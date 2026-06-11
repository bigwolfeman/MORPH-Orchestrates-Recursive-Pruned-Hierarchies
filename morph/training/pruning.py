"""MORPH pruning schedule — thin coordinator that drives CMSBlockLinear topology.

Five-phase orchestration:
  Phase 1  (0 … prune_start):        dense training, accumulate_scores every step
  Phase 2  (prune_start … compact):  topology_step every prune_interval steps
  Phase 3  (compact … route_start):  post-compact sparse, settling
  Phase 4  (route_start … end):      per-token ReMoE routing over pruned tile-groups

Usage:
    schedule = PruningSchedule.from_cfg(cfg)
    # in training loop (AFTER loss.backward(), BEFORE optimizer.zero_grad()):
    stats = schedule.step(model, global_step)
    if stats:
        wandb.log(stats, step=global_step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

__all__ = ["PruningSchedule"]


def _find_cms_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Walk the model and return all (name, module) pairs for CMSBlockLinear."""
    try:
        from morph.model.titans_core.block_sparse import CMSBlockLinear
    except ImportError:
        return []

    results = []
    for name, module in model.named_modules():
        if isinstance(module, CMSBlockLinear):
            results.append((name, module))
    return results


def _find_block_ell_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Walk the model and return all (name, module) pairs for BlockELLLinear."""
    from morph.model.sparsity import BlockELLLinear

    results = []
    for name, module in model.named_modules():
        if isinstance(module, BlockELLLinear):
            results.append((name, module))
    return results


@dataclass
class PruningSchedule:
    """Drives the CMS five-phase topology schedule.

    Attributes:
        prune_start:      Global step to start topology_step calls.
        prune_interval:   How many steps between topology_step calls.
        prune_rate:       Fraction of blocks removed per topology_step.
        target_density:   Stop pruning once overall density <= this value.
        compact_step:     Global step to call compact on all CMS layers.
        route_start:      Global step to activate ReMoE routing (0 = disabled).
        n_clusters:       Number of tile-group clusters for routing.
        activation_ratio: Fraction of tile-groups activated per token.
        aux_loss_coeff:   Load-balance penalty coefficient for routing.
        _is_compact:      Internal flag, set after compact() is called.
        _is_routed:       Internal flag, set after routing layers are swapped in.
    """

    prune_start: int = 6_000
    prune_interval: int = 3_000
    prune_rate: float = 0.05
    target_density: float = 0.25
    compact_step: int = 66_000
    route_start: int = 70_000
    n_clusters: int = 16
    activation_ratio: float = 0.5
    aux_loss_coeff: float = 1e-2
    # Sparse execution backend after the density transition:
    #   "blockell" — legacy 16×16 Block-ELL compact() (kernel measured SLOWER than dense)
    #   "mortar"   — 128×128 BCSR carve() via vendored stk (3.09× FASTER than dense, G1).
    # MORTAR also switches pruning to block-aligned prune_step_blocks (score at tile,
    # prune/execute at block) so the carve is lossless.
    sparse_backend: str = "blockell"
    carve_blocking: int = 128
    # Detach the router input so the load-balance gradient does NOT flow into the looped carrier
    # x (required for memory: it otherwise extends BPTT depth → +7 GB/step at deploy shape). The
    # router still trains (params get grad from the detached input + gates from the main loss).
    aux_detach_input: bool = True
    # Which MLPs get ReMoE routing at route_start:
    #   "core" — only the looped core block (B5 / legacy behaviour, default)
    #   "all"  — the WHOLE body: prelude + core + coda (Wolfe 2026-06-10).
    # Prelude/coda run once (not looped) → routed with n_iters=1; core keeps n_iters=max_depth.
    route_scope: str = "core"
    _is_compact: bool = field(default=False, repr=False)
    _is_routed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.sparse_backend not in ("blockell", "mortar"):
            raise ValueError(
                f"sparse_backend={self.sparse_backend!r}; choices: 'blockell', 'mortar'"
            )
        if self.route_scope not in ("core", "all"):
            raise ValueError(
                f"route_scope={self.route_scope!r}; choices: 'core', 'all'"
            )

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "PruningSchedule":
        tr = cfg.training
        rt = getattr(cfg, "routing", None)
        return cls(
            prune_start=int(getattr(tr, "prune_start", 6_000)),
            prune_interval=int(getattr(tr, "prune_interval", 3_000)),
            prune_rate=float(getattr(tr, "prune_rate", 0.05)),
            target_density=float(getattr(tr, "target_density", 0.25)),
            compact_step=int(getattr(tr, "compact_step", 66_000)),
            route_start=int(getattr(rt, "route_start", 0) if rt else 0),
            n_clusters=int(getattr(rt, "n_clusters", 16) if rt else 16),
            activation_ratio=float(getattr(rt, "activation_ratio", 0.5) if rt else 0.5),
            aux_loss_coeff=float(getattr(rt, "aux_loss_coeff", 1e-2) if rt else 1e-2),
            aux_detach_input=bool(getattr(rt, "aux_detach_input", True) if rt else True),
            route_scope=str(getattr(rt, "route_scope", "core") if rt else "core"),
            sparse_backend=str(getattr(tr, "sparse_backend", "blockell")),
            carve_blocking=int(getattr(tr, "carve_blocking", 128)),
        )

    @property
    def is_compact(self) -> bool:
        return self._is_compact

    @property
    def is_routed(self) -> bool:
        return self._is_routed

    def step(self, model: nn.Module, global_step: int) -> Optional[dict]:
        """Orchestrate CMS calls for the current training step.

        MUST be called between loss.backward() and optimizer.zero_grad()
        so that weight.grad is populated when accumulate_scores() runs.

        Returns a dict of metrics for wandb logging, or None if no topology
        action happened this step.
        """
        layers = _find_cms_layers(model)
        if not layers and not self._is_routed:
            return None

        stats: Optional[dict] = None

        # ── Every step: keep already-pruned tiles dead (before scoring/optimizer) ──
        # Re-zeros dead tiles + their grads in the live weight so neither optimizer
        # momentum nor the ternary STE can revive a pruned tile.
        if not self._is_compact:
            for _name, layer in layers:
                layer.apply_prune_mask()

        # ── Phase 1 / 2: accumulate gradient scores (pre-compact only) ───
        if not self._is_compact:
            for _name, layer in layers:
                layer.accumulate_scores()

            if global_step % 10 == 0:
                for _name, layer in layers:
                    layer.score_step()

        # ── Phase 2: structured masked-dense pruning ──────────────────────
        # Zero the lowest-saliency tiles down toward target_density. Auto-stops
        # once density <= target (prune_step is a no-op at target). This is the
        # REAL density reduction; the old topology_step only swapped at fixed K.
        if (
            global_step >= self.prune_start
            and not self._is_compact
            and global_step % self.prune_interval == 0
        ):
            cur_density = self._current_density(layers)
            if cur_density > self.target_density + 1e-6:
                pruned_total = 0
                for _name, layer in layers:
                    if self.sparse_backend == "mortar":
                        # Block-aligned pruning (128×128 execution blocks, global
                        # top-k with ≥1-per-row floor) so carve() is lossless.
                        res = layer.prune_step_blocks(
                            self.prune_rate, self.target_density, self.carve_blocking
                        )
                    else:
                        res = layer.prune_step(self.prune_rate, self.target_density)
                    pruned_total += int(res.get("pruned", 0))
                new_density = self._current_density(layers)
                stats = self.log_stats(model)
                stats["pruning/prune_step"] = 1
                stats["pruning/tiles_pruned"] = pruned_total
                stats["pruning/density"] = new_density
                # stdout so monitors/smoke-gates can SEE density fall without wandb.
                print(f"[prune] step {global_step}: tiles_pruned={pruned_total} "
                      f"density={new_density:.4f} (target {self.target_density})", flush=True)

        # ── Phase 3: compact ──────────────────────────────────────────────
        # Hidden-neuron routing (Wolfe's choice) does NOT need compact_with_groups'
        # output-cluster metadata — the router builds its own d_ff neuron→cluster map.
        # So plain compact() everywhere. For ternary layers we compact the SMOOTH shadow
        # (restore_parametrized=False) — NOT the baked discrete ternary — so QAT keeps a
        # real gradient signal, then re-register a 4-D-aware ternary STE on `values`.
        # Net effect: "keep pretraining the ternary model, now compacted." Optimizer is
        # rebuilt afterward (weight→values is a new param set).
        if global_step == self.compact_step and not self._is_compact:
            import torch.nn.utils.parametrize as parametrize
            from morph.model.ternary_qat import reparametrize_compacted_values_ternary

            total_live = 0
            n_reternary = 0
            block_ell_layers = _find_block_ell_layers(model)
            reparam: list[tuple[nn.Module, float]] = []
            for _name, layer in block_ell_layers:
                cms = layer._cms
                was_ternary = parametrize.is_parametrized(cms, "weight")
                thr = 0.5
                if was_ternary:
                    # Capture the threshold, then restore the SMOOTH shadow as the leaf
                    # weight (leave_parametrized=False) so compact carries continuous
                    # survivor values, not the discrete ternary.
                    try:
                        thr = float(cms.parametrizations.weight[0].threshold)
                    except (AttributeError, IndexError):
                        thr = 0.5
                    parametrize.remove_parametrizations(cms, "weight",
                                                        leave_parametrized=False)
                if self.sparse_backend == "mortar":
                    n_alive = layer.carve(blocking=self.carve_blocking)
                else:
                    n_alive = layer.compact()
                # Phase C never prunes again → the activation/gradient scoring hooks are dead
                # compute and per-forward memory we cannot afford once routing activations land.
                cms.set_scoring(False)
                total_live += n_alive
                if was_ternary:
                    reparam.append((cms, thr))
            # Re-register ternary QAT on the compacted 4-D `values` (continues QAT).
            for cms, thr in reparam:
                if reparametrize_compacted_values_ternary(cms, thr):
                    n_reternary += 1
            self._is_compact = True
            stats = self.log_stats(model)
            stats["pruning/compacted"] = 1
            stats["pruning/total_live_blocks"] = total_live
            stats["pruning/n_reternary_layers"] = n_reternary
            stats["_rebuild_optimizer"] = True
            verb = "carved (mortar)" if self.sparse_backend == "mortar" else "compacted"
            print(f"[compact] step {global_step}: {len(block_ell_layers)} layers {verb}, "
                  f"{total_live} live blocks, {n_reternary} re-ternarized → optimizer rebuild",
                  flush=True)

        # ── Phase 4: activate routing ─────────────────────────────────────
        # Routing normally follows compaction (route_start > compact_step), so we
        # require _is_compact. EXCEPTION — a pure-DENSE ablation that disables carve
        # (compact_step set to the never-reached sentinel) still wants ReMoE: the
        # hidden-neuron gating (TileRouter, _SwiGLUBlockELL.forward) operates on the
        # dense gate_up output and does NOT need the BCSR/carved structure. So allow
        # routing on an un-compacted model IFF carve is explicitly disabled. Normal
        # sparse runs (compact_step a real step) are byte-identical to before.
        _carve_disabled = self.compact_step > 10_000_000
        if (
            self.route_start > 0
            and global_step == self.route_start
            and (self._is_compact or _carve_disabled)
            and not self._is_routed
        ):
            n_enabled = self._activate_routing(model)
            self._is_routed = True
            stats = stats or {}
            stats["routing/activated"] = 1
            stats["routing/n_core_mlps_routed"] = n_enabled
            stats["routing/activation_ratio"] = self.activation_ratio
            stats["_rebuild_optimizer"] = True
            print(f"[route] step {global_step}: routing enabled on {n_enabled} core MLPs "
                  f"({self.n_clusters} clusters, {self.activation_ratio:.0%} active) → "
                  f"optimizer rebuild", flush=True)

        return stats

    def _activate_routing(self, model: nn.Module) -> int:
        """Enable iteration-aware hidden-neuron routing on the selected MLPs.

        Routing gates the d_ff hidden neuron bank of each block's _SwiGLUBlockELL MLP
        (clean PEER/MoE expert selection), via a shared TileRouter whose zero-init
        iteration embedding makes routing un-specialized at turn-on and specialize
        through training. Adds router params → optimizer must be rebuilt after.

        Scope (self.route_scope):
          "core" — only the looped CORE block (legacy / B5). Iteration-aware:
                   n_iters = max core-loop depth (each loop iteration gets its own
                   routing embedding row).
          "all"  — the WHOLE body: prelude + core + coda (Wolfe 2026-06-10). Prelude
                   and coda run ONCE (iter_idx always 0, not looped), so they are
                   routed with n_iters=1; the core stays iteration-aware. Aux losses
                   from every router are collected automatically (collect_routing_aux_losses
                   is a generic model.modules() walk on _last_aux_loss).

        Returns the number of MLPs on which routing was enabled.
        """
        root = getattr(model, "_orig_mod", model)   # unwrap torch.compile
        max_depth = int(getattr(root.cfg, "max_depth", 8))

        # (group-name, module-list, n_iters). Core is iteration-aware; prelude/coda are
        # single-pass → n_iters=1. Order is prelude→core→coda for readable logs.
        groups = [("core", root.core, max_depth)]
        if self.route_scope == "all":
            groups = (
                [("prelude", root.prelude, 1)]
                + groups
                + [("coda", root.coda, 1)]
            )

        enabled = 0
        for gname, group, gi in groups:
            for bi, blk in enumerate(group):
                mlp = getattr(blk, "mlp", None)
                if mlp is None or not hasattr(mlp, "enable_routing"):
                    # The MLP is a _SwiGLUBlockELL (optionally wrapped in _KwargSequential
                    # for dropout), both of which expose enable_routing. Anything else means
                    # the block structure changed out from under Phase C — fail LOUD, do not
                    # silently enable 0 routers (theater we forbid).
                    raise RuntimeError(
                        f"_activate_routing: {gname} block {bi} has no routable MLP "
                        f"(mlp type={type(mlp).__name__}); cannot attach ReMoE router. "
                        f"Expected _SwiGLUBlockELL or _KwargSequential wrapping it."
                    )
                mlp.enable_routing(
                    n_clusters=self.n_clusters,
                    activation_ratio=self.activation_ratio,
                    aux_loss_coeff=self.aux_loss_coeff,
                    n_iters=gi,
                    detach_input=self.aux_detach_input,
                )
                enabled += 1
                print(f"  Routing enabled: {gname} MLP {bi} "
                      f"(d_ff={mlp.d_ff}, {self.n_clusters} clusters, "
                      f"{self.activation_ratio:.0%} active, n_iters={gi})")

        if enabled == 0:
            raise RuntimeError(
                f"_activate_routing: no MLPs found to route (route_scope={self.route_scope!r})"
            )
        print(f"  Routing scope={self.route_scope!r}: {enabled} MLPs routed", flush=True)
        return enabled

    def log_stats(self, model: nn.Module) -> dict:
        """Compute per-layer and aggregate density metrics for wandb."""
        layers = _find_cms_layers(model)
        if not layers:
            return {}

        log: dict = {}
        total_alive = 0
        total_blocks = 0

        for name, layer in layers:
            try:
                R = int(getattr(layer, "R", 0))
                C = int(getattr(layer, "C", 0))
                if R == 0 or C == 0:
                    continue
                if getattr(layer, "_prune_mask", None) is not None:
                    n_alive = int(layer._prune_mask.sum().item())
                    n_total = R * C
                elif not getattr(layer, "_dense_mode", True):
                    n_alive = R * int(getattr(layer, "K", C))
                    n_total = R * C
                else:
                    n_alive = R * C
                    n_total = R * C
            except Exception:
                continue

            density = n_alive / max(n_total, 1)
            safe_name = name.replace(".", "_")
            log[f"pruning/density_{safe_name}"] = density
            total_alive += n_alive
            total_blocks += n_total

        if total_blocks > 0:
            log["pruning/total_density"] = total_alive / total_blocks
            log["pruning/n_blocks_alive"] = total_alive
            log["pruning/n_blocks_total"] = total_blocks

        return log

    def _current_density(self, layers: list) -> float:
        """Overall density = alive tiles / total tiles across all CMS layers.

        Pre-compact this reads the structured prune mask (prune_density); post-compact
        it falls back to K/C. A layer never pruned reports density 1.0.
        """
        alive = 0
        total = 0
        for _name, layer in layers:
            try:
                R = int(getattr(layer, "R", 0))
                C = int(getattr(layer, "C", 0))
                if R == 0 or C == 0:
                    continue
                if getattr(layer, "_prune_mask", None) is not None:
                    alive += int(layer._prune_mask.sum().item())
                    total += R * C
                elif not getattr(layer, "_dense_mode", True):
                    # post-compact: K active columns per row
                    alive += R * int(getattr(layer, "K", C))
                    total += R * C
                else:
                    # dense, never pruned → fully dense
                    alive += R * C
                    total += R * C
            except Exception:
                pass
        return alive / max(total, 1)
