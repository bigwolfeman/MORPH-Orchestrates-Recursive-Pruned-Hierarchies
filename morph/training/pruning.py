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
    _is_compact: bool = field(default=False, repr=False)
    _is_routed: bool = field(default=False, repr=False)

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

        # ── Phase 1 / 2: accumulate gradient scores (pre-compact only) ───
        if not self._is_compact:
            for _name, layer in layers:
                layer.accumulate_scores()

            if global_step % 10 == 0:
                for _name, layer in layers:
                    layer.score_step()

        # ── Phase 2: topology decisions ───────────────────────────────────
        if (
            global_step >= self.prune_start
            and not self._is_compact
            and global_step % self.prune_interval == 0
        ):
            cur_density = self._current_density(layers)
            if cur_density > self.target_density:
                for _name, layer in layers:
                    layer.topology_step(global_step=global_step)
                stats = self.log_stats(model)
                stats["pruning/topology_step"] = 1

        # ── Phase 3: compact ──────────────────────────────────────────────
        if global_step == self.compact_step and not self._is_compact:
            use_groups = self.route_start > 0
            total_live = 0
            block_ell_layers = _find_block_ell_layers(model)
            for _name, layer in block_ell_layers:
                if use_groups:
                    n_alive = layer.compact_with_groups(self.n_clusters)
                else:
                    n_alive = layer.compact()
                total_live += n_alive
            self._is_compact = True
            stats = self.log_stats(model)
            stats["pruning/compacted"] = 1
            stats["pruning/total_live_blocks"] = total_live
            if use_groups:
                stats["pruning/n_clusters"] = self.n_clusters

        # ── Phase 4: activate routing ─────────────────────────────────────
        if (
            self.route_start > 0
            and global_step == self.route_start
            and self._is_compact
            and not self._is_routed
        ):
            n_swapped = self._activate_routing(model)
            self._is_routed = True
            stats = stats or {}
            stats["routing/activated"] = 1
            stats["routing/n_layers_swapped"] = n_swapped
            stats["routing/activation_ratio"] = self.activation_ratio

        return stats

    def _activate_routing(self, model: nn.Module) -> int:
        """Swap BlockELLLinear modules for RoutedBlockELLLinear in-place.

        Walks the model tree, finds all BlockELLLinear that are post-compact
        (have cluster metadata), wraps them in RoutedBlockELLLinear, and
        replaces them on the parent module.

        Returns the number of layers swapped.
        """
        from morph.model.routing import RoutedBlockELLLinear
        from morph.model.sparsity import BlockELLLinear

        swapped = 0
        # Build a list first to avoid modifying during iteration
        targets: list[tuple[nn.Module, str, BlockELLLinear]] = []
        for name, module in model.named_modules():
            for attr_name, child in module.named_children():
                if isinstance(child, BlockELLLinear) and not child._cms._dense_mode:
                    if hasattr(child._cms, "n_clusters"):
                        targets.append((module, attr_name, child))

        for parent, attr_name, layer in targets:
            d_model = layer.in_features
            routed = RoutedBlockELLLinear(
                block_ell_linear=layer,
                d_model=d_model,
                activation_ratio=self.activation_ratio,
                aux_loss_coeff=self.aux_loss_coeff,
            )
            setattr(parent, attr_name, routed)
            swapped += 1
            print(f"  Routing activated: {parent.__class__.__name__}.{attr_name} "
                  f"({layer.in_features}→{layer.out_features}, "
                  f"{self.n_clusters} clusters, {self.activation_ratio:.0%} active)")

        return swapped

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
                if hasattr(layer, "tile_mask"):
                    mask = layer.tile_mask
                    n_alive = int(mask.sum().item())
                    n_total = int(mask.numel())
                elif hasattr(layer, "block_mask"):
                    mask = layer.block_mask
                    n_alive = int(mask.sum().item())
                    n_total = int(mask.numel())
                elif hasattr(layer, "K_active"):
                    n_alive = int(layer.K_active)
                    R = getattr(layer, "R", 1)
                    K = getattr(layer, "K", 1)
                    n_total = R * K
                else:
                    continue
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
        """Estimate overall density from alive vs total blocks across all layers."""
        alive = 0
        total = 0
        for _name, layer in layers:
            try:
                if hasattr(layer, "tile_mask"):
                    alive += int(layer.tile_mask.sum().item())
                    total += int(layer.tile_mask.numel())
                elif hasattr(layer, "block_mask"):
                    alive += int(layer.block_mask.sum().item())
                    total += int(layer.block_mask.numel())
            except Exception:
                pass
        return alive / max(total, 1)
