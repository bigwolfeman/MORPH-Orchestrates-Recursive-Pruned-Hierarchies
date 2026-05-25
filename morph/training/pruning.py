"""MORPH pruning schedule — thin coordinator that drives CMSBlockLinear topology.

Three-phase orchestration:
  Phase 1  (0 … prune_start):        dense training, accumulate_scores every step
  Phase 2  (prune_start … compact):  topology_step every prune_interval steps
  Phase 3  (compact … end):          post-compact sparse, routing

Usage:
    schedule = PruningSchedule.from_cfg(cfg)
    # in training loop (AFTER loss.backward(), BEFORE optimizer.zero_grad()):
    stats = schedule.step(model, global_step)
    if stats:
        wandb.log(stats, step=global_step)
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class PruningSchedule:
    """Drives the CMS three-phase topology schedule.

    Attributes:
        prune_start:    Global step to start topology_step calls.
        prune_interval: How many steps between topology_step calls.
        prune_rate:     Fraction of blocks removed per topology_step.
        target_density: Stop pruning once overall density <= this value.
        compact_step:   Global step to call compact() on all CMS layers.
        _is_compact:    Internal flag, set after compact() is called.
    """

    prune_start: int = 50_000
    prune_interval: int = 3_000
    prune_rate: float = 0.05
    target_density: float = 0.25
    compact_step: int = 60_000
    _is_compact: bool = False

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "PruningSchedule":
        tr = cfg.training
        return cls(
            prune_start=int(getattr(tr, "prune_start", 50_000)),
            prune_interval=int(getattr(tr, "prune_interval", 3_000)),
            prune_rate=float(getattr(tr, "prune_rate", 0.05)),
            target_density=float(getattr(tr, "target_density", 0.25)),
            compact_step=int(getattr(tr, "compact_step", 60_000)),
        )

    @property
    def is_compact(self) -> bool:
        return self._is_compact

    def step(self, model: nn.Module, global_step: int) -> Optional[dict]:
        """Orchestrate CMS calls for the current training step.

        MUST be called between loss.backward() and optimizer.zero_grad()
        so that weight.grad is populated when accumulate_scores() runs.

        Returns a dict of metrics for wandb logging, or None if no topology
        action happened this step.
        """
        layers = _find_cms_layers(model)
        if not layers:
            return None

        # ── Phase 1 / 2 / 3: always accumulate gradient scores ───────────
        # (accumulate_scores reads p.grad, so must be called before zero_grad)
        for _name, layer in layers:
            layer.accumulate_scores()

        # Every 10 steps: normalize scores, age blocks.
        if global_step % 10 == 0:
            for _name, layer in layers:
                layer.score_step()

        stats: Optional[dict] = None

        # ── Phase 2: topology decisions ───────────────────────────────────
        if (
            global_step >= self.prune_start
            and not self._is_compact
            and global_step % self.prune_interval == 0
        ):
            # Only prune if we're above the density target.
            cur_density = self._current_density(layers)
            if cur_density > self.target_density:
                for _name, layer in layers:
                    layer.topology_step(global_step=global_step)
                stats = self.log_stats(model)
                stats["pruning/topology_step"] = 1

        # ── Phase 3: compact ──────────────────────────────────────────────
        if global_step == self.compact_step and not self._is_compact:
            total_live = 0
            for _name, layer in layers:
                n_alive = layer.compact()
                total_live += n_alive
            self._is_compact = True
            stats = self.log_stats(model)
            stats["pruning/compacted"] = 1
            stats["pruning/total_live_blocks"] = total_live

        return stats

    def log_stats(self, model: nn.Module) -> dict:
        """Compute per-layer and aggregate density metrics for wandb.

        Returns a flat dict with keys:
          pruning/density_<layer_name>   — per-layer alive-block fraction
          pruning/total_density          — aggregate across all layers
          pruning/n_blocks_alive         — total live block count
          pruning/n_blocks_total         — total possible block count
        """
        layers = _find_cms_layers(model)
        if not layers:
            return {}

        log: dict = {}
        total_alive = 0
        total_blocks = 0

        for name, layer in layers:
            try:
                # CMSBlockLinear exposes these as properties / attributes.
                # tile_mask: [R, K] bool — True for alive blocks.
                if hasattr(layer, "tile_mask"):
                    mask = layer.tile_mask
                    n_alive = int(mask.sum().item())
                    n_total = int(mask.numel())
                elif hasattr(layer, "block_mask"):
                    mask = layer.block_mask
                    n_alive = int(mask.sum().item())
                    n_total = int(mask.numel())
                else:
                    # Fall back to counting non-zero rows in indices tensor
                    # CMSBlockLinear v2 stores alive blocks in indices
                    if hasattr(layer, "K_active"):
                        n_alive = int(layer.K_active)
                        R = getattr(layer, "R", 1)
                        K = getattr(layer, "K", 1)
                        n_total = R * K
                    else:
                        continue
            except Exception:
                continue

            density = n_alive / max(n_total, 1)
            # Clean up the name for logging (e.g. "core.0.mlp.0._swiglu.gate_up")
            safe_name = name.replace(".", "_")
            log[f"pruning/density_{safe_name}"] = density
            total_alive += n_alive
            total_blocks += n_total

        if total_blocks > 0:
            log["pruning/total_density"] = total_alive / total_blocks
            log["pruning/n_blocks_alive"] = total_alive
            log["pruning/n_blocks_total"] = total_blocks

        return log

    # ── Internal helpers ───────────────────────────────────────────────────

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
