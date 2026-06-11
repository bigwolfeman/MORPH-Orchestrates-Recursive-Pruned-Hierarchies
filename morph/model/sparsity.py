"""Block-ELL sparsity integration for MORPH.

BlockELLLinear wraps CMSBlockLinear with a clean two-mode interface:
  - Dense mode  (pre-compact): forward is a plain nn.Linear cuBLAS call.
    All CMS scoring machinery accumulates during this phase.
  - Sparse mode (post-compact): forward uses Triton Block-ELL kernels.
    Topology is frozen after compact() — no further swaps.

SparsitySchedule coordinates the three-level CMS update cadence across
all BlockELLLinear modules in a model and fires compact() at the right step.

Critical ordering rule (from CLAUDE.md / CMSBlockLinear contract):
  loss.backward()
  schedule.step(global_step, loss)   # accumulate_scores() is called here
  optimizer.step()
  optimizer.zero_grad()

Source lineage: CMSBlockLinear from morph/model/titans_core/block_sparse.py
Date: 2026-05-25
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .titans_core.block_sparse import CMSBlockLinear


# =============================================================================
# BlockELLLinear — drop-in replacement for nn.Linear
# =============================================================================


class BlockELLLinear(nn.Module):
    """Drop-in replacement for nn.Linear using Block-ELL format internally.

    Wraps CMSBlockLinear with two stable modes:

    Dense mode (default, pre-compact):
        Behaves exactly like nn.Linear — uses cuBLAS GEMM via F.linear.
        CMSBlockLinear stores weights as a flat [out, in] tensor in this phase.
        CMS scoring hooks run silently in the background (activation/gradient norms).
        Topology decisions (prune/grow blocks) fire at the usual intervals.

    Sparse mode (post-compact):
        After compact() is called, weights are rebuilt into Block-ELL format
        [R, K_active, B, B] + col_indices [R, K_active].  Forward uses the
        Triton Block-ELL kernel (or the vectorized PyTorch fallback when Triton
        is unavailable).  Topology is frozen after compaction.

    The mode switch happens exactly once, at compaction time, not at runtime.

    Args:
        in_features:        Input dimension (must be divisible by tile_size).
        out_features:       Output dimension (must be divisible by tile_size).
        bias:               Include bias term.  Default True.
        tile_size:          Block tile size.  16 is the WMMA-safe default for sm_120.
        initial_density:    Starting density (fraction of blocks active per row).
                            Set to 1.0 for fully-dense pre-compact phase.
        score_ema_alpha:    EMA decay for gradient importance scores.
        swap_threshold:     Required score improvement ratio for a block swap.
        exploration_epsilon: Probability of a random (non-greedy) swap.
        topology_warmup_steps: Global steps before first topology swap.
        device:             Target device.
        dtype:              Parameter dtype (bf16 recommended for training).

    Example:
        >>> layer = BlockELLLinear(1536, 6144)
        >>> x = torch.randn(4, 256, 1536)
        >>> y = layer(x)               # [4, 256, 6144]
        >>> loss = y.mean()
        >>> loss.backward()
        >>> layer.accumulate_scores()   # call BEFORE optimizer.step()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tile_size: int = 16,
        initial_density: float = 1.0,
        score_ema_alpha: float = 0.95,
        swap_threshold: float = 2.5,
        exploration_epsilon: float = 0.05,
        topology_warmup_steps: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # Clamp density to valid range (CMSBlockLinear requires [0.1, 1.0])
        density = max(0.1, min(1.0, initial_density))

        self._cms = CMSBlockLinear(
            in_features=in_features,
            out_features=out_features,
            tile_size=tile_size,
            density=density,
            bias=bias,
            score_ema_alpha=score_ema_alpha,
            swap_threshold=swap_threshold,
            exploration_epsilon=exploration_epsilon,
            topology_warmup_steps=topology_warmup_steps,
            device=device,
            dtype=dtype,
        )

        # Expose dimensions for external inspection
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size

    # ── Mode queries ─────────────────────────────────────────────────────────

    @property
    def is_compact(self) -> bool:
        """True if weights have been converted to Block-ELL sparse format."""
        return not self._cms._dense_mode

    @property
    def bias(self) -> Optional[nn.Parameter]:
        """Bias parameter (None if bias=False)."""
        return self._cms.bias

    # ── CMS update hooks ─────────────────────────────────────────────────────

    def accumulate_scores(self) -> None:
        """Accumulate gradient statistics for importance scoring.

        Must be called AFTER loss.backward() and BEFORE optimizer.zero_grad().
        Reads weight.grad (dense mode) or values.grad (sparse mode) and
        updates the gradient-EMA block importance scores.

        Safe to call in either mode; no-op if grad is None.
        """
        self._cms.accumulate_scores()

    def score_step(self) -> None:
        """Level-1 update: normalize accumulators and increment block ages.

        Call every ~10 training steps.  Does NOT change topology.
        """
        self._cms.score_step()

    def topology_step(self, global_step: int) -> None:
        """Level-2 update: prune/grow topology decisions.

        Call every ~100 training steps (before compaction only).
        After compaction the CMSBlockLinear internal flag `disable_topology_updates`
        is set to True, so this becomes a safe no-op.

        Args:
            global_step: Current global training step (used for deterministic
                         DDP topology and churn-cooldown tracking).
        """
        self._cms.topology_step(global_step=global_step)

    # ── Compaction ───────────────────────────────────────────────────────────

    def compact(self) -> int:
        """Convert from dense to Block-ELL sparse format.

        Detects alive tiles by Frobenius norm, discards dead-weight blocks,
        rebuilds weights into [R, K_active, B, B] + col_indices [R, K_active].
        After this call the forward path switches to the Triton kernel and
        topology updates are permanently disabled.

        Returns:
            new_K: Number of active blocks per row after compaction.
        """
        return self._cms.compact()

    def compact_with_groups(self, n_clusters: int = 16) -> int:
        """Compact and register cluster metadata for routed forward.

        Same as compact() but also divides row-groups into n_clusters
        contiguous clusters.  Required before wrapping with
        RoutedBlockELLLinear.

        Returns:
            new_K: Number of active blocks per row after compaction.
        """
        return self._cms.compact_with_groups(n_clusters=n_clusters)

    def carve(self, blocking: int = 128) -> int:
        """MORTAR alternative to compact(): pack the masked-dense weight into
        128×128 BCSR blocks executed by the vendored stk Triton backend
        (3.09× faster than dense at 0.25 density — Gate G1; the 16×16 Block-ELL
        kernel was 1.1× SLOWER). Pair with prune_step_blocks for lossless carving.

        Returns:
            nnz: Number of kept 128×128 blocks.
        """
        return self._cms.carve(blocking=blocking)

    # ── Density ──────────────────────────────────────────────────────────────

    def get_density(self) -> float:
        """Current density ratio K / C (fraction of input block-columns active per row)."""
        return self._cms.get_density()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass — delegates to CMSBlockLinear which handles mode dispatch.

        Pre-compact:  F.linear(x, weight, bias) via cuBLAS.
        Post-compact: Triton Block-ELL kernel (or PyTorch fallback).

        Args:
            x: [..., in_features]

        Returns:
            [..., out_features]
        """
        return self._cms(x)

    def extra_repr(self) -> str:
        mode = "sparse" if self.is_compact else "dense"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"tile={self.tile_size}, density={self.get_density():.2f}, mode={mode}"
        )


# =============================================================================
# SparsitySchedule — coordinate CMS across all BlockELLLinear modules
# =============================================================================


class SparsitySchedule:
    """Manages the Block-ELL pruning schedule across a model.

    Finds all BlockELLLinear modules, then handles the three-level CMS cadence:
      - Every step:       accumulate_scores()
      - Every 10 steps:   score_step()
      - Every 100 steps:  topology_step()  (only in the prune window)
      - At compact_step:  compact() once

    Usage in the training loop::

        schedule = SparsitySchedule(
            model,
            prune_start=3_000,
            prune_interval=3_000,
            prune_rate=0.10,
            target_density=0.25,
            compact_step=39_000,
        )

        for step, batch in enumerate(dataloader):
            loss = model(batch)["loss"]
            loss.backward()

            schedule.step(step, loss.item())   # accumulate + topology

            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                wandb.log(schedule.log_stats())

    Args:
        model:          nn.Module to scan for BlockELLLinear layers.
        prune_start:    Global step at which topology_step() starts firing.
                        Scores are still accumulated before this — just no swaps.
        prune_interval: Steps between topology_step() calls (default 100).
        prune_rate:     Fraction of blocks to prune per topology step.
                        Not directly used here — CMSBlockLinear handles the
                        internal swap logic; this is stored for logging only.
        target_density: Target density after pruning completes (stored for logging).
        compact_step:   Global step at which compact() fires (exactly once).
    """

    def __init__(
        self,
        model: nn.Module,
        prune_start: int = 3_000,
        prune_interval: int = 100,
        prune_rate: float = 0.10,
        target_density: float = 0.25,
        compact_step: int = 39_000,
    ) -> None:
        self.prune_start = prune_start
        self.prune_interval = prune_interval
        self.prune_rate = prune_rate
        self.target_density = target_density
        self.compact_step = compact_step

        self._compacted = False

        # Collect all BlockELLLinear modules with their path names
        self._layers: List[Tuple[str, BlockELLLinear]] = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, BlockELLLinear)
        ]

        if not self._layers:
            import warnings
            warnings.warn(
                "SparsitySchedule: no BlockELLLinear modules found in model. "
                "Did you replace nn.Linear layers?"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_compact(self) -> bool:
        """True after compact() has fired for all layers."""
        return self._compacted

    @property
    def num_layers(self) -> int:
        """Number of BlockELLLinear layers tracked by this schedule."""
        return len(self._layers)

    def step(self, global_step: int, loss: float) -> None:
        """Main per-step entry point. Call after loss.backward(), before optimizer.step().

        Handles:
        - accumulate_scores()     — every step (reads .grad, safe no-op if None)
        - score_step()            — every 10 steps
        - topology_step()         — every prune_interval steps, only in [prune_start, compact_step)
        - compact()               — once, at compact_step

        Args:
            global_step: Current training step (0-indexed).
            loss:        Current loss value (stored for diagnostics, not used internally).
        """
        # 1. Accumulate gradient scores — must happen BEFORE optimizer.zero_grad()
        for _, layer in self._layers:
            layer.accumulate_scores()

        # 2. Score normalisation every 10 steps
        if global_step % 10 == 0:
            for _, layer in self._layers:
                layer.score_step()

        # 3. Topology decisions: only during the prune window, not after compact
        if (
            not self._compacted
            and global_step >= self.prune_start
            and global_step < self.compact_step
            and global_step % self.prune_interval == 0
        ):
            for _, layer in self._layers:
                layer.topology_step(global_step=global_step)

        # 4. Compaction — exactly once
        if not self._compacted and global_step >= self.compact_step:
            self._run_compact()

    def log_stats(self) -> Dict[str, float]:
        """Return a flat dict of per-layer stats suitable for wandb.log().

        Keys follow the pattern::
            sparsity/<layer_name>/density
            sparsity/<layer_name>/avg_block_score
            sparsity/<layer_name>/avg_block_age
            sparsity/global/mean_density
            sparsity/global/is_compact

        Returns:
            dict mapping string keys to float values.
        """
        stats: Dict[str, float] = {}
        densities: List[float] = []

        for name, layer in self._layers:
            cms = layer._cms
            topo = cms.get_topology_stats()
            safe_name = name.replace(".", "/")

            stats[f"sparsity/{safe_name}/density"] = topo.density
            stats[f"sparsity/{safe_name}/avg_block_score"] = topo.avg_block_score
            stats[f"sparsity/{safe_name}/avg_block_age"] = topo.avg_block_age
            stats[f"sparsity/{safe_name}/column_entropy"] = topo.column_entropy
            stats[f"sparsity/{safe_name}/crystallized_count"] = float(topo.crystallized_count)
            densities.append(topo.density)

        if densities:
            stats["sparsity/global/mean_density"] = sum(densities) / len(densities)
            stats["sparsity/global/min_density"] = min(densities)

        stats["sparsity/global/is_compact"] = float(self._compacted)
        return stats

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_compact(self) -> None:
        """Fire compact() on all layers and record the result."""
        for name, layer in self._layers:
            new_k = layer.compact()

        self._compacted = True
