"""MORTAR sparsity integration for MORPH.

MortarLinear wraps CMSBlockLinear with a clean two-mode interface:
  - Dense mode  (pre-carve): forward is a plain nn.Linear cuBLAS call.
    Tile-saliency scoring (block_score_ema) accumulates during this phase and
    prune_step_blocks zeroes 128×128 blocks in the masked-dense weight.
  - MORTAR mode (post-carve): carve() packs the masked weight into 128×128
    BCSR storage executed by the vendored stk Triton backend.

The schedule that drives prune → carve → route lives in
morph/training/pruning.py (PruningSchedule).

Source lineage: CMSBlockLinear from morph/model/titans_core/block_sparse.py
Date: 2026-05-25 (Block-ELL backend removed 2026-06-11 — MORTAR only)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .titans_core.block_sparse import CMSBlockLinear


# =============================================================================
# MortarLinear — drop-in replacement for nn.Linear
# =============================================================================


class MortarLinear(nn.Module):
    """Drop-in replacement for nn.Linear: dense pre-carve, MORTAR BCSR post-carve.

    Wraps CMSBlockLinear with two stable modes:

    Dense mode (default, pre-carve):
        Behaves exactly like nn.Linear — uses cuBLAS GEMM via F.linear.
        CMSBlockLinear stores weights as a flat [out, in] tensor in this phase.
        Gradient tile-saliency accumulates silently for the pruning schedule.

    MORTAR mode (post-carve):
        After carve() is called, the masked-dense weight is rebuilt into 128×128
        BCSR blocks (mortar_data + 6 index buffers).  Forward uses the vendored
        stk Triton dds kernel.  Topology is frozen after carving.

    The mode switch happens exactly once, at carve time, not at runtime.

    Args:
        in_features:        Input dimension (must be divisible by tile_size).
        out_features:       Output dimension (must be divisible by tile_size).
        bias:               Include bias term.  Default True.
        tile_size:          Saliency tile size.  16 is the default; execution
                            blocks are 128×128 (carve blocking).
        initial_density:    Starting density (fraction of tiles active per row).
                            Set to 1.0 for the fully-dense pre-carve phase.
        score_ema_alpha:    EMA decay for gradient importance scores.
        device:             Target device.
        dtype:              Parameter dtype (bf16 recommended for training).

    Example:
        >>> layer = MortarLinear(1536, 6144)
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
        """True if weights have been carved into MORTAR BCSR format."""
        return not self._cms._dense_mode

    @property
    def bias(self) -> Optional[nn.Parameter]:
        """Bias parameter (None if bias=False)."""
        return self._cms.bias

    # ── Scoring / carving ────────────────────────────────────────────────────

    def accumulate_scores(self) -> None:
        """Accumulate gradient statistics for importance scoring.

        Must be called AFTER loss.backward() and BEFORE optimizer.zero_grad().
        Reads weight.grad (dense mode) and updates the gradient-EMA tile
        importance scores.  Safe to call in either mode; no-op if grad is None
        or the layer is already carved.
        """
        self._cms.accumulate_scores()

    def carve(self, blocking: int = 128) -> int:
        """Pack the masked-dense weight into 128×128 BCSR blocks executed by the
        vendored stk Triton backend (3.09× faster than dense at 0.25 density —
        Gate G1). Pair with prune_step_blocks for lossless carving.

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

        Pre-carve:  F.linear(x, weight, bias) via cuBLAS.
        Post-carve: MORTAR BCSR Triton kernel (stk dds).

        Args:
            x: [..., in_features]

        Returns:
            [..., out_features]
        """
        return self._cms(x)

    def extra_repr(self) -> str:
        mode = "mortar" if self.is_compact else "dense"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"tile={self.tile_size}, density={self.get_density():.2f}, mode={mode}"
        )
