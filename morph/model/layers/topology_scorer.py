"""Tile-saliency scoring utilities for CMS block-sparse layers.

TopologyScorer maintains the gradient-EMA tile importance (block_score_ema)
that CMSBlockLinear.accumulate_scores feeds and prune_step_blocks pools into
128×128 MORTAR block saliency.

The legacy CMS Level-2 swap machinery (candidate scoring, epsilon-greedy
exploration, top-K column selection, column entropy) was removed 2026-06-11
together with the Block-ELL backend — pruning is the only topology action.
"""

import torch
from torch import Tensor


class TopologyScorer:
    """Maintains the gradient-EMA tile importance for block-sparse layers.

    Args:
        R: Number of output block-rows
        C: Number of input block-columns
        K: Active blocks per row
        ema_alpha: Momentum for gradient EMA (default 0.95)

    Example:
        >>> scorer = TopologyScorer(R=160, C=40, K=20)
        >>> grad_norms = torch.randn(160, 20).abs()  # [R, K]
        >>> ema = scorer.update_gradient_ema(grad_norms, torch.zeros(160, 20))
    """

    def __init__(
        self,
        R: int,
        C: int,
        K: int,
        ema_alpha: float = 0.95,
    ) -> None:
        """Initialize topology scorer."""
        if K > C:
            raise ValueError(f"K ({K}) cannot exceed C ({C})")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha ({ema_alpha}) must be in [0, 1]")

        self.R = R
        self.C = C
        self.K = K
        self.ema_alpha = ema_alpha

    def update_gradient_ema(
        self,
        grad_norms: Tensor,
        current_ema: Tensor,
    ) -> Tensor:
        """Update gradient EMA with new gradient norms.

        Implements: ema = alpha * new + (1 - alpha) * old

        Args:
            grad_norms: New gradient Frobenius norms [R, K]
            current_ema: Current EMA values [R, K]

        Returns:
            Updated EMA tensor [R, K]
        """
        # EMA formula: new_ema = alpha * grad_norms + (1 - alpha) * current_ema
        return self.ema_alpha * grad_norms + (1.0 - self.ema_alpha) * current_ema


def compute_gradient_frobenius_norms(grad: Tensor) -> Tensor:
    """Compute Frobenius norm of gradients per block.

    Frobenius norm = sqrt(sum of squared elements) per block.

    Args:
        grad: Gradient tensor [R, K, B, B]

    Returns:
        Frobenius norms [R, K]
    """
    # Frobenius norm: sqrt(sum of squared elements) per block
    # grad shape is [R, K, B, B], we want output [R, K]
    # Sum over the last two dimensions (B, B), then sqrt
    return torch.sqrt(torch.sum(grad * grad, dim=(-2, -1)))
