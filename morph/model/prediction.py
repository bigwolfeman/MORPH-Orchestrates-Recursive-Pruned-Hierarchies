"""STP — Semantic Tube Prediction regularizer (PyTorch).

Port/clean-room of the STP system from:
  TPU/subq-attention/looped_model_mhc.py  (PyTorch source — legacy name, pre-MRR rename)

Modules
-------
STPLoss
    Semantic Tube Prediction (Huang, LeCun, Balestriero 2026).
    Full-sequence vectorized geodesic smoothness regularizer.
    Always uses multi-scale geodesic mode: strides {1, 2, 4, ..., tau}.
    Zero parameters, zero runtime branches.

Used during pretraining as a geometric regularizer (the paper tested it during
fine-tuning only). Does not improve teacher-forced PPL — it improves generation
quality by enforcing locally-linear hidden-state trajectories.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── STPLoss ─────────────────────────────────────────────────────────────────────


class STPLoss(nn.Module):
    """Semantic Tube Prediction geodesic smoothness regularizer.

    Zero parameters. Tests that hidden-state trajectories are locally linear
    across multiple temporal scales (strides 1, 2, 4, ..., tau).

    Loss = mean over strides k of:
        (1 - cos(h[t+2k] - h[t+k], h[t+k] - h[t])) averaged over positions t.

    Args:
        tau: Maximum stride (default 64). Strides are powers of 2 up to tau.
    """

    def forward(self, h: Tensor, tau: int = 64) -> Tensor:
        """Compute multi-scale geodesic STP loss.

        Args:
            h: Post-norm hidden states [B, T, D]. Should be float32 for
               numerical stability (cast before calling if needed).
            tau: Maximum stride bound. Strides are {1, 2, 4, ..., tau}.

        Returns:
            Scalar loss in [0, 2]. Returns 0.0 tensor if T < 3.
        """
        T = h.shape[1]
        if T < 3:
            return h.new_zeros(())

        # Clamp tau so we always have at least one valid triplet
        effective_tau = min(tau, T // 2)

        losses: list[Tensor] = []
        k = 1
        while k <= effective_tau and T > 2 * k:
            # v_fwd[t] = h[t+2k] - h[t+k]  →  [B, T-2k, D]
            v_fwd = h[:, 2 * k:] - h[:, k:-k]
            # v_bwd[t] = h[t+k]  - h[t]    →  [B, T-2k, D]
            v_bwd = h[:, k:-k] - h[:, :-2 * k]
            losses.append(
                (1.0 - F.cosine_similarity(v_fwd, v_bwd, dim=-1)).mean()
            )
            k *= 2

        if not losses:
            return h.new_zeros(())

        return torch.stack(losses).mean()
