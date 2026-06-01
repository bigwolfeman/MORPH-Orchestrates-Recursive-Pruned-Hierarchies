"""STP — Semantic Tube Prediction regularizer (JAX/Flax).

Port of the STP system from:
  TPU/subq-attention/looped_model_mrr.py  (PyTorch)

Modules
-------
STPLoss
    Semantic Tube Prediction (Huang, LeCun, Balestriero 2026).
    Full-sequence vectorized geodesic smoothness regularizer.
    Modes:
      "geodesic"    — multi-scale strides {1, 2, 4, ..., tau}, tau=64.
      "consecutive" — stride-1 all positions.
    Loss: 1 - cos(v_fwd, v_bwd) averaged over positions and scales.
    Zero parameters.

Used during pretraining as a geometric regularizer (the paper tested it during
fine-tuning only). Improves generation quality, not teacher-forced PPL.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


# ── STPLoss ────────────────────────────────────────────────────────────────────


class STPLoss(nn.Module):
    """Semantic Tube Prediction (STP) loss — zero parameters.

    Enforces geodesic linearity in the representation manifold:
    for any triplet (a, b, c) along a trajectory, the forward difference
    b→c should align with the backward difference a→b in cosine similarity.

    Modes
    -----
    "geodesic"
        Multi-scale: stride k triplets for k in {1, 2, 4, ..., tau}, tau ≤ stp_tau.
        Tests representational linearity at multiple scales simultaneously.
        Loss = mean over all scales and positions of (1 - cosine_similarity).

    "consecutive"
        Stride-1 only: consecutive triplets (t-2, t-1, t).
        Loss = mean over all positions of (1 - cosine_similarity).

    Attributes
    ----------
    mode    : str   "geodesic" | "consecutive"
    tau     : int   Maximum stride (geodesic mode). Power-of-2 halting.
    lam     : float Loss weight applied before adding to total loss.
    """

    mode: str = "geodesic"
    tau: int = 64
    lam: float = 0.02

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        """Compute STP loss.

        Args:
            h: [B, T, D] hidden states (float32 recommended; cast internally).

        Returns:
            Scalar loss (unweighted by lam — caller multiplies by self.lam).
        """
        h = h.astype(jnp.float32)  # STP in fp32 for numerical stability
        T = h.shape[1]

        if self.mode == "consecutive":
            return self._consecutive_loss(h, T)
        elif self.mode == "geodesic":
            return self._geodesic_loss(h, T)
        else:
            raise ValueError(f"Unknown STP mode: {self.mode!r}. Use 'geodesic' or 'consecutive'.")

    def _cosine_loss(self, v_fwd: jnp.ndarray, v_bwd: jnp.ndarray) -> jnp.ndarray:
        """1 - cos_sim(v_fwd, v_bwd), averaged over batch and position."""
        eps = 1e-8
        norm_fwd = jnp.linalg.norm(v_fwd, axis=-1, keepdims=True)
        norm_bwd = jnp.linalg.norm(v_bwd, axis=-1, keepdims=True)
        dot = (v_fwd * v_bwd).sum(axis=-1)
        cos_sim = dot / (norm_fwd.squeeze(-1) * norm_bwd.squeeze(-1) + eps)
        return (1.0 - cos_sim).mean()

    def _consecutive_loss(self, h: jnp.ndarray, T: int) -> jnp.ndarray:
        if T < 3:
            return jnp.zeros((), dtype=jnp.float32)
        v_fwd = h[:, 2:] - h[:, 1:-1]   # [B, T-2, D]
        v_bwd = h[:, 1:-1] - h[:, :-2]   # [B, T-2, D]
        return self._cosine_loss(v_fwd, v_bwd)

    def _geodesic_loss(self, h: jnp.ndarray, T: int) -> jnp.ndarray:
        if T < 3:
            return jnp.zeros((), dtype=jnp.float32)

        tau = min(self.tau, T // 2)
        losses = []
        k = 1
        while k <= tau and T > 2 * k:
            v_fwd = h[:, 2*k:] - h[:, k:-k]       # [B, T-2k, D]
            v_bwd = h[:, k:-k] - h[:, :-2*k]      # [B, T-2k, D]
            losses.append(self._cosine_loss(v_fwd, v_bwd))
            k *= 2

        if not losses:
            return jnp.zeros((), dtype=jnp.float32)

        return jnp.stack(losses).mean()
