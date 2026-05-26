"""STP + LeJEPA split_nsm z-latent prediction modules (PyTorch).

Port/clean-room of the STP and z-latent (split_nsm) systems from:
  TPU/subq-attention/looped_model_mhc.py  (PyTorch source — legacy name, pre-MRR rename)
  titans_core/memory/sigreg.py            (SIGReg)

Modules
-------
SIGReg
    Epps-Pulley characteristic function anti-collapse regularizer.
    Enforces isotropic Gaussian distribution on embedding vectors.
    No learned parameters.

STPLoss
    Semantic Tube Prediction (Huang, LeCun, Balestriero 2026).
    Full-sequence vectorized geodesic smoothness regularizer.
    Always uses multi-scale geodesic mode: strides {1, 2, 4, ..., tau}.
    Zero parameters, zero runtime branches.

ZLatentHeads
    LeJEPA split_nsm dual-head projection system.
    z_projector: d_model → d_z (Linear + LayerNorm), used on coda and prelude.
    z_memory_head: d_model → d_z, predicts next prelude from memory retrieval.
    z_backbone_head: d_model → d_z, predicts mean(next z_coda) from prelude.

split_nsm_loss
    Pure function: accumulates delayed cross-segment z-losses.
    Backbone: mean(prelude[i]) → z_backbone_head → target = mean(z_coda[i+1])
    Memory:   mean(z_memory_raw[i]) → target = mean(z_prelude[i+1])
    Returns combined SmoothL1 loss + SIGReg term.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── SIGReg ─────────────────────────────────────────────────────────────────────


class SIGReg(nn.Module):
    """Epps-Pulley characteristic function test for anti-collapse regularization.

    Computes a statistic measuring deviation of the input distribution from
    an isotropic Gaussian. Returns a scalar loss — add to total loss with
    weight lambda (default 0.02).

    No learnable parameters. Random projection directions are regenerated
    each call for stochasticity.

    Args:
        knots: Number of quadrature points on [0, 3]. Default 17.
        n_slices: Number of random projection slices. Default 256.
    """

    def __init__(self, knots: int = 17, n_slices: int = 256):
        super().__init__()
        self.n_slices = n_slices

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """Compute SIGReg statistic.

        Args:
            proj: Embedding tensor [..., N, D] — operates on last two dims.

        Returns:
            Scalar loss measuring deviation from isotropic Gaussian.
        """
        # Random unit-norm projection directions
        A = torch.randn(proj.size(-1), self.n_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # [..., N, n_slices]
        projected = proj @ A

        # Characteristic function test at quadrature points: [..., N, n_slices, knots]
        x_t = projected.unsqueeze(-1) * self.t

        cos_mean = x_t.cos().mean(-3)   # mean over N dim
        sin_mean = x_t.sin().mean(-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integral, scaled by sample size
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


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


# ── ZLatentHeads ────────────────────────────────────────────────────────────────


class ZLatentHeads(nn.Module):
    """LeJEPA split_nsm dual-head projection system.

    Provides:
      z_projector      — projects any d_model tensor to z-latent space with
                         normalisation (shared between coda and prelude paths).
      z_memory_head    — linear predictor: memory retrieval → z_prelude target.
      z_backbone_head  — linear predictor: prelude hidden → mean(z_coda) target.

    Both prediction heads are Linear with no bias (following paper convention).
    z_projector uses LayerNorm for collapse resistance without EMA teacher.

    Args:
        d_model: Hidden dimension from the model.
        d_z: Z-latent dimension. Default 256.
    """

    def __init__(self, d_model: int, d_z: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_z = d_z

        # Shared projector — used for both coda and prelude inputs
        self.z_projector = nn.Sequential(
            nn.Linear(d_model, d_z),
            nn.LayerNorm(d_z),
        )
        nn.init.xavier_uniform_(self.z_projector[0].weight)
        nn.init.zeros_(self.z_projector[0].bias)

        # Memory path: retrieval → predict next z_prelude
        self.z_memory_head = nn.Linear(d_model, d_z, bias=False)
        nn.init.xavier_uniform_(self.z_memory_head.weight)

        # Backbone path: prelude hidden → predict mean of next z_coda
        self.z_backbone_head = nn.Linear(d_model, d_z, bias=False)
        nn.init.xavier_uniform_(self.z_backbone_head.weight)

    def project_coda(self, x: Tensor) -> Tensor:
        """Project coda hidden states to z-latent space.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_z]
        """
        return self.z_projector(x)

    def project_prelude(self, x: Tensor) -> Tensor:
        """Project prelude hidden states to z-latent space.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_z]
        """
        return self.z_projector(x)

    def memory_predict(self, retrieval: Tensor) -> Tensor:
        """Predict next segment's prelude z-latent from memory retrieval.

        Args:
            retrieval: [B, T, d_model] — output of memory.retrieve()

        Returns:
            [B, T, d_z]
        """
        return self.z_memory_head(retrieval)

    def backbone_predict(self, prelude: Tensor) -> Tensor:
        """Predict mean of next segment's z_coda from prelude hidden states.

        Args:
            prelude: [B, T, d_model] — raw (un-projected) prelude output

        Returns:
            [B, T, d_z]
        """
        return self.z_backbone_head(prelude)


# ── split_nsm_loss ───────────────────────────────────────────────────────────────


def split_nsm_loss(
    seg_z_codas: List[Tensor],
    seg_z_memories: List[Tensor],
    seg_z_preludes: List[Tensor],
    seg_x_preludes_raw: List[Tensor],
    backbone_head: nn.Module,
    sigreg: SIGReg,
    sigreg_weight: float = 0.02,
) -> Tensor:
    """Delayed cross-segment z-latent loss (split_nsm mode).

    Two orthogonal prediction tasks computed across consecutive segments:

      Backbone task:
        prelude[i] (d_model) → backbone_head → predict mean(z_coda[i+1])
        Backbone handles fast-path "what broad content comes next?"

      Memory task:
        z_memory_raw[i].mean → predict z_prelude[i+1].mean
        Memory handles slow-path "what detailed state will the next segment start in?"

    Both use SmoothL1 loss (robust to outliers, matches source implementation).
    SIGReg prevents collapse of the z-latent space.

    Args:
        seg_z_codas: List of [B, T, d_z] — projected coda outputs per segment.
                     Index 0 = first segment. Detached (no backprop through targets).
        seg_z_memories: List of [B, T, d_z] — memory retrieval projected to z-space.
        seg_z_preludes: List of [B, T, d_z] — projected prelude outputs per segment.
        seg_x_preludes_raw: List of [B, T, d_model] — raw (un-projected) prelude
                            hidden states, input to backbone_head.
        backbone_head: ZLatentHeads.z_backbone_head or equivalent Linear(d_model, d_z, bias=False).
        sigreg: SIGReg instance for anti-collapse regularization.
        sigreg_weight: Multiplier on SIGReg term. Default 0.02.

    Returns:
        Scalar loss. Zero tensor if fewer than 2 segments or no valid pairs.
    """
    losses: list[Tensor] = []

    # How many cross-segment pairs can we form?
    n_pairs = min(len(seg_z_memories), len(seg_z_codas) - 1)

    for i in range(n_pairs):
        # ── Backbone: prelude[i] → predict mean(z_coda[i+1]) ──────────────
        if i < len(seg_x_preludes_raw):
            # [B, T, d_z] → mean over T → [B, 1, d_z]
            bb_pred = backbone_head(seg_x_preludes_raw[i])
            bb_pred_mean = bb_pred.mean(dim=1, keepdim=True)
            # Target: detached mean of next coda → [B, 1, d_z]
            bb_target = seg_z_codas[i + 1].mean(dim=1, keepdim=True).detach()
            losses.append(F.smooth_l1_loss(bb_pred_mean, bb_target))

        # ── Memory: z_memory_raw[i] → predict z_prelude[i+1] ─────────────
        if i + 1 < len(seg_z_preludes):
            mem_mean = seg_z_memories[i].mean(dim=1, keepdim=True)
            prelude_target = seg_z_preludes[i + 1].mean(dim=1, keepdim=True).detach()
            losses.append(F.smooth_l1_loss(mem_mean, prelude_target))

    if not losses:
        # Return zero-grad tensor (no pairs)
        if seg_z_codas:
            return seg_z_codas[0].new_zeros(())
        return torch.zeros((), dtype=torch.float32)

    z_loss = torch.stack(losses).mean()

    # SIGReg: prevent z-space collapse. Concatenate all z_coda projections
    # across segments and measure isotropy.
    # Using coda projections (they're the richest z signal).
    if len(seg_z_codas) >= 1:
        # Stack across segments along the time dim: [B, sum_T, d_z]
        all_z = torch.cat(seg_z_codas, dim=1)
        # Collapse along B and T to get a flat set of d_z vectors
        B, T_total, d_z = all_z.shape
        all_z_flat = all_z.reshape(B * T_total, d_z)
        reg = sigreg(all_z_flat.unsqueeze(0))   # [1, N, d_z] → scalar
        z_loss = z_loss + sigreg_weight * reg

    return z_loss
