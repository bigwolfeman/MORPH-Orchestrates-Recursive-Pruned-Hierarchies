"""STP — Semantic Tube Prediction regularizer (PyTorch).

Paper-faithful implementation of:
  Semantic Tube Prediction (Huang, LeCun, Balestriero 2026, arXiv:2602.22617).

Modules
-------
STPLoss
    Semantic Tube Prediction geodesic smoothness regularizer. Zero parameters.

Paper definition (single random triplet)
-----------------------------------------
For a last-layer post-norm hidden-state sequence h ∈ [B, T, D], pick a SINGLE
random ordered triplet of positions s < r < t whose span is bounded by a window
τ (``|t - s| ≤ τ``, default 64), and minimize:

    L_STP = 1 − cos(h_t − h_r, h_r − h_s)

i.e. enforce that the forward increment (h_t − h_r) is colinear with the
backward increment (h_r − h_s) — a locally-linear (geodesic) trajectory.

The indices are chosen RANDOMLY (uniformly over the constraint set), not via a
multi-stride ladder. An earlier version of this module summed a fixed
``{1, 2, 4, …, τ}`` stride ladder over ALL positions; that multi-scale ladder
was a NON-PAPER extension and has been REMOVED.

Intentional deviation
----------------------
We average ``n_samples`` independent triplet draws per sequence. This is a
variance-reduced Monte-Carlo estimate of the paper's single-triplet loss: the
estimator has the SAME expectation as the single random triplet (each draw is
an unbiased sample of E[1 − cos(·,·)]), but lower gradient variance. Set
``n_samples=1`` to recover the exact paper estimator.

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
    """Semantic Tube Prediction geodesic smoothness regularizer (paper-faithful).

    Zero parameters. For each sequence, samples random ordered triplets
    ``s < r < t`` with span ``(t − s) ≤ τ`` and minimizes
    ``1 − cos(h_t − h_r, h_r − h_s)``, encouraging locally-linear (geodesic)
    hidden-state trajectories.
    """

    def forward(
        self,
        h: Tensor,
        tau: int = 64,
        seq_lens: Tensor | None = None,
        n_samples: int = 4,
        generator: torch.Generator | None = None,
        return_indices: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Compute the paper-faithful random-triplet STP loss.

        Args:
            h: Post-norm hidden states ``[B, T, D]``. Should be float32 for
               numerical stability (cast before calling if needed).
            tau: Window bound on the triplet span ``(t − s) ≤ tau`` (default 64).
            seq_lens: Optional ``[B]`` long tensor of the number of VALID
               (non-pad) tokens per sequence. If ``None``, every sequence's
               valid length is ``T`` (the packed/pretraining use). Pad positions
               (``index ≥ L_b``) are never sampled.
            n_samples: Number of independent triplet draws per sequence to
               average (variance reduction). ``n_samples=1`` recovers the exact
               paper single-triplet estimator.
            generator: Optional ``torch.Generator`` used for ALL randomness, so
               the sampling is reproducible/seedable.
            return_indices: If ``True``, also return the sampled ``(s, r, t)``
               index tensors (each ``[B, n_samples]``). Used by tests to exercise
               the REAL sampler; the loss path is unchanged.

        Returns:
            Scalar loss in ``[0, 2]`` (or ``(loss, (s, r, t))`` if
            ``return_indices``). Returns a ``0`` tensor if NO sequence has valid
            length ``≥ 3``.
        """
        B, T, _ = h.shape
        device = h.device

        # Per-sequence valid lengths (default: full T).
        if seq_lens is None:
            L = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            L = seq_lens.to(device=device, dtype=torch.long).clamp(max=T)

        # A sequence can supply a triplet only if it has ≥ 3 valid positions.
        valid_seq = L >= 3                                   # [B] bool
        if not bool(valid_seq.any()):
            zero = h.new_zeros(())
            if return_indices:
                empty = torch.zeros(B, n_samples, dtype=torch.long, device=device)
                return zero, (empty, empty, empty)
            return zero

        # ── Vectorized constraint-respecting sampler over (B, n_samples) ──
        # Strategy (per (b, sample)):
        #   t  ∈ [2, L_b)                       (need ≥ 2 positions below t)
        #   d  ∈ [2, min(t, tau)]               (span; ≥ 2 so that s < r < t fits)
        #   s  = t − d                          (⇒ s ≥ 0, t − s = d ≤ tau)
        #   r  = s + 1 + randint(0, d − 1)      (⇒ s < r < t)
        # Floats in [0,1) from a single generator are mapped into each integer
        # range; everything is gathered with torch.gather (no python position loop).
        N = n_samples
        Lb = L.unsqueeze(1).expand(B, N)                     # [B, N] valid len per draw

        # u_t: t ∈ [2, L_b - 1]  →  t = 2 + floor(u * (L_b - 2))
        u_t = torch.rand(B, N, device=device, generator=generator)
        t_range = (Lb - 2).clamp(min=1)                      # #choices for t (≥1 where valid)
        t = 2 + (u_t * t_range.float()).floor().long()
        t = t.clamp(max=T - 1)                               # safety; t < T always

        # d: span ∈ [2, min(t, tau)]  →  d = 2 + floor(u * (dmax - 1))
        dmax = torch.minimum(t, torch.full_like(t, tau))     # [B, N], = min(t, tau)
        dmax = dmax.clamp(min=2)                             # ensure ≥ 2 (valid where t≥2)
        u_d = torch.rand(B, N, device=device, generator=generator)
        d = 2 + (u_d * (dmax - 1).float()).floor().long()
        d = torch.minimum(d, dmax)                           # safety clamp

        s = t - d                                            # ≥ 0
        # r ∈ (s, t):  r = s + 1 + floor(u * (d - 1)),  d ≥ 2 ⇒ ≥1 choice
        u_r = torch.rand(B, N, device=device, generator=generator)
        r = s + 1 + (u_r * (d - 1).float()).floor().long()
        r = r.clamp(min=s + 1, max=t - 1)                    # safety: keep s < r < t

        # ── Gather hidden states at sampled indices ──
        idx_s = s.unsqueeze(-1).expand(B, N, h.shape[-1])    # [B, N, D]
        idx_r = r.unsqueeze(-1).expand(B, N, h.shape[-1])
        idx_t = t.unsqueeze(-1).expand(B, N, h.shape[-1])
        h_s = torch.gather(h, 1, idx_s)                      # [B, N, D]
        h_r = torch.gather(h, 1, idx_r)
        h_t = torch.gather(h, 1, idx_t)

        # ── Geodesic colinearity loss per (b, sample) ──
        v_fwd = h_t - h_r                                    # forward increment
        v_bwd = h_r - h_s                                    # backward increment
        per_sample = 1.0 - F.cosine_similarity(v_fwd, v_bwd, dim=-1)   # [B, N]

        # Mask out sequences with L_b < 3 (their sampled triplets are garbage),
        # then average over all valid (b, sample) entries.
        mask = valid_seq.unsqueeze(1).expand(B, N)           # [B, N] bool
        denom = mask.sum()
        if denom == 0:
            loss = h.new_zeros(())
        else:
            loss = (per_sample * mask).sum() / denom
        if return_indices:
            return loss, (s, r, t)
        return loss
