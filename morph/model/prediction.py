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

    def forward_step_boundary(self, h: Tensor, boundary_mask: Tensor) -> Tensor:
        """Consecutive STEP-BOUNDARY STP (arXiv:2604.18464, Eq 3) — the paper's central variant.

        Reads the hidden state at each reasoning-step boundary, forming the per-sequence step
        trajectory ``z_0..z_K``, and penalizes non-colinear consecutive STEP increments:

            L = mean_k [ 1 − cos(z_{k+1} − z_k,  z_k − z_{k−1}) ]

        averaged over all consecutive boundary triplets in the batch. Identical in FORM to the
        random-triplet ``forward`` but sampled at SEMANTIC step boundaries (the paper's
        168x-vs-4x point: *where* you sample dominates, not what the loss computes). ``h`` should
        be the HEAD-INPUT latent (post-coda; the exact state the LM head decodes) so the geometry
        being smoothed is the one the decode-fidelity metric reads.

        Args:
            h: ``[B, T, D]`` hidden states (pass float32 for numerical stability).
            boundary_mask: ``[B, T]`` bool, True at step-boundary token positions.
        Returns:
            Scalar loss in ``[0, 2]``; a ``0`` tensor if NO sequence has ≥3 boundaries.
        """
        B = h.shape[0]
        total = h.new_zeros(())
        n_terms = 0
        for b in range(B):
            idx = boundary_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [K] boundary positions
            if idx.numel() < 3:
                continue
            z = h[b].index_select(0, idx)                                # [K, D] step trajectory
            v = z[1:] - z[:-1]                                           # [K-1, D] step increments
            per = 1.0 - F.cosine_similarity(v[1:], v[:-1], dim=-1)       # [K-2] consecutive colinearity
            total = total + per.sum()
            n_terms += int(per.numel())
        if n_terms == 0:
            return h.new_zeros(())
        return total / n_terms


# ── LatentForecast ──────────────────────────────────────────────────────────────


class _AttnPredictor(nn.Module):
    """Causal single-layer self-attention over the hidden trajectory: predict each position's target
    by attending over h_{≤t} (Wolfe: 'predict the next hidden from PAST hidden, not just past
    context'). Causal mask ⇒ position t never sees the future latent it is asked to forecast."""

    def __init__(self, d_model: int, n_head: int = 4):
        super().__init__()
        self.nh = n_head
        self.dh = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, h: Tensor) -> Tensor:               # h [B,T,D] -> pred [B,T,D]
        B, T, D = h.shape
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, T, self.nh, self.dh).transpose(1, 2)   # [B,nh,T,dh]
        k = k.view(B, T, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, T, self.nh, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # causal: t attends ≤ t
        a = a.transpose(1, 2).reshape(B, T, D)
        return self.o(a)


class LatentForecast(nn.Module):
    """Forward latent-PREDICTION objective (distinct from STP's smoothness).

    From the head-input latent h_t, a small MLP head PREDICTS a future latent; the loss is
    1 − cos(pred, target.detach()) (JEPA-style: target is a stop-grad of the model's own future
    state, so the head must forecast, not the encoder collapse). Three horizons (Wolfe):

      h1    : predict h_{t+1}                      ("what am I doing at the very next step")
      end   : predict h_{b(t)}  (next period > t)  ("what am I doing at the END of this proposition")
      start : predict h_{b(t)+1}                   ("what am I doing at the START of the next one")

    `mode` is an underscore-joined subset, e.g. "h1", "end", "h1_end", "h1_end_start". One head per
    active horizon (so off-arm vs on-arm differ ONLY by which heads receive gradient). Boundary
    horizons need `boundary_mask`; if absent they contribute 0.
    """

    HORIZONS = ("h1", "end", "start")

    def __init__(self, d_model: int, hidden: int | None = None, predictor: str = "mlp",
                 n_head: int = 4):
        super().__init__()
        h = hidden or d_model
        self.predictor = predictor
        if predictor == "attn":
            # Wolfe's "give attention a chance to predict the next hidden from PAST hidden":
            # each head forecasts its target by CAUSALLY attending over the hidden trajectory
            # h_{≤t} (not from h_t alone). Causal ⇒ position t never sees the future it predicts.
            self.heads = nn.ModuleDict({k: _AttnPredictor(d_model, n_head) for k in self.HORIZONS})
        else:
            self.heads = nn.ModuleDict({
                k: nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, d_model))
                for k in self.HORIZONS
            })

    @staticmethod
    def _next_boundary_after(boundary_mask: Tensor) -> Tensor:
        """[B,T] long: for each t, the index of the next boundary STRICTLY after t (T if none).

        ⚠️ TESTING-ONLY TARGET HEURISTIC. `end` uses this boundary index; `start` uses +1 from it.
        The "+1 = first token of the next proposition" assumption holds ONLY for clean single-space
        "X. Word" text (verified: the space BPE-fuses into ' Word', period is a bare '.'). It is
        WRONG on terminator+newline (+1='\\n'), multi-space (+1=' '), and closing-punct clusters
        ('."' '.)' which BPE-fuse so the boundary is MISSED upstream in punct_boundary.py). On real
        prose the boundary targets are therefore NOISY — a candidate reason any forecast signal is
        weak. PROPER FIX (deferred): a real sentence segmenter → explicit boundary + next-content
        index map at data-prep. See punct_boundary.py warning + Ai-notes 06-25-2026 + CLAUDE.md."""
        B, T = boundary_mask.shape
        pos = torch.arange(T, device=boundary_mask.device).unsqueeze(0).expand(B, T)
        BIG = T
        bp = torch.where(boundary_mask, pos, torch.full_like(pos, BIG))      # [B,T] boundary idx or BIG
        # shift so position t sees boundaries at j>t, then reverse-cummin
        shifted = torch.cat([bp[:, 1:], bp.new_full((B, 1), BIG)], dim=1)    # bp[t+1..], pad BIG
        rev = torch.flip(shifted, dims=[1])
        rev_cummin = torch.cummin(rev, dim=1).values
        return torch.flip(rev_cummin, dims=[1])                              # [B,T] next-boundary-after-t

    def forward(self, h: Tensor, boundary_mask: Tensor | None, seq_lens: Tensor | None,
                mode: str) -> Tensor:
        """h: [B,T,D] head-input latent (float32). Returns mean over active horizons of
        1 − cos(head(h_t), h_target.detach()), masked to valid (in-seq, target-exists) positions."""
        B, T, D = h.shape
        dev = h.device
        horizons = [k for k in mode.split("_") if k in self.HORIZONS]
        if not horizons:
            return h.new_zeros(())
        pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)          # [B,T]
        L = (seq_lens.to(dev).clamp(max=T).unsqueeze(1) if seq_lens is not None
             else torch.full((B, 1), T, device=dev, dtype=torch.long))       # [B,1] valid len
        nb = self._next_boundary_after(boundary_mask) if boundary_mask is not None else None

        losses = []
        for k in horizons:
            if k == "h1":
                tgt_idx = (pos + 1)
            elif k == "end":
                if nb is None:
                    continue
                tgt_idx = nb
            else:  # start
                if nb is None:
                    continue
                tgt_idx = (nb + 1)
            valid = (pos < (L - 1)) & (tgt_idx < L) & (tgt_idx < T)          # [B,T] bool
            if not bool(valid.any()):
                continue
            tgt_idx_c = tgt_idx.clamp(max=T - 1)
            target = torch.gather(h, 1, tgt_idx_c.unsqueeze(-1).expand(B, T, D)).detach()
            pred = self.heads[k](h)                                           # [B,T,D]
            per = 1.0 - F.cosine_similarity(pred, target, dim=-1)            # [B,T]
            losses.append((per * valid).sum() / valid.sum().clamp(min=1))
        if not losses:
            return h.new_zeros(())
        return torch.stack(losses).mean()
