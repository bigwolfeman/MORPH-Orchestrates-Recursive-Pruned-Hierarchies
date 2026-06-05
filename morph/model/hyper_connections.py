"""Hyper-Connections residual (HC / mHC / JPmHC) — manifold-constrained n-stream skip.

This is the *real* Hyper-Connection mechanism that the per-channel ``MultiRateResidual``
(``mhc.py``) was a "half-assed" stand-in for. It widens the residual stream from a single
``C``-dim stream to ``n`` parallel ``C``-dim streams and routes signal with three learnable,
input-dependent mappings, exactly as in:

  - Hyper-Connections (HC), Zhu et al. 2024 (arXiv 2409.19606) — the base mechanism.
  - mHC, DeepSeek-AI 2025 (arXiv 2512.24880) — constrains H^res to the **Birkhoff polytope**
    (doubly-stochastic) via Sinkhorn-Knopp so the *depth-composite* ``∏ H^res`` stays
    norm-preserving (no exploding/vanishing residual stream across deep / looped stacks).
  - JPmHC, 2026 (arXiv 2602.18308) — generalises the manifold; the **Stiefel/orthogonal**
    mixer via a Cayley transform gives *exact* dynamical isometry (all singular values 1)
    with NO iterative normalisation, and on a weight-tied *recursive* transformer (the same
    regime as MORPH's looped core) converges faster + scores higher than the bistochastic
    Sinkhorn variant at lower compute (Table 1/2 of the paper).

Mechanism (per token, identical for both manifolds — only the H^res projection differs):

    x_streams ∈ ℝ^{n×C}                       # n parallel residual streams
    [H̃pre | H̃post | H̃res] = W_fused · Norm(vec(x_streams))      # one fused projection
    Hpre  = softmax(H̃pre  / τ, dim=-1)        # row-stochastic  (read / aggregate streams)
    Hpost = softmax(H̃post / τ, dim=-2)        # col-stochastic  (write / fan output back)
    Hres  = P_M(H̃res)                          # manifold projection of the stream mixer
    x̄in = mean_n( Hpre · x_streams )           # single C-dim input the sublayer F sees
    y   = F(x̄in)                               # attention or MLP — unchanged, run ONCE
    x_out = Hres · x_streams + Hpost · (y ⊗ 1ₙ)   # mix streams + scatter sublayer output

where P_M is:
  - ``cayley``   (JPmHC, DEFAULT): orthogonal O(n) via Cayley transform of skew(H̃res).
  - ``sinkhorn`` (mHC):            doubly-stochastic via Sinkhorn-Knopp iteration.

Design notes
------------
- Same ``forward(h, sublayer_fn, *args, **kwargs)`` interface as MultiRateResidual /
  StandardResidual, so ``MORPHBlock`` swaps it in at construction (branch-free hot path).
  The ONLY difference is the carrier shape: HC carries ``[B, S, n, C]`` instead of ``[B, S, C]``.
- The sublayer F always receives a single ``[B, S, C]`` tensor (the stream-averaged input),
  so attention / MLP modules are untouched and run at unchanged FLOPs.
- The mappings are computed in fp32 (the projections need precision; entries are bounded
  ≤1 so the bf16 *apply* is safe). torch.compile-friendly: no data-dependent control flow.
- Init ≈ plain residual: small W_fused init ⇒ H̃≈0 ⇒ Hpre,Hpost≈uniform and
  Hres≈I (Cayley(0)=I; Sinkhorn(0)=uniform→use a diagonal bias). With all streams equal at
  expand time the whole module reduces to ``x + F(mean(x))`` at step 0 (verified in the gate).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def cayley_orthogonal(A: Tensor, iters: int = 2, alpha: float = 0.1) -> Tensor:
    """Project a per-token matrix onto the orthogonal group O(n) via the Cayley transform.

    The Cayley map ``(I − W/2)(I + W/2)⁻¹`` sends a skew-symmetric ``W`` to an orthogonal
    matrix. The closed form needs a matrix inverse (expensive batched/per-token), so we use
    the inverse-free fixed-point iteration of Li et al. 2020 (JPmHC §3.1):

        W = A − Aᵀ                              # skew-symmetrise → so(n)
        Y₀ = I + α·W
        Yᵢ₊₁ = I + (α/2)·W·(I + Yᵢ),   i = 0…iters−1

    ``iters=2`` already gives ``‖YᵀY − I‖_max < 1e-3`` for n=4. Each step is one batched
    matmul; with n=4 the cost is negligible next to the sublayer.

    Args:
        A:     [..., n, n] unconstrained matrices (fp32 recommended).
        iters: fixed-point steps (s in the paper). Default 2.
        alpha: step size. Default 0.1.

    Returns:
        [..., n, n] approximately orthogonal matrices.
    """
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    W = A - A.transpose(-1, -2)
    Y = I + alpha * W
    half = alpha * 0.5
    for _ in range(iters):
        Y = I + half * (W @ (I + Y))
    return Y


def sinkhorn_bistochastic(A: Tensor, iters: int = 20) -> Tensor:
    """Project a per-token matrix onto the Birkhoff polytope (doubly-stochastic) — mHC.

    ``M⁰ = exp(A)`` then alternately column- and row-normalise (T_r ∘ T_c) ``iters`` times.
    The final op is row-normalisation, so rows sum to 1 exactly; columns converge to 1 as
    iters→∞ (mHC uses 20). A per-matrix max is subtracted before ``exp`` for stability — it
    is a constant scale that cancels in the normalisation, so the result is unchanged.

    Args:
        A:     [..., n, n] unconstrained logits (fp32 recommended).
        iters: Sinkhorn iterations. Default 20 (mHC value).

    Returns:
        [..., n, n] approximately doubly-stochastic matrices (rows exactly sum to 1).
    """
    M = (A - A.amax(dim=(-2, -1), keepdim=True)).exp()
    for _ in range(iters):
        M = M / M.sum(dim=-2, keepdim=True)   # T_c: column normalise
        M = M / M.sum(dim=-1, keepdim=True)   # T_r: row normalise
    return M


class HyperConnectionResidual(nn.Module):
    """n-stream manifold-constrained Hyper-Connection residual wrapper.

    Drop-in for MultiRateResidual / StandardResidual but carries ``[B, S, n, C]``.

    Args:
        d_model:       per-stream feature width C.
        n_streams:     expansion rate n (paper default 4). n=1 ≡ plain residual.
        manifold:      "cayley" (orthogonal, JPmHC — DEFAULT) | "sinkhorn" (bistochastic, mHC).
        tau:           softmax temperature for Hpre / Hpost.
        cayley_iters:  Cayley fixed-point steps (s). Default 2.
        cayley_alpha:  Cayley step size α. Default 0.1.
        sinkhorn_iters: Sinkhorn iterations. Default 20.
        init_gain:     W_fused init std = init_gain / sqrt(n*d_model). Small ⇒ H̃≈0 ⇒
                       module ≈ plain residual at init. Default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        manifold: str = "cayley",
        tau: float = 1.0,
        cayley_iters: int = 3,
        cayley_alpha: float = 0.1,
        sinkhorn_iters: int = 20,
        init_gain: float = 0.1,
    ):
        super().__init__()
        assert manifold in ("cayley", "sinkhorn"), f"unknown manifold {manifold!r}"
        self.d_model = d_model
        self.n = n_streams
        self.manifold = manifold
        self.tau = tau
        self.cayley_iters = cayley_iters
        self.cayley_alpha = cayley_alpha
        self.sinkhorn_iters = sinkhorn_iters

        nd = n_streams * d_model
        # One fused projection to the three n×n coefficient blocks: [Hpre | Hpost | Hres]
        # (mHC/JPmHC Eq. 12). The RMSNorm of the flattened stream is reordered to AFTER this
        # matmul (mHC §4.3.1): RMSNorm(x)·proj = proj(x)/rms since proj is linear and rms is a
        # per-token scalar — so we store only a [B,S,1] scalar, never the [B,S,nC] normalised
        # tensor. (RMSNorm affine would be absorbed into proj; we omit it.)
        self.eps = 1e-6
        self.proj = nn.Linear(nd, 3 * n_streams * n_streams, bias=True)
        nn.init.normal_(self.proj.weight, std=init_gain / math.sqrt(nd))
        nn.init.zeros_(self.proj.bias)

        # Sinkhorn cannot reach I from a zero logit (exp(0)=1 → uniform). Bias its H̃res
        # diagonal up so Sinkhorn(bias) ≈ I at init (matches the Cayley(0)=I behaviour).
        # Stored as a per-(i,j) additive bias on the H̃res block only; pre/post bias stays 0.
        if manifold == "sinkhorn":
            diag_bias = 6.0 * torch.eye(n_streams)        # exp(6) diag ⇒ ~0.99 after Sinkhorn
            self.register_buffer("_res_bias", diag_bias.reshape(1, 1, n_streams, n_streams))
        else:
            self.register_buffer("_res_bias", None, persistent=False)

    def _mappings(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (Hpre, Hpost, Hres) per token from the n-stream carrier X [B,S,n,C]."""
        B, S, n, C = X.shape
        x_flat = X.reshape(B, S, n * C)
        # RMSNorm reordered past the projection (see __init__): proj(x)/rms ≡ proj(RMSNorm(x)),
        # storing only the per-token scalar rms instead of the [B,S,nC] normalised tensor.
        rms = x_flat.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()   # [B,S,1]
        h = (self.proj(x_flat).float() / rms).reshape(B, S, 3, n, n)              # fp32 mappings
        pre_raw, post_raw, res_raw = h[:, :, 0], h[:, :, 1], h[:, :, 2]

        Hpre = torch.softmax(pre_raw / self.tau, dim=-1)    # row-stochastic
        Hpost = torch.softmax(post_raw / self.tau, dim=-2)  # column-stochastic
        if self.manifold == "cayley":
            Hres = cayley_orthogonal(res_raw, self.cayley_iters, self.cayley_alpha)
        else:
            Hres = sinkhorn_bistochastic(res_raw + self._res_bias, self.sinkhorn_iters)
        return Hpre, Hpost, Hres

    def forward(
        self,
        h: Tensor,
        sublayer_fn: Callable[..., Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        """Apply the HC residual.

        Args:
            h:           [B, S, n, C] n-stream residual carrier.
            sublayer_fn: callable F taking a single [B, S, C] tensor → [B, S, C].
            *args/**kwargs: forwarded to sublayer_fn.

        Returns:
            [B, S, n, C] updated carrier.
        """
        dt = h.dtype
        Hpre, Hpost, Hres = self._mappings(h)
        Hpost_row = Hpost.sum(dim=-1).to(dt)               # rowsum_i, since y is shared across streams

        # Read: aggregate streams → single sublayer input. x̄ = mean_i(Hpre·streams) folds the
        # mean into a column-mean of Hpre, contracting straight to [B,S,C] and never
        # materialising the [B,S,n,C] mixed-stream intermediate (exact). Memory + one launch.
        Hpre_cm = Hpre.mean(dim=-2).to(dt)                 # [B,S,j]  (mean over output stream i)
        x_bar = torch.einsum("bsj,bsjc->bsc", Hpre_cm, h)  # [B,S,C]

        y = sublayer_fn(x_bar, *args, **kwargs)            # [B,S,C]

        # Skip: mix streams through the manifold-constrained mixer.
        x_mix = torch.einsum("bsij,bsjc->bsic", Hres.to(dt), h)   # [B,S,n,C]
        # Write: Hpost · (y ⊗ 1ₙ) = rowsum_i(Hpost) · y  (y is identical across input streams).
        x_post = Hpost_row.unsqueeze(-1) * y.unsqueeze(2)         # [B,S,n,C]
        return x_mix + x_post
