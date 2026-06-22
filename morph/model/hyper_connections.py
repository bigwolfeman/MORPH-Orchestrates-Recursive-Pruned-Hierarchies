"""Hyper-Connections residual (JPmHC) — orthogonal-manifold n-stream skip.

Widens the residual stream from a single ``C``-dim stream to ``n`` parallel ``C``-dim
streams and routes signal with three learnable, input-dependent mappings (Hyper-Connections,
Zhu et al. 2024, arXiv 2409.19606). The stream mixer ``H^res`` is constrained to the
**Stiefel/orthogonal** manifold via a Cayley transform (JPmHC, arXiv 2602.18308), giving
*exact* dynamical isometry (all singular values 1) with no iterative normalisation — on a
weight-tied *recursive* transformer (MORPH's looped core) this converges faster and scores
higher than a doubly-stochastic (Sinkhorn/mHC) mixer at lower compute (Table 1/2).

Mechanism (per token, identical for both manifolds — only the H^res projection differs):

    x_streams ∈ ℝ^{n×C}                       # n parallel residual streams
    [H̃pre | H̃post | H̃res] = W_fused · Norm(vec(x_streams))      # one fused projection
    Hpre  = softmax(H̃pre  / τ, dim=-1)        # row-stochastic  (read / aggregate streams)
    Hpost = softmax(H̃post / τ, dim=-2)        # col-stochastic  (write / fan output back)
    Hres  = P_M(H̃res)                          # manifold projection of the stream mixer
    x̄in = mean_n( Hpre · x_streams )           # single C-dim input the sublayer F sees
    y   = F(x̄in)                               # attention or MLP — unchanged, run ONCE
    x_out = Hres · x_streams + Hpost · (y ⊗ 1ₙ)   # mix streams + scatter sublayer output

where P_M = ``cayley``: orthogonal O(n) via Cayley transform of skew(H̃res).

Design notes
------------
- ``forward(h, sublayer_fn, *args, **kwargs)`` interface; ``MORPHBlock`` builds it at
  construction (branch-free hot path). The carrier is ``[B, S, n, C]``.
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


class HyperConnectionResidual(nn.Module):
    """n-stream manifold-constrained Hyper-Connection residual wrapper.

    Carries an ``[B, S, n, C]`` n-stream residual.

    Args:
        d_model:       per-stream feature width C.
        n_streams:     expansion rate n (paper default 4). n=1 ≡ plain residual.
        tau:           softmax temperature for Hpre / Hpost.
        cayley_iters:  Cayley fixed-point steps (s). Default 3.
        cayley_alpha:  Cayley step size α. Default 0.1.
        init_gain:     W_fused init std = init_gain / sqrt(n*d_model). Small ⇒ H̃≈0 ⇒
                       module ≈ plain residual at init. Default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        tau: float = 1.0,
        cayley_iters: int = 3,
        cayley_alpha: float = 0.1,
        init_gain: float = 0.1,
        use_kernel: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n = n_streams
        self.tau = tau
        self.cayley_iters = cayley_iters
        self.cayley_alpha = cayley_alpha

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

        # Branch-free hot path: resolve the carrier-op implementation at construction.
        # The fused Triton kernels (PRE x_bar / POST x_mix+x_post) run on CUDA; hc_pre / hc_post
        # fall back to their eager references on CPU / force_eager, so binding them here is safe
        # and adds NO runtime flag check to the math. `use_kernel=False` forces the eager
        # references even on cayley+cuda — the bit-faithful, slower reference arm for the
        # fused-vs-eager A/B (and the wandb-logged eager baseline).
        if use_kernel:
            from morph.kernels.triton.fused_hyper_connection import hc_pre_map, hc_post
            # Round 2: hc_pre_map fuses the WHOLE pre phase (rms+proj+softmax×2+cayley+
            # reductions+x_bar) into one kernel (+ a cuBLAS addmm GEMV) and returns
            # (x_bar, Hres, Hpost_row) directly — collapsing the ~65-launch eager mapping
            # storm. hc_post is unchanged (round 1).
            self._hc_pre_map = hc_pre_map
            self._hc_post = hc_post
            self._use_fused_premap = True
        else:
            from morph.kernels.triton.fused_hyper_connection import hc_post_reference
            self._hc_pre_map = None
            self._hc_post = hc_post_reference
            self._use_fused_premap = False

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
        Hres = cayley_orthogonal(res_raw, self.cayley_iters, self.cayley_alpha)
        return Hpre, Hpost, Hres

    def forward(
        self,
        h: Tensor,
        sublayer_fn: Callable[..., Tensor],
        *args,
        post_inject: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Apply the HC residual.

        Args:
            h:           [B, S, n, C] n-stream residual carrier.
            sublayer_fn: callable F taking a single [B, S, C] tensor → [B, S, C].
            post_inject: [B, S, C] | None — carrier-engine: the NEXT layer's single-stream
                         injection term, folded into the POST write (broadcast-added to every
                         output stream) so a separate _apply_injection carrier pass is skipped.
            *args/**kwargs: forwarded to sublayer_fn.

        Returns:
            [B, S, n, C] updated carrier.
        """
        dt = h.dtype
        if self._use_fused_premap:
            # Round 2: ONE fused kernel (+ cuBLAS GEMV) does rms+proj+softmax×2+cayley+
            # reductions+x_bar and returns (x_bar, Hres, Hpost_row) directly. Branch-free
            # on the cayley hot path (resolved at __init__); hc_pre_map itself falls back to
            # its reference on CPU / force_eager / n≠4 / iters≠3 so this call is always safe.
            x_bar, Hres, Hpost_row = self._hc_pre_map(
                h, self.proj.weight, self.proj.bias,
                self.tau, self.cayley_alpha, self.cayley_iters, self.eps,
            )
            Hres = Hres.to(dt)
            Hpost_row = Hpost_row.to(dt)
        else:
            # Sinkhorn / non-cayley: eager mapping (unchanged).
            Hpre, Hpost, Hres = self._mappings(h)
            Hpost_row = Hpost.sum(dim=-1).to(dt)
            Hpre_cm = Hpre.mean(dim=-2).to(dt)
            x_bar = torch.einsum("bsj,bsjc->bsc", Hpre_cm, h)
            Hres = Hres.to(dt)

        y = sublayer_fn(x_bar, *args, **kwargs)            # [B,S,C]

        # Skip: mix streams through the manifold-constrained mixer, then scatter the
        # shared sublayer output and add — fused into one carrier pass. The optional
        # post_inject term (carrier-engine) is broadcast-added to every output stream in
        # the SAME write, folding the next layer's _apply_injection into this kernel.
        return self._hc_post(Hres, Hpost_row, h, y, post_inject)  # [B,S,n,C]
