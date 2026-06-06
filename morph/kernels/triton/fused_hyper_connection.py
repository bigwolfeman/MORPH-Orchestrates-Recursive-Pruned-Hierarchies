"""Fused Triton kernels for the JPmHC Hyper-Connection residual (sm_120 / Blackwell).

Targets ``HyperConnectionResidual.forward`` (``morph/model/hyper_connections.py``),
the production-default n-stream manifold residual. The carrier is ``h[B,S,n,C]``
(n=4 streams, C=768) = ~100 MB at B4/S4096 in bf16. The eager forward touches that
100 MB carrier *several* times (rms, proj, x_bar, x_mix, x_post) and fires ~10 tiny
launches for the n×n mapping math. This module fuses the BANDWIDTH-bound carrier
passes so h is read ONCE in the pre-phase and ONCE in the post-phase, and collapses
the post-phase's 3 carrier-touching ops (mix einsum + post broadcast-mul + add) into
a single kernel.

DECOMPOSITION — eager-manifold fallback (the sound, documented path)
-------------------------------------------------------------------
The n×n mapping (rms scalar, proj GEMV, softmax×2, Cayley 3-iter, the two reductions)
operates on ``[B,S,16]`` tensors — < 2 MB total, negligible bytes. It is kept in EAGER
PyTorch: autograd handles the hard softmax+Cayley backward EXACTLY, and a fully-fused
analytic Triton backward through 3 Cayley fixed-point iterations would add correctness
risk for a marginal (sub-1%-of-bytes) gain. We fuse only the BIG carrier passes:

    PRE  (``_FusedHCPre``):  x_bar[b,s,c]   = Σ_j Hpre_cm[b,s,j] · h[b,s,j,c]
    POST (``_FusedHCPost``): out[b,s,i,c]   = Σ_j Hres[b,s,i,j] · h[b,s,j,c]
                                            + Hpost_row[b,s,i] · y[b,s,c]

``h`` is an autograd input to BOTH functions; grad_h sums automatically across them —
that is intended; the two Functions straddle the Python sublayer call.

Hardware (RTX 5090 / sm_120):
  * num_stages=1, num_warps=8 (consumer Blackwell has no TMA pipeline).
  * bf16 in/out, fp32 accumulation for all reductions.
  * One program = one (b, s) token. The n=4 streams × C=768 carrier row is processed
    with C tiled into a single BLOCK_C = next_pow2(C) register block; n=4 unrolled
    with ``tl.static_range``. All reductions over C are in-register (fp32).

Author: TileProver (Claude Code, Opus 4.8)
Date:   2026-06-05
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


_LAUNCH = dict(num_stages=1, num_warps=8)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


# ===========================================================================
# Triton kernels
# ===========================================================================

if TRITON_AVAILABLE:

    # -----------------------------------------------------------------------
    # Device helpers for the n=4 mapping (Triton @jit functions return tuples).
    # -----------------------------------------------------------------------
    @triton.jit
    def _sm4(x0, x1, x2, x3):
        """4-way softmax (numerically stable)."""
        m = tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3))
        e0 = tl.exp(x0 - m); e1 = tl.exp(x1 - m); e2 = tl.exp(x2 - m); e3 = tl.exp(x3 - m)
        d = e0 + e1 + e2 + e3
        return e0 / d, e1 / d, e2 / d, e3 / d

    @triton.jit
    def _smjac4(g0, g1, g2, g3, p0, p1, p2, p3):
        """Softmax VJP for a 4-vector: g_z[k] = p[k]*(g[k] - Σ_l g[l] p[l])."""
        dot = g0 * p0 + g1 * p1 + g2 * p2 + g3 * p3
        return p0 * (g0 - dot), p1 * (g1 - dot), p2 * (g2 - dot), p3 * (g3 - dot)

    @triton.jit
    def _cayley_fwd_step(
        w00, w01, w02, w03, w10, w11, w12, w13,
        w20, w21, w22, w23, w30, w31, w32, w33,
        y00, y01, y02, y03, y10, y11, y12, y13,
        y20, y21, y22, y23, y30, y31, y32, y33, half,
    ):
        """One Cayley fixed-point step: Y' = I + half*(W @ (I + Y)). Returns Y'."""
        t00 = 1.0 + y00; t01 = y01;       t02 = y02;       t03 = y03
        t10 = y10;       t11 = 1.0 + y11; t12 = y12;       t13 = y13
        t20 = y20;       t21 = y21;       t22 = 1.0 + y22; t23 = y23
        t30 = y30;       t31 = y31;       t32 = y32;       t33 = 1.0 + y33
        p00 = w00*t00 + w01*t10 + w02*t20 + w03*t30
        p01 = w00*t01 + w01*t11 + w02*t21 + w03*t31
        p02 = w00*t02 + w01*t12 + w02*t22 + w03*t32
        p03 = w00*t03 + w01*t13 + w02*t23 + w03*t33
        p10 = w10*t00 + w11*t10 + w12*t20 + w13*t30
        p11 = w10*t01 + w11*t11 + w12*t21 + w13*t31
        p12 = w10*t02 + w11*t12 + w12*t22 + w13*t32
        p13 = w10*t03 + w11*t13 + w12*t23 + w13*t33
        p20 = w20*t00 + w21*t10 + w22*t20 + w23*t30
        p21 = w20*t01 + w21*t11 + w22*t21 + w23*t31
        p22 = w20*t02 + w21*t12 + w22*t22 + w23*t32
        p23 = w20*t03 + w21*t13 + w22*t23 + w23*t33
        p30 = w30*t00 + w31*t10 + w32*t20 + w33*t30
        p31 = w30*t01 + w31*t11 + w32*t21 + w33*t31
        p32 = w30*t02 + w31*t12 + w32*t22 + w33*t32
        p33 = w30*t03 + w31*t13 + w32*t23 + w33*t33
        return (1.0+half*p00, half*p01, half*p02, half*p03,
                half*p10, 1.0+half*p11, half*p12, half*p13,
                half*p20, half*p21, 1.0+half*p22, half*p23,
                half*p30, half*p31, half*p32, 1.0+half*p33)

    @triton.jit
    def _cayley_bwd_step(
        gW00, gW01, gW02, gW03, gW10, gW11, gW12, gW13,
        gW20, gW21, gW22, gW23, gW30, gW31, gW32, gW33,
        gY00, gY01, gY02, gY03, gY10, gY11, gY12, gY13,
        gY20, gY21, gY22, gY23, gY30, gY31, gY32, gY33,
        w00, w01, w02, w03, w10, w11, w12, w13,
        w20, w21, w22, w23, w30, w31, w32, w33,
        t00, t01, t02, t03, t10, t11, t12, t13,
        t20, t21, t22, t23, t30, t31, t32, t33, half,
    ):
        """Reverse of one Cayley step. Given gY (grad wrt Y_k) and the forward T_{k-1}=I+Y_{k-1},
        accumulate gW += half*gY@T^T and return (gW_acc, gY_prev=half*W^T@gY)."""
        gW00 += half*(gY00*t00 + gY01*t01 + gY02*t02 + gY03*t03)
        gW01 += half*(gY00*t10 + gY01*t11 + gY02*t12 + gY03*t13)
        gW02 += half*(gY00*t20 + gY01*t21 + gY02*t22 + gY03*t23)
        gW03 += half*(gY00*t30 + gY01*t31 + gY02*t32 + gY03*t33)
        gW10 += half*(gY10*t00 + gY11*t01 + gY12*t02 + gY13*t03)
        gW11 += half*(gY10*t10 + gY11*t11 + gY12*t12 + gY13*t13)
        gW12 += half*(gY10*t20 + gY11*t21 + gY12*t22 + gY13*t23)
        gW13 += half*(gY10*t30 + gY11*t31 + gY12*t32 + gY13*t33)
        gW20 += half*(gY20*t00 + gY21*t01 + gY22*t02 + gY23*t03)
        gW21 += half*(gY20*t10 + gY21*t11 + gY22*t12 + gY23*t13)
        gW22 += half*(gY20*t20 + gY21*t21 + gY22*t22 + gY23*t23)
        gW23 += half*(gY20*t30 + gY21*t31 + gY22*t32 + gY23*t33)
        gW30 += half*(gY30*t00 + gY31*t01 + gY32*t02 + gY33*t03)
        gW31 += half*(gY30*t10 + gY31*t11 + gY32*t12 + gY33*t13)
        gW32 += half*(gY30*t20 + gY31*t21 + gY32*t22 + gY33*t23)
        gW33 += half*(gY30*t30 + gY31*t31 + gY32*t32 + gY33*t33)
        ngY00 = half*(w00*gY00 + w10*gY10 + w20*gY20 + w30*gY30)
        ngY01 = half*(w00*gY01 + w10*gY11 + w20*gY21 + w30*gY31)
        ngY02 = half*(w00*gY02 + w10*gY12 + w20*gY22 + w30*gY32)
        ngY03 = half*(w00*gY03 + w10*gY13 + w20*gY23 + w30*gY33)
        ngY10 = half*(w01*gY00 + w11*gY10 + w21*gY20 + w31*gY30)
        ngY11 = half*(w01*gY01 + w11*gY11 + w21*gY21 + w31*gY31)
        ngY12 = half*(w01*gY02 + w11*gY12 + w21*gY22 + w31*gY32)
        ngY13 = half*(w01*gY03 + w11*gY13 + w21*gY23 + w31*gY33)
        ngY20 = half*(w02*gY00 + w12*gY10 + w22*gY20 + w32*gY30)
        ngY21 = half*(w02*gY01 + w12*gY11 + w22*gY21 + w32*gY31)
        ngY22 = half*(w02*gY02 + w12*gY12 + w22*gY22 + w32*gY32)
        ngY23 = half*(w02*gY03 + w12*gY13 + w22*gY23 + w32*gY33)
        ngY30 = half*(w03*gY00 + w13*gY10 + w23*gY20 + w33*gY30)
        ngY31 = half*(w03*gY01 + w13*gY11 + w23*gY21 + w33*gY31)
        ngY32 = half*(w03*gY02 + w13*gY12 + w23*gY22 + w33*gY32)
        ngY33 = half*(w03*gY03 + w13*gY13 + w23*gY23 + w33*gY33)
        return (gW00, gW01, gW02, gW03, gW10, gW11, gW12, gW13,
                gW20, gW21, gW22, gW23, gW30, gW31, gW32, gW33,
                ngY00, ngY01, ngY02, ngY03, ngY10, ngY11, ngY12, ngY13,
                ngY20, ngY21, ngY22, ngY23, ngY30, ngY31, ngY32, ngY33)

    # -----------------------------------------------------------------------
    # PRE forward: x_bar[b,s,c] = sum_j Hpre_cm[b,s,j] * h[b,s,j,c]
    # One program = one (b,s) token. n streams unrolled, C in one register block.
    # -----------------------------------------------------------------------
    @triton.jit
    def _hc_pre_fwd_kernel(
        h_ptr,            # [B, S, n, C] bf16
        hpre_ptr,         # [B, S, n]    (Hpre_cm)
        xbar_ptr,         # [B, S, C]    out
        N: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)                 # over B*S
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        h_base = tok * (N * C)
        for j in tl.static_range(N):
            hj = tl.load(h_ptr + h_base + j * C + c, mask=cmask, other=0.0).to(tl.float32)
            wj = tl.load(hpre_ptr + tok * N + j).to(tl.float32)
            acc += wj * hj

        tl.store(xbar_ptr + tok * C + c, acc.to(xbar_ptr.dtype.element_ty), mask=cmask)

    # -----------------------------------------------------------------------
    # PRE backward:
    #   grad_h[b,s,j,c]      = Hpre_cm[b,s,j] * grad_xbar[b,s,c]
    #   grad_Hpre_cm[b,s,j]  = sum_c grad_xbar[b,s,c] * h[b,s,j,c]
    # -----------------------------------------------------------------------
    @triton.jit
    def _hc_pre_bwd_kernel(
        gxbar_ptr,        # [B, S, C]
        h_ptr,            # [B, S, n, C]
        hpre_ptr,         # [B, S, n]
        gh_ptr,           # [B, S, n, C]   out (grad wrt h, this path)
        ghpre_ptr,        # [B, S, n]      out (grad wrt Hpre_cm)
        N: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        g = tl.load(gxbar_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)
        h_base = tok * (N * C)
        for j in tl.static_range(N):
            wj = tl.load(hpre_ptr + tok * N + j).to(tl.float32)
            hj = tl.load(h_ptr + h_base + j * C + c, mask=cmask, other=0.0).to(tl.float32)
            # grad wrt h (this path only; the post kernel adds its own)
            gh = wj * g
            tl.store(gh_ptr + h_base + j * C + c, gh.to(gh_ptr.dtype.element_ty), mask=cmask)
            # grad wrt Hpre_cm[j] = sum_c g * h_j
            ghpre = tl.sum(g * hj, axis=0)
            tl.store(ghpre_ptr + tok * N + j, ghpre)

    # -----------------------------------------------------------------------
    # PRE-MAPPING forward (round 2): the full n×n mapping + x_bar in ONE kernel.
    #
    # Per (b,s) token, given raw_full[48] = proj_w @ x_flat + proj_b (cuBLAS addmm,
    # pre-rms) and the carrier h[n,C]:
    #   rms      = sqrt(mean_{nC}(h^2) + eps)                 (folds the BIG pow/mean)
    #   raw[48]  = raw_full / rms   -> 3 blocks of [4,4]: pre | post | res
    #   Hpre     = softmax(pre/τ, dim=-1)   (rows)
    #   Hpost    = softmax(post/τ, dim=-2)  (cols)
    #   Hres     = cayley(res, iters, α)    (3 unrolled 4×4 fixed-point iters)
    #   Hpre_cm  = colmean(Hpre)            [4]   (mean over output stream i)
    #   Hpost_row= rowsum(Hpost)            [4]
    #   x_bar[c] = Σ_j Hpre_cm[j]·h[j,c]
    # Every lane redundantly computes the 48-float mapping in fp32 scalar registers
    # (n=4 fully unrolled, no cross-lane reduction, no shared memory); the rms reduction
    # and x_bar contraction use the BLOCK_C register tile. Outputs Hres[4,4],
    # Hpost_row[4], x_bar[C]. N is fixed to 4 (NN=16) — the mapping unroll assumes it.
    # -----------------------------------------------------------------------
    @triton.jit
    def _hc_premap_fwd_kernel(
        h_ptr,            # [B, S, n, C] bf16
        raw_ptr,          # [B, S, 48]   fp32  (proj_w @ x_flat + proj_b, pre-rms)
        xbar_ptr,         # [B, S, C]    bf16  out
        hres_ptr,         # [B, S, 4, 4] fp32  out
        hpostrow_ptr,     # [B, S, 4]    fp32  out
        hprecm_ptr,       # [B, S, 4]    fp32  out  (saved for backward)
        rms_ptr,          # [B, S, 1]    fp32  out  (saved for backward)
        TAU: tl.constexpr, ALPHA: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
        N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        h_base = tok * (N * C)

        # --- rms over the n*C carrier (fp32) ---
        ssq = tl.zeros((), dtype=tl.float32)
        for j in tl.static_range(N):
            hj = tl.load(h_ptr + h_base + j * C + c, mask=cmask, other=0.0).to(tl.float32)
            ssq += tl.sum(hj * hj, axis=0)
        rms = tl.sqrt(ssq / (N * C) + EPS)
        inv_rms = 1.0 / rms
        tl.store(rms_ptr + tok, rms)

        rb = tok * 48
        # raw blocks: pre = raw[0:16], post = raw[16:32], res = raw[32:48]; row-major [4,4].
        # Each loaded scalar already /rms and (for softmax) /tau folded in.
        # ---- Hpre = softmax(pre/tau, dim=-1) : per row i over j ----
        # ---- Hpost = softmax(post/tau, dim=-2) : per col j over i ----
        # We fully unroll n=4.
        inv_tau = 1.0 / TAU

        # load pre (16), post (16), res (16) as fp32 scalars, scaled by inv_rms
        # pre[i,j] => raw[i*4+j]; post[i,j] => raw[16 + i*4+j]; res[i,j] => raw[32 + i*4+j]
        # Hpre_cm[j] = (1/4) sum_i Hpre[i,j]
        hprecm0 = tl.zeros((), dtype=tl.float32)
        hprecm1 = tl.zeros((), dtype=tl.float32)
        hprecm2 = tl.zeros((), dtype=tl.float32)
        hprecm3 = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(4):
            p0 = tl.load(raw_ptr + rb + i * 4 + 0) * inv_rms * inv_tau
            p1 = tl.load(raw_ptr + rb + i * 4 + 1) * inv_rms * inv_tau
            p2 = tl.load(raw_ptr + rb + i * 4 + 2) * inv_rms * inv_tau
            p3 = tl.load(raw_ptr + rb + i * 4 + 3) * inv_rms * inv_tau
            s0, s1, s2, s3 = _sm4(p0, p1, p2, p3)
            hprecm0 += s0
            hprecm1 += s1
            hprecm2 += s2
            hprecm3 += s3
        hprecm0 *= 0.25; hprecm1 *= 0.25; hprecm2 *= 0.25; hprecm3 *= 0.25
        tl.store(hprecm_ptr + tok * 4 + 0, hprecm0)
        tl.store(hprecm_ptr + tok * 4 + 1, hprecm1)
        tl.store(hprecm_ptr + tok * 4 + 2, hprecm2)
        tl.store(hprecm_ptr + tok * 4 + 3, hprecm3)

        # Hpost = softmax over dim=-2 (columns): for each column j, softmax across rows i.
        # Hpost_row[i] = sum_j Hpost[i,j].
        hpr0 = tl.zeros((), dtype=tl.float32)
        hpr1 = tl.zeros((), dtype=tl.float32)
        hpr2 = tl.zeros((), dtype=tl.float32)
        hpr3 = tl.zeros((), dtype=tl.float32)
        for j in tl.static_range(4):
            q0 = tl.load(raw_ptr + rb + 16 + 0 * 4 + j) * inv_rms * inv_tau
            q1 = tl.load(raw_ptr + rb + 16 + 1 * 4 + j) * inv_rms * inv_tau
            q2 = tl.load(raw_ptr + rb + 16 + 2 * 4 + j) * inv_rms * inv_tau
            q3 = tl.load(raw_ptr + rb + 16 + 3 * 4 + j) * inv_rms * inv_tau
            s0, s1, s2, s3 = _sm4(q0, q1, q2, q3)
            hpr0 += s0
            hpr1 += s1
            hpr2 += s2
            hpr3 += s3
        tl.store(hpostrow_ptr + tok * 4 + 0, hpr0)
        tl.store(hpostrow_ptr + tok * 4 + 1, hpr1)
        tl.store(hpostrow_ptr + tok * 4 + 2, hpr2)
        tl.store(hpostrow_ptr + tok * 4 + 3, hpr3)

        # Hres = cayley(res): W = A - A^T (skew), Y0 = I + alpha*W,
        #        Y_{k+1} = I + (alpha/2) * W @ (I + Y_k).  Unroll ITERS.
        # res[i,j] = raw[32 + i*4+j]*inv_rms (NO tau on res).
        a00 = tl.load(raw_ptr + rb + 32 + 0) * inv_rms
        a01 = tl.load(raw_ptr + rb + 32 + 1) * inv_rms
        a02 = tl.load(raw_ptr + rb + 32 + 2) * inv_rms
        a03 = tl.load(raw_ptr + rb + 32 + 3) * inv_rms
        a10 = tl.load(raw_ptr + rb + 32 + 4) * inv_rms
        a11 = tl.load(raw_ptr + rb + 32 + 5) * inv_rms
        a12 = tl.load(raw_ptr + rb + 32 + 6) * inv_rms
        a13 = tl.load(raw_ptr + rb + 32 + 7) * inv_rms
        a20 = tl.load(raw_ptr + rb + 32 + 8) * inv_rms
        a21 = tl.load(raw_ptr + rb + 32 + 9) * inv_rms
        a22 = tl.load(raw_ptr + rb + 32 + 10) * inv_rms
        a23 = tl.load(raw_ptr + rb + 32 + 11) * inv_rms
        a30 = tl.load(raw_ptr + rb + 32 + 12) * inv_rms
        a31 = tl.load(raw_ptr + rb + 32 + 13) * inv_rms
        a32 = tl.load(raw_ptr + rb + 32 + 14) * inv_rms
        a33 = tl.load(raw_ptr + rb + 32 + 15) * inv_rms
        # skew W = A - A^T  (diagonal = 0)
        w00 = 0.0;            w01 = a01 - a10;     w02 = a02 - a20;     w03 = a03 - a30
        w10 = a10 - a01;      w11 = 0.0;           w12 = a12 - a21;     w13 = a13 - a31
        w20 = a20 - a02;      w21 = a21 - a12;     w22 = 0.0;           w23 = a23 - a32
        w30 = a30 - a03;      w31 = a31 - a13;     w32 = a32 - a23;     w33 = 0.0
        half = ALPHA * 0.5
        # Y0 = I + alpha*W
        y00 = 1.0 + ALPHA * w00; y01 = ALPHA * w01;       y02 = ALPHA * w02;       y03 = ALPHA * w03
        y10 = ALPHA * w10;       y11 = 1.0 + ALPHA * w11; y12 = ALPHA * w12;       y13 = ALPHA * w13
        y20 = ALPHA * w20;       y21 = ALPHA * w21;       y22 = 1.0 + ALPHA * w22; y23 = ALPHA * w23
        y30 = ALPHA * w30;       y31 = ALPHA * w31;       y32 = ALPHA * w32;       y33 = 1.0 + ALPHA * w33
        for _ in tl.static_range(ITERS):
            (y00, y01, y02, y03, y10, y11, y12, y13,
             y20, y21, y22, y23, y30, y31, y32, y33) = _cayley_fwd_step(
                w00, w01, w02, w03, w10, w11, w12, w13,
                w20, w21, w22, w23, w30, w31, w32, w33,
                y00, y01, y02, y03, y10, y11, y12, y13,
                y20, y21, y22, y23, y30, y31, y32, y33, half)
        hb = tok * 16
        tl.store(hres_ptr + hb + 0, y00);  tl.store(hres_ptr + hb + 1, y01)
        tl.store(hres_ptr + hb + 2, y02);  tl.store(hres_ptr + hb + 3, y03)
        tl.store(hres_ptr + hb + 4, y10);  tl.store(hres_ptr + hb + 5, y11)
        tl.store(hres_ptr + hb + 6, y12);  tl.store(hres_ptr + hb + 7, y13)
        tl.store(hres_ptr + hb + 8, y20);  tl.store(hres_ptr + hb + 9, y21)
        tl.store(hres_ptr + hb + 10, y22); tl.store(hres_ptr + hb + 11, y23)
        tl.store(hres_ptr + hb + 12, y30); tl.store(hres_ptr + hb + 13, y31)
        tl.store(hres_ptr + hb + 14, y32); tl.store(hres_ptr + hb + 15, y33)

        # --- x_bar[c] = Σ_j Hpre_cm[j] * h[j,c] (reuse the BLOCK_C tile) ---
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        h0 = tl.load(h_ptr + h_base + 0 * C + c, mask=cmask, other=0.0).to(tl.float32)
        h1 = tl.load(h_ptr + h_base + 1 * C + c, mask=cmask, other=0.0).to(tl.float32)
        h2 = tl.load(h_ptr + h_base + 2 * C + c, mask=cmask, other=0.0).to(tl.float32)
        h3 = tl.load(h_ptr + h_base + 3 * C + c, mask=cmask, other=0.0).to(tl.float32)
        acc = hprecm0 * h0 + hprecm1 * h1 + hprecm2 * h2 + hprecm3 * h3
        tl.store(xbar_ptr + tok * C + c, acc.to(xbar_ptr.dtype.element_ty), mask=cmask)

    # -----------------------------------------------------------------------
    # PRE-MAPPING backward (round 2): analytic VJP through the whole mapping.
    #
    # Given upstream grads (g_xbar[C], g_Hres[4,4], g_Hpostrow[4]) and saved
    # (h[n,C], raw_full[48], rms scalar), recompute the forward mapping and
    # backprop:  x_bar -> Hpre_cm + h ; Hpostrow -> rowsum -> softmax(-2) -> post ;
    # Hpre_cm -> colmean -> softmax(-1) -> pre ; Hres -> cayley(3) -> W -> res.
    # Stack [pre|post|res] grads (post-rms), then /rms VJP -> grad_raw_full[48]
    # (fed to the addmm backward eagerly for grad_w/grad_b/grad_h_proj) AND the
    # rms-path grad on h. Outputs: grad_raw_full[48], grad_h_partial[n,C]
    # (= x_bar-path + rms-path; the proj-path is added eagerly via cuBLAS).
    # Validated against autograd to fp32 (rel ~1e-7, ignore/derive_pre_backward.py).
    # -----------------------------------------------------------------------
    @triton.jit
    def _hc_premap_bwd_kernel(
        gxbar_ptr,        # [B,S,C]    upstream grad on x_bar
        ghres_ptr,        # [B,S,4,4]  upstream grad on Hres
        ghpostrow_ptr,    # [B,S,4]    upstream grad on Hpost_row
        h_ptr,            # [B,S,n,C]
        raw_ptr,          # [B,S,48]   raw_full (pre-rms)
        rms_ptr,          # [B,S,1]
        graw_ptr,         # [B,S,48]   out: grad wrt raw_full (pre-rms)
        ghpart_ptr,       # [B,S,n,C]  out: grad on h (xbar-path + rms-path)
        TAU: tl.constexpr, ALPHA: tl.constexpr, ITERS: tl.constexpr,
        N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        h_base = tok * (N * C)
        rb = tok * 48
        hb = tok * 16

        rms = tl.load(rms_ptr + tok)
        inv_rms = 1.0 / rms
        inv_tau = 1.0 / TAU

        # ============ recompute forward mapping (fp32 scalars) ============
        # Hpre = softmax(pre/tau, -1) rows ; keep Hpre[i,j] and Hpre_cm[j].
        # store per-row exp/denom so we can do softmax jacobian in bwd.
        # We recompute fully; cheap (48 floats).
        # ---- pre (rows softmax) ----
        # row i values
        # We'll need Hpre[i,j]. Keep all 16.
        # load pre raw scaled (inv_rms*inv_tau already includes tau)
        pre00 = tl.load(raw_ptr+rb+0)*inv_rms*inv_tau; pre01 = tl.load(raw_ptr+rb+1)*inv_rms*inv_tau
        pre02 = tl.load(raw_ptr+rb+2)*inv_rms*inv_tau; pre03 = tl.load(raw_ptr+rb+3)*inv_rms*inv_tau
        pre10 = tl.load(raw_ptr+rb+4)*inv_rms*inv_tau; pre11 = tl.load(raw_ptr+rb+5)*inv_rms*inv_tau
        pre12 = tl.load(raw_ptr+rb+6)*inv_rms*inv_tau; pre13 = tl.load(raw_ptr+rb+7)*inv_rms*inv_tau
        pre20 = tl.load(raw_ptr+rb+8)*inv_rms*inv_tau; pre21 = tl.load(raw_ptr+rb+9)*inv_rms*inv_tau
        pre22 = tl.load(raw_ptr+rb+10)*inv_rms*inv_tau; pre23 = tl.load(raw_ptr+rb+11)*inv_rms*inv_tau
        pre30 = tl.load(raw_ptr+rb+12)*inv_rms*inv_tau; pre31 = tl.load(raw_ptr+rb+13)*inv_rms*inv_tau
        pre32 = tl.load(raw_ptr+rb+14)*inv_rms*inv_tau; pre33 = tl.load(raw_ptr+rb+15)*inv_rms*inv_tau

        P00, P01, P02, P03 = _sm4(pre00, pre01, pre02, pre03)
        P10, P11, P12, P13 = _sm4(pre10, pre11, pre12, pre13)
        P20, P21, P22, P23 = _sm4(pre20, pre21, pre22, pre23)
        P30, P31, P32, P33 = _sm4(pre30, pre31, pre32, pre33)

        # ---- post (cols softmax, dim=-2): for col j softmax across rows i ----
        po00 = tl.load(raw_ptr+rb+16+0)*inv_rms*inv_tau; po01 = tl.load(raw_ptr+rb+16+1)*inv_rms*inv_tau
        po02 = tl.load(raw_ptr+rb+16+2)*inv_rms*inv_tau; po03 = tl.load(raw_ptr+rb+16+3)*inv_rms*inv_tau
        po10 = tl.load(raw_ptr+rb+16+4)*inv_rms*inv_tau; po11 = tl.load(raw_ptr+rb+16+5)*inv_rms*inv_tau
        po12 = tl.load(raw_ptr+rb+16+6)*inv_rms*inv_tau; po13 = tl.load(raw_ptr+rb+16+7)*inv_rms*inv_tau
        po20 = tl.load(raw_ptr+rb+16+8)*inv_rms*inv_tau; po21 = tl.load(raw_ptr+rb+16+9)*inv_rms*inv_tau
        po22 = tl.load(raw_ptr+rb+16+10)*inv_rms*inv_tau; po23 = tl.load(raw_ptr+rb+16+11)*inv_rms*inv_tau
        po30 = tl.load(raw_ptr+rb+16+12)*inv_rms*inv_tau; po31 = tl.load(raw_ptr+rb+16+13)*inv_rms*inv_tau
        po32 = tl.load(raw_ptr+rb+16+14)*inv_rms*inv_tau; po33 = tl.load(raw_ptr+rb+16+15)*inv_rms*inv_tau
        # column j softmax across i: col0 = (po00,po10,po20,po30) etc.
        Q00, Q10, Q20, Q30 = _sm4(po00, po10, po20, po30)   # col 0 -> Hpost[i,0]
        Q01, Q11, Q21, Q31 = _sm4(po01, po11, po21, po31)   # col 1
        Q02, Q12, Q22, Q32 = _sm4(po02, po12, po22, po32)   # col 2
        Q03, Q13, Q23, Q33 = _sm4(po03, po13, po23, po33)   # col 3

        # ---- cayley forward, save Y_k for each iter (need I+Y_{k-1} & W) ----
        a00 = tl.load(raw_ptr+rb+32+0)*inv_rms;  a01 = tl.load(raw_ptr+rb+32+1)*inv_rms
        a02 = tl.load(raw_ptr+rb+32+2)*inv_rms;  a03 = tl.load(raw_ptr+rb+32+3)*inv_rms
        a10 = tl.load(raw_ptr+rb+32+4)*inv_rms;  a11 = tl.load(raw_ptr+rb+32+5)*inv_rms
        a12 = tl.load(raw_ptr+rb+32+6)*inv_rms;  a13 = tl.load(raw_ptr+rb+32+7)*inv_rms
        a20 = tl.load(raw_ptr+rb+32+8)*inv_rms;  a21 = tl.load(raw_ptr+rb+32+9)*inv_rms
        a22 = tl.load(raw_ptr+rb+32+10)*inv_rms; a23 = tl.load(raw_ptr+rb+32+11)*inv_rms
        a30 = tl.load(raw_ptr+rb+32+12)*inv_rms; a31 = tl.load(raw_ptr+rb+32+13)*inv_rms
        a32 = tl.load(raw_ptr+rb+32+14)*inv_rms; a33 = tl.load(raw_ptr+rb+32+15)*inv_rms
        w01 = a01 - a10; w02 = a02 - a20; w03 = a03 - a30
        w12 = a12 - a21; w13 = a13 - a31; w23 = a23 - a32
        w10 = -w01; w20 = -w02; w30 = -w03
        w21 = -w12; w31 = -w13; w32 = -w23
        w00 = 0.0; w11 = 0.0; w22 = 0.0; w33 = 0.0
        half = ALPHA * 0.5

        # forward, storing (I+Y_k) snapshots. We need T_k = I + Y_k as the right factor
        # of step k+1, and W for the gradient. Save them into small arrays via static unroll.
        # Y0:
        y00 = 1.0+ALPHA*w00; y01 = ALPHA*w01; y02 = ALPHA*w02; y03 = ALPHA*w03
        y10 = ALPHA*w10; y11 = 1.0+ALPHA*w11; y12 = ALPHA*w12; y13 = ALPHA*w13
        y20 = ALPHA*w20; y21 = ALPHA*w21; y22 = 1.0+ALPHA*w22; y23 = ALPHA*w23
        y30 = ALPHA*w30; y31 = ALPHA*w31; y32 = ALPHA*w32; y33 = 1.0+ALPHA*w33
        # Forward replay storing T_k = I+Y_k (right factor of step k+1). ITERS is a constexpr
        # fixed to 3 (HyperConnectionResidual default); fully unrolled to avoid Triton's lack
        # of Python-list mutation inside @jit. The kernel asserts ITERS==3 at the autograd layer.
        # T0 (right factor of step 1) = I + Y0
        T0_00=1.0+y00; T0_01=y01; T0_02=y02; T0_03=y03
        T0_10=y10; T0_11=1.0+y11; T0_12=y12; T0_13=y13
        T0_20=y20; T0_21=y21; T0_22=1.0+y22; T0_23=y23
        T0_30=y30; T0_31=y31; T0_32=y32; T0_33=1.0+y33
        (y00,y01,y02,y03,y10,y11,y12,y13,y20,y21,y22,y23,y30,y31,y32,y33) = _cayley_fwd_step(
            w00,w01,w02,w03,w10,w11,w12,w13,w20,w21,w22,w23,w30,w31,w32,w33,
            y00,y01,y02,y03,y10,y11,y12,y13,y20,y21,y22,y23,y30,y31,y32,y33, half)
        # T1 = I + Y1
        T1_00=1.0+y00; T1_01=y01; T1_02=y02; T1_03=y03
        T1_10=y10; T1_11=1.0+y11; T1_12=y12; T1_13=y13
        T1_20=y20; T1_21=y21; T1_22=1.0+y22; T1_23=y23
        T1_30=y30; T1_31=y31; T1_32=y32; T1_33=1.0+y33
        (y00,y01,y02,y03,y10,y11,y12,y13,y20,y21,y22,y23,y30,y31,y32,y33) = _cayley_fwd_step(
            w00,w01,w02,w03,w10,w11,w12,w13,w20,w21,w22,w23,w30,w31,w32,w33,
            y00,y01,y02,y03,y10,y11,y12,y13,y20,y21,y22,y23,y30,y31,y32,y33, half)
        # T2 = I + Y2
        T2_00=1.0+y00; T2_01=y01; T2_02=y02; T2_03=y03
        T2_10=y10; T2_11=1.0+y11; T2_12=y12; T2_13=y13
        T2_20=y20; T2_21=y21; T2_22=1.0+y22; T2_23=y23
        T2_30=y30; T2_31=y31; T2_32=y32; T2_33=1.0+y33
        # (Y3 = final Hres, not needed in bwd — gY comes from upstream.)

        # ============ BACKWARD ============
        # ---- x_bar = Σ_j Hpre_cm[j] h[j,c] : g_Hpre_cm[j] = Σ_c gxbar[c]*h[j,c];
        #      grad_h_xbar[j,c] = Hpre_cm[j]*gxbar[c] ----
        # Hpre_cm[j] = 0.25 * Σ_i Hpre[i,j]
        cm0 = 0.25*(P00+P10+P20+P30); cm1 = 0.25*(P01+P11+P21+P31)
        cm2 = 0.25*(P02+P12+P22+P32); cm3 = 0.25*(P03+P13+P23+P33)
        gx = tl.load(gxbar_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)
        h0 = tl.load(h_ptr + h_base + 0*C + c, mask=cmask, other=0.0).to(tl.float32)
        h1 = tl.load(h_ptr + h_base + 1*C + c, mask=cmask, other=0.0).to(tl.float32)
        h2 = tl.load(h_ptr + h_base + 2*C + c, mask=cmask, other=0.0).to(tl.float32)
        h3 = tl.load(h_ptr + h_base + 3*C + c, mask=cmask, other=0.0).to(tl.float32)
        gcm0 = tl.sum(gx*h0, axis=0); gcm1 = tl.sum(gx*h1, axis=0)
        gcm2 = tl.sum(gx*h2, axis=0); gcm3 = tl.sum(gx*h3, axis=0)
        # grad_h xbar-path
        ghx0 = cm0*gx; ghx1 = cm1*gx; ghx2 = cm2*gx; ghx3 = cm3*gx

        # ---- Hpre_cm -> Hpre: g_Hpre[i,j] = 0.25 * g_Hpre_cm[j]  (same for all i) ----
        # softmax(dim=-1) jacobian per row i: g_z[i,j] = P[i,j]*(g[i,j] - Σ_k g[i,k]P[i,k])
        # g_Hpre[i,j] = 0.25 * gcm_j ; then /tau folded at the end.
        gpre_pre = inv_tau  # the /tau from pre/tau
        # row i: g[i,:] = (0.25*gcm0,...). dot = Σ_k g[i,k]*P[i,k]
        gg0 = 0.25*gcm0; gg1 = 0.25*gcm1; gg2 = 0.25*gcm2; gg3 = 0.25*gcm3
        gpre00,gpre01,gpre02,gpre03 = _smjac4(gg0,gg1,gg2,gg3, P00,P01,P02,P03)
        gpre10,gpre11,gpre12,gpre13 = _smjac4(gg0,gg1,gg2,gg3, P10,P11,P12,P13)
        gpre20,gpre21,gpre22,gpre23 = _smjac4(gg0,gg1,gg2,gg3, P20,P21,P22,P23)
        gpre30,gpre31,gpre32,gpre33 = _smjac4(gg0,gg1,gg2,gg3, P30,P31,P32,P33)
        # apply /tau
        gpre00*=gpre_pre; gpre01*=gpre_pre; gpre02*=gpre_pre; gpre03*=gpre_pre
        gpre10*=gpre_pre; gpre11*=gpre_pre; gpre12*=gpre_pre; gpre13*=gpre_pre
        gpre20*=gpre_pre; gpre21*=gpre_pre; gpre22*=gpre_pre; gpre23*=gpre_pre
        gpre30*=gpre_pre; gpre31*=gpre_pre; gpre32*=gpre_pre; gpre33*=gpre_pre

        # ---- Hpost_row[i] = Σ_j Hpost[i,j] -> g_Hpost[i,j] = g_Hpostrow[i] ----
        gpr0 = tl.load(ghpostrow_ptr + tok*4 + 0)
        gpr1 = tl.load(ghpostrow_ptr + tok*4 + 1)
        gpr2 = tl.load(ghpostrow_ptr + tok*4 + 2)
        gpr3 = tl.load(ghpostrow_ptr + tok*4 + 3)
        # softmax(dim=-2): each COLUMN j is a softmax across rows i. column j vector =
        # (Hpost[0,j],Hpost[1,j],Hpost[2,j],Hpost[3,j]) = (Q0j,Q1j,Q2j,Q3j).
        # incoming grad on column j rows: (gpr0,gpr1,gpr2,gpr3) (since g_Hpost[i,j]=gpr_i).
        # col 0: Q00,Q10,Q20,Q30
        gpo00,gpo10,gpo20,gpo30 = _smjac4(gpr0,gpr1,gpr2,gpr3, Q00,Q10,Q20,Q30)
        gpo01,gpo11,gpo21,gpo31 = _smjac4(gpr0,gpr1,gpr2,gpr3, Q01,Q11,Q21,Q31)
        gpo02,gpo12,gpo22,gpo32 = _smjac4(gpr0,gpr1,gpr2,gpr3, Q02,Q12,Q22,Q32)
        gpo03,gpo13,gpo23,gpo33 = _smjac4(gpr0,gpr1,gpr2,gpr3, Q03,Q13,Q23,Q33)
        # /tau
        gpo00*=inv_tau; gpo01*=inv_tau; gpo02*=inv_tau; gpo03*=inv_tau
        gpo10*=inv_tau; gpo11*=inv_tau; gpo12*=inv_tau; gpo13*=inv_tau
        gpo20*=inv_tau; gpo21*=inv_tau; gpo22*=inv_tau; gpo23*=inv_tau
        gpo30*=inv_tau; gpo31*=inv_tau; gpo32*=inv_tau; gpo33*=inv_tau

        # ---- Hres = cayley: backprop gY through ITERS steps ----
        gY00 = tl.load(ghres_ptr+hb+0); gY01 = tl.load(ghres_ptr+hb+1)
        gY02 = tl.load(ghres_ptr+hb+2); gY03 = tl.load(ghres_ptr+hb+3)
        gY10 = tl.load(ghres_ptr+hb+4); gY11 = tl.load(ghres_ptr+hb+5)
        gY12 = tl.load(ghres_ptr+hb+6); gY13 = tl.load(ghres_ptr+hb+7)
        gY20 = tl.load(ghres_ptr+hb+8); gY21 = tl.load(ghres_ptr+hb+9)
        gY22 = tl.load(ghres_ptr+hb+10); gY23 = tl.load(ghres_ptr+hb+11)
        gY30 = tl.load(ghres_ptr+hb+12); gY31 = tl.load(ghres_ptr+hb+13)
        gY32 = tl.load(ghres_ptr+hb+14); gY33 = tl.load(ghres_ptr+hb+15)
        gW00 = 0.0; gW01 = 0.0; gW02 = 0.0; gW03 = 0.0
        gW10 = 0.0; gW11 = 0.0; gW12 = 0.0; gW13 = 0.0
        gW20 = 0.0; gW21 = 0.0; gW22 = 0.0; gW23 = 0.0
        gW30 = 0.0; gW31 = 0.0; gW32 = 0.0; gW33 = 0.0
        # reverse (3 steps, last-to-first): step3 used T2, step2 used T1, step1 used T0.
        (gW00,gW01,gW02,gW03,gW10,gW11,gW12,gW13,gW20,gW21,gW22,gW23,gW30,gW31,gW32,gW33,
         gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33) = _cayley_bwd_step(
            gW00,gW01,gW02,gW03,gW10,gW11,gW12,gW13,gW20,gW21,gW22,gW23,gW30,gW31,gW32,gW33,
            gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33,
            w00,w01,w02,w03,w10,w11,w12,w13,w20,w21,w22,w23,w30,w31,w32,w33,
            T2_00,T2_01,T2_02,T2_03,T2_10,T2_11,T2_12,T2_13,T2_20,T2_21,T2_22,T2_23,T2_30,T2_31,T2_32,T2_33, half)
        (gW00,gW01,gW02,gW03,gW10,gW11,gW12,gW13,gW20,gW21,gW22,gW23,gW30,gW31,gW32,gW33,
         gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33) = _cayley_bwd_step(
            gW00,gW01,gW02,gW03,gW10,gW11,gW12,gW13,gW20,gW21,gW22,gW23,gW30,gW31,gW32,gW33,
            gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33,
            w00,w01,w02,w03,w10,w11,w12,w13,w20,w21,w22,w23,w30,w31,w32,w33,
            T1_00,T1_01,T1_02,T1_03,T1_10,T1_11,T1_12,T1_13,T1_20,T1_21,T1_22,T1_23,T1_30,T1_31,T1_32,T1_33, half)
        (gW00,gW01,gW02,gW03,gW10,gW11,gW12,gW13,gW20,gW21,gW22,gW23,gW30,gW31,gW32,gW33,
         gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33) = _cayley_bwd_step(
            gW00,gW01,gW02,gW03,gW10,gW11,gW12,gW13,gW20,gW21,gW22,gW23,gW30,gW31,gW32,gW33,
            gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33,
            w00,w01,w02,w03,w10,w11,w12,w13,w20,w21,w22,w23,w30,w31,w32,w33,
            T0_00,T0_01,T0_02,T0_03,T0_10,T0_11,T0_12,T0_13,T0_20,T0_21,T0_22,T0_23,T0_30,T0_31,T0_32,T0_33, half)
        # Y0 = I + alpha*W : gW += alpha * gY
        gW00 += ALPHA*gY00; gW01 += ALPHA*gY01; gW02 += ALPHA*gY02; gW03 += ALPHA*gY03
        gW10 += ALPHA*gY10; gW11 += ALPHA*gY11; gW12 += ALPHA*gY12; gW13 += ALPHA*gY13
        gW20 += ALPHA*gY20; gW21 += ALPHA*gY21; gW22 += ALPHA*gY22; gW23 += ALPHA*gY23
        gW30 += ALPHA*gY30; gW31 += ALPHA*gY31; gW32 += ALPHA*gY32; gW33 += ALPHA*gY33
        # W = A - A^T : g_res[i,j] = gW[i,j] - gW[j,i]  (diagonal -> 0)
        gres00 = 0.0;           gres01 = gW01 - gW10; gres02 = gW02 - gW20; gres03 = gW03 - gW30
        gres10 = gW10 - gW01;   gres11 = 0.0;         gres12 = gW12 - gW21; gres13 = gW13 - gW31
        gres20 = gW20 - gW02;   gres21 = gW21 - gW12; gres22 = 0.0;         gres23 = gW23 - gW32
        gres30 = gW30 - gW03;   gres31 = gW31 - gW13; gres32 = gW32 - gW23; gres33 = 0.0

        # ---- stack [pre|post|res] grads (these are grads wrt raw_scaled = raw_full/rms) ----
        # grad wrt raw_full[k] = grad_raw_scaled[k] / rms ; also accumulate g_rms.
        # g_rms = Σ_k g_raw_scaled[k] * (-raw_full[k]/rms^2) = -(1/rms)*Σ_k g_scaled[k]*raw_scaled[k]
        # but note pre/post grads already include the /tau; raw_scaled here means raw_full/rms
        # WITHOUT tau. The /tau was applied as part of the softmax-arg path, so the gradient
        # w.r.t. raw_scaled is exactly gpre*?? -- careful: pre_scaled_arg = raw_scaled*inv_tau,
        # and gpre_* above already had inv_tau multiplied in => they ARE d/d(raw_scaled).
        # res had NO tau; gres_* are d/d(raw_scaled) directly. Good.
        inv_rms2 = inv_rms  # grad_raw_full = grad_scaled * inv_rms
        # write grad_raw_full[48]
        grb = tok * 48
        # pre block
        gsf00=gpre00; gsf01=gpre01; gsf02=gpre02; gsf03=gpre03
        gsf10=gpre10; gsf11=gpre11; gsf12=gpre12; gsf13=gpre13
        gsf20=gpre20; gsf21=gpre21; gsf22=gpre22; gsf23=gpre23
        gsf30=gpre30; gsf31=gpre31; gsf32=gpre32; gsf33=gpre33
        # accumulate g_rms over all 48 scaled grads * raw_scaled.
        # raw_scaled (no tau) for pre = pre_scaled*tau? pre00 above = raw_full*inv_rms*inv_tau.
        # raw_scaled_pre = raw_full*inv_rms = pre00*tau. So contribution to g_rms uses raw_scaled.
        # g_rms = -(1/rms) * Σ g_scaled * raw_scaled.
        # For pre: g_scaled = gpre (d/d raw_scaled), raw_scaled = pre*TAU.
        grms = 0.0
        grms += gpre00*(pre00*TAU)+gpre01*(pre01*TAU)+gpre02*(pre02*TAU)+gpre03*(pre03*TAU)
        grms += gpre10*(pre10*TAU)+gpre11*(pre11*TAU)+gpre12*(pre12*TAU)+gpre13*(pre13*TAU)
        grms += gpre20*(pre20*TAU)+gpre21*(pre21*TAU)+gpre22*(pre22*TAU)+gpre23*(pre23*TAU)
        grms += gpre30*(pre30*TAU)+gpre31*(pre31*TAU)+gpre32*(pre32*TAU)+gpre33*(pre33*TAU)
        grms += gpo00*(po00*TAU)+gpo01*(po01*TAU)+gpo02*(po02*TAU)+gpo03*(po03*TAU)
        grms += gpo10*(po10*TAU)+gpo11*(po11*TAU)+gpo12*(po12*TAU)+gpo13*(po13*TAU)
        grms += gpo20*(po20*TAU)+gpo21*(po21*TAU)+gpo22*(po22*TAU)+gpo23*(po23*TAU)
        grms += gpo30*(po30*TAU)+gpo31*(po31*TAU)+gpo32*(po32*TAU)+gpo33*(po33*TAU)
        grms += gres00*a00+gres01*a01+gres02*a02+gres03*a03
        grms += gres10*a10+gres11*a11+gres12*a12+gres13*a13
        grms += gres20*a20+gres21*a21+gres22*a22+gres23*a23
        grms += gres30*a30+gres31*a31+gres32*a32+gres33*a33
        grms = -inv_rms * grms   # d L / d rms

        # store grad_raw_full = g_scaled * inv_rms
        tl.store(graw_ptr+grb+0,  gpre00*inv_rms2); tl.store(graw_ptr+grb+1,  gpre01*inv_rms2)
        tl.store(graw_ptr+grb+2,  gpre02*inv_rms2); tl.store(graw_ptr+grb+3,  gpre03*inv_rms2)
        tl.store(graw_ptr+grb+4,  gpre10*inv_rms2); tl.store(graw_ptr+grb+5,  gpre11*inv_rms2)
        tl.store(graw_ptr+grb+6,  gpre12*inv_rms2); tl.store(graw_ptr+grb+7,  gpre13*inv_rms2)
        tl.store(graw_ptr+grb+8,  gpre20*inv_rms2); tl.store(graw_ptr+grb+9,  gpre21*inv_rms2)
        tl.store(graw_ptr+grb+10, gpre22*inv_rms2); tl.store(graw_ptr+grb+11, gpre23*inv_rms2)
        tl.store(graw_ptr+grb+12, gpre30*inv_rms2); tl.store(graw_ptr+grb+13, gpre31*inv_rms2)
        tl.store(graw_ptr+grb+14, gpre32*inv_rms2); tl.store(graw_ptr+grb+15, gpre33*inv_rms2)
        tl.store(graw_ptr+grb+16, gpo00*inv_rms2);  tl.store(graw_ptr+grb+17, gpo01*inv_rms2)
        tl.store(graw_ptr+grb+18, gpo02*inv_rms2);  tl.store(graw_ptr+grb+19, gpo03*inv_rms2)
        tl.store(graw_ptr+grb+20, gpo10*inv_rms2);  tl.store(graw_ptr+grb+21, gpo11*inv_rms2)
        tl.store(graw_ptr+grb+22, gpo12*inv_rms2);  tl.store(graw_ptr+grb+23, gpo13*inv_rms2)
        tl.store(graw_ptr+grb+24, gpo20*inv_rms2);  tl.store(graw_ptr+grb+25, gpo21*inv_rms2)
        tl.store(graw_ptr+grb+26, gpo22*inv_rms2);  tl.store(graw_ptr+grb+27, gpo23*inv_rms2)
        tl.store(graw_ptr+grb+28, gpo30*inv_rms2);  tl.store(graw_ptr+grb+29, gpo31*inv_rms2)
        tl.store(graw_ptr+grb+30, gpo32*inv_rms2);  tl.store(graw_ptr+grb+31, gpo33*inv_rms2)
        tl.store(graw_ptr+grb+32, gres00*inv_rms2); tl.store(graw_ptr+grb+33, gres01*inv_rms2)
        tl.store(graw_ptr+grb+34, gres02*inv_rms2); tl.store(graw_ptr+grb+35, gres03*inv_rms2)
        tl.store(graw_ptr+grb+36, gres10*inv_rms2); tl.store(graw_ptr+grb+37, gres11*inv_rms2)
        tl.store(graw_ptr+grb+38, gres12*inv_rms2); tl.store(graw_ptr+grb+39, gres13*inv_rms2)
        tl.store(graw_ptr+grb+40, gres20*inv_rms2); tl.store(graw_ptr+grb+41, gres21*inv_rms2)
        tl.store(graw_ptr+grb+42, gres22*inv_rms2); tl.store(graw_ptr+grb+43, gres23*inv_rms2)
        tl.store(graw_ptr+grb+44, gres30*inv_rms2); tl.store(graw_ptr+grb+45, gres31*inv_rms2)
        tl.store(graw_ptr+grb+46, gres32*inv_rms2); tl.store(graw_ptr+grb+47, gres33*inv_rms2)

        # ---- rms-path grad on h: rms = sqrt(mean(h^2)+eps);
        #      d rms/d h[j,c] = h[j,c]/(N*C*rms). grad_h_rms = grms * h/(N*C*rms) ----
        scale = grms / (N * C * rms)
        ghr0 = scale * h0; ghr1 = scale * h1; ghr2 = scale * h2; ghr3 = scale * h3
        # total grad_h_partial = xbar-path + rms-path
        tl.store(ghpart_ptr + h_base + 0*C + c, (ghx0+ghr0).to(ghpart_ptr.dtype.element_ty), mask=cmask)
        tl.store(ghpart_ptr + h_base + 1*C + c, (ghx1+ghr1).to(ghpart_ptr.dtype.element_ty), mask=cmask)
        tl.store(ghpart_ptr + h_base + 2*C + c, (ghx2+ghr2).to(ghpart_ptr.dtype.element_ty), mask=cmask)
        tl.store(ghpart_ptr + h_base + 3*C + c, (ghx3+ghr3).to(ghpart_ptr.dtype.element_ty), mask=cmask)

    # -----------------------------------------------------------------------
    # POST forward:
    #   out[b,s,i,c] = sum_j Hres[b,s,i,j]*h[b,s,j,c] + Hpost_row[b,s,i]*y[b,s,c]
    # One program = one (b,s) token. Load all n streams of h, all n*n Hres,
    # n Hpost_row, and y once; emit n output streams.
    # -----------------------------------------------------------------------
    @triton.jit
    def _hc_post_fwd_kernel(
        hres_ptr,         # [B, S, n, n]
        hpost_ptr,        # [B, S, n]   (Hpost_row)
        h_ptr,            # [B, S, n, C]
        y_ptr,            # [B, S, C]
        out_ptr,          # [B, S, n, C]
        N: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        h_base = tok * (N * C)
        yv = tl.load(y_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)

        # preload all n stream rows of h into registers (n=4, C-block each)
        # use a python list comprehension over static N via static_range accumulation.
        for i in tl.static_range(N):
            acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for j in tl.static_range(N):
                hij = tl.load(hres_ptr + tok * (N * N) + i * N + j).to(tl.float32)
                hj = tl.load(h_ptr + h_base + j * C + c, mask=cmask, other=0.0).to(tl.float32)
                acc += hij * hj
            post_i = tl.load(hpost_ptr + tok * N + i).to(tl.float32)
            acc += post_i * yv
            tl.store(out_ptr + h_base + i * C + c,
                     acc.to(out_ptr.dtype.element_ty), mask=cmask)

    # -----------------------------------------------------------------------
    # POST backward, given grad_out[b,s,i,c]:
    #   grad_h[b,s,j,c]       = sum_i Hres[b,s,i,j] * grad_out[b,s,i,c]
    #   grad_y[b,s,c]         = sum_i Hpost_row[b,s,i] * grad_out[b,s,i,c]
    #   grad_Hres[b,s,i,j]    = sum_c grad_out[b,s,i,c] * h[b,s,j,c]
    #   grad_Hpost_row[b,s,i] = sum_c grad_out[b,s,i,c] * y[b,s,c]
    # -----------------------------------------------------------------------
    @triton.jit
    def _hc_post_bwd_kernel(
        gout_ptr,         # [B, S, n, C]
        hres_ptr,         # [B, S, n, n]
        hpost_ptr,        # [B, S, n]
        h_ptr,            # [B, S, n, C]
        y_ptr,            # [B, S, C]
        gh_ptr,           # [B, S, n, C]  out (grad wrt h, this path)
        gy_ptr,           # [B, S, C]     out (grad wrt y)
        ghres_ptr,        # [B, S, n, n]  out
        ghpost_ptr,       # [B, S, n]     out
        N: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        h_base = tok * (N * C)
        yv = tl.load(y_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)

        gy = tl.zeros((BLOCK_C,), dtype=tl.float32)
        # grad_h for stream j accumulates over output streams i.
        for j in tl.static_range(N):
            ghj = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for i in tl.static_range(N):
                go = tl.load(gout_ptr + h_base + i * C + c, mask=cmask, other=0.0).to(tl.float32)
                hij = tl.load(hres_ptr + tok * (N * N) + i * N + j).to(tl.float32)
                ghj += hij * go
            tl.store(gh_ptr + h_base + j * C + c, ghj.to(gh_ptr.dtype.element_ty), mask=cmask)

        # grad_y, grad_Hres, grad_Hpost_row — loop over output streams i.
        for i in tl.static_range(N):
            go = tl.load(gout_ptr + h_base + i * C + c, mask=cmask, other=0.0).to(tl.float32)
            post_i = tl.load(hpost_ptr + tok * N + i).to(tl.float32)
            gy += post_i * go
            # grad_Hpost_row[i] = sum_c go * y
            ghpost_i = tl.sum(go * yv, axis=0)
            tl.store(ghpost_ptr + tok * N + i, ghpost_i)
            # grad_Hres[i,j] = sum_c go * h_j
            for j in tl.static_range(N):
                hj = tl.load(h_ptr + h_base + j * C + c, mask=cmask, other=0.0).to(tl.float32)
                ghres_ij = tl.sum(go * hj, axis=0)
                tl.store(ghres_ptr + tok * (N * N) + i * N + j, ghres_ij)

        tl.store(gy_ptr + tok * C + c, gy.to(gy_ptr.dtype.element_ty), mask=cmask)


# ===========================================================================
# autograd.Function — PRE (x_bar contraction)
# ===========================================================================

class _FusedHCPre(torch.autograd.Function):
    """x_bar[b,s,c] = sum_j Hpre_cm[b,s,j] * h[b,s,j,c]."""

    @staticmethod
    def forward(ctx, h: Tensor, hpre_cm: Tensor):
        B, S, N, C = h.shape
        h = h.contiguous()
        hpre_cm = hpre_cm.contiguous()
        xbar = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        BLOCK_C = _next_pow2(C)
        _hc_pre_fwd_kernel[(B * S,)](
            h, hpre_cm, xbar, N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        ctx.save_for_backward(h, hpre_cm)
        ctx.shape = (B, S, N, C)
        return xbar

    @staticmethod
    def backward(ctx, grad_xbar: Tensor):
        h, hpre_cm = ctx.saved_tensors
        B, S, N, C = ctx.shape
        grad_xbar = grad_xbar.contiguous()
        grad_h = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        grad_hpre = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        BLOCK_C = _next_pow2(C)
        _hc_pre_bwd_kernel[(B * S,)](
            grad_xbar, h, hpre_cm, grad_h, grad_hpre,
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        return grad_h, grad_hpre.to(hpre_cm.dtype)


# ===========================================================================
# autograd.Function — PRE-MAPPING (round 2): rms+proj+softmax+cayley+x_bar fused.
#   Forward:  raw_full = addmm(proj_b, x_flat, proj_w.T)   [cuBLAS, tensor-core]
#             kernel:  (h, raw_full) -> x_bar, Hres, Hpost_row  (+ saved Hpre_cm, rms)
#   Backward: kernel:  grads on (x_bar, Hres, Hpost_row) -> grad_raw_full, grad_h_partial
#             grad_w = grad_raw_full^T @ x_flat ; grad_b = grad_raw_full.sum   [cuBLAS]
#             grad_h = grad_h_partial + (grad_raw_full @ proj_w).reshape(B,S,n,C) [cuBLAS]
# ===========================================================================

class _FusedHCPreMap(torch.autograd.Function):
    """Fused n×n mapping + x_bar. Returns (x_bar, Hres, Hpost_row)."""

    @staticmethod
    def forward(ctx, h, proj_w, proj_b, tau, alpha, iters, eps):
        B, S, N, C = h.shape
        assert N == 4, "fused premap kernel assumes n=4 (mapping unroll is 4×4)"
        assert int(iters) == 3, "fused premap backward unrolls cayley to exactly 3 iters"
        h = h.contiguous()
        x_flat = h.reshape(B * S, N * C)
        # raw_full = x_flat @ proj_w.T + proj_b  (pre-rms). Compute the GEMV in the carrier
        # dtype (bf16 -> tensor-core, fp32 accumulate inside cuBLAS) to AVOID materialising a
        # 200 MB fp32 copy of x_flat and the slow SIMT sgemm. The precision-sensitive part of
        # the mapping (softmax / Cayley) runs fp32 in-kernel from raw_full, so a bf16-input
        # GEMV with fp32 accumulate is faithful (verified: grad cosines == 1.0 on proj/h).
        dt = h.dtype
        raw_full = torch.addmm(
            proj_b.to(dt), x_flat, proj_w.to(dt).t()
        ).float().reshape(B, S, 48).contiguous()

        xbar = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
        hpostrow = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        hprecm = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        rms = torch.empty(B, S, 1, device=h.device, dtype=torch.float32)
        BLOCK_C = _next_pow2(C)
        _hc_premap_fwd_kernel[(B * S,)](
            h, raw_full, xbar, hres, hpostrow, hprecm, rms,
            TAU=float(tau), ALPHA=float(alpha), ITERS=int(iters), EPS=float(eps),
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        ctx.save_for_backward(h, raw_full, rms, proj_w)
        ctx.shape = (B, S, N, C)
        ctx.cfg = (float(tau), float(alpha), int(iters))
        return xbar, hres, hpostrow

    @staticmethod
    def backward(ctx, grad_xbar, grad_hres, grad_hpostrow):
        h, raw_full, rms, proj_w = ctx.saved_tensors
        B, S, N, C = ctx.shape
        tau, alpha, iters = ctx.cfg
        grad_xbar = grad_xbar.contiguous()
        grad_hres = grad_hres.contiguous().float()
        grad_hpostrow = grad_hpostrow.contiguous().float()

        graw = torch.empty(B, S, 48, device=h.device, dtype=torch.float32)
        gh_partial = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        BLOCK_C = _next_pow2(C)
        _hc_premap_bwd_kernel[(B * S,)](
            grad_xbar, grad_hres, grad_hpostrow, h, raw_full, rms,
            graw, gh_partial,
            TAU=tau, ALPHA=alpha, ITERS=iters,
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        # proj VJP via cuBLAS. raw_full = x_flat @ proj_w.T + proj_b. To avoid the 200 MB
        # fp32 copy of x_flat and the slow SIMT sgemm, run the two GEMMs with bf16 inputs
        # (cuBLAS accumulates in fp32). grad_b stays an exact fp32 reduction over the small
        # [B·S,48] graw. grad_h_proj is added to the kernel's xbar+rms grad on h.
        dt = h.dtype
        graw2 = graw.reshape(B * S, 48)
        graw2_dt = graw2.to(dt)
        x_flat = h.reshape(B * S, N * C)
        grad_w = (graw2_dt.t() @ x_flat).float()                  # [48, N*C]  (fp32 accum)
        grad_b = graw2.sum(0)                                     # [48]       (exact fp32)
        grad_h_proj = (graw2_dt @ proj_w.to(dt)).reshape(B, S, N, C)  # [B,S,n,C]
        grad_h = (gh_partial.to(dt) + grad_h_proj)
        return (grad_h, grad_w.to(proj_w.dtype), grad_b.to(proj_w.dtype),
                None, None, None, None)


# ===========================================================================
# autograd.Function — POST (x_mix + x_post + add)
# ===========================================================================

class _FusedHCPost(torch.autograd.Function):
    """out[b,s,i,c] = sum_j Hres[i,j]*h[j,c] + Hpost_row[i]*y[c]."""

    @staticmethod
    def forward(ctx, hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor):
        B, S, N, C = h.shape
        hres = hres.contiguous()
        hpost_row = hpost_row.contiguous()
        h = h.contiguous()
        y = y.contiguous()
        out = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        BLOCK_C = _next_pow2(C)
        _hc_post_fwd_kernel[(B * S,)](
            hres, hpost_row, h, y, out, N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        ctx.save_for_backward(hres, hpost_row, h, y)
        ctx.shape = (B, S, N, C)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        hres, hpost_row, h, y = ctx.saved_tensors
        B, S, N, C = ctx.shape
        grad_out = grad_out.contiguous()
        grad_h = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        grad_y = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        grad_hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
        grad_hpost = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        BLOCK_C = _next_pow2(C)
        _hc_post_bwd_kernel[(B * S,)](
            grad_out, hres, hpost_row, h, y,
            grad_h, grad_y, grad_hres, grad_hpost,
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        return (grad_hres.to(hres.dtype), grad_hpost.to(hpost_row.dtype),
                grad_h, grad_y.to(y.dtype))


# ===========================================================================
# Public API
# ===========================================================================

def hc_pre(h: Tensor, hpre_cm: Tensor) -> Tensor:
    """Fused stream-aggregation: x_bar = einsum('bsj,bsjc->bsc', Hpre_cm, h).

    Args:
        h:       [B, S, n, C] n-stream carrier.
        hpre_cm: [B, S, n]    column-mean of Hpre (mean over output stream i).

    Returns:
        x_bar: [B, S, C] sublayer input.
    """
    from morph.kernels.triton._eager_flag import force_eager, hc_force_eager
    if force_eager() or hc_force_eager() or not TRITON_AVAILABLE or not h.is_cuda:
        return hc_pre_reference(h, hpre_cm)
    return _FusedHCPre.apply(h, hpre_cm)


def hc_post(hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor) -> Tensor:
    """Fused stream-mix + scatter + add: x_mix + x_post.

    Args:
        hres:      [B, S, n, n] manifold stream mixer.
        hpost_row: [B, S, n]    row-sum of Hpost.
        h:         [B, S, n, C] n-stream carrier.
        y:         [B, S, C]    sublayer output.

    Returns:
        out: [B, S, n, C] updated carrier.
    """
    from morph.kernels.triton._eager_flag import force_eager, hc_force_eager
    if force_eager() or hc_force_eager() or not TRITON_AVAILABLE or not h.is_cuda:
        return hc_post_reference(hres, hpost_row, h, y)
    return _FusedHCPost.apply(hres, hpost_row, h, y)


def hc_pre_map(
    h: Tensor, proj_w: Tensor, proj_b: Tensor,
    tau: float, alpha: float, iters: int, eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused full PRE mapping: rms + proj + softmax×2 + cayley + reductions + x_bar.

    Computes, per (b,s) token, from the n-stream carrier and the projection weights:
      rms       = sqrt(mean(vec(h)^2) + eps)
      raw48     = (proj_w @ vec(h) + proj_b) / rms  -> [pre|post|res] each [n,n]
      Hpre      = softmax(pre/tau, -1) ; Hpost = softmax(post/tau, -2) ; Hres = cayley(res)
      Hpre_cm   = colmean(Hpre) ; Hpost_row = rowsum(Hpost)
      x_bar[c]  = Σ_j Hpre_cm[j] h[j,c]

    Args:
        h:      [B,S,n,C] carrier (n must be 4).
        proj_w: [3*n*n, n*C] projection weight.
        proj_b: [3*n*n]      projection bias.
        tau, alpha, iters, eps: softmax temperature, cayley step, cayley iters, rms eps.

    Returns:
        (x_bar[B,S,C], Hres[B,S,n,n], Hpost_row[B,S,n]).
    """
    from morph.kernels.triton._eager_flag import force_eager, hc_force_eager
    if (force_eager() or hc_force_eager() or not TRITON_AVAILABLE or not h.is_cuda
            or h.shape[2] != 4 or int(iters) != 3):
        return hc_pre_map_reference(h, proj_w, proj_b, tau, alpha, iters, eps)
    return _FusedHCPreMap.apply(h, proj_w, proj_b, tau, alpha, iters, eps)


# ===========================================================================
# Pure-PyTorch references (the spec)
# ===========================================================================

def _cayley_ref(A: Tensor, iters: int, alpha: float) -> Tensor:
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    W = A - A.transpose(-1, -2)
    Y = I + alpha * W
    half = alpha * 0.5
    for _ in range(iters):
        Y = I + half * (W @ (I + Y))
    return Y


def hc_pre_map_reference(
    h: Tensor, proj_w: Tensor, proj_b: Tensor,
    tau: float, alpha: float, iters: int, eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch spec for the fused full PRE mapping (matches HyperConnectionResidual)."""
    B, S, n, C = h.shape
    x_flat = h.reshape(B, S, n * C)
    rms = x_flat.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    raw = (F.linear(x_flat.float(), proj_w.float(), proj_b.float()) / rms).reshape(B, S, 3, n, n)
    pre_raw, post_raw, res_raw = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
    Hpre = torch.softmax(pre_raw / tau, dim=-1)
    Hpost = torch.softmax(post_raw / tau, dim=-2)
    Hres = _cayley_ref(res_raw, iters, alpha)
    Hpre_cm = Hpre.mean(dim=-2)
    Hpost_row = Hpost.sum(dim=-1)
    x_bar = torch.einsum("bsj,bsjc->bsc", Hpre_cm.to(h.dtype), h)
    return x_bar, Hres, Hpost_row.to(h.dtype)


def hc_pre_reference(h: Tensor, hpre_cm: Tensor) -> Tensor:
    return torch.einsum("bsj,bsjc->bsc", hpre_cm.to(h.dtype), h)


def hc_post_reference(hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor) -> Tensor:
    x_mix = torch.einsum("bsij,bsjc->bsic", hres.to(h.dtype), h)
    x_post = hpost_row.to(h.dtype).unsqueeze(-1) * y.unsqueeze(2)
    return x_mix + x_post


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    dev = torch.device("cuda")
    dt = torch.bfloat16
    print("=" * 100)
    print("fused_hyper_connection — PRE + POST forward + backward correctness")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    n, C = 4, 768

    def make(B, S):
        h = torch.randn(B, S, n, C, device=dev, dtype=dt)
        # Hpre_cm: row-stochastic Hpre, column-mean -> roughly in [0, 1/n .. ]; use realistic
        pre = torch.softmax(torch.randn(B, S, n, n, device=dev) / 1.0, dim=-1)
        hpre_cm = pre.mean(dim=-2).to(dt)                      # [B,S,n]
        post = torch.softmax(torch.randn(B, S, n, n, device=dev) / 1.0, dim=-2)
        hpost_row = post.sum(dim=-1).to(dt)                    # [B,S,n]
        # Hres ~ orthogonal-ish (cayley of small skew), entries bounded
        A = torch.randn(B, S, n, n, device=dev) * 0.3
        W = A - A.transpose(-1, -2)
        I = torch.eye(n, device=dev)
        Y = I + 0.1 * W
        for _ in range(3):
            Y = I + 0.05 * (W @ (I + Y))
        hres = Y.to(dt)                                        # [B,S,n,n]
        y = torch.randn(B, S, C, device=dev, dtype=dt)
        return h, hpre_cm, hpost_row, hres, y

    def stats(f, r):
        e = (f.float() - r.float()).abs()
        return e.max().item(), e.mean().item(), F.cosine_similarity(
            f.reshape(-1).float(), r.reshape(-1).float(), dim=0).item()

    def gcos(a, b):
        return F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0).item()

    all_ok = True

    # ---------------------------------------------------------------------
    # PRE-MAP (round 2): full mapping (rms+proj+softmax×2+cayley+reductions+x_bar)
    # vs hc_pre_map_reference. Gate: fwd cosines + grad cosines (h/proj_w/proj_b)
    # + (bf16) closeness-to-fp32-truth at-least-as-good-as the reference.
    # ---------------------------------------------------------------------
    print("\n[PRE-MAP — rms+proj+softmax×2+cayley+reductions+x_bar (round 2)]")
    tau, alpha, iters, eps = 1.0, 0.1, 3, 1e-6
    for B in (2, 4):
        for S in (512, 2048, 4096):
            h = torch.randn(B, S, n, C, device=dev, dtype=dt) * 0.7
            pw = (torch.randn(48, n * C, device=dev) * 0.02).to(dt)
            pb = (torch.randn(48, device=dev) * 0.05).to(dt)
            hf = h.clone().requires_grad_(True); wf = pw.clone().requires_grad_(True); bf = pb.clone().requires_grad_(True)
            hr = h.clone().requires_grad_(True); wr = pw.clone().requires_grad_(True); br = pb.clone().requires_grad_(True)

            xf, resf, prf = _FusedHCPreMap.apply(hf, wf, bf, tau, alpha, iters, eps)
            xr, resr, prr = hc_pre_map_reference(hr, wr, br, tau, alpha, iters, eps)
            x32, res32, pr32 = hc_pre_map_reference(h.float(), pw.float(), pb.float(), tau, alpha, iters, eps)

            xmx, xmn, xcs = stats(xf, xr)
            rcs = gcos(resf, resr); pcs = gcos(prf, prr)
            kvt = (xf.float() - x32.float()).abs().max().item()
            rvt = (xr.float() - x32.float()).abs().max().item()
            gx = torch.randn_like(xf); gr = torch.randn_like(resf); gp = torch.randn_like(prf)
            (xf.float()*gx.float()).sum().add_((resf*gr).sum()).add_((prf.float()*gp.float()).sum()).backward()
            (xr.float()*gx.float()).sum().add_((resr*gr).sum()).add_((prr.float()*gp.float()).sum()).backward()
            gh_c = gcos(hf.grad, hr.grad); gw_c = gcos(wf.grad, wr.grad); gb_c = gcos(bf.grad, br.grad)
            ok = (xcs > 0.9999) and (rcs > 0.999) and (pcs > 0.999) \
                and (kvt <= rvt + 5e-3) and min(gh_c, gw_c, gb_c) > 0.995
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] B={B} S={S:<5} "
                  f"x_max={xmx:.2e} x_cos={xcs:.6f} res_cos={rcs:.6f} pr_cos={pcs:.6f} "
                  f"vs-truth(k={kvt:.2e}<=r={rvt:.2e}) | gh={gh_c:.4f} gw={gw_c:.4f} gb={gb_c:.4f}")

    print("\n[PRE — x_bar contraction]")
    for B in (2, 4):
        for S in (512, 2048, 4096):
            h, hpre_cm, *_ = make(B, S)
            hf = h.detach().clone().requires_grad_(True)
            pf = hpre_cm.detach().clone().requires_grad_(True)
            hr = h.detach().clone().requires_grad_(True)
            pr = hpre_cm.detach().clone().requires_grad_(True)

            xf = _FusedHCPre.apply(hf, pf)
            xr = hc_pre_reference(hr, pr)
            mx, mn, cs = stats(xf, xr)
            go = torch.randn_like(xf)
            (xf.float() * go.float()).sum().backward()
            (xr.float() * go.float()).sum().backward()
            gh_c = gcos(hf.grad, hr.grad)
            gp_c = gcos(pf.grad, pr.grad)
            ok = (mx < 2e-2) and (gh_c > 0.995) and (gp_c > 0.995)
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] B={B} S={S:<5} "
                  f"fwd_max={mx:.2e} fwd_cos={cs:.6f} | gh={gh_c:.4f} gHpre={gp_c:.4f}")

    print("\n[POST — x_mix + x_post + add]")
    # NOTE on the max-err gate: the eager reference rounds x_mix and x_post to bf16
    # SEPARATELY before adding; this kernel accumulates the whole sum once in fp32.
    # So the kernel differs from the bf16 reference by up to ~4 bf16 ULP (6.25e-2 at
    # |x|~4) — that gap is BELOW the bf16 representation floor of this op and is
    # unachievable by ANY bf16 impl vs the bf16 reference (same situation documented
    # in fused_cca_prologue). We instead gate on the contracts that ARE meetable and
    # prove correctness: MEAN abs err, forward cosine, all grad cosines, AND that the
    # kernel is at least as close to fp32-truth as the bf16 reference is.
    for B in (2, 4):
        for S in (512, 2048, 4096):
            h, hpre_cm, hpost_row, hres, y = make(B, S)
            hf = h.detach().clone().requires_grad_(True)
            postf = hpost_row.detach().clone().requires_grad_(True)
            hresf = hres.detach().clone().requires_grad_(True)
            yf = y.detach().clone().requires_grad_(True)
            hr = h.detach().clone().requires_grad_(True)
            postr = hpost_row.detach().clone().requires_grad_(True)
            hresr = hres.detach().clone().requires_grad_(True)
            yr = y.detach().clone().requires_grad_(True)

            of = _FusedHCPost.apply(hresf, postf, hf, yf)
            orr = hc_post_reference(hresr, postr, hr, yr)
            o32 = hc_post_reference(hres.float(), hpost_row.float(), h.float(), y.float())
            mx, mn, cs = stats(of, orr)
            kvt = (of.float() - o32.float()).abs().max().item()       # kernel vs truth
            rvt = (orr.float() - o32.float()).abs().max().item()      # bf16 ref vs truth
            go = torch.randn_like(of)
            (of.float() * go.float()).sum().backward()
            (orr.float() * go.float()).sum().backward()
            gh_c = gcos(hf.grad, hr.grad)
            gy_c = gcos(yf.grad, yr.grad)
            gres_c = gcos(hresf.grad, hresr.grad)
            gpost_c = gcos(postf.grad, postr.grad)
            ok = (mn < 3e-3) and (cs > 0.9999) and (kvt <= rvt + 1e-6) \
                and min(gh_c, gy_c, gres_c, gpost_c) > 0.995
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] B={B} S={S:<5} "
                  f"fwd_max={mx:.2e} fwd_mean={mn:.2e} fwd_cos={cs:.6f} "
                  f"vs-truth(ker={kvt:.2e}<=ref={rvt:.2e}) | gh={gh_c:.4f} gy={gy_c:.4f} "
                  f"gHres={gres_c:.4f} gHpost={gpost_c:.4f}")

    # finite check
    h, hpre_cm, hpost_row, hres, y = make(4, 2048)
    xb = _FusedHCPre.apply(h.requires_grad_(True), hpre_cm.requires_grad_(True))
    out = _FusedHCPost.apply(hres.requires_grad_(True), hpost_row.requires_grad_(True),
                             h, y.requires_grad_(True))
    finite = torch.isfinite(xb).all().item() and torch.isfinite(out).all().item()
    print(f"\n[finite] fwd all-finite: {finite}")
    all_ok &= finite

    # ----- speed: fused PRE+POST vs eager references, B4/S4096 -----
    print("\n[Speed — fused vs eager (PRE+POST fwd+bwd), B=4 S=4096]")
    B, S = 4, 4096
    h0, hpre_cm0, hpost_row0, hres0, y0 = make(B, S)

    def fused_fb():
        h = h0.detach().clone().requires_grad_(True)
        pre = hpre_cm0.detach().clone().requires_grad_(True)
        post = hpost_row0.detach().clone().requires_grad_(True)
        res = hres0.detach().clone().requires_grad_(True)
        y = y0.detach().clone().requires_grad_(True)
        xb = _FusedHCPre.apply(h, pre)
        out = _FusedHCPost.apply(res, post, h, y)
        (xb.sum() + out.sum()).backward()

    def eager_fb():
        h = h0.detach().clone().requires_grad_(True)
        pre = hpre_cm0.detach().clone().requires_grad_(True)
        post = hpost_row0.detach().clone().requires_grad_(True)
        res = hres0.detach().clone().requires_grad_(True)
        y = y0.detach().clone().requires_grad_(True)
        xb = hc_pre_reference(h, pre)
        out = hc_post_reference(res, post, h, y)
        (xb.sum() + out.sum()).backward()

    def bench(fn, nrep=50, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(nrep):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / nrep * 1e3

    t_f = bench(fused_fb)
    t_e = bench(eager_fb)
    print(f"  Eager refs fwd+bwd:  {t_e:.3f} ms")
    print(f"  Fused      fwd+bwd:  {t_f:.3f} ms")
    print(f"  Speedup:             {t_e / t_f:.2f}x")

    print("\n" + "=" * 100)
    print("ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 100)
    assert all_ok, "fused_hyper_connection self-test FAILED"
