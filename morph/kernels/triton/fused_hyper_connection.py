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

    # NOTE: the old inverse-free fixed-point Cayley iteration helpers
    # (_cayley_fwd_step / _cayley_bwd_step) were REMOVED — they diverged for
    # ‖A−Aᵀ‖ ≥ 2/α = 20 and amplified the carrier 10–3000×. Replaced by the exact
    # closed-form Cayley + analytic VJP below (unconditionally orthogonal, VJP ‖·‖₂ ≤ 1).

    # -----------------------------------------------------------------------
    # EXACT closed-form Cayley for n=4 (replaces the divergent fixed-point iter).
    #   B = ½α·(A − Aᵀ)  (skew);  p = ½‖B‖²_F ;  q = Pf(B)²
    #   Y = (I + 2B + B²)·[(1+p)I + B²] / (1 + p + q)          (orthogonal, denom ≥ 1)
    # VJP:  M = [(1+p)I + B²]·(I + B) / (1+p+q) = (I−B)⁻¹
    #       gB = (I+Y)ᵀ·gY·Mᵀ ;  gW = ½α·gB ;  gA = gW − gWᵀ
    # Mirrors morph/model/hyper_connections.cayley_orthogonal + cayley_vjp_ref.py exactly.
    # -----------------------------------------------------------------------
    @triton.jit
    def _mm4(
        a00, a01, a02, a03, a10, a11, a12, a13,
        a20, a21, a22, a23, a30, a31, a32, a33,
        b00, b01, b02, b03, b10, b11, b12, b13,
        b20, b21, b22, b23, b30, b31, b32, b33,
    ):
        """4×4 · 4×4 matmul, 16 scalars in ×2 → 16 scalars out (row-major)."""
        c00 = a00*b00 + a01*b10 + a02*b20 + a03*b30
        c01 = a00*b01 + a01*b11 + a02*b21 + a03*b31
        c02 = a00*b02 + a01*b12 + a02*b22 + a03*b32
        c03 = a00*b03 + a01*b13 + a02*b23 + a03*b33
        c10 = a10*b00 + a11*b10 + a12*b20 + a13*b30
        c11 = a10*b01 + a11*b11 + a12*b21 + a13*b31
        c12 = a10*b02 + a11*b12 + a12*b22 + a13*b32
        c13 = a10*b03 + a11*b13 + a12*b23 + a13*b33
        c20 = a20*b00 + a21*b10 + a22*b20 + a23*b30
        c21 = a20*b01 + a21*b11 + a22*b21 + a23*b31
        c22 = a20*b02 + a21*b12 + a22*b22 + a23*b32
        c23 = a20*b03 + a21*b13 + a22*b23 + a23*b33
        c30 = a30*b00 + a31*b10 + a32*b20 + a33*b30
        c31 = a30*b01 + a31*b11 + a32*b21 + a33*b31
        c32 = a30*b02 + a31*b12 + a32*b22 + a33*b32
        c33 = a30*b03 + a31*b13 + a32*b23 + a33*b33
        return (c00, c01, c02, c03, c10, c11, c12, c13,
                c20, c21, c22, c23, c30, c31, c32, c33)

    @triton.jit
    def _skewB4(a00, a01, a02, a03, a10, a11, a12, a13,
                a20, a21, a22, a23, a30, a31, a32, a33, half):
        """B = half·(A − Aᵀ), the skew so(4) generator (diagonal = 0)."""
        B01 = half*(a01 - a10); B02 = half*(a02 - a20); B03 = half*(a03 - a30)
        B12 = half*(a12 - a21); B13 = half*(a13 - a31); B23 = half*(a23 - a32)
        return (0.0,  B01,  B02,  B03,
                -B01, 0.0,  B12,  B13,
                -B02, -B12, 0.0,  B23,
                -B03, -B13, -B23, 0.0)

    @triton.jit
    def _cayley4_pq(B00, B01, B02, B03, B10, B11, B12, B13,
                    B20, B21, B22, B23, B30, B31, B32, B33):
        """p = ½‖B‖²_F ; Pf = Pfaffian ; q = Pf². (B is 4×4 skew.)"""
        p = B01*B01 + B02*B02 + B03*B03 + B12*B12 + B13*B13 + B23*B23
        Pf = B01*B23 - B02*B13 + B03*B12
        return p, Pf*Pf

    @triton.jit
    def _cayley4_fwd(
        a00, a01, a02, a03, a10, a11, a12, a13,
        a20, a21, a22, a23, a30, a31, a32, a33, ALPHA,
    ):
        """Exact closed-form orthogonal Cayley Y for n=4. Returns Y (16 scalars)."""
        half = ALPHA * 0.5
        (B00, B01, B02, B03, B10, B11, B12, B13,
         B20, B21, B22, B23, B30, B31, B32, B33) = _skewB4(
            a00, a01, a02, a03, a10, a11, a12, a13,
            a20, a21, a22, a23, a30, a31, a32, a33, half)
        (S00, S01, S02, S03, S10, S11, S12, S13,
         S20, S21, S22, S23, S30, S31, S32, S33) = _mm4(
            B00, B01, B02, B03, B10, B11, B12, B13,
            B20, B21, B22, B23, B30, B31, B32, B33,
            B00, B01, B02, B03, B10, B11, B12, B13,
            B20, B21, B22, B23, B30, B31, B32, B33)      # B² = B@B
        p, q = _cayley4_pq(B00, B01, B02, B03, B10, B11, B12, B13,
                           B20, B21, B22, B23, B30, B31, B32, B33)
        inv = 1.0 / (1.0 + p + q)
        # L = I + 2B + B² ; R = (1+p)I + B²
        L00 = 1.0 + S00;       L01 = 2.0*B01 + S01;   L02 = 2.0*B02 + S02;   L03 = 2.0*B03 + S03
        L10 = 2.0*B10 + S10;   L11 = 1.0 + S11;       L12 = 2.0*B12 + S12;   L13 = 2.0*B13 + S13
        L20 = 2.0*B20 + S20;   L21 = 2.0*B21 + S21;   L22 = 1.0 + S22;       L23 = 2.0*B23 + S23
        L30 = 2.0*B30 + S30;   L31 = 2.0*B31 + S31;   L32 = 2.0*B32 + S32;   L33 = 1.0 + S33
        pp = 1.0 + p
        R00 = pp + S00; R01 = S01;      R02 = S02;      R03 = S03
        R10 = S10;      R11 = pp + S11; R12 = S12;      R13 = S13
        R20 = S20;      R21 = S21;      R22 = pp + S22; R23 = S23
        R30 = S30;      R31 = S31;      R32 = S32;      R33 = pp + S33
        (Y00, Y01, Y02, Y03, Y10, Y11, Y12, Y13,
         Y20, Y21, Y22, Y23, Y30, Y31, Y32, Y33) = _mm4(
            L00, L01, L02, L03, L10, L11, L12, L13,
            L20, L21, L22, L23, L30, L31, L32, L33,
            R00, R01, R02, R03, R10, R11, R12, R13,
            R20, R21, R22, R23, R30, R31, R32, R33)
        return (Y00*inv, Y01*inv, Y02*inv, Y03*inv, Y10*inv, Y11*inv, Y12*inv, Y13*inv,
                Y20*inv, Y21*inv, Y22*inv, Y23*inv, Y30*inv, Y31*inv, Y32*inv, Y33*inv)

    @triton.jit
    def _cayley4_vjp(
        a00, a01, a02, a03, a10, a11, a12, a13,
        a20, a21, a22, a23, a30, a31, a32, a33,
        gY00, gY01, gY02, gY03, gY10, gY11, gY12, gY13,
        gY20, gY21, gY22, gY23, gY30, gY31, gY32, gY33, ALPHA,
    ):
        """Exact analytic VJP of the closed-form Cayley. Returns gA = grad wrt res_raw A."""
        half = ALPHA * 0.5
        (B00, B01, B02, B03, B10, B11, B12, B13,
         B20, B21, B22, B23, B30, B31, B32, B33) = _skewB4(
            a00, a01, a02, a03, a10, a11, a12, a13,
            a20, a21, a22, a23, a30, a31, a32, a33, half)
        (S00, S01, S02, S03, S10, S11, S12, S13,
         S20, S21, S22, S23, S30, S31, S32, S33) = _mm4(
            B00, B01, B02, B03, B10, B11, B12, B13,
            B20, B21, B22, B23, B30, B31, B32, B33,
            B00, B01, B02, B03, B10, B11, B12, B13,
            B20, B21, B22, B23, B30, B31, B32, B33)      # B²
        p, q = _cayley4_pq(B00, B01, B02, B03, B10, B11, B12, B13,
                           B20, B21, B22, B23, B30, B31, B32, B33)
        inv = 1.0 / (1.0 + p + q)
        pp = 1.0 + p
        # R = (1+p)I + B²
        R00 = pp + S00; R01 = S01;      R02 = S02;      R03 = S03
        R10 = S10;      R11 = pp + S11; R12 = S12;      R13 = S13
        R20 = S20;      R21 = S21;      R22 = pp + S22; R23 = S23
        R30 = S30;      R31 = S31;      R32 = S32;      R33 = pp + S33
        # Y = (I + 2B + B²)·R·inv   →   need Y for (I+Y)ᵀ
        L00 = 1.0 + S00;       L01 = 2.0*B01 + S01;   L02 = 2.0*B02 + S02;   L03 = 2.0*B03 + S03
        L10 = 2.0*B10 + S10;   L11 = 1.0 + S11;       L12 = 2.0*B12 + S12;   L13 = 2.0*B13 + S13
        L20 = 2.0*B20 + S20;   L21 = 2.0*B21 + S21;   L22 = 1.0 + S22;       L23 = 2.0*B23 + S23
        L30 = 2.0*B30 + S30;   L31 = 2.0*B31 + S31;   L32 = 2.0*B32 + S32;   L33 = 1.0 + S33
        (Y00, Y01, Y02, Y03, Y10, Y11, Y12, Y13,
         Y20, Y21, Y22, Y23, Y30, Y31, Y32, Y33) = _mm4(
            L00, L01, L02, L03, L10, L11, L12, L13,
            L20, L21, L22, L23, L30, L31, L32, L33,
            R00, R01, R02, R03, R10, R11, R12, R13,
            R20, R21, R22, R23, R30, R31, R32, R33)
        Y00 *= inv; Y01 *= inv; Y02 *= inv; Y03 *= inv
        Y10 *= inv; Y11 *= inv; Y12 *= inv; Y13 *= inv
        Y20 *= inv; Y21 *= inv; Y22 *= inv; Y23 *= inv
        Y30 *= inv; Y31 *= inv; Y32 *= inv; Y33 *= inv
        # M = R·(I+B)·inv = (I−B)⁻¹
        IB00 = 1.0;  IB01 = B01;  IB02 = B02;  IB03 = B03
        IB10 = B10;  IB11 = 1.0;  IB12 = B12;  IB13 = B13
        IB20 = B20;  IB21 = B21;  IB22 = 1.0;  IB23 = B23
        IB30 = B30;  IB31 = B31;  IB32 = B32;  IB33 = 1.0
        (M00, M01, M02, M03, M10, M11, M12, M13,
         M20, M21, M22, M23, M30, M31, M32, M33) = _mm4(
            R00, R01, R02, R03, R10, R11, R12, R13,
            R20, R21, R22, R23, R30, R31, R32, R33,
            IB00, IB01, IB02, IB03, IB10, IB11, IB12, IB13,
            IB20, IB21, IB22, IB23, IB30, IB31, IB32, IB33)
        M00 *= inv; M01 *= inv; M02 *= inv; M03 *= inv
        M10 *= inv; M11 *= inv; M12 *= inv; M13 *= inv
        M20 *= inv; M21 *= inv; M22 *= inv; M23 *= inv
        M30 *= inv; M31 *= inv; M32 *= inv; M33 *= inv
        # U = gY · Mᵀ   (Mᵀ passed by transposing the arg block)
        (U00, U01, U02, U03, U10, U11, U12, U13,
         U20, U21, U22, U23, U30, U31, U32, U33) = _mm4(
            gY00, gY01, gY02, gY03, gY10, gY11, gY12, gY13,
            gY20, gY21, gY22, gY23, gY30, gY31, gY32, gY33,
            M00, M10, M20, M30, M01, M11, M21, M31,
            M02, M12, M22, M32, M03, M13, M23, M33)
        # (I+Y)ᵀ : transpose of (I+Y)
        T00 = 1.0 + Y00; T01 = Y01;       T02 = Y02;       T03 = Y03
        T10 = Y10;       T11 = 1.0 + Y11; T12 = Y12;       T13 = Y13
        T20 = Y20;       T21 = Y21;       T22 = 1.0 + Y22; T23 = Y23
        T30 = Y30;       T31 = Y31;       T32 = Y32;       T33 = 1.0 + Y33
        # gB = (I+Y)ᵀ · U   ((I+Y)ᵀ passed by transposing the arg block)
        (gB00, gB01, gB02, gB03, gB10, gB11, gB12, gB13,
         gB20, gB21, gB22, gB23, gB30, gB31, gB32, gB33) = _mm4(
            T00, T10, T20, T30, T01, T11, T21, T31,
            T02, T12, T22, T32, T03, T13, T23, T33,
            U00, U01, U02, U03, U10, U11, U12, U13,
            U20, U21, U22, U23, U30, U31, U32, U33)
        # gW = half·gB ; gA = gW − gWᵀ  (diagonal cancels)
        gA00 = 0.0
        gA01 = half*(gB01 - gB10); gA02 = half*(gB02 - gB20); gA03 = half*(gB03 - gB30)
        gA10 = half*(gB10 - gB01); gA11 = 0.0
        gA12 = half*(gB12 - gB21); gA13 = half*(gB13 - gB31)
        gA20 = half*(gB20 - gB02); gA21 = half*(gB21 - gB12); gA22 = 0.0
        gA23 = half*(gB23 - gB32)
        gA30 = half*(gB30 - gB03); gA31 = half*(gB31 - gB13); gA32 = half*(gB32 - gB23)
        gA33 = 0.0
        return (gA00, gA01, gA02, gA03, gA10, gA11, gA12, gA13,
                gA20, gA21, gA22, gA23, gA30, gA31, gA32, gA33)

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
        raw_ptr,          # [B, S, 48]   fp32  (proj_w @ x_flat, bias-free, pre-rms)
        pb_ptr,           # [48]         fp32  proj_b (added OUTSIDE the rms divide)
        xbar_ptr,         # [B, S, C]    bf16  out
        hres_ptr,         # [B, S, 4, 4] fp32  out
        hpostrow_ptr,     # [B, S, 4]    fp32  out
        hprecm_ptr,       # [B, S, 4]    fp32  out  (saved for backward)
        rms_ptr,          # [B, S, 1]    fp32  out  (saved for backward)
        nout_ptr,         # [rows, *] strided out — RMSNorm(x_bar)·nw (decode fold)
        nw_ptr,           # [C] norm weight (decode fold)
        TAU: tl.constexpr, ALPHA: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
        N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr,
        HAS_NOUT: tl.constexpr = False, neps=1e-6, snout=0,
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
            # bias added OUTSIDE the rms divide: h_map = raw·inv_rms + b, then ·inv_tau.
            p0 = (tl.load(raw_ptr + rb + i * 4 + 0) * inv_rms + tl.load(pb_ptr + i * 4 + 0)) * inv_tau
            p1 = (tl.load(raw_ptr + rb + i * 4 + 1) * inv_rms + tl.load(pb_ptr + i * 4 + 1)) * inv_tau
            p2 = (tl.load(raw_ptr + rb + i * 4 + 2) * inv_rms + tl.load(pb_ptr + i * 4 + 2)) * inv_tau
            p3 = (tl.load(raw_ptr + rb + i * 4 + 3) * inv_rms + tl.load(pb_ptr + i * 4 + 3)) * inv_tau
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
            q0 = (tl.load(raw_ptr + rb + 16 + 0 * 4 + j) * inv_rms + tl.load(pb_ptr + 16 + 0 * 4 + j)) * inv_tau
            q1 = (tl.load(raw_ptr + rb + 16 + 1 * 4 + j) * inv_rms + tl.load(pb_ptr + 16 + 1 * 4 + j)) * inv_tau
            q2 = (tl.load(raw_ptr + rb + 16 + 2 * 4 + j) * inv_rms + tl.load(pb_ptr + 16 + 2 * 4 + j)) * inv_tau
            q3 = (tl.load(raw_ptr + rb + 16 + 3 * 4 + j) * inv_rms + tl.load(pb_ptr + 16 + 3 * 4 + j)) * inv_tau
            s0, s1, s2, s3 = _sm4(q0, q1, q2, q3)
            hpr0 += s0
            hpr1 += s1
            hpr2 += s2
            hpr3 += s3
        tl.store(hpostrow_ptr + tok * 4 + 0, hpr0)
        tl.store(hpostrow_ptr + tok * 4 + 1, hpr1)
        tl.store(hpostrow_ptr + tok * 4 + 2, hpr2)
        tl.store(hpostrow_ptr + tok * 4 + 3, hpr3)

        # Hres = cayley(res): EXACT closed form (unconditionally orthogonal). ITERS ignored.
        # res A[i,j] = raw[32 + i*4+j]·inv_rms + b (bias OUTSIDE rms divide; NO tau on res).
        a00 = tl.load(raw_ptr + rb + 32 + 0) * inv_rms + tl.load(pb_ptr + 32 + 0)
        a01 = tl.load(raw_ptr + rb + 32 + 1) * inv_rms + tl.load(pb_ptr + 32 + 1)
        a02 = tl.load(raw_ptr + rb + 32 + 2) * inv_rms + tl.load(pb_ptr + 32 + 2)
        a03 = tl.load(raw_ptr + rb + 32 + 3) * inv_rms + tl.load(pb_ptr + 32 + 3)
        a10 = tl.load(raw_ptr + rb + 32 + 4) * inv_rms + tl.load(pb_ptr + 32 + 4)
        a11 = tl.load(raw_ptr + rb + 32 + 5) * inv_rms + tl.load(pb_ptr + 32 + 5)
        a12 = tl.load(raw_ptr + rb + 32 + 6) * inv_rms + tl.load(pb_ptr + 32 + 6)
        a13 = tl.load(raw_ptr + rb + 32 + 7) * inv_rms + tl.load(pb_ptr + 32 + 7)
        a20 = tl.load(raw_ptr + rb + 32 + 8) * inv_rms + tl.load(pb_ptr + 32 + 8)
        a21 = tl.load(raw_ptr + rb + 32 + 9) * inv_rms + tl.load(pb_ptr + 32 + 9)
        a22 = tl.load(raw_ptr + rb + 32 + 10) * inv_rms + tl.load(pb_ptr + 32 + 10)
        a23 = tl.load(raw_ptr + rb + 32 + 11) * inv_rms + tl.load(pb_ptr + 32 + 11)
        a30 = tl.load(raw_ptr + rb + 32 + 12) * inv_rms + tl.load(pb_ptr + 32 + 12)
        a31 = tl.load(raw_ptr + rb + 32 + 13) * inv_rms + tl.load(pb_ptr + 32 + 13)
        a32 = tl.load(raw_ptr + rb + 32 + 14) * inv_rms + tl.load(pb_ptr + 32 + 14)
        a33 = tl.load(raw_ptr + rb + 32 + 15) * inv_rms + tl.load(pb_ptr + 32 + 15)
        (y00, y01, y02, y03, y10, y11, y12, y13,
         y20, y21, y22, y23, y30, y31, y32, y33) = _cayley4_fwd(
            a00, a01, a02, a03, a10, a11, a12, a13,
            a20, a21, a22, a23, a30, a31, a32, a33, ALPHA)
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

        if HAS_NOUT:
            # decode-engine fold: RMSNorm(x_bar)·nw written straight into a strided
            # slot (the attention x-history staging row). Mirrors rmsnorm_rows: acc
            # is the same fp32 value the separate kernel would re-load.
            ms = tl.sum(acc * acc, axis=0) / C
            rn = 1.0 / tl.sqrt(ms + neps)
            nw = tl.load(nw_ptr + c, mask=cmask, other=0.0)
            tl.store(nout_ptr + tok * snout + c, (acc * rn) * nw, mask=cmask)

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
        raw_ptr,          # [B,S,48]   raw_full (bias-free, pre-rms)
        pb_ptr,           # [48]       proj_b
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
        # h_map = raw·inv_rms + b, then softmax-arg ·inv_tau (bias OUTSIDE the rms divide).
        pre00 = (tl.load(raw_ptr+rb+0)*inv_rms+tl.load(pb_ptr+0))*inv_tau;  pre01 = (tl.load(raw_ptr+rb+1)*inv_rms+tl.load(pb_ptr+1))*inv_tau
        pre02 = (tl.load(raw_ptr+rb+2)*inv_rms+tl.load(pb_ptr+2))*inv_tau;  pre03 = (tl.load(raw_ptr+rb+3)*inv_rms+tl.load(pb_ptr+3))*inv_tau
        pre10 = (tl.load(raw_ptr+rb+4)*inv_rms+tl.load(pb_ptr+4))*inv_tau;  pre11 = (tl.load(raw_ptr+rb+5)*inv_rms+tl.load(pb_ptr+5))*inv_tau
        pre12 = (tl.load(raw_ptr+rb+6)*inv_rms+tl.load(pb_ptr+6))*inv_tau;  pre13 = (tl.load(raw_ptr+rb+7)*inv_rms+tl.load(pb_ptr+7))*inv_tau
        pre20 = (tl.load(raw_ptr+rb+8)*inv_rms+tl.load(pb_ptr+8))*inv_tau;  pre21 = (tl.load(raw_ptr+rb+9)*inv_rms+tl.load(pb_ptr+9))*inv_tau
        pre22 = (tl.load(raw_ptr+rb+10)*inv_rms+tl.load(pb_ptr+10))*inv_tau; pre23 = (tl.load(raw_ptr+rb+11)*inv_rms+tl.load(pb_ptr+11))*inv_tau
        pre30 = (tl.load(raw_ptr+rb+12)*inv_rms+tl.load(pb_ptr+12))*inv_tau; pre31 = (tl.load(raw_ptr+rb+13)*inv_rms+tl.load(pb_ptr+13))*inv_tau
        pre32 = (tl.load(raw_ptr+rb+14)*inv_rms+tl.load(pb_ptr+14))*inv_tau; pre33 = (tl.load(raw_ptr+rb+15)*inv_rms+tl.load(pb_ptr+15))*inv_tau

        P00, P01, P02, P03 = _sm4(pre00, pre01, pre02, pre03)
        P10, P11, P12, P13 = _sm4(pre10, pre11, pre12, pre13)
        P20, P21, P22, P23 = _sm4(pre20, pre21, pre22, pre23)
        P30, P31, P32, P33 = _sm4(pre30, pre31, pre32, pre33)

        # ---- post (cols softmax, dim=-2): for col j softmax across rows i ----
        po00 = (tl.load(raw_ptr+rb+16+0)*inv_rms+tl.load(pb_ptr+16+0))*inv_tau;  po01 = (tl.load(raw_ptr+rb+16+1)*inv_rms+tl.load(pb_ptr+16+1))*inv_tau
        po02 = (tl.load(raw_ptr+rb+16+2)*inv_rms+tl.load(pb_ptr+16+2))*inv_tau;  po03 = (tl.load(raw_ptr+rb+16+3)*inv_rms+tl.load(pb_ptr+16+3))*inv_tau
        po10 = (tl.load(raw_ptr+rb+16+4)*inv_rms+tl.load(pb_ptr+16+4))*inv_tau;  po11 = (tl.load(raw_ptr+rb+16+5)*inv_rms+tl.load(pb_ptr+16+5))*inv_tau
        po12 = (tl.load(raw_ptr+rb+16+6)*inv_rms+tl.load(pb_ptr+16+6))*inv_tau;  po13 = (tl.load(raw_ptr+rb+16+7)*inv_rms+tl.load(pb_ptr+16+7))*inv_tau
        po20 = (tl.load(raw_ptr+rb+16+8)*inv_rms+tl.load(pb_ptr+16+8))*inv_tau;  po21 = (tl.load(raw_ptr+rb+16+9)*inv_rms+tl.load(pb_ptr+16+9))*inv_tau
        po22 = (tl.load(raw_ptr+rb+16+10)*inv_rms+tl.load(pb_ptr+16+10))*inv_tau; po23 = (tl.load(raw_ptr+rb+16+11)*inv_rms+tl.load(pb_ptr+16+11))*inv_tau
        po30 = (tl.load(raw_ptr+rb+16+12)*inv_rms+tl.load(pb_ptr+16+12))*inv_tau; po31 = (tl.load(raw_ptr+rb+16+13)*inv_rms+tl.load(pb_ptr+16+13))*inv_tau
        po32 = (tl.load(raw_ptr+rb+16+14)*inv_rms+tl.load(pb_ptr+16+14))*inv_tau; po33 = (tl.load(raw_ptr+rb+16+15)*inv_rms+tl.load(pb_ptr+16+15))*inv_tau
        # column j softmax across i: col0 = (po00,po10,po20,po30) etc.
        Q00, Q10, Q20, Q30 = _sm4(po00, po10, po20, po30)   # col 0 -> Hpost[i,0]
        Q01, Q11, Q21, Q31 = _sm4(po01, po11, po21, po31)   # col 1
        Q02, Q12, Q22, Q32 = _sm4(po02, po12, po22, po32)   # col 2
        Q03, Q13, Q23, Q33 = _sm4(po03, po13, po23, po33)   # col 3

        # ---- cayley res A[i,j] = raw·inv_rms + b (bias OUTSIDE rms; NO tau). Keep the
        #      bias-free wxs = raw·inv_rms separately for the shared /rms grms accumulation. ----
        aw00 = tl.load(raw_ptr+rb+32+0)*inv_rms;  aw01 = tl.load(raw_ptr+rb+32+1)*inv_rms
        aw02 = tl.load(raw_ptr+rb+32+2)*inv_rms;  aw03 = tl.load(raw_ptr+rb+32+3)*inv_rms
        aw10 = tl.load(raw_ptr+rb+32+4)*inv_rms;  aw11 = tl.load(raw_ptr+rb+32+5)*inv_rms
        aw12 = tl.load(raw_ptr+rb+32+6)*inv_rms;  aw13 = tl.load(raw_ptr+rb+32+7)*inv_rms
        aw20 = tl.load(raw_ptr+rb+32+8)*inv_rms;  aw21 = tl.load(raw_ptr+rb+32+9)*inv_rms
        aw22 = tl.load(raw_ptr+rb+32+10)*inv_rms; aw23 = tl.load(raw_ptr+rb+32+11)*inv_rms
        aw30 = tl.load(raw_ptr+rb+32+12)*inv_rms; aw31 = tl.load(raw_ptr+rb+32+13)*inv_rms
        aw32 = tl.load(raw_ptr+rb+32+14)*inv_rms; aw33 = tl.load(raw_ptr+rb+32+15)*inv_rms
        a00 = aw00 + tl.load(pb_ptr+32+0);  a01 = aw01 + tl.load(pb_ptr+32+1)
        a02 = aw02 + tl.load(pb_ptr+32+2);  a03 = aw03 + tl.load(pb_ptr+32+3)
        a10 = aw10 + tl.load(pb_ptr+32+4);  a11 = aw11 + tl.load(pb_ptr+32+5)
        a12 = aw12 + tl.load(pb_ptr+32+6);  a13 = aw13 + tl.load(pb_ptr+32+7)
        a20 = aw20 + tl.load(pb_ptr+32+8);  a21 = aw21 + tl.load(pb_ptr+32+9)
        a22 = aw22 + tl.load(pb_ptr+32+10); a23 = aw23 + tl.load(pb_ptr+32+11)
        a30 = aw30 + tl.load(pb_ptr+32+12); a31 = aw31 + tl.load(pb_ptr+32+13)
        a32 = aw32 + tl.load(pb_ptr+32+14); a33 = aw33 + tl.load(pb_ptr+32+15)

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

        # ---- Hres = cayley: EXACT analytic VJP (closed form). gres = gA (grad wrt res A). ----
        gY00 = tl.load(ghres_ptr+hb+0); gY01 = tl.load(ghres_ptr+hb+1)
        gY02 = tl.load(ghres_ptr+hb+2); gY03 = tl.load(ghres_ptr+hb+3)
        gY10 = tl.load(ghres_ptr+hb+4); gY11 = tl.load(ghres_ptr+hb+5)
        gY12 = tl.load(ghres_ptr+hb+6); gY13 = tl.load(ghres_ptr+hb+7)
        gY20 = tl.load(ghres_ptr+hb+8); gY21 = tl.load(ghres_ptr+hb+9)
        gY22 = tl.load(ghres_ptr+hb+10); gY23 = tl.load(ghres_ptr+hb+11)
        gY30 = tl.load(ghres_ptr+hb+12); gY31 = tl.load(ghres_ptr+hb+13)
        gY32 = tl.load(ghres_ptr+hb+14); gY33 = tl.load(ghres_ptr+hb+15)
        (gres00,gres01,gres02,gres03,gres10,gres11,gres12,gres13,
         gres20,gres21,gres22,gres23,gres30,gres31,gres32,gres33) = _cayley4_vjp(
            a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33,
            gY00,gY01,gY02,gY03,gY10,gY11,gY12,gY13,gY20,gY21,gY22,gY23,gY30,gY31,gY32,gY33, ALPHA)

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
        # pre/post grms terms need bias-free wxs = raw·inv_rms = pre·TAU − b_pre.
        grms = 0.0
        grms += gpre00*(pre00*TAU-tl.load(pb_ptr+0))+gpre01*(pre01*TAU-tl.load(pb_ptr+1))+gpre02*(pre02*TAU-tl.load(pb_ptr+2))+gpre03*(pre03*TAU-tl.load(pb_ptr+3))
        grms += gpre10*(pre10*TAU-tl.load(pb_ptr+4))+gpre11*(pre11*TAU-tl.load(pb_ptr+5))+gpre12*(pre12*TAU-tl.load(pb_ptr+6))+gpre13*(pre13*TAU-tl.load(pb_ptr+7))
        grms += gpre20*(pre20*TAU-tl.load(pb_ptr+8))+gpre21*(pre21*TAU-tl.load(pb_ptr+9))+gpre22*(pre22*TAU-tl.load(pb_ptr+10))+gpre23*(pre23*TAU-tl.load(pb_ptr+11))
        grms += gpre30*(pre30*TAU-tl.load(pb_ptr+12))+gpre31*(pre31*TAU-tl.load(pb_ptr+13))+gpre32*(pre32*TAU-tl.load(pb_ptr+14))+gpre33*(pre33*TAU-tl.load(pb_ptr+15))
        grms += gpo00*(po00*TAU-tl.load(pb_ptr+16+0))+gpo01*(po01*TAU-tl.load(pb_ptr+16+1))+gpo02*(po02*TAU-tl.load(pb_ptr+16+2))+gpo03*(po03*TAU-tl.load(pb_ptr+16+3))
        grms += gpo10*(po10*TAU-tl.load(pb_ptr+16+4))+gpo11*(po11*TAU-tl.load(pb_ptr+16+5))+gpo12*(po12*TAU-tl.load(pb_ptr+16+6))+gpo13*(po13*TAU-tl.load(pb_ptr+16+7))
        grms += gpo20*(po20*TAU-tl.load(pb_ptr+16+8))+gpo21*(po21*TAU-tl.load(pb_ptr+16+9))+gpo22*(po22*TAU-tl.load(pb_ptr+16+10))+gpo23*(po23*TAU-tl.load(pb_ptr+16+11))
        grms += gpo30*(po30*TAU-tl.load(pb_ptr+16+12))+gpo31*(po31*TAU-tl.load(pb_ptr+16+13))+gpo32*(po32*TAU-tl.load(pb_ptr+16+14))+gpo33*(po33*TAU-tl.load(pb_ptr+16+15))
        # res grms term uses the BIAS-FREE wxs (raw·inv_rms); bias does not depend on rms.
        grms += gres00*aw00+gres01*aw01+gres02*aw02+gres03*aw03
        grms += gres10*aw10+gres11*aw11+gres12*aw12+gres13*aw13
        grms += gres20*aw20+gres21*aw21+gres22*aw22+gres23*aw23
        grms += gres30*aw30+gres31*aw31+gres32*aw32+gres33*aw33
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

    # =======================================================================
    # GENERIC (n-arbitrary) PRE-MAPPING — [N,N] register-tile reformulation.
    #
    # Numerically EQUIVALENT to _hc_premap_fwd/bwd_kernel (the hand-unrolled 4×4
    # scalar path) but parameterized by constexpr N: the n×n mapping math lives in
    # small [N,N] fp32 register tiles instead of a00..a33 scalars. This serves any
    # N (production uses it for n=2; the 4×4 scalar kernel stays the tuned default
    # for n=4). Validated two ways in ignore/verify_hc_ngeneric.py: generic@N=4 vs
    # the scalar kernel (bit-level oracle) AND generic@N=2 vs the eager reference.
    #
    # Tile algebra (one program = one (b,s) token):
    #   matmul  A@B   = tl.sum(A[:,:,None]*B[None,:,:], axis=1)      (_tile_mm)
    #   skew    W     = res - resᵀ                                   (tl.trans)
    #   softmax along an axis = exp(x - max)/sum(exp)                (axis reduce)
    # Cayley iters fixed to 3 (HC architectural constant; the scalar path is the
    # same). The backward tile recursion reduces exactly to _cayley_bwd_step /
    # _smjac4 (verified by hand): gW += half·gY@Tᵀ ; gY ← half·Wᵀ@gY.
    # -----------------------------------------------------------------------
    @triton.jit
    def _tile_mm(A, B, N: tl.constexpr):
        """[N,N]@[N,N] via broadcast contraction (N tiny: 2 or 4)."""
        return tl.sum(A[:, :, None] * B[None, :, :], axis=1)

    @triton.jit
    def _cayley_tile_YM(B, Ieye, N: tl.constexpr):
        """Exact orthogonal Cayley Y and M=(I−B)⁻¹ from skew tile B (N∈{2,4}).

        N=2: closed 2×2 rotation.  N=4: Cayley–Hamilton closed form (matches the
        scalar path / cayley_orthogonal). Both are exact at any ‖B‖ (denom ≥ 1).
        """
        p = 0.5 * tl.sum(B * B)                              # ½‖B‖²_F
        if N == 2:
            d = 1.0 + p
            Y = ((1.0 - p) * Ieye + 2.0 * B) / d
            M = (Ieye + B) / d
        else:                                               # N == 4
            B2 = _tile_mm(B, B, N)
            rr = tl.arange(0, N)
            b01 = tl.sum(tl.where((rr[:, None] == 0) & (rr[None, :] == 1), B, 0.0))
            b23 = tl.sum(tl.where((rr[:, None] == 2) & (rr[None, :] == 3), B, 0.0))
            b02 = tl.sum(tl.where((rr[:, None] == 0) & (rr[None, :] == 2), B, 0.0))
            b13 = tl.sum(tl.where((rr[:, None] == 1) & (rr[None, :] == 3), B, 0.0))
            b03 = tl.sum(tl.where((rr[:, None] == 0) & (rr[None, :] == 3), B, 0.0))
            b12 = tl.sum(tl.where((rr[:, None] == 1) & (rr[None, :] == 2), B, 0.0))
            Pf = b01 * b23 - b02 * b13 + b03 * b12
            d = 1.0 + p + Pf * Pf
            R = (1.0 + p) * Ieye + B2
            Y = _tile_mm(Ieye + 2.0 * B + B2, R, N) / d
            M = _tile_mm(R, Ieye + B, N) / d
        return Y, M

    @triton.jit
    def _hc_premap_fwd_kernel_g(
        h_ptr,            # [B, S, N, C] bf16
        raw_ptr,          # [B, S, 3*N*N] fp32  (proj_w @ x_flat, bias-free, pre-rms)
        pb_ptr,           # [3*N*N]      fp32  proj_b (added OUTSIDE the rms divide)
        xbar_ptr,         # [B, S, C]    bf16  out
        hres_ptr,         # [B, S, N, N] fp32  out
        hpostrow_ptr,     # [B, S, N]    fp32  out
        rms_ptr,          # [B, S, 1]    fp32  out  (saved for backward)
        TAU: tl.constexpr, ALPHA: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
        N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        rr = tl.arange(0, N)
        ij = rr[:, None] * N + rr[None, :]          # [N,N] flat offsets
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        NN = N * N
        h_base = tok * (N * C)
        rb = tok * (3 * NN)

        # carrier tile h[N, BLOCK_C] (fp32) — loaded once, reused for rms + x_bar.
        hh = tl.load(h_ptr + h_base + rr[:, None] * C + c[None, :],
                     mask=cmask[None, :], other=0.0).to(tl.float32)
        ssq = tl.sum(hh * hh)                        # scalar over [N, C]
        rms = tl.sqrt(ssq / (N * C) + EPS)
        inv_rms = 1.0 / rms
        inv_tau = 1.0 / TAU
        tl.store(rms_ptr + tok, rms)

        # raw blocks (pre|post|res), each [N,N]. Bias added OUTSIDE the rms divide:
        # h_map = raw·inv_rms + b ; softmax-arg then ·inv_tau (res gets NO tau).
        b_pre = tl.load(pb_ptr + 0 * NN + ij).to(tl.float32)
        b_post = tl.load(pb_ptr + 1 * NN + ij).to(tl.float32)
        b_res = tl.load(pb_ptr + 2 * NN + ij).to(tl.float32)
        pre_s = (tl.load(raw_ptr + rb + 0 * NN + ij).to(tl.float32) * inv_rms + b_pre) * inv_tau
        post_s = (tl.load(raw_ptr + rb + 1 * NN + ij).to(tl.float32) * inv_rms + b_post) * inv_tau
        res_s = tl.load(raw_ptr + rb + 2 * NN + ij).to(tl.float32) * inv_rms + b_res

        # Hpre = softmax(pre_s, dim=-1) rows ; Hpre_cm[j] = mean_i Hpre[i,j]
        pm = tl.max(pre_s, axis=1)[:, None]
        pe = tl.exp(pre_s - pm)
        Hpre = pe / tl.sum(pe, axis=1)[:, None]
        Hpre_cm = tl.sum(Hpre, axis=0) / N           # [N]

        # Hpost = softmax(post_s, dim=-2) cols ; Hpost_row[i] = sum_j Hpost[i,j]
        qm = tl.max(post_s, axis=0)[None, :]
        qe = tl.exp(post_s - qm)
        Hpost = qe / tl.sum(qe, axis=0)[None, :]
        Hpost_row = tl.sum(Hpost, axis=1)            # [N]

        # Hres = cayley(res_s): EXACT closed form (N∈{2,4}), unconditionally orthogonal.
        Ieye = (rr[:, None] == rr[None, :]).to(tl.float32)
        Bm = (ALPHA * 0.5) * (res_s - tl.trans(res_s))       # B = ½α·(A−Aᵀ), skew
        Y, _M = _cayley_tile_YM(Bm, Ieye, N)

        # x_bar[c] = Σ_j Hpre_cm[j] h[j,c]
        x_bar = tl.sum(Hpre_cm[:, None] * hh, axis=0)   # [BLOCK_C]

        tl.store(hres_ptr + tok * NN + ij, Y)
        tl.store(hpostrow_ptr + tok * N + rr, Hpost_row)
        tl.store(xbar_ptr + tok * C + c, x_bar.to(xbar_ptr.dtype.element_ty), mask=cmask)

    @triton.jit
    def _hc_premap_bwd_kernel_g(
        gxbar_ptr,        # [B,S,C]    upstream grad on x_bar
        ghres_ptr,        # [B,S,N,N]  upstream grad on Hres
        ghpostrow_ptr,    # [B,S,N]    upstream grad on Hpost_row
        h_ptr,            # [B,S,N,C]
        raw_ptr,          # [B,S,3*N*N] raw_full (bias-free, pre-rms)
        pb_ptr,           # [3*N*N]     proj_b
        rms_ptr,          # [B,S,1]
        graw_ptr,         # [B,S,3*N*N] out: grad wrt raw_full (pre-rms)
        ghpart_ptr,       # [B,S,N,C]   out: grad on h (xbar-path + rms-path)
        TAU: tl.constexpr, ALPHA: tl.constexpr, ITERS: tl.constexpr,
        N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        tok = tl.program_id(0)
        rr = tl.arange(0, N)
        ij = rr[:, None] * N + rr[None, :]
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        NN = N * N
        h_base = tok * (N * C)
        rb = tok * (3 * NN)

        rms = tl.load(rms_ptr + tok)
        inv_rms = 1.0 / rms
        inv_tau = 1.0 / TAU

        # ---- recompute forward mapping (fp32 tiles) with bias OUTSIDE the /rms divide ----
        b_pre = tl.load(pb_ptr + 0 * NN + ij).to(tl.float32)
        b_post = tl.load(pb_ptr + 1 * NN + ij).to(tl.float32)
        b_res = tl.load(pb_ptr + 2 * NN + ij).to(tl.float32)
        # bias-free wxs = raw·inv_rms (needed for the shared /rms grms accumulation)
        wxs_pre = tl.load(raw_ptr + rb + 0 * NN + ij).to(tl.float32) * inv_rms
        wxs_post = tl.load(raw_ptr + rb + 1 * NN + ij).to(tl.float32) * inv_rms
        wxs_res = tl.load(raw_ptr + rb + 2 * NN + ij).to(tl.float32) * inv_rms
        pre_s = (wxs_pre + b_pre) * inv_tau
        post_s = (wxs_post + b_post) * inv_tau
        res_s = wxs_res + b_res

        pm = tl.max(pre_s, axis=1)[:, None]
        pe = tl.exp(pre_s - pm)
        Hpre = pe / tl.sum(pe, axis=1)[:, None]
        qm = tl.max(post_s, axis=0)[None, :]
        qe = tl.exp(post_s - qm)
        Hpost = qe / tl.sum(qe, axis=0)[None, :]

        Ieye = (rr[:, None] == rr[None, :]).to(tl.float32)
        half = ALPHA * 0.5
        Bm = half * (res_s - tl.trans(res_s))       # B = ½α·(A−Aᵀ), skew
        Y, M = _cayley_tile_YM(Bm, Ieye, N)         # exact Y and M=(I−B)⁻¹

        # ---- x_bar path ----
        hh = tl.load(h_ptr + h_base + rr[:, None] * C + c[None, :],
                     mask=cmask[None, :], other=0.0).to(tl.float32)     # [N, BLOCK_C]
        gx = tl.load(gxbar_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)  # [BLOCK_C]
        # Hpre_cm[j] = mean_i Hpre[i,j]
        Hpre_cm = tl.sum(Hpre, axis=0) / N                              # [N]
        g_Hpre_cm = tl.sum(gx[None, :] * hh, axis=1)                    # [N]  Σ_c gx·h_j
        grad_h_xbar = Hpre_cm[:, None] * gx[None, :]                    # [N, BLOCK_C]

        # ---- Hpre_cm -> Hpre -> softmax(row) VJP -> pre ----
        g_Hpre = (g_Hpre_cm[None, :]) / N                              # [1,N] -> [N,N] broadcast
        g_Hpre = tl.broadcast_to(g_Hpre, (N, N))
        dot_p = tl.sum(g_Hpre * Hpre, axis=1)[:, None]
        g_pre_s = Hpre * (g_Hpre - dot_p)                              # d/d pre_s
        g_rs_pre = g_pre_s * inv_tau                                   # d/d raw_scaled_pre

        # ---- Hpost_row -> Hpost -> softmax(col) VJP -> post ----
        g_hpr = tl.load(ghpostrow_ptr + tok * N + rr)                  # [N]
        g_Hpost = tl.broadcast_to(g_hpr[:, None], (N, N))             # g_Hpost[i,j]=g_hpr[i]
        dot_q = tl.sum(g_Hpost * Hpost, axis=0)[None, :]
        g_post_s = Hpost * (g_Hpost - dot_q)
        g_rs_post = g_post_s * inv_tau

        # ---- Hres = cayley : EXACT analytic VJP (closed form) ----
        # gB = (I+Y)ᵀ·gY·Mᵀ ; gW = ½α·gB ; g_res = gW − gWᵀ = ½α·(gB − gBᵀ).
        gY = tl.load(ghres_ptr + tok * NN + ij).to(tl.float32)        # [N,N]
        U = _tile_mm(gY, tl.trans(M), N)                             # gY·Mᵀ
        gB = _tile_mm(tl.trans(Ieye + Y), U, N)                      # (I+Y)ᵀ·U
        gW = half * gB
        g_rs_res = gW - tl.trans(gW)                                 # d/d res_s (= res A)

        # ---- g_rms (folds back through the shared /rms). Uses the BIAS-FREE wxs; the
        #      bias does not depend on rms. ----
        grms_inner = (tl.sum(g_rs_pre * wxs_pre)
                      + tl.sum(g_rs_post * wxs_post)
                      + tl.sum(g_rs_res * wxs_res))
        grms = -inv_rms * grms_inner

        # ---- grad wrt raw_full = g_rs * inv_rms ----
        tl.store(graw_ptr + rb + 0 * NN + ij, g_rs_pre * inv_rms)
        tl.store(graw_ptr + rb + 1 * NN + ij, g_rs_post * inv_rms)
        tl.store(graw_ptr + rb + 2 * NN + ij, g_rs_res * inv_rms)

        # ---- rms-path grad on h + xbar-path -> grad_h_partial ----
        scale = grms / (N * C * rms)
        grad_h = grad_h_xbar + scale * hh                             # [N, BLOCK_C]
        tl.store(ghpart_ptr + h_base + rr[:, None] * C + c[None, :],
                 grad_h.to(ghpart_ptr.dtype.element_ty), mask=cmask[None, :])

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
        term_ptr,         # [B, S, C] or dummy — single-stream inject for the NEXT layer
        N: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr, HAS_TERM: tl.constexpr,
    ):
        tok = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        h_base = tok * (N * C)
        yv = tl.load(y_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)
        # carrier-engine: fold the next layer's broadcast inject into this POST write
        # (out[i] += term for every output stream i), killing a separate _apply_injection
        # carrier read+write. term is the same for all i, loaded once.
        if HAS_TERM:
            termv = tl.load(term_ptr + tok * C + c, mask=cmask, other=0.0).to(tl.float32)

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
            if HAS_TERM:
                acc += termv
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
        gterm_ptr,        # [B, S, C] or dummy  out (grad wrt the folded inject term)
        N: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr, HAS_TERM: tl.constexpr,
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
        # term is added to EVERY output stream i, so grad_term = sum_i grad_out[i].
        gterm = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for i in tl.static_range(N):
            go = tl.load(gout_ptr + h_base + i * C + c, mask=cmask, other=0.0).to(tl.float32)
            post_i = tl.load(hpost_ptr + tok * N + i).to(tl.float32)
            gy += post_i * go
            gterm += go
            # grad_Hpost_row[i] = sum_c go * y
            ghpost_i = tl.sum(go * yv, axis=0)
            tl.store(ghpost_ptr + tok * N + i, ghpost_i)
            # grad_Hres[i,j] = sum_c go * h_j
            for j in tl.static_range(N):
                hj = tl.load(h_ptr + h_base + j * C + c, mask=cmask, other=0.0).to(tl.float32)
                ghres_ij = tl.sum(go * hj, axis=0)
                tl.store(ghres_ptr + tok * (N * N) + i * N + j, ghres_ij)

        tl.store(gy_ptr + tok * C + c, gy.to(gy_ptr.dtype.element_ty), mask=cmask)
        if HAS_TERM:
            tl.store(gterm_ptr + tok * C + c, gterm.to(gterm_ptr.dtype.element_ty), mask=cmask)


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
        # bias-under-rms fix: raw_full = x·Wᵀ WITHOUT bias (no addmm). The kernel adds
        # proj_b OUTSIDE the /rms divide (h_map = raw·inv_rms + b), matching eager _mappings.
        dt = h.dtype
        raw_full = torch.mm(
            x_flat, proj_w.to(dt).t()
        ).float().reshape(B, S, 48).contiguous()
        proj_b_f = proj_b.float().contiguous()

        xbar = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
        hpostrow = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        hprecm = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        rms = torch.empty(B, S, 1, device=h.device, dtype=torch.float32)
        BLOCK_C = _next_pow2(C)
        _hc_premap_fwd_kernel[(B * S,)](
            h, raw_full, proj_b_f, xbar, hres, hpostrow, hprecm, rms, xbar, xbar,
            TAU=float(tau), ALPHA=float(alpha), ITERS=int(iters), EPS=float(eps),
            N=N, C=C, BLOCK_C=BLOCK_C, HAS_NOUT=False, **_LAUNCH,
        )
        ctx.save_for_backward(h, raw_full, proj_b_f, rms, proj_w)
        ctx.shape = (B, S, N, C)
        ctx.cfg = (float(tau), float(alpha), int(iters))
        return xbar, hres, hpostrow

    @staticmethod
    def backward(ctx, grad_xbar, grad_hres, grad_hpostrow):
        h, raw_full, proj_b_f, rms, proj_w = ctx.saved_tensors
        B, S, N, C = ctx.shape
        tau, alpha, iters = ctx.cfg
        grad_xbar = grad_xbar.contiguous()
        grad_hres = grad_hres.contiguous().float()
        grad_hpostrow = grad_hpostrow.contiguous().float()

        graw = torch.empty(B, S, 48, device=h.device, dtype=torch.float32)
        gh_partial = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        BLOCK_C = _next_pow2(C)
        _hc_premap_bwd_kernel[(B * S,)](
            grad_xbar, grad_hres, grad_hpostrow, h, raw_full, proj_b_f, rms,
            graw, gh_partial,
            TAU=tau, ALPHA=alpha, ITERS=iters,
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        # proj VJP via cuBLAS. h_map = (x·Wᵀ)·inv_rms + b, so graw = grad wrt (x·Wᵀ) =
        # grad_hmap·inv_rms (unchanged by the bias fix) → grad_w / grad_h_proj unchanged.
        # grad_b = Σ_tok grad_hmap = Σ_tok graw·rms (NOT graw.sum), since d h_map/d b = 1.
        dt = h.dtype
        graw2 = graw.reshape(B * S, 48)
        graw2_dt = graw2.to(dt)
        x_flat = h.reshape(B * S, N * C)
        grad_w = (graw2_dt.t() @ x_flat).float()                  # [48, N*C]  (fp32 accum)
        grad_b = (graw2 * rms.reshape(B * S, 1)).sum(0)           # [48]  Σ grad_hmap (exact fp32)
        grad_h_proj = (graw2_dt @ proj_w.to(dt)).reshape(B, S, N, C)  # [B,S,n,C]
        grad_h = (gh_partial.to(dt) + grad_h_proj)
        return (grad_h, grad_w.to(proj_w.dtype), grad_b.to(proj_w.dtype),
                None, None, None, None)


class _FusedHCPreMapGeneric(torch.autograd.Function):
    """n-generic fused n×n mapping + x_bar (tile form). Returns (x_bar, Hres, Hpost_row).

    Same contract + cuBLAS proj wrapper as _FusedHCPreMap, but the in-kernel mapping
    is [N,N]-tile (constexpr N) so it serves any N (production: n=2). The 4×4 scalar
    _FusedHCPreMap stays the tuned default for n=4; this is gated == it at N=4.
    """

    @staticmethod
    def forward(ctx, h, proj_w, proj_b, tau, alpha, iters, eps):
        B, S, N, C = h.shape
        assert int(iters) == 3, "fused premap backward unrolls cayley to exactly 3 iters"
        h = h.contiguous()
        NN = N * N
        x_flat = h.reshape(B * S, N * C)
        # bias-under-rms fix: bias-free GEMM (bf16 in, fp32 accumulate); kernel adds proj_b
        # OUTSIDE the /rms divide, matching eager _mappings.
        dt = h.dtype
        raw_full = torch.mm(
            x_flat, proj_w.to(dt).t()
        ).float().reshape(B, S, 3 * NN).contiguous()
        proj_b_f = proj_b.float().contiguous()

        xbar = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
        hpostrow = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        rms = torch.empty(B, S, 1, device=h.device, dtype=torch.float32)
        BLOCK_C = _next_pow2(C)
        _hc_premap_fwd_kernel_g[(B * S,)](
            h, raw_full, proj_b_f, xbar, hres, hpostrow, rms,
            TAU=float(tau), ALPHA=float(alpha), ITERS=int(iters), EPS=float(eps),
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        ctx.save_for_backward(h, raw_full, proj_b_f, rms, proj_w)
        ctx.shape = (B, S, N, C)
        ctx.cfg = (float(tau), float(alpha), int(iters))
        return xbar, hres, hpostrow

    @staticmethod
    def backward(ctx, grad_xbar, grad_hres, grad_hpostrow):
        h, raw_full, proj_b_f, rms, proj_w = ctx.saved_tensors
        B, S, N, C = ctx.shape
        tau, alpha, iters = ctx.cfg
        NN = N * N
        grad_xbar = grad_xbar.contiguous()
        grad_hres = grad_hres.contiguous().float()
        grad_hpostrow = grad_hpostrow.contiguous().float()

        graw = torch.empty(B, S, 3 * NN, device=h.device, dtype=torch.float32)
        gh_partial = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        BLOCK_C = _next_pow2(C)
        _hc_premap_bwd_kernel_g[(B * S,)](
            grad_xbar, grad_hres, grad_hpostrow, h, raw_full, proj_b_f, rms,
            graw, gh_partial,
            TAU=tau, ALPHA=alpha, ITERS=iters,
            N=N, C=C, BLOCK_C=BLOCK_C, **_LAUNCH,
        )
        # proj VJP via cuBLAS. graw = grad wrt (x·Wᵀ) = grad_hmap·inv_rms (bias fix leaves
        # grad_w/grad_h_proj unchanged); grad_b = Σ_tok grad_hmap = Σ_tok graw·rms.
        dt = h.dtype
        graw2 = graw.reshape(B * S, 3 * NN)
        graw2_dt = graw2.to(dt)
        x_flat = h.reshape(B * S, N * C)
        grad_w = (graw2_dt.t() @ x_flat).float()
        grad_b = (graw2 * rms.reshape(B * S, 1)).sum(0)
        grad_h_proj = (graw2_dt @ proj_w.to(dt)).reshape(B, S, N, C)
        grad_h = (gh_partial.to(dt) + grad_h_proj)
        return (grad_h, grad_w.to(proj_w.dtype), grad_b.to(proj_w.dtype),
                None, None, None, None)


# ===========================================================================
# autograd.Function — POST (x_mix + x_post + add)
# ===========================================================================

class _FusedHCPost(torch.autograd.Function):
    """out[b,s,i,c] = sum_j Hres[i,j]*h[j,c] + Hpost_row[i]*y[c] (+ term[c] folded inject)."""

    @staticmethod
    def forward(ctx, hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor, term=None):
        B, S, N, C = h.shape
        hres = hres.contiguous()
        hpost_row = hpost_row.contiguous()
        h = h.contiguous()
        y = y.contiguous()
        has_term = term is not None
        term_arg = term.contiguous() if has_term else y   # dummy ptr when unused (not read)
        out = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        BLOCK_C = _next_pow2(C)
        _hc_post_fwd_kernel[(B * S,)](
            hres, hpost_row, h, y, out, term_arg,
            N=N, C=C, BLOCK_C=BLOCK_C, HAS_TERM=has_term, **_LAUNCH,
        )
        ctx.save_for_backward(hres, hpost_row, h, y)
        ctx.shape = (B, S, N, C)
        ctx.has_term = has_term
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        hres, hpost_row, h, y = ctx.saved_tensors
        B, S, N, C = ctx.shape
        has_term = ctx.has_term
        grad_out = grad_out.contiguous()
        grad_h = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
        grad_y = torch.empty(B, S, C, device=h.device, dtype=h.dtype)
        grad_hres = torch.empty(B, S, N, N, device=h.device, dtype=torch.float32)
        grad_hpost = torch.empty(B, S, N, device=h.device, dtype=torch.float32)
        grad_term = torch.empty(B, S, C, device=h.device, dtype=h.dtype) if has_term else grad_y
        BLOCK_C = _next_pow2(C)
        _hc_post_bwd_kernel[(B * S,)](
            grad_out, hres, hpost_row, h, y,
            grad_h, grad_y, grad_hres, grad_hpost, grad_term,
            N=N, C=C, BLOCK_C=BLOCK_C, HAS_TERM=has_term, **_LAUNCH,
        )
        gterm_ret = grad_term.to(y.dtype) if has_term else None
        return (grad_hres.to(hres.dtype), grad_hpost.to(hpost_row.dtype),
                grad_h, grad_y.to(y.dtype), gterm_ret)


# ===========================================================================
# Dynamo fences — the Triton autograd Functions are opaque to Dynamo (tracing
# INTO their Triton IR mis-launches the kernel / feeds fp64 to tl.dot). These
# @torch.compiler.disable dispatchers force a graph break AT the kernel so the
# surrounding model still compiles with kernels ON. The reference branches in
# the public wrappers above stay OUTSIDE the fence → inductor fuses them (the
# kernels-OFF compile path the d=256 seed relies on). No effect in eager runs.
# ===========================================================================

@torch.compiler.disable
def _hc_pre_dispatch(h: Tensor, hpre_cm: Tensor) -> Tensor:
    return _FusedHCPre.apply(h, hpre_cm)


@torch.compiler.disable
def _hc_post_dispatch(hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor,
                      term: Tensor | None) -> Tensor:
    return _FusedHCPost.apply(hres, hpost_row, h, y, term)


@torch.compiler.disable
def _hc_pre_map_dispatch(h: Tensor, proj_w: Tensor, proj_b: Tensor,
                         tau: float, alpha: float, iters: int, eps: float,
                         N: int) -> tuple[Tensor, Tensor, Tensor]:
    if N == 4:
        # tuned hand-unrolled 4×4 scalar path (production default for n=4).
        return _FusedHCPreMap.apply(h, proj_w, proj_b, tau, alpha, iters, eps)
    # n-generic tile path (n=2 and any other power-of-2 stream count).
    return _FusedHCPreMapGeneric.apply(h, proj_w, proj_b, tau, alpha, iters, eps)


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
        return hc_pre_reference(h, hpre_cm)  # traceable: inductor fuses it
    return _hc_pre_dispatch(h, hpre_cm)


def hc_post(hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor,
            term: Tensor | None = None) -> Tensor:
    """Fused stream-mix + scatter + add: x_mix + x_post (+ optional folded inject term).

    Args:
        hres:      [B, S, n, n] manifold stream mixer.
        hpost_row: [B, S, n]    row-sum of Hpost.
        h:         [B, S, n, C] n-stream carrier.
        y:         [B, S, C]    sublayer output.
        term:      [B, S, C] | None — carrier-engine: the NEXT layer's single-stream
                   inject, broadcast-added to every output stream (folds a separate
                   _apply_injection carrier read+write into this POST write).

    Returns:
        out: [B, S, n, C] updated carrier.
    """
    from morph.kernels.triton._eager_flag import force_eager, hc_force_eager
    if force_eager() or hc_force_eager() or not TRITON_AVAILABLE or not h.is_cuda:
        return hc_post_reference(hres, hpost_row, h, y, term)  # traceable
    return _hc_post_dispatch(hres, hpost_row, h, y, term)


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
    N = h.shape[2]
    # The kernels implement the EXACT closed-form Cayley only for n∈{2,4} (n=4 scalar path;
    # n=2 closed 2×2 in the generic tile path). Any other n falls back to the exact eager
    # reference (n=4 closed form / n≠4 solve) — no silent divergent iteration anywhere.
    supported_n = N == 2 or N == 4
    if (force_eager() or hc_force_eager() or not TRITON_AVAILABLE or not h.is_cuda
            or int(iters) != 3 or not supported_n):
        return hc_pre_map_reference(h, proj_w, proj_b, tau, alpha, iters, eps)  # traceable
    return _hc_pre_map_dispatch(h, proj_w, proj_b, tau, alpha, iters, eps, N)


# ===========================================================================
# Pure-PyTorch references (the spec)
# ===========================================================================

def _cayley_ref(A: Tensor, iters: int, alpha: float) -> Tensor:
    """EXACT orthogonal Cayley (matches morph.model.hyper_connections.cayley_orthogonal).

    n=4: Cayley–Hamilton closed form. n≠4: true Cayley via solve. ``iters`` is retained
    for API compatibility but ignored (the old divergent fixed-point iteration is gone).
    """
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    B = (alpha * 0.5) * (A - A.transpose(-1, -2))                    # skew so(n)
    B2 = B @ B
    p = 0.5 * (B * B).sum(dim=(-1, -2))
    if n == 4:
        Pf = (B[..., 0, 1] * B[..., 2, 3]
              - B[..., 0, 2] * B[..., 1, 3]
              + B[..., 0, 3] * B[..., 1, 2])
        q = Pf * Pf
        num = (I + 2.0 * B + B2) @ ((1.0 + p)[..., None, None] * I + B2)
        return num / (1.0 + p + q)[..., None, None]
    Yt = torch.linalg.solve((I - B).transpose(-1, -2), (I + B).transpose(-1, -2))
    return Yt.transpose(-1, -2)


def hc_pre_map_reference(
    h: Tensor, proj_w: Tensor, proj_b: Tensor,
    tau: float, alpha: float, iters: int, eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch spec for the fused full PRE mapping (matches HyperConnectionResidual)."""
    B, S, n, C = h.shape
    x_flat = h.reshape(B, S, n * C)
    rms = x_flat.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    # bias-under-rms fix: (x·Wᵀ)/rms + b, NOT (x·Wᵀ + b)/rms.
    wx = F.linear(x_flat.float(), proj_w.float())
    raw = (wx / rms + proj_b.float()).reshape(B, S, 3, n, n)
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


def hc_post_reference(hres: Tensor, hpost_row: Tensor, h: Tensor, y: Tensor,
                      term: Tensor | None = None) -> Tensor:
    x_mix = torch.einsum("bsij,bsjc->bsic", hres.to(h.dtype), h)
    x_post = hpost_row.to(h.dtype).unsqueeze(-1) * y.unsqueeze(2)
    out = x_mix + x_post
    if term is not None:
        out = out + term.to(h.dtype).unsqueeze(2)   # broadcast over n streams
    return out


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
