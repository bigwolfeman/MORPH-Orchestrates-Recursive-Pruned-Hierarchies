"""Row-batched decode kernels for one LXTUL block: every weight tile read ONCE per event.

The LXTUL fused decoder (``morph/inference/lx_fused_decode.py``, ``layers="fused"``) runs
one MORPHBlock application for all the query rows of an event (the B token rows, or the
cells of a slot bucket) in eight launches; design and measurements:
``.agents/notes/proposed/feature/2026-10-10-lx-decode-megakernel.md``.

What changes against the plain engine's kernels (``fused_decode_step.py``), each measured:

* The engine recomputes the CCA projection of the conv's 6 history rows at every step; here a
  position's projection (its "latent" row: ``q_lat | k_lat | W_v_prev x``, ``LQ + LK + VH``
  floats) is computed once and kept in a per-site latent cache.
* The grouped conv weight is read TAP-major (``conv_taps_major``): the engine's
  ``[C, D, KC]`` tiles stride the input channel by KC, and that gather was 6.5 of the
  site front-end's 8.5 us (``lat_post`` 3.5 us at one row).
* Fewer phases: the HC-projection GEMV rides on the post kernel that produced the carrier
  (partial sums per channel tile), the sublayer norm on the pre-map, SwiGLU on gate_up.
* The grid is (weight tile, row block). ``lat_gemm`` and ``gu_swiglu`` (from 4 rows) and
  ``up_post`` (from 16 rows) take 4 rows per CTA with one explicit 2D accumulator per row,
  so each weight tile loads (and decodes) once for 4 rows and every row's sum is the
  one-row sum bit for bit (``_rb`` has the measurements); the other kernels take one row
  per CTA, the engine's schedule, where a row's weight tiles come from L2 after the first
  row. A version that looped ALL rows inside the CTA with 3D [rows, out, k] accumulators
  measured 3-5x SLOWER at 4-32 rows (layout conversions through shared memory, and far
  fewer CTAs in flight).

One block, in order (rows ``r < R``; ``N`` = 4 Hyper-Connection streams; ``C`` = d_model):

1. ``hc_pre``      the HC mapping from the projection's partial sums (pre/post softmaxes,
                   exact Cayley), ``x_bar``, the sublayer RMSNorm -> ``xn`` [R, C]
2. ``lat_gemm``    the CCA projection ``Wqkv x`` of the CURRENT row, K-split partials
3. ``lat_post``    partial sums + the two stacked convs over 6 cached latent rows and the
                   current one + qk-mean + RMS + temp + RoPE + value assembly -> q, K/V
                   store, the raw residual, the gate hidden; writes the current latent row
4. ``lx_attn``     (``lx_decode_attn.py``, unchanged)
5. ``up_post``     ``W_up``, the HC post, and the MLP side's HC-projection partials
6. ``hc_pre``      MLP side
7. ``gu_swiglu``   ternary gate_up, gate row o and up row o in one CTA -> ``silu(g) * u``
8. ``dn_post``     ternary down, the HC post with the next layer's injection term, and the
                   next layer's attention HC-projection partials

A group entry (no producing post kernel) gets its partials from ``hc_partials``.

History index encoding (``lat_post``): ``idx >= 0`` a row of the latent table; ``-1`` the
segment's zero row (the projections and convs have no bias, so a zero latent is exactly the
latent of the zero input the segment cut gives); ``idx <= -2`` row ``-2 - idx`` of THIS
event (its K-split partial sums, read the same way the row reads its own).

Numerics: fp32 FMA throughout, no ``tl.dot`` (fp32 ``tl.dot`` runs at TF32 by default).
Deviation from the engine kernels = reduction order only (the GEMV trees, the order the
K-split and channel-tile partials are summed in, fixed per launch). Weights may be fp32 or
bf16 copies (loaded and widened to fp32); the MLP weights are the 2-bit strip-packed
ternary codes of ``fused_decode_step.ternary_pack`` with the tensor scale applied after
the sum (the engine's rule).

Every launch has a fixed grid and reads the row count ``R`` as a host int: graph-capturable
(one graph per event shape, as the decoder already captures).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from morph.kernels.triton.fused_hyper_connection import _cayley_tile_YM

__all__ = ["HC_NO", "hc_partials", "hc_pre", "lat_gemm", "lat_post", "up_post",
           "gu_swiglu", "dn_post", "PART_TILE", "conv_taps_major"]

HC_NO = 48              # 3 * N * N outputs of the HC mapping projection at N = 4
_NOP = 64               # HC_NO padded to a power of two (masked lanes never load)
PART_TILE = 8           # channels per CTA of the partial-producing kernels


def _p2(n: int) -> int:
    return triton.next_power_of_2(int(n))


# ─────────────────────────────────────────────────────────────────────────────
# HC-projection partial sums over one channel tile (shared by 3 kernels)
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _hc_part_row(WN, RAWP, r, t, o0, o1, o2, o3, N: tl.constexpr, C: tl.constexpr,
                 BC: tl.constexpr, NT: tl.constexpr, NO: tl.constexpr, NOP: tl.constexpr):
    """RAWP[r, t, :] = sum over the 4 streams i and the tile's channels c of
    ``WN[o, i*C + c] * o_i[c]`` (o0..o3 [BC] the carrier's 4 streams on channel tile t)."""
    o = tl.arange(0, NOP)
    c = t * BC + tl.arange(0, BC)
    mo = (o < NO)[:, None]
    wrow = WN + o[:, None] * (N * C) + c[None, :]
    p = tl.load(wrow, mask=mo, other=0.0).to(tl.float32) * o0[None, :]
    p += tl.load(wrow + C, mask=mo, other=0.0).to(tl.float32) * o1[None, :]
    p += tl.load(wrow + 2 * C, mask=mo, other=0.0).to(tl.float32) * o2[None, :]
    p += tl.load(wrow + 3 * C, mask=mo, other=0.0).to(tl.float32) * o3[None, :]
    tl.store(RAWP + (r * NT + t) * NO + o, tl.sum(p, 1), mask=o < NO)


@triton.jit
def _hc_partials_kernel(H, WN, RAWP, N: tl.constexpr, C: tl.constexpr,
                        BC: tl.constexpr, NT: tl.constexpr, NO: tl.constexpr,
                        NOP: tl.constexpr):
    t = tl.program_id(0)
    r = tl.program_id(1)
    c = t * BC + tl.arange(0, BC)
    hb = H + r * (N * C) + c
    _hc_part_row(WN, RAWP, r, t, tl.load(hb), tl.load(hb + C), tl.load(hb + 2 * C),
                 tl.load(hb + 3 * C), N, C, BC, NT, NO, NOP)


def hc_partials(h: Tensor, w_proj: Tensor, out: Tensor | None = None) -> Tensor:
    """``[R, C // PART_TILE, 48]`` partial sums of the HC mapping projection (bias-free,
    pre-rms) of the carrier ``h`` [R, N, C] (contiguous fp32); ``w_proj`` [48, N*C] (fp32 or
    bf16, contiguous). ``hc_pre`` sums them over the tiles."""
    R, N, C = h.shape
    assert N == 4 and w_proj.shape == (HC_NO, N * C) and w_proj.is_contiguous()
    assert h.is_contiguous() and h.dtype == torch.float32 and C % PART_TILE == 0
    NT = C // PART_TILE
    if out is None:
        out = torch.empty(R, NT, HC_NO, device=h.device, dtype=torch.float32)
    _hc_partials_kernel[(NT, R)](h, w_proj, out, N=N, C=C, BC=PART_TILE, NT=NT, NO=HC_NO,
                                 NOP=_NOP, num_stages=1, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1 / 6. hc_pre — HC mapping from partials + x_bar + sublayer RMSNorm
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _hc_pre_kernel(H, RAWP, PB, NW, XN, HRES, HPOST,
                   TAU: tl.constexpr, ALPHA: tl.constexpr, EPS: tl.constexpr, neps,
                   N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr,
                   NT: tl.constexpr, BNT: tl.constexpr, NO: tl.constexpr):
    """One CTA per row. The mapping math is ``_hc_premap_fwd_kernel_g``'s (bias outside the
    rms divide, softmax over j for pre, over i for post, exact Cayley for res); the
    projection comes as NT channel-tile partials, summed in tile order by one tl.sum."""
    r = tl.program_id(0)
    rr = tl.arange(0, N)
    NN: tl.constexpr = N * N
    ij = rr[:, None] * N + rr[None, :]
    c = tl.arange(0, BLOCK_C)
    cmask = c < C
    hh = tl.load(H + r * (N * C) + rr[:, None] * C + c[None, :], mask=cmask[None, :],
                 other=0.0)
    ssq = tl.sum(hh * hh)
    inv_rms = 1.0 / tl.sqrt(ssq / (N * C) + EPS)
    inv_tau = 1.0 / TAU
    tt = tl.arange(0, BNT)
    tm = (tt < NT)[:, None]
    pb = RAWP + r * (NT * NO) + tt[:, None] * NO
    raw_pre = tl.sum(tl.load(pb + 0 * NN + tl.arange(0, NN)[None, :], mask=tm, other=0.0), 0)
    raw_post = tl.sum(tl.load(pb + 1 * NN + tl.arange(0, NN)[None, :], mask=tm, other=0.0), 0)
    raw_res = tl.sum(tl.load(pb + 2 * NN + tl.arange(0, NN)[None, :], mask=tm, other=0.0), 0)
    b_pre = tl.load(PB + 0 * NN + ij)
    b_post = tl.load(PB + 1 * NN + ij)
    b_res = tl.load(PB + 2 * NN + ij)
    pre_s = (tl.reshape(raw_pre, (N, N)) * inv_rms + b_pre) * inv_tau
    post_s = (tl.reshape(raw_post, (N, N)) * inv_rms + b_post) * inv_tau
    res_s = tl.reshape(raw_res, (N, N)) * inv_rms + b_res
    pm = tl.max(pre_s, axis=1)[:, None]
    pe = tl.exp(pre_s - pm)
    Hpre = pe / tl.sum(pe, axis=1)[:, None]
    Hpre_cm = tl.sum(Hpre, axis=0) / N
    qm = tl.max(post_s, axis=0)[None, :]
    qe = tl.exp(post_s - qm)
    Hpost = qe / tl.sum(qe, axis=0)[None, :]
    Hpost_row = tl.sum(Hpost, axis=1)
    Ieye = (rr[:, None] == rr[None, :]).to(tl.float32)
    Bm = (ALPHA * 0.5) * (res_s - tl.trans(res_s))
    Y, _M = _cayley_tile_YM(Bm, Ieye, N)
    x_bar = tl.sum(Hpre_cm[:, None] * hh, axis=0)
    rn = 1.0 / tl.sqrt(tl.sum(x_bar * x_bar, axis=0) / C + neps)
    nw = tl.load(NW + c, mask=cmask, other=0.0)
    tl.store(XN + r * C + c, (x_bar * rn) * nw, mask=cmask)
    tl.store(HRES + r * NN + ij, Y)
    tl.store(HPOST + r * N + rr, Hpost_row)


def hc_pre(h: Tensor, rawp: Tensor, bias: Tensor, mod, norm_w: Tensor, norm_eps: float,
           xn: Tensor, hres: Tensor, hpost: Tensor) -> None:
    """``h`` [R, N, C] fp32 contiguous; ``rawp`` [R, NT, 48] (``hc_partials`` / ``up_post`` /
    ``dn_post``); ``bias`` [48] fp32; ``mod`` the HyperConnectionResidual (tau, alpha,
    eps); writes ``xn`` [R, C] = RMSNorm(x_bar) * norm_w, ``hres`` [R, 16], ``hpost`` [R, 4]."""
    R, N, C = h.shape
    NT = rawp.shape[1]
    assert N == 4 and h.is_contiguous() and rawp.shape == (R, NT, HC_NO) and rawp.is_contiguous()
    assert xn.shape == (R, C) and xn.is_contiguous() and hres.shape == (R, N * N)
    assert hpost.shape == (R, N) and bias.numel() == HC_NO and bias.dtype == torch.float32
    # one warp: 2.2-2.3 us at 1-16 rows against 2.9-3.1 (4 warps) and 3.4-3.5 (8 warps)
    # (/home/wolfe/morph-scratch/megadec/mb_layer.py, in-graph, the model's widths)
    _hc_pre_kernel[(R,)](h, rawp, bias, norm_w, xn, hres, hpost,
                         TAU=float(mod.tau), ALPHA=float(mod.cayley_alpha),
                         EPS=float(mod.eps), neps=float(norm_eps), N=N, C=C,
                         BLOCK_C=_p2(C), NT=NT, BNT=_p2(NT), NO=HC_NO,
                         num_stages=1, num_warps=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. lat_gemm — the CCA projection of the current row, all rows, K-split partials
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _lat_gemm_kernel(X, W, PART, R, O: tl.constexpr, KDIM: tl.constexpr, KS: tl.constexpr,
                     BK: tl.constexpr, BO: tl.constexpr, RB: tl.constexpr):
    """PART[r, s, o] = sum_{k in slice s} W[o, k] * X[r, k]; CTA = (tile of BO outputs,
    slice s, rows RB*g .. RB*g + RB-1). The products accumulate elementwise in [BO, BK]
    over the slice's chunks and reduce once (the engine's ``_front_gemm_kernel`` schedule);
    with RB = 4 each weight tile loads once for 4 rows, one accumulator per row (each row's
    sum is the RB = 1 sum, bit for bit)."""
    pid = tl.program_id(0)
    r = tl.program_id(1) * RB
    t = pid // KS
    s = pid % KS
    offs_o = t * BO + tl.arange(0, BO)
    KCH: tl.constexpr = KDIM // KS
    acc = tl.zeros((BO, BK), tl.float32)
    acc1 = tl.zeros((BO, BK), tl.float32)
    acc2 = tl.zeros((BO, BK), tl.float32)
    acc3 = tl.zeros((BO, BK), tl.float32)
    for k0 in range(s * KCH, (s + 1) * KCH, BK):
        kk = k0 + tl.arange(0, BK)
        w = tl.load(W + offs_o[:, None] * KDIM + kk[None, :]).to(tl.float32)
        acc += w * tl.load(X + r * KDIM + kk)[None, :]
        if RB == 4:
            acc1 += w * tl.load(X + (r + 1) * KDIM + kk, mask=r + 1 < R, other=0.0)[None, :]
            acc2 += w * tl.load(X + (r + 2) * KDIM + kk, mask=r + 2 < R, other=0.0)[None, :]
            acc3 += w * tl.load(X + (r + 3) * KDIM + kk, mask=r + 3 < R, other=0.0)[None, :]
    tl.store(PART + r * (KS * O) + s * O + offs_o, tl.sum(acc, 1))
    if RB == 4:
        mo = offs_o < O
        tl.store(PART + (r + 1) * (KS * O) + s * O + offs_o, tl.sum(acc1, 1), mask=mo & (r + 1 < R))
        tl.store(PART + (r + 2) * (KS * O) + s * O + offs_o, tl.sum(acc2, 1), mask=mo & (r + 2 < R))
        tl.store(PART + (r + 3) * (KS * O) + s * O + offs_o, tl.sum(acc3, 1), mask=mo & (r + 3 < R))


def lat_gemm(x: Tensor, w: Tensor, part: Tensor) -> None:
    """``x`` [R, C] fp32 contiguous (``xn``); ``w`` [O, C] (Wqkv, fp32 or bf16, contiguous);
    ``part`` [R, KS, O] fp32 (KS from its shape; C % (KS * 32) == 0, O % 32 == 0)."""
    R, KDIM = x.shape
    O = w.shape[0]
    KS = part.shape[1]
    assert w.shape == (O, KDIM) and w.is_contiguous() and x.is_contiguous()
    assert part.shape == (R, KS, O) and part.is_contiguous() and x.dtype == torch.float32
    BO = 32
    assert O % BO == 0 and KDIM % KS == 0
    BK = 64 if (KDIM // KS) % 64 == 0 else 32
    assert (KDIM // KS) % BK == 0
    RB = _rb(R)
    _lat_gemm_kernel[((O // BO) * KS, triton.cdiv(R, RB))](x, w, part, R, O=O, KDIM=KDIM,
                                                           KS=KS, BK=BK, BO=BO, RB=RB,
                                                           num_stages=2, num_warps=4)


def _rb(R: int, from_rows: int = 4) -> int:
    """Rows per CTA (RB) of lat_gemm, gu_swiglu (from 4 rows) and up_post (from 16 rows):
    with RB = 4 each weight tile loads (and, for the 2-bit codes, decodes) once for 4
    rows, one accumulator per row, so every row's sum is the RB = 1 sum bit for bit.
    Measured 2026-10-10, in-graph, the model's widths, RB = 1 -> RB = 4 in us
    (/home/wolfe/morph-scratch/megadec/rb/check_rb.py, outputs bit-equal at R = 4, 5, 7,
    8, 16, 32):
        rows        4             8             16            32
        lat_gemm    4.00 -> 2.27  7.76 -> 3.28  16.0 -> 5.13  28.0 -> 9.49
        gu_swiglu   4.98 -> 3.69  8.12 -> 6.57  17.5 -> 8.91  30.1 -> 16.6
        up_post     3.48 -> 6.20  5.84 -> 6.56  10.9 -> 8.41  19.9 -> 12.7
    up_post's CTA tile is 8 channels x 512 inputs, so below 16 rows its cost is the rows'
    inputs and HC post, which RB = 4 serialises. 2 rows per CTA at R = 2 was slower on
    gu_swiglu (3.49 -> 3.91, rb/mb_gu_rb.py)."""
    return 4 if R >= from_rows else 1


def conv_taps_major(wgp: Tensor) -> Tensor:
    """``[CL, D, KC]`` grouped conv weight (``conv_q_gp.weight`` / ``conv_k_gp.weight``) ->
    the tap-major ``[KC, CL, D]`` copy ``lat_post`` reads. Same values, new layout."""
    return wgp.permute(2, 0, 1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# 3. lat_post — conv over cached latents + prologue + K/V store + latent write
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _psum1(PB, col, offs, OP: tl.constexpr, KS: tl.constexpr):
    """Sum over the KS k-slices of one row's partials ``PB`` [KS, OP] at columns col+offs,
    in slice order (every reader of a row uses this one order)."""
    a = tl.load(PB + col + offs)
    for s in tl.static_range(1, KS):
        a += tl.load(PB + s * OP + col + offs)
    return a


@triton.jit
def _lrow(LAT, PART, idx, col, offs, slat, KS: tl.constexpr, OP: tl.constexpr):
    """[D] latent values at column col.. of a history row: idx >= 0 the table row, -1 zero,
    <= -2 event row (-2 - idx) as its K-split partial sum (``_psum1``, the order the row's
    own CTA sums it in). One runtime branch per row: only the source in use is loaded."""
    if idx >= 0:
        v = tl.load(LAT + idx.to(tl.int64) * slat + col + offs)
    elif idx == -1:
        v = tl.zeros_like(offs).to(tl.float32)
    else:
        v = _psum1(PART + (-2 - idx).to(tl.int64) * (KS * OP), col, offs, OP, KS)
    return v


@triton.jit
def _conv7t(a0, a1, a2, a3, a4, a5, a6, WDW, WGPT, c0, offs,
            D: tl.constexpr, KC: tl.constexpr, CL: tl.constexpr):
    """The two stacked causal convs at the last position (``fused_decode_step._conv7``'s
    math) on 7 rows of D channels of one head group starting at flat channel c0:
    ``y_p[i] = sum_j a_{p+j}[i] wdw[c0+i, j]`` (same order as ``_conv7``), then
    ``out[c] = sum_p sum_i wgp[c0+c, i, p] y_p[i]``. ``WGPT`` is the grouped weight laid out
    TAP-major, ``[KC, CL, D]`` (``wgp.permute(2, 0, 1)``): each tap's [D, D] tile is a
    contiguous, coalesced load (``wgp``'s own layout strides the input channel by KC). The
    taps accumulate elementwise and reduce over i once."""
    wd = WDW + (c0 + offs) * KC
    wd0 = tl.load(wd + 0)
    wd1 = tl.load(wd + 1)
    wd2 = tl.load(wd + 2)
    wd3 = tl.load(wd + 3)
    y0 = a0 * wd0 + a1 * wd1 + a2 * wd2 + a3 * wd3
    y1 = a1 * wd0 + a2 * wd1 + a3 * wd2 + a4 * wd3
    y2 = a2 * wd0 + a3 * wd1 + a4 * wd2 + a5 * wd3
    y3 = a3 * wd0 + a4 * wd1 + a5 * wd2 + a6 * wd3
    tile = (c0 + offs)[:, None] * D + offs[None, :]
    acc = tl.load(WGPT + tile) * y0[None, :]
    acc += tl.load(WGPT + CL * D + tile) * y1[None, :]
    acc += tl.load(WGPT + 2 * CL * D + tile) * y2[None, :]
    acc += tl.load(WGPT + 3 * CL * D + tile) * y3[None, :]
    return tl.sum(acc, 1)


@triton.jit
def _lat_post_kernel(
    PART, LAT, IDX, COS, SIN, WDWQ, WGPQ, WDWK, WGPK, QNW, KNW, ETEMP,
    Q, WK, WV, XRES, GATEH, POSP, KVROW,
    eps,
    H: tl.constexpr, HKV: tl.constexpr, NREP: tl.constexpr,
    D: tl.constexpr, HD2: tl.constexpr,
    LQ: tl.constexpr, LK: tl.constexpr, VH: tl.constexpr, GH: tl.constexpr,
    KC: tl.constexpr, KS: tl.constexpr, OP: tl.constexpr, NU: tl.constexpr,
    HAS_KVROW: tl.constexpr, WWIN: tl.constexpr,
    slat, sq_b, sq_h, sk_b, sk_h, sk_w, sxr_b, sgh_b, scos_b,
):
    """``_front_post_kernel`` (fused_decode_step.py) with the 7 conv rows read through
    ``_lrow`` instead of 7 recomputed projections. IDX [B', 8] int32: entries 0..5 the
    history rows of positions p-6..p-1 (encoding in the module docstring), 7 the latent
    table row this query's own latents go to (-1: none); the current row is event row b.
    Unit u < H: one query head; u < H + HKV: one kv group; then the gate-hidden chunks."""
    pid = tl.program_id(0)
    b = pid // NU
    u = pid % NU
    offs = tl.arange(0, D)
    offs_h = tl.arange(0, HD2)
    pb = PART + b * (KS * OP)                           # this query's own partials
    ib = IDX + b * 8
    i0 = tl.load(ib + 0); i1 = tl.load(ib + 1); i2 = tl.load(ib + 2)
    i3 = tl.load(ib + 3); i4 = tl.load(ib + 4); i5 = tl.load(ib + 5)
    wl = tl.load(ib + 7)
    wlm = (offs >= 0) & (wl >= 0)
    lw = LAT + tl.maximum(wl, 0).to(tl.int64) * slat

    if u < H:
        g = u // NREP
        col = u * D
        a0 = _lrow(LAT, PART, i0, col, offs, slat, KS, OP)
        a1 = _lrow(LAT, PART, i1, col, offs, slat, KS, OP)
        a2 = _lrow(LAT, PART, i2, col, offs, slat, KS, OP)
        a3 = _lrow(LAT, PART, i3, col, offs, slat, KS, OP)
        a4 = _lrow(LAT, PART, i4, col, offs, slat, KS, OP)
        a5 = _lrow(LAT, PART, i5, col, offs, slat, KS, OP)
        a6 = _psum1(pb, col, offs, OP, KS)
        k6 = _psum1(pb, LQ + g * D, offs, OP, KS)
        tl.store(lw + col + offs, a6, mask=wlm)
        qc = _conv7t(a0, a1, a2, a3, a4, a5, a6, WDWQ, WGPQ, col, offs, D, KC, LQ)
        qv = qc + (a6 + k6) * 0.5
        rq = 1.0 / tl.sqrt(tl.sum(qv * qv, 0) / D + eps)
        qv = (qv * rq) * tl.load(QNW + offs)
        q2 = tl.trans(tl.reshape(qv, (2, HD2)))
        q_lo, q_hi = tl.split(q2)
        cos_lo = tl.load(COS + b * scos_b + offs_h)
        cos_hi = tl.load(COS + b * scos_b + HD2 + offs_h)
        sin_lo = tl.load(SIN + b * scos_b + offs_h)
        sin_hi = tl.load(SIN + b * scos_b + HD2 + offs_h)
        tl.store(Q + b * sq_b + u * sq_h + offs_h, q_lo * cos_lo - q_hi * sin_lo)
        tl.store(Q + b * sq_b + u * sq_h + HD2 + offs_h, q_hi * cos_hi + q_lo * sin_hi)
        tl.store(XRES + b * sxr_b + u * D + offs, a6)
    elif u < H + HKV:
        g = u - H
        col = LQ + g * D
        a0 = _lrow(LAT, PART, i0, col, offs, slat, KS, OP)
        a1 = _lrow(LAT, PART, i1, col, offs, slat, KS, OP)
        a2 = _lrow(LAT, PART, i2, col, offs, slat, KS, OP)
        a3 = _lrow(LAT, PART, i3, col, offs, slat, KS, OP)
        a4 = _lrow(LAT, PART, i4, col, offs, slat, KS, OP)
        a5 = _lrow(LAT, PART, i5, col, offs, slat, KS, OP)
        a6 = _psum1(pb, col, offs, OP, KS)
        tl.store(lw + col + offs, a6, mask=wlm)
        if g < VH // D:
            # this position's W_v_prev x: the NEXT position's value shift reads it
            vp6 = _psum1(pb, LQ + LK + g * D, offs, OP, KS)
            tl.store(lw + LQ + LK + g * D + offs, vp6, mask=wlm)
        kc_ = _conv7t(a0, a1, a2, a3, a4, a5, a6, WDWK, WGPK, g * D, offs, D, KC, LK)
        qkm = (_psum1(pb, (g * NREP + 0) * D, offs, OP, KS) + a6) * 0.5
        for rr in tl.static_range(1, NREP):
            qkm += (_psum1(pb, (g * NREP + rr) * D, offs, OP, KS) + a6) * 0.5
        kv_ = kc_ + qkm / NREP
        rk = 1.0 / tl.sqrt(tl.sum(kv_ * kv_, 0) / D + eps)
        kv_ = ((kv_ * rk) * tl.load(KNW + offs)) * tl.load(ETEMP + g)
        if g < HKV // 2:
            av = _psum1(pb, LQ + LK + VH + g * D, offs, OP, KS)          # v_curr, this row
        else:
            av = _lrow(LAT, PART, i5, LQ + LK + (g - HKV // 2) * D, offs, slat, KS, OP)
        k2 = tl.trans(tl.reshape(kv_, (2, HD2)))
        k_lo, k_hi = tl.split(k2)
        cos_lo = tl.load(COS + b * scos_b + offs_h)
        cos_hi = tl.load(COS + b * scos_b + HD2 + offs_h)
        sin_lo = tl.load(SIN + b * scos_b + offs_h)
        sin_hi = tl.load(SIN + b * scos_b + HD2 + offs_h)
        ky_lo = k_lo * cos_lo - k_hi * sin_lo
        ky_hi = k_hi * cos_hi + k_lo * sin_hi
        v2 = tl.trans(tl.reshape(av, (2, HD2)))
        v_lo, v_hi = tl.split(v2)
        wslot = (tl.load(POSP + b) % WWIN).to(tl.int32)
        if HAS_KVROW:
            kvb = tl.load(KVROW + b).to(tl.int64)
        else:
            kvb = b
        for r in tl.static_range(NREP):
            hh = g * NREP + r
            wkb = WK + kvb * sk_b + hh * sk_h + wslot * sk_w
            tl.store(wkb + offs_h, ky_lo)
            tl.store(wkb + HD2 + offs_h, ky_hi)
            wvb = WV + kvb * sk_b + hh * sk_h + wslot * sk_w
            tl.store(wvb + offs_h, v_lo)
            tl.store(wvb + HD2 + offs_h, v_hi)
    else:
        tc = u - (H + HKV)
        cols = tc * 256 + tl.arange(0, 256)
        mc = cols < GH
        base = LQ + LK + 2 * VH
        a = tl.load(pb + base + cols, mask=mc, other=0.0)
        for s in tl.static_range(1, KS):
            a += tl.load(pb + s * OP + base + cols, mask=mc, other=0.0)
        tl.store(GATEH + b * sgh_b + cols, a, mask=mc)


def lat_post(part: Tensor, lat: Tensor, idx: Tensor, pos_dev: Tensor, cos: Tensor,
             sin: Tensor, wdwq: Tensor, wgpq: Tensor, wdwk: Tensor, wgpk: Tensor,
             qnw: Tensor, knw: Tensor, etemp: Tensor, win_k: Tensor, win_v: Tensor,
             q_out: Tensor, xres: Tensor, gateh: Tensor,
             n_heads: int, n_kv: int, d_head: int, lq: int, lk: int, vh: int, gh: int,
             k_conv: int, eps: float, kv_row: Tensor | None = None) -> None:
    """``part`` [B', KS, O] (``lat_gemm``); ``lat`` [rows, LQ+LK+VH] fp32 (the site's
    latent table, last dim contiguous; any row count, a dummy 1-row tensor for sites with
    none); ``idx`` [B', 8] int32 (entries 0..5 history rows, 7 the write row or -1);
    ``pos_dev`` [B'] int64 K/V entry; ``kv_row`` [B'] the cache row (None: row b'). ``wgpq``
    / ``wgpk`` are the grouped conv weights TAP-MAJOR, ``[KC, LQ, D]`` / ``[KC, LK, D]``
    contiguous (``conv_taps_major``). The rest as ``fused_decode_step.decode_front``."""
    B = part.shape[0]
    KS, OP = part.shape[1], part.shape[2]
    assert part.is_contiguous() and lat.stride(1) == 1 and lat.shape[1] == lq + lk + vh
    assert idx.shape == (B, 8) and idx.dtype == torch.int32 and idx.is_contiguous()
    assert pos_dev.numel() >= B and pos_dev.is_contiguous()
    # the value read of unit g >= n_kv // 2 is latent column lq + lk + (g - n_kv // 2) * D:
    # inside the row iff 2 * vh == n_kv * d_head (Z3, 2026-10-10: with only vh % D == 0 a
    # site with vh = D reads column 894 of an 832-wide latent row)
    assert 2 * vh == n_kv * d_head and lq == n_heads * d_head and lk == n_kv * d_head
    assert OP == lq + lk + 2 * vh + gh and k_conv == 4
    assert wgpq.shape == (k_conv, lq, d_head) and wgpk.shape == (k_conv, lk, d_head)
    assert wgpq.is_contiguous() and wgpk.is_contiguous()
    if kv_row is not None:
        assert kv_row.shape == (B,) and kv_row.is_contiguous()
    NU = n_heads + n_kv + triton.cdiv(gh, 256)
    _lat_post_kernel[(B * NU,)](
        part, lat, idx, cos, sin, wdwq, wgpq, wdwk, wgpk, qnw, knw, etemp,
        q_out, win_k, win_v, xres, gateh, pos_dev,
        kv_row if kv_row is not None else pos_dev, eps,
        H=n_heads, HKV=n_kv, NREP=n_heads // n_kv, D=d_head, HD2=d_head // 2,
        LQ=lq, LK=lk, VH=vh, GH=gh, KC=k_conv, KS=KS, OP=OP, NU=NU,
        HAS_KVROW=kv_row is not None, WWIN=win_k.shape[2],
        slat=lat.stride(0), sq_b=q_out.stride(0), sq_h=q_out.stride(1),
        sk_b=win_k.stride(0), sk_h=win_k.stride(1), sk_w=win_k.stride(2),
        sxr_b=xres.stride(0), sgh_b=gateh.stride(0),
        scos_b=cos.stride(0) if cos.dim() > 1 else 0,
        # w1: 3.5 / 3.7 / 6.7 us at 1 / 4 / 32 query rows (w2 3.9 / 4.1 / 7.9, w4 3.8 / 3.9
        # / 7.4; the engine's _front_post_kernel 8.5 / 8.6 / 33.7, its strided conv tiles):
        # /home/wolfe/morph-scratch/megadec/mb_post.py, in-graph, RTX 5090, D 64
        num_stages=1, num_warps=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. up_post — W_up + HC post + the next HC projection's partials
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _post_stream(HRES, HPOST, r, i: tl.constexpr, h0, h1, h2, h3, y, tv,
                 N: tl.constexpr, HAS_TERM: tl.constexpr):
    """Stream i of the HC post on one channel tile: sum_j Hres[r, i, j] h_j + Hpost_row[r, i]
    y (+ term), in ``_hc_post_fwd_kernel``'s order."""
    hb = HRES + r * (N * N) + i * N
    acc = tl.load(hb + 0) * h0
    acc += tl.load(hb + 1) * h1
    acc += tl.load(hb + 2) * h2
    acc += tl.load(hb + 3) * h3
    acc += tl.load(HPOST + r * N + i) * y
    if HAS_TERM:
        acc += tv
    return acc


@triton.jit
def _post_row(HRES, HPOST, H, OUT, TERM, WN, RAWP, r, t, y, c, N: tl.constexpr,
              C: tl.constexpr, BC: tl.constexpr, NT: tl.constexpr, NO: tl.constexpr,
              NOP: tl.constexpr, HAS_TERM: tl.constexpr, HAS_NEXT: tl.constexpr, sterm):
    """The HC post of row r on channel tile t (4 streams), its store, and (HAS_NEXT) the
    next HC projection's partials of the result."""
    hb = H + r * (N * C) + c
    h0 = tl.load(hb)
    h1 = tl.load(hb + C)
    h2 = tl.load(hb + 2 * C)
    h3 = tl.load(hb + 3 * C)
    tv = y
    if HAS_TERM:
        tv = tl.load(TERM + r * sterm + c)
    o0 = _post_stream(HRES, HPOST, r, 0, h0, h1, h2, h3, y, tv, N, HAS_TERM)
    o1 = _post_stream(HRES, HPOST, r, 1, h0, h1, h2, h3, y, tv, N, HAS_TERM)
    o2 = _post_stream(HRES, HPOST, r, 2, h0, h1, h2, h3, y, tv, N, HAS_TERM)
    o3 = _post_stream(HRES, HPOST, r, 3, h0, h1, h2, h3, y, tv, N, HAS_TERM)
    ob = OUT + r * (N * C) + c
    tl.store(ob, o0)
    tl.store(ob + C, o1)
    tl.store(ob + 2 * C, o2)
    tl.store(ob + 3 * C, o3)
    if HAS_NEXT:
        _hc_part_row(WN, RAWP, r, t, o0, o1, o2, o3, N, C, BC, NT, NO, NOP)


@triton.jit
def _up_post_kernel(X, WUP, HRES, HPOST, H, OUT, TERM, WN, RAWP, R,
                    N: tl.constexpr, C: tl.constexpr, I: tl.constexpr, BK: tl.constexpr,
                    BC: tl.constexpr, NT: tl.constexpr, NO: tl.constexpr, NOP: tl.constexpr,
                    HAS_TERM: tl.constexpr, HAS_NEXT: tl.constexpr, sterm, RB: tl.constexpr):
    """CTA = (channel tile t, rows RB*g ..): y = W_up x per row, then each row's HC post
    (``_post_row``); RB = 4 loads each W_up tile once for 4 rows (``_lat_gemm_kernel``)."""
    t = tl.program_id(0)
    r = tl.program_id(1) * RB
    c = t * BC + tl.arange(0, BC)
    acc = tl.zeros((BC, BK), tl.float32)
    acc1 = tl.zeros((BC, BK), tl.float32)
    acc2 = tl.zeros((BC, BK), tl.float32)
    acc3 = tl.zeros((BC, BK), tl.float32)
    for k0 in range(0, I, BK):
        kk = k0 + tl.arange(0, BK)
        w = tl.load(WUP + c[:, None] * I + kk[None, :]).to(tl.float32)
        acc += w * tl.load(X + r * I + kk)[None, :]
        if RB == 4:
            acc1 += w * tl.load(X + (r + 1) * I + kk, mask=r + 1 < R, other=0.0)[None, :]
            acc2 += w * tl.load(X + (r + 2) * I + kk, mask=r + 2 < R, other=0.0)[None, :]
            acc3 += w * tl.load(X + (r + 3) * I + kk, mask=r + 3 < R, other=0.0)[None, :]
    _post_row(HRES, HPOST, H, OUT, TERM, WN, RAWP, r, t, tl.sum(acc, 1), c, N, C, BC, NT,
              NO, NOP, HAS_TERM, HAS_NEXT, sterm)
    if RB == 4:
        if r + 1 < R:
            _post_row(HRES, HPOST, H, OUT, TERM, WN, RAWP, r + 1, t, tl.sum(acc1, 1), c, N, C,
                      BC, NT, NO, NOP, HAS_TERM, HAS_NEXT, sterm)
        if r + 2 < R:
            _post_row(HRES, HPOST, H, OUT, TERM, WN, RAWP, r + 2, t, tl.sum(acc2, 1), c, N, C,
                      BC, NT, NO, NOP, HAS_TERM, HAS_NEXT, sterm)
        if r + 3 < R:
            _post_row(HRES, HPOST, H, OUT, TERM, WN, RAWP, r + 3, t, tl.sum(acc3, 1), c, N, C,
                      BC, NT, NO, NOP, HAS_TERM, HAS_NEXT, sterm)


def _post_args(h, out, term, w_next, rawp, R, N, C):
    assert N == 4 and h.is_contiguous() and out.shape == h.shape and out.is_contiguous()
    term2 = None
    if term is not None:
        assert term.shape[-1] == C and term.stride(-1) == 1
        term2 = term.reshape(-1, C)
        assert term2.shape[0] == R
    NT = C // PART_TILE
    if w_next is not None:
        assert w_next.shape == (HC_NO, N * C) and rawp.shape == (R, NT, HC_NO)
    return term2, NT


def up_post(x: Tensor, w_up: Tensor, hres: Tensor, hpost: Tensor, h: Tensor,
            out: Tensor, term: Tensor | None, w_next: Tensor | None,
            rawp: Tensor | None) -> None:
    """``out`` [R, N, C] = HC post of ``h`` with ``y = x @ w_up.T`` (``x`` [R, I] the
    attention output, ``w_up`` [C, I]); ``term`` [R, C] (row-contiguous) added to every
    stream; ``w_next`` [48, N*C]: also write ``rawp`` [R, C/PART_TILE, 48], the partials of
    the next HC projection of ``out``."""
    R, N, C = h.shape
    I = x.shape[1]
    assert x.shape == (R, I) and x.is_contiguous() and w_up.shape == (C, I)
    assert w_up.is_contiguous() and hres.shape == (R, N * N) and hpost.shape == (R, N)
    term2, NT = _post_args(h, out, term, w_next, rawp, R, N, C)
    BK = 64 if I % 64 == 0 else 32
    assert I % BK == 0
    RB = _rb(R, 16)
    _up_post_kernel[(NT, triton.cdiv(R, RB))](
        x, w_up, hres, hpost, h, out, term2 if term2 is not None else h,
        w_next if w_next is not None else h, rawp if rawp is not None else h, R,
        N=N, C=C, I=I, BK=BK, BC=PART_TILE, NT=NT, NO=HC_NO, NOP=_NOP,
        HAS_TERM=term is not None, HAS_NEXT=w_next is not None,
        sterm=term2.stride(0) if term2 is not None else 0, RB=RB,
        num_stages=2, num_warps=4)


# ─────────────────────────────────────────────────────────────────────────────
# 7. gu_swiglu — ternary gate_up, all rows, -> silu(g) * u
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _tdec(p32, st: tl.constexpr):
    """2-bit strip code -> fp32 {-1, 0, 1} (the engine's magic-number dequant)."""
    return (((p32 >> (2 * st)) & 3) | 0x4B400000).to(tl.float32, bitcast=True) - 12582913.0


@triton.jit
def _gu_swiglu_kernel(X, W2, SCALE, OUT, R, I: tl.constexpr, FF: tl.constexpr,
                      BO: tl.constexpr, BK: tl.constexpr, RB: tl.constexpr):
    """CTA = (BO gate rows o and the BO up rows FF + o, rows RB*g ..). A code byte carries
    4 strips; each chunk's bytes load and decode once into the 4 strips' weights, for RB
    rows (one gate / up accumulator pair per row: each row's sums are the RB = 1 sums)."""
    t = tl.program_id(0)
    r = tl.program_id(1) * RB
    I4: tl.constexpr = I // 4
    o = t * BO + tl.arange(0, BO)
    ag = tl.zeros((BO, BK), tl.float32)
    au = tl.zeros((BO, BK), tl.float32)
    ag1 = tl.zeros((BO, BK), tl.float32)
    au1 = tl.zeros((BO, BK), tl.float32)
    ag2 = tl.zeros((BO, BK), tl.float32)
    au2 = tl.zeros((BO, BK), tl.float32)
    ag3 = tl.zeros((BO, BK), tl.float32)
    au3 = tl.zeros((BO, BK), tl.float32)
    for j0 in range(0, I4, BK):
        j = j0 + tl.arange(0, BK)
        mj = j < I4
        pg = tl.load(W2 + o[:, None] * I4 + j[None, :], mask=mj[None, :], other=0).to(tl.int32)
        pu = tl.load(W2 + (FF + o)[:, None] * I4 + j[None, :], mask=mj[None, :],
                     other=0).to(tl.int32)
        for st in tl.static_range(4):
            wg = _tdec(pg, st)
            wu = _tdec(pu, st)
            x = tl.load(X + r * I + st * I4 + j, mask=mj, other=0.0)[None, :]
            ag += wg * x
            au += wu * x
            if RB == 4:
                x = tl.load(X + (r + 1) * I + st * I4 + j, mask=mj & (r + 1 < R),
                            other=0.0)[None, :]
                ag1 += wg * x
                au1 += wu * x
                x = tl.load(X + (r + 2) * I + st * I4 + j, mask=mj & (r + 2 < R),
                            other=0.0)[None, :]
                ag2 += wg * x
                au2 += wu * x
                x = tl.load(X + (r + 3) * I + st * I4 + j, mask=mj & (r + 3 < R),
                            other=0.0)[None, :]
                ag3 += wg * x
                au3 += wu * x
    sc = tl.load(SCALE)
    g = tl.sum(ag, 1) * sc
    u = tl.sum(au, 1) * sc
    tl.store(OUT + r * FF + o, g * tl.sigmoid(g) * u)
    if RB == 4:
        mo = o < FF
        g = tl.sum(ag1, 1) * sc
        u = tl.sum(au1, 1) * sc
        tl.store(OUT + (r + 1) * FF + o, g * tl.sigmoid(g) * u, mask=mo & (r + 1 < R))
        g = tl.sum(ag2, 1) * sc
        u = tl.sum(au2, 1) * sc
        tl.store(OUT + (r + 2) * FF + o, g * tl.sigmoid(g) * u, mask=mo & (r + 2 < R))
        g = tl.sum(ag3, 1) * sc
        u = tl.sum(au3, 1) * sc
        tl.store(OUT + (r + 3) * FF + o, g * tl.sigmoid(g) * u, mask=mo & (r + 3 < R))


def gu_swiglu(x: Tensor, codes: Tensor, scale: Tensor, out: Tensor) -> None:
    """``x`` [R, I] fp32 (the normed MLP input); ``codes`` [2*FF, I/4] uint8 (ternary_pack
    of gate_up, gate rows first); ``scale`` [1]; ``out`` [R, FF] = silu(g) * u."""
    R, I = x.shape
    F2, I4 = codes.shape
    FF = F2 // 2
    assert I4 * 4 == I and F2 == 2 * FF and out.shape == (R, FF) and x.is_contiguous()
    BO = 8
    BK = 64
    assert FF % BO == 0
    RB = _rb(R)
    _gu_swiglu_kernel[(FF // BO, triton.cdiv(R, RB))](x, codes, scale, out, R, I=I, FF=FF,
                                                      BO=BO, BK=BK, RB=RB, num_stages=2,
                                                      num_warps=4)


# ─────────────────────────────────────────────────────────────────────────────
# 8. dn_post — ternary down + HC post (+ next term) + next attention partials
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _dn_post_kernel(X, W2, SCALE, HRES, HPOST, H, OUT, TERM, WN, RAWP,
                    N: tl.constexpr, C: tl.constexpr, FF: tl.constexpr, BK: tl.constexpr,
                    BC: tl.constexpr, NT: tl.constexpr, NO: tl.constexpr, NOP: tl.constexpr,
                    HAS_TERM: tl.constexpr, HAS_NEXT: tl.constexpr, sterm):
    t = tl.program_id(0)
    r = tl.program_id(1)
    I4: tl.constexpr = FF // 4
    c = t * BC + tl.arange(0, BC)
    acc = tl.zeros((BC, BK), tl.float32)
    for j0 in range(0, I4, BK):
        j = j0 + tl.arange(0, BK)
        mj = j < I4                          # a masked lane: code byte 0 (-1) times x = 0.0
        p = tl.load(W2 + c[:, None] * I4 + j[None, :], mask=mj[None, :], other=0).to(tl.int32)
        for st in tl.static_range(4):
            acc += _tdec(p, st) * tl.load(X + r * FF + st * I4 + j, mask=mj, other=0.0)[None, :]
    y = tl.sum(acc, 1) * tl.load(SCALE)
    _post_row(HRES, HPOST, H, OUT, TERM, WN, RAWP, r, t, y, c, N, C, BC, NT, NO, NOP,
              HAS_TERM, HAS_NEXT, sterm)


def dn_post(x: Tensor, codes: Tensor, scale: Tensor, hres: Tensor, hpost: Tensor,
            h: Tensor, out: Tensor, term: Tensor | None, w_next: Tensor | None,
            rawp: Tensor | None) -> None:
    """``out`` [R, N, C] = HC post of ``h`` with ``y = gamma * codes(x)`` (``x`` [R, FF] the
    swiglu output, ``codes`` [C, FF/4] the down weight's ternary pack); ``term``, ``w_next``,
    ``rawp`` as ``up_post``."""
    R, N, C = h.shape
    FF = x.shape[1]
    assert codes.shape == (C, FF // 4) and FF % 4 == 0 and x.is_contiguous()
    term2, NT = _post_args(h, out, term, w_next, rawp, R, N, C)
    BK = 64                                  # d_ff 2816: I/4 = 704 = 11 chunks, no tail
    # one row per CTA at every R: the CTA's code tile is 8 x FF/4 bytes, so its cost is
    # the row's FF inputs and the HC post, which 4 rows per CTA only serialise (an RB = 4
    # variant measured 4.96 -> 12.5 us at 4 rows, 22.0 -> 20.5 at 32:
    # /home/wolfe/morph-scratch/megadec/rb/check_rb.py)
    _dn_post_kernel[(NT, R)](
        x, codes, scale, hres, hpost, h, out, term2 if term2 is not None else h,
        w_next if w_next is not None else h, rawp if rawp is not None else h,
        N=N, C=C, FF=FF, BK=BK, BC=PART_TILE, NT=NT, NO=HC_NO, NOP=_NOP,
        HAS_TERM=term is not None, HAS_NEXT=w_next is not None,
        sterm=term2.stride(0) if term2 is not None else 0,
        num_stages=2, num_warps=4)
