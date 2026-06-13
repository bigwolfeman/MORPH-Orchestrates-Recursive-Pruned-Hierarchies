"""Fused single-token decode kernels for the MORPH static decode engine (#245 Phase 3).

The graph-replayed decode step was OVERHEAD-bound: ~5k kernels/token at ~1.5 µs each, with
~45% of GPU time in tiny eager elementwise glue (profile: ignore/profile_static_decode.py).
These kernels collapse the per-site glue:

  decode_attn   — ONE kernel per attention site (after the cuBLAS projections): window branch
                  (q·K over the 128-slot ring, masked softmax, ·V) + compressed branch (HCA
                  dense-over-blocks / CSA gathered top-k, sink logit, validity masking) +
                  sigmoid gate blend + residual-alpha. Replaces ~30 eager kernels.
  rmsnorm_rows  — RMSNorm over contiguous rows (replaces the 6-kernel eager chain).
                  Optional fused per-row-group scale would live here later.
  swiglu_rows   — silu(gate)·up on the packed [rows, 2F] gate_up output (3 kernels → 1).

Numerics: all fp32 (the eval stack is fp32), mirroring the golden eager math op-for-op;
deviations are reduction-tree order only. Gated by ignore/verify_static_decode.py
(greedy token_match 1.0 vs the eager golden decoder).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_LAUNCH = dict(num_stages=1, num_warps=4)

_dummy_cache: dict = {}


def _dummy_idx(dev) -> Tensor:
    """Persistent dummy top-idx for the HCA branch (avoids a per-call fill kernel)."""
    t = _dummy_cache.get(dev)
    if t is None:
        t = torch.zeros(1, 1, dtype=torch.long, device=dev)
        _dummy_cache[dev] = t
    return t


# ─────────────────────────────────────────────────────────────────────────────
# decode_attn — fused window + compressed attention + gate combine (one site)
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _decode_attn_kernel(
    Q, WK, WV, CC, TIDX, GLOG, XRES, SINK, ALPHA, CNT, WMASK, OUT, WG2, POSP,
    scale,
    H: tl.constexpr, D: tl.constexpr, WWIN: tl.constexpr, NB: tl.constexpr,
    IS_CSA: tl.constexpr, GH: tl.constexpr, BG: tl.constexpr,
    sq_b, sq_h,                # Q [B,H,D] (D contiguous)
    sk_b, sk_h, sk_w,          # WK/WV [B,H,WWIN,D]
    sc_b, sc_n,                # CC [B,*,D]
    st_b,                      # TIDX [B,NB]
    sg_b,                      # GLOG [B,H*2] | GATEH [B,GH] when GH>0
    sx_b,                      # XRES [B,H*D]
    so_b,                      # OUT [B,H*D]
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sq_b + h * sq_h + offs_d)                       # [D] fp32

    # ── window branch: masked softmax over the RING, loaded in CHRONOLOGICAL
    # order (index j ↔ position p-WWIN+1+j ↔ ring slot (p+1+j)%WWIN) so every
    # reduction keeps the golden's operand ordering — the ring changes only
    # WHERE a key lives, never the order it enters the sums. WMASK is in
    # chronological j-space (slot j==WWIN-1 is the current position: invalid).
    offs_w = tl.arange(0, WWIN)
    posv = tl.load(POSP)
    slot = ((posv + 1 + offs_w) % WWIN).to(tl.int32)
    kw = tl.load(WK + b * sk_b + h * sk_h + slot[:, None] * sk_w + offs_d[None, :])
    s_win = tl.sum(kw * q[None, :], 1) * scale                          # [WWIN]
    wm = tl.load(WMASK + offs_w)
    s_win = tl.where(wm != 0, s_win, float("-inf"))
    m_w = tl.max(s_win, 0)
    p_w = tl.exp(s_win - m_w)
    vw = tl.load(WV + b * sk_b + h * sk_h + slot[:, None] * sk_w + offs_d[None, :])
    ow = tl.sum(p_w[:, None] * vw, 0) / tl.sum(p_w, 0)                  # [D]

    # ── compressed branch: HCA dense blocks / CSA gathered top-k, with sink ──
    offs_n = tl.arange(0, NB)
    cnt = tl.load(CNT)
    if IS_CSA:
        idx = tl.load(TIDX + b * st_b + offs_n)                         # [NB] int64
        valid = idx < cnt
        safe = tl.minimum(idx, tl.maximum(cnt - 1, 0))
        cc = tl.load(CC + b * sc_b + safe[:, None] * sc_n + offs_d[None, :])
        s_c = tl.sum(cc * q[None, :], 1) * scale
        s_c = tl.where(valid, s_c, float("-inf"))
    else:
        cc = tl.load(CC + b * sc_b + offs_n[:, None] * sc_n + offs_d[None, :])
        s_c = tl.sum(cc * q[None, :], 1) * scale
        s_c = tl.where(offs_n < cnt, s_c, float("-inf"))
    sink = tl.load(SINK + h)
    m_c = tl.maximum(tl.max(s_c, 0), sink)
    p_c = tl.exp(s_c - m_c)
    denom = tl.sum(p_c, 0) + tl.exp(sink - m_c)                         # sink in denominator
    oc = tl.sum(p_c[:, None] * cc, 0) / denom                           # [D]; cnt==0 → 0 exact

    # ── sigmoid gate blend + residual alpha ──
    if GH > 0:
        # gate2 GEMV folded in: GLOG holds the RAW gate hidden [B,GH]; per head we
        # need logits rows 2h, 2h+1 of WG2 [H*2, GH] applied to silu(gate_h). Same
        # padded-256 reduction shape the _gemv_row_kernel used (order-equal).
        offs_g = tl.arange(0, BG)
        mg = offs_g < GH
        ghid = tl.load(GLOG + b * sg_b + offs_g, mask=mg, other=0.0)
        ghid = ghid * tl.sigmoid(ghid)                                  # silu (SILU_X)
        w0 = tl.load(WG2 + (h * 2 + 0) * GH + offs_g, mask=mg, other=0.0).to(tl.float32)
        w1 = tl.load(WG2 + (h * 2 + 1) * GH + offs_g, mask=mg, other=0.0).to(tl.float32)
        g0 = tl.sigmoid(tl.sum(ghid * w0, 0))
        g1 = tl.sigmoid(tl.sum(ghid * w1, 0))
    else:
        g0 = tl.sigmoid(tl.load(GLOG + b * sg_b + h * 2 + 0))
        g1 = tl.sigmoid(tl.load(GLOG + b * sg_b + h * 2 + 1))
    xres = tl.load(XRES + b * sx_b + h * D + offs_d)
    alpha = tl.load(ALPHA + h)
    out = g0 * oc + g1 * ow + alpha * xres
    tl.store(OUT + b * so_b + h * D + offs_d, out)


def decode_attn(q: Tensor, win_k: Tensor, win_v: Tensor, c_comp: Tensor,
                top_idx: Tensor | None, gate_logits: Tensor, x_res: Tensor,
                sink: Tensor, alpha: Tensor, cnt: Tensor, win_mask_i8: Tensor,
                scale: float, w_gate2: Tensor | None = None,
                pos_dev: Tensor | None = None) -> Tensor:
    """Fused per-site decode attention.

    q [B,H,D]; win_k/win_v [B,H,WWIN,D]; c_comp [B,NBLK,D]; top_idx [B,tk] (CSA) or None
    (HCA); gate_logits: raw logits [B,H*2] (sigmoid inside) OR, when w_gate2 [H*2,GH] is
    given, the RAW gate hidden [B,GH] (silu + gate2 GEMV folded in); x_res [B,H*D]
    (=q_lat layout); sink/alpha [H]; cnt [1] int64 visible-block count; win_mask_i8
    [WWIN] int8. Returns [B, H*D] (W_up-ready layout).
    """
    B, H, D = q.shape
    WWIN = win_k.shape[2]
    is_csa = top_idx is not None
    NB = top_idx.shape[1] if is_csa else c_comp.shape[1]
    out = torch.empty(B, H * D, device=q.device, dtype=q.dtype)
    tidx = top_idx if is_csa else _dummy_idx(q.device)
    GH = gate_logits.shape[1] if w_gate2 is not None else 0
    _decode_attn_kernel[(B * H,)](
        q, win_k, win_v, c_comp, tidx, gate_logits, x_res, sink, alpha, cnt,
        win_mask_i8, out, w_gate2 if w_gate2 is not None else gate_logits,
        pos_dev, scale,
        H=H, D=D, WWIN=WWIN, NB=NB, IS_CSA=is_csa,
        GH=GH, BG=triton.next_power_of_2(max(GH, 1)),
        sq_b=q.stride(0), sq_h=q.stride(1),
        sk_b=win_k.stride(0), sk_h=win_k.stride(1), sk_w=win_k.stride(2),
        sc_b=c_comp.stride(0), sc_n=c_comp.stride(1),
        st_b=tidx.stride(0),
        sg_b=gate_logits.stride(0),
        sx_b=x_res.stride(0),
        so_b=out.stride(0),
        num_stages=1, num_warps=8,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# csa_select — fused CSA block scoring + exact top-k selection (2 launches)
# ─────────────────────────────────────────────────────────────────────────────
#
# Replaces bmm + relu + masked_fill + torch.topk (gatherTopK + radixSortKVInPlace +
# index kernels ≈ 6 launches, ~14 µs/site). torch.topk(sorted=True) on CUDA orders by
# (value desc, index asc) — equal values keep gather (index) order through the stable
# radix sort. We reproduce that EXACT order with a single in-register sort over packed
# int64 keys (monotone-float-bits << 32 | reversed index).


@triton.jit
def _csa_scores_kernel(QI, KI, CNT, OUT, DI: tl.constexpr, BN: tl.constexpr,
                       sq_b, sk_b, sk_n, so_b):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, DI)
    q = tl.load(QI + b * sq_b + offs_d)
    k = tl.load(KI + b * sk_b + offs_n[:, None] * sk_n + offs_d[None, :])
    s = tl.sum(k * q[None, :], 1)
    s = tl.maximum(s, 0.0)
    s = tl.where(s == 0.0, 0.0, s)            # canonicalize -0 → +0 (tie-break safety)
    cnt = tl.load(CNT)
    s = tl.where(offs_n < cnt, s, float("-inf"))
    tl.store(OUT + b * so_b + offs_n, s)


@triton.jit
def _csa_select_kernel(S, OUT, NB: tl.constexpr, K: tl.constexpr, ss_b, so_b):
    b = tl.program_id(0)
    offs = tl.arange(0, NB)
    v = tl.load(S + b * ss_b + offs)
    bits = v.to(tl.int32, bitcast=True).to(tl.int64)
    ub = bits & 0xFFFFFFFF                                       # value bits as uint32
    mono = tl.where(ub >= 0x80000000, 0xFFFFFFFF - ub, ub + 0x80000000)
    # bias into the SIGNED int64 range before shifting (mono ≥ 2^31 would set the sign
    # bit after <<32 and sort -inf entries FIRST — the falsified first cut did that).
    key = ((mono - 0x80000000) << 32) | (NB - 1 - offs).to(tl.int64)
    skey = tl.sort(key, descending=True)
    idx = (NB - 1) - (skey & 0xFFFFFFFF)
    tl.store(OUT + b * so_b + offs, idx, mask=offs < K)


def csa_select(q_i: Tensor, k_i: Tensor, cnt: Tensor, scratch: Tensor, k: int) -> Tensor:
    """q_i [B,DI] (any row stride); k_i [B,NB,DI]; cnt [1] int64; scratch [B,NB] fp32.
    Returns top-k indices [B,k] int64 — exact torch.topk(value desc, index asc) order."""
    B, NB, DI = k_i.shape
    out = torch.empty(B, k, device=q_i.device, dtype=torch.long)
    BN = 128                                    # 8 CTAs + w8: 1.67 -> 1.06 µs measured
    _csa_scores_kernel[(NB // BN, B)](
        q_i, k_i, cnt, scratch, DI=DI, BN=BN,
        sq_b=q_i.stride(0), sk_b=k_i.stride(0), sk_n=k_i.stride(1),
        so_b=scratch.stride(0), num_stages=1, num_warps=8)
    _csa_select_kernel[(B,)](scratch, out, NB=NB, K=k, ss_b=scratch.stride(0),
                             so_b=out.stride(0), num_stages=1, num_warps=8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# gla_step — single-token GLA recurrence + GroupNorm + output gate (one launch)
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirrors GatedLinearAttention._chunked at S=1 (incl. the −30 log-gate clamp and the
# e^{b}·e^{-b} decay-to-end roundings) + the GroupNorm(o)·silu(r) readout. GroupNorm
# groups == heads, so each (b,h) CTA normalizes its own dh channels. Updates the fp32
# state IN PLACE (graph-stable address). Replaces ~10 launches per retention site.


@triton.jit
def _gla_step_kernel(P, STATE, GB, GNW, GNB, OUT,
                     H: tl.constexpr, DH: tl.constexpr, eps,
                     sp_b, ss_b, ss_h, so_b):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs = tl.arange(0, DH)
    d = H * DH
    base = P + b * sp_b + h * DH
    q = tl.load(base + offs)
    k = tl.load(base + d + offs)
    v = tl.load(base + 2 * d + offs)
    g = tl.load(base + 3 * d + offs) + tl.load(GB + h * DH + offs)
    r = tl.load(base + 4 * d + offs)

    # log_alpha = logsigmoid(g) (stable form), clamped at −30 like the chunked path.
    la = tl.where(g >= 0.0, -tl.log(1.0 + tl.exp(-g)), g - tl.log(1.0 + tl.exp(g)))
    la = tl.maximum(la, -30.0)
    a = tl.exp(la)
    qd = q * a                                              # q ⊙ e^{b}
    kd = k * tl.exp(-la)                                    # k ⊙ e^{-b}
    score = tl.sum(qd * kd, 0)                              # intra (L=1, causal diag)
    ke = k * (a * tl.exp(-la))                              # k ⊙ e^{b_L}·e^{-b_j}

    st_ptr = STATE + b * ss_b + h * ss_h + offs[:, None] * DH + tl.arange(0, DH)[None, :]
    st = tl.load(st_ptr)                                    # [dk, dv] fp32
    o = tl.sum(qd[:, None] * st, 0) + score * v             # inter + intra
    tl.store(st_ptr, a[:, None] * st + ke[:, None] * v[None, :])

    # GroupNorm over this head's dv channels (+ affine), then swish output gate.
    mean = tl.sum(o, 0) / DH
    var = tl.sum((o - mean) * (o - mean), 0) / DH
    on = (o - mean) / tl.sqrt(var + eps)
    on = on * tl.load(GNW + h * DH + offs) + tl.load(GNB + h * DH + offs)
    out = on * (r * tl.sigmoid(r))
    tl.store(OUT + b * so_b + h * DH + offs, out)


def gla_step(p: Tensor, state: Tensor, gate_bias: Tensor, gn_w: Tensor, gn_b: Tensor,
             eps: float) -> Tensor:
    """p [B,5d] stacked q|k|v|g|r; state [B,H,dh,dh] fp32 (updated IN PLACE).
    Returns gated GroupNorm'd output [B,d] (feed o_proj)."""
    B = p.shape[0]
    Bh, H, dh, _ = state.shape
    d = H * dh
    out = torch.empty(B, d, device=p.device, dtype=p.dtype)
    _gla_step_kernel[(B * H,)](
        p, state, gate_bias, gn_w, gn_b, out, H=H, DH=dh, eps=eps,
        sp_b=p.stride(0), ss_b=state.stride(0), ss_h=state.stride(1),
        so_b=out.stride(0), num_stages=1, num_warps=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# rmsnorm_rows — RMSNorm over contiguous rows
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _rmsnorm_kernel(X, W, OUT, C: tl.constexpr, BC: tl.constexpr, eps, so):
    row = tl.program_id(0)
    offs = tl.arange(0, BC)
    mask = offs < C
    x = tl.load(X + row * C + offs, mask=mask, other=0.0)
    ms = tl.sum(x * x, 0) / C
    r = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + offs, mask=mask, other=0.0)
    tl.store(OUT + row * so + offs, (x * r) * w, mask=mask)


def rmsnorm_rows(x: Tensor, weight: Tensor, eps: float,
                 out: Tensor | None = None, out_row_stride: int | None = None) -> Tensor:
    """RMSNorm over the last dim. x [..., C] CONTIGUOUS, fp32. Mirrors attention.RMSNorm.
    `out` (with `out_row_stride`) lets the caller write straight into a strided slot
    (e.g. the x-history staging slice) and skip a copy kernel."""
    C = x.shape[-1]
    rows = x.numel() // C
    if out is None:
        out = torch.empty_like(x)
        so = C
    else:
        so = out_row_stride if out_row_stride is not None else C
    # w2 measured-best at C=768; wide rows (d=8192) need more lanes or the
    # BLOCK_C tile register-spills (latency-bound either way — 276M unchanged).
    _rmsnorm_kernel[(rows,)](x, weight, out, C=C, BC=triton.next_power_of_2(C), eps=eps,
                             so=so, num_stages=1, num_warps=2 if C <= 2048 else 16)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# swiglu_rows — silu(gate) * up from packed gate_up output
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# decode_prologue — full CCA prologue post-GEMM in ONE kernel (per site)
# ─────────────────────────────────────────────────────────────────────────────
#
# Consumes the stacked projection GEMM output LAT [B, 7, LQ+LK+Vh] (= W_down_q/W_down_k/
# W_v_prev over the conv window incl. the current token) + MISC (v_curr in its head):
# two stacked causal convs (depthwise k=4 → head-grouped k=4, last position), QK-mean
# coupling, per-head RMSNorm, exp(temp) key scale, CoPE-RoPE at the current position,
# value assembly (cat[v_curr,v_prev] head split), and the GQA-expanded staging writes
# into the window ring. Replaces ~35 eager kernels per site. Mirrors kv_cache._prologue_one.


@triton.jit
def _conv_unit(BASE, WDW, WGP, offs, sl_r, c0, D: tl.constexpr, KC: tl.constexpr):
    """Two stacked causal convs (depthwise→grouped) at the LAST position for one head
    group of D channels starting at flat channel c0. BASE points at the group's columns.
    y1[3+p, c] = Σ_j x[p+j, c]·wdw[c, j];  out[c] = Σ_p Σ_i wgp[c, i, p]·y1[3+p, i]."""
    acc = tl.zeros((D,), tl.float32)
    for p in tl.static_range(KC):
        y1 = tl.zeros((D,), tl.float32)
        for j in tl.static_range(KC):
            xv = tl.load(BASE + (p + j) * sl_r + offs)
            wd = tl.load(WDW + (c0 + offs) * KC + j)
            y1 += xv * wd
        wgp = tl.load(WGP + (c0 + offs)[:, None] * (D * KC) + offs[None, :] * KC + p)
        acc += tl.sum(wgp * y1[None, :], 1)
    return acc


@triton.jit
def _decode_prologue_kernel(
    LAT, MISC, COS, SIN, WDWQ, WGPQ, WDWK, WGPK, QNW, KNW, ETEMP,
    Q, WK, WV,
    eps,
    H: tl.constexpr, HKV: tl.constexpr, NREP: tl.constexpr,
    D: tl.constexpr, HD2: tl.constexpr,
    LQ: tl.constexpr, LK: tl.constexpr, KC: tl.constexpr, NH: tl.constexpr,
    WSLOT: tl.constexpr,
    sl_b, sl_r, sm_b, sq_b, sq_h, sk_b, sk_h, sk_w,
):
    # grid: B * (H + HKV). pid_u < H → one q head; else one kv group (k + v + staging).
    pid = tl.program_id(0)
    b = pid // (H + HKV)
    u = pid % (H + HKV)
    offs = tl.arange(0, D)
    offs_h = tl.arange(0, HD2)
    cos_lo = tl.load(COS + offs_h)
    cos_hi = tl.load(COS + HD2 + offs_h)
    sin_lo = tl.load(SIN + offs_h)
    sin_hi = tl.load(SIN + HD2 + offs_h)

    if u < H:
        # ── one q head ──
        h = u
        g = h // NREP
        base_q = LAT + b * sl_b + h * D
        qc = _conv_unit(base_q, WDWQ, WGPQ, offs, sl_r, h * D, D, KC)
        q_lat6 = tl.load(base_q + (NH - 1) * sl_r + offs)
        k_lat6 = tl.load(LAT + b * sl_b + LQ + g * D + (NH - 1) * sl_r + offs)
        qv = qc + (q_lat6 + k_lat6) * 0.5
        rq = 1.0 / tl.sqrt(tl.sum(qv * qv, 0) / D + eps)
        qv = (qv * rq) * tl.load(QNW + offs)
        q2 = tl.trans(tl.reshape(qv, (2, HD2)))                       # halves for RoPE
        q_lo, q_hi = tl.split(q2)
        tl.store(Q + b * sq_b + h * sq_h + offs_h, q_lo * cos_lo - q_hi * sin_lo)
        tl.store(Q + b * sq_b + h * sq_h + HD2 + offs_h, q_hi * cos_hi + q_lo * sin_hi)
    else:
        # ── one kv group: k conv + qk-mean-mean + norm + temp + rope; v; staging ×NREP ──
        g = u - H
        base_k = LAT + b * sl_b + LQ + g * D
        kc = _conv_unit(base_k, WDWK, WGPK, offs, sl_r, g * D, D, KC)
        k_lat6 = tl.load(base_k + (NH - 1) * sl_r + offs)
        qkm_sum = tl.zeros((D,), tl.float32)
        for r in tl.static_range(NREP):
            q_lat6 = tl.load(LAT + b * sl_b + (g * NREP + r) * D + (NH - 1) * sl_r + offs)
            qkm_sum += (q_lat6 + k_lat6) * 0.5
        kv_ = kc + qkm_sum / NREP
        rk = 1.0 / tl.sqrt(tl.sum(kv_ * kv_, 0) / D + eps)
        kv_ = ((kv_ * rk) * tl.load(KNW + offs)) * tl.load(ETEMP + g)
        k2 = tl.trans(tl.reshape(kv_, (2, HD2)))
        k_lo, k_hi = tl.split(k2)
        ky_lo = k_lo * cos_lo - k_hi * sin_lo
        ky_hi = k_hi * cos_hi + k_lo * sin_hi

        # V: flat cat([v_curr(Vh), v_prev(Vh)]) reshaped [Hkv, D] — golden layout:
        # head g<Hkv/2 reads v_curr[g*D:], head g>=Hkv/2 reads v_prev[(g-Hkv/2)*D:].
        vc = tl.load(MISC + b * sm_b + g * D + offs, mask=(g < HKV // 2), other=0.0)
        vp = tl.load(LAT + b * sl_b + (NH - 2) * sl_r + LQ + LK + (g - HKV // 2) * D + offs,
                     mask=(g >= HKV // 2), other=0.0)
        v2 = tl.trans(tl.reshape(vc + vp, (2, HD2)))
        v_lo, v_hi = tl.split(v2)
        for r in tl.static_range(NREP):
            h = g * NREP + r
            wkb = WK + b * sk_b + h * sk_h + WSLOT * sk_w
            tl.store(wkb + offs_h, ky_lo)
            tl.store(wkb + HD2 + offs_h, ky_hi)
            wvb = WV + b * sk_b + h * sk_h + WSLOT * sk_w
            tl.store(wvb + offs_h, v_lo)
            tl.store(wvb + HD2 + offs_h, v_hi)


def decode_prologue(lat: Tensor, misc: Tensor, cos: Tensor, sin: Tensor,
                    wdwq: Tensor, wgpq: Tensor, wdwk: Tensor, wgpk: Tensor,
                    qnw: Tensor, knw: Tensor, etemp: Tensor,
                    win_k: Tensor, win_v: Tensor,
                    n_heads: int, n_kv: int, d_head: int, lq: int, lk: int,
                    k_conv: int, eps: float) -> Tensor:
    """Fused prologue. lat [B,NH,LQ+LK+Vh] contiguous; misc [B,*] (v_curr first);
    cos/sin [D]; conv weights flattened [C,k]/[C,Cg,k]; etemp [Hkv].
    Writes the win_k/win_v staging slots; returns q [B,H,D]."""
    B, NH, _ = lat.shape
    q = torch.empty(B, n_heads, d_head, device=lat.device, dtype=lat.dtype)
    wslot = win_k.shape[2] - 1
    _decode_prologue_kernel[(B * (n_heads + n_kv),)](
        lat, misc, cos, sin, wdwq, wgpq, wdwk, wgpk, qnw, knw, etemp,
        q, win_k, win_v, eps,
        H=n_heads, HKV=n_kv, NREP=n_heads // n_kv, D=d_head, HD2=d_head // 2,
        LQ=lq, LK=lk, KC=k_conv, NH=NH, WSLOT=wslot,
        sl_b=lat.stride(0), sl_r=lat.stride(1), sm_b=misc.stride(0),
        sq_b=q.stride(0), sq_h=q.stride(1),
        sk_b=win_k.stride(0), sk_h=win_k.stride(1), sk_w=win_k.stride(2),
        num_stages=1, num_warps=1,
    )
    return q


# ─────────────────────────────────────────────────────────────────────────────
# decode_front — Wqkv GEMM + full prologue + gate-hidden + indexer-q, ONE kernel
# ─────────────────────────────────────────────────────────────────────────────
#
# Replaces, per site: the cuBLAS [7,768]x[864,768]^T gemmSN (~5.5 us), decode_prologue
# (~3.9 us) and the gate2 GEMV (~1.5 us, folded into decode_attn) with ONE launch.
# The stacked-projection GEMM is computed IN-REGISTER per unit (each unit owns the D
# weight rows it consumes; the lat intermediate never touches memory). Unit map, grid
# B*(H + HKV + GH/D [+1 CSA]):
#   u < H:        one q head — lat_q[7,D] + k_lat6, conv, qk-mean, RMS, RoPE -> Q;
#                 raw q_lat row6 -> XRES (decode_attn residual).
#   u < H+HKV:    one kv group — lat_k[7,D] + NREP q_lat6 rows + v (curr row6 /
#                 prev row5), conv, qk-mean-mean, RMS, temp, RoPE -> GQA staging
#                 writes into the window ring (v un-roped, half-interleave only).
#   u < H+HKV+GU: one gate slice — gate_h[gu*D:(gu+1)*D] = W_g rows x x6 -> GATEH
#                 (raw; silu+gate2 are folded into decode_attn).
#   last (CSA):   indexer query q_I = W_IQ x x6 -> QI.
# fp32 broadcast-FMA only (no tl.dot -> no tf32); deviation class = reduction-tree
# order, gated by the greedy token-match like every other kernel here.


@triton.jit
def _conv7(a0, a1, a2, a3, a4, a5, a6, WDW, WGP, c0, offs,
           D: tl.constexpr, KC: tl.constexpr):
    """Two stacked causal convs on the 7 in-register lat rows (mirrors _conv_unit)."""
    acc = tl.zeros((D,), tl.float32)
    for p in tl.static_range(KC):
        y1 = tl.zeros((D,), tl.float32)
        for j in tl.static_range(KC):
            if p + j == 0:
                lr = a0
            elif p + j == 1:
                lr = a1
            elif p + j == 2:
                lr = a2
            elif p + j == 3:
                lr = a3
            elif p + j == 4:
                lr = a4
            elif p + j == 5:
                lr = a5
            else:
                lr = a6
            y1 += lr * tl.load(WDW + (c0 + offs) * KC + j)
        wgp = tl.load(WGP + (c0 + offs)[:, None] * (D * KC) + offs[None, :] * KC + p)
        acc += tl.sum(wgp * y1[None, :], 1)
    return acc


@triton.jit
def _front_gemm_kernel(
    X, XOFF, WQKV, WSC, PART,
    LQK: tl.constexpr, VH: tl.constexpr, O: tl.constexpr, OP: tl.constexpr,
    KDIM: tl.constexpr, KS: tl.constexpr, BK: tl.constexpr,
    NT: tl.constexpr, NU: tl.constexpr, HAS_SC: tl.constexpr,
    PACK4: tl.constexpr,
    sx_b, sx_r,
):
    """K-split lat GEMM: PART[b, s, r, c] = Σ_{k in slice s} W[c,k]·x[r,k].
    Tiles 0..LQK/32-1 (q|k rows) emit rows 0..6; v_prev tiles emit row 5; the rest
    (v_curr | gate | q_I) emit row 6 only. Zero weight duplication; 7-row reuse of
    each weight tile; NT·KS CTAs give the occupancy a single-launch design cannot.
    HAS_SC: WQKV holds int8 per-output-row codes (Int8RowLinear deploy format); the
    fp32 row scale WSC[c] is applied at the partial store (scale·Σ ≡ Σ scale· up to
    fp distribution order — tolerance-gated on the quantized stack).
    PACK4 (⇒ HAS_SC): WQKV holds NIBBLE-packed int4 codes — byte j of a row carries
    codes for k = j (lo nibble, biased +8) and k = j + KDIM (hi); KDIM is then the
    PACKED row width (= real_d/2). Unpack is lossless for codes in [-7, 7].
    XOFF [7] int32: RING row indices of x positions p-6..p in the X history buffer
    (x history is a ring keyed pos%W — no per-step roll; see _precompute_step)."""
    pid = tl.program_id(0)
    b = pid // NU
    t = (pid % NU) // KS
    s = (pid % NU) % KS
    offs = tl.arange(0, 32)
    offs_k = tl.arange(0, BK)
    c0 = t * 32
    KCH: tl.constexpr = KDIM // KS
    xb = X + b * sx_b
    ro0 = tl.load(XOFF + 0); ro1 = tl.load(XOFF + 1); ro2 = tl.load(XOFF + 2)
    ro3 = tl.load(XOFF + 3); ro4 = tl.load(XOFF + 4); ro5 = tl.load(XOFF + 5)
    ro6 = tl.load(XOFF + 6)
    pb = PART + b * (KS * 7 * OP) + s * (7 * OP)
    NTQK: tl.constexpr = LQK // 32
    NTVP: tl.constexpr = VH // 32
    if HAS_SC:
        sc = tl.load(WSC + c0 + offs)
    else:
        sc = tl.zeros((32,), tl.float32) + 1.0

    if t < NTQK:
        if HAS_SC:
            # int8/int4 deploy schedule: per-iteration row reduction → [32]-scalar
            # accumulators instead of seven [32,BK] register tiles (57 KB at BK=64
            # capped occupancy at ~4 CTAs/SM and defeated the ns pipeliner), and a
            # CONSTANT-TRIP K loop (runtime `range(s·KCH, …)` bounds blocked the
            # pipeliner; base hoisted into the address). Microbenched 20.2→12.0
            # ms/tok at BK=256 on the 30B working set. Reduction order differs from
            # the fp32 schedule — quantized stacks are tolerance-gated.
            a0 = tl.zeros((32,), tl.float32); a1 = tl.zeros((32,), tl.float32)
            a2 = tl.zeros((32,), tl.float32); a3 = tl.zeros((32,), tl.float32)
            a4 = tl.zeros((32,), tl.float32); a5 = tl.zeros((32,), tl.float32)
            a6 = tl.zeros((32,), tl.float32)
            kbase = s * KCH
            for i in range(0, KCH, BK):
                kk = kbase + i + offs_k
                w8 = tl.load(WQKV + (c0 + offs)[:, None] * KDIM + kk[None, :])
                if PACK4:
                    wl = (w8 & 15).to(tl.float32) - 8.0
                    wh = ((w8 >> 4) & 15).to(tl.float32) - 8.0
                    a0 += tl.sum(wl * tl.load(xb + ro0 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro0 * sx_r + KDIM + kk)[None, :], 1)
                    a1 += tl.sum(wl * tl.load(xb + ro1 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro1 * sx_r + KDIM + kk)[None, :], 1)
                    a2 += tl.sum(wl * tl.load(xb + ro2 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro2 * sx_r + KDIM + kk)[None, :], 1)
                    a3 += tl.sum(wl * tl.load(xb + ro3 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro3 * sx_r + KDIM + kk)[None, :], 1)
                    a4 += tl.sum(wl * tl.load(xb + ro4 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro4 * sx_r + KDIM + kk)[None, :], 1)
                    a5 += tl.sum(wl * tl.load(xb + ro5 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro5 * sx_r + KDIM + kk)[None, :], 1)
                    a6 += tl.sum(wl * tl.load(xb + ro6 * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ro6 * sx_r + KDIM + kk)[None, :], 1)
                else:
                    w = w8.to(tl.float32)
                    a0 += tl.sum(w * tl.load(xb + ro0 * sx_r + kk)[None, :], 1)
                    a1 += tl.sum(w * tl.load(xb + ro1 * sx_r + kk)[None, :], 1)
                    a2 += tl.sum(w * tl.load(xb + ro2 * sx_r + kk)[None, :], 1)
                    a3 += tl.sum(w * tl.load(xb + ro3 * sx_r + kk)[None, :], 1)
                    a4 += tl.sum(w * tl.load(xb + ro4 * sx_r + kk)[None, :], 1)
                    a5 += tl.sum(w * tl.load(xb + ro5 * sx_r + kk)[None, :], 1)
                    a6 += tl.sum(w * tl.load(xb + ro6 * sx_r + kk)[None, :], 1)
            tl.store(pb + 0 * OP + c0 + offs, a0 * sc)
            tl.store(pb + 1 * OP + c0 + offs, a1 * sc)
            tl.store(pb + 2 * OP + c0 + offs, a2 * sc)
            tl.store(pb + 3 * OP + c0 + offs, a3 * sc)
            tl.store(pb + 4 * OP + c0 + offs, a4 * sc)
            tl.store(pb + 5 * OP + c0 + offs, a5 * sc)
            tl.store(pb + 6 * OP + c0 + offs, a6 * sc)
        else:
            # validated 276M fp32 schedule — byte-identical reduction tree.
            p0 = tl.zeros((32, BK), tl.float32); p1 = tl.zeros((32, BK), tl.float32)
            p2 = tl.zeros((32, BK), tl.float32); p3 = tl.zeros((32, BK), tl.float32)
            p4 = tl.zeros((32, BK), tl.float32); p5 = tl.zeros((32, BK), tl.float32)
            p6 = tl.zeros((32, BK), tl.float32)
            for k0 in range(s * KCH, (s + 1) * KCH, BK):
                kk = k0 + offs_k
                w = tl.load(WQKV + (c0 + offs)[:, None] * KDIM + kk[None, :]).to(tl.float32)
                p0 += w * tl.load(xb + ro0 * sx_r + kk)[None, :]
                p1 += w * tl.load(xb + ro1 * sx_r + kk)[None, :]
                p2 += w * tl.load(xb + ro2 * sx_r + kk)[None, :]
                p3 += w * tl.load(xb + ro3 * sx_r + kk)[None, :]
                p4 += w * tl.load(xb + ro4 * sx_r + kk)[None, :]
                p5 += w * tl.load(xb + ro5 * sx_r + kk)[None, :]
                p6 += w * tl.load(xb + ro6 * sx_r + kk)[None, :]
            tl.store(pb + 0 * OP + c0 + offs, tl.sum(p0, 1))
            tl.store(pb + 1 * OP + c0 + offs, tl.sum(p1, 1))
            tl.store(pb + 2 * OP + c0 + offs, tl.sum(p2, 1))
            tl.store(pb + 3 * OP + c0 + offs, tl.sum(p3, 1))
            tl.store(pb + 4 * OP + c0 + offs, tl.sum(p4, 1))
            tl.store(pb + 5 * OP + c0 + offs, tl.sum(p5, 1))
            tl.store(pb + 6 * OP + c0 + offs, tl.sum(p6, 1))
    else:
        r = tl.where(t < NTQK + NTVP, 5, 6)                 # v_prev ← x5; rest ← x6
        ror = tl.load(XOFF + r)
        if HAS_SC:
            ar = tl.zeros((32,), tl.float32)
            kbase = s * KCH
            for i in range(0, KCH, BK):
                kk = kbase + i + offs_k
                w8 = tl.load(WQKV + (c0 + offs)[:, None] * KDIM + kk[None, :])
                if PACK4:
                    wl = (w8 & 15).to(tl.float32) - 8.0
                    wh = ((w8 >> 4) & 15).to(tl.float32) - 8.0
                    ar += tl.sum(wl * tl.load(xb + ror * sx_r + kk)[None, :], 1) \
                        + tl.sum(wh * tl.load(xb + ror * sx_r + KDIM + kk)[None, :], 1)
                else:
                    ar += tl.sum(w8.to(tl.float32)
                                 * tl.load(xb + ror * sx_r + kk)[None, :], 1)
            tl.store(pb + r * OP + c0 + offs, ar * sc)
        else:
            pr = tl.zeros((32, BK), tl.float32)
            for k0 in range(s * KCH, (s + 1) * KCH, BK):
                kk = k0 + offs_k
                w = tl.load(WQKV + (c0 + offs)[:, None] * KDIM + kk[None, :]).to(tl.float32)
                pr += w * tl.load(xb + ror * sx_r + kk)[None, :]
            tl.store(pb + r * OP + c0 + offs, tl.sum(pr, 1))


@triton.jit
def _psum(PB, r, col0, offs, OP: tl.constexpr, KS: tl.constexpr):
    """Σ over the KS k-slices of PART row r at columns col0+offs (fixed order)."""
    a = tl.load(PB + 0 * (7 * OP) + r * OP + col0 + offs)
    for s in tl.static_range(1, KS):
        a += tl.load(PB + s * (7 * OP) + r * OP + col0 + offs)
    return a


@triton.jit
def _psum_r(PB, r, col0, offs, OP: tl.constexpr, KS: tl.constexpr):
    """Same as _psum with a runtime row index."""
    a = tl.load(PB + 0 * (7 * OP) + r * OP + col0 + offs)
    for s in tl.static_range(1, KS):
        a += tl.load(PB + s * (7 * OP) + r * OP + col0 + offs)
    return a


@triton.jit
def _front_post_kernel(
    PART, COS, SIN, WDWQ, WGPQ, WDWK, WGPK, QNW, KNW, ETEMP,
    Q, WK, WV, XRES, GATEH, QI, POSP,
    eps,
    H: tl.constexpr, HKV: tl.constexpr, NREP: tl.constexpr,
    D: tl.constexpr, HD2: tl.constexpr,
    LQ: tl.constexpr, LK: tl.constexpr, VH: tl.constexpr, GH: tl.constexpr,
    KC: tl.constexpr, KS: tl.constexpr, OP: tl.constexpr,
    IS_CSA: tl.constexpr, WWIN: tl.constexpr, NU: tl.constexpr,
    sq_b, sq_h, sk_b, sk_h, sk_w, sxr_b, sgh_b, sqi_b,
):
    """Combine the K-split partials + conv + qk-mean + RMS + temp + RoPE + v-assembly
    + staging/gate/q_I stores. All inputs are the just-written PART scratch (L2-hot)."""
    pid = tl.program_id(0)
    b = pid // NU
    u = pid % NU
    offs = tl.arange(0, D)
    offs_h = tl.arange(0, HD2)
    pb = PART + b * (KS * 7 * OP)

    if u < H:
        g = u // NREP
        a0 = _psum(pb, 0, u * D, offs, OP, KS)
        a1 = _psum(pb, 1, u * D, offs, OP, KS)
        a2 = _psum(pb, 2, u * D, offs, OP, KS)
        a3 = _psum(pb, 3, u * D, offs, OP, KS)
        a4 = _psum(pb, 4, u * D, offs, OP, KS)
        a5 = _psum(pb, 5, u * D, offs, OP, KS)
        a6 = _psum(pb, 6, u * D, offs, OP, KS)
        k6 = _psum(pb, 6, LQ + g * D, offs, OP, KS)
        qc = _conv7(a0, a1, a2, a3, a4, a5, a6, WDWQ, WGPQ, u * D, offs, D, KC)
        qv = qc + (a6 + k6) * 0.5
        rq = 1.0 / tl.sqrt(tl.sum(qv * qv, 0) / D + eps)
        qv = (qv * rq) * tl.load(QNW + offs)
        q2 = tl.trans(tl.reshape(qv, (2, HD2)))
        q_lo, q_hi = tl.split(q2)
        cos_lo = tl.load(COS + offs_h); cos_hi = tl.load(COS + HD2 + offs_h)
        sin_lo = tl.load(SIN + offs_h); sin_hi = tl.load(SIN + HD2 + offs_h)
        tl.store(Q + b * sq_b + u * sq_h + offs_h, q_lo * cos_lo - q_hi * sin_lo)
        tl.store(Q + b * sq_b + u * sq_h + HD2 + offs_h, q_hi * cos_hi + q_lo * sin_hi)
        tl.store(XRES + b * sxr_b + u * D + offs, a6)       # raw q_lat6 residual
    elif u < H + HKV:
        g = u - H
        a0 = _psum(pb, 0, LQ + g * D, offs, OP, KS)
        a1 = _psum(pb, 1, LQ + g * D, offs, OP, KS)
        a2 = _psum(pb, 2, LQ + g * D, offs, OP, KS)
        a3 = _psum(pb, 3, LQ + g * D, offs, OP, KS)
        a4 = _psum(pb, 4, LQ + g * D, offs, OP, KS)
        a5 = _psum(pb, 5, LQ + g * D, offs, OP, KS)
        a6 = _psum(pb, 6, LQ + g * D, offs, OP, KS)
        kc_ = _conv7(a0, a1, a2, a3, a4, a5, a6, WDWK, WGPK, g * D, offs, D, KC)
        qkm = (_psum(pb, 6, (g * NREP + 0) * D, offs, OP, KS) + a6) * 0.5
        # generic GQA ratio (was unrolled to NREP<=3); same accumulation order at NREP=3.
        for rr in tl.static_range(1, NREP):
            qkm += (_psum(pb, 6, (g * NREP + rr) * D, offs, OP, KS) + a6) * 0.5
        kv_ = kc_ + qkm / NREP
        rk = 1.0 / tl.sqrt(tl.sum(kv_ * kv_, 0) / D + eps)
        kv_ = ((kv_ * rk) * tl.load(KNW + offs)) * tl.load(ETEMP + g)
        vrow = tl.where(g < HKV // 2, LQ + LK + VH + g * D, LQ + LK + (g - HKV // 2) * D)
        vr = tl.where(g < HKV // 2, 6, 5)
        av = _psum_r(pb, vr, vrow, offs, OP, KS)
        k2 = tl.trans(tl.reshape(kv_, (2, HD2)))
        k_lo, k_hi = tl.split(k2)
        cos_lo = tl.load(COS + offs_h); cos_hi = tl.load(COS + HD2 + offs_h)
        sin_lo = tl.load(SIN + offs_h); sin_hi = tl.load(SIN + HD2 + offs_h)
        ky_lo = k_lo * cos_lo - k_hi * sin_lo
        ky_hi = k_hi * cos_hi + k_lo * sin_hi
        v2 = tl.trans(tl.reshape(av, (2, HD2)))
        v_lo, v_hi = tl.split(v2)
        wslot = (tl.load(POSP) % WWIN).to(tl.int32)         # ring slot of position p
        for r in tl.static_range(NREP):
            hh = g * NREP + r
            wkb = WK + b * sk_b + hh * sk_h + wslot * sk_w
            tl.store(wkb + offs_h, ky_lo)
            tl.store(wkb + HD2 + offs_h, ky_hi)
            wvb = WV + b * sk_b + hh * sk_h + wslot * sk_w
            tl.store(wvb + offs_h, v_lo)
            tl.store(wvb + HD2 + offs_h, v_hi)
    else:
        # gate hidden (GH) + CSA indexer query (DI=D), 256-col chunks per unit
        # (row6 partial sums). One unit at GH<=224 (the 276M shape — identical math);
        # ceil((GH+D)/256) units at larger gate widths (e.g. d=8192 → GH=2048).
        tc = u - (H + HKV)
        cols = tc * 256 + tl.arange(0, 256)
        ncols = GH + D if IS_CSA else GH
        mc = cols < ncols
        base = LQ + LK + 2 * VH
        a = tl.load(pb + 0 * (7 * OP) + 6 * OP + base + cols, mask=mc, other=0.0)
        for s in tl.static_range(1, KS):
            a += tl.load(pb + s * (7 * OP) + 6 * OP + base + cols, mask=mc, other=0.0)
        tl.store(GATEH + b * sgh_b + cols, a, mask=cols < GH)
        if IS_CSA:
            tl.store(QI + b * sqi_b + (cols - GH), a,
                     mask=(cols >= GH) & (cols < GH + D))


def decode_front(x_hist: Tensor, x_off: Tensor, pos_dev: Tensor,
                 wqkv: Tensor, cos: Tensor, sin: Tensor,
                 wdwq: Tensor, wgpq: Tensor, wdwk: Tensor, wgpk: Tensor,
                 qnw: Tensor, knw: Tensor, etemp: Tensor,
                 win_k: Tensor, win_v: Tensor,
                 q_out: Tensor, xres: Tensor, gateh: Tensor, qi: Tensor | None,
                 part: Tensor,
                 n_heads: int, n_kv: int, d_head: int, lq: int, lk: int, vh: int,
                 gh: int, k_conv: int, eps: float,
                 wqkv_scale: Tensor | None = None, wqkv_pack4: bool = False) -> None:
    """Two-launch site front-end replacing cuBLAS-GEMM + decode_prologue + gate2 GEMV:
      A) _front_gemm_kernel — K-split (KS=4) lat GEMM into the `part` scratch
         [B, KS, 7, OP]; NT·KS CTAs, zero weight duplication.
      B) _front_post_kernel — partial combine + conv/qk-mean/RMS/temp/RoPE/v-assembly,
         window staging + XRES/GATEH/QI stores (scratch is L2-hot).
    x_hist [B,Wx,d] RING buffer (x_off [7] int32 = ring rows of positions p-6..p;
    pos_dev [1] int64 = p, staging slot p%WWIN); wqkv [LQ+LK+2VH+GH(+D), d] contiguous —
    fp32/bf16 weights, or int8 codes with wqkv_scale [O] fp32 per-row scales
    (Int8RowLinear deploy stacks)."""
    B = x_hist.shape[0]
    is_csa = qi is not None
    O, KDIM = wqkv.shape
    OP = part.shape[-1]
    KS = part.shape[1]
    assert O <= OP and part.shape[2] == 7 and KDIM % KS == 0
    NT = O // 32
    assert O % 32 == 0 and (lq + lk) % 32 == 0 and vh % 32 == 0
    # ns=1 is the validated 276M launch; the int8 deploy path (wqkv_scale set)
    # pipelines its K loop (ns=3) — dependent HBM loads, measured 0.77 TB/s at ns=1.
    assert not (wqkv_pack4 and wqkv_scale is None)
    has_sc = wqkv_scale is not None
    # BK=64/ns=1/w8 = the validated 276M fp32 launch; the int8/int4 deploy schedule
    # wants BK=256 + ns=3 + w4 with the engine's KS=8 split (lab7c winner 5.93 vs
    # 6.73 ms/tok on the d4 cold-L2 working set; BK must divide KDIM/KS).
    bk = 256 if (has_sc and (KDIM // KS) % 256 == 0) else 64
    _front_gemm_kernel[(B * NT * KS,)](
        x_hist, x_off, wqkv, wqkv_scale if has_sc else x_hist, part,
        LQK=lq + lk, VH=vh, O=O, OP=OP, KDIM=KDIM, KS=KS, BK=bk,
        NT=NT, NU=NT * KS, HAS_SC=has_sc, PACK4=wqkv_pack4,
        sx_b=x_hist.stride(0), sx_r=x_hist.stride(1),
        num_stages=3 if has_sc else 1, num_warps=4 if has_sc else 8,
    )
    ntail = triton.cdiv(gh + (d_head if is_csa else 0), 256)
    NU = n_heads + n_kv + ntail
    _front_post_kernel[(B * NU,)](
        part, cos, sin, wdwq, wgpq, wdwk, wgpk, qnw, knw, etemp,
        q_out, win_k, win_v, xres, gateh, qi if is_csa else gateh, pos_dev, eps,
        H=n_heads, HKV=n_kv, NREP=n_heads // n_kv, D=d_head, HD2=d_head // 2,
        LQ=lq, LK=lk, VH=vh, GH=gh, KC=k_conv, KS=KS, OP=OP,
        IS_CSA=is_csa, WWIN=win_k.shape[2], NU=NU,
        sq_b=q_out.stride(0), sq_h=q_out.stride(1),
        sk_b=win_k.stride(0), sk_h=win_k.stride(1), sk_w=win_k.stride(2),
        sxr_b=xres.stride(0), sgh_b=gateh.stride(0),
        sqi_b=qi.stride(0) if is_csa else gateh.stride(0),
        num_stages=1, num_warps=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# csa_emit — fused CSA block emission (GatedPoolCompressor + comp_norm + indexer)
# ─────────────────────────────────────────────────────────────────────────────
#
# The eager emit path (every csa_m=4 tokens, 21 sites) was ~20 kernels/site
# (6 cuBLAS GEMMs + pad/cat/softmax/mul/sum/norm glue) ≈ +1.2 ms on emit tokens.
# Two launches per site instead:
#   E1 — six [4,768]→[4,32] GEMVs (aKV/aZ on x[4:8], bKV/bZ on x[0:4], iKV/iZ on
#        x[4:8]) into a [6,4,32] scratch; stacked weight tensor, front_gemm pattern.
#   E2 — one CTA: +B_a/+B_b bias, joint 8-way softmax (max-subtract like torch),
#        gated pooling, comp_norm RMS → C_comp[idx]; indexer 4-way softmax pooling
#        → K_I[idx] (idx read from the device CNT counter).
# Mirrors GatedPoolCompressor.forward for the LAST block at pos≥8 (stream B = real
# tokens, no -inf pad). fp32 throughout; deviation = reduction order, gated.


@triton.jit
def _csa_emit_gemm_kernel(X, XOFF, WEMIT, SCR,
                          KDIM: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr,
                          M: tl.constexpr, NUNIT: tl.constexpr,
                          sx_b, sx_r):
    pid = tl.program_id(0)
    NU: tl.constexpr = NUNIT * KS
    b = pid // NU
    u = (pid % NU) // KS
    s = (pid % NU) % KS
    offs = tl.arange(0, 32)
    offs_k = tl.arange(0, BK)
    KCH: tl.constexpr = KDIM // KS
    # units 0,1 (aKV,aZ) and 4,5 (iKV,iZ) read rows M..2M-1; units 2,3 rows 0..M-1.
    # X is a RING (+ staging row); XOFF [2M] int32 = ring rows of positions p-7..p.
    roff = tl.where((u == 2) | (u == 3), 0, M)
    xb = X + b * sx_b
    e0 = tl.load(XOFF + roff + 0); e1 = tl.load(XOFF + roff + 1)
    e2 = tl.load(XOFF + roff + 2); e3 = tl.load(XOFF + roff + 3)
    p0 = tl.zeros((32, BK), tl.float32); p1 = tl.zeros((32, BK), tl.float32)
    p2 = tl.zeros((32, BK), tl.float32); p3 = tl.zeros((32, BK), tl.float32)
    for k0 in range(s * KCH, (s + 1) * KCH, BK):
        kk = k0 + offs_k
        w = tl.load(WEMIT + (u * 32 + offs)[:, None] * KDIM + kk[None, :])
        p0 += w * tl.load(xb + e0 * sx_r + kk)[None, :]
        p1 += w * tl.load(xb + e1 * sx_r + kk)[None, :]
        p2 += w * tl.load(xb + e2 * sx_r + kk)[None, :]
        p3 += w * tl.load(xb + e3 * sx_r + kk)[None, :]
    sb = SCR + b * (KS * NUNIT * 4 * 32) + s * (NUNIT * 4 * 32) + u * (4 * 32)
    tl.store(sb + 0 * 32 + offs, tl.sum(p0, 1))
    tl.store(sb + 1 * 32 + offs, tl.sum(p1, 1))
    tl.store(sb + 2 * 32 + offs, tl.sum(p2, 1))
    tl.store(sb + 3 * 32 + offs, tl.sum(p3, 1))


@triton.jit
def _emit_unit(SCR, b, u: tl.constexpr, KS: tl.constexpr, NUNIT: tl.constexpr):
    """Sum the K-split partials (fixed order) for emit unit u: a [4,32] tile."""
    offs = tl.arange(0, 32)
    r4 = tl.arange(0, 4)
    a = tl.load(SCR + b * (KS * NUNIT * 128) + u * 128
                + r4[:, None] * 32 + offs[None, :])
    for s in tl.static_range(1, KS):
        a += tl.load(SCR + b * (KS * NUNIT * 128) + s * (NUNIT * 128) + u * 128
                     + r4[:, None] * 32 + offs[None, :])
    return a


@triton.jit
def _csa_emit_combine_kernel(SCR, BA, BB, BIA, CNW, CC, KI, CNT,
                             eps,
                             KS: tl.constexpr, NUNIT: tl.constexpr,
                             sc_b, sc_n, ski_b, ski_n):
    b = tl.program_id(0)
    offs = tl.arange(0, 32)
    r4 = tl.arange(0, 4)

    c_a = _emit_unit(SCR, b, 0, KS, NUNIT)
    z_a = _emit_unit(SCR, b, 1, KS, NUNIT) + tl.load(BA + r4[:, None] * 32 + offs[None, :])
    c_b = _emit_unit(SCR, b, 2, KS, NUNIT)
    z_b = _emit_unit(SCR, b, 3, KS, NUNIT) + tl.load(BB + r4[:, None] * 32 + offs[None, :])
    # joint softmax over the 8 rows (per feature c): max-subtract like torch.softmax
    mx = tl.maximum(tl.max(z_a, 0), tl.max(z_b, 0))                       # [32]
    ea = tl.exp(z_a - mx[None, :])
    eb = tl.exp(z_b - mx[None, :])
    den = tl.sum(ea, 0) + tl.sum(eb, 0)
    out = (tl.sum(ea * c_a, 0) + tl.sum(eb * c_b, 0)) / den               # [32]
    # comp_norm (RMSNorm over c=32)
    rms = 1.0 / tl.sqrt(tl.sum(out * out, 0) / 32 + eps)
    out = (out * rms) * tl.load(CNW + offs)
    idx = tl.load(CNT)
    tl.store(CC + b * sc_b + idx * sc_n + offs, out)

    # indexer: single-stream softmax over m=4 rows
    c_i = _emit_unit(SCR, b, 4, KS, NUNIT)
    z_i = _emit_unit(SCR, b, 5, KS, NUNIT) + tl.load(BIA + r4[:, None] * 32 + offs[None, :])
    mi = tl.max(z_i, 0)
    ei = tl.exp(z_i - mi[None, :])
    ki = tl.sum(ei * c_i, 0) / tl.sum(ei, 0)
    tl.store(KI + b * ski_b + idx * ski_n + offs, ki)


def csa_emit(x_hist: Tensor, x_off: Tensor, w_emit: Tensor, b_a: Tensor, b_b: Tensor,
             b_ia: Tensor, cn_w: Tensor, cn_eps: float, c_comp: Tensor, k_i: Tensor,
             cnt: Tensor, scratch: Tensor) -> None:
    """x_hist [B, Wx, d] RING (x_off [8] int32 = ring rows of positions p-7..p,
    tokens of blocks j-1 | j). Writes
    C_comp[b, cnt] (comp_norm'd gated pool) and K_I[b, cnt] (indexer pool).
    w_emit [6*32, d] = cat[aKV,aZ,bKV,bZ,iKV,iZ]; scratch [B, KS, 6, 4, 32] fp32."""
    B = x_hist.shape[0]
    KDIM = x_hist.shape[-1]
    KS = scratch.shape[1]
    _csa_emit_gemm_kernel[(B * 6 * KS,)](
        x_hist, x_off, w_emit, scratch, KDIM=KDIM, BK=64, KS=KS, M=4, NUNIT=6,
        sx_b=x_hist.stride(0), sx_r=x_hist.stride(1),
        num_stages=1, num_warps=4)
    _csa_emit_combine_kernel[(B,)](
        scratch, b_a, b_b, b_ia, cn_w, c_comp, k_i, cnt, cn_eps,
        KS=KS, NUNIT=6,
        sc_b=c_comp.stride(0), sc_n=c_comp.stride(1),
        ski_b=k_i.stride(0), ski_n=k_i.stride(1),
        num_stages=1, num_warps=1)


# ─────────────────────────────────────────────────────────────────────────────
# bf16_gemv — bf16-WEIGHT GEMV with fp32 x / fp32 accumulate / fp32 out
# ─────────────────────────────────────────────────────────────────────────────
#
# For the big frozen read-only matrices (LM head 151 MB fp32 = ~94 µs/tok at M=1).
# The ONLY deviation vs F.linear(x, W_fp32) is the one-time bf16 rounding of the
# stored weights (x stays fp32, every product/accumulation fp32) — re-gated by the
# greedy token-match. Halves the weight traffic.


@triton.jit
def _bf16_gemv_kernel(X, W, OUT, I: tl.constexpr, O: tl.constexpr,
                      BI: tl.constexpr, BO: tl.constexpr, sx_b, so_b):
    pid_o = tl.program_id(0)
    b = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    acc = tl.zeros((BO, BI), tl.float32)
    for i0 in range(0, I, BI):
        offs_i = i0 + tl.arange(0, BI)
        x = tl.load(X + b * sx_b + offs_i)
        w = tl.load(W + offs_o[:, None] * I + offs_i[None, :]).to(tl.float32)
        acc += w * x[None, :]
    tl.store(OUT + b * so_b + offs_o, tl.sum(acc, 1))


def bf16_gemv(x: Tensor, w_bf16: Tensor) -> Tensor:
    """x [B, I] fp32; w_bf16 [O, I] bf16 contiguous (I % BI == 0). Returns [B, O] fp32."""
    O, I = w_bf16.shape
    B = x.shape[0]
    out = torch.empty(B, O, device=x.device, dtype=torch.float32)
    BO, BI = 16, 64
    assert O % BO == 0 and I % BI == 0, (O, I)
    _bf16_gemv_kernel[(O // BO, B)](
        x, w_bf16, out, I=I, O=O, BI=BI, BO=BO,
        sx_b=x.stride(0), so_b=out.stride(0), num_stages=2, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# loop_ssm_term — LoopSSM ctx-slice update + next-layer inject term, ONE kernel
# ─────────────────────────────────────────────────────────────────────────────
#
# Replaces, per core iteration: mul + mul + add (ssm) + cat + broadcast term-add
# (≈5 eager launches) with one. out = cat([h[:lo], A·h[lo:hi] + dt·e[lo:hi], h[hi:]])
# + term (term broadcast over the N streams). NOT in-place: h at t=0 aliases e and
# e is re-read every iteration.


@triton.jit
def _loop_ssm_kernel(H, E, A, DT, TERM, OUT, lo, hi,
                     C: tl.constexpr, BC: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)                              # b*N + stream
    b = row // N
    c = tl.arange(0, BC)
    m = c < C
    hv = tl.load(H + row * C + c, mask=m, other=0.0)
    inctx = (c >= lo) & (c < hi)
    av = tl.load(A + (c - lo), mask=m & inctx, other=0.0)
    dv = tl.load(DT + (c - lo), mask=m & inctx, other=0.0)
    ev = tl.load(E + row * C + c, mask=m & inctx, other=0.0)
    sv = av * hv + dv * ev
    tv = tl.load(TERM + b * C + c, mask=m, other=0.0)
    tl.store(OUT + row * C + c, tl.where(inctx, sv, hv) + tv, mask=m)


def loop_ssm_term(h: Tensor, e: Tensor, A: Tensor, dt: Tensor, term: Tensor,
                  lo: int, hi: int) -> Tensor:
    """h/e [B,1,N,C] contiguous fp32; A/dt [hi-lo]; term [B,1,C]. Returns new h."""
    B, S, N, C = h.shape
    out = torch.empty_like(h)
    _loop_ssm_kernel[(B * S * N,)](
        h, e, A, dt, term, out, lo, hi,
        C=C, BC=triton.next_power_of_2(C), N=N, num_stages=1, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# small_gemv — single-launch fp32 GEMV for the M=1 decode projections
# ─────────────────────────────────────────────────────────────────────────────
#
# cuBLAS dispatches the tiny decode GEMVs ([768→384] W_up, [3072→48] HC proj, …) to
# dot+reduce(+splitK) kernel TRIPLES at these shapes; at B=1 each launch is ~1.5 µs of
# pure overhead. One Triton launch replaces them. fp32, order-only deviation, gated.


@triton.jit
def _small_gemv_kernel(X, W, BIAS, OUT, I: tl.constexpr, O: tl.constexpr,
                       BI: tl.constexpr, BO: tl.constexpr, HAS_BIAS: tl.constexpr,
                       sx_b, so_b):
    pid_o = tl.program_id(0)
    b = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    mo = offs_o < O
    acc = tl.zeros((BO,), tl.float32)
    for i0 in range(0, I, BI):
        offs_i = i0 + tl.arange(0, BI)
        mi = offs_i < I
        x = tl.load(X + b * sx_b + offs_i, mask=mi, other=0.0)
        w = tl.load(W + offs_o[:, None] * I + offs_i[None, :],
                    mask=mo[:, None] & mi[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_o, mask=mo, other=0.0)
    tl.store(OUT + b * so_b + offs_o, acc, mask=mo)


@triton.jit
def _gemv_row_kernel(X, W, BIAS, OUT, I: tl.constexpr, BI: tl.constexpr,
                     HAS_BIAS: tl.constexpr, SILU_X: tl.constexpr, sx_b, so_b):
    # one CTA per output ROW: parallel across O, chunked deterministic dot over I.
    o = tl.program_id(0)
    b = tl.program_id(1)
    acc = tl.zeros((BI,), tl.float32)
    for i0 in range(0, I, BI):
        offs = i0 + tl.arange(0, BI)
        mi = offs < I
        x = tl.load(X + b * sx_b + offs, mask=mi, other=0.0)
        if SILU_X:
            x = x * tl.sigmoid(x)
        w = tl.load(W + o * I + offs, mask=mi, other=0.0).to(tl.float32)
        acc += x * w
    y = tl.sum(acc, 0)
    if HAS_BIAS:
        y += tl.load(BIAS + o)
    tl.store(OUT + b * so_b + o, y)


def small_gemv(x: Tensor, w: Tensor, bias: Tensor | None = None,
               silu_x: bool = False) -> Tensor:
    """x [B, I] (row-contiguous, any row stride); w [O, I] contiguous → [B, O] fp32.
    silu_x applies SiLU to x inside the kernel (gate-MLP path). Two schedules:
    row-parallel (one CTA/row — wins for small O / wide I, e.g. the [48,3072] HC proj
    where cuBLAS gemv2T burns 10.8 µs) and O-tiled (larger O)."""
    O, I = w.shape
    B = x.shape[0]
    out = torch.empty(B, O, device=x.device, dtype=torch.float32)
    if O <= 64 or I >= 2048 or (O <= 1024 and I <= 1024):
        BI = min(triton.next_power_of_2(I), 1024)
        # ns=1 is the validated 276M launch (I<=4096 there); wide rows (W_x0/W_mix
        # at d=8192: I=8192+) chunk-loop over I → pipeline the dependent loads.
        _gemv_row_kernel[(O, B)](
            x, w, bias if bias is not None else w, out, I=I, BI=BI,
            HAS_BIAS=bias is not None, SILU_X=silu_x,
            sx_b=x.stride(0), so_b=out.stride(0),
            num_stages=1 if I <= 4096 else 3, num_warps=4)
        return out
    assert not silu_x, "silu_x only wired for the row schedule"
    BO = 64
    BI = 256
    _small_gemv_kernel[(triton.cdiv(O, BO), B)](
        x, w, bias if bias is not None else w, out, I=I, O=O, BI=BI, BO=BO,
        HAS_BIAS=bias is not None, sx_b=x.stride(0), so_b=out.stride(0), **_LAUNCH)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ternary_gemv — int8-code GEMV for ternary-QAT weights (backbone MLPs etc.)
# ─────────────────────────────────────────────────────────────────────────────
#
# The materialized ternary weight is exactly γ·code with code ∈ {-1,0,+1} (tensor-mode
# scale). Storing int8 codes cuts GEMV weight traffic 4× vs fp32 (the decode GEMVs are
# pure weight-bandwidth at M=1). y = γ·Σ code_i·x_i — every product is exact (±x or 0);
# the deviation vs cuBLAS fp32 is accumulation order + γ-after-sum (vs per-element),
# both order-of-ulp; gated by the greedy token-match.


@triton.jit
def _ternary_gemv_kernel(X, W2, SCALE, WN, OUT, I: tl.constexpr, O: tl.constexpr,
                         BI4: tl.constexpr, BO: tl.constexpr, BC: tl.constexpr,
                         X_MODE: tl.constexpr, eps):
    # 2-BIT PACKED codes: byte j of row o holds codes for i ∈ {j, j+I/4, j+I/2, j+3I/4}
    # (strip-packed → x chunks stay contiguous), code = ((byte >> 2s) & 3) − 1.
    # X_MODE: 0 = plain x; 1 = RMSNorm(x)·WN on the fly (gate_up after norm_mlp);
    #         2 = swiglu from packed gate_up output (X holds [.., 2I], h=silu(g)·u).
    # BI4 = next_pow2(I/4) capped at 512 → the whole code row loads in 1-2 masked
    # shots (the old 64-byte chunk loop serialized 3 dependent HBM round-trips).
    pid_o = tl.program_id(0)
    b = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    I4: tl.constexpr = I // 4
    if X_MODE == 1:
        offs_c = tl.arange(0, BC)
        xv = tl.load(X + b * I + offs_c, mask=offs_c < I, other=0.0)
        rms = 1.0 / tl.sqrt(tl.sum(xv * xv, 0) / I + eps)
    acc = tl.zeros((BO,), dtype=tl.float32)
    for j0 in range(0, I4, BI4):
        offs_j = j0 + tl.arange(0, BI4)
        mj = offs_j < I4
        p = tl.load(W2 + offs_o[:, None] * I4 + offs_j[None, :],
                    mask=mj[None, :], other=0)                         # uint8 [BO,BI4]
        for st in tl.static_range(4):
            ii = st * I4 + offs_j
            if X_MODE == 2:
                g = tl.load(X + b * 2 * I + ii, mask=mj, other=0.0)
                u = tl.load(X + b * 2 * I + I + ii, mask=mj, other=0.0)
                x = g * tl.sigmoid(g) * u
            else:
                x = tl.load(X + b * I + ii, mask=mj, other=0.0)
                if X_MODE == 1:
                    x = (x * rms) * tl.load(WN + ii, mask=mj, other=0.0)
            w = ((p >> (2 * st)) & 3).to(tl.float32) - 1.0
            # masked p bytes decode to code 0-1 = -1 → force masked lanes to 0·x
            w = tl.where(mj[None, :], w, 0.0)
            acc += tl.sum(w * x[None, :], 1)
    y = acc * tl.load(SCALE)
    tl.store(OUT + b * O + offs_o, y)


def ternary_pack(w: Tensor, rel_tol: float = 1e-5) -> tuple[Tensor, Tensor] | None:
    """Pack a ternary fp32 weight into (int8 codes, fp32 scale γ). Returns None if the
    tensor is not γ·{-1,0,1} within `rel_tol` relative (caller falls back to F.linear).

    The QAT STE materializes ``w + (γ·code − w)`` whose double-rounding leaves a 1–2 ulp
    wobble around ±γ, so a bit-strict check would always fail; the tolerance admits ONLY
    that ulp wobble (1e-5 ≪ any real weight difference) and the snap is the deployment
    semantics of a ternary weight anyway. Deviation class = fp-ulp, gated by token-match."""
    nz = w[w != 0]
    if nz.numel() == 0:
        return None
    mags = nz.abs()
    gamma = mags.median()
    if ((mags - gamma).abs().max() > rel_tol * gamma).item():
        return None                                  # not (close enough to) ternary
    O, I = w.shape
    assert I % 4 == 0, I
    I4 = I // 4
    codes = (torch.sign(w).to(torch.int8) + 1).to(torch.uint8)   # {-1,0,1} → {0,1,2}
    # strip-pack 4 codes/byte: byte j ← codes at i = j, j+I/4, j+I/2, j+3I/4.
    strips = codes.view(O, 4, I4)
    packed = (strips[:, 0] | (strips[:, 1] << 2) | (strips[:, 2] << 4)
              | (strips[:, 3] << 6)).contiguous()
    return packed, gamma.reshape(1).contiguous().float()


def ternary_gemv(x: Tensor, codes: Tensor, scale: Tensor,
                 rms_weight: Tensor | None = None, rms_eps: float = 1e-6,
                 swiglu_x: bool = False) -> Tensor:
    """x [B,1,I] fp32 contiguous; codes [O,I] int8; scale [1] fp32 → [B,1,O].
    rms_weight: fold RMSNorm(x)·w on the fly (the MLP norm). swiglu_x: x is the packed
    [B,1,2I] gate_up output, h = silu(g)·u computed on load. (Mutually exclusive.)"""
    O, I4 = codes.shape
    I = I4 * 4
    B = x.shape[0]
    out = torch.empty(B, 1, O, device=x.device, dtype=x.dtype)
    # schedule: BO=8 fills the GPU with CTAs (the BO=32 grid left the [768,2048] down
    # GEMV at 24 CTAs — latency-bound). Microbenched in-graph (ignore/bench_decode_kernels):
    # gate_up [4096,768] 5.58→3.56 µs, down [768,2048] 4.08→2.95 µs. Same math, same
    # per-output reduction order (BI4 unchanged) → identical results to the BO=32 grid.
    BO = 8
    # whole code row in 1-2 masked shots. CAUTION: BI4=256 with X_MODE=1 miscomputes
    # (+phantom codes) on Triton 3.6 — ignore/probe_t4.py is the regression probe;
    # 128-masked and 256/512 on modes 0/2 verified exact vs fp64.
    BI4 = min(triton.next_power_of_2(I4), 512)
    if rms_weight is not None:
        BI4 = min(BI4, 128)
    assert O % BO == 0, (O, I)
    assert not (rms_weight is not None and swiglu_x)
    mode = 1 if rms_weight is not None else (2 if swiglu_x else 0)
    _ternary_gemv_kernel[(triton.cdiv(O, BO), B)](
        x, codes, scale, rms_weight if rms_weight is not None else x, out,
        I=I, O=O, BI4=BI4, BO=BO, BC=triton.next_power_of_2(I),
        X_MODE=mode, eps=rms_eps,
        num_stages=1, num_warps=8 if swiglu_x else 4)
    return out


@triton.jit
def _swiglu_kernel(GU, OUT, FF: tl.constexpr, BF: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BF)
    mask = offs < FF
    g = tl.load(GU + row * 2 * FF + offs, mask=mask, other=0.0)
    u = tl.load(GU + row * 2 * FF + FF + offs, mask=mask, other=0.0)
    tl.store(OUT + row * FF + offs, g * tl.sigmoid(g) * u, mask=mask)


def swiglu_rows(gu: Tensor) -> Tensor:
    """gu [..., 2F] contiguous → silu(gu[...,:F]) * gu[...,F:], [..., F]."""
    F2 = gu.shape[-1]
    FF = F2 // 2
    rows = gu.numel() // F2
    out = torch.empty(*gu.shape[:-1], FF, device=gu.device, dtype=gu.dtype)
    _swiglu_kernel[(rows,)](gu, out, FF=FF, BF=triton.next_power_of_2(FF), **_LAUNCH)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# int8_gemv — per-output-row int8 GEMV (Int8RowLinear deploy format)
# ─────────────────────────────────────────────────────────────────────────────
#
# The 30B deploy stack stores attention projections as int8 codes + fp32 row scales
# (EXACT IntNLinearSTE semantics, packed_ternary_infer.Int8RowLinear). At M=1 the
# GEMV is pure weight bandwidth: int8 codes = 4× less traffic than fp32 / 2× less
# than bf16. y[o] = scale[o] · Σ codes[o,i]·x[i] — codes are exact in fp32; the
# deviation vs dequant-then-F.linear is scale-after-sum + reduction order
# (tolerance-gated on the quantized stack; the 276M fp32 path never calls this).


@triton.jit
def _i8row_gemv_kernel(X, W8, SC, OUT, I: tl.constexpr, O: tl.constexpr,
                       BI: tl.constexpr, BO: tl.constexpr, PACK4: tl.constexpr,
                       sx_b, so_b):
    # PACK4: W8 holds nibble-packed int4 (byte j of a row = codes k=j lo / k=I+j hi,
    # biased +8; I is then the PACKED row width = real_in/2).
    pid_o = tl.program_id(0)
    b = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    acc = tl.zeros((BO, BI), tl.float32)
    for i0 in range(0, I, BI):
        offs_i = i0 + tl.arange(0, BI)
        w8 = tl.load(W8 + offs_o[:, None] * I + offs_i[None, :])
        if PACK4:
            xl = tl.load(X + b * sx_b + offs_i)
            xh = tl.load(X + b * sx_b + I + offs_i)
            acc += ((w8 & 15).to(tl.float32) - 8.0) * xl[None, :] \
                + (((w8 >> 4) & 15).to(tl.float32) - 8.0) * xh[None, :]
        else:
            x = tl.load(X + b * sx_b + offs_i)
            acc += w8.to(tl.float32) * x[None, :]
    y = tl.sum(acc, 1) * tl.load(SC + offs_o)
    tl.store(OUT + b * so_b + offs_o, y)


def int8_gemv(x: Tensor, codes: Tensor, row_scale: Tensor,
              pack4: bool = False) -> Tensor:
    """x [B, I] fp32 (row-contiguous); codes [O, I] int8 contiguous (or [O, I/2]
    nibble-packed uint8 when pack4); row_scale [O] fp32. Returns [B, O] fp32."""
    O, I = codes.shape
    B = x.shape[0]
    out = torch.empty(B, O, device=x.device, dtype=torch.float32)
    BO, BI = 16, 128
    assert O % BO == 0 and I % BI == 0, (O, I)
    _i8row_gemv_kernel[(O // BO, B)](
        x, codes, row_scale, out, I=I, O=O, BI=BI, BO=BO, PACK4=pack4,
        sx_b=x.stride(0), so_b=out.stride(0), num_stages=4, num_warps=4)
    return out


def pack_nibbles(codes: Tensor) -> Tensor:
    """int8 codes in [-7,7] → nibble-packed uint8 [O, I/2]: byte j = (c_j + 8) |
    ((c_{I/2+j} + 8) << 4). Lossless for the int4 (hi=7) row-quant range."""
    O, I = codes.shape
    assert I % 2 == 0
    assert int(codes.abs().max()) <= 7, "codes exceed int4 range — not an int4 layer"
    u = (codes.to(torch.int16) + 8).to(torch.uint8)
    return (u[:, : I // 2] | (u[:, I // 2:] << 4)).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# mortar_gemv — carved-BCSR ternary 2-bit GEMV (MORTAR deploy MLP at M=1)
# ─────────────────────────────────────────────────────────────────────────────
#
# The 30B MLP is MORTAR-carved (128×128 BCSR, density 0.25) with per-tensor-scale
# ternary codes packed 2-bit (packed_ternary_infer.pack_mortar_ternary). At M=1 the
# carved GEMV reads ONLY the kept blocks' codes — 4× less MLP weight traffic than a
# dense-expanded ternary GEMV (9.7 vs 38.7 GB/token on the 30B config).
#
# Storage (built once by the engine): strip-packed codes [nnz, BLK, BLK//4] uint8 —
# byte (j, lr, b) holds the codes of block j, row lr, columns {b, 32+b, 64+b, 96+b}
# (code+1 in 2 bits, little-endian) — plus the layer's BCSR offsets [R+1] (block-row
# CSR pointers) and column_indices [nnz] (block-column ids), straight from carve().
#
# y[r·BLK + lr] = scale · Σ_{j ∈ row r} Σ_c codes[j, lr, c] · x[col_j·BLK + c]
# Every product is exact (±x or 0); deviation vs the eager stk path = reduction
# order + scale-after-sum (tolerance-gated). SWIGLU: x is the packed [B, 2F]
# gate_up output; h = silu(g)·u computed on load (the down projection).


@triton.jit
def _mortar_gemv_kernel(X, CODES, COLIDX, OFFS, ROWS, J0S, SLOTS, PART,
                        ROWACT, COLACT,
                        BLK: tl.constexpr, BO: tl.constexpr, CB: tl.constexpr,
                        OTOT: tl.constexpr, NB: tl.constexpr,
                        SWIGLU: tl.constexpr, FF: tl.constexpr,
                        HAS_RACT: tl.constexpr, HAS_CACT: tl.constexpr,
                        sx_b):
    # BALANCED WORK-LIST schedule. The carve is RAGGED (real 30B build: gate_up
    # rows 6..26 kept blocks, down 31..59 — probe_ragged.py), so the old
    # K-split-by-NSPL grid issued 33-38% WASTED block loads (masked iterations
    # clamped j to `lo` and re-read real bytes) and left the slowest row binding
    # every CTA. The work list (built once at pack time — topology is frozen)
    # gives every CTA exactly CB consecutive blocks of ONE row: zero waste
    # beyond the <=CB-1 row tail (masked loads, ~2%), perfect balance, and a
    # CONSTANT trip count (static_range) for the pipeliner. Deferred reduction:
    # accumulate the [BO, BLK/4] tile, ONE tl.sum at the end (the per-iteration
    # log-tree reduces were ~10% of the kernel). Cold-L2 lab (ragged shapes,
    # depth-4 visit order): prod 11.08 -> 5.13 ms/tok pair (roofline 2.78).
    pid = tl.program_id(0)
    b = tl.program_id(1)
    PER: tl.constexpr = BLK // BO
    e = pid // PER
    lr0 = (pid % PER) * BO
    r = tl.load(ROWS + e)
    j0 = tl.load(J0S + e)
    sl = tl.load(SLOTS + e)
    hi = tl.load(OFFS + r + 1)
    offs_r = tl.arange(0, BO)
    offs_c = tl.arange(0, BLK // 4)
    if HAS_RACT:
        # ReMoE route-then-GATHER: inactive OUTPUT block-row (its h is gated to
        # exact 0 by the combine epilogue) → the WHOLE CTA early-returns after
        # storing a zero partial (the combine reads it; torch.empty scratch may
        # hold NaN bits and 0·NaN = NaN). Masked-load-only gathering saved no
        # time — the kernel is unpack-ALU/issue-bound, so predicated-off loads
        # still paid the arithmetic; the early return skips it all.
        if tl.load(ROWACT + r) == 0:
            tl.store(PART + (sl * NB + b) * OTOT + r * BLK + lr0 + offs_r,
                     tl.zeros((BO,), tl.float32))
            return
    acc = tl.zeros((BO, BLK // 4), tl.float32)
    for jj in tl.static_range(CB):
        j = j0 + jj
        ok = j < hi
        cb = tl.load(COLIDX + j, mask=ok, other=0)
        if HAS_CACT:
            # down proj: skip blocks whose 128 input neurons are all gated 0
            ok = ok & (tl.load(COLACT + cb, mask=ok, other=0) != 0)
        p = tl.load(CODES + j * (BLK * (BLK // 4))
                    + (lr0 + offs_r)[:, None] * (BLK // 4) + offs_c[None, :],
                    mask=ok, other=0)
        m = tl.where(ok, 1.0, 0.0)
        for st in tl.static_range(4):
            xi = cb * BLK + st * (BLK // 4) + offs_c
            if SWIGLU:
                g = tl.load(X + b * sx_b + xi)
                u = tl.load(X + b * sx_b + FF + xi)
                xs = g * tl.sigmoid(g) * u
            else:
                xs = tl.load(X + b * sx_b + xi)
            w = ((p >> (2 * st)) & 3).to(tl.float32) - 1.0
            acc += w * (xs * m)[None, :]
    tl.store(PART + (sl * NB + b) * OTOT + r * BLK + lr0 + offs_r, tl.sum(acc, 1))


@triton.jit
def _mortar_combine_kernel(PART, CNTS, OUT, SCALE, GATES,
                           O: tl.constexpr, NSPLMAX: tl.constexpr,
                           NB: tl.constexpr, BS: tl.constexpr,
                           BLK: tl.constexpr, SWIGLU_OUT: tl.constexpr,
                           FF: tl.constexpr, HAS_GATES: tl.constexpr,
                           CLS: tl.constexpr, so_b):
    """Combine the per-chunk partials. CNTS [R] = chunks per row → masked reads
    (no PART zero-fill; absent slots are never touched). SWIGLU_OUT: PART holds
    the packed [2F] gate_up sums; OUT[c] = silu(g)·u computed ONCE here instead
    of per block visit inside the down kernel (bit-identical relocation — the
    partial-sum order and the silu inputs are unchanged)."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BS + tl.arange(0, BS)
    if SWIGLU_OUT:
        rg = offs // BLK
        ru = (offs + FF) // BLK
        cg = tl.load(CNTS + rg)
        cu = tl.load(CNTS + ru)
        g = tl.zeros((BS,), tl.float32)
        u = tl.zeros((BS,), tl.float32)
        for s in tl.static_range(NSPLMAX):
            g += tl.load(PART + (s * NB + b) * O + offs, mask=s < cg, other=0.0)
            u += tl.load(PART + (s * NB + b) * O + FF + offs, mask=s < cu, other=0.0)
        sc = tl.load(SCALE)
        g = g * sc
        u = u * sc
        h = g * tl.sigmoid(g) * u
        if HAS_GATES:
            # ReMoE: h gated per neuron-cluster (cluster = neuron // CLS); the
            # router's continuous relu gates — inactive clusters are exact 0.
            NCLS: tl.constexpr = FF // CLS
            h = h * tl.load(GATES + b * NCLS + offs // CLS)
        tl.store(OUT + b * so_b + offs, h)
    else:
        mo = offs < O
        r = offs // BLK
        cnt = tl.load(CNTS + r, mask=mo, other=0)
        a = tl.zeros((BS,), tl.float32)
        for s in tl.static_range(NSPLMAX):
            a += tl.load(PART + (s * NB + b) * O + offs,
                         mask=mo & (s < cnt), other=0.0)
        tl.store(OUT + b * so_b + offs, a * tl.load(SCALE), mask=mo)


_MORTAR_CB = 4                                  # blocks per work-list entry (lab6c winner)


def mortar_pack_strips(cms) -> tuple:
    """Convert a pack_mortar_ternary'd CMSBlockLinear into the strip-packed engine
    format + the balanced work list (one entry per <=CB consecutive blocks of a
    row). Returns (strips [nnz,BLK,BLK//4] uint8, col_idx int32, offsets int32,
    scale [1] fp32, out_features, ent [3,E] int32 (row|j0|slot), nspl_max,
    cnts [R] int32). One-time; the codes are the SAME ternary codes the eager
    `_mortar_effective_data` dequantizes (re-gated)."""
    from morph.model.packed_ternary_infer import unpack_ternary
    nnz, blk, blk2 = cms._packed_shape
    assert blk == blk2 and blk % 4 == 0, cms._packed_shape
    codes = unpack_ternary(cms.mortar_packed, cms._packed_numel, torch.float32)
    u = (codes.view(nnz, blk, blk) + 1.0).to(torch.uint8).view(nnz, blk, 4, blk // 4)
    strips = (u[:, :, 0] | (u[:, :, 1] << 2)
              | (u[:, :, 2] << 4) | (u[:, :, 3] << 6)).contiguous()
    offsets = cms.mortar_offsets.to(torch.int32).contiguous()
    out_features = (int(offsets.numel()) - 1) * blk
    dev = strips.device
    offs_c = offsets.cpu()
    CB = _MORTAR_CB
    rows, j0s, slots = [], [], []
    R = offs_c.numel() - 1
    nspl_max = 0
    for r in range(R):
        lo, hi = int(offs_c[r]), int(offs_c[r + 1])
        nch = -(-(hi - lo) // CB)
        nspl_max = max(nspl_max, nch)
        for s in range(nch):
            rows.append(r); j0s.append(lo + s * CB); slots.append(s)
    ent = torch.stack([torch.tensor(rows, dtype=torch.int32),
                       torch.tensor(j0s, dtype=torch.int32),
                       torch.tensor(slots, dtype=torch.int32)]).to(dev).contiguous()
    cnts = (-(-(offs_c[1:] - offs_c[:-1]) // CB)).to(torch.int32).to(dev).contiguous()
    return (strips, cms.mortar_column_indices.to(torch.int32).contiguous(),
            offsets, cms.mortar_scale.detach().reshape(1).float().contiguous(),
            out_features, ent, nspl_max, cnts)


def mortar_gemv(x: Tensor, pack: tuple, swiglu_x: bool = False,
                swiglu_out: bool = False, row_act: Tensor | None = None,
                col_act: Tensor | None = None, gates: Tensor | None = None,
                cluster_size: int = 0) -> Tensor:
    """x [B, I] fp32 row-contiguous (or [B, 2F] packed gate_up when swiglu_x);
    pack = mortar_pack_strips(...) output. Returns [B, O] fp32 — or, with
    swiglu_out (the gate_up call), [B, O/2] = silu(g)·u so the down projection
    reads a plain vector instead of re-deriving swiglu per block visit."""
    strips, col_idx, offsets, scale, O, ent, nspl_max, cnts = pack
    nnz, blk, _ = strips.shape
    B = x.shape[0]
    E = ent.shape[1]
    FF = x.shape[-1] // 2 if swiglu_x else 0
    BO = 32
    assert not (swiglu_x and swiglu_out)
    dummy = offsets
    part = torch.empty(nspl_max, B, O, device=x.device, dtype=torch.float32)
    _mortar_gemv_kernel[(E * (blk // BO), B)](
        x, strips, col_idx, offsets, ent[0], ent[1], ent[2], part,
        row_act if row_act is not None else dummy,
        col_act if col_act is not None else dummy,
        BLK=blk, BO=BO, CB=_MORTAR_CB, OTOT=O, NB=B, SWIGLU=swiglu_x, FF=FF,
        HAS_RACT=row_act is not None, HAS_CACT=col_act is not None,
        sx_b=x.stride(0), num_stages=3, num_warps=4)
    Oout = O // 2 if swiglu_out else O
    out = torch.empty(B, Oout, device=x.device, dtype=torch.float32)
    _mortar_combine_kernel[(triton.cdiv(Oout, 512), B)](
        part, cnts, out, scale, gates if gates is not None else scale,
        O=O, NSPLMAX=nspl_max, NB=B, BS=512,
        BLK=blk, SWIGLU_OUT=swiglu_out, FF=O // 2 if swiglu_out else 0,
        HAS_GATES=gates is not None, CLS=max(cluster_size, 1),
        so_b=out.stride(0), num_stages=1, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Wide-carrier HyperConnection decode path (d_model > 2048, e.g. the 30B d=8192)
# ─────────────────────────────────────────────────────────────────────────────
#
# The validated 276M premap/post kernels run ONE CTA per token (the whole n·C
# carrier in registers) and the mapping projection ran a 48-CTA serial-K GEMV:
# at C=8192 that is ~21 µs of pure latency per _hc call (296 calls/token at
# depth-4 ⇒ ~6.4 ms/tok). This suite splits the same math across the GPU:
#   W1 — K-split mapping GEMV (grid O×KS) + carrier ssq partials (o==0 lane).
#   W2 — one tiny CTA/token: combine partials → raw[48], rms; softmax² + Cayley
#        on [4,4] tiles (vectorized; same math as the scalar 276M kernel, fp
#        reduction-tree order differs) → hres/hpostrow/hprecm.
#   W3 — xbar chunks: xbar[c] = Σ_j hprecm[j]·h[j,c] (+ xbar² partials for W4).
#   W4 — (attn side only) RMSNorm(xbar)·nw into the x-history staging slot.
#   W5 — post: out[i,c] = Σ_j hres[i,j]·h[j,c] + hpostrow[i]·y[c] (+term).
# 276M (C=768) keeps the original single-CTA kernels byte-identical; this path
# is tolerance-gated on the 30B (reduction order + K-split only — same inputs).
# N=4 only (asserted): the [4,4] tile math mirrors _sm4/_cayley_fwd_step.


@triton.jit
def _hcw_gemv_kernel(H, W, SSQ, PART, IDIM: tl.constexpr, KS: tl.constexpr,
                     BI: tl.constexpr, O: tl.constexpr):
    pid = tl.program_id(0)                  # o * KS + s
    row = tl.program_id(1)
    o = pid // KS
    s = pid % KS
    offs = s * BI + tl.arange(0, BI)
    x = tl.load(H + row * IDIM + offs)
    w = tl.load(W + o * IDIM + offs).to(tl.float32)
    tl.store(PART + (row * O + o) * KS + s, tl.sum(w * x, 0))
    if o == 0:
        tl.store(SSQ + row * KS + s, tl.sum(x * x, 0))


@triton.jit
def _hcw_map_kernel(PART, BIAS, SSQ, HRES, HPOSTROW, HPRECM,
                    KS: tl.constexpr, NC: tl.constexpr,
                    TAU: tl.constexpr, ALPHA: tl.constexpr, ITERS: tl.constexpr,
                    EPS: tl.constexpr):
    row = tl.program_id(0)
    i4 = tl.arange(0, 4)
    j4 = tl.arange(0, 4)
    sK = tl.arange(0, KS)
    pb = PART + row * 48 * KS
    # [4,4,KS] tiles for the three raw blocks; reduce the K split (fixed tree).
    idx = (i4[:, None, None] * 4 + j4[None, :, None]) * KS + sK[None, None, :]
    pre = tl.sum(tl.load(pb + idx), 2) + tl.load(BIAS + i4[:, None] * 4 + j4[None, :]).to(tl.float32)
    post = tl.sum(tl.load(pb + 16 * KS + idx), 2) \
        + tl.load(BIAS + 16 + i4[:, None] * 4 + j4[None, :]).to(tl.float32)
    res = tl.sum(tl.load(pb + 32 * KS + idx), 2) \
        + tl.load(BIAS + 32 + i4[:, None] * 4 + j4[None, :]).to(tl.float32)
    ssq = tl.sum(tl.load(SSQ + row * KS + sK), 0)
    inv_rms = 1.0 / tl.sqrt(ssq / NC + EPS)
    inv_tau = 1.0 / TAU

    # Hpre: softmax over j (rows), column-mean → hprecm[j]
    p = pre * inv_rms * inv_tau
    m = tl.max(p, 1)
    e = tl.exp(p - m[:, None])
    hpre = e / tl.sum(e, 1)[:, None]
    hprecm = tl.sum(hpre, 0) * 0.25
    tl.store(HPRECM + row * 4 + j4, hprecm)

    # Hpost: softmax over i (columns), row-sum → hpostrow[i]
    q = post * inv_rms * inv_tau
    mq = tl.max(q, 0)
    eq = tl.exp(q - mq[None, :])
    hpost = eq / tl.sum(eq, 0)[None, :]
    tl.store(HPOSTROW + row * 4 + i4, tl.sum(hpost, 1))

    # Hres: Cayley fixed point on the skew part of res/rms.
    a = res * inv_rms
    w = a - tl.trans(a)
    eye = (i4[:, None] == j4[None, :]).to(tl.float32)
    y = eye + ALPHA * w
    half: tl.constexpr = ALPHA * 0.5
    for _ in tl.static_range(ITERS):
        t = eye + y
        p3 = tl.sum(w[:, :, None] * t[None, :, :], 1)       # W @ (I + Y)
        y = eye + half * p3
    tl.store(HRES + row * 16 + i4[:, None] * 4 + j4[None, :], y)


@triton.jit
def _hcw_xbar_kernel(H, HPRECM, XBAR, XSQ, C: tl.constexpr, BC: tl.constexpr,
                     NCH: tl.constexpr, HAS_SQ: tl.constexpr):
    s = tl.program_id(0)
    row = tl.program_id(1)
    offs = s * BC + tl.arange(0, BC)
    hb = H + row * (4 * C)
    g0 = tl.load(HPRECM + row * 4 + 0)
    g1 = tl.load(HPRECM + row * 4 + 1)
    g2 = tl.load(HPRECM + row * 4 + 2)
    g3 = tl.load(HPRECM + row * 4 + 3)
    acc = g0 * tl.load(hb + 0 * C + offs) + g1 * tl.load(hb + 1 * C + offs) \
        + g2 * tl.load(hb + 2 * C + offs) + g3 * tl.load(hb + 3 * C + offs)
    tl.store(XBAR + row * C + offs, acc)
    if HAS_SQ:
        tl.store(XSQ + row * NCH + s, tl.sum(acc * acc, 0))


@triton.jit
def _hcw_nout_kernel(XBAR, XSQ, NW, NOUT, C: tl.constexpr, BC: tl.constexpr,
                     NCH: tl.constexpr, neps, snout):
    s = tl.program_id(0)
    row = tl.program_id(1)
    sq = tl.sum(tl.load(XSQ + row * NCH + tl.arange(0, NCH)), 0)
    rn = 1.0 / tl.sqrt(sq / C + neps)
    offs = s * BC + tl.arange(0, BC)
    xv = tl.load(XBAR + row * C + offs)
    nw = tl.load(NW + offs)
    tl.store(NOUT + row * snout + offs, (xv * rn) * nw)


@triton.jit
def _hcw_post_kernel(HRES, HPOSTROW, H, Y, OUT, TERM,
                     C: tl.constexpr, BC: tl.constexpr, NCH: tl.constexpr,
                     HAS_TERM: tl.constexpr):
    pid = tl.program_id(0)                  # i * NCH + s
    row = tl.program_id(1)
    i = pid // NCH
    s = pid % NCH
    offs = s * BC + tl.arange(0, BC)
    hb = H + row * (4 * C)
    rb = HRES + row * 16 + i * 4
    acc = tl.load(rb + 0) * tl.load(hb + 0 * C + offs) \
        + tl.load(rb + 1) * tl.load(hb + 1 * C + offs) \
        + tl.load(rb + 2) * tl.load(hb + 2 * C + offs) \
        + tl.load(rb + 3) * tl.load(hb + 3 * C + offs)
    acc += tl.load(HPOSTROW + row * 4 + i) * tl.load(Y + row * C + offs)
    if HAS_TERM:
        acc += tl.load(TERM + row * C + offs)
    tl.store(OUT + row * (4 * C) + i * C + offs, acc)


def hc_premap_wide(h: Tensor, w_proj: Tensor, bias: Tensor, tau: float,
                   alpha: float, iters: int, eps: float,
                   hres: Tensor, hpostrow: Tensor, hprecm: Tensor,
                   norm_w: Tensor | None = None, norm_eps: float = 1e-6,
                   norm_out: Tensor | None = None, norm_stride: int = 0) -> Tensor:
    """Wide-carrier HC pre-mapping. h [B,S,4,C] fp32 contiguous; w_proj [48, 4C]
    bf16; hres/hpostrow/hprecm are caller scratch ([B,S,4,4]/[B,S,4]/[B,S,4] fp32).
    Returns xbar [B,S,C] fp32 (+ optional fused RMSNorm(xbar)·nw strided store)."""
    B, S, N, C = h.shape
    assert N == 4, "wide HC path is n=4 only"
    rows = B * S
    NC = N * C
    KS = 16
    BI = NC // KS
    assert NC % KS == 0
    dev = h.device
    part = torch.empty(rows, 48, KS, device=dev, dtype=torch.float32)
    ssq = torch.empty(rows, KS, device=dev, dtype=torch.float32)
    _hcw_gemv_kernel[(48 * KS, rows)](h, w_proj, ssq, part, IDIM=NC, KS=KS,
                                      BI=BI, O=48, num_stages=2, num_warps=4)
    _hcw_map_kernel[(rows,)](part, bias, ssq, hres, hpostrow, hprecm,
                             KS=KS, NC=NC, TAU=float(tau), ALPHA=float(alpha),
                             ITERS=int(iters), EPS=float(eps),
                             num_stages=1, num_warps=1)
    xbar = torch.empty(B, S, C, device=dev, dtype=h.dtype)
    has_nout = norm_out is not None
    BC = 1024
    NCH = C // BC
    assert C % BC == 0
    xsq = torch.empty(rows, NCH, device=dev, dtype=torch.float32) if has_nout else xbar
    _hcw_xbar_kernel[(NCH, rows)](h, hprecm, xbar, xsq, C=C, BC=BC, NCH=NCH,
                                  HAS_SQ=has_nout, num_stages=1, num_warps=4)
    if has_nout:
        _hcw_nout_kernel[(NCH, rows)](xbar, xsq, norm_w, norm_out, C=C, BC=BC,
                                      NCH=NCH, neps=norm_eps, snout=norm_stride,
                                      num_stages=1, num_warps=4)
    return xbar


def hc_post_wide(hres: Tensor, hpostrow: Tensor, h: Tensor, y: Tensor,
                 term: Tensor | None = None) -> Tensor:
    """Wide-carrier HC post: out[i] = Σ_j Hres[i,j]·h[j] + Hpost_row[i]·y (+term)."""
    B, S, N, C = h.shape
    assert N == 4
    rows = B * S
    out = torch.empty(B, S, N, C, device=h.device, dtype=h.dtype)
    BC = 1024
    NCH = C // BC
    _hcw_post_kernel[(N * NCH, rows)](
        hres, hpostrow, h, y.view(rows, C), out,
        term.view(rows, C) if term is not None else y,
        C=C, BC=BC, NCH=NCH, HAS_TERM=term is not None,
        num_stages=1, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ternary_gemv_rs — row-scaled 2-bit ternary GEMV (stacked per-layer-γ tensors)
# ─────────────────────────────────────────────────────────────────────────────
#
# The deploy stack's x0-inject / value-embed / lm_mixer projections are
# PackedTernaryLinear (exact γ·{-1,0,+1} per LAYER). The engine stacks 43 x0
# projections into one GEMV — per-tensor-γ ternary_gemv can't serve a stack of
# DIFFERENT γs, so this variant reads a per-output-row fp32 scale (γ_layer
# expanded). 2-bit codes = 8× less traffic than the bf16 stack it replaces
# (W_x0: 1.92 GB → 0.24 GB per token). Same strip packing as ternary_pack.


@triton.jit
def _ternary_rs_kernel(X, W2, RS, OUT, I: tl.constexpr, O: tl.constexpr,
                       BI4: tl.constexpr, BO: tl.constexpr, sx_b, so_b):
    pid_o = tl.program_id(0)
    b = tl.program_id(1)
    offs_o = pid_o * BO + tl.arange(0, BO)
    I4: tl.constexpr = I // 4
    acc = tl.zeros((BO,), dtype=tl.float32)
    for j0 in range(0, I4, BI4):
        offs_j = j0 + tl.arange(0, BI4)
        mj = offs_j < I4
        p = tl.load(W2 + offs_o[:, None] * I4 + offs_j[None, :],
                    mask=mj[None, :], other=0)
        for st in tl.static_range(4):
            ii = st * I4 + offs_j
            x = tl.load(X + b * sx_b + ii, mask=mj, other=0.0)
            w = ((p >> (2 * st)) & 3).to(tl.float32) - 1.0
            w = tl.where(mj[None, :], w, 0.0)
            acc += tl.sum(w * x[None, :], 1)
    tl.store(OUT + b * so_b + offs_o, acc * tl.load(RS + offs_o))


def ternary_gemv_rs(x: Tensor, codes: Tensor, row_scale: Tensor) -> Tensor:
    """x [B, I] fp32 (row-contiguous); codes [O, I/4] strip-packed uint8;
    row_scale [O] fp32. Returns [B, O] fp32."""
    O, I4 = codes.shape
    I = I4 * 4
    B = x.shape[0]
    out = torch.empty(B, O, device=x.device, dtype=torch.float32)
    BO = 8 if O % 8 == 0 else 2                  # ctx_w=2730 stacks: 2 | 2730
    assert O % BO == 0, (O,)
    BI4 = min(triton.next_power_of_2(I4), 512)
    _ternary_rs_kernel[(O // BO, B)](
        x, codes, row_scale, out, I=I, O=O, BI4=BI4, BO=BO,
        sx_b=x.stride(0), so_b=out.stride(0), num_stages=2, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ring-buffer step metadata + staging-row commit (kills the per-step rolls)
# ─────────────────────────────────────────────────────────────────────────────
#
# The engine's window rings and x-history buffers were shifted with torch.roll +
# copy_ every token (4× the state bytes; 1.7 ms/tok at d=8192). Ring addressing
# removes the window roll entirely (front_post stages k/v at slot p%WWIN, the
# validity mask excludes the slot being overwritten) and reduces the x-history
# roll to ONE staging-row commit (2× one row instead of 4× the whole buffer).
# X layout: [B, WR+1, d] — rows 0..WR-1 = ring of PAST tokens (row q%WR holds
# x_q), row WR = fixed staging row for the CURRENT token (so the HC premap's
# fused norm-store keeps its host-computable address). All consumers read via
# XOFF ring-row indices computed once per step by _ring_meta_kernel.


@triton.jit
def _ring_meta_kernel(POS, WMASK, XC, XH, XE,
                      WWIN: tl.constexpr, WXC: tl.constexpr, WXH: tl.constexpr):
    p = tl.load(POS)
    s = tl.arange(0, WWIN)
    # chronological j-space mask (j ↔ position p-WWIN+1+j; decode_attn rotates
    # ring slots into this order): EXACTLY the pre-ring formula — valid j in
    # [w1 - min(p, w1), w1) with w1 = WWIN-1; j = w1 (current position) invalid.
    w1 = WWIN - 1
    nv = tl.minimum(p, w1)
    valid = (s >= (w1 - nv)) & (s < w1)
    tl.store(WMASK + s, valid.to(tl.int8))
    i8 = tl.arange(0, 8)
    # x positions p-6..p-1 → ring rows; current token → staging row WXC/WXH.
    tl.store(XC + i8, tl.where(i8 == 6, WXC, (p - 6 + i8) % WXC).to(tl.int32),
             mask=i8 < 7)
    tl.store(XH + i8, tl.where(i8 == 6, WXH, (p - 6 + i8) % WXH).to(tl.int32),
             mask=i8 < 7)
    tl.store(XE + i8, tl.where(i8 == 7, WXC, (p - 7 + i8) % WXC).to(tl.int32))


@triton.jit
def _ring_commit_kernel(X, POSP, WR: tl.constexpr, D: tl.constexpr,
                        BD: tl.constexpr, s_row):
    # X stacked [N·B, WR+1, D] rows: commit staging row WR → ring row (p-1)%WR.
    nb = tl.program_id(0)
    ch = tl.program_id(1)
    tgt = ((tl.load(POSP) - 1) % WR).to(tl.int32)
    offs = ch * BD + tl.arange(0, BD)
    m = offs < D
    base = X + nb * ((WR + 1) * s_row)
    v = tl.load(base + WR * s_row + offs, mask=m, other=0)
    tl.store(base + tgt * s_row + offs, v, mask=m)


def ring_meta(pos_dev: Tensor, wmask: Tensor, xoff_csa: Tensor, xoff_hca: Tensor,
              xoff_emit: Tensor, wwin: int, wxc: int, wxh: int) -> None:
    _ring_meta_kernel[(1,)](pos_dev, wmask, xoff_csa, xoff_hca, xoff_emit,
                            WWIN=wwin, WXC=wxc, WXH=wxh,
                            num_stages=1, num_warps=4)


def ring_commit(x_all: Tensor, pos_dev: Tensor) -> None:
    """x_all [N, B, WR+1, D] (contiguous rows): staging row → ring row (p-1)%WR."""
    N, B, W1, D = x_all.shape
    BD = min(triton.next_power_of_2(D), 2048)
    _ring_commit_kernel[(N * B, triton.cdiv(D, BD))](
        x_all, pos_dev, WR=W1 - 1, D=D, BD=BD, s_row=x_all.stride(2),
        num_stages=1, num_warps=4)


@triton.jit
def _route_flags_kernel(GATES, RLO, RHI, CLO, CHI, RACT, CACT,
                        NR: tl.constexpr, NC: tl.constexpr, BL: tl.constexpr):
    """ReMoE gather flags from the per-token cluster gates (B=1): a gate_up
    output block-row / down input block-column is ACTIVE iff either of the (≤2)
    neuron-clusters it spans has gate > 0. One tiny launch per routed MLP."""
    i = tl.arange(0, BL)
    mr = i < NR
    glo = tl.load(GATES + tl.load(RLO + i, mask=mr, other=0), mask=mr, other=0.0)
    ghi = tl.load(GATES + tl.load(RHI + i, mask=mr, other=0), mask=mr, other=0.0)
    tl.store(RACT + i, ((glo > 0) | (ghi > 0)).to(tl.int32), mask=mr)
    mc = i < NC
    clo = tl.load(GATES + tl.load(CLO + i, mask=mc, other=0), mask=mc, other=0.0)
    chi = tl.load(GATES + tl.load(CHI + i, mask=mc, other=0), mask=mc, other=0.0)
    tl.store(CACT + i, ((clo > 0) | (chi > 0)).to(tl.int32), mask=mc)


def route_flags(gates: Tensor, rlo: Tensor, rhi: Tensor, clo: Tensor, chi: Tensor,
                ract: Tensor, cact: Tensor) -> None:
    """gates [1, n_clusters] fp32; rlo/rhi [NR], clo/chi [NC] int32 cluster spans;
    ract/cact int32 out buffers (persistent scratch)."""
    NR, NC = rlo.numel(), clo.numel()
    BL = triton.next_power_of_2(max(NR, NC))
    _route_flags_kernel[(1,)](gates, rlo, rhi, clo, chi, ract, cact,
                              NR=NR, NC=NC, BL=BL, num_stages=1, num_warps=4)
