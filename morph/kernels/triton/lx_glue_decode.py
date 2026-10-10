"""The LXTUL fused decoder's token-step glue: the ENTRY (embedding, bigram, injection terms,
the first carrier) in 2 launches, the prelude-to-coda hand-off in 1, and the EXIT (readout,
head, pointer mixture, log-softmax, argmax) in 9.

Entry and hand-off (replacing about 38 small PyTorch kernels, about 58 us at B = 1):

  E1 ``tok_entry``     per row and channel tile: the token's embedding row (the head table's
                       row: int code x row scale | Lorentz block, or the fp32 table), the
                       bigram row (the hashed (prev, tok) key into the cached int6 table),
                       the span-buffer writes, the injection terms OUTSIDE the ctx slice
                       (``lam_l * bigram``), the first carrier on those channels, the RoPE rows
                       (the model's own ``lm_weight()`` rows equal ``m.embed(ids)`` to the bit
                       at B = 16 and within 1 ulp (3e-8) on 3061 of 49169 Lorentz rows at
                       B = 1, which is the reference's own batch-shape noise: measured
                       2026-10-10, /home/wolfe/morph-scratch/megadec/glue/check_embed_rows.py)
  E2 ``tok_x0``        the stacked ternary x0 GEMV of every layer and the ctx-slice terms
                       ``lam_l * bigram + scale_l * x0f``, the first carrier's ctx channels
  M1 ``tok_mid``       per row: ``input_norm`` of each stream, the span's register input
                       (stream mean), the coda's entry carrier ``e + term``

``LxFusedDecoder._token_step`` (``morph/inference/lx_fused_decode.py``, ``layers="fused"``)
ends every token step with the model's readout, the tied head, the TULPointer copy mixture
and the greedy pick. Written with PyTorch ops that is 59 small kernels (about 97 us of
device time at B = 1 beside the head itself, measured 2026-10-10 with torch.profiler on the
LXTUL-pointer step-5000 checkpoint). Here it is:

  X1 ``readout_gemv``   stream mean x lm_scale, then the ternary ``lm_mixer`` GEMV
  X2 ``rmsnorm_rows``   final_norm (``fused_decode_step``, unchanged) -> xh
  X3 the head           ``lm_head_codes.head_codes_logits`` (or the table GEMV) -> logits
  X4 ``ptr_proj``       the pointer's q, k and gate rows of xh, one stacked GEMV
  X5 ``ptr_attn``       per row: the candidate update, q / k RMS, scores against the key
                        cache, the softmax with the null key, the gate softmax -> w0, beta;
                        stores the new key
  X6 ``row_lse``        per row: max and sum of exp over the logits (slot column excluded)
  X7 ``ptr_scatter``    per candidate: p = sum of beta over the keys with the same token,
                        written into a dense [B, V] buffer (deterministic: duplicates write
                        the same bits; no atomics)
  X8 ``mix``            per vocab tile: log-softmax, ``log(w0 exp(lm) + p)``, the store into
                        the step's log-prob row, the dense buffer zeroed for the next step,
                        a partial argmax
  X9 ``argmax_rows``    the partials -> the greedy token (first index among equal maxima,
                        ``torch.argmax``'s rule)

Function: each kernel is the formula of the PyTorch code it replaces
(``LxFusedDecoder._readout`` / ``_logp`` / ``_pointer``, ``torch.log_softmax``,
``torch.argmax``) with fp32 sums in another order. Two elementwise functions differ by
implementation, not by formula: ``rsqrt`` (libdevice here, CUDA's ``rsqrtf`` in torch) and
``exp`` / ``log`` (libdevice here); both are within a few ulp of the exact value. The
gates that bound the total: tests/test_lx_glue_decode.py (each kernel against a float64
reference) and tests/test_lx_fused_decode.py (the decoder's log-prob rows against the
graphed decoder).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from morph.kernels.triton.lx_layer_decode import HC_NO, PART_TILE, _NOP, _hc_part_row, _tdec

__all__ = ["tok_entry", "tok_x0", "tok_mid", "readout_gemv", "ptr_proj", "ptr_attn",
           "row_lse", "ptr_scatter", "mix_logp", "argmax_rows", "MIX_BLOCK", "cell_norm",
           "route", "loop_entry", "prefix_write"]

# morph/model/embeddings.py BigramEmbedding.compute's hash: (A * tok ^ B * prev) % hash_vocab
# (int64; both products are below 2**63 for ids below 2**31, and non-negative)
from morph.model.embeddings import _BIGRAM_HASH_A, _BIGRAM_HASH_B  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# E1 tok_entry — embedding + bigram rows, span buffers, non-ctx terms, first carrier
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _tok_entry_kernel(TOK, PREV, POS, J, CODES, SC, WL, BGT, LAM, COSF, SINF,
                      X2, BG2, SPE, SPB, TERMS, H0, COS, SIN,
                      HV, HA, HB, CS, CEX, B,
                      IS_CODES: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
                      CE: tl.constexpr, NL: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                      BC: tl.constexpr):
    """Row b, channels t*BC..: x = emb(tok) (IS_CODES: fp32(CODES[tok, c]) * SC[tok] for
    c < CE, WL[tok, c - CE] after; else the fp32 table CODES[tok, c]), bg = BGT[hash],
    SPE / SPB[b, j] = x / bg, TERMS[l, b, c] = LAM[l] * bg outside [CS, CEX), H0[b, i, c] =
    x + TERMS[0, b, c] on those channels; tile 0 also writes the RoPE rows of pos."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    c = t * BC + tl.arange(0, BC)
    tok = tl.load(TOK + b)
    prev = tl.load(PREV + b)
    j = tl.load(J + b)
    if IS_CODES:
        ke = c < CE
        we = tl.load(CODES + tok * CE + c, mask=ke, other=0)
        wl = tl.load(WL + tok * (C - CE) + (c - CE), mask=c >= CE, other=0.0)
        x = tl.where(ke, we.to(tl.float32) * tl.load(SC + tok), wl)
    else:
        x = tl.load(CODES + tok * C + c)
    hsh = ((tok * HA) ^ (prev * HB)) % HV
    bg = tl.load(BGT + hsh * C + c)
    tl.store(X2 + b * C + c, x)
    tl.store(BG2 + b * C + c, bg)
    tl.store(SPE + (b * P + j) * C + c, x)
    tl.store(SPB + (b * P + j) * C + c, bg)
    out = (c < CS) | (c >= CEX)
    for l in tl.static_range(NL):
        tv = tl.load(LAM + l) * bg
        tl.store(TERMS + (l * B + b) * C + c, tv, mask=out)
        if l == 0:
            hv = x + tv
            for i in tl.static_range(N):
                tl.store(H0 + (b * N + i) * C + c, hv, mask=out)
    if t == 0:
        pos = tl.load(POS + b)
        d = tl.arange(0, D)
        tl.store(COS + b * D + d, tl.load(COSF + pos * D + d))
        tl.store(SIN + b * D + d, tl.load(SINF + pos * D + d))


def tok_entry(tok: Tensor, prev: Tensor, pos: Tensor, j: Tensor, table: tuple,
              bg_table: Tensor, lambdas: Tensor, cos_flat: Tensor, sin_flat: Tensor,
              cs: int, ce: int, x2: Tensor, bg2: Tensor, span_emb: Tensor, span_bg: Tensor,
              terms: Tensor, h0: Tensor, cos: Tensor, sin: Tensor) -> None:
    """``table`` = (codes [V, CE] int8, scale [V], w_lor [V, C - CE]) (``head_codes_pack``)
    or (w [V, C] fp32,). ``tok``, ``prev``, ``pos``, ``j`` [B] int64; ``bg_table`` [HV, C]
    fp32 (the bigram embedding's forward weight); ``lambdas`` [NL]; outputs ``x2``, ``bg2``
    [B, C], ``span_emb`` / ``span_bg`` [R, P, C] (row b, entry j), ``terms`` [NL, B, C]
    (channels outside [cs, ce)), ``h0`` [B, N, C] (same channels), ``cos`` / ``sin`` [B, D]."""
    B, C = x2.shape
    NL = lambdas.shape[0]
    N = h0.shape[1]
    P = span_emb.shape[1]
    D = cos.shape[1]
    HV = bg_table.shape[0]
    assert terms.shape == (NL, B, C) and h0.shape == (B, N, C) and bg2.shape == (B, C)
    assert span_emb.shape[2] == C and span_bg.shape == span_emb.shape
    for t_ in (x2, bg2, span_emb, span_bg, terms, h0, cos, sin, bg_table, cos_flat, sin_flat):
        assert t_.is_contiguous() and t_.dtype == torch.float32
    for t_ in (tok, prev, pos, j):
        assert t_.shape == (B,) and t_.dtype == torch.int64 and t_.is_contiguous()
    if len(table) == 3:
        codes, sc, wl = table
        CE = codes.shape[1]
        assert codes.dtype == torch.int8 and wl.shape == (codes.shape[0], C - CE)
    else:
        (codes,) = table
        sc = wl = codes
        CE = C
        assert codes.dtype == torch.float32 and codes.shape[1] == C
    BC = 256
    assert C % BC == 0 and cos_flat.shape[1] == D
    _tok_entry_kernel[(B, C // BC)](
        tok, prev, pos, j, codes, sc, wl, bg_table, lambdas, cos_flat, sin_flat,
        x2, bg2, span_emb, span_bg, terms, h0, cos, sin,
        HV, _BIGRAM_HASH_A, _BIGRAM_HASH_B, cs, ce, B,
        IS_CODES=len(table) == 3, N=N, C=C, CE=CE, NL=NL, P=P, D=D, BC=BC,
        num_stages=1, num_warps=2)


# ─────────────────────────────────────────────────────────────────────────────
# E2 tok_x0 — stacked x0 GEMV, ctx-slice terms, first carrier's ctx channels
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _tok_x0_kernel(X2, BG2, W2, RS, X0S, LAM, VE, TERMS, H0, B, CS,
                   N: tl.constexpr, C: tl.constexpr, CW: tl.constexpr, NVE: tl.constexpr,
                   BO: tl.constexpr, BK: tl.constexpr):
    """x0f[o] = RS[o] * sum_i code(W2)[o, i] * x2[b, i] for o = l * CW + k; then
    TERMS[l, b, CS + k] = LAM[l] * bg2[b, CS + k] + (x0f[o] * X0S[l] (+ VE[l, b, k] for
    the value-embedding layers l < NVE)) (``_terms``'s order: the ctx part first, then
    added to the bigram term), and for l = 0 the carrier H0 = x2 + term."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    I4: tl.constexpr = C // 4
    o = t * BO + tl.arange(0, BO)
    acc = tl.zeros((BO, BK), tl.float32)
    for j0 in range(0, I4, BK):
        jj = j0 + tl.arange(0, BK)
        p = tl.load(W2 + o[:, None] * I4 + jj[None, :]).to(tl.int32)
        for st in tl.static_range(4):
            acc += _tdec(p, st) * tl.load(X2 + b * C + st * I4 + jj)[None, :]
    x0f = tl.sum(acc, 1) * tl.load(RS + o)
    l = o // CW
    c = CS + o % CW
    ctx = x0f * tl.load(X0S + l)
    if NVE > 0:
        ctx = tl.where(l < NVE, ctx + tl.load(VE + (l * B + b) * CW + o % CW, mask=l < NVE,
                                              other=0.0), ctx)
    tv = tl.load(LAM + l) * tl.load(BG2 + b * C + c) + ctx
    tl.store(TERMS + (l * B + b) * C + c, tv)
    first = l == 0
    hv = tl.load(X2 + b * C + c, mask=first, other=0.0) + tv
    for i in tl.static_range(N):
        tl.store(H0 + (b * N + i) * C + c, hv, mask=first)


def tok_x0(x2: Tensor, bg2: Tensor, codes: Tensor, row_scale: Tensor, x0_scales: Tensor,
           lambdas: Tensor, cs: int, terms: Tensor, h0: Tensor, ve: Tensor | None) -> None:
    """``codes`` [NL * CW, C/4] (the stacked ternary x0 projections, ``ternary_pack``),
    ``row_scale`` [NL * CW]; ``x0_scales`` [NL]; ``ve`` [NVE, B, CW] the value-embedding
    signals of layers 0..NVE-1 (``LxFusedDecoder._ve_signal``) or None; writes ``terms``
    [NL, B, C] on [cs, cs+CW) and ``h0`` [B, N, C] on the same channels."""
    B, C = x2.shape
    NL = lambdas.shape[0]
    O = codes.shape[0]
    CW = O // NL
    N = h0.shape[1]
    assert O == NL * CW and codes.shape[1] == C // 4 and row_scale.shape == (O,)
    assert x0_scales.numel() == NL and x0_scales.is_contiguous() and terms.shape == (NL, B, C)
    BO, BK = 8, 64
    assert O % BO == 0 and (C // 4) % BK == 0
    NVE = 0 if ve is None else ve.shape[0]
    if ve is not None:
        assert ve.shape == (NVE, B, CW) and ve.is_contiguous() and NVE <= NL
    _tok_x0_kernel[(O // BO, B)](x2, bg2, codes, row_scale, x0_scales, lambdas,
                                 ve if ve is not None else x2, terms, h0, B, cs, N=N, C=C,
                                 CW=CW, NVE=NVE, BO=BO, BK=BK, num_stages=2, num_warps=4)


# ─────────────────────────────────────────────────────────────────────────────
# M1 tok_mid — input_norm per stream, the register's stream mean, the coda carrier
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _tok_mid_kernel(H, NW, TERM, XC, XN, J, eps, sterm,
                    HAS_XN: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
                    P: tl.constexpr, BC: tl.constexpr):
    """e_i = (h_i * (1 / sqrt(mean(h_i^2) + eps))) * w (``_rmsnorm_kernel``'s order) for the
    N streams of row b; XN[b, j] = mean_i e_i (HAS_XN); XC[b, i] = e_i + TERM[b]."""
    b = tl.program_id(0)
    i = tl.arange(0, N)
    c = tl.arange(0, BC)
    mc = c < C
    h = tl.load(H + (b * N + i[:, None]) * C + c[None, :], mask=mc[None, :], other=0.0)
    r = 1.0 / tl.sqrt(tl.sum(h * h, 1) / C + eps)
    e = (h * r[:, None]) * tl.load(NW + c, mask=mc, other=0.0)[None, :]
    if HAS_XN:
        j = tl.load(J + b)
        tl.store(XN + (b * P + j) * C + c, tl.sum(e, 0) / N, mask=mc)
    tv = tl.load(TERM + b * sterm + c, mask=mc, other=0.0)
    tl.store(XC + (b * N + i[:, None]) * C + c[None, :], e + tv[None, :], mask=mc[None, :])


def tok_mid(h: Tensor, norm_w: Tensor, eps: float, term: Tensor, xc: Tensor,
            span_xn: Tensor | None, j: Tensor) -> None:
    """``h`` [B, N, C] the prelude's carrier; ``term`` [B, C] (row-contiguous) the coda's
    first injection term; ``xc`` [B, N, C]; ``span_xn`` [R, P, C] or None."""
    B, N, C = h.shape
    assert h.is_contiguous() and xc.shape == h.shape and xc.is_contiguous()
    assert term.shape == (B, C) and term.stride(1) == 1 and j.shape == (B,)
    has = span_xn is not None
    P = span_xn.shape[1] if has else 1
    _tok_mid_kernel[(B,)](h, norm_w, term, xc, span_xn if has else h, j, eps, term.stride(0),
                          HAS_XN=has, N=N, C=C, P=P, BC=triton.next_power_of_2(C),
                          num_stages=1, num_warps=4)

MIX_BLOCK = 2048        # vocab entries per X8 CTA (the partial argmax count is V / this)


# ─────────────────────────────────────────────────────────────────────────────
# X1 readout_gemv — stream mean x lm_scale -> ternary lm_mixer GEMV
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _readout_kernel(H, SC, W2, RS, OUT, sh, N: tl.constexpr, C: tl.constexpr,
                    BO: tl.constexpr, BK: tl.constexpr):
    """OUT[b, o] = RS[o] * sum_i code(W2)[o, i] * xs[b, i] with
    ``xs = (h_0 + h_1 + h_2 + h_3) / 4 * SC`` (``x.mean(dim=1) * lm_scale``, N = 4 streams).
    The [BO, BK] products accumulate elementwise and reduce once."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    I4: tl.constexpr = C // 4
    o = t * BO + tl.arange(0, BO)
    acc = tl.zeros((BO, BK), tl.float32)
    hb = H + b * sh
    for j0 in range(0, I4, BK):
        j = j0 + tl.arange(0, BK)
        p = tl.load(W2 + o[:, None] * I4 + j[None, :]).to(tl.int32)
        for st in tl.static_range(4):
            i = st * I4 + j
            s4 = (tl.load(hb + i) + tl.load(hb + C + i)) + (tl.load(hb + 2 * C + i)
                                                            + tl.load(hb + 3 * C + i))
            xs = (s4 * 0.25) * tl.load(SC + i)
            acc += _tdec(p, st) * xs[None, :]
    tl.store(OUT + b * C + o, tl.sum(acc, 1) * tl.load(RS + o))


def readout_gemv(h: Tensor, lm_scale: Tensor, codes: Tensor, row_scale: Tensor,
                 out: Tensor) -> Tensor:
    """``h`` [B, 4, C] carrier rows (row stride any, inner contiguous); ``codes`` [C, C/4]
    (``ternary_pack`` of ``lm_mixer.mix.weight``), ``row_scale`` [C]; ``out`` [B, C]."""
    B, N, C = h.shape
    assert N == 4 and h.stride(2) == 1 and h.stride(1) == C and h.dtype == torch.float32
    assert codes.shape == (C, C // 4) and codes.is_contiguous() and row_scale.shape == (C,)
    assert lm_scale.shape == (C,) and out.shape == (B, C) and out.is_contiguous()
    BO, BK = 8, 64
    assert C % BO == 0 and (C // 4) % BK == 0
    _readout_kernel[(C // BO, B)](h, lm_scale, codes, row_scale, out, h.stride(0), N=N, C=C,
                                  BO=BO, BK=BK, num_stages=2, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# X4 ptr_proj — q | k | gate rows of xh (one stacked fp32 GEMV)
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _ptr_proj_kernel(X, W, BIAS, OUT, OPN: tl.constexpr, NQK: tl.constexpr,
                     C: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    """OUT[b, o] = sum_k W[o, k] X[b, k] (+ BIAS[o - NQK] for the gate rows o >= NQK;
    the q / k rows have no bias and get no + 0.0)."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    o = t * BO + tl.arange(0, BO)
    mo = o < OPN
    acc = tl.zeros((BO, BK), tl.float32)
    for k0 in range(0, C, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(W + o[:, None] * C + k[None, :], mask=mo[:, None], other=0.0)
        acc += w * tl.load(X + b * C + k)[None, :]
    y = tl.sum(acc, 1)
    gb = tl.load(BIAS + (o - NQK), mask=(o >= NQK) & mo, other=0.0)
    y = tl.where(o >= NQK, y + gb, y)
    tl.store(OUT + b * OPN + o, y, mask=mo)


def ptr_proj(xh: Tensor, w: Tensor, gate_bias: Tensor, nqk: int, out: Tensor) -> Tensor:
    """``xh`` [B, C]; ``w`` [NQK + G, C] = cat(q.weight, k.weight, gate.weight) fp32;
    ``gate_bias`` [G]; ``out`` [B, NQK + G]."""
    B, C = xh.shape
    OPN = w.shape[0]
    assert w.shape == (OPN, C) and w.is_contiguous() and w.dtype == torch.float32
    assert xh.is_contiguous() and out.shape == (B, OPN) and out.is_contiguous()
    assert gate_bias.shape == (OPN - nqk,) and C % 128 == 0
    BO, BK = 8, 128
    _ptr_proj_kernel[(triton.cdiv(OPN, BO), B)](xh, w, gate_bias, out, OPN=OPN, NQK=nqk, C=C,
                                                BO=BO, BK=BK, num_stages=2, num_warps=4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# X5 ptr_attn — one CTA per row: candidates, q/k RMS, scores, softmaxes -> w0, beta
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _ptr_attn_kernel(PROJ, KEYS, CAND, TI, TIM1, TOK, NULLK, SCALE, S, BETA, W0,
                     eps, NK, HP: tl.constexpr, H: tl.constexpr, DH: tl.constexpr,
                     OPN: tl.constexpr, BN: tl.constexpr, GP: tl.constexpr):
    """``LxFusedDecoder._pointer`` for row b, up to the scatter (X7 does that). S is a
    [B, HP, NK] fp32 scratch for the scores (written in pass 1, read back after a CTA
    barrier). HP = H padded to a power of two (the padded heads are masked off). Every
    pass runs over the key blocks below ti only: a key n >= ti has score -inf and beta 0
    in ``_pointer``, and X7 reads beta below ti only."""
    b = tl.program_id(0)
    hh = tl.arange(0, HP)
    mh = hh < H
    d = tl.arange(0, DH)
    ti = tl.load(TI + b)
    tim1 = tl.load(TIM1 + b)
    tok = tl.load(TOK + b)
    # the previous entry gets its follower (this token); X7 reads the updated candidates
    nn0 = tl.arange(0, BN)
    tl.store(CAND + b * NK + tim1 + nn0, tok + nn0 * 0, mask=nn0 == 0)
    # q, k: _rms over each head's DH channels (x * rsqrt(mean(x^2) + eps))
    pq = tl.load(PROJ + b * OPN + hh[:, None] * DH + d[None, :], mask=mh[:, None], other=0.0)
    pk = tl.load(PROJ + b * OPN + H * DH + hh[:, None] * DH + d[None, :], mask=mh[:, None],
                 other=0.0)
    q = pq * tl.math.rsqrt(tl.sum(pq * pq, 1) / DH + eps)[:, None]
    k = pk * tl.math.rsqrt(tl.sum(pk * pk, 1) / DH + eps)[:, None]
    sc = tl.load(SCALE + hh, mask=mh, other=1.0)
    nk = tl.load(NULLK + hh[:, None] * DH + d[None, :], mask=mh[:, None], other=0.0)
    s_null = tl.sum(q * nk, 1) * sc                                       # [HP]
    # pass 1: scores (masked to n < ti) into S, running max
    m = s_null
    sb = S + b * (HP * NK) + hh[:, None] * NK
    kb = KEYS + b * (H * NK * DH) + hh[:, None, None] * (NK * DH)
    for n0 in range(0, ti, BN):
        n = n0 + tl.arange(0, BN)
        kt = tl.load(kb + n[None, :, None] * DH + d[None, None, :],
                     mask=mh[:, None, None] & (n < ti)[None, :, None], other=0.0)
        s = tl.sum(q[:, None, :] * kt, 2) * sc[:, None]                   # [HP, BN]
        s = tl.where((n < ti)[None, :], s, float("-inf"))
        tl.store(sb + n[None, :], s, mask=mh[:, None] & (n < NK)[None, :])
        m = tl.maximum(m, tl.max(s, 1))
    tl.debug_barrier()
    # pass 2: the softmax denominator over [scores, null]
    zacc = tl.zeros((HP, BN), tl.float32)
    for n0 in range(0, ti, BN):
        n = n0 + tl.arange(0, BN)
        s = tl.load(sb + n[None, :], mask=mh[:, None] & (n < NK)[None, :], other=float("-inf"))
        zacc += tl.exp(s - m[:, None])
    e_null = tl.exp(s_null - m)
    z = tl.sum(zacc, 1) + e_null
    null = e_null / z                                                      # [HP]
    # the gate: softmax over the 1 + H gate logits
    g = tl.arange(0, GP)
    gl = tl.load(PROJ + b * OPN + 2 * H * DH + g, mask=g < H + 1, other=float("-inf"))
    ge = tl.exp(gl - tl.max(gl, 0))
    a = ge / tl.sum(ge, 0)                                                 # [GP]
    a0 = tl.sum(tl.where(g == 0, a, 0.0), 0)
    # a_h for head h = a[1 + h], as an [HP] vector
    ah = tl.sum(tl.where(g[None, :] == hh[:, None] + 1, a[None, :], 0.0), 1)
    w0 = a0 + tl.sum(tl.where(mh, ah * null, 0.0), 0)
    tl.store(W0 + b, w0)
    # pass 3: alpha = e / z, beta_n = sum_h alpha_hn a_h (x valid)
    for n0 in range(0, ti, BN):
        n = n0 + tl.arange(0, BN)
        s = tl.load(sb + n[None, :], mask=mh[:, None] & (n < NK)[None, :], other=float("-inf"))
        alpha = tl.exp(s - m[:, None]) / z[:, None]
        # (``_pointer`` also multiplies by the valid mask; alpha is exp(-inf) = 0 at every
        # n >= ti already, so the product is the identity and is left out)
        beta = tl.sum(tl.where(mh[:, None], alpha * ah[:, None], 0.0), 0)
        tl.store(BETA + b * NK + n, beta, mask=n < NK)
    # this token's key joins the cache after it is read
    tl.store(KEYS + b * (H * NK * DH) + hh[:, None] * (NK * DH) + ti * DH + d[None, :], k,
             mask=mh[:, None])


def ptr_attn(proj: Tensor, keys: Tensor, cand: Tensor, ti: Tensor, ti_m1: Tensor,
             tok: Tensor, null_k: Tensor, scale: Tensor, s_scratch: Tensor, beta: Tensor,
             w0: Tensor, eps: float = 1e-6) -> None:
    """``proj`` [B, 2 H DH + 1 + H] (``ptr_proj``); ``keys`` [B, H, NK, DH] fp32 (the key
    cache, entry ``ti`` written); ``cand`` [B, NK] int64 (entry ``ti_m1`` set to ``tok``);
    ``ti``, ``ti_m1``, ``tok`` [B] int64; ``null_k`` [H, DH] = ``_rms(null_k_bias)``;
    ``scale`` [H] = ``exp(log_scale_bias) / sqrt(DH)``; ``s_scratch`` [B, HP, NK] fp32;
    out: ``beta`` [B, NK], ``w0`` [B]."""
    B, H, NK, DH = keys.shape
    HP = triton.next_power_of_2(H)
    OPN = proj.shape[1]
    assert OPN == 2 * H * DH + 1 + H and proj.is_contiguous() and keys.is_contiguous()
    assert cand.shape == (B, NK) and cand.dtype == torch.int64 and cand.is_contiguous()
    assert s_scratch.shape == (B, HP, NK) and beta.shape == (B, NK) and w0.shape == (B,)
    for t in (ti, ti_m1, tok):
        assert t.shape == (B,) and t.is_contiguous() and t.dtype == torch.int64
    _ptr_attn_kernel[(B,)](proj, keys, cand, ti, ti_m1, tok, null_k, scale, s_scratch, beta,
                           w0, eps, NK, HP=HP, H=H, DH=DH, OPN=OPN, BN=32,
                           GP=triton.next_power_of_2(H + 1), num_stages=1, num_warps=4)


# ─────────────────────────────────────────────────────────────────────────────
# X6 row_lse — per row max and sum exp over the logits, slot column excluded
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _row_lse_kernel(LOG, MX, LS, V, sl, SLOT, BV: tl.constexpr):
    """MX[b] = max_v x_v, LS[b] = log(sum_v exp(x_v - MX[b])) over v != SLOT (the slot
    column is -inf in the decoder's logits: exp(-inf) = 0, so it adds nothing)."""
    b = tl.program_id(0)
    v0 = tl.arange(0, BV)
    mx = tl.full((BV,), float("-inf"), tl.float32)
    for c0 in range(0, V, BV):
        v = c0 + v0
        x = tl.load(LOG + b * sl + v, mask=(v < V) & (v != SLOT), other=float("-inf"))
        mx = tl.maximum(mx, x)
    m = tl.max(mx, 0)
    acc = tl.zeros((BV,), tl.float32)
    for c0 in range(0, V, BV):
        v = c0 + v0
        x = tl.load(LOG + b * sl + v, mask=(v < V) & (v != SLOT), other=float("-inf"))
        acc += tl.exp(x - m)
    tl.store(MX + b, m)
    tl.store(LS + b, tl.log(tl.sum(acc, 0)))


def row_lse(logits: Tensor, slot_col: int, mx: Tensor, ls: Tensor) -> None:
    """``logits`` [B, V] (inner contiguous); writes ``mx`` [B], ``ls`` [B]."""
    B, V = logits.shape
    assert logits.stride(1) == 1 and mx.shape == (B,) and ls.shape == (B,)
    # the kernel's offsets b * stride + v are int32: wrap-free for every lane, the masked
    # tail lanes v < V + BV included (Z3, 2026-10-10)
    assert (B - 1) * logits.stride(0) + V + 4096 < 2 ** 31
    _row_lse_kernel[(B,)](logits, mx, ls, V, logits.stride(0), slot_col, BV=4096,
                          num_stages=2, num_warps=8)


# ─────────────────────────────────────────────────────────────────────────────
# X7 ptr_scatter — p(v) = sum of beta over the keys whose candidate is v
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _ptr_scatter_kernel(CAND, BETA, TI, PD, NK, V, BN: tl.constexpr, BM: tl.constexpr):
    """For each valid key n (n < ti) of this CTA's block: p_n = sum over valid n' with
    cand[n'] == cand[n] of beta[n'] (n' in index order: two keys with the same candidate
    get the same bits), stored at PD[b, max(cand[n], 0)] (``scatter_add`` index rule)."""
    b = tl.program_id(0)
    i = tl.program_id(1)
    ti = tl.load(TI + b)
    n = i * BN + tl.arange(0, BN)
    vn = n < ti
    cn = tl.load(CAND + b * NK + n, mask=vn, other=-1)
    acc = tl.zeros((BN, BM), tl.float32)
    hi = tl.where(i * BN < ti, ti, 0)                  # a block past ti has no valid key
    for m0 in range(0, hi, BM):
        mm = m0 + tl.arange(0, BM)
        vm = mm < ti
        cm = tl.load(CAND + b * NK + mm, mask=vm, other=-2)
        bm = tl.load(BETA + b * NK + mm, mask=vm, other=0.0)
        acc += tl.where((cn[:, None] == cm[None, :]) & vm[None, :], bm[None, :], 0.0)
    p = tl.sum(acc, 1)
    tl.store(PD + b * V + tl.maximum(cn, 0), p, mask=vn)


def ptr_scatter(cand: Tensor, beta: Tensor, ti: Tensor, p_dense: Tensor) -> None:
    """``cand`` [B, NK] int64, ``beta`` [B, NK], ``ti`` [B]; ``p_dense`` [B, V] fp32 (all
    zeros on entry except what this writes; ``mix_logp`` re-zeroes it)."""
    B, NK = cand.shape
    V = p_dense.shape[1]
    assert beta.shape == (B, NK) and p_dense.is_contiguous() and cand.is_contiguous()
    BN = 64
    _ptr_scatter_kernel[(B, triton.cdiv(NK, BN))](cand, beta, ti, p_dense, NK, V, BN=BN,
                                                 BM=64, num_stages=1, num_warps=4)


# ─────────────────────────────────────────────────────────────────────────────
# X8 mix_logp — log-softmax, the pointer mixture, store, partial argmax
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _mix_kernel(LOG, MX, LS, W0, PD, OUT, PAV, PAI, V, sl, so, SLOT, NT,
                HAS_PTR: tl.constexpr, BV: tl.constexpr):
    """lm = (x - max) - log(sum) (``torch.log_softmax``), -inf at SLOT;
    HAS_PTR: out = log(w0 exp(lm) + p) (``torch.log(w0 * lm.exp() + p_ptr)``), and PD is
    zeroed behind the read. Partial argmax per CTA in ``torch.argmax``'s order: the first
    NaN if the tile has one (value NaN), else the maximum and the first index of it."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    v = t * BV + tl.arange(0, BV)
    mv = v < V
    x = tl.load(LOG + b * sl + v, mask=mv, other=0.0)
    lm = (x - tl.load(MX + b)) - tl.load(LS + b)
    lm = tl.where(v == SLOT, float("-inf"), lm)
    if HAS_PTR:
        p = tl.load(PD + b * V + v, mask=mv, other=0.0)
        out = tl.log(tl.load(W0 + b) * tl.exp(lm) + p)
        tl.store(PD + b * V + v, tl.zeros_like(p), mask=mv)
    else:
        out = lm
    tl.store(OUT + b * so + v, out, mask=mv)
    o2 = tl.where(mv, out, float("-inf"))
    nidx = tl.min(tl.where(o2 != o2, v, 2147483647), 0)          # the first NaN lane
    best = tl.max(o2, 0)
    idx = tl.min(tl.where(o2 == best, v, 2147483647), 0)
    hasn = nidx < 2147483647
    tl.store(PAV + b * NT + t, tl.where(hasn, float("nan"), best))
    tl.store(PAI + b * NT + t, tl.where(hasn, nidx, idx))


def mix_logp(logits: Tensor, mx: Tensor, ls: Tensor, w0: Tensor | None,
             p_dense: Tensor | None, slot_col: int, out: Tensor, pav: Tensor,
             pai: Tensor) -> None:
    """``out`` [B, V] = the step's log-prob rows; ``pav`` / ``pai`` [B, cdiv(V, MIX_BLOCK)]
    the partial argmax (value, int32 index). ``w0`` / ``p_dense`` None: no pointer."""
    B, V = logits.shape
    NT = triton.cdiv(V, MIX_BLOCK)
    assert pav.shape == (B, NT) and pai.shape == (B, NT) and pai.dtype == torch.int32
    assert out.stride(1) == 1 and logits.stride(1) == 1
    # the kernel's offsets b * stride + v are int32: wrap-free for every lane, the masked
    # tail lanes v < V + MIX_BLOCK included (Z3, 2026-10-10)
    lim = 2 ** 31 - V - MIX_BLOCK
    assert (B - 1) * logits.stride(0) < lim and (B - 1) * out.stride(0) < lim
    has = w0 is not None
    _mix_kernel[(B, NT)](logits, mx, ls, w0 if has else mx, p_dense if has else logits, out,
                         pav, pai, V, logits.stride(0), out.stride(0), slot_col, NT,
                         HAS_PTR=has, BV=MIX_BLOCK, num_stages=1, num_warps=8)


# ─────────────────────────────────────────────────────────────────────────────
# X9 argmax_rows — the partials -> the token
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _argmax_kernel(PAV, PAI, OUT, NT, BT: tl.constexpr):
    b = tl.program_id(0)
    t = tl.arange(0, BT)
    mt = t < NT
    val = tl.load(PAV + b * NT + t, mask=mt, other=float("-inf"))
    idx = tl.load(PAI + b * NT + t, mask=mt, other=2147483647)
    # torch.argmax's order: the first NaN if any (a NaN partial carries its tile's first
    # NaN, and tiles are in index order), else the first index of the maximum. Without
    # the NaN branch an all-NaN row would give no index (2**31 - 1) and next_tok out of
    # range (Z3, 2026-10-10)
    nidx = tl.min(tl.where((val != val) & mt, idx, 2147483647), 0)
    best = tl.max(val, 0)
    midx = tl.min(tl.where(val == best, idx, 2147483647), 0)
    tl.store(OUT + b, tl.where(nidx < 2147483647, nidx, midx).to(tl.int64))


def argmax_rows(pav: Tensor, pai: Tensor, out: Tensor) -> None:
    """``out`` [B] int64 = the first index of each row's maximum (``torch.argmax``)."""
    B, NT = pav.shape
    assert out.shape == (B,) and out.dtype == torch.int64
    _argmax_kernel[(B,)](pav, pai, out, NT, BT=triton.next_power_of_2(NT), num_stages=1,
                         num_warps=1)


# ─────────────────────────────────────────────────────────────────────────────
# Slot-loop glue: per pass L1 cell_norm -> L2 route -> L3 loop_entry (the next pass's
# carrier + its first HC partials); at the exit L4 prefix_write
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _cell_norm_kernel(H, NW, HN, LN, eps, LSEL: tl.constexpr, N: tl.constexpr,
                      C: tl.constexpr, BC: tl.constexpr):
    """Cell q: hn_i = (h_i * (1 / sqrt(mean(h_i^2) + eps))) * w per stream i (the cell
    norm, ``rmsnorm_rows``' order); LSEL: cr = mean_i hn_i (``hs.mean(dim=2)``) and
    LN[q] = (cr - mean(cr)) * rsqrt(var(cr) + 1e-5) (``F.layer_norm`` without affine)."""
    q = tl.program_id(0)
    i = tl.arange(0, N)
    c = tl.arange(0, BC)
    mc = c < C
    h = tl.load(H + (q * N + i[:, None]) * C + c[None, :], mask=mc[None, :], other=0.0)
    r = 1.0 / tl.sqrt(tl.sum(h * h, 1) / C + eps)
    hn = (h * r[:, None]) * tl.load(NW + c, mask=mc, other=0.0)[None, :]
    tl.store(HN + (q * N + i[:, None]) * C + c[None, :], hn, mask=mc[None, :])
    if LSEL:
        cr = tl.sum(hn, 0) / N
        mu = tl.sum(cr, 0) / C
        dv = tl.where(mc, cr - mu, 0.0)
        var = tl.sum(dv * dv, 0) / C
        tl.store(LN + q * C + c, dv * tl.math.rsqrt(var + 1e-5), mask=mc)


def cell_norm(h: Tensor, norm_w: Tensor, eps: float, hn: Tensor, ln: Tensor | None) -> None:
    """``h`` [Q, N, C] the cells after a pass (Q = nb * M); ``hn`` [Q, N, C] the normed
    cells; ``ln`` [Q, C] the layer-normed stream mean (None: no router)."""
    Q, N, C = h.shape
    assert h.is_contiguous() and hn.shape == h.shape and hn.is_contiguous()
    if ln is not None:
        assert ln.shape == (Q, C) and ln.is_contiguous()
    _cell_norm_kernel[(Q,)](h, norm_w, hn, ln if ln is not None else hn, eps,
                            LSEL=ln is not None, N=N, C=C, BC=triton.next_power_of_2(C),
                            num_stages=1, num_warps=4)


@triton.jit
def _route_part_kernel(LN, WC, XP, VV, SP, M: tl.constexpr, RK: tl.constexpr,
                       C: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    """Cell q, rank tile t: SP[q, t] = sum over the tile's ranks o of
    v[o] * relu((W_c ln_q)[o] + xp_b[o]) (``FanRouter.scores``' sum over the rank, split
    into RK / BR partial sums)."""
    t = tl.program_id(0)
    q = tl.program_id(1)
    b = q // M
    o = t * BR + tl.arange(0, BR)
    acc = tl.zeros((BR, BK), tl.float32)
    for k0 in range(0, C, BK):
        k = k0 + tl.arange(0, BK)
        acc += tl.load(WC + o[:, None] * C + k[None, :]) * tl.load(LN + q * C + k)[None, :]
    a = tl.sum(acc, 1) + tl.load(XP + b * RK + o)
    # torch.relu keeps a NaN; tl.maximum (fmax) would turn it into 0 and make a NaN
    # cell's score finite
    ra = tl.where(a != a, a, tl.maximum(a, 0.0))
    tl.store(SP + q * (RK // BR) + t, tl.sum(ra * tl.load(VV + o), 0))


@triton.jit
def _route_pick_kernel(SP, BIAS, PICK, M: tl.constexpr, NTR: tl.constexpr):
    """Slot b: score_i = sum_t SP[b * M + i, t]; PICK[b] = the first index of the maximum
    of score + bias (``FanRouter.select``'s winner: the bias chooses)."""
    b = tl.program_id(0)
    mi = tl.arange(0, M)
    tt = tl.arange(0, NTR)
    sc = tl.sum(tl.load(SP + (b * M + mi[:, None]) * NTR + tt[None, :]), 1)
    sb = sc + tl.load(BIAS + mi)
    # torch.argmax's order: the first NaN cell if any, else the first index of the max.
    # Without the NaN branch all-NaN scores gave PICK = M, and loop_entry / prefix_write
    # then read the next slot's cells or past the buffer (Z3, 2026-10-10)
    nidx = tl.min(tl.where(sb != sb, mi, M), 0)
    best = tl.max(sb, 0)
    midx = tl.min(tl.where(sb == best, mi, M), 0)
    tl.store(PICK + b, tl.where(nidx < M, nidx, midx).to(tl.int64))


def route(ln: Tensor, w_c: Tensor, xp: Tensor, v: Tensor, bias: Tensor, pick: Tensor,
          sp: Tensor) -> None:
    """``ln`` [nb * M, C]; ``w_c`` [RK, C]; ``xp`` [nb, RK]; ``v`` [RK]; ``bias`` [M];
    ``pick`` [nb] int64; ``sp`` [nb * M, RK / 8] scratch (the partial scores)."""
    nb, RK = xp.shape
    M = bias.shape[0]
    C = ln.shape[1]
    BR, BK = 8, 64
    NTR = RK // BR
    assert ln.shape == (nb * M, C) and w_c.shape == (RK, C) and w_c.is_contiguous()
    assert pick.shape == (nb,) and pick.dtype == torch.int64 and C % BK == 0
    assert RK % BR == 0 and NTR & (NTR - 1) == 0 and M & (M - 1) == 0
    assert xp.is_contiguous() and sp.shape == (nb * M, NTR) and sp.is_contiguous()
    # (rank tile, cell) CTAs: one CTA per slot reading all of W_c took 25 us at nb = 1
    _route_part_kernel[(NTR, nb * M)](ln, w_c, xp, v, sp, M=M, RK=RK, C=C, BR=BR, BK=BK,
                                      num_stages=2, num_warps=4)
    _route_pick_kernel[(nb,)](sp, bias, pick, M=M, NTR=NTR, num_stages=1, num_warps=1)


@triton.jit
def _loop_entry_kernel(SRC, PICK, E, A, DT, TERM, WN, OUT, RAWP, lo, hi, sterm,
                       RESET: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                       C: tl.constexpr, BC: tl.constexpr, NT: tl.constexpr,
                       NO: tl.constexpr, NOP: tl.constexpr):
    """Cell q = b * M + i, channel tile t: the pass input h = SRC[b * M + PICK[b]] (RESET,
    ``reset_to_winner``) or SRC[q]; out_s = where(lo <= c < hi, A h_s + dt e_s, h_s) +
    term_b (``_loop_ssm_kernel``'s formula) for the N streams; OUT[q] = out; RAWP[q, t] the
    first core layer's HC-projection partials of out (``hc_partials``)."""
    t = tl.program_id(0)
    q = tl.program_id(1)
    b = q // M
    c = t * BC + tl.arange(0, BC)
    if RESET:
        src = b * M + tl.load(PICK + b)
    else:
        src = q
    inctx = (c >= lo) & (c < hi)
    av = tl.load(A + (c - lo), mask=inctx, other=0.0)
    dv = tl.load(DT + (c - lo), mask=inctx, other=0.0)
    tv = tl.load(TERM + b * sterm + c)
    sb = SRC + src * (N * C) + c
    eb = E + q * (N * C) + c
    ob = OUT + q * (N * C) + c
    h0 = tl.load(sb)
    o0 = tl.where(inctx, av * h0 + dv * tl.load(eb, mask=inctx, other=0.0), h0) + tv
    h1 = tl.load(sb + C)
    o1 = tl.where(inctx, av * h1 + dv * tl.load(eb + C, mask=inctx, other=0.0), h1) + tv
    h2 = tl.load(sb + 2 * C)
    o2 = tl.where(inctx, av * h2 + dv * tl.load(eb + 2 * C, mask=inctx, other=0.0), h2) + tv
    h3 = tl.load(sb + 3 * C)
    o3 = tl.where(inctx, av * h3 + dv * tl.load(eb + 3 * C, mask=inctx, other=0.0), h3) + tv
    tl.store(ob, o0)
    tl.store(ob + C, o1)
    tl.store(ob + 2 * C, o2)
    tl.store(ob + 3 * C, o3)
    _hc_part_row(WN, RAWP, q, t, o0, o1, o2, o3, N, C, BC, NT, NO, NOP)


def loop_entry(src: Tensor, pick: Tensor | None, e: Tensor, A: Tensor, dt: Tensor,
               term: Tensor, lo: int, hi: int, w_next: Tensor, out: Tensor,
               rawp: Tensor, m_cells: int) -> None:
    """``src`` / ``e`` / ``out`` [nb * M, N, C]; ``pick`` [nb] (None: no reset); ``A`` /
    ``dt`` [hi - lo]; ``term`` [nb, C] (row-contiguous); ``w_next`` [48, N * C];
    ``rawp`` [nb * M, C / PART_TILE, 48]."""
    Q, N, C = src.shape
    NT = C // PART_TILE
    assert N == 4 and src.is_contiguous() and e.shape == src.shape and e.is_contiguous()
    assert out.shape == src.shape and out.is_contiguous() and term.stride(1) == 1
    assert term.shape == (Q // m_cells, C) and rawp.shape == (Q, NT, HC_NO)
    assert w_next.shape == (HC_NO, N * C) and w_next.is_contiguous()
    _loop_entry_kernel[(NT, Q)](src, pick if pick is not None else src, e, A, dt, term,
                                w_next, out, rawp, lo, hi, term.stride(0),
                                RESET=pick is not None,
                                M=m_cells, N=N, C=C, BC=PART_TILE, NT=NT, NO=HC_NO,
                                NOP=_NOP, num_stages=1, num_warps=4)


@triton.jit
def _prefix_kernel(HN, PICK, W, OUT, HAS_PICK: tl.constexpr, KP: tl.constexpr,
                   N: tl.constexpr, C: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    """OUT[b, k, s, o] = sum_d HN[b, k, s, d] W[k, d, o] for cell k of slot row b; with
    HAS_PICK only the winner k = PICK[b] is computed and the other cells get 0
    (``hw = hs * onehot(pick)`` then ``matmul(hw[:, k], W_prefix[k])``: every product of
    a masked cell is a zero). Grid (C / BO, nb) with HAS_PICK, (C / BO, nb * KP) without."""
    t = tl.program_id(0)
    g = tl.program_id(1)
    o = t * BO + tl.arange(0, BO)
    s = tl.arange(0, N)
    if HAS_PICK:
        b = g
        k = tl.load(PICK + b)
    else:
        b = g // KP
        k = g % KP
    q = b * KP + k
    acc = tl.zeros((N, BK, BO), tl.float32)
    for d0 in range(0, C, BK):
        d = d0 + tl.arange(0, BK)
        w = tl.load(W + k * (C * C) + d[:, None] * C + o[None, :])          # [BK, BO]
        x = tl.load(HN + (q * N + s[:, None]) * C + d[None, :])              # [N, BK]
        acc += x[:, :, None] * w[None, :, :]
    y = tl.sum(acc, 1)
    if HAS_PICK:
        for kk in tl.static_range(KP):
            tl.store(OUT + ((b * KP + kk) * N + s[:, None]) * C + o[None, :],
                     tl.where(kk == k, y, 0.0))
    else:
        tl.store(OUT + (q * N + s[:, None]) * C + o[None, :], y)


def prefix_write(hn: Tensor, pick: Tensor | None, w_prefix: Tensor, out: Tensor) -> None:
    """``hn`` [nb * Kp, N, C] the normed exit cells (M == Kp); ``pick`` [nb] or None;
    ``w_prefix`` [Kp, C, C] fp32; ``out`` [nb * Kp, N, C] the coda cells' input."""
    Q, N, C = hn.shape
    KP = w_prefix.shape[0]
    assert w_prefix.shape == (KP, C, C) and w_prefix.is_contiguous() and Q % KP == 0
    assert hn.is_contiguous() and out.shape == hn.shape and out.is_contiguous()
    BO, BK = 16, 64
    assert C % BO == 0 and C % BK == 0
    # only the winner's [C, C] block is read (4 MB at C = 1024); BO = 16 spreads it over
    # C / 16 CTAs per slot (a 64-wide tile with one CTA per cell took 17.5 us at nb = 1)
    grid = (C // BO, Q // KP if pick is not None else Q)
    _prefix_kernel[grid](hn, pick if pick is not None else hn, w_prefix, out,
                         HAS_PICK=pick is not None, KP=KP, N=N, C=C, BO=BO, BK=BK,
                         num_stages=2, num_warps=4)
