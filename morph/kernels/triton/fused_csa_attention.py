"""Fused CSA compressed-attention — Triton forward AND backward (sm_120 / Blackwell).

Fuses the sparse top-k compressed-attention core of ``_CCACSAAttention.forward``:
the GATHER of the per-query-position top-k selected compressed blocks ON THE FLY,
the q × C_sel → out_comp masked attention, the validity (future-block) mask, and
the per-head attention sink folded into the online-softmax denominator.

The point of this kernel (the headline win)
--------------------------------------------
The eager path materialises ``C_sel = C_comp[batch_idx, top_idx]`` of shape
[B, S, top_k, D]. At B=32, S=8192, top_k=128, D=32 that is

    32 * 8192 * 128 * 32 * 2 bytes  =  2.0 GiB  (bf16)  *per layer*,

plus the [B,H,S,top_k] scores/weights tensors and autograd-saved copies of all of
them. This kernel NEVER materialises C_sel: each program gathers only the tk block
rows it needs straight from C_comp via top_idx, streams them through an online
softmax, and writes only out_comp [B,H,S,D] + a per-row LSE [B,H,S]. C_sel and the
[B,H,S,tk] scores tensor are eliminated.

Computation fused (byte-faithful to the eager block in attention.py)
--------------------------------------------------------------------
Inputs:
  q            [B, H, S, D]      per query head
  C_comp       [B, n_blocks, D]  compressed block keys/values, SHARED across heads
  top_idx      [B, S, tk]  int   selected block indices (NON-differentiable; the
                                 LightningIndexer gets no gradient from this path)
  invalid_mask [B, S, tk]  bool  True → gathered block is a future (invalid) block
  sink_logits  [H]               per-head learnable sink logit
  scale = D ** -0.5

  C_sel[b,s,t,:]    = C_comp[b, top_idx[b,s,t], :]           # gather on the fly
  scores[b,h,s,t]   = (q[b,h,s] · C_sel[b,s,t]) * scale
  scores            = where(invalid_mask[b,s,t], -inf, scores)
  scores_aug        = concat([scores, sink_logits[h]], -1)
  attn_w            = softmax(scores_aug, fp32)[..., :-1]     # drop sink column
  out[b,h,s,:]      = sum_t attn_w[b,h,s,t] * C_sel[b,s,t,:]

NOTE — no early-query guard (unlike HCA): the sink logit is finite, so the softmax
denominator is exp(sink - m) > 0 for every row, even when ALL tk gathered blocks
are invalid. In that case every attn_w is ~0 and out is ~0 naturally, matching the
eager path exactly (the eager path also has no no_valid guard for CSA).

Design for sm_120 (RTX 5090 / Blackwell)
----------------------------------------
  * num_stages=1, num_warps=8 (consumer Blackwell has NO TMA pipeline).
  * bf16 in/out, fp32 accumulation; softmax entirely in fp32.
  * ONE program = one (b, h, q-row). The gather differs per query position s, so
    a clean 2D Q@C^T matmul over a q-tile is impossible (each row selects its own
    tk blocks). With a single row, the gathered C_sel [tk, D] is shared within the
    program → every contraction is a genuine ``tl.dot``:
        scores[tk] = C_sel @ q       (M=tk, K=D, N=1, N padded ≥16)
        out[D]     = w   @ C_sel      (M=1 padded ≥16, K=tk, N=D)
    This sidesteps the Triton-3.6/sm_120 interior-axis 3D-reduce miscompile
    ENTIRELY — there is no ``tl.sum(a[:,:,None]*b[None,:,:], axis=1)`` anywhere.
  * tk streamed in BLOCK_T tiles with an online (flash) softmax; the sink term is
    added to the denominator after the streaming loop.
  * Branchless: invalid mask is a masked compare (-inf via tl.where); the gather
    bounds are guaranteed by construction (top_idx ∈ [0, n_blocks)).

Author: TileProver (Claude Code, Opus 4.8)
Date:   2026-05-31
Branch: 006-looped-block-ell
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
    # Forward: one program = one (b, h, q-row). Streams the tk selected blocks
    # in BLOCK_T tiles with an online softmax. Gathers C_sel rows on the fly via
    # top_idx; never materialises [B,S,tk,D]. Writes out [B,H,S,D] and the
    # per-row LSE [B,H,S] (= M + log l, sink folded in) for the backward pass.
    # -----------------------------------------------------------------------
    @triton.jit
    def _csa_fwd_kernel(
        q_ptr,          # [B, H, S, D]
        c_ptr,          # [B, NB, D]   (shared across heads)
        idx_ptr,        # [B, S, TK]   int32
        inv_ptr,        # [B, S, TK]   int8 (1 = invalid/future block)
        sink_ptr,       # [H]
        out_ptr,        # [B, H, S, D]
        lse_ptr,        # [B, H, S]    (M + log l)
        B, S,
        scale,
        H: tl.constexpr, NB: tl.constexpr, D: tl.constexpr, TK: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        # ONE program = one (b, s): ALL heads at this query position. The gather
        # (top_idx, invalid_mask) depends only on (b, s), so C_sel is gathered ONCE
        # and reused across all H heads — H-fold gather amortisation + meaningful
        # matmul N-dims. Per-head online-softmax stats are [BLOCK_H] vectors.
        pid = tl.program_id(0)          # over B * S
        s = pid % S
        b = pid // S

        d = tl.arange(0, BLOCK_D)                        # [BLOCK_D]
        dmask = d < D
        hh = tl.arange(0, BLOCK_H)                       # [BLOCK_H]
        hmask = hh < H

        # Q for all heads at (b, s): [BLOCK_H, BLOCK_D]. q[b,h,s,d] stride over h is
        # S*D. base for h=0 is ((b*H)*S + s)*D.
        q_base0 = ((b * H) * S + s) * D
        q_ptrs = q_ptr + q_base0 + hh[:, None] * (S * D) + d[None, :]
        q = tl.load(q_ptrs, mask=hmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        q = q * scale                                    # [BLOCK_H, BLOCK_D]

        sink = tl.load(sink_ptr + hh, mask=hmask, other=0.0).to(tl.float32)  # [BLOCK_H]

        idx_base = (b * S + s) * TK
        c_base = b * NB * D

        # online-softmax running stats per head. Sink seeded as the initial running
        # max with weight 1 (= exp(sink - sink)) and value 0 (dropped from the value
        # sum). Seeding the sink keeps m_i FINITE, so an all-invalid tile gives
        # m_new==m_i, alpha=exp(0)=1, p=0 — no NaN. This is exactly why CSA needs no
        # early-query guard: the sink always normalises the denominator.
        m_i = sink                                       # [BLOCK_H]
        l_i = tl.where(hmask, 1.0, 0.0)                  # [BLOCK_H]
        acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

        for t0 in tl.range(0, TK, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)               # [BLOCK_T]
            tmask = t < TK
            idx = tl.load(idx_ptr + idx_base + t, mask=tmask, other=0)   # [BLOCK_T]
            inv = tl.load(inv_ptr + idx_base + t, mask=tmask, other=1)   # [BLOCK_T] int8
            # gathered C rows: C_sel [BLOCK_T, BLOCK_D] (one gather, all heads)
            c_ptrs = c_ptr + c_base + idx[:, None] * D + d[None, :]
            c = tl.load(c_ptrs, mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

            # scores [BLOCK_H, BLOCK_T] = Q @ C_sel^T  (tl.dot: M=H, K=D, N=T)
            s_ht = tl.dot(q, tl.trans(c), out_dtype=tl.float32)          # [BLOCK_H, BLOCK_T]
            valid = (tmask & (inv == 0))[None, :]                        # [1, BLOCK_T]
            s_ht = tl.where(valid, s_ht, float("-inf"))

            # online softmax over the tk (last) axis — per head
            m_new = tl.maximum(m_i, tl.max(s_ht, axis=1))               # [BLOCK_H]
            alpha = tl.exp(m_i - m_new)                                 # [BLOCK_H]
            p = tl.where(valid, tl.exp(s_ht - m_new[:, None]), 0.0)     # [BLOCK_H, BLOCK_T]
            l_i = l_i * alpha + tl.sum(p, axis=1)                       # last-axis sum
            # acc = acc*alpha + P @ C_sel  (tl.dot: M=H, K=T, N=D)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), c, out_dtype=tl.float32)
            m_i = m_new

        out = acc / l_i[:, None]                                        # [BLOCK_H, BLOCK_D]

        # store out[b,:,s,:] for all heads
        out_ptrs = out_ptr + q_base0 + hh[:, None] * (S * D) + d[None, :]
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty),
                 mask=hmask[:, None] & dmask[None, :])

        # LSE[b,:,s] = m_i + log(l_i); strictly finite (sink seed → l_i > 0).
        lse = m_i + tl.log(l_i)                                         # [BLOCK_H]
        lse_ptrs = lse_ptr + (b * H + hh) * S + s
        tl.store(lse_ptrs, lse, mask=hmask)

    # -----------------------------------------------------------------------
    # Backward dQ + dSink + dC: one program = one (b, h, q-row). Recomputes the
    # softmax weights p from the LSE, streams the tk blocks (re-gathering C_sel),
    # accumulates dq and dsink locally, and ATOMICALLY scatters dC into the shared
    # dC[B,NB,D] (multiple (s,t) — and all heads — can select the same block).
    #
    #   D_acc        = sum_t p_t * (do · C_sel_t)
    #   dscore_t     = p_t * ((do · C_sel_t) - D_acc)
    #   dq           = scale * sum_t dscore_t * C_sel_t
    #   dsink        = -p_sink * D_acc                 ; p_sink = exp(sink - lse)
    #   dC_sel_t    += scale * dscore_t * q  +  p_t * do   (scattered to top_idx_t)
    # -----------------------------------------------------------------------
    @triton.jit
    def _csa_bwd_kernel(
        q_ptr, c_ptr, idx_ptr, inv_ptr, sink_ptr,
        do_ptr,         # [B, H, S, D]
        lse_ptr,        # [B, H, S]
        dq_ptr,         # [B, H, S, D]
        dsink_ptr,      # [B, H, S]   per-row dsink partial
        dc_ptr,         # [B, NB, D]  fp32, atomic-accumulated
        B, S, scale,
        H: tl.constexpr, NB: tl.constexpr, D: tl.constexpr, TK: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        # ONE program = one (b, s): all heads. Recomputes p from LSE, streams the tk
        # gathered blocks (re-gathering C_sel once per t-tile, shared across heads),
        # accumulates dq/dsink per head, and ATOMICALLY scatters dC into the shared
        # dC[B,NB,D] (heads AND query rows can select the same block).
        pid = tl.program_id(0)
        s = pid % S
        b = pid // S

        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        hh = tl.arange(0, BLOCK_H)
        hmask = hh < H

        q_base0 = ((b * H) * S + s) * D
        idx_base = (b * S + s) * TK
        c_base = b * NB * D

        q = tl.load(q_ptr + q_base0 + hh[:, None] * (S * D) + d[None, :],
                    mask=hmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)  # [BLOCK_H, BLOCK_D]
        qs = q * scale
        do = tl.load(do_ptr + q_base0 + hh[:, None] * (S * D) + d[None, :],
                     mask=hmask[:, None] & dmask[None, :], other=0.0).to(tl.float32) # [BLOCK_H, BLOCK_D]
        lse = tl.load(lse_ptr + (b * H + hh) * S + s, mask=hmask, other=0.0).to(tl.float32)  # [BLOCK_H]
        sink = tl.load(sink_ptr + hh, mask=hmask, other=0.0).to(tl.float32)          # [BLOCK_H]

        # Pass 1: D_acc[h] = sum_t p[h,t] * (do[h] · C_sel_t)
        D_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for t0 in tl.range(0, TK, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            tmask = t < TK
            idx = tl.load(idx_ptr + idx_base + t, mask=tmask, other=0)
            inv = tl.load(inv_ptr + idx_base + t, mask=tmask, other=1)
            c = tl.load(c_ptr + c_base + idx[:, None] * D + d[None, :],
                        mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            valid = (tmask & (inv == 0))[None, :]                          # [1, BLOCK_T]
            s_ht = tl.dot(qs, tl.trans(c), out_dtype=tl.float32)           # [BLOCK_H, BLOCK_T]
            p = tl.where(valid, tl.exp(s_ht - lse[:, None]), 0.0)
            doc = tl.dot(do, tl.trans(c), out_dtype=tl.float32)            # [BLOCK_H, BLOCK_T]
            D_acc += tl.sum(p * doc, axis=1)                               # last-axis sum

        # Pass 2: dq, dsink, scatter dC
        dq = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        for t0 in tl.range(0, TK, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            tmask = t < TK
            idx = tl.load(idx_ptr + idx_base + t, mask=tmask, other=0)
            inv = tl.load(inv_ptr + idx_base + t, mask=tmask, other=1)
            c = tl.load(c_ptr + c_base + idx[:, None] * D + d[None, :],
                        mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            valid_t = tmask & (inv == 0)                                   # [BLOCK_T]
            valid = valid_t[None, :]                                       # [1, BLOCK_T]
            s_ht = tl.dot(qs, tl.trans(c), out_dtype=tl.float32)           # [BLOCK_H, BLOCK_T]
            p = tl.where(valid, tl.exp(s_ht - lse[:, None]), 0.0)
            doc = tl.dot(do, tl.trans(c), out_dtype=tl.float32)            # [BLOCK_H, BLOCK_T]
            dscore = p * (doc - D_acc[:, None])                            # [BLOCK_H, BLOCK_T]

            # dq[h] += scale * sum_t dscore[h,t] * C_sel_t = scale * (dscore @ C_sel)
            #   tl.dot: M=H, K=T, N=D
            dq += scale * tl.dot(dscore.to(tl.float32), c, out_dtype=tl.float32)

            # dC_sel_t += sum_h (scale*dscore[h,t]*q[h] + p[h,t]*do[h])
            #   = scale*(dscore^T @ q) + (p^T @ do)   → [BLOCK_T, BLOCK_D]
            #   tl.dot contracts over the head axis (K=H).
            term_score = tl.dot(tl.trans((scale * dscore).to(tl.float32)), q,
                                out_dtype=tl.float32)                      # [BLOCK_T, BLOCK_D]
            term_val = tl.dot(tl.trans(p.to(tl.float32)), do,
                              out_dtype=tl.float32)                        # [BLOCK_T, BLOCK_D]
            dc_tile = term_score + term_val
            store_mask = valid_t[:, None] & dmask[None, :]
            dc_ptrs = dc_ptr + c_base + idx[:, None] * D + d[None, :]
            tl.atomic_add(dc_ptrs, dc_tile, mask=store_mask)

        tl.store(dq_ptr + q_base0 + hh[:, None] * (S * D) + d[None, :],
                 dq.to(dq_ptr.dtype.element_ty), mask=hmask[:, None] & dmask[None, :])

        # dsink[h] = -p_sink[h] * D_acc[h] ; p_sink = exp(sink - lse)
        p_sink = tl.exp(sink - lse)                                       # [BLOCK_H]
        tl.store(dsink_ptr + (b * H + hh) * S + s, -p_sink * D_acc, mask=hmask)


# ===========================================================================
# Python wrappers
# ===========================================================================

def _csa_forward(q, C_comp, top_idx, invalid_mask, sink_logits, scale):
    B, H, S, D = q.shape
    NB = C_comp.shape[1]
    TK = top_idx.shape[-1]
    BLOCK_D = max(16, _next_pow2(D))
    BLOCK_H = max(16, _next_pow2(H))
    BLOCK_T = min(_next_pow2(TK), 64) if TK >= 16 else 16
    dev = q.device
    odt = q.dtype

    idx_i32 = top_idx.to(torch.int32).contiguous()
    inv_i8 = invalid_mask.to(torch.int8).contiguous()

    out = torch.empty(B, H, S, D, device=dev, dtype=odt)
    lse = torch.empty(B, H, S, device=dev, dtype=torch.float32)
    grid = (B * S,)
    _csa_fwd_kernel[grid](
        q, C_comp, idx_i32, inv_i8, sink_logits, out, lse,
        B, S, scale,
        H, NB, D, TK, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T, **_LAUNCH,
    )
    return out, lse, idx_i32, inv_i8, BLOCK_H, BLOCK_D, BLOCK_T


def _csa_backward(grad_out, q, C_comp, idx_i32, inv_i8, sink_logits, lse,
                  scale, BLOCK_H, BLOCK_D, BLOCK_T):
    B, H, S, D = q.shape
    NB = C_comp.shape[1]
    TK = idx_i32.shape[-1]
    dev = q.device
    f32 = torch.float32

    grad_out = grad_out.contiguous()
    dq = torch.empty(B, H, S, D, device=dev, dtype=f32)
    dsink_row = torch.empty(B, H, S, device=dev, dtype=f32)
    dc = torch.zeros(B, NB, D, device=dev, dtype=f32)

    grid = (B * S,)
    _csa_bwd_kernel[grid](
        q, C_comp, idx_i32, inv_i8, sink_logits, grad_out, lse,
        dq, dsink_row, dc,
        B, S, scale,
        H, NB, D, TK, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T, **_LAUNCH,
    )

    dsink = dsink_row.sum(dim=(0, 2))   # [H]
    return dq, dc, dsink


class _FusedCSAAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, C_comp, top_idx, invalid_mask, sink_logits, scale):
        q = q.contiguous()
        C_comp = C_comp.contiguous()
        sink_logits = sink_logits.contiguous()
        out, lse, idx_i32, inv_i8, BH, BD, BT = _csa_forward(
            q, C_comp, top_idx, invalid_mask, sink_logits, scale)
        ctx.save_for_backward(q, C_comp, idx_i32, inv_i8, sink_logits, lse)
        ctx.cfg = (scale, BH, BD, BT)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, C_comp, idx_i32, inv_i8, sink_logits, lse = ctx.saved_tensors
        scale, BH, BD, BT = ctx.cfg
        dq, dc, dsink = _csa_backward(
            grad_out, q, C_comp, idx_i32, inv_i8, sink_logits, lse, scale, BH, BD, BT)
        # grads only for q, C_comp, sink_logits (top_idx/invalid_mask non-diff)
        return (dq.to(q.dtype), dc.to(C_comp.dtype), None, None,
                dsink.to(sink_logits.dtype), None)


# ===========================================================================
# Public API
# ===========================================================================

def fused_csa_attention(q: Tensor, C_comp: Tensor, top_idx: Tensor,
                        invalid_mask: Tensor, sink_logits: Tensor,
                        scale: float) -> Tensor:
    """Fused CSA top-k compressed attention (gather-on-the-fly, fwd + bwd in Triton).

    Args:
        q:            [B, H, S, D]      per query head
        C_comp:       [B, n_blocks, D]  compressed block keys/values (shared/MQA)
        top_idx:      [B, S, tk] int    selected block indices (NON-differentiable)
        invalid_mask: [B, S, tk] bool   True → gathered block is a future block
        sink_logits:  [H]               per-head learnable sink logit
        scale:        softmax scale (D ** -0.5)

    Returns:
        out_comp:     [B, H, S, D]   matches the eager CSA einsum path exactly.

    Never materialises C_sel = [B, S, tk, D] (the ~2 GiB/layer tensor at scale).
    """
    if not TRITON_AVAILABLE or not q.is_cuda:
        return csa_attention_reference(q, C_comp, top_idx, invalid_mask,
                                       sink_logits, scale)
    return _FusedCSAAttention.apply(q, C_comp, top_idx, invalid_mask,
                                    sink_logits, scale)


# ===========================================================================
# Pure-PyTorch reference — EXACTLY the eager block in _CCACSAAttention.forward
# ===========================================================================

def csa_attention_reference(q: Tensor, C_comp: Tensor, top_idx: Tensor,
                            invalid_mask: Tensor, sink_logits: Tensor,
                            scale: float) -> Tensor:
    """Byte-faithful reference of the CSA top-k compressed-attention core.

    Mirrors lines 427-440 of _CCACSAAttention.forward (the gather + masked
    softmax + value-weighted sum), with top_idx/invalid_mask provided.
    """
    B, H, S, D = q.shape
    device = q.device

    batch_idx = torch.arange(B, device=device)[:, None, None]
    C_sel = C_comp[batch_idx, top_idx]                      # [B, S, tk, D]

    attn_scores = torch.einsum("bhsd,bstd->bhst", q, C_sel) * scale
    attn_scores = attn_scores.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))

    sink = sink_logits.view(1, H, 1, 1).expand(B, -1, S, 1)
    scores_aug = torch.cat([attn_scores, sink], dim=-1)
    attn_w = F.softmax(scores_aug.float(), dim=-1).to(q.dtype)[..., :-1]

    out_comp = torch.einsum("bhst,bstd->bhsd", attn_w, C_sel)   # [B, H, S, D]
    return out_comp


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required for the self-test"
    dev = torch.device("cuda")
    dt = torch.bfloat16
    print("=" * 100)
    print("fused_csa_attention — forward + backward correctness test")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    d_model, H = 768, 12
    comp = 2
    D = d_model // (comp * H)        # 32
    m = 4                            # csa_compress_ratio
    top_k = 128
    scale = D ** -0.5

    def make_inputs(B, S):
        nb = S // m
        tk = min(top_k, nb)
        # build a causal compressed mask and a plausible top-k selection, so that
        # invalid_mask actually exercises both valid and future blocks.
        block_end = (torch.arange(nb, device=dev) + 1) * m - 1            # [nb]
        qpos = torch.arange(S, device=dev)                               # [S]
        causal = block_end[None, :] < qpos[:, None]                      # [S, nb] valid
        scores = torch.rand(B, S, nb, device=dev)
        scores = scores.masked_fill(~causal[None], float("-inf"))
        # relu-like: future blocks score 0 (mirror LightningIndexer), top-k picks
        scores = torch.where(causal[None], scores, torch.zeros_like(scores))
        _, top_idx = scores.topk(tk, dim=-1)                             # [B, S, tk]
        gathered_valid = causal[None].expand(B, -1, -1).gather(-1, top_idx)
        invalid_mask = ~gathered_valid                                   # [B, S, tk]
        return {
            "q": torch.randn(B, H, S, D, device=dev, dtype=dt),
            "C_comp": torch.randn(B, nb, D, device=dev, dtype=dt),
            "sink": torch.randn(H, device=dev, dtype=dt).mul_(0.5),
            "top_idx": top_idx,
            "invalid_mask": invalid_mask,
        }

    def clone_req(ins):
        out = {}
        for k, v in ins.items():
            if k in ("q", "C_comp", "sink"):
                out[k] = v.detach().clone().requires_grad_(True)
            else:
                out[k] = v  # non-diff, shared
        return out

    def cos(a, b):
        return F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0).item()

    def run_case(B, S, mean_tol=2e-3, fcos_thresh=0.9999, gcos_thresh=0.995):
        ins = make_inputs(B, S)
        a = clone_req(ins)   # fused
        r = clone_req(ins)   # reference

        of = fused_csa_attention(a["q"], a["C_comp"], a["top_idx"],
                                 a["invalid_mask"], a["sink"], scale)
        orf = csa_attention_reference(r["q"], r["C_comp"], r["top_idx"],
                                      r["invalid_mask"], r["sink"], scale)
        # fp32 ground-truth of the SAME math (bf16-rounded inputs upcast)
        truth = csa_attention_reference(
            ins["q"].float(), ins["C_comp"].float(), ins["top_idx"],
            ins["invalid_mask"], ins["sink"].float(), scale)

        emax = (of.float() - orf.float()).abs().max().item()
        emean = (of.float() - orf.float()).abs().mean().item()
        fcos = cos(of, orf)
        fvt = (of.float() - truth).abs().max().item()
        rvt = (orf.float() - truth).abs().max().item()

        # ── 2x-doubling footgun check (interior-axis 3D-reduce miscompile) ──
        # If the kernel silently doubled (or 4x'd) any contraction, the fused
        # output norm would be a clean multiple of the reference norm. Assert the
        # ratio is ~1.0, NOT ~2.0 or ~4.0. (Each row of out is a convex combo of
        # C_sel rows → ||out_fused|| ≈ ||out_ref|| when correct.)
        rn = orf.float().norm().item()
        fn = of.float().norm().item()
        ratio = fn / max(rn, 1e-9)
        doubled = abs(ratio - 2.0) < 0.15 or abs(ratio - 4.0) < 0.3
        not_doubled = (0.9 < ratio < 1.1)

        go = torch.randn_like(of)
        (of.float() * go.float()).sum().backward()
        (orf.float() * go.float()).sum().backward()

        gq = cos(a["q"].grad, r["q"].grad)
        gc = cos(a["C_comp"].grad, r["C_comp"].grad)
        gs = cos(a["sink"].grad, r["sink"].grad)
        gworst = min(gq, gc, gs)

        as_correct = fvt <= rvt + 1e-3
        fwd_ok = (emean < mean_tol) and (fcos > fcos_thresh) and as_correct
        bwd_ok = gworst > gcos_thresh
        status = "PASS" if (fwd_ok and bwd_ok and not_doubled and not doubled) else "FAIL"
        print(f"  [{status}] B={B} S={S:<5} nb={S // m:<4} tk={ins['top_idx'].shape[-1]:<3} "
              f"fwd_max={emax:.2e} fwd_mean={emean:.2e} fwd_cos={fcos:.6f} | "
              f"vs-truth: ker={fvt:.2e} ref={rvt:.2e} | "
              f"norm_ratio={ratio:.4f} | gcos q={gq:.4f} C={gc:.4f} sink={gs:.4f}")
        return fwd_ok and bwd_ok and not_doubled and not doubled

    print("\n[Correctness — forward + backward across dim sweep]")
    print("  (norm_ratio must be ~1.0 — proves NO 2x/4x interior-reduce miscompile)")
    all_ok = True
    for S in (512, 1024, 2048, 4096):
        all_ok &= run_case(2, S)

    # =====================================================================
    # End-to-end vs the REAL _CCACSAAttention compressed-attention sub-path.
    # =====================================================================
    print("\n[End-to-end vs real _CCACSAAttention compressed path]")
    e2e_ok = True
    try:
        from morph.model.attention import _CCACSAAttention, _compressed_causal_mask

        for S in (512, 1024, 2048):
            mod = _CCACSAAttention(
                d_model=768, n_heads=12, n_kv_heads=4, compression=2,
                csa_compress_ratio=4, top_k=128, d_indexer=32,
                max_seq_len=8192, context_len=4096,
                window_size=128, init_alpha=0.1, conv_kernel=4).to(dev)
            with torch.no_grad():
                mod.cca.sink_logits.normal_(0, 0.5)
            x = torch.randn(2, S, 768, device=dev, dtype=dt)

            with torch.autocast("cuda", dtype=dt):
                q, k, v = mod.cca._cca_project(x, 0)
                Hh, Dd = mod.cca.n_heads, mod.cca.d_head
                mm = mod.compress_ratio
                nb = S // mm
                sc = Dd ** -0.5
                B = x.shape[0]

                C_comp = mod.comp_norm(mod.compressor(x))          # [B, nb, D]
                causal = _compressed_causal_mask(S, nb, mm, x.device)
                causal_3d = causal.unsqueeze(0).expand(B, -1, -1)
                scores = mod.indexer(x, causal_3d)
                tk = min(mod.top_k, nb)
                _, top_idx = scores.topk(tk, dim=-1)               # [B, S, tk]
                gathered_valid = causal_3d.gather(-1, top_idx)
                invalid_mask = ~gathered_valid

                # eager out_comp (exact block from forward)
                batch_idx = torch.arange(B, device=x.device)[:, None, None]
                C_sel = C_comp[batch_idx, top_idx]                 # [B,S,tk,D]
                attn_scores = torch.einsum("bhsd,bstd->bhst", q, C_sel) * sc
                attn_scores = attn_scores.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))
                sink = mod.cca.sink_logits.view(1, Hh, 1, 1).expand(B, -1, S, 1)
                scores_aug = torch.cat([attn_scores, sink], dim=-1)
                attn_w = F.softmax(scores_aug.float(), dim=-1).to(x.dtype)[..., :-1]
                out_eager = torch.einsum("bhst,bstd->bhsd", attn_w, C_sel)

                qf = q.detach().clone().requires_grad_(True)
                Cf = C_comp.detach().clone().requires_grad_(True)
                sf = mod.cca.sink_logits.detach().clone().requires_grad_(True)
                qr = q.detach().clone().requires_grad_(True)
                Cr = C_comp.detach().clone().requires_grad_(True)
                sr = mod.cca.sink_logits.detach().clone().requires_grad_(True)

                out_fused = fused_csa_attention(qf, Cf, top_idx, invalid_mask, sf, sc)
                out_ref = csa_attention_reference(qr, Cr, top_idx, invalid_mask, sr, sc)

            truth = csa_attention_reference(
                q.float(), C_comp.float(), top_idx, invalid_mask,
                mod.cca.sink_logits.float(), sc)
            emax = (out_fused.float() - out_eager.float()).abs().max().item()
            cmin = cos(out_fused, out_eager)
            fvt = (out_fused.float() - truth).abs().max().item()
            rvt = (out_eager.float() - truth).abs().max().item()

            go = torch.randn_like(out_fused)
            (out_fused.float() * go.float()).sum().backward()
            (out_ref.float() * go.float()).sum().backward()
            gq = cos(qf.grad, qr.grad)
            gc = cos(Cf.grad, Cr.grad)
            gs = cos(sf.grad, sr.grad)

            ok = (cmin > 0.999) and (fvt <= rvt + 1e-3) and min(gq, gc, gs) > 0.99
            e2e_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] S={S:<5} real_csa: "
                  f"max={emax:.2e} cos={cmin:.6f} | vs-truth ker={fvt:.2e} "
                  f"eager={rvt:.2e} | gcos q={gq:.4f} C={gc:.4f} sink={gs:.4f}")
    except Exception as ex:
        e2e_ok = False
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] end-to-end raised: {type(ex).__name__}: {ex}")
    all_ok &= e2e_ok

    # =====================================================================
    # Speed + peak-memory comparison (the headline: C_sel never alloc'd)
    # =====================================================================
    def speed_mem_case(B, S):
        print(f"\n[Speed + peak memory — eager vs fused, B={B}, S={S}]")
        ins = make_inputs(B, S)
        nb = S // m
        tk = ins["top_idx"].shape[-1]

        def fused_fb(d):
            of = fused_csa_attention(d["q"], d["C_comp"], d["top_idx"],
                                     d["invalid_mask"], d["sink"], scale)
            of.sum().backward()

        def eager_fb(d):
            orf = csa_attention_reference(d["q"], d["C_comp"], d["top_idx"],
                                          d["invalid_mask"], d["sink"], scale)
            orf.sum().backward()

        def bench(fn, d, n=20, warmup=5):
            for _ in range(warmup):
                for k_ in ("q", "C_comp", "sink"):
                    d[k_].grad = None
                fn(d)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                for k_ in ("q", "C_comp", "sink"):
                    d[k_].grad = None
                fn(d)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e3

        def peak_mem(fn, d):
            for k_ in ("q", "C_comp", "sink"):
                d[k_].grad = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn(d)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            return (peak - base) / 2**20

        csel_bytes = B * S * tk * D * 2 / 2**20
        scores_bytes = B * H * S * tk * 2 / 2**20

        # speed (only safe when eager fits; guard OOM)
        try:
            a = clone_req(ins); r = clone_req(ins)
            t_fused = bench(fused_fb, a)
            try:
                t_eager = bench(eager_fb, r)
                print(f"  Eager fwd+bwd:  {t_eager:.3f} ms")
                print(f"  Fused fwd+bwd:  {t_fused:.3f} ms   ({t_eager / t_fused:.2f}x)")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  Eager fwd+bwd:  OOM (could not allocate C_sel/scores)")
                print(f"  Fused fwd+bwd:  {t_fused:.3f} ms")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  Fused fwd+bwd:  OOM")

        # peak mem
        try:
            a2 = clone_req(ins)
            m_fused = peak_mem(fused_fb, a2)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); m_fused = float("nan")
        try:
            r2 = clone_req(ins)
            m_eager = peak_mem(eager_fb, r2)
            mem_line = (f"  Eager peak: {m_eager:.1f} MiB | Fused peak: {m_fused:.1f} MiB"
                        f" | reduction: {m_eager - m_fused:.1f} MiB "
                        f"({m_eager / max(m_fused, 1e-6):.2f}x)")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mem_line = (f"  Eager peak: OOM | Fused peak: {m_fused:.1f} MiB "
                        f"(eager cannot allocate)")
        print(mem_line)
        print(f"  (C_sel [B,S,tk,D] bf16 = {csel_bytes:.1f} MiB; "
              f"scores [B,H,S,tk] bf16 = {scores_bytes:.1f} MiB; eager allocs both ×autograd)")

    speed_mem_case(8, 4096)
    speed_mem_case(32, 8192)

    print("\n" + "=" * 100)
    print("ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 100)
    assert all_ok, "fused_csa_attention self-test FAILED — see rows marked FAIL above"
