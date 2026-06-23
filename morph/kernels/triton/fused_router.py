"""fused_router.py — fused ReMoE decode router for the MORPH StaticDecodeEngine.

Replaces the eager-Python router pile in kv_cache_static.py:777-787 (per routed-MLP
decode visit, 42×/token) with a 2-launch path:

  (1) qv = small_gemv(x, Wq, bias=iter_vec)          # proven MULTI-CTA gate projection
                                                      # (one CTA per output row; cublas-like
                                                      #  parallelism — a 768×768 GEMV must NOT
                                                      #  be serialized on a single CTA).
  (2) _router_tail_kernel  (ONE CTA): everything after the projection —
        qv     -> layer_norm(qv, ln_w, ln_b, eps)
        sa     = qn[:d/2] @ ska_t   ;  sb = qn[d/2:] @ skb_t          # [nsk] each
        logits = (sa[:,None] + sb[None,:]).reshape(ncls) + gbias      # 1:1 product map
        kth    = topk(logits, k).values[-1]                          # desc-rank threshold
        gates  = relu(logits - kth)                                  # ReMoE soft gate
        gates *= k / sum(gates).clamp(min=1e-6)                      # magnitude-preserving

The tail is all ≤d-wide elementwise/tiny-reduction work, so ONE CTA is optimal — and it
collapses the eager pile's layer_norm + 2 sub-GEMVs + topk(the 254µs cublas/bitonic pair) +
relu + sum + normalize (~7 launches) into a SINGLE launch. The heavy 768×768 GEMV stays
multi-CTA (kernel 1). Net: ~8 eager router launches/visit -> 2 launches/visit.

Op-for-op identical to TileRouter.forward (routing.py) and the eager mirror. The top-k
threshold is an EXACT in-register desc-rank selection over ncls(=16) logits reproducing
torch.topk's value-desc / index-asc rule for the THRESHOLD VALUE; since gates depend only on
(logit - kth_value), tie-break among equal logits is output-irrelevant. The deploy router
runs FP32 (params stay fp32 through to_deploy_inference), so the GEMV+tail accumulate fp32
and bit-match the eager pile (parity gate: active_mismatch=0, max_gate_err=2.1e-6).

Constraints: B==1; nsk*nsk==ncls (DIRECT 1:1; MORPH nsk=4,ncls=16); d even.
Target: RTX 5090 / sm_120. num_stages=1, no TMA/TMEM. num_warps Z3/microbench-tuned.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from morph.kernels.triton.fused_decode_step import small_gemv


@triton.jit
def _router_tail_kernel(
    QV,           # [d]      fp32 projected query (x@Wq.T + iter_vec), from small_gemv
    LNW, LNB,     # [d]      layernorm weight/bias
    SKA_T,        # [d2, nsk] sub_keys_a.T contiguous
    SKB_T,        # [d2, nsk]
    GBIAS,        # [ncls]
    GATES_OUT,    # [ncls]   fp32 output gates
    LN_EPS,       # scalar
    K,            # int active count
    D: tl.constexpr, D2: tl.constexpr, NSK: tl.constexpr, NCLS: tl.constexpr,
    DPAD: tl.constexpr, D2PAD: tl.constexpr,
):
    # LayerNorm(QV) over D (mask the pow2 tail so it never enters the reductions).
    d_idx = tl.arange(0, DPAD)
    dm = d_idx < D
    qv = tl.load(QV + d_idx, mask=dm, other=0.0)
    mean = tl.sum(qv, axis=0) / D
    xc = tl.where(dm, qv - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    # Normalized halves (per-element LN; no [D2,D] gather).
    i_lo = tl.arange(0, D2PAD)
    lm = i_lo < D2
    lo_qv = tl.load(QV + i_lo, mask=lm, other=0.0)
    hi_qv = tl.load(QV + i_lo + D2, mask=lm, other=0.0)
    qn_lo = tl.where(lm, (lo_qv - mean) * rstd * tl.load(LNW + i_lo, mask=lm, other=0.0)
                     + tl.load(LNB + i_lo, mask=lm, other=0.0), 0.0)
    qn_hi = tl.where(lm, (hi_qv - mean) * rstd * tl.load(LNW + i_lo + D2, mask=lm, other=0.0)
                     + tl.load(LNB + i_lo + D2, mask=lm, other=0.0), 0.0)
    c_idx = tl.arange(0, NSK)
    ska = tl.load(SKA_T + i_lo[:, None] * NSK + c_idx[None, :], mask=lm[:, None], other=0.0)
    skb = tl.load(SKB_T + i_lo[:, None] * NSK + c_idx[None, :], mask=lm[:, None], other=0.0)
    sa = tl.sum(qn_lo[:, None] * ska, axis=0)        # [NSK]
    sb = tl.sum(qn_hi[:, None] * skb, axis=0)        # [NSK]

    # product logits: logits[a*NSK+b] = sa[a]+sb[b]+gbias
    g_idx = tl.arange(0, NCLS)
    a_of = g_idx // NSK
    b_of = g_idx % NSK
    sa_g = tl.sum(tl.where(c_idx[None, :] == a_of[:, None], sa[None, :], 0.0), axis=1)
    sb_g = tl.sum(tl.where(c_idx[None, :] == b_of[:, None], sb[None, :], 0.0), axis=1)
    logits = sa_g + sb_g + tl.load(GBIAS + g_idx)    # [NCLS]

    # k-th largest VALUE via exact desc-rank (strict-gt + tie-by-index == torch.topk).
    gt = (logits[None, :] > logits[:, None]).to(tl.int32)
    eq = (logits[None, :] == logits[:, None])
    le_idx = (g_idx[None, :] <= g_idx[:, None])
    rank = tl.sum(gt, axis=1) + tl.sum((eq & le_idx).to(tl.int32), axis=1)
    kth = tl.sum(tl.where(rank == K, logits, 0.0), axis=0)

    gates = tl.maximum(logits - kth, 0.0)
    gsum = tl.maximum(tl.sum(gates, axis=0), 1e-6)
    gates = gates * (K / gsum)
    tl.store(GATES_OUT + g_idx, gates)


def fused_router(
    x: Tensor, wq: Tensor, iter_vec: Tensor,
    ln_w: Tensor, ln_b: Tensor, ln_eps: float,
    ska_t: Tensor, skb_t: Tensor, gbias: Tensor, k: int,
    out: Tensor | None = None, qv_scr: Tensor | None = None,
    num_warps: int = 2,
) -> Tensor:
    """2-launch ReMoE router. Returns gates [1, ncls] fp32 (active>0, sum==k).

    qv_scr [d] fp32 holds the projection between the two launches (persistent scratch).
    """
    d = wq.shape[1]
    ncls = gbias.numel()
    nsk = ska_t.shape[1]
    d2 = d // 2
    assert wq.shape == (d, d), f"wq must be square [d,d], got {tuple(wq.shape)}"
    assert nsk * nsk == ncls, f"fused router needs nsk^2==ncls (1:1 map); {nsk}^2!={ncls}"
    assert ska_t.shape == (d2, nsk) and skb_t.shape == (d2, nsk)
    xf = x.reshape(1, d)
    if out is None:
        out = torch.empty(1, ncls, device=x.device, dtype=torch.float32)

    # (1) gate projection: qv = x @ Wq.T + iter_vec — proven MULTI-CTA small_gemv (one CTA/row).
    qv = small_gemv(xf.to(wq.dtype), wq, bias=iter_vec.reshape(-1))   # [1, d] fp32
    if qv_scr is not None:
        qv_scr.copy_(qv.reshape(-1)); qv_flat = qv_scr
    else:
        qv_flat = qv.reshape(-1)

    # (2) fused tail: LN + subkeys + product logits + topk + relu + normalize -> gates.
    dpad = triton.next_power_of_2(d)
    d2pad = triton.next_power_of_2(d2)
    _router_tail_kernel[(1,)](
        qv_flat, ln_w, ln_b, ska_t, skb_t, gbias.reshape(-1),
        out.reshape(-1), ln_eps, int(k),
        D=d, D2=d2, NSK=nsk, NCLS=ncls, DPAD=dpad, D2PAD=d2pad,
        num_warps=num_warps, num_stages=1,
    )
    return out
