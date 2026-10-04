"""One-pass row softmax-gradient for the chunked linear cross-entropy (``fused_ce.py``).

WHY (2026-10-04 training-speed pass). Per ``[c, V]`` logits tile the eager chunk body runs
``.float()`` (bf16 -> fp32 copy), two masked column writes, ``logsumexp``, a gather,
``softmax``, a ``scatter_add_`` of -1 at the label, a row-weight multiply and ``.to(bf16)``:
about ten full passes over a tile of ``c x 49280`` fp32 values. On the lead slot-loop arm
that elementwise work is about 45 ms of GPU time per training step (the span decoder's CE
over 12288 rows plus the token CE over 7680 rows), more than the three GEMMs it sits
between. This kernel reads the bf16 logits tile twice (once for the row's log-sum-exp,
once to write the gradient) and writes the bf16 gradient IN PLACE over it.

What it computes, per row ``i`` with label ``y`` (clamped >= 0), weight ``w`` (0 on an
ignored row), real vocab ``V`` and masked column ``mask_id`` (< 0 = none):

    x_j   = fp32(logits[i, j]), -inf for j >= V and j == mask_id
    lse   = max_j x_j + log(sum_j exp(x_j - max_j x_j))     (one online pass)
    loss  = (lse - x_y) * w
    out_j = bf16((exp(x_j - lse) - [j == y]) * w)            (written over logits[i, :])

NOT BIT-IDENTICAL to the eager body: the eager path takes the max and the sum in ATen's
reduction order and forms the probability as ``exp(x - max) / sum``; here the sum is
accumulated block by block with a running max and the probability is ``exp(x - lse)``.
Both are within a few fp32 ulps of the exact value, far below the bf16 rounding of the
gradient tile. ``tests/test_ce_softmax_grad.py`` bounds the difference against an fp64
reference and against the eager kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ce_softmax_grad_kernel(
    logit_ptr, lab_ptr, w_ptr, loss_ptr,
    stride_row, V, V_pad, mask_id,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = logit_ptr + row * stride_row
    lab = tl.load(lab_ptr + row)
    w = tl.load(w_ptr + row)
    cols = tl.arange(0, BLOCK_V)
    neg_inf = float("-inf")

    m = tl.full((BLOCK_V,), neg_inf, tl.float32)
    s = tl.zeros((BLOCK_V,), tl.float32)
    for start in range(0, V_pad, BLOCK_V):
        c = start + cols
        x = tl.load(base + c, mask=c < V_pad, other=0.0).to(tl.float32)
        x = tl.where((c < V) & (c != mask_id), x, neg_inf)
        m_new = tl.maximum(m, x)
        # lanes still at -inf have nothing to rescale (exp(-inf - -inf) would be NaN)
        alpha = tl.where(m_new == neg_inf, 0.0, tl.exp2((m - m_new) * 1.4426950408889634))
        s = s * alpha + tl.where(x == neg_inf, 0.0, tl.exp2((x - m_new) * 1.4426950408889634))
        m = m_new
    m_row = tl.max(m, axis=0)
    s_row = tl.sum(tl.where(m == neg_inf, 0.0, s * tl.exp2((m - m_row) * 1.4426950408889634)),
                   axis=0)
    lse = m_row + tl.log(s_row)
    x_y = tl.load(base + lab).to(tl.float32)
    tl.store(loss_ptr + row, (lse - x_y) * w)

    for start in range(0, V_pad, BLOCK_V):
        c = start + cols
        inb = c < V_pad
        x = tl.load(base + c, mask=inb, other=0.0).to(tl.float32)
        live = (c < V) & (c != mask_id)
        p = tl.where(live, tl.exp2((x - lse) * 1.4426950408889634), 0.0)
        p = tl.where(c == lab, p - 1.0, p)
        tl.store(base + c, (p * w).to(logit_ptr.dtype.element_ty), mask=inb)


def ce_softmax_grad_(logits: torch.Tensor, labels: torch.Tensor, w: torch.Tensor, V: int,
                     mask_id: int = -1) -> torch.Tensor:
    """In place: ``logits`` ``[c, V_pad]`` (bf16/fp16, row-contiguous) becomes the
    weighted softmax gradient; returns the per-row weighted loss ``[c]`` fp32. ``labels``
    ``[c]`` must already be clamped to ``>= 0``; ``w`` ``[c]`` fp32 is 0 on ignored rows."""
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError("ce_softmax_grad_ wants a row-contiguous [c, V_pad] tile")
    if logits.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"ce_softmax_grad_ wants a bf16/fp16 tile, got {logits.dtype}")
    c, v_pad = logits.shape
    loss = torch.empty(c, device=logits.device, dtype=torch.float32)
    if c == 0:
        return loss
    _ce_softmax_grad_kernel[(c,)](
        logits, labels.contiguous(), w.contiguous().to(torch.float32), loss,
        logits.stride(0), int(V), int(v_pad), int(mask_id),
        BLOCK_V=2048, num_warps=8,
    )
    return loss
