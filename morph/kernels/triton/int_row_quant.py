"""The per-row int-N fake quantisation of an embedding table in ONE pass (2026-10-10,
training-speed pass; ``embed_quant``, morph/model/embed_quant.py ``IntNRowSTE``).

WHY. The eager ``IntNRowSTE.forward`` is nine elementwise / reduction kernels over the whole
table (abs, amax, scale, clamp, divide, round, clamp, multiply, subtract, add), each a full
read and write of a [V, d] fp32 table: 1.4 ms for the euclidean table and 2.3 ms for the
bigram table on the lxtul_pointer_fast step (2026-10-10 profile). This kernel reads the table
once and writes the result once.

EXACT. Every operation is the eager one, in the eager order, in fp32, so the result is bit
for bit the eager result (pinned in tests/test_int_row_quant.py):
  * ``s = amax(|w_row|) * (1 / hi)``: ATen divides a tensor by a CPU scalar as a multiply by
    the scalar's fp32 reciprocal (BinaryDivTrueKernel.cu), so the kernel takes that
    reciprocal (``inv_hi``, rounded once on the host in fp32) and multiplies; the max is
    exact in any order; a row with a NaN reads NaN, as ``amax`` does.
  * ``clamp(min=1e-8)``, ``clamp(lo, hi)``: NaN passes through, as ATen's clamp does.
  * ``w / s``: the IEEE-rounded division (``div_rn``), as nvcc compiles ATen's ``a / b``.
  * ``round``: round half to even (libdevice ``rint``), as ``torch.round``.
  * ``q = r * s``, then ``w + (q - w)``: three separately rounded operations; the launch
    turns floating-point contraction OFF (``enable_fp_fusion=False``), so no multiply-add is
    fused into an FMA.
"""
from __future__ import annotations

from ._pdl import pdl_opts

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _int_row_quant_kernel(w_ptr, out_ptr, n_rows, D, inv_hi, lo, hi,
                          BLOCK_D: tl.constexpr, ROWS: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    rows = (pid * ROWS + tl.arange(0, ROWS))[:, None]
    cols = tl.arange(0, BLOCK_D)[None, :]
    m = (rows < n_rows) & (cols < D)
    off = rows.to(tl.int64) * D + cols
    w = tl.load(w_ptr + off, mask=m, other=0.0)
    amax = tl.max(tl.abs(w), axis=1)
    nan_row = tl.sum((w != w).to(tl.int32), axis=1) > 0
    amax = tl.where(nan_row, float("nan"), amax)
    s = amax * inv_hi
    s = tl.where(s != s, s, tl.maximum(s, 1e-8))
    x = libdevice.div_rn(w, s[:, None])
    r = libdevice.rint(x)
    r = tl.where(r != r, r, tl.minimum(tl.maximum(r, lo), hi))
    q = r * s[:, None]
    out = w + (q - w)
    tl.store(out_ptr + off, out, mask=m)


def int_row_quant(w: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """``IntNRowSTE``'s forward value for a CUDA fp32 ``[V, d]`` table, in one pass."""
    if not (w.is_cuda and w.dtype == torch.float32 and w.dim() == 2):
        raise ValueError(f"int_row_quant: a CUDA fp32 [V, d] table, got {w.dtype} "
                         f"{tuple(w.shape)} on {w.device}")
    w = w.contiguous()
    V, D = w.shape
    out = torch.empty_like(w)
    BLOCK_D = triton.next_power_of_2(D)
    ROWS = max(1, min(16, 4096 // BLOCK_D))
    inv_hi = float(np.float32(1.0) / np.float32(hi))
    _int_row_quant_kernel[(triton.cdiv(V, ROWS),)](
        w, out, V, D, inv_hi, float(lo), float(hi), BLOCK_D=BLOCK_D, ROWS=ROWS,
        num_warps=4 if BLOCK_D <= 1024 else 8, enable_fp_fusion=False, **pdl_opts())
    return out


class IntRowQuantSTE(torch.autograd.Function):
    """``w + (q - w).detach()`` with the value from :func:`int_row_quant`: the backward is
    the identity, the gradient tensor handed through unchanged (the eager ``AddBackward0``
    does the same with alpha 1)."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return int_row_quant(w.detach(), lo, hi)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return g, None, None
