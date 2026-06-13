# torch.library custom-op wrappers around the vendored stk BCSR Triton kernels.
# NOT part of upstream stk — added for MORPH (torch.compile support).
"""Compile-friendly entry points for the stk dds/sdd kernels.

Why this exists: the vendored stk autograd path (backend/autograd.py — bare
torch.autograd.Functions fed by a per-call Matrix wrapper object) is hostile to
torch.compile. Dynamo 2.11 *can* speculate through it, but the trace is fragile
(stride-sniffing helpers, recursive autograd.Function calls in backward, autotuned
kernels behind host-side asserts), and any failure poisons the whole containing MLP.
These custom ops expose the SAME kernels through torch.library.custom_op with
register_fake metas: the op is opaque to Inductor (launched as-is, bit-identical),
while the surrounding elementwise (SwiGLU gate, ternary STE, bias) fuses normally.

Numerics: byte-faithful to backend/autograd.py DDS — identical backend.dds /
backend.sdd calls with identical tensors and strides. Autocast casting is the
CALLER's job (mirroring stk's custom_fwd: cast lhs+data to the autocast dtype
before calling; the kernels themselves are dtype-agnostic).

Semantics (matching backend/triton_kernels.py):
  stk_dds(lhs, …, shape0, shape1, transpose_b) -> [M, shape1]
    transpose_b=True : stored BCSR matrix is (shape1, shape0); computes lhs @ Wᵀ.
                       (carved-linear forward: x [M,in] @ Wᵀ, W (out,in),
                       shape0=in, shape1=out)
    transpose_b=False: stored BCSR matrix is (shape0, shape1); computes lhs @ W.
                       (dx backward: dy [M,out] @ W, shape0=out, shape1=in)
  stk_sdd_at_b(a, b, data, …) -> [nnz, blk, blk]
    aᵀ @ b restricted to the BCSR topology (the dW backward).
"""
import torch
from torch import Tensor

from morph.sparse.stk.backend import triton_kernels as backend


@torch.library.custom_op("morph_stk::dds", mutates_args=(), device_types="cuda")
def stk_dds(
    lhs: Tensor,
    data: Tensor,
    offsets: Tensor,
    row_indices: Tensor,
    column_indices: Tensor,
    offsets_t: Tensor,
    column_indices_t: Tensor,
    block_offsets_t: Tensor,
    shape0: int,
    shape1: int,
    transpose_b: bool,
) -> Tensor:
    out = torch.empty((lhs.size(0), shape1), dtype=lhs.dtype, device=lhs.device)
    backend.dds(
        lhs, (shape0, shape1), data,
        offsets, row_indices, column_indices,
        offsets_t, column_indices_t, block_offsets_t,
        transpose_b, out,
    )
    return out


@stk_dds.register_fake
def _stk_dds_fake(
    lhs, data, offsets, row_indices, column_indices,
    offsets_t, column_indices_t, block_offsets_t,
    shape0, shape1, transpose_b,
):
    return lhs.new_empty((lhs.size(0), shape1))


@torch.library.custom_op("morph_stk::sdd_at_b", mutates_args=(), device_types="cuda")
def stk_sdd_at_b(
    a: Tensor,
    b: Tensor,
    data: Tensor,
    offsets: Tensor,
    row_indices: Tensor,
    column_indices: Tensor,
) -> Tensor:
    # aᵀ @ b on the topology of `data`. Mirrors stk SDD.forward(a.t(), b, …):
    # the kernel host fn reads strides off the transposed VIEW, so pass the view
    # (bit-identical launch to the legacy autograd path). `data` supplies only
    # the output block shape — its values are never read.
    lhs = a.t()
    out = torch.empty(data.shape, dtype=lhs.dtype, device=lhs.device)
    backend.sdd(lhs, b, (a.size(1), b.size(1)), out, offsets, row_indices, column_indices)
    return out


@stk_sdd_at_b.register_fake
def _stk_sdd_fake(a, b, data, offsets, row_indices, column_indices):
    return a.new_empty(data.shape)


def _dds_setup_context(ctx, inputs, output):
    (lhs, data, offsets, row_indices, column_indices,
     offsets_t, column_indices_t, block_offsets_t,
     shape0, shape1, transpose_b) = inputs
    ctx.save_for_backward(lhs, data, offsets, row_indices, column_indices,
                          offsets_t, column_indices_t, block_offsets_t)
    ctx.shape0 = shape0
    ctx.shape1 = shape1
    ctx.transpose_b = transpose_b


def _dds_backward(ctx, grad):
    """Byte-faithful port of stk DDS.backward for the contiguous-lhs case
    (the only case the carved forward produces; backend kernels stride-sniff,
    so even a non-contiguous lhs would still compute correctly).

      dlhs  = grad @ (W if transpose_b else Wᵀ)        — another dds
      ddata = (gradᵀ @ lhs  if transpose_b else lhsᵀ @ grad) on the topology — sdd
    """
    (lhs, data, offsets, row_indices, column_indices,
     offsets_t, column_indices_t, block_offsets_t) = ctx.saved_tensors
    shape0, shape1, transpose_b = ctx.shape0, ctx.shape1, ctx.transpose_b

    dy = grad if grad.is_contiguous() else grad.contiguous()

    dlhs = None
    if ctx.needs_input_grad[0]:
        dlhs = stk_dds(
            dy, data, offsets, row_indices, column_indices,
            offsets_t, column_indices_t, block_offsets_t,
            shape1, shape0, not transpose_b,
        )
    ddata = None
    if ctx.needs_input_grad[1]:
        a, b = (dy, lhs) if transpose_b else (lhs, dy)
        ddata = stk_sdd_at_b(a, b, data, offsets, row_indices, column_indices)
    return (dlhs, ddata, None, None, None, None, None, None, None, None, None)


torch.library.register_autograd(
    "morph_stk::dds", _dds_backward, setup_context=_dds_setup_context
)
