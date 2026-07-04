"""Fused ReMoE TileRouter elementwise tail (perf: launch-count cut, class A).

Three tiny fusions around the router's top-k gate tail (routing.py
``TileRouter.forward`` steps 3-6 + the load-balance activation mask), each an
``autograd.Function`` with a Triton forward on CUDA and an aten fallback
elsewhere (CPU / ``force_eager`` / unsupported dtype):

  K1 ``pk_logits``       : logits[n, i*ns+j] = (sa[n,i] + sb[n,j]) + bias[i*ns+j]
                           (product-key broadcast add + group-bias add; the
                           n_sub_keys² == n_tile_groups branch only)
  K2 ``sub_relu``        : gates0 = relu(logits - threshold)
  K3 ``normalize_mask``  : gates = gates0 * t ;  mask = (gates > 0).float()
                           (t = activation_k / clamp(sum) stays EAGER — see below)

Bit-exactness contract (class A):
  * Every fused op is ELEMENTWISE and correctly rounded (IEEE add/mul/max/cmp).
    A fused chain of per-element ops is bitwise-identical to the eager per-op
    chain — no reduction is touched, so no reassociation is possible.
  * K1's mixed-dtype case (bf16 scores + fp32 bias under autocast) replicates
    aten exactly: the two bf16 scores are added via an exact fp32 sum rounded
    to bf16 (aten opmath; the 24-bit intermediate makes the double rounding
    innocuous for 8-bit-mantissa operands), then type-promoted to fp32 for the
    bias add — the same op sequence eager runs.
  * Every REDUCTION stays in aten: forward (topk / sum / mean / var untouched)
    and backward (the sum_to_size-equivalent ``.sum`` calls below are the same
    kernels autograd's broadcast backward uses).
  * The ``activation_k / gate_sum`` division is NOT fused: Python-scalar /
    Tensor lowering is aten-internal (reciprocal-vs-div is version-dependent),
    so it is computed eager and its result ``t`` is passed in. K3's backward
    returns d(t) to autograd, which routes it through the eager clamp/div
    nodes — no div-derivative replication anywhere.
  * NaN semantics replicated: relu(NaN) = NaN (torch clamp_min propagates), so
    K2 uses an explicit NaN-passthrough around ``tl.maximum`` (libdevice fmax
    would drop the NaN); (NaN > 0) = False in both aten and Triton.

Backward is analytic aten mirroring the eager derivative sequence op-for-op;
it is shared by the Triton and fallback paths and is verified bitwise on CPU
in ``scratchpad/parity_router_tail.py``.

GPU-blind edges (single-5090 discipline: this module was written without a GPU
run): the Triton kernels compile/launch only on CUDA, so the CPU parity
validates the fallback forward + the shared analytic backward; the kernels'
first compile and their forward bitwise-ness need the module probe + loss-trace
noise-floor gate. Kill switch: ``MORPH_FUSED_ROUTER_TAIL=0`` (env, read by
routing.py) or ``routing.set_fused_router_tail(False)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._eager_flag import force_eager, kernel_fence

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


_BLOCK = 1024


if TRITON_AVAILABLE:

    @triton.jit
    def _k1_pk_logits(sa_ptr, sb_ptr, bias_ptr, out_ptr, total, G, NS,
                      SRC_BF16: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        n = offs // G
        g = offs % G
        i = g // NS
        j = g % NS
        a = tl.load(sa_ptr + n * NS + i, mask=m, other=0.0).to(tl.float32)
        b = tl.load(sb_ptr + n * NS + j, mask=m, other=0.0).to(tl.float32)
        u = a + b                       # exact in fp32 for bf16 operands
        if SRC_BF16:
            u = u.to(tl.bfloat16).to(tl.float32)   # aten opmath round-to-bf16
        bias = tl.load(bias_ptr + g, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, u + bias, mask=m)

    @triton.jit
    def _k2_sub_relu(x_ptr, t_ptr, out_ptr, total, G, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        n = offs // G
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        t = tl.load(t_ptr + n, mask=m, other=0.0)
        d = x - t
        # relu = clamp_min(d, 0): propagate NaN like aten (fmax would drop it)
        y = tl.where(d != d, d, tl.maximum(d, 0.0))
        tl.store(out_ptr + offs, y, mask=m)

    @triton.jit
    def _k3_norm_mask(g0_ptr, t_ptr, gates_ptr, mask_ptr, total, G,
                      BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        n = offs // G
        g0 = tl.load(g0_ptr + offs, mask=m, other=0.0)
        t = tl.load(t_ptr + n, mask=m, other=0.0)
        gates = g0 * t                                  # one IEEE mul, same as aten
        mask_val = tl.where(gates > 0, 1.0, 0.0)        # (NaN > 0) = False, like aten
        tl.store(gates_ptr + offs, gates, mask=m)
        tl.store(mask_ptr + offs, mask_val.to(tl.float32), mask=m)


def _use_triton(*tensors: Tensor) -> bool:
    return (TRITON_AVAILABLE and not force_eager()
            and all(t.is_cuda for t in tensors))


# ─── K1: product-key broadcast add + group-bias add ──────────────────────────


class _PKLogits(torch.autograd.Function):
    """logits = (sa ⊕ sb).reshape(N, ns²) + bias — the 1:1 product-key branch."""

    @staticmethod
    def forward(ctx, sa: Tensor, sb: Tensor, bias: Tensor) -> Tensor:
        N, ns = sa.shape
        G = ns * ns
        out_dtype = torch.promote_types(sa.dtype, bias.dtype)
        supported = (sa.dtype == sb.dtype
                     and bias.dtype == torch.float32
                     and sa.dtype in (torch.float32, torch.bfloat16))
        if supported and _use_triton(sa, sb, bias):
            sa_c, sb_c, bias_c = sa.contiguous(), sb.contiguous(), bias.contiguous()
            out = torch.empty(N, G, device=sa.device, dtype=out_dtype)
            total = N * G
            grid = (triton.cdiv(total, _BLOCK),)
            _k1_pk_logits[grid](sa_c, sb_c, bias_c, out, total, G, ns,
                                SRC_BF16=(sa.dtype == torch.bfloat16), BLOCK=_BLOCK)
        else:
            # aten fallback — the exact eager op sequence
            out = (sa.unsqueeze(2) + sb.unsqueeze(1)).reshape(N, G) + bias.unsqueeze(0)
        ctx.ns = ns
        ctx.dtypes = (sa.dtype, sb.dtype, bias.dtype)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        ns = ctx.ns
        sa_dt, sb_dt, bias_dt = ctx.dtypes
        N = grad.shape[0]
        # eager: bias-add bwd = sum_to_size([1,G]) → squeeze (fp32, bias dtype);
        # the mixed-dtype promotion casts the grad back to the SOURCE dtype at the
        # promotion boundary — i.e. BEFORE the broadcast-add backward — so the
        # dsa/dsb reductions run in the source (bf16) dtype, exactly like eager
        # (verified bitwise in parity_router_tail.py; sum-then-cast fails at
        # bf16-quantum scale).
        dbias = grad.sum(0)
        if dbias.dtype != bias_dt:
            dbias = dbias.to(bias_dt)
        g3 = grad.to(sa_dt).reshape(N, ns, ns)
        dsa = g3.sum(2)
        dsb = g3.sum(1)
        if dsb.dtype != sb_dt:
            dsb = dsb.to(sb_dt)
        return dsa, dsb, dbias


# ─── K2: threshold-shift + relu ───────────────────────────────────────────────


class _SubRelu(torch.autograd.Function):
    """gates0 = relu(logits - threshold), threshold [N,1] broadcast."""

    @staticmethod
    def forward(ctx, logits: Tensor, threshold: Tensor) -> Tensor:
        if logits.dtype == torch.float32 and _use_triton(logits, threshold):
            lc, tc = logits.contiguous(), threshold.contiguous()
            out = torch.empty_like(lc)
            total = lc.numel()
            grid = (triton.cdiv(total, _BLOCK),)
            _k2_sub_relu[grid](lc, tc, out, total, lc.shape[-1], BLOCK=_BLOCK)
        else:
            out = torch.relu(logits - threshold)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        (out,) = ctx.saved_tensors
        # eager: relu bwd = threshold_backward(grad, result, 0) — call the SAME
        # aten op (1 kernel; a where() recomposition costs 2 and re-derives the
        # mask); sub bwd = grad (self, pass-through) and sum_to_size(-grad)
        # (broadcast other).
        dmasked = torch.ops.aten.threshold_backward(grad, out, 0)
        dthr = (-dmasked).sum(-1, keepdim=True)
        return dmasked, dthr


# ─── K3: gate normalization + activation mask ────────────────────────────────


class _NormalizeMask(torch.autograd.Function):
    """gates = gates0 * t;  mask = (gates > 0).float() (non-differentiable).

    t = activation_k / clamp(gate_sum) is computed EAGER by the caller; d(t)
    flows back through the eager clamp/div autograd nodes.
    """

    @staticmethod
    def forward(ctx, gates0: Tensor, t: Tensor):
        ctx.set_materialize_grads(False)   # no zero-fill for the non-diff mask output
        if gates0.dtype == torch.float32 and _use_triton(gates0, t):
            gc, tc = gates0.contiguous(), t.contiguous()
            gates = torch.empty_like(gc)
            mask = torch.empty_like(gc, dtype=torch.float32)
            total = gc.numel()
            grid = (triton.cdiv(total, _BLOCK),)
            _k3_norm_mask[grid](gc, tc, gates, mask, total, gc.shape[-1],
                                BLOCK=_BLOCK)
        else:
            gates = gates0 * t
            mask = (gates > 0).float()
        ctx.save_for_backward(gates0, t)
        ctx.mark_non_differentiable(mask)
        return gates, mask

    @staticmethod
    def backward(ctx, grad: Tensor, _grad_mask):
        if grad is None:   # only the non-diff mask was consumed downstream
            return None, None
        gates0, t = ctx.saved_tensors
        # eager mul bwd: d(gates0) = grad * t (broadcast); d(t) = sum_to_size(grad * gates0)
        dg0 = grad * t
        dt = (grad * gates0).sum(-1, keepdim=True)
        return dg0, dt


# ─── Dynamo-fenced entry points (autograd.Functions are opaque to compile) ───


@kernel_fence
def pk_logits(scores_a: Tensor, scores_b: Tensor, group_bias: Tensor) -> Tensor:
    return _PKLogits.apply(scores_a, scores_b, group_bias)


@kernel_fence
def sub_relu(group_logits: Tensor, threshold: Tensor) -> Tensor:
    return _SubRelu.apply(group_logits, threshold)


@kernel_fence
def normalize_mask(gates0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    return _NormalizeMask.apply(gates0, t)
