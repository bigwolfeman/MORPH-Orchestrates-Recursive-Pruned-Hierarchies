"""Fused Triton kernels for looped transformer hot-path operations.

Targets the inner loop of MORPH's looped core, which runs the core transformer
layers x 8-14 iterations. Each elementwise op is a separate CUDA kernel launch;
fusing them eliminates memory round-trips and launch overhead.

Kernels:
    1. fused_rmsnorm           — standalone RMSNorm (for input_norm, output norm)
    2. fused_residual_rmsnorm  — residual add + RMSNorm in one pass
    3. fused_diagonal_injection — decay*h + dt*e with precomputed decay/dt
    4. fused_gelu               — GELU activation (tanh approximation)

Hardware: RTX 5090 (SM120) — num_stages=1, num_warps=4.

All kernels:
    - Accept bf16 inputs, accumulate in fp32, output bf16
    - Handle arbitrary batch*seq flattened as rows
    - Backward passes avoid atomic operations (use PyTorch reductions for
      parameter gradients instead — this is critical for performance)
"""

import math
import os
import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = not os.environ.get("DISABLE_FUSED_KERNELS", "")
except ImportError:
    TRITON_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# Launch config helper
# ─────────────────────────────────────────────────────────────────────

def _get_block_d(D: int) -> int:
    """Next power of 2 >= D for Triton block size."""
    return triton.next_power_of_2(D)


def _get_launch_kwargs() -> dict:
    """SM120-safe launch kwargs."""
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 12:
        return dict(num_stages=1, num_warps=4)
    return dict(num_stages=2, num_warps=4)


# ─────────────────────────────────────────────────────────────────────
# Kernel 1: Fused RMSNorm forward
# Fuses: variance = mean(x^2) → rms = sqrt(var + eps) → x/rms * scale
# ─────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, Scale_ptr, Out_ptr,
        D: tl.constexpr,
        stride_x_row, stride_out_row,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        clamp_scale: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        x = tl.load(X_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        rms = tl.sqrt(var + eps)
        x_norm = x / rms

        scale = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        if clamp_scale:
            scale = tl.minimum(tl.maximum(scale, 0.1), 10.0)

        tl.store(Out_ptr + row * stride_out_row + offs, (x_norm * scale).to(tl.bfloat16), mask=mask)

    # ─────────────────────────────────────────────────────────────────────
    # Kernel 1b: RMSNorm backward (grad_x only, NO atomic accumulation)
    # grad_scale is computed in Python via torch reduction — much faster.
    # ─────────────────────────────────────────────────────────────────────

    @triton.jit
    def _rmsnorm_bwd_kernel(
        Grad_ptr, X_ptr, Scale_ptr, GradX_ptr,
        D: tl.constexpr,
        stride_grad_row, stride_x_row, stride_gx_row,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        clamp_scale: tl.constexpr,
    ):
        """RMSNorm backward: compute grad_x only (no atomics)."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        grad = tl.load(Grad_ptr + row * stride_grad_row + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        if clamp_scale:
            scale = tl.minimum(tl.maximum(scale, 0.1), 10.0)

        # Recompute forward
        var = tl.sum(x * x, axis=0) / D
        rms = tl.sqrt(var + eps)
        rms_inv = 1.0 / rms
        x_norm = x * rms_inv

        # grad_x = scale/rms * (grad - x_norm * mean(grad * scale * x_norm))
        grad_normed = grad * scale
        dot = tl.sum(grad_normed * x_norm, axis=0) / D
        grad_x = (grad_normed - x_norm * dot) * rms_inv

        tl.store(GradX_ptr + row * stride_gx_row + offs, grad_x.to(tl.bfloat16), mask=mask)

    # ─────────────────────────────────────────────────────────────────────
    # Kernel 2: Fused Residual Add + RMSNorm
    # Pattern: hidden = residual + sublayer_out; norm = RMSNorm(hidden)
    # ─────────────────────────────────────────────────────────────────────

    @triton.jit
    def _rmsnorm_residual_fwd_kernel(
        X_ptr, Residual_ptr, Scale_ptr, NormOut_ptr, ResOut_ptr,
        D: tl.constexpr,
        stride_x_row, stride_res_row, stride_nout_row, stride_rout_row,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        clamp_scale: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        x = tl.load(X_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(Residual_ptr + row * stride_res_row + offs, mask=mask, other=0.0).to(tl.float32)

        hidden = res + x
        tl.store(ResOut_ptr + row * stride_rout_row + offs, hidden.to(tl.bfloat16), mask=mask)

        var = tl.sum(hidden * hidden, axis=0) / D
        rms = tl.sqrt(var + eps)
        normed = hidden / rms

        scale = tl.load(Scale_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        if clamp_scale:
            scale = tl.minimum(tl.maximum(scale, 0.1), 10.0)

        tl.store(NormOut_ptr + row * stride_nout_row + offs, (normed * scale).to(tl.bfloat16), mask=mask)

    # ─────────────────────────────────────────────────────────────────────
    # Kernel 3: Fused Diagonal Injection
    # out = decay * h + dt * e  (3 elementwise → 1 kernel)
    # ─────────────────────────────────────────────────────────────────────

    @triton.jit
    def _diagonal_injection_kernel(
        H_ptr, E_ptr, Decay_ptr, Dt_ptr, Out_ptr,
        D: tl.constexpr,
        stride_h_row, stride_e_row, stride_out_row,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D

        h = tl.load(H_ptr + row * stride_h_row + offs, mask=mask, other=0.0).to(tl.float32)
        e = tl.load(E_ptr + row * stride_e_row + offs, mask=mask, other=0.0).to(tl.float32)
        decay = tl.load(Decay_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dt = tl.load(Dt_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        tl.store(Out_ptr + row * stride_out_row + offs, (decay * h + dt * e).to(tl.bfloat16), mask=mask)

    # ─────────────────────────────────────────────────────────────────────
    # Kernel 4: Fused GELU (tanh approximation)
    # ─────────────────────────────────────────────────────────────────────

    @triton.jit
    def _gelu_fwd_kernel(
        X_ptr, Out_ptr,
        N: tl.constexpr,
        stride_x, stride_out,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        COEFF: tl.constexpr = 0.044715
        inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
        # Clamp to prevent exp overflow: exp(2*44) ~ 3.5e38 ≈ fp32 max.
        # |inner| > 10 → tanh ≈ ±1 to >10 decimal places, so clamping is lossless.
        inner = tl.minimum(tl.maximum(inner, -44.0), 44.0)
        exp_2inner = tl.exp(2.0 * inner)
        tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0)
        out = 0.5 * x * (1.0 + tanh_val)

        tl.store(Out_ptr + row * stride_out + offs, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gelu_bwd_kernel(
        Grad_ptr, X_ptr, Out_ptr,
        N: tl.constexpr,
        stride_grad, stride_x, stride_out,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        grad = tl.load(Grad_ptr + row * stride_grad + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        COEFF: tl.constexpr = 0.044715

        inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
        # Clamp to prevent exp overflow (same as forward — see comment there)
        inner = tl.minimum(tl.maximum(inner, -44.0), 44.0)
        exp_2inner = tl.exp(2.0 * inner)
        tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0)

        dtanh = 1.0 - tanh_val * tanh_val
        dinner_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x)
        dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * dinner_dx

        tl.store(Out_ptr + row * stride_out + offs, (grad * dgelu).to(tl.bfloat16), mask=mask)


# ─────────────────────────────────────────────────────────────────────
# Autograd wrappers
# ─────────────────────────────────────────────────────────────────────

class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, eps: float, clamp_scale: bool) -> Tensor:
        shape = x.shape
        D = shape[-1]
        x_2d = x.reshape(-1, D).contiguous()
        rows = x_2d.shape[0]

        out = torch.empty_like(x_2d)
        BLOCK_D = _get_block_d(D)
        kw = _get_launch_kwargs()

        _rmsnorm_fwd_kernel[(rows,)](
            x_2d, scale, out,
            D=D, stride_x_row=x_2d.stride(0), stride_out_row=out.stride(0),
            eps=eps, BLOCK_D=BLOCK_D, clamp_scale=clamp_scale, **kw,
        )

        ctx.save_for_backward(x_2d, scale)
        ctx.eps = eps
        ctx.clamp_scale = clamp_scale
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_2d, scale = ctx.saved_tensors
        D = x_2d.shape[-1]
        rows = x_2d.shape[0]
        BLOCK_D = _get_block_d(D)
        kw = _get_launch_kwargs()

        grad_2d = grad_output.reshape(-1, D).contiguous()
        grad_x = torch.empty_like(x_2d)

        # Triton kernel computes grad_x (per-row, no atomics)
        _rmsnorm_bwd_kernel[(rows,)](
            grad_2d, x_2d, scale, grad_x,
            D=D,
            stride_grad_row=grad_2d.stride(0),
            stride_x_row=x_2d.stride(0),
            stride_gx_row=grad_x.stride(0),
            eps=ctx.eps, BLOCK_D=BLOCK_D, clamp_scale=ctx.clamp_scale,
            **kw,
        )

        # grad_scale via PyTorch reduction (fast, no atomics)
        # Recompute x_norm for grad_scale: x_norm = x / rms
        var = (x_2d.float() ** 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(var + ctx.eps)
        x_norm = x_2d.float() / rms
        grad_scale = (grad_2d.float() * x_norm).sum(dim=0).to(scale.dtype)

        return grad_x.reshape(ctx.shape), grad_scale, None, None


class _FusedResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sublayer_output: Tensor, residual: Tensor,
                scale: Tensor, eps: float, clamp_scale: bool):
        shape = sublayer_output.shape
        D = shape[-1]
        x_2d = sublayer_output.reshape(-1, D).contiguous()
        res_2d = residual.reshape(-1, D).contiguous()
        rows = x_2d.shape[0]

        norm_out = torch.empty_like(x_2d)
        res_out = torch.empty_like(x_2d)
        BLOCK_D = _get_block_d(D)
        kw = _get_launch_kwargs()

        _rmsnorm_residual_fwd_kernel[(rows,)](
            x_2d, res_2d, scale, norm_out, res_out,
            D=D,
            stride_x_row=x_2d.stride(0), stride_res_row=res_2d.stride(0),
            stride_nout_row=norm_out.stride(0), stride_rout_row=res_out.stride(0),
            eps=eps, BLOCK_D=BLOCK_D, clamp_scale=clamp_scale, **kw,
        )

        # Save hidden (res_out) for backward
        ctx.save_for_backward(res_out, scale)
        ctx.eps = eps
        ctx.clamp_scale = clamp_scale
        ctx.shape = shape
        return res_out.reshape(shape), norm_out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_hidden: Tensor, grad_norm: Tensor):
        hidden_2d, scale = ctx.saved_tensors
        D = hidden_2d.shape[-1]
        rows = hidden_2d.shape[0]
        BLOCK_D = _get_block_d(D)
        kw = _get_launch_kwargs()

        grad_norm_2d = grad_norm.reshape(-1, D).contiguous()
        grad_x = torch.empty_like(hidden_2d)

        # Triton: grad w.r.t. hidden through RMSNorm (no atomics)
        _rmsnorm_bwd_kernel[(rows,)](
            grad_norm_2d, hidden_2d, scale, grad_x,
            D=D,
            stride_grad_row=grad_norm_2d.stride(0),
            stride_x_row=hidden_2d.stride(0),
            stride_gx_row=grad_x.stride(0),
            eps=ctx.eps, BLOCK_D=BLOCK_D, clamp_scale=ctx.clamp_scale,
            **kw,
        )

        # grad_scale via PyTorch reduction
        var = (hidden_2d.float() ** 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(var + ctx.eps)
        x_norm = hidden_2d.float() / rms
        grad_scale = (grad_norm_2d.float() * x_norm).sum(dim=0).to(scale.dtype)

        # Total gradient through hidden = grad from norm backward + direct
        grad_hidden_2d = grad_hidden.reshape(-1, D)
        total_grad = (grad_x + grad_hidden_2d).reshape(ctx.shape)

        return total_grad, total_grad, grad_scale, None, None


class _FusedDiagonalInjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: Tensor, e: Tensor, decay: Tensor, dt: Tensor) -> Tensor:
        shape = h.shape
        D = shape[-1]
        h_2d = h.reshape(-1, D).contiguous()
        e_2d = e.reshape(-1, D).contiguous()
        rows = h_2d.shape[0]

        out = torch.empty_like(h_2d)
        BLOCK_D = _get_block_d(D)
        kw = _get_launch_kwargs()

        _diagonal_injection_kernel[(rows,)](
            h_2d, e_2d, decay, dt, out,
            D=D,
            stride_h_row=h_2d.stride(0), stride_e_row=e_2d.stride(0),
            stride_out_row=out.stride(0),
            BLOCK_D=BLOCK_D, **kw,
        )

        ctx.save_for_backward(decay, dt)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        decay, dt = ctx.saved_tensors
        # Simple elementwise: grad_h = grad * decay, grad_e = grad * dt
        return grad_output * decay, grad_output * dt, None, None


class _FusedGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        shape = x.shape
        N = shape[-1]
        x_2d = x.reshape(-1, N).contiguous()
        rows = x_2d.shape[0]

        out = torch.empty_like(x_2d)
        BLOCK_N = _get_block_d(N)
        kw = _get_launch_kwargs()

        _gelu_fwd_kernel[(rows,)](
            x_2d, out,
            N=N, stride_x=x_2d.stride(0), stride_out=out.stride(0),
            BLOCK_N=BLOCK_N, **kw,
        )

        ctx.save_for_backward(x_2d)
        ctx.shape = shape
        ctx.N = N
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_2d, = ctx.saved_tensors
        N = ctx.N
        rows = x_2d.shape[0]
        BLOCK_N = _get_block_d(N)
        kw = _get_launch_kwargs()

        grad_2d = grad_output.reshape(-1, N).contiguous()
        grad_x = torch.empty_like(x_2d)

        _gelu_bwd_kernel[(rows,)](
            grad_2d, x_2d, grad_x,
            N=N,
            stride_grad=grad_2d.stride(0), stride_x=x_2d.stride(0),
            stride_out=grad_x.stride(0),
            BLOCK_N=BLOCK_N, **kw,
        )

        return grad_x.reshape(ctx.shape)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def fused_rmsnorm(x: Tensor, scale: Tensor, eps: float = 1e-5,
                  clamp_scale: bool = True) -> Tensor:
    """Drop-in replacement for RMSNorm.forward()."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        raise RuntimeError("Requires CUDA + Triton")
    return _FusedRMSNorm.apply(x.contiguous(), scale, eps, clamp_scale)


def fused_residual_rmsnorm(sublayer_output: Tensor, residual: Tensor,
                           scale: Tensor, eps: float = 1e-5,
                           clamp_scale: bool = True):
    """Fused residual add + RMSNorm.

    Returns (hidden, norm_out) where hidden = residual + sublayer_output
    and norm_out = RMSNorm(hidden).
    """
    if not TRITON_AVAILABLE or not sublayer_output.is_cuda:
        raise RuntimeError("Requires CUDA + Triton")
    return _FusedResidualRMSNorm.apply(
        sublayer_output.contiguous(), residual.contiguous(),
        scale, eps, clamp_scale
    )


def fused_diagonal_injection(h: Tensor, e: Tensor,
                             decay: Tensor, dt: Tensor) -> Tensor:
    """Fused diagonal injection: out = decay * h + dt * e."""
    if not TRITON_AVAILABLE or not h.is_cuda:
        raise RuntimeError("Requires CUDA + Triton")
    return _FusedDiagonalInjection.apply(h.contiguous(), e.contiguous(), decay, dt)


def fused_gelu(x: Tensor) -> Tensor:
    """Fused GELU activation with tanh approximation."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        raise RuntimeError("Requires CUDA + Triton")
    return _FusedGELU.apply(x.contiguous())
