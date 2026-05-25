"""Fused Triton kernel: sigmoid gating + weighted branch combine + residual attention.

Eliminates 6 intermediate tensor allocations per layer:
  - gates tensor [B, H, S, 3]
  - 3 intermediate scaled branches
  - combined [B, H, S, D]
  - x_heads [B, H, S, D]

The kernel reads 4 tensors (compress, selected, window, gate_logits, x_input),
fuses sigmoid + weighted sum + alpha*residual + transpose, and writes a single
output [B, S, D_model] ready for W_o.

With 42 effective layers (6 × ~7 loop iterations), this saves 252 allocations per
forward pass.

sm_120 (RTX 5090 / Blackwell) constraints:
  - num_stages=1, num_warps=4
  - tl.sigmoid supported natively
  - BLOCK_SIZE=1024 fits well in L1

Date: 2026-05-06
Branch: 006-looped-block-ell
"""

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# Triton kernel
# =============================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def fused_gate_combine_kernel(
        # Branch outputs — all [B, H, S, D], row-major (contiguous)
        compress_ptr,
        selected_ptr,
        window_ptr,
        # Gate logits [B, S, H*3] — raw pre-sigmoid
        gate_logits_ptr,
        # Per-head residual weight [H] (scalar per head, broadcast over S and D)
        alpha_ptr,
        # Original input x [B, S, H*D] (same layout as D_model dim = H*D)
        x_input_ptr,
        # Output [B, S, H*D]  (transposed layout, ready for W_o)
        out_ptr,
        # Dimensions
        B,   # batch size
        H,   # number of heads
        S,   # sequence length
        D,   # head dim (d_model // n_heads)
        # Strides for [B, H, S, D] tensors (compress, selected, window)
        stride_b_bhs,   # stride over batch   dim in [B,H,S,D]
        stride_h_bhs,   # stride over head    dim in [B,H,S,D]
        stride_s_bhs,   # stride over seq     dim in [B,H,S,D]
        # stride_d is 1 (contiguous), no need to pass
        # Strides for gate_logits [B, S, H*3]
        stride_b_gate,  # stride over batch   dim
        stride_s_gate,  # stride over seq     dim
        # stride over H*3 dim is 1 (contiguous)
        # Strides for x_input [B, S, H*D]
        stride_b_x,
        stride_s_x,
        # Strides for out [B, S, H*D]
        stride_b_out,
        stride_s_out,
        # Block size (constexpr for compiler)
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused gate + combine + residual kernel.

        Grid: (N_blocks,) where N_blocks = cdiv(B*H*S*D, BLOCK_SIZE)

        Each program handles BLOCK_SIZE elements of the flattened [B,H,S,D] space.
        We decode (b, h, s, d) from the flat index, load gate[b,s,h,0..2],
        compute the fused result, and store to out[b, s, h*D+d].
        """
        pid = tl.program_id(0)
        # Flat offsets for this block
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = B * H * S * D
        mask = offsets < total

        # Decode flat index into (b, h, s, d) — row-major [B,H,S,D] order
        # flat = b*(H*S*D) + h*(S*D) + s*D + d
        HSD = H * S * D
        SD = S * D

        b_idx = offsets // HSD
        rem = offsets % HSD
        h_idx = rem // SD
        rem2 = rem % SD
        s_idx = rem2 // D
        d_idx = rem2 % D

        # ----- Load 3 branch values -----
        # [B,H,S,D] with strides (stride_b_bhs, stride_h_bhs, stride_s_bhs, 1)
        bhs_offset = b_idx * stride_b_bhs + h_idx * stride_h_bhs + s_idx * stride_s_bhs + d_idx
        v_compress = tl.load(compress_ptr + bhs_offset, mask=mask, other=0.0)
        v_selected = tl.load(selected_ptr + bhs_offset, mask=mask, other=0.0)
        v_window   = tl.load(window_ptr   + bhs_offset, mask=mask, other=0.0)

        # ----- Load gate logits & compute sigmoid gates -----
        # gate_logits [B, S, H*3]: for (b, s, h) load 3 consecutive values
        # offset into [B, S, H*3]: b*stride_b_gate + s*stride_s_gate + h*3 + {0,1,2}
        gate_base = b_idx * stride_b_gate + s_idx * stride_s_gate + h_idx * 3
        g0 = tl.sigmoid(tl.load(gate_logits_ptr + gate_base,     mask=mask, other=0.0).to(tl.float32))
        g1 = tl.sigmoid(tl.load(gate_logits_ptr + gate_base + 1, mask=mask, other=0.0).to(tl.float32))
        g2 = tl.sigmoid(tl.load(gate_logits_ptr + gate_base + 2, mask=mask, other=0.0).to(tl.float32))

        # ----- Load alpha [H] — per-head scalar -----
        alpha = tl.load(alpha_ptr + h_idx, mask=mask, other=0.0).to(tl.float32)

        # ----- Load x_input residual -----
        # x_input [B, S, H*D]: offset = b*stride_b_x + s*stride_s_x + h*D + d
        x_offset = b_idx * stride_b_x + s_idx * stride_s_x + h_idx * D + d_idx
        x_val = tl.load(x_input_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)

        # ----- Fused compute -----
        v_compress = v_compress.to(tl.float32)
        v_selected = v_selected.to(tl.float32)
        v_window   = v_window.to(tl.float32)

        result = g0 * v_compress + g1 * v_selected + g2 * v_window + alpha * x_val

        # ----- Store to out[b, s, h*D + d] -----
        # out [B, S, H*D]: this IS the transpose+reshape step
        out_offset = b_idx * stride_b_out + s_idx * stride_s_out + h_idx * D + d_idx
        tl.store(out_ptr + out_offset, result.to(v_compress.dtype), mask=mask)


# =============================================================================
# Python wrapper
# =============================================================================


def _fused_gate_combine_triton(out_compress, out_selected, out_window, gate_logits, alpha, x_input):
    """Raw Triton forward — no autograd."""
    B, H, S, D = out_compress.shape
    D_model = H * D
    alpha_flat = alpha.reshape(H)

    out_compress  = out_compress.contiguous()
    out_selected  = out_selected.contiguous()
    out_window    = out_window.contiguous()
    gate_logits   = gate_logits.contiguous()
    alpha_flat    = alpha_flat.contiguous()
    x_input       = x_input.contiguous()

    out = torch.empty(B, S, D_model, dtype=out_compress.dtype, device=out_compress.device)

    total_elements = B * H * S * D
    BLOCK_SIZE = 1024

    cap = torch.cuda.get_device_capability(out_compress.device)
    launch_kwargs = dict(num_stages=1, num_warps=4) if cap[0] >= 12 else dict(num_warps=4)

    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    fused_gate_combine_kernel[grid](
        out_compress, out_selected, out_window, gate_logits,
        alpha_flat, x_input, out,
        B, H, S, D,
        out_compress.stride(0), out_compress.stride(1), out_compress.stride(2),
        gate_logits.stride(0), gate_logits.stride(1),
        x_input.stride(0), x_input.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE, **launch_kwargs,
    )
    return out


class _FusedGateCombineFunction(torch.autograd.Function):
    """Fused forward (Triton), reference backward (correct gradients)."""

    @staticmethod
    def forward(ctx, out_compress, out_selected, out_window, gate_logits, alpha, x_input):
        ctx.save_for_backward(out_compress, out_selected, out_window, gate_logits, alpha, x_input)
        return _fused_gate_combine_triton(out_compress, out_selected, out_window, gate_logits, alpha, x_input)

    @staticmethod
    def backward(ctx, grad_output):
        out_compress, out_selected, out_window, gate_logits, alpha, x_input = ctx.saved_tensors
        with torch.enable_grad():
            c_ = out_compress.detach().requires_grad_(True)
            s_ = out_selected.detach().requires_grad_(True)
            w_ = out_window.detach().requires_grad_(True)
            g_ = gate_logits.detach().requires_grad_(True)
            a_ = alpha.detach().requires_grad_(True)
            x_ = x_input.detach().requires_grad_(True)
            out = fused_gate_combine_reference(c_, s_, w_, g_, a_, x_)
            out.backward(grad_output)
        return c_.grad, s_.grad, w_.grad, g_.grad, a_.grad, x_.grad


def fused_gate_combine(
    out_compress: Tensor,   # [B, H, S, D]
    out_selected: Tensor,   # [B, H, S, D]
    out_window:   Tensor,   # [B, H, S, D]
    gate_logits:  Tensor,   # [B, S, H*3]
    alpha:        Tensor,   # [H] or [H, 1, 1] — will be flattened
    x_input:      Tensor,   # [B, S, D_model] where D_model = H*D
) -> Tensor:                # [B, S, D_model]
    """Fused sigmoid gating + branch combination + residual attention.

    Forward: Triton kernel (eliminates 6 intermediate allocations).
    Backward: PyTorch reference (correct gradients for all inputs).

    Args:
        out_compress: [B, H, S, D] compressed branch output
        out_selected: [B, H, S, D] MoSA selected branch output
        out_window:   [B, H, S, D] sliding window branch output
        gate_logits:  [B, S, H*3] raw gate logits (pre-sigmoid)
        alpha:        [H] or [H, 1, 1] per-head residual weight
        x_input:      [B, S, D_model] original input for residual

    Returns:
        [B, S, D_model] combined output ready for W_o projection
    """
    if not TRITON_AVAILABLE:
        return fused_gate_combine_reference(
            out_compress, out_selected, out_window, gate_logits, alpha, x_input
        )

    return _FusedGateCombineFunction.apply(
        out_compress, out_selected, out_window, gate_logits, alpha, x_input
    )


# =============================================================================
# Reference implementation (pure PyTorch, for testing and CPU fallback)
# =============================================================================


def fused_gate_combine_reference(
    out_compress: Tensor,
    out_selected: Tensor,
    out_window:   Tensor,
    gate_logits:  Tensor,
    alpha:        Tensor,
    x_input:      Tensor,
) -> Tensor:
    """Pure PyTorch reference — numerically equivalent to the kernel.

    Args / Returns: same as fused_gate_combine.
    """
    B, H, S, D = out_compress.shape
    # Sigmoid gates and reshape to [B, H, S, 3]
    gates = torch.sigmoid(gate_logits).reshape(B, S, H, 3).permute(0, 2, 1, 3)
    # Weighted branch sum [B, H, S, D]
    combined = (
        gates[..., 0:1] * out_compress
        + gates[..., 1:2] * out_selected
        + gates[..., 2:3] * out_window
    )
    # Residual: alpha [H,1,1] * x_heads [B,H,S,D]
    x_heads = x_input.reshape(B, S, H, D).transpose(1, 2)  # [B,H,S,D]
    alpha_bcast = alpha.reshape(H, 1, 1)
    out = combined + alpha_bcast * x_heads
    # Transpose+reshape: [B,H,S,D] → [B,S,H*D]
    return out.transpose(1, 2).reshape(B, S, H * D)


# =============================================================================
# Test + benchmark
# =============================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = "cuda"

    B, H, S, D = 4, 12, 512, 64
    D_model = H * D  # 768

    print(f"Config: B={B}, H={H}, S={S}, D={D}, D_model={D_model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")
    print()

    # Create random inputs (bf16 like real training)
    dtype = torch.bfloat16
    out_compress  = torch.randn(B, H, S, D, device=device, dtype=dtype)
    out_selected  = torch.randn(B, H, S, D, device=device, dtype=dtype)
    out_window    = torch.randn(B, H, S, D, device=device, dtype=dtype)
    gate_logits   = torch.randn(B, S, H * 3, device=device, dtype=dtype)
    alpha         = torch.full((H,), 0.1, device=device, dtype=dtype)
    x_input       = torch.randn(B, S, D_model, device=device, dtype=dtype)

    # ── Correctness check ────────────────────────────────────────────────────
    print("=== Correctness check ===")
    ref = fused_gate_combine_reference(
        out_compress, out_selected, out_window, gate_logits, alpha, x_input
    )
    kernel_out = fused_gate_combine(
        out_compress, out_selected, out_window, gate_logits, alpha, x_input
    )

    # Compare in fp32 to avoid bf16 precision noise
    ref_f32    = ref.float()
    kernel_f32 = kernel_out.float()
    max_err    = (ref_f32 - kernel_f32).abs().max().item()
    rel_err    = (ref_f32 - kernel_f32).abs().mean().item() / (ref_f32.abs().mean().item() + 1e-8)

    print(f"Max absolute error : {max_err:.6f}")
    print(f"Mean relative error: {rel_err:.6f}")
    # bf16 ULP at |x|~2 is 2^-6 = 0.015625; allow 2 ULPs for rounding accumulation
    # The reference itself runs in bf16 mixed precision so the threshold is loose
    assert max_err < 0.05, f"Max error {max_err} too large for bf16 (threshold 0.05 = ~3 ULPs)"
    print("✓ Correctness: PASS")
    print()

    # ── Benchmark ────────────────────────────────────────────────────────────
    N_WARMUP = 20
    N_ITER   = 100

    # Warm up both paths
    for _ in range(N_WARMUP):
        _ = fused_gate_combine_reference(
            out_compress, out_selected, out_window, gate_logits, alpha, x_input
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = fused_gate_combine_reference(
            out_compress, out_selected, out_window, gate_logits, alpha, x_input
        )
    torch.cuda.synchronize()
    ref_time_ms = (time.perf_counter() - t0) * 1000 / N_ITER

    for _ in range(N_WARMUP):
        _ = fused_gate_combine(
            out_compress, out_selected, out_window, gate_logits, alpha, x_input
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = fused_gate_combine(
            out_compress, out_selected, out_window, gate_logits, alpha, x_input
        )
    torch.cuda.synchronize()
    kernel_time_ms = (time.perf_counter() - t0) * 1000 / N_ITER

    ref_steps_s    = 1000.0 / ref_time_ms
    kernel_steps_s = 1000.0 / kernel_time_ms

    print("=== Benchmark (100 iterations) ===")
    print(f"Reference (PyTorch): {ref_time_ms:.3f} ms/iter  ({ref_steps_s:.1f} step/s)")
    print(f"Kernel  (Triton):    {kernel_time_ms:.3f} ms/iter  ({kernel_steps_s:.1f} step/s)")
    print(f"Speedup: {ref_time_ms / kernel_time_ms:.2f}×")
    print()

    # ── Memory allocation estimate ────────────────────────────────────────────
    element_bytes = 2  # bf16
    tensor_elems  = B * H * S * D
    model_elems   = B * S * D_model

    # Intermediates eliminated per layer:
    #   gates [B,S,H,3], 3 scaled branches [B,H,S,D] each, combined [B,H,S,D], x_heads [B,H,S,D]
    intermediates = (
        B * S * H * 3 +      # gates
        3 * tensor_elems +    # 3 scaled branches
        tensor_elems +        # combined
        tensor_elems          # x_heads
    )
    saved_mb_per_layer = intermediates * element_bytes / (1024 ** 2)
    n_effective_layers = 42
    saved_mb_total = saved_mb_per_layer * n_effective_layers

    print("=== Memory savings ===")
    print(f"Intermediate tensors eliminated per layer: ~{intermediates * element_bytes / 1024:.1f} KB")
    print(f"  ({saved_mb_per_layer:.2f} MB per layer)")
    print(f"Total across {n_effective_layers} effective layers: {saved_mb_total:.1f} MB saved per forward pass")
