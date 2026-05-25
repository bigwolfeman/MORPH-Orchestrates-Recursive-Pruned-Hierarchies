"""Triton kernels for Block-ELL routed forward pass.

This module provides GPU-accelerated forward pass for block-sparse matrix
multiplication with per-token variable K (number of active blocks per row).
This enables routing to skip inactive macro-tile columns at forward time.

Key difference from block_ell_forward.py:
- K is a compile-time upper bound K_MAX (same as old K)
- Each token can have a different k_active <= K_MAX
- Inner loop always runs K_MAX iterations but masks inactive tokens
- Masked loads (other=0.0) + tl.where make inactive iterations zero-cost in
  terms of output contribution (loads still issue, but dot products are zeroed)

Shapes:
- R     = out_features // tile_size (output block-rows)
- C     = in_features  // tile_size (input block-columns)
- K_MAX = max active blocks per row (compile-time constexpr)
- B     = tile_size (default 16)
- k_active: [total_batch] int32 — per-token active block count (<=K_MAX)

Date: 2026-05-06
Branch: 006-looped-block-ell
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

# Triton import with fallback for CPU-only systems
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# Triton kernel: Block-ELL routed sparse GEMM forward
# =============================================================================


def _check_triton_available() -> None:
    """Raise error if Triton is not available."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is required for block_ell_routed_forward kernels. "
            "Install with: pip install triton"
        )


if TRITON_AVAILABLE:

    @triton.jit
    def block_ell_routed_forward_kernel(
        # Input tensor
        x_ptr,
        # Block-ELL values [R, K_MAX, B, B]
        values_ptr,
        # Column indices [R, K_MAX]
        col_indices_ptr,
        # Per-token active block count [total_batch] — int32
        k_active_ptr,
        # Output tensor
        out_ptr,
        # Dimensions
        batch_size,
        in_features,
        out_features,
        R,
        K_MAX: tl.constexpr,
        B: tl.constexpr,
        # Strides for x [batch, in_features]
        stride_x_batch,
        stride_x_feat,
        # Strides for values [R, K_MAX, B, B]
        stride_v_r,
        stride_v_k,
        stride_v_b1,
        stride_v_b2,
        # Strides for col_indices [R, K_MAX]
        stride_ci_r,
        stride_ci_k,
        # Strides for output [batch, out_features]
        stride_out_batch,
        stride_out_feat,
        # Block dimensions
        BLOCK_BATCH: tl.constexpr,
    ):
        """Block-ELL routed sparse matrix multiplication forward kernel.

        Computes: out = x @ W^T  (per-token variable K)

        Each program instance handles one output block-row for a batch of inputs.
        Tokens with k_active[token] < K_MAX skip the tail K iterations via masking.

        Grid: (R, cdiv(batch_size, BLOCK_BATCH))

        Algorithm:
        - Load k_active for all tokens in this batch block => k_active_vals [BLOCK_BATCH]
        - For k in range(K_MAX):
            - Compute per-token mask: k_mask = (k < k_active_vals)
            - Load x_block with combined mask (batch_mask & k_mask) => zero-fill inactive
            - Load weight tile (no mask needed — weight is shared across the batch block)
            - Accumulate: acc += dot(x_block, weight.T)
              (zeroed x_block rows automatically produce zero contribution)
        """
        # Program IDs
        pid_r = tl.program_id(0)  # Output block-row
        pid_batch = tl.program_id(1)  # Batch block

        # Batch indices for this program: [BLOCK_BATCH]
        batch_offs = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_offs < batch_size  # [BLOCK_BATCH] — valid token guard

        # Output block offsets within the block-row: [B]
        out_b_offs = tl.arange(0, B)
        in_b_offs = tl.arange(0, B)

        # Load per-token active K for this batch block: [BLOCK_BATCH] int32
        # Out-of-bound tokens get k_active=0 so they never contribute.
        k_active_vals = tl.load(k_active_ptr + batch_offs, mask=batch_mask, other=0)

        # Initialize accumulator: [BLOCK_BATCH, B] in fp32 for numerical stability
        acc = tl.zeros((BLOCK_BATCH, B), dtype=tl.float32)

        # Loop over K_MAX active blocks (compile-time constant — no break in Triton)
        # Per-token masking via k_mask makes tail iterations zero-cost contribution-wise.
        for k in tl.static_range(K_MAX):
            # Per-token mask: which tokens in this batch block are still active at step k
            # k_mask shape: [BLOCK_BATCH] bool
            k_mask = k < k_active_vals

            # Combined load mask: valid batch index AND token still active
            load_mask = batch_mask & k_mask  # [BLOCK_BATCH]

            # Load column index for this (r, k) pair — scalar, shared for all tokens
            col_idx = tl.load(col_indices_ptr + pid_r * stride_ci_r + k * stride_ci_k)

            # Load input block: x[batch_offs, col_idx*B : (col_idx+1)*B]
            # Shape: [BLOCK_BATCH, B] — inactive token rows filled with 0.0
            x_block_ptr = (
                x_ptr
                + batch_offs[:, None] * stride_x_batch          # [BLOCK_BATCH, 1]
                + (col_idx * B + in_b_offs[None, :]) * stride_x_feat  # [1, B]
            )
            x_block = tl.load(x_block_ptr, mask=load_mask[:, None], other=0.0)  # [BLOCK_BATCH, B]

            # Load weight tile: values[r, k, :, :]  shape [B_out, B_in] = [B, B]
            # Weight is shared across all tokens in the batch block — no per-token mask needed.
            weight_ptr = (
                values_ptr
                + pid_r * stride_v_r
                + k * stride_v_k
                + out_b_offs[:, None] * stride_v_b1   # [B, 1] for B_out
                + in_b_offs[None, :] * stride_v_b2    # [1, B] for B_in
            )
            weight = tl.load(weight_ptr)  # [B_out, B_in]

            # Cast weight to match input dtype (bf16 under autocast, fp32 otherwise)
            weight = weight.to(x_block.dtype)

            # Matrix multiply: x_block @ weight^T
            # x_block: [BLOCK_BATCH, B_in],  weight.T: [B_in, B_out]
            # Result:  [BLOCK_BATCH, B_out]
            # Rows for inactive tokens are already zero from the masked load,
            # so their contribution to acc is naturally zero.
            block_out = tl.dot(x_block, tl.trans(weight), allow_tf32=True)

            acc += block_out

        # Store output: out[batch_offs, pid_r*B : (pid_r+1)*B]
        out_ptr_block = (
            out_ptr
            + batch_offs[:, None] * stride_out_batch          # [BLOCK_BATCH, 1]
            + (pid_r * B + out_b_offs[None, :]) * stride_out_feat  # [1, B]
        )
        tl.store(out_ptr_block, acc, mask=batch_mask[:, None])


# =============================================================================
# Python wrapper: block_ell_routed_forward
# =============================================================================


def block_ell_routed_forward(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    k_active: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Block-ELL routed sparse matrix multiplication forward pass.

    Computes: y = x @ W^T + bias  where W is in Block-ELL format and each
    token uses only its first k_active[token] block columns.

    Args:
        x:           Input tensor [batch, in_features] or [batch, seq, in_features]
        values:      Block weight values [R, K_MAX, B, B]
        col_indices: Column indices for each block [R, K_MAX] (int32)
        k_active:    Per-token active block count [batch] or [batch, seq] (int32, <= K_MAX)
        bias:        Optional bias vector [out_features]

    Returns:
        Output tensor [batch, out_features] or [batch, seq, out_features]
    """
    _check_triton_available()

    # Validate standard block-ELL inputs
    _validate_routed_inputs(x, values, col_indices, k_active, bias)

    # Get dimensions
    R, K_MAX, B, _ = values.shape
    in_features = x.shape[-1]
    out_features = R * B

    # Handle 2D vs 3D input — flatten to 2D for kernel
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, in_features)
        reshape_output = True
    else:
        x_flat = x
        batch_size = x.shape[0]
        seq_len = 1
        reshape_output = False

    total_batch = x_flat.shape[0]

    # Flatten k_active to [total_batch] int32 on the same device
    k_active_flat = k_active.reshape(total_batch).to(dtype=torch.int32, device=x.device)

    # Clamp to [0, K_MAX] for safety (bad values would cause OOB reads in kernel)
    k_active_flat = k_active_flat.clamp(0, K_MAX)

    # Ensure contiguous layout for pointer arithmetic
    x_flat = x_flat.contiguous()
    values = values.contiguous()
    col_indices = col_indices.contiguous()
    k_active_flat = k_active_flat.contiguous()

    # Allocate output
    output = torch.empty(total_batch, out_features, dtype=x_flat.dtype, device=x_flat.device)

    # Choose BLOCK_BATCH based on total batch size
    if total_batch <= 16:
        BLOCK_BATCH = 16
    elif total_batch <= 32:
        BLOCK_BATCH = 32
    else:
        BLOCK_BATCH = 64

    # Detect sm_120 (Blackwell/5090) — needs reduced shared memory usage
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 12:
        BLOCK_BATCH = min(BLOCK_BATCH, 32)
        launch_kwargs = dict(num_stages=1, num_warps=4)
    else:
        launch_kwargs = {}

    # B must be a power of 2 for efficient tl.dot
    assert B in (8, 16, 32, 64), f"Block size B must be power of 2, got {B}"

    # Grid: (R, cdiv(total_batch, BLOCK_BATCH))
    grid = (R, triton.cdiv(total_batch, BLOCK_BATCH))

    # Launch kernel
    block_ell_routed_forward_kernel[grid](
        # Pointers
        x_flat,
        values,
        col_indices,
        k_active_flat,
        output,
        # Dimensions
        total_batch,
        in_features,
        out_features,
        R,
        K_MAX,
        B,
        # Strides for x [batch, in_features]
        x_flat.stride(0),
        x_flat.stride(1),
        # Strides for values [R, K_MAX, B, B]
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        # Strides for col_indices [R, K_MAX]
        col_indices.stride(0),
        col_indices.stride(1),
        # Strides for output [batch, out_features]
        output.stride(0),
        output.stride(1),
        # Block dimensions
        BLOCK_BATCH=BLOCK_BATCH,
        **launch_kwargs,
    )

    # Add bias if present
    if bias is not None:
        output = output + bias

    # Reshape output back to original batch/seq shape
    if reshape_output:
        output = output.view(batch_size, seq_len, out_features)

    return output


# =============================================================================
# Autograd wrapper
# =============================================================================


class _BlockELLRoutedAutograd(torch.autograd.Function):
    """Autograd Function for routed Block-ELL forward pass.

    Forward:  uses per-token k_active masking (routed, efficient)
    Backward: uses the existing (non-routed) backward kernels.
              - dW: all K_MAX tiles receive gradient (safe — router is a separate
                    selection mechanism, not a differentiable gate)
              - dx: existing block_ell_backward over full K_MAX
    """

    @staticmethod
    def forward(ctx, x, values, col_indices, k_active, bias):
        # Run the routed forward kernel
        output = block_ell_routed_forward(x, values, col_indices, k_active, bias)

        # Save for backward
        ctx.save_for_backward(x, values, col_indices, bias)
        ctx.k_active = k_active  # not a tensor we differentiate through
        R, K_MAX, B, _ = values.shape
        ctx.R = R
        ctx.K_MAX = K_MAX
        ctx.B = B
        ctx.in_features = x.shape[-1]
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from morph.kernels.triton.block_ell_backward import block_ell_backward

        x, values, col_indices, bias = ctx.saved_tensors

        needs_bias_grad = ctx.has_bias and (bias is not None and bias.requires_grad)
        dx, dvalues, dbias = block_ell_backward(
            dout=grad_output,
            x=x,
            values=values,
            col_indices=col_indices,
            needs_input_grad=x.requires_grad,
            needs_weight_grad=values.requires_grad,
            needs_bias_grad=needs_bias_grad,
        )

        # Gradients: x, values, col_indices (None — not differentiable), k_active (None), bias
        return dx, dvalues, None, None, dbias


def block_ell_routed_forward_autograd(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    k_active: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Autograd-compatible routed Block-ELL forward pass.

    Wraps block_ell_routed_forward in a torch.autograd.Function so that
    gradients flow correctly through x, values, and bias.

    The backward pass uses the existing (non-routed) backward kernels —
    all K_MAX tiles receive gradient updates for weights, which is correct
    because the router is a forward-time selection mechanism, not a
    differentiable gating function. dx also uses the full K_MAX backward.

    Args:
        x:           Input tensor [batch, in_features] or [batch, seq, in_features]
        values:      Block weight values [R, K_MAX, B, B]
        col_indices: Column indices [R, K_MAX] (int32)
        k_active:    Per-token active block count [batch] or [batch, seq] (int32)
        bias:        Optional bias vector [out_features]

    Returns:
        Output tensor with gradient support
    """
    return _BlockELLRoutedAutograd.apply(x, values, col_indices, k_active, bias)


# =============================================================================
# Reference implementation (pure PyTorch, for testing)
# =============================================================================


def block_ell_routed_forward_reference(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    k_active: Tensor,
    bias: Optional[Tensor] = None,
    tile_size: int = 16,
) -> Tensor:
    """Reference (pure PyTorch) implementation for correctness testing.

    Numerically equivalent to block_ell_routed_forward_kernel but runs on CPU
    or GPU via standard PyTorch ops. Slow — use for testing only.

    Args:
        x:           Input tensor [batch, in_features] or [batch, seq, in_features]
        values:      Block weight values [R, K_MAX, B, B]
        col_indices: Column indices [R, K_MAX] (int32)
        k_active:    Per-token active block count [batch] or [batch, seq] (int32, <=K_MAX)
        bias:        Optional bias [out_features]
        tile_size:   Block size B (must match values.shape[-1])

    Returns:
        Output tensor [batch, out_features] or [batch, seq, out_features]
    """
    R, K_MAX, B_out, B_in = values.shape
    assert B_out == B_in == tile_size, (
        f"Tile size mismatch: values has B={B_out}x{B_in}, tile_size={tile_size}"
    )

    # Handle 2D vs 3D input
    if x.dim() == 2:
        x = x.unsqueeze(1)
        k_active = k_active.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    # x is now [batch, seq, in_features]
    batch_size, seq_len, in_features = x.shape
    C = in_features // tile_size
    out_features = R * tile_size

    # k_active: [batch, seq] — flatten to [batch*seq]
    k_active_flat = k_active.reshape(batch_size * seq_len).long().clamp(0, K_MAX)

    # Reshape input to block view: [batch, seq, C, B]
    x_blocks = x.reshape(batch_size, seq_len, C, tile_size)

    # Initialize output: [batch, seq, R, B]
    output = torch.zeros(batch_size, seq_len, R, tile_size, dtype=x.dtype, device=x.device)

    for r in range(R):
        cols = col_indices[r]       # [K_MAX] int32
        weights = values[r]         # [K_MAX, B_out, B_in]

        for k in range(K_MAX):
            c = cols[k].long().item()
            w = weights[k]  # [B_out, B_in]

            # Input slice at column c: [batch, seq, B_in]
            x_col = x_blocks[:, :, c, :]  # [batch, seq, B_in]

            # Compute contribution: x_col @ w.T  =>  [batch, seq, B_out]
            contrib = torch.einsum("bsi,oi->bso", x_col, w)

            # Per-token mask: active only when k < k_active[token]
            # k_active_flat: [batch*seq], reshape to [batch, seq, 1]
            active_mask = (k < k_active_flat).reshape(batch_size, seq_len, 1).to(x.dtype)

            output[:, :, r, :] += active_mask * contrib

    # Reshape output: [batch, seq, R, B] -> [batch, seq, out_features]
    output = output.reshape(batch_size, seq_len, out_features)

    if bias is not None:
        output = output + bias

    if squeeze_output:
        output = output.squeeze(1)

    return output


# =============================================================================
# Validation
# =============================================================================


def _validate_routed_inputs(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    k_active: Tensor,
    bias: Optional[Tensor],
) -> None:
    """Validate input tensors for routed forward pass."""
    if values.dim() != 4:
        raise ValueError(f"values must be 4D [R, K_MAX, B, B], got {values.dim()}D")

    R, K_MAX, B_out, B_in = values.shape
    if B_out != B_in:
        raise ValueError(f"Block dimensions must be square: got {B_out}x{B_in}")

    tile_size = B_out

    if col_indices.shape != (R, K_MAX):
        raise ValueError(
            f"col_indices shape {tuple(col_indices.shape)} must match [R={R}, K_MAX={K_MAX}]"
        )

    if x.dim() not in (2, 3):
        raise ValueError(f"Input x must be 2D or 3D, got {x.dim()}D")

    in_features = x.shape[-1]
    if in_features % tile_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by tile_size ({tile_size})"
        )

    C = in_features // tile_size
    if col_indices.numel() > 0:
        max_col = col_indices.max().item()
        if max_col >= C:
            raise ValueError(f"col_indices contains value {max_col} >= C ({C})")

    # k_active shape: must broadcast to the token count
    if x.dim() == 2:
        expected_k_active_numel = x.shape[0]
    else:
        expected_k_active_numel = x.shape[0] * x.shape[1]
    if k_active.numel() != expected_k_active_numel:
        raise ValueError(
            f"k_active has {k_active.numel()} elements, expected {expected_k_active_numel} "
            f"(batch{'*seq' if x.dim()==3 else ''})"
        )

    out_features = R * tile_size
    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != out_features:
            raise ValueError(
                f"bias shape {tuple(bias.shape)} must be [out_features={out_features}]"
            )

    if values.device != x.device:
        raise ValueError(f"values ({values.device}) and x ({x.device}) must be on same device")
    if col_indices.device != x.device:
        raise ValueError(
            f"col_indices ({col_indices.device}) and x ({x.device}) must be on same device"
        )
    if k_active.device != x.device:
        raise ValueError(
            f"k_active ({k_active.device}) and x ({x.device}) must be on same device"
        )
    if bias is not None and bias.device != x.device:
        raise ValueError(f"bias ({bias.device}) and x ({x.device}) must be on same device")
