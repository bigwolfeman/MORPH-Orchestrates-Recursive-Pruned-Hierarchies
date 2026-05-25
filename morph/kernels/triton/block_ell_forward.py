"""Triton kernels for Block-ELL sparse forward pass.

This module provides GPU-accelerated forward pass for block-sparse matrix
multiplication using the Block-ELL format. The kernels are optimized for:
- Tensor core operations on 16x16 blocks (WMMA)
- Coalesced memory access patterns
- Fixed K blocks per row for regular execution

Shapes:
- R = out_features // tile_size (output block-rows)
- C = in_features // tile_size (input block-columns)
- K = active blocks per row
- B = tile_size (default 16)

Date: 2025-12-26
Branch: 001-cms-block-sparse
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
# Triton kernel: Block-ELL sparse GEMM forward
# =============================================================================


def _check_triton_available() -> None:
    """Raise error if Triton is not available."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is required for block_ell_forward kernels. " "Install with: pip install triton"
        )


if TRITON_AVAILABLE:

    @triton.jit
    def block_ell_forward_kernel(
        # Input tensor
        x_ptr,
        # Block-ELL values [R, K, B, B]
        values_ptr,
        # Column indices [R, K]
        col_indices_ptr,
        # Output tensor
        out_ptr,
        # Dimensions
        batch_size,
        in_features,
        out_features,
        R,
        K,
        B: tl.constexpr,
        # Strides for x [batch, in_features]
        stride_x_batch,
        stride_x_feat,
        # Strides for values [R, K, B, B]
        stride_v_r,
        stride_v_k,
        stride_v_b1,
        stride_v_b2,
        # Strides for col_indices [R, K]
        stride_ci_r,
        stride_ci_k,
        # Strides for output [batch, out_features]
        stride_out_batch,
        stride_out_feat,
        # Block dimensions
        BLOCK_BATCH: tl.constexpr,
    ):
        """Block-ELL sparse matrix multiplication forward kernel.

        Computes: out = x @ W^T where W is stored in Block-ELL format

        Each program instance handles one output block-row for a batch of inputs.

        Grid: (R, cdiv(batch_size, BLOCK_BATCH))

        Algorithm:
        - For the assigned block-row r:
          - For each active column k in [0, K):
            - Load column index c = col_indices[r, k]
            - Load input block x[:, c*B:(c+1)*B]
            - Load weight tile values[r, k, :, :]
            - Compute contribution: x_block @ weight.T
            - Accumulate into output block
        """
        # Program IDs
        pid_r = tl.program_id(0)  # Output block-row
        pid_batch = tl.program_id(1)  # Batch block

        # Batch indices for this program
        batch_offs = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_offs < batch_size

        # Output block indices (within the block-row)
        out_b_offs = tl.arange(0, B)  # [0, 1, ..., B-1]

        # Initialize accumulator for this output block: [BLOCK_BATCH, B]
        acc = tl.zeros((BLOCK_BATCH, B), dtype=tl.float32)

        # Loop over K active blocks in this row
        for k in range(K):
            # Load column index for this (r, k) pair
            col_idx_ptr = col_indices_ptr + pid_r * stride_ci_r + k * stride_ci_k
            col_idx = tl.load(col_idx_ptr)  # Scalar

            # Load input block: x[batch_offs, col_idx*B : (col_idx+1)*B]
            # x_block shape: [BLOCK_BATCH, B]
            in_b_offs = tl.arange(0, B)
            x_block_ptr = (
                x_ptr
                + batch_offs[:, None] * stride_x_batch  # [BLOCK_BATCH, 1]
                + (col_idx * B + in_b_offs[None, :]) * stride_x_feat  # [1, B]
            )
            # Mask for valid batch indices
            x_mask = batch_mask[:, None]  # [BLOCK_BATCH, 1] broadcasts to [BLOCK_BATCH, B]
            x_block = tl.load(x_block_ptr, mask=x_mask, other=0.0)  # [BLOCK_BATCH, B]

            # Load weight tile: values[r, k, :, :]
            # Shape: [B, B] where first dim is output, second is input
            # We need to compute x_block @ weight.T = x_block @ weight^T
            # x_block: [BLOCK_BATCH, B_in], weight: [B_out, B_in]
            # Result: [BLOCK_BATCH, B_out]

            # Load weight in transposed order for efficient matmul
            # We load as [B_out, B_in] and compute x @ W^T
            weight_ptr = (
                values_ptr
                + pid_r * stride_v_r
                + k * stride_v_k
                + out_b_offs[:, None] * stride_v_b1  # [B, 1] for B_out
                + in_b_offs[None, :] * stride_v_b2  # [1, B] for B_in
            )
            weight = tl.load(weight_ptr)  # [B_out, B_in] = [B, B]

            # Cast weight to match input dtype (bf16 under autocast, fp32 otherwise)
            weight = weight.to(x_block.dtype)

            # Matrix multiply: x_block @ weight^T
            # x_block: [BLOCK_BATCH, B_in]
            # weight.T: [B_in, B_out]
            # Result: [BLOCK_BATCH, B_out]
            # In Triton: tl.dot(a, tl.trans(b)) computes a @ b^T
            # Note: TF32 enabled for tensor core acceleration (3-8x speedup)
            # TF32 has ~1e-3 error vs FP32's ~1e-7, acceptable for training
            block_out = tl.dot(x_block, tl.trans(weight), allow_tf32=True)  # [BLOCK_BATCH, B]

            # Accumulate
            acc += block_out

        # Store output: out[batch_offs, pid_r*B : (pid_r+1)*B]
        out_ptr_block = (
            out_ptr
            + batch_offs[:, None] * stride_out_batch  # [BLOCK_BATCH, 1]
            + (pid_r * B + out_b_offs[None, :]) * stride_out_feat  # [1, B]
        )
        out_mask = batch_mask[:, None]
        tl.store(out_ptr_block, acc, mask=out_mask)


def block_ell_forward(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Block-ELL sparse matrix multiplication forward pass.

    Computes: y = x @ W^T + bias where W is in Block-ELL format

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        values: Block weight values [R, K, B, B]
        col_indices: Column indices for each block [R, K] (int32)
        bias: Optional bias vector [out_features]

    Returns:
        Output tensor [batch, out_features] or [batch, seq, out_features]
    """
    _check_triton_available()

    # Validate inputs
    validate_forward_inputs(x, values, col_indices, bias)

    # Get dimensions
    R, K, B, _ = values.shape
    in_features = x.shape[-1]
    out_features = R * B

    # Handle 2D vs 3D input - flatten to 2D for kernel
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(batch_size * seq_len, in_features)
        reshape_output = True
    else:
        x_flat = x
        batch_size = x.shape[0]
        seq_len = 1
        reshape_output = False

    total_batch = x_flat.shape[0]

    # Ensure tensors are contiguous
    x_flat = x_flat.contiguous()
    values = values.contiguous()
    col_indices = col_indices.contiguous()

    # Allocate output
    output = torch.empty(total_batch, out_features, dtype=x_flat.dtype, device=x_flat.device)

    # Choose BLOCK_BATCH based on batch size
    # Using smaller blocks for small batches to maintain occupancy
    if total_batch <= 16:
        BLOCK_BATCH = 16
    elif total_batch <= 32:
        BLOCK_BATCH = 32
    else:
        BLOCK_BATCH = 64

    # Detect sm_120 (Blackwell/5090) — force num_stages=1 (no TMA pipeline)
    # Z3-verified: BLOCK_BATCH=64, num_warps=8 gives 100% occupancy on SM120.
    # Shared memory: 64*16*2 + 16*16*2 = 2560 bytes (well under 48KB default).
    # Occupancy: 6 blocks * 8 warps = 48 warps = 100%.
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 12:
        launch_kwargs = dict(num_stages=1, num_warps=8)
    else:
        launch_kwargs = {}

    # Ensure B is power of 2 for efficient tl.dot
    assert B in (8, 16, 32, 64), f"Block size B must be power of 2, got {B}"

    # Grid: (R, cdiv(batch_size, BLOCK_BATCH))
    grid = (R, triton.cdiv(total_batch, BLOCK_BATCH))

    # Launch kernel
    block_ell_forward_kernel[grid](
        # Pointers
        x_flat,
        values,
        col_indices,
        output,
        # Dimensions
        total_batch,
        in_features,
        out_features,
        R,
        K,
        B,
        # Strides for x [batch, in_features]
        x_flat.stride(0),
        x_flat.stride(1),
        # Strides for values [R, K, B, B]
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        # Strides for col_indices [R, K]
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


def block_ell_forward_autograd(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    bias: Optional[Tensor] = None,
    use_triton: bool = True,
    column_index: Optional[Tuple[Tensor, Tensor, int]] = None,
) -> Tensor:
    """Autograd-compatible wrapper for block_ell_forward.

    This function wraps the kernel in a torch.autograd.Function for
    gradient computation during training.

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        values: Block weight values [R, K, B, B]
        col_indices: Column indices for each block [R, K]
        bias: Optional bias vector [out_features]
        use_triton: Whether to use Triton kernels when available (default True)
        column_index: Optional precomputed column index (pairs, counts, max_pairs)
                     from build_column_index(). If provided, enables the optimized
                     column-parallel dx backward kernel (7.3x faster).

    Returns:
        Output tensor with gradient support
    """
    from morph.kernels.triton.block_ell_backward import block_ell_autograd

    R, K, B, _ = values.shape
    in_features = x.shape[-1]

    return block_ell_autograd(
        x=x,
        values=values,
        col_indices=col_indices,
        bias=bias,
        in_features=in_features,
        R=R,
        K=K,
        B=B,
        use_triton=use_triton,
        column_index=column_index,
    )


# =============================================================================
# Reference implementation for testing
# =============================================================================


def block_ell_forward_reference(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    bias: Optional[Tensor] = None,
    tile_size: int = 16,
) -> Tensor:
    """Reference (slow) implementation for correctness testing.

    This PyTorch implementation is numerically correct but not optimized.
    Use for testing and debugging only.

    Computes: y = x @ W^T + bias where W is in Block-ELL format

    Algorithm:
        For each output block-row r:
            output[r] = sum over k of (input[:, col_indices[r,k]] @ values[r,k])

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        values: Block weight values [R, K, B, B] where B is tile_size
        col_indices: Column indices [R, K] indicating which input block-column
                     each weight block is connected to
        bias: Optional bias [out_features]
        tile_size: Block size B (must match values.shape[-1])

    Returns:
        Output tensor [batch, out_features] or [batch, seq, out_features]
    """
    # Get dimensions
    R, K, B_out, B_in = values.shape
    assert B_out == B_in == tile_size, f"Tile size mismatch: {B_out} vs {B_in} vs {tile_size}"

    # Handle 2D vs 3D input
    if x.dim() == 2:
        # [batch, in_features] -> [batch, 1, in_features]
        x = x.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    # x is now [batch, seq, in_features]
    batch_size, seq_len, in_features = x.shape
    C = in_features // tile_size
    out_features = R * tile_size

    # Reshape input to block view: [batch, seq, C, B]
    x_blocks = x.view(batch_size, seq_len, C, tile_size)

    # Initialize output: [batch, seq, R, B]
    output = torch.zeros(batch_size, seq_len, R, tile_size, dtype=x.dtype, device=x.device)

    # Block-sparse matmul: for each output row, gather K input blocks,
    # multiply with weight tiles, and sum
    for r in range(R):
        # Get column indices for this row: [K]
        cols = col_indices[r]  # [K]

        # Gather input blocks at these columns: [batch, seq, K, B]
        input_gathered = x_blocks[:, :, cols.long(), :]  # [batch, seq, K, B]

        # Weight tiles for this row: [K, B, B]
        weights = values[r]  # [K, B_out, B_in]

        # Compute: input [b, s, k, B_in] @ weight [k, B_out, B_in]^T
        # Using einsum: 'bski,koi->bsko' then sum over k
        # weights layout: [K, B_out, B_in] where B_out=B_in=B
        block_outputs = torch.einsum("bski,koi->bsko", input_gathered, weights)

        # Sum over K blocks: [batch, seq, B]
        row_output = block_outputs.sum(dim=2)

        output[:, :, r, :] = row_output

    # Reshape output: [batch, seq, R, B] -> [batch, seq, out_features]
    output = output.view(batch_size, seq_len, out_features)

    # Add bias if present
    if bias is not None:
        output = output + bias

    # Squeeze back if input was 2D
    if squeeze_output:
        output = output.squeeze(1)  # [batch, out_features]

    return output


# =============================================================================
# Utility functions
# =============================================================================


def compute_output_shape(
    input_shape: Tuple[int, ...],
    out_features: int,
) -> Tuple[int, ...]:
    """Compute output shape for given input shape.

    Args:
        input_shape: Input tensor shape (batch,) or (batch, seq, feat) etc.
        out_features: Output feature dimension

    Returns:
        Output shape with last dim replaced by out_features
    """
    return (*input_shape[:-1], out_features)


def validate_forward_inputs(
    x: Tensor,
    values: Tensor,
    col_indices: Tensor,
    bias: Optional[Tensor],
) -> None:
    """Validate input tensors for forward pass.

    Args:
        x: Input tensor [batch, in_features] or [batch, seq, in_features]
        values: Block values [R, K, B, B]
        col_indices: Column indices [R, K]
        bias: Optional bias [out_features]

    Raises:
        ValueError: If tensor shapes are invalid
    """
    # Check values shape
    if values.dim() != 4:
        raise ValueError(f"values must be 4D [R, K, B, B], got {values.dim()}D")

    R, K, B_out, B_in = values.shape
    if B_out != B_in:
        raise ValueError(f"Block dimensions must match: {B_out} vs {B_in}")

    tile_size = B_out

    # Check col_indices shape
    if col_indices.dim() != 2:
        raise ValueError(f"col_indices must be 2D [R, K], got {col_indices.dim()}D")
    if col_indices.shape[0] != R or col_indices.shape[1] != K:
        raise ValueError(
            f"col_indices shape {tuple(col_indices.shape)} doesn't match values shape R={R}, K={K}"
        )

    # Check input dimensions
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

    # Check bias shape
    out_features = R * tile_size
    if bias is not None:
        if bias.dim() != 1:
            raise ValueError(f"bias must be 1D, got {bias.dim()}D")
        if bias.shape[0] != out_features:
            raise ValueError(
                f"bias shape {bias.shape[0]} doesn't match out_features {out_features}"
            )

    # Check device compatibility
    if values.device != col_indices.device:
        raise ValueError(
            f"values ({values.device}) and col_indices ({col_indices.device}) must be on same device"
        )
    if values.device != x.device:
        raise ValueError(f"values ({values.device}) and x ({x.device}) must be on same device")
    if bias is not None and bias.device != values.device:
        raise ValueError(
            f"bias ({bias.device}) and values ({values.device}) must be on same device"
        )
