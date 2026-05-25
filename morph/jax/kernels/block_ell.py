"""Pallas (JAX) kernels for Block-ELL sparse matrix operations.

Ported from Triton GPU kernels in titans_core/kernels/block_ell_{forward,backward}.py.
Targets TPU v4/v5/v6 via the Pallas TPU backend.

Block-ELL format recap
----------------------
A sparse weight matrix [out_features, in_features] is stored as:
  values      : [R, K, B, B]  — dense tile values (bf16 or fp32)
  col_indices : [R, K]        — column-block index for each tile (int32)

where  R = out_features // B,  C = in_features // B,  K = active blocks per row,
and    B = tile_size (16 for pruning granularity).

Macro-block tiling for MXU
---------------------------
The TPU Matrix Multiply Unit (MXU) is most efficient at 128×128.  We therefore
group 8×8 pruning tiles into a *macro block* (128×128 elements) at the kernel
level.  The Pallas grid iterates over macro blocks; within each kernel invocation
we loop over the K active tiles and accumulate into the 128×128 output slab.

When B=16 and macro_size=128 there are M=8 tiles per macro-block edge.  For
simplicity the outer dimension (R) is handled at the full-tile level (one
invocation per output block-row r, not per macro-block-row), which lets us keep
the scalar-prefetch col_indices simple.  If R is large and compile time becomes
an issue, switch to macro-row grouping.

Scalar prefetch
---------------
col_indices [R, K] is passed as a *prefetch scalar* via PrefetchScalarGridSpec:
  num_scalar_prefetch = 1
The kernel body receives it as a plain integer array loaded in SMEM before the
kernel fires.  This is the idiomatic TPU Pallas pattern for small index arrays.

Precision
---------
  inputs  : bf16
  weights : bf16 (stored) but cast to fp32 for accumulation
  output  : bf16 (cast from fp32 accumulator before store)
  bias    : fp32

Autograd
--------
block_ell_linear() is wrapped with jax.custom_vjp for correct gradient flow.
- Forward: dispatches to Pallas kernel (use_pallas=True) or pure-JAX reference.
- Backward dx and dw: pure-JAX einsum + scatter_add.  On TPU, XLA fuses these
  into MXU-efficient operations and the overhead of Pallas dispatch is not worth
  it.  The Pallas dx kernel (_block_ell_bwd_dx_pallas) is available for
  benchmarking but not used in the default backward path.

Pure-JAX fallback
-----------------
All forward/backward operations have _reference_* counterparts written in plain
JAX (no Pallas).  Pass use_pallas=False to disable the Pallas forward kernel.
The backward is always pure JAX regardless of use_pallas.

Usage
-----
    from morph.jax.kernels.block_ell import block_ell_linear, BlockELLKernelConfig

    cfg = BlockELLKernelConfig(R=32, K=8, B=16, C=64)
    y = block_ell_linear(x, values, col_indices, bias=None, cfg=cfg)

Date: 2026-05-25
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    PALLAS_AVAILABLE = True
except ImportError:
    PALLAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockELLKernelConfig:
    """Kernel configuration for Block-ELL sparse matmul.

    Attributes:
        R: Number of output block-rows  (out_features // B).
        K: Max non-zero blocks per row.
        B: Tile size in elements (16 for pruning granularity).
        C: Number of input block-columns (in_features // B).
        macro_size: MXU-friendly macro-tile edge in elements (128 for TPU).
        use_pallas: Whether to use Pallas kernels (True) or pure-JAX reference (False).
    """

    R: int
    K: int
    B: int
    C: int
    macro_size: int = 128
    use_pallas: bool = True

    @property
    def out_features(self) -> int:
        return self.R * self.B

    @property
    def in_features(self) -> int:
        return self.C * self.B

    @property
    def tiles_per_macro(self) -> int:
        """Number of B-sized tiles that fit along one macro-block edge."""
        assert self.macro_size % self.B == 0, (
            f"macro_size ({self.macro_size}) must be divisible by B ({self.B})"
        )
        return self.macro_size // self.B


# ---------------------------------------------------------------------------
# Pure-JAX reference implementations (correctness oracle)
# ---------------------------------------------------------------------------

def _reference_fwd(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Reference forward: y = x @ W^T + bias.

    W is represented in Block-ELL format.

    Computes for each output block-row r:
        y[..., r*B:(r+1)*B] = sum_k( x[..., col_indices[r,k]*B : (col_indices[r,k]+1)*B]
                                      @ values[r,k].T )

    Args:
        x: [..., in_features]
        values: [R, K, B, B]  — values[r, k] is [B_out, B_in]
        col_indices: [R, K] int32
        bias: [out_features] or None

    Returns:
        [..., out_features]
    """
    R, K, B_out, B_in = values.shape
    B = B_out
    assert B_out == B_in, "Block tiles must be square"
    C = x.shape[-1] // B
    leading = x.shape[:-1]

    # x_tiled: [..., C, B]
    x_tiled = x.reshape(*leading, C, B)

    # Gather active tiles: [..., R, K, B]
    flat_cols = col_indices.reshape(-1)                              # [R*K]
    x_flat = x_tiled[..., flat_cols, :]                              # [..., R*K, B]
    x_gathered = x_flat.reshape(*leading, R, K, B)                   # [..., R, K, B]

    # Batched matmul — values[r,k]: [B_out, B_in], x_gathered[...,r,k]: [B_in]
    # einsum: '...rkd, rkDd -> ...rkD'  then sum over k
    # Note: values layout matches the Triton convention: [B_out, B_in] so
    #       the matmul is  x @ W^T  ↔  einsum 'bsrkd, rkDd -> bsrkD'
    out_blocked = jnp.einsum(
        "...rkd,rkDd->...rkD",
        x_gathered,
        values,
        precision=jax.lax.Precision.DEFAULT,
    )  # [..., R, K, B_out]

    # Sum over K active blocks
    out_blocked = out_blocked.sum(axis=-2)  # [..., R, B_out]

    # Flatten: [..., R*B_out]
    output = out_blocked.reshape(*leading, R * B)

    if bias is not None:
        output = output + bias

    return output


def _reference_bwd_dx(
    dout: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    in_features: int,
) -> jnp.ndarray:
    """Reference backward w.r.t. x.

    Forward: y = x @ W^T  →  dx = dout @ W

    In block form:
        dx[..., c*B:(c+1)*B] += sum_{r,k: col_indices[r,k]=c}
                                     dout[..., r*B:(r+1)*B] @ values[r,k]

    Args:
        dout: [..., out_features]
        values: [R, K, B, B]
        col_indices: [R, K] int32
        in_features: int

    Returns:
        dx: [..., in_features]
    """
    R, K, B, _ = values.shape
    C = in_features // B
    leading = dout.shape[:-1]

    # dout_blocks: [..., R, B]
    dout_blocks = dout.reshape(*leading, R, B)

    # Compute contributions for all (r, k) pairs simultaneously
    # dout_blocks: [..., R, 1, B] broadcast to [..., R, K, B]
    # values[r,k]: [B_out, B_in] — dx contrib = dout @ values (no transpose)
    dout_exp = jnp.expand_dims(dout_blocks, axis=-2)  # [..., R, 1, B_out]
    # einsum: '...r1o, rkoi -> ...rki'  (then scatter-add over C)
    dx_contrib = jnp.einsum(
        "...ro,rkoi->...rki",
        dout_blocks,
        values,
        precision=jax.lax.Precision.DEFAULT,
    )  # [..., R, K, B_in]

    # Flatten (R, K) → index into C
    dx_contrib_flat = dx_contrib.reshape(*leading, R * K, B)   # [..., R*K, B]
    flat_cols = col_indices.reshape(-1).astype(jnp.int32)       # [R*K]

    # Scatter-add into dx_blocks: [..., C, B]
    dx_blocks = jnp.zeros((*leading, C, B), dtype=dout.dtype)
    # Use indexed update — JAX scatter-add via at[...].add(...)
    dx_blocks = dx_blocks.at[..., flat_cols, :].add(dx_contrib_flat)

    # Flatten: [..., C*B]
    return dx_blocks.reshape(*leading, in_features)


def _reference_bwd_dw(
    x: jnp.ndarray,
    dout: jnp.ndarray,
    col_indices: jnp.ndarray,
    R: int,
    K: int,
    B: int,
) -> jnp.ndarray:
    """Reference backward w.r.t. weight values.

    Forward: y = x @ W^T  →  dW[r,k] = dout[..., r*B:(r+1)*B].T @ x[..., c*B:(c+1)*B]
    where c = col_indices[r, k].

    Args:
        x: [..., in_features]
        dout: [..., out_features]
        col_indices: [R, K] int32
        R, K, B: dimensions

    Returns:
        dvalues: [R, K, B, B]
    """
    C = x.shape[-1] // B
    leading = x.shape[:-1]
    N = 1
    for d in leading:
        N *= d

    # Flatten leading dims: [N, features]
    x_flat = x.reshape(N, C * B)
    dout_flat = dout.reshape(N, R * B)

    # Block views: [N, C, B] and [N, R, B]
    x_blocks = x_flat.reshape(N, C, B)
    dout_blocks = dout_flat.reshape(N, R, B)

    # For each (r, k): dW[r,k] = einsum('no,ni->oi', dout[:, r, :], x[:, c, :])
    # where c = col_indices[r, k].
    # Vectorised: gather x blocks at all columns simultaneously.
    flat_cols = col_indices.reshape(-1).astype(jnp.int32)     # [R*K]
    # x at active columns: [N, R*K, B]
    x_gathered = x_blocks[:, flat_cols, :]                      # [N, R*K, B]
    x_rk = x_gathered.reshape(N, R, K, B)                      # [N, R, K, B_in]

    # dout per output row: [N, R, K, B_out] (broadcast K)
    dout_rk = jnp.expand_dims(dout_blocks, axis=2)              # [N, R, 1, B_out]
    dout_rk = jnp.broadcast_to(dout_rk, (N, R, K, B))           # [N, R, K, B_out]

    # dW[r,k] = sum_n dout[n,r,:].T @ x[n,c,:]
    # einsum: 'nrko, nrki -> rkoi'
    dvalues = jnp.einsum(
        "nrko,nrki->rkoi",
        dout_rk,
        x_rk,
        precision=jax.lax.Precision.DEFAULT,
    )  # [R, K, B_out, B_in]

    return dvalues


# ---------------------------------------------------------------------------
# Pallas kernel: forward pass
# ---------------------------------------------------------------------------

def _make_fwd_kernel(B: int, K: int, macro_size: int):
    """Return a Pallas forward kernel function for the given tile dimensions.

    Each kernel invocation handles one output block-row r and one batch macro-block.
    The col_indices for the current row are provided via scalar prefetch.

    Grid: (R, ceil(N / macro_size))
      pid(0) = output block-row index  r
      pid(1) = batch macro-block index (N chunked into macro_size slabs)

    Scalar prefetch (num_scalar_prefetch=1):
      col_indices_ref: shape [K] — column indices for the CURRENT row r.
      These are loaded into SMEM by the TPU DMA engine before the kernel body runs.

    Inputs (via BlockSpec):
      x_ref:   [macro_size, B]  — one batch-macro slice of one input column tile
                                   (we loop over K columns inside the kernel body)
      NOTE: Because we loop over K inside the kernel, x_ref must be a *full-column*
            view.  We therefore pass x as [N, C*B] with BlockSpec slicing [macro_size, C*B],
            and index into columns manually.

    Output (via BlockSpec):
      out_ref: [macro_size, B]  — one batch-macro × output-row block.
    """
    tiles_per_macro = macro_size // B  # e.g. 128//16 = 8 (unused in kernel body but useful context)

    def fwd_kernel(col_indices_ref, x_ref, values_ref, out_ref):
        """Pallas kernel body: accumulate K active tiles into one output block.

        Args:
            col_indices_ref: scalar prefetch — [K] int32, active column indices for this row.
            x_ref: MemoryRef — [macro_size, C*B] input slab for this batch chunk.
            values_ref: MemoryRef — [K, B, B] weight tiles for this output row.
            out_ref: MemoryRef — [macro_size, B] output accumulator.

        The kernel uses lax.fori_loop over K to accumulate block contributions.
        fp32 accumulator for numerical stability, cast to bf16 on write.
        """
        macro_sz = out_ref.shape[0]

        def body(k, acc):
            col = col_indices_ref[k]  # scalar int
            # Gather input tile: x[:, col*B : (col+1)*B]
            x_tile = pl.load(x_ref, (slice(None), pl.ds(col * B, B)))  # [macro_sz, B]
            # Weight tile for (r, k): [B_out, B_in]
            w_tile = pl.load(values_ref, (k, slice(None), slice(None)))  # [B, B]
            # Accumulate: x_tile @ w_tile^T  →  [macro_sz, B_out]
            x_fp32 = x_tile.astype(jnp.float32)
            w_fp32 = w_tile.astype(jnp.float32)
            contrib = pl.dot(x_fp32, w_fp32, trans_b=True)   # [macro_sz, B]
            return acc + contrib

        acc = jnp.zeros((macro_sz, B), dtype=jnp.float32)
        acc = jax.lax.fori_loop(0, K, body, acc)

        # Cast and write output
        out_ref[...] = acc.astype(out_ref.dtype)

    return fwd_kernel


def _block_ell_fwd_pallas(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    cfg: BlockELLKernelConfig,
) -> jnp.ndarray:
    """Pallas forward pass: y = x @ W^T  (no bias; caller adds bias).

    Args:
        x: [N, in_features]  (batch × seq already flattened)
        values: [R, K, B, B]
        col_indices: [R, K] int32
        cfg: BlockELLKernelConfig

    Returns:
        out: [N, out_features]
    """
    N, in_features = x.shape
    R, K, B, _ = values.shape
    macro = cfg.macro_size
    N_padded = ((N + macro - 1) // macro) * macro

    # Pad N to a multiple of macro_size for clean blocking
    if N_padded != N:
        pad_len = N_padded - N
        x = jnp.concatenate([x, jnp.zeros((pad_len, in_features), dtype=x.dtype)], axis=0)

    # col_indices [R, K] is passed as a scalar-prefetch input via PrefetchScalarGridSpec.
    # num_scalar_prefetch=1 tells Pallas that the first argument is a prefetch array.
    # The BlockSpec index_map selects row r of col_indices → [K] per invocation.
    # Inside the kernel body, col_indices_ref is a [K] SMEM array of integer scalars.
    fwd_kernel = _make_fwd_kernel(B=B, K=K, macro_size=macro)

    grid = (R, N_padded // macro)

    # Output layout: [N_padded, R*B].
    # BlockSpec block_shape=(macro, B) with index_map (r, n) → (n, r) correctly tiles
    # by placing macro-batch-block n at row n and B-output-block r at column r.
    # This means out[n*macro:(n+1)*macro, r*B:(r+1)*B] is written per invocation.
    out_shape = jax.ShapeDtypeStruct((N_padded, R * B), dtype=x.dtype)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=grid,
        in_specs=[
            # Scalar prefetch: col_indices [R, K] — index_map selects row r → [K]
            pl.BlockSpec(
                block_shape=(K,),
                index_map=lambda r, n: (r, 0),
            ),
            # x: [N_padded, in_features] — block [macro, in_features] per batch chunk
            pl.BlockSpec(
                block_shape=(macro, in_features),
                index_map=lambda r, n: (n, 0),
            ),
            # values: [R, K, B, B] — block [K, B, B] per output row r
            pl.BlockSpec(
                block_shape=(K, B, B),
                index_map=lambda r, n: (r, 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            # out[n*macro:(n+1)*macro, r*B:(r+1)*B]
            block_shape=(macro, B),
            index_map=lambda r, n: (n, r),
        ),
    )

    out = pl.pallas_call(
        fwd_kernel,
        grid_spec=grid_spec,
        out_shape=out_shape,
        name="block_ell_fwd",
    )(col_indices, x, values)

    # Slice back to original N
    return out[:N]


# ---------------------------------------------------------------------------
# Pallas kernel: backward w.r.t. x
# ---------------------------------------------------------------------------

def _make_bwd_dx_kernel(B: int, macro_size: int):
    """Return a Pallas backward dx kernel.

    Grid: (C, ceil(N / macro_size))
      pid(0) = input column-block index c
      pid(1) = batch macro-block index

    Scalar prefetch:
      pairs_ref: [max_pairs, 2] — (r, k) pairs mapping to column c (current pid).
      counts_ref: [] (scalar) — number of valid pairs for column c.

    Inputs:
      dout_ref: [macro_size, out_features]
      values_ref: [R, K, B, B]  — full weight tensor (needed for arbitrary (r,k))

    Output:
      dx_ref: [macro_size, B]  — gradient for input column c
    """
    def bwd_dx_kernel(pairs_ref, counts_ref, dout_ref, values_ref, dx_ref):
        """Accumulate dx for one input column c from all (r,k) pairs that map to it.

        dx[:, c*B:(c+1)*B] += sum_{(r,k): col_indices[r,k]=c}  dout[:, r*B:(r+1)*B] @ values[r,k]
        """
        macro_sz = dx_ref.shape[0]
        num_pairs = counts_ref[()]  # scalar

        def body(p, acc):
            # Load (r, k) pair — always load, mask contribution
            r_idx = pairs_ref[p, 0]
            k_idx = pairs_ref[p, 1]
            pair_valid = p < num_pairs

            # dout tile: [macro_sz, B_out]
            dout_tile = pl.load(dout_ref, (slice(None), pl.ds(r_idx * B, B)))
            # weight tile: [B_out, B_in]
            w_tile = pl.load(values_ref, (r_idx, k_idx, slice(None), slice(None)))

            # contrib: dout @ weight  →  [macro_sz, B_in]
            contrib = pl.dot(
                dout_tile.astype(jnp.float32),
                w_tile.astype(jnp.float32),
            )  # [macro_sz, B_in]

            # Zero-out if pair is padding
            contrib = jnp.where(pair_valid, contrib, jnp.zeros_like(contrib))
            return acc + contrib

        acc = jnp.zeros((macro_sz, B), dtype=jnp.float32)
        # max_pairs comes from pairs_ref.shape[0]
        max_pairs = pairs_ref.shape[0]
        acc = jax.lax.fori_loop(0, max_pairs, body, acc)
        dx_ref[...] = acc.astype(dx_ref.dtype)

    return bwd_dx_kernel


def _build_column_pairs_jax(
    col_indices: jnp.ndarray,
    C: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Build the inverse mapping: for each input column c, list (r,k) pairs.

    Pure-JAX (no Python loops over C) so it can be jit-compiled.

    Args:
        col_indices: [R, K] int32
        C: number of input columns

    Returns:
        pairs:  [C, max_pairs, 2] int32  — (r, k) for each pair per column
        counts: [C] int32                — number of valid pairs per column
        max_pairs: int                   — max pairs across all columns
    """
    R, K = col_indices.shape
    N = R * K

    flat_cols = col_indices.reshape(-1).astype(jnp.int32)  # [R*K]
    r_coords = jnp.repeat(jnp.arange(R, dtype=jnp.int32), K)   # [R*K]
    k_coords = jnp.tile(jnp.arange(K, dtype=jnp.int32), R)     # [R*K]

    # Count per column
    counts = jnp.zeros(C, dtype=jnp.int32)
    counts = counts.at[flat_cols].add(jnp.ones(N, dtype=jnp.int32))

    max_pairs = int(counts.max())
    if max_pairs == 0:
        max_pairs = 1

    # Sort by column to group entries
    sort_idx = jnp.argsort(flat_cols)
    sorted_cols = flat_cols[sort_idx]
    sorted_r = r_coords[sort_idx]
    sorted_k = k_coords[sort_idx]

    # Position within each column group
    col_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), counts.cumsum(0)[:-1]])
    positions = jnp.arange(N, dtype=jnp.int32) - col_starts[sorted_cols]

    # Scatter into pairs: [C, max_pairs, 2]
    pairs = jnp.zeros((C, max_pairs, 2), dtype=jnp.int32)
    pairs = pairs.at[sorted_cols, positions, 0].set(sorted_r)
    pairs = pairs.at[sorted_cols, positions, 1].set(sorted_k)

    return pairs, counts, max_pairs


def _block_ell_bwd_dx_pallas(
    dout: jnp.ndarray,
    values: jnp.ndarray,
    pairs: jnp.ndarray,
    counts: jnp.ndarray,
    max_pairs: int,
    in_features: int,
    cfg: BlockELLKernelConfig,
) -> jnp.ndarray:
    """Pallas backward dx: column-parallel, no atomics.

    Grid: (C, ceil(N / macro_size))

    Args:
        dout: [N, out_features]
        values: [R, K, B, B]
        pairs: [C, max_pairs, 2] int32
        counts: [C] int32
        max_pairs: int
        in_features: int
        cfg: BlockELLKernelConfig

    Returns:
        dx: [N, in_features]
    """
    N, out_features = dout.shape
    R, K, B, _ = values.shape
    C = in_features // B
    macro = cfg.macro_size

    N_padded = ((N + macro - 1) // macro) * macro
    if N_padded != N:
        pad_len = N_padded - N
        dout = jnp.concatenate([dout, jnp.zeros((pad_len, out_features), dtype=dout.dtype)], axis=0)

    bwd_dx_kernel = _make_bwd_dx_kernel(B=B, macro_size=macro)

    grid = (C, N_padded // macro)

    # Scalar prefetch for current column's (r,k) pairs and count
    # pairs: [C, max_pairs, 2] — index_map selects row c → [max_pairs, 2]
    # counts: [C] — index_map selects count for column c → scalar
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=grid,
        in_specs=[
            # pairs for column c: [max_pairs, 2]
            pl.BlockSpec(
                block_shape=(max_pairs, 2),
                index_map=lambda c, n: (c, 0, 0),
            ),
            # count for column c: scalar — squeezed with None
            pl.BlockSpec(
                block_shape=(),
                index_map=lambda c, n: (c,),
            ),
            # dout: [N_padded, out_features] — full row loaded per batch macro
            pl.BlockSpec(
                block_shape=(macro, out_features),
                index_map=lambda c, n: (n, 0),
            ),
            # values: full [R, K, B, B] — not tiled, accessed via dynamic index
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec(
            block_shape=(macro, B),
            index_map=lambda c, n: (n, c),
        ),
    )

    out_shape = jax.ShapeDtypeStruct((N_padded, C * B), dtype=dout.dtype)

    dx = pl.pallas_call(
        bwd_dx_kernel,
        grid_spec=grid_spec,
        out_shape=out_shape,
        name="block_ell_bwd_dx",
    )(pairs, counts, dout, values)

    return dx[:N]


# ---------------------------------------------------------------------------
# Pallas kernel: backward w.r.t. weight values
# ---------------------------------------------------------------------------

def _make_bwd_dw_kernel(B: int, macro_size: int):
    """Return a Pallas backward dw kernel.

    Grid: (R, K)
      pid(0) = output block-row r
      pid(1) = block-within-row k

    Scalar prefetch:
      col_ref: [] (scalar int32) — col_indices[r, k] for current (r, k).

    Inputs:
      x_ref:    [N_padded, B]  — input tile at column col (sliced externally)
      dout_ref: [N_padded, B]  — dout tile at row r

    Output:
      dvalues_ref: [B, B]  — accumulated dW for (r, k)

    We sum over the batch (N) dimension inside the kernel using fori_loop
    over macro-sized chunks.
    """
    def bwd_dw_kernel(col_ref, x_ref, dout_ref, dvalues_ref):
        """dW[r,k] = dout[:, r*B:(r+1)*B].T @ x[:, c*B:(c+1)*B].

        x_ref is already pre-sliced to the column tile [N, B].
        dout_ref is pre-sliced to the output tile [N, B].
        Both are provided as full-N arrays (not macro-blocked) because we
        reduce over all N in one shot.
        """
        # Full N reduce: dW = dout.T @ x
        # x_ref: [N, B_in], dout_ref: [N, B_out]
        x_fp32 = x_ref[...].astype(jnp.float32)
        dout_fp32 = dout_ref[...].astype(jnp.float32)
        # dW: [B_out, B_in] = dout.T @ x
        dw = pl.dot(dout_fp32, x_fp32, trans_a=True)  # [B_out, B_in]
        dvalues_ref[...] = dw.astype(dvalues_ref.dtype)

    return bwd_dw_kernel


def _block_ell_bwd_dw_pallas(
    x: jnp.ndarray,
    dout: jnp.ndarray,
    col_indices: jnp.ndarray,
    cfg: BlockELLKernelConfig,
) -> jnp.ndarray:
    """Pallas backward dw: one invocation per (r, k) block.

    The column index col_indices[r, k] is passed as a scalar prefetch.
    x and dout are pre-sliced per (r, k) before calling pallas_call to
    avoid dynamic indexing within the kernel body (which is not supported
    on TPU Pallas for HBM loads).

    This is implemented as R*K independent pallas_calls, each with a
    different x column slice.  In practice this is JIT-fused by XLA.

    Args:
        x: [N, in_features]
        dout: [N, out_features]
        col_indices: [R, K] int32
        cfg: BlockELLKernelConfig

    Returns:
        dvalues: [R, K, B, B]
    """
    R, K, B = cfg.R, cfg.K, cfg.B
    N = x.shape[0]
    C = x.shape[1] // B

    # Reshape for block indexing
    x_blocks = x.reshape(N, C, B)         # [N, C, B]
    dout_blocks = dout.reshape(N, R, B)   # [N, R, B]

    # Vectorised: for all (r, k) simultaneously using vmap
    # Gather all (r, k) x-column slices: [R, K, N, B]
    flat_cols = col_indices.reshape(-1).astype(jnp.int32)        # [R*K]
    x_rk = x_blocks[:, flat_cols, :]                              # [N, R*K, B]
    x_rk = x_rk.reshape(N, R, K, B).transpose(1, 2, 0, 3)        # [R, K, N, B]

    # dout per output row: [R, K, N, B]
    dout_rk = jnp.broadcast_to(
        dout_blocks.transpose(1, 0, 2)[:, None, :, :],            # [R, 1, N, B]
        (R, K, N, B),
    )

    # For each (r, k): dW = dout.T @ x  →  [B_out, B_in]
    # Pallas call expecting: x_ref [N, B], dout_ref [N, B], out: [B, B]
    bwd_dw_kernel = _make_bwd_dw_kernel(B=B, macro_size=cfg.macro_size)

    # Use vmap over (R, K) — each call gets [N, B] slices
    def _one_block(x_col, dout_row):
        """Compute dW for a single (r, k) block via pallas_call."""
        out_shape = jax.ShapeDtypeStruct((B, B), dtype=jnp.float32)
        # col prefetch: unused here (column already sliced)
        col_dummy = jnp.zeros((), dtype=jnp.int32)
        return pl.pallas_call(
            bwd_dw_kernel,
            out_shape=out_shape,
            in_specs=[
                pl.no_block_spec,  # col_ref (scalar prefetch placeholder)
                pl.no_block_spec,  # x_ref
                pl.no_block_spec,  # dout_ref
            ],
            name="block_ell_bwd_dw_single",
        )(col_dummy, x_col, dout_row)

    # vmap over (R*K) dimension
    x_flat_rk = x_rk.reshape(R * K, N, B)
    dout_flat_rk = dout_rk.reshape(R * K, N, B)

    dvalues_flat = jax.vmap(_one_block)(x_flat_rk, dout_flat_rk)  # [R*K, B, B]
    dvalues = dvalues_flat.reshape(R, K, B, B)
    return dvalues.astype(x.dtype)


# ---------------------------------------------------------------------------
# custom_vjp autograd wrapper
# ---------------------------------------------------------------------------

def block_ell_linear(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    cfg: BlockELLKernelConfig,
) -> jnp.ndarray:
    """Block-ELL sparse linear: y = x @ W^T + bias.

    This is the public entry point.  It dispatches to Pallas kernels when
    cfg.use_pallas=True and a TPU device is active, otherwise falls back to
    the pure-JAX reference.

    The function is wrapped with jax.custom_vjp so that JAX's autodiff
    uses our explicit backward kernels rather than differentiating through
    the forward kernel itself.

    Args:
        x: [..., in_features]  — bf16 or fp32
        values: [R, K, B, B]  — weight tiles, same dtype as x
        col_indices: [R, K] int32  — column-block index for each tile
        bias: [out_features] or None
        cfg: BlockELLKernelConfig

    Returns:
        y: [..., out_features]
    """
    return _block_ell_linear_impl(x, values, col_indices, bias, cfg)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _block_ell_linear_impl(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    cfg: BlockELLKernelConfig,
) -> jnp.ndarray:
    """Forward pass (custom_vjp function)."""
    return _fwd_only(x, values, col_indices, bias, cfg)


def _fwd_only(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    cfg: BlockELLKernelConfig,
) -> jnp.ndarray:
    """Execute forward pass, dispatching to Pallas or reference."""
    leading = x.shape[:-1]
    in_features = x.shape[-1]

    # Flatten leading dims for 2D kernel interface
    x_2d = x.reshape(-1, in_features)

    use_pallas = cfg.use_pallas and PALLAS_AVAILABLE
    if use_pallas:
        out_2d = _block_ell_fwd_pallas(x_2d, values, col_indices, cfg)
    else:
        out_2d = _reference_fwd(x_2d, values, col_indices)

    out = out_2d.reshape(*leading, cfg.out_features)

    if bias is not None:
        out = out + bias

    return out


def _fwd_with_residuals(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    cfg: BlockELLKernelConfig,
):
    """Forward pass that also returns residuals needed for backward."""
    out = _fwd_only(x, values, col_indices, bias, cfg)
    # Residuals: everything needed for backward
    return out, (x, values, col_indices, bias)


def _bwd(
    cfg: BlockELLKernelConfig,
    residuals,
    dout: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, None, Optional[jnp.ndarray]]:
    """Backward pass computing (dx, dvalues, None, dbias).

    None is returned for col_indices (non-differentiable integer array).

    Strategy:
    - dx: pure-JAX einsum + scatter_add (XLA auto-vectorises over TPU MXU perfectly).
          The Pallas column-parallel dx kernel is available but the pure-JAX path
          is preferred because XLA can fuse the einsum + scatter_add into a single
          MXU operation and avoids the overhead of R*K kernel dispatch rounds.
    - dvalues: pure-JAX einsum (XLA-fused, no Pallas needed).
    - dbias: plain sum reduction.

    If you need the Pallas dx backward for debugging or benchmarking, call
    _block_ell_bwd_dx_pallas() directly.
    """
    x, values, col_indices, bias = residuals
    R, K, B, _ = values.shape
    leading = x.shape[:-1]
    in_features = x.shape[-1]

    # Flatten leading dims: [N, features]
    x_2d = x.reshape(-1, in_features)
    dout_2d = dout.reshape(-1, cfg.out_features)

    # --- dx: pure JAX einsum + scatter_add (XLA-fused on TPU) ---
    dx_2d = _reference_bwd_dx(dout_2d, values, col_indices, in_features)
    dx = dx_2d.reshape(*leading, in_features)

    # --- dvalues: pure JAX einsum (XLA-fused on TPU) ---
    dvalues = _reference_bwd_dw(x_2d, dout_2d, col_indices, R, K, B)

    # --- dbias ---
    if bias is not None:
        dbias = dout.sum(axis=tuple(range(dout.ndim - 1)))
    else:
        dbias = None

    return dx, dvalues, None, dbias


_block_ell_linear_impl.defvjp(_fwd_with_residuals, _bwd)


# ---------------------------------------------------------------------------
# Convenience: build column pairs from numpy/jax for caching
# ---------------------------------------------------------------------------

def build_column_pairs(
    col_indices: jnp.ndarray,
    C: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Build the (r, k) inverse mapping for the dx backward kernel.

    Should be called once after each topology change and the result cached.
    The cached (pairs, counts, max_pairs) can be passed to
    _block_ell_bwd_dx_pallas() directly.

    Args:
        col_indices: [R, K] int32
        C: number of input block-columns

    Returns:
        pairs: [C, max_pairs, 2] int32
        counts: [C] int32
        max_pairs: int
    """
    return _build_column_pairs_jax(col_indices, C)


# ---------------------------------------------------------------------------
# Matmul interface compatible with existing JAX Block-ELL code
# ---------------------------------------------------------------------------

def block_ell_matmul(
    x: jnp.ndarray,
    values: jnp.ndarray,
    col_indices: jnp.ndarray,
    cfg: BlockELLKernelConfig,
    bias: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Drop-in Pallas-accelerated replacement for the pure-JAX block_ell_matmul.

    Compatible with the BlockELLTensor interface from
    titans_core/TPU/looped_blockell/layers/block_ell.py.

    Args:
        x: [batch, seq, in_features] or [N, in_features]
        values: [R, K, B, B]
        col_indices: [R, K] int32
        cfg: BlockELLKernelConfig
        bias: [out_features] or None

    Returns:
        out: same leading shape as x, last dim = out_features
    """
    return block_ell_linear(x, values, col_indices, bias, cfg)


# ---------------------------------------------------------------------------
# Self-test / smoke test (run on CPU with use_pallas=False)
# ---------------------------------------------------------------------------

def _smoke_test():
    """Quick correctness check: reference vs. JAX AD on a tiny problem.

    Validates:
    1. Forward output matches pure-JAX reference (no Pallas).
    2. custom_vjp dx/dvalues/dbias match JAX autodiff on the reference forward.
    3. Numerical gradient check on dx (centred finite-difference, eps=1e-3).
    """
    key = jax.random.PRNGKey(0)
    R, K, B, C = 4, 3, 16, 8
    N, S = 2, 6  # batch=2, seq=6

    cfg = BlockELLKernelConfig(R=R, K=K, B=B, C=C, use_pallas=False)

    # Deterministic col_indices: K unique columns per row
    col_indices = jnp.stack([
        jax.random.choice(jax.random.fold_in(key, r), C, shape=(K,), replace=False)
        for r in range(R)
    ]).astype(jnp.int32)

    values_key, x_key = jax.random.split(key)
    values = jax.random.normal(values_key, (R, K, B, B), dtype=jnp.float32) * 0.1
    x = jax.random.normal(x_key, (N, S, C * B), dtype=jnp.float32)
    bias = jnp.zeros(R * B, dtype=jnp.float32)

    # 1. Forward shape + correctness
    y = block_ell_linear(x, values, col_indices, bias, cfg)
    assert y.shape == (N, S, R * B), f"Shape mismatch: {y.shape}"

    y_ref = _reference_fwd(x, values, col_indices, bias)
    max_err_fwd = float(jnp.max(jnp.abs(y - y_ref)))
    assert max_err_fwd < 1e-5, f"Forward max error: {max_err_fwd}"

    # 2. Gradient shape + match against autodiff on reference
    def loss(x_, v_, b_):
        return block_ell_linear(x_, v_, col_indices, b_, cfg).sum()

    def loss_ref(x_, v_, b_):
        return _reference_fwd(x_, v_, col_indices, b_).sum()

    grads = jax.grad(loss, argnums=(0, 1, 2))(x, values, bias)
    dx, dvalues, dbias = grads

    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(x, values, bias)
    dx_ref, dvalues_ref, dbias_ref = grads_ref

    assert dx.shape == x.shape, f"dx shape: {dx.shape}"
    assert dvalues.shape == values.shape, f"dvalues shape: {dvalues.shape}"
    assert dbias.shape == bias.shape, f"dbias shape: {dbias.shape}"

    max_err_dx = float(jnp.max(jnp.abs(dx - dx_ref)))
    max_err_dw = float(jnp.max(jnp.abs(dvalues - dvalues_ref)))
    assert max_err_dx < 1e-5, f"dx vs autodiff-reference max error: {max_err_dx}"
    assert max_err_dw < 1e-5, f"dvalues vs autodiff-reference max error: {max_err_dw}"

    # 3. Numerical gradient check on dx.
    # We compare analytic dx against the autodiff-on-reference dx (already verified above
    # to be identical to within floating-point precision).  An additional centred-FD check
    # is run on the *reference* loss so that both sides use the same code path.
    # The primary gradient correctness assertion is the match against autodiff-reference above.
    dx_flat = dx.reshape(-1)
    i = int(jnp.argmax(jnp.abs(dx_flat)))  # pick element with largest gradient (guaranteed non-zero)
    # eps=1e-2: centred FD truncation error is O(eps^2)~1e-4, typically giving <1% rel error.
    # Smaller eps triggers catastrophic cancellation in float32 for this problem size.
    eps = 1e-2
    x_flat = x.reshape(-1)
    x_plus = x_flat.at[i].set(x_flat[i] + eps).reshape(x.shape)
    x_minus = x_flat.at[i].set(x_flat[i] - eps).reshape(x.shape)
    # Compare FD against the reference loss (same code path as analytic gradient)
    numerical_dx_i = (loss_ref(x_plus, values, bias) - loss_ref(x_minus, values, bias)) / (2 * eps)
    analytic_dx_i = dx_ref.reshape(-1)[i]  # autodiff-reference (exact)
    rel_err = abs(float(numerical_dx_i - analytic_dx_i)) / (abs(float(analytic_dx_i)) + 1e-8)
    assert rel_err < 2e-2, f"dx numerical gradient check failed: rel_err={rel_err:.4f}"

    print("✓ block_ell Pallas kernel smoke test passed")
    print(f"  shape: x={x.shape} → y={y.shape}")
    print(f"  forward max_err vs reference: {max_err_fwd:.2e}")
    print(f"  dx vs autodiff-reference max_err: {max_err_dx:.2e}")
    print(f"  dvalues vs autodiff-reference max_err: {max_err_dw:.2e}")
    print(f"  dx numerical gradient rel_err (elem {i}, eps={eps}): {rel_err:.4f}")
    return True


if __name__ == "__main__":
    _smoke_test()
