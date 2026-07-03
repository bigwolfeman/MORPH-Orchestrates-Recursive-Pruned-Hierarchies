# Vendored from stanford-futuredata/stk (Apache-2.0) — see NOTICE for provenance.
# MODIFIED: int16->int32 casts on loaded block indices (upstream PR #17 pattern; Triton>=3.2 promotion overflow fix).
import functools
import os

import torch
import triton
import triton.language as tl
from dataclasses import dataclass

# Perf killswitch for the output-tiled SDD (occupancy fix). Set MORPH_SDD_SPLIT=0
# to fall back to the stock one-CTA-per-block kernel (for A/B measurement). The
# split path is bitwise-identical, so this only affects speed, never numerics.
_SDD_SPLIT_ENABLED = os.environ.get("MORPH_SDD_SPLIT", "1") != "0"

@dataclass
class TritonConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 32
    BLOCK_SIZE: int = 128
    NUM_STAGES: int = 4
    NUM_WARPS: int = 4

def _validate_matmul_dims(M: int, K: int, N: int):
    error_string = "incompatible dimensions: tensor has dim with length: {}, which must be divisible by {}"
    assert M % TritonConfig.BLOCK_M == 0, error_string.format(M, TritonConfig.BLOCK_M)
    assert K % TritonConfig.BLOCK_K == 0, error_string.format(K, TritonConfig.BLOCK_K)
    assert N % TritonConfig.BLOCK_N == 0, error_string.format(N, TritonConfig.BLOCK_N)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': TritonConfig.BLOCK_M,
            'BLOCK_N': TritonConfig.BLOCK_N,
            'BLOCK_K': TritonConfig.BLOCK_K,
            'BLOCK_SIZE': TritonConfig.BLOCK_SIZE
        }, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _sdd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_m = tl.load(row_indices + pid).to(tl.int32)
    pid_n = tl.load(column_indices + pid).to(tl.int32)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    #Store to sparse matrix
    acc = acc.to(C.dtype.element_ty)
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    cm = tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)
    C = C + pid * BLOCK_ELEMENTS + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': TritonConfig.BLOCK_M,
            'BLOCK_N': TritonConfig.BLOCK_N,
            'BLOCK_K': TritonConfig.BLOCK_K,
            'BLOCK_SIZE': TritonConfig.BLOCK_SIZE
        }, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dsd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)

    start_inx = tl.load(offsets + pid_m).to(tl.int32)
    end_inx = tl.load(offsets + pid_m + 1).to(tl.int32)

    # pointers to sparse matrix
    rm =  tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)

    A += (rm[:, None] * stride_am + rak[None, :] * stride_ak)

    # pointers to dense matrix
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    ak_sub_incr = BLOCK_K * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk
    bk_block_incr = BLOCK_SIZE * stride_bk

    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks

        if trans_A:
            ptr_A = A + tl.load(block_offsets_t + start_inx + block_inx).to(tl.int32) * BLOCK_ELEMENTS + sub_block_inx * ak_sub_incr
        else:
            ptr_A = A + (start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * ak_sub_incr

        ptr_B = B + tl.load(column_indices + start_inx + block_inx).to(tl.int32) * bk_block_incr + sub_block_inx * bk_sub_incr

        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b)

    acc = acc.to(C.dtype.element_ty)

    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({
            'BLOCK_M': TritonConfig.BLOCK_M,
            'BLOCK_N': TritonConfig.BLOCK_N,
            'BLOCK_K': TritonConfig.BLOCK_K,
            'BLOCK_SIZE': TritonConfig.BLOCK_SIZE
        }, num_stages=TritonConfig.NUM_STAGES, num_warps=TritonConfig.NUM_WARPS),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dds_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)

    start_inx = tl.load(offsets + pid_n).to(tl.int32)
    end_inx = tl.load(offsets + pid_n + 1).to(tl.int32)

    # pointers to dense matrix
    rm =  pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)

    A += (rm[:, None] * stride_am + rak[None, :] * stride_ak)

    # pointers to sparse matrix
    rn = tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE

    ak_sub_incr = BLOCK_K * stride_ak
    ak_block_incr = BLOCK_SIZE * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk

    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks

        if trans_B:
            ptr_B = B + (start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr
        else:
            ptr_B = B + tl.load(block_offsets_t + start_inx + block_inx).to(tl.int32) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr

        ptr_A = A + tl.load(column_indices + start_inx + block_inx).to(tl.int32) * ak_block_incr + sub_block_inx * ak_sub_incr
        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b)

    acc = acc.to(C.dtype.element_ty)
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

def dsd(shape,
        data,
        offsets,
        row_indices,
        column_indices,
        offsets_t,
        column_indices_t,
        block_offsets_t,
        transpose_a,
        rhs,
        out
    ):

    device = rhs.device
    trans_A = transpose_a
    trans_B = False

    if rhs.stride(0) > 1 and rhs.stride(1) > 1:
        trans_B = True

    # checks constraints
    assert shape[1] == rhs.shape[0], "incompatible dimensions"
    M, K = shape
    _, N = rhs.shape

    _validate_matmul_dims(M, K, N)

    # accumulator types
    ACC_TYPE = tl.float32 if rhs.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    stride_am, stride_ak = data.stride(1), data.stride(2)
    stride_bk, stride_bn = rhs.stride(0), rhs.stride(1)
    a_column_indices  = column_indices
    a_offsets = offsets

    # launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    if trans_A:
        stride_am, stride_ak = data.stride(2), data.stride(1)
        a_column_indices, a_offsets = column_indices_t, offsets_t

    if trans_B:
        stride_bk, stride_bn = rhs.stride(1), rhs.stride(0)

    _dsd_kernel[grid](
        data.data, rhs, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        row_indices, a_column_indices, a_offsets,
        block_offsets_t, trans_A, trans_B,
        GROUP_M=128, ACC_TYPE=ACC_TYPE
    )
    # return out

def dds(lhs,
        shape,
        data,
        offsets,
        row_indices,
        column_indices,
        offsets_t,
        column_indices_t,
        block_offsets_t,
        transpose_b,
        out
    ):

    device = lhs.device
    trans_B = transpose_b
    trans_A = False

    if lhs.stride(0) > 1 and lhs.stride(1) > 1:
        trans_A = True

    # checks constraints
    assert lhs.shape[1] == shape[0], "incompatible dimensions"
    M, K = lhs.shape
    _, N = shape

    _validate_matmul_dims(M, K, N)

    # accumulator types
    ACC_TYPE = tl.float32 if lhs.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    stride_am, stride_ak = lhs.stride(0), lhs.stride(1)
    stride_bk, stride_bn = data.stride(1), data.stride(2)
    b_column_indices  = column_indices_t
    b_offsets = offsets_t

    # launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    if trans_B:
        stride_bk, stride_bn = data.stride(2), data.stride(1)
        b_column_indices, b_offsets = column_indices, offsets

    _dds_kernel[grid](
        lhs, data, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        row_indices, b_column_indices, b_offsets,
        block_offsets_t, trans_A, trans_B,
        GROUP_M=128, ACC_TYPE=ACC_TYPE
    )

# ── Output-tiled SDD (MORPH SM120 occupancy fix) ────────────────────────────
# The stock _sdd_kernel launches exactly ONE CTA per nonzero output block
# (grid = (nnz,)). Each CTA computes the full [128,128] block by reducing over
# the entire K=B·S dimension. At MORPH deploy shape nnz≈48 blocks/layer, so only
# ~48 CTAs run on a 170-SM GB202 (RTX 5090 / RTX Pro 6000) — the kernel starves
# >70% of the machine. Registers cap occupancy at 2 CTAs/SM (measured n_regs≈217),
# so 340 CTAs would saturate; 48 does not. num_warps/num_stages CANNOT fix this
# (occupancy is register-bound, verified) — the fix is MORE CTAs.
#
# This kernel partitions each 128×128 output block into SPLIT_M×SPLIT_N sub-tiles,
# each computed by an independent CTA (grid = (nnz·SPLIT_M·SPLIT_N,)). Every output
# element c[i,j] = Σ_k a[i,k]·b[k,j] is still accumulated over the SAME K in the
# SAME order (BLOCK_K, tl.dot unchanged) — only the output PARTITION changes. This
# is bitwise-identical to _sdd_kernel (proven: torch.equal across all split factors,
# ignore/perf/sdd_split.py). Measured 2.17× at deploy shape (427µs→197µs).
@triton.jit
def _sdd_split_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, SPLIT_M: tl.constexpr, SPLIT_N: tl.constexpr,
            GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):
    pid = tl.program_id(0)
    nsub = SPLIT_M * SPLIT_N
    blk = pid // nsub
    sub = pid % nsub
    sm = sub // SPLIT_N
    sn = sub % SPLIT_N
    pid_m = tl.load(row_indices + blk).to(tl.int32)
    pid_n = tl.load(column_indices + blk).to(tl.int32)
    rm = pid_m * BLOCK_SIZE + sm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_SIZE + sn * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    cm = sm * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = sn * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + blk * BLOCK_ELEMENTS + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)


@functools.lru_cache(maxsize=32)
def _sdd_split_factors(nnz: int, block_size: int, sm_count: int):
    """Pick (split_m, split_n) so nnz·split_m·split_n saturates the SMs.
    Sub-tiles stay ≥32 (tensor-core min) → split factor ≤4 per axis. Returns
    (1,1) (use stock kernel) when nnz already fills the machine or block≠128."""
    if not _SDD_SPLIT_ENABLED or block_size != 128 or nnz >= 2 * sm_count:
        return (1, 1)
    target = 2 * sm_count
    best = (1, 1)
    for sm_, sn_ in [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4)]:
        if nnz * sm_ * sn_ <= 4 * sm_count:
            best = (sm_, sn_)
        if nnz * best[0] * best[1] >= target:
            break
    return best


def sdd(lhs,
        rhs,
        shape,
        out,
        offsets,
        row_indices,
        column_indices
    ):

    device = out.device
    trans_A = False
    trans_B = False

    if lhs.stride(0) > 1 and lhs.stride(1) > 1:
        trans_A = True
    if rhs.stride(0) > 1 and rhs.stride(1) > 1:
        trans_B = True

    # checks constraints
    assert lhs.shape[1] == rhs.shape[0], "incompatible dimensions"
    M, K = lhs.shape
    _, N = rhs.shape

    _validate_matmul_dims(M, K, N)

    # accumulator types
    ACC_TYPE = tl.float32 if out.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    # launch kernel
    nnz_blocks = len(row_indices)

    stride_am, stride_ak = lhs.stride(0), lhs.stride(1)
    stride_bk, stride_bn = rhs.stride(0), rhs.stride(1)

    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    if trans_B:
        stride_bk, stride_bn = rhs.stride(1), rhs.stride(0)

    block_size = out.shape[-1]
    sm_count = torch.cuda.get_device_properties(out.device).multi_processor_count
    split_m, split_n = _sdd_split_factors(nnz_blocks, block_size, sm_count)

    if split_m == 1 and split_n == 1:
        grid = lambda META: (nnz_blocks,)
        _sdd_kernel[grid](
            lhs, rhs, out, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            out.stride(1), out.stride(2),
            row_indices, column_indices,
            GROUP_M=128, ACC_TYPE=ACC_TYPE
            )
    else:
        # Output-tiled path: bitwise-identical to _sdd_kernel, more CTAs.
        grid = (nnz_blocks * split_m * split_n,)
        _sdd_split_kernel[grid](
            lhs, rhs, out, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            out.stride(1), out.stride(2),
            row_indices, column_indices,
            BLOCK_M=block_size // split_m, BLOCK_N=block_size // split_n,
            BLOCK_K=32, BLOCK_SIZE=block_size,
            SPLIT_M=split_m, SPLIT_N=split_n,
            GROUP_M=128, ACC_TYPE=ACC_TYPE,
            num_warps=4, num_stages=4,
        )

@triton.jit
def _row_indices_kernel(offsets, out):
    pid = tl.program_id(0)
    row_offset = tl.load(offsets + pid)
    nnz_blocks = tl.load(offsets + pid + 1) - row_offset
    for nnz_block in range(nnz_blocks):
        tl.store(out + row_offset + nnz_block, pid)

def row_indices(
    shape, data, offsets, column_indices, out
):
    block_rows = len(offsets) - 1
    _row_indices_kernel[(block_rows, )](offsets, out)
