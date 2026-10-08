"""ATen's CUDA reduction ORDER at a fixed shape, for a row count held on the device.

WHY (2026-10-07, the identity decomposition of the graph-captured training step). Under
``model.graph_safe`` the fixed-point and fan readers keep every row (live rows first) instead
of gathering a host-counted row set, and must still give the gathered eager result bit for
bit. ATen's CUDA reductions (Reduce.cuh) choose their thread layout from the row count, so
a row's sum or a column's mean depends on how many rows the eager call saw.
``morph.model.host_shadow`` states each layout and first emulated it with torch ops: exact,
but about 4000 tiny kernels and 11.6 ms of GPU time per training step. These kernels run
the same arithmetic in one launch each.

Every kernel performs exactly the eager kernel's fp32 operations in the eager order:
sequential adds per accumulator, the accumulators combined in order, then the tree
``lane[x] += lane[x + w]`` for ``w = width / 2 .. 1`` (shared-memory halving, then the
warp's ``shfl_down``; both are this tree). Adds of an exact 0 (masked loads past the count)
are exact. Divisions and square roots are the IEEE-rounded ones (``div_rn`` / ``sqrt_rn``),
and the Welford update's fused multiply-adds are explicit ``fma`` calls, as nvcc contracts
them in ATen's WelfordOps; no other multiply feeds an add.

Pinned bit for bit against the eager ops in tests/test_graph_safe_exact.py.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_LANES = 512          # Reduce.cuh's MAX_NUM_THREADS for fp32: the widest thread layout


@triton.jit
def _lane_tree(v, scratch, lane, width, LANES: tl.constexpr):
    """Lane 0 of the tree ``v[x] += v[x + w]`` (``w = width / 2 .. 1``) over the first
    ``width`` lanes of ``v``, through ``scratch`` (``LANES`` fp32 of global memory)."""
    tl.store(scratch + lane, v)
    w = width // 2
    while w >= 1:
        tl.debug_barrier()
        lo = tl.load(scratch + lane, mask=lane < w, other=0.0, cache_modifier=".cg")
        hi = tl.load(scratch + lane + w, mask=lane < w, other=0.0, cache_modifier=".cg")
        tl.debug_barrier()
        tl.store(scratch + lane, lo + hi, mask=lane < w)
        w = w // 2
    tl.debug_barrier()
    return tl.load(scratch, cache_modifier=".cg")


@triton.jit
def _mean1d_kernel(x_ptr, n_ptr, scratch_ptr, out_ptr, LANES: tl.constexpr):
    """``x[:n].mean()``, fp32, one output: Reduce.cuh's one-output configuration."""
    n = tl.load(n_ptr).to(tl.int32)
    lane = tl.arange(0, LANES)
    a0 = tl.zeros([LANES], tl.float32)
    a1 = tl.zeros([LANES], tl.float32)
    a2 = tl.zeros([LANES], tl.float32)
    a3 = tl.zeros([LANES], tl.float32)
    width = 1
    if n >= 128:
        # Vectorised by 4: width = min(last_pow2(n // 4), 512); thread x adds the vectors
        # x, x + width, ... into its four lane accumulators; the n % 4 tail goes to the
        # first accumulator of threads 0..3 after the loop.
        nv = n // 4
        while (width * 2 <= nv) & (width < LANES):
            width = width * 2
        for start in range(0, nv, width):
            v = start + lane
            m = (lane < width) & (v < nv)
            a0 += tl.load(x_ptr + v * 4 + 0, mask=m, other=0.0)
            a1 += tl.load(x_ptr + v * 4 + 1, mask=m, other=0.0)
            a2 += tl.load(x_ptr + v * 4 + 2, mask=m, other=0.0)
            a3 += tl.load(x_ptr + v * 4 + 3, mask=m, other=0.0)
        tail = n - n % 4
        a0 += tl.load(x_ptr + tail + lane, mask=(tail + lane) < n, other=0.0)
    else:
        # width = last_pow2(n); element e to thread e % width, accumulator (e // width) % 4.
        while width * 2 <= n:
            width = width * 2
        for start in range(0, n, 4 * width):
            m = lane < width
            a0 += tl.load(x_ptr + start + lane, mask=m & (start + lane < n), other=0.0)
            a1 += tl.load(x_ptr + start + width + lane,
                          mask=m & (start + width + lane < n), other=0.0)
            a2 += tl.load(x_ptr + start + 2 * width + lane,
                          mask=m & (start + 2 * width + lane < n), other=0.0)
            a3 += tl.load(x_ptr + start + 3 * width + lane,
                          mask=m & (start + 3 * width + lane < n), other=0.0)
    s = _lane_tree(((a0 + a1) + a2) + a3, scratch_ptr, lane, width, LANES)
    # MeanOps.project: the sum times fl(1/n).
    tl.store(out_ptr, s * tl.math.div_rn(1.0, n.to(tl.float32)))


def mean1d_aten_order(buf: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """``buf[:n].mean()`` in ATen's order (fp32 1-D, ``n`` a 0-dim int64 device tensor);
    values at ``n`` and beyond are never read. Forward only."""
    out = torch.empty((), device=buf.device, dtype=torch.float32)
    scratch = torch.empty(_LANES, device=buf.device, dtype=torch.float32)
    _mean1d_kernel[(1,)](buf, n, scratch, out, LANES=_LANES, num_warps=4)
    return out


@triton.jit
def _rowsum_kernel(x_ptr, width_ptr, scratch_ptr, out_ptr, n_cols, LANES: tl.constexpr):
    """``x[r].sum()`` for one fp32 row of ``n_cols`` (a multiple of 4) with the row's own
    thread count ``width[r]``: thread t adds the 4-wide vectors t, t + width, ... into four
    lane accumulators."""
    row = tl.program_id(0).to(tl.int64)
    width = tl.load(width_ptr + row)
    lane = tl.arange(0, LANES)
    base = x_ptr + row * n_cols
    nv = n_cols // 4
    a0 = tl.zeros([LANES], tl.float32)
    a1 = tl.zeros([LANES], tl.float32)
    a2 = tl.zeros([LANES], tl.float32)
    a3 = tl.zeros([LANES], tl.float32)
    for start in range(0, nv, width):
        v = start + lane
        m = (lane < width) & (v < nv)
        a0 += tl.load(base + v * 4 + 0, mask=m, other=0.0)
        a1 += tl.load(base + v * 4 + 1, mask=m, other=0.0)
        a2 += tl.load(base + v * 4 + 2, mask=m, other=0.0)
        a3 += tl.load(base + v * 4 + 3, mask=m, other=0.0)
    s = _lane_tree(((a0 + a1) + a2) + a3, scratch_ptr + row * LANES, lane, width, LANES)
    tl.store(out_ptr + row, s)


def rowsum_aten_order(x: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """``x.sum(-1)`` of a contiguous fp32 ``[R, F]`` with row ``r`` reduced by ``width[r]``
    threads (int32 ``[R]``, device; ``host_shadow.aten_rowsum_width`` gives ATen's). Forward
    only."""
    R, F = x.shape
    out = torch.empty(R, device=x.device, dtype=torch.float32)
    scratch = torch.empty(R, _LANES, device=x.device, dtype=torch.float32)
    _rowsum_kernel[(R,)](x, width, scratch, out, F, LANES=_LANES, num_warps=4)
    return out


@triton.jit
def _welford_step(mean, m2, nf, x, live):
    """WelfordOps.reduce, the m2 update as one fma, skipped where not ``live``."""
    new_nf = nf + 1.0
    delta = x - mean
    new_mean = mean + tl.math.div_rn(delta, new_nf)
    new_m2 = tl.math.fma(delta, x - new_mean, m2)
    return (tl.where(live, new_mean, mean), tl.where(live, new_m2, m2),
            tl.where(live, new_nf, nf))


@triton.jit
def _welford_merge(am, a2, an, bm, b2, bn):
    """WelfordOps.combine with its two fmas and its empty-side returns."""
    delta = bm - am
    cnt = an + bn
    nb = tl.math.div_rn(bn, cnt)
    mm = tl.math.fma(delta, nb, am)
    m2 = tl.math.fma((delta * delta) * an, nb, a2 + b2)
    a_empty = an == 0.0
    b_empty = bn == 0.0
    return (tl.where(a_empty, bm, tl.where(b_empty, am, mm)),
            tl.where(a_empty, b2, tl.where(b_empty, a2, m2)),
            tl.where(a_empty, bn, tl.where(b_empty, an, cnt)))


@triton.jit
def _col_std_kernel(x_ptr, n_ptr, out_ptr, n_rows, n_cols, thr,
                    LANES: tl.constexpr, BLOCK_C: tl.constexpr):
    """``x[:n].std(0, unbiased=False)`` in fp32: WelfordOps with two accumulators per
    thread; from ``thr`` rows up row e goes to y-lane e % LANES and accumulator
    (e // LANES) % 2, then the y-lanes merge in Reduce.cuh's block_y_reduce tree; below,
    one thread takes row e into accumulator e % 2."""
    pid = tl.program_id(0)
    n = tl.load(n_ptr).to(tl.int32)
    cols = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = cols < n_cols
    if n >= thr:
        y = tl.arange(0, LANES)
        mean = tl.zeros([LANES, BLOCK_C, 2], tl.float32)
        m2 = tl.zeros([LANES, BLOCK_C, 2], tl.float32)
        nf = tl.zeros([LANES, BLOCK_C, 2], tl.float32)
        acc = tl.arange(0, 2)
        for start in range(0, n, 2 * LANES):
            e = start + acc[None, None, :] * LANES + y[:, None, None]          # [L, 1, 2]
            live = (e < n) & cm[None, :, None]
            xv = tl.load(x_ptr + e * n_cols + cols[None, :, None], mask=live, other=0.0)
            mean, m2, nf = _welford_step(mean, m2, nf, xv.to(tl.float32), live)
        m_0, m_1 = tl.split(mean)
        q_0, q_1 = tl.split(m2)
        f_0, f_1 = tl.split(nf)
        vm, vq, vf = _welford_merge(m_0, q_0, f_0, m_1, q_1, f_1)            # [L, C]
        # block_y_reduce: lane y merges lane y + h, h = LANES/2 .. 1.
        vm, vq, vf = _merge_halves(vm, vq, vf, LANES, BLOCK_C)
        res = tl.math.sqrt_rn(tl.math.div_rn(vq, vf))
    else:
        # (names apart from the branch above: Triton types each name once per branch pair)
        s_mean = tl.zeros([BLOCK_C, 2], tl.float32)
        s_m2 = tl.zeros([BLOCK_C, 2], tl.float32)
        s_nf = tl.zeros([BLOCK_C, 2], tl.float32)
        k2 = tl.arange(0, 2)
        for s_start in range(0, n, 2):
            s_e = s_start + k2[None, :]
            s_live = (s_e < n) & cm[:, None]
            s_x = tl.load(x_ptr + s_e * n_cols + cols[:, None], mask=s_live, other=0.0)
            s_mean, s_m2, s_nf = _welford_step(s_mean, s_m2, s_nf, s_x.to(tl.float32), s_live)
        sm0, sm1 = tl.split(s_mean)
        sq0, sq1 = tl.split(s_m2)
        sf0, sf1 = tl.split(s_nf)
        _, s_q, s_f = _welford_merge(sm0, sq0, sf0, sm1, sq1, sf1)
        res = tl.math.sqrt_rn(tl.math.div_rn(s_q, s_f))
    tl.store(out_ptr + cols, res, mask=cm)


@triton.jit
def _merge_halves(vm, vq, vf, LANES: tl.constexpr, BLOCK_C: tl.constexpr):
    """Merge ``[LANES, C]`` Welford lanes down to ``[C]``: lane y with lane y + LANES/2,
    halving to one lane (LANES a power of two, at most 8)."""
    if LANES >= 8:
        vm, vq, vf = _merge_once(vm, vq, vf, 8, BLOCK_C)
    if LANES >= 4:
        vm, vq, vf = _merge_once(vm, vq, vf, 4, BLOCK_C)
    if LANES >= 2:
        vm, vq, vf = _merge_once(vm, vq, vf, 2, BLOCK_C)
    return tl.reshape(vm, [BLOCK_C]), tl.reshape(vq, [BLOCK_C]), tl.reshape(vf, [BLOCK_C])


@triton.jit
def _merge_once(vm, vq, vf, W: tl.constexpr, BLOCK_C: tl.constexpr):
    """``[W, C]`` -> ``[W/2, C]``: lane y merged with lane y + W/2."""
    h: tl.constexpr = W // 2
    vm = tl.permute(tl.reshape(vm, [2, h, BLOCK_C]), [1, 2, 0])
    vq = tl.permute(tl.reshape(vq, [2, h, BLOCK_C]), [1, 2, 0])
    vf = tl.permute(tl.reshape(vf, [2, h, BLOCK_C]), [1, 2, 0])
    am, bm = tl.split(vm)
    aq, bq = tl.split(vq)
    af, bf = tl.split(vf)
    return _welford_merge(am, aq, af, bm, bq, bf)


def col_std_aten_order(x: torch.Tensor, n: torch.Tensor, lanes: int, thr: int) -> torch.Tensor:
    """``x[:n].std(0, unbiased=False)`` for a contiguous fp32/bf16 ``[R, C]`` in ATen's
    Welford order (``lanes`` / ``thr`` from ``host_shadow._col_split_rows``), fp32 out."""
    R, C = x.shape
    if lanes not in (1, 2, 4, 8):
        raise ValueError(f"col_std_aten_order: {lanes} y-lanes (the kernel merges up to 8)")
    out = torch.empty(C, device=x.device, dtype=torch.float32)
    block_c = 64
    _col_std_kernel[(triton.cdiv(C, block_c),)](x, n, out, R, C, thr, LANES=lanes,
                                                BLOCK_C=block_c, num_warps=4)
    return out


@triton.jit
def _col_sum_seq_kernel(x_ptr, n_ptr, out_ptr, n_cols, thr, BLOCK_C: tl.constexpr):
    """``x[:min(n, thr)].sum(0)`` with one thread per column: row e into accumulator e % 4,
    the four combined in order (Reduce.cuh below the split threshold)."""
    pid = tl.program_id(0)
    n = tl.minimum(tl.load(n_ptr).to(tl.int32), thr)
    cols = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = cols < n_cols
    acc = tl.zeros([BLOCK_C, 4], tl.float32)
    k = tl.arange(0, 4)
    for start in range(0, n, 4):
        e = start + k[None, :]
        acc += tl.load(x_ptr + e * n_cols + cols[:, None], mask=(e < n) & cm[:, None],
                       other=0.0).to(tl.float32)
    # acc[c, 2a + b] -> [c, a, b]: split b, then a.
    even, odd = tl.split(tl.reshape(acc, [BLOCK_C, 2, 2]))      # k in {0, 2} / {1, 3}
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    tl.store(out_ptr + cols, ((a0 + a1) + a2) + a3, mask=cm)


def col_sum_seq(x: torch.Tensor, n: torch.Tensor, thr: int) -> torch.Tensor:
    """fp32 column sums of ``x[:min(n, thr)]`` in ATen's one-thread order (fp32/bf16 in;
    fp64 is summed by ``host_shadow`` in torch, this kernel is fp32)."""
    R, C = x.shape
    out = torch.empty(C, device=x.device, dtype=torch.float32)
    block_c = 64
    _col_sum_seq_kernel[(triton.cdiv(C, block_c),)](x, n, out, C, thr, BLOCK_C=block_c,
                                                    num_warps=4)
    return out
