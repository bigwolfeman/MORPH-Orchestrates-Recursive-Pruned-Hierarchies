"""Host shadows: the CPU copy of a small device tensor whose value the host already knows.

WHY (2026-10-04 training-speed pass). A boolean-mask index ``x[mask]``, a ``bool(m.any())``
and an ``int(m.sum().max())`` on a CUDA tensor each stop the host until the GPU has run
everything queued before them. The slot loop did about 190 of these per training step,
most of them on masks the host had in hand before the batch ever reached the GPU (the
layout's ``slot_valid`` and ``slot_mask`` come from the CPU packer) or could have in hand
after the ONE sync it cannot avoid (the per-slot depth table, drawn on the GPU, whose max
sets the Python loop count). Every such sync drains the queue, so the host cannot launch
the next region while the GPU still runs the last one.

WHAT. :func:`attach` records a numpy copy of a device tensor's value on the tensor object
itself, keyed by the tensor's ``_version`` (an in-place write invalidates it).
:func:`shadow` returns it, or ``None``. Derived tensors (``tensor[perm]``, ``.repeat``,
``torch.where``...) are NEW objects without a shadow, so a shadow can never describe a
different value than its tensor: a reader that finds none takes the old synchronising
path. That fallback is the contract, not a degradation: every reader below must give the
same result with or without a shadow, which is what ``tests/test_host_shadow.py`` pins.

The readers are exact replacements:

* :func:`true_index` — the flat indices of a mask's True entries in ascending order, the
  same tensor as ``mask.flatten().nonzero().squeeze(1)``. With a shadow the count is known
  on the host and a stable argsort finds the positions on the device without a sync.
* :func:`masked_rows` — ``x[mask]`` for a mask over ``x``'s leading dims, as
  ``x.flatten(...)[true_index(mask)]``. Forward values are identical, and the backward is
  the same ``index_put_(accumulate=True)`` with the same (unique) indices, so the gradient
  is bit-identical too.
* :func:`any_true` / :func:`count_true` — ``bool(mask.any())`` / ``int(mask.sum())`` from
  the shadow when there is one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

_ATTR = "_morph_host_shadow"


def attach(t: Tensor, host: np.ndarray) -> Tensor:
    """Record ``host`` (the exact value of ``t``) on ``t`` and return ``t``."""
    host = np.asarray(host)
    if tuple(host.shape) != tuple(t.shape):
        raise ValueError(f"host shadow shape {host.shape} != tensor shape {tuple(t.shape)}")
    setattr(t, _ATTR, (t._version, host))
    return t


def shadow(t: Tensor | None) -> np.ndarray | None:
    """The host copy recorded by :func:`attach`, or ``None`` (none recorded, or the
    tensor was written in place since)."""
    if t is None:
        return None
    rec = getattr(t, _ATTR, None)
    if rec is None or rec[0] != t._version:
        return None
    return rec[1]


def to_device(t: Tensor, device, non_blocking: bool = True) -> Tensor:
    """``t.to(device)`` with the source's value recorded as the result's shadow when the
    source is a CPU tensor (the value the host already holds)."""
    out = t.to(device, non_blocking=non_blocking)
    if t.device.type == "cpu":
        # A CPU target returns `t` itself; recording its own value on it is harmless and
        # lets a CPU run take the same reader paths a CUDA run takes.
        attach(out, t.numpy().copy())
    elif out is not t:
        h = shadow(t)
        if h is not None:
            attach(out, h)
    return out


def derive(out: Tensor, host: np.ndarray | None) -> Tensor:
    """Attach ``host`` to ``out`` when it is known (``None`` leaves ``out`` bare)."""
    if host is not None:
        attach(out, host)
    return out


def any_true(mask: Tensor) -> bool:
    """``bool(mask.any())``, from the shadow when there is one."""
    h = shadow(mask)
    return bool(h.any()) if h is not None else bool(mask.any())


def count_true(mask: Tensor) -> int:
    """``int(mask.sum())``, from the shadow when there is one."""
    h = shadow(mask)
    return int(h.sum()) if h is not None else int(mask.sum())


def true_index(mask: Tensor) -> Tensor:
    """Flat indices of ``mask``'s True entries, ascending: ``mask.flatten().nonzero()``.

    With a shadow: the count is the host's, and a STABLE argsort of ``~mask`` puts the True
    positions first in their original order, so the first ``n`` entries are exactly the
    nonzero indices, found without a host sync."""
    h = shadow(mask)
    flat = mask.reshape(-1)
    if h is None:
        return flat.nonzero().squeeze(1)
    n = int(h.sum())
    return torch.argsort((~flat).to(torch.uint8), stable=True)[:n]


def masked_rows(x: Tensor, mask: Tensor) -> Tensor:
    """``x[mask]`` for a boolean ``mask`` over ``x``'s leading ``mask.dim()`` dims."""
    k = mask.dim()
    if tuple(x.shape[:k]) != tuple(mask.shape):
        raise ValueError(f"masked_rows: mask {tuple(mask.shape)} does not lead x "
                         f"{tuple(x.shape)}")
    if shadow(mask) is None:
        return x[mask]
    return x.reshape(-1, *x.shape[k:])[true_index(mask)]


# ── model.graph_safe: the readers above at a FIXED shape, bit for bit ────────────────
# A captured CUDA graph replays one op sequence at one set of shapes, so a reader whose row
# count is the host's count of a mask (`masked_rows`, `true_index`) or whose branch is the
# host's answer to "is any row set?" (`any_true`) is wrong on the next batch even with a
# shadow. The graph-captured step must also train bit for bit like the eager one (2026-10-07
# identity decomposition: the first fixed forms, a masked sum over every row in another
# order, broke identity at steps 1-11 of a 45-step gate). Two facts make an exact
# fixed-shape form possible:
#   * every per-row op of those readers (elementwise, a last-dim reduction over >= 16 rows,
#     a small batched GEMM, a batched slogdet) gives each row the same bits whether the rows
#     are gathered or not; only reductions ACROSS rows depend on the row count;
#   * ATen's CUDA reduction order for a 1-D fp32 tensor (Reduce.cuh: block width, vector
#     lanes, tail, shared-memory and warp-shuffle tree) is a deterministic function of the
#     count, which `_aten_sum_order_1d` evaluates at a fixed length for a count held on
#     the device.
# Pinned bit for bit against the gathered eager ops in tests/test_graph_safe.py.


def valid_first(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """``(rows, n)``: ``x``'s rows over ``mask``'s leading dims with the True rows FIRST, in
    the order ``masked_rows(x, mask)`` returns them, then the others set to exactly 0;
    ``n`` is the True count as a 0-dim int64 DEVICE tensor. ``rows[:n]`` is
    ``masked_rows(x, mask)`` bit for bit, and its gradient reaches the same elements of
    ``x`` with the same values (a permutation gather; its backward adds each row once)."""
    k = mask.dim()
    if tuple(x.shape[:k]) != tuple(mask.shape):
        raise ValueError(f"valid_first: mask {tuple(mask.shape)} does not lead x "
                         f"{tuple(x.shape)}")
    flat = mask.reshape(-1)
    perm = torch.argsort((~flat).to(torch.uint8), stable=True)
    n = flat.sum()
    rows = x.reshape(-1, *x.shape[k:]).index_select(0, perm)
    keep = (torch.arange(flat.numel(), device=flat.device) < n).view(-1, *([1] * (rows.dim() - 1)))
    return torch.where(keep, rows, torch.zeros_like(rows)), n


def _last_pow2(x: Tensor) -> Tensor:
    """Reduce.cuh's ``last_pow2`` (the largest power of two <= x, at least 1) for a device
    int64 tensor, by comparisons (``log2`` of a power of two need not round to it)."""
    lp = torch.ones_like(x)
    for p in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        lp = torch.where(x >= p, torch.full_like(x, p), lp)
    return lp


def _halve_to_lane0(v: Tensor) -> Tensor:
    """Reduce.cuh's ``block_x_reduce`` over the last dim (a power of two): the shared-memory
    halving down to 32 lanes, then the warp's ``shfl_down`` offsets 16, 8, 4, 2, 1. Every
    step is ``lane[x] + lane[x + h]`` for ``x < h``, so lane 0 ends with exactly this tree."""
    w = v.shape[-1]
    while w > 1:
        w //= 2
        v = v[..., :w] + v[..., w:2 * w]
    return v[..., 0]


def _aten_sum_order_1d(buf: Tensor, n: Tensor) -> Tensor:
    """``buf[:n].sum()`` with ATen's CUDA reduction order for a contiguous, 16-byte-aligned
    fp32 tensor of ``n`` elements and one output, at the fixed length ``buf.shape[0]``
    (``buf``'s values at ``n`` and beyond are ignored). Reduce.cuh's ``setReduceConfig`` for
    one output: ``n >= 128`` vectorises by 4 and uses ``bw = min(last_pow2(n // 4), 512)``
    threads, thread ``x`` adding vectors ``x, x + bw, ...`` into four lane accumulators, the
    ``n % 4`` tail elements to thread ``x``'s first accumulator after the loop; ``n < 128``
    uses ``bw = last_pow2(n)`` threads, element ``e`` going to thread ``e % bw``,
    accumulator ``(e // bw) % 4``. Then the four accumulators combine in order and the
    threads by :func:`_halve_to_lane0`. Every candidate width is evaluated and the one the
    count selects is kept, so nothing is read on the host. The elements at ``n`` and beyond
    are zeroed first; adding an exact 0 is exact, so they may run through the loops."""
    L = buf.shape[0]
    dev, dt = buf.device, buf.dtype
    pos = torch.arange(L, device=dev)
    outs, picks = [], []
    # Vectorised regime (n >= 128).
    bw_vec = _last_pow2((n // 4).clamp(min=1)).clamp(max=512)
    r = n % 4
    tail_start = n - r
    main = torch.where(pos < tail_start, buf, torch.zeros_like(buf))
    tail = torch.where(pos[:4] < r, buf[(tail_start + pos[:4]).clamp(max=L - 1)],
                       torch.zeros(4, device=dev, dtype=dt))
    bw = 32
    while bw <= 512 and 4 * bw <= max(L, 128):
        k_steps = -(-L // (4 * bw))
        m = torch.cat([main, main.new_zeros(k_steps * 4 * bw - L)]).view(k_steps, bw, 4)
        acc = m[0]
        for k in range(1, k_steps):
            acc = acc + m[k]
        a0 = acc[:, 0] + torch.cat([tail, tail.new_zeros(bw - 4)])
        outs.append(_halve_to_lane0(((a0 + acc[:, 1]) + acc[:, 2]) + acc[:, 3]))
        picks.append((n >= 128) & (bw_vec == bw))
        bw *= 2
    # Plain regime (n < 128): only the first 128 elements can be live.
    small = torch.where(pos[:min(L, 128)] < n, buf[:128], torch.zeros_like(buf[:128]))
    small = torch.cat([small, small.new_zeros(256 - small.shape[0])])
    bw_small = _last_pow2(n.clamp(min=1))
    for bw in (1, 2, 4, 8, 16, 32, 64):
        k_steps = max(1, 128 // (4 * bw))
        m = small[:k_steps * 4 * bw].view(k_steps, 4, bw)
        acc = m[0]
        for k in range(1, k_steps):
            acc = acc + m[k]
        outs.append(_halve_to_lane0(((acc[0] + acc[1]) + acc[2]) + acc[3]))
        picks.append((n < 128) & (bw_small == bw))
    # Exactly one pick is True; `where` selects it without arithmetic on the others.
    out = torch.zeros((), device=dev, dtype=dt)
    for o, p in zip(outs, picks):
        out = torch.where(p, o, out)
    return out


class _ExactMean1d(torch.autograd.Function):
    """``buf[:n].mean()`` exactly as the eager op computes it, forward and backward."""

    @staticmethod
    def forward(ctx, buf: Tensor, n: Tensor) -> Tensor:
        ctx.save_for_backward(n)
        ctx.length = buf.shape[0]
        if buf.is_cuda:                              # the same order in one launch
            from morph.kernels.triton.exact_reduce import mean1d_aten_order
            return mean1d_aten_order(buf, n)
        # MeanOps.project: the sum times fl(1/n) (`factor`, a float32 reciprocal).
        return _aten_sum_order_1d(buf, n) * torch.reciprocal(n.to(buf.dtype))

    @staticmethod
    def backward(ctx, g: Tensor):
        (n,) = ctx.saved_tensors
        # MeanBackward: the gradient times fl(1/n) (measured bit for bit, 1800 of 1800
        # draws, where g / n differs from it in a third of them), 0 beyond n.
        gi = (g * torch.reciprocal(n.to(g.dtype))).expand(ctx.length)
        live = torch.arange(ctx.length, device=g.device) < n
        return torch.where(live, gi, torch.zeros_like(gi)), None


def exact_mean_1d(buf: Tensor, n: Tensor) -> Tensor:
    """``buf[:n].mean()`` bit for bit (fp32, ``n`` a 0-dim int64
    device tensor), forward and gradient, at the fixed length of ``buf``. ``n == 0`` reads
    NaN, as the eager mean of nothing does; callers guard it as the gathered path did."""
    if buf.dtype != torch.float32 or buf.dim() != 1:
        raise ValueError(f"exact_mean_1d: fp32 1-D only, got {buf.dtype} {tuple(buf.shape)}")
    return _ExactMean1d.apply(buf.contiguous(), n)


def aten_rowsum_width(rows: int, n_cols: int) -> int:
    """Threads per row Reduce.cuh runs for ``x.sum(-1)`` of a contiguous fp32
    ``[rows, n_cols]`` tensor (``n_cols % 4 == 0`` and ``>= 128``: vectorised by 4). A row's
    sum depends on it, and it depends on ``rows`` below 16 (e.g. 4096 columns: 512 threads
    for 1 row, 256 for 2-3, 128 for 4-7, 64 for 8-15, 32 from 16 up)."""
    if n_cols % 4 or n_cols < 128:
        raise ValueError(f"aten_rowsum_width: {n_cols} columns (needs a multiple of 4, >= 128)")
    d0 = min(n_cols // 4, 512)
    d0 = 512 if n_cols // 4 >= 512 else _last_pow2_int(d0)
    d1 = 512 if rows >= 512 else _last_pow2_int(rows)
    bh = min(d1, 512 // min(d0, 32))
    return min(d0, 512 // bh)


def aten_rowsum_order(x: Tensor, bw: int) -> Tensor:
    """``x.sum(-1)`` for a contiguous fp32 ``[R, F]`` (``F % 4 == 0``, 16-byte-aligned rows)
    in ATen's order when the reduction runs ``bw`` threads per row
    (:func:`aten_rowsum_width`): thread ``t`` adds the 4-wide vectors ``t, t + bw, ...`` into
    four lane accumulators, which combine in order, then :func:`_halve_to_lane0`. The
    fixed-point term needs the orders of the passes that finish fewer than 16 cells."""
    R, F = x.shape
    nv = F // 4
    k_steps = -(-nv // bw)
    if k_steps * bw != nv:                          # threads past the last vector add 0
        x = torch.cat([x, x.new_zeros(R, (k_steps * bw - nv) * 4)], dim=1)
    m = x.reshape(R, k_steps, bw, 4)
    acc = m[:, 0]
    for k in range(1, k_steps):
        acc = acc + m[:, k]
    return _halve_to_lane0(((acc[..., 0] + acc[..., 1]) + acc[..., 2]) + acc[..., 3])


class _RowsumByWidth(torch.autograd.Function):
    """``x.sum(-1)`` with row ``r`` in the order of ``width[r]`` threads; the backward is
    SumBackward's (the gradient expanded over the row, no arithmetic)."""

    @staticmethod
    def forward(ctx, x: Tensor, width: Tensor) -> Tensor:
        ctx.n_cols = x.shape[1]
        if x.is_cuda:
            from morph.kernels.triton.exact_reduce import rowsum_aten_order
            return rowsum_aten_order(x, width.to(torch.int32))
        out = x.new_zeros(x.shape[0])
        for bw in sorted({int(w) for w in width.tolist()}):  # CPU reference path only
            out = torch.where(width == bw, aten_rowsum_order(x, bw), out)
        return out

    @staticmethod
    def backward(ctx, g: Tensor):
        return g.unsqueeze(-1).expand(g.shape[0], ctx.n_cols), None


def rowsum_by_width(x: Tensor, width: Tensor) -> Tensor:
    """``x.sum(-1)`` of a contiguous fp32 ``[R, F]`` where row ``r`` is summed as ATen sums
    it with ``width[r]`` threads per row (:func:`aten_rowsum_width`), forward and gradient."""
    return _RowsumByWidth.apply(x.contiguous(), width)


def _col_split_rows(n_cols: int) -> tuple[int, int]:
    """``(lanes, threshold)`` of Reduce.cuh's configuration for a reduction over dim 0 of a
    contiguous ``[R, n_cols]`` tensor (output vectorised by 4, 128 threads per vector lane
    group): from ``threshold`` rows up the rows split over ``lanes`` y-threads (row ``e`` to
    lane ``e % lanes``); below it one thread walks every row of its columns."""
    if n_cols % 4:
        raise ValueError(f"exact column reductions: {n_cols} columns is not a multiple of 4 "
                         f"(ATen's output vector width would shrink; not emulated)")
    dim0 = n_cols // 4
    max_thr = 512 // 4
    d0 = max_thr if dim0 >= max_thr else _last_pow2_int(dim0)
    lanes = max_thr // min(d0, 32)
    return lanes, min(lanes * 16, 256)


def _last_pow2_int(x: int) -> int:
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _col_lane_sum(x: Tensor, n: Tensor, vt0: int, lanes: int) -> Tensor:
    """fp32 accumulators of ATen's dim-0 sum order for ``x[:n]`` (``x`` fp32 ``[R, C]``,
    ``R`` a multiple of ``vt0 * lanes``, rows at ``n`` and beyond ignored): row ``e`` goes to
    lane ``e % lanes``, accumulator ``(e // lanes) % vt0``, in row order; the accumulators
    combine in order, then the lanes in Reduce.cuh's ``block_y_reduce`` tree (offsets
    ``lanes / 2 .. 1``). ``lanes == 1`` is the one-thread regime."""
    R, C = x.shape
    live = (torch.arange(R, device=x.device) < n).unsqueeze(1)
    x = torch.where(live, x, torch.zeros_like(x)).view(R // (vt0 * lanes), vt0, lanes, C)
    acc = x[0]
    for k in range(1, x.shape[0]):
        acc = acc + x[k]
    v = acc[0]
    for i in range(1, vt0):
        v = v + acc[i]                                                 # [lanes, C]
    while v.shape[0] > 1:
        h = v.shape[0] // 2
        v = v[:h] + v[h:]
    return v[0]


class _ExactColMean(torch.autograd.Function):
    """``x[:n].mean(0)`` exactly as the eager op computes it, forward and backward."""

    @staticmethod
    def forward(ctx, x: Tensor, n: Tensor) -> Tensor:
        R, C = x.shape
        lanes, thr = _col_split_rows(C)
        opm = torch.float64 if x.dtype == torch.float64 else torch.float32
        xo = x.to(opm)
        # From `thr` rows up the split order does not depend on the count, so the sum over
        # the zero-padded rows IS the eager sum (adding an exact 0 is exact). Past
        # 256 * lanes rows ATen would also split across blocks, which is not emulated.
        if R > 256 * lanes:
            raise ValueError(f"exact_col_mean: {R} rows > {256 * lanes}, where ATen splits "
                             f"the reduction across blocks (not emulated)")
        if x.is_cuda and opm == torch.float32:       # one thread per column, n < thr
            from morph.kernels.triton.exact_reduce import col_sum_seq
            s = col_sum_seq(x, n, thr)
        else:
            k = -(-thr // 4) * 4
            head = torch.cat([xo[:min(R, k)], xo.new_zeros(max(0, k - R), C)])
            s = _col_lane_sum(head, n, 4, 1)
        if R >= thr:
            live = (torch.arange(R, device=x.device) < n).unsqueeze(1)
            s = torch.where(n >= thr, torch.where(live, xo, torch.zeros_like(xo)).sum(0), s)
        ctx.save_for_backward(n)
        ctx.shape, ctx.dtype = (R, C), x.dtype
        return (s * torch.reciprocal(n.to(opm))).to(x.dtype)

    @staticmethod
    def backward(ctx, g: Tensor):
        (n,) = ctx.saved_tensors
        R, C = ctx.shape
        opm = torch.float64 if g.dtype == torch.float64 else torch.float32
        gi = (g.to(opm) * torch.reciprocal(n.to(opm))).to(ctx.dtype).expand(R, C)
        live = (torch.arange(R, device=g.device) < n).unsqueeze(1)
        return torch.where(live, gi, torch.zeros_like(gi)), None


def exact_col_mean(x: Tensor, n: Tensor) -> Tensor:
    """``x[:n].mean(0)`` bit for bit for a contiguous ``[R, C]`` tensor whose first ``n``
    rows are the live ones (``valid_first``), forward and gradient (MeanBackward's
    ``g * fl(1/n)``), with ``n`` on the device."""
    return _ExactColMean.apply(x.contiguous(), n)


def _welford_reduce(acc, x, live):
    """WelfordOps.reduce on fp32 tensors, with the kernel's FMA contraction
    (``m2 + delta * new_delta``) kept as one rounding (``torch.addcmul``)."""
    mean, m2, nf = acc
    new_nf = nf + 1.0
    delta = x - mean
    new_mean = mean + delta / new_nf
    new_m2 = torch.addcmul(m2, delta, x - new_mean)
    return (torch.where(live, new_mean, mean), torch.where(live, new_m2, m2),
            torch.where(live, new_nf, nf))


def _welford_combine(a, b):
    """WelfordOps.combine with the kernel's FMAs (``a.mean + delta * nb_over_n`` and
    ``(a.m2 + b.m2) + (delta * delta * a.nf) * nb_over_n``) and its empty-side returns."""
    am, a2, an = a
    bm, b2, bn = b
    delta = bm - am
    cnt = an + bn
    nb = bn / cnt
    out = (torch.addcmul(am, delta, nb), torch.addcmul(a2 + b2, (delta * delta) * an, nb), cnt)
    return tuple(torch.where(an == 0, y, torch.where(bn == 0, x, z))
                 for x, y, z in zip(a, b, out))


@torch.no_grad()
def exact_col_std(x: Tensor, n: Tensor) -> Tensor:
    """``x[:n].std(0, unbiased=False)`` bit for bit (WelfordOps in fp32, unroll 2, Reduce.cuh's
    dim-0 split) for a contiguous fp32/bf16 ``[R, C]`` tensor whose first ``n`` rows are
    the live ones, ``n`` on the device. Forward only (the caller is a no-grad readout).
    Both regimes are evaluated (from ``threshold`` rows up: row ``e`` to lane
    ``e % lanes``, accumulator ``(e // lanes) % 2``; below: one thread, accumulator
    ``e % 2``) and the count picks one."""
    R, C = x.shape
    lanes, thr = _col_split_rows(C)
    if x.is_cuda:                                    # both regimes in one launch
        from morph.kernels.triton.exact_reduce import col_std_aten_order
        return col_std_aten_order(x.contiguous(), n, lanes, thr).to(x.dtype)
    xf = x.float()
    out = []
    for ln, rows in ((lanes, R), (1, min(R, thr))):
        step = 2 * ln
        rr = -(-rows // step) * step
        xs = torch.cat([xf[:rows], xf.new_zeros(rr - rows, C)]).view(rr // step, 2, ln, C)
        e = torch.arange(rr, device=x.device).view(rr // step, 2, ln, 1)
        z = xf.new_zeros(2, ln, C)
        acc = (z, z, z)
        for k in range(rr // step):
            acc = _welford_reduce(acc, xs[k], e[k] < n)
        v = _welford_combine(tuple(a[0] for a in acc), tuple(a[1] for a in acc))
        while v[0].shape[0] > 1:
            h = v[0].shape[0] // 2
            v = _welford_combine(tuple(a[:h] for a in v), tuple(a[h:] for a in v))
        mean, m2, nf = (a[0] for a in v)
        out.append(torch.sqrt(m2 / nf))
    return torch.where(n >= thr, out[0], out[1]).to(x.dtype)
