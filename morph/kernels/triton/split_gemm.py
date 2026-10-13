"""``model.split_wgrad_triton``: Triton GEMMs for the three matmuls of ``split_linear``.

``split_linear`` (``morph/model/split_wgrad.py``) does, per covered linear ``y = x W^T``:

  * forward  ``y  = x  @ W^T``   [M, K] x [K, N] -> [M, N]
  * dgrad    ``dx = dy @ W``     [M, N] x [N, K] -> [M, K]   (main stream)
  * wgrad    ``dW += dy^T @ x``  [N, M] x [M, K] -> [N, K]   (grad stream)

cuBLAS picks 128x128 / 256x128 tiles and a few wmma 32x32 kernels at the core's small, varying row
counts (M = 408 ... 1,536), so some GEMMs leave SMs idle. One plain tiled ``tl.dot`` kernel with
64x64 / 64x128 tiles filled the 170 SMs better: 0.83 x cuBLAS time summed over the core shapes
(isolated, ``2026-10-12-core-pass-persistent-kernel-plan.md`` Q2).

THE WGRAD EPILOGUE. Today the weight grad is ``torch.mm`` into a bf16 scratch, then ``buf.copy_``
(first read of a backward) or ``buf.add_`` (later reads) into the fp32 sum. Here the kernel rounds
the fp32 accumulator to bf16 (the scratch's rounding), widens it, and writes it (``MODE_FIRST``) or
adds it to ``buf`` in fp32 (``MODE_ADD``): the same arithmetic as the cast and add kernels, without
the scratch and without two launches per read.

TUNING. ``tuned_mm`` times every config of ``CONFIGS`` once per key ``(mode, M, N, K, strides)`` and
keeps the fastest. It runs in the eager warm-up. A key that is not tuned when a CUDA graph is
being captured RAISES (benchmarking inside a capture would corrupt it, and a first launch of a
config compiles it), except inside ``CAPTURE_TUNER.recording()`` (``GraphStep``'s hook): there the
key is collected, the GEMM runs its cuBLAS equivalent, and ``GraphStep`` tunes the collected keys
on synthetic tensors, throws the graphs away and records again. Tuning also compiles the other wgrad mode for the winning config, so a launch
in a capture never compiles. Every config accumulates K in the same sequential order (a block of
``BK`` is ``BK/16`` mma steps into one accumulator), so the winner changes speed, not bits.
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["CONFIGS", "MODE_PLAIN", "MODE_FIRST", "MODE_ADD", "tuned_mm", "linear_fwd",
           "linear_dgrad", "linear_wgrad", "tuned_keys", "clear_tuned", "CAPTURE_TUNER"]

MODE_PLAIN, MODE_FIRST, MODE_ADD = 0, 1, 2

# (BM, BN, BK, num_warps, num_stages): the list of the isolated measurement (triton_gemm_bench.py)
CONFIGS: tuple[tuple[int, int, int, int, int], ...] = (
    (128, 128, 64, 8, 3), (128, 128, 32, 4, 4), (128, 128, 64, 4, 3), (64, 128, 64, 4, 4),
    (128, 64, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 32, 4, 5), (128, 256, 32, 8, 3),
    (256, 128, 32, 8, 3), (64, 256, 32, 4, 4), (32, 128, 64, 4, 4),
)
_GM = 8                                   # grouped tile order: M-tiles per group (L2 reuse)
_TUNE_REPS = 3                            # timed launches per config; the fastest counts


@triton.jit
def _mm_kernel(a, b, c, M, N, K, sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
               MODE: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    width = GM * npn
    first = (pid // width) * GM
    gsz = min(npm - first, GM)
    pm = first + (pid % width) % gsz
    pn = (pid % width) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a + rm[:, None] * sam + rk[None, :] * sak
    bp = b + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        km = rk < K - k * BK
        av = tl.load(ap, mask=(rm[:, None] < M) & km[None, :], other=0.0)
        bv = tl.load(bp, mask=km[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(av, bv, acc)
        ap += BK * sak
        bp += BK * sbk
    cp = c + rm[:, None] * scm + rn[None, :] * scn
    cm = (rm[:, None] < M) & (rn[None, :] < N)
    if MODE == 0:
        tl.store(cp, acc.to(c.dtype.element_ty), mask=cm)
    elif MODE == 1:      # the bf16 scratch's rounding, then the fp32 copy
        tl.store(cp, acc.to(tl.bfloat16).to(tl.float32).to(c.dtype.element_ty), mask=cm)
    else:                # ... and the fp32 add into the sum
        old = tl.load(cp, mask=cm, other=0.0).to(tl.float32)
        tl.store(cp, (old + acc.to(tl.bfloat16).to(tl.float32)).to(c.dtype.element_ty), mask=cm)


_TUNED: dict[tuple, tuple[int, int, int, int, int]] = {}


def tuned_keys() -> list[tuple]:
    return list(_TUNED)


def clear_tuned() -> None:
    _TUNED.clear()
    _PENDING.clear()


def _launch(a: Tensor, b: Tensor, c: Tensor, mode: int, cfg: tuple) -> None:
    M, K = a.shape
    N = b.shape[1]
    bm, bn, bk, nw, ns = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _mm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), BM=bm, BN=bn, BK=bk, GM=_GM, MODE=mode,
                     num_warps=nw, num_stages=ns)


def _tune(a: Tensor, b: Tensor, c: Tensor, mode: int) -> tuple:
    """Time every config on throwaway output (``c`` is never touched) and return the fastest."""
    scratch = torch.empty_like(c)
    if mode != MODE_PLAIN:
        scratch.zero_()
    best, best_t = None, float("inf")
    for cfg in CONFIGS:
        _launch(a, b, scratch, mode, cfg)                  # compile + warm
        ts = []
        for _ in range(_TUNE_REPS):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            _launch(a, b, scratch, mode, cfg)
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        if min(ts) < best_t:
            best, best_t = cfg, min(ts)
    if mode != MODE_PLAIN:                                 # the other wgrad mode never compiles later
        for m in (MODE_FIRST, MODE_ADD):
            _launch(a, b, scratch, m, best)
    torch.cuda.synchronize()
    return best


def _cublas_equivalent(a: Tensor, b: Tensor, c: Tensor, mode: int) -> None:
    """The cuBLAS form of each mode (same rounding rule), for a recording that only collects."""
    if mode == MODE_PLAIN:
        torch.mm(a, b, out=c)
    elif mode == MODE_FIRST:
        c.copy_(torch.mm(a, b))
    else:
        c.add_(torch.mm(a, b))


def tuned_mm(a: Tensor, b: Tensor, c: Tensor, mode: int = MODE_PLAIN) -> None:
    """``c (op)= a @ b`` for strided 2-D bf16 ``a``, ``b``; ``mode`` is the epilogue (module notes).
    An untuned shape tunes in eager mode; under a CUDA graph capture it RAISES, unless the
    capture is inside ``CAPTURE_TUNER.recording()``: then the key is collected and the GEMM runs
    its cuBLAS equivalent so the recording can end (the caller tunes the keys and records again)."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"split_wgrad_triton: GEMM inputs must be bf16, got {a.dtype} and {b.dtype}")
    if a.shape[1] != b.shape[0] or c.shape != (a.shape[0], b.shape[1]):
        raise ValueError(f"split_wgrad_triton: shapes {tuple(a.shape)} x {tuple(b.shape)} -> "
                         f"{tuple(c.shape)}")
    # the first-read / add modes share one tuning (and one winner)
    key = (MODE_PLAIN if mode == MODE_PLAIN else MODE_FIRST, a.shape[0], b.shape[1], a.shape[1],
           a.stride(), b.stride(), c.stride(), c.dtype, a.device.index)
    cfg = _TUNED.get(key)
    if cfg is None:
        if torch.cuda.is_current_stream_capturing():
            if not _COLLECT:
                raise RuntimeError(
                    f"split_wgrad_triton: GEMM (mode {mode}, M={a.shape[0]}, N={b.shape[1]}, "
                    f"K={a.shape[1]}, strides {a.stride()} {b.stride()} {c.stride()}) was not "
                    f"tuned in the eager warm-up; it cannot be tuned inside a CUDA graph capture")
            _PENDING[key] = (key[0], tuple(a.shape), a.stride(), tuple(b.shape), b.stride(),
                             c.stride(), c.dtype, a.device)
            _cublas_equivalent(a, b, c, mode)
            return
        cfg = _TUNED[key] = _tune(a, b, c, mode)
    _launch(a, b, c, mode, cfg)


_COLLECT = False
# key -> (mode, a shape, a strides, b shape, b strides, c strides, c dtype, device): all that a
# tuning reads (shapes, strides, dtype and mode, never values)
_PENDING: dict[tuple, tuple] = {}


class _CaptureTuner:
    """The hook ``GraphStep`` takes (``capture_tuner``): ``recording()`` turns collect mode on
    around a recording pass, ``pending()`` counts the collected untuned keys, ``tune()`` tunes
    them on synthetic tensors and returns how many."""

    @staticmethod
    @contextlib.contextmanager
    def recording():
        global _COLLECT
        prev, _COLLECT = _COLLECT, True
        try:
            yield
        finally:
            _COLLECT = prev

    @staticmethod
    def pending() -> int:
        return len(_PENDING)

    @staticmethod
    def tune() -> int:
        n = 0
        for key, (mode, ash, ast, bsh, bst, cst, cdt, dev) in list(_PENDING.items()):
            a = torch.empty_strided(ash, ast, dtype=torch.bfloat16, device=dev).normal_()
            b = torch.empty_strided(bsh, bst, dtype=torch.bfloat16, device=dev).normal_()
            c = torch.empty_strided((ash[0], bsh[1]), cst, dtype=cdt, device=dev)
            _TUNED[key] = _tune(a, b, c, mode)
            del _PENDING[key]
            n += 1
        return n


CAPTURE_TUNER = _CaptureTuner()


def linear_fwd(x2d: Tensor, w: Tensor) -> Tensor:
    """``x2d @ w^T``: [M, K] x [N, K] -> [M, N] bf16."""
    y = torch.empty(x2d.shape[0], w.shape[0], dtype=torch.bfloat16, device=x2d.device)
    if x2d.shape[0]:
        tuned_mm(x2d, w.t(), y)
    return y


def linear_dgrad(dy2d: Tensor, w: Tensor) -> Tensor:
    """``dy2d @ w``: [M, N] x [N, K] -> [M, K] bf16."""
    dx = torch.empty(dy2d.shape[0], w.shape[1], dtype=torch.bfloat16, device=dy2d.device)
    if dy2d.shape[0]:
        tuned_mm(dy2d, w, dx)
    return dx


def linear_wgrad(dy2d: Tensor, x2d: Tensor, buf: Tensor, add: bool) -> None:
    """``buf (+)= bf16(dy2d^T @ x2d)``: [N, M] x [M, K] -> [N, K]. ``add`` False writes (the
    first read of a backward), True adds. An empty reduction (M = 0) writes zeros or adds
    nothing."""
    if dy2d.shape[0] == 0:
        if not add:
            buf.zero_()
        return
    tuned_mm(dy2d.t(), x2d, buf, MODE_ADD if add else MODE_FIRST)
