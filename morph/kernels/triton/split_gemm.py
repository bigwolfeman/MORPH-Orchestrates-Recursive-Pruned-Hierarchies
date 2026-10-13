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

THE SWIGLU EPILOGUES (``model.gemm_epilogue_fusion``). The MLP's elementwise work moves into two
of these GEMMs, with the tile grid kept (one program per output tile, K walked in the same
order). ``swiglu_fwd``: the ``gate_up`` GEMM computes the gate tile and the up tile of the SAME
hidden columns in two accumulators, writes both pre-activations (the backward needs them) and
``h = silu(gate) * up``. ``swiglu_bwd``: the ``down`` data-grad GEMM ``dh = dy @ W_down`` reads the
saved gate and up tiles in its epilogue and writes ``d_gate`` and ``d_up`` (the SwiGLU backward),
so ``dh`` never reaches memory. Both round the accumulator to bf16 first (the GEMM output's
rounding) and compute in fp32 from there, as the inductor kernels they replace do.

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
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["CONFIGS", "SWIGLU_FWD_CONFIGS", "MODE_PLAIN", "MODE_FIRST", "MODE_ADD",
           "MODE_SWIGLU_FWD", "MODE_SWIGLU_BWD", "tuned_mm", "tuned_swiglu", "linear_fwd",
           "linear_dgrad", "linear_wgrad", "swiglu_fwd", "swiglu_bwd", "swiglu_fwd_torch",
           "swiglu_bwd_torch", "tuned_keys", "clear_tuned", "CAPTURE_TUNER"]

MODE_PLAIN, MODE_FIRST, MODE_ADD = 0, 1, 2
MODE_SWIGLU_FWD, MODE_SWIGLU_BWD = 3, 4

# (BM, BN, BK, num_warps, num_stages): the list of the isolated measurement (triton_gemm_bench.py)
CONFIGS: tuple[tuple[int, int, int, int, int], ...] = (
    (128, 128, 64, 8, 3), (128, 128, 32, 4, 4), (128, 128, 64, 4, 3), (64, 128, 64, 4, 4),
    (128, 64, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 32, 4, 5), (128, 256, 32, 8, 3),
    (256, 128, 32, 8, 3), (64, 256, 32, 4, 4), (32, 128, 64, 4, 4),
)
# model.gemm_epilogue_fusion, forward: two accumulators per tile (gate and up), so the tiles are
# smaller than the plain list's. Shared memory per stage (BM*BK + 2*BK*BN) * 2 bytes, at most
# ~80 KB (the card allows 99 KB per block).
SWIGLU_FWD_CONFIGS: tuple[tuple[int, int, int, int, int], ...] = (
    (64, 64, 64, 4, 3), (64, 64, 32, 4, 4), (128, 64, 32, 4, 4), (128, 64, 64, 8, 2),
    (64, 128, 32, 4, 4), (128, 128, 32, 8, 3), (32, 128, 32, 4, 4), (64, 32, 64, 4, 4),
    (256, 64, 32, 8, 3), (64, 64, 32, 4, 5),
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


@triton.jit
def _swiglu_fwd_kernel(a, b, gu, h, M, DFF, K, sam, sak, sbk, sbn, sgm, shm,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    """``gu = a @ b`` ([M, 2*DFF]: gate columns, then up columns) and ``h = silu(gate) * up``
    ([M, DFF]). One program owns the gate tile and the up tile of the same DFF columns."""
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(DFF, BN)
    width = GM * npn
    first = (pid // width) * GM
    gsz = min(npm - first, GM)
    pm = first + (pid % width) % gsz
    pn = (pid % width) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a + rm[:, None] * sam + rk[None, :] * sak
    bgp = b + rk[:, None] * sbk + rn[None, :] * sbn
    bup = bgp + DFF * sbn
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        km = rk < K - k * BK
        av = tl.load(ap, mask=(rm[:, None] < M) & km[None, :], other=0.0)
        bm = km[:, None] & (rn[None, :] < DFF)
        acc_g = tl.dot(av, tl.load(bgp, mask=bm, other=0.0), acc_g)
        acc_u = tl.dot(av, tl.load(bup, mask=bm, other=0.0), acc_u)
        ap += BK * sak
        bgp += BK * sbk
        bup += BK * sbk
    cm = (rm[:, None] < M) & (rn[None, :] < DFF)
    g16 = acc_g.to(tl.bfloat16)
    u16 = acc_u.to(tl.bfloat16)
    gp = gu + rm[:, None] * sgm + rn[None, :]
    tl.store(gp, g16, mask=cm)
    tl.store(gp + DFF, u16, mask=cm)
    gf = g16.to(tl.float32)
    uf = u16.to(tl.float32)
    hv = gf * tl.sigmoid(gf) * uf
    tl.store(h + rm[:, None] * shm + rn[None, :], hv.to(tl.bfloat16), mask=cm)


@triton.jit
def _swiglu_bwd_kernel(a, b, gu, dgu, M, DFF, K, sam, sak, sbk, sbn, sgm, sdm,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    """``dh = a @ b`` ([M, DFF]) then, in the epilogue, the SwiGLU backward against the saved
    pre-activations ``gu`` ([M, 2*DFF]): ``d_gate = dh * up * silu'(gate)``,
    ``d_up = dh * silu(gate)``, written to ``dgu`` in ``gu``'s layout."""
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(DFF, BN)
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
        bv = tl.load(bp, mask=km[:, None] & (rn[None, :] < DFF), other=0.0)
        acc = tl.dot(av, bv, acc)
        ap += BK * sak
        bp += BK * sbk
    cm = (rm[:, None] < M) & (rn[None, :] < DFF)
    gp = gu + rm[:, None] * sgm + rn[None, :]
    gf = tl.load(gp, mask=cm, other=0.0).to(tl.float32)
    uf = tl.load(gp + DFF, mask=cm, other=0.0).to(tl.float32)
    dh = acc.to(tl.bfloat16).to(tl.float32)
    sig = tl.sigmoid(gf)
    dgate = dh * uf * (sig * (1.0 + gf * (1.0 - sig)))
    dup = dh * (gf * sig)
    dp = dgu + rm[:, None] * sdm + rn[None, :]
    tl.store(dp, dgate.to(tl.bfloat16), mask=cm)
    tl.store(dp + DFF, dup.to(tl.bfloat16), mask=cm)


_TUNED: dict[tuple, tuple[int, int, int, int, int]] = {}


def tuned_keys() -> list[tuple]:
    return list(_TUNED)


def clear_tuned() -> None:
    _TUNED.clear()
    _PENDING.clear()


def _configs(mode: int) -> tuple:
    return SWIGLU_FWD_CONFIGS if mode == MODE_SWIGLU_FWD else CONFIGS


def _launch(a: Tensor, b: Tensor, c: Tensor, mode: int, cfg: tuple, aux: Tensor | None = None
            ) -> None:
    """One launch. ``aux`` is the second tensor of a SwiGLU epilogue: the hidden ``h`` it writes
    (forward; ``c`` is ``gu``) or the saved ``gu`` it reads (backward; ``c`` is ``d_gu``)."""
    M, K = a.shape
    N = b.shape[1]
    bm, bn, bk, nw, ns = cfg
    if mode == MODE_SWIGLU_FWD:
        dff = N // 2
        grid = (triton.cdiv(M, bm) * triton.cdiv(dff, bn),)
        _swiglu_fwd_kernel[grid](a, b, c, aux, M, dff, K, a.stride(0), a.stride(1), b.stride(0),
                                 b.stride(1), c.stride(0), aux.stride(0), BM=bm, BN=bn, BK=bk,
                                 GM=_GM, num_warps=nw, num_stages=ns)
        return
    if mode == MODE_SWIGLU_BWD:
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        _swiglu_bwd_kernel[grid](a, b, aux, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                 b.stride(1), aux.stride(0), c.stride(0), BM=bm, BN=bn, BK=bk,
                                 GM=_GM, num_warps=nw, num_stages=ns)
        return
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _mm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1), BM=bm, BN=bn, BK=bk, GM=_GM, MODE=mode,
                     num_warps=nw, num_stages=ns)


def _tune(a: Tensor, b: Tensor, c: Tensor, mode: int, aux: Tensor | None = None) -> tuple:
    """Time every config on throwaway output (``c`` and a written ``aux`` are never touched)
    and return the fastest."""
    scratch = torch.empty_like(c)
    wgrad = mode in (MODE_FIRST, MODE_ADD)
    if wgrad:
        scratch.zero_()
    aux_s = torch.empty_like(aux) if mode == MODE_SWIGLU_FWD else aux
    best, best_t = None, float("inf")
    for cfg in _configs(mode):
        _launch(a, b, scratch, mode, cfg, aux_s)           # compile + warm
        ts = []
        for _ in range(_TUNE_REPS):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            _launch(a, b, scratch, mode, cfg, aux_s)
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        if min(ts) < best_t:
            best, best_t = cfg, min(ts)
    if wgrad:                                              # the other wgrad mode never compiles later
        for m in (MODE_FIRST, MODE_ADD):
            _launch(a, b, scratch, m, best)
    torch.cuda.synchronize()
    return best


def swiglu_fwd_torch(gu: Tensor) -> Tensor:
    """``silu(gate) * up`` of ``gu`` ([..., 2*DFF], bf16): fp32 inside, one rounding to bf16 (the
    inductor kernel's arithmetic, and the forward epilogue's)."""
    gate, up = gu.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(torch.bfloat16)


def swiglu_bwd_torch(dh: Tensor, gu: Tensor) -> Tensor:
    """``d_gu = [d_gate, d_up]`` for ``h = silu(gate) * up`` (the backward epilogue's arithmetic)."""
    gate, up = gu.float().chunk(2, dim=-1)
    dhf = dh.float()
    sig = torch.sigmoid(gate)
    return torch.cat([dhf * up * (sig * (1.0 + gate * (1.0 - sig))), dhf * (gate * sig)],
                     dim=-1).to(torch.bfloat16)


def _cublas_equivalent(a: Tensor, b: Tensor, c: Tensor, mode: int, aux: Tensor | None) -> None:
    """The cuBLAS form of each mode (same rounding rule), for a recording that only collects."""
    if mode == MODE_PLAIN:
        torch.mm(a, b, out=c)
    elif mode == MODE_FIRST:
        c.copy_(torch.mm(a, b))
    elif mode == MODE_ADD:
        c.add_(torch.mm(a, b))
    elif mode == MODE_SWIGLU_FWD:
        gu = torch.mm(a, b)
        c.copy_(gu)
        aux.copy_(swiglu_fwd_torch(gu))
    else:
        c.copy_(swiglu_bwd_torch(torch.mm(a, b), aux))


def _run(a: Tensor, b: Tensor, c: Tensor, mode: int, aux: Tensor | None) -> None:
    # the first-read / add modes share one tuning (and one winner)
    kmode = MODE_FIRST if mode in (MODE_FIRST, MODE_ADD) else mode
    key = (kmode, a.shape[0], b.shape[1], a.shape[1], a.stride(), b.stride(), c.stride(), c.dtype,
           a.device.index) + (() if aux is None else (aux.stride(),))
    cfg = _TUNED.get(key)
    if cfg is None:
        if torch.cuda.is_current_stream_capturing():
            if not _COLLECT:
                raise RuntimeError(
                    f"split_wgrad_triton: GEMM (mode {mode}, M={a.shape[0]}, N={b.shape[1]}, "
                    f"K={a.shape[1]}, strides {a.stride()} {b.stride()} {c.stride()}) was not "
                    f"tuned in the eager warm-up; it cannot be tuned inside a CUDA graph capture")
            _PENDING[key] = (kmode, tuple(a.shape), a.stride(), tuple(b.shape), b.stride(),
                             c.stride(), c.dtype, a.device,
                             None if aux is None else aux.stride())
            _cublas_equivalent(a, b, c, mode, aux)
            return
        cfg = _TUNED[key] = _tune(a, b, c, kmode, aux)
    _launch(a, b, c, mode, cfg, aux)


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
    if mode in (MODE_SWIGLU_FWD, MODE_SWIGLU_BWD):
        raise ValueError("split_wgrad_triton: the SwiGLU epilogues go through tuned_swiglu")
    _run(a, b, c, mode, None)


def tuned_swiglu(a: Tensor, b: Tensor, c: Tensor, aux: Tensor, mode: int) -> None:
    """The SwiGLU epilogue GEMMs (module notes), tuned like ``tuned_mm``.

    ``MODE_SWIGLU_FWD``: ``a`` [M, K] x ``b`` [K, 2*DFF] -> ``c`` = gu [M, 2*DFF] and ``aux`` = h
    [M, DFF] (both written). ``MODE_SWIGLU_BWD``: ``a`` = dy [M, K] x ``b`` = W_down [K, DFF] ->
    ``c`` = d_gu [M, 2*DFF] (written), ``aux`` = the saved gu [M, 2*DFF] (read)."""
    if mode not in (MODE_SWIGLU_FWD, MODE_SWIGLU_BWD):
        raise ValueError(f"split_wgrad_triton: tuned_swiglu takes a SwiGLU mode, got {mode}")
    for t in (a, b, c, aux):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"split_wgrad_triton: SwiGLU GEMM tensors must be bf16, got {t.dtype}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"split_wgrad_triton: shapes {tuple(a.shape)} x {tuple(b.shape)}")
    m, n = a.shape[0], b.shape[1]
    two_dff = n if mode == MODE_SWIGLU_FWD else 2 * n
    want_aux = (m, two_dff // 2) if mode == MODE_SWIGLU_FWD else (m, two_dff)
    if two_dff % 2 or tuple(c.shape) != (m, two_dff) or tuple(aux.shape) != want_aux:
        raise ValueError(f"split_wgrad_triton: SwiGLU shapes a {tuple(a.shape)} b {tuple(b.shape)} "
                         f"c {tuple(c.shape)} aux {tuple(aux.shape)}")
    if c.stride(1) != 1 or aux.stride(1) != 1:
        raise ValueError("split_wgrad_triton: the SwiGLU epilogue tensors need unit column stride")
    _run(a, b, c, mode, aux)


_COLLECT = False
# key -> (mode, a shape, a strides, b shape, b strides, c strides, c dtype, device, aux strides):
# all that a tuning reads (shapes, strides, dtype and mode, never values)
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
        for key, (mode, ash, ast, bsh, bst, cst, cdt, dev, aust) in list(_PENDING.items()):
            a = torch.empty_strided(ash, ast, dtype=torch.bfloat16, device=dev).normal_()
            b = torch.empty_strided(bsh, bst, dtype=torch.bfloat16, device=dev).normal_()
            aux = None
            if mode == MODE_SWIGLU_FWD:
                c = torch.empty_strided((ash[0], bsh[1]), cst, dtype=cdt, device=dev)
                aux = torch.empty_strided((ash[0], bsh[1] // 2), aust, dtype=torch.bfloat16,
                                          device=dev)
            elif mode == MODE_SWIGLU_BWD:
                c = torch.empty_strided((ash[0], 2 * bsh[1]), cst, dtype=cdt, device=dev)
                aux = torch.empty_strided((ash[0], 2 * bsh[1]), aust, dtype=torch.bfloat16,
                                          device=dev).normal_()
            else:
                c = torch.empty_strided((ash[0], bsh[1]), cst, dtype=cdt, device=dev)
            _TUNED[key] = _tune(a, b, c, mode, aux)
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


def swiglu_fwd(x2d: Tensor, w_gu: Tensor) -> tuple[Tensor, Tensor]:
    """``(gu, h)``: ``gu = x2d @ w_gu^T`` ([M, 2*DFF]) and ``h = silu(gate) * up`` ([M, DFF]) in
    one kernel. ``w_gu`` is [2*DFF, K]."""
    m, dff = x2d.shape[0], w_gu.shape[0] // 2
    gu = torch.empty(m, 2 * dff, dtype=torch.bfloat16, device=x2d.device)
    h = torch.empty(m, dff, dtype=torch.bfloat16, device=x2d.device)
    if m:
        tuned_swiglu(x2d, w_gu.t(), gu, h, MODE_SWIGLU_FWD)
    return gu, h


def swiglu_bwd(dy2d: Tensor, w_dn: Tensor, gu: Tensor) -> Tensor:
    """``d_gu`` ([M, 2*DFF]): the SwiGLU backward applied to ``dh = dy2d @ w_dn`` ([M, DFF]) in
    the GEMM's epilogue. ``w_dn`` is [N_out, DFF]; ``gu`` is the forward's saved pre-activation."""
    dgu = torch.empty_like(gu)
    if dy2d.shape[0]:
        tuned_swiglu(dy2d, w_dn, dgu, gu, MODE_SWIGLU_BWD)
    return dgu


def linear_wgrad(dy2d: Tensor, x2d: Tensor, buf: Tensor, add: bool) -> None:
    """``buf (+)= bf16(dy2d^T @ x2d)``: [N, M] x [M, K] -> [N, K]. ``add`` False writes (the
    first read of a backward), True adds. An empty reduction (M = 0) writes zeros or adds
    nothing."""
    if dy2d.shape[0] == 0:
        if not add:
            buf.zero_()
        return
    tuned_mm(dy2d.t(), x2d, buf, MODE_ADD if add else MODE_FIRST)
