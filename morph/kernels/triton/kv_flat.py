"""``model.kv_flat_kernels``: the K/V side of a core pass, re-tiled for parallelism.

WHY. The core pass runs the CCA conv and the CCA prologue over EVERY cell (batch 6 x 256 cells =
1,536 rows), however few cells are active. Their backward kernels were launched with one program
per (batch, channel group, 128-position tile): 144 programs of 256 threads for the grouped conv
weight grad, on 170 SMs, each running four taps in series and writing a per-tile fp32 partial slab
that a separate torch reduce kernel summed. The prologue rows backward used 16 rows per program
(96 programs at 1,536 rows). Measured on the 5090 at those shapes the kernels are latency bound,
not byte bound: the whole input is 2.4 MB per tensor and sits in L2.

WHAT CHANGES. Same math, same fp32 accumulation, a different split of the position reduction.

* ``gp_dx`` and ``gp_dw`` of the grouped conv run in ONE launch (two roles split by program id),
  and the weight grad gets one program per (group, tap, position chunk): a program loops over its
  chunk's tiles with the ``[CG, CG]`` accumulator held in registers, then writes ONE partial.
* The depthwise backward tiles ``CB`` channels x one position tile per program (was one channel).
* The partials of every weight grad are summed by ONE Triton kernel, in a fixed order, and
  written in the weight's dtype (was: torch ``sum`` kernels plus cast kernels). No atomics.
* The prologue rows backward uses fewer rows per program; its three partial vectors are summed
  by the same fixed-order kernel.

The sums run in a different order than the in-tree kernels (more, smaller partials), so the
results are not bit-identical to the key-off path; ``tests/test_kv_flat.py`` scores both against
an fp64 reference.
"""
from __future__ import annotations

from ._pdl import pdl_opts

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    # -----------------------------------------------------------------------
    # Grouped conv backward: d_input and d_weight in ONE launch.
    #   pid <  n_dx : the d_input tile, (b, group, BT_DX-tile)   (== _gp_bwd_dx_kernel)
    #   pid >= n_dx : the d_weight chunk, (group, chunk, tap)
    # dW[co, ci, j] = sum_t go[ch0+co, t] * x[ch0+ci, t-p+j]; the chunk's tiles are visited in
    # ascending order and accumulate in one fp32 [CG, CG] register tile.
    # part layout [n_chunks, K, C, CG] fp32 (rows contiguous along ci).
    # -----------------------------------------------------------------------
    @triton.jit
    def _kvf_gp_bwd_kernel(
        go_ptr, w_ptr, x_ptr, dx_ptr, part_ptr,
        B, C, S, n_dx, n_chunks, tiles_per_chunk,
        CG: tl.constexpr, K: tl.constexpr, BT_DX: tl.constexpr, BT_W: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
        pid = tl.program_id(0)
        G = C // CG
        p = K - 1
        co = tl.arange(0, CG)
        ci = tl.arange(0, CG)
        if pid < n_dx:
            n_tt_x = (S + BT_DX - 1) // BT_DX
            t_tile_x = pid % n_tt_x
            bg = pid // n_tt_x
            gx = bg % G
            bx = bg // G
            ch0_x = gx * CG
            Tx = t_tile_x * BT_DX + tl.arange(0, BT_DX)
            tmask_x = Tx < S
            acc_x = tl.zeros((CG, BT_DX), dtype=tl.float32)
            for j in tl.static_range(K):
                w_off = (ch0_x + co)[None, :] * (CG * K) + ci[:, None] * K + j
                WjT = tl.load(w_ptr + w_off)
                gsrc = Tx + p - j
                gmask = (gsrc >= 0) & (gsrc < S) & tmask_x[None, :]
                g_off = (bx * C + ch0_x + co)[:, None] * S + gsrc[None, :]
                Gj = tl.load(go_ptr + g_off, mask=gmask, other=0.0)
                acc_x = tl.dot(WjT, Gj, acc_x)
            dx_off = (bx * C + ch0_x + ci)[:, None] * S + Tx[None, :]
            tl.store(dx_ptr + dx_off, acc_x.to(dx_ptr.dtype.element_ty), mask=tmask_x[None, :])
        else:
            r = pid - n_dx
            jw = r % K
            r = r // K
            c = r % n_chunks
            g = r // n_chunks
            ch0 = g * CG
            n_tt = (S + BT_W - 1) // BT_W
            n_tiles = B * n_tt
            u0 = c * tiles_per_chunk
            u1 = tl.minimum(u0 + tiles_per_chunk, n_tiles)
            acc_w = tl.zeros((CG, CG), dtype=tl.float32)
            for u in range(u0, u1):
                b = u // n_tt
                Tw = (u % n_tt) * BT_W + tl.arange(0, BT_W)
                tmask_w = Tw < S
                go_off = (b * C + ch0 + co)[:, None] * S + Tw[None, :]
                Go = tl.load(go_ptr + go_off, mask=tmask_w[None, :], other=0.0)
                src = Tw - p + jw
                xmask = (src >= 0) & (src < S) & tmask_w[None, :]
                x_off = (b * C + ch0 + ci)[:, None] * S + src[None, :]
                Xj = tl.load(x_ptr + x_off, mask=xmask, other=0.0)
                acc_w = tl.dot(Go, tl.trans(Xj), acc_w)
            o_off = ((c * K + jw) * C + ch0 + co)[:, None] * CG + ci[None, :]
            tl.store(part_ptr + o_off, acc_w)

    # -----------------------------------------------------------------------
    # Depthwise backward: d_input and the d_weight partials. One program = (b, CB channels,
    # position tile). part layout [B * n_tt, K, C] fp32.
    # -----------------------------------------------------------------------
    @triton.jit
    def _kvf_dw_bwd_kernel(
        go_ptr, w_ptr, x_ptr, dx_ptr, part_ptr,
        B, C, S,
        K: tl.constexpr, BT: tl.constexpr, CB: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
        pid = tl.program_id(0)
        n_tt = (S + BT - 1) // BT
        n_cb = C // CB
        t_tile = pid % n_tt
        r = pid // n_tt
        cb = r % n_cb
        b = r // n_cb
        p = K - 1
        cc = cb * CB + tl.arange(0, CB)
        t = t_tile * BT + tl.arange(0, BT)
        tmask = t < S
        row = ((b * C + cc) * S)[:, None]
        gv_cur = tl.load(go_ptr + row + t[None, :], mask=tmask[None, :], other=0.0).to(tl.float32)
        slab = b * n_tt + t_tile
        dx_acc = tl.zeros((CB, BT), dtype=tl.float32)
        for j in tl.static_range(K):
            gsrc = t + p - j
            gmask = tmask & (gsrc >= 0) & (gsrc < S)
            gv_sh = tl.load(go_ptr + row + gsrc[None, :], mask=gmask[None, :],
                            other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + cc * K + j).to(tl.float32)
            dx_acc += wv[:, None] * gv_sh
            src = t - p + j
            smask = tmask & (src >= 0) & (src < S)
            xv = tl.load(x_ptr + row + src[None, :], mask=smask[None, :], other=0.0).to(tl.float32)
            tl.store(part_ptr + (slab * K + j) * C + cc, tl.sum(gv_cur * xv, axis=1))
        tl.store(dx_ptr + row + t[None, :], dx_acc.to(dx_ptr.dtype.element_ty), mask=tmask[None, :])

    # -----------------------------------------------------------------------
    # Fixed-order partial sums, written in the output dtype. One launch sums up to three
    # buffers: part_i is [n_i, M_i] fp32 row-major, out_i is [M_i] in the output dtype. The
    # partial's flat column e is laid out [K_i, M_i // K_i]; the output is laid out
    # [M_i // K_i, K_i] (the conv weights keep the tap axis last), so the store moves e to
    # (e % inner) * K_i + e // inner. K_i == 1 is the identity.
    # -----------------------------------------------------------------------
    @triton.jit
    def _kvf_sum_one(part, out, M, n, K, e, m, NB: tl.constexpr):
        # A [NB, BLK] tile per step: n up to a few hundred rows must not be a serial chain of
        # loads. The order is fixed (rows in blocks of NB, ascending, then the tile's axis 0).
        BLK: tl.constexpr = e.shape[0]
        rows = tl.arange(0, NB)
        acc = tl.zeros((NB, BLK), dtype=tl.float32)
        for u0 in range(0, n, NB):
            u = u0 + rows
            acc += tl.load(part + u[:, None] * M + e[None, :],
                           mask=(u < n)[:, None] & m[None, :], other=0.0)
        tot = tl.sum(acc, axis=0)
        inner = M // K
        tl.store(out + (e % inner) * K + e // inner, tot.to(out.dtype.element_ty), mask=m)

    @triton.jit
    def _kvf_sum_kernel(
        p0, o0, M0, n0, K0, nb0,
        p1, o1, M1, n1, K1, nb1,
        p2, o2, M2, n2, K2,
        BLK: tl.constexpr, NB: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
        pid = tl.program_id(0)
        offs = tl.arange(0, BLK)
        if pid < nb0:
            e = pid * BLK + offs
            _kvf_sum_one(p0, o0, M0, n0, K0, e, e < M0, NB)
        elif pid < nb0 + nb1:
            e = (pid - nb0) * BLK + offs
            _kvf_sum_one(p1, o1, M1, n1, K1, e, e < M1, NB)
        else:
            e = (pid - nb0 - nb1) * BLK + offs
            _kvf_sum_one(p2, o2, M2, n2, K2, e, e < M2, NB)


# ===========================================================================
# Wrappers
# ===========================================================================

_SUM_BLK = 128
_SUM_NB = 16


def _sum_partials(bufs: list[tuple[Tensor, Tensor, int]]) -> None:
    """Sum each ``(part [n, M], out [M], K)`` over ``n`` in ascending order into ``out``, one
    launch. ``out`` may be 1-3 entries; missing ones are skipped by a zero block count."""
    assert 1 <= len(bufs) <= 3
    args, nbs = [], []
    for part, out, K in bufs:
        n, M = part.shape
        assert part.dtype == torch.float32 and part.is_contiguous() and out.is_contiguous()
        assert out.numel() == M and M % K == 0
        nb = triton.cdiv(M, _SUM_BLK)
        args += [part, out, M, n, K]
        nbs.append(nb)
    dummy = bufs[0]
    while len(args) < 15:
        args += [dummy[0], dummy[1], dummy[0].shape[1], 0, 1]
    _kvf_sum_kernel[(sum(nbs),)](
        args[0], args[1], args[2], args[3], args[4], nbs[0],
        args[5], args[6], args[7], args[8], args[9], nbs[1] if len(nbs) > 1 else 0,
        args[10], args[11], args[12], args[13], args[14],
        BLK=_SUM_BLK, NB=_SUM_NB, num_warps=4, **pdl_opts(),
    )


# Tile config, from the microbench in lab/perf/bench_kv_flat.py at B=6, C=768, S=256, CG=64.
_GP_BT_DX = 128
_GP_BT_W = 128
_GP_TILES_PER_CHUNK = 3
_GP_LAUNCH = dict(num_stages=2, num_warps=8)
_DW_BT = 256
_DW_CB = 4
_DW_LAUNCH = dict(num_stages=1, num_warps=4)


def cca_conv_bwd_flat(grad_out: Tensor, x: Tensor, y_dw: Tensor, w_dw2: Tensor, w_gp: Tensor,
                      CG: int, K: int) -> tuple[Tensor, Tensor, Tensor]:
    """``fused_cca_conv._cca_conv_bwd_impl`` on the re-tiled kernels: ``(dx, dw_dw [C, K],
    dw_gp [C, CG, K])``, each in its input's dtype. Three launches (was seven plus casts)."""
    B, C, S = grad_out.shape
    assert C % CG == 0 and C % _DW_CB == 0
    grad_out = grad_out.contiguous()
    G = C // CG
    dev = grad_out.device
    n_tt_x = triton.cdiv(S, _GP_BT_DX)
    n_dx = B * G * n_tt_x
    n_tiles_w = B * triton.cdiv(S, _GP_BT_W)
    n_chunks = triton.cdiv(n_tiles_w, _GP_TILES_PER_CHUNK)
    d_ydw = torch.empty_like(grad_out)
    part_gp = torch.empty(n_chunks, K * C * CG, device=dev, dtype=torch.float32)
    _kvf_gp_bwd_kernel[(n_dx + G * n_chunks * K,)](
        grad_out, w_gp, y_dw, d_ydw, part_gp, B, C, S, n_dx, n_chunks, _GP_TILES_PER_CHUNK,
        CG=CG, K=K, BT_DX=_GP_BT_DX, BT_W=_GP_BT_W, **_GP_LAUNCH, **pdl_opts(),
    )
    n_tt = triton.cdiv(S, _DW_BT)
    dx = torch.empty_like(grad_out)
    part_dw = torch.empty(B * n_tt, K * C, device=dev, dtype=torch.float32)
    _kvf_dw_bwd_kernel[(B * (C // _DW_CB) * n_tt,)](
        d_ydw, w_dw2, x, dx, part_dw, B, C, S, K=K, BT=_DW_BT, CB=_DW_CB, **_DW_LAUNCH, **pdl_opts(),
    )
    dw_gp = torch.empty_like(w_gp)
    dw_dw = torch.empty_like(w_dw2)
    _sum_partials([(part_gp, dw_gp, K), (part_dw, dw_dw, K)])
    return dx.to(x.dtype), dw_dw, dw_gp
