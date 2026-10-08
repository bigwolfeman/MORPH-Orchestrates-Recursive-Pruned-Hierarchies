"""Fused strict-geometry TG attention — Triton forward AND backward, both branches in one pass.

``model.tg_fused_attention`` (2026-10-07, graph-step task 2.1). Under ``tul.tg_restrict`` +
``tul.tg_geometry: strict`` every ``MORPHAttention`` call runs two attention branches over the
SAME per-position CCA tensors q, k, v ``[B, H, S, D]`` (attention.py `_CCACSAAttention.forward`):

  out_comp = _tg_slot_attention(...)   softmax over the comp relation plus a per-head SINK
                                       logit whose value is the zero vector
  out_win  = _window_fallback(...)     softmax over the window relation, no sink; a row
                                       with no allowed key reads exactly 0

This module computes both from ONE score tile ``s = q k^T * scale`` per (query tile, key
tile), with the relation evaluated in the kernel from index tensors instead of a dense
``[B, 1, S, S]`` mask. The relations, ``i`` the query position and ``j`` the key position:

  MODE_PRELUDE (``tg_strict_allow(layout, "prelude")``)
      base(i, j) = j <= i  AND  bag[i] == bag[j]
  MODE_CODA    (``tg_strict_allow(layout, "coda", coda_prefix_reach="all")``)
      base(i, j) = j == i                                          if slot[i]
                 = j <= i AND (bag[i] == bag[j] OR (slot[j] AND bag[j] < bag[i]))   else
  MODE_CELLS   (``slot_cell_relation(n_slots, cells)[0]``, the Thought Register at reach 0)
      base(i, j) = j // cells <= i // cells

  window(i, j) = base AND i != j AND |i - j| < window        (XSA + the sliding window)
  comp(i, j)   = base AND slot[j]       (prelude / coda: the dense slot-column form,
                                         ``dense_slot_cols``)
               = base                   (cells: every position is a cell)

Nothing else is supported; the model refuses every other geometry at build
(``MORPHTransformer`` under ``model.tg_fused_attention``).

The sink enters as the comp branch's initial online-softmax state (max = sink logit,
normaliser 1, accumulator 0): it is a key every row sees, with a zero value. Its gradient is
``-sum_i p_sink(i) * (dO_i . O_i)``, reduced in torch (fixed order, deterministic).

NOT bit-identical to the eager path (stated, by mechanism): the eager comp branch scores the
dense row with an fp32 GEMM, softmaxes it in Inductor's reduction order, rounds the NORMALISED
weights to bf16 and multiplies them with cuBLAS; the window branch is the SDPA
memory-efficient kernel. This kernel runs a flash online softmax (tile order, unnormalised P
rounded to bf16 before the PV product), so it sums the same terms in a different order. Its
error against fp64 is pinned in tests/test_tg_strict_attention.py.

Precision: fp32 accumulation everywhere; P is rounded to the input dtype for the PV / P^T dO
products (the flash convention); dS stays fp32 and its two products run on tf32 operands; the
backward's delta = rowsum(dO * O) reads an fp32 copy of each output written by the forward
(only when a backward will run). An fp32 forward (no autocast) runs full-precision dots,
libdevice exp2 and 32-wide tiles.

Deterministic: no atomics. dK/dV and dQ are two kernels, each output written by one program.
Fully masked tiles: MODE_PRELUDE skips a (query tile, key tile) pair whose bag ranges do not
overlap (per-tile min/max, correct for any bag order); every mode skips tiles past the
causal / cell-block bound.

Registered as ``torch.library.custom_op``s, so a compiled block traces through the call
without a graph break and a CUDA graph captures it (fixed shapes, no host reads).
"""
from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False

MODE_PRELUDE, MODE_CODA, MODE_CELLS = 0, 1, 2
_MODES = {"prelude": MODE_PRELUDE, "coda": MODE_CODA, "cells": MODE_CELLS}
# query tile == key tile (the bag min/max tables share it); fp32 tiles at 64 overflow sm_120's
# 99 KB of shared memory in the dK/dV kernel, so an fp32 forward runs 32-wide tiles
def _tile(dtype: torch.dtype) -> int:
    return 32 if dtype == torch.float32 else 64
_LOG2E = 1.4426950408889634

if TRITON_AVAILABLE:

    @triton.jit
    def _exp2(x, PREC: tl.constexpr):
        """ex2.approx for the 16-bit model; the accurate libdevice exp2 for an fp32 forward."""
        if PREC == "ieee":
            return libdevice.exp2(x)
        else:
            return tl.exp2(x)

    @triton.jit
    def _log2(x, PREC: tl.constexpr):
        if PREC == "ieee":
            return libdevice.log2(x)
        else:
            return tl.log2(x)

    @triton.jit
    def _tile_masks(offs_m, offs_n, mrow, ncol, bag_i, slot_i, bag_j, slot_j, W,
                    MODE: tl.constexpr, CELLS: tl.constexpr):
        """(window, comp) bool [BM, BN] for one (query tile, key tile)."""
        valid = mrow[:, None] & ncol[None, :]
        diff = offs_m[:, None] - offs_n[None, :]
        if MODE == 2:
            base = (offs_n[None, :] // CELLS) <= (offs_m[:, None] // CELLS)
        else:
            causal = diff >= 0
            same = bag_i[:, None] == bag_j[None, :]
            if MODE == 0:
                base = causal & same
            else:
                tok = same | ((slot_j[None, :] != 0) & (bag_j[None, :] < bag_i[:, None]))
                base = tl.where(slot_i[:, None] != 0, diff == 0, causal & tok)
        base = base & valid
        win = base & (diff != 0) & (diff < W) & (diff > -W)
        if MODE == 2:
            comp = base
        else:
            comp = base & (slot_j[None, :] != 0)
        return win, comp

    @triton.jit
    def _fwd_kernel(Q, K, V, SINK, BAG, SLOT, TMIN, TMAX, OC, OW, LC, LW, OC32, OW32,
                    H, S, NT, scale_log2, W,
                    MODE: tl.constexpr, CELLS: tl.constexpr, D: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, SAVE32: tl.constexpr,
                    PREC: tl.constexpr):
        qt = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        base = bh.to(tl.int64) * S * D
        offs_m = qt * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        mrow = offs_m < S
        q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None],
                    other=0.0)
        if MODE == 2:
            bag_i = offs_m
            slot_i = offs_m
            hi = tl.minimum(S, ((qt * BM + BM - 1) // CELLS + 1) * CELLS)
        else:
            bag_i = tl.load(BAG + b * S + offs_m, mask=mrow, other=-1)
            slot_i = tl.load(SLOT + b * S + offs_m, mask=mrow, other=0)
            hi = tl.minimum(S, qt * BM + BM)
        if MODE == 0:
            qmin = tl.load(TMIN + b * NT + qt)
            qmax = tl.load(TMAX + b * NT + qt)
        sink = tl.load(SINK + h).to(tl.float32) * 1.4426950408889634
        m_c = tl.zeros([BM], dtype=tl.float32) + sink
        l_c = tl.zeros([BM], dtype=tl.float32) + 1.0
        acc_c = tl.zeros([BM, D], dtype=tl.float32)
        m_w = tl.zeros([BM], dtype=tl.float32) - float("inf")
        l_w = tl.zeros([BM], dtype=tl.float32)
        acc_w = tl.zeros([BM, D], dtype=tl.float32)
        for start in range(0, hi, BN):
            run = True
            if MODE == 0:
                kmin = tl.load(TMIN + b * NT + start // BN)
                kmax = tl.load(TMAX + b * NT + start // BN)
                run = (kmin <= qmax) & (kmax >= qmin)
            if run:
                offs_n = start + tl.arange(0, BN)
                ncol = offs_n < S
                kv_ptr = base + offs_n[:, None] * D + offs_d[None, :]
                k = tl.load(K + kv_ptr, mask=ncol[:, None], other=0.0)
                v = tl.load(V + kv_ptr, mask=ncol[:, None], other=0.0)
                if MODE == 2:
                    bag_j = offs_n
                    slot_j = offs_n
                else:
                    bag_j = tl.load(BAG + b * S + offs_n, mask=ncol, other=-2)
                    slot_j = tl.load(SLOT + b * S + offs_n, mask=ncol, other=0)
                win, comp = _tile_masks(offs_m, offs_n, mrow, ncol, bag_i, slot_i,
                                        bag_j, slot_j, W, MODE, CELLS)
                s = tl.dot(q, tl.trans(k), input_precision=PREC) * scale_log2
                # comp: the sink started the state, so the running max is finite
                sc = tl.where(comp, s, float("-inf"))
                mc_new = tl.maximum(m_c, tl.max(sc, 1))
                alpha = _exp2(m_c - mc_new, PREC)
                p = _exp2(sc - mc_new[:, None], PREC)
                l_c = l_c * alpha + tl.sum(p, 1)
                acc_c = acc_c * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
                m_c = mc_new
                # window: the running max may still be -inf (no allowed key yet)
                sw = tl.where(win, s, float("-inf"))
                mw_new = tl.maximum(m_w, tl.max(sw, 1))
                mw_use = tl.where(mw_new == float("-inf"), 0.0, mw_new)
                alpha = _exp2(m_w - mw_use, PREC)
                p = _exp2(sw - mw_use[:, None], PREC)
                l_w = l_w * alpha + tl.sum(p, 1)
                acc_w = acc_w * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
                m_w = mw_new
        o_ptr = base + offs_m[:, None] * D + offs_d[None, :]
        o_c = acc_c / l_c[:, None]
        has_w = l_w > 0.0
        l_safe = tl.where(has_w, l_w, 1.0)
        o_w = tl.where(has_w[:, None], acc_w / l_safe[:, None], 0.0)
        tl.store(OC + o_ptr, o_c.to(OC.dtype.element_ty), mask=mrow[:, None])
        tl.store(OW + o_ptr, o_w.to(OW.dtype.element_ty), mask=mrow[:, None])
        # fp32 copies for the backward's delta = rowsum(dO * O): with the bf16 outputs its
        # rounding fed every dS of the row and put dq / dsink past the eager path's error
        if SAVE32:
            tl.store(OC32 + o_ptr, o_c, mask=mrow[:, None])
            tl.store(OW32 + o_ptr, o_w, mask=mrow[:, None])
        r_ptr = bh.to(tl.int64) * S + offs_m
        tl.store(LC + r_ptr, m_c + _log2(l_c, PREC), mask=mrow)
        # an empty window row stores +inf: exp2(s - inf) == 0 in the backward
        tl.store(LW + r_ptr, tl.where(has_w, m_w + _log2(l_safe, PREC), float("inf")),
                 mask=mrow)

    @triton.jit
    def _bwd_pre_kernel(OC, OW, DOC, DOW, LC, SINK, DC, DW, PSD, H, S,
                        D: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
        """delta_c / delta_w = rowsum(dO * O) and the sink's per-row gradient term."""
        qt = tl.program_id(0)
        bh = tl.program_id(1)
        h = bh % H
        offs_m = qt * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        mrow = offs_m < S
        ptr = bh.to(tl.int64) * S * D + offs_m[:, None] * D + offs_d[None, :]
        oc = tl.load(OC + ptr, mask=mrow[:, None], other=0.0)
        doc = tl.load(DOC + ptr, mask=mrow[:, None], other=0.0).to(tl.float32)
        ow = tl.load(OW + ptr, mask=mrow[:, None], other=0.0)
        dow = tl.load(DOW + ptr, mask=mrow[:, None], other=0.0).to(tl.float32)
        dc = tl.sum(oc * doc, 1)
        dw = tl.sum(ow * dow, 1)
        r_ptr = bh.to(tl.int64) * S + offs_m
        lc = tl.load(LC + r_ptr, mask=mrow, other=0.0)
        sink = tl.load(SINK + h).to(tl.float32) * 1.4426950408889634
        tl.store(DC + r_ptr, dc, mask=mrow)
        tl.store(DW + r_ptr, dw, mask=mrow)
        tl.store(PSD + r_ptr, -_exp2(sink - lc, PREC) * dc, mask=mrow)

    @triton.jit
    def _bwd_dkdv_kernel(Q, K, V, DOC, DOW, LC, LW, DC, DW, BAG, SLOT, TMIN, TMAX, DK, DV,
                         H, S, NT, scale, scale_log2, W,
                         MODE: tl.constexpr, CELLS: tl.constexpr, D: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
        kt = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        base = bh.to(tl.int64) * S * D
        rbase = bh.to(tl.int64) * S
        offs_n = kt * BN + tl.arange(0, BN)
        offs_d = tl.arange(0, D)
        ncol = offs_n < S
        kv_ptr = base + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(K + kv_ptr, mask=ncol[:, None], other=0.0)
        v = tl.load(V + kv_ptr, mask=ncol[:, None], other=0.0)
        if MODE == 2:
            bag_j = offs_n
            slot_j = offs_n
            lo = ((kt * BN) // CELLS) * CELLS
        else:
            bag_j = tl.load(BAG + b * S + offs_n, mask=ncol, other=-2)
            slot_j = tl.load(SLOT + b * S + offs_n, mask=ncol, other=0)
            lo = kt * BN
        if MODE == 0:
            kmin = tl.load(TMIN + b * NT + kt)
            kmax = tl.load(TMAX + b * NT + kt)
        lo = (lo // BM) * BM
        dk = tl.zeros([BN, D], dtype=tl.float32)
        dv = tl.zeros([BN, D], dtype=tl.float32)
        for start in range(lo, S, BM):
            run = True
            if MODE == 0:
                qmin = tl.load(TMIN + b * NT + start // BM)
                qmax = tl.load(TMAX + b * NT + start // BM)
                run = (kmin <= qmax) & (kmax >= qmin)
            if run:
                offs_m = start + tl.arange(0, BM)
                mrow = offs_m < S
                q_ptr = base + offs_m[:, None] * D + offs_d[None, :]
                q = tl.load(Q + q_ptr, mask=mrow[:, None], other=0.0)
                doc = tl.load(DOC + q_ptr, mask=mrow[:, None], other=0.0)
                dow = tl.load(DOW + q_ptr, mask=mrow[:, None], other=0.0)
                lc = tl.load(LC + rbase + offs_m, mask=mrow, other=0.0)
                lw = tl.load(LW + rbase + offs_m, mask=mrow, other=float("inf"))
                dc = tl.load(DC + rbase + offs_m, mask=mrow, other=0.0)
                dw = tl.load(DW + rbase + offs_m, mask=mrow, other=0.0)
                if MODE == 2:
                    bag_i = offs_m
                    slot_i = offs_m
                else:
                    bag_i = tl.load(BAG + b * S + offs_m, mask=mrow, other=-1)
                    slot_i = tl.load(SLOT + b * S + offs_m, mask=mrow, other=0)
                win, comp = _tile_masks(offs_m, offs_n, mrow, ncol, bag_i, slot_i,
                                        bag_j, slot_j, W, MODE, CELLS)
                s = tl.dot(q, tl.trans(k), input_precision=PREC) * scale_log2
                pc = tl.where(comp, _exp2(s - lc[:, None], PREC), 0.0)
                pw = tl.where(win, _exp2(s - lw[:, None], PREC), 0.0)
                dv += tl.dot(tl.trans(pc.to(doc.dtype)), doc, input_precision=PREC)
                dv += tl.dot(tl.trans(pw.to(dow.dtype)), dow, input_precision=PREC)
                dpc = tl.dot(doc, tl.trans(v), input_precision=PREC)
                dpw = tl.dot(dow, tl.trans(v), input_precision=PREC)
                ds = pc * (dpc - dc[:, None]) + pw * (dpw - dw[:, None])
                # ds stays fp32 (tf32 operands): rounding it to bf16 put the coda's dq
                # error past the eager path's, whose score gradient is an fp32 GEMM
                dk += tl.dot(tl.trans(ds), q.to(tl.float32), input_precision=PREC)
        tl.store(DK + kv_ptr, (dk * scale).to(DK.dtype.element_ty), mask=ncol[:, None])
        tl.store(DV + kv_ptr, dv.to(DV.dtype.element_ty), mask=ncol[:, None])

    @triton.jit
    def _bwd_dq_kernel(Q, K, V, DOC, DOW, LC, LW, DC, DW, BAG, SLOT, TMIN, TMAX, DQ,
                       H, S, NT, scale, scale_log2, W,
                       MODE: tl.constexpr, CELLS: tl.constexpr, D: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
        qt = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        base = bh.to(tl.int64) * S * D
        rbase = bh.to(tl.int64) * S
        offs_m = qt * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        mrow = offs_m < S
        q_ptr = base + offs_m[:, None] * D + offs_d[None, :]
        q = tl.load(Q + q_ptr, mask=mrow[:, None], other=0.0)
        doc = tl.load(DOC + q_ptr, mask=mrow[:, None], other=0.0)
        dow = tl.load(DOW + q_ptr, mask=mrow[:, None], other=0.0)
        lc = tl.load(LC + rbase + offs_m, mask=mrow, other=0.0)
        lw = tl.load(LW + rbase + offs_m, mask=mrow, other=float("inf"))
        dc = tl.load(DC + rbase + offs_m, mask=mrow, other=0.0)
        dw = tl.load(DW + rbase + offs_m, mask=mrow, other=0.0)
        if MODE == 2:
            bag_i = offs_m
            slot_i = offs_m
            hi = tl.minimum(S, ((qt * BM + BM - 1) // CELLS + 1) * CELLS)
        else:
            bag_i = tl.load(BAG + b * S + offs_m, mask=mrow, other=-1)
            slot_i = tl.load(SLOT + b * S + offs_m, mask=mrow, other=0)
            hi = tl.minimum(S, qt * BM + BM)
        if MODE == 0:
            qmin = tl.load(TMIN + b * NT + qt)
            qmax = tl.load(TMAX + b * NT + qt)
        dq = tl.zeros([BM, D], dtype=tl.float32)
        for start in range(0, hi, BN):
            run = True
            if MODE == 0:
                kmin = tl.load(TMIN + b * NT + start // BN)
                kmax = tl.load(TMAX + b * NT + start // BN)
                run = (kmin <= qmax) & (kmax >= qmin)
            if run:
                offs_n = start + tl.arange(0, BN)
                ncol = offs_n < S
                kv_ptr = base + offs_n[:, None] * D + offs_d[None, :]
                k = tl.load(K + kv_ptr, mask=ncol[:, None], other=0.0)
                v = tl.load(V + kv_ptr, mask=ncol[:, None], other=0.0)
                if MODE == 2:
                    bag_j = offs_n
                    slot_j = offs_n
                else:
                    bag_j = tl.load(BAG + b * S + offs_n, mask=ncol, other=-2)
                    slot_j = tl.load(SLOT + b * S + offs_n, mask=ncol, other=0)
                win, comp = _tile_masks(offs_m, offs_n, mrow, ncol, bag_i, slot_i,
                                        bag_j, slot_j, W, MODE, CELLS)
                s = tl.dot(q, tl.trans(k), input_precision=PREC) * scale_log2
                pc = tl.where(comp, _exp2(s - lc[:, None], PREC), 0.0)
                pw = tl.where(win, _exp2(s - lw[:, None], PREC), 0.0)
                dpc = tl.dot(doc, tl.trans(v), input_precision=PREC)
                dpw = tl.dot(dow, tl.trans(v), input_precision=PREC)
                ds = pc * (dpc - dc[:, None]) + pw * (dpw - dw[:, None])
                dq += tl.dot(ds, k.to(tl.float32), input_precision=PREC)
        tl.store(DQ + q_ptr, (dq * scale).to(DQ.dtype.element_ty), mask=mrow[:, None])


def _launch(dtype: torch.dtype) -> dict:
    """fp32 tiles do not fit two pipeline stages in sm_120's 99 KB of shared memory."""
    return dict(num_warps=4, num_stages=1 if dtype == torch.float32 else 2)


def _prec(dtype: torch.dtype) -> str:
    """tl.dot precision: tf32 for the fp32 dS products of a 16-bit model (the 16-bit dots
    ignore it); full fp32 ("ieee") when q/k/v themselves are fp32 (an fp32 forward)."""
    return "ieee" if dtype == torch.float32 else "tf32"


def _check(q: Tensor, k: Tensor, v: Tensor, mode: int, bag, slot, cells: int) -> None:
    if not TRITON_AVAILABLE or not q.is_cuda:
        raise RuntimeError("tg_strict_attention needs Triton and CUDA tensors "
                           "(model.tg_fused_attention has no CPU path)")
    if q.shape != k.shape or q.shape != v.shape or q.dim() != 4:
        raise ValueError(f"q/k/v must share one [B, H, S, D] shape, got "
                         f"{tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    if (q.dtype not in (torch.bfloat16, torch.float16, torch.float32) or k.dtype != q.dtype
            or v.dtype != q.dtype):
        raise TypeError(f"q/k/v must share one float dtype, got {q.dtype} {k.dtype} {v.dtype}")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError(f"d_head {q.shape[-1]} is not a supported tile width")
    if mode == MODE_CELLS:
        if cells < 1 or bag is not None or slot is not None:
            raise ValueError("MODE_CELLS takes cells >= 1 and no bag / slot tensors")
    elif mode in (MODE_PRELUDE, MODE_CODA):
        B, _, S, _ = q.shape
        if bag is None or slot is None or bag.shape != (B, S) or slot.shape != (B, S):
            raise ValueError(f"bag / slot must be [B, S] = {(B, S)}")
        if bag.dtype != torch.int32 or slot.dtype != torch.int32:
            raise TypeError("bag / slot must be int32 (tg_strict_index builds them)")
    else:
        raise ValueError(f"unknown mode {mode}")


def _tile_bounds(bag: Tensor | None, S: int, device, bt: int) -> tuple[Tensor, Tensor]:
    """Per-tile [min, max] of ``bag`` ([B, NT] int32 each) — the prelude's tile skip. Cells
    mode reads neither: a one-element device placeholder (no host copy under a capture)."""
    if bag is None:
        z = torch.zeros(1, dtype=torch.int32, device=device)
        return z, z
    B = bag.shape[0]
    nt = triton.cdiv(S, bt)
    pad = nt * bt - S
    lo = torch.nn.functional.pad(bag, (0, pad), value=torch.iinfo(torch.int32).max)
    hi = torch.nn.functional.pad(bag, (0, pad), value=torch.iinfo(torch.int32).min)
    return (lo.view(B, nt, bt).amin(-1).contiguous(),
            hi.view(B, nt, bt).amax(-1).contiguous())


@torch.library.custom_op("morph::tg_strict_attn_fwd", mutates_args=())
def _fwd_op(q: Tensor, k: Tensor, v: Tensor, sink: Tensor, bag: Tensor | None,
            slot: Tensor | None, mode: int, cells: int, window: int,
            scale: float, save32: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _check(q, k, v, mode, bag, slot, cells)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, S, D = q.shape
    oc, ow = torch.empty_like(q), torch.empty_like(q)
    lc = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    lw = torch.empty_like(lc)
    n32 = (B, H, S, D) if save32 else (0,)        # no backward -> no fp32 copies
    oc32 = torch.empty(n32, device=q.device, dtype=torch.float32)
    ow32 = torch.empty_like(oc32)
    bt = _tile(q.dtype)
    tmin, tmax = _tile_bounds(bag, S, q.device, bt)
    if bag is None:
        bag = slot = tmin
    nt = triton.cdiv(S, bt)
    _fwd_kernel[(nt, B * H)](q, k, v, sink, bag, slot, tmin, tmax, oc, ow, lc, lw, oc32, ow32,
                             H, S, nt, scale * _LOG2E, window,
                             MODE=mode, CELLS=max(cells, 1), D=D, BM=bt, BN=bt,
                             SAVE32=save32, PREC=_prec(q.dtype), **_launch(q.dtype))
    return oc, ow, lc, lw, oc32, ow32


@_fwd_op.register_fake
def _fwd_fake(q, k, v, sink, bag, slot, mode, cells, window, scale, save32):
    B, H, S, D = q.shape
    n32 = (B, H, S, D) if save32 else (0,)
    return (torch.empty_like(q), torch.empty_like(q),
            q.new_empty(B, H, S, dtype=torch.float32), q.new_empty(B, H, S, dtype=torch.float32),
            q.new_empty(n32, dtype=torch.float32), q.new_empty(n32, dtype=torch.float32))


@torch.library.custom_op("morph::tg_strict_attn_bwd", mutates_args=())
def _bwd_op(q: Tensor, k: Tensor, v: Tensor, sink: Tensor, bag: Tensor | None,
            slot: Tensor | None, oc: Tensor, ow: Tensor, lc: Tensor, lw: Tensor,
            doc: Tensor, dow: Tensor, mode: int, cells: int, window: int,
            scale: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    doc, dow = doc.contiguous().to(q.dtype), dow.contiguous().to(q.dtype)
    B, H, S, D = q.shape
    bt = _tile(q.dtype)
    nt = triton.cdiv(S, bt)
    dc = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    dw, psd = torch.empty_like(dc), torch.empty_like(dc)
    _bwd_pre_kernel[(nt, B * H)](oc, ow, doc, dow, lc, sink, dc, dw, psd, H, S,
                                 D=D, BM=bt, PREC=_prec(q.dtype), num_warps=4)
    tmin, tmax = _tile_bounds(bag, S, q.device, bt)
    if bag is None:
        bag = slot = tmin
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    common = (H, S, nt, scale, scale * _LOG2E, window)
    meta = dict(MODE=mode, CELLS=max(cells, 1), D=D, BM=bt, BN=bt, PREC=_prec(q.dtype),
                **_launch(q.dtype))
    _bwd_dkdv_kernel[(nt, B * H)](q, k, v, doc, dow, lc, lw, dc, dw, bag, slot, tmin, tmax,
                                  dk, dv, *common, **meta)
    _bwd_dq_kernel[(nt, B * H)](q, k, v, doc, dow, lc, lw, dc, dw, bag, slot, tmin, tmax,
                                dq, *common, **meta)
    dsink = psd.sum(dim=(0, 2)).to(sink.dtype)
    return dq, dk, dv, dsink


@_bwd_op.register_fake
def _bwd_fake(q, k, v, sink, bag, slot, oc, ow, lc, lw, doc, dow, mode, cells, window, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(sink)


def _setup(ctx, inputs, output):
    q, k, v, sink, bag, slot, mode, cells, window, scale, save32 = inputs
    _, _, lc, lw, oc, ow = output          # the backward reads the fp32 outputs
    ctx.save_for_backward(q, k, v, sink, bag, slot, oc, ow, lc, lw)
    ctx.meta = (mode, cells, window, scale)


def _backward(ctx, doc, dow, _dlc, _dlw, _doc32, _dow32):
    q, k, v, sink, bag, slot, oc, ow, lc, lw = ctx.saved_tensors
    if oc.numel() == 0:
        raise RuntimeError("tg_strict_attention: a backward through a forward that ran "
                           "without grad (no fp32 outputs were saved)")
    if doc is None:
        doc = torch.zeros_like(q)
    if dow is None:
        dow = torch.zeros_like(q)
    dq, dk, dv, dsink = _bwd_op(q, k, v, sink, bag, slot, oc, ow, lc, lw, doc, dow,
                                *ctx.meta)
    return dq, dk, dv, dsink, None, None, None, None, None, None, None


_fwd_op.register_autograd(_backward, setup_context=_setup)


def tg_strict_attention(q: Tensor, k: Tensor, v: Tensor, sink: Tensor, index: dict,
                        window: int, scale: float) -> tuple[Tensor, Tensor]:
    """``(out_comp, out_win)``, each ``[B, H, S, D]`` in q's dtype (see the module docstring).

    ``index`` is :func:`tg_strict_index`'s dict: ``{"mode": "prelude" | "coda", "bag": [B, S]
    int32, "slot": [B, S] int32}`` or ``{"mode": "cells", "cells": M}``.
    """
    mode = _MODES[index["mode"]]
    save32 = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, sink))
    oc, ow, _, _, _, _ = _fwd_op(q, k, v, sink, index.get("bag"), index.get("slot"), mode,
                                 int(index.get("cells", 0)), int(window), float(scale), save32)
    return oc, ow


def tg_strict_index(mode: str, bag_id: Tensor | None = None, slot_mask: Tensor | None = None,
                    cells: int = 0) -> dict:
    """The index dict :func:`tg_strict_attention` reads, built ONCE per region per forward."""
    if mode == "cells":
        if cells < 1:
            raise ValueError("tg_strict_index('cells') needs cells >= 1")
        return {"mode": "cells", "cells": int(cells)}
    if mode not in ("prelude", "coda"):
        raise ValueError(f"tg_strict_index mode must be prelude / coda / cells, got {mode!r}")
    return {"mode": mode, "bag": bag_id.to(torch.int32).contiguous(),
            "slot": slot_mask.to(torch.int32).contiguous()}
