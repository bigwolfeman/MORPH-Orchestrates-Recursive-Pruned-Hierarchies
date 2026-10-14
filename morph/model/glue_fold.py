"""``model.glue_fold``: compiled forms of the eager elementwise glue around the custom ops.

The speed base runs its blocks, GEMMs and attention as fused kernels, but a number of small
modules still run op by op: the span decoder's RMS norms and SwiGLU, the pointer head's
``[B, H, N, N]`` score chain, the tied head's table build, the per-layer injection term and
the loop's diagonal injection. Each op of such a module is a read and a write of a full
tensor, and its backward is as long again. ``torch.compile`` fuses each into one forward
kernel and a few backward kernels.

Every fold here is a rewrite of a function that already exists, behind ``model.glue_fold``:
the unfolded function is still the code that runs with the key off, and the folded one is the
SAME Python function passed through :func:`compile_glue`. The results are NOT bit-identical to
the eager ops (Inductor reduces a row in its own order, keeps fp32 between fused bf16 ops and
may contract a multiply-add); ``tests/test_glue_fold.py`` bounds each against an fp64
reference and against the eager function, in value and in every gradient.

The fused-CE head weight is the one fold that is not compiled (``fused_ce._pad_cast``) and it
is exact.
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
from torch import Tensor

from .loop_glue import rms_norm_fp32


@functools.lru_cache(maxsize=None)
def compile_glue(fn: Callable, dynamic: bool) -> Callable:
    """``torch.compile(fn)``, ONE compiled object per (function, dynamic) for the process.

    ``dynamic=True`` for a function whose row count differs between calls (the cell count
    ``tul.cell_count`` changes every tensor's length, eval runs another batch): one symbolic
    compile serves all of them, where static shapes would exhaust Dynamo's per-function
    recompile limit and fall back to eager without a word (the ``loop_glue`` lesson)."""
    return torch.compile(fn, dynamic=dynamic)


def rms_norm_cast(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``attention.RMSNorm`` followed by the caller's ``.to(x.dtype)``: the cast sits inside
    the compiled region, so the fp32 product never reaches memory."""
    return rms_norm_fp32(x, weight, eps).to(x.dtype)


def silu_mul(g: Tensor, u: Tensor) -> Tensor:
    """``silu(g) * u``: the SwiGLU product, one kernel instead of two."""
    return torch.nn.functional.silu(g) * u


@functools.lru_cache(maxsize=None)
def injection_term(cs: int, ce: int) -> Callable:
    """The tensor part of ``MORPHTransformer._build_injection_term`` for the ctx slice
    ``[cs, ce)``: ``lam * bigram`` (full width) with ``ctx`` added into the slice. Eager; the
    fold compiles it, which assembles the term in ONE kernel instead of a cast, a multiply, a
    slice add and a cat. ``ctx`` has the carrier's dtype already (``x0`` term plus any
    value-embed term, built by the caller)."""
    def fn(bigram_emb: Tensor, lam: Tensor, ctx: Tensor) -> Tensor:
        dtype = ctx.dtype
        full = lam.to(dtype) * bigram_emb.to(dtype)
        return torch.cat([full[..., :cs], full[..., cs:ce] + ctx, full[..., ce:]], dim=-1)
    return fn


@functools.lru_cache(maxsize=None)
def diag_inject(start: int, end: int, keep_dtype: bool) -> Callable:
    """``DiagonalInjection.forward`` without the ``B`` map: ``h_ctx <- A * h_ctx + dt * e_ctx``
    on the slice ``[start, end)`` of the carrier ``h``, and the cat that puts it back. Eager;
    the fold compiles it into one kernel over the carrier, in place of an exp, a clamp, two
    multiplies, an add, a slice copy and a cat. ``e_ctx`` is already the slice's source
    (``source_view``)."""
    def fn(h: Tensor, e_ctx: Tensor, log_A: Tensor, log_dt: Tensor) -> Tensor:
        A = log_A.exp().clamp(max=0.9999)
        dt = log_dt.exp()
        new_ctx = A * h[..., start:end] + dt * e_ctx
        if keep_dtype:
            new_ctx = new_ctx.to(h.dtype)
        return torch.cat([h[..., :start], new_ctx, h[..., end:]], dim=-1)
    return fn
