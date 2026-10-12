"""Compiled forms of the slot loop's elementwise glue (``model.loop_glue_compile``).

The slot loop (``MORPHTransformer._tul_core``) keeps a few full-width fp32 passes outside the
compiled blocks: the entry norm of the prelude output (``input_norm``, one [B, L, n, C] pass
over the token positions) and the slot-cell norm after every pass (``tul_cell_norm``). Run
eagerly, ``attention.RMSNorm`` is six kernels forward and about a dozen backward over the
whole carrier, each one a read and a write of the full tensor. ``torch.compile`` fuses them
into one reduction kernel forward and two backward.

NOT bit-identical to the eager norm: Inductor reduces the row in its own order and may
contract a multiply-add. ``tests/test_loop_glue.py`` bounds the difference against an fp64
reference, and against the eager norm, in value and in both gradients.
"""
from __future__ import annotations

import torch
from torch import Tensor


def rms_norm_fp32(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``attention.RMSNorm.forward``, statement for statement: the mean square in fp32, the
    normalised value cast back to ``x.dtype``, then the (fp32) weight, which promotes."""
    norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * norm).to(x.dtype) * weight


def compile_rms_norm():
    """The compiled ``rms_norm_fp32``. Dynamic shapes: the slot-cell norm sees a different row
    count at every pass under ``model.slot_compact_norm`` (the compacted rows), and one
    symbolic compile serves all of them where static shapes would exhaust Dynamo's per-function
    recompile limit and fall back to eager without a word."""
    return torch.compile(rms_norm_fp32, dynamic=True)
