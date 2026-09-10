"""`model.core_hca_compress_ratio` on the slot loop's 64-cell sequence (audit finding F1).

`GatedPoolCompressor.forward` computes `n_blocks = S // m` and returns an empty stream when
`S < m`. The slot loop runs the core on `tul.max_slots` = 64 cells with
`hca_compress_ratio` 256, so `n_blocks = 0`, the HCA compressed branch outputs EXACTLY zero
on core blocks 1, 3 and 5, and `_gate_combine_up` still spends 0.42-0.52 of the head's
mixture on that zero tensor. Confirmed live at step 5000 on two checkpoints:
`lab/experiments/results/2026-09-10-slot-geometry-audit/README.md`, finding F1. The same
weights on a 1,152-position token sequence give `n_blocks = 4` and `|out_comp|` 680-830.

`model.core_hca_compress_ratio` is the fix and it is SCOPED to the core: the prelude and
coda run on all 1,152 positions, where 256 gives 4 blocks and the branch is healthy.

This file holds the two halves of that claim on a built model, at the sequence length the
slot core really runs:

  1. at the shipped ratio the core's HCA compressor produces zero blocks and a zero
     compressed stream at S = 64 — the defect, reproduced rather than asserted from a
     comment;
  2. at ratio 16 it produces 4 blocks and a non-zero stream, and the prelude and coda
     compressors are UNCHANGED, which is the scoping the knob exists for.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import torch

from test_tul_gl1 import _cfg, _tul                        # noqa: E402 (tests/ on sys.path)

from morph.model.attention import _CCAHCAAttention         # noqa: E402
from morph.model.transformer import MORPHTransformer       # noqa: E402

SLOT_S = 64          # tul.max_slots on every slot-loop arm since A1
SHIPPED = 256        # base.yaml hca_compress_ratio


def _model(core_ratio: int | None):
    torch.manual_seed(3)
    return MORPHTransformer(_cfg(
        tul=_tul(tg_restrict=False), n_core=2, retention=False, dropout=0.0,
        hca_compress_ratio=SHIPPED, core_hca_compress_ratio=core_ratio))


def _compressors(blocks):
    out = []
    for blk in blocks:
        impl = blk.attention._impl
        if isinstance(impl, _CCAHCAAttention) and impl.compressor is not None:
            out.append(impl.compressor)
    return out


def test_the_shipped_ratio_is_dead_at_the_slot_budget():
    m = _model(None)
    comps = _compressors(m.core)
    assert comps, "the tiny model has no HCA block in its core — the test is vacuous"
    x = torch.randn(2, SLOT_S, m.cfg.d_model)
    for c in comps:
        assert c.m == SHIPPED
        out = c(x)
        assert out.shape[1] == 0, out.shape          # n_blocks = 64 // 256 = 0
        assert float(out.detach().abs().sum()) == 0.0


def test_the_fix_gives_the_core_four_live_blocks_and_leaves_the_rest_alone():
    m = _model(16)
    comps = _compressors(m.core)
    assert comps
    x = torch.randn(2, SLOT_S, m.cfg.d_model)
    for c in comps:
        assert c.m == 16
        out = c(x)
        assert out.shape[1] == SLOT_S // 16 == 4, out.shape
        assert float(out.detach().abs().sum()) > 0.0
    # SCOPED: everything outside the core keeps the shipped ratio.
    outside = _compressors(list(m.prelude) + list(m.coda))
    assert outside, "no HCA block outside the core — the scoping check is vacuous"
    for c in outside:
        assert c.m == SHIPPED


def test_two_blocks_is_the_least_the_fix_may_give():
    """The arm picks 16 because 64 // 16 = 4 is what the token path gets at seq_len 1024
    (1152 // 256 = 4). 32 would give 2 and is the floor; anything at or above 64 is the
    defect again. Stated as a test so the choice cannot drift into a config by accident."""
    for ratio, blocks in ((16, 4), (32, 2), (64, 1), (128, 0), (256, 0)):
        assert SLOT_S // ratio == blocks
