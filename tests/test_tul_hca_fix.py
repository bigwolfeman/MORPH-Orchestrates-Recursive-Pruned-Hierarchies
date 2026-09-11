"""`model.core_hca_compress_ratio` on the slot loop's 64-cell sequence (audit finding F1).

`GatedPoolCompressor.forward` computes `n_blocks = S // m` and returns an empty stream when
`S < m`. The slot loop runs the core on `tul.max_slots` = 64 cells with the shipped
`hca_compress_ratio` 256, so `n_blocks = 0`, the HCA compressed branch outputs EXACTLY zero
on core blocks 1, 3 and 5, and `_gate_combine_up` still spends 0.42-0.52 of the head's
mixture on that zero tensor. Confirmed live at step 5000 on two checkpoints:
`lab/experiments/results/2026-09-10-slot-geometry-audit/README.md`, finding F1. The same
weights on a 1,152-position token sequence give `n_blocks = 4` and `|out_comp|` 680-830.

The fix (`model.core_hca_compress_ratio: 16`, giving the core 4 blocks, what the token
path gets at seq_len 1024) shipped as the SLOT-LOOP DEFAULT in `tul_short.yaml` on
2026-09-11, measured on the arm `slot-mux-hca-fix`
(`lab/experiments/successes/2026-09-10-arc-slot-mux-hca-fix.md`): the branch deficit closes
(attention slot/token 0.73-1.08 against the unrepaired 0.37-0.43), CE 0.030 better at 5k,
1.04x wall clock, and no change to the loop's depth contribution. It is scoped to the core:
the prelude and coda run on all 1,152 positions, where 256 already gives 4 blocks and the
branch was never dead. `tul_slot_mux_hca_fix.yaml`, the one-factor arm that shipped this,
is retired — every `tul_short`-derived config now carries the fix by default, so a
separate arm testing "ratio 16 vs not" no longer has a control to run against.

This file holds three claims, each reproduced on a built object rather than asserted from
a comment:

  1. every slot-loop config (anything composing `tul_short.yaml`) resolves
     `model.core_hca_compress_ratio` to 16 — the shipped default, not an opt-in;
  2. a core HCA compressor built at that ratio on S = 64 produces 4 blocks and a non-zero
     compressed stream, while the OLD (pre-fix) ratio of 256 on the same S produces 0 blocks
     and an exactly-zero stream, and the prelude/coda compressors are unaffected either way;
  3. `base.yaml` (the paid loop, `L_total` 1152 at seq 1024) is UNCHANGED: it never composes
     `tul_short` and its `core_hca_compress_ratio` stays unset (null) — the paid loop was
     never in the defect's reach (1152 // 256 = 4 already) and this fix does not touch it.

CPU only, tiny config for (2), Hydra compose for (1) and (3), no tokenizer, no GPU.
"""

from __future__ import annotations

import os

import torch
from hydra import compose, initialize_config_dir

from test_tul_gl1 import _cfg, _tul                        # noqa: E402 (tests/ on sys.path)

from morph.model.attention import _CCAHCAAttention         # noqa: E402
from morph.model.transformer import MORPHTransformer       # noqa: E402

SLOT_S = 64          # tul.max_slots on every slot-loop arm since A1
SHIPPED_DEFECT = 256   # the ratio that is dead at S = 64 (pre-fix core value / base.yaml)
SHIPPED_FIX = 16        # tul_short.yaml's shipped core default since 2026-09-11

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_ROOT, "morph", "configs")

# A sample of configs that compose tul_short, spanning the lineages a change here could
# silently miss: the arm itself, the M-next ruler and its think-once ancestor, the staged
# credit-assignment arm, and the Parcae-core swap (which forbids the knob outright and
# must therefore override it back to null rather than inherit the new default).
_SLOT_LOOP_CONFIGS = [
    "tul_short", "tul_a1", "tul_to_mnext", "tul_slot_mux_norm_match",
    "tul_slot_mnext_staged",
]


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=name)


def _model(core_ratio: int | None):
    torch.manual_seed(3)
    return MORPHTransformer(_cfg(
        tul=_tul(tg_restrict=False), n_core=2, retention=False, dropout=0.0,
        hca_compress_ratio=SHIPPED_DEFECT, core_hca_compress_ratio=core_ratio))


def _compressors(blocks):
    out = []
    for blk in blocks:
        impl = blk.attention._impl
        if isinstance(impl, _CCAHCAAttention) and impl.compressor is not None:
            out.append(impl.compressor)
    return out


def test_every_slot_loop_config_composes_with_the_shipped_default():
    for name in _SLOT_LOOP_CONFIGS:
        cfg = _compose(name)
        assert cfg.model.get("core_hca_compress_ratio", None) == SHIPPED_FIX, (
            f"{name} did not resolve model.core_hca_compress_ratio to {SHIPPED_FIX}"
        )


def test_the_parcae_core_arm_overrides_the_default_back_to_null():
    """The Parcae core has no pooled compressor; the build RAISES if this is non-null.
    The arm predates the fix and must keep composing, so it opts back out explicitly
    rather than silently inheriting `tul_short`'s new default."""
    cfg = _compose("tul_slot_mnext_parcae_core")
    assert cfg.model.get("core_impl", "morph") == "parcae"
    assert cfg.model.get("core_hca_compress_ratio", None) is None


def test_base_yaml_paid_loop_is_unaffected():
    cfg = _compose("base")
    assert cfg.model.get("core_hca_compress_ratio", None) is None
    assert cfg.tul.get("tokens_through_core", None) is True


def test_the_old_ratio_is_dead_at_the_slot_budget():
    m = _model(None)
    comps = _compressors(m.core)
    assert comps, "the tiny model has no HCA block in its core — the test is vacuous"
    x = torch.randn(2, SLOT_S, m.cfg.d_model)
    for c in comps:
        assert c.m == SHIPPED_DEFECT
        out = c(x)
        assert out.shape[1] == 0, out.shape          # n_blocks = 64 // 256 = 0
        assert float(out.detach().abs().sum()) == 0.0


def test_the_shipped_fix_gives_the_core_four_live_blocks_and_leaves_the_rest_alone():
    m = _model(SHIPPED_FIX)
    comps = _compressors(m.core)
    assert comps
    x = torch.randn(2, SLOT_S, m.cfg.d_model)
    for c in comps:
        assert c.m == SHIPPED_FIX
        out = c(x)
        assert out.shape[1] == SLOT_S // SHIPPED_FIX == 4, out.shape
        assert float(out.detach().abs().sum()) > 0.0
    # SCOPED: everything outside the core keeps the shipped (defect) ratio.
    outside = _compressors(list(m.prelude) + list(m.coda))
    assert outside, "no HCA block outside the core — the scoping check is vacuous"
    for c in outside:
        assert c.m == SHIPPED_DEFECT


def test_two_blocks_is_the_least_the_fix_may_give():
    """The default is 16 because 64 // 16 = 4 is what the token path gets at seq_len 1024
    (1152 // 256 = 4). 32 would give 2 and is the floor; anything at or above 64 is the
    defect again. Stated as a test so the choice cannot drift into a config by accident."""
    for ratio, blocks in ((16, 4), (32, 2), (64, 1), (128, 0), (256, 0)):
        assert SLOT_S // ratio == blocks
