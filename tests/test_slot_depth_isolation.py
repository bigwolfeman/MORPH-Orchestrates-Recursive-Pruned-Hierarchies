"""`slot_depths` — the eval-only per-slot depth table, and the map it is read through.

`lab/divergence/slot_depth_isolation.py` asks what ONE slot's passes are worth. No scalar
knob can say "slot 7 at depth 1, every other slot at 6", so `MORPHTransformer.forward`
grew a per-forward DATA argument, the `slot_layout` pattern. This file is its contract:

* **Off is nothing.** `slot_depths=None` reproduces the shipped forward BIT for BIT.
* **A constant table is the shipped knob.** `full(c)` equals `tul.slot_depth_fixed=c`,
  bit for bit, so the override is the same function the sweep already measures.
* **Pad slots loop once,** whatever the table says — `_sample_slot_depths`' rule, in one
  home (`_pad_slot_depths`).
* **It raises rather than clamps / ignores:** out of range, wrong shape, wrong dtype,
  training mode, no layout, the paid loop, an FM planner, `halt`.
* **The isolation is CAUSAL, two-sided.** Changing slot s's depth moves NOTHING before
  slot s's cell and DOES move something after it. A one-sided test would pass on a
  forward that ignored the table completely.

and the position map `lab/divergence/_next_span.py`:

* `labels.gather(1, pos)` reproduces `morph.model.tul_spandec.next_span_slots` token for
  token, so offset 0 of span s+1 is the position at span s's LAST TOKEN — before the slot
  cell, `span_budget_profile.py`'s convention.
* no slot cell is ever scored, and the last slot of a row has no target.

CPU only, fp32, tiny config, no tokenizer — the `tests/test_tul_oracle_z.py` fixtures.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))
from _next_span import next_span_mask, next_span_positions  # noqa: E402
from slot_depth_isolation import bucket_of, score, tile_layout  # noqa: E402

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, slot_depth_fixed=3, slot_max_depth=4)
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 140, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed: int = 99, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    return m.eval().float()


def _table(layout, d: int) -> torch.Tensor:
    return torch.full(layout.slot_index.shape, int(d), dtype=torch.long)


# ── off is nothing ───────────────────────────────────────────────────────────

def test_none_is_bit_identical():
    m = _model()
    inp, lab, layout, _ = _batch()
    a = m(inp, labels=lab, slot_layout=layout)
    b = m(inp, labels=lab, slot_layout=layout, slot_depths=None)
    assert torch.equal(a["loss"], b["loss"])
    # and the label-free forward, which is what every probe in lab/divergence scores
    assert torch.equal(m(inp, slot_layout=layout)["logits"],
                       m(inp, slot_layout=layout, slot_depths=None)["logits"])


def test_constant_table_is_the_shipped_fixed_depth():
    """`full(c)` must BE `tul.slot_depth_fixed=c` — the override is the same function."""
    m = _model(slot_depth_fixed=3)
    inp, lab, layout, _ = _batch()
    ref = m(inp, labels=lab, slot_layout=layout)["loss"]
    got = m(inp, labels=lab, slot_layout=layout,
            slot_depths=_table(layout, 3))["loss"]
    assert torch.equal(ref, got)
    # and a DIFFERENT constant must be the model's own forward at that constant
    m.cfg.tul.slot_depth_fixed = 2
    ref2 = m(inp, labels=lab, slot_layout=layout)["loss"]
    m.cfg.tul.slot_depth_fixed = 3
    got2 = m(inp, labels=lab, slot_layout=layout,
             slot_depths=_table(layout, 2))["loss"]
    assert torch.equal(ref2, got2)
    assert not torch.equal(ref, ref2), "depth 2 and depth 3 give the same loss — the " \
                                       "fixture's loop does nothing and proves nothing"


def test_pad_slots_loop_once_whatever_the_table_says():
    m = _model()
    _inp, _lab, layout, _ = _batch()
    assert not bool(layout.slot_valid.all()), "fixture has no pad slot to check"
    d = m._slot_depth_override(layout, _table(layout, 4), torch.device("cpu"))
    assert torch.equal(d[~layout.slot_valid],
                       torch.ones_like(d[~layout.slot_valid]))
    assert torch.equal(d[layout.slot_valid],
                       torch.full_like(d[layout.slot_valid], 4))


# ── it raises rather than clamping or ignoring ───────────────────────────────

@pytest.mark.parametrize("bad,msg", [(0, "slot_max_depth"), (5, "slot_max_depth")])
def test_out_of_range_raises(bad, msg):
    m = _model()
    inp, lab, layout, _ = _batch()
    with pytest.raises(ValueError, match=msg):
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, bad))


def test_wrong_shape_raises():
    m = _model()
    inp, lab, layout, _ = _batch()
    with pytest.raises(ValueError, match="max_slots"):
        m(inp, labels=lab, slot_layout=layout,
          slot_depths=torch.full((layout.slot_index.shape[0], 3), 2, dtype=torch.long))


def test_float_table_raises():
    m = _model()
    inp, lab, layout, _ = _batch()
    with pytest.raises(ValueError, match="integer"):
        m(inp, labels=lab, slot_layout=layout,
          slot_depths=_table(layout, 2).float())


def test_training_raises():
    m = _model().train()
    inp, lab, layout, _ = _batch()
    with pytest.raises(RuntimeError, match="EVAL ONLY"):
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, 2))


def test_no_layout_raises():
    m = _model()
    inp, lab, layout, _ = _batch()
    with pytest.raises(ValueError, match="requires slot_layout"):
        m(inp, labels=lab, slot_depths=_table(layout, 2))


def test_paid_loop_raises():
    m = _model(tokens_through_core=True, slot_depth_fixed=0)
    inp, lab, layout, _ = _batch()
    with pytest.raises(NotImplementedError, match="paid loop"):
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, 2))


def test_halt_raises():
    m = _model()
    inp, _lab, layout, _ = _batch()
    x, x0, bg = m._tul_front(inp, layout)
    with pytest.raises(ValueError, match="halt=True and slot_depths"):
        m._tul_core(x, x0, bg, layout, halt=True, slot_depths=_table(layout, 2))


# ── the isolation is causal, two-sided ───────────────────────────────────────

def test_changing_one_slot_moves_only_what_comes_after_it():
    """The whole instrument rests on this. Two-sided: nothing before, something after."""
    m = _model()
    inp, lab, layout, _ = _batch()
    s = 2
    assert bool(layout.slot_valid[:, s].all())
    base = _table(layout, 3)
    other = base.clone()
    other[:, s] = 1
    a = m(inp, slot_layout=layout, slot_depths=base)["logits"]
    b = m(inp, slot_layout=layout, slot_depths=other)["logits"]
    for r in range(inp.shape[0]):
        cut = int(layout.slot_index[r, s])
        assert torch.equal(a[r, :cut], b[r, :cut]), \
            f"row {r}: slot {s}'s depth changed a position BEFORE its cell — not causal"
        assert not torch.equal(a[r, cut:], b[r, cut:]), \
            f"row {r}: slot {s}'s depth changed NOTHING after its cell — the table is " \
            "being ignored"


# ── the position map ─────────────────────────────────────────────────────────

def test_positions_predict_the_next_span_tokens():
    """The map's own check, exercised: it raises when the convention slips by one."""
    inp, lab, layout, _ = _batch()
    pos, valid = next_span_positions(layout, lab, inp)
    assert int(valid.sum()) > 50
    got = lab.gather(1, pos.reshape(pos.shape[0], -1)).reshape(pos.shape)
    from morph.model.tul_spandec import next_span_slots
    ids_ref, _ = next_span_slots(inp, layout, pos.shape[2])
    assert torch.equal(got[valid], ids_ref[valid])


def test_offset_zero_sits_before_the_slot_cell():
    inp, lab, layout, _ = _batch()
    pos, valid = next_span_positions(layout, lab, inp)
    B, S, _ = pos.shape
    for b in range(B):
        for s in range(S):
            if not bool(valid[b, s, 0]):
                continue
            assert int(pos[b, s, 0]) == int(layout.slot_index[b, s]) - 1
            assert not bool(layout.slot_mask[b, pos[b, s, 0]])


def test_no_slot_cell_and_no_span_zero_token_is_a_target_body():
    inp, lab, layout, _ = _batch()
    _pos, _valid, slot_of = next_span_mask(layout, lab, inp)
    scored = slot_of >= 0
    assert int(scored[layout.slot_mask].sum()) == 0, "a slot cell was scored"
    # span 0's tokens are scored only at its LAST one (which predicts span 1's first)
    span0 = (~layout.slot_mask) & (layout.bag_id == 0)
    assert int((scored & span0).sum()) == int(inp.shape[0])


def test_the_last_slot_of_a_row_has_no_target():
    inp, lab, layout, _ = _batch()
    _pos, valid = next_span_positions(layout, lab, inp)
    n = layout.slot_valid.long().sum(dim=1)
    for b in range(inp.shape[0]):
        last = int(n[b]) - 1
        assert not bool(valid[b, last].any())


# ── the script's own helpers ─────────────────────────────────────────────────

def test_score_reads_exactly_one_slots_span():
    inp, lab, layout, _ = _batch()
    pos, valid = next_span_positions(layout, lab, inp)
    J = pos.shape[2]
    # a CE map that is 1.0 exactly at slot 1's scored positions and 0 everywhere else
    ce = torch.zeros(inp.shape, dtype=torch.float32)
    ce.scatter_(1, pos[:, 1, :], valid[:, 1, :].float())
    rs, rc, os_, oc = score(ce, pos, valid, 1, bucket_of(np.arange(J)))
    assert np.allclose(rs, rc), "score did not read slot 1's own positions"
    rs0, rc0, _, _ = score(ce, pos, valid, 0, bucket_of(np.arange(J)))
    assert rc0.sum() > 0 and np.allclose(rs0, 0.0), \
        "score on slot 0 picked up slot 1's positions"
    assert np.allclose(oc.sum(axis=1), rc)


def test_bucket_of_caps_at_eight():
    assert list(bucket_of(np.arange(12))) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]


def test_tile_layout_is_row_major_copies():
    _inp, _lab, layout, _ = _batch()
    t = tile_layout(layout, 3)
    assert t.slot_index.shape[0] == layout.slot_index.shape[0] * 3
    for b in range(layout.slot_index.shape[0]):
        for c in range(3):
            assert torch.equal(t.slot_index[b * 3 + c], layout.slot_index[b])
            assert torch.equal(t.slot_mask[b * 3 + c], layout.slot_mask[b])
    assert tile_layout(layout, 1) is layout


def test_tiled_forward_equals_the_untiled_one():
    """`--slot-chunk` must be an arrangement, not a different measurement."""
    m = _model()
    inp, lab, layout, _ = _batch()
    base = _table(layout, 3)
    v = [base.clone(), base.clone()]
    v[1][:, 1] = 1
    single = [m(inp, slot_layout=layout, slot_depths=t)["logits"] for t in v]
    tiled = m(inp.repeat_interleave(2, dim=0),
              slot_layout=tile_layout(layout, 2),
              slot_depths=torch.stack(v, dim=1).reshape(-1, base.shape[1]))["logits"]
    for i in range(2):
        assert torch.allclose(tiled[i::2], single[i], atol=1e-5), \
            f"variant {i} differs between the tiled and the untiled forward"
