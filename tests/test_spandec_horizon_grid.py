"""`lab/divergence/spandec_horizon_grid.py` — the depth x horizon grid's contract.

The instrument holds the span-decoder TARGET fixed and moves the loop depth, so every
claim it makes rests on two things being true:

* **The exit column IS the model's own reading.** Column `exit` at the model's own eval
  depth reproduces the `spandec_ce` the shipped forward reports, to 1e-5. If it does not,
  the grid is scoring some other function and its K-curve means nothing.
* **The mask is depth-independent**, and the pass columns read `pos_pass` while the exit
  column reads `pos` — one parameter cannot be two offsets at once
  (`SpanDecoder.__init__`), so a grid that mixed them would compare a trained table
  against an untrained one.

Plus the housekeeping that decides whether a column runs at all: a checkpoint with no
`pos_pass` reports the exit column ALONE, with a reason, rather than faking the per-pass
geometry through `pos`.

Two-sided throughout: the pos/pos_pass test perturbs one table and asserts the OTHER
column does not move, so it cannot pass on a grid that ignores the table entirely.

CPU only, fp32, tiny config — the `tests/test_tul_oracle_z.py` fixtures.
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
from spandec_horizon_grid import (  # noqa: E402
    choose_columns,
    column_targets,
    exit_state,
    row_ce,
)

V = 64
DOT = 10
OWN_DEPTH = 3


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
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                spandec_per_pass=True, spandec_pass_tokens=3, spandec_pass_horizon_max=3,
                slot_depth_fixed=OWN_DEPTH, slot_max_depth=4)
    base.update(kw)
    return TULConfig(**base)


def _batch(B: int = 2, n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(seed: int = 99, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    m = m.eval().float()
    # the zero-init position tables make every offset identical at init, which would let
    # a pos / pos_pass mix-up pass unnoticed. Give both tables real content.
    with torch.no_grad():
        m.tul_spandec.pos.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(1))
        if m.tul_spandec.pos_pass is not None:
            m.tul_spandec.pos_pass.normal_(
                0.0, 0.5, generator=torch.Generator().manual_seed(2))
    return m


def _grid_ce(model, inp, layout, name: str, kind: str, h: int, depth: int) -> float:
    z = model._readout(exit_state(model, inp, layout, depth))
    ids, valid, table, _J = column_targets(model, inp, layout, kind, h)
    s, c = row_ce(model, z, ids, valid, table)
    return float(s.sum() / c.sum())


# ── the exit column is the model's own reading ───────────────────────────────

def test_exit_column_equals_the_models_own_spandec_ce():
    m = _model()
    inp, lab, layout, _ = _batch()
    own = float(m(inp, labels=lab, slot_layout=layout)["spandec_ce"])
    got = _grid_ce(m, inp, layout, "exit", "exit", 1, OWN_DEPTH)
    assert abs(own - got) < 1e-5, f"exit column {got:.6f} vs the model's {own:.6f}"


def test_row_ce_aggregates_to_one_batch_call():
    """Per-row scoring must be an ARRANGEMENT of the shipped kernel, not a re-derivation."""
    from morph.model.fused_ce import fused_linear_cross_entropy

    m = _model()
    inp, _lab, layout, _ = _batch()
    z = m._readout(exit_state(m, inp, layout, OWN_DEPTH))
    ids, valid, table, _J = column_targets(m, inp, layout, "exit", 1)
    s, c = row_ce(m, z, ids, valid, table)
    dec, tc = m.tul_spandec, m.cfg.tul
    w = m.embed.lm_weight()
    st = dec.decode(z, ids, valid, w.detach(), pos=table)
    labs = torch.where(valid, ids, torch.full_like(ids, -100))
    with torch.no_grad():
        one = float(fused_linear_cross_entropy(
            st.reshape(-1, st.shape[-1]), w.detach(), labs.reshape(-1), ignore_index=-100,
            chunk_size=m.cfg.ce_chunk_size, mask_token_id=tc.slot_id))
    assert abs(float(s.sum() / c.sum()) - one) < 1e-5


# ── the depth actually moves, and the mask does not ──────────────────────────

def test_exit_state_runs_the_forced_number_of_passes():
    m = _model()
    inp, _lab, layout, _ = _batch()
    a = exit_state(m, inp, layout, 1)
    b = exit_state(m, inp, layout, 3)
    assert not torch.allclose(a, b), "depth 1 and depth 3 give the same exit state — " \
                                     "the forced depth is being ignored"


def test_the_target_is_identical_at_every_depth():
    m = _model()
    inp, _lab, layout, _ = _batch()
    ref = column_targets(m, inp, layout, "pass", 2)
    for d in (1, 2, 3):
        _z = m._readout(exit_state(m, inp, layout, d))
        got = column_targets(m, inp, layout, "pass", 2)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_every_column_scores_a_nonempty_and_complete_population():
    m = _model()
    inp, _lab, layout, _ = _batch()
    for kind, h in (("exit", 1), ("pass", 1), ("pass", 3)):
        _ids, valid, _t, _J = column_targets(m, inp, layout, kind, h)
        assert int(valid.sum()) > 0, f"column {kind}/{h} scores nothing"
        # a pad slot is never graded, and neither is the last slot of a row
        assert int(valid[~layout.slot_valid].sum()) == 0


# ── pos vs pos_pass ──────────────────────────────────────────────────────────

def test_pass_columns_read_pos_pass_and_the_exit_column_reads_pos():
    m = _model()
    inp, _lab, layout, _ = _batch()
    assert column_targets(m, inp, layout, "pass", 1)[2] is m.tul_spandec.pos_pass
    assert column_targets(m, inp, layout, "exit", 1)[2] is m.tul_spandec.pos


def test_perturbing_pos_pass_moves_only_the_pass_columns():
    """Two-sided: a grid that ignored the table would pass neither half of this."""
    m = _model()
    inp, _lab, layout, _ = _batch()
    before = {n: _grid_ce(m, inp, layout, n, k, h, OWN_DEPTH)
              for n, k, h in (("pass_h1", "pass", 1), ("exit", "exit", 1))}
    with torch.no_grad():
        m.tul_spandec.pos_pass.add_(1.0)
    after = {n: _grid_ce(m, inp, layout, n, k, h, OWN_DEPTH)
             for n, k, h in (("pass_h1", "pass", 1), ("exit", "exit", 1))}
    assert abs(after["pass_h1"] - before["pass_h1"]) > 1e-4, \
        "pos_pass moved and the pass column did not — it is not reading that table"
    assert abs(after["exit"] - before["exit"]) < 1e-6, \
        "pos_pass moved and the EXIT column moved with it — the tables are crossed"


def test_a_longer_horizon_scores_strictly_more_tokens():
    m = _model()
    inp, _lab, layout, _ = _batch()
    n1 = int(column_targets(m, inp, layout, "pass", 1)[1].sum())
    n3 = int(column_targets(m, inp, layout, "pass", 3)[1].sum())
    assert n3 > n1


# ── which columns run ────────────────────────────────────────────────────────

def test_per_pass_checkpoint_runs_the_pass_columns():
    m = _model()
    cols, skipped = choose_columns(m, [1, 3])
    assert [c[0] for c in cols] == ["pass_h1", "pass_h3", "exit"]
    assert skipped == {}


def test_non_per_pass_checkpoint_reports_the_exit_column_alone():
    m = _model(spandec_per_pass=False)
    assert m.tul_spandec.pos_pass is None
    cols, skipped = choose_columns(m, [1, 3])
    assert [c[0] for c in cols] == ["exit"]
    assert "pass_h*" in skipped and "pos_pass" in skipped["pass_h*"]


def test_a_horizon_past_the_cap_is_skipped_with_a_reason():
    m = _model()
    cols, skipped = choose_columns(m, [1, 9])
    assert [c[0] for c in cols] == ["pass_h1", "exit"]
    assert "spandec_pass_horizon_max" in skipped["pass_h9"]


def test_exit_state_refuses_a_depth_the_model_never_ran():
    m = _model()
    inp, _lab, layout, _ = _batch()
    with pytest.raises(ValueError, match="slot_max_depth"):
        exit_state(m, inp, layout, 9)
