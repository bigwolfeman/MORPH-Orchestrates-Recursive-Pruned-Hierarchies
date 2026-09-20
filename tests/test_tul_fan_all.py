"""``tul.fan_mix="all"`` — write every stream, winner-takes-all responsibility (2026-09-20).

WHAT THIS FILE HAS TO PROVE:

1. THE WRITE IS 1:1. Under ``all`` the forward hands ``prefix_project`` the K cells
   themselves (``cells=``, the Thought Register's route), so stream i reaches the coda in
   prefix cell i; a mixing fan hands it nothing (``cells`` None) and one state.
2. "STREAM i ALONE" IS THE DEPLOYED GEOMETRY MINUS THE LOSERS: the shared per-stream
   write blanks the other K-1 cells (they carry ``E_pass`` only), and under a mixing fan it
   is the single-source broadcast the select arm used (bit-identical to before).
3. THE TERM IS CHARGED AND REACHES THE STREAMS: the train step reports ``fan_wta_*``, the
   loss moves by exactly ``lambda * wta_ce`` when the knob moves, and the winner's stream
   gets gradient through the term alone.
4. THE MIX IS THE MEAN WITH NO GATE, refuses a ``choice``; ``fan/mix_entropy`` reads ln K.
5. REFUSALS: ``all`` needs ``prefix_k == fan_k``; ``fan_all_wta_lambda`` is refused off
   ``all``; ``fan_select_eps`` is readable under ``all``; the gate knob is not.
6. The new key composes through ``tul_setup``.

CPU only, fp32, the ``tests/test_tul_fan.py`` fixtures.
Record: lab/experiments/planned/2026-09-20-lxtul-fan4-all.md
"""
from __future__ import annotations

import math

import pytest
import torch

from morph.model.tul_fan import TULFanMix
from test_tul_fan import _batch, _model, _tul


def _all_model(**kw):
    kw.setdefault("fan_select_eps", 0.0)
    return _model(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", **kw)


# ── 4. the mix ────────────────────────────────────────────────────────────────

def test_fan_mix_all_is_the_mean_with_uniform_weights_and_no_gate():
    torch.manual_seed(0)
    mix = TULFanMix(8, 4, mode="all")
    assert mix.gate is None
    cells = torch.randn(2, 3, 4, 8)
    state, w = mix(cells)
    assert torch.allclose(state, cells.mean(dim=2))
    assert torch.allclose(w, torch.full((2, 3, 4), 0.25))
    with pytest.raises(ValueError, match="takes no choice"):
        mix(cells, torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(RuntimeError, match="no gate"):
        mix.logits(cells)


# ── 1. the write ──────────────────────────────────────────────────────────────

def _spy_write(m, inp, lab, layout):
    seen: dict = {}
    real = m.tul.prefix_project

    def spy(h_slots, lay, l_total, cells=None):
        seen.setdefault("calls", []).append(None if cells is None else cells.detach().clone())
        seen.setdefault("h", []).append(h_slots.detach().clone())
        return real(h_slots, lay, l_total, cells=cells)

    m.tul.prefix_project = spy
    try:
        with torch.no_grad():
            out = m(inp, labels=lab, slot_layout=layout)
    finally:
        m.tul.prefix_project = real
    return out, seen


def test_all_writes_the_k_cells_one_to_one_and_a_mixing_fan_writes_one_state():
    ids, inp, lab, layout = _batch()
    m = _all_model()
    out, seen = _spy_write(m, inp, lab, layout)
    # eval: the shipped write, then the oracle's K one-stream replays
    shipped = seen["calls"][0]
    assert shipped is not None and shipped.shape[2] == 4, "the all-fan did not take the cells route"
    assert torch.allclose(seen["h"][0], shipped.mean(dim=2)), "h_slots under all is the cells' mean"
    m2 = _model(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="mean")
    _out2, seen2 = _spy_write(m2, inp, lab, layout)
    assert seen2["calls"][0] is None, "a mixing fan must write ONE state (cells None)"


# ── 2. stream i alone ─────────────────────────────────────────────────────────

def test_stream_write_blanks_the_other_cells_under_all_and_broadcasts_under_a_mix():
    ids, inp, lab, layout = _batch()
    m = _all_model()
    S = layout.slot_index.shape[1]
    B = inp.shape[0]
    L = inp.shape[1]
    torch.manual_seed(3)
    C = m.cfg.d_model
    cells = torch.randn(B, S, 4, C)
    v1, pos = m._tul_fan_stream_write(cells, 1, layout, L)
    blank = torch.zeros_like(cells)
    blank[:, :, 1] = cells[:, :, 1]
    v_ref, _ = m.tul.prefix_project(cells[:, :, 1], layout, L, cells=blank)
    assert torch.equal(v1, v_ref)
    # prefix cell k of slot s is value index s*K + k: cell 1 carries the stream, the
    # others carry exactly what a zero cell projects to (E_pass only, or 0)
    zero_v, _ = m.tul.prefix_project(torch.zeros_like(cells[:, :, 0]), layout, L,
                                     cells=torch.zeros_like(cells))
    v1 = v1.view(B, S, 4, -1)
    zv = zero_v.view(B, S, 4, -1)
    for k in (0, 2, 3):
        assert torch.equal(v1[:, :, k], zv[:, :, k])
    assert not torch.allclose(v1[:, :, 1], zv[:, :, 1])
    # a mixing fan: the single-source broadcast (the select arm's write), unchanged
    m2 = _model(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="select")
    v2, _ = m2._tul_fan_stream_write(cells, 1, layout, L)
    v_bc, _ = m2.tul.prefix_project(cells[:, :, 1], layout, L)
    assert torch.equal(v2, v_bc)


# ── 3. the term ───────────────────────────────────────────────────────────────

def test_all_train_step_charges_the_wta_term_and_reports_the_winners():
    m = _all_model().train()
    ids, inp, lab, layout = _batch()
    out = m(inp, labels=lab, slot_layout=layout)
    for k in ("fan_wta_ce", "fan_wta_weighted", "fan_wta_oracle_ce", "fan_wta_single_ce",
              "fan_wta_pick0", "fan_wta_forced", "fan_wta_share_k0", "fan_wta_share_k3",
              "fan_mix_entropy"):
        assert k in out, k
    assert float(out["fan_wta_forced"]) == 0.0
    assert float(out["fan_mix_entropy"]) == pytest.approx(math.log(4.0), abs=1e-5)
    shares = sum(float(out[f"fan_wta_share_k{i}"]) for i in range(4))
    assert shares == pytest.approx(1.0, abs=1e-5)
    # eps 0: the winner IS the argmin. The with-grad replay is NOT the oracle's number
    # exactly: the table scored slot s with stream i in EVERY slot, the replay writes each
    # slot its own winner, and the strict coda reads earlier slots' cells, so the two
    # differ by the cross-slot read (0.005 nats on this batch). Same coda, no dropout, so
    # they are close, and the replay never beats the table's per-slot minimum by much.
    assert abs(float(out["fan_wta_ce"]) - float(out["fan_wta_oracle_ce"])) < 0.05
    assert float(out["fan_wta_weighted"]) == pytest.approx(float(out["fan_wta_ce"]), rel=1e-6)
    assert "fan_select_gate_ce" not in out
    out["loss"].backward()
    assert m.tul_register is not None
    grads = [p.grad for p in m.tul_register.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0.0 for g in grads)


def test_all_loss_moves_by_lambda_times_the_term():
    ids, inp, lab, layout = _batch()
    torch.manual_seed(7)
    a = _all_model(fan_all_wta_lambda=1.0).train()
    out_a = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(7)
    b = _all_model(fan_all_wta_lambda=0.5).train()
    out_b = b(inp, labels=lab, slot_layout=layout)
    assert float(out_a["fan_wta_ce"]) == pytest.approx(float(out_b["fan_wta_ce"]), rel=1e-6)
    delta = float(out_a["loss"]) - float(out_b["loss"])
    assert delta == pytest.approx(0.5 * float(out_a["fan_wta_ce"]), rel=1e-4)


def test_all_eval_reports_the_oracle_and_no_gate_agreement():
    m = _all_model()
    ids, inp, lab, layout = _batch()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    for k in ("fan_mixed_ce", "fan_oracle_ce", "fan_single_ce", "fan_stream_ce_k0",
              "fan_stream_ce_k3", "fan_oracle_pick0"):
        assert k in out, k
    assert "fan_gate_agree" not in out
    assert "fan_wta_ce" not in out
    ces = [float(out[f"fan_stream_ce_k{i}"]) for i in range(4)]
    assert float(out["fan_oracle_ce"]) <= min(ces) + 1e-5
    assert float(out["fan_single_ce"]) == pytest.approx(ces[0], rel=1e-6)


def test_all_eps_one_forces_every_valid_slot():
    m = _all_model(fan_select_eps=0.999).train()
    ids, inp, lab, layout = _batch()
    torch.manual_seed(1)
    out = m(inp, labels=lab, slot_layout=layout)
    assert float(out["fan_wta_forced"]) > 0.9


# ── 5/6. refusals and keys ────────────────────────────────────────────────────

def test_all_refusals_and_the_knobs_it_reads():
    with pytest.raises(ValueError, match="prefix_k=4"):
        _tul(fan_k=4, slot_cells=4, prefix_k=2, fan_mix="all")
    with pytest.raises(ValueError, match="read only under"):
        _tul(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="mean", fan_all_wta_lambda=0.5)
    with pytest.raises(ValueError, match="read only under"):
        _tul(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_select_gate_lambda=0.5)
    with pytest.raises(ValueError, match="fan_all_wta_lambda"):
        _tul(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_all_wta_lambda=-1.0)
    _tul(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_select_eps=0.2,
         fan_all_wta_lambda=2.0)
    _tul(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="select", fan_select_eps=0.2)


def test_all_key_composes_through_tul_setup():
    from morph.training.tul_setup import KNOWN_TUL_KEYS
    assert "fan_all_wta_lambda" in KNOWN_TUL_KEYS
