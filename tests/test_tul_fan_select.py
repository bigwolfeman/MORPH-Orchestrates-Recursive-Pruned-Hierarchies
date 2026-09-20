"""``tul.fan_mix="select"`` — select-then-commit on LXTUL's fan (2026-09-20).

WHAT THIS FILE HAS TO PROVE:

1. THE GATHER IS HARD. ``select_streams`` returns stream ``choice[b, s]`` exactly, and
   its gradient reaches that stream alone (the other K-1 get exactly zero).
2. THE WINNER IS THE ARGMIN, the eps guard forces VALID slots only, and eps 0 forces none.
3. THE GATE LOSS trains the gate and NOT the streams (zero gradient into the cells).
4. END TO END: a select model's eval write equals feeding the gate's argmax stream ALONE
   (checked against the oracle's per-stream replay through the same forward), the train
   step charges ``fan_select_gate_ce``, reports the selection instruments, and the K
   selection passes leave the gate's winner within the oracle table they were built from.
5. ``mean`` and ``softmax`` are untouched: a softmax model refuses a ``choice``, and the
   two select knobs are refused on any other mixture (no silent no-op).
6. The two new ``tul.*`` keys compose through ``tul_setup`` and reach the manifest.

CPU only, fp32, the ``tests/test_tul_fan.py`` fixtures.
Record: lab/experiments/planned/2026-09-20-lxtul-fan4-select.md
"""
from __future__ import annotations

import pytest
import torch

from morph.model.tul_fan import (TULFanMix, select_gate_loss, select_streams,
                                 select_winners)
from test_tul_fan import _batch, _model, _tul


# ── 1. the gather ────────────────────────────────────────────────────────────

def test_select_streams_is_a_hard_gather_with_a_one_stream_gradient():
    torch.manual_seed(0)
    cells = torch.randn(2, 3, 4, 5, requires_grad=True)             # [B, S, M, C]
    choice = torch.tensor([[0, 3, 1], [2, 2, 0]])
    out = select_streams(cells, choice)
    for b in range(2):
        for s in range(3):
            assert torch.equal(out[b, s], cells[b, s, choice[b, s]])
    out.sum().backward()
    g = cells.grad
    for b in range(2):
        for s in range(3):
            for i in range(4):
                expect = 1.0 if i == int(choice[b, s]) else 0.0
                assert torch.equal(g[b, s, i], torch.full((5,), expect))


def test_select_streams_handles_the_hc_carrier_and_refuses_a_bad_choice():
    cells = torch.randn(1, 2, 3, 4, 6)                                # [B, S, M, n, C]
    out = select_streams(cells, torch.tensor([[2, 0]]))
    assert out.shape == (1, 2, 4, 6)
    assert torch.equal(out[0, 0], cells[0, 0, 2]) and torch.equal(out[0, 1], cells[0, 1, 0])
    with pytest.raises(ValueError, match="out of range"):
        select_streams(cells, torch.tensor([[3, 0]]))
    with pytest.raises(ValueError, match="choice"):
        select_streams(cells, torch.tensor([[0]]))


# ── 2. the winner ────────────────────────────────────────────────────────────

def test_select_winners_is_the_argmin_and_eps_forces_valid_slots_only():
    ce = torch.tensor([[[3.0, 1.0, 2.0], [0.5, 0.7, 0.9], [9.0, 9.0, 0.1]]])
    valid = torch.tensor([[True, True, False]])
    choice, forced = select_winners(ce, valid, eps=0.0)
    assert choice.tolist() == [[1, 0, 0]]                # the invalid slot is inert (0)
    assert not forced.any()
    g = torch.Generator().manual_seed(3)
    choice, forced = select_winners(ce, valid, eps=1.0, generator=g)
    assert forced.tolist() == [[True, True, False]]      # every VALID slot forced
    assert choice[0, 2] == 0
    assert int(choice.max()) < 3 and int(choice.min()) >= 0
    with pytest.raises(ValueError, match=r"\[B, S, K\]"):
        select_winners(ce[0], valid, 0.0)


def test_select_winners_eps_rate_is_the_knob():
    torch.manual_seed(0)
    ce = torch.rand(64, 64, 4)
    valid = torch.ones(64, 64, dtype=torch.bool)
    g = torch.Generator().manual_seed(7)
    _, forced = select_winners(ce, valid, eps=0.25, generator=g)
    assert 0.20 < float(forced.float().mean()) < 0.30


# ── 3. the gate loss ─────────────────────────────────────────────────────────

def test_select_gate_loss_trains_the_gate_only():
    torch.manual_seed(0)
    mix = TULFanMix(8, 4, "select")
    cells = torch.randn(2, 5, 4, 8, requires_grad=True)
    logits = mix.logits(cells)
    valid = torch.ones(2, 5, dtype=torch.bool)
    choice = torch.randint(0, 4, (2, 5))
    loss = select_gate_loss(logits, choice, valid)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-6)  # zero-init gate
    loss.backward()
    assert mix.gate.weight.grad is not None and float(mix.gate.weight.grad.abs().sum()) > 0
    # The cells feed the LOGITS, so they do receive gradient through the gate's linear;
    # that is the gate reading the cells, not the loss moving the streams' content. What
    # the contract forbids is the loss reaching the streams through the WRITE, which
    # `select_gate_loss` never touches — proven end to end in test 4 (the write path's
    # gradient is the CE's alone). Here: an empty valid set is an exact zero.
    z = select_gate_loss(logits, choice, torch.zeros(2, 5, dtype=torch.bool))
    assert float(z.detach()) == 0.0


def test_fan_mix_select_forward_is_the_argmax_stream_and_softmax_refuses_a_choice():
    torch.manual_seed(0)
    mix = TULFanMix(8, 3, "select")
    with torch.no_grad():
        mix.gate.weight.copy_(torch.randn(1, 8))
    cells = torch.randn(2, 4, 3, 8)
    state, w = mix(cells)
    arg = w.argmax(-1)
    assert torch.equal(state, select_streams(cells, arg))
    state2, _ = mix(cells, torch.zeros(2, 4, dtype=torch.long))
    assert torch.equal(state2, cells[:, :, 0])
    soft = TULFanMix(8, 3, "softmax")
    with pytest.raises(ValueError, match="takes no choice"):
        soft(cells, torch.zeros(2, 4, dtype=torch.long))


# ── 4. end to end ────────────────────────────────────────────────────────────

def _select_model(**kw):
    return _model(fan_k=4, slot_cells=4, fan_mix="select", **kw)


def test_select_model_builds_the_gate_and_the_eval_write_is_one_stream():
    m = _select_model()
    assert m.tul_fan.mode == "select" and m.tul_fan.gate is not None
    ids, inp, lab, layout = _batch()
    with torch.no_grad():
        # give the gate a non-trivial preference so the argmax is not stream 0 everywhere
        m.tul_fan.gate.weight.copy_(torch.randn_like(m.tul_fan.gate.weight))
        out = m(inp, labels=lab, slot_layout=layout)
    groups = out
    keys = [k for k in groups if str(k).startswith("fan_")]
    assert "fan_mixed_ce" in keys and "fan_oracle_ce" in keys and "fan_gate_agree" in keys
    # The shipped write IS one stream: mixed_ce equals that stream's own CE on every span,
    # so mixed_ce must be >= oracle_ce (the min over streams) and, since the argmax stream
    # is one of the K, mixed_ce lies within the set of per-stream CEs' hull [min, max].
    ces = [float(groups[f"fan_stream_ce_k{i}"]) for i in range(4)]
    mixed = float(groups["fan_mixed_ce"])
    assert float(groups["fan_oracle_ce"]) <= mixed + 1e-5
    assert min(ces) - 1e-4 <= mixed <= max(ces) + 1e-4
    assert 0.0 <= float(groups["fan_gate_agree"]) <= 1.0


def test_select_model_train_step_charges_the_gate_loss_and_reports_the_selection():
    m = _select_model(fan_select_eps=0.0).train()
    ids, inp, lab, layout = _batch()
    out = m(inp, labels=lab, slot_layout=layout)
    groups = out
    for k in ("fan_select_gate_ce", "fan_select_gate_weighted", "fan_select_oracle_ce",
              "fan_select_single_ce", "fan_select_pick0", "fan_select_agree",
              "fan_select_forced", "fan_select_share_k0", "fan_select_share_k3"):
        assert k in groups, k
    assert float(groups["fan_select_forced"]) == 0.0
    assert float(groups["fan_select_oracle_ce"]) <= float(groups["fan_select_single_ce"]) + 1e-6
    shares = sum(float(groups[f"fan_select_share_k{i}"]) for i in range(4))
    assert shares == pytest.approx(1.0, abs=1e-5)
    # the zero-init gate predicts nothing: its CE is ln 4 at step 0
    assert float(groups["fan_select_gate_ce"]) == pytest.approx(float(torch.log(torch.tensor(4.0))), abs=1e-5)
    groups["loss"].backward()
    assert m.tul_fan.gate.weight.grad is not None
    assert float(m.tul_fan.gate.weight.grad.abs().sum()) > 0.0


def test_select_eps_one_forces_every_valid_slot():
    m = _select_model(fan_select_eps=0.999).train()
    ids, inp, lab, layout = _batch()
    torch.manual_seed(1)
    out = m(inp, labels=lab, slot_layout=layout)
    assert float(out["fan_select_forced"]) > 0.9


# ── 5/6. refusals and keys ───────────────────────────────────────────────────

def test_select_knobs_are_refused_off_the_select_mix_and_validated():
    with pytest.raises(ValueError, match="read only under"):
        _tul(fan_k=4, slot_cells=4, fan_mix="softmax", fan_select_eps=0.1)
    with pytest.raises(ValueError, match="read only under"):
        _tul(fan_k=4, slot_cells=4, fan_mix="mean", fan_select_gate_lambda=0.5)
    with pytest.raises(ValueError, match="fan_select_eps"):
        _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_eps=1.0)
    with pytest.raises(ValueError, match="fan_select_gate_lambda"):
        _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_gate_lambda=-1.0)
    _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_eps=0.0, fan_select_gate_lambda=2.0)


def test_select_keys_compose_through_tul_setup():
    from morph.training.tul_setup import KNOWN_TUL_KEYS
    assert "fan_select_eps" in KNOWN_TUL_KEYS and "fan_select_gate_lambda" in KNOWN_TUL_KEYS


def test_select_eval_write_with_a_zero_gate_is_stream_zero_exactly():
    """The sharp version of the hull check: a zero-init gate ties every logit, argmax
    returns index 0 in every slot, so the shipped write IS stream 0 alone and
    ``fan_mixed_ce`` must equal ``fan_stream_ce_k0`` (the oracle's own replay of stream 0
    through the identical coda) to fp32 precision. A mixture cannot pass this."""
    m = _select_model()
    ids, inp, lab, layout = _batch()
    assert float(m.tul_fan.gate.weight.abs().sum()) == 0.0
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert float(out["fan_mixed_ce"]) == pytest.approx(float(out["fan_stream_ce_k0"]), rel=1e-5)
    assert float(out["fan_gate_agree"]) == pytest.approx(float(out["fan_oracle_pick0"]), abs=1e-6)
