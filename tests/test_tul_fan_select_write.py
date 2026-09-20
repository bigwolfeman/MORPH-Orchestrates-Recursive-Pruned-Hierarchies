"""``tul.fan_select_write`` — the select arm writes the GATE'S pick at train (2026-09-20).

WHAT THIS FILE HAS TO PROVE:

1. ``oracle`` (the filed arm) IS UNTOUCHED: the default draws no RNG, reports no write
   stats, and a train forward under it is bit-identical to the pre-knob select model.
2. ``gate`` WRITES THE GATE'S ARGMAX: with a zero-init gate every slot writes stream 0,
   so the written CE equals the table's stream-0 CE, ``from_gate`` is 1 on every valid
   slot, and the gate is still trained on the TABLE's winner (its CE reads ln K at step 0,
   its weight gets gradient). A forced (eps) slot keeps its random stream.
3. ``anneal`` FOLLOWS THE STEP BUFFER: step 0 writes the winner (``p_gate`` 0, written CE
   equals the oracle CE), a mid step writes a mixture, a step past the horizon writes the
   gate on every valid slot.
4. REFUSALS: the write knob is refused off ``select``, a bad value is refused, the anneal
   horizon is refused off ``anneal`` and at 0.
5. The two new keys compose through ``tul_setup`` and the shipped config resolves to
   ``anneal`` over 1500 steps.

CPU only, fp32, the ``tests/test_tul_fan.py`` fixtures.
Record: lab/experiments/planned/2026-09-20-lxtul-fan4-select-gate.md
"""
from __future__ import annotations

import math

import pytest
import torch

from test_tul_fan import _batch, _model, _tul


def _sel(**kw):
    return _model(fan_k=4, slot_cells=4, fan_mix="select", **kw)


def _train_forward(m, seed: int = 7):
    ids, inp, lab, layout = _batch()
    torch.manual_seed(seed)
    return m.train()(inp, labels=lab, slot_layout=layout)


# ── 1. oracle is untouched ───────────────────────────────────────────────────

def test_oracle_write_reports_no_write_stats_and_matches_the_default():
    a = _train_forward(_sel(fan_select_eps=0.0))
    b = _train_forward(_sel(fan_select_eps=0.0, fan_select_write="oracle"))
    assert "fan_select_write_p_gate" not in a and "fan_select_written_ce" not in a
    assert float(a["loss"]) == float(b["loss"])
    assert float(a["fan_select_gate_ce"]) == float(b["fan_select_gate_ce"])


# ── 2. gate writes the gate's argmax ─────────────────────────────────────────

def test_gate_write_is_stream_zero_under_a_zero_gate_and_the_gate_still_learns():
    m = _sel(fan_select_eps=0.0, fan_select_write="gate")
    out = _train_forward(m)
    assert float(out["fan_select_write_p_gate"]) == 1.0
    assert float(out["fan_select_write_from_gate"]) == 1.0
    # the zero-init gate ties every logit; argmax is 0 in every slot, so the written
    # stream IS stream 0 and its table CE is the table's stream-0 column
    assert float(out["fan_select_written_ce"]) == pytest.approx(
        float(out["fan_select_single_ce"]), abs=1e-6)
    assert float(out["fan_select_written_agree"]) == pytest.approx(
        float(out["fan_select_pick0"]), abs=1e-6)
    assert float(out["fan_select_written_share_k0"]) == 1.0
    assert sum(float(out[f"fan_select_written_share_k{i}"]) for i in range(4)) == pytest.approx(1.0, abs=1e-6)
    # the gate's label is still the table's winner: ln K at step 0, gradient on its weight
    assert float(out["fan_select_gate_ce"]) == pytest.approx(math.log(4.0), abs=1e-5)
    out["loss"].backward()
    assert float(m.tul_fan.gate.weight.grad.abs().sum()) > 0.0


def test_gate_write_differs_from_the_oracle_write_in_the_loss():
    """Same seed, same table: the oracle arm writes the winner, the gate arm writes
    stream 0, and the coda's CE (hence the loss) differs whenever the winner is not 0."""
    a = _train_forward(_sel(fan_select_eps=0.0))
    b = _train_forward(_sel(fan_select_eps=0.0, fan_select_write="gate"))
    assert float(a["fan_select_pick0"]) < 1.0, "the fixture has a slot whose winner is not 0"
    assert float(a["fan_select_oracle_ce"]) == pytest.approx(float(b["fan_select_oracle_ce"]), abs=1e-6)
    assert float(a["loss"]) != float(b["loss"])


def test_gate_write_keeps_a_forced_slot_random():
    m = _sel(fan_select_eps=0.999, fan_select_write="gate")
    out = _train_forward(m)
    assert float(out["fan_select_forced"]) > 0.9
    assert float(out["fan_select_write_from_gate"]) < 0.1
    # streams other than 0 were written (a zero gate alone would write 0 everywhere)
    assert sum(float(out[f"fan_select_share_k{i}"]) for i in (1, 2, 3)) > 0.5


# ── 3. anneal follows the step buffer ────────────────────────────────────────

def test_anneal_write_moves_from_the_winner_to_the_gate_with_the_step():
    m = _sel(fan_select_eps=0.0, fan_select_write="anneal", fan_select_write_anneal=1000)
    m.fan_select_step.fill_(0)
    o0 = _train_forward(m)
    assert float(o0["fan_select_write_p_gate"]) == 0.0
    assert float(o0["fan_select_write_from_gate"]) == 0.0
    assert float(o0["fan_select_written_ce"]) == pytest.approx(float(o0["fan_select_oracle_ce"]), abs=1e-6)
    assert float(o0["fan_select_written_agree"]) == 1.0
    m.fan_select_step.fill_(500)
    o5 = _train_forward(m)
    assert float(o5["fan_select_write_p_gate"]) == 0.5
    assert 0.0 < float(o5["fan_select_write_from_gate"]) < 1.0
    m.fan_select_step.fill_(5000)
    o9 = _train_forward(m)
    assert float(o9["fan_select_write_p_gate"]) == 1.0
    assert float(o9["fan_select_write_from_gate"]) == 1.0
    assert float(o9["fan_select_written_ce"]) == pytest.approx(float(o9["fan_select_single_ce"]), abs=1e-6)


def test_eval_write_is_the_gate_argmax_under_every_mode():
    ids, inp, lab, layout = _batch()
    outs = []
    for mode in ("oracle", "gate", "anneal"):
        m = _sel(fan_select_eps=0.0, fan_select_write=mode).eval()
        with torch.no_grad():
            outs.append(float(m(inp, labels=lab, slot_layout=layout)["loss"]))
    assert outs[0] == outs[1] == outs[2]


# ── 4/5. refusals and keys ───────────────────────────────────────────────────

def test_write_knobs_are_refused_off_select_and_validated():
    with pytest.raises(ValueError, match="read only under"):
        _tul(fan_k=4, slot_cells=4, fan_mix="softmax", fan_select_write="gate")
    with pytest.raises(ValueError, match="read only under"):
        _tul(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_select_write="anneal")
    with pytest.raises(ValueError, match="fan_select_write must be"):
        _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_write="random")
    with pytest.raises(ValueError, match="fan_select_write_anneal is read only"):
        _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_write="gate",
             fan_select_write_anneal=300)
    with pytest.raises(ValueError, match="fan_select_write_anneal must be"):
        _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_write="anneal",
             fan_select_write_anneal=0)
    _tul(fan_k=4, slot_cells=4, fan_mix="select", fan_select_write="anneal",
         fan_select_write_anneal=300)


def test_write_keys_compose_through_tul_setup_and_the_config_resolves():
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import KNOWN_TUL_KEYS
    import os
    assert "fan_select_write" in KNOWN_TUL_KEYS and "fan_select_write_anneal" in KNOWN_TUL_KEYS
    cdir = os.path.abspath("morph/configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name="tul_slot_spandec_strict_fan4_select_gate")
    assert cfg.tul.fan_mix == "select"
    assert cfg.tul.fan_select_write == "anneal"
    assert int(cfg.tul.fan_select_write_anneal) == 1500
    assert float(cfg.tul.fan_select_eps) == 0.05
    assert cfg.wandb.name == "slot-spandec-strict-fan4-select-gate"
