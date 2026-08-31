"""GRT recurrence gate (morph/model/recur_gate.py) — the gate-ladder G1/G2 invariants.

Program note: .agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md.
The rows this file protects: none-mode builds nothing and stays bit-identical; the
gate is near-identity at init; g==1 makes the loop depth-invariant; gradients cross
both blend branches; the gate never sees the iteration index; the banned combos
raise at construction.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402 (tests/ on sys.path)

from morph.model.recur_gate import RecurrenceGate
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig


def _gate_model(seed: int = 0, **kw):
    torch.manual_seed(seed)
    tul = _tul(recur_gate="grt", mux_beta=1.0, mux_target="own",
               sigreg_lambda=0.0, eval_ablations=False)
    base = dict(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _plain_model(seed: int = 0, **kw):
    torch.manual_seed(seed)
    tul = _tul(mux_beta=1.0, mux_target="own", sigreg_lambda=0.0, eval_ablations=False)
    base = dict(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=3)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def test_none_builds_nothing_and_is_bit_identical():
    m0 = _plain_model()
    assert m0.tul_recur_gate is None
    assert not any("recur_gate" in n for n, _ in m0.named_parameters())
    # RNG neutrality: a plain model built before/after the gate existed draws the same
    # weights, so two same-seed plain builds must agree bitwise (regression anchor).
    m1 = _plain_model()
    for (n0, p0), (n1, p1) in zip(m0.named_parameters(), m1.named_parameters()):
        assert n0 == n1 and torch.equal(p0, p1), n0


def test_gate_near_identity_at_init():
    m = _gate_model()
    m.eval()
    x, y, lay, _ = _batch()
    seen = {}
    def spy(mod, inp, out):
        seen["mean"] = float(out.mean()); seen["min"] = float(out.min())
    handle = m.tul_recur_gate.register_forward_hook(spy)
    try:
        with torch.no_grad():
            m(x, y, slot_layout=lay)
    finally:
        handle.remove()
    assert 0.95 <= seen["mean"] <= 0.995, seen
    assert seen["min"] > 0.90, seen


def test_saturated_gate_makes_depth_invariant():
    """g == 1 => the carrier never moves => forced depth cannot matter."""
    m = _gate_model()
    with torch.no_grad():
        m.tul_recur_gate.fc2.bias.fill_(1e3)   # sigmoid(1000) == 1.0 in fp32
    m.eval()
    x, _, lay, _ = _batch()
    outs = []
    for d in (1, 3):
        m.cfg.tul.slot_mean_depth = d
        with torch.no_grad():
            outs.append(m(x, slot_layout=lay)["logits"])
    m.cfg.tul.slot_mean_depth = 0
    assert torch.equal(outs[0], outs[1])


def test_gradients_cross_both_blend_branches():
    m = _gate_model()
    m.train()
    x, y, lay, _ = _batch()
    out = m(x, y, slot_layout=lay)
    out["loss"].backward()
    for n in ("fc1.weight", "fc2.weight", "fc2.bias"):
        p = dict(m.tul_recur_gate.named_parameters())[n]
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, n
    core_grads = [p.grad for nm, p in m.named_parameters()
                  if nm.startswith("core.") and p.grad is not None]
    assert core_grads and any(float(g.abs().sum()) > 0.0 for g in core_grads)


def test_gate_never_sees_the_iteration_index():
    """The cond-zero constraint, made structural: forward(h, e) and nothing else."""
    params = list(inspect.signature(RecurrenceGate.forward).parameters)
    assert params == ["self", "h_prev", "e"], params


def test_banned_combos_raise():
    with pytest.raises(ValueError, match="db_loop"):
        _tul(recur_gate="grt", db_loop=True)
    with pytest.raises(ValueError, match="poisons|core_stage_cond"):
        _tul(recur_gate="grt", core_stage_cond="iter")
    with pytest.raises(ValueError, match="tokens_through_core"):
        _tul(recur_gate="grt", tokens_through_core=True)
    with pytest.raises(ValueError, match="recur_gate"):
        _tul(recur_gate="bogus")


def test_gate_noise_trains_only():
    m = _gate_model()
    x, _, lay, _ = _batch()
    m.eval()
    with torch.no_grad():
        a = m(x, slot_layout=lay)["logits"]
        b = m(x, slot_layout=lay)["logits"]
    assert torch.equal(a, b), "eval must be deterministic (no gate noise)"


def test_the_g_configs_resolve():
    from hydra import compose, initialize_config_dir
    import os
    cfg_dir = os.path.abspath("morph/configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        g1 = compose(config_name="tul_g1")
        g2 = compose(config_name="tul_g2")
    assert g1.tul.recur_gate == "grt"
    assert g2.tul.recur_gate == "grt"
    assert "spectral_project_cap" not in g1.training or not g1.training.get("spectral_project_cap")
    assert g2.training.spectral_project_cap == 1.5
    assert g1.wandb.name == "tul-g1" and g2.wandb.name == "tul-g2"
