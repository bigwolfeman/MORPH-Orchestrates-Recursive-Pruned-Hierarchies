"""Staged targets (tul.mux_stage_own_iters, arc E3): the state after iteration k is
supervised toward the span the slot terminates ("own"), the final state toward the
forecast; the carry stays live. Contract, one test per line:
  * off (0) keeps no trajectory; on keeps T+1 states with the carry live;
  * the reported local loss is EXACTLY 0.5·(own on traj[k], slots with depth ≥ k)
    + 0.5·(forecast on the final state), and eval exposes both targets on the final state;
  * the own term's gradient reaches the slot path;
  * the config refuses the combinations that have no meaning; k past the loop raises."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_tul_mux import _batch, _loss, _model, _spec  # noqa: E402

from morph.model.tul import TULConfig  # noqa: E402


def _stage_model(k: int, seed: int = 1234, **kw):
    return _model(TULConfig(slot_id=4, mux_beta=1.0, mux_stage_own_iters=k, **kw), seed=seed)


def test_off_keeps_no_trajectory_and_on_keeps_a_live_one():
    x, y, layout, _ = _batch(_spec())
    m0 = _model(TULConfig(slot_id=4, mux_beta=1.0))
    m2 = _stage_model(2)
    for m in (m0, m2):
        m.eval()
    xx, x0, bg = m0._tul_front(x, layout)
    assert m0._tul_core(xx, x0, bg, layout)[4] is None
    xx, x0, bg = m2._tul_front(x, layout)
    traj = m2._tul_core(xx, x0, bg, layout)[4]
    assert traj is not None and len(traj) == m2.cfg.mean_depth + 1   # eval: uniform fill
    # live carry: the final state's graph reaches the seed THROUGH iteration 1's state
    m2.train()
    xx, x0, bg = m2._tul_front(x, layout)
    _xn, _h, depths, _g, traj, _gr = m2._tul_core(xx, x0, bg, layout)
    assert len(traj) == int(depths.max()) + 1                          # train: deepest slot
    g = torch.autograd.grad(traj[-1].float().pow(2).sum(), traj[1], retain_graph=True,
                            allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0.0


def test_local_loss_is_the_mean_of_the_two_staged_terms():
    x, y, layout, _ = _batch(_spec())
    m = _stage_model(2)
    m.eval()
    torch.manual_seed(7)
    out = _loss(m, x, y, layout)
    torch.manual_seed(7)
    xx, x0, bg = m._tul_front(x, layout)
    _xn, h, depths, _g, traj, _gr = m._tul_core(xx, x0, bg, layout)
    own = m._tul_mux_loss(traj[2], x, layout, slot_keep=(depths >= 2), target="own")
    nxt = m._tul_mux_loss(h, x, layout)
    assert torch.allclose(out["mux_local"], 0.5 * (own + nxt), atol=1e-6)
    assert torch.allclose(out["mux_stage_own"], own, atol=1e-6)
    assert torch.allclose(out["mux_stage_next"], nxt, atol=1e-6)
    # eval exposes both targets on the FINAL state (the depth sweep's columns)
    own_final = m._tul_mux_loss(h, x, layout, target="own")
    assert torch.allclose(out["mux_local_own_final"], own_final, atol=1e-6)
    assert torch.allclose(out["mux_local_next_final"], nxt, atol=1e-6)
    assert float(out["mux_n_supervised_own"]) > 0 and float(out["mux_n_supervised_next"]) > 0


def test_training_forward_reports_no_final_columns():
    x, y, layout, _ = _batch(_spec())
    m = _stage_model(2)
    m.train()
    out = m(x, labels=y, bag_size=0, slot_layout=layout)
    assert "mux_local_own_final" not in out and "mux_stage_own" in out


def test_own_term_gradient_reaches_the_slot_path():
    x, y, layout, _ = _batch(_spec())

    def grad(k):
        m = _model(TULConfig(slot_id=4, mux_beta=1.0, mux_stage_own_iters=k), seed=1234)
        torch.manual_seed(7)
        out = _loss(m, x, y, layout)
        out["loss"].backward()
        return m.tul.E_slot.grad.clone()

    g0a, g0b, g2 = grad(0), grad(0), grad(2)
    assert torch.equal(g0a, g0b), "RNG pinning broken — the comparison below is void"
    assert not torch.allclose(g0a, g2), "the staged own term never reached E_slot"


@pytest.mark.parametrize("kw", [dict(mux_beta=0.0), dict(db_loop=True),
                                dict(tokens_through_core=True)])
def test_config_refuses_undefined_combinations(kw):
    base = dict(slot_id=4, mux_beta=1.0, mux_stage_own_iters=2)
    base.update(kw)
    with pytest.raises(ValueError):
        TULConfig(**base)


def test_stage_past_the_loop_raises_at_construction():
    with pytest.raises(ValueError, match="exceeds the loop"):
        _stage_model(5)          # max_depth 3 in the tiny config


def test_stage_at_the_last_iteration_is_allowed_and_clamps_in_a_shallow_batch():
    x, y, layout, _ = _batch(_spec())
    m = _stage_model(3)          # == max_depth: legal; eval fills depth 2 < 3
    m.eval()
    out = _loss(m, x, y, layout)
    assert float(out["mux_stage_own"]) == 0.0 and torch.isfinite(out["mux_local"])
