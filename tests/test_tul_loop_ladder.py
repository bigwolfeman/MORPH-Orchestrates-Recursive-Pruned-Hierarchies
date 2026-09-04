"""Loop-ladder arms (L1/L2/L3/L4) — the loop restored on the GL winner.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_loop_ladder.py -v

Prereg: ``lab/experiments/planned/2026-08-29-tul-loop-ladder.md``. The db_loop
contract (arm L3) is: the loop runs in the FORWARD, but no gradient ever crosses
an iteration boundary — the carry and the retention state are detached — while
the seed injection stays live so every loss reaches the write through exactly ONE
core application. These tests enshrine that contract and the resolved configs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402 (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer


def _db_model(seed: int = 0, db: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(db_loop=db, db_mux_iters=3, mux_beta=1.0, mux_target="own",
               sigreg_lambda=0.0)
    # retention_layers=(2,) puts the GLA retention INSIDE the core (prelude is layers
    # 0-1), so the cross-iteration retention carry exists and the detach is exercised.
    base = dict(tul=tul, n_core=2, mean_depth=3, max_depth=3, bptt_depth=1,
                retention_layers=(2,))
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


# ── resolved configs (the gl1b mux_target lesson: test the RESOLVED values) ──

def test_the_ladder_configs_resolve():
    from hydra import compose, initialize_config_dir
    import os
    cdir = os.path.abspath("morph/configs")
    want = {
        "tul_l1": {("model", "n_core"): 6, ("model", "bptt_depth"): 8,
                   ("model", "max_depth"): 8, ("tul", "mux_beta"): 1.0,
                   ("tul", "tg_restrict"): True, ("tul", "db_loop", None): None},
        "tul_l2": {("training", "spectral_project_cap"): 1.5, ("model", "n_core"): 6},
        "tul_l3": {("tul", "db_loop"): True, ("tul", "db_mux_iters"): 4,
                   ("model", "bptt_depth"): 8},
    }
    with initialize_config_dir(config_dir=cdir, version_base=None):
        for name, checks in want.items():
            cfg = compose(config_name=name)
            for key, val in checks.items():
                if len(key) == 3:      # key absent → OmegaConf .get default
                    assert cfg[key[0]].get(key[1]) is val
                else:
                    assert cfg[key[0]][key[1]] == val, (name, key)
        # L4 is tul_l1 + a CLI override, not a file: the override must resolve.
        cfg = compose(config_name="tul_l1", overrides=["training.optimizer=adamw"])
        assert cfg.training.optimizer == "adamw"
        # full BPTT means bptt_depth covers every realizable depth
        cfg1 = compose(config_name="tul_l1")
        assert cfg1.model.bptt_depth >= cfg1.model.max_depth


# ── db_loop: the gradient contract ───────────────────────────────────────────

def test_db_loop_no_gradient_crosses_an_iteration_boundary():
    """h_final w.r.t. any earlier trajectory state is EXACTLY no-graph (None): in eval
    the depth is uniform, every slot updates every iteration, and the carry into each
    core application is detached. One leak here and L3 is L1 with extra steps."""
    x, _y, lay, _ = _batch()
    m = _db_model()
    m.eval()
    xf, x0, bg = m._tul_front(x, lay)
    xf = xf.requires_grad_(True)
    _xn, h, _d, _g, traj, _gr = m._tul_core(xf, x0, bg, lay)
    assert traj is not None and len(traj) == 4          # seed + 3 iterations
    # The contract forbids gradient through the CORE MAP across iterations. A slot
    # frozen at depth d keeps traj[d] as its final state via the where-carry, and the
    # IDENTITY gradient of that carry is by design (it is how the final state of a
    # shallow slot stays supervised). In eval the real slots run the full uniform
    # depth, so for THEM any nonzero gradient to an earlier state is a leak. PAD
    # slots freeze at depth 1 and carry the identity — excluded.
    vm = lay.slot_valid.view(*lay.slot_valid.shape, *([1] * (h.dim() - 2)))
    for t in range(len(traj) - 1):
        g = torch.autograd.grad(h.sum(), traj[t], retain_graph=True,
                                allow_unused=True)[0]
        if g is None:
            continue
        leak = float((g * vm.to(g.dtype)).abs().sum())
        assert leak == 0.0, f"core-map gradient crossed the boundary at t={t}: {leak}"


def test_db_loop_every_iteration_carries_grad_despite_bptt_depth():
    """bptt_depth=1 would freeze iterations 0-1 under no_grad in the normal loop; under
    db_loop n_nograd is forced to 0 (a frozen iteration would silently drop its local
    loss). Every trajectory state after the seed must be in a graph."""
    x, _y, lay, _ = _batch()
    m = _db_model()          # bptt_depth=1, max_depth=3
    m.train()
    xf, x0, bg = m._tul_front(x, lay)
    _xn, _h, _d, _g, traj, _gr = m._tul_core(xf, x0, bg, lay)
    for t in range(1, len(traj)):
        assert traj[t].grad_fn is not None, f"iteration {t} lost its graph"


def test_db_loop_forward_value_is_bitwise_the_undetached_loop():
    """detach() changes gradients, never values: at uniform (eval) depth the db model's
    logits equal the plain looped model's exactly, weight for weight."""
    x, _y, lay, _ = _batch()
    a = _db_model(seed=3, db=True)
    b = _db_model(seed=3, db=False)
    b.load_state_dict(a.state_dict())
    a.eval(); b.eval()
    with torch.no_grad():
        la = a(x, slot_layout=lay)["logits"]
        lb = b(x, slot_layout=lay)["logits"]
    assert torch.equal(la, lb)


def test_db_loop_ce_still_reaches_the_write():
    """The GL mechanism must survive the loop: a later span's CE reaches W_sent (the
    boundary tap) through the live seed injection of the final core application."""
    x, y, lay, _ = _batch()
    m = _db_model()
    m.train()
    out = m(x, y, slot_layout=lay)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert float(m.tul.W_sent.weight.grad.abs().sum()) > 0


def test_db_loop_local_mux_runs_and_reports_its_iterations():
    x, y, lay, _ = _batch()
    m = _db_model()
    m.train()
    out = m(x, y, slot_layout=lay)
    assert out.get("mux_local") is not None and torch.isfinite(out["mux_local"])
    assert float(out["mux_db_n_iters"]) == 3.0


def test_db_loop_under_scse_raises():
    """The carry under SCSE is the DEVIATION; detaching it detaches h* reconstruction.
    Explicitly not defined — a silent wrong graph is worse than a raise."""
    x, _y, lay, _ = _batch()
    m = _db_model(scse_enable=True, core_init_scale=0.1) if False else None
    # SCSE construction differs per config; assert at the seam instead: a model whose
    # scse attr is not None must raise inside _tul_core.
    mm = _db_model()
    mm.scse = object.__new__(type("FakeSCSE", (), {}))   # non-None sentinel
    xf, x0, bg = mm._tul_front(x, lay)
    with pytest.raises(NotImplementedError, match="db_loop"):
        mm._tul_core(xf, x0, bg, lay)
