"""Think-once panel knobs (branch tul/think-once): ``tul.cond_layers`` and ``tul.detach_z``.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_think_once.py -v

Drawing board: ``.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md``
(Revision 2026-09-03 evening, arms R7/R8). Contracts, each chosen so the test FAILS when
the shipped code breaks:

1. ``cond_layers > 0`` builds exactly that many extra blocks, drawn AFTER every shared
   parameter, so the shared weights are byte-identical to the ``cond_layers=0`` build
   from the same seed (the retention / TULSlots RNG-neutrality contract).
2. The stack's block parameters equal the block parameters of ``n_coda + N`` — the
   fairness contract behind R7 (cond4 + coda4) vs R8 (coda8).
3. The stack changes z (the coda's read and the mux loss both move) and the layer-pass
   accounting counts it once per REAL slot.
4. ``detach_z`` cuts the token CE's gradient into the loop AND the stack exactly (all
   None), while the mux local loss still reaches them and the reader side (W_prefix)
   still trains.
5. The undefined combinations RAISE (A2, db_loop) instead of silently running.
6. The sweep's paired bootstrap is token-weighted, contains its own point estimate,
   is deterministic, and collapses to zero width on identical inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer

N_COND = 2


def _loop_tul(**kw):
    """R4-shaped TUL settings on the tiny model: boundary seed, mux next, no mask."""
    base = dict(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
                mux_detach_head=False)
    base.update(kw)
    return _tul(**base)


def _model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = kw.pop("tul", None) or _loop_tul()
    base = dict(tul=tul, n_core=2, mean_depth=2, max_depth=2, bptt_depth=2,
                retention=False)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _forward(m: MORPHTransformer, seed: int = 7):
    x, y, layout, _ = _batch()
    torch.manual_seed(seed)          # pins the Poisson depth draw
    m.eval()                         # dropout off; graph still built
    return m(x, labels=y, slot_layout=layout), layout


# ── 1. RNG neutrality + 2. block-parameter parity ─────────────────────────────

def test_cond_stack_is_built_last_and_shared_weights_are_byte_identical():
    m0 = _model(seed=3)
    m4 = _model(seed=3, tul=_loop_tul(cond_layers=N_COND))
    assert m0.tul_cond is None and m4.tul_cond is not None
    assert len(m4.tul_cond) == N_COND
    sd0, sd4 = m0.state_dict(), m4.state_dict()
    extra = {k for k in sd4 if k not in sd0}
    assert extra and all(k.startswith("tul_cond.") for k in extra)
    for k, v in sd0.items():
        assert torch.equal(v, sd4[k]), f"shared parameter {k} changed"


def test_cond_block_parameters_equal_the_same_number_of_extra_coda_blocks():
    m_cond = _model(seed=3, tul=_loop_tul(cond_layers=N_COND))
    m_coda = _model(seed=3, n_coda=2 + N_COND)
    n_cond = sum(p.numel() for p in m_cond.tul_cond.parameters())
    n_extra_coda = sum(p.numel() for blk in list(m_coda.coda)[2:] for p in blk.parameters())
    assert n_cond == n_extra_coda > 0


# ── 3. the stack changes z, and the accounting sees it ───────────────────────

def test_cond_stack_changes_the_read_and_the_mux_loss_and_layer_passes():
    m0 = _model(seed=3)
    m4 = _model(seed=3, tul=_loop_tul(cond_layers=N_COND))
    # zero-init blocks would make the stack an identity; nudge it so the contract is
    # tested on a stack that actually computes something
    with torch.no_grad():
        for p in m4.tul_cond.parameters():
            p.add_(0.05 * torch.randn_like(p))
    o0, layout = _forward(m0)
    o4, _ = _forward(m4)
    assert not torch.allclose(o0["mux_local"], o4["mux_local"])
    assert not torch.allclose(o0["ce_main"], o4["ce_main"])
    n_valid = int(layout.slot_valid.sum())
    assert n_valid > 0
    assert float(o4["layer_passes"] - o0["layer_passes"]) == pytest.approx(N_COND * n_valid)


# ── 4. frozen z ───────────────────────────────────────────────────────────────

def _loop_and_stack_params(m: MORPHTransformer) -> list[torch.Tensor]:
    ps = [p for p in m.core.parameters() if p.requires_grad]
    if m.tul_cond is not None:
        ps += [p for p in m.tul_cond.parameters() if p.requires_grad]
    return ps


def test_detach_z_cuts_the_token_ce_gradient_into_loop_and_stack():
    # mux off: the token CE is the ONLY loss, so any gradient into the loop or the
    # stack can only come through the coda's read of z
    m = _model(seed=5, tul=_loop_tul(cond_layers=N_COND, mux_beta=0.0, detach_z=True))
    with torch.no_grad():
        for p in m.tul_cond.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out, _ = _forward(m)
    grads = torch.autograd.grad(out["loss"], _loop_and_stack_params(m), allow_unused=True)
    assert all(g is None for g in grads), "detach_z leaked token-CE gradient into the loop"
    # the reader side still trains: W_prefix sits AFTER the detach
    out2, _ = _forward(m)
    (gw,) = torch.autograd.grad(out2["loss"], [m.tul.W_prefix], allow_unused=True)
    assert gw is not None and float(gw.abs().sum()) > 0.0


def test_without_detach_the_token_ce_reaches_loop_and_stack():
    m = _model(seed=5, tul=_loop_tul(cond_layers=N_COND, mux_beta=0.0, detach_z=False))
    with torch.no_grad():
        for p in m.tul_cond.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out, _ = _forward(m)
    grads = torch.autograd.grad(out["loss"], _loop_and_stack_params(m), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_detach_z_keeps_the_mux_gradient_into_loop_and_stack():
    m = _model(seed=5, tul=_loop_tul(cond_layers=N_COND, mux_beta=1.0, detach_z=True))
    with torch.no_grad():
        for p in m.tul_cond.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out, _ = _forward(m)
    grads = torch.autograd.grad(out["loss"], _loop_and_stack_params(m), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


# ── 5. undefined combinations raise ───────────────────────────────────────────

def test_detach_z_with_tokens_through_core_raises():
    with pytest.raises(ValueError, match="detach_z"):
        _loop_tul(detach_z=True, tokens_through_core=True)


def test_cond_layers_with_db_loop_raises():
    with pytest.raises(NotImplementedError, match="cond_layers"):
        _model(seed=1, tul=_loop_tul(cond_layers=1, db_loop=True, mux_target="own"))


def test_cond_layers_with_tokens_through_core_raises():
    with pytest.raises(NotImplementedError, match="cond_layers"):
        _model(seed=1, tul=_loop_tul(cond_layers=1, tokens_through_core=True, mux_beta=0.0))


# ── 6. the paired bootstrap ───────────────────────────────────────────────────

def test_paired_bootstrap_is_token_weighted_deterministic_and_covers_its_point():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lab", "divergence", "_stats.py")
    spec = importlib.util.spec_from_file_location("_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    cnt = rng.integers(50, 200, size=40).astype(float)
    a = cnt * 4.0 + rng.normal(0, 3, size=40)
    b = cnt * 3.9 + rng.normal(0, 3, size=40)
    r1 = mod.paired_bootstrap_ci(a, b, cnt, n_boot=500, seed=1)
    r2 = mod.paired_bootstrap_ci(a, b, cnt, n_boot=500, seed=1)
    assert r1 == r2
    assert r1["point"] == pytest.approx(a.sum() / cnt.sum() - b.sum() / cnt.sum())
    assert r1["lo"] <= r1["point"] <= r1["hi"]
    same = mod.paired_bootstrap_ci(a, a, cnt, n_boot=200, seed=2)
    assert same["point"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0
    with pytest.raises(ValueError):
        mod.paired_bootstrap_ci(a[:3], b[:4], cnt[:3])
