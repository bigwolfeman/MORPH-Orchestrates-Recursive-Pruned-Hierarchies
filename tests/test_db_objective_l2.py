"""Option A — L2 embedding denoising (the paper's AR objective). CPU only.

The point of the objective (checklist, "The objective fork"): under EDM the network's
raw output regresses a unit-variance target at EVERY σ, so no σ decodes for free. The
CE+tied-readout path (Option B) gives CE≈0 at low σ at INIT — ``c_skip·z ≈ y`` passes
straight through the tied head — which is the free ride that collapsed the killed run.
Under L2 there is no readout in training, so the free ride cannot occur by construction.
These tests pin that, plus the plumbing around it.

Pattern follows ``tests/test_db_forward.py``: tiny config, ``use_kernels=False``, no CUDA.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from morph.model.diffusion_blocks import DBConfig, DBStep
from morph.training.db_setup import build_db_step, db_loss

from test_db_forward import _model, _RT, V

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Low → high. Low σ is where the CE free ride lives; the L2 loss must be O(1) everywhere.
SIGMA_GRID = [0.05, 0.3, 1.0, 3.0, 9.79]


def _fixed_sigma_step(m, rt, labels: torch.Tensor, sv: float,
                      seed: int = 0) -> DBStep:
    """A DBStep at one FIXED σ (build_db_step samples σ; the probes need a grid)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        y = m.embed(labels)
        scaler = getattr(m, "db_scaler", None)
        if scaler is not None:
            y = scaler(y)
    B = labels.shape[0]
    sigma = torch.full((B,), sv)
    eps = torch.randn(y.shape, dtype=torch.float32, generator=g)
    z = (y.float() + sigma.view(B, 1, 1) * eps).to(y.dtype)
    block = int(rt.schedule.block_of_sigma(sigma[:1])[0])
    return DBStep(block_idx=block, sigma=sigma, z_noisy=z, y_clean=y, labels=labels)


# ── validation ───────────────────────────────────────────────────────────────

def test_loss_kind_validation_raises_on_bad_value():
    with pytest.raises(ValueError, match="loss_kind"):
        DBConfig(mode="b1", conditioning="x0_inject", loss_kind="nll")


def test_db_loss_raises_on_bad_loss_kind():
    cfg = DBConfig(mode="b1", conditioning="x0_inject")
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 16))
    step = build_db_step(rt, m, labels)
    out = m(torch.randint(0, V, (2, 16)), db_step=step, db_precond=rt.precond)
    with pytest.raises(ValueError, match="loss_kind"):
        db_loss(out["denoised"], step, rt.precond, loss_kind="nll")


def test_db_loss_l2_rejects_logits_shaped_pred():
    """Passing logits into the L2 branch must fail loudly, not broadcast silently."""
    cfg = DBConfig(mode="b1", conditioning="x0_inject", loss_kind="ce")
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 16))
    step = build_db_step(rt, m, labels)
    bad = torch.randn(2, 16, V + 1)          # not the target's shape
    with pytest.raises(ValueError, match="denoised"):
        db_loss(bad, step, rt.precond, loss_kind="l2")


# ── the forward contract ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["b1", "b3"])
def test_l2_forward_returns_denoised_and_no_logits(mode):
    """Under loss_kind='l2' the tied-readout matmul is SKIPPED entirely."""
    cfg = DBConfig(mode=mode, conditioning="x0_inject", loss_kind="l2")
    m = _model(cfg)
    rt = _RT(cfg)
    ids = torch.randint(0, V, (2, 32))
    labels = torch.randint(0, V, (2, 32))
    step = build_db_step(rt, m, labels)
    out = m(ids, db_step=step, db_precond=rt.precond)
    assert "logits" not in out, "l2 forward must not build the tied readout"
    assert out["denoised"].shape == (2, 32, m.cfg.d_model)
    assert torch.isfinite(out["denoised"]).all()


# ── the free ride, and its absence ───────────────────────────────────────────

def test_no_free_ride_under_l2_at_init():
    """At a FRESH init the weighted L2 loss is clearly nonzero at EVERY σ — including
    low σ, where the CE+tied path scores ≈0 for free.

    Why the L2 number cannot be ~0: the loss is w(σ)·mean((D̂−y)²) with w = 1/c_out², and
    D̂−y = c_out·(hidden − target_of_the_raw_net), so the weighted loss equals the raw
    network output's squared error against a UNIT-VARIANCE target. An untrained network
    cannot match that target, so the loss is O(1) at every σ. The contrast test below
    shows the CE path giving itself the low-σ answer at the same init.
    """
    torch.manual_seed(0)
    cfg = DBConfig(mode="b1", conditioning="x0_inject", loss_kind="l2")
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 32))
    ids = torch.randint(0, V, (2, 32))

    for sv in SIGMA_GRID:
        step = _fixed_sigma_step(m, rt, labels, sv)
        with torch.no_grad():
            out = m(ids, db_step=step, db_precond=rt.precond)
        loss, metrics = db_loss(out["denoised"], step, rt.precond,
                                weighting="edm", loss_kind="l2")
        assert torch.isfinite(loss), f"σ={sv}: non-finite L2 loss"
        assert float(loss) > 0.1, (
            f"σ={sv}: weighted L2 loss {float(loss):.5f} ≈ 0 at init — free ride")
        assert metrics["db/l2_raw"] > 0.0


def test_ce_tied_path_has_the_free_ride_l2_does_not():
    """The contrast that motivated the fork: at the SAME init, tied-readout CE at low σ
    is a fraction of its high-σ value (z ≈ y passes through the tied head and decodes
    the target for free), while the weighted L2 stays O(1) across the grid."""
    torch.manual_seed(0)
    cfg = DBConfig(mode="b1", conditioning="x0_inject", loss_kind="ce")
    m = _model(cfg)
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 32))
    ids = torch.randint(0, V, (2, 32))

    ce, l2w = {}, {}
    for sv in (SIGMA_GRID[0], SIGMA_GRID[-1]):
        step = _fixed_sigma_step(m, rt, labels, sv)
        with torch.no_grad():
            out = m(ids, db_step=step, db_precond=rt.precond)
        c, _ = db_loss(out["logits"], step, rt.precond, loss_kind="ce")
        lw, _ = db_loss(out["denoised"], step, rt.precond,
                        weighting="edm", loss_kind="l2")
        ce[sv], l2w[sv] = float(c), float(lw)

    lo, hi = SIGMA_GRID[0], SIGMA_GRID[-1]
    # The free ride: low-σ CE collapses toward 0 while high-σ CE sits near chance (ln V).
    assert ce[lo] < 0.25 * ce[hi], (
        f"expected the tied-CE free ride at init: CE(σ={lo})={ce[lo]:.4f} "
        f"vs CE(σ={hi})={ce[hi]:.4f}")
    assert ce[lo] < 0.5 * math.log(V)
    # No free ride under L2: both ends of the grid stay O(1).
    assert l2w[lo] > 0.1 and l2w[hi] > 0.1, (l2w[lo], l2w[hi])


# ── gradient reaches the denoiser ────────────────────────────────────────────

def test_gradient_reaches_the_denoiser_under_l2():
    """loss.backward() under L2 must move the core AND coda (b1 = whole-net denoiser).

    σ is fixed mid-range (1.0): at very low σ a near-zero DENOISER gradient is correct
    diffusion behavior (c_out→0 — checklist §9), so low σ would be a flaky non-test.
    """
    torch.manual_seed(0)
    cfg = DBConfig(mode="b1", conditioning="x0_inject", loss_kind="l2")
    m = _model(cfg)
    m.train()
    rt = _RT(cfg)
    labels = torch.randint(0, V, (2, 24))
    ids = torch.randint(0, V, (2, 24))
    step = _fixed_sigma_step(m, rt, labels, 1.0)
    out = m(ids, db_step=step, db_precond=rt.precond)
    loss, _ = db_loss(out["denoised"], step, rt.precond, weighting="edm", loss_kind="l2")
    loss.backward()
    for name in ("prelude", "core", "coda"):
        got = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                  for p in getattr(m, name).parameters())
        assert got, f"{name} received no gradient under L2"
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


def test_l2_objective_can_learn_a_single_batch():
    """The honest signal check: raw L2 on a fixed batch at fixed σ must go down."""
    torch.manual_seed(0)
    cfg = DBConfig(mode="b1", conditioning="x0_inject", loss_kind="l2")
    m = _model(cfg)
    m.train()
    rt = _RT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = torch.randint(0, V, (2, 16))
    labels = torch.randint(0, V, (2, 16))

    first = last = None
    for i in range(40):
        step = _fixed_sigma_step(m, rt, labels, 1.0, seed=i)   # resampled noise
        out = m(ids, db_step=step, db_precond=rt.precond)
        loss, metrics = db_loss(out["denoised"], step, rt.precond,
                                weighting="edm", loss_kind="l2")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        raw = metrics["db/l2_raw"]
        if i < 5:
            first = raw if first is None else min(first, raw)
        if i >= 35:
            last = raw if last is None else min(last, raw)
    assert last < first, f"raw L2 did not improve: first {first:.5f} -> last {last:.5f}"


# ── DB-off parity and config plumbing ────────────────────────────────────────

def test_db_off_is_bit_identical_with_l2_modules_built():
    """Building L2-flavored DB modules must not perturb the DB-off path by one bit."""
    ids = torch.randint(0, V, (2, 32))
    labels = torch.randint(0, V, (2, 32))

    plain = _model(None, seed=1)
    with torch.no_grad():
        a = plain(ids, labels=labels)["loss"]

    withdb = _model(DBConfig(mode="b1", conditioning="x0_inject", loss_kind="l2"), seed=1)
    with torch.no_grad():
        b = withdb(ids, labels=labels)["loss"]

    assert torch.equal(a, b), f"DB-off drifted: {a.item()!r} vs {b.item()!r}"


def test_db_b1_l2_config_resolves_to_option_a():
    """The Phase-1 arm config must land exactly on Option A: b1, x0_inject, l2+edm."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from morph.training.db_setup import build_db_runtime

    with initialize_config_dir(config_dir=os.path.join(REPO, "morph", "configs"),
                               version_base=None):
        cfg = compose(config_name="db_b1_l2")
    rt = build_db_runtime(cfg)
    assert rt is not None
    assert rt.model_cfg.loss_kind == "l2"
    assert rt.model_cfg.loss_weighting == "edm"
    assert rt.model_cfg.conditioning == "x0_inject"
    assert rt.model_cfg.mode == "b1"
    assert rt.activate_at == 0.0
    assert rt.manifest["db/loss_kind"] == "l2"
    assert rt.manifest["db/loss_weighting"] == "edm"


def test_db_b1_concat_config_stays_pinned_to_ce():
    """The historical concat arm keeps Option B explicitly (lineage protection)."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from morph.training.db_setup import build_db_runtime

    with initialize_config_dir(config_dir=os.path.join(REPO, "morph", "configs"),
                               version_base=None):
        cfg = compose(config_name="db_b1_concat")
    rt = build_db_runtime(cfg)
    assert rt.model_cfg.loss_kind == "ce"
    assert rt.model_cfg.loss_weighting == "unweighted"
