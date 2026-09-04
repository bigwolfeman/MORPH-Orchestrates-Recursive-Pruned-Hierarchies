"""Contract for the core-map Jacobian probe (morph/training/core_jacobian.py).

The probe answers a question no other log can: is the looped core's map EXPANSIVE, or has
the realized backward direction merely rotated into an amplifying direction the map always
had? A wrong answer there sends the whole divergence programme after the wrong cure, so
every claim the probe makes is pinned here against a case whose answer is known
independently.

CPU only, tiny config, no tokenizer — the same fixture style as test_tul_forward.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.training.core_jacobian import CoreJacobianProbe, _jacobian_stats

V = 64
DOT = 10


# ── reference cases: maps whose Jacobian is known in closed form ────────────────────────

def test_sigma_matches_svdvals_on_a_linear_map():
    """sigma_max(J) of h -> h W^T is sigma_max(W). Anything else is an estimator bug."""
    torch.manual_seed(0)
    n = 40
    W = torch.randn(n, n)
    true = float(torch.linalg.svdvals(W)[0])
    mask = torch.ones(1, 3, 1)
    est, rel, _ = _jacobian_stats(lambda h: h @ W.t(), torch.randn(1, 3, n), mask, 80, 0)
    assert abs(est - true) / true < 1e-4, f"est {est} true {true}"
    assert rel < 1e-4, f"power iteration did not converge (rel={rel})"


def test_sigma_matches_a_residual_jacobian_known_at_the_origin():
    """h -> h + tanh(h W^T)/4 has J = I + W^T/4 exactly at h = 0.

    A residual Jacobian has its singular values clustered near 1, so this also documents
    that power iteration needs many passes here — a property of the operator, which is
    why `rel_change` ships with every measurement.
    """
    torch.manual_seed(1)
    n = 40
    W = torch.randn(n, n) / n ** 0.5
    true = float(torch.linalg.svdvals(torch.eye(n) + W.t() / 4.0)[0])
    mask = torch.ones(1, 3, 1)
    est, _, _ = _jacobian_stats(lambda h: h + torch.tanh(h @ W.t()) / 4.0,
                                torch.zeros(1, 3, n), mask, 800, 1)
    assert abs(est - true) / true < 1e-3, f"est {est} true {true}"


def test_typical_gain_matches_the_frobenius_norm():
    """The Hutchinson estimate of E||Jv||/||v|| is ||W||_F / sqrt(n) for a linear map."""
    torch.manual_seed(2)
    n = 40
    W = torch.randn(n, n) / 3.0
    true = float(W.norm() / n ** 0.5)
    mask = torch.ones(1, 3, 1)
    _, _, rms = _jacobian_stats(lambda h: h @ W.t(), torch.randn(1, 3, n), mask, 3, 2,
                                n_probe=64)
    assert abs(rms - true) / true < 1e-2, f"rms {rms} true {true}"


def test_mask_restricts_the_operator():
    """A 10x direction outside the mask must not be reported.

    This is the defect that made the first real measurement read 1.5e6: a pad slot enters
    the core loop at h = 0, an RMSNorm at h = 0 has a Jacobian of order 1/eps, and the top
    singular direction sat entirely in the pad subspace.
    """
    scale = torch.tensor([0.5, 10.0, 1.0]).view(1, 3, 1)
    m0 = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1)
    est, _, rms = _jacobian_stats(lambda h: h * scale, torch.randn(1, 3, 8), m0, 40, 3)
    assert abs(est - 0.5) < 1e-5, est
    assert abs(rms - 0.5) < 1e-5, rms


def test_sigma_is_at_least_the_typical_gain():
    """sigma_max >= ||J||_F / sqrt(n) always. A violation means one of them is wrong."""
    torch.manual_seed(4)
    n = 32
    W = torch.randn(n, n)
    mask = torch.ones(1, 2, 1)
    est, _, rms = _jacobian_stats(lambda h: h @ W.t(), torch.randn(1, 2, n), mask, 60, 4,
                                  n_probe=16)
    assert est >= rms - 1e-4, f"sigma {est} < rms {rms}"


# ── the real model ──────────────────────────────────────────────────────────────────────

def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
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
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _fixture(seed=1234):
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    rng = np.random.default_rng(0)
    ids = rng.integers(5, V, size=(2, 90))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)
    torch.manual_seed(seed)
    model = MORPHTransformer(_tiny(tul=TULConfig(slot_id=spec.slot_id,
                                                 prefix_k=spec.prefix_k,
                                                 token_state_dropout=0.0)))
    model.train()
    return model, x, y, layout


def test_capture_is_off_by_default_and_the_forward_is_unchanged():
    """`_jac_capture` None must be a Python-level no-op: same logits, to the last bit."""
    model, x, y, layout = _fixture()
    assert model._jac_capture is None
    torch.manual_seed(7)
    a = model(x, labels=y, slot_layout=layout)["loss"]
    probe = CoreJacobianProbe(model, n_iter=3)
    torch.manual_seed(7)
    with probe.capture():
        b = model(x, labels=y, slot_layout=layout)["loss"]
    assert model._jac_capture is None, "capture list leaked past the context manager"
    assert torch.equal(a, b), f"capture changed the forward: {a} vs {b}"


def _padded_fixture(seed=1234):
    """A layout that really does contain PAD slots — few boundaries, generous budget."""
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=8, slot_id=4)
    rule = BoundaryRule(is_boundary=np.zeros(V, dtype=bool), min_span=4, span_cap=16,
                        eos_id=0)
    rng = np.random.default_rng(5)
    ids = rng.integers(5, V, size=(3, 200))
    ids[ids == spec.slot_id] = 5
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), rule, spec)
    assert bool((~layout.slot_valid).any()), "this fixture must contain pad slots"
    torch.manual_seed(seed)
    model = MORPHTransformer(_tiny(tul=TULConfig(slot_id=spec.slot_id,
                                                 prefix_k=spec.prefix_k,
                                                 token_state_dropout=0.0)))
    model.train()
    return model, x, y, layout


def test_capture_collects_one_point_per_loop_iteration_and_excludes_pads():
    # A PADDED layout, on purpose. A pad slot enters the loop at h = 0 with depth 1, so it
    # is "active" at t = 0; without the validity mask the probe's top singular direction
    # sits in the pad subspace, where an RMSNorm Jacobian is of order 1/eps. Measured on
    # the real model: 1.5e6 before the mask, 198 after. A fixture with no pad slots cannot
    # catch that, so this test uses one that has them.
    model, x, y, layout = _padded_fixture()
    assert bool((~layout.slot_valid).any())
    probe = CoreJacobianProbe(model, n_iter=3)
    with probe.capture() as pts:
        with torch.no_grad():
            model(x, labels=y, slot_layout=layout)
    assert len(pts) >= 1
    assert [p["iter_idx"] for p in pts] == list(range(len(pts)))
    for p in pts:
        assert p["h"].shape[:2] == layout.slot_valid.shape
        # every captured "active" position must be a REAL slot
        assert bool((p["active"] & ~layout.slot_valid).sum() == 0), \
            "a pad slot survived into the probe mask"
        assert not p["h"].requires_grad, "captured state must be detached"


def test_measure_returns_a_finite_operator_norm_on_the_real_core():
    model, x, y, layout = _fixture()
    probe = CoreJacobianProbe(model, n_iter=40)
    with probe.capture() as pts:
        with torch.no_grad():
            model(x, labels=y, slot_layout=layout)
    res = probe.measure(pts[0])
    assert np.isfinite(res.sigma_step) and res.sigma_step > 0
    assert np.isfinite(res.rms_step) and res.rms_step > 0
    assert res.sigma_step >= res.rms_step - 1e-5
    assert len(res.sigma_blocks) == model.cfg.n_core
    assert len(res.rms_blocks) == model.cfg.n_core
    # submultiplicativity: sigma of the composition <= product of the factors' sigmas.
    assert res.sigma_step <= res.block_product * 1.05


def test_measure_tracks_the_operator_when_the_core_is_made_expansive():
    """THE test that fails when the probe is broken.

    Scaling every core MLP weight up makes the core map strictly more expansive. A probe
    that returns a plausible constant, or that reads the carrier's magnitude instead of
    the operator, passes everything above and fails here.
    """
    model, x, y, layout = _fixture()
    probe = CoreJacobianProbe(model, n_iter=60, per_block=False)
    with probe.capture() as pts:
        with torch.no_grad():
            model(x, labels=y, slot_layout=layout)
    before = probe.measure(pts[0])

    with torch.no_grad():
        touched = 0
        for name, p in model.named_parameters():
            if name.startswith("core.") and "mlp" in name and p.dim() == 2:
                p.mul_(3.0)
                touched += 1
    assert touched > 0, "fixture has no core MLP matrices — the test proves nothing"

    after = probe.measure(pts[0])       # SAME operating point, changed operator
    assert after.sigma_step > before.sigma_step * 1.2, \
        f"sigma did not follow the operator: {before.sigma_step} -> {after.sigma_step}"
    assert after.rms_step > before.rms_step * 1.05, \
        f"typical gain did not follow the operator: {before.rms_step} -> {after.rms_step}"


def test_probe_is_rng_neutral():
    """A probed run must be bit-identical to an unprobed one.

    The probe runs an extra forward in TRAINING mode, which draws the Poisson slot depths
    and the dropout mask. Without the save/restore in `train._jacobian_probe` that shifts
    every later step and destroys the bit-reproducibility the divergence programme rests
    on. This test asserts the save/restore, not the good intention.
    """
    from morph.training.train import _jacobian_probe

    model, x, y, layout = _fixture()
    probe = CoreJacobianProbe(model, n_iter=3, per_block=False)

    torch.manual_seed(11)
    baseline = torch.randn(5)

    torch.manual_seed(11)
    out = _jacobian_probe(model, probe, x, y, layout, 0, [0])
    after = torch.randn(5)

    assert out, "probe returned nothing — the capture never fired"
    assert torch.equal(baseline, after), \
        f"the probe consumed RNG: {baseline.tolist()} vs {after.tolist()}"


def test_probe_measurement_is_rng_neutral_with_dropout():
    """The MEASUREMENT, not only the capture forward, must leave the generator untouched.

    `probe.measure` rebuilds the core step in training mode; with block dropout > 0 (the
    production value is 0.1) every power iteration draws a mask from the global generator.
    The 2026-09-03 onset-capture smoke found the replay diverging one step after every
    probed step with weights and gradients bit-identical — the restore wrapped the capture
    forward only. This test builds the fixture WITH dropout, probes per block, and compares
    the generator STATE, which is stricter than a handful of draws.
    """
    from morph.training.train import _jacobian_probe

    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    rng = np.random.default_rng(0)
    ids = rng.integers(5, V, size=(2, 90))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)
    torch.manual_seed(1234)
    model = MORPHTransformer(_tiny(dropout=0.1, tul=TULConfig(slot_id=spec.slot_id,
                                                              prefix_k=spec.prefix_k,
                                                              token_state_dropout=0.0)))
    model.train()
    assert any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()), \
        "fixture has no live dropout; the test would prove nothing"
    probe = CoreJacobianProbe(model, n_iter=3, per_block=True)

    # Sensitivity: a BARE measurement on this fixture must move the generator, or the
    # assertion below would pass on any implementation.
    with probe.capture() as points:
        with torch.no_grad():
            model(x, labels=y, bag_size=0, slot_layout=layout)
    torch.manual_seed(11)
    bare_before = torch.get_rng_state().clone()
    probe.measure(points[0])
    assert not torch.equal(bare_before, torch.get_rng_state()), \
        "a bare measurement did not touch the generator; the fixture cannot detect the leak"

    torch.manual_seed(11)
    state_before = torch.get_rng_state().clone()
    out = _jacobian_probe(model, probe, x, y, layout, 0, [0])
    state_after = torch.get_rng_state()
    assert out and "jac/sigma_b0_t0" in out, "per-block measurement did not run"
    assert torch.equal(state_before, state_after), "the measurement consumed RNG"
