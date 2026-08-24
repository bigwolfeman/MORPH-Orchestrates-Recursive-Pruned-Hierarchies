"""Contract for the core-map spectral-norm penalty (morph/training/spectral_penalty.py).

This is the shipped cure for the TUL core takeover
(.agents/notes/implemented/architecture/2026-08-24-core-spectral-cap.md), and until now its
only check was a `__main__` gate nobody runs in CI. The properties that have to hold:

* the sigma it measures is the real spectral norm of the EFFECTIVE map, not of the raw
  parameter — the core MLP runs ternary QAT, so those are different matrices;
* it is exactly zero while every linear is below the cap, so a healthy run is untouched;
* its gradient reaches the core MLP weights and NOTHING else;
* it enumerates every core MLP linear, and refuses to run as a silent no-op if it finds
  none.

CPU only, tiny config — the same fixture style as test_tul_forward.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.training.spectral_penalty import CoreSpectralPenalty, _power_iter_sigma

V = 64


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=3, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(**kw) -> MORPHTransformer:
    torch.manual_seed(1234)
    return MORPHTransformer(_tiny(**kw))


# ── the estimator ───────────────────────────────────────────────────────────────────────

def test_sigma_matches_svdvals():
    torch.manual_seed(0)
    lin = nn.Linear(48, 72, bias=False).double()
    v = torch.randn(48, dtype=torch.float64)
    with torch.enable_grad():
        sig, _ = _power_iter_sigma(lin, v, n_iter=300)
    true = float(torch.linalg.svdvals(lin.weight.detach())[0])
    got = float(sig.detach())
    assert abs(got - true) / true < 1e-5, f"{got} vs {true}"


def test_sigma_reads_the_effective_map_not_the_raw_parameter():
    """A module whose forward scales its weight must report the SCALED spectral norm.

    The core MLP is ternarised by a parametrization, so `mod.weight` and the matrix the
    forward applies are different. A probe that read the parameter would pass every other
    test here and be wrong by the ternary scale.
    """
    torch.manual_seed(0)

    class Scaled(nn.Module):
        def __init__(self, w, k):
            super().__init__()
            self.weight = nn.Parameter(w)
            self.k = k

        def forward(self, x):
            return x @ (self.weight * self.k).t()

    W = torch.randn(32, 32).double()
    true = float(torch.linalg.svdvals(W)[0])
    mod = Scaled(W, 3.0).double()
    with torch.enable_grad():
        sig, _ = _power_iter_sigma(mod, torch.randn(32, dtype=torch.float64), 300)
    got = float(sig.detach())
    assert abs(got - 3.0 * true) / (3.0 * true) < 1e-5, \
        f"read {got}, raw sigma is {true}, effective is {3.0 * true}"


# ── the penalty on a real model ─────────────────────────────────────────────────────────

def test_it_finds_every_core_mlp_linear():
    m = _model()
    pen = CoreSpectralPenalty(m, cap=1.0, lam=1.0)
    names = [n for n, _, _ in pen._linears]
    assert len(names) == 2 * m.cfg.n_core, names        # gate_up and down per core block
    assert all(n.startswith("core.") and "mlp" in n for n in names), names


def test_it_refuses_to_be_a_silent_no_op():
    """n_core = 0 means there is nothing to penalise. Constructing it must RAISE, not
    return an object whose penalty() is quietly always zero."""
    m = _model(n_core=0)
    with pytest.raises(RuntimeError, match="0 core MLP linears"):
        CoreSpectralPenalty(m, cap=1.0, lam=1.0)


def test_lambda_zero_is_exactly_zero_and_still_reports_sigmas():
    """The logging-only construction train.py uses on EVERY run. It must cost the loss
    exactly nothing while still measuring."""
    m = _model()
    pen = CoreSpectralPenalty(m, cap=0.0, lam=0.0)
    p = pen.penalty()
    assert float(p) == 0.0
    assert p.grad_fn is None, "a lambda=0 penalty must not even build a graph"
    sig = pen.sigmas()
    assert len(sig) == 2 * m.cfg.n_core
    assert all(v > 0 for v in sig.values()), sig


def test_below_the_cap_the_penalty_is_exactly_zero():
    m = _model()
    pen = CoreSpectralPenalty(m, cap=1e6, lam=10.0)
    assert float(pen.penalty().detach()) == 0.0


def _warm(pen, n=25):
    """One power iteration per call is the training setting; the vectors are warm-started
    ACROSS steps. A cold estimate is an underestimate — see the test below."""
    for _ in range(n):
        pen.penalty()
    return pen


def test_above_the_cap_the_penalty_is_positive_and_grows_with_the_excess():
    m = _model()
    worst = max(CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas().values())
    lo = float(_warm(CoreSpectralPenalty(m, cap=worst * 0.9, lam=1.0)).penalty().detach())
    hi = float(_warm(CoreSpectralPenalty(m, cap=worst * 0.5, lam=1.0)).penalty().detach())
    assert lo > 0.0, (lo, worst)
    assert hi > lo * 2.0, (lo, hi)


def test_one_power_iteration_from_cold_underestimates_sigma():
    """Documented, not incidental. `penalty()` runs n_iter=1 against a cached vector that
    starts RANDOM, so the first few steps of a run see a soft version of the hinge; the
    cache then warm-starts and it converges. A reader who assumes step 0 already enforces
    the cap would misread the first hundred steps of every arm."""
    m = _model()
    true_worst = max(CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas().values())
    cold = CoreSpectralPenalty(m, cap=true_worst * 0.9, lam=1.0)
    first = float(cold.penalty().detach())
    warmed = float(_warm(cold).penalty().detach())
    assert first < warmed, (first, warmed)
    assert warmed > 0.0


def test_the_gradient_reaches_the_core_mlp_weights_and_nothing_else():
    """THE contract. A penalty whose gradient does not reach the weights it names is a
    no-op with a plausible number attached; one that reaches other regions is a silent
    second objective on them."""
    m = _model()
    sig = CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas()
    pen = _warm(CoreSpectralPenalty(m, cap=min(sig.values()) * 0.5, lam=1.0))
    m.zero_grad(set_to_none=True)
    pen.penalty().backward()

    touched = {n for n, p in m.named_parameters() if p.grad is not None
               and float(p.grad.abs().sum()) > 0}
    assert touched, "the penalty reached no parameter at all"
    for n in touched:
        assert n.startswith("core.") and "mlp" in n, \
            f"the penalty put gradient on {n}, which it does not claim to constrain"
    # and it must reach ALL of them, not just the worst one
    blocks = {n.split(".")[1] for n in touched}
    assert len(blocks) == m.cfg.n_core, (blocks, touched)


def test_the_penalty_actually_pulls_sigma_down():
    """Optimise the penalty alone for a few steps; the worst sigma must fall toward the
    cap. A hinge that is differentiable but points the wrong way passes everything above."""
    m = _model()
    probe = CoreSpectralPenalty(m, cap=0.0, lam=0.0)
    before = max(probe.sigmas().values())
    cap = before * 0.6
    pen = CoreSpectralPenalty(m, cap=cap, lam=1.0, n_iter=3)
    params = [p for n, p in m.named_parameters() if n.startswith("core.") and "mlp" in n]
    opt = torch.optim.SGD(params, lr=0.5)
    for _ in range(60):
        opt.zero_grad()
        pen.penalty().backward()
        opt.step()
    after = max(CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas().values())
    assert after < before - (before - cap) * 0.5, \
        f"sigma barely moved: {before:.4f} -> {after:.4f} against a cap of {cap:.4f}"


# ── the opt-in attention scope ──────────────────────────────────────────────────────────

def test_include_attn_adds_attention_linears_and_keeps_the_mlp_ones():
    m = _model()
    mlp_only = CoreSpectralPenalty(m, cap=1.0, lam=1.0)
    both = CoreSpectralPenalty(m, cap=1.0, lam=1.0, include_attn=True)
    assert both._n_mlp == mlp_only._n_mlp == 2 * m.cfg.n_core
    assert len(both._linears) > len(mlp_only._linears)
    added = [n for n, _, _ in both._linears[both._n_mlp:]]
    assert added and all(".attention." in n for n in added), added
    # every added module must be a plain Linear — the CCA convolutions are not rank-2 maps
    # on the last dim and the [1, in_features] power-iteration probe does not apply to them.
    for _, mod, inf in both._linears[both._n_mlp:]:
        assert type(mod) is nn.Linear, type(mod)
        assert mod.in_features == inf


def test_include_attn_penalty_reaches_the_attention_weights():
    m = _model()
    pen = _warm(CoreSpectralPenalty(m, cap=0.05, lam=1.0, include_attn=True))
    m.zero_grad(set_to_none=True)
    pen.penalty().backward()
    touched = {n for n, p in m.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert any(".attention." in n for n in touched), sorted(touched)[:8]
    assert any("mlp" in n for n in touched), sorted(touched)[:8]
    for n in touched:
        assert n.startswith("core."), n


def test_include_attn_refuses_to_be_a_silent_no_op():
    """The hook this replaces was a parameter that was accepted and ignored. If the
    enumeration ever stops finding attention linears it must RAISE, not quietly fall back
    to the MLP-only penalty while the config says otherwise."""
    m = _model()
    for blk in m.core:
        blk.attention = nn.Identity()
    with pytest.raises(RuntimeError, match="0 attention linears"):
        CoreSpectralPenalty(m, cap=1.0, lam=1.0, include_attn=True)


def test_sigmas_covers_every_selected_linear():
    m = _model()
    pen = CoreSpectralPenalty(m, cap=0.0, lam=0.0, include_attn=True)
    sig = pen.sigmas()
    assert len(sig) == len(pen._linears)
    assert all(v > 0 for v in sig.values())


# ── the hard projection ─────────────────────────────────────────────────────────────────

def test_projection_enforces_the_cap_on_the_effective_map():
    """THE contract. After one step() every selected linear's EFFECTIVE sigma is at the cap
    or below. A soft hinge only argues for this; the projection must deliver it."""
    from morph.training.spectral_penalty import CoreSpectralProjection
    m = _model()
    before = CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas()
    cap = min(before.values()) * 0.7
    proj = CoreSpectralProjection(m, cap=cap, n_iter=40, verify=True)
    stats = proj.step()
    assert stats["specproj/n_projected"] == len(proj._linears), stats
    after = CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas()
    for name, s in after.items():
        assert s <= cap * 1.05, f"{name} still at {s:.4f} against cap {cap:.4f}"


def test_projection_leaves_an_under_cap_model_untouched_bit_for_bit():
    from morph.training.spectral_penalty import CoreSpectralProjection
    m = _model()
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    stats = CoreSpectralProjection(m, cap=1e6, n_iter=40).step()
    assert stats["specproj/n_projected"] == 0.0
    for n, p in m.named_parameters():
        assert torch.equal(p.detach(), snap[n]), n


def test_projection_writes_through_the_parametrization():
    """MORPH ternarises the core MLP with a weight parametrization, so `mod.weight` is a
    computed property and the trainable leaf is `parametrizations.weight.original`. A
    projection that wrote to `mod.weight` would be discarded on the next forward."""
    from morph.training.spectral_penalty import CoreSpectralProjection, raw_weight
    import torch.nn.utils.parametrize as parametrize

    class Double(nn.Module):
        def forward(self, w):
            return 2.0 * w

    m = _model()
    lin = CoreSpectralPenalty(m, cap=0.0, lam=0.0)._linears[0][1]
    inner = lin._cms
    parametrize.register_parametrization(inner, "weight", Double())
    assert raw_weight(inner) is inner.parametrizations["weight"].original
    orig = raw_weight(inner).detach().clone()
    sig = CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas()
    cap = min(sig.values()) * 0.5
    CoreSpectralProjection(m, cap=cap, n_iter=40).step()
    assert not torch.equal(raw_weight(inner).detach(), orig), \
        "the projection did not reach the parametrized leaf"


def test_projection_verify_catches_a_non_homogeneous_map():
    """The projection assumes `W -> cW` gives `W_eff -> c W_eff`. That is true of the ternary
    parametrization in use and NOT true of CMSBlockLinear.enable_ternary, whose scale is a
    frozen buffer. verify=True must RAISE rather than report a cap it did not enforce."""
    from morph.training.spectral_penalty import CoreSpectralProjection
    import torch.nn.utils.parametrize as parametrize

    class FrozenScale(nn.Module):
        """Effective weight = sign(W) * const — magnitude of W is discarded, so scaling W
        does not scale the map."""
        def __init__(self, ref):
            super().__init__()
            self.register_buffer("g", ref.detach().abs().mean().clamp(min=1e-8))

        def forward(self, w):
            return torch.sign(w) * self.g

    m = _model()
    lin = CoreSpectralPenalty(m, cap=0.0, lam=0.0)._linears[0][1]
    inner = lin._cms
    parametrize.register_parametrization(inner, "weight", FrozenScale(inner.weight))
    sig = CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas()
    with pytest.raises(RuntimeError, match="did not land on the cap"):
        CoreSpectralProjection(m, cap=min(sig.values()) * 0.5, n_iter=40, verify=True).step()


def test_projection_and_penalty_agree_on_what_the_core_linears_are():
    from morph.training.spectral_penalty import CoreSpectralProjection
    m = _model()
    for attn in (False, True):
        a = CoreSpectralPenalty(m, cap=1.0, lam=1.0, include_attn=attn)
        b = CoreSpectralProjection(m, cap=1.0, include_attn=attn)
        assert [n for n, _, _ in a._linears] == [n for n, _, _ in b._linears]
        assert a._n_mlp == b._n_mlp


def test_projection_converges_sigma_before_the_first_step():
    """Two power iterations from a RANDOM start under-read sigma. Measured on the real
    model at step 0: 1.2674 against a converged 1.4293, an 11 % under-read, which made the
    first projection land 13 % above the cap it was asked for. The constructor converges the
    vectors once so `n_iter` per step only has to TRACK, not to find."""
    from morph.training.spectral_penalty import CoreSpectralProjection
    m = _model()
    cold = CoreSpectralProjection(m, cap=1e6, n_iter=2, warmup_iters=0)
    warm = CoreSpectralProjection(m, cap=1e6, n_iter=2, warmup_iters=120)
    ref = CoreSpectralProjection(m, cap=1e6, n_iter=2, warmup_iters=600)
    name, lin, inf = warm._linears[0]
    truth = ref._sigma(name, lin, inf, 600)
    cold_est = cold._sigma(name, lin, inf, 2)
    warm_est = warm._sigma(name, lin, inf, 2)
    assert cold_est < truth * 0.99, (cold_est, truth)
    assert abs(warm_est - truth) / truth < 5e-3, (warm_est, truth)


def test_projection_with_verify_survives_a_full_step_on_a_warmed_model():
    """The end-to-end contract the smoke run exercises: warm up, project, verify, no raise."""
    from morph.training.spectral_penalty import CoreSpectralProjection
    m = _model()
    cap = min(CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas().values()) * 0.8
    proj = CoreSpectralProjection(m, cap=cap, n_iter=2, verify=True, warmup_iters=80)
    stats = proj.step()          # must not raise
    assert stats["specproj/n_projected"] > 0
    after = CoreSpectralPenalty(m, cap=0.0, lam=0.0).sigmas()
    assert max(after.values()) <= cap * 1.05, max(after.values())


# ── the isometry penalty: flatten the spectrum, do not shrink it ────────────────────────

def _make_isometries(pen, scale=3.0):
    from morph.training.spectral_penalty import raw_weight
    with torch.no_grad():
        for _n, lin, _inf in pen._linears:
            w = raw_weight(lin)
            o, i = w.shape
            q = torch.linalg.qr(torch.randn(max(o, i), min(o, i)))[0]
            w.copy_((q if o >= i else q.t())[:o, :i] * scale)


def _plant_spike(pen, rel=2.0):
    """Add a rank-1 term so one direction dominates — the thing the penalty must see."""
    from morph.training.spectral_penalty import raw_weight
    with torch.no_grad():
        for _n, lin, _inf in pen._linears:
            w = raw_weight(lin)
            u = torch.randn(w.shape[0]); u /= u.norm()
            v = torch.randn(w.shape[1]); v /= v.norm()
            w.add_(torch.outer(u, v) * float(w.norm()) * rel)


def test_isometry_penalty_separates_a_spiked_spectrum_from_a_flat_one():
    """It does not reach exactly zero, and the reason is worth knowing: for a WIDE matrix
    with orthonormal rows, ||W v||^2 over random v concentrates but does not become
    constant — the estimator's floor is of order 1/rank, 2/64 on this fixture and 2/1024 on
    the real `down` projections. So the contract is SEPARATION between a spectrum with a
    dominant direction and a flat one, not a zero. Kaiming init already sits close to the
    floor, which is why the comparison is against a planted spike."""
    from morph.training.spectral_penalty import CoreIsometryPenalty
    m = _model()
    pen = CoreIsometryPenalty(m, mu=1.0, n_probe=64, seed=1)
    _plant_spike(pen)
    spiked = float(CoreIsometryPenalty(m, mu=1.0, n_probe=64, seed=1).penalty().detach())
    _make_isometries(pen)
    flat = float(CoreIsometryPenalty(m, mu=1.0, n_probe=64, seed=1).penalty().detach())
    assert spiked > flat * 5.0, (spiked, flat)


def test_isometry_penalty_is_scale_free():
    """THE property that distinguishes it from the spectral cap: doubling every weight must
    not change it. A term that moved would be constraining size, which was measured not to
    be the lever."""
    from morph.training.spectral_penalty import CoreIsometryPenalty, raw_weight
    m = _model()
    a = CoreIsometryPenalty(m, mu=1.0, n_probe=64, seed=7)
    before = float(a.penalty().detach())
    with torch.no_grad():
        for _n, lin, _inf in a._linears:
            raw_weight(lin).mul_(2.0)
    b = CoreIsometryPenalty(m, mu=1.0, n_probe=64, seed=7)
    after = float(b.penalty().detach())
    assert abs(after - before) / max(before, 1e-9) < 0.05, (before, after)


def test_isometry_penalty_is_positive_on_a_spread_spectrum_and_falls_under_optimisation():
    """Optimising it alone must FLATTEN the spectrum. A term that is differentiable but
    points nowhere useful passes both tests above."""
    from morph.training.spectral_penalty import CoreIsometryPenalty, raw_weight
    m = _model()
    with torch.no_grad():                       # plant a dominant direction
        for _n, lin, _inf in CoreIsometryPenalty(m, mu=0.0)._linears:
            w = raw_weight(lin)
            u = torch.randn(w.shape[0]); u /= u.norm()
            v = torch.randn(w.shape[1]); v /= v.norm()
            w.add_(torch.outer(u, v) * float(w.norm()) * 0.5)
    pen = CoreIsometryPenalty(m, mu=1.0, n_probe=32, seed=3)
    before = float(pen.penalty().detach())
    spread_before = max(pen.spread().values())
    assert before > 1e-3, before
    params = [p for n, p in m.named_parameters() if n.startswith("core.") and "mlp" in n]
    opt = torch.optim.SGD(params, lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        pen.penalty().backward()
        opt.step()
    after = float(CoreIsometryPenalty(m, mu=1.0, n_probe=32, seed=3).penalty().detach())
    spread_after = max(CoreIsometryPenalty(m, mu=0.0, n_probe=32, seed=3).spread().values())
    assert after < before * 0.5, (before, after)
    assert spread_after < spread_before * 0.7, (spread_before, spread_after)


def test_isometry_penalty_mu_zero_is_an_exact_zero_with_no_graph():
    from morph.training.spectral_penalty import CoreIsometryPenalty
    m = _model()
    p = CoreIsometryPenalty(m, mu=0.0).penalty()
    assert float(p) == 0.0 and p.grad_fn is None


def test_isometry_penalty_reaches_only_the_core_linears():
    from morph.training.spectral_penalty import CoreIsometryPenalty
    m = _model()
    m.zero_grad(set_to_none=True)
    CoreIsometryPenalty(m, mu=1.0, n_probe=16).penalty().backward()
    touched = {n for n, p in m.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert touched
    for n in touched:
        assert n.startswith("core.") and "mlp" in n, n
