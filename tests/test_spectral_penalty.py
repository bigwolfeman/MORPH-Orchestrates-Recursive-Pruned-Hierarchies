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
