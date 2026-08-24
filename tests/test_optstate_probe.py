"""Contracts for the offline AdEMAMix state reader, `lab/divergence/optstate_probe.py`.

Every test here asserts a VALUE the module is supposed to produce, not a type or a shape.
Two of them exist because the module got the answer wrong once:

* `test_eps_inside_changes_the_denominator` — the first version hardcoded the floored
  denominator `sqrt(nu/bc2 + eps)` while every MORPH run uses `sqrt(nu/bc2) + eps`. With
  gradients at the 5e-5 scale the two differ by a factor of ~100, and the floored form
  reported that 99 % of core coordinates sat on the floor.
* `test_dequant_rejects_an_unknown_layout` — a reader that returns zeros for a layout it
  does not know reports "the slow channel is empty", which is the conclusion the
  instrument exists to test.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.divergence.optstate_probe import (           # noqa: E402
    aggregate, dequant_state, drift_between, param_names_in_optimizer_order,
    param_stats, region_of,
)
from lab.divergence.score_optstate import spearman    # noqa: E402


# ── naming ───────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,want", [
    ("core.3.mlp.0.gate_up.weight", "core"),
    ("_orig_mod.core.3.attention.q.weight", "core"),
    # torch.compile wraps SUBMODULES, so the marker shows up mid-name.
    ("prelude.0.mlp._orig_mod.0.down._cms.parametrizations.weight.original", "prelude"),
    ("embed.hybrid.euc_embed.parametrizations.weight.original", "embed"),
    ("tul.E_slot", "tul"),
])
def test_region_of(name, want):
    assert region_of(name) == want


def test_region_rule_matches_the_trainer():
    """The bucket must be the same one `_preclip_probe` uses, or `slow_rms(core)` and
    `preclip/core` would be counting different tensors and could not be compared."""
    for name in ("_orig_mod.core.0.x", "coda.2.mlp.w", "prelude.1._orig_mod.a.b"):
        parts = name.replace("_orig_mod.", "").split(".")
        assert region_of(name) == parts[0]


def test_param_names_follow_the_decay_split():
    """`optimizer.state_dict()` indexes decay params first, then no-decay, each in
    `named_parameters()` order. `norm` and `bias` are no-decay keywords."""
    m = torch.nn.Module()
    m.w1 = torch.nn.Parameter(torch.zeros(2))
    m.norm_a = torch.nn.Parameter(torch.zeros(3))
    m.w2 = torch.nn.Parameter(torch.zeros(4))
    m.bias_b = torch.nn.Parameter(torch.zeros(5))
    assert param_names_in_optimizer_order(m, 0.1) == ["w1", "w2", "norm_a", "bias_b"]


# ── the decomposition ────────────────────────────────────────────────────────────────
def _flat(v, n=64):
    return torch.full((n,), float(v))


def test_param_stats_hits_the_analytic_answer():
    """nu=4, bc2=1 -> denom=2; m2=1, alpha=2 -> slow=1 exactly; fast = sqrt(q)/denom = 1."""
    n = 64
    s = param_stats(_flat(1.0, n), _flat(4.0, n), alpha=2.0, bc2=1.0, eps=1e-8,
                    eps_inside=False, update_clip=5.0)
    assert s["n"] == n
    assert s["slow_sq_sum"] == pytest.approx(n, rel=1e-5)
    assert s["fast_sq_sum"] == pytest.approx(n, rel=1e-5)
    # coh = RMS(m2)/RMS_ema(g) = sqrt(sum m2^2 / sum nu) = sqrt(1/4)
    a = aggregate({"core.0.w": s})
    assert a["core"]["coh"] == pytest.approx(0.5, rel=1e-5)
    assert a["core"]["slow_over_fast"] == pytest.approx(1.0, rel=1e-5)


def test_slow_over_fast_scales_with_alpha():
    """The slow channel is linear in alpha, which is what `ademamix_alpha_cap` changes."""
    lo = aggregate({"core.w": param_stats(_flat(1.0), _flat(4.0), 1.0, 1.0, 1e-8,
                                          False, 5.0)})
    hi = aggregate({"core.w": param_stats(_flat(1.0), _flat(4.0), 3.5, 1.0, 1e-8,
                                          False, 5.0)})
    assert hi["core"]["slow_over_fast"] / lo["core"]["slow_over_fast"] == pytest.approx(3.5, rel=1e-5)


def test_eps_inside_changes_the_denominator():
    """At MORPH's gradient scale the two forms differ by about 100x. A reader that ignores
    the flag reports the wrong regime, which is the bug this test exists to catch."""
    m2, nu = _flat(1e-6), _flat(1e-12)
    outside = param_stats(m2, nu, 1.0, 1.0, 1e-8, eps_inside=False, update_clip=0.0)
    inside = param_stats(m2, nu, 1.0, 1.0, 1e-8, eps_inside=True, update_clip=0.0)
    # eps-outside: denom = 1e-6 + 1e-8; eps-inside: denom = sqrt(1e-12 + 1e-8) ~ 1e-4
    assert outside["slow_sq_sum"] / inside["slow_sq_sum"] > 1e3
    # and the fast channel is ~1 only under eps-outside
    assert (outside["fast_sq_sum"] / outside["n"]) ** 0.5 == pytest.approx(0.99, abs=0.02)
    assert (inside["fast_sq_sum"] / inside["n"]) ** 0.5 < 0.02


def test_clip_counts_are_counts_not_fractions():
    n = 64
    s = param_stats(_flat(1.0, n), _flat(4.0, n), alpha=20.0, bc2=1.0, eps=1e-8,
                    eps_inside=False, update_clip=5.0)
    # slow = 20*1/2 = 10, so every coordinate is above 1 AND above the clip of 5
    assert s["n_slow_gt1"] == n
    assert s["n_slow_gt_clip"] == n
    a = aggregate({"core.w": s})
    assert a["core"]["frac_slow_gt1"] == pytest.approx(1.0)


def test_aggregate_splits_core_from_noncore():
    core = param_stats(_flat(1.0), _flat(4.0), 2.0, 1.0, 1e-8, False, 5.0)
    coda = param_stats(_flat(0.0), _flat(4.0), 2.0, 1.0, 1e-8, False, 5.0)
    a = aggregate({"core.0.w": core, "coda.0.w": coda})
    assert a["core"]["slow_over_fast"] == pytest.approx(1.0, rel=1e-5)
    assert a["noncore"]["slow_over_fast"] == pytest.approx(0.0, abs=1e-9)
    assert a["all"]["n"] == core["n"] + coda["n"]
    # `noncore` must EXCLUDE core, not merely be everything-else-shaped
    assert a["noncore"]["n"] == coda["n"]


# ── dequantisation ───────────────────────────────────────────────────────────────────
def test_dequant_passes_fp32_state_through():
    e = {"m2": torch.arange(8.0), "nu": torch.ones(8)}
    m2, nu = dequant_state(e, 8, helper=None)
    assert torch.equal(m2, torch.arange(8.0))
    assert torch.equal(nu, torch.ones(8))


def test_dequant_rejects_an_unknown_layout():
    with pytest.raises(KeyError):
        dequant_state({"exp_avg": torch.zeros(4)}, 4, helper=None)


def test_dequant_round_trips_the_dynamic_qmap_layout():
    """Quantise a known tensor with the optimizer's own quantiser, then read it back."""
    import bitsandbytes.functional as bnbF
    from lab.divergence.optstate_probe import _deq_helper
    helper = _deq_helper("cpu")
    torch.manual_seed(0)
    ref = torch.randn(512) * 1e-5
    code = helper._code(torch.device("cpu"), True)
    q, qs = bnbF.quantize_blockwise(ref.contiguous(), code=code, blocksize=256)
    e = {"m2_dcode": q, "m2_damax": qs.absmax,
         "nu_dcode": q, "nu_damax": qs.absmax}
    m2, _nu = dequant_state(e, 512, helper)
    assert m2.shape == (512,)
    assert float((m2 - ref).norm() / ref.norm()) < 0.02


# ── drift ────────────────────────────────────────────────────────────────────────────
def test_drift_between_is_per_region_and_signed():
    prev = {"core.0.w": torch.zeros(4), "coda.0.w": torch.zeros(4)}
    cur = {"core.0.w": torch.ones(4), "coda.0.w": -torch.ones(4)}
    d = drift_between(prev, cur, list(prev))
    assert torch.equal(d["core"], torch.ones(4))
    assert d["all"].numel() == 8
    # a region moving the other way must not cancel inside `core`
    assert float(d["core"].sum()) == pytest.approx(4.0)


def test_drift_skips_names_absent_from_one_side():
    d = drift_between({"core.a": torch.zeros(2)}, {"core.a": torch.ones(2),
                                                   "core.b": torch.ones(2)},
                      ["core.a", "core.b"])
    assert d["core"].numel() == 2


# ── the scorer ───────────────────────────────────────────────────────────────────────
def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # one adjacent swap in 5 items: rho = 1 - 6*2/(5*24) = 0.9
    assert spearman([1, 2, 3, 4, 5], [1, 3, 2, 4, 5]) == pytest.approx(0.9)


def test_spearman_handles_ties():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
