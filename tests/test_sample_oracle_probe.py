"""Contracts for the pure logic of `lab/divergence/sample_oracle_probe.py`.

Four claims the gate's reading rests on, each one that a plausible table would hide if
false:

1. THE NOISE. `relative_noise` scales by each slot's OWN rms, touches only valid slots,
   and is the identity at sigma 0.
2. THE ATTRIBUTION. `span_sums` sums a row's CE per span over scored tokens only, and
   index 0 is the span with no preceding slot (the caller drops it).
3. THE ORACLE. `oracle(N)` is the per-unit minimum over the FIRST N samples, token-weighted;
   `oracle(1)` equals one sample; the gain is non-decreasing in N.
4. THE COLLAPSE MEASURE. `pairwise_cos` is 1.0 for identical samples, ~0 for orthogonal
   ones, and ignores invalid slots.

No model, no checkpoint, no GPU.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "lab/divergence")
from sample_oracle_probe import (                                  # noqa: E402
    oracle_curve, pairwise_cos, relative_noise, row_units, span_sums,
)


# ── 1. the noise ───────────────────────────────────────────────────────────────────

def test_relative_noise_is_identity_at_sigma_zero():
    x = torch.randn(2, 3, 4, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    out = relative_noise(x, 0.0, valid, torch.randn_like(x))
    assert torch.equal(out, x)


def test_relative_noise_touches_only_valid_slots_and_scales_by_slot_rms():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 4, 8)
    x[0, 1] = 10.0 * x[0, 0]                        # slot 1 is slot 0 at 10x the scale
    valid = torch.tensor([[True, True, False]])
    eps = torch.ones_like(x)                        # deterministic direction
    out = relative_noise(x, 0.5, valid, eps)
    delta = out - x
    assert torch.equal(delta[0, 2], torch.zeros(4, 8))            # invalid slot untouched
    rms0 = x[0, 0].pow(2).mean().sqrt()
    rms1 = x[0, 1].pow(2).mean().sqrt()
    assert torch.allclose(delta[0, 0], 0.5 * rms0 * torch.ones(4, 8), atol=1e-5)
    assert torch.allclose(delta[0, 1], 0.5 * rms1 * torch.ones(4, 8), atol=1e-4)
    assert (delta[0, 1].abs().mean() / delta[0, 0].abs().mean()) == pytest.approx(10.0, rel=0.05)


def test_relative_noise_keeps_dtype_and_works_on_a_plain_carrier():
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    valid = torch.ones(2, 5, dtype=torch.bool)
    out = relative_noise(x, 0.3, valid, torch.randn(2, 5, 16))
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    assert not torch.equal(out, x)


# ── 2. the attribution ─────────────────────────────────────────────────────────────

def test_span_sums_sum_scored_tokens_per_span_and_keep_index_zero():
    ce = np.array([1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 6.0])
    span = np.array([0, 0, -1, 1, 1, 2, 2])
    istok = np.array([True, True, False, True, False, True, True])   # pos 4 unscored
    s, n = span_sums(ce, span, istok, n_spans=3)
    assert s.tolist() == [3.0, 3.0, 11.0]
    assert n.tolist() == [2.0, 1.0, 2.0]


def test_span_sums_ignores_negative_span_index_even_if_marked_token():
    ce = np.array([1.0, 2.0])
    span = np.array([-1, 0])
    istok = np.array([True, True])
    s, n = span_sums(ce, span, istok, n_spans=1)
    assert s.tolist() == [2.0] and n.tolist() == [1.0]


# ── 3. the oracle ──────────────────────────────────────────────────────────────────

def test_oracle_is_the_per_unit_min_over_the_first_n_samples_token_weighted():
    det = np.array([4.0, 6.0])                     # two units
    count = np.array([2.0, 2.0])                   # 4 tokens
    samp = np.array([[5.0, 7.0],                   # sample 0: worse everywhere
                     [3.0, 8.0],                   # sample 1: better on unit 0
                     [6.0, 2.0]])                  # sample 2: better on unit 1
    out = oracle_curve(samp, det, count, ns=[1, 2, 4])
    assert out["det"] == pytest.approx(10.0 / 4)
    assert out["oracle_1"] == pytest.approx(12.0 / 4)             # one sample = that sample
    assert out["mean_samples"] == pytest.approx((12.0 + 11.0 + 8.0) / 3 / 4)
    assert out["oracle_2"] == pytest.approx((3.0 + 7.0) / 4)      # min over samples 0,1
    assert out["gain_2"] == pytest.approx(out["det"] - out["oracle_2"])
    assert "oracle_4" not in out                                   # only 3 samples exist


def test_oracle_gain_is_non_decreasing_in_n():
    rng = np.random.default_rng(0)
    samp = rng.normal(5.0, 1.0, size=(16, 50))
    det = rng.normal(5.0, 1.0, size=50)
    count = np.ones(50)
    out = oracle_curve(samp, det, count, ns=[1, 2, 4, 8, 16])
    gains = [out[f"gain_{n}"] for n in (1, 2, 4, 8, 16)]
    assert all(b >= a for a, b in zip(gains, gains[1:])), gains


def test_oracle_curve_refuses_no_tokens():
    with pytest.raises(ValueError):
        oracle_curve(np.zeros((2, 3)), np.zeros(3), np.zeros(3), ns=[1])


# ── 4. the collapse measure ────────────────────────────────────────────────────────

def test_pairwise_cos_reads_one_for_identical_samples_and_zero_for_orthogonal():
    base = np.random.default_rng(1).normal(size=(1, 2, 8))
    same = np.stack([base, base, base])                 # [3, B=1, S=2, D=8]
    valid = np.ones((1, 2), dtype=bool)
    assert pairwise_cos(same, valid) == pytest.approx(1.0)
    orth = np.zeros((4, 1, 1, 4))
    for i in range(4):
        orth[i, 0, 0, i] = 1.0                          # four orthogonal unit vectors
    assert pairwise_cos(orth, np.ones((1, 1), dtype=bool)) == pytest.approx(0.0)


def test_pairwise_cos_ignores_invalid_slots():
    a = np.random.default_rng(2).normal(size=(1, 1, 6))
    st = np.zeros((2, 1, 2, 6))
    st[0, 0, 0] = a[0, 0]; st[1, 0, 0] = a[0, 0]         # slot 0: identical  -> 1.0
    st[0, 0, 1] = [1, 0, 0, 0, 0, 0]; st[1, 0, 1] = [0, 1, 0, 0, 0, 0]   # slot 1: 0.0
    assert pairwise_cos(st, np.array([[True, False]])) == pytest.approx(1.0)
    assert pairwise_cos(st, np.array([[True, True]])) == pytest.approx(0.5)


def test_pairwise_cos_needs_two_samples():
    with pytest.raises(ValueError):
        pairwise_cos(np.zeros((1, 1, 1, 3)), np.ones((1, 1), dtype=bool))


# ── row units ──────────────────────────────────────────────────────────────────────

def test_row_units_sums_per_row():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    row = np.array([0, 1, 1, 3])
    assert row_units(vals, row, n_rows=4).tolist() == [1.0, 5.0, 0.0, 4.0]
