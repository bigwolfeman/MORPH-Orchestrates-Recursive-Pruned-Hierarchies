"""``lab/divergence/fan_mixture_probe.py`` — the mixture readings, on hand-built NLLs.

Every test asserts a VALUE the maths fixes, not a shape: K identical streams collapse
the whole ladder onto one number; one stream that is perfect on one span puts the
uniform mixture between the oracle and ``oracle + log K / span_len``; the prefix
mixture telescopes to the uniform mixture per span; winner persistence reads a
hand-built sequence; and the offset bins partition the tokens exactly once.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab", "divergence"))
from fan_mixture_probe import (BINS, block_bootstrap, offset_bin_index, oracle_per_span,  # noqa: E402
                               prefix_mix_token_nll, span_stream_nll, uniform_mix_nll,
                               winner_persistence)

S, K, LMAX = 7, 4, 10


def _mask(lens, lmax=LMAX):
    m = torch.zeros(len(lens), lmax, dtype=torch.bool)
    for i, n in enumerate(lens):
        m[i, :n] = True
    return m


def _rand_nll(s=S, k=K, lmax=LMAX, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(s, k, lmax, generator=g).double() * 4.0 + 0.1


# ── (a) K identical streams: every reading is the same number ───────────────

def test_identical_streams_collapse_the_whole_ladder():
    lens = [1, 2, 3, 5, 8, 10, 4]
    mask = _mask(lens)
    one = _rand_nll(s=S, k=1, seed=1)
    nll = one.expand(S, K, LMAX).contiguous()
    ell = span_stream_nll(nll, mask)
    best, arg = oracle_per_span(nll, mask)
    umix = uniform_mix_nll(nll, mask)
    ptok = prefix_mix_token_nll(nll, mask)
    single = ell[:, 0]
    assert torch.allclose(ell, single.unsqueeze(1).expand(S, K))
    assert torch.allclose(best, single, atol=1e-12)
    # -log((1/K) * K * exp(-l)) == l exactly, so the mixture buys nothing off copies.
    assert torch.allclose(umix, single, atol=1e-12)
    assert torch.allclose(ptok.sum(dim=1), single, atol=1e-12)
    # the prefix mixture's PER-TOKEN nll is the stream's own token nll, not just the sum
    assert torch.allclose(ptok, nll[:, 0, :] * mask.double(), atol=1e-12)
    assert int(arg.min()) == 0 and int(arg.max()) == 0     # ties resolve to stream 0


# ── (b) one stream perfect on one span: the log K bound ─────────────────────

def test_one_perfect_stream_and_the_log_k_bound():
    lens = [4, 4, 4]
    mask = _mask(lens, lmax=4)
    nll = torch.full((3, K, 4), 2.0, dtype=torch.float64)
    nll[0, 2, :] = 0.01                                    # stream 2 is perfect on span 0
    nll[1:, 2, :] = 9.0                                    # and terrible elsewhere
    best, arg = oracle_per_span(nll, mask)
    umix = uniform_mix_nll(nll, mask)
    assert int(arg[0]) == 2
    assert best[0] == pytest.approx(0.04)
    assert best[1] == pytest.approx(8.0) and best[2] == pytest.approx(8.0)
    log_k = math.log(K)
    for s in range(3):
        assert best[s] <= umix[s] + 1e-12
        assert umix[s] <= best[s] + log_k + 1e-12
    # span 0: three streams at 8.0 and one at 0.04 -> the mixture pays ~log 4 over the
    # oracle because the prior gives the winner only 1/4.
    expect0 = -(torch.logsumexp(-torch.tensor([8.0, 8.0, 0.04, 8.0], dtype=torch.float64),
                                dim=0) - log_k)
    assert umix[0] == pytest.approx(float(expect0), abs=1e-12)
    # the gap is log K minus log(1 + 3 exp(-7.96)), i.e. log K to a part in a thousand:
    # the losers are not quite dead, so the mixture is a hair better than the bound.
    gap = float(umix[0] - best[0])
    assert gap == pytest.approx(log_k - math.log1p(3 * math.exp(-7.96)), abs=1e-12)
    assert log_k - 2e-3 < gap < log_k
    # and per TOKEN that gap is log K / span_len
    assert float((umix[0] - best[0]) / lens[0]) == pytest.approx(log_k / 4.0, abs=1e-3)
    # span 1: all four streams are 8.0 except the terrible one at 36.0 -> the mixture is
    # strictly better than the WORST and strictly worse than the oracle.
    assert 8.0 < float(umix[1]) < 36.0


def test_uniform_mixture_beats_the_best_single_when_winners_differ():
    """The reading the probe exists for: no single stream is the best everywhere."""
    mask = _mask([4, 4], lmax=4)
    nll = torch.full((2, 2, 4), 3.0, dtype=torch.float64)
    nll[0, 0, :] = 0.5                                     # stream 0 wins span 0
    nll[1, 1, :] = 0.5                                     # stream 1 wins span 1
    ell = span_stream_nll(nll, mask)
    best, _ = oracle_per_span(nll, mask)
    umix = uniform_mix_nll(nll, mask)
    assert float(ell.sum(0).min()) == pytest.approx(14.0)  # 2.0 + 12.0, either stream
    assert float(best.sum()) == pytest.approx(4.0)
    assert float(best.sum()) < float(umix.sum()) < float(ell.sum(0).min())


# ── (c) the telescoping identity ────────────────────────────────────────────

def test_prefix_mixture_telescopes_to_the_uniform_mixture():
    lens = [1, 2, 3, 5, 8, 10, 4]
    mask = _mask(lens)
    nll = _rand_nll(seed=7)
    umix = uniform_mix_nll(nll, mask)
    ptok = prefix_mix_token_nll(nll, mask)
    assert float((ptok.sum(dim=1) - umix).abs().max()) < 1e-9
    assert torch.allclose(ptok[~mask], torch.zeros_like(ptok[~mask]))
    # The FIRST token of a span pays the uniform-prior mixture; later tokens use the
    # prefix, so on a span with a clear winner the prefix mixture must improve with j.
    mask2 = _mask([6], lmax=6)
    n2 = torch.full((1, 3, 6), 4.0, dtype=torch.float64)
    n2[0, 1, :] = 0.2
    p2 = prefix_mix_token_nll(n2, mask2)[0]
    assert float(p2[0]) > float(p2[1]) > float(p2[2])
    assert float(p2[0]) == pytest.approx(0.2 + math.log(3.0), abs=5e-2)
    assert float(p2[5]) == pytest.approx(0.2, abs=1e-3)    # the prefix has identified it


def test_prefix_mixture_first_token_is_the_uniform_prior_read():
    """w_{0,k} = 1/K by construction, so token 0's nll is the log-mean of the K streams."""
    mask = _mask([3], lmax=3)
    nll = torch.tensor([[[1.0, 5.0, 5.0], [3.0, 5.0, 5.0]]], dtype=torch.float64)
    p = prefix_mix_token_nll(nll, mask)[0, 0]
    want = -(torch.logsumexp(-torch.tensor([1.0, 3.0], dtype=torch.float64), dim=0)
             - math.log(2.0))
    assert float(p) == pytest.approx(float(want), abs=1e-12)


# ── (d) winner persistence ──────────────────────────────────────────────────

def test_winner_persistence_on_a_hand_built_sequence():
    #        row 0: 0 0 1 1        row 1: 2 2 2
    winners = np.array([0, 0, 1, 1, 2, 2, 2])
    rows = np.array([0, 0, 0, 0, 1, 1, 1])
    w = winner_persistence(winners, rows, k=4)
    assert w["n_pairs"] == 5                               # 3 in row 0, 2 in row 1
    assert w["p_repeat"] == pytest.approx(4.0 / 5.0)       # only 0->1 at index 1->2 breaks
    assert w["shares"] == pytest.approx([2 / 7, 2 / 7, 3 / 7, 0.0])
    assert w["chance"] == pytest.approx((2 / 7) ** 2 * 2 + (3 / 7) ** 2)
    assert w["n_spans"] == 7


def test_winner_persistence_never_pairs_across_rows():
    winners = np.array([3, 3])
    rows = np.array([0, 1])
    w = winner_persistence(winners, rows, k=4)
    assert w["n_pairs"] == 0 and math.isnan(w["p_repeat"])
    assert w["chance"] == pytest.approx(1.0)


def test_winner_persistence_at_chance_on_iid_draws():
    rng = np.random.default_rng(3)
    winners = rng.integers(0, 4, size=20000)
    rows = np.repeat(np.arange(200), 100)
    w = winner_persistence(winners, rows, k=4)
    assert w["p_repeat"] == pytest.approx(0.25, abs=0.02)
    assert w["chance"] == pytest.approx(0.25, abs=0.01)


# ── (e) the offset bins partition the tokens ────────────────────────────────

def test_offset_bins_partition_the_tokens_and_land_in_the_right_bin():
    lens = [1, 2, 3, 5, 8, 10, 4]
    mask = _mask(lens)
    b = offset_bin_index(mask)
    assert int((b >= 0).sum()) == sum(lens)                # every scored token binned
    assert torch.equal(b >= 0, mask)                       # and nothing else
    counts = torch.bincount(b[mask], minlength=len(BINS))
    assert int(counts.sum()) == sum(lens)
    # BINS = [0],[1],[2],[3],[4-7],[8-15],[16+]; offsets 0..len-1 per span.
    want = [0] * len(BINS)
    for n in lens:
        for j in range(n):
            want[0 if j == 0 else 1 if j == 1 else 2 if j == 2 else 3 if j == 3
                 else 4 if j <= 7 else 5 if j <= 15 else 6] += 1
    assert counts.tolist() == want
    assert counts.tolist() == [7, 6, 5, 4, 9, 2, 0]
    # a long span reaches the 16+ bin
    long_mask = _mask([20], lmax=20)
    lb = offset_bin_index(long_mask)[0]
    assert lb[16].item() == 6 and lb[15].item() == 5 and lb[8].item() == 5


def test_offset_bins_ignore_padded_columns():
    mask = _mask([2], lmax=6)
    b = offset_bin_index(mask)[0]
    assert b.tolist() == [0, 1, -1, -1, -1, -1]


# ── the bootstrap ───────────────────────────────────────────────────────────

def test_block_bootstrap_point_is_the_token_weighted_mean_and_brackets_it():
    rng = np.random.default_rng(11)
    tok = rng.integers(5, 40, size=60).astype(np.float64)
    a = tok * (2.0 + rng.normal(0, 0.1, 60))
    b = tok * (2.5 + rng.normal(0, 0.1, 60))
    out = block_bootstrap({"a": a, "b": b}, tok, [("b", "a")], n_boot=400, seed=1)
    assert out["a"]["point"] == pytest.approx(a.sum() / tok.sum())
    assert out["b"]["point"] == pytest.approx(b.sum() / tok.sum())
    assert out["b-a"]["point"] == pytest.approx(out["b"]["point"] - out["a"]["point"])
    assert out["a"]["lo"] < out["a"]["point"] < out["a"]["hi"]
    assert out["b-a"]["lo"] > 0.0                          # b is worse, and the CI says so
    assert out["a"]["n_units"] == 60 and out["a"]["n_boot"] == 400


def test_block_bootstrap_is_deterministic_in_the_seed():
    tok = np.full(20, 10.0)
    v = np.arange(20, dtype=np.float64)
    one = block_bootstrap({"v": v}, tok, [], n_boot=200, seed=5)
    two = block_bootstrap({"v": v}, tok, [], n_boot=200, seed=5)
    three = block_bootstrap({"v": v}, tok, [], n_boot=200, seed=6)
    assert one["v"] == two["v"]
    assert one["v"]["lo"] != three["v"]["lo"]


# ── guards ──────────────────────────────────────────────────────────────────

def test_shape_guards_raise():
    with pytest.raises(ValueError):
        span_stream_nll(torch.zeros(3, 4), torch.ones(3, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        span_stream_nll(torch.zeros(3, 4, 5), torch.ones(3, 6, dtype=torch.bool))
    with pytest.raises(ValueError):
        winner_persistence(np.zeros(3), np.zeros(4), k=2)
    with pytest.raises(ValueError):
        block_bootstrap({"a": np.zeros(3)}, np.zeros(3), [])


def test_masked_columns_never_reach_a_sum():
    """A pad carrying garbage must change nothing: the mask, not the value, decides."""
    mask = _mask([2], lmax=4)
    clean = torch.tensor([[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]], dtype=torch.float64)
    dirty = clean.clone()
    dirty[0, :, 2:] = 99.0
    assert torch.allclose(span_stream_nll(clean, mask), span_stream_nll(dirty, mask))
    assert torch.allclose(uniform_mix_nll(clean, mask), uniform_mix_nll(dirty, mask))
    assert torch.allclose(prefix_mix_token_nll(clean, mask),
                          prefix_mix_token_nll(dirty, mask))


def test_winner_persistence_block_bootstrap_brackets_the_point():
    rng = np.random.default_rng(9)
    rows = np.repeat(np.arange(60), 20)
    # a sticky chain: each row starts somewhere and repeats its winner 80 % of the time
    w = np.empty(rows.shape, dtype=np.int64)
    cur = 0
    for i, r in enumerate(rows):
        if i == 0 or rows[i - 1] != r or rng.random() > 0.8:
            cur = int(rng.integers(0, 4))
        w[i] = cur
    out = winner_persistence(w, rows, k=4, n_boot=300, seed=2)
    assert out["n_pairs"] == 60 * 19
    assert out["p_repeat"] > 0.7                      # sticky by construction
    assert out["excess"] == pytest.approx(out["p_repeat"] - out["chance"])
    assert out["p_repeat_ci"]["lo"] < out["p_repeat"] < out["p_repeat_ci"]["hi"]
    assert out["excess_ci"]["lo"] > 0.3               # and the interval says it is real
    assert out["p_repeat_ci"]["n_units"] == 60 and out["p_repeat_ci"]["n_boot"] == 300


def test_winner_persistence_bootstrap_finds_no_excess_on_iid_draws():
    rng = np.random.default_rng(4)
    rows = np.repeat(np.arange(120), 20)
    w = rng.integers(0, 4, size=rows.shape)
    out = winner_persistence(w, rows, k=4, n_boot=400, seed=3)
    assert out["excess"] == pytest.approx(0.0, abs=0.02)
    assert out["excess_ci"]["lo"] < 0.0 < out["excess_ci"]["hi"]


def test_winner_persistence_without_boot_carries_no_interval():
    out = winner_persistence(np.array([0, 0, 1]), np.array([0, 0, 0]), k=2)
    assert "p_repeat_ci" not in out and out["excess"] == pytest.approx(0.5 - (4 / 9 + 1 / 9))
