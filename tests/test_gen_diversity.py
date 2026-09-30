"""`lab/divergence/gen_diversity.py` — hand-built sequences with exactly computable
answers. Each metric's test is worked out by hand in the comment beside it so a broken
implementation cannot pass by accident.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab", "divergence"))
from gen_diversity import (  # noqa: E402
    bootstrap_ci, bootstrap_ci_rows, distinct_n, ngrams, rep_l, seq_rep_n,
)


# ── ngrams ───────────────────────────────────────────────────────────────────────────
def test_ngrams_basic_count_and_content():
    assert ngrams([1, 2, 3, 4], 2) == [(1, 2), (2, 3), (3, 4)]
    assert ngrams([1, 2], 4) == []          # shorter than n
    assert ngrams([1, 2, 3], 3) == [(1, 2, 3)]


def test_ngrams_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        ngrams([1, 2, 3], 0)


# ── seq_rep_n ────────────────────────────────────────────────────────────────────────
def test_seq_rep_n_total_collapse():
    # 8 identical tokens: every 4-gram is (0,0,0,0) -> 1 unique of (8-4+1)=5 total.
    # seq_rep_4 = 1 - 1/5 = 0.8 exactly.
    ids = [0] * 8
    assert seq_rep_n(ids, 4) == pytest.approx(0.8)


def test_seq_rep_n_no_repeats_is_zero():
    ids = list(range(20))
    assert seq_rep_n(ids, 4) == 0.0
    assert seq_rep_n(ids, 1) == 0.0


def test_seq_rep_n_periodic_pattern():
    # period-4 pattern repeated 5 times = 20 tokens. 4-grams: 20-4+1 = 17 total.
    # The 4-gram starting at offset k (mod 4) is the SAME tuple every time it recurs, so
    # there are exactly 4 unique 4-grams (one per phase) as long as >= 2 full periods
    # exist for every phase, which 5 repeats guarantees.
    ids = [1, 2, 3, 4] * 5
    total = 20 - 4 + 1
    expected = 1.0 - 4 / total
    assert seq_rep_n(ids, 4) == pytest.approx(expected)


def test_seq_rep_n_short_sequence_is_zero_not_crash():
    assert seq_rep_n([1, 2], 4) == 0.0


def test_seq_rep_n_minimum_length_never_fully_collapses():
    # A sequence of exactly length n has exactly one n-gram -> seq_rep_n is always 0.0
    # (1 unique of 1 total), regardless of content.
    assert seq_rep_n([5, 5, 5, 5], 4) == 0.0


# ── rep_l ────────────────────────────────────────────────────────────────────────────
def test_rep_l_worked_example_window_3():
    # ids = [1,2,3,1,2,3], window=3.
    #  i=0 lookback=[]            -> not repeat
    #  i=1 lookback=[1]           -> 2 not in           -> not repeat
    #  i=2 lookback=[1,2]         -> 3 not in            -> not repeat
    #  i=3 lookback=[1,2,3]       -> 1 IS in             -> repeat
    #  i=4 lookback=[2,3,1]       -> 2 IS in             -> repeat
    #  i=5 lookback=[3,1,2]       -> 3 IS in             -> repeat
    # 3 repeats / 6 tokens = 0.5
    assert rep_l([1, 2, 3, 1, 2, 3], window=3) == pytest.approx(0.5)


def test_rep_l_no_repeats_is_zero():
    assert rep_l(list(range(50)), window=128) == 0.0


def test_rep_l_all_same_token_is_near_one():
    # Every position except the first (empty lookback) is a repeat: (n-1)/n.
    ids = [7] * 10
    assert rep_l(ids, window=128) == pytest.approx(9 / 10)


def test_rep_l_respects_the_window_boundary():
    # Token 0 falls OUT of a window of size 2 by the time we reach position 3.
    # ids = [9, 1, 2, 9], window=2.
    #  i=0 lookback=[]        -> not repeat
    #  i=1 lookback=[9]       -> not repeat
    #  i=2 lookback=[9,1]     -> not repeat
    #  i=3 lookback=[1,2]     -> 9 NOT in (fell out of the window) -> not repeat
    assert rep_l([9, 1, 2, 9], window=2) == 0.0
    # The SAME sequence with window=3 puts 9 back in range at i=3: lookback=[9,1,2].
    assert rep_l([9, 1, 2, 9], window=3) == pytest.approx(0.25)


def test_rep_l_empty_sequence_is_zero():
    assert rep_l([], window=128) == 0.0


def test_rep_l_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        rep_l([1, 2, 3], window=0)


# ── distinct_n ───────────────────────────────────────────────────────────────────────
def test_distinct_n_worked_example_two_rows():
    # Row A: 8 zeros -> unigrams: 8 total, 1 unique ({0}).
    # Row B: [1..8]  -> unigrams: 8 total, 8 unique.
    # Corpus distinct-1 = (1 + 8) / (8 + 8) = 9/16 = 0.5625 exactly.
    rows = [[0] * 8, list(range(1, 9))]
    assert distinct_n(rows, 1) == pytest.approx(9 / 16)


def test_distinct_n_pools_across_rows_not_averages_per_row():
    # Two rows that are each internally repeat-free but SHARE every bigram with each
    # other: pooling must dedupe across rows, not just within a row.
    rows = [[1, 2, 3], [1, 2, 3]]
    # bigrams per row: (1,2),(2,3) -> 2 unique total across the pooled 4.
    assert distinct_n(rows, 2) == pytest.approx(2 / 4)


def test_distinct_n_empty_corpus_is_zero_not_crash():
    assert distinct_n([[1], [2]], 4) == 0.0   # every row shorter than n=4
    assert distinct_n([], 1) == 0.0


# ── bootstrap_ci ─────────────────────────────────────────────────────────────────────
def test_bootstrap_ci_constant_array_collapses_to_the_constant():
    lo, hi = bootstrap_ci([3.0] * 50, n_boot=200, seed=1)
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(3.0)


def test_bootstrap_ci_brackets_the_true_mean_on_a_known_distribution():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=10.0, scale=1.0, size=500).tolist()
    lo, hi = bootstrap_ci(values, n_boot=2000, seed=2)
    assert lo < 10.0 < hi, f"95% CI [{lo}, {hi}] should bracket the true mean 10.0"
    # The CI must not be absurdly wide for n=500, sigma=1 (SEM ~ 0.045).
    assert hi - lo < 0.6, f"CI width {hi - lo} is implausibly wide for n=500"


def test_bootstrap_ci_single_value():
    lo, hi = bootstrap_ci([5.0])
    assert (lo, hi) == (5.0, 5.0)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    vals = list(range(30))
    a = bootstrap_ci(vals, n_boot=500, seed=7)
    b = bootstrap_ci(vals, n_boot=500, seed=7)
    assert a == b


# ── bootstrap_ci_rows (corpus-level statistics) ─────────────────────────────────────
def test_bootstrap_ci_rows_constant_corpus_collapses():
    rows = [[1, 2, 3, 4]] * 20
    lo, hi = bootstrap_ci_rows(rows, lambda rs: distinct_n(rs, 2), n_boot=100, seed=1)
    exact = distinct_n(rows, 2)
    assert lo == pytest.approx(exact)
    assert hi == pytest.approx(exact)


def test_bootstrap_ci_rows_matches_mean_bootstrap_for_a_mean_statistic():
    # bootstrap_ci_rows with stat_fn = mean of a per-row scalar must agree with
    # bootstrap_ci on that same list of scalars (same resampling scheme, same seed).
    vals = [float(x) for x in range(40)]
    rows = [[v] for v in vals]  # one "row" per scalar
    a = bootstrap_ci(vals, n_boot=300, seed=3)
    b = bootstrap_ci_rows(rows, lambda rs: float(np.mean([r[0] for r in rs])),
                          n_boot=300, seed=3)
    assert a == pytest.approx(b)


def test_bootstrap_ci_rows_single_row():
    lo, hi = bootstrap_ci_rows([[1, 1, 2]], lambda rs: distinct_n(rs, 1))
    exact = distinct_n([[1, 1, 2]], 1)
    assert (lo, hi) == (exact, exact)


def test_bootstrap_ci_rows_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci_rows([], lambda rs: 0.0)
