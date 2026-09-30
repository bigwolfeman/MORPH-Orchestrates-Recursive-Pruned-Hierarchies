"""Repetition-and-diversity metrics for generated text (docs/gen-diversity task, 2026-09-30).

Wolfe: "Main thing we can measure at this early training is repetition and diversity."
This module holds the METRIC FUNCTIONS only — pure token-id math, no model, no torch
device, so they are cheap to unit-test against hand-built sequences with exactly
computable answers (`tests/test_gen_diversity.py`). The eval driver that loads
checkpoints, generates continuations and calls these lives in
`lab/divergence/gen_diversity_eval.py`.

Four families:

  seq_rep_n(ids, n)        per-CONTINUATION: 1 - unique n-grams / total n-grams
                            (Welleck, Holtzman, Ma, Zettlemoyer & Bosselut 2019,
                            "Neural Text Generation with Unlikelihood Training", eq. 1's
                            seq-rep-n). Near 0 for natural text, -> 1 for a collapsed
                            repetition loop. Average this over continuations.

  rep_l(ids, window)       per-CONTINUATION: fraction of positions i whose token already
                            occurred in the PRECEDING `window` tokens of the SAME
                            continuation (a windowed, position-level cousin of Welleck et
                            al.'s token-level "rep" statistic — their rep is read off the
                            greedy next-token argmax against the running context; this one
                            is read off the emitted sequence itself against a fixed
                            lookback, which is what a fixed-length continuation affords).
                            Position 0..window-1 have a shrinking lookback (whatever
                            history exists); the first token always scores "not a repeat"
                            (empty lookback).

  distinct_n(seqs, n)      CORPUS-level (Li, Galley, Brockett, Gao, Dolan 2016, "A
                            Diversity-Promoting Objective Function for Neural
                            Conversation Models"): unique n-grams / total n-grams, pooled
                            over ALL sequences passed in. This is the number that is
                            comparable ACROSS models at a fixed N and length, because it
                            pools the whole sample instead of averaging a per-row ratio
                            that a single very short row can dominate.

  bootstrap_ci(values)     nonparametric percentile bootstrap 95% CI over the per-row
                            values (rows are prompts/continuations, resampled with
                            replacement).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = ["ngrams", "seq_rep_n", "rep_l", "distinct_n", "bootstrap_ci", "bootstrap_ci_rows"]


def ngrams(ids: list[int], n: int) -> list[tuple[int, ...]]:
    """All contiguous n-grams of `ids`, in order. Empty list if len(ids) < n."""
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(ids) < n:
        return []
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


def seq_rep_n(ids: list[int], n: int) -> float:
    """1 - |unique n-grams(ids)| / |n-grams(ids)| (Welleck et al. 2019, seq-rep-n).

    0.0 when the continuation is shorter than n (nothing to repeat; the caller — not
    this function — decides whether such a short row belongs in the average at all).
    1.0 - 1/T when every one of T n-grams is the SAME n-gram (total collapse); never
    exactly 1.0 for a finite sequence, since a sequence of length >= n always has at
    least one unique n-gram.
    """
    grams = ngrams(ids, n)
    if not grams:
        return 0.0
    n_unique = len(set(grams))
    return 1.0 - n_unique / len(grams)


def rep_l(ids: list[int], window: int = 128) -> float:
    """Fraction of tokens that already occurred in the previous `window` tokens of the
    SAME continuation (a fixed-lookback, position-level repeat rate).

    For position i (0-indexed), the lookback is ids[max(0, i-window):i] — i.e. up to
    `window` tokens strictly BEFORE i, never including i itself. Token 0 always has an
    empty lookback and therefore never counts as a repeat. Returns 0.0 for an empty
    sequence.
    """
    if window <= 0:
        raise ValueError(f"window must be >= 1, got {window}")
    n = len(ids)
    if n == 0:
        return 0.0
    seen_in_window = 0
    lo = 0
    window_set: Counter = Counter()
    for i in range(n):
        if ids[i] in window_set:
            seen_in_window += 1
        window_set[ids[i]] += 1
        # Slide the window: once it holds `window` tokens ending at i, drop the token
        # that is about to fall off the back for position i+1.
        if i - lo + 1 > window:
            drop = ids[lo]
            window_set[drop] -= 1
            if window_set[drop] == 0:
                del window_set[drop]
            lo += 1
    return seen_in_window / n


def distinct_n(seqs: list[list[int]], n: int) -> float:
    """Corpus-level distinct-n (Li et al. 2016): unique n-grams / total n-grams, pooled
    over every sequence in `seqs`. 0.0 if no sequence has >= n tokens.
    """
    total = 0
    uniq: set = set()
    for ids in seqs:
        grams = ngrams(ids, n)
        total += len(grams)
        uniq.update(grams)
    if total == 0:
        return 0.0
    return len(uniq) / total


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Nonparametric percentile bootstrap `(1-alpha)` CI of the MEAN of `values`.

    Resamples `values` with replacement `n_boot` times, takes the mean of each
    resample, and returns the `(alpha/2, 1-alpha/2)` percentiles of that distribution.
    A constant array collapses to `(value, value)` — that is the correct answer, not a
    bug, since every resample of a constant array has the same mean.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_rows(rows: list, stat_fn, n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat_fn(resampled_rows)` over ROWS (not over a
    pre-reduced per-row scalar).

    For a CORPUS-level statistic like `distinct_n`, which needs the whole pooled corpus
    to compute (it is not a mean of a per-row number), `bootstrap_ci` cannot be used
    directly: resample the ROWS themselves with replacement, recompute the statistic on
    each resampled corpus, and take percentiles of THAT distribution. `stat_fn` must
    accept a list of rows and return one float, e.g. ``lambda rs: distinct_n(rs, 4)``.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("bootstrap_ci_rows needs at least one row")
    if n == 1:
        v = float(stat_fn(rows))
        return v, v
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = stat_fn([rows[i] for i in idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
