"""Bucket assignment and distance-bin unit tests for `lab/divergence/exact_recall_gap.py`.

Hand-built token arrays, no tokenizer, no shards, no npz files. Checks `row_buckets`
(the far bigram / far token / novel classification and its reported distance) and
`dist_submask` (the 33-64 / 65-256 / 257+ distance split) against values computed by
hand, per the prereg's EXISTENCE-based bucket definition
(lab/experiments/planned/2026-10-04-exact-recall-gap-probe.md): a bucket is "far" if
SOME earlier occurrence sits at distance >= far_gap, not necessarily the globally
nearest occurrence at any distance.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "lab", "divergence"))
from exact_recall_gap import DIST_BINS, dist_submask, row_buckets  # noqa: E402

NOVEL, FAR_TOKEN, FAR_BIGRAM = 0, 1, 2


def test_row_buckets_hand_computed():
    # positions: 0  1  2  3  4   5   6  7  8
    x = np.array([7, 8, 9, 7, 8, 10, 7, 8, 11])
    # far_gap=3, min_i=3 (small values so the hand trace below stays short):
    #
    # i=3 x=7: nearest tok j<=0 -> j=0 (dist 3); no (9,7) bigram before -> FAR_TOKEN, 3
    # i=4 x=8: nearest tok j<=1 -> j=1 (dist 3); (7,8) bigram at j=1<=1 -> FAR_BIGRAM, 3
    # i=5 x=10: never seen before -> NOVEL, -1
    # i=6 x=7: nearest tok j<=3 -> j=3 (dist 3); no (10,7) bigram before -> FAR_TOKEN, 3
    # i=7 x=8: nearest tok j<=4 -> j=4 (dist 3); (7,8) bigram at j=4<=4 -> FAR_BIGRAM, 3
    # i=8 x=11: never seen before -> NOVEL, -1
    local_idx, bucket, dist = row_buckets(x, min_i=3, far_gap=3)

    np.testing.assert_array_equal(local_idx, [2, 3, 4, 5, 6, 7])  # i-1 for i=3..8
    np.testing.assert_array_equal(bucket,
                                   [FAR_TOKEN, FAR_BIGRAM, NOVEL, FAR_TOKEN, FAR_BIGRAM, NOVEL])
    np.testing.assert_array_equal(dist, [3, 3, -1, 3, 3, -1])


def test_close_repeat_is_not_far():
    """A token (or bigram) that repeats, but only within the far_gap window, is NOVEL —
    the EXISTENCE test must gate on distance, not on "did this ever repeat"."""
    # x[0]=5, x[2]=5 — repeats at distance 2, which is < far_gap=3.
    x = np.array([5, 6, 5, 9, 9])
    local_idx, bucket, dist = row_buckets(x, min_i=2, far_gap=3)
    # i=2 (local_idx=1): x[2]=5 repeats x[0] at distance 2 < 3 -> NOVEL, not FAR_TOKEN.
    # i=3 (local_idx=2): x[3]=9, first occurrence -> NOVEL.
    # i=4 (local_idx=3): x[4]=9 repeats x[3] at distance 1 < 3 -> NOVEL, not FAR_TOKEN.
    np.testing.assert_array_equal(local_idx, [1, 2, 3])
    np.testing.assert_array_equal(bucket, [NOVEL, NOVEL, NOVEL])
    np.testing.assert_array_equal(dist, [-1, -1, -1])


def test_bigram_requires_the_full_pair_not_just_the_token():
    """x[i] matching something far back is only a FAR_BIGRAM if (x[i-1], x[i]) also
    matched there; a lone token match with a different predecessor is FAR_TOKEN."""
    # positions: 0  1  2  3   4  5
    x = np.array([1, 2, 3, 99, 4, 2])
    # i=5 (local_idx=4): x[5]=2, matches x[1]=2 at distance 4 (>= far_gap=3) -> token
    # match exists, but the pair (x[4], x[5]) = (4, 2) != (x[0], x[1]) = (1, 2), so no
    # bigram match -> FAR_TOKEN, distance 4.
    local_idx, bucket, dist = row_buckets(x, min_i=4, far_gap=3)
    assert list(local_idx) == [3, 4]
    assert bucket[-1] == FAR_TOKEN
    assert dist[-1] == 4


def test_dist_submask_boundaries():
    bucket_all = np.array([FAR_BIGRAM] * 7)
    dist_all = np.array([32, 33, 64, 65, 256, 257, 1000])
    # 32 is below the far_gap floor used anywhere in this probe (far_gap=33 by
    # default) and must land in no DIST_BINS bucket.
    none_mask = np.zeros(7, dtype=bool)
    for _, lo, hi in DIST_BINS:
        none_mask |= dist_submask(bucket_all, dist_all, FAR_BIGRAM, lo, hi)
    assert not none_mask[0]  # dist=32 -> no bin
    assert list(DIST_BINS[0][:1]) == ["33-64"]
    m_33_64 = dist_submask(bucket_all, dist_all, FAR_BIGRAM, 33, 64)
    m_65_256 = dist_submask(bucket_all, dist_all, FAR_BIGRAM, 65, 256)
    m_257p = dist_submask(bucket_all, dist_all, FAR_BIGRAM, 257, None)
    np.testing.assert_array_equal(m_33_64, [False, True, True, False, False, False, False])
    np.testing.assert_array_equal(m_65_256, [False, False, False, True, True, False, False])
    np.testing.assert_array_equal(m_257p, [False, False, False, False, False, True, True])
    # every distance >= 33 lands in exactly one bin.
    far = dist_all >= 33
    coverage = m_33_64 | m_65_256 | m_257p
    np.testing.assert_array_equal(coverage, far)


def test_dist_submask_respects_bucket_code():
    # same distances, but FAR_TOKEN rows must not show up when querying FAR_BIGRAM.
    bucket_all = np.array([FAR_TOKEN, FAR_BIGRAM])
    dist_all = np.array([100, 100])
    m = dist_submask(bucket_all, dist_all, FAR_BIGRAM, 65, 256)
    np.testing.assert_array_equal(m, [False, True])
