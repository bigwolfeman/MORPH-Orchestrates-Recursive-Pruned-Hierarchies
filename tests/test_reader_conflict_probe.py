"""Contracts for the reader-gradient-conflict probe.

The load-bearing one is `test_alignment_normalises_out_the_reader_count`. Raw
`||sum g_r|| / sum ||g_r||` falls like `1/sqrt(K)` for INDEPENDENT gradients, so comparing
it between slots with different reader counts compares the reader counts. Every verdict in
`docs/experiments/failures/2026-08-24-tul-reader-gradient-conflict.md` is read off the
`sqrt(K)`-normalised statistic for that reason.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.divergence.reader_conflict_probe import alignment, conflict_stats  # noqa: E402


def test_alignment_normalises_out_the_reader_count():
    """K orthogonal unit gradients: ||sum|| = sqrt(K), sum||.|| = K, so conflict =
    1/sqrt(K) and alignment = 1 for every K. That invariance is the whole point."""
    for k in (4, 16, 64, 256):
        g = [torch.eye(k)[i] for i in range(k)]
        s = conflict_stats(g, None)
        assert s["conflict"] == pytest.approx(1 / math.sqrt(k), rel=1e-5)
        assert s["alignment"] == pytest.approx(1.0, rel=1e-5)


def test_identical_readers_are_perfect_agreement():
    g = [torch.tensor([1.0, 0.0, 0.0])] * 8
    s = conflict_stats(g, None)
    assert s["conflict"] == pytest.approx(1.0)
    assert s["alignment"] == pytest.approx(math.sqrt(8), rel=1e-5)
    assert s["mean_pair_cos"] == pytest.approx(1.0, abs=1e-5)


def test_opposed_readers_cancel():
    g = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    s = conflict_stats(g, None)
    assert s["conflict"] == pytest.approx(0.0, abs=1e-6)
    assert s["alignment"] == pytest.approx(0.0, abs=1e-6)
    assert s["mean_pair_cos"] == pytest.approx(-1.0, abs=1e-5)


def test_alignment_helper_matches_the_formula():
    assert alignment(3.0, 12.0, 16) == pytest.approx(3.0 / 12.0 * 4.0)
    assert math.isnan(alignment(1.0, 0.0, 4))
    assert math.isnan(alignment(1.0, 4.0, 0))


def test_route_frac_compares_direct_against_the_reader_SUM_of_norms():
    """route_frac must divide by sum of NORMS, not by the norm of the sum. Dividing by the
    cancelled sum would report the direct route as dominant whenever readers disagree,
    which is exactly the conclusion the probe exists to test."""
    readers = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]   # norm of sum = 0
    s = conflict_stats(readers, torch.tensor([0.0, 1.0]))
    assert s["direct_norm"] == pytest.approx(1.0)
    assert s["route_frac"] == pytest.approx(1.0 / 3.0, rel=1e-5)      # 1 / (1 + 2)


def test_cos_direct_readers_is_signed():
    readers = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    assert conflict_stats(readers, torch.tensor([1.0, 0.0]))["cos_direct_readers"] \
        == pytest.approx(1.0, abs=1e-5)
    assert conflict_stats(readers, torch.tensor([-1.0, 0.0]))["cos_direct_readers"] \
        == pytest.approx(-1.0, abs=1e-5)
    assert conflict_stats(readers, torch.tensor([0.0, 1.0]))["cos_direct_readers"] \
        == pytest.approx(0.0, abs=1e-5)


def test_gradients_are_flattened_not_reduced():
    """A slot gradient is [n_streams, C] under hyper-connections. The statistic is over the
    WHOLE tensor, so a 2-D gradient must behave exactly like its flattening.

    The two gradients here are orthogonal when flattened (alignment 1.0) but become
    IDENTICAL under any per-row reduction (alignment sqrt(2)), so a probe that reduced
    instead of flattening would report agreement that is not there."""
    a = [torch.tensor([[1.0, 0.0], [0.0, 0.0]]), torch.tensor([[0.0, 1.0], [0.0, 0.0]])]
    flat = [g.reshape(-1) for g in a]
    assert conflict_stats(a, None)["alignment"] == pytest.approx(
        conflict_stats(flat, None)["alignment"], rel=1e-6)
    assert conflict_stats(a, None)["alignment"] == pytest.approx(1.0, rel=1e-5)
    # and the reduction it must NOT do would give a different, larger answer
    reduced = [g.mean(-1) for g in a]
    assert conflict_stats(reduced, None)["alignment"] == pytest.approx(math.sqrt(2), rel=1e-5)


def test_empty_reader_set_returns_nothing():
    assert conflict_stats([], None) == {}
