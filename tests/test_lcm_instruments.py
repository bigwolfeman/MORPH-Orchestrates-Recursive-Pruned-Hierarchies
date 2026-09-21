"""The three LCM instruments' pure functions, on hand-built vectors.

Every test asserts a VALUE the maths fixes, not a shape:

* ``cos_flat`` flattens the CELL axis before the cosine, so two stacks that disagree per
  cell but agree as a whole read 1;
* a prediction that IS a real span round trips to itself — ``move`` 0 and ``gap`` 0 —
  while a prediction that is the AVERAGE of two real spans is pulled onto one of them by
  a snap-to-manifold decoder, which is LCM's Base-LCM result reproduced in miniature;
* ``context_mi`` is a per-token difference, so doubling a span's length halves its MI;
* ``permute_valid`` moves a valid slot's cells and leaves an invalid slot's exactly alone;
* ``pick_ordinals`` reaches the END of a row, not only its first slots;
* ``pair_distinct`` ignores everything past a span's own length;
* ``mean_pairwise_dist`` reads 1 on orthogonal decodes and 0 on identical ones;
* ``span_pool`` averages the NEXT span's token states and zeroes a slot with no span.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab", "divergence"))
from code_context_mi_probe import context_mi, permute_valid, pick_ordinals  # noqa: E402
from code_roundtrip_probe import cos_flat, roundtrip_readings  # noqa: E402
from fan_stream_decode_probe import (first_token_agree, mean_pairwise_dist,  # noqa: E402
                                     pair_distinct, span_pool)


# ────────────────────────── cos_flat ──────────────────────────

def test_cos_flat_is_over_the_whole_cell_stack():
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])          # [1, 2, 2]
    assert cos_flat(a, a).item() == pytest.approx(1.0, abs=1e-12)
    # b matches a per CELL (cosine 1 on both), so a per-cell average would read 1.0.
    # Flattened, <a, b> = 3 and |a||b| = sqrt(2) sqrt(5): the whole-stack cosine is
    # 0.9487. This is the flattening choice, asserted rather than described.
    b = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    assert cos_flat(a, b).item() == pytest.approx(3.0 / (2.0 ** 0.5 * 5.0 ** 0.5), abs=1e-12)
    c = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    assert cos_flat(a, c).item() == pytest.approx(0.0, abs=1e-12)
    d = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    assert cos_flat(a, d).item() == pytest.approx(1.0, abs=1e-12)   # scale free


def test_cos_flat_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        cos_flat(torch.zeros(2, 3), torch.zeros(2, 4))


# ────────────────────────── the round trip ──────────────────────────

def _snap(z, manifold):
    """The toy decoder+encoder: the nearest point of ``manifold`` by cosine.

    Ties go to the LAST maximiser, so the average of two truths lands on the second one.
    """
    x = torch.nn.functional.normalize(z.reshape(z.shape[0], -1).double(), dim=1)
    m = torch.nn.functional.normalize(manifold.reshape(manifold.shape[0], -1).double(), dim=1)
    sim = x @ m.t()
    n = sim.shape[1]
    # argmax with ties to the last index
    pick = (n - 1) - sim.flip(1).argmax(dim=1)
    return manifold[pick]


def test_a_real_span_round_trips_to_itself():
    t = torch.randn(5, 2, 8, generator=torch.Generator().manual_seed(0))
    z_re = _snap(t, t)
    r = roundtrip_readings(t, z_re, t)
    assert torch.allclose(r["move"], torch.zeros(5).double(), atol=1e-12)
    assert torch.allclose(r["gap"], torch.zeros(5).double(), atol=1e-12)
    assert torch.allclose(r["cos_hat_true"], torch.ones(5).double(), atol=1e-12)


def test_the_average_of_two_truths_is_pulled_off_its_own_point():
    g = torch.Generator().manual_seed(7)
    manifold = torch.randn(2, 2, 8, generator=g)
    t0, t1 = manifold[0:1], manifold[1:2]
    z_hat = 0.5 * (t0 + t1)

    # each truth's own round trip moves nothing
    for t in (t0, t1):
        r = roundtrip_readings(t, _snap(t, manifold), t)
        assert r["move"].item() == pytest.approx(0.0, abs=1e-12)

    # the average is snapped onto the OTHER truth (ties to the last index), so it moves
    r = roundtrip_readings(z_hat, _snap(z_hat, manifold), t0)
    assert r["move"].item() > 0.1
    # ... and lands further from the truth than the prediction was: LCM's l2-r > l2
    assert r["gap"].item() > 0.0
    assert r["d_re"].item() > r["d_hat"].item()


def test_gap_is_d_re_minus_d_hat_exactly():
    g = torch.Generator().manual_seed(3)
    z_hat, z_re, z_true = (torch.randn(4, 2, 6, generator=g) for _ in range(3))
    r = roundtrip_readings(z_hat, z_re, z_true)
    assert torch.allclose(r["gap"], r["d_re"] - r["d_hat"], atol=1e-12)
    assert torch.allclose(r["d_hat"], 1.0 - r["cos_hat_true"], atol=1e-12)
    assert torch.allclose(r["move"], 1.0 - r["cos_re_hat"], atol=1e-12)


# ────────────────────────── context MI ──────────────────────────

def test_context_mi_is_per_token():
    out = context_mi(np.array([12.0, 12.0]), np.array([8.0, 8.0]), np.array([4.0, 8.0]))
    assert out["mi"].tolist() == pytest.approx([1.0, 0.5])
    assert out["ce_cut"].tolist() == pytest.approx([3.0, 1.5])
    assert out["ce_full"].tolist() == pytest.approx([2.0, 1.0])


def test_context_mi_is_zero_when_the_context_does_nothing():
    out = context_mi(np.array([5.0]), np.array([5.0]), np.array([5.0]))
    assert out["mi"].item() == pytest.approx(0.0)


def test_context_mi_can_be_negative():
    # the cut makes the span CHEAPER: LCM's negative MI (Base-LCM, C4: -0.105)
    out = context_mi(np.array([4.0]), np.array([6.0]), np.array([2.0]))
    assert out["mi"].item() == pytest.approx(-1.0)


def test_context_mi_refuses_an_empty_span():
    with pytest.raises(ValueError):
        context_mi(np.array([1.0]), np.array([1.0]), np.array([0.0]))


# ────────────────────────── permute_valid / pick_ordinals ──────────────────────────

def test_permute_valid_moves_valid_cells_and_leaves_invalid_ones():
    z = torch.arange(12.0).reshape(1, 3, 4)                      # slots 0,1,2
    valid = torch.tensor([[True, True, False]])
    perm = torch.tensor([[2, 0, 1]])
    out = permute_valid(z, valid, perm)
    assert out[0, 0].tolist() == z[0, 2].tolist()                # valid: takes slot 2
    assert out[0, 1].tolist() == z[0, 0].tolist()                # valid: takes slot 0
    assert out[0, 2].tolist() == z[0, 2].tolist()                # INVALID: untouched


def test_pick_ordinals_spans_the_row_and_pads_with_minus_one():
    elig = torch.zeros(2, 10, dtype=torch.bool)
    elig[0, [1, 3, 5, 7, 9]] = True
    elig[1, [4, 6]] = True
    out = pick_ordinals(elig, 3)
    assert out[0].tolist() == [1, 5, 9]          # first, middle, LAST — not [1, 3, 5]
    assert out[1].tolist() == [4, 6, -1]


def test_pick_ordinals_on_an_empty_row():
    out = pick_ordinals(torch.zeros(1, 4, dtype=torch.bool), 2)
    assert out[0].tolist() == [-1, -1]


# ────────────────────────── the fan's decode statistics ──────────────────────────

def test_pair_distinct_ignores_padding_past_the_span_length():
    # both decodes agree on the first 2 tokens and differ only at position 2, which is
    # PAST the span's length: they must read identical.
    tokens = torch.tensor([[[5, 6, 1], [5, 6, 9]]])              # [S=1, K=2, J=3]
    allsame, rate = pair_distinct(tokens, torch.tensor([2]))
    assert bool(allsame[0]) is False
    assert rate[0].item() == pytest.approx(0.0)
    allsame, rate = pair_distinct(tokens, torch.tensor([3]))
    assert bool(allsame[0]) is True
    assert rate[0].item() == pytest.approx(1.0)


def test_pair_distinct_counts_pairs_not_decodes():
    # three decodes, two of them identical: 1 of the 3 pairs differ... actually 2 of 3.
    tokens = torch.tensor([[[1, 1], [1, 1], [2, 2]]])            # K = 3
    allsame, rate = pair_distinct(tokens, torch.tensor([2]))
    assert bool(allsame[0]) is False
    assert rate[0].item() == pytest.approx(2.0 / 3.0)


def test_first_token_agree_ignores_everything_after_position_zero():
    # all three decodes open with 7 and then diverge: agreement on token 0 is still 1.0
    same_open = torch.tensor([[[7, 1, 2], [7, 3, 4], [7, 5, 6]]])
    assert first_token_agree(same_open)[0].item() == pytest.approx(1.0)
    # one decode opens differently: 0.0, even though the rest matches
    diff_open = torch.tensor([[[7, 1, 2], [8, 1, 2], [7, 1, 2]]])
    assert first_token_agree(diff_open)[0].item() == pytest.approx(0.0)


def test_mean_pairwise_dist_reads_zero_on_identical_and_one_on_orthogonal():
    same = torch.ones(1, 3, 4)
    assert mean_pairwise_dist(same)[0].item() == pytest.approx(0.0, abs=1e-12)
    orth = torch.eye(3).unsqueeze(0)                              # [1, 3, 3]
    assert mean_pairwise_dist(orth)[0].item() == pytest.approx(1.0, abs=1e-12)
    # one pair identical, two pairs orthogonal -> 2/3
    mixed = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    assert mean_pairwise_dist(mixed)[0].item() == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_span_pool_averages_the_next_span_and_zeroes_a_slot_with_no_span():
    # L = 5: token(bag 0), slot(bag 0), token(bag 1), token(bag 1), slot(bag 1)
    xn = torch.tensor([[[1.0], [99.0], [2.0], [4.0], [99.0]]])          # [1, 5, 1]
    bag = torch.tensor([[0, 0, 1, 1, 1]])
    smask = torch.tensor([[False, True, False, False, True]])
    out = span_pool(xn, bag, smask, 2)                                  # [1, 2, 1]
    assert out[0, 0, 0].item() == pytest.approx(3.0)       # slot 0 -> span 1: (2 + 4)/2
    assert out[0, 1, 0].item() == pytest.approx(0.0)       # slot 1 -> no span 2
