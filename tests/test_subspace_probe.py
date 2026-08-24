"""Contracts for the input-energy probes.

`test_energy_curve_is_a_cumulative_share` and `test_curve_ends_at_one` pin the self-test
the probe relies on: if a curve does not reach exactly 1.0 at k = in_dim the eigenbasis is
wrong and every other number is meaningless.

`test_eff_rank_is_participation_ratio` exists because the first version of the probe used a
99 %-energy PROJECTOR instead, which on the real model spans 935 of 1024 directions and
therefore captures everything trivially. The participation ratio is what actually
distinguishes a concentrated input distribution from a spread one.

`test_centred_rows_*` pin the slot-row statistic: the rows share a common mean by
construction, so an uncentred cosine measures the mean and not the diversity.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.divergence.slot_rows_probe import find_e_slot, row_stats      # noqa: E402
from lab.divergence.subspace_probe import (                            # noqa: E402
    adamix_update, at_k, energy_curve, input_eigenbasis,
)


# ── input concentration ──────────────────────────────────────────────────────────────
def test_eff_rank_is_participation_ratio():
    """n equal eigenvalues -> eff_rank n; one dominant -> eff_rank ~1."""
    flat = input_eigenbasis(torch.eye(16))[1]
    assert flat["eff_rank"] == pytest.approx(16.0, rel=1e-4)
    d = torch.zeros(16, 16)
    d[0, 0] = 1.0
    d[1:, 1:] = torch.eye(15) * 1e-8
    assert input_eigenbasis(d)[1]["eff_rank"] == pytest.approx(1.0, abs=1e-4)


def test_k_quantiles_count_directions_not_energy():
    lam = torch.zeros(10, 10)
    for i in range(10):
        lam[i, i] = 1.0 if i < 5 else 0.0        # 5 equal directions, 5 empty
    c = input_eigenbasis(lam)[1]
    assert c["k50"] == 3       # 3 of 5 needed to EXCEED half
    assert c["k90"] == 5
    assert c["eff_rank"] == pytest.approx(5.0, rel=1e-4)


def test_eigenbasis_is_ordered_descending():
    g = torch.diag(torch.tensor([1.0, 9.0, 3.0]))
    V, _ = input_eigenbasis(g)
    # the first eigenvector must be the one with eigenvalue 9, i.e. axis 1
    assert int(V[:, 0].abs().argmax()) == 1


# ── energy curve ─────────────────────────────────────────────────────────────────────
def test_energy_curve_is_a_cumulative_share():
    V = torch.eye(4)
    m = torch.tensor([[3.0, 0.0, 4.0, 0.0]])     # energy 9 on axis 0, 16 on axis 2
    c = energy_curve(m, V)
    assert at_k(c, 1) == pytest.approx(9 / 25)
    assert at_k(c, 2) == pytest.approx(9 / 25)
    assert at_k(c, 3) == pytest.approx(1.0)


def test_curve_ends_at_one():
    """The self-test the probe gates on."""
    torch.manual_seed(0)
    V, _ = input_eigenbasis(torch.randn(12, 40) @ torch.randn(40, 12) + 12 * torch.eye(12))
    for m in (torch.randn(5, 12), torch.randn(1, 12) * 1e-9):
        assert float(energy_curve(m, V)[-1]) == pytest.approx(1.0, abs=1e-5)


def test_curve_is_monotone():
    torch.manual_seed(1)
    V = torch.linalg.qr(torch.randn(8, 8))[0]
    c = energy_curve(torch.randn(3, 8), V)
    assert torch.all(c[1:] >= c[:-1] - 1e-6)


def test_energy_curve_is_basis_dependent():
    """A curve that ignored V would not distinguish these, and the probe would be blind."""
    m = torch.tensor([[1.0, 0.0]])
    assert at_k(energy_curve(m, torch.eye(2)), 1) == pytest.approx(1.0)
    flipped = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert at_k(energy_curve(m, flipped), 1) == pytest.approx(0.0, abs=1e-9)


# ── the update ───────────────────────────────────────────────────────────────────────
def test_adamix_update_is_eps_outside():
    """eps OUTSIDE the sqrt. MORPH sets ademamix_eps_inside: false; the floored form is a
    ~100x error at MORPH's gradient scale."""
    g = torch.full((4,), 1e-5)
    m2 = torch.zeros(4)
    nu = torch.full((4,), 1e-12)
    u = adamix_update(g, m2, nu, alpha=1.0, bc2=1.0, eps=1e-8)
    assert float(u[0]) == pytest.approx(1e-5 / (1e-6 + 1e-8), rel=1e-4)


def test_adamix_update_uses_alpha_and_bc2():
    g, m2, nu = torch.ones(3), torch.ones(3), torch.full((3,), 4.0)
    # bc2 = 1 -> denom = 2 + eps; (1 + 3*1)/2 = 2
    assert float(adamix_update(g, m2, nu, 3.0, 1.0, 0.0)[0]) == pytest.approx(2.0)
    # bc2 = 0.25 -> nu/bc2 = 16 -> denom 4 -> 4/4 = 1
    assert float(adamix_update(g, m2, nu, 3.0, 0.25, 0.0)[0]) == pytest.approx(1.0)


# ── slot rows ────────────────────────────────────────────────────────────────────────
def test_centred_rows_see_diversity_the_raw_cosine_hides():
    """Rows = one shared mean plus small distinct offsets. Raw cosine is near 1 and says
    nothing; the centred statistics report the offsets, which is the quantity of interest."""
    torch.manual_seed(0)
    mean = torch.ones(1, 32) * 10.0
    e = mean + torch.randn(16, 32) * 0.01
    st = row_stats(e)
    assert st["mean_pairwise_cos_raw"] > 0.99
    assert abs(st["mean_pairwise_cos_centred"]) < 0.2
    assert st["eff_rank_centred"] > 8


def test_centred_eff_rank_collapses_when_rows_do():
    mean = torch.ones(1, 32)
    direction = torch.randn(1, 32)
    e = mean + torch.linspace(-1, 1, 16).unsqueeze(1) * direction   # rank 1 after centring
    assert row_stats(e)["eff_rank_centred"] == pytest.approx(1.0, abs=1e-3)


def test_find_e_slot_rejects_the_shared_vector():
    with pytest.raises(ValueError):
        find_e_slot({"tul.E_slot": torch.zeros(768)})


def test_find_e_slot_rejects_a_missing_key():
    with pytest.raises(KeyError):
        find_e_slot({"tul.E_mask": torch.zeros(4, 768)})


def test_find_e_slot_strips_orig_mod():
    e = find_e_slot({"_orig_mod.tul.E_slot": torch.zeros(64, 768)})
    assert tuple(e.shape) == (64, 768)
