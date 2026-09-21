"""``lab/divergence/fan_gain_probe.py``'s pure function against a map whose mean and
deviation gains are known exactly.

The probe claims to separate two gains of one core pass: what the map does to a
perturbation that is the SAME on all K streams of a slot (the mean direction, the K
averaging projector P) and what it does to a perturbation whose stream-sum is zero (the
deviation subspace, Q = I - P). The synthetic map here is

    Z' = a * (P Z) + b * (Q Z) R

with ``R`` a fixed orthogonal matrix on the channel axis. Its Jacobian is exactly ``a``
on every mean direction and exactly ``b`` on every deviation direction, whatever ``R`` is
— so the rotation is in the map on purpose: a probe that read the direction rather than
the magnitude would fail on it. No model, no checkpoint, no GPU.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))

from fan_gain_probe import _d_eff, pass_gains  # noqa: E402


def _split_map(m: int, c: int, a: float, b: float, seed: int = 0):
    """``jvp(v) -> a P v + b (Q v) R`` on ``[B, S*M, C]`` states, plus ``R``."""
    g = torch.Generator().manual_seed(seed)
    r, _ = torch.linalg.qr(torch.randn(c, c, generator=g, dtype=torch.float64))
    r = r.float()

    def jvp(v: torch.Tensor) -> torch.Tensor:
        bsz, sm = v.shape[0], v.shape[1]
        z = v.reshape(bsz, sm // m, m, c)
        mu = z.mean(dim=2, keepdim=True)
        dev = z - mu
        out = a * mu.expand_as(z) + b * (dev @ r)
        return out.reshape(bsz, sm, c)

    return jvp, r


def _state(bsz: int, s: int, m: int, c: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(bsz, s * m, c, generator=g)


@pytest.mark.parametrize("a,b", [(0.5, 1.0), (1.0, 0.3), (0.25, 0.9)])
def test_reads_the_two_gains_exactly(a, b):
    m, c, s, bsz = 4, 8, 5, 2
    jvp, _ = _split_map(m, c, a, b)
    state = _state(bsz, s, m, c)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=8, seed=3)

    assert g["gain_mean_dir"] == pytest.approx(a, abs=1e-5)
    assert g["gain_dev_dir"] == pytest.approx(b, abs=1e-5)
    assert g["ratio_dev_over_mean"] == pytest.approx(b / a, abs=1e-5)
    # the state's OWN components are read by the same two gains
    assert g["gain_realised_mean"] == pytest.approx(a, abs=1e-5)
    assert g["gain_realised_dev"] == pytest.approx(b, abs=1e-5)
    # the whole-state reading agrees: the map is block-diagonal over slots
    assert g["gain_global_mean_dir"] == pytest.approx(a, abs=1e-5)
    assert g["gain_global_dev_dir"] == pytest.approx(b, abs=1e-5)
    # and nothing leaves the perturbed slot
    assert g["leak_mean_dir"] == pytest.approx(0.0, abs=1e-6)
    assert g["leak_dev_dir"] == pytest.approx(0.0, abs=1e-6)
    # every slot is probed and the spread across slots is zero for a linear map
    assert g["slots_probed"] == bsz * s
    assert g["se_mean_dir"] == pytest.approx(0.0, abs=1e-6)
    assert g["se_dev_dir"] == pytest.approx(0.0, abs=1e-6)


def test_uniform_map_reads_equal_gains():
    """a == b: the map is a uniform scaling and the two families must not separate."""
    m, c, s, bsz = 4, 8, 4, 2
    jvp, _ = _split_map(m, c, 0.7, 0.7, seed=1)
    state = _state(bsz, s, m, c, seed=11)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=8, seed=5)
    assert g["gain_mean_dir"] == pytest.approx(0.7, abs=1e-5)
    assert g["gain_dev_dir"] == pytest.approx(0.7, abs=1e-5)
    assert g["ratio_dev_over_mean"] == pytest.approx(1.0, abs=1e-5)
    assert abs(g["gain_mean_dir"] - g["gain_dev_dir"]) < 1e-5


def test_hyper_connection_carrier_is_handled():
    """A ``[B, S*M, n, C]`` carrier: the split is on the CELL axis, the carrier axis rides
    along, and the two gains are still read exactly."""
    m, c, s, bsz, n = 4, 6, 3, 2, 4
    a, b = 0.4, 1.1

    def jvp(v: torch.Tensor) -> torch.Tensor:
        z = v.reshape(v.shape[0], v.shape[1] // m, m, n, c)
        mu = z.mean(dim=2, keepdim=True)
        out = a * mu.expand_as(z) + b * (z - mu)
        return out.reshape(v.shape)

    g0 = torch.Generator().manual_seed(2)
    state = torch.randn(bsz, s * m, n, c, generator=g0)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=6, seed=1)
    assert g["gain_mean_dir"] == pytest.approx(a, abs=1e-5)
    assert g["gain_dev_dir"] == pytest.approx(b, abs=1e-5)
    assert g["gain_realised_mean"] == pytest.approx(a, abs=1e-5)
    assert g["gain_realised_dev"] == pytest.approx(b, abs=1e-5)


def test_mask_restricts_the_operator():
    """A map that amplifies 10x on slot 1 and 0.5x everywhere else: with slot 1 inactive
    the reading must be 0.5, not something between. This is the pad-slot guard —
    ``core_jacobian``'s Gate 3 in the fan probe's own units."""
    m, c, s, bsz = 4, 5, 3, 1
    scale = torch.tensor([0.5, 10.0, 0.5]).view(1, s, 1, 1)

    def jvp(v: torch.Tensor) -> torch.Tensor:
        z = v.reshape(v.shape[0], s, m, c) * scale
        return z.reshape(v.shape)

    state = _state(bsz, s, m, c, seed=3)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    active[:, m:2 * m] = False
    g = pass_gains(jvp, state, active, m, draws=4, seed=0)
    assert g["slots_probed"] == 2
    assert g["gain_mean_dir"] == pytest.approx(0.5, abs=1e-5)
    assert g["gain_dev_dir"] == pytest.approx(0.5, abs=1e-5)
    assert g["gain_global_mean_dir"] == pytest.approx(0.5, abs=1e-5)


def test_cross_slot_leak_is_reported():
    """A map that sends slot s's perturbation into slot s+1 and keeps a 0.5x diagonal:
    the per-slot gain must still read 0.5 and the leak must be the off-block share."""
    m, c, s, bsz = 4, 4, 4, 1

    def jvp(v: torch.Tensor) -> torch.Tensor:
        z = v.reshape(bsz, s, m, c)
        out = 0.5 * z
        out[:, 1:] = out[:, 1:] + 1.5 * z[:, :-1]
        return out.reshape(v.shape)

    state = _state(bsz, s, m, c, seed=5)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=4, seed=2)
    assert g["gain_mean_dir"] == pytest.approx(0.5, abs=1e-5)
    assert g["gain_dev_dir"] == pytest.approx(0.5, abs=1e-5)
    # slots 0..2 leak 1.5 out against 0.5 kept -> 1.5/sqrt(0.5^2+1.5^2); slot 3 leaks 0
    one = 1.5 / (0.5 ** 2 + 1.5 ** 2) ** 0.5
    assert g["leak_dev_dir"] == pytest.approx(3.0 * one / 4.0, abs=1e-5)


def test_slots_per_row_caps_and_spreads():
    m, c, s, bsz = 4, 4, 10, 2
    jvp, _ = _split_map(m, c, 0.6, 0.9, seed=4)
    state = _state(bsz, s, m, c, seed=9)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=2, seed=0, slots_per_row=3)
    assert g["slots_probed"] == bsz * 3
    assert g["gain_mean_dir"] == pytest.approx(0.6, abs=1e-5)
    assert g["gain_dev_dir"] == pytest.approx(0.9, abs=1e-5)


def test_mixed_slot_mask_raises():
    m, c, s, bsz = 4, 4, 2, 1
    jvp, _ = _split_map(m, c, 0.5, 0.5)
    state = _state(bsz, s, m, c)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    active[0, 1] = False                       # one cell of slot 0 only
    with pytest.raises(ValueError, match="not constant within a slot"):
        pass_gains(jvp, state, active, m, draws=1)


def test_uniform_subspace_has_no_within_family_spread():
    """``a P + b Q R`` scales each subspace uniformly, so every draw inside a family
    reads the same gain: the coefficient of variation is 0 and ``d_eff`` is undefined."""
    m, c, s, bsz = 4, 8, 4, 2
    jvp, _ = _split_map(m, c, 0.5, 0.9, seed=6)
    state = _state(bsz, s, m, c, seed=13)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=8, seed=1)
    assert g["cv_mean_dir"] == pytest.approx(0.0, abs=1e-6)
    assert g["cv_dev_dir"] == pytest.approx(0.0, abs=1e-6)
    # a uniform scaling spreads its gain over EVERY direction of each subspace
    assert g["dim_mean_dir"] == c
    assert g["dim_dev_dir"] == (m - 1) * c
    assert g["d_eff_mean_dir"] == pytest.approx(float(c), rel=1e-6)
    assert g["d_eff_dev_dir"] == pytest.approx(float((m - 1) * c), rel=1e-6)


def test_anisotropic_deviation_subspace_is_seen():
    """A map that scales HALF the channels of the deviation by 1.0 and the other half by
    0 leaves the RMS deviation gain at ``1/sqrt(2)`` and a nonzero spread across draws.

    With ``d`` the deviation subspace dimension and half its singular values 1 and half
    0, ``d_eff = (sum s^2)^2 / sum s^4 = d/2`` exactly, which is what the estimator must
    land near. The mean family is untouched and must still read ``a`` with zero spread.
    """
    m, c, s, bsz = 4, 64, 3, 2
    a = 0.5
    keep = torch.zeros(c)
    keep[: c // 2] = 1.0

    def jvp(v: torch.Tensor) -> torch.Tensor:
        z = v.reshape(v.shape[0], v.shape[1] // m, m, c)
        mu = z.mean(dim=2, keepdim=True)
        out = a * mu.expand_as(z) + (z - mu) * keep
        return out.reshape(v.shape)

    state = _state(bsz, s, m, c, seed=17)
    active = torch.ones(bsz, s * m, dtype=torch.bool)
    g = pass_gains(jvp, state, active, m, draws=64, seed=4)
    assert g["gain_mean_dir"] == pytest.approx(a, abs=1e-5)
    assert g["cv_mean_dir"] == pytest.approx(0.0, abs=1e-6)
    # RMS over an isotropic draw in the (m-1)*c deviation subspace: half the singular
    # values are 1 -> E||Jv||^2 = 1/2, so the mean gain sits just under 1/sqrt(2).
    assert g["gain_dev_dir"] == pytest.approx(0.5 ** 0.5, rel=0.02)
    d = (m - 1) * c
    assert g["d_eff_dev_dir"] == pytest.approx(d / 2.0, rel=0.25)
    assert g["cv_dev_dir"] > 0.01


def test_d_eff_formula():
    """Invert the exact relation both ways on known spectra."""
    for d in (16, 192, 4096):
        # uniform subspace: no spread, every direction counts
        assert _d_eff(0.0, d) == pytest.approx(float(d), rel=1e-12)
        # half the singular values 1, half 0 -> d_eff = d/2, cv(X) = sqrt(2/(d+2))
        cv = 0.5 * (2.0 / (d + 2)) ** 0.5
        assert _d_eff(cv, d) == pytest.approx(d / 2.0, rel=1e-9)
        # one live direction -> d_eff = 1
        cv1 = 0.5 * ((2.0 * (d - 1)) / ((d + 2) * 1.0)) ** 0.5
        assert _d_eff(cv1, d) == pytest.approx(1.0, rel=1e-9)
    assert _d_eff(float("nan"), 8) != _d_eff(float("nan"), 8)     # nan in, nan out
