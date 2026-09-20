"""``tul.fan_repel_mode: "epi"`` — the epiplexity diversity term (morph/model/tul_fan.py).

What the term must do that the cosine could not: score a constant two-against-two split
at ZERO, score input-dependent spread over more directions HIGHER, and be blind to the
streams' scale. Plus the contracts every fan knob keeps: ``"cos"`` is the default and
builds nothing new, the reservoir is frozen and RNG-neutral, the passes charged are
``1 .. fan_repel_passes`` and every pass is reported.
"""
import math

import pytest
import torch

from morph.model.tul import TULConfig
from morph.model.tul_fan import FanReservoir, epi_score, fan_epi_term, ridge_map
from morph.training.tul_setup import reject_unknown_tul_keys
from test_tul_fan import _batch, _model, _state


def _reservoir_map(n: int, c: int, f: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, c, generator=g)
    res = FanReservoir(c, f, seed=1, hidden=32)
    return x, ridge_map(res(x), 3.0)


def test_score_is_zero_for_a_constant_offset_and_for_identical_streams():
    n, c = 64, 24
    _x, a = _reservoir_map(n, c)
    const = torch.ones(n, c) * 3.0                          # the same deviation in every slot
    assert epi_score(const, a, 30.0).item() == pytest.approx(0.0, abs=1e-9)
    assert epi_score(torch.zeros(n, c), a, 30.0).item() == pytest.approx(0.0, abs=1e-9)


def test_score_rises_with_the_rank_of_input_dependent_variation():
    n, c = 128, 24
    x, a = _reservoir_map(n, c)
    # Both deviations are linear functions of the reservoir's input (so predictable), with
    # the same Frobenius norm; one lives on one direction, the other on eight.
    g = torch.Generator().manual_seed(3)
    u = torch.nn.functional.normalize(torch.randn(c, generator=g), dim=0)
    coef = x @ torch.randn(c, 8, generator=g)                # [n, 8] input-dependent
    rank1 = coef[:, :1] * u.unsqueeze(0)                     # [n, c] on one line
    basis = torch.linalg.qr(torch.randn(c, 8, generator=g))[0]   # [c, 8]
    rank8 = coef @ basis.T
    rank8 = rank8 * (rank1.norm() / rank8.norm())
    s1, s8 = epi_score(rank1, a, 30.0).item(), epi_score(rank8, a, 30.0).item()
    assert s8 > s1 > 0.0


def test_sylvester_side_matches_the_full_log_det():
    n, c, f = 40, 12, 6
    _x, a = _reservoir_map(n, c, f=f)
    g = torch.Generator().manual_seed(5)
    z = torch.randn(n, c, generator=g)
    w = a @ (z.double() - z.double().mean(0))                # [f, c]
    full = 0.5 * torch.linalg.slogdet(torch.eye(c, dtype=w.dtype) + 30.0 * w.T @ w)[1] / math.log(2)
    assert epi_score(z, a, 30.0).item() == pytest.approx(full.item(), rel=1e-9)


def _traj(seed: int, s: int = 6, m: int = 4, c: int = 24, scale: float = 1.0,
          spread: float = 0.3, n_entries: int = 4):
    """A seed entry and three passes: streams = a per-slot base plus per-stream deviations
    that DEPEND on the base (so the reservoir can read them)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(s, c, generator=g)
    mix = torch.randn(c, m * c, generator=g) / math.sqrt(c)
    dev = (base @ mix).reshape(s, m, c) * spread
    out = []
    for t in range(n_entries):
        st = base.unsqueeze(1) + (0.0 if t == 0 else 1.0) * dev * (1.0 + 0.1 * t)
        out.append((scale * _state(st)).clone().requires_grad_(True))
    return out


def test_term_charges_the_first_passes_only_and_reports_every_pass():
    traj = _traj(seed=0)
    valid = torch.ones(1, 6, dtype=torch.bool)
    res = FanReservoir(24, 16, seed=1, hidden=32)
    stats: dict = {}
    term = fan_epi_term(traj, valid, 4, n_passes=2, reservoir=res, ridge=3.0, eta=30.0,
                        stats=stats)
    assert sorted(stats) == ["epi_t0", "epi_t1", "epi_t2", "epi_t3", "repel_terms"]
    assert stats["repel_terms"] == 2.0
    assert stats["epi_t0"] == pytest.approx(0.0, abs=1e-9)   # identical streams at the seed
    assert stats["epi_t1"] > 0.0 and stats["epi_t2"] > 0.0
    assert term.item() == pytest.approx(-0.5 * (stats["epi_t1"] + stats["epi_t2"]), rel=1e-5)
    gs = torch.autograd.grad(term, traj, allow_unused=True)
    assert gs[0] is None, "the SEED state must carry no gradient"
    assert gs[1] is not None and float(gs[1].abs().sum()) > 0.0
    assert gs[2] is not None and float(gs[2].abs().sum()) > 0.0
    assert gs[3] is None, "pass 3 is past fan_repel_passes=2 and must carry no gradient"


def test_term_is_scale_free():
    res = FanReservoir(24, 16, seed=1, hidden=32)
    valid = torch.ones(1, 6, dtype=torch.bool)
    a = fan_epi_term(_traj(seed=0, scale=1.0), valid, 4, 2, res, 3.0, 30.0)
    b = fan_epi_term(_traj(seed=0, scale=25.0), valid, 4, 2, res, 3.0, 30.0)
    assert a.item() == pytest.approx(b.item(), rel=1e-5)


def test_term_ignores_pad_slots_and_survives_a_tiny_batch():
    res = FanReservoir(24, 16, seed=1, hidden=32)
    traj = _traj(seed=0)
    valid = torch.tensor([[True, True, True, True, False, False]])
    stats: dict = {}
    term = fan_epi_term(traj, valid, 4, 2, res, 3.0, 30.0, stats=stats)
    assert torch.isfinite(term)
    one = torch.tensor([[True, False, False, False, False, False]])
    term1 = fan_epi_term(traj, one, 4, 2, res, 3.0, 30.0)
    assert term1.item() == 0.0                                # < 2 slots: no ridge, exact 0


def test_reservoir_is_frozen_rng_neutral_and_excluded():
    torch.manual_seed(11)
    before = torch.random.get_rng_state()
    res = FanReservoir(32, 8, seed=4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert list(res.parameters()) == []
    assert sorted(k for k, _ in res.named_buffers()) == ["w1", "w2"]
    assert res._ternary_exclude is True
    x = torch.randn(5, 32, requires_grad=True)
    assert res(x).requires_grad is False
    same = FanReservoir(32, 8, seed=4)
    assert torch.equal(res.w1, same.w1) and torch.equal(res.w2, same.w2)


def test_cos_is_the_default_and_builds_no_reservoir():
    m = _model(fan_k=4, slot_cells=4, fan_mix="mean", fan_repel_lambda=0.5)
    assert m.cfg.tul.fan_repel_mode == "cos"
    assert m.tul_fan_epi is None
    assert not any(k.startswith("tul_fan_epi") for k in m.state_dict())


def test_epi_mode_adds_buffers_only_and_reaches_the_loss():
    cos = _model(fan_k=4, slot_cells=4, fan_mix="mean", fan_repel_lambda=0.5)
    epi = _model(fan_k=4, slot_cells=4, fan_mix="mean", fan_repel_lambda=0.5,
                 fan_repel_mode="epi", fan_epi_features=8)
    assert [n for n, _ in cos.named_parameters()] == [n for n, _ in epi.named_parameters()]
    # same seed, same draw: the base weights are byte-identical (the reservoir is private)
    for (n, p), (_, q) in zip(cos.named_parameters(), epi.named_parameters()):
        assert torch.equal(p, q), n
    assert sorted(k for k in epi.state_dict() if k.startswith("tul_fan_epi")) == [
        "tul_fan_epi.w1", "tul_fan_epi.w2"]
    _ids0, inp, lab, layout = _batch(4)
    epi.train()
    torch.manual_seed(7)
    out = epi(inp, labels=lab, slot_layout=layout)
    assert "fan_repel" in out and "fan_repel_weighted" in out
    assert torch.isfinite(out["fan_repel"])
    assert float(out["fan_repel_weighted"]) == pytest.approx(0.5 * float(out["fan_repel"]), rel=1e-6)
    assert any(str(k).startswith("fan_epi_t") for k in out)
    assert any(str(k).startswith("fan_stream_cos_t") for k in out), "the cosines stay instruments"
    out["loss"].backward()
    assert epi.tul_register.W_o.weight.grad is not None


def test_config_refusals_and_known_keys():
    with pytest.raises(ValueError, match="fan_repel_mode"):
        TULConfig(prefix_k=4, fan_k=4, slot_cells=4, fan_repel_mode="cosine")
    with pytest.raises(ValueError, match="fan_epi_features"):
        TULConfig(prefix_k=4, fan_k=4, slot_cells=4, fan_repel_mode="epi", fan_epi_features=1)
    with pytest.raises(ValueError, match="fan_epi_ridge"):
        TULConfig(prefix_k=4, fan_k=4, slot_cells=4, fan_repel_mode="epi", fan_epi_eta=0.0)
    with pytest.raises(ValueError, match="fan_k=0"):
        TULConfig(prefix_k=2, fan_repel_mode="epi")
    reject_unknown_tul_keys({"fan_k": 4, "fan_repel_mode": "epi", "fan_epi_features": 64,
                             "fan_epi_ridge": 3.0, "fan_epi_eta": 30.0})
