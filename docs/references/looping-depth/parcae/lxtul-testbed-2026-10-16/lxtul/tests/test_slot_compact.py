"""CPU tests of `slot_depth_stratified` and `slot_compact` (lxtul/slot_compact.py, lxtul/model.py).

What is pinned:
  1. the stratified draw: each slot's marginal is clamp(Poisson), the active slots of pass t never exceed
     the capacity C_t (pads included), and the bound is attained;
  2. "gather" (C_t * M cells per pass) equals "full" (every cell, same caches) in loss and EVERY
     parameter gradient, in float64, with the gain hinge and the cot clip engaged, with and without
     checkpointed passes, and beside static_depth, cot_clip_fn, ce_rows and spandec_packed;
  3. "full" is a DIFFERENT function from off (frozen cells recomputed): it differs when an earlier slot is
     still active after a later one finished, and is identical when no cell finishes early;
  4. a sabotaged cache read is caught by (2); the capacity assert and the build refusals fire;
  5. eval is unchanged.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_slot_compact.py
"""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import lxtul.attention as A
import lxtul.model as lm
from lxtul.slot_compact import (clamped_poisson_cdf, compact_order, put_rows, slot_depth_capacity,
                                stratified_slot_depths, take_rows)
from lxtul.tests.test_compact import (J, K, S, V, _assert_same, _batch, _build, _run, _twin,  # noqa: F401
                                      cpu_reference_kernels)

ST = dict(slot_depth_stratified=True)


@pytest.fixture(scope="module", params=["default", "hot"])
def base(request):
    """Stratified draw on, compaction off. `hot`: the gain hinge and the cot clip are engaged."""
    if request.param == "hot":
        return _build(gain_target=0.05, cot_clip=0.2, **ST)
    return _build(**ST)


@pytest.fixture(scope="module")
def data():
    x, y, lay = _batch(1)
    nv = lay.slot_valid.sum(1)
    assert nv.min() < S and nv.min() != nv.max()
    return x, y, lay


# ------------------------------------------------------------------ 1. the draw
def test_capacity_at_the_lxtul_shape():
    assert slot_depth_capacity(6.0, 8, 64) == (64, 63, 61, 55, 46, 36, 26, 17)


@pytest.mark.parametrize("mean,max_d,n", [(6.0, 8, 64), (6.0, 8, 10), (3.0, 5, 7), (2.0, 4, 50), (6.0, 8, 16)])
def test_capacity_bound_holds_and_is_attained(mean, max_d, n):
    torch.manual_seed(0)
    rows = 20000
    d = stratified_slot_depths(rows, n, mean, max_d, "cpu")
    assert d.min() >= 1 and d.max() <= max_d
    caps = slot_depth_capacity(mean, max_d, n)
    n_valid = torch.randint(0, n + 1, (rows, 1))
    dp = torch.where(torch.arange(n) < n_valid, d, torch.ones_like(d))        # pads loop once
    for t in range(max_d):
        assert int((d > t).sum(1).max()) == caps[t], (t, "the bound must be attained")
        assert int((dp > t).sum(1).max()) <= caps[t], t


def test_marginal_is_the_clamped_poisson_law():
    torch.manual_seed(2)
    rows, n, mean, max_d = 30000, 16, 6.0, 8
    d = stratified_slot_depths(rows, n, mean, max_d, "cpu")
    cdf = clamped_poisson_cdf(mean, max_d)
    pmf = [cdf[t] - cdf[t - 1] for t in range(1, max_d + 1)]
    for slot in (0, 5, 15):                                  # any slot: the permutation is uniform
        freq = torch.bincount(d[:, slot], minlength=max_d + 1)[1:].double() / rows
        for t in range(max_d):
            se = math.sqrt(pmf[t] * (1 - pmf[t]) / rows)
            assert abs(float(freq[t]) - pmf[t]) < 5 * se + 1e-9, (slot, t + 1, float(freq[t]), pmf[t])
    pooled = torch.bincount(d.flatten(), minlength=max_d + 1)[1:].double() / d.numel()
    assert torch.allclose(pooled, torch.tensor(pmf, dtype=torch.float64), atol=3e-3)
    # the joint law differs from independent draws: the spread of a row's mean depth is far smaller
    ind = torch.poisson(torch.full((rows, n), mean)).clamp(1, max_d).double()
    assert d.double().mean(1).std() < 0.5 * ind.mean(1).std()


def test_row_moves_are_a_permutation_and_roundtrip():
    g = torch.Generator().manual_seed(3)
    act = torch.rand(4, 20, generator=g) < 0.4
    perm = compact_order(act)
    for b in range(4):
        n = int(act[b].sum())
        assert perm[b, :n].tolist() == act[b].nonzero().squeeze(1).tolist()
        assert sorted(perm[b].tolist()) == list(range(20))
    n = int(act.sum(1).max())
    idx = perm[:, :n]
    x = torch.randn(4, 20, 3, 5, dtype=torch.float64, requires_grad=True)
    fill = torch.randn(4, 20, 3, 5, dtype=torch.float64, requires_grad=True)
    out = put_rows(take_rows(x, idx) * 2, fill, idx, act)
    want = torch.where(act[..., None, None], x * 2, fill)
    assert torch.equal(out, want)
    w = torch.randn_like(out)
    (out * w).sum().backward()
    assert torch.equal(x.grad, 2 * w * act[..., None, None])
    assert torch.equal(fill.grad, w * (~act)[..., None, None])


# ------------------------------------------------------------------ 2. gather == full
def test_gather_equals_full(base, data):
    x, y, lay = data
    full = _twin(base, slot_compact="full")
    a, ga = _run(full, x, y, lay)
    gat = _twin(base, slot_compact="gather")
    b, gb = _run(gat, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_gather_equals_full_with_checkpointed_passes(base, data):
    x, y, lay = data
    a, ga = _run(_twin(base, slot_compact="full", ckpt_passes=0), x, y, lay)
    b, gb = _run(_twin(base, slot_compact="gather", ckpt_passes=4), x, y, lay)
    _assert_same(a, ga, b, gb)


def test_gather_beside_the_other_keys(base, data):
    x, y, lay = data
    a, ga = _run(_twin(base, slot_compact="full"), x, y, lay)
    on = _twin(base, slot_compact="gather", static_depth=True, cot_clip_fn=True, ce_rows=True,
               spandec_packed=True)
    b, gb = _run(on, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_the_compacted_pass_computes_fewer_cells(base, data):
    x, y, lay = data
    sizes = []
    real = A.compact_layer

    def spy(block, xx, *a):
        sizes.append(xx.shape[1])
        return real(block, xx, *a)
    gat = _twin(base, slot_compact="gather", gain_lambda=0.0)      # no hinge: one call per layer per pass
    gat._compact_layer = spy
    with torch.no_grad():
        gat(x, y, lay)
    per_layer = len(gat.transformer.core_block)
    n = sizes[::per_layer]
    caps = slot_depth_capacity(6.0, 8, S)
    assert n[0] == min(caps[0] * K, S * K)
    assert all(n[t] == min(caps[t] * K, S * K) for t in range(len(n))), n
    assert n[-1] < n[0], "the last pass must be smaller than the first"


# ------------------------------------------------------------------ 3. off vs full
def _craft(depths):
    d = torch.tensor(depths, dtype=torch.long)

    def draw(B, Sn, mean, max_d, device):
        return d.expand(B, Sn).clone()
    return draw


def test_full_differs_from_off_when_an_earlier_slot_stays_active(base, data, monkeypatch):
    x, y, lay = data
    # slot 0 runs 8 passes, slot 3 stops after 2: slot 3 is finished while slot 0 (earlier) still moves
    pat = [8, 6, 5, 2, 7, 3, 8, 1, 4, 6, 2, 8, 5, 3, 7, 4]
    monkeypatch.setattr(lm, "stratified_slot_depths", _craft(pat))
    a, ga = _run(base, x, y, lay)
    b, gb = _run(_twin(base, slot_compact="full"), x, y, lay)
    assert abs(float(a["loss"]) - float(b["loss"])) > 1e-6, "off and full must be different functions"


def test_full_equals_off_when_nothing_finishes_early(base, data, monkeypatch):
    x, y, lay = data
    monkeypatch.setattr(lm, "stratified_slot_depths", _craft([8] * S))
    a, ga = _run(base, x, y, lay)
    b, gb = _run(_twin(base, slot_compact="full"), x, y, lay)
    _assert_same(a, ga, b, gb)


# ------------------------------------------------------------------ 4. controls and refusals
def test_a_sabotaged_cache_read_is_caught(base, data, monkeypatch):
    x, y, lay = data
    a, ga = _run(_twin(base, slot_compact="full"), x, y, lay)
    gat = _twin(base, slot_compact="gather")
    real = A.compact_layer

    def stale(block, xx, freqs, mask, ve, kc, vc, idx, act):
        if kc is not None:
            kc, vc = kc * 0.9, vc * 0.9                 # a frozen cell reads a wrong key and value
        return real(block, xx, freqs, mask, ve, kc, vc, idx, act)
    gat._compact_layer = stale
    b, gb = _run(gat, x, y, lay)
    with pytest.raises(AssertionError):
        _assert_same(a, ga, b, gb)


def test_capacity_assert_fires_when_the_draw_breaks_the_bound(base, data, monkeypatch):
    x, y, lay = data
    monkeypatch.setattr(lm, "stratified_slot_depths", _craft([8] * S))      # every slot active at every pass
    with pytest.raises(RuntimeError):
        _run(_twin(base, slot_compact="gather"), x, y, lay)


def test_refusals(base):
    with pytest.raises(ValueError, match="stratified"):
        _twin(_build(), slot_compact="gather")
    with pytest.raises(ValueError, match="not combined"):
        _twin(base, slot_compact="gather", compact_slots=True)
    with pytest.raises(ValueError, match="one of"):
        _twin(base, slot_compact="bogus")


def test_eval_is_unchanged(base, data):
    x, y, lay = data
    off, on = base.eval(), _twin(base, slot_compact="gather").eval()
    with torch.no_grad():
        for depth in (None, 3):
            h0 = off(x, None, lay, depth=depth)["hidden"]
            h1 = on(x, None, lay, depth=depth)["hidden"]
            assert torch.equal(h0, h1)
    base.train()


@pytest.mark.fp32
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
def test_gather_equals_full_in_fp32_on_the_real_kernels(device):
    """Rounding bound at training precision; on CUDA this runs the real FlexAttention with Q_LEN != KV_LEN."""
    import parcae_lm
    from lxtul.model import LXTULConfig, LXTULParcae
    from lxtul.tests.test_compact import _randomize, _to
    x, y, lay = _to(_batch(1), device)
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=128 + K * S)
    tul = LXTULConfig(slot_id=4, max_slots=S, cells=K, span_cap=J, spandec_layers=1, gain_target=0.05,
                      cot_clip=0.2, slot_depth_stratified=True, slot_compact="full")
    m = _randomize(LXTULParcae(cfg, tul)).to(device).train()
    a, ga = _run(m, x, y, lay)
    on = LXTULParcae(cfg, dataclasses.replace(tul, slot_compact="gather")).to(device).train()
    on.load_state_dict(m.state_dict())
    b, gb = _run(on, x, y, lay)
    assert abs(float(a["loss"]) - float(b["loss"])) <= 2e-5 * abs(float(a["loss"]))
    for n in ga:
        assert float((ga[n] - gb[n]).abs().max()) / max(float(ga[n].abs().max()), 1e-12) <= 2e-3, n
