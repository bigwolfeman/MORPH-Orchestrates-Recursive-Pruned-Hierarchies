"""`static_index` and `mask_cache` against the original code paths, on CPU.

`static_index` is rounding-only: the same maths with weights in place of boolean indexing and
an F x F Cholesky ridge solve in place of the float64 QR. These tests compare value AND gradient
to the original on random inputs with random `valid` patterns, and check that invalid rows are
inert (changing them changes nothing).

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_static_index.py
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

import lxtul.data  # noqa: F401  (puts MORPH_ROOT on sys.path)
import parcae_lm
from lxtul.model import LXTULConfig, LXTULParcae

V, SLOT, SEQ, K, S, D = 512, 4, 128, 4, 16, 64


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=D, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=D,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    return LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=12,
                                        spandec_layers=1, epi_features=16, epi_hidden=32))


def _traj(B, valid_p, seed, passes=3):
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(B, S, generator=g) < valid_p
    traj = [torch.randn(B, S * K, D, generator=g).requires_grad_() for _ in range(passes)]
    return traj, valid


def _static(m, on):
    return dataclasses.replace(m.tul, static_index=on)


@pytest.mark.parametrize("valid_p", [1.0, 0.7, 0.2])
def test_epivol_matches_original(model, valid_p):
    traj, valid = _traj(3, valid_p, 1)
    out = {}
    for on in (False, True):
        model.tul = _static(model, on)
        for t in traj:
            t.grad = None
        loss = model._epivol(traj, valid, K)
        loss.backward()
        out[on] = (loss.detach().clone(), [None if t.grad is None else t.grad.clone() for t in traj], dict(model.metrics))
    (l0, g0, m0), (l1, g1, m1) = out[False], out[True]
    assert l0.abs() > 1e-3, "the loss must be non-trivial for the comparison to mean anything"
    torch.testing.assert_close(l1, l0, rtol=1e-4, atol=1e-5)
    assert g0[0] is None and g1[0] is None, "pass 0 is read under no_grad"
    for a, b in zip(g1[1:], g0[1:]):
        assert b.abs().max() > 0
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-6)
    for k in ("epi", "vol"):
        torch.testing.assert_close(m1[k], m0[k], rtol=1e-4, atol=1e-5)


def test_epivol_ignores_invalid_rows(model):
    model.tul = _static(model, True)
    traj, valid = _traj(3, 0.5, 2)
    base = model._epivol(traj, valid, K).detach()
    bad = [t.detach().clone() for t in traj]
    inv = ~valid.repeat_interleave(K, dim=1)
    for t in bad:
        t[inv] = torch.randn_like(t[inv]) * 50
    torch.testing.assert_close(model._epivol(bad, valid, K).detach(), base, rtol=0, atol=0)


@pytest.mark.parametrize("n_ok", [0, 1, 2, 9])
def test_lsel_enc_floor_matches_original(model, n_ok):
    g = torch.Generator().manual_seed(5)
    B = 3
    ok = torch.zeros(B, S, dtype=torch.bool)
    ok.view(-1)[torch.randperm(B * S, generator=g)[:n_ok]] = True
    hv = torch.randn(B, S, K, D, generator=g)
    zo0 = 0.02 * torch.randn(B, S, D, generator=g)      # std 0.02 < gamma 0.1: the floor is active
    z = torch.randn(B, S, D, generator=g)
    res = {}
    for on in (False, True):
        model.tul = _static(model, on)
        zo = zo0.detach().clone().requires_grad_()
        _, l_enc, _ = model._lsel_exit(hv, z, ok, zo, torch.zeros(()), torch.zeros(()))
        if l_enc.requires_grad:
            l_enc.backward()
        res[on] = (l_enc.detach(), torch.zeros_like(zo) if zo.grad is None else zo.grad)
    torch.testing.assert_close(res[True][0], res[False][0], rtol=1e-4, atol=1e-6)
    assert torch.isfinite(res[True][1]).all()
    torch.testing.assert_close(res[True][1], res[False][1], rtol=1e-3, atol=1e-7)
    if n_ok > 1:
        assert res[False][0] > 0, "floor must be active for the comparison to mean anything"


def test_mask_cache_reuses_only_for_the_same_unmodified_layout(model, monkeypatch):
    calls = []
    monkeypatch.setattr("lxtul.model.prelude_mask", lambda bag, sl, op: calls.append("p") or object())
    monkeypatch.setattr("lxtul.model.coda_mask", lambda bag, sl, op: calls.append("c") or object())
    bag, sl = torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.bool)
    model._mask_memo = None
    model.tul = dataclasses.replace(model.tul, mask_cache=False)
    a, b = model._masks(bag, sl, False, K), model._masks(bag, sl, False, K)
    assert len(calls) == 4 and a[0] is not b[0], "key off: a build per call"
    calls.clear()
    model.tul = dataclasses.replace(model.tul, mask_cache=True)
    a = model._masks(bag, sl, False, K)
    assert model._masks(bag, sl, False, K) is a and calls == ["p", "c"], "same tensors: one build"
    bag[0, 0] = 1                                   # in-place edit bumps the version
    c = model._masks(bag, sl, False, K)
    assert c is not a and len(calls) == 4, "edited layout: rebuilt"
    assert model._masks(bag.clone(), sl, False, K) is not c and len(calls) == 6, "new tensor: rebuilt"
    model._masks(bag, sl, False, K)
    n = len(calls)
    model._masks(bag, sl, True, K)
    assert len(calls) == n + 2, "geometry flag change: rebuilt"


def test_mask_cache_is_keyed_by_the_cell_count(model, monkeypatch):
    """The same layout tensors under another M must rebuild (a varying cell count cannot get a stale pair)."""
    calls = []
    monkeypatch.setattr("lxtul.model.prelude_mask", lambda bag, sl, op: calls.append("p") or object())
    monkeypatch.setattr("lxtul.model.coda_mask", lambda bag, sl, op: calls.append("c") or object())
    bag, sl = torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.bool)
    model._mask_memo = None
    model.tul = dataclasses.replace(model.tul, mask_cache=True)
    a = model._masks(bag, sl, False, 2)
    assert model._masks(bag, sl, False, 2) is a and len(calls) == 2
    assert model._masks(bag, sl, False, 3) is not a and len(calls) == 4, "other M: rebuilt"
