"""Host shadows (`morph/model/host_shadow.py`, 2026-10-04 speed pass): sync-free readers of
masks whose value the host already knows must return EXACTLY what the synchronising code
returned, forward and backward, and must fall back to it when no shadow exists.

Pinned here:
  * `true_index` == `nonzero()`; `masked_rows` == `x[mask]` (values and gradients);
    `any_true` / `count_true`; an in-place write voids a shadow; derived tensors carry none.
  * `SlotLayout.to` records the host value of `slot_mask` / `slot_valid`.
  * The fan's terms (`fan_stream_cos`, the epi and vol passes, `TULFanMix.entropy`) and the
    compressed branch's slot gather (`_tg_slot_attention`) give the same tensors with a
    shadowed mask as with a bare one.
  * The lead arm's full TRAIN forward + backward (latent-selected loop, joint, router-
    followed, rank-only head, cell norm, fixed-point term, gain hinge) is bit-identical with
    a shadowed layout and with a bare one: loss, every output, every gradient.
  * `_train_instruments=False` (the trainer's non-logging steps) leaves the loss and every
    gradient bit-identical and drops only the instrument readings, whose values on a
    logging step are those of the old code.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model import host_shadow as hs
from morph.model.attention import _tg_slot_attention
from morph.model.tul_fan import (TULFanMix, _fan_epi_pass, _fan_vol_pass, fan_epi_term,
                                 fan_repel_term, fan_stream_cos, fan_vol_term,
                                 FanReservoir, ridge_map)
from morph.model.tul_layout import SlotLayout
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M


def _mask(shape, p=0.6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) < p


def _shadowed(m: torch.Tensor) -> torch.Tensor:
    t = m.clone()
    return hs.attach(t, t.numpy().copy())


def test_true_index_is_nonzero():
    for seed in range(5):
        m = _mask((3, 17), seed=seed)
        ref = m.flatten().nonzero().squeeze(1)
        assert torch.equal(hs.true_index(m), ref)                    # bare: the old op
        assert torch.equal(hs.true_index(_shadowed(m)), ref)         # shadowed: sync-free
    empty = torch.zeros(4, 5, dtype=torch.bool)
    assert hs.true_index(_shadowed(empty)).numel() == 0


def test_masked_rows_equals_boolean_index_forward_and_backward():
    m = _mask((3, 9, 4), seed=3)
    x = torch.randn(3, 9, 4, 2, 5, dtype=torch.float32)
    w = torch.randn(int(m.sum()), 2, 5)
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ya = xa[m]
    yb = hs.masked_rows(xb, _shadowed(m))
    assert torch.equal(ya, yb)
    (ya * w).sum().backward()
    (yb * w).sum().backward()
    assert torch.equal(xa.grad, xb.grad)


def test_counts_and_invalidation():
    m = _mask((5, 6), seed=1)
    s = _shadowed(m)
    assert hs.any_true(s) == bool(m.any()) and hs.count_true(s) == int(m.sum())
    assert hs.shadow(s) is not None
    s[0, 0] = ~s[0, 0]                     # in-place write: the shadow no longer applies
    assert hs.shadow(s) is None
    assert hs.count_true(s) == int(s.sum())
    assert hs.shadow(_shadowed(m) & m) is None          # derived tensors carry nothing
    with pytest.raises(ValueError):
        hs.attach(m.clone(), np.zeros((2, 2), dtype=bool))


def test_layout_to_records_the_host_value():
    _ids, _inp, _lab, layout = _batch(M)
    d = layout.to("cpu")
    assert np.array_equal(hs.shadow(d.slot_valid), layout.slot_valid.numpy())
    assert np.array_equal(hs.shadow(d.slot_mask), layout.slot_mask.numpy())


def _bare(layout: SlotLayout) -> SlotLayout:
    """The same layout with fresh tensors (no shadow anywhere)."""
    c = lambda t: None if t is None else t.clone()  # noqa: E731
    return SlotLayout(slot_mask=c(layout.slot_mask), bag_id=c(layout.bag_id),
                      slot_index=c(layout.slot_index), slot_valid=c(layout.slot_valid),
                      prefix_k=layout.prefix_k, stats=layout.stats,
                      span_len=c(layout.span_len), len_supervised=c(layout.len_supervised))


def test_fan_terms_agree_with_and_without_a_shadow():
    torch.manual_seed(0)
    B, S, m, n, C = 2, 6, 4, 2, 16
    valid = _mask((B, S), p=0.7, seed=2)
    valid[0, 0] = True
    traj = [torch.randn(B, S * m, n, C) for _ in range(4)]
    res = FanReservoir(C, 8, seed=3)
    outs = []
    for v in (valid.clone(), _shadowed(valid)):
        st: dict = {}
        cos = fan_stream_cos(traj[1], v, m)
        epi = fan_epi_term(traj, v, m, 2, res, 3.0, 30.0, stats=st)
        vol = fan_vol_term(traj, v, m, 2, 30.0, stats=st)
        rep = fan_repel_term(traj, v, m, 2, stats=st)
        a = ridge_map(res(traj[0].reshape(B, S, m, n, C).mean(2).mean(2)[valid]), 3.0)
        ep = _fan_epi_pass(traj[2], v, m, a, 30.0)
        vp = _fan_vol_pass(traj[2], v, m, 30.0)
        wts = torch.softmax(torch.randn(B, S, m, generator=torch.Generator().manual_seed(4)), -1)
        ent = TULFanMix.entropy(wts, v)
        outs.append((cos, epi, vol, rep, ep, vp, ent, st))
    for x, y in zip(outs[0][:7], outs[1][:7]):
        assert torch.equal(x, y)
    assert outs[0][7].keys() == outs[1][7].keys()
    for k in outs[0][7]:
        assert torch.equal(torch.as_tensor(outs[0][7][k]), torch.as_tensor(outs[1][7][k])), k


def test_instruments_off_keeps_the_terms_and_drops_only_readings():
    torch.manual_seed(1)
    B, S, m, n, C = 2, 5, 4, 2, 16
    valid = torch.ones(B, S, dtype=torch.bool)
    traj = [torch.randn(B, S * m, n, C) for _ in range(5)]
    res = FanReservoir(C, 8, seed=3)
    for fn, args in ((fan_epi_term, (res, 3.0, 30.0)), (fan_vol_term, (30.0,)),
                     (fan_repel_term, ())):
        on: dict = {}
        off: dict = {}
        a = fn(traj, valid, m, 2, *args, stats=on)
        b = fn(traj, valid, m, 2, *args, stats=off, instruments=False)
        assert torch.equal(a, b), fn.__name__
        assert set(off) < set(on)
        for k in off:
            assert torch.equal(torch.as_tensor(off[k]), torch.as_tensor(on[k])), k


def test_tg_slot_attention_slot_count_from_the_shadow():
    torch.manual_seed(2)
    B, H, S, D = 2, 2, 12, 8
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    sm = _mask((B, S), p=0.4, seed=5)
    sink = torch.zeros(H)
    a = _tg_slot_attention(q, k, v, sm.clone(), sink, D ** -0.5)
    b = _tg_slot_attention(q, k, v, _shadowed(sm), sink, D ** -0.5)
    assert torch.equal(a, b)


_LEAD = dict(fan_loop_select="joint", fan_lsel_train_follow="router", fan_lsel_lambda=1.0,
             fan_lsel_head_input="detached", slot_cell_pass_norm="rms")


def _run(layout, instruments=True, dropout=0.1):
    _ids, inp, lab, _ = _batch(M)
    m = _lsel("joint", model_kw={"dropout": dropout},
              **{k: v for k, v in _LEAD.items() if k != "fan_loop_select"}).train()
    m._train_instruments = instruments
    torch.manual_seed(7)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out, grads


def test_lead_arm_forward_backward_is_bit_identical_with_shadows():
    _ids, _inp, _lab, layout = _batch(M)
    bare = _bare(layout)
    shad = _bare(layout).to("cpu")
    assert hs.shadow(shad.slot_valid) is not None and hs.shadow(bare.slot_valid) is None
    oa, ga = _run(bare)
    ob, gb = _run(shad)
    assert torch.equal(oa["loss"], ob["loss"])
    assert oa.keys() == ob.keys()
    for k in oa:
        if torch.is_tensor(oa[k]):
            assert torch.equal(oa[k], ob[k]), k
    assert ga.keys() == gb.keys() and len(ga) > 50
    for n in ga:
        assert torch.equal(ga[n], gb[n]), n


def test_lead_arm_instruments_off_keeps_loss_and_gradients():
    _ids, _inp, _lab, layout = _batch(M)
    oa, ga = _run(_bare(layout).to("cpu"), instruments=True)
    ob, gb = _run(_bare(layout).to("cpu"), instruments=False)
    assert torch.equal(oa["loss"], ob["loss"])
    for n in ga:
        assert torch.equal(ga[n], gb[n]), n
    dropped = set(oa) - set(ob)
    assert dropped and all(k.startswith("fan_") for k in dropped), dropped
    assert any(k.startswith("fan_epi_t") for k in dropped)
    for k in ob:
        if torch.is_tensor(ob[k]):
            assert torch.equal(oa[k], ob[k]), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
def test_pinned_prefetch_keeps_every_value():
    """The trainer's prefetcher pins each batch on its producer thread (pin=True) so the
    copies to the GPU can be non_blocking; the values must be the ones it was given."""
    from morph.training.data_placement import Prefetcher
    _ids, inp, lab, layout = _batch(M)
    src = [(inp, lab, layout), (inp + 1, lab, layout)]
    got = list(Prefetcher(iter(src), depth=2, name="t", pin=True))
    assert len(got) == 2
    for (x0, y0, l0), (x1, y1, l1) in zip(src, got):
        assert x1.is_pinned() and y1.is_pinned() and l1.slot_valid.is_pinned()
        assert torch.equal(x0, x1) and torch.equal(y0, y1)
        for f in ("slot_mask", "bag_id", "slot_index", "slot_valid", "span_len",
                  "len_supervised"):
            a, b = getattr(l0, f), getattr(l1, f)
            assert (a is None and b is None) or torch.equal(a, b), f
        assert l1.prefix_k == l0.prefix_k and l1.stats == l0.stats
        d = l1.to("cuda")
        assert np.array_equal(hs.shadow(d.slot_valid), l0.slot_valid.numpy())
        assert torch.equal(d.slot_valid.cpu(), l0.slot_valid)
