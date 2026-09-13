"""The reread (2026-09-10): ``tul.reread`` lets the looping slot cross-attend the frozen
prelude token states each pass. Contracts: the knob is bit-identical at its default and at
step 0 (W_o zero); the allow mask reads exactly the span (or the causal prefix) and never a
slot cell; the read is live once W_o moves; the paid loop refuses it.
"""
from __future__ import annotations

import pytest
import torch

from morph.model import tul as tul_mod
from morph.model.tul import TULConfig, reread_allow
from tests.test_tul_forward import V, _model
from tests.test_tul_unpack import _ids, _pack, _tul


def _slots(model, x, lay):
    """The looped slot states handed to prefix_project (the spy the state probe uses)."""
    real = model.tul.prefix_project
    box = {}

    def spy(h_slots, layout, l_total, cells=None):
        box["h"] = h_slots.detach().clone()
        return real(h_slots, layout, l_total, cells=cells)

    model.tul.prefix_project = spy
    try:
        model.eval()
        with torch.no_grad():
            model(x, labels=None, slot_layout=lay)
    finally:
        model.tul.prefix_project = real
    return box["h"]


def test_reread_allow_reads_the_span_or_the_causal_prefix_and_never_a_slot_cell():
    _, _, lay = _pack(_ids())
    B, L = lay.bag_id.shape
    S = lay.slot_index.shape[1]
    span = reread_allow(lay, "span")
    causal = reread_allow(lay, "causal")
    assert span.shape == (B, S, L) and causal.shape == (B, S, L)
    for b in range(B):
        for s in range(S):
            if not bool(lay.slot_valid[b, s]):
                assert span[b, s, 0] and not span[b, s, 1:].any()
                assert causal[b, s, 0] and not causal[b, s, 1:].any()
                continue
            for p in range(L):
                tok = not bool(lay.slot_mask[b, p])
                k = int(lay.bag_id[b, p])
                assert bool(span[b, s, p]) == (tok and k == s), (b, s, p)
                assert bool(causal[b, s, p]) == (tok and k <= s), (b, s, p)
            assert span[b, s].any(), "every valid slot has a span to read (min span 4)"
    with pytest.raises(ValueError):
        reread_allow(lay, "all")


def test_reread_default_builds_nothing_and_zero_init_is_bit_identical():
    x, _, lay = _pack(_ids())
    base = _model(_tul())
    rr = _model(_tul(reread=True))
    assert base.tul_reread is None and rr.tul_reread is not None
    extra = {k for k in rr.state_dict() if k.startswith("tul_reread.")}
    assert extra == {"tul_reread.W_q", "tul_reread.W_k", "tul_reread.W_v",
                     "tul_reread.W_o", "tul_reread.q_scale"}
    # the private-generator init leaves every other weight byte-identical
    sb, sr = base.state_dict(), rr.state_dict()
    assert set(sb) == set(sr) - extra
    for k in sb:
        assert torch.equal(sb[k], sr[k]), k
    assert torch.equal(rr.tul_reread.W_o, torch.zeros_like(rr.tul_reread.W_o))
    assert torch.equal(_slots(base, x, lay), _slots(rr, x, lay))
    base.eval(); rr.eval()
    with torch.no_grad():
        lb = base(x, labels=None, slot_layout=lay)["logits"]
        lr = rr(x, labels=None, slot_layout=lay)["logits"]
    assert torch.equal(lb, lr)


def test_reread_is_live_and_causal_once_w_o_moves(monkeypatch):
    ids = _ids()
    x, _, lay = _pack(ids)
    rr = _model(_tul(reread=True))
    with torch.no_grad():
        rr.tul_reread.W_o.normal_(0.0, 0.05, generator=torch.Generator().manual_seed(3))
    h = _slots(rr, x, lay)
    zero = _model(_tul(reread=True))
    assert not torch.equal(h, _slots(zero, x, lay)), "the read must move the slot state"

    # perturb a token INSIDE span 2 (not its boundary, so the layout is unchanged) — slot 0
    # and slot 1 must not move: they read only bags <= their own index
    b = 0
    span2 = [p for p in range(lay.bag_id.shape[1])
             if int(lay.bag_id[b, p]) == 2 and not bool(lay.slot_mask[b, p])]
    assert len(span2) >= 3
    x2 = x.clone()
    p = span2[1]
    x2[b, p] = (int(x2[b, p]) % (V - 20)) + 12       # never a boundary id (DOT=10, 11, 0) or slot 4
    assert torch.equal(_pack(ids)[2].bag_id, lay.bag_id)
    h2 = _slots(rr, x2, lay)
    assert torch.equal(h[b, :2], h2[b, :2]), "a later span leaked into an earlier slot"
    assert not torch.equal(h[b, 2], h2[b, 2]), "the slot of the perturbed span must read it"

    # fixture sensitivity: with the mask broken open, the earlier slots DO move
    monkeypatch.setattr(tul_mod, "reread_allow",
                        lambda layout, scope: torch.ones_like(reread_allow(layout, scope)))
    h_open = _slots(rr, x, lay)
    h2_open = _slots(rr, x2, lay)
    assert not torch.equal(h_open[b, :2], h2_open[b, :2]), \
        "the test cannot see a leak: a broken mask must move slot 0/1"


def test_reread_refuses_the_paid_loop_and_a_missing_core():
    with pytest.raises(NotImplementedError):
        TULConfig(prefix_k=2, slot_id=4, tokens_through_core=True, reread=True)
    with pytest.raises(ValueError):
        TULConfig(prefix_k=2, slot_id=4, reread=True, reread_scope="all")
    with pytest.raises(ValueError):
        _model(_tul(reread=True), n_core=0)
