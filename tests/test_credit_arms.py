"""Credit-assignment arms (2026-10-10): ``tul.fan_lsel_pick`` and ``tul.coda_prelude_detach``.

Files: morph/model/tul.py (``_check_credit_arms``), morph/model/transformer.py
(``_lsel_pick_random`` in ``_lsel_pass``, ``_coda_prelude_detach`` at the coda seam in
``_forward_tul``), morph/training/tul_setup.py, configs ``lxtul_pointer_fast_randpick`` and
``..._preldet``.

  1. Both defaults are bit-identical to the key being absent (loss, every grad).
  2. random pick: the followed winner is not the router's argmax everywhere, the coda's routed
     cell is the followed one, all M cells get picked over draws, eval follows random too,
     refusals.
  3. prelude detach: loss and logits bit-identical to off; every grad equals a HAND-DETACHED
     reference (off model, ``scatter_positions`` base detached by a wrapper) and the prelude
     grads differ from off, so removing the detach fails the test; refusals.

CPU, fp32, the ``tests/test_tul_fan_lsel.py`` strict LXTUL fixtures (``M = 4`` fan cells).
"""
from __future__ import annotations

import pytest
import torch

import morph.model.transformer as tr
from morph.model.transformer import MORPHTransformer
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M

M = int(M)
R = dict(fan_lsel_train_follow="router")


def _fwd(m, seed=7, train=True):
    _ids, inp, lab, layout = _batch(M)
    m.train(train)
    torch.manual_seed(seed)
    if train:
        return m(inp, labels=lab, slot_layout=layout)
    with torch.no_grad():
        return m(inp, labels=lab, slot_layout=layout)


def _logits(m):
    _ids, inp, _lab, layout = _batch(M)
    m.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        return m(inp, slot_layout=layout)


def _grads(m):
    out = _fwd(m)
    out["loss"].backward()
    return out, {n: p.grad for n, p in m.named_parameters()}


def _assert_same(a, b, ga, gb):
    assert torch.equal(a["loss"], b["loss"])
    n = 0
    for k in ga:
        if ga[k] is None:
            assert gb[k] is None, k
            continue
        assert torch.equal(ga[k], gb[k]), k
        n += 1
    assert n > 0


# ── 1. defaults are the tree ────────────────────────────────────────────────────────


@pytest.mark.parametrize("key,val", [("fan_lsel_pick", "router"),
                                     ("coda_prelude_detach", False)])
def test_default_is_bit_identical_to_key_absent(key, val):
    a, b = _lsel("joint"), _lsel("joint", **{key: val})
    assert not a._lsel_pick_random and not b._lsel_pick_random
    assert not a._coda_prelude_detach and not b._coda_prelude_detach
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    oa, ga = _grads(a)
    ob, gb = _grads(b)
    _assert_same(oa, ob, ga, gb)


# ── 2. random pick ──────────────────────────────────────────────────────────────────


def _random_run(seed, train=True):
    m = _lsel("joint", fan_lsel_pick="random", **R)
    assert m._lsel_pick_random
    cap = []
    m._lsel_capture = cap
    routed = []
    orig = MORPHTransformer._fan_route_cells

    def spy(cells, route):
        out = orig(cells, route)
        routed.append((route["winner"].detach().clone(), out.detach().clone()))
        return out

    MORPHTransformer._fan_route_cells = staticmethod(spy)
    try:
        _fwd(m, seed=seed, train=train)
    finally:
        MORPHTransformer._fan_route_cells = staticmethod(orig)
    return m, cap, routed


def test_followed_winner_differs_from_router_and_coda_reads_it():
    m, cap, routed = _random_run(3)
    assert cap and routed
    diff = sum(int(((c["follow"] != c["router"]) & c["act"]).sum()) for c in cap)
    assert diff > 0, "random follow equals the router's argmax everywhere"
    winner, out = routed[0]
    # the winner is the LAST active pass's follow for each slot
    last = torch.zeros_like(winner)
    for c in cap:
        last = torch.where(c["act"], c["follow"], last)
    valid = m._lsel_out["winner"] == winner
    assert bool(valid.all())
    act_any = torch.zeros_like(winner, dtype=torch.bool)
    for c in cap:
        act_any |= c["act"]
    assert torch.equal(winner[act_any], last[act_any])
    # coda's routed stack: only the followed cell is non-zero, per slot
    B, S = winner.shape
    nz = out.flatten(3).abs().sum(-1) > 0 if out.dim() > 3 else out.abs() > 0   # [B,S,M]
    oh = torch.nn.functional.one_hot(winner, M).bool()
    assert not bool((nz & ~oh).any())
    assert bool((nz & oh).any())


def test_router_pick_is_still_recorded():
    m, cap, _ = _random_run(3)
    # final_router is the router's own pick, not the followed one: it is stashed in the
    # exit stats as a regret/agreement instrument; here check the capture keeps both.
    assert any(bool((c["router"] != c["follow"]).any()) for c in cap)


def test_all_cells_get_picked_over_draws():
    seen = set()
    for s in range(12):
        m, _, routed = _random_run(100 + s)
        seen |= set(routed[0][0].flatten().tolist())
    assert seen == set(range(M))


def test_eval_follows_random_too():
    _, _, r1 = _random_run(1, train=False)
    _, _, r2 = _random_run(2, train=False)
    assert not torch.equal(r1[0][0], r2[0][0])


def test_random_refusals():
    with pytest.raises(ValueError, match="'router' or 'random'"):
        _lsel("joint", fan_lsel_pick="best", **R)
    with pytest.raises(ValueError, match="fan_loop_select != 'off'"):
        _lsel("off", fan_lsel_pick="random")
    with pytest.raises(ValueError, match="fan_lsel_train_follow"):
        _lsel("joint", fan_lsel_pick="random", fan_lsel_train_follow="teacher")


# ── 3. prelude detach ───────────────────────────────────────────────────────────────


def _hand_detached(monkeypatch):
    orig = tr.scatter_positions
    n = []

    def wrapped(base, pos, values):
        n.append(1)
        return orig(base.detach(), pos, values)

    monkeypatch.setattr(tr, "scatter_positions", wrapped)
    return n


def test_detach_forward_is_bit_identical_and_backward_is_cut(monkeypatch):
    off = _lsel("joint")
    on = _lsel("joint", coda_prelude_detach=True)
    assert on._coda_prelude_detach
    for k, v in off.state_dict().items():
        assert torch.equal(v, on.state_dict()[k]), k
    ref = _lsel("joint")
    o_off, g_off = _grads(off)
    o_on, g_on = _grads(on)
    # (a) forward identical: loss and logits
    assert torch.equal(o_off["loss"], o_on["loss"])
    e_off, e_on = _logits(off), _logits(on)
    assert e_off["logits"] is not None and torch.equal(e_off["logits"], e_on["logits"])
    # (b) the on model's grads equal the hand-detached reference, param by param
    n = _hand_detached(monkeypatch)
    o_ref, g_ref = _grads(ref)
    assert n, "the reference never detached; the seam moved"
    assert torch.equal(o_ref["loss"], o_on["loss"])
    differ = 0
    for k in g_on:
        if g_on[k] is None:
            assert g_ref[k] is None, k
            continue
        assert torch.equal(g_on[k], g_ref[k]), k
        if g_off[k] is not None and not torch.equal(g_off[k], g_on[k]):
            differ += 1
    # (c) and the detach changed something: grads under off are not the on grads
    assert differ > 0, "detach left every gradient unchanged; it is not cutting anything"


def test_detach_refusals():
    with pytest.raises(ValueError, match="coda_token_input"):
        _lsel("joint", coda_prelude_detach=True, coda_token_input="embed")
    with pytest.raises((ValueError, NotImplementedError)):
        _lsel("joint", coda_prelude_detach=True, tokens_through_core=True)
