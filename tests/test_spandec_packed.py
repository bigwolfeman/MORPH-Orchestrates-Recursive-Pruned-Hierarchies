"""``model.spandec_packed`` (2026-10-11): the span decoder's blocks on a PACKED layout.

Files: morph/model/tul_spandec.py (``pack_spans``, ``chunk_segment_mask``, ``PackedCtx``,
``SpanDecoder.decode_packed``, the ``attn_mask`` / ``slot`` arguments of ``_SpanDecBlock``
and ``_SpanDecCross``), morph/model/transformer.py (``MORPHConfig.spandec_packed``, the
build refusals, ``_spandec_packed_cap``, the branch in ``_tul_spandec_loss``),
morph/training/train.py and morph/configs/base.yaml (the key).

What each test pins:
  a. ``pack_spans`` puts every real position at its slot-order destination, with the right
     slot, offset, label and previous token, and flags a row over capacity.
  b. Packed on vs off, SAME weights, on a batch with spans longer than J, pad slots, and a
     row with few slots: the spandec loss and EVERY parameter gradient (the decoder's, the
     cross-attention's and the upstream cells') agree. The zero-init weights (position
     table, cross projection) are randomised first, or the cross-attention and the
     position path would carry no gradient to compare.
  c. SABOTAGE: with the segment mask opened to cross-segment attention, test b fails.
  d. A batch over the capacity gives a NaN loss, not a silent drop.
  e. The refusals, by name; the key builds the same parameters; it is wired in
     ``build_morph_config`` and ``base.yaml``.

CPU, fp32, the ``tests/test_tul_fan_lsel.py`` ungraded-fan fixtures with M = 4 cells and
``tul.spandec_reads_cells``.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

import morph.model.tul_spandec as sd
from morph.model.tul_layout import slot_layout_from_ids
from test_tul_fan import DOT, V, _rule, _spec
from test_tul_fan_lsel import _build, _nowta_kw
from test_tul_lx_credit import M

M = int(M)
J_SMALL = 6          # spandec_max_tokens: shorter than the 8-token spans, so spans are cut


def _ids() -> np.ndarray:
    """Three rows: dense boundaries (8-token spans, some shorter), NO boundary at all (two
    32-token spans, then pad slots), and a few boundaries (few slots, longer spans)."""
    rng = np.random.default_rng(3)
    ids = rng.integers(5, V, size=(3, 120))
    ids[ids == 4] = 5
    ids[ids == 11] = 5
    ids[ids == DOT] = 5
    ids[0, ::8] = DOT
    ids[0, 3::13] = 11
    ids[2, 15::20] = DOT
    return ids.astype(np.int64)


def _batch3():
    inp, lab, layout, _ = slot_layout_from_ids(_ids(), _rule(), _spec(M))
    return inp, lab, layout


def _model(packed: bool, **kw):
    return _build(model_kw={"spandec_packed": packed},
                  **_nowta_kw(fan_loop_select="off", spandec_reads_cells=True,
                              spandec_max_tokens=J_SMALL, **kw))


def _randomise_zero_inits(m, seed=11):
    """The position table and every cross-attention projection start at exactly 0, which
    would hide the position and cross paths from a gradient comparison."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.tul_spandec.parameters():
            if float(p.abs().max()) == 0.0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def _run(m, inp, lab, layout):
    """``(spandec loss, {param name: grad})`` of one train forward."""
    cap = {}
    orig = m._tul_spandec_loss

    def wrapped(*a, **kw):
        cap["loss"] = orig(*a, **kw)
        return cap["loss"]

    m._tul_spandec_loss = wrapped
    m.train()
    torch.manual_seed(5)
    m(inp, labels=lab, slot_layout=layout)
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    gs = torch.autograd.grad(cap["loss"], [p for _, p in named], allow_unused=True)
    return cap["loss"].detach(), {n: g for (n, _), g in zip(named, gs)}


def _compare():
    inp, lab, layout = _batch3()
    off, on = _model(False), _model(True)
    assert on._spandec_packed and not off._spandec_packed
    _randomise_zero_inits(off)
    on.load_state_dict(off.state_dict())
    return _run(off, inp, lab, layout), _run(on, inp, lab, layout)


def _assert_same(a, b):
    (l0, g0), (l1, g1) = a, b
    torch.testing.assert_close(l1, l0, rtol=1e-5, atol=0)
    assert g0.keys() == g1.keys()
    n_dec = n_up = 0
    for k in g0:
        if g0[k] is None:
            assert g1[k] is None, k
            continue
        rel = float((g1[k] - g0[k]).norm() / (g0[k].norm() + 1e-12))
        assert rel < 1e-4, f"{k}: relative grad error {rel:.3e}"
        if float(g0[k].norm()) > 0:
            if k.startswith("tul_spandec."):
                n_dec += 1
            else:
                n_up += 1
    # real gradients, not an all-zero comparison: the decoder (cross included) and upstream
    assert n_dec >= 20 and n_up >= 5, (n_dec, n_up)
    cross = [k for k in g0 if ".cross." in k and g0[k] is not None]
    assert cross and all(float(g0[k].norm()) > 0 for k in cross)


# ── a. the pack ─────────────────────────────────────────────────────────────────────

def test_pack_spans_destinations():
    inp, _lab, layout = _batch3()
    ids, valid = sd.horizon_span_slots(inp, layout, J_SMALL, 1, start=1)
    n = valid.sum(-1)
    assert int(n.max()) == J_SMALL and 0 < int((n == 0).sum()), "need cut spans and pad slots"
    assert 0 < int(((n > 0) & (n < J_SMALL)).sum()), "need a short span too"
    most = int(n.sum(-1).max())
    pk = sd.pack_spans(ids, valid, most + 7)
    assert not bool(pk.over)
    for b in range(ids.shape[0]):
        p = 0
        for s in range(ids.shape[1]):
            for j in range(int(n[b, s])):
                assert (int(pk.slot[b, p]), int(pk.j[b, p]), int(pk.ids[b, p])) == \
                    (s, j, int(ids[b, s, j]))
                assert bool(pk.ok[b, p])
                assert int(pk.prev[b, p]) == (int(ids[b, s, j - 1]) if j else 0)
                p += 1
        assert not bool(pk.ok[b, p:].any()) and int(pk.ids[b, p:].abs().sum()) == 0
    assert bool(sd.pack_spans(ids, valid, most - 1).over)
    assert not bool(sd.pack_spans(ids, valid, most).over)


# ── b. packed == per-slot ───────────────────────────────────────────────────────────

def test_packed_matches_per_slot_loss_and_every_grad():
    _assert_same(*_compare())


# ── c. sabotage ─────────────────────────────────────────────────────────────────────

def test_sabotaged_segment_mask_is_caught(monkeypatch):
    """Let every position attend across segments (causal inside the window only): the
    comparison MUST fail, or test b cannot tell a segment mask from none."""
    def causal_only(seg, chunk):
        B, P = seg.shape
        i = torch.arange(chunk).view(chunk, 1)
        k = torch.arange(2 * chunk).view(1, 2 * chunk)
        return (k <= i + chunk).expand(B * (P // chunk), 1, chunk, 2 * chunk)

    monkeypatch.setattr(sd, "chunk_segment_mask", causal_only)
    with pytest.raises(AssertionError):
        _assert_same(*_compare())


# ── d. capacity ─────────────────────────────────────────────────────────────────────

def test_over_capacity_batch_gives_nan_loss(monkeypatch):
    inp, lab, layout = _batch3()
    on = _model(True)
    monkeypatch.setattr(type(on), "_spandec_packed_cap", staticmethod(lambda L, lay, dec: 8))
    loss, _ = _run(on, inp, lab, layout)
    assert torch.isnan(loss)


def test_capacity_covers_the_batch():
    inp, lab, layout = _batch3()
    on = _model(True)
    cap = on._spandec_packed_cap(inp.shape[1], layout, on.tul_spandec)
    K, S, L = M, layout.max_slots, inp.shape[1]
    assert cap == max(min(L - K * n, J_SMALL * max(n - 1, 0)) for n in range(S + 1))
    _ids_, valid = sd.horizon_span_slots(inp, layout, J_SMALL, 1, start=1)
    assert int(valid.sum((1, 2)).max()) <= cap
    assert torch.isfinite(_run(on, inp, lab, layout)[0])


# ── e. refusals, wiring ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw, match", [
    (dict(spandec_horizon=2), "spandec_horizon"),
    (dict(spandec_target_offset=2), "spandec_target_offset"),
    (dict(spandec_per_pass=True), "spandec_per_pass"),
])
def test_unsupported_modes_are_refused(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _build(model_kw={"spandec_packed": True},
               **_nowta_kw(fan_loop_select="off", spandec_reads_cells=True, **kw))


def test_refused_without_a_decoder_and_with_read_each():
    with pytest.raises(ValueError, match="spandec"):
        _build(model_kw={"spandec_packed": True}, **_nowta_kw(spandec=False))
    with pytest.raises(ValueError, match="spandec_read"):
        _build(model_kw={"spandec_packed": True}, **_nowta_kw(spandec_read="each"))


def test_the_key_builds_the_same_parameters():
    a = _build(**_nowta_kw(fan_loop_select="off", spandec_reads_cells=True))
    b = _model(True)
    assert not a._spandec_packed and b._spandec_packed
    assert a.state_dict().keys() == b.state_dict().keys()


def test_key_is_wired_through_hydra():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "spandec_packed: false" in (root / "morph/configs/base.yaml").read_text()
    assert "spandec_packed=bool(" in (root / "morph/training/train.py").read_text()
