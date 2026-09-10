"""The coda reads the thought (Wolfe 2026-09-09): ``tul.bcast`` (the unpack),
``tul.coda_token_input='embed'`` and ``tul.tg_restrict_scope='coda'``.

The contract these protect: under the three knobs a token's logits can reach an earlier
span ONLY through the slot cells and the unpack term, i.e. through z. The shipped §3.4
path let every token carry its own global prelude state into the coda, which is why z
was optional to the reader. Also: the knobs are bit-identical at their defaults, the
zero-init unpack changes nothing at step 0, and the paid loop refuses them.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.tul import TULConfig, unpack_index
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from tests.test_tul_forward import DOT, V, _model, _rule, _spec


def _ids(B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::6] = DOT
    return ids.astype(np.int64)


def _pack(ids, spec=None):
    """(input_ids, labels, layout) for a raw id block."""
    x, y, lay, _ = slot_layout_from_ids(ids, _rule(), spec or _spec())
    return x, y, lay


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, token_state_dropout=0.0, slot_seed="boundary")
    base.update(kw)
    return TULConfig(**base)


# ── the index helper ──────────────────────────────────────────────────────────

def test_unpack_index_offsets_restart_at_each_span_and_point_at_the_previous_slot():
    ids = _ids()
    _, _, lay = _pack(ids)
    src, off, valid = unpack_index(lay)
    B, L = lay.bag_id.shape
    S = lay.slot_index.shape[1]
    for b in range(B):
        for p in range(L):
            k = int(lay.bag_id[b, p])
            if bool(lay.slot_mask[b, p]) or k == 0:
                assert not valid[b, p]
                continue
            prev = min(k - 1, S - 1)
            if not bool(lay.slot_valid[b, prev]):
                assert not valid[b, p]
                continue
            assert valid[b, p], (b, p, k)
            assert int(src[b, p]) == prev
            start = int(lay.slot_index[b, prev]) + lay.prefix_k
            assert int(off[b, p]) == p - start
            assert p >= start
    # every valid span-first token reads offset 0, and offsets are contiguous inside a span
    for b in range(B):
        for s in range(S):
            if not bool(lay.slot_valid[b, s]):
                continue
            first = int(lay.slot_index[b, s]) + lay.prefix_k
            if first < L and not bool(lay.slot_mask[b, first]):
                assert valid[b, first] and int(off[b, first]) == 0
    assert valid.any()
    # the invalid entries are clamped to 0, never a stray index
    assert int(src[~valid].max()) == 0 if (~valid).any() else True
    assert int(off[~valid].max()) == 0 if (~valid).any() else True


def test_unpack_index_covers_the_unterminated_tail():
    """The span being decoded at generation time is never terminated: the tail must
    still decode the last slot (mux_span_targets requires termination; this must not)."""
    ids = _ids()
    _, _, lay = _pack(ids)
    src, off, valid = unpack_index(lay)
    S = lay.slot_index.shape[1]
    for b in range(ids.shape[0]):
        last = int(lay.slot_valid[b].nonzero().max())
        tail = (lay.bag_id[b] == last + 1) & ~lay.slot_mask[b]
        if tail.any():
            assert valid[b][tail].all()
            assert (src[b][tail] == last).all()


# ── config contract ───────────────────────────────────────────────────────────

def test_bcast_is_implemented_and_the_paid_loop_refuses_it():
    _tul(bcast=True)
    with pytest.raises(NotImplementedError):
        _tul(bcast=True, tokens_through_core=True)
    with pytest.raises(NotImplementedError):
        _tul(coda_token_input="embed", tokens_through_core=True)
    with pytest.raises(ValueError):
        _tul(tg_restrict_scope="coda")            # no restriction to scope
    with pytest.raises(ValueError):
        _tul(coda_token_input="bogus")


# ── bit-identity at the defaults / at zero init ───────────────────────────────

def _logits(model, x, lay, plan_mode="normal"):
    """Logits with the slot_id column dropped (it is -inf everywhere, and inf - inf is NaN)."""
    model.eval()
    with torch.no_grad():
        res = model.tul_forward_ablated(x, None, lay, plan_mode=plan_mode)
    keep = [c for c in range(V) if c != 4]
    return res["logits"].float()[..., keep]


def test_zero_init_unpack_is_bit_identical_and_trains():
    ids = _ids()
    x, y, lay = _pack(ids)
    off = _model(_tul(bcast=False))
    on = _model(_tul(bcast=True))
    assert on.tul.W_bcast is not None and on.tul.W_bcast.shape[1:] == (64, 64)
    assert torch.equal(_logits(off, x, lay), _logits(on, x, lay))
    # a nonzero unpack changes the logits at decoded positions and NOT in span 0
    with torch.no_grad():
        on.tul.W_bcast.normal_(std=0.5)
    a, b = _logits(off, x, lay), _logits(on, x, lay)
    span0 = (lay.bag_id == 0) & ~lay.slot_mask
    assert torch.allclose(a[span0], b[span0], atol=1e-5)
    _, _, valid = unpack_index(lay)
    assert (a[valid] - b[valid]).abs().max() > 1e-3
    # the parameter receives gradient through the token CE
    on.train()
    out = on(x, labels=y, slot_layout=lay)
    out["loss"].backward()
    assert on.tul.W_bcast.grad is not None and on.tul.W_bcast.grad.abs().sum() > 0


def test_coda_token_input_embed_changes_the_coda_input_only():
    ids = _ids()
    x, _, lay = _pack(ids)
    pre = _model(_tul())
    emb = _model(_tul(coda_token_input="embed"))
    # same parameters (the knob builds nothing)
    for (n1, p1), (n2, p2) in zip(pre.named_parameters(), emb.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    assert not torch.equal(_logits(pre, x, lay), _logits(emb, x, lay))


# ── THE contract: earlier spans reach a token only through z ─────────────────

def _perturb_span0(ids, lay, safe_gap=4):
    """New ids with span 0's tokens replaced at ``safe_gap`` or more positions before its
    boundary (boundaries and layout fixed).

    Why the gap: CCA's causal depthwise convolutions on Q and K (kernel 4) and its
    previous-position value term mix the three positions before a cell into that cell,
    across span boundaries and outside any attention mask. So a slot cell's keys and
    values carry the last three tokens of the span it terminates. That is a 3-token
    local window, not a route to the span's content; the contract is stated outside it.
    """
    ids2 = ids.copy()
    for b in range(ids.shape[0]):
        n_span0 = int(((lay.bag_id[b] == 0) & ~lay.slot_mask[b]).sum())
        boundary = n_span0 - 1          # the raw row's span 0 = its first n_span0 tokens
        for p in range(n_span0):
            if ids2[b, p] != DOT and boundary - p >= safe_gap:
                ids2[b, p] = 5 + (ids2[b, p] + 7) % (V - 5)
                if ids2[b, p] == DOT:
                    ids2[b, p] = 6
    assert (ids2 != ids).any(), "nothing perturbed: widen the span or lower safe_gap"
    return ids2


def test_under_the_arm_the_coda_sees_earlier_spans_only_through_z():
    ids = _ids()
    x, _, lay = _pack(ids)
    ids2 = _perturb_span0(ids, lay)
    x2, _, lay2 = _pack(ids2)
    assert torch.equal(lay.bag_id, lay2.bag_id) and torch.equal(lay.slot_index, lay2.slot_index)
    arm = _model(_tul(coda_token_input="embed", tg_restrict=True, tg_restrict_scope="coda",
                      bcast=True))
    with torch.no_grad():
        arm.tul.W_bcast.normal_(std=0.3)
    later = (lay.bag_id >= 2) & ~lay.slot_mask
    # with the thought ablated (zero), span >= 2 tokens cannot tell span 0 changed
    a = _logits(arm, x, lay, plan_mode="zero")
    b = _logits(arm, x2, lay2, plan_mode="zero")
    assert torch.allclose(a[later], b[later], atol=1e-4), float((a[later] - b[later]).abs().max())
    # with the thought present they can
    a = _logits(arm, x, lay)
    b = _logits(arm, x2, lay2)
    assert (a[later] - b[later]).abs().max() > 1e-3
    # the shipped path leaks span 0 into span >= 2 even with the thought zeroed
    ship = _model(_tul())
    a = _logits(ship, x, lay, plan_mode="zero")
    b = _logits(ship, x2, lay2, plan_mode="zero")
    assert (a[later] - b[later]).abs().max() > 1e-3


def test_scope_coda_keeps_the_prelude_global():
    """Under scope 'coda' the prelude runs with NO mask and the coda with the coda rule;
    under scope 'all' (the shipped TG arms) both get the mask. Spied at the seams
    (`_tul_front` / `_back_region`), the two forwards that exist: training and the
    frozen-feature read-out."""
    ids = _ids()
    x, y, lay = _pack(ids)
    for scope, front_masked in (("coda", False), ("all", True)):
        m = _model(_tul(tg_restrict=True, tg_restrict_scope=scope))
        seen = {}
        real_front, real_back = m._tul_front, m._back_region

        def front(*a, _seen=seen, _real=real_front, **kw):
            _seen["front"] = kw.get("attn_kwargs")
            return _real(*a, **kw)

        def back(*a, _seen=seen, _real=real_back, **kw):
            _seen["back"] = kw.get("attn_kwargs")
            return _real(*a, **kw)

        m._tul_front, m._back_region = front, back
        m.eval()
        with torch.no_grad():
            m(x, labels=y, slot_layout=lay)
        assert (seen["front"] is not None) == front_masked, scope
        assert seen["back"] is not None and "tg_allow" in seen["back"]
        assert ("tg_seg" in seen["back"]) == (scope == "coda")
        seen.clear()
        with torch.no_grad():
            m.prelude_states(x, layout=lay)
        assert (seen["front"] is not None) == front_masked, scope


def test_coda_rule_slot_cells_attend_slot_cells_only():
    from morph.model.tul_layout import tg_allow_mask
    ids = _ids()
    _, _, lay = _pack(ids)
    base = tg_allow_mask(lay)[:, 0]
    coda = tg_allow_mask(lay, slot_queries_slots_only=True)[:, 0]
    slot_q = lay.slot_mask.unsqueeze(2)
    # token queries: unchanged
    assert torch.equal(base[~lay.slot_mask], coda[~lay.slot_mask])
    # slot queries: only slot keys, causal
    L = lay.bag_id.shape[1]
    causal = (torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1))
    want = lay.slot_mask.unsqueeze(1) & causal.unsqueeze(0)
    assert torch.equal(coda[lay.slot_mask], want.expand_as(coda)[lay.slot_mask])
    assert (base[lay.slot_mask] & ~coda[lay.slot_mask]).any()   # the rule removed something


def test_segment_causal_conv_matches_the_reference_and_cuts_at_boundaries():
    import torch.nn.functional as F
    from morph.kernels.triton.fused_cca_conv import causal_conv_reference
    from morph.model.attention import segment_causal_conv
    torch.manual_seed(0)
    B, C, S, k, G = 2, 8, 20, 4, 2
    x = torch.randn(B, C, S)
    w_dw = torch.randn(C, 1, k)
    w_gp = torch.randn(C, C // G, k)
    ref = causal_conv_reference(x, w_dw, w_gp, k)
    seg0 = torch.zeros(B, S, dtype=torch.long)
    assert torch.allclose(segment_causal_conv(x, w_dw, w_gp, seg0), ref, atol=1e-5)
    # two segments: outputs in the second segment are independent of the first
    seg = torch.zeros(B, S, dtype=torch.long)
    seg[:, 9:] = 1
    a = segment_causal_conv(x, w_dw, w_gp, seg)
    x2 = x.clone()
    x2[:, :, :9] = torch.randn(B, C, 9)
    b = segment_causal_conv(x2, w_dw, w_gp, seg)
    assert torch.allclose(a[:, :, 9:], b[:, :, 9:], atol=1e-6)
    assert not torch.allclose(a[:, :, :9], b[:, :, :9])
    # and inside a segment the taps are the reference's (the first segment alone)
    ref1 = causal_conv_reference(x[:, :, :9], w_dw, w_gp, k)
    assert torch.allclose(a[:, :, :9], ref1, atol=1e-5)
