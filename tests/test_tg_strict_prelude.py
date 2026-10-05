"""`tul.tg_strict_prelude` — the PRELUDE's own history (2026-10-05).

Wolfe: "we need to test the prelude and loop seeing full token history." Under
``tul.tg_geometry="strict"`` the prelude is SAME-SPAN only (``tg_strict_allow(layout,
"prelude")`` — ``causal AND bag_id[i] == bag_id[j]``), so the slot loop's seed (pooled
from the span's own prelude token states) is the only thing that ever sees an earlier
span. ``tg_strict_prelude="causal"`` widens a TOKEN query's prelude relation to every
earlier TOKEN of the row (``causal AND NOT slot_mask[j]``), so the prelude itself
becomes a causal model of the row's history; a SLOT-CELL query's own relation is
UNCHANGED (its own span's tokens and its own earlier cells — the loop reaches the row
through its ordinary seed, not through a second route). The conv / value-shift / GLA
retention carry in the PRELUDE must stop resetting at segment boundaries under
"causal" too, or the attention relation would widen while the conv/retention state
kept forgetting at every span edge — not a causal model of anything. The CODA's own
geometry (relation, conv/value-shift/retention reset, the zeroed per-cell injections)
is UNCHANGED by this knob at every value.

THE SIX THINGS THIS FILE PROVES, each by perturbation through the REAL seam
(:meth:`MORPHTransformer._tul_tg_kwargs`, never a hand-rolled kwargs dict — the
2026-09-22 lesson named in ``morph/model/CLAUDE.md``: a mask-capture test that builds
its own kwargs can call a geometry "verified" while the real forward's conv relays
across a segment the hand-rolled dict never exercised):

1. "span" is bit-identical to a model that never set the key (state_dict, loss, logits).
2. "causal" — a token 2 spans back moves the PRELUDE output at span s's tokens, and
   moves the loop's seed/state for slot s. "span" does neither (two-sided).
3. "causal" + ``coda_token_input="prelude"`` — the bypass is OPEN: an earlier span's
   edit still moves the coda's logits at a later span with EVERY slot cell zeroed.
   Documented, not failed: the token carrier at a coda TOKEN position is already a
   global-context summary under this setting, whatever the prelude's relation is.
4. "causal" + ``coda_token_input="embed"`` — the bypass is CLOSED: with every cell
   zeroed, perturbing ANY earlier span's tokens changes NOTHING in the coda's logits at
   a later span, at forced depth 1 AND 2. With the cells live, the same perturbation
   DOES move the logits — the loop is the only route left, and it is connected.
5. "span" + ``coda_token_input="embed"`` is also exact-zero with the cells zeroed (the
   bypass this arm closes does not depend on which prelude mode is running).
6. Config refusals: an unknown value, and the knob outside ``tg_geometry="strict"``.

Two sabotage checks close the file (per the 2026-09-22 lesson): leaving the prelude's
conv/retention reset ON under "causal" is NOT caught by test 2's attention-only
perturbation (the attention relation alone already carries a far edit, so the bulk
logit/state check is blind to a LOCAL conv bug) and IS caught by a dedicated
boundary-adjacent check; letting the CODA's conv cross a segment is caught by the
existing `tests/test_tul_strict_geometry.py` sabotage #4, re-run here to confirm this
change did not touch the coda's partition.

CPU, fp32, `use_kernels=False`, the `tests/test_tul_strict_geometry.py` tiny fixtures —
same reasons as that file (fp32 carries the exact zeros this file asserts; the
`_window_fallback` fp64 mismatch is irrelevant to every dtype MORPH runs).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.transformer as transformer_mod
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (
    slot_layout_from_ids,
    tg_reset_from_ids,
    tg_segment_ids,
    tg_strict_allow,
)
from test_tul_strict_geometry import _edit, _pack, _rule, _spec, _tiny, _tul

# ── helpers: build models and run the REAL `_tul_tg_kwargs` seam ─────────────


def _model(seed: int = 1234, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", **tul_kw)))
    # Zero-init bigram lambdas would make the bigram route inert; the strict-geometry
    # file's own lesson (an id-perturbation test must exercise every live route).
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _prelude_and_slots(m: MORPHTransformer, ids: np.ndarray, depth: int | None = None):
    """``(x, h_slots, layout)`` through the model's OWN `_tul_tg_kwargs` seam.

    `x` is the prelude output over EVERY position (pre-`input_norm`, exactly what
    `_tul_front` returns); `h_slots` the loop's exit state per slot. Both come from the
    production call sequence `_tul_tg_kwargs` -> `_tul_front` -> `_tul_core`
    (`morph/model/transformer.py`'s own `_forward_latent_pre`, `tul_forward_with_plan_nats`
    and `_forward_tul` all make this exact sequence) — never a hand-rolled kwargs dict.
    """
    spec, rule = _spec(), _rule()
    inp, _lab, layout, _ = slot_layout_from_ids(ids, rule, spec)
    front_kw, front_reset, _ckw, _creset = m._tul_tg_kwargs(layout)
    slot_depths = None
    if depth is not None:
        slot_depths = torch.full(layout.slot_index.shape, depth, dtype=torch.long)
        slot_depths = m._pad_slot_depths(slot_depths, layout)
    with torch.no_grad():
        x, x0, bigram = m._tul_front(inp, layout, attn_kwargs=front_kw,
                                     ret_reset_mask=front_reset)
        xn, h_slots, _depths, _g, _db, _gr, _mep = m._tul_core(
            x, x0, bigram, layout, input_ids=inp, slot_depths=slot_depths)
    return x, h_slots, layout


def _logits(m: MORPHTransformer, ids: np.ndarray, plan_mode: str,
           slot_depths: torch.Tensor | None = None):
    spec, rule = _spec(), _rule()
    inp, lab, layout, _ = slot_layout_from_ids(ids, rule, spec)
    with torch.no_grad():
        if plan_mode == "normal" and slot_depths is None:
            out = m(inp, labels=None, slot_layout=layout)
        else:
            out = m.tul_forward_ablated(inp, None, layout, plan_mode=plan_mode,
                                        slot_depths=slot_depths)
    return out["logits"], layout


def _forced_depths(layout, depth: int, m: MORPHTransformer) -> torch.Tensor:
    d = torch.full(layout.slot_index.shape, depth, dtype=torch.long)
    return m._pad_slot_depths(d, layout)


# ── (1) tg_strict_allow's prelude_history knob, on its own ───────────────────


def test_span_prelude_history_is_the_old_call_bit_for_bit():
    ids, *_ = _pack()
    _ids0, _inp, _lab, layout = _pack()
    old = tg_strict_allow(layout, "prelude")
    new = tg_strict_allow(layout, "prelude", prelude_history="span")
    assert torch.equal(old, new)


def test_causal_widens_token_rows_and_leaves_slot_rows_untouched():
    """Direct structural check of the relation tensor, before any model is involved."""
    _ids0, _inp, _lab, layout = _pack()
    span = tg_strict_allow(layout, "prelude", prelude_history="span")
    causal = tg_strict_allow(layout, "prelude", prelude_history="causal")
    sm = layout.slot_mask
    # slot-cell QUERY rows (dim -2) are byte-identical between the two relations.
    row_is_slot = sm.view(*sm.shape, 1)
    assert torch.equal(torch.where(row_is_slot.unsqueeze(1), causal, span),
                       torch.where(row_is_slot.unsqueeze(1), span, span))
    # token QUERY rows strictly widen: every "span" pair still allowed, plus new ones,
    # and every new pair's KEY is a token (never a slot cell).
    added = causal & ~span
    assert bool(added.any()), "causal added nothing — the fixture is inert"
    assert not bool((span & ~causal).any()), "causal must only widen"
    key_is_slot = sm.view(sm.shape[0], 1, 1, -1)
    assert not bool((added & key_is_slot).any()), \
        "a widened pair read a SLOT CELL key — the prelude causal history is tokens only"
    query_is_slot = sm.view(*sm.shape, 1).unsqueeze(1)
    assert not bool((added & query_is_slot).any()), \
        "a slot-cell QUERY gained a pair under causal — its relation must be unchanged"


def test_prelude_history_refuses_a_coda_stage_and_a_bad_value():
    _ids0, _inp, _lab, layout = _pack()
    with pytest.raises(ValueError, match="PRELUDE relation"):
        tg_strict_allow(layout, "coda", prelude_history="causal")
    with pytest.raises(ValueError, match="'span' or 'causal'"):
        tg_strict_allow(layout, "prelude", prelude_history="loose")


# ── (2) condition 1: "span" is bit-identical to a model without the key ──────


def test_span_is_bit_identical_to_a_model_that_never_set_the_key():
    m_key = _model(tg_strict_prelude="span")
    m_none = _model()
    assert m_key.cfg.tul.tg_strict_prelude == "span"
    assert list(m_key.state_dict().keys()) == list(m_none.state_dict().keys())
    for a, b in zip(m_key.state_dict().values(), m_none.state_dict().values()):
        assert torch.equal(a, b), "tg_strict_prelude='span' must draw no RNG"
    ids, *_ = _pack()
    a, _ = _logits(m_key, ids, "normal")
    b, _ = _logits(m_none, ids, "normal")
    assert torch.equal(a, b), "tg_strict_prelude='span' changed the forward — not a no-op"


def test_causal_changes_the_forward_so_the_knob_is_not_inert():
    ids, *_ = _pack()
    a, _ = _logits(_model(), ids, "normal")
    b, _ = _logits(_model(tg_strict_prelude="causal"), ids, "normal")
    assert not torch.equal(a, b), "tg_strict_prelude='causal' changed nothing"


# ── (3) condition 2: causal widens the PRELUDE and the LOOP's seed ───────────


def test_causal_prelude_token_reads_two_spans_back_and_moves_the_loop_seed():
    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)

    m_causal = _model(tg_strict_prelude="causal")
    xa, ha, lay = _prelude_and_slots(m_causal, ids)
    xb, hb, _ = _prelude_and_slots(m_causal, edited)
    tok_s = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    assert bool(tok_s.any()), "fixture: span s must hold token positions"
    d_tok = (xa[0, tok_s] - xb[0, tok_s]).abs().max()
    assert float(d_tok) > 0.0, (
        "causal prelude: an edit 2 spans back did not move span s's prelude output")
    seed_idx = layout.slot_index[0, s]
    assert float((xa[0, seed_idx] - xb[0, seed_idx]).abs().max()) > 0.0, (
        "causal prelude: the edit did not move slot s's own prelude-output seed")
    d_h = (ha[0, s] - hb[0, s]).abs().max()
    assert float(d_h) > 0.0, (
        "causal prelude: the loop's state for slot s did not move — the seed changed but "
        "the loop never saw it")


def test_span_prelude_is_two_sided_against_causal_on_the_same_edit():
    """The control: 'span' (today) shows NEITHER prelude effect on the same perturbation.

    `h_slots` (the loop's exit state) is deliberately NOT compared here. Under
    `tg_geometry="strict"` with the default `loop_reach=0` (unlimited), the LOOP's own
    cross-slot attention already carries slot s-2's post-loop state into slot s within
    one pass, at EVERY value of `tg_strict_prelude` — that is the geometry doing exactly
    its documented job ("the loop is the only cross-span channel"), not an effect of
    this knob. Comparing h_slots here would conflate the two. What this knob can change
    in isolation is whether the PRELUDE itself (before any loop pass) already carries
    the edit — checked at a span-s TOKEN position and at slot s's own SEED position,
    both of which the causal twin (`test_causal_prelude_token_reads_two_spans_back_...`)
    shows move, and both of which must NOT move here.
    """
    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)

    m_span = _model(tg_strict_prelude="span")
    xa, _ha, lay = _prelude_and_slots(m_span, ids)
    xb, _hb, _ = _prelude_and_slots(m_span, edited)
    tok_s = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    d_tok = (xa[0, tok_s] - xb[0, tok_s]).abs().max()
    assert float(d_tok) == 0.0, (
        "LEAK: under tg_strict_prelude='span' an edit 2 spans back moved span s's "
        "prelude output — the prelude is supposed to be same-span only")
    seed_idx = layout.slot_index[0, s]
    d_seed = (xa[0, seed_idx] - xb[0, seed_idx]).abs().max()
    assert float(d_seed) == 0.0, (
        "LEAK: under 'span' an edit 2 spans back moved slot s's own prelude-output seed")


def _model1(seed: int = 1234, **tul_kw) -> MORPHTransformer:
    """A ONE-PRELUDE-LAYER model — isolates "the slot cell's own relation never changed"
    from "a later layer composes a cross-span read the EARLIER layer already picked up"
    (`test_causal_prelude_token_reads_two_spans_back...` measures the composed case;
    with `n_prelude=1` there is no earlier layer, so a slot cell's output can only
    depend on VALUES its (unchanged) relation actually names).
    """
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(n_prelude=1, tul=_tul(tg_geometry="strict", **tul_kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def test_causal_slot_cell_seed_is_unchanged_with_no_layer_to_compose_the_leak():
    """'slot positions keep exactly what they read today' — the ATTENTION RELATION, not
    necessarily the resulting value: at `n_prelude >= 2` slot s's seed DOES move from an
    edit 2 spans back (measured above), but only because span s's OWN tokens already
    absorbed it one layer earlier through THEIR widened relation, and slot s's
    (unchanged) same-span relation then reads those already-contaminated token states —
    composition, not a changed relation. At `n_prelude=1` there is no earlier layer: a
    slot cell's output depends only on the UNCONTAMINATED embeddings its relation names,
    and must be EXACTLY unchanged by an edit 2 spans away, under 'causal' same as
    'span'. Two-sided: a token of span s itself still moves (its own relation DID widen),
    so the fixture is not simply inert at one layer.
    """
    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)
    m = _model1(tg_strict_prelude="causal")
    xa, _ha, lay = _prelude_and_slots(m, ids)
    xb, _hb, _ = _prelude_and_slots(m, edited)
    seed_idx = lay.slot_index[0, s]
    assert float((xa[0, seed_idx] - xb[0, seed_idx]).abs().max()) == 0.0, (
        "at n_prelude=1 the slot cell's own seed moved from an edit 2 spans back — its "
        "relation is supposed to be unchanged (same span, own earlier cells only)")
    tok_s = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    assert float((xa[0, tok_s] - xb[0, tok_s]).abs().max()) > 0.0, (
        "the fixture is inert at one prelude layer — a token of span s itself did not "
        "move, so the zero above proves nothing")


# ── (4) condition 3: causal + coda_token_input='prelude' leaves the bypass open ──


def test_causal_with_prelude_token_input_the_bypass_is_open():
    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)
    m = _model(tg_strict_prelude="causal", coda_token_input="prelude")
    a, lay = _logits(m, ids, "all_slots")
    b, _ = _logits(m, edited, "all_slots")
    tgt = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    d = (a[0] - b[0]).abs().nan_to_num(0.0)
    d = torch.where(a[0] != b[0], d, torch.zeros_like(d))
    assert float(d[tgt].max()) > 0.0, (
        "expected the OPEN bypass: with every cell zeroed, a causal prelude's own-"
        "position token carrier should still carry an earlier span's edit forward")


# ── (5)/(6) condition 4/5: 'embed' closes the bypass, span or causal, depth 1 and 2 ──


@pytest.mark.parametrize("prelude_mode", ["span", "causal"])
@pytest.mark.parametrize("depth", [1, 2])
def test_embed_coda_input_closes_the_bypass_with_cells_zeroed(prelude_mode, depth):
    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)
    m = _model(tg_strict_prelude=prelude_mode, coda_token_input="embed")
    forced = _forced_depths(layout, depth, m)
    a, lay = _logits(m, ids, "all_slots", slot_depths=forced)
    b, _ = _logits(m, edited, "all_slots", slot_depths=forced)
    tgt = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    d = (a[0] - b[0]).abs().nan_to_num(0.0)
    d = torch.where(a[0] != b[0], d, torch.zeros_like(d))
    assert float(d[tgt].max()) == 0.0, (
        f"LEAK at depth {depth}, prelude_mode={prelude_mode!r}: with every slot cell "
        f"zeroed and coda_token_input='embed', an earlier span's edit still moved span "
        f"s's logits — the loop was supposed to be the only cross-span route left")


@pytest.mark.parametrize("depth", [1, 2])
def test_embed_coda_input_with_cells_live_the_loop_carries_the_edit(depth):
    """Two-sided: 'embed' is not simply a model that reads nothing."""
    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)
    m = _model(tg_strict_prelude="causal", coda_token_input="embed")
    forced = _forced_depths(layout, depth, m)
    a, lay = _logits(m, ids, "normal", slot_depths=forced)
    b, _ = _logits(m, edited, "normal", slot_depths=forced)
    tgt = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    d = (a[0] - b[0]).abs().nan_to_num(0.0)
    d = torch.where(a[0] != b[0], d, torch.zeros_like(d))
    assert float(d[tgt].max()) > 0.0, (
        f"depth {depth}: with the slot cells LIVE and coda_token_input='embed', span "
        f"s's edit never reached span s's own logits at all — the loop is not connected")


# ── (6) config refusals ────────────────────────────────────────────────────────


def test_tg_strict_prelude_refuses_an_unknown_value():
    with pytest.raises(ValueError, match="'span' or 'causal'"):
        _tul(tg_geometry="strict", tg_strict_prelude="loose")


def test_tg_strict_prelude_is_refused_outside_strict_geometry():
    with pytest.raises(ValueError, match="tg_geometry='strict'"):
        _tul(tg_strict_prelude="causal")          # default tg_geometry='restrict'
    # 'span' at the default tg_geometry is fine — it is the value every existing arm has.
    TULConfig(**{**_default_kwargs(), "tg_strict_prelude": "span"})


def _default_kwargs():
    return dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0)


# ── sabotage checks (the 2026-09-22 lesson) ───────────────────────────────────


def test_sabotage_conv_reset_left_on_under_causal_is_not_caught_by_the_attention_probe(
        monkeypatch):
    """Bug: `tg_strict_prelude='causal'` but the prelude's conv/retention still reset at
    every segment (as 'span' does) — i.e. only `tg_allow`/`tg_comp_allow` widened, not
    `tg_seg`/the reset mask. `test_causal_prelude_token_reads_two_spans_back...` above
    perturbs a token TWO SPANS back: the attention relation alone already carries that
    edit (it is unmasked for any distance under causal), so this bug does NOT change
    that test's reading — reported here as a finding, not hidden.
    """
    real = MORPHTransformer._tul_tg_kwargs

    def fake(self, layout):
        front_kw, front_reset, coda_kw, coda_reset = real(self, layout)
        if self.cfg.tul.tg_strict_prelude == "causal" and front_kw is not None:
            front_kw = dict(front_kw)
            seg = tg_segment_ids(layout)
            front_kw["tg_seg"] = seg
            front_reset = tg_reset_from_ids(seg)
        return front_kw, front_reset, coda_kw, coda_reset
    monkeypatch.setattr(MORPHTransformer, "_tul_tg_kwargs", fake)

    ids, _inp, _lab, layout = _pack()
    s, back = 4, 2
    edited = _edit(ids, layout, 0, s - back)
    m = _model(tg_strict_prelude="causal")
    xa, ha, lay = _prelude_and_slots(m, ids)
    xb, hb, _ = _prelude_and_slots(m, edited)
    tok_s = (~lay.slot_mask[0]) & (lay.bag_id[0] == s)
    d_tok = float((xa[0, tok_s] - xb[0, tok_s]).abs().max())
    assert d_tok > 0.0, (
        "FINDING CONFIRMED: the conv-reset sabotage did not hide the 2-spans-back "
        "attention perturbation — a bulk prelude-output check cannot see this bug")


def test_sabotage_conv_reset_left_on_under_causal_is_caught_at_a_slot_cell(monkeypatch):
    """The dedicated check that DOES catch the sabotage above.

    A TOKEN readout cannot isolate this bug (shown above: attention dominates at any
    distance for a token query under causal). A SLOT CELL's own relation, by contrast,
    is UNCHANGED at every value of `tg_strict_prelude` (same span only — proven
    structurally in `test_causal_widens_token_rows_and_leaves_slot_rows_untouched`), so
    the ONLY way an edit in the PREVIOUS span can reach the next span's slot cell is
    through the conv contaminating that span's own first token's KEY features across
    the segment boundary — exactly the route this sabotage reopens incorrectly (by
    leaving the cut in place where "causal" must remove it). `n_prelude=1` removes the
    composition confound (no earlier layer to have already mixed anything in).
    """
    real = MORPHTransformer._tul_tg_kwargs

    def fake(self, layout):
        front_kw, front_reset, coda_kw, coda_reset = real(self, layout)
        if self.cfg.tul.tg_strict_prelude == "causal" and front_kw is not None:
            front_kw = dict(front_kw)
            seg = tg_segment_ids(layout)
            front_kw["tg_seg"] = seg
            front_reset = tg_reset_from_ids(seg)
        return front_kw, front_reset, coda_kw, coda_reset

    ids, _inp, _lab, layout = _pack()
    bag, sm = layout.bag_id[0].numpy(), layout.slot_mask[0].numpy()
    k = 2
    k_last = int(np.flatnonzero((bag == k) & ~sm)[-1])
    out = ids.copy()
    raw = int((~sm[:k_last + 1]).sum()) - 1
    # Swap the boundary token (a DOT) for the OTHER boundary char (11): changes the
    # embedding without moving the cut (the trick
    # `test_the_bigram_cannot_see_the_previous_span_because_a_cell_sits_between` uses).
    assert out[0, raw] == 10, "fixture: expected span k to end on a DOT boundary"
    out[0, raw] = 11
    edited = out

    m = _model1(tg_strict_prelude="causal")
    seed_idx = layout.slot_index[0, k + 1]           # the NEXT span's own slot cell

    good_a, _h1, _l1 = _prelude_and_slots(m, ids)
    good_b, _h2, _l2 = _prelude_and_slots(m, edited)
    d_good = float((good_a[0, seed_idx] - good_b[0, seed_idx]).abs().max())
    assert d_good > 0.0, (
        "fixture/correct-code sanity: the correct (unsegmented) causal conv must mix "
        "span k's boundary token into span k+1's first token's key features, which "
        "slot k+1's cell then reads through its own (unchanged) same-span relation")

    monkeypatch.setattr(MORPHTransformer, "_tul_tg_kwargs", fake)
    bad_a, _h3, _l3 = _prelude_and_slots(m, ids)
    bad_b, _h4, _l4 = _prelude_and_slots(m, edited)
    d_bad = float((bad_a[0, seed_idx] - bad_b[0, seed_idx]).abs().max())
    assert d_bad == 0.0, (
        "sabotage not caught: leaving the prelude conv segment-reset ON under causal "
        "should cut the key-feature contamination this check targets")


def test_sabotage_coda_conv_crossing_a_segment_is_still_caught():
    """Re-run of `test_tul_strict_geometry.py`'s sabotage #4 on a causal-prelude model.

    `coda_token_input="embed"` is required here: with the default "prelude" the coda's
    own TOKEN carrier already inherits the (now causal) prelude output directly, which
    is condition 3's documented OPEN bypass, not a defect in the CODA's own partition —
    using it as the base leak test would wrongly blame the coda for a bypass this file
    already names elsewhere. With "embed" the loop is the only remaining route, so the
    base `_leak` test (loop zeroed -> exactly 0 outside the edited span) applies exactly
    as it does on a plain strict model, and this confirms the CODA's conv/value-shift
    partition (tg_segment_ids, shared code with the prelude) was not touched by this
    change.
    """
    from test_tul_strict_geometry import _leak

    real = transformer_mod.tg_segment_ids

    ids, *_ = _pack()
    m = _model(tg_strict_prelude="causal", coda_token_input="embed")
    out_d, own_d = _leak(m, ids)
    assert own_d > 0 and out_d == 0.0, "the causal-prelude + embed model fails its own leak test"

    def fake(layout):
        return torch.zeros_like(layout.bag_id)
    transformer_mod.tg_segment_ids = fake
    try:
        out_d2, own_d2 = _leak(
            _model(tg_strict_prelude="causal", coda_token_input="embed"), ids)
    finally:
        transformer_mod.tg_segment_ids = real
    assert own_d2 > 0, "the sabotage made the fixture inert, so it proves nothing"
    assert out_d2 > 0.0, (
        "sabotage not caught: dropping the segment partition (shared code — this zeros "
        "it in the coda AND the prelude) should reopen a cross-span leak")
