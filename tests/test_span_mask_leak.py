"""`model.span_mask` — THE LEAK TEST for the cross-span information budget.

The pair `budget-web-full` / `budget-web-span`
(`lab/experiments/planned/2026-09-11-arc-span-budget.md`) reads the CE gap between two
models that differ only in whether anything may cross a span boundary. That number is
worth nothing if one route still crosses, so this file is the gate the arms are launched
behind, and every assertion here is two-sided: the restricted model must show EXACTLY
zero cross-span influence, and the `"row"` control run through the SAME code must show a
NONZERO one. A one-sided test would pass just as happily on a mask that is applied to
both arms, or to neither.

Three instruments, because they fail differently:

* `test_no_id_from_an_earlier_span_can_move_a_later_span`: the end-to-end one. Change a
  token id inside span 0 and every logit in spans >= 1 must be bit-identical. This
  catches EVERY route at once — attention, the CCA conv, the value shift, the hash
  bigram, x0 / value-embed injections, the Parcae core — and needs no hooks.
* `test_cross_span_jacobian_is_zero_at_every_region`: autograd through the embedding,
  read at the prelude exit, the core exit, the coda exit and the logits, so a leak is
  located rather than merely detected.
* `test_bigram_ignores_the_previous_span`: the one route the embedding Jacobian cannot
  see, because the bigram reads token IDS and not the embedding output.

CPU only, fp32, `use_kernels=False`, tiny config, no tokenizer — the
`tests/test_tul_forward.py` fixtures. fp32 and not fp64 deliberately:
`_window_fallback` hands `F.scaled_dot_product_attention` an fp32 mask whatever the
dtype of q, which is silently WRONG at fp64 (`morph/model/CLAUDE.md`). The quantity
under test is an EXACT zero, which fp32 carries perfectly — a masked softmax column has
weight exactly 0 and contributes gradient exactly 0.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul_layout import (BoundaryRule, span_allow_mask, span_ids_from_ids,
                                    span_reach_allow)

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0, core_impl="parcae",
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _ids(B: int = 2, S: int = 32, seed: int = 0) -> np.ndarray:
    """A batch whose rows carry at least four spans under `_rule`."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(12, V, size=(B, S))          # 12.. : never a boundary id
    ids[:, ::6] = DOT                               # a boundary every 6 tokens
    return ids.astype(np.int64)


def _model(mode: str, seed: int = 1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(span_mask=mode,
                               span_rule=_rule() if mode == "span" else None, **cfg_kw))
    # `BigramEmbedding.lambdas` is ZERO-init, so on a fresh model the bigram route is
    # switched off and an id-perturbation test would pass without ever exercising it.
    # A trained model has non-zero lambdas; give the fixture some, or this file quietly
    # stops testing the one route the attention masks do not cover (verified: with the
    # bigram cut removed and lambdas at 0, every assertion below still passes).
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


# ── model.span_reach: the relation builder ────────────────────────────────────
#
# `span_reach_allow` is the ONE thing `model.span_reach` changes (`_span_context`'s
# ``allow`` tensor); `tg_seg` and the bigram ``cut`` stay `span_id` / `span_start_mask`
# at every reach, unchanged from plain `span_mask: "span"`. These tests cover the pure
# relation (no model) and the attention route at reach 1, isolated to ONE layer so a
# multi-layer forward's relay cannot be mistaken for the relation itself — see
# `test_reach1_relation_is_reach1_at_a_single_attention_layer`'s docstring for why an
# END-TO-END logit check is the wrong instrument for "two spans back must not move".

def _expected_reach_allow(span: list[int], reach: int) -> torch.Tensor:
    """Hand-built ``[S, S]`` bool expected relation, independent of `span_reach_allow`."""
    S = len(span)
    out = torch.zeros(S, S, dtype=torch.bool)
    for i in range(S):
        for j in range(S):
            if j > i:
                continue
            out[i, j] = True if reach == -1 else span[j] >= span[i] - reach
    return out


def test_span_reach_allow_matches_span_allow_mask_at_reach_zero():
    ids = _ids()
    span_id = torch.from_numpy(span_ids_from_ids(ids, _rule()))
    assert torch.equal(span_reach_allow(span_id, 0), span_allow_mask(span_id))


def test_span_reach_allow_synthetic_row():
    """A hand-picked row with 6 spans; every reach checked against a hand-built matrix,
    and every mask asserted causal (``allow(i, j)`` False for ``j > i``)."""
    span = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    span_id = torch.tensor([span], dtype=torch.int64)
    S = len(span)
    row = torch.arange(S).unsqueeze(1)
    col = torch.arange(S).unsqueeze(0)
    for reach in (0, 1, 2, -1):
        got = span_reach_allow(span_id, reach)[0, 0]
        expected = _expected_reach_allow(span, reach)
        assert torch.equal(got, expected), f"reach={reach} mismatch"
        assert not (got & (col > row)).any(), f"reach={reach} is not causal"

    # reach 1 from span 3's last position (index 9): own span and span 2, not 0/1.
    q = 9
    seen = {span[j] for j in range(S) if span_reach_allow(span_id, 1)[0, 0, q, j]}
    assert seen == {2, 3}
    # reach 2: adds span 1.
    seen2 = {span[j] for j in range(S) if span_reach_allow(span_id, 2)[0, 0, q, j]}
    assert seen2 == {1, 2, 3}
    # reach -1: plain causal, every earlier span reachable.
    seen_all = {span[j] for j in range(S) if span_reach_allow(span_id, -1)[0, 0, q, j]}
    assert seen_all == {0, 1, 2, 3}


def test_span_reach_zero_is_bit_identical_to_the_field_absent():
    """``span_reach=0`` explicit vs the field left at its dataclass default: same seed,
    same architecture, same forward — the two models must be bit-identical, and a
    forward on `span_mask: "span"` at `span_reach=0` must match `span_allow_mask`'s own
    arm exactly (same call, `_span_context` routes both through `span_reach_allow`)."""
    ids = torch.from_numpy(_ids())
    m_default = _model("span")                       # span_reach: MORPHConfig default (0)
    m_explicit = _model("span", span_reach=0)
    assert m_default.cfg.span_reach == 0 and m_explicit.cfg.span_reach == 0
    with torch.no_grad():
        a = m_default(ids)["logits"]
        b = m_explicit(ids)["logits"]
    assert torch.equal(a, b)


def test_reach1_relation_is_reach1_at_a_single_attention_layer():
    """`model.span_reach=1` widens the PER-LAYER relation by exactly one span; it does
    NOT bound the model's end-to-end receptive field, which grows with depth like any
    stack of locally-restricted attention layers does (measured on this fixture's
    prelude+core+coda model, ~6+ sequential attention layers: a perturbation TWO spans
    back moves span-2+ END-TO-END logits by ~0.87 — real relay through the intervening
    span's updated representation after layer 1, not a leak in the mask; see the note
    `hop-staircase-on-prev-reach1` for the same phenomenon elsewhere in this tree). An
    end-to-end logit check is therefore the WRONG instrument for "two spans back must
    not move" at reach >= 1 — it is right only at reach 0, where every layer's relation
    already forbids ANY cross-span read (`test_no_id_from_an_earlier_span_can_move_a_
    later_span`).

    This test isolates the ONE relation `span_reach_allow` actually builds: the same
    embedding-Jacobian instrument `test_cross_span_jacobian_is_zero_at_every_region`
    uses, but read at `prelude[0]` — the FIRST block, one hop from the raw embedding —
    instead of `prelude[-1]`, so no second attention layer can compose a relay.
    """
    ids = _ids()
    span = span_ids_from_ids(ids, _rule())
    assert span.max() >= 3, "fixture must carry at least 4 spans per row"

    m = _model("span", span_reach=1)
    B, S = ids.shape
    eps = torch.zeros(B, S, m.cfg.d_model, requires_grad=True)
    grabbed: dict[str, torch.Tensor] = {}
    handles = [
        m.embed.hybrid.register_forward_hook(lambda _m, _i, o: o + eps),
        m.prelude[0].register_forward_hook(lambda _m, _i, o: grabbed.__setitem__("p0", o)),
    ]
    try:
        m(torch.from_numpy(ids))
    finally:
        for h in handles:
            h.remove()
    y = grabbed["p0"]

    queries = [int(np.flatnonzero(span[0] == k)[-1])
               for k in range(1, int(span[0].max()) + 1)
               if int((span[0] == k).sum()) > 1]
    assert len(queries) >= 2

    saw_two_back = False
    for i in queries:
        g, = torch.autograd.grad(y[0, i].sum(), eps, retain_graph=True)
        g = g[0].abs().sum(-1)
        own_span = int(span[0][i])
        one_back = torch.from_numpy(span[0] == own_span - 1)
        two_back = torch.from_numpy(span[0] <= own_span - 2)
        if one_back.any():
            assert g[one_back].max() > 0, (
                f"query span {own_span}: reach=1 must read its ONE previous span "
                "directly (one attention layer, no relay needed)")
        if two_back.any():
            saw_two_back = True
            assert g[two_back].max().item() == 0.0, (
                f"query span {own_span}: reach=1's own relation let it read a span "
                "TWO OR MORE back at a SINGLE attention layer — max |d out/d embed| = "
                f"{g[two_back].max().item():.3e}")
    assert saw_two_back, "fixture needs at least one query with 2+ earlier spans"


# ── the end-to-end instrument ────────────────────────────────────────────────

@pytest.mark.parametrize("site", ["interior", "boundary_token"])
@pytest.mark.parametrize("mode,leaks", [("span", False), ("row", True)])
def test_no_id_from_an_earlier_span_can_move_a_later_span(mode, leaks, site):
    """Edit one token of span 0; every logit in spans >= 1 must not move (mode 'span').

    Two edit sites, because they leak through different routes. ``interior`` is an
    ordinary token in the middle of span 0 and only attention (and the conv's k-1 taps)
    can carry it forward. ``boundary_token`` is span 0's LAST token, which is also the
    token the hash bigram of span 1's FIRST token reads — the one route no amount of
    attention masking touches, and exactly the datum a slot seed would carry.
    """
    ids = _ids()
    span = span_ids_from_ids(ids, _rule())
    assert span.max() >= 3, "fixture must carry at least 4 spans per row"

    # The cut must not move, so a non-boundary id is swapped for another non-boundary id
    # and the boundary token for another BOUNDARY id. Asserted below, not assumed.
    edited = ids.copy()
    if site == "interior":
        pos = int(np.flatnonzero(span[0] == 0)[1])
        edited[0, pos] = 12 if ids[0, pos] != 12 else 13
    else:
        pos = int(np.flatnonzero(span[0] == 0)[-1])
        edited[0, pos] = 11 if ids[0, pos] != 11 else DOT
    assert span[0][pos] == 0 and pos > 0
    assert np.array_equal(span_ids_from_ids(edited, _rule()), span), \
        "the edit moved the cut — the fixture, not the model, is wrong"

    m = _model(mode)
    with torch.no_grad():
        a = m(torch.from_numpy(ids))["logits"]
        b = m(torch.from_numpy(edited))["logits"]
    later = torch.from_numpy(span[0] > 0)
    moved = (a[0][later] != b[0][later]).any().item()
    if leaks:
        assert moved, ("model.span_mask='row' must NOT hide an earlier token: if this "
                       "passes, the mask is applied to the control too and the budget "
                       "pair measures nothing")
    else:
        assert not moved, (
            "LEAK: an id in span 0 moved logits in a later span. Max |delta| = "
            f"{(a[0][later] - b[0][later]).abs().max().item():.3e}")
        # and the edited span itself MUST move, or the model is simply ignoring its input
        own = torch.from_numpy(span[0] == 0)
        assert (a[0][own] != b[0][own]).any().item(), \
            "the edited token moved nothing at all — the fixture is inert"


# ── the located instrument ───────────────────────────────────────────────────

def _region_jacobian(m: MORPHTransformer, ids: np.ndarray):
    """Run one forward with a leaf perturbation on the embedding.

    Returns ``(eps, {region: tensor})`` where every tensor is ``[B, S, ...]`` and shares
    the graph, so ``autograd.grad(region[:, i].sum(), eps)`` is the region's Jacobian
    row at position ``i`` w.r.t. every input position.
    """
    B, S = ids.shape
    eps = torch.zeros(B, S, m.cfg.d_model, requires_grad=True)
    grabbed: dict[str, torch.Tensor] = {}
    handles = [m.embed.hybrid.register_forward_hook(lambda _m, _i, o: o + eps)]
    for name, mod in (("prelude", m.prelude[-1]), ("core", m.core[-1]),
                      ("coda", m.coda[-1])):
        def _keep(_m, _i, o, _n=name):
            grabbed[_n] = o          # last call wins: the core's final loop iteration
        handles.append(mod.register_forward_hook(_keep))
    try:
        grabbed["logits"] = m(torch.from_numpy(ids))["logits"]
    finally:
        for h in handles:
            h.remove()
    return eps, grabbed


@pytest.mark.parametrize("mode,leaks", [("span", False), ("row", True)])
def test_cross_span_jacobian_is_zero_at_every_region(mode, leaks):
    ids = _ids()
    span = span_ids_from_ids(ids, _rule())
    m = _model(mode)
    eps, regions = _region_jacobian(m, ids)

    # one query per span: its LAST position, which has the most same-span history.
    # Spans of length 1 (a row's trailing stub) are skipped — they have no own history
    # to prove the mask left alone, so they cannot carry the two-sided assertion.
    queries = [int(np.flatnonzero(span[0] == k)[-1])
               for k in range(1, int(span[0].max()) + 1)
               if int((span[0] == k).sum()) > 1]
    assert len(queries) >= 3

    for name, y in regions.items():
        for i in queries:
            g, = torch.autograd.grad(y[0, i].sum(), eps, retain_graph=True)
            g = g[0].abs().sum(-1)                       # [S] per input position
            assert g.sum() > 0, (
                f"{name}@{i}: the whole Jacobian row is zero, so the cross-span "
                "assertion below would pass vacuously")
            earlier = torch.from_numpy(span[0] < span[0][i])
            same_past = torch.from_numpy((span[0] == span[0][i]) & (np.arange(len(g)) < i))
            assert same_past.any(), "fixture: every probed query needs same-span history"
            if leaks:
                assert g[earlier].max() > 0, (
                    f"{name}@{i}: the 'row' control hid an earlier position — the mask "
                    "is being applied to the control")
            else:
                assert g[earlier].max().item() == 0.0, (
                    f"LEAK at region {name}, query position {i} (span {span[0][i]}): "
                    f"max |d out / d embed| over earlier spans = {g[earlier].max().item():.3e} "
                    f"at position {int(g[earlier].argmax())}")
                assert g[same_past].max() > 0, (
                    f"{name}@{i}: the query reads NOTHING from its own span's history — "
                    "the mask is too tight, not just leak-free")


@pytest.mark.parametrize("mode,leaks", [("span", False), ("row", True)])
def test_the_training_path_is_masked_too(mode, leaks):
    """train() mode, where the core loop SORTS the batch by its per-sample depth.

    `_core_region` permutes the carrier into depth order and processes only the active
    prefix each iteration, so it must permute the span relation the same way — a mask
    that stayed in batch order would put sample i under sample j's spans, and the arms
    would train on a relation nobody chose. Eval draws a uniform depth and never
    exercises that path, so this test is the one that covers the code the RUN uses.
    """
    ids = _ids(B=4, seed=3)                      # 4 rows -> distinct sampled depths
    span = span_ids_from_ids(ids, _rule())
    # bptt_depth == max_depth, as the budget arms run it (8 and 8). Below that,
    # `n_nograd = total_iters - bptt_depth` is GLOBAL, so every sample whose own depth
    # falls under it runs entirely no-grad and its embedding Jacobian is EXACTLY ZERO
    # (the E7 "global no-grad prefix silences 27-57 % of samples" defect). A zero row
    # would make the assertions below pass without testing anything, so the fixture
    # removes the condition and the row-sum check below refuses it if it comes back.
    m = _model(mode, bptt_depth=3)
    m.train()                                    # dropout is 0 in the fixture
    torch.manual_seed(7)
    eps, regions = _region_jacobian(m, ids)
    y = regions["logits"]
    seen = 0
    for b in range(ids.shape[0]):
        for k in range(1, int(span[b].max()) + 1):
            pos = np.flatnonzero(span[b] == k)
            if pos.size < 2:
                continue
            i = int(pos[-1])
            g, = torch.autograd.grad(y[b, i].sum(), eps, retain_graph=True)
            other = g.abs().sum(-1)
            own_row = other[b].clone()          # clone FIRST: `other[b] = 0` below
                                                # would otherwise zero this view too
            assert own_row.sum() > 0, (
                f"row {b} has a ZERO embedding Jacobian — the assertions below would "
                "pass vacuously (a no-grad prefix silenced the sample?)")
            other[b] = 0.0
            assert other.max().item() == 0.0, \
                f"row {b} position {i} read ANOTHER ROW of the batch"
            earlier = torch.from_numpy(span[b] < k)
            seen += 1
            if leaks:
                assert own_row[earlier].max() > 0
            else:
                assert own_row[earlier].max().item() == 0.0, (
                    f"LEAK in train() mode: row {b}, span {k}, position {i} reads "
                    f"{own_row[earlier].max().item():.3e} from an earlier span")
    assert seen >= 8, "fixture did not produce enough probed positions"


# ── the route autograd cannot see ────────────────────────────────────────────

@pytest.mark.parametrize("mode,leaks", [("span", False), ("row", True)])
def test_bigram_ignores_the_previous_span(mode, leaks):
    """The hash bigram reads IDS, so it needs an id perturbation, not a Jacobian."""
    ids = _ids()
    span = span_ids_from_ids(ids, _rule())
    m = _model(mode)
    cut = m._span_context(torch.from_numpy(ids))[2]
    assert cut is not None, "a span_mask model must always build the bigram cut mask"

    # the first position of span 1, and the boundary token of span 0 right before it
    first = int(np.flatnonzero(span[0] == 1)[0])
    assert first > 0 and span[0][first - 1] == 0
    assert bool(cut[0, first]) is (mode == "span"), \
        "span_start_mask must mark a span's first position under 'span' and not under 'row'"

    edited = ids.copy()
    edited[0, first - 1] = 11 if ids[0, first - 1] != 11 else DOT   # another boundary id
    with torch.no_grad():
        a = m.embed.get_bigram(torch.from_numpy(ids), cut)[0, first]
        b = m.embed.get_bigram(torch.from_numpy(edited), cut)[0, first]
    if leaks:
        assert not torch.equal(a, b), \
            "the 'row' control must still read the previous token into its bigram"
    else:
        assert torch.equal(a, b), (
            "LEAK: the bigram at a span's first position still hashes the previous "
            "span's boundary token — exactly the datum a slot seed carries")
    # the position's OWN id must still reach its bigram, in both modes
    own = ids.copy()
    own[0, first] = 12 if ids[0, first] != 12 else 13
    with torch.no_grad():
        c = m.embed.get_bigram(torch.from_numpy(own), cut)[0, first]
    assert not torch.equal(a, c), "the bigram stopped reading its own token"


# ── the refusals ─────────────────────────────────────────────────────────────

def test_span_mask_refuses_a_morph_core():
    with pytest.raises(NotImplementedError, match="core_impl='parcae'"):
        MORPHTransformer(_tiny(span_mask="row", core_impl="morph"))


def test_span_mask_refuses_the_fused_kernels():
    with pytest.raises(ValueError, match="use_kernels=false"):
        MORPHTransformer(_tiny(span_mask="row", use_kernels=True))


def test_span_mask_span_needs_a_rule_and_row_refuses_one():
    with pytest.raises(ValueError, match="span_rule"):
        MORPHTransformer(_tiny(span_mask="span", span_rule=None))
    with pytest.raises(ValueError, match="must not"):
        MORPHTransformer(_tiny(span_mask="row", span_rule=_rule()))


def test_span_mask_off_is_the_default_and_builds_the_pooled_compressor():
    m = MORPHTransformer(_tiny())
    assert m.cfg.span_mask == "off" and not m._span_mask
    assert m.prelude[0].attention._impl.compressor is not None
    assert m._span_context(torch.zeros(1, 4, dtype=torch.long)) == (None, None, None)


def test_span_reach_refuses_below_minus_one():
    with pytest.raises(ValueError, match="span_reach must be >= -1"):
        MORPHTransformer(_tiny(span_mask="span", span_rule=_rule(), span_reach=-2))


@pytest.mark.parametrize("mode,rule", [("row", None), ("off", None)])
def test_span_reach_nonzero_requires_span_mask_span(mode, rule):
    with pytest.raises(ValueError, match="span_reach != 0 requires model.span_mask='span'"):
        MORPHTransformer(_tiny(span_mask=mode, span_rule=rule, span_reach=1))
