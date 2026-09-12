"""`tul.tg_geometry="strict"` — THE LEAK TEST for "the slot loop is the only channel".

`tg_restrict`'s relation is ``causal AND (same span OR j is ANY slot cell)``, in the
prelude AND in the coda. That "OR any slot cell" is a cross-span channel the LOOP never
touches: in the prelude every token and every cell may read every earlier cell, so a
cell's SEED — a bag-mean of its own span's token embeddings — reaches later spans with no
pass of the loop in between. Measured on `slot-spandec-mask` at 5,000 steps
(`lab/experiments/planned/2026-09-11-arc-span-decoder.md`, Results part 1): the whole
slot channel is worth 0.182 nats and the loop's own prefix write 0.078.

`strict` cuts every cross-span route that is not the loop. This file is the gate the
arms launch behind, and every assertion is TWO-SIDED: with the loop's write zeroed the
strict model must show EXACTLY zero cross-span influence, and the same probe on the
`restrict` model — run through the same code — must show a NONZERO one. A one-sided test
would pass just as happily on a model that reads nothing at all.

THE SABOTAGE TABLE (`test_sabotage_*`). Each case re-opens ONE cut route and asserts the
leak test FAILS. All five are caught:

  1. prelude WINDOW branch widened to the restrict relation  (token -> any earlier cell)
  2. prelude COMPRESSED branch unmasked                       (the same route, other branch)
  3. coda CELL query widened to the restrict relation         (cell -> its own span)
  4. CCA conv / value-shift segment reset dropped             (`tg_seg`)
  5. coda per-layer injections at the cells restored          (`slot_cell_inject_keep`)

The hash bigram is the sixth route the 2026-09-11 budget arms had to cut, and on a TUL
row it is cut BY CONSTRUCTION rather than by a mask: a span's first token is always
preceded by the previous slot's prefix cells, whose input id is the constant `slot_id`.
`test_the_bigram_cannot_see_the_previous_span_because_a_cell_sits_between` proves that
two-sided instead of sabotaging a cut that does not exist.

CPU only, fp32, `use_kernels=False`, tiny config, no tokenizer — the
`tests/test_tg_restrict.py` fixtures. fp32 and not fp64 deliberately: `_window_fallback`
hands SDPA an fp32 mask whatever the dtype of q, which is silently WRONG at fp64
(`morph/model/CLAUDE.md`). The quantity under test is an EXACT zero, which fp32 carries
perfectly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.transformer as transformer_mod
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (
    BoundaryRule,
    SlotLayout,
    TulLayoutSpec,
    slot_layout_from_ids,
    tg_allow_mask,
    tg_segment_ids,
    tg_strict_allow,
)

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5                      # slot_id must not occur in the stream
    ids[:, ::8] = DOT                      # a boundary every 8 tokens
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4,
                tg_restrict=True, tg_restrict_scope="all", emit_weight=0.0,
                token_state_dropout=0.0, mux_beta=0.0)
    base.update(kw)
    return TULConfig(**base)


def _model(seed: int = 1234, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    # `BigramEmbedding.lambdas` is ZERO-init, so on a fresh model the bigram route is
    # switched off and an id-perturbation test would pass without exercising it at all
    # (the `test_span_mask_leak.py` lesson).
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _pack(B: int = 2, seed: int = 0):
    spec, rule = _spec(), _rule()
    ids = _ids(B=B, seed=seed)
    inp, lab, layout, _stats = slot_layout_from_ids(ids, rule, spec)
    return ids, inp, lab, layout


def _edit(ids: np.ndarray, layout: SlotLayout, row: int, span: int) -> np.ndarray:
    """Change one NON-boundary token of ``span`` in ``row`` without moving the cut."""
    bag = layout.bag_id[row].numpy()
    tok = (~layout.slot_mask[row].numpy())
    pos = np.flatnonzero((bag == span) & tok)
    assert pos.size >= 3, f"span {span} of row {row} is too short to edit safely"
    # the middle of the span: never the boundary token (which would move the cut)
    p = int(pos[len(pos) // 2])
    out = ids.copy()
    # positions in `ids` (the raw buffer) and in the packed row differ; the packer
    # consumes tokens in order, so raw index = the count of token positions before p.
    raw = int(tok[:p].sum())
    assert out[row, raw] != DOT and out[row, raw] != 11
    out[row, raw] = 12 if out[row, raw] != 12 else 13
    return out


def _logits(m: MORPHTransformer, ids: np.ndarray, plan_mode: str):
    spec, rule = _spec(), _rule()
    inp, lab, layout, _ = slot_layout_from_ids(ids, rule, spec)
    with torch.no_grad():
        if plan_mode == "normal":
            out = m(inp, labels=None, slot_layout=layout)
        else:
            out = m.tul_forward_ablated(inp, None, layout, plan_mode=plan_mode)
    return out["logits"], layout


# ── (a) the default is the shipped relation, and strict is NOT it ─────────────

def test_restrict_is_the_default_and_builds_no_new_parameter():
    m_def = _model()
    m_str = _model(tg_geometry="strict")
    assert m_def.cfg.tul.tg_geometry == "restrict" and not m_def._tg_strict
    assert m_str._tg_strict
    assert list(m_def.state_dict().keys()) == list(m_str.state_dict().keys()), \
        "strict must add no parameter — it is a mask, not a module"
    for a, b in zip(m_def.state_dict().values(), m_str.state_dict().values()):
        assert torch.equal(a, b), "strict must draw no RNG: the weights must be identical"


def test_restrict_logits_are_unchanged_and_strict_logits_are_not():
    """The regression guard: `tg_geometry` defaults to the relation the arms already ran.

    `restrict` is pinned by construction — the same `tg_allow_mask` call, which
    `tests/test_tg_restrict.py` already pins against a brute-force reference — so what
    this adds is the two-sided half: strict must actually CHANGE the forward, or the
    whole file could be passing on a knob that does nothing.
    """
    ids, inp, _lab, _lay = _pack()
    a, _ = _logits(_model(), ids, "normal")
    b, _ = _logits(_model(tg_geometry="restrict"), ids, "normal")
    assert torch.equal(a, b)
    c, _ = _logits(_model(tg_geometry="strict"), ids, "normal")
    assert not torch.equal(a, c), "strict changed nothing — the knob is inert"


# ── (b) THE LEAK TEST ────────────────────────────────────────────────────────

def _leak(m: MORPHTransformer, ids: np.ndarray, row: int = 0, span: int = 0,
          plan_mode: str = "zero") -> tuple[float, float]:
    """``(max |delta| outside the edited span, max |delta| inside it)``.

    With ``plan_mode="zero"`` the loop's prefix write is zeroed
    (``_tul_plan_ablate``), so the LOOP carries nothing. Under strict, nothing else
    crosses a span boundary either, and the outside-delta must be exactly 0.
    """
    _ids0, _inp, _lab, layout = _pack()
    edited = _edit(ids, layout, row, span)
    a, lay = _logits(m, ids, plan_mode)
    b, _ = _logits(m, edited, plan_mode)
    bag = lay.bag_id[row]
    tok = ~lay.slot_mask[row]
    own = tok & (bag == span)
    other = tok & (bag != span)
    # The slot_id vocab column is forced to -inf at every position (spec §3.1), and
    # -inf - (-inf) is NaN even where the two runs agree exactly. Compare by INEQUALITY
    # and report the delta over the finite columns.
    d = (a[row] - b[row]).abs().nan_to_num(0.0)
    d = torch.where(a[row] != b[row], d, torch.zeros_like(d))
    return float(d[other].max()), float(d[own].max())


def test_with_the_write_zeroed_no_earlier_span_can_move_a_later_one_under_strict():
    ids, *_ = _pack()
    out_d, own_d = _leak(_model(tg_geometry="strict"), ids)
    assert own_d > 0, "the edited token moved nothing at all — the fixture is inert"
    assert out_d == 0.0, (
        f"LEAK: with the loop's write zeroed, a token id in span 0 moved a later span's "
        f"logits by {out_d:.3e} under tg_geometry='strict'")


def test_the_restrict_model_leaks_through_the_same_probe():
    """The control. Same code, same ablation, `tg_geometry='restrict'`: it MUST leak."""
    ids, *_ = _pack()
    out_d, own_d = _leak(_model(), ids)
    assert own_d > 0
    assert out_d > 0, (
        "the restrict model hid an earlier span with its loop write zeroed — then the "
        "strict assertion above is vacuous and this file measures nothing")


def test_with_the_write_ON_the_loop_carries_the_edit_across_the_boundary():
    """The other side: strict must not be a model that simply reads nothing."""
    ids, *_ = _pack()
    out_d, own_d = _leak(_model(tg_geometry="strict"), ids, plan_mode="normal")
    assert own_d > 0
    assert out_d > 0, (
        "under strict with the loop's write ON, span 0's edit reached no later span — "
        "the slot loop is not connected to the coda at all")


@pytest.mark.parametrize("reach", ["all", "prev"])
def test_the_leak_test_holds_at_both_coda_reaches(reach):
    ids, *_ = _pack()
    out_d, own_d = _leak(_model(tg_geometry="strict", tg_coda_prefix_reach=reach), ids)
    assert own_d > 0 and out_d == 0.0, f"reach={reach}: outside delta {out_d:.3e}"


def test_zero_and_all_slots_are_the_same_ablation_under_strict():
    """Bit-forced, and it is the geometry's own check.

    `all_slots` cuts three routes; under strict two of them (the cell's per-layer
    injections and the cell's read of its own span) are already cut on EVERY forward, so
    the instrument reduces to `zero`. If these two ever differ, one of those cuts stopped
    firing.
    """
    ids, *_ = _pack()
    m = _model(tg_geometry="strict")
    a, _ = _logits(m, ids, "zero")
    b, _ = _logits(m, ids, "all_slots")
    assert torch.equal(a, b)
    # and on a RESTRICT model they must differ, or the comparison above is vacuous
    m2 = _model()
    c, _ = _logits(m2, ids, "zero")
    d, _ = _logits(m2, ids, "all_slots")
    assert not torch.equal(c, d)


# ── the route no mask covers: the hash bigram ────────────────────────────────

def test_the_bigram_cannot_see_the_previous_span_because_a_cell_sits_between():
    """A TUL row needs no bigram cut — the packer already put `slot_id` in between.

    `BigramEmbedding.compute` hashes (previous position's id, this position's id). In a
    packed TUL row a span's FIRST token is always preceded by the previous slot's
    `prefix_k` cells, whose input id is the constant `slot_id`, so the hash carries no
    information about the previous span. Two-sided: the same bigram at a MID-span
    position must still move when its own predecessor changes.
    """
    ids, inp, _lab, layout = _pack()
    bag, sm = layout.bag_id[0].numpy(), layout.slot_mask[0].numpy()
    first = int(np.flatnonzero((bag == 1) & ~sm)[0])
    assert sm[first - 1], "fixture: a span's first token must follow a slot cell"
    assert int(inp[0, first - 1]) == 4, "the cell's input id must be slot_id"

    # change the previous SPAN's last (boundary) token for another boundary id
    prev_last = int(np.flatnonzero((bag == 0) & ~sm)[-1])
    edited = inp.clone()
    assert int(edited[0, prev_last]) == DOT
    edited[0, prev_last] = 11
    with torch.no_grad():
        m = _model(tg_geometry="strict")
        a = m.embed.get_bigram(inp)[0, first]
        b = m.embed.get_bigram(edited)[0, first]
    assert torch.equal(a, b), (
        "LEAK: the bigram at a span's first token read the previous span — the packer's "
        "slot cell is supposed to stand between them")
    # two-sided: a mid-span position DOES read its predecessor
    mid = int(np.flatnonzero((bag == 1) & ~sm)[2])
    e2 = inp.clone()
    e2[0, mid - 1] = 12 if int(e2[0, mid - 1]) != 12 else 13
    with torch.no_grad():
        c = m.embed.get_bigram(inp)[0, mid]
        d = m.embed.get_bigram(e2)[0, mid]
    assert not torch.equal(c, d), "the bigram stopped reading its own predecessor"


# ── (d) THE SABOTAGE TABLE ───────────────────────────────────────────────────

def _assert_leak_test_fails(monkeypatch, patch_fn):
    ids, *_ = _pack()
    patch_fn(monkeypatch)
    m = _model(tg_geometry="strict")
    out_d, own_d = _leak(m, ids)
    assert own_d > 0, "the sabotage made the fixture inert, so it proves nothing"
    assert out_d > 0.0, (
        "the sabotage re-opened a route and the leak test still passed — the test does "
        "not cover that route")


def test_sabotage_1_prelude_window_widened_to_the_restrict_relation(monkeypatch):
    def patch(mp):
        real = tg_strict_allow

        def fake(layout, stage, coda_prefix_reach="all"):
            if stage == "prelude":
                return tg_allow_mask(layout)          # token -> any earlier cell
            return real(layout, stage, coda_prefix_reach=coda_prefix_reach)
        mp.setattr(transformer_mod, "tg_strict_allow", fake)
    _assert_leak_test_fails(monkeypatch, patch)


def test_sabotage_2_prelude_compressed_branch_unmasked(monkeypatch):
    """Only `tg_comp_allow` is dropped; the window branch stays strict.

    This is defect class F1 of the 2026-09-10 audit: one branch restricted, the other
    not, and the model looks masked.
    """
    def patch(mp):
        real = MORPHTransformer._tul_front

        def fake(self, input_ids, layout, attn_kwargs=None, ret_reset_mask=None):
            if attn_kwargs is not None:
                attn_kwargs = dict(attn_kwargs)
                attn_kwargs.pop("tg_comp_allow", None)
            return real(self, input_ids, layout, attn_kwargs=attn_kwargs,
                        ret_reset_mask=ret_reset_mask)
        mp.setattr(MORPHTransformer, "_tul_front", fake)
    _assert_leak_test_fails(monkeypatch, patch)


def test_sabotage_3_coda_cell_query_widened(monkeypatch):
    def patch(mp):
        real = tg_strict_allow

        def fake(layout, stage, coda_prefix_reach="all"):
            if stage == "coda":
                # the cell re-summarises its own span from the coda's token states
                return tg_allow_mask(layout)
            return real(layout, stage, coda_prefix_reach=coda_prefix_reach)
        mp.setattr(transformer_mod, "tg_strict_allow", fake)
    _assert_leak_test_fails(monkeypatch, patch)


def test_sabotage_4_conv_and_value_shift_reset_dropped(monkeypatch):
    def patch(mp):
        mp.setattr(transformer_mod, "tg_segment_ids",
                   lambda layout: torch.zeros_like(layout.bag_id))
    _assert_leak_test_fails(monkeypatch, patch)


def test_sabotage_5_coda_cell_injections_restored(monkeypatch):
    def patch(mp):
        mp.setattr(transformer_mod, "slot_cell_inject_keep",
                   lambda layout, dtype: torch.ones(*layout.slot_mask.shape, 1, dtype=dtype))
    _assert_leak_test_fails(monkeypatch, patch)


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(tg_restrict=False), "tg_restrict=true"),
    (dict(tg_restrict_scope="coda"), "tg_restrict_scope='all'"),
    (dict(tokens_through_core=True), "paid loop"),
    (dict(tg_soft_prev_span=True), "soft term"),
    (dict(tg_span_comp=True), "E-SAC"),
    (dict(coda_sees_slots=False), "GATHERED"),
])
def test_strict_refuses_what_it_has_no_relation_for(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(tg_geometry="strict", **kw)


def test_the_reach_knob_is_refused_at_restrict():
    with pytest.raises(ValueError, match="silently ignored"):
        _tul(tg_coda_prefix_reach="prev")


def test_an_unknown_geometry_raises():
    with pytest.raises(ValueError, match="tg_geometry"):
        _tul(tg_geometry="loose")


def test_the_segment_ids_separate_a_span_from_its_own_cells():
    """The conv / value shift / retention partition, asserted directly.

    `tg_reset_mask` reads `bag_id`, which gives a span's tokens and that span's cells the
    SAME id — so under strict the reset is driven by `tg_segment_ids` instead. Retention
    is off on every strict arm; this is the assertion that says what the carry WOULD be
    segmented by if one were turned on.
    """
    _ids0, _inp, _lab, layout = _pack()
    seg = tg_segment_ids(layout)
    bag, sm = layout.bag_id, layout.slot_mask
    # a span's tokens and its own cells share bag_id but never a segment
    same_bag = bag.unsqueeze(2) == bag.unsqueeze(1)
    diff_kind = sm.unsqueeze(2) != sm.unsqueeze(1)
    overlap = same_bag & diff_kind
    assert bool(overlap.any()), "fixture: a row must contain a span AND its cells"
    same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)
    assert not bool((overlap & same_seg).any()), \
        "tg_segment_ids put a span's tokens in the same segment as its own cells"


# ── the arms themselves: compose the Hydra config and BUILD the model ────────
#
# 1,149 green unit tests said nothing about whether an arm could START, because every
# test built its own tiny TULConfig by hand (the 2026-09-12 latent-z batch: a refusal in
# `TULConfig.__post_init__` blocked all three of its arms and no test saw it). So every
# config this change adds is composed through Hydra, mapped through the SHIPPED
# `build_tul_runtime` key mapping, and run.

_CONFIG_DIR = __import__("os").path.abspath("morph/configs")
STRICT_CONFIGS = ["tul_slot_spandec_strict", "tul_slot_spandec_strict_prev",
                  "tul_slot_spandec_strict_reach1",
                  "tul_slot_spandec_strict_oracle",
                  "tul_slot_spandec_strict_h3"]


class _StubTok:
    """Enough of a tokenizer for `build_tul_runtime`'s id resolution, no download."""

    @staticmethod
    def from_pretrained(_name):
        return _StubTok()

    def convert_tokens_to_ids(self, _tok):
        return 4


def _runtime(name: str, monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0,
                                                   ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name=name)
    return cfg, tul_setup.build_tul_runtime(cfg)


@pytest.mark.parametrize("name", STRICT_CONFIGS)
def test_the_strict_arms_compose_and_build_and_run(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    assert rt is not None, f"{name}: tul is off — the arm would run the plain model"
    tc = rt.model_cfg
    assert tc.tg_geometry == "strict" and tc.tg_restrict and tc.tg_restrict_scope == "all"
    assert tc.spandec, f"{name} must inherit the span-decoder target"
    assert tc.tg_coda_prefix_reach == ("prev" if name.endswith("_prev") else "all")
    assert tc.loop_reach == (1 if name.endswith("_reach1") else 0)
    assert tc.oracle_z == name.endswith("_oracle")
    assert tc.spandec_horizon == (3 if name.endswith("_h3") else 1)
    assert not bool(cfg.model.use_kernels), "tg_restrict forces model.use_kernels: false"

    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc)).eval().float()
    _ids0, inp, lab, layout = _pack()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]), f"{name}: loss is not finite"


# ── DEPTH AS REACH (tul.loop_reach) ──────────────────────────────────────────
#
# Inside `_tul_core` the compact sequence updates every cell in parallel per pass
# (Jacobi), so at reach w a slot m spans back needs ceil(m/w) passes to reach slot k. The
# acceptance test is that recurrence, read on the loop's own states: bit-exact zero
# movement before pass ceil(m/w), nonzero at it. Every cross-cell route inside the core
# has to obey it or the count is wrong, which is what the sabotages check.


def _h_slots(m: MORPHTransformer, ids: np.ndarray):
    """The loop's exit states `[B, S, ...]` for one batch, eval, no grad."""
    spec, rule = _spec(), _rule()
    inp, _lab, layout, _ = slot_layout_from_ids(ids, rule, spec)
    with torch.no_grad():
        x, x0, bigram = m._tul_front(*_front_args(m, inp, layout))
        out = m._tul_core(x, x0, bigram, layout, input_ids=inp)
    return out[1], layout


def _front_args(m: MORPHTransformer, inp, layout):
    """`_tul_front`'s arguments WITH the strict prelude masks `_forward_tul` builds."""
    from morph.model.tul_layout import tg_reset_from_ids
    seg = tg_segment_ids(layout)
    pre = tg_strict_allow(layout, "prelude")
    kw = {"tg_allow": pre, "tg_slot_mask": layout.slot_mask, "tg_comp_allow": pre,
          "tg_seg": seg}
    return inp, layout, kw, tg_reset_from_ids(seg)


def _reach_model(reach: int, depth: int, seed: int = 1234) -> MORPHTransformer:
    return _model(seed=seed, tg_geometry="strict", loop_reach=reach,
                  slot_depth_fixed=depth, slot_max_depth=8)


@pytest.mark.parametrize("reach,m_back,first_pass", [(1, 2, 2), (1, 3, 3), (2, 3, 2)])
def test_a_slot_state_moves_first_at_pass_ceil_m_over_w(reach, m_back, first_pass):
    """Perturb slot ``k - m``'s seed; slot ``k``'s state must move at pass ceil(m/w).

    The seed is perturbed through the DATA — one token id inside span ``k - m`` — because
    that is the perturbation the arm actually experiences. Under the strict prelude that
    edit reaches exactly one slot's seed, which
    `test_the_edit_reaches_exactly_one_slot_seed` asserts separately, so the pass count
    read here is the LOOP's and not the prelude's.
    """
    ids, _inp, _lab, layout = _pack()
    k = 4
    edited = _edit(ids, layout, 0, k - m_back)
    for passes in range(1, first_pass + 1):
        mdl = _reach_model(reach, passes)
        a, lay = _h_slots(mdl, ids)
        b, _ = _h_slots(mdl, edited)
        d = float((a[0, k] - b[0, k]).abs().max())
        if passes < first_pass:
            assert d == 0.0, (
                f"reach {reach}: slot {k} moved at pass {passes} from a perturbation "
                f"{m_back} slots back — expected first movement at pass {first_pass}, "
                f"delta {d:.3e}")
        else:
            assert d > 0.0, (
                f"reach {reach}: slot {k} did NOT move at pass {first_pass} from a "
                f"perturbation {m_back} slots back — the loop carries nothing, so the "
                f"zeros above are vacuous")


def test_the_edit_reaches_exactly_one_slot_seed():
    """The fixture's own gate: under strict, editing span j moves slot j's seed and no other.

    Without this, every zero in the reach test could be a prelude that carries nothing.
    """
    ids, _inp, _lab, layout = _pack()
    j = 2
    edited = _edit(ids, layout, 0, j)
    mdl = _reach_model(1, 1)
    spec, rule = _spec(), _rule()
    moved = []
    for arr in (ids, edited):
        inp, _l, lay, _ = slot_layout_from_ids(arr, rule, spec)
        with torch.no_grad():
            x, _x0, _bg = mdl._tul_front(*_front_args(mdl, inp, lay))
        moved.append(mdl.input_norm(x))
    d = (moved[0] - moved[1]).abs().flatten(2).max(-1).values[0]     # [L]
    seeds = layout.slot_index[0]
    hit = [int(s) for s in range(layout.max_slots)
           if bool(layout.slot_valid[0, s]) and float(d[seeds[s]].detach()) > 0]
    assert hit == [j], f"the edit moved slot seeds {hit}, expected exactly [{j}]"


def test_unlimited_reach_moves_a_far_slot_in_one_pass():
    """The control for the whole family: at loop_reach 0 one pass crosses any distance."""
    ids, _inp, _lab, layout = _pack()
    mdl = _model(tg_geometry="strict", slot_depth_fixed=1, slot_max_depth=8)
    edited = _edit(ids, layout, 0, 1)
    a, _ = _h_slots(mdl, ids)
    b, _ = _h_slots(mdl, edited)
    assert float((a[0, 4] - b[0, 4]).abs().max()) > 0.0, \
        "at unlimited reach a single pass did not carry slot 1 to slot 4"


def _drop(attn_kw, key: str):
    """Remove one mask from the core's per-layer kwargs (`tul.loop_reach` hands a tuple)."""
    if attn_kw is None:
        return None
    if isinstance(attn_kw, (list, tuple)):
        return tuple({k: v for k, v in d.items() if k != key} for d in attn_kw)
    return {k: v for k, v in attn_kw.items() if k != key}


def test_sabotage_6_core_conv_reset_dropped_breaks_the_reach_count(monkeypatch):
    """Drop the per-cell `tg_seg` in the core: the conv alone reaches ~6 cells per block."""
    real = MORPHTransformer._apply_core_step

    def fake(self, *a, attn_kw=None, **kw):
        attn_kw = _drop(attn_kw, "tg_seg")
        return real(self, *a, attn_kw=attn_kw, **kw)
    monkeypatch.setattr(MORPHTransformer, "_apply_core_step", fake)
    ids, _inp, _lab, layout = _pack()
    edited = _edit(ids, layout, 0, 2)
    mdl = _reach_model(1, 1)
    a, _ = _h_slots(mdl, ids)
    b, _ = _h_slots(mdl, edited)
    assert float((a[0, 4] - b[0, 4]).abs().max()) > 0.0, \
        "dropping the core conv reset changed nothing — the reach test does not cover it"


def test_sabotage_7_core_compressed_branch_unmasked_breaks_the_reach_count(monkeypatch):
    """Drop `tg_comp_allow` in the core: the compressed branch is dense causal over cells."""
    real = MORPHTransformer._apply_core_step

    def fake(self, *a, attn_kw=None, **kw):
        attn_kw = _drop(attn_kw, "tg_comp_allow")
        return real(self, *a, attn_kw=attn_kw, **kw)
    monkeypatch.setattr(MORPHTransformer, "_apply_core_step", fake)
    ids, _inp, _lab, layout = _pack()
    edited = _edit(ids, layout, 0, 2)
    mdl = _reach_model(1, 1)
    a, _ = _h_slots(mdl, ids)
    b, _ = _h_slots(mdl, edited)
    assert float((a[0, 4] - b[0, 4]).abs().max()) > 0.0, \
        "dropping the core compressed mask changed nothing — the branch is not a route"


def test_sabotage_8_core_window_branch_unmasked_breaks_the_reach_count(monkeypatch):
    real = MORPHTransformer._apply_core_step

    def fake(self, *a, attn_kw=None, **kw):
        attn_kw = _drop(attn_kw, "tg_allow")
        return real(self, *a, attn_kw=attn_kw, **kw)
    monkeypatch.setattr(MORPHTransformer, "_apply_core_step", fake)
    ids, _inp, _lab, layout = _pack()
    edited = _edit(ids, layout, 0, 2)
    mdl = _reach_model(1, 1)
    a, _ = _h_slots(mdl, ids)
    b, _ = _h_slots(mdl, edited)
    assert float((a[0, 4] - b[0, 4]).abs().max()) > 0.0, \
        "dropping the core window mask changed nothing — the branch is not a route"


def test_the_core_has_no_pooled_compressor_under_the_restriction():
    """Audit finding F1 cannot occur on a strict arm, and here is why, not a claim.

    F1 is the HCA pooled compressor returning `[B, 0, c]` when the ratio does not divide
    the sequence. Under `tg_restrict` that compressor is NOT BUILT on any layer — the
    compressed branch is `_tg_slot_attention`, which at the compact shape is dense causal
    over the cells and therefore a REAL cross-cell route that `loop_reach` must mask
    (sabotage 7 proves it is).
    """
    mdl = _reach_model(1, 2)
    for blk in list(mdl.core) + list(mdl.prelude) + list(mdl.coda):
        assert blk.attention._impl.compressor is None


@pytest.mark.parametrize("kw,match", [
    (dict(loop_reach=1, tg_geometry="restrict"), "tg_geometry='strict'"),
    (dict(loop_reach=1, tg_geometry="strict", slot_chain=True), "slot_chain"),
    (dict(loop_reach=-1), "loop_reach must be"),
])
def test_loop_reach_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)


@pytest.mark.parametrize("reach,moves", [(1, False), (2, True)])
def test_reach_prev_plus_loop_reach_needs_depth_to_carry_three_spans(reach, moves):
    """The two restrictions composed, read at the LOGITS — item 1's test (c).

    Coda reach "prev" means a token of span k reads only slot k-1's cells. With the loop
    run for ONE pass at `loop_reach` w, slot k-1 holds slots k-1-w .. k-1. So an edit in
    span k-3 reaches span k at w = 2 and NOT at w = 1 — the only way to carry it at w = 1
    is a second pass, which is the arm's whole claim.

    Span k-1's own tokens DO move at w = 1 (they read slot k-2, which holds slot k-3), so
    the assertion is on span k's token positions alone.
    """
    ids, _inp, _lab, layout = _pack()
    k = 4
    mdl = _model(tg_geometry="strict", tg_coda_prefix_reach="prev", loop_reach=reach,
                 slot_depth_fixed=1, slot_max_depth=8)
    edited = _edit(ids, layout, 0, k - 3)
    a, lay = _logits(mdl, ids, "normal")
    b, _ = _logits(mdl, edited, "normal")
    tgt = (~lay.slot_mask[0]) & (lay.bag_id[0] == k)
    assert bool(tgt.any()), "fixture: span k must hold token positions"
    d = (a[0] - b[0]).abs().nan_to_num(0.0)
    d = torch.where(a[0] != b[0], d, torch.zeros_like(d))
    got = float(d[tgt].max())
    if moves:
        assert got > 0.0, (
            f"reach {reach}: one pass did NOT carry an edit three spans back to span {k} "
            "— then the reach-1 zero below is vacuous")
    else:
        assert got == 0.0, (
            f"reach {reach}: one pass carried an edit three spans back to span {k} "
            f"(delta {got:.3e}) — depth is not required after all")
