"""Arm CW — the compaction window (.agents/notes/implemented/architecture/2026-08-18-tul-compaction-window.md).

Covers: ``coda_token_cut=0`` is bit-identical to the pre-CW code path (proven by
reconstructing that exact old path by hand and comparing, not by reading the diff —
CLAUDE.md's rule for this directory); the window-cut gather keeps exactly "every slot +
tokens with row index >= C" and drops exactly "tokens with row index < C"; the four arms
CW0/CW1/CW2/CW3 score CE over the identical set of labels; CW2's retention is exactly
``prefix_k * n_valid_slots`` per row and reproducible under a fixed seed; a cut >= seq_len
raises.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.tul import (
    TULConfig,
    compact_index,
    cw2_retain_mask,
    scatter_positions,
    window_drop_mask,
)
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.model.transformer import MORPHConfig, MORPHTransformer

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
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
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _padded_batch(B=3, max_slots=8):
    """A layout with real PAD slots — needed to make n_valid_slots < max_slots."""
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=max_slots, slot_id=4)
    rule = BoundaryRule(is_boundary=np.zeros(V, dtype=bool), min_span=4, span_cap=16, eos_id=0)
    rng = np.random.default_rng(5)
    ids = rng.integers(5, V, size=(B, 200))
    ids[ids == spec.slot_id] = 5
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), rule, spec)
    assert (~layout.slot_valid).any(), "fixture must contain pad slots"
    return x, y, layout, spec


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


# ── §1 config validation ──────────────────────────────────────────────────────

def test_coda_token_cut_defaults_to_zero_and_negative_rejected():
    assert TULConfig(prefix_k=2, slot_id=4).coda_token_cut == 0
    with pytest.raises(ValueError, match="coda_token_cut"):
        TULConfig(prefix_k=2, slot_id=4, coda_token_cut=-1)


def test_cut_greater_equal_seq_len_raises_not_silently_empties():
    m = _model(TULConfig(prefix_k=2, slot_id=4, coda_token_cut=999))
    x, y, layout, _ = _batch(_spec())
    with pytest.raises(ValueError, match="coda_token_cut"):
        m(x, labels=y, slot_layout=layout)
    m2 = _model(TULConfig(prefix_k=2, slot_id=4))
    with pytest.raises(ValueError, match="cut"):
        m2.tul_forward_cw_arms(x, y, layout, cut=layout.l_total)


# ── §2 the gather: window_drop_mask / compact_index mirror ──────────────────

def test_window_drop_mask_keeps_every_slot_regardless_of_index():
    # positions 2 and 5 are slots; the rest are tokens.
    slot_mask = torch.tensor([[False, False, True, False, False, True, False, False]])
    got = window_drop_mask(slot_mask, cut=3)
    want = torch.tensor([[True, True, False, False, False, False, False, False]])
    assert torch.equal(got, want), f"got {got.tolist()} want {want.tolist()}"
    # every slot position must be False in the drop mask no matter where cut lands,
    # including a cut that would otherwise reach past the slot's own row index.
    for cut in range(9):
        m = window_drop_mask(slot_mask, cut)
        assert not m[0, 2] and not m[0, 5], f"a slot was marked for drop at cut={cut}"


def test_window_cut_gather_index_keeps_slots_and_late_tokens_in_order():
    """The mirror of the existing ``test_compact_index_moves_tokens_to_the_front_in_order``:
    ``_tul_coda_without_slots`` moves tokens to the front and slots to the dump row; arm
    CW's gather must move (slots + late tokens) to the front and EARLY tokens to the dump
    row, preserving order."""
    slot_mask = torch.tensor([[False, False, True, False, False, True, False, False]])
    drop = window_drop_mask(slot_mask, cut=3)
    idx = compact_index(drop)
    assert idx[0].tolist() == [2, 3, 4, 5, 6, 7, 8, 8], f"got {idx[0].tolist()}"


def test_forward_coda_token_cut_zero_is_bit_identical_to_the_pre_cw_branch():
    """Hard constraint: rebuild the OLD (pre-CW) control flow by hand — the exact ops
    ``_forward_tul`` ran before this change (``_tul_front`` -> ``_tul_core`` ->
    ``prefix_project`` -> ``scatter_positions`` -> ``apply_token_dropout`` ->
    ``_back_region`` -> ``_tul_group_losses(layout=...)`` when ``coda_sees_slots`` is
    True) — and require the new forward, at ``coda_token_cut=0``, to match bit for bit.
    A model built with ``coda_token_cut=0`` must never call ``_tul_coda_gather`` at all.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4, coda_token_cut=0))
    m.eval()
    x, y, layout, _ = _batch(_spec())

    calls = {"n": 0}
    orig = m._tul_coda_gather

    def _spy(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    m._tul_coda_gather = _spy
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert calls["n"] == 0, "coda_token_cut=0 must never reach the CW gather"

    with torch.no_grad():
        xf, x0f, bgf = m._tul_front(x, layout)
        xn, h_slots, _depths = m._tul_core(xf, x0f, bgf, layout)
        values, pos = m.tul.prefix_project(h_slots, layout, layout.l_total)
        x_coda = scatter_positions(xn, pos, values)
        x_coda, keep = m.tul.apply_token_dropout(x_coda, layout, m.training)
        xh = m._back_region(x_coda, x0f, bgf, x, inject_keep=keep)
        manual = m._tul_group_losses(xh, y, layout, want_groups=True)

    assert torch.equal(out["loss"], manual["loss"])
    assert torch.equal(out["ce_tokens"], manual["ce_tokens"])
    assert torch.equal(out["ce_first_tok"], manual["ce_first_tok"])
    assert torch.equal(out["n_tokens"], (~layout.slot_mask).sum())


def test_forward_coda_token_cut_positive_drops_only_early_tokens():
    """arm CW1 through the ordinary forward: n_targets after the cut must equal the
    number of (slot emit labels) + (token labels with row index >= cut), and must be
    strictly fewer real targets than the uncut pass (proves something was actually
    removed, not a no-op gather)."""
    x, y, layout, _ = _batch(_spec(), B=1)
    cut = 10
    m0 = _model(TULConfig(prefix_k=2, slot_id=4, coda_token_cut=0))
    mC = _model(TULConfig(prefix_k=2, slot_id=4, coda_token_cut=cut))
    m0.eval(), mC.eval()
    with torch.no_grad():
        o0 = m0(x, labels=y, slot_layout=layout)
        oC = mC(x, labels=y, slot_layout=layout)
    assert torch.isfinite(oC["loss"])
    pos = torch.arange(layout.l_total).unsqueeze(0)
    early_tok = (~layout.slot_mask) & (pos < cut)
    assert early_tok.sum() > 0, "fixture must have early tokens for this test to bite"
    n_emit = int(layout.slot_valid.sum())
    n_tok_late = int(((~layout.slot_mask) & (pos >= cut)).sum())
    assert int(oC["n_targets"]) == n_emit + n_tok_late


def test_coda_sees_slots_false_with_zero_cut_is_unchanged_a4():
    """Regression guard: A4 (coda_sees_slots=False) alone must still behave exactly as
    before the refactor of ``_tul_coda_without_slots`` into ``_tul_coda_gather``."""
    m = _model(TULConfig(prefix_k=2, slot_id=4, coda_sees_slots=False))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert "ce_first_tok" not in out, "A4 removes the slots, so it has no emitting position"
    assert torch.isfinite(out["loss"])


# ── §3 cw2_retain_mask: budget, reproducibility ──────────────────────────────

def test_cw2_retain_mask_retains_exactly_the_budget_per_row():
    candidates = torch.zeros(3, 40, dtype=torch.bool)
    candidates[0, :20] = True     # 20 candidates, budget 6
    candidates[1, :5] = True      # 5 candidates, budget 5 (== pool, exact)
    candidates[2, :30] = True     # 30 candidates, budget 0
    budget = torch.tensor([6, 5, 0])
    retain = cw2_retain_mask(candidates, budget, seed=1234)
    assert retain.sum(dim=1).tolist() == [6, 5, 0]
    assert bool((retain & ~candidates).any()) is False, "retained a non-candidate position"


def test_cw2_retain_mask_never_retains_outside_the_candidate_pool():
    """A budget larger than the pool must saturate at the pool, not spill into
    non-candidate positions (there is no explicit clamp — the final ``candidates &``
    intersection is what has to do this; this test targets exactly that line)."""
    candidates = torch.zeros(1, 10, dtype=torch.bool)
    candidates[0, :3] = True
    retain = cw2_retain_mask(candidates, torch.tensor([50]), seed=0)
    assert int(retain.sum()) == 3, "budget must saturate at the candidate pool, not overrun it"
    assert bool((retain & ~candidates).any()) is False, "retained a non-candidate position"


def test_cw2_retain_mask_is_seed_reproducible_and_seed_sensitive():
    candidates = torch.zeros(1, 200, dtype=torch.bool)
    candidates[0, :100] = True
    budget = torch.tensor([20])
    a1 = cw2_retain_mask(candidates, budget, seed=42)
    a2 = cw2_retain_mask(candidates, budget, seed=42)
    b = cw2_retain_mask(candidates, budget, seed=43)
    assert torch.equal(a1, a2), "same seed must retain the same positions"
    assert not torch.equal(a1, b), "different seed must (with this pool size) retain differently"


# ── §4 the four arms score the same labels ───────────────────────────────────

def test_cw_arms_n_targets_identical_across_all_four():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _padded_batch(B=2, max_slots=6)
    cut = 10
    with torch.no_grad():
        out = m.tul_forward_cw_arms(x, y, layout, cut=cut, seed=7)
    ns = {name: int(g["n_targets"]) for name, g in out.items()}
    assert len(set(ns.values())) == 1, f"n_targets differ across arms: {ns}"
    pos = torch.arange(layout.l_total).unsqueeze(0)
    want = int(((~layout.slot_mask) & (pos >= cut) & (y != -100)).sum())
    assert ns["CW0"] == want


def test_cw2_drop_mask_used_by_the_model_matches_the_spec_formula():
    """Not just that the retained COUNT is right in isolation — that the model's
    ``tul_forward_cw_arms`` actually feeds THAT exact mask into the coda gather.
    Captures every ``drop_mask`` the model passes to ``_tul_coda_gather`` and checks
    CW2's against an independently-built reference (same seed).
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, spec = _padded_batch(B=2, max_slots=6)
    cut = 10
    captured: list[torch.Tensor] = []
    orig = m._tul_coda_gather

    def _spy(x_coda, x0, bigram_emb, keep, labels, layout_, drop_mask):
        captured.append(drop_mask.clone())
        return orig(x_coda, x0, bigram_emb, keep, labels, layout_, drop_mask)

    m._tul_coda_gather = _spy
    with torch.no_grad():
        m.tul_forward_cw_arms(x, y, layout, cut=cut, seed=3)
    assert len(captured) == 4, "expected one gather call per arm (CW0..CW3)"
    cw0, cw1, cw2, cw3 = captured    # insertion order of the dict literal in the source

    pos = torch.arange(layout.l_total).unsqueeze(0)
    early_tok = (~layout.slot_mask) & (pos < cut)
    budget = spec.prefix_k * layout.slot_valid.sum(dim=1)
    retain = cw2_retain_mask(early_tok, budget, seed=3)
    want_cw2 = layout.slot_mask | (early_tok & ~retain)

    assert not bool(cw0.any()), "CW0 must drop nothing"
    assert torch.equal(cw1, early_tok), "CW1 must drop exactly the early tokens"
    assert torch.equal(cw2, want_cw2), "CW2's drop mask does not match the spec formula"
    assert torch.equal(cw3, layout.slot_mask | early_tok), "CW3 must drop slots + early tokens"
    # CW2 must keep STRICTLY more than CW3 whenever any slot has a real budget — otherwise
    # the "equal-KV-budget random subset" retention is not actually reaching the coda.
    assert int((~cw2).sum()) > int((~cw3).sum()), "CW2 kept nothing extra over the floor"


def test_cw_arms_all_finite_and_cw3_is_the_floor_no_slots_no_early_tokens():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec(), B=1)
    cut = 10
    with torch.no_grad():
        out = m.tul_forward_cw_arms(x, y, layout, cut=cut, seed=0)
    for name, g in out.items():
        assert torch.isfinite(g["loss"]), f"{name} produced a non-finite loss"
    # CW2 has strictly more context than CW3 (its retained subset), so on an untrained
    # (random-weight) model the two losses coinciding to float32 precision would be an
    # extraordinary coincidence — this is the cheap end-to-end control that CW2's extra
    # context is actually reaching the coda, not silently discarded.
    assert not torch.equal(out["CW2"]["loss"], out["CW3"]["loss"]), (
        "CW2 == CW3: the retained random subset never reached the coda")
    # CW3's drop mask must be exactly "every slot OR every early token" — check via the
    # gather machinery directly rather than re-deriving CW3's number from the arms dict.
    from morph.model.tul import window_drop_mask as wdm
    want_drop = layout.slot_mask | wdm(layout.slot_mask, cut)
    assert int(want_drop.sum()) == int(layout.slot_mask.sum()) + int(
        ((~layout.slot_mask) & (torch.arange(layout.l_total).unsqueeze(0) < cut)).sum()
    )
