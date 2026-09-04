"""TUL span layout contract (docs/tul-spec.md §3.1, §4, §6, §9 invariants 1/3/5).

Invariant 1 is the reason this file exists: the boundary rule is ONE function used by
the loader and by the generator, and a train/generation mismatch would silently decode
without the plan (the coconut ``assert_layout_parity`` lesson). Everything here runs on
CPU with a hand-built vocabulary — no tokenizer, no shards.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.tul_layout import (
    BoundaryRule,
    TulDataConfig,
    TulLayoutSpec,
    boundary_lut_from_strings,
    pack_tul_batch,
    pack_tul_row,
)

# A 32-token toy vocabulary: 0 = EOS, 10 = ".", 11 = "\n", 12 = " --", rest are words.
V = 32
DOT, NL, DASH = 10, 11, 12


def _lut() -> np.ndarray:
    strings = [""] * V
    strings[0] = "<|endoftext|>"
    strings[DOT] = "."
    strings[NL] = "\n"
    strings[DASH] = " --"
    strings[13] = "Mr."          # ends in "." → in B (the documented abbreviation mis-cut)
    strings[14] = " word"
    for i in (5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20):
        strings[i] = f" w{i}"
    return boundary_lut_from_strings(strings, eos_id=0)


def _rule(**kw) -> BoundaryRule:
    base = dict(is_boundary=_lut(), min_span=4, span_cap=8, eos_id=0)
    base.update(kw)
    return BoundaryRule(**base)


# ── the rule itself ──────────────────────────────────────────────────────────

def test_lut_membership_matches_the_spec_rule():
    lut = _lut()
    assert lut[DOT] and lut[NL] and lut[DASH], "punctuation / newline / dash must be in B"
    assert lut[0], "EOS must be in B (spec §3.1 rule 1)"
    assert lut[13], "'Mr.' ends in '.' → in B; the id-rule cannot see the abbreviation"
    assert not lut[14] and not lut[5], "ordinary word pieces must not cut"


def test_run_of_boundary_tokens_yields_exactly_one_boundary():
    """`.` + `\\n` is ONE boundary (spec §3.1 rule 2's purpose), via min_span."""
    ids = np.array([5, 6, 7, DOT, NL, 5, 6, 7, 8], dtype=np.int64)
    pos, _ = _rule().cut(ids)
    assert pos.tolist() == [3], f"expected one boundary at the '.', got {pos.tolist()}"


def test_min_span_suppresses_a_short_span():
    ids = np.array([5, DOT, 6, DOT, 7, 8, 9, DOT], dtype=np.int64)
    pos, _ = _rule(min_span=4).cut(ids)
    # index 1 (span_len 2) and index 3 (span_len 4 → allowed) …
    assert pos.tolist() == [3, 7], f"got {pos.tolist()}"
    # control: with min_span 1 every punctuation token cuts.
    pos1, _ = _rule(min_span=1).cut(ids)
    assert pos1.tolist() == [1, 3, 7], f"min_span=1 control got {pos1.tolist()}"


def test_span_cap_forces_a_boundary():
    ids = np.full(20, 5, dtype=np.int64)          # no punctuation at all
    pos, _ = _rule(span_cap=8).cut(ids)
    assert pos.tolist() == [7, 15], f"cap must fire every 8 tokens, got {pos.tolist()}"


def test_eos_cuts_even_below_min_span():
    """Spec §3.1: EOS is a boundary; a span must never straddle a document."""
    ids = np.array([5, 6, 0, 7, 8, 9, 10], dtype=np.int64)
    pos, _ = _rule(min_span=4).cut(ids)
    assert 2 in pos.tolist(), f"EOS at index 2 must cut, got {pos.tolist()}"


def test_fixed_stride_arm_replaces_the_punctuation_rule():
    ids = np.array([5, DOT, 6, DOT, 7, 8, 9, DOT, 5, 5, 5, 5], dtype=np.int64)
    pos, _ = _rule(fixed_stride=5).cut(ids)
    assert pos.tolist() == [4, 9], f"A5 must cut every 5 tokens only, got {pos.tolist()}"


def test_incremental_parity_one_token_at_a_time():
    """INVARIANT 1: the generator's per-token call and the loader's whole-row call agree.

    1000 random rows, each replayed one token at a time through the same state machine.
    """
    rng = np.random.default_rng(0)
    rule = _rule()
    for trial in range(1000):
        n = int(rng.integers(1, 60))
        ids = rng.choice([5, 6, 7, 8, 9, DOT, NL, DASH, 0], size=n,
                         p=[0.16, 0.16, 0.16, 0.13, 0.13, 0.11, 0.06, 0.06, 0.03])
        ids = ids.astype(np.int64)
        whole, tail = rule.cut(ids)
        step, span = [], 0
        for i, t in enumerate(ids):
            cuts, span = rule.cut(np.array([t], dtype=np.int64), span)
            step += [i + int(c) for c in cuts]
        assert whole.tolist() == step, (
            f"trial {trial}: whole-row {whole.tolist()} != incremental {step} for {ids.tolist()}")
        assert tail == span, f"trial {trial}: tail span_len {tail} != {span}"


def test_incremental_parity_across_arbitrary_chunk_splits():
    """The state machine is resumable at ANY split, not just per-token."""
    rng = np.random.default_rng(7)
    rule = _rule()
    for _ in range(200):
        ids = rng.choice([5, 6, 7, DOT, NL, 0], size=int(rng.integers(2, 50))).astype(np.int64)
        whole, tail = rule.cut(ids)
        cut_at = int(rng.integers(1, len(ids)))
        a, span = rule.cut(ids[:cut_at], 0)
        b, span = rule.cut(ids[cut_at:], span)
        joined = a.tolist() + [cut_at + int(x) for x in b]
        assert whole.tolist() == joined and tail == span


# ── the packer ───────────────────────────────────────────────────────────────

def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=4, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _stream(n, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.choice([5, 6, 7, 8, 9, DOT, NL], size=n,
                     p=[0.2, 0.2, 0.2, 0.13, 0.13, 0.1, 0.04]).astype(np.int64)
    return ids


def test_l_total_is_fixed_and_token_count_varies():
    """INVARIANT 5: L_total = tokens + prefix_k·slots is fixed; token count varies."""
    spec, rule = _spec(), _rule()
    counts = set()
    for seed in range(20):
        arrays, n_tok, stats = pack_tul_row(_stream(200, seed), rule, spec)
        for name, arr in arrays.items():
            want = spec.max_slots if name in ("slot_index", "slot_valid") else spec.l_total
            assert arr.shape == (want,), f"{name} has shape {arr.shape}, want ({want},)"
        n_slots = int(arrays["slot_valid"].sum())
        n_pad = int(spec.l_total - n_tok - spec.prefix_k * n_slots)
        assert n_tok + spec.prefix_k * n_slots + n_pad == spec.l_total
        assert 0 <= n_pad <= spec.prefix_k or n_slots == spec.max_slots, (
            f"tail padding {n_pad} exceeds one unit without the max_slots cap firing")
        counts.add(n_tok)
    assert len(counts) > 1, "token count must vary across rows (spec §3.1)"


def test_every_boundary_inside_the_row_gets_its_slot():
    """Spec §3.1: the packer never drops a boundary inside the row."""
    spec, rule = _spec(seq_len=64, max_slots=16), _rule()
    ids = _stream(400, 3)
    arrays, n_tok, _ = pack_tul_row(ids, rule, spec)
    expected, _ = rule.cut(ids[:n_tok])
    assert int(arrays["slot_valid"].sum()) == len(expected), (
        f"{int(arrays['slot_valid'].sum())} slots for {len(expected)} boundaries in the row")


def test_labels_are_the_double_label_of_spec_3_4():
    spec, rule = _spec(), _rule()
    arrays, n_tok, _ = pack_tul_row(_stream(200, 1), rule, spec)
    ins, labs = arrays["input_ids"], arrays["labels"]
    K = spec.prefix_k
    for s, first in enumerate(arrays["slot_index"][arrays["slot_valid"]]):
        assert (ins[first:first + K] == spec.slot_id).all(), "slot positions carry slot_id"
        assert (labs[first:first + K - 1] == -100).all(), (
            "the plan-carrying prefix positions must have NO label (spec §3.1)")
        emit, t_last = labs[first + K - 1], labs[first - 1]
        assert emit == t_last, (
            f"slot {s}: the emitting position ({emit}) and t_last ({t_last}) must predict "
            f"the SAME token — that is the counterfactual pair of §7.2")
        assert emit != -100


def test_pad_slots_and_tail_pads_never_carry_a_label():
    spec, rule = _spec(), _rule()
    arrays, _, _ = pack_tul_row(_stream(200, 2), rule, spec)
    pads = arrays["slot_mask"].copy()
    for first in arrays["slot_index"][arrays["slot_valid"]]:
        pads[first:first + spec.prefix_k] = False       # real slots
    assert (arrays["labels"][pads] == -100).all(), "tail pads must be −100 (invariant 3)"
    assert (arrays["input_ids"][pads] == spec.slot_id).all()
    inv = ~arrays["slot_valid"]
    assert arrays["slot_index"][inv].tolist() == [0] * int(inv.sum())


def test_bag_id_maps_tokens_to_the_slot_that_closes_their_span():
    spec, rule = _spec(), _rule()
    arrays, n_tok, _ = pack_tul_row(_stream(200, 4), rule, spec)
    bag, mask, S = arrays["bag_id"], arrays["slot_mask"], spec.max_slots
    for s, first in enumerate(arrays["slot_index"][arrays["slot_valid"]]):
        assert (bag[first:first + spec.prefix_k] == s).all(), "a slot position carries its own id"
        # the tokens immediately before the slot belong to its span
        assert bag[first - 1] == s
    # tokens after the last real slot have no slot → the dump bin
    last = int(arrays["slot_index"][arrays["slot_valid"]][-1]) + spec.prefix_k
    trailing = ~mask[last:]
    if trailing.any():
        assert (bag[last:][trailing] == S).all(), "trailing tokens must use the dump bin"


def test_slot_id_present_in_the_stream_is_a_hard_error():
    spec, rule = _spec(), _rule()
    ids = _stream(200, 5).copy()
    ids[17] = spec.slot_id
    with pytest.raises(ValueError, match="slot_id"):
        pack_tul_row(ids, rule, spec)


def test_max_slots_budget_ends_the_row_early():
    """Spec §3.1: 'If a row would exceed max_slots the packer ends the row early.'"""
    spec = _spec(seq_len=64, max_slots=2)        # L_total 68, but only 2 slots allowed
    rule = _rule(span_cap=5, min_span=4)
    ids = np.full(300, 5, dtype=np.int64)
    arrays, n_tok, stats = pack_tul_row(ids, rule, spec)
    assert int(arrays["slot_valid"].sum()) == 2, "the slot budget must bind"
    # The row stops at the 3rd forced boundary (index 14), so it keeps the two complete
    # spans plus the tokens of the next one — and still drops no boundary.
    assert n_tok == 14, f"expected the row to end at the 3rd boundary, got n_tok={n_tok}"
    assert len(rule.cut(ids[:n_tok])[0]) == 2, "no boundary inside the row may be unslotted"
    assert stats["pad_frac"] > 0.5, "the early row end must show up as tail padding"


def test_pack_tul_batch_consumes_only_what_it_uses():
    spec, rule = _spec(), _rule()
    buf = _stream(4 * (spec.l_total + 1) + 500, 6).tolist()
    before = len(buf)
    x, y, layout = pack_tul_batch(buf, rule, spec, batch_size=4)
    assert x.shape == y.shape == (4, spec.l_total)
    assert layout.slot_index.shape == (4, spec.max_slots)
    used = before - len(buf)
    assert 4 <= used <= 4 * spec.l_total, f"consumed {used} tokens for 4 rows"
    assert layout.stats and layout.stats["mean_span"] > 0


def test_tul_data_config_derives_max_slots_from_seq_len():
    cfg = TulDataConfig(rule=_rule(), prefix_k=2, slot_id=4)
    spec = cfg.spec_for(1024)
    assert spec.max_slots == 128, "spec §8: max_slots = seq_len // 8"
    assert spec.l_total == 1280, "spec §3.1: L_total = 1024 + 2·128"


def test_span_cap_below_min_span_is_rejected():
    with pytest.raises(ValueError, match="span_cap"):
        _rule(min_span=8, span_cap=4)


# ── generator ↔ loader parity (spec §6, invariant 1) ─────────────────────────

def test_generator_row_builder_matches_the_packer():
    """The incremental builder and the batch packer produce the SAME layout."""
    from morph.inference.tul_generate import TulRowBuilder

    spec, rule = _spec(seq_len=64, max_slots=16), _rule()
    ids = _stream(60, 11)
    b = TulRowBuilder(rule=rule, spec=spec)
    for t in ids:
        b.append(int(t))
    # The packer needs spare tokens (it peeks one for the last label and fills to
    # L_total), so give it more and compare over the builder's prefix.
    arrays, n_tok, _ = pack_tul_row(np.concatenate([ids, _stream(80, 12)]), rule, spec)
    assert n_tok >= len(ids), f"the packer consumed only {n_tok} of {len(ids)} tokens"
    n = len(b.ids)
    assert b.ids == arrays["input_ids"][:n].tolist(), "input ids diverge"
    assert b.slot_mask == arrays["slot_mask"][:n].tolist(), "slot mask diverges"
    packer_slots = [s for s in arrays["slot_index"][arrays["slot_valid"]].tolist() if s < n]
    assert b.slot_first == packer_slots, (
        "slot positions diverge — this is the train/generation mismatch invariant 1 forbids")
    _ids, layout = b.tensors(torch.device("cpu"))
    # bag_id agrees over every CLOSED span. The builder's row ends mid-span, so its
    # trailing tokens correctly sit in the dump bin while the packer — which saw the
    # span close — has already assigned them to their slot.
    end = b.slot_first[-1] + spec.prefix_k
    assert torch.equal(layout.bag_id[0, :end], torch.from_numpy(arrays["bag_id"][:end]))
    assert (layout.bag_id[0, end:] == spec.max_slots).all(), (
        "tokens of the still-open span must sit in the dump bin, not in a slot that "
        "does not exist yet")
