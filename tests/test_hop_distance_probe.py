"""Contracts for the pure logic of `lab/divergence/hop_distance_probe.py`.

Three claims the probe's reading rests on, and each is a claim that a plausible-looking
table would hide if it were false:

1. THE SCHEDULE. Over the H+1 corruption phases every scored token must get exactly ONE
   reading at each distance 1…H — no distance twice, none missing. A wrong modulus still
   prints a full ΔCE profile; it just prints one of a different quantity.
2. THE PLANT. The planted pair must sit at the span distance the probe claims, under the
   REAL packer's boundaries — and the scored position's label must really be the planted
   repeat, which is a property of `pack_tul_row`, not of the probe.
3. THE AGGREGATION. A hop bin's CE must be the token-weighted mean over that bin's tokens
   and its bootstrap units must be rows.

No model, no checkpoint, no GPU: this file is the logic gate, the probe's own `--planted`
positive control (g = 0) is the measurement gate.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "lab/divergence")
from hop_distance_probe import (                                   # noqa: E402
    HopTable, bin_sums, corrupt_mask_for_phase, donor_replacement,
    hop_before_row_start, hop_of, hops_scored_by_token, hops_seen_by_token,
    next_token_corrupted, own_prefix_mask, plant_sites, planted_triple,
    source_position, span_index_from_layout,
)

from morph.model.tul_layout import (                               # noqa: E402
    BoundaryRule, TulDataConfig, pack_tul_batch,
)

H = 6


# ── 1. the corruption schedule ─────────────────────────────────────────────────────

def test_every_token_is_scored_once_at_each_distance():
    """The schedule's whole point: distances 1..H, each exactly once, per token."""
    for span in range(0, 40):
        assert hops_seen_by_token(span, H) == list(range(1, H + 1)), span


def test_schedule_contract_holds_through_the_mask_path():
    """Same claim, but read the way the probe reads it: hop_of over the phases."""
    spans = np.repeat(np.arange(30), 5)          # 30 spans of 5 tokens
    seen = [[] for _ in range(spans.size)]
    for c in range(H + 1):
        cor = corrupt_mask_for_phase(spans, c, H)
        hops = hop_of(spans, c, H)
        for t in range(spans.size):
            if cor[t]:
                assert hops[t] == 0             # corrupted <=> distance 0
                continue
            seen[t].append(int(hops[t]))
    for t in range(spans.size):
        assert sorted(seen[t]) == list(range(1, H + 1)), (t, seen[t])


def test_an_early_span_has_no_reading_past_its_own_start():
    """The wrap-around drop: for a token of span 2 at H 6 the spans at distances 3..6 are
    not in the row, so phase c corrupted nothing that token can causally read and the
    reading must be dropped rather than counted as a ΔCE of 0."""
    assert hops_scored_by_token(2, H) == [1, 2]
    assert hops_scored_by_token(0, H) == []
    assert hops_scored_by_token(H, H) == list(range(1, H + 1))
    assert hops_scored_by_token(50, H) == list(range(1, H + 1))


def test_wrap_around_is_dropped_through_the_mask_path():
    spans = np.repeat(np.arange(4), 3)            # spans 0..3, three tokens each
    seen = [[] for _ in range(spans.size)]
    for c in range(H + 1):
        cor = corrupt_mask_for_phase(spans, c, H)
        hops = hop_of(spans, c, H)
        drop = hop_before_row_start(spans, hops)
        for t in range(spans.size):
            if cor[t] or drop[t]:
                continue
            seen[t].append(int(hops[t]))
    for t in range(spans.size):
        assert sorted(seen[t]) == hops_scored_by_token(int(spans[t]), H), (t, seen[t])
    # and the drop is exactly the wrap-around, never a real reading
    assert not hop_before_row_start(np.array([9, 9]), np.array([1, 6])).any()
    assert hop_before_row_start(np.array([-1]), np.array([3])).tolist() == [False]


def test_wrap_around_left_in_would_bias_h_star_low():
    """Two-sided: without the drop an early-span token DOES get a full 1..H reading, which
    is the bias this probe would otherwise print as a hop profile."""
    spans = np.repeat(np.arange(4), 3)
    for t, sp in enumerate(spans):
        undropped = sorted(int(hop_of(spans, c, H)[t]) for c in range(H + 1)
                           if not corrupt_mask_for_phase(spans, c, H)[t])
        assert undropped == list(range(1, H + 1))
        if sp < H:
            assert undropped != hops_scored_by_token(int(sp), H)


def test_a_wrong_modulus_is_caught():
    """The sabotage this file exists to catch: H instead of H+1 in the schedule."""
    def broken(span: int, hops: int) -> list[int]:
        return sorted(h for h in ((span - c) % hops for c in range(hops + 1)) if h != 0)

    assert any(broken(s, H) != list(range(1, H + 1)) for s in range(40))


def test_slot_positions_and_the_tail_span_get_their_own_index():
    bag = np.array([0, 0, 0, 0, 1, 1, 1, 9, 9], dtype=np.int64)     # 9 = max_slots dump
    slot = np.array([False, False, True, True, False, False, False, False, True])
    sp = span_index_from_layout(bag, slot, max_slots=9, n_slots=2)
    assert sp.tolist() == [0, 0, -1, -1, 1, 1, 1, 2, -1]


def test_next_token_corrupted_marks_the_position_whose_label_moved():
    istok = np.array([True, True, False, True, True])               # position 2 is a slot
    corrupt = np.array([False, False, False, True, False])
    got = next_token_corrupted(istok, corrupt)
    # token order is 0, 1, 3, 4; position 3 is corrupted, so position 1 carries its id
    assert got.tolist() == [False, True, False, False, False]


def test_own_prefix_mask_splits_each_span_and_scores_only_the_tail():
    spans = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, -1], dtype=np.int64)
    cor, sc = own_prefix_mask(spans)
    assert cor.tolist() == [True, True, True, False, False, False, True, True, False, False]
    assert sc.tolist() == [False, False, False, True, True, False, False, False, True, False]
    assert not (cor & sc).any()
    assert not cor[spans < 0].any() and not sc[spans < 0].any()


def test_donor_replacement_only_uses_the_donors_own_tokens():
    donor = np.array([11, 12, 13], dtype=np.int64)
    rep = donor_replacement(donor, 7)
    assert rep.tolist() == [11, 12, 13, 11, 12, 13, 11]
    assert set(rep.tolist()) <= set(donor.tolist())
    with pytest.raises(ValueError):
        donor_replacement(np.empty(0, dtype=np.int64), 3)


# ── 2. the planted pair, under the real packer ─────────────────────────────────────

def _packed_row(seed: int = 0):
    lut = np.zeros(64, dtype=bool)
    lut[10] = True                                     # the "." token
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)
    rng = np.random.default_rng(seed)
    toks = rng.integers(11, 60, size=4000)             # never eos (0), "." (10), slot_id (4)
    toks[rng.random(4000) < 0.12] = 10
    spec = TulDataConfig(rule=rule, prefix_k=2, slot_id=4).spec_for(512)
    inp, labels, layout = pack_tul_batch(toks.tolist(), rule, spec, batch_size=1)
    n_slots = int(layout.slot_valid[0].sum())
    spans = span_index_from_layout(layout.bag_id[0].numpy(), layout.slot_mask[0].numpy(),
                                   spec.max_slots, n_slots)
    istok = (~layout.slot_mask[0].numpy()) & (labels[0].numpy() >= 0)
    return inp[0].numpy(), labels[0].numpy(), np.where(istok, spans, -1), n_slots


def test_planted_pair_lands_at_the_claimed_span_distance():
    inp, labels, spans, n_slots = _packed_row()
    rng = np.random.default_rng(7)
    sites = plant_sites(int(spans.max()) + 1, H)
    assert len(sites) >= 3, sites
    for s in sites:
        scored, repeat = planted_triple(spans, s)
        assert spans[scored] == s
        assert spans[repeat] == s + 1
        assert scored < repeat
        for g in range(H + 1):
            src = source_position(spans, s, g, rng)
            assert spans[src] == s - g, (s, g, spans[src])
            assert src != scored                     # g=0 must not overwrite the scorer


def test_the_scored_position_really_carries_the_repeat_as_its_label():
    """`pack_tul_row`'s invariant, checked rather than assumed: the last token of span s
    is the position whose label is the FIRST token of span s+1."""
    inp, labels, spans, _ = _packed_row(seed=3)
    for s in plant_sites(int(spans.max()) + 1, H):
        scored, repeat = planted_triple(spans, s)
        assert labels[scored] == inp[repeat], (s, labels[scored], inp[repeat])


def test_plant_windows_never_overlap():
    sites = plant_sites(200, H)
    windows = [set(range(s - H, s + 2)) for s in sites]
    for a, b in zip(windows, windows[1:]):
        assert not (a & b), (a, b)
    assert min(sites) >= H                            # span s-H must exist


def test_plant_sites_rejects_a_degenerate_hop_count():
    with pytest.raises(ValueError):
        plant_sites(50, 0)
    assert plant_sites(3, H) == []                    # too few spans: no site, not a crash


# ── 3. the aggregation ─────────────────────────────────────────────────────────────

def test_bin_sums_are_per_row_and_count_each_token_once():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    row = np.array([0, 0, 1, 1, 1])
    keep = np.array([True, False, True, True, False])
    s, n = bin_sums(vals, row, keep, n_rows=3)
    assert s.tolist() == [1.0, 7.0, 0.0]
    assert n.tolist() == [1.0, 2.0, 0.0]


def test_hop_table_means_are_token_weighted_over_the_bin():
    hstar = np.array([1, 1, 2, 2, 2, -1])
    row = np.array([0, 1, 0, 1, 1, 0])
    ce = {1: np.array([2.0, 4.0, 1.0, 1.0, 1.0, 9.0]),
          6: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 9.0])}
    tab = HopTable(hstar, row, n_rows=2)
    assert tab.bins() == [1, 2]

    def boot(a, b, n):
        return {"point": float(a.sum() / n.sum() - b.sum() / n.sum())}

    out = tab.table(ce, [(1, 6)], boot)
    assert out["1"]["n_tokens"] == 2
    assert out["1"]["ce"]["1"] == pytest.approx(3.0)      # (2+4)/2, not a mean of rows
    assert out["1"]["K1-K6"]["point"] == pytest.approx(2.0)
    assert out["2"]["ce"]["1"] == pytest.approx(1.0)
    assert out["2"]["K1-K6"]["point"] == pytest.approx(0.0)
    assert "-1" not in out                                # incomplete tokens are dropped


def test_hop_table_refuses_mismatched_inputs():
    with pytest.raises(ValueError):
        HopTable(np.zeros(3), np.zeros(4), n_rows=1)


# ── 4. the second pass (2026-09-19): source pairs and the reach cut ────────────────
#
# Two options the plateau-and-dilution prereg needs before it runs. The pair is pure
# logic; the cut has to be proved on a real tiny model, because its contract is EXACT
# invisibility of a span beyond the cut, and only a forward can show that.

from hop_distance_probe import install_reach_cut, reach0_kw, source_positions   # noqa: E402


def test_source_positions_are_consecutive_and_inside_the_span():
    span = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = source_positions(span, 2, 1, rng, 2)
        assert p[1] == p[0] + 1 and span[p[0]] == 1 and span[p[1]] == 1
        q = source_positions(span, 2, 0, rng, 2)
        assert q[1] == q[0] + 1 and span[q[0]] == 2 and q[1] < 11      # never the scored pos


def test_source_positions_refuses_a_span_with_no_run_of_n():
    span = np.array([0, 0, 0, -1, 1, -1, 1, -1, 2, 2, 2, 2])              # span 1 has no pair
    with pytest.raises(ValueError, match="no run of 2"):
        source_positions(span, 2, 1, np.random.default_rng(0), 2)
    assert source_positions(span, 2, 1, np.random.default_rng(0), 1)[0] in (4, 6)


def test_source_position_is_the_n1_form():
    span = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    a = source_position(span, 2, 1, np.random.default_rng(3))
    b = source_positions(span, 2, 1, np.random.default_rng(3), 1)[0]
    assert a == b and span[a] == 1


def test_reach0_kw_is_the_diagonal_and_keeps_the_segment():
    import torch
    S = 5
    ii = torch.arange(S).unsqueeze(1); jj = torch.arange(S).unsqueeze(0)
    m = ((ii >= jj) & (jj >= ii - 1)).view(1, 1, S, S)
    seg = torch.arange(S).unsqueeze(0)
    out = reach0_kw({"tg_allow": m, "tg_comp_allow": m, "tg_seg": seg})
    assert torch.equal(out["tg_allow"], torch.eye(S, dtype=torch.bool).view(1, 1, S, S))
    assert torch.equal(out["tg_comp_allow"], out["tg_allow"])
    assert out["tg_seg"] is seg
    with pytest.raises(ValueError, match="tg_relation"):
        reach0_kw({"tg_relation": m})


def _reach_model(depth: int):
    from test_tul_strict_geometry import _model
    return _model(tg_geometry="strict", tg_coda_prefix_reach="prev", loop_reach=1,
                  slot_depth_fixed=depth, slot_max_depth=8)


def _span_delta(m, ids, edited, row: int, span: int):
    """max |logit delta| over the tokens of ``span`` in ``row`` (inequality-based, as in
    `test_tul_strict_geometry._leak`: the slot_id column is -inf everywhere)."""
    import torch
    from test_tul_strict_geometry import _logits
    a, lay = _logits(m, ids, "normal")
    b, _ = _logits(m, edited, "normal")
    tok = (~lay.slot_mask[row]) & (lay.bag_id[row] == span)
    d = (a[row] - b[row]).abs().nan_to_num(0.0)
    d = torch.where(a[row] != b[row], d, torch.zeros_like(d))
    return float(d[tok].max())


def test_reach_cut_at_or_past_max_depth_is_bit_identical():
    import torch
    from test_tul_strict_geometry import _ids, _logits
    m = _reach_model(3)
    ids = _ids()
    a, _ = _logits(m, ids, "normal")
    restore = install_reach_cut(m, 3)                    # passes 0,1,2 run; index 3 never
    b, _ = _logits(m, ids, "normal")
    restore()
    assert torch.equal(a.nan_to_num(0.0), b.nan_to_num(0.0))
    c, _ = _logits(m, ids, "normal")
    assert torch.equal(a.nan_to_num(0.0), c.nan_to_num(0.0))     # restore restores


def test_reach_cut_makes_spans_beyond_the_cut_exactly_invisible():
    """Coda reads cell j-1, loop reach 1, depth 3. Uncut, span j sees h <= 4. With the cut
    at pass index 1 (only pass 0 mixes) it sees h <= 2: an edit two spans back still moves
    span j, an edit three spans back moves it by EXACTLY zero."""
    from test_tul_strict_geometry import _edit, _ids, _pack
    m = _reach_model(3)
    ids = _ids()
    _i, _inp, _lab, layout = _pack()
    j = 5
    e2 = _edit(ids, layout, 0, j - 2)
    e3 = _edit(ids, layout, 0, j - 3)
    assert _span_delta(m, ids, e3, 0, j) > 0.0, "uncut, h=3 must move span j at depth 3"
    restore = install_reach_cut(m, 1)
    try:
        assert _span_delta(m, ids, e2, 0, j) > 0.0, "h=2 arrives at pass 0 and must move"
        assert _span_delta(m, ids, e3, 0, j) == 0.0, "h=3 needs pass 1, which is cut"
    finally:
        restore()
    restore0 = install_reach_cut(m, 0)
    try:
        assert _span_delta(m, ids, e2, 0, j) == 0.0, "with no mixing pass, h=2 is invisible"
    finally:
        restore0()


def test_install_reach_cut_refuses_negative():
    with pytest.raises(ValueError):
        install_reach_cut(object(), -1)


# ── 5. the span swap (2026-09-22): the --planted replacement ───────────────────────
#
# --planted plants an id absent from the WHOLE batch (control CE 15.5 nats, 4.7 above
# uniform) and is an exact-induction test a 5k-step model barely passes. --swap replaces
# span k-g's tokens with NATURAL text (another row of the batch, or the span's own tokens
# reordered) and reads span k's own CE against it — see the module docstring, part 3.

from hop_distance_probe import (                                  # noqa: E402
    build_swap_plan, build_swap_sites, donor_window, kept_fraction, shuffle_ids,
    span_token_positions, swap_replacement, swap_sites,
)


def test_swap_sites_windows_never_overlap():
    sites = swap_sites(200, H)
    windows = [set(range(s - H, s + 1)) for s in sites]
    for a, b in zip(windows, windows[1:]):
        assert not (a & b), (a, b)
    assert min(sites) >= H                             # span k-H must exist


def test_swap_sites_rejects_a_degenerate_hop_count():
    with pytest.raises(ValueError):
        swap_sites(50, 0)
    assert swap_sites(3, H) == []                       # too few spans: no site, not a crash


def test_span_token_positions_is_span_local():
    span = np.array([0, 0, -1, 1, 1, 1])
    assert span_token_positions(span, 0).tolist() == [0, 1]
    assert span_token_positions(span, 1).tolist() == [3, 4, 5]
    assert span_token_positions(span, 2).tolist() == []


# ── (1) the swap changes ONLY span k-g's positions ──────────────────────────────────

def test_swap_replacement_changes_only_the_source_spans_positions():
    span = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    inp = np.array([10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33])
    donor_span = span.copy()
    donor_ids = np.array([90, 91, 92, 93, 80, 81, 82, 83, 70, 71, 72, 73])
    pos, ids = swap_replacement(span, k=2, g=1, donor_span_idx=donor_span,
                                donor_ids=donor_ids)
    assert pos.tolist() == span_token_positions(span, 1).tolist()
    out = inp.copy()
    out[pos] = ids
    diff = out != inp
    assert diff.tolist() == (span == 1).tolist(), \
        "the swap must change span k-g's tokens and nothing else"
    assert ids.tolist() == donor_ids[donor_span == 1].tolist(), \
        "index-matched, same-length donor span: its own tokens, unmodified"


def test_swap_replacement_is_none_when_the_source_span_is_absent_here():
    span = np.array([0, 0, 0, 0, 2, 2, 2, 2])            # span 1 does not exist in this row
    donor_span = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    donor_ids = np.arange(100, 108)
    assert swap_replacement(span, k=2, g=1, donor_span_idx=donor_span,
                            donor_ids=donor_ids) is None


# ── (2) the donor text is natural ───────────────────────────────────────────────────

def test_donor_window_prefers_the_index_matched_span():
    donor_span = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    win = donor_window(donor_span, s=1, length=3)
    assert win.tolist() == [3, 4, 5]


def test_donor_window_falls_back_to_any_contiguous_run_of_the_right_length():
    # span 1 is only 2 tokens long (no exact match); span 2 has 5, a 4-run sits inside it
    donor_span = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    win = donor_window(donor_span, s=1, length=4)
    assert win.tolist() == [5, 6, 7, 8]                  # first 4 of span 2's 5 positions
    assert len({int(donor_span[p]) for p in win}) == 1, "the window must not cross a boundary"


def test_donor_window_is_none_when_no_window_of_that_length_exists():
    donor_span = np.array([0, 0, 1, 1, 2, 2])            # every span only 2 tokens
    assert donor_window(donor_span, s=0, length=3) is None


def test_swap_replacement_donor_text_is_natural():
    """Every swapped id occurs in the donor row at the position taken from it: the
    replacement is real donor content, never synthesized."""
    span = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    donor_span = np.array([0, 0, 5, 5, 5, 5, 1, 1])      # donor's own span 1 is only 2 long
    donor_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    pos, ids = swap_replacement(span, k=1, g=1, donor_span_idx=donor_span,
                                donor_ids=donor_ids)
    win = donor_window(donor_span, s=1, length=4)        # the fallback window this must use
    assert win.tolist() == [2, 3, 4, 5]
    for p, tid in zip(win.tolist(), ids.tolist()):
        assert donor_ids[p] == tid, (p, tid)


# ── (3) shuffle is a permutation of the span's own ids ──────────────────────────────

def test_shuffle_ids_is_a_permutation_of_the_spans_own_ids():
    own = np.array([10, 11, 12, 13, 14, 90, 91, 92])
    pos = np.array([2, 3, 4])
    out = shuffle_ids(own, pos, np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(own[pos].tolist())
    assert out.shape == pos.shape


def test_shuffle_ids_is_deterministic_in_the_rng():
    own = np.arange(20)
    pos = np.arange(5, 15)
    a = shuffle_ids(own, pos, np.random.default_rng(1))
    b = shuffle_ids(own, pos, np.random.default_rng(1))
    assert a.tolist() == b.tolist()
    assert a.tolist() != own[pos].tolist(), \
        "seed 1's permutation of 10 items happened to be identity; pick another seed"


# ── kept_fraction, the g-table's own summary number ─────────────────────────────────

def test_kept_fraction_is_the_deepest_over_the_first_nonzero():
    benefit = {"1": 0.0, "2": 0.1, "6": 0.2}
    assert kept_fraction(benefit, [1, 2, 6]) == pytest.approx(2.0)


def test_kept_fraction_is_nan_when_every_depth_reads_zero():
    benefit = {"1": 0.0, "2": 0.0, "6": 0.0}
    v = kept_fraction(benefit, [1, 2, 6])
    assert v != v                                        # NaN


def test_kept_fraction_is_nan_when_the_deepest_reading_is_missing_or_nan():
    v = kept_fraction({"1": 0.3, "6": float("nan")}, [1, 6])
    assert v != v


# ── build_swap_sites / build_swap_plan: site coverage and skip counting ────────────

def test_build_swap_sites_matches_swap_sites_filtered_to_real_spans():
    span = np.repeat(np.arange(10), 5)                   # 10 spans of 5 tokens, all present
    sites = build_swap_sites(span, H)
    assert [k for k, _ in sites] == swap_sites(10, H)
    for k, pos in sites:
        assert pos.tolist() == span_token_positions(span, k).tolist()


def test_build_swap_plan_visits_every_site_once_per_g_and_counts_skips():
    """A row with two sites at hops=1: site k=1 needs a length-2 source (satisfiable in
    the donor) and site k=3 needs a length-5 source (the donor never has one that long).
    Every (site, g) pair must appear exactly once, and the skip count must match an
    independent recomputation through `swap_replacement` directly."""
    span_idx = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    donor_span = np.array([0, 0, 1, 1, 1, 2])             # donor's longest span is length 3
    donor_ids = np.arange(200, 206)
    hops = 1
    plan = build_swap_plan(span_idx, hops, "row", donor_span, donor_ids, None,
                           np.random.default_rng(0))
    sites = build_swap_sites(span_idx, hops)
    assert [k for k, _ in sites] == [1, 3]
    expected_keys = {(k, g) for k, _ in sites for g in range(1, hops + 1)}
    assert set(plan.keys()) == expected_keys, "every site must be visited once per g"

    n_skipped = sum(1 for v in plan.values() if v is None)
    n_expected_skip = sum(
        1 for k, _ in sites for g in range(1, hops + 1)
        if swap_replacement(span_idx, k, g, donor_span, donor_ids) is None)
    assert n_skipped == n_expected_skip
    assert n_skipped == 1, "fixture must produce exactly one skip and one hit"
    assert plan[(1, 1)] is not None, "the length-2 request must be satisfied"
    assert plan[(3, 1)] is None, "the length-5 request must be skipped: no donor span "\
                                 "is long enough"


def test_build_swap_plan_shuffle_mode_never_needs_a_donor():
    span_idx = np.array([0, 0, 0, 1, 1, 1])
    own_ids = np.array([1, 2, 3, 4, 5, 6])
    plan = build_swap_plan(span_idx, 1, "shuffle", None, None, own_ids,
                           np.random.default_rng(2))
    site = build_swap_sites(span_idx, 1)
    assert [k for k, _ in site] == [1]
    pos, ids = plan[(1, 1)]
    assert pos.tolist() == span_token_positions(span_idx, 0).tolist()
    assert sorted(ids.tolist()) == sorted(own_ids[pos].tolist())


def test_build_swap_plan_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        build_swap_plan(np.array([0, 0]), 1, "bogus", None, None, None,
                        np.random.default_rng(0))


# ── (4) GEOMETRY: the reach staircase, through the swap machinery end to end ────────
#
# `tul_slot_spandec_strict_fan4_all_reach1`'s composition (fan_k 4, fan_mix all,
# loop_reach 1, tg_coda_prefix_reach prev, tg_geometry strict): a token of span j reads
# its own span and slot j-1's cells only; at forced depth T slot j-1's cells hold spans
# j-1 .. j-1-T (`test_lxtul_r_composition.py::test_loop_reach1_register_reaches_exactly_
# one_slot_per_pass`). So at depth 1, span j-2 reaches span j and span j-3 does not; at
# depth 2, span j-3 reaches and span j-4 does not. This proves the SAME staircase reading
# through `swap_replacement` + a real forward, not the perturbation helper that file uses.

from test_lxtul_r_composition import _r1_model                    # noqa: E402
from test_tul_fan import _batch                                   # noqa: E402
from hop_distance_probe import span_index_from_layout as _spans_from_layout  # noqa: E402


def _swap_and_run(depth: int, g: int):
    import torch
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    n_slots0 = int(layout.slot_valid[0].sum())
    n_slots1 = int(layout.slot_valid[1].sum())
    spans0 = _spans_from_layout(layout.bag_id[0].numpy(), layout.slot_mask[0].numpy(),
                                layout.max_slots, n_slots0)
    spans1 = _spans_from_layout(layout.bag_id[1].numpy(), layout.slot_mask[1].numpy(),
                                layout.max_slots, n_slots1)
    donor_ids = inp[1].numpy()
    nb = int(layout.bag_id[0].max())
    j = min(nb - 1, 7)
    replacement = swap_replacement(spans0, j, g, spans1, donor_ids)
    assert replacement is not None, \
        f"fixture must offer a donor window for g={g} at depth={depth}"
    pos, ids_ = replacement
    swapped = inp.clone()
    row = swapped[0].numpy().copy()
    row[pos] = ids_
    swapped[0] = torch.from_numpy(row)

    m = _r1_model(seed=12, tul_kw=dict(slot_mean_depth=depth, slot_max_depth=depth))
    m.eval()
    tgt = span_token_positions(spans0, j)
    with torch.no_grad():
        a = m(inp, labels=None, slot_layout=layout)["logits"][0, tgt]
        b = m(swapped, labels=None, slot_layout=layout)["logits"][0, tgt]
    d = (a - b)
    d[:, m.cfg.tul.slot_id] = 0.0                        # the masked column is -inf on both
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    return float(d.abs().max())


def test_swap_at_forced_depth_1_reaches_g1_and_not_g3():
    lit = _swap_and_run(depth=1, g=1)
    dark = _swap_and_run(depth=1, g=3)
    assert lit > 0.0, f"depth 1: g=1 must move span j's logits, got {lit}"
    assert dark == 0.0, f"depth 1: g=3 must NOT move span j's logits, got {dark}"


def test_swap_at_forced_depth_2_does_not_reach_g4():
    dark = _swap_and_run(depth=2, g=4)
    assert dark == 0.0, f"depth 2: g=4 must NOT move span j's logits, got {dark}"


# ── `_swap`'s OWN bookkeeping, no model ─────────────────────────────────────────────
#
# The tests above cover every function `_swap` calls, in isolation, plus one real
# forward. This one exercises `_swap` ITSELF — site indexing across TWO rows of one
# batch, the control/swapped pairing, the skip count, `bin_sums`/`paired_bootstrap_ci`
# wiring — via a stub `forward_ce` whose output is a pure function of the token ids, so
# every number `_swap` reports can be recomputed independently by hand and checked.

from hop_distance_probe import _swap                               # noqa: E402
from _stats import paired_bootstrap_ci                             # noqa: E402


def test_swap_runner_matches_an_independent_recomputation_on_a_fake_forward():
    import torch

    # row 0 (target): spans of length 2, 3, 5, 2 — the length-5 site (k=3) has no
    # length-5 donor window anywhere in row 1, so it must be skipped at g=1.
    span0 = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    # row 1 (donor for row 0, and its own target via the row+1 mod B pairing): longest
    # span is length 3, so it can satisfy a length-2 request but never a length-5 one.
    span1 = np.array([0, 0, 1, 1, 1, 2, -1, -1, -1, -1, -1, -1])
    inp = torch.from_numpy(np.stack([np.arange(0, 12), np.arange(100, 112)]))
    labels = torch.zeros_like(inp)

    def forward_ce(inp_t, labels_t, layout_, d):
        # id/100, plus a ROW-LEVEL offset (row sum / 1e5) so editing any position
        # changes CE everywhere in that row — a stand-in for "the model saw the edit
        # somewhere," which a purely position-local stub could not exercise.
        arr = inp_t.numpy().astype(np.float64)
        return arr / 100.0 + arr.sum(axis=1, keepdims=True) / 100000.0

    meta = [{"spans": np.stack([span0, span1]),
            "istok": np.ones((2, 12), dtype=bool),
            "row0": 0, "donor_local": np.array([1, 0])}]
    packed = [(inp, labels, None, None)]

    out = _swap(packed, meta, depths=[1], H=1, forward_ce=forward_ce,
               boot=paired_bootstrap_ci, n_rows=2, rng=np.random.default_rng(0),
               torch=torch, mode="row")

    assert out["mode"] == "row" and out["hops"] == 1
    # 3 sites: row0 k in {1, 3} (n_sp=4), row1 k in {1} (n_sp=3)
    assert out["n_sites"] == 3

    def own_mean_ce(row_ids: np.ndarray, span: np.ndarray, k: int) -> float:
        pos = np.flatnonzero(span == k)
        row_sum = row_ids.sum()
        return float((row_ids[pos] / 100.0 + row_sum / 100000.0).mean())

    expect_ctrl = np.mean([
        own_mean_ce(inp[0].numpy(), span0, 1),      # site A: row0 span 1
        own_mean_ce(inp[0].numpy(), span0, 3),      # site B: row0 span 3
        own_mean_ce(inp[1].numpy(), span1, 1),      # site C: row1 span 1
    ])
    assert out["control"]["1"] == pytest.approx(expect_ctrl)

    g1 = out["g1"]
    assert g1["n_sites"] == 2, "site B (length-5 source) has no donor window and must skip"
    assert g1["n_skipped"] == 1

    swapped_row0 = np.array([100, 101, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])   # site A's g=1 edit
    swapped_row1 = np.array([0, 1, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])  # C's
    expect_ce_A = own_mean_ce(swapped_row0, span0, 1)
    expect_ce_C = own_mean_ce(swapped_row1, span1, 1)
    assert g1["ce"]["1"] == pytest.approx((expect_ce_A + expect_ce_C) / 2)

    ctrl_A = own_mean_ce(inp[0].numpy(), span0, 1)
    ctrl_C = own_mean_ce(inp[1].numpy(), span1, 1)
    expect_benefit = ((expect_ce_A - ctrl_A) + (expect_ce_C - ctrl_C)) / 2
    assert g1["benefit"]["1"] == pytest.approx(expect_benefit)
    assert g1["frac_hurt"]["1"] == pytest.approx(0.5), \
        "site A's edit raises its row sum (hurt); site C's lowers it (helped)"
    assert "1" in g1["benefit_ci"]
    assert g1["benefit_ci"]["1"]["point"] == pytest.approx(expect_benefit)
