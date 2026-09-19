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
