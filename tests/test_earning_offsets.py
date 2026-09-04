"""Arc E0's offset-in-span convention has ONE meaning across the plain-row path
(`offsets_from_ids`, the boundary rule on input ids) and the packed-layout path
(`offsets_from_layout`, worth_profile's rule): a span starts after a boundary token,
offset 0 is its first input token. The packed row inserts slot positions, so the two
are compared as the SEQUENCE of offsets over token positions."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "lab/divergence")
from _earning import BINS, EarningProfile, bin_of, offsets_from_ids, offsets_from_layout  # noqa: E402

from morph.model.tul_layout import BoundaryRule, TulDataConfig, pack_tul_batch  # noqa: E402


def _rule(vocab: int = 64) -> BoundaryRule:
    lut = np.zeros(vocab, dtype=bool)
    lut[10] = True          # "." token
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def test_offsets_from_ids_restart_after_every_boundary():
    rule = _rule()
    ids = np.array([5, 6, 7, 10, 8, 9, 11, 12, 10, 13, 14], dtype=np.int64)
    #               0  1  2  3| 0  1  2   3   4| 0   1     ("." at 3 and 8; min_span 4)
    off = offsets_from_ids(ids, rule)
    assert off.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1]


def test_offsets_from_ids_min_span_suppresses_early_boundary():
    rule = _rule()
    ids = np.array([5, 10, 6, 7, 10, 8], dtype=np.int64)   # "." at 1 is inside min_span
    off = offsets_from_ids(ids, rule)
    assert off.tolist() == [0, 1, 2, 3, 4, 0]


def test_layout_and_ids_agree_on_token_offsets():
    rule = _rule()
    rng = np.random.default_rng(0)
    toks = rng.integers(11, 40, size=400)          # never eos (0), '.' (10) or slot_id (4)
    toks[rng.random(400) < 0.15] = 10
    spec = TulDataConfig(rule=rule, prefix_k=2, slot_id=4).spec_for(96)
    inp, labels, layout = pack_tul_batch(toks.tolist(), rule, spec, batch_size=1)
    off_layout = offsets_from_layout(layout, 0)
    tok_mask = ~layout.slot_mask[0].cpu().numpy()
    ids_row = inp[0].cpu().numpy()[tok_mask]
    off_ids = offsets_from_ids(ids_row, rule)
    got = off_layout[tok_mask]
    keep = got >= 0                       # the layout marks the first span / dump bin −1
    assert keep.sum() > 10
    assert np.array_equal(got[keep], off_ids[keep])


def test_bins_cover_every_offset_once():
    for o in range(0, 64):
        assert BINS[bin_of(o)][0] <= o <= BINS[bin_of(o)][1]


def test_profile_counts_tokens_once_and_sums_per_depth():
    import torch
    prof = EarningProfile([1, 2], n_rows=1)
    ce1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ce2 = torch.tensor([0.5, 1.0, 1.5, 2.0])
    valid = torch.tensor([True, True, False, True])
    off = np.array([0, 1, 2, 3])
    prof.add(1, 0, ce1, valid, off)
    prof.add(2, 0, ce2, valid, off)
    j = prof.to_json()
    assert j["row_n_tokens"] == [3.0]
    assert j["row_ce_sum"]["1"] == [7.0] and j["row_ce_sum"]["2"] == [3.5]
    assert j["bin_n_tokens"][0][:4] == [1.0, 1.0, 0.0, 1.0]
    assert j["bin_ce_sum"]["1"][0][:4] == [1.0, 2.0, 0.0, 4.0]
