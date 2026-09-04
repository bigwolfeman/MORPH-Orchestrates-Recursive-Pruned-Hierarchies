"""Per-row and per-offset accumulation of depth earning (arc E0,
lab/experiments/planned/2026-09-04-arc-e0-where-depth-earns.md).

ONE home for the offset-in-span bins (worth_profile.py imports them from here) and
for the two ways a token gets its offset: from a packed `SlotLayout` (TUL rows) or
from the boundary rule run over a plain row's input ids (no-TUL rows). Both give the
SAME convention: a span starts at the position after a boundary token, offset 0 is
the first input token of the span, and the CE at a position is the CE of predicting
that position's label.
"""
from __future__ import annotations

import numpy as np
import torch

BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 10 ** 9)]


def bin_of(off: int) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= off <= hi:
            return i
    raise AssertionError(off)


def offsets_from_ids(ids: np.ndarray, rule) -> np.ndarray:
    """Offset-in-span of every position of a plain row (all positions scoreable)."""
    n = int(ids.shape[0])
    cuts, _ = rule.cut(ids, 0)
    off = np.empty(n, dtype=np.int64)
    start = 0
    for p in cuts.tolist():
        off[start:p + 1] = np.arange(p + 1 - start)
        start = p + 1
    off[start:n] = np.arange(n - start)
    return off


def offsets_from_layout(layout, b: int) -> np.ndarray:
    """Offset-in-span per position of packed row ``b``; −1 where the position is a
    slot, in the first span (no preceding slot) or in the dump bin — the same rule as
    worth_profile.token_strata."""
    L = int(layout.slot_mask.shape[1])
    dump = int(layout.slot_valid.shape[1])
    slot = layout.slot_mask[b].cpu().numpy()
    bag = layout.bag_id[b].cpu().numpy()
    off = np.full(L, -1, dtype=np.int64)
    counts: dict[int, int] = {}
    for p in range(L):
        if slot[p]:
            continue
        k = int(bag[p])
        o = counts.get(k, 0)
        counts[k] = o + 1
        if k == 0 or k >= dump:
            continue
        off[p] = o
    return off


class EarningProfile:
    """Accumulates, per row: CE sum per depth, token count, CE sum per (depth, bin),
    token count per bin. ``add(depth, row_index, ce[L], valid[L], offsets[L])``."""

    def __init__(self, depths: list[int], n_rows: int):
        self.depths = list(depths)
        self.row_ce = {d: np.zeros(n_rows) for d in depths}
        self.row_n = np.zeros(n_rows)
        self.bin_ce = {d: np.zeros((n_rows, len(BINS))) for d in depths}
        self.bin_n = np.zeros((n_rows, len(BINS)))
        self._counted: set[int] = set()

    def add(self, d: int, r: int, ce: torch.Tensor, valid: torch.Tensor, off: np.ndarray):
        ce = ce.detach().float().cpu().numpy()
        v = valid.cpu().numpy().astype(bool)
        self.row_ce[d][r] += float(ce[v].sum())
        first = r not in self._counted
        if first:
            self.row_n[r] += float(v.sum())
        for i, (lo, hi) in enumerate(BINS):
            m = v & (off >= lo) & (off <= hi)
            self.bin_ce[d][r, i] += float(ce[m].sum())
            if first:
                self.bin_n[r, i] += float(m.sum())
        self._counted.add(r)

    def to_json(self) -> dict:
        return {
            "bins": [list(b) for b in BINS],
            "row_ce_sum": {str(d): self.row_ce[d].tolist() for d in self.depths},
            "row_n_tokens": self.row_n.tolist(),
            "bin_ce_sum": {str(d): self.bin_ce[d].tolist() for d in self.depths},
            "bin_n_tokens": self.bin_n.tolist(),
        }
