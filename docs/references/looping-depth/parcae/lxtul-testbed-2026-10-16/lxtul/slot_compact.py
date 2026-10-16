"""Stratified slot depths and the row moves of the compacted slot loop (keys `slot_depth_stratified`,
`slot_compact`; lxtul/model.py). Ported from MORPH's `morph/model/slot_compact.py`
(lab/experiments/successes/2026-10-08-slot-compaction-paired.md: no visible K1-K6 drop).

`slot_depth_stratified`: per row, the S slot depths are a SYSTEMATIC sample of D = clamp(Poisson(mean),
1, max): one U ~ Uniform[0,1) per row, sorted-sample slot k gets the quantile at (k + U) / S, then a
random permutation spreads the sample over the row's slots. Slot k sees (k + U) / S with k uniform, which
is Uniform[0,1), so each slot's marginal is exactly the clamped Poisson law. Only the joint law changes: a
row cannot draw many deep slots at once.

Capacity. Let G_t = S * P(D <= t). A sorted-sample slot is deeper than t iff x_k = fl(k + U) > G_t, and
x_k lies in [k, k + 1], so that needs k + 1 > G_t: at most S - floor(G_t) slots of a row are deeper than t,
whatever the rounding. Pad slots get depth 1, which only lowers the count. So pass t has at most
C_t = S - floor(G_t) active slots per row (64, 63, 61, 55, 46, 36, 26, 17 at S 64, mean 6, max 8), a
static number: the compacted pass computes C_t * M cells per row with no host sync.

`slot_compact` (the execution; function change in training only, see lxtul/model.py): a finished cell keeps
serving every core layer's attention with the keys and values of its last ACTIVE pass. "full" computes every
cell and keeps the active ones; "gather" computes only the C_t * M cells of the pass.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

SLOT_COMPACT_MODES = ("off", "full", "gather")


def clamped_poisson_cdf(mean: float, max_d: int) -> list[float]:
    """[P(D <= t) for t = 0 .. max_d], D = clamp(Poisson(mean), 1, max_d), float64."""
    out, acc, term = [], 0.0, math.exp(-float(mean))
    for t in range(max_d + 1):
        if t > 0:
            term *= float(mean) / t
        acc += term
        out.append(0.0 if t == 0 else (1.0 if t >= max_d else acc))
    return out


def stratified_thresholds(mean: float, max_d: int, n_slots: int) -> list[float]:
    """G_t = n_slots * P(D <= t) for t = 1 .. max_d - 1: what the draw compares k + U against."""
    cdf = clamped_poisson_cdf(mean, max_d)
    return [n_slots * cdf[t] for t in range(1, max_d)]


def slot_depth_capacity(mean: float, max_d: int, n_slots: int) -> tuple[int, ...]:
    """C_t, t = 0 .. max_d - 1: the most slots of a row the stratified draw leaves active at pass t."""
    g = [0.0] + stratified_thresholds(mean, max_d, n_slots)
    return tuple(n_slots - int(math.floor(x)) for x in g)


_THRESHOLDS: dict[tuple, Tensor] = {}


def stratified_slot_depths(n_rows: int, n_slots: int, mean: float, max_d: int, device) -> Tensor:
    """[n_rows, n_slots] int64 depths in [1, max_d]: per row a systematic sample of the clamped
    Poisson law, permuted over the slots. Two float64 draws from the global device stream (U and the
    permutation keys); no host sync (the thresholds are copied to the device once per shape)."""
    key = (int(n_slots), float(mean), int(max_d), str(torch.device(device)))
    g = _THRESHOLDS.get(key)
    if g is None:
        g = _THRESHOLDS[key] = torch.tensor(stratified_thresholds(mean, max_d, n_slots),
                                            dtype=torch.float64, device=device)
    u = torch.rand(n_rows, 1, dtype=torch.float64, device=device)
    x = torch.arange(n_slots, dtype=torch.float64, device=device) + u          # [R, S], sorted
    d = 1 + (x.unsqueeze(-1) > g).sum(-1)
    perm = torch.rand(n_rows, n_slots, dtype=torch.float64, device=device).argsort(dim=1)
    return d.gather(1, perm)


def compact_order(active: Tensor) -> Tensor:
    """[B, N] bool -> [B, N] int64: position at each rank, active first, then the rest, each in
    position order (a stable sort)."""
    return torch.argsort((~active).to(torch.int8), dim=1, stable=True)


def _view(idx: Tensor, like: Tensor) -> Tensor:
    """idx [B, n] broadcast to like's trailing dims, for gather/scatter on dim 1."""
    return idx.reshape(*idx.shape, *([1] * (like.dim() - 2))).expand(*idx.shape, *like.shape[2:])


def take_rows(x: Tensor, idx: Tensor) -> Tensor:
    """x [B, N, ...] -> [B, n, ...], the rows at idx [B, n] (distinct positions per row). The
    backward scatters into zeros, one write per row (distinct indices: deterministic)."""
    return x.gather(1, _view(idx, x))


def put_rows(src: Tensor, fill: Tensor, idx: Tensor, keep: Tensor) -> Tensor:
    """fill [B, N, ...] with the rows at idx [B, n] replaced by src [B, n, ...] where keep [B, N] is
    True (read at idx); rows of src whose keep is False leave fill as it was."""
    keep_c = keep.gather(1, idx).reshape(*idx.shape, *([1] * (src.dim() - 2)))
    src = torch.where(keep_c, src.to(fill.dtype), take_rows(fill, idx))
    return fill.scatter(1, _view(idx, src), src)
