"""Hop-distance probe: does the slot loop earn depth only where the content is HOPS away?

THE QUESTION. Twelve slot-loop arms read K1−K6 ≈ +0.002 nats while the plain looped model
reads 0.033 (`.claude/.../slot-loop-k-curve-yardstick`). Every stability lever is on both,
so the flat K-curve is not a stability story. The standing hypothesis this probe tests: a
looped transformer earns depth through GATHER — pass k reads pass k−1 states of OTHER
positions, one more attention hop per pass — and under the strict geometry most of what a
token needs is already ONE hop away, so no pass after the first can add anything.

The geometry, from `morph/model/tul_layout.py::tg_strict_allow`:

    prelude  a token sees its own span's tokens; a cell sees its own span and its own
             earlier cells. A cell's seed summarises ONE span.
    loop     cell k at pass t reads cells k−`loop_reach` … k at pass t−1 (`loop_reach` 0 =
             every earlier cell).
    coda     a token of span j reads its own span plus the prefix cells of earlier slots —
             ALL of them (`tg_coda_prefix_reach: all`) or only cell j−1 (`prev`).

So on the shipped strict arm (`all` + `loop_reach` 0) span i's content reaches a token of
span j in ONE hop at any depth: the hypothesis PREDICTS a flat hop profile there, and that
arm is the null control, not the test. The test is `prev` + `loop_reach` 1, where span j−h
reaches cell j−1 only at pass h−1, so depth is required BY CONSTRUCTION and the earning
must appear in the far hop bins and plateau at d ≈ h−1.

TWO INSTRUMENTS, deliberately not one.

1. THE CORRUPTION LOCALISER (default). For c = 0…H one forward replaces the tokens of
   every span i with i ≡ c (mod H+1) by tokens taken from another row of the batch, the
   LAYOUT held fixed (`slot_layout` is a forward argument, so the boundaries, the cells and
   every mask are byte-identical to the clean forward). A token of span j is scored in the
   H forwards that do not corrupt its own span, and in the forward with phase c its nearest
   corrupted span sits at distance (j − c) mod (H+1) — so across the H+1 forwards each
   token gets exactly one reading at each distance 1…H. ΔCE(h) = CE at that reading minus
   the clean CE at the same position and the same depth; h* = argmax_h ΔCE(h) is the hop
   distance of the span that token most needs. Then the clean per-token K-curve is binned
   by h*.

   Named honestly, three limits of the localiser:
   * phase c corrupts EVERY span at distance ≡ h (mod H+1), so ΔCE(h) is the effect of
     corrupting distances h, h+H+1, h+2(H+1) … together, not h alone. At H 6 and a mean
     span of ~12 tokens the confound sits ≥ 7 spans away.
   * distance 0 (the token's own span) cannot be read this way at all — corrupting it
     replaces the scored token itself. It gets its OWN forward, which corrupts the first
     half of every span and scores only the second half, and it is reported separately and
     kept OUT of the argmax.
   * a token whose label is the first token of a corrupted span is dropped from that
     phase's reading (its target changed, not just its context), and so is a token of span
     j at a distance h > j — the phase corrupted nothing the token could causally read, so
     that reading would be a spurious ΔCE of 0. Only tokens with a finite reading at every
     distance 1…H are binned, which means span index >= H.

2. THE PLANTED COPY PAIR (`--planted`). No localiser, no assumption about what a token
   needs: a token id absent from the whole batch is written at a random position of span
   s−g AND as the first token of span s+1, and the CE of predicting that repeat — read at
   the LAST token of span s, the position whose label it is — is measured against (g, d).
   The scoring positions are IDENTICAL for every g and for the no-source control, so the
   copy benefit CE_control − CE_source(g, d) is paired at the token level. g = 0 plants the
   source inside span s itself; that is the instrument's positive control, and if it shows
   no benefit the model cannot copy at all and no larger g can say anything.

Depth forcing is `core_depth_sweep.py`'s: `tul.slot_mean_depth` (and `slot_max_depth`, and
`slot_depth_fixed` on a k-fixed arm) between evals. Depth 0 is NOT valid on a slot arm —
`_sample_slot_depths` reads `tc.slot_mean_depth or self.cfg.mean_depth`, so 0 silently
becomes the trained mean — and is refused here. The CE map comes from
`core_depth_sweep.ce_maps`, i.e. from `model.tul_forward_ablated`, so every TG mask is the
model's own; a bare `_tul_front` would score a strict arm from an UNRESTRICTED prelude and
flip the sign of the reading (measured 2026-09-13).

Usage:
  PYTHONPATH=. python lab/divergence/hop_distance_probe.py \
    --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
    --rows 24 --batch 4 --depths 1,2,3,6,9,12,16 --hops 6 --planted \
    --out lab/experiments/results/2026-09-18-hop-distance-probe/hop_strict_5000.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = [
    "corrupt_mask_for_phase",
    "donor_replacement",
    "hop_before_row_start",
    "hop_of",
    "hops_scored_by_token",
    "hops_seen_by_token",
    "next_token_corrupted",
    "own_prefix_mask",
    "plant_sites",
    "planted_triple",
    "source_positions",
    "reach0_kw",
    "install_reach_cut",
    "source_position",
    "span_index_from_layout",
    "bin_sums",
    "HopTable",
]


# ── pure logic: the corruption schedule ────────────────────────────────────────────

def span_index_from_layout(bag_id_row: np.ndarray, slot_mask_row: np.ndarray,
                           max_slots: int, n_slots: int) -> np.ndarray:
    """``[L]`` span index of every TOKEN position; −1 at slot positions and tail pads.

    `pack_tul_row` gives a token the index of the slot that TERMINATES its span, and the
    dump bin `max_slots` to the tokens after the last boundary. Those tokens are a real
    span (the row's open tail) with no cell of its own, so they get index `n_slots` — one
    past the last slot — which keeps the distance arithmetic contiguous.
    """
    sp = np.asarray(bag_id_row, dtype=np.int64).copy()
    sp[sp == max_slots] = n_slots
    sp[np.asarray(slot_mask_row, dtype=bool)] = -1
    return sp


def hop_of(span_idx: np.ndarray, c: int, hops: int) -> np.ndarray:
    """Distance from each token's span to the NEAREST span corrupted in phase ``c``.

    Phase ``c`` corrupts every span i with ``i % (hops+1) == c``, so the nearest corrupted
    span at or before span j sits at ``(j - c) % (hops+1)`` spans back. 0 means the token's
    own span is the corrupted one.
    """
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")
    out = np.full(np.shape(span_idx), -1, dtype=np.int64)
    valid = np.asarray(span_idx) >= 0
    out[valid] = (np.asarray(span_idx)[valid] - c) % (hops + 1)
    return out


def corrupt_mask_for_phase(span_idx: np.ndarray, c: int, hops: int) -> np.ndarray:
    """``[L]`` bool — token positions whose span is corrupted in phase ``c``."""
    return hop_of(span_idx, c, hops) == 0


def hops_seen_by_token(span: int, hops: int) -> list[int]:
    """The distances one token of span ``span`` is scored at, over phases 0…hops.

    The contract the schedule exists for: exactly one reading at each of 1…hops, and none
    at 0 (phase ``span % (hops+1)`` corrupts the token itself, so it is not scored).
    """
    seen = []
    for c in range(hops + 1):
        h = (span - c) % (hops + 1)
        if h != 0:
            seen.append(h)
    return sorted(seen)


def hop_before_row_start(span_idx: np.ndarray, hop: np.ndarray) -> np.ndarray:
    """``[L]`` bool — the span at that distance does not exist in this row.

    THE WRAP-AROUND BUG this exists to kill. Phase ``c`` corrupts every span
    ``i ≡ c (mod hops+1)``, so a token of span j is assigned distance ``(j-c) % (hops+1)``.
    For ``j < h`` the span ``j-h`` is not in the row: every span the phase corrupted sits
    AFTER the token, where a causal model cannot read it, so that reading is ΔCE ≈ 0 for a
    reason that has nothing to do with what the token needs. Left in, it drags h* DOWN for
    every token of the first H spans of a row. Marked invalid here instead, which makes
    ``complete`` (a finite reading at every distance 1…H) require ``j >= H`` by itself.
    """
    span_idx = np.asarray(span_idx, dtype=np.int64)
    hop = np.asarray(hop, dtype=np.int64)
    return (span_idx >= 0) & (hop >= 1) & ((span_idx - hop) < 0)


def hops_scored_by_token(span: int, hops: int) -> list[int]:
    """The distances a token of span ``span`` actually contributes a reading at.

    :func:`hops_seen_by_token` is the raw schedule; this is the schedule AFTER the
    wrap-around drop, so a token of span 2 at H 6 contributes only at distances 1 and 2.
    """
    return [h for h in hops_seen_by_token(span, hops) if span - h >= 0]


def next_token_corrupted(is_token: np.ndarray, corrupt: np.ndarray) -> np.ndarray:
    """``[L]`` bool — True at a token position whose LABEL is a corrupted token.

    A row's labels are next-token over the TOKEN positions only (the slot positions in
    between carry their own emit label), so the label of token position p is the id at the
    next token position. Such a position is dropped from the phase: its target moved, and a
    CE change there is not a context effect.
    """
    corrupt = np.asarray(corrupt, dtype=bool)
    out = np.zeros_like(corrupt)
    pos = np.flatnonzero(np.asarray(is_token, dtype=bool))
    if pos.size > 1:
        out[pos[:-1]] = corrupt[pos[1:]]
    return out


def own_prefix_mask(span_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance-0 forward: ``(corrupt, scored)`` — corrupt each span's first half.

    Returns the positions to replace (the first ``ceil(len/2)`` tokens of every span) and
    the positions to score (the rest of the span). The scored tokens keep their own input
    and their own label; only their within-span context moved.
    """
    span_idx = np.asarray(span_idx, dtype=np.int64)
    corrupt = np.zeros(span_idx.shape, dtype=bool)
    scored = np.zeros(span_idx.shape, dtype=bool)
    for s in np.unique(span_idx[span_idx >= 0]):
        pos = np.flatnonzero(span_idx == s)
        if pos.size < 2:
            continue
        half = (pos.size + 1) // 2
        corrupt[pos[:half]] = True
        scored[pos[half:]] = True
    return corrupt, scored


def donor_replacement(donor_ids: np.ndarray, n_positions: int) -> np.ndarray:
    """Replacement ids for ``n_positions`` corrupted token positions, in row order.

    The donor's own token ids, cycled. Taking them from the donor's TOKEN positions (never
    from its layout positions) is what keeps `slot_id` — which `pack_tul_row` refuses to
    see in a token stream — out of the corrupted row.
    """
    donor_ids = np.asarray(donor_ids, dtype=np.int64)
    if donor_ids.size == 0:
        raise ValueError("donor row has no token positions")
    return donor_ids[np.arange(n_positions) % donor_ids.size]


# ── pure logic: the planted copy pair ──────────────────────────────────────────────

def plant_sites(n_slots: int, hops: int, phase: int = 0) -> list[int]:
    """Scoring spans for the planted probe, spaced so the plants cannot overlap.

    A site ``s`` owns spans ``s-hops … s+1``; consecutive sites are ``hops+2`` apart, so
    the next site's window starts at ``s+2`` and the two never touch. The SAME sites serve
    every source distance g and the no-source control, which is what makes the copy benefit
    paired at the token level.
    """
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")
    step = hops + 2
    first = hops + (phase % step)
    return [s for s in range(first, n_slots - 1, step)]


def planted_triple(span_idx: np.ndarray, s: int) -> tuple[int, int]:
    """``(scored_pos, repeat_pos)`` for scoring span ``s``.

    ``repeat_pos`` is the FIRST token position of span ``s+1`` — the position whose id is
    overwritten with the rare token. ``scored_pos`` is the LAST token position of span
    ``s``, the position whose label is that id (`pack_tul_row`: ``labels[tok_pos] =
    ids[i+1]``). The slot's own emit position carries the same label, but the token head is
    the trained one (`emit_source="token"`, `emit_weight` 0), so the token position is what
    is scored.
    """
    span_idx = np.asarray(span_idx, dtype=np.int64)
    cur = np.flatnonzero(span_idx == s)
    nxt = np.flatnonzero(span_idx == s + 1)
    if cur.size == 0 or nxt.size == 0:
        raise ValueError(f"span {s} or {s + 1} has no token position in this row")
    return int(cur[-1]), int(nxt[0])


def source_positions(span_idx: np.ndarray, s: int, g: int, rng, n: int = 1) -> list[int]:
    """``n`` CONSECUTIVE token positions inside span ``s-g``, chosen by ``rng``.

    ``g == 0`` puts the source inside the scoring span itself and excludes that span's LAST
    token, which is the scored position: overwriting it would change the probe's own input.
    With ``n == 2`` the pair is ``(cue, X)``; the scored position then carries the cue in
    every condition and the source pair is what the planted condition adds, so the benefit
    is an induction copy (cue -> X) rather than a bare "seen once" boost.
    """
    if g < 0:
        raise ValueError(f"g must be >= 0, got {g}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    span_idx = np.asarray(span_idx, dtype=np.int64)
    pos = np.flatnonzero(span_idx == s - g)
    if g == 0:
        pos = pos[:-1]
    # consecutive: every run of n adjacent positions inside the span
    starts = [int(pos[i]) for i in range(pos.size - n + 1)
              if int(pos[i + n - 1]) - int(pos[i]) == n - 1]
    if not starts:
        raise ValueError(f"span {s - g} has no run of {n} usable token positions in this row")
    p0 = starts[rng.integers(0, len(starts))]
    return [p0 + k for k in range(n)]


def source_position(span_idx: np.ndarray, s: int, g: int, rng) -> int:
    """The ``n == 1`` form of `source_positions`."""
    return source_positions(span_idx, s, g, rng, 1)[0]


# ── pure logic: aggregation ────────────────────────────────────────────────────────

def bin_sums(values: np.ndarray, row: np.ndarray, keep: np.ndarray,
             n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(sum, count)`` of ``values`` over the tokens in ``keep``.

    The unit of the paired bootstrap is the ROW, so a bin's interval resamples rows and a
    row that holds no token of the bin contributes an honest zero of both.
    """
    keep = np.asarray(keep, dtype=bool)
    row = np.asarray(row, dtype=np.int64)
    s = np.bincount(row[keep], weights=np.asarray(values, dtype=np.float64)[keep],
                    minlength=n_rows)
    n = np.bincount(row[keep], minlength=n_rows).astype(np.float64)
    return s[:n_rows], n[:n_rows]


class HopTable:
    """Per-hop-bin CE curves with paired row bootstraps.

    ``hstar`` is one bin index per scored token, ``row`` its row, and ``ce`` the per-token
    clean CE at each forced depth on the SAME tokens — so every bin's K-curve is paired.
    """

    def __init__(self, hstar: np.ndarray, row: np.ndarray, n_rows: int):
        self.hstar = np.asarray(hstar, dtype=np.int64)
        self.row = np.asarray(row, dtype=np.int64)
        self.n_rows = int(n_rows)
        if self.hstar.shape != self.row.shape:
            raise ValueError("hstar and row must have the same shape")

    def bins(self) -> list[int]:
        return sorted(int(b) for b in np.unique(self.hstar) if b >= 0)

    def table(self, ce: dict[int, np.ndarray], pairs: list[tuple[int, int]],
              boot) -> dict[str, dict]:
        out: dict[str, dict] = {}
        depths = sorted(ce)
        for b in self.bins():
            keep = self.hstar == b
            entry: dict[str, object] = {"n_tokens": int(keep.sum()), "ce": {}}
            for d in depths:
                s, n = bin_sums(ce[d], self.row, keep, self.n_rows)
                entry["ce"][str(d)] = float(s.sum() / n.sum()) if n.sum() else float("nan")
            for a, z in pairs:
                if a in ce and z in ce and a != z:
                    sa, n = bin_sums(ce[a], self.row, keep, self.n_rows)
                    sz, _ = bin_sums(ce[z], self.row, keep, self.n_rows)
                    if n.sum() > 0:
                        entry[f"K{a}-K{z}"] = boot(sa, sz, n)
            out[str(b)] = entry
        return out


# ── the run ────────────────────────────────────────────────────────────────────────

def reach0_kw(kw: dict) -> dict:
    """The reach-0 form of one core layer's reach kwargs: a cell reads ITSELF only.

    Built from the layer's own ``tg_allow`` (so shape, device and ``tg_seg`` travel with
    it) rather than from the model, which keeps this a pure function of one dict.
    """
    if "tg_allow" not in kw:
        raise ValueError("reach0_kw needs the loop_reach kwargs (tg_allow); a register arm "
                         "(slot_cells > 1) carries tg_relation and is not cut here")
    import torch
    m = kw["tg_allow"]
    S = m.shape[-1]
    eye = torch.eye(S, dtype=torch.bool, device=m.device).view(1, 1, S, S)
    out = dict(kw)
    out["tg_allow"] = eye
    out["tg_comp_allow"] = eye
    return out


def install_reach_cut(model, cut_after: int):
    """Passes with 0-based index >= ``cut_after`` read no other cell (reach 0 on EVERY core
    layer); passes before it keep the arm's reach. Returns ``restore``.

    Wraps ``model._apply_core_step``, the one call every core pass goes through
    (``_core_step`` in `_tul_core` hands it ``iter_idx=t`` and ``attn_kw=_core_akw``).
    With ``cut_after >= max depth`` nothing changes and the forward is bit-identical, which
    `tests/test_hop_distance_probe.py` asserts.
    """
    if cut_after < 0:
        raise ValueError(f"cut_after must be >= 0, got {cut_after}")
    real = model._apply_core_step
    cache: dict[int, tuple] = {}

    def cut_step(*args, iter_idx=0, attn_kw=None, **kw):
        if attn_kw is not None and int(iter_idx) >= cut_after:
            key = id(attn_kw)
            if key not in cache:
                cache[key] = tuple(reach0_kw(k) for k in attn_kw)
            attn_kw = cache[key]
        return real(*args, iter_idx=iter_idx, attn_kw=attn_kw, **kw)

    model._apply_core_step = cut_step

    def restore() -> None:
        model._apply_core_step = real

    return restore


def _depth_forcer(model, tc):
    """Set/restore the slot loop's forced eval depth. Slot-loop arms only."""
    orig = (int(tc.slot_mean_depth), int(tc.slot_max_depth),
            int(getattr(tc, "slot_depth_fixed", 0)), int(model.cfg.max_depth))

    def set_depth(d: int) -> None:
        if d < 1:
            raise ValueError(
                f"depth {d} is not valid on a slot arm: _sample_slot_depths reads "
                f"`tc.slot_mean_depth or self.cfg.mean_depth`, so 0 silently becomes the "
                f"trained mean and the sweep would print a plausible lie")
        tc.slot_mean_depth = d
        tc.slot_max_depth = max(d, orig[1] or int(model.cfg.max_depth))
        if orig[2] > 0:
            tc.slot_depth_fixed = d

    def restore() -> None:
        tc.slot_mean_depth, tc.slot_max_depth = orig[0], orig[1]
        if orig[2] > 0:
            tc.slot_depth_fixed = orig[2]

    return set_depth, restore


def _row_spans(layout, b: int, max_slots: int) -> np.ndarray:
    n_slots = int(layout.slot_valid[b].sum())
    return span_index_from_layout(layout.bag_id[b].cpu().numpy(),
                                  layout.slot_mask[b].cpu().numpy(), max_slots, n_slots)


def _rare_ids(n: int, used: set[int], is_boundary: np.ndarray, slot_id: int,
              vocab: int, rng) -> list[int]:
    """``n`` distinct token ids absent from the batch, never boundary, never `slot_id`.

    Absent from the batch is what makes the planted repeat a COPY task: the only place the
    model can have seen the id is the source we planted.
    """
    out: list[int] = []
    seen = set(used)
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 200 * n + 10000:
            raise RuntimeError(f"could not find {n} rare ids absent from the batch")
        t = int(rng.integers(1000, vocab))
        if t in seen or t == slot_id:
            continue
        if t < is_boundary.shape[0] and bool(is_boundary[t]):
            continue
        seen.add(t)
        out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--depths", default="1,2,3,6,9,12,16")
    ap.add_argument("--hops", type=int, default=6, help="H: distances 1..H are localised")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planted", action="store_true", help="also run the planted copy pair")
    ap.add_argument("--planted-len", type=int, default=1, choices=(1, 2),
                    help="1: a rare id seen once at the source; 2: a (cue, X) pair at the "
                         "source with the cue at the scored position (induction copy)")
    ap.add_argument("--cut-after", type=int, default=-1,
                    help="passes with 0-based index >= this read no other cell (reach 0). "
                         "-1: off. Needs tul.loop_reach > 0")
    ap.add_argument("--skip-localiser", action="store_true",
                    help="planted probe only (no corruption forwards, no h* bins)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from _build import ROOT, build_cfg, uses_sample_depth
    from _rows import pack_rows, stream_from_loader
    from _stats import paired_bootstrap_ci
    from core_depth_sweep import ce_maps, warn_if_frozen_reader

    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    device = a.device
    depths = sorted({int(x) for x in a.depths.split(",")})
    if min(depths) < 1:
        raise SystemExit("depth 0 is not valid on a slot arm (see --help / the docstring)")
    H = int(a.hops)
    rng = np.random.default_rng(a.seed)

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise SystemExit("this probe needs a TUL arm: a plain model has no spans and no "
                         "cells, so hop distance is undefined on it")
    if uses_sample_depth(tul_rt.model_cfg):
        raise SystemExit("this arm runs tokens through the core (`tokens_through_core` or "
                         "`loop_reads_tokens`). Its loop is not the slot chain this probe "
                         "measures hops along, and its forced-depth knob is "
                         "`model.cfg.mean_depth`, not `tul.slot_mean_depth`")
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, tul_rt.model_cfg)
    model.eval()
    warn_if_frozen_reader(cfg, label)
    tc = model.cfg.tul
    set_depth, restore = _depth_forcer(model, tc)
    if a.cut_after >= 0:
        if int(getattr(tc, "loop_reach", 0)) <= 0:
            raise SystemExit("--cut-after needs tul.loop_reach > 0: at loop_reach 0 the "
                             "core builds no reach kwargs and there is nothing to cut")
        _restore_cut = install_reach_cut(model, a.cut_after)
        print(f"[arm] reach cut: passes with index >= {a.cut_after} read no other cell",
              flush=True)
    train_depth = int(getattr(tc, "slot_depth_fixed", 0) or tc.slot_mean_depth
                      or cfg.model.mean_depth)
    if train_depth not in depths:
        depths = sorted(depths + [train_depth])

    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, a.rows * (spec.l_total + 1))
    n_batches = -(-a.rows // a.batch)
    packed = pack_rows(stream, tul_rt, cfg, a.batch, plain=False)[:n_batches]
    rows_done = sum(inp.shape[0] for inp, _, _, _ in packed)

    arm = {
        "label": label, "config": config, "ckpt": path, "step": step,
        "rows": rows_done, "batch": a.batch, "hops": H, "depths": depths,
        "train_depth": train_depth, "seed": a.seed,
        # the hypothesis predicts DIFFERENT curves for reach=all and reach=prev, so the
        # reach travels with the reading and a scorer cannot read the two as one number
        "tg_geometry": str(getattr(tc, "tg_geometry", "none")),
        "tg_restrict": bool(getattr(tc, "tg_restrict", False)),
        "tg_coda_prefix_reach": str(getattr(tc, "tg_coda_prefix_reach", "all")),
        "loop_reach": int(getattr(tc, "loop_reach", 0)),
        "spandec": bool(getattr(tc, "spandec", False)),
        "slot_seed": str(getattr(tc, "slot_seed", "")),
        "cut_after": int(a.cut_after),
        "planted_len": int(a.planted_len),
    }
    print(f"[arm] {label} step={step} rows={rows_done} geometry={arm['tg_geometry']} "
          f"coda_reach={arm['tg_coda_prefix_reach']} loop_reach={arm['loop_reach']} "
          f"train_depth={train_depth}", flush=True)

    # per-row span maps and token masks — computed once on the CPU layout, then the
    # layouts move to the device ONCE and every forward reuses them (a `.to` per forward
    # would copy two [B, L] int64 tensors per depth per phase for nothing).
    meta = []
    row0 = 0
    for inp, labels, layout, idx in packed:
        B, L = inp.shape
        if B < 2:
            raise SystemExit("--batch must be >= 2: the corruption donor is another row "
                             "of the same batch")
        spans = np.stack([_row_spans(layout, b, spec.max_slots) for b in range(B)])
        istok = (~layout.slot_mask).numpy() & (labels.numpy() >= 0)
        meta.append({"spans": spans, "istok": istok, "row0": row0,
                     "donor_local": (np.arange(B) + 1) % B})
        row0 += B
    n_rows = row0
    packed = [(inp, labels, layout.to(device), idx) for inp, labels, layout, idx in packed]

    def forward_ce(inp, labels, layout, d):
        set_depth(d)
        ce, _ = ce_maps(model, inp, layout, labels, device, want_mux=False)
        return ce.float().cpu().numpy()

    results: dict[str, dict] = {label: arm}
    npz_arrays: dict[str, np.ndarray] = {}

    try:
        if not a.skip_localiser:
            # ── clean CE at every depth, kept per token ────────────────────────────
            tok_row, tok_index, tok_span = [], [], []
            for (inp, labels, layout, idx), m in zip(packed, meta):
                keep = m["istok"]
                B = inp.shape[0]
                tok_row.append(np.repeat(np.arange(B) + m["row0"], keep.sum(axis=1)))
                tok_index.append(idx.numpy()[keep])
                tok_span.append(m["spans"][keep])
            tok_row = np.concatenate(tok_row).astype(np.int64)
            tok_index = np.concatenate(tok_index).astype(np.int64)
            tok_span = np.concatenate(tok_span).astype(np.int64)

            ce_clean: dict[int, np.ndarray] = {}
            for d in depths:
                acc = []
                for (inp, labels, layout, _), m in zip(packed, meta):
                    ce = forward_ce(inp, labels, layout, d)
                    acc.append(ce[m["istok"]])
                ce_clean[d] = np.concatenate(acc).astype(np.float32)
                print(f"  clean depth={d:>2d}  ce={ce_clean[d].mean():.4f}", flush=True)

            # ── the H+1 corruption phases, at the training depth ───────────────────
            n_tok = tok_row.shape[0]
            dce = np.full((H + 1, n_tok), np.nan, dtype=np.float32)   # row 0 = own span
            for c in range(H + 1):
                acc_d, acc_h = [], []
                for (inp, labels, layout, _), m in zip(packed, meta):
                    B, L = inp.shape
                    cor = np.zeros((B, L), dtype=bool)
                    ci = inp.numpy().copy()
                    for b in range(B):
                        cm = corrupt_mask_for_phase(m["spans"][b], c, H) & m["istok"][b]
                        cor[b] = cm
                        dn = m["donor_local"][b]
                        donor_ids = inp.numpy()[dn][m["istok"][dn]]
                        ci[b, cm] = donor_replacement(donor_ids, int(cm.sum()))
                    ce = forward_ce(torch.from_numpy(ci), labels, layout,
                                    train_depth)
                    for b in range(B):
                        hb = hop_of(m["spans"][b], c, H)
                        # corrupted itself | its label moved | the span at that distance is
                        # not in the row (the wrap-around, see hop_before_row_start)
                        bad = (cor[b] | next_token_corrupted(m["istok"][b], cor[b])
                               | hop_before_row_start(m["spans"][b], hb))
                        keep = m["istok"][b]
                        acc_d.append(np.where(bad[keep], np.nan, ce[b][keep]))
                        acc_h.append(np.where(bad[keep], -1, hb[keep]))
                cc = np.concatenate(acc_d)
                hh = np.concatenate(acc_h)
                for h in range(1, H + 1):
                    sel = hh == h
                    dce[h, sel] = cc[sel] - ce_clean[train_depth][sel]
                print(f"  phase c={c}  scored={int((hh > 0).sum())}/{n_tok}", flush=True)

            # ── distance 0: corrupt each span's first half, score the second ───────
            acc_d, acc_m = [], []
            for (inp, labels, layout, _), m in zip(packed, meta):
                B, L = inp.shape
                ci = inp.numpy().copy()
                sc = np.zeros((B, L), dtype=bool)
                for b in range(B):
                    cm, sm = own_prefix_mask(np.where(m["istok"][b], m["spans"][b], -1))
                    dn = m["donor_local"][b]
                    donor_ids = inp.numpy()[dn][m["istok"][dn]]
                    ci[b, cm] = donor_replacement(donor_ids, int(cm.sum()))
                    sc[b] = sm & ~next_token_corrupted(m["istok"][b], cm)
                ce = forward_ce(torch.from_numpy(ci), labels, layout, train_depth)
                for b in range(B):
                    keep = m["istok"][b]
                    acc_d.append(np.where(sc[b][keep], ce[b][keep], np.nan))
                    acc_m.append(sc[b][keep])
            own = np.concatenate(acc_d) - ce_clean[train_depth]
            own[~np.concatenate(acc_m)] = np.nan
            dce[0] = own
            print(f"  phase own  scored={int(np.isfinite(own).sum())}/{n_tok}", flush=True)

            # ── h*, the bins and the table ─────────────────────────────────────────
            far = dce[1:]                       # distances 1..H only; 0 is reported apart
            complete = np.isfinite(far).all(axis=0)
            hstar = np.full(n_tok, -1, dtype=np.int64)
            hstar[complete] = np.nanargmax(far[:, complete], axis=0) + 1
            tab = HopTable(hstar, tok_row, n_rows)
            pairs = [(1, 6), (1, 3), (1, max(depths))]
            arm["bins"] = tab.table({d: ce_clean[d] for d in depths}, pairs,
                                    paired_bootstrap_ci)
            arm["n_tokens_complete"] = int(complete.sum())
            arm["n_tokens_scored"] = int(n_tok)
            # a token needs a finite reading at every distance 1..H to be binned, and the
            # wrap-around drop alone makes that require span index >= H
            arm["complete_frac"] = float(complete.sum() / n_tok) if n_tok else float("nan")
            print(f"  complete tokens: {int(complete.sum())}/{n_tok} "
                  f"({100.0 * complete.sum() / max(n_tok, 1):.1f} %) — a token is binned "
                  f"only with a finite reading at every distance 1..{H}", flush=True)
            arm["dce_profile"] = {str(h): float(np.nanmean(dce[h]))
                                  for h in range(0, H + 1)}
            arm["dce_profile_n"] = {str(h): int(np.isfinite(dce[h]).sum())
                                    for h in range(0, H + 1)}
            arm["all_tokens"] = {}
            allkeep = np.ones(n_tok, dtype=bool)
            for d in depths:
                s, n = bin_sums(ce_clean[d], tok_row, allkeep, n_rows)
                arm["all_tokens"][str(d)] = float(s.sum() / n.sum())
            npz_arrays.update({"tok_index": tok_index.astype(np.int32),
                               "tok_row": tok_row.astype(np.int32),
                               "tok_span": tok_span.astype(np.int32),
                               "hstar": hstar.astype(np.int16), "dce": dce,
                               **{f"ce_{d}": ce_clean[d] for d in depths}})

        if a.planted:
            arm["planted"] = _planted(packed, meta, spec, tul_rt, cfg, depths, H,
                                      forward_ce, paired_bootstrap_ci, n_rows, rng, torch,
                                      planted_len=a.planted_len)
    finally:
        restore()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if npz_arrays:
        npz = a.out.rsplit(".", 1)[0] + f".{label}.tokens.npz"
        np.savez_compressed(npz, **npz_arrays)
        arm["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    _print_tables(arm, depths, H)
    print(f"wrote {a.out}", flush=True)


def _planted(packed, meta, spec, tul_rt, cfg, depths, H, forward_ce, boot,
             n_rows, rng, torch, planted_len: int = 1) -> dict:
    """The planted copy pair: CE of a rare repeat against (source distance g, depth d).

    ``planted_len 1``: the rare id X sits at the source and its label at the scored
    position. ``planted_len 2``: the source holds ``(C, X)``, the scored position's INPUT is
    C in every condition (control included), so the planted condition adds exactly the
    (cue -> X) evidence and the benefit is an induction copy.
    """
    if planted_len not in (1, 2):
        raise ValueError(f"planted_len must be 1 or 2, got {planted_len}")
    is_boundary = tul_rt.data_cfg.rule.is_boundary
    vocab = int(cfg.model.vocab_size)
    sites = []          # (batch_idx, b, scored_pos, repeat_pos, {g: source_pos}, X, row)
    for bi, ((inp, labels, layout, _), m) in enumerate(zip(packed, meta)):
        B = inp.shape[0]
        used = set(int(t) for t in np.unique(inp.numpy()))
        for b in range(B):
            spans = np.where(m["istok"][b], m["spans"][b], -1)
            n_sp = int(spans.max()) + 1 if (spans >= 0).any() else 0
            cand = plant_sites(n_sp, H, phase=0)
            good = []
            for s in cand:
                try:
                    sc, rp = planted_triple(spans, s)
                    src = {g: source_positions(spans, s, g, rng, planted_len)
                           for g in range(H + 1)}
                except ValueError:
                    continue
                good.append((s, sc, rp, src))
            if not good:
                continue
            rare = _rare_ids(planted_len * len(good), used, is_boundary, spec.slot_id,
                             vocab, rng)
            used.update(rare)
            for k, (s, sc, rp, src) in enumerate(good):
                ids_k = rare[planted_len * k: planted_len * (k + 1)]
                # pair = (C, X) or (X,): X is always the LAST id and the copy target
                sites.append({"bi": bi, "b": b, "scored": sc, "repeat": rp,
                              "src": src, "pair": ids_k, "X": ids_k[-1],
                              "row": m["row0"] + b})
    if not sites:
        raise SystemExit("no planted site fits: the rows hold too few spans for hops=H")
    print(f"[planted] {len(sites)} sites over {n_rows} rows", flush=True)

    def run(g: int | None, d: int) -> tuple[np.ndarray, np.ndarray]:
        """CE of every site's repeat, with the source at distance g (None = no source)."""
        vals, rows = [], []
        for bi, (inp, labels, layout, _) in enumerate(packed):
            mine = [s for s in sites if s["bi"] == bi]
            if not mine:
                continue
            ci = inp.numpy().copy()
            lab = np.full(labels.shape, -100, dtype=np.int64)
            for s in mine:
                ci[s["b"], s["repeat"]] = s["X"]
                if len(s["pair"]) == 2:
                    ci[s["b"], s["scored"]] = s["pair"][0]      # the cue, every condition
                if g is not None:
                    for pos, tid in zip(s["src"][g], s["pair"]):
                        ci[s["b"], pos] = tid
                lab[s["b"], s["scored"]] = s["X"]
            ce = forward_ce(torch.from_numpy(ci), torch.from_numpy(lab), layout, d)
            for s in mine:
                vals.append(float(ce[s["b"], s["scored"]]))
                rows.append(s["row"])
        return np.asarray(vals, dtype=np.float64), np.asarray(rows, dtype=np.int64)

    out: dict[str, object] = {"n_sites": len(sites), "hops": H, "control": {},
                              "planted_len": int(planted_len)}
    ctrl: dict[int, np.ndarray] = {}
    site_rows = np.asarray([s["row"] for s in sites], dtype=np.int64)
    for d in depths:
        v, rows = run(None, d)
        if not np.array_equal(rows, site_rows):
            raise RuntimeError("the planted forward returned sites in another order than "
                               "the site list; the per-row pairing would be wrong")
        ctrl[d] = v
        out["control"][str(d)] = float(v.mean())
        print(f"  planted control depth={d:>2d}  ce={v.mean():.4f}", flush=True)
    keep = np.ones(site_rows.shape, dtype=bool)
    for g in range(H + 1):
        cell: dict[str, object] = {"ce": {}, "benefit": {}}
        got: dict[int, np.ndarray] = {}
        for d in depths:
            v, _ = run(g, d)
            got[d] = v
            cell["ce"][str(d)] = float(v.mean())
            cell["benefit"][str(d)] = float(ctrl[d].mean() - v.mean())
        for a_, z_ in [(1, 6), (1, max(depths))]:
            if a_ in got and z_ in got and a_ != z_:
                sa, n = bin_sums(got[a_], site_rows, keep, n_rows)
                sz, _ = bin_sums(got[z_], site_rows, keep, n_rows)
                cell[f"K{a_}-K{z_}"] = boot(sa, sz, n)
        out[f"g{g}"] = cell
        print(f"  planted g={g}  " + "  ".join(
            f"d{d}={cell['ce'][str(d)]:.3f}" for d in depths), flush=True)
    return out


def _print_tables(arm: dict, depths: list[int], H: int) -> None:
    print()
    print(f"== {arm['label']} step {arm['step']} — geometry {arm['tg_geometry']}, "
          f"coda_reach {arm['tg_coda_prefix_reach']}, loop_reach {arm['loop_reach']} ==")
    if "dce_profile" in arm:
        print("\n-- corruption profile (mean dCE at the training depth, nats) --")
        print("  hop     dCE      n")
        for h in range(0, H + 1):
            k = str(h)
            tag = "own" if h == 0 else f"{h:>3d}"
            print(f"  {tag}  {arm['dce_profile'][k]:+8.4f}  {arm['dce_profile_n'][k]:>7d}")
    if "bins" in arm:
        print("\n-- clean CE by depth, binned on h* = argmax_h dCE(h) --")
        head = "  h*      n  " + "".join(f"  d={d:<7d}" for d in depths)
        print(head + "   K1-K6            K1-K3")
        for b, e in sorted(arm["bins"].items(), key=lambda kv: int(kv[0])):
            ces = "".join(f"  {e['ce'][str(d)]:9.4f}" for d in depths)
            k16 = e.get("K1-K6")
            k13 = e.get("K1-K3")
            s16 = (f"{k16['point']:+.4f}[{k16['lo']:+.4f},{k16['hi']:+.4f}]"
                   if k16 else "n/a")
            s13 = (f"{k13['point']:+.4f}[{k13['lo']:+.4f},{k13['hi']:+.4f}]"
                   if k13 else "n/a")
            print(f"  {int(b):>2d} {e['n_tokens']:>7d}{ces}   {s16}  {s13}")
        print("  all tokens:   " + "".join(
            f"  {arm['all_tokens'][str(d)]:9.4f}" for d in depths))
    if "planted" in arm:
        p = arm["planted"]
        print(f"\n-- planted copy pair ({p['n_sites']} sites); g = source distance --")
        print("  g   " + "".join(f"  d={d:<7d}" for d in depths))
        print("  ctl " + "".join(f"  {p['control'][str(d)]:9.4f}" for d in depths))
        for g in range(H + 1):
            e = p[f"g{g}"]
            print(f"  {g:>2d}  " + "".join(
                f"  {e['ce'][str(d)]:9.4f}" for d in depths))
        print("  benefit = control - planted (nats); >0 means the copy was used")
        for g in range(H + 1):
            e = p[f"g{g}"]
            print(f"  {g:>2d}  " + "".join(
                f"  {e['benefit'][str(d)]:+9.4f}" for d in depths))


if __name__ == "__main__":
    main()
