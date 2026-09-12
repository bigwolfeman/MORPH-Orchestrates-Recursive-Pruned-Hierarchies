"""What is ONE slot's thinking worth? Per-slot depth isolation for the slot loop.

WHY THIS EXISTS. Every depth reading in this arc is GLOBAL: `core_depth_sweep.py` forces
the same depth on every slot in the row and reads the token CE, and eleven arms have come
back flat (K1-K6 <= 0.0023 across the nine arms of `lab/experiments/failures/`,
2026-09-10 and 2026-09-11). A global sweep cannot separate two very different worlds:

  * the loop is worth nothing anywhere;  or
  * the loop refines the CURRENT thought and the reading is drowned because the same
    knob also lobotomises the fifty ARCHIVED slots the coda reads for history.

At depth 1 the whole row's history channel is shallow at once. This instrument moves ONE
slot at a time and scores ONLY that slot's own next span, so the current-thought reading
and the archive reading are two different columns:

  arm ISOLATE     slot s at depth d, every other slot at the reference depth.
                  Scored on span s+1. "What does THIS slot lose by thinking less,
                  with the archive intact?"
  arm COMPLEMENT  slot s at the reference depth, every other slot at depth d.
                  Scored on span s+1. "What does this slot lose when the ARCHIVE
                  thinks less, with its own thinking intact?"
  arm UNIFORM     every slot at depth d — `core_depth_sweep.py`'s column, recomputed
                  here on exactly these positions so the three are paired.

If ISOLATE moves and UNIFORM does not, the global K-curve is hiding a real per-slot
effect behind the archive. If neither moves, the loop is flat for this arm and the
"archive drowns it" story is dead. Both readings are useful; that is the point of an
instrument.

THE FORWARD ARGUMENT. This needs a per-slot depth TABLE, which no config knob can say, so
`MORPHTransformer.forward` grew `slot_depths: [B, max_slots] | None` — a per-forward DATA
argument, the `slot_layout` pattern, EVAL ONLY, and `None` is bit-identical to before it
existed (`tests/test_slot_depth_isolation.py`). It overrides `tul.slot_depth_fixed` and
`tul.slot_mean_depth` and RAISES outside [1, slot_max_depth] rather than clamping.

WHAT IT CANNOT SAY. It scores the ordinary token CE, so it measures what the CODA got out
of that slot's extra passes, never what the slot state itself holds (`slot_z_optimize.py`
is the instrument for that) and never what a BETTER slot state would be worth. It is an
EVAL-time intervention on a trained model: a slot that never learned to use depth 6 reads
flat here, and that is not evidence that a model trained with per-slot depth would (the
2026-09-12 reading "training depth moves CE, eval depth does not" is exactly this trap —
pair any result with a depth-1-TRAINED control). It says nothing about span 0 (no slot
precedes it) or about the row's open tail (no complete next span).

POSITIONS. `_next_span.py` owns the map, with the tree's offset convention: the CE at p
is the cost of the token at p+1, and offset 0 of span s+1 is predicted at span s's LAST
TOKEN — a position BEFORE the slot cell. The per-offset table reuses
`span_budget_profile.py`'s buckets (0..7 then 8+).

COST, arithmetic and not a guess. One forward per (slot, arm, depth) per batch, plus a
reference and a uniform forward per depth. At 480 rows, batch 3 (160 batches), ~51 valid
slots a row and `--depths 1,3` against `--ref-depth 6`, that is
160 x (1 + 2 x (1 + 2 x 51)) = 33,120 batched forwards = 99,360 row-forwards — roughly
40x a `core_depth_sweep` run over the same rows. Run it at `--rows 48` first; the CI over
rows is what says whether more rows are needed. `--slot-chunk c` tiles the batch c times
so c slot variants share one forward (c x the activation memory, same FLOPs).

Usage:
  python lab/divergence/slot_depth_isolation.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --rows 48 --batch 3 --depths 1,3 --ref-depth 6 \
      --out .../depth_isolation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _next_span import next_span_positions  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

OFFSET_BUCKETS = [str(k) for k in range(8)] + ["8+"]


def bucket_of(j: np.ndarray) -> np.ndarray:
    """Offset -> bucket index, `span_budget_profile.py`'s buckets (0..7, then 8+)."""
    return np.minimum(j, 8)


def tile_layout(layout, n: int):
    """The layout of `n` copies of every row, in row-major order (row 0 n times, ...).

    A plain `repeat_interleave` on every batched field. The dataclass carries no other
    state that depends on the batch, and `stats` is per-batch bookkeeping the forward
    never reads.
    """
    from morph.model.tul_layout import SlotLayout

    if n == 1:
        return layout
    def _r(t):
        return None if t is None else t.repeat_interleave(n, dim=0)

    return SlotLayout(
        slot_mask=_r(layout.slot_mask), bag_id=_r(layout.bag_id),
        slot_index=_r(layout.slot_index), slot_valid=_r(layout.slot_valid),
        prefix_k=layout.prefix_k, stats=None,
        span_len=_r(layout.span_len), len_supervised=_r(layout.len_supervised))


@torch.no_grad()
def ce_map(model, inp, labels, layout, depths, device) -> torch.Tensor:
    """`[B, L]` per-position CE at the forced per-slot depths."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp, None, layout, slot_depths=depths)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.clone()
    lab[lab < 0] = 0
    return F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L)


class Accum:
    """Per-ROW sums and counts, keyed by (arm, depth) and by offset bucket.

    The bootstrap unit is a row, matching every other paired readout in this directory
    (`_stats.paired_bootstrap_ci`). A row contributes one sum and one count per key, so a
    resample re-weighs whole rows and the interval carries the within-row correlation the
    per-token interval would throw away.
    """

    def __init__(self, n_rows_hint: int = 0):
        self.sums: dict[str, list[float]] = {}
        self.counts: dict[str, list[float]] = {}
        self.n_rows = 0

    def new_rows(self, n: int) -> None:
        self.n_rows += n
        for k in self.sums:
            self.sums[k].extend([0.0] * n)
            self.counts[k].extend([0.0] * n)

    def _key(self, key: str) -> None:
        if key not in self.sums:
            self.sums[key] = [0.0] * self.n_rows
            self.counts[key] = [0.0] * self.n_rows

    def add(self, key: str, row0: int, sums: np.ndarray, counts: np.ndarray) -> None:
        self._key(key)
        for i in range(sums.shape[0]):
            self.sums[key][row0 + i] += float(sums[i])
            self.counts[key][row0 + i] += float(counts[i])

    def mean(self, key: str) -> float:
        c = float(np.sum(self.counts[key]))
        return float(np.sum(self.sums[key]) / c) if c > 0 else float("nan")

    def ci(self, key: str, ref: str, seed: int = 0, n_boot: int = 2000) -> dict:
        a = np.asarray(self.sums[key])
        b = np.asarray(self.sums[ref])
        n = np.asarray(self.counts[key])
        if not np.allclose(n, self.counts[ref]):
            raise RuntimeError(
                f"arm {key!r} and reference {ref!r} were scored on different position "
                "counts — they are not paired and the difference means nothing")
        return paired_bootstrap_ci(a, b, n, seed=seed, n_boot=n_boot)


def score(ce: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, s: int,
          off_bucket: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray]:
    """Per-row (sum, count) over slot `s`'s next span, and the same split by offset.

    Returns `(row_sum, row_cnt, off_sum [B, 9], off_cnt [B, 9])`.
    """
    B = ce.shape[0]
    p = pos[:, s, :]                                       # [B, J]
    v = valid[:, s, :]
    x = ce.gather(1, p) * v.to(ce.dtype)                   # [B, J]
    row_sum = x.sum(dim=1).double().cpu().numpy()
    row_cnt = v.sum(dim=1).double().cpu().numpy()
    xn = x.double().cpu().numpy()
    vn = v.cpu().numpy()
    off_sum = np.zeros((B, len(OFFSET_BUCKETS)))
    off_cnt = np.zeros((B, len(OFFSET_BUCKETS)))
    for b in range(B):
        np.add.at(off_sum[b], off_bucket[vn[b]], xn[b][vn[b]])
        np.add.at(off_cnt[b], off_bucket[vn[b]], 1.0)
    return row_sum, row_cnt, off_sum, off_cnt


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="1,3",
                    help="the isolated depths; the reference depth is always included "
                         "as the identity check")
    ap.add_argument("--ref-depth", type=int, default=6)
    ap.add_argument("--slot-chunk", type=int, default=1,
                    help="how many slot variants share one forward (tiles the batch; "
                         "c x the activation memory, same FLOPs)")
    ap.add_argument("--max-slots", type=int, default=0,
                    help="score only the first N slots of a row (0 = all). A cheap way "
                         "to trade coverage for wall clock; the JSON records it.")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    depths = [int(x) for x in a.depths.split(",")]
    ref = int(a.ref_depth)

    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_depth_isolation needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false): the paid loop has no per-slot "
                         "depth to isolate.")
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            a.device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    max_d = int(tc.slot_max_depth or model.cfg.max_depth)
    for d in [ref, *depths]:
        if not (1 <= d <= max_d):
            raise SystemExit(f"depth {d} is outside [1, slot_max_depth={max_d}] — the "
                             "model has never run it, so it is not a measurement of it")

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    acc = Accum()
    n_slots_seen: set[int] = set()
    identity_err = 0.0
    for bi, (inp, labels, layout, _idx) in enumerate(batches):
        inp, labels = inp.to(a.device), labels.to(a.device)
        layout = layout.to(a.device)
        B = inp.shape[0]
        row0 = acc.n_rows
        acc.new_rows(B)
        pos, valid = next_span_positions(layout, labels, inp)
        J = pos.shape[2]
        off_bucket = bucket_of(np.arange(J))
        S = layout.max_slots if a.max_slots <= 0 else min(layout.max_slots, a.max_slots)
        # slots with at least one scored position anywhere in the batch
        live = [s for s in range(S) if bool(valid[:, s, :].any())]
        n_slots_seen.update(live)

        full_ref = torch.full(layout.slot_index.shape, ref, dtype=torch.long,
                              device=a.device)
        ce_ref = ce_map(model, inp, labels, layout, full_ref, a.device)
        for s in live:
            rs, rc, os_, oc = score(ce_ref, pos, valid, s, off_bucket)
            acc.add("ref", row0, rs, rc)
            for k, name in enumerate(OFFSET_BUCKETS):
                acc.add(f"ref/off{name}", row0, os_[:, k], oc[:, k])

        # THE IDENTITY CHECK, run once. A table filled with the model's OWN eval depth
        # must reproduce a `slot_depths=None` forward bit for bit. That is the only check
        # here that tests the override PLUMBING rather than repeatability: every other
        # table this script builds differs from the shipped path by construction, so a
        # wrong override would look like a result. A non-zero value and the arms below
        # are unreadable.
        if bi == 0:
            own = int(tc.slot_depth_fixed or tc.slot_mean_depth or model.cfg.mean_depth)
            identity_err = float(
                (ce_map(model, inp, labels, layout,
                        torch.full_like(full_ref, own), a.device)
                 - ce_map(model, inp, labels, layout, None, a.device)).abs().max())

        for d in depths:
            full_d = torch.full_like(full_ref, d)
            ce_uni = ce_map(model, inp, labels, layout, full_d, a.device)
            for s in live:
                rs, rc, os_, oc = score(ce_uni, pos, valid, s, off_bucket)
                acc.add(f"uniform/d{d}", row0, rs, rc)
                for k, name in enumerate(OFFSET_BUCKETS):
                    acc.add(f"uniform/d{d}/off{name}", row0, os_[:, k], oc[:, k])
            for arm, base, other in (("isolate", full_ref, d), ("complement", full_d, ref)):
                for chunk0 in range(0, len(live), max(1, a.slot_chunk)):
                    sel = live[chunk0:chunk0 + max(1, a.slot_chunk)]
                    c = len(sel)
                    tab = base.repeat_interleave(c, dim=0)
                    for i, s in enumerate(sel):
                        tab[i::c, s] = other
                    ce_c = ce_map(model, inp.repeat_interleave(c, dim=0),
                                  labels.repeat_interleave(c, dim=0),
                                  tile_layout(layout, c), tab, a.device)
                    for i, s in enumerate(sel):
                        rs, rc, os_, oc = score(ce_c[i::c], pos, valid, s, off_bucket)
                        acc.add(f"{arm}/d{d}", row0, rs, rc)
                        for k, name in enumerate(OFFSET_BUCKETS):
                            acc.add(f"{arm}/d{d}/off{name}", row0, os_[:, k], oc[:, k])
        print(f"  batch {bi + 1}/{len(batches)}: {B} rows, {len(live)} slots", flush=True)

    if identity_err != 0.0:
        raise SystemExit(
            f"IDENTITY CHECK FAILED: slot_depths filled with the model's own eval depth "
            f"moved the CE map by {identity_err:.3e} against slot_depths=None. The "
            "override is not reproducing the shipped forward; do not report these "
            "numbers.")

    res: dict = {"ce_ref": acc.mean("ref"), "arms": {}}
    for d in depths:
        row: dict = {}
        for arm in ("isolate", "complement", "uniform"):
            k = f"{arm}/d{d}"
            row[arm] = {"ce": acc.mean(k), "n": float(np.sum(acc.counts[k])),
                        **acc.ci(k, "ref", n_boot=a.boot)}
            row[arm]["by_offset"] = {
                name: {"n": float(np.sum(acc.counts[f"{k}/off{name}"])),
                       "delta": acc.mean(f"{k}/off{name}") - acc.mean(f"ref/off{name}"),
                       **acc.ci(f"{k}/off{name}", f"ref/off{name}",
                                n_boot=a.boot)}
                for name in OFFSET_BUCKETS
                if np.sum(acc.counts[f"{k}/off{name}"]) > 0}
        res["arms"][str(d)] = row

    rec = {
        "label": label, "config": config, "step": step, "rows": acc.n_rows,
        "batch": a.batch, "ref_depth": ref, "depths": depths,
        "slots_scored": sorted(n_slots_seen), "max_slots_arg": a.max_slots,
        "slot_chunk": a.slot_chunk, "identity_check_abs_err": identity_err,
        "notes": {
            "arms": "isolate = slot s at d and every other slot at ref; complement = "
                    "slot s at ref and every other slot at d; uniform = every slot at d",
            "scored": "ONLY span s+1's tokens, by the _next_span.py map (offset 0 is "
                      "predicted at span s's LAST TOKEN, before the slot cell)",
            "delta": "arm minus reference, in nats; POSITIVE means the arm is worse",
            "unit": "row (paired bootstrap, _stats.paired_bootstrap_ci)",
            "eval_only": "an eval-time depth intervention on a trained model; pair any "
                         "moving result with a depth-1-TRAINED control before reading it "
                         "as 'the loop earns'",
        },
        "results": res,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)

    print(f"\n{label} step {step} — {acc.n_rows} rows, ref depth {ref}, "
          f"{len(n_slots_seen)} slot indices")
    print(f"  identity check |delta| {identity_err:.3e} (must be 0)")
    print(f"  ce_ref {res['ce_ref']:.4f} over {np.sum(acc.counts['ref']):.0f} positions")
    print("  " + "arm".ljust(22) + "delta".rjust(10) + "95% CI".rjust(22))
    for d in depths:
        for arm in ("isolate", "complement", "uniform"):
            v = res["arms"][str(d)][arm]
            print(f"  {arm + ' d' + str(d):22s}{v['point']:+10.4f}"
                  f"   [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
