"""The cross-span information budget, overall and by offset after a span start.

Two arms (`budget-web-full`, `budget-web-span`) differ in ONE thing: whether anything may
cross a span boundary (`model.span_mask`). Their paired per-token CE gap is the number of
nats that live across span boundaries, and its profile by offset is the shape a slot
could ever fill. Prereg: `lab/experiments/planned/2026-09-11-arc-span-budget.md`.

Input is the per-token sweep artifact `core_depth_sweep.py` already writes —
`<outdir>/sweep_<arm>_<step>.<arm>.tokens.npz`, holding `tok_index` (int32, the STREAM
index of the input token at each scored position) and `ce_<depth>` (float32). Arms pair
on `tok_index`, so a plain arm and a slot arm that cut the same stream differently still
pair at the token level.

    python lab/divergence/span_budget_profile.py \
        --full  OUT/sweep_budget-web-full_5000.budget-web-full.tokens.npz \
        --span  OUT/sweep_budget-web-span_5000.budget-web-span.tokens.npz \
        --config budget_web_full --depths 6,1

WHAT "OFFSET" MEANS HERE, because the answer depends on it. The CE at a scored position
is the cost of predicting the NEXT token, and the offset reported is the offset of THAT
PREDICTED TOKEN inside its own span. Offset 0 is therefore a span's first token,
predicted from the last token of the previous span — the position where a slot would earn
its living, and the position the 2026-09-11 worth profile reads. The last position of
every row is dropped: its label lies outside the row the rule was cut on, so it has no
offset.

`--rows` must cover every `tok_index` in the files (the runner sweeps 480), and `--config`
names the config whose ROW CUT defines the spans — both arms of a budget pair share it.
Pairing two arms whose row structure differs (a plain arm against a slot arm) is still
valid for the OVERALL number and is reported, but the per-offset table is then computed
on `--config`'s rows alone and the script says so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _build import build_cfg                                   # noqa: E402
from _rows import pack_rows, stream_from_loader                # noqa: E402

BLOCK = 1024        # bootstrap unit: one block of stream indices (sweep_score.py's)
N_BOOT = 400
SEED = 0


def offset_map(config: str, rows: int, batch: int = 3) -> tuple[dict[int, int], int]:
    """``{stream index of a scored position: offset of its PREDICTED token}``, row count.

    Built from the SAME four calls every `lab/divergence` readout uses, so the rows are
    the sweep's rows: `create_dataloader(..., 2048, 8, split="validation",
    skip_samples=0, bag_size=0, tul=None)` -> `stream_from_loader` -> `pack_rows`.
    """
    from morph.model.tul_layout import span_ids_from_ids
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_boundary_rule, build_tul_runtime

    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    plain = tul_rt is None
    rule, _lut, _eos, _sub = build_boundary_rule(cfg)
    row_tokens = (int(cfg.data.seq_len) + 1 if plain
                  else tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                              split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, batch, plain)[: -(-rows // batch)]

    out: dict[int, int] = {}
    n_rows = 0
    for inp, _labels, layout, idx in batches:
        ids = inp.numpy().astype(np.int64)
        # The cut the MODEL ran on: `_span_context` cuts the row's own input ids.
        span = span_ids_from_ids(ids, rule)
        tokpos = (idx >= 0).numpy() if layout is not None else np.ones_like(ids, dtype=bool)
        for b in range(ids.shape[0]):
            n_rows += 1
            # offset of position p inside its span, from the row's span ids
            off = np.zeros(ids.shape[1], dtype=np.int64)
            run = 0
            for p in range(1, ids.shape[1]):
                run = 0 if span[b, p] != span[b, p - 1] else run + 1
                off[p] = run
            for p in range(ids.shape[1] - 1):        # the last position has no label offset
                if tokpos[b, p]:
                    out[int(idx[b, p])] = int(off[p + 1])
    return out, n_rows


def _boot(d: np.ndarray, blk: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95 % CI of the mean of ``d``, resampling whole blocks of stream indices."""
    if d.size == 0:
        return float("nan"), float("nan")
    uniq = np.unique(blk)
    if uniq.size < 2:
        return float("nan"), float("nan")
    order = np.argsort(blk, kind="stable")
    d_s, blk_s = d[order], blk[order]
    starts = np.searchsorted(blk_s, uniq, side="left")
    ends = np.searchsorted(blk_s, uniq, side="right")
    sums = np.add.reduceat(d_s, starts) if d_s.size else np.zeros(uniq.size)
    cnts = (ends - starts).astype(np.float64)
    draws = rng.integers(0, uniq.size, size=(N_BOOT, uniq.size))
    means = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1.0)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def profile(full_npz: str, span_npz: str, offsets: dict[int, int] | None,
            depths: list[int]) -> dict:
    a, b = np.load(full_npz), np.load(span_npz)
    common, pa, pb = np.intersect1d(a["tok_index"], b["tok_index"],
                                    assume_unique=True, return_indices=True)
    res: dict = {"n_paired": int(common.size),
                 "n_full": int(a["tok_index"].size), "n_span": int(b["tok_index"].size),
                 "depths": {}}
    if common.size == 0:
        raise SystemExit("the two files share no token index — wrong pair?")
    off = None
    if offsets is not None:
        off = np.array([offsets.get(int(t), -1) for t in common], dtype=np.int64)
        res["n_without_offset"] = int((off < 0).sum())
    blk = common // BLOCK
    for d in depths:
        ka, kb = f"ce_{d}", f"ce_{d}"
        if ka not in a or kb not in b:
            res["depths"][d] = {"missing": True}
            continue
        gap = b[kb][pb].astype(np.float64) - a[ka][pa].astype(np.float64)
        rng = np.random.default_rng(SEED)
        lo, hi = _boot(gap, blk, rng)
        row = {"overall": float(gap.mean()), "ci": [lo, hi],
               "ce_full": float(a[ka][pa].mean()), "ce_span": float(b[kb][pb].mean()),
               "by_offset": {}}
        if off is not None:
            for name, sel in ([(str(k), off == k) for k in range(8)]
                              + [("8+", off >= 8)]):
                g = gap[sel]
                if g.size == 0:
                    continue
                lo_o, hi_o = _boot(g, blk[sel], np.random.default_rng(SEED))
                row["by_offset"][name] = {"n": int(g.size), "gap": float(g.mean()),
                                          "ci": [lo_o, hi_o]}
        res["depths"][d] = row
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="the UNRESTRICTED arm's tokens.npz")
    ap.add_argument("--span", required=True, help="the RESTRICTED arm's tokens.npz")
    ap.add_argument("--config", default=None,
                    help="config whose row cut defines the spans; omit for the overall "
                         "number only")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="6,1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    depths = [int(x) for x in a.depths.split(",")]
    offsets = None
    n_rows = 0
    if a.config:
        offsets, n_rows = offset_map(a.config, a.rows, a.batch)
    res = profile(a.full, a.span, offsets, depths)
    res["config"] = a.config
    res["rows_reconstructed"] = n_rows

    print(f"paired {res['n_paired']:,} tokens "
          f"(full {res['n_full']:,}, span {res['n_span']:,})")
    if offsets is not None:
        print(f"offsets from {a.config} over {n_rows} rows; "
              f"{res.get('n_without_offset', 0):,} paired tokens have none")
    for d in depths:
        row = res["depths"][d]
        if row.get("missing"):
            print(f"\ndepth {d}: MISSING from one of the files")
            continue
        print(f"\ndepth {d}:  CE full {row['ce_full']:.4f}   CE span {row['ce_span']:.4f}"
              f"   gap {row['overall']:+.4f} nats "
              f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]")
        if row["by_offset"]:
            print("   offset      n        gap        95% CI")
            for k, v in row["by_offset"].items():
                print(f"   {k:>6}  {v['n']:>8,}  {v['gap']:+8.4f}  "
                      f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
