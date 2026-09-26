"""Token-paired CE gaps of code arms against a ruler sweep, at every depth the arm swept.

Wraps `sweep_score.paired`: every arm scores the same validation stream, so the per-token
CE arrays (`tok_index` in the sweep's tokens npz) intersect exactly and the bootstrap runs
over 1,024-token stream blocks. Output has the shape of the 5k panel's
`paired_vs_strict_ruler_5000.json` (`ce_<k>_minus_ruler_ce<d>: [point, lo, hi]`).

    python lab/divergence/paired_vs_ruler.py --ruler RESULTS/sweep_slot-spandec-strict-20k_20000.json \
        --ruler_depth 6 --arms tul-code-20k=RESULTS/sweep_tul-code-20k_20000.json \
        --out .../paired_vs_strict_ruler_20000.json
"""
from __future__ import annotations

import argparse
import json
import os

from sweep_score import load_sweep, paired


def within_arm(arm: dict, search: tuple[str, ...] = ()) -> dict:
    """The arm's own K1-K6 and K3-K6 (``arm_K<a>-K<b>``), token-paired against itself with
    the same stream-block bootstrap as the ruler gaps (``sweep_score.paired``). Appended
    beside the ruler rows (2026-09-26): forced depth 1 is off the training distribution
    (about 1.7 % of Poisson(6) draws), K3-K6 is the within-distribution read."""
    out = {}
    for a, b in ((1, 6), (3, 6)):
        if str(a) in arm["row_ce_sum"] and str(b) in arm["row_ce_sum"]:
            p = paired(arm, str(a), arm, str(b), search)
            out[f"arm_K{a}-K{b}"] = [p[0], p[1], p[2]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", required=True, help="the ruler's core_depth_sweep JSON")
    ap.add_argument("--ruler_depth", default="6")
    ap.add_argument("--arms", required=True, nargs="+", help="LABEL=sweep.json ...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ruler = load_sweep(a.ruler)
    assert ruler is not None, a.ruler
    search = (os.path.dirname(os.path.abspath(a.ruler)),)
    out = {}
    for spec in a.arms:
        label, path = spec.split("=", 1)
        arm = load_sweep(path)
        assert arm is not None, path
        row = {}
        for k in arm["row_ce_sum"]:
            p = paired(arm, str(k), ruler, a.ruler_depth, search)
            row[f"ce_{k}_minus_ruler_ce{a.ruler_depth}"] = [p[0], p[1], p[2]]
            row.setdefault("pairing", p[3])
        within = within_arm(arm, search)
        row.update(within)
        out[label] = row
        print(label, row["pairing"])
        for k, v in row.items():
            if k.startswith("ce_"):
                print(f"  {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
        for k, v in within.items():
            print(f"  {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
