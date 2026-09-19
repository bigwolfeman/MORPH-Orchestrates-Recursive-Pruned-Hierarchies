"""Arm-vs-arm paired CE on the depth ladder, with a 480-ROW bootstrap.

`core_depth_sweep.py` writes, beside every sweep JSON, a `<...>.tokens.npz` holding
`tok_index` (int32, the stream index of every scored position, in row order) and
`ce_<depth>` (float32). The ladder's arms are all plain `notul_norm_match_20k`
descendants: they cut the same validation stream the same way, so their `tok_index`
arrays are IDENTICAL and no intersection is needed. This scorer REFUSES to pair two
files whose indices differ rather than silently intersecting them, because on this
ladder a mismatch means the row cut moved and the comparison is no longer the one the
record asks for.

Why not `sweep_score.paired` / `span_budget_profile`: both bootstrap over 1,024-token
stream BLOCKS. The ladder prereg asks for the interval over the 480 validation ROWS,
the same unit `core_depth_sweep` uses for its own within-arm K-curve intervals, so the
between-arm numbers and the within-arm ones are read on one unit. The bootstrap itself
is `_stats.paired_bootstrap_ci` — one home, no second implementation.

    python lab/divergence/ladder_score.py --spec spec.json --out out.json --csv out.csv

The spec is `{"pairs": [{"name": ..., "a": {"json": ..., "depth": 3},
                         "b": {"json": ..., "depth": 6}, "bar": ..., "note": ...}]}`
and every pair reports `CE(a @ depth) - CE(b @ depth)`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _stats import paired_bootstrap_ci                              # noqa: E402


def load_arm(path: str) -> dict:
    """The one arm inside a `core_depth_sweep.py` JSON."""
    with open(path) as f:
        d = json.load(f)
    if len(d) != 1:
        raise ValueError(f"{path}: expected exactly one arm, found {sorted(d)}")
    return next(iter(d.values()))


def tokens_npz_path(arm: dict, json_path: str) -> str:
    """The per-token file the sweep wrote, tried beside the JSON when the path moved."""
    p = arm.get("tokens_npz")
    if not p:
        raise ValueError(f"{json_path}: the sweep wrote no tokens_npz")
    if os.path.exists(p):
        return p
    beside = os.path.join(os.path.dirname(os.path.abspath(json_path)), os.path.basename(p))
    if os.path.exists(beside):
        return beside
    raise FileNotFoundError(f"{p} (also tried {beside})")


def load_tokens(npz_path: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """``tok_index`` and ``{depth: per-token CE}`` from a sweep's token file."""
    z = np.load(npz_path)
    ce = {int(k[3:]): np.asarray(z[k], dtype=np.float64)
          for k in z.files if k.startswith("ce_")}
    return np.asarray(z["tok_index"]), ce


def require_same_index(idx_a: np.ndarray, idx_b: np.ndarray,
                       label_a: str = "a", label_b: str = "b") -> None:
    """Raise unless the two arms scored exactly the same stream positions in one order."""
    if idx_a.shape != idx_b.shape:
        raise ValueError(f"tok_index length differs: {label_a} has {idx_a.size}, "
                         f"{label_b} has {idx_b.size}; these arms cut the stream "
                         f"differently and must not be paired position by position")
    if not np.array_equal(idx_a, idx_b):
        n_bad = int((idx_a != idx_b).sum())
        first = int(np.flatnonzero(idx_a != idx_b)[0])
        raise ValueError(f"tok_index differs between {label_a} and {label_b} at "
                         f"{n_bad} of {idx_a.size} positions (first at {first}); "
                         f"these arms cut the stream differently and must not be paired")


def row_ids(counts: np.ndarray, n_tokens: int) -> np.ndarray:
    """Row id per scored position, from the sweep's per-row token counts."""
    counts = np.asarray(counts, dtype=np.int64)
    if int(counts.sum()) != int(n_tokens):
        raise ValueError(f"row counts sum to {int(counts.sum())} but the token arrays "
                         f"hold {int(n_tokens)} positions")
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts)


def paired_delta(ce_a: np.ndarray, ce_b: np.ndarray, rid: np.ndarray, n_rows: int,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Token-weighted ``mean(ce_a) - mean(ce_b)`` with a 95 % bootstrap over rows.

    The positions are already paired one-to-one (the caller has checked `tok_index`), so
    a row's contribution is the SUM of its tokens on both arms and the resample is over
    whole rows — the unit `core_depth_sweep` uses for its own K-curve intervals.
    """
    if ce_a.shape != ce_b.shape or ce_a.shape != rid.shape:
        raise ValueError("ce_a, ce_b and rid must have the same shape")
    sum_a = np.bincount(rid, weights=ce_a, minlength=n_rows)
    sum_b = np.bincount(rid, weights=ce_b, minlength=n_rows)
    cnt = np.bincount(rid, minlength=n_rows).astype(np.float64)
    out = paired_bootstrap_ci(sum_a, sum_b, cnt, n_boot=n_boot, seed=seed)
    out["ce_a"] = float(sum_a.sum() / cnt.sum())
    out["ce_b"] = float(sum_b.sum() / cnt.sum())
    out["n_tokens"] = int(cnt.sum())
    return out


def score_pair(pair: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """One spec entry -> its paired delta, CEs and interval."""
    a_json, b_json = pair["a"]["json"], pair["b"]["json"]
    da, db = int(pair["a"]["depth"]), int(pair["b"]["depth"])
    arm_a, arm_b = load_arm(a_json), load_arm(b_json)
    idx_a, ce_a = load_tokens(tokens_npz_path(arm_a, a_json))
    idx_b, ce_b = load_tokens(tokens_npz_path(arm_b, b_json))
    require_same_index(idx_a, idx_b, os.path.basename(a_json), os.path.basename(b_json))
    for d, ce, src in ((da, ce_a, a_json), (db, ce_b, b_json)):
        if d not in ce:
            raise ValueError(f"{src}: no depth {d} in the sweep (has {sorted(ce)})")
    rid = row_ids(np.asarray(arm_a["row_n_tokens"]), idx_a.size)
    n_rows = len(arm_a["row_n_tokens"])
    res = paired_delta(ce_a[da], ce_b[db], rid, n_rows, n_boot=n_boot, seed=seed)
    res.update({"name": pair.get("name", ""), "bar": pair.get("bar", ""),
                "note": pair.get("note", ""),
                "a_json": a_json, "a_depth": da, "a_step": arm_a["step"],
                "b_json": b_json, "b_depth": db, "b_step": arm_b["step"],
                "n_rows": n_rows})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, help="JSON file with a 'pairs' list")
    ap.add_argument("--out", help="write the scored pairs here as JSON")
    ap.add_argument("--csv", help="write the scored pairs here as CSV")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.spec) as f:
        spec = json.load(f)
    rows = [score_pair(p, n_boot=a.n_boot, seed=a.seed) for p in spec["pairs"]]
    for r in rows:
        print(f"{r['name']:34s} {r['ce_a']:.4f} - {r['ce_b']:.4f} = {r['point']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  bar={r['bar']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"n_boot": a.n_boot, "seed": a.seed, "pairs": rows}, f, indent=1)
        print(f"wrote {a.out}")
    if a.csv:
        cols = ["name", "a_json", "a_step", "a_depth", "ce_a", "b_json", "b_step",
                "b_depth", "ce_b", "point", "lo", "hi", "n_rows", "n_tokens", "n_boot",
                "level", "bar", "note"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "n_boot": a.n_boot})
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
