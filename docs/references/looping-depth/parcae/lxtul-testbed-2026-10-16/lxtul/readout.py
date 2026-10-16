"""Panel readout: CE, K1-K6 and the paired gap to a reference arm, from each run's sweep.json.

    python -m lxtul.readout --runs /home/wolfe/parcae-runs --ref plain-5k plain-5k lxtul-5k ...

Every arm must have scored the same rows: the per-row token counts are checked equal, so a
gap is paired per row (MORPH `_stats.paired_bootstrap_ci`, 2000 draws, seed 0). tok/s is
the median of the logged training windows after step 100; peak memory is the last logged.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from lxtul.evaluate import _stats


def _arm(run: Path) -> dict:
    sweep = json.loads((run / "sweep.json").read_text())
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    tr = [r for r in rows if "perf/tok_s" in r and r["step"] > 100]
    return {"sweep": sweep, "tok_s": statistics.median(r["perf/tok_s"] for r in tr) if tr else None,
            "peak_gib": tr[-1]["perf/peak_alloc_gib"] if tr else None}


def readout(runs: Path, names: list[str], ref: str, depth: int = 6) -> list[dict]:
    ci = _stats().paired_bootstrap_ci
    arms = {n: _arm(runs / n) for n in names}
    base = arms[ref]["sweep"]
    cnt = np.asarray(base["count"])
    out = []
    for n, a in arms.items():
        sw = a["sweep"]
        if not np.array_equal(np.asarray(sw["count"]), cnt):
            raise ValueError(f"{n} scored different rows or tokens than {ref}: not paired")
        s = {k: np.asarray(v) for k, v in sw["sums"].items()}
        row = {"arm": n, "ce": float(s[str(depth)].sum() / cnt.sum()),
               "ce_by_depth": {k: float(v.sum() / cnt.sum()) for k, v in s.items()},
               "K1_minus_K6": ci(s["1"], s[str(depth)], cnt),
               "tok_s": a["tok_s"], "peak_gib": a["peak_gib"]}
        if n != ref:
            row["gap_to_ref"] = ci(s[str(depth)], np.asarray(base["sums"][str(depth)]), cnt)
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("/home/wolfe/parcae-runs"))
    p.add_argument("--ref", required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("names", nargs="+")
    a = p.parse_args()
    res = readout(a.runs, a.names, a.ref)
    f = lambda c: f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
    print(f"{'arm':<18} {'CE@6':>7} {'K1-K6':>28} {'gap to ' + a.ref:>30} {'tok/s':>7} {'GiB':>5}")
    for r in res:
        g = f(r["gap_to_ref"]) if "gap_to_ref" in r else "-"
        print(f"{r['arm']:<18} {r['ce']:7.4f} {f(r['K1_minus_K6']):>28} {g:>30} "
              f"{r['tok_s'] or 0:7.0f} {r['peak_gib'] or 0:5.1f}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
