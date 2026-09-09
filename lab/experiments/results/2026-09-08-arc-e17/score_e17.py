#!/usr/bin/env python
"""Score the E17 Sudoku-Extreme depth grid (P17a-h) from the sweep JSONs, probes and run logs.

The instrument is a GRID: answer-cell accuracy and whole-board solve rate by rating bucket
(0 easiest .. 4 hardest, 600 boards each) by forced loop depth T, per arm and checkpoint.
Every K-difference is paired over boards with a 2,000-draw bootstrap; between-arm
differences pair the same 3,000 boards (the sweeps score the whole held-out shard in file
order). Wall clocks and peaks come from the queue log and run logs, not from memory.

Usage: python lab/experiments/results/2026-09-08-arc-e17/score_e17.py [RESULTS_DIR]
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ARMS = ["sud-mask", "sud-notul"]
TRAINED = {"sud-mask": "12", "sud-notul": "6"}  # each arm's trained depth (slot mean 12 / core mean 6)
CKS = [1500, 3000, 4500, 6000]
DEPTHS = ["1", "2", "3", "4", "6", "9", "12", "16"]
BUCKETS = [0, 1, 2, 3, 4]
HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT, SEED = 2000, 0
# From arc/queue.log START/DONE stamps (2026-09-08): mask 11:42:54-15:18:13, notul 15:36:15-19:00:28.
WALL_H = {"sud-mask": 3 + 35 / 60 + 19 / 3600, "sud-notul": 3 + 24 / 60 + 13 / 3600}


def load(d: str, arm: str, ck: int) -> dict | None:
    p = os.path.join(d, f"sweep_{arm}_{ck}.json")
    return json.load(open(p))[arm] if os.path.exists(p) else None


def arrays(s: dict, depth: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-board (correct answer cells, answer cells, solved flag, bucket) at ``depth``."""
    c = np.asarray(s["doc_ans_correct"][depth], dtype=float)
    n = np.asarray(s["doc_n_answer"], dtype=float)
    return c, n, (c == n).astype(float), np.asarray(s["doc_band"], dtype=int)


def boot_ratio(num: np.ndarray, den: np.ndarray) -> tuple[float, float, float]:
    """sum(num)/sum(den) over boards, with a bootstrap CI over boards."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def acc_gain(s: dict, d_from: str, d_to: str, buckets: list[int]) -> tuple[float, float, float]:
    """Answer-cell accuracy at d_to minus at d_from on the given buckets, paired over boards."""
    c0, n, _, band = arrays(s, d_from)
    c1 = arrays(s, d_to)[0]
    keep = np.isin(band, buckets)
    return boot_ratio((c1 - c0)[keep], n[keep])


def solve_gain(s: dict, d_from: str, d_to: str, buckets: list[int]) -> tuple[float, float, float]:
    _, _, s0, band = arrays(s, d_from)
    s1 = arrays(s, d_to)[2]
    keep = np.isin(band, buckets)
    return boot_ratio((s1 - s0)[keep], np.ones(keep.sum()))


def diagonal(s: dict, d_from: str, d_to: str, hard: int, easy: int, solve: bool) -> tuple[float, float, float]:
    """gain(hard bucket) - gain(easy bucket); the buckets hold different boards so the two
    bootstraps are independent draws over each bucket's boards."""
    c0, n, s0, band = arrays(s, d_from)
    c1, _, s1, _ = arrays(s, d_to)
    num = (s1 - s0) if solve else (c1 - c0)
    den = np.ones_like(n) if solve else n
    rng = np.random.default_rng(SEED)
    out, boots = [], []
    for b in (hard, easy):
        k = band == b
        idx = rng.integers(0, k.sum(), size=(N_BOOT, k.sum()))
        out.append(num[k].sum() / den[k].sum())
        boots.append(num[k][idx].sum(1) / den[k][idx].sum(1))
    d = boots[0] - boots[1]
    return float(out[0] - out[1]), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def between(a: dict, da: str, b: dict, db: str) -> tuple[tuple, tuple]:
    """Arm a at depth da minus arm b at depth db: (accuracy, solve rate), paired over boards."""
    assert a["doc_stage"] == b["doc_stage"], "sweeps scored different boards"
    ca, n, sa, _ = arrays(a, da)
    cb, _, sb, _ = arrays(b, db)
    return boot_ratio(ca - cb, n), boot_ratio(sa - sb, np.ones_like(n))


def val_curve(log: str) -> dict[int, float]:
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"\[VAL\s+(\d+)\] loss=([0-9.]+)", open(log).read())}


def peak_gb(log: str) -> float:
    return max(float(x) for x in re.findall(r"peak=([0-9.]+)GB", open(log).read()))


def probe_stats(path: str) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    w = [r for r in rows if 1000 <= r["step"] <= 6000]
    g = [r["loss/gain_est"] for r in w if "loss/gain_est" in r]  # notul has no slot loop
    out = {"preclip_max_after_200": max((r["preclip/total"], r["step"]) for r in rows if r["step"] >= 200)}
    if g:
        out.update(gain_mean=float(np.mean(g)), gain_max=float(max(g)),
                   hinge_frac=sum(1 for r in w if r.get("loss/gain_reg_weighted", 0) > 0) / len(w))
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    S = {(arm, ck): s for arm in ARMS for ck in CKS if (s := load(root, arm, ck)) is not None}
    print("=== THE GRID: answer-cell accuracy (solve rate) by rating bucket x forced depth T")
    for (arm, ck), s in S.items():
        print(f"\n{arm} @ {ck}   (trained depth {TRAINED[arm]})")
        print("bucket " + " ".join(f"{'T=' + d:>14s}" for d in DEPTHS))
        for b in BUCKETS:
            row = [s["depths"][d] for d in DEPTHS]
            print(f"  {b}    " + " ".join(f"{r['band_acc_answer'][str(b)]:.4f} ({r['band_solve_rate'][str(b)]:.3f})"
                                          for r in row))
        print("  all  " + " ".join(f"{s['depths'][d]['acc_answer']:.4f} ({s['depths'][d]['solve_rate']:.3f})"
                                   for d in DEPTHS))
        print("  ce_ans " + " ".join(f"{s['depths'][d]['ce_answer']:14.4f}" for d in DEPTHS))

    print("\n=== P17b (does anything learn) at 6000")
    for arm in ARMS:
        s = S.get((arm, 6000))
        if s is None:
            print(f"  {arm}: no 6000 sweep yet")
            continue
        t = s["depths"][TRAINED[arm]]
        print(f"  {arm}: acc@{TRAINED[arm]} {t['acc_answer']:.4f}  bucket-0 solve {t['band_solve_rate']['0']:.4f}  "
              f"bucket-4 solve {t['band_solve_rate']['4']:.4f}  solve(all) {t['solve_rate']:.4f}")

    print("\n=== P17c (depth beyond 3) at 6000: accuracy gains paired over boards, buckets 3-4 pooled")
    for arm in ARMS:
        s = S.get((arm, 6000))
        if s is None:
            continue
        print(f"  {arm}: acc@12-acc@3 {fmt(acc_gain(s, '3', '12', [3, 4]))}   "
              f"acc@3-acc@1 {fmt(acc_gain(s, '1', '3', [3, 4]))}   "
              f"acc@16-acc@3 {fmt(acc_gain(s, '3', '16', [3, 4]))}   "
              f"[all buckets] acc@12-acc@3 {fmt(acc_gain(s, '3', '12', BUCKETS))}")

    print("\n=== P17d (the diagonal) at 6000: gain(3->12) on bucket 4 minus on bucket 0")
    for arm in ARMS:
        s = S.get((arm, 6000))
        if s is None:
            continue
        print(f"  {arm}: accuracy {fmt(diagonal(s, '3', '12', 4, 0, solve=False))}   "
              f"solve {fmt(diagonal(s, '3', '12', 4, 0, solve=True))}   "
              f"bucket-4 solve@12-solve@3 {fmt(solve_gain(s, '3', '12', [4]))}")

    print("\n=== P17e (the control) at 6000: notul@6 minus mask@12, paired over the same boards")
    if ("sud-mask", 6000) in S and ("sud-notul", 6000) in S:
        acc, solve = between(S["sud-notul", 6000], "6", S["sud-mask", 6000], "12")
        print(f"  accuracy {fmt(acc)}   solve rate {fmt(solve)}")
        for ck in CKS:
            if ("sud-mask", ck) in S and ("sud-notul", ck) in S:
                acc, _ = between(S["sud-notul", ck], "6", S["sud-mask", ck], "12")
                print(f"  @{ck}: accuracy {fmt(acc)}")
    else:
        print("  (needs both 6000 sweeps)")

    print("\n=== P17f/P17g/P17h: wall clock, peak, probes, val curve, learning between 3000 and 6000")
    for arm in ARMS:
        log = os.path.join(root, f"run_{arm}.log")
        probe = os.path.join(root, f"probe_{arm}.jsonl")  # 47 MB: kept out of git
        if not os.path.exists(probe):
            probe = os.path.join(HERE, "..", "..", "..", "..", "ignored", "experiment-artifacts",
                                 "2026-09-08-arc-e17", f"probe_{arm}.jsonl")
        if not os.path.exists(log):
            print(f"  {arm}: no run log")
            continue
        v = val_curve(log)
        ps = probe_stats(probe) if os.path.exists(probe) else {}
        line = (f"  {arm}: wall {WALL_H[arm]:.2f} h  peak {peak_gb(log):.2f} GB  "
                f"val@1500 {v.get(1500, float('nan')):.4f} @3000 {v.get(3000, float('nan')):.4f} "
                f"@4500 {v.get(4500, float('nan')):.4f} last@{max(v)} {v[max(v)]:.4f}  {ps}")
        if (arm, 3000) in S and (arm, 6000) in S:
            t = TRAINED[arm]
            a3, a6 = S[arm, 3000]["depths"][t]["acc_answer"], S[arm, 6000]["depths"][t]["acc_answer"]
            line += f"\n         acc@{t}: 3000 {a3:.4f} -> 6000 {a6:.4f} (delta {a6 - a3:+.4f})"
        print(line)


if __name__ == "__main__":
    main()
