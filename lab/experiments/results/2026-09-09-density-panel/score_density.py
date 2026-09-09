#!/usr/bin/env python
"""Score the density panel (P-den-a..g) from the core_depth_sweep JSONs (+ per-token files),
the anatomy and init-probe JSONs, the probes, run logs and the queue log. References: the
dense Parcae-entry arm at depth 6 (the end point) and the depth-1 control at depth 1 (the
price), both from the Parcae-entry panel's results directory. Pairing and readers:
`lab/divergence/sweep_score.py`.

Usage (repo root): PYTHONPATH=. python lab/experiments/results/2026-09-09-density-panel/score_density.py
                   [RESULTS_DIR] [QUEUE_LOG] [PARCAE_ENTRY_RESULTS_DIR]
"""
from __future__ import annotations

import json
import os
import sys

from lab.divergence.sweep_score import (ci, fmt, last_prune, load_sweep, paired, peak_gb,
                                        probe_stats, prune_at, val_curve, verdicts, wall_h)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARMS = ["density-half", "density-quarter"]
TARGET = {"density-half": 0.5, "density-quarter": 0.25}
DENSE, D1 = "parcae-entry", "plain-depth1"  # the Parcae-entry panel's arms, renamed at filing
DENSE_ALIASES = {"parcae-entry": ["parcae-entry", "e19-loop"], "plain-depth1": ["plain-depth1", "e19-d1"]}
CKS = [2500, 5000]
DEPTHS = ["0", "1", "2", "3", "6", "9", "12", "16"]
ART = tuple(os.path.join(REPO, "ignored", "experiment-artifacts", d)
            for d in ("2026-09-09-density-panel", "2026-09-09-arc-e19"))
VAL_GAP_LIMIT = 0.05
DENSE_WALL_H = 1.00


def yes(b: bool) -> str:
    return "TRUE" if b else "FALSE"


def find(d: str, pattern: str, names: list[str]) -> str | None:
    for n in names:
        p = os.path.join(d, pattern.format(n))
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    queue = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "queue_density.log")
    ref = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "2026-09-09-arc-e19")
    dirs = (root, HERE, ref) + ART
    S = {(a, ck): s for a in ARMS for ck in CKS
         if (s := load_sweep(os.path.join(root, f"sweep_{a}_{ck}.json"))) is not None}
    logs = {a: os.path.join(root, f"run_{a}.log") for a in ARMS}
    for r in (DENSE, D1):
        p = find(ref, "sweep_{}_5000.json", DENSE_ALIASES[r])
        if p:
            S[r, 5000] = load_sweep(p)
        lp = find(ref, "run_{}.log", DENSE_ALIASES[r])
        if lp:
            logs[r] = lp
    vals = {a: val_curve(p, final=True) for a, p in logs.items() if os.path.exists(p)}

    def pair(a, da, b, db):
        return paired(S[a], da, S[b], db, dirs) if (a in S and b in S) else None

    print("=== P-den-b: did density fall? (the trainer's own [prune] lines)")
    for a in ARMS:
        if not os.path.exists(logs[a]):
            print(f"  {a}: no run log")
            continue
        lp = last_prune(logs[a])
        at2500 = prune_at(logs[a], 2500)
        if lp is None:
            print(f"  {a:16s} NO PRUNE LINES -> the run never pruned; nothing below is read for it")
            continue
        print(f"  {a:16s} last prune step {lp[0]} density {lp[1]:.4f} (target {TARGET[a]}) -> within 0.02: {yes(abs(lp[1] - TARGET[a]) <= 0.02)};"
              f"  density at 2500: {at2500:.4f}")

    print("\n=== the E17 rule: sweep CE at depth 6 vs the trainer's held-out loss at that step")
    for (a, ck), s in S.items():
        d = "1" if a == D1 else "6"
        v = vals.get(a, {})
        tv = v.get(ck if ck in v else ck + 1)
        gap = s["depths"][d]["ce_tokens"] - tv if tv is not None else float("nan")
        print(f"  {a:16s}@{ck} depth {d}: sweep {s['depths'][d]['ce_tokens']:.4f}  trainer {tv if tv is not None else float('nan'):.4f}"
              f"  gap {gap:+.4f} {'OK' if abs(gap) < VAL_GAP_LIMIT else 'BUG?'}")

    print("\n=== token CE by forced depth (480 rows) and the paired K-differences")
    print(f"{'arm':16s} {'ck':>5s} " + " ".join(f"{'T=' + d:>7s}" for d in DEPTHS) + " | K1-K6 | K3-K6 | K6-K12")
    for (a, ck), s in S.items():
        k612 = paired(s, "6", s, "12", dirs)
        print(f"{a:16s} {ck:5d} " + " ".join(f"{s['depths'][d]['ce_tokens']:7.4f}" if d in s["depths"] else f"{'-':>7s}" for d in DEPTHS)
              + f" | {ci(s, 'K1-K6')} | {ci(s, 'K3-K6')} | {k612[0]:+.4f}")

    print("\n=== P-den-d (depth) at 5000")
    dense_k16 = S[DENSE, 5000]["ci_ce_tokens"]["K1-K6"]["point"] if (DENSE, 5000) in S else float("nan")
    for a in ARMS:
        if (a, 5000) not in S:
            continue
        s = S[a, 5000]
        k36, k16 = s["ci_ce_tokens"]["K3-K6"], s["ci_ce_tokens"]["K1-K6"]
        k612 = paired(s, "6", s, "12", dirs)
        print(f"  {a:16s} K3-K6 {k36['point']:+.4f} lo {k36['lo']:+.4f} -> {yes(k36['point'] > 0.02 and k36['lo'] > 0)};"
              f"  K1-K6 {k16['point']:+.4f} vs dense {dense_k16:+.4f} -> larger by >0.02: {yes(k16['point'] - dense_k16 > 0.02)};"
              f"  K6-K12 {k612[0]:+.4f} -> within 0.005: {yes(abs(k612[0]) <= 0.005)}")

    print("\n=== P-den-c (the tax): arm@6 - dense@6, token-paired")
    for a in ARMS:
        r = pair((a, 5000), "6", (DENSE, 5000), "6")
        if r:
            band = "0.03-0.10" if a == "density-half" else "0.10-0.25"
            lo, hi = (0.03, 0.10) if a == "density-half" else (0.10, 0.25)
            print(f"  {a:16s} {fmt(r)}  in {band}: {yes(lo <= r[0] <= hi)}  >0.25: {yes(r[0] > 0.25)}  within 0.02: {yes(abs(r[0]) <= 0.02)}")

    print("\n=== P-den-e (the price): arm@6 - depth-1 control@1, token-paired (loses iff CI lo > 0)")
    for a in ARMS + [DENSE]:
        r = pair((a, 5000), "6", (D1, 5000), "1")
        if r:
            print(f"  {a:16s} {fmt(r)}  loses to d1: {yes(r[1] > 0)}  beats d1: {yes(r[2] < 0)}")

    print("\n=== P-den-f (anatomy at 5000, depth 8, 3 rows) and init probes")
    dense_max = float("nan")
    for a in [DENSE] + ARMS:
        ap = find(root if a in ARMS else ref, "anatomy_{}.json", [a] if a in ARMS else DENSE_ALIASES[a])
        if not ap:
            continue
        an = json.load(open(ap))
        mv = an["consecutive"]
        mv6 = mv[5] if len(mv) > 5 else float("nan")
        attn6, mlp6 = an["branch_out_in"]["attn"].get("6", []), an["branch_out_in"]["mlp"].get("6", [])
        mx = max(attn6 + mlp6)
        if a == DENSE:
            dense_max = mx
        inj = an["injection"]
        print(f"  {a:16s} movement {[round(x, 3) for x in mv]}  iter6 > 10 %: {yes(mv6 > 0.10)};  branch out/in @6 max {mx:.3f}"
              f"{'' if a == DENSE else f' (lower than dense by >0.05: {yes(dense_max - mx > 0.05)})'};"
              f"  B diag {inj.get('B_diag_mean', float('nan')):.4f} A {inj['A_mean']:.4f} dt {inj['dt_mean']:.4f};"
              f"  carrier norm t8 {an['carrier'][-1]['norm']:.0f} rank {an['carrier'][-1]['rank']:.1f}")
    for a in [DENSE] + ARMS:
        ip = find(root if a in ARMS else ref, "init_probe_{}.json", [a] if a in ARMS else DENSE_ALIASES[a])
        if not ip:
            continue
        pr = json.load(open(ip))
        at8 = {m: (c.get("8") if "8" in c else c.get(8)) for m, c in pr["ce"].items()}
        spread = max(at8.values()) - min(at8.values())
        print(f"  init probe {a:16s} T8 by start: " + " ".join(f"{m}={v:.4f}" for m, v in at8.items())
              + f"  spread {spread:.4f} -> within 0.01: {yes(spread <= 0.01)}")

    print("\n=== P-den-a/g: survival, tripwire, wall clock, peak, held-out curve")
    wh = wall_h(queue, "density-") if os.path.exists(queue) else {}
    vd = verdicts(queue, "density-") if os.path.exists(queue) else {}
    for a in ARMS + [DENSE]:
        log = logs.get(a)
        if not log or not os.path.exists(log):
            print(f"  {a}: no run log")
            continue
        names = [a] if a in ARMS else DENSE_ALIASES[a]
        probe = next((p for d in dirs for n in names if os.path.exists(p := os.path.join(d, f"probe_{n}.jsonl"))), None)
        ps = probe_stats(probe) if probe else "no probe file"
        v = vals[a]
        w = wh.get(a, float("nan"))
        print(f"  {a:16s} verdict={vd.get(a, '?' if a in ARMS else 'dense ref')}  wall {w:.2f} h ({w / DENSE_WALL_H:.2f}x dense)  "
              f"peak {peak_gb(log):.2f} GB  val@2500 {v.get(2500, float('nan')):.4f}  final {v[max(v)]:.4f}  {ps}")
    if DENSE in vals:
        for a in ARMS:
            if a in vals:
                steps = sorted(set(vals[a]) & set(vals[DENSE]))
                print(f"  {a} - dense held-out (trainer's batches): " +
                      " ".join(f"{s}:{vals[a][s] - vals[DENSE][s]:+.3f}" for s in steps if s % 500 == 0 or s == max(steps)))
    for a in ARMS:
        if a in wh:
            print(f"  P-den-g {a}: wall within 1.1x dense: {yes(wh[a] <= 1.1 * DENSE_WALL_H)}; peak < 16 GB: {yes(peak_gb(logs[a]) < 16)}")


if __name__ == "__main__":
    main()
