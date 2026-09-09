#!/usr/bin/env python
"""Score the E18 slot-channel width sweep (P18a-g) from the core_depth_sweep JSONs, probes,
run logs and the queue log.

Between-arm CE differences pair at the TOKEN level: every arm scores the same validation
stream, but each cut packs it differently (a row of 1024 + 64·k positions holds a different
number of tokens at each width; the plain cut is 1024 tokens per row), so ROWS are not the
same text across arms — the row means of k2 and k4 correlate at 0.098 across the 480 rows
against 1.000 within an arm. The re-swept JSONs carry a per-token CE array keyed by stream
index (`core_depth_sweep.py` at the fixed commit); the scorer intersects the indices and
bootstraps over 1,024-token stream BLOCKS (tokens in a row are not independent). Without
the token files it falls back to the row-mean comparison and labels it UNPAIRED.
K-differences within an arm are the sweep's own paired bootstraps. The matched-inference-cost metric is block-passes per row (prereg,
Method): plain at core depth T costs 6144 + 6144*T; mask-k at slot depth T' costs
6*(1024 + 64*k) + 384*T'.

Pairing and readers: `lab/divergence/sweep_score.py`.

Usage (repo root): PYTHONPATH=. python lab/experiments/results/2026-09-08-arc-e18/score_e18.py
                   [RESULTS_DIR] [QUEUE_LOG]
"""
from __future__ import annotations

import os
import sys

from lab.divergence.sweep_score import (ci, fmt, load_sweep, paired, peak_gb, probe_stats,
                                        val_curve, verdicts, wall_h)

ARMS = ["e18-mask-k2", "e18-mask-k4", "e18-mask-k8", "e18-notul"]
K = {"e18-mask-k2": 2, "e18-mask-k4": 4, "e18-mask-k8": 8}
TRAINED = {"e18-mask-k2": "12", "e18-mask-k4": "12", "e18-mask-k8": "12", "e18-notul": "6"}
CKS = [2500, 5000]
DEPTHS = ["1", "2", "3", "6", "9", "12", "16"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(d: str, arm: str, ck: int) -> dict | None:
    return load_sweep(os.path.join(d, f"sweep_{arm}_{ck}.json"))


ART = (os.path.join(HERE, "..", "..", "..", "..", "ignored", "experiment-artifacts", "2026-09-08-arc-e18"),)


def cost(arm: str, depth: int) -> int:
    if arm == "e18-notul":
        return 6144 + 6144 * depth
    return 6 * (1024 + 64 * K[arm]) + 384 * depth


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    queue = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "queue_e18.log")
    S = {(a, ck): s for a in ARMS for ck in CKS if (s := load(root, a, ck)) is not None}
    print("=== token CE by forced depth (480 rows), the sweep's paired K-differences")
    print(f"{'arm':12s} {'ck':>5s} " + " ".join(f"{'T=' + d:>7s}" for d in DEPTHS) + " | K1-K6 | K3-K6")
    for (a, ck), s in S.items():
        print(f"{a:12s} {ck:5d} " + " ".join(f"{s['depths'][d]['ce_tokens']:7.4f}" for d in DEPTHS)
              + f" | {ci(s, 'K1-K6')} | {ci(s, 'K3-K6')}")

    print("\n=== P18b (width) at 5000: paired token CE at the trained depth 12")
    for a, b in (("e18-mask-k4", "e18-mask-k2"), ("e18-mask-k8", "e18-mask-k4"), ("e18-mask-k8", "e18-mask-k2")):
        if (a, 5000) in S and (b, 5000) in S:
            print(f"  {a} - {b} @12: {fmt(paired(S[a, 5000], '12', S[b, 5000], '12', ART))}")
    for a in K:
        if (a, 5000) in S and ("e18-notul", 5000) in S:
            print(f"  {a}@12 - notul@6: {fmt(paired(S[a, 5000], '12', S['e18-notul', 5000], '6', ART))}")

    print("\n=== P18c (depth) at 5000: K3-K6 > 0.005 with CI above 0? K1-K6 > 0.03?")
    for a in K:
        if (a, 5000) in S:
            c3, c1 = S[a, 5000]["ci_ce_tokens"]["K3-K6"], S[a, 5000]["ci_ce_tokens"]["K1-K6"]
            print(f"  {a}: K3-K6 {c3['point']:+.4f} lo {c3['lo']:+.4f} -> {'YES' if c3['point'] > 0.005 and c3['lo'] > 0 else 'no'};"
                  f"  K1-K6 {c1['point']:+.4f} -> {'YES' if c1['point'] > 0.03 else 'no'}")

    print("\n=== P18d (matched inference cost) at 5000: block-passes per row and paired CE")
    print(f"  {'arm@depth':22s} {'passes':>7s} {'CE':>8s}   vs notul@1 (12288)      vs notul@2 (18432)")
    for a in K:
        if (a, 5000) not in S:
            continue
        for d in ("3", "6", "12", "16"):
            line = f"  {a + '@' + d:22s} {cost(a, int(d)):7d} {S[a, 5000]['depths'][d]['ce_tokens']:8.4f}"
            if ("e18-notul", 5000) in S:
                line += f"   {fmt(paired(S[a, 5000], d, S['e18-notul', 5000], '1', ART))}   {fmt(paired(S[a, 5000], d, S['e18-notul', 5000], '2', ART))}"
            print(line)
    if ("e18-notul", 5000) in S:
        for d in ("1", "2", "3", "6", "12"):
            print(f"  {'notul@' + d:22s} {cost('e18-notul', int(d)):7d} {S['e18-notul', 5000]['depths'][d]['ce_tokens']:8.4f}")

    print("\n=== P18a/e/f/g: survival, hinge, wall clock, peak, val order")
    wh = wall_h(queue, 'e18-') if os.path.exists(queue) else {}
    vd = verdicts(queue, 'e18-') if os.path.exists(queue) else {}
    finals = {}
    for a in ARMS:
        log = os.path.join(root, f"run_{a}.log")
        probe = os.path.join(root, f"probe_{a}.jsonl")
        if not os.path.exists(probe):
            probe = os.path.join(HERE, "..", "..", "..", "..", "ignored", "experiment-artifacts",
                                 "2026-09-08-arc-e18", f"probe_{a}.jsonl")
        if not os.path.exists(log):
            print(f"  {a}: no run log")
            continue
        v = val_curve(log)
        finals[a] = v[max(v)]
        ps = probe_stats(probe) if os.path.exists(probe) else {}
        print(f"  {a:12s} verdict={vd.get(a, '?')}  wall {wh.get(a, float('nan')):.2f} h  peak {peak_gb(log):.2f} GB  "
              f"val@2500 {v.get(2500, float('nan')):.4f} last@{max(v)} {v[max(v)]:.4f}  {ps}")
    if finals:
        print("  final val order: " + " < ".join(a for a, _ in sorted(finals.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
