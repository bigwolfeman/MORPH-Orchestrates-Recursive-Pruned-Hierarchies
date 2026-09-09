#!/usr/bin/env python
"""Score ARC E20 (three loop-depth candidates on the E19 loop arm; P20a-f) from the
core_depth_sweep JSONs (+ per-token files), the anatomy and init-probe JSONs, the probes, run
logs and the queue log. References: the E19 loop arm at depth 6 (the end point) and the E19
depth-1-trained arm at depth 1 (the price), both from `../2026-09-09-arc-e19/`. Pairing and
readers: `lab/divergence/sweep_score.py`.

Usage (repo root): PYTHONPATH=. python lab/experiments/results/2026-09-09-arc-e20/score_e20.py
                   [RESULTS_DIR] [QUEUE_LOG] [E19_RESULTS_DIR]
"""
from __future__ import annotations

import json
import os
import sys

from lab.divergence.sweep_score import (ci, fmt, load_sweep, paired, peak_gb, probe_stats,
                                        val_curve, verdicts, wall_h)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARMS = ["e20-dense-core", "e20-carry-lr20x", "e20-draw8-bptt4"]
TRAINED = {"e20-dense-core": "6", "e20-carry-lr20x": "6", "e20-draw8-bptt4": "8",
           "e19-loop": "6", "e19-d1": "1"}
LOOP, D1 = "e19-loop", "e19-d1"
CKS = [2500, 5000]
DEPTHS = ["0", "1", "2", "3", "6", "8", "9", "12", "16"]
ART = tuple(os.path.join(REPO, "ignored", "experiment-artifacts", d)
            for d in ("2026-09-09-arc-e20", "2026-09-09-arc-e19"))
VAL_GAP_LIMIT = 0.05  # sweep-vs-trainer gap above this is a packing bug (E18/E19: ~0.025)
LOOP_WALL_H = 1.00  # the E19 loop arm's wall clock (queue stamps 02:33:03 -> 03:33:06)


def yes(b: bool) -> str:
    return "TRUE" if b else "FALSE"


def kdiff(s: dict, a: str, b: str, dirs) -> tuple:
    """CE at depth a minus depth b within one sweep (exact token pairing)."""
    return paired(s, a, s, b, dirs)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    queue = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "queue_e20.log")
    e19 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "2026-09-09-arc-e19")
    dirs = (root, HERE, e19) + ART
    S = {(a, ck): s for a in ARMS for ck in CKS
         if (s := load_sweep(os.path.join(root, f"sweep_{a}_{ck}.json"))) is not None}
    for ref in (LOOP, D1):
        r = load_sweep(os.path.join(e19, f"sweep_{ref}_5000.json"))
        if r is not None:
            S[ref, 5000] = r
    vals = {}
    for a in ARMS + [LOOP, D1]:
        log = os.path.join(root if a in ARMS else e19, f"run_{a}.log")
        if os.path.exists(log):
            vals[a] = val_curve(log, final=True)

    def pair(a, da, b, db):
        return paired(S[a], da, S[b], db, dirs) if (a in S and b in S) else None

    print("=== the E17 rule: sweep CE at the trained depth vs the trainer's held-out loss at that step")
    for (a, ck), s in S.items():
        d = TRAINED[a]
        v = vals.get(a, {})
        tv = v.get(ck if ck in v else ck + 1)
        gap = s["depths"][d]["ce_tokens"] - tv if tv is not None else float("nan")
        print(f"  {a:16s}@{ck} depth {d}: sweep {s['depths'][d]['ce_tokens']:.4f}  trainer {tv if tv is not None else float('nan'):.4f}"
              f"  gap {gap:+.4f} {'OK' if abs(gap) < VAL_GAP_LIMIT else 'BUG?'}")

    print("\n=== token CE by forced depth (480 rows) and the paired K-differences (Kt = trained depth)")
    print(f"{'arm':16s} {'ck':>5s} " + " ".join(f"{'T=' + d:>7s}" for d in DEPTHS) + " | K1-Kt | K3-Kt | Kt-K12")
    for (a, ck), s in S.items():
        t = TRAINED[a]
        k1, k3, k12 = kdiff(s, "1", t, dirs), kdiff(s, "3", t, dirs), kdiff(s, t, "12", dirs)
        print(f"{a:16s} {ck:5d} " + " ".join(f"{s['depths'][d]['ce_tokens']:7.4f}" if d in s["depths"] else f"{'-':>7s}" for d in DEPTHS)
              + f" | {k1[0]:+.4f} | {k3[0]:+.4f} [{k3[1]:+.4f},{k3[2]:+.4f}] | {k12[0]:+.4f}")

    print("\n=== P20b (depth) at 5000: K3-Kt > 0.02 with CI > 0; K1-Kt > 0.10; Kt-K12 within +-0.005")
    for a in ARMS:
        if (a, 5000) not in S:
            continue
        s, t = S[a, 5000], TRAINED[a]
        k1, k3, k12 = kdiff(s, "1", t, dirs), kdiff(s, "3", t, dirs), kdiff(s, t, "12", dirs)
        print(f"  {a:16s} K3-K{t} {k3[0]:+.4f} lo {k3[1]:+.4f} -> {yes(k3[0] > 0.02 and k3[1] > 0)};"
              f"  K1-K{t} {k1[0]:+.4f} -> {yes(k1[0] > 0.10)};  K{t}-K12 {k12[0]:+.4f} -> {yes(abs(k12[0]) <= 0.005)};"
              f"  sweep's own K1-K6 {ci(s, 'K1-K6')} K3-K6 {ci(s, 'K3-K6')}")

    print("\n=== P20c (end point): arm@trained - e19-loop@6, token-paired")
    for a in ARMS:
        r = pair((a, 5000), TRAINED[a], (LOOP, 5000), "6")
        if r:
            print(f"  {a:16s} {fmt(r)}  better by >0.02: {yes(r[2] < -0.02)}  within +-0.02: {yes(abs(r[0]) <= 0.02)}  worse by >0.03: {yes(r[1] > 0.03)}")

    print("\n=== P20d (the price): arm@trained - e19-d1@1, token-paired (beats d1 iff CI hi < 0)")
    for a in ARMS + [LOOP]:
        r = pair((a, 5000), TRAINED[a], (D1, 5000), "1")
        if r:
            print(f"  {a:16s} {fmt(r)}  beats d1: {yes(r[2] < 0)}")

    print("\n=== P20e (anatomy at 5000, depth 8, 3 rows) and init probes")
    for a in ARMS + [LOOP]:
        ap = os.path.join(root if a in ARMS else e19, f"anatomy_{a}.json")
        if not os.path.exists(ap):
            continue
        an = json.load(open(ap))
        mv = an["consecutive"]
        mv6 = mv[5] if len(mv) > 5 else float("nan")
        inj = an["injection"]
        attn6, mlp6 = an["branch_out_in"]["attn"].get("6", []), an["branch_out_in"]["mlp"].get("6", [])
        bd = inj.get("B_diag_mean", float("nan"))
        print(f"  {a:16s} movement {[round(x, 3) for x in mv]}  iter6 > 10 %: {yes(mv6 > 0.10)};  B diag {bd:.4f} (moves > 0.3: {yes(abs(bd - 1.0) > 0.3)});"
              f"  A {inj['A_mean']:.4f} (moves > 0.05: {yes(abs(inj['A_mean'] - 0.447) > 0.05)}) dt {inj['dt_mean']:.4f};"
              f"  branch out/in @6 max {max(attn6 + mlp6):.3f}; carrier norm t8 {an['carrier'][-1]['norm']:.0f} rank {an['carrier'][-1]['rank']:.1f}")
    for a in ARMS + [LOOP]:
        ip = os.path.join(root if a in ARMS else e19, f"init_probe_{a}.json")
        if not os.path.exists(ip):
            continue
        pr = json.load(open(ip))
        at8 = {m: (c.get("8") if "8" in c else c.get(8)) for m, c in pr["ce"].items()}
        spread = max(at8.values()) - min(at8.values())
        print(f"  init probe {a:16s} T8 by start: " + " ".join(f"{m}={v:.4f}" for m, v in at8.items())
              + f"  spread {spread:.4f} -> within 0.01: {yes(spread <= 0.01)}")
        print(f"    {'':16s} full curves: " + "; ".join(f"{m}: " + " ".join(f"T{d}={v:.3f}" for d, v in c.items()) for m, c in pr["ce"].items()))

    print("\n=== P20a/f: survival, tripwire, wall clock, peak, held-out curve")
    wh = wall_h(queue, "e20-") if os.path.exists(queue) else {}
    vd = verdicts(queue, "e20-") if os.path.exists(queue) else {}
    for a in ARMS + [LOOP]:
        log = os.path.join(root if a in ARMS else e19, f"run_{a}.log")
        if not os.path.exists(log):
            print(f"  {a}: no run log")
            continue
        probe = next((p for p in [os.path.join(d, f"probe_{a}.jsonl") for d in dirs] if os.path.exists(p)), None)
        ps = probe_stats(probe) if probe else "no probe file"
        v = vals[a]
        w = wh.get(a, float("nan"))
        print(f"  {a:16s} verdict={vd.get(a, '?' if a in ARMS else 'E19')}  wall {w:.2f} h ({w / LOOP_WALL_H:.2f}x loop)  "
              f"peak {peak_gb(log):.2f} GB  val@2500 {v.get(2500, float('nan')):.4f}  final {v[max(v)]:.4f}  {ps}")
    if LOOP in vals:
        for a in ARMS:
            if a in vals:
                steps = sorted(set(vals[a]) & set(vals[LOOP]))
                print(f"  {a} - loop held-out (trainer's batches): " +
                      " ".join(f"{s}:{vals[a][s] - vals[LOOP][s]:+.3f}" for s in steps if s % 1000 == 0 or s == max(steps)))
    if "e20-dense-core" in wh:
        print(f"  P20f: dense-core within 1.1x loop wall: {yes(wh['e20-dense-core'] <= 1.1 * LOOP_WALL_H)}")
    if "e20-draw8-bptt4" in wh:
        r = wh["e20-draw8-bptt4"] / LOOP_WALL_H
        print(f"  P20f: draw8-bptt4 wall {r:.2f}x loop -> in [1.15, 1.5]: {yes(1.15 <= r <= 1.5)}")
    if "e20-draw8-bptt4" in vals and LOOP in vals and "e20-draw8-bptt4" in wh:
        # matched wall clock: the draw8 arm's held-out loss at the step it has reached when the
        # (faster) loop arm finishes its 5000, against the loop arm's final
        step_at = int(5000 * LOOP_WALL_H / wh["e20-draw8-bptt4"] / 250) * 250
        dv = vals["e20-draw8-bptt4"].get(step_at)
        lf = vals[LOOP][max(vals[LOOP])]
        if dv is not None:
            print(f"  matched wall clock: draw8@{step_at} {dv:.4f} vs loop final {lf:.4f} -> {dv - lf:+.4f}"
                  f" (trainer's own batches; within 0.03: {yes(dv - lf <= 0.03)})")

if __name__ == "__main__":
    main()
