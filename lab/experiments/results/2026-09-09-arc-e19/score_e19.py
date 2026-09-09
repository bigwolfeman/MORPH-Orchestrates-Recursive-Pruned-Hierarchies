#!/usr/bin/env python
"""Score ARC E19 (Parcae's loop entry on the plain model; P19a-g) from the core_depth_sweep
JSONs (+ per-token files), the anatomy and init-probe JSONs, the probes, run logs and the
queue log. The reference arm is the E18 plain model (`../2026-09-08-arc-e18/`, its 5000
sweep + token file). Pairing and readers: `lab/divergence/sweep_score.py`.

Usage (repo root): PYTHONPATH=. python lab/experiments/results/2026-09-09-arc-e19/score_e19.py
                   [RESULTS_DIR] [QUEUE_LOG]
"""
from __future__ import annotations

import json
import os
import sys

from lab.divergence.sweep_score import (ci, fmt, load_sweep, paired, peak_gb, probe_stats,
                                        val_curve, verdicts, wall_h)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARMS = ["parcae-entry", "plain-no-fixed-point", "plain-depth1"]
TRAINED = {"parcae-entry": "6", "plain-no-fixed-point": "6", "plain-depth1": "1"}
PLAIN = "e18-notul"  # the E18 plain arm: same recipe, seed and stream; trained depth 6
CKS = [2500, 5000]
DEPTHS = ["0", "1", "2", "3", "6", "9", "12", "16"]
E18_DIR = os.path.join(HERE, "..", "2026-09-08-arc-e18")
ART = tuple(os.path.join(REPO, "ignored", "experiment-artifacts", d)
            for d in ("2026-09-09-arc-e19", "2026-09-08-arc-e18"))  # the first is the Parcae-entry panel
VAL_GAP_LIMIT = 0.05  # sweep-vs-trainer gap above this is a packing bug (E18 plain: 0.026)


def yes(b: bool) -> str:
    return "TRUE" if b else "FALSE"


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    queue = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "queue_parcae_entry.log")
    dirs = (root, HERE) + ART
    S = {(a, ck): s for a in ARMS for ck in CKS
         if (s := load_sweep(os.path.join(root, f"sweep_{a}_{ck}.json"))) is not None}
    P = load_sweep(os.path.join(E18_DIR, f"sweep_{PLAIN}_5000.json"))
    if P is not None:
        S[PLAIN, 5000] = P
    vals = {a: val_curve(os.path.join(root, f"run_{a}.log"), final=True)
            for a in ARMS if os.path.exists(os.path.join(root, f"run_{a}.log"))}
    if os.path.exists(os.path.join(E18_DIR, f"run_{PLAIN}.log")):
        vals[PLAIN] = val_curve(os.path.join(E18_DIR, f"run_{PLAIN}.log"), final=True)

    def pair(a, da, b, db):
        return paired(S[a], da, S[b], db, dirs) if (a in S and b in S) else None

    print("=== the E17 rule: sweep CE at the trained depth vs the trainer's held-out loss at that step")
    for (a, ck), s in S.items():
        d = TRAINED.get(a, "6")
        v = vals.get(a, {})
        tv = v.get(ck if ck in v else ck + 1)
        gap = s["depths"][d]["ce_tokens"] - tv if tv is not None else float("nan")
        print(f"  {a:10s}@{ck} depth {d}: sweep {s['depths'][d]['ce_tokens']:.4f}  trainer {tv if tv is not None else float('nan'):.4f}"
              f"  gap {gap:+.4f} {'OK' if abs(gap) < VAL_GAP_LIMIT else 'BUG?'}")

    print("\n=== token CE by forced depth (480 rows) and the sweep's paired K-differences")
    print(f"{'arm':10s} {'ck':>5s} " + " ".join(f"{'T=' + d:>7s}" for d in DEPTHS) + " | K1-K6 | K3-K6 | K6-K12")
    for (a, ck), s in S.items():
        k612 = paired(s, "6", s, "12", dirs)
        print(f"{a:10s} {ck:5d} " + " ".join(f"{s['depths'][d]['ce_tokens']:7.4f}" if d in s["depths"] else f"{'-':>7s}" for d in DEPTHS)
              + f" | {ci(s, 'K1-K6')} | {ci(s, 'K3-K6')} | {k612[0]:+.4f}")

    if ("parcae-entry", 5000) in S:
        s = S["parcae-entry", 5000]
        k16, k36 = s["ci_ce_tokens"]["K1-K6"], s["ci_ce_tokens"]["K3-K6"]
        k612 = paired(s, "6", s, "12", dirs)
        t0 = s["depths"]["0"]["ce_tokens"] - s["depths"]["1"]["ce_tokens"]
        print("\n=== P19b (the loop arm earns depth) at 5000")
        print(f"  K1-K6 > 0.15:            {k16['point']:+.4f} -> {yes(k16['point'] > 0.15)}")
        print(f"  K3-K6 > 0.02, CI > 0:    {k36['point']:+.4f} lo {k36['lo']:+.4f} -> {yes(k36['point'] > 0.02 and k36['lo'] > 0)}")
        print(f"  K6-K12 within +-0.005:   {k612[0]:+.4f} -> {yes(abs(k612[0]) <= 0.005)}")
        print(f"  CE(T=0) - CE(T=1) > 1.0: {t0:+.4f} -> {yes(t0 > 1.0)}")
        print(f"  binding clause 'P19b FALSE (K3-K6 < 0.01)': {yes(k36['point'] < 0.01)}")

    print("\n=== P19c (the loop arm's end point): loop@6 - E18 plain@6, token-paired")
    r = pair(("parcae-entry", 5000), "6", (PLAIN, 5000), "6")
    if r:
        print(f"  {fmt(r)}  within +0.05: {yes(r[0] < 0.05)}  better: {yes(r[2] < 0)}  >0.15 behind: {yes(r[0] > 0.15)}")

    print("\n=== P19d (fp0 alone): K3-K6 > 0.005 with CI > 0; fp0@6 - E18 plain@6 within 0.02, token-paired")
    if ("plain-no-fixed-point", 5000) in S:
        c = S["plain-no-fixed-point", 5000]["ci_ce_tokens"]["K3-K6"]
        print(f"  K3-K6 {c['point']:+.4f} lo {c['lo']:+.4f} -> {yes(c['point'] > 0.005 and c['lo'] > 0)}")
        r = pair(("plain-no-fixed-point", 5000), "6", (PLAIN, 5000), "6")
        if r:
            print(f"  fp0@6 - plain@6: {fmt(r)} -> within 0.02: {yes(abs(r[0]) <= 0.02)}")

    print("\n=== P19e (the price of the loop): CE(x@6) - CE(d1@1), token-paired")
    if ("plain-depth1", 5000) in S:
        for a in (PLAIN, "parcae-entry", "plain-no-fixed-point"):
            r = pair((a, 5000), "6", ("plain-depth1", 5000), "1")
            if r:
                print(f"  {a:10s}@6 - d1@1: {fmt(r)}")
        r = pair((PLAIN, 5000), "6", ("plain-depth1", 5000), "1")
        if r:
            print(f"  plain: > 0.05 {yes(r[0] > 0.05)}; > 0.15 {yes(r[0] > 0.15)}; < 0.02 (H19'': the loop was free) {yes(r[0] < 0.02)}")
        r = pair(("parcae-entry", 5000), "6", ("plain-depth1", 5000), "1")
        if r:
            print(f"  loop@6 beats d1@1 (CI hi < 0): {yes(r[2] < 0)}")
        print("  d1's own K-curve (a depth-1-trained model run deeper) is in the table above; its trained depth is 1.")

    print("\n=== P19f (anatomy of the loop arm at 5000, depth 8, 3 rows)")
    ap = os.path.join(root, "anatomy_parcae-entry.json")
    if os.path.exists(ap):
        an = json.load(open(ap))
        mv = an["consecutive"]
        mv6 = mv[5] if len(mv) > 5 else float("nan")  # movement into iteration 6 (t5 -> t6)
        t6 = "6"
        attn6, mlp6 = an["branch_out_in"]["attn"].get(t6, []), an["branch_out_in"]["mlp"].get(t6, [])
        inj = an["injection"]
        print(f"  consecutive movement: {[round(x, 4) for x in mv]}")
        print(f"  movement at iteration 6 > 10 %: {mv6:.4f} -> {yes(mv6 > 0.10)}")
        print(f"  a block out/in > 0.2 at iteration 6: max {max(attn6 + mlp6):.4f} (attn {[round(x, 3) for x in attn6]}, "
              f"mlp {[round(x, 3) for x in mlp6]}) -> {yes(max(attn6 + mlp6) > 0.2)}")
        bd = inj.get("B_diag_mean", float("nan"))
        print(f"  B diag mean moves > 0.3 from 1.0: {bd:.4f} -> {yes(abs(bd - 1.0) > 0.3)}")
        print(f"  injection A mean {inj['A_mean']:.4f} [{inj['A_min']:.4f}, {inj['A_max']:.4f}] (init 0.447)  dt {inj['dt_mean']:.4f} (init 0.8)  span {inj['span']}")
        print(f"  h0 norm {an['h0']['norm']:.1f} rank {an['h0']['rank']:.1f}; carrier norm t1..t8 {[round(c['norm']) for c in an['carrier']]}; "
              f"rank {[round(c['rank'], 1) for c in an['carrier']]}; cos to h0 {[round(c['cos_to_h0'], 3) for c in an['carrier']]}")
    for a in ARMS:
        ip = os.path.join(root, f"init_probe_{a}.json")
        if not os.path.exists(ip):
            continue
        pr = json.load(open(ip))
        print(f"  init probe {a} (96 rows, noise std {pr['noise_std']}):")
        for mode, curve in pr["ce"].items():
            print(f"    {mode:12s} " + " ".join(f"T{d}={v:.3f}" for d, v in curve.items()))
        if a == "parcae-entry" and (a, 5000) in S:
            s = S[a, 5000]
            first = min(pr["rows"], len(s["row_n_tokens"]))
            sw6 = sum(s["row_ce_sum"]["6"][:first]) / sum(s["row_n_tokens"][:first])
            ns6 = pr["ce"]["noise_small"].get("6") or pr["ce"]["noise_small"].get(6)
            print(f"    sanity: noise_small@T6 {ns6:.4f} vs the sweep's trained init on the same first {first} rows @T6 {sw6:.4f}"
                  f" -> within 0.01: {yes(abs(ns6 - sw6) <= 0.01)} (the prereg named T8; the sweep has no depth 8)")

    print("\n=== P19a/g: survival, tripwire, wall clock, peak, held-out curve")
    wh = wall_h(queue, "") if os.path.exists(queue) else {}
    vd = verdicts(queue, "") if os.path.exists(queue) else {}
    for a in ARMS + [PLAIN]:
        log = os.path.join(root if a != PLAIN else E18_DIR, f"run_{a}.log")
        if not os.path.exists(log):
            print(f"  {a}: no run log")
            continue
        probe = next((p for p in [os.path.join(d, f"probe_{a}.jsonl") for d in dirs] if os.path.exists(p)), None)
        ps = probe_stats(probe) if probe else "no probe file"
        v = vals[a]
        print(f"  {a:10s} verdict={vd.get(a, '?' if a != PLAIN else 'E18')}  wall {wh.get(a, float('nan')):.2f} h  "
              f"peak {peak_gb(log):.2f} GB  val@2500 {v.get(2500, float('nan')):.4f}  final {v[max(v)]:.4f}  {ps}")
    if "parcae-entry" in vals and PLAIN in vals:
        steps = sorted(set(vals["parcae-entry"]) & set(vals[PLAIN]))
        print("  loop - plain held-out (trainer's own batches): " +
              " ".join(f"{s}:{vals['parcae-entry'][s] - vals[PLAIN][s]:+.3f}" for s in steps if s % 500 == 0 or s == max(steps)))
    if "parcae-entry" in wh:
        print(f"  P19g: loop <= 1.3 h: {yes(wh['parcae-entry'] <= 1.3)}; peak < 16 GB: {yes(peak_gb(os.path.join(root, 'run_parcae-entry.log')) < 16)}")


if __name__ == "__main__":
    main()
