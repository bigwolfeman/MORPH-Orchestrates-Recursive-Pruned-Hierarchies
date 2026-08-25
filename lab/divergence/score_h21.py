"""Score H21 -- does the forcing bias predict WHICH A1 seeds diverge? -- without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-24-tul-forcing-bias-predicts-divergence.md

Every threshold here is copied from that file. Nothing is re-derived and nothing is tuned.
The validity gate runs first and refuses the whole panel if it fails, because this campaign
has twice read verdicts off a panel that had not earned them.

TWO DEFINITIONS ARE DECLARED HERE RATHER THAN HIDDEN, because each one changes a verdict:

1. `turnaround()` -- nats risen from the minimum to the LAST validation point -- is inherited
   VERBATIM from `score_stage0.py`, which was committed before any seed in this sweep started.
   It is not re-chosen here. A seed whose CE dips and fully recovers scores 0.0 under it.
   The alternative statistic, the largest rise from the minimum to ANY later point, is also
   computed and printed as `peak_rise`, so the reader can see exactly what the choice costs.
   `peak_rise` NEVER decides a verdict. It is printed because a rise that recovers and a rise
   that does not are different phenomena and the scored rule cannot tell them apart.

2. Turnaround STEP is the step of the validation minimum. This matches the Method amendment,
   which called seed 0's minimum at step 250 the point where "the turnaround began".
   A seed that never turns around gets +inf and ranks last.

Usage:
    python lab/divergence/score_h21.py --dir /home/wolfe/morph-scratch/seedsweep \
        --probe-glob 'drift_s{seed}.json' --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

TURNAROUND_NATS = 0.10   # P1: a seed "turns around" at >= this rise
P1_MIN_SEEDS = 2         # P1: at least this many of four
P2_MAX_INVERSIONS = 1    # P2: "at most one adjacent swap" == one discordant pair
P3_MARGIN = 0.10         # P3: turned-group mean exceeds not-turned mean by >= 10 %
RUNG_P2 = 1000           # P2/P3 read b_t at this step
RUNG_BASE = 500          # P4 compares against this step
R0_TOL = 1e-3            # validity gate on the paper's identity R_0 = 1
GATE_TOL = 1e-2          # the probe's own trajectory-replay tolerance


def b_by_step(path: str) -> dict[int, float]:
    """`{training_step: ||b||/||h*||}`. Raises if b_rel is not flat across loop iterations."""
    out = {}
    for r in json.load(open(path)):
        vals = [x["b_rel"] for x in r["forcing_bias"]]
        spread = (max(vals) - min(vals)) / max(abs(vals[0]), 1e-30)
        if spread > 1e-3:
            raise RuntimeError(
                f"{path} step {r['step']}: b_rel varies {spread:.2e} across loop iterations. "
                f"It is constant by construction before route_start, so either the core map "
                f"gained a t dependence or the probe is reading the wrong operating point.")
        out[int(r["step"])] = vals[0]
    return out


def gate_by_step(path: str) -> dict[int, tuple[float, float]]:
    """`{step: (R_0, trajectory_gate_rel_err)}`."""
    return {int(r["step"]): (r["forcing_bias"][0]["R_t"], r["gate_rel_err"])
            for r in json.load(open(path))}


def val_curve(log_path: str) -> list[tuple[int, float]]:
    pat = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")
    out = []
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def turnaround(curve: list[tuple[int, float]]) -> float:
    """Nats risen from the minimum to the last point. 0.0 if it never rose.

    Inherited verbatim from score_stage0.py. Do not change it here to fit a seed.
    """
    if len(curve) < 2:
        return 0.0
    lo = min(v for _, v in curve)
    return max(0.0, curve[-1][1] - lo)


def peak_rise(curve: list[tuple[int, float]]) -> float:
    """Largest rise from the running minimum to any later point. DIAGNOSTIC ONLY."""
    best = 0.0
    lo = math.inf
    for _, v in curve:
        lo = min(lo, v)
        best = max(best, v - lo)
    return best


def recovery_profile(curve: list[tuple[int, float]]) -> tuple[float, int]:
    """(largest rise that LATER recovered to a new minimum, evals since the last new minimum).

    STRICTLY DESCRIPTIVE. Added on 2026-08-24 AFTER seeing seeds 0 and 1, so it is barred
    from every verdict in this file and is printed only to make one thing visible: a rise
    that recovers and a rise that does not are different phenomena, and the pre-registered
    0.1-nat threshold cannot tell them apart. Seed 1 rose 0.168 nats above its running
    minimum and then set a new minimum, which puts the pre-registered threshold BELOW this
    metric's own noise floor. That is a defect in the pre-registration and it is owned in
    the writeup, not repaired here by moving the threshold after the fact.
    """
    lo = math.inf
    largest_recovered = 0.0
    for i, (_, v) in enumerate(curve):
        if v > lo and any(v2 < lo for _, v2 in curve[i + 1:]):
            largest_recovered = max(largest_recovered, v - lo)
        lo = min(lo, v)
    best = min(v for _, v in curve)
    last_min_i = max(i for i, (_, v) in enumerate(curve) if v == best)
    return largest_recovered, len(curve) - 1 - last_min_i


def min_step(curve: list[tuple[int, float]]) -> int:
    return min(curve, key=lambda sv: sv[1])[0]


def inversions(order_a: list[int], order_b: list[int]) -> int:
    """Adjacent transpositions to turn order_a into order_b (bubble-sort distance)."""
    pos = {s: i for i, s in enumerate(order_b)}
    seq = [pos[s] for s in order_a]
    n = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="seedsweep dir holding s<N>.log")
    ap.add_argument("--probe-glob", default="drift_s{seed}.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()

    seeds, B, gates, curves = [], {}, {}, {}
    for sd in a.seeds:
        p = os.path.join(a.dir, a.probe_glob.format(seed=sd))
        lg = os.path.join(a.dir, f"s{sd}.log")
        if not os.path.exists(p):
            print(f"!! seed {sd}: no probe output at {p} -- EXCLUDED")
            continue
        if not os.path.exists(lg):
            print(f"!! seed {sd}: no training log at {lg} -- EXCLUDED")
            continue
        seeds.append(sd)
        B[sd] = b_by_step(p)
        gates[sd] = gate_by_step(p)
        curves[sd] = val_curve(lg)
    if len(seeds) < 2:
        raise SystemExit("fewer than two seeds usable; nothing to compare")

    print("VALIDITY GATE -- R_0 = 1.000 and trajectory gate ~ 0 at EVERY checkpoint")
    bad = []
    worst_gate = 0.0
    for sd in seeds:
        for step, (r0, ge) in sorted(gates[sd].items()):
            worst_gate = max(worst_gate, ge)
            if abs(r0 - 1.0) > R0_TOL:
                bad.append(f"s{sd}@{step}: R_0={r0:.6f}")
            if ge > GATE_TOL:
                bad.append(f"s{sd}@{step}: gate={ge:.2e}")
    if bad:
        print("  FAILED: " + "; ".join(bad))
        print("  R_0 = 1 is an identity under h* = h_0 = e. It failing means the probe is not "
              "measuring the anchor response, so NOTHING below is readable.")
        raise SystemExit(1)
    print(f"  passed at every checkpoint of every seed (worst trajectory gate {worst_gate:.1e})\n")

    # ---- turnaround table -------------------------------------------------
    print("VALIDATION CURVES")
    print(f"{'seed':>5} {'min CE':>8} {'@step':>6} {'last CE':>8} {'@step':>6} "
          f"{'rise(scored)':>13} {'peak_rise':>10} {'turned?':>8} | {'recov_rise':>10} "
          f"{'evals_since_min':>15}")
    print(f"{'':>5} {'':>8} {'':>6} {'':>8} {'':>6} {'<-- these decide -->':>13} {'':>10} "
          f"{'':>8} | {'<-- descriptive only, no verdict -->':>10}")
    turned, not_turned, tstep = [], [], {}
    for sd in seeds:
        c = curves[sd]
        lo = min(v for _, v in c)
        ms = min_step(c)
        r = turnaround(c)
        pr = peak_rise(c)
        t = r >= TURNAROUND_NATS
        (turned if t else not_turned).append(sd)
        tstep[sd] = ms if t else math.inf
        rr, since = recovery_profile(c)
        print(f"{sd:>5} {lo:>8.4f} {ms:>6} {c[-1][1]:>8.4f} {c[-1][0]:>6} "
              f"{r:>13.3f} {pr:>10.3f} {'YES' if t else 'no':>8} | {rr:>10.3f} {since:>15}")
    print(f"\n  turned around: {turned or '(none)'}   held: {not_turned or '(none)'}")
    floor = max(recovery_profile(curves[sd])[0] for sd in seeds)
    if floor >= TURNAROUND_NATS:
        marginal = [sd for sd in turned
                    if turnaround(curves[sd]) < floor]
        print(f"\n  !! THRESHOLD BELOW NOISE FLOOR. Some seed rose {floor:.3f} nats above its "
              f"running minimum and then set a NEW minimum, so a rise of that size is noise in "
              f"this metric. The pre-registered threshold is {TURNAROUND_NATS:.2f}.")
        if marginal:
            print(f"     Seed(s) {marginal} are scored as turned around on a rise SMALLER than "
                  f"that noise floor. The pre-registered verdict stands as written and is "
                  f"reported as such, but it does not distinguish those seeds from noise. This "
                  f"is a defect in the pre-registration, and the next planned run must set the "
                  f"threshold from a measured noise floor and require N consecutive evals with "
                  f"no new minimum.")
    disagree = [sd for sd in seeds
                if (turnaround(curves[sd]) >= TURNAROUND_NATS)
                != (peak_rise(curves[sd]) >= TURNAROUND_NATS)]
    if disagree:
        print(f"  NOTE: the two rise statistics DISAGREE on seed(s) {disagree}. The scored rule "
              f"(rise to the LAST point) is the one that decides; peak_rise is reported so the "
              f"cost of that inherited choice is visible.")

    # ---- b_t table --------------------------------------------------------
    shared = sorted(set.intersection(*(set(B[sd]) for sd in seeds)))
    print(f"\nFORCING BIAS b_t/|h*| at rungs common to all seeds: {shared}")
    hdr = "".join(f"{('s%d' % sd):>9}" for sd in seeds)
    print(f"{'step':>6}{hdr}")
    for s in shared:
        print(f"{s:>6}" + "".join(f"{B[sd][s]:>9.3f}" for sd in seeds))
    missing = {sd: sorted(set(B[sd]) - set(shared)) for sd in seeds}
    for sd, m in missing.items():
        if m:
            print(f"  s{sd} also has rungs {m} (a run the guard aborted has extra/short rungs)")

    # ---- P1 ---------------------------------------------------------------
    p1 = len(turned) >= P1_MIN_SEEDS
    print(f"\nP1  >= {P1_MIN_SEEDS} of {len(seeds)} seeds rise >= {TURNAROUND_NATS} nats: "
          f"{len(turned)} -> {'HELD' if p1 else 'FAILED'}")
    if not p1:
        print("    Per the pre-registration this makes the experiment UNDERPOWERED rather than "
              "informative: a protocol failure, filed under failures/, and the next planned run "
              "must extend the horizon or raise ademamix_alpha_cap.")

    # ---- P2 ---------------------------------------------------------------
    # Scored on DISCORDANT PAIRS, not bubble-sort distance. Seeds that never turn around
    # are TIED at +inf, and an adjacent-swap count over ties is not defined -- it silently
    # falls back to the tie-break key (the seed index) and can print HELD off a panel where
    # nothing turned over at all. Only pairs with DISTINCT turnaround steps carry
    # information, so only those are counted. One adjacent swap in a strict ordering makes
    # exactly one discordant pair, so the pre-registered "at most one adjacent swap"
    # translates to "at most one discordant pair" without loosening it.
    have1000 = [sd for sd in seeds if RUNG_P2 in B[sd]]
    if len(have1000) < len(seeds):
        print(f"\nP2  NOT SCORABLE: seeds {sorted(set(seeds)-set(have1000))} have no "
              f"step-{RUNG_P2} rung.")
    else:
        comparable = [(i, j) for i in seeds for j in seeds
                      if i < j and tstep[i] != tstep[j]]
        discordant = [(i, j) for (i, j) in comparable
                      if (tstep[i] < tstep[j]) != (B[i][RUNG_P2] > B[j][RUNG_P2])]
        by_b = sorted(seeds, key=lambda sd: -B[sd][RUNG_P2])
        by_t = sorted(seeds, key=lambda sd: (tstep[sd], sd))
        print(f"\nP2  b_t@{RUNG_P2} rank-orders seeds by turnaround step, <= "
              f"{P2_MAX_INVERSIONS} discordant pair")
        print(f"    by b_t desc : {by_b}   ({', '.join('s%d=%.3f' % (sd, B[sd][RUNG_P2]) for sd in by_b)})")
        print(f"    by turnaround: {by_t}   "
              f"({', '.join('s%d=%s' % (sd, tstep[sd] if tstep[sd] != math.inf else 'never') for sd in by_t)})")
        if len(comparable) < 3:
            print(f"    NOT SCORABLE: only {len(comparable)} pair(s) have distinct turnaround "
                  f"steps. With this many ties the ordering carries almost no information, "
                  f"and scoring it would report a rank result the panel cannot support.")
        else:
            p2 = len(discordant) <= P2_MAX_INVERSIONS
            print(f"    {len(discordant)} discordant of {len(comparable)} comparable pairs"
                  + (f" {discordant}" if discordant else "")
                  + f" -> {'HELD' if p2 else 'FAILED'}")
        early = [sd for sd in seeds if tstep[sd] != math.inf and tstep[sd] < RUNG_BASE]
        if early:
            print(f"    PREMISE FAILURE (Method amendment 1): seed(s) {early} turned around "
                  f"BEFORE step {RUNG_BASE}, so b_t@{RUNG_P2} is not 'before any turnaround' "
                  f"for them. Scored as written; the premise failure stands in the record.")

    # ---- P3 ---------------------------------------------------------------
    if not turned or not not_turned:
        print(f"\nP3  NOT SCORABLE: one group is empty "
              f"(turned={turned or 'none'}, held={not_turned or 'none'}).")
    elif not all(RUNG_P2 in B[sd] for sd in seeds):
        print(f"\nP3  NOT SCORABLE: a seed has no step-{RUNG_P2} rung.")
    else:
        mt = sum(B[sd][RUNG_P2] for sd in turned) / len(turned)
        mh = sum(B[sd][RUNG_P2] for sd in not_turned) / len(not_turned)
        p3 = mt >= mh * (1 + P3_MARGIN)
        print(f"\nP3  mean b_t@{RUNG_P2}, turned vs held, margin >= {P3_MARGIN:.0%}: "
              f"{mt:.3f} vs {mh:.3f} = {mt/mh:.3f}x -> {'HELD' if p3 else 'FAILED'}")

    # ---- P4 ---------------------------------------------------------------
    print(f"\nP4  within each turned seed, b_t at the last rung BEFORE turnaround exceeds "
          f"b_t@{RUNG_BASE}")
    if not turned:
        print("    NOT SCORABLE: no seed turned around.")
    for sd in turned:
        rungs = sorted(B[sd])
        pre = [s for s in rungs if s <= tstep[sd]]
        if RUNG_BASE not in B[sd] or not pre or max(pre) <= RUNG_BASE:
            print(f"    s{sd}: NOT SCORABLE -- minimum at step {tstep[sd]}, rungs {rungs}. "
                  f"No rung strictly between {RUNG_BASE} and the turnaround.")
            continue
        last = max(pre)
        ok = B[sd][last] > B[sd][RUNG_BASE]
        print(f"    s{sd}: b@{last}={B[sd][last]:.3f} vs b@{RUNG_BASE}={B[sd][RUNG_BASE]:.3f} "
              f"-> {'HELD' if ok else 'FAILED'}")

    # ---- refuter ----------------------------------------------------------
    print("\nREFUTER  b_t ranges overlap between groups with no ordering")
    if not turned or not not_turned or not all(RUNG_P2 in B[sd] for sd in seeds):
        print("  NOT SCORABLE (a group is empty or a rung is missing).")
    else:
        lo_t = min(B[sd][RUNG_P2] for sd in turned)
        hi_h = max(B[sd][RUNG_P2] for sd in not_turned)
        overlap = lo_t <= hi_h
        print(f"  min(turned)={lo_t:.3f} vs max(held)={hi_h:.3f} -> "
              f"{'RANGES OVERLAP' if overlap else 'ranges separate'}")
        if overlap:
            print("  H21 REFUTED as written: the forcing bias stays arm-intrinsic but is NOT "
                  "predictive of failure. SCSE then loses its causal claim on MORPH, and the "
                  "port rests only on the structural argument that Delta_0 = 0 makes the whole "
                  "trajectory a propagated forcing response.")

    # ---- exploratory ------------------------------------------------------
    print(f"\nEXPLORATORY (Method amendment 1 -- no verdict rests on this): b_t@{RUNG_BASE}")
    for sd in seeds:
        v = B[sd].get(RUNG_BASE)
        print(f"    s{sd}: {v:.3f}" if v is not None else f"    s{sd}: (no rung)")


if __name__ == "__main__":
    main()
