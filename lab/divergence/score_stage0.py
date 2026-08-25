"""Score Stage 0 against its pre-registered predictions, without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-24-tul-forcing-bias-arm-control.md

Written and committed BEFORE either arm produced a checkpoint, so the thresholds cannot be
fitted to the data. Every number the verdicts use is read from the two `drift_probe.py`
outputs and the two training logs; nothing here re-derives a threshold.

The validity gate runs first and refuses the whole panel if it fails, in the same spirit as
`score_arms.py` refusing a firing step when the probe cadence is too coarse: the campaign has
twice read verdicts off a panel whose control had not earned them.

Usage:
    python lab/divergence/score_stage0.py --a0 drift_a0.json --a1 drift_a1.json \
        [--a0-log tul_a0.log --a1-log tul_a1.log]
"""
from __future__ import annotations

import argparse
import json
import re


def b_by_step(path: str) -> dict[int, float]:
    """`{training_step: ||b||/||h*||}` from a drift_probe run over a checkpoint dir.

    CORRECTED 2026-08-25. This docstring used to claim MORPH's core map "has no `t`
    dependence before `route_start`". That is FALSE. Measured on seedsweep-s1/step_3500.pt:
    the probe is bit-exact run to run (0.00e+00), pinning `ret_state` to iteration 0's
    collapses the across-iteration spread to exactly 0.00e+00, and pinning `iter_idx`
    instead leaves it unchanged. `T_t` depends on `t` through the GLA retention state
    carried across loop iterations.

    Stage 0's numbers are unaffected: its `b_rel` values are 1.4-2.5, where the measured
    spread stays under the 1e-3 tolerance below, so the guard never fired and iteration 0
    and the mean agree to well within the arm gap being resolved. The claim was still
    wrong. See docs/experiments/failures/2026-08-24-tul-forcing-bias-predicts-divergence.md
    and `B_SPREAD_MAX` in score_h21.py, which reads the mean instead.
    """
    out = {}
    for r in json.load(open(path)):
        fb = r["forcing_bias"]
        vals = [x["b_rel"] for x in fb]
        spread = (max(vals) - min(vals)) / max(abs(vals[0]), 1e-30)
        if spread > 1e-3:
            raise RuntimeError(
                f"{path} step {r['step']}: b_rel varies {spread:.2e} across loop iterations, "
                f"over this file's 1e-3 tolerance. The core map IS t-dependent through the GLA "
                f"retention carry (see the docstring), so a spread this size means the rung is "
                f"outside the range where iteration 0 stands in for the mean. Read it with "
                f"score_h21.b_by_step, which averages over iterations.")
        out[int(r["step"])] = vals[0]
    return out


def r0_by_step(path: str) -> dict[int, float]:
    return {int(r["step"]): r["forcing_bias"][0]["R_t"] for r in json.load(open(path))}


def val_curve(log_path: str) -> list[tuple[int, float]]:
    """`[(step, val_loss)]` from a training log's eval lines."""
    pat = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")
    out = []
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def turnaround(curve: list[tuple[int, float]]) -> float:
    """Nats risen from the minimum to the last point. 0.0 if it never rose."""
    if len(curve) < 2:
        return 0.0
    lo = min(v for _, v in curve)
    return max(0.0, curve[-1][1] - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a0-log")
    ap.add_argument("--a1-log")
    a = ap.parse_args()

    b0, b1 = b_by_step(a.a0), b_by_step(a.a1)
    shared = sorted(set(b0) & set(b1))
    if not shared:
        raise SystemExit("no checkpoint steps in common between the arms")

    print("VALIDITY GATE — R_0 must be 1.000 at every checkpoint of both arms")
    bad = []
    for tag, p in (("A0", a.a0), ("A1", a.a1)):
        for step, r0 in sorted(r0_by_step(p).items()):
            if abs(r0 - 1.0) > 1e-3:
                bad.append(f"{tag}@{step}: R_0={r0:.6f}")
    if bad:
        print("  FAILED: " + "; ".join(bad))
        print("  The identity R_0 = 1 holds under h* = h_0 = e. It failing means the probe is "
              "not measuring the anchor response, so NOTHING below is readable.")
        raise SystemExit(1)
    print("  passed at every checkpoint of both arms\n")

    print(f"{'step':>6} {'A0 b/|h*|':>10} {'A1 b/|h*|':>10} {'A1/A0':>7}")
    for s in shared:
        print(f"{s:>6} {b0[s]:>10.3f} {b1[s]:>10.3f} {b1[s]/b0[s]:>7.3f}")

    first, last = shared[0], shared[-1]
    g0 = b0[last] / b0[first] - 1.0
    g1 = b1[last] / b1[first] - 1.0
    ratio_last = b1[last] / b0[last]
    within10 = all(abs(b1[s] / b0[s] - 1.0) <= 0.10 for s in shared)

    print(f"\nP1  b rises for BOTH arms {first}->{last}: "
          f"A0 {g0:+.1%}, A1 {g1:+.1%} -> {'HELD' if g0 > 0 and g1 > 0 else 'FAILED'}")
    print(f"P2  A1 exceeds A0 by >=15% at the last rung: {ratio_last:.3f} "
          f"-> {'HELD' if ratio_last >= 1.15 else 'FAILED'}")
    print(f"P3  A1 grows faster than A0: {g1:+.1%} vs {g0:+.1%} "
          f"-> {'HELD' if g1 > g0 else 'FAILED'}")

    if a.a0_log and a.a1_log:
        c0, c1 = val_curve(a.a0_log), val_curve(a.a1_log)
        t0, t1 = turnaround(c0), turnaround(c1)
        p4 = t1 >= 0.1 and t0 < 0.1
        print(f"P4  A1 val CE turns around >=0.1 nats and A0 does not: "
              f"A1 +{t1:.3f}, A0 +{t0:.3f} -> {'HELD' if p4 else 'FAILED'}")
        if not p4:
            print("    P1-P3 are then readable as an arm comparison at matched steps, NOT as "
                  "sick-against-healthy. The writeup must say so and must not borrow the "
                  "5090 run's takeover.")
    else:
        print("P4  not evaluated (pass --a0-log and --a1-log)")

    print(f"\nREFUTER  A0 within 10% of A1 at EVERY rung: {within10} "
          f"-> {'H20 REFUTED, forcing bias is a recipe property' if within10 else 'not refuted'}")


if __name__ == "__main__":
    main()
