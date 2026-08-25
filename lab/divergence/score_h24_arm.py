"""Score the H24 arm against its pre-registered predictions, without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-arm.md

Committed BEFORE the runs, so no threshold here can be fitted to the data. The validity
gate runs first and refuses the whole panel, as in `score_stage0.py`, `score_scse.py` and
`score_h18.py`.

Usage:
    python lab/divergence/score_h24_arm.py --dir /home/wolfe/morph-scratch/h24arm
"""
from __future__ import annotations

import argparse
import os
import re

# ── pre-registered constants ───────────────────────────────────────────────────────
SEEDS = (0, 1, 2, 3)
RISE_DIVERGED = 0.35     # nats above a run's own minimum; healthy noise floor is 0.168
V1_MIN_SURVIVORS = 2     # of control seeds 1,2,3
V1_SURVIVOR_STEP = 3250
P2_MIN_LATER = 0.20      # arm's failure step at least 20 % later
P3_MIN_SEEDS = 3         # arm's final CE lower on this many seeds
P3_MIN_MEAN_GAIN = 0.10  # nats, mean over surviving seeds
REFUTE_CE_TOL = 0.05     # nats

_VAL = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")


def curve(path: str) -> list[tuple[int, float]]:
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = _VAL.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def summarise(path: str, exit_code: int | None) -> dict:
    """`diverged`, the step it happened, and the CE numbers the panel reads."""
    c = curve(path)
    if not c:
        return {"n_evals": 0, "diverged": None, "fail_step": None,
                "final": None, "min": None, "rise": None, "last_step": None,
                "exit": exit_code}
    lo = min(v for _, v in c)
    final = c[-1][1]
    rise = final - lo
    # First eval at or past the threshold, measured against the minimum seen SO FAR, so a
    # late minimum cannot retro-actively erase an earlier turnaround.
    fail_step, run_lo = None, c[0][1]
    for st, v in c:
        run_lo = min(run_lo, v)
        if v - run_lo >= RISE_DIVERGED:
            fail_step = st
            break
    guard = exit_code is not None and exit_code != 0
    if guard and fail_step is None:
        fail_step = c[-1][0]
    return {"n_evals": len(c), "diverged": bool(guard or rise >= RISE_DIVERGED),
            "guard": guard, "fail_step": fail_step, "final": final, "min": lo,
            "rise": rise, "last_step": c[-1][0], "exit": exit_code}


def exits(d: str) -> dict[str, int]:
    """`{tag: exit_code}` from the runner's own echo lines."""
    out = {}
    for name in ("runner.log", "h24_runner.log"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            for line in open(p, errors="replace"):
                m = re.match(r"(\S+) exit=(\d+)", line.strip())
                if m:
                    out[m.group(1)] = int(m.group(2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    ex = exits(a.dir)

    R = {}
    for arm in ("ctrl", "hca16"):
        for sd in SEEDS:
            tag = f"{arm}-s{sd}"
            p = os.path.join(a.dir, f"{tag}.log")
            R[tag] = summarise(p, ex.get(tag)) if os.path.exists(p) else None

    print(f"{'run':>10} {'evals':>6} {'last':>6} {'min':>8} {'final':>8} {'rise':>7} "
          f"{'exit':>5} {'DIVERGED':>9} {'at':>6}")
    for tag in sorted(R):
        r = R[tag]
        if r is None:
            print(f"{tag:>10}   MISSING")
            continue
        f = lambda k, w, p=4: ("%*.*f" % (w, p, r[k])) if r[k] is not None else " " * w
        print(f"{tag:>10} {r['n_evals']:>6} {str(r['last_step']):>6} {f('min',8)} "
              f"{f('final',8)} {f('rise',7,3)} {str(r['exit']):>5} "
              f"{str(r['diverged']):>9} {str(r['fail_step']):>6}")

    # ── validity gate ──────────────────────────────────────────────────────────────
    print("\nVALIDITY GATE")
    fails = []
    missing = [t for t, r in R.items() if r is None or r["n_evals"] == 0]
    if missing:
        fails.append(f"V2: no eval lines for {sorted(missing)}")
    c0 = R.get("ctrl-s0")
    if c0 and c0["n_evals"] and not c0["diverged"]:
        fails.append("V1: control seed 0 did NOT diverge; every prior control lost it")
    surv = [sd for sd in (1, 2, 3)
            if (r := R.get(f"ctrl-s{sd}")) and r["n_evals"]
            and not r["diverged"] and r["last_step"] >= V1_SURVIVOR_STEP]
    if len(surv) < V1_MIN_SURVIVORS:
        fails.append(f"V1: only {len(surv)} of control seeds 1-3 survived to "
                     f"{V1_SURVIVOR_STEP} (needed {V1_MIN_SURVIVORS}): {surv}")
    if fails:
        print("  FAILED: " + "; ".join(fails))
        print("  V3 was checked with attn_sink_probe.py --geometry before the runs.")
        raise SystemExit(1)
    print(f"  V1 control seed 0 diverged; {len(surv)} of seeds 1-3 survived to "
          f"{V1_SURVIVOR_STEP}")
    print("  V2 every run produced eval lines")
    print("  V3 checked before the runs (geometry probe on both configs)\n")

    # ── panel ──────────────────────────────────────────────────────────────────────
    dc = {sd for sd in SEEDS if R[f"ctrl-s{sd}"]["diverged"]}
    da = {sd for sd in SEEDS if R[f"hca16-s{sd}"]["diverged"]}
    p1 = len(da) < len(dc)
    both = sorted(dc & da)
    later = [(sd, R[f"ctrl-s{sd}"]["fail_step"], R[f"hca16-s{sd}"]["fail_step"])
             for sd in both]
    p2 = all(b is not None and c is not None and b >= c * (1 + P2_MIN_LATER)
             for _, c, b in later) if later else None
    lower = [sd for sd in SEEDS
             if R[f"hca16-s{sd}"]["final"] < R[f"ctrl-s{sd}"]["final"]]
    alive = sorted(set(SEEDS) - dc - da)
    gain = ([R[f"ctrl-s{sd}"]["final"] - R[f"hca16-s{sd}"]["final"] for sd in alive]
            if alive else [])
    mean_gain = sum(gain) / len(gain) if gain else float("nan")
    p3 = len(lower) >= P3_MIN_SEEDS and (mean_gain >= P3_MIN_MEAN_GAIN if gain else False)
    p4 = len(da) >= 1

    print(f"P1  arm diverges on fewer seeds: ctrl {sorted(dc)} vs arm {sorted(da)} "
          f"-> {'HELD' if p1 else 'FAILED'}")
    if later:
        print("P2  arm fails later where both fail: " +
              ", ".join(f"s{sd} {c}->{b}" for sd, c, b in later) +
              f" -> {'HELD' if p2 else 'FAILED'}")
    else:
        print("P2  no seed diverged in BOTH arms; not evaluated")
    print(f"P3  arm CE lower on >={P3_MIN_SEEDS}/4 seeds and mean gain >= "
          f"{P3_MIN_MEAN_GAIN}: lower on {sorted(lower)}, mean gain over surviving "
          f"seeds {alive} = {mean_gain:+.4f} -> {'HELD' if p3 else 'FAILED'}")
    print(f"P4  arm still diverges somewhere: {sorted(da)} -> "
          f"{'HELD' if p4 else 'FAILED — and that is BETTER than predicted, not a miss'}")

    mc = sum(R[f"ctrl-s{sd}"]["final"] for sd in SEEDS) / len(SEEDS)
    ma = sum(R[f"hca16-s{sd}"]["final"] for sd in SEEDS) / len(SEEDS)
    refuted = dc == da and abs(mc - ma) < REFUTE_CE_TOL
    print(f"\nREFUTER  same divergence set ({dc == da}) and mean final CE within "
          f"{REFUTE_CE_TOL}: {mc:.4f} vs {ma:.4f} = {abs(mc - ma):.4f} "
          f"-> {'H24 REFUTED as a lever' if refuted else 'not refuted'}")
    held = [n for n, v in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4)) if v]
    print(f"\nSUMMARY  {len(held)} held: {', '.join(held) or 'none'}")


if __name__ == "__main__":
    main()
