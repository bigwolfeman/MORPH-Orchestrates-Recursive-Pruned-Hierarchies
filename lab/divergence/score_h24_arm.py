"""Score the H24 arm against its pre-registered predictions, without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-arm.md

Committed BEFORE the runs, so no threshold here can be fitted to the data. The validity
gate runs first and refuses the whole panel, as in `score_stage0.py`, `score_scse.py` and
`score_h18.py`.

ONE SEED [W, 2026-08-25]. What that costs is written into the plan file and repeated in
the output: a HELD panel here is a SCREEN result that licenses a multi-seed arm, not a
claim that the branch is a lever. A REFUTED panel is strong at n=1, because "the arm blew
up the same way at the same step" needs no seed count.

Usage:
    python lab/divergence/score_h24_arm.py --dir /home/wolfe/morph-scratch/h24arm
"""
from __future__ import annotations

import argparse
import os
import re

# ── pre-registered constants ───────────────────────────────────────────────────────
RISE_DIVERGED = 0.35     # nats above a run's own minimum; healthy noise floor is 0.168
V1_CONTROL_BY = 3000     # the control must fail by here; the seed sweep aborted at 2040
P2_MIN_GAIN = 0.20       # nats, arm below control at the control's failure step
P3_MIN_MIN_GAIN = 0.10   # nats, arm's run minimum below the control's
REFUTE_STEP_TOL = 0.20   # arm aborts within this fraction of the control's abort step
REFUTE_CE_TOL = 0.05     # nats

_VAL = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")
_ABORT = re.compile(r"\[ABORT\][^\n]*?step[_ ](\d+)")


def curve(path: str) -> list[tuple[int, float]]:
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = _VAL.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def abort_step(path: str) -> int | None:
    """The step in the run's own `[ABORT] ... step_N` line, or None if it never fired.

    Read from the LOG and not from an exit code: the control of this experiment was
    orphaned when the rejected 4-seed runner was stopped, so no wrapper recorded its exit.
    The log line is the run's own statement and is available either way.
    """
    with open(path, errors="replace") as f:
        for line in f:
            m = _ABORT.search(line)
            if m:
                return int(m.group(1))
    return None


def summarise(path: str) -> dict:
    """Everything the panel reads from one run's log."""
    c = curve(path)
    ab = abort_step(path)
    if not c:
        return {"n_evals": 0, "curve": [], "abort": ab, "diverged": None,
                "fail_step": ab, "final": None, "min": None, "rise": None,
                "last_step": None}
    lo = min(v for _, v in c)
    final = c[-1][1]
    # First eval at or past the threshold measured against the minimum seen SO FAR, so a
    # late minimum cannot retro-actively erase an earlier turnaround.
    fail_step, run_lo = None, c[0][1]
    for st, v in c:
        run_lo = min(run_lo, v)
        if v - run_lo >= RISE_DIVERGED:
            fail_step = st
            break
    if ab is not None and (fail_step is None or ab < fail_step):
        fail_step = ab
    return {"n_evals": len(c), "curve": c, "abort": ab,
            "diverged": bool(ab is not None or (final - lo) >= RISE_DIVERGED),
            "fail_step": fail_step, "final": final, "min": lo, "rise": final - lo,
            "last_step": c[-1][0]}


def at_or_before(c: list[tuple[int, float]], step: int) -> tuple[int, float] | None:
    """The last evaluation at or before `step`."""
    hits = [(s, v) for s, v in c if s <= step]
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    R = {}
    for tag in ("ctrl-s0", "hca16-s0"):
        p = os.path.join(a.dir, f"{tag}.log")
        R[tag] = summarise(p) if os.path.exists(p) else None

    print(f"{'run':>10} {'evals':>6} {'last':>6} {'min':>8} {'final':>8} {'rise':>7} "
          f"{'ABORT':>7} {'DIVERGED':>9} {'at':>6}")
    for tag in ("ctrl-s0", "hca16-s0"):
        r = R[tag]
        if r is None:
            print(f"{tag:>10}   MISSING")
            continue
        f = lambda k, w, p=4: ("%*.*f" % (w, p, r[k])) if r[k] is not None else " " * w
        print(f"{tag:>10} {r['n_evals']:>6} {str(r['last_step']):>6} {f('min', 8)} "
              f"{f('final', 8)} {f('rise', 7, 3)} {str(r['abort']):>7} "
              f"{str(r['diverged']):>9} {str(r['fail_step']):>6}")

    # ── validity gate ──────────────────────────────────────────────────────────────
    print("\nVALIDITY GATE")
    fails = []
    for tag in ("ctrl-s0", "hca16-s0"):
        if R[tag] is None or R[tag]["n_evals"] == 0:
            fails.append(f"V2: no eval lines for {tag}")
    if not fails:
        c, arm = R["ctrl-s0"], R["hca16-s0"]
        if not (c["diverged"] and c["fail_step"] is not None
                and c["fail_step"] <= V1_CONTROL_BY):
            fails.append(f"V1: the control did not fail by step {V1_CONTROL_BY} "
                         f"(diverged={c['diverged']}, at={c['fail_step']}); the seed sweep "
                         f"aborted at 2040, so there is nothing to compare against")
        for tag, r in (("ctrl-s0", c), ("hca16-s0", arm)):
            if r["abort"] is None and r["last_step"] is not None and r["last_step"] < 5750:
                fails.append(f"V2: {tag} stopped at step {r['last_step']} with no [ABORT] "
                             f"line — a crash or an interrupt, not a result")
    if fails:
        print("  FAILED: " + "; ".join(fails))
        print("  V3 was checked with attn_sink_probe.py --geometry before the runs.")
        raise SystemExit(1)
    c, arm = R["ctrl-s0"], R["hca16-s0"]
    print(f"  V1 the control failed at step {c['fail_step']} (needed <= {V1_CONTROL_BY})")
    print("  V2 both runs finished or aborted on their own guard")
    print("  V3 checked before the runs (geometry probe on both configs)\n")

    # ── panel ──────────────────────────────────────────────────────────────────────
    fs = c["fail_step"]
    p1 = (arm["fail_step"] is None or arm["fail_step"] > fs) and (
        arm["last_step"] is not None and arm["last_step"] >= fs)
    ca, aa = at_or_before(c["curve"], fs), at_or_before(arm["curve"], fs)
    gain = (ca[1] - aa[1]) if (ca and aa) else float("nan")
    p2 = bool(ca and aa and gain >= P2_MIN_GAIN)
    min_gain = c["min"] - arm["min"]
    p3 = min_gain >= P3_MIN_MIN_GAIN

    print(f"P1  arm still training at the control's failure step {fs}: "
          f"arm fail_step={arm['fail_step']}, arm reached {arm['last_step']} "
          f"-> {'HELD' if p1 else 'FAILED'}")
    if ca and aa:
        print(f"P2  arm CE below control by >= {P2_MIN_GAIN} at the last common eval "
              f"<= {fs} (step {ca[0]} vs {aa[0]}): {ca[1]:.4f} - {aa[1]:.4f} = "
              f"{gain:+.4f} -> {'HELD' if p2 else 'FAILED'}")
    else:
        print(f"P2  no common evaluation at or before step {fs}; not evaluable")
    print(f"P3  arm run-minimum below the control's by >= {P3_MIN_MIN_GAIN}: "
          f"{c['min']:.4f} - {arm['min']:.4f} = {min_gain:+.4f} "
          f"-> {'HELD' if p3 else 'FAILED'}")

    # ── refuter ────────────────────────────────────────────────────────────────────
    close_step = (arm["fail_step"] is not None and fs
                  and abs(arm["fail_step"] - fs) <= REFUTE_STEP_TOL * fs)
    close_ce = abs(min_gain) < REFUTE_CE_TOL
    refuted = bool(close_step and close_ce)
    print(f"\nREFUTER  arm fails within {REFUTE_STEP_TOL:.0%} of the control's step "
          f"({close_step}) AND run-minima within {REFUTE_CE_TOL} ({close_ce}) "
          f"-> {'H24 REFUTED as a lever at m=16' if refuted else 'not refuted'}")

    held = [n for n, v in (("P1", p1), ("P2", p2), ("P3", p3)) if v]
    print(f"\nSUMMARY  {len(held)}/3 held: {', '.join(held) or 'none'}")
    print("\nSCOPE, fixed before the run: n = 1. A HELD panel is a SCREEN result that "
          "licenses a\nmulti-seed arm. It is NOT a claim that the branch is a lever. A "
          "REFUTED panel is strong\nat n = 1, because 'the arm blew up the same way at the "
          "same step' needs no seed count.")


if __name__ == "__main__":
    main()
