"""Score the H24 arm against its pre-registered predictions, without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-25-h24-hca-branch-arm-binary.md

Committed BEFORE the runs, so no threshold here can be fitted to the data. The validity
gate runs first and refuses the whole panel, as in `score_stage0.py`, `score_scse.py` and
`score_h18.py`.

THE SIGNAL IS BINARY. `diverged` means the run's own guard wrote an `[ABORT] ... step_N`
line. Not a CE threshold, not a rise, not a judgement. Validation CE appears in one place
only, P3, and only to catch the degenerate pass — a run that dodges the guard by learning
nothing would satisfy P1 and mean nothing.

Usage:
    python lab/divergence/score_h24_arm.py --dir /home/wolfe/morph-scratch/h24bin
"""
from __future__ import annotations

import argparse
import os
import re

# ── pre-registered constants ───────────────────────────────────────────────────────
SEEDS = (0, 1)
V1_BY_STEP = 6000        # both controls must abort by here; the RCA aborts are 3240, 4540
P2_MIN_LATER = 0.50      # arm aborts at least this much later than its own control
P3_MAX_CE = 5.0          # arm must actually be training, not just surviving
REFUTE_STEP_TOL = 0.20   # arm aborts within this fraction of its control's step

_VAL = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")
_ABORT = re.compile(r"\[ABORT\][^\n]*?step[_ ](\d+)")
_STEP = re.compile(r"^\[\s*(\d+)/\d+\]")
# The completion marker. CORRECTED 2026-08-25, AFTER the runs, and it cannot change any
# verdict: V1 already failed on its own (control seed 1 did not abort), so the panel is
# refused either way. The bug: V2 read `last_step` from the periodic progress line, whose
# cadence is 200 steps and which the loop's FINAL step does not emit — so a run that
# completed all 6000 steps showed last_step=5800 and was called "a crash or an interrupt".
# Both ctrl-s1 and hca16-s0 wrote `Final checkpoint: .../step_6000.pt` and exited 0.
_DONE = re.compile(r"Final checkpoint:.*?step_(\d+)\.pt")


def read_run(path: str) -> dict:
    """`abort` step or None, the val curve, and the last training step reached."""
    curve, abort, last, done = [], None, None, None
    with open(path, errors="replace") as f:
        for line in f:
            m = _VAL.search(line)
            if m:
                curve.append((int(m.group(1)), float(m.group(2))))
                continue
            m = _STEP.match(line)
            if m:
                last = int(m.group(1))
                continue
            m = _DONE.search(line)
            if m:
                done = int(m.group(1))
                continue
            if abort is None:
                m = _ABORT.search(line)
                if m:
                    abort = int(m.group(1))
    # A run that wrote its final checkpoint reached that step, whatever the progress
    # cadence last printed.
    if done is not None:
        last = done if last is None else max(last, done)
    return {"curve": curve, "abort": abort, "last_step": last, "completed": done,
            "final": curve[-1][1] if curve else None,
            "min": min((v for _, v in curve), default=None),
            "n_evals": len(curve)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    R = {}
    for arm in ("ctrl", "hca16"):
        for sd in SEEDS:
            tag = f"{arm}-s{sd}"
            p = os.path.join(a.dir, f"{tag}.log")
            R[tag] = read_run(p) if os.path.exists(p) else None

    print(f"{'run':>10} {'evals':>6} {'last':>6} {'min CE':>8} {'final CE':>9} "
          f"{'ABORT at':>9}")
    for tag in sorted(R):
        r = R[tag]
        if r is None:
            print(f"{tag:>10}   MISSING")
            continue
        g = lambda k, w, p=4: ("%*.*f" % (w, p, r[k])) if r[k] is not None else " " * w
        print(f"{tag:>10} {r['n_evals']:>6} {str(r['last_step']):>6} {g('min', 8)} "
              f"{g('final', 9)} {str(r['abort']):>9}")

    # ── validity gate ──────────────────────────────────────────────────────────────
    print("\nVALIDITY GATE")
    fails = []
    for tag, r in R.items():
        if r is None or r["n_evals"] == 0:
            fails.append(f"V2: no eval lines for {tag}")
    if not fails:
        for sd in SEEDS:
            c = R[f"ctrl-s{sd}"]
            if c["abort"] is None:
                fails.append(f"V1: control seed {sd} did NOT abort (reached step "
                             f"{c['last_step']}); the RCA regime is 2 of 2 at 3240 and 4540")
            elif c["abort"] > V1_BY_STEP:
                fails.append(f"V1: control seed {sd} aborted at {c['abort']} > {V1_BY_STEP}")
        for tag, r in R.items():
            if r["abort"] is None and r["last_step"] is not None and r["last_step"] < 5900:
                fails.append(f"V2: {tag} stopped at step {r['last_step']} with no [ABORT] "
                             f"line — a crash or an interrupt, not a result")
    if fails:
        print("  FAILED: " + "; ".join(fails))
        print("  V3 was checked with attn_sink_probe.py --geometry before the runs.")
        raise SystemExit(1)
    ctl_at = ", ".join(f"s{sd}@{R[f'ctrl-s{sd}']['abort']}" for sd in SEEDS)
    print(f"  V1 both control seeds aborted by step {V1_BY_STEP}: {ctl_at}")
    print("  V2 every run reached 6000 or aborted on its own guard")
    print("  V3 checked before the runs (geometry probe on both configs)\n")

    # ── panel ──────────────────────────────────────────────────────────────────────
    arm_aborts = {sd: R[f"hca16-s{sd}"]["abort"] for sd in SEEDS}
    ctl_aborts = {sd: R[f"ctrl-s{sd}"]["abort"] for sd in SEEDS}
    p1 = all(v is None for v in arm_aborts.values())

    both = [sd for sd in SEEDS if arm_aborts[sd] is not None]
    p2 = (all(arm_aborts[sd] >= ctl_aborts[sd] * (1 + P2_MIN_LATER) for sd in both)
          if both else None)
    ce = {sd: R[f"hca16-s{sd}"]["final"] for sd in SEEDS}
    p3 = all(v is not None and v < P3_MAX_CE for v in ce.values())

    print("P1  NEITHER arm seed aborts: " +
          ", ".join(f"s{sd}={arm_aborts[sd]}" for sd in SEEDS) +
          f" -> {'HELD' if p1 else 'FAILED'}")
    if both:
        print(f"P2  where the arm aborts, it aborts >= {P2_MIN_LATER:.0%} later than its "
              "own control: " +
              ", ".join(f"s{sd} {ctl_aborts[sd]}->{arm_aborts[sd]}" for sd in both) +
              f" -> {'HELD' if p2 else 'FAILED'}")
    else:
        print("P2  the arm aborted nowhere; not evaluated (P1 covers it)")
    print(f"P3  arm final val CE < {P3_MAX_CE} on both seeds (not a degenerate pass): " +
          ", ".join(f"s{sd}={ce[sd]:.4f}" if ce[sd] is not None else f"s{sd}=none"
                    for sd in SEEDS) + f" -> {'HELD' if p3 else 'FAILED'}")

    close = [sd for sd in both
             if abs(arm_aborts[sd] - ctl_aborts[sd]) <= REFUTE_STEP_TOL * ctl_aborts[sd]]
    refuted = len(both) == len(SEEDS) and len(close) == len(SEEDS)
    print(f"\nREFUTER  arm aborts on BOTH seeds within {REFUTE_STEP_TOL:.0%} of its own "
          f"control: {sorted(close)} of {sorted(SEEDS)} "
          f"-> {'H24 DEAD, as a cure and as a lever' if refuted else 'not refuted'}")

    held = [n for n, v in (("P1", p1), ("P2", p2), ("P3", p3)) if v]
    print(f"\nSUMMARY  {len(held)} held: {', '.join(held) or 'none'}")
    if p1 and p3:
        print("\nP1 AND P3 both held against a control that failed 2 of 2. That is the "
              "strongest\nresult in this campaign — and it is TWO SEEDS. It licenses "
              "seeds 2 and 3 and the\nfull schedule. It does not license the word "
              "\"solved\" yet.")


if __name__ == "__main__":
    main()
