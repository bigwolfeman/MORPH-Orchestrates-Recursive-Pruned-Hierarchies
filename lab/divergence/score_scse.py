"""Score the full-SCSE experiment against its pre-registered predictions.

Pre-registration: docs/experiments/planned/2026-08-25-scse-full-method.md
Implementation:   docs/scse-spec.md

Committed BEFORE the first arm produced a checkpoint, so no threshold here can have been
fitted to the data. Every number a verdict uses is read from the training logs and the
drift-probe JSONs; nothing is re-derived and nothing is a judgement call.

Two lessons from the H21 scorer are enforced structurally:

* **No tie-break fallbacks.** H21's P2 sorted by turnaround step, tied every entry at +inf,
  silently fell back to the seed index, and printed HELD on a panel where nothing had
  happened. Here, a comparison that cannot be made prints NOT SCORABLE and the panel fails.
* **The validity gate runs first and can refuse the whole panel.** Reading arm verdicts off
  a control that has not earned them has happened twice in this campaign.

Usage:
    python lab/divergence/score_scse.py --dir /home/wolfe/morph-scratch/scse2/runs \
        [--probe-dir /home/wolfe/morph-scratch/scse2/probe]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

# ── pre-registered thresholds (do not edit after the first run) ────────────────────────
CE_GAIN_MIN = 0.10      # P2: mean CE improvement, nats. Half the paper's 0.19-0.21.
P3_MIN_PAIRS = 3        # P3: fewest complete pairs that can be scored at all.
DELTA0_MIN = 1e-3       # P1: ||Delta_0||/||h*|| must exceed this on the SCSE arm.
B_ZERO_TOL = 0.0        # P1: the SCSE forcing bias must be EXACTLY zero.
CE_FALL_MIN = 1.0       # P5: nats the val curve must fall, first eval to last.

VAL_RE = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")
ABORT_RE = re.compile(r"\[ABORT\] run aborted at step (\d+)")


def val_curve(path: str) -> list[tuple[int, float]]:
    return [(int(m.group(1)), float(m.group(2)))
            for line in open(path, errors="replace") if (m := VAL_RE.search(line))]


def aborted_at(path: str) -> int | None:
    for line in open(path, errors="replace"):
        if (m := ABORT_RE.search(line)):
            return int(m.group(1))
    return None


def load_arm(d: str, arm: str) -> dict[int, str]:
    """`{seed: log_path}` for `<dir>/<arm>-s<N>.log`."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"{arm}-s*.log"))):
        m = re.search(rf"{arm}-s(\d+)\.log$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def probe_rows(path: str) -> list[dict]:
    return json.load(open(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="holds ctrl-sN.log and scse-sN.log")
    ap.add_argument("--probe-dir", help="holds ctrl-sN.json / scse-sN.json from drift_probe")
    a = ap.parse_args()

    ctrl, scse = load_arm(a.dir, "ctrl"), load_arm(a.dir, "scse")
    print(f"found  ctrl seeds {sorted(ctrl)}   scse seeds {sorted(scse)}\n")

    # ── P1 VALIDITY GATE ───────────────────────────────────────────────────────────────
    print("P1  VALIDITY GATE — the arms must be the two things they claim to be")
    if not a.probe_dir:
        print("  NOT SCORABLE: no --probe-dir. P1 is a GATE, so the panel stops here.")
        print("  Run drift_probe over both arms' checkpoints, then re-score.")
        raise SystemExit(2)
    bad = []
    for arm, seeds in (("scse", scse), ("ctrl", ctrl)):
        for sd in sorted(seeds):
            f = os.path.join(a.probe_dir, f"{arm}-s{sd}.json")
            if not os.path.exists(f):
                bad.append(f"{arm}-s{sd}: no probe file")
                continue
            for r in probe_rows(f):
                fb = r["forcing_bias"]
                bmax = max(abs(x["b_rel"]) for x in fb)
                d0 = fb[0].get("delta0_rel")
                if arm == "scse":
                    if bmax > B_ZERO_TOL:
                        bad.append(f"scse-s{sd}@{r['step']}: b_rel max {bmax:.3e} != 0")
                    if d0 is None or d0 <= DELTA0_MIN:
                        bad.append(f"scse-s{sd}@{r['step']}: delta0_rel {d0} <= {DELTA0_MIN}")
                else:
                    r0 = fb[0]["R_t"]
                    if abs(r0 - 1.0) > 1e-3:
                        bad.append(f"ctrl-s{sd}@{r['step']}: R_0 {r0:.6f} != 1.000")
                    if d0 not in (None, 0.0):
                        bad.append(f"ctrl-s{sd}@{r['step']}: delta0_rel {d0} != 0")
    if bad:
        print("  FAILED:")
        for b in bad[:12]:
            print(f"    {b}")
        print("  Nothing below is readable. The panel is refused.")
        raise SystemExit(1)
    print("  passed at every checkpoint of both arms\n")

    # ── pairs ──────────────────────────────────────────────────────────────────────────
    # Seed 0 is the known-pathological divergence probe and is excluded from P2/P3 by the
    # pre-registration. It is still scored under P4.
    pairs = sorted(set(ctrl) & set(scse) - {0})
    rows = []
    for sd in pairs:
        cc, sc = val_curve(ctrl[sd]), val_curve(scse[sd])
        if not cc or not sc:
            print(f"  seed {sd}: NOT SCORABLE (an arm produced no [VAL] line)")
            continue
        rows.append((sd, cc[-1][1], sc[-1][1], sc, cc))

    print(f"{'seed':>5} {'ctrl CE':>9} {'scse CE':>9} {'delta':>8}  (negative = SCSE better)")
    for sd, c, s, _, _ in rows:
        print(f"{sd:>5} {c:>9.4f} {s:>9.4f} {s - c:>+8.4f}")

    if len(rows) < P3_MIN_PAIRS:
        print(f"\nNOT SCORABLE: {len(rows)} complete pairs, the pre-registration requires "
              f"at least {P3_MIN_PAIRS}. No verdict is issued.")
        raise SystemExit(2)

    # ── P2 ─────────────────────────────────────────────────────────────────────────────
    mc = sum(r[1] for r in rows) / len(rows)
    ms = sum(r[2] for r in rows) / len(rows)
    gain = mc - ms                                  # positive = SCSE better
    p2 = gain >= CE_GAIN_MIN
    print(f"\nP2  mean CE improves by >= {CE_GAIN_MIN} nats: ctrl {mc:.4f} - scse {ms:.4f} "
          f"= {gain:+.4f} -> {'HELD' if p2 else 'FAILED'}")

    # ── P3 ─────────────────────────────────────────────────────────────────────────────
    wins = sum(1 for _, c, s, _, _ in rows if s < c)
    need = len(rows) if len(rows) == 3 else 3
    p3 = wins >= need
    print(f"P3  SCSE lower on >= {need} of {len(rows)} pairs: {wins} "
          f"-> {'HELD' if p3 else 'FAILED'}")

    # ── P4 ─────────────────────────────────────────────────────────────────────────────
    print("P4  no SCSE seed diverges whose control did not:")
    p4 = True
    for sd in sorted(set(ctrl) & set(scse)):
        ca, sa = aborted_at(ctrl[sd]), aborted_at(scse[sd])
        if sa is not None and ca is None:
            p4 = False
            print(f"    seed {sd}: SCSE aborted at {sa}, control did NOT -> NEW instability")
        elif sa is not None or ca is not None:
            print(f"    seed {sd}: ctrl abort={ca}  scse abort={sa}")
    print(f"    -> {'HELD' if p4 else 'FAILED'}")

    # ── P5 ─────────────────────────────────────────────────────────────────────────────
    print(f"P5  SCSE val CE falls >= {CE_FALL_MIN} nats (the loop is not frozen):")
    p5 = True
    for sd, _c, _s, sc, _cc in rows:
        fall = sc[0][1] - min(v for _, v in sc)
        if fall < CE_FALL_MIN:
            p5 = False
        print(f"    seed {sd}: {sc[0][1]:.4f} -> {min(v for _, v in sc):.4f} "
              f"= {fall:.3f} nats")
    print(f"    -> {'HELD' if p5 else 'FAILED'}")

    # ── REFUTER ────────────────────────────────────────────────────────────────────────
    losses = sum(1 for _, c, s, _, _ in rows if s > c)
    refuted = (gain < 0) or (losses >= 3)
    print(f"\nREFUTER  mean CE worse ({gain < 0}) OR worse on >= 3 pairs ({losses}) "
          f"-> {'FIRED — the full method does not transfer to MORPH at this scale' if refuted else 'not fired'}")
    print(f"\nSUMMARY  P2 {'HELD' if p2 else 'FAILED'} | P3 {'HELD' if p3 else 'FAILED'} | "
          f"P4 {'HELD' if p4 else 'FAILED'} | P5 {'HELD' if p5 else 'FAILED'} | "
          f"refuter {'FIRED' if refuted else 'silent'}")


if __name__ == "__main__":
    main()
