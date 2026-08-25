"""Score H23 -- does SCSE Stage 1 stop the A1 takeover? -- without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-25-scse-stage1-initial-deviation.md

Written and committed BEFORE any Stage 1 checkpoint was probed. Every threshold is copied
from that file. The control is H21's sweep, which is legitimate because `core_init_scale=0.0`
is bit-identical to the code that produced it (tests/test_scse_core_init.py).

THE VALIDITY GATE IS ARM-DEPENDENT, and getting that wrong would invert the result.
`R_0 = 1` is an identity only when `Delta_0 = 0`, i.e. when the loop enters at the anchor.
That is TRUE of the baseline and FALSE of Stage 1 by construction -- Stage 1 exists to make
`Delta_0 != 0`. So:

  * baseline seeds MUST report `R_0 = 1.000` and `delta0_rel = 0`;
  * Stage 1 seeds MUST report `delta0_rel > 0`, and `R_0 = 1.000` there would mean the
    mechanism did NOT engage.

A gate that demanded `R_0 = 1` of both arms would pass exactly when Stage 1 had failed.

Usage:
    python lab/divergence/score_h23.py --scse <dir> --base <dir> --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

DELTA0_MIN = 1e-3        # P1: the mechanism is live
P2_MIN_SEEDS = 3         # P2: b_t max lower in at least this many of four
CE_TOL = 0.17            # P4: one MEASURED noise floor (H21 amendment 2), not a guess
B_SPREAD_MAX = 1e-2      # as in score_h21: the GLA retention carry makes b_t t-dependent
GATE_TOL = 1e-2


def _rows(path: str) -> list[dict]:
    return json.load(open(path))


def b_by_step(path: str) -> dict[int, float]:
    """`{step: mean_t b_rel}`. Mean because the core map is t-dependent through the GLA
    retention carry -- measured 2026-08-25, see score_h21.b_by_step."""
    out = {}
    for r in _rows(path):
        v = [x["b_rel"] for x in r["forcing_bias"]]
        m = sum(v) / len(v)
        if (max(v) - min(v)) / max(abs(m), 1e-30) > B_SPREAD_MAX:
            raise RuntimeError(f"{path} step {r['step']}: b_rel t-spread over "
                               f"{B_SPREAD_MAX:.0e}; the rung is not readable.")
        out[int(r["step"])] = m
    return out


def delta0_by_step(path: str) -> dict[int, float]:
    return {int(r["step"]): r["forcing_bias"][0]["delta0_rel"] for r in _rows(path)}


def r0_by_step(path: str) -> dict[int, float]:
    return {int(r["step"]): r["forcing_bias"][0]["R_t"] for r in _rows(path)}


def gate_by_step(path: str) -> dict[int, float]:
    return {int(r["step"]): r["gate_rel_err"] for r in _rows(path)}


def val_curve(log_path: str) -> list[tuple[int, float]]:
    pat = re.compile(r"\[VAL\s+(\d+)\]\s+loss=([0-9.]+)")
    out = []
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def diverged(ckpt_dir: str, log_path: str) -> bool:
    """The divergence guard aborted this run.

    Read from the guard's own artefact -- a DIVERGED_*.pt checkpoint -- and from the log,
    rather than inferred from the loss curve. H21 showed a loss-shape rule cannot separate
    divergence from noise at this metric's spread; the guard firing is unambiguous.
    """
    if os.path.isdir(ckpt_dir) and any(f.startswith("DIVERGED") for f in os.listdir(ckpt_dir)):
        return True
    if os.path.exists(log_path):
        for line in open(log_path, errors="replace"):
            if "DIVERG" in line:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scse", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--ckpt-root", default="checkpoints/morph")
    a = ap.parse_args()

    S, B = {}, {}
    for sd in a.seeds:
        for tag, d, pre, store in (("scse", a.scse, "scse1", S), ("base", a.base, "seedsweep", B)):
            p = os.path.join(d, f"drift_s{sd}.json")
            lg = os.path.join(d, f"s{sd}.log")
            if not (os.path.exists(p) and os.path.exists(lg)):
                print(f"!! {tag} seed {sd}: missing probe or log -- EXCLUDED")
                continue
            store[sd] = {
                "b": b_by_step(p), "d0": delta0_by_step(p), "r0": r0_by_step(p),
                "gate": gate_by_step(p), "curve": val_curve(lg),
                "diverged": diverged(os.path.join(a.ckpt_root, f"{pre}-s{sd}"), lg),
            }
    seeds = sorted(set(S) & set(B))
    if not seeds:
        raise SystemExit("no seed has both arms")

    # ── validity gate, arm-dependent ────────────────────────────────────────
    print("VALIDITY GATE")
    bad = []
    for sd in seeds:
        for st, g in S[sd]["gate"].items():
            if g > GATE_TOL:
                bad.append(f"scse s{sd}@{st}: trajectory gate {g:.2e}")
        for st, g in B[sd]["gate"].items():
            if g > GATE_TOL:
                bad.append(f"base s{sd}@{st}: trajectory gate {g:.2e}")
        # baseline: R_0 = 1 is an identity, and Delta_0 must be exactly 0
        for st, v in B[sd]["r0"].items():
            if abs(v - 1.0) > 1e-3:
                bad.append(f"base s{sd}@{st}: R_0={v:.6f} (identity broken)")
        for st, v in B[sd]["d0"].items():
            if v > 1e-9:
                bad.append(f"base s{sd}@{st}: delta0_rel={v:.2e} (baseline must be 0)")
    if bad:
        print("  FAILED: " + "; ".join(bad[:8]) + ("" if len(bad) <= 8 else f" (+{len(bad)-8})"))
        print("  Nothing below is readable.")
        raise SystemExit(1)
    print("  passed: trajectory gates clean; baseline holds R_0 = 1 and Delta_0 = 0\n")

    # ── P1: the mechanism is live ───────────────────────────────────────────
    print(f"P1  Stage 1 |Delta_0|/|h*| >= {DELTA0_MIN:.0e} at EVERY checkpoint")
    worst = {sd: min(S[sd]["d0"].values()) for sd in seeds}
    for sd in seeds:
        print(f"    s{sd}: min over rungs = {worst[sd]:.4f}"
              f"   (R_0 = {list(S[sd]['r0'].values())[0]:.3f}, must NOT be 1.000)")
    p1 = all(v >= DELTA0_MIN for v in worst.values())
    print(f"    -> {'HELD' if p1 else 'FAILED'}")
    if not p1:
        print("    The run is VOID per the pre-registration: a mechanism that did not "
              "engage tests nothing.")
        raise SystemExit(1)

    # ── P2: the forcing bias ────────────────────────────────────────────────
    print(f"\nP2  max b_t over rungs is LOWER than the control seed's, in >= {P2_MIN_SEEDS} of "
          f"{len(seeds)}")
    print(f"    {'seed':>5} {'base max b':>12} {'scse max b':>12} {'ratio':>8}  lower?")
    n_lower = 0
    for sd in seeds:
        bb, sb = max(B[sd]["b"].values()), max(S[sd]["b"].values())
        lo = sb < bb
        n_lower += lo
        print(f"    {sd:>5} {bb:>12.3f} {sb:>12.3f} {sb/bb:>8.3f}  {'YES' if lo else 'no'}")
    p2 = n_lower >= P2_MIN_SEEDS
    print(f"    {n_lower} of {len(seeds)} lower -> {'HELD' if p2 else 'FAILED'}")

    # ── P3: divergence count (declared low power) ───────────────────────────
    dv_s = [sd for sd in seeds if S[sd]["diverged"]]
    dv_b = [sd for sd in seeds if B[sd]["diverged"]]
    p3 = not dv_s
    print(f"\nP3  no Stage 1 seed trips the divergence guard  [LOW POWER, declared in advance]")
    print(f"    control diverged: {dv_b or 'none'}   Stage 1 diverged: {dv_s or 'none'}")
    print(f"    -> {'HELD' if p3 else 'FAILED'}")
    print(f"    Under the null with the control's {len(dv_b)}-of-{len(seeds)} base rate, a clean "
          f"sweep has probability {((len(seeds)-len(dv_b))/len(seeds))**len(seeds):.2f}. "
          f"This endpoint CANNOT decide H23 alone and is not read as if it could.")

    # ── P4: CE ──────────────────────────────────────────────────────────────
    healthy = [sd for sd in seeds if not S[sd]["diverged"] and not B[sd]["diverged"]]
    print(f"\nP4  mean final val CE not worse than control by more than {CE_TOL} nats")
    print(f"    (threshold = the MEASURED within-run noise floor, H21 amendment 2)")
    if not healthy:
        print("    NOT SCORABLE: no seed is healthy in both arms.")
    else:
        mb = sum(B[sd]["curve"][-1][1] for sd in healthy) / len(healthy)
        ms = sum(S[sd]["curve"][-1][1] for sd in healthy) / len(healthy)
        for sd in healthy:
            print(f"    s{sd}: base {B[sd]['curve'][-1][1]:.4f} -> scse "
                  f"{S[sd]['curve'][-1][1]:.4f}  ({S[sd]['curve'][-1][1]-B[sd]['curve'][-1][1]:+.4f})")
        p4 = (ms - mb) <= CE_TOL
        print(f"    mean over seeds healthy in BOTH arms {healthy}: "
              f"{mb:.4f} -> {ms:.4f}  ({ms-mb:+.4f}) -> {'HELD' if p4 else 'FAILED'}")

    # ── refuter ─────────────────────────────────────────────────────────────
    print("\nREFUTER  b_t maxima >= control in 2+ seeds AND a seed still diverges")
    n_not_lower = len(seeds) - n_lower
    fired = n_not_lower >= 2 and bool(dv_s)
    print(f"  not-lower seeds = {n_not_lower}, Stage 1 divergences = {len(dv_s)} -> "
          f"{'REFUTER FIRED' if fired else 'not fired'}")
    if fired:
        print("  H23 REFUTED: a non-zero initial deviation does not touch the mechanism. "
              "Go to Stage 2 (deviation coordinates, source injection moved into the "
              "anchor) or abandon the port.")

    # ── b_t tables ──────────────────────────────────────────────────────────
    print("\nb_t/|h*| by rung  (base -> scse)")
    rungs = sorted(set().union(*[set(S[sd]["b"]) | set(B[sd]["b"]) for sd in seeds]))
    print(f"{'step':>6}" + "".join(f"{'s%d' % sd:>20}" for sd in seeds))
    for st in rungs:
        cells = []
        for sd in seeds:
            b, s_ = B[sd]["b"].get(st), S[sd]["b"].get(st)
            cells.append(f"{'-' if b is None else f'{b:.2f}':>8} ->"
                         f"{'-' if s_ is None else f'{s_:.2f}':>9}")
        print(f"{st:>6}" + "".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
