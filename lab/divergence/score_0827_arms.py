"""Score the 2026-08-27 arms against their frozen predictions.

Pre-registration: lab/experiments/planned/2026-08-27-warmup-sigreg-ntpdrop.md
(W1, W2, S1, S2, N1, C1-C4, P). Written to be runnable before any arm finishes,
so the thresholds cannot be fitted to the data.

The rule this file exists to enforce: **a letter whose input is missing is
printed NOT MEASURED, never as a pass and never as a fail.** The campaign has
twice reported a verdict off a panel that had not earned it.

Two worths, and they are NOT the same number:

  val/plan_nats   logged every eval by `tul_forward_with_plan_nats`, which runs
                  the coda with the slots GATHERED OUT. Free, present on every
                  arm and on the control, so it is the right ARM-VS-CONTROL
                  comparison.
  plan-off worth  `slot_path_worth.py`, which zeroes what `prefix_project`
                  writes while leaving the layout intact, and reports the cost
                  on ce_main. This is the 0.0191 nats the pre-registration
                  names, and **P is scored on this one only.**

Reading P off val/plan_nats would compare two different ablations against one
threshold. This file refuses to do it.

Usage:
    python lab/divergence/score_0827_arms.py \
        ctrl-s1=/home/wolfe/morph-scratch/seedsweep/s1.log \
        warmup-s0=/home/wolfe/morph-scratch/queue2/tul-warmup-s0/run.log \
        [--wandb]        also query val/mux_local for W1
        [--worth DIR]    directory of slot_path_worth.py --out JSONs, named <arm>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st

# ── Baselines. Every one is a measurement with a named source, not a guess. ──
# Control = tul_a1 seedsweep (lab/divergence/seedsweep.sh), seeds 1/2/3. Seed 0
# diverged (ppl_tok 1779 at step 2000) and is excluded from every median below,
# which is what the pre-registration's "median 91.77" already did.
CTRL_PPL_TOK_3250 = {1: 105.54, 2: 91.77, 3: 90.28}
CTRL_PLAN_NATS_3000 = {1: 0.0049, 2: 0.0037, 3: 0.0027}
PLAN_OFF_BASELINE = 0.0191        # slot_path_worth.py on ce_main; P's threshold
MUX_HEAD_BEST = 7.023             # v1a's best val/mux_local; W1 must beat it
SHARE_THRESHOLD = 0.5             # "crosses 0.5" — the pre-registration's wording
SHIPPED_WINDOW, SHIPPED_FRAC = 50, 0.3   # score_arms.py's stricter rule, reported too

VAL_RE = re.compile(
    r"\[VAL\s+(?P<step>\d+)\]\s+loss=(?P<loss>[0-9.]+)\s+ppl=(?P<ppl>[0-9.]+)"
    r"\s+ppl_tok=(?P<ppl_tok>[0-9.]+)\s+first_tok=(?P<first_tok>[0-9.]+)"
    r"\s+plan_nats=(?P<plan_nats>[-+0-9.]+)")


def val_rows(path: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                d = m.groupdict()
                out[int(d.pop("step"))] = {k: float(v) for k, v in d.items()}
    return out


def share_series(path: str) -> list[tuple[int, float]]:
    """`[(step, preclip core share)]` from the probe mirror. Empty if absent."""
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a killed run leaves a torn last line
            tot = r.get("preclip/total") or 0.0
            if tot > 0.0:
                out.append((int(r["step"]), r.get("preclip/core", 0.0) / tot))
    return out


def cadence(series: list[tuple[int, float]]) -> int:
    if len(series) < 3:
        return 0
    return max(1, min(series[i + 1][0] - series[i][0] for i in range(len(series) - 1)))


def first_cross(series: list[tuple[int, float]], thr: float) -> int | None:
    for step, v in series:
        if v > thr:
            return step
    return None


def shipped_rule(series: list[tuple[int, float]]) -> int | None | str:
    """score_arms.py's rule, reused unchanged. Returns a step, None, or a refusal."""
    c = cadence(series)
    if c == 0:
        return "no series"
    n = SHIPPED_WINDOW // c
    if n < 20:
        return f"REFUSED (cadence {c}: {n} samples in a {SHIPPED_WINDOW}-step window, <20)"
    buf: list[bool] = []
    for step, v in series:
        buf.append(v > SHARE_THRESHOLD)
        if len(buf) > n:
            buf.pop(0)
        if len(buf) == n and sum(buf) / n > SHIPPED_FRAC:
            return step
    return None


def mux_local(names: list[str]) -> dict[str, list[tuple[int, float]]]:
    import wandb
    api = wandb.Api()
    runs = {r.name: r for r in api.runs("adew-me/morph-tul")}
    out = {}
    for nm in names:
        r = runs.get(nm)
        if r is None:
            continue
        h = r.history(keys=["val/mux_local"], samples=2000).dropna()
        out[nm] = [(int(a), float(b)) for a, b in zip(h["_step"], h["val/mux_local"])]
    return out


def verdict(ok: bool | None, note: str = "") -> str:
    if ok is None:
        return "NOT MEASURED" + (f" ({note})" if note else "")
    return ("HELD" if ok else "FAILED") + (f" ({note})" if note else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="label=/path/run.log")
    ap.add_argument("--wandb", action="store_true", help="query val/mux_local for W1")
    ap.add_argument("--worth", default="", help="dir of slot_path_worth.py --out JSONs")
    a = ap.parse_args()

    arms: dict[str, dict] = {}
    for spec in a.arms:
        label, path = spec.split("=", 1)
        probe = os.path.join(os.path.dirname(path), "probe.jsonl")
        arms[label] = {"log": path, "val": val_rows(path), "share": share_series(probe),
                       "probe": probe}

    # ── 1. The curve table ───────────────────────────────────────────────────
    print(f"{'arm':<18} {'evals':>5} {'last':>6} {'ppl_tok@3000':>12} {'ppl_tok@3250':>12} "
          f"{'plan@3000':>10} {'plan@3250':>10} {'first_tok@3250':>14}")
    print("-" * 96)
    for label, d in arms.items():
        v = d["val"]
        if not v:
            print(f"{label:<18} {'0':>5}  no VAL lines in {d['log']}")
            continue
        last = max(v)
        def g(step, key):
            return f"{v[step][key]:>12.4f}" if step in v else f"{'--':>12}"
        def gp(step, key):
            return f"{v[step][key]:>10.4f}" if step in v else f"{'--':>10}"
        print(f"{label:<18} {len(v):>5} {last:>6} {g(3000,'ppl_tok')} {g(3250,'ppl_tok')} "
              f"{gp(3000,'plan_nats')} {gp(3250,'plan_nats')} "
              f"{v[3250]['first_tok'] if 3250 in v else float('nan'):>14.4f}")

    # ── 2. The takeover ──────────────────────────────────────────────────────
    print(f"\n{'arm':<18} {'probed':>7} {'cadence':>8} {'max share':>10} "
          f"{'first >0.5':>11}  shipped 30%-of-50 rule")
    print("-" * 96)
    for label, d in arms.items():
        s = d["share"]
        if not s:
            print(f"{label:<18} {'0':>7} {'--':>8} {'--':>10} {'--':>11}  "
                  f"NOT MEASURED (no {os.path.basename(d['probe'])})")
            continue
        print(f"{label:<18} {len(s):>7} {cadence(s):>8} {max(v for _, v in s):>10.4f} "
              f"{str(first_cross(s, SHARE_THRESHOLD)):>11}  {shipped_rule(s)}")

    # ── 3. plan_nats against the control, at a MATCHED definition ────────────
    ctrl_plan = st.median(CTRL_PLAN_NATS_3000.values())
    ctrl_ppl = st.median(CTRL_PPL_TOK_3250.values())
    print(f"\ncontrol medians (tul_a1 seedsweep s1/s2/s3): "
          f"ppl_tok@3250 {ctrl_ppl:.2f}, val/plan_nats@3000 {ctrl_plan:.4f} "
          f"(seed range {min(CTRL_PLAN_NATS_3000.values()):.4f}"
          f"-{max(CTRL_PLAN_NATS_3000.values()):.4f})")
    print("val/plan_nats gathers the slots OUT. It is NOT the 0.0191-nats plan-off worth,")
    print("which zeroes prefix_project's values instead. P is scored on the latter only.")

    # ── 4. W1: did the head learn? ───────────────────────────────────────────
    print(f"\nW1  val/mux_local at 3000 < {MUX_HEAD_BEST} (v1a's best)")
    if a.wandb:
        series = mux_local([f"tul-{lbl}" if not lbl.startswith("tul-") else lbl
                            for lbl in arms])
        if not series:
            print("    " + verdict(None, "no matching wandb run carried val/mux_local"))
        for nm, rows in series.items():
            at3000 = [v for s, v in rows if 2900 <= s <= 3100]
            if not at3000:
                print(f"    {nm:<22} " + verdict(None, "no val/mux_local near step 3000"))
            else:
                v = min(at3000)
                print(f"    {nm:<22} {v:.4f}  " + verdict(v < MUX_HEAD_BEST))
    else:
        print("    " + verdict(None, "pass --wandb"))

    # ── 5. P / C3: the criterion that decides ────────────────────────────────
    print(f"\nP/C3  plan-off worth at step 3000 > {PLAN_OFF_BASELINE} nats "
          f"(slot_path_worth.py, ce_main)")
    for label in arms:
        f = os.path.join(a.worth, f"{label}.json") if a.worth else ""
        if not f or not os.path.exists(f):
            print(f"    {label:<22} " + verdict(None, "no slot_path_worth JSON"))
            continue
        r = json.load(open(f))
        full = r["full"]["ce_main"]
        noplan = next(v for k, v in r.items() if k.startswith("no-plan"))["ce_main"]
        w = noplan - full
        print(f"    {label:<22} {w:+.4f}  " + verdict(w > PLAN_OFF_BASELINE))

    print("\nAn arm that passes its own letter and fails P is a PARTIAL result, never a")
    print("success — the pre-registration fixes that and this file does not soften it.")


if __name__ == "__main__":
    main()
