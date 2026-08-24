"""Join the offline optimizer-state probe to the run's own pre-clip core share.

`optstate_probe.py` measures the AdEMAMix state at a checkpoint. The trainer's
`grad_probe_path` jsonl records `preclip/core` and `preclip/total` at the same steps. This
puts them in one table and asks the only question that matters for a severity measure:
does it rank the runs the way their harm does?

Reported next to the per-block backward gain, which
`docs/experiments/failures/2026-08-24-tul-takeover-cure.md` shows is the WRONG severity
measure — `bptt_depth 2` scores 2.784 against its control's 2.445 with 64 % less harm. A
replacement has to beat that, not merely differ from it.

Usage:
    python lab/divergence/score_optstate.py \
        --pair onset-ladder=optstate_ladder.json:capture.jsonl \
        --pair b10-ctrl=optstate_b10-ctrl.json:b10-ctrl.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def load_probe(path: str) -> dict[int, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[int(r["step"])] = r
    return rows


def share_at(probe: dict[int, dict], step: int, tol: int = 30):
    """Core share at the probed step nearest `step`, or None if none is within `tol`.

    A tolerance and not an interpolation: the probe cadence is 25 steps here and the share
    moves from 0.02 to 0.37 in 50 steps at onset, so interpolating would invent a number
    in exactly the window the reader cares about.
    """
    cand = [s for s in probe if abs(s - step) <= tol]
    if not cand:
        return None, None
    s = min(cand, key=lambda x: abs(x - step))
    r = probe[s]
    tot = r.get("preclip/total")
    if not tot:
        return s, None
    return s, r["preclip/core"] / tot


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, average ranks on ties. n is small enough that O(n^2) is free."""
    def ranks(v):
        out = [0.0] * len(v)
        order = sorted(range(len(v)), key=lambda i: v[i])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


# Every arm with a measured harm, transcribed from the ONE table in
# docs/experiments/failures/2026-08-24-tul-takeover-cure.md. `harm` is the validation-CE
# rise above the arm's own minimum, lower is better. `gain` is the per-block backward
# gain, the measure being replaced. Nothing else in this file is typed by hand.
ARM_HARM = {
    "a35-ctrl": 1.186, "b10-ctrl": 0.533, "b10-bptt2": 0.192, "s0-stack": 0.148,
    "s0-slotembed": 0.119, "cure-a1r-ctrl": 0.015, "b10-slotembed": 0.000,
}
ARM_GAIN = {
    "a35-ctrl": 2.772, "b10-ctrl": 2.445, "b10-bptt2": 2.784, "s0-stack": 2.838,
    "s0-slotembed": 2.501, "cure-a1r-ctrl": 1.045, "b10-slotembed": 1.052,
}
# The three spectral arms carry a CORE-LOCAL penalty, so their gradients — and therefore
# m2 and nu — contain the regulariser as well as the model. They are scored separately
# for exactly the reason `_preclip_probe` warns about, not dropped.
PENALISED_HARM = {"a35-spec": 2.737, "a35-proj15": 3.496, "a35-proj15attn": 1.108}
PENALISED_GAIN = {"a35-spec": 2.038, "a35-proj15": 1.659, "a35-proj15attn": 1.828}

# Every candidate severity measure, as (label, extractor). The extractor sees the row at
# the comparison step plus the last row's drift.
CANDIDATES = [
    ("per-block gain (incumbent)", None),
    ("core slow/fast", lambda r, d: r["regions"]["core"]["slow_over_fast"]),
    ("core coherence", lambda r, d: r["regions"]["core"]["coh"]),
    ("core/noncore coherence", lambda r, d: r["regions"]["core"]["coh"]
     / (r["regions"]["noncore"]["coh"] or 1e-12)),
    ("core share of dW", lambda r, d: d),
]


def _arm_row(rows: list[dict], step: int):
    """The row at `step`, and the last row's core dW share."""
    at = [r for r in rows if abs(r["step"] - step) <= 5]
    last_share = rows[-1].get("drift", {}).get("core", {}).get("share", float("nan"))
    return (at[0] if at else None), last_share


def rank_report(vals: dict[str, dict], harm_of: dict[str, float],
                gain_of: dict[str, float], title: str) -> None:
    """Which candidate severity measure orders these arms the way their harm does."""
    arms = [a for a in harm_of if a in vals and vals[a]["row"] is not None]
    if len(arms) < 3:
        print(f"\n{title}: only {len(arms)} arms measured, not ranked")
        return
    harm = [harm_of[a] for a in arms]
    print(f"\n{title}  (n={len(arms)}, comparison step 2000)")
    head = f"{'arm':<16}{'harm':>7}{'gain':>7}"
    cols = []
    for label, fn in CANDIDATES:
        if fn is None:
            cols.append((label, [gain_of[a] for a in arms]))
        else:
            cols.append((label, [fn(vals[a]["row"], vals[a]["dw_share"]) for a in arms]))
        if fn is not None:
            head += f"{label.split()[0][:9]:>10}"
    print(head)
    for i, a in enumerate(arms):
        line = f"{a:<16}{harm_of[a]:>7.3f}{gain_of[a]:>7.3f}"
        for label, series in cols:
            if label != "per-block gain (incumbent)":
                line += f"{series[i]:>10.4f}"
        print(line)
    print()
    for label, series in cols:
        print(f"  spearman(harm, {label:<28}) = {spearman(harm, series):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[],
                    help="label=optstate.json:probe.jsonl")
    ap.add_argument("--tol", type=int, default=30)
    ap.add_argument("--rank", action="store_true",
                    help="rank the cure arms by each candidate severity measure")
    a = ap.parse_args()

    print(f"{'run':<16}{'step':>7}{'share':>8}{'slow/fast':>10}{'coh':>8}"
          f"{'coh_nc':>8}{'dW share':>10}")
    print("-" * 67)
    x_measure, y_share = [], []
    arm_vals: dict[str, dict] = {}
    for spec in a.pair:
        label, rest = spec.split("=", 1)
        oj, pj = rest.split(":", 1)
        rows = json.load(open(oj))["rows"]
        if label in ARM_HARM or label in PENALISED_HARM:
            row, share = _arm_row(rows, 2000)
            arm_vals[label] = {"row": row, "dw_share": share}
        probe = load_probe(pj) if os.path.exists(pj) else {}
        for r in rows:
            st = r["step"]
            _s, sh = share_at(probe, st, a.tol) if probe else (None, None)
            core, nc = r["regions"]["core"], r["regions"]["noncore"]
            d = r.get("drift", {}).get("core", {})
            print(f"{label:<16}{st:>7}"
                  f"{('%.4f' % sh) if sh is not None else '     —':>8}"
                  f"{core['slow_over_fast']:>10.4f}{core['coh']:>8.4f}{nc['coh']:>8.4f}"
                  + (f"{d['share']:>10.4f}" if "share" in d else f"{'—':>10}"))
            if sh is not None:
                x_measure.append(core["coh"] / (nc["coh"] or 1e-12))
                y_share.append(sh)
    if len(x_measure) >= 3:
        print(f"\nspearman(core/noncore coherence, pre-clip core share) = {spearman(x_measure, y_share):+.3f}"
              f"   over n={len(x_measure)} checkpoints")
    if a.rank and arm_vals:
        rank_report(arm_vals, ARM_HARM, ARM_GAIN, "CLEAN ARMS")
        rank_report(arm_vals, PENALISED_HARM, PENALISED_GAIN,
                    "PENALISED ARMS (core-local regulariser in the gradient)")


if __name__ == "__main__":
    main()
