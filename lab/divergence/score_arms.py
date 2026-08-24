"""Score divergence arms from their grad-probe mirrors and their console logs.

ONE verdict rule, fixed in docs/experiments/results/2026-08-24-tul-takeover-rca.md and
reused unchanged here so arms scored on different days are comparable:

    TAKEN OVER = core share > 0.5 on more than 30 % of the last 50 probed steps.

Also reported, because the share is the SYMPTOM and the block backward gain is the
mechanism: the median block gain and its r2 over the same window, the first step at which
each criterion fires, and — read off the console log — the loss minimum and the loss at the
end, because the harm this whole programme is about is a loss TURNAROUND, not a spike.

Usage:
    python lab/divergence/score_arms.py name=/path/a.jsonl other=/path/b.jsonl
    python lab/divergence/score_arms.py --window 4000 name=/path/a.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st

LOSS_RE = re.compile(r"^\[\s*(\d+)/\s*\d+\]\s+loss=([0-9.]+)")
VAL_RE = re.compile(r"val[/ ]ce[_a-z]*[=: ]+([0-9.]+)", re.I)


def load_probe(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # a killed run can leave a torn last line
            if r.get("preclip/total"):
                out[int(r["step"])] = r
    return out


def load_loss(path: str) -> list[tuple[int, float]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = LOSS_RE.match(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def fires(rows: list[tuple[int, float]], thr: float, window: int, frac: float,
          gate: list[bool] | None = None) -> int | None:
    """First step at which more than `frac` of the trailing `window` probed steps exceed
    `thr`. `gate` masks steps that do not qualify (the r2 floor for the block gain)."""
    buf: list[bool] = []
    for i, (step, v) in enumerate(rows):
        ok = v is not None and v > thr and (gate is None or gate[i])
        buf.append(ok)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window and sum(buf) / window > frac:
            return step
    return None


def summarise(probe: dict[int, dict], loss: list[tuple[int, float]], upto: int | None):
    ks = sorted(k for k in probe if upto is None or k <= upto)
    if len(ks) < 10:
        return None
    share = [(k, probe[k]["preclip/core"] / probe[k]["preclip/total"]) for k in ks]
    gain = [(k, probe[k].get("preclip/core_block_gain")) for k in ks]
    r2 = [probe[k].get("preclip/core_block_gain_r2") or 0.0 for k in ks]
    tail = [v for _, v in share[-50:]]
    gtail = [v for _, v in gain[-50:] if v]
    rtail = [probe[k].get("preclip/core_block_gain_r2") for k in ks[-50:]
             if probe[k].get("preclip/core_block_gain_r2") is not None]
    ls = [(s, v) for s, v in loss if upto is None or s <= upto]
    lo = min(ls, key=lambda t: t[1]) if ls else (None, float("nan"))
    return {
        "n": len(ks), "last": ks[-1],
        "end_share": st.median(tail),
        "took": sum(v > 0.5 for v in tail) / len(tail) > 0.3,
        "gain": st.median(gtail) if gtail else float("nan"),
        "r2": st.median(rtail) if rtail else float("nan"),
        "share_fires": fires(share, 0.5, 50, 0.3),
        "gain_fires": fires(gain, 1.0, 200, 0.3, gate=[x >= 0.5 for x in r2]),
        "loss_min": lo[1], "loss_min_at": lo[0],
        "loss_end": ls[-1][1] if ls else float("nan"),
        "loss_rise": (ls[-1][1] - lo[1]) if ls else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=None,
                    help="score only steps <= this, so arms of different length compare")
    ap.add_argument("arms", nargs="+", help="name=/path/to/probe.jsonl (log is inferred)")
    a = ap.parse_args()

    hdr = (f"{'arm':<20}{'probed':>7}{'last':>7}{'endshare':>9}{'gain':>7}{'r2':>6}"
           f"{'shareAt':>9}{'gainAt':>8}{'lossMin':>9}{'@':>7}{'lossEnd':>9}{'rise':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for spec in a.arms:
        name, _, path = spec.partition("=")
        s = summarise(load_probe(path), load_loss(path.replace(".jsonl", ".log")), a.window)
        if s is None:
            print(f"{name:<20}{'(too few probed steps)':>40}")
            continue
        print(f"{name:<20}{s['n']:>7}{s['last']:>7}{s['end_share']:>9.4f}{s['gain']:>7.3f}"
              f"{s['r2']:>6.2f}{str(s['share_fires']):>9}{str(s['gain_fires']):>8}"
              f"{s['loss_min']:>9.4f}{str(s['loss_min_at']):>7}{s['loss_end']:>9.4f}"
              f"{s['loss_rise']:>7.3f}  {'TOOK OVER' if s['took'] else 'held'}")


if __name__ == "__main__":
    main()
