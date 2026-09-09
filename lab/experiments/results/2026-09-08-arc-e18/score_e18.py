#!/usr/bin/env python
"""Score the E18 slot-channel width sweep (P18a-g) from the core_depth_sweep JSONs, probes,
run logs and the queue log.

Between-arm CE differences pair at the TOKEN level: every arm scores the same validation
stream, but each cut packs it differently (a row of 1024 + 64·k positions holds a different
number of tokens at each width; the plain cut is 1024 tokens per row), so ROWS are not the
same text across arms — the row means of k2 and k4 correlate at 0.098 across the 480 rows
against 1.000 within an arm. The re-swept JSONs carry a per-token CE array keyed by stream
index (`core_depth_sweep.py` at the fixed commit); the scorer intersects the indices and
bootstraps over 1,024-token stream BLOCKS (tokens in a row are not independent). Without
the token files it falls back to the row-mean comparison and labels it UNPAIRED.
K-differences within an arm are the sweep's own paired bootstraps. The matched-inference-cost metric is block-passes per row (prereg,
Method): plain at core depth T costs 6144 + 6144*T; mask-k at slot depth T' costs
6*(1024 + 64*k) + 384*T'.

Usage: python lab/experiments/results/2026-09-08-arc-e18/score_e18.py [RESULTS_DIR] [QUEUE_LOG]
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ARMS = ["e18-mask-k2", "e18-mask-k4", "e18-mask-k8", "e18-notul"]
K = {"e18-mask-k2": 2, "e18-mask-k4": 4, "e18-mask-k8": 8}
TRAINED = {"e18-mask-k2": "12", "e18-mask-k4": "12", "e18-mask-k8": "12", "e18-notul": "6"}
CKS = [2500, 5000]
DEPTHS = ["1", "2", "3", "6", "9", "12", "16"]
HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT, SEED = 2000, 0


def load(d: str, arm: str, ck: int) -> dict | None:
    p = os.path.join(d, f"sweep_{arm}_{ck}.json")
    return json.load(open(p))[arm] if os.path.exists(p) else None


def ci(s: dict, key: str) -> str:
    c = s["ci_ce_tokens"][key]
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


BLOCK = 1024  # stream tokens per bootstrap unit
_TOK: dict[str, dict] = {}


def tokens(s: dict) -> dict | None:
    """The per-token CE arrays of a sweep (``tok_index`` + ``ce_<depth>``), if it wrote them."""
    p = s.get("tokens_npz")
    if not p:
        return None
    if p not in _TOK:
        cands = [p, os.path.join(os.path.dirname(sys.argv[1]) if len(sys.argv) > 1 else HERE, os.path.basename(p)),
                 os.path.join(HERE, os.path.basename(p))]
        hit = next((c for c in cands if os.path.exists(c)), None)
        _TOK[p] = dict(np.load(hit)) if hit else None
    return _TOK[p]


def paired(a: dict, da: str, b: dict, db: str) -> tuple[float, float, float, str]:
    """Token CE of arm a at depth da minus arm b at depth db.

    Token-level when both sweeps carry per-token arrays: the stream indices are
    intersected, the mean difference is over the shared tokens, and the CI resamples
    1,024-token stream blocks. Otherwise the row-mean fallback (UNPAIRED across cuts)."""
    ta, tb = tokens(a), tokens(b)
    rng = np.random.default_rng(SEED)
    if ta is not None and tb is not None:
        ia, ib = ta["tok_index"], tb["tok_index"]
        common, pa, pb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        d = ta[f"ce_{da}"][pa].astype(np.float64) - tb[f"ce_{db}"][pb].astype(np.float64)
        blk = common // BLOCK
        ub, inv = np.unique(blk, return_inverse=True)
        bsum = np.bincount(inv, weights=d, minlength=len(ub))
        bcnt = np.bincount(inv, minlength=len(ub)).astype(float)
        idx = rng.integers(0, len(ub), size=(N_BOOT, len(ub)))
        boots = bsum[idx].sum(1) / bcnt[idx].sum(1)
        return (float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)),
                f"tokens n={len(common)} blocks={len(ub)}")
    na, nb = np.asarray(a["row_n_tokens"], float), np.asarray(b["row_n_tokens"], float)
    n = min(len(na), len(nb))
    d = np.asarray(a["row_ce_sum"][da][:n]) / na[:n] - np.asarray(b["row_ce_sum"][db][:n]) / nb[:n]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = (d * nb[:n])[idx].sum(1) / nb[:n][idx].sum(1)
    return (float((d * nb[:n]).sum() / nb[:n].sum()), float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)), "UNPAIRED rows")


def fmt(t: tuple) -> str:
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}] ({t[3]})"


def cost(arm: str, depth: int) -> int:
    if arm == "e18-notul":
        return 6144 + 6144 * depth
    return 6 * (1024 + 64 * K[arm]) + 384 * depth


def val_curve(log: str) -> dict[int, float]:
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"\[VAL\s+(\d+)\] loss=([0-9.]+)", open(log).read())}


def peak_gb(log: str) -> float:
    return max(float(x) for x in re.findall(r"peak=([0-9.]+)GB", open(log).read()))


def wall_h(queue: str) -> dict[str, float]:
    t: dict[str, dict[str, int]] = {}
    for m in re.finditer(r"(START|DONE) (e18-[\w-]+) .*?epoch=(\d+)", open(queue).read()):
        t.setdefault(m.group(2), {})[m.group(1)] = int(m.group(3))
    return {a: (v["DONE"] - v["START"]) / 3600 for a, v in t.items() if "START" in v and "DONE" in v}


def verdicts(queue: str) -> dict[str, str]:
    return {m.group(1): m.group(2)
            for m in re.finditer(r"DONE (e18-[\w-]+) exit=\d+ epoch=\d+ verdict=(\S+.*?) Final", open(queue).read())}


def probe_stats(path: str) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    w = [r for r in rows if 1000 <= r["step"] <= 5000]
    g = [r["loss/gain_est"] for r in w if "loss/gain_est" in r]
    out = {"preclip_max_after_200": max((round(r["preclip/total"]), r["step"]) for r in rows if r["step"] >= 200)}
    if g:
        out.update(gain_mean=round(float(np.mean(g)), 4), gain_max=round(float(max(g)), 4),
                   hinge_frac=round(sum(1 for r in w if r.get("loss/gain_reg_weighted", 0) > 0) / len(w), 4))
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    queue = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "queue_e18.log")
    S = {(a, ck): s for a in ARMS for ck in CKS if (s := load(root, a, ck)) is not None}
    print("=== token CE by forced depth (480 rows), the sweep's paired K-differences")
    print(f"{'arm':12s} {'ck':>5s} " + " ".join(f"{'T=' + d:>7s}" for d in DEPTHS) + " | K1-K6 | K3-K6")
    for (a, ck), s in S.items():
        print(f"{a:12s} {ck:5d} " + " ".join(f"{s['depths'][d]['ce_tokens']:7.4f}" for d in DEPTHS)
              + f" | {ci(s, 'K1-K6')} | {ci(s, 'K3-K6')}")

    print("\n=== P18b (width) at 5000: paired token CE at the trained depth 12")
    for a, b in (("e18-mask-k4", "e18-mask-k2"), ("e18-mask-k8", "e18-mask-k4"), ("e18-mask-k8", "e18-mask-k2")):
        if (a, 5000) in S and (b, 5000) in S:
            print(f"  {a} - {b} @12: {fmt(paired(S[a, 5000], '12', S[b, 5000], '12'))}")
    for a in K:
        if (a, 5000) in S and ("e18-notul", 5000) in S:
            print(f"  {a}@12 - notul@6: {fmt(paired(S[a, 5000], '12', S['e18-notul', 5000], '6'))}")

    print("\n=== P18c (depth) at 5000: K3-K6 > 0.005 with CI above 0? K1-K6 > 0.03?")
    for a in K:
        if (a, 5000) in S:
            c3, c1 = S[a, 5000]["ci_ce_tokens"]["K3-K6"], S[a, 5000]["ci_ce_tokens"]["K1-K6"]
            print(f"  {a}: K3-K6 {c3['point']:+.4f} lo {c3['lo']:+.4f} -> {'YES' if c3['point'] > 0.005 and c3['lo'] > 0 else 'no'};"
                  f"  K1-K6 {c1['point']:+.4f} -> {'YES' if c1['point'] > 0.03 else 'no'}")

    print("\n=== P18d (matched inference cost) at 5000: block-passes per row and paired CE")
    print(f"  {'arm@depth':22s} {'passes':>7s} {'CE':>8s}   vs notul@1 (12288)      vs notul@2 (18432)")
    for a in K:
        if (a, 5000) not in S:
            continue
        for d in ("3", "6", "12", "16"):
            line = f"  {a + '@' + d:22s} {cost(a, int(d)):7d} {S[a, 5000]['depths'][d]['ce_tokens']:8.4f}"
            if ("e18-notul", 5000) in S:
                line += f"   {fmt(paired(S[a, 5000], d, S['e18-notul', 5000], '1'))}   {fmt(paired(S[a, 5000], d, S['e18-notul', 5000], '2'))}"
            print(line)
    if ("e18-notul", 5000) in S:
        for d in ("1", "2", "3", "6", "12"):
            print(f"  {'notul@' + d:22s} {cost('e18-notul', int(d)):7d} {S['e18-notul', 5000]['depths'][d]['ce_tokens']:8.4f}")

    print("\n=== P18a/e/f/g: survival, hinge, wall clock, peak, val order")
    wh = wall_h(queue) if os.path.exists(queue) else {}
    vd = verdicts(queue) if os.path.exists(queue) else {}
    finals = {}
    for a in ARMS:
        log = os.path.join(root, f"run_{a}.log")
        probe = os.path.join(root, f"probe_{a}.jsonl")
        if not os.path.exists(probe):
            probe = os.path.join(HERE, "..", "..", "..", "..", "ignored", "experiment-artifacts",
                                 "2026-09-08-arc-e18", f"probe_{a}.jsonl")
        if not os.path.exists(log):
            print(f"  {a}: no run log")
            continue
        v = val_curve(log)
        finals[a] = v[max(v)]
        ps = probe_stats(probe) if os.path.exists(probe) else {}
        print(f"  {a:12s} verdict={vd.get(a, '?')}  wall {wh.get(a, float('nan')):.2f} h  peak {peak_gb(log):.2f} GB  "
              f"val@2500 {v.get(2500, float('nan')):.4f} last@{max(v)} {v[max(v)]:.4f}  {ps}")
    if finals:
        print("  final val order: " + " < ".join(a for a, _ in sorted(finals.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
