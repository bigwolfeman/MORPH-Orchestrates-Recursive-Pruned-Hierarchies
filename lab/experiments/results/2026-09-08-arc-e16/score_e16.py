#!/usr/bin/env python
"""Score the E16 Olympiad curriculum panel (P16a-j) from the sweep JSONs, probes and run logs.

Two sweep sets exist: ``sweeps/`` (the runner's 1000-doc draw from the ORIGINAL eval_holdout
files, 41 % of whose docs sit verbatim in the training bands) and ``sweeps_clean/`` (all
1906 docs that appear in no training band view and are unique within the holdout). The
predictions were written against the original files; both are scored and the clean set is
the one that counts. See the prereg for why.

Usage: python lab/experiments/results/2026-09-08-arc-e16/score_e16.py [RESULTS_DIR]
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ARMS = ["oly-mask", "oly-notul", "oly-mnext", "oly-a1"]
CKS = [1500, 3000, 4500, 6000]
DEPTHS = ["1", "2", "3", "6", "9", "12", "16"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(d: str, arm: str, ck: int) -> dict | None:
    p = os.path.join(d, f"sweep_{arm}_{ck}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))[arm]


def ci(s: dict, metric: str, key: str) -> str:
    c = s[metric][key]
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def band_acc(s: dict, depth: str, lo: int, hi: int) -> float:
    d = s["depths"][depth]
    num = sum(d["band_acc_answer"][b] * d["band_n_answer"][b]
              for b in d["band_n_answer"] if lo <= int(b) <= hi)
    den = sum(d["band_n_answer"][b] for b in d["band_n_answer"] if lo <= int(b) <= hi)
    return num / den if den else float("nan")


def paired_ce(a: dict, b: dict, depth: str) -> tuple[float, float, float]:
    """Token CE of arm a minus arm b at ``depth``, paired over docs, 95 % bootstrap CI.

    The same docs in the same order (seeded interleave) but a plain-model sweep counts a
    few more loss-bearing tokens per doc than a TUL sweep (mean +1.9 of ~72 at 6000; the
    packer's padded tail), so the pairing is on each doc's MEAN CE under its own count,
    weighted by arm b's token count."""
    assert a["doc_stage"] == b["doc_stage"], "sweeps scored different docs"
    na, nb = np.asarray(a["doc_n_tokens"]), np.asarray(b["doc_n_tokens"])
    keep = (na > 0) & (nb > 0)  # a doc the packer left without a loss-bearing token scores nothing
    na, nb = na[keep], nb[keep]
    da = np.asarray(a["doc_ce_sum"][depth])[keep] / na - np.asarray(b["doc_ce_sum"][depth])[keep] / nb
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(nb), size=(2000, len(nb)))
    boots = (da * nb)[idx].sum(1) / nb[idx].sum(1)
    return float((da * nb).sum() / nb.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def val_curve(log: str) -> dict[int, float]:
    out = {}
    for m in re.finditer(r"\[VAL\s+(\d+)\] loss=([0-9.]+)", open(log).read()):
        out[int(m.group(1))] = float(m.group(2))
    return out


def probe_stats(path: str) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    w = [r for r in rows if 1000 <= r["step"] <= 6000]
    g = [r["loss/gain_est"] for r in w if "loss/gain_est" in r]  # notul has no slot loop
    out = {"preclip_max_after_200": max((r["preclip/total"], r["step"]) for r in rows if r["step"] >= 200)}
    if g:
        out.update(gain_mean=float(np.mean(g)), gain_max=float(max(g)),
                   hinge_frac=sum(1 for r in w if r.get("loss/gain_reg_weighted", 0) > 0) / len(w))
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    for tag, sub in (("ORIGINAL eval_holdout (contaminated)", "sweeps"),
                     ("CLEAN holdout (no doc in any training band)", "sweeps_clean")):
        d = os.path.join(root, sub)
        print(f"\n=== {tag}: {d}")
        print(f"{'arm':10s} {'ck':>5s} {'ce@1':>7s} {'ce@3':>7s} {'ce@6':>7s} {'ce@12':>7s} "
              f"{'acc@6':>7s} {'acc@12':>7s} | tok K1-K6 | ans K1-K6 | ans K3-K6 | ans K6-K12 | acc K1-K6")
        S = {}
        for arm in ARMS:
            for ck in CKS:
                s = load(d, arm, ck)
                if s is None:
                    continue
                S[arm, ck] = s
                dp = s["depths"]
                print(f"{arm:10s} {ck:5d} {dp['1']['ce_tokens']:7.4f} {dp['3']['ce_tokens']:7.4f} "
                      f"{dp['6']['ce_tokens']:7.4f} {dp['12']['ce_tokens']:7.4f} "
                      f"{dp['6']['acc_answer']:7.4f} {dp['12']['acc_answer']:7.4f} | "
                      f"{ci(s, 'ci_ce_tokens', 'K1-K6')} | {ci(s, 'ci_ce_answer', 'K1-K6')} | "
                      f"{ci(s, 'ci_ce_answer', 'K3-K6')} | {ci(s, 'ci_ce_answer', 'K6-K12')} | "
                      f"{ci(s, 'ci_acc_answer', 'K1-K6')}")
        print("\nP16f band accuracy at depth 12 (bands 2-3 / 6-7 / 8-10 / 11-13):")
        for (arm, ck), s in S.items():
            print(f"  {arm:10s} {ck:5d}  " + "  ".join(
                f"{lo}-{hi}: {band_acc(s, '12', lo, hi):.3f}" for lo, hi in ((2, 3), (6, 7), (8, 10), (11, 13))))
        if ("oly-mask", 6000) in S and ("oly-mnext", 6000) in S:
            p, lo, hi = paired_ce(S["oly-mask", 6000], S["oly-mnext", 6000], "12")
            print(f"\nP16g mask - mnext token CE at depth 12 @6000, paired over docs: {p:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        for arm in ARMS:
            if (arm, 6000) in S and ("oly-notul", 6000) in S and arm != "oly-notul":
                p, lo, hi = paired_ce(S[arm, 6000], S["oly-notul", 6000], "12")
                print(f"     {arm} - notul token CE at depth 12 @6000, paired: {p:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\n=== training-time instruments")
    for arm in ARMS:
        log = os.path.join(root, f"run_{arm}.log")
        probe = os.path.join(root, f"probe_{arm}.jsonl")  # 15-20 MB each: kept out of git
        if not os.path.exists(probe):
            probe = os.path.join(HERE, "..", "..", "..", "..", "ignored", "experiment-artifacts",
                                 "2026-09-08-arc-e16", f"probe_{arm}.jsonl")
        if not os.path.exists(log):
            continue
        v = val_curve(log)
        jumps = {k: v.get(k + 250, float("nan")) - v.get(k, float("nan")) for k in (1500, 3000, 4500)}
        ps = probe_stats(probe) if os.path.exists(probe) else {}
        print(f"{arm:10s} val@1500 {v.get(1500):.4f} @3000 {v.get(3000):.4f} @4500 {v.get(4500):.4f} "
              f"@5750 {v.get(5750):.4f}  switch jumps {jumps}  {ps}")


if __name__ == "__main__":
    main()
