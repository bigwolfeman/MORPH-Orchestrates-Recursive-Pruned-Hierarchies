"""Score the slot-map levers panel and the reread arm against the unpack arm.

Reads the arc runner's result dirs (the copies under lab/experiments/results/ once filed):
sweeps (token + forecast K-curves with CIs), worth profiles, slot-state probes, slot
anatomies, the queue log (survival, rate) and the trainer logs (val curve, peak).
Rerun from the repo root:
  PYTHONPATH=. python lab/experiments/results/2026-09-10-slot-map-levers/score_levers.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, "lab/divergence")
from sweep_score import load_sweep, val_curve, peak_gb  # noqa: E402

Q = os.environ.get("ARC_Q", "/home/wolfe/morph-scratch/arc")
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(Q, "results", "2026-09-10-slot-map-levers"),
        os.path.join(Q, "results", "2026-09-09-slot-loop-norm-match"),
        HERE, os.path.join(HERE, "..", "2026-09-09-slot-loop-norm-match")]
ARMS = ["slot-unpack-norm-match", "slot-unpack-free", "slot-unpack-noise-entry",
        "slot-unpack-fixed-depth", "slot-unpack-reread"]
REF = "slot-unpack-norm-match"


def find(name: str) -> str | None:
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def jload(name: str):
    p = find(name)
    if p is None:
        return None
    d = json.load(open(p))
    return next(iter(d.values())) if isinstance(d, dict) and len(d) == 1 and isinstance(next(iter(d.values())), dict) and "step" in next(iter(d.values())) else d


def cis(s: dict, block: str, key: str) -> str:
    c = s.get(block, {}).get(key)
    if c is None:
        return "n/a"
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def above(s: dict, block: str, key: str, thr: float) -> bool | None:
    c = s.get(block, {}).get(key)
    if c is None:
        return None
    return c["point"] > thr and c["lo"] > 0


def queue_lines(arm: str) -> list[str]:
    out = []
    for name in ("queue.log",):
        p = os.path.join(Q, name)
        if os.path.exists(p):
            out += [l.rstrip() for l in open(p) if f" {arm}" in l or f" {arm}:" in l or f" {arm}@" in l]
    return out


def main() -> None:
    print("slot-map levers panel + reread, scored against", REF)
    print("prereg: lab/experiments/planned/2026-09-10-arc-slot-map-levers.md, 2026-09-10-arc-slot-reread.md\n")
    rows = {}
    for arm in ARMS:
        r = {"arm": arm}
        ql = queue_lines(arm)
        done = [l for l in ql if " DONE " in l and "smoke" not in l]
        r["verdict"] = re.search(r"verdict=(\S+)", done[-1]).group(1) if done else "not run"
        rate = [l for l in ql if "RATE " in l]
        r["rate"] = re.search(r"tok/s=(\d+)", rate[-1]).group(1) if rate else "n/a"
        log = find(f"run_{arm}.log") or os.path.join(Q, arm, "run.log")
        if os.path.exists(log):
            vc = val_curve(log)
            r["val5000"] = vc.get(5000, vc.get(max(vc), float("nan")) if vc else float("nan"))
            r["peak"] = peak_gb(log)
        for ck in (2500, 5000):
            s = jload(f"sweep_{arm}_{ck}.json")
            if s:
                r[f"sweep{ck}"] = s
        w = jload(f"worth_{arm}_5000.json")
        if w:
            r["worth0"] = w["modes"]["zero"]["mean"][0]
            r["shuffle0"] = w["modes"]["shuffle"]["mean"][0]
        pr = jload(f"slot_state_{arm}_5000.json")
        if pr:
            pd = pr["per_depth"]
            r["probe_move"] = [pd[k].get("rel_dist_from_prev_depth_mean") for k in sorted(pd, key=int)][1:]
        an = jload(f"slot_anatomy_{arm}_5000.json")
        if an:
            r["anat_first"] = an["first_pass_from_h0"]
            r["anat_move"] = an["consecutive"]
            r["anat_mlp"] = an["branch_out_in"]["mlp"][str(max(int(k) for k in an["branch_out_in"]["mlp"]))]
            r["anat_attn"] = an["branch_out_in"]["attn"][str(max(int(k) for k in an["branch_out_in"]["attn"]))]
        rows[arm] = r

    print("survival / rate / val@5000 / peak")
    for arm, r in rows.items():
        print(f"  {arm:26s} {r['verdict']:10s} tok/s {r['rate']:>6s}  val {r.get('val5000', float('nan')):.4f}  peak {r.get('peak', float('nan')):.1f} GB")
    print("\nK-curves (tokens | forecast), 2500 then 5000")
    for arm, r in rows.items():
        for ck in (2500, 5000):
            s = r.get(f"sweep{ck}")
            if not s:
                continue
            print(f"  {arm:26s}@{ck}: tok K1-K6 {cis(s, 'ci_ce_tokens', 'K1-K6')}  K3-K6 {cis(s, 'ci_ce_tokens', 'K3-K6')}"
                  f" | fc K1-K6 {cis(s, 'ci_mux_local', 'K1-K6')}  K3-K6 {cis(s, 'ci_mux_local', 'K3-K6')}")
            d6 = s["depths"].get("6", {}).get("ce_tokens")
            print(f"  {'':26s}       CE@6 {d6:.4f}" if d6 else "")
    print("\nworth (zero / shuffle at offset 0), probe movement per forced depth, anatomy movement per pass")
    for arm, r in rows.items():
        pm = r.get("probe_move")
        am = r.get("anat_move")
        print(f"  {arm:26s} zero {r.get('worth0', float('nan')):+.3f} shuffle {r.get('shuffle0', float('nan')):+.3f}"
              f" | probe {[round(x, 3) for x in pm] if pm else 'n/a'}"
              f" | anatomy first {r.get('anat_first', float('nan')):.3f} then {[round(x, 3) for x in am] if am else 'n/a'}")
        if r.get("anat_mlp"):
            print(f"  {'':26s} branch out/in last pass: mlp {[round(x, 3) for x in r['anat_mlp']]} attn {[round(x, 3) for x in r['anat_attn']]}")
    print("\nprediction checks (P-lev-b/c/d/e, P-rr-c): tokens K1-K6 > 0.005, K3-K6 > 0.002, forecast K1-K6 > 0.010, CI low > 0")
    any_k3 = False
    for arm, r in rows.items():
        s = r.get("sweep5000")
        if not s or arm == REF:
            continue
        k1 = above(s, "ci_ce_tokens", "K1-K6", 0.005)
        k3 = above(s, "ci_ce_tokens", "K3-K6", 0.002)
        f1 = above(s, "ci_mux_local", "K1-K6", 0.010)
        any_k3 = any_k3 or bool(k3)
        ref6 = rows[REF].get("sweep5000", {}).get("depths", {}).get("6", {}).get("ce_tokens")
        d6 = s["depths"].get("6", {}).get("ce_tokens")
        gap = (d6 - ref6) if (d6 and ref6) else float("nan")
        print(f"  {arm:26s} tok K1-K6>0.005: {k1}  K3-K6>0.002: {k3}  fc K1-K6>0.010: {f1}  CE@6 minus ref {gap:+.4f}")
    print(f"\nP-lev-e (at least one lever arm moves tokens K3-K6 above 0.002 with CI > 0): {any_k3}")


if __name__ == "__main__":
    main()
