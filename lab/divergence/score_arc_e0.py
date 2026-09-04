"""Score arc E0 (lab/experiments/planned/2026-09-04-arc-e0-where-depth-earns.md) from a
depth-sweep JSON written with --profile (token_depth_sweep.py or a2_depth_sweep.py).

  python lab/divergence/score_arc_e0.py --sweep LABEL=path.json [--sweep ...] --lo 1 --hi 6

Per arm: Spearman(row CE at depth lo, row earning lo−hi) with a row bootstrap CI (P0a);
earning at offset 0 against the mean earning over offsets 4..31 (P0b); the top decile of
rows by CE at depth lo: its share of total earning against its share of total loss (P0c).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True, help="LABEL=path.json")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for spec in a.sweep:
        label, path = spec.split("=", 1)
        d = json.load(open(path))
        arm = d[label] if label in d else next(iter(d.values()))
        pr = arm["profile"]
        lo, hi = str(a.lo), str(a.hi)
        n = np.asarray(pr["row_n_tokens"])
        ok = n > 0
        c_lo = np.asarray(pr["row_ce_sum"][lo])[ok] / n[ok]
        c_hi = np.asarray(pr["row_ce_sum"][hi])[ok] / n[ok]
        earn = c_lo - c_hi
        R = earn.shape[0]
        rho = spearman(c_lo, earn)
        boots = []
        for _ in range(a.n_boot):
            idx = rng.integers(0, R, R)
            boots.append(spearman(c_lo[idx], earn[idx]))
        b = np.percentile(boots, [2.5, 97.5])
        print(f"{label}: rows {R}  K{a.lo}−K{a.hi} = {float(earn.mean()):+.4f} (row mean)")
        print(f"  P0a Spearman(row CE_{a.lo}, row earning) = {rho:+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]")
        # P0b: offset profile (token-weighted over rows, bootstrap over rows)
        bins = pr["bins"]
        bce_lo = np.asarray(pr["bin_ce_sum"][lo])
        bce_hi = np.asarray(pr["bin_ce_sum"][hi])
        bn = np.asarray(pr["bin_n_tokens"])
        def prof(idx):
            s_lo = bce_lo[idx].sum(0)
            s_hi = bce_hi[idx].sum(0)
            c = np.maximum(bn[idx].sum(0), 1)
            return (s_lo - s_hi) / c
        pt = prof(np.arange(R))
        bb = np.stack([prof(rng.integers(0, R, R)) for _ in range(a.n_boot)])
        lo_ci, hi_ci = np.percentile(bb, 2.5, axis=0), np.percentile(bb, 97.5, axis=0)
        print("  offset profile (earning per token by offset-in-span):")
        for i, (lo_b, hi_b) in enumerate(bins):
            print(f"    {lo_b:>2}-{min(hi_b, 31):<2}: {pt[i]:+.4f} [{lo_ci[i]:+.4f}, {hi_ci[i]:+.4f}]  n={int(bn[:, i].sum())}")
        tail = [i for i, (lo_b, _) in enumerate(bins) if lo_b >= 4]
        tail_earn = (bce_lo[:, tail].sum() - bce_hi[:, tail].sum()) / max(bn[:, tail].sum(), 1)
        ratio = pt[0] / tail_earn if tail_earn != 0 else float("inf")
        print(f"  P0b offset-0 earning / mean earning over offsets 4..31 = {pt[0]:+.4f} / {tail_earn:+.4f} = {ratio:.2f}x")
        # P0c: top decile by CE_lo
        order = np.argsort(-c_lo)
        top = order[: max(1, R // 10)]
        loss_share = float((c_lo[top] * n[ok][top]).sum() / (c_lo * n[ok]).sum())
        tot_earn = float((earn * n[ok]).sum())
        earn_share = float((earn[top] * n[ok][top]).sum() / tot_earn) if tot_earn != 0 else float("nan")
        print(f"  P0c top-decile rows: loss share {loss_share:.3f}, earning share {earn_share:.3f} -> {'below' if earn_share < loss_share else 'NOT below'}")


if __name__ == "__main__":
    main()
