"""Post-hoc reads on ``superposition_probe.py`` token files (NOT preregistered, 2026-09-23).

1. P-7 of the superposition prereg: ``row`` worth on the CONTEXT top2 bucket at offsets
   8..15 (the probe bins offsets on the own bucket only).
2. The token-difficulty check: the ``row`` worth ratio top2 / top1 (context bucket) with
   the offsets 1..n tokens stratified into deciles of ``ce_zero`` (the token's CE with the
   cells zeroed). The ratio of decile-weighted means, block bootstrap over 1,024-token
   stream blocks (the probe's unit).

Usage: python lab/divergence/superposition_strat.py OUT.json PROBE.tokens.npz [...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

BLOCK = 1024
N_BOOT = 2000
N_STRATA = 10


def _resample_weights(blocks: np.ndarray, rng: np.random.Generator):
    ub, inv = np.unique(blocks, return_inverse=True)
    idx = rng.integers(0, len(ub), size=(N_BOOT, len(ub)))
    W = np.zeros((N_BOOT, len(ub)))
    np.add.at(W, (np.arange(N_BOOT)[:, None], idx), 1.0)
    return inv, W


def _boot_mean(w: np.ndarray, sel: np.ndarray, inv: np.ndarray, W: np.ndarray) -> np.ndarray:
    nb = W.shape[1]
    s = np.bincount(inv[sel], weights=w[sel], minlength=nb)
    c = np.bincount(inv[sel], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ s) / (W @ c)


def read(path: str) -> dict:
    z = np.load(path)
    off, bc = z["offset"], z["bucket_ctx"]
    w = (z["ce_row"] - z["ce_own"]).astype(np.float64)
    inv, W = _resample_weights(z["tok_index"] // BLOCK, np.random.default_rng(0))
    out: dict = {}
    m = (bc == 1) & (off >= 8) & (off <= 15)
    b = _boot_mean(w, m, inv, W)
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    out["p7_row_ctx_top2_off8_15"] = {"mean": float(w[m].mean()), "lo": float(lo),
                                      "hi": float(hi), "n": int(m.sum())}
    pos = off >= 1
    edges = np.quantile(z["ce_zero"][pos], np.linspace(0, 1, N_STRATA + 1))
    dec = np.digitize(z["ce_zero"], edges[1:-1])
    share = np.array([(pos & (dec == k)).sum() for k in range(N_STRATA)], np.float64)
    share /= share.sum()
    num = den = 0.0
    bnum = np.zeros(N_BOOT)
    bden = np.zeros(N_BOOT)
    strata = []
    for k in range(N_STRATA):
        s1 = pos & (dec == k) & (bc == 0)
        s2 = pos & (dec == k) & (bc == 1)
        m1, m2 = float(w[s1].mean()), float(w[s2].mean())
        strata.append({"decile": k, "top1": m1, "top2": m2, "n_top1": int(s1.sum()),
                       "n_top2": int(s2.sum())})
        num += share[k] * m2
        den += share[k] * m1
        bnum += share[k] * _boot_mean(w, s2, inv, W)
        bden += share[k] * _boot_mean(w, s1, inv, W)
    lo, hi = np.nanquantile(bnum / bden, [0.025, 0.975])
    out["strat_ratio_row_ctx"] = {"point": num / den, "lo": float(lo), "hi": float(hi)}
    out["strata"] = strata
    return out


def main() -> None:
    res = {p.rsplit("/", 1)[-1]: read(p) for p in sys.argv[2:]}
    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)
    for k, r in res.items():
        p7, sr = r["p7_row_ctx_top2_off8_15"], r["strat_ratio_row_ctx"]
        print(f"{k}\n  P-7 {p7['mean']:+.4f} [{p7['lo']:+.4f}, {p7['hi']:+.4f}] n={p7['n']}"
              f"\n  stratified ratio {sr['point']:.3f} [{sr['lo']:.3f}, {sr['hi']:.3f}]")


if __name__ == "__main__":
    main()
