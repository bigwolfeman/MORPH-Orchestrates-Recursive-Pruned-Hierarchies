"""LXTUL-E Stage 1 readers against notul on IDENTICAL tokens, overall and by span offset.

Joins three per-token files: the Stage 1 scorer's (``lxtul_e_stage1_score.py``), the plain
model's ``core_depth_sweep`` tokens npz (notul, read at depth 6, its training mean) and the
Stage 0 scorer's (the ruler's teacher-forced span decoder, ``ce_tf``). Each file indexes a
token in its own convention: the coda set lines up with the plain sweep at shift 0, the head
set and Stage 0 at +1. The shift is DETECTED per set by correlation against notul's CE (one
shift must correlate > 0.5 and the others < 0.3, or the script raises), and every join runs
in notul's index space. Joining two sets on their raw indexes pairs each token with its
neighbour; that mistake is what this layout prevents.

Usage:
  python lab/divergence/lxtul_e_notul_pair.py --s1 stage1_score.tokens.npz \\
      --plain sweep_plain-panel-norm-match_5000...tokens.npz --s0 stage0_score.tokens.npz
"""
from __future__ import annotations

import argparse

import numpy as np

OFFSET_BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8-15", 8, 15),
                  ("16+", 16, 10**6)]


def detect_shift(idx: np.ndarray, ce: np.ndarray, plain_idx: np.ndarray,
                 plain_ce: np.ndarray, name: str) -> int:
    """The shift s with ``idx == plain_idx + s`` on the same token, found by correlation."""
    corr = {}
    for sh in (-1, 0, 1):
        _, a, b = np.intersect1d(idx, plain_idx + sh, return_indices=True)
        corr[sh] = float(np.corrcoef(ce[a], plain_ce[b])[0, 1]) if len(a) > 1000 else np.nan
    best = max(corr, key=lambda k: np.nan_to_num(corr[k], nan=-9.0))
    print(f"[align {name}] corr by shift {({k: round(v, 3) for k, v in corr.items()})} "
          f"-> {best:+d}")
    others = [np.nan_to_num(v) for k, v in corr.items() if k != best]
    if not (corr[best] > 0.5 and all(o < 0.3 for o in others)):
        raise SystemExit(f"ambiguous alignment for {name}: {corr}")
    return best


def boot_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def scatter_onto(n: int, at: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A length-n array holding ``values`` at ``at`` and NaN elsewhere."""
    out = np.full(n, np.nan)
    out[at] = values
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s1", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--s0", required=True)
    args = ap.parse_args()
    z = np.load(args.s1)
    p = np.load(args.plain)
    s0 = np.load(args.s0)
    pi = p["tok_index"].astype(np.int64)
    notul = p["ce_6"]

    # coda set: every token the coda is charged on
    ci = z["coda_idx"]
    sh_c = detect_shift(ci, z["e1_coda_6"], pi, notul, "coda")
    _, a, b = np.intersect1d(ci, pi + sh_c, return_indices=True)
    cols = {"notul": notul[b], "e1 coda": z["e1_coda_6"][a], "e4 coda mix": z["e4_coda_mix_6"][a]}
    if "ruler_coda_6" in z:
        cols["ruler coda"] = z["ruler_coda_6"][a]
    print(f"\nCODA tokens: {len(a)} of {len(ci)} matched")
    for k, v in cols.items():
        d = v - cols["notul"]
        lo, hi = boot_ci(d)
        print(f"  {k:14s} {v.mean():.4f}   minus notul {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # head set: the next-span tokens the parallel head predicts, by offset in the span
    hi_ = z["head_idx"]
    sh_h = detect_shift(hi_, z["e1_par_6"], pi, notul, "head")
    _, a, b = np.intersect1d(hi_, pi + sh_h, return_indices=True)
    head_plain = hi_[a] - sh_h                                  # notul index space
    off = z["head_off"].astype(np.int64)[a]
    cols = {"notul": notul[b], "e1 par": z["e1_par_6"][a], "e4 par mix": z["e4_par_mix_6"][a],
            "e4 par mix d1": z["e4_par_mix_1"][a], "e1 par d1": z["e1_par_1"][a]}
    _, ca, ha = np.intersect1d(ci - sh_c, head_plain, return_indices=True)
    sh_0 = detect_shift(s0["tok_index"], s0["ce_tf"], pi, notul, "stage0 TF")
    _, sa, sb = np.intersect1d(s0["tok_index"] - sh_0, head_plain, return_indices=True)
    extra = {"e1 coda": scatter_onto(len(a), ha, z["e1_coda_6"][ca]),
             "e4 coda mix": scatter_onto(len(a), ha, z["e4_coda_mix_6"][ca]),
             "ruler TF": scatter_onto(len(a), sb, s0["ce_tf"][sa])}
    print(f"\nHEAD tokens: {len(a)} of {len(hi_)} matched to notul; coda overlap {len(ca)}; "
          f"stage0 TF overlap {len(sa)}")
    print("  offset    n       " + "".join(f"{k:>14s}" for k in cols)
          + "".join(f"{k:>13s}" for k in extra))
    buckets = [("all", np.ones(len(a), bool))] + [
        (lab, (off >= lo) & (off <= hi)) for lab, lo, hi in OFFSET_BUCKETS]
    for lab, m in buckets:
        row = "".join(f"{v[m].mean():14.4f}" for v in cols.values())
        row += "".join(f"{np.nanmean(v[m]):13.4f}" if np.isfinite(v[m]).any() else f"{'-':>13s}"
                       for v in extra.values())
        print(f"  {lab:6s} {int(m.sum()):8d}  {row}")


if __name__ == "__main__":
    main()
