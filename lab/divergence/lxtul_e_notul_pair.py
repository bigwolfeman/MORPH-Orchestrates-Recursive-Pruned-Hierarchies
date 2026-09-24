"""LXTUL-E readers (Stage 1 and Stage 2) against notul on IDENTICAL tokens, overall and by span offset.

Joins per-token files: a scorer's (``lxtul_e_stage1_score.py``, fixed keys ``coda_idx`` /
``e1_coda_6`` / ...; or ``lxtul_e_stage2_score.py``, per-arm keys ``<arm>_coda_idx`` /
``<arm>_coda_mix_6`` / ...), the plain model's ``core_depth_sweep`` tokens npz (notul, read
at depth 6, its training mean) and, optionally, the Stage 0 scorer's (the ruler's
teacher-forced span decoder, ``ce_tf``). Each file indexes a
token in its own convention: the coda set lines up with the plain sweep at shift 0, the head
set and Stage 0 at +1. The shift is DETECTED per set by correlation against notul's CE (one
shift must correlate > 0.5 and the others < 0.3, or the script raises), and every join runs
in notul's index space. Joining two sets on their raw indexes pairs each token with its
neighbour; that mistake is what this layout prevents.

Usage:
  python lab/divergence/lxtul_e_notul_pair.py --scores stage{1,2}_score.tokens.npz \\
      --plain sweep_plain-panel-norm-match_5000...tokens.npz [--s0 stage0_score.tokens.npz]
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


def readers(z) -> tuple[list, list]:
    """The coda readers ``(label, idx, ce@6, ce@1)`` and head readers ``(label, idx, offset,
    ce@6, ce@1)`` of a Stage 1 or a Stage 2 tokens npz."""
    coda, head = [], []
    if "coda_idx" in z:                                             # Stage 1 layout
        for lab, key in (("e1", "e1_coda"), ("e4", "e4_coda_mix")):
            coda.append((f"{lab} coda", z["coda_idx"], z[f"{key}_6"], z[f"{key}_1"]))
        if "ruler_coda_6" in z:
            coda.append(("ruler coda", z["coda_idx"], z["ruler_coda_6"], None))
        for lab, key in (("e1", "e1_par"), ("e4", "e4_par_mix")):
            head.append((f"{lab} par", z["head_idx"], z["head_off"], z[f"{key}_6"],
                         z[f"{key}_1"]))
        return coda, head
    arms = sorted(k[:-len("_coda_idx")] for k in z.files if k.endswith("_coda_idx"))
    if not arms:
        raise SystemExit("no coda readers in this npz (neither Stage 1 nor Stage 2 keys)")
    for arm in arms:
        coda.append((f"{arm} coda", z[f"{arm}_coda_idx"], z[f"{arm}_coda_mix_6"],
                     z[f"{arm}_coda_mix_1"]))
        head.append((f"{arm} par", z[f"{arm}_head_idx"], z[f"{arm}_head_off"],
                     z[f"{arm}_par_mix_6"], z[f"{arm}_par_mix_1"]))
    return coda, head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", required=True, help="a Stage 1 or Stage 2 tokens npz")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--s0", default=None, help="Stage 0 tokens npz (the ruler's TF decoder)")
    args = ap.parse_args()
    z = np.load(args.scores)
    p = np.load(args.plain)
    pi = p["tok_index"].astype(np.int64)
    notul = p["ce_6"]
    coda, head = readers(z)

    print("\nCODA tokens (each reader joined to notul in notul index space):")
    for lab, idx, ce6, ce1 in coda:
        sh = detect_shift(idx, ce6, pi, notul, lab)
        _, a, b = np.intersect1d(idx, pi + sh, return_indices=True)
        d = ce6[a] - notul[b]
        lo, hi = boot_ci(d)
        k = "" if ce1 is None else f"   K1-K6 {(ce1[a] - ce6[a]).mean():+.4f}"
        print(f"  {lab:16s} n={len(a)}  {ce6[a].mean():.4f}  notul {notul[b].mean():.4f}  "
              f"minus notul {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{k}")

    tf = None
    if args.s0:
        s0 = np.load(args.s0)
        sh0 = detect_shift(s0["tok_index"], s0["ce_tf"], pi, notul, "stage0 TF")
        tf = (s0["tok_index"] - sh0, s0["ce_tf"])
    print("\nHEAD tokens by span offset (each reader on its OWN tokens, notul on the same):")
    for lab, idx, off_all, ce6, ce1 in head:
        sh = detect_shift(idx, ce6, pi, notul, lab)
        _, a, b = np.intersect1d(idx, pi + sh, return_indices=True)
        off = off_all.astype(np.int64)[a]
        cols = {"notul": notul[b], lab: ce6[a], f"{lab} d1": ce1[a]}
        if tf is not None:
            _, ta, tb = np.intersect1d(tf[0], idx[a] - sh, return_indices=True)
            cols["ruler TF"] = scatter_onto(len(a), tb, tf[1][ta])
        print(f"  [{lab}] {len(a)} of {len(idx)} matched")
        print("    offset    n     " + "".join(f"{k:>16s}" for k in cols))
        buckets = [("all", np.ones(len(a), bool))] + [
            (bl, (off >= lo) & (off <= hi)) for bl, lo, hi in OFFSET_BUCKETS]
        for bl, m in buckets:
            if not m.any():
                continue
            row = "".join(f"{np.nanmean(v[m]):16.4f}" if np.isfinite(v[m]).any()
                          else f"{'-':>16s}" for v in cols.values())
            print(f"    {bl:6s} {int(m.sum()):8d}  {row}")


if __name__ == "__main__":
    main()
