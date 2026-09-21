"""Does SONAR carry sequential structure on MORPH's spans?  (LXTUL-P rung P3 precondition)

Reads a SONAR span cache (`morph/model/sonar_cache.py`) and its `spans.keys.npy` (stream
order) and reports, over the first `--n` stream keys:

  C-1  mean cos(span_i, span_{i+1}) - mean cos(span_i, span_random)
  C-2  successor retrieval top-1 among 1 + `--cands` candidates by cosine to span_i

Both with a percentile bootstrap. Adjacent stream keys are the packer's successive spans
except at row boundaries (about 1 in 30) and where the dedupe dropped a repeat; both count
AGAINST the excess. Prereg: lab/experiments/planned/2026-09-21-lctul-tlow-sonar.md.

    python lab/divergence/sonar_cache_structure.py --cache ~/sonar-cache/tul_code_cfg_tlow \
        --n 20000 --out RESULTS/sonar_cache_structure_2k.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from morph.model.sonar_cache import SonarSpanCache


def _boot(vals: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    m = float(vals.mean())
    idx = rng.integers(0, vals.shape[0], size=(n_boot, vals.shape[0]))
    bs = vals[idx].mean(axis=1)
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def structure(cache: SonarSpanCache, keys: np.ndarray, n: int, cands: int,
              n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k = keys[:n]
    rows = cache.lookup(k)
    if int((rows < 0).sum()):
        raise RuntimeError(f"{int((rows < 0).sum())} of the first {n} stream keys miss the cache")
    e = cache.take(rows)
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-6)
    cur, nxt = e[:-1], e[1:]
    cos_next = (cur * nxt).sum(1)
    perm = rng.permutation(cur.shape[0])
    cos_rand = (cur * cur[perm]).sum(1)
    excess = cos_next - cos_rand
    # C-2: for each query i, the true successor against `cands` random keys.
    q = rng.choice(cur.shape[0], size=min(2000, cur.shape[0]), replace=False)
    top1 = np.zeros(q.shape[0], dtype=np.float64)
    for j, i in enumerate(q):
        r = rng.choice(e.shape[0], size=cands, replace=False)
        r = r[r != i + 1]
        s_true = float(cur[i] @ nxt[i])
        s_rand = e[r] @ cur[i]
        top1[j] = float(s_true > s_rand.max())
    c1 = _boot(excess, rng, n_boot)
    c2 = _boot(top1, rng, n_boot)
    return {
        "n_keys": int(n), "n_pairs": int(cur.shape[0]), "n_queries": int(q.shape[0]),
        "cands": int(cands),
        "cos_next_mean": float(cos_next.mean()), "cos_rand_mean": float(cos_rand.mean()),
        "c1_excess": {"point": c1[0], "lo": c1[1], "hi": c1[2], "threshold": 0.10},
        "c2_top1": {"point": c2[0], "lo": c2[1], "hi": c2[2], "threshold": 0.05,
                    "chance": 1.0 / (cands + 1)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--cands", type=int, default=999)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache = SonarSpanCache(os.path.expanduser(a.cache))
    keys = np.load(os.path.join(os.path.expanduser(a.cache), "spans.keys.npy"))
    out = structure(cache, keys, a.n, a.cands, a.boot, a.seed)
    out["cache"] = os.path.expanduser(a.cache)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"cos next {out['cos_next_mean']:.4f}  rand {out['cos_rand_mean']:.4f}  "
          f"C-1 excess {out['c1_excess']['point']:+.4f} [{out['c1_excess']['lo']:+.4f}, "
          f"{out['c1_excess']['hi']:+.4f}] (>= 0.10)  "
          f"C-2 top1 {out['c2_top1']['point']:.4f} [{out['c2_top1']['lo']:.4f}, "
          f"{out['c2_top1']['hi']:.4f}] (>= 0.05, chance {out['c2_top1']['chance']:.4f})")


if __name__ == "__main__":
    main()
