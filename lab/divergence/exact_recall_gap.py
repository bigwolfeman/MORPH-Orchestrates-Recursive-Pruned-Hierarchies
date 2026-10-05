"""Is LXTUL's CE deficit against plain concentrated on tokens plain can copy from far back?

Prereg: lab/experiments/planned/2026-10-04-exact-recall-gap-probe.md (read it before
touching this file; the Method there is authoritative and is NOT edited by this script).
Offline, CPU only, no model forward, no checkpoint load, no GPU touched: the question is
answered from the two existing per-token sweep files (`core_depth_sweep.py`'s
`*.tokens.npz`) plus the raw validation token stream, reproduced bit-for-bit with the SAME
loader call `core_depth_sweep.py` used (`create_dataloader(tok, ds, 2048, 8,
split="validation", skip_samples=0, bag_size=0, tul=None)` -> `stream_from_loader`).

Row / position convention
--------------------------
`core_depth_sweep.py`'s plain control packs the stream into 480 non-overlapping rows of
``seq_len + 1 = 1025`` raw tokens (`_rows.pack_rows`, the `plain` branch): row r holds
``stream[r*1025 : (r+1)*1025]``; local position p = 0..1023 is an INPUT position, and
``stream[r*1025 + p + 1]`` is the label it is scored against (`_rows.py`: "the label of a
scored position is stream[pos_index + 1]"). Every `tok_index` plain and LXTUL share is
exactly ``row*1025 + p`` for row in 0..479, p in 0..1023 (checked below: plain's full
tok_index set is a SUBSET of LXTUL's, so every plain index is a valid LXTUL one too).

This script re-indexes by the PREDICTED token's own position in the row's raw 1025-token
sequence: ``i = p + 1`` (1 <= i <= 1024), so `x[i]` in the prereg's bucket definitions is
literally ``row_tokens[i]``, the target label, and the scored `tok_index` for that target
is ``row*1025 + i - 1``. Two checks confirm `i` lines up with the PREDICTED token and not
the input token: (1) `x[i] == stream[row*1025 + i]` holds by construction from the
`_rows.py` identity above; (2) printed below, plain's own mean CE is lower on the far
bigram/token repeat buckets than on novel -- a model that can copy is better, on average,
at predicting a token it has seen before than one it has not, so a mis-aligned `i` (CE
paired with the WRONG target) would most likely wash this out or invert it.

Bucket definition (prereg Method, read literally: EXISTENCE at distance >= far_gap, not
necessarily the single globally-nearest occurrence at any distance). For target position
i (i >= min_i, default 64), with x = the row's own 1025-token sequence:

  * far bigram repeat: exists j in [1, i-1], j <= i - far_gap, with
    (x[j-1], x[j]) == (x[i-1], x[i])
  * far token repeat:  no such bigram j, but exists j in [0, i-1], j <= i - far_gap,
    with x[j] == x[i]
  * novel:             neither

`far_gap` defaults to 33 (span cap 32, so distance >= 33 guarantees the earlier copy sits
outside the target's own TUL span -- prereg). The reported distance for a far bucket is to
the NEAREST qualifying (already-far) j: a per-row, per-position loop over a running
position index (value -> sorted position list, `bisect_right`) -- ~1025 steps x 480 rows,
not a loop over the whole ~500k-token corpus. Every reduction after that (gap, share,
bootstrap) is vectorised numpy over the ~460k classified tokens.

Usage:
    python lab/divergence/exact_recall_gap.py \
        --out /home/wolfe/morph-scratch/recall/exact_recall_gap.json
(all other paths default to the inputs named in the prereg's CE test queue item).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_right
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import build_cfg                # noqa: E402
from _rows import stream_from_loader         # noqa: E402
from sweep_score import load_sweep, paired   # noqa: E402

ROW_STRIDE = 1025   # plain control: seq_len (1024) + 1 raw tokens per row
SEQ_LEN = 1024
N_ROWS = 480
FAR_GAP = 33
MIN_I = 64
DIST_BINS = (("33-64", 33, 64), ("65-256", 65, 256), ("257+", 257, None))
N_BOOT = 1000
BOOT_SEED = 0

DEFAULT_LX = ("/home/wolfe/morph-scratch/abc/"
              "slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm-10k/"
              "sweep_slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm-10k"
              "_10000.slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm-10k"
              ".tokens.npz")
DEFAULT_LX_JSON = ("/home/wolfe/morph-scratch/abc/"
                    "slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm-10k/"
                    "sweep_slot-spandec-strict-fan4-all-fp01-lsel-joint-rf-lam1-rank-cnorm-10k"
                    "_10000.json")
DEFAULT_PL = ("/home/wolfe/morph-scratch/plain-10k/"
              "sweep_plain-panel-norm-match_10000.plain-panel-norm-match.tokens.npz")
DEFAULT_PL_JSON = "/home/wolfe/morph-scratch/plain-10k/sweep_plain-panel-norm-match_10000.json"

# The reproduction target this script must hit before it trusts anything else
# (gap_vs_plain10k.json, ce_6_minus_ruler_ce6): point, lo, hi.
EXPECTED_GAP = (0.3564, 0.3356, 0.3791)
EXPECTED_TOL = 2e-3


def row_buckets(x: np.ndarray, min_i: int = MIN_I, far_gap: int = FAR_GAP
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify every target position i (min_i <= i <= len(x)-1) of one row.

    Returns ``(local_idx, bucket, dist)`` aligned to the classified positions:
    ``local_idx = i - 1`` (the scored INPUT position, matching the npz's ``tok_index``
    convention relative to the row's own start), ``bucket`` in {0: novel, 1: far token
    repeat, 2: far bigram repeat}, ``dist = i - j`` to the NEAREST qualifying j (-1 if
    bucket is novel).
    """
    n = len(x)
    tok_pos: dict[int, list[int]] = defaultdict(list)
    bi_pos: dict[tuple[int, int], list[int]] = defaultdict(list)
    local_idx: list[int] = []
    bucket: list[int] = []
    dist: list[int] = []
    for i in range(n):
        if i >= min_i:
            thr = i - far_gap
            j_bi = None
            if i >= 1:
                lst = bi_pos.get((int(x[i - 1]), int(x[i])))
                if lst:
                    k = bisect_right(lst, thr) - 1
                    if k >= 0:
                        j_bi = lst[k]
            if j_bi is not None:
                local_idx.append(i - 1); bucket.append(2); dist.append(i - j_bi)
            else:
                j_tok = None
                lst = tok_pos.get(int(x[i]))
                if lst:
                    k = bisect_right(lst, thr) - 1
                    if k >= 0:
                        j_tok = lst[k]
                if j_tok is not None:
                    local_idx.append(i - 1); bucket.append(1); dist.append(i - j_tok)
                else:
                    local_idx.append(i - 1); bucket.append(0); dist.append(-1)
        tok_pos[int(x[i])].append(i)
        if i >= 1:
            bi_pos[(int(x[i - 1]), int(x[i]))].append(i)
    return (np.asarray(local_idx, dtype=np.int64),
            np.asarray(bucket, dtype=np.int8),
            np.asarray(dist, dtype=np.int64))


def dist_submask(bucket_all: np.ndarray, dist_all: np.ndarray, code: int,
                  lo: int, hi: int | None) -> np.ndarray:
    """Boolean mask: this bucket code, and distance in ``[lo, hi]`` (``hi=None`` -> open)."""
    hi_bound = hi if hi is not None else np.inf
    return (bucket_all == code) & (dist_all >= lo) & (dist_all <= hi_bound)


def gather(tok_index: np.ndarray, ce: np.ndarray, query: np.ndarray) -> np.ndarray:
    """``ce`` values at the positions in ``tok_index`` matching ``query`` (every query
    value MUST be present — asserted). No assumption that ``tok_index`` arrives sorted."""
    order = np.argsort(tok_index, kind="stable")
    sorted_idx = tok_index[order]
    pos = np.searchsorted(sorted_idx, query)
    ok = (pos < len(sorted_idx)) & (sorted_idx[np.clip(pos, 0, len(sorted_idx) - 1)] == query)
    if not ok.all():
        raise RuntimeError(f"{(~ok).sum()} query indices not found in tok_index — "
                            "the two sweeps do not share this token after all")
    return ce[order[pos]]


def bootstrap_ci(row_sum: np.ndarray, row_cnt: np.ndarray, draws: np.ndarray
                  ) -> tuple[float, float]:
    """95% CI of the ratio-of-sums statistic, resampled over the 480-row draw matrix."""
    bsum = row_sum[draws].sum(axis=1)
    bcnt = row_cnt[draws].sum(axis=1)
    with np.errstate(invalid="ignore"):
        boot = bsum / bcnt
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lx", default=DEFAULT_LX)
    ap.add_argument("--lx-json", default=DEFAULT_LX_JSON)
    ap.add_argument("--pl", default=DEFAULT_PL)
    ap.add_argument("--pl-json", default=DEFAULT_PL_JSON)
    ap.add_argument("--config", default="notul_panel_norm_match",
                     help="Hydra config read ONLY for cfg.data.{tokenizer,dataset} — "
                          "identical on both arms (checked at run time via the hydra "
                          "config dumps under each run's hy/.hydra/config.yaml).")
    ap.add_argument("--depth", default="6")
    ap.add_argument("--min-i", type=int, default=MIN_I)
    ap.add_argument("--far-gap", type=int, default=FAR_GAP)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--expect-gap-json", default=None,
                    help="a paired_vs_ruler gap json ({arm: {ce_<d>_minus_ruler_ce<d>: "
                         "[pt, lo, hi]}}) whose gap this run must reproduce before it "
                         "trusts anything; default: the shipped 10k pair (EXPECTED_GAP)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.depth

    # ---- 1. sanity: reproduce the shipped +0.3564 gap with the SAME function that
    # produced it (sweep_score.paired), before trusting anything built below. ----
    lx_sweep = load_sweep(a.lx_json)
    pl_sweep = load_sweep(a.pl_json)
    assert lx_sweep is not None, a.lx_json
    assert pl_sweep is not None, a.pl_json
    search = (os.path.dirname(a.lx_json), os.path.dirname(a.pl_json))
    point, lo, hi, pairing = paired(lx_sweep, d, pl_sweep, d, search)
    print(f"[sanity] paired(lx, ce_{d}, plain, ce_{d}) = {point:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"({pairing})")
    exp_pt, exp_lo, exp_hi = EXPECTED_GAP
    exp_src = "gap_vs_plain10k.json"
    if a.expect_gap_json is not None:
        # Any other pair (an arm against plain 5k): the target is THAT pair's own
        # paired_vs_ruler output, read from disk, never a number typed in here.
        with open(a.expect_gap_json) as f:
            gj = json.load(f)
        if len(gj) != 1:
            raise ValueError(f"{a.expect_gap_json}: expected one arm, got {list(gj)}")
        exp_pt, exp_lo, exp_hi = next(iter(gj.values()))[f"ce_{d}_minus_ruler_ce{d}"]
        exp_src = a.expect_gap_json
    if (abs(point - exp_pt) > EXPECTED_TOL or abs(lo - exp_lo) > EXPECTED_TOL
            or abs(hi - exp_hi) > EXPECTED_TOL):
        raise RuntimeError(
            f"STOP: sanity reproduction FAILED. Got {point:+.4f} [{lo:+.4f}, {hi:+.4f}], "
            f"expected {exp_pt:+.4f} [{exp_lo:+.4f}, {exp_hi:+.4f}] from "
            f"{exp_src}. Do not trust the table below; the pairing or depth "
            f"is wrong.")
    print(f"[sanity] OK — reproduces {exp_src}'s ce_{d}_minus_ruler_ce{d} "
          f"within {EXPECTED_TOL}.")

    # ---- 2. load the per-token npz's the table below actually reads. ----
    lx = np.load(a.lx)
    pl = np.load(a.pl)
    lx_idx, lx_ce = lx["tok_index"], lx[f"ce_{d}"].astype(np.float64)
    pl_idx, pl_ce = pl["tok_index"], pl[f"ce_{d}"].astype(np.float64)
    if not np.isin(pl_idx, lx_idx).all():
        raise RuntimeError("STOP: plain's tok_index is not a subset of LXTUL's; the "
                            "pairing assumption this script relies on does not hold.")
    if pl_idx.max() >= N_ROWS * ROW_STRIDE or len(pl_idx) != N_ROWS * SEQ_LEN:
        raise RuntimeError(f"STOP: plain tok_index shape ({len(pl_idx)} entries, max "
                            f"{pl_idx.max()}) does not match the assumed {N_ROWS} rows x "
                            f"{ROW_STRIDE} stride; re-check --rows/--batch of the sweep.")

    # ---- 3. reproduce the raw token stream offline, no model, same loader call as
    # core_depth_sweep.py (`create_dataloader(..., 2048, 8, split="validation", "
    # skip_samples=0, bag_size=0, tul=None)` -> `stream_from_loader`). ----
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from morph.training.data import create_dataloader
    cfg = build_cfg(a.config, [])
    n_need = N_ROWS * ROW_STRIDE
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = np.asarray(stream_from_loader(loader, n_need)[:n_need], dtype=np.int64)
    X = stream.reshape(N_ROWS, ROW_STRIDE)
    print(f"[stream] reproduced {len(stream)} tokens ({cfg.data.tokenizer}, "
          f"{cfg.data.dataset}); X.shape={X.shape}")

    # ---- 4. classify every target position i >= min_i of every row. ----
    idx_all, bucket_all, dist_all, row_all = [], [], [], []
    for r in range(N_ROWS):
        local_idx, bucket, dist = row_buckets(X[r], a.min_i, a.far_gap)
        idx_all.append(r * ROW_STRIDE + local_idx)
        bucket_all.append(bucket)
        dist_all.append(dist)
        row_all.append(np.full(len(local_idx), r, dtype=np.int32))
    idx_all = np.concatenate(idx_all)
    bucket_all = np.concatenate(bucket_all)
    dist_all = np.concatenate(dist_all)
    row_all = np.concatenate(row_all)
    n_tok = len(idx_all)
    print(f"[classify] {n_tok} target positions (i >= {a.min_i}) over {N_ROWS} rows")

    pl_at = gather(pl_idx, pl_ce, idx_all)
    lx_at = gather(lx_idx, lx_ce, idx_all)
    gap = lx_at - pl_at

    # sanity check #2: an i that lines up with the PREDICTED token should be easier
    # to predict on repeat than on novel, for the plain model, on average.
    bucket_names_short = {0: "novel", 1: "far_token", 2: "far_bigram"}
    print("[sanity] plain mean CE by bucket (should be lowest on repeats if `i` "
          "indexes the predicted token, not the input token):")
    for code, name in bucket_names_short.items():
        m = bucket_all == code
        if m.any():
            print(f"    {name:10s} plain_ce={pl_at[m].mean():.4f} n={int(m.sum())}")

    # ---- 5. per-row sums (gap, plain CE, LXTUL CE, token count) for every reported
    # bucket, so the bootstrap below resamples ROWS, not tokens. ----
    rng = np.random.default_rng(a.seed)
    draws = rng.integers(0, N_ROWS, size=(a.n_boot, N_ROWS))

    def bucket_stats(mask: np.ndarray, name: str) -> dict:
        cnt = int(mask.sum())
        if cnt == 0:
            return {"name": name, "n_tokens": 0}
        row_sum_gap = np.bincount(row_all[mask], weights=gap[mask], minlength=N_ROWS)
        row_cnt = np.bincount(row_all[mask], minlength=N_ROWS).astype(np.float64)
        row_sum_pl = np.bincount(row_all[mask], weights=pl_at[mask], minlength=N_ROWS)
        row_sum_lx = np.bincount(row_all[mask], weights=lx_at[mask], minlength=N_ROWS)
        point = float(row_sum_gap.sum() / row_cnt.sum())
        lo, hi = bootstrap_ci(row_sum_gap, row_cnt, draws)
        return {
            "name": name, "n_tokens": cnt, "token_share": cnt / n_tok,
            "mean_gap": point, "mean_gap_ci": [lo, hi],
            "plain_mean_ce": float(row_sum_pl.sum() / row_cnt.sum()),
            "lxtul_mean_ce": float(row_sum_lx.sum() / row_cnt.sum()),
            "total_gap": float(row_sum_gap.sum()),
        }

    results: dict[str, dict] = {}
    results["far_bigram_repeat"] = bucket_stats(bucket_all == 2, "far_bigram_repeat")
    results["far_token_repeat"] = bucket_stats(bucket_all == 1, "far_token_repeat")
    results["novel"] = bucket_stats(bucket_all == 0, "novel")
    for code, label in ((2, "far_bigram_repeat"), (1, "far_token_repeat")):
        for bname, blo, bhi in DIST_BINS:
            m = dist_submask(bucket_all, dist_all, code, blo, bhi)
            results[f"{label}@{bname}"] = bucket_stats(m, f"{label}@{bname}")

    grand_total_gap = float(gap.sum())
    for r in results.values():
        if r.get("n_tokens", 0) > 0:
            r["gap_share"] = r["total_gap"] / grand_total_gap
            r["share_ratio"] = (r["gap_share"] / r["token_share"]
                                 if r["token_share"] > 0 else float("nan"))

    # ---- 6. predictions, read against the numbers (not written to the prereg). ----
    def g(name: str) -> float:
        return results[name]["mean_gap"]

    def ci(name: str) -> tuple[float, float]:
        return tuple(results[name]["mean_gap_ci"])

    preds = {
        "R1_far_bigram_ge_2x_novel": {
            "far_bigram_gap": g("far_bigram_repeat"), "novel_gap": g("novel"),
            "ratio": g("far_bigram_repeat") / g("novel") if g("novel") else float("inf"),
            "holds": g("far_bigram_repeat") >= 2 * g("novel"),
        },
        "R2_far_bigram_share_ge_1.5x_tokens": {
            "share_ratio": results["far_bigram_repeat"].get("share_ratio"),
            "holds": (results["far_bigram_repeat"].get("share_ratio") or 0) >= 1.5,
        },
        "R3_novel_gap_ci_above_0": {
            "novel_gap": g("novel"), "novel_ci": ci("novel"),
            "holds": ci("novel")[0] > 0,
        },
        "R4_dist33to64_le_dist257plus": {
            "d33_64": results["far_bigram_repeat@33-64"].get("mean_gap"),
            "d257p": results["far_bigram_repeat@257+"].get("mean_gap"),
            "holds": (results["far_bigram_repeat@33-64"].get("mean_gap", 0)
                      <= results["far_bigram_repeat@257+"].get("mean_gap", 1e18)),
        },
    }

    out = {
        "sanity_reproduction": {"point": point, "lo": lo, "hi": hi, "pairing": pairing,
                                 "expected": list(EXPECTED_GAP)},
        "n_classified_tokens": n_tok,
        "grand_total_gap": grand_total_gap,
        "buckets": results,
        "predictions": preds,
        "config": {"min_i": a.min_i, "far_gap": a.far_gap, "n_boot": a.n_boot,
                    "seed": a.seed, "depth": d},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")

    # ---- 7. the table. ----
    header = (f"{'bucket':26s} {'n_tok':>8s} {'tok_share':>9s} {'mean_gap':>22s} "
              f"{'gap_share':>9s} {'ratio':>7s} {'plain_ce':>8s} {'lxtul_ce':>8s}")
    print(header)
    print("-" * len(header))
    order = ["far_bigram_repeat", "far_bigram_repeat@33-64", "far_bigram_repeat@65-256",
             "far_bigram_repeat@257+", "far_token_repeat", "far_token_repeat@33-64",
             "far_token_repeat@65-256", "far_token_repeat@257+", "novel"]
    for name in order:
        r = results[name]
        if r.get("n_tokens", 0) == 0:
            print(f"{name:26s} {0:8d}")
            continue
        lo_, hi_ = r["mean_gap_ci"]
        print(f"{name:26s} {r['n_tokens']:8d} {r['token_share']:9.4f} "
              f"{r['mean_gap']:+7.4f} [{lo_:+.4f},{hi_:+.4f}] {r['gap_share']:9.4f} "
              f"{r['share_ratio']:7.2f} {r['plain_mean_ce']:8.4f} {r['lxtul_mean_ce']:8.4f}")
    print()
    for k, v in preds.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
