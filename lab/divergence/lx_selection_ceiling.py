"""LX selection ceiling: how concentrated is the per-span credit over an LXTUL-E arm's K
rollouts, and how much could ANY per-span selector over them buy. Re-scores the STORED
per-rollout coda log-probs; CPU only, no forward pass, no fitting.

Prereg: ``lab/experiments/planned/2026-09-26-lx-selection-ceiling.md`` (asked for by the
external council reviewers after LX-Carry Stage 0). The row rebuild, the stream-index join,
today's read, the EOS handling and the offsets are ``lx_carry_stage0.rescore``; the
bootstrap is the Stage 1 scorer's ``_ci``; the mixture is
``morph.model.rollout_mixture.log_mean_exp``. This script only groups tokens into spans and
reads the span scores.

TERMS (one meaning each):

  coda token     a position the Stage 1/2 scorer scored; ``lp_k(i)`` = rollout k's log-prob
                 of its label (``-<label>_coda_code_<d>[k]``).
  span           a run of one ``bag_id`` on a packed row holding >= 1 coda token
                 (``(row, seg)`` from ``rescore``): the unit of the training mixture and of
                 the eval restart. A row's first span and its dump bin are spans too.
  S_k(s)         sum of ``lp_k`` over span s's coda tokens.
  credit         ``c_k(s) = softmax_k S_k(s)``: the posterior over rollouts after the WHOLE
                 span; ``KL(c || uniform) = sum_k c_k log(K c_k)``; ``c_max`` its largest
                 entry, ``argmax`` its rollout.
  Bayes read     today's per-token read: uniform restart per span, sequential Bayes inside
                 it (``MORPHTransformer._enum_position_nll``, recomputed by ``rescore``).
  span mixture   ``-log (1/K) sum_k exp S_k(s)``. By the chain rule of a mixture a span's
                 Bayes-read tokens SUM to it exactly (asserted per span), so it is reported
                 overall only: it has no other per-token split.
  oracle         per span, ``-lp_{k*}`` on every token, ``k* = argmax_k S_k(s)``: the best
                 any per-span selector can do (it looks ahead at the span's own tokens).
  best fixed     the one rollout with the lowest total NLL over all spans, on every token
                 (chosen on the same rows; 4 candidates).
  ceiling bound  ``log K * spans / tokens``: the most (Bayes - oracle) can be per token,
                 since ``mixture - oracle = log(K c_max) <= log K`` per span.
  token-shuffle  (post-hoc diagnostic, added after the first run; not in the prereg's
  null           predictions) the rollout axis permuted independently at every token:
                 each token keeps its K values, but no rollout's advantage persists across
                 a span. A lookahead oracle beats the mixture even then (the max of K noisy
                 span sums), so (Bayes - oracle) minus its null is the part of the ceiling
                 that comes from span-coherent rollout differences. Overall only (the
                 Bayes read's overall mean is the span mixture's, so no re-read is needed).
  offset / block ``_earning.BINS`` offsets (-1: first span or dump bin); stream index //
                 1,024, the paired bootstrap unit. Span-level readings use one unit per
                 span, in the block of its first coda token.

READINGS (per checkpoint and forced depth; 95 % CI, paired block bootstrap, 2,000
resamples, seed 0):

  credit     mean KL, mean entropy, fraction of spans with c_max > 0.5 / > 0.9, argmax
             shares; per checkpoint, P(argmax at depth a == argmax at depth b) with the
             chance level ``sum_k p_a(k) p_b(k)`` from the two marginals.
  per-token  CE of the Bayes read, span mixture, oracle, best fixed, each single rollout;
             Bayes - oracle (the selection ceiling), best fixed - Bayes, best fixed -
             oracle, overall, at offset 0 and in every bin.
  null       mean KL and overall Bayes - oracle under the token-shuffle null
             (``--null-draws`` draws, seed ``--seed``), and the observed minus the null.

SELF-CHECKS (each RAISES): everything ``rescore`` checks (rebuilt positions == stored
``coda_idx``; per-row switching likelihood); today's read vs the stored ``coda_mix`` (max
|dev| <= ``--tol``); per span, the Bayes tokens sum to the span mixture (``--span-tol``),
oracle <= mixture <= oracle + log K, mixture - oracle == log(K c_max); oracle <= best
fixed overall; the spans are the same at every depth.

Usage (CPU, the UPS rule while the GPU trains):
  PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    nice -n 19 taskset -c 0 python lab/divergence/lx_selection_ceiling.py \\
    --npz /home/wolfe/morph-scratch/lxtul-10k/score.tokens.npz \\
    --label fp01_5k --label fp01_10k --config tul_slot_spandec_strict_e4probe_fp01 \\
    --out .../selection_ceiling_fp01.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _earning import BINS  # noqa: E402
from lx_carry_stage0 import rescore  # noqa: E402
from lxtul_e_stage1_score import N_BOOT, _blocks, _ci, _fmt, val_batches  # noqa: E402

from morph.model.rollout_mixture import log_mean_exp  # noqa: E402

CREDIT_CUTS = (0.5, 0.9)


def span_index(row: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(span [N], first [G])``: each coda token's span number (spans in row, then run
    order) and the stored-order index of each span's first coda token."""
    key = row.astype(np.int64) * (int(seg.max()) + 1) + seg.astype(np.int64)
    _u, first, span = np.unique(key, return_index=True, return_inverse=True)
    return span.reshape(-1), first


def span_scores(lp: np.ndarray, span: np.ndarray, G: int) -> np.ndarray:
    """``S [K, G]`` fp64: per rollout, the sum of ``lp [K, N]`` over each span's tokens."""
    return np.stack([np.bincount(span, weights=lp[k].astype(np.float64), minlength=G)
                     for k in range(lp.shape[0])])


def credit(S: np.ndarray) -> dict:
    """Per span, from ``S [K, G]``: ``kl`` = KL(softmax_k S || uniform) (nats),
    ``entropy``, ``c_max``, ``argmax`` (the first maximum on a tie)."""
    K = S.shape[0]
    logc = torch.log_softmax(torch.from_numpy(S.astype(np.float64)), dim=0)
    c = logc.exp()
    return {"kl": (c * (logc + math.log(K))).sum(0).numpy(),
            "entropy": (-(c * logc).sum(0)).numpy(),
            "c_max": c.max(0).values.numpy(),
            "argmax": np.argmax(S, axis=0)}


def selection_reads(lp: np.ndarray, span: np.ndarray, G: int) -> dict:
    """All span-level and per-token reads of ``lp [K, N]`` (log-probs, fp64) grouped by
    ``span [N]`` into ``G`` spans (module doc, TERMS)."""
    K, N = lp.shape
    lp = lp.astype(np.float64)
    S = span_scores(lp, span, G)
    cr = credit(S)
    oracle = -lp[cr["argmax"][span], np.arange(N)]
    total = -lp.sum(1)                                            # [K]
    k_fixed = int(np.argmin(total))
    return {"S": S, **cr,
            "mix_span": (-log_mean_exp(torch.from_numpy(S), dim=0)).numpy(),
            "oracle_span": -S.max(0), "oracle": oracle,
            "k_fixed": k_fixed, "fixed": -lp[k_fixed],
            "single_ce": (total / N).tolist(), "n_tokens": N, "n_spans": G}


def token_shuffle_null(lp: np.ndarray, span: np.ndarray, G: int, draws: int,
                       seed: int) -> dict:
    """The token-shuffle null (module doc): per draw, mean span KL and the overall
    (Bayes - oracle) per token, ``sum_s (mixture_s - oracle_s) / N``."""
    K, N = lp.shape
    rng = np.random.default_rng(seed)
    kl, gap = [], []
    for _ in range(draws):
        perm = np.argsort(rng.random((K, N)), axis=0)
        rd = selection_reads(np.take_along_axis(lp, perm, axis=0), span, G)
        kl.append(float(rd["kl"].mean()))
        gap.append(float((rd["mix_span"] - rd["oracle_span"]).sum() / N))
    return {"draws": draws, "kl": kl, "bayes_minus_oracle": gap,
            "kl_mean": float(np.mean(kl)), "bayes_minus_oracle_mean": float(np.mean(gap))}


def check_identities(rd: dict, span: np.ndarray, today: np.ndarray | None,
                     span_tol: float) -> dict:
    """RAISE unless the per-span identities of the module doc hold; return the largest
    deviations. ``today [N]`` (optional) the Bayes read, whose span sums must equal the
    span mixture."""
    K = rd["S"].shape[0]
    mix, orc = rd["mix_span"], rd["oracle_span"]
    tiny = 1e-9 * np.maximum(1.0, np.abs(mix))
    if (orc > mix + tiny).any():
        raise RuntimeError("oracle NLL above the span mixture on some span")
    if (mix > orc + math.log(K) + tiny).any():
        raise RuntimeError("span mixture above oracle + log K on some span")
    gap_dev = float(np.abs((mix - orc) - np.log(K * rd["c_max"])).max())
    if gap_dev > 1e-9 * max(1.0, float(np.abs(mix).max())):
        raise RuntimeError(f"mixture - oracle != log(K c_max): max dev {gap_dev:.3g}")
    if rd["fixed"].sum() < rd["oracle"].sum() - 1e-9 * abs(rd["oracle"].sum()):
        raise RuntimeError("best fixed rollout beats the per-span oracle")
    osum = np.bincount(span, weights=rd["oracle"], minlength=rd["S"].shape[1])
    if not np.allclose(osum, orc, rtol=1e-12, atol=1e-9):
        raise RuntimeError("the oracle's per-token NLL does not sum to -max_k S_k per span")
    out = {"gap_identity_max_dev": gap_dev}
    if today is not None:
        bsum = np.bincount(span, weights=today, minlength=rd["S"].shape[1])
        dev = float(np.abs(bsum - mix).max())
        if dev > span_tol:
            raise RuntimeError(f"the Bayes read's span sums differ from the span mixture "
                               f"by {dev:.3g} (> {span_tol})")
        out["bayes_span_sum_vs_mixture_max_abs_dev"] = dev
    return out


def _span_ci(x: np.ndarray, sblocks: np.ndarray, nb: int, sd: int) -> dict:
    """Mean over spans of ``x [G]`` with the block bootstrap, one unit per span."""
    r = _ci(x.astype(np.float64), None, sblocks, None, nb, sd)
    r["n_spans"] = r.pop("n_tokens")
    return r


def _fmt_span(d: dict) -> str:
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] n={d['n_spans']} spans"


def _deltas(a: np.ndarray, b: np.ndarray, off: np.ndarray, blocks: np.ndarray, nb: int,
            sd: int) -> dict:
    """Per-token mean(a) - mean(b), overall, at offset 0 and in every bin."""
    res = {"overall": _ci(a, b, blocks, None, nb, sd), "by_bin": {}}
    for lo, hi in BINS:
        res["by_bin"][f"{lo}-{hi}"] = _ci(a, b, blocks, (off >= lo) & (off <= hi), nb, sd)
    res["by_bin"]["first_span_or_dump"] = _ci(a, b, blocks, off < 0, nb, sd)
    return res


def summarise(rd: dict, today: np.ndarray, off: np.ndarray, blocks: np.ndarray,
              sblocks: np.ndarray, nb: int, sd: int) -> dict:
    K = rd["S"].shape[0]
    N, G = rd["n_tokens"], rd["n_spans"]
    cr = {"kl": _span_ci(rd["kl"], sblocks, nb, sd),
          "entropy": _span_ci(rd["entropy"], sblocks, nb, sd),
          "argmax_share": (np.bincount(rd["argmax"], minlength=K) / G).tolist(),
          "mean_log_K_cmax": float(np.log(K * rd["c_max"]).mean())}
    for t in CREDIT_CUTS:
        cr[f"frac_cmax_gt_{t}"] = _span_ci((rd["c_max"] > t).astype(np.float64), sblocks,
                                           nb, sd)
    ce = {"bayes": float(today.mean()), "span_mixture": float(rd["mix_span"].sum() / N),
          "oracle": float(rd["oracle"].mean()), "best_fixed": float(rd["fixed"].mean()),
          "best_fixed_k": rd["k_fixed"], "single": rd["single_ce"],
          "mean_single": float(np.mean(rd["single_ce"]))}
    return {"n_tokens": N, "n_spans": G, "tokens_per_span": N / G,
            "ceiling_bound": math.log(K) * G / N, "credit": cr, "ce": ce,
            "bayes_minus_oracle": _deltas(today, rd["oracle"], off, blocks, nb, sd),
            "fixed_minus_bayes": _deltas(rd["fixed"], today, off, blocks, nb, sd),
            "fixed_minus_oracle": _deltas(rd["fixed"], rd["oracle"], off, blocks, nb, sd)}


def argmax_agreement(am_a: np.ndarray, am_b: np.ndarray, cmax_a: np.ndarray,
                     cmax_b: np.ndarray, K: int, sblocks: np.ndarray, nb: int,
                     sd: int) -> dict:
    """P(argmax at one depth == argmax at the other), its chance level from the marginals,
    and the same on the spans decisive at both depths (c_max > 0.9)."""
    agree = (am_a == am_b).astype(np.float64)
    pa = np.bincount(am_a, minlength=K) / am_a.shape[0]
    pb = np.bincount(am_b, minlength=K) / am_b.shape[0]
    dec = (cmax_a > 0.9) & (cmax_b > 0.9)
    return {"agree": _span_ci(agree, sblocks, nb, sd), "chance": float((pa * pb).sum()),
            "decisive_n": int(dec.sum()),
            "decisive_agree": float(agree[dec].mean()) if dec.any() else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, help="lxtul_e_stage2_score.py's tokens npz")
    ap.add_argument("--label", action="append", required=True,
                    help="arm label inside the npz (repeatable), e.g. fp01_5k")
    ap.add_argument("--config", required=True, help="the arms' Hydra config (packing)")
    ap.add_argument("--depths", default="1,6", help="forced depths to read")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max |today recomputed - stored coda_mix| per token")
    ap.add_argument("--null-draws", type=int, default=5,
                    help="token-shuffle null draws (0 skips it)")
    ap.add_argument("--span-tol", type=float, default=1e-3,
                    help="max |sum of Bayes tokens - span mixture| per span (fp32 read)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    depths = [int(x) for x in a.depths.split(",")]
    from _build import build_cfg

    from morph.training.tul_setup import build_tul_runtime
    cfg = build_cfg(a.config, ["model.use_kernels=false"])
    rt = build_tul_runtime(cfg)
    eos_id = int(rt.data_cfg.rule.eos_id)
    batches, rows = val_batches(cfg, rt, a.rows, a.batch)
    if rows != a.rows:
        raise SystemExit(f"packed {rows} rows, the stored sweep scored {a.rows}")
    z = np.load(a.npz)
    nb, sd = a.n_boot, a.seed
    res: dict = {"npz": os.path.abspath(a.npz), "config": a.config, "rows": rows,
                 "depths": depths, "eos_id": eos_id, "bins": BINS, "arms": {}}
    for label in a.label:
        cidx = z[f"{label}_coda_idx"]
        blocks = _blocks(cidx)
        arm: dict = {"n_coda_tokens": int(cidx.shape[0]), "by_depth": {}}
        per_d: dict[int, dict] = {}
        span0 = first0 = None
        for d in depths:
            code = z[f"{label}_coda_code_{d}"]
            stored = z[f"{label}_coda_mix_{d}"].astype(np.float64)
            r = rescore(batches, cidx, code, [1.0], eos_id)
            dev = float(np.abs(r["today"] - stored).max())
            if dev > a.tol:
                raise RuntimeError(f"{label} d{d}: today's read recomputed differs from the "
                                   f"stored coda_mix by {dev:.3g} (> {a.tol})")
            span, first = span_index(r["row"], r["seg"])
            if span0 is None:
                span0, first0 = span, first
            elif not np.array_equal(span, span0):
                raise RuntimeError(f"{label} d{d}: the spans differ across depths")
            G = int(first.shape[0])
            rd = selection_reads(-code.astype(np.float64), span, G)
            chk = check_identities(rd, span, r["today"], a.span_tol)
            s = summarise(rd, r["today"], r["off"], blocks, blocks[first], nb, sd)
            s["checks"] = {"today_vs_stored_max_abs_dev": dev, **chk}
            if a.null_draws:
                nl = token_shuffle_null(-code.astype(np.float64), span, G, a.null_draws,
                                        sd)
                nl["observed_minus_null_kl"] = s["credit"]["kl"]["point"] - nl["kl_mean"]
                nl["observed_minus_null_gap"] = (s["bayes_minus_oracle"]["overall"]["point"]
                                                 - nl["bayes_minus_oracle_mean"])
                s["token_shuffle_null"] = nl
            per_d[d] = rd
            arm["by_depth"][d] = s
            c, ce = s["credit"], s["ce"]
            print(f"{label} d{d}: {G} spans, {s['n_tokens']} tokens "
                  f"({s['tokens_per_span']:.2f}/span), ceiling bound log4*G/N "
                  f"{s['ceiling_bound']:.4f}; today vs stored max|dev| {dev:.2e}, Bayes span "
                  f"sums vs mixture max|dev| "
                  f"{chk['bayes_span_sum_vs_mixture_max_abs_dev']:.2e}", flush=True)
            print(f"    credit: KL {_fmt_span(c['kl'])}  H {c['entropy']['point']:.4f}  "
                  f"P(cmax>0.5) {c['frac_cmax_gt_0.5']['point']:.4f}  P(cmax>0.9) "
                  f"{c['frac_cmax_gt_0.9']['point']:.4f}  argmax share "
                  f"{[round(x, 3) for x in c['argmax_share']]}", flush=True)
            print(f"    CE: Bayes {ce['bayes']:.4f}  span mixture {ce['span_mixture']:.4f}  "
                  f"oracle {ce['oracle']:.4f}  best fixed (k={ce['best_fixed_k']}) "
                  f"{ce['best_fixed']:.4f}  singles {[round(x, 4) for x in ce['single']]}",
                  flush=True)
            for key in ("bayes_minus_oracle", "fixed_minus_bayes", "fixed_minus_oracle"):
                dd = s[key]
                print(f"    {key:19s} overall {_fmt(dd['overall'])}  off0 "
                      f"{_fmt(dd['by_bin']['0-0'])}", flush=True)
            for k, v in s["bayes_minus_oracle"]["by_bin"].items():
                print(f"      Bayes-oracle bin {k:18s} {_fmt(v)}")
            if a.null_draws:
                nl = s["token_shuffle_null"]
                print(f"    token-shuffle null ({nl['draws']} draws): KL {nl['kl_mean']:.4f} "
                      f"(range {min(nl['kl']):.4f}-{max(nl['kl']):.4f}), Bayes-oracle "
                      f"{nl['bayes_minus_oracle_mean']:.4f} (range "
                      f"{min(nl['bayes_minus_oracle']):.4f}-"
                      f"{max(nl['bayes_minus_oracle']):.4f}); observed - null: KL "
                      f"{nl['observed_minus_null_kl']:+.4f}, Bayes-oracle "
                      f"{nl['observed_minus_null_gap']:+.4f}", flush=True)
        if len(depths) >= 2:
            da, db = depths[0], depths[-1]
            ra, rb = per_d[da], per_d[db]
            ag = argmax_agreement(ra["argmax"], rb["argmax"], ra["c_max"], rb["c_max"],
                                  ra["S"].shape[0], blocks[first0], nb, sd)
            arm[f"argmax_agree_d{da}_d{db}"] = ag
            print(f"[{label}] argmax d{da} == d{db}: {_fmt_span(ag['agree'])} (chance "
                  f"{ag['chance']:.4f}); on spans decisive at both (cmax>0.9, n="
                  f"{ag['decisive_n']}): {ag['decisive_agree']}", flush=True)
        res["arms"][label] = arm
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
