"""LX-Carry Stage 0: re-score an LXTUL-E arm's STORED per-rollout coda log-probs under a
fixed-share carry across spans, instead of the per-span uniform restart. CPU only.

Design: ``/home/wolfe/morph-scratch/tulv2/opus.md`` D2 "LX-Carry" (Stage 0), the same idea
as gpt.md B1 and fable.md V2-B. Prereg: ``lab/experiments/planned/*-lx-carry-stage0.md``.
The math is ``morph/model/rollout_mixture.py`` (THE CARRIED READ); this script only feeds
it the stored arrays of ``lxtul_e_stage2_score.py``.

TERMS (one meaning each):

  coda token     a position the Stage 1/2 scorer scored (``lxtul_e_stage1_score.py``): a
                 token position with a label; ``<label>_coda_idx`` holds its stream index.
  rollout lp     ``-<label>_coda_code_<d>[k]``: rollout k's log-prob of the coda token's
                 label at forced depth d, as the labelled forward computed it.
  run            a contiguous run of one ``bag_id`` on a packed row (a span's tokens and
                 its slot's cells, ``rollout_mixture.span_segment_start``): the unit the
                 deployed read restarts at.
  eps            the fixed-share rate: ``pi_{s+1} = (1 - eps) alpha_s + eps / K``. eps = 1
                 is today's read (uniform at every run); it is ASSERTED bit-identical to
                 ``MORPHTransformer._enum_position_nll`` on the same arrays.
  document reset a run that holds an EOS input token ends a document (EOS always ends a
                 span, ``BoundaryRule.cut``), so the next run starts at uniform. Each packed
                 row also starts at uniform: the model has no memory across rows.
  offset         a coda token's position in its span (``_earning.offsets_from_layout``;
                 -1 for the first span and the dump bin); bins ``_earning.BINS``.
  fit / score    rows [0, --fit-rows) choose eps* (lowest mean coda CE at --fit-depth);
                 rows [--fit-rows, --rows) report it. Every CI is on the score rows only.
  block          stream index // 1,024, the paired bootstrap unit (``sweep_score.BLOCK``).

READINGS (95 % CI, paired block bootstrap, 2,000 resamples, seed 0):

  per depth, per eps, per half   mean coda CE; CE(eps) - CE(1) overall and at offset 0;
  at eps*, score half            CE(eps*) - CE(1) in every offset bin;
                                 K1-K6 carried vs today (the carry held equal at both
                                 depths) and the paired change in K1-K6;
  diagnostics (depth --fit-depth) mean normalised entropy of the carried prior at the
                                 runs' starts; the plug-in mutual information between
                                 consecutive runs' best rollout (argmax_k S_k) inside one
                                 document, with a 200-shuffle null (fable.md V2-B's
                                 pre-check).

SELF-CHECKS (each RAISES): the rebuilt rows' scored positions are exactly the stored
``coda_idx``; today's read recomputed here matches the stored ``coda_mix`` (max |dev| <=
``--tol``); eps = 1 is bit-identical to today's read; every row's carried NLL sums to the
switching model's log-likelihood computed by an independent per-run loop (rel. 1e-9).

Usage (CPU, the UPS rule while the GPU trains):
  PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    nice -n 19 taskset -c 0 python lab/divergence/lx_carry_stage0.py \\
    --npz /home/wolfe/morph-scratch/lxtul-10k/score.tokens.npz \\
    --label fp01_5k --label fp01_10k --config tul_slot_spandec_strict_e4probe_fp01 \\
    --out .../carry_stage0.json
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

from _earning import BINS, offsets_from_layout  # noqa: E402
from lxtul_e_stage1_score import N_BOOT, _blocks, _ci, _fmt, val_batches  # noqa: E402

from morph.model.rollout_mixture import (  # noqa: E402
    carried_position_nll, evidence_labels, fixed_share_log_prior, segment_index,
    span_segment_start)
from morph.model.transformer import MORPHTransformer  # noqa: E402

EPS_DEFAULT = "1,0.5,0.2,0.05,0.01"


def _scored(labels: torch.Tensor, layout) -> torch.Tensor:
    """The scorer's weighted positions on a model with ``emit_weight`` 0 and
    ``plast_weight`` 1 (``_tul_half_weights``): token positions with a label."""
    return (~layout.slot_mask) & (labels >= 0)


def switching_loglik(lp: np.ndarray, seg: np.ndarray, scored: np.ndarray,
                     reset: np.ndarray, eps: float) -> float:
    """One row's fixed-share log-likelihood by a plain per-run loop, the independent
    reference for the vectorised read. ``lp [K, L]``, ``seg / scored / reset [L]``."""
    K = lp.shape[0]
    log_pi = np.full(K, -math.log(K))
    total = 0.0
    for g in range(int(seg.max()) + 1):
        sel = (seg == g) & scored
        if not sel.any():
            continue
        S = lp[:, sel].astype(np.float64).sum(1)
        a = log_pi + S
        m = a.max()
        lse = m + math.log(np.exp(a - m).sum())
        total += lse
        alpha = np.exp(a - lse)
        log_pi = np.log((1.0 - eps) * alpha + eps / K)
        if reset[seg == g].any():
            log_pi = np.full(K, -math.log(K))
    return total


def rescore(batches: list, coda_idx: np.ndarray, coda_code: np.ndarray,
            eps_list: list[float], eos_id: int, check_rows: bool = True) -> dict:
    """Per coda token (stored order): today's NLL, the carried NLL per eps, its row, its run
    index within the row (``seg``; ``(row, seg)`` names its span), offset, and per run-start
    diagnostics. ``batches`` are ``val_batches`` output
    ``[(inp, labels, layout, idx)]``; ``coda_code [K, N]`` the stored ``-lp``."""
    if 1.0 not in eps_list:
        raise ValueError("eps 1 (today's read) must be in the list: it is the reference")
    if not np.all(np.diff(coda_idx) > 0):
        raise RuntimeError("coda_idx is not strictly increasing: the stream map is ambiguous")
    K, N = coda_code.shape
    out = {"today": np.empty(N), "row": np.empty(N, dtype=np.int64),
           "seg": np.empty(N, dtype=np.int64), "off": np.empty(N, dtype=np.int64),
           "eps": {e: np.empty(N) for e in eps_list},
           "prior_entropy": {e: [] for e in eps_list},
           "run_best": [], "n_resets": 0}
    seen = 0
    row0 = 0
    for inp, labels, layout, idx in batches:
        B, L = labels.shape
        sc = _scored(labels, layout)
        sidx = idx[sc].numpy()
        pos = np.searchsorted(coda_idx, sidx)
        if (pos >= N).any() or not np.array_equal(coda_idx[np.minimum(pos, N - 1)], sidx):
            raise RuntimeError("a rebuilt scored position is not in the stored coda_idx: "
                               "different rows, packer or labels")
        seen += sidx.shape[0]
        lp = torch.zeros(K, B, L, dtype=torch.float32)
        lp[:, sc] = -torch.from_numpy(coda_code[:, pos])
        tok = ~layout.slot_mask
        reset = tok & (inp == eos_id)
        out["n_resets"] += int(reset.sum())
        today = MORPHTransformer._enum_position_nll(lp, inp, layout)
        out["today"][pos] = today[sc].double().numpy()
        rows = torch.arange(B).view(B, 1).expand(B, L) + row0
        out["row"][pos] = rows[sc].numpy()
        off = torch.from_numpy(np.stack([offsets_from_layout(layout, b) for b in range(B)]))
        out["off"][pos] = off[sc].numpy()
        seg_start = span_segment_start(layout.bag_id)
        seg = segment_index(seg_start)
        out["seg"][pos] = seg[sc].numpy()
        _ev_l, _ev = evidence_labels(inp, layout)
        for e in eps_list:
            nll = carried_position_nll(lp, inp, layout, sc, e, reset)
            if e == 1.0 and not torch.equal(nll, today):
                raise RuntimeError("eps = 1 is not bit-identical to today's read")
            out["eps"][e][pos] = nll[sc].double().numpy()
            # the carried prior at each run's first position (its entropy / log K)
            prior = fixed_share_log_prior(lp, sc, seg_start, e, reset)          # log(K pi)
            first = seg_start == torch.arange(L).view(1, L)
            p = (prior - math.log(K)).exp()                                    # pi
            ent = -(p * (prior - math.log(K))).sum(0) / math.log(K)            # [B, L]
            has = torch.zeros(B, L, dtype=torch.bool)
            for b in range(B):
                gs = seg[b][sc[b]].unique()
                has[b] = first[b] & torch.isin(seg[b], gs)
            out["prior_entropy"][e].append(ent[has].numpy())
            if check_rows:
                lpn = lp.numpy()
                for b in range(B):
                    ref = switching_loglik(lpn[:, b], seg[b].numpy(), sc[b].numpy(),
                                           reset[b].numpy(), e)
                    got = -float(nll[b][sc[b]].double().sum())
                    if abs(got - ref) > 1e-9 * max(1.0, abs(ref)) + 1e-4:
                        raise RuntimeError(f"row {row0 + b} eps {e}: carried read sums to "
                                           f"{got:.6f}, switching likelihood {ref:.6f}")
        # consecutive runs' best rollout inside one document (MI diagnostic)
        S = torch.zeros(K, B, int(seg.max()) + 1, dtype=torch.float64)
        S.scatter_add_(-1, seg.unsqueeze(0).expand(K, -1, -1),
                       torch.where(sc.unsqueeze(0), lp.double(), torch.zeros(1).double()))
        for b in range(B):
            gs = seg[b][sc[b]].unique().tolist()
            ends_doc = {int(g) for g in seg[b][reset[b]].unique().tolist()}
            best = S[:, b].argmax(0)
            for g0, g1 in zip(gs[:-1], gs[1:]):
                if g1 == g0 + 1 and g0 not in ends_doc:
                    out["run_best"].append((int(best[g0]), int(best[g1]), row0 + b))
        row0 += B
    if check_rows and seen != N:
        raise RuntimeError(f"the rebuilt rows score {seen} tokens, the stored arrays {N}")
    out["n_rows"] = row0
    out["prior_entropy"] = {e: np.concatenate(v) for e, v in out["prior_entropy"].items()}
    return out


def plugin_mi(pairs: np.ndarray, K: int) -> float:
    """Plug-in mutual information (nats) of the integer pairs ``[n, 2]`` in ``[0, K)``."""
    j = np.zeros((K, K))
    np.add.at(j, (pairs[:, 0], pairs[:, 1]), 1.0)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum())


def summarise(r: dict, eps_list: list[float], n_fit: int, nb: int, sd: int,
              blocks: np.ndarray) -> dict:
    fit = r["row"] < n_fit
    score = ~fit
    off0 = r["off"] == 0
    res: dict = {"per_eps": {}}
    for e in eps_list:
        a = r["eps"][e]
        res["per_eps"][str(e)] = {
            "fit_ce": float(a[fit].mean()), "score_ce": float(a[score].mean()),
            "score_delta": _ci(a, r["today"], blocks, score, nb, sd),
            "score_delta_off0": _ci(a, r["today"], blocks, score & off0, nb, sd),
            "fit_delta": float(a[fit].mean() - r["today"][fit].mean()),
            "prior_entropy_mean": float(r["prior_entropy"][e].mean()),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True, help="lxtul_e_stage2_score.py's tokens npz")
    ap.add_argument("--label", action="append", required=True,
                    help="arm label inside the npz (repeatable), e.g. fp01_5k")
    ap.add_argument("--config", required=True, help="the arms' Hydra config (packing)")
    ap.add_argument("--depths", default="1,6", help="forced depths to re-score")
    ap.add_argument("--fit-depth", type=int, default=6)
    ap.add_argument("--eps", default=EPS_DEFAULT)
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--fit-rows", type=int, default=240)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max |today recomputed - stored coda_mix| per token")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    eps_list = sorted({float(x) for x in a.eps.split(",")}, reverse=True)
    depths = [int(x) for x in a.depths.split(",")]
    if a.fit_depth not in depths:
        raise SystemExit("--fit-depth must be one of --depths")
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
                 "fit_rows": a.fit_rows, "eps": eps_list, "depths": depths,
                 "fit_depth": a.fit_depth, "eos_id": eos_id, "bins": BINS, "arms": {}}
    for label in a.label:
        cidx = z[f"{label}_coda_idx"]
        blocks = _blocks(cidx)
        arm: dict = {"n_coda_tokens": int(cidx.shape[0]), "by_depth": {}}
        per_d: dict[int, dict] = {}
        for d in depths:
            code = z[f"{label}_coda_code_{d}"]
            stored = z[f"{label}_coda_mix_{d}"].astype(np.float64)
            r = rescore(batches, cidx, code, eps_list, eos_id)
            dev = float(np.abs(r["today"] - stored).max())
            if dev > a.tol:
                raise RuntimeError(f"{label} d{d}: today's read recomputed differs from the "
                                   f"stored coda_mix by {dev:.3g} (> {a.tol})")
            per_d[d] = r
            s = summarise(r, eps_list, a.fit_rows, nb, sd, blocks)
            s["today_vs_stored_max_abs_dev"] = dev
            s["n_resets"] = r["n_resets"]
            arm["by_depth"][d] = s
            print(f"{label} d{d}: today {r['today'].mean():.4f} (stored "
                  f"{stored.mean():.4f}, max|dev| {dev:.2e}), {r['n_resets']} EOS resets",
                  flush=True)
            for e in eps_list:
                pe = s["per_eps"][str(e)]
                print(f"    eps {e:<5}: fit {pe['fit_ce']:.4f} ({pe['fit_delta']:+.5f})  "
                      f"score {pe['score_ce']:.4f}  d {_fmt(pe['score_delta'])}  "
                      f"off0 {_fmt(pe['score_delta_off0'])}  H(prior) "
                      f"{pe['prior_entropy_mean']:.3f}", flush=True)
        rf = per_d[a.fit_depth]
        fit = rf["row"] < a.fit_rows
        eps_star = min(eps_list, key=lambda e: float(rf["eps"][e][fit].mean()))
        score = ~fit
        head = {"eps_star": eps_star,
                "delta_overall": _ci(rf["eps"][eps_star], rf["today"], blocks, score, nb,
                                     sd),
                "delta_off0": _ci(rf["eps"][eps_star], rf["today"], blocks,
                                  score & (rf["off"] == 0), nb, sd),
                "delta_by_bin": {}}
        for i, (lo, hi) in enumerate(BINS):
            m = score & (rf["off"] >= lo) & (rf["off"] <= hi)
            head["delta_by_bin"][f"{lo}-{hi}"] = _ci(rf["eps"][eps_star], rf["today"],
                                                     blocks, m, nb, sd)
        m = score & (rf["off"] < 0)
        head["delta_by_bin"]["first_span_or_dump"] = _ci(rf["eps"][eps_star], rf["today"],
                                                         blocks, m, nb, sd)
        if 1 in per_d and 6 in per_d:
            c1, c6 = per_d[1]["eps"][eps_star], per_d[6]["eps"][eps_star]
            t1, t6 = per_d[1]["today"], per_d[6]["today"]
            head["K1-K6_today"] = _ci(t1, t6, blocks, score, nb, sd)
            head["K1-K6_carried"] = _ci(c1, c6, blocks, score, nb, sd)
            head["K1-K6_carried - today"] = _ci(c1 - c6, t1 - t6, blocks, score, nb, sd)
        pairs = np.array([(p, q) for p, q, _r in rf["run_best"]], dtype=np.int64)
        K = int(z[f"{label}_coda_code_{a.fit_depth}"].shape[0])
        mi = plugin_mi(pairs, K)
        rng = np.random.default_rng(a.seed)
        null = [plugin_mi(np.stack([pairs[:, 0], rng.permutation(pairs[:, 1])], 1), K)
                for _ in range(200)]
        head["run_best_mi"] = {"mi_nats": mi, "null_mean": float(np.mean(null)),
                               "null_q95": float(np.quantile(null, 0.95)),
                               "n_pairs": int(pairs.shape[0]),
                               "best_share": np.bincount(pairs[:, 0], minlength=K).tolist()}
        arm["headline"] = head
        res["arms"][label] = arm
        print(f"[{label}] eps* = {eps_star} (fit on rows < {a.fit_rows}, depth "
              f"{a.fit_depth}); score rows:")
        print(f"    overall  {_fmt(head['delta_overall'])}")
        print(f"    offset 0 {_fmt(head['delta_off0'])}")
        for k, v in head["delta_by_bin"].items():
            print(f"    bin {k:18s} {_fmt(v)}")
        for k in ("K1-K6_today", "K1-K6_carried", "K1-K6_carried - today"):
            if k in head:
                print(f"    {k:24s} {_fmt(head[k])}")
        mi_r = head["run_best_mi"]
        print(f"    run-best MI {mi_r['mi_nats']:.4f} nats (null mean {mi_r['null_mean']:.4f}, "
              f"q95 {mi_r['null_q95']:.4f}, {mi_r['n_pairs']} pairs)", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
