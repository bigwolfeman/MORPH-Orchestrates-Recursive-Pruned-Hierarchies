"""LX-Concept step 1: is the span-autoencoder teacher aligned with fp01, and how well does
it decode a span from its own code?

Prereg: ``lab/experiments/planned/2026-09-26-lx-concept-precheck.md`` (Method, step 1).
Terms: ``_concept.py``'s module docstring (slot s, next span, coded slot, span key, span
positions, span NLL).

(a) ALIGNMENT. The teacher (``tul_code_vae``, LCTUL VAE stage) and the student (fp01,
``tul_slot_spandec_strict_e4probe_fp01``) must cut the stream into the SAME spans, or a
concept label of "span s+1" names a different span than the one fp01's slot s precedes. The
audit prints the tokenizer, dataset and boundary-rule parameters of both composed configs
(``build_tul_runtime``: the rule's boundary lookup table, ``min_span``, ``span_cap``,
``eos_id``, ``fixed_stride``; the layout spec ``seq_len``, ``prefix_k``, ``max_slots``,
``slot_id``), then PACKS THE SAME VALIDATION STREAM with each runtime and compares every row
(input ids, labels, slot mask, bag ids, slot index, slot validity, stream index) and every
coded slot's next span (key and token ids, joined on the key). CPU-only (``--align-only``):
no model is built.

(b) ORACLE CE. The teacher's coda decoding every span from E's code of that span (the
``encoder`` eval mode, the trainer's ``val/ce_tf``), teacher-forced, per scored position,
on the 480 validation rows ``core_depth_sweep.py`` reads; E runs on the TEACHER's own
prelude. Twice: the clean code (the eval statistic) and the code at the training statistic
(``rmsnorm(z + 3.0 eps)``, seeded; ``--noise``). Beside it fp01's coda CE at forced depth 6
(the per-span Bayes read over its 4 rollouts, ``lxtul_e_stage1_score.score_arm``) on the same
rows, token-paired by stream index; paired block bootstrap over 1,024-token stream blocks.
Also per span: the teacher's span NLL distribution (mean, quantiles), the input to the
attribution step.

Usage:
  python lab/divergence/concept_audit.py --align-only --out .../audit_align.json     # CPU
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/concept_audit.py \\
      --rows 480 --out /home/wolfe/morph-scratch/concept/audit.json                  # GPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _concept as cc  # noqa: E402
from _build import build_cfg, parse_ckpt_spec  # noqa: E402
from lxtul_e_stage1_score import (  # noqa: E402
    _blocks, _ci, _fmt, abs_path, load_arm, score_arm, val_batches)


def runtime_params(cfg, rt) -> dict:
    """The parameters that decide a packing: tokenizer, dataset, rule, layout spec."""
    spec = rt.data_cfg.spec_for(cfg.data.seq_len)
    rule = rt.data_cfg.rule
    lut = np.asarray(rule.is_boundary)
    return {"tokenizer": str(cfg.data.tokenizer), "dataset": str(cfg.data.dataset),
            "seq_len": int(cfg.data.seq_len),
            "spec": {"seq_len": spec.seq_len, "prefix_k": spec.prefix_k,
                     "max_slots": spec.max_slots, "slot_id": spec.slot_id,
                     "l_total": spec.l_total},
            "rule": {"min_span": int(rule.min_span), "span_cap": int(rule.span_cap),
                     "eos_id": int(rule.eos_id), "fixed_stride": int(rule.fixed_stride),
                     "n_boundary_ids": int(lut.sum()), "lut_size": int(lut.size),
                     "lut_sha": __import__("hashlib").sha1(lut.tobytes()).hexdigest()}}


def align(teacher_cfg: str, student_cfg: str, rows: int, batch: int) -> dict:
    from morph.training.tul_setup import build_tul_runtime
    ct = build_cfg(teacher_cfg, ["model.use_kernels=false"])
    cs = build_cfg(student_cfg, ["model.use_kernels=false"])
    rt_t, rt_s = build_tul_runtime(ct), build_tul_runtime(cs)
    pt, ps = runtime_params(ct, rt_t), runtime_params(cs, rt_s)
    diff = {k: (pt[k], ps[k]) for k in pt if pt[k] != ps[k]}
    bt, _ = val_batches(ct, rt_t, rows, batch)
    bs, _ = val_batches(cs, rt_s, rows, batch)
    cmp = cc.compare_packing(bt, bs)
    return {"teacher_config": teacher_cfg, "student_config": student_cfg,
            "teacher_params": pt, "student_params": ps, "param_differences": diff,
            "packing": cmp,
            "aligned": (not diff and cmp["spans_only_a"] == 0 and cmp["spans_only_b"] == 0
                        and cmp["span_token_mismatches"] == 0
                        and all(v == 0 for v in cmp["row_field_mismatches"].values()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", default=f"vae={cc.TEACHER_CONFIG}={cc.TEACHER_CKPT}")
    ap.add_argument("--student",
                    default=f"fp01_10k={cc.STUDENT_CONFIG}={cc.STUDENT_DIR}/step_10000.pt")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--noise", type=float, default=3.0,
                    help="the teacher's training code_noise; 0 skips the noised read")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tl, tcfg, tpath, tovr = parse_ckpt_spec(a.teacher)
    sl, scfg, spath, sovr = parse_ckpt_spec(a.student)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res: dict = {"align": align(tcfg, scfg, a.rows, a.batch)}
    al = res["align"]
    print(f"[align] params differ: {list(al['param_differences']) or 'none'}")
    print(f"[align] packing: {json.dumps(al['packing'])}")
    print(f"[align] ALIGNED={al['aligned']}", flush=True)
    if a.align_only:
        res["wall_s"] = round(time.time() - t0, 1)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(f"wrote {a.out}")
        return
    if not al["aligned"]:
        raise SystemExit("teacher and student packings differ; the oracle pairing below "
                         "would join different spans. See the align block.")

    # ── (b) the teacher's oracle CE ──────────────────────────────────────────────────
    mt, st, cfg_t, rt_t = load_arm(tcfg, tpath, a.device, tovr)
    for p in mt.parameters():
        p.requires_grad_(False)
    batches, rows = val_batches(cfg_t, rt_t, a.rows, a.batch)
    reads = {"clean": 0.0} | ({"noise": a.noise} if a.noise > 0 else {})
    tok: dict[str, list] = {k: [] for k in reads}
    idx_all, span_nll = [], {k: [] for k in reads}
    devs = {k: 0.0 for k in reads}
    for bi, (inp, lab, lay, idx) in enumerate(batches):
        keys = cc.span_keys(lay, idx)
        for name, nz in reads.items():
            r = cc.teacher_read(mt, inp, lab, lay, a.device, noise=nz,
                                seed=a.noise_seed + bi, tol=a.tol)
            devs[name] = max(devs[name], r["dev"])
            tok[name].append(r["ce"][r["scored"]].numpy())
            span_nll[name].append(cc.span_nll(r["ce"], r["scored"], lay, keys))
            if name == "clean":
                idx_all.append(idx[r["scored"]].numpy())
    t_idx = np.concatenate(idx_all)
    t_ce = {k: np.concatenate(v) for k, v in tok.items()}
    t_span = {k: np.concatenate(v) for k, v in span_nll.items()}
    print(f"teacher {tpath} step {st}: {t_idx.size} scored tokens, "
          + "  ".join(f"{k} {v.mean():.4f}" for k, v in t_ce.items()), flush=True)
    del mt
    if a.device == "cuda":
        torch.cuda.empty_cache()

    # ── fp01's coda CE at depth 6 on the same rows ───────────────────────────────────
    ms, ss, cfg_s, rt_s = load_arm(scfg, spath, a.device, sovr)
    bs, _ = val_batches(cfg_s, rt_s, a.rows, a.batch)
    r6 = score_arm(ms, bs, [6], a.device, a.tol)["per"][6]
    s_idx, s_ce = r6["coda_idx"], r6["coda_mix"].astype(np.float64)
    common, it, is_ = np.intersect1d(t_idx, s_idx, assume_unique=True, return_indices=True)
    blk = _blocks(common)
    nb = a.n_boot
    res["oracle"] = {
        "teacher": abs_path(tpath), "teacher_step": st, "student": abs_path(spath),
        "student_step": ss, "rows": rows, "teacher_scored_tokens": int(t_idx.size),
        "student_scored_tokens": int(s_idx.size), "paired_tokens": int(common.size),
        "self_check_max_abs_dev": devs,
        "student_coda_d6": _ci(s_ce[is_], None, blk, None, nb, 0),
    }
    for k in reads:
        o = res["oracle"]
        o[f"teacher_{k}"] = _ci(t_ce[k][it], None, blk, None, nb, 0)
        o[f"teacher_{k}_minus_student_d6"] = _ci(t_ce[k][it], s_ce[is_], blk, None, nb, 0)
        sp = t_span[k]
        o[f"teacher_{k}_span_nll"] = {
            "n_spans": int(sp.size), "mean": float(sp.mean()),
            "q05_q50_q95": [float(x) for x in np.quantile(sp, [0.05, 0.5, 0.95])]}
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    o = res["oracle"]
    print(f"[oracle] {o['paired_tokens']} paired tokens ({rows} rows)")
    print(f"  fp01 coda @6                     {_fmt(o['student_coda_d6'])}")
    for k in reads:
        print(f"  teacher {k:5s} oracle CE          {_fmt(o[f'teacher_{k}'])}")
        print(f"  teacher {k:5s} - fp01 coda @6     {_fmt(o[f'teacher_{k}_minus_student_d6'])}")
        s = o[f"teacher_{k}_span_nll"]
        print(f"  teacher {k:5s} span NLL mean {s['mean']:.3f} q05/50/95 {s['q05_q50_q95']} "
              f"over {s['n_spans']} spans")
    print(f"  self-check max |dev|: {devs}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
