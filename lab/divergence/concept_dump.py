"""LX-Concept step 2: E's code of every span, and fp01's slot states, keyed by span.

Prereg: ``lab/experiments/planned/2026-09-26-lx-concept-precheck.md`` (Method, step 2).
Terms: ``_concept.py``'s module docstring (coded slot, next span, span key, code, entry,
exit, exit_raw, span NLL).

Two REGIONS of one packer (the student's runtime; ``concept_audit.py`` proved the teacher's
cuts the same spans):

  val    the first ``--val-rows`` validation rows (480: the rows ``core_depth_sweep.py`` and
         the LX scorers read); stream indices as those scorers have them.
  train  ``--train-rows`` rows of the OWT stream after ``--skip-docs`` documents (default
         200,000, past what fp01 and the teacher trained on); stream indices + 10^12.

Per region, into ``<out-dir>/<region>/``:

  keys.npz              row, slot, first, last, n_tok of every coded slot (the join key)
  codes.npy             ``[n, 2048]`` fp16: E's clean code of the next span (2 cells x 1024)
  teacher_nll.npy       ``[n]`` fp32: the teacher's span NLL with every cell at E's code
  <label>/entry.npy     ``[n, 1024]`` fp16 per student checkpoint (the slot, not its span)
  <label>/exit.npy      ``[n, 1024]`` fp16
  <label>/exit_raw.npy  ``[n, 1024]`` fp16
  meta.json             the packing (config, region, rows, batch, skip), counts, checks

The teacher's rows and each student's rows are packed separately and JOINED ON THE SPAN KEY
(``_concept.join_keys``); a key present in one packing and not the other RAISES, so row
order can never mis-assign a state to a span. ~52 coded slots per row: 2,000 train rows is
~100k spans (the prereg's range), about 200 MB of codes and 600 MB of states per student.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/concept_dump.py \\
      --region val --val-rows 480 --out-dir /home/wolfe/morph-scratch/concept/dump
  ... --region train --train-rows 2000 ...
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
from _build import parse_ckpt_spec  # noqa: E402
from lxtul_e_stage1_score import abs_path, load_arm  # noqa: E402


def _gather(t: torch.Tensor, keys: dict) -> torch.Tensor:
    """``t[row, slot]`` for every key, in key order (``t`` is ``[B, S, ...]``)."""
    return t[torch.as_tensor(keys["row"]), torch.as_tensor(keys["slot"])]


def dump_teacher(spec: str, region: str, rows: int, batch: int, skip: int, device: str,
                 out: str, tol: float) -> dict:
    label, config, path, ovr = parse_ckpt_spec(spec)
    m, step, cfg, rt = load_arm(config, path, device, ovr)
    for p in m.parameters():
        p.requires_grad_(False)
    batches = cc.region_batches(cfg, rt, region, rows, batch, skip)
    keys_all, codes, nll = [], [], []
    dev = 0.0
    for bi, (inp, lab, lay, idx) in enumerate(batches):
        keys = cc.span_keys(lay, idx)
        r = cc.teacher_read(m, inp, lab, lay, device, tol=tol)
        dev = max(dev, r["dev"])
        if not bool(r["ok"][torch.as_tensor(keys["row"]), torch.as_tensor(keys["slot"])].all()):
            raise RuntimeError("E's ok mask disagrees with code_target_valid on a keyed slot")
        z = _gather(r["z"], keys)                                       # [n, M, C]
        codes.append(z.reshape(z.shape[0], -1).half().numpy())
        nll.append(cc.span_nll(r["ce"], r["scored"], lay, keys).astype(np.float32))
        keys["row"] = keys["row"] + bi * batch
        keys_all.append(keys)
        if bi % 50 == 0:
            print(f"  teacher batch {bi}/{len(batches)}", flush=True)
    keys = cc.concat_keys(keys_all)
    np.savez(os.path.join(out, "keys.npz"), **keys)
    np.save(os.path.join(out, "codes.npy"), np.concatenate(codes))
    np.save(os.path.join(out, "teacher_nll.npy"), np.concatenate(nll))
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"teacher": abs_path(path), "teacher_config": config, "teacher_step": step,
            "n_spans": int(keys["row"].size), "rows": int(sum(b[0].shape[0] for b in batches)),
            "self_check_max_abs_dev": dev,
            "teacher_nll_mean": float(np.concatenate(nll).mean())}


def dump_student(spec: str, region: str, rows: int, batch: int, skip: int, device: str,
                 out: str, depth: int, tkeys: dict) -> dict:
    label, config, path, ovr = parse_ckpt_spec(spec)
    m, step, cfg, rt = load_arm(config, path, device, ovr)
    batches = cc.region_batches(cfg, rt, region, rows, batch, skip)
    parts: dict[str, list] = {"entry": [], "exit": [], "exit_raw": []}
    keys_all = []
    for bi, (inp, lab, lay, idx) in enumerate(batches):
        keys = cc.span_keys(lay, idx)
        st = cc.student_states(m, inp, lab, lay, device, depth=depth)
        for k in parts:
            parts[k].append(_gather(st[k], keys).half().numpy())
        keys["row"] = keys["row"] + bi * batch
        keys_all.append(keys)
        if bi % 50 == 0:
            print(f"  {label} batch {bi}/{len(batches)}", flush=True)
    skeys = cc.concat_keys(keys_all)
    it, is_ = cc.join_keys(tkeys, skeys)
    if it.size != tkeys["row"].size or is_.size != skeys["row"].size:
        raise RuntimeError(f"{label}: {tkeys['row'].size} teacher spans, {skeys['row'].size} "
                           f"student spans, {it.size} joined; the packings differ")
    order = np.empty_like(is_)
    order[it] = is_                                  # student row for each teacher span
    d = os.path.join(out, label)
    os.makedirs(d, exist_ok=True)
    for k, v in parts.items():
        np.save(os.path.join(d, f"{k}.npy"), np.concatenate(v)[order])
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"student": abs_path(path), "student_config": config, "student_step": step,
            "depth": depth, "n_spans": int(skeys["row"].size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, choices=["val", "train"])
    ap.add_argument("--teacher", default=f"vae={cc.TEACHER_CONFIG}={cc.TEACHER_CKPT}")
    ap.add_argument("--student", action="append", default=None,
                    help="LABEL=CONFIG=PATH (repeatable); default fp01 steps 5000 and 10000")
    ap.add_argument("--val-rows", type=int, default=480)
    ap.add_argument("--train-rows", type=int, default=2000)
    ap.add_argument("--skip-docs", type=int, default=cc.TRAIN_SKIP_DOCS)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    students = a.student or [f"fp01_5k={cc.STUDENT_CONFIG}={cc.STUDENT_DIR}/step_5000.pt",
                             f"fp01_10k={cc.STUDENT_CONFIG}={cc.STUDENT_DIR}/step_10000.pt"]
    rows = a.val_rows if a.region == "val" else a.train_rows
    out = os.path.join(a.out_dir, a.region)
    os.makedirs(out, exist_ok=True)
    meta = {"region": a.region, "rows": rows, "batch": a.batch,
            "skip_docs": a.skip_docs if a.region == "train" else 0, "depth": a.depth}
    meta["teacher"] = dump_teacher(a.teacher, a.region, rows, a.batch, a.skip_docs, a.device,
                                   out, a.tol)
    print(f"[{a.region}] teacher: {json.dumps(meta['teacher'])}", flush=True)
    tkeys = dict(np.load(os.path.join(out, "keys.npz")))
    meta["students"] = {}
    for spec in students:
        lab = spec.split("=", 1)[0]
        meta["students"][lab] = dump_student(spec, a.region, rows, a.batch, a.skip_docs,
                                             a.device, out, a.depth, tkeys)
        print(f"[{a.region}] {lab}: {json.dumps(meta['students'][lab])}", flush=True)
    meta["wall_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"wrote {out}/ ({meta['teacher']['n_spans']} spans, {meta['wall_s']} s)")


if __name__ == "__main__":
    main()
