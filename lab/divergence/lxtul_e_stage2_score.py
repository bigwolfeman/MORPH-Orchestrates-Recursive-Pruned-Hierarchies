"""LXTUL-E Stage 2: the width and depth readings of one-factor arms on top of `lxtul-e4`,
each paired against the reference arm (e4) on the same packed validation rows.

Arms (``morph/configs/``): ``tul_slot_spandec_strict_e4j4.yaml`` (the parallel head's target
cut to the first 4 tokens of the next span, ``tul.spandec_parallel_span_cap: 4``) and
``tul_slot_spandec_strict_e4probe.yaml`` (the head reads the exit state detached,
``tul.spandec_parallel_detach: true``). Why: the Stage 1 filing,
``lab/experiments/failures/2026-09-24-lxtul-e-stage1.md``. Design note:
``.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md``.

TERMS. Every term of ``lxtul_e_stage1_score.py``'s docstring (depth d, head token, par CE,
coda token, coda CE, width gain, exit sep, block) carries over unchanged; this script
imports that scorer's forward, its per-arm loop and its paired bootstrap, and adds:

  head offset    j, the position of a head token inside its span (0 = the span's first
                 token). A capped head (J = c) grades offsets 0 .. c-1 only.
  shared tokens  the head tokens two arms BOTH grade, joined by stream index
                 (``head_idx``). For j4 vs e4 that is e4's head tokens at offsets 0-3; for
                 the probe it is every head token. Every par comparison ACROSS arms is on
                 the shared tokens only; a par CE of one arm is on its own tokens.

READINGS (95 % CI, paired block bootstrap where a difference is paired):

  per arm, per depth   par mix CE, coda mix CE, coda and par width gain of the best single
                       code (one code minus the mixture), exit separation abs / rel;
  per arm              par K1-K6 (own head tokens), coda K1-K6;
  per arm vs ref       coda mix @6 (arm - ref); par mix @6 on the shared tokens (arm - ref);
                       both arms' par K1-K6 on the shared tokens, and their paired
                       difference (arm K1-K6 - ref K1-K6);
  per arm vs ruler     coda mix @6 (arm - ruler), when ``--ruler`` is given.

SELF-CHECK. ``score_arm`` compares the offline recomputation with each labelled forward's own
``par_ce`` and ``ce_tokens`` per batch and RAISES above ``--tol``.

Usage (GPU):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/lxtul_e_stage2_score.py \\
      --arm e4=tul_slot_spandec_strict_e4=checkpoints/morph/lxtul-e4/step_5000.pt \\
      --arm e4j4=tul_slot_spandec_strict_e4j4=checkpoints/morph/lxtul-e4j4/step_5000.pt \\
      --arm e4probe=tul_slot_spandec_strict_e4probe=checkpoints/morph/lxtul-e4probe/step_5000.pt \\
      --ref e4 --ruler tul_slot_spandec_strict=checkpoints/morph/slot-spandec-strict/step_5000.pt \\
      --rows 480 --batch 3 --out .../stage2.json
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

from _build import parse_ckpt_spec  # noqa: E402  (THE LABEL=CONFIG=PATH[=ovr] parser)
from lxtul_e_stage1_score import (  # noqa: E402  (ONE copy of the Stage 1 instrument)
    N_BOOT, _blocks, _ci, _fmt, abs_path, load_arm, score_arm, val_batches)
from sweep_score import BLOCK  # noqa: E402


def _per_depth(p: dict, depths: list[int], cblk: np.ndarray, hblk: np.ndarray, nb: int,
               sd: int) -> dict:
    """One arm's by-depth readings on its own tokens."""
    out = {}
    for d in depths:
        r = p[d]
        row = {"par_mix": _ci(r["par_mix"], None, hblk, None, nb, sd),
               "coda_mix": _ci(r["coda_mix"], None, cblk, None, nb, sd)}
        if "coda_code" in r:
            bc = int(np.argmin(r["coda_code"].mean(1)))
            row["coda_code"] = [float(x) for x in r["coda_code"].mean(1)]
            row["coda_width_gain_best"] = _ci(r["coda_code"][bc], r["coda_mix"], cblk, None,
                                              nb, sd) | {"code": bc}
        if "par_code" in r:
            bp = int(np.argmin(r["par_code"].mean(1)))
            row["par_code"] = [float(x) for x in r["par_code"].mean(1)]
            row["par_width_gain_best"] = _ci(r["par_code"][bp], r["par_mix"], hblk, None,
                                             nb, sd) | {"code": bp}
        if "sep_abs" in r:
            row["exit_sep_abs"] = r["sep_abs"]
            row["exit_sep_rel"] = r["sep_rel"]
        out[d] = row
    return out


def _shared(arm: dict, ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """Indices into ``arm`` and ``ref`` head arrays of the tokens both grade, joined by
    stream index (unique per head token), in ``ref`` order."""
    ai, ri = arm["head_idx"], ref["head_idx"]
    if np.unique(ai).shape[0] != ai.shape[0] or np.unique(ri).shape[0] != ri.shape[0]:
        raise RuntimeError("head_idx is not unique within an arm: the join would be wrong")
    _common, ia, ir = np.intersect1d(ai, ri, assume_unique=True, return_indices=True)
    order = np.argsort(ir)
    return ia[order], ir[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=ovr,...] (repeatable; one must be --ref)")
    ap.add_argument("--ref", required=True, help="the LABEL of the reference arm (e4)")
    ap.add_argument("--ruler", default=None, help="CONFIG=PATH of the ruler (optional)")
    ap.add_argument("--depths", default="1,2,3,4,5,6")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ovr", action="append", default=[],
                    help="extra Hydra override for every arm (repeatable). A CPU run needs "
                         "model.tg_scoped_kernels=false and model.hc_use_kernel=false.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    if 1 not in depths or 6 not in depths:
        raise SystemExit("--depths must include 1 and 6 (K1-K6 and the @6 pairings)")
    arms = [parse_ckpt_spec(s) for s in a.arm]
    labels = [x[0] for x in arms]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate --arm labels: {labels}")
    if a.ref not in labels:
        raise SystemExit(f"--ref {a.ref!r} is not an --arm label ({labels})")
    arms.sort(key=lambda x: x[0] != a.ref)                   # the reference first
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()

    per: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    batches = rows = None
    for label, config, path, ovr in arms:
        m, step, cfg, rt = load_arm(config, path, a.device, [*a.ovr, *ovr])
        if m.tul_spandec_par is None:
            raise SystemExit(f"--arm {label}: no parallel head (tul.spandec_parallel)")
        if batches is None:
            batches, rows = val_batches(cfg, rt, a.rows, a.batch)
        tc = m.cfg.tul
        meta[label] = {"config": config, "path": abs_path(path), "overrides": ovr,
                       "step": step, "K": max(int(m._code_enum_k), 1),
                       "head_J": int(m.tul_spandec_par.max_tokens),
                       "span_cap": int(tc.spandec_parallel_span_cap),
                       "detach": bool(tc.spandec_parallel_detach)}
        print(f"{label} {path} (step {step}, J={meta[label]['head_J']}, "
              f"K={meta[label]['K']})", flush=True)
        r = score_arm(m, batches, depths, a.device, a.tol)
        per[label], meta[label]["self_check_max_abs_dev"] = r["per"], r["devs"]
        del m
        if a.device == "cuda":
            torch.cuda.empty_cache()
    rr = None
    if a.ruler:
        _l, rcfg, rpath, rovr = parse_ckpt_spec(f"ruler={a.ruler}")
        mr, sr, _cr, _rtr = load_arm(rcfg, rpath, a.device, [*a.ovr, *rovr])
        print(f"ruler {rpath} (step {sr})", flush=True)
        rr = score_arm(mr, batches, [6], a.device, a.tol)
        meta["ruler"] = {"config": rcfg, "path": rpath, "step": sr,
                         "self_check_max_abs_dev": rr["devs"]}
        del mr

    nb, sd = a.n_boot, a.seed
    ref = per[a.ref]
    coda_idx = ref[1]["coda_idx"]
    for label, p in per.items():
        for d in depths:
            if not np.array_equal(p[d]["coda_idx"], coda_idx):
                raise RuntimeError(f"{label} depth {d}: the coda token set differs from the "
                                   f"reference's (different rows or labels)")
            if not np.array_equal(p[d]["head_idx"], p[1]["head_idx"]):
                raise RuntimeError(f"{label} depth {d}: the head token set moved with depth")
    cblk = _blocks(coda_idx)
    res: dict = {"rows": rows, "batch": a.batch, "depths": depths, "ref": a.ref,
                 "arms": meta, "block_tokens": BLOCK,
                 "n_coda_tokens": int(coda_idx.shape[0]), "per_arm": {}, "vs_ref": {}}
    for label, p in per.items():
        hblk = _blocks(p[1]["head_first"])
        res["per_arm"][label] = {
            "n_head_tokens": int(p[1]["head_idx"].shape[0]),
            "head_offsets": sorted(int(x) for x in np.unique(p[1]["head_off"])),
            "by_depth": _per_depth(p, depths, cblk, hblk, nb, sd),
            "par_K1-K6": _ci(p[1]["par_mix"], p[6]["par_mix"], hblk, None, nb, sd),
            "coda_K1-K6": _ci(p[1]["coda_mix"], p[6]["coda_mix"], cblk, None, nb, sd),
        }
        if rr is not None:
            if not np.array_equal(rr["per"][6]["coda_idx"], coda_idx):
                raise RuntimeError("the ruler's coda token set differs from the reference's")
            res["per_arm"][label]["coda_mix@6 - ruler"] = _ci(
                p[6]["coda_mix"], rr["per"][6]["coda_mix"], cblk, None, nb, sd)
        if label == a.ref:
            continue
        ia, ir = _shared(p[1], ref[1])
        sblk = _blocks(ref[1]["head_first"][ir])
        offs = sorted(int(x) for x in np.unique(ref[1]["head_off"][ir]))
        if not np.array_equal(p[1]["head_off"][ia], ref[1]["head_off"][ir]):
            raise RuntimeError(f"{label}: a shared head token sits at a different offset")
        k_arm = p[1]["par_mix"][ia] - p[6]["par_mix"][ia]
        k_ref = ref[1]["par_mix"][ir] - ref[6]["par_mix"][ir]
        res["vs_ref"][label] = {
            "n_shared_head_tokens": int(ia.shape[0]),
            "shared_offsets": offs,
            "coda_mix@6 (arm - ref)": _ci(p[6]["coda_mix"], ref[6]["coda_mix"], cblk, None,
                                          nb, sd),
            "par_mix@6 shared (arm - ref)": _ci(p[6]["par_mix"][ia], ref[6]["par_mix"][ir],
                                                sblk, None, nb, sd),
            "par_mix@1 shared (arm - ref)": _ci(p[1]["par_mix"][ia], ref[1]["par_mix"][ir],
                                                sblk, None, nb, sd),
            "arm par K1-K6 shared": _ci(k_arm, None, sblk, None, nb, sd),
            "ref par K1-K6 shared": _ci(k_ref, None, sblk, None, nb, sd),
            "par K1-K6 shared (arm - ref)": _ci(k_arm, k_ref, sblk, None, nb, sd),
        }
    res["wall_s"] = round(time.time() - t0, 1)

    npz = a.out.rsplit(".", 1)[0] + ".tokens.npz"
    save: dict[str, np.ndarray] = {}
    for label, p in per.items():
        save[f"{label}_coda_idx"] = p[1]["coda_idx"].astype(np.int64)
        save[f"{label}_head_idx"] = p[1]["head_idx"].astype(np.int64)
        save[f"{label}_head_first"] = p[1]["head_first"].astype(np.int64)
        save[f"{label}_head_off"] = p[1]["head_off"].astype(np.int16)
        for d in depths:
            for key in ("coda_mix", "coda_code", "par_mix", "par_code"):
                if key in p[d]:
                    save[f"{label}_{key}_{d}"] = p[d][key]
        if "sep_abs" in p[depths[0]]:
            save[f"{label}_exit_sep_abs"] = np.array([p[d]["sep_abs"] for d in depths])
            save[f"{label}_exit_sep_rel"] = np.array([p[d]["sep_rel"] for d in depths])
    save["depths"] = np.array(depths, dtype=np.int16)
    if rr is not None:
        save["ruler_coda_mix_6"] = rr["per"][6]["coda_mix"]
    np.savez_compressed(npz, **save)
    res["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)

    print(f"Stage 2: rows {rows}, {res['n_coda_tokens']} coda tokens, ref {a.ref}")
    for label, r in res["per_arm"].items():
        print(f"  [{label}] {r['n_head_tokens']} head tokens, offsets "
              f"{r['head_offsets'][0]}..{r['head_offsets'][-1]}")
        for d in (1, 6):
            bd = r["by_depth"][d]
            extra = ""
            if "par_width_gain_best" in bd:
                extra = (f"  par gain {bd['par_width_gain_best']['point']:+.4f}"
                         f"  coda gain {bd['coda_width_gain_best']['point']:+.4f}")
            if "exit_sep_abs" in bd:
                extra += f"  sep {bd['exit_sep_abs']:.4f}/{bd['exit_sep_rel']:.4f}"
            print(f"    d{d}: par {bd['par_mix']['point']:.4f}  coda "
                  f"{bd['coda_mix']['point']:.4f}{extra}")
        for key in ("par_K1-K6", "coda_K1-K6", "coda_mix@6 - ruler"):
            if key in r:
                print(f"    {key:38s} {_fmt(r[key])}")
    for label, v in res["vs_ref"].items():
        print(f"  [{label} vs {a.ref}] {v['n_shared_head_tokens']} shared head tokens, "
              f"offsets {v['shared_offsets'][0]}..{v['shared_offsets'][-1]}")
        for key, val in v.items():
            if isinstance(val, dict):
                print(f"    {key:38s} {_fmt(val)}")
    print(f"  self-check max |dev|: "
          f"{ {k: m.get('self_check_max_abs_dev') for k, m in meta.items()} }")
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
