"""Is the code-target loop's predicted cell a conditional prediction or one constant vector?

The code-target arm (docs/tul-code-spec.md §17) regresses the slot loop's exit state onto
the frozen VAE code z = E(next span). Its exit cosine plateaued at 0.14-0.17 by step 1500
(arm A, 2026-09-17). A cosine of that size is what a CONSTANT output reaches when the codes
share a common direction, so the reading needs a baseline. On one checkpoint, over the valid
slots of the validation rows, this probe measures:

- cos(pred, z_own)                    the trained reading (val/code_target_cos);
- cos(pred, z_shuf)                   pred against another slot's code (the generic floor);
- cos(pred, zbar), cos(z, zbar)       both against the corpus-mean code;
- centred cos(pred - pbar, z - zbar)  what is left once the mean direction is removed;
- mean pairwise cos among preds vs among codes, and the effective rank of each
  (a constant predictor: pairwise ~1, rank ~1).

    python lab/divergence/code_target_mean_probe.py \
        --ckpt A=tul_code_target=/path/step_10000.pt --rows 96 --depths 1,6,16 \
        --out .../mean_probe.json

``--depths`` reads the SAME rows at each forced loop depth (`slot_mean_depth` /
`slot_max_depth`, the eval dial `core_depth_sweep.py` uses, restored afterwards). The
frozen coda's CE falls with depth (arm A at 20k: 9.142 at depth 1, 9.046 at depth 16)
while the exit cosine to the code is flat across passes, so the question is WHICH of the
two the passes move: a cell that is closer to its own code, or a cell that is simply less
confident (nearer the corpus mean, lower norm, lower pairwise cosine). That is what the
per-depth columns answer.

Read-only. Same validation stream and packer as code_subspace_probe.py; the predicted cells
come from the ordinary eval forward (``code_cells``), the codes from ``code_mode="encoder"``.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.tul_code import code_target_valid  # noqa: E402


@torch.no_grad()
def _cells(model, inp, layout, device, code_mode):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        code_mode=code_mode)
    assert "code_cells" in res, "the eval forward did not expose code_cells"
    return res["code_cells"].float()


def _eff_rank(X: torch.Tensor) -> float:
    Xc = (X - X.mean(0)).double()
    sv = torch.linalg.svdvals(Xc)
    p = sv ** 2
    p = p / p.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))


def _pairwise(X: torch.Tensor, n_pairs: int, g: torch.Generator) -> float:
    N = X.shape[0]
    i = torch.randint(0, N, (n_pairs,), generator=g)
    j = torch.randint(0, N, (n_pairs,), generator=g)
    keep = i != j
    return float(F.cosine_similarity(X[i[keep]], X[j[keep]], dim=-1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=k=v,k=v]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="", help="forced loop depths, e.g. 1,6,16 "
                                                "(default: the model's own eval depth)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(cfg.tul.code_target), "a code-target checkpoint is required"
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, tul_rt.model_cfg)
    model.eval()
    M = int(model.tul_code_enc.m)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    batches = [(inp, lay.to(device)) for inp, _labels, lay, _ in batches]
    n_rows = sum(inp.shape[0] for inp, _ in batches)

    # The codes are the FROZEN encoder's read of the prelude: the loop's depth cannot move
    # them, so they are collected once and every depth column is scored against the same Z.
    zs = [_cells(model, inp, lay, device, "encoder")[code_target_valid(lay)].cpu()
          for inp, lay in batches]
    Z = torch.cat(zs)
    N, _, C = Z.shape
    print(f"{label}: step {step}, {n_rows} rows, N={N} valid slots, M={M}, C={C}", flush=True)

    tc = model.cfg.tul
    depths = [int(x) for x in a.depths.split(",") if x.strip()] or [0]
    orig_mean, orig_max = int(tc.slot_mean_depth), int(tc.slot_max_depth)
    orig_fixed = int(getattr(tc, "slot_depth_fixed", 0))
    out = {"label": label, "step": step, "rows": n_rows, "n_slots": N, "M": M, "C": C,
           "eval_depth": orig_fixed or orig_mean, "by_depth": {}}
    P0 = None                     # the FIRST depth's cells: every later depth is read against it
    try:
        for d in depths:
            if d > 0:
                # the eval dial core_depth_sweep.py uses (restored in the finally below)
                tc.slot_mean_depth = d
                tc.slot_max_depth = max(d, orig_max)
                if orig_fixed > 0:
                    tc.slot_depth_fixed = d
            P = torch.cat([_cells(model, inp, lay, device, None)[code_target_valid(lay)].cpu()
                           for inp, lay in batches])
            if P0 is None:
                P0 = P
            out["by_depth"][str(d)] = _stats(P, Z, M, C, N, d, P0)
    finally:
        tc.slot_mean_depth, tc.slot_max_depth = orig_mean, orig_max
        if orig_fixed > 0:
            tc.slot_depth_fixed = orig_fixed
    # the flat keys the first version wrote, from the first depth column
    out["cells"] = out["by_depth"][str(depths[0])]["cells"]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


def _stats(P: torch.Tensor, Z: torch.Tensor, M: int, C: int, N: int, depth: int,
           P0: torch.Tensor) -> dict:
    """The readings for ONE set of predicted cells against the codes, and against the cells
    the FIRST depth column produced (``P0``): how far the extra passes moved the cell at
    all, and in which direction. ``cos_to_d0`` near 1 with a falling coda CE says the
    passes are polishing one vector, not choosing a different one; ``cos_delta_zbar``
    (the change's cosine to the corpus-mean code) says whether the polish is a hedge."""
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    res = {"depth": depth, "cells": []}
    for m in range(M):
        p, z = P[:, m], Z[:, m]
        pbar, zbar = p.mean(0), z.mean(0)
        r = {
            "cos_own": float(F.cosine_similarity(p, z, dim=-1).mean()),
            "cos_shuf": float(F.cosine_similarity(p, z[perm], dim=-1).mean()),
            "cos_pred_zbar": float(F.cosine_similarity(p, zbar.expand_as(p), dim=-1).mean()),
            "cos_z_zbar": float(F.cosine_similarity(z, zbar.expand_as(z), dim=-1).mean()),
            "cos_pbar_zbar": float(F.cosine_similarity(pbar, zbar, dim=0)),
            "cos_centred_own": float(F.cosine_similarity(p - pbar, z - zbar, dim=-1).mean()),
            "cos_centred_shuf": float(F.cosine_similarity(p - pbar, (z - zbar)[perm],
                                                          dim=-1).mean()),
            "pair_pred": _pairwise(p, 4000, g),
            "pair_z": _pairwise(z, 4000, g),
            "eff_rank_pred": _eff_rank(p),
            "eff_rank_z": _eff_rank(z),
            "norm_pbar_over_rms": float(pbar.norm() / p.norm(dim=-1).mean()),
            "norm_zbar_over_rms": float(zbar.norm() / z.norm(dim=-1).mean()),
            "rms_pred": float(p.pow(2).mean(-1).sqrt().mean()),
        }
        p0 = P0[:, m]
        dlt = p - p0
        r["cos_to_d0"] = float(F.cosine_similarity(p, p0, dim=-1).mean())
        r["delta_rel"] = float((dlt.norm(dim=-1) / p0.norm(dim=-1)).mean())
        _dn = dlt.norm(dim=-1) > 1e-8
        r["cos_delta_zbar"] = (float(F.cosine_similarity(dlt[_dn], zbar.expand_as(dlt)[_dn],
                                                         dim=-1).mean()) if _dn.any() else 0.0)
        r["cos_delta_z"] = (float(F.cosine_similarity(dlt[_dn], z[_dn], dim=-1).mean())
                            if _dn.any() else 0.0)
        res["cells"].append(r)
        print(f"  [depth {depth}] cell {m}: cos own {r['cos_own']:.4f}  shuf {r['cos_shuf']:.4f}  "
              f"pred·zbar {r['cos_pred_zbar']:.4f}  z·zbar {r['cos_z_zbar']:.4f}  "
              f"centred own {r['cos_centred_own']:.4f} shuf {r['cos_centred_shuf']:.4f}  "
              f"pair pred {r['pair_pred']:.4f} z {r['pair_z']:.4f}  "
              f"rank pred {r['eff_rank_pred']:.1f} z {r['eff_rank_z']:.1f}  "
              f"rms {r['rms_pred']:.3f}\n"
              f"             to-d0 {r['cos_to_d0']:.4f}  |delta|/|p| {r['delta_rel']:.4f}  "
              f"delta.zbar {r['cos_delta_zbar']:+.4f}  delta.z {r['cos_delta_z']:+.4f}",
              flush=True)
    return res


if __name__ == "__main__":
    main()
