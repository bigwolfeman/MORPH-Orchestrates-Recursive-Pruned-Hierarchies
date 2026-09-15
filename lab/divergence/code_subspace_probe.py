"""Where in the code does the coda's content live, and does the thinker's sample reach it?

TUL-Code (docs/tul-code-spec.md) trains the thinker with an L2 flow loss on the encoder's
code z. An L2 loss is weighted by variance: if z is anisotropic (val/code_eff_rank ~60 of
2048 on the 2026-09-14 arms), the flow loss is dominated by the few high-variance
directions, and the coda's word-level content may sit in a low-amplitude tail the flow
loss barely sees. This probe measures that directly on a trained checkpoint:

1. the code's variance spectrum (PCA over the encoder codes of the valid slots, per cell);
2. `ce_tf` with the code PROJECTED onto its top-r principal components (tail zeroed) and,
   separately, onto the tail alone (head zeroed), for a ladder of r — where the CE lives
   is where the content lives;
3. the thinker's sample residual (sample − z) split into the same head/tail subspaces,
   as a fraction of z's own energy there — whether the sampler reaches the content.

    python lab/divergence/code_subspace_probe.py \
        --ckpt tul-code=tul_code=/path/step_5000.pt --rows 96 --out .../probe.json

Read-only. The encoder's forward is wrapped (never edited) for the projected passes and
restored after. Same validation stream and packer as core_depth_sweep.py.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.tul_code import code_target_valid  # noqa: E402


@torch.no_grad()
def _forward(model, inp, layout, device, code_mode, code_steps=None):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        code_mode=code_mode, code_steps=code_steps)
    assert "code_cells" in res, "the eval forward did not expose code_cells (not a code model?)"
    return res["logits"].float(), res["code_cells"].float()


def _token_ce(logits, labels, layout, device):
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    keep = (lab >= 0) & (~layout.slot_mask)
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    return float(ce[keep].sum()), int(keep.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=k=v,k=v]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--ranks", default="4,16,32,64,128,256,512,1024")
    ap.add_argument("--steps", type=int, default=8, help="sampler steps for the residual")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    ranks = [int(x) for x in a.ranks.split(",")]

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(cfg.tul.code), "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, tul_rt.model_cfg)
    model.eval()
    enc = model.tul_code_enc
    M = int(enc.m)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    batches = [(inp, labels, lay.to(device)) for inp, labels, lay, _ in batches]
    n_rows = sum(inp.shape[0] for inp, _, _ in batches)
    print(f"{label}: step {step}, {n_rows} rows, M={M}, C={model.tul_code_cell.shape[1]}",
          flush=True)

    # ── pass 1: encoder codes, sampled codes, the two baseline CEs ──────────────────
    zs, ss = [], []
    ce_enc = ce_smp = 0.0
    n_tok = 0
    for inp, labels, lay in batches:
        lg_e, cells_e = _forward(model, inp, lay, device, "encoder")
        lg_s, cells_s = _forward(model, inp, lay, device, "sampled", a.steps)
        ok = code_target_valid(lay)  # [B, S]: slot s valid AND its next span exists
        zs.append(cells_e[ok].cpu())  # [n, M, C]
        ss.append(cells_s[ok].cpu())
        c_e, n = _token_ce(lg_e, labels, lay, device)
        c_s, _ = _token_ce(lg_s, labels, lay, device)
        ce_enc += c_e
        ce_smp += c_s
        n_tok += n
    Z = torch.cat(zs)  # [N, M, C]
    S_ = torch.cat(ss)
    N, _, C = Z.shape
    ce_enc /= n_tok
    ce_smp /= n_tok
    print(f"  N={N} valid slots  ce_tf={ce_enc:.4f}  ce_sampled(k={a.steps})={ce_smp:.4f}",
          flush=True)

    # ── PCA per cell (the cells carry different embeddings, so their spectra differ) ──
    mean = Z.mean(0)  # [M, C]
    comps, spectrum = [], []
    for m in range(M):
        X = (Z[:, m] - mean[m]).double()
        _, sv, Vt = torch.linalg.svd(X, full_matrices=False)
        var = (sv ** 2) / max(N - 1, 1)
        comps.append(Vt.float())  # [K, C] rows = components
        spectrum.append(var)
    out = {"label": label, "step": step, "rows": n_rows, "n_slots": N, "M": M, "C": C,
           "ce_tf": ce_enc, "ce_sampled": ce_smp, "steps": a.steps, "cells": []}
    for m in range(M):
        var = spectrum[m]
        cum = torch.cumsum(var, 0) / var.sum()
        pr = float(var.sum() ** 2 / (var ** 2).sum())
        row = {"participation_ratio": pr, "var_total": float(var.sum()),
               "cum_var_at_rank": {r: float(cum[min(r, len(cum)) - 1]) for r in ranks}}
        # sample residual split: energy of (s − z) and of z in the head (top r) / tail
        R = (S_[:, m] - Z[:, m]).double()
        Xz = (Z[:, m] - mean[m]).double()
        Vt = comps[m].double()
        split = {}
        for r in ranks:
            H = Vt[:r]  # [r, C]
            r_head = (R @ H.T).pow(2).sum(1).mean()
            r_tail = R.pow(2).sum(1).mean() - r_head
            z_head = (Xz @ H.T).pow(2).sum(1).mean()
            z_tail = Xz.pow(2).sum(1).mean() - z_head
            split[r] = {"resid_over_z_head": float(r_head / z_head.clamp_min(1e-12)),
                        "resid_over_z_tail": float(r_tail / z_tail.clamp_min(1e-12)),
                        "z_head_frac": float(z_head / (z_head + z_tail))}
        row["sample_residual"] = split
        out["cells"].append(row)
        print(f"  cell {m}: participation ratio {pr:.1f}; cumulative variance at rank "
              + ", ".join(f"{r}:{row['cum_var_at_rank'][r]:.3f}" for r in ranks), flush=True)

    # ── pass 2: ce_tf with the code projected onto head / tail ─────────────────────
    comps_d = [c.to(device) for c in comps]
    mean_d = mean.to(device)
    orig_forward = enc.forward

    def _wrapped(part: str, r: int):
        def fwd(xs, layout):
            z, ok = orig_forward(xs, layout)  # [B, S, M, C]
            z32 = z.float()
            outz = torch.empty_like(z32)
            for m in range(M):
                H = comps_d[m][:r]  # [r, C]
                x = z32[..., m, :] - mean_d[m]
                head = (x @ H.T) @ H
                outz[..., m, :] = mean_d[m] + (head if part == "head" else x - head)
            return outz.to(z.dtype), ok
        return fwd

    ladder = {"head": {}, "tail": {}}
    for part in ("head", "tail"):
        for r in ranks:
            enc.forward = _wrapped(part, r)
            tot, n = 0.0, 0
            for inp, labels, lay in batches:
                lg, _ = _forward(model, inp, lay, device, "encoder")
                c, k = _token_ce(lg, labels, lay, device)
                tot += c
                n += k
            ladder[part][r] = tot / n
            print(f"  ce_tf with the {part} of rank {r:>5}: {tot / n:.4f}", flush=True)
    enc.forward = orig_forward
    # sanity: full-rank head must reproduce the baseline
    enc.forward = _wrapped("head", C)
    tot, n = 0.0, 0
    for inp, labels, lay in batches:
        lg, _ = _forward(model, inp, lay, device, "encoder")
        c, k = _token_ce(lg, labels, lay, device)
        tot += c
        n += k
    enc.forward = orig_forward
    out["ce_tf_fullrank_check"] = tot / n
    print(f"  full-rank check: {tot / n:.4f} (baseline {ce_enc:.4f})", flush=True)
    out["ce_ladder"] = ladder
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
