"""Does the TUL-Code coda read the NORM of a code as a "this is a sample" flag?

In training (transformer.py `_tul_code_core`, phase 3) a truth code reaches the coda as
`z + code_noise * N(0, I)` with NO renorm (RMS ≈ sqrt(1 + noise²) ≈ 1.118 at noise 0.5),
while a sampled code reaches it as `code_rmsnorm(z_hat)` (RMS exactly 1). At eval the
encoder code is fed bare (RMS 1, no noise). If the coda learned the norm as the flag,
then at eval EVERY code looks like a sample — which would explain ce_tf climbing through
phase 3 (0.37 → 1.25 on tul-code-20k) and ORACLE == GREEDY cuts. This probe scores the
coda's token CE with the encoder code presented at each statistic, and the sample scaled
to the truth statistic:

    enc              z                          (eval baseline)
    enc_scaled       z * sqrt(1 + noise²)       (truth norm, no noise texture)
    enc_noise        z + noise * N              (the exact training truth input)
    enc_noise_renorm rmsnorm(z + noise * N)     (noise texture at RMS 1)
    sampled          rmsnorm(z_hat)             (eval baseline)
    sampled_scaled   rmsnorm(z_hat) * sqrt(1 + noise²)

    python lab/divergence/code_norm_flag_probe.py \
        --ckpt tul-code-20k=tul_code=/path/step_20000.pt --rows 96 --out .../probe.json

Read-only: `enc.forward` and `transformer.code_rmsnorm` are wrapped per mode and restored.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

import morph.model.transformer as _tr  # noqa: E402
from morph.model.tul_code import code_rmsnorm  # noqa: E402


@torch.no_grad()
def _ce(model, inp, labels, layout, device, code_mode):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        code_mode=code_mode)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    keep = (lab >= 0) & (~layout.slot_mask)
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    return ce[keep].double().sum().item(), int(keep.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(getattr(tul_rt.model_cfg, "code", False)), \
        "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    enc = model.tul_code_enc
    noise = float(tul_rt.model_cfg.code_noise)
    scale = math.sqrt(1.0 + noise * noise)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    batches = [(inp, labels, lay.to(device)) for inp, labels, lay, _ in batches]
    n_rows = sum(inp.shape[0] for inp, _, _ in batches)
    print(f"{label}: step {step}, {n_rows} rows, code_noise={noise} → truth RMS {scale:.4f}",
          flush=True)

    orig_enc = enc.forward
    orig_norm = _tr.code_rmsnorm

    def enc_wrapped(kind):
        g = torch.Generator(device=device)
        g.manual_seed(0)

        def fwd(xs, layout):
            z, ok = orig_enc(xs, layout)
            z32 = z.float()
            if kind == "scaled":
                out = z32 * scale
            else:
                n = torch.randn(z32.shape, generator=g, device=z32.device, dtype=z32.dtype)
                out = z32 + noise * n
                if kind == "noise_renorm":
                    out = code_rmsnorm(out)
            return out.to(z.dtype), ok
        return fwd

    def norm_scaled(z, eps=1e-6):
        return orig_norm(z, eps) * scale

    modes = {
        "enc": ("encoder", None, None),
        "enc_scaled": ("encoder", "scaled", None),
        "enc_noise": ("encoder", "noise", None),
        "enc_noise_renorm": ("encoder", "noise_renorm", None),
        "sampled": ("sampled", None, None),
        "sampled_scaled": ("sampled", None, norm_scaled),
    }
    out = {"label": label, "step": step, "rows": n_rows, "code_noise": noise,
           "truth_rms": scale, "ce": {}, "per_batch": {}}
    for name, (code_mode, enc_kind, norm_fn) in modes.items():
        enc.forward = enc_wrapped(enc_kind) if enc_kind else orig_enc
        _tr.code_rmsnorm = norm_fn if norm_fn else orig_norm
        try:
            sums, cnts = [], []
            for inp, labels, lay in batches:
                s, n = _ce(model, inp, labels, lay, device, code_mode)
                sums.append(s)
                cnts.append(n)
        finally:
            enc.forward = orig_enc
            _tr.code_rmsnorm = orig_norm
        sums, cnts = np.array(sums), np.array(cnts, dtype=np.float64)
        out["ce"][name] = float(sums.sum() / cnts.sum())
        out["per_batch"][name] = {"sum": sums.tolist(), "n": cnts.tolist()}
        print(f"  {name:<18} ce={out['ce'][name]:.4f}", flush=True)
    # paired bootstrap over batches for the two decisive contrasts
    rng = np.random.default_rng(0)
    cnts = np.array(out["per_batch"]["enc"]["n"], dtype=np.float64)
    for a_, b_ in (("enc_noise", "enc"), ("enc_scaled", "enc"), ("sampled_scaled", "sampled")):
        d = np.array(out["per_batch"][a_]["sum"]) - np.array(out["per_batch"][b_]["sum"])
        bs = []
        for _ in range(500):
            idx = rng.integers(0, len(d), len(d))
            bs.append(d[idx].sum() / cnts[idx].sum())
        out[f"{a_}_minus_{b_}"] = [float(d.sum() / cnts.sum()), float(np.percentile(bs, 2.5)),
                                   float(np.percentile(bs, 97.5))]
        print(f"  {a_} − {b_}: {out[f'{a_}_minus_{b_}'][0]:+.4f} "
              f"[{out[f'{a_}_minus_{b_}'][1]:+.4f}, {out[f'{a_}_minus_{b_}'][2]:+.4f}]", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
