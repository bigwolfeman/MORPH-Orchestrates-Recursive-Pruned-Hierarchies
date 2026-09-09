"""CE vs forced depth on a PLAIN checkpoint under four loop-state inits (eval only).

(a) the prelude output (the shipped entry), (b) Gaussian noise at the prelude output's
per-element RMS, (c) zeros, (d) Gaussian noise at ``--noise-std`` per element (Parcae's
like-init scale, 0.02). Same rows for every init and depth. The 2026-09-09 reading on the
E18 plain arm: (a) 3.939 -> 3.918 (T=4) -> 3.948 (T=24); (b) 8.20 -> 6.30 (T=6) then worse;
(c) 6.36 -> 4.67 (T=8) -> 5.21 (T=24): the fixed point depends on where the loop starts.

Usage:
  python lab/divergence/core_init_probe.py --ckpt .../step_5000.pt --config notul_e18 \
      --rows 96 --out .../init_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

DEPTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24)


class _Init(nn.Module):
    def __init__(self, mode: str, std: float):
        super().__init__()
        self.mode, self.std = mode, std

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        if self.mode == "prelude":
            return e
        if self.mode == "zero":
            return torch.zeros_like(e)
        if self.mode == "noise_rms":
            rms = e.float().pow(2).mean().sqrt()
            return (torch.randn_like(e.float()) * rms).to(e.dtype)
        if self.mode == "noise_small":
            return torch.randn_like(e) * self.std
        raise ValueError(self.mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    cfg = build_cfg(a.config, ["model.use_kernels=false"])
    if build_tul_runtime(cfg) is not None:
        raise SystemExit("core_init_probe reads the PLAIN forward; pass a tul.activate_at=never config")
    model, step = load_ckpt(cfg, a.ckpt, "cuda", None)
    model.eval()
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    seq = int(cfg.data.seq_len)
    stream = stream_from_loader(loader, a.rows * (seq + 1))
    batches = pack_rows(stream, None, cfg, a.batch, True)[: -(-a.rows // a.batch)]
    orig_init, orig_depth = model.core_init, int(model.cfg.mean_depth)
    res: dict[str, dict[int, float]] = {}
    try:
        for mode in ("prelude", "noise_rms", "zero", "noise_small"):
            model.core_init = _Init(mode, a.noise_std).cuda()
            res[mode] = {}
            for d in DEPTHS:
                model.cfg.mean_depth = d
                torch.manual_seed(0)
                tot = n = 0.0
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for inp, lab, _, _ in batches:
                        lg = model(inp.cuda(), labels=None)["logits"].float()
                        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.cuda().reshape(-1),
                                               reduction="sum").item()
                        n += lab.numel()
                res[mode][d] = tot / n
            print(mode, {d: round(v, 4) for d, v in res[mode].items()}, flush=True)
    finally:
        model.core_init, model.cfg.mean_depth = orig_init, orig_depth
    json.dump({"step": step, "rows": a.rows, "noise_std": a.noise_std, "ce": res}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
