"""torch.profiler over real l2cap training fwd+bwd steps: where the 645 ms goes.

GPU ONLY — do not run while a trainer owns the 5090 (one-trainer rule).
Loads a real checkpoint (default: the 450-step bench arm) so magnitudes and the
QAT wrappers are the shipped ones. Profiles forward+backward with the training
autocast; the optimizer step and spectral projection are NOT included (they are
timed separately by the trainer's own logs).

Usage:
  python lab/perf/profile_l2cap_step.py \
    --ckpt checkpoints/morph/bench-l2cap-asis/step_450.pt \
    [--config tul_l2] [--batch 6] [--steps 5] [--compile-attention]
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/divergence")
from _build import ROOT, build_cfg  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tul_l2")
    ap.add_argument("--ckpt", default="checkpoints/morph/bench-l2cap-asis/step_450.pt")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compile-attention", action="store_true")
    ap.add_argument("--trace-out", default="")
    a = ap.parse_args()

    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    cfg = build_cfg(a.config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    model, step = load_ckpt(cfg, a.ckpt if a.ckpt.startswith("/") else f"{ROOT}/{a.ckpt}",
                            "cuda", tul_rt.model_cfg if tul_rt else None)
    model.train()
    if a.compile_attention:
        for group in [model.prelude, model.core, model.coda]:
            dyn = True if group is model.core else None
            for layer in group:
                if hasattr(layer, "mlp"):
                    layer.mlp = torch.compile(layer.mlp, dynamic=dyn)
                if hasattr(layer, "attention"):
                    layer.attention = torch.compile(layer.attention, dynamic=dyn)

    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    rule = tul_rt.data_cfg.rule
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    buf: list[int] = []
    need = a.batch * (spec.l_total + 1)
    while len(buf) < need:
        buf.extend(next(loader)[0].reshape(-1).tolist())
    inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
    inp, labels, layout = inp.cuda(), labels.cuda(), layout.to("cuda")

    def one_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp, labels=labels, slot_layout=layout)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)

    for _ in range(a.warmup):
        one_step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(a.steps):
            one_step()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))
    total_cuda_us = sum(getattr(e, "device_time_total", 0) for e in ka)
    print(f"\nTotal CUDA time: {total_cuda_us/1e3/a.steps:.1f} ms/step over {a.steps} steps"
          f" (ckpt step {step}, batch {a.batch}, L={spec.l_total})")
    if a.trace_out:
        prof.export_chrome_trace(a.trace_out)
        print(f"trace -> {a.trace_out}")


if __name__ == "__main__":
    main()
