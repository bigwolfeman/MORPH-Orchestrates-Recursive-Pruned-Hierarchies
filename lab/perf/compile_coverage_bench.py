"""Compile-coverage ladder for the l2cap eager step (measured 2026-08-31, 5090).

Variants over identical fwd+bwd (batch 6, L=1152, bench-l2cap-asis/step_450,
this worktree's GATHERED _tg_slot_attention in all variants):

  (a) eager                                   623.8 ms
  (b) torch.compile(layer.mlp, layer.attention)  558.6 ms   (the shipped compile_attention)
  (c) torch.compile(whole MORPHBlock)         FAILS — Inductor SplitScan codegen
      bug (TypeError: list indices ... NoneType) in the GLA chunked cumsum
      ([B, 256, 8, 128] fp32). Upstream torch bug, not ours.
  (d) torch.compile(whole MORPHBlock) with layer.retention.forward wrapped in
      torch.compiler.disable (GLA runs the SAME eager code via graph break)
                                              339.6 ms   1.84x vs (a), 1.64x vs (b)

Correctness (eval mode, paired batch, vs eager):
  null (eager vs eager): logits BIT-IDENTICAL; grad cos 0.99998 (atomics only)
  (b): logits max|d| 1.92 mean 0.0078 (scale 4.9); loss |d| 6e-4;
       grad cos 0.990-0.994, rel 11-14%
  (d): logits max|d| 2.00 mean 0.0083; loss |d| 4e-4; grad cos 0.987-0.994
  => (d) adds NO drift class beyond the already-shipped (b).

Profile context (uncompiled step): self-CPU 820 ms vs self-CUDA 371 ms — the
eager step is launch-bound (~27k kernel launches; 75k copy_ + 56k mul calls per
5 steps). Whole-block compile collapses the glue (norms, HC residual mixing,
injections, casts); GLA stays eager pending the upstream fix or a reset_mask
fused kernel (gla.py documents why tg_restrict forbids the existing one).

Run: python lab/perf/compile_coverage_bench.py [--variant a|b|d] [--check]
GPU ONLY; one-trainer rule applies.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/divergence")
from _build import ROOT, build_cfg  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


def make_batch(cfg, tul_rt, batch):
    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    buf: list[int] = []
    while len(buf) < batch * (spec.l_total + 1):
        buf.extend(next(loader)[0].reshape(-1).tolist())
    inp, labels, layout = pack_tul_batch(buf, tul_rt.data_cfg.rule, spec, batch)
    return inp.cuda(), labels.cuda(), layout.to("cuda")


def apply_variant(model, variant):
    if variant == "a":
        return model
    for group in [model.prelude, model.core, model.coda]:
        dyn = True if group is model.core else None
        for i in range(len(group)):
            layer = group[i]
            if variant == "b":
                layer.mlp = torch.compile(layer.mlp, dynamic=dyn)
                layer.attention = torch.compile(layer.attention, dynamic=dyn)
            elif variant == "d":
                if layer.retention is not None:
                    layer.retention.forward = torch.compiler.disable(layer.retention.forward)
                group[i] = torch.compile(layer, dynamic=dyn)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tul_l2")
    ap.add_argument("--ckpt", default="checkpoints/morph/bench-l2cap-asis/step_450.pt")
    ap.add_argument("--variant", default="d", choices=["a", "b", "d"])
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--check", action="store_true",
                    help="paired eval correctness vs eager instead of timing")
    a = ap.parse_args()

    from morph.training.tul_setup import build_tul_runtime
    cfg = build_cfg(a.config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    path = a.ckpt if a.ckpt.startswith("/") else f"{ROOT}/{a.ckpt}"
    inp, labels, layout = make_batch(cfg, tul_rt, a.batch)

    if a.check:
        outs = {}
        for var in ("a", a.variant):
            model, _ = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
            model = apply_variant(model, var)
            model.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp, None, slot_layout=layout)["logits"].detach().float()
                out = model(inp, labels=labels, slot_layout=layout)
            outs[var] = (logits, float(out["loss"]))
            del model
            torch.cuda.empty_cache()
        le, se = outs["a"]
        lc, sc = outs[a.variant]
        fin = torch.isfinite(le) & torch.isfinite(lc)
        d = (le[fin] - lc[fin]).abs()
        print(f"variant {a.variant} vs eager: logits max|d| {d.max():.4f} "
              f"mean|d| {d.mean():.6f}  loss |d| {abs(se-sc):.5f}")
        return

    model, _ = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
    model = apply_variant(model, a.variant)
    model.train()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp, labels=labels, slot_layout=layout)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
        return float(out["loss"])

    for _ in range(6):
        last = step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        last = step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"variant {a.variant}: {ms:.1f} ms/fwdbwd  loss {last:.3f}")


if __name__ == "__main__":
    main()
