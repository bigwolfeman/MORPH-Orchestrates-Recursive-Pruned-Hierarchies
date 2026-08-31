"""Micro-benchmark: dense vs gathered _tg_slot_attention at the 5090 shapes.

GPU ONLY — do not run while a trainer owns the 5090 (one-trainer rule).
Correctness is already covered by tests/test_tg_slot_attention_gather.py;
this measures wall-clock and peak memory, fwd and fwd+bwd.

Usage:  python lab/perf/bench_tg_slot_attn.py [--B 6 --H 8 --S 1152 --D 64 --M 64]
"""
from __future__ import annotations

import argparse

import torch

from morph.model.attention import _tg_slot_attention


def _dense_reference(q, k, v, slot_mask, sink_logits, scale):
    B, H, S, D = q.shape
    device = q.device
    row = torch.arange(S, device=device).unsqueeze(1)
    col = torch.arange(S, device=device).unsqueeze(0)
    causal = (col <= row).unsqueeze(0)
    allow = causal if slot_mask is None else causal & slot_mask.unsqueeze(1)
    scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale
    scores = scores.masked_fill(~allow.unsqueeze(1), float("-inf"))
    sink = sink_logits.view(1, H, 1, 1).to(scores.dtype).expand(B, H, S, 1)
    scores = torch.cat([scores, sink], dim=-1)
    weights = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.einsum("bhij,bhjd->bhid", weights[..., :S], v)


def bench(fn, args, bwd: bool, iters=50, warmup=10) -> tuple[float, float]:
    q, k, v, slot_mask, sink, scale = args
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize(); start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(q, k, v, slot_mask, sink, scale)
        if bwd:
            out.square().sum().backward()
            for t in (q, k, v, sink):
                t.grad = None
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--S", type=int, default=1152)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--M", type=int, default=64)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    q, k, v = (torch.randn(a.B, a.H, a.S, a.D, device=dev, dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3))
    sink = torch.randn(a.H, device=dev, requires_grad=True)
    slot_mask = torch.zeros(a.B, a.S, dtype=torch.bool, device=dev)
    for b in range(a.B):
        slot_mask[b, torch.randperm(a.S)[: a.M]] = True
    args = (q, k, v, slot_mask, sink, a.D ** -0.5)
    print(f"shapes B={a.B} H={a.H} S={a.S} D={a.D} M={a.M}")
    for bwd in (False, True):
        tag = "fwd+bwd" if bwd else "fwd    "
        td, md = bench(_dense_reference, args, bwd)
        tg, mg = bench(_tg_slot_attention, args, bwd)
        print(f"{tag}  dense {td:7.3f} ms  peak {md:8.1f} MiB   "
              f"gathered {tg:7.3f} ms  peak {mg:8.1f} MiB   speedup {td/tg:5.2f}x")


if __name__ == "__main__":
    main()
