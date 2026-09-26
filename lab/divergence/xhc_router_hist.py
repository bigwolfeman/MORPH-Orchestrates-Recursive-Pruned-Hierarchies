"""Plan C router probe: per-pass stream-use histogram of the xHC slot loop on ONE val batch.

Usage (from the worktree root, under the GPU lock):
  PYTHONPATH=.:lab/divergence python lab/divergence/xhc_router_hist.py \
      CONFIG CKPT [BATCH] [DEPTH] [DEVICE]

Loads the checkpoint the way the lab probes do (`_capture_lab.load_model_and_batch`: eager
kernels, compile off), runs ONE eval forward at a forced per-slot depth, and records every
`XHCResidual.route` call. Eval calls each of the 2 * n_core residuals once per pass, in
block order, so call c belongs to pass c // (2 * n_core). Reports, per pass, the share of
valid (slot, rollout) positions whose active set contains each stream, pooled over the
residuals, and the share of positions whose ROUTED pair differs from pass 0's.
"""
from __future__ import annotations

import sys

import torch

from _capture_lab import load_model_and_batch

config, ckpt = sys.argv[1], sys.argv[2]
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 2
depth = int(sys.argv[4]) if len(sys.argv) > 4 else 6
device = sys.argv[5] if len(sys.argv) > 5 else "cuda"

model, x, y, layout, step = load_model_and_batch(config, ckpt, batch, device)
model.eval()
root = getattr(model, "_orig_mod", model)
mods = [mm for blk in root.core for mm in (blk.mrr_attn, blk.mrr_mlp)]
assert all(type(mm).__name__ == "XHCResidual" for mm in mods), "not an xHC model"
log: list[torch.Tensor] = []
for mm in mods:
    orig = mm.route

    def _w(h, fixed_route=None, _o=orig):
        r = _o(h, fixed_route)
        log.append(r[0].detach())
        return r
    mm.route = _w

depths = torch.full(layout.slot_index.shape, depth, dtype=torch.long, device=x.device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
    model(x, slot_layout=layout, slot_depths=depths)
n_mod = len(mods)
assert len(log) == depth * n_mod, (len(log), depth, n_mod)
N = mods[0].n
k_enum = root._code_enum_k or 1
valid = layout.slot_valid.repeat(k_enum, 1)
nv = int(valid.sum())
print(f"step {step}  config {config}  valid positions {nv} (x{n_mod} residuals)  depth {depth}")
print("pass | share of (position, residual) whose active set holds stream i, i = 0..15 "
      "| routed pair != pass 0")
base = [log[j] for j in range(n_mod)]
for t in range(depth):
    calls = log[t * n_mod:(t + 1) * n_mod]
    counts = torch.zeros(N)
    moved = 0
    for j, idx in enumerate(calls):
        sel = idx[valid]                                          # [nv, k]
        counts += torch.bincount(sel.flatten().cpu(), minlength=N).float()
        a = sel[:, mods[j].m:].sort(-1).values
        b = base[j][valid][:, mods[j].m:].sort(-1).values
        moved += int((a != b).any(-1).sum())
    share = counts / (nv * n_mod)
    print(f"  t{t} | " + " ".join(f"{float(s):.2f}" for s in share)
          + f" | {moved / (nv * n_mod):.3f}")
# per residual at pass 0: which routed streams dominate (collapse check)
print("pass-0 routed-stream mode per residual (stream: share):")
for j, mm in enumerate(mods):
    sel = base[j][valid][:, mm.m:].flatten().cpu()
    c = torch.bincount(sel, minlength=N).float() / sel.numel()
    top = torch.topk(c, 3)
    print(f"  {mm.route_key:12s} " + "  ".join(f"{int(i)}:{float(v):.2f}"
                                              for v, i in zip(top.values, top.indices)))
