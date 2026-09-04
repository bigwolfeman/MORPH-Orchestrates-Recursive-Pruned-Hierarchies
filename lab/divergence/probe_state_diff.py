"""Does the Jacobian probe's extra forward + measurement change anything the next
training step sees — parameters, buffers, gradients, module attributes, or the RNG?

    python lab/divergence/probe_state_diff.py --ckpt checkpoints/morph/run/ROLL_step_40.pt \
        --config tul_cap_c1 [--batch 6] [--power 20]

Procedure on ONE batch, model in training mode, deterministic settings as the trainer:
  1. F1: forward+backward (the trainer's step). Record loss and every parameter's .grad.
  2. Snapshot every parameter, buffer, .grad, the CPU and CUDA generator states, and every
     scalar Python attribute of every module (ints, floats, bools, small tensors).
  3. Run exactly what `_jacobian_probe` runs (capture forward + measure).
  4. Snapshot again and print every name whose value changed.
  5. F3: restore the RNG to its pre-F1 state and repeat the forward; report whether the
     loss equals F1's bit for bit.

Exit 0 if nothing changed and F3 == F1; 5 otherwise (the names are the finding).
Result 2026-09-03 on smoke-cap-a/ROLL_step_40: before the fix, ONLY `rng:cuda` changed
(the measurement's dropout draws); after the fix in `train._jacobian_probe`, nothing.
"""
from __future__ import annotations

import argparse
import sys

import torch

from _capture_lab import load_model_and_batch, require_deterministic_env

from morph.training.core_jacobian import CoreJacobianProbe  # noqa: E402
from morph.training.train import _jacobian_probe  # noqa: E402


def _snapshot(model):
    root = getattr(model, "_orig_mod", model)
    snap = {}
    for n, p in root.named_parameters():
        snap[f"param:{n}"] = p.detach().clone()
        snap[f"grad:{n}"] = None if p.grad is None else p.grad.detach().clone()
    for n, b in root.named_buffers():
        snap[f"buffer:{n}"] = None if b is None else b.detach().clone()
    snap["rng:cpu"] = torch.get_rng_state().clone()
    snap["rng:cuda"] = torch.cuda.get_rng_state().clone()
    for n, m in root.named_modules():
        for k, v in vars(m).items():
            if k.startswith(("_parameters", "_buffers", "_modules")):
                continue
            if isinstance(v, (int, float, bool, str)) or v is None:
                snap[f"attr:{n}.{k}"] = v
            elif torch.is_tensor(v) and v.numel() <= 4096:
                snap[f"attr:{n}.{k}"] = v.detach().clone()
    return snap


def _diff(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape:
                out.append((k, "shape changed", None))
            elif not torch.equal(va, vb):
                out.append((k, "tensor changed", float((va.float() - vb.float()).abs().max())))
        elif torch.is_tensor(va) != torch.is_tensor(vb):
            out.append((k, "kind changed", None))
        elif va != vb:
            out.append((k, f"{va!r} -> {vb!r}", None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_cap_c1")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--power", type=int, default=20)
    a = ap.parse_args()
    require_deterministic_env()
    model, x, y, layout, step = load_model_and_batch(a.config, a.ckpt, a.batch)
    root = getattr(model, "_orig_mod", model)
    root._probe_loop = True
    root._probe_cot = True
    root._probe_rank = True

    torch.manual_seed(1234)
    rng0 = (torch.get_rng_state(), torch.cuda.get_rng_state())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=y, bag_size=0, slot_layout=layout)
    out["loss"].backward()
    loss1 = out["loss"].detach().clone()
    print(f"F1 loss {float(loss1):.10f} at step {step}")
    s0 = _snapshot(model)

    probe = CoreJacobianProbe(model, n_iter=a.power, seed=0)
    row = _jacobian_probe(model, probe, x, y, layout, 0, [0, 3])
    print("probe row:", {k: round(v, 5) for k, v in row.items() if k.startswith(("jac/sigma_t", "jac/rms_t"))})
    s1 = _snapshot(model)
    changed = [c for c in _diff(s0, s1) if c[0] != "attr:._jac_capture"]   # the capture list is set to None by design
    print(f"changed after probe: {len(changed)}")
    for k, what, mag in changed[:60]:
        print("   ", k, what, mag)

    torch.set_rng_state(rng0[0]); torch.cuda.set_rng_state(rng0[1])
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out3 = model(x, labels=y, bag_size=0, slot_layout=layout)
    same = bool(torch.equal(out3["loss"].detach(), loss1))
    print(f"F3 loss {float(out3['loss'].detach()):.10f}  equals F1: {same}")
    return 0 if (same and not changed) else 5


if __name__ == "__main__":
    sys.exit(main())
