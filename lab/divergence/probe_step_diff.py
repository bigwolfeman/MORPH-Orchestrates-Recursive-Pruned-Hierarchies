"""Does the Jacobian probe change the OPTIMIZER STEP that follows it?

    python lab/divergence/probe_step_diff.py --ckpt checkpoints/morph/run/ROLL_step_40.pt

The trainer's sequence at a probed step is: scaled forward+backward -> Jacobian probe ->
unscale -> clip -> scaler.step -> scaler.update. The forward and backward after the
probe are bit-identical (probe_alloc_diff.py), so this replicates the whole step, with
the checkpoint's optimizer and scaler state, three times from the same weights, RNG and
batch: S1 without the probe, S2 with it, S3 without it again. It compares the weights
after the step. S1 != S3 means the step itself is not reproducible; S1 == S3 != S2 means
the probe changes the update.
"""
from __future__ import annotations

import argparse
import copy
import sys

import torch
import torch.nn as nn

from _build import build_cfg
from _capture_lab import load_model_and_batch, require_deterministic_env

from morph.training.core_jacobian import CoreJacobianProbe  # noqa: E402
from morph.training.optimizer import align_optimizer_state, create_optimizer  # noqa: E402
from morph.training.train import _jacobian_probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_cap_c1")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--power", type=int, default=20)
    a = ap.parse_args()
    require_deterministic_env()
    cfg = build_cfg(a.config, ["model.use_kernels=false", "training.compile=false"])
    model, x, y, layout, step = load_model_and_batch(a.config, a.ckpt, a.batch)
    root = getattr(model, "_orig_mod", model)
    root._probe_loop = True
    root._probe_cot = True
    root._probe_rank = False

    ckpt = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    optimizer = create_optimizer(model, cfg)
    state, added = align_optimizer_state(ckpt["optimizer"], model, set(ckpt["model"].keys()))
    optimizer.load_state_dict(state)
    scaler = torch.amp.GradScaler("cuda")
    scaler.load_state_dict(ckpt["scaler"])
    print(f"step {step}; optimizer {type(optimizer).__name__}, {len(added)} params without state; "
          f"scaler scale {scaler.get_scale()}")
    w0 = copy.deepcopy(root.state_dict())
    o0 = copy.deepcopy(optimizer.state_dict())
    s0 = copy.deepcopy(scaler.state_dict())
    torch.manual_seed(1234)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
    grad_clip = float(cfg.training.grad_clip)
    probe = CoreJacobianProbe(model, n_iter=a.power, seed=0)

    def one_step(with_probe: bool) -> dict:
        root.load_state_dict(w0)
        optimizer.load_state_dict(copy.deepcopy(o0))
        scaler.load_state_dict(copy.deepcopy(s0))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y, bag_size=0, slot_layout=layout)
        scaler.scale(out["loss"]).backward()
        if with_probe:
            _jacobian_probe(model, probe, x, y, layout, 0, [3])
        scaler.unscale_(optimizer)
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        scaler.step(optimizer)
        scaler.update()
        print(f"  {'probe' if with_probe else 'plain'}: loss {float(out['loss']):.10f} gnorm {gn:.6f} "
              f"scale {scaler.get_scale()}")
        return {k: v.detach().clone() for k, v in root.state_dict().items()}

    s1 = one_step(False)
    s2 = one_step(True)
    s3 = one_step(False)
    ok = True
    for name, sa, sb in (("S3 vs S1 (plain twice)", s1, s3), ("S2 vs S1 (probe vs plain)", s1, s2)):
        worst, first = 0.0, None
        for k in sa:
            if sa[k].dtype.is_floating_point:
                d = float((sa[k].float() - sb[k].float()).abs().max())
                if d > worst:
                    worst = d
                if d > 0 and first is None:
                    first = k
        print(f"{name}: max|dw| {worst:.3e} first {first}")
        ok &= (worst == 0.0)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
