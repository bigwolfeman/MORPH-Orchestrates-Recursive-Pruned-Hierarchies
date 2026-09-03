"""Does the looped core map care how big its input is?

SCSE evolves a DEVIATION `Delta` from a fixed anchor, not the state. That only works if
the core map is roughly HOMOGENEOUS: feed it a small input, get a small output. If the map
instead ignores the size of its input, then a deliberately small `Delta_0` comes back out
at whatever size the weights choose, the anchor stops mattering after one iteration, and
the deviation recurrence is a state recurrence wearing a different name.

MORPH's blocks are PRE-NORM. `MORPHBlock` applies `RMSNorm` at the input of each sublayer,
and RMSNorm divides by the RMS of what it is given. That makes each sublayer's output size
a function of the WEIGHTS, not of the input size. This script measures whether that is
true of the whole 6-block core step, at the real operating point of a real checkpoint.

Method: capture one live operating point `(h, e, inj)`, then evaluate the SAME core step at
`h * s` for a ladder of `s`, and report the output RMS. A homogeneous map gives output RMS
proportional to `s`. A scale-free map gives the same output RMS for every `s`.

    python lab/divergence/scale_probe.py --ckpt checkpoints/morph/onset-capture/ROLL_step_1750.pt
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence.jac_ladder import build            # noqa: E402
from morph.training.core_jacobian import CoreJacobianProbe   # noqa: E402
from morph.training.train import load_checkpoint       # noqa: E402


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--iters", default="0,3", help="which loop iterations to probe")
    ap.add_argument("--scales", default="1.0,0.5,0.1,0.01,0.001")
    a = ap.parse_args()

    overrides = ["training.batch_size=6", "model.use_kernels=false"]
    cfg, model, x, y, layout = build(a.config, overrides)

    # THE loader the trainer uses. A plain `load_state_dict(strict=False)` here loaded
    # NOTHING into the core MLPs: checkpoints written under torch.compile carry an
    # `_orig_mod.` prefix (`mlp._orig_mod.0.gate_up...`) that a fresh model does not have,
    # so 48 core keys silently found no home and the probe measured fresh-init weights.
    # `load_checkpoint` aligns the prefix on both sides and RAISES on a homeless tensor.
    root = getattr(model, "_orig_mod", model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, torch.device("cuda"))
    model.eval()

    probe = CoreJacobianProbe(model)
    with probe.capture() as pts:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
    print(f"  captured {len(pts)} operating points\n")

    want = {int(v) for v in a.iters.split(",")}
    scales = [float(v) for v in a.scales.split(",")]

    for p in pts:
        t = int(p["iter_idx"])
        if t not in want:
            continue
        h, e, inj, ret = p["h"], p["e"], p["inj"], p["ret_state"]
        base_in = rms(h)
        print(f"iteration {t}   input RMS at the real operating point = {base_in:.4f}")
        print(f"  {'scale s':>9} {'RMS(h*s)':>12} {'RMS(step)':>12} "
              f"{'out/in':>10} {'vs s=1':>9}  homogeneous?")
        ref_out = None
        for s in scales:
            hs = h * s
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                out, _ = root._apply_core_step(hs.float(), e.float(), None, None, None,
                                               ret_state=ret, iter_idx=t,
                                               inj_terms=inj.float())
            oi, oo = rms(hs), rms(out)
            if ref_out is None:
                ref_out = oo
            # A homogeneous map would give out/ref == s. A scale-free map gives 1.0.
            print(f"  {s:>9.4g} {oi:>12.4f} {oo:>12.4f} {oo / max(oi, 1e-12):>10.3f} "
                  f"{oo / ref_out:>9.4f}  {'yes' if abs(oo / ref_out - s) < 0.1 * max(s, 1e-3) else 'NO'}")
        print()


if __name__ == "__main__":
    main()
