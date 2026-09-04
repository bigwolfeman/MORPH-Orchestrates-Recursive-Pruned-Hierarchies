"""Operator gain of the slot loop across saved checkpoints, on ONE fixed batch.

    python lab/divergence/jac_sweep.py --out sweep.jsonl \
        tul_cap_c1:checkpoints/morph/cap-c1-det/ROLL_step_650.pt \
        tul_cap_c0:checkpoints/morph/cap-c0-nt/step_2500.pt ...

Why: the onset capture (2026-09-04, `lab/experiments/planned/2026-09-03-tul-onset-capture.md`)
measured the loop's typical gain `jac/rms_t3` drifting 0.87 -> 1.00 over 1200 steps on the
ternary run and crossing 1 at the spike onset, while the same arm with ternary OFF ran 5000
steps clean. C0 logged no Jacobian, so this sweep asks the missing question offline: does the
gain drift to 1 WITHOUT ternary, and on the healthy (non-MUX) arms? Every checkpoint is
loaded eager (kernels off, compile off) into the model its config describes, and measured
at the same batch of validation rows with the same probe seed, so rows are comparable.

Per checkpoint, one JSON line: the `jac/*` dict of `train._jacobian_probe` at the requested
iterations (whole-map sigma and rms, per block) plus the `loop/*` rows of `train._preclip_probe`
from ONE forward+backward (delta_mean, delta_ratio, core_gain, eff_rank, cot_norm per
iteration). Gradients are computed only to drive the cotangent hook; nothing is stepped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from _capture_lab import load_model_and_batch

from morph.training.core_jacobian import CoreJacobianProbe  # noqa: E402
from morph.training.train import _jacobian_probe, _preclip_probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="config:ckpt_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--power", type=int, default=60)
    ap.add_argument("--iters", default="0,3,7")
    a = ap.parse_args()
    iters = [int(t) for t in a.iters.split(",")]
    torch.backends.cudnn.benchmark = False
    ref = None
    with open(a.out, "a") as fh:
        for pair in a.pairs:
            config, ckpt = pair.split(":", 1)
            t0 = time.time()
            model, x, y, layout, step = load_model_and_batch(config, ckpt, a.batch)
            if ref is None:
                ref = (x.clone(), y.clone())
            elif not (torch.equal(ref[0], x) and torch.equal(ref[1], y)):
                raise SystemExit(f"batch differs for {pair}; rows would not be comparable")
            root = getattr(model, "_orig_mod", model)
            root._probe_loop = True
            root._probe_rank = True
            root._probe_cot = True
            torch.manual_seed(1234)
            probe = CoreJacobianProbe(model, n_iter=a.power, seed=0, per_block=True)
            row = {"run": os.path.basename(os.path.dirname(ckpt)), "ckpt": ckpt,
                   "config": config, "step": step}
            row.update(_jacobian_probe(model, probe, x, y, layout, 0, iters))
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1234)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, labels=y, bag_size=0, slot_layout=layout)
            out["loss"].backward()
            row["loss"] = float(out["loss"].detach())
            row.update({k: v for k, v in _preclip_probe(model).items()
                        if k.startswith(("loop/", "preclip/total", "preclip/core"))})
            row["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"{row['run']:14s} step {step:5d} rms_t3 {row.get('jac/rms_t3', float('nan')):.3f} "
                  f"sigma_t3 {row.get('jac/sigma_t3', float('nan')):.1f} rms_t0 {row.get('jac/rms_t0', float('nan')):.3f} "
                  f"loss {row['loss']:.3f} ({row['seconds']}s)", flush=True)
            del model, probe, out
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
