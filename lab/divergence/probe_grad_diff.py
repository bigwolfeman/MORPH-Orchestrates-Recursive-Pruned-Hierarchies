"""Do the loop-probe readings change the BACKWARD's gradients?

    python lab/divergence/probe_grad_diff.py --ckpt checkpoints/morph/run/ROLL_step_40.pt

Same batch, same weights, same RNG, deterministic settings as the trainer. Runs
forward+backward four times and compares every parameter gradient bit for bit:
  G1: rank probe ON,  cot hook ON
  G2: rank probe OFF, cot hook ON
  G3: rank probe ON,  cot hook OFF
  G4: rank probe ON,  cot hook ON  (a second G1 — is the backward reproducible at all?)
Prints the max |diff| and the first differing parameter for each pair. Exit 0 if all four
agree, 5 otherwise. Result 2026-09-03 on smoke-cap-a/ROLL_step_40: all four agree.
"""
from __future__ import annotations

import argparse
import sys

import torch

from _capture_lab import cmp_grads, fwd_bwd, load_model_and_batch, require_deterministic_env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_cap_c1")
    ap.add_argument("--batch", type=int, default=6)
    a = ap.parse_args()
    require_deterministic_env()
    model, x, y, layout, step = load_model_and_batch(a.config, a.ckpt, a.batch)
    torch.manual_seed(1234)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())

    l1, g1 = fwd_bwd(model, x, y, layout, rng, rank=True, cot=True)
    l2, g2 = fwd_bwd(model, x, y, layout, rng, rank=False, cot=True)
    l3, g3 = fwd_bwd(model, x, y, layout, rng, rank=True, cot=False)
    l4, g4 = fwd_bwd(model, x, y, layout, rng, rank=True, cot=True)
    print(f"step {step}; losses: G1 {l1:.10f} G2 {l2:.10f} G3 {l3:.10f} G4 {l4:.10f}")
    ok = True
    for name, (ga, gb) in {"G1 vs G4 (repeat)": (g1, g4), "G1 vs G2 (rank off)": (g1, g2),
                           "G1 vs G3 (cot off)": (g1, g3)}.items():
        w, f = cmp_grads(ga, gb)
        print(f"{name}: max|dgrad| {w:.3e} first differing {f}")
        ok &= (w == 0.0)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
