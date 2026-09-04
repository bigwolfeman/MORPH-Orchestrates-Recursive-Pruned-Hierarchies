"""Is the replay divergence allocator-layout dependence, and does the Jacobian probe
trigger it through the trainer-shaped sequence (backward -> probe -> next backward)?

    python lab/divergence/probe_alloc_diff.py --ckpt checkpoints/morph/run/ROLL_step_40.pt

Same batch, weights and RNG each time (deterministic settings as the trainer):
  G1: forward+backward                                   — the reference gradients
  G2: forward+backward again                             — reproducibility floor
  G3: after the Jacobian probe (exactly `_jacobian_probe` + measure), forward+backward
  G4: after allocating and KEEPING a few odd-sized junk tensors, forward+backward
  G5: after `torch.cuda.empty_cache()`, forward+backward
Each is compared with G1 bit for bit. If G3 and G4 differ while G2 and G5 match, the
forward/backward is deterministic for a fixed memory layout and the probe moves the layout.
"""
from __future__ import annotations

import argparse
import sys

import torch

from _capture_lab import cmp_grads, fwd_bwd, load_model_and_batch, require_deterministic_env

from morph.training.core_jacobian import CoreJacobianProbe  # noqa: E402
from morph.training.train import _jacobian_probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_cap_c1")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--power", type=int, default=20)
    a = ap.parse_args()
    require_deterministic_env()
    model, x, y, layout, step = load_model_and_batch(a.config, a.ckpt, a.batch)
    torch.manual_seed(1234)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())

    l1, g1 = fwd_bwd(model, x, y, layout, rng)
    l2, g2 = fwd_bwd(model, x, y, layout, rng)
    probe = CoreJacobianProbe(model, n_iter=a.power, seed=0)
    row = _jacobian_probe(model, probe, x, y, layout, 0, [3])
    l3, g3 = fwd_bwd(model, x, y, layout, rng)
    junk = [torch.empty(int(1e6) + 16 * k, dtype=torch.uint8, device="cuda") for k in range(1, 6)]
    l4, g4 = fwd_bwd(model, x, y, layout, rng)
    del junk
    torch.cuda.empty_cache()
    l5, g5 = fwd_bwd(model, x, y, layout, rng)
    print(f"step {step}; probe sigma_t3={row.get('jac/sigma_t3'):.4f} rms_t3={row.get('jac/rms_t3'):.4f}")
    print(f"losses: G1 {l1:.10f} G2 {l2:.10f} G3 {l3:.10f} G4 {l4:.10f} G5 {l5:.10f}")
    ok = True
    for name, g in (("G2 repeat", g2), ("G3 after jac probe", g3), ("G4 after junk allocs", g4),
                    ("G5 after empty_cache", g5)):
        w, f = cmp_grads(g1, g)
        print(f"{name:22s} vs G1: max|dgrad| {w:.3e} first {f}")
        ok &= (w == 0.0)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
