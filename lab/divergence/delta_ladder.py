"""How big does the deviation get, iteration by iteration, under each SCSE arm?

The 2026-08-25 SCSE port failed in ONE loop iteration. Measured from its saved probe data
(lab/experiments/results/2026-08-25-scse-full-method/scse-s1.json, training step 1000),
the carrier RMS ran 6.5 -> 1499.6 -> ... -> 4706.8 across eight iterations: 230x in the
FIRST step and 722x overall, against the control's 64.2 -> 124.2, i.e. 1.9x. After that
first step SCSE's per-iteration relative change actually FELL (1.20, 0.93, 0.78, 0.67,
0.61, 0.65, 0.58) and was better behaved than the control's steady ~0.90.

`lab/divergence/scale_probe.py` measured the cause on trained weights: MORPH's blocks are
PRE-NORM, so the core map's output size comes from the weights, not from the input. Shrink
its input 1000x and the output moves 31 %.

This script runs the SCSE recurrence FORWARD ONLY on a real checkpoint and reports
||Delta_t|| per iteration for each arm. No training, no optimizer, minutes not hours.

    python lab/divergence/delta_ladder.py --ckpt checkpoints/morph/onset-capture/ROLL_step_1750.pt
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence.jac_ladder import build              # noqa: E402
from morph.model.transformer import _SCSE                # noqa: E402
from morph.training.core_jacobian import CoreJacobianProbe    # noqa: E402
from morph.training.train import load_checkpoint         # noqa: E402


def rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def run_arm(root, scse: _SCSE, e: torch.Tensor, ret, n_iter: int) -> dict:
    """One arm's deviation trajectory. Mirrors the live recurrence in `_tul_core`:
    `rec = recurrent_input(Delta, h*)`, `g = core(rec, source_free=True)`,
    `Delta <- update(Delta, g, rec)`."""
    h_star, delta = scse.entry(e)
    out = {"h_star_rms": rms(h_star), "delta": [rms(delta)], "state": [rms(h_star + delta)],
           "ratio": []}
    for t in range(n_iter):
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            rec = scse.recurrent_input(delta, h_star)
            g, _ = root._apply_core_step(rec, h_star, None, None, None, ret_state=ret,
                                         iter_idx=t, inj_terms=None, source_free=True)
            out["ratio"].append(rms(g) / max(rms(rec), 1e-12))
            delta = scse.update(delta, g, rec)
        out["delta"].append(rms(delta))
        out["state"].append(rms(h_star + delta))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a1")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg, model, x, y, layout = build(
        a.config, ["training.batch_size=6", "model.use_kernels=false"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, torch.device("cuda"))
    model.eval()
    root = getattr(model, "_orig_mod", model)

    # One live operating point gives the anchor input `e` the real forward produced. The
    # CONTROL forward is used, so `e` is the same tensor every arm below is entered with.
    probe = CoreJacobianProbe(model)
    with probe.capture() as pts:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
    p0 = [p for p in pts if int(p["iter_idx"]) == 0][0]
    e = p0["e"].float()
    ctrl_state = rms(p0["h"])
    print(f"\ncontrol state RMS at iteration 0 = {ctrl_state:.4f}   "
          f"(what a state-sized Delta_0 should match)\n")

    d = root.cfg.d_model
    torch.manual_seed(a.seed)

    # Arm A's init_scale is CHOSEN BY MEASUREMENT, not guessed: build the entry at
    # init_scale = 1 and read how big Delta_0 comes out, then scale linearly. Delta_0 is
    # `init_scale*init_proj(e) - anchor_scale*a_omega(e)`, so it is affine in init_scale,
    # not proportional — the anchor term does not scale with it. Solve on the measured
    # endpoints instead of assuming proportionality.
    torch.manual_seed(a.seed)
    ref = _SCSE(d, step_scale=cfg.model.scse_step_scale, anchor_scale=cfg.model.scse_anchor_scale,
                init_scale=1.0, eps=cfg.model.scse_eps, kappa=0.0).cuda().float()
    with torch.no_grad():
        _, d_at_1 = ref.entry(e)
        ref.init_scale = 0.0
        _, d_at_0 = ref.entry(e)
    r1, r0 = rms(d_at_1), rms(d_at_0)
    # rms is not linear in init_scale, but over this range it is close enough to solve on;
    # the arm then REPORTS its realised ||Delta_0|| so the choice is checkable, not assumed.
    a_scale = max(1e-3, (ctrl_state - r0) / max(r1 - r0, 1e-9))
    print(f"arm A calibration: ||Delta_0|| = {r0:.4f} at init_scale 0, {r1:.4f} at 1.0 "
          f"-> init_scale {a_scale:.3f} targets {ctrl_state:.4f}\n")

    arms = [
        ("SCSE as built", dict(init_scale=cfg.model.scse_init_scale,
                               input_mode="deviation", delta_clip=0.0)),
        ("A: state-size Delta_0", dict(init_scale=a_scale,
                                       input_mode="deviation", delta_clip=0.0)),
        ("C: state input", dict(init_scale=cfg.model.scse_init_scale,
                                input_mode="state", delta_clip=0.0)),
        ("C+A: state input, big D0", dict(init_scale=a_scale,
                                          input_mode="state", delta_clip=0.0)),
        ("C+clip", dict(init_scale=cfg.model.scse_init_scale,
                        input_mode="state", delta_clip=ctrl_state)),
    ]

    rows = []
    print(f"{'arm':<28} {'||D_0||':>8} {'||D_T||':>8} {'D_T/state':>10} {'growth':>8}   "
          f"||Delta_t|| per iteration")
    for name, kw in arms:
        torch.manual_seed(a.seed)      # same projection draw for every arm
        s = _SCSE(d, step_scale=cfg.model.scse_step_scale,
                  anchor_scale=cfg.model.scse_anchor_scale,
                  eps=cfg.model.scse_eps, kappa=0.0, **kw).cuda().float()
        r = run_arm(root, s, e, p0["ret_state"], a.iters)
        dl = r["delta"]
        growth = dl[-1] / max(dl[0], 1e-12)
        rows.append((name, r))
        print(f"{name:<28} {dl[0]:>8.3f} {dl[-1]:>8.3f} {dl[-1] / ctrl_state:>10.2f} "
              f"{growth:>7.1f}x   " + " ".join(f"{v:7.2f}" for v in dl))
    print()
    print(f"{'arm':<28} {'':>8} core amplification ||G(rec)||/||rec|| per iteration")
    for name, r in rows:
        print(f"{name:<28} {'':>8} " + " ".join(f"{v:7.2f}" for v in r["ratio"]))
    print()
    print("D_T/state compares the FINAL deviation to the control's state size. That is the")
    print("quantity that matters: a deviation the size of the state means the anchor is")
    print("swamped, whatever ratio it grew by from a tiny start.")
    print()
    print("CAVEAT: these are fresh SCSE projections on a checkpoint from a NON-SCSE run.")
    print("This measures the ENTRY condition, not the pathology of a model that already")
    print("trained 1000 steps under SCSE. Absolute values are NOT comparable to the")
    print("training probe's h_rms, which uses a different reduction.")


if __name__ == "__main__":
    main()
