"""sigma_max(J_core) across the onset-capture checkpoint ladder, and under a spectral cap.

Two questions the gradient logs cannot answer, both about the OPERATOR rather than about
magnitudes (see morph/training/core_jacobian.py):

  Q1  Does sigma_max(J_core) rise as the run takes over, or was it always above 1 with the
      realized backward direction merely rotating into it? The ROLL_1625..1850 ladder
      brackets the onset, so measuring every rung answers it directly.

  Q2  What weight cap does it take to pull sigma_max(J_core) back below 1? Projecting the
      core linears onto {sigma_max(W) <= c} for a grid of c and re-measuring turns the
      spectral cap from a tuned number into a derived one.

ONE model, ONE batch, ONE operating point per checkpoint: the batch and the Poisson slot
depths are fixed by a seeded generator so every rung is compared at the same input.

Usage:
    PYTHONPATH=$PWD python lab/divergence/jac_ladder.py \
        --ckpt-dir checkpoints/morph/onset-capture --out ladder.json
    PYTHONPATH=$PWD python lab/divergence/jac_ladder.py \
        --ckpt-dir checkpoints/morph/onset-capture --sweep-caps 3.5,3.0,2.5,2.0,1.5,1.2,1.0 \
        --sweep-ckpt ROLL_step_1850.pt --out sweep.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch
from hydra import compose, initialize_config_dir

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from morph.model.transformer import MORPHTransformer          # noqa: E402
from morph.training.core_jacobian import CoreJacobianProbe    # noqa: E402
from morph.training.data import create_dataloader             # noqa: E402
from morph.training.train import build_morph_config           # noqa: E402
from morph.training.tul_setup import build_tul_runtime        # noqa: E402


def build(config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(_ROOT, "morph", "configs")):
        cfg = compose(config_name=config_name, overrides=overrides)
    tul_rt = build_tul_runtime(cfg)
    model = MORPHTransformer(build_morph_config(cfg, tul=tul_rt.model_cfg if tul_rt else None))
    model = model.cuda()
    loader = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                                    int(cfg.data.seq_len), int(cfg.training.batch_size),
                                    split="validation", skip_samples=50_000,
                                    tul=tul_rt.val_data_cfg if tul_rt else None))
    batch = next(loader)
    if len(batch) == 3:
        x, y, layout = batch
        layout = layout.to("cuda")
    else:
        (x, y), layout = batch, None
    return cfg, model, x.cuda(), y.cuda(), layout


def operating_points(model, x, y, layout, seed: int):
    """Capture one core step per loop iteration at a FIXED depth draw."""
    probe = CoreJacobianProbe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with probe.capture() as pts:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
    return [dict(p) for p in pts]


@torch.no_grad()
def project_core_linears(model, cap: float, n_iter: int = 60) -> dict:
    """Scale every core 2-D weight so sigma_max(W) <= cap. Returns what actually moved.

    This is the HARD version of the soft hinge in morph/training/spectral_penalty.py: a
    projection onto the spectral ball, applied to the loaded weights so the probe can read
    sigma_max(J) as a function of the cap alone.
    """
    moved = {}
    for name, p in model.named_parameters():
        if not name.startswith("core.") or p.dim() != 2 or p.numel() < 1024:
            continue
        w = p.data.float()
        u = torch.randn(w.shape[1], device=w.device,
                        generator=torch.Generator(device=w.device).manual_seed(0))
        u /= u.norm()
        s = torch.zeros((), device=w.device)
        for _ in range(n_iter):
            v = w @ u
            v = v / (v.norm() + 1e-12)
            u = w.t() @ v
            s = u.norm()
            u = u / (s + 1e-12)
        s = float(s)
        if s > cap:
            p.data.mul_(cap / s)
            moved[name] = (s, cap)
    return moved


def measure(model, points, iters, power_iters, per_block):
    probe = CoreJacobianProbe(model, n_iter=power_iters, seed=0, per_block=per_block)
    by_iter = {int(p["iter_idx"]): p for p in points}
    out = {}
    for t in iters:
        if t not in by_iter:
            continue
        r = probe.measure(by_iter[t])
        out[f"t{t}"] = {"sigma": r.sigma_step, "conv": r.rel_change,
                        "blocks": r.sigma_blocks, "blockprod": r.block_product}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a1")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--iters", default="0,3")
    ap.add_argument("--power-iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-caps", default="")
    ap.add_argument("--sweep-ckpt", default="")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    iters = [int(t) for t in a.iters.split(",") if t.strip()]
    ov = [o for o in a.overrides.split(",") if o.strip()]
    cfg, model, x, y, layout = build(a.config_name, ov)
    model.train()                       # Poisson depths, as in training

    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if a.sweep_ckpt:
        ckpts = [os.path.join(a.ckpt_dir, a.sweep_ckpt)]
    caps = [float(c) for c in a.sweep_caps.split(",") if c.strip()] or [None]

    results = []
    for path in ckpts:
        raw = torch.load(path, map_location="cuda", weights_only=False)
        base_sd = raw.get("model", raw)
        step = int(raw.get("step", -1))
        for cap in caps:
            model.load_state_dict(base_sd, strict=False)
            moved = project_core_linears(model, cap) if cap else {}
            pts = operating_points(model, x, y, layout, a.seed)
            m = measure(model, pts, iters, a.power_iters, not a.no_blocks)
            row = {"ckpt": os.path.basename(path), "step": step, "cap": cap,
                   "n_projected": len(moved), "sigma": m}
            results.append(row)
            head = m.get(f"t{iters[0]}", {})
            print(f"{os.path.basename(path):<24} cap={cap} proj={len(moved):>3} "
                  f"sigma_t{iters[0]}={head.get('sigma', float('nan')):.4f} "
                  f"conv={head.get('conv', float('nan')):.1e} "
                  f"blockprod={head.get('blockprod', float('nan')):.4f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
