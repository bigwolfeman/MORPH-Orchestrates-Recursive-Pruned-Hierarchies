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


def _raw_weight(mod):
    """The tensor an optimizer actually owns for `mod`, through any parametrization."""
    par = getattr(mod, "parametrizations", None)
    if par is not None and "weight" in par:
        return par["weight"].original
    w = getattr(mod, "weight", None)
    return w if (w is not None and w.dim() == 2 and w.numel() >= 1024) else None


def core_linears(model, scope: str):
    """(name, module) for the core's 2-D linear maps, filtered by `scope`.

    scope: `mlp` = the SwiGLU gate_up/down (what morph/training/spectral_penalty.py
    covers), `attn` = the CCA projections, `all` = both plus the hyper-connection
    projections.
    """
    out = []
    root = getattr(model, "_orig_mod", model)
    for li, blk in enumerate(root.core):
        for sub_name, sub in blk.named_modules():
            if _raw_weight(sub) is None:
                continue
            name = f"core.{li}.{sub_name}"
            is_mlp = ".mlp" in name
            is_attn = ".attention" in name
            if scope == "mlp" and not is_mlp:
                continue
            if scope == "attn" and not is_attn:
                continue
            out.append((name, sub))
    return out


def project_core_linears(model, cap: float, scope: str = "all", n_iter: int = 60) -> dict:
    """Scale each selected core linear so sigma_max of its EFFECTIVE map is <= cap.

    THROUGH THE MODULE'S FORWARD, not off the raw parameter. The core MLP runs ternary QAT
    (`ternary_scope: backbone`), so the raw weight and the weight the forward applies are
    different matrices with different spectra; the soft penalty in
    morph/training/spectral_penalty.py measures the effective one, and a sweep that
    measured the raw one would report caps in units the penalty does not use.

    Scaling the raw weight scales the effective map exactly ON THE PATH IN USE. MORPH
    ternarises through `TernarySTE` registered as a weight parametrization, whose
    per-tensor scale is `gamma = mean(|W|)` recomputed from `W` on every forward, so
    `W -> cW` gives `gamma -> c gamma`, an unchanged code pattern, and `W_eff -> c W_eff`.
    That is NOT true of `CMSBlockLinear.enable_ternary`, whose scale is a buffer frozen at
    the transition — under that path scaling `W` only moves entries across the threshold.
    Rather than trust which path is live, every projection RE-MEASURES sigma afterwards and
    raises if it did not land on the cap.

    This is the HARD projection onto the spectral ball. The shipped cure is the SOFT
    hinge; this exists so the cap can be read off a curve instead of tuned.
    """
    from morph.training.spectral_penalty import _power_iter_sigma
    moved = {}
    for name, mod in core_linears(model, scope):
        raw = _raw_weight(mod)
        v = torch.randn(raw.shape[1], device=raw.device, dtype=raw.dtype,
                        generator=torch.Generator(device=raw.device).manual_seed(0))
        with torch.enable_grad():
            sig, _ = _power_iter_sigma(mod, v, n_iter)
        s = float(sig.detach())
        if s > cap:
            with torch.no_grad():
                raw.data.mul_(cap / s)
            with torch.enable_grad():
                sig2, _ = _power_iter_sigma(mod, v, n_iter)
            s2 = float(sig2.detach())
            if abs(s2 - cap) / cap > 0.02:
                raise RuntimeError(
                    f"projection of {name} did not land on the cap: sigma {s:.4f} -> "
                    f"{s2:.4f}, wanted {cap:.4f}. The effective map is not homogeneous in "
                    f"the raw weight on this path (see the docstring), so this sweep would "
                    f"report caps in units nothing else uses.")
            moved[name] = (s, s2)
    return moved


@torch.no_grad()
def core_sigmas(model, scope: str = "all", n_iter: int = 60) -> dict:
    """Effective sigma_max per core linear — the diagnostic the projection is built on."""
    from morph.training.spectral_penalty import _power_iter_sigma
    out = {}
    for name, mod in core_linears(model, scope):
        raw = _raw_weight(mod)
        v = torch.randn(raw.shape[1], device=raw.device, dtype=raw.dtype,
                        generator=torch.Generator(device=raw.device).manual_seed(0))
        with torch.enable_grad():
            sig, _ = _power_iter_sigma(mod, v, n_iter)
        out[name] = float(sig.detach())
    return out


def measure(model, points, iters, power_iters, per_block):
    probe = CoreJacobianProbe(model, n_iter=power_iters, seed=0, per_block=per_block)
    by_iter = {int(p["iter_idx"]): p for p in points}
    out = {}
    for t in iters:
        if t not in by_iter:
            continue
        r = probe.measure(by_iter[t])
        out[f"t{t}"] = {"sigma": r.sigma_step, "conv": r.rel_change, "rms": r.rms_step,
                        "blocks": r.sigma_blocks, "blockprod": r.block_product,
                        "rms_blocks": r.rms_blocks, "rms_block_gain": r.rms_block_gain}
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
    ap.add_argument("--project-scope", default="all", choices=["all", "mlp", "attn"])
    ap.add_argument("--dump-sigmas", action="store_true")
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
            pre_sigmas = core_sigmas(model) if a.dump_sigmas else None
            moved = (project_core_linears(model, cap, a.project_scope)
                     if cap else {})
            pts = operating_points(model, x, y, layout, a.seed)
            m = measure(model, pts, iters, a.power_iters, not a.no_blocks)
            row = {"ckpt": os.path.basename(path), "step": step, "cap": cap,
                   "scope": a.project_scope, "n_projected": len(moved), "sigma": m}
            if pre_sigmas is not None:
                row["weight_sigmas"] = pre_sigmas
            results.append(row)
            head = m.get(f"t{iters[0]}", {})
            print(f"{os.path.basename(path):<22} cap={cap} proj={len(moved):>3} "
                  f"sigma={head.get('sigma', float('nan')):>10.4f} "
                  f"rms={head.get('rms', float('nan')):>8.4f} "
                  f"blk_rms_gain={head.get('rms_block_gain', float('nan')):>7.4f} "
                  f"conv={head.get('conv', float('nan')):.0e}", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
