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

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from lab.divergence._build import build_cfg, build_model       # noqa: E402
from morph.training.core_jacobian import CoreJacobianProbe    # noqa: E402
from morph.training.data import create_dataloader             # noqa: E402
from morph.training.train import load_checkpoint              # noqa: E402


def build(config_name: str, overrides: list[str]):
    cfg = build_cfg(config_name, overrides)
    model, tul_rt = build_model(cfg, device="cuda")
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
    # `named_modules()` also yields the parametrization machinery itself, whose `.weight` is
    # a ParametrizationList, not a tensor — hence the isinstance guard rather than a
    # duck-typed `.dim()`.
    if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.numel() < 1024:
        return None
    return w


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
            if sub_name.endswith("parametrizations") or ".parametrizations" in sub_name:
                continue          # the STE modules, not the linear maps
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


def cotangent_rank(model, x, y, layout, seed: int, token_path: bool = False) -> dict:
    """Effective number of positions carrying the backward cotangent at the core.

    The alignment reading says the loop power-iterates the cotangent onto the core map's
    top singular direction. Power iteration can only concentrate on a direction the
    cotangent is free to occupy, and the cotangent at a core block is a SUM over the
    active positions — so its effective rank is bounded by how many positions there are.
    That is the standing explanation for why arm A1 (64 slots) fails where A0 (1024 token
    positions) does not, and this measures it instead of assuming it. `token_path=True`
    runs the SAME weights with `slot_layout=None`, i.e. the core looping over every token
    position — arm A0's code path — so the comparison holds the operator fixed and varies
    only how many positions the cotangent is spread over.

    Participation ratio over positions, `PR = (sum a_p)^2 / (n * sum a_p^2)` with
    `a_p = ||dL/dh_p||^2`; `PR * n` is the effective number of positions. Read off
    `register_full_backward_hook` on each core block, so it uses only public API and needs
    no model change.
    """
    root = getattr(model, "_orig_mod", model)
    per_block: dict[int, list[float]] = {i: [] for i in range(len(root.core))}
    handles = []

    def mk(i):
        def hook(_mod, _gin, gout):
            g = gout[0]
            if g is None:
                return
            a = g.detach().float().flatten(2).pow(2).sum(-1)      # [B, S]
            s1 = a.sum(dim=1)
            s2 = a.pow(2).sum(dim=1)
            n = a.shape[1]
            pr = (s1 * s1) / (s2.clamp_min(1e-30) * n)            # [B] in (0, 1]
            per_block[i].append(float(pr.mean()) * n)
        return hook

    for i, blk in enumerate(root.core):
        handles.append(blk.register_full_backward_hook(mk(i)))
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = (model(x, labels=y) if token_path
                   else model(x, labels=y, slot_layout=layout))
        out["loss"].backward()
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
    if token_path:
        n_pos = int(x.shape[1])
        n_valid = float(n_pos)
    else:
        n_pos = int(layout.slot_valid.shape[1])
        n_valid = float(layout.slot_valid.float().sum(dim=1).mean())
    return {"n_positions": n_pos, "mean_valid_slots": n_valid,
            "eff_positions_per_block": {i: (sum(v) / len(v) if v else float("nan"))
                                        for i, v in per_block.items()},
            "n_calls_per_block": {i: len(v) for i, v in per_block.items()}}


def spectral_gap(model, n_iter: int = 200) -> dict:
    """sigma_1 / sigma_2 per core linear, by DEFLATED power iteration.

    Power iteration onto a matrix's top singular direction converges like
    `(sigma_1 / sigma_2)^k`, so the gap is the quantity that governs how fast the
    weight-shared loop can align its cotangent. It is NOT what a spectral norm cap
    constrains (a uniform rescale leaves every ratio untouched), and it is NOT what the
    isometry spread sees either — a random direction puts only `1/n` of its energy on the
    top singular vector, so in 1024 dimensions a single dominant direction is invisible to a
    bulk statistic.

    sigma_2 comes from the same iteration run in the orthogonal complement of the converged
    top right-singular vector, re-projected every step so it cannot drift back.
    """
    from morph.training.spectral_penalty import _power_iter_sigma, collect_core_linears
    lins, _ = collect_core_linears(model, True, "spectral_gap")
    out = {}
    for name, lin, inf in lins:
        w = next(lin.parameters())
        g = torch.Generator(device=w.device).manual_seed(0)
        v1 = torch.randn(inf, device=w.device, dtype=w.dtype, generator=g)
        with torch.enable_grad():
            s1, v1 = _power_iter_sigma(lin, v1, n_iter)
        s1 = float(s1.detach())
        # sigma_2: power iteration restricted to v1's orthogonal complement.
        v2 = torch.randn(inf, device=w.device, dtype=w.dtype, generator=g)
        v2 = v2 - (v2 @ v1) * v1
        v2 = v2 / (v2.norm() + 1e-12)
        s2 = 0.0
        for _ in range(n_iter):
            v2 = v2.detach().requires_grad_(True)
            wv = lin(v2.unsqueeze(0)).squeeze(0)
            (g2,) = torch.autograd.grad(0.5 * (wv * wv).sum(), v2)
            g2 = g2.detach()
            g2 = g2 - (g2 @ v1) * v1                     # stay in the complement
            n = g2.norm()
            if float(n) < 1e-20:
                break
            v2 = g2 / n
        with torch.no_grad():
            s2 = float(lin(v2.unsqueeze(0)).squeeze(0).norm())
        out[name] = {"sigma1": s1, "sigma2": s2, "gap": s1 / max(s2, 1e-12)}
    return out


def state_geometry(model, x, y, layout, seed: int) -> dict:
    """Is the FORWARD collapsing, or only the backward concentrating?

    The competing reading of the same numbers: the 57 slot states are built from one shared
    `E_slot` plus a span bag-mean, so they start near-parallel, and then the SAME six blocks
    are applied to them 6 to 8 times. That is the oversmoothing regime. If the slot states
    are losing rank across the loop, the backward's concentration is a SYMPTOM and the lever
    is state diversity, not anything in the backward.

    Per loop iteration, on the valid slots of each row: the participation ratio of the
    squared singular values of the [S_valid, C] state matrix (its effective rank), and the
    mean pairwise cosine between slot states. Both read off the operating points the Jacobian
    probe already captures, so this costs one forward.

    Also returns WHICH slots carry the backward cotangent, and whether the same ones carry it
    at every core block — a stable sink is a different disease from a drifting one.
    """
    root = getattr(model, "_orig_mod", model)
    probe = CoreJacobianProbe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with probe.capture() as pts:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
    per_iter = []
    for p in pts:
        h = p["h"].float()
        h = h.mean(dim=2) if h.dim() == 4 else h            # reduce the n hyper-connection streams
        m = p["active"]
        erank, cos, urank = [], [], []
        for b in range(h.shape[0]):
            hb = h[b][m[b]]
            if hb.shape[0] < 2:
                continue
            sv = torch.linalg.svdvals(hb.double())
            e = sv ** 2
            erank.append(float((e.sum() ** 2) / (e ** 2).sum()))
            hn = hb / (hb.norm(dim=1, keepdim=True) + 1e-12)
            # UNIT-NORMALISED rank too. The participation ratio of squared singular values
            # is norm-weighted, so ONE slot with a huge carrier norm reads as low rank even
            # when the directions are perfectly spread. Normalising first separates "one
            # slot is big" from "the directions merged", and the two disagreeing is itself
            # a finding.
            svu = torch.linalg.svdvals(hn.double()) ** 2
            urank.append(float((svu.sum() ** 2) / (svu ** 2).sum()))
            g = hn @ hn.t()
            n = g.shape[0]
            cos.append(float((g.sum() - n) / (n * (n - 1))))
        per_iter.append({"iter": int(p["iter_idx"]),
                         "eff_rank": sum(erank) / max(len(erank), 1),
                         "eff_rank_unit": sum(urank) / max(len(urank), 1),
                         "mean_cos": sum(cos) / max(len(cos), 1),
                         "n_slots": int(m[0].sum())})

    # which slots carry the cotangent, per core block
    share, handles = {i: [] for i in range(len(root.core))}, []

    def mk(i):
        def hook(_m, _gi, go):
            g = go[0]
            if g is None:
                return
            a = g.detach().float().flatten(2).pow(2).sum(-1)[0]     # row 0, [S]
            share[i].append((a / a.sum().clamp_min(1e-30)).cpu())
        return hook

    for i, blk in enumerate(root.core):
        handles.append(blk.register_full_backward_hook(mk(i)))
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y, slot_layout=layout)
        out["loss"].backward()
    finally:
        for h_ in handles:
            h_.remove()
        model.zero_grad(set_to_none=True)
    tops = {}
    for i, lst in share.items():
        if not lst:
            continue
        a = sum(lst) / len(lst)
        v, idx = torch.topk(a, k=min(5, a.numel()))
        tops[i] = {"top_idx": idx.tolist(), "top_share": [round(float(t), 4) for t in v]}
    agree = None
    if len(tops) >= 2:
        sets = [set(t["top_idx"][:3]) for t in tops.values()]
        inter = set.intersection(*sets)
        agree = len(inter) / 3.0
    return {"per_iter": per_iter, "cotangent_top": tops, "top3_agreement_across_blocks": agree}


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
    ap.add_argument("--rank-probe", action="store_true",
                    help="effective positions carrying the cotangent, instead of sigma")
    ap.add_argument("--state-probe", action="store_true",
                    help="forward slot-state effective rank per loop iteration, and which "
                         "slots carry the cotangent")
    ap.add_argument("--gap-probe", action="store_true",
                    help="sigma_1/sigma_2 per core linear, instead of the Jacobian")
    ap.add_argument("--rank-token-path", action="store_true",
                    help="with --rank-probe: run the SAME weights on the token path (A0)")
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
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for path in ckpts:
        step = int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
        for cap in caps:
            # THE loader the trainer uses: `_orig_mod` key alignment on both sides, and it
            # RAISES when a checkpoint tensor finds no home. Re-loaded per cap because the
            # projection mutates the weights in place.
            load_checkpoint(path, model, scaler, torch.device("cuda"))
            pre_sigmas = core_sigmas(model) if a.dump_sigmas else None
            moved = (project_core_linears(model, cap, a.project_scope)
                     if cap else {})
            if a.state_probe:
                st = state_geometry(model, x, y, layout, a.seed)
                results.append({"ckpt": os.path.basename(path), "step": step, "state": st})
                er = " ".join(f"{r['eff_rank']:.2f}" for r in st["per_iter"])
                eu = " ".join(f"{r['eff_rank_unit']:.2f}" for r in st["per_iter"])
                co = " ".join(f"{r['mean_cos']:+.3f}" for r in st["per_iter"])
                print(f"{os.path.basename(path):<22} slots={st['per_iter'][0]['n_slots']:>3} "
                      f"eff_rank/iter={er}", flush=True)
                print(f"{'':<22} unit_rank/iter={eu}", flush=True)
                print(f"{'':<22} mean_cos/iter={co}", flush=True)
                print(f"{'':<22} cotangent top3 block0={st['cotangent_top'][0]['top_idx'][:3]} "
                      f"share={st['cotangent_top'][0]['top_share'][:3]} "
                      f"block-agreement={st['top3_agreement_across_blocks']}", flush=True)
                continue
            if a.gap_probe:
                gp = spectral_gap(model, n_iter=a.power_iters)
                results.append({"ckpt": os.path.basename(path), "step": step, "gap": gp})
                gaps = sorted(gp.items(), key=lambda kv: -kv[1]["gap"])
                worst = gaps[0]
                med = sorted(v["gap"] for v in gp.values())[len(gp) // 2]
                print(f"{os.path.basename(path):<22} median gap={med:.3f} "
                      f"worst={worst[1]['gap']:.3f} ({worst[0]}) "
                      f"s1={worst[1]['sigma1']:.3f} s2={worst[1]['sigma2']:.3f}", flush=True)
                continue
            if a.rank_probe:
                rk = cotangent_rank(model, x, y, layout, a.seed, a.rank_token_path)
                row = {"ckpt": os.path.basename(path), "step": step, "cap": cap, "rank": rk}
                results.append(row)
                eff = rk["eff_positions_per_block"]
                print(f"{os.path.basename(path):<22} valid_slots={rk['mean_valid_slots']:.1f} "
                      f"eff_positions=" + " ".join(f"{eff[i]:.2f}" for i in sorted(eff)),
                      flush=True)
                continue
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
