"""Leak-or-anisotropy diagnostic for the band-0 rel fall (R6's unanticipated cell).

Band 0's preconditioned target is ~ -eps. The pre-reg said a fall there "means a leak".
The rival explanation: targets are rank-~25, so on the ~999 quasi-dead dims eps is
recoverable from z alone (z_j = y_j + sigma*eps_j with y_j ~ deterministic).

Discriminator: split the F-space squared error along the target PCA subspace.
  LEAK       -> error falls in BOTH subspaces (model predicts eps everywhere).
  ANISOTROPY -> error falls ONLY on the dead complement; live-subspace rel stays ~1.
"""
import sys, torch
sys.path.insert(0, "/home/wolfe/morph-perf")
from omegaconf import OmegaConf
from lab.tulfm.train_p1 import build_backbone, make_loader
from morph.training.tul_setup import build_boundary_rule
from lab.tulfm.fm_planner import (FMPlanner, FMPlannerConfig, build_schedule,
                                  segment_rows, pool_targets, build_masks)

CKPT = "checkpoints/tulfm/p1/step_4000.pt"
device = torch.device("cuda")
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = OmegaConf.create(blob["cfg"]); bcfg = OmegaConf.create(blob["backbone_cfg"])
backbone = build_backbone(cfg, bcfg, device)
rule, _l, _e, _s = build_boundary_rule(bcfg)
pcfg = FMPlannerConfig(**blob["planner_cfg"])
planner = FMPlanner(pcfg).to(device); planner.load_state_dict(blob["planner"]); planner.eval()
schedule = build_schedule(float(cfg.sigma.p_mean), float(cfg.sigma.p_std), float(cfg.sigma.sigma_data))
loader = make_loader(bcfg, 1024, 8, skip_samples=int(cfg.data.val_skip_samples))
torch.manual_seed(7)

# pass 1: collect targets for the PCA basis
ys, batches = [], []
for _ in range(6):
    ids = next(loader)[0].to(device)
    geom = segment_rows(ids, rule, pcfg.max_slots)
    with torch.no_grad():
        h = backbone.prelude_states(ids, apply_input_norm=True).float()
    y = pool_targets(h, geom)
    ys.append(y[geom.valid]); batches.append((h, geom))
Y = torch.cat(ys)                                  # [N, 1024]
mu = Y.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(Y - mu, full_matrices=False)
K = 25
P_live = Vh[:K]                                    # [25, 1024]
var_live = (S[:K]**2).sum() / (S**2).sum()
print(f"targets N={Y.shape[0]}  top-{K} PCs carry {var_live:.4f} of variance")

# pass 2: band-0 sigma, same-eps null, subspace split of the F-space error
lo, hi = 0.002, 0.094
tot = {"live_err": 0., "dead_err": 0., "live_null": 0., "dead_null": 0., "n": 0.}
gen = torch.Generator(device=device).manual_seed(11)
for h, geom in batches:
    y = pool_targets(h, geom)
    B, Ssl, d = y.shape
    u = torch.rand(B, Ssl, device=device, generator=gen)
    import math
    sigma = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))   # log-uniform in band 0
    eps = torch.randn(y.shape, device=device, generator=gen)
    z = y + sigma[..., None] * eps
    ctx = planner.encode_ctx(h)
    sm, cm = build_masks(geom)
    with torch.no_grad():
        d_hat = planner.denoise(z, sigma, ctx, geom, sm, cm).float()
        c_skip, _, _, _ = planner.precond.coeffs(sigma)
        d_null = c_skip[..., None] * z
        w = planner.precond.weight(sigma)
        for name, D in (("err", d_hat), ("null", d_null)):
            r = D - y                                            # [B,S,1024]
            r_live = r @ P_live.T                                # [B,S,25]
            live = (r_live**2).sum(-1)
            dead = (r**2).sum(-1) - live
            v = geom.valid.float()
            tot[f"live_{name}"] += (w * live * v).sum().item()
            tot[f"dead_{name}"] += (w * dead * v).sum().item()
        tot["n"] += geom.valid.float().sum().item()

rl = tot["live_err"] / max(tot["live_null"], 1e-9)
rd = tot["dead_err"] / max(tot["dead_null"], 1e-9)
print(f"band0  live-subspace rel = {rl:.4f}   (leak predicts <<1, anisotropy predicts ~1)")
print(f"band0  dead-subspace rel = {rd:.4f}   (anisotropy predicts <<1)")
print(f"overall band0 rel check  = {(tot['live_err']+tot['dead_err'])/(tot['live_null']+tot['dead_null']):.4f}")
