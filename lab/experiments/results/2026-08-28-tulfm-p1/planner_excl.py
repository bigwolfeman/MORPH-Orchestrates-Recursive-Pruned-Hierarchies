"""Best planner (cfm_white 12k) rescored with the previous-span candidate excluded — apples to apples with the copy baseline."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/wolfe/morph-perf")
from omegaconf import OmegaConf
from lab.tulfm.train_p1 import build_backbone, make_loader
from morph.training.tul_setup import build_boundary_rule
from lab.tulfm.fm_planner import (FMPlanner, FMPlannerConfig, build_schedule,
                                  segment_rows, pool_targets, generate_plans)
from lab.tulfm.retrieval_probe import run_probe  # not used; loading pattern only

CK = "checkpoints/tulfm/p1c_cfm_white_12k/step_12000.pt"
blob = torch.load(CK, map_location="cpu", weights_only=False)
cfg = OmegaConf.create(blob["cfg"]); bcfg = OmegaConf.create(blob["backbone_cfg"])
device = torch.device("cuda")
backbone = build_backbone(cfg, bcfg, device)
rule, _l, _e, _s = build_boundary_rule(bcfg)
pcfg = FMPlannerConfig(**blob["planner_cfg"])
planner = FMPlanner(pcfg).to(device); planner.load_state_dict(blob["planner"]); planner.eval()
schedule = build_schedule(float(cfg.sigma.p_mean), float(cfg.sigma.p_std), float(cfg.sigma.sigma_data))
loader = make_loader(bcfg, 1024, 8, skip_samples=int(cfg.data.val_skip_samples))
torch.manual_seed(1234)
gen = torch.Generator(device=device).manual_seed(1234)

# whitener, if the blob carries one
whiten = blob.get("whitener")
def to_space(y):
    if whiten is None: return y
    W = torch.tensor(whiten["basis"], device=device).float()
    mu = torch.tensor(whiten["mu"], device=device).float()
    sc = torch.tensor(whiten["scales"], device=device).float()
    return (y - mu) @ W.T / sc
print("whitener present:", whiten is not None, list(blob.keys()))

ranks, cands = [], []
for _ in range(8):
    ids = next(loader)[0].to(device)
    geom = segment_rows(ids, rule, pcfg.max_slots)
    with torch.no_grad():
        h = backbone.prelude_states(ids, apply_input_norm=True).float()
        y_raw = pool_targets(h, geom)
        y = to_space(y_raw)
        zhat = generate_plans(planner, h, geom, schedule, int(cfg.sigma.infer_steps), generator=gen)
    B = y.shape[0]
    for b in range(B):
        vs = geom.valid[b].nonzero().flatten().tolist()
        k = F.normalize(y[b, vs].float(), dim=-1)
        q_all = F.normalize(zhat[b, vs].float(), dim=-1)
        idx = {s: j for j, s in enumerate(vs)}
        for s in vs:
            if s - 1 not in idx:   # same query population as the copy baseline
                continue
            sim = k @ q_all[idx[s]]
            allowed = torch.ones(len(vs), dtype=torch.bool, device=device)
            allowed[idx[s - 1]] = False
            gold = idx[s]
            r = ((sim >= sim[gold]) & allowed).sum().item()
            ranks.append(r); cands.append(int(allowed.sum().item()))
import statistics
rt = torch.tensor(ranks, dtype=torch.float); ct = torch.tensor(cands, dtype=torch.float)
print(f"queries={len(ranks)}  N={ct.mean():.1f}  chance={float((1/ct).mean()):.4f}")
print(f"top1={float((rt==1).float().mean()):.4f}  top5={float((rt<=5).float().mean()):.4f}  "
      f"mrr={float((1/rt).mean()):.4f}  med={statistics.median(ranks):.1f}")
