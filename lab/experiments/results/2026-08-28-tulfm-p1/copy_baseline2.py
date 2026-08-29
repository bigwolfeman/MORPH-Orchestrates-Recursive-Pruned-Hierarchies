"""Copy baseline, self-match excluded: candidate y_{i-1} (== the guess itself) is masked out."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/wolfe/morph-perf")
from omegaconf import OmegaConf
from lab.tulfm.train_p1 import build_backbone, make_loader
from morph.training.tul_setup import build_boundary_rule
from lab.tulfm.fm_planner import FMPlannerConfig, segment_rows, pool_targets

blob = torch.load("checkpoints/tulfm/p1/step_4000.pt", map_location="cpu", weights_only=False)
cfg = OmegaConf.create(blob["cfg"]); bcfg = OmegaConf.create(blob["backbone_cfg"])
device = torch.device("cuda")
backbone = build_backbone(cfg, bcfg, device)
rule, _l, _e, _s = build_boundary_rule(bcfg)
pcfg = FMPlannerConfig(**blob["planner_cfg"])
loader = make_loader(bcfg, 1024, 8, skip_samples=int(cfg.data.val_skip_samples))
torch.manual_seed(1234)

ranks, cands = [], []
for _ in range(8):
    ids = next(loader)[0].to(device)
    geom = segment_rows(ids, rule, pcfg.max_slots)
    with torch.no_grad():
        h = backbone.prelude_states(ids, apply_input_norm=True).float()
    y = pool_targets(h, geom)
    B, S, d = y.shape
    valid = geom.valid
    for b in range(B):
        vs = valid[b].nonzero().flatten().tolist()
        k = F.normalize(y[b, vs].float(), dim=-1)          # candidates: this row's targets
        idx = {s: j for j, s in enumerate(vs)}
        for s in vs:
            if s - 1 not in idx:            # need a previous span to copy
                continue
            q = k[idx[s - 1]]               # guess = pooled current span = y_{s-1}
            sim = k @ q
            allowed = torch.ones(len(vs), dtype=torch.bool, device=device)
            allowed[idx[s - 1]] = False     # exclude the self-match candidate
            gold = idx[s]
            r = ((sim >= sim[gold]) & allowed).sum().item()
            ranks.append(r); cands.append(int(allowed.sum().item()))
import statistics
ranks_t = torch.tensor(ranks, dtype=torch.float)
cands_t = torch.tensor(cands, dtype=torch.float)
print(f"queries={len(ranks)}  N={cands_t.mean():.1f}  chance={float((1/cands_t).mean()):.4f}")
print(f"top1={float((ranks_t==1).float().mean()):.4f}  top5={float((ranks_t<=5).float().mean()):.4f}  "
      f"mrr={float((1/ranks_t).mean()):.4f}  med={statistics.median(ranks):.1f}")
