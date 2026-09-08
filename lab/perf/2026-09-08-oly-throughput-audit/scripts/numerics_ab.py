"""In-process A/B: fused kernels ON vs OFF on the SAME tul_oly_mask weights and batch."""
import sys, json, torch
sys.path.insert(0, "/home/wolfe/morph-to"); sys.path.insert(0, "/home/wolfe/morph-to/lab/divergence")
from _build import build_cfg, build_model
from morph.kernels.triton._eager_flag import set_force_eager

MICRO = 6
cfg = build_cfg("tul_oly_mask", ["training.steps=60",
    f"curriculum.stages=[{{seq_len:512,context_len:512,micro_batch:{MICRO},steps:60}}]"])
torch.manual_seed(int(cfg.training.seed))
model, tul_rt = build_model(cfg, "cuda")
model.train()
from morph.training.curriculum_data import MultiSourceCurriculumLoader
from morph.training.data_placement import DataRuntimeConfig
cc = cfg.curriculum
loader = MultiSourceCurriculumLoader(str(cc.pretok_dir),
    {str(k): float(v) for k, v in dict(cc.blend).items()},
    [int(s.seq_len) for s in cc.stages], seed=int(cfg.training.seed),
    allowed_roles=[str(x) for x in cc.allowed_source_roles],
    data_runtime=DataRuntimeConfig.resolve(getattr(cfg, "data_runtime", None)))
it = loader.batches(MICRO, 0, tul_rt.data_cfg)
x, y, layout = next(it)
x, y, layout = x.cuda(), y.cuda(), layout.to("cuda")

def run(eager: bool):
    set_force_eager(eager)
    torch.manual_seed(1234)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=y, slot_layout=layout)
    print("KEYS", sorted([k for k,v in out.items() if torch.is_tensor(v) and v.numel()==1]))
    d = {k: float(v.detach()) for k, v in out.items()
         if torch.is_tensor(v) and v.numel() == 1}
    # gradient of the CE only (drop the gain penalty so the two are comparable)
    model.zero_grad(set_to_none=True)
    (out.get("ce_main") if out.get("ce_main") is not None else out["loss"]).backward()
    gn = {n: p.grad.detach().float().norm().item()
          for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    del out
    torch.cuda.empty_cache()
    return d, gn

a, ga = run(True)     # eager: what the config as-shipped runs
b, gb = run(False)    # fused: what +model.tg_scoped_kernels=true runs
print("EAGER ", json.dumps({k: round(v, 6) for k, v in a.items()}))
print("FUSED ", json.dumps({k: round(v, 6) for k, v in b.items()}))
keys = sorted(set(a) & set(b))
print("\ndelta (fused - eager):")
for k in keys:
    print(f"  {k:24s} {a[k]:14.6f} -> {b[k]:14.6f}  d={b[k]-a[k]:+.6f}")
rel = [(abs(gb[n]-ga[n])/max(ga[n],1e-12), n) for n in ga if n in gb]
rel.sort(reverse=True)
print("\nCE-grad-norm relative diff, worst 8 params:")
for r, n in rel[:8]:
    print(f"  {r:9.4%}  {n}  {ga[n]:.6g} -> {gb[n]:.6g}")
import statistics
print("median rel grad-norm diff:", f"{statistics.median([r for r,_ in rel]):.4%}")

# repeatability of the gain reading WITHIN one mode (is 1.03 vs 0.90 above noise?)
for mode, name in ((True, "eager"), (False, "fused")):
    vals = []
    for s in range(5):
        set_force_eager(mode); torch.manual_seed(1234 + s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, labels=y, slot_layout=layout)
        vals.append(float(o["gain_est"].detach()))
        model.zero_grad(set_to_none=True)
    print(f"gain_est {name}: {[round(v,4) for v in vals]}")
