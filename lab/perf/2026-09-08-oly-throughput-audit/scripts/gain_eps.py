"""Is the eager-vs-fused gain gap a real map difference or finite-difference noise?
Sweep slot_gain_eps in both kernel modes on the SAME weights and batch."""
import sys, torch, statistics
sys.path.insert(0, "/home/wolfe/morph-to"); sys.path.insert(0, "/home/wolfe/morph-to/lab/divergence")
from _build import build_cfg, build_model
from morph.kernels.triton._eager_flag import set_force_eager
MICRO = 6
cfg = build_cfg("tul_oly_mask", ["training.steps=60",
    f"curriculum.stages=[{{seq_len:512,context_len:512,micro_batch:{MICRO},steps:60}}]"])
torch.manual_seed(int(cfg.training.seed))
model, tul_rt = build_model(cfg, "cuda"); model.train()
from morph.training.curriculum_data import MultiSourceCurriculumLoader
from morph.training.data_placement import DataRuntimeConfig
cc = cfg.curriculum
loader = MultiSourceCurriculumLoader(str(cc.pretok_dir),
    {str(k): float(v) for k, v in dict(cc.blend).items()},
    [int(s.seq_len) for s in cc.stages], seed=int(cfg.training.seed),
    allowed_roles=[str(x) for x in cc.allowed_source_roles],
    data_runtime=DataRuntimeConfig.resolve(getattr(cfg, "data_runtime", None)))
it = loader.batches(MICRO, 0, tul_rt.data_cfg)
x, y, layout = next(it); x, y, layout = x.cuda(), y.cuda(), layout.to("cuda")
print("cfg slot_gain_eps", model.cfg.slot_gain_eps, "target", model.cfg.slot_gain_target)
print(f"{'eps':>6} {'eager mean':>11} {'eager sd':>9} {'fused mean':>11} {'fused sd':>9} {'delta':>8}")
for eps in (0.02, 0.05, 0.1, 0.2, 0.4):
    model.cfg.slot_gain_eps = eps
    res = {}
    for mode, nm in ((True, "eager"), (False, "fused")):
        vals = []
        for s in range(5):
            set_force_eager(mode); torch.manual_seed(1234 + s)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pass
            torch.manual_seed(1234 + s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x, labels=y, slot_layout=layout)
            vals.append(float(o["gain_est"].detach()))
            del o; model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        res[nm] = vals
    e, f = res["eager"], res["fused"]
    print(f"{eps:6.2f} {statistics.mean(e):11.4f} {statistics.pstdev(e):9.4f} "
          f"{statistics.mean(f):11.4f} {statistics.pstdev(f):9.4f} "
          f"{statistics.mean(f)-statistics.mean(e):+8.4f}")
