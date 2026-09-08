"""Phase profile of ONE tul_oly_mask training step. Scratch script, no repo file touched.

Wraps model methods in CUDA-event timers (forward phases, exact), times the whole
backward, and optionally runs torch.profiler for the op-level backward split.
"""
from __future__ import annotations
import argparse, os, sys, json, time, statistics
import torch

ROOT = "/home/wolfe/morph-to"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/lab/divergence")
from _build import build_cfg, build_model            # noqa: E402


class Ev:
    def __init__(self):
        self.acc = {}
        self.pend = []
    def wrap(self, obj, name, tag):
        fn = getattr(obj, name)
        def inner(*a, **k):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            try:
                return fn(*a, **k)
            finally:
                e1.record(); self.pend.append((tag, e0, e1))
        setattr(obj, name, inner)
    def drain(self):
        torch.cuda.synchronize()
        for tag, e0, e1 in self.pend:
            self.acc[tag] = self.acc.get(tag, 0.0) + e0.elapsed_time(e1)
        self.pend.clear()
    def reset(self):
        self.acc = {}; self.pend = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tul_oly_mask")
    ap.add_argument("--micro", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--torch-profile", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ov = ["training.steps=60",
          f"curriculum.stages=[{{seq_len:512,context_len:512,micro_batch:{a.micro},steps:60}}]"]
    if a.overrides:
        ov += a.overrides.split(",")
    cfg = build_cfg(a.config, ov)
    torch.manual_seed(int(cfg.training.seed))
    model, tul_rt = build_model(cfg, "cuda")
    model.train()

    from morph.training.curriculum_data import MultiSourceCurriculumLoader
    from morph.training.data_placement import DataRuntimeConfig
    from morph.training.optimizer import create_optimizer
    cc = cfg.curriculum
    loader = MultiSourceCurriculumLoader(
        str(cc.pretok_dir), {str(k): float(v) for k, v in dict(cc.blend).items()},
        [int(s.seq_len) for s in cc.stages], seed=int(cfg.training.seed),
        allowed_roles=[str(x) for x in cc.allowed_source_roles],
        data_runtime=DataRuntimeConfig.resolve(getattr(cfg, "data_runtime", None)))
    it = loader.batches(a.micro, 0, tul_rt.data_cfg)
    opt = create_optimizer(model, cfg)

    ev = Ev()
    ev.wrap(model, "_tul_front", "fwd/front(embed+prelude)")
    ev.wrap(model, "_tul_core", "fwd/slot_loop")
    ev.wrap(model, "_back_region", "fwd/back(coda)")
    ev.wrap(model, "_tul_group_losses", "fwd/CE+heads")
    ev.wrap(model, "_slot_gain_penalty", "fwd/gain_hinge")
    if hasattr(model, "prefix_project"):
        ev.wrap(model, "prefix_project", "fwd/prefix_project")

    times = {"fwd": [], "bwd": [], "opt": [], "data": [], "step": []}
    phases = []
    prof = None
    for i in range(a.warmup + a.steps):
        if i == a.warmup:
            torch.cuda.reset_peak_memory_stats()
            if a.torch_profile:
                prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, with_stack=False)
                prof.__enter__()
        t0 = time.perf_counter()
        x, y, layout = next(it)
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); layout = layout.to("cuda")
        torch.cuda.synchronize(); t1 = time.perf_counter()
        ev.reset()
        e_f0 = torch.cuda.Event(enable_timing=True); e_f1 = torch.cuda.Event(enable_timing=True)
        e_b1 = torch.cuda.Event(enable_timing=True); e_o1 = torch.cuda.Event(enable_timing=True)
        e_f0.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y, slot_layout=layout)
        loss = out["loss"]
        e_f1.record()
        loss.backward()
        e_b1.record()
        opt.step(); opt.zero_grad(set_to_none=True)
        e_o1.record()
        ev.drain()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if i >= a.warmup:
            times["data"].append((t1 - t0) * 1e3)
            times["fwd"].append(e_f0.elapsed_time(e_f1))
            times["bwd"].append(e_f1.elapsed_time(e_b1))
            times["opt"].append(e_b1.elapsed_time(e_o1))
            times["step"].append((t2 - t0) * 1e3)
            phases.append(dict(ev.acc))
        print(f"  step {i} fwd {e_f0.elapsed_time(e_f1):.0f} bwd {e_f1.elapsed_time(e_b1):.0f} "
              f"opt {e_b1.elapsed_time(e_o1):.0f} loss {float(loss):.4f} "
              f"phases {({k: round(v,1) for k,v in ev.acc.items()})}", flush=True)
    if prof is not None:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35))
    res = {"config": a.config, "micro": a.micro, "overrides": a.overrides,
           "peak_alloc_GB": torch.cuda.max_memory_allocated() / 2**30,
           "peak_resv_GB": torch.cuda.max_memory_reserved() / 2**30}
    for k, v in times.items():
        res[k + "_ms_mean"] = round(statistics.mean(v), 1)
        res[k + "_ms_med"] = round(statistics.median(v), 1)
    keys = sorted({k for p in phases for k in p})
    res["phases_ms_mean"] = {k: round(statistics.mean([p.get(k, 0.0) for p in phases]), 1)
                             for k in keys}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
