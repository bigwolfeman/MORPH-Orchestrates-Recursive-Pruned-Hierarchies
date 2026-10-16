"""Throughput: plain Parcae vs Parcae-LXTUL on MORPH's real rows (lxtul/configs/bench.yaml).

Each arm: build, compile, load `n_batches` real batches onto the GPU, run `warmup_steps`
full training steps (forward, backward, clip, MuonAdamW step, EMA twin update), then time
`measured_steps`. Wall time per step is the number reported (it includes host syncs, as a
training loop sees them). tok/s = batch * data.seq_len / median wall step, MORPH's formula.
Raises if Dynamo compiles inside the timed window, or if any loss is non-finite.

    PYTHONPATH=.:$MORPH_ROOT python -m lxtul.bench
"""
from __future__ import annotations

import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from lxtul.build import build, make_optimizer
from lxtul.data import make_loader


def _dynamo_counts():
    return {k: dict(v) for k, v in torch._dynamo.utils.counters.items() if v}


def _batches(mcfg, rt, n: int, bs: int):
    it = make_loader(mcfg, "train", bs, tul_rt=rt, prefetch=0)
    out = []
    for _ in range(n):
        b = next(it)
        if rt is None:
            x, y = b
            out.append((x.cuda(), y.cuda(), None))
        else:
            x, y, lay = b
            for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
                setattr(lay, f, getattr(lay, f).cuda())
            out.append((x.cuda(), y.cuda(), lay))
    return out


def _step(model, opt, batch, arm: str, clip: float):
    x, y, lay = batch
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=y) if arm == "plain" else model(x, y, lay)
    loss = out["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    if arm == "lxtul":
        model.ema_update()
    return loss.detach(), out.get("ce", loss).detach()


def bench_arm(cfg: DictConfig, arm: str) -> dict:
    torch.manual_seed(int(cfg.run.seed))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch._dynamo.config.recompile_limit = 64
    torch.cuda.empty_cache()
    model, mcfg, rt = build(cfg, arm)
    opt = make_optimizer(model, cfg)
    bs = int(cfg.bench.batch_size)
    batches = _batches(mcfg, rt, int(cfg.bench.n_batches), bs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.perf_counter()
    for i in range(int(cfg.bench.warmup_steps)):
        loss, ce = _step(model, opt, batches[i % len(batches)], arm, cfg.bench.grad_clip)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{arm}: non-finite loss {loss.item()} at warmup step {i}")
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0
    before = _dynamo_counts()
    torch.cuda.reset_peak_memory_stats()
    walls, losses, ces = [], [], []
    for i in range(int(cfg.bench.measured_steps)):
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss, ce = _step(model, opt, batches[i % len(batches)], arm, cfg.bench.grad_clip)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t)
        losses.append(loss)
        ces.append(ce)
    after = _dynamo_counts()
    if after.get("stats", {}).get("unique_graphs") != before.get("stats", {}).get("unique_graphs"):
        raise RuntimeError(f"{arm}: Dynamo compiled inside the timed window: {before} -> {after}")
    losses = torch.stack(losses).float().cpu().tolist()
    if not all(map(lambda v: v == v and abs(v) < 1e9, losses)):
        raise RuntimeError(f"{arm}: non-finite loss in the timed window")
    seq = int(mcfg.data.seq_len)
    med = statistics.median(walls)
    res = {"arm": arm, "params": n_params, "median_step_s": med, "mean_step_s": sum(walls) / len(walls),
           "tok_s": bs * seq / med, "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
           "warmup_s": warm_s, "loss_first": losses[0], "loss_last": losses[-1],
           "ce_last": float(ces[-1]), "batch": bs, "seq_len": seq}
    if rt is not None:
        res["lxtul_config"] = dataclasses.asdict(model.tul)
        res["valid_slots_mean"] = float(sum(b[2].slot_valid.float().sum(1).mean() for b in batches) / len(batches))
    del model, opt, batches
    return res


@hydra.main(config_path="configs", config_name="bench", version_base=None)
def main(cfg: DictConfig) -> None:
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    power = os.popen("nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits").read().strip()
    run = wandb.init(project=cfg.run.wandb_project, name=cfg.run.name, mode=cfg.run.wandb_mode,
                     config=OmegaConf.to_container(cfg, resolve=True) | {"gpu_power_limit_w": power},
                     dir=str(out_dir))
    results = []
    for arm in cfg.bench.arms:
        r = bench_arm(cfg, str(arm))
        print(json.dumps(r), flush=True)
        run.log({f"{arm}/{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        if "lxtul_config" in r:
            run.config.update({"lxtul_resolved": r["lxtul_config"]})
        results.append(r)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
