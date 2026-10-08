"""Train one testbed arm on MORPH's stream, then sweep depth on 480 held-out rows.

    PYTHONPATH=/home/wolfe/parcae:$MORPH_ROOT python -m lxtul.train arm=plain

Recipe: Parcae's (MuonAdamW, lr 0.008 * sqrt(batch/256), constant then linear cooldown over
the last half, Muon momentum warmup over 300 steps, Muon wd decayed to 0, grad clip 1.0), plus a linear LR warmup
over `training.warmup_steps` (Wolfe 2026-10-06: Parcae's recipe has none).
Logs the full resolved config, the resolved LXTULConfig and the git commit to wandb.
Writes `<out_root>/<run.name>/`: metrics.jsonl, final.pt (model state), sweep.json (per-row
sums per depth), summary.json. A non-finite loss or grad norm raises.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from experiments.runtime import apply_schedule, schedule
from lxtul.build import build, make_optimizer
from lxtul.data import make_loader, morph_cfg, tul_runtime
from lxtul.evaluate import depth_sweep, eval_rows, summarize

ROOT = Path(__file__).resolve().parents[1]


def _to_cuda(batch, kind: str):
    if kind in ("plain", "gpt"):
        x, y = batch
        return x.cuda(non_blocking=True), y.cuda(non_blocking=True), None
    x, y, lay = batch
    for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
        setattr(lay, f, getattr(lay, f).cuda(non_blocking=True))
    return x.cuda(non_blocking=True), y.cuda(non_blocking=True), lay


def _forward(model, kind, x, y, lay):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x, labels=y) if kind in ("plain", "gpt") else model(x, y, lay)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    t, kind = cfg.training, str(cfg.arm.kind)
    out = Path(t.out_root) / str(cfg.run.name)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"{out} already holds a finished run")
    torch.manual_seed(int(cfg.run.seed))
    # every Parcae block shares one forward code object; ve tensor/None x grad/no-grad x
    # train/eval exceeds Dynamo's default limit of 8 (Parcae's own trainer uses 64)
    torch._dynamo.config.recompile_limit = int(t.recompile_limit)
    torch._dynamo.config.accumulated_recompile_limit = int(t.recompile_limit) * 4
    overrides = OmegaConf.to_container(cfg.arm.lxtul, resolve=True) if "lxtul" in cfg.arm else {}
    model, mcfg, rt = build(cfg, kind, overrides=overrides, compile=bool(t.compile))
    eval_rt = rt if rt is not None else tul_runtime(morph_cfg(str(cfg.morph.config)))
    slot_id = int(eval_rt.data_cfg.slot_id)
    opt = make_optimizer(model, cfg)
    bs, steps, seq = int(t.batch_size), int(t.steps), int(mcfg.data.seq_len)
    sched_cfg = {"training": OmegaConf.to_container(t, resolve=True),
                 "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True)}
    rows = eval_rows(mcfg, eval_rt, int(cfg.eval.rows), int(cfg.eval.batch_size))
    val_rows = rows[: int(t.val_rows) // int(cfg.eval.batch_size)]
    eval_depth = 6
    resolved = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(project=cfg.run.wandb_project, name=cfg.run.name, mode=cfg.run.wandb_mode,
                     dir=str(out), config=resolved | {
                         "lxtul_resolved": dataclasses.asdict(model.tul) if kind == "lxtul" else None,
                         "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                         "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "lxtul"], cwd=ROOT, text=True).strip()),
                         "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                         "gpu_power_limit_w": os.popen("nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits").read().strip()})
    (out / "config.json").write_text(json.dumps(dict(run.config), indent=2, default=str))
    loader = make_loader(mcfg, "train", bs, tul_rt=rt, prefetch=int(t.prefetch))
    mlog = (out / "metrics.jsonl").open("a")
    window, t_win = [], time.perf_counter()
    open_at = int(model.tul.open_at) if kind == "lxtul" else -1
    for step in range(steps):
        if step == open_at:                    # strict -> open geometry (LXTULConfig.open_at)
            model.tul = dataclasses.replace(model.tul, geometry="open")
            print(f"[geometry] step {step}: strict -> open", flush=True)
        x, y, lay = _to_cuda(next(loader), kind)
        if kind == "plain":
            model.step = step
        vals = schedule(sched_cfg, bs, step)
        warm = min(1.0, (step + 1) / int(t.warmup_steps)) if int(t.warmup_steps) > 0 else 1.0
        vals["lr"], vals["muon_lr"] = vals["lr"] * warm, vals["muon_lr"] * warm
        apply_schedule(opt, vals)
        opt.zero_grad(set_to_none=True)
        o = _forward(model, kind, x, y, lay)
        loss = o["loss"]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.grad_clip))
        opt.step()
        if kind == "lxtul":
            model.ema_update()
        window.append((loss.detach(), o.get("ce", loss).detach(), gn.detach()))
        done = step + 1
        if done % int(t.log_every) == 0 or done == 1:
            ls, cs, gs = (torch.stack(v).float() for v in zip(*window))
            if not (torch.isfinite(ls).all() and torch.isfinite(gs).all()):
                raise FloatingPointError(f"non-finite loss or grad norm by step {done}")
            dt = time.perf_counter() - t_win
            m = {"step": done, "train/loss": float(ls.mean()), "train/ce": float(cs.mean()),
                 "train/grad_norm": float(gs.mean()), "perf/tok_s": len(window) * bs * seq / dt,
                 "perf/peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                 **{f"sched/{k}": v for k, v in vals.items()}}
            if kind == "lxtul":
                m |= {f"tul/{k}": float(v) for k, v in model.metrics.items()}
                if "spandec" in o:
                    m["train/spandec"] = float(o["spandec"])
                if "ce_model" in o:                 # copy-cache arms: the model's own CE
                    m["train/ce_model"] = float(o["ce_model"])
            run.log(m, step=done)
            mlog.write(json.dumps(m) + "\n"); mlog.flush()
            print(json.dumps(m), flush=True)
            window, t_win = [], time.perf_counter()
        if done % int(t.val_every) == 0 or done == steps:
            t_val = time.perf_counter()
            sw = depth_sweep(model, kind, val_rows, [eval_depth], slot_id)
            v = sum(sw["sums"][str(eval_depth)]) / sum(sw["count"])
            run.log({"val/ce_d6": v}, step=done)
            mlog.write(json.dumps({"step": done, "val/ce_d6": v}) + "\n"); mlog.flush()
            print(f"[val] step {done} ce_d6 {v:.4f}", flush=True)
            t_win += time.perf_counter() - t_val          # val time is not training time
    if t.save_final:
        torch.save({"model": model.state_dict(), "step": steps, "config": resolved}, out / "final.pt")
    sweep = depth_sweep(model, kind, rows, list(cfg.eval.depths), slot_id)
    (out / "sweep.json").write_text(json.dumps(sweep))
    if open_at >= 0:
        # the switched arm, scored back under the strict mask: how much of the loop survived
        model.tul = dataclasses.replace(model.tul, geometry="strict")
        sw_s = depth_sweep(model, kind, rows, list(cfg.eval.depths), slot_id)
        (out / "sweep_strict.json").write_text(json.dumps(sw_s))
        model.tul = dataclasses.replace(model.tul, geometry="open")
    summ = summarize(sweep, ref=eval_depth)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    for k, ce in summ["ce"].items():
        run.summary[f"sweep/ce_d{k}"] = ce
    k1 = summ["minus_ref"]["1"]
    run.summary["sweep/K1_minus_K6"] = k1["point"]
    print(f"[sweep] ce {summ['ce']}  K1-K6 {k1['point']:+.4f} [{k1['lo']:+.4f}, {k1['hi']:+.4f}]", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
