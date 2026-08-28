"""TUL-FM Phase 1 — train the flow-matching planner on a FROZEN backbone.

Arc:     ``.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md``
Gates:   ``docs/tul-fm-probing.md`` §5 (P1)
Pre-reg: ``lab/experiments/planned/2026-08-28-tulfm-p1.md``
Config:  ``morph/configs/tulfm_p1.yaml``

    PYTHONPATH=$PWD python -m lab.tulfm.train_p1

The backbone is loaded, put in ``eval()``, and every one of its parameters has
``requires_grad`` set False — asserted, not assumed, and asserted AGAIN against the
optimizer's own parameter set so a future edit cannot quietly start training it. Only the
planner is checkpointed.

WHY THE BACKBONE'S CONFIG IS COMPOSED RATHER THAN COPIED. ``backbone.config_name`` names
a Hydra config in ``morph/configs``; it supplies the architecture, the tokenizer, the
dataset and the whole ``tul:`` span rule. Duplicating any of those into
``tulfm_p1.yaml`` would let the planner read features from one tokenizer while cutting
spans with another — a failure that produces a perfectly plausible loss curve and a
meaningless probe.
"""

from __future__ import annotations

import math
import os
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from lab.tulfm.fm_planner import (
    FMPlanner,
    FMPlannerConfig,
    band_edges,
    build_schedule,
    effective_rank,
    fm_loss,
    generate_plans,
    mean_pairwise_cos,
    pool_targets,
    segment_rows,
)
from lab.tulfm.retrieval_probe import retrieval_scores, row_index_of_valid
from morph.model.transformer import MORPHTransformer
from morph.training.data import create_dataloader
from morph.training.train import build_morph_config, load_weights_only
from morph.training.tul_setup import build_boundary_rule

__all__ = ["build_backbone", "make_loader", "compose_backbone_cfg", "main"]


# ── backbone ─────────────────────────────────────────────────────────────────

def compose_backbone_cfg(config_name: str, device: torch.device,
                         extra: list | None = None) -> DictConfig:
    """Compose the frozen model's own Hydra config.

    On CPU the Triton/CUDA kernel paths cannot run, so ``use_kernels`` and
    ``hc_use_kernel`` are forced off. That is a DIFFERENT numeric path from the GPU run
    and it is stated in the log rather than hidden: CPU here exists for the test suite and
    for smoke runs, never for a result.

    ``extra`` (``backbone.overrides`` in the config) are Hydra overrides applied to the
    BACKBONE's config. They exist so a smoke run can shrink the frozen model without
    editing a shipped arm file. Anything set here also lands in the wandb config, because
    the whole resolved backbone config is logged.
    """
    from hydra import compose

    overrides = [str(o) for o in (extra or [])]
    if device.type == "cpu":
        overrides = overrides + ["model.use_kernels=false", "model.hc_use_kernel=false"]
        print("[backbone] CPU: forcing model.use_kernels=false, hc_use_kernel=false "
              "(Triton kernels are CUDA-only). This is NOT the GPU numeric path.")
    if overrides:
        print(f"[backbone] compose overrides: {overrides}")
    return compose(config_name=config_name, overrides=overrides)


def build_backbone(cfg: DictConfig, bcfg: DictConfig,
                   device: torch.device) -> MORPHTransformer:
    """Build the frozen model, load its weights COMPLETELY, freeze it, and prove both.

    ``apply_quantization`` is not optional and it must run BEFORE the load. Every QAT
    transform registers a ``torch.nn.utils.parametrize`` hook, which renames the affected
    tensor in ``state_dict`` (``w.weight`` -> ``w.parametrizations.weight.original``). A
    checkpoint written by a QAT run therefore only loads into a model that has had the
    same transforms applied, in the same order — ``morph/training/quant_setup.py`` says
    so in its own docstring, and a sampling script already paid for ignoring it once
    (2026-08-18, 45 missing / 45 unexpected).

    MEASURED on ``a3-s2/step_4500`` while building this: without it, 27 of 348 tensors
    fail to load — the euclidean embedding table (which is also the tied LM head), the
    bigram table, all 8 MLP ``gate_up``/``down`` banks, all 8 ``x0_injects`` projections
    and ``lm_mixer``. ``load_weights_only``'s own guard passes, because it counts tensors
    and 321/348 is comfortably over half. The features would have been half random and
    every P1 number meaningless. Hence the hard ``missing == 0`` check below.
    """
    from morph.training.quant_setup import apply_quantization

    morph_cfg = build_morph_config(bcfg, tul=None)
    model = MORPHTransformer(morph_cfg).to(device)
    qm = apply_quantization(model, bcfg)
    live_q = [k for k, v in qm.items() if v is not None]
    print(f"[backbone] quantization transforms applied: {live_q or 'none'}")

    ckpt = cfg.backbone.get("checkpoint", None)
    if ckpt:
        missing, unexpected = load_weights_only(str(ckpt), model, device)
        if missing:
            raise RuntimeError(
                f"{len(missing)} tensors did NOT load from {ckpt} — the frozen backbone "
                f"would be part random and every P1 number meaningless. First 5: "
                f"{list(missing)[:5]}. Almost always a QAT/parametrize mismatch: the "
                f"checkpoint's config and `backbone.config_name` must agree on "
                f"training.ternary / embed_quant / lm_head_quant / attn_proj_quant.")
        if unexpected:
            print(f"[backbone] {len(unexpected)} unexpected tensors in the checkpoint "
                  f"(ignored): {list(unexpected)[:5]}")
    else:
        print("[backbone] checkpoint is null → RANDOMLY INITIALISED backbone. "
              "Smoke path only; a random backbone's features are not a plan target.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"backbone is not frozen: {len(live)} params still require "
                           f"grad (first: {live[0]})")
    return model


def make_loader(bcfg: DictConfig, seq_len: int, batch_size: int, skip_samples: int = 0):
    """The ordinary MORPH OWT loader. No TUL packing — P1 segments rows itself."""
    return create_dataloader(
        tokenizer_name=str(bcfg.data.tokenizer),
        dataset_name=str(bcfg.data.dataset),
        seq_len=int(seq_len),
        batch_size=int(batch_size),
        split="train",
        skip_samples=int(skip_samples),
    )


# ── schedule ─────────────────────────────────────────────────────────────────

def lr_at(step: int, total: int, base_lr: float, warmup: int, min_frac: float) -> float:
    """Linear warmup → cosine decay to ``min_frac * base_lr``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    t = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return base_lr * (min_frac + (1.0 - min_frac) * cos)


# ── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(planner: FMPlanner, backbone: MORPHTransformer, loader, rule, cfg: DictConfig,
             schedule, edges, max_slots: int, device: torch.device,
             n_batches: int, generator: torch.Generator) -> dict:
    """Held-out FM loss, per-band loss, retrieval top-1, and the target effective rank.

    Retrieval runs at EVERY eval, not only at the end — the arc note names target gaming
    as the risk the loss curve cannot see, and a probe that only runs once cannot catch a
    target space that collapsed at step 900 and recovered by step 4000.
    """
    planner.eval()
    tot_loss, tot_top1, tot_mrr, tot_rank, tot_chance, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    tot_rel, tot_wtop1, tot_wchance = 0.0, 0.0, 0.0
    bands: dict[str, list[float]] = {}
    for _ in range(n_batches):
        ids = next(loader)[0].to(device)
        geom = segment_rows(ids, rule, max_slots)
        if int(geom.valid.sum()) < 2:
            continue
        h = backbone.prelude_states(
            ids, apply_input_norm=bool(cfg.backbone.apply_input_norm)).float()
        y = pool_targets(h, geom)
        loss, stats = fm_loss(planner, h, geom, schedule, generator=generator,
                              edges=edges, y=y)
        z = generate_plans(planner, h, geom, schedule,
                           n_steps=int(cfg.sigma.infer_steps), generator=generator)
        r = retrieval_scores(z, y, geom.valid)
        rw = retrieval_scores(z, y, geom.valid, row_of=row_index_of_valid(geom.valid))

        tot_loss += float(loss.item())
        tot_top1 += r["top1"]
        tot_mrr += r["mrr"]
        tot_rank += r["median_rank"]
        tot_chance += r["chance"]
        tot_rel += stats["rel_loss"]
        tot_wtop1 += rw["top1"]
        tot_wchance += rw["chance"]
        for k, v in stats.items():
            if k.startswith("band"):
                bands.setdefault(k, []).append(v)
        n += 1
        last_rank = effective_rank(y, geom.valid)
        last_cos = mean_pairwise_cos(y, geom.valid)

    planner.train()
    if n == 0:
        raise RuntimeError("eval produced no usable batch (fewer than 2 valid slots in "
                           "every one) — the span rule or the data is wrong")
    out = {
        "val/fm_loss": tot_loss / n,
        "val/rel_loss": tot_rel / n,
        "val/retrieval_top1": tot_top1 / n,
        "val/retrieval_mrr": tot_mrr / n,
        "val/retrieval_median_rank": tot_rank / n,
        "val/retrieval_chance": tot_chance / n,
        "val/retrieval_top1_within_row": tot_wtop1 / n,
        "val/retrieval_chance_within_row": tot_wchance / n,
        "val/target_effective_rank": last_rank,
        "val/target_mean_pairwise_cos": last_cos,
    }
    for k, v in bands.items():
        out[f"val/{k}"] = sum(v) / len(v)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../morph/configs", config_name="tulfm_p1", version_base=None)
def main(cfg: DictConfig) -> None:
    tr = cfg.training
    torch.manual_seed(int(tr.seed))

    device = torch.device(str(tr.device) if torch.cuda.is_available()
                          or str(tr.device) == "cpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but no CUDA device is visible")
    use_amp = bool(tr.amp) and device.type == "cuda"
    print(f"[p1] device={device} amp_bf16={use_amp} seed={int(tr.seed)}")

    # ── frozen backbone + its own config (architecture, tokenizer, data, span rule) ──
    bcfg = compose_backbone_cfg(str(cfg.backbone.config_name), device,
                                list(cfg.backbone.get("overrides", []) or []))
    backbone = build_backbone(cfg, bcfg, device)
    rule, lut, eos_id, substrings = build_boundary_rule(bcfg)
    d_ctx = int(bcfg.model.d_model)
    seq_len = int(cfg.data.seq_len)
    max_slots = int(cfg.data.max_slots) or (seq_len // rule.min_span)
    print(f"[p1] backbone d_model={d_ctx} n_core={int(bcfg.model.n_core)} "
          f"seq_len={seq_len} max_slots={max_slots} "
          f"rule(min_span={rule.min_span}, span_cap={rule.span_cap}, |B|={int(lut.sum())})")

    # ── planner ──────────────────────────────────────────────────────────
    pcfg = FMPlannerConfig(
        d_ctx=d_ctx, d_p=int(cfg.planner.d_p), n_layers=int(cfg.planner.n_layers),
        n_heads=int(cfg.planner.n_heads), d_ff=int(cfg.planner.d_ff),
        cond_dim=int(cfg.planner.cond_dim), max_slots=max_slots, max_ctx_len=seq_len,
        sigma_data=float(cfg.sigma.sigma_data), dropout=float(cfg.planner.dropout),
    )
    planner = FMPlanner(pcfg).to(device)
    n_p = planner.n_params()
    lo, hi = float(cfg.planner.min_params_m) * 1e6, float(cfg.planner.max_params_m) * 1e6
    if not lo <= n_p <= hi:
        raise ValueError(
            f"planner has {n_p/1e6:.2f}M params, outside the declared P1 band "
            f"[{lo/1e6:.1f}, {hi/1e6:.1f}]M. Either the dims changed or the band did; "
            f"say which in the config rather than letting the size drift.")
    print(f"[p1] planner: {n_p/1e6:.2f}M params (d_p={pcfg.d_p} x{pcfg.n_layers} layers)")

    # ── the freeze, proven against the optimizer's own parameter set ─────
    backbone_ids = {id(p) for p in backbone.parameters()}
    planner_params = list(planner.parameters())
    leaked = [i for i, p in enumerate(planner_params) if id(p) in backbone_ids]
    if leaked:
        raise RuntimeError(f"{len(leaked)} backbone tensors reached the planner optimizer")
    opt = torch.optim.AdamW(planner_params, lr=float(tr.lr),
                            betas=tuple(float(b) for b in tr.betas),
                            weight_decay=float(tr.weight_decay))

    # ── σ machinery ──────────────────────────────────────────────────────
    schedule = build_schedule(float(cfg.sigma.p_mean), float(cfg.sigma.p_std),
                              float(cfg.sigma.sigma_data))
    edges = band_edges(schedule, int(cfg.sigma.n_bands)).to(device)
    sigma_manifest = {
        "sigma/band_edges_ascending": [float(x) for x in edges.tolist()],
        "sigma/inference_ladder_descending":
            [float(x) for x in schedule.inference_sigmas(int(cfg.sigma.infer_steps))],
        **{k: v for k, v in schedule.manifest().items()
           if k.split("/")[-1] in ("sigma_min", "sigma_max", "p_mean", "p_std",
                                   "sigma_data", "overlap_gamma", "n_blocks")},
    }
    print(f"[p1] sigma bands (ascending): "
          f"{[round(float(x), 4) for x in edges.tolist()]}")

    # ── wandb: the FULL resolved config, both halves, plus every derived number ──
    full_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    wandb.init(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.get("entity", None),
        name=cfg.wandb.get("name", None),
        mode=str(cfg.wandb.get("mode", "online")),
        config={
            **full_cfg,
            "backbone_cfg": OmegaConf.to_container(bcfg, resolve=True,
                                                   throw_on_missing=False),
            "derived": {
                "planner_params": n_p,
                "planner_cfg": {k: getattr(pcfg, k) for k in pcfg.__dataclass_fields__},
                "d_ctx": d_ctx,
                "max_slots": max_slots,
                "boundary_ids": int(lut.sum()),
                "eos_id": eos_id,
                "boundary_substrings": list(substrings),
                "min_span": rule.min_span,
                "span_cap": rule.span_cap,
                **sigma_manifest,
            },
        },
        settings=wandb.Settings(_service_wait=60),
    )

    # ── data ─────────────────────────────────────────────────────────────
    batch_size = int(cfg.data.batch_size)
    train_loader = make_loader(bcfg, seq_len, batch_size)
    val_loader = make_loader(bcfg, seq_len, batch_size,
                             skip_samples=int(cfg.data.val_skip_samples))
    gen = torch.Generator(device=device).manual_seed(int(tr.seed))

    ckpt_dir = str(tr.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    total = int(tr.steps)
    planner.train()

    def _save(step: int) -> str:
        path = os.path.join(ckpt_dir, f"step_{step}.pt")
        torch.save({
            "step": step,
            "planner": planner.state_dict(),
            "planner_cfg": {k: getattr(pcfg, k) for k in pcfg.__dataclass_fields__},
            "cfg": full_cfg,
            "backbone_cfg": OmegaConf.to_container(bcfg, resolve=True,
                                                   throw_on_missing=False),
            "sigma_manifest": sigma_manifest,
        }, path)
        print(f"[p1] saved planner-only checkpoint {path}")
        return path

    t0 = time.perf_counter()
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, float(tr.lr), int(tr.warmup),
                            float(tr.min_lr_frac))

        ids = next(train_loader)[0].to(device)
        geom = segment_rows(ids, rule, max_slots)
        if int(geom.valid.sum()) < 2:
            continue

        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = backbone.prelude_states(
                        ids, apply_input_norm=bool(cfg.backbone.apply_input_norm))
            else:
                h = backbone.prelude_states(
                    ids, apply_input_norm=bool(cfg.backbone.apply_input_norm))
        h = h.float()

        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = fm_loss(planner, h, geom, schedule, generator=gen,
                                      edges=edges)
        else:
            loss, stats = fm_loss(planner, h, geom, schedule, generator=gen, edges=edges)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(planner_params, float(tr.grad_clip))
        opt.step()

        if step % int(tr.log_every) == 0:
            rec = {
                "step": step,
                "train/fm_loss": float(loss.item()),
                "train/grad_norm": float(gnorm),
                "train/lr": opt.param_groups[0]["lr"],
                "slots/n_valid": stats["n_valid"],
                "slots/dropped_frac": geom.dropped_fraction,
                "slots/n_dropped_budget": geom.n_dropped_budget,
                "train/sq_mean": stats["sq_mean"],
                "train/y_norm_mean": stats["y_norm_mean"],
                "train/sigma_mean": stats["sigma_mean"],
                "train/null_loss": stats["null_loss"],
                "train/rel_loss": stats["rel_loss"],
                "perf/steps_per_s": (step + 1) / max(time.perf_counter() - t0, 1e-9),
            }
            for k, v in stats.items():
                if k.startswith("band"):
                    rec[f"train/{k}"] = v
            wandb.log(rec, step=step)
            print(f"  step {step:>6}  loss {float(loss.item()):.4f}  "
                  f"slots {stats['n_valid']:.0f}  drop {geom.dropped_fraction:.3f}  "
                  f"|g| {float(gnorm):.2e}", flush=True)

        if int(tr.eval_every) > 0 and step > 0 and step % int(tr.eval_every) == 0:
            ev = evaluate(planner, backbone, val_loader, rule, cfg, schedule, edges,
                          max_slots, device, int(tr.n_eval_batches), gen)
            wandb.log({**ev, "step": step}, step=step)
            print(f"  EVAL {step:>6}  fm {ev['val/fm_loss']:.4f} "
                  f"(rel {ev['val/rel_loss']:.4f})  "
                  f"top1 {ev['val/retrieval_top1']:.4f} "
                  f"(chance {ev['val/retrieval_chance']:.4f})  "
                  f"in-row {ev['val/retrieval_top1_within_row']:.4f} "
                  f"(chance {ev['val/retrieval_chance_within_row']:.4f})  "
                  f"eff_rank {ev['val/target_effective_rank']:.1f}", flush=True)

        if int(tr.ckpt_every) > 0 and step > 0 and step % int(tr.ckpt_every) == 0:
            _save(step)

    ev = evaluate(planner, backbone, val_loader, rule, cfg, schedule, edges, max_slots,
                  device, int(tr.n_eval_batches), gen)
    wandb.log({**ev, "step": total}, step=total)
    _save(total)
    print(f"[p1] done: {total} steps in {(time.perf_counter()-t0)/60:.1f} min; "
          f"final top1 {ev['val/retrieval_top1']:.4f} vs chance "
          f"{ev['val/retrieval_chance']:.4f}; within-row "
          f"{ev['val/retrieval_top1_within_row']:.4f} vs "
          f"{ev['val/retrieval_chance_within_row']:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
