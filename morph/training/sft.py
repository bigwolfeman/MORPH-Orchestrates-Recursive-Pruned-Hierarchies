"""MORPH instruction-tuning (SFT) — fine-tune a routed/compact pretrained checkpoint.

Prediction-2 experiment: does a short instruction-following SFT (with STP, the paper's
intended use) erase the pretraining-STP repetition penalty found in prediction 1? Both
arms (base_off, base_on) run THIS script with IDENTICAL config except `sft.init_ckpt`.

Design (see sft_data.py for the packed/response-masked data contract):
  - Build the model with the SAME quant stack the ckpt was saved with (ternary + embed
    int6), reconstruct routed (ReMoE) + carved (BCSR) topology via load_checkpoint, then
    build a FRESH optimizer on the reconstructed param set and reset the step axis to 0.
  - STP is enabled by `model.stp_lambda` (set in sft.yaml) — the forward always adds
    `stp_lambda * stp_loss`, so "both bases get STP-SFT" = stp_lambda>0 for both arms.
  - RESEED after load: load_checkpoint restores the ckpt's RNG; the two arms have
    different saved RNG, which would break the paired comparison. Reseeding with the
    shared sft.seed gives both arms identical Poisson-depth/dropout streams.
  - Topology is FROZEN (no prune/carve/route): pure forward→backward→clip→step.

Run:  PYTHONPATH=$PWD python -m morph.training.sft \
          sft.init_ckpt=checkpoints/morph/tst_stp_off_50k/step_50000.pt \
          wandb.name=sft_off_stp
"""
from __future__ import annotations

import math
import os
import random
import time

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf

from morph.model.routing import collect_routing_aux_losses
from morph.model.transformer import MORPHTransformer
from morph.training.optimizer import create_optimizer
from morph.training.pruning import PruningSchedule
from morph.training.sft_data import (
    build_dolly_examples,
    build_mcq_examples,
    count_steps,
    count_steps_padded,
    sft_batches,
    sft_batches_padded,
)
from morph.training.train import build_morph_config, load_checkpoint, save_checkpoint

_MORPH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model_with_quant(cfg: DictConfig, device) -> nn.Module:
    """Mirror train.py main()'s config-driven quant-stack build (no torch.compile).

    The ckpt was saved WITH these parametrizations (ternary `.original` + scales, embed
    int6) — they MUST exist on the live model before load or those keys are 'unexpected'
    and load_checkpoint RAISES. torch.compile is skipped: load_checkpoint strips the
    ckpt's `mlp._orig_mod.` prefix to match the uncompiled model, and carved-eager is the
    validated fast path. The fused Triton HC/attn/CE kernels (use_kernels/hc_use_kernel)
    are NOT torch.compile and stay on.
    """
    morph_cfg = build_morph_config(cfg)
    model = MORPHTransformer(morph_cfg).to(device)
    tr = cfg.training

    if bool(getattr(tr, "ternary", False)):
        from morph.model.ternary_qat import apply_ternary_qat
        apply_ternary_qat(
            model,
            scope=str(getattr(tr, "ternary_scope", "backbone")),
            threshold=float(getattr(tr, "ternary_threshold", 0.5)),
            scale_mode=str(getattr(tr, "ternary_scale_mode", "symmetric")),
            scale_group=str(getattr(tr, "ternary_scale_group", "tensor")),
            scale_dtype=str(getattr(tr, "ternary_scale_dtype", "fp16")),
            scale_clip_mult=float(getattr(tr, "ternary_scale_clip_mult", 0.0)),
        )
    eq = str(getattr(tr, "embed_quant", "off")).strip().lower()
    eq = "off" if eq in ("false", "none", "") else eq
    lq = str(getattr(tr, "lm_head_quant", "off")).strip().lower()
    lq = "off" if lq in ("false", "none", "") else lq
    if eq != "off":
        from morph.model.embed_quant import apply_embed_quant
        apply_embed_quant(model, embed_quant=eq, lm_head_quant=lq)

    ap = str(getattr(tr, "attn_proj_quant", "off")).strip().lower()
    ap = "off" if ap in ("false", "none", "") else ap
    if ap != "off":
        from morph.model.attn_proj_quant import apply_attn_proj_quant
        apply_attn_proj_quant(model, attn_proj_quant=ap)

    return model


def make_lr_schedule(total_steps: int, warmup_frac: float, lr_max: float, lr_min: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def sched(step: int) -> float:
        if step < warmup:
            return lr_max * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return sched


@hydra.main(config_path="../configs", config_name="sft", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft = cfg.sft
    init_ckpt = str(sft.init_ckpt)
    if not os.path.isfile(init_ckpt):
        raise FileNotFoundError(f"sft.init_ckpt not found: {init_ckpt}")

    # ── Data (built before model so total_steps drives the LR schedule) ──────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    examples = build_dolly_examples(
        tokenizer, n_examples=int(sft.n_examples), seed=int(sft.seed),
        max_len=int(sft.seq_len),
    )
    n_dolly = len(examples)
    # ── Optional MCQ mix-in (ARC train split) ───────────────────────────────
    # When sft.mcq_frac > 0, target an MCQ fraction of the COMBINED set: with f = mcq_frac,
    # mcq = round(f/(1-f) * n_dolly) capped at sft.mcq_n, then CONCATENATE and shuffle the
    # combined list with sft.seed BEFORE the batcher consumes it (so MCQ is interspersed,
    # not clustered). mcq_frac=0.0 ⇒ bit-identical to Dolly-only (this branch is skipped).
    mcq_frac = float(getattr(sft, "mcq_frac", 0.0))
    n_mcq = 0
    if mcq_frac > 0.0:
        if not (0.0 < mcq_frac < 1.0):
            raise ValueError(f"sft.mcq_frac must be in (0,1), got {mcq_frac}")
        mcq_n_cap = int(getattr(sft, "mcq_n", 4000))
        n_mcq_target = min(mcq_n_cap, round(mcq_frac / (1.0 - mcq_frac) * n_dolly))
        mcq_examples = build_mcq_examples(
            tokenizer, n_examples=n_mcq_target, seed=int(sft.seed),
            max_len=int(sft.seq_len), split="train",
        )
        n_mcq = len(mcq_examples)
        examples = examples + mcq_examples
        # Shuffle the COMBINED list so MCQ is interspersed (sft_batches re-shuffles ORDER
        # per epoch, but it concatenates the LIST as-is for stream packing on epoch 0's
        # base order; a one-time combined shuffle here guarantees interspersion regardless).
        random.Random(int(sft.seed)).shuffle(examples)
    print(f"[sft] data mix: dolly={n_dolly} mcq={n_mcq} (mcq_frac={mcq_frac}) "
          f"→ combined={len(examples)}", flush=True)
    # Resolve pad_id ONCE (only used by the padded path). tokenizer.pad_token_id may be
    # None (starcoder2 has no pad token) → fall back to eos (real tokens are right-padded,
    # pads carry label=-100 so the pad-token choice never enters the loss/attention).
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    packing = bool(getattr(sft, "packing", True))

    if packing:
        total_steps = count_steps(examples, int(sft.seq_len), int(sft.batch_size), int(sft.epochs))
    else:
        total_steps = count_steps_padded(examples, int(sft.batch_size), int(sft.epochs))
    if total_steps < 1:
        raise RuntimeError(f"computed total_steps={total_steps}; raise epochs/n_examples")
    print(f"[sft] total optimizer steps = {total_steps} "
          f"(packing={packing}, epochs={sft.epochs}, batch={sft.batch_size}, "
          f"seq_len={sft.seq_len}, pad_id={pad_id})", flush=True)

    # ── Model + topology reconstruction ─────────────────────────────────────
    print(f"[sft] building model + quant stack…", flush=True)
    model = build_model_with_quant(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    pruning = PruningSchedule.from_cfg(cfg)
    scaler = torch.amp.GradScaler("cuda")
    print(f"[sft] reconstructing topology from {init_ckpt}", flush=True)
    _next, _opt_state, needs_rebuild = load_checkpoint(init_ckpt, model, scaler, device, pruning)
    del _opt_state  # SFT uses a FRESH optimizer — discard the pretraining optimizer state
    print(f"[sft] reconstructed: is_compact={pruning.is_compact} is_routed={pruning.is_routed}",
          flush=True)

    # FRESH optimizer on the reconstructed (routed/carved) param set.
    optimizer = create_optimizer(model, cfg)
    lr_fn = make_lr_schedule(total_steps, float(sft.warmup_frac),
                             float(cfg.training.lr), float(sft.min_lr))
    grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))

    # RESEED for a PAIRED comparison (override the ckpt's restored RNG so both arms share
    # an identical Poisson-depth / dropout stream).
    torch.manual_seed(int(sft.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sft.seed))

    # ── wandb (full config + derived facts; reproducible from config alone) ──
    full_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    full_cfg["n_params"] = n_params
    full_cfg["sft_total_steps"] = total_steps
    full_cfg["sft_n_examples_used"] = len(examples)
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
               name=cfg.wandb.name, config=full_cfg)

    ckpt_dir = os.path.join(_MORPH_ROOT, "checkpoints", "morph",
                            wandb.run.name if wandb.run else "sft_run")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[sft] START stp_lambda={cfg.model.stp_lambda} lr={cfg.training.lr} "
          f"opt={cfg.training.optimizer} adam8bit={cfg.training.adam8bit} "
          f"init={os.path.basename(os.path.dirname(init_ckpt))}", flush=True)

    # Data iterator: packed (current default) yields (ids, labels); padded yields
    # (ids, labels, seq_lens). Both share the epoch-seeded order convention.
    if packing:
        data_iter = sft_batches(examples, int(sft.seq_len), int(sft.batch_size),
                                int(sft.epochs), int(sft.seed))
    else:
        data_iter = sft_batches_padded(examples, int(sft.batch_size),
                                       int(sft.epochs), int(sft.seed), int(pad_id))

    model.train()
    step = 0
    t0 = time.perf_counter()
    for batch in data_iter:
        if packing:
            ids, labels = batch
            seq_lens = None
        else:
            ids, labels, seq_lens = batch
            seq_lens = seq_lens.to(device)
        ids, labels = ids.to(device), labels.to(device)
        lr = lr_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, labels=labels, bag_size=0, seq_lens=seq_lens)
        loss = out["loss"]
        if pruning.is_routed:
            loss = loss + collect_routing_aux_losses(model)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        lv = loss.item()
        if not math.isfinite(lv):
            ep = os.path.join(ckpt_dir, f"NONFINITE_step_{step}.pt")
            save_checkpoint(ep, step, model, optimizer, scaler, pruning, next_step=step)
            raise RuntimeError(f"[sft] non-finite loss {lv} at step {step}; saved {ep}")

        sps = (step + 1) / (time.perf_counter() - t0)
        log = {
            "sft/loss": lv,
            "sft/ppl": math.exp(min(lv, 20.0)),
            "sft/stp_loss": out["stp_loss"].item(),
            "sft/lr": lr,
            "perf/steps_per_sec": sps,
            "perf/step": step,
        }
        if not packing:
            log["sft/seq_lens_mean"] = float(seq_lens.float().mean().item())
        wandb.log(log, step=step)
        if step % 20 == 0:
            extra = "" if packing else f" slen={float(seq_lens.float().mean().item()):.0f}"
            print(f"[sft {step:4d}/{total_steps}] loss={lv:.4f} ppl={math.exp(min(lv,20.0)):.1f} "
                  f"stp={out['stp_loss'].item():.4f} lr={lr:.2e} sps={sps:.2f}{extra}", flush=True)
        step += 1

    final = os.path.join(ckpt_dir, f"step_{step}.pt")
    save_checkpoint(final, step, model, optimizer, scaler, pruning, next_step=step)
    print(f"[sft] DONE {step} steps → {final}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
