"""MORPH Step-Boundary STP Reasoning SFT — arXiv:2604.18464 reproduction.

Fine-tunes a ROUTED+COMPACT Phase-C checkpoint on reasoning traces (GSM8K or MATH),
computing the paper's Semantic Step Prediction (STP) objective at reasoning-step
boundaries.  Three arms differ in a SINGLE config knob:

    model.stp_mode=step_boundary  → paper STP on LAST-TOKEN-of-step latents  (model A)
    model.stp_mode=random_token   → paper STP on uniformly sampled latents    (model C)
    model.stp_mode=off            → baseline: no paper STP (model B, bit-exact)

KEY DESIGN DECISIONS:
  - The base ckpt is routed+compact (Phase-C) → MUST reconstruct topology via
    load_phasec_model (plain load_state_dict → 24 missing + 82 unexpected → garbage ppl).
  - Fresh optimizer each run (SFT not a resume).
  - Topology is FROZEN: no prune/carve/route events.
  - Padded batches (one example per row, dynamic padding) via reasoning_batches_padded
    which returns (ids, labels, seq_lens, boundary_mask).
  - boundary_mask is passed as step_boundary_mask to model.forward; model uses it only
    if stp_mode="step_boundary" (else arg is ignored → fully bit-identical baseline).

Run (example — step_boundary arm):

    PYTHONPATH=$PWD python -m morph.training.sft_reasoning \\
        sft.init_ckpt=/abs/path/to/step_50000.pt \\
        model.stp_mode=step_boundary \\
        wandb.name=sft_reasoning_step_boundary

All other config comes from sft_reasoning.yaml (which composes base.yaml).
"""
from __future__ import annotations

import math
import os
import sys
import time

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf

from morph.model.routing import collect_routing_aux_losses
from morph.training.optimizer import create_optimizer
from morph.training.reasoning_data import (
    build_gsm8k_examples,
    build_math_examples,
    build_olympiad_examples,
    count_reasoning_steps,
    reasoning_batches_padded,
)
from morph.training.punct_boundary import (
    punctuation_boundary_mask,
    resolve_punct_token_ids,
)
from morph.training.train import save_checkpoint

_MORPH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Canonical load of a routed+compact Phase-C ckpt ────────────────────────────────────────
# import from ignore/ — it's in the PYTHONPATH root, not a package, so we use sys.path.

def _import_load_phasec():
    """Import load_phasec_model from ignore/fixed_eval_phasec.py.

    ignore/ is NOT a Python package (no __init__.py) — we import the module directly via
    importlib so we don't need to assume it's on sys.path as a dotted module name.
    """
    import importlib.util
    phasec_path = os.path.join(_MORPH_ROOT, "ignore", "fixed_eval_phasec.py")
    if not os.path.isfile(phasec_path):
        raise FileNotFoundError(
            f"fixed_eval_phasec.py not found at {phasec_path}. "
            "Ensure you are running from the repo root with PYTHONPATH=$PWD."
        )
    spec = importlib.util.spec_from_file_location("fixed_eval_phasec", phasec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_phasec_model


load_phasec_model = _import_load_phasec()


# ── LR schedule ────────────────────────────────────────────────────────────────────────────

def _make_lr_schedule(total_steps: int, warmup_frac: float, lr_max: float, lr_min: float):
    """Warmup + cosine decay.  lr_min == lr_max → flat LR (the winning SFT recipe)."""
    warmup = max(1, int(total_steps * warmup_frac))

    def sched(step: int) -> float:
        if step < warmup:
            return lr_max * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return sched


# ── Entry point ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="sft_reasoning", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft = cfg.sft

    # ── Validate required fields ──────────────────────────────────────────────
    init_ckpt = str(sft.init_ckpt)
    if not os.path.isfile(init_ckpt):
        raise FileNotFoundError(f"sft.init_ckpt not found: {init_ckpt}")

    data_source = str(getattr(sft, "data_source", "gsm8k")).strip().lower()
    if data_source not in ("gsm8k", "math", "olympiad"):
        raise ValueError(f"sft.data_source must be 'gsm8k', 'math', or 'olympiad', got {data_source!r}")

    stp_mode = str(OmegaConf.select(cfg, "model.stp_mode") or "off")
    step_stp_lambda = float(OmegaConf.select(cfg, "model.step_stp_lambda") or 1.0)
    # Boundary source for stp_mode="step_boundary": "none"/"step" → the reasoning-step
    # mask carried by the data (paper arm A); "punctuation" → derive boundaries from
    # punctuation in the SAME token sequence (the punctuation-vs-semantic-step ablation).
    # Everything else is held identical, so this is a single-variable A/B vs arm A.
    boundary_source = str(OmegaConf.select(cfg, "model.stp_boundary_source") or "none").lower()
    use_punct = boundary_source == "punctuation"
    print(f"[sft_reasoning] stp_mode={stp_mode!r}  step_stp_lambda={step_stp_lambda}  "
          f"boundary_source={boundary_source!r}", flush=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # ── Punctuation boundary ids (only when boundary_source="punctuation") ─────
    punct_ids = None
    punct_min_gap = int(OmegaConf.select(cfg, "model.stp_punct_min_gap") or 2)
    if use_punct:
        include_comma = bool(OmegaConf.select(cfg, "model.stp_punct_include_comma"))
        include_newline = OmegaConf.select(cfg, "model.stp_punct_include_newline")
        include_newline = True if include_newline is None else bool(include_newline)
        punct_ids = resolve_punct_token_ids(
            tokenizer, include_comma=include_comma, include_newline=include_newline
        )
        print(f"[sft_reasoning] punctuation boundaries: {len(punct_ids)} ids  "
              f"(comma={include_comma}, newline={include_newline}, min_gap={punct_min_gap})",
              flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    n_examples = int(sft.n_examples)
    epochs = int(sft.epochs)
    batch_size = int(sft.batch_size)
    max_len = int(sft.max_len)
    seed = int(sft.seed)

    print(f"[sft_reasoning] building {data_source} examples "
          f"(n={n_examples}, seed={seed}, max_len={max_len})", flush=True)
    if data_source == "gsm8k":
        examples = build_gsm8k_examples(tokenizer, n_examples=n_examples, seed=seed,
                                         max_len=max_len)
    elif data_source == "olympiad":
        examples = build_olympiad_examples(tokenizer, n_examples=n_examples, seed=seed,
                                           max_len=max_len)
    else:
        examples = build_math_examples(tokenizer, n_examples=n_examples, seed=seed,
                                        max_len=max_len)

    total_steps = count_reasoning_steps(examples, batch_size, epochs)
    if total_steps < 1:
        raise RuntimeError(
            f"total_steps={total_steps}; too few examples ({len(examples)}) "
            f"for batch_size={batch_size} × epochs={epochs}. "
            "Increase sft.n_examples or lower sft.batch_size."
        )
    print(f"[sft_reasoning] {len(examples)} examples → {total_steps} steps "
          f"(batch={batch_size}, epochs={epochs})", flush=True)

    # ── Model: faithful Phase-C topology reconstruction ───────────────────────
    # load_phasec_model: build dense+QAT, conform CMS layers to ckpt sparse shapes,
    # enable routing (routed=True for the 50k ckpt), load weights, ABORT on missing.
    # It returns model.eval(); we immediately .train() it.
    print(f"[sft_reasoning] reconstructing Phase-C topology from {init_ckpt}", flush=True)
    model = load_phasec_model(cfg, init_ckpt, device, routed=True, verbose=True)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sft_reasoning] model ready: {n_params:,} params", flush=True)

    # ── Optimizer: fresh (SFT is NOT a resume) ────────────────────────────────
    optimizer = create_optimizer(model, cfg)
    lr_fn = _make_lr_schedule(
        total_steps,
        float(sft.warmup_frac),
        float(cfg.training.lr),
        float(sft.min_lr),
    )
    grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))
    scaler = torch.amp.GradScaler("cuda")

    # RESEED so all arms share an identical stochastic stream (analogous to sft.py):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── wandb ─────────────────────────────────────────────────────────────────
    full_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    full_cfg["n_params"] = n_params
    full_cfg["reasoning_total_steps"] = total_steps
    full_cfg["reasoning_n_examples_used"] = len(examples)
    full_cfg["data_source"] = data_source
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        config=full_cfg,
    )

    ckpt_dir = os.path.join(
        _MORPH_ROOT, "checkpoints", "morph",
        wandb.run.name if wandb.run else "sft_reasoning_run",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    log_every = int(getattr(sft, "log_every", 20))

    print(
        f"[sft_reasoning] START  data={data_source}  stp_mode={stp_mode!r}  "
        f"step_stp_lambda={step_stp_lambda}  lr={cfg.training.lr}  "
        f"total_steps={total_steps}  ckpt_dir={ckpt_dir}",
        flush=True,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    data_iter = reasoning_batches_padded(
        examples, batch_size=batch_size, epochs=epochs, seed=seed, pad_id=pad_id
    )

    step = 0
    t0 = time.perf_counter()

    for batch in data_iter:
        ids, labels, seq_lens, boundary_mask = [t.to(device) for t in batch]

        # Punctuation arm: replace the reasoning-step mask with a punctuation-derived
        # one over the SAME ids (single changed variable vs arm A). Pad positions are
        # never punctuation, so they stay unmarked.
        if use_punct:
            boundary_mask = punctuation_boundary_mask(ids, punct_ids, min_gap=punct_min_gap)

        bnd_per_seq = float(boundary_mask.sum(dim=1).float().mean().item())

        lr = lr_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                ids,
                labels=labels,
                bag_size=0,
                seq_lens=seq_lens,
                step_boundary_mask=boundary_mask,
            )

        loss = out["loss"]

        # Routing auxiliary loss (present on routed ckpts; adds load-balancing signal).
        # Collect here so the topology stays frozen (no prune.step events).
        try:
            aux = collect_routing_aux_losses(model)
            if aux is not None:
                loss = loss + aux
        except Exception:
            pass  # no routers or already zero — ignore

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        lv = loss.item()
        if not math.isfinite(lv):
            ep = os.path.join(ckpt_dir, f"NONFINITE_step_{step}.pt")
            torch.save({"model": model.state_dict(), "step": step}, ep)
            raise RuntimeError(
                f"[sft_reasoning] non-finite loss {lv} at step {step}; "
                f"emergency ckpt saved to {ep}"
            )

        paper_stp_lv = out.get("paper_stp_loss")
        paper_stp_lv = paper_stp_lv.item() if paper_stp_lv is not None else 0.0
        ce_lv = out["loss"].item() - step_stp_lambda * paper_stp_lv  # approx CE portion

        sps = (step + 1) / (time.perf_counter() - t0)
        log_dict = {
            "sft_reasoning/loss": lv,
            "sft_reasoning/ppl": math.exp(min(lv, 20.0)),
            "sft_reasoning/paper_stp_loss": paper_stp_lv,
            "sft_reasoning/boundaries_per_seq": bnd_per_seq,
            "sft_reasoning/lr": lr,
            "sft_reasoning/seq_lens_mean": float(seq_lens.float().mean().item()),
            "perf/steps_per_sec": sps,
            "perf/step": step,
        }
        wandb.log(log_dict, step=step)

        if step % log_every == 0:
            print(
                f"[sft_reasoning {step:4d}/{total_steps}] "
                f"loss={lv:.4f} ppl={math.exp(min(lv, 20.0)):.1f} "
                f"paper_stp={paper_stp_lv:.4f} bnd/seq={bnd_per_seq:.1f} "
                f"lr={lr:.2e} sps={sps:.2f} "
                f"slen={float(seq_lens.float().mean().item()):.0f}",
                flush=True,
            )

        step += 1

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final = os.path.join(ckpt_dir, f"step_{step}.pt")
    # Use train.py's save_checkpoint for a full-format ckpt (includes scaler, step, etc.).
    # Pass pruning=None because topology is frozen (no pruning state to save).
    save_checkpoint(final, step, model, optimizer, scaler, pruning=None, next_step=step)
    print(f"[sft_reasoning] DONE {step} steps → {final}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
