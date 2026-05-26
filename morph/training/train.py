"""MORPH training entry point.

Usage:
    python morph/training/train.py                        # base config
    python morph/training/train.py training.steps=50000   # override
    python morph/training/train.py +training.ternary=true # phase 2

Config is managed by Hydra. See morph/configs/base.yaml for defaults.
All hyperparameters are logged to wandb at run start (full config dict).
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

# ── torch.compile + neural memory safety ──────────────────────────────────────
# Neural memory uses autograd.grad with retain_graph, which conflicts with
# donated buffer optimisation.  Must be set before torch import.
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch._functorch.config.donated_buffer = False  # type: ignore[attr-defined]
except AttributeError:
    torch._inductor.config.donated_buffer = False  # type: ignore[attr-defined]

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path so morph.* imports work when run from repo root.
_MORPH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Package should be pip-installed; no sys.path hack needed
# sys.path.insert(0, _MORPH_ROOT)

import wandb

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.routing import collect_routing_aux_losses, collect_routing_stats
from morph.training.data import create_dataloader
from morph.training.optimizer import create_optimizer, create_lr_schedule, TernaryShadowOptimizer
from morph.training.pruning import PruningSchedule


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    device: torch.device,
    loader,
    n_batches: int = 20,
) -> tuple[float, float]:
    """Return (avg_loss, ppl) over n_batches validation steps."""
    model.eval()
    if hasattr(model, "memory") and model.memory is not None:
        model.memory.reset_momentum()
    losses: list[float] = []
    for _ in range(n_batches):
        try:
            x, y = next(loader)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        losses.append(out["loss"].item())
    model.train()
    avg = sum(losses) / max(len(losses), 1)
    return avg, math.exp(min(avg, 20.0))


# ── Generation test ────────────────────────────────────────────────────────────

def run_generation_test(
    model: nn.Module,
    device: torch.device,
    tokenizer_name: str,
    seq_len: int,
    step: int,
    n_tokens: int = 100,
) -> str:
    """Run a short greedy generation and return the text."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "[transformers not installed]"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompts = [
        "The theory of relativity states that",
        "Once upon a time in a distant land, there lived a",
        "In machine learning, the key insight is that",
    ]
    output_lines: list[str] = []
    model.eval()

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)
        gen = ids.clone()
        with torch.no_grad():
            for _ in range(n_tokens):
                if gen.shape[1] >= seq_len:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(gen)
                logits = out["logits"][:, -1, :] / 0.8
                topk_v, _ = logits.topk(50, dim=-1)
                logits[logits < topk_v[:, -1:]] = float("-inf")
                next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
                gen = torch.cat([gen, next_tok], dim=1)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        output_lines.append(f"PROMPT: {prompt}\nOUTPUT: {text}")

    model.train()
    return "\n---\n".join(output_lines)


# ── Config → MORPHConfig ───────────────────────────────────────────────────────

def build_morph_config(cfg: DictConfig) -> MORPHConfig:
    m = cfg.model
    tr = cfg.training

    d_ff_raw = int(getattr(m, "d_ff", 0))

    ch = m.get("channel_dims", [384, 256, 128])
    channel_dims = tuple(int(c) for c in ch)

    return MORPHConfig(
        d_model=int(m.d_model),
        n_heads=int(m.n_heads),
        d_ff=d_ff_raw,
        vocab_size=int(m.vocab_size),
        max_seq_len=int(m.max_seq_len),
        n_prelude=int(m.n_prelude),
        n_core=int(m.n_core),
        n_coda=int(m.n_coda),
        mean_depth=int(m.mean_depth),
        max_depth=int(m.max_depth),
        bptt_depth=int(m.bptt_depth),
        channel_dims=channel_dims,
        compression=int(m.compression),
        n_kv_heads=int(m.n_kv_heads),
        csa_compress_ratio=int(m.csa_compress_ratio),
        hca_compress_ratio=int(m.hca_compress_ratio),
        top_k=int(m.top_k),
        window_size=int(m.window_size),
        context_len=int(m.context_len),
        lorentz_fraction=float(m.lorentz_fraction),
        bigram_hash_vocab=int(m.bigram_hash_vocab),
        n_memory_layers=int(m.n_memory_layers),
        n_memory_tokens=int(m.n_memory_tokens),
        stp_lambda=float(m.stp_lambda),
        stp_tau=int(m.stp_tau),
        d_z=int(m.d_z),
        segment_size=int(m.segment_size),
        dropout=float(tr.dropout),
        mac_warmup_steps=int(tr.mac_warmup_steps),
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optimizer,
    scaler: torch.amp.GradScaler,
    pruning: Optional[PruningSchedule],
) -> None:
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if pruning is not None:
        ckpt["pruning_compact"] = pruning.is_compact
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Strip ._orig_mod. prefix inserted by torch.compile
    state = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"  Warning: could not restore optimizer state: {e}")
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = ckpt.get("step", 0)
    # Restore model's internal step counter so memory warmup is correct.
    if hasattr(model, "_step"):
        model._step = step
    print(f"  Resumed from step {step}")
    return step


# ── Main training loop ────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Resolve paths ────────────────────────────────────────────────────
    tr = cfg.training
    data_cfg = cfg.data
    wb_cfg = cfg.wandb

    total_steps = int(tr.steps)
    batch_size = int(tr.batch_size)
    seq_len = int(data_cfg.seq_len)
    grad_clip = float(getattr(tr, "grad_clip", 1.0))
    eval_every = int(getattr(tr, "eval_every", 500))
    ckpt_every = int(getattr(tr, "ckpt_every", 2500))
    gen_every = int(getattr(tr, "gen_every", 0))  # 0 = disabled
    n_eval_batches = int(getattr(tr, "n_eval_batches", 20))
    resume_path: Optional[str] = getattr(tr, "resume", None)

    use_compile = bool(getattr(tr, "compile", True))
    compile_mode = str(getattr(tr, "compile_mode", "default"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── W&B init — log FULL config dict ──────────────────────────────────
    # OmegaConf → plain Python dict so wandb can serialise it.
    full_config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    wandb.init(
        project=wb_cfg.project,
        entity=getattr(wb_cfg, "entity", None),
        name=getattr(wb_cfg, "name", None),
        config=full_config_dict,
        settings=wandb.Settings(_service_wait=60),
    )

    # ── Build model ───────────────────────────────────────────────────────
    morph_cfg = build_morph_config(cfg)
    model = MORPHTransformer(morph_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # ── torch.compile ─────────────────────────────────────────────────────
    # Compile only the MLP sub-modules (attention uses Triton/SDPA kernels,
    # memory uses autograd.grad — both incompatible with fullgraph compile).
    if use_compile:
        for group in [model.prelude, model.core, model.coda]:
            for layer in group:
                if hasattr(layer, "mlp"):
                    layer.mlp = torch.compile(layer.mlp, mode=compile_mode)
        print(f"  MLPs compiled (mode={compile_mode})")

    # ── Optimizer + LR schedule ───────────────────────────────────────────
    optimizer = create_optimizer(model, cfg)
    lr_fn = create_lr_schedule(cfg)
    scaler = torch.amp.GradScaler("cuda")

    # ── Pruning schedule ──────────────────────────────────────────────────
    pruning = PruningSchedule.from_cfg(cfg)

    # ── Data loaders (train + val share the same generator; val uses a
    #    separate iterator so they don't pollute each other) ────────────────
    tokenizer_name = data_cfg.tokenizer
    dataset_name = data_cfg.dataset

    train_loader = iter(
        create_dataloader(tokenizer_name, dataset_name, seq_len, batch_size, split="train")
    )
    val_loader = iter(
        create_dataloader(tokenizer_name, dataset_name, seq_len, batch_size,
                         split="validation", skip_samples=50_000)
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(_MORPH_ROOT, "checkpoints", "morph",
                            wandb.run.name if wandb.run else "run")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Optional resume ───────────────────────────────────────────────────
    start_step = 0
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        start_step = load_checkpoint(resume_path, model, optimizer, scaler, device)

    # ── Optimizer step closure (resolved once, no per-step isinstance) ───
    _has_ternary = isinstance(optimizer, TernaryShadowOptimizer)
    if _has_ternary:
        def _step_optimizer():
            scaler.step(optimizer._opt)
            scaler.update()
            optimizer.step(_skip_inner=True)
    else:
        def _step_optimizer():
            scaler.step(optimizer)
            scaler.update()

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    step_times: list[float] = []
    t_start = time.perf_counter()

    for step in range(start_step, total_steps):
        lr = lr_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            x, y = next(train_loader)
        except StopIteration:
            train_loader = iter(
                create_dataloader(tokenizer_name, dataset_name, seq_len, batch_size,
                                  split="train")
            )
            x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        loss = out["loss"]

        # Routing aux loss (load balance) — only active after route_start
        if pruning.is_routed:
            routing_aux = collect_routing_aux_losses(model)
            loss = loss + routing_aux

        scaler.scale(loss).backward()

        prune_stats = pruning.step(model, step)

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        _step_optimizer()

        # ── Timing ────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        step_times.append(t_now - t_start)
        t_start = t_now
        if len(step_times) > 100:
            step_times = step_times[-100:]

        # ── Logging (every 20 steps) ──────────────────────────────────────
        if step % 20 == 0:
            sps = 1.0 / (sum(step_times) / max(len(step_times), 1))
            log: dict = {
                "train/loss": loss.item(),
                "train/ppl": math.exp(min(loss.item(), 20.0)),
                "train/lr": lr,
                "perf/steps_per_sec": sps,
                "perf/step": step,
            }
            if "memory_loss" in out:
                log["train/memory_loss"] = out["memory_loss"].item()
            if "z_loss" in out:
                log["train/z_loss"] = out["z_loss"].item()
            if "stp_loss" in out:
                log["train/stp_loss"] = out["stp_loss"].item()

            # Pruning stats
            if prune_stats:
                log.update(prune_stats)

            # Ternary stats (every 100 steps to keep overhead low)
            if step % 100 == 0 and isinstance(optimizer, TernaryShadowOptimizer):
                tern = optimizer.ternary_stats()
                log.update({f"ternary/{k}": v for k, v in tern.items()})

            # Routing diagnostics (every 100 steps, only when routed)
            if step % 100 == 0 and pruning.is_routed:
                rt_stats = collect_routing_stats(model)
                log.update(rt_stats)

            # Memory diagnostics (every 100 steps)
            if step % 100 == 0 and hasattr(model, "memory") and model.memory is not None:
                mem = model.memory
                if hasattr(mem, "get_memory_stats"):
                    ms = mem.get_memory_stats()
                    for k, v in ms.items():
                        if isinstance(v, (int, float)):
                            log[f"memory/{k}"] = v

            wandb.log(log, step=step)

            if step % 200 == 0:
                print(
                    f"[{step:7d}/{total_steps}] loss={loss.item():.4f}  "
                    f"ppl={math.exp(min(loss.item(), 20.0)):.1f}  "
                    f"lr={lr:.2e}  sps={sps:.2f}"
                )

        # ── Validation (every eval_every steps) ──────────────────────────
        if step % eval_every == 0 and step > 0:
            val_loss, val_ppl = evaluate(model, device, val_loader, n_eval_batches)
            val_log: dict = {"val/loss": val_loss, "val/ppl": val_ppl}

            # Memory contribution diagnostic
            if (
                hasattr(model, "memory")
                and model.memory is not None
                and step >= morph_cfg.mac_warmup_steps
            ):
                if hasattr(model.memory, "_zero_output"):
                    model.memory._zero_output = True
                    zl, zp = evaluate(model, device, val_loader, n_eval_batches // 2)
                    model.memory._zero_output = False
                    val_log["val/ppl_mem_zeroed"] = zp
                    val_log["val/mem_contribution"] = val_ppl - zp

            wandb.log(val_log, step=step)
            print(
                f"  [VAL {step:7d}] loss={val_loss:.4f}  ppl={val_ppl:.2f}"
            )
            model.train()

        # ── Generation test ───────────────────────────────────────────────
        if gen_every > 0 and step % gen_every == 0 and step > 0:
            gen_text = run_generation_test(
                model, device, tokenizer_name, seq_len, step
            )
            wandb.log(
                {"gen/sample": wandb.Html(f"<pre>{gen_text}</pre>")}, step=step
            )
            print(f"  [GEN {step}]\n{gen_text[:300]}...")
            model.train()

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % ckpt_every == 0 and step > 0:
            ck_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            save_checkpoint(ck_path, step, model, optimizer, scaler, pruning)
            print(f"  Checkpoint: {ck_path}")

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
    save_checkpoint(final_path, total_steps, model, optimizer, scaler, pruning)
    print(f"Final checkpoint: {final_path}")

    # ── Final eval + generation ───────────────────────────────────────────
    val_loss, val_ppl = evaluate(model, device, val_loader, n_eval_batches)
    wandb.log({"val/loss_final": val_loss, "val/ppl_final": val_ppl}, step=total_steps)
    print(f"Final val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")

    if gen_every > 0 or bool(getattr(tr, "gen_test", False)):
        gen_text = run_generation_test(
            model, device, tokenizer_name, seq_len, total_steps, n_tokens=200
        )
        wandb.log(
            {"gen/final": wandb.Html(f"<pre>{gen_text}</pre>")}, step=total_steps
        )
        print(f"\nGeneration test:\n{gen_text}")

    wandb.finish()


if __name__ == "__main__":
    main()
