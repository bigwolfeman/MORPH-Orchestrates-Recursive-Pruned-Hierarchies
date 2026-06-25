"""MORPH training entry point.

Usage:
    python morph/training/train.py                        # base config
    python morph/training/train.py training.steps=50000   # override
    python morph/training/train.py +training.ternary=true # phase 2

Config is managed by Hydra. See morph/configs/base.yaml for defaults.
All hyperparameters are logged to wandb at run start (full config dict).
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from typing import Optional

# Diagnostic-only: env-guarded single-shot faulthandler to capture the stack of an
# intermittent step-0 backward hang (gradient-checkpoint recompute). Single dump
# (NOT repeat=True — repeated dumps SIGSEGV under live CUDA). Set MORPH_FAULT_TIMEOUT
# to a value past a healthy step-0 (~40s) so it only fires when actually wedged.
if os.environ.get("MORPH_FAULT_TIMEOUT"):
    import faulthandler as _fh
    _fh.dump_traceback_later(int(os.environ["MORPH_FAULT_TIMEOUT"]), repeat=False)

# ── torch.compile safety ───────────────────────────────────────────────────────
# The looped core uses gradient checkpointing (use_reentrant=False); disabling
# donated buffers below avoids buffer-aliasing conflicts under compile.
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable donated-buffer reuse: the looped core's compiled + checkpointed
# (use_reentrant=False) backward can otherwise alias input buffers. The knob
# lives in torch._functorch.config but that submodule is not auto-imported by
# `import torch`, so import it explicitly and set it where it exists (the path
# has shifted across torch versions — guard rather than guess).
import torch._functorch.config as _functorch_config  # noqa: E402

if hasattr(_functorch_config, "donated_buffer"):
    _functorch_config.donated_buffer = False

# Inductor compile workers via SPAWN, not the default fork-based "subprocess" pool.
# WHY: any torch.compile RECOMPILE during the training loop forks compile workers /
# gcc while background threads (wandb asyncio + status, HF-streaming httpx, the
# inductor read-thread) hold a glibc malloc-arena lock → the forked child deadlocks
# in __triton_launcher.c (intermittent; cost a full night — see Ai-notes 06-01-2026/
# MORPH-eval-recompile-hang). Recompiles are unavoidable here (the active-set's
# grad_mode/dtype/size guards leak past warmup). Spawn workers are FRESH processes
# that never inherit the main thread-lock state, so compilation can never fork-deadlock
# regardless of when a recompile fires. Verified: 60 real training steps with live
# recompiles, no wedge. Pair with the single-threaded warmup below (handles the initial
# bulk compile). This applies to ALL runs incl. the future pruning run.
# Spawn workers (MORPH_COMPILE_CARVED) are only needed when compiling the carved path.
# At d=768 carved-eager is faster than carved-compiled (grad_mode guard thrashing) and
# spawn caused a BrokenProcessPool on startup compile. Default: eager inductor workers +
# carved path runs eager via eager_on_recompile. Gate kept for cloud-scale revisit.
import torch._inductor.config as _inductor_config  # noqa: E402
if os.environ.get("MORPH_COMPILE_CARVED"):
    if hasattr(_inductor_config, "worker_start_method"):
        _inductor_config.worker_start_method = "spawn"
    os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")

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
from morph.training.optimizer import create_optimizer, create_lr_schedule
from morph.training.pruning import PruningSchedule
from morph.training.punct_boundary import resolve_punct_token_ids, punctuation_boundary_mask


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

@torch.compiler.set_stance("force_eager")
def run_generation_test(
    model: nn.Module,
    device: torch.device,
    tokenizer_name: str,
    seq_len: int,
    step: int,
    n_tokens: int = 100,
) -> str:
    """Run a short greedy generation and return the text.

    Decorated with ``force_eager``: generation runs the model token-by-token at
    batch=1 with a sequence length that grows by one every step, so the MLPs
    (``torch.compile``d for the *training* shapes B×S) see a brand-new shape on
    every token. Under the training stance (``eager_on_recompile``) dynamo still
    pays per-token guard-eval to route each novel shape to its eager fallback —
    measured >10× slower and it tripped the 90s watchdog mid-gen. ``force_eager``
    makes the compiled MLPs run their original eager code directly (~42 ms/tok,
    stable as seqlen grows; verified ignore/gen_isolated.py), and the decorator
    restores the prior stance on return. Eval (full-batch fixed shape) is
    unaffected and stays on the compiled path.
    """
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
        ckpt_grad_iters=int(getattr(m, "ckpt_grad_iters", -1)),
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
        stp_lambda=float(m.stp_lambda),
        stp_tau=int(m.stp_tau),
        loop_stp_lambda=float(getattr(m, "loop_stp_lambda", 0.0)),
        stp_mode=str(getattr(m, "stp_mode", "off")),
        step_stp_lambda=float(getattr(m, "step_stp_lambda", 1.0)),
        latent_forecast_mode=str(getattr(m, "latent_forecast_mode", "off")),
        latent_forecast_lambda=float(getattr(m, "latent_forecast_lambda", 1.0)),
        latent_forecast_hidden=int(getattr(m, "latent_forecast_hidden", 0)),
        latent_forecast_predictor=str(getattr(m, "latent_forecast_predictor", "mlp")),
        ce_chunk_size=int(getattr(m, "ce_chunk_size", 1024)),
        use_kernels=bool(getattr(m, "use_kernels", True)),
        hc_streams=int(getattr(m, "hc_streams", 4)),
        hc_tau=float(getattr(m, "hc_tau", 1.0)),
        hc_cayley_iters=int(getattr(m, "hc_cayley_iters", 3)),
        hc_cayley_alpha=float(getattr(m, "hc_cayley_alpha", 0.1)),
        hc_init_gain=float(getattr(m, "hc_init_gain", 0.1)),
        hc_use_kernel=bool(getattr(m, "hc_use_kernel", True)),
        l2_persist=bool(getattr(m, "l2_persist", False)),
        retention=bool(getattr(m, "retention", True)),
        retention_layers=tuple(int(x) for x in getattr(m, "retention_layers", (1,))),
        retention_heads=int(getattr(m, "retention_heads", 0)),
        retention_chunk=int(getattr(m, "retention_chunk", 128)),
        retention_gate_init=float(getattr(m, "retention_gate_init", -6.0)),
        retention_carry=bool(getattr(m, "retention_carry", True)),
        retention_gate_bias=float(getattr(m, "retention_gate_bias", 2.0)),
        core_gain_clip=float(getattr(m, "core_gain_clip", 0.0)),
        dropout=float(tr.dropout),
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optimizer,
    scaler: torch.amp.GradScaler,
    pruning: Optional[PruningSchedule],
    *,
    next_step: Optional[int] = None,
) -> None:
    """Save a full training checkpoint.

    `step` is the label for the checkpoint filename/logs. `next_step` is the loop
    index to execute on resume. They differ for ordinary post-step checkpoints:
    after completing loop step N, resume must start at N+1 and fast-forward N+1
    batches. Pre-step transition checkpoints pass next_step=N.
    """
    resume_step = int(step if next_step is None else next_step)
    ckpt = {
        "step": int(step),
        "next_step": resume_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        # RNG state so a resume continues the SAME stochastic stream (per-sequence Poisson
        # depth draws, dropout, etc.) — "like nothing happened". CPU + all CUDA devices.
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if pruning is not None:
        # Both topology-phase flags are needed to RECONSTRUCT module structure (carve +
        # routers) before load_state_dict on resume. _is_compact alone is insufficient:
        # a routed checkpoint also needs its routers re-attached or their params are
        # silently dropped (strict=False) and routing comes back OFF.
        ckpt["pruning_compact"] = pruning.is_compact
        ckpt["pruning_routed"] = pruning.is_routed
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pruning: Optional[PruningSchedule] = None,
) -> tuple[int, dict, bool]:
    """FULL resume — restore the run exactly ("like nothing happened").

    Restores: model weights + topology (carve/BCSR + ReMoE routers), the pre-carve
    dead-tile prune mask (now a buffer), the saliency EMA, the GradScaler, CPU+CUDA RNG,
    the training step, and the pruning-schedule phase flags. Reconstructs module structure
    in the SAME order the live run mutated it, so every saved tensor finds a home:

        routers (if routed)  →  load_state_dict (auto-rebuilds carve)  →  rng/scaler

    The OPTIMIZER is handled by the caller: a carved/routed checkpoint's optimizer state is
    keyed on a DIFFERENT param set (mortar_data / router params) than the freshly-built
    dense optimizer, so the caller must REBUILD the optimizer on the reconstructed topology
    BEFORE loading state. Returns (next_step, optimizer_state_dict, needs_optimizer_rebuild).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    is_compact = bool(ckpt.get("pruning_compact", False))
    is_routed = bool(ckpt.get("pruning_routed", False))

    # 1. Re-attach ReMoE routers BEFORE load_state_dict. CMSBlockLinear._load_from_state_dict
    #    auto-reconstructs the BCSR carve, but routers are separate submodules it does NOT
    #    rebuild — without this their params have no home and strict=False drops them.
    if is_routed:
        if pruning is None:
            raise RuntimeError(
                "load_checkpoint: checkpoint is ROUTED but no PruningSchedule was passed to "
                "reconstruct the routers — cannot resume faithfully."
            )
        pruning._activate_routing(model)
    if pruning is not None:
        pruning._is_compact = is_compact
        pruning._is_routed = is_routed

    # 2. Load weights. CMSBlockLinear._load_from_state_dict rebuilds carve (BCSR) storage
    #    from the mortar_* keys; the _prune_mask buffer restores the pre-carve dead tiles.
    #    KEY ALIGNMENT: torch.compile wraps the MLPs in-place (layer.mlp = compile(...)), so
    #    BOTH the checkpoint and the live model nest keys under `mlp._orig_mod.…`. The old
    #    code stripped `_orig_mod.` from ONLY the checkpoint → every compiled-MLP tensor
    #    mismatched the (still-`_orig_mod`) model and strict=False silently dropped them (a
    #    near-empty "resume" — latent theater). Fix: align the checkpoint's key CONVENTION to
    #    the model's, but pass ALL keys through INTACT so the carve/router load-hooks fire
    #    (pre-filtering to the dense model's keys would drop mortar_data before it exists).
    ckpt_model = ckpt["model"]
    model_keys = list(model.state_dict().keys())
    model_has_orig = any("_orig_mod" in k for k in model_keys)
    ckpt_has_orig = any("_orig_mod" in k for k in ckpt_model)
    if ckpt_has_orig and not model_has_orig:
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt_model.items()}
    else:
        # Same convention (both compiled, or neither) → as-is. (model-compiled/ckpt-not is
        # not produced by this codebase — compile is applied unconditionally before save.)
        state = dict(ckpt_model)
    # Let load_state_dict report truthfully AFTER the hooks reconstruct mortar_data/routers.
    missing, unexpected = model.load_state_dict(state, strict=False)
    # No-theater: an UNEXPECTED key means a saved tensor found no home (structure drift) →
    # state was silently lost. Fail loud. MISSING keys are tolerated only for back-compat
    # buffers a pre-this-change checkpoint legitimately lacks (e.g. _prune_mask), and warned.
    if unexpected:
        raise RuntimeError(
            f"load_checkpoint: {len(unexpected)} checkpoint tensors had no home in the "
            f"reconstructed model (structure mismatch — state would be silently lost): "
            f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    _benign_missing = tuple(m for m in missing if not m.endswith("_prune_mask"))
    if _benign_missing:
        print(f"  Warning: {len(_benign_missing)} model tensors absent from checkpoint "
              f"(kept their init): {_benign_missing[:8]}"
              f"{'...' if len(_benign_missing) > 8 else ''}")

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    # 3. RNG — continue the SAME stochastic stream (Poisson depth draws, dropout).
    if ckpt.get("rng_cpu") is not None:
        torch.set_rng_state(ckpt["rng_cpu"].cpu().to(torch.uint8))
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ckpt["rng_cuda"]])
        except Exception as e:  # device-count mismatch etc. — surface, don't pretend
            print(f"  Warning: could not restore CUDA RNG state ({e}); RNG continues fresh")

    ckpt_step = int(ckpt.get("step", 0))
    if "next_step" in ckpt:
        step = int(ckpt["next_step"])
    else:
        # Legacy periodic checkpoints were written after completing their loop step but
        # only persisted `step`. Treat them as post-step saves so resume executes the next
        # unseen batch. Legacy pre-step transition checkpoints are ambiguous; the warning is
        # intentional because exact replay cannot be inferred from old metadata alone.
        step = ckpt_step + 1
        print(f"  Warning: checkpoint lacks next_step metadata; assuming legacy post-step "
              f"save and resuming at step {step} (saved step label {ckpt_step})")
    needs_rebuild = bool(is_compact or is_routed)
    print(f"  Resumed model+scaler+RNG from checkpoint step {ckpt_step}; "
          f"next_step={step} "
          f"(compact={is_compact} routed={is_routed} → optimizer_rebuild={needs_rebuild})")
    return step, ckpt["optimizer"], needs_rebuild


def load_weights_only(path: str, model: nn.Module, device: torch.device) -> None:
    """Initialise model WEIGHTS from a checkpoint, but reset the run to step 0.

    Unlike load_checkpoint (full resume: weights + optimizer + scaler + step), this
    loads ONLY the model tensors and leaves the optimizer/scaler FRESH and the step
    counter at 0. Used by `training.init_from` to seed a brand-new schedule (e.g. the
    25k whole-body gradual-prune run) from dense pretrained weights while keeping the
    schedule's step axis absolute from 0 — and sidestepping the optimizer-resume ppl
    spike (a fresh optimizer + the schedule's dense warmup absorb any startup bump).
    The weight-load path is byte-identical to load_checkpoint's (strict=False, same
    ._orig_mod. strip), which the live B5 resume already proved loads this seed cleanly.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt["model"]
    # _orig_mod-robust key alignment. The seed may be saved from a torch.compile'd model
    # (keys carry `mlp._orig_mod.…`) AND this model is compiled too (same prefix) — in
    # which case the keys match NATIVELY and the usual `._orig_mod.` strip would BREAK the
    # match, silently dropping every compiled-MLP weight to random init (this was the real
    # cause of the mid-Phase-C resume ppl spike). So pick whichever key-form lands more
    # tensors on the model, rather than always stripping.
    model_keys = set(model.state_dict().keys())
    stripped = {k.replace("._orig_mod.", "."): v for k, v in raw.items()}
    n_raw = sum(1 for k in raw if k in model_keys)
    n_strip = sum(1 for k in stripped if k in model_keys)
    state = raw if n_raw >= n_strip else stripped
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Hard guard against a silent partial load: the MLP backbone (gate_up/down shadows)
    # MUST land. If almost nothing matched, the seed is incompatible — fail LOUD.
    n_loaded = len(model_keys) - len(missing)
    if n_loaded < 0.5 * len(model_keys):
        raise RuntimeError(
            f"init_from {path}: only {n_loaded}/{len(model_keys)} model tensors matched "
            f"(raw-match={n_raw}, strip-match={n_strip}). Seed/model key structure mismatch "
            f"— refusing to train from a mostly-random model."
        )
    print(f"  init_from {path}: loaded WEIGHTS only (step reset → 0, fresh optimizer); "
          f"seed step was {ckpt.get('step', '?')}; matched {n_loaded}/{len(model_keys)} "
          f"tensors via {'raw' if state is raw else 'stripped'} keys; "
          f"{len(missing)} missing / {len(unexpected)} unexpected", flush=True)


@torch.no_grad()
def diag_prune_optstate(model, optimizer, step: int, path: str) -> None:
    """Root-cause the AdEMAMix prune divergence (env MORPH_DIAG_OPT=<path>).

    For every CMSBlockLinear, dequant the optimizer's slow-EMA m₂ and second-moment ν,
    pair with the live grad, and reconstruct the per-element update (g+α·m₂)/(√(ν/bc2)+ε).
    Split positions DEAD (pruned, _prune_mask==0) vs LIVE and report the GLOBAL max|update|
    with its components — so we see EXACTLY what blows up: numerator (α·m₂ on a charged
    slow EMA) vs denominator (ν collapse), and whether it is a dead or a LIVE param. Handles
    all three state formats (fused linear-int8 m2_code, de-fused dynamic-map m2_q, fp32 m2).
    """
    opt = optimizer
    if not hasattr(opt, "state"):
        return
    from morph.model.layers.block_sparse import CMSBlockLinear

    def _grp(p):
        for g in opt.param_groups:
            for q in g["params"]:
                if q is p:
                    return g
        return None

    def _deq_any(st, key, signed, p):
        if key == "nu" and "nu_sqrt_code" in st:         # fused sqrt-ν int8 (BLOCK=256)
            code = st["nu_sqrt_code"].float()
            amax = st["nu_sqrt_amax"].float()
            scale = (amax / 127.0).repeat_interleave(256)[: code.numel()]
            nu_sqrt = code * scale
            return (nu_sqrt * nu_sqrt).view_as(p)
        if f"{key}_code" in st:                          # fused linear-int8 (BLOCK=256)
            code = st[f"{key}_code"].float()
            amax = st[f"{key}_amax"].float()
            scale = (amax / 127.0).repeat_interleave(256)[: code.numel()]
            return (code * scale).view_as(p)
        if f"{key}_q" in st:                             # de-fused dynamic-map
            return opt._deq(st[f"{key}_q"], st[f"{key}_amax"], signed, p)
        if key in st:                                    # fp32
            return st[key].view_as(p)
        return None

    root = getattr(model, "_orig_mod", model)
    worst = (-1.0, None)
    amom_dead = amom_live = 0.0
    minnu_dead = minnu_live = float("inf")
    zero_nu_dead = zero_nu_live = 0
    floor_den_dead = floor_den_live = 0
    total_dead = total_live = 0
    n_layers = n_with_state = 0
    per_layer = {}   # NEW: name -> (zeroNu_live_count, maxU_in_layer) to localize ν-collapse
    clip10_live = clip25_live = 0   # NEW: live coords whose ACTUAL |update| would be clipped at 10/25
    max_rms = (-1.0, None)          # NEW: max per-tensor update-RMS (collective-move magnitude) + layer
    max_rel = (-1.0, None)          # NEW: max per-tensor rel-step lr·‖u‖/‖w‖ (trust-ratio quantity) + layer
    # β1=0 root-cause measurement: snr = |m₂|/denom ≈ |mean|/rms. Low snr on failing
    # (saturating) coords → magnitude problem → SNR gate is appropriate. Sign-flip between
    # g and m₂ → oscillation → directional gate. Measures both globally + on the saturating
    # subset so the mechanism can be confirmed from data.
    snr_lt01_live = snr_lt03_live = 0          # live coords with snr<0.1 / <0.3
    sat_count = 0                              # live coords with |update|>10 (the "failing" coords)
    sat_sign_agree = 0                         # of those, how many have sign(g)==sign(m₂)
    sat_snr_sum = 0.0                          # Σ snr over saturating coords (÷ sat_count = mean)
    for name, layer in root.named_modules():
        if not isinstance(layer, CMSBlockLinear):
            continue
        n_layers += 1
        p = layer._prune_target_weight()                 # the param the optimizer holds
        st = opt.state.get(p)
        if not st or st.get("init"):
            continue
        m2 = _deq_any(st, "m2", True, p)
        nu = _deq_any(st, "nu", False, p)
        if m2 is None or nu is None or p.grad is None:
            continue
        n_with_state += 1
        grp = _grp(p)
        a_t, b2, b3_t = opt._sched(grp["step"], grp)
        bc2 = 1.0 - b2 ** grp["step"]
        eps = grp["eps"]
        g = p.grad.float()
        # Match the optimizer's ACTUAL eps placement so maxU is the REAL update magnitude
        # (eps-OUTSIDE runs were being under-reported by the old always-eps-inside reconstruction).
        eps_inside = bool(getattr(opt, "eps_inside", True))
        denom = (nu / bc2 + eps).sqrt() if eps_inside else ((nu / bc2).sqrt() + eps)
        amom = a_t * m2
        upd = ((g + amom) / denom).reshape(-1).abs()
        # dead mask: expand [R,C] _prune_mask → [out,in] elementwise (True=alive)
        B = layer.tile_size
        keep = layer._prune_mask.view(layer.R, 1, layer.C, 1).expand(
            layer.R, B, layer.C, B).reshape(layer.out_features, layer.in_features).reshape(-1)
        dead = ~keep.bool().to(upd.device)
        amf = amom.reshape(-1).abs()
        nuf = nu.reshape(-1)
        zero_nu = (nuf == 0)
        floor_den = (denom.reshape(-1) <= (eps ** 0.5) * 1.0001)
        if dead.any():
            amom_dead = max(amom_dead, float(amf[dead].max()))
            minnu_dead = min(minnu_dead, float(nuf[dead].min()))
            zero_nu_dead += int(zero_nu[dead].sum())
            floor_den_dead += int(floor_den[dead].sum())
            total_dead += int(dead.sum())
        live = ~dead
        layer_zero_live = 0
        if live.any():
            amom_live = max(amom_live, float(amf[live].max()))
            minnu_live = min(minnu_live, float(nuf[live].min()))
            layer_zero_live = int(zero_nu[live].sum())
            zero_nu_live += layer_zero_live
            floor_den_live += int(floor_den[live].sum())
            total_live += int(live.sum())
            updl = upd[live]
            clip10_live += int((updl > 10.0).sum())
            clip25_live += int((updl > 25.0).sum())
            # SNR + sign-agreement measurement: snr uses raw |m₂| (matches the optimizer
            # gate exactly, not amf=|α·m₂|).
            m2f = m2.reshape(-1).abs()
            snr = m2f / denom.reshape(-1)                    # |m₂|/denom ≈ |mean|/rms, per coord
            snr_live = snr[live]
            snr_lt01_live += int((snr_live < 0.1).sum())
            snr_lt03_live += int((snr_live < 0.3).sum())
            sign_agree = g.reshape(-1).sign() == m2.reshape(-1).sign()
            sat = live & (upd > 10.0)                        # the FAILING (saturating) coords
            if sat.any():
                sat_count += int(sat.sum())
                sat_sign_agree += int(sign_agree[sat].sum())
                sat_snr_sum += float(snr[sat].sum())
        lrms = float(upd.pow(2).mean().sqrt())                   # NEW: per-tensor update-RMS
        if lrms > max_rms[0]:
            max_rms = (lrms, name)
        lr_p = float(grp.get("lr", 0.0))                         # NEW: rel-step = lr·‖u‖/‖w‖
        rel = lr_p * float(upd.norm()) / (float(p.float().norm()) + 1e-12)
        if rel > max_rel[0]:
            max_rel = (rel, name)
        per_layer[name] = (layer_zero_live, float(upd.max()), rel)  # localize collapse + rel-step
        j = int(upd.argmax())
        if float(upd[j]) > worst[0]:
            gf, df = g.reshape(-1), denom.reshape(-1)
            worst = (float(upd[j]), dict(
                layer=name, dead=bool(dead[j]), g=float(gf[j]), amom=float(amf[j]),
                denom=float(df[j]), nu=float(nuf[j]), m2=float(m2.reshape(-1)[j]), a_t=a_t, b3=b3_t))
    b = worst[1] or {}
    with open(path, "a") as f:
        f.write(
            f"step={step} layers={n_with_state}/{n_layers} maxU={worst[0]:.3e} "
            f"dead={b.get('dead')} layer={b.get('layer')} g={b.get('g',0):.3e} "
            f"amom={b.get('amom',0):.3e} denom={b.get('denom',0):.3e} nu={b.get('nu',0):.3e} "
            f"m2={b.get('m2',0):.3e} a_t={b.get('a_t',0):.2f} b3={b.get('b3',0):.5f} "
            f"| amomMax d/l={amom_dead:.3e}/{amom_live:.3e} "
            f"minNu d/l={minnu_dead:.2e}/{minnu_live:.2e} "
            f"zeroNu d/l={zero_nu_dead}/{total_dead}:{zero_nu_live}/{total_live} "
            f"floorDen d/l={floor_den_dead}/{total_dead}:{floor_den_live}/{total_live} "
            f"clipLive >10={clip10_live} >25={clip25_live} "
            f"maxUpdRMS={max_rms[0]:.3f}@{max_rms[1]} "
            f"relStep={max_rel[0]:.2e}@{max_rel[1]}\n"
        )
        # NEW per-layer line: top layers by ν-collapse (only when any) + top-3 by maxU.
        # Localizes whether collapse concentrates at prelude.0.down (first-layer/arch) or spreads.
        z_top = sorted(((z, k) for k, (z, _, _) in per_layer.items() if z > 0), reverse=True)[:8]
        u_top = sorted(((u, k) for k, (_, u, _) in per_layer.items()), reverse=True)[:3]
        r_top = sorted(((r, k) for k, (_, _, r) in per_layer.items()), reverse=True)[:6]
        zstr = " ".join(f"{k}={z}" for z, k in z_top) if z_top else "none"
        ustr = " ".join(f"{k}={u:.2f}" for u, k in u_top)
        rstr = " ".join(f"{k}={r:.2e}" for r, k in r_top)
        f.write(f"  PERLAYER zeroNuLive[{zstr}] topMaxU[{ustr}] topRelStep[{rstr}]\n")
        # SNR distribution (live) + sign-agreement on saturating (|update|>10) coords.
        # Low sat-SNR + high sat-signAgree → magnitude problem → SNR gate appropriate.
        # Low sat-signAgree → oscillation → directional/sign gate.
        tl = max(total_live, 1)
        sat_sa = (sat_sign_agree / sat_count) if sat_count else float("nan")
        sat_snr = (sat_snr_sum / sat_count) if sat_count else float("nan")
        f.write(
            f"  SNRGATE snr<0.1={snr_lt01_live}/{total_live}({snr_lt01_live/tl:.3f}) "
            f"snr<0.3={snr_lt03_live}/{total_live}({snr_lt03_live/tl:.3f}) "
            f"| satCoords(|u|>10)={sat_count} satSignAgree={sat_sa:.3f} satMeanSNR={sat_snr:.3e}\n"
        )


def diag_optstate_allparams(model, optimizer, step: int, path: str) -> None:
    """All-param, gate-aware blow-up localizer (env MORPH_DIAG_OPT=<path>).

    Sweeps every optimizer-state param, reconstructs the real update (gate·g + α·m₂)/denom
    with the SNR gate applied, and decomposes it into the gated-g term |gate·g|/denom vs the
    slow-EMA term |α·m₂|/denom. Reports the global-worst param by full name + which term
    drives it, plus the worst per name-category (prelude/core/coda/embed/attn/hc/etc.).
    """
    opt = getattr(optimizer, "_opt", optimizer)
    if not hasattr(opt, "state"):
        return
    root = getattr(model, "_orig_mod", model)
    kappa = float(getattr(opt, "g_snr_gate_kappa", 0.0))
    floor = float(getattr(opt, "g_snr_gate_floor", 0.1))
    eps_inside = bool(getattr(opt, "eps_inside", True))
    grp_of = {}
    for g in opt.param_groups:
        for q in g["params"]:
            grp_of[id(q)] = g

    def _deq(st, key, signed, p):
        if f"{key}_q" in st:                       # de-fused dynamic-map (8-bit)
            return opt._deq(st[f"{key}_q"], st[f"{key}_amax"], signed, p).reshape(-1)
        if key in st:                              # fp32 (no-decay group: embed/HC/norm)
            return st[key].reshape(-1)
        return None

    def _cat(nm):
        for k in ("prelude", "core", "coda", "embed", "attn", "mhc", "hc",
                  "inject", "norm", "lm", "stp", "log_"):
            if k in nm:
                return k
        return "other"

    worst = (-1.0, None)
    max_g = (-1.0, None)
    max_am = (-1.0, None)
    cat_worst = {}
    for nm, p in root.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        st = opt.state.get(p)
        if not st or st.get("init"):
            continue
        grp = grp_of.get(id(p))
        if grp is None:
            continue
        a_t, b2, b3_t = opt._sched(grp["step"], grp)
        bc2 = 1.0 - b2 ** grp["step"]
        eps = grp["eps"]
        m2 = _deq(st, "m2", True, p)
        nu = _deq(st, "nu", False, p)
        if m2 is None or nu is None:
            continue
        g = p.grad.float().reshape(-1)
        denom = (nu / bc2 + eps).sqrt() if eps_inside else ((nu / bc2).sqrt() + eps)
        if kappa > 0.0:
            gate = (m2.abs() / denom / kappa).clamp(0.0, 1.0).mul_(1.0 - floor).add_(floor)
        else:
            gate = torch.ones_like(g)
        g_term = (gate * g).abs() / denom              # gated raw-g contribution
        am_term = (a_t * m2).abs() / denom             # UNGATED slow-EMA contribution
        upd = ((gate * g + a_t * m2) / denom).abs()
        mu = float(upd.max())
        cat = _cat(nm)
        if mu > cat_worst.get(cat, (-1.0, None))[0]:
            cat_worst[cat] = (mu, nm)
        mg, ma = float(g_term.max()), float(am_term.max())
        if mg > max_g[0]:
            max_g = (mg, nm)
        if ma > max_am[0]:
            max_am = (ma, nm)
        if mu > worst[0]:
            j = int(upd.argmax())
            worst = (mu, dict(name=nm, cat=cat, gterm=float(g_term[j]), amterm=float(am_term[j]),
                              gate=float(gate[j]), denom=float(denom[j]), m2=float(m2[j]),
                              nu=float(nu[j]), g=float(g[j]), a_t=a_t))
    b = worst[1] or {}
    driver = "amom" if b.get("amterm", 0) > b.get("gterm", 0) else "g"
    with open(path, "a") as f:
        f.write(
            f"  ALLPARAM step={step} worstU={worst[0]:.3e}@{b.get('name')}[{b.get('cat')}] "
            f"driver={driver} gTerm={b.get('gterm', 0):.3e} amTerm={b.get('amterm', 0):.3e} "
            f"gate={b.get('gate', 0):.3f} denom={b.get('denom', 0):.2e} m2={b.get('m2', 0):.2e} "
            f"nu={b.get('nu', 0):.2e} a_t={b.get('a_t', 0):.2f} "
            f"| maxGterm={max_g[0]:.3e}@{max_g[1]} maxAMterm={max_am[0]:.3e}@{max_am[1]}\n"
        )
        cats = " ".join(
            f"{c}={v:.2e}" for c, (v, _n) in sorted(cat_worst.items(), key=lambda x: -x[1][0])
        )
        f.write(f"  ALLPARAM_CAT {cats}\n")


_M2G_SNAP: dict = {}   # name -> (snap_step, m2_cpu_fp32_vec) for slow-EMA self-coherence


def diag_m2g_geometry(model, optimizer, step: int, path: str, snap_every: int = 50) -> None:
    """Slow-EMA geometry localizer on the CORE params (env MORPH_DIAG_M2G=<path>).

    Measures two diagnostic quantities for the slow-EMA / gradient geometry:
      (i)  cos(m₂, g): drops if the slow EMA has rotated away from the current gradient
           → indicates stale off-manifold direction → cure: directional-trust gate.
      (ii) cos(m₂_t, m₂_{t-k}): sustained ≈1 if drift is a coherent ramp → cure: damp
           persistence rather than mis-alignment.
    Logs both per core tensor each step + medians. Off by default (core params only; cheap).
    """
    opt = getattr(optimizer, "_opt", optimizer)
    if not hasattr(opt, "state"):
        return
    root = getattr(model, "_orig_mod", model)

    def _deq(st, key, signed, p):
        if f"{key}_q" in st:
            return opt._deq(st[f"{key}_q"], st[f"{key}_amax"], signed, p).reshape(-1)
        if key in st:
            return st[key].reshape(-1)
        return None

    rows = []
    cos_mg_all, cos_self_all = [], []
    for nm, p in root.named_parameters():
        if "core." not in nm or not p.requires_grad or p.grad is None:
            continue
        st = opt.state.get(p)
        if not st or st.get("init"):
            continue
        m2 = _deq(st, "m2", True, p)
        if m2 is None:
            continue
        g = p.grad.float().reshape(-1)
        m2n, gn = float(m2.norm()), float(g.norm())
        cos_mg = float((m2 @ g) / (m2n * gn + 1e-12))
        cos_self, age = float("nan"), -1
        prev = _M2G_SNAP.get(nm)
        if prev is not None:
            ps, pv = prev
            pv = pv.to(m2.device)
            cos_self = float((m2 @ pv) / (m2n * float(pv.norm()) + 1e-12))
            age = step - ps
        if prev is None or (step - prev[0]) >= snap_every:
            _M2G_SNAP[nm] = (step, m2.detach().to("cpu"))
        rows.append((nm, m2n, gn, cos_mg, cos_self, age))
        cos_mg_all.append(cos_mg)
        if cos_self == cos_self:
            cos_self_all.append(cos_self)
    if not rows:
        return
    import statistics as _stat
    med_mg = _stat.median(cos_mg_all)
    med_self = _stat.median(cos_self_all) if cos_self_all else float("nan")
    rows.sort(key=lambda r: r[3])     # ascending cos(m₂,g) → most-rotated-from-g first
    with open(path, "a") as f:
        f.write(f"M2G step={step} n={len(rows)} medCos_m2g={med_mg:.3f} medCos_self={med_self:.3f} "
                "| lowestCos_m2g: "
                + " ".join(f"{r[0].split('core.', 1)[1]}={r[3]:.2f}" for r in rows[:4]) + "\n")
        for nm, m2n, gn, cmg, cself, age in rows:
            f.write(f"  M2G_T step={step} {nm} m2n={m2n:.3e} gn={gn:.3e} "
                    f"cos_m2g={cmg:.4f} cos_self={cself:.4f} age={age}\n")


def diag_m2g_numerator(optimizer, model, step: int, path: str, dense: bool,
                       name_filter=None):
    """Drain the optimizer's g-vs-numerator geometry capture.

    Metrics (computed inside the optimizer step from the exact working copies):
      cos(g, m₂)            → if this drops after a prune event, the slow EMA is stale relative
                               to the new landscape.
      cos(g, α·m₂+gated_g) → whether the full numerator still aligns with the current gradient.
      ‖α·m₂‖ / ‖gated_g‖   → how much the stale term outweighs the fresh (gated) gradient.
    Re-arms capture lazily after any optimizer rebuild (compact/route swap the param objects).
    Writes medians every call; per-tensor rows only when `dense` (near prune events) to keep
    log size bounded. Returns a dict of medians for wandb.
    """
    base = getattr(optimizer, "_opt", optimizer)
    if not hasattr(base, "set_diag_capture"):
        return None                                   # not an AdEMAMixB1Zero (e.g. AdamW)
    if not base._diag_capture or not base._diag_names:
        base.set_diag_capture(model, name_filter=name_filter)  # (re)arm after build/rebuild
    rows = base._diag_rows
    base._diag_rows = []                              # drain — bounded memory
    if not rows:
        return None
    import statistics as _stat
    cos_m2g = [r[1] for r in rows]
    cos_num = [r[2] for r in rows]
    ratio = [r[3] for r in rows]
    med = {
        "diag/cos_g_m2_med": _stat.median(cos_m2g),
        "diag/cos_g_num_med": _stat.median(cos_num),
        "diag/alpha_m2_over_gatedg_med": _stat.median(ratio),
        "diag/cos_g_m2_min": min(cos_m2g),
    }
    rows.sort(key=lambda r: r[1])                     # ascending cos(g,m₂) → most-stale first
    with open(path, "a") as f:
        f.write(f"M2N step={step} n={len(rows)} medCos_g_m2={med['diag/cos_g_m2_med']:.3f} "
                f"medCos_g_num={med['diag/cos_g_num_med']:.3f} "
                f"medRatio_am_gg={med['diag/alpha_m2_over_gatedg_med']:.3f} "
                f"minCos_g_m2={med['diag/cos_g_m2_min']:.3f}"
                + (" PRUNE_WINDOW" if dense else "") + "\n")
        if dense:
            for nm, cgm, cgn, rt, m2n, gn, amn, ggn in rows:
                short = nm.split("core.", 1)[-1]
                f.write(f"  M2N_T step={step} {short} cos_g_m2={cgm:.4f} cos_g_num={cgn:.4f} "
                        f"am/gg={rt:.3f} m2n={m2n:.3e} gn={gn:.3e} amn={amn:.3e} ggn={ggn:.3e}\n")
    return med


def diag_forward_norms(model, step: int, path: str) -> None:
    """Forward-side blow-up localizer (env MORPH_DIAG_FWD=1, writes to MORPH_DIAG_OPT path).

    When optimizer-state diagnostics show no anomaly but loss explodes, the cause is likely
    forward-side: a residual-stream activation blowup that the looped core amplifies. This
    probe captures, per training step:

      FWDNORM : per-block output (residual-stream) L2 norm — localizes which block's
                activations grow first. Core blocks run T× per step; the max over iterations
                is reported.
      TERNFLIP: count of backbone ternary weights that changed {-1,0,+1} state since the
                previous step — tests the hypothesis of mass ternary sign-flip leading to a
                discontinuous effective-weight change. Pre-carve _ternary_mode path only.

    Hooks are registered once (idempotent). Run with MORPH_DIAG_OPT_EVERY=1 for per-step norms.
    """
    import torch
    import torch.nn.utils.parametrize as _P
    from morph.model.ternary_qat import TernarySTE

    root = getattr(model, "_orig_mod", model)

    # ── Lazy one-time hook registration ────────────────────────────────────
    if not getattr(model, "_diag_fwd_hooked", False):
        model._diag_fwd = {}

        def _pre_hook(_m, _inp):
            model._diag_fwd.clear()

        def _mk(name):
            def _hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    n = t.detach().float().norm().item()
                    d = model._diag_fwd
                    if n > d.get(name, 0.0):
                        d[name] = n
            return _hook

        root.register_forward_pre_hook(_pre_hook)
        for sec in ("prelude", "core", "coda"):
            mods = getattr(root, sec, None)
            if mods is None:
                continue
            for i, blk in enumerate(mods):
                blk.register_forward_hook(_mk(f"{sec}.{i}"))
        for nm in ("embed", "final_norm"):
            sub = getattr(root, nm, None)
            if sub is not None:
                sub.register_forward_hook(_mk(nm))
        model._diag_fwd_hooked = True
        model._diag_prev_tern = {}
        # First call: hooks fire on the NEXT forward → nothing to report yet.
        return

    fwd = dict(model._diag_fwd)
    if not fwd:
        return

    # ── Ternary {-1,0,+1} state-flip count since previous step ─────────────────
    # Ternary backbone = a TernarySTE PARAMETRIZATION on module.weight (NOT the CMSBlockLinear
    # _ternary_mode flag). The parametrization runs on .weight access → m.weight IS the realized
    # ternary (code·scale, scale>0) → sign(m.weight) == the {-1,0,+1} code exactly. Count how
    # many codes changed vs the previous step, totalled + per-section, to test whether a CORE-layer
    # flip burst coincides with the single-step core residual-stream excursion.
    prev = model._diag_prev_tern
    flip_total = 0
    flip_worst = (-1, None)
    flip_sec = {}   # section -> flip count this step
    with torch.no_grad():
        for name, m in root.named_modules():
            if not _P.is_parametrized(m, "weight"):
                continue
            if not any(isinstance(p, TernarySTE) for p in m.parametrizations.weight):
                continue
            tern = torch.sign(m.weight.detach()).to(torch.int8)   # {-1,0,+1} code
            old = prev.get(name)
            if old is not None and old.shape == tern.shape:
                f = int((tern != old).sum().item())
                flip_total += f
                sec = name.split(".")[0]
                flip_sec[sec] = flip_sec.get(sec, 0) + f
                if f > flip_worst[0]:
                    flip_worst = (f, name)
            prev[name] = tern
    flip_sec_str = " ".join(f"{s}={flip_sec[s]}" for s in sorted(flip_sec, key=lambda k: -flip_sec[k]))

    # ── Worst block + per-section max residual-stream norm ──────────────────
    worst = max(fwd.items(), key=lambda kv: kv[1])
    sec_max = {}
    for k, v in fwd.items():
        sec = k.split(".")[0]
        if v > sec_max.get(sec, 0.0):
            sec_max[sec] = v
    sec_str = " ".join(f"{s}={sec_max[s]:.3e}" for s in
                       ("embed", "prelude", "core", "coda", "final_norm") if s in sec_max)
    with open(path, "a") as f:
        f.write(
            f"  FWDNORM step={step} worstBlock={worst[0]}={worst[1]:.3e} | {sec_str}\n"
            f"  TERNFLIP step={step} total={flip_total} worst={flip_worst[0]}@{flip_worst[1]} | {flip_sec_str}\n"
        )


def warmup_compile_all_shapes(
    model, batch_size: int, seq_len: int, device, passes_per_size: int,
    tag: str = "startup",
) -> None:
    """Forced-depth fwd+bwd passes so EVERY compile variant builds NOW, not mid-loop.

    Forces the active-set to hit every sub-batch size (incl. n_active==1, the rare
    Poisson draw) in BOTH the no_grad prefix and the checkpointed BPTT window, so
    fwd AND bwd variants of every torch.compile guard-set and every hand-written
    Triton kernel size-specialization compile here. Shared by the thread-free
    startup window and the MORTAR/route phase-boundary recompile — see the two
    call sites for the (different) fork-safety reasoning at each.
    """
    mx = int(model.cfg.max_depth)

    def _forced(K):
        d = [1] * batch_size
        for j in range(min(K, batch_size)):
            d[j] = mx
        return torch.tensor(d, device=device, dtype=torch.long)

    orig_sample = model._sample_depths
    sizes = list(range(batch_size, 0, -1))   # [B, B-1, ..., 1] — size>1 AND size==1
    print(f"  Warmup compile [{tag}] (active-set sizes {sizes} × {passes_per_size})...",
          flush=True)
    t0 = time.perf_counter()
    try:
        for K in sizes:
            model._sample_depths = (lambda _b, _dev, _K=K: _forced(_K))
            for _ in range(passes_per_size):
                ids = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len),
                                    device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids, labels=ids)
                out["loss"].backward()
                model.zero_grad(set_to_none=True)
                del ids, out
    finally:
        model._sample_depths = orig_sample          # restore real Poisson sampling
    torch.cuda.synchronize()
    print(f"  Warmup compile [{tag}] done in {time.perf_counter()-t0:.1f}s "
          f"({len(sizes) * passes_per_size} passes, all active-set sizes)", flush=True)


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
    init_from_path: Optional[str] = getattr(tr, "init_from", None)

    # ── Token-Superposition Training (TST, #231 — arXiv 2605.06546) ──────────
    # Two phases in ONE run: superposition (bag_size=s, multi-hot CE) for the first
    # tst_ratio·total_steps, then recovery (bag_size=0, standard NTP — code path
    # inactive). bag_size is a per-forward kwarg (eval/gen always use 0). The switch
    # is in-process: only the MLP submodules are compiled and they see [*, L, d] in
    # BOTH phases (bagging happens before the loop), so the switch triggers no
    # recompile. tst_bag_size=0 → bit-identical to the pre-TST baseline.
    tst_bag_size = int(getattr(tr, "tst_bag_size", 0))
    tst_ratio = float(getattr(tr, "tst_ratio", 0.0))
    tst_phase1_steps = int(tst_ratio * total_steps) if tst_bag_size > 0 else 0
    if tst_bag_size > 0:
        print(f"  TST ON: bag_size={tst_bag_size} ratio={tst_ratio} → superposition "
              f"steps [0,{tst_phase1_steps}), recovery [{tst_phase1_steps},{total_steps})")

    use_compile = bool(getattr(tr, "compile", True))
    compile_mode = str(getattr(tr, "compile_mode", "default"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model ───────────────────────────────────────────────────────
    # ORDERING IS LOAD-BEARING: model build + torch.compile + warmup run BEFORE
    # wandb.init() and the streaming dataloader. All Triton/Inductor compilation —
    # including the gcc subprocess fork that builds each kernel launcher — therefore
    # happens in a single-threaded process. A fork deadlocks only when a thread holds a
    # non-reentrant lock (glibc malloc arena) at fork time; with no wandb/httpx threads
    # alive yet, every compile fork is safe. The fused CCA kernels JIT-specialize size==1
    # separately; without this ordering, the first n_active==1 Poisson draw would compile
    # against live threads. wandb.init() is deferred to just after the warmup.
    morph_cfg = build_morph_config(cfg)
    # Seed the CPU RNG before model construction so the seed perturbs weight INIT (Linear params are
    # created on CPU via the default generator, THEN .to(device)), not just data-loader order. Lets a
    # replication vary BOTH init and data across seeds. (No-op for prior runs that used seed=0.)
    # ⚠️ CPU-ONLY on purpose: do NOT call torch.cuda.manual_seed_all here — it eagerly inits the CUDA
    # context/driver threads BEFORE the single-threaded compile-fork window (see comment ~L1165), so a
    # later Inductor compile fork deadlocks on a held lock (observed: 119 threads, futex_wait hang).
    # CUDA RNG is restored from the ckpt anyway, so seeding it here is both unnecessary and unsafe.
    torch.manual_seed(int(getattr(cfg.training, "seed", 0)))
    model = MORPHTransformer(morph_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")

    # Core-map spectral-norm penalty: soft hinge L=λ·Σ relu(σ_max(W)−cap)² over core MLP
    # linears. Binds module references (stable across compile/resume). penalty() calls
    # forward() at train time so the ternary STE is applied live. OFF when cap or lambda ≤ 0
    # (exact baseline, never constructed).
    _spec_pen = None
    _sp_cap = float(getattr(cfg.training, "spectral_penalty_cap", 0.0))
    _sp_lam = float(getattr(cfg.training, "spectral_penalty_lambda", 0.0))
    if _sp_cap > 0.0 and _sp_lam > 0.0:
        from morph.training.spectral_penalty import CoreSpectralPenalty
        _spec_pen = CoreSpectralPenalty(model, cap=_sp_cap, lam=_sp_lam,
                                        n_iter=int(getattr(cfg.training, "spectral_penalty_n_iter", 1)))
        print(f"  Core spectral-norm penalty ON: cap={_sp_cap} lambda={_sp_lam} "
              f"on {len(_spec_pen._linears)} core MLP linears")

    # ── Ternary QAT (forward-STE) ──────────────────────────────────────────
    # MUST run BEFORE torch.compile (so the STE is captured in the compiled graph)
    # and BEFORE create_optimizer (so the optimizer binds the smooth `.original`
    # params). When active, the forward uses {-1,0,+1}×scale weights → training/val
    # ppl already reflects the deployed-ternary quality. See morph/model/ternary_qat.py.
    ternary_manifest = None
    if bool(getattr(cfg.training, "ternary", False)):
        from morph.model.ternary_qat import apply_ternary_qat
        ternary_manifest = apply_ternary_qat(
            model,
            scope=str(getattr(cfg.training, "ternary_scope", "backbone")),
            threshold=float(getattr(cfg.training, "ternary_threshold", 0.5)),
            scale_mode=str(getattr(cfg.training, "ternary_scale_mode", "symmetric")),
            scale_group=str(getattr(cfg.training, "ternary_scale_group", "tensor")),
            scale_dtype=str(getattr(cfg.training, "ternary_scale_dtype", "fp16")),
            scale_clip_mult=float(getattr(cfg.training, "ternary_scale_clip_mult", 0.0)),
        )
        print(
            f"  Ternary QAT ON: scope={ternary_manifest['scope']} "
            f"threshold={ternary_manifest['threshold']} "
            f"mode={ternary_manifest['scale_mode']} "
            f"group={ternary_manifest['scale_group']} "
            f"dtype={ternary_manifest['scale_dtype']} "
            f"modules={ternary_manifest['n_modules_ternary']} "
            f"({ternary_manifest['counts']}) "
            f"params_ternary={ternary_manifest['n_params_ternary'] / 1e6:.1f}M "
            f"({ternary_manifest['frac_params_ternary'] * 100:.1f}% of model)",
            flush=True,
        )

    # ── Embedding QAT (int8/int6 per-row, Ablation E) ─────────────────────
    # Applies AFTER ternary (disjoint: ternary targets Linear/CMSBlockLinear,
    # embed_quant targets nn.Embedding). BEFORE torch.compile so the parametrize
    # hook is in the compiled graph. Lorentz space embed is ALWAYS skipped.
    embed_quant_manifest = None
    # Normalize defensively: bare `off`/`on` in YAML parse as bools (YAML 1.1), so a
    # config value can arrive as False/"False" rather than "off". Map those to "off".
    _embed_quant_mode = str(getattr(cfg.training, "embed_quant", "off")).strip().lower()
    if _embed_quant_mode in ("false", "none", ""):
        _embed_quant_mode = "off"
    _lm_head_quant_mode = str(getattr(cfg.training, "lm_head_quant", "off")).strip().lower()
    if _lm_head_quant_mode in ("false", "none", ""):
        _lm_head_quant_mode = "off"
    if _embed_quant_mode != "off":
        from morph.model.embed_quant import apply_embed_quant
        embed_quant_manifest = apply_embed_quant(
            model,
            embed_quant=_embed_quant_mode,
            lm_head_quant=_lm_head_quant_mode,
        )
        print(
            f"  Embed QAT ON: mode={embed_quant_manifest['embed_quant']} "
            f"modules={embed_quant_manifest['n_modules_quantized']} "
            f"({embed_quant_manifest['module_names']}). "
            f"LM head: {embed_quant_manifest['lm_head_note'][:80]}",
            flush=True,
        )

    # ── CMS importance-scoring mode (for structured pruning) ──────────────
    # Sets the saliency criterion used by accumulate_scores / prune_step on every
    # CMSBlockLinear. Default "grad" is bit-identical to the pre-pruning behaviour.
    #   grad → ‖∇W‖_F · taylor → ‖W⊙∇W‖_F (Molchanov) · magnitude → ‖W‖_F
    _cms_score_mode = str(getattr(cfg.training, "cms_score_mode", "grad")).strip().lower()
    if _cms_score_mode not in ("grad", "taylor", "magnitude"):
        raise ValueError(f"cms_score_mode must be grad|taylor|magnitude, got {_cms_score_mode!r}")
    if _cms_score_mode != "grad":
        from morph.model.layers.block_sparse import CMSBlockLinear
        _n_cms = 0
        for _m in model.modules():
            if isinstance(_m, CMSBlockLinear):
                _m.score_mode = _cms_score_mode
                _n_cms += 1
        print(f"  CMS score_mode={_cms_score_mode} on {_n_cms} CMSBlockLinear layers", flush=True)

    # ── Attention-projection int-N QAT (Ablation #205) ────────────────────
    # Gentler-than-ternary per-row int8/int6/int4 on the CCA attention projections —
    # the Efull-recovery lever. Runs AFTER ternary (disjointness: the #205 stack uses
    # ternary scope=backbone, so attention Linears are free) and BEFORE torch.compile so
    # the STE is captured. attn_proj_quant=off → bit-identical bf16. See attn_proj_quant.py.
    attn_proj_quant_manifest = None
    _attn_proj_mode = str(getattr(cfg.training, "attn_proj_quant", "off")).strip().lower()
    if _attn_proj_mode in ("false", "none", ""):
        _attn_proj_mode = "off"
    if _attn_proj_mode != "off":
        from morph.model.attn_proj_quant import apply_attn_proj_quant
        attn_proj_quant_manifest = apply_attn_proj_quant(
            model,
            attn_proj_quant=_attn_proj_mode,
            ternary_module_names=(ternary_manifest or {}).get("module_names"),
        )
        print(
            f"  Attn-proj QAT ON: mode={attn_proj_quant_manifest['attn_proj_quant']} "
            f"bits={attn_proj_quant_manifest['bits']} "
            f"modules={attn_proj_quant_manifest['n_modules_quantized']} "
            f"params={attn_proj_quant_manifest['n_params_quantized'] / 1e6:.2f}M "
            f"skipped_already_param={len(attn_proj_quant_manifest['skipped_already_parametrized'])}",
            flush=True,
        )

    # ── FP8 training (torchao float8) ──────────────────────────────────────
    # MUST run AFTER ternary QAT (for the disjointness guard) and BEFORE torch.compile
    # (so Float8Linear is compiled). Converts only the scoped dense GEMMs; dynamic
    # scaling (stateless — safe in the reused-weight loop). See morph/model/fp8_scope.py.
    fp8_manifest = None
    if bool(getattr(cfg.training, "fp8", False)):
        from morph.model.fp8_scope import apply_fp8_training
        fp8_manifest = apply_fp8_training(
            model,
            scope=str(getattr(cfg.training, "fp8_scope", "mlp")),
            recipe=str(getattr(cfg.training, "fp8_recipe", "dynamic")),
            min_dim=int(getattr(cfg.training, "fp8_filter_min_dim", 256)),
            ternary_module_names=(ternary_manifest or {}).get("module_names"),
        )
        print(
            f"  FP8 training ON: scope={fp8_manifest['scope']} recipe={fp8_manifest['recipe']} "
            f"min_dim={fp8_manifest['min_dim']} converted={fp8_manifest['n_converted']} Linears",
            flush=True,
        )

    # ── torch.compile ─────────────────────────────────────────────────────
    # Compile only the MLP sub-modules (attention uses Triton/SDPA kernels,
    # which are incompatible with fullgraph compile).
    if use_compile:
        for group in [model.prelude, model.core, model.coda]:
            # Core MLPs see a VARIABLE batch each loop iteration (active-set
            # shrinking processes the still-active prefix), so compile them with
            # dynamic batch to avoid a recompile per distinct sub-batch size.
            # Prelude/coda see a fixed batch → let Dynamo auto-decide (None).
            dyn = True if group is model.core else None
            for layer in group:
                if hasattr(layer, "mlp"):
                    layer.mlp = torch.compile(layer.mlp, mode=compile_mode, dynamic=dyn)
        print(f"  MLPs compiled (mode={compile_mode}, core dynamic-batch)")

        # ── Warmup compile — runs in the THREAD-FREE window (pre-wandb, pre-dataloader) ──
        # Two compilation systems fork subprocesses here and must finish before any thread
        # spawns: (a) torch.compile/Inductor for the MLPs, which lazily compiles a variant
        # per (sub-batch size × grad_mode × autocast dtype) guard on the first few forwards;
        # (b) the hand-written fused CCA Triton kernels, which JIT-specialize size==1 apart
        # from size>1. Either, if compiled DURING the training loop, forks gcc/Inductor
        # workers while the HF-streaming httpx + wandb threads hold a glibc malloc-arena lock
        # → the forked child deadlocks in __triton_launcher.c (intermittent; cost us a night
        # — Ai-notes 06-01-2026/MORPH-eval-recompile-hang). Mitigation is layered: (1) build +
        # warm up BEFORE wandb.init/dataloader so the fork window is single-threaded → safe by
        # construction; (2) the forced-size loop below compiles EVERY Triton variant (incl.
        # the rare size==1) here, so none JIT-compiles at runtime; (3) eager_on_recompile
        # (set after warmup) catches any leftover MLP guard → runs it eager (no compile, no
        # fork) rather than recompiling mid-loop. Raise the Dynamo cache limit so all variants
        # coexist without eviction.
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = max(getattr(_dynamo.config, "cache_size_limit", 8), 64)
        _dynamo.config.accumulated_cache_size_limit = max(
            getattr(_dynamo.config, "accumulated_cache_size_limit", 256), 512)

        # Force the active-set to hit EVERY sub-batch size (incl. n_active==1) so all
        # Triton kernel variants compile HERE, in this thread-free window. The fused CCA
        # attention kernels are hand-written Triton (NOT torch.compile), so the stance
        # below does NOT govern them — Triton JIT-specializes size==1 separately from
        # size>1. If the size-1 variant is left to compile on the first runtime n_active==1
        # (a rare Poisson draw with one sequence far deeper than the rest), it forks gcc for
        # its launcher stub while wandb/httpx threads hold the glibc malloc-arena lock → the
        # forked child deadlocks (the intermittent step-0 wedge; py-spy caught the autograd
        # engine blocked mid-recompute on the next malloc). Patterns: K sequences at
        # max_depth, rest at depth 1 → n_active==K in BOTH the no_grad prefix and the
        # checkpointed BPTT window, so fwd AND bwd Triton variants for every size compile now.
        warmup_compile_all_shapes(
            model, int(cfg.training.batch_size), seq_len, device,
            int(getattr(tr, "warmup_passes_per_size", 4)), tag="startup thread-free",
        )

        # Final safety net: forbid NEW compilation during the training loop. The warmup
        # above + the @torch.compiler.disable on the CMS stats hook cover the COMMON shape
        # space (verified: 100 steps, 0 recompiles), so this rarely fires. But a rare
        # Poisson-depth draw can still produce an (size × grad_mode × dtype) combo the
        # warmup missed — and any such recompile would fork gcc for the Triton launcher in
        # the MAIN process (NOT covered by the spawn worker pool) against wandb/httpx
        # threads → intermittent deadlock (this bit the real campaign at step 1 while
        # 160 diag steps ran clean — pure timing luck). "eager_on_recompile" makes that
        # rare uncovered shape run EAGER (no compile, no fork, no deadlock) instead — one
        # slightly-slow step, never a hang. Common shapes keep their compiled kernels.
        torch.compiler.set_stance("eager_on_recompile")
        print("  torch.compiler stance = eager_on_recompile (rare uncovered shapes run eager, never recompile/fork)", flush=True)

    # ── W&B init — log FULL config dict ──────────────────────────────────
    # DEFERRED until AFTER the warmup: the compile/gcc-fork window above must be
    # thread-free (no wandb asyncio/httpx threads) for the fork to be deadlock-safe.
    # OmegaConf → plain Python dict so wandb can serialise it; fold n_params in directly.
    full_config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    full_config_dict["n_params"] = n_params
    if ternary_manifest is not None:
        # Derived ternary facts (scope/threshold already live in cfg). Drop the
        # verbose module_names list from the logged config; keep the greppable counts.
        full_config_dict["ternary_manifest"] = {
            k: v for k, v in ternary_manifest.items() if k != "module_names"
        }
    if fp8_manifest is not None:
        full_config_dict["fp8_manifest"] = {
            "scope": fp8_manifest["scope"], "recipe": fp8_manifest["recipe"],
            "min_dim": fp8_manifest["min_dim"], "n_converted": fp8_manifest["n_converted"],
        }
    if embed_quant_manifest is not None:
        full_config_dict["embed_quant_manifest"] = {
            k: v for k, v in embed_quant_manifest.items()
            if k not in ("module_names", "lm_head_note")
        }
    if attn_proj_quant_manifest is not None:
        # Keep the greppable counts/bits; drop the verbose per-module name list.
        full_config_dict["attn_proj_quant_manifest"] = {
            k: v for k, v in attn_proj_quant_manifest.items() if k != "module_names"
        }
    # Resume the same wandb run (continuous metric history) when resuming a checkpoint:
    # the prior run wrote its id to a wandb_id.txt sidecar next to its checkpoints.
    _wandb_resume_id = None
    if resume_path and os.path.isfile(resume_path):
        _sidecar = os.path.join(os.path.dirname(resume_path), "wandb_id.txt")
        if os.path.isfile(_sidecar):
            _wandb_resume_id = (open(_sidecar).read().strip() or None)
            if _wandb_resume_id:
                print(f"  [wandb] resuming run id {_wandb_resume_id}", flush=True)
    wandb.init(
        project=wb_cfg.project,
        entity=getattr(wb_cfg, "entity", None),
        name=getattr(wb_cfg, "name", None),
        config=full_config_dict,
        id=_wandb_resume_id,
        resume=("allow" if _wandb_resume_id else None),
        settings=wandb.Settings(_service_wait=60),
    )

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

    def _make_train_loader(bag: int, skip_batches: int = 0):
        it = iter(create_dataloader(tokenizer_name, dataset_name, seq_len,
                                    batch_size, split="train", bag_size=bag))
        # Resume: the stream is DETERMINISTIC and UNSHUFFLED (fixed shard order, no per-epoch
        # seed), so replaying `skip_batches` next() calls advances to the EXACT batch the
        # interrupted run would serve next — "like nothing happened" for data too. Cost is
        # re-tokenizing the skipped prefix (CPU, ~1-2 min for a few-k-step resume); logged.
        if skip_batches > 0:
            print(f"  [data] fast-forwarding train stream by {skip_batches} batches "
                  f"(deterministic replay to exact resume position)…", flush=True)
            t_ff = time.perf_counter()
            for _ in range(skip_batches):
                next(it)
            print(f"  [data] fast-forward done in {time.perf_counter() - t_ff:.1f}s", flush=True)
        return it

    # val/gen ALWAYS use standard NTP (bag_size=0) so val ppl is comparable to the
    # baseline regardless of which TST phase training is in.
    val_loader = iter(
        create_dataloader(tokenizer_name, dataset_name, seq_len, batch_size,
                         split="validation", skip_samples=50_000)
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(_MORPH_ROOT, "checkpoints", "morph",
                            wandb.run.name if wandb.run else "run")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Persist the wandb run id so a future resume continues the same run (read back as
    # the wandb_id.txt sidecar above, before wandb.init).
    if wandb.run is not None:
        try:
            with open(os.path.join(ckpt_dir, "wandb_id.txt"), "w") as _f:
                _f.write(str(wandb.run.id))
        except OSError as e:
            print(f"  [wandb] could not write run-id sidecar ({e}); resume will start a new run")

    # Generation samples go to a sidecar file, NOT stdout. Generated text is
    # uncontrolled token output that can contain substrings ("RuntimeError:",
    # "Killed", "Traceback...") which would false-trigger a log-scraping watcher
    # (ignore/ab_watch.sh) into reporting a crash. Stdout gets only a safe summary.
    gen_samples_path = os.path.join(ckpt_dir, "generation_samples.txt")

    def _emit_gen(label: str, gen_text: str) -> None:
        with open(gen_samples_path, "a") as _f:
            _f.write(f"\n===== {label} =====\n{gen_text}\n")
        print(f"  [GEN {label}] {len(gen_text)} chars → {gen_samples_path}", flush=True)

    # ── Optional resume (FULL: model+topology+optimizer+scaler+RNG+step) ────
    start_step = 0
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        start_step, _opt_state, _needs_rebuild = load_checkpoint(
            resume_path, model, scaler, device, pruning)
        if _needs_rebuild:
            # Carve/route changed the param set → the dense optimizer built above is stale.
            # Free it (bnb keeps optimizer↔state↔param ref-cycles → explicit clear+gc, same
            # pattern as the in-loop phase-boundary rebuild) and rebuild on the NOW-
            # reconstructed (carved/routed) topology so its state keys line up before load.
            optimizer.zero_grad(set_to_none=True)
            if hasattr(optimizer, "state"):
                optimizer.state.clear()
            optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
            optimizer = create_optimizer(model, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_fn(start_step)
            print("  [opt] rebuilt optimizer on reconstructed (carved/routed) topology",
                  flush=True)
        # Restore momentum/variance. HARD-FAIL on mismatch — swallowing it here would
        # silently continue training with ZERO momentum (the theater this whole task kills).
        # EXCEPTION — resume_fresh_optimizer=True (FORK-continue / new experiment off a ckpt):
        # deliberately start with a FRESH optimizer (no inherited momentum). This is the correct
        # choice for an A/B fork (both arms start identical; no stale 50k slow-EMA confound) and
        # it also sidesteps the bnb-8bit per-param "step"-key restore path. Topology + weights +
        # RNG are still restored above; ONLY the optimizer STATE is intentionally dropped.
        if bool(getattr(cfg.training, "resume_fresh_optimizer", False)):
            print("  [opt] resume_fresh_optimizer=True → FRESH optimizer (momentum starts at "
                  "zero; topology+weights+RNG restored). Fork-continue mode.", flush=True)
        else:
            optimizer.load_state_dict(_opt_state)
            _n_restored = sum(len(g["params"]) for g in optimizer.param_groups)
            print(f"  [opt] optimizer state restored ({_n_restored} param tensors)", flush=True)
        # MEMORY: optimizer.load_state_dict deep-copies into the live optimizer's tensors;
        # the checkpoint copy is a dead duplicate (~1.7GB on GPU for this model) that
        # otherwise lingers for the whole run (a local held by the train() frame). Dropping
        # it + empty_cache() also returns freed-but-reserved blocks from torch.load.
        del _opt_state
        gc.collect()
        torch.cuda.empty_cache()
    elif init_from_path:
        # Weights-only seed (step stays 0, fresh optimizer). resume takes precedence if both set.
        if not os.path.isfile(init_from_path):
            raise FileNotFoundError(f"training.init_from not found: {init_from_path}")
        print(f"Init-from (weights only) {init_from_path}")
        load_weights_only(init_from_path, model, device)

    # TST: start in superposition unless resuming past the phase-1 boundary.
    cur_bag = tst_bag_size if (tst_phase1_steps > 0 and start_step < tst_phase1_steps) else 0
    # Data fast-forward to the exact resume position (deterministic unshuffled stream). Only
    # for the base (non-curriculum) loader — the curriculum multi-source loader is rebuilt
    # below with its own stage logic, so skipping here would be wasted re-tokenization.
    _curr_on = bool(getattr(cfg, "curriculum", None) is not None
                    and getattr(cfg.curriculum, "enabled", False))
    # Fork-continue (fresh optimizer) is explicitly NOT a faithful resume → no point replaying
    # the deterministic stream to the exact position; start from the stream head (saves the
    # ~10min/arm CPU re-tokenization). Faithful resume still replays for exact continuation.
    _fork_continue = bool(getattr(cfg.training, "resume_fresh_optimizer", False))
    _resume_skip = start_step if (start_step > 0 and not _curr_on and not _fork_continue) else 0
    train_loader = _make_train_loader(cur_bag, skip_batches=_resume_skip)

    # ── Curriculum pretraining (Phase P) — length-bucketed multi-source ramp ──
    # GATED: absent/disabled → base.yaml path is byte-identical (curriculum_enabled False,
    # cur_grad_accum 1, no transitions, total_steps unchanged). When ON: overrides total_steps
    # and train_loader, and ramps seq_len / RoPE-context / micro-batch per stage with a
    # checkpoint-before-step-up (the stage transition fires at the top of the loop below).
    _curr_cfg = getattr(cfg, "curriculum", None)
    curriculum_enabled = bool(_curr_cfg is not None and getattr(_curr_cfg, "enabled", False))
    cur_grad_accum = 1
    cur_stage = -1
    if curriculum_enabled:
        from morph.training.curriculum import CurriculumScheduler
        from morph.training.curriculum_data import MultiSourceCurriculumLoader
        from morph.model.attention import CoPEEmbedding
        _stages = list(_curr_cfg.stages)
        _boundaries = [int(s.seq_len) for s in _stages]
        _contexts = [int(s.context_len) for s in _stages]
        _microbatch = [int(s.micro_batch) for s in _stages]
        _stage_steps = [int(s.steps) for s in _stages]
        _eff_batch = int(getattr(_curr_cfg, "eff_batch", 8))
        _weights = {str(k): float(v) for k, v in dict(_curr_cfg.blend).items()}
        _allowed_roles = [str(x) for x in getattr(
            _curr_cfg, "allowed_source_roles", ("pretrain_bulk", "reasoning_midtrain")
        )]
        _sched = CurriculumScheduler(_stage_steps)
        total_steps = _sched.total_steps                      # override training.steps
        _curr_loader = MultiSourceCurriculumLoader(
            str(_curr_cfg.pretok_dir), _weights, _boundaries,
            seed=int(getattr(tr, "seed", 0)), allowed_roles=_allowed_roles)
        # RoPE modules to re-anchor on each step-up (attention is EAGER → safe to mutate
        # cos/sin cache mid-run; compile only wraps the MLPs). Reach through _orig_mod.
        _rope_mods = [m for m in getattr(model, "_orig_mod", model).modules()
                      if isinstance(m, CoPEEmbedding)]
        def _ceil_div(a, b):
            return max(1, -(-a // b))
        print(f"[curriculum] ENABLED: {len(_stages)} stages seq={_boundaries} "
              f"context={_contexts} micro_batch={_microbatch} eff_batch={_eff_batch} "
              f"stage_steps={_stage_steps} total_steps={total_steps} "
              f"allowed_roles={_allowed_roles} | "
              f"{len(_rope_mods)} RoPE modules", flush=True)

    # ── Optimizer step closure (resolved once, no per-step isinstance) ───
    def _step_optimizer():
        scaler.step(optimizer)
        scaler.update()

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    step_times: list[float] = []
    t_start = time.perf_counter()
    # In-process divergence guard: counts consecutive eval-cadence points with train ppl
    # over a hard ceiling (after step 2000); aborts after K consecutive strikes so finite
    # prune-bounces (ppl≲70) don't trip it but a real divergence (ppl→1e5) does. Env-tunable.
    _div_ceiling = float(os.environ.get("MORPH_DIV_PPL", "1000"))
    _div_strikes_max = int(os.environ.get("MORPH_DIV_STRIKES", "2"))
    _div_strikes = 0
    _aborted = False  # set by the non-finite / divergence guards → skip the post-loop final
                      # save+eval so a DIVERGED_step_N.pt is NOT shadowed by a misleading
                      # "completed" step_{total_steps}.pt (no-theater: don't fake a finished run).

    # ── Activation-memory probe (MORPH_MEM_PROBE) ──────────────────────────────
    # Root-causes the post-compact activation regression (+8 GB at b4). When set, we
    # reset the peak counter at the START of each step and print THAT step's fwd+bwd
    # high-water mark — correlate with the [compact]/[route] log lines to read the
    # masked-dense → sparse → routed deltas WITHIN ONE faithful training process.
    # MORPH_MEM_SNAPSHOT_STEP=N additionally dumps a full allocation snapshot (every
    # block + its Python alloc stack) at the first step >= N, for line-level attribution.
    _mem_probe = bool(os.environ.get("MORPH_MEM_PROBE"))
    _diag_optstate_path = os.environ.get("MORPH_DIAG_OPT")  # prune-divergence root-cause probe
    _diag_optstate_every = int(os.environ.get("MORPH_DIAG_OPT_EVERY", "1"))  # stride (1 = every step)
    _diag_fwd = bool(os.environ.get("MORPH_DIAG_FWD"))      # forward-side blow-up localizer
    _diag_m2g_path = os.environ.get("MORPH_DIAG_M2G")       # slow-EMA m₂/g geometry localizer
    # Prune cadence (for the M2N stale-m₂ diagnostic: log per-tensor rows DENSELY in a ±2-step
    # window around each prune event so cos(g,m₂) before vs after the topology change is visible).
    _prune_start = int(getattr(cfg.training, "prune_start", 10**9))
    _prune_interval = max(1, int(getattr(cfg.training, "prune_interval", 1)))
    # Core/CMS MLP params are where pruning acts → the stale-m₂ candidates we track.
    _m2g_filter = (lambda nm: "core." in nm)
    _mem_snap_step = int(os.environ.get("MORPH_MEM_SNAPSHOT_STEP", "-1"))
    _mem_snapped = False
    # History recording installs CUDA-allocator hooks that can make a Triton
    # autograd.Function (the fused HC kernels) return NULL — so we ONLY enable it when a
    # snapshot is explicitly requested (MORPH_MEM_SNAPSHOT_STEP>=0). The default probe is
    # peak-only (reset_peak_memory_stats + max_memory_allocated) which touches no hooks.
    if _mem_probe and _mem_snap_step >= 0:
        torch.cuda.memory._record_memory_history(max_entries=300_000)
        print(f"[memprobe] recording allocation history (snapshot @ step>={_mem_snap_step})",
              flush=True)
    elif _mem_probe:
        print("[memprobe] peak-only mode (no allocator hooks; set MORPH_MEM_SNAPSHOT_STEP for a snapshot)",
              flush=True)

    # ── Punctuation-boundary STP (step_boundary_mask for pretraining) ─────────────────────
    # Resolve punct token ids ONCE at startup when stp_mode=="step_boundary" and
    # stp_boundary_source=="punctuation".  Each training step then computes a [B,T] bool
    # mask from the raw input tokens; the mask is passed to model() so paper-STP fires at
    # punctuation positions in general pretraining text (bag_size=0 / TST OFF path only).
    # When stp_boundary_source=="none" OR stp_mode!="step_boundary", _punct_ids is None and
    # the model call is bit-identical to the pre-STP baseline (step_boundary_mask not passed).
    _stp_mode = str(getattr(cfg.model, "stp_mode", "off"))
    _stp_boundary_source = str(getattr(cfg.model, "stp_boundary_source", "none"))
    _stp_punct_include_comma = bool(getattr(cfg.model, "stp_punct_include_comma", False))
    _punct_ids: dict | None = None
    if _stp_mode == "step_boundary" and _stp_boundary_source == "punctuation":
        from transformers import AutoTokenizer as _AutoTok
        _ptok = _AutoTok.from_pretrained(tokenizer_name)
        print(f"[punct_boundary] Resolving punctuation token ids for stp_mode=step_boundary "
              f"(tokenizer={tokenizer_name}, include_comma={_stp_punct_include_comma})", flush=True)
        _punct_ids = resolve_punct_token_ids(
            _ptok,
            include_comma=_stp_punct_include_comma,
            include_newline=True,
        )
        print(f"[punct_boundary] Ready: {len(_punct_ids)} boundary token ids", flush=True)
        del _ptok  # tokenizer no longer needed; don't hold it in GPU memory

    # Diagnostic-only (MORPH_DEBUG_STEP): per-step wall time + the exact Poisson depths
    # that step sampled, to catch the intermittent slow step and its trigger. Wrap
    # _sample_depths to stash the last-returned depths; printed in the timing block.
    _dbg_step = bool(os.environ.get("MORPH_DEBUG_STEP"))
    _dbg = {"depths": None, "step_start": time.perf_counter(), "cur_step": -1, "dumped": False}
    if _dbg_step:
        _real_sample = model._sample_depths
        def _logged_sample(b, dev_, _f=_real_sample):
            d = _f(b, dev_)
            _dbg["depths"] = d.tolist()
            # print at SAMPLE time (step start) so a step that wedges still has its
            # trigger logged (the end-of-step timing print would never fire).
            print(f"  [dbg] >>> forward start depths={_dbg['depths']} max={int(d.max())}", flush=True)
            return d
        model._sample_depths = _logged_sample

        # Watchdog: if any single step exceeds 60s (vs ~0.6s normal), dump the FULL
        # all-thread stack ONCE — captures the wedge in situ (the failure only manifests
        # in the real campaign launcher, not in controlled repros). faulthandler shows the
        # autograd-engine thread's Python frame + every other thread → what is actually stuck.
        import faulthandler as _fh, threading as _thr
        def _watchdog():
            while True:
                time.sleep(5)
                wedged = time.perf_counter() - _dbg["step_start"]
                if wedged > 60 and not _dbg["dumped"] and _dbg["cur_step"] >= 0:
                    _dbg["dumped"] = True
                    print(f"\n[dbg] !!! WEDGE: step {_dbg['cur_step']} running {wedged:.0f}s "
                          f"depths={_dbg['depths']} — dumping all threads:\n", flush=True)
                    _fh.dump_traceback()
        _thr.Thread(target=_watchdog, daemon=True).start()

    for step in range(start_step, total_steps):
        if _dbg_step:
            _dbg["step_start"] = time.perf_counter()
            _dbg["cur_step"] = step
        if _mem_probe:
            torch.cuda.reset_peak_memory_stats()

        # ── TST phase switch: superposition → recovery (once, at tst_phase1_steps) ──
        if cur_bag != 0 and step >= tst_phase1_steps:
            sw_path = os.path.join(ckpt_dir, f"tst_switch_step_{step}.pt")
            save_checkpoint(sw_path, step, model, optimizer, scaler, pruning, next_step=step)
            print(f"[TST] phase switch @ step {step}: superposition (bag={cur_bag}) → "
                  f"recovery (bag=0). Switch ckpt: {sw_path}", flush=True)
            cur_bag = 0
            train_loader = _make_train_loader(0)

        # ── Curriculum stage transition: checkpoint → RoPE re-anchor → loader.set_stage →
        #    micro-batch/grad-accum swap. Two independent risks at a step-up (activation OOM
        #    and the PE-shift loss spike) → the pre-step-up checkpoint is the recovery point. ──
        if curriculum_enabled and _sched.stage_at(step) != cur_stage:
            _k = _sched.stage_at(step)
            if step > start_step:                                  # nothing to save at step 0
                _cp = os.path.join(ckpt_dir, f"curriculum_pre_stage{_k}_step{step}.pt")
                save_checkpoint(_cp, step, model, optimizer, scaler, pruning, next_step=step)
                print(f"[curriculum] stage {cur_stage}→{_k} @ step {step}: pre-step-up ckpt {_cp}",
                      flush=True)
            for _m in _rope_mods:                                  # re-anchor taper + rebuild cache
                _m.set_context(_contexts[_k])
            _curr_loader.set_stage(_k)
            cur_stage = _k
            cur_grad_accum = _ceil_div(_eff_batch, _microbatch[_k])
            seq_len = _boundaries[_k]
            batch_size = _microbatch[_k] * cur_grad_accum          # effective, for tok/s logging
            train_loader = _curr_loader.batches(_microbatch[_k], bag_size=cur_bag)
            print(f"[curriculum] → stage {_k}: seq_len={seq_len} context={_contexts[_k]} "
                  f"micro_batch={_microbatch[_k]} grad_accum={cur_grad_accum} eff_batch={batch_size} "
                  f"(RoPE re-anchored on {len(_rope_mods)} modules)", flush=True)

        lr = lr_fn(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # Grad accumulation: _ga micro-steps before one optimizer step. _ga==1 (no curriculum)
        # → byte-identical to the single fwd/bwd path (loss/1 == loss). The curriculum uses it to
        # hold a constant effective batch as the per-stage micro-batch drops with context length.
        _ga = cur_grad_accum if curriculum_enabled else 1
        for _micro in range(_ga):
            try:
                x, y = next(train_loader)
            except StopIteration:
                train_loader = _make_train_loader(cur_bag)
                x, y = next(train_loader)
            x, y = x.to(device), y.to(device)

            # ── Punctuation step-boundary mask (bag_size==0 path only) ──────────────
            # When stp_mode="step_boundary" + stp_boundary_source="punctuation", derive
            # the [B,T] bool boundary mask from punctuation token positions in x.
            # Only applied on the raw-token (bag_size=0) path: TST bagging scrambles
            # punctuation positions (bag_size>0 groups s tokens per position), so the
            # mask is meaningless under superposition and intentionally skipped there.
            # When _punct_ids is None (stp_mode!="step_boundary" or source="none"),
            # step_boundary_mask is not passed → bit-identical to pre-STP baseline.
            _step_boundary_mask: torch.Tensor | None = None
            if _punct_ids is not None and cur_bag == 0:
                _step_boundary_mask = punctuation_boundary_mask(x, _punct_ids, min_gap=2)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, labels=y, bag_size=cur_bag,
                            step_boundary_mask=_step_boundary_mask)
            loss = out["loss"]

            # Routing aux loss (load balance) — only active after route_start
            if pruning.is_routed:
                routing_aux = collect_routing_aux_losses(model)
                loss = loss + routing_aux

            # Core-map spectral-norm penalty. Zero while every core MLP linear is below
            # `cap` (bit-exact); only fires on σ_max runaway. Loss-side → optimizer-agnostic.
            if _spec_pen is not None:
                _sp = _spec_pen.penalty()
                loss = loss + _sp.to(loss.dtype)

            scaler.scale(loss / _ga).backward()

        if _mem_probe:
            _pk = torch.cuda.max_memory_allocated() / 2**30
            _rsv = torch.cuda.max_memory_reserved() / 2**30
            print(f"[memprobe] step={step} routed={pruning.is_routed} "
                  f"fwdbwd_peak_alloc={_pk:.2f}GB reserved={_rsv:.2f}GB", flush=True)
            if _mem_snap_step >= 0 and step >= _mem_snap_step and not _mem_snapped:
                _snap_path = os.environ.get("MORPH_MEM_SNAPSHOT_PATH",
                                            "experiments/mem_snapshot.pickle")
                torch.cuda.memory._dump_snapshot(_snap_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                _mem_snapped = True
                print(f"[memprobe] dumped allocation snapshot → {_snap_path} "
                      f"(recording stopped)", flush=True)

        prune_stats = pruning.step(model, step)

        # Phase boundary (compact / routing) changed the param set → rebuild a FRESH
        # optimizer (Wolfe: fresh optimizer after compact). This step's backward grads
        # live on the OLD params (weight, pre-router); the new params (values, router)
        # have no grads yet, so we skip this step's update and train normally next step.
        # _step_optimizer closes over `optimizer` by name → reassigning here is picked up.
        if prune_stats and prune_stats.pop("_rebuild_optimizer", False):
            wandb.log({k: v for k, v in prune_stats.items()
                       if isinstance(v, (int, float))}, step=step)
            # FREE the old optimizer BEFORE building the new one. The old AdamW8bit holds
            # 8-bit moment tensors for the PRE-rebuild param set (e.g. the now-deleted dense
            # `weight` Parameters at compact). bitsandbytes optimizers keep internal reference
            # CYCLES (optimizer ↔ state ↔ param), so plain reassignment does NOT free them via
            # refcounting — they linger as LIVE GPU memory until a cyclic GC pass. Without this
            # the dense-weight optimizer state survives compact+route and stacks on top of the
            # new sparse state → b4 OOM even though the compacted model is smaller. So: clear
            # state, drop the name, gc.collect() to break the cycle, empty_cache() to return the
            # freed blocks to the driver — THEN allocate the new optimizer into the cleared pool.
            _mem_before = torch.cuda.memory_allocated() / 1e9
            optimizer.zero_grad(set_to_none=True)
            if hasattr(optimizer, "state"):
                optimizer.state.clear()
            del loss
            optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
            _mem_freed = torch.cuda.memory_allocated() / 1e9
            optimizer = create_optimizer(model, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            _mem_after = torch.cuda.memory_allocated() / 1e9
            _n_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
            print(f"[opt] rebuilt optimizer @ step {step}: {_n_opt:,} params; "
                  f"cuda_alloc {_mem_before:.2f}→{_mem_freed:.2f} (freed)→{_mem_after:.2f} GB",
                  flush=True)

            # ── Phase-boundary controlled recompile (MORTAR carve / route) ────
            # GATED OFF BY DEFAULT (MORPH_COMPILE_CARVED). MEASURED NET-NEGATIVE at
            # d=768: carved-COMPILED 742ms vs carved-EAGER 698ms (-6.2%) — the carved
            # path's compute is the opaque BCSR custom-op GEMM (not fusable), the
            # surrounding elementwise is cheap, and compiling it thrashes on grad_mode
            # guards (recompile_limit-64 hit). The eager_on_recompile fallback below IS
            # the fast path. The ~+5% Wolfe saw is overhead dilution of the +22%
            # model-compute carving win, NOT lost fusion. Kept for cloud-scale revisit
            # where a larger d_model changes the GEMM-vs-elementwise economics.
            # When ON: open ONE controlled recompile window (default stance, warm every
            # active-set size, re-arm the stance), fork-safe via spawn workers.
            #   Fork-safety vs the step-0 wedge: Inductor codegen still runs in
            # the worker pool PRE-SPAWNED at startup (worker_start_method=
            # subprocess → no new forks from this now-threaded process). The
            # residual risk is the main-process cc launch for new Triton
            # launcher stubs — the SAME class of risk the pre-fix code already
            # took when the carved stk kernels JIT'd on their first post-carve
            # eager forward. Taking it here, in a bounded window we control,
            # beats letting it fire on a random later training step.
            #   RNG: fork_rng so the warmup's randint doesn't shift the
            # training stream's draw sequence.
            if use_compile and os.environ.get("MORPH_COMPILE_CARVED"):
                torch.compiler.set_stance("default")
                try:
                    with torch.random.fork_rng():
                        warmup_compile_all_shapes(
                            model, int(cfg.training.batch_size), seq_len, device,
                            int(getattr(tr, "warmup_passes_per_size", 4)),
                            tag=f"phase-boundary step {step}",
                        )
                finally:
                    torch.compiler.set_stance("eager_on_recompile")
                    print("  torch.compiler stance restored = eager_on_recompile",
                          flush=True)

            t_start = time.perf_counter()
            continue

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        _step_optimizer()

        # ── Prune-divergence diagnostic (env MORPH_DIAG_OPT=<path>) ─────────
        # Post-step, grads still live (zero_grad is top-of-next-iter). Dequants m₂/ν and
        # attributes the worst update dead-vs-live, numerator-vs-denominator. Off by default.
        if _diag_optstate_path and (step % _diag_optstate_every == 0 or step <= 5):
            diag_prune_optstate(model, optimizer, step, _diag_optstate_path)
            diag_optstate_allparams(model, optimizer, step, _diag_optstate_path)
        # Slow-EMA geometry localizer (MORPH_DIAG_M2G=<path>) — decides the de-coherence operator.
        if _diag_m2g_path:
            diag_m2g_geometry(model, optimizer, step, _diag_m2g_path)
            # Authoritative g-vs-numerator capture (drained from the optimizer): dense per-tensor
            # logging in a ±2-step window around prune events to expose cos(g,m₂) before/after.
            _is_prune = bool(prune_stats and prune_stats.get("pruning/prune_step"))
            _near_prune = (step >= _prune_start
                           and ((step - _prune_start) % _prune_interval) in (0, 1, 2,
                                _prune_interval - 1, _prune_interval - 2))
            _m2n = diag_m2g_numerator(optimizer, model, step, _diag_m2g_path,
                                      dense=(_is_prune or _near_prune or step % 200 == 0),
                                      name_filter=_m2g_filter)
            if _m2n:
                wandb.log(_m2n, step=step)
        # Forward-side blow-up localizer (MORPH_DIAG_FWD=1): per-block residual-stream norm +
        # backbone ternary {-1,0,+1} flip count — tracks STE-cusp flip spikes
        # (calm optimizer, exploding loss). Registered lazily; run every step here for per-step norms.
        if _diag_fwd and _diag_optstate_path:
            diag_forward_norms(model, step, _diag_optstate_path)

        # ── Timing ────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        _dt = t_now - t_start
        step_times.append(_dt)
        # t_start is reset at the END of the loop body (after eval/gen/ckpt) so
        # those non-training blocks are excluded from the NEXT step's _dt — keeps
        # steps_per_sec a pure training-throughput metric regardless of eval cadence.
        if len(step_times) > 100:
            step_times = step_times[-100:]
        if _dbg_step:
            _flag = "  <<< SLOW" if _dt > 3.0 else ""
            print(f"  [dbg] step {step}: {_dt:.2f}s depths={_dbg['depths']}{_flag}", flush=True)

        # ── Logging (every 20 steps) ──────────────────────────────────────
        if step % 20 == 0:
            sps = 1.0 / (sum(step_times) / max(len(step_times), 1))
            # Memory: allocated = peak live tensors; reserved = what the caching
            # allocator grabbed from the driver (alloc overhead/fragmentation).
            # The eager-vs-kernel gap in BOTH is the real "alloc overhead" delta.
            peak_alloc = torch.cuda.max_memory_allocated() / 2**20
            peak_resv = torch.cuda.max_memory_reserved() / 2**20
            _lv = loss.item()
            # ── Non-finite self-abort (no-theater: the αcap35 run spewed 600 steps of NaN
            #    after its external watchdog died in a power loss). A NaN/Inf loss NEVER
            #    recovers — save an emergency ckpt for forensics and stop, instead of burning
            #    the GPU. (Finite-but-huge prune bounces are NOT caught here — that's the
            #    external watchdog's job; this only fires on genuine non-finite.) ──
            if not math.isfinite(_lv):
                _ep = os.path.join(ckpt_dir, f"NONFINITE_step_{step}.pt")
                print(f"[ABORT] non-finite loss={_lv} at step {step} — saving {_ep} and stopping",
                      flush=True)
                try:
                    save_checkpoint(_ep, step, model, optimizer, scaler, pruning, next_step=step)
                except Exception as _e:
                    print(f"[ABORT] emergency ckpt failed: {_e}", flush=True)
                _aborted = True
                break
            # Finite-divergence guard: sustained ppl over the ceiling past the warmup descent.
            _ppl_now = math.exp(min(_lv, 20.0))
            if step > 2000 and _ppl_now > _div_ceiling:
                _div_strikes += 1
                print(f"[DIV-GUARD] strike {_div_strikes}/{_div_strikes_max}: ppl={_ppl_now:.1f} "
                      f"> {_div_ceiling:.0f} at step {step}", flush=True)
                if _div_strikes >= _div_strikes_max:
                    _ep = os.path.join(ckpt_dir, f"DIVERGED_step_{step}.pt")
                    print(f"[ABORT] sustained divergence — saving {_ep} and stopping", flush=True)
                    try:
                        save_checkpoint(_ep, step, model, optimizer, scaler, pruning, next_step=step)
                    except Exception as _e:
                        print(f"[ABORT] emergency ckpt failed: {_e}", flush=True)
                    _aborted = True
                    break
            else:
                _div_strikes = 0
            log: dict = {
                "train/loss": _lv,
                "train/ppl": math.exp(min(_lv, 20.0)),
                "train/lr": lr,
                "perf/steps_per_sec": sps,
                # TST superposition ingests s× raw tokens per step (same FLOPs); count them.
                "perf/tokens_per_sec": sps * batch_size * seq_len * (cur_bag if cur_bag > 0 else 1),
                "perf/peak_mem_alloc_mib": peak_alloc,
                "perf/peak_mem_reserved_mib": peak_resv,
                "perf/step": step,
                "train/tst_bag": cur_bag,
            }
            if "stp_loss" in out:
                log["train/stp_loss"] = out["stp_loss"].item()
            if "loop_stp_loss" in out:
                log["train/loop_stp_loss"] = float(out["loop_stp_loss"])
            if "paper_stp_loss" in out:
                log["train/paper_stp_loss"] = float(out["paper_stp_loss"])
            if "latent_forecast_loss" in out:
                log["train/latent_forecast_loss"] = float(out["latent_forecast_loss"])

            # Retention gate diagnostic (#230): sigmoid(ret_gate) per retention block — THE key
            # signal for whether the model actually USES the retention branch (gate opens from ~0)
            # vs treats it as dead weight (stays ~0). A few scalars; log every step.
            _rm = getattr(model, "_orig_mod", model)
            if getattr(_rm.cfg, "retention", False):
                for _nm, _sec in (("prelude", _rm.prelude), ("core", _rm.core), ("coda", _rm.coda)):
                    for _i, _blk in enumerate(_sec):
                        if getattr(_blk, "ret_gate", None) is not None:
                            log[f"retention/gate_{_nm}{_i}"] = torch.sigmoid(_blk.ret_gate).item()

            # Pruning stats
            if prune_stats:
                log.update(prune_stats)

            # Routing diagnostics (every 100 steps, only when routed)
            if step % 100 == 0 and pruning.is_routed:
                rt_stats = collect_routing_stats(model)
                log.update(rt_stats)

            # β1=0 SNR-gate activity (only when the gate is active) — mean gate applied + how many
            # coords were heavily noise-gated (<0.5), reset each log interval. Direct evidence the
            # gate fires (vs inferring from the loss curve).
            _o = getattr(optimizer, "_opt", optimizer)
            if getattr(_o, "g_snr_gate_kappa", 0.0) > 0.0 and getattr(_o, "_gate_n", 0) > 0:
                log["optim/snr_gate_mean"] = _o._gate_sum / _o._gate_n
                log["optim/snr_gate_lt0.5_coords"] = _o._gate_low
                _o._gate_sum, _o._gate_n, _o._gate_low = 0.0, 0, 0

            wandb.log(log, step=step)

            if step % 200 == 0:
                _lstp = f"  loop_stp={float(out['loop_stp_loss']):.4f}" if (
                    "loop_stp_loss" in out and morph_cfg.loop_stp_lambda > 0.0) else ""
                _pstp = f"  paper_stp={float(out['paper_stp_loss']):.4f}" if (
                    "paper_stp_loss" in out and getattr(morph_cfg, "stp_mode", "off") != "off") else ""
                _lf = f"  lf={float(out['latent_forecast_loss']):.4f}" if (
                    "latent_forecast_loss" in out
                    and str(getattr(morph_cfg, "latent_forecast_mode", "off")) != "off") else ""
                _lstp = _lstp + _pstp + _lf
                print(
                    f"[{step:7d}/{total_steps}] loss={loss.item():.4f}  "
                    f"ppl={math.exp(min(loss.item(), 20.0)):.1f}  "
                    f"lr={lr:.2e}  sps={sps:.2f}{_lstp}"
                )

        # ── Validation (every eval_every steps) ──────────────────────────
        if step % eval_every == 0 and step > 0:
            val_loss, val_ppl = evaluate(model, device, val_loader, n_eval_batches)
            val_log: dict = {"val/loss": val_loss, "val/ppl": val_ppl}

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
            _emit_gen(f"step {step}", gen_text)
            model.train()

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % ckpt_every == 0 and step > 0:
            ck_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            save_checkpoint(ck_path, step, model, optimizer, scaler, pruning, next_step=step + 1)
            print(f"  Checkpoint: {ck_path}")

        # ── Reset step timer ───────────────────────────────────────────────
        # Anchor the next step's _dt here, AFTER logging/eval/gen/ckpt, so those
        # non-training blocks don't inflate steps_per_sec (see Timing block above).
        t_start = time.perf_counter()

    # ── Final checkpoint ──────────────────────────────────────────────────
    if _aborted:
        # The run hit the non-finite/divergence guard — a DIVERGED_/NONFINITE_ ckpt already
        # holds the real (failed) state. Do NOT write step_{total_steps}.pt: that would label
        # diverged weights as a completed run (theater that misleads the next resume).
        print(f"[ABORT] run aborted at step {step}; skipping final save+eval "
              f"(forensic ckpt already written). No completed-run checkpoint.", flush=True)
        wandb.finish()
        return
    final_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
    save_checkpoint(final_path, total_steps, model, optimizer, scaler, pruning, next_step=total_steps)
    print(f"Final checkpoint: {final_path}")

    # ── Final eval + generation ───────────────────────────────────────────
    # The eval()/train() + grad-mode toggle is safe under eager_on_recompile (set
    # after warmup): a guard miss runs that region eager instead of recompiling, so
    # there is no recompilation storm. We still skip the final eval when periodic
    # eval is disabled (eval_every > total_steps) — a pure throughput/mem run has no
    # val_loader worth touching and the skip lets it exit promptly.
    if eval_every <= total_steps:
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
        _emit_gen(f"FINAL step {total_steps}", gen_text)

    wandb.finish()


if __name__ == "__main__":
    main()
