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
import torch._inductor.config as _inductor_config  # noqa: E402
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
        use_cla=bool(getattr(tr, "use_cla", False)),
        cla_share_start=int(getattr(tr, "cla_share_start", -1)),
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
        ce_chunk_size=int(getattr(m, "ce_chunk_size", 1024)),
        use_kernels=bool(getattr(m, "use_kernels", True)),
        use_mrr=bool(getattr(m, "use_mrr", True)),
        residual_mode=(str(m.residual_mode)
                       if getattr(m, "residual_mode", None) not in (None, "null") else None),
        hc_streams=int(getattr(m, "hc_streams", 4)),
        hc_tau=float(getattr(m, "hc_tau", 1.0)),
        hc_cayley_iters=int(getattr(m, "hc_cayley_iters", 3)),
        hc_cayley_alpha=float(getattr(m, "hc_cayley_alpha", 0.1)),
        hc_sinkhorn_iters=int(getattr(m, "hc_sinkhorn_iters", 20)),
        hc_init_gain=float(getattr(m, "hc_init_gain", 0.1)),
        hc_use_kernel=bool(getattr(m, "hc_use_kernel", True)),
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

    # ── Build model ───────────────────────────────────────────────────────
    # ORDERING IS LOAD-BEARING: model build + torch.compile + warmup run BEFORE
    # wandb.init() and the streaming dataloader (both below). All Triton/Inductor
    # compilation — and the gcc subprocess fork that builds each kernel launcher stub —
    # therefore happens in a SINGLE-THREADED process. A fork only deadlocks when another
    # thread holds a non-reentrant lock (glibc malloc arena) at fork time; with no
    # wandb/httpx threads alive yet, every compile fork is safe by construction. This is
    # the root-cause fix for the intermittent step-0 wedge (see Ai-notes 06-01-2026/
    # MORPH-eval-recompile-hang): the fused CCA Triton kernels JIT-specialize size==1
    # separately, so the first runtime n_active==1 (a rare Poisson draw) used to compile
    # + fork against live threads. wandb.init() is deferred to just after the warmup.
    morph_cfg = build_morph_config(cfg)
    model = MORPHTransformer(morph_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")

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
        _wb = int(cfg.training.batch_size)
        _mx = int(model.cfg.max_depth)

        def _forced_depths(K):
            d = [1] * _wb
            for _j in range(min(K, _wb)):
                d[_j] = _mx
            return torch.tensor(d, device=device, dtype=torch.long)

        _orig_sample = model._sample_depths
        _passes_per_size = int(getattr(tr, "warmup_passes_per_size", 4))
        _sizes = list(range(_wb, 0, -1))   # [B, B-1, ..., 1] — covers size>1 AND size==1
        print(f"  Warmup compile (thread-free; active-set sizes {_sizes} × {_passes_per_size})...",
              flush=True)
        _t_warm = time.perf_counter()
        for _K in _sizes:
            model._sample_depths = (lambda _b, _dev, __K=_K: _forced_depths(__K))
            for _wi in range(_passes_per_size):
                _ids = torch.randint(0, model.cfg.vocab_size, (_wb, seq_len), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _w = model(_ids, labels=_ids)
                _w["loss"].backward()
                model.zero_grad(set_to_none=True)
                del _ids, _w
        model._sample_depths = _orig_sample          # restore real Poisson sampling
        torch.cuda.synchronize()
        print(f"  Warmup compile done in {time.perf_counter()-_t_warm:.1f}s "
              f"({len(_sizes) * _passes_per_size} passes, all active-set sizes)", flush=True)

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
    wandb.init(
        project=wb_cfg.project,
        entity=getattr(wb_cfg, "entity", None),
        name=getattr(wb_cfg, "name", None),
        config=full_config_dict,
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

    # Generation samples go to a sidecar file, NOT stdout. Generated text is
    # uncontrolled token output that can contain substrings ("RuntimeError:",
    # "Killed", "Traceback...") which would false-trigger a log-scraping watcher
    # (ignore/ab_watch.sh) into reporting a crash. Stdout gets only a safe summary.
    gen_samples_path = os.path.join(ckpt_dir, "generation_samples.txt")

    def _emit_gen(label: str, gen_text: str) -> None:
        with open(gen_samples_path, "a") as _f:
            _f.write(f"\n===== {label} =====\n{gen_text}\n")
        print(f"  [GEN {label}] {len(gen_text)} chars → {gen_samples_path}", flush=True)

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
            log: dict = {
                "train/loss": loss.item(),
                "train/ppl": math.exp(min(loss.item(), 20.0)),
                "train/lr": lr,
                "perf/steps_per_sec": sps,
                "perf/tokens_per_sec": sps * batch_size * seq_len,
                "perf/peak_mem_alloc_mib": peak_alloc,
                "perf/peak_mem_reserved_mib": peak_resv,
                "perf/step": step,
            }
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
            save_checkpoint(ck_path, step, model, optimizer, scaler, pruning)
            print(f"  Checkpoint: {ck_path}")

        # ── Reset step timer ───────────────────────────────────────────────
        # Anchor the next step's _dt here, AFTER logging/eval/gen/ckpt, so those
        # non-training blocks don't inflate steps_per_sec (see Timing block above).
        t_start = time.perf_counter()

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
    save_checkpoint(final_path, total_steps, model, optimizer, scaler, pruning)
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
