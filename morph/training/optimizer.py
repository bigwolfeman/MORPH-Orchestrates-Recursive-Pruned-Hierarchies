"""MORPH optimizer setup.

create_optimizer builds the deploy optimizer (AdEMAMixB1Zero 8-bit, default) or the
AdamW8bit / AdamW fallbacks; create_lr_schedule returns the step -> lr-multiplier fn.

  create_optimizer(model, cfg)  -> torch.optim.Optimizer
  create_lr_schedule(cfg)       -> Callable[[int], float]
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from omegaconf import DictConfig

__all__ = [
    "create_optimizer",
    "create_lr_schedule",
]

# Parameter groups: names matching these patterns are excluded from weight decay.
_NO_DECAY_KEYWORDS = (
    "norm", "bias", "log_A", "log_dt",            # Norms + biases + SSM
    "injection",                                    # DiagonalInjection scalars
    "channel_scales", "alpha_raw", "gamma_raw",    # HC/legacy residual scalar params
    "log_scale",                                   # ChannelInject scales
    "x0_injects", "value_embeds",                  # Skip/value inject gates
    "lm_mixer",                                    # LM head mixer
    "tul_gate",                                    # TUL span-length gate: a norm, a rank-1
                                                   # readout direction and a small budget
                                                   # table, all zero-init. Decay on a
                                                   # zero-init readout direction pulls
                                                   # against the only gradient it has —
                                                   # the "provably cannot move" class that
                                                   # docs/tul-gate-spec.md §10 exists for.
    "embed",                                       # Embedding tables
    "ste_gain", "ste_temp",                        # LSTE per-layer params
    # ── TUL (docs/tul-spec.md §3.1-§3.4) ──
    # E_slot / E_mask are EMBEDDINGS (E_slot is initialised to the mean of the embedding
    # table, spec §5) and every other embedding here is no-decay; they were landing in the
    # decay group only because their names do not contain "embed".
    # W_prefix is the ONLY channel carrying h_i into the coda, and the plan position
    # (k < prefix_k-1) has NO label — its sole gradient is the coda's attention to it.
    # Weight decay pulls that channel toward zero, i.e. toward val/plan_nats = 0, which is
    # the quantity the arm exists to measure; a null result would then be unattributable
    # (decay vs the model ignoring the plan). Excluded deliberately — reviewer, 2026-08-16.
    "E_slot", "E_mask", "W_prefix",
)


def _split_by_decay(model: nn.Module) -> tuple[list, list, list[str], list[str]]:
    """`(decay_params, no_decay_params, decay_names, no_decay_names)`, in group order.

    ONE walk of `named_parameters()` produces both the tensors and their names, so the
    optimizer's group layout and any name-based view of it cannot drift apart. They did
    drift once: `param_group_names` below is what re-aligns optimizer state on a resume,
    and a second independent walk would have made it silently wrong the first time the
    split rule changed.
    """
    decay, no_decay, decay_n, no_decay_n = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in _NO_DECAY_KEYWORDS):
            no_decay.append(p)
            no_decay_n.append(name)
        else:
            decay.append(p)
            decay_n.append(name)
    return decay, no_decay, decay_n, no_decay_n


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decay / no-decay groups."""
    decay_params, no_decay_params, _, _ = _split_by_decay(model)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def param_group_names(model: nn.Module) -> list[list[str]]:
    """Parameter NAMES per optimizer group, in the exact order `_param_groups` uses.

    `torch.optim.Optimizer.load_state_dict` matches saved state to live parameters by
    POSITION within the flattened group order. That is fine for a plain resume and wrong
    the moment a resume adds parameters: an intervention arm that introduces a module
    (SCSE's two projections, for example) inserts them wherever the module sits in
    `named_parameters()` — for SCSE that is the MIDDLE of the decay group, not the end —
    which shifts every later index by one and silently pairs each parameter with another
    parameter's moments. Torch does not detect that; it only raises when the group SIZES
    differ, which is the lucky case.

    :func:`align_optimizer_state` uses these names to re-index instead.
    """
    _, _, decay_n, no_decay_n = _split_by_decay(model)
    return [decay_n, no_decay_n]


def align_optimizer_state(state: dict, model: nn.Module,
                          ckpt_param_names: set[str]) -> tuple[dict, list[str]]:
    """Re-index a checkpoint's optimizer state onto `model`'s CURRENT parameters.

    Returns `(aligned_state, added_names)`. Parameters the checkpoint does not contain get
    NO entry in `state`, so the optimizer creates their moments lazily on the first step —
    which is what "the new module starts cold" should mean.

    Args:
        state:            the checkpoint's `optimizer` state dict.
        model:            the live model, which may have parameters the checkpoint lacks.
        ckpt_param_names: names present in the checkpoint's MODEL state dict. Anything the
                          live model has and this set does not is treated as added.

    Only ADDITIONS are supported. If the checkpoint holds state for more parameters in a
    group than the live model has after removing the added ones, the alignment is not a
    pure insertion and this raises rather than guessing — a wrong alignment pairs a
    parameter with another parameter's moments and shows up as a plausible-looking run.
    """
    live_names = param_group_names(model)
    added = [n for g in live_names for n in g if n not in ckpt_param_names]
    added_set = set(added)

    # Validate BEFORE the no-additions shortcut. A checkpoint holding MORE parameters than
    # the live model has is a REMOVAL, which reports zero additions and would otherwise
    # take the shortcut and hand torch a mismatched dict. Caught by
    # tests/test_optimizer_align.py::test_a_REMOVED_parameter_raises_instead_of_guessing.
    old_groups = state.get("param_groups", [])
    if len(old_groups) != len(live_names):
        raise ValueError(
            f"checkpoint has {len(old_groups)} optimizer groups, model needs "
            f"{len(live_names)}; alignment handles ADDED PARAMETERS, not a changed "
            f"group layout")

    for gi, names in enumerate(live_names):
        kept = len([n for n in names if n not in added_set])
        want = len(old_groups[gi]["params"])
        if kept != want:
            raise ValueError(
                f"optimizer group {gi}: checkpoint holds {want} parameters but the live "
                f"model has {kept} once the {len(added)} added ones are removed. "
                f"This is not a pure insertion; refusing to guess an alignment.")
    if not added:
        return state, []

    # Flattened old index -> flattened new index. The old order is the live order with the
    # added names removed, because additions do not reorder what was already there.
    old_to_new: dict[int, int] = {}
    new_flat = 0
    old_flat = 0
    for gi, names in enumerate(live_names):
        for n in names:
            if n not in added_set:
                old_to_new[old_flat] = new_flat
                old_flat += 1
            new_flat += 1

    aligned = dict(state)
    aligned["state"] = {old_to_new[k]: v for k, v in state["state"].items()
                        if k in old_to_new}
    groups = []
    flat = 0
    for gi, names in enumerate(live_names):
        g = dict(old_groups[gi])
        g["params"] = list(range(flat, flat + len(names)))
        flat += len(names)
        groups.append(g)
    aligned["param_groups"] = groups
    return aligned, added


def _register_embedding_32bit_overrides(model: nn.Module, opt) -> None:
    """Override 8-bit optimizer state to 32-bit for all nn.Embedding parameters.

    bnb's 8-bit Adam is documented to be unstable on embedding layer state
    (sparse, large-range gradients). This function registers all nn.Embedding
    weight parameters to use 32-bit state via GlobalOptimManager.

    Must be called AFTER the optimizer is constructed (pid→config binding
    happens at first optimizer step, so override_config just needs to be
    registered before the first step).

    Args:
        model: The model whose nn.Embedding modules to override.
        opt:   A constructed bnb.optim.AdamW8bit (or any bnb 8-bit optimizer).
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return

    mng = bnb.optim.GlobalOptimManager.get_instance()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            mng.register_module_override(module, "weight", {"optim_bits": 32})


def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Build the optimizer from config.

    Reads:
        cfg.training.lr           — base learning rate
        cfg.training.weight_decay (optional, default 0.1)
        cfg.training.adam8bit     (optional bool — bitsandbytes 8-bit optimizer state)
        cfg.training.optimizer    ("adamw" (default) | "ademamix_b1zero")
        cfg.training.beta3        (AdEMAMix slow-EMA decay)
        cfg.training.ademamix_alpha / ademamix_t_alpha / ademamix_t_beta3
                                  (AdEMAMix mix weight + α/β3 warmup horizons; the warmup
                                   horizons default to total steps — essential for stability,
                                   NOT the same as LR warmup)

    Returns one of (deploy default first):
      - AdEMAMixB1Zero (optimizer=ademamix_b1zero) — β1=0 AdEMAMix, blockwise-8bit state
      - bnb AdamW8bit  (optimizer=adamw + adam8bit=true) — fallback
      - torch AdamW    (optimizer=adamw, bitsandbytes absent) — last-resort fallback
    The 8-bit quantization applies only to optimizer state (m,v → uint8); bnb does NOT
    quantize the parameters themselves (bf16 weights unchanged).
    """
    tr = cfg.training
    lr = float(tr.lr)
    wd = float(getattr(tr, "weight_decay", 0.1))
    betas = (
        float(getattr(tr, "beta1", 0.9)),
        float(getattr(tr, "beta2", 0.95)),
    )

    groups = _param_groups(model, wd)

    use_8bit = bool(getattr(tr, "adam8bit", False))
    opt_name = str(getattr(tr, "optimizer", "adamw")).lower()

    if opt_name == "ademamix":
        raise ValueError(
            "optimizer=ademamix (stock bnb 3-buffer AdEMAMix) was removed — it cost +50% "
            "optimizer memory vs AdamW8bit for no deploy benefit. Use "
            "optimizer=ademamix_b1zero (β1=0 fork: 2 buffers, AdamW8bit memory parity).")
    elif opt_name == "ademamix_b1zero":
        # β1=0 AdEMAMix with blockwise-8bit state — 2 buffers (m2, ν) matching AdamW8bit
        # memory (~2 B/param); bnb's β1=0 "trick" still allocates 3 buffers. Uses bnb's
        # own blockwise dynamic quant.
        from morph.training.ademamix_b1zero import AdEMAMixB1Zero
        beta3 = float(getattr(tr, "beta3", 0.9999))
        # b1zero uses its OWN beta2 (AdEMAMix paper recommends 0.999) decoupled from the shared
        # AdamW beta2, so the deploy default is the validated coordcap05 winner without moving the
        # AdamW baseline. Override per-arm with training.ademamix_beta2.
        b1zero_beta2 = float(getattr(tr, "ademamix_beta2", 0.999))
        alpha = float(getattr(tr, "ademamix_alpha", 8.0))
        _ta = getattr(tr, "ademamix_t_alpha", None)
        _tb = getattr(tr, "ademamix_t_beta3", None)
        t_alpha = int(_ta) if _ta is not None else int(tr.steps)
        t_beta3 = int(_tb) if _tb is not None else int(tr.steps)
        b3_start = float(getattr(tr, "ademamix_beta3_warmup_start", 0.9))
        bits = 8 if use_8bit else 32
        # fused=True (DEFAULT) → custom Triton kernel. With fused_dynamic_qmap=True (also default)
        # it uses bnb's dynamic blockwise qmap — the same quantizer as the de-fused path, so no
        # quality tax at the fastest opt.step. fused_dynamic_qmap=False falls back to linear-int8
        # state (lossy on small-ν lanes). fused=False → de-fused path (bnb dynamic qmap, preserves
        # small ν). See base.yaml for full flag documentation.
        fused = bool(getattr(tr, "ademamix_fused", True))
        # eps_inside: True = √(ν/bc2+ε) denom floor; False = √(ν/bc2)+ε true-Adam normalization
        # (DEFAULT, correct for the dynamic-qmap/de-fused ν). Honored by BOTH the de-fused step and
        # the fused kernel (the linear-int8 fused path needs the floor; dynamic-qmap does not).
        eps_inside = bool(getattr(tr, "ademamix_eps_inside", False))
        # per-coordinate update clamp (Adam units) — bounds the (g+α·m₂)/denom step so a prune
        # topology-shock can't detonate a few coords in one step. 0 = off. The fast+stable lever.
        update_clip = float(getattr(tr, "ademamix_update_clip", 0.0))
        # Per-coordinate stale-push cap: |α·m₂_i| ≤ c·|g_i| each coord — catches per-coord
        # magnitude domination that the per-tensor norm hides. The deploy-gauntlet-winning cure.
        # 0=off. (Supported on BOTH the fused kernel and the de-fused path.)
        stale_push_cap_coord = float(getattr(tr, "ademamix_stale_push_cap_coord", 0.0))
        # β1=0 noise-gating fix (stateless, no m1 buffer; see ademamix_b1zero.py for details):
        #   g_coef (γ<1) = constant downscale of the raw-g numerator term (uniform control);
        #   g_snr_gate_kappa (κ>0) = soft per-coord SNR gate: snr=|m₂|/denom → gate∈[floor,1].
        g_coef = float(getattr(tr, "ademamix_g_coef", 1.0))
        g_snr_gate_kappa = float(getattr(tr, "ademamix_g_snr_gate_kappa", 0.0))
        g_snr_gate_floor = float(getattr(tr, "ademamix_g_snr_gate_floor", 0.1))
        # track_diag: per-tensor optimizer telemetry (snr-gate, clip/cap counts). OFF by
        # default — sync loops cost ~67ms/step on large models; param update is bit-identical.
        track_diag = bool(getattr(tr, "ademamix_track_diag", False))
        # fused_dynamic_qmap: replace the fused kernel's linear-int8 state quant with bnb's
        # own non-linear dynamic map (the de-fused reference quantizer), eliminating the
        # systematic bias of the linear path. nu_floor only affects the linear fused path.
        # Both no-op when ademamix_fused=false.
        fused_dynamic_qmap = bool(getattr(tr, "ademamix_fused_dynamic_qmap", True))
        fused_nu_floor = bool(getattr(tr, "ademamix_fused_nu_floor", True))
        # Keep the no-decay group (which holds nn.Embedding tables) in 32-bit state —
        # bnb's 8-bit is unstable on sparse/large-range embedding grads. The no-decay group
        # otherwise holds only sub-4096 tensors (already fp32), so this mirrors AdamW8bit.
        groups[1]["optim_bits"] = 32
        base_opt = AdEMAMixB1Zero(
            groups, lr=lr, betas=(0.0, b1zero_beta2, beta3), alpha=alpha,
            alpha_cap=float(getattr(tr, "ademamix_alpha_cap", 0.0)),
            t_alpha=t_alpha, t_beta3=t_beta3, beta3_warmup_start=b3_start,
            eps=1e-8, weight_decay=wd, bits=bits, fused=fused, eps_inside=eps_inside,
            update_clip=update_clip, stale_push_cap_coord=stale_push_cap_coord,
            g_coef=g_coef, g_snr_gate_kappa=g_snr_gate_kappa,
            g_snr_gate_floor=g_snr_gate_floor, track_diag=track_diag,
            fused_dynamic_qmap=fused_dynamic_qmap, fused_nu_floor=fused_nu_floor)
    elif use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "cfg.training.adam8bit=true requires bitsandbytes. "
                "Install with: uv pip install bitsandbytes"
            ) from e
        base_opt = bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=1e-8,
                                        weight_decay=wd)
        # Override optimizer state to 32-bit for all embedding tables —
        # bnb 8-bit is unstable on sparse/large-range embedding gradients.
        _register_embedding_32bit_overrides(model, base_opt)
    else:
        base_opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)

    # cfg.training.ternary means FORWARD-STE QAT, applied at model-build time via
    # morph.model.ternary_qat.apply_ternary_qat — the smooth weight is the live parameter,
    # the STE lives in the forward, and the optimizer (AdamW / AdamW8bit / AdEMAMixB1Zero)
    # updates it directly. The optimizer needs no ternary awareness.

    return base_opt


def create_lr_schedule(cfg: DictConfig) -> Callable[[int], float]:
    """Return a function step -> lr_multiplier (relative to base lr).

    Schedule: linear warmup from 0 to 1 over `warmup` steps,
    then cosine decay from 1 to `min_lr / lr` over the remaining steps.

    `lr_decay_steps` decouples the cosine DECAY HORIZON from the run length: set it to
    a longer horizon (e.g. tokens_target / tokens_per_step) to keep LR near peak across
    a short continue-pretrain (the schedule only traverses run_steps/lr_decay_steps of the
    cosine). Unset/0 → decay over `steps` (original behavior). Only affects LR; TST and the
    loop still key off `steps`.
    """
    tr = cfg.training
    total_steps = int(getattr(tr, "lr_decay_steps", 0)) or int(tr.steps)
    warmup_steps = int(getattr(tr, "warmup", 1500))
    lr_max = float(tr.lr)
    lr_min = float(getattr(tr, "min_lr", lr_max * 0.1))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return lr_max * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine

    return schedule
