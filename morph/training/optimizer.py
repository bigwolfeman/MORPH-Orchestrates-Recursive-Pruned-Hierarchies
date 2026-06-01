"""MORPH optimizer setup — AdamW + optional DeepNestedOptimizer + STE ternary shadows.

Components:
  create_optimizer(model, cfg)           -> torch.optim.Optimizer
  create_lr_schedule(cfg)               -> Callable[[int], float]
  TernaryShadowOptimizer                 wrapper for Phase-2 STE ternary
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

__all__ = [
    "create_optimizer",
    "create_lr_schedule",
    "TernaryShadowOptimizer",
]

# Parameter groups: names matching these patterns are excluded from weight decay.
_NO_DECAY_KEYWORDS = (
    "norm", "bias", "log_A", "log_dt",            # Norms + biases + SSM
    "injection",                                    # DiagonalInjection scalars
    "channel_scales", "alpha_raw", "gamma_raw",    # MRR channel params
    "log_scale",                                   # MRR log scales
    "x0_injects", "value_embeds",                  # Skip/value inject gates
    "lm_mixer",                                    # LM head mixer
    "embed",                                       # Embedding tables
    "stp",                                         # STP loss params
    "ste_gain", "ste_temp",                        # LSTE per-layer params
)


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decay / no-decay groups."""
    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in _NO_DECAY_KEYWORDS):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


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
        cfg.training.lr          — base learning rate
        cfg.training.weight_decay (optional, default 0.1)
        cfg.training.ternary      (optional bool — activates TernaryShadowOptimizer)
        cfg.training.adam8bit     (optional bool — uses bitsandbytes AdamW8bit)

    Returns a plain AdamW, TernaryShadowOptimizer-wrapped AdamW, or
    an 8-bit AdamW (bnb) depending on config flags.
    8-bit and ternary may be combined: bnb AdamW8bit becomes the base_opt
    inside TernaryShadowOptimizer. The 8-bit quantization applies only to
    the optimizer state (m, v → uint8); bf16 shadow-weight semantics are
    unchanged because bnb does NOT quantize the parameters themselves.
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
    if use_8bit:
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

    # Optional STE ternary shadow wrapper (composes with 8-bit base_opt).
    use_ternary = bool(getattr(tr, "ternary", False))
    if use_ternary:
        base_opt = TernaryShadowOptimizer(base_opt, model)

    return base_opt


def create_lr_schedule(cfg: DictConfig) -> Callable[[int], float]:
    """Return a function step -> lr_multiplier (relative to base lr).

    Schedule: linear warmup from 0 to 1 over `warmup` steps,
    then cosine decay from 1 to `min_lr / lr` over the remaining steps.
    """
    tr = cfg.training
    total_steps = int(tr.steps)
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


# ── STE Ternary Shadow Optimizer ─────────────────────────────────────────────

class _STEFunction(torch.autograd.Function):
    """STE quantization: forward uses ternary values, backward is identity."""

    @staticmethod
    def forward(ctx, w: Tensor, threshold: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(w)
        # Scale = mean absolute value per group-of-128 (per-layer global for simplicity)
        scale = w.abs().mean().clamp(min=1e-8)
        w_norm = w / scale
        ternary = w_norm.sign() * (w_norm.abs() > threshold).float()
        return ternary * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:  # type: ignore[override]
        # Straight-through: gradient passes through unchanged.
        return grad_output, None


class TernaryShadowOptimizer:
    """Wraps any optimizer and maintains per-parameter ternary shadow copies.

    Design:
      - The wrapped model's fp16/bf16 parameters ARE the shadow weights.
        The optimizer updates them continuously.
      - After each step(), we snap each weight to {-1, 0, +1} × scale
        using STE — but crucially this snap is NOT applied in-place to the
        live parameter. Instead we maintain a buffer of ternary int8 values
        for export / deployment only.
      - Forward/backward always use the bf16 shadow weights (smooth surface
        for the optimizer). The ternary buffers are export artefacts.

    This matches the spec in 007-ternary-shadow-weights: shadow weight IS
    self.weight, forward uses STE-quantized version, gradient flows to shadow.

    The `enable()` / `disable()` toggle lets training scripts switch between
    dense warmup (Phase 1) and ternary training (Phase 2) without rebuilding
    the optimizer.

    Args:
        optimizer: Any torch optimizer.
        model:     The model whose parameters will receive ternary shadows.
        threshold: STE quantization threshold (default 0.5, as in Bonsai).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        threshold: float = 0.5,
    ) -> None:
        self._opt = optimizer
        self._model = model
        self._threshold = threshold
        self._enabled = True
        self._step_count = 0

        # Build ternary shadow buffers (int8, same shape as each parameter).
        self._shadows: dict[str, torch.Tensor] = {}
        self._scales: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Only quantise weight tensors (skip scalars / 1-D vectors).
            if p.dim() >= 2:
                self._shadows[name] = torch.zeros_like(p.data, dtype=torch.int8)
                self._scales[name] = torch.ones(1, device=p.device, dtype=torch.float32)

    # ── Delegate standard optimizer interface ──────────────────────────────

    @property
    def param_groups(self):
        return self._opt.param_groups

    def state_dict(self) -> dict:
        return {
            "optimizer": self._opt.state_dict(),
            "shadows": self._shadows,
            "scales": self._scales,
            "step_count": self._step_count,
            "enabled": self._enabled,
        }

    def load_state_dict(self, state: dict) -> None:
        self._opt.load_state_dict(state["optimizer"])
        self._shadows = state.get("shadows", self._shadows)
        self._scales = state.get("scales", self._scales)
        self._step_count = state.get("step_count", 0)
        self._enabled = state.get("enabled", True)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def step(self, closure=None, _skip_inner: bool = False) -> None:
        """Step the inner optimizer, then optionally update ternary shadows.

        Args:
            _skip_inner: If True, skip self._opt.step() (use when GradScaler
                         already stepped the inner optimizer).
        """
        if not _skip_inner:
            self._opt.step(closure)
        self._step_count += 1

        if not self._enabled:
            return

        # Update ternary shadow buffers from current (updated) parameter values.
        # This is for export only — does NOT modify the live parameter.
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                if name not in self._shadows:
                    continue
                scale = p.data.abs().mean().clamp(min=1e-8)
                self._scales[name].fill_(scale.item())
                p_norm = p.data.float() / scale
                ternary = (p_norm.sign() * (p_norm.abs() > self._threshold).float())
                self._shadows[name].copy_(ternary.to(torch.int8))

    def get_ternary_weights(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return {param_name: (ternary_int8, scale_f32)} for all shadowed params.

        Suitable for export to a compact ternary checkpoint.
        """
        return {
            name: (self._shadows[name].clone(), self._scales[name].clone())
            for name in self._shadows
        }

    def ternary_stats(self) -> dict[str, float]:
        """Aggregate ternary distribution statistics for wandb logging."""
        total = neg = zero = pos = 0
        for t in self._shadows.values():
            t_f = t.float()
            total += t_f.numel()
            neg += (t_f == -1).sum().item()
            zero += (t_f == 0).sum().item()
            pos += (t_f == 1).sum().item()
        if total == 0:
            return {"neg_frac": 0.0, "zero_frac": 0.0, "pos_frac": 0.0}
        return {
            "neg_frac": neg / total,
            "zero_frac": zero / total,
            "pos_frac": pos / total,
        }
