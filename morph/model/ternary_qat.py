"""Ternary quantization-aware training (QAT) via straight-through estimator.

This is REAL QAT, not the export-only path: the STE-quantized weight is used in
the *forward* pass, so the smooth bf16 shadow weight (the live parameter) learns
to be robust to the {-1, 0, +1}×scale snap. Gradients pass straight through the
quantizer (identity backward) to the shadow weight; the optimizer (plain AdamW or
bitsandbytes AdamW8bit) updates the shadow as usual.

Mechanism: ``torch.nn.utils.parametrize`` registers ``TernarySTE`` on a module's
``weight``. After registration, ``module.weight`` transparently returns the
quantized tensor inside the forward, while the smooth parameter lives at
``module.parametrizations.weight.original``. This composes with ``state_dict``,
works for both ``nn.Linear`` and ``nn.Embedding``, and traces cleanly under
``torch.compile`` (sign/abs/mean/compare/detach are all capturable).

Because the forward already uses the ternary weight, the training/val perplexity
IS the deployed-ternary perplexity (modulo final int8 packing) — there is no
train-smooth / deploy-ternary gap to measure separately.

Scope ablation (per the research question — does ternarizing attention +
embeddings hurt more than just the backbone?):
  - "backbone"      : all weight matrices that are NOT attention and NOT embeddings
                      (MLP gate_up/down for both _SwiGLU and _SwiGLUBlockELL cores,
                      the LM-head mixer, channel-inject projections).
  - "backbone_attn" : backbone + attention projections (nn.Linear inside MORPHAttention).
  - "full"          : backbone + attention + token/space/euclidean/bigram embeddings.

Embeddings are the riskiest target: the 8-bit optimizer already force-keeps
embedding *state* in fp32 because their gradients are sparse/large-range, so
snapping embedding *weights* to ternary is expected to be the dominant quality
sink. The scope knob exists precisely to quantify that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

# Which weight categories each scope quantizes.
SCOPES: dict[str, set[str]] = {
    "backbone": {"backbone"},
    "backbone_attn": {"backbone", "attention"},
    "full": {"backbone", "attention", "embeddings"},
}


class TernarySTE(nn.Module):
    """Per-tensor ternary straight-through estimator parametrization.

    forward(w) = scale * sign(w/scale) * 1[|w/scale| > threshold]
    where scale = mean(|w|) is treated as a constant (detached) so the backward
    is a pure identity straight-through to the shadow weight w.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, w: Tensor) -> Tensor:
        # Detached per-tensor scale → constant for the STE (pure straight-through).
        scale = w.detach().abs().mean().clamp(min=1e-8)
        w_norm = w / scale
        q = torch.sign(w_norm) * (w_norm.abs() > self.threshold).to(w.dtype)
        # forward value == scale*q ; gradient flows to w as identity (rest detached).
        return w + (scale * q - w).detach()

    def right_inverse(self, w: Tensor) -> Tensor:
        # Map an assigned weight back to the underlying smooth parameter (identity).
        return w


def _attention_linear_ids(model: nn.Module) -> set[int]:
    """ids of every nn.Linear that lives inside an attention module."""
    ids: set[int] = set()
    for m in model.modules():
        if "Attention" in type(m).__name__:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    ids.add(id(sub))
    return ids


def _categorize(name: str, module: nn.Module, attn_ids: set[int]) -> str | None:
    """Return the ternary category for a module, or None if it carries no
    quantizable weight matrix (norms, scalars, biases, container modules)."""
    if isinstance(module, nn.Embedding):
        return "embeddings"
    # CMSBlockLinear (inside BlockELLLinear) holds a dense [out, in] nn.Parameter
    # named `weight` in dense mode — quantize it like any other linear.
    is_cms = type(module).__name__ == "CMSBlockLinear"
    if isinstance(module, nn.Linear) or is_cms:
        w = getattr(module, "weight", None)
        if not isinstance(w, nn.Parameter) or w.dim() < 2:
            return None  # e.g. CMSBlockLinear in sparse mode (no dense .weight)
        return "attention" if id(module) in attn_ids else "backbone"
    return None


def apply_ternary_qat(
    model: nn.Module,
    scope: str = "backbone",
    threshold: float = 0.5,
) -> dict:
    """Register the ternary STE parametrization on every weight matrix selected
    by ``scope``. Call AFTER model construction and BEFORE torch.compile and
    optimizer construction.

    Returns a manifest: {scope, threshold, counts per category, n_params_ternary,
    n_params_total, module_names}.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown ternary_scope={scope!r}; choices={list(SCOPES)}")
    wanted = SCOPES[scope]
    attn_ids = _attention_linear_ids(model)

    counts: dict[str, int] = {"backbone": 0, "attention": 0, "embeddings": 0}
    names: list[str] = []
    n_tern = 0
    for name, module in model.named_modules():
        cat = _categorize(name, module, attn_ids)
        if cat is None or cat not in wanted:
            continue
        if parametrize.is_parametrized(module, "weight"):
            continue  # idempotent — don't double-register
        parametrize.register_parametrization(module, "weight", TernarySTE(threshold))
        counts[cat] += 1
        names.append(f"{name} [{cat}]")
        n_tern += module.weight.numel()

    n_total = sum(p.numel() for p in model.parameters())
    return {
        "scope": scope,
        "threshold": threshold,
        "counts": counts,
        "n_modules_ternary": sum(counts.values()),
        "n_params_ternary": int(n_tern),
        "n_params_total": int(n_total),
        "frac_params_ternary": float(n_tern) / max(1, n_total),
        "module_names": names,
    }
