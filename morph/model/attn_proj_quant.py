"""Attention-projection quantization-aware training (QAT) for MORPH.

Ablation #205 — the Efull-recovery lever.

Background: the "full" ternary scope (Efull) ternarized EVERY weight matrix
including the CCA attention projections, and lost ~2.85 ppl vs the
backbone-only ternary stack D (44.92 vs 42.07). Hypothesis: the attention
projections (latent down-proj ``W_down_q/k``, value ``W_v``, the compressor /
LightningIndexer linears, the gate, ``W_up``) are more quant-sensitive than the
MLP backbone — ternary's 1.58 bits is too coarse for them. A *gentler*
per-output-row int8/int6/int4 QAT keeps the same memory-compression direction
but with enough resolution to preserve attention selectivity.

Mechanism: ``torch.nn.utils.parametrize`` registers ``IntNLinearSTE`` on each
selected ``nn.Linear.weight``. After registration the effective forward weight is
``round_clip(w_row / s_row, lo, hi) * s_row`` — exactly int{n}-representable per
output row — while the smooth fp32 shadow lives at
``parametrizations.weight.original``. Gradient passes straight through the
quantizer (STE identity backward) to the shadow; AdamW / AdamW8bit updates the
shadow. This is the same per-row STE math as ``embed_quant.IntNRowSTE`` (the
embedding-table version), generalized to int4 and applied to the attention
Linears that ``ternary_qat._attention_linear_ids`` already enumerates.

Design rules (CLAUDE.md):
- NO runtime ``if`` in the forward. Mode is resolved at construction via
  ``parametrize``. ``attn_proj_quant=off`` → no parametrization → bit-identical bf16.
- Per-output-row ABSMAX scale (``max(|w_row|)/hi``) — standard for weight QAT, so a
  single large-magnitude channel can't force the whole tensor's grid coarse. (Ternary
  uses MEAN-based scaling because its threshold semantics differ; int-N uses absmax to
  guarantee no clipping.)
- Disjoint from ternary: this targets attention Linears; the #205 stack runs ternary on
  ``scope=backbone`` (NOT ``backbone_attn``), so the two module sets never overlap. A
  guard SKIPS any Linear already parametrized (e.g. an accidental backbone_attn ternary)
  so we never double-wrap a weight.
- The CSA top-k routing INDICES are never quantized by this — we quantize the indexer's
  *weight* (a projection), and the selection (argmax/topk) is recomputed in full
  precision from the quantized-weight output. The integer indices are never stored
  quantized. (That is the KV-QAT constraint, satisfied here.)

Int{n} representation: symmetric signed, max-absolute per row.
  int8 → [-127, 127] × s_row
  int6 → [-31,  31]  × s_row
  int4 → [-7,   7]   × s_row
(Symmetric ranges avoid asymmetric zero-point complexity — matches the BitNet /
embed_quant per-vector-scale convention.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

from morph.model.ternary_qat import _attention_linear_ids  # reuse — no drift

# Symmetric int{n} ranges (we avoid the negative-extreme for symmetry).
_RANGES: dict[str, tuple[int, int]] = {
    "int8": (-127, 127),
    "int6": (-31, 31),
    "int4": (-7, 7),
}

VALID_MODES = ("off", "int8", "int6", "int4")


class IntNLinearSTE(nn.Module):
    """Per-output-row int-N straight-through estimator for Linear weights.

    For each output row ``w_row`` of the weight ``W`` (shape ``[out, in]``):
        s_row = max(|w_row|) / hi          (detached, per-row scale)
        q_row = round(w_row / s_row).clamp(lo, hi) * s_row

    The effective forward weight is EXACTLY int{n}-representable × s_row. Gradient
    flows straight through to the smooth shadow (identity backward).

    Args:
        bits: quantization bits — 8, 6, or 4.
    """

    def __init__(self, bits: int) -> None:
        super().__init__()
        if bits not in (4, 6, 8):
            raise ValueError(f"IntNLinearSTE: bits must be 4, 6, or 8, got {bits}")
        lo, hi = _RANGES[f"int{bits}"]
        self.bits = bits
        self.lo: int = lo
        self.hi: int = hi

    def forward(self, w: Tensor) -> Tensor:
        """Return the per-row quantized weight. Backward is identity STE."""
        # w: [out, in] (fp32 shadow stored by parametrize). Per-row scale over the
        # input dim → [out, 1] for broadcast. Detached so it is a constant for STE.
        s = w.detach().abs().amax(dim=-1, keepdim=True).div(self.hi).clamp(min=1e-8)
        q_int = (w / s).round().clamp(self.lo, self.hi)   # [out, in] on the int grid
        q = q_int * s                                      # back to float domain
        # STE: forward value = q, backward = identity through w.
        return w + (q - w).detach()

    def right_inverse(self, w: Tensor) -> Tensor:
        """Identity: the shadow lives in the same space as the effective weight."""
        return w


def apply_attn_proj_quant(
    model: nn.Module,
    attn_proj_quant: str = "off",
    ternary_module_names: list[str] | None = None,
) -> dict:
    """Register per-row int-N QAT on every attention-projection ``nn.Linear``.

    Targets (when ``attn_proj_quant != "off"``): every ``nn.Linear`` that lives
    inside a module whose class name contains ``"Attention"`` — identical set to
    the ``"attention"`` category in ``ternary_qat`` (``W_down_q/k``, ``W_v``, the
    compressor + LightningIndexer linears, gate, ``W_up``). This is the SAME
    coverage Efull's ternary attention scope used, so the arms are apples-to-apples
    (Efull ternarized these; #205 int-N's them instead).

    Disjointness: any Linear already parametrized (e.g. a backbone_attn ternary
    arm) is SKIPPED and reported in ``skipped_already_parametrized`` — we never
    double-wrap a weight. In the intended #205 stack ternary runs ``scope=backbone``
    so nothing here is pre-parametrized.

    Call AFTER ``apply_ternary_qat`` (for the disjointness check) and BEFORE
    ``torch.compile`` + optimizer construction (so the STE is in the compiled graph
    and the optimizer binds the smooth ``.original`` shadow).

    Args:
        model: the model to parametrize.
        attn_proj_quant: "off" | "int8" | "int6" | "int4".
        ternary_module_names: the ternary manifest's ``module_names`` (each entry is
            ``"<name> [<category>]"``); used only for a sanity cross-check in the
            manifest. The authoritative disjointness guard is the live
            ``is_parametrized`` check, not this list.

    Returns:
        manifest dict with mode, bits, counts, module names, and any skips.
    """
    if attn_proj_quant not in VALID_MODES:
        raise ValueError(
            f"attn_proj_quant must be one of {VALID_MODES}, got {attn_proj_quant!r}"
        )

    if attn_proj_quant == "off":
        return {
            "attn_proj_quant": "off",
            "bits": None,
            "n_modules_quantized": 0,
            "n_params_quantized": 0,
            "module_names": [],
            "skipped_already_parametrized": [],
        }

    bits = int(attn_proj_quant[3:])  # "int8" → 8, "int6" → 6, "int4" → 4
    attn_ids = _attention_linear_ids(model)

    registered: list[str] = []
    skipped: list[str] = []
    n_params = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if id(module) not in attn_ids:
            continue
        w = getattr(module, "weight", None)
        if not isinstance(w, nn.Parameter) or w.dim() < 2:
            continue
        if parametrize.is_parametrized(module, "weight"):
            # Already wrapped (e.g. ternary) — do NOT double-wrap. Disjointness guard.
            skipped.append(name)
            continue
        # Each module gets its own STE instance (stateless, but distinct objects avoid
        # any accidental aliasing).
        parametrize.register_parametrization(module, "weight", IntNLinearSTE(bits=bits))
        registered.append(name)
        n_params += module.weight.numel()

    return {
        "attn_proj_quant": attn_proj_quant,
        "bits": bits,
        "n_modules_quantized": len(registered),
        "n_params_quantized": int(n_params),
        "module_names": registered,
        "skipped_already_parametrized": skipped,
    }
