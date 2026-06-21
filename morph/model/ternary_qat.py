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
                      (MLP gate_up/down for both _SwiGLU and _SwiGLUMortar cores,
                      the LM-head mixer, channel-inject projections).
  - "backbone_attn" : backbone + attention projections (nn.Linear inside MORPHAttention).
  - "embeddings"    : token/euclidean/bigram embedding tables ONLY (no backbone, no
                      attention). The Lorentz space embedding is ALWAYS excluded.
                      Use this for the isolation arm: ternary embeddings alone, to
                      cleanly separate embedding cost from backbone cost.
  - "full"          : backbone + attention + token/euclidean/bigram embeddings.
                      The Lorentz space embedding (``lor_embed.space_embed``) is
                      ALWAYS excluded — it is geometry-critical (trained geodesic
                      radius ~0.97) and must stay bf16. This scope is the "full
                      ternary done right" arm: everything except Lorentz.

Embeddings are the riskiest target: the 8-bit optimizer already force-keeps
embedding *state* in fp32 because their gradients are sparse/large-range, so
snapping embedding *weights* to ternary is expected to be the dominant quality
sink. The scope knob exists precisely to quantify that.

Scale modes (Ablations A–B):
  - "symmetric" (default): γ = mean(|W|), detached constant (pure STE). BIT-IDENTICAL
    to the pre-ablation behaviour.
  - "ttq":  Trained Ternary Quantization — separate LEARNABLE nn.Parameter scales
    γ₊ (for +1 codes) and γ₋ (for -1 codes). Both carry real gradients.
  - "dual": TernaryLLM Dual Learnable Ternarization — per-(output-row) learnable
    shift δ and scale γ as nn.Parameters. Effective weight = γ·codes + δ.

Group axis (scale_group):
  - "tensor" (default): one scale per whole weight tensor.
  - "128" / "64": one scale per group of `g` output-rows (handles small CCA
    projections like W_down_k=128, W_v=64).
  For nn.Embedding weights (shape [vocab, dim]), the group axis is dim 0 (vocab
  rows = tokens). group=64 means one scale per 64-token block. This is the
  correct axis — per-token-group scale, not per-feature-group. Same code path
  as nn.Linear; no special case needed.

Scale dtype (Ablation B, scale_dtype):
  - "fp16" (default): scales stored and applied as float16.
  - "int8": each group scale quantized to int8 with a per-group meta-scale (two
    levels: outer fp16 meta-scale × inner int8 encoded scale).
  - "pow2": each scale rounded to the nearest power of two (via torch.log2/round).
    Deploy needs only a bit-shift; the effective weight is still fp16-range.

Branch-free discipline: all mode/group/dtype decisions are resolved at parametrize
CONSTRUCTION (__init__), stored as fixed constants, and dispatched by a single
_forward_fn reference set at init. torch.compile sees a clean static graph.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor
from typing import Callable

# Which weight categories each scope quantizes.
# NOTE: "embeddings" category excludes lor_embed.space_embed regardless of scope.
# That exclusion is enforced in _categorize (path-name guard) and double-checked
# by a Lorentz safety assertion in apply_ternary_qat.
SCOPES: dict[str, set[str]] = {
    "backbone": {"backbone"},
    "backbone_attn": {"backbone", "attention"},
    "embeddings": {"embeddings"},          # token/euclidean/bigram only, Lorentz always excluded
    "full": {"backbone", "attention", "embeddings"},  # everything except Lorentz (always excluded)
}

# Valid choices for each knob.
VALID_MODES = {"symmetric", "ttq", "dual"}
VALID_GROUPS = {"tensor", "128", "64"}
VALID_DTYPES = {"fp16", "int8", "pow2"}


# ─────────────────────────────────────────────────────────────────────────────
# Scale encoding helpers (branch-free; called once per forward with fixed args)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_scale_fp16(raw: Tensor) -> Tensor:
    """Return raw scale unchanged (fp16 label means 'no extra quantization').

    The scale is applied in the weight's native dtype (bf16 in the forward),
    so no cast is needed here — this matches the original TernarySTE exactly
    and preserves bit-identical behaviour for the symmetric path.
    """
    return raw


def _encode_scale_int8(raw: Tensor) -> Tensor:
    """Quantize raw scale tensor to int8 + meta-scale (two-level).

    For each group's scale s: encode as round(s / meta) → int8, where
    meta = max(|s|) / 127 clamped to ≥ 1e-8. On decode: s_hat = int8_val * meta.
    Returns a stacked tensor [meta_f16, int8_as_f32] that decode() can unpack.
    Actually we fuse them back to fp16 immediately (training path); the int8
    encoding round-trip is the quality gate, not a real storage format here.
    """
    meta = raw.abs().max().clamp(min=1e-8) / 127.0
    quantized = (raw / meta).round().clamp(-127, 127).to(torch.int8)
    # Decode back to fp16 for the actual multiply (training QAT path).
    return (quantized.to(torch.float32) * meta).to(torch.float16)


def _encode_scale_pow2(raw: Tensor) -> Tensor:
    """Round each scale to the nearest power of two (shift-only deploy)."""
    clamped = raw.abs().clamp(min=1e-8)
    log2_rounded = torch.log2(clamped).round()
    return (2.0 ** log2_rounded).to(torch.float16) * raw.sign().clamp(min=1.0)


# Map string → encoder function (resolved at init, no runtime branch).
_SCALE_ENCODERS: dict[str, Callable[[Tensor], Tensor]] = {
    "fp16": _encode_scale_fp16,
    "int8": _encode_scale_int8,
    "pow2": _encode_scale_pow2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Group-scale utilities
# ─────────────────────────────────────────────────────────────────────────────

def _compute_group_scales(w: Tensor, group_size: int) -> Tensor:
    """Compute per-(output-row-group) mean absolute value.

    Parameters
    ----------
    w : Tensor, shape [out, in]
    group_size : int
        If group_size <= 0 or >= w.shape[0], treat as per-tensor (one scale).

    Returns
    -------
    scales : Tensor, shape [n_groups, 1, 1]  (broadcast-ready after reshape)
        where n_groups = ceil(out / group_size).
    """
    out_dim = w.shape[0]
    if group_size <= 0 or group_size >= out_dim:
        # Per-tensor: single scale, broadcast to full tensor.
        return w.detach().abs().mean().clamp(min=1e-8).view(1, 1)

    # Pad out_dim to a multiple of group_size, then reshape.
    n_full = out_dim // group_size
    remainder = out_dim % group_size

    # Compute scales for full groups.
    scales_list: list[Tensor] = []
    if n_full > 0:
        w_full = w.detach()[:n_full * group_size].view(n_full, -1)
        scales_list.append(w_full.abs().mean(dim=1, keepdim=True))  # [n_full, 1]
    if remainder > 0:
        w_tail = w.detach()[n_full * group_size:]
        scales_list.append(w_tail.abs().mean(keepdim=False).unsqueeze(0).unsqueeze(1))  # [1,1]

    return torch.cat(scales_list, dim=0).clamp(min=1e-8)  # [n_groups, 1]


def _apply_grouped_ste(
    w: Tensor,
    group_size: int,
    threshold: float,
    encode_scale: Callable[[Tensor], Tensor],
    scale_cap: Tensor | None = None,
) -> Tensor:
    """Symmetric grouped STE: effective_weight = encode(mean(|W_g|)) * codes_g.

    All scales are detached → pure straight-through (symmetric mode).

    scale_cap (Task #276 "B" — MORTAR-tile weight-explosion guard): optional per-group
    UPPER BOUND on the ternary scale γ=mean(|W_g|), shape [n_groups]. When given, each
    group's scale is min'd against its cap (= clip_mult × initial mean(|W_g|)). γ is the
    ENTIRE magnitude carrier of the {-1,0,+1}×γ weight, so capping it bounds the layer's
    output magnitude — a structural backstop against the g-collapse→α·m₂ blowup, independent
    of the optimizer. The cap is DETACHED (no grad) so the identity STE is unchanged.
    """
    out_dim, in_dim = w.shape

    if group_size <= 0 or group_size >= out_dim:
        # Per-tensor (group_size=tensor).
        scale = encode_scale(w.detach().abs().mean().clamp(min=1e-8).unsqueeze(0))[0]
        if scale_cap is not None:
            scale = torch.minimum(scale, scale_cap[0])
        w_norm = w / scale
        q = torch.sign(w_norm) * (w_norm.abs() > threshold).to(w.dtype)
        return w + (scale * q - w).detach()

    # Per-group VECTORIZED fast path — only when encode is identity (fp16) AND divisible.
    # int8/pow2 encode has PER-GROUP-max semantics (encode called on each 1-elem scale in
    # the loop), so those keep the loop to stay bit-identical. fp16 (default) = identity.
    if group_size and out_dim % group_size == 0 and encode_scale is _encode_scale_fp16:
        gs = group_size
        ng = out_dim // gs
        wr = w.reshape(ng, gs * in_dim)
        s = wr.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)   # encode=identity
        if scale_cap is not None:
            s = torch.minimum(s, scale_cap.view(ng, 1))
        w_n = wr / s
        q = torch.sign(w_n) * (w_n.abs() > threshold).to(w.dtype)
        out = wr + (s * q - wr).detach()
        return out.reshape(out_dim, in_dim)

    # Per-group: iterate over groups, collect, reassemble (non-divisible OR int8/pow2).
    n_full = out_dim // group_size
    parts: list[Tensor] = []

    for g in range(n_full):
        w_g = w[g * group_size:(g + 1) * group_size]  # [gs, in]
        s_raw = w_g.detach().abs().mean().clamp(min=1e-8).unsqueeze(0)
        s = encode_scale(s_raw)[0]
        if scale_cap is not None:
            s = torch.minimum(s, scale_cap[g])
        w_n = w_g / s
        q_g = torch.sign(w_n) * (w_n.abs() > threshold).to(w.dtype)
        parts.append(w_g + (s * q_g - w_g).detach())

    remainder = out_dim % group_size
    if remainder > 0:
        w_g = w[n_full * group_size:]
        s_raw = w_g.detach().abs().mean().clamp(min=1e-8).unsqueeze(0)
        s = encode_scale(s_raw)[0]
        if scale_cap is not None:
            s = torch.minimum(s, scale_cap[n_full])
        w_n = w_g / s
        q_g = torch.sign(w_n) * (w_n.abs() > threshold).to(w.dtype)
        parts.append(w_g + (s * q_g - w_g).detach())

    return torch.cat(parts, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# TernarySTE parametrization
# ─────────────────────────────────────────────────────────────────────────────

class TernarySTE(nn.Module):
    """Ternary straight-through estimator parametrization.

    Supports three scale modes, per-group scales, and scale-precision ablations.
    All dispatch decisions are resolved at construction time (branch-free hotpath).

    Parameters
    ----------
    threshold : float
        |w_norm| ≤ threshold → code 0.  Default 0.5 (Bonsai).
    weight_shape : tuple[int, ...]
        Shape of the weight tensor being parametrized. Required to pre-allocate
        learnable parameters (ttq / dual modes).
    mode : str
        "symmetric" | "ttq" | "dual".
    group : int
        Group size along output dim. 0 → per-tensor (default).
    scale_dtype : str
        "fp16" | "int8" | "pow2".
    """

    def __init__(
        self,
        threshold: float = 0.5,
        weight_shape: tuple[int, ...] = (1,),
        mode: str = "symmetric",
        group: int = 0,
        scale_dtype: str = "fp16",
        device: torch.device | None = None,
        weight_init: torch.Tensor | None = None,
        scale_clip_mult: float = 0.0,
    ) -> None:
        super().__init__()
        assert mode in VALID_MODES, f"unknown mode={mode!r}"
        assert scale_dtype in VALID_DTYPES, f"unknown scale_dtype={scale_dtype!r}"

        self.threshold = float(threshold)
        self.mode = mode
        self.group = int(group)  # 0 → per-tensor
        self.scale_dtype = scale_dtype
        self._encode_scale = _SCALE_ENCODERS[scale_dtype]
        self.scale_clip_mult = float(scale_clip_mult)

        out_dim = weight_shape[0] if len(weight_shape) >= 1 else 1
        in_dim = weight_shape[1] if len(weight_shape) >= 2 else 1
        n_groups = self._n_groups(out_dim)

        if mode == "ttq":
            # TTQ: one LEARNABLE γ₊ and γ₋ per group (separate positive/negative scales).
            # Initialized from mean(|W|) per group so the threshold is scale-aware.
            # Created on the same device as the weight so bnb optimizer sees all on GPU.
            gamma_init = self._compute_group_scale_init(weight_init, out_dim, n_groups, device)
            self.gamma_pos = nn.Parameter(gamma_init.clone())
            self.gamma_neg = nn.Parameter(gamma_init.clone())
            self._forward_fn = self._forward_ttq
        elif mode == "dual":
            # Dual: per-group learnable shift δ and scale γ.
            # δ initialized to 0; γ initialized from mean(|W|) so the threshold works
            # correctly at init (|w/γ| > threshold is meaningful for the true weight scale).
            gamma_init = self._compute_group_scale_init(weight_init, out_dim, n_groups, device)
            self.delta = nn.Parameter(torch.zeros(n_groups, dtype=torch.float32, device=device))
            self.gamma = nn.Parameter(gamma_init)
            self._forward_fn = self._forward_dual
        else:
            # symmetric: no learnable parameters — pure straight-through.
            self._forward_fn = self._forward_symmetric

        # Task #276 "B": per-group ternary-scale UPPER BOUND, anchored to the INITIAL scale
        # (mult × mean(|W_g|) at registration). An explosion can't ratchet the cap up because
        # the anchor is fixed. Registered as a buffer (moves with .to(), persists in state_dict,
        # reloads identically on resume). None → off (branch resolved once, here, not per-forward).
        cap = None
        if self.scale_clip_mult > 0.0:
            ref = self._compute_group_scale_init(weight_init, out_dim, n_groups, device)
            cap = (self.scale_clip_mult * ref).float()
        self.register_buffer("_scale_cap", cap)  # None placeholder when off (valid buffer value)

    def _compute_group_scale_init(
        self,
        weight_init: torch.Tensor | None,
        out_dim: int,
        n_groups: int,
        device: torch.device | None,
    ) -> torch.Tensor:
        """Compute per-group mean(|W|) for initializing learnable scales.

        If weight_init is not provided (None), fall back to 1.0.
        The scale is detached (initialization only) and cast to fp32.
        """
        if weight_init is None or weight_init.numel() == 0:
            return torch.ones(n_groups, dtype=torch.float32, device=device)

        w = weight_init.detach().float()
        gs = self.group  # 0 = per-tensor

        if gs <= 0 or gs >= out_dim:
            # Per-tensor: one global mean(|W|).
            s = w.abs().mean().clamp(min=1e-8)
            return s.unsqueeze(0).to(device=device)

        # Per-group.
        scales = []
        n_full = out_dim // gs
        for g_idx in range(n_full):
            w_g = w[g_idx * gs:(g_idx + 1) * gs]
            scales.append(w_g.abs().mean().clamp(min=1e-8))
        if out_dim % gs:
            w_tail = w[n_full * gs:]
            scales.append(w_tail.abs().mean().clamp(min=1e-8))
        return torch.tensor(scales, dtype=torch.float32, device=device)

    def _n_groups(self, out_dim: int) -> int:
        if self.group <= 0 or self.group >= out_dim:
            return 1
        n_full = out_dim // self.group
        remainder = out_dim % self.group
        return n_full + (1 if remainder > 0 else 0)

    # ── Forward dispatch (selected at init, no Python branch at runtime) ──────

    def forward(self, w: Tensor) -> Tensor:
        return self._forward_fn(w)

    def _forward_symmetric(self, w: Tensor) -> Tensor:
        """BIT-IDENTICAL to the original TernarySTE for group=tensor, dtype=fp16.

        For group>0 or non-fp16 dtype: same math, grouped/encoded.
        """
        return _apply_grouped_ste(w, self.group, self.threshold, self._encode_scale,
                                  scale_cap=self._scale_cap)

    def _forward_ttq(self, w: Tensor) -> Tensor:
        """TTQ: separate LEARNABLE γ₊, γ₋ per group (real grad through both).

        Codes = round_clip(w / γ_init_scale_detached, -1, +1) where
        γ_init_scale_detached = mean(|W|) used as STE anchor only;
        effective_weight = γ₊·pos_codes + γ₋·(−neg_codes).

        Both γ₊ and γ₋ carry real gradients; the code rounding is pure STE.
        """
        out_dim = w.shape[0]

        if self.group <= 0 or self.group >= out_dim:
            # Per-tensor TTQ.
            ste_scale = w.detach().abs().mean().clamp(min=1e-8)
            w_norm = w / ste_scale
            codes = torch.sign(w_norm) * (w_norm.abs() > self.threshold).to(w.dtype)
            # codes are {-1, 0, +1}, detached from W for the scale grads.
            codes_sg = codes.detach()
            pos_mask = (codes_sg > 0).to(w.dtype)
            neg_mask = (codes_sg < 0).to(w.dtype)
            gp = self.gamma_pos[0].to(w.dtype)
            gn = self.gamma_neg[0].to(w.dtype)
            w_eff = gp * pos_mask - gn * neg_mask  # learnable magnitude
            # STE: gradient flows to w through the codes path, and to γ₊/γ₋ directly.
            return w + (w_eff - w).detach() + (w_eff - w_eff.detach())

        # Grouped TTQ — VECTORIZED (bit-identical to the per-group loop, but no Python
        # iteration). For embeddings [49152,768] @ group=64 the loop was 768 iters/table;
        # this is a single reshape→reduce→broadcast. Falls back to the loop only when
        # out_dim is not divisible by group (never happens for MORPH's dims; kept for safety).
        if out_dim % self.group == 0:
            gs = self.group
            ng = out_dim // gs
            wr = w.reshape(ng, gs * w.shape[1])                    # group rows row-major
            s = wr.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
            w_n = wr / s
            codes_sg = (torch.sign(w_n) * (w_n.abs() > self.threshold).to(w.dtype)).detach()
            gp = self.gamma_pos.to(w.dtype).reshape(ng, 1)
            gn = self.gamma_neg.to(w.dtype).reshape(ng, 1)
            w_eff = gp * (codes_sg > 0).to(w.dtype) - gn * (codes_sg < 0).to(w.dtype)
            out = wr + (w_eff - wr).detach() + (w_eff - w_eff.detach())
            return out.reshape(out_dim, w.shape[1])

        # Non-divisible fallback (per-group loop).
        n_full = out_dim // self.group
        remainder = out_dim % self.group
        parts: list[Tensor] = []
        for g_idx in range(n_full):
            w_g = w[g_idx * self.group:(g_idx + 1) * self.group]
            s = w_g.detach().abs().mean().clamp(min=1e-8)
            w_n = w_g / s
            codes = torch.sign(w_n) * (w_n.abs() > self.threshold).to(w.dtype)
            codes_sg = codes.detach()
            gp = self.gamma_pos[g_idx].to(w.dtype)
            gn = self.gamma_neg[g_idx].to(w.dtype)
            w_eff = gp * (codes_sg > 0).to(w.dtype) - gn * (codes_sg < 0).to(w.dtype)
            parts.append(w_g + (w_eff - w_g).detach() + (w_eff - w_eff.detach()))
        if remainder > 0:
            g_idx = n_full
            w_g = w[n_full * self.group:]
            s = w_g.detach().abs().mean().clamp(min=1e-8)
            w_n = w_g / s
            codes = torch.sign(w_n) * (w_n.abs() > self.threshold).to(w.dtype)
            codes_sg = codes.detach()
            gp = self.gamma_pos[g_idx].to(w.dtype)
            gn = self.gamma_neg[g_idx].to(w.dtype)
            w_eff = gp * (codes_sg > 0).to(w.dtype) - gn * (codes_sg < 0).to(w.dtype)
            parts.append(w_g + (w_eff - w_g).detach() + (w_eff - w_eff.detach()))
        return torch.cat(parts, dim=0)

    def _forward_dual(self, w: Tensor) -> Tensor:
        """TernaryLLM Dual: learnable per-group shift δ + scale γ.

        codes = round_clip((w − δ) / γ_detached, −1, +1)  [STE through round]
        w_eff = γ · codes + δ

        γ in the multiply gets real gradient; γ in the divide is detached (STE anchor).
        δ gets gradient via the additive term.
        """
        out_dim = w.shape[0]

        if self.group <= 0 or self.group >= out_dim:
            d = self.delta[0].to(w.dtype)
            g = self.gamma[0].to(w.dtype)
            g_det = g.detach()
            w_shifted = w - d
            w_norm = w_shifted / g_det.clamp(min=1e-8)
            codes = torch.sign(w_norm) * (w_norm.abs() > self.threshold).to(w.dtype)
            w_eff = g * codes.detach() + d  # codes detached, g/d carry grad
            # STE for the codes path: grad flows straight to w as identity.
            return w + (w_eff - w).detach() + (w_eff - w_eff.detach())

        # Grouped Dual — VECTORIZED (bit-identical to the per-group loop). δ carries grad
        # only through the additive +δ term (codes are detached, so δ's grad through the
        # shift in w_n is killed); γ carries grad through the multiply, γ.detach() is the
        # STE divisor. Loop fallback only when out_dim not divisible by group.
        if out_dim % self.group == 0:
            gs = self.group
            ng = out_dim // gs
            wr = w.reshape(ng, gs * w.shape[1])
            d = self.delta.to(w.dtype).reshape(ng, 1)
            g_scale = self.gamma.to(w.dtype).reshape(ng, 1)
            g_det = g_scale.detach().clamp(min=1e-8)
            w_n = (wr - d) / g_det
            codes = (torch.sign(w_n) * (w_n.abs() > self.threshold).to(w.dtype)).detach()
            w_eff = g_scale * codes + d
            out = wr + (w_eff - wr).detach() + (w_eff - w_eff.detach())
            return out.reshape(out_dim, w.shape[1])

        # Non-divisible fallback (per-group loop).
        n_full = out_dim // self.group
        remainder = out_dim % self.group
        parts: list[Tensor] = []
        for g_idx in range(n_full):
            w_g = w[g_idx * self.group:(g_idx + 1) * self.group]
            d = self.delta[g_idx].to(w.dtype)
            g_scale = self.gamma[g_idx].to(w.dtype)
            g_det = g_scale.detach().clamp(min=1e-8)
            w_shifted = w_g - d
            w_n = w_shifted / g_det
            codes = torch.sign(w_n) * (w_n.abs() > self.threshold).to(w.dtype)
            w_eff = g_scale * codes.detach() + d
            parts.append(w_g + (w_eff - w_g).detach() + (w_eff - w_eff.detach()))
        if remainder > 0:
            g_idx = n_full
            w_g = w[n_full * self.group:]
            d = self.delta[g_idx].to(w.dtype)
            g_scale = self.gamma[g_idx].to(w.dtype)
            g_det = g_scale.detach().clamp(min=1e-8)
            w_shifted = w_g - d
            w_n = w_shifted / g_det
            codes = torch.sign(w_n) * (w_n.abs() > self.threshold).to(w.dtype)
            w_eff = g_scale * codes.detach() + d
            parts.append(w_g + (w_eff - w_g).detach() + (w_eff - w_eff.detach()))
        return torch.cat(parts, dim=0)

    def right_inverse(self, w: Tensor) -> Tensor:
        # Map an assigned weight back to the underlying smooth parameter (identity).
        return w


# ─────────────────────────────────────────────────────────────────────────────
# Module categorization
# ─────────────────────────────────────────────────────────────────────────────

def _attention_linear_ids(model: nn.Module) -> set[int]:
    """ids of every nn.Linear that lives inside an attention module."""
    ids: set[int] = set()
    for m in model.modules():
        if "Attention" in type(m).__name__:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    ids.add(id(sub))
    return ids


# The Lorentz space embedding is geometry-critical (trained geodesic radius ~0.97)
# and MUST NEVER be ternarized. It is excluded by matching its exact attribute path.
# The path is relative to the top-level model, e.g. "embed.hybrid.lor_embed.space_embed".
_LORENTZ_SPACE_EMBED_SUFFIX = "lor_embed.space_embed"


def _categorize(name: str, module: nn.Module, attn_ids: set[int]) -> str | None:
    """Return the ternary category for a module, or None if it carries no
    quantizable weight matrix (norms, scalars, biases, container modules).

    The Lorentz space embedding (lor_embed.space_embed) always returns None —
    it is geometry-critical and must stay bf16 regardless of scope.
    """
    if isinstance(module, nn.Embedding):
        # Guard: never ternarize the Lorentz space embedding.
        # Match by path suffix — robust across nesting depth (e.g., model.embed.hybrid.
        # lor_embed.space_embed, or embed.hybrid.lor_embed.space_embed, etc.).
        if name == _LORENTZ_SPACE_EMBED_SUFFIX or name.endswith("." + _LORENTZ_SPACE_EMBED_SUFFIX):
            return None  # Lorentz space embed — always excluded
        return "embeddings"
    # Guard: never ternarize the Hyper-Connection coefficient projection (W_fused). It
    # is a tiny, precision-sensitive CONTROL path that generates the orthogonal /
    # doubly-stochastic stream mixer via Cayley / Sinkhorn — ternarizing it to {-1,0,+1}
    # would destroy the manifold constraint (same rationale as excluding norms/gates).
    # Always bf16, regardless of scope. Lives at ``*.mrr_attn.proj`` / ``*.mrr_mlp.proj``.
    if name.endswith("mrr_attn.proj") or name.endswith("mrr_mlp.proj"):
        return None
    # Guard: never ternarize the loop-axis SSM (DiagonalInjection) control paths.
    # Same rationale as the HC proj above: tiny, precision-sensitive control matrices.
    # Always bf16, regardless of scope.
    if (name.endswith("injection.W_a") or name.endswith("injection.W_dt")
            or name.endswith("injection.B")):
        return None
    # CMSBlockLinear (inside MortarLinear) holds a dense [out, in] nn.Parameter
    # named `weight` in dense mode — quantize it like any other linear.
    is_cms = type(module).__name__ == "CMSBlockLinear"
    if isinstance(module, nn.Linear) or is_cms:
        w = getattr(module, "weight", None)
        if not isinstance(w, nn.Parameter) or w.dim() < 2:
            return None  # e.g. CMSBlockLinear in sparse mode (no dense .weight)
        return "attention" if id(module) in attn_ids else "backbone"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_ternary_qat(
    model: nn.Module,
    scope: str = "backbone",
    threshold: float = 0.5,
    scale_mode: str = "symmetric",
    scale_group: str = "tensor",
    scale_dtype: str = "fp16",
    scale_clip_mult: float = 0.0,
) -> dict:
    """Register the ternary STE parametrization on every weight matrix selected
    by ``scope``. Call AFTER model construction and BEFORE torch.compile and
    optimizer construction (so any nn.Parameter scales are picked up by the optimizer).

    The Lorentz space embedding (``lor_embed.space_embed``) is ALWAYS excluded,
    regardless of scope. This is enforced by ``_categorize`` (path-name guard) and
    double-checked by a safety assertion below. It is geometry-critical (trained
    geodesic radius ~0.97) and must stay bf16.

    For nn.Embedding weights (shape [vocab, dim]), group scales are per-token-block:
    group=64 → one scale per 64 vocabulary rows. The group axis is dim 0 (vocab),
    which is correct — same as "output rows" for nn.Linear.

    Parameters
    ----------
    model : nn.Module
        The model to parametrize.
    scope : str
        "backbone" | "backbone_attn" | "embeddings" | "full".
        "embeddings" ternarizes only embedding tables (token/euclidean/bigram),
        leaving backbone and attention as bf16. "full" adds backbone+attention on
        top, but still excludes Lorentz. Use "embeddings" for the isolation arm
        and "full" for the complete stack.
    threshold : float
        Ternary threshold (default 0.5 — Bonsai).
    scale_mode : str
        "symmetric" (default, bit-identical to pre-ablation) | "ttq" | "dual".
    scale_group : str
        "tensor" (default) | "128" | "64" — group size along output dim (dim 0).
        For embeddings, dim 0 is the vocabulary axis (per-token-block scales).
    scale_dtype : str
        "fp16" (default) | "int8" | "pow2" — scale encoding for deploy fidelity test.

    Returns
    -------
    manifest : dict
        {scope, threshold, scale_mode, scale_group, scale_dtype,
         counts, n_modules_ternary, n_params_ternary, n_params_total,
         frac_params_ternary, module_names, lorentz_untouched}.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown ternary_scope={scope!r}; choices={list(SCOPES)}")
    if scale_mode not in VALID_MODES:
        raise ValueError(f"unknown scale_mode={scale_mode!r}; choices={list(VALID_MODES)}")
    if scale_group not in VALID_GROUPS:
        raise ValueError(f"unknown scale_group={scale_group!r}; choices={list(VALID_GROUPS)}")
    if scale_dtype not in VALID_DTYPES:
        raise ValueError(f"unknown scale_dtype={scale_dtype!r}; choices={list(VALID_DTYPES)}")

    wanted = SCOPES[scope]
    attn_ids = _attention_linear_ids(model)

    # Parse group size: "tensor" → 0 (per-tensor), "128" → 128, "64" → 64.
    group_size = 0 if scale_group == "tensor" else int(scale_group)

    counts: dict[str, int] = {"backbone": 0, "attention": 0, "embeddings": 0}
    names: list[str] = []
    n_tern = 0

    for name, module in model.named_modules():
        cat = _categorize(name, module, attn_ids)
        if cat is None or cat not in wanted:
            continue
        if parametrize.is_parametrized(module, "weight"):
            continue  # idempotent — don't double-register

        w = module.weight  # pre-parametrize original weight
        w_shape = tuple(w.shape)
        w_device = w.device  # pass device so learnable params live on GPU

        ste = TernarySTE(
            threshold=threshold,
            weight_shape=w_shape,
            mode=scale_mode,
            group=group_size,
            scale_dtype=scale_dtype,
            device=w_device,
            weight_init=w.detach(),  # for mean(|W|) gamma initialization
            scale_clip_mult=scale_clip_mult,  # Task #276 "B" weight-explosion guard
        )
        parametrize.register_parametrization(module, "weight", ste)
        counts[cat] += 1
        names.append(f"{name} [{cat}]")
        n_tern += module.weight.numel()

    # ── Lorentz safety assertion (belt + suspenders after _categorize guard) ──
    # Walk ALL nn.Embedding modules and confirm lor_embed.space_embed was NOT touched.
    # This is an unconditional post-condition check regardless of scope.
    lorentz_guarded: list[str] = []
    for lor_name, lor_mod in model.named_modules():
        if not isinstance(lor_mod, nn.Embedding):
            continue
        if lor_name == _LORENTZ_SPACE_EMBED_SUFFIX or lor_name.endswith("." + _LORENTZ_SPACE_EMBED_SUFFIX):
            if parametrize.is_parametrized(lor_mod, "weight"):
                raise RuntimeError(
                    "apply_ternary_qat: LORENTZ SAFETY VIOLATION — "
                    f"{lor_name!r} was parametrized. This must NEVER happen. "
                    "The Lorentz space embedding is geometry-critical and must stay bf16."
                )
            lorentz_guarded.append(lor_name)

    n_total = sum(p.numel() for p in model.parameters())
    return {
        "scope": scope,
        "threshold": threshold,
        "scale_mode": scale_mode,
        "scale_group": scale_group,
        "scale_dtype": scale_dtype,
        "counts": counts,
        "n_modules_ternary": sum(counts.values()),
        "n_params_ternary": int(n_tern),
        "n_params_total": int(n_total),
        "frac_params_ternary": float(n_tern) / max(1, n_total),
        "module_names": names,
        "lorentz_untouched": lorentz_guarded,  # list of Lorentz embed paths confirmed bf16
    }


def reparametrize_compacted_values_ternary(cms: nn.Module, threshold: float = 0.5) -> bool:
    """Continue ternary QAT on a compacted CMS layer's ``values`` after compaction.

    Call AFTER compact() and BEFORE the optimizer rebuild. Idempotent. Returns True if
    values-ternary QAT was newly enabled. The smooth survivor values stay the trainable
    shadow; the sparse forward applies a per-tensor symmetric STE so the effective sparse
    weight is ternary ("keep pretraining the ternary model, now compacted").

    NB: this uses the CMS internal STE flag (enable_values_ternary), NOT torch.parametrize
    — register_parametrization cannot bind a tensor named "values" (its parametrizations
    ModuleDict reserves the .values() method), so the parametrize path is impossible here.
    """
    if getattr(cms, "_dense_mode", True):
        raise RuntimeError(
            "reparametrize_compacted_values_ternary requires a compacted (sparse) layer"
        )
    if getattr(cms, "_values_ternary_mode", False):
        return False  # idempotent
    cms.enable_values_ternary(threshold)
    return True
