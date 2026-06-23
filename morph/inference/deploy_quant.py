"""Packed ternary / int-N INFERENCE materialization for MORPH scale tests.

Training stores quantized weights as QAT *shadows* (bf16/fp32 + STE) — fine at 276M,
fatal at 30B (60-120 GB). This module materializes the DEPLOY-EFFECTIVE weights in
their real storage cost:

  - MLP backbone (post prune_step_blocks + carve, MORTAR BCSR 0.25): mortar_data is
    ternarized with the EXACT ``CMSBlockLinear._mortar_effective_data`` formula
    (per-tensor symmetric, threshold 0.5) and packed 4 codes/byte (2-bit). The bf16
    mortar_data parameter is DELETED; an instance-bound ``_mortar_effective_data``
    override dequantizes into a transient bf16 buffer per forward, feeding the
    unchanged ``_forward_mortar`` / stk BCSR kernel path.
  - Dense backbone Linears (x0/value ChannelInject projs, LMHeadMixer.mix): same
    per-tensor symmetric ternary (``TernarySTE`` scope=backbone semantics), packed
    2-bit, exposed as an ``nn.Linear`` subclass whose ``weight`` is a dequantizing
    property (so ``isinstance(..., nn.Linear)`` guards and direct ``.weight`` reads
    — e.g. ``ChannelInject.precompute`` — keep working byte-for-byte).
  - Attention projections: per-output-row int8 (EXACT ``IntNLinearSTE`` semantics:
    s_row = absmax/127, codes = round(w/s).clamp(±127)), stored as int8 codes +
    fp32 row scales.
  - euc/bigram embeddings: per-row int6 (EXACT ``IntNRowSTE`` semantics, hi=31),
    int6 values in an int8 container + fp32 row scales. Lorentz space embedding and
    value-embed tables stay bf16 — the real deploy stack never quantizes them.

Inference-only: no gradients, no STE, no optimizer. The dequantized forward values
are IDENTICAL to what the corresponding QAT forward would produce for the same
underlying weights (same formulas, baked once instead of recomputed per step).
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from morph.model.sparsity import MortarLinear

__all__ = [
    "pack_ternary_codes", "unpack_ternary", "extract_ternary_from_parametrized",
    "PackedTernaryLinear", "Int8RowLinear", "Int6RowEmbedding",
    "pack_mortar_ternary", "strip_cms_inference", "shrink_mlp_to_mortar_ternary",
    "quantize_attention_linears", "resident_bytes_report",
    "to_deploy_inference",
]

_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2-bit packing primitives
# ─────────────────────────────────────────────────────────────────────────────

def pack_ternary_codes(q: Tensor) -> Tensor:
    """Pack ternary codes {-1,0,+1} (any float/int dtype) into uint8, 4 codes/byte.

    Code mapping: c+1 ∈ {0,1,2} stored in 2 bits, little-endian within the byte.
    Length is padded to a multiple of 4 with zeros (code 0 ⇒ value -1·scale for the
    pad slots, but they are never read back: unpack slices to the true numel).
    """
    flat = q.reshape(-1)
    n = flat.numel()
    pad = (-n) % 4
    u = (flat.to(torch.int16) + 1).to(torch.uint8)
    if pad:
        u = torch.cat([u, torch.zeros(pad, dtype=torch.uint8, device=u.device)])
    u = u.view(-1, 4)
    packed = (u[:, 0] | (u[:, 1] << 2) | (u[:, 2] << 4) | (u[:, 3] << 6)).contiguous()
    return packed


def unpack_ternary(packed: Tensor, numel: int, dtype: torch.dtype) -> Tensor:
    """Inverse of pack_ternary_codes → flat tensor of {-1,0,+1} in `dtype`."""
    shifts = _SHIFTS.to(packed.device)
    u = (packed.unsqueeze(-1) >> shifts) & 3          # [n//4, 4] uint8
    return u.reshape(-1)[:numel].to(dtype) - 1.0


def _ternary_quantize(w: Tensor, threshold: float = 0.5) -> tuple[Tensor, Tensor]:
    """EXACT TernarySTE symmetric per-tensor effective-weight quantization.

    Returns (codes ∈ {-1,0,+1} int8, scale fp32 scalar). Effective weight = scale·codes.

    NOTE: this re-derives the scale from `mean(|w|)`. It is CORRECT only when `w` is the
    smooth *shadow* weight (the live parameter). It is WRONG if `w` is the already-ternarized
    STE *output* (mean(|γ·codes|) ≠ mean(|shadow|)). For a parametrized module use
    ``extract_ternary_from_parametrized`` which reads the shadow from the parametrization.
    """
    wf = w.detach().float()
    scale = wf.abs().mean().clamp(min=1e-8)
    wn = wf / scale
    q = (torch.sign(wn) * (wn.abs() > threshold)).to(torch.int8)
    return q, scale


def extract_ternary_from_parametrized(
    module: nn.Module, param_name: str = "weight",
) -> tuple[Tensor, Tensor, dict]:
    """Extract (codes_int8 [out,in] ∈ {-1,0,+1}, scale, meta) from a TernarySTE-parametrized
    module, reproducing the STE forward output ``module.weight`` BIT-EXACTLY.

    This mirrors ``ternary_qat._apply_grouped_ste`` exactly (symmetric mode), but reads the
    smooth *shadow* parameter from the parametrization so the scale is the true ``mean(|W|)``
    the forward uses — NOT the degenerate ``mean(|γ·codes|)`` you would get by re-quantizing
    the STE output. Handles:
      - per-tensor scale (group=0, the deployed case) and grouped scale (group=64/128),
      - fp16 / int8 / pow2 scale-encoding (the same ``_encode_scale_*`` the forward applies),
      - the optional per-group ``_scale_cap`` buffer (scale_clip_mult > 0).

    Returns
    -------
    codes : int8 [out, in], values in {-1, 0, +1}
    scale : fp32. Shape [1] for per-tensor; [n_groups] for grouped (per-output-row-group).
    meta  : dict {group, n_groups, mode, scale_dtype, out, in, group_size}
    """
    import torch.nn.utils.parametrize as parametrize
    from morph.model.ternary_qat import TernarySTE, _SCALE_ENCODERS

    if not parametrize.is_parametrized(module, param_name):
        raise ValueError(f"{module} is not parametrized on {param_name!r}")
    plist = module.parametrizations[param_name]
    ste = plist[0]
    if not isinstance(ste, TernarySTE):
        raise TypeError(f"parametrization is {type(ste).__name__}, not TernarySTE")
    if ste.mode != "symmetric":
        raise NotImplementedError(
            f"extract_ternary_from_parametrized supports symmetric mode only "
            f"(got mode={ste.mode!r}); ttq/dual carry learnable γ that the packed "
            f"per-tensor/group scale format does not represent.")

    shadow = plist.original.detach()              # smooth weight, native dtype
    out_dim, in_dim = shadow.shape
    thr = ste.threshold
    group = ste.group                             # 0 → per-tensor
    encode = _SCALE_ENCODERS[ste.scale_dtype]
    cap = getattr(ste, "_scale_cap", None)        # buffer or None

    def _scale_for(w_part: Tensor, gidx: int) -> Tensor:
        # mean(|.|) clamped, encoded, then capped — EXACT order of _apply_grouped_ste.
        s_raw = w_part.abs().mean().clamp(min=1e-8).unsqueeze(0)
        s = encode(s_raw)[0]
        if cap is not None:
            s = torch.minimum(s, cap[gidx])
        return s

    if group <= 0 or group >= out_dim:
        # ── per-tensor ──
        scale = _scale_for(shadow, 0)
        w_n = shadow / scale
        q = torch.sign(w_n) * (w_n.abs() > thr).to(shadow.dtype)
        codes = q.to(torch.int8)
        scale_out = scale.reshape(1).float().contiguous()
        meta = {"group": 0, "n_groups": 1, "mode": "symmetric",
                "scale_dtype": ste.scale_dtype, "out": out_dim, "in": in_dim,
                "group_size": 0}
        return codes.contiguous(), scale_out, meta

    # ── grouped (per output-row-group) ──
    n_full = out_dim // group
    remainder = out_dim % group
    n_groups = n_full + (1 if remainder else 0)
    codes = torch.empty_like(shadow, dtype=torch.int8)
    scales: list[Tensor] = []
    for g in range(n_full):
        sl = slice(g * group, (g + 1) * group)
        s = _scale_for(shadow[sl], g)
        w_n = shadow[sl] / s
        codes[sl] = (torch.sign(w_n) * (w_n.abs() > thr).to(shadow.dtype)).to(torch.int8)
        scales.append(s.reshape(1))
    if remainder:
        sl = slice(n_full * group, out_dim)
        s = _scale_for(shadow[sl], n_full)
        w_n = shadow[sl] / s
        codes[sl] = (torch.sign(w_n) * (w_n.abs() > thr).to(shadow.dtype)).to(torch.int8)
        scales.append(s.reshape(1))
    scale_out = torch.cat(scales).float().contiguous()  # [n_groups]
    meta = {"group": group, "n_groups": n_groups, "mode": "symmetric",
            "scale_dtype": ste.scale_dtype, "out": out_dim, "in": in_dim,
            "group_size": group}
    return codes.contiguous(), scale_out, meta


# ─────────────────────────────────────────────────────────────────────────────
# Module replacements (nn.Linear subclasses → isinstance + .weight reads survive)
# ─────────────────────────────────────────────────────────────────────────────

class PackedTernaryLinear(nn.Linear):
    """Frozen inference Linear with 2-bit-packed ternary weight.

    Subclasses nn.Linear so isinstance checks pass; ``weight`` is a property that
    dequantizes on access (forward reads it exactly once per call).

    Scale layout:
      - per-tensor: ``scale`` is shape [1]; broadcast over the whole weight.
      - grouped:    ``scale`` is shape [n_groups]; ``group_size`` rows share one scale
                    (per-output-row-group, dim-0 axis — matches TernarySTE grouping).
    """

    def __init__(self, in_features: int, out_features: int,
                 packed: Tensor, scale: Tensor, dtype: torch.dtype = torch.bfloat16,
                 group_size: int = 0):
        nn.Module.__init__(self)          # skip nn.Linear's dense alloc
        self.in_features = in_features
        self.out_features = out_features
        self._act_dtype = dtype
        self.group_size = int(group_size)        # 0 → per-tensor
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)     # fp32 [1] or [n_groups]
        self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, threshold: float = 0.5) -> "PackedTernaryLinear":
        """Quantize a PLAIN (non-parametrized) Linear from its dense .weight.

        For a TernarySTE-parametrized module use ``from_parametrized`` instead —
        re-deriving the scale from the STE *output* gives the wrong γ.
        """
        assert lin.bias is None, "packed ternary path expects bias-free Linears"
        q, scale = _ternary_quantize(lin.weight.data, threshold)
        return cls(lin.in_features, lin.out_features,
                   pack_ternary_codes(q), scale.reshape(1).float(),
                   dtype=lin.weight.dtype, group_size=0)

    @classmethod
    def from_parametrized(cls, lin: nn.Module) -> "PackedTernaryLinear":
        """Build from a TernarySTE-parametrized Linear (symmetric mode), reproducing
        the STE forward output bit-exactly. Reads the smooth shadow for the true scale.
        """
        assert getattr(lin, "bias", None) is None, "packed ternary path expects bias-free Linears"
        codes, scale, meta = extract_ternary_from_parametrized(lin, "weight")
        # dtype of the deployed forward = the STE output dtype = shadow dtype.
        dtype = lin.parametrizations.weight.original.dtype
        return cls(meta["in"], meta["out"], pack_ternary_codes(codes), scale,
                   dtype=dtype, group_size=meta["group_size"])

    @property
    def weight(self) -> Tensor:  # type: ignore[override]
        n = self.out_features * self.in_features
        w = unpack_ternary(self.packed, n, self._act_dtype).view(
            self.out_features, self.in_features)
        if self.group_size <= 0 or self.scale.numel() == 1:
            return w * self.scale.to(self._act_dtype)[0]
        # grouped: expand the per-group scale over its rows (handles a ragged tail group).
        out = self.out_features
        gs = self.group_size
        idx = (torch.arange(out, device=w.device) // gs).clamp_(max=self.scale.numel() - 1)
        s = self.scale.to(self._act_dtype)[idx].unsqueeze(1)   # [out, 1]
        return w * s

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    def extra_repr(self) -> str:
        g = "per-tensor" if self.group_size <= 0 else f"group={self.group_size}"
        return f"in={self.in_features}, out={self.out_features}, packed-2bit ternary ({g})"


class Int8RowLinear(nn.Linear):
    """Frozen inference Linear with per-output-row int8 weight (IntNLinearSTE semantics)."""

    def __init__(self, in_features: int, out_features: int,
                 codes: Tensor, scale: Tensor, dtype: torch.dtype = torch.bfloat16):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self._act_dtype = dtype
        self.register_buffer("codes", codes)      # int8 [out, in]
        self.register_buffer("scale", scale)      # fp32 [out, 1]
        self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int = 8) -> "Int8RowLinear":
        assert lin.bias is None, "attention projections are bias-free"
        hi = {8: 127, 6: 31, 4: 7}[bits]
        w = lin.weight.data.detach().float()
        s = w.abs().amax(dim=-1, keepdim=True).div(hi).clamp(min=1e-8)
        codes = (w / s).round().clamp(-hi, hi).to(torch.int8)
        return cls(lin.in_features, lin.out_features, codes, s, dtype=lin.weight.dtype)

    @property
    def weight(self) -> Tensor:  # type: ignore[override]
        return self.codes.to(self._act_dtype) * self.scale.to(self._act_dtype)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8/row"


class Int6RowEmbedding(nn.Module):
    """Frozen inference Embedding with per-row int6 codes in an int8 container
    (IntNRowSTE semantics, hi=31). ``weight`` is a dequantizing property with an
    explicit cache for the weight-tied LM head (``HybridEmbedding.lm_weight`` reads
    ``euc_embed.weight`` every logits call — cache once, count its bytes honestly).
    """

    def __init__(self, codes: Tensor, scale: Tensor, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.num_embeddings, self.embedding_dim = codes.shape
        self._act_dtype = dtype
        self.register_buffer("codes", codes)      # int8 [V, d], values in [-31, 31]
        self.register_buffer("scale", scale)      # fp32 [V, 1]
        self._weight_cache: Tensor | None = None

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, bits: int = 6) -> "Int6RowEmbedding":
        hi = {8: 127, 6: 31}[bits]
        w = emb.weight.data.detach().float()
        s = w.abs().amax(dim=-1, keepdim=True).div(hi).clamp(min=1e-8)
        codes = (w / s).round().clamp(-hi, hi).to(torch.int8)
        return cls(codes, s, dtype=emb.weight.dtype)

    @property
    def weight(self) -> Tensor:
        if self._weight_cache is None:
            self._weight_cache = (
                self.codes.to(self._act_dtype) * self.scale.to(self._act_dtype)
            )
        return self._weight_cache

    def drop_weight_cache(self) -> None:
        self._weight_cache = None

    def forward(self, ids: Tensor) -> Tensor:
        return self.codes[ids].to(self._act_dtype) * self.scale[ids].to(self._act_dtype)

    def extra_repr(self) -> str:
        return f"V={self.num_embeddings}, d={self.embedding_dim}, int6-in-int8/row"


# ─────────────────────────────────────────────────────────────────────────────
# MORTAR carved-MLP packing
# ─────────────────────────────────────────────────────────────────────────────

def pack_mortar_ternary(cms: nn.Module, threshold: float = 0.5) -> dict:
    """Ternarize + 2-bit-pack a carved CMSBlockLinear's mortar_data IN PLACE.

    Quantization is the EXACT ``_mortar_effective_data`` per-tensor symmetric formula.
    The bf16 ``mortar_data`` parameter is deleted; ``_mortar_effective_data`` is
    rebound on the INSTANCE to dequantize from the packed buffer, so the unchanged
    ``_forward_mortar`` / stk BCSR kernel path consumes it transparently.
    """
    assert getattr(cms, "_mortar", False), "pack_mortar_ternary requires a carved layer"
    assert not getattr(cms, "_values_ternary_mode", False), \
        "pack bakes the ternary snap; values-ternary STE must be OFF"

    data = cms.mortar_data.data
    shape = tuple(data.shape)                       # [nnz, blk, blk]
    q, scale = _ternary_quantize(data, threshold)
    packed = pack_ternary_codes(q)

    del cms._parameters["mortar_data"]              # free the bf16 shadow
    cms.register_buffer("mortar_packed", packed)
    cms.register_buffer("mortar_scale", scale)
    cms._packed_shape = shape
    cms._packed_numel = shape[0] * shape[1] * shape[2]

    def _packed_effective_data(self) -> Tensor:
        w = unpack_ternary(self.mortar_packed, self._packed_numel, torch.bfloat16)
        return (w * self.mortar_scale.to(torch.bfloat16)).view(self._packed_shape)

    cms._mortar_effective_data = types.MethodType(_packed_effective_data, cms)
    nz = int(q.ne(0).sum().item())
    return {"nnz_blocks": shape[0], "packed_bytes": packed.numel(),
            "scale": float(scale), "nonzero_code_frac": nz / max(1, q.numel())}


# Training-only CMS buffer(s) to empty for inference. (The legacy topology buffers
# — score_history, col_indices, block_age, … — no longer exist on CMSBlockLinear.)
_BIG_CMS_BUFFERS = ("block_score_ema",)


def strip_cms_inference(cms: nn.Module) -> int:
    """Replace the (training-only) CMS scoring buffer(s) with empty tensors.

    Post-carve the saliency EMA is dead weight (block_score_ema is [R,K] fp32 —
    ~8 MB for gate_up at d=8192). Returns bytes freed. Attribute access stays
    valid (empty tensors, not deletion) so reprs/log paths can't crash.
    """
    freed = 0
    for name in _BIG_CMS_BUFFERS:
        buf = cms._buffers.get(name)
        if buf is not None and buf.numel() > 0:
            freed += buf.numel() * buf.element_size()
            cms._buffers[name] = torch.empty(0, dtype=buf.dtype, device=buf.device)
    cms._prune_mask = None
    cms._prune_elem_mask = None
    return freed


def shrink_mlp_to_mortar_ternary(
    bel: MortarLinear, target_density: float = 0.25, blocking: int = 128,
    threshold: float = 0.5, generator: torch.Generator | None = None,
) -> dict:
    """Full deploy-format pipeline for ONE MortarLinear, in place (on its device):

      random saliency → prune_step_blocks(→ target_density, 128-aligned)
      → carve(128) [real MORTAR BCSR] → ternary 2-bit pack → strip CMS buffers.

    Random saliency: weights are random in this systems test, so importance scores
    are meaningless — uniform-random scores give a representative *unstructured*
    kept-block pattern (with prune_step_blocks' ≥1-per-row floor), rather than the
    degenerate all-ties pattern of the zero-initialized EMA buffer.
    """
    cms = bel._cms
    with torch.no_grad():
        if generator is not None:
            cms.block_score_ema.copy_(
                torch.rand(cms.block_score_ema.shape, generator=generator,
                           device=cms.block_score_ema.device))
        else:
            cms.block_score_ema.uniform_()
    pr = cms.prune_step_blocks(prune_rate=1.0 - target_density,
                               target_density=target_density, blocking=blocking)
    nnz = cms.carve(blocking=blocking)
    info = pack_mortar_ternary(cms, threshold)
    freed = strip_cms_inference(cms)
    info.update(block_density=pr["density"], nnz=nnz, buffers_freed=freed)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Block / model orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _inner_mlp(block_mlp: nn.Module) -> nn.Module:
    """Unwrap _KwargSequential(mlp, Dropout) → the _SwiGLUMortar (or return as-is)."""
    if isinstance(block_mlp, nn.Sequential):
        return block_mlp[0]
    return block_mlp


def shrink_block(block: nn.Module, target_density: float = 0.25,
                 device: str | torch.device = "cuda",
                 attn_bits: int = 8, threshold: float = 0.5) -> dict:
    """Move ONE MORPHBlock to `device` and materialize it in deploy format, in place.

    MLP gate_up/down → prune(0.25)+carve(128)+ternary-2bit-pack; attention Linears →
    int8/row; norms/convs/HC/scalars stay bf16. Returns per-block stats (original
    param count by category + carve info). Used both post-construction (small gate)
    and from the streaming __init__ patch (30B build — keeps peak host RAM at ~one
    block instead of the whole 60 GB bf16 model).
    """
    # Tally ORIGINAL (pre-quantization) parameter counts — this is how the Stage B
    # build measures the exact 30B total without ever holding the full bf16 model.
    n_attn = sum(p.numel() for p in block.attention.parameters())
    n_mlp = sum(p.numel() for p in _inner_mlp(block.mlp).parameters())
    n_total = sum(p.numel() for p in block.parameters())

    block.to(device)
    mlp = _inner_mlp(block.mlp)
    carve_info = []
    for sub in (mlp.gate_up, mlp.down):
        assert isinstance(sub, MortarLinear), type(sub)
        carve_info.append(shrink_mlp_to_mortar_ternary(
            sub, target_density=target_density, threshold=threshold))
    n_lin = quantize_attention_linears(block.attention, bits=attn_bits)
    return {
        "params_total": n_total, "params_attn": n_attn, "params_mlp": n_mlp,
        "params_other": n_total - n_attn - n_mlp,
        "attn_linears_int8": n_lin, "carve": carve_info,
    }


def _pack_backbone_proj(proj: nn.Linear, threshold: float) -> PackedTernaryLinear:
    """Pack a backbone dense proj (x0/value ChannelInject.proj, LMHeadMixer.mix) to 2-bit ternary.

    CORRECTNESS: when the proj is TernarySTE-parametrized (the deploy config — scope=backbone
    covers these modules), ``from_linear`` would re-derive the scale from the already-ternarized
    STE *output* (mean(|γ·codes|) ≠ mean(|shadow|)) — a known scale bug. Read the smooth shadow
    via ``from_parametrized`` so the deployed scale is exact. Plain (non-parametrized) Linears
    fall back to ``from_linear``.
    """
    import torch.nn.utils.parametrize as parametrize
    if parametrize.is_parametrized(proj, "weight"):
        return PackedTernaryLinear.from_parametrized(proj)
    return PackedTernaryLinear.from_linear(proj, threshold)


def _proj_numel(proj: nn.Module) -> int:
    """Original weight numel of a (possibly TernarySTE-parametrized) Linear, read from the
    smooth shadow so the tally is correct whether or not the module is parametrized."""
    import torch.nn.utils.parametrize as parametrize
    if parametrize.is_parametrized(proj, "weight"):
        return proj.parametrizations.weight.original.numel()
    return proj.weight.numel()


def materialize_top_level(model: nn.Module, device: str | torch.device = "cuda",
                          embed_bits: int = 6, threshold: float = 0.5) -> dict:
    """Deploy-format the NON-block modules of a MORPHTransformer, in place.

    euc/bigram embeds → int6-in-int8 rows; x0/value ChannelInject projs + LMHeadMixer.mix
    → packed-2bit ternary (TernarySTE scope=backbone covers them); Lorentz space embed,
    value-embed tables, norms, LoopSSM → bf16 (deploy stack keeps them full precision).
    Finishes with model.to(device) for the bf16 remainder.

    Backbone-proj packing reads the TernarySTE shadow via ``from_parametrized`` when the proj
    is parametrized (the deploy case) — re-deriving the scale from the STE output is wrong.
    """
    import torch.nn.utils.parametrize as parametrize
    stats: dict = {"int6_embeds": [], "ternary_dense": []}
    hy = model.embed.hybrid
    n_emb = hy.euc_embed.weight.numel() + model.embed.bigram.embed.weight.numel()
    hy.euc_embed = Int6RowEmbedding.from_embedding(hy.euc_embed, bits=embed_bits)
    model.embed.bigram.embed = Int6RowEmbedding.from_embedding(
        model.embed.bigram.embed, bits=embed_bits)
    stats["int6_embeds"] = ["embed.hybrid.euc_embed", "embed.bigram.embed"]
    stats["params_embed_int6"] = n_emb

    # A backbone proj is either a plain nn.Linear or a TernarySTE-parametrized nn.Linear
    # (parametrize keeps the class as nn.Linear, so `type(...) is nn.Linear` stays True).
    def _is_packable_linear(m: nn.Module) -> bool:
        return type(m) is nn.Linear and (
            parametrize.is_parametrized(m, "weight") or getattr(m, "weight", None) is not None)

    n_tern = 0
    for inj in list(model.x0_injects) + list(model.value_embeds):
        if _is_packable_linear(inj.proj):
            n_tern += _proj_numel(inj.proj)
            inj.proj = _pack_backbone_proj(inj.proj, threshold)
            stats["ternary_dense"].append("ChannelInject.proj")
    if _is_packable_linear(model.lm_mixer.mix):
        n_tern += _proj_numel(model.lm_mixer.mix)
        model.lm_mixer.mix = _pack_backbone_proj(model.lm_mixer.mix, threshold)
        stats["ternary_dense"].append("lm_mixer.mix")
    stats["params_dense_ternary"] = n_tern

    model.to(device)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Attention + reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def quantize_attention_linears(module: nn.Module, bits: int = 8) -> int:
    """Replace every plain nn.Linear under `module` with Int8RowLinear (recursive).

    Apply to a MORPHAttention subtree → identical coverage to apply_attn_proj_quant
    (W_down_q/k, W_v_curr/prev, W_up, gate MLP, compressor, indexer). Convs/norms/
    scalar params untouched (stay bf16, exactly like the QAT config).
    """
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            setattr(module, name, Int8RowLinear.from_linear(child, bits=bits))
            n += 1
        else:
            n += quantize_attention_linears(child, bits=bits)
    return n


def to_deploy_inference(model: nn.Module, device: str | torch.device = "cuda",
                        attn_bits: int = 8, embed_bits: int = 6,
                        threshold: float = 0.5) -> dict:
    """Compose the full DEPLOY-QUANT inference build for an ALREADY-CARVED, ALREADY-TERNARY-QAT
    real trained checkpoint, IN PLACE. After this, the StaticDecodeEngine auto-detects
    ``deploy_quant=True`` (via the int8/row attention Linears) and uses the int8/2-bit/bf16
    fast kernels.

    Steps (the proven 310 tok/s recipe):
      1. ``materialize_top_level`` — euc/bigram embeds → int6/row; x0/value ChannelInject
         projs + lm_mixer.mix → packed-2bit ternary (from_parametrized when parametrized).
      2. ``quantize_attention_linears`` over every Attention subtree → int8/row.
      3. For each CARVED CMSBlockLinear (``_mortar`` True): disable values-ternary STE,
         ``pack_mortar_ternary`` (2-bit BCSR), ``strip_cms_inference`` (free training buffers).

    This is for the carved MLPs that ALREADY exist from ``load_checkpoint`` — it does NOT call
    ``shrink_block`` / ``shrink_mlp_to_mortar_ternary`` (those re-prune with random saliency,
    which would destroy a trained model's learned block pattern).

    Returns a stats dict with counts (attention int8 linears, mortar MLPs packed) and the
    resident-bytes report after quantization.
    """
    top = materialize_top_level(model, device=device, embed_bits=embed_bits, threshold=threshold)

    # Attention: int8/row over every MORPHAttention subtree.
    attn_int8 = 0
    n_attn_modules = 0
    for m in model.modules():
        if "Attention" in type(m).__name__:
            attn_int8 += quantize_attention_linears(m, bits=attn_bits)
            n_attn_modules += 1

    # Carved MLPs: pack mortar ternary (2-bit BCSR) + strip training buffers.
    mlps_packed = 0
    mortar_freed = 0
    for m in model.modules():
        if type(m).__name__ == "CMSBlockLinear" and getattr(m, "_mortar", False):
            # The packer bakes the ternary snap; the live values-ternary STE must be OFF.
            m._values_ternary_mode = False
            pack_mortar_ternary(m, threshold=threshold)
            mortar_freed += strip_cms_inference(m)
            mlps_packed += 1

    report = resident_bytes_report(model)
    return {
        "attn_modules": n_attn_modules,
        "attn_int8_linears": attn_int8,
        "mlps_packed": mlps_packed,
        "mortar_buffers_freed_bytes": mortar_freed,
        "top_level": top,
        "resident_bytes": report["total_bytes"],
        "resident_mb": report["total_bytes"] / (1024 ** 2),
        "resident_report": report,
    }


def resident_bytes_report(model: nn.Module) -> dict:
    """Measured bytes of every parameter+buffer, grouped by dtype and by component."""
    by_dtype: dict[str, int] = {}
    by_comp: dict[str, int] = {}
    total = 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        b = t.numel() * t.element_size()
        total += b
        by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + b
        if "mortar_packed" in name:
            comp = "mlp_packed_2bit"
        elif ".codes" in name or ".scale" in name or ".packed" in name:
            comp = "quant_codes_and_scales"
        elif name.startswith("embed.") or "value_embed_tables" in name:
            comp = "embeddings_bf16_remainder"
        else:
            comp = "bf16_remainder(norms/convs/HC/scalars)"
        by_comp[comp] = by_comp.get(comp, 0) + b
    return {"total_bytes": total, "by_dtype": by_dtype, "by_component": by_comp}
