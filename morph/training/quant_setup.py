"""Quantization / QAT setup, shared by the trainer and by anything that has to
REBUILD a trained model from a checkpoint.

This lives in its own module because it is load-bearing twice. The trainer applies
these transforms after building the model and before ``torch.compile``; every one of
them registers a ``torch.nn.utils.parametrize`` hook, which RENAMES the affected
tensors in ``state_dict`` (``w.weight`` becomes ``w.parametrizations.weight.original``).
So a checkpoint written by a QAT run can only be loaded into a model that has had the
same transforms applied, in the same order.

Sampling scripts learned this the hard way on 2026-08-18: loading the arm checkpoints
into a plain model reported 45 missing and 45 unexpected tensors, and with
``strict=False`` it would have silently sampled a half-initialised network. Copying the
block into the sampler would have fixed that run and drifted from the trainer by the
next config knob, so the block moved here instead and both callers use it.

ORDER IS LOAD-BEARING and is documented per-step below: ternary first (later steps
check disjointness against its module list), then embedding QAT, CMS scoring, attention
projection QAT, then FP8.
"""
from __future__ import annotations

from omegaconf import DictConfig
from torch import nn


def apply_quantization(model: nn.Module, cfg: DictConfig) -> dict:
    """Apply every configured QAT/quantization transform to ``model``, in place.

    Returns the manifests keyed ``ternary`` / ``embed_quant`` / ``attn_proj_quant`` /
    ``fp8``; each is ``None`` when that transform is off. The trainer logs them into
    the wandb config, so every run records exactly which transforms were live.

    Must be called BEFORE ``torch.compile`` (so the STE is captured in the compiled
    graph) and BEFORE ``create_optimizer`` (so the optimizer binds the smooth
    ``.original`` params).
    """
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
            scale_ema_beta=float(getattr(cfg.training, "ternary_scale_ema_beta", 0.0)),
        )
        _ema_b = float(getattr(cfg.training, "ternary_scale_ema_beta", 0.0))
        if _ema_b > 0.0:
            print(f"  TERNARY SCALE EMA ON: beta={_ema_b} — gamma advances once per "
                  f"optimizer step (cusp-vault fix); forward reads the buffer")
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

    return {
        "ternary": ternary_manifest,
        "embed_quant": embed_quant_manifest,
        "attn_proj_quant": attn_proj_quant_manifest,
        "fp8": fp8_manifest,
    }

