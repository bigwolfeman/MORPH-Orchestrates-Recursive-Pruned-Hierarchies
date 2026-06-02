"""FP8 training scope selection for MORPH (torchao float8, dynamic scaling).

Implements the design in docs/fp8_mixed_precision_training.md: convert ONLY the
large dense linear GEMMs to torchao `Float8Linear`, keep attention math / norms /
softmax / fused-CE / embeddings / the loop's injection in bf16 (the DeepSeek-V3 /
Ling-1T "mixed" recipe). Verified on RTX 5090 sm_120 (2.4–4.3× GEMM speedup).

Scopes (by Linear leaf name — name-based so `LMHeadMixer.mix` and the gate/indexer
Linears, which ARE nn.Linear, stay bf16):
  - "mlp"           : _SwiGLU.gate_up / .down (prelude+coda dense MLP). Core MLP is
                      BlockELLLinear (NOT nn.Linear) → auto-excluded; fixed-shape,
                      no loop/pruning interaction. Safest phase-1 surface.
  - "mlp_attn_proj" : + CCA projections W_down_q/k, W_v_curr/prev, W_up (these live in
                      the variable-M core loop → exercises the active-set/recompile path).
  - "all_gemm"      : == mlp_attn_proj here (the LM head is inside fused_linear_cross_entropy,
                      not an nn.Linear, so it can't be auto-converted; stays bf16).

Scaling is DYNAMIC (stateless, torchao default) — mandated for the looped core: the
6 core weights are reused ~6× per forward, and delayed/amax-history scaling would
conflate intra-step reuse into the cross-step history (corrupt scale). It also avoids
the truncated-BPTT no_grad iterations polluting a persistent amax buffer. See doc §7.
"""

from __future__ import annotations

import torch.nn as nn

# Linear leaf names eligible for FP8, grouped by category.
_MLP_LEAVES = {"gate_up", "down"}                                  # _SwiGLU dense MLP
_ATTN_PROJ_LEAVES = {"W_down_q", "W_down_k", "W_v_curr", "W_v_prev", "W_up"}  # CCA projections

FP8_SCOPES: dict[str, set[str]] = {
    "mlp": _MLP_LEAVES,
    "attn_proj": _ATTN_PROJ_LEAVES,           # attention projections ONLY — disjoint from
                                              # ternary_scope=backbone, for the 3-way combo
                                              # (MLP=ternary, attn-proj=FP8, 8-bit optimizer).
    "mlp_attn_proj": _MLP_LEAVES | _ATTN_PROJ_LEAVES,
    "all_gemm": _MLP_LEAVES | _ATTN_PROJ_LEAVES,
}


def build_fp8_filter(scope: str, min_dim: int):
    """Return a torchao module_filter_fn(module, fqn) -> bool.

    Converts a Linear iff its leaf name is in the scope's allow-set AND both weight
    dims >= min_dim (FP8 only helps large GEMMs). Everything else (attention kernels,
    norms, embeddings, Block-ELL core MLP, gate/indexer, LMHeadMixer.mix, fused-CE)
    is left bf16 by virtue of not matching.
    """
    if scope not in FP8_SCOPES:
        raise ValueError(f"unknown fp8_scope={scope!r}; choices={list(FP8_SCOPES)}")
    allow = FP8_SCOPES[scope]

    def module_filter_fn(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if fqn.split(".")[-1] not in allow:
            return False
        return min(module.weight.shape) >= min_dim

    return module_filter_fn


def _eligible_modules(model: nn.Module, scope: str, min_dim: int) -> list[str]:
    """FQNs of the Linears the filter would convert (for logging + disjointness check)."""
    filt = build_fp8_filter(scope, min_dim)
    return [name for name, m in model.named_modules() if filt(m, name)]


def apply_fp8_training(
    model: nn.Module,
    scope: str = "mlp",
    recipe: str = "dynamic",
    min_dim: int = 256,
    ternary_module_names: list[str] | None = None,
) -> dict:
    """Convert the scoped Linears to torchao Float8Linear (in place). Returns a manifest.

    Call AFTER model.to(device) and BEFORE torch.compile (so FP8 modules are compiled).
    `ternary_module_names`: FQNs already parametrized for ternary QAT — asserted disjoint
    from the FP8 set (a Linear cannot be both ternary and FP8; doc §6).
    """
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig

    if recipe == "dynamic":
        config = Float8LinearConfig()                      # torchao default = dynamic tensorwise
    elif recipe == "rowwise":
        config = Float8LinearConfig.from_recipe_name("rowwise")
    elif recipe == "delayed":
        raise ValueError(
            "fp8_recipe='delayed' is unsafe in MORPH's looped core (amax-history "
            "conflates intra-step weight reuse; doc §7.1/§7.2). Use 'dynamic'."
        )
    else:
        raise ValueError(f"unknown fp8_recipe={recipe!r}; choices=[dynamic, rowwise]")

    targets = _eligible_modules(model, scope, min_dim)

    # Disjointness with ternary QAT — a Linear cannot be both (doc §6).
    if ternary_module_names:
        tern = {n.split(" [")[0] for n in ternary_module_names}
        overlap = sorted(set(targets) & tern)
        if overlap:
            raise ValueError(
                f"fp8_scope and ternary_scope overlap on {len(overlap)} module(s): "
                f"{overlap[:5]}... — a Linear cannot be both FP8 and ternary. "
                "Choose disjoint scopes."
            )

    filt = build_fp8_filter(scope, min_dim)
    convert_to_float8_training(model, module_filter_fn=filt, config=config)

    return {
        "scope": scope,
        "recipe": recipe,
        "min_dim": min_dim,
        "n_converted": len(targets),
        "converted_modules": targets,
    }
