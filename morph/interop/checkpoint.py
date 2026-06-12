"""Bidirectional PyTorch ↔ JAX/Flax checkpoint converter for MORPH.

Handles the full MORPH parameter namespace as of the post-removal refactor
(neural memory + LeJEPA removed, Triton-fused attention active).

Verified PT → JAX path mapping (one representative example per param type):

  PT model path                               JAX path
  ─────────────────────────────────────────── ────────────────────────────────────────────────────────
  embed.hybrid.euc_embed.weight               embed/hybrid/euc_embed/embedding
  embed.hybrid.lor_embed.space_embed.weight   embed/hybrid/lor_embed/space_embed/embedding
  embed.bigram.embed.weight                   embed/bigram/embed/embedding
  embed.bigram.lambdas                        embed/bigram/lambdas

  prelude.0.norm_attn.weight                  prelude_0/norm_attn/scale
  prelude.0.norm_mlp.weight                   prelude_0/norm_mlp/scale
  prelude.0.mrr_attn.alpha_raw                prelude_0/mrr_attn/alpha_raw
  prelude.0.mrr_attn.gamma_raw                prelude_0/mrr_attn/gamma_raw
  prelude.0.mrr_mlp.alpha_raw                 prelude_0/mrr_mlp/alpha_raw
  prelude.0.mlp.gate_up.weight                prelude_0/mlp_module/gate_up/kernel
  prelude.0.mlp.down.weight                   prelude_0/mlp_module/down/kernel
  prelude.0.attention._impl.cca.W_down_q.weight     prelude_0/attn_module/cca_csa/cca_base/W_down_q/kernel
  prelude.0.attention._impl.cca.W_down_k.weight     prelude_0/attn_module/cca_csa/cca_base/W_down_k/kernel
  prelude.0.attention._impl.cca.W_up.weight         prelude_0/attn_module/cca_csa/cca_base/W_up/kernel
  prelude.0.attention._impl.cca.W_v_curr.weight     prelude_0/attn_module/cca_csa/cca_base/W_v_curr/kernel
  prelude.0.attention._impl.cca.W_v_prev.weight     prelude_0/attn_module/cca_csa/cca_base/W_v_prev/kernel
  prelude.0.attention._impl.cca.alpha               prelude_0/attn_module/cca_csa/cca_base/alpha
  prelude.0.attention._impl.cca.sink_logits         prelude_0/attn_module/cca_csa/sink_logits
  prelude.0.attention._impl.cca.temp                prelude_0/attn_module/cca_csa/cca_base/temp
  prelude.0.attention._impl.cca.q_norm.weight       prelude_0/attn_module/cca_csa/cca_base/q_norm/scale
  prelude.0.attention._impl.cca.k_norm.weight       prelude_0/attn_module/cca_csa/cca_base/k_norm/scale
  prelude.0.attention._impl.cca.gate.0.weight       prelude_0/attn_module/cca_csa/cca_base/gate_w1/kernel
  prelude.0.attention._impl.cca.gate.2.weight       prelude_0/attn_module/cca_csa/cca_base/gate_w2/kernel
  prelude.0.attention._impl.cca.conv_q_dw.weight    prelude_0/attn_module/cca_csa/cca_base/conv_q_dw_w
  prelude.0.attention._impl.cca.conv_q_gp.weight    prelude_0/attn_module/cca_csa/cca_base/conv_q_gp_w
  prelude.0.attention._impl.cca.conv_k_dw.weight    prelude_0/attn_module/cca_csa/cca_base/conv_k_dw_w
  prelude.0.attention._impl.cca.conv_k_gp.weight    prelude_0/attn_module/cca_csa/cca_base/conv_k_gp_w
  prelude.0.attention._impl.comp_norm.weight         prelude_0/attn_module/cca_csa/comp_norm/scale
  prelude.0.attention._impl.compressor.W_aKV.weight  prelude_0/attn_module/cca_csa/compressor/W_aKV/kernel
  prelude.0.attention._impl.compressor.B_a           prelude_0/attn_module/cca_csa/compressor/B_a
  prelude.0.attention._impl.indexer.W_IQ.weight      prelude_0/attn_module/cca_csa/indexer/W_IQ/kernel
  prelude.0.attention._impl.indexer.compressor.*     prelude_0/attn_module/cca_csa/indexer/indexer_compressor/*

  core.N.*                                    core_N/*   (with cca_csa or cca_hca depending on layer_idx % 2)
  coda.N.*                                    coda_N/*

  x0_injects.N.log_scale                     x0_inject_N/log_scale
  x0_injects.N.proj.weight                   x0_inject_N/proj/kernel
  value_embeds.N.log_scale                   value_embed_N/log_scale
  value_embeds.N.proj.weight                 value_embed_N/proj/kernel
  value_embed_tables.N.weight                value_embed_table_N/embedding

  injection.log_A                             injection/log_A
  injection.log_dt                            injection/log_dt
  input_norm.weight                           input_norm/scale
  final_norm.weight                           final_norm/scale
  lm_mixer.channel_scales                    lm_mixer/channel_scales
  lm_mixer.mix.weight                        lm_mixer/mix/kernel

  stp.*                                       (no parameters — zero-param module)

  MortarLinear / _cms.*                       (pruning state — skipped, no JAX equivalent)

Key differences from older converter:
  - PT MORPHBlock: `block.norm_attn`, `block.attention`, `block.mlp`
    → JAX MORPHTransformerBlock: `block/norm_attn`, `block/attn_module`, `block/mlp_module`
  - PT _CCABase: stored as `attention._impl.cca.*`
    → JAX _CCABase: stored as `attn_module/cca_csa/cca_base/*` or `cca_hca/cca_base/*`
  - PT conv layers (nn.Conv1d): `conv_q_dw.weight` [C, 1, k]
    → JAX raw params: `conv_q_dw_w` [C, 1, k] — no transpose needed (3D conv kernel)
  - PT gate Sequential: `gate.0.weight`, `gate.2.weight`
    → JAX named Dense: `gate_w1/kernel`, `gate_w2/kernel`
  - PT RMSNorm: `.weight` → JAX RMSNorm: `/scale`
  - PT sink_logits is in cca (base), JAX has it one level up (in cca_csa/cca_hca)
  - MRR norms: PT `block.norm_attn` is standalone, JAX `block/norm_attn` is also standalone
    (NOT nested inside mrr_attn — they share the same block dict level)

Usage:
    python -m morph.interop.checkpoint \\
        --from pt --to jax \\
        --input ckpt.pt --output params.msgpack
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


# ── Skip list: PT state-dict entries that have no JAX equivalent ──────────────

# These are either non-parameter buffers (RoPE cache, CoPE freqs) or
# Legacy CMS topology state (col_indices, score_history, etc. — kept in the skip
# list so OLD checkpoints that still carry these keys convert cleanly)
_PT_SKIP_SUBSTRINGS = [
    "freqs", "momentum_S", "_param_offsets",
    # CMS state (pruning topology — not model params):
    "_cms.col_indices", "_cms.block_age", "_cms.block_score_ema",
    "_cms.block_score_historical_ema", "_cms.score_history",
    "_cms.crystallized_mask", "_cms.col_usage_count", "_cms.swap_count",
    "_cms.last_swap_step", "_cms.error_norm_acc", "_cms.activation_norm_acc",
    "_cms._score_snapshot",
    # Topology global counters (model-level state, not per-layer params):
    "_acc_steps", "_score_history_idx", "_swap_rate_history", "_topology_step_count",
]

# PT param names with these substrings are CMS weight tensors (actual params, keep):
_CMS_WEIGHT_SUFFIX = "_cms.weight"


# ── Embedding parent names (weight → embedding in JAX) ────────────────────────

_EMBEDDING_PARENTS = {
    "euc_embed", "lor_embed", "space_embed", "embed",
    "value_embed_table_0", "value_embed_table_1", "value_embed_table_2",
}


# ── Leaf names that must NOT be transposed ────────────────────────────────────

_NO_TRANSPOSE_LEAVES = {
    # Scalars / 1D:
    "scale", "embedding", "lambdas", "log_A", "log_dt",
    "alpha_raw", "gamma_raw", "log_scale", "channel_scales",
    "sink_logits", "alpha", "temp", "B_a", "B_b",
    # 3D conv kernels: [C, groups_factor, k] — no transpose
    "conv_q_dw_w", "conv_q_gp_w", "conv_k_dw_w", "conv_k_gp_w",
}


# ── Utility: nested dict helpers ──────────────────────────────────────────────

def _set_nested(d: dict, path: list[str], value: Any):
    for p in path[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[path[-1]] = value


def _get_nested(d: dict, path: list[str]) -> Any | None:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _flatten_dict(d: dict, prefix: list[str] | None = None) -> list[tuple[list[str], np.ndarray]]:
    """Recursively flatten nested dict to [(path_list, array), ...] pairs."""
    out = []
    prefix = prefix or []
    for k, v in d.items():
        cur_path = prefix + [k]
        if isinstance(v, dict):
            out.extend(_flatten_dict(v, cur_path))
        else:
            # Accept numpy arrays or jax arrays
            try:
                arr = np.asarray(v)
                out.append((cur_path, arr))
            except Exception:
                pass
    return out


# ── PT → JAX name conversion ─────────────────────────────────────────────────

def _pt_name_to_jax_path(pt_name: str, layer_indices: dict[str, int] | None = None
                          ) -> tuple[list[str], bool] | None:
    """Convert a PyTorch param name to a JAX nested path list + transpose flag.

    Returns:
        (path, needs_transpose) or None if this param should be skipped.
        path is the full nested key list including leaf.
        needs_transpose is True for 2D Dense kernels [out, in] → [in, out].

    Args:
        pt_name: dot-separated PT parameter name.
        layer_indices: optional dict mapping block_prefix (e.g. "prelude_0") to
            the global layer_idx. Used to determine cca_csa vs cca_hca.
            If None, the layer type is inferred from the layer position
            (even → CSA, odd → HCA matching the default alternation).
    """
    parts = pt_name.split(".")

    # ── 1. Skip non-parameter tensors ─────────────────────────────────────────
    for skip_sub in _PT_SKIP_SUBSTRINGS:
        if skip_sub in pt_name:
            return None

    # ── 2. Parse block prefix and remainder ───────────────────────────────────
    # Examples:
    #   "prelude.0.norm_attn.weight" → prefix=prelude_0, rest=["norm_attn","weight"]
    #   "core.1.attention._impl.cca.W_down_q.weight" → prefix=core_1, rest=[...]
    #   "embed.bigram.embed.weight" → no block prefix

    jax_path: list[str] = []
    i = 0

    # ── Top-level block groups ─────────────────────────────────────────────────
    # prelude/core/coda: "prelude.N." → "prelude_N"
    # x0_injects: "x0_injects.N." → "x0_inject_N"
    # value_embeds: "value_embeds.N." → "value_embed_N"
    # value_embed_tables: "value_embed_tables.N." → "value_embed_table_N"

    def _take_indexed_module():
        """Consume 'module.N' → append 'module_N', return True if done."""
        nonlocal i
        name = parts[i]
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            jax_path.append(f"{name}_{parts[i+1]}")
            i += 2
            return True
        return False

    p0 = parts[0]

    # ── Embed ──────────────────────────────────────────────────────────────────
    if p0 == "embed":
        # embed.hybrid.euc_embed.weight → embed/hybrid/euc_embed/embedding
        # embed.bigram.embed.weight     → embed/bigram/embed/embedding
        # embed.bigram.lambdas          → embed/bigram/lambdas
        rest = parts[1:]  # ["hybrid", "euc_embed", "weight"] etc.
        jax_path.append("embed")
        for j, rp in enumerate(rest):
            if rp == "weight":
                parent = rest[j - 1] if j > 0 else ""
                if parent in _EMBEDDING_PARENTS:
                    jax_path.append("embedding")
                else:
                    jax_path.append("kernel")
                    return jax_path, True
            else:
                jax_path.append(rp)
        return jax_path, False

    # ── Scalar non-block params ────────────────────────────────────────────────
    if p0 in ("injection", "stp"):
        # injection.log_A / injection.log_dt — direct map
        # stp.* — no params
        if p0 == "stp":
            return None
        jax_path.extend(parts)
        return jax_path, False

    if p0 in ("input_norm", "final_norm"):
        # input_norm.weight → input_norm/scale
        jax_path.append(p0)
        leaf = parts[-1]
        if leaf == "weight":
            jax_path.append("scale")
            return jax_path, False
        jax_path.append(leaf)
        return jax_path, False

    if p0 == "lm_mixer":
        # lm_mixer.channel_scales → lm_mixer/channel_scales
        # lm_mixer.mix.weight     → lm_mixer/mix/kernel
        jax_path.append("lm_mixer")
        rest = parts[1:]
        leaf = rest[-1]
        for rp in rest[:-1]:
            jax_path.append(rp)
        if leaf == "weight":
            jax_path.append("kernel")
            return jax_path, True
        jax_path.append(leaf)
        return jax_path, False

    # ── x0_injects / value_embeds / value_embed_tables ───────────────────────
    if p0 == "x0_injects":
        # x0_injects.N.log_scale     → x0_inject_N/log_scale
        # x0_injects.N.proj.weight   → x0_inject_N/proj/kernel
        idx = parts[1]
        jax_path.append(f"x0_inject_{idx}")
        rest = parts[2:]
        return _map_channel_inject_rest(jax_path, rest)

    if p0 == "value_embeds":
        idx = parts[1]
        jax_path.append(f"value_embed_{idx}")
        rest = parts[2:]
        return _map_channel_inject_rest(jax_path, rest)

    if p0 == "value_embed_tables":
        idx = parts[1]
        jax_path.append(f"value_embed_table_{idx}")
        leaf = parts[-1]
        if leaf == "weight":
            jax_path.append("embedding")
            return jax_path, False
        jax_path.append(leaf)
        return jax_path, False

    # ── Block groups: prelude.N / core.N / coda.N ────────────────────────────
    if p0 in ("prelude", "core", "coda"):
        block_idx = int(parts[1])
        block_prefix = f"{p0}_{block_idx}"
        jax_path.append(block_prefix)

        # Compute global layer_idx for CSA/HCA alternation.
        # This requires knowing n_prelude / n_core from config.
        # We use layer_indices if provided; otherwise caller passes it.
        if layer_indices is not None:
            global_layer_idx = layer_indices.get(block_prefix, 0)
        else:
            # Default: can't determine without config — use block_idx within section
            # (wrong for core/coda but converter will accept override via layer_indices)
            global_layer_idx = block_idx

        cca_branch = "cca_csa" if global_layer_idx % 2 == 0 else "cca_hca"

        rest = parts[2:]  # e.g. ["norm_attn","weight"] or ["attention","_impl","cca",...]
        return _map_block_rest(jax_path, rest, cca_branch)

    # Fallthrough: unknown param
    # Pass through as-is (best-effort)
    return parts, False


def _map_channel_inject_rest(jax_path: list[str], rest: list[str]) -> tuple[list[str], bool]:
    """Map remainder of x0_injects.N.* or value_embeds.N.* to JAX path."""
    # log_scale → log_scale (scalar)
    # proj.weight → proj/kernel (Dense, needs transpose)
    if not rest:
        return jax_path, False
    leaf = rest[-1]
    interior = rest[:-1]
    for rp in interior:
        jax_path.append(rp)
    if leaf == "weight":
        jax_path.append("kernel")
        return jax_path, True
    jax_path.append(leaf)
    return jax_path, False


def _map_block_rest(jax_path: list[str], rest: list[str], cca_branch: str
                    ) -> tuple[list[str], bool]:
    """Map the sub-path within a prelude/core/coda block.

    rest examples:
      ["norm_attn", "weight"]
      ["norm_mlp", "weight"]
      ["mrr_attn", "alpha_raw"]
      ["mrr_mlp", "gamma_raw"]
      ["mlp", "gate_up", "weight"]     (prelude/coda plain SwiGLU)
      ["mlp", "down", "weight"]
      ["mlp", "gate_up", "_cms", "weight"]  (core CMS weight)
      ["mlp", "down", "_cms", "weight"]
      ["attention", "_impl", "cca", "W_down_q", "weight"]
      ["attention", "_impl", "cca", "sink_logits"]
      ["attention", "_impl", "cca", "conv_q_dw", "weight"]
      ["attention", "_impl", "cca", "gate", "0", "weight"]
      ["attention", "_impl", "comp_norm", "weight"]
      ["attention", "_impl", "compressor", "B_a"]
      ["attention", "_impl", "indexer", "W_IQ", "weight"]
      ["attention", "_impl", "indexer", "compressor", "B_a"]
    """
    if not rest:
        return jax_path, False

    r0 = rest[0]

    # ── Norms and MRR (standalone at block level) ──────────────────────────────
    if r0 in ("norm_attn", "norm_mlp"):
        jax_path.append(r0)
        leaf = rest[-1]
        if leaf == "weight":
            jax_path.append("scale")
            return jax_path, False
        jax_path.append(leaf)
        return jax_path, False

    if r0 in ("mrr_attn", "mrr_mlp"):
        jax_path.append(r0)
        leaf = rest[-1]
        jax_path.append(leaf)  # alpha_raw / gamma_raw — no transform
        return jax_path, False

    # ── MLP (SwiGLU, possibly CMS-wrapped) ───────────────────────────────────
    if r0 == "mlp":
        jax_path.append("mlp_module")
        # rest[1:] = e.g. ["gate_up", "weight"] or ["gate_up", "_cms", "weight"]
        # MortarLinear stores the actual dense weight at "._cms.weight".
        # Both forms map to the same JAX Dense kernel (plain SwiGLU or post-compact).
        mlp_rest = rest[1:]
        if len(mlp_rest) == 0:
            return jax_path, False
        layer_name = mlp_rest[0]  # "gate_up" or "down"
        jax_path.append(layer_name)
        # Detect CMS wrapper: ["gate_up", "_cms", "weight"] → kernel
        if len(mlp_rest) >= 3 and mlp_rest[1] == "_cms" and mlp_rest[2] == "weight":
            jax_path.append("kernel")
            return jax_path, True
        # Plain SwiGLU: ["gate_up", "weight"] → kernel
        leaf = mlp_rest[-1]
        if leaf == "weight":
            jax_path.append("kernel")
            return jax_path, True
        jax_path.append(leaf)
        return jax_path, False

    # ── Attention ─────────────────────────────────────────────────────────────
    if r0 == "attention":
        # rest[1] == "_impl", rest[2:] is the impl sub-path
        jax_path.append("attn_module")
        jax_path.append(cca_branch)
        impl_rest = rest[2:]  # drop "attention", "_impl"
        return _map_impl_rest(jax_path, impl_rest)

    # Fallthrough
    for rp in rest:
        jax_path.append(rp)
    return jax_path, False


def _map_impl_rest(jax_path: list[str], impl_rest: list[str]) -> tuple[list[str], bool]:
    """Map the sub-path within _impl (after stripping 'attention._impl').

    impl_rest examples starting from after "_impl":
      ["cca", "W_down_q", "weight"]
      ["cca", "alpha"]
      ["cca", "sink_logits"]           → one level up (JAX: cca_csa/sink_logits)
      ["cca", "temp"]
      ["cca", "conv_q_dw", "weight"]
      ["cca", "gate", "0", "weight"]
      ["cca", "q_norm", "weight"]
      ["comp_norm", "weight"]
      ["compressor", "B_a"]
      ["compressor", "W_aKV", "weight"]
      ["indexer", "W_IQ", "weight"]
      ["indexer", "compressor", "B_a"]
    """
    if not impl_rest:
        return jax_path, False

    r0 = impl_rest[0]

    # ── CCA base ──────────────────────────────────────────────────────────────
    if r0 == "cca":
        cca_rest = impl_rest[1:]  # sub-path within the CCA base
        return _map_cca_base(jax_path, cca_rest)

    # ── comp_norm (RMSNorm on compressed KV) ─────────────────────────────────
    if r0 == "comp_norm":
        jax_path.append("comp_norm")
        leaf = impl_rest[-1]
        if leaf == "weight":
            jax_path.append("scale")
            return jax_path, False
        jax_path.append(leaf)
        return jax_path, False

    # ── compressor (GatedPoolCompressor) ──────────────────────────────────────
    if r0 == "compressor":
        jax_path.append("compressor")
        comp_rest = impl_rest[1:]
        return _map_gated_pool_rest(jax_path, comp_rest)

    # ── indexer (LightningIndexer) ────────────────────────────────────────────
    if r0 == "indexer":
        jax_path.append("indexer")
        idx_rest = impl_rest[1:]
        return _map_indexer_rest(jax_path, idx_rest)

    # Fallthrough
    for rp in impl_rest:
        jax_path.append(rp)
    return jax_path, False


def _map_cca_base(jax_path: list[str], cca_rest: list[str]) -> tuple[list[str], bool]:
    """Map sub-path within _CCABase.

    In JAX, most CCA params are under cca_base/ — EXCEPT sink_logits which is
    one level up (under cca_csa or cca_hca directly).

    cca_rest examples:
      ["W_down_q", "weight"]      → cca_base/W_down_q/kernel  (T)
      ["alpha"]                   → cca_base/alpha
      ["sink_logits"]             → sink_logits (NOT in cca_base!)
      ["temp"]                    → cca_base/temp
      ["q_norm", "weight"]        → cca_base/q_norm/scale
      ["k_norm", "weight"]        → cca_base/k_norm/scale
      ["gate", "0", "weight"]     → cca_base/gate_w1/kernel  (T)
      ["gate", "2", "weight"]     → cca_base/gate_w2/kernel  (T)
      ["conv_q_dw", "weight"]     → cca_base/conv_q_dw_w  (NO transpose, 3D kernel)
      ["conv_q_gp", "weight"]     → cca_base/conv_q_gp_w
      ["conv_k_dw", "weight"]     → cca_base/conv_k_dw_w
      ["conv_k_gp", "weight"]     → cca_base/conv_k_gp_w
      ["W_v_curr", "weight"]      → cca_base/W_v_curr/kernel  (T)
      ["W_v_prev", "weight"]      → cca_base/W_v_prev/kernel  (T)
      ["W_up", "weight"]          → cca_base/W_up/kernel  (T)
    """
    if not cca_rest:
        return jax_path, False

    r0 = cca_rest[0]

    # sink_logits lives one level up (not inside cca_base) in JAX
    if r0 == "sink_logits":
        jax_path.append("sink_logits")
        return jax_path, False

    # Everything else goes into cca_base/
    jax_path.append("cca_base")

    # Norms
    if r0 in ("q_norm", "k_norm"):
        jax_path.append(r0)
        leaf = cca_rest[-1]
        if leaf == "weight":
            jax_path.append("scale")
            return jax_path, False
        jax_path.append(leaf)
        return jax_path, False

    # Gate Sequential: gate.0.weight → gate_w1/kernel, gate.2.weight → gate_w2/kernel
    if r0 == "gate":
        gate_idx = cca_rest[1] if len(cca_rest) > 1 else "0"
        gate_name = "gate_w1" if gate_idx == "0" else "gate_w2"
        jax_path.append(gate_name)
        leaf = cca_rest[-1]
        if leaf == "weight":
            jax_path.append("kernel")
            return jax_path, True
        jax_path.append(leaf)
        return jax_path, False

    # Conv kernels: conv_{q,k}_{dw,gp}.weight → conv_{q,k}_{dw,gp}_w  (3D, no T)
    if r0 in ("conv_q_dw", "conv_q_gp", "conv_k_dw", "conv_k_gp"):
        jax_path.append(f"{r0}_w")
        # No "kernel" sub-key and no transpose — this is a raw param, not nn.Dense
        return jax_path, False

    # alpha, temp: scalars
    if r0 in ("alpha", "temp"):
        jax_path.append(r0)
        return jax_path, False

    # Dense layers: W_down_q, W_down_k, W_up, W_v_curr, W_v_prev
    if r0 in ("W_down_q", "W_down_k", "W_up", "W_v_curr", "W_v_prev", "W_down"):
        jax_path.append(r0)
        leaf = cca_rest[-1]
        if leaf == "weight":
            jax_path.append("kernel")
            return jax_path, True
        jax_path.append(leaf)
        return jax_path, False

    # Fallthrough
    for rp in cca_rest:
        jax_path.append(rp)
    return jax_path, False


def _map_gated_pool_rest(jax_path: list[str], comp_rest: list[str]) -> tuple[list[str], bool]:
    """Map GatedPoolCompressor sub-path.

    comp_rest examples:
      ["B_a"]                  → B_a (no transpose)
      ["B_b"]                  → B_b
      ["W_aKV", "weight"]      → W_aKV/kernel  (T)
      ["W_aZ", "weight"]       → W_aZ/kernel   (T)
      ["W_bKV", "weight"]      → W_bKV/kernel  (T)
      ["W_bZ", "weight"]       → W_bZ/kernel   (T)
    """
    if not comp_rest:
        return jax_path, False
    r0 = comp_rest[0]
    leaf = comp_rest[-1]
    if r0 in ("B_a", "B_b"):
        jax_path.append(r0)
        return jax_path, False
    # Dense projections
    jax_path.append(r0)
    if leaf == "weight":
        jax_path.append("kernel")
        return jax_path, True
    jax_path.append(leaf)
    return jax_path, False


def _map_indexer_rest(jax_path: list[str], idx_rest: list[str]) -> tuple[list[str], bool]:
    """Map LightningIndexer sub-path.

    idx_rest examples:
      ["W_IQ", "weight"]               → W_IQ/kernel  (T)
      ["compressor", "B_a"]            → indexer_compressor/B_a
      ["compressor", "W_aKV", "weight"] → indexer_compressor/W_aKV/kernel  (T)
      ["compressor", "W_aZ", "weight"]  → indexer_compressor/W_aZ/kernel   (T)
    """
    if not idx_rest:
        return jax_path, False
    r0 = idx_rest[0]

    if r0 == "W_IQ":
        jax_path.append("W_IQ")
        leaf = idx_rest[-1]
        if leaf == "weight":
            jax_path.append("kernel")
            return jax_path, True
        jax_path.append(leaf)
        return jax_path, False

    if r0 == "compressor":
        # In JAX, the LightningIndexer's internal compressor is named "indexer_compressor"
        jax_path.append("indexer_compressor")
        return _map_gated_pool_rest(jax_path, idx_rest[1:])

    # Fallthrough
    for rp in idx_rest:
        jax_path.append(rp)
    return jax_path, False


# ── JAX → PT path conversion ─────────────────────────────────────────────────

def _jax_path_to_pt_name(jax_path: list[str],
                          layer_indices: dict[str, int] | None = None) -> str:
    """Reverse-map a JAX path list to a PT param name.

    This is more complex than the forward direction because we need to
    recover the exact PT module hierarchy from the JAX flat naming.
    """
    # Reconstruct by calling _pt_name_to_jax_path backwards is hard;
    # instead we rebuild the PT name from the JAX path directly.

    if not jax_path:
        return ""

    p0 = jax_path[0]

    # ── embed ──────────────────────────────────────────────────────────────────
    if p0 == "embed":
        rest = jax_path[1:]
        pt_parts = ["embed"]
        for j, rp in enumerate(rest):
            if rp == "embedding":
                pt_parts.append("weight")
            elif rp == "kernel":
                pt_parts.append("weight")
            else:
                pt_parts.append(rp)
        return ".".join(pt_parts)

    # ── injection ─────────────────────────────────────────────────────────────
    if p0 == "injection":
        return ".".join(jax_path)

    # ── input_norm / final_norm ───────────────────────────────────────────────
    if p0 in ("input_norm", "final_norm"):
        leaf = jax_path[-1]
        if leaf == "scale":
            return f"{p0}.weight"
        return ".".join(jax_path)

    # ── lm_mixer ─────────────────────────────────────────────────────────────
    if p0 == "lm_mixer":
        if jax_path[-1] == "kernel":
            inner = ".".join(jax_path[1:-1])
            return f"lm_mixer.{inner}.weight"
        return ".".join(jax_path)

    # ── x0_inject_N ──────────────────────────────────────────────────────────
    if p0.startswith("x0_inject_") and p0[len("x0_inject_"):].isdigit():
        idx = p0[len("x0_inject_"):]
        rest = jax_path[1:]
        leaf = rest[-1]
        interior = rest[:-1]
        prefix = f"x0_injects.{idx}"
        if leaf == "kernel":
            return f"{prefix}.{'.' .join(interior)}.weight"
        return f"{prefix}.{'.'.join(rest)}"

    # ── value_embed_N ─────────────────────────────────────────────────────────
    if p0.startswith("value_embed_") and not p0.startswith("value_embed_table_"):
        suffix = p0[len("value_embed_"):]
        if suffix.isdigit():
            idx = suffix
            rest = jax_path[1:]
            leaf = rest[-1]
            interior = rest[:-1]
            prefix = f"value_embeds.{idx}"
            if leaf == "kernel":
                return f"{prefix}.{'.'.join(interior)}.weight"
            return f"{prefix}.{'.'.join(rest)}"

    # ── value_embed_table_N ───────────────────────────────────────────────────
    if p0.startswith("value_embed_table_"):
        suffix = p0[len("value_embed_table_"):]
        if suffix.isdigit():
            idx = suffix
            leaf = jax_path[-1]
            if leaf == "embedding":
                return f"value_embed_tables.{idx}.weight"
            return f"value_embed_tables.{idx}.{'.'.join(jax_path[1:])}"

    # ── prelude_N / core_N / coda_N ──────────────────────────────────────────
    for prefix in ("prelude_", "core_", "coda_"):
        if p0.startswith(prefix) and p0[len(prefix):].isdigit():
            block_type = prefix.rstrip("_")
            block_idx = p0[len(prefix):]
            rest = jax_path[1:]
            return _jax_block_to_pt(block_type, block_idx, rest)

    # Fallthrough: join with dots, swap leaf names
    pt_parts = []
    for rp in jax_path:
        if rp == "kernel":
            pt_parts.append("weight")
        elif rp == "scale":
            pt_parts.append("weight")
        elif rp == "embedding":
            pt_parts.append("weight")
        else:
            pt_parts.append(rp)
    return ".".join(pt_parts)


def _jax_block_to_pt(block_type: str, block_idx: str, rest: list[str]) -> str:
    """Reconstruct PT param name for a block (prelude/core/coda).

    rest is the JAX sub-path after the block prefix, e.g.:
      ["norm_attn", "scale"]
      ["mrr_attn", "alpha_raw"]
      ["mlp_module", "gate_up", "kernel"]
      ["attn_module", "cca_csa", "cca_base", "W_down_q", "kernel"]
      ["attn_module", "cca_csa", "sink_logits"]
      ["attn_module", "cca_csa", "cca_base", "conv_q_dw_w"]
      ["attn_module", "cca_csa", "cca_base", "gate_w1", "kernel"]
    """
    prefix = f"{block_type}.{block_idx}"
    if not rest:
        return prefix

    r0 = rest[0]

    if r0 in ("norm_attn", "norm_mlp"):
        leaf = rest[-1]
        if leaf == "scale":
            return f"{prefix}.{r0}.weight"
        return f"{prefix}.{r0}.{'.'.join(rest[1:])}"

    if r0 in ("mrr_attn", "mrr_mlp"):
        return f"{prefix}.{r0}.{rest[1]}"

    if r0 == "mlp_module":
        inner = rest[1:]
        leaf = inner[-1]
        mid = inner[:-1]
        mid_str = ".".join(mid)
        if leaf == "kernel":
            return f"{prefix}.mlp.{mid_str}.weight" if mid_str else f"{prefix}.mlp.weight"
        return f"{prefix}.mlp.{'.'.join(inner)}"

    if r0 == "attn_module":
        # rest[1] = "cca_csa" or "cca_hca"
        # rest[2:] = ["cca_base", ...] or ["sink_logits"] or ["comp_norm","scale"] etc.
        impl_rest = rest[2:]  # drop "attn_module" and "cca_csa"/"cca_hca"
        return _jax_impl_to_pt(prefix, impl_rest)

    # Fallthrough
    pt_parts = [prefix]
    for rp in rest:
        if rp == "kernel":
            pt_parts.append("weight")
        elif rp == "scale":
            pt_parts.append("weight")
        else:
            pt_parts.append(rp)
    return ".".join(pt_parts)


def _jax_impl_to_pt(prefix: str, impl_rest: list[str]) -> str:
    """Reconstruct PT path from JAX path after stripping attn_module/cca_*/."""
    if not impl_rest:
        return f"{prefix}.attention"

    r0 = impl_rest[0]

    if r0 == "cca_base":
        # impl_rest[1:] = cca params
        return _jax_cca_base_to_pt(prefix, impl_rest[1:])

    if r0 == "sink_logits":
        return f"{prefix}.attention._impl.cca.sink_logits"

    if r0 == "comp_norm":
        leaf = impl_rest[-1]
        if leaf == "scale":
            return f"{prefix}.attention._impl.comp_norm.weight"
        return f"{prefix}.attention._impl.comp_norm.{leaf}"

    if r0 == "compressor":
        comp_rest = impl_rest[1:]
        return _jax_compressor_to_pt(f"{prefix}.attention._impl.compressor", comp_rest)

    if r0 == "indexer":
        idx_rest = impl_rest[1:]
        return _jax_indexer_to_pt(f"{prefix}.attention._impl.indexer", idx_rest)

    # Fallthrough
    return f"{prefix}.attention._impl.{'.' .join(impl_rest)}"


def _jax_cca_base_to_pt(prefix: str, cca_rest: list[str]) -> str:
    """Map JAX cca_base/* back to PT _impl.cca.*"""
    if not cca_rest:
        return f"{prefix}.attention._impl.cca"
    r0 = cca_rest[0]
    base = f"{prefix}.attention._impl.cca"

    if r0 in ("q_norm", "k_norm"):
        leaf = cca_rest[-1]
        if leaf == "scale":
            return f"{base}.{r0}.weight"
        return f"{base}.{r0}.{leaf}"

    if r0 in ("gate_w1", "gate_w2"):
        gate_idx = "0" if r0 == "gate_w1" else "2"
        leaf = cca_rest[-1]
        if leaf == "kernel":
            return f"{base}.gate.{gate_idx}.weight"
        return f"{base}.gate.{gate_idx}.{leaf}"

    # conv params: conv_q_dw_w → conv_q_dw.weight
    if r0 in ("conv_q_dw_w", "conv_q_gp_w", "conv_k_dw_w", "conv_k_gp_w"):
        conv_name = r0[:-2]  # strip "_w"
        return f"{base}.{conv_name}.weight"

    if r0 in ("alpha", "temp", "sink_logits"):
        return f"{base}.{r0}"

    # Dense: W_down_q, W_up, etc.
    leaf = cca_rest[-1]
    mid = cca_rest[:-1]
    if leaf == "kernel":
        return f"{base}.{'.'.join(mid)}.weight"
    return f"{base}.{'.'.join(cca_rest)}"


def _jax_compressor_to_pt(base: str, comp_rest: list[str]) -> str:
    if not comp_rest:
        return base
    r0 = comp_rest[0]
    if r0 in ("B_a", "B_b"):
        return f"{base}.{r0}"
    leaf = comp_rest[-1]
    mid = comp_rest[:-1]
    if leaf == "kernel":
        return f"{base}.{'.'.join(mid)}.weight"
    return f"{base}.{'.'.join(comp_rest)}"


def _jax_indexer_to_pt(base: str, idx_rest: list[str]) -> str:
    if not idx_rest:
        return base
    r0 = idx_rest[0]
    if r0 == "W_IQ":
        leaf = idx_rest[-1]
        if leaf == "kernel":
            return f"{base}.W_IQ.weight"
        return f"{base}.W_IQ.{leaf}"
    if r0 == "indexer_compressor":
        return _jax_compressor_to_pt(f"{base}.compressor", idx_rest[1:])
    # Fallthrough
    return f"{base}.{'.'.join(idx_rest)}"


def _needs_transpose(jax_path: list[str]) -> bool:
    """True iff this JAX param needs to be transposed when going back to PT."""
    leaf = jax_path[-1]
    if leaf != "kernel":
        return False
    # Conv kernels are stored as raw params (not /kernel) in JAX, so they never
    # appear here. The only "kernel" leaves are Dense layers.
    return True


# ── pt_to_jax ─────────────────────────────────────────────────────────────────

def pt_to_jax(pt_state_dict: dict,
              layer_indices: dict[str, int] | None = None) -> dict:
    """Convert PyTorch model state dict to JAX/Flax nested param dict.

    Args:
        pt_state_dict: dict mapping PT param name → torch.Tensor (or numpy).
                       Accepts a full PT checkpoint dict with "model_state_dict"
                       key (auto-extracted).
        layer_indices: optional dict mapping block_prefix (e.g. "prelude_0",
            "core_0", "coda_0") to the global layer index used for CSA/HCA
            alternation. If None, uses the block's position within its section
            as the layer index (which may be wrong for core/coda blocks).
            Provide this for models where n_prelude != 0.

    Returns:
        Nested dict suitable for flax model.apply({"params": ...}).
        Leaf values are float32 numpy arrays.
    """
    if "model_state_dict" in pt_state_dict:
        sd = pt_state_dict["model_state_dict"]
    else:
        sd = pt_state_dict

    params: dict = {}
    skipped: list[str] = []
    mapped: int = 0

    for pt_name, tensor in sd.items():
        # Convert to numpy
        if _HAS_TORCH:
            import torch as _t
            if isinstance(tensor, _t.Tensor):
                arr = tensor.detach().float().cpu().numpy()
            else:
                arr = np.array(tensor, dtype=np.float32)
        else:
            arr = np.array(tensor, dtype=np.float32)

        result = _pt_name_to_jax_path(pt_name, layer_indices)
        if result is None:
            skipped.append(pt_name)
            continue

        jax_path, needs_t = result

        # Transpose 2D kernels: PT [out, in] → JAX [in, out]
        if needs_t and arr.ndim == 2:
            arr = arr.T

        _set_nested(params, jax_path, arr)
        mapped += 1

    if skipped:
        print(f"[pt_to_jax] Skipped {len(skipped)} non-param entries "
              f"(col_indices, CMS state, etc.): {skipped[:3]}...")

    print(f"[pt_to_jax] Mapped {mapped} PT params → JAX nested dict")
    return params


# ── jax_to_pt ─────────────────────────────────────────────────────────────────

def jax_to_pt(jax_params: dict,
              layer_indices: dict[str, int] | None = None) -> dict:
    """Convert JAX/Flax nested param dict to PyTorch state dict.

    Args:
        jax_params: Nested dict of numpy/jax arrays (JAX convention).
        layer_indices: see pt_to_jax.

    Returns:
        Flat dict mapping PT param name → torch.Tensor.
        Suitable for model.load_state_dict(..., strict=False).
    """
    assert _HAS_TORCH, "torch is required for jax_to_pt()"
    import torch

    if _HAS_JAX:
        jax_params = jax.tree_util.tree_map(lambda x: np.array(x), jax_params)

    state_dict: dict = {}
    pairs = _flatten_dict(jax_params)

    for path, arr in pairs:
        leaf = path[-1]
        if leaf in ("col_indices", "alive_mask"):
            continue
        pt_name = _jax_path_to_pt_name(path, layer_indices)
        if _needs_transpose(path) and arr.ndim == 2:
            arr = arr.T  # [in, out] → [out, in]
        state_dict[pt_name] = torch.tensor(arr)

    return state_dict


# ── diff_param_sets ────────────────────────────────────────────────────────────

def diff_param_sets(pt_names: list[str], jax_params: dict,
                    layer_indices: dict[str, int] | None = None) -> dict:
    """Compare PT param names against converted JAX params.

    Returns:
        {
          "pt_only":  PT params not found in JAX after conversion,
          "jax_only": JAX params not found in PT,
          "matched":  params found in both,
        }
    """
    jax_pairs = _flatten_dict(jax_params)
    jax_pt_names = set()
    for path, arr in jax_pairs:
        leaf = path[-1]
        if leaf in ("col_indices", "alive_mask"):
            continue
        jax_pt_names.add(_jax_path_to_pt_name(path, layer_indices))

    pt_set = set(pt_names)
    pt_only  = pt_set - jax_pt_names
    jax_only = jax_pt_names - pt_set
    matched  = pt_set & jax_pt_names

    return {
        "pt_only":  sorted(pt_only),
        "jax_only": sorted(jax_only),
        "matched":  sorted(matched),
    }


# ── Serialization helpers ─────────────────────────────────────────────────────

def save_jax_params(params: dict, path: str | Path):
    """Save JAX params to msgpack or pickle."""
    assert _HAS_JAX, "jax/flax is required"
    import flax.serialization as ser
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".msgpack":
        with open(path, "wb") as f:
            f.write(ser.to_bytes(params))
    else:
        with open(path, "wb") as f:
            pickle.dump(params, f)
    print(f"[save_jax_params] Saved to {path}")


def load_jax_params(path: str | Path, template: dict | None = None) -> dict:
    """Load JAX params from msgpack or pickle."""
    assert _HAS_JAX, "jax/flax is required"
    path = Path(path)
    if path.suffix == ".msgpack" and template is not None:
        import flax.serialization as ser
        with open(path, "rb") as f:
            return ser.from_bytes(template, f.read())
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_pt_checkpoint(pt_state_dict: dict, path: str | Path):
    assert _HAS_TORCH, "torch is required"
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": pt_state_dict}, path)
    print(f"[save_pt_checkpoint] Saved to {path}")


def load_pt_checkpoint(path: str | Path) -> dict:
    assert _HAS_TORCH, "torch is required"
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def count_params_jax(params: dict) -> int:
    try:
        import jax
        return sum(x.size for x in jax.tree_util.tree_leaves(params))
    except ImportError:
        return sum(arr.size for _, arr in _flatten_dict(params))


def count_params_pt(state_dict: dict) -> int:
    total = 0
    for v in state_dict.values():
        if _HAS_TORCH:
            import torch as _t
            if isinstance(v, _t.Tensor):
                total += v.numel()
                continue
        if isinstance(v, np.ndarray):
            total += v.size
    return total


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert MORPH checkpoints between PyTorch and JAX/Flax.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--from", dest="src_fmt", required=True, choices=["pt", "jax"])
    parser.add_argument("--to",   dest="dst_fmt", required=True, choices=["pt", "jax"])
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    # layer_indices: "prelude_0:0,core_0:3,core_1:4,coda_0:9"
    parser.add_argument("--layer-indices", default=None,
                        help="Comma-separated block:global_idx pairs for CSA/HCA alternation. "
                             "Example: 'prelude_0:0,core_0:3,core_1:4,coda_0:9'")
    args = parser.parse_args()

    layer_indices = None
    if args.layer_indices:
        layer_indices = {}
        for pair in args.layer_indices.split(","):
            k, v = pair.strip().split(":")
            layer_indices[k.strip()] = int(v.strip())

    if args.src_fmt == "pt" and args.dst_fmt == "jax":
        print(f"Converting PyTorch → JAX: {args.input} → {args.output}")
        sd = load_pt_checkpoint(args.input)
        jax_params = pt_to_jax(sd, layer_indices)
        save_jax_params(jax_params, args.output)
        if args.verify:
            n_pt  = count_params_pt(sd)
            n_jax = count_params_jax(jax_params)
            print(f"  PT  params: {n_pt:,}")
            print(f"  JAX params: {n_jax:,}")
            diff = diff_param_sets(list(sd.keys()), jax_params, layer_indices)
            if diff["pt_only"]:
                print(f"  PT-only (not mapped): {diff['pt_only'][:10]}")
            if diff["jax_only"]:
                print(f"  JAX-only (extra): {diff['jax_only'][:10]}")
            print(f"  Matched: {len(diff['matched'])} params")
    elif args.src_fmt == "jax" and args.dst_fmt == "pt":
        print(f"Converting JAX → PyTorch: {args.input} → {args.output}")
        jax_p = load_jax_params(args.input)
        sd = jax_to_pt(jax_p, layer_indices)
        save_pt_checkpoint(sd, args.output)
        if args.verify:
            print(f"  JAX params: {count_params_jax(jax_p):,}")
            print(f"  PT  params: {count_params_pt(sd):,}")
    else:
        print("No-op.")
    print("Done.")


if __name__ == "__main__":
    main()
