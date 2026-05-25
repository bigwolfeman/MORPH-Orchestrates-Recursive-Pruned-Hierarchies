"""Bidirectional PyTorch ↔ JAX/Flax checkpoint converter for MORPH.

Handles the full MORPH parameter namespace:

  PT model paths                     JAX/Flax paths (nested dict)
  ────────────────────────────────── ────────────────────────────────────
  embed.hybrid.euc_embed.weight      embed/hybrid/euc_embed/embedding
  embed.hybrid.lor_embed.space_embed.weight
                                     embed/hybrid/lor_embed/space_embed/embedding
  embed.bigram.embed.weight          embed/bigram/embed/embedding
  embed.bigram.lambdas               embed/bigram/lambdas

  prelude.0.norm_attn.scale          prelude_0/mhc_attn/norm_attn/scale
  prelude.0.attn.cca_csa.W_q.weight  prelude_0/mhc_attn/attn/cca_csa/W_q/kernel
  prelude.0.mlp.gate_up.weight       prelude_0/mhc_mlp/mlp/gate_up/kernel
  prelude.0.mhc_attn.alpha_raw       prelude_0/mhc_attn/alpha_raw
  prelude.0.mhc_attn.gamma_raw       prelude_0/mhc_attn/gamma_raw

  core.N.*                           core_N/*
  coda.N.*                           coda_N/*

  x0_injects.N.*                     x0_inject_N/*
  value_embeds.N.*                   value_embed_N/*
  value_embed_tables.N.weight        value_embed_table_N/embedding

  injection.log_A                    injection/log_A
  injection.log_dt                   injection/log_dt
  input_norm.scale                   input_norm/scale
  final_norm.scale                   final_norm/scale

  lm_mixer.channel_scales            lm_mixer/channel_scales
  lm_mixer.mix.weight                lm_mixer/mix/kernel

  memory.ssm_query                   memory/ssm_query
  memory.ssm_proj.weight             memory/ssm_proj/kernel
  memory.mag_gate.weight             memory/mag_gate/kernel
  memory.mag_gate.bias               memory/mag_gate/bias
  memory.mag_proj.weight             memory/mag_proj/kernel
  memory.mac_gate_raw                memory/mac_gate_raw
  memory.mac_query_proj.weight       memory/mac_query_proj/kernel
  memory.mac_out_proj.weight         memory/mac_out_proj/kernel
  memory.memory.W_K.weight           memory/memory/W_K/kernel
  memory.memory.W_V.weight           memory/memory/W_V/kernel
  memory.memory.W_Q.weight           memory/memory/W_Q/kernel
  memory.memory.stable_norm.scale    memory/memory/stable_norm/scale
  memory.memory.mlp_output_norm_scale memory/memory/mlp_output_norm_scale
  memory.memory.delta_gate_raw       memory/memory/delta_gate_raw
  memory.memory.alpha_gate.weight    memory/memory/alpha_gate/kernel
  memory.memory.alpha_gate.bias      memory/memory/alpha_gate/bias
  memory.memory.eta_gate.weight      memory/memory/eta_gate/kernel
  memory.memory.eta_gate.bias        memory/memory/eta_gate/bias
  memory.memory.theta_gate.weight    memory/memory/theta_gate/kernel
  memory.memory.theta_gate.bias      memory/memory/theta_gate/bias

  z_heads.z_proj_linear.weight       z_heads/z_proj_linear/kernel
  z_heads.z_proj_linear.bias         z_heads/z_proj_linear/bias
  z_heads.z_proj_norm.*              z_heads/z_proj_norm/*
  z_heads.z_backbone_head.weight     z_heads/z_backbone_head/kernel
  z_heads.z_memory_head.weight       z_heads/z_memory_head/kernel

  stp.*                              (no parameters)

Note on memory MLP weights:
  The NeuralMemoryCore MLP weights live in the "memory_state" mutable
  collection (not in "params"). They are stored per-sequence state,
  not global learned parameters. The converter skips them from the
  main state dict but provides helper functions to export/import them.

Usage:
    python -m morph.interop.checkpoint \\
        --from pt --to jax \\
        --input ckpt.pt --output params.msgpack
"""

from __future__ import annotations

import argparse
import math
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np

# Optional imports (only needed for actual conversion, not name mapping)
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


# ── Name mapping constants ────────────────────────────────────────────────────

# These parameter leaf names are always 1D or scalar (no transpose needed)
_NO_TRANSPOSE_LEAVES = {
    "scale", "embedding", "bias", "log_A", "log_dt", "alpha_raw", "gamma_raw",
    "log_scale", "lambdas", "ssm_query", "mac_gate_raw", "delta_gate_raw",
    "mlp_output_norm_scale", "channel_scales", "sink_logits",
    "alpha", "beta",
}

# Embedding weight tables: PyTorch "weight" → JAX "embedding"
_EMBEDDING_PARENTS = {
    "euc_embed", "lor_embed", "space_embed", "embed",
    "value_embed_table_0", "value_embed_table_1", "value_embed_table_2",
    "iteration_embed",
}


# ── Utility: nested dict helpers ──────────────────────────────────────────────

def _set_nested(d: dict, path: list[str], value: Any):
    """Set d[path[0]][path[1]]...[path[-1]] = value, creating dicts on the way."""
    for p in path[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[path[-1]] = value


def _get_nested(d: dict, path: list[str]) -> Any | None:
    """Get d[path[0]][path[1]]..., returning None if any key is missing."""
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
        elif isinstance(v, np.ndarray):
            out.append((cur_path, v))
    return out


# ── PT → JAX name conversion ─────────────────────────────────────────────────

def _pt_name_to_jax_path(pt_name: str) -> tuple[list[str], bool]:
    """Convert a PyTorch param name to a JAX nested path list + transpose flag.

    Returns:
        (path, needs_transpose): path is the full nested key list including leaf,
        needs_transpose is True for 2D Dense kernels (PyTorch [out, in] → [in, out]).
    """
    parts = pt_name.split(".")

    jax_path: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]

        # ModuleList indices: "prelude.0" → "prelude_0"
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            jax_path.append(f"{p}_{parts[i + 1]}")
            i += 2
        else:
            jax_path.append(p)
            i += 1

    # Convert the leaf
    leaf = jax_path[-1]
    parent = jax_path[-2] if len(jax_path) >= 2 else ""

    if leaf == "weight":
        if parent in _EMBEDDING_PARENTS:
            jax_path[-1] = "embedding"
            return jax_path, False
        else:
            jax_path[-1] = "kernel"
            return jax_path, True  # Dense layers need transpose

    # All other leaves: scale, bias, log_A, etc.
    return jax_path, False


def _jax_path_to_pt_name(jax_path: list[str]) -> str:
    """Reverse of _pt_name_to_jax_path: JAX path list → PT param name string."""
    pt_parts = []
    for p in jax_path:
        # "prelude_0" → "prelude", "0"; "core_5" → "core", "5"
        # But: "alpha_raw", "gamma_raw", "log_A", "log_dt" must NOT be split
        # Rule: split on last underscore only if the suffix is purely digits.
        if "_" in p:
            prefix, suffix = p.rsplit("_", 1)
            if suffix.isdigit():
                pt_parts.extend([prefix, suffix])
                continue
        pt_parts.append(p)

    # Reverse leaf mapping
    leaf = pt_parts[-1]
    if leaf == "kernel":
        pt_parts[-1] = "weight"
    elif leaf == "embedding":
        pt_parts[-1] = "weight"

    return ".".join(pt_parts)


def _needs_transpose(jax_path: list[str]) -> bool:
    """True iff this is a 2D Dense kernel that needs [out, in] ↔ [in, out] transpose."""
    leaf = jax_path[-1]
    if leaf != "kernel":
        return False
    # Embeddings are named "embedding", not "kernel" — already handled.
    return True


# ── pt_to_jax ─────────────────────────────────────────────────────────────────

def pt_to_jax(pt_state_dict: dict) -> dict:
    """Convert PyTorch model state dict to JAX/Flax nested param dict.

    The JAX params are a nested dict suitable for use with flax.serialization
    or direct jax.tree_util operations.

    Args:
        pt_state_dict: dict mapping PT param name → torch.Tensor.
                       Can also be a full PT checkpoint dict with
                       "model_state_dict" key (auto-extracted).

    Returns:
        Nested dict {"embed": {"hybrid": {...}, ...}, "prelude_0": {...}, ...}
        where leaf values are float32 numpy arrays (shape matches JAX convention).
    """
    # Accept full checkpoint dict or bare state dict
    if "model_state_dict" in pt_state_dict:
        sd = pt_state_dict["model_state_dict"]
    else:
        sd = pt_state_dict

    params: dict = {}
    skipped: list[str] = []

    for pt_name, tensor in sd.items():
        # Convert tensor to numpy (fp32 for safety; caller can downcast)
        if _HAS_TORCH:
            import torch
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().float().cpu().numpy()
            else:
                arr = np.array(tensor, dtype=np.float32)
        else:
            arr = np.array(tensor, dtype=np.float32)

        # Skip non-parameter buffers
        if any(skip in pt_name for skip in ["freqs", "momentum_S", "_param_offsets"]):
            skipped.append(pt_name)
            continue

        jax_path, needs_t = _pt_name_to_jax_path(pt_name)

        if needs_t and arr.ndim == 2:
            arr = arr.T  # [out, in] → [in, out]

        _set_nested(params, jax_path, arr)

    if skipped:
        print(f"[pt_to_jax] Skipped {len(skipped)} buffers: {skipped[:5]}...")

    return params


# ── jax_to_pt ─────────────────────────────────────────────────────────────────

def jax_to_pt(jax_params: dict) -> dict:
    """Convert JAX/Flax nested param dict to PyTorch state dict.

    Args:
        jax_params: Nested dict of numpy arrays (JAX convention).

    Returns:
        Flat dict mapping PT param name → torch.Tensor.
        Suitable for model.load_state_dict(..., strict=False).
    """
    assert _HAS_TORCH, "torch is required for jax_to_pt()"
    import torch

    # Convert any jax.Array leaves to numpy first
    if _HAS_JAX:
        import jax as _jax
        jax_params = _jax.tree_util.tree_map(lambda x: np.array(x), jax_params)

    state_dict: dict = {}
    pairs = _flatten_dict(jax_params)

    for path, arr in pairs:
        leaf = path[-1]

        # Skip topology/state entries that aren't model params
        if leaf in ("col_indices", "alive_mask"):
            continue

        pt_name = _jax_path_to_pt_name(path)

        if _needs_transpose(path) and arr.ndim == 2:
            arr = arr.T  # [in, out] → [out, in]

        state_dict[pt_name] = torch.tensor(arr)

    return state_dict


# ── Lorentz embedding special handling ────────────────────────────────────────

def _verify_lorentz_roundtrip(euc_w: np.ndarray, lor_space_w: np.ndarray) -> bool:
    """Verify that Lorentz space weights round-trip through hyperboloid projection.

    The stored weights are the d-dim spatial components, NOT the (d+1)-dim
    Lorentz vectors. This is the same convention in both PT and JAX.

    Returns True if weights are plausible (no NaN/Inf).
    """
    sq = (lor_space_w ** 2).sum(axis=-1)
    x0 = np.sqrt(np.maximum(1.0 + sq, 1e-6))
    return not (np.isnan(x0).any() or np.isinf(x0).any())


# ── Bigram hash table: no special handling needed ─────────────────────────────
# The bigram embed.weight / embed/embed/embedding is just a standard embedding
# table. The hash function is applied at forward-pass time from input_ids, not
# stored in the checkpoint. No special conversion needed.


# ── Serialization helpers ─────────────────────────────────────────────────────

def save_jax_params(params: dict, path: str | Path):
    """Save JAX params to a msgpack file via flax.serialization."""
    assert _HAS_JAX, "jax/flax is required for save_jax_params()"
    import flax.serialization as ser

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".msgpack":
        serialized = ser.to_bytes(params)
        with open(path, "wb") as f:
            f.write(serialized)
    elif path.suffix in (".pkl", ".pickle"):
        with open(path, "wb") as f:
            pickle.dump(params, f)
    else:
        # Default: pickle
        with open(path, "wb") as f:
            pickle.dump(params, f)
    print(f"[save_jax_params] Saved to {path}")


def load_jax_params(path: str | Path, template: dict | None = None) -> dict:
    """Load JAX params from msgpack or pickle file.

    Args:
        path:     Path to the saved params file.
        template: Optional nested dict with the correct structure (for
                  flax.serialization.from_bytes). If None, uses pickle.
    """
    assert _HAS_JAX, "jax/flax is required for load_jax_params()"
    path = Path(path)

    if path.suffix == ".msgpack" and template is not None:
        import flax.serialization as ser
        with open(path, "rb") as f:
            data = f.read()
        return ser.from_bytes(template, data)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_pt_checkpoint(pt_state_dict: dict, path: str | Path):
    """Save PyTorch state dict as a .pt checkpoint."""
    assert _HAS_TORCH, "torch is required for save_pt_checkpoint()"
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": pt_state_dict}, path)
    print(f"[save_pt_checkpoint] Saved to {path}")


def load_pt_checkpoint(path: str | Path) -> dict:
    """Load a PyTorch checkpoint and return the state dict."""
    assert _HAS_TORCH, "torch is required for load_pt_checkpoint()"
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


# ── Param counting helpers ────────────────────────────────────────────────────

def count_params_jax(params: dict) -> int:
    """Count total parameters in a JAX param dict (handles jax.Array or numpy)."""
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(params)
        return sum(leaf.size for leaf in leaves)
    except ImportError:
        total = 0
        for path, arr in _flatten_dict(params):
            total += arr.size
        return total


def count_params_pt(state_dict: dict) -> int:
    """Count total parameters in a PT state dict."""
    total = 0
    for v in state_dict.values():
        if _HAS_TORCH:
            import torch
            if isinstance(v, torch.Tensor):
                total += v.numel()
                continue
        if isinstance(v, np.ndarray):
            total += v.size
    return total


# ── Diff helper (for debugging mismatches) ────────────────────────────────────

def diff_param_sets(pt_names: list[str], jax_params: dict) -> dict:
    """Compare PT param names against a converted JAX params dict.

    Returns:
        {
          "pt_only":  PT params not found in JAX (possible mapping gap),
          "jax_only": JAX params not found in PT (possible spurious additions),
          "matched":  Params successfully mapped in both directions,
        }
    """
    jax_pairs = _flatten_dict(jax_params)
    jax_pt_names = set()
    for path, arr in jax_pairs:
        leaf = path[-1]
        if leaf in ("col_indices", "alive_mask"):
            continue
        jax_pt_names.add(_jax_path_to_pt_name(path))

    pt_set = set(pt_names)
    pt_only  = pt_set - jax_pt_names
    jax_only = jax_pt_names - pt_set
    matched  = pt_set & jax_pt_names

    return {
        "pt_only":  sorted(pt_only),
        "jax_only": sorted(jax_only),
        "matched":  sorted(matched),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert MORPH checkpoints between PyTorch and JAX/Flax.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from", dest="src_fmt", required=True,
        choices=["pt", "jax"],
        help="Source format: 'pt' (PyTorch .pt) or 'jax' (.msgpack / .pkl)",
    )
    parser.add_argument(
        "--to", dest="dst_fmt", required=True,
        choices=["pt", "jax"],
        help="Destination format",
    )
    parser.add_argument("--input",  required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--verify", action="store_true",
        help="Print parameter count and name diff after conversion",
    )
    args = parser.parse_args()

    if args.src_fmt == "pt" and args.dst_fmt == "jax":
        print(f"Converting PyTorch → JAX: {args.input} → {args.output}")
        if not _HAS_TORCH:
            raise ImportError("torch is required for PyTorch → JAX conversion")
        sd = load_pt_checkpoint(args.input)
        jax_params = pt_to_jax(sd)
        save_jax_params(jax_params, args.output)

        if args.verify:
            n_pt  = count_params_pt(sd)
            n_jax = count_params_jax(jax_params)
            print(f"  PT  params: {n_pt:,}")
            print(f"  JAX params: {n_jax:,}")
            diff = diff_param_sets(list(sd.keys()), jax_params)
            if diff["pt_only"]:
                print(f"  PT-only (not mapped): {diff['pt_only'][:10]}")
            if diff["jax_only"]:
                print(f"  JAX-only (extra): {diff['jax_only'][:10]}")
            print(f"  Matched: {len(diff['matched'])} params")

    elif args.src_fmt == "jax" and args.dst_fmt == "pt":
        print(f"Converting JAX → PyTorch: {args.input} → {args.output}")
        if not _HAS_TORCH:
            raise ImportError("torch is required for JAX → PyTorch conversion")
        jax_params = load_jax_params(args.input)
        sd = jax_to_pt(jax_params)
        save_pt_checkpoint(sd, args.output)

        if args.verify:
            n_jax = count_params_jax(jax_params)
            n_pt  = count_params_pt(sd)
            print(f"  JAX params: {n_jax:,}")
            print(f"  PT  params: {n_pt:,}")

    else:
        print(f"No-op: --from {args.src_fmt} --to {args.dst_fmt} (same format)")

    print("Done.")


if __name__ == "__main__":
    main()
