"""MORPH inference stack — deploy-quant build + fast decode engine.

Decode-exclusive code, separated from training/model-core. Depends on the
shared Triton kernels in ``morph.kernels.triton`` (HC, CCA, decode-step, router)
and on ``morph.model.kv_quant`` for KV-cache quantization; those stay in place.

Public entry points
-------------------
- :func:`to_deploy_inference` — compose the full deploy-quant build (int6 embeds
  + PackedTernary backbone + Int8Row attention + 2-bit-packed carved MLPs).
- :class:`StaticDecodeEngine` — CUDA-graph-captured fused decode (B>=1).
- :class:`MORPHKVCache` / :func:`prefill` / :func:`decode_step` — eager AR cache.
"""

from morph.inference.engine import StaticDecodeEngine, materialize_quant
from morph.inference.deploy_quant import (
    to_deploy_inference,
    materialize_top_level,
    shrink_block,
    shrink_mlp_to_mortar_ternary,
    pack_mortar_ternary,
    quantize_attention_linears,
    strip_cms_inference,
    resident_bytes_report,
    extract_ternary_from_parametrized,
    pack_ternary_codes,
    unpack_ternary,
    PackedTernaryLinear,
    Int8RowLinear,
    Int6RowEmbedding,
)
from morph.inference.kv_cache import (
    MORPHKVCache,
    AttnSiteCache,
    prefill,
    decode_step,
)

__all__ = [
    "StaticDecodeEngine",
    "materialize_quant",
    "to_deploy_inference",
    "materialize_top_level",
    "shrink_block",
    "shrink_mlp_to_mortar_ternary",
    "pack_mortar_ternary",
    "quantize_attention_linears",
    "strip_cms_inference",
    "resident_bytes_report",
    "extract_ternary_from_parametrized",
    "pack_ternary_codes",
    "unpack_ternary",
    "PackedTernaryLinear",
    "Int8RowLinear",
    "Int6RowEmbedding",
    "MORPHKVCache",
    "AttnSiteCache",
    "prefill",
    "decode_step",
]
