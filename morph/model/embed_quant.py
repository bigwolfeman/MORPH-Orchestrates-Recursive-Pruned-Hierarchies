"""Embedding quantization-aware training (QAT) for MORPH.

Ablation E: per-row int8/int6 QAT on token (euclidean) and bigram-hash embedding
tables. The Lorentz space embedding ALWAYS stays bf16 — it is geometry-sensitive
(trained geodesic radius ~0.97 from the Lorentz curvature measurement) and must
not be quantized.

Mechanism: ``torch.nn.utils.parametrize`` registers ``IntNRowSTE`` on the
selected ``nn.Embedding.weight`` parameters. After registration, the effective
forward weight is ``round_clip(w_row/s_row, lo, hi) * s_row`` — exactly int{n}-
representable — while the smooth fp32 shadow is at
``module.parametrizations.weight.original``. Gradients pass straight through the
quantizer (STE identity backward) to the shadow; AdamW/AdamW8bit updates the
shadow. This mirrors the ``TernarySTE`` approach in ``ternary_qat.py``.

Design rules (from CLAUDE.md + spec §3):
- NO runtime ``if`` in forward. Mode is resolved at construction via
  ``parametrize``. ``embed_quant=off`` → no parametrization → bit-identical to bf16.
- Lorentz space embed (``lor_embed.space_embed``) is NEVER touched.
- LM head is TIED to the euclidean + Lorentz embeddings via
  ``HybridEmbedding.lm_weight()``. Quantizing ``euc_embed.weight`` already
  propagates into the LM head logits. Since the head is tied (no separate weight),
  ``lm_head_quant`` is documented as a no-op at the weight level — what matters
  for logit precision is handled in ``fused_ce.py`` where logits are already cast
  to fp32 (``logits_c = (x_c @ wT).float()``). We document this clearly in the
  manifest rather than quantizing something that does not exist separately.
- fp32 scales (per row), fp32 shadow (via parametrize), bf16 activation path.

Int6 representation: symmetric, 6-bit signed → values in [-31, 31] × s_row.
(Not [-32, 31] — we use symmetric range to avoid the asymmetric zero-point
complexity; this matches BitNet's per-vector scale convention.)
Int8: symmetric signed → [-127, 127] × s_row.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import Tensor

# Symmetric int{n} ranges (we avoid the negative-extreme for symmetry).
_RANGES: dict[str, tuple[int, int]] = {
    "int8": (-127, 127),
    "int6": (-31, 31),
}


class IntNRowSTE(nn.Module):
    """Per-row int-N straight-through estimator for embedding tables.

    For each row w_row of the embedding weight W (shape [V, d]):
        s_row = max(|w_row|) / hi   (detached, per-row scale, fp32)
        q_row = round(w_row / s_row).clamp(lo, hi) * s_row

    The effective forward weight is EXACTLY int{n}-representable × s_row.
    Gradient flows straight through to the smooth shadow (identity backward).

    Args:
        bits: quantization bits — 8 or 6.
    """

    def __init__(self, bits: int) -> None:
        super().__init__()
        if bits not in (6, 8):
            raise ValueError(f"IntNRowSTE: bits must be 6 or 8, got {bits}")
        key = f"int{bits}"
        lo, hi = _RANGES[key]
        self.bits = bits
        self.lo: int = lo
        self.hi: int = hi

    def forward(self, w: Tensor) -> Tensor:
        """Return the per-row quantized weight.  Backward is identity STE."""
        # w: [V, d] fp32 (shadow stored as fp32 by parametrize)
        # Per-row scale: max(|w_row|) / hi, detached → constant for STE.
        # Shape: [V, 1] so broadcasting works.
        s = w.detach().abs().amax(dim=-1, keepdim=True).div(self.hi).clamp(min=1e-8)
        # Quantize to the integer grid, cast back to float.
        q_int = (w / s).round().clamp(self.lo, self.hi)   # [V, d] on int grid
        q = q_int * s                                       # [V, d] float-domain
        # STE: forward = q, backward = identity through w.
        return w + (q - w).detach()

    def right_inverse(self, w: Tensor) -> Tensor:
        """Identity: the shadow lives in the same space as the effective weight."""
        return w


def apply_embed_quant(
    model: nn.Module,
    embed_quant: str = "off",
    lm_head_quant: str = "off",
) -> dict:
    """Register per-row int-N QAT on the selected embedding tables.

    Targets (when embed_quant != "off"):
      - ``embed.hybrid.euc_embed``     (euclidean token embedding, [V, euc_dim])
      - ``embed.bigram.embed``         (bigram hash table, [hash_V, d_model])

    ALWAYS skipped (regardless of knobs):
      - ``embed.hybrid.lor_embed.space_embed``  (Lorentz, geometry-sensitive)
      - Any other module not matched above.

    LM head: the head is WEIGHT-TIED to the euclidean embedding via
    ``HybridEmbedding.lm_weight()``. There is NO separate LM-head weight.
    Quantizing ``euc_embed.weight`` already affects LM-head logits.
    ``lm_head_quant`` is therefore a NO-OP at the weight level; the manifest
    records this explicitly. Logit precision (fp32 dequant before softmax) is
    already handled in ``fused_ce.py`` unconditionally.

    Call AFTER model construction and BEFORE torch.compile + optimizer.

    Returns:
        manifest dict with mode, counts, module names, and LM-head note.
    """
    if embed_quant not in ("off", "int8", "int6"):
        raise ValueError(f"embed_quant must be off|int8|int6, got {embed_quant!r}")
    if lm_head_quant not in ("off", "int6"):
        raise ValueError(f"lm_head_quant must be off|int6, got {lm_head_quant!r}")

    lm_head_note = (
        "LM head is WEIGHT-TIED to euc_embed (no separate weight). "
        "lm_head_quant is a NO-OP at the weight level. "
        "Logits are cast to fp32 in fused_ce.py before the chunked softmax."
    )

    if embed_quant == "off":
        return {
            "embed_quant": "off",
            "lm_head_quant": lm_head_quant,
            "n_modules_quantized": 0,
            "module_names": [],
            "lm_head_note": lm_head_note,
        }

    bits = int(embed_quant[3:])  # "int8" → 8, "int6" → 6
    ste = IntNRowSTE(bits=bits)

    # The two target modules, identified by path fragments.
    # We walk named_modules and match by path to be explicit — no fuzzy isinstance
    # scan that might accidentally catch value_embed_tables or future modules.
    TARGET_PATHS = (
        "embed.hybrid.euc_embed",
        "embed.bigram.embed",
    )
    LORENTZ_PATH = "embed.hybrid.lor_embed.space_embed"

    registered: list[str] = []
    lorentz_guarded: list[str] = []

    # Build a name→module map for exact-path matching.
    name_to_module = dict(model.named_modules())

    for path in TARGET_PATHS:
        if path not in name_to_module:
            # Model may use a different attribute layout (warn but don't crash).
            continue
        mod = name_to_module[path]
        if not isinstance(mod, nn.Embedding):
            raise RuntimeError(
                f"apply_embed_quant: expected nn.Embedding at {path!r}, "
                f"got {type(mod).__name__}"
            )
        if parametrize.is_parametrized(mod, "weight"):
            continue  # idempotent
        # Each module gets its own IntNRowSTE instance (they share no state,
        # but distinct objects avoids any accidental state aliasing).
        parametrize.register_parametrization(
            mod, "weight", IntNRowSTE(bits=bits)
        )
        registered.append(path)

    # Safety guard: assert we did NOT touch the Lorentz space embed.
    if LORENTZ_PATH in name_to_module:
        mod_lor = name_to_module[LORENTZ_PATH]
        if parametrize.is_parametrized(mod_lor, "weight"):
            raise RuntimeError(
                "apply_embed_quant: LORENTZ SAFETY VIOLATION — "
                f"{LORENTZ_PATH} was parametrized. This must never happen."
            )
        lorentz_guarded.append(LORENTZ_PATH)

    return {
        "embed_quant": embed_quant,
        "lm_head_quant": lm_head_quant,
        "n_modules_quantized": len(registered),
        "module_names": registered,
        "lorentz_untouched": lorentz_guarded,
        "lm_head_note": lm_head_note,
    }
