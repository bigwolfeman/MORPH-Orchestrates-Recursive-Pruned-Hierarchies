"""Two-source causal attention for the faithful DiffusionBlocks concat.

Design: ``docs/diffusionblocks-concat-design.md``.

The paper's App. E.4 causal consistency concatenates a CLEAN and a NOISY copy of
the sequence under a modified causal mask, so a noisy target position attends the
CLEAN past but never its own clean answer. MORPH cannot run that as one ``2L``
tensor — its CCA causal conv, value-shift and CoPE-RoPE encode position
structurally and would cross the clean|noisy boundary. So MORPH takes the paper's
own stated alternative (line 1428): each stream is a normal length-``L`` causal
sequence, and the NOISY stream's attention additionally attends the CLEAN stream's
keys, causally.

This module holds the PURE, side-effect-free pieces of that two-source attention —
the block-bank causal masks and the reference two-source window — so the index
arithmetic (the part a leak hides in) is unit-testable without a model. The
attention modules in ``attention.py`` call these; the clean-stream capture and the
noisy-stream merge live there.

Visibility, MORPH next-token convention (``labels[t] = input_ids[t+1]``):

  * ``clean_i = embed(input_ids[i])``      — clean CURRENT token; plain causal LM context.
  * ``noisy_i = embed(input_ids[i+1]) + σε`` — the NOISED TARGET, read out at position i.

  A noisy query ``i`` may attend:
    - clean positions ``j <= i``  (context up to AND INCLUDING the current token;
      ``clean_i`` is the current token, not the target ``clean_{i+1}``), and
    - noisy positions per the baseline rule (its own causal past).
  It must NEVER attend ``clean_{i+1}`` — that is its target. Every rule below is the
  arithmetic that enforces exactly this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "merged_block_causal_mask",
    "clean_block_causal_mask",
    "two_source_window",
]


def clean_block_causal_mask(S: int, n_blocks: int, m: int, device) -> Tensor:
    """[S, n_blocks] bool: CLEAN block j is visible to noisy query i iff the whole
    block lies at positions ``<= i`` (inclusive of the current token i).

    Contrast ``attention._compressed_causal_mask`` (the NOISY rule), which is
    strict (``block_end < i``). The clean rule is ``block_end <= i`` so the clean
    CURRENT token is visible, while a block that ends at ``i+1`` (which would carry
    the target ``clean_{i+1}``) stays hidden. A block that STRADDLES i is excluded
    (its ``<= i`` part is served by the window); this guarantees no block can leak
    the target.
    """
    block_end = (torch.arange(n_blocks, device=device) + 1) * m - 1   # [nb]
    query_pos = torch.arange(S, device=device)                          # [S]
    return block_end.unsqueeze(0) <= query_pos.unsqueeze(1)             # [S, nb]


def merged_block_causal_mask(S: int, nb_clean: int, nb_noisy: int, m: int,
                             device) -> Tensor:
    """[S, nb_clean + nb_noisy] bool causal mask over the MERGED block bank
    ``cat([clean_blocks, noisy_blocks], dim=block)``.

    Clean half uses the inclusive rule (:func:`clean_block_causal_mask`); noisy
    half uses the baseline strict rule (``block_end < i``), identical to
    ``attention._compressed_causal_mask`` so the noisy-on-noisy path is bit-identical
    to a single-stream forward.
    """
    block_end = (torch.arange(nb_noisy, device=device) + 1) * m - 1
    query_pos = torch.arange(S, device=device)
    noisy = block_end.unsqueeze(0) < query_pos.unsqueeze(1)             # [S, nb_noisy]
    clean = clean_block_causal_mask(S, nb_clean, m, device)            # [S, nb_clean]
    return torch.cat([clean, noisy], dim=1)                            # [S, nb_clean+nb_noisy]


def two_source_window(q: Tensor,
                      k_clean: Tensor, v_clean: Tensor,
                      k_noisy: Tensor, v_noisy: Tensor,
                      window_size: int, scale: float,
                      n_skip_rope: int = 0) -> Tensor:
    """Reference two-source causal sliding-window attention (v1; not yet fused).

    A noisy query ``i`` attends, within the local window:
      * clean keys ``j`` with ``0 <= i - j < window_size`` — INCLUDING ``j == i``
        (``clean_i`` is the current token, legitimate context), and
      * noisy keys ``j`` with ``0 < i - j < window_size`` — EXCLUDING ``j == i``
        (XSA self-exclusion, exactly the baseline ``_window_fallback`` rule).

    Suffix (``n_skip_rope``) positions are always mutually visible, matching the
    baseline. Implemented as one SDPA over ``cat([k_clean, k_noisy])`` with a
    ``[S, 2S]`` additive mask. Correct but O(S·window) dense; the fused two-source
    window is a follow-up optimization (design §5, risk 4). Labeled reference so it
    is never reported as the fused path.

    Shapes: q/k/v ``[B, H, S, D]`` (K/V already GQA-expanded). Returns ``[B, H, S, D]``.
    """
    S = q.shape[2]
    device = q.device
    row = torch.arange(S, device=device).unsqueeze(1)                  # i
    col = torch.arange(S, device=device).unsqueeze(0)                  # j
    dist = row - col

    in_window = (dist >= 0) & (dist < window_size)
    noisy_mask = in_window & (dist != 0)                               # XSA: exclude self
    clean_mask = in_window                                             # include current token

    if n_skip_rope > 0:
        is_suffix_col = col >= S - n_skip_rope
        is_suffix_row = row >= S - n_skip_rope
        suffix = is_suffix_col | is_suffix_row
        noisy_mask = noisy_mask | suffix
        clean_mask = clean_mask | suffix

    merged = torch.cat([clean_mask, noisy_mask], dim=1)                # [S, 2S]
    bias = torch.where(merged, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
    k = torch.cat([k_clean, k_noisy], dim=2)                           # [B, H, 2S, D]
    v = torch.cat([v_clean, v_noisy], dim=2)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
