"""Generation for a MORPH model built WITHOUT TUL — the A0 baseline arm.

This file exists because the first TUL generation table had no baseline in it. Arm A0
sets `tul.activate_at: never`, so it constructs no slot parameters and
`build_tul_runtime` returns None; the sampling script's response was to print
"SKIP: this arm builds no TUL layout" and move on. The table that came out compared
TUL against TUL and could not answer the one question it was written to answer —
whether the slot loop reduces repetition against a plain model.

The decode procedure here is deliberately IDENTICAL to `generate_tul`: eager, one full
recompute per step, no KV cache, and the same `sample_next`. That is slow and it is the
point — a repetition difference between the arms must come from the weights, not from
one arm decoding through a cache the other does not have.
"""

from __future__ import annotations

import torch
from torch import Tensor

from morph.inference.sampling import sample_next

__all__ = ["generate_plain"]


@torch.no_grad()
def generate_plain(
    model,
    prompt_ids: list[int] | Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device=None,
) -> list[int]:
    """Generate `max_new_tokens` tokens with no slot layout. Returns the NEW tokens."""
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=str(device)).manual_seed(seed)

    if isinstance(prompt_ids, Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    if not prompt_ids:
        raise ValueError("generate_plain needs at least one prompt token")

    row = [int(t) for t in prompt_ids]
    emitted: list[int] = []
    try:
        for _ in range(max_new_tokens):
            ids = torch.tensor(row, dtype=torch.long, device=device)[None]
            res = model(ids)
            logits = res["logits"] if isinstance(res, dict) else res
            nxt = sample_next(logits[0, -1], temperature, top_k, gen)
            emitted.append(nxt)
            row.append(nxt)
    finally:
        if was_training:
            model.train()
    return emitted
