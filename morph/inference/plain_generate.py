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


@torch.no_grad()
def generate_plain_batch(
    model,
    prompts: list[list[int]],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seeds: list[int] | None = None,
    device=None,
    pad_id: int = 0,
) -> list[list[int]]:
    """`generate_plain` for B rows at once. Returns one list of NEW tokens per row.

    Why this exists: the A0-vs-A1 repetition question needs statistical power, and one
    row at a time does not buy it. rep4 has a per-sample paired standard deviation of
    ~0.34 at top-k while the A1-A0 gap is ~0.03, so n=12 resolves nothing and n in the
    low thousands is what the question actually costs. At 33 s per 512-token row that is
    a 12-hour job; batched it is under two.

    RAGGED BY DESIGN. Rows may start from different prompt lengths and, in the TUL twin
    of this function, grow at different rates. Each row keeps its own write cursor and
    its logits are read at its OWN last real position; everything past that cursor is
    padding. This is only sound because the model is strictly causal at position
    granularity -- compressed blocks are visible only once fully in the past
    (`_compressed_causal_mask`: block j is causal for query i iff (j+1)*m - 1 < i), and
    the CCA convolutions are causal. That reasoning is NOT taken on trust:
    `tests/test_generation_sampling.py` asserts batched greedy equals single-row greedy
    token for token, which fails if any path peeks past a row's cursor.
    """
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("generate_plain_batch needs a non-empty prompt per row")
    B = len(prompts)
    if seeds is not None and len(seeds) != B:
        raise ValueError(f"seeds must have one entry per row, got {len(seeds)} for {B}")
    gens = ([torch.Generator(device=str(device)).manual_seed(int(s)) for s in seeds]
            if seeds is not None else [None] * B)

    lens = [len(p) for p in prompts]
    total = max(lens) + max_new_tokens
    ids = torch.full((B, total), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
    cur = torch.tensor(lens, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)
    emitted: list[list[int]] = [[] for _ in range(B)]
    try:
        for _ in range(max_new_tokens):
            T = int(cur.max())
            res = model(ids[:, :T])
            logits = res["logits"] if isinstance(res, dict) else res
            last = logits[rows, cur - 1]                     # [B, V], each row's own tail
            for i in range(B):
                nxt = sample_next(last[i], temperature, top_k, gens[i])
                ids[i, cur[i]] = nxt
                emitted[i].append(nxt)
            cur += 1
    finally:
        if was_training:
            model.train()
    return emitted
