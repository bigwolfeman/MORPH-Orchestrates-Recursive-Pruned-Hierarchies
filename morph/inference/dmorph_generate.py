"""Eager dmorph generation: the clean pass, then the ``K``-step ladder with the hard
bridge, one token at a time (design note, "Arm dm-tok", inference).

The same recompute-per-step shape as :mod:`morph.inference.tul_generate` (no KV cache,
by design: "every step is a fresh forward, which is slow but cannot be subtly wrong").
Each step:

    run the clean pass over the grown row with the K/V capture (the shipped forward)
    run the B-step Euler ladder at every position from pure noise, one block per step
    in order 0 → B−1, the HARD bridge between steps (tok arm)
    sample the next token from EITHER head at the row's last emitting position

``head="clean"`` is the shipped head (always available on every arm); ``head="ladder"``
is the ladder's last unbridged ``D̂`` through the tied head. Both are scored in the
panel (prereg P3 compares their greedy accuracy on the same rows). The layout is built
by the SAME :class:`~morph.inference.tul_generate.TulRowBuilder` the TUL generator uses,
so the boundary rule and the slot insertion are the loader's (runtime-invariants §6b).

The diversity guard travels with every generation number: score the returned tokens
with :func:`morph.inference.gen_metrics.generation_metrics` (rep4 / distinct-3) — a
repetition loop scores a better gen-PPL than real text (memory ``gen-PPL needs a
diversity guard``; ``db-testbed-ladder.md`` B).
"""

from __future__ import annotations

import torch
from torch import Tensor

from morph.inference.sampling import sample_next
from morph.inference.tul_generate import TulRowBuilder
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec

__all__ = ["HEADS", "generate_dmorph"]

HEADS = ("clean", "ladder")


@torch.no_grad()
def generate_dmorph(
    model,
    prompt_ids: list[int] | Tensor,
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device=None,
    head: str = "clean",
) -> tuple[list[int], TulRowBuilder]:
    """Generate ``max_new_tokens`` TOKENS (slots are inserted by the rule, not counted).

    ``temperature = 0`` → greedy. Returns ``(token_ids, builder)``; the builder carries
    the realised layout so a caller can assert parity against the loader's packer or read
    the span-length distribution.

    The emitting position follows ``tul_generate``'s ``emit_source="token"`` rule: when
    the row ends with a freshly inserted slot, the next token is read from the boundary
    TOKEN's position (the only position trained at weight 1 under ``emit_weight 0``),
    ``1 + prefix_k`` back; otherwise from the last position.
    """
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}")
    if getattr(model, "dmorph", None) is None:
        raise RuntimeError("generate_dmorph needs a model built with MORPHConfig(dmorph=...)")
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=str(device)).manual_seed(seed)

    if isinstance(prompt_ids, Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    if not prompt_ids:
        raise ValueError("generate_dmorph needs at least one prompt token")

    builder = TulRowBuilder(rule=rule, spec=spec)
    for t in prompt_ids:
        builder.append(int(t))

    key = "logits" if head == "clean" else "ladder_logits"
    emitted: list[int] = []
    try:
        for _ in range(max_new_tokens):
            ids, layout = builder.tensors(device)
            res = model.dmorph_infer(ids, layout)
            back = 1 + spec.prefix_k if builder.slot_mask[-1] else 1
            logits = res[key][0, -back]
            nxt = sample_next(logits, temperature, top_k, gen)
            emitted.append(nxt)
            builder.append(nxt)
    finally:
        if was_training:
            model.train()
    return emitted, builder
