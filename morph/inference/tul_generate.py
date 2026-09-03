"""Eager TUL generation (docs/tul-spec.md §6, v1).

The spec defers the inference-engine port ("separate work; eager generation for the
test" [W]), so this is a plain recompute-per-step sampler: it holds no KV cache and
re-runs prelude → core → coda over the whole grown row each step. That is O(n²) and
slow on purpose — its job is to prove that the slot machinery generates and that the
layout it builds is the SAME layout the loader builds (invariant 1, tested by
``tests/test_tul_layout.py::test_generator_layout_matches_loader``).

The loop follows §6 exactly:

    emit token from the last position's coda logits (slot_id masked)
    span_len += 1
    if the shared boundary rule cuts here:
        insert the slot's prefix_k positions; run prelude → core → coda on them
        emit the first token of the next span from the slot's logits

Because the whole row is recomputed, "run the core on the new slot" happens as part of
the recompute rather than as an incremental state update; the emitted sequence is
identical either way, and the state-carrying version belongs with the KV-cache port.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from morph.inference.sampling import sample_next
from morph.model.tul_layout import BoundaryRule, SlotLayout, TulLayoutSpec

__all__ = ["TulRowBuilder", "generate_tul"]


@dataclass
class TulRowBuilder:
    """Incremental twin of ``pack_tul_row`` — one token at a time (spec §6).

    Feeds every token through the SAME :meth:`BoundaryRule.cut` state machine the
    loader uses, threading ``span_len`` across single-token calls. Everything the model
    needs (``input_ids`` plus the four layout tensors) is appended as it goes.
    """

    rule: BoundaryRule
    spec: TulLayoutSpec
    ids: list[int] = field(default_factory=list)
    slot_mask: list[bool] = field(default_factory=list)
    bag_id: list[int] = field(default_factory=list)
    slot_first: list[int] = field(default_factory=list)
    span_len: int = 0

    @property
    def n_slots(self) -> int:
        return len(self.slot_first)

    def append(self, token_id: int) -> bool:
        """Append one emitted token; insert its slot if the rule cuts here.

        Returns True when a slot was inserted (the caller's next logits then come from
        the slot's emitting position rather than from the token).
        """
        if token_id == self.spec.slot_id:
            raise ValueError(
                f"generator emitted slot_id {token_id}: its logit must be −inf "
                f"(spec §3.1, invariant 4)")
        self.ids.append(int(token_id))
        self.slot_mask.append(False)
        self.bag_id.append(self.n_slots)          # the slot that will close this span
        cuts, self.span_len = self.rule.cut(np.array([token_id], dtype=np.int64), self.span_len)
        if cuts.size == 0:
            return False
        if self.n_slots >= self.spec.max_slots:
            # Out of slot budget: the loader would end the row here (spec §3.1). Keep
            # generating tokens rather than silently dropping the boundary's slot, and
            # say so — a silent divergence from the training layout is the failure this
            # whole module exists to prevent.
            raise RuntimeError(
                f"generation exceeded max_slots={self.spec.max_slots}; raise the budget "
                f"(it is sized from seq_len // 8 by default)")
        s = self.n_slots
        self.slot_first.append(len(self.ids))
        for _ in range(self.spec.prefix_k):
            self.ids.append(self.spec.slot_id)
            self.slot_mask.append(True)
            self.bag_id.append(s)
        return True

    def tensors(self, device) -> tuple[Tensor, SlotLayout]:
        """``(input_ids [1, L], layout)`` for the current row."""
        S = self.spec.max_slots
        idx = np.zeros(S, dtype=np.int64)
        valid = np.zeros(S, dtype=bool)
        if self.slot_first:
            idx[: self.n_slots] = np.asarray(self.slot_first, dtype=np.int64)
            valid[: self.n_slots] = True
        bag = np.asarray(self.bag_id, dtype=np.int64)
        bag[bag >= self.n_slots] = S                       # open span → the dump bin
        layout = SlotLayout(
            slot_mask=torch.from_numpy(np.asarray(self.slot_mask)[None]).to(device),
            bag_id=torch.from_numpy(bag[None]).to(device),
            slot_index=torch.from_numpy(idx[None]).to(device),
            slot_valid=torch.from_numpy(valid[None]).to(device),
            prefix_k=self.spec.prefix_k,
        )
        ids = torch.tensor(self.ids, dtype=torch.long, device=device)[None]
        return ids, layout


@torch.no_grad()
def generate_tul(
    model,
    prompt_ids: list[int] | Tensor,
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    device=None,
    emit_source: str = "token",
) -> tuple[list[int], TulRowBuilder]:
    """Generate ``max_new_tokens`` TOKENS (slots are inserted by the rule, not counted).

    ``temperature = 0`` → greedy. Returns ``(token_ids, builder)``; the builder carries
    the realised layout so a caller can assert parity against the loader's packer or
    read the span-length distribution (spec §7.2 generation metrics).

    ``emit_source`` decides which position's logits produce a span's FIRST token when
    the row currently ends with a freshly inserted slot:
      - ``"token"`` — the boundary TOKEN's position, the one position that is always
        trained at weight 1 (``pack_tul_row``: token labels skip slots). The default and
        the right choice for the shipped recipe (``tul.emit_weight == 0``): its slot
        readout is untrained, measured as the missing-space-after-period artifact
        (lab/divergence/emit_space_probe.py — emit-position space mass 0.47–0.58 vs
        0.81 at the token position).
      - ``"slot"``  — the slot's emitting position (spec §6 v1). Correct ONLY when
        training put weight on the slot emit label (``tul.emit_weight > 0``).
    Slot insertion, layouts, and every other part of the procedure are identical.
    """
    if emit_source not in ("slot", "token"):
        raise ValueError(f"emit_source must be 'slot' or 'token', got {emit_source!r}")
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    gen = None
    if seed is not None:
        gen = torch.Generator(device=str(device)).manual_seed(seed)

    if isinstance(prompt_ids, Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    if not prompt_ids:
        raise ValueError("generate_tul needs at least one prompt token")

    builder = TulRowBuilder(rule=rule, spec=spec)
    for t in prompt_ids:
        builder.append(int(t))

    emitted: list[int] = []
    try:
        for _ in range(max_new_tokens):
            ids, layout = builder.tensors(device)
            res = model(ids, slot_layout=layout)
            # Row ends with the K slot positions exactly when the last append cut a
            # boundary; `emit_source="token"` then reads the last TOKEN position.
            back = (1 + spec.prefix_k
                    if emit_source == "token" and builder.slot_mask[-1] else 1)
            logits = res["logits"][0, -back]
            # ONE sampling step for both generators — see morph/inference/sampling.py.
            nxt = sample_next(logits, temperature, top_k, gen)
            emitted.append(nxt)
            builder.append(nxt)
    finally:
        if was_training:
            model.train()
    return emitted, builder


@torch.no_grad()
def generate_tul_batch(
    model,
    prompts: list[list[int]],
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    seeds: list[int] | None = None,
    device=None,
    pad_id: int = 0,
    emit_source: str = "token",
) -> tuple[list[list[int]], list[TulRowBuilder]]:
    """`generate_tul` for B rows at once. Returns (new tokens per row, builders).

    Same motivation and same ragged contract as `plain_generate.generate_plain_batch`,
    with one extra source of raggedness that is specific to TUL: rows insert slots at
    their own boundaries, so two rows that started at the same length are different
    lengths a few tokens later. Each row therefore keeps its own cursor, its logits are
    read at its own last position, and every row's layout is padded out to the batch's
    current maximum with `bag_id = max_slots` (the dump bin) and `slot_mask = False`, so
    a padded position is neither a slot nor a member of any span.

    Parity against the single-row generator is asserted in
    `tests/test_generation_sampling.py`, because the whole point of this file is that the
    TUL and non-TUL arms are decoded by procedures that differ in nothing but the layout.
    """
    if emit_source not in ("slot", "token"):
        raise ValueError(f"emit_source must be 'slot' or 'token', got {emit_source!r}")
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("generate_tul_batch needs a non-empty prompt per row")
    B = len(prompts)
    if seeds is not None and len(seeds) != B:
        raise ValueError(f"seeds must have one entry per row, got {len(seeds)} for {B}")
    gens = ([torch.Generator(device=str(device)).manual_seed(int(s)) for s in seeds]
            if seeds is not None else [None] * B)

    builders = [TulRowBuilder(rule=rule, spec=spec) for _ in range(B)]
    for bld, p in zip(builders, prompts):
        for t in p:
            bld.append(int(t))
    emitted: list[list[int]] = [[] for _ in range(B)]
    S = spec.max_slots
    rows = torch.arange(B, device=device)
    try:
        for _ in range(max_new_tokens):
            L = max(len(b.ids) for b in builders)
            ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            smask = torch.zeros((B, L), dtype=torch.bool, device=device)
            bag = torch.full((B, L), S, dtype=torch.long, device=device)
            sidx = torch.zeros((B, S), dtype=torch.long, device=device)
            svalid = torch.zeros((B, S), dtype=torch.bool, device=device)
            cur = torch.empty(B, dtype=torch.long, device=device)
            for i, b in enumerate(builders):
                n = len(b.ids)
                cur[i] = n
                ids[i, :n] = torch.tensor(b.ids, dtype=torch.long, device=device)
                smask[i, :n] = torch.tensor(b.slot_mask, dtype=torch.bool, device=device)
                bg = np.asarray(b.bag_id, dtype=np.int64)
                bg[bg >= b.n_slots] = S            # open span → the dump bin, as in tensors()
                bag[i, :n] = torch.from_numpy(bg).to(device)
                if b.slot_first:
                    sidx[i, :b.n_slots] = torch.tensor(b.slot_first, dtype=torch.long,
                                                       device=device)
                    svalid[i, :b.n_slots] = True
            layout = SlotLayout(slot_mask=smask, bag_id=bag, slot_index=sidx,
                                slot_valid=svalid, prefix_k=spec.prefix_k)
            res = model(ids, slot_layout=layout)
            # Same rule as the single-row generator, applied per row.
            if emit_source == "token":
                back = torch.tensor(
                    [1 + spec.prefix_k if b.slot_mask[-1] else 1 for b in builders],
                    dtype=torch.long, device=device)
            else:
                back = torch.ones(B, dtype=torch.long, device=device)
            last = res["logits"][rows, cur - back]
            for i, b in enumerate(builders):
                nxt = sample_next(last[i], temperature, top_k, gens[i])
                emitted[i].append(nxt)
                b.append(nxt)
    finally:
        if was_training:
            model.train()
    return emitted, builders
