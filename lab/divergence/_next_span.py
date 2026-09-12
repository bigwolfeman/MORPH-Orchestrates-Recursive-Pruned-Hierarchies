"""WHICH row positions a slot is answerable for, and at what offset.

The slot loop's job is the NEXT span: slot ``s`` thinks about span ``s + 1``
(``morph/model/tul_spandec.py::next_span_slots``, ``mux_target="next"``). Two of the
2026-09-12 causal instruments — ``slot_z_causal_fit.py`` and ``slot_depth_isolation.py``
— have to turn that relation into ROW POSITIONS, because they score, replace or mask the
model's own token stream rather than a per-slot gather. One home for the arithmetic, so
the two cannot drift apart.

THE CONVENTION, and it is the tree's (``span_budget_profile.py``): the CE at position
``p`` is the cost of predicting the token at ``p + 1``, and the OFFSET reported is the
offset of that PREDICTED token inside its own span. Offset 0 is therefore a span's first
token, predicted from the LAST TOKEN OF THE PREVIOUS SPAN — a position that sits before
the slot cell, not after it. Getting this wrong shifts the whole profile by one token and
silently drops the 0.96-nat first-position spike
(``lab/experiments/failures/2026-09-11-arc-span-budget.md``).

So for slot ``s`` and its span ``s + 1`` of length ``n``:

    offset 0      -> row position ``slot_index[s] - 1``  (span s's boundary token)
    offset j >= 1 -> row position ``slot_index[s] + prefix_k + (j - 1)``

and ``n = slot_index[s+1] - slot_index[s] - prefix_k``. The slot's own emitting cell
carries the same label as offset 0 but is a SLOT position: it is never scored here
(``tul.emit_weight`` is 0 on every arm in this arc).

VALIDITY is :func:`morph.model.tul_spandec.span_slots`' rule at ``shift=1``, unchanged:
slot ``s`` must exist AND slot ``s + 1`` must exist (which is what proves span ``s + 1``
complete). The row's open tail after the last slot is never a target.

The mapping is CHECKED, not asserted by reading: :func:`next_span_positions` verifies
that ``labels.gather(1, pos)`` reproduces ``next_span_slots``' ids token for token on
every valid entry. That is a two-sided check — it fails if the offset convention slips by
one in either direction.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["next_span_positions", "next_span_mask"]


def next_span_positions(layout, labels: Tensor, input_ids: Tensor, max_tokens: int = 0
                        ) -> tuple[Tensor, Tensor]:
    """``(pos [B, S, J] long, valid [B, S, J] bool)`` for every slot's NEXT span.

    ``pos[b, s, j]`` is the row position whose LABEL is token ``j`` of span ``s + 1``;
    ``valid[b, s, j]`` says whether that token exists. Invalid entries address position 0
    and must be masked by the caller, never read.

    ``max_tokens`` (0 -> the layout's own span cap, i.e. the longest span in the batch)
    caps ``J``. The check against :func:`~morph.model.tul_spandec.next_span_slots` runs on
    the same ``J``, so a smaller cap is checked on the prefix it keeps.
    """
    from morph.model.tul_spandec import next_span_slots

    idx = layout.slot_index                                   # [B, S]
    val = layout.slot_valid                                   # [B, S]
    B, S = idx.shape
    K = int(layout.prefix_k)
    dev = idx.device
    # slot s is graded on span s+1, so BOTH slots must exist (span_slots' shift=1 rule).
    nxt_val = torch.cat([val[:, 1:], torch.zeros_like(val[:, :1])], dim=1)
    nxt_idx = torch.cat([idx[:, 1:], torch.zeros_like(idx[:, :1])], dim=1)
    pair = val & nxt_val                                      # [B, S]
    n_tok = (nxt_idx - idx - K).clamp(min=0)                  # span s+1's token count
    n_tok = torch.where(pair, n_tok, torch.zeros_like(n_tok))
    J = int(max_tokens) if max_tokens > 0 else max(1, int(n_tok.max().item()))
    j = torch.arange(J, device=dev).view(1, 1, J)
    valid = pair.unsqueeze(-1) & (j < n_tok.unsqueeze(-1))
    # offset 0 sits BEFORE the slot cell (the previous span's boundary token); every
    # later offset sits after the prefix cells.
    first = (idx - 1).clamp(min=0).unsqueeze(-1)                          # [B, S, 1]
    rest = (idx + K).unsqueeze(-1) + (j - 1)                             # [B, S, J]
    pos = torch.where(j == 0, first, rest)
    pos = torch.where(valid, pos, torch.zeros_like(pos))

    # THE CHECK. The production relation says WHICH tokens; this says WHICH positions
    # predict them. They must agree token for token, or the offset convention has slipped.
    ids_ref, val_ref = next_span_slots(input_ids, layout, J)
    if not torch.equal(val_ref, valid):
        raise RuntimeError(
            "next_span_positions disagrees with next_span_slots about WHICH (slot, "
            "offset) pairs exist — the layout arithmetic here is wrong, not the rule.")
    got = labels.gather(1, pos.reshape(B, S * J)).reshape(B, S, J)
    if not torch.equal(got[valid], ids_ref[valid]):
        n_bad = int((got[valid] != ids_ref[valid]).sum())
        raise RuntimeError(
            f"next_span_positions: {n_bad} of {int(valid.sum())} positions do not predict "
            "the token next_span_slots names. The CE at p predicts the token at p+1; a "
            "mismatch means the offset convention slipped by one.")
    return pos, valid


def next_span_mask(layout, labels: Tensor, input_ids: Tensor, max_tokens: int = 0
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """``(pos, valid, slot_of [B, L] long)`` — the per-position map back to its slot.

    ``slot_of[b, p]`` is the slot whose next span the label at ``p`` belongs to, or ``-1``
    where no slot is answerable for it (span 0's tokens, the dump bin, the last span, pad
    positions and every slot cell). The instruments use it to score one slot's column, and
    the union over slots is exactly "every position a slot could ever have helped".
    """
    pos, valid = next_span_positions(layout, labels, input_ids, max_tokens)
    B, S, _J = pos.shape
    L = labels.shape[1]
    s_idx = torch.arange(S, device=pos.device).view(1, S, 1).expand_as(pos)
    flat_pos = torch.where(valid, pos, torch.full_like(pos, L))
    out = labels.new_full((B, L + 1), -1)
    out.scatter_(1, flat_pos.reshape(B, -1), s_idx.reshape(B, -1))
    slot_of = out[:, :L]
    return pos, valid, slot_of
