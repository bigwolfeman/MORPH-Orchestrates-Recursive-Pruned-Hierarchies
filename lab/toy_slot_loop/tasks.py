"""Two synthetic span tasks for the toy slot loop.

Both tasks pack a row as n_spans spans of span_len symbols. The model never sees the
answer as an INPUT token: the answer alphabet is output-only, and it appears only as the
LABEL at the first token position of the NEXT span. That is the double-label shape of the
real model (spec sec 5) and it is what keeps the answer off the token path.

Vocabulary (12 ids):
  0..5   op symbols   -- the six elements of S_3, the only ids that ever appear as inputs
  6..11  value symbols -- output only, the answer alphabet

Task "compose" (iterative state tracking).
  Each span is span_len group elements. The span product P_i is their ordered product in
  S_3. The register is R_i = R_{i-1} . P_i with R_{-1} = e. The label at the first token
  of span i+1 is VALUE(R_i). Composition in S_3 is not commutative, so no single
  bag/average step computes R_i from the raw symbols: slot i must combine its own span's
  product with the running register carried by earlier slots, and one round of slot->slot
  attention advances that carry by a bounded number of spans.

Task "summary" (one-pass control).
  The label at the first token of span i+1 is VALUE(mode of span i's symbols), ties to the
  smallest id. Slot i's own prelude already sees every symbol of span i, so this needs no
  iteration at all.

Every other token position carries the ordinary next-symbol label, which is uniform noise
by construction (CE floor ln 6 = 1.7918). That mirrors the real model, where most of the
token CE is local statistics the loop cannot help with.
"""

from __future__ import annotations

import torch

# S_3 as permutations of (0,1,2). Index = symbol id.
PERMS = [
    (0, 1, 2),  # e
    (1, 2, 0),  # r
    (2, 0, 1),  # r^2
    (0, 2, 1),  # s
    (2, 1, 0),  # rs
    (1, 0, 2),  # r^2 s
]
N_GROUP = len(PERMS)
VALUE_BASE = N_GROUP
VOCAB = N_GROUP + N_GROUP


def _mul_table() -> torch.Tensor:
    """t[a, b] = index of the permutation a then b (apply a, then b)."""
    t = torch.empty(N_GROUP, N_GROUP, dtype=torch.long)
    for a in range(N_GROUP):
        for b in range(N_GROUP):
            comp = tuple(PERMS[b][PERMS[a][i]] for i in range(3))
            t[a, b] = PERMS.index(comp)
    return t


MUL = _mul_table()


def make_batch(
    task: str,
    B: int,
    n_spans: int,
    span_len: int,
    generator: torch.Generator | None = None,
    device="cpu",
) -> dict[str, torch.Tensor]:
    """Return tokens [B, n_spans*span_len], labels (same shape), mux_next/mux_own [B, n_spans]."""
    mul = MUL.to(device)
    toks = torch.randint(0, N_GROUP, (B, n_spans, span_len), generator=generator, device=device)

    if task == "compose":
        span_prod = toks[:, :, 0].clone()
        for j in range(1, span_len):
            span_prod = mul[span_prod, toks[:, :, j]]
        reg = torch.empty_like(span_prod)
        acc = torch.zeros(B, dtype=torch.long, device=device)  # identity
        for i in range(n_spans):
            acc = mul[acc, span_prod[:, i]]
            reg[:, i] = acc
        answer_next = reg  # slot i's target: the register after span i
        answer_own = span_prod  # slot i's own-span target: this span's product
    elif task == "summary":
        counts = torch.zeros(B, n_spans, N_GROUP, dtype=torch.long, device=device)
        counts.scatter_add_(2, toks, torch.ones_like(toks))
        answer_next = counts.argmax(dim=2)  # ties -> smallest id (argmax picks first max)
        answer_own = answer_next
    else:
        raise ValueError(task)

    flat = toks.reshape(B, n_spans * span_len)
    labels = torch.full_like(flat, -100)
    labels[:, :-1] = flat[:, 1:]  # ordinary next-symbol label
    labels[:, -1] = -100
    # overwrite the first token of each span i+1 with the answer of span i
    for i in range(1, n_spans):
        labels[:, i * span_len] = VALUE_BASE + answer_next[:, i - 1]

    value_pos = torch.tensor([i * span_len for i in range(1, n_spans)], device=device)
    return {
        "tokens": flat,
        "labels": labels,
        "mux_next": VALUE_BASE + answer_next,
        "mux_own": VALUE_BASE + answer_own,
        "value_pos": value_pos,
    }


def chance_ce() -> float:
    import math

    return math.log(N_GROUP)
