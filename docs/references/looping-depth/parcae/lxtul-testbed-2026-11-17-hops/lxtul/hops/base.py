"""The common interface of a hop-task source, the row builder and the packer glue.

A source makes whole DOCUMENTS: token ids plus two per-token arrays that mark the TARGET token of an answer
(the token a model must predict): `hops` (0 = not an answer; h > 0 = the answer needs h hops) and `family`
(0 = not an answer; else an id of FAMILIES). A row is whole documents from its start (as many as fit under
BOTH `seq_len` tokens and `max_spans` spans by the boundary rule), then padding. `pack_row` hands a row to
MORPH's own packer (`pack_tul_row`) on a spec that fits the row exactly, then extends the result with the
packer's tail-pad convention (input slot_id, label -100, slot_mask True, no slot), so a short row is the
packer's output plus pad and never carries filler tokens or extra spans. `plain_row` is the same documents
as an `(x, y)` row for the flat and token-loop arms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

import lxtul.data  # noqa: F401  (puts MORPH_ROOT on sys.path)
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_row

# Answer families. A family names WHAT the answer is, the hop label says how many lookups it needs.
FAMILIES = {"chain": 1, "reach_yes": 2, "reach_no": 3, "sp": 4, "edge_yes": 5, "edge_no": 6}
FAMILY_NAME = {v: k for k, v in FAMILIES.items()}


@dataclass(frozen=True)
class Document:
    """One document: `tokens` ends in EOS; `hops`, `family` are per token (the target token of an answer)."""
    tokens: np.ndarray       # [n] int64
    hops: np.ndarray         # [n] uint8
    family: np.ndarray       # [n] uint8

    def __post_init__(self):
        if not (self.tokens.shape == self.hops.shape == self.family.shape and self.tokens.ndim == 1):
            raise ValueError("a document's tokens, hops and family must be 1-D arrays of one length")
        if ((self.hops > 0) != (self.family > 0)).any():
            raise ValueError("hops and family must mark the same positions")


@dataclass(frozen=True)
class HopRow:
    """Whole documents laid end to end, no padding (the packers pad)."""
    tokens: np.ndarray
    hops: np.ndarray
    family: np.ndarray

    @property
    def n_answers(self) -> int:
        return int((self.hops > 0).sum())


@dataclass(frozen=True)
class HopEnv:
    """What a source needs from the testbed: the tokenizer's token strings and encoder, the boundary rule."""
    token_text: list[str]                         # id -> decoded string
    encode: Callable[[str], list[int]]            # text -> ids, no special tokens
    rule: BoundaryRule
    eos: int
    slot_id: int


class HopSource(ABC):
    """A document factory. `train_document` draws from the training ranges; `eval_buckets` lists the fixed
    evaluation buckets as (name, document factory); every factory takes only an rng."""
    name: str

    @abstractmethod
    def train_document(self, rng: np.random.Generator) -> Document: ...

    @abstractmethod
    def eval_buckets(self) -> list[tuple[str, Callable[[np.random.Generator], Document]]]: ...


def n_spans(rule: BoundaryRule, tokens: np.ndarray) -> int:
    return int(rule.cut(tokens.astype(np.int64), 0)[0].shape[0])


def build_row(make_doc: Callable[[np.random.Generator], Document], rng: np.random.Generator, seq_len: int,
              max_spans: int, rule: BoundaryRule) -> HopRow:
    """Whole documents from the row start while both budgets hold: tokens <= seq_len and spans <= max_spans.
    A document ends in EOS, which the rule always cuts, so spans add up document by document. The first
    document that does not fit is dropped and ends the row. A first document that does not fit raises."""
    toks, hops, fams = [], [], []
    n_tok = n_sp = 0
    while True:
        d = make_doc(rng)
        s = n_spans(rule, d.tokens)
        if n_tok + len(d.tokens) > seq_len or n_sp + s > max_spans:
            break
        toks.append(d.tokens), hops.append(d.hops), fams.append(d.family)
        n_tok, n_sp = n_tok + len(d.tokens), n_sp + s
    if not toks:
        raise ValueError(f"a document of {len(d.tokens)} tokens / {s} spans does not fit a row of "
                         f"{seq_len} tokens / {max_spans} spans")
    return HopRow(np.concatenate(toks), np.concatenate(hops), np.concatenate(fams))


def _tail(a: np.ndarray, length: int, fill) -> np.ndarray:
    out = np.full(length, fill, dtype=a.dtype)
    out[: a.shape[0]] = a
    return out


def pack_row(row: HopRow, rule: BoundaryRule, spec: TulLayoutSpec) -> dict[str, np.ndarray]:
    """One packed row of `spec` (arrays as `pack_tul_row`, plus `hops` and `family` aligned to the LABELS:
    position p carries the hop label of the token that position predicts).

    The tokens go through `pack_tul_row` on a spec of exactly their own size (N tokens, their own span
    count), which consumes all of them and creates no pad. The result is then extended to `spec`'s shape
    with the packer's tail-pad convention. The row's last token and the slot after it have no next token
    to predict: their labels are -100 (the packer would label them with its one-token peek)."""
    N, K, S = len(row.tokens), spec.prefix_k, spec.max_slots
    ns = n_spans(rule, row.tokens)
    if N > spec.seq_len or ns > S:
        raise ValueError(f"row of {N} tokens / {ns} spans exceeds the spec ({spec.seq_len} / {S})")
    ids = row.tokens.astype(np.int64)
    tight = TulLayoutSpec(seq_len=N, prefix_k=K, max_slots=ns, slot_id=spec.slot_id)
    a, used, _ = pack_tul_row(np.append(ids, ids[0]), rule, tight)     # ids[0]: the label peek, never a cut
    if used != N or tight.l_total != a["input_ids"].shape[0]:
        raise AssertionError("the packer did not consume exactly the row's tokens")
    tok_pos = np.flatnonzero(~a["slot_mask"])
    emit_last = int(a["slot_index"][ns - 1]) + K - 1
    if a["labels"][tok_pos[-1]] != ids[0] or a["labels"][emit_last] != ids[0]:
        raise AssertionError("the packer's peek label is not where the row's end was expected")
    a["labels"][tok_pos[-1]] = -100
    a["labels"][emit_last] = -100
    hops = np.zeros(tight.l_total, dtype=np.uint8)
    fam = np.zeros(tight.l_total, dtype=np.uint8)
    hops[tok_pos[:-1]], fam[tok_pos[:-1]] = row.hops[1:], row.family[1:]
    if (a["bag_id"] == ns).any():
        raise AssertionError("a tight row has no dump-bin position")
    L = spec.l_total
    return {
        "input_ids": _tail(a["input_ids"], L, spec.slot_id), "labels": _tail(a["labels"], L, -100),
        "slot_mask": _tail(a["slot_mask"], L, True), "bag_id": _tail(a["bag_id"], L, S),
        "slot_index": _tail(a["slot_index"], S, 0), "slot_valid": _tail(a["slot_valid"], S, False),
        "hops": _tail(hops, L, 0), "family": _tail(fam, L, 0),
    }


def widen_batch(batch, spec: TulLayoutSpec):
    """A stock TUL batch `(x, y, SlotLayout)` laid out at `spec`'s wider shape (same seq_len and prefix_k, more
    slots): the same tokens, slots and labels at the same positions, then more tail pads at the END of each row,
    in the packer's convention (input slot_id, label -100, slot_mask True, bag_id = the dump bin, no slot_index
    entry). The dump bin moves from the stock slot count to `spec.max_slots`; per-slot arrays are padded with
    0 / False (invalid slots), `ditto_prev` with -1. The model reads pads as inert, so a widened row gives the
    stock row's hidden states at every real position (tests/test_hops_widen.py)."""
    x, y, lay = batch
    B, L0 = x.shape
    K, S0, S1, L1 = int(lay.prefix_k), int(lay.slot_index.shape[1]), spec.max_slots, spec.l_total
    if K != spec.prefix_k or L0 - K * S0 != spec.seq_len or S1 < S0:
        raise ValueError(f"cannot widen a row of {L0} positions / {S0} slots / prefix_k {K} to {spec}")
    if L1 == L0:
        return batch

    def tail(t, length, fill):
        out = t.new_full((B, length) + tuple(t.shape[2:]), fill)
        out[:, : t.shape[1]] = t
        return out
    opt = lambda t, length, fill: None if t is None else tail(t, length, fill)      # noqa: E731
    bag = tail(torch.where(lay.bag_id == S0, torch.full_like(lay.bag_id, S1), lay.bag_id), L1, S1)
    new = type(lay)(slot_mask=tail(lay.slot_mask, L1, True), bag_id=bag, slot_index=tail(lay.slot_index, S1, 0),
                    slot_valid=tail(lay.slot_valid, S1, False), prefix_k=K, stats=lay.stats,
                    span_len=opt(lay.span_len, S1, 0), len_supervised=opt(lay.len_supervised, S1, False),
                    ditto_prev=opt(lay.ditto_prev, L1, -1), cell_roles=lay.cell_roles)
    return tail(x, L1, spec.slot_id), tail(y, L1, -100), new


def plain_row(row: HopRow, seq_len: int, pad_id: int) -> dict[str, np.ndarray]:
    """The same documents as a flat `(x, y)` row of `seq_len`: y is x shifted left, -100 on the pad tail
    (and on the last token), `hops` and `family` aligned to y."""
    N = len(row.tokens)
    if N > seq_len:
        raise ValueError(f"row of {N} tokens exceeds seq_len {seq_len}")
    ids = row.tokens.astype(np.int64)
    y = np.full(seq_len, -100, dtype=np.int64)
    y[: N - 1] = ids[1:]
    hops, fam = np.zeros(seq_len, dtype=np.uint8), np.zeros(seq_len, dtype=np.uint8)
    hops[: N - 1], fam[: N - 1] = row.hops[1:], row.family[1:]
    return {"x": _tail(ids, seq_len, pad_id), "y": y, "hops": hops, "family": fam}


def token_texts(tokenizer, n: int | None = None) -> list[str]:
    """id -> decoded string, as MORPH's boundary table reads it (`boundary_lut_from_tokenizer`)."""
    n = len(tokenizer) if n is None else n
    return tokenizer.batch_decode([[i] for i in range(n)])


def single_token(env: HopEnv, text: str, expect: int | None = None) -> int:
    """The id of `text` when it is exactly one token (and `expect`, when given); else raise, naming the text."""
    ids = env.encode(text)
    if len(ids) != 1 or (expect is not None and ids[0] != expect):
        raise ValueError(f"{text!r} must be one token under the tokenizer, got {ids}")
    return ids[0]
