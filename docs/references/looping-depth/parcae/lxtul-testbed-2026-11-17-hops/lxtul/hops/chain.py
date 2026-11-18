"""Hop chain documents: text whose answers need a known number of sequential lookups (hops). Token ids, numpy.

Ported from hnet/chaintask.py (the 2026-10-30 two-stage H-Net smoke; same document, `stream`, `rows`, `solve`).

Pseudocode (one document; K chains, chain k of length l_k; names and values = fresh draws of single-token words):
    chain k:   Y_l is Y_(l-1).  ...  Y_2 is Y_1.  Y_1 is V_k.        # 4 tokens per statement: name " is" name "."
               its statements keep this order (the head first, the value last); the K chains are merged at random,
               so no statement can resolve its right side from the text before it (a causal model must chase
               backward from the question, one lookup per hop)
    questions: `ask="heads_half"` (the eval): a random half of the chains (rounded up), in random order, each at its head:
               Find Y_l now.  It is V_k.                              # 4 + 4 tokens; hops = l_k
               `ask="nodes_all_but_one"` (training): K - 1 chains (at least 1), each ONCE at a uniformly random node
               Y_j (j lookups from the value, 1 <= j <= l); hops = j; the statements above Y_j stay as distractors
    document = statements, questions, EOS
The answer V_k needs l_k lookups: Y_l -> Y_(l-1) -> ... -> Y_1 -> V_k. Each chain is asked at most once, so no
earlier answer in the text shortens a later chase. The chains that are not asked keep the last answer from being
the one value left over. Chance for a model that picks a value of the document: 1 / K.
Every statement and question is a 4-token sentence that ends in "." (the testbed rule: min_span 4, "." cuts),
and the question is a whole sentence BEFORE its answer: a model whose only path between sentences is a coarse loop
sees the asked name one sentence ahead of the answer token.
    stream(n_tokens): documents with K ~ uniform(k_range); a size cap drawn from `len_caps`, l_k ~ uniform(1, cap)
    rows(n, L):       n rows of L + 1 tokens: whole documents of `k` chains with l_k ~ uniform(hop_list), EOS padded;
                      hops[row, i] = l_k where token i is an answer value, else 0
Testbed use: `ChainSource` (the `HopSource` of this file) feeds `hops.base.build_row`; the answer token is the value
V_k, the label position is the " is" before it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lxtul.hops.base import FAMILIES, Document, HopEnv, HopSource, single_token

WORDS = dict(is_=" is", stop=".", find=" Find", now=" now", it=" It")
MIN_NAMES = 256


@dataclass(frozen=True)
class ChainVocab:
    names: np.ndarray        # token ids of single-token names
    values: np.ndarray       # token ids of single-token values (no id is in both pools)
    is_: int
    stop: int
    find: int
    now: int
    it: int
    eos: int

    def __post_init__(self):
        fixed = {self.is_, self.stop, self.find, self.now, self.it, self.eos}
        pools = set(self.names.tolist()) | set(self.values.tolist())
        if len(pools) != len(self.names) + len(self.values) or pools & fixed or len(fixed) != 6:
            raise ValueError("chain vocab: names, values and the fixed words must be distinct token ids")


def pick_vocab(env: HopEnv, n_names: int = 1024, n_values: int = 256, min_names: int = MIN_NAMES) -> ChainVocab:
    """Names: ' Xxxx' capitalised words of 4 to 8 letters. Values: ' xxxx' lower-case words of 4 to 8 letters.
    The first n of each by token id (frequent words), so the choice is a fixed function of the tokenizer.

    Every fixed word must be ONE token under the tokenizer, "." must cut a span under the testbed's boundary
    rule, and every picked word must encode back to its own single id. Takes as many names as exist up to
    `n_names`; raises under `min_names` names or `n_values` values."""
    ids = {}
    for i, t in enumerate(env.token_text):
        if t:
            ids.setdefault(t, i)
    fixed = {k: single_token(env, w, ids.get(w)) for k, w in WORDS.items()}
    if not env.rule.is_boundary[fixed["stop"]]:
        raise ValueError('"." is not a span boundary under the boundary rule: a 4-token statement would not be a span')
    word = lambda t: len(t) in range(5, 10) and t[0] == " " and t[1:].isascii() and t[1:].isalpha()  # noqa: E731
    taken = set(fixed.values()) | {env.eos, env.slot_id}
    names = [i for i, t in enumerate(env.token_text) if word(t) and t[1].isupper() and t[2:].islower() and i not in taken]
    values = [i for i, t in enumerate(env.token_text) if word(t) and t[1:].islower() and i not in taken]
    names, values = _round_trip(env, names, n_names), _round_trip(env, values, n_values)
    if len(names) < min_names or len(values) < n_values:
        raise ValueError(f"chain vocab: {len(names)} names, {len(values)} values; need {min_names}, {n_values}")
    return ChainVocab(np.array(names), np.array(values), eos=env.eos, **fixed)


def _round_trip(env: HopEnv, cand: list[int], n: int) -> list[int]:
    """The first `n` candidates whose text encodes to exactly their own id (a decoded string can map back to
    other ids when two ids share one string)."""
    out = []
    for i in cand:
        if env.encode(env.token_text[i]) == [i]:
            out.append(i)
            if len(out) == n:
                break
    return out


ASK_MODES = ("heads_half", "nodes_all_but_one")


def document(v: ChainVocab, lengths: np.ndarray, rng: np.random.Generator,
             ask: str = "heads_half") -> tuple[np.ndarray, np.ndarray]:
    """lengths (K,) chain lengths -> (tokens, hops), both (4 sum(l) + 8 n_asked + 1,); the last token is EOS.
    n_asked = ceil(K / 2) for `heads_half`, max(K - 1, 1) for `nodes_all_but_one` (see the module docstring)."""
    if ask not in ASK_MODES:
        raise ValueError(f"ask must be one of {ASK_MODES}, got {ask!r}")
    K, n = len(lengths), int(lengths.sum())
    names = v.names[rng.permutation(len(v.names))[:n]]
    values = v.values[rng.permutation(len(v.values))[:K]]
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]])     # chain k owns names[start_k : start_k + l_k], head first
    chain = rng.permutation(np.repeat(np.arange(K), lengths))  # a random merge: statement slot -> its chain
    rank = np.zeros(n, dtype=np.int64)                         # the slot's index inside its chain (stable order)
    order = np.argsort(chain, kind="stable")
    rank[order] = np.arange(n) - np.repeat(start, lengths)
    left = names[start[chain] + rank]
    last = rank == lengths[chain] - 1
    right = np.where(last, values[chain], names[np.minimum(start[chain] + rank + 1, n - 1)])
    stmts = np.stack([left, np.full(n, v.is_), right, np.full(n, v.stop)], axis=1).reshape(-1)
    if ask == "heads_half":
        asked = rng.permutation(K)[: (K + 1) // 2]
        at_hops = lengths[asked]                                   # the head: l lookups
    else:
        asked = rng.permutation(K)[: max(K - 1, 1)]
        at_hops = np.array([int(rng.integers(1, lengths[k] + 1)) for k in asked], dtype=np.int64)   # node j of its chain
    asked_name = names[start[asked] + lengths[asked] - at_hops]    # Y_j sits j - 1 names before the chain's last name
    q = np.stack([np.full(len(asked), v.find), asked_name, np.full(len(asked), v.now), np.full(len(asked), v.stop),
                  np.full(len(asked), v.it), np.full(len(asked), v.is_), values[asked], np.full(len(asked), v.stop)], axis=1)
    qh = np.zeros_like(q)
    qh[:, 6] = at_hops
    toks = np.concatenate([stmts, q.reshape(-1), [v.eos]])
    hops = np.concatenate([np.zeros(4 * n, dtype=np.int64), qh.reshape(-1), [0]])
    return toks.astype(np.uint16), hops.astype(np.uint8)


def draw_lengths(rng: np.random.Generator, k_range: tuple[int, int], len_caps) -> np.ndarray:
    """Chain lengths of one training document: K ~ U(k_range), a cap drawn uniformly from `len_caps`, then each
    length ~ U(1, cap). `len_caps = [16]` is the plain U(1, 16) mix."""
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    cap = int(len_caps[int(rng.integers(0, len(len_caps)))])
    return rng.integers(1, cap + 1, k)


def stream(v: ChainVocab, n_tokens: int, k_range: tuple[int, int], len_caps,
           rng: np.random.Generator, ask: str = "heads_half") -> np.ndarray:
    """At least n_tokens of whole documents, cut to n_tokens."""
    out, have = [], 0
    while have < n_tokens:
        toks, _ = document(v, draw_lengths(rng, k_range, len_caps), rng, ask)
        out.append(toks)
        have += len(toks)
    return np.concatenate(out)[:n_tokens]


def rows(v: ChainVocab, n_rows: int, seq_len: int, k: int, hop_list: list[int],
         rng: np.random.Generator, ask: str = "heads_half") -> tuple[np.ndarray, np.ndarray]:
    """(tokens, hops), both (n_rows, seq_len + 1): whole documents from the row start, then EOS to the row end."""
    toks = np.full((n_rows, seq_len + 1), v.eos, dtype=np.uint16)
    hops = np.zeros((n_rows, seq_len + 1), dtype=np.uint8)
    hl = np.asarray(hop_list)
    for r in range(n_rows):
        at = 1                                                 # token 0 = EOS: every document starts after a separator
        while True:
            t, h = document(v, hl[rng.integers(0, len(hl), k)], rng, ask)
            if at + len(t) > seq_len + 1:
                break
            toks[r, at: at + len(t)], hops[r, at: at + len(t)] = t, h
            at += len(t)
        assert at > 1, f"a document of {k} chains does not fit a row of {seq_len} tokens"
    return toks, hops


def solve(v: ChainVocab, toks: np.ndarray) -> list[tuple[int, int, int]]:
    """Reference reader for tests: [(index of the answer token, the value the text gives, hops)] of one whole document."""
    t = toks.tolist()
    table, out, i = {}, [], 0
    while i + 3 < len(t):
        if t[i] == v.find:
            name, n = t[i + 1], 0
            while name in table:
                name, n = table[name], n + 1
            out.append((i + 6, name, n))
            i += 8
        else:
            table[t[i]] = t[i + 2]
            i += 4
    return out


class ChainSource(HopSource):
    """`task.sources.chain`: K ~ U(k_range) chains per training document, a size cap drawn from `len_caps`, chain
    lengths ~ U(1, cap), questions in mode `ask` (training: `nodes_all_but_one`); evaluation buckets hold
    documents of `eval_k` chains that all have one length from `eval_hops`, always asked `heads_half`."""
    name = "chain"
    KEYS = {"row_share", "k_range", "len_caps", "ask", "n_names", "n_values"}

    def __init__(self, vocab: ChainVocab, k_range, len_caps, ask: str, eval_k: int, eval_hops):
        self.v, self.n_names = vocab, len(vocab.names)
        self.k_range, self.len_caps, self.ask = tuple(int(x) for x in k_range), [int(c) for c in len_caps], str(ask)
        self.eval_k, self.eval_hops = int(eval_k), [int(h) for h in eval_hops]
        if min(self.k_range) < 1 or not self.len_caps or min(self.len_caps) < 1:
            raise ValueError("chain: k_range must start at 1 or more and len_caps must be a non-empty list of 1 or more")
        if self.ask not in ASK_MODES:
            raise ValueError(f"chain: ask must be one of {ASK_MODES}, got {self.ask!r}")
        if max(self.eval_hops + self.len_caps) > 255:
            raise ValueError("chain: hop counts must fit uint8")

    @classmethod
    def from_cfg(cls, cfg: dict, eval_cfg: dict, env: HopEnv) -> "ChainSource":
        unknown = set(cfg) - cls.KEYS
        if unknown:
            raise ValueError(f"unknown task.sources.chain keys: {sorted(unknown)}")
        vocab = pick_vocab(env, int(cfg["n_names"]), int(cfg["n_values"]))
        return cls(vocab, cfg["k_range"], cfg["len_caps"], cfg["ask"], eval_cfg["k"], eval_cfg["hops"])

    def _doc(self, lengths: np.ndarray, rng: np.random.Generator, ask: str) -> Document:
        toks, hops = document(self.v, lengths, rng, ask)
        return Document(toks.astype(np.int64), hops, np.where(hops > 0, FAMILIES["chain"], 0).astype(np.uint8))

    def train_document(self, rng):
        return self._doc(draw_lengths(rng, self.k_range, self.len_caps), rng, self.ask)

    def eval_buckets(self):
        return [(f"chain/h{h}", lambda rng, h=h: self._doc(np.full(self.eval_k, h), rng, "heads_half")) for h in self.eval_hops]
