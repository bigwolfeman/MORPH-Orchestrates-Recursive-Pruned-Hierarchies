"""Hop chain documents (lxtul/hops/chain.py), ported from hnet/tests/test_chaintask.py, plus the testbed parts:
the vocabulary picker on the real StarCoder2 tokenizer and the boundary rule.

  1. a document asks half its chains, no statement resolves early, the reader solve() reproduces every answer;
  2. stream and rows hold whole documents, hops land on the answer tokens;
  3. pick_vocab: each fixed word is ONE token, "." cuts a span, 256+ names, no slot token, and it RAISES when
     a fixed word is not one token, "." is not a boundary or too few names exist;
  4. under the real boundary rule every statement and question of a real chain document is exactly one span.

    PYTHONPATH=.:$MORPH_ROOT HF_HUB_OFFLINE=1 python -m pytest -q lxtul/tests/test_hops_chain.py
"""
from __future__ import annotations

import collections
import dataclasses

import numpy as np
import pytest

from lxtul.hops import chain as C
from lxtul.hops.base import FAMILIES, HopEnv
from lxtul.tests.hops_support import real

V = C.ChainVocab(names=np.arange(100, 400), values=np.arange(400, 500), is_=1, stop=2, find=3, now=4, it=5, eos=0)


def test_a_document_asks_half_its_chains_and_no_statement_resolves_early():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lengths = rng.integers(1, 9, int(rng.integers(2, 9)))
        toks, hops = C.document(V, lengths, rng)
        n, K = int(lengths.sum()), len(lengths)
        assert len(toks) == 4 * n + 8 * ((K + 1) // 2) + 1 and toks[-1] == V.eos
        st = toks[: 4 * n].reshape(n, 4)
        assert (st[:, 1] == V.is_).all() and (st[:, 3] == V.stop).all()
        defined_at = {int(left): i for i, left in enumerate(st[:, 0])}
        assert len(defined_at) == n                                              # every name is defined once
        for i, right in enumerate(st[:, 2].tolist()):                            # a right side is a value, or a name
            assert right in V.values or defined_at[right] > i                    # that is defined LATER in the text
        answers = C.solve(V, toks[:-1])
        assert len(answers) == (K + 1) // 2 and sorted(h for _, _, h in answers) == sorted(hops[hops > 0].tolist())
        assert len({v for _, v, _ in answers}) == len(answers)                   # each chain is asked at most once
        for at, value, h in answers:
            assert toks[at] == value and hops[at] == h and value in V.values


def test_stream_and_rows_hold_whole_documents():
    rng = np.random.default_rng(1)
    s = C.stream(V, 5000, (2, 8), [6], rng)
    assert len(s) == 5000 and s.dtype == np.uint16
    toks, hops = C.rows(V, 6, 512, 8, [1, 2, 4, 8, 12], rng)
    assert toks.shape == hops.shape == (6, 513) and (toks[:, 0] == V.eos).all() and not hops[:, -1].any()
    assert set(np.unique(hops).tolist()) <= {0, 1, 2, 4, 8, 12}
    for row, hrow in zip(toks, hops):
        docs = np.split(row[1:], np.flatnonzero(row[1:] == V.eos) + 1)
        seen = 0
        for d in docs:
            if len(d) > 1:
                for at, value, h in C.solve(V, d[:-1]):
                    assert d[at] == value and hrow[1 + seen + at] == h
            seen += len(d)
        assert int((hrow > 0).sum()) == sum(len(C.solve(V, d[:-1])) for d in docs if len(d) > 1) > 0


def test_the_solver_catches_a_wrong_hop_label():
    """Sabotage check: the reader must disagree with a document whose hop label is off by one."""
    rng = np.random.default_rng(3)
    toks, hops = C.document(V, np.array([3, 5]), rng)
    at, _, h = C.solve(V, toks[:-1])[0]
    assert hops[at] == h
    hops[at] += 1
    assert hops[at] != C.solve(V, toks[:-1])[0][2]


# ------------------------------------------------------------------------- the real tokenizer and rule
def test_pick_vocab_on_the_real_tokenizer():
    tok, env, _, _, _ = real()
    v = C.pick_vocab(env)
    assert len(v.names) == 1024 and len(v.values) == 256
    for k, w in C.WORDS.items():                                  # each fixed word is exactly one token
        assert tok(w, add_special_tokens=False)["input_ids"] == [getattr(v, k)]
    assert env.rule.is_boundary[v.stop]
    pool = np.concatenate([v.names, v.values])
    assert env.slot_id not in pool and env.eos not in pool and len(set(pool.tolist())) == len(pool)
    for i in np.concatenate([v.names[:50], v.values[:50]]):       # names and values encode back to their own id
        assert tok(env.token_text[i], add_special_tokens=False)["input_ids"] == [int(i)]
    assert not any(env.rule.is_boundary[pool])                    # a name never cuts a span by itself


def test_pick_vocab_raises_when_the_tokenizer_cannot_carry_the_task():
    _, env, _, _, _ = real()
    with pytest.raises(ValueError, match="one token"):            # " Find" splits into two tokens
        C.pick_vocab(dataclasses.replace(env, encode=lambda s: [1, 2] if s == " Find" else env.encode(s)))
    lut = env.rule.is_boundary.copy()
    lut[env.encode(".")[0]] = False
    with pytest.raises(ValueError, match="boundary"):
        C.pick_vocab(dataclasses.replace(env, rule=dataclasses.replace(env.rule, is_boundary=lut)))
    few = HopEnv(env.token_text[:2000], env.encode, env.rule, env.eos, env.slot_id)
    with pytest.raises(ValueError, match="need 256"):             # under 256 names
        C.pick_vocab(few, n_names=10, n_values=10)


def test_every_statement_and_question_is_one_span_under_the_real_rule():
    _, env, _, _, _ = real()
    v = C.pick_vocab(env)
    src = C.ChainSource(v, (2, 5), [16], "heads_half", 4, [1, 2, 3])
    rng = np.random.default_rng(5)
    for _ in range(40):
        d = src.train_document(rng)
        t = d.tokens
        cuts = env.rule.cut(t, 0)[0]
        stops = np.flatnonzero(t == v.stop)
        assert cuts.tolist() == stops.tolist() + [len(t) - 1]     # a cut after every "." and at EOS, nowhere else
        assert (np.diff(np.concatenate([[-1], stops])) == 4).all()  # every sentence is exactly 4 tokens
        assert (d.family[d.hops > 0] == FAMILIES["chain"]).all()
        assert not (t == env.slot_id).any()


# ----------------------------------------------------------------- ask modes and the size mix
def test_nodes_all_but_one_asks_k_minus_one_chains_once_each_at_a_random_node():
    rng = np.random.default_rng(7)
    seen_below_head = seen_head = False
    for _ in range(300):
        lengths = rng.integers(1, 12, int(rng.integers(1, 7)))
        toks, hops = C.document(V, lengths, rng, ask="nodes_all_but_one")
        n, K = int(lengths.sum()), len(lengths)
        asked = max(K - 1, 1)
        assert len(toks) == 4 * n + 8 * asked + 1 and toks[-1] == V.eos
        answers = C.solve(V, toks[:-1])                         # the reader follows the text, not the generator
        assert len(answers) == asked and len({v for _, v, _ in answers}) == asked   # each chain once
        assert sorted(h for _, _, h in answers) == sorted(hops[hops > 0].tolist())
        for at, value, h in answers:
            assert toks[at] == value and hops[at] == h and 1 <= h <= lengths.max()
        seen_head |= any(h == lengths.max() for _, _, h in answers)
        seen_below_head |= any(h < lengths.max() for _, _, h in answers) and lengths.max() > 1
        if K > 1:
            assert asked < K                                    # one chain stays unasked
    assert seen_head and seen_below_head


def test_nodes_all_but_one_hop_label_is_at_most_the_asked_chains_length():
    rng = np.random.default_rng(8)
    for _ in range(100):
        lengths = np.array([int(rng.integers(1, 9)) for _ in range(3)])
        toks, hops = C.document(V, lengths, rng, ask="nodes_all_but_one")
        n = int(lengths.sum())
        statements = toks[: 4 * n].reshape(n, 4)
        table = {int(a): int(b) for a, _, b, _ in statements}
        for at, value, h in C.solve(V, toks[:-1]):
            q = toks[at - 5]                                    # the asked name: "Find Y now. It is V." sits 6 tokens before V
            assert q in table
            name, steps = int(q), 0
            while name in table:
                name, steps = table[name], steps + 1
            assert steps == h and name == value                 # h lookups from the asked node reach the value


def test_heads_half_is_unchanged_and_len_caps_draws_the_size_mix():
    rng = np.random.default_rng(9)
    lengths = np.array([3, 5, 2])
    a = C.document(V, lengths, np.random.default_rng(1))
    b = C.document(V, lengths, np.random.default_rng(1), ask="heads_half")
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    for _, _, h in C.solve(V, a[0][:-1]):
        assert h in lengths.tolist()                            # asked at a head: the hop label is a chain length
    caps = collections.Counter()
    for _ in range(4000):
        caps[int(C.draw_lengths(rng, (2, 5), [1, 2, 4, 8, 16]).max())] += 1
    assert max(caps) == 16 and caps[1] > 0 and caps[2] > 0                    # mixed sizes, small ones included
    assert set(int(x) for x in np.concatenate([C.draw_lengths(rng, (2, 5), [16]) for _ in range(500)])) == set(range(1, 17))
    with pytest.raises(ValueError, match="ask must be"):
        C.document(V, lengths, rng, ask="all")
