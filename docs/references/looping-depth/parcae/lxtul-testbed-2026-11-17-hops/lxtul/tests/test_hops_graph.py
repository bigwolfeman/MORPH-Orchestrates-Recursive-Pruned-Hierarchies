"""GraphQA-style graph documents (lxtul/hops/graph.py) on the real StarCoder2 tokenizer and boundary rule.

  1. the name pool: 64+ names, each one token in BOTH the sentence-initial and the mid-sentence form;
  2. a document decodes to the GraphQA text and encodes back to the same ids (whole-text tokenization agrees
     with the pieces it was built from);
  3. under the boundary rule every edge sentence, every question sentence and every answer sentence is its own
     span, the labelled answer is exactly one token, EOS closes the document, the slot token never occurs;
  4. the reference reader (BFS on the edges parsed from the token ids ALONE) reproduces every answer token,
     hop label and family of the generator, and it disagrees with a corrupted document;
  5. forests have two trees; questions are balanced and stratified by distance; no pair is asked twice;
  6. the eval buckets ask only their own question, with a path of the bucket's length built into the forest.

    PYTHONPATH=.:$MORPH_ROOT HF_HUB_OFFLINE=1 python -m pytest -q lxtul/tests/test_hops_graph.py
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from lxtul.hops import graph as G
from lxtul.hops.base import FAMILIES, FAMILY_NAME
from lxtul.tests.hops_support import real, real_task


@pytest.fixture(scope="module")
def src():
    return next(s for s in real_task().sources if s.name == "graph")


def _docs(src, n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [src.train_document(rng) for _ in range(n)]


def test_name_pool_is_single_token_in_both_forms():
    tok, env, _, _, _ = real()
    v = G.pick_vocab(env, 128)
    assert len(v.init) == 128 >= G.MIN_NAMES
    for a, b in zip(v.init[:128], v.mid[:128]):
        t0, t1 = env.token_text[a], env.token_text[b]
        assert t1 == " " + t0 and tok(t0, add_special_tokens=False)["input_ids"] == [int(a)]
        assert tok(t1, add_special_tokens=False)["input_ids"] == [int(b)]
    pool = {env.token_text[i] for i in v.init}
    kept = [n for n in G.GRAPHQA_NAMES if n in pool]
    assert kept == ["James", "John", "Michael", "David"], kept   # the StarCoder2 vocabulary keeps 4 of GraphQA's 20
    assert [env.token_text[i] for i in v.init[:4]] == kept       # and they lead the pool
    with pytest.raises(ValueError, match="need 5000"):
        G.pick_vocab(env, 128, min_names=5000)
    assert len(G.pick_vocab(env, 8, min_names=8).init) == 8


def test_documents_decode_to_graphqa_text_and_encode_back():
    tok, _, _, _, _ = real()
    for d in _docs(next(s for s in real_task().sources if s.name == "graph"), n=20):
        text = tok.decode(d.tokens[:-1])
        assert text.startswith("G describes a friendship graph among nodes ")
        assert "\nWe have the following edges in G:\n" in text and text.count(" are friends.") >= 2
        assert "\nQ: " in text and "\nA: " in text
        assert tok(text, add_special_tokens=False)["input_ids"] == d.tokens[:-1].tolist()
        assert d.tokens[-1] == real()[1].eos


def test_every_sentence_is_one_span_and_the_answer_is_one_token(src):
    _, env, _, _, _ = real()
    v, rule = src.v, env.rule
    nl = v.p["nl"][0]
    for d in _docs(src, n=40, seed=1):
        t = d.tokens
        cuts = set(rule.cut(t, 0)[0].tolist())
        assert not (t == env.slot_id).any() and t[-1] == env.eos
        starts = np.flatnonzero(np.isin(t, v.init))              # a sentence-initial name starts an edge sentence
        # the header may be cut by the 32-token cap before its "."; everything after it is sentence by sentence
        dots = np.flatnonzero(t == v.p["dot"][0])
        dots = dots[dots >= starts[0]]
        qmarks = np.flatnonzero(t == v.p["qmark"][0])
        # an edge, a question and an answer sentence each end in "." or "?", and that is a cut; so is EOS
        assert set(dots.tolist()) | set(qmarks.tolist()) | {len(t) - 1} <= cuts
        for s in starts:
            e = int(dots[np.searchsorted(dots, s)])
            assert not (set(range(s, e)) & cuts), "a cut inside an edge sentence"
            prev = max([c for c in cuts if c < s], default=-1)
            assert all(t[i] == nl for i in range(prev + 1, s)), "something but a newline before an edge sentence"
        for q in np.flatnonzero((t == v.p["q_reach"][0]) | (t == v.p["q_sp"][0]) | (t == v.p["q_edge"][0])):
            e = int(qmarks[np.searchsorted(qmarks, q)])          # Q ... ? is one span (a newline may open it)
            assert not (set(range(q, e)) & cuts)
            prev = max(c for c in cuts if c < q)
            assert all(t[i] == nl for i in range(prev + 1, q))
            e2 = int(dots[np.searchsorted(dots, e)])             # then "A: ... ." is the next span
            assert not (set(range(e + 1, e2)) & cuts) and e in cuts and e2 in cuts
        ans = np.flatnonzero(d.hops > 0)
        assert len(ans) == int((d.family > 0).sum()) >= 1
        for a in ans:                                            # exactly one token: the answer, then "."
            assert t[a + 1] == v.p["dot"][0] and (t[a] in (v.p["yes"][0], v.p["no"][0]) or t[a] in v.digits)


def test_the_reference_reader_reproduces_every_answer_label_and_family(src):
    for d in _docs(src, n=50, seed=2):
        solved = G.solve(src.v, d.tokens)
        assert [a for a, *_ in solved] == np.flatnonzero(d.hops > 0).tolist()
        for at, tokid, hop, fam in solved:
            assert d.tokens[at] == tokid and d.hops[at] == hop and d.family[at] == fam
    yes, no = src.v.p["yes"][0], src.v.p["no"][0]                # a flipped Yes / No: the reader still says the truth
    d = _docs(src, n=1, seed=9)[0]
    solved = G.solve(src.v, d.tokens)
    at, truth = next((a, t) for a, t, *_ in solved if t in (yes, no))
    bad = d.tokens.copy()
    bad[at] = no if truth == yes else yes
    assert [x[1] for x in G.solve(src.v, bad)] == [x[1] for x in solved] and bad[at] != truth


def test_forests_questions_are_balanced_stratified_and_not_repeated(src):
    rng = np.random.default_rng(4)
    for _ in range(30):
        g = G.Forest(rng, int(rng.integers(8, 41)))
        assert len(g.edges) == g.n - 2 and set(g.comp.tolist()) == {0, 1}
        assert all(g.comp[a] == g.comp[b] for a, b in g.edges)
        assert (g.dist[g.comp[:, None] != g.comp[None]] == -1).all()
    yes = collections.Counter()
    dist = collections.Counter()
    for d in _docs(src, n=300, seed=5):
        fams = [FAMILY_NAME[int(f)] for f in d.family[d.family > 0]]
        yes.update(fams)
        dist.update(int(h) for h, f in zip(d.hops[d.family > 0], d.family[d.family > 0]) if f == FAMILIES["sp"])
        text = d.tokens
        asked = []
        for at, *_ in G.solve(src.v, text):
            q0 = int(np.flatnonzero(text[:at] == src.v.p["nl"][0])[-2])   # the question's own newline
            asked.append(frozenset(text[q0:at][np.isin(text[q0:at], src.v.mid)].tolist()))
        assert len(asked) == len(set(asked)), "a pair was asked twice"
    assert 0.7 < yes["reach_yes"] / yes["reach_no"] < 1.4 and 0.7 < yes["edge_yes"] / yes["edge_no"] < 1.4
    assert set(dist) == set(range(1, 10))                        # every shortest-path distance of the range appears
    g = G.Forest(rng, 40, spine=9)                               # a forest that has every distance 1 .. 9
    have = sorted(d for d in g.pairs if d <= 9)
    assert have == list(range(1, 10))
    drawn = collections.Counter(G.ask(rng, g, "sp", set(), (1, 9)).hop for _ in range(1800))
    assert set(drawn) == set(have) and all(0.6 * 200 < c < 1.4 * 200 for c in drawn.values()), drawn    # uniform in d


def test_eval_documents_mix_their_questions_and_every_answer_bucket_is_filled(src):
    """Buckets are per answer, never per document: every document mixes Yes and No questions of several
    distances in random order, and the planted path makes distance 24 exist."""
    (name, make), = src.eval_buckets()
    assert name == "graph/mixed" and src.eval_rows == 120
    rng = np.random.default_rng(6)
    count = collections.Counter()
    for _ in range(60):
        d = make(rng)
        solved = G.solve(src.v, d.tokens)
        fams = collections.Counter(FAMILY_NAME[f] for *_, f in solved)
        assert len(solved) == sum(src.ev["per_doc"].values())
        assert sum(fams[f] for f in ("reach_yes", "reach_no")) == 6 and sum(fams[f] for f in ("edge_yes", "edge_no")) == 4
        assert fams["sp"] == 4
        count.update(["yes"] * fams["reach_yes"] + ["no"] * fams["reach_no"])
        assert len({(h, f) for _, _, h, f in solved if f != FAMILIES["reach_no"] and f != FAMILIES["edge_no"]}) > 3
        count.update((FAMILY_NAME[f], h) for _, _, h, f in solved if f in (FAMILIES["reach_yes"], FAMILIES["sp"]))
        order = [FAMILY_NAME[f] for *_, f in solved]
        assert order != sorted(order), "questions must come in random order"
    assert abs(count["yes"] / (count["yes"] + count["no"]) - 0.5) < 4 * np.sqrt(0.25 / 360)    # a fair coin
    for dd in src.ev["reach_dists"]:
        assert count[("reach_yes", dd)] >= 8, (dd, count[("reach_yes", dd)])
    for dd in src.ev["sp_dists"]:
        assert count[("sp", dd)] >= 12, (dd, count[("sp", dd)])


def test_a_path_longer_than_the_forest_allows_raises():
    with pytest.raises(ValueError, match="cannot hold a path"):
        G.Forest(np.random.default_rng(0), 10, spine=9)


def test_training_documents_mix_sizes_scale_questions_and_build_at_n_4_and_6(src):
    assert src.n_choices == [4, 6, 8, 12, 16, 24, 32, 40] and src.questions_max == 12
    rng = np.random.default_rng(10)
    sizes, per_n = collections.Counter(), collections.defaultdict(list)
    kinds = collections.defaultdict(collections.Counter)
    for _ in range(600):
        d = src.train_document(rng)
        solved = G.solve(src.v, d.tokens)                        # the reader parses every document it is given
        text_nodes = len(_header_names(src, d.tokens))
        sizes[text_nodes] += 1
        per_n[text_nodes].append(len(solved))
        for _, _, _, f in solved:
            kinds[text_nodes][FAMILY_NAME[f]] += 1
    assert set(sizes) == {4, 6, 8, 12, 16, 24, 32, 40}, sizes
    for n, counts in per_n.items():
        want = min(max(n // 3, 2), 12)
        assert max(counts) <= want and np.mean(counts) > 0.6 * want, (n, want, np.mean(counts))
    for n in (4, 6):                                             # two trees on 4 and 6 nodes: every task can be drawn
        assert {"reach_yes", "reach_no", "sp", "edge_yes", "edge_no"} <= set(kinds[n]), (n, kinds[n])


def _header_names(src, toks):
    """Node names listed in the header: the mid-form name ids before the first edge line."""
    stop = int(np.flatnonzero(toks == src.v.p["dot"][0])[0])
    return [t for t in toks[:stop + 1] if t in set(src.v.mid.tolist())]


def test_a_small_graph_never_asks_the_same_pair_twice_and_stops_early_when_nothing_is_left(src):
    rng = np.random.default_rng(11)
    for _ in range(300):
        g = G.Forest(rng, 4, spine=1)
        used, qs = set(), []
        for _ in range(40):
            q = G.ask(rng, g, "edge", used, (1, 1), yes=bool(rng.random() < 0.5))
            if q is not None:
                qs.append(q)
        assert len({frozenset((q.u, q.v)) for q in qs}) == len(qs) == 6      # all C(4,2) pairs, once each
    docs = [src.train_document(np.random.default_rng(i)) for i in range(300)]
    for d in docs:
        asked = [at for at, *_ in G.solve(src.v, d.tokens)]
        assert asked and len(asked) <= 12
