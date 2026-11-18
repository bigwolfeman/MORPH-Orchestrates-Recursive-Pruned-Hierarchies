"""No answer of a hop row can be guessed from the other answers of the row (real tokenizer, real task config).

An untrained model must read chance. A document whose questions share one answer (all Yes, all No, one distance
digit) is solved by "repeat the previous answer": the first graph eval design did that and read 0.94 after 60 steps.
Pinned, on the eval rows and on a sample of training rows, for every task (reachability = reach_yes + reach_no,
edge, shortest path, chain):
  1. "repeat the previous answer of the task in this row" and "the majority answer so far" sit at the chance
     the DATA gives them: the mean of the same baseline under random re-assignment of the answers over the answer
     positions (200 permutations), within 4 standard deviations of that null;
  2. the null itself is near the textbook chance (0.5 for Yes / No, 1 / len(sp_dists), about 0 for chain);
  3. "always Yes" answers half of reachability and of edge existence;
  4. the same baselines catch a document built the old way (all questions one answer).

    PYTHONPATH=.:$MORPH_ROOT HF_HUB_OFFLINE=1 python -m pytest -q lxtul/tests/test_hops_leaks.py
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from lxtul.hops import graph as G
from lxtul.hops.base import FAMILIES, build_row
from lxtul.tests.hops_support import real_task

TASKS = {"reach": ("reach_yes", "reach_no"), "edge": ("edge_yes", "edge_no"), "sp": ("sp",), "chain": ("chain",)}


def _answers(rows, task):
    """Per row, the answer tokens of `task` in text order."""
    ids = {FAMILIES[f] for f in TASKS[task]}
    return [r.tokens[np.isin(r.family, list(ids))].tolist() for r in rows]


def _repeat(rowsans):
    hit = tot = 0
    for a in rowsans:
        hit += sum(x == y for x, y in zip(a[:-1], a[1:]))
        tot += max(len(a) - 1, 0)
    return hit / tot, tot


def _majority(rowsans):
    hit = tot = 0
    for a in rowsans:
        for i in range(1, len(a)):
            c = collections.Counter(a[:i])
            top = max(c.values())
            guess = min(t for t, n in c.items() if n == top)
            hit += guess == a[i]
            tot += 1
    return hit / tot, tot


def _null(rowsans, fn, n_perm=200, seed=0):
    """(mean, std) of `fn` when the answers are re-assigned at random over the answer positions of the sample."""
    flat = np.array([t for a in rowsans for t in a])
    cuts = np.cumsum([len(a) for a in rowsans])[:-1]
    rng = np.random.default_rng(seed)
    vals = [fn([x.tolist() for x in np.split(rng.permutation(flat), cuts)])[0] for _ in range(n_perm)]
    return float(np.mean(vals)), float(np.std(vals))


def _check(rowsans, label):
    for name, fn in (("repeat", _repeat), ("majority", _majority)):
        acc, n = fn(rowsans)
        mean, std = _null(rowsans, fn)
        assert abs(acc - mean) <= 4 * std + 1.0 / n, f"{label} {name}: {acc:.3f} vs null {mean:.3f} +- {std:.3f} (n={n})"
    return _null(rowsans, _repeat)[0]


@pytest.fixture(scope="module")
def samples():
    t = real_task()
    ev = [r for _, r in t.eval_rows()]
    train = [t.train_row(s, r) for s in (0, 1) for r in range(150)]
    return t, ev, train


def test_baselines_sit_at_chance_on_eval_and_training_rows(samples):
    t, ev, train = samples
    n_sp = len(next(s for s in t.sources if s.name == "graph").ev["sp_dists"])
    for label, rows, tol in (("eval", ev, 0.03), ("train", train, 0.06)):
        for task, chance in (("reach", 0.5), ("edge", 0.5), ("sp", 1 / n_sp), ("chain", 0.0)):
            m = _check(_answers(rows, task), f"{label}/{task}")
            assert abs(m - chance) < (tol if task != "sp" else (0.03 if label == "eval" else 0.06)), (label, task, m, chance)


def test_always_yes_answers_half_of_reachability_and_edge(samples):
    t, ev, train = samples
    yes = G_YES = next(s for s in t.sources if s.name == "graph").v.p["yes"][0]
    for label, rows in (("eval", ev), ("train", train)):
        for task in ("reach", "edge"):
            toks = np.array([x for a in _answers(rows, task) for x in a])
            frac, n = float((toks == yes).mean()), len(toks)
            assert abs(frac - 0.5) <= 4 * np.sqrt(0.25 / n), (label, task, frac, n)


def test_the_baselines_catch_the_old_eval_design(samples):
    """Negative control: documents whose questions all share one answer (the first eval design) are not at chance."""
    t, _, _ = samples
    src = next(s for s in t.sources if s.name == "graph")

    def old_doc(rng):                                    # a Yes document or a No document, never both
        g = G.Forest(rng, 40, spine=3)
        d, used, qs, yes = 3, set(), [], bool(rng.random() < 0.5)
        for _ in range(8):
            q = G.ask(rng, g, "reach", used, (d, d), yes=yes, d=d if yes else None)
            if q is not None:
                qs.append(q)
        return G.render(src.v, rng, g, qs)
    rows = [build_row(old_doc, np.random.default_rng([5, r]), t.seq_len, t.max_slots, t.rule) for r in range(60)]
    ans = _answers(rows, "reach")
    acc, n = _repeat(ans)
    mean, std = _null(ans, _repeat)
    assert acc > 0.95 and acc > mean + 4 * std + 1.0 / n, (acc, mean, std)
