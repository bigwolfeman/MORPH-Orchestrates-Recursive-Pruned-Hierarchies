"""GraphQA-style friendship graphs as hop documents (Fatemi et al. 2024, `graphqa/graph_text_encoder.py`).

One document (text; names are single tokens in BOTH the sentence-initial and the mid-sentence form):
    G describes a friendship graph among nodes James, Robert, John, and Michael.
    We have the following edges in G:
    James and Robert are friends.                        # one sentence per edge, one span each
    ...
    Q: Is there a path from node James to node John?     # one question sentence, one span
    A: Yes.                                              # one answer sentence, one span
EOS closes the document. The graph is a random forest of 2 trees (random parent attachment), edge order
shuffled, the two endpoints of an edge in random order. Questions, three tasks with config weights:
    reachability   `Q: Is there a path from node X to node Y?`        Yes / No balanced
                   Yes: hops = shortest path length d (family reach_yes). No: hops = the source's component size
                   (family reach_no): the reader must rule out a whole tree.
    shortest_path  `Q: What is the length of the shortest path from node X to node Y?`   `A: 3.` (ONE digit token)
                   connected pairs only, hops = d = the answer (family sp)
    edge_existence `Q: Is node X connected to node Y?`                Yes / No balanced
                   hops 1 (families edge_yes, edge_no)
Pairs are drawn stratified by distance: d is uniform over the distances the graph has inside the config range,
then a pair at that d. A pair is asked at most once per document. Training documents mix sizes (`n_choices`), ask
clip(n // 3, 2, questions_max) questions, and plant a path of up to max(reach_dist, sp_dist) edges (as far as n - 3
allows) so the far distances exist. EVALUATION documents mix their questions the way training
documents do (a fixed count per task, a fair coin per Yes / No question, distances drawn uniformly over the eval lists,
random order). Exactly balanced documents would leak the other way: the last answer of a document is then
predictable from the counts, and repeating the previous answer is wrong more than half the time.; buckets are per ANSWER (family, hops), never per document: a document whose questions share one
answer would be solved by repeating the previous answer token. The answer token is the Yes / No / digit token;
the label position is the token before it ("A:" colon, or the space before a digit).

Documents are assembled from token ids of template pieces (each encoded once), not by encoding the whole text;
tests decode a document and encode it again to prove the two agree. `solve` is the reference reader: BFS on the
edges parsed from the token ids alone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lxtul.hops.base import FAMILIES, Document, HopEnv, HopSource, single_token

GRAPHQA_NAMES = ["James", "Robert", "John", "Michael", "David", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
                 "William", "Richard", "Joseph", "Thomas", "Christopher", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
MIN_NAMES = 64
PIECES = dict(                      # text -> token list; the single-token ones are checked at build
    hdr="G describes a friendship graph among nodes", edges="We have the following edges in G:",
    are=" are friends", q_reach="Q: Is there a path from node", q_sp="Q: What is the length of the shortest path from node",
    q_edge="Q: Is node", to_node=" to node", conn=" connected to node", a="A:")
SINGLES = dict(comma=",", and_=" and", dot=".", nl="\n", qmark="?", yes=" Yes", no=" No", space=" ")


@dataclass(frozen=True)
class GraphVocab:
    init: np.ndarray                 # [P] id of "Name" (sentence-initial form)
    mid: np.ndarray                  # [P] id of " Name"
    p: dict[str, list[int]]          # template pieces (ids)
    digits: np.ndarray               # [10] ids of "0".."9"
    eos: int

    def __post_init__(self):
        pool = set(self.init.tolist()) | set(self.mid.tolist())
        other = {i for v in self.p.values() for i in v} | set(self.digits.tolist()) | {self.eos}
        if len(pool) != 2 * len(self.init) or pool & other:
            raise ValueError("graph vocab: a name id is also a template or digit id, or two names share an id")


def pick_vocab(env: HopEnv, n_names: int = 128, min_names: int = MIN_NAMES) -> GraphVocab:
    """GraphQA's 20 names first, then capitalised alphabetic words (3 to 8 letters) in token-id order, keeping
    a word only when " Word" AND "Word" are each one token that encodes back to its own id. Raises under
    `min_names` names; takes up to `n_names`."""
    ids = {}
    for i, t in enumerate(env.token_text):
        if t:
            ids.setdefault(t, i)
    cand = list(GRAPHQA_NAMES)
    seen = set(cand)
    for t in env.token_text:
        if len(t) in range(4, 10) and t[0] == " " and t[1:].isascii() and t[1:].isalpha() and t[1].isupper() \
                and t[2:].islower() and t[1:] not in seen:
            cand.append(t[1:])
            seen.add(t[1:])
    init, mid = [], []
    for w in cand:
        a, b = env.encode(w), env.encode(" " + w)
        if len(a) == len(b) == 1 and a[0] != b[0] and env.token_text[a[0]] == w and env.token_text[b[0]] == " " + w:
            init.append(a[0]), mid.append(b[0])
            if len(init) == n_names:
                break
    if len(init) < min_names:
        raise ValueError(f"graph vocab: only {len(init)} names are one token in both forms; need {min_names}")
    p = {k: env.encode(t) for k, t in PIECES.items()}
    p |= {k: [single_token(env, t)] for k, t in SINGLES.items()}
    if not all(p.values()):
        raise ValueError("graph vocab: an empty template piece")
    if not env.rule.is_boundary[p["dot"][0]] or not env.rule.is_boundary[p["qmark"][0]]:
        raise ValueError('"." and "?" must cut a span under the boundary rule')
    digits = np.array([single_token(env, str(d)) for d in range(10)])
    return GraphVocab(np.array(init), np.array(mid), p, digits, env.eos)


# --------------------------------------------------------------------------------------------- the graph
class Forest:
    """Two trees on n nodes: `edges` (shuffled, endpoints in random order), `comp` (node -> tree), `dist` ([n, n]
    shortest path, -1 across trees) and `pairs` (d -> unordered pairs at distance d)."""

    def __init__(self, rng: np.random.Generator, n: int, spine: int = 0):
        lo = max(2, spine + 1)
        if n - 2 < lo:
            raise ValueError(f"a forest of two trees on {n} nodes cannot hold a path of {spine} edges")
        n1 = int(rng.integers(lo, n - 1))                          # first tree: lo .. n - 2 nodes; second: the rest
        self.n, self.comp = n, np.array([0] * n1 + [1] * (n - n1))
        edges = []
        for first, size in ((0, n1), (n1, n - n1)):
            path = spine + 1 if first == 0 else 0
            for i in range(1, size):
                edges.append((first + i - 1 if i < path else first + int(rng.integers(0, i)), first + i))
        order = rng.permutation(len(edges))
        flip = rng.random(len(edges)) < 0.5
        self.edges = [(b, a) if f else (a, b) for (a, b), f in ((edges[i], flip[j]) for j, i in enumerate(order))]
        self.dist = _all_pairs(n, self.edges)
        self.pairs: dict[int, list[tuple[int, int]]] = {}
        for u in range(n):
            for v in range(u + 1, n):
                if self.dist[u, v] > 0:
                    self.pairs.setdefault(int(self.dist[u, v]), []).append((u, v))

    def comp_size(self, u: int) -> int:
        return int((self.comp == self.comp[u]).sum())


def _all_pairs(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b), adj[b].append(a)
    d = np.full((n, n), -1, dtype=np.int64)
    for s in range(n):
        d[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if d[s, w] < 0:
                    d[s, w] = d[s, u] + 1
                    q.append(w)
    return d


# ------------------------------------------------------------------------------------------- the questions
@dataclass(frozen=True)
class Question:
    kind: str          # reach | sp | edge
    u: int
    v: int
    answer: int        # reach / edge: 1 = Yes, 0 = No; sp: the distance
    hop: int
    family: str


def ask(rng: np.random.Generator, g: Forest, kind: str, used: set, d_range: tuple[int, int],
        yes: bool | None = None, d: int | None = None) -> Question | None:
    """One question of `kind` on forest `g` about a pair not in `used`, or None when the graph has none.
    `yes` fixes the answer of reach / edge (None: a fair coin); `d` fixes the distance of a Yes / sp question
    (None: uniform over the distances the graph has inside `d_range`)."""
    if kind == "reach" or kind == "sp":
        want_yes = True if kind == "sp" else (bool(rng.random() < 0.5) if yes is None else yes)
        if want_yes:
            have = [x for x in range(d_range[0], d_range[1] + 1)
                    if x in g.pairs and any(frozenset(p) not in used for p in g.pairs[x])]
            if not have or (d is not None and d not in have):
                return None
            dd = d if d is not None else int(have[int(rng.integers(0, len(have)))])
            free = [p for p in g.pairs[dd] if frozenset(p) not in used]
            u, v = free[int(rng.integers(0, len(free)))]
            u, v = (v, u) if rng.random() < 0.5 else (u, v)
            used.add(frozenset((u, v)))
            return Question(kind, u, v, dd if kind == "sp" else 1, dd, "sp" if kind == "sp" else "reach_yes")
        free = [(u, v) for u in range(g.n) for v in range(u + 1, g.n)
                if g.comp[u] != g.comp[v] and frozenset((u, v)) not in used]
        if not free:
            return None
        u, v = free[int(rng.integers(0, len(free)))]
        u, v = (v, u) if rng.random() < 0.5 else (u, v)
        used.add(frozenset((u, v)))
        return Question("reach", u, v, 0, g.comp_size(u), "reach_no")
    want_yes = bool(rng.random() < 0.5) if yes is None else yes
    if want_yes:
        free = [e for e in g.edges if frozenset(e) not in used]
    else:
        free = [(u, v) for u in range(g.n) for v in range(u + 1, g.n)
                if g.dist[u, v] != 1 and frozenset((u, v)) not in used]
    if not free:
        return None
    u, v = free[int(rng.integers(0, len(free)))]
    u, v = (v, u) if rng.random() < 0.5 else (u, v)
    used.add(frozenset((u, v)))
    return Question("edge", u, v, int(want_yes), 1, "edge_yes" if want_yes else "edge_no")


def render(vocab: GraphVocab, rng: np.random.Generator, g: Forest, questions: list[Question]) -> Document:
    """Tokens, hops and family of one document: header, edge sentences, then each question and its answer."""
    p, nl, dot = vocab.p, vocab.p["nl"], vocab.p["dot"]
    who = rng.permutation(len(vocab.init))[: g.n]                  # node -> name index, distinct
    toks: list[int] = list(p["hdr"])
    for i in range(g.n):
        toks += ([] if i == 0 else p["comma"] + (p["and_"] if i == g.n - 1 else [])) + [int(vocab.mid[who[i]])]
    toks += dot + nl + p["edges"] + nl
    for k, (a, b) in enumerate(g.edges):
        toks += (nl if k else []) + [int(vocab.init[who[a]])] + p["and_"] + [int(vocab.mid[who[b]])] + p["are"] + dot
    head = {"reach": p["q_reach"], "sp": p["q_sp"], "edge": p["q_edge"]}
    link = {"reach": p["to_node"], "sp": p["to_node"], "edge": p["conn"]}
    answers = []                                                   # (token index, hop label, family id)
    for q in questions:
        toks += nl + head[q.kind] + [int(vocab.mid[who[q.u]])] + link[q.kind] + [int(vocab.mid[who[q.v]])] + p["qmark"]
        toks += nl + p["a"] + (p["space"] if q.kind == "sp" else [])
        answers.append((len(toks), q.hop, FAMILIES[q.family]))
        toks += ([int(vocab.digits[q.answer])] if q.kind == "sp" else p["yes"] if q.answer else p["no"]) + dot
    toks.append(vocab.eos)
    hops, fam = np.zeros(len(toks), dtype=np.uint8), np.zeros(len(toks), dtype=np.uint8)
    for at, h, f in answers:
        hops[at], fam[at] = h, f
    return Document(np.array(toks, dtype=np.int64), hops, fam)


def solve(vocab: GraphVocab, toks: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Reference reader for tests: [(index of the answer token, the answer token the text gives, hop label,
    family id)] of one whole document, from the token ids alone (parse the edges, BFS the questions)."""
    t, p = toks.tolist(), vocab.p
    init = {int(x): i for i, x in enumerate(vocab.init)}
    mid = {int(x): i for i, x in enumerate(vocab.mid)}
    nl = p["nl"]

    def eat(i: int, piece: list[int]) -> int:
        if t[i: i + len(piece)] != piece:
            raise ValueError(f"expected {piece} at {i}, got {t[i: i + len(piece)]}")
        return i + len(piece)

    at = eat(_find(t, p["edges"]) + len(p["edges"]), nl)
    edges, node = [], {}
    while True:                                                    # one sentence per edge, newline-separated
        a = node.setdefault(init[t[at]], len(node))
        at = eat(at + 1, p["and_"])
        b = node.setdefault(mid[t[at]], len(node))
        at = eat(eat(at + 1, p["are"]), p["dot"])
        edges.append((a, b))
        if t[at + 1] not in init:
            break
        at = eat(at, nl)
    dist = _all_pairs(len(node), edges)
    heads = {"reach": p["q_reach"], "sp": p["q_sp"], "edge": p["q_edge"]}
    out = []
    while t[at] != vocab.eos:
        at = eat(at, nl)
        kind = next(k for k, h in heads.items() if t[at: at + len(h)] == h)
        at = eat(at, heads[kind])
        u = node[mid[t[at]]]
        at = eat(at + 1, p["conn"] if kind == "edge" else p["to_node"])
        v = node[mid[t[at]]]
        at = eat(eat(at + 1, p["qmark"]), nl + p["a"])
        d = int(dist[u, v])
        if kind == "sp":
            at = eat(at, p["space"])
            out.append((at, int(vocab.digits[d]), d, FAMILIES["sp"]))
        elif kind == "reach":
            out.append((at, p["yes" if d >= 0 else "no"][0], d if d >= 0 else int((dist[u] >= 0).sum()),
                        FAMILIES["reach_yes" if d >= 0 else "reach_no"]))
        else:
            out.append((at, p["yes" if d == 1 else "no"][0], 1, FAMILIES["edge_yes" if d == 1 else "edge_no"]))
        at = eat(at + 1, p["dot"])
    return out


def _find(t: list[int], piece: list[int]) -> int:
    for i in range(len(t) - len(piece) + 1):
        if t[i: i + len(piece)] == piece:
            return i
    raise ValueError(f"piece {piece} not found")


class GraphSource(HopSource):
    """`task.sources.graph`: n drawn uniformly from `n_choices`, clip(n // 3, 2, questions_max) questions per graph,
    task per question from `task_weights` (a task with no unused pair left is skipped for that question). Evaluation: ONE row set (`eval.rows` rows of mixed documents on forests of `eval.n` nodes, see
    `_eval_doc`); accuracy is bucketed per answer later."""
    name = "graph"
    KEYS = {"row_share", "n_choices", "questions_max", "task_weights", "reach_dist", "sp_dist", "names"}
    TASKS = {"reachability": "reach", "shortest_path": "sp", "edge_existence": "edge"}

    def __init__(self, vocab: GraphVocab, n_choices, questions_max: int, weights: dict, reach_dist, sp_dist, ev: dict):
        self.v, self.n_names = vocab, len(vocab.init)
        self.n_choices, self.questions_max = [int(x) for x in n_choices], int(questions_max)
        if set(weights) != set(self.TASKS):
            raise ValueError(f"graph task_weights keys must be {sorted(self.TASKS)}, got {sorted(weights)}")
        w = np.array([float(weights[k]) for k in self.TASKS])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("graph task_weights must be non-negative with a positive sum")
        self.kinds, self.w = list(self.TASKS.values()), w / w.sum()
        self.reach_dist, self.sp_dist = tuple(int(x) for x in reach_dist), tuple(int(x) for x in sp_dist)
        if not self.n_choices or min(self.n_choices) < 4 or max(self.n_choices) > len(vocab.init):
            raise ValueError(f"graph n_choices must be non-empty and lie in [4, {len(vocab.init)}] (names in the pool)")
        if self.questions_max < 2:
            raise ValueError("graph questions_max must be at least 2")
        if self.sp_dist[1] > 9:
            raise ValueError("graph sp_dist must stay at or under 9: the answer is ONE digit token")
        ev = dict(ev)
        need = {"n", "per_doc", "rows", "val_rows", "reach_dists", "sp_dists"}
        if set(ev) != need:
            raise ValueError(f"graph eval keys must be {sorted(need)}, got {sorted(ev)}")
        pd = {k: int(ev["per_doc"][k]) for k in ("reach", "sp", "edge")}
        if min(pd.values()) < 0 or sum(pd.values()) < 1:
            raise ValueError("graph eval per_doc: counts must be non-negative with a positive sum")
        self.ev = {"n": int(ev["n"]), "per_doc": pd, "reach_dists": [int(x) for x in ev["reach_dists"]],
                   "sp_dists": [int(x) for x in ev["sp_dists"]]}
        self.eval_rows, self.eval_val_rows = int(ev["rows"]), int(ev["val_rows"])
        if not 0 < self.eval_val_rows <= self.eval_rows:
            raise ValueError("graph eval val_rows must lie in [1, rows]")
        if self.ev["n"] > len(vocab.init) or max(self.ev["sp_dists"]) > 9:
            raise ValueError("graph eval: n exceeds the name pool, or an sp distance exceeds 9")
        if self.ev["n"] - 3 < max(self.ev["reach_dists"]):
            raise ValueError(f"graph eval: {self.ev['n']} nodes cannot hold a path of {max(self.ev['reach_dists'])} edges "
                             "beside a second tree")

    @classmethod
    def from_cfg(cls, cfg: dict, eval_cfg: dict, env: HopEnv) -> "GraphSource":
        unknown = set(cfg) - cls.KEYS
        if unknown:
            raise ValueError(f"unknown task.sources.graph keys: {sorted(unknown)}")
        return cls(pick_vocab(env, int(cfg["names"])), cfg["n_choices"], cfg["questions_max"], dict(cfg["task_weights"]),
                   cfg["reach_dist"], cfg["sp_dist"], eval_cfg)

    def train_document(self, rng):
        n = int(self.n_choices[int(rng.integers(0, len(self.n_choices)))])
        g = Forest(rng, n, spine=min(max(self.reach_dist[1], self.sp_dist[1]), n - 3))   # plant only what fits
        used, qs = set(), []
        for _ in range(min(max(n // 3, 2), self.questions_max)):
            left, q = list(range(len(self.kinds))), None
            while left and q is None:                              # a task with no unused pair: draw another task
                w = self.w[left] / self.w[left].sum() if self.w[left].sum() > 0 else None
                if w is None:
                    break
                i = left.pop(int(rng.choice(len(left), p=w)))
                kind = self.kinds[i]
                d_range = self.reach_dist if kind == "reach" else self.sp_dist
                if kind == "sp":
                    q = ask(rng, g, kind, used, d_range)
                else:                                              # a fair coin picks Yes / No; an empty side falls back
                    coin = bool(rng.random() < 0.5)
                    q = ask(rng, g, kind, used, d_range, yes=coin) or ask(rng, g, kind, used, d_range, yes=not coin)
            if q is None:                                          # nothing left to ask
                break
            qs.append(q)
        if not qs:                                                 # a forest always has an edge to ask about
            qs.append(ask(rng, g, "edge", used, (1, 1)))
        return render(self.v, rng, g, qs)

    def _eval_doc(self, rng) -> Document:
        """One evaluation document: a forest with a path of max(reach_dists) edges planted, then per_doc questions per
        task in random order. Reach / edge: a fair coin per question. Yes distances and sp distances drawn uniformly from
        the eval lists (another distance is drawn when the forest has no unused pair at it)."""
        g = Forest(rng, self.ev["n"], spine=max(self.ev["reach_dists"]))
        used, qs = set(), []
        for kind, dists in (("reach", self.ev["reach_dists"]), ("sp", self.ev["sp_dists"]), ("edge", None)):
            n = self.ev["per_doc"][kind if kind != "sp" else "sp"]
            for i in range(n):
                yes = None if kind == "sp" else bool(rng.random() < 0.5)       # a fair coin per question
                for _ in range(60):
                    d = int(dists[int(rng.integers(0, len(dists)))]) if dists and yes is not False else None
                    q = ask(rng, g, kind, used, (d, d) if d else (1, 1), yes=yes, d=d)
                    if q is not None:
                        qs.append(q)
                        break
                else:
                    raise ValueError(f"graph eval: no free {kind} question on a forest of {self.ev['n']} nodes")
        return render(self.v, rng, g, [qs[i] for i in rng.permutation(len(qs))])

    def eval_buckets(self):
        return [("graph/mixed", self._eval_doc)]
