"""Two synthetic span tasks for the toy slot loop.

Both tasks pack a row as n_spans spans of span_len symbols. The model never sees the
answer as an INPUT token: the answer alphabet is output-only, and it appears only as the
LABEL at the first token position of the NEXT span. That is the double-label shape of the
real model (spec sec 5) and it is what keeps the answer off the token path.

Vocabulary (12 ids):
  0..5   op symbols   -- the six elements of S_3, the only ids that ever appear as inputs
  6..11  value symbols -- output only, the answer alphabet

Task "compose" (iterative state tracking).
  A span's OPERATOR is its FIRST symbol; the other span_len-1 symbols are distractors that
  keep the ordinary next-symbol loss non-trivial and keep both tasks on one layout. The
  register is R_i = R_{i-1} . P_i with P_i the span's operator and R_{-1} = e. The label at
  the first token of span i+1 is VALUE(R_i). Composition in S_3 is not commutative, so no
  single bag/average step computes R_i from the raw symbols: slot i must combine its own
  operator with the running register carried by earlier slots, and one round of slot->slot
  attention advances that carry by a bounded number of spans.

  An earlier version made P_i the ordered product of ALL span_len symbols. The capacity
  ladder (2026-09-10, `ignored/.../ladder/`) showed nothing learned the scan at any depth:
  value CE plateaued at 1.23-1.34 against a chance of 1.79 at every fixed depth from 1 to
  3, i.e. about two of the seven answers. Folding the within-span product into the prelude
  was consuming the whole budget. The scan is the object of study, so the within-span work
  was removed and the ladder re-run.

Task "summary" (one-pass control).
  The label at the first token of span i+1 is VALUE(mode of span i's symbols), ties to the
  smallest id. Slot i's own prelude already sees every symbol of span i, so this needs no
  iteration at all.

Every other token position carries the ordinary next-symbol label, which is uniform noise
by construction (CE floor ln 6 = 1.7918). That mirrors the real model, where most of the
token CE is local statistics the loop cannot help with.
"""

from __future__ import annotations

from dataclasses import dataclass

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


# --------------------------------------------------------------------------------------
# task "eliminate" (2026-09-18): deferred commitment
# --------------------------------------------------------------------------------------
#
# Added for the superposition question (Zhu et al. NeurIPS 2025; Rizvi-Martel et al. 2026):
# a task whose optimal intermediate state is a SET of candidates that later evidence prunes,
# not a point. `compose` and `summary` are untouched by this section; they keep the 12-id
# S_3 vocabulary and every number already filed under 2026-09-10.
#
# Alphabets (18 ids, disjoint):
#   0..5    candidate symbols   (input)
#   6..11   distractor symbols  (input)
#   12..17  answer symbols      (OUTPUT ONLY) -- answer(c) = ELIM_VALUE_BASE + c, a fixed
#           bijection from candidate symbols, so the answer never appears as an input token.
#
# One instance is 4 spans of 3 cells:
#   span s+0  three DISTINCT candidates            -> the alive set, size 3
#   span s+1  [first elimination][distractor x2]   -> alive set size 2
#   span s+2  [second elimination][distractor x2]  -> alive set size 1 (the survivor)
#   span s+3  three distractors; its HEAD carries the survivor's answer id as the label
# The survivor is uniform over the three candidates and the elimination order is uniform,
# so from span s+0 alone the answer is uniform over 3, after span s+1 uniform over 2, and
# after span s+2 it is determined. A row holds n_spans // 4 independent instances.
N_CAND = 6
N_DIST = 6
ELIM_CAND_BASE = 0
ELIM_DIST_BASE = N_CAND
ELIM_VALUE_BASE = N_CAND + N_DIST
ELIM_VOCAB = N_CAND + N_DIST + N_CAND
ELIM_SPANS = 4  # spans per instance


# --------------------------------------------------------------------------------------
# task "eliminate6" (2026-09-18, iteration 2): a SIX-hop elimination chain
# --------------------------------------------------------------------------------------
#
# `eliminate` was solved by every cell of its grid, so its escape rate measured nothing: a
# 2-pass chain has no basin. `eliminate6` deepens the chain and grades the hop distance.
#
# Alphabets (22 ids, disjoint):
#   0..7    candidate symbols   (input)  -- an instance uses SIX of the eight
#   8..13   distractor symbols  (input)
#   14..21  answer symbols      (OUTPUT ONLY), answer(c) = E6_VALUE_BASE + c
#
# One instance is the whole row, 8 spans of 3 cells:
#   span 0,1  the six DISTINCT candidates, three per span, in a uniformly random order
#   span 2..6 [elimination][distractor x2], five eliminations in a uniformly random order
#   span 7    three distractors; its HEAD carries the survivor's answer id as the label
#
# Two deliberate departures from `eliminate`, both forced by arithmetic:
#   - FIVE eliminations, not four. Four eliminations of six candidates leave TWO alive, so
#     the answer would not be determined and the commitment ladder could not reach 0.
#   - EIGHT candidate symbols, of which an instance uses six. With |C| = 6 every instance's
#     candidate set would be the whole alphabet, the candidate spans would carry nothing,
#     and "does the loop carry a SET" would collapse into "does it carry the exclusions".
# The answer slot is slot 6 and its window after t passes is spans 6-t .. 6, so the full
# candidate set (span 0) arrives only at pass 6: a six-hop chain with one fact per hop.
N6_CAND = 8
N6_DIST = 6
N6_ALIVE = 6  # candidates per instance
E6_CAND_BASE = 0
E6_DIST_BASE = N6_CAND
E6_VALUE_BASE = N6_CAND + N6_DIST
E6_VOCAB = N6_CAND + N6_DIST + N6_CAND
E6_SPANS = 8
E6_CAND_SPANS = (0, 1)
E6_ELIM_SPANS = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class ElimSpec:
    """What an elimination task's row looks like, for the instruments to read it.

    Spans are offsets INSIDE one instance. `roles` are the slots the instruments report,
    by their offset inside the instance. Under strict geometry slot `r` after `t` passes
    holds spans r-t .. r, so this dataclass is all a probe needs to know what a state
    COULD hold, which is what every reading here is scored against.
    """

    task: str
    spans: int  # spans per instance
    cand_spans: tuple[int, ...]
    elim_spans: tuple[int, ...]
    n_symbols: int  # |C|, the candidate alphabet
    n_alive: int  # candidates per instance
    value_base: int
    roles: tuple[tuple[str, int], ...]
    probe_depth: int  # the forced depth the per-pass instruments run at

    @property
    def answer_slot(self) -> int:
        return self.elim_spans[-1]


ELIMINATE_SPEC = ElimSpec(
    task="eliminate",
    spans=ELIM_SPANS,
    cand_spans=(0,),
    elim_spans=(1, 2),
    n_symbols=N_CAND,
    n_alive=3,
    value_base=ELIM_VALUE_BASE,
    roles=(("cand", 0), ("elim1", 1), ("answer", 2)),
    probe_depth=6,
)

ELIMINATE6_SPEC = ElimSpec(
    task="eliminate6",
    spans=E6_SPANS,
    cand_spans=E6_CAND_SPANS,
    elim_spans=E6_ELIM_SPANS,
    n_symbols=N6_CAND,
    n_alive=N6_ALIVE,
    value_base=E6_VALUE_BASE,
    # `cand` holds the whole set from pass 1; `mid` is the third elimination, whose set
    # completes at pass 4; `answer` is the slot the label is read from, complete at pass 6
    roles=(("cand", 1), ("mid", 4), ("answer", 6)),
    probe_depth=8,
)

SPECS = {"eliminate": ELIMINATE_SPEC, "eliminate6": ELIMINATE6_SPEC}


def spec_for(task: str) -> ElimSpec:
    if task not in SPECS:
        raise ValueError(f"{task} is not an elimination task")
    return SPECS[task]


def vocab_for(task: str) -> int:
    """Vocabulary size of a task. `compose` and `summary` keep their 12 ids."""
    if task == "eliminate":
        return ELIM_VOCAB
    if task == "eliminate6":
        return E6_VOCAB
    if task in {"compose", "summary"}:
        return VOCAB
    raise ValueError(task)


def _eliminate_draw(B, n_inst, generator, device):
    """Sample (candidates, survivor, elim) for every instance.

    candidates [B, I, 3]  the three DISTINCT candidate symbols, in span order
    survivor   [B, I]     the surviving candidate symbol
    elim       [B, I, 2]  the eliminated symbols, in elimination order
    """
    # a random 3-subset in a random order: the first 3 of a random permutation of C
    perm = torch.rand(B, n_inst, N_CAND, generator=generator, device=device).argsort(dim=-1)
    cands = perm[..., :3] + ELIM_CAND_BASE
    s = torch.randint(0, 3, (B, n_inst), generator=generator, device=device)
    flip = torch.randint(0, 2, (B, n_inst), generator=generator, device=device)
    o1 = (s + 1) % 3
    o2 = (s + 2) % 3
    first = torch.where(flip == 0, o1, o2)
    second = torch.where(flip == 0, o2, o1)
    survivor = cands.gather(2, s.unsqueeze(-1)).squeeze(-1)
    e1 = cands.gather(2, first.unsqueeze(-1)).squeeze(-1)
    e2 = cands.gather(2, second.unsqueeze(-1)).squeeze(-1)
    return cands, survivor, torch.stack([e1, e2], dim=-1)


def _make_eliminate(B, n_spans, span_len, generator, device, twin: bool = False):
    if n_spans % ELIM_SPANS != 0:
        raise ValueError(f"eliminate needs n_spans divisible by {ELIM_SPANS}, got {n_spans}")
    if span_len != 3:
        raise ValueError(f"eliminate needs span_len 3 (one cell per candidate), got {span_len}")
    n_inst = n_spans // ELIM_SPANS
    cands, survivor, elim = _eliminate_draw(B, n_inst, generator, device)
    toks = ELIM_DIST_BASE + torch.randint(
        0, N_DIST, (B, n_spans, span_len), generator=generator, device=device
    )
    out = [_pack_eliminate(toks, cands, survivor, elim, n_spans, span_len, device)]
    if twin:
        # the twin differs ONLY in span s+1: the roles of the survivor and the FIRST
        # eliminated candidate are swapped. Span s+0 (the candidate set and its order),
        # span s+2 (the second elimination) and every distractor are bit-identical.
        t_surv = elim[..., 0]
        t_elim = torch.stack([survivor, elim[..., 1]], dim=-1)
        out.append(_pack_eliminate(toks, cands, t_surv, t_elim, n_spans, span_len, device))
    return out


def _pack_eliminate(toks, cands, survivor, elim, n_spans, span_len, device):
    B = toks.shape[0]
    n_inst = cands.shape[1]
    toks = toks.clone()
    for k in range(n_inst):
        base = k * ELIM_SPANS
        toks[:, base, :] = cands[:, k]
        toks[:, base + 1, 0] = elim[:, k, 0]
        toks[:, base + 2, 0] = elim[:, k, 1]

    flat = toks.reshape(B, n_spans * span_len)
    labels = torch.full_like(flat, -100)
    labels[:, :-1] = flat[:, 1:]  # ordinary next-symbol label
    labels[:, -1] = -100
    mux_next = torch.full((B, n_spans), -100, dtype=torch.long, device=device)
    mux_own = torch.full((B, n_spans), -100, dtype=torch.long, device=device)
    for k in range(n_inst):
        base = k * ELIM_SPANS
        # the answer sits at the head of the instance's LAST span, i.e. the next span
        # after the slot that must know it (slot base+2), exactly as in `compose`
        labels[:, (base + 3) * span_len] = ELIM_VALUE_BASE + survivor[:, k]
        mux_next[:, base + 2] = ELIM_VALUE_BASE + survivor[:, k]
        # the staged attachment's own-span target: the span's OWN elimination
        mux_own[:, base + 1] = ELIM_VALUE_BASE + elim[:, k, 0]
        mux_own[:, base + 2] = ELIM_VALUE_BASE + elim[:, k, 1]

    value_pos = torch.tensor(
        [(k * ELIM_SPANS + 3) * span_len for k in range(n_inst)], device=device
    )
    return {
        "tokens": flat,
        "labels": labels,
        "mux_next": mux_next,
        "mux_own": mux_own,
        "value_pos": value_pos,
        # instance metadata, read by the eliminate instruments only
        "candidates": cands,
        "survivor": survivor,
        "elim": elim,
        "inst_base_span": torch.tensor(
            [k * ELIM_SPANS for k in range(n_inst)], device=device
        ),
    }


def make_twin_batch(B, n_spans, span_len, generator=None, device="cpu", task="eliminate"):
    """Two batches identical through the candidate spans and differing ONLY in the FIRST
    elimination: the survivor and the first eliminated candidate swap roles.

    Returns (a, b). The instrument reading is a reachability test: a slot state may not
    move until the changed span enters its reachable window.
    """
    if task == "eliminate6":
        return _make_eliminate6(B, n_spans, span_len, generator, device, twin=True)
    return _make_eliminate(B, n_spans, span_len, generator, device, twin=True)


def _eliminate6_draw(B, generator, device):
    """(candidates [B,6] in span order, survivor [B], elim [B,5] in elimination order)."""
    perm = torch.rand(B, N6_CAND, generator=generator, device=device).argsort(dim=-1)
    cands = perm[:, :N6_ALIVE] + E6_CAND_BASE
    s = torch.randint(0, N6_ALIVE, (B,), generator=generator, device=device)
    survivor = cands.gather(1, s.unsqueeze(-1)).squeeze(-1)
    # the other five, in a uniformly random order: drop the survivor's column, then shuffle
    keep = torch.arange(N6_ALIVE, device=device).expand(B, N6_ALIVE) != s.unsqueeze(-1)
    others = cands[keep].view(B, N6_ALIVE - 1)
    order = torch.rand(B, N6_ALIVE - 1, generator=generator, device=device).argsort(dim=-1)
    elim = others.gather(1, order)
    return cands, survivor, elim


def _make_eliminate6(B, n_spans, span_len, generator, device, twin: bool = False):
    if n_spans != E6_SPANS or span_len != 3:
        raise ValueError(f"eliminate6 needs n_spans {E6_SPANS} and span_len 3, got {n_spans}/{span_len}")
    cands, survivor, elim = _eliminate6_draw(B, generator, device)
    toks = E6_DIST_BASE + torch.randint(
        0, N6_DIST, (B, n_spans, span_len), generator=generator, device=device
    )
    out = [_pack_eliminate6(toks, cands, survivor, elim, span_len, device)]
    if twin:
        # identical through the candidate spans: the survivor and the FIRST eliminated
        # candidate swap roles, so only the head of span 2 changes
        t_surv = elim[:, 0]
        t_elim = torch.cat([survivor.unsqueeze(-1), elim[:, 1:]], dim=-1)
        out.append(_pack_eliminate6(toks, cands, t_surv, t_elim, span_len, device))
    return out


def _pack_eliminate6(toks, cands, survivor, elim, span_len, device):
    B = toks.shape[0]
    toks = toks.clone()
    for j, sp in enumerate(E6_CAND_SPANS):
        toks[:, sp, :] = cands[:, j * span_len : (j + 1) * span_len]
    for j, sp in enumerate(E6_ELIM_SPANS):
        toks[:, sp, 0] = elim[:, j]

    flat = toks.reshape(B, E6_SPANS * span_len)
    labels = torch.full_like(flat, -100)
    labels[:, :-1] = flat[:, 1:]
    labels[:, -1] = -100
    answer_span = E6_ELIM_SPANS[-1] + 1
    labels[:, answer_span * span_len] = E6_VALUE_BASE + survivor

    mux_next = torch.full((B, E6_SPANS), -100, dtype=torch.long, device=device)
    mux_own = torch.full((B, E6_SPANS), -100, dtype=torch.long, device=device)
    mux_next[:, E6_ELIM_SPANS[-1]] = E6_VALUE_BASE + survivor
    for j, sp in enumerate(E6_ELIM_SPANS):
        mux_own[:, sp] = E6_VALUE_BASE + elim[:, j]

    return {
        "tokens": flat,
        "labels": labels,
        "mux_next": mux_next,
        "mux_own": mux_own,
        "value_pos": torch.tensor([answer_span * span_len], device=device),
        "candidates": cands.unsqueeze(1),  # [B, 1 instance, 6]
        "survivor": survivor.unsqueeze(1),
        "elim": elim.unsqueeze(1),
        "inst_base_span": torch.tensor([0], device=device),
    }


def make_batch(
    task: str,
    B: int,
    n_spans: int,
    span_len: int,
    generator: torch.Generator | None = None,
    device="cpu",
) -> dict[str, torch.Tensor]:
    """Return tokens [B, n_spans*span_len], labels (same shape), mux_next/mux_own [B, n_spans]."""
    if task == "eliminate":
        return _make_eliminate(B, n_spans, span_len, generator, device)[0]
    if task == "eliminate6":
        return _make_eliminate6(B, n_spans, span_len, generator, device)[0]
    mul = MUL.to(device)
    toks = torch.randint(0, N_GROUP, (B, n_spans, span_len), generator=generator, device=device)

    if task == "compose":
        span_prod = toks[:, :, 0].clone()  # the span's operator is its FIRST symbol
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


# --------------------------------------------------------------------------------------
# the ceilings of `eliminate`, computed by enumeration
# --------------------------------------------------------------------------------------
#
# The instance distribution is uniform over the 6*5*4 = 120 ordered triples
# (survivor, first eliminated, second eliminated) of distinct candidate symbols: a uniform
# 3-subset (20) times a uniform survivor (3) times a uniform elimination order (2).
# Everything below is a conditional entropy of the survivor under that distribution, in
# nats, which is the lowest value CE any predictor with that information can reach.


def _elim_universe():
    """[(prob, survivor, e1, e2)] over the instance distribution."""
    out = []
    p = 1.0 / (N_CAND * (N_CAND - 1) * (N_CAND - 2))
    for s in range(N_CAND):
        for a in range(N_CAND):
            if a == s:
                continue
            for b in range(N_CAND):
                if b in (s, a):
                    continue
                out.append((p, s, a, b))
    return out


def _cond_entropy(pairs) -> float:
    """H(survivor | observation) in nats, from [(prob, observation, survivor)]."""
    import math
    from collections import defaultdict

    joint: dict = defaultdict(lambda: defaultdict(float))
    for p, obs, s in pairs:
        joint[obs][s] += p
    h = 0.0
    for obs, dist in joint.items():
        tot = sum(dist.values())
        for s, p in dist.items():
            q = p / tot
            h -= p * math.log(q)
    return h


def eliminate_ceilings() -> dict:
    """Every ceiling of `eliminate`, in nats. Enumerated, never asserted.

    reachability[d]  the lowest value CE at forced depth d. Under strict geometry the
                     answer slot (span s+2) reaches spans s+2-d .. s+2 after d passes, so
                     depth d buys exactly the eliminations and the candidate set inside
                     that window. This is the K-curve's floor.
    commitment[t]    the lowest value CE for a predictor that is a function of spans
                     s+0 .. s+t ONLY, i.e. one that commits at span s+t and then ignores
                     every later elimination. ln 3, ln 2, 0.
    point_carry[t]   the lowest value CE for a predictor that carries ONE candidate id
                     sampled from the alive set after span s+t, and still reads the later
                     eliminations locally. This is the plateau a loop that collapses its
                     state to a point should sit at; it is BELOW commitment[t] because the
                     later eliminations are still readable at the answer slot.
    """
    import math

    uni = _elim_universe()
    reach = {}
    for d in range(0, 4):
        # window spans s+2-d .. s+2: d=0 sees the second elimination, d=1 both, d>=2 also
        # the candidate set
        if d == 0:
            obs = lambda p, s, a, b: (b,)  # noqa: E731
        elif d == 1:
            obs = lambda p, s, a, b: (a, b)  # noqa: E731
        else:
            obs = lambda p, s, a, b: (s, a, b)  # noqa: E731
        reach[d] = _cond_entropy([(p, obs(p, s, a, b), s) for p, s, a, b in uni])

    commit = {}
    # the candidate set is seen as an unordered set; the survivor's position in span s+0 is
    # uniform, so the order carries nothing
    commit[0] = _cond_entropy([(p, tuple(sorted((s, a, b))), s) for p, s, a, b in uni])
    commit[1] = _cond_entropy([(p, (tuple(sorted((s, a, b))), a), s) for p, s, a, b in uni])
    commit[2] = _cond_entropy([(p, (tuple(sorted((s, a, b))), a, b), s) for p, s, a, b in uni])

    point = {}
    alive_at = {0: lambda s, a, b: (s, a, b), 1: lambda s, a, b: (s, b), 2: lambda s, a, b: (s,)}
    for t, alive in alive_at.items():
        pairs = []
        for p, s, a, b in uni:
            live = alive(s, a, b)
            for g in live:
                pairs.append((p / len(live), (g, a, b), s))
        point[t] = _cond_entropy(pairs)

    return {
        "chance": math.log(N_CAND),
        "reachability": reach,
        "commitment": commit,
        "point_carry": point,
    }


def _elim6_universe():
    """[(prob, survivor, (e0..e4))] over the `eliminate6` instance distribution.

    An instance is a survivor and an ORDERED tuple of five eliminated symbols, all six
    distinct, drawn from eight: 8*7*6*5*4*3 = 20160 equiprobable configurations. That is
    exactly a uniform 6-subset (28) times a uniform survivor (6) times a uniform
    elimination order (120).
    """
    from itertools import permutations

    out = []
    n = N6_CAND
    total = 1
    for i in range(N6_ALIVE):
        total *= n - i
    p = 1.0 / total
    for s in range(n):
        rest = [c for c in range(n) if c != s]
        for e in permutations(rest, N6_ALIVE - 1):
            out.append((p, s, e))
    return out


def eliminate6_ceilings() -> dict:
    """Every ceiling of `eliminate6`, in nats, enumerated over the 20,160 configurations.

    reachability[d]  the lowest value CE at forced depth d. The answer slot is slot 6 and
                     its window after d passes is spans 6-d .. 6, so d = 0 buys the last
                     elimination, d = 4 buys all five, d = 5 adds the three candidates in
                     span 1 and d = 6 adds span 0 and determines the answer. One fact per
                     hop, which is what makes this task a hop-distance instrument.
    commitment[k]    the lowest value CE for a predictor that is a function of the
                     candidate spans and the first k eliminations only: ln 6 … ln 2, 0.
    point_carry[k]   the lowest value CE for a predictor that collapses to ONE candidate id
                     sampled from the alive set after k eliminations and still reads all
                     five eliminations locally at the answer slot.
    """
    import math
    from itertools import combinations

    uni = _elim6_universe()
    ne = N6_ALIVE - 1  # five eliminations

    reach = {}
    for d in range(0, ne):  # d = 0..4: the last d+1 eliminations, nothing else
        seen = ne - 1 - d
        reach[d] = _cond_entropy([(p, e[seen:], s) for p, s, e in uni])
    # d = 5 adds span 1: three of the six candidates, a uniform 3-subset. Enumerated over
    # the C(6,3) = 20 equally likely placements rather than argued.
    pairs = []
    for p, s, e in uni:
        cands = (s,) + e
        for sub in combinations(range(N6_ALIVE), N6_ALIVE // 2):
            s1 = frozenset(cands[i] for i in sub)
            pairs.append((p / 20.0, (e, s1), s))
    reach[ne] = _cond_entropy(pairs)
    reach[ne + 1] = _cond_entropy([(p, (frozenset((s,) + e), e), s) for p, s, e in uni])

    commit = {}
    for k in range(ne + 1):
        commit[k] = _cond_entropy(
            [(p, (frozenset((s,) + e), e[:k]), s) for p, s, e in uni]
        )

    point = {}
    for k in range(ne + 1):
        pairs = []
        for p, s, e in uni:
            alive = (s,) + e[k:]  # the survivor plus the not-yet-eliminated candidates
            for g in alive:
                pairs.append((p / len(alive), (g, e), s))
        point[k] = _cond_entropy(pairs)

    return {
        "chance": math.log(N6_CAND),
        "reachability": reach,
        "commitment": commit,
        "point_carry": point,
    }


def ceilings_for(task: str) -> dict:
    return eliminate6_ceilings() if task == "eliminate6" else eliminate_ceilings()
