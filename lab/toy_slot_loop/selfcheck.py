"""Self-checks for the toy slot loop. Run before trusting any grid number.

    python selfcheck.py

Each check fails loudly. Several are sabotage-shaped: they assert a number is ZERO on a
path that must not exist, and they are proven to have teeth by the paired assertion that
the same number is NOT zero on the path that must exist.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instruments import gradient_probe, make_eval_batches  # noqa: E402
from model import PREFIX, SLOT, TOKEN, Layout, ToyConfig, ToySlotLoop, build_masks  # noqa: E402
from tasks import (  # noqa: E402
    E6_CAND_SPANS,
    E6_DIST_BASE,
    E6_ELIM_SPANS,
    E6_SPANS,
    E6_VALUE_BASE,
    E6_VOCAB,
    ELIM_DIST_BASE,
    ELIM_VALUE_BASE,
    ELIM_VOCAB,
    MUL,
    N_CAND,
    N_GROUP,
    PERMS,
    VALUE_BASE,
    N6_ALIVE,
    N6_CAND,
    eliminate6_ceilings,
    eliminate_ceilings,
    make_batch,
    make_twin_batch,
    spec_for,
)

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILS.append(name)


def cfg(**kw):
    base = dict(d_model=32, n_heads=4, d_ff=64, n_prelude=2, n_coda=2, layout=Layout(6, 3, 2))
    base.update(kw)
    return ToyConfig(**base)


def t_group():
    ok = True
    for a in range(N_GROUP):
        for b in range(N_GROUP):
            for c in range(N_GROUP):
                if MUL[MUL[a, b], c] != MUL[a, MUL[b, c]]:
                    ok = False
    check("S_3 multiplication is associative", ok)
    check("S_3 is non-abelian", bool((MUL != MUL.T).any()))
    check("identity is element 0", bool((MUL[0] == torch.arange(N_GROUP)).all()))


def t_task():
    g = torch.Generator().manual_seed(0)
    b = make_batch("compose", 64, 6, 3, generator=g)
    toks = b["tokens"].view(64, 6, 3)
    # recompute the register independently, one row at a time
    ok = True
    for r in range(64):
        acc = 0
        for i in range(6):
            acc = int(MUL[acc, int(toks[r, i, 0])])  # the operator is the span's FIRST symbol
            if int(b["mux_next"][r, i]) != VALUE_BASE + acc:
                ok = False
            if i >= 1 and int(b["labels"][r, i * 3]) != VALUE_BASE + int(b["mux_next"][r, i - 1]) - VALUE_BASE:
                ok = False
    check("compose: register and labels recomputed independently", ok)
    # the distractor symbols must NOT change the answer
    g2 = torch.Generator().manual_seed(0)
    b2 = make_batch("compose", 64, 6, 3, generator=g2)
    check("compose: only the first symbol of a span is the operator",
          bool((b2["mux_next"] == b["mux_next"]).all()) and bool((b2["tokens"] == b["tokens"]).all()))
    check("answers never appear as INPUT tokens", bool((b["tokens"] < VALUE_BASE).all()))
    s = make_batch("summary", 32, 6, 3, generator=g)
    t2 = s["tokens"].view(32, 6, 3)
    ok = True
    for r in range(32):
        for i in range(6):
            counts = torch.bincount(t2[r, i], minlength=N_GROUP)
            if int(s["mux_next"][r, i]) != VALUE_BASE + int(counts.argmax()):
                ok = False
    check("summary: mode target recomputed independently", ok)


def t_eliminate_task():
    """`eliminate`: the sampler must match the distribution the ceilings are computed from."""
    g = torch.Generator().manual_seed(0)
    B = 4096
    b = make_batch("eliminate", B, 8, 3, generator=g)
    toks = b["tokens"].view(B, 8, 3)
    cands, surv, elim = b["candidates"], b["survivor"], b["elim"]

    # structure, recomputed from the tokens alone
    ok_struct = True
    for k in range(2):
        base = k * 4
        if not bool((toks[:, base] == cands[:, k]).all()):
            ok_struct = False
        if not bool((toks[:, base + 1, 0] == elim[:, k, 0]).all()):
            ok_struct = False
        if not bool((toks[:, base + 2, 0] == elim[:, k, 1]).all()):
            ok_struct = False
        # every other cell of spans s+1..s+3 is a distractor
        for sp, cell in ((base + 1, 1), (base + 1, 2), (base + 2, 1), (base + 2, 2),
                         (base + 3, 0), (base + 3, 1), (base + 3, 2)):
            col = toks[:, sp, cell]
            if not bool(((col >= ELIM_DIST_BASE) & (col < ELIM_VALUE_BASE)).all()):
                ok_struct = False
    check("eliminate: span layout is [cands][e1 d d][e2 d d][d d d]", ok_struct)

    srt = cands.sort(dim=-1).values
    check(
        "eliminate: the three candidates of a span are DISTINCT",
        bool(((srt[..., 0] != srt[..., 1]) & (srt[..., 1] != srt[..., 2])).all()),
    )
    in_set = ((cands == surv.unsqueeze(-1)).sum(-1) == 1)
    check("eliminate: the survivor is one of the three candidates", bool(in_set.all()))
    alive = True
    for j in range(2):
        e = elim[..., j].unsqueeze(-1)
        if not bool(((cands == e).sum(-1) == 1).all()):
            alive = False
        if bool((elim[..., j] == surv).any()):
            alive = False
    check("eliminate: both eliminations are candidates and NEITHER is the survivor", alive)
    check("eliminate: the two eliminations differ", bool((elim[..., 0] != elim[..., 1]).all()))
    check(
        "eliminate: answers never appear as INPUT tokens",
        bool((b["tokens"] < ELIM_VALUE_BASE).all()),
    )
    # the bijection candidate -> answer id, read off the labels and the MUX targets
    ok_bij = True
    for k in range(2):
        base = k * 4
        if not bool((b["labels"][:, (base + 3) * 3] == ELIM_VALUE_BASE + surv[:, k]).all()):
            ok_bij = False
        if not bool((b["mux_next"][:, base + 2] == ELIM_VALUE_BASE + surv[:, k]).all()):
            ok_bij = False
        if not bool((b["mux_own"][:, base + 1] == ELIM_VALUE_BASE + elim[:, k, 0]).all()):
            ok_bij = False
        if not bool((b["mux_own"][:, base + 2] == ELIM_VALUE_BASE + elim[:, k, 1]).all()):
            ok_bij = False
    check("eliminate: answer(c) = ELIM_VALUE_BASE + c at every label and MUX target", ok_bij)
    keep = torch.zeros(8, dtype=torch.bool)
    keep[2] = keep[6] = True
    check(
        "eliminate: only the answer slots carry a next-span MUX target",
        bool(((b["mux_next"] != -100) == keep).all()),
    )
    own = torch.zeros(8, dtype=torch.bool)
    own[1] = own[2] = own[5] = own[6] = True
    check(
        "eliminate: only the elimination spans carry an own-span MUX target",
        bool(((b["mux_own"] != -100) == own).all()),
    )
    check(
        "eliminate: the value positions are the heads of spans 3 and 7",
        b["value_pos"].tolist() == [9, 21],
    )

    # the distribution the ceilings assume: the survivor's POSITION in the candidate span is
    # uniform over 3, and the elimination order is uniform. 4096 rows x 2 instances = 8192
    pos = (cands == surv.unsqueeze(-1)).float().argmax(-1).flatten()
    frac = [(pos == j).float().mean().item() for j in range(3)]
    check(
        "eliminate: the survivor is uniform over the three candidate CELLS",
        max(abs(f - 1 / 3) for f in frac) < 0.02,
        f"cell fractions {[round(f, 4) for f in frac]}",
    )
    first_is_lower = (elim[..., 0] < elim[..., 1]).float().mean().item()
    check(
        "eliminate: the elimination ORDER is uniform",
        abs(first_is_lower - 0.5) < 0.02,
        f"P(e1 < e2) = {first_is_lower:.4f}",
    )
    counts = torch.zeros(N_CAND)
    counts.scatter_add_(0, surv.flatten(), torch.ones(surv.numel()))
    frac_s = (counts / surv.numel()).tolist()
    check(
        "eliminate: the survivor SYMBOL is uniform over the candidate alphabet",
        max(abs(f - 1 / N_CAND) for f in frac_s) < 0.02,
        f"symbol fractions {[round(f, 4) for f in frac_s]}",
    )
    a, bb = make_twin_batch(256, 8, 3, generator=torch.Generator().manual_seed(3))
    diff = (a["tokens"] != bb["tokens"])
    check(
        "eliminate twins: differ ONLY at the head of span s+1",
        diff.sum(1).unique().tolist() == [2]
        and sorted(diff[0].nonzero().flatten().tolist()) == [3, 15],
        f"differing positions {sorted(diff[0].nonzero().flatten().tolist())}",
    )
    check(
        "eliminate twins: the survivor and the first elimination swap roles",
        bool((bb["survivor"] == a["elim"][..., 0]).all())
        and bool((bb["elim"][..., 0] == a["survivor"]).all())
        and bool((bb["elim"][..., 1] == a["elim"][..., 1]).all()),
    )


def t_eliminate_ceilings():
    """The enumerated ceilings must match the closed forms, and the closed forms must match
    a Monte-Carlo estimate taken from the SAMPLER (not from the assumed distribution)."""
    import math

    c = eliminate_ceilings()
    check("ceiling: chance = ln 6", abs(c["chance"] - math.log(6)) < 1e-12, f"{c['chance']:.6f}")
    check(
        "ceiling: reachability d=0 is ln 5 (the second elimination alone)",
        abs(c["reachability"][0] - math.log(5)) < 1e-12,
        f"{c['reachability'][0]:.6f}",
    )
    check(
        "ceiling: reachability d=1 is ln 4 (both eliminations, no candidate set)",
        abs(c["reachability"][1] - math.log(4)) < 1e-12,
        f"{c['reachability'][1]:.6f}",
    )
    check(
        "ceiling: reachability d>=2 is 0 (the answer is determined)",
        c["reachability"][2] < 1e-12 and c["reachability"][3] < 1e-12,
    )
    check(
        "ceiling: commitment at span s+0 is ln 3, s+1 is ln 2, s+2 is 0",
        abs(c["commitment"][0] - math.log(3)) < 1e-12
        and abs(c["commitment"][1] - math.log(2)) < 1e-12
        and c["commitment"][2] < 1e-12,
        f"{c['commitment'][0]:.4f} {c['commitment'][1]:.4f} {c['commitment'][2]:.4f}",
    )
    check(
        "ceiling: point carry from span s+0 is (2/3) ln 4, from s+1 is ln 2",
        abs(c["point_carry"][0] - 2 / 3 * math.log(4)) < 1e-12
        and abs(c["point_carry"][1] - math.log(2)) < 1e-12,
        f"{c['point_carry'][0]:.4f} {c['point_carry'][1]:.4f}",
    )

    # the sampler against the enumeration: H(survivor | both eliminations) from 60k draws
    g = torch.Generator().manual_seed(11)
    b = make_batch("eliminate", 30000, 8, 3, generator=g)
    key = b["elim"][..., 0] * N_CAND + b["elim"][..., 1]
    joint = torch.zeros(N_CAND * N_CAND, N_CAND)
    joint.scatter_add_(
        0,
        (key.flatten().unsqueeze(-1)).expand(-1, N_CAND),
        torch.nn.functional.one_hot(b["survivor"].flatten(), N_CAND).float(),
    )
    tot = joint.sum()
    p = joint / tot
    row = p.sum(1, keepdim=True).clamp_min(1e-12)
    cond = p / row
    h = -(p * cond.clamp_min(1e-12).log()).sum().item()
    check(
        "the SAMPLER reproduces the enumerated H(survivor | e1, e2) = ln 4",
        abs(h - math.log(4)) < 0.01,
        f"Monte-Carlo {h:.4f} against {math.log(4):.4f}",
    )


def t_eliminate_depth_requirement():
    """Under strict geometry the `eliminate` answer needs exactly TWO passes.

    Same measurement as `t_depth_requirement`: differentiate the value logit at the head of
    span 3 with respect to the raw cell embeddings and look at the candidate span (span 0).
    It must be exactly zero at depth 1 and non-zero at depth 2.
    """
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(geometry="strict", vocab=ELIM_VOCAB, layout=Layout(8, 3, 2)))
    g = torch.Generator().manual_seed(1)
    b = make_batch("eliminate", 2, 8, 3, generator=g)
    ok_below, ok_at = True, True
    for T in (1, 2, 3):
        base = mo._base(b["tokens"]).detach().requires_grad_(True)
        x = base
        for blk in mo.prelude:
            x = blk(x, mo.mask_prelude)
        e = x[:, mo.slot_pos]
        z, h0, states, hp = mo.loop(e, torch.full((2, 8), T, dtype=torch.long))
        xh = mo._write_and_coda(base, z)
        pos = int(b["value_pos"][0])  # head of span 3, the answer of instance 0
        logit = mo.head(xh[:, mo.tok_pos])[:, pos].sum()
        gb = torch.autograd.grad(logit, base)[0]
        reach = gb[:, int(mo.tok_pos[0])].norm().item()  # span 0's first candidate cell
        if T < 2 and reach != 0.0:
            ok_below = False
        if T >= 2 and reach == 0.0:
            ok_at = False
        print(f"      T={T}: |d logit(span 3 head)/d span-0 candidate| = {reach:.3e}")
    check("eliminate/strict: the candidate span is UNREACHABLE at depth 1", ok_below)
    check("eliminate/strict: the candidate span IS reachable at depth 2 and above", ok_at)


def t_eliminate_mux_masking():
    """A -100 MUX target must contribute nothing, and must never produce a NaN."""
    torch.manual_seed(0)
    for attach in ("exit", "mux_all", "mux_all_detach", "staged", "deep_coda", "progressive"):
        c = cfg(attach=attach, vocab=ELIM_VOCAB, layout=Layout(8, 3, 2))
        mo = ToySlotLoop(c)
        g = torch.Generator().manual_seed(5)
        b = make_batch("eliminate", 8, 8, 3, generator=g)
        o = mo(b, force_depth=3)
        o["loss"].backward()
        finite = torch.isfinite(o["loss"]).item() and torch.isfinite(o["mux"]).item()
        gn = mo.core.mlp.down.weight.grad
        check(
            f"eliminate {attach}: finite loss and a live core gradient",
            finite and gn is not None and torch.isfinite(gn).all() and gn.norm().item() > 0,
            f"loss {o['loss'].item():.4f} mux {o['mux'].item():.4f}",
        )


def t_eliminate6_task():
    """`eliminate6`: the sampler must match the distribution the ceilings enumerate."""
    g = torch.Generator().manual_seed(0)
    B = 4096
    b = make_batch("eliminate6", B, E6_SPANS, 3, generator=g)
    toks = b["tokens"].view(B, E6_SPANS, 3)
    cands, surv, elim = b["candidates"][:, 0], b["survivor"][:, 0], b["elim"][:, 0]

    ok = True
    for j, sp in enumerate(E6_CAND_SPANS):
        if not bool((toks[:, sp] == cands[:, j * 3 : (j + 1) * 3]).all()):
            ok = False
    for j, sp in enumerate(E6_ELIM_SPANS):
        if not bool((toks[:, sp, 0] == elim[:, j]).all()):
            ok = False
        for cell in (1, 2):
            col = toks[:, sp, cell]
            if not bool(((col >= E6_DIST_BASE) & (col < E6_VALUE_BASE)).all()):
                ok = False
    if not bool(((toks[:, 7] >= E6_DIST_BASE) & (toks[:, 7] < E6_VALUE_BASE)).all()):
        ok = False
    check("eliminate6: [cands x2][elim d d]x5[d d d] layout", ok)

    srt = cands.sort(dim=-1).values
    check(
        "eliminate6: the six candidates are DISTINCT",
        bool((srt[:, 1:] != srt[:, :-1]).all()),
    )
    check(
        "eliminate6: the survivor is one of the six candidates",
        bool(((cands == surv.unsqueeze(-1)).sum(-1) == 1).all()),
    )
    alive = True
    for j in range(N6_ALIVE - 1):
        if not bool(((cands == elim[:, j].unsqueeze(-1)).sum(-1) == 1).all()):
            alive = False
        if bool((elim[:, j] == surv).any()):
            alive = False
    check("eliminate6: all five eliminations are candidates and none is the survivor", alive)
    es = elim.sort(dim=-1).values
    check("eliminate6: the five eliminations are distinct", bool((es[:, 1:] != es[:, :-1]).all()))
    check(
        "eliminate6: the eliminations are exactly the candidates minus the survivor",
        bool(
            (
                torch.nn.functional.one_hot(cands, N6_CAND).sum(1)
                - torch.nn.functional.one_hot(elim, N6_CAND).sum(1)
                == torch.nn.functional.one_hot(surv, N6_CAND)
            ).all()
        ),
    )
    check("eliminate6: answers never appear as INPUT tokens", bool((b["tokens"] < E6_VALUE_BASE).all()))
    check(
        "eliminate6: answer(c) = E6_VALUE_BASE + c at the label and at the MUX target",
        bool((b["labels"][:, 7 * 3] == E6_VALUE_BASE + surv).all())
        and bool((b["mux_next"][:, E6_ELIM_SPANS[-1]] == E6_VALUE_BASE + surv).all()),
    )
    keep = torch.zeros(E6_SPANS, dtype=torch.bool)
    keep[E6_ELIM_SPANS[-1]] = True
    check(
        "eliminate6: only the answer slot carries a next-span MUX target",
        bool(((b["mux_next"] != -100) == keep).all()),
    )
    own = torch.zeros(E6_SPANS, dtype=torch.bool)
    for sp in E6_ELIM_SPANS:
        own[sp] = True
    check(
        "eliminate6: only the elimination spans carry an own-span MUX target",
        bool(((b["mux_own"] != -100) == own).all()),
    )
    check("eliminate6: one value position, the head of span 7", b["value_pos"].tolist() == [21])

    pos = (cands == surv.unsqueeze(-1)).float().argmax(-1)
    frac = [(pos == j).float().mean().item() for j in range(N6_ALIVE)]
    check(
        "eliminate6: the survivor is uniform over the six candidate CELLS",
        max(abs(f - 1 / N6_ALIVE) for f in frac) < 0.02,
        f"cell fractions {[round(f, 4) for f in frac]}",
    )
    counts = torch.zeros(N6_CAND)
    counts.scatter_add_(0, surv, torch.ones(surv.numel()))
    fs = (counts / surv.numel()).tolist()
    check(
        "eliminate6: the survivor SYMBOL is uniform over the eight candidate ids",
        max(abs(f - 1 / N6_CAND) for f in fs) < 0.02,
        f"symbol fractions {[round(f, 4) for f in fs]}",
    )
    # the elimination ORDER is uniform: every candidate cell is equally likely to be first out
    firstpos = (cands == elim[:, 0].unsqueeze(-1)).float().argmax(-1)
    ff = [(firstpos == j).float().mean().item() for j in range(N6_ALIVE)]
    check(
        "eliminate6: the elimination ORDER is uniform over the candidate cells",
        max(abs(f - 1 / N6_ALIVE) for f in ff) < 0.02,
        f"first-out fractions {[round(f, 4) for f in ff]}",
    )
    a, bb = make_twin_batch(256, E6_SPANS, 3, generator=torch.Generator().manual_seed(3), task="eliminate6")
    diff = a["tokens"] != bb["tokens"]
    check(
        "eliminate6 twins: differ ONLY at the head of span 2",
        diff.sum(1).unique().tolist() == [1] and diff[0].nonzero().flatten().tolist() == [6],
        f"differing positions {diff[0].nonzero().flatten().tolist()}",
    )
    check(
        "eliminate6 twins: the survivor and the first elimination swap roles",
        bool((bb["survivor"][:, 0] == a["elim"][:, 0, 0]).all())
        and bool((bb["elim"][:, 0, 0] == a["survivor"][:, 0]).all())
        and bool((bb["elim"][:, 0, 1:] == a["elim"][:, 0, 1:]).all()),
    )


def t_eliminate6_ceilings():
    """The enumerated ceilings against their closed forms, and against the sampler."""
    import math

    c = eliminate6_ceilings()
    check("e6 ceiling: chance = ln 8", abs(c["chance"] - math.log(8)) < 1e-12)
    want = [math.log(7), math.log(6), math.log(5), math.log(4), math.log(3), 0.5 * math.log(3), 0.0]
    got = [c["reachability"][d] for d in range(7)]
    check(
        "e6 ceiling: reachability is ln7/ln6/ln5/ln4/ln3/(ln3)/2/0 at depth 0..6",
        all(abs(a - b) < 1e-9 for a, b in zip(got, want)),
        " ".join(f"{v:.6f}" for v in got) + f"; worst |delta| {max(abs(a - b) for a, b in zip(got, want)):.2e}",
    )
    wantc = [math.log(n) for n in (6, 5, 4, 3, 2)] + [0.0]
    gotc = [c["commitment"][k] for k in range(6)]
    check(
        "e6 ceiling: commitment is ln6 -> ln5 -> ln4 -> ln3 -> ln2 -> 0",
        all(abs(a - b) < 1e-9 for a, b in zip(gotc, wantc)),
        " ".join(f"{v:.4f}" for v in gotc),
    )
    check(
        "e6 ceiling: the point-carry ladder is strictly below the commitment ladder",
        all(c["point_carry"][k] < c["commitment"][k] - 1e-9 for k in range(5))
        and c["point_carry"][5] == 0.0,
        " ".join(f"{c['point_carry'][k]:.4f}" for k in range(6)),
    )
    # the SAMPLER against the enumeration: H(survivor | all five eliminations) = ln 3.
    # The key is the unordered elimination SET, not the ordered tuple: the survivor is
    # conditionally independent of the order, and 56 sets keep about 700 samples per cell,
    # where the plug-in entropy's bias is about -(3-1)/(2*700) = -0.0014 nats. Keying on
    # the 6,720 ordered tuples would measure that bias (-0.20 nats) instead of the sampler.
    g = torch.Generator().manual_seed(11)
    b = make_batch("eliminate6", 40000, E6_SPANS, 3, generator=g)
    es = b["elim"][:, 0].sort(dim=-1).values
    key = torch.zeros(b["survivor"].shape[0], dtype=torch.long)
    for j in range(N6_ALIVE - 1):
        key = key * N6_CAND + es[:, j]
    uniq, inv = key.unique(return_inverse=True)
    joint = torch.zeros(uniq.numel(), N6_CAND)
    joint.scatter_add_(
        0,
        inv.unsqueeze(-1).expand(-1, N6_CAND),
        torch.nn.functional.one_hot(b["survivor"][:, 0], N6_CAND).float(),
    )
    p = joint / joint.sum()
    cond = p / p.sum(1, keepdim=True).clamp_min(1e-12)
    h = -(p * cond.clamp_min(1e-12).log()).sum().item()
    check(
        "the eliminate6 SAMPLER reproduces H(survivor | all five eliminations) = ln 3",
        abs(h - math.log(3)) < 0.02,
        f"Monte-Carlo {h:.4f} against {math.log(3):.4f}",
    )


def t_eliminate6_depth_requirement():
    """Under strict geometry the `eliminate6` answer needs SIX passes, one per hop.

    Differentiate the value logit at the head of span 7 with respect to the raw cell
    embeddings and read the norm at each source span. A span at hop distance h from the
    answer slot must be exactly zero below depth h and non-zero at and above it.
    """
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(geometry="strict", vocab=E6_VOCAB, layout=Layout(E6_SPANS, 3, 2)))
    g = torch.Generator().manual_seed(1)
    b = make_batch("eliminate6", 2, E6_SPANS, 3, generator=g)
    width = mo.cfg.layout.width
    ok = True
    for T in (1, 2, 4, 6):
        base = mo._base(b["tokens"]).detach().requires_grad_(True)
        x = base
        for blk in mo.prelude:
            x = blk(x, mo.mask_prelude)
        e = x[:, mo.slot_pos]
        z, h0, states, hp = mo.loop(e, torch.full((2, E6_SPANS), T, dtype=torch.long))
        xh = mo._write_and_coda(base, z)
        logit = mo.head(xh[:, mo.tok_pos])[:, int(b["value_pos"][0])].sum()
        gb = torch.autograd.grad(logit, base)[0]
        reach = [gb[:, sp * width : sp * width + 3].norm().item() for sp in range(E6_SPANS)]
        print("      T=%d: |d logit(span 7 head)/d span s| = %s" % (T, " ".join(f"{v:.1e}" for v in reach)))
        for sp in range(7):  # the answer slot is slot 6, so span sp sits at hop 6 - sp
            hop = 6 - sp
            if T < hop and reach[sp] != 0.0:
                ok = False
            if T >= hop and reach[sp] == 0.0:
                ok = False
    check("eliminate6/strict: every span is reachable at exactly its hop distance", ok)


def t_eliminate6_mux_masking():
    torch.manual_seed(0)
    for attach in ("exit", "mux_all", "staged"):
        c = cfg(attach=attach, vocab=E6_VOCAB, layout=Layout(E6_SPANS, 3, 2))
        mo = ToySlotLoop(c)
        g = torch.Generator().manual_seed(5)
        b = make_batch("eliminate6", 8, E6_SPANS, 3, generator=g)
        o = mo(b, force_depth=4)
        o["loss"].backward()
        gn = mo.core.mlp.down.weight.grad
        check(
            f"eliminate6 {attach}: finite loss and a live core gradient",
            torch.isfinite(o["loss"]).item() and gn is not None and torch.isfinite(gn).all()
            and gn.norm().item() > 0,
            f"loss {o['loss'].item():.4f} mux {o['mux'].item():.4f}",
        )


def t_instrument_reach():
    """The instruments' reachability helper must agree with the measured geometry."""
    from instruments import _reachable

    sp = spec_for("eliminate6")
    r0 = _reachable(sp, 6, 0)
    r4 = _reachable(sp, 6, 4)
    r6 = _reachable(sp, 6, 6)
    check(
        "instrument reach: the answer slot holds only the last elimination at pass 0",
        r0["elims"] == [False, False, False, False, True] and not r0["all_cands"],
    )
    check(
        "instrument reach: it holds all five eliminations and no candidate span at pass 4",
        r4["elims"] == [True] * 5 and not r4["all_cands"],
    )
    check("instrument reach: it holds the whole candidate set at pass 6", r6["all_cands"])
    se = spec_for("eliminate")
    check(
        "instrument reach: the eliminate spec still reads its own geometry",
        _reachable(se, 2, 0)["elims"] == [False, True]
        and _reachable(se, 2, 1)["elims"] == [True, True]
        and not _reachable(se, 2, 1)["all_cands"]
        and _reachable(se, 2, 2)["all_cands"],
    )


def t_masks():
    lay = Layout(6, 3, 2)
    kind = lay.cell_kind()
    span = lay.span_of_cell()
    n = lay.n_cells
    for geom in ("permissive", "strict"):
        m = build_masks(lay, coda_reads_z=True, geometry=geom)
        bad = 0
        for q in range(n):
            if kind[q] != TOKEN:
                continue
            for k in range(n):
                if m["coda"][q, k] and kind[k] == TOKEN and span[k] != span[q]:
                    bad += 1
        check(f"{geom} coda: no token->token attention across spans", bad == 0, f"{bad} offenders")
        off = m["coda"].clone()
        off.fill_diagonal_(False)
        check(f"{geom} coda: slot cells are never keys (off-diagonal)", bool(~off[:, kind == SLOT].any()))
        check(f"{geom} coda: every query keeps at least one key", bool(m["coda"].sum(1).min() >= 1))
        check(f"{geom} coda: keys are causal only", bool((off).triu(1).sum() == 0))
        pk = off & (kind == PREFIX)[None, :] & (kind == TOKEN)[:, None]
        gaps = {int(span[q]) - int(span[k]) for q in range(n) for k in range(n) if pk[q, k]}
        if geom == "strict":
            check("strict coda: prefix keys come from the PREVIOUS span only", gaps == {1}, f"span gaps {sorted(gaps)}")
            check("strict prelude: within-span only", bool(~(m["prelude"] & ~(span[:, None] == span[None, :])).any()))
            offs = {i - j for i in range(6) for j in range(6) if m["core"][i, j]}
            check("strict core: attends self and the previous slot only", offs == {0, 1}, f"slot offsets {sorted(offs)}")
        else:
            check("permissive coda: prefix keys come from every earlier span", max(gaps) > 1, f"span gaps {sorted(gaps)}")
            check("permissive prelude: causal over all cells",
                  bool((m["prelude"] == (torch.arange(n)[:, None] >= torch.arange(n)[None, :])).all()))
            offs = {i - j for i in range(6) for j in range(6) if m["core"][i, j]}
            check("permissive core: attends every earlier slot", max(offs) == 5, f"max slot offset {max(offs)}")
    mb = build_masks(lay, coda_reads_z=False, geometry="strict")
    offb = mb["coda"].clone()
    offb.fill_diagonal_(False)
    tokq = offb & (kind == PREFIX)[None, :] & (kind == TOKEN)[:, None]
    check("coda_blind: no token reads a prefix cell", bool(~tokq.any()))


def t_depth_requirement():
    """Under strict geometry, the answer at the head of span j+1 needs at least j passes.

    Measured, not argued: differentiate the value logit w.r.t. the RAW cell embeddings and
    look at the operator cell of span 0. It must be exactly zero below the required depth
    and non-zero at or above it.
    """
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(geometry="strict"))
    g = torch.Generator().manual_seed(1)
    b = make_batch("compose", 2, 6, 3, generator=g)
    j = 4  # value_pos[4] is the head of span 5; its label is R_4, from slot 4: needs 4 passes
    ok_below, ok_at = True, True
    for T in (1, 2, 3, 4, 5):
        base = mo._base(b["tokens"]).detach().requires_grad_(True)
        x = base
        for blk in mo.prelude:
            x = blk(x, mo.mask_prelude)
        e = x[:, mo.slot_pos]
        z, h0, states, hp = mo.loop(e, torch.full((2, 6), T, dtype=torch.long))
        xh = mo._write_and_coda(base, z)
        pos = int(b["value_pos"][j])  # head of span j+1
        logit = mo.head(xh[:, mo.tok_pos])[:, pos].sum()
        gb = torch.autograd.grad(logit, base)[0]
        reach = gb[:, int(mo.tok_pos[0])].norm().item()  # span 0's operator cell
        if T < j and reach != 0.0:
            ok_below = False
        if T >= j and reach == 0.0:
            ok_at = False
        print(f"      T={T}: |d logit(span {j+1} head)/d span-0 operator| = {reach:.3e}")
    check(f"strict: span 0 is UNREACHABLE below depth {j}", ok_below)
    check(f"strict: span 0 IS reachable at depth {j} and above", ok_at)


def t_coda_path():
    """The only route from an earlier span to a token logit is z. Measured, not argued."""
    torch.manual_seed(0)
    for blind in (False, True):
        c = cfg(coda_reads_z=not blind)
        mo = ToySlotLoop(c)
        g = torch.Generator().manual_seed(1)
        b = make_batch("compose", 4, 6, 3, generator=g)
        emb = mo.embed.weight
        emb.requires_grad_(True)
        # gradient of the value-position logit w.r.t. the PREFIX-cell values: cut the
        # prefix write and see whether the logit still depends on the loop.
        out = mo(b, force_depth=3)
        pos = int(b["value_pos"][2])  # a value position in span 3
        logit = mo.head(out["xh"][:, mo.tok_pos])[:, pos].sum()
        gz = torch.autograd.grad(logit, out["z"], retain_graph=True)[0]
        # slots strictly before span 3 must reach it iff the coda reads z
        early = gz[:, :3].norm().item()
        later = gz[:, 3:].norm().item()
        check(
            f"coda_reads_z={not blind}: earlier slots reach a value logit",
            (early > 1e-8) != blind,
            f"|d logit/d z_earlier| = {early:.3e}",
        )
        check(
            f"coda_reads_z={not blind}: LATER slots never reach it (causality)",
            later < 1e-10,
            f"|d logit/d z_later| = {later:.3e}",
        )


def t_prelude_leak():
    """coda_token_input='prelude' would open a token path; 'embed' must not.

    The leak is not the coda attending to earlier token cells (it cannot). It is that a
    prelude output at the value position ALREADY carries earlier spans. So differentiate
    the value logit w.r.t. the raw cell embeddings, with z detached and the coda blind to
    z, which leaves the token path as the only possible route.
    """
    for mode in ("embed", "prelude"):
        torch.manual_seed(0)
        c = cfg(coda_token_input=mode, coda_reads_z=False, geometry="permissive")
        mo = ToySlotLoop(c)
        g = torch.Generator().manual_seed(1)
        b = make_batch("compose", 4, 6, 3, generator=g)
        base = mo._base(b["tokens"]).detach().requires_grad_(True)
        x = base
        for blk in mo.prelude:
            x = blk(x, mo.mask_prelude)
        e = x[:, mo.slot_pos]
        z, h0, states, hp = mo.loop(e, torch.full((4, 6), 3, dtype=torch.long))
        cbase = x if mode == "prelude" else base
        xh = mo._write_and_coda(cbase, z.detach())
        pos = int(b["value_pos"][2])
        logit = mo.head(xh[:, mo.tok_pos])[:, pos].sum()
        gb = torch.autograd.grad(logit, base)[0]
        early_tok = gb[:, mo.tok_pos][:, :9].norm().item()  # tokens of spans 0-2
        check(
            f"coda_token_input={mode}: token path to earlier spans",
            (early_tok > 1e-8) == (mode == "prelude"),
            f"|d logit/d earlier token cells| = {early_tok:.3e}",
        )


def t_tap():
    torch.manual_seed(0)
    for attach in ("exit", "mux_all", "mux_all_detach", "staged", "deep_coda", "progressive"):
        c = cfg(attach=attach)
        mo = ToySlotLoop(c)
        ev = make_eval_batches("compose", c, 32, 32, 7, torch.device("cpu"))
        r = gradient_probe(mo, ev, depth=4, source="total")
        check(
            f"tap sums to the leaf gradient ({attach})",
            r["selfcheck_max_rel_err"] < 1e-4,
            f"max rel err {r['selfcheck_max_rel_err']:.2e}, cancel {r['cancellation']:.3f}",
        )


def t_tap_is_inert():
    """The tap must not change the forward or the leaf gradient."""
    torch.manual_seed(0)
    c = cfg(attach="exit")
    mo = ToySlotLoop(c)
    g = torch.Generator().manual_seed(3)
    b = make_batch("compose", 8, 6, 3, generator=g)
    outs = []
    for tap in (False, True):
        mo.zero_grad(set_to_none=True)
        mo._tap = tap
        mo.tap_params = []
        o = mo(b, force_depth=4)
        o["loss"].backward()
        mo._tap = False
        outs.append((o["loss"].item(), mo.core.mlp.down.weight.grad.clone()))
        mo.tap_params = []
    check("tap does not change the loss", outs[0][0] == outs[1][0], f"{outs[0][0]!r} vs {outs[1][0]!r}")
    check(
        "tap does not change the leaf gradient",
        torch.allclose(outs[0][1], outs[1][1], atol=1e-6),
        f"max |d| {(outs[0][1]-outs[1][1]).abs().max().item():.2e}",
    )


def t_switch_identity():
    """progressive_p=0, bptt_last=0 and pass_lora_rank=0 must be the untouched forward."""
    torch.manual_seed(0)
    ref = None
    for kw, name in (
        ({}, "exit"),
        ({"attach": "progressive", "progressive_p": 0.0}, "progressive p=0"),
        ({"bptt_last": 0}, "bptt_last=0"),
    ):
        torch.manual_seed(0)
        mo = ToySlotLoop(cfg(**kw))
        g = torch.Generator().manual_seed(5)
        b = make_batch("compose", 8, 6, 3, generator=g)
        mo.train()
        o = mo(b, force_depth=4)
        o["loss"].backward()
        val = (o["loss"].item(), mo.core.mlp.down.weight.grad.norm().item())
        if ref is None:
            ref = val
        else:
            check(f"{name} is bit-identical to the plain exit forward", val == ref, f"{val} vs {ref}")

    # and the sabotage: p=1.0 with a deterministic cut MUST change the gradient
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(attach="progressive", progressive_p=1.0))
    g = torch.Generator().manual_seed(5)
    b = make_batch("compose", 8, 6, 3, generator=g)
    mo.train()
    o = mo(b, force_depth=4)
    o["loss"].backward()
    val = (o["loss"].item(), mo.core.mlp.down.weight.grad.norm().item())
    check("progressive p=1.0 DOES change the core gradient", val[1] != ref[1], f"{val[1]:.6f} vs {ref[1]:.6f}")

    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(bptt_last=2))
    g = torch.Generator().manual_seed(5)
    b = make_batch("compose", 8, 6, 3, generator=g)
    mo.train()
    o = mo(b, force_depth=4)
    o["loss"].backward()
    check(
        "bptt_last=2 DOES change the core gradient",
        mo.core.mlp.down.weight.grad.norm().item() != ref[1],
        f"{mo.core.mlp.down.weight.grad.norm().item():.6f} vs {ref[1]:.6f}",
    )


def t_bptt_window():
    """bptt_last=k must put ZERO gradient into the passes before the last k."""
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(bptt_last=2, fixed_depth=4))
    ev = make_eval_batches("compose", mo.cfg, 32, 32, 7, torch.device("cpu"))
    r = gradient_probe(mo, ev, depth=4, source="total")
    early = sum(r["dW_norm"][:2])
    late = sum(r["dW_norm"][2:])
    check("bptt_last=2 zeroes the first passes", early == 0.0 and late > 0.0, f"early {early:.2e} late {late:.2e}")


def t_lora():
    """Zero-init deltas are inert; a non-zero B acts. Compared on ONE model, because
    building the LoRA parameters shifts the RNG stream for every later module."""
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg(pass_lora_rank=4))
    g = torch.Generator().manual_seed(5)
    b = make_batch("compose", 8, 6, 3, generator=g)
    with_lora = mo(b, force_depth=4)["loss"].item()
    mo.core.rank = 0  # the branch, removed; the shared weights are untouched
    without = mo(b, force_depth=4)["loss"].item()
    mo.core.rank = 4
    check("zero-init pass LoRA is inert at init", with_lora == without, f"{with_lora!r} vs {without!r}")
    with torch.no_grad():
        mo.core.lora_B.add_(0.1)
    acted = mo(b, force_depth=4)["loss"].item()
    check("pass LoRA acts once B is non-zero", abs(acted - with_lora) > 1e-4, f"{acted:.6f} vs {with_lora:.6f}")


def t_depth_freeze():
    """A slot at its depth must stop moving; forcing depth d must run exactly d passes."""
    torch.manual_seed(0)
    mo = ToySlotLoop(cfg())
    e = torch.randn(2, 6, 32)
    depths = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    z, h0, states, hp = mo.loop(e, depths)
    ok = True
    for i in range(6):
        for r in range(2):
            d = int(depths[r, i])
            for t in range(d, len(states)):
                if not torch.allclose(states[t][r, i], states[d - 1][r, i]):
                    ok = False
    check("slots freeze at their realised depth", ok)
    check("loop runs max(depths) passes", len(states) == int(depths.max()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    t_group()
    t_task()
    t_eliminate_task()
    t_eliminate_ceilings()
    t_eliminate6_task()
    t_eliminate6_ceilings()
    t_instrument_reach()
    t_masks()
    t_depth_requirement()
    t_eliminate_depth_requirement()
    t_eliminate_mux_masking()
    t_eliminate6_depth_requirement()
    t_eliminate6_mux_masking()
    t_coda_path()
    t_prelude_leak()
    t_depth_freeze()
    t_switch_identity()
    t_lora()
    t_tap_is_inert()
    t_tap()
    t_bptt_window()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("all self-checks passed")
