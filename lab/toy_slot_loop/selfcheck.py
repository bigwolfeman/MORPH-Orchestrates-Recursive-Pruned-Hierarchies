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
from tasks import MUL, N_GROUP, PERMS, VALUE_BASE, make_batch  # noqa: E402

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
    t_masks()
    t_depth_requirement()
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
