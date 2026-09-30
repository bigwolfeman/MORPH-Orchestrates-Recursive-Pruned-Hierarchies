"""``tul.fan_route`` (arms R "reader" and T "latent", 2026-09-30): a top-1 router over the
write-all fan's M cells; the coda reads only the router's pick.

Files: morph/model/tul_fan_route.py (FanRouter, FanTeacherMap), morph/model/transformer.py
(`_tul_fan_route`, `_fan_route_cells`, the routed write at the register's 1:1 seam, the
`route` argument of `_tul_fan_stream_write` / `_tul_fan_oracle`, the `rlat_weighted`
fold, `tul_fan_after_step`, `tul_fan_reset_route_load`).
Note: .agents/notes/proposed/architecture/2026-09-30-fan-opf-and-routers.md

What each test pins:
  * Only the winner reaches the coda: perturb a LOSER cell inside the write and every
    logit is bit-equal; perturb the winner and they move (R and T).
  * The write: R scales the winner by p_winner, T writes it unscaled; a loser's prefix
    cell is exactly zero (no E_pass under prefix_source exit).
  * The losers stay visible to the loop: the loop runs before the write, on the cells.
  * R: the reader's CE reaches the router. T: the router's gradient is EXACTLY the
    teacher term's (the reader's CE contributes nothing), and the teacher term sends no
    gradient into the cells.
  * T's teacher pick is argmin ||g(sg cell) - z||^2 (hand check); the router CE's target
    is that pick; `teacher_router_agree` is its hand value.
  * The bias: its update direction, it chooses and never enters p, it is checkpointed
    (the pending counts are not), and counts accrue only in a training forward with grad.
  * The selection is the same function at train and eval.
  * The val oracle's router instruments equal a hand recomputation from the per-cell
    coda CE table the oracle built.
  * RNG-neutral build: every shared weight equals nowta's.

CPU, fp32, the tests/test_tul_fan.py / tests/test_tul_lxfan.py strict fixtures.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import morph.model.transformer as T
from morph.model.transformer import MORPHTransformer, span_ce_index, span_token_counts
from morph.model.tul_fan_route import FanRouter
from test_tul_fan import _batch
from test_tul_lx_credit import M
from test_tul_lxfan import _build, _fan_kw, _Spy


def _kw(route: str, **kw) -> dict:
    base = dict(spandec=True, fan_all_wta_lambda=0.0, fan_route=route)
    base.update(kw)
    return _fan_kw(M, **base)


def _arm(route: str, model_kw: dict | None = None, **kw) -> MORPHTransformer:
    m = _build(model_kw=model_kw, **_kw(route, **kw))
    assert m.tul_fan_target_build() == (route == "latent")
    return m


def _route_call(m, inp, lab, layout, seed: int = 5):
    spy = _Spy(m, "_tul_fan_route")
    torch.manual_seed(seed)
    out = m(inp, labels=lab, slot_layout=layout)
    delattr(m, "_tul_fan_route")
    assert len(spy.calls) == 1
    return out, spy.calls[0][0], spy.outs[0]


# ── 1. only the winner reaches the coda ─────────────────────────────────────────────


@pytest.mark.parametrize("route", ["reader", "latent"])
def test_a_loser_cell_never_reaches_the_coda_and_the_winner_does(route):
    _ids0, inp, _lab, layout = _batch(M)
    m = _arm(route).eval()
    orig = m._fan_route_cells
    which = {"target": None}

    def _perturbed(cells, rt):
        w = rt["winner"]
        c = cells.clone()
        # pick, per slot, one cell that is (loser) or is not (winner) the router's pick
        idx = w if which["target"] == "winner" else (w + 1) % M
        sel = F.one_hot(idx, M).to(c.dtype).view(*idx.shape, M, *([1] * (c.dim() - 3)))
        c = c + 3.0 * sel * torch.randn_like(c)
        return orig(c, rt)

    with torch.no_grad():
        base = m(inp, labels=None, slot_layout=layout)["logits"]
        m._fan_route_cells = _perturbed
        which["target"] = "loser"
        lose = m(inp, labels=None, slot_layout=layout)["logits"]
        which["target"] = "winner"
        win = m(inp, labels=None, slot_layout=layout)["logits"]
    del m._fan_route_cells
    assert torch.equal(lose, base)
    assert not torch.allclose(win, base)


@pytest.mark.parametrize("route", ["reader", "latent"])
def test_the_routed_write_scales_under_reader_only_and_a_loser_cell_is_zero(route):
    m = _arm(route)
    torch.manual_seed(3)
    B, S = 2, 5
    cells = torch.randn(B, S, M, 4, 64)
    winner = torch.randint(0, M, (B, S))
    p = torch.softmax(torch.randn(B, S, M), dim=-1)
    rt = {"winner": winner, "scale": p if route == "reader" else None}
    out = m._fan_route_cells(cells, rt)
    for b in range(B):
        for s in range(S):
            for i in range(M):
                if i == int(winner[b, s]):
                    exp = cells[b, s, i] * (p[b, s, i] if route == "reader" else 1.0)
                    assert torch.allclose(out[b, s, i], exp)
                else:
                    assert torch.equal(out[b, s, i], torch.zeros_like(out[b, s, i]))
    # Through the 1:1 write a loser's prefix cell is EXACTLY zero: these arms keep
    # nowta's `prefix_source: exit`, which builds no E_pass (the per-index constant a
    # blank would otherwise carry).
    assert m.tul.E_pass is None
    _ids0, _inp, _lab, layout = _batch(M)
    Sx = layout.slot_valid.shape[1]
    cells = torch.randn(layout.slot_valid.shape[0], Sx, M, 4, 64)
    rt = {"winner": torch.zeros(cells.shape[:2], dtype=torch.long),
          "scale": torch.full((*cells.shape[:2], M), 0.5) if route == "reader" else None}
    vals, _pos = m.tul.prefix_project(cells[:, :, 0], layout, 256,
                                      cells=m._fan_route_cells(cells, rt))
    vals = vals.view(cells.shape[0], Sx, M, 4, 64)
    for i in range(1, M):
        assert torch.equal(vals[:, :, i], torch.zeros_like(vals[:, :, i]))
    assert float(vals[:, :, 0].abs().sum()) > 0


def test_the_loop_runs_before_the_write_so_losers_stay_visible_to_later_slots():
    """The router reads the loop's EXIT cells: the loop (where cell i of slot k reads the
    cells of slots < k) has already run on every cell when the pick is made, and the
    pick only changes what the prefix write hands the coda. Checked by the call order:
    `_tul_core` returns before `_tul_fan_route` is called, and the cells the router gets
    are `_tul_core`'s own output, every cell present."""
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("reader").train()
    order = []
    core_spy = _Spy(m, "_tul_core")
    orig_route = m._tul_fan_route

    def _r(*a, **k):
        order.append("route")
        return orig_route(*a, **k)
    orig_core = m._tul_core

    def _c(*a, **k):
        order.append("core")
        return orig_core(*a, **k)
    m._tul_core = _c
    m._tul_fan_route = _r
    m(inp, labels=lab, slot_layout=layout)
    assert order == ["core", "route"]
    del core_spy


# ── 2. who trains the router ────────────────────────────────────────────────────────


def _router_params(m):
    return [p for p in m.tul_fan_router.parameters()]


def test_reader_arm_the_coda_ce_trains_the_router():
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("reader").train()
    out, _args, rt = _route_call(m, inp, lab, layout)
    assert rt["loss"] is None and "rlat_weighted" not in out
    g = torch.autograd.grad(out["loss"], _router_params(m), allow_unused=True)
    assert all(x is not None for x in g)
    assert sum(float(x.abs().sum()) for x in g) > 0


def test_latent_arm_the_router_learns_from_the_teacher_alone_and_no_cell_is_touched():
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("latent").train()
    out, args, rt = _route_call(m, inp, lab, layout)
    cells = args[0]
    params = _router_params(m)
    g_all = torch.autograd.grad(out["loss"], params, retain_graph=True, allow_unused=True)
    g_t = torch.autograd.grad(rt["loss"], params, retain_graph=True, allow_unused=True)
    for a, b in zip(g_all, g_t):
        assert b is not None and float(b.abs().sum()) > 0
        assert torch.allclose(a, b, atol=1e-7, rtol=0)
    g_c, = torch.autograd.grad(rt["loss"], cells, retain_graph=True, allow_unused=True)
    assert g_c is None or float(g_c.abs().sum()) == 0.0
    # g learns (its MSE), and gets nothing from the coda either
    gp = list(m.tul_fan_teacher.parameters())
    g_all = torch.autograd.grad(out["loss"], gp, retain_graph=True)
    g_t = torch.autograd.grad(rt["loss"], gp, retain_graph=True)
    for a, b in zip(g_all, g_t):
        assert float(b.abs().sum()) > 0 and torch.allclose(a, b, atol=1e-7, rtol=0)
    assert float(out["rlat_weighted"]) == pytest.approx(float(rt["loss"]))


def test_teacher_pick_router_target_and_agreement_by_hand():
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("latent", fan_rlat_lambda=0.5).train()
    out, args, rt = _route_call(m, inp, lab, layout)
    cells, xn, lay, tgt = args[0], args[1], args[2], args[3]
    cr = cells.detach()
    while cr.dim() > 4:
        cr = cr.mean(dim=-2)
    ln = F.layer_norm(cr, (cr.shape[-1],))
    gz = ln @ m.tul_fan_teacher.g.weight.t() + m.tul_fan_teacher.g.bias
    d2 = ((gz - tgt["z"].unsqueeze(2)) ** 2).mean(-1)
    teacher = d2.argmin(-1)
    assert torch.equal(rt["teacher"], teacher)
    ok = tgt["ok"]
    ctx = T.gather_valid(xn, lay.slot_index, lay.slot_valid).mean(dim=2)
    sc = m.tul_fan_router.scores(cr, ctx.detach())
    ce = F.cross_entropy(sc[ok], teacher[ok])
    mse = d2.mean(-1)[ok].mean()
    assert float(rt["loss"]) == pytest.approx(0.5 * float(ce + mse), rel=1e-5)
    agree = float((rt["winner"][ok] == teacher[ok]).float().mean())
    assert float(out["fan_teacher_router_agree"]) == pytest.approx(agree, abs=1e-6)
    assert float(out["fan_rlat_ce"]) == pytest.approx(float(ce), rel=1e-5)


# ── 3. the balance bias ─────────────────────────────────────────────────────────────


def test_bias_update_direction_and_its_edge_cases():
    r = FanRouter(16, 4, 8)
    r.pending.copy_(torch.tensor([7.0, 1.0, 1.0, 1.0]))
    r.balance_step_(0.01)
    assert torch.allclose(r.bias, torch.tensor([-0.01, 0.01, 0.01, 0.01]))
    assert float(r.pending.abs().sum()) == 0.0
    r.pending.copy_(torch.tensor([2.0, 2.0, 2.0, 2.0]))         # balanced: no move
    r.balance_step_(0.01)
    assert torch.allclose(r.bias, torch.tensor([-0.01, 0.01, 0.01, 0.01]))
    r.balance_step_(0.01)                                          # no counts: no move
    assert torch.allclose(r.bias, torch.tensor([-0.01, 0.01, 0.01, 0.01]))


def test_bias_chooses_and_never_enters_p():
    r = FanRouter(16, 4, 8)
    torch.manual_seed(0)
    sc = torch.randn(3, 5, 4)
    w0, p0 = r.select(sc)
    assert torch.equal(p0, torch.softmax(sc, -1))
    r.bias.copy_(torch.tensor([0.0, 0.0, 100.0, 0.0]))
    w1, p1 = r.select(sc)
    assert torch.equal(w1, torch.full_like(w1, 2)) and torch.equal(p1, p0)
    assert torch.equal(w0, sc.argmax(-1))


def test_counts_accrue_only_in_training_forwards_with_grad_and_after_step_applies_them():
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("reader")
    r = m.tul_fan_router
    m.eval()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    assert float(r.pending.sum()) == 0.0
    m.train()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    assert float(r.pending.sum()) == 0.0
    out, _a, rt = _route_call(m, inp, lab, layout)
    n_valid = int(layout.slot_valid.sum())
    assert float(r.pending.sum()) == n_valid
    counts = torch.bincount(rt["winner"][layout.slot_valid], minlength=M).float()
    assert torch.equal(r.pending, counts)
    load = counts / n_valid
    exp = 1e-3 * torch.sign(0.25 - load)
    m.tul_fan_after_step()
    assert torch.allclose(r.bias, exp)
    for i in range(M):
        assert float(out[f"fan_route_load_k{i}"]) == pytest.approx(float(load[i]))
    r.pending.fill_(3.0)
    m.tul_fan_reset_route_load()
    assert float(r.pending.sum()) == 0.0


def test_bias_is_checkpointed_and_pending_is_not():
    m = _arm("reader")
    m.tul_fan_router.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    m.tul_fan_router.pending.fill_(5.0)
    sd = m.state_dict()
    assert "tul_fan_router.bias" in sd and "tul_fan_router.pending" not in sd
    m2 = _arm("reader")
    m2.load_state_dict(sd)
    assert torch.equal(m2.tul_fan_router.bias, m.tul_fan_router.bias)


def test_selection_is_the_same_function_at_train_and_eval():
    _ids0, inp, lab, layout = _batch(M)
    for route in ("reader", "latent"):
        m = _arm(route)
        m.eval()
        _out, args, rt_eval = _route_call(m, inp, lab, layout)
        m.train()
        with torch.no_grad():
            rt_train = m._tul_fan_route(args[0], args[1], args[2], args[3], {})
        assert torch.equal(rt_train["winner"], rt_eval["winner"])
        if route == "reader":
            assert torch.allclose(rt_train["scale"], rt_eval["scale"])
        else:
            assert rt_train["scale"] is None and rt_eval["scale"] is None


def test_router_build_is_rng_neutral():
    a = _build(**_fan_kw(M, spandec=True, fan_all_wta_lambda=0.0))
    for route in ("reader", "latent"):
        b = _arm(route)
        sa, sb = a.state_dict(), b.state_dict()
        extra = set(sb) - set(sa)
        assert all(k.startswith(("tul_fan_router.", "tul_fan_teacher.")) for k in extra)
        for k, v in sa.items():
            assert torch.equal(v, sb[k]), k


# ── 4. the val oracle's router instruments ──────────────────────────────────────────


@pytest.mark.parametrize("route", ["reader", "latent"])
def test_oracle_router_instruments_match_a_hand_recomputation(route, monkeypatch):
    _ids0, inp, lab, layout = _batch(M)
    m = _arm(route).eval()
    tables: list = []
    orig = T.accumulate_span_ce

    def _rec(*a, **k):
        r = orig(*a, **k)
        tables.append(r.detach().clone())
        return r
    monkeypatch.setattr(T, "accumulate_span_ce", _rec)
    writes = _Spy(m, "_tul_fan_stream_write")
    with torch.no_grad():
        out, args, rt = _route_call(m, inp, lab, layout)
    # the table rows are the ROUTER's write with the winner forced to i
    assert len(writes.calls) == M
    for i, (a, k) in enumerate(writes.calls):
        assert a[1] == i
        if route == "reader":
            assert torch.equal(k["route_scale"], rt["scale"])
        else:
            assert k["route_scale"] is None
    assert len(tables) == M + 1                      # M per-cell passes + the mixed read
    ce = torch.stack([t[:, 1:] for t in tables[:M]], dim=-1)
    gid, keep_tok, _l, G = span_ce_index(lab, layout)
    n_tok = span_token_counts(gid, keep_tok, G)[:, 1:]
    ok = layout.slot_valid & (n_tok > 0)
    denom = n_tok[ok].sum()
    best, arg = ce.min(-1)
    w = rt["winner"]
    assert float(out["fan_router_coda_agree"]) == pytest.approx(
        float((w[ok] == arg[ok]).float().mean()), abs=1e-6)
    picked = ce.gather(-1, w.unsqueeze(-1)).squeeze(-1)
    assert float(out["fan_router_pick_regret"]) == pytest.approx(
        float((picked - best)[ok].sum() / denom), abs=1e-5)
    assert float(out["fan_rand_pick_regret"]) == pytest.approx(
        float((ce.mean(-1) - best)[ok].sum() / denom), abs=1e-5)
    if route == "latent":
        assert float(out["fan_teacher_coda_agree"]) == pytest.approx(
            float((rt["teacher"][ok] == arg[ok]).float().mean()), abs=1e-6)
    else:
        assert "fan_teacher_coda_agree" not in out
    # the regret of a pick is never below 0 and the random floor is at or above it
    assert float(out["fan_router_pick_regret"]) >= 0.0


def test_the_oracle_refuses_a_router_model_without_its_route():
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("reader").eval()
    orig = m._tul_fan_oracle

    def _drop_route(*a, **k):
        k.pop("route", None)
        return orig(*a, **k)
    m._tul_fan_oracle = _drop_route
    with torch.no_grad(), pytest.raises(RuntimeError, match="must hand in its route"):
        m(inp, labels=lab, slot_layout=layout)


@pytest.mark.parametrize("route", ["reader", "latent"])
def test_the_oracle_row_i_is_the_routers_write_with_the_winner_forced_to_i(route):
    """Built independently: cell i scaled by p_i (reader) or unscaled (latent) in ITS
    prefix cell, every other cell zero, through the ordinary 1:1 prefix write."""
    _ids0, _inp, _lab, layout = _batch(M)
    m = _arm(route)
    B, S = layout.slot_valid.shape
    torch.manual_seed(4)
    cells = torch.randn(B, S, M, 4, 64)
    p = torch.softmax(torch.randn(B, S, M), dim=-1)
    for i in range(M):
        got, pos = m._tul_fan_stream_write(cells, i, layout, 256,
                                           route_scale=p if route == "reader" else None)
        blank = torch.zeros_like(cells)
        blank[:, :, i] = cells[:, :, i] * (p[:, :, i, None, None] if route == "reader"
                                           else 1.0)
        exp, pos2 = m.tul.prefix_project(blank[:, :, i], layout, 256, cells=blank)
        assert torch.equal(pos, pos2)
        assert torch.allclose(got, exp, atol=1e-6), i
    with pytest.raises(RuntimeError, match="route_scale"):
        m._tul_fan_stream_write(cells, 0, layout, 256,
                                route_scale=None if route == "reader" else p)


def test_the_router_context_is_detached():
    """ctx (the slot's loop input) is the router's conditioning, not a gradient path into
    the prelude: under "reader" the gate p has gradient w.r.t. the cells and none w.r.t.
    the carrier the context is gathered from."""
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("reader").train()
    _out, args, _rt = _route_call(m, inp, lab, layout)
    cells = args[0].detach().requires_grad_(True)
    xn = args[1].detach().requires_grad_(True)
    rt = m._tul_fan_route(cells, xn, args[2], None, {})
    g_x, g_c = torch.autograd.grad((rt["scale"] ** 2).sum(), [xn, cells], allow_unused=True)
    assert g_x is None or float(g_x.abs().sum()) == 0.0
    assert g_c is not None and float(g_c.abs().sum()) > 0


@pytest.mark.parametrize("arm", [dict(fan_route="reader"), dict(fan_route="latent"),
                                 dict(fan_opf=True)])
def test_the_arm_build_never_reseeds_a_cuda_generator(arm, monkeypatch):
    """`torch.manual_seed` also reseeds every CUDA generator (queued lazily before CUDA
    init), and `fork_rng(devices=[])` restores only the CPU one. On the 5090 that moved
    every dropout mask of arms F / R / T off nowta's, which no CPU test could see: this
    pins that the build makes NO CUDA seed call at all, and leaves the CPU stream where
    nowta's build leaves it."""
    calls = []
    for fn in ("manual_seed", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, fn, lambda *a, _fn=fn, **k: calls.append(_fn))
    kw = dict(spandec=True, fan_all_wta_lambda=0.0)
    _build(**_fan_kw(M, **kw))
    after_nowta = torch.random.get_rng_state()
    calls.clear()
    _build(**_fan_kw(M, **kw, **arm))
    assert torch.equal(torch.random.get_rng_state(), after_nowta)
    # `_build` seeds the CPU stream itself (torch.manual_seed -> one CUDA call pair per
    # build); the arm's modules must add none.
    base = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: base.append(1))
    torch.manual_seed(0)
    assert calls == ["manual_seed_all"] * len(base), calls
