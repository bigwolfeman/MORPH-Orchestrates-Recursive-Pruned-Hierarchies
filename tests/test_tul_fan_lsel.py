"""``tul.fan_loop_select`` (the latent-selected loop, 2026-09-30): the fan's selection
moves INTO the slot loop and a latent objective trains the loop.

Files: morph/model/tul_fan_route.py (FanLatentHead, lsel_distance, lsel_exit_loss,
reset_to_winner, cell_spread), morph/model/tul.py (the keys, `_check_fan_loop_select`),
morph/model/transformer.py (`_lsel_begin / _lsel_pass / _lsel_finish`, the `_tul_core`
hooks, the detach of the detached variant, the routed write, the `fan_lsel_weighted`
fold, the `lsel_follow` eval switch), morph/training/train.py (both subtraction tuples,
the val routing, the teacher-pick val pass), morph/training/tul_setup.py, the configs.
Note: .agents/notes/proposed/architecture/2026-09-30-latent-selected-loop.md

What each test pins:
  * Every new key at its default is the tree: the ungraded fan's pins (measured before
    any of the fan route keys existed) with the keys implicit AND explicit, and the three
    existing route arms' pins measured on the UNMODIFIED tree at 213b585.
  * RNG-neutral build (every shared weight equal, no CUDA seed call) and an RNG-neutral
    forward (the generator state after a train forward equals the ungraded fan's).
  * The target of slot s depends on span s+1 only.
  * The reset: every pass, only for slots whose depth continues; the cells of a reset
    slot are the winner's pre-reset state; finished and pad slots are untouched; the next
    pass reads the reset carrier. The cells stay DISTINCT after a reset.
  * The latent loss is charged at the EXIT only (g runs with grad once per forward; its
    gradient equals the exit term's), and equals a hand recomputation.
  * The router's CE trains the router alone (no gradient into cells or loop).
  * Train follows the teacher on slots with a target; eval follows the router;
    `lsel_follow='teacher'` at eval follows the teacher. The coda reads the final winner
    alone (a loser perturbed inside the write moves no logit).
  * DETACHED: no token-CE gradient reaches the loop's parameters; JOINT: it does; the
    detached loop still trains through the latent loss.
  * Readings: switch rate and per-pass agreement by hand.
  * Checkpoint round trip of the EMA twin, the head and the router.
  * Refusals; configs compose one key from the ungraded fan and reach the model; train.py
    subtracts `fan_lsel_weighted` in both tuples, routes `lsel_*` and runs the teacher pass.

CPU, fp32, the tests/test_tul_fan.py / tests/test_tul_lxfan.py strict fixtures.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch.nn.functional as F

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_fan_route import (cell_spread, lsel_distance, lsel_exit_loss,
                                       reset_to_winner)
from test_tul_fan import _batch, _tiny
from test_tul_fan_opf import NOWTA_PINS, _swap, _tuple_after
from test_tul_lx_credit import M, _fan_pin_run
from test_tul_lxfan import _build, _fan_kw, _Spy

NEW_KEY_DEFAULTS = dict(
    fan_loop_select="off", fan_lsel_lambda=10.0, fan_lsel_eps=0.05,
    fan_lsel_enc_lambda=0.2, fan_lsel_enc_gamma=0.1, fan_lsel_router_lambda=1.0,
    fan_lsel_router_rank=64, fan_lsel_hidden=0, fan_lsel_train_follow="teacher", fan_lsel_read="winner")

# Measured 2026-09-30 on the UNMODIFIED tree at 213b585 (`git archive 213b585`, before
# any `fan_lsel_*` key existed) with these fixtures and one CPU thread, by
# /home/wolfe/morph-scratch/lsel-smoke/pin_arms.py: `_fan_pin_run` of the factor fan and
# the two router arms (their twin built where the arm has one), dropout 0.0.
ARM_PINS = {
    "opf": (10.776350021362305, 1358.1454057991505, 10.8211669921875, 2036.2593129592565, 200),
    "rmoe": (9.775012016296387, 1184.021321195265, 9.819000244140625, 1957.2451360127582, 199),
    "rlat": (12.512540817260742, 1173.9946138417436, 12.584196090698242, 2032.928871618136, 201),
}
_ARM_KW = {"opf": dict(fan_opf=True), "rmoe": dict(fan_route="reader"),
           "rlat": dict(fan_route="latent")}


def _nowta_kw(**kw) -> dict:
    base = dict(spandec=True, fan_all_wta_lambda=0.0)
    base.update(kw)
    return _fan_kw(M, **base)


def _lsel(mode: str = "joint", model_kw: dict | None = None, **kw) -> MORPHTransformer:
    m = _build(model_kw=model_kw, **_nowta_kw(fan_loop_select=mode, **kw))
    assert m.tul_fan_target_build()
    return m


def _cap(m: MORPHTransformer) -> list:
    m._lsel_capture = []
    return m._lsel_capture


def _per_cell(d_slot: torch.Tensor) -> torch.Tensor:
    return d_slot.repeat_interleave(M, dim=1)


# ── 1. the defaults are the tree ────────────────────────────────────────────────────


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_ungraded_fan_pins_hold_with_the_keys_implicit_and_explicit(dropout):
    a = _build(model_kw={"dropout": dropout}, **_nowta_kw())
    b = _build(model_kw={"dropout": dropout}, **_nowta_kw(**NEW_KEY_DEFAULTS))
    for m in (a, b):
        assert m.tul_fan_lsel_head is None and m.tul_fan_lsel_router is None
        assert m._lsel_mode == "off"
        assert not m.tul_fan_target_build()
    assert a.state_dict().keys() == b.state_dict().keys()
    assert not any(k.startswith("tul_fan_lsel") for k in a.state_dict())
    assert _fan_pin_run(a) == NOWTA_PINS[dropout]
    assert _fan_pin_run(b) == NOWTA_PINS[dropout]


@pytest.mark.parametrize("arm", sorted(ARM_PINS))
def test_the_existing_route_arms_are_unchanged(arm):
    m = _build(**_nowta_kw(**_ARM_KW[arm], **NEW_KEY_DEFAULTS))
    m.tul_fan_target_build()
    assert ARM_PINS[arm] is not None, "pin not measured"
    assert _fan_pin_run(m) == ARM_PINS[arm]


# ── 2. RNG neutrality ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["joint", "detached"])
def test_build_is_rng_neutral_and_never_reseeds_cuda(mode, monkeypatch):
    calls = []
    for fn in ("manual_seed", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, fn, lambda *a, _fn=fn, **k: calls.append(_fn))
    a = _build(**_nowta_kw())
    after_a = torch.random.get_rng_state()
    calls.clear()
    b = _lsel(mode)
    assert torch.equal(torch.random.get_rng_state(), after_a)
    assert calls == ["manual_seed_all"], calls          # `_build`'s own torch.manual_seed
    sa, sb = a.state_dict(), b.state_dict()
    extra = set(sb) - set(sa)
    assert extra and all(k.startswith(("tul_fan_lsel_head.", "tul_fan_lsel_router."))
                         for k in extra), sorted(extra)
    for k, v in sa.items():
        assert torch.equal(v, sb[k]), k


@pytest.mark.parametrize("mode", ["joint", "detached"])
def test_the_forward_draws_no_extra_rng(mode):
    """The per-pass selection, the twin's front and the exit term draw nothing: after a
    seeded training forward the CPU generator is where the ungraded fan's leaves it, so
    every dropout mask and depth draw of a later step is the same."""
    _ids0, inp, lab, layout = _batch(M)
    a = _build(model_kw={"dropout": 0.1}, **_nowta_kw()).train()
    b = _lsel(mode, model_kw={"dropout": 0.1}).train()
    torch.manual_seed(21)
    a(inp, labels=lab, slot_layout=layout)
    ra = torch.random.get_rng_state()
    torch.manual_seed(21)
    b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(torch.random.get_rng_state(), ra)


# ── 3. the target ───────────────────────────────────────────────────────────────────


def test_target_of_slot_s_depends_on_span_s_plus_1_only():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel().eval()

    def _z(ids):
        fkw, freset, _, _ = m._tul_tg_kwargs(layout)
        x, _, _ = m._tul_front(ids, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        return m._tul_fan_target(ids, lab, layout, fkw, freset, x)

    b, s = 0, 3
    with torch.no_grad():
        z0 = _z(inp)
    assert bool(z0["ok"][b, s]) and z0["zo"] is not None
    other = [p for p in range(inp.shape[1])
             if not bool(layout.slot_mask[b, p]) and int(layout.bag_id[b, p]) != s + 1]
    assert len(other) > 20
    inp2 = inp.clone()
    for p in other:
        inp2[b, p] = _swap(int(inp[b, p]))
    with torch.no_grad():
        z1 = _z(inp2)
    assert torch.equal(z1["z"][b, s], z0["z"][b, s])
    assert not torch.equal(z1["z"][b, s + 2], z0["z"][b, s + 2])
    own = [int(p) for p in torch.nonzero((layout.bag_id[b] == s + 1)
                                         & ~layout.slot_mask[b]).flatten()]
    inp3 = inp.clone()
    inp3[b, own[1]] = _swap(int(inp[b, own[1]]))
    with torch.no_grad():
        z2 = _z(inp3)
    assert not torch.equal(z2["z"][b, s], z0["z"][b, s])


# ── 4. the reset ────────────────────────────────────────────────────────────────────


def test_reset_to_winner_by_hand():
    torch.manual_seed(0)
    B, S, n, C = 2, 3, 4, 5
    h = torch.randn(B, S * M, n, C)
    w = torch.randint(0, M, (B, S))
    reset = torch.tensor([[True, False, True], [False, True, True]])
    out = reset_to_winner(h, w, reset, M)
    hc, oc = h.view(B, S, M, n, C), out.view(B, S, M, n, C)
    for b in range(B):
        for s in range(S):
            for i in range(M):
                exp = hc[b, s, int(w[b, s])] if bool(reset[b, s]) else hc[b, s, i]
                assert torch.equal(oc[b, s, i], exp)
    # BPTT through the winner: a reset slot's gradient lands on the winner's cell only
    hg = h.clone().requires_grad_(True)
    reset_to_winner(hg, w, reset, M).sum().backward()
    g = hg.grad.view(B, S, M, n, C)
    for b in range(B):
        for s in range(S):
            for i in range(M):
                if bool(reset[b, s]):
                    exp = float(M) if i == int(w[b, s]) else 0.0
                else:
                    exp = 1.0
                assert torch.all(g[b, s, i] == exp)


def test_the_loop_resets_every_pass_only_the_continuing_slots():
    """Forced per-slot depths 1..3 at eval: after pass t a slot of depth > t+1 has all
    M cells equal to its winner's PRE-reset state; a slot that finished at t keeps its M
    distinct candidates; a slot that stopped earlier and every pad are untouched; and the
    next pass starts from the reset carrier (a finished slot's cells read the same at the
    next capture)."""
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel().eval()
    B, S = layout.slot_valid.shape
    d = (torch.arange(S).view(1, S) % 3 + 1).expand(B, S).contiguous()
    cap = _cap(m)
    with torch.no_grad():
        m.tul_forward_ablated(inp, lab, layout, slot_depths=_per_cell(d))
    valid = layout.slot_valid
    dpad = torch.where(valid, d, torch.ones_like(d))
    assert [c["t"] for c in cap] == [0, 1, 2]
    for c in cap:
        t = c["t"]
        pre = c["pre"].view(B, S, M, *c["pre"].shape[2:])
        post = c["post"].view(B, S, M, *c["post"].shape[2:])
        reset = valid & (dpad > t + 1)
        assert bool(reset.any()) == (t < 2)
        assert torch.equal(c["act"], valid & (dpad > t))
        for b in range(B):
            for s in range(S):
                if bool(reset[b, s]):
                    w = int(c["follow"][b, s])
                    for i in range(M):
                        assert torch.equal(post[b, s, i], pre[b, s, w])
                else:
                    assert torch.equal(post[b, s], pre[b, s])
    # A slot that stops at pass t is frozen from then on: its cells at the next capture
    # are the ones it left with (and they are M distinct candidates, not copies).
    nxt = cap[1]["pre"].view(B, S, M, *cap[1]["pre"].shape[2:])
    post0 = cap[0]["post"].view(B, S, M, *cap[0]["post"].shape[2:])
    stopped = valid & (dpad == 1)
    assert bool(stopped.any())
    assert torch.equal(nxt[stopped], post0[stopped])
    # The next pass READ the reset carrier: a continuing slot's cells were identical when
    # pass 1 began and pass 1 moved them.
    cont = valid & (dpad >= 2)
    assert not torch.equal(nxt[cont], post0[cont])


def test_the_cells_stay_distinct_after_a_reset():
    """After one reset all M cells of a slot are the SAME state; pass t+1 must pull them
    apart again or the fan is dead. They differ through each cell's own loop seed
    `e_i` (the prelude output at its own prefix position, re-injected every pass by
    DiagonalInjection), its register term and its own core position. Read at the fresh
    init (register W_o = 0, P_cell = 0: the weakest case)."""
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel().eval()
    B, S = layout.slot_valid.shape
    d = torch.full((B, S), 3, dtype=torch.long)
    cap = _cap(m)
    with torch.no_grad():
        m.tul_forward_ablated(inp, lab, layout, slot_depths=_per_cell(d))
    valid = layout.slot_valid
    post0 = cap[0]["post"].view(B, S, M, *cap[0]["post"].shape[2:]).mean(dim=-2)
    pre1 = cap[1]["pre"].view(B, S, M, *cap[1]["pre"].shape[2:]).mean(dim=-2)
    # identical going in ...
    assert float(cell_spread(post0, valid)) == 0.0
    # ... distinct coming out: every pair of cells of every valid slot differs
    spread = float(cell_spread(pre1, valid))
    print(f"[lsel] cell spread after one reset at init: {spread:.4g}")
    assert spread > 1e-2
    for b in range(B):
        for s in range(S):
            if bool(valid[b, s]):
                for i in range(M):
                    for j in range(i + 1, M):
                        assert not torch.allclose(pre1[b, s, i], pre1[b, s, j], atol=1e-4)


# ── 5. the exit-only latent loss and the router ─────────────────────────────────────


def _finish_spy(m):
    seen = {}
    orig = m._lsel_finish

    def _f(st, h):
        seen["st"], seen["h"] = st, h
        return orig(st, h)
    m._lsel_finish = _f
    return seen


def test_the_latent_loss_is_charged_at_the_exit_only_and_equals_a_hand_computation():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel(fan_lsel_lambda=3.0, fan_lsel_enc_lambda=0.0, fan_lsel_router_lambda=0.0,
              fan_lsel_eps=0.1).train()
    head = m.tul_fan_lsel_head
    grad_calls = []
    h0 = head.register_forward_hook(
        lambda mod, a, o: grad_calls.append(torch.is_grad_enabled()))
    seen = _finish_spy(m)
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    h0.remove()
    assert grad_calls.count(True) == 1, grad_calls       # ONE graded g call: the exit
    st, h = seen["st"], seen["h"]
    B, S = layout.slot_valid.shape
    cr = h.view(B, S, M, *h.shape[2:]).mean(dim=-2)
    dist = lsel_distance(head(cr), st["z"])
    tex = dist.detach().argmin(-1)
    ok = st["ok"]
    # relaxed WTA by hand
    hand = 0.0
    for b in range(B):
        for s in range(S):
            if bool(ok[b, s]):
                w = int(tex[b, s])
                hand += sum((0.9 if i == w else 0.1 / (M - 1)) * float(dist[b, s, i])
                            for i in range(M))
    hand /= float(ok.sum())
    assert float(out["fan_lsel_lat"]) == pytest.approx(hand, rel=1e-5)
    assert float(out["fan_lsel_weighted"]) == pytest.approx(3.0 * hand, rel=1e-5)
    # the head's gradient from the whole objective IS the exit term's
    params = list(head.parameters())
    g_all = torch.autograd.grad(out["loss"], params, retain_graph=True)
    g_exit = torch.autograd.grad(3.0 * lsel_exit_loss(dist, tex, ok, 0.1), params)
    for a, b_ in zip(g_all, g_exit):
        assert float(b_.abs().sum()) > 0
        assert torch.allclose(a, b_, atol=1e-6, rtol=1e-5)
    # train: the final winner IS the exit's teacher on every slot with a target
    assert torch.equal(st["final"][ok], tex[ok])


def _loop_params(m) -> list:
    ps = list(m.core.parameters()) + list(m.injection.parameters())
    ps += list(m.tul_register.parameters())
    np_, nc = m.cfg.n_prelude, m.cfg.n_core
    for x in list(m.x0_injects)[np_:np_ + nc]:
        ps += list(x.parameters())
    return [p for p in ps if p.requires_grad]


def test_the_router_ce_trains_the_router_alone():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel(fan_lsel_lambda=0.0, fan_lsel_enc_lambda=0.0, fan_lsel_router_lambda=1.0
              ).train()
    seen = {}
    orig = m._lsel_finish

    def _f(st, h):
        orig(st, h)
        seen["loss"] = m._lsel_out["loss"]
    m._lsel_finish = _f
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    loss = seen["loss"]
    assert float(out["fan_lsel_router_ce"]) > 0
    g_r = torch.autograd.grad(loss, list(m.tul_fan_lsel_router.parameters()),
                              retain_graph=True, allow_unused=True)
    assert all(g is not None for g in g_r) and sum(float(g.abs().sum()) for g in g_r) > 0
    g_l = torch.autograd.grad(loss, _loop_params(m) + list(m.tul_fan_lsel_head.parameters()),
                              retain_graph=True, allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in g_l)


# ── 6. who picks ────────────────────────────────────────────────────────────────────


def test_train_follows_the_teacher_eval_follows_the_router():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel()
    seen = _finish_spy(m)
    cap = _cap(m)
    m.train()
    torch.manual_seed(5)
    m(inp, labels=lab, slot_layout=layout)
    ok = seen["st"]["ok"]
    assert cap
    n_diff = 0
    for c in cap:
        exp = torch.where(ok, c["teacher"], c["router"])
        assert torch.equal(c["follow"], exp)
        n_diff += int((c["teacher"] != c["router"])[ok & c["act"]].sum())
    assert n_diff > 0          # the two picks differ somewhere, so the test can tell
    m.eval()
    cap.clear()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    for c in cap:
        assert c["teacher"] is not None
        assert torch.equal(c["follow"], c["router"])
    cap.clear()
    with torch.no_grad():
        m(inp, labels=None, slot_layout=layout)
    for c in cap:
        assert c["teacher"] is None and torch.equal(c["follow"], c["router"])
    cap.clear()
    with torch.no_grad():
        m.tul_forward_ablated(inp, lab, layout, lsel_follow="teacher")
    for c in cap:
        assert torch.equal(c["follow"], torch.where(ok, c["teacher"], c["router"]))
    with pytest.raises(ValueError, match="eval-only"):
        m.train()
        m.tul_forward_ablated(inp, lab, layout, lsel_follow="teacher")


def test_router_followed_train_follows_the_router_and_still_trains_it():
    """`fan_lsel_train_follow: router` (2026-10-01): at train the loop follows the ROUTER
    on every slot, the teacher still exists (it labels the router and names the exit
    loss's winner), and the router CE and the exit loss are both charged. The same
    weights under the default follow the teacher, so the key is what moves the follow."""
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel(fan_lsel_train_follow="router").train()
    seen = _finish_spy(m)
    cap = _cap(m)
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    ok = seen["st"]["ok"]
    assert cap and seen["st"]["teacher_drives"] is False
    n_diff = 0
    for c in cap:
        assert c["teacher"] is not None
        assert torch.equal(c["follow"], c["router"])
        n_diff += int((c["teacher"] != c["router"])[ok & c["act"]].sum())
    assert n_diff > 0          # teacher and router differ, so following one is visible
    assert float(seen["st"]["rce_n"]) > 0
    assert float(out["fan_lsel_weighted"]) > 0
    out["loss"].backward()
    assert float(sum(p.grad.abs().sum() for p in m.tul_fan_lsel_router.parameters()
                     if p.grad is not None)) > 0
    # same weights, default key: the train forward follows the teacher on `ok` slots
    m2 = _lsel()
    m2.load_state_dict(m.state_dict())
    m2.tul_fan_target_build()
    m2.train()
    cap2 = _cap(m2)
    torch.manual_seed(5)
    m2(inp, labels=lab, slot_layout=layout)
    assert any(not torch.equal(a["follow"], b["follow"]) for a, b in zip(cap, cap2))


def test_train_follow_key_refusals():
    with pytest.raises(ValueError, match="fan_lsel_train_follow"):
        _lsel(fan_lsel_train_follow="oracle")
    with pytest.raises(ValueError, match="silent no-op"):
        _build(**_nowta_kw(fan_lsel_train_follow="router"))


@pytest.mark.parametrize("name,parent,wb", [
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf",
     "tul_slot_spandec_strict_fan4_all_fp01_lsel_joint",
     "slot-spandec-strict-fan4-all-fp01-lsel-joint-rf"),
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf",
     "tul_slot_spandec_strict_fan4_all_fp01_lsel_det",
     "slot-spandec-strict-fan4-all-fp01-lsel-det-rf"),
])
def test_router_followed_configs_change_one_key_and_reach_the_model(
        name, parent, wb, monkeypatch):
    import dataclasses

    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {"tul.fan_lsel_train_follow", "wandb.name"}, sorted(diff)
    assert c["wandb.name"] == wb
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    pmc = build_morph_config(pcfg, tul=prt.model_cfg)
    tdiff = {f.name for f in dataclasses.fields(mc.tul)
             if getattr(mc.tul, f.name) != getattr(pmc.tul, f.name)}
    assert tdiff == {"fan_lsel_train_follow"}, sorted(tdiff)
    assert mc.tul.fan_lsel_train_follow == "router"
    assert rt.manifest.get("fan_lsel_train_follow") == "router"


def test_read_all_writes_every_final_candidate_and_the_oracle_runs_unrouted():
    """`fan_lsel_read: all`: the write is the ungraded fan's (all M cells 1:1, no routed
    zeros), the selection still resets the loop, the exit loss is still charged, and the
    val oracle runs as the write-all fan's (no route)."""
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel("detached", fan_lsel_train_follow="router", fan_lsel_read="all")
    seen = {}
    orig = m.tul.prefix_project

    def _pp(h_slots, layout_, l_total, cells=None):
        seen["cells"] = None if cells is None else cells.detach().clone()
        return orig(h_slots, layout_, l_total, cells=cells)
    m.tul.prefix_project = _pp
    cap = _cap(m)
    m.train()
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    c = seen["cells"]
    valid = layout.slot_valid
    assert c is not None and c.shape[2] == M
    # every cell of every valid slot is live (no routed zeros) ...
    nz = c.flatten(3).abs().sum(dim=-1) > 0                       # [B, S, M]
    assert bool(nz[valid].all())
    # ... and they are the loop's final candidates, which differ from each other
    assert any(not torch.allclose(c[b, s, 0], c[b, s, 1]) for b, s in valid.nonzero().tolist())
    assert cap and float(out["fan_lsel_weighted"]) > 0
    # eval: the val oracle runs unrouted and reports
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout)
    assert "fan_oracle_ce" in o and torch.isfinite(o["fan_oracle_ce"])
    # the winner read zeroes the losers (the contrast that makes the test meaningful)
    w = _lsel("detached", fan_lsel_train_follow="router")
    w.load_state_dict(m.state_dict())
    w.tul_fan_target_build()
    seen.clear()
    orig_w = w.tul.prefix_project

    def _ppw(h_slots, layout_, l_total, cells=None):
        seen["cells"] = cells.detach().clone()
        return orig_w(h_slots, layout_, l_total, cells=cells)
    w.tul.prefix_project = _ppw
    w.train()
    torch.manual_seed(5)
    w(inp, labels=lab, slot_layout=layout)
    nzw = seen["cells"].flatten(3).abs().sum(dim=-1) > 0
    assert bool((nzw[valid].sum(dim=-1) == 1).all())


def test_read_key_refusals():
    with pytest.raises(ValueError, match="fan_lsel_read"):
        _lsel(fan_lsel_read="mean")
    with pytest.raises(ValueError, match="silent no-op"):
        _build(**_nowta_kw(fan_lsel_read="all"))


@pytest.mark.parametrize("name,key,val,wb", [
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1", "fan_lsel_lambda", 1.0,
     "slot-spandec-strict-fan4-all-fp01-lsel-det-rf-lam1"),
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_all", "fan_lsel_read", "all",
     "slot-spandec-strict-fan4-all-fp01-lsel-det-rf-all"),
])
def test_det_rf_children_change_one_key_and_reach_the_model(name, key, val, wb, monkeypatch):
    import dataclasses

    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {f"tul.{key}", "wandb.name"}, sorted(diff)
    assert c["wandb.name"] == wb
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    pmc = build_morph_config(pcfg, tul=prt.model_cfg)
    tdiff = {f.name for f in dataclasses.fields(mc.tul)
             if getattr(mc.tul, f.name) != getattr(pmc.tul, f.name)}
    assert tdiff == {key}, sorted(tdiff)
    assert getattr(mc.tul, key) == val and rt.manifest.get(key) == val


def test_the_coda_reads_the_final_winner_alone():
    _ids0, inp, _lab, layout = _batch(M)
    m = _lsel().eval()
    orig = m._fan_route_cells
    which = {"target": None}

    def _perturbed(cells, rt):
        w = rt["winner"]
        idx = w if which["target"] == "winner" else (w + 1) % M
        sel = F.one_hot(idx, M).to(cells.dtype).view(*idx.shape, M,
                                                     *([1] * (cells.dim() - 3)))
        return orig(cells + 3.0 * sel * torch.randn_like(cells), rt)

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


def test_the_val_oracle_prices_the_final_winner():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel().eval()
    spy = _Spy(m, "_tul_fan_oracle")
    seen = _finish_spy(m)
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert len(spy.calls) == 1
    rt = spy.calls[0][1]["route"]
    assert rt["scale"] is None and torch.equal(rt["winner"], seen["st"]["final"])
    for k in ("fan_router_coda_agree", "fan_router_pick_regret", "fan_teacher_coda_agree",
              "fan_oracle_ce", "fan_mixed_ce"):
        assert k in out, k


# ── 7. joint vs detached: who trains the loop ───────────────────────────────────────

_TOKEN_ONLY = dict(core_fixed_point_lambda=0.0, slot_gain_lambda=0.0,
                   slot_gain_tail_lambda=0.0)


def _token_only(mode: str) -> MORPHTransformer:
    """Every loop term at weight 0 (fixed point, gain hinges, diversity, latent, floor,
    router), so the objective is the token CE + the span decoder's CE alone."""
    return _lsel(mode, model_kw=_TOKEN_ONLY, fan_repel_lambda=0.0, fan_lsel_lambda=0.0,
                 fan_lsel_enc_lambda=0.0, fan_lsel_router_lambda=0.0).train()


def test_detached_no_token_ce_gradient_reaches_the_loop():
    _ids0, inp, lab, layout = _batch(M)
    m = _token_only("detached")
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    ps = _loop_params(m)
    assert len(ps) > 10
    for p in ps:
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0
    # ... while the token CE still trains the write, the coda and the span decoder
    assert float(m.tul.W_prefix.grad.abs().sum()) > 0
    assert float(sum(p.grad.abs().sum() for p in m.coda.parameters()
                     if p.grad is not None)) > 0
    assert float(sum(p.grad.abs().sum() for p in m.tul_spandec.parameters()
                     if p.grad is not None)) > 0


def test_joint_the_token_ce_reaches_the_loop():
    _ids0, inp, lab, layout = _batch(M)
    m = _token_only("joint")
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    tot = sum(float(p.grad.abs().sum()) for p in m.core.parameters() if p.grad is not None)
    assert tot > 0


def test_detached_the_loop_still_trains_on_the_latent_loss():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel("detached", model_kw=_TOKEN_ONLY, fan_repel_lambda=0.0).train()
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    tot = sum(float(p.grad.abs().sum()) for p in m.core.parameters() if p.grad is not None)
    assert tot > 0
    assert float(out["fan_lsel_weighted"]) > 0


# ── 8. readings ─────────────────────────────────────────────────────────────────────


def test_switch_rate_and_agreement_by_hand():
    _ids0, inp, lab, layout = _batch(M)
    m = _lsel()
    seen = _finish_spy(m)
    cap = _cap(m)
    m.train()
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    ok = seen["st"]["ok"]
    sw = n = 0
    for prev, c in zip(cap, cap[1:]):
        sw += int(((c["follow"] != prev["follow"]) & c["act"]).sum())
        n += int(c["act"].sum())
    assert float(out["fan_lsel_switch_rate"]) == pytest.approx(sw / max(n, 1), abs=1e-6)
    ag = tot = 0
    for c in cap:
        okp = ok & c["act"]
        a = int(((c["router"] == c["teacher"]) & okp).sum())
        assert float(out[f"fan_lsel_teacher_router_agree_t{c['t']}"]) == pytest.approx(
            a / max(int(okp.sum()), 1), abs=1e-6)
        ag += a
        tot += int(okp.sum())
    assert float(out["fan_lsel_teacher_router_agree"]) == pytest.approx(ag / tot, abs=1e-6)
    share = sum(float(out[f"fan_lsel_share_k{i}"]) for i in range(M))
    assert share == pytest.approx(1.0, abs=1e-6)
    assert -1.0 < float(out["fan_lsel_r2"]) < 1.0


# ── 9. checkpoint ───────────────────────────────────────────────────────────────────


def test_twin_head_and_router_round_trip_through_the_checkpoint(tmp_path):
    from morph.training.train import read_checkpoint_side_state, save_checkpoint
    m = _lsel(fan_target_ema=0.5)
    with torch.no_grad():
        for p in m.prelude.parameters():
            p.add_(0.1 * torch.randn_like(p))
        for p in list(m.tul_fan_lsel_head.parameters()) + list(
                m.tul_fan_lsel_router.parameters()):
            p.add_(0.1 * torch.randn_like(p))
    m.tul_fan_after_step()
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, 3, m, torch.optim.SGD(m.parameters(), lr=0.1),
                    torch.amp.GradScaler("cpu", enabled=False), None)
    state = read_checkpoint_side_state(path, "fan_target")
    assert state is not None
    m2 = _lsel(fan_target_ema=0.5)
    m2.load_state_dict(torch.load(path, weights_only=False)["model"])
    assert m2.tul_fan_target_sync(state) == "resume"
    a, b = m.__dict__["_fan_target"].state_dict(), m2.__dict__["_fan_target"].state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    for name in ("tul_fan_lsel_head", "tul_fan_lsel_router"):
        sa, sb = getattr(m, name).state_dict(), getattr(m2, name).state_dict()
        assert sa.keys() == sb.keys() and len(sa) > 0
        for k in sa:
            assert torch.equal(sa[k], sb[k]), (name, k)


# ── 10. refusals ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,match", [
    (dict(fan_loop_select="bogus"), ValueError, "fan_loop_select must be"),
    (dict(fan_lsel_lambda=1.0), ValueError, "fan_loop_select='off'"),
    (dict(fan_lsel_hidden=8), ValueError, "fan_loop_select='off'"),
    (dict(fan_loop_select="joint", fan_lsel_eps=1.0), ValueError, "fan_lsel_eps"),
    (dict(fan_loop_select="joint", fan_lsel_lambda=-1.0), ValueError, "fan_lsel_lambda"),
    (dict(fan_loop_select="joint", fan_lsel_router_rank=0), ValueError, "router_rank"),
    (dict(fan_loop_select="joint", fan_opf=True), NotImplementedError, "fan_opf"),
    (dict(fan_loop_select="joint", fan_route="reader"), NotImplementedError, "fan_route"),
    (dict(fan_loop_select="joint", fan_route="latent"), NotImplementedError, "fan_route"),
    (dict(fan_loop_select="joint", fan_all_wta_lambda=1.0), NotImplementedError,
     "fan_all_wta_lambda"),
    (dict(fan_loop_select="joint", code_enum_k=4), NotImplementedError, "code_enum_k=4"),
    (dict(fan_loop_select="joint", bcast=True), NotImplementedError, "bcast"),
    (dict(fan_loop_select="joint", prefix_source="trajectory"), NotImplementedError,
     "prefix_source"),
    (dict(fan_loop_select="joint", loop_carry="persist"), NotImplementedError,
     "loop_carry"),
    (dict(fan_loop_select="detached", mux_beta=1.0), NotImplementedError, "mux_beta"),
    (dict(fan_loop_select="detached", oracle_z=True), NotImplementedError, "oracle_z"),
    (dict(fan_loop_select="detached", spandec_per_pass=True), NotImplementedError,
     "spandec_per_pass"),
])
def test_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        TULConfig(**_nowta_kw(**kw))


def test_needs_the_write_all_fan():
    with pytest.raises(NotImplementedError, match="fan_k=0"):
        TULConfig(fan_loop_select="joint")
    with pytest.raises(NotImplementedError, match="fan_mix='mean'"):
        TULConfig(**_fan_kw(M, fan_mix="mean", fan_all_wta_lambda=1.0, fan_select_eps=0.05,
                            fan_loop_select="joint"))
    TULConfig(**_nowta_kw(fan_loop_select="joint"))
    TULConfig(**_nowta_kw(fan_loop_select="detached"))


def test_off_model_refuses_the_eval_switch():
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_nowta_kw()).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="fan_loop_select"):
        m.tul_forward_ablated(inp, lab, layout, lsel_follow="teacher")


# ── 11. configs and train.py ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,mode,wb", [
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_joint", "joint",
     "slot-spandec-strict-fan4-all-fp01-lsel-joint"),
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_det", "detached",
     "slot-spandec-strict-fan4-all-fp01-lsel-det"),
])
def test_configs_compose_one_key_from_the_ungraded_fan_reach_the_model_and_train(
        name, mode, wb, monkeypatch):
    import dataclasses

    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_fan4_all_fp01_nowta"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {"tul.fan_loop_select", "wandb.name"}, sorted(diff)
    assert c["wandb.name"] == wb and c["training.steps"] == p["training.steps"] == 5000
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    pmc = build_morph_config(pcfg, tul=prt.model_cfg)
    assert mc.tul.fan_loop_select == mode == rt.model_cfg.fan_loop_select
    tdiff = {f.name for f in dataclasses.fields(mc.tul)
             if getattr(mc.tul, f.name) != getattr(pmc.tul, f.name)}
    print(f"[lsel] {name}: TULConfig diff vs the ungraded fan = "
          f"{ {k: (getattr(pmc.tul, k), getattr(mc.tul, k)) for k in sorted(tdiff)} }")
    assert tdiff == {"fan_loop_select"}, sorted(tdiff)
    for key, v in NEW_KEY_DEFAULTS.items():
        if key != "fan_loop_select":
            assert getattr(mc.tul, key) == v, key
    assert rt.manifest.get("fan_loop_select") == mode
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=96,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)
                         ).train().float()
    assert m.tul_fan_target_build()
    _ids0, inp, lab, layout = _batch(M)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and "fan_lsel_weighted" in out
    m.tul_fan_after_step()


def test_both_train_py_subtraction_tuples_list_the_key():
    src = pathlib.Path("morph/training/train.py").read_text()
    assert "fan_lsel_weighted" in _tuple_after(src, "_aux2")
    assert "fan_lsel_weighted" in _tuple_after(src, "_ak")


class _Cfg:
    class tul:
        fan_loop_select = "joint"


class _LselStub(torch.nn.Module):
    cfg = _Cfg()

    def __init__(self):
        super().__init__()
        self.follow_calls = []

    def tul_forward_with_plan_nats(self, x, y, layout):
        return {"loss": torch.tensor(7.0), "fan_lsel_weighted": torch.tensor(2.0),
                "fan_lsel_r2": torch.tensor(0.3), "fan_lsel_switch_rate": torch.tensor(0.4),
                "fan_router_coda_agree": torch.tensor(0.5),
                "ce_tokens": 4.5, "layer_passes": 8.0, "n_tokens": 4.0}

    def tul_forward_ablated(self, x, y, layout, lsel_follow=None):
        self.follow_calls.append(lsel_follow)
        return {"ce_tokens": torch.tensor(4.25)}


class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


def test_evaluate_subtracts_the_term_routes_the_readings_and_runs_the_teacher_pass():
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    extra: dict = {}
    stub = _LselStub()
    avg, _ppl = evaluate(stub, torch.device("cpu"), iter([(x, x, _Layout())] * 2),
                         n_batches=2, tul=True, extra=extra)
    assert avg == pytest.approx(5.0)
    assert stub.follow_calls == ["teacher", "teacher"]
    assert extra["fan/lsel_teacher_pick_ce"] == pytest.approx(4.25)
    assert extra["fan/lsel_teacher_pick_gap"] == pytest.approx(0.25)
    for k, v in (("lsel_r2", 0.3), ("lsel_switch_rate", 0.4), ("lsel_weighted", 2.0)):
        assert extra[f"val/fan_{k}"] == pytest.approx(v)
        assert f"fan/{k}" not in extra
    assert extra["fan/router_coda_agree"] == pytest.approx(0.5)


def test_the_source_has_no_per_pass_latent_term():
    """SOURCE check: inside `_lsel_pass` the head runs only under no_grad (the teacher's
    pick); the only graded head call is in `_lsel_finish`."""
    src = pathlib.Path("morph/model/transformer.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    body = ast.get_source_segment(src, fns["_lsel_pass"])
    i_ng = body.index("with torch.no_grad():")
    i_head = body.index("self.tul_fan_lsel_head(")
    assert i_ng < i_head and body.count("self.tul_fan_lsel_head(") == 1
