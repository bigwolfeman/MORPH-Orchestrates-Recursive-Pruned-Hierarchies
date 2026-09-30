"""``tul.fan_opf`` (arm F, 2026-09-30): Orthogonal Predictive Factorization on the
write-all fan's cells, one factor away from nowta. Also the pieces arm T shares with it
(the EMA-prelude target and its twin's lifecycle) and the train.py / config contracts
of all three new arms.

Files: morph/model/tul_fan_route.py (FanTargetFront, FanOPF, opf_terms, the pooling),
morph/model/tul.py (the keys, `_check_fan_opf_route`), morph/model/transformer.py
(`_tul_front(twin=)`, `_tul_fan_target`, `_tul_fan_opf`, the `opf_weighted` fold, the twin
lifecycle `tul_fan_target_build / _sync / _state`, `tul_fan_after_step`),
morph/training/train.py (both subtraction tuples, the val routing, the checkpoint key,
the post-step hook), morph/training/tul_setup.py, the three configs.
Note: .agents/notes/proposed/architecture/2026-09-30-fan-opf-and-routers.md

What each test pins:
  * Every new key at its default is the tree: nowta's loss / eval logits / eval loss /
    grad sum measured on the UNMODIFIED worktree at 622fb13 (one CPU thread; value pins
    only hold on this machine), with the keys implicit AND explicit; no new state keys.
  * Arm F is nowta plus a term: RNG-neutral build (every shared weight equal), the same
    token CE at the same seed under dropout 0.1, loss = nowta loss + opf_weighted.
  * The target of slot s is a function of span s+1's tokens only; it is computed by the
    TWIN (not the live prelude), draws no RNG, and equals the live pooling at build time.
  * L_pred's gradient reaches the cells and the loop; L_fac's reaches P and not the
    cells; L_enc's reaches the online prelude; nothing reaches the twin.
  * opf_terms equals an independent masked-index computation; R^2 hand check.
  * The QR retraction makes P orthonormal (1e-5) with QR's sign canonicalised, and is
    the identity on an orthonormal P. The EMA update is m*twin + (1-m)*live.
  * The twin round-trips through save_checkpoint / read_checkpoint_side_state / sync.
  * Every refusal fires. The configs compose one key away from nowta and reach the model.
  * evaluate() subtracts opf_weighted / rlat_weighted and routes the readings; both
    train.py tuples list both keys.

CPU, fp32, the tests/test_tul_fan.py / tests/test_tul_lxfan.py strict fixtures.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from morph.model.transformer import MORPHTransformer, span_ce_index, span_token_counts
from morph.model.tul import TULConfig
from morph.model.tul_fan_route import (FanOPF, batch_std, opf_terms, participation_ratio,
                                       pooled_span_states)
from test_tul_fan import DOT, _batch, _tiny
from test_tul_lx_credit import M, _fan_pin_run
from test_tul_lxfan import _build, _fan_kw, _Spy

# Measured 2026-09-30 on the UNMODIFIED worktree at 622fb13 (before any of these keys
# existed) with these fixtures and one CPU thread, by
# /home/wolfe/morph-scratch/opf-smoke/pin_head.py: (train loss, eval logit sum, eval loss,
# sum |grad|, n grads) of nowta's tiny twin.
NOWTA_PINS = {
    0.0: (9.765828132629395, 1358.1454057991505, 9.81228256225586, 1962.4300315045568, 195),
    0.1: (9.74213981628418, 1358.1454057991505, 9.81228256225586, 1970.5855368406962, 195),
}
NEW_KEY_DEFAULTS = dict(
    fan_opf=False, fan_opf_lambda_pred=1.0, fan_opf_lambda_fac=0.05,
    fan_opf_lambda_enc=0.02, fan_opf_gamma_fac=0.1, fan_opf_gamma_enc=0.1,
    fan_opf_pred_hidden=0, fan_target_ema=0.996, fan_route="none", fan_route_rank=64,
    fan_route_bias_u=1e-3, fan_rlat_lambda=1.0)


def _nowta_kw(**kw) -> dict:
    base = dict(spandec=True, fan_all_wta_lambda=0.0)
    base.update(kw)
    return _fan_kw(M, **base)


def _opf(model_kw: dict | None = None, **kw) -> MORPHTransformer:
    m = _build(model_kw=model_kw, **_nowta_kw(fan_opf=True, **kw))
    assert m.tul_fan_target_build()
    return m


def _target(m: MORPHTransformer, inp, lab, layout, x_online=None) -> dict:
    fkw, freset, _, _ = m._tul_tg_kwargs(layout)
    if x_online is None:
        x_online, _, _ = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    return m._tul_fan_target(inp, lab, layout, fkw, freset, x_online)


def _capture(m: MORPHTransformer, name: str, inp, lab, layout, seed: int = 5):
    spy = _Spy(m, name)
    torch.manual_seed(seed)
    out = m(inp, labels=lab, slot_layout=layout)
    delattr(m, name)
    assert len(spy.calls) == 1, name
    return out, spy


# ── 1. the defaults are the tree ────────────────────────────────────────────────────


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_nowta_pins_hold_with_the_keys_implicit_and_explicit(dropout):
    a = _build(model_kw={"dropout": dropout}, **_nowta_kw())
    b = _build(model_kw={"dropout": dropout}, **_nowta_kw(**NEW_KEY_DEFAULTS))
    assert a.tul_fan_opf is None and a.tul_fan_router is None and a.tul_fan_teacher is None
    assert not a.tul_fan_target_build()                    # builds nothing, returns False
    assert a.__dict__["_fan_target"] is None
    assert a.state_dict().keys() == b.state_dict().keys()
    assert not any(k.startswith(("tul_fan_opf", "tul_fan_router", "tul_fan_teacher"))
                   for k in a.state_dict())
    assert _fan_pin_run(a) == NOWTA_PINS[dropout]
    assert _fan_pin_run(b) == NOWTA_PINS[dropout]


def test_after_step_is_a_no_op_on_nowta():
    m = _build(**_nowta_kw())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.tul_fan_after_step()
    m.tul_fan_reset_route_load()
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k


# ── 2. arm F is nowta plus a term ───────────────────────────────────────────────────


def test_opf_build_is_rng_neutral_and_the_token_ce_is_nowtas():
    _ids0, inp, lab, layout = _batch(M)
    a = _build(model_kw={"dropout": 0.1}, **_nowta_kw()).train()
    b = _opf(model_kw={"dropout": 0.1}).train()
    sa, sb = a.state_dict(), b.state_dict()
    extra = set(sb) - set(sa)
    assert extra and all(k.startswith("tul_fan_opf.") for k in extra), sorted(extra)
    for k, v in sa.items():
        assert torch.equal(v, sb[k]), k
    torch.manual_seed(11)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(11)
    ob = b(inp, labels=lab, slot_layout=layout)
    # The twin's front draws no RNG, so every dropout mask and depth draw of the live
    # forward is nowta's: every reading the arm does not touch is bit-equal, and the
    # objective differs by the OPF term alone.
    for key in ("spandec_ce", "fixed_point", "fan_repel", "n_tokens", "gain_est"):
        assert torch.equal(oa[key], ob[key]), key
    assert float(ob["opf_weighted"]) > 0.0
    assert float(ob["ce_main"]) == pytest.approx(
        float(oa["ce_main"]) + float(ob["opf_weighted"]), rel=0, abs=1e-5)
    assert float(ob["loss"]) == pytest.approx(float(oa["loss"]) + float(ob["opf_weighted"]),
                                              rel=0, abs=1e-5)
    assert "opf_weighted" not in oa and "rlat_weighted" not in ob


def test_labelled_forward_without_a_twin_raises_and_label_free_is_nowta():
    _ids0, inp, lab, layout = _batch(M)
    m = _build(**_nowta_kw(fan_opf=True)).eval()
    with pytest.raises(RuntimeError, match="twin was never built"):
        m(inp, labels=lab, slot_layout=layout)
    n = _build(**_nowta_kw()).eval()
    with torch.no_grad():
        lm = m(inp, labels=None, slot_layout=layout)["logits"]
        ln = n(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(lm, ln)


# ── 3. the target ───────────────────────────────────────────────────────────────────


def _span_positions(layout, b: int, bag: int) -> list[int]:
    sel = (layout.bag_id[b] == bag) & (~layout.slot_mask[b])
    return [int(p) for p in torch.nonzero(sel).flatten()]


def _swap(tok: int) -> int:
    """A different ordinary token (never a boundary, the slot id or eos; < V = 64)."""
    t = 5 + (tok + 2) % 59
    while t in (DOT, 11, 4, 0, tok):
        t = 5 + (t - 4) % 59
    return t


def test_target_of_slot_s_depends_on_span_s_plus_1_only():
    _ids0, inp, lab, layout = _batch(M)
    m = _opf().eval()
    b, s = 0, 3
    assert bool(layout.slot_valid[b, s + 1])
    with torch.no_grad():
        z0 = _target(m, inp, lab, layout)
    assert bool(z0["ok"][b, s])
    other = [p for p in range(inp.shape[1])
             if not bool(layout.slot_mask[b, p]) and int(layout.bag_id[b, p]) != s + 1]
    assert len(other) > 20
    inp2 = inp.clone()
    for p in other:
        inp2[b, p] = _swap(int(inp[b, p]))
    with torch.no_grad():
        z1 = _target(m, inp2, lab, layout)
    assert torch.equal(z1["z"][b, s], z0["z"][b, s])
    assert not torch.equal(z1["z"][b, s + 2], z0["z"][b, s + 2])   # the edit did land
    own = _span_positions(layout, b, s + 1)
    inp3 = inp.clone()
    inp3[b, own[1]] = _swap(int(inp[b, own[1]]))
    with torch.no_grad():
        z2 = _target(m, inp3, lab, layout)
    assert not torch.equal(z2["z"][b, s], z0["z"][b, s])


def test_target_is_the_twin_draws_no_rng_and_equals_the_live_pool_at_build():
    _ids0, inp, lab, layout = _batch(M)
    m = _opf().eval()                                        # dropout 0.0
    with torch.no_grad():
        rng = torch.get_rng_state()
        t0 = _target(m, inp, lab, layout)
        assert torch.equal(rng, torch.get_rng_state())
        # The twin is a copy of the live prelude at build: the same pooling of the live
        # front reproduces the target.
        assert torch.allclose(t0["z"], t0["zo"], atol=1e-6)
        # Move the LIVE prelude only: the target does not move (it is the twin's) ...
        for p in m.prelude.parameters():
            p.add_(0.05 * torch.randn_like(p))
        t1 = _target(m, inp, lab, layout)
        assert torch.equal(t1["z"], t0["z"])
        assert not torch.allclose(t1["zo"], t0["zo"])
        # ... until the twin is synced to it.
        assert m.tul_fan_target_sync(None) == "snapshot"
        t2 = _target(m, inp, lab, layout)
        assert torch.allclose(t2["z"], t1["zo"], atol=1e-6)
    n_tok = span_token_counts(*span_ce_index(lab, layout)[:2],
                              span_ce_index(lab, layout)[3])[:, 1:]
    assert torch.equal(t0["ok"], layout.slot_valid & (n_tok > 0))


def test_pooled_span_states_is_the_bin_mean_then_layernorm():
    _ids0, inp, lab, layout = _batch(M)
    torch.manual_seed(0)
    x = torch.randn(inp.shape[0], inp.shape[1], 3, 8)
    gid, keep_tok, _lab, G = span_ce_index(lab, layout)
    got = pooled_span_states(x, gid, keep_tok, G)
    xs = x.mean(dim=2)
    for b in range(inp.shape[0]):
        for s in range(G - 1):
            sel = keep_tok[b] & (layout.bag_id[b].clamp(0, G - 1) == s + 1)
            if not bool(sel.any()):
                assert torch.equal(got[b, s], torch.zeros(8))
                continue
            ref = torch.nn.functional.layer_norm(xs[b][sel].mean(dim=0), (8,))
            assert torch.allclose(got[b, s], ref, atol=1e-5), (b, s)


# ── 4. who gets which gradient ──────────────────────────────────────────────────────


def _opf_graph(m, inp, lab, layout):
    m.train()
    out, spy = _capture(m, "_tul_fan_opf", inp, lab, layout)
    cells = spy.calls[0][0][0]
    loss = spy.outs[0]
    return out, cells, loss


def test_l_pred_reaches_the_cells_and_the_loop_and_never_the_twin():
    _ids0, inp, lab, layout = _batch(M)
    m = _opf()
    out, cells, loss = _opf_graph(m, inp, lab, layout)
    g_cells, = torch.autograd.grad(loss, cells, retain_graph=True)
    assert float(g_cells.abs().sum()) > 0
    core = [p for p in m.core.parameters() if p.requires_grad]
    g_core = torch.autograd.grad(loss, core, retain_graph=True, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in g_core if g is not None) > 0
    g_p, = torch.autograd.grad(loss, [m.tul_fan_opf.P], retain_graph=True)
    assert float(g_p.abs().sum()) > 0
    twin = m.__dict__["_fan_target"]
    assert all(not p.requires_grad for p in twin.parameters())
    out["loss"].backward()
    assert all(p.grad is None for p in twin.parameters())


def test_l_fac_reaches_p_only_and_l_enc_reaches_the_online_prelude():
    _ids0, inp, lab, layout = _batch(M)
    # L_fac alone with an active hinge (gamma 10 >> every std): P only, never the cells.
    m = _opf(fan_opf_lambda_pred=0.0, fan_opf_lambda_enc=0.0, fan_opf_gamma_fac=10.0)
    _out, cells, loss = _opf_graph(m, inp, lab, layout)
    g_p, = torch.autograd.grad(loss, [m.tul_fan_opf.P], retain_graph=True)
    assert float(g_p.abs().sum()) > 0
    g_c, = torch.autograd.grad(loss, cells, retain_graph=True, allow_unused=True)
    assert g_c is None or float(g_c.abs().sum()) == 0.0
    # L_enc alone with an active hinge: the online prelude, never the cells or P.
    m = _opf(fan_opf_lambda_pred=0.0, fan_opf_lambda_fac=0.0, fan_opf_gamma_enc=10.0)
    _out, cells, loss = _opf_graph(m, inp, lab, layout)
    pre = [p for p in m.prelude.parameters() if p.requires_grad]
    g_pre = torch.autograd.grad(loss, pre, retain_graph=True, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in g_pre if g is not None) > 0
    g_c, g_p = torch.autograd.grad(loss, [cells, m.tul_fan_opf.P], retain_graph=True,
                                   allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in (g_c, g_p))


def test_inactive_hinges_carry_no_gradient_at_the_default_gamma():
    """On this fixture every std sits above 0.1, so the default L_fac and L_enc read 0."""
    _ids0, inp, lab, layout = _batch(M)
    m = _opf()
    out = m.train()(inp, labels=lab, slot_layout=layout)
    assert float(out["fan_opf_fac"]) == 0.0 and float(out["fan_opf_enc"]) == 0.0
    assert float(out["fan_opf_enc_std_min"]) > 0.1


# ── 5. the math ─────────────────────────────────────────────────────────────────────


def test_opf_terms_equal_an_independent_masked_index_computation():
    torch.manual_seed(1)
    N, K, r, d = 12, 4, 3, 12
    q, zf, zo = torch.randn(N, K, r), torch.randn(N, K, r), 0.2 * torch.randn(N, d)
    w = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.bool)
    lp, lf, le, st = opf_terms(q, zf, zo, w, 0.9, 0.25)
    qv, fv, ov = q[w], zf[w], zo[w]

    def pstd(x):
        return torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
    assert lp == pytest.approx(float(((qv - fv) ** 2).mean()), abs=1e-6)
    assert lf == pytest.approx(float(torch.relu(0.9 - pstd(fv)).mean()), abs=1e-6)
    assert le == pytest.approx(float(torch.relu(0.25 - pstd(ov)).mean()), abs=1e-6)
    for k in range(K):
        mse = float(((qv[:, k] - fv[:, k]) ** 2).mean())
        var = float(fv[:, k].var(dim=0, unbiased=False).mean())
        assert float(st[f"opf_r2_k{k}"]) == pytest.approx(1 - mse / var, abs=1e-5)
    assert float(st["opf_enc_std_min"]) == pytest.approx(float(pstd(ov).min()), abs=1e-6)
    # A prediction of the batch mean reads R^2 = 0, a perfect one 1.
    _, _, _, st0 = opf_terms(fv.mean(0, keepdim=True).expand_as(fv), fv, ov,
                             torch.ones(len(fv), dtype=torch.bool), 0.1, 0.1)
    _, _, _, st1 = opf_terms(fv, fv, ov, torch.ones(len(fv), dtype=torch.bool), 0.1, 0.1)
    assert float(st0["opf_r2_mean"]) == pytest.approx(0.0, abs=1e-5)
    assert float(st1["opf_r2_mean"]) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_reads_rank():
    torch.manual_seed(2)
    w = torch.ones(40, dtype=torch.bool)
    one = torch.randn(40, 1) * torch.randn(1, 16)
    assert float(participation_ratio(one, w)) == pytest.approx(1.0, abs=1e-4)
    iso = torch.linalg.qr(torch.randn(40, 16))[0] * 3.0          # 16 equal directions
    assert float(participation_ratio(iso, w)) == pytest.approx(16.0, rel=0.1)
    # A zero-weight row is ignored: an outlier outside the mask does not move it.
    x = one.clone()
    x[0] = 1e3
    w2 = w.clone()
    w2[0] = False
    assert float(participation_ratio(x, w2)) == pytest.approx(1.0, abs=1e-4)
    assert float(batch_std(x, w2).max()) < 1e2


def test_qr_retraction_makes_p_orthonormal_and_fixes_an_orthonormal_p():
    opf = FanOPF(16, 4, 0)
    with torch.no_grad():
        opf.P.add_(0.3 * torch.randn(16, 16))
    assert float(opf.orth_err()) > 1e-2
    raw = opf.P.detach().clone()
    opf.retract_()
    p = opf.P.detach()
    assert float(opf.orth_err()) < 1e-5
    q, r = torch.linalg.qr(raw.t())
    s = torch.sign(torch.diagonal(r))
    assert torch.allclose(p, (q * s).t(), atol=1e-5)
    # The retraction spans the same row space per leading block (QR's triangularity):
    # the first r rows of P are an orthonormal basis of the first r raw rows.
    proj = p[:4].t() @ p[:4]
    assert torch.allclose(proj @ raw[:4].t(), raw[:4].t(), atol=1e-4)
    opf.retract_()
    assert torch.allclose(opf.P.detach(), p, atol=1e-6)


def test_after_step_retracts_p_and_moves_the_twin_by_the_ema_rule():
    _ids0, inp, lab, layout = _batch(M)
    m = _opf(fan_target_ema=0.9)
    twin = m.__dict__["_fan_target"]
    old = {k: v.detach().clone() for k, v in twin.own_tensors().items()}
    with torch.no_grad():
        for p in m.prelude.parameters():
            p.add_(0.1 * torch.randn_like(p))
        for p in m.x0_injects[0].parameters():
            p.add_(0.1 * torch.randn_like(p))
        m.tul_fan_opf.P.add_(0.2 * torch.randn_like(m.tul_fan_opf.P))
    live = {k: v.detach().clone() for k, v in twin.live_tensors(m, m.cfg.n_prelude).items()}
    m.tul_fan_after_step()
    moved = 0
    for k, v in twin.own_tensors().items():
        exp = 0.9 * old[k] + 0.1 * live[k] if v.is_floating_point() else live[k]
        assert torch.allclose(v, exp, atol=1e-6), k
        moved += int(not torch.equal(old[k], live[k]))
    assert moved > 10
    assert float(m.tul_fan_opf.orth_err()) < 1e-5
    # Only the prelude's first n_prelude x0 injections are mirrored (the coda's are not).
    assert len(twin.x0_injects) == m.cfg.n_prelude < len(m.x0_injects)


def test_twin_round_trips_through_the_checkpoint(tmp_path):
    from morph.training.train import read_checkpoint_side_state, save_checkpoint
    m = _opf(fan_target_ema=0.5)
    with torch.no_grad():
        for p in m.prelude.parameters():
            p.add_(0.1 * torch.randn_like(p))
    m.tul_fan_after_step()                       # the twin now differs from both inits
    path = str(tmp_path / "ck.pt")
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    save_checkpoint(path, 3, m, opt, torch.amp.GradScaler("cpu", enabled=False), None)
    state = read_checkpoint_side_state(path, "fan_target")
    assert state is not None
    m2 = _opf()
    assert m2.tul_fan_target_sync(state) == "resume"
    a, b = m.__dict__["_fan_target"].state_dict(), m2.__dict__["_fan_target"].state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    # the twin is NOT inside the model state
    assert not any("_fan_target" in k for k in torch.load(path, weights_only=False)["model"])
    # a nowta checkpoint carries no twin: the side reader returns None (snapshot case)
    n = _build(**_nowta_kw())
    path2 = str(tmp_path / "n.pt")
    save_checkpoint(path2, 1, n, torch.optim.SGD(n.parameters(), lr=0.1),
                    torch.amp.GradScaler("cpu", enabled=False), None)
    assert read_checkpoint_side_state(path2, "fan_target") is None


# ── 6. refusals ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,match", [
    (dict(fan_opf=True, fan_route="reader"), NotImplementedError, "exclude each other"),
    (dict(fan_opf=True, fan_all_wta_lambda=1.0), NotImplementedError, "fan_all_wta_lambda"),
    (dict(fan_route="latent", fan_all_wta_lambda=1.0), NotImplementedError,
     "fan_all_wta_lambda"),
    (dict(fan_opf=True, code_enum_k=4), NotImplementedError, "code_enum_k=4"),
    (dict(fan_route="reader", code_enum_k=4), NotImplementedError, "code_enum_k=4"),
    (dict(fan_opf=True, bcast=True), NotImplementedError, "bcast"),
    (dict(fan_route="bogus"), ValueError, "fan_route must be"),
    (dict(fan_opf=True, fan_target_ema=1.0), ValueError, "fan_target_ema"),
    (dict(fan_opf=True, fan_opf_gamma_fac=-1.0), ValueError, "fan_opf_gamma_fac"),
    (dict(fan_route="reader", fan_route_rank=0), ValueError, "fan_route_rank"),
    (dict(fan_opf_lambda_fac=0.5), ValueError, "without the arm"),
    (dict(fan_target_ema=0.99), ValueError, "without the arm"),
    (dict(fan_route="reader", fan_target_ema=0.99), ValueError, "without the arm"),
    (dict(fan_route_bias_u=0.1), ValueError, "without the arm"),
    (dict(fan_route="reader", fan_rlat_lambda=0.5), ValueError, "without the arm"),
    (dict(fan_opf=True, fan_route_rank=8), ValueError, "without the arm"),
])
def test_refusals(kw, exc, match):
    with pytest.raises(exc, match=match):
        TULConfig(**_nowta_kw(**kw))


@pytest.mark.parametrize("arm", [dict(fan_opf=True), dict(fan_route="reader"),
                                 dict(fan_route="latent")])
def test_arms_need_the_write_all_fan(arm):
    with pytest.raises(NotImplementedError, match="fan_k=0"):
        TULConfig(**arm)
    with pytest.raises(NotImplementedError, match="fan_mix='mean'"):
        TULConfig(**_fan_kw(M, fan_mix="mean", fan_all_wta_lambda=1.0, fan_select_eps=0.05,
                            **arm))
    TULConfig(**_nowta_kw(**arm))


def test_opf_refuses_a_fan_that_does_not_divide_d_model():
    with pytest.raises(ValueError, match="divide d_model"):
        MORPHTransformer(_tiny(tul=TULConfig(**_fan_kw(3, spandec=True,
                                                         fan_all_wta_lambda=0.0,
                                                         fan_opf=True)), d_ff=96))


# ── 7. configs ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,stated,wb", [
    ("tul_slot_spandec_strict_fan4_all_fp01_opf", {"tul.fan_opf": True},
     "slot-spandec-strict-fan4-all-fp01-opf"),
    ("tul_slot_spandec_strict_fan4_all_fp01_rmoe", {"tul.fan_route": "reader"},
     "slot-spandec-strict-fan4-all-fp01-rmoe"),
    ("tul_slot_spandec_strict_fan4_all_fp01_rlat", {"tul.fan_route": "latent"},
     "slot-spandec-strict-fan4-all-fp01-rlat"),
])
def test_configs_compose_one_key_from_nowta_reach_the_model_and_train(name, stated, wb,
                                                                       monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_fan4_all_fp01_nowta"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == set(stated) | {"wandb.name"}, sorted(diff)
    assert c["wandb.name"] == wb and c["training.steps"] == p["training.steps"] == 5000
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    for k, v in stated.items():
        key = k.split(".", 1)[1]
        assert getattr(rt.model_cfg, key) == v, k
        assert getattr(mc.tul, key) == v, k
    for key, v in NEW_KEY_DEFAULTS.items():
        if f"tul.{key}" not in stated:
            assert getattr(mc.tul, key) == v, key
    assert mc.tul.fan_all_wta_lambda == 0.0 and mc.tul.fan_mix == "all"
    assert mc.core_fixed_point_lambda == 0.1
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=96,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)
                         ).train().float()
    assert m.tul_fan_target_build() == name.endswith(("_opf", "_rlat"))
    _ids0, inp, lab, layout = _batch(M)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert ("opf_weighted" in out) == name.endswith("_opf")
    assert ("rlat_weighted" in out) == name.endswith("_rlat")
    assert ("fan_route_entropy" in out) == name.endswith(("_rmoe", "_rlat"))
    m.tul_fan_after_step()


# ── 8. train.py: val loss is the model's CE; the readings are routed ────────────────


class _AuxStub(torch.nn.Module):
    def tul_forward_with_plan_nats(self, x, y, layout):
        return {"loss": torch.tensor(7.0), "opf_weighted": torch.tensor(1.5),
                "rlat_weighted": torch.tensor(1.0),
                "fan_opf_r2_k0": torch.tensor(0.3), "fan_route_load_k0": torch.tensor(0.4),
                "fan_rlat_ce": torch.tensor(1.2), "fan_teacher_router_agree": torch.tensor(0.6),
                "fan_router_coda_agree": torch.tensor(0.5),
                "fan_router_pick_regret": torch.tensor(0.01),
                "fan_rand_pick_regret": torch.tensor(0.02),
                "fan_teacher_coda_agree": torch.tensor(0.3),
                "ce_tokens": 4.5, "layer_passes": 8.0, "n_tokens": 4.0}


class _Layout:
    stats: dict = {}

    def to(self, device):
        return self


def test_evaluate_subtracts_both_terms_and_routes_the_readings():
    from morph.training.train import evaluate
    x = torch.zeros(1, 4, dtype=torch.long)
    extra: dict = {}
    avg, _ppl = evaluate(_AuxStub(), torch.device("cpu"), iter([(x, x, _Layout())] * 2),
                         n_batches=2, tul=True, extra=extra)
    assert avg == pytest.approx(4.5)
    # built at train too -> val/fan_*, never contending with the train series fan/*
    for k, v in (("opf_r2_k0", 0.3), ("route_load_k0", 0.4), ("rlat_ce", 1.2),
                 ("teacher_router_agree", 0.6)):
        assert extra[f"val/fan_{k}"] == pytest.approx(v)
        assert f"fan/{k}" not in extra
    # eval-only oracle readings stay in fan/
    for k, v in (("router_coda_agree", 0.5), ("router_pick_regret", 0.01),
                 ("rand_pick_regret", 0.02), ("teacher_coda_agree", 0.3)):
        assert extra[f"fan/{k}"] == pytest.approx(v)


def _tuple_after(src: str, var: str) -> set[str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == var and isinstance(node.iter, ast.Tuple)):
            return {e.value for e in node.iter.elts if isinstance(e, ast.Constant)}
    raise AssertionError(f"no `for {var} in (...)` loop in train.py")


def test_both_train_py_subtraction_tuples_list_both_keys():
    src = pathlib.Path("morph/training/train.py").read_text()
    for key in ("opf_weighted", "rlat_weighted"):
        assert key in _tuple_after(src, "_aux2"), key
        assert key in _tuple_after(src, "_ak"), key


def test_train_py_calls_the_lifecycle_hooks():
    """SOURCE check: the trainer builds the twin right after quantisation and before the
    compile, re-syncs it after the weights load, steps it after the optimizer and clears
    the router's warmup counts before the loop."""
    src = pathlib.Path("morph/training/train.py").read_text()
    i_q = src.index("_qm = apply_quantization(model, cfg)")
    i_b = src.index("model.tul_fan_target_build()")
    i_c = src.index("# ── torch.compile ──")
    i_s = src.index("_mdl0.tul_fan_target_sync(")
    i_l = src.index("_mdl0.tul_fan_reset_route_load()")
    i_loop = src.index("# ── Training loop ──")
    i_opt = src.index("_mdl0.tul_fan_after_step()")
    assert i_q < i_b < i_c < i_s < i_l < i_loop < i_opt
    assert 'ckpt["fan_target"] = _fan_tgt' in src


def test_the_lab_loader_restores_the_twin_and_refuses_a_checkpoint_without_it(
        tmp_path, monkeypatch):
    """`scripts/tul_samples.py::load_ckpt` — every readout the queue runs (the depth sweep,
    the Stage-2 scorer, worth_profile) loads through it — builds the twin after the
    weights and loads it STRICTLY from `fan_target`; without the key it refuses."""
    import importlib.util
    import sys as _sys
    from morph.training.train import save_checkpoint
    spec = importlib.util.spec_from_file_location(
        "tul_samples_under_test", pathlib.Path("scripts/tul_samples.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    src = _opf(fan_target_ema=0.5)
    with torch.no_grad():
        for p in src.prelude.parameters():
            p.add_(0.1 * torch.randn_like(p))
    src.tul_fan_after_step()
    path = str(tmp_path / "f.pt")
    save_checkpoint(path, 7, src, torch.optim.SGD(src.parameters(), lr=0.1),
                    torch.amp.GradScaler("cpu", enabled=False), None)
    monkeypatch.setattr(ts, "build_model_with_quant",
                        lambda cfg, device, tul=None: _build(**_nowta_kw(fan_opf=True)))
    m, step = ts.load_ckpt(None, path, "cpu", None)
    assert step == 7
    a, b = src.__dict__["_fan_target"].state_dict(), m.__dict__["_fan_target"].state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    ck = torch.load(path, weights_only=False)
    del ck["fan_target"]
    torch.save(ck, path)
    with pytest.raises(SystemExit):
        ts.load_ckpt(None, path, "cpu", None)
    _sys.modules.pop("tul_samples_under_test", None)
