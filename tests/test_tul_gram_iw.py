"""`tul.gram_objective: "iw"` — LXTUL-GK, the multi-sample bound over K prior rollouts.

Prereg lab/experiments/planned/2026-09-23-lxtul-gk-multisample.md; model
morph/model/tul_gram.py ("The multi-sample objective"); the per-token Bayesian read it must
equal is lab/divergence/lxtul_g_probe.py::bayes_read.

Contracts, one test (or one parametrised family) each:

  * DEFAULT is the shipped LXTUL-G: the key absent and `gram_objective="elbo"` give the same
    loss, logits, parameters and gradients (gram and non-gram models). (The stronger gate,
    this tree against 9e71efa on the ruler and two gram models, is an out-of-tree script
    run at build time; see the build report.)
  * THE BOUND: on random tables, per span it equals the probe's `bayes_read` sum, weighted
    spans equal the direct formula, rows never merge, K = 1 is the weighted sum exactly,
    and its gradient into the per-rollout span log-likelihood is the softmax credit.
  * THE MODEL: the TRAINING forward's objective, with its noise drawn from the eval
    generators, equals `bayes_read` over K single-sample eval forwards of the probe; the
    eval `gram_mode="iw"` read equals it too; K = 1 equals the ruler-style weighted CE of
    that rollout (eval) and the LXTUL-G training loss at init (train, where q == p).
  * ONE FRONT, K LOOPS: the prelude runs once per forward, the core and the coda see K*B
    rows; the K rollouts share the depth draw and the dropout mask and differ in noise.
  * EVAL is unchanged: a gk4 model's default eval forward equals an "elbo" model's on the
    same weights, and no posterior is built.
  * NO LEAK AT TRAIN: an edit to span s+1 leaves slot s's written cells bit-exact in the
    TRAINING forward of every rollout (there is no posterior); an "elbo" model's training
    forward moves them (the positive control).
  * the per-row log-prob kernel against autograd; every refusal; the two configs compose.

CPU, fp32, the `tests/test_tul_strict_geometry.py` fixtures.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from morph.model.fused_ce import fused_linear_cross_entropy, fused_linear_label_logprob
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_gram import iw_span_bound, iw_span_groups
from test_tul_gram import _gram, _open_e_next
from test_tul_strict_geometry import _edit, _pack, _rule, _spec, _tiny, _tul

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAB = os.path.join(_ROOT, "lab", "divergence")
if _LAB not in sys.path:
    sys.path.insert(0, _LAB)


def _gk(K: int = 4, **kw) -> MORPHTransformer:
    return _gram(gram_objective="iw", gram_iw_k=K, **kw)


def _train(m, inp, lab, layout, seed: int = 0, **kw):
    m.train()
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout, **kw)


def _seed_train_noise(m: MORPHTransformer, seed: int) -> None:
    """Make the TRAINING forward draw rollout k's Gaussian steps from the generator an eval
    forward seeded ``seed + k`` uses (the eval `gram_mode="iw"` rule), so the training
    objective can be compared with the probe's single-sample eval forwards."""
    real = m._gram_ctx

    def patched(xn, layout, gram_mode, gram_seed, n_nograd, iw_rollouts=1):
        ctx = real(xn, layout, gram_mode, gram_seed, n_nograd, iw_rollouts)
        ctx["gen"] = torch.Generator(device=xn.device)
        ctx["iw_seeds"] = [seed + k for k in range(iw_rollouts)]
        return ctx
    m._gram_ctx = patched


def _probe_read(m, inp, lab, layout, seeds) -> float:
    """``-sum_j log p_read(tok_j) / n``: the probe's per-token Bayesian read over the
    single-sample eval forwards at ``seeds``, per row, as `gram_probe` computes it."""
    from lxtul_g_probe import bayes_read, label_logp
    m.eval()                                  # gram_probe's own first line
    tp = (~layout.slot_mask) & (lab >= 0)
    rows = torch.arange(inp.shape[0]).view(-1, 1).expand_as(tp)[tp].numpy()
    bags = layout.bag_id[tp].numpy()
    st = np.stack([label_logp(m, inp, layout, lab, "cpu", gram_mode="prior", seed=s)[0][tp]
                   .numpy().astype(np.float64) for s in seeds])
    tot = sum(float(bayes_read(st[:, rows == r], bags[rows == r]).sum())
              for r in np.unique(rows))
    return -tot / float(tp.sum())


# ── DEFAULT is the shipped LXTUL-G ───────────────────────────────────────────

def _full_run(m, inp, lab, layout):
    out = _train(m, inp, lab, layout)
    out["loss"].backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.eval()
    with torch.no_grad():
        logits = m(inp, labels=None, slot_layout=layout)["logits"]
    return out, grads, logits


@pytest.mark.parametrize("gram", [True, False])
def test_default_objective_is_elbo_bit_identical(gram):
    _ids, inp, lab, layout = _pack()
    kw = dict(tg_geometry="strict", token_state_dropout=0.15, spandec=True)
    a = _gram(**kw) if gram else _gram_off(**kw)
    b = (_gram(gram_objective="elbo", **kw) if gram
         else _gram_off(gram_objective="elbo", **kw))
    assert a._gram_iw_k == 0 and b._gram_iw_k == 0
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka
    oa, ga, la = _full_run(a, inp, lab, layout)
    ob, gb, lb = _full_run(b, inp, lab, layout)
    assert torch.equal(oa["loss"], ob["loss"])
    assert set(ga) == set(gb) and all(torch.equal(ga[n], gb[n]) for n in ga)
    assert torch.equal(la, lb)
    assert not any(k.startswith("gk_") for k in oa)
    if gram:
        assert "gram_kl_weighted" in oa and a.tul_gram.pool is not None


def _gram_off(**kw) -> MORPHTransformer:
    from test_tul_strict_geometry import _model
    kw = dict(kw)
    kw.pop("tg_geometry", None)
    return _model(tg_geometry="strict", **kw)


# ── THE BOUND on tables ──────────────────────────────────────────────────────

def _table(K=5, B=3, L=30, S=6, seed=0):
    rng = np.random.default_rng(seed)
    lp = np.log(rng.uniform(0.01, 1.0, size=(K, B * L)))
    bag = np.sort(rng.integers(0, S + 1, size=(B, L)), axis=1)     # contiguous runs per row
    return lp, bag


def test_bound_equals_bayes_read_per_span_on_random_tables():
    from lxtul_g_probe import bayes_read
    for K in (1, 2, 5):
        lp, bag = _table(K=K, seed=K)
        B, L = bag.shape
        S = int(bag.max())
        grp = iw_span_groups(torch.as_tensor(bag), S).reshape(-1)
        w = torch.ones(B * L)
        total, Ssum, scored = iw_span_bound(torch.as_tensor(lp), w, grp, B * (S + 1))
        ref = 0.0
        for r in range(B):
            sl = slice(r * L, (r + 1) * L)
            out = bayes_read(lp[:, sl], bag[r])
            ref += out.sum()
            # per span: the group's own bound equals the read summed over the span
            lme = torch.logsumexp(Ssum, 0) - math.log(K)
            for b in np.unique(bag[r]):
                g = r * (S + 1) + int(b)
                assert bool(scored[g])
                assert abs(float(lme[g]) - out[bag[r] == b].sum()) < 1e-4
        assert abs(float(total) - ref) < 1e-3, (K, float(total), ref)


def test_weighted_spans_and_rows_never_merge():
    """Weights w_j in {0, 0.5, 1} (the ruler's plast / unscored positions): the bound is
    the direct formula on weighted sums, per (row, span). Two rows with the SAME span id
    are two groups: merging them changes the value."""
    K, B, L, S = 3, 2, 20, 4
    lp, bag = _table(K=K, B=B, L=L, S=S, seed=7)
    rng = np.random.default_rng(1)
    w = rng.choice([0.0, 0.5, 1.0], size=B * L)
    grp = iw_span_groups(torch.as_tensor(bag), S).reshape(-1)
    total, _S, _sc = iw_span_bound(torch.as_tensor(lp), torch.as_tensor(w), grp, B * (S + 1))
    ref = 0.0
    flat_bag = bag.reshape(-1)
    for r in range(B):
        for b in range(S + 1):
            sel = np.zeros(B * L, dtype=bool)
            sel[r * L:(r + 1) * L] = flat_bag[r * L:(r + 1) * L] == b
            if not (sel & (w != 0)).any():
                continue
            sk = (lp[:, sel] * w[sel]).sum(1)
            ref += np.log(np.mean(np.exp(sk)))
    assert abs(float(total) - ref) < 1e-4
    merged = iw_span_bound(torch.as_tensor(lp), torch.as_tensor(w),
                           torch.as_tensor(flat_bag), S + 1)[0]
    assert abs(float(merged) - ref) > 1e-3, "merging rows did not change the bound"


def test_k1_bound_is_the_weighted_sum_exactly():
    lp, bag = _table(K=1, seed=3)
    w = torch.rand(lp.shape[1])
    S = int(bag.max())
    grp = iw_span_groups(torch.as_tensor(bag), S).reshape(-1)
    lp_t = torch.as_tensor(lp).float()
    total, _S, _sc = iw_span_bound(lp_t, w, grp, bag.shape[0] * (S + 1))
    ref = (lp_t[0] * w).sum()
    assert torch.allclose(total, ref, rtol=1e-6, atol=1e-5)


def test_bound_gradient_is_the_softmax_credit():
    K, B, L, S = 4, 2, 12, 3
    lp, bag = _table(K=K, B=B, L=L, S=S, seed=11)
    w = torch.as_tensor(np.random.default_rng(2).choice([0.5, 1.0], size=B * L)).float()
    grp = iw_span_groups(torch.as_tensor(bag), S).reshape(-1)
    x = torch.as_tensor(lp).float().requires_grad_(True)
    total, Ssum, _sc = iw_span_bound(x, w, grp, B * (S + 1))
    (g,) = torch.autograd.grad(total, x)
    credit = torch.softmax(Ssum.detach(), dim=0)                     # [K, G]
    want = credit[:, grp] * w.unsqueeze(0)
    assert torch.allclose(g, want, rtol=1e-5, atol=1e-7)
    # the credit is NOT uniform (else the test would pass for a mean of CEs too)
    assert float(credit.std(0).max()) > 1e-2


# ── THE MODEL: the training objective equals the probe's Bayesian read ──────

@pytest.mark.parametrize("K", [4, 3])
def test_training_objective_equals_probe_bayes_read(K):
    """The TRAINING forward (front once, K rollouts, the bound), its noise drawn from the
    eval generators, against the probe's per-token read over K single-sample eval forwards.
    Depth fixed and dropout off, so train and eval rollouts are the same function."""
    _ids, inp, lab, layout = _pack()
    m = _gk(K, plast_weight=1.0, slot_depth_fixed=2)
    # the probe first: `_seed_train_noise` patches the model's noise for good
    ref = _probe_read(m, inp, lab, layout, [5 + k for k in range(K)])
    single = _probe_read(m, inp, lab, layout, [5])
    _seed_train_noise(m, seed=5)
    out = _train(m, inp, lab, layout)
    assert abs(float(out["gk_ce_iw"]) - ref) < 2e-5, (float(out["gk_ce_iw"]), ref)
    assert abs(float(out["loss"]) - ref) < 2e-5
    assert abs(float(out["gk_ce_single"]) - single) < 2e-5
    assert abs(float(out["gk_width_gain"]) - (single - ref)) < 4e-5
    assert float(out["gk_width_gain"]) != 0.0


def test_eval_iw_read_equals_probe_bayes_read():
    _ids, inp, lab, layout = _pack()
    m = _gk(4, plast_weight=1.0)
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout, gram_mode="iw", gram_sample_seed=3)
    assert o["gram_mode"] == "iw"
    ref = _probe_read(m, inp, lab, layout, [3, 4, 5, 6])
    assert abs(float(o["loss"]) - ref) < 2e-5
    ent = float(o["gk_w_entropy"])
    assert 0.0 < ent <= 1.0 + 1e-6


def test_k1_equals_the_ruler_style_weighted_ce_of_that_rollout():
    """plast_weight 0.5 (the fixture default) so the weights are exercised: the K = 1 bound
    is the weighted CE `_tul_group_losses` charges, on the same seeded prior rollout."""
    _ids, inp, lab, layout = _pack()
    m = _gk(1)
    assert m.cfg.tul.plast_weight == 0.5
    m.eval()
    with torch.no_grad():
        o_iw = m(inp, labels=lab, slot_layout=layout, gram_mode="iw", gram_sample_seed=2)
        o_ce = m(inp, labels=lab, slot_layout=layout, gram_sample_seed=2)
    assert "ce_main" in o_ce and "gk_ce_iw" not in o_ce         # the ruler's CE path
    assert abs(float(o_iw["loss"]) - float(o_ce["loss"])) < 1e-5
    assert float(o_iw["gk_width_gain"]) == 0.0 and float(o_iw["gk_w_entropy"]) == 0.0


def test_k1_training_loss_equals_lxtul_g_at_init():
    """At init the posterior IS the prior (q == p, KL exactly 0) and both draw the same
    shapes from the global stream, so the K = 1 training forward and the LXTUL-G training
    forward are the same rollout: their losses agree (the CE through two kernels)."""
    _ids, inp, lab, layout = _pack()
    kw = dict(token_state_dropout=0.15, spandec=True)
    a = _gk(1, **kw)
    b = _gram(**kw)
    oa = _train(a, inp, lab, layout)
    ob = _train(b, inp, lab, layout)
    assert float(ob["gram_kl_weighted"]) == 0.0
    assert abs(float(oa["loss"]) - float(ob["loss"])) < 1e-5
    assert torch.equal(oa["spandec"], ob["spandec"])


# ── ONE FRONT, K LOOPS ───────────────────────────────────────────────────────

def test_front_runs_once_and_core_and_coda_see_k_rollouts():
    _ids, inp, lab, layout = _pack()
    B = inp.shape[0]
    for K in (4, 1):
        m = _gk(K)
        calls = {"prelude": [], "core": [], "coda": []}
        hs = []
        for name in calls:
            for blk in getattr(m, name):
                hs.append(blk.register_forward_hook(
                    lambda _m, a, _o, _n=name: calls[_n].append(a[0].shape[0])))
        out = _train(m, inp, lab, layout)
        for h in hs:
            h.remove()
        assert calls["prelude"] == [B] * m.cfg.n_prelude, calls["prelude"]
        assert calls["coda"] and set(calls["coda"]) == {B * K}
        assert calls["core"] and set(calls["core"]) == {B * K}
        assert int(out["n_tokens"]) == int((~layout.slot_mask).sum())
        out["loss"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.prelude.parameters())
        for head in (m.tul_gram.prior_s, m.tul_gram.prior_m):
            assert head.w_down.weight.grad is not None


def test_rollouts_share_depth_and_dropout_and_differ_in_noise():
    _ids, inp, lab, layout = _pack()
    B = inp.shape[0]
    m = _gk(4, token_state_dropout=0.3)
    seen = {}
    real_drop = m.tul.apply_token_dropout
    real_depth = m._sample_slot_depths

    def drop(x, lay, training, n_rep=1):
        xo, keep = real_drop(x, lay, training, n_rep=n_rep)
        seen["keep"] = keep
        return xo, keep

    def depth(lay, device):
        d = real_depth(lay, device)
        seen["depth_shape"] = tuple(d.shape)
        return d
    m.tul.apply_token_dropout = drop
    m._sample_slot_depths = depth
    m._gram_capture = []
    _train(m, inp, lab, layout)
    cap, m._gram_capture = m._gram_capture, None
    keep = seen["keep"].reshape(4, B, -1)
    assert bool((keep == 0).any())
    assert all(torch.equal(keep[0], keep[k]) for k in range(1, 4))
    assert seen["depth_shape"] == tuple(layout.slot_index.shape)   # drawn on the base rows
    for c in cap:
        eps = c["eps"].reshape(4, B, *c["eps"].shape[1:])
        for k in range(1, 4):
            assert not torch.allclose(eps[0], eps[k]), f"pass {c['t']}: rollouts share noise"



def test_loop_depth_table_is_the_base_draw_tiled_rollout_major():
    """The depth table the loop RUNS (returned by `_tul_core`) is the base rows' draw,
    tiled rollout-major: rows k*B + r all equal base row r. A row-major tiling
    (`repeat_interleave`) would hand rollout 0 the depths of base row 0 four times."""
    _ids, inp, lab, layout = _pack()
    B, K = inp.shape[0], 4
    m = _gk(K)
    seen = {}
    real_depth, real_core = m._sample_slot_depths, m._tul_core

    def depth(lay, device):
        d = real_depth(lay, device)
        seen["base"] = d.clone()
        return d

    def core(*a, **kw):
        out = real_core(*a, **kw)
        seen["used"] = out[2].clone()
        return out
    m._sample_slot_depths = depth
    m._tul_core = core
    _train(m, inp, lab, layout, seed=11)
    base, used = seen["base"], seen["used"]
    assert used.shape[0] == K * B and base.shape[0] == B
    # the draw must differ across base rows, or a mis-tiled table could not be seen
    assert any(not torch.equal(base[0], base[r]) for r in range(1, B)), base
    for k in range(K):
        assert torch.equal(used[k * B:(k + 1) * B], base), k


@pytest.mark.parametrize("spandec", [False, True])
def test_identical_rollouts_give_the_k1_gradient_on_every_parameter(spandec):
    """All K rollouts drawn from ONE seed are the same function, so the bound is that
    rollout's weighted CE and each rollout carries credit 1/K. The gradient summed back
    through the repeats must then equal the K = 1 gradient on EVERY parameter: the front
    (it gets the sum over the K copies of its output), the loop and the coda. A front fed
    by rollout 0 alone would read 1/K of it."""
    _ids, inp, lab, layout = _pack()
    grads, losses = {}, {}
    for K in (1, 4):
        m = _gk(K, token_state_dropout=0.0, spandec=spandec)
        real = m._gram_ctx

        def patched(xn, lay, gram_mode, gram_seed, n_nograd, iw_rollouts=1, _r=real):
            ctx = _r(xn, lay, gram_mode, gram_seed, n_nograd, iw_rollouts)
            ctx["gen"] = torch.Generator(device=xn.device)
            ctx["iw_seeds"] = [7] * iw_rollouts          # every rollout: the same noise
            return ctx
        m._gram_ctx = patched
        out = _train(m, inp, lab, layout, seed=3)
        out["loss"].backward()
        losses[K] = float(out["loss"])
        if K == 4:
            assert abs(float(out["gk_width_gain"])) < 1e-5
        grads[K] = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    assert abs(losses[1] - losses[4]) < 1e-5, losses
    assert grads[1].keys() == grads[4].keys()
    front = [n for n in grads[1] if n.startswith("prelude.")]
    assert front and all(float(grads[1][n].abs().sum()) > 0 for n in front[:1])
    if spandec:     # the decoder (checkpointed at K > 1) is in the compared set
        assert any(n.startswith("tul_spandec.") or ".spandec" in n or "spandec" in n
                   for n in grads[1]), sorted(grads[1])[:5]
    for n in grads[1]:
        torch.testing.assert_close(grads[4][n], grads[1][n], rtol=1e-4, atol=1e-6,
                                   msg=lambda e, _n=n: f"{_n}: {e}")

# ── EVAL is unchanged ────────────────────────────────────────────────────────

def test_no_posterior_built_and_eval_equals_an_elbo_model_on_the_same_weights():
    _ids, inp, lab, layout = _pack()
    gk = _gk(4)
    el = _gram()
    assert gk.tul_gram.pool is None and gk.tul_gram.post_s is None
    assert gk.tul_gram.post_m is None
    gs, es = gk.state_dict(), el.state_dict()
    assert set(gs) <= set(es)
    extra = set(es) - set(gs)
    assert extra and all(k.startswith(("tul_gram.pool.", "tul_gram.post_")) for k in extra)
    for k, v in gs.items():
        assert torch.equal(v, es[k]), f"{k}: the posterior-free build moved a weight"
    for mode_kw in ({}, {"gram_sample_seed": 7}, {"gram_mode": "mean"}):
        gk.eval()
        el.eval()
        with torch.no_grad():
            a = gk(inp, labels=None, slot_layout=layout, **mode_kw)["logits"]
            b = el(inp, labels=None, slot_layout=layout, **mode_kw)["logits"]
            la = gk(inp, labels=lab, slot_layout=layout, **mode_kw)["loss"]
            lb = el(inp, labels=lab, slot_layout=layout, **mode_kw)["loss"]
        assert torch.equal(a, b) and torch.equal(la, lb), mode_kw


# ── NO LEAK AT TRAIN ─────────────────────────────────────────────────────────

def _written_train(m, inp, lab, layout):
    rec = {}
    real = m.tul.prefix_project

    def spy(h_slots, lay, L, cells=None):
        values, pos = real(h_slots, lay, L, cells=cells)
        rec["v"] = values.detach().clone()
        return values, pos
    m.tul.prefix_project = spy
    try:
        _train(m, inp, lab, layout, seed=3)
    finally:
        m.tul.prefix_project = real
    v = rec["v"]
    S, k = layout.slot_index.shape[1], layout.prefix_k
    return v.reshape(v.shape[0], S, k, *v.shape[2:])


def test_no_posterior_leak_in_the_training_forward():
    from morph.model.tul_layout import slot_layout_from_ids
    ids, inp, lab, layout = _pack()
    B = inp.shape[0]
    row, s = 0, 1
    edited = _edit(ids, layout, row, s + 1)
    inp2, lab2, layout2, _ = slot_layout_from_ids(edited, _rule(), _spec())
    assert torch.equal(layout2.bag_id, layout.bag_id)
    m = _gk(4, token_state_dropout=0.15)
    a = _written_train(m, inp, lab, layout)
    b = _written_train(m, inp2, lab2, layout2)
    assert a.shape[0] == 4 * B
    for k in range(4):
        r = k * B + row
        assert torch.equal(a[r, :s + 1], b[r, :s + 1]), (
            f"LEAK: rollout {k}: an edit to span s+1 moved slot s's cells at train")
        assert not torch.equal(a[r, s + 1:], b[r, s + 1:]), "the edit moved nothing"
    # positive control: the ELBO model's training forward reads span s+1 at slot s
    el = _gram(token_state_dropout=0.15)
    _open_e_next(el)
    c = _written_train(el, inp, lab, layout)
    d = _written_train(el, inp2, lab2, layout2)
    assert not torch.equal(c[row, s], d[row, s]), "the positive control is inert"


# ── the kernel ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("V", [101, 128])
def test_label_logprob_kernel_matches_autograd(V):
    torch.manual_seed(0)
    N, d = 45, 16
    x = torch.randn(N, d, requires_grad=True)
    w = torch.randn(V, d, requires_grad=True)
    lab = torch.randint(0, V, (N,))
    lab[lab == 7] = 8
    lab[[3, 10, 44]] = -100
    g = torch.randn(N)
    lp = fused_linear_label_logprob(x, w, lab, chunk_size=8, mask_token_id=7)
    (lp * g).sum().backward()
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    logits = (x @ w.t()).masked_fill(torch.arange(V) == 7, float("-inf"))
    ref = F.log_softmax(logits, -1).gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    ref = ref * (lab != -100)
    (ref * g).sum().backward()
    assert torch.allclose(lp, ref, atol=2e-6)
    assert torch.count_nonzero(lp[[3, 10, 44]]) == 0
    assert torch.allclose(gx, x.grad, atol=2e-5) and torch.allclose(gw, w.grad, atol=2e-5)
    ce = fused_linear_cross_entropy(x.detach(), w.detach(), lab, chunk_size=8,
                                    mask_token_id=7)
    assert torch.allclose(ce, -lp[lab != -100].mean(), atol=1e-6)


# ── refusals ─────────────────────────────────────────────────────────────────

_IW_REFUSED = [
    dict(gram_beta=1.0), dict(gram_kl_balance=0.5), dict(gram_free_bits=1.0),
    dict(emit_weight=0.5), dict(coda_logit_l2=1e-3), dict(core_token_aux=True),
    dict(grad_pass=True), dict(row_contrast_lambda=0.1), dict(sigreg_lambda=0.1),
    dict(coda_span_heads=2),
]


@pytest.mark.parametrize("bad", _IW_REFUSED, ids=[next(iter(b)) for b in _IW_REFUSED])
def test_iw_refusals(bad):
    kw = dict(prefix_k=2, slot_id=4, gram=True, gram_objective="iw", emit_weight=0.0)
    kw.update(bad)
    with pytest.raises((ValueError, NotImplementedError), match="gram_objective='iw'"):
        TULConfig(**kw)


@pytest.mark.parametrize("bad", [dict(fan_k=4), dict(slot_cells=2), dict(code=True),
                                 dict(coda_sees_slots=False), dict(tokens_through_core=True)],
                         ids=["fan_k", "slot_cells", "code", "gathered", "paid"])
def test_inherited_gram_refusals(bad):
    with pytest.raises((ValueError, NotImplementedError), match="tul.gram"):
        TULConfig(prefix_k=2, slot_id=4, gram=True, gram_objective="iw", emit_weight=0.0,
                  **bad)


def test_knob_validation():
    with pytest.raises(ValueError, match="gram_"):
        TULConfig(prefix_k=2, slot_id=4, gram_objective="iw")
    with pytest.raises(ValueError, match="gram_"):
        TULConfig(prefix_k=2, slot_id=4, gram_iw_k=2)
    with pytest.raises(ValueError, match="gram_objective must be"):
        TULConfig(prefix_k=2, slot_id=4, gram=True, gram_objective="vae")
    with pytest.raises(ValueError, match="gram_iw_k must be >= 1"):
        TULConfig(prefix_k=2, slot_id=4, gram=True, gram_objective="iw", emit_weight=0.0,
                  gram_iw_k=0)
    with pytest.raises(ValueError, match="gram_objective='elbo'"):
        TULConfig(prefix_k=2, slot_id=4, gram=True, gram_iw_k=2)


def test_model_and_runtime_refusals():
    with pytest.raises(NotImplementedError, match="mtp_heads"):
        MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", gram=True,
                                        gram_objective="iw"), mtp_heads=2))
    _ids, inp, lab, layout = _pack()
    el = _gram()
    el.eval()
    with pytest.raises(ValueError, match="gram_objective='iw'"):
        el(inp, labels=lab, slot_layout=layout, gram_mode="iw")
    gk = _gk(4)
    gk.eval()
    with pytest.raises(ValueError, match="no posterior"):
        gk(inp, labels=lab, slot_layout=layout, gram_mode="post")
    with pytest.raises(ValueError, match="needs labels"):
        gk(inp, labels=None, slot_layout=layout, gram_mode="iw")
    with pytest.raises(ValueError, match="needs labels"):
        gk.tul_forward_ablated(inp, lab, layout, plan_mode="zero", gram_mode="iw")
    gk.train()
    with pytest.raises(ValueError, match="EVAL-ONLY"):
        gk(inp, labels=lab, slot_layout=layout, gram_mode="iw")
    with pytest.raises(RuntimeError, match="without one"):
        gk.tul_gram.posterior(torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))


# ── the configs ──────────────────────────────────────────────────────────────

def test_the_two_configs_compose_through_hydra_and_the_runtime():
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import build_tul_runtime
    want = {"tul_slot_spandec_strict_gk4": ("lxtul-gk4", 4),
            "tul_slot_spandec_strict_gk1": ("lxtul-gk1", 1)}
    for name, (wb, k) in want.items():
        with initialize_config_dir(config_dir=os.path.join(_ROOT, "morph", "configs"),
                                   version_base=None):
            cfg = compose(config_name=name)
        assert cfg.wandb.name == wb
        mc = build_tul_runtime(cfg).model_cfg
        assert mc.gram and mc.gram_objective == "iw" and mc.gram_iw_k == k
        assert mc.gram_beta == 0.1 and mc.gram_kl_balance == 0.8 and mc.gram_free_bits == 0.0
        assert mc.tg_geometry == "strict" and mc.spandec and mc.emit_weight == 0.0
        assert mc.plast_weight == 1.0


def test_the_probe_runs_on_a_model_without_a_posterior():
    """`lab/divergence/lxtul_g_probe.py::gram_probe` on a GK model: the posterior readings
    are None (there is no posterior), and the prior-side Bayesian read equals the model's
    own eval `gram_mode="iw"` bound on the same seeds."""
    from lxtul_g_probe import gram_probe
    _ids, inp, lab, layout = _pack()
    m = _gk(4, plast_weight=1.0)
    res, arr = gram_probe(m, [(inp, lab, layout, None)], "cpu", n_list=(1, 4), seed=3)
    assert res["has_posterior"] is False
    for k in ("ce_post", "ce_elbo", "kl_per_token", "exposure_gap"):
        assert res[k] is None, k
    assert "post" not in arr
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout, gram_mode="iw", gram_sample_seed=3)
    assert abs(res["ce_iw@4"] - float(o["loss"])) < 2e-5, (res["ce_iw@4"], float(o["loss"]))


def test_rollouts_share_every_dropout_mask():
    """Model dropout ON (0.3) and every rollout on the SAME noise seed: the K rollouts are
    then the same function, so the coda readout's K row blocks must be bit-identical and
    the width gain 0 to fp32 rounding. With per-row nn.Dropout masks in the core and coda they are
    not (the 2026-09-23 defect this guards)."""
    _ids, inp, lab, layout = _pack()
    B, K = inp.shape[0], 4
    m = _gk(K, token_state_dropout=0.15)
    m2 = _gk(K, token_state_dropout=0.15, core_kw={"dropout": 0.3})
    from morph.model.rollout_dropout import RolloutSharedDropout
    n_shared = sum(isinstance(x, RolloutSharedDropout) for x in m2.modules())
    assert n_shared == m2._n_rollout_dropout > 0
    assert not any(isinstance(x, RolloutSharedDropout) for x in m2.prelude.modules())
    assert m._n_rollout_dropout == 0            # dropout 0: nothing to share
    real, seen = m2._gram_ctx, {}

    def patched(xn, lay, gram_mode, gram_seed, n_nograd, iw_rollouts=1):
        ctx = real(xn, lay, gram_mode, gram_seed, n_nograd, iw_rollouts)
        ctx["gen"] = torch.Generator(device=xn.device)
        ctx["iw_seeds"] = [7] * iw_rollouts
        return ctx
    m2._gram_ctx = patched
    real_loss = m2._gram_iw_losses

    def cap(xh, labels, lay, K_):
        seen["xh"] = xh.detach().clone()
        return real_loss(xh, labels, lay, K_)
    m2._gram_iw_losses = cap
    out = _train(m2, inp, lab, layout, seed=5)
    xh = seen["xh"].reshape(K, B, *seen["xh"].shape[1:])
    for k in range(1, K):
        assert torch.equal(xh[0], xh[k]), f"rollout {k} differs from rollout 0"
    assert abs(float(out["gk_width_gain"])) < 1e-5     # fp32 logsumexp - log K


def test_gk_training_forward_without_labels_raises():
    _ids, inp, lab, layout = _pack()
    m = _gk(4, core_kw={"dropout": 0.1})
    m.train()
    with pytest.raises(RuntimeError, match="needs labels"):
        m(inp, labels=None, slot_layout=layout)
    m.eval()
    with torch.no_grad():
        m(inp, labels=None, slot_layout=layout)
