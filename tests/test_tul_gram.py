"""`tul.gram` — LXTUL-G, the stochastic slot loop (morph/model/tul_gram.py).

Contracts, one test (or one parametrised family) each:

  * OFF is bit-identical: the knob absent and `gram=False` give the same loss, logits,
    parameters and gradients, and a gram model shares every base weight with its
    same-seed ruler (the heads draw from a private generator).
  * INIT: the posterior equals the prior bit for bit and the KL is exactly 0; after one
    optimizer step with real next spans, the KL is > 0 and the two heads differ.
  * EVAL draws the PRIOR, deterministically under the fixed seed, differently under
    another `gram_sample_seed`, never reads the next span, and reuses the same noise at
    pass t under every forced depth.
  * The fixed-point term reads u_T, the deterministic part: at depth 1 it sends NO
    gradient into the sigma heads, and at depth 3 its value is ||u_T - h_{T-1}||^2 /
    ||u_T||^2 from the captured passes (and not the h_T version).
  * mean-free builds no m heads and its mu is 0; d1 composes and trains.
  * every refusal raises, naming `tul.gram`.
  * THE POSTERIOR LEAK GATE: in prior mode an edit to span s+1 leaves slot s's written
    cells bit-exact; in post mode (with the e_next path opened) it moves them. And the
    strict geometry's own leak test still holds on a gram model in prior mode.
  * the probe (lab/divergence/lxtul_g_probe.py): the identity null, ce_iw@1 == ce_prior@1,
    the Bayesian read's multi-sample-bound identity and its causality.

CPU, fp32, the `tests/test_tul_strict_geometry.py` fixtures.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from test_tul_strict_geometry import _leak, _model, _pack, _tiny, _tul, _edit

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAB = os.path.join(_ROOT, "lab", "divergence")
if _LAB not in sys.path:
    sys.path.insert(0, _LAB)


def _gram(seed: int = 1234, core_kw: dict | None = None, **tul_kw) -> MORPHTransformer:
    """A strict gram model on the tiny fixture (fp32, eval mode)."""
    kw = dict(tg_geometry="strict", gram=True)
    kw.update(tul_kw)
    if core_kw:
        torch.manual_seed(seed)
        m = MORPHTransformer(_tiny(tul=_tul(**kw), **core_kw))
        with torch.no_grad():
            m.embed.bigram.lambdas.fill_(0.5)
        return m.eval().float()
    return _model(seed=seed, **kw)


def _open_e_next(m: MORPHTransformer, scale: float = 0.5) -> None:
    """Give the posterior's zero-init e_next path (and the zero-init output layers) values,
    as training would. At init the posterior does not read e_next at all, so a leak test
    run on a fresh model would pass vacuously."""
    g = torch.Generator().manual_seed(7)
    heads = [m.tul_gram.post_s] + ([m.tul_gram.post_m] if m.tul_gram.post_m is not None else [])
    with torch.no_grad():
        for h in heads:
            for lin in (h.x_gate, h.x_up, h.w_down):
                lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * scale
                                 / lin.weight.shape[1] ** 0.5)


def _train_out(m, inp, lab, layout):
    m.train()
    torch.manual_seed(0)
    return m(inp, labels=lab, slot_layout=layout)


def _eval(m, inp, layout, **kw):
    m.eval()
    with torch.no_grad():
        return m(inp, labels=None, slot_layout=layout, **kw)


# ── OFF is bit-identical ─────────────────────────────────────────────────────

def test_off_is_bit_identical_loss_logits_params_grads():
    _ids, inp, lab, layout = _pack()
    a = _model(tg_geometry="strict")
    b = _model(tg_geometry="strict", gram=False)
    assert a.tul_gram is None and b.tul_gram is None
    for k, (x, y) in enumerate(zip(a.state_dict().values(), b.state_dict().values())):
        assert torch.equal(x, y), f"param {k} differs"
    oa, ob = _train_out(a, inp, lab, layout), _train_out(b, inp, lab, layout)
    assert torch.equal(oa["loss"], ob["loss"])
    oa["loss"].backward()
    ob["loss"].backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert (pa.grad is None) == (pb.grad is None), na
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad), f"grad {na} differs"
    assert torch.equal(_eval(a, inp, layout)["logits"], _eval(b, inp, layout)["logits"])
    assert not any(k.startswith("gram_") for k in oa)


def test_gram_shares_every_base_weight_with_its_ruler():
    r = _model(tg_geometry="strict")
    g = _gram()
    gs = g.state_dict()
    rs = r.state_dict()
    extra = [k for k in gs if k not in rs]
    assert extra and all(k.startswith("tul_gram.") for k in extra)
    assert all(k in gs for k in rs)
    for k, v in rs.items():
        assert torch.equal(v, gs[k]), f"{k}: the gram build moved the global RNG"
    # and the global stream after the build is the ruler's
    torch.manual_seed(99)
    MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict")))
    a = torch.rand(4)
    torch.manual_seed(99)
    MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", gram=True)))
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_every_gram_leaf_is_ternary_excluded():
    g = _gram()
    lins = [m for m in g.tul_gram.modules() if isinstance(m, torch.nn.Linear)]
    assert lins and all(getattr(m, "_ternary_exclude", False) for m in lins)
    from morph.model.ternary_qat import _categorize
    for name, mod in g.named_modules():
        if name.startswith("tul_gram"):
            assert _categorize(name, mod, set()) is None, name


# ── INIT and one step ────────────────────────────────────────────────────────

def test_kl_is_exactly_zero_and_posterior_equals_prior_at_init():
    _ids, inp, lab, layout = _pack()
    m = _gram()
    out = _train_out(m, inp, lab, layout)
    assert float(out["gram_kl"]) == 0.0
    assert float(out["gram_kl_weighted"]) == 0.0
    assert float(out["gram_n_slots"]) > 0
    assert abs(float(out["gram_sigma_ratio_prior"]) - 0.1) < 1e-6
    torch.manual_seed(3)
    uh = torch.randn(2, 5, 64)
    e = torch.randn(2, 5, 64)
    mp, sp = m.tul_gram.prior(uh)
    mq, sq = m.tul_gram.posterior(uh, e)
    assert torch.equal(mp, mq) and torch.equal(sp, sq)
    # The output layers are zero at init, so the two lines above would hold even if the
    # hidden layers were NOT copied. Give prior and posterior the SAME nonzero output layer:
    # q == p must still hold bit for bit (the copy covers the hidden layers), and the
    # e_next path must still be exactly 0.
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for pri, post in ((m.tul_gram.prior_s, m.tul_gram.post_s),
                          (m.tul_gram.prior_m, m.tul_gram.post_m)):
            w = torch.randn(pri.w_down.weight.shape, generator=g) * 0.1
            pri.w_down.weight.copy_(w)
            post.w_down.weight.copy_(w)
    mp, sp = m.tul_gram.prior(uh)
    mq, sq = m.tul_gram.posterior(uh, e)
    assert torch.equal(mp, mq) and torch.equal(sp, sq)
    assert mp.abs().sum() > 0 and float(sp.std()) > 0


def test_one_optimizer_step_opens_the_kl():
    _ids, inp, lab, layout = _pack()
    m = _gram()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    out = _train_out(m, inp, lab, layout)
    out["loss"].backward()
    opt.step()
    opt.zero_grad()
    out2 = _train_out(m, inp, lab, layout)
    assert float(out2["gram_kl"]) > 0.0
    assert float(out2["gram_kl_weighted"]) > 0.0
    assert torch.isfinite(out2["loss"])
    torch.manual_seed(3)
    uh, e = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
    mp, sp = m.tul_gram.prior(uh)
    mq, sq = m.tul_gram.posterior(uh, e)
    assert not (torch.equal(mp, mq) and torch.equal(sp, sq))
    # the loss carries exactly beta * KL / n_tokens on top of everything else
    n_tok = ((~layout.slot_mask) & (lab >= 0)).sum().float()
    assert torch.allclose(out2["gram_kl_weighted"],
                          0.1 * out2["gram_kl_per_token"], rtol=1e-6)
    assert float(out2["gram_kl_per_token"]) > 0 and n_tok > 0


def test_kl_balancing_splits_the_gradient():
    """``a * KL(sg(q) || p) + (1 - a) * KL(q || sg(p))``: value = KL, the prior side gets
    fraction ``a`` of the plain KL's gradient and the posterior side ``1 - a``."""
    from morph.model.tul_gram import gram_kl, gram_kl_balanced
    torch.manual_seed(0)
    mq, sq, mp, sp = (torch.randn(6), torch.rand(6) + 0.1, torch.randn(6),
                      torch.rand(6) + 0.1)
    ts = [t.clone().requires_grad_(True) for t in (mq, sq, mp, sp)]
    bal, raw = gram_kl_balanced(*ts, alpha=0.8)
    assert torch.allclose(bal, raw) and torch.allclose(raw, gram_kl(mq, sq, mp, sp))
    g_bal = torch.autograd.grad(bal.sum(), ts)
    ts2 = [t.clone().requires_grad_(True) for t in (mq, sq, mp, sp)]
    g_raw = torch.autograd.grad(gram_kl(*ts2).sum(), ts2)
    for i, w in enumerate((0.2, 0.2, 0.8, 0.8)):
        assert torch.allclose(g_bal[i], w * g_raw[i], rtol=1e-5, atol=1e-7)


def test_kl_is_accurate_when_q_is_close_to_p():
    """fp32 KL against the fp64 textbook form, from tiny to large differences. The
    textbook form in fp32 reads a NEGATIVE sum at a 1e-6 log-sigma gap (cancellation)."""
    from morph.model.tul_gram import gram_kl
    torch.manual_seed(0)
    for scale in (1e-6, 1e-4, 1e-2, 0.3):
        sp = torch.rand(4000, dtype=torch.float64) + 0.05
        sq = sp * torch.exp(scale * torch.randn(4000, dtype=torch.float64))
        mp = torch.randn(4000, dtype=torch.float64)
        mq = mp + scale * torch.randn(4000, dtype=torch.float64)
        ref = (torch.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5).sum()
        got = gram_kl(mq.float(), sq.float(), mp.float(), sp.float()).double().sum()
        assert float(got) >= 0.0
        assert abs(float(got - ref)) <= 0.05 * float(ref) + 1e-9, (scale, float(got), float(ref))


def test_sigma_head_resolves_small_updates_under_bf16_autocast():
    """The sigma head's output layer runs in fp32: a 1e-4 change of its bias must move
    sigma under bf16 autocast (the bf16 spacing at the init bias -2.25 is 0.0156)."""
    m = _gram()
    torch.manual_seed(0)
    uh = torch.randn(2, 5, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, s0 = m.tul_gram.prior(uh)
    assert s0.dtype == torch.float32 and abs(float(s0.mean()) - 0.1) < 1e-6
    with torch.no_grad():
        m.tul_gram.prior_s.w_down.bias.add_(1e-4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, s1 = m.tul_gram.prior(uh)
    assert float((s1 - s0).abs().min()) > 0


def test_model_reads_the_balance_knob():
    """At ``gram_kl_balance: 1.0`` the KL term trains the PRIOR only: its gradient into the
    posterior heads is exactly 0; at 0.8 it is not."""
    _ids, inp, lab, layout = _pack()
    for alpha, zero in ((1.0, True), (0.8, False)):
        m = _gram(gram_kl_balance=alpha)
        _open_e_next(m)
        out = _train_out(m, inp, lab, layout)
        assert float(out["gram_kl"]) > 0
        post = list(m.tul_gram.post_s.parameters()) + list(m.tul_gram.post_m.parameters())
        # The KL term's own gradient is the loss gradient at beta 0.1 minus the SAME
        # forward's at beta 0 (beta only scales the term, so the CE part is bit-identical).
        grads = torch.autograd.grad(out["loss"], post, allow_unused=True, retain_graph=True)
        m2 = _gram(gram_kl_balance=alpha, gram_beta=0.0)
        _open_e_next(m2)
        out2 = _train_out(m2, inp, lab, layout)
        grads2 = torch.autograd.grad(out2["loss"], list(m2.tul_gram.post_s.parameters())
                                     + list(m2.tul_gram.post_m.parameters()),
                                     allow_unused=True)
        diff = sum(float((a - b).abs().sum()) for a, b in zip(grads, grads2)
                   if a is not None and b is not None)
        assert (diff == 0.0) is zero, f"alpha={alpha}: KL grad into posterior {diff}"


def test_kl_is_masked_to_slots_with_a_next_span_and_their_realised_passes():
    """The row's LAST valid slot precedes the partial tail and every pad has no next span:
    their KL is masked out, at every pass. A slot whose realised depth does not reach pass
    t contributes nothing at t."""
    from morph.model.tul_gram import gram_has_next
    _ids, inp, lab, layout = _pack()
    m = _gram()
    _open_e_next(m)
    has = gram_has_next(layout)
    assert bool(has.any()) and bool((layout.slot_valid & ~has).any())
    out = _eval(m, inp, layout, gram_mode="post")
    kp, km = out["gram_kl_pass"], out["gram_kl_mask"]
    T = kp.shape[-1]
    assert torch.equal(km, has.unsqueeze(-1).expand(-1, -1, T))
    assert torch.count_nonzero(kp[~km]) == 0 and bool((kp[km] > 0).all())
    # training: a per-slot Poisson depth, and the per-pass KL mean is over the slots that
    # REACH the pass
    tr = _train_out(m, inp, lab, layout)
    assert float(tr["gram_n_slots"]) == float(has.sum())
    # a forced mixed-depth table at eval: slot 1 of row 0 stops after pass 1
    d = torch.full(layout.slot_index.shape, 3, dtype=torch.long)
    d[0, 1] = 1
    m.eval()
    with torch.no_grad():
        o = m.tul_forward_ablated(inp, None, layout, slot_depths=d, gram_mode="post")
    assert bool(o["gram_kl_mask"][0, 1, 0]) and not bool(o["gram_kl_mask"][0, 1, 1:].any())


# ── EVAL: prior, seeded, paired across depths ────────────────────────────────

def test_eval_uses_the_prior_and_never_reads_the_next_span():
    _ids, inp, _lab, layout = _pack()
    m = _gram()
    m._gram_capture = []

    def _boom(*a, **k):
        raise AssertionError("the posterior pool ran in prior mode")
    real = m.tul_gram.pool.forward
    m.tul_gram.pool.forward = _boom
    try:
        o1 = _eval(m, inp, layout)
        o2 = _eval(m, inp, layout, gram_mode="mean")
    finally:
        m.tul_gram.pool.forward = real
    assert {c["mode"] for c in m._gram_capture} == {"prior", "mean"}
    assert o1["gram_mode"] == "prior" and "gram_kl_pass" not in o1
    assert o2["gram_mode"] == "mean"
    o3 = _eval(m, inp, layout, gram_mode="post")
    assert o3["gram_kl_pass"].shape[:2] == layout.slot_valid.shape


def test_eval_is_deterministic_under_the_seed_and_moves_with_another_seed():
    _ids, inp, _lab, layout = _pack()
    m = _gram()
    a = _eval(m, inp, layout)["logits"]
    b = _eval(m, inp, layout)["logits"]
    c = _eval(m, inp, layout, gram_sample_seed=5)["logits"]
    d = _eval(m, inp, layout, gram_sample_seed=0)["logits"]      # == gram_eval_seed
    assert torch.equal(a, b)
    assert torch.equal(a, d)
    assert not torch.equal(a, c)
    m.gram_eval_seed = 5
    assert torch.equal(_eval(m, inp, layout)["logits"], c)


def test_forced_depths_reuse_the_same_noise_per_pass():
    _ids, inp, _lab, layout = _pack()
    m = _gram()
    tc = m.cfg.tul
    caps = {}
    for d in (1, 6):
        tc.slot_mean_depth, tc.slot_max_depth = d, 6
        m._gram_capture = []
        _eval(m, inp, layout)
        caps[d] = m._gram_capture
    m._gram_capture = None
    assert len(caps[1]) == 1 and len(caps[6]) == 6
    assert torch.equal(caps[1][0]["u"], caps[6][0]["u"])
    assert torch.equal(caps[1][0]["eps"], caps[6][0]["eps"])
    assert torch.equal(caps[1][0]["h"], caps[6][0]["h"])
    assert caps[1][0]["eps"].abs().sum() > 0
    # pass 2 draws a DIFFERENT noise from pass 1 (the per-pass seed moves)
    n1 = caps[6][0]["eps"] / caps[6][0]["u"].pow(2).mean((-1, -2), keepdim=True).sqrt()[..., 0, :]
    n2 = caps[6][1]["eps"] / caps[6][1]["u"].pow(2).mean((-1, -2), keepdim=True).sqrt()[..., 0, :]
    assert not torch.allclose(n1, n2)


def test_eval_mode_arguments_are_refused_at_train_and_on_a_non_gram_model():
    _ids, inp, lab, layout = _pack()
    m = _gram()
    m.train()
    with pytest.raises(ValueError, match="EVAL-ONLY"):
        m(inp, labels=lab, slot_layout=layout, gram_mode="prior")
    with pytest.raises(ValueError, match="EVAL-ONLY"):
        m(inp, labels=lab, slot_layout=layout, gram_sample_seed=3)
    r = _model(tg_geometry="strict")
    with pytest.raises(ValueError, match="tul.gram=true"):
        _eval(r, inp, layout, gram_mode="prior")
    with pytest.raises(ValueError, match="gram_mode must be"):
        _eval(m, inp, layout, gram_mode="posterior")


# ── the fixed-point term reads u_T ───────────────────────────────────────────

def test_fixed_point_term_sends_no_gradient_into_sigma_at_depth_1():
    _ids, inp, lab, layout = _pack()
    m = _gram(core_kw=dict(core_fixed_point_lambda=1.0), slot_depth_fixed=1)
    out = _train_out(m, inp, lab, layout)
    fp = out["fp_weighted"]
    assert float(fp) > 0, "the fixed-point term is not on — the test would be vacuous"
    s_params = (list(m.tul_gram.prior_s.parameters())
                + list(m.tul_gram.post_s.parameters()))
    grads = torch.autograd.grad(fp, s_params, allow_unused=True, retain_graph=True)
    for g in grads:
        assert g is None or torch.count_nonzero(g) == 0
    # two-sided: the CE DOES reach the sigma heads (so "no gradient" above is not a dead
    # parameter)
    g_ce = torch.autograd.grad(out["ce_main"], s_params, allow_unused=True)
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in g_ce)


def test_fixed_point_value_is_computed_on_u_T_not_h_T():
    _ids, inp, lab, layout = _pack()
    m = _gram(core_kw=dict(core_fixed_point_lambda=1.0), slot_depth_fixed=3)
    m._gram_capture = []
    out = _train_out(m, inp, lab, layout)
    cap = m._gram_capture
    m._gram_capture = None
    assert len(cap) == 3
    v = layout.slot_valid
    uT = cap[2]["u"].flatten(2).float()[v]
    hT = cap[2]["h"].flatten(2).float()[v]
    hTm1 = cap[1]["h"].flatten(2).float()[v]
    on_u = ((uT - hTm1).pow(2).sum(-1) / (uT.pow(2).sum(-1) + 1e-6)).mean()
    on_h = ((hT - hTm1).pow(2).sum(-1) / (hT.pow(2).sum(-1) + 1e-6)).mean()
    assert torch.allclose(out["fixed_point"], on_u, rtol=1e-5, atol=0)
    assert not torch.allclose(out["fixed_point"], on_h, rtol=1e-3, atol=0)


def test_gain_hinge_probes_f_alone():
    """The hinge re-applies the core step at the pass's operating point; the Gaussian step
    of that pass is added AFTER the core step returns. So the hinge's unperturbed
    application f(h) must return exactly the loop's u_t, never u_t + eps. Recorded on
    `_apply_core_step`: the two calls that share one input (the loop's, then the hinge's
    f0) must both output the captured u_t, and that u_t must differ from the captured h_t
    (else the check is vacuous)."""
    _ids, inp, lab, layout = _pack()
    m = _gram(core_kw=dict(slot_gain_lambda=1.0, slot_gain_all_iters=True),
              slot_depth_fixed=2)
    calls = []
    real = m._apply_core_step

    def rec(h_in, *a, **k):
        out = real(h_in, *a, **k)
        calls.append((h_in.detach().clone(), out[0].detach().clone()))
        return out
    m._apply_core_step = rec
    m._gram_capture = []
    out = _train_out(m, inp, lab, layout)
    cap = m._gram_capture
    m._gram_capture = None
    m._apply_core_step = real
    assert "gain_est" in out and len(cap) == 2
    n_checked = 0
    for c in cap:
        same = [o for (i, o) in calls
                if i.shape == c["u"].shape and torch.equal(o, c["u"])]
        # the loop's own call AND the hinge's f0 (same input, same map) both give u_t
        assert len(same) >= 2, f"pass {c['t']}: the hinge did not read f at the loop's point"
        assert not torch.equal(c["u"], c["h"])
        assert not any(torch.equal(o, c["h"]) for (_i, o) in calls)
        n_checked += 1
    assert n_checked == 2


# ── mean-free and d1 ─────────────────────────────────────────────────────────

def test_meanfree_builds_no_mean_heads_and_mu_is_zero():
    _ids, inp, lab, layout = _pack()
    m = _gram(gram_mean=False)
    assert m.tul_gram.prior_m is None and m.tul_gram.post_m is None
    assert not any("prior_m" in k or "post_m" in k for k in m.state_dict())
    out = _train_out(m, inp, lab, layout)
    assert float(out["gram_mu_ratio_prior"]) == 0.0
    assert float(out["gram_mu_ratio_post"]) == 0.0
    m._gram_capture = []
    _eval(m, inp, layout, gram_mode="mean")
    assert all(torch.count_nonzero(c["eps"]) == 0 for c in m._gram_capture)
    m._gram_capture = None


def test_d1_trains_a_step():
    _ids, inp, lab, layout = _pack()
    m = _gram(slot_depth_fixed=1)
    out = _train_out(m, inp, lab, layout)
    assert torch.isfinite(out["loss"])
    assert "gram_kl_t1" in out and "gram_kl_t2" not in out
    out["loss"].backward()


def test_the_three_configs_compose_through_hydra_and_the_runtime():
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import build_tul_runtime
    want = {"tul_slot_spandec_strict_gram": ("lxtul-g", True, 0),
            "tul_slot_spandec_strict_gram_meanfree": ("lxtul-g-meanfree", False, 0),
            "tul_slot_spandec_strict_gram_d1": ("lxtul-g-d1", True, 1)}
    for name, (wb, mean, fixed) in want.items():
        with initialize_config_dir(config_dir=os.path.join(_ROOT, "morph", "configs"),
                                   version_base=None):
            cfg = compose(config_name=name)
        assert cfg.wandb.name == wb
        mc = build_tul_runtime(cfg).model_cfg
        assert mc.gram and mc.gram_beta == 0.1 and mc.gram_kl_balance == 0.8
        assert mc.gram_free_bits == 0.0 and mc.gram_sigma_init == 0.1
        assert mc.gram_mean is mean and mc.slot_depth_fixed == fixed
        assert mc.tg_geometry == "strict" and mc.spandec


# ── refusals ─────────────────────────────────────────────────────────────────

_REFUSED = [
    dict(tokens_through_core=True), dict(loop_reads_tokens=True), dict(code=True),
    dict(code_target=True), dict(loop_denoise=True), dict(fan_k=4), dict(slot_cells=2),
    dict(vq_codes=2), dict(prefix_source="trajectory"), dict(pass_readout="gated"),
    dict(core_stage_cond="sigma"), dict(db_loop=True), dict(progressive_p=0.5),
    dict(detach_z=True), dict(coda_sees_slots=False), dict(coda_token_cut=8),
]


@pytest.mark.parametrize("bad", _REFUSED, ids=[next(iter(b)) for b in _REFUSED])
def test_tul_side_refusals(bad):
    with pytest.raises((ValueError, NotImplementedError), match="tul.gram"):
        TULConfig(prefix_k=2, slot_id=4, gram=True, **bad)


def test_gram_knobs_without_gram_raise():
    with pytest.raises(ValueError, match="gram_"):
        TULConfig(prefix_k=2, slot_id=4, gram_beta=0.2)
    for bad in (dict(gram_beta=-1.0), dict(gram_kl_balance=1.5), dict(gram_free_bits=-1.0),
                dict(gram_sigma_init=0.0), dict(gram_hidden=0)):
        with pytest.raises(ValueError):
            TULConfig(prefix_k=2, slot_id=4, gram=True, **bad)


def test_model_side_refusals():
    with pytest.raises(NotImplementedError, match="tul.gram under SCSE"):
        MORPHTransformer(_tiny(tul=_tul(gram=True), scse_enabled=True))
    with pytest.raises(ValueError, match="tul.gram needs a core loop"):
        MORPHTransformer(_tiny(tul=_tul(gram=True), n_core=0))
    from morph.model.tul_fm import FMArmConfig
    with pytest.raises(NotImplementedError, match="tul.gram with an FM planner"):
        MORPHTransformer(_tiny(tul=TULConfig(prefix_k=2, slot_id=4, gram=True,
                                             emit_weight=0.0, token_state_dropout=0.0,
                                             mux_beta=0.0),
                               fm=FMArmConfig(d_p=16, n_layers=1, n_heads=2, d_ff=32,
                                              cond_dim=16, max_slots=10, l_total=84)))


def test_runtime_refusals():
    _ids, inp, lab, layout = _pack()
    m = _gram(core_kw=dict(bptt_depth=1))
    m.train()
    with pytest.raises(NotImplementedError, match="truncated BPTT"):
        m(inp, labels=lab, slot_layout=layout)
    m2 = _gram()
    m2.train()
    with pytest.raises(NotImplementedError, match="db1"):
        m2(inp, labels=lab, slot_layout=layout, tul_step_mode="db1")
    with pytest.raises(ValueError, match="require slot_layout"):
        _model()(inp, gram_mode="prior")


# ── THE POSTERIOR LEAK GATE ──────────────────────────────────────────────────

def _written(m, inp, layout, **kw):
    """The values `prefix_project` writes into the coda, ``[B, S, K, ...]``."""
    rec = {}
    real = m.tul.prefix_project

    def spy(h_slots, lay, L, cells=None):
        values, pos = real(h_slots, lay, L, cells=cells)
        rec["v"] = values.detach().clone()
        return values, pos
    m.tul.prefix_project = spy
    try:
        _eval(m, inp, layout, **kw)
    finally:
        m.tul.prefix_project = real
    B, S = layout.slot_index.shape
    return rec["v"].reshape(B, S, layout.prefix_k, *rec["v"].shape[2:])


@pytest.mark.parametrize("mean", [True, False])
def test_posterior_leak_gate(mean):
    from test_tul_strict_geometry import _rule, _spec
    from morph.model.tul_layout import slot_layout_from_ids
    ids, inp, _lab, layout = _pack()
    m = _gram(gram_mean=mean)
    _open_e_next(m)
    row, s = 0, 1
    assert bool(layout.slot_valid[row, s + 1]), "fixture: span s+1 must be a full span"
    edited = _edit(ids, layout, row, s + 1)
    inp2, _l2, layout2, _ = slot_layout_from_ids(edited, _rule(), _spec())
    assert torch.equal(layout2.bag_id, layout.bag_id), "the edit moved a cut"
    assert not torch.equal(inp2, inp)
    # PRIOR mode: slot s (and every earlier slot) is bit-exact
    a = _written(m, inp, layout)
    b = _written(m, inp2, layout2)
    assert torch.equal(a[row, :s + 1], b[row, :s + 1]), (
        "LEAK: in prior mode an edit to span s+1 moved slot s's written cells")
    assert not torch.equal(a[row, s + 1:], b[row, s + 1:]), \
        "the edit moved no later slot either — the fixture is inert"
    # POST mode: the posterior reads span s+1, so slot s's cells MUST move (positive
    # control; the same seed, so only e_next differs)
    c = _written(m, inp, layout, gram_mode="post")
    d = _written(m, inp2, layout2, gram_mode="post")
    assert torch.equal(c[row, :s], d[row, :s]), "post mode: an edit to span s+1 moved slot < s"
    assert not torch.equal(c[row, s], d[row, s]), (
        "post mode did NOT move slot s: the posterior is not reading the next span, so "
        "the prior-mode assertion above proves nothing")


def test_strict_leak_test_still_holds_under_gram_in_prior_mode():
    ids, *_ = _pack()
    m = _gram()
    _open_e_next(m)
    out_d, own_d = _leak(m, ids)
    assert own_d > 0 and out_d == 0.0, f"outside delta {out_d:.3e}"
    out_n, own_n = _leak(m, ids, plan_mode="normal")
    assert own_n > 0 and out_n > 0, "with the write ON the loop must carry the edit"


# ── the probe ────────────────────────────────────────────────────────────────

def test_bayes_read_equals_the_multi_sample_bound_and_is_causal():
    from lxtul_g_probe import bayes_read
    rng = np.random.default_rng(0)
    lp = np.log(rng.uniform(0.01, 1.0, size=(5, 12)))
    bag = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3])
    out = bayes_read(lp, bag)
    for b in np.unique(bag):
        sel = bag == b
        bound = np.log(np.mean(np.exp(lp[:, sel].sum(1))))
        assert abs(out[sel].sum() - bound) < 1e-10
    lp2 = lp.copy()
    lp2[:, 5] += rng.normal(size=5)
    out2 = bayes_read(lp2, bag)
    assert np.array_equal(out[:5], out2[:5]) and np.array_equal(out[7:], out2[7:])
    assert np.array_equal(bayes_read(np.repeat(lp[:1], 4, 0), bag), lp[0])
    assert np.array_equal(bayes_read(lp[:1], bag), lp[0])


def test_probe_core_on_the_fixture():
    from lxtul_g_probe import gram_probe
    _ids, inp, lab, layout = _pack()
    m = _gram()
    res, arr = gram_probe(m, [(inp, lab, layout, None)], "cpu", n_list=(1, 4),
                          depths=[1, 3])
    assert res["ce_iw@1"] == res["ce_prior@1"]
    assert res["ce_iw_identity@4"] == res["ce_prior@1"]
    assert res["ce_iw@4"] <= res["ce_prior@1"] + 0.5      # finite and sane
    # at init q == p and the post draw uses the same seed: identical to the prior sample,
    # and the ELBO equals ce_post (KL exactly 0)
    assert res["ce_post"] == res["ce_prior@1"] and res["ce_elbo"] == res["ce_post"]
    assert res["ce_zero"] != res["ce_prior@1"]
    assert set(res["depths"]) == {"1", "3"}
    assert arr["iw4"].shape == arr["prior1"].shape == arr["_tok_index"].shape
    # a trained-looking posterior: the KL is positive and the ELBO sits above ce_post
    _open_e_next(m)
    res2, _ = gram_probe(m, [(inp, lab, layout, None)], "cpu", n_list=(1,))
    assert res2["kl_per_token"] > 0 and res2["ce_elbo"] > res2["ce_post"]
    assert res2["kl_pass"]["1"]["n_slots"] > 0
