"""Gradient-conditioned passes on the slot loop (``tul.grad_pass``).

Marino, Yue & Mandt 2018 (Iterative Amortized Inference) and Greff et al. 2019 (IODINE):
an iterative inference network is handed, at every step, the gradient of its own objective
with respect to the state it is refining. Here that objective is LOCAL and CAUSAL — the
slot's OWN span through the tied head (``_tul_mux_loss(target="own")``) — and the exit
target is unchanged (the ordinary M-next forecast MUX). The loop's loss is not touched at
all: this knob adds an INPUT.

One test per contract:

  1. ``grad_pass = False`` is the forward from before the knob existed: same loss, same
     gradient on every parameter, no module built, and a sensitivity test so the check
     cannot pass vacuously.
  2. zero-init identity: with the knob ON and ``W_g`` at its zero init, the loss and every
     other parameter's gradient are the off model's — and ``W_g`` still receives a nonzero
     gradient, so it escapes zero on the first step.
  3. fixture sensitivity on the mechanism: with ``W_g`` randomised the state after pass 1
     moves, and the injected term is exactly proportional to ``tul.grad_pass_scale``.
  4. the feature IS the gradient: what pass ``t`` injects equals ``autograd.grad`` of the
     own-span loss at that pass's state, recomputed by hand.
  5. the feature is DETACHED: the injected gradient carries no ``grad_fn``, so no outer
     gradient (and no second-order term) flows through the inner backward.
  6. refusals, including the one that keeps a caller from silently losing the mechanism.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULGradPass

MAX_DEPTH = 3


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False, **tul_kw)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1,
                core_fixed_point_lambda=1.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, seed: int = 7):
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    return out, x, layout


def _grads(m: MORPHTransformer, skip: str = "tul_grad_pass"):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters() if not n.startswith(skip)}


def _same(ga, gb) -> bool:
    return set(ga) == set(gb) and all(
        ((ga[k] is None) == (gb[k] is None))
        and (ga[k] is None or torch.equal(ga[k], gb[k])) for k in ga)


def _feature_spy(m: MORPHTransformer) -> list[dict]:
    """Record, per pass: the state the feature was taken at, the mask, the raw gradient
    the model computed, and the term it injected."""
    seen: list[dict] = []
    real_grad, real_map = m._own_span_grad, m.tul_grad_pass.forward

    def grad(h, input_ids, layout, mask):
        g, own = real_grad(h, input_ids, layout, mask)
        seen.append({"h": h.detach().clone(), "mask": mask.detach().clone(), "g": g,
                     "own": own})
        return g, own

    def wg(g, slot_valid):
        term = real_map(g, slot_valid)
        seen[-1]["term"] = term
        return term

    m._own_span_grad = grad
    m.tul_grad_pass.forward = wg
    return seen


def _fixed_depths(m: MORPHTransformer, depth: int):
    def draw(layout, device):
        d = torch.full(layout.slot_valid.shape, depth, dtype=torch.long, device=device)
        return torch.where(layout.slot_valid, d, torch.ones_like(d))

    m._sample_slot_depths = draw


def _randomise_wg(m: MORPHTransformer, seed: int = 11, std: float = 0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.tul_grad_pass.W_g.copy_(
            torch.empty_like(m.tul_grad_pass.W_g).normal_(0.0, std, generator=g))


# ── 1. off is the forward from before the knob ───────────────────────────────

def test_off_is_bit_identical_loss_and_every_gradient():
    m_a = _model()
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    rng_a = torch.get_rng_state()

    m_b = _model(tul_grad_pass=False)
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()

    assert m_a.tul_grad_pass is None and m_b.tul_grad_pass is None
    assert torch.equal(rng_a, torch.get_rng_state()), "the off path consumed random numbers"
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert _same(_grads(m_a), _grads(m_b)), "the off path changed a gradient"


def test_the_check_is_sensitive_on_changes_the_loss_and_the_gradients():
    """Guards the test above and the zero-init test below: were the mechanism a no-op it
    would pass both for the wrong reason. `W_g` at its zero INIT is deliberately inert, so
    the sensitivity fixture is the same model with `W_g` moved off zero — which is where
    the first optimiser step puts it."""
    m_a = _model(tul_grad_pass=False)
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()

    m_b = _model(tul_grad_pass=True)
    _randomise_wg(m_b)
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()

    assert not torch.equal(out_a["loss"], out_b["loss"])
    assert not _same(_grads(m_a), _grads(m_b))


# ── 2. zero-init identity, and the escape from zero ──────────────────────────

def test_zero_init_is_the_ruler_and_W_g_still_gets_a_gradient():
    """The arm starts as the ruler, bit for bit — `W_g(g) == 0` exactly, and `h + 0.0 == h`
    for every float. It does not STAY the ruler: `dL/dW_g = (dL/dh_in) (x) g` does not
    vanish at `W_g = 0`, so the first backward moves it."""
    m_off = _model(tul_grad_pass=False)
    out_off, _, _ = _run(m_off)
    out_off["loss"].backward()

    m_on = _model(tul_grad_pass=True)
    assert isinstance(m_on.tul_grad_pass, TULGradPass)
    assert float(m_on.tul_grad_pass.W_g.detach().abs().max()) == 0.0, "W_g is not zero-initialised"
    out_on, _, _ = _run(m_on)
    out_on["loss"].backward()

    assert torch.equal(out_off["loss"], out_on["loss"]), "the zero-init arm moved the loss"
    assert _same(_grads(m_off), _grads(m_on)), "the zero-init arm moved a gradient"
    wg = m_on.tul_grad_pass.W_g
    assert wg.grad is not None and float(wg.grad.abs().sum()) > 0.0, "W_g cannot escape 0"


def test_the_feature_runs_at_eval_too_it_is_part_of_the_map():
    """`core_depth_sweep.py` reads the arm at forced depths under `model.eval()` and
    `torch.no_grad()`. The feature is an INPUT to the map, not a training term, so it must
    be built there as well — otherwise the sweep measures a different function than the one
    that trained."""
    m = _model(tul_grad_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _randomise_wg(m)
    _fixed_depths(m, 2)
    seen = _feature_spy(m)
    x, y, layout, _ = _batch()
    m.eval()
    with torch.no_grad():
        m(x, labels=y, slot_layout=layout)
    assert len(seen) == 2, len(seen)
    assert float(seen[0]["term"].detach().abs().max()) > 0.0


def test_building_the_module_draws_no_random_numbers():
    """An arm with the knob on must hold the ruler's weights everywhere else, so the
    construction of `W_g` may not touch the model's RNG stream."""
    m_off, m_on = _model(tul_grad_pass=False), _model(tul_grad_pass=True)
    a = {n: p for n, p in m_off.named_parameters()}
    b = {n: p for n, p in m_on.named_parameters()}
    assert set(b) - set(a) == {"tul_grad_pass.W_g"}
    for n in a:
        assert torch.equal(a[n], b[n]), f"{n} moved when the module was built"


# ── 3. fixture sensitivity: the term acts, and scales ────────────────────────

def test_the_injected_term_scales_with_grad_pass_scale_and_moves_the_state():
    """Pass 0's feature is taken at the loop ENTRY, which `W_g` cannot have touched yet, so
    the two arms differ at that pass by the scale alone and the comparison is exact."""
    terms, states = {}, {}
    for scale in (0.1, 0.2):
        m = _model(tul_grad_pass=True, tul_grad_pass_scale=scale, slot_gain_lambda=0.0,
                   dropout=0.0)
        _randomise_wg(m)
        _fixed_depths(m, 2)
        seen = _feature_spy(m)
        _run(m)
        terms[scale] = seen[0]["term"].detach()
        states[scale] = seen[1]["h"]          # the state pass 1 was handed
    assert torch.allclose(terms[0.2], 2.0 * terms[0.1], rtol=1e-5, atol=1e-7)
    assert float(terms[0.1].abs().max()) > 0.0, "the check is vacuous — the term is zero"

    m0 = _model(tul_grad_pass=False, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m0, 2)
    off_states: list[torch.Tensor] = []
    real = m0._tul_core

    def spy(*a, **k):
        res = real(*a, **k)
        off_states.append(res[1].detach().clone())
        return res

    m0._tul_core = spy
    _run(m0)
    assert not torch.allclose(states[0.1], off_states[0]), "pass 1 saw an unchanged state"


# ── 4. the feature IS the gradient of the own-span loss ──────────────────────

def test_the_injected_feature_equals_autograd_grad_of_the_own_span_loss():
    """Recomputed by hand at the SAME state the loop was at, from the definition: the
    gradient of `_tul_mux_loss(target="own")` with respect to the slot state, reduced over
    the Hyper-Connection streams the MUX head's readout means over."""
    m = _model(tul_grad_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _randomise_wg(m)
    _fixed_depths(m, 3)
    seen = _feature_spy(m)
    _out, x, layout = _run(m)
    assert len(seen) == 3, len(seen)

    for t, rec in enumerate(seen):
        h_d = rec["h"].detach().requires_grad_(True)
        loss = m._tul_mux_loss(h_d, x, layout, slot_keep=rec["mask"], target="own")
        by_hand = torch.autograd.grad(loss, h_d)[0].mean(dim=2)
        assert torch.allclose(rec["g"], by_hand, atol=1e-5, rtol=1e-4), f"pass {t}"
        assert float(by_hand.abs().max()) > 0.0, f"pass {t}: the check is vacuous"


def test_the_feature_follows_the_full_stream_readout_too():
    """`tul.mux_readout: full` (finding F2) normalises each Hyper-Connection stream
    separately, so the per-stream gradients genuinely differ and the mean over them is a
    summary rather than a recovery of one vector. The contract is the same either way: what
    is injected is `autograd.grad` of the own loss under the readout the loss used."""
    m = _model(tul_grad_pass=True, tul_mux_readout="full", slot_gain_lambda=0.0,
               dropout=0.0)
    assert m.cfg.tul.mux_readout == "full"
    _randomise_wg(m)
    _fixed_depths(m, 2)
    seen = _feature_spy(m)
    _out, x, layout = _run(m)
    for t, rec in enumerate(seen):
        h_d = rec["h"].detach().requires_grad_(True)
        loss = m._tul_mux_loss(h_d, x, layout, slot_keep=rec["mask"], target="own")
        full = torch.autograd.grad(loss, h_d)[0]
        assert torch.allclose(rec["g"], full.mean(dim=2), atol=1e-5, rtol=1e-4), f"pass {t}"
        assert float(full.mean(dim=2).abs().max()) > 0.0, f"pass {t}: vacuous"
    # The fixture is not vacuous: under "full" the streams are NOT one scaled vector.
    h_d = seen[0]["h"].detach().requires_grad_(True)
    loss = m._tul_mux_loss(h_d, x, layout, slot_keep=seen[0]["mask"], target="own")
    full = torch.autograd.grad(loss, h_d)[0]
    assert not torch.allclose(full[:, :, 0], full[:, :, 1], atol=1e-7)


def test_the_own_target_is_not_the_exit_target():
    """The toy study's finding, held as a contract: the local target the feature
    differentiates must DIFFER from the target the exit is trained on. The arm runs
    `mux_target: next` at the exit and `target="own"` inside the loop."""
    m = _model(tul_grad_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _randomise_wg(m)
    seen = _feature_spy(m)
    _out, x, layout = _run(m)
    h = seen[0]["h"].detach().requires_grad_(True)
    own = m._tul_mux_loss(h, x, layout, slot_keep=seen[0]["mask"], target="own")
    nxt = m._tul_mux_loss(h, x, layout, slot_keep=seen[0]["mask"], target="next")
    assert not torch.allclose(own, nxt)
    assert m.cfg.tul.mux_target == "next"


# ── 5. the feature is detached ───────────────────────────────────────────────

def test_no_outer_gradient_flows_through_the_injected_feature():
    """`create_graph=False`: the gradient is a FEATURE. The tensor the loop injects carries
    no `grad_fn`, so the token CE never differentiates through the inner backward and no
    second-order term exists. `W_g` is the only route the outer graph has into it, and it
    is a live one."""
    m = _model(tul_grad_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _randomise_wg(m)
    seen = _feature_spy(m)
    out, _, _ = _run(m)
    assert seen, "the feature was never built"
    for t, rec in enumerate(seen):
        assert rec["g"].grad_fn is None, f"pass {t}: the feature kept a graph"
        assert not rec["g"].requires_grad, f"pass {t}: the feature requires grad"
        assert rec["term"].grad_fn is not None, f"pass {t}: W_g lost its graph"
    out["loss"].backward()
    assert float(m.tul_grad_pass.W_g.grad.abs().sum()) > 0.0


def test_the_per_pass_readouts_report_the_local_target_and_the_terms_size():
    """`loop/own_pass_t{t}` is the arm's own instrument: the value of the target whose
    gradient each pass is handed. `loop/gp_rel_t{t}` says whether the injected term is
    moving the state at all. Both are training-only, so an eval forward cannot overwrite
    the step the trainer is about to read."""
    m = _model(tul_grad_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _randomise_wg(m)
    _fixed_depths(m, 3)
    seen = _feature_spy(m)
    _run(m)
    stats = m._loop_gradpass
    assert set(stats) == {f"{k}_t{t}" for t in range(3) for k in ("own_pass", "gp_rel")}
    for t, rec in enumerate(seen):
        assert float(stats[f"own_pass_t{t}"]) == pytest.approx(float(rec["own"]), rel=1e-6)
        assert float(stats[f"gp_rel_t{t}"]) > 0.0

    m.eval()
    x, y, layout, _ = _batch()
    with torch.no_grad():
        m(x, labels=y, slot_layout=layout)
    assert m._loop_gradpass is stats, "an eval forward overwrote the training readouts"


def test_the_readouts_are_absent_when_the_knob_is_off():
    m = _model(tul_grad_pass=False)
    _run(m)
    assert m._loop_gradpass is None


# ── 6. refusals ──────────────────────────────────────────────────────────────

def test_the_paid_loop_refuses_the_knob():
    with pytest.raises(NotImplementedError, match="SLOT-LOOP lever"):
        _tul(grad_pass=True, mux_beta=1.0, tokens_through_core=True)


def test_a_zero_mux_beta_refuses_the_knob():
    with pytest.raises(ValueError, match="mux_beta"):
        _tul(grad_pass=True, mux_beta=0.0)


def test_db_loop_refuses_the_knob():
    with pytest.raises(ValueError, match="db_loop"):
        _tul(grad_pass=True, mux_beta=1.0, db_loop=True)


def test_a_zero_scale_and_an_unknown_norm_are_refused():
    with pytest.raises(ValueError, match="grad_pass_scale"):
        _tul(grad_pass=True, mux_beta=1.0, grad_pass_scale=0.0)
    with pytest.raises(ValueError, match="grad_pass_norm"):
        _tul(grad_pass=True, mux_beta=1.0, grad_pass_norm="l2")


def test_a_coreless_model_refuses_the_knob():
    with pytest.raises(ValueError, match="n_core"):
        _model(tul_grad_pass=True, n_core=0)


def test_a_caller_that_omits_input_ids_raises_rather_than_losing_the_feature():
    """`_tul_core`'s other callers pass four positional arguments. A new one that forgets
    `input_ids` must not run the loop with the mechanism silently switched off."""
    m = _model(tul_grad_pass=True)
    x, _y, layout, _ = _batch()
    front = m._tul_front(x, layout)
    with pytest.raises(RuntimeError, match="input_ids"):
        m._tul_core(*front, layout)


def test_scse_refuses_the_knob():
    """Under SCSE the carry is the DEVIATION, so the own-span readout of it is not a
    readout of the slot and its gradient is not the gradient of the loss."""
    m = _model(tul_grad_pass=True)
    m.scse = object.__new__(type("FakeSCSE", (), {}))       # non-None sentinel
    x, _y, layout, _ = _batch()
    front = m._tul_front(x, layout)
    with pytest.raises(NotImplementedError, match="SCSE"):
        m._tul_core(*front, layout, input_ids=x)
