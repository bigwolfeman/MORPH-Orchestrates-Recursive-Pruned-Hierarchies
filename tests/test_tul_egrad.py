"""The latent-z gradient arms: energies, the residual bound, and the refused knob.

``tul.grad_pass`` hands every pass of the slot loop the gradient of a local objective with
respect to the state it is refining. Until 2026-09-12 there was ONE energy — the slot's own
span against an order-free bag through the tied head — and the arm that ran it descended
that energy 0.366 nats in the FIRST pass and then sat flat. This file covers the two
energies built to be harder than that (``morph/model/tul_egrad.py``), the per-pass residual
bound, and the knob that is refused because the tree already does what it asks for.

One test per contract:

  1. ``grad_pass_energy='own_mux'`` builds nothing and is the forward from before.
  2. ``own_span_slots`` gathers exactly the tokens ``mux_span_targets(target='own')``
     supervises, token for token.
  3. the SOFT target is EXACTLY the Label-Forcing mixture (checked against an eager
     full-logits reference, and against the hard CE at mix 0 and 1).
  4. ``recon``: the energy decoder's parameters receive gradient, and the energy's own
     training loss reaches ``z`` NOWHERE — the only route from the energy into the loop is
     ``W_g``, and the feature that crosses it is detached.
  5. ``disc``: the critic's loss reaches only its own parameters, and its label is the
     coda's next-span CE against the batch median.
  6. ``pass_residual_lambda``: 0 is bit-identical to the parent; > 0 adds exactly
     ``pass_res_weighted`` to the loss and nothing else to the forward.
  7. ``reinject_seed_every_pass`` RAISES, and the reason it raises is true: the slot's
     prelude-entry state reaches every pass through the injection.
  8. the plain path (``slot_layout=None``) stays bit-identical with every module built.
  9. ``KNOWN_TUL_KEYS`` accepts the new keys and rejects a misspelling.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer
from morph.model.tul import mux_span_targets
from morph.model.tul_egrad import DiscEnergy, ReconEnergy, slot_outcome_labels
from morph.model.tul_spandec import own_span_slots

MAX_DEPTH = 3


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    tul_base = dict(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0,
                    mux_target="next", mux_detach_head=True)
    tul_base.update(tul_kw)          # a `tul_*` kwarg OVERRIDES the default, never collides
    tul = _tul(**tul_base)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.0,
                core_fixed_point_lambda=1.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, seed: int = 7):
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    return out, x, y, layout


def _grads(m: MORPHTransformer, skip: tuple[str, ...] = ()):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters() if not n.startswith(skip)}


def _same(ga, gb) -> bool:
    return set(ga) == set(gb) and all(
        ((ga[k] is None) == (gb[k] is None))
        and (ga[k] is None or torch.equal(ga[k], gb[k])) for k in ga)


# ── 1. off is nothing ────────────────────────────────────────────────────────

def test_own_mux_energy_builds_nothing_and_is_the_old_forward():
    off = _model(tul_grad_pass=True)
    assert off.tul_egrad is None
    a, *_ = _run(off)
    a["loss"].backward()
    ga = _grads(off)

    # The same model built with the key spelled out explicitly.
    on = _model(tul_grad_pass=True, tul_grad_pass_energy="own_mux")
    assert on.tul_egrad is None
    b, *_ = _run(on)
    b["loss"].backward()
    assert torch.equal(a["loss"].detach(), b["loss"].detach())
    assert _same(ga, _grads(on))
    assert "egrad" not in b and "pass_residual" not in b


def test_the_mux_beta_guard_is_scoped_to_the_mux_energy():
    """`tul.grad_pass` used to demand `mux_beta > 0` unconditionally. That is right for the
    MUX energy — it IS the MUX head's loss — and WRONG for the two energies that own their
    own scorer. It blocked the whole batch: every arm here descends from
    `tul_slot_spandec_mask`, which sets `mux_beta: 0` because the span decoder REPLACES the
    MUX. Caught by composing the configs, not by a test, so here is the test."""
    from morph.model.tul import TULConfig
    with pytest.raises(ValueError, match="grad_pass_energy='own_mux' needs tul.mux_beta"):
        TULConfig(grad_pass=True, mux_beta=0.0, grad_pass_energy="own_mux")
    for energy in ("recon", "disc"):
        cfg = TULConfig(grad_pass=True, mux_beta=0.0, grad_pass_energy=energy)
        assert cfg.grad_pass_energy == energy
    # ... and the shipped gradpass arm's setting still passes the guard it was written for.
    assert TULConfig(grad_pass=True, mux_beta=1.0).grad_pass_energy == "own_mux"
    with pytest.raises(ValueError, match="grad_pass_energy must be"):
        TULConfig(grad_pass=True, mux_beta=1.0, grad_pass_energy="nope")


def test_the_two_energy_arms_build_at_mux_beta_zero():
    """End to end at the arms' own setting: mux_beta 0, the span decoder on, an energy on."""
    for energy in ("recon", "disc"):
        m = _model(tul_mux_beta=0.0, tul_grad_pass=True, tul_grad_pass_energy=energy,
                   tul_egrad_max_tokens=6, tul_pass_residual_lambda=0.01)
        out, *_ = _run(m)
        assert torch.isfinite(out["loss"]) and "egrad" in out and "pass_residual" in out
        assert "mux_local" not in out          # the MUX really is off
        out["loss"].backward()
        assert float(m.tul_grad_pass.W_g.grad.abs().sum()) > 0.0


def test_an_energy_without_grad_pass_is_refused():
    with pytest.raises(ValueError, match="without tul.grad_pass"):
        _model(tul_grad_pass=False, tul_grad_pass_energy="recon")
    with pytest.raises(ValueError, match="must be 'own_mux'"):
        _model(tul_grad_pass=True, tul_grad_pass_energy="reconstruct")


# ── 2. the own-span gather IS the own-span target ────────────────────────────

def test_own_span_slots_agrees_with_mux_span_targets_token_for_token():
    x, _y, layout, _ = _batch()
    J = 32
    ids, valid = own_span_slots(x, layout, J)
    pos_valid, _alpha, tgt_slot, _sup = mux_span_targets(x, layout, 0.9, target="own")

    # Every position the MUX supervises must appear exactly once in the gather, under the
    # slot it supervises, carrying its own token id.
    got: dict[tuple[int, int], list[int]] = {}
    B, S, _ = ids.shape
    for b in range(B):
        for s in range(S):
            got[(b, s)] = [int(v) for v, ok in zip(ids[b, s], valid[b, s]) if ok]
    want: dict[tuple[int, int], list[int]] = {(b, s): [] for b in range(B) for s in range(S)}
    for b in range(B):
        for p in range(x.shape[1]):
            if bool(pos_valid[b, p]):
                want[(b, int(tgt_slot[b, p]))].append(int(x[b, p]))
    assert any(v for v in want.values()), "fixture supervises nothing — test is vacuous"
    assert got == want


def test_own_span_gather_is_not_the_next_span_gather():
    """Fixture sensitivity: the two shifts must actually differ on this batch."""
    from morph.model.tul_spandec import next_span_slots
    x, _y, layout, _ = _batch()
    own, ov = own_span_slots(x, layout, 32)
    nxt, nv = next_span_slots(x, layout, 32)
    assert not torch.equal(own * ov, nxt * nv)


# ── 3. the soft target IS the Label-Forcing mixture ──────────────────────────

def _eager_soft_ce(st, w, ids, valid, mix):
    """Reference: full logits, the exact mixture target, no kernel."""
    import torch.nn.functional as F
    B, S, J, C = st.shape
    logp = F.log_softmax((st.reshape(-1, C).float() @ w.float().t()), dim=-1)
    logp = logp.reshape(B, S, J, -1)
    tot, n = 0.0, 0
    for b in range(B):
        for s in range(S):
            span = [int(ids[b, s, j]) for j in range(J) if bool(valid[b, s, j])]
            if not span:
                continue
            for j in range(J):
                if not bool(valid[b, s, j]):
                    continue
                hard = -logp[b, s, j, int(ids[b, s, j])]
                bag = -sum(logp[b, s, j, t] for t in span) / len(span)
                tot = tot + (1 - mix) * hard + mix * bag
                n += 1
    return tot / max(n, 1)


def test_soft_labels_match_an_eager_mixture_reference():
    torch.manual_seed(0)
    x, _y, layout, _ = _batch()
    d, V = 16, 64
    eg = ReconEnergy(d_model=d, n_heads=2, d_ff=32, n_layers=1, max_tokens=8,
                     soft_labels=True, soft_mix=0.5)
    w = torch.randn(V, d) * 0.05
    ids_in = x.clamp(max=V - 1)
    z = torch.randn(x.shape[0], layout.slot_index.shape[1], d)
    ids, valid = own_span_slots(ids_in, layout, 8)
    valid = valid & layout.slot_valid.unsqueeze(-1)
    st = eg.dec.decode(z, ids, valid, w)
    got = eg.loss(z, ids_in, layout, w, w, chunk=256, mask_token_id=-1,
                  slot_keep=layout.slot_valid)
    want = _eager_soft_ce(st, w, ids, valid, 0.5)
    assert torch.allclose(got.float(), want.float(), atol=2e-5), (float(got), float(want))

    # mix 0 is the hard CE; mix 1 is the pure bag. Both must differ, or the knob is inert.
    eg.soft_mix = 0.0
    hard = eg.loss(z, ids_in, layout, w, w, 256, -1, slot_keep=layout.slot_valid)
    eg.soft_mix = 1.0
    bag = eg.loss(z, ids_in, layout, w, w, 256, -1, slot_keep=layout.slot_valid)
    assert abs(float(hard) - float(bag)) > 1e-3
    assert torch.allclose(got.float(), (0.5 * hard + 0.5 * bag).float(), atol=2e-5)
    eg.soft_labels = False
    assert torch.equal(eg.loss(z, ids_in, layout, w, w, 256, -1,
                               slot_keep=layout.slot_valid), hard)


# ── 4. recon: the energy trains itself and NEVER z ───────────────────────────

def test_recon_energy_trains_its_own_parameters_and_never_z():
    m = _model(tul_grad_pass=True, tul_grad_pass_energy="recon",
               tul_egrad_max_tokens=6, tul_egrad_soft_labels=True)
    assert isinstance(m.tul_egrad, ReconEnergy)
    out, x, y, layout = _run(m)
    assert "egrad" in out and float(out["egrad"]) > 0.0
    assert "egrad_weighted" in out
    out["loss"].backward()

    eps = [p for n, p in m.named_parameters() if n.startswith("tul_egrad")]
    assert eps, "no energy parameters"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in eps)
    # W_g is the ONLY route from the feature into the loop, and it escapes zero.
    wg = m.tul_grad_pass.W_g
    assert wg.grad is not None and float(wg.grad.abs().sum()) > 0.0

    # The energy's OWN loss reaches z nowhere: rebuild it on a live h_slots and ask
    # autograd. This is the contract stated by construction in `_egrad_train_loss`.
    m.zero_grad(set_to_none=True)
    h = torch.randn(x.shape[0], layout.slot_index.shape[1], m.cfg.hc_streams,
                    m.cfg.d_model, requires_grad=True)
    loss = m._egrad_train_loss(h, x, layout, None, torch.zeros(1), y, {})
    g = torch.autograd.grad(loss, h, allow_unused=True, retain_graph=True)[0]
    assert g is None, "the energy's training loss reached z — z must be detached there"
    ge = torch.autograd.grad(loss, m.tul_egrad.dec.z_in.weight, allow_unused=True)[0]
    assert ge is not None and float(ge.abs().sum()) > 0.0


def test_the_recon_feature_is_detached():
    m = _model(tul_grad_pass=True, tul_grad_pass_energy="recon", tul_egrad_max_tokens=6)
    seen = []
    real = m.tul_grad_pass.forward

    def spy(g, slot_valid):
        seen.append(g)
        return real(g, slot_valid)

    m.tul_grad_pass.forward = spy
    _run(m)
    assert seen, "the feature never ran"
    for g in seen:
        assert g.grad_fn is None and not g.requires_grad


# ── 5. disc: the critic scores the outcome and trains only itself ────────────

def test_disc_critic_loss_reaches_only_its_own_parameters():
    m = _model(tul_grad_pass=True, tul_grad_pass_energy="disc")
    assert isinstance(m.tul_egrad, DiscEnergy)
    out, x, y, layout = _run(m)
    assert "egrad" in out and "egrad_auc" in out and "egrad_pos_frac" in out
    assert 0.0 <= float(out["egrad_auc"]) <= 1.0
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.tul_egrad.parameters())

    m.zero_grad(set_to_none=True)
    B, S, C = x.shape[0], layout.slot_index.shape[1], m.cfg.d_model
    z = torch.randn(B, S, C, requires_grad=True)
    ctx = torch.randn(B, S, C, requires_grad=True)
    lab = (torch.rand(B, S) > 0.5).float()
    bce = m.tul_egrad.bce(z.detach(), ctx.detach(), lab, layout.slot_valid)
    assert torch.autograd.grad(bce, z, allow_unused=True, retain_graph=True)[0] is None
    assert torch.autograd.grad(bce, ctx, allow_unused=True, retain_graph=True)[0] is None
    g = torch.autograd.grad(bce, m.tul_egrad.fc2.weight, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0.0

    # ... and the model's own `_egrad_train_loss` detaches z before it gets there, which is
    # the statement that matters: the critic is a reader of the loop, never a writer to it.
    h = torch.randn(B, S, m.cfg.hc_streams, C, requires_grad=True)
    xh = torch.randn(B, x.shape[1], C)
    loss = m._egrad_train_loss(h, x, layout, torch.randn(B, S, C), xh, y, {})
    assert torch.autograd.grad(loss, h, allow_unused=True, retain_graph=True)[0] is None
    gc = torch.autograd.grad(loss, m.tul_egrad.fc1.weight, allow_unused=True)[0]
    assert gc is not None and float(gc.abs().sum()) > 0.0


def test_the_outcome_label_is_the_next_span_ce_against_the_median():
    torch.manual_seed(0)
    x, y, layout, _ = _batch()
    B, L = x.shape
    d, V = 8, 64
    xh = torch.randn(B, L, d)
    w = torch.randn(V, d) * 0.1
    lab = y.clamp(max=V - 1)
    lab = torch.where(y == -100, torch.full_like(y, -100), lab)
    yy, scored, ce = slot_outcome_labels(xh, lab, layout, w, chunk=128)
    assert bool(scored.any()), "no slot scored — test is vacuous"
    vals = ce[scored]
    med = float(vals.median())
    # Exactly the thresholding claim, checked per slot.
    assert torch.equal(yy[scored] > 0.5, ce[scored] < med)
    # And the label set is balanced-ish by construction (a median split).
    frac = float(yy[scored].mean())
    assert 0.0 <= frac <= 0.6


# ── 6. the per-pass residual bound ───────────────────────────────────────────

def test_pass_residual_lambda_zero_is_bit_identical():
    a = _model()
    oa, *_ = _run(a)
    oa["loss"].backward()
    b = _model(tul_pass_residual_lambda=0.0)
    ob, *_ = _run(b)
    ob["loss"].backward()
    assert torch.equal(oa["loss"].detach(), ob["loss"].detach())
    assert _same(_grads(a), _grads(b))
    assert "pass_residual" not in ob


def test_pass_residual_adds_exactly_its_weighted_term():
    lam = 0.25
    a = _model()
    oa, *_ = _run(a)
    b = _model(tul_pass_residual_lambda=lam)
    ob, *_ = _run(b)
    assert "pass_residual" in ob and float(ob["pass_residual"]) > 0.0
    assert torch.allclose(ob["pass_res_weighted"],
                          lam * ob["pass_residual"], atol=0, rtol=1e-6)
    # The FORWARD is untouched: the only difference in the total loss is the term.
    assert torch.allclose(ob["loss"].detach() - ob["pass_res_weighted"],
                          oa["loss"].detach(), atol=1e-6)
    # ... and it is NOT the terminal fixed-point term: both are reported, and they differ
    # because the residual averages every pass while the fixed point charges the last.
    assert "fixed_point" in ob
    assert abs(float(ob["pass_residual"]) - float(ob["fixed_point"])) > 1e-9


def test_pass_residual_changes_the_core_gradient():
    """Sabotage guard: a term that is added to the loss but never reaches the map would
    pass the two tests above and do nothing. This one fails if it does not train."""
    a = _model()
    oa, *_ = _run(a)
    oa["loss"].backward()
    b = _model(tul_pass_residual_lambda=0.5)
    ob, *_ = _run(b)
    ob["loss"].backward()
    ga, gb = _grads(a), _grads(b)
    moved = [k for k in ga if k.startswith("core.") and ga[k] is not None
             and not torch.equal(ga[k], gb[k])]
    assert moved, "the residual term reached no core parameter"


# ── 7. the refused knob ──────────────────────────────────────────────────────

def test_reinject_seed_every_pass_is_refused():
    with pytest.raises(NotImplementedError, match="NO-OP by construction"):
        _model(tul_reinject_seed_every_pass=True)


def test_the_reason_it_is_refused_is_true_the_seed_reaches_every_pass():
    """The refusal claims `_apply_core_step` injects the slot's prelude-entry state at
    EVERY pass. Zeroing that argument must change every pass's output, not only pass 0."""
    m = _model()
    x, y, layout, _ = _batch()
    m.train()
    seen: list[int] = []

    def spy(_mod, args):
        e_in = args[1] if len(args) > 1 else None
        seen.append(0 if e_in is None else int(e_in.abs().sum() > 0))

    h = m.injection.register_forward_pre_hook(spy)
    torch.manual_seed(7)
    m(x, labels=y, slot_layout=layout)
    h.remove()
    # `_apply_core_step` is the ONLY caller, once per core pass; what matters is that
    # every one of those calls carries a non-zero source.
    assert len(seen) >= MAX_DEPTH
    assert sum(seen) >= MAX_DEPTH, "the entry state did not reach every pass"


# ── 8. the plain path is untouched with every module built ───────────────────

def test_plain_path_is_bit_identical_with_every_new_module_built():
    x, y, layout, _ = _batch()
    base = _model(seed=5)
    torch.manual_seed(11)
    a = base(x, labels=y)                       # slot_layout=None — the plain forward
    rich = _model(seed=5, tul_grad_pass=True, tul_grad_pass_energy="recon",
                  tul_egrad_max_tokens=6, tul_pass_residual_lambda=0.3)
    torch.manual_seed(11)
    b = rich(x, labels=y)
    assert torch.equal(a["loss"].detach(), b["loss"].detach())
    assert "egrad" not in b and "pass_residual" not in b


# ── 9. the config keys ───────────────────────────────────────────────────────

def test_known_tul_keys_accepts_the_new_keys_and_rejects_a_misspelling():
    from morph.training.tul_setup import KNOWN_TUL_KEYS, reject_unknown_tul_keys
    new = {"grad_pass_energy", "egrad_weight", "egrad_layers", "egrad_heads",
           "egrad_max_tokens", "egrad_soft_labels", "egrad_soft_mix",
           "egrad_disc_hidden", "pass_residual_lambda", "reinject_seed_every_pass"}
    assert new <= KNOWN_TUL_KEYS
    reject_unknown_tul_keys({k: 0 for k in new})
    with pytest.raises(ValueError, match="egrad_soft_label"):
        reject_unknown_tul_keys({"egrad_soft_label": True})
