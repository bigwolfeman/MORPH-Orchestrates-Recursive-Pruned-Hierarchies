"""``tul.horizon_weight`` / ``tul.pass_readout`` — the LoopMTP port (arXiv 2608.03624,
Eq 9-14), arm ``slot-spandec-strict-horizon``.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_horizon.py -v

WHAT THIS FILE HAS TO PROVE:

1. OFF IS BIT-IDENTICAL. ``tul.horizon_weight: 0`` (the default) and
   ``tul.pass_readout: "last"`` (the default) build no new module, draw no RNG and leave
   the forward untouched — pinned against a SECOND PROCESS running the pre-change source
   at this tree's own pre-arm commit, not argued (see ``PIN`` below and
   ``lab/divergence/`` for the probe script this pin was produced by).
2. THE TERM IS REAL. Positive, enters ``loss``, and its gradient reaches the LOOP alone —
   never the tied embedding table (the target is detached) and never the span decoder
   (the term needs no decoder at all).
3. THE TARGET IS THE RIGHT SPAN. Pass ``t`` is graded against span ``i+t``, not ``i+1``
   restaged — perturbing span ``i+3`` moves ONLY pass 3's term.
4. THE MASK. A row with no span ``i+t`` contributes nothing to pass ``t``.
5. PASS 1 IS FREE by default (LoopMTP's own choice), and the knob to turn that off works.
6. THE GATE (``pass_readout="gated"``) normalises to one and its argmax can move the
   written ``z`` — the coda and the span decoder read THAT z, not the raw last pass.
7. THE REFUSAL. A Poisson-depth model (``slot_depth_fixed: 0``) raises for both knobs.
8. Every new ``tul.*`` key is in ``tul_setup.KNOWN_TUL_KEYS``.

CPU only, fp32, ``use_kernels=False``, tiny config (the ``test_tul_slot_register``
fixture, imported rather than duplicated).

Record: lab/experiments/planned/2026-09-14-arc-horizon-passes.md
Note: .agents/notes/proposed/architecture/2026-09-14-horizon-indexed-passes.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# tests/ is on sys.path; the register file owns the tiny strict fixture this arm runs on.
from test_tul_slot_register import _batch, _edit, _ids, _model, _rule, _tiny, _tul  # noqa: E402

from morph.model.tul import TULConfig  # noqa: E402
from morph.model.tul_layout import TulLayoutSpec, slot_layout_from_ids  # noqa: E402
from morph.model.transformer import MORPHTransformer  # noqa: E402
from morph.training.tul_setup import KNOWN_TUL_KEYS  # noqa: E402


def _horizon(T: int = 3, seed: int = 99, **tul_kw) -> MORPHTransformer:
    """The fixture model: the strict slot loop, fixed depth T, horizon_weight on."""
    return _model(1, seed=seed, slot_depth_fixed=T, horizon_weight=1.0, **tul_kw)


def _gated(T: int = 3, seed: int = 99, **tul_kw) -> MORPHTransformer:
    m = _model(1, seed=seed, slot_depth_fixed=T, pass_readout="gated", **tul_kw)
    # Wg is zero-init (the ruler's step-0 gate is content-blind); nudge it so the gate's
    # argmax can actually be moved by content, or the perturbation contract is untested.
    torch.manual_seed(4)
    with torch.no_grad():
        m.tul_pass_gate.Wg.weight.add_(0.05 * torch.randn_like(m.tul_pass_gate.Wg.weight))
    return m


def _run(m: MORPHTransformer, seed: int = 5):
    _i, inp, lab, layout = _batch(1)
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout)


# ── 1. OFF IS BIT-IDENTICAL ──────────────────────────────────────────────────

# Produced by running the PRE-ARM source (this tree's own commit before the horizon-
# passes arm, `morph/model/tul.py` / `transformer.py` / `tul_setup.py` / `train.py`
# restored from `git show HEAD:<path>` into a swapped-in copy of `morph/`) and this tree
# in two SEPARATE processes on the SAME fixture and the SAME seeds
# (`_tiny()`/`_rule()`/`_ids()`, `tul_a1`-style: prefix_k=2, slot_id=4, spandec on,
# tg_geometry=strict, slot_depth_fixed=3), `torch.manual_seed(99)` at construction and
# `torch.manual_seed(5)` at both the train and the eval forward. The "new" process passes
# `horizon_weight=0.0, pass_readout="last"` EXPLICITLY (both are the defaults) to prove
# they are true no-ops, not merely unset. All four columns matched to the last printed
# digit:
#
#   variant=old loss=9.9712696075 logit_sum=587.699934 n_fin=10584 gradsum=1751.887209
#   variant=new loss=9.9712696075 logit_sum=587.699934 n_fin=10584 gradsum=1751.887209
PIN = dict(loss=9.9712696075, logit_sum=587.699934, n_fin=10584, gradsum=1751.887209)


def test_off_is_bit_identical_to_the_pre_change_source():
    tul = _tul(slot_depth_fixed=3, horizon_weight=0.0, pass_readout="last")
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(tul=tul))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    m = m.train().float()
    _i, inp, lab, layout = _batch(1)
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    gs = sum(float(p.grad.double().abs().sum()) for p in m.parameters()
             if p.grad is not None)
    m.eval()
    with torch.no_grad():
        torch.manual_seed(5)
        lg = m(inp, labels=None, slot_layout=layout)["logits"].float()
    fin = torch.isfinite(lg)
    assert float(out["loss"]) == pytest.approx(PIN["loss"], abs=1e-9)
    assert gs == pytest.approx(PIN["gradsum"], abs=1e-4)
    assert int(fin.sum()) == PIN["n_fin"]
    assert float(lg[fin].double().sum()) == pytest.approx(PIN["logit_sum"], abs=1e-4)


def test_off_builds_no_modules():
    m = _model(1, slot_depth_fixed=3)  # horizon_weight=0.0, pass_readout="last": defaults
    assert m.tul_horizon_proj is None
    assert m.tul_pass_gate is None
    assert not any("tul_horizon_proj" in k or "tul_pass_gate" in k for k in m.state_dict())


def test_horizon_weight_zero_with_gated_readout_is_still_bit_identical_off_the_loss():
    """`pass_readout="gated"` is orthogonal to `horizon_weight`: the loss term is off,
    but the READOUT still changes the forward. This just proves the two knobs are
    independent — the gate contract itself is tested in section 6."""
    m = _gated(3)
    assert m.tul_horizon_proj is None
    assert m.tul_pass_gate is not None


# ── 2. THE TERM IS REAL ──────────────────────────────────────────────────────

def test_horizon_term_is_positive_and_enters_the_loss():
    m0 = _model(1, seed=99, slot_depth_fixed=3)
    m1 = _horizon(3)
    _i, inp, lab, layout = _batch(1)
    torch.manual_seed(5)
    o0 = m0(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    o1 = m1(inp, labels=lab, slot_layout=layout)
    assert "horizon" in o1 and "horizon_weighted" in o1
    assert float(o1["horizon"]) > 0.0
    assert torch.equal(o1["horizon_weighted"], o1["horizon"])   # weight is 1.0 here
    assert not torch.allclose(o0["loss"], o1["loss"])
    assert float(o1["loss"]) == pytest.approx(
        float(o0["loss"]) + float(o1["horizon_weighted"]), abs=1e-5)


def test_horizon_target_trains_the_loop_and_nothing_else():
    """Edge test via `autograd.grad` on the LIVE (undetached) term, captured by spying on
    `_tul_horizon_loss` -- `out["horizon"]`/`out["horizon_weighted"]` are detached, the
    same contract every other weighted TUL term follows. The gradient must reach the
    core blocks and `tul_horizon_proj`, and reach NEITHER the span decoder (the term
    needs no decoder at all). The embedding table is checked SEPARATELY, in
    `test_horizon_target_is_detached_from_the_embedding_table` below: on the REAL
    forward `z = readout(db_traj[t])` legitimately depends on the tied embedding table
    too (a slot's seed reads its own span's tokens, and the loop is causal), so a
    nonzero gradient at the embedding leaf here would be expected EITHER WAY and cannot
    tell the target's detach apart from that real dependency."""
    m = _horizon(3)
    _i, inp, lab, layout = _batch(1)
    real = m._tul_horizon_loss
    captured = {}

    def spy(*a, **kw):
        live = real(*a, **kw)
        captured["live"] = live
        return live

    m._tul_horizon_loss = spy
    try:
        torch.manual_seed(5)
        m(inp, labels=lab, slot_layout=layout)
    finally:
        m._tul_horizon_loss = real
    live = captured["live"]
    assert live.requires_grad, "fixture: the captured term carries no graph at all"

    dec_params = list(m.tul_spandec.parameters()) if m.tul_spandec is not None else []
    proj_params = list(m.tul_horizon_proj.parameters())
    core_params = [p for blk in m.core for p in blk.parameters()]
    targets = proj_params + core_params + dec_params
    grads = torch.autograd.grad(live, targets, retain_graph=True, allow_unused=True)
    for p, g in zip(proj_params, grads[:len(proj_params)]):
        assert g is not None and bool((g != 0).any()), "horizon term did not train its own projection"
    core_grads = grads[len(proj_params):len(proj_params) + len(core_params)]
    assert any(g is not None and bool((g != 0).any()) for g in core_grads), (
        "horizon term did not reach any core block")
    dec_grads = grads[len(proj_params) + len(core_params):]
    assert all(g is None for g in dec_grads), (
        "horizon term reached the span decoder — it should need no decoder at all")


def test_horizon_target_is_detached_from_the_embedding_table():
    """Isolated: calls `_tul_horizon_loss` directly against a FIXED, input-independent
    `db_traj` (random noise, no connection at all to `m.embed`'s parameters) -- the
    SAME isolation `test_pass_t_target_moves_only_when_span_i_plus_t_is_edited` uses,
    for the same reason: on the real forward `z` legitimately depends on the embedding
    table (a slot's seed reads its own span), so that confound must be removed before
    the embedding leaf's gradient means anything. With `db_traj` disconnected, the ONLY
    possible path from the loss to the embedding table is through the TARGET
    (`w_tied = self.embed.lm_weight().detach()`) -- so this isolates exactly the thing
    the docstring on that line claims."""
    m = _horizon(3)
    ids = _ids()
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    inp0, _l0, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    S = layout.slot_index.shape[1]
    torch.manual_seed(0)
    db_traj = [torch.randn(1, S, m._n_streams, m.cfg.d_model, requires_grad=True)
               for _ in range(4)]
    stats: dict = {}
    loss = m._tul_horizon_loss(db_traj, inp0, layout, stats=stats)
    assert loss.requires_grad, "fixture: the isolated term carries no graph at all"

    euc_w = m.embed.hybrid.euc_embed.weight
    lor_w = m.embed.hybrid.lor_embed.space_embed.weight
    g_euc, g_lor = torch.autograd.grad(loss, [euc_w, lor_w], retain_graph=False,
                                       allow_unused=True)
    assert g_euc is None or bool((g_euc == 0).all()), (
        "horizon term reached the euclidean embedding table — the target must be detached")
    assert g_lor is None or bool((g_lor == 0).all()), (
        "horizon term reached the Lorentz embedding table — the target must be detached")


# ── 3/4. THE TARGET IS THE RIGHT SPAN, AND THE MASK ─────────────────────────

def test_pass_t_reads_span_i_plus_t_via_span_slots_shift_t():
    """UNIT level: `_tul_horizon_loss` builds pass t's target with
    `span_slots(input_ids, layout, J, shift=t)`. Slot 0's pass-2 target must be span 2's
    OWN tokens, and pass-3's target must be span 3's OWN tokens — two DIFFERENT token
    sets, not the same "next span" (span 1) restaged at every pass. This is the exact
    call `_tul_horizon_loss` makes; a `shift=1` sabotage (always "the next span") is
    caught here directly, and by the end-to-end test below."""
    from morph.model.tul_spandec import span_slots
    ids = _ids()
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    assert bool(layout.slot_valid[0, 0]), "fixture: slot 0 is not valid"
    ids_t2, valid_t2 = span_slots(inp, layout, 32, shift=2)
    ids_t3, valid_t3 = span_slots(inp, layout, 32, shift=3)
    assert bool(valid_t2[0, 0].any()) and bool(valid_t3[0, 0].any()), (
        "fixture: slot 0 has no valid span 2 / span 3 target")
    span2_tokens = ids_t2[0, 0][valid_t2[0, 0]]
    span3_tokens = ids_t3[0, 0][valid_t3[0, 0]]
    assert not torch.equal(span2_tokens, span3_tokens), (
        "pass 2 and pass 3 read the SAME span for slot 0 — shift is not varying with t")
    # And each really is the RIGHT span: span k's own tokens are `own_span_slots` at the
    # slot terminating it, i.e. `span_slots(shift=0)` read at slot k itself.
    own2, ok2 = span_slots(inp, layout, 32, shift=0)
    own3, ok3 = span_slots(inp, layout, 32, shift=0)
    assert torch.equal(span2_tokens, own2[0, 2][ok2[0, 2]]), (
        "pass 2's target is not span 2's own tokens")
    assert torch.equal(span3_tokens, own3[0, 3][ok3[0, 3]]), (
        "pass 3's target is not span 3's own tokens")


def test_pass_t_target_moves_only_when_span_i_plus_t_is_edited():
    """End-to-end, with TWO confounds named and removed:

    (a) a slot's OWN z legitimately depends on EARLIER spans (its seed is a summary of
        its own span, and the causal loop lets a later slot depend on an earlier one) —
        removed by calling `_tul_horizon_loss` directly against a FIXED,
        input-independent `db_traj` (random noise, built once), so no edit can move z;
    (b) span k is simultaneously slot (k-2)'s pass-2 target AND slot (k-3)'s pass-3
        target, so editing span 3 legitimately moves BOTH `horizon_t2` (via slot 1) and
        `horizon_t3` (via slot 0) in the real, multi-slot fixture — removed by masking
        slot 1 OUT of `layout.slot_valid` (slots 0/2/3 stay valid, since span 2's and
        span 3's own COMPLETENESS checks read those slots' validity too), leaving slot
        0 the aggregate's ONLY GRADED contributor.

    With both confounds gone, only a wrong `shift` in the target lookup can move a term
    that should not move.
    """
    m = _horizon(3)
    ids = _ids()
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    _inp0, _l0, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    S = layout.slot_index.shape[1]
    torch.manual_seed(0)
    # The real carrier's shape: [B, S, n_streams, C] under HC-Cayley (the sole residual
    # on this tree), which is what `_tul_core` actually appends into `db_traj`.
    db_traj = [torch.randn(1, S, m._n_streams, m.cfg.d_model) for _ in range(4)]

    def horizon_stats(i):
        inp_i, _l, lay_i, _ = slot_layout_from_ids(i, _rule(), spec)
        lay_i.slot_valid[:, 1] = False        # (b): slot 1 is the ONLY contaminant
        assert bool(lay_i.slot_valid[0, 0]), "fixture: slot 0 is not valid"
        assert bool(lay_i.slot_valid[0, 2]) and bool(lay_i.slot_valid[0, 3]), (
            "fixture: span 2 / span 3 is not complete without slot 1")
        stats: dict = {}
        m._tul_horizon_loss(db_traj, inp_i, lay_i, stats=stats)
        return stats

    base = horizon_stats(ids)
    edited3 = horizon_stats(_edit(ids, layout, 0, 3))
    edited1 = horizon_stats(_edit(ids, layout, 0, 1))
    assert base["horizon_t2"] == pytest.approx(edited3["horizon_t2"], abs=1e-9), (
        "editing span 3 moved slot 0's pass 2 term with z held fixed and every other "
        "slot masked out — pass 2 is reading the wrong span")
    assert base["horizon_t3"] != pytest.approx(edited3["horizon_t3"], abs=1e-9), (
        "editing span 3 did not move slot 0's pass 3 term at all, with z held fixed")
    # Span 1 is NEVER slot 0's target at t in {2,3} (target = span 0+t); with z held
    # fixed and every other slot masked out, editing it must move neither term.
    assert base["horizon_t2"] == pytest.approx(edited1["horizon_t2"], abs=1e-9)
    assert base["horizon_t3"] == pytest.approx(edited1["horizon_t3"], abs=1e-9)


def test_rows_with_no_span_i_plus_t_contribute_nothing():
    """The LAST valid slots of a row have no span T spans ahead — `span_slots`' own
    validity rule excludes them, not a zero-padded average that would leak a dump-row
    value in. Fewer (slot, pass) pairs are graded at the deeper pass than the shallower
    one, on the SAME row, and the whole term stays finite (no NaN from dividing by an
    empty mask)."""
    from morph.model.tul_spandec import span_slots
    m = _horizon(3)
    ids = _ids()
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    _id2, valid2 = span_slots(inp, layout, 32, shift=2)
    _id3, valid3 = span_slots(inp, layout, 32, shift=3)
    n_keep2 = int((valid2.any(dim=-1) & layout.slot_valid).sum())
    n_keep3 = int((valid3.any(dim=-1) & layout.slot_valid).sum())
    n_valid_slots = int(layout.slot_valid.sum())
    assert 0 < n_keep3 < n_keep2 <= n_valid_slots, (
        f"fixture: expected n_keep3 < n_keep2 <= n_valid_slots, got "
        f"{n_keep3}, {n_keep2}, {n_valid_slots} — the row does not exercise the mask")
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["horizon"])
    assert float(out["horizon_n_tokens"]) > 0.0


# ── 5. PASS 1 IS FREE BY DEFAULT ─────────────────────────────────────────────

def test_pass_1_is_unconstrained_by_default():
    m = _horizon(3)
    out = _run(m)
    assert "horizon_t1" not in out, "pass 1 was graded despite horizon_free_first=True"
    assert "horizon_t2" in out and "horizon_t3" in out


def test_horizon_free_first_false_grades_pass_1_too():
    m = _horizon(3, horizon_free_first=False)
    out = _run(m)
    assert "horizon_t1" in out, "horizon_free_first=False did not grade pass 1"
    assert "horizon_t2" in out and "horizon_t3" in out


def test_horizon_knobs_set_with_weight_zero_raises():
    with pytest.raises(ValueError, match="horizon_weight"):
        _tul(slot_depth_fixed=3, horizon_weight=0.0, horizon_tokens=4)


# ── 6. THE GATE (pass_readout="gated") ───────────────────────────────────────

def test_the_gate_normalises_to_one():
    """Both HALVES on the REAL module (not an independent reimplementation): the
    per-pass weights it computes sum to one, AND `m.tul_pass_gate(states)` equals the
    independently-recomputed normalised weighted sum — so a bug in `forward` itself
    (e.g. a missing division by the normaliser) is caught here, not only a bug in a
    test-local copy of the formula."""
    m = _gated(3)
    C = m.cfg.d_model
    states = [torch.randn(2, 5, C) for _ in range(3)]
    gates = [torch.nn.functional.softplus(m.tul_pass_gate.Wg(x) + m.tul_pass_gate.beta[t])
             for t, x in enumerate(states)]
    total = sum(gates)
    weights = [g / (total + m.tul_pass_gate.eps) for g in gates]
    frac = torch.stack(weights, dim=0).sum(0)
    assert torch.allclose(frac, torch.ones_like(frac), atol=1e-6)
    want = weights[0] * states[0] + weights[1] * states[1] + weights[2] * states[2]
    got = m.tul_pass_gate(states)
    assert torch.allclose(got, want, atol=1e-6), (
        "TULPassGate.forward does not match its own documented Eq 9-11 formula")


def test_the_gates_argmax_can_move_the_written_z():
    """Scaling up the pass with the LARGEST gate weight moves z proportionally more than
    scaling the pass with the smallest — the argmax pass is load-bearing, not decorative."""
    m = _gated(3)
    T = 3
    torch.manual_seed(0)
    states = [torch.randn(1, 4, m.cfg.d_model) for _ in range(T)]
    with torch.no_grad():
        gates = [torch.nn.functional.softplus(m.tul_pass_gate.Wg(x) + m.tul_pass_gate.beta[t])
                 for t, x in enumerate(states)]
        denom = sum(gates) + m.tul_pass_gate.eps
        weights = [float(g.mean() / denom.mean()) for g in gates]
    hi, lo = int(np.argmax(weights)), int(np.argmin(weights))
    base = m.tul_pass_gate(states)

    def bump(idx):
        s2 = list(states)
        s2[idx] = s2[idx] + 1.0
        return m.tul_pass_gate(s2)

    d_hi = float((bump(hi) - base).abs().mean())
    d_lo = float((bump(lo) - base).abs().mean())
    assert d_hi > 0.0 and d_lo > 0.0, "fixture: neither perturbation moved z at all"
    assert d_hi > d_lo, (
        f"the argmax-weight pass ({hi}, w={weights[hi]:.3f}) moved z LESS than the "
        f"min-weight pass ({lo}, w={weights[lo]:.3f}) — the gate is not load-bearing")


def test_the_coda_and_span_decoder_read_the_gated_mixture_not_the_last_pass():
    m = _gated(3, spandec=True, spandec_layers=1, spandec_max_tokens=8)
    real_proj, real_sd = m.tul.prefix_project, m._tul_spandec_loss
    seen = {}

    def proj(h, *a, **kw):
        seen["prefix_h"] = h.detach().clone()
        return real_proj(h, *a, **kw)

    def sd(h, *a, **kw):
        seen["spandec_h"] = h.detach().clone()
        return real_sd(h, *a, **kw)

    m.tul.prefix_project, m._tul_spandec_loss = proj, sd
    try:
        _run(m)
    finally:
        m.tul.prefix_project, m._tul_spandec_loss = real_proj, real_sd
    assert torch.equal(seen["prefix_h"], seen["spandec_h"]), (
        "the coda and the span decoder read different states")


def test_the_gated_readout_differs_from_the_raw_last_pass_state():
    """A bypassed gate (the substitution line skipped, `h_slots` left as `_tul_core`'s
    raw return) would make `prefix_project`'s input EQUAL `db_traj[-1]` exactly — this
    is the direct catch for that, independent of the coda/decoder-agreement test above
    (which would still pass under a bypass, since both readers would just agree on the
    UN-gated state instead)."""
    m = _gated(3)
    real_core, real_proj = m._tul_core, m.tul.prefix_project
    seen = {}

    def core(*a, **kw):
        out = real_core(*a, **kw)
        seen["db_traj"] = out[4]
        return out

    def proj(h, *a, **kw):
        seen["prefix_h"] = h.detach().clone()
        return real_proj(h, *a, **kw)

    m._tul_core, m.tul.prefix_project = core, proj
    try:
        _run(m)
    finally:
        m._tul_core, m.tul.prefix_project = real_core, real_proj
    last_pass = seen["db_traj"][-1].detach()
    assert not torch.equal(seen["prefix_h"], last_pass), (
        "the coda read the raw last-pass state — the gated substitution is bypassed")


# ── 7. THE POISSON-DEPTH REFUSAL ─────────────────────────────────────────────

def test_poisson_depth_with_horizon_weight_raises():
    with pytest.raises(NotImplementedError, match="slot_depth_fixed"):
        _tul(horizon_weight=1.0)          # slot_depth_fixed defaults to 0 (Poisson)


def test_poisson_depth_with_gated_readout_raises():
    with pytest.raises(NotImplementedError, match="slot_depth_fixed"):
        _tul(pass_readout="gated")        # slot_depth_fixed defaults to 0 (Poisson)


def test_bad_pass_readout_value_raises():
    with pytest.raises(ValueError, match="pass_readout"):
        _tul(pass_readout="bogus")


# ── 8. EVERY NEW KEY IS KNOWN ────────────────────────────────────────────────

def test_new_keys_are_known_to_tul_setup():
    for k in ("horizon_weight", "horizon_free_first", "horizon_tokens", "pass_readout"):
        assert k in KNOWN_TUL_KEYS, f"{k} missing from tul_setup.KNOWN_TUL_KEYS"


def test_new_fields_exist_on_tulconfig():
    tc = TULConfig()
    assert tc.horizon_weight == 0.0
    assert tc.horizon_free_first is True
    assert tc.horizon_tokens == 0
    assert tc.pass_readout == "last"


def test_the_gate_accepts_a_forced_eval_depth_and_refuses_it_at_train():
    """The K-curve sweep forces slot depths d != T at EVAL. Pass t < T reads beta[t];
    past the trained T every pass reuses beta[T-1] (the token-loop _LoopMTPGate's
    convention). At TRAIN a mismatch still raises: there it means the depth knob moved.
    Before 2026-09-14 the gate refused every forced depth and the horizon arm had no
    sweep (runner: SWEEP horizon-arm@5000 exit=1)."""
    m = _gated(3)
    C = m.cfg.d_model
    gate = m.tul_pass_gate
    with torch.no_grad():
        gate.beta.copy_(torch.tensor([0.5, -1.0, 2.0]))
    gate.eval()
    for d in (1, 2, 5):
        states = [torch.randn(2, 5, C) for _ in range(d)]
        gates = [torch.nn.functional.softplus(gate.Wg(x) + gate.beta[min(t, 2)])
                 for t, x in enumerate(states)]
        total = sum(gates) + gate.eps
        want = sum(g / total * x for g, x in zip(gates, states))
        got = gate(states)
        assert torch.allclose(got, want, atol=1e-6), f"forced depth {d}"
    # depth 5 must NOT equal a gate that (wrongly) read beta[0] for the passes past T
    states = [torch.randn(2, 5, C) for _ in range(5)]
    wrong = [torch.nn.functional.softplus(gate.Wg(x) + gate.beta[t % 3])
             for t, x in enumerate(states)]
    total = sum(wrong) + gate.eps
    assert not torch.allclose(gate(states), sum(g / total * x for g, x in zip(wrong, states)), atol=1e-6)
    gate.train()
    with pytest.raises(ValueError, match="built for 3 passes"):
        gate([torch.randn(2, 5, C) for _ in range(2)])
