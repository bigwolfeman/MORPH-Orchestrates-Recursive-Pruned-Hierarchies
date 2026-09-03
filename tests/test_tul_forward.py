"""TUL forward contract — the PAID loop (docs/tul-paid-loop-recipe.md; docs/tul-spec.md
§3.2, §3.4, §5, §7.2; runtime-invariants §6b).

Protects the invariants that a silent break would hide inside the loss curve:
``slot_layout=None`` is bit-identical to the baseline, the packed row runs through the
SAME per-sample core loop as a plain row (every position pays the loop — nothing is
gathered, projected or scattered), pad slots never receive loss, and ``slot_id`` can
never be emitted. CPU only (``use_kernels=False``), tiny config, no tokenizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, TULSlots, bag_mean
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


TUL_PARAMS = ("E_slot", "E_mask", "W_sent.weight")


def _tul_param(m: MORPHTransformer, name: str) -> torch.Tensor:
    obj = m.tul
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


# ── invariant: slot_layout=None is the baseline, bit for bit ─────────────────

def test_tul_params_do_not_perturb_the_plain_path():
    """runtime-invariants §6b: `slot_layout=None` is bit-identical to today's forward.

    Building the TUL parameters must not move a single weight, gradient or RNG draw —
    every TUL init is deterministic (W_sent draws from a private generator) and they are
    constructed last, so a TUL model and a baseline model built with the same seed are
    the same model on the plain path.
    """
    x = torch.randint(0, V, (2, 32))
    y = torch.randint(0, V, (2, 32))
    outs, grads, params = [], [], []
    for tul in (None, TULConfig(prefix_k=2, slot_id=4)):
        m = _model(tul, dropout=0.1)
        m.train()
        torch.manual_seed(99)
        out = m(x, labels=y)
        out["loss"].backward()
        outs.append(out["loss"].detach().clone())
        named = [(n, p) for n, p in sorted(m.named_parameters()) if not n.startswith("tul.")]
        params.append(torch.cat([p.detach().flatten() for _, p in named]))
        grads.append(torch.cat([p.grad.flatten() for _, p in named if p.grad is not None]))
        for n, p in m.named_parameters():
            if n.startswith("tul."):
                assert p.grad is None, f"{n} must receive NO gradient on the plain path"
    assert torch.equal(outs[0], outs[1]), f"loss moved: {outs[0].item()} vs {outs[1].item()}"
    assert torch.equal(params[0], params[1]), "TUL construction perturbed the init RNG"
    assert torch.equal(grads[0], grads[1]), "TUL construction changed the baseline gradients"


def test_forward_without_tul_config_rejects_a_layout():
    m = _model(None)
    x, y, layout, _ = _batch(_spec())
    with pytest.raises(RuntimeError, match="tul="):
        m(x, labels=y, slot_layout=layout)


def test_bag_size_and_slot_layout_are_mutually_exclusive():
    m = _model(TULConfig())
    x, y, layout, _ = _batch(_spec())
    with pytest.raises(ValueError, match="mutually exclusive"):
        m(x, labels=y, bag_size=2, slot_layout=layout)


# ── §3.2 the slot input ──────────────────────────────────────────────────────

def test_bag_mean_is_the_mean_over_the_span_tokens_only():
    sig = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    bag = torch.tensor([[0, 0, 2, 1, 1, 2]])          # 2 = the dump bin (n_bags=2)
    sel = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]])   # positions 2,5 are slots
    out = bag_mean(sig, bag, sel, n_bags=2)
    assert out.shape == (1, 3, 4)
    assert torch.allclose(out[0, 0], (sig[0, 0] + sig[0, 1]) / 2)
    assert torch.allclose(out[0, 1], (sig[0, 3] + sig[0, 4]) / 2)
    assert torch.all(out[0, 2] == 0), "the dump bin must be exactly zero"


def test_slot_input_equals_e_slot_plus_the_span_mean_in_bag_mean_mode():
    """Spec §3.2: slot input embedding = E_slot + mean_j embed(t_j) over its span.

    ``slot_seed="boundary"`` (the shipped default) is pinned in tests/test_slot_seed.py.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4, slot_seed="bag_mean"))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    with torch.no_grad():
        # E_slot is zero-initialised until the activation step; give it a real value or
        # this test would pass even if E_slot were dropped from the sum.
        m.tul.E_slot.normal_(std=1.0)
        emb = m.embed(x)
        got = m.tul.slot_input(emb, layout, add_e_slot=True)
    b, s = 0, 0
    first = int(layout.slot_index[b, s])
    span = [p for p in range(layout.l_total)
            if int(layout.bag_id[b, p]) == s and not bool(layout.slot_mask[b, p])]
    assert span, "span 0 must own token positions"
    want = emb[b, span].mean(0) + m.tul.E_slot
    for k in range(spec.prefix_k):
        assert torch.allclose(got[b, first + k], want, atol=1e-6), (
            f"prefix position {k} must carry the same slot input (spec §3.1 prelude)")
    tok = [p for p in range(layout.l_total) if not bool(layout.slot_mask[b, p])]
    assert torch.equal(got[b, tok], emb[b, tok]), "token positions must be untouched"


def test_gather_scatter_round_trip_on_the_hc_carrier():
    """The FM planner's write path (``prefix_project`` → ``scatter_positions``)."""
    from morph.model.tul import gather_positions, scatter_positions

    B, L, n, C = 2, 9, 4, 6
    x = torch.randn(B, L, n, C)
    idx = torch.tensor([[1, 4, 7], [0, 3, 9]])         # 9 == the dump row for row 1
    xp = torch.cat([x, x.new_zeros(B, 1, n, C)], dim=1)
    g = gather_positions(xp, idx)
    assert torch.equal(g[0, 1], x[0, 4])
    assert torch.all(g[1, 2] == 0), "the dump row must gather zeros"
    back = scatter_positions(x, idx, g)
    assert torch.equal(back, x), "gather → scatter must be the identity"
    other = scatter_positions(x, idx, torch.full_like(g, 7.0))
    assert torch.all(other[1, 0] == 7.0) and torch.equal(other[1, 1], x[1, 1])
    assert torch.equal(other[1, 9 - 1], x[1, 8]), "a dump-row write must not touch the row"


# ── the paid loop: the packed row runs through the SAME core as a plain row ──

def test_tul_forward_runs_the_whole_packed_row_through_the_core():
    """docs/tul-paid-loop-recipe.md: the slot is a looped POSITION, not a gathered state.

    The TUL forward must call the per-sample core loop (`_core_region`) exactly once,
    on the FULL packed row `[B, L_total, streams, d]` — every token and every slot
    position pays the loop. A gather/scatter regression would show up here as a
    `[B, S, …]` call, and a coreless regression as no call at all.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    seen: list[tuple] = []
    orig = m._core_region

    def spy(x_in, x0, bigram_emb, input_ids=None, **kw):
        seen.append((tuple(x_in.shape), tuple(x0.shape),
                     None if input_ids is None else tuple(input_ids.shape)))
        return orig(x_in, x0, bigram_emb, input_ids, **kw)

    m._core_region = spy
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert len(seen) == 1, f"the core region must run exactly once, ran {len(seen)}x"
    x_shape, x0_shape, ids_shape = seen[0]
    B, L = x.shape
    assert L == layout.l_total
    assert x_shape[:2] == (B, L) and x0_shape[:2] == (B, L), (
        f"the core must see the FULL packed row (B={B}, L={L}); it saw {x_shape}")
    assert x_shape[-1] == m.cfg.d_model
    assert ids_shape == (B, L), "the core needs the packed ids (bigram / injection)"
    assert torch.isfinite(out["loss"])


def test_layer_passes_charge_every_packed_position_for_the_loop():
    """`layer_passes` is the cost accountant behind the wall-clock claims. Under the paid
    loop every one of the L_total positions runs prelude + n_core×mean_depth + coda, so
    the per-TOKEN figure is at or ABOVE the dense baseline (slots are extra positions).
    The old slot-only arm asserted the opposite; a regression to it flips this test."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    B, L = x.shape
    c = m.cfg
    want = B * L * (c.n_prelude + c.n_coda + c.n_core * c.mean_depth)
    assert float(out["layer_passes"]) == pytest.approx(want)
    assert int(out["n_tokens"]) == int((~layout.slot_mask).sum())
    per_tok = float(out["layer_passes"]) / float(out["n_tokens"])
    dense = c.n_prelude + c.n_coda + c.n_core * c.mean_depth
    assert per_tok >= dense, f"paid loop cannot be cheaper per token than dense: {per_tok}"


def test_plan_ablations_are_refused_on_the_paid_loop():
    """There is no separate plan tensor to zero or shuffle; a `val/plan_worth_*` that is
    0 by construction must never be reported, so the forward raises instead."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        for mode in ("zero", "shuffle"):
            with pytest.raises(ValueError, match="plan_mode"):
                m._forward_tul(x, y, layout, plan_mode=mode)


def test_eval_forward_is_deterministic_at_the_mean_depth():
    """Eval runs the loop at exactly `mean_depth` for every sample (no Poisson draw), so
    two eval forwards on the same packed row are bit-identical — the property every
    same-rows depth sweep in lab/divergence/ rests on."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        a = m(x, labels=y, slot_layout=layout)["loss"]
        b = m(x, labels=y, slot_layout=layout)["loss"]
    assert torch.equal(a, b)


# ── §3.1 / invariant 4: the slot id ──────────────────────────────────────────

def test_slot_id_logit_is_minus_inf_and_never_top_1():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, _y, layout, _ = _batch(_spec())
    with torch.no_grad():
        logits = m(x, slot_layout=layout)["logits"]
    assert torch.isinf(logits[..., 4]).all() and (logits[..., 4] < 0).all(), (
        "invariant 4: the slot_id logit must be −inf everywhere")
    assert (logits.argmax(-1) != 4).all(), "slot_id must never be the top-1 token"


def test_masked_vocab_row_receives_zero_gradient():
    """Spec §12: 'fused CE with slot_id masked: gradient to that row is zero.'"""
    from morph.model.fused_ce import fused_linear_cross_entropy

    torch.manual_seed(0)
    xf = torch.randn(16, 8, requires_grad=True)
    w = torch.randn(V, 8, requires_grad=True)
    labels = torch.randint(5, V, (16,))
    fused_linear_cross_entropy(xf, w, labels, mask_token_id=4).backward()
    assert torch.all(w.grad[4] == 0), "the masked row must receive exactly zero gradient"
    assert w.grad[5].abs().sum() > 0, "control: unmasked rows do receive gradient"
    # and the loss must equal a reference that removes the row from the partition function
    ref_logits = (xf.detach() @ w.detach().t())
    ref_logits[:, 4] = float("-inf")
    ref = torch.nn.functional.cross_entropy(ref_logits.float(), labels)
    got = fused_linear_cross_entropy(xf.detach(), w.detach(), labels, mask_token_id=4)
    assert torch.allclose(ref, got, atol=1e-5), f"masked CE {got} != reference {ref}"


# ── §5 the loss groups ───────────────────────────────────────────────────────

def test_pad_slots_receive_no_loss_and_the_denominator_is_distinct_targets():
    """Invariant 3 + spec §5's half-weight double label."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    g = m._tul_group_losses(torch.randn(*x.shape, m.cfg.d_model), y, layout)
    n_valid = int(layout.slot_valid.sum())
    assert int(g["n_emit"]) == n_valid, (
        f"{int(g['n_emit'])} emitting labels for {n_valid} real slots — pad slots leaked "
        f"into the loss")
    assert int(g["n_plast"]) == n_valid
    n_labels = int((y != -100).sum())
    assert int(g["n_main"]) == n_labels - 2 * n_valid, (
        "the two half-weight groups must be removed from the main group exactly once")
    assert float(g["n_targets"]) == pytest.approx(n_labels - n_valid), (
        "the loss denominator must be the number of DISTINCT target tokens (spec §5)")


def test_weighted_ce_equals_the_explicit_half_weight_combination():
    """The single weighted CE call IS the spec §5 formula, to fp tolerance.

    The training loss folds the 0.5 weights into the kernel's reduction; this pins that
    against the explicit three-group combination it replaces, so the optimisation can
    never quietly change the objective.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4, emit_weight=0.5, plast_weight=0.5))
    x, y, layout, _ = _batch(_spec())
    h = torch.randn(*x.shape, m.cfg.d_model)
    g = m._tul_group_losses(h, y, layout)
    explicit = (g["ce_main"] * g["n_main"]
                + 0.5 * (g["ce_plast"] * g["n_plast"] + g["ce_emit"] * g["n_emit"])) \
        / (g["n_main"] + 0.5 * (g["n_plast"] + g["n_emit"]))
    assert torch.allclose(g["loss"], explicit, atol=1e-5), (
        f"weighted CE {float(g['loss']):.6f} != explicit combination {float(explicit):.6f}")


def test_shipped_weights_drop_the_emit_term_and_keep_the_plain_token_label():
    """base.yaml ships ``emit_weight: 0.0, plast_weight: 1.0``: the slot's own emit label
    carries NO loss and the boundary token keeps its ordinary weight-1 label. With those
    weights the training loss must equal the plain CE over the token labels alone."""
    m = _model(TULConfig(prefix_k=2, slot_id=4, emit_weight=0.0, plast_weight=1.0))
    x, y, layout, _ = _batch(_spec())
    h = torch.randn(*x.shape, m.cfg.d_model)
    g = m._tul_group_losses(h, y, layout)
    explicit = (g["ce_main"] * g["n_main"] + g["ce_plast"] * g["n_plast"]) \
        / (g["n_main"] + g["n_plast"])
    assert torch.allclose(g["loss"], explicit, atol=1e-5)
    assert float(g["n_targets"]) == pytest.approx(float(g["n_main"] + g["n_plast"]))
    assert torch.allclose(g["loss"], g["ce_tokens"], atol=1e-5), (
        "with emit_weight=0 the training loss IS the token CE")


def test_weights_in_the_fused_ce_reduce_to_the_plain_mean():
    from morph.model.fused_ce import fused_linear_cross_entropy

    torch.manual_seed(0)
    xf = torch.randn(32, 8)
    w = torch.randn(V, 8)
    labels = torch.randint(0, V, (32,))
    labels[::5] = -100
    plain = fused_linear_cross_entropy(xf, w, labels)
    ones = fused_linear_cross_entropy(xf, w, labels, weights=torch.ones(32))
    assert torch.allclose(plain, ones, atol=1e-6), "weights=1 must be the plain mean"
    # and a weight vector must reproduce a hand-computed weighted mean
    rw = torch.rand(32) + 0.1
    got = fused_linear_cross_entropy(xf, w, labels, weights=rw)
    per = torch.nn.functional.cross_entropy((xf @ w.t()).float(), labels,
                                            ignore_index=-100, reduction="none")
    valid = (labels != -100).float()
    want = (per * rw * valid).sum() / (rw * valid).sum()
    assert torch.allclose(got, want, atol=1e-5), f"weighted CE {got} != reference {want}"


def test_weighted_ce_gradient_matches_the_reference():
    from morph.model.fused_ce import fused_linear_cross_entropy

    torch.manual_seed(1)
    labels = torch.randint(0, V, (24,))
    labels[::7] = -100
    rw = torch.rand(24) + 0.1
    grads = []
    for fused in (True, False):
        torch.manual_seed(2)
        w = (torch.randn(V, 8) * 0.1).requires_grad_(True)
        torch.manual_seed(3)
        xf = torch.randn(24, 8).requires_grad_(True)
        if fused:
            loss = fused_linear_cross_entropy(xf, w, labels, weights=rw)
        else:
            per = torch.nn.functional.cross_entropy((xf @ w.t()).float(), labels,
                                                    ignore_index=-100, reduction="none")
            valid = (labels != -100).float()
            loss = (per * rw * valid).sum() / (rw * valid).sum()
        loss.backward()
        grads.append((xf.grad.clone(), w.grad.clone()))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5), "grad wrt x differs"
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-5), "grad wrt w differs"


def _padded_batch(B=3):
    """A layout that really does contain PAD slots (few boundaries, generous budget)."""
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=8, slot_id=4)
    rule = BoundaryRule(is_boundary=np.zeros(V, dtype=bool), min_span=4, span_cap=16, eos_id=0)
    rng = np.random.default_rng(5)
    ids = rng.integers(5, V, size=(B, 200))
    ids[ids == spec.slot_id] = 5
    x, y, layout, _ = slot_layout_from_ids(ids.astype(np.int64), rule, spec)
    n_valid = int(layout.slot_valid.sum())
    assert n_valid < B * spec.max_slots, "this fixture must contain pad slots"
    assert (~layout.slot_valid).any()
    return x, y, layout, spec


def test_pad_slots_are_excluded_from_every_loss_group():
    """Invariant 3, with a layout that actually HAS pad slots.

    A pad slot's ``slot_index`` is 0, so a missing validity mask would silently point
    its t_last at the previous ROW's last position — a real label, silently trained on.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    x, y, layout, spec = _padded_batch()
    n_valid = int(layout.slot_valid.sum())
    g = m._tul_group_losses(torch.randn(*x.shape, m.cfg.d_model), y, layout)
    assert int(g["n_emit"]) == n_valid, (
        f"{int(g['n_emit'])} emitting labels for {n_valid} real slots — pad slots leaked in")
    assert int(g["n_plast"]) == n_valid, (
        f"{int(g['n_plast'])} t_last labels for {n_valid} real slots — a pad slot's "
        f"index 0 pointed at another row's token")
    assert int(g["n_main"]) == int((y != -100).sum()) - 2 * n_valid
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"])


def test_counterfactual_is_the_two_predictions_of_the_same_token():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    assert torch.allclose(out["first_tok_counterfactual"],
                          out["ce_first_tok_plain"] - out["ce_first_tok"])
    lo = min(float(out["ce_first_tok"]), float(out["ce_tokens"]))
    hi = max(float(out["ce_first_tok"]), float(out["ce_tokens"]))
    assert lo > 0 and hi < 20


# ── §3.4 token-state dropout ─────────────────────────────────────────────────

def test_token_state_dropout_only_touches_token_positions():
    m = _model(TULConfig(prefix_k=2, slot_id=4, token_state_dropout=1.0))
    x, _y, layout, _ = _batch(_spec())
    h = torch.randn(x.shape[0], layout.l_total, 4, m.cfg.d_model)
    out, keep = m.tul.apply_token_dropout(h, layout, training=True)
    tok = ~layout.slot_mask
    assert torch.equal(out[layout.slot_mask], h[layout.slot_mask]), (
        "spec §3.4: dropout is NEVER applied to slot positions")
    assert torch.allclose(out[tok], m.tul.E_mask.expand_as(out[tok])), (
        "at p=1 every token state must become E_mask")
    assert torch.equal(keep.squeeze(-1).bool(), layout.slot_mask)


def test_token_state_dropout_is_off_at_eval_and_at_p_zero():
    for p, training in ((1.0, False), (0.0, True)):
        m = _model(TULConfig(prefix_k=2, slot_id=4, token_state_dropout=p))
        x, _y, layout, _ = _batch(_spec())
        h = torch.randn(x.shape[0], layout.l_total, 4, m.cfg.d_model)
        out, keep = m.tul.apply_token_dropout(h, layout, training=training)
        assert torch.equal(out, h) and keep is None


# ── shapes, parameters, checkpoints ──────────────────────────────────────────

def test_prefix_k_shapes_and_no_prefix_projection_on_the_paid_loop():
    """`W_prefix` is the FM planner's write path; the paid loop builds none of it."""
    for k in (1, 2, 3):
        spec = _spec(prefix_k=k)
        x, y, layout, _ = _batch(spec)
        assert x.shape[1] == spec.seq_len + k * spec.max_slots
        m = _model(TULConfig(prefix_k=k, slot_id=4))
        m.eval()
        with torch.no_grad():
            out = m(x, labels=y, slot_layout=layout)
        assert torch.isfinite(out["loss"])
        assert m.tul.W_prefix is None, "the paid loop must not build W_prefix"
        assert not any(n.startswith("tul.W_prefix") for n, _ in m.named_parameters())
        with_fm = TULSlots(m.cfg.d_model, TULConfig(prefix_k=k, slot_id=4), with_prefix=True)
        assert with_fm.W_prefix.shape == (k, m.cfg.d_model, m.cfg.d_model)


def test_prefix_project_refuses_a_model_without_the_projection():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    x, _y, layout, _ = _batch(_spec())
    h = torch.randn(x.shape[0], layout.max_slots, 4, m.cfg.d_model)
    with pytest.raises(RuntimeError, match="W_prefix"):
        m.tul.prefix_project(h, layout, layout.l_total)


def test_layout_prefix_k_must_match_the_model():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    x, y, layout, _ = _batch(_spec(prefix_k=1))
    with pytest.raises(ValueError, match="prefix_k"):
        m(x, labels=y, slot_layout=layout)


def test_train_step_reaches_every_new_parameter():
    m = _model(TULConfig(prefix_k=2, slot_id=4, token_state_dropout=0.5))
    m.train()
    x, y, layout, _ = _batch(_spec())
    out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for name in TUL_PARAMS:
        g = _tul_param(m, name).grad
        assert g is not None, f"{name} received no gradient"
        assert torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} gradient is zero/NaN"
    core_grads = [p.grad for p in m.core.parameters() if p.grad is not None]
    assert core_grads and sum(float(g.abs().sum()) for g in core_grads) > 0, (
        "the core received no gradient — the paid loop is disconnected from the loss")


def test_checkpoint_round_trip_carries_the_tul_parameters():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    with torch.no_grad():
        for name in TUL_PARAMS:
            _tul_param(m, name).normal_()
    sd = m.state_dict()
    assert {f"tul.{n}" for n in TUL_PARAMS} <= set(sd), (
        f"TUL parameters missing from the state dict: {sorted(k for k in sd if 'tul' in k)}")
    assert "tul.W_prefix" not in sd
    m2 = _model(TULConfig(prefix_k=2, slot_id=4), seed=777)
    missing, unexpected = m2.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected}"
    assert not missing, f"missing keys {missing}"
    for name in TUL_PARAMS:
        assert torch.equal(_tul_param(m, name), _tul_param(m2, name)), f"{name} did not load"
    x, y, layout, _ = _batch(_spec())
    m.eval(), m2.eval()
    with torch.no_grad():
        assert torch.equal(m(x, labels=y, slot_layout=layout)["loss"],
                           m2(x, labels=y, slot_layout=layout)["loss"])


def test_e_slot_activation_init_is_the_embedding_mean():
    """Spec §5 / Block Transformer §3.7: init the new position as the average token."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    assert torch.all(m.tul.E_slot == 0), "E_slot starts at zero until activation"
    m.tul.init_at_activation(m.embed.lm_weight())
    assert torch.allclose(m.tul.E_slot, m.embed.lm_weight().mean(0), atol=1e-6)
    # The FM planner's projection starts as the identity so the extra coda position
    # costs nothing at activation (the same init contract, on the module that has it).
    with_fm = TULSlots(m.cfg.d_model, TULConfig(prefix_k=2, slot_id=4), with_prefix=True)
    with_fm.init_at_activation(m.embed.lm_weight())
    assert torch.all(with_fm.W_prefix[0] == torch.eye(m.cfg.d_model))


def test_gather_valid_zeroes_invalid_rows():
    """An invalid (pad) slot's ``slot_index`` is 0 — a REAL token position. Anything that
    gathers slot states (the FM planner, the lab probes) must zero those rows, or a
    reduction over the slot axis silently counts another position's content."""
    from morph.model.tul import gather_valid

    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3) + 1.0
    index = torch.tensor([[4, 0, 0], [1, 5, 0]])
    valid = torch.tensor([[True, False, False], [True, True, False]])
    g = gather_valid(x, index, valid)
    assert torch.equal(g[0, 0], x[0, 4]) and torch.equal(g[1, 1], x[1, 5])
    assert torch.all(g[0, 1:] == 0) and torch.all(g[1, 2] == 0), (
        "invalid slots gathered real content instead of zeros")
