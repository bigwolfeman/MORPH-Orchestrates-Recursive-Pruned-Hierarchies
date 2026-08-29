"""TUL forward contract (docs/tul-spec.md §3.2-§3.4, §5, §7.2; runtime-invariants §6b).

Protects the invariants that a silent break would hide inside the loss curve:
``slot_layout=None`` is bit-identical to the baseline, the per-slot depth really is a
masked update, pad slots never receive loss, and ``slot_id`` can never be emitted.
CPU only (``use_kernels=False``), tiny config, no tokenizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, bag_mean, compact_index
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


# ── invariant: slot_layout=None is the baseline, bit for bit ─────────────────

def test_tul_params_do_not_perturb_the_plain_path():
    """runtime-invariants §6b: `slot_layout=None` is bit-identical to today's forward.

    Building the TUL parameters must not move a single weight, gradient or RNG draw —
    all three inits are deterministic and they are constructed last, so a TUL model and
    a baseline model built with the same seed are the same model on the plain path.
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


# ── §3.2 the slot input is the span's bag-mean ───────────────────────────────

def test_bag_mean_is_the_mean_over_the_span_tokens_only():
    sig = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    bag = torch.tensor([[0, 0, 2, 1, 1, 2]])          # 2 = the dump bin (n_bags=2)
    sel = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]])   # positions 2,5 are slots
    out = bag_mean(sig, bag, sel, n_bags=2)
    assert out.shape == (1, 3, 4)
    assert torch.allclose(out[0, 0], (sig[0, 0] + sig[0, 1]) / 2)
    assert torch.allclose(out[0, 1], (sig[0, 3] + sig[0, 4]) / 2)
    assert torch.all(out[0, 2] == 0), "the dump bin must be exactly zero"


def test_slot_input_equals_e_slot_plus_the_span_mean():
    """Spec §3.2: slot input embedding = E_slot + mean_j embed(t_j) over its span."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
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


# ── §3.3 the core: gather/scatter and the masked per-slot depth ──────────────

def test_gather_scatter_round_trip_on_the_hc_carrier():
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


def _force_depths(model, table):
    model._sample_slot_depths = lambda layout, device: table.to(device)


def test_per_slot_depth_is_a_masked_update_not_a_gather():
    """Invariant 2: slot i's trajectory must not depend on any LATER slot's depth.

    Causal attention means slot 0 never reads slot 1, so raising slot 1's depth must
    leave slot 0's looped state bit-identical. A per-position gather (the thing the
    invariant forbids) would change what the frozen slots serve and break this.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    spec = _spec()
    x, y, layout, _ = _batch(spec, B=1)
    n_valid = int(layout.slot_valid.sum())
    assert n_valid >= 3, "need ≥3 real slots for this test"

    with torch.no_grad():
        front = m._tul_front(x, layout)

    def run(depths):
        _force_depths(m, depths)
        with torch.no_grad():
            return m._tul_core(*front, layout)[1]

    base = layout.slot_valid.long().clone()
    d_a = base.clone()
    d_a[:] = 1
    d_b = d_a.clone()
    d_b[0, n_valid - 1] = 3                       # deepen the LAST slot only
    h_a, h_b = run(d_a), run(d_b)
    assert torch.equal(h_a[:, : n_valid - 1], h_b[:, : n_valid - 1]), (
        "earlier slots changed when a later slot's depth changed — the mask is wrong")
    assert not torch.equal(h_a[:, n_valid - 1], h_b[:, n_valid - 1]), (
        "the deepened slot did not change: the depth mask never fired (control)")


def test_core_runs_with_the_gla_retention_carry():
    """base.yaml ships `retention: true`, so the slot loop must carry GLA state.

    Also pins the known behaviour that pad slots take part in that carry: they sit at
    the END of the compact sequence, so causal attention never lets a real slot read
    them WITHIN an iteration, and their content is a constant (the zero dump row) that
    carries no row information ACROSS iterations. Changing a pad's gathered content must
    therefore not move any real slot's state.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4), retention=True, retention_layers=(1,),
               retention_heads=2, retention_chunk=8, retention_carry=True)
    m.eval()
    x, y, layout, spec = _padded_batch(B=2)
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"]), "the GLA carry produced a non-finite loss"
    n_valid_row = layout.slot_valid.sum(1)
    with torch.no_grad():
        front = m._tul_front(x, layout)
        h_a = m._tul_core(*front, layout)[1]
        moved = type(layout)(
            slot_mask=layout.slot_mask, bag_id=layout.bag_id,
            slot_index=torch.where(layout.slot_valid, layout.slot_index, 3),
            slot_valid=layout.slot_valid, prefix_k=layout.prefix_k)
        h_b = m._tul_core(*front, moved)[1]
    for b in range(x.shape[0]):
        nv = int(n_valid_row[b])
        assert torch.equal(h_a[b, :nv], h_b[b, :nv]), (
            f"row {b}: a pad slot's gathered content changed a REAL slot's state")


def test_uniform_per_slot_depth_reduces_to_a_plain_loop():
    """Spec §12: depth d on every slot == d unmasked iterations."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec(), B=1)
    n_valid = int(layout.slot_valid.sum())
    _force_depths(m, torch.where(layout.slot_valid, 3, 1))
    with torch.no_grad():
        xa, x0a, bga = m._tul_front(x, layout)
        _, h_masked, _, _, *_ = m._tul_core(xa, x0a, bga, layout)
        # every slot at depth 3 → nothing is ever frozen
        _force_depths(m, torch.full_like(layout.slot_valid, 3, dtype=torch.long))
        _, h_plain, _, _, *_ = m._tul_core(xa, x0a, bga, layout)
    assert torch.equal(h_masked[:, :n_valid], h_plain[:, :n_valid]), (
        "masked update over real slots must equal the unmasked loop at uniform depth")


def test_batch_of_mixed_depths_matches_separate_single_row_runs():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    spec = _spec()
    x, y, layout, _ = _batch(spec, B=2, seed=3)
    depths = torch.where(layout.slot_valid, 1, 1)
    depths[1] = torch.where(layout.slot_valid[1], 3, 1)      # row 1 runs deeper
    _force_depths(m, depths)
    with torch.no_grad():
        xa, x0a, bga = m._tul_front(x, layout)
        _, h_both, _, _, *_ = m._tul_core(xa, x0a, bga, layout)
    for b in range(2):
        row = type(layout)(
            slot_mask=layout.slot_mask[b: b + 1], bag_id=layout.bag_id[b: b + 1],
            slot_index=layout.slot_index[b: b + 1], slot_valid=layout.slot_valid[b: b + 1],
            prefix_k=layout.prefix_k)
        _force_depths(m, depths[b: b + 1])
        with torch.no_grad():
            xr, x0r, bgr = m._tul_front(x[b: b + 1], row)
            _, h_one, _, _, *_ = m._tul_core(xr, x0r, bgr, row)
        nv = int(row.slot_valid.sum())
        assert torch.allclose(h_both[b, :nv], h_one[0, :nv], atol=1e-5), (
            f"row {b} differs between the batched and the single-row run")


# ── §3.4 / §5 the head: slot_id, pad slots, the double label ─────────────────

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
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    x, y, layout, _ = _batch(_spec())
    h = torch.randn(*x.shape, m.cfg.d_model)
    g = m._tul_group_losses(h, y, layout)
    explicit = (g["ce_main"] * g["n_main"]
                + 0.5 * (g["ce_plast"] * g["n_plast"] + g["ce_emit"] * g["n_emit"])) \
        / (g["n_main"] + 0.5 * (g["n_plast"] + g["n_emit"]))
    assert torch.allclose(g["loss"], explicit, atol=1e-5), (
        f"weighted CE {float(g['loss']):.6f} != explicit combination {float(explicit):.6f}")


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


# ── §7.1 arms and §7.2 metrics ───────────────────────────────────────────────

def test_compact_index_moves_tokens_to_the_front_in_order():
    mask = torch.tensor([[False, True, True, False, False, True]])
    idx = compact_index(mask)
    assert idx[0].tolist() == [0, 3, 4, 6, 6, 6], f"got {idx[0].tolist()}"


def test_plan_nats_path_runs_and_differs_from_the_normal_path():
    """Spec §7.2: CE with the slots gathered OUT of the coda sequence."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        out = m.tul_forward_with_plan_nats(x, y, layout)
    assert "ce_tokens_no_slots" in out and torch.isfinite(out["ce_tokens_no_slots"])
    assert not torch.equal(out["ce_tokens_no_slots"], out["ce_tokens"]), (
        "removing the slots changed nothing — the coda is not reading them at all")


def test_arm_a4_drops_slots_and_arm_a2_runs_tokens_through_the_core():
    x, y, layout, _ = _batch(_spec())
    for kw in (dict(coda_sees_slots=False), dict(tokens_through_core=True)):
        m = _model(TULConfig(prefix_k=2, slot_id=4, **kw))
        m.train()
        out = m(x, labels=y, slot_layout=layout)
        assert torch.isfinite(out["loss"]), f"{kw} produced a non-finite loss"
        out["loss"].backward()
    # A4 has no slot positions in the coda → no first-token term at all
    m4 = _model(TULConfig(prefix_k=2, slot_id=4, coda_sees_slots=False))
    m4.eval()
    with torch.no_grad():
        o4 = m4(x, labels=y, slot_layout=layout)
    assert "ce_first_tok" not in o4, "A4 removes the slots, so it has no emitting position"


def test_prefix_k_one_and_two_have_the_right_shapes():
    for k in (1, 2, 3):
        spec = _spec(prefix_k=k)
        x, y, layout, _ = _batch(spec)
        assert x.shape[1] == spec.seq_len + k * spec.max_slots
        m = _model(TULConfig(prefix_k=k, slot_id=4))
        m.eval()
        with torch.no_grad():
            out = m(x, labels=y, slot_layout=layout)
        assert torch.isfinite(out["loss"])
        assert m.tul.W_prefix.shape == (k, m.cfg.d_model, m.cfg.d_model)


def test_layout_prefix_k_must_match_the_model():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    x, y, layout, _ = _batch(_spec(prefix_k=1))
    with pytest.raises(ValueError, match="prefix_k"):
        m(x, labels=y, slot_layout=layout)


def test_layer_passes_per_token_beats_the_dense_baseline():
    """Spec §2: the whole point — the core runs on 9-19× fewer positions."""
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    x, y, layout, _ = _batch(_spec())
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    per_tok = float(out["layer_passes"]) / float(out["n_tokens"])
    dense = m.cfg.n_prelude + m.cfg.n_coda + m.cfg.n_core * m.cfg.mean_depth
    assert per_tok < dense, f"TUL layer-passes/token {per_tok:.2f} not below dense {dense}"


# ── unimplemented arms must fail loudly, not silently ────────────────────────

@pytest.mark.parametrize("kw", [dict(stp_lambda=0.1), dict(set_lambda=0.1),
                                dict(carry=True), dict(xattn=True), dict(bcast=True)])
def test_unimplemented_arm_keys_raise(kw):
    with pytest.raises(NotImplementedError):
        TULConfig(**kw)


# ── training step + checkpoint ───────────────────────────────────────────────

def test_train_step_reaches_every_new_parameter():
    m = _model(TULConfig(prefix_k=2, slot_id=4, token_state_dropout=0.5))
    m.train()
    x, y, layout, _ = _batch(_spec())
    out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for name in ("E_slot", "E_mask", "W_prefix"):
        g = getattr(m.tul, name).grad
        assert g is not None, f"{name} received no gradient"
        assert torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} gradient is zero/NaN"
    core_grads = [p.grad for p in m.core.parameters() if p.grad is not None]
    assert core_grads and sum(float(g.abs().sum()) for g in core_grads) > 0, (
        "the core received no gradient — the slot loop is disconnected from the loss")


def test_checkpoint_round_trip_carries_the_tul_parameters():
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    with torch.no_grad():
        m.tul.E_slot.normal_()
        m.tul.E_mask.normal_()
        m.tul.W_prefix.normal_()
    sd = m.state_dict()
    assert {"tul.E_slot", "tul.E_mask", "tul.W_prefix"} <= set(sd), (
        f"TUL parameters missing from the state dict: {sorted(k for k in sd if 'tul' in k)}")
    m2 = _model(TULConfig(prefix_k=2, slot_id=4), seed=777)
    missing, unexpected = m2.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected}"
    for name in ("E_slot", "E_mask", "W_prefix"):
        assert torch.equal(getattr(m.tul, name), getattr(m2.tul, name)), f"{name} did not load"
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
    assert torch.all(m.tul.W_prefix[0] == torch.eye(m.cfg.d_model)), (
        "W_prefix must start as the identity so the extra coda position costs nothing")


# ── reviewer additions, 2026-08-16 (gaps found by mutation testing the PR) ───
# Each of these was written because a deliberate break of the code it covers left the
# whole suite GREEN. The mutation that motivated it is named in the docstring.

def test_prefix_projections_are_distinct_channels():
    """Mutation RM3: making every prefix position use W_prefix[0] left the suite green.

    prefix_k = 2 [W] exists so the plan position and the emitting position can hold
    DIFFERENT functions of the one looped state (Block Transformer App. F.2 / Fig 3f).
    W_prefix is identity-initialised, so the two are equal at build — a test that never
    perturbs it cannot see the collapse. Perturb, then require the coda inputs to differ.
    """
    m = _model(TULConfig(prefix_k=2, slot_id=4))
    m.eval()
    spec = _spec()
    x, y, layout, _ = _batch(spec, B=1)
    with torch.no_grad():
        m.tul.W_prefix[1].mul_(2.0)                    # W_2 = 2·W_1
        front = m._tul_front(x, layout)
        _xn, h, _d, _g, *_ = m._tul_core(*front, layout)
        values, pos = m.tul.prefix_project(h, layout, layout.l_total)
    s = 0
    assert bool(layout.slot_valid[0, s]), "need a real slot"
    v0, v1 = values[0, s * 2], values[0, s * 2 + 1]     # slot-major: s·K + k
    assert not torch.allclose(v0, v1), (
        "both prefix positions carry the same vector — W_prefix[k] is not being indexed, "
        "so prefix_k=2 is prefix_k=1 with a wasted position")
    assert torch.allclose(v1, v0 * 2.0, atol=1e-5), "position k must use W_prefix[k]"
    assert int(pos[0, s * 2 + 1]) == int(pos[0, s * 2]) + 1, "prefix positions are adjacent"


def test_gather_valid_zeroes_invalid_rows():
    """Mutation RM6: dropping the validity mask in the core gather left the suite green.

    An invalid (pad) slot's ``slot_index`` is 0 — a REAL token position. Without the
    mask the core would loop on that token's state. It is inert for the value of every
    real slot (pads sort last, attention is causal) but it is not inert for anything
    that reduces over the slot axis, which is why the mask must be tested directly.
    """
    from morph.model.tul import gather_valid

    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3) + 1.0
    index = torch.tensor([[4, 0, 0], [1, 5, 0]])
    valid = torch.tensor([[True, False, False], [True, True, False]])
    g = gather_valid(x, index, valid)
    assert torch.equal(g[0, 0], x[0, 4]) and torch.equal(g[1, 1], x[1, 5])
    assert torch.all(g[0, 1:] == 0) and torch.all(g[1, 2] == 0), (
        "invalid slots gathered real content instead of zeros")


def test_pad_slots_do_not_enter_the_core_gain_norm():
    """A per-SAMPLE reduction over the slot axis must exclude pad slots.

    ``core_gain_clip`` is 0.0 in every arm config, so this is dormant today — but it is
    the documented knob for ρ(J_core) (CLAUDE.md § nested dynamical system). It reduces
    over the whole ``[B, S, n, C]`` carrier, so with pads in the norm a row's PAD COUNT
    scales its REAL slots. The pad count is data-dependent (how many spans the row
    happens to contain, i.e. tokens in that row's FUTURE) and differs between the packer
    and the incremental generator, so this is a train/generate mismatch, not only noise.

    Measured 2026-08-16 at τ=0.5, tiny config: masked 1.8e-7 (fp32 tiling roundoff —
    attention reduces over a different sequence length), unmasked 3.6e-2. The tolerance
    sits between the two, five orders of magnitude apart.
    """
    from dataclasses import replace

    # τ must BIND: the HC residual is norm-preserving, so gain ≈ 1 and any τ ≥ 1 leaves
    # the clip an identity — a test at τ = 1.05 passes no matter what the norm counts.
    m = _model(TULConfig(prefix_k=2, slot_id=4), core_gain_clip=0.5)
    m.eval()
    spec = _spec(max_slots=8)
    x, y, layout, _ = _batch(spec, B=1, n=40)
    n_valid = int(layout.slot_valid.sum())
    assert 0 < n_valid < layout.max_slots, "this test needs at least one pad slot"
    with torch.no_grad():
        front = m._tul_front(x, layout)
        h_pad = m._tul_core(*front, layout)[1]
        trimmed = replace(layout,
                          slot_index=layout.slot_index[:, :n_valid].contiguous(),
                          slot_valid=layout.slot_valid[:, :n_valid].contiguous())
        h_trim = m._tul_core(*front, trimmed)[1]
    gap = (h_pad[:, :n_valid] - h_trim).abs().max().item()
    assert gap < 1e-5, (
        f"real slots moved by {gap:.3e} when the PAD COUNT changed — a reduction over "
        f"the slot axis (the gain-clip norm) is counting pad slots")


# ── per-slot input embedding (lab/experiments/failures/2026-08-24-tul-takeover-cure.md) ──

def test_per_slot_embed_off_is_the_shipped_shape_and_forward():
    """Default MUST be one shared vector and a bit-identical forward — every existing
    checkpoint and every existing arm depends on it."""
    x, y, layout, _ = _batch(_spec())
    a = _model(TULConfig(prefix_k=2, slot_id=4))
    assert a.tul.E_slot.dim() == 1
    torch.manual_seed(3)
    la = a(x, labels=y, slot_layout=layout)["loss"]
    b = _model(TULConfig(prefix_k=2, slot_id=4, per_slot_embed=0))
    torch.manual_seed(3)
    lb = b(x, labels=y, slot_layout=layout)["loss"]
    assert torch.equal(la, lb)


def test_per_slot_embed_gives_each_slot_index_its_own_row():
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    m = _model(TULConfig(prefix_k=2, slot_id=4, per_slot_embed=spec.max_slots))
    assert m.tul.E_slot.shape == (spec.max_slots, m.cfg.d_model)
    # plant a distinct, large value in one row and check ONLY that slot's input moves
    with torch.no_grad():
        m.tul.E_slot.zero_()
        m.tul.E_slot[1] = 100.0
    emb = torch.randn(x.shape[0], layout.l_total, m.cfg.d_model)
    got = m.tul.slot_input(emb, layout, add_e_slot=True)
    base = m.tul.slot_input(emb, layout, add_e_slot=False)
    moved = ((got - base).abs().sum(-1) > 1.0)
    for b in range(x.shape[0]):
        for p in range(layout.l_total):
            is_slot1 = bool(layout.slot_mask[b, p]) and int(layout.bag_id[b, p]) == 1
            assert bool(moved[b, p]) == is_slot1, (b, p, int(layout.bag_id[b, p]))


def test_per_slot_embed_seating_starts_every_row_equal_unless_jittered():
    """Seating with std 0 must leave the rows identical, so the activation step is the same
    forward the shared version gives. With std > 0 the rows must differ — that is the whole
    point, since the measured problem is that the slot states start near-parallel."""
    spec = _spec()
    lm = torch.randn(V, 64)
    flat = _model(TULConfig(prefix_k=2, slot_id=4, per_slot_embed=spec.max_slots))
    flat.tul.init_at_activation(lm)
    assert torch.allclose(flat.tul.E_slot[0], flat.tul.E_slot[1])
    jit = _model(TULConfig(prefix_k=2, slot_id=4, per_slot_embed=spec.max_slots,
                           per_slot_embed_std=0.5))
    jit.tul.init_at_activation(lm)
    assert not torch.allclose(jit.tul.E_slot[0], jit.tul.E_slot[1])
    # deterministic: a second model seats to exactly the same rows
    jit2 = _model(TULConfig(prefix_k=2, slot_id=4, per_slot_embed=spec.max_slots,
                            per_slot_embed_std=0.5))
    jit2.tul.init_at_activation(lm)
    assert torch.equal(jit.tul.E_slot, jit2.tul.E_slot)


def test_per_slot_embed_raises_the_slot_states_effective_rank():
    """THE reason it exists. With one shared row the slot inputs are the bag-means plus a
    constant and their effective rank is low; with jittered per-slot rows it must rise."""
    spec = _spec()
    x, y, layout, _ = _batch(spec)
    emb = torch.randn(x.shape[0], layout.l_total, 64) * 0.05     # low-variance bag-means

    def erank(m):
        s = m.tul.slot_input(emb, layout, add_e_slot=True)
        h = s[0][layout.slot_mask[0]]
        sv = torch.linalg.svdvals(h.double()) ** 2
        return float(((sv.sum() ** 2) / (sv ** 2).sum()).detach())

    lm = torch.randn(V, 64)
    shared = _model(TULConfig(prefix_k=2, slot_id=4))
    shared.tul.init_at_activation(lm)
    per = _model(TULConfig(prefix_k=2, slot_id=4, per_slot_embed=spec.max_slots,
                           per_slot_embed_std=1.0))
    per.tul.init_at_activation(lm)
    assert erank(per) > erank(shared) * 1.5, (erank(shared), erank(per))
