"""TUL span-length gate — the §9 invariants of docs/tul-gate-spec.md.

One test per row of the spec's invariant table, plus the §3 data claims the label
depends on. Every test here protects something a silent break would hide inside a
falling loss curve: the predecessor lost a whole ladder to a gate that trained to a
low loss while emitting a constant.

CPU only, tiny config, no tokenizer — the same conventions as ``test_tul_forward.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, TULGate, TULGateConfig
from morph.model.tul_layout import (BoundaryRule, TulGateSpec, TulLayoutSpec,
                                    insert_truncations, pack_tul_row,
                                    slot_layout_from_ids)

V = 64
DOT = 10
K_MAX = 8                       # == span_cap: the label must not saturate (§3.3)


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


def _rule(span_cap=K_MAX) -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=2, span_cap=span_cap, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _ids(B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n)).astype(np.int64)
    ids[ids == 4] = 5
    ids[:, ::6] = DOT
    return ids


def _batch(spec, gate=None, seed=0, B=2, n=90, rng_seed=7):
    return slot_layout_from_ids(_ids(B, n, seed), _rule(), spec, gate=gate,
                                rng=np.random.default_rng(rng_seed))


def _gcfg(**kw) -> TULGateConfig:
    base = dict(k_max=K_MAX, lam=1.0, budget_cond=True)
    base.update(kw)
    return TULGateConfig(**base)


def _tulcfg(gate=None, **kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, token_state_dropout=0.0, gate=gate)
    base.update(kw)
    return TULConfig(**base)


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


# ── §3: the data ────────────────────────────────────────────────────────────

def test_gate_off_leaves_the_packer_byte_identical():
    """§9 invariant 1, data half: no gate ⇒ no new array and NO random draw."""
    ids = _ids(1, 90)[0]
    a0, n0, s0 = pack_tul_row(ids, _rule(), _spec())
    a1, n1, s1 = pack_tul_row(ids, _rule(), _spec(), gate=None, rng=None)
    assert n0 == n1 and set(a0) == set(a1)
    assert "span_len" not in a0 and "trunc_frac" not in s0
    for k in a0:
        assert np.array_equal(a0[k], a1[k]), k


def test_truncation_is_consistent_with_the_rule():
    """§3.2: an inserted cut is a pure INSERTION — restarting the state machine at it
    still yields the same next DATA boundary.

    This is what lets the generator force a cut at ``k`` tokens and stay in sync with the
    loader's segmentation. If it were false, a wrong ``k`` would desynchronise every
    later span in the row instead of costing one span's quality.
    """
    ids = _ids(1, 400, seed=3)[0]
    rule = _rule(span_cap=32)
    gate = TulGateSpec(k_max=32, truncate_p=0.9)
    bpos, _ = rule.cut(ids, 0)
    aug, is_rng = insert_truncations(bpos, rule, gate, np.random.default_rng(1))
    assert is_rng.sum() > 0, "the fixture must actually truncate something"
    assert np.array_equal(np.sort(aug), aug)
    assert set(bpos.tolist()) <= set(aug.tolist())          # nothing removed
    base = -1
    for pos, rng_made in zip(aug.tolist(), is_rng.tolist()):
        seg = ids[base + 1: pos + 1]
        cuts, _ = rule.cut(seg, 0)
        if not rng_made:
            assert cuts.size and int(cuts[0]) + base + 1 == pos, (
                f"data boundary {pos} is not what the rule finds after {base}")
        base = pos


def test_the_label_is_the_NEXT_span_length():
    """§3.3 (amended): slot i plans span i+1 — the span it can causally condition.

    A gate graded on span i's own length would be reading out the past, while generation
    asks it for the future. The last slot's next span is the row's open tail: conditioned
    on, never graded.
    """
    ids = _ids(1, 200, seed=5)[0]
    rule, spec = _rule(span_cap=32), _spec(seq_len=64, max_slots=8)
    gate = TulGateSpec(k_max=32, truncate_p=0.0)
    arrays, n_tok, _ = pack_tul_row(ids, rule, spec, gate=gate,
                                    rng=np.random.default_rng(0))
    bpos, _ = rule.cut(ids, 0)
    row_b = bpos[bpos < n_tok]
    n = int(row_b.shape[0])
    lens = np.diff(np.concatenate([[-1], row_b]))
    assert n >= 3
    for i in range(n - 1):
        assert arrays["span_len"][i] == min(lens[i + 1], 32), i
        assert arrays["len_supervised"][i]
    assert arrays["span_len"][n - 1] == max(1, (n_tok - 1) - int(row_b[-1]))
    assert not arrays["len_supervised"][n - 1], "the open tail is not the data's answer"


def test_an_rng_truncated_span_is_not_graded():
    """§3.2 / §9: the truncation point is OUR rng, so its length is not a label."""
    ids = _ids(1, 400, seed=3)[0]
    rule, spec = _rule(span_cap=32), _spec(seq_len=128, max_slots=16)
    gate = TulGateSpec(k_max=32, truncate_p=0.9)
    arrays, n_tok, stats = pack_tul_row(ids, rule, spec, gate=gate,
                                        rng=np.random.default_rng(1))
    bpos, _ = rule.cut(ids, 0)
    aug, is_rng = insert_truncations(bpos, rule, gate, np.random.default_rng(1))
    n = int((aug < n_tok).sum())
    assert stats["trunc_frac"] > 0.0
    for i in range(n - 1):
        # slot i is graded iff the span it plans (i+1) ended on the DATA's boundary
        assert bool(arrays["len_supervised"][i]) == (not bool(is_rng[i + 1])), i


def test_pad_slots_carry_no_label():
    """§9: a pad slot's slot_index is 0, so a missing mask trains on row 0's first span."""
    _x, _y, layout, _st = _batch(_spec(max_slots=32), gate=TulGateSpec(k_max=K_MAX))
    pad = ~layout.slot_valid
    assert bool(pad.any())
    assert int(layout.span_len[pad].sum()) == 0
    assert not bool(layout.len_supervised[pad].any())


def test_truncation_never_changes_l_total():
    """§9: fixed shapes. Both arms must pack the same number of positions."""
    ids = _ids(1, 400, seed=3)[0]
    rule, spec = _rule(span_cap=32), _spec(seq_len=128, max_slots=16)
    a0, _, _ = pack_tul_row(ids, rule, spec)
    a1, _, _ = pack_tul_row(ids, rule, spec, gate=TulGateSpec(k_max=32, truncate_p=0.9),
                            rng=np.random.default_rng(1))
    assert a0["input_ids"].shape == a1["input_ids"].shape == (spec.l_total,)


def test_span_cap_above_k_max_is_a_hard_error():
    """§3.3: a span longer than k_max cannot be expressed as span_len/k_max."""
    with pytest.raises(ValueError, match="saturate"):
        pack_tul_row(_ids(1, 90)[0], _rule(span_cap=16), _spec(),
                     gate=TulGateSpec(k_max=8), rng=np.random.default_rng(0))


# ── §9 invariant 1: gate off is bit-identical ───────────────────────────────

def test_building_the_gate_draws_no_random_number():
    """§9 invariant 1. nn.Linear/nn.Embedding draw in reset_parameters; a draw would
    advance the global stream and change every later Poisson depth and dropout mask,
    so gate_lambda=0 would NOT reproduce arm A1."""
    a = _model(_tulcfg(gate=None))
    ra = torch.rand(4)
    b = _model(_tulcfg(gate=_gcfg()))
    rb = torch.rand(4)
    assert torch.equal(ra, rb), "the gate's construction moved the RNG stream"
    pa = dict(a.named_parameters())
    for n, p in b.named_parameters():
        if n.startswith("tul_gate."):
            continue
        assert torch.equal(p, pa[n]), n


def test_gate_parameters_start_at_exactly_zero():
    g = TULGate(8, _gcfg())
    assert torch.equal(g.w, torch.zeros(8))
    assert torch.equal(g.b, torch.zeros(1))
    assert torch.equal(g.budget, torch.zeros(K_MAX + 1, 8))
    assert torch.equal(g.norm.weight, torch.ones(8))


def test_lambda_zero_and_no_budget_cond_is_bit_identical_to_a1():
    """§9 invariant 1: same loss, same gradients on every shared parameter."""
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    outs = {}
    for name, gate in (("a1", None), ("off", _gcfg(lam=0.0, budget_cond=False))):
        m = _model(_tulcfg(gate=gate))
        m.train()
        torch.manual_seed(99)
        o = m(x, labels=y, slot_layout=layout)
        o["loss"].backward()
        outs[name] = (float(o["loss"].detach()),
                      {n: p.grad.clone() for n, p in m.named_parameters()
                       if p.grad is not None and not n.startswith("tul_gate.")})
    assert outs["a1"][0] == outs["off"][0]
    assert set(outs["a1"][1]) == set(outs["off"][1])
    for n, g in outs["a1"][1].items():
        assert torch.equal(g, outs["off"][1][n]), n


# ── §6: the loss ────────────────────────────────────────────────────────────

def _fake_traj(B=2, S=5, T=3):
    g = TULGate(64, _gcfg())
    return g, torch.rand(B, S, T, requires_grad=True)


def test_train_zeros_gives_the_two_part_target_of_the_original_spec():
    """``train_zeros=True`` is the predecessor's shape, kept as a switch (§6 amended)."""
    g = TULGate(64, _gcfg(train_zeros=True))
    traj = torch.rand(2, 5, 3, requires_grad=True)
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX))
    depths = torch.full(layout.slot_index.shape, 2, dtype=torch.long)
    res = g.loss(traj, depths, layout)
    # rebuild the target by hand and check the loss equals the masked Huber
    t = torch.arange(3).view(1, 1, 3)
    last = (depths - 1).unsqueeze(-1)
    tgt = torch.where(t == last, (layout.span_len.float() / K_MAX).unsqueeze(-1),
                      torch.zeros(()))
    sup = layout.len_supervised.unsqueeze(-1) & layout.slot_valid.unsqueeze(-1)
    mask = (sup & (t == last)) | (layout.slot_valid.unsqueeze(-1) & (t < last))
    ref = (torch.nn.functional.smooth_l1_loss(traj, tgt, reduction="none")
           * mask).sum() / mask.sum()
    assert torch.allclose(res["loss_gate"], ref)


def test_an_unsupervised_slot_supervises_the_zeros_but_not_the_value():
    """§9: flipping the VALUE at an unsupervised slot must not move the loss; flipping
    it at a supervised one must."""
    g = TULGate(64, _gcfg(train_zeros=True))
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX, truncate_p=0.0))
    # force one supervised and one unsupervised slot in row 0
    layout.len_supervised[0, 0] = True
    layout.len_supervised[0, 1] = False
    layout.slot_valid[0, :2] = True
    depths = torch.full(layout.slot_index.shape, 2, dtype=torch.long)
    base = torch.full((2, 5, 3), 0.3)
    l0 = float(g.loss(base, depths, layout)["loss_gate"])
    hit = base.clone()
    hit[0, 0, 1] = 0.9                 # the supervised slot's final iteration
    miss = base.clone()
    miss[0, 1, 1] = 0.9                # the unsupervised slot's final iteration
    assert float(g.loss(hit, depths, layout)["loss_gate"]) != l0
    assert float(g.loss(miss, depths, layout)["loss_gate"]) == l0
    # …but its ZEROS are still supervised
    zero = base.clone()
    zero[0, 1, 0] = 0.9
    assert float(g.loss(zero, depths, layout)["loss_gate"]) != l0


def test_pad_slots_contribute_exactly_zero_to_the_gate_loss():
    g, _ = _fake_traj()
    _x, _y, layout, _ = _batch(_spec(max_slots=32), gate=TulGateSpec(k_max=K_MAX))
    depths = torch.full(layout.slot_index.shape, 2, dtype=torch.long)
    a = torch.full((2, 32, 3), 0.3)
    b = a.clone()
    pad = ~layout.slot_valid
    b[pad] = 0.99
    assert float(g.loss(a, depths, layout)["loss_gate"]) == \
           float(g.loss(b, depths, layout)["loss_gate"])


def test_the_gate_reads_a_scalar_and_regresses_nothing_vector_valued():
    """§9 invariant 2 — the narrow exception to "slot core states have no loss".

    A scalar/discrete READOUT is allowed; a vector regression onto a target
    representation is not. Structurally: the readout is [B, S] (one number per slot),
    the loss takes only that trajectory, and no gate parameter has d_model as an OUTPUT
    dimension of the loss path.
    """
    g = TULGate(64, _gcfg())
    h = torch.randn(2, 5, 4, 64)
    out = g.readout(h)
    assert out.shape == (2, 5), "the readout must be one scalar per slot"
    import inspect
    params = list(inspect.signature(g.loss).parameters)
    assert params[0] == "g_traj", "the loss must be a function of the scalar trajectory"
    # `budget` maps a LENGTH into d_model — it is an input to the coda, never a target.
    assert not any(p.requires_grad and p.shape[-1] == 64 and n == "w"
                   for n, p in g.named_parameters() if p.dim() > 1)


# ── §4/§5: the forward ──────────────────────────────────────────────────────

def test_the_gate_loss_is_added_and_reaches_every_gate_parameter():
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg()))
    m.train()
    # `w` starts at exactly zero, so at init the norm's own gradient is zero by
    # construction (d g / d z = w). Move it first, then require EVERY gate parameter to
    # be reachable — a parameter that is unreachable once the path is live is dead.
    with torch.no_grad():
        m.tul_gate.w.normal_(0.0, 0.1)
    out = m(x, labels=y, slot_layout=layout)
    assert "gate/loss_gate" in out and "loss_tokens" in out
    assert float(out["loss"]) != float(out["loss_tokens"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if n.startswith("tul_gate."):
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            assert float(p.grad.abs().sum()) > 0.0, f"{n} got no gradient"


def test_the_gate_gradient_reaches_the_core_inside_the_bptt_window():
    """§4: the readout runs OUTSIDE the checkpoint, so it shapes the core state on the
    iterations the token loss also backprops through."""
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg()))
    m.train()
    with torch.no_grad():
        m.tul_gate.w.normal_(0.0, 0.1)      # d g / d h is w; at w = 0 there is no path
    xf, x0f, bgf = m._tul_front(x, layout)
    _xn, _h, depths, traj = m._tul_core(xf, x0f, bgf, layout)
    assert traj is not None and traj.shape[:2] == layout.slot_index.shape
    g = m.tul_gate.loss(traj, depths, layout)["loss_gate"]
    core_w = next(p for p in m.core[0].parameters() if p.dim() == 2)
    (grad,) = torch.autograd.grad(g, core_w, retain_graph=True, allow_unused=True)
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_budget_conditioning_changes_the_coda_only_once_trained():
    """§5: zero-init ⇒ an exact no-op at step 0, so the arm STARTS as A1 and diverges
    only as the table learns. A budget table that never leaves zero is a dead §5."""
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg(lam=0.0)))
    m.eval()
    with torch.no_grad():
        a = float(m(x, labels=y, slot_layout=layout)["loss"])
        m.tul_gate.budget.normal_(0.0, 0.5)
        b = float(m(x, labels=y, slot_layout=layout)["loss"])
    m2 = _model(_tulcfg(gate=None))
    m2.eval()
    with torch.no_grad():
        ref = float(m2(x, labels=y, slot_layout=layout)["loss"])
    assert a == ref and b != ref


def test_training_uses_the_realised_length_and_generation_the_predicted_one():
    """§9: never a mixture. The layout carrying a label IS the switch."""
    spec = _spec()
    x, _y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg()))
    m.eval()
    ids = m._tul_budget_ids(layout, torch.full(layout.slot_index.shape, 2),
                            torch.rand(*layout.slot_index.shape, 3))
    assert torch.equal(ids, layout.span_len)
    layout.span_len, layout.len_supervised = None, None
    with torch.no_grad():
        k = m(x, slot_layout=layout)["gate_k"]
    assert k.shape == layout.slot_index.shape
    assert int(k[layout.slot_valid].min()) >= 1 and int(k.max()) <= K_MAX
    assert int(k[~layout.slot_valid].sum()) == 0
    # …and it is genuinely k = round(g·k_max), not a constant that happens to be in
    # range: drive the gate to each end and the budget must follow it there.
    with torch.no_grad():
        m.tul_gate.b.fill_(30.0)
        k_hi = m(x, slot_layout=layout)["gate_k"]
        m.tul_gate.b.fill_(-30.0)
        k_lo = m(x, slot_layout=layout)["gate_k"]
    v = layout.slot_valid
    assert int(k_hi[v].min()) == K_MAX, "g→1 must ask for the whole budget"
    assert int(k_lo[v].max()) == 1, "g→0 must ask for the floor (§8 clamps k≥1)"


def test_gate_with_arm_a2_raises():
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg(), tokens_through_core=True))
    with pytest.raises(NotImplementedError, match="A2"):
        m(x, labels=y, slot_layout=layout)


# ── §7: the halting arm ─────────────────────────────────────────────────────

def test_halt_is_eval_only():
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg(drives_depth=True)))
    m.train()
    with pytest.raises(RuntimeError, match="EVAL ONLY"):
        m.tul_forward_halt(x, y, layout)


def test_halt_depths_stay_inside_one_and_max_depth():
    """§9: a slot that never asks to stop must not hang the generator — it takes the cap."""
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg(drives_depth=True)))
    m.eval()
    with torch.no_grad():
        out = m.tul_forward_halt(x, y, layout)
        # a gate pinned shut: every slot must run to the cap and stop there
        m.tul_gate.b.fill_(-30.0)
        shut = m.tul_forward_halt(x, y, layout)
    d = float(out["gate/depth_mean"])
    assert 1.0 <= d <= m.cfg.max_depth
    assert float(shut["gate/depth_mean"]) == float(m.cfg.max_depth)


def test_halt_and_fixed_depth_share_every_weight():
    """§11: the two arms are ONE training run scored twice, so the comparison is paired."""
    spec = _spec()
    x, y, layout, _ = _batch(spec, gate=TulGateSpec(k_max=K_MAX))
    m = _model(_tulcfg(gate=_gcfg(drives_depth=True)))
    m.eval()
    before = {n: p.clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        m.tul_forward_halt(x, y, layout)
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]), n


# ── §8: generation ──────────────────────────────────────────────────────────

def test_the_generator_forces_a_cut_when_the_budget_runs_out():
    from morph.inference.tul_generate import TulRowBuilder
    spec = _spec(max_slots=16)
    rule = _rule(span_cap=32)
    b = TulRowBuilder(rule=rule, spec=spec)
    b.budget = 3
    cuts = [b.append(t) for t in [7, 8, 9, 12, 13, 14]]     # no boundary token at all
    assert cuts == [False, False, True, False, False, True]
    assert b.n_slots == 2 and b.slot_first == [3, 8]
    # …and a budget of 0 is the pre-gate behaviour: the rule alone decides
    b2 = TulRowBuilder(rule=rule, spec=spec)
    assert [b2.append(t) for t in [7, 8, 9, 12, 13, 14]] == [False] * 6


def test_a_forced_cut_leaves_the_rule_in_the_same_state_as_a_real_one():
    """§8: a wrong k costs quality, never synchronisation."""
    from morph.inference.tul_generate import TulRowBuilder
    rule, spec = _rule(span_cap=32), _spec(max_slots=16)
    b = TulRowBuilder(rule=rule, spec=spec)
    b.budget = 3
    for t in [7, 8, 9]:
        b.append(t)
    assert b.span_len == 0, "a forced cut must reset the open span exactly like a real cut"


# ── §10: the instruments ────────────────────────────────────────────────────

def test_seat_bias_puts_the_gate_at_the_corpus_base_rate():
    from morph.training.gate_audit import seat_gate_bias
    g = TULGate(64, _gcfg())
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX))
    st = seat_gate_bias(g, layout)
    sel = layout.slot_valid & (layout.span_len > 0)
    want = float((layout.span_len[sel].float().mean() / K_MAX))
    assert abs(float(torch.sigmoid(g.b)) - want) < 1e-5
    assert abs(st["gate_target_mean"] - want) < 1e-5


def test_the_audit_refuses_a_gate_that_is_not_in_the_optimizer():
    from morph.training.gate_audit import audit_gate_travel, seat_gate_bias
    g = TULGate(64, _gcfg())
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX))
    st = seat_gate_bias(g, layout)
    full = torch.optim.AdamW(list(g.parameters()), lr=1e-4)
    audit_gate_travel(g, full, st, total_steps=20000, z_norm=8.0)
    partial = torch.optim.AdamW([g.w, g.b], lr=1e-4)
    with pytest.raises(RuntimeError, match="param groups"):
        audit_gate_travel(g, partial, st, total_steps=20000, z_norm=8.0)
    frozen = torch.optim.AdamW(list(g.parameters()), lr=0.0)
    with pytest.raises(RuntimeError, match="lr=0"):
        audit_gate_travel(g, frozen, st, total_steps=20000, z_norm=8.0)


def test_the_audit_refuses_a_step_budget_too_small_to_move_the_gate():
    from morph.training.gate_audit import audit_gate_travel, seat_gate_bias
    g = TULGate(64, _gcfg())
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX))
    st = seat_gate_bias(g, layout)
    opt = torch.optim.AdamW(list(g.parameters()), lr=1e-9)
    with pytest.raises(RuntimeError, match="REFUSED"):
        audit_gate_travel(g, opt, st, total_steps=10, z_norm=8.0)


def test_a_frozen_gate_direction_fails_the_alive_check():
    from morph.training.gate_audit import assert_gate_is_alive
    g = TULGate(64, _gcfg())
    with pytest.raises(RuntimeError, match="DEAD"):
        assert_gate_is_alive(g, step=2000)
    with torch.no_grad():
        g.w.normal_()
    assert_gate_is_alive(g, step=2000)


def test_separation_is_zero_for_a_constant_gate_and_positive_for_a_working_one():
    """§10: a gate can sit at a low loss and still not discriminate. This is the number
    that tells the difference, and the predecessor defined it and never measured it."""
    g = TULGate(64, _gcfg())
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX))
    depths = torch.full(layout.slot_index.shape, 3, dtype=torch.long)
    flat = torch.full((2, 5, 3), 0.4)
    assert abs(float(g.loss(flat, depths, layout)["gate_separation"])) < 1e-6
    work = flat.clone()
    work[:, :, 2] = 0.8
    assert float(g.loss(work, depths, layout)["gate_separation"]) > 0.3


# ── §12: unbuilt keys ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [{"scheduled_sampling": 0.5}, {"stop_head": True},
                                {"ponder_lambda": 0.1}])
def test_unimplemented_gate_keys_raise(kw):
    with pytest.raises(NotImplementedError):
        TULGateConfig(**kw)


def test_both_halves_of_a_truncation_clear_min_span():
    """§3.2: a truncated span is a real training target, so it must be a plausible one.

    A 1-token first half wastes a slot and gives the gate a budget it can never learn;
    a short second half makes the next span start almost where it ended. Both halves
    respect ``min_span``, which is also why the insertion stays rule-consistent.
    """
    ids = _ids(1, 600, seed=11)[0]
    rule = BoundaryRule(is_boundary=_rule().is_boundary, min_span=4, span_cap=32, eos_id=0)
    gate = TulGateSpec(k_max=32, truncate_p=1.0)
    bpos, _ = rule.cut(ids, 0)
    aug, is_rng = insert_truncations(bpos, rule, gate, np.random.default_rng(2))
    assert is_rng.sum() > 0
    base = -1
    for pos, rng_made in zip(aug.tolist(), is_rng.tolist()):
        assert pos - base >= rule.min_span, (
            f"span ending at {pos} is only {pos - base} tokens, below min_span")
        base = pos


def test_the_gate_lands_in_the_no_decay_optimizer_group():
    """§10: weight decay on a zero-init readout direction pulls against the only
    gradient it has — the "provably cannot move" class the audit exists to catch."""
    from omegaconf import OmegaConf
    from morph.training.optimizer import create_optimizer
    m = _model(_tulcfg(gate=_gcfg()))
    cfg = OmegaConf.create({"training": {"lr": 1e-4, "weight_decay": 0.1,
                                         "optimizer": "adamw", "betas": [0.9, 0.95]}})
    opt = create_optimizer(m, cfg)
    for n, p in m.named_parameters():
        if not n.startswith("tul_gate."):
            continue
        hit = [g for g in opt.param_groups if any(q is p for q in g["params"])]
        assert len(hit) == 1, n
        assert float(hit[0].get("weight_decay", 0.0)) == 0.0, f"{n} is being decayed"


def test_no_label_lands_on_the_sigmoid_asymptote():
    """§3.3 headroom: ``gate_k_max`` exceeds ``span_cap``, so the largest target is < 1.

    Measured on OpenWebText at span_cap 32, 24.5 % of labels ARE a span of exactly 32.
    With ``k_max = span_cap`` a quarter of the training signal would sit on ``g = 1.0``,
    where the sigmoid's gradient vanishes and the head can only under-predict.
    """
    cap, k_max = 8, 10
    rule = BoundaryRule(is_boundary=_rule().is_boundary, min_span=2, span_cap=cap, eos_id=0)
    gate = TulGateSpec(k_max=k_max, truncate_p=0.0)
    # a stream with NO boundary token: every span is cut by span_cap, so every label is
    # the maximum one — the case that is 24.5 % of real OpenWebText.
    ids = np.random.default_rng(2).integers(12, V, size=300).astype(np.int64)
    arrays, _n, _st = pack_tul_row(ids, rule, _spec(max_slots=16),
                                   gate=gate, rng=np.random.default_rng(0))
    lab = arrays["span_len"]
    assert lab.max() == cap, "the fixture must contain a capped span"
    assert float(lab.max()) / k_max < 1.0
    g = TULGate(16, TULGateConfig(k_max=k_max, k_decode_max=cap, lam=1.0))
    # the biggest target is reachable at a finite logit, and it decodes back exactly
    top = float(lab.max()) / k_max
    assert int(g.choose_k(torch.tensor([top]))) == cap


def test_choose_k_never_exceeds_the_rule_s_span_cap():
    """§3.3: a budget above span_cap would index a budget row no example ever trains."""
    g = TULGate(16, TULGateConfig(k_max=10, k_decode_max=8, lam=1.0))
    assert g.budget.shape[0] == 9, "the table holds only reachable budgets"
    assert int(g.choose_k(torch.ones(1))) == 8
    assert int(g.choose_k(torch.zeros(1))) == 0


def test_the_default_target_is_the_length_at_every_iteration():
    """§6 AMENDED. The Poisson depth is independent of the input, so the two-part target's
    optimum is the HAZARD and the length is multiplied away — measured at k=5.00 against
    gold 18.98. With the zeros off the head regresses the length at every iteration, so a
    generation that reads it at ANY depth gets an unbiased length.
    """
    g = TULGate(64, _gcfg())
    assert g.gate.train_zeros is False
    _x, _y, layout, _ = _batch(_spec(), gate=TulGateSpec(k_max=K_MAX))
    depths = torch.full(layout.slot_index.shape, 3, dtype=torch.long)
    tgt = (layout.span_len.float() / K_MAX).unsqueeze(-1).expand(2, 5, 3)
    perfect = tgt.clone()
    assert float(g.loss(perfect, depths, layout)["loss_gate"]) == 0.0
    # …and under the ORIGINAL target that same trajectory is heavily penalised, because
    # it refuses to emit zero on the early iterations.
    g0 = TULGate(64, _gcfg(train_zeros=True))
    assert float(g0.loss(perfect, depths, layout)["loss_gate"]) > 0.0
    # an early iteration now carries the SAME signal as the last one
    a = torch.full((2, 5, 3), 0.3)
    b = a.clone()
    b[0, 0, 0] = 0.9
    assert float(g.loss(b, depths, layout)["loss_gate"]) != float(g.loss(a, depths, layout)["loss_gate"])


def test_k_corr_is_zero_for_a_constant_gate_and_one_for_a_perfect_one():
    """§10: with one target at every iteration, gate_separation is ~0 BY DESIGN, so the
    dead-gate detector is the correlation between the chosen k and the gold length."""
    g = TULGate(64, _gcfg())
    _x, _y, layout, _ = _batch(_spec(max_slots=32), gate=TulGateSpec(k_max=K_MAX))
    depths = torch.full(layout.slot_index.shape, 3, dtype=torch.long)
    const = torch.full((2, 32, 3), 0.5)
    assert abs(float(g.loss(const, depths, layout)["gate_k_corr"])) < 1e-3
    perfect = (layout.span_len.float() / K_MAX).unsqueeze(-1).expand(2, 32, 3).contiguous()
    assert float(g.loss(perfect, depths, layout)["gate_k_corr"]) > 0.99


def test_the_open_tail_label_is_clamped_to_span_cap_not_to_k_max():
    """§3.3: the one length that is not a real span is the row's open tail, and it can be
    arbitrarily long. Clamping it to ``k_max`` instead of ``span_cap`` would index a budget
    row that has no training example and hand the coda a zero vector."""
    cap, k_max = 8, 10
    rule = BoundaryRule(is_boundary=_rule().is_boundary, min_span=2, span_cap=cap, eos_id=0)
    # the row's slot budget runs out long before its token budget, so the tail is long
    arrays, _n, _st = pack_tul_row(
        _ids(1, 400, seed=4)[0], rule, _spec(seq_len=120, max_slots=2),
        gate=TulGateSpec(k_max=k_max, truncate_p=0.0), rng=np.random.default_rng(0))
    lab = arrays["span_len"]
    assert lab[1] > 0, "the fixture must actually reach the slot budget"
    assert int(lab.max()) <= cap, f"a label of {lab.max()} exceeds span_cap {cap}"
    # …and it holds for every shape the packer can end a row in, not just this one.
    for seed in range(8):
        for slots in (2, 4, 8, 16):
            a, _n, _s = pack_tul_row(
                _ids(1, 400, seed=seed)[0], rule, _spec(seq_len=120, max_slots=slots),
                gate=TulGateSpec(k_max=k_max, truncate_p=0.3),
                rng=np.random.default_rng(seed))
            assert int(a["span_len"].max()) <= cap, (seed, slots, a["span_len"])
            assert int(a["span_len"][a["span_len"] > 0].min()) >= 1


def test_generation_metrics_catch_a_repetition_loop():
    """§10: a degenerate repetition loop scores an EXCELLENT perplexity — measured 1.46
    against real text's 32.44 — so the fluency number is meaningless without the
    diversity number beside it."""
    from morph.inference.gen_metrics import ngram_stats, span_stats
    from morph.inference.tul_generate import TulRowBuilder
    rep4, d4 = ngram_stats([1, 2, 3, 4] * 40)
    assert rep4 > 0.9 and d4 < 0.1
    rep4, d4 = ngram_stats(list(range(200)))
    assert rep4 == 0.0 and d4 == 1.0
    # a span that ended because the BUDGET ran out is not a span that ended on a boundary
    rule, spec = _rule(span_cap=32), _spec(max_slots=16)
    b = TulRowBuilder(rule=rule, spec=spec)
    b.budget = 5
    for t in [7, 8, 9, 12, 13]:            # no boundary token anywhere
        b.append(t)
    st = span_stats(b, rule)
    assert st["n_spans"] == 1 and st["mean_span"] == 5 and st["boundary_frac"] == 0.0
    b2 = TulRowBuilder(rule=rule, spec=spec)
    for t in [7, 8, 9, 12, DOT]:           # ends on a real boundary
        b2.append(t)
    assert span_stats(b2, rule)["boundary_frac"] == 1.0


def test_gate_skill_is_measured_against_the_best_constant_predictor():
    """§10: `gate_k_abs_err` alone is unreadable. Predicting ONE number forever scores
    9.03 tokens on real OpenWebText and the gate scored 8.62 at step 500, so the entire
    claim lives in the 0.41 between them — which means the floor has to be in the run,
    not in someone's head."""
    g = TULGate(64, _gcfg())
    _x, _y, layout, _ = _batch(_spec(max_slots=32), gate=TulGateSpec(k_max=K_MAX))
    depths = torch.full(layout.slot_index.shape, 3, dtype=torch.long)
    # a gate that emits the gold median for every slot has, by definition, ZERO skill
    med = layout.span_len[layout.slot_valid].float().median()
    const = torch.full((2, 32, 3), float(med) / K_MAX)
    r = g.loss(const, depths, layout)
    assert abs(float(r["gate_k_skill"])) < 1e-4, "a constant predictor cannot have skill"
    assert float(r["gate_k_mae_const"]) > 0.0
    # a perfect gate's skill IS the constant predictor's error
    perfect = (layout.span_len.float() / K_MAX).unsqueeze(-1).expand(2, 32, 3).contiguous()
    r2 = g.loss(perfect, depths, layout)
    assert float(r2["gate_k_abs_err"]) < 0.51          # k = round(g·k_max) rounding only
    assert float(r2["gate_k_skill"]) > float(r2["gate_k_mae_const"]) - 0.51
