"""``tul.fan_history_streams`` — split the fan's K streams into HISTORY and PLAN
(LXTUL-R, 2026-09-22).

Design record: ``.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md``.
LXTUL-R Step 1b (``lab/experiments/results/2026-09-22-lxtul-r-step1b/``) filed with the
relay (``tul.loop_reach``, the K-1 span-back chain) and the diversity push
(``fan_repel_lambda``) sharing the SAME K cells: the repulsion term reads the whole
``fan_k`` axis every pass, which includes whatever cell is carrying content forward from
the previous slot. This key gives the relay its own ``h`` HISTORY streams (the ONLY
streams that read cross-slot, restricted to their own stream index — the ``fan_lineage``
narrowing) and lets the repulsion push apart the remaining ``K - h`` PLAN streams alone,
which never read another slot directly.

WHAT THIS FILE HAS TO PROVE, in the order it matters:

1. OFF IS NOTHING. ``fan_history_streams: 0`` (the default) is bit-identical to the tree
   before the key existed, and the mask builder at ``history_streams=0`` (the default
   argument) equals the pre-existing builder's output across several ``(S, M, reach,
   lineage)`` combinations.
2. THE MASK IS THE RELATION THE DOCSTRING SAYS: at ``h=1, M=4, reach=1``, cross-slot
   entries exist only for ``(query stream < h) AND (query stream == key stream)``; a PLAN
   query row has NO cross-slot entry at all; the own-slot block is unconditionally True.
3. THE PERTURBATION: end to end on the tiny model, a PLAN cell's seed cannot reach the
   next slot at forced depth 1 (exactly 0.0) but does by depth 2; a HISTORY cell's seed
   reaches the next slot at both depths.
4. THE REACH STAIRCASE IS UNCHANGED: the SAME one-slot-per-pass span-level law
   (``test_lxtul_r_composition.py::test_loop_reach1_register_reaches_exactly_one_slot_per_pass``)
   holds at ``h=0`` and ``h=1`` alike — the split narrows WHICH CELL carries the relay,
   not whether the relay happens or how far it reaches per pass (the whole slot's cells
   re-mix every pass, and the coda reads the whole slot).
5. THE REPULSION IS CHARGED ON THE PLAN STREAMS ALONE: the term equals one computed BY
   HAND on the sliced streams, and the reported ``fan/stream_cos_t{t}`` instrument still
   covers all K streams. ``plan_streams`` is a pure slicing helper, tested on its own.
6. REFUSALS: ``history_streams`` with ``reach=0``, ``>= fan_k - 1``, and with ``fan_k=0``
   all raise the documented messages, at both the function and the ``TULConfig`` layer.
7. A TRAINING STEP (forward + backward) is finite on the ``h=1`` composition with
   ``fan_mix: all``, and the shipped config composes through Hydra +
   ``build_tul_runtime``.

CPU only, fp32, ``use_kernels=False`` — the ``tests/test_tul_fan.py`` fixtures (``_tiny``,
``_rule``, ``_spec``, ``_ids``, ``_batch``, ``_state``, ``DOT``) and
``tests/test_lxtul_r_composition.py``'s composed-model fixtures (``_r1_tul``, ``_r1_model``,
``_span_logit_delta``), reused the way ``tests/test_tul_fan_all.py`` reuses ``test_tul_fan``'s.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHTransformer, slot_cell_relation
from morph.model.tul import TULConfig
from morph.model.tul_fan import fan_repel_term, fan_stream_cos, plan_streams
from morph.training.tul_setup import KNOWN_TUL_KEYS, build_tul_runtime, reject_unknown_tul_keys

from test_lxtul_r_composition import _r1_model, _r1_tul, _span_logit_delta
from test_tul_fan import _batch, _state


def _hist_tul(h: int = 1, **kw) -> TULConfig:
    return _r1_tul(fan_history_streams=h, **kw)


def _hist_model(seed: int = 1234, h: int = 1, tul_kw: dict | None = None,
                **model_kw) -> MORPHTransformer:
    tk = dict(fan_history_streams=h)
    tk.update(tul_kw or {})
    return _r1_model(seed=seed, tul_kw=tk, **model_kw)


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_slots,m_cells,reach,lineage", [
    (3, 2, 0, False), (3, 2, 1, False), (3, 2, 1, True),
    (4, 4, 0, False), (4, 4, 2, False), (4, 4, 1, True),
])
def test_history_streams_zero_matches_the_call_with_no_history_arg(
        n_slots, m_cells, reach, lineage):
    dev = torch.device("cpu")
    blk_a, same_a = slot_cell_relation(n_slots, m_cells, dev, reach=reach, lineage=lineage)
    blk_b, same_b = slot_cell_relation(n_slots, m_cells, dev, reach=reach, lineage=lineage,
                                       history_streams=0)
    assert torch.equal(blk_a, blk_b)
    assert torch.equal(same_a, same_b)


def test_fan_history_streams_zero_is_the_default_and_a_known_key():
    assert TULConfig(prefix_k=1, slot_id=4).fan_history_streams == 0
    assert "fan_history_streams" in KNOWN_TUL_KEYS


def test_history_streams_zero_model_is_bit_identical_to_the_key_absent():
    """``fan_history_streams: 0`` (default) vs the key never mentioned: loss, logits,
    total gradient and parameter names, to the last bit."""
    ids, inp, lab, layout = _batch(prefix_k=4, seed=2)
    m_absent = _r1_model(seed=11)
    m_zero = _hist_model(seed=11, h=0)
    assert set(m_absent.state_dict()) == set(m_zero.state_dict())

    def _run(m):
        # `slot_mean_depth` / `slot_max_depth` are 0 here (the base `_r1_tul` does not
        # set them), so the forward draws a real per-batch Poisson depth from the GLOBAL
        # RNG — reseed immediately before the call, the `test_tul_fan.py::
        # test_fan_k_zero_is_bit_identical_to_the_pre_key_tree` pattern, or the two runs
        # would (correctly) diverge on the depth draw alone, which has nothing to do with
        # `fan_history_streams`.
        torch.manual_seed(7)
        m = m.train()
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        g = sum(float(p.grad.double().abs().sum())
                for p in m.parameters() if p.grad is not None)
        return out["loss"].item(), g

    loss_a, g_a = _run(m_absent)
    loss_b, g_b = _run(m_zero)
    assert loss_a == loss_b
    assert g_a == g_b


# ── 2. THE MASK ───────────────────────────────────────────────────────────────


def test_history_mask_cross_slot_is_exactly_history_index_matched():
    n_slots, m_cells, h = 3, 4, 1
    blk, same = slot_cell_relation(n_slots, m_cells, torch.device("cpu"), reach=1,
                                   history_streams=h)
    blk, same = blk[0, 0], same[0, 0]
    sm = n_slots * m_cells
    slot_of = torch.arange(sm) // m_cells
    cell_of = torch.arange(sm) % m_cells

    for i in range(sm):
        for j in range(sm):
            si, sj = int(slot_of[i]), int(slot_of[j])
            ci, cj = int(cell_of[i]), int(cell_of[j])
            if si == sj:
                expected = True                    # own slot: always True
            else:
                expected = (sj == si - 1) and (ci < h) and (ci == cj)
            assert bool(blk[i, j]) == expected, (i, j, si, sj, ci, cj)
            assert bool(same[i, j]) == (si == sj)


def test_history_mask_plan_rows_have_no_cross_slot_entry():
    n_slots, m_cells, h = 4, 4, 1
    blk, _ = slot_cell_relation(n_slots, m_cells, torch.device("cpu"), reach=1,
                                history_streams=h)
    blk = blk[0, 0]
    slot_of = torch.arange(n_slots * m_cells) // m_cells
    cell_of = torch.arange(n_slots * m_cells) % m_cells
    for i in range(n_slots * m_cells):
        if int(cell_of[i]) < h:
            continue                               # history row, checked above
        si = int(slot_of[i])
        for j in range(n_slots * m_cells):
            sj = int(slot_of[j])
            if sj == si:
                continue
            assert not bool(blk[i, j]), \
                f"plan cell {i} (slot {si}) must not read cell {j} of a different slot"


def test_history_mask_own_slot_block_is_unconditionally_true():
    n_slots, m_cells, h = 3, 4, 1
    blk, _ = slot_cell_relation(n_slots, m_cells, torch.device("cpu"), reach=1,
                                history_streams=h)
    blk = blk[0, 0]
    slot_of = torch.arange(n_slots * m_cells) // m_cells
    for i in range(n_slots * m_cells):
        for j in range(n_slots * m_cells):
            if int(slot_of[i]) == int(slot_of[j]):
                assert bool(blk[i, j]), f"own-slot pair ({i},{j}) must be True"


def test_history_narrowing_is_a_no_op_composed_with_lineage():
    """``lineage`` alone narrows the cross-slot half to same-index; ``history_streams``
    narrows it further to (same-index AND a history stream). Composing both must equal
    ``history_streams`` alone (the AND is redundant on the history half, as documented)."""
    n_slots, m_cells, h = 4, 4, 1
    dev = torch.device("cpu")
    hist_only, _ = slot_cell_relation(n_slots, m_cells, dev, reach=1, history_streams=h)
    both, _ = slot_cell_relation(n_slots, m_cells, dev, reach=1, lineage=True,
                                 history_streams=h)
    assert torch.equal(hist_only, both)


# ── 3. THE PERTURBATION ──────────────────────────────────────────────────────


def _perturb_register_cell_and_measure(m: MORPHTransformer, inp, layout, slot_idx: int,
                                       cell_idx: int, m_cells: int, tgt_span: int,
                                       delta: torch.Tensor) -> float:
    """max |logit| change on ``tgt_span``'s tokens when a random vector is added to ONE
    cell's SEED — the register's own additive output, ``[B, S*M, C]``, the tensor that
    feeds ``core_init`` before any pass runs (``morph/model/transformer.py``, the
    ``_reg_term`` seam). A forward hook on ``tul_register`` perturbs exactly that one
    cell and nothing else — the cleanest way to isolate ONE cell's content, since a
    token-id perturbation (``test_lxtul_r_composition.py::_span_logit_delta``) moves an
    entire span's worth of cells at once through the shared pooling keys/values.
    """
    bag = layout.bag_id[0]
    slot_mask = layout.slot_mask[0]
    tgt = ((bag == tgt_span) & ~slot_mask).nonzero(as_tuple=True)[0]
    assert tgt.numel() > 0, tgt_span

    def hook(module, inputs, output):
        out = output.clone()
        idx = slot_idx * m_cells + cell_idx
        out[:, idx, :] = out[:, idx, :] + delta.to(out.dtype)
        return out

    with torch.no_grad():
        a = m(inp, labels=None, slot_layout=layout)["logits"][0, tgt]
    handle = m.tul_register.register_forward_hook(hook)
    try:
        with torch.no_grad():
            b = m(inp, labels=None, slot_layout=layout)["logits"][0, tgt]
    finally:
        handle.remove()
    d = a - b
    d[:, m.cfg.tul.slot_id] = 0.0                  # masked -inf column on both sides
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    return float(d.abs().max())


@pytest.mark.parametrize("depth,plan_must_be_zero", [(1, True), (2, False)])
def test_plan_cell_seed_reaches_next_slot_only_at_depth_2(depth, plan_must_be_zero):
    """``h=1``, ``M=4``: at forced depth 1, a query slot's cells read the PREVIOUS slot
    ONLY through the history cell (index 0), and layer 0 of pass 1 computes every
    position's attention output from the ENTRY state, so the previous slot's PLAN cells
    (index 1-3) have not yet been mixed into ITS OWN history cell when slot k reads it —
    a plan cell's seed therefore moves span k+1's logits by EXACTLY 0.0. By depth 2, the
    previous slot's own within-slot mixing (layers >= 1 of pass 0, unrestricted for the
    own-slot term) has already folded its plan cells into its history cell, so pass 1's
    cross-slot read carries it forward and the plan perturbation moves the logits.
    A larger core (``n_core=4``, vs the composition fixture's 2) gives the two-hop path
    enough layers to clear a >1e-4 floor; the HISTORY-cell path clears it at n_core=2
    already (`test_lxtul_r_composition.py`'s own perturbation test uses n_core=2).
    """
    m = _hist_model(seed=12, h=1, n_core=4,
                    tul_kw=dict(slot_mean_depth=depth, slot_max_depth=depth))
    ids, inp, lab, layout = _batch(prefix_k=4, seed=3)
    nb = int(layout.bag_id[0].max())
    j = min(nb - 1, 7)
    slot_pert = j - 2
    assert slot_pert >= 0
    g = torch.Generator().manual_seed(777)
    delta = torch.randn(m.cfg.d_model, generator=g) * 50.0

    d_hist = _perturb_register_cell_and_measure(m, inp, layout, slot_pert, 0, 4, j, delta)
    d_plan = _perturb_register_cell_and_measure(m, inp, layout, slot_pert, 1, 4, j, delta)

    assert d_hist > 1e-4, \
        f"depth {depth}: the HISTORY cell's seed must reach span {j}, got {d_hist}"
    if plan_must_be_zero:
        assert d_plan == 0.0, \
            f"depth {depth}: a PLAN cell's seed must NOT reach span {j} yet, got {d_plan}"
    else:
        assert d_plan > 1e-4, \
            f"depth {depth}: a PLAN cell's seed must reach span {j} by now, got {d_plan}"


# ── 4. THE REACH STAIRCASE IS UNCHANGED ─────────────────────────────────────


@pytest.mark.parametrize("h", [0, 1])
@pytest.mark.parametrize("depth,first_dark", [(1, 3), (2, 4)])
def test_loop_reach1_register_reaches_exactly_one_slot_per_pass_with_history_streams(
        h, depth, first_dark):
    """The SAME one-slot-per-pass law as
    ``test_lxtul_r_composition.py::test_loop_reach1_register_reaches_exactly_one_slot_per_pass``,
    at ``h=0`` (the exact original arm) and ``h=1`` (this key on): a TOKEN-level span
    perturbation moves the entire perturbed slot's cells (the register's shared pooling
    keys/values), so the span-level reach law is a property of the WHOLE slot's content,
    not of one cell — narrowing which cell carries it across the boundary does not change
    how many passes the content needs to arrive, only how much of it does. Batch seed 11 /
    span 4 give a comfortable margin over the 1e-4 floor at every (h, depth) combination
    (batch seed 3 / span 7, used elsewhere in this file, does not clear it at h=1 depth=2:
    ONE fewer bits-per-cell of bandwidth costs magnitude, not reach)."""
    m = _hist_model(seed=12, h=h, tul_kw=dict(slot_mean_depth=depth, slot_max_depth=depth))
    ids, inp, lab, layout = _batch(prefix_k=4, seed=11)
    j = 4
    lit = _span_logit_delta(m, inp, layout, j, first_dark - 1)
    dark = _span_logit_delta(m, inp, layout, j, first_dark)
    assert lit > 1e-4, \
        f"h={h} depth {depth}: span j-{first_dark - 1} must reach span j, got {lit}"
    assert dark == 0.0, \
        f"h={h} depth {depth}: span j-{first_dark} must NOT reach span j, got {dark}"


# ── 5. THE REPULSION IS CHARGED ON PLAN STREAMS ALONE ───────────────────────


def test_plan_streams_slices_the_stream_axis_and_keeps_the_slot_axis():
    b, s, m, c = 2, 3, 4, 5
    h = 1
    t = torch.arange(b * s * m * c, dtype=torch.float32).reshape(b, s * m, c)
    tp, mp = plan_streams(t, m, h)
    assert mp == m - h
    assert tp.shape == (b, s * mp, c)
    back = t.reshape(b, s, m, c)
    want = back[:, :, h:].reshape(b, s * mp, c)
    assert torch.equal(tp, want)


def test_plan_streams_refusals():
    t = torch.zeros(1, 3 * 4, 5)
    with pytest.raises(ValueError, match="h must be >= 1"):
        plan_streams(t, 4, 0)
    with pytest.raises(ValueError, match="must be < m_cells"):
        plan_streams(t, 4, 4)
    with pytest.raises(ValueError, match="not divisible"):
        plan_streams(torch.zeros(1, 10, 5), 4, 1)


def test_repel_term_on_plan_streams_equals_the_hand_computed_term_and_full_k_stats_unmoved():
    c = 8
    valid = torch.ones(1, 2, dtype=torch.bool)
    h = 1
    mixes = (0.0, 0.25, 0.5, 1.0)
    traj = []
    for mix in mixes:
        base = torch.eye(4, c).unsqueeze(0).expand(2, 4, c).contiguous().clone()
        st = (1.0 - mix) * base + mix * torch.ones_like(base)
        if 0.0 < mix < 1.0:
            # Break the fixture's symmetry on the HISTORY stream alone at the charged
            # passes: with four interchangeable streams every pair has the same cosine,
            # so the plan-only mean (streams 1..3) would EQUAL the full-K mean and the
            # final assertion below could not tell the two apart. Stream 0 gets a
            # ramp; streams 1..3 (what the hand term reads) are untouched, and the
            # t0 / t3 endpoints (mix 0 and 1) keep their exact 0 / 1 cosines.
            st[:, 0] = st[:, 0] + 0.37 * torch.arange(c, dtype=st.dtype)
        traj.append(_state(st))

    stats_full: dict = {}
    with torch.no_grad():
        fan_repel_term(traj, valid, 4, n_passes=2, stats=stats_full)

    traj_plan = [plan_streams(t, 4, h)[0] for t in traj]
    term_plan = fan_repel_term(traj_plan, valid, 4 - h, n_passes=2, stats=None)

    # by hand: streams 1..3 of the SAME underlying tensors, sliced directly (not through
    # `plan_streams`, so this is an independent check of what the term SHOULD read).
    hand = []
    for t, mix in enumerate(mixes):
        if not (1 <= t <= 2):
            continue
        base = torch.eye(4, c).unsqueeze(0).expand(2, 4, c).contiguous().clone()
        st = (1.0 - mix) * base + mix * torch.ones_like(base)
        st_plan = st[:, h:]
        hand.append(fan_stream_cos(_state(st_plan), valid, 4 - h))
    hand_term = torch.stack(hand).mean()
    assert term_plan.item() == pytest.approx(hand_term.item(), abs=1e-6)

    # the full-K instrument is unaffected by the plan-only charged term: it was computed
    # entirely separately, on the UNSLICED trajectory.
    assert sorted(stats_full) == ["repel_terms", "stream_cos_t0", "stream_cos_t1",
                                  "stream_cos_t2", "stream_cos_t3"]
    assert stats_full["stream_cos_t0"] == pytest.approx(0.0, abs=1e-6)
    assert stats_full["stream_cos_t3"] == pytest.approx(1.0, abs=1e-6)
    # and the plan-only term is a DIFFERENT number from the full-K one at t1/t2 (streams
    # 0..3 vs 1..3 are not the same set, so a term that silently read the full K would
    # coincide with a hand-mean of stats_full's t1/t2 instead of the plan-only hand term).
    full_hand = 0.5 * (stats_full["stream_cos_t1"] + stats_full["stream_cos_t2"])
    assert term_plan.item() != pytest.approx(full_hand, abs=1e-9)


def test_history_streams_model_reports_full_k_instruments_and_plan_rank():
    m = _hist_model(seed=7, h=1, tul_kw=dict(fan_repel_lambda=0.5, fan_repel_mode="cos"))
    ids, inp, lab, layout = _batch(prefix_k=4, seed=2)
    with torch.no_grad():
        out = m.eval()(inp, labels=lab, slot_layout=layout)
    assert any(str(k).startswith("fan_stream_cos_t") for k in out)
    assert any(str(k).startswith("fan_plan_rank_t") for k in out), \
        "eval forward with fan_history_streams > 0 must report fan/plan_rank_t{t}"
    assert any(str(k).startswith("fan_stream_rank_t") for k in out)


# ── 6. REFUSALS ───────────────────────────────────────────────────────────────


def test_function_level_refusals():
    with pytest.raises(ValueError, match="reach=0"):
        slot_cell_relation(3, 4, torch.device("cpu"), reach=0, history_streams=1)
    with pytest.raises(ValueError, match="must be < m_cells"):
        slot_cell_relation(3, 4, torch.device("cpu"), reach=1, history_streams=4)


def test_config_level_refusals():
    with pytest.raises(ValueError, match="tul.fan_history_streams=1 needs tul.loop_reach"):
        _hist_tul(h=1, loop_reach=0)
    with pytest.raises(ValueError, match="fan_k - 2"):
        _hist_tul(h=3)                                 # fan_k=4 (from _r1_tul), h<=2 only
    with pytest.raises(ValueError, match="tul.fan_k=0"):
        TULConfig(prefix_k=1, slot_id=4, fan_history_streams=1)
    with pytest.raises(ValueError, match="must be >= 0"):
        _hist_tul(h=-1)
    # h=2 (== fan_k - 2) is the boundary and must NOT raise.
    _hist_tul(h=2)


# ── 7. A TRAINING STEP, AND THE SHIPPED CONFIG COMPOSES ─────────────────────


def test_history_streams_train_step_is_finite_with_fan_mix_all():
    ids, inp, lab, layout = _batch(prefix_k=4, seed=2)
    m = _hist_model(seed=11, h=1).train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    for step in range(2):
        opt.zero_grad()
        out = m(inp, labels=lab, slot_layout=layout)
        loss = out["loss"]
        assert torch.isfinite(loss), f"step {step}: loss is not finite ({loss})"
        loss.backward()
        n_finite_grads = 0
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "a gradient is not finite"
                n_finite_grads += 1
        assert n_finite_grads > 0, f"step {step}: no parameter received a gradient"
        opt.step()


def test_the_shipped_config_composes_through_hydra_and_build_tul_runtime():
    import os

    from hydra import compose, initialize_config_dir

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(root, "morph", "configs")):
        cfg = compose(config_name="tul_slot_spandec_strict_fan4_all_reach1_hist1")
    rt = build_tul_runtime(cfg)
    assert rt.model_cfg.fan_history_streams == 1
    assert rt.model_cfg.fan_k == 4
    assert rt.model_cfg.loop_reach == 1


def test_unknown_key_still_raises_and_the_shipped_configs_use_known_keys_only():
    with pytest.raises(ValueError, match="fan_history_stream"):
        reject_unknown_tul_keys({"fan_history_stream": 1})    # a plausible typo
    import glob

    from omegaconf import OmegaConf
    for path in sorted(glob.glob("morph/configs/*.yaml")):
        raw = OmegaConf.load(path)
        tc = raw.get("tul", None)
        if tc is None or not hasattr(tc, "keys"):
            continue
        reject_unknown_tul_keys(tc)
