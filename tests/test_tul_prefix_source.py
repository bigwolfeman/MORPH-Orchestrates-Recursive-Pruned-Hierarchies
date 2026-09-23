"""`tul.prefix_source` — what each of a slot's coda cells carries.

"exit" is the shipped write: every one of a slot's ``prefix_k`` cells holds the loop's
EXIT state through its own ``W_prefix[k]``. "trajectory" gives cell ``k`` (``k < K-1``)
the state AFTER PASS ``k+1`` and keeps the EXIT in the last cell, so every written pass
gets its own reader and its own direct gradient edge into the coda's CE. "exit_repeat" is
the matched-count CONTROL: the same cells, the same parameters, the same zero-init
per-cell embedding, and the exit in all of them.

WHAT THIS FILE HAS TO PROVE, in the order it matters:

1. OFF IS NOTHING. `prefix_source: exit` is bit-identical to the tree before the knob
   existed. Verified against commit `d778845` on 2026-09-13 by running the same fixture in
   a worktree of that commit and diffing loss, the finite logit sum, the total gradient sum
   and the state_dict key count on FOUR configurations (strict k2, strict k4, restrict k2,
   plain-TUL k2): identical to the last printed digit. `test_exit_is_pinned_to_the_pre_knob_values`
   carries those numbers so the claim does not depend on that worktree still existing.
2. THE CELLS HOLD WHAT THE DOCSTRING SAYS. Cell ``k`` is `db_traj[k+1]`, the last cell is
   the exit, a forced depth ``d`` writes exactly ``min(d, K)`` non-pad cells, and the exit
   is present at EVERY forced depth.
3. A PAD IS INERT. Its carrier is exactly zero — `E_pass` included, which is not free:
   the embedding is added AFTER the projection and the CCA conv carries a cell into the
   later cells of its OWN slot, so a pad that kept `E_pass[k]` would hand the depth draw
   to the exit cell. Two-sided: writing a constant into a pad moves NO logit, writing the
   same constant into a WRITTEN cell moves one.
4. EVERY WRITTEN PASS RECEIVES GRADIENT, which is the arm's entire claim.
5. THE STRICT GEOMETRY STILL HOLDS, unchanged, at prefix_k 6 under trajectory.
6. THE EVAL TOOLING RUNS: forced per-slot depths, the `slot_mean_depth` lever the depth
   sweep uses, and the zero / shuffle / all_slots plan ablations `worth_profile` reads.

CPU only, fp32, `use_kernels=False`, tiny config, the `tests/test_tul_strict_geometry.py`
fixtures. fp32 and not fp64: `_window_fallback` hands SDPA an fp32 mask whatever the dtype
of q, which is silently WRONG at fp64 (morph/model/CLAUDE.md).

Record: lab/experiments/planned/2026-09-13-arc-trajectory-prefix.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (BoundaryRule, TulLayoutSpec, slot_layout_from_ids)

V = 64
DOT = 10
K6 = 6


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=3,
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
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _spec(prefix_k: int = 2) -> TulLayoutSpec:
    return TulLayoutSpec(seq_len=64, prefix_k=prefix_k, max_slots=10, slot_id=4)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5                      # slot_id must not occur in the stream
    ids[:, ::8] = DOT                      # a boundary every 8 tokens
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
                tg_geometry="strict", emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0)
    base.update(kw)
    return TULConfig(**base)


def _model(seed: int = 1234, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    # `BigramEmbedding.lambdas` is ZERO-init, so on a fresh model the bigram route is
    # switched off and a perturbation test would pass without exercising it at all
    # (the `test_span_mask_leak.py` lesson).
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _batch(prefix_k: int = 2, seed: int = 0):
    ids = _ids(seed=seed)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec(prefix_k))
    return ids, inp, lab, layout


def _finite_logit_sum(lg: torch.Tensor) -> float:
    """The slot_id column is forced to −inf at every position (spec §3.1)."""
    return float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

def test_exit_is_the_default_and_builds_no_parameter():
    m = _model()
    assert m.cfg.tul.prefix_source == "exit"
    assert m.tul.E_pass is None, "the pass-index embedding must not exist at 'exit'"
    assert not any(k.endswith("E_pass") for k in m.state_dict())


@pytest.mark.parametrize("tag,kw,loss,logit_sum,grad_sum,n_keys", [
    ("strict_k2", dict(tg_geometry="strict"),
     5.044249534606934, 842.074198674527, 968.7720451450658, 211),
    ("strict_k4", dict(tg_geometry="strict", prefix_k=4),
     5.020976543426514, 1136.2373420511503, 962.8389462181175, 211),
    ("restrict_k2", dict(tg_geometry="restrict"),
     4.719430446624756, 802.7991237437302, 865.1620777188826, 211),
    ("plain_tul_k2", dict(tg_restrict=False, tg_geometry="restrict"),
     4.7814178466796875, 1107.3154941663088, 912.4137765946576, 256),
])
def test_exit_is_pinned_to_the_pre_knob_values(tag, kw, loss, logit_sum, grad_sum, n_keys):
    """The OFF state, pinned to numbers measured on the tree BEFORE `prefix_source`.

    Produced 2026-09-13 by running this exact fixture in a `git worktree` of `d778845`
    (the commit this change sits on) and again on the working tree: identical. The pin is
    here so a later refactor of `prefix_project` cannot move the shipped write in silence.

    2026-09-23: the `cells is None` write moved from a broadcast matmul (which expanded
    W_prefix to [B, S, K, C, C] and saved it for the backward: 1.5 GB at the panel shape,
    6.0 GB under LXTUL-GK) to K plain matmuls. Loss and logits are unchanged bit for bit;
    the four grad sums moved by 1e-11 to 2e-9 relative (fp32 summation order of the
    W_prefix gradient) and are re-pinned to the new tree's values.
    """
    k = int(kw.get("prefix_k", 2))
    _ids0, inp, lab, layout = _batch(k)
    m = _model(**kw).train()
    torch.manual_seed(7)
    res = m(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    assert res["loss"].item() == loss
    assert _finite_logit_sum(lg) == logit_sum
    assert g == grad_sum
    assert len(m.state_dict()) == n_keys


def test_exit_repeat_at_init_is_the_exit_model_at_the_same_width():
    """`E_pass` is zeros, so at step 0 the control IS `exit` at prefix_k 6.

    This is what makes the pair one factor: after a step the two differ only by what the
    embedding has learned, and `trajectory` differs from `exit_repeat` only by content.
    """
    _ids0, inp, lab, layout = _batch(K6)
    a = _model(prefix_k=K6)
    b = _model(prefix_k=K6, prefix_source="exit_repeat")
    assert b.tul.E_pass is not None and torch.equal(b.tul.E_pass, torch.zeros(K6, 64))
    with torch.no_grad():
        la = a(inp, labels=lab, slot_layout=layout)
        lb = b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(la["loss"], lb["loss"])
    with torch.no_grad():
        assert torch.equal(a(inp, labels=None, slot_layout=layout)["logits"],
                           b(inp, labels=None, slot_layout=layout)["logits"])


def test_exit_repeat_is_not_exit_once_the_embedding_moves():
    """Two-sided: if `E_pass` were dead the test above would be vacuous."""
    _ids0, inp, lab, layout = _batch(K6)
    a = _model(prefix_k=K6)
    b = _model(prefix_k=K6, prefix_source="exit_repeat")
    with torch.no_grad():
        b.tul.E_pass.normal_(std=0.5)
    with torch.no_grad():
        assert not torch.equal(a(inp, labels=None, slot_layout=layout)["logits"],
                               b(inp, labels=None, slot_layout=layout)["logits"])


# ── 2. THE CELLS HOLD WHAT THE DOCSTRING SAYS ────────────────────────────────

def _cells_of(m: MORPHTransformer, inp, layout, depths=None):
    """``(cells, pad_cell, pad_pos, h_slots, realised_depths)`` from the SHIPPED path.

    `_tul_core` and `_tul_prefix_cells` are called exactly as `_forward_tul` calls them,
    including the strict PRELUDE relation — a bare `_tul_front` would run the prelude
    unrestricted and every state downstream of it would be off-distribution (the
    2026-09-13 horizon-grid defect, `_tul_tg_kwargs`' docstring).
    """
    fkw, freset, _ckw, _creset = m._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        _xn, h_slots, d, _g, db_traj, _gr, _mk = m._tul_core(
            x, x0, bg, layout, input_ids=inp, slot_depths=depths)
        cells, pad_cell, pad_pos = m._tul_prefix_cells(h_slots, db_traj, d, layout)
    return cells, pad_cell, pad_pos, h_slots, d, db_traj


def test_trajectory_cells_hold_the_recorded_per_pass_states():
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=6,
               slot_depth_fixed=5)
    _ids0, inp, _lab, layout = _batch(K6)
    cells, pad, _pp, h_slots, d, traj = _cells_of(m, inp, layout)
    assert int(d[layout.slot_valid].min()) == 5 and int(d.max()) == 5
    for k in range(K6 - 1):
        w = layout.slot_valid & (d >= k + 2)
        if not bool(w.any()):
            continue
        assert torch.equal(cells[:, :, k][w], traj[k + 1][w]), (
            f"cell {k} does not hold the state after pass {k + 1}")
        # ... and it is NOT the exit, which is the whole point of a per-pass cell.
        assert not torch.equal(cells[:, :, k][w], h_slots[w])
    assert torch.equal(cells[:, :, K6 - 1], h_slots), "the last cell must be the EXIT"
    assert not bool(pad[:, :, K6 - 1][layout.slot_valid].any())


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 8])
def test_a_forced_depth_writes_min_d_k_cells_and_pads_the_rest(d):
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8)
    _ids0, inp, _lab, layout = _batch(K6)
    forced = torch.full_like(layout.slot_index, d)
    cells, pad, _pp, h_slots, got, _traj = _cells_of(m, inp, layout, depths=forced)
    assert int(got[layout.slot_valid].min()) == d
    n_written = (~pad)[layout.slot_valid].sum(dim=-1)
    assert torch.equal(n_written, torch.full_like(n_written, min(d, K6))), (
        f"depth {d} must write exactly min(d, {K6}) cells, got {n_written.tolist()[:4]}")
    # The EXIT is present at every depth, and always in the SAME cell.
    assert torch.equal(cells[:, :, K6 - 1], h_slots)
    # A pad's carrier source is exactly zero.
    for k in range(K6 - 1):
        p = pad[:, :, k] & layout.slot_valid
        if bool(p.any()):
            assert float(cells[:, :, k][p].abs().max()) == 0.0


def test_exit_repeat_cells_are_all_equal_to_the_exit_and_pad_nothing():
    m = _model(prefix_k=K6, prefix_source="exit_repeat", slot_max_depth=6)
    _ids0, inp, _lab, layout = _batch(K6)
    cells, pad, pad_pos, h_slots, _d, _t = _cells_of(m, inp, layout)
    for k in range(K6):
        assert torch.equal(cells[:, :, k], h_slots)
    assert pad is None and pad_pos is None, (
        "a mode that pads nothing must return None, so the caller's masking is a "
        "Python-level branch and not all-False tensor work on the hot path")


def test_entry_exit_carries_the_entry_in_cell_zero_and_the_exit_everywhere_else():
    """The information control: `I((z_1..z_T); Y) = I(z_1; Y)`, so the entry is the
    ceiling the trajectory arm has to be read against."""
    m = _model(prefix_k=K6, prefix_source="entry_exit", slot_max_depth=6)
    _ids0, inp, _lab, layout = _batch(K6)
    cells, pad, pad_pos, h_slots, _d, traj = _cells_of(m, inp, layout)
    assert pad is None and pad_pos is None
    assert torch.equal(cells[:, :, 0], traj[0]), "cell 0 is not the loop's ENTRY state"
    v = layout.slot_valid
    assert not torch.equal(cells[:, :, 0][v], h_slots[v]), (
        "the entry equals the exit on this fixture — the control is vacuous here")
    for k in range(1, K6):
        assert torch.equal(cells[:, :, k], h_slots)


def test_the_entry_state_is_core_init_of_the_gathered_prelude():
    """Pins WHAT `db_traj[0]` is, so the control cannot quietly become something else."""
    from morph.model.tul import gather_valid
    m = _model(prefix_k=K6, prefix_source="entry_exit", slot_max_depth=6)
    _ids0, inp, _lab, layout = _batch(K6)
    fkw, freset, _c, _r = m._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        xn, _h, _d, _g, traj, _gr, _mk = m._tul_core(x, x0, bg, layout, input_ids=inp)
        want = m.core_init(gather_valid(xn, layout.slot_index, layout.slot_valid))
    assert torch.equal(traj[0], want)


def test_entry_exit_and_exit_repeat_differ_in_exactly_one_cell():
    m_a = _model(prefix_k=K6, prefix_source="entry_exit", slot_max_depth=6)
    m_b = _model(prefix_k=K6, prefix_source="exit_repeat", slot_max_depth=6)
    _ids0, inp, _lab, layout = _batch(K6)
    ca, _p, _q, _h, _d, _t = _cells_of(m_a, inp, layout)
    cb, _p, _q, _h, _d, _t = _cells_of(m_b, inp, layout)
    assert not torch.equal(ca[:, :, 0], cb[:, :, 0])
    for k in range(1, K6):
        assert torch.equal(ca[:, :, k], cb[:, :, k])


def test_the_pad_positions_land_on_the_rows_cell_positions():
    """`pad_pos` is `pad_cell` at ROW positions, and nowhere else."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8)
    _ids0, inp, _lab, layout = _batch(K6)
    forced = torch.full_like(layout.slot_index, 2)      # cells 1..4 are pads
    _c, pad, pad_pos, _h, _d, _t = _cells_of(m, inp, layout, depths=forced)
    assert bool((pad_pos & ~layout.slot_mask).sum() == 0), "a TOKEN position was flagged"
    B, S = layout.slot_index.shape
    for b in range(B):
        for s in range(S):
            if not bool(layout.slot_valid[b, s]):
                continue
            base = int(layout.slot_index[b, s])
            for k in range(K6):
                assert bool(pad_pos[b, base + k]) == bool(pad[b, s, k])


# ── 3. A PAD IS INERT ────────────────────────────────────────────────────────

def _logits_with_cell_edit(m, inp, layout, which: str, const: float = 7.0):
    """Run the shipped forward with a constant written into PAD or WRITTEN cells."""
    real = m._tul_prefix_cells

    def patched(h_slots, db_traj, depths, lay):
        cells, pad, pad_pos = real(h_slots, db_traj, depths, lay)
        if which == "none":
            return cells, pad, pad_pos
        sel = pad if which == "pad" else (~pad)
        # never touch the EXIT cell, which both selections would otherwise disagree on
        sel = sel.clone()
        sel[:, :, -1] = False
        sel = sel & lay.slot_valid.unsqueeze(-1)
        v = sel.view(*sel.shape, *([1] * (cells.dim() - 3))).to(cells.dtype)
        return cells * (1 - v) + const * v, pad, pad_pos

    m._tul_prefix_cells = patched
    try:
        with torch.no_grad():
            return m(inp, labels=None, slot_layout=layout)["logits"].clone()
    finally:
        m._tul_prefix_cells = real


def test_editing_a_pad_cell_moves_no_logit_and_editing_a_written_one_does():
    """The pad is out of the key set AND out of the conv — two-sided, one fixture."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=3)                       # cells 2,3,4 are pads
    _ids0, inp, _lab, layout = _batch(K6)
    base = _logits_with_cell_edit(m, inp, layout, "none", const=0.0)
    with torch.no_grad():
        plain = m(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(base, plain), "the patch harness changed the forward by itself"
    padded = _logits_with_cell_edit(m, inp, layout, "pad")
    written = _logits_with_cell_edit(m, inp, layout, "written")
    dp = (padded - plain).abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
    dw = (written - plain).abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
    assert float(dp.max()) == 0.0, (
        f"a PAD cell reached the coda (max |delta| {float(dp.max()):.3e}) — it is not "
        "masked out of the key set, or the CCA conv carries it into its slot's later cells")
    assert float(dw.max()) > 0.0, "a WRITTEN cell reached nothing — the probe is blind"


def test_the_pad_cells_are_out_of_the_codas_key_set():
    """The structural half of "a pad is inert", on the SHIPPED coda relation.

    The logit test above cannot see this one on its own: a pad's carrier is also zeroed,
    so with the narrowing gone a pad still contributes a zero VALUE — but it contributes a
    softmax weight, and it is a key the conv and the value shift walk over. This asserts
    the relation directly, two-sided: a pad's column is closed for every query but itself,
    and a WRITTEN cell's column is still open.
    """
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=3)                       # cells 2,3,4 are pads
    _ids0, inp, _lab, layout = _batch(K6)
    _c, pad, pad_pos, _h, _d, _t = _cells_of(m, inp, layout)
    _f, _fr, ckw, _cr = m._tul_tg_kwargs(layout)
    narrowed = m._tul_pad_cell_narrow(ckw, pad_pos)
    assert bool(pad_pos.any()), "fixture: no pad cell at this depth"
    L = pad_pos.shape[1]
    eye = torch.eye(L, dtype=torch.bool).view(1, 1, L, L)
    for key in ("tg_allow", "tg_comp_allow"):
        a_, b_ = ckw[key], narrowed[key]
        cols = pad_pos.view(pad_pos.shape[0], 1, 1, L) & ~eye
        assert not bool((b_ & cols).any()), f"{key}: a pad cell is still a readable key"
        # two-sided: the NON-pad columns are untouched, so the narrowing cut only pads
        assert torch.equal(b_ & ~cols, a_ & ~cols), f"{key}: the narrowing cut a real cell"
        assert bool((a_ & cols).any()), "the unnarrowed relation never allowed a pad — vacuous"


def test_the_shipped_coda_is_handed_the_narrowed_relation():
    """The SHIPPED PATH, not the helper. `test_the_pad_cells_are_out_of_the_codas_key_set`
    calls `_tul_pad_cell_narrow` itself, so it stays green when the FORWARD stops calling
    it — measured: that sabotage was MISSED on 2026-09-13, the same defect class as the
    core-token aux test's C1 miss. This spies on what `_back_region` actually receives."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=3)
    _ids0, inp, _lab, layout = _batch(K6)
    _c, _p, pad_pos, _h, _d, _t = _cells_of(m, inp, layout)
    assert bool(pad_pos.any()), "fixture: no pad cell at this depth"
    seen: dict = {}
    real = m._back_region

    def spy(x, x0, bg, ids=None, inject_keep=None, attn_kwargs=None, **kw):
        seen["kw"] = attn_kwargs
        return real(x, x0, bg, ids, inject_keep=inject_keep, attn_kwargs=attn_kwargs, **kw)

    m._back_region = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
    finally:
        m._back_region = real
    L = pad_pos.shape[1]
    eye = torch.eye(L, dtype=torch.bool).view(1, 1, L, L)
    cols = pad_pos.view(pad_pos.shape[0], 1, 1, L) & ~eye
    for key in ("tg_allow", "tg_comp_allow"):
        got = seen["kw"][key]
        assert not bool((got & cols).any()), (
            f"the coda was handed a relation that still lets a query read a PAD cell "
            f"({key}) — the forward is not calling `_tul_pad_cell_narrow`")


def test_a_pad_cell_keeps_its_own_self_edge():
    """Deliberate: the compressed branch keeps ``j == i``, and an all-`-inf` softmax row is
    0 under SDPA and NaN under an explicit one (morph/model/CLAUDE.md). Nothing reads a pad
    cell's output, so its self-edge costs nothing and removes a NaN class."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=3)
    _ids0, inp, _lab, layout = _batch(K6)
    _c, _p, pad_pos, _h, _d, _t = _cells_of(m, inp, layout)
    _f, _fr, ckw, _cr = m._tul_tg_kwargs(layout)
    nb = m._tul_pad_cell_narrow(ckw, pad_pos)["tg_comp_allow"]
    diag = nb[:, 0].diagonal(dim1=-2, dim2=-1)                      # [B, L]
    assert bool(diag[pad_pos].all()), "a pad cell lost its own self-edge"


def test_a_pad_cells_carrier_is_exactly_zero_including_the_pass_embedding():
    """`E_pass` is added AFTER the projection, so zeroing the SOURCE is not enough.

    Measured before the fix: with a non-zero `E_pass` the pad cells' own gradient summed
    to 7407 on this fixture against 0.25 on the `exit_repeat` twin, and the CCA conv
    carried that learned constant into the EXIT cell of the same slot.
    """
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=2)
    with torch.no_grad():
        m.tul.E_pass.normal_(std=1.0)
    _ids0, inp, _lab, layout = _batch(K6)
    seen: dict = {}
    real = m.tul.prefix_project

    def spy(h_slots, lay, l_total, cells=None):
        v, p = real(h_slots, lay, l_total, cells=cells)
        seen["v"], seen["p"] = v, p
        return v, p

    m.tul.prefix_project = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
    finally:
        m.tul.prefix_project = real
    v_live = seen["v"].clone()
    # `> 0` is vacuous — the projection is non-zero whether or not E_pass was added, and an
    # independent review pointed out that deleting the add keeps such a test green. So the
    # claim is stated as a DIFFERENCE: zero E_pass and the same forward must produce a
    # different carrier, and the pad cells specifically must go to exactly zero.
    with torch.no_grad():
        m.tul.E_pass.zero_()
    seen.clear()
    m.tul.prefix_project = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
    finally:
        m.tul.prefix_project = real
    v_zero = seen["v"]
    assert not torch.equal(v_live, v_zero), (
        "zeroing E_pass changed nothing: the per-cell pass embedding never reaches the "
        "carrier, so `trajectory` and `exit_repeat` are not the one-factor pair they claim "
        "to be")
    # A pad's SOURCE state is zeroed in `_tul_prefix_cells`, so with E_pass zeroed its
    # projected carrier is EXACTLY zero — and with E_pass live it is exactly the E_pass
    # term, which is the learned constant that used to announce a slot's depth.
    assert float(v_zero.abs().min()) == 0.0
    d = (v_live - v_zero).abs()
    assert float(d.max()) > 0.0


# ── 4. EVERY WRITTEN PASS RECEIVES GRADIENT ──────────────────────────────────

def test_every_written_cell_carries_gradient_and_every_pad_carries_none():
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=4).train()
    _ids0, inp, lab, layout = _batch(K6)
    grabbed: dict = {}
    real = m._tul_prefix_cells

    def patched(h_slots, db_traj, depths, lay):
        cells, pad, pad_pos = real(h_slots, db_traj, depths, lay)
        leaf = cells.detach().clone().requires_grad_(True)
        grabbed["leaf"], grabbed["pad"] = leaf, pad
        return leaf, pad, pad_pos

    m._tul_prefix_cells = patched
    try:
        torch.manual_seed(3)
        out = m(inp, labels=lab, slot_layout=layout)
        g, = torch.autograd.grad(out["loss"], grabbed["leaf"])
    finally:
        m._tul_prefix_cells = real
    pad, valid = grabbed["pad"], layout.slot_valid
    per_cell = g.flatten(3).abs().sum(-1) if g.dim() > 3 else g.abs().sum(-1)   # [B, S, K]
    for k in range(K6):
        w = valid & ~pad[:, :, k]
        if bool(w.any()):
            assert float(per_cell[:, :, k][w].max()) > 0.0, (
                f"cell {k} receives NO gradient — that pass has no reader, which is the "
                "one thing this arm exists to create")
        p = valid & pad[:, :, k]
        if bool(p.any()):
            assert float(per_cell[:, :, k][p].abs().max()) == 0.0, (
                f"a PAD at cell {k} receives gradient — it reached the loss")


def test_the_gradient_reaches_each_recorded_pass_state():
    """The other half: `db_traj[t]` itself, not the cell built from it."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8,
               slot_depth_fixed=5).train()
    _ids0, inp, lab, layout = _batch(K6)
    fkw, freset, ckw, _cr = m._tul_tg_kwargs(layout)
    x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    _xn, h_slots, d, _g, traj, _gr, _mk = m._tul_core(x, x0, bg, layout, input_ids=inp)
    cells, pad, _pp = m._tul_prefix_cells(h_slots, traj, d, layout)
    gs = torch.autograd.grad(cells.sum(), [traj[t] for t in range(1, K6)],
                             allow_unused=True, retain_graph=True)
    for t, gt in enumerate(gs, start=1):
        assert gt is not None and float(gt.abs().sum()) > 0.0, (
            f"pass {t}'s recorded state is not in the cells' graph")


# ── 5. THE STRICT GEOMETRY STILL HOLDS ───────────────────────────────────────

def _edit(ids: np.ndarray, layout, row: int, span: int) -> np.ndarray:
    bag = layout.bag_id[row].numpy()
    tok = ~layout.slot_mask[row].numpy()
    pos = np.flatnonzero((bag == span) & tok)
    assert pos.size >= 3
    p = int(pos[len(pos) // 2])
    out = ids.copy()
    raw = int(tok[:p].sum())
    assert out[row, raw] not in (DOT, 11)
    out[row, raw] = 12 if out[row, raw] != 12 else 13
    return out


def _leak(m, prefix_k: int, plan_mode: str = "zero", row: int = 0, span: int = 0):
    ids = _ids()
    spec = _spec(prefix_k)

    def run(i):
        inp, _l, lay, _s = slot_layout_from_ids(i, _rule(), spec)
        with torch.no_grad():
            return m.tul_forward_ablated(inp, None, lay, plan_mode=plan_mode)["logits"], lay

    a, lay = run(ids)
    b, _ = run(_edit(ids, lay, row, span))
    bag, tok = lay.bag_id[row], ~lay.slot_mask[row]
    d = (a[row] - b[row]).abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
    own = tok & (bag == span)
    other = tok & (bag != span)
    return float(d[other].max()), float(d[own].max())


def test_the_strict_leak_test_still_holds_under_trajectory():
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=6)
    out, own = _leak(m, K6)
    assert own > 0.0, "fixture: the edit did not move its OWN span"
    assert out == 0.0, (
        "with the loop's write zeroed a span-0 token id moved a later span — the "
        "trajectory cells opened a route strict had cut")


def test_the_restrict_control_leaks_through_the_same_probe():
    """Two-sided: the same probe on a NON-strict trajectory model must see the route."""
    m = _model(prefix_k=K6, prefix_source="trajectory", tg_geometry="restrict",
               slot_max_depth=6)
    out, own = _leak(m, K6)
    assert own > 0.0 and out > 0.0, "the probe measures nothing"


# ── 6. THE EVAL TOOLING RUNS ─────────────────────────────────────────────────

@pytest.mark.parametrize("src", ["trajectory", "exit_repeat", "entry_exit"])
@pytest.mark.parametrize("mode", ["normal", "zero", "shuffle", "all_slots"])
def test_the_worth_profile_modes_run_on_every_prefix_source(mode, src):
    m = _model(prefix_k=K6, prefix_source=src, slot_max_depth=6)
    _ids0, inp, lab, layout = _batch(K6)
    with torch.no_grad():
        res = m.tul_forward_ablated(inp, lab, layout, plan_mode=mode)
    assert torch.isfinite(res["loss"])


def test_zero_and_all_slots_are_the_same_ablation_under_strict_trajectory():
    """The strict geometry's CHECK, carried over to the wider write."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=6)
    _ids0, inp, lab, layout = _batch(K6)
    with torch.no_grad():
        a = m.tul_forward_ablated(inp, lab, layout, plan_mode="zero")["loss"]
        b = m.tul_forward_ablated(inp, lab, layout, plan_mode="all_slots")["loss"]
    assert torch.equal(a, b)


def test_the_depth_sweeps_slot_mean_depth_lever_moves_the_forward():
    """`core_depth_sweep.py` forces depth by writing `model.cfg.tul.slot_mean_depth`."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8)
    _ids0, inp, lab, layout = _batch(K6)
    seen = []
    for d in (1, 3, 6):
        m.cfg.tul.slot_mean_depth = d
        with torch.no_grad():
            seen.append(float(m.tul_forward_ablated(inp, lab, layout)["loss"]))
    assert len(set(seen)) == 3, f"the depth lever did nothing: {seen}"


def test_a_forced_slot_depth_table_runs_and_is_the_identity_at_the_eval_depth():
    """`slot_depth_isolation.py`'s lever, and its own identity check."""
    m = _model(prefix_k=K6, prefix_source="trajectory", slot_max_depth=8)
    m.cfg.tul.slot_mean_depth = 4
    _ids0, inp, lab, layout = _batch(K6)
    with torch.no_grad():
        a = m.tul_forward_ablated(inp, lab, layout)["loss"]
        b = m.tul_forward_ablated(inp, lab, layout,
                                  slot_depths=torch.full_like(layout.slot_index, 4))["loss"]
    assert torch.equal(a, b)


# ── refusals ─────────────────────────────────────────────────────────────────

def test_an_unknown_prefix_source_raises():
    with pytest.raises(ValueError, match="prefix_source"):
        TULConfig(prefix_source="last_two")


def test_trajectory_without_an_allow_relation_raises():
    with pytest.raises(NotImplementedError, match="coda ALLOW relation"):
        TULConfig(prefix_source="trajectory", prefix_k=4, tg_restrict=False,
                  tg_geometry="restrict")


@pytest.mark.parametrize("src", ["trajectory", "entry_exit"])
def test_a_two_state_source_at_prefix_k_1_raises(src):
    with pytest.raises(ValueError, match="prefix_k >= 2"):
        TULConfig(prefix_source=src, prefix_k=1, tg_geometry="strict")


def test_entry_exit_needs_no_allow_relation():
    """It pads nothing, so unlike `trajectory` it has nothing to mask out."""
    TULConfig(prefix_source="entry_exit", prefix_k=4, tg_restrict=False,
              tg_geometry="restrict")


@pytest.mark.parametrize("kw", [dict(tokens_through_core=True),
                                dict(loop_reads_tokens=True),
                                dict(db_loop=True)])
def test_trajectory_refuses_the_forwards_that_have_no_per_pass_write(kw):
    with pytest.raises(NotImplementedError):
        TULConfig(prefix_source="trajectory", prefix_k=4, tg_geometry="strict", **kw)


def test_exit_repeat_takes_the_same_refusals():
    with pytest.raises(NotImplementedError):
        TULConfig(prefix_source="exit_repeat", prefix_k=4, tokens_through_core=True)


def test_the_shipped_configs_compose_and_carry_the_knobs():
    """The `compose-every-config-before-queueing` rule: green units do not prove a start."""
    from hydra import compose, initialize_config_dir
    import os
    from morph.training.tul_setup import build_tul_runtime
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(root, "morph", "configs")):
        for name, src in (("tul_slot_spandec_strict_traj", "trajectory"),
                          ("tul_slot_spandec_strict_trajrep", "exit_repeat"),
                          ("tul_slot_spandec_strict_entryexit", "entry_exit")):
            cfg = compose(config_name=name, overrides=[])
            rt = build_tul_runtime(cfg)
            assert rt.model_cfg.prefix_source == src
            assert rt.model_cfg.prefix_k == 6
            assert rt.model_cfg.tg_geometry == "strict"
            assert rt.data_cfg.spec_for(cfg.data.seq_len).l_total == 1024 + 6 * 64
