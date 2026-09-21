"""`tul.slot_cells` — the Thought Register: M mutable looped cells per span, not one.

THE DEFECT. A row's written slot states sit at effective rank 5.7-7.3 in 1024 dimensions
with mean pairwise cosine 0.72-0.77 (the strict ruler 6.34 / 0.74; prefix-4 7.12), 1.7-4.8
in the 2026-09-10 geometry audit, 18-23 across 4,906 slots at step 0. The slots of a row
are near copies, and a single vector gives pass 2 nothing to relate pass 1's result TO.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. `slot_cells: 1` builds no module, draws no RNG and leaves the forward
   bit-identical (the shared pin in `tests/test_tul_prefix_source.py` carries the numbers).
2. THE ARM STARTS AT ITS RULER. `W_o` is zero and `P_cell` is zeros, so the register's
   term is EXACTLY 0 at step 0 and every cell's seed is the ruler's single seed. That is
   what makes the m4/sameinit pair one factor: they are identical until `W_o` moves.
3. THE SHAPE AND THE WRITE. The compact sequence is S*M cells; cell i lands in prefix cell
   i, 1:1, through the shared `W_prefix[i]`; the M cells of a slot loop at ONE depth.
4. THE IN-LOOP RELATION, AS EXECUTED. Cell i of slot k reads every cell of its OWN slot
   (including cells AFTER it) and every cell of earlier slots, and reads NO cell of a
   later slot. Held on the MASK and, two-sided and NUMERICALLY, on the FORWARD: the
   register's loss differs from plain flattened causal and equals an independently
   written all-true-within-slot mask, and perturbing cell 3 of a slot moves cell 0 of
   that slot (it does NOT under a causal build) while moving nothing in any earlier slot.
   The mask travels as `tg_relation`, which REPLACES a branch's causal term;
   `tg_allow`/`tg_comp_allow` only ever narrow and delivered plain flattened causal.
5. NO NaN. A tail-pad slot has no key to pool from; an all-`-inf` softmax row makes NaN in
   the BACKWARD, which is how `Q` and `W_k` first read `grad = nan` here. Two-sided: the
   pad's term is exactly 0 AND every register parameter receives a finite gradient.
6. THE STRICT GEOMETRY STILL HOLDS at M = 4.
7. THE INSTRUMENTS. `val/slot_cell_eff_rank` (the rank WITHIN a slot) exists and is the
   number to read first.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-thought-register.md
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, TULSlotRegister
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
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


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    base.update(kw)
    return TULConfig(**base)


def _batch(M: int = 1, seed: int = 0):
    spec = TulLayoutSpec(seq_len=64, prefix_k=max(M, 2), max_slots=10, slot_id=4)
    ids = _ids(seed=seed)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), spec)
    return ids, inp, lab, layout


def _model(M: int = 1, init: str = "distinct", seed: int = 99, **tul_kw):
    kw = dict(prefix_k=max(M, 2))
    if M > 1:
        kw.update(slot_cells=M, slot_cell_init=init)
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

def test_m1_is_the_default_and_builds_no_register():
    m = _model(1)
    assert m.cfg.tul.slot_cells == 1
    assert m.tul_register is None
    assert not any("tul_register" in k for k in m.state_dict())


def test_the_register_adds_exactly_five_tensors_and_moves_no_base_weight():
    """Every real draw comes from a PRIVATE generator (the `W_sent` precedent), so an m4
    model's base weights are byte-identical to a prefix_k-4 ruler's and the arm differs by
    the mechanism alone."""
    a, b = _model(1, prefix_k=4), _model(4)     # the SAME prefix_k: one factor
    ka, kb = set(a.state_dict()), set(b.state_dict())
    assert kb - ka == {"tul_register.Q", "tul_register.W_k.weight",
                       "tul_register.W_v.weight", "tul_register.W_o.weight",
                       "tul_register.P_cell"}
    assert ka - kb == set()
    for k in sorted(ka):
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), (
            f"{k} differs — the register drew from the GLOBAL RNG stream")


# ── 2. THE ARM STARTS AT ITS RULER ───────────────────────────────────────────

def test_the_register_term_is_exactly_zero_at_init():
    m = _model(4)
    _i, inp, _l, layout = _batch(4)
    with torch.no_grad():
        fkw, freset, _c, _r = m._tul_tg_kwargs(layout)
        x, _x0, _bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        xn = m.input_norm(x)
        term = m.tul_register(xn.mean(dim=2) if m._is_hc else xn, layout)
    assert float(term.abs().max()) == 0.0, (
        "W_o is not zero-init (or P_cell is not zeros): the register arm does not start "
        "at its ruler and the m4-vs-sameinit pair is not one factor")


def test_distinct_and_same_are_identical_at_init_and_differ_once_w_o_moves():
    _i, inp, lab, layout = _batch(4)
    a, b = _model(4, "distinct"), _model(4, "same")
    torch.manual_seed(3)
    la = a(inp, labels=lab, slot_layout=layout)["loss"]
    torch.manual_seed(3)
    lb = b(inp, labels=lab, slot_layout=layout)["loss"]
    assert torch.equal(la.detach(), lb.detach())
    for m in (a, b):
        with torch.no_grad():
            m.tul_register.W_o.weight.normal_(std=0.05, generator=torch.Generator().manual_seed(5))
    torch.manual_seed(3)
    la2 = a(inp, labels=lab, slot_layout=layout)["loss"]
    torch.manual_seed(3)
    lb2 = b(inp, labels=lab, slot_layout=layout)["loss"]
    assert not torch.equal(la2.detach(), lb2.detach()), (
        "the shared-query control is the same model as the arm — `slot_cell_init` is inert")


def test_the_registers_own_weights_do_not_depend_on_the_ambient_rng():
    """The private-generator claim, stated so a global draw fails it. Two m4 models built
    under DIFFERENT ambient seeds must carry the same register."""
    a, b = _model(4, seed=1), _model(4, seed=2)
    for n in ("Q", "W_k.weight", "W_v.weight", "W_o.weight", "P_cell"):
        assert torch.equal(a.state_dict()["tul_register." + n],
                           b.state_dict()["tul_register." + n]), (
            f"tul_register.{n} moved with the ambient seed — it is a GLOBAL draw")
    assert not torch.equal(a.state_dict()["embed.hybrid.euc_embed.weight"],
                           b.state_dict()["embed.hybrid.euc_embed.weight"]), (
        "fixture: the two seeds built the same model, so this proves nothing")


def test_building_the_register_moves_no_global_rng_state():
    """`nn.Linear` draws kaiming on the GLOBAL stream before the weight is overwritten.
    Three Linears = three draws, which would shift anything constructed after the module."""
    torch.manual_seed(11)
    before = torch.random.get_rng_state()
    TULSlotRegister(64, 4)
    assert torch.equal(before, torch.random.get_rng_state())


def test_one_query_is_stored_for_same_and_m_for_distinct():
    assert _model(4, "distinct").tul_register.Q.shape == (4, 64)
    assert _model(4, "same").tul_register.Q.shape == (1, 64)


# ── 3. THE SHAPE AND THE WRITE ───────────────────────────────────────────────

def _core_out(m, inp, layout):
    fkw, freset, _c, _r = m._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        return m._tul_core(x, x0, bg, layout, input_ids=inp)


def test_the_compact_sequence_is_s_times_m_cells():
    m = _model(4)
    _i, inp, _l, layout = _batch(4)
    _xn, h, depths, *_ = _core_out(m, inp, layout)
    S = layout.slot_index.shape[1]
    assert h.shape[1] == S * 4
    assert depths.shape[1] == S * 4


def test_a_slots_m_cells_loop_at_one_depth():
    m = _model(4)
    _i, inp, _l, layout = _batch(4)
    _xn, _h, depths, *_ = _core_out(m, inp, layout)
    S = layout.slot_index.shape[1]
    d = depths.view(depths.shape[0], S, 4)
    assert torch.equal(d, d[:, :, :1].expand_as(d)), (
        "the M cells of a slot drew different depths — they are one register and they "
        "iterate together")


def test_cell_i_is_written_into_prefix_cell_i():
    """1:1 through the shared `W_prefix[i]`, which is why prefix_k must equal slot_cells."""
    m = _model(4).eval()
    _i, inp, _l, layout = _batch(4)
    seen: dict = {}
    real = m.tul.prefix_project

    def spy(h_slots, lay, l_total, cells=None):
        seen["cells"] = cells
        return real(h_slots, lay, l_total, cells=cells)

    m.tul.prefix_project = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
    finally:
        m.tul.prefix_project = real
    c = seen["cells"]
    assert c is not None and c.shape[2] == 4, "the register did not reach prefix_project"
    _xn, h, _d, *_ = _core_out(m, inp, layout)
    S = layout.slot_index.shape[1]
    assert torch.equal(c, h.reshape(h.shape[0], S, 4, *h.shape[2:]))


def test_prefix_k_must_equal_slot_cells():
    with pytest.raises(ValueError, match="prefix_k"):
        TULConfig(slot_cells=4, prefix_k=2)


# ── 4. THE IN-LOOP RELATION ──────────────────────────────────────────────────

def _core_kwargs(m, inp, layout):
    seen: list = []
    real = m._apply_core_step

    def spy(*a, **kw):
        seen.append(kw.get("attn_kw"))
        return real(*a, **kw)

    m._apply_core_step = spy
    try:
        _core_out(m, inp, layout)
    finally:
        m._apply_core_step = real
    return seen[0]


def _relation_of(akw: dict) -> torch.Tensor:
    """The relation a core layer was handed, with the delivery contract checked.

    It must arrive as `tg_relation` — the ONE attention kwarg that REPLACES a branch's
    causal term. `tg_allow`/`tg_comp_allow` are ANDed into an already-causal relation and
    can only NARROW, so delivering this superset-of-causal mask through them executed as
    plain flattened causal (measured 2026-09-13, fixed the same day).
    """
    assert "tg_relation" in akw, (
        f"the register's relation arrived as {sorted(akw)} — a narrowing kwarg cannot "
        f"express a cell reading a LATER cell of its own slot")
    assert "tg_allow" not in akw and "tg_comp_allow" not in akw, (
        "a narrowing mask beside the relation would silently win on every entry the "
        "relation added")
    return akw["tg_relation"]


def test_a_cell_reads_its_own_slots_other_cells_and_earlier_slots_only():
    m = _model(4)
    _i, inp, _l, layout = _batch(4)
    akw = _core_kwargs(m, inp, layout)
    assert akw is not None, "the register built no in-loop relation"
    mask = _relation_of(akw[0])[0, 0]               # [S*M, S*M]
    S = layout.slot_index.shape[1]
    M = 4
    sl = torch.arange(S * M) // M
    for p in (0, 1, M + 2, 3 * M + 1, S * M - 1):
        row = mask[p]
        # every cell of its OWN slot, including the ones AFTER it — the half plain causal
        # on the flattened axis would get wrong
        own = (sl == sl[p])
        assert bool(row[own].all()), f"cell {p} cannot read all of its own slot"
        assert bool(row[sl < sl[p]].all()), f"cell {p} cannot read an earlier slot"
        assert not bool(row[sl > sl[p]].any()), f"cell {p} reads a LATER slot"
    # two-sided: plain causal on the flattened axis is NOT this relation
    L = mask.shape[0]
    causal = torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1)
    assert not torch.equal(mask, causal), "the relation is just flattened causal"


def test_the_relation_is_the_same_on_every_core_layer_without_a_reach_budget():
    m = _model(4)
    _i, inp, _l, layout = _batch(4)
    akw = _core_kwargs(m, inp, layout)
    assert len(akw) == m.cfg.n_core
    for d in akw[1:]:
        assert torch.equal(_relation_of(d), _relation_of(akw[0]))


def test_loop_reach_narrows_across_slots_and_keeps_the_slots_own_cells():
    m = _model(4, loop_reach=1)
    _i, inp, _l, layout = _batch(4)
    akw = _core_kwargs(m, inp, layout)
    M, S = 4, layout.slot_index.shape[1]
    sl = torch.arange(S * M) // M
    m0 = _relation_of(akw[0])[0, 0]
    p = 3 * M + 1
    assert bool(m0[p][sl == sl[p]].all()), "reach cut a slot's own cells"
    assert bool(m0[p][sl == sl[p] - 1].all()), "reach 1 must reach the previous slot"
    assert not bool(m0[p][sl < sl[p] - 1].any()), "reach 1 reached two slots back"
    m1 = _relation_of(akw[1])[0, 0]
    assert bool(m1[p][sl == sl[p]].all())
    assert not bool(m1[p][sl != sl[p]].any()), (
        "the later layers must be SLOT-local under a reach budget")


# ── 4b. THE RELATION AS EXECUTED — two-sided and NUMERICAL ───────────────────
#
# The mask tests above grade what `slot_cell_relation` BUILDS. Until 2026-09-13 the mask
# was handed to the attention branches as `tg_allow`/`tg_comp_allow`, which are ANDed into
# an already-causal relation and can only NARROW — and this mask is a SUPERSET of
# flattened causal, so it executed as plain flattened causal and a cell never read a LATER
# cell of its own slot. The mask tests passed the whole time. These do not.

@contextlib.contextmanager
def patch_relation(mode: str):
    """Swap `slot_cell_relation`'s `blk` for a named control, everywhere it is called.

    ``blk``     the shipped relation, ``slot(p) >= slot(q)``.
    ``flat``    plain flattened causal, ``p >= q`` — exactly what the forward executed
                before the fix, and the control that must give a DIFFERENT loss.
    ``alltrue`` an INDEPENDENTLY written all-true-within-slot mask, ``causal OR same
                slot``. It equals ``blk`` by construction, so the forward must give the
                SAME loss under it. Together the two sides say the executed relation is
                the documented one and not merely "some mask".
    """
    import morph.model.transformer as _T
    real = _T.slot_cell_relation

    def patched(n_slots, m_cells, device, r=0, lineage=False):
        # `lineage` is `tul.fan_lineage: "relation"` (a fan arm). Every model in this
        # file is a plain register, so it must arrive False here — accepted and ASSERTED
        # rather than swallowed, so a caller that starts passing True cannot have its
        # narrowing silently dropped by this stub.
        assert lineage is False, "the register fixtures build no fan lineage"
        blk, same = real(n_slots, m_cells, device, r)
        sm = n_slots * m_cells
        idx = torch.arange(sm, device=device)
        if mode == "flat":
            blk = (idx.unsqueeze(1) >= idx.unsqueeze(0)).view(1, 1, sm, sm)
        elif mode == "alltrue":
            sl = idx // m_cells
            blk = ((idx.unsqueeze(1) >= idx.unsqueeze(0))
                   | (sl.unsqueeze(1) == sl.unsqueeze(0))).view(1, 1, sm, sm)
        elif mode != "blk":
            raise ValueError(mode)
        return blk, same

    _T.slot_cell_relation = patched
    try:
        yield
    finally:
        _T.slot_cell_relation = real


def loss_under(mode: str, M: int = 4, seed: int = 99, **tul_kw) -> float:
    """One forward's loss with the cell relation delivered as `mode`."""
    with patch_relation(mode):
        m = _model(M, seed=seed, **tul_kw)
        with torch.no_grad():
            # W_o is zero-init, so at step 0 every cell carries the SAME seed and no
            # relation can tell the three modes apart. Move it first.
            m.tul_register.W_o.weight.normal_(std=0.05)
        _i, inp, lab, layout = _batch(M)
        torch.manual_seed(5)
        return float(m(inp, labels=lab, slot_layout=layout)["loss"].detach())


def test_the_executed_loop_relation_is_not_flattened_causal():
    """The half the old delivery silently dropped. Two-sided in ONE test so neither side
    can be satisfied by a forward that ignores the relation altogether."""
    blk = loss_under("blk")
    assert blk != loss_under("flat"), (
        "the register's relation gives the SAME loss as plain flattened causal — the "
        "in-slot forward read is not executing (it is being ANDed into causality again)")
    assert blk == loss_under("alltrue"), (
        "the register's relation differs from an all-true-within-slot mask — the "
        "executed relation is neither the documented one nor plain causal")


def test_a_reach_budget_still_narrows_the_executed_relation():
    """Fixture check for the pair above: with a reach budget the mask really does cut
    across slots, so `blk` must differ from the unbudgeted all-true-within-slot mask."""
    assert loss_under("blk", loop_reach=1) != loss_under("alltrue", loop_reach=1)


def core_exit(m, inp, layout, cell=None, mag: float = 1.0, seed: int = 5):
    """The loop's compact exit, optionally with ONE cell of the loop's INPUT nudged.

    The per-sample Poisson depth draw is RNG, so the seed is pinned around the call —
    without it the two runs differ by their depths and the probe reads noise.
    """
    real = m._apply_core_step
    n = {"i": 0}

    def spy(h_in, *a, **kw):
        if cell is not None and n["i"] == 0:
            h_in = h_in.clone()
            h_in[:, cell] = h_in[:, cell] + mag
        n["i"] += 1
        return real(h_in, *a, **kw)

    m._apply_core_step = spy
    try:
        torch.manual_seed(seed)
        _xn, h, _d, *_ = _core_out(m, inp, layout)
    finally:
        m._apply_core_step = real
    return h


def _cell_deltas(m, inp, layout, cell, M=4):
    S = layout.slot_index.shape[1]
    base = core_exit(m, inp, layout)
    hit = core_exit(m, inp, layout, cell=cell)
    return (hit - base).abs().flatten(2).amax(-1)[0].view(S, M)      # [S, M]


def test_cell_zero_reads_a_later_cell_of_its_own_slot_in_the_forward():
    """THE CELL-LEVEL PROBE, and the reason the register is a register.

    Nudge cell 3 of slot k at the loop's input. Cell 0 of slot k must move — it is a
    LATER sibling reading an EARLIER-indexed one, which plain causal on the flattened
    axis forbids. Non-vacuous: under a `flat` build the same nudge moves cell 0 by
    EXACTLY zero. No cell of any earlier slot may move under either build.
    """
    M = 4
    _i, inp, _l, layout = _batch(M)
    k = int(layout.slot_valid[0].nonzero().flatten()[2])
    assert k >= 2, "fixture: need a slot with two earlier valid slots"

    with patch_relation("blk"):
        m = _model(M)
        with torch.no_grad():
            m.tul_register.W_o.weight.normal_(std=0.05)
        d = _cell_deltas(m, inp, layout, k * M + 3, M)
    with patch_relation("flat"):
        mc = _model(M)
        with torch.no_grad():
            mc.tul_register.W_o.weight.normal_(std=0.05)
        dc = _cell_deltas(mc, inp, layout, k * M + 3, M)

    for i in (0, 1, 2):
        assert float(d[k, i]) > 1e-4, (
            f"cell {i} of slot {k} did not move when cell 3 of its own slot did — the "
            f"in-slot forward read is not executing")
        assert float(dc[k, i]) == 0.0, (
            f"fixture: cell {i} moved under a plain-causal build too, so the probe is "
            f"not reading the relation")
    assert float(d[:k].max()) == 0.0, "an EARLIER slot moved — causality across slots"
    assert float(dc[:k].max()) == 0.0


def test_a_pad_cell_is_never_a_key_of_a_valid_cell():
    """The direction that matters for a pad. (The reverse is true and harmless and is
    NOT asserted: a tail pad cell sits after every valid cell on the compact axis, so it
    reads them — exactly as it did before the register's relation widened, and its own
    output goes to `prefix_project`'s dump row.)"""
    M = 4
    _i, inp, _l, layout = _batch(M)
    pad = ~layout.slot_valid[0]
    assert bool(pad.any()), "fixture: every slot is valid, the pad path is untested"
    with patch_relation("blk"):
        m = _model(M)
        with torch.no_grad():
            m.tul_register.W_o.weight.normal_(std=0.05)
        d = _cell_deltas(m, inp, layout, int(pad.nonzero()[0]) * M, M)
    assert float(d[~pad].max()) == 0.0, "a valid cell read a PAD cell"


# ── 4c. BOTH BRANCHES, not one ───────────────────────────────────────────────
#
# A layer is two attention branches. If only one of them honours the relation, every
# whole-model assertion above still passes — the other branch carries the in-slot read on
# its own. These grade each branch separately, and grade that the core stage hands the
# relation to both.

def test_the_window_branch_honours_a_forward_pointing_relation():
    """`_window_fallback` with a relation must let query 0 read a LATER key. Without one
    query 0's row is empty (causal + XSA excludes the self token) and SDPA returns 0, so
    the reading is unambiguous."""
    from morph.model.attention import _window_fallback
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 4) for _ in range(3))
    causal = _window_fallback(q, k, v, 16, q.device, 0.5)
    rel = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    wide = _window_fallback(q, k, v, 16, q.device, 0.5, relation=rel)
    assert float(causal[:, :, 0].abs().max()) == 0.0, (
        "fixture: query 0 already reads something without a relation")
    assert float(wide[:, :, 0].abs().max()) > 0.0, (
        "the window branch ANDs the relation into causality — it cannot widen")


def test_the_compressed_branch_honours_a_forward_pointing_relation():
    """`_tg_slot_attention`'s dense form (the compact cell axis) must do the same."""
    from morph.model.attention import _tg_slot_attention
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 4) for _ in range(3))
    sink = torch.zeros(2)
    causal = _tg_slot_attention(q, k, v, None, sink, 0.5)
    rel = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    wide = _tg_slot_attention(q, k, v, None, sink, 0.5, relation=rel)
    assert not torch.allclose(causal[:, :, 0], wide[:, :, 0]), (
        "the compressed branch ANDs the relation into causality — it cannot widen")


def _loss_with_branch_relation_dropped(branch: str, M: int = 4) -> float:
    """`loss_under("blk")` with ONE branch's `relation=` forced back to None."""
    import morph.model.attention as A
    win, comp = A._window_fallback, A._tg_slot_attention

    def drop(fn):
        def inner(*a, **kw):
            kw["relation"] = None
            return fn(*a, **kw)
        return inner

    if branch == "win":
        A._window_fallback = drop(win)
    else:
        A._tg_slot_attention = drop(comp)
    try:
        return loss_under("blk", M)
    finally:
        A._window_fallback, A._tg_slot_attention = win, comp


def test_each_branchs_relation_is_load_bearing_in_the_forward():
    """The model-level twin of the two function tests above. Force ONE branch's relation
    back to None and the register's loss must move. A branch that had quietly gone back
    to plain causal would give the SAME loss and every whole-model number would still
    look right."""
    base = loss_under("blk")
    for br, what in (("win", "window"), ("comp", "compressed")):
        assert base != _loss_with_branch_relation_dropped(br), (
            f"dropping the {what} branch's relation changed nothing — that branch is "
            f"already running plain causal")


def test_both_attention_branches_receive_the_relation_on_the_core_stage():
    """Wiring, not arithmetic: every core-stage attention call must hand the relation to
    BOTH halves of the layer. Handing it to one leaves the other running plain causal
    while every whole-model number still moves."""
    import morph.model.attention as A
    seen = {"win": [], "comp": []}
    win, comp = A._window_fallback, A._tg_slot_attention

    def spy_win(*a, **kw):
        seen["win"].append(kw.get("relation") is not None)
        return win(*a, **kw)

    def spy_comp(*a, **kw):
        seen["comp"].append(kw.get("relation") is not None)
        return comp(*a, **kw)

    m = _model(4)
    _i, inp, _l, layout = _batch(4)
    # The prelude runs OUTSIDE the spies on purpose: it carries no relation and would
    # dilute the reading. Only the core stage is graded.
    fkw, freset, _c, _r = m._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        A._window_fallback, A._tg_slot_attention = spy_win, spy_comp
        try:
            m._tul_core(x, x0, bg, layout, input_ids=inp)
        finally:
            A._window_fallback, A._tg_slot_attention = win, comp
    for br in ("win", "comp"):
        calls = seen[br]
        assert calls, f"fixture: the core stage made no {br} attention call"
        assert all(calls), (
            f"the {br} branch was handed no relation on {sum(1 for t in calls if not t)} "
            f"of {len(calls)} core-stage calls — it runs plain causal there")


# ── 5. NO NaN ────────────────────────────────────────────────────────────────

def test_a_pad_slot_gets_exactly_zero_and_no_nan():
    m = _model(4)
    with torch.no_grad():
        m.tul_register.W_o.weight.normal_(std=0.05)
    _i, inp, _l, layout = _batch(4)
    with torch.no_grad():
        fkw, freset, _c, _r = m._tul_tg_kwargs(layout)
        x, _x0, _bg = m._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        xn = m.input_norm(x)
        term = m.tul_register(xn.mean(dim=2) if m._is_hc else xn, layout)
    assert torch.isfinite(term).all()
    S, M = layout.slot_index.shape[1], 4
    t = term.view(term.shape[0], S, M, -1)
    pad = ~layout.slot_valid
    assert bool(pad.any()), "fixture: every slot is valid, the pad path is untested"
    assert float(t[pad].abs().max()) == 0.0


def test_every_register_parameter_receives_a_finite_gradient():
    """The pad slot's empty span used to make an all-`-inf` softmax row, whose NaN
    survives into the BACKWARD even with `nan_to_num` on the output: `Q` and `W_k` read
    `grad = nan` on this exact fixture before the mask was opened on those rows."""
    m = _model(4)
    with torch.no_grad():
        m.tul_register.W_o.weight.normal_(std=0.05)     # W_o zero => the pool is dead
    _i, inp, lab, layout = _batch(4)
    torch.manual_seed(3)
    m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    for n, p in m.tul_register.named_parameters():
        assert p.grad is not None, f"tul_register.{n} is not in the graph"
        assert torch.isfinite(p.grad).all(), f"tul_register.{n}.grad is not finite"
        assert float(p.grad.abs().sum()) > 0.0, f"tul_register.{n} receives no gradient"


# ── 6. THE STRICT GEOMETRY STILL HOLDS ───────────────────────────────────────

def _edit(ids, layout, row: int, span: int):
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


def test_the_strict_leak_test_still_holds_with_four_cells():
    m = _model(4).eval()
    spec = TulLayoutSpec(seq_len=64, prefix_k=4, max_slots=10, slot_id=4)
    ids = _ids()

    def run(i):
        inp, _l, lay, _s = slot_layout_from_ids(i, _rule(), spec)
        with torch.no_grad():
            return m.tul_forward_ablated(inp, None, lay, plan_mode="zero")["logits"], lay

    a, lay = run(ids)
    b, _ = run(_edit(ids, lay, 0, 0))
    bag, tok = lay.bag_id[0], ~lay.slot_mask[0]
    d = (a[0] - b[0]).abs().nan_to_num(0.0, posinf=0.0, neginf=0.0)
    assert float(d[tok & (bag == 0)].max()) > 0.0, "fixture: the edit moved nothing"
    assert float(d[tok & (bag != 0)].max()) == 0.0, (
        "with the loop's write zeroed a span-0 token moved a later span — four cells "
        "opened a route strict had cut")


@pytest.mark.parametrize("mode", ["normal", "zero", "shuffle", "all_slots"])
def test_the_worth_profile_modes_run_on_a_register_model(mode):
    m = _model(4).eval()
    _i, inp, lab, layout = _batch(4)
    with torch.no_grad():
        res = m.tul_forward_ablated(inp, lab, layout, plan_mode=mode)
    assert torch.isfinite(res["loss"])


def test_zero_and_all_slots_agree_under_strict_with_four_cells():
    m = _model(4).eval()
    _i, inp, lab, layout = _batch(4)
    with torch.no_grad():
        a = m.tul_forward_ablated(inp, lab, layout, plan_mode="zero")["loss"]
        b = m.tul_forward_ablated(inp, lab, layout, plan_mode="all_slots")["loss"]
    assert torch.equal(a, b)


def test_the_depth_lever_still_moves_a_register_model():
    m = _model(4).eval()
    _i, inp, lab, layout = _batch(4)
    seen = []
    for d in (1, 2, 4):
        m.cfg.tul.slot_mean_depth = d
        with torch.no_grad():
            seen.append(float(m.tul_forward_ablated(inp, lab, layout)["loss"]))
    assert len(set(seen)) == 3, f"the depth lever did nothing: {seen}"


# ── 7. THE INSTRUMENTS ───────────────────────────────────────────────────────

def test_the_probe_reports_the_within_slot_rank():
    m = _model(4).eval()
    _i, inp, _l, layout = _batch(4)
    with torch.no_grad():
        pr = m.tul_slot_state_probe(inp, layout)
    for k in ("slot_eff_rank", "slot_pairwise_cos", "slot_cell_eff_rank",
              "slot_cell_pairwise_cos", "slot_cells"):
        assert k in pr, f"{k} missing"
    assert pr["slot_cells"] == 4.0
    assert 1.0 <= pr["slot_cell_eff_rank"] <= 4.0


def test_the_probe_on_an_m1_model_reports_no_within_slot_keys():
    m = _model(1).eval()
    _i, inp, _l, layout = _batch(1)
    with torch.no_grad():
        pr = m.tul_slot_state_probe(inp, layout)
    assert "slot_cell_eff_rank" not in pr and "slot_cells" not in pr


# ── refusals ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    dict(tokens_through_core=True),
    dict(loop_reads_tokens=True),
    dict(db_loop=True),
    dict(reread=True),
    dict(slot_chain=True),
    dict(grad_pass=True),
    dict(oracle_z=True),
    dict(spandec_per_pass=True),
    dict(mux_every_pass=True, mux_beta=1.0),
    dict(mux_stage_own_iters=2, mux_beta=1.0),
    dict(bcast=True),
    dict(pass_lora_rank=8),
    dict(progressive_p=0.5),
    dict(core_token_aux=True),
    dict(coda_sees_slots=False),
    dict(prefix_source="trajectory"),
])
def test_every_one_state_per_slot_mechanism_raises(kw):
    with pytest.raises((NotImplementedError, ValueError)):
        _tul(slot_cells=4, prefix_k=4, **kw)


def test_slot_cell_init_without_a_register_raises():
    with pytest.raises(ValueError, match="slot_cells=1"):
        TULConfig(slot_cell_init="same")


def test_a_coreless_register_raises_at_build():
    with pytest.raises(ValueError, match="needs a core loop"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(slot_cells=4, prefix_k=4)))


def test_the_register_module_refuses_m_below_two():
    with pytest.raises(ValueError, match="m_cells >= 2"):
        TULSlotRegister(64, 1)


def test_the_shipped_configs_compose_and_carry_the_knobs():
    from hydra import compose, initialize_config_dir
    import os
    from morph.training.tul_setup import build_tul_runtime
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    want = {"tul_slot_register_m4": (4, "distinct"),
            "tul_slot_register_m4_sameinit": (4, "same"),
            "tul_slot_register_m8": (8, "distinct")}
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.join(root, "morph", "configs")):
        for name, (m_, init) in want.items():
            cfg = compose(config_name=name, overrides=[])
            rt = build_tul_runtime(cfg)
            assert rt.model_cfg.slot_cells == m_
            assert rt.model_cfg.prefix_k == m_
            assert rt.model_cfg.slot_cell_init == init
            assert rt.model_cfg.tg_geometry == "strict"
            assert rt.data_cfg.spec_for(cfg.data.seq_len).l_total == 1024 + m_ * 64
        # the ultralight-coda control, which is a config-only arm and needs no register
        cfg = compose(config_name="tul_slot_ultralight_coda", overrides=[])
        rt = build_tul_runtime(cfg)
        assert (cfg.model.n_prelude, cfg.model.n_core, cfg.model.n_coda) == (7, 6, 1)
        assert cfg.model.n_prelude + cfg.model.n_core + cfg.model.n_coda == 14
        assert rt.model_cfg.slot_cells == 1


def test_an_ultralight_coda_builds_and_runs_a_forward_and_backward():
    """A one-block coda is the arm's whole content, so it has to actually build: the TG
    kwargs, the per-layer injections and the slot-cell inject_keep all index the coda's
    blocks."""
    _i, inp, lab, layout = _batch(1)
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(n_prelude=3, n_core=2, n_coda=1,
                               tul=_tul())).train().float()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.parameters())
