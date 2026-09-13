"""``tul.cond_layers`` on the strict slot loop — arms ``slot-spandec-strict-cond4`` and
``slot-register-m4-cond4``.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_tul_cond4_strict.py -v

THE STACK. ``cond_layers`` NON-SHARED ``MORPHBlock``s run ONCE over the compact slot
sequence between the loop's exit and the readers of ``z``. Wolfe's 2026-09-03 sketch
(prelude -> core loop -> a few blocks -> z -> coda); its only two runs (R7f/R7d) detonated
under ``warmup: 0`` + absmean and it has never been measured.

WHAT THIS FILE HAS TO PROVE, and each contract is chosen so the test FAILS when the
shipped code breaks:

1. THE STACK IS THE READER'S INPUT. The cells ``prefix_project`` writes and the state the
   span decoder grades are the STACK's output, not the loop's raw exit. A bypassed stack
   is caught, two-sided.
2. IT RUNS OVER THE CELLS, NOT THE MEAN (the register arm). With ``slot_cells > 1`` the
   stack runs on the S*M compact CELL axis straight out of ``_tul_core``, under the SAME
   relation the register loops with — own slot full, earlier slots causal — so BOTH
   readers (the coda's 1:1 prefix write and the decoder's mean) read its output.
3. M == 1 IS THE OLD PLACEMENT, BIT-EXACT. Plain causal over S slots IS the relation at
   M = 1, and the register block is a no-op there, so moving the call above it changes
   nothing. Pinned against the pre-change source, extracted from git.
4. THE INTERACTIONS HOLD. Strict geometry's leak cut, the pad narrowing, the slot-loop
   gain constraint, and the forced-depth lever the K-curve instrument uses.
5. THE REFUSALS. ``loop_reach`` with a stack raises rather than silently giving the stack
   a WIDER relation than the loop's own pass.

NOT DUPLICATED HERE. ``cond_layers: 0`` builds no stack, draws no RNG and leaves every
shared weight byte-identical: ``tests/test_tul_think_once.py::
test_cond_stack_is_built_last_and_shared_weights_are_byte_identical`` (and the block-count
parity and ``detach_z`` contracts beside it). This file starts where that one stops.

CPU only, fp32, ``use_kernels=False``, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-cond4-reader.md
Note: .agents/notes/proposed/architecture/2026-09-13-cond4-as-the-reader-of-the-register.md
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch

# tests/ is on sys.path; the register file owns the tiny strict fixture this arm runs on.
from test_tul_slot_register import (  # noqa: E402
    DOT, V, _batch, _edit, _ids, _model, _relation_of, _rule, _tiny, _tul,
    patch_relation,
)

from morph.model.tul_layout import TulLayoutSpec, slot_layout_from_ids  # noqa: E402
from morph.model.transformer import MORPHTransformer                    # noqa: E402

N_COND = 2


def _cond(M: int = 1, n_cond: int = N_COND, seed: int = 99, **tul_kw) -> MORPHTransformer:
    """The fixture model: the strict slot loop with a conditioning stack on it."""
    m = _model(M, seed=seed, cond_layers=n_cond, **tul_kw)
    # A zero-init stack would be an identity and every contract below would pass on a
    # model that computes nothing. Nudge it so the stack really moves the state.
    torch.manual_seed(4)
    with torch.no_grad():
        for p in m.tul_cond.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


class _Seam:
    """What each seam of the forward saw, captured by spying on the real methods."""

    def __init__(self, m: MORPHTransformer):
        self.m = m
        self.stack_in = self.stack_out = None
        self.prefix_h = self.prefix_cells = None
        self.spandec_h = None

    def __enter__(self):
        m = self.m
        self._cond, self._proj, self._sd = (
            m._tul_cond_apply, m.tul.prefix_project, m._tul_spandec_loss)

        def cond(h, *a, **kw):
            self.stack_in = h.detach().clone()
            out = self._cond(h, *a, **kw)
            self.stack_out = out.detach().clone()
            return out

        def proj(h, *a, **kw):
            self.prefix_h = h.detach().clone()
            c = kw.get("cells")
            self.prefix_cells = None if c is None else c.detach().clone()
            return self._proj(h, *a, **kw)

        def sd(h, *a, **kw):
            self.spandec_h = h.detach().clone()
            return self._sd(h, *a, **kw)

        m._tul_cond_apply, m.tul.prefix_project, m._tul_spandec_loss = cond, proj, sd
        return self

    def __exit__(self, *exc):
        self.m._tul_cond_apply = self._cond
        self.m.tul.prefix_project = self._proj
        self.m._tul_spandec_loss = self._sd
        return False


def _run(m: MORPHTransformer, M: int = 1, seed: int = 5) -> _Seam:
    _i, inp, lab, layout = _batch(M)
    with _Seam(m) as seam:
        torch.manual_seed(seed)
        m(inp, labels=lab, slot_layout=layout)["loss"]
    return seam


# ── 1. THE STACK IS THE READER'S INPUT ───────────────────────────────────────

def test_the_coda_reads_the_stack_output_not_the_loop_exit():
    m = _cond()
    s = _run(m)
    assert s.stack_in is not None, "the stack never ran"
    assert torch.equal(s.prefix_h, s.stack_out), (
        "prefix_project was handed something other than the stack's output")
    assert not torch.allclose(s.prefix_h, s.stack_in), (
        "the stack is an identity on this fixture — the contract is untested")


def test_the_span_decoder_grades_the_stack_output():
    m = _cond()
    s = _run(m)
    assert s.spandec_h is not None, "the span decoder never ran"
    assert torch.equal(s.spandec_h, s.stack_out), (
        "the span decoder graded the loop's raw exit, not the stack's output")


def test_sabotage_a_bypassed_stack_is_caught_at_both_readers():
    """The sabotage: ``_tul_cond_apply`` returns its input. Both contracts above must
    then be violated, which is what makes them load-bearing rather than decorative."""
    m = _cond()
    m._tul_cond_apply = lambda h, *a, **kw: h          # the bypass
    s = _run(m)
    assert torch.equal(s.prefix_h, s.stack_in), "fixture: the bypass did not bypass"
    # the two assertions of the two tests above, inverted
    assert torch.allclose(s.prefix_h, s.stack_in)
    assert torch.equal(s.spandec_h, s.stack_in)


def test_the_stack_moves_the_loss_and_the_layer_pass_count():
    m0, m1 = _model(1, seed=99), _cond(1)
    _i, inp, lab, layout = _batch(1)
    torch.manual_seed(5)
    o0 = m0(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    o1 = m1(inp, labels=lab, slot_layout=layout)
    assert not torch.allclose(o0["loss"], o1["loss"])
    n_valid = int(layout.slot_valid.sum())
    assert n_valid > 0
    assert float(o1["layer_passes"] - o0["layer_passes"]) == pytest.approx(N_COND * n_valid)


# ── 4. THE INTERACTIONS ──────────────────────────────────────────────────────

def test_the_strict_leak_cut_still_holds_with_a_stack():
    """With the loop's write zeroed, a span-0 token must move NOTHING outside span 0.
    The stack sits INSIDE that cut: it reads slot states and writes slot states, so a
    stack that leaked would show up here."""
    m = _cond().eval()
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
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
        "the conditioning stack opened a cross-span route strict had cut")


def test_zero_and_all_slots_still_agree_under_strict_with_a_stack():
    """On a strict arm `worth_profile`'s `all_slots` equals its `zero` by construction —
    the check the profile's reading rests on."""
    m = _cond().eval()
    _i, inp, lab, layout = _batch(1)
    with torch.no_grad():
        a = m.tul_forward_ablated(inp, lab, layout, plan_mode="zero")["loss"]
        b = m.tul_forward_ablated(inp, lab, layout, plan_mode="all_slots")["loss"]
    assert torch.equal(a, b)


def test_a_pad_slot_still_writes_to_the_dump_row_with_a_stack():
    """The pad narrowing: an invalid slot's prefix positions address `l_total`, the dump
    row, whatever the stack computed at those positions."""
    m = _cond().eval()
    _i, inp, lab, layout = _batch(1)
    s = _run(m)
    L = int(layout.slot_mask.shape[1])
    _v, pos = m.tul.prefix_project(s.stack_out, layout, L)
    S, K = layout.slot_index.shape[1], m.cfg.tul.prefix_k
    pos = pos.reshape(-1, S, K)
    pad = ~layout.slot_valid
    assert bool(pad.any()), "fixture: every slot is valid, the pad path is untested"
    assert bool((pos[pad] == L).all()), "a pad slot's prefix write escaped the dump row"


def test_the_slot_gain_constraint_still_acts_and_prints_no_inert_notice():
    """`model.slot_gain_lambda` acts only inside `_tul_core`. `cond_layers` does not move
    the core stage, so the constraint must still read a gain and the build must NOT print
    the INERT notice."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        torch.manual_seed(99)
        m = MORPHTransformer(_tiny(tul=_tul(prefix_k=2, cond_layers=N_COND),
                                   slot_gain_lambda=100.0, slot_gain_target=0.9,
                                   slot_cot_clip=4.0)).train().float()
    assert "INERT" not in buf.getvalue(), buf.getvalue()
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    _i, inp, lab, layout = _batch(1)
    torch.manual_seed(5)
    out = m(inp, labels=lab, slot_layout=layout)
    assert "gain_est" in out and float(out["gain_n_iters"]) > 0.0, sorted(out)
    assert torch.isfinite(out["gain_est"])


def test_the_forced_depth_lever_reaches_the_stack_and_the_coda():
    """`lab/divergence/core_depth_sweep.py` forces a slot arm's depth with
    `cfg.tul.slot_mean_depth`. A forced depth changes the loop's exit; the stack then runs
    on THAT state and the coda reads the stack. So every depth must move all three, or the
    K-curve on this arm would be measuring a state nobody reads."""
    m = _cond().eval()
    seen: list[tuple[float, float, float]] = []
    for d in (1, 2, 4):
        m.cfg.tul.slot_mean_depth = d
        m.cfg.tul.slot_max_depth = max(d, 4)
        s = _run(m)
        seen.append((float(s.stack_in.float().sum()), float(s.stack_out.float().sum()),
                     float(s.prefix_h.float().sum())))
    for col, what in enumerate(("the loop exit", "the stack output", "the coda's read")):
        vals = [round(r[col], 4) for r in seen]
        assert len(set(vals)) == 3, f"the depth lever did not move {what}: {vals}"


def test_the_stack_and_the_loop_run_the_same_cross_slot_relation_at_m1():
    """Interaction (e). At `slot_cells: 1` with `loop_reach: 0` the core is handed NO
    attention kwargs on the compact sequence (`_core_akw is None`) and `_tul_cond_apply`
    passes none either, so BOTH run the block's own causal attention over the same S
    positions: the stack's relation is exactly a pass of the loop's, never wider. Strict
    geometry does not narrow it — `_tul_core` is never handed `tg_attn_kwargs`."""
    from test_tul_slot_register import _core_kwargs
    m = _cond()
    _i, inp, _l, layout = _batch(1)
    assert _core_kwargs(m, inp, layout) is None, (
        "the loop now carries a relation the stack does not; give the stack the same one")
    seen: list[tuple] = []
    real = m.tul_cond[0].forward

    def spy(h, *a, **kw):
        seen.append((tuple(a), dict(kw)))
        return real(h, *a, **kw)

    m.tul_cond[0].forward = spy
    _run(m)
    assert seen and seen[0] == ((), {"attn_kwargs": None}), (
        f"the stack was handed {seen[0]}")


# ── 5. THE REFUSALS ──────────────────────────────────────────────────────────

def test_cond_layers_with_a_reach_budget_raises():
    """`loop_reach` spends the whole per-pass cross-cell budget in core layer 0. A stack
    running plain causal (or own-slot-full) afterwards would carry cells the reach budget
    cut, silently re-opening the route the arm exists to close."""
    with pytest.raises(NotImplementedError, match="cond_layers"):
        _model(1, cond_layers=1, loop_reach=1)


# ── 2/3. THE REGISTER ARM: THE STACK RUNS ON THE CELLS ───────────────────────

# The pins for contract 3, produced by running the PRE-change source (commit b1075f2,
# extracted into a symlink tree whose only real file is that transformer.py) and this tree
# in two processes on the same fixture and the same seeds. All four columns matched to the
# last printed digit, which is what says the reorder is a no-op at `slot_cells: 1`:
#
#   cond=2 loss=9.9817762375 logit_sum=553.297651 n_fin=10584 passes=950.0 gradsum=1812.370773
#   cond=4 loss=9.9952659607 logit_sum=414.534669 n_fin=10584 passes=982.0 gradsum=1916.135578
M1_PIN = {
    2: dict(loss=9.9817762375, logit_sum=553.297651, n_fin=10584, passes=950.0,
            gradsum=1812.370773),
    4: dict(loss=9.9952659607, logit_sum=414.534669, n_fin=10584, passes=982.0,
            gradsum=1916.135578),
}


@pytest.mark.parametrize("n_cond", [2, 4])
def test_m1_placement_is_bit_identical_to_the_pre_change_source(n_cond):
    """Contract 3. Moving `_tul_cond_apply` above the register's mean cannot change a
    `slot_cells: 1` model: the register block is a no-op there and plain causal over S
    slots IS `slot_cell_relation` at M = 1. Pinned, not argued."""
    m = _cond(1, n_cond=n_cond)
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
    pin = M1_PIN[n_cond]
    assert float(out["loss"]) == pytest.approx(pin["loss"], abs=1e-9)
    assert float(out["layer_passes"]) == pin["passes"]
    assert gs == pytest.approx(pin["gradsum"], abs=1e-4)
    assert int(fin.sum()) == pin["n_fin"]
    assert float(lg[fin].double().sum()) == pytest.approx(pin["logit_sum"], abs=1e-4)


def test_the_stack_runs_on_the_cell_axis_not_on_the_mean():
    m = _cond(4)
    s = _run(m, 4)
    B, S = m.cfg.tul.prefix_k, None
    _i, inp, lab, layout = _batch(4)
    S = layout.slot_index.shape[1]
    assert s.stack_in.shape[1] == S * 4, (
        f"the stack read {s.stack_in.shape[1]} positions, not the S*M={S * 4} cell axis")
    assert s.stack_out.shape == s.stack_in.shape


def test_the_coda_prefix_cells_are_the_stacks_output_cell_by_cell():
    """The register writes cell i into prefix cell i. With the stack on the cell axis
    those cells ARE the stack's output — not the raw loop exit."""
    m = _cond(4)
    s = _run(m, 4)
    assert s.prefix_cells is not None, "the register did not hand prefix_project cells"
    B = s.stack_out.shape[0]
    S = s.stack_out.shape[1] // 4
    want = s.stack_out.reshape(B, S, 4, *s.stack_out.shape[2:])
    assert torch.equal(s.prefix_cells, want), (
        "the coda's prefix cells are not the stack's output cells")
    raw = s.stack_in.reshape(B, S, 4, *s.stack_in.shape[2:])
    assert not torch.allclose(s.prefix_cells, raw), (
        "the stack is an identity on this fixture — the contract is untested")


def test_the_span_decoder_grades_the_mean_of_the_stacks_output_cells():
    m = _cond(4)
    s = _run(m, 4)
    B = s.stack_out.shape[0]
    S = s.stack_out.shape[1] // 4
    want = s.stack_out.reshape(B, S, 4, *s.stack_out.shape[2:]).mean(dim=2)
    assert torch.equal(s.spandec_h, want), (
        "the span decoder graded the mean of the RAW loop cells, not the stack's")


def test_the_layer_pass_count_is_per_cell_on_a_register_model():
    m0, m1 = _model(4, seed=99), _cond(4)
    _i, inp, lab, layout = _batch(4)
    torch.manual_seed(5)
    o0 = m0(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    o1 = m1(inp, labels=lab, slot_layout=layout)
    n_valid = int(layout.slot_valid.sum())
    assert float(o1["layer_passes"] - o0["layer_passes"]) == pytest.approx(
        N_COND * 4 * n_valid), "the stack's passes are counted per SLOT, not per CELL"


# ── the stack's relation, at M > 1 ───────────────────────────────────────────

def _stack_allow(m: MORPHTransformer, M: int) -> torch.Tensor:
    """The relation the stack was actually handed, captured off the real forward.

    It must arrive as `tg_relation` — the one attention kwarg that REPLACES a branch's
    causal term — exactly as the loop's own core stage delivers it. Through
    `tg_allow`/`tg_comp_allow` this superset-of-causal mask executed as plain flattened
    causal (measured and fixed 2026-09-13).
    """
    seen: list = []
    real = m.tul_cond[0].forward

    def spy(h, attn_kwargs=None, **kw):
        seen.append(attn_kwargs)
        return real(h, attn_kwargs=attn_kwargs, **kw)

    m.tul_cond[0].forward = spy
    try:
        _run(m, M)
    finally:
        m.tul_cond[0].forward = real
    assert seen and seen[0] is not None, "the stack was handed no relation at all"
    return _relation_of(seen[0])[0, 0]


def test_the_stack_uses_the_same_relation_builder_as_the_loop():
    """ONE builder. If the stack's mask and the core's mask ever stop being the same
    tensor content, the stack has quietly become a different mechanism."""
    from test_tul_slot_register import _core_kwargs
    m = _cond(4)
    _i, inp, _l, layout = _batch(4)
    core = _core_kwargs(m, inp, layout)
    assert core is not None
    assert torch.equal(_stack_allow(m, 4), _relation_of(core[0])[0, 0])


def test_the_stacks_mask_says_own_slot_full_and_earlier_slots_causal():
    """THE MASK. Cell i of slot k is allowed every cell of its own slot (including the
    ones after it) and every cell of slots < k, and no cell of a later slot.

    THIS GRADES THE MASK. `test_the_stacks_executed_relation_is_not_flattened_causal`
    below grades the FORWARD, two-sided, which is the assertion that would have caught
    the 2026-09-13 defect: the mask was right and the delivery (`tg_allow`, which only
    narrows) threw its forward-in-slot half away."""
    m = _cond(4)
    a = _stack_allow(m, 4)
    S = a.shape[0] // 4
    sl = torch.arange(S * 4) // 4
    for p in (0, 1, 4 + 2, 3 * 4 + 1, S * 4 - 1):
        assert bool(a[p][sl == sl[p]].all()), f"cell {p} cannot read all of its own slot"
        assert bool(a[p][sl < sl[p]].all()), f"cell {p} cannot read an earlier slot"
        assert not bool(a[p][sl > sl[p]].any()), f"cell {p} reads a LATER slot"
    flat = torch.arange(a.shape[0]).unsqueeze(0) <= torch.arange(a.shape[0]).unsqueeze(1)
    assert not torch.equal(a, flat), "the stack's MASK is flattened causal"


def _stack_out(m: MORPHTransformer, M: int, edit=None, perturb=None) -> torch.Tensor:
    """Run the forward with the stack's relation (and/or its input) edited."""
    from morph.model.transformer import slot_cell_relation
    real = m._tul_cond_apply
    box: dict = {}

    def patched(h, n_slots=0, m_cells=1):
        allow, _same = slot_cell_relation(n_slots, m_cells, h.device)
        if edit is not None:
            allow = edit(allow.clone())
        if perturb is not None:
            h = perturb(h.clone())
        akw = {"tg_relation": allow}
        for layer in m.tul_cond:
            h = layer(h, attn_kwargs=akw)
        box["out"] = h.detach().clone()
        return h

    m._tul_cond_apply = patched
    try:
        _run(m, M)
    finally:
        m._tul_cond_apply = real
    return box["out"]


def _nudge(h: torch.Tensor, cell: int, mag: float = 1.0) -> torch.Tensor:
    h[:, cell] = h[:, cell] + mag
    return h


def _cond_loss(mode: str, M: int = 4) -> float:
    """One forward's loss on a cond-stack register model, with the relation set to
    `mode` (see `patch_relation`: blk / flat / alltrue).

    No `loop_reach` variant: a reach budget with a stack RAISES (`TULConfig`), so the
    reach fixture check for this pair lives on the loop, in
    `tests/test_tul_slot_register.py::test_a_reach_budget_still_narrows_the_executed_relation`.
    The non-vacuity fixture HERE is `test_the_stacks_mask_is_load_bearing`.
    """
    with patch_relation(mode):
        m = _cond(M)
        with torch.no_grad():
            # W_o is zero at init, so every cell carries the same seed and no relation
            # can be told from another. Move it first.
            m.tul_register.W_o.weight.normal_(std=0.05)
        _i, inp, lab, layout = _batch(M)
        torch.manual_seed(5)
        return float(m(inp, labels=lab, slot_layout=layout)["loss"].detach())


def test_the_stacks_mask_is_load_bearing():
    """Non-vacuity. Blank ONE entry the causal base already allows — `slot 1 cell 3 reads
    slot 1 cell 0` — and the stack's output must move. Without this the whole relation
    could be ignored and every assertion above would still pass."""
    m = _cond(4)
    base = _stack_out(m, 4)
    ref = _run(m, 4).stack_out
    assert torch.equal(base, ref), "fixture: the patched runner is not the shipped path"

    def blank(a):
        a[0, 0, 4 + 3, 4 + 0] = False        # slot 1, cell 3 <- slot 1, cell 0
        return a

    assert not torch.equal(_stack_out(m, 4, edit=blank), base), (
        "closing a within-slot backward read changed nothing — the stack ignores its mask")


def test_the_stacks_executed_relation_is_not_flattened_causal():
    """THE FORWARD, two-sided, on the STACK (the loop's own twin lives in
    `tests/test_tul_slot_register.py`).

    `blk[p][q] = slot(p) >= slot(q)` is a SUPERSET of flattened causal `p >= q`. Handed to
    `tg_allow`/`tg_comp_allow` it was ANDed back into causality and the stack executed
    plain flattened causal — a cell never read a LATER cell of its own slot. It now
    travels as `tg_relation`, which REPLACES the causal term. So the stack's loss must
    DIFFER from a flattened-causal build and EQUAL an independently written
    all-true-within-slot build."""
    blk = _cond_loss("blk")
    assert blk != _cond_loss("flat"), (
        "the stack gives the SAME loss as plain flattened causal — its in-slot forward "
        "read is not executing")
    assert blk == _cond_loss("alltrue"), (
        "the stack's executed relation is neither the documented one nor plain causal")


def test_a_stack_cell_reads_a_later_cell_of_its_own_slot():
    """The cell-level probe on the STACK. Nudge the loop exit at cell 3 of slot k; cell 0
    of slot k's STACK output must move, and must NOT move under a `flat` build. No cell
    of an earlier slot may move under either."""
    M, k_idx = 4, 2
    _i, _inp, _l, layout = _batch(M)
    S = layout.slot_index.shape[1]
    k = int(layout.slot_valid[0].nonzero().flatten()[k_idx])

    def deltas(mode: str) -> torch.Tensor:
        with patch_relation(mode):
            m = _cond(M)
            base = _stack_out(m, M)
            hit = _stack_out(m, M, perturb=lambda h: _nudge(h, k * M + 3))
        return (hit - base).abs().flatten(2).amax(-1)[0].view(S, M)

    d, dc = deltas("blk"), deltas("flat")
    for i in (0, 1, 2):
        assert float(d[k, i]) > 1e-6, (
            f"stack cell {i} of slot {k} did not move when cell 3 of its own slot did")
        assert float(dc[k, i]) == 0.0, (
            f"fixture: stack cell {i} moved under a plain-causal build too")
    assert float(d[:k].max()) == 0.0, "an EARLIER slot's stack output moved"
    assert float(dc[:k].max()) == 0.0


def test_the_stack_is_causal_across_slots():
    """Perturb the loop exit at EVERY cell of the last valid slot. No earlier slot's
    stack output may move."""
    m = _cond(4)
    _i, inp, lab, layout = _batch(4)
    S = layout.slot_index.shape[1]
    last = int(layout.slot_valid[0].nonzero()[-1])
    assert last >= 2, "fixture: too few valid slots to test causality"
    base = _stack_out(m, 4)

    def hit(h):
        h[:, last * 4:(last + 1) * 4] += 1.0
        return h

    got = _stack_out(m, 4, perturb=hit)
    earlier = slice(0, last * 4)
    assert torch.equal(got[:, earlier], base[:, earlier]), (
        "a later slot's cells moved an earlier slot's stack output")
    assert not torch.equal(got[:, last * 4:], base[:, last * 4:]), (
        "fixture: the perturbation did nothing at all")


def test_pad_cells_never_reach_a_valid_cell_and_their_write_goes_to_the_dump_row():
    """Pad slots sit at the TAIL of the compact axis, so the causal relation already
    excludes them from every valid cell's key set; and `prefix_project` sends an invalid
    slot's K positions to the dump row whatever the stack computed there."""
    m = _cond(4).eval()
    _i, inp, lab, layout = _batch(4)
    S = layout.slot_index.shape[1]
    a = _stack_allow(m, 4)
    sl = torch.arange(S * 4) // 4
    pad_slot = ~layout.slot_valid[0]
    assert bool(pad_slot.any()), "fixture: every slot is valid, the pad path is untested"
    pad_cell = pad_slot[sl]
    valid_cell = ~pad_cell
    assert not bool(a[valid_cell][:, pad_cell].any()), (
        "a valid cell may read a pad cell through the stack")
    s = _run(m, 4)
    L = int(layout.slot_mask.shape[1])
    _v, pos = m.tul.prefix_project(s.spandec_h, layout, L, cells=s.prefix_cells)
    pos = pos.reshape(-1, S, 4)
    assert bool((pos[~layout.slot_valid] == L).all()), (
        "a pad slot's prefix write escaped the dump row")


# ── the instruments ──────────────────────────────────────────────────────────

def test_the_rank_probe_reads_the_stacks_output():
    """`val/slot_eff_rank` and `val/slot_cell_eff_rank` are this arm's headline numbers.
    The probe's contract is "the written slot states, read at the point the coda reads
    them", so on a `cond_layers` model it has to run the stack — otherwise the instrument
    is blind to the mechanism the arm exists to test."""
    m = _cond(4).eval()
    _i, inp, _l, layout = _batch(4)
    with torch.no_grad():
        pr = m.tul_slot_state_probe(inp, layout)
    for k in ("slot_eff_rank", "slot_pairwise_cos", "slot_cell_eff_rank",
              "slot_cell_pairwise_cos", "slot_cells"):
        assert k in pr, f"{k} missing"
    assert pr["slot_cells"] == 4.0
    real = m._tul_cond_apply
    m._tul_cond_apply = lambda h, *a, **kw: h            # SABOTAGE: probe skips the stack
    try:
        with torch.no_grad():
            bypass = m.tul_slot_state_probe(inp, layout)
    finally:
        m._tul_cond_apply = real
    assert pr["slot_eff_rank"] != bypass["slot_eff_rank"], (
        "the rank probe reads the loop exit, not the stack's output")
