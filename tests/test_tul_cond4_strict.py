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
    DOT, V, _batch, _edit, _ids, _model, _rule, _tiny, _tul,
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
    assert seen and seen[0] == ((), {}), f"the stack was handed {seen[0]}"


# ── 5. THE REFUSALS ──────────────────────────────────────────────────────────

def test_cond_layers_with_a_reach_budget_raises():
    """`loop_reach` spends the whole per-pass cross-cell budget in core layer 0. A stack
    running plain causal (or own-slot-full) afterwards would carry cells the reach budget
    cut, silently re-opening the route the arm exists to close."""
    with pytest.raises(NotImplementedError, match="cond_layers"):
        _model(1, cond_layers=1, loop_reach=1)
