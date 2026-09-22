"""``tul.loop_carry`` — the slot loop KEEPS what it read from its neighbours.

WHAT THIS FILE HAS TO PROVE, in the order it matters:

1. OFF IS NOTHING. ``loop_carry: "none"`` is bit-identical to the tree before the key
   existed. Pinned against the numbers ``tests/test_tul_prefix_source.py`` already carries
   for the strict ruler at ``prefix_k 2`` — the same fixture, the same seed, measured
   before this change — and the parameter NAMES are compared set-to-set against a config
   built with no carry key at all.
2. THE STATE IS WHAT THE DOCSTRING SAYS. At a forced depth ``T``, ``sum``'s carry at cell
   ``k`` after pass ``T`` is EXACTLY the sum of the ``T`` reads the loop captured, and
   ``gate``'s at ``W_g = 0`` is exactly half of it. The reads come off the model's own
   ``_carry_capture`` hook, so the test reads the tensors the forward used and not a
   re-derivation of them.
3. THE CARRY DOES NOT WIDEN THE REACH. Under ``loop_reach 1`` with the coda reading cell
   ``j-1`` only, an edit two spans back moves span ``j`` at depth 1 (it always did), an
   edit THREE spans back still does NOT at depth 1, and DOES at depth 3. A mechanism that
   let a cell read further per pass would break the middle clause; one that carried
   nothing would break the last.
4. THE NORMALISATION IS LIVE. The injected term's per-cell RMS equals the carrier's, at
   every pass, on every cell that receives one — read off the model, not recomputed from
   the rule.
5. THE FOUR REFUSALS RAISE, each with its own message.
6. ``"persist"`` (LXTUL-R Step 2, 2026-09-22) ACCUMULATES LIKE ``sum`` BUT NEVER
   RE-INJECTS. Its per-pass carrier — ``_apply_core_step``'s own return, upstream of any
   exit add — is bit-identical to ``"none"`` at every pass, which ``sum`` is NOT (it
   diverges from pass 1 on); the exit state ``_tul_core`` returns is exactly
   ``h_none + carry_rms_match(sum of the T reads, h_none)``; ``carry/persist_ratio`` is
   reported only on this mode; and it is NOT refused on a Thought Register
   (``tul.slot_cells > 1`` / ``tul.fan_k > 0``), where ``sum``/``gate`` stay refused.

CPU only, fp32, ``use_kernels=False``, tiny config, the
``tests/test_tul_strict_geometry.py`` fixtures. fp32 and not fp64: ``_window_fallback``
hands SDPA an fp32 mask whatever the dtype of q, which is silently WRONG at fp64
(morph/model/CLAUDE.md).

NAMED HERE because it is the opposite of this tree's habit and a reader will look for it:
``loop_carry`` is NOT a no-op at initialisation. The RMS match cancels any constant scale
in front of the carry, so ``gate`` at ``W_g = 0`` injects exactly what ``sum`` injects at
pass 1 and the two modes give the SAME loss on an untrained model
(``test_sum_and_gate_agree_at_init_because_the_rms_match_cancels_the_half``). The
bit-identity claim of this key is ``"none"``.

Record: lab/experiments/planned/2026-09-19-loop-carry-prev-reach1.md
Note: .agents/notes/proposed/architecture/2026-09-19-loop-carry-reinjection.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_carry import TULLoopCarry, carry_rms_match
from morph.model.tul_layout import (BoundaryRule, SlotLayout, TulLayoutSpec,
                                    slot_layout_from_ids)
from morph.training.tul_setup import reject_unknown_tul_keys

V = 64
DOT = 10


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


def _edit(ids: np.ndarray, layout: SlotLayout, row: int, span: int) -> np.ndarray:
    """Change one NON-boundary token of ``span`` in ``row`` without moving the cut.

    Copied in behaviour from `tests/test_tul_strict_geometry.py::_edit`; the two files
    share the fixture and the perturbation has to be the same one.
    """
    bag = layout.bag_id[row].numpy()
    tok = (~layout.slot_mask[row].numpy())
    pos = np.flatnonzero((bag == span) & tok)
    assert pos.size >= 3, f"span {span} of row {row} is too short to edit safely"
    p = int(pos[len(pos) // 2])
    out = ids.copy()
    raw = int(tok[:p].sum())
    assert out[row, raw] != DOT and out[row, raw] != 11
    out[row, raw] = 12 if out[row, raw] != 12 else 13
    return out


def _logits(m: MORPHTransformer, ids: np.ndarray):
    inp, _lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec())
    with torch.no_grad():
        out = m(inp, labels=None, slot_layout=layout)
    return out["logits"], layout


def _reads(m: MORPHTransformer, inp, layout):
    """Run one forward with the capture hook attached; return the per-pass records."""
    m._carry_capture = []
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
        return list(m._carry_capture)
    finally:
        m._carry_capture = None


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

# The strict ruler at prefix_k 2, pinned in tests/test_tul_prefix_source.py
# (`test_exit_is_pinned_to_the_pre_knob_values`, case "strict_k2") against a worktree of
# `d778845` on 2026-09-13 — i.e. measured on the tree BEFORE `tul.loop_carry` existed.
HEAD_K2_LOSS = 5.044249534606934
HEAD_K2_LOGIT_SUM = 842.074198674527
HEAD_K2_GRAD_SUM = 968.7720451547807
HEAD_K2_KEYS = 211


def test_none_is_the_default_and_builds_nothing():
    m = _model()
    assert m.cfg.tul.loop_carry == "none"
    assert m.tul_carry is None
    assert not any("tul_carry" in k for k in m.state_dict())


def test_none_is_bit_identical_to_the_pre_key_tree():
    """Loss, logits, total gradient and the state-dict size, to the last printed digit."""
    _ids0, inp, lab, layout = _batch()
    m = _model(loop_carry="none").train()
    torch.manual_seed(7)
    res = m(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    assert res["loss"].item() == HEAD_K2_LOSS
    assert _finite_logit_sum(lg) == HEAD_K2_LOGIT_SUM
    assert g == HEAD_K2_GRAD_SUM
    assert len(m.state_dict()) == HEAD_K2_KEYS


def test_none_has_the_parameter_names_of_a_config_with_no_carry_key():
    a = set(_model().state_dict())
    b = set(_model(loop_carry="none").state_dict())
    assert a == b


def test_sum_builds_no_parameter_and_gate_builds_exactly_one():
    a = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum")
    b = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="gate")
    assert a.tul_carry is not None and a.tul_carry.W_g is None
    assert not any(k.startswith("tul_carry.") for k in a.state_dict())
    assert b.tul_carry is not None and b.tul_carry.W_g is not None
    assert [k for k in b.state_dict() if k.startswith("tul_carry.")] == ["tul_carry.W_g"]
    assert torch.equal(b.tul_carry.W_g, torch.zeros(64, 128)), "W_g must be zero-init"
    # RNG-neutral: the gate draws nothing, so every OTHER weight is the ruler's.
    r = _model(tg_coda_prefix_reach="prev", loop_reach=1)
    for k, v in r.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), f"{k} moved — the gate drew RNG"


def test_the_carry_arm_changes_the_forward():
    """The other side of clause 1: `sum` must not be inert."""
    _ids0, inp, lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1)
    a = _model(**kw)
    b = _model(**kw, loop_carry="sum")
    with torch.no_grad():
        la = a(inp, labels=lab, slot_layout=layout)["loss"]
        lb = b(inp, labels=lab, slot_layout=layout)["loss"]
    assert not torch.equal(la, lb), "loop_carry='sum' changed nothing — the knob is inert"


# ── 2. THE STATE IS WHAT THE DOCSTRING SAYS ──────────────────────────────────

@pytest.mark.parametrize("depth", [2, 3])
def test_sum_carry_is_the_sum_of_its_reads_at_forced_depth(depth):
    """c_k(T) == sum over t of r_k(t), read off the model's own capture hook."""
    _ids0, inp, _lab, layout = _batch()
    m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum",
               slot_depth_fixed=depth, slot_max_depth=8)
    recs = _reads(m, inp, layout)
    assert len(recs) == depth, f"forced depth {depth} ran {len(recs)} passes"
    want = torch.zeros_like(recs[0]["read"])
    for r in recs:
        want = want + r["read"]
    got = recs[-1]["carry"]
    v = layout.slot_valid.unsqueeze(-1)
    assert torch.allclose(got * v, want * v, atol=1e-6), \
        f"depth {depth}: the carry is not the sum of its reads"
    assert float((want * v).abs().max()) > 0, "the reads are all zero — the test is vacuous"
    assert all(r["gate"] is None for r in recs), "'sum' must build no gate"


@pytest.mark.parametrize("depth", [2, 3])
def test_gate_carry_at_zero_weight_is_half_the_sum(depth):
    """W_g = 0 ⇒ sigmoid = 1/2 at every channel ⇒ c = (1/2) Σ r."""
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1, slot_depth_fixed=depth,
              slot_max_depth=8)
    _ids0, inp, _lab, layout = _batch()
    a = _model(**kw, loop_carry="sum")
    b = _model(**kw, loop_carry="gate")
    ra, rb = _reads(a, inp, layout), _reads(b, inp, layout)
    assert len(ra) == len(rb) == depth
    v = layout.slot_valid.unsqueeze(-1)
    ca, cb = ra[-1]["carry"] * v, rb[-1]["carry"] * v
    assert float(ca.abs().max()) > 0
    assert torch.allclose(cb, 0.5 * ca, atol=1e-5), \
        "the zero-init gate did not open at exactly 1/2"
    # The two runs must have taken the SAME reads, or the halving above would be a
    # coincidence of two different trajectories: the RMS match cancels the 1/2, so the
    # injected term is identical and so is every later read.
    for t, (x, y) in enumerate(zip(ra, rb)):
        assert torch.equal(x["read"], y["read"]), f"pass {t}: the reads diverged"
        assert float((y["gate"] - 0.5).abs().max()) == 0.0


def test_sum_and_gate_agree_at_init_because_the_rms_match_cancels_the_half():
    """NOT a bug, and the reason this arm has no zero-init no-op state.

    `gate`'s carry is half `sum`'s at W_g = 0, and the injected term is the carry scaled
    to the carrier's RMS — a scale that cancels any constant factor. So the two modes
    inject the SAME tensor at pass 1 and differ only once W_g moves.
    """
    _ids0, inp, lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1)
    with torch.no_grad():
        a = _model(**kw, loop_carry="sum")(inp, labels=lab, slot_layout=layout)["loss"]
        b = _model(**kw, loop_carry="gate")(inp, labels=lab, slot_layout=layout)["loss"]
    assert torch.allclose(a, b, atol=1e-6)


def test_the_carry_resets_every_forward():
    """Two identical forwards in a row must give identical carries, not a growing one."""
    _ids0, inp, _lab, layout = _batch()
    m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum",
               slot_depth_fixed=3, slot_max_depth=8)
    a = _reads(m, inp, layout)[-1]["carry_prev"]
    b = _reads(m, inp, layout)[-1]["carry_prev"]
    assert torch.equal(a, b)


def test_the_carry_is_exactly_zero_for_cell_zero_and_for_pads():
    """The fixture's own gate for the normalisation clause below.

    Under `loop_reach w` core layer 0's window row for cell 0 is cells −w..−1, which is
    empty, and the branch's XSA excludes the self token — so cell 0's READ is exactly 0
    and its carry stays 0 for the whole forward. A pad slot DOES read (its window row
    holds a real cell), but it is never valid, so the update masks it out and ITS carry
    stays 0 too. Both must therefore add EXACTLY nothing, which is what the `where` in
    `carry_rms_match` is for — and the reason `test_the_rms_match_gradient_is_finite_at_
    a_zero_carry` is not a hypothetical.
    """
    _ids0, inp, _lab, layout = _batch()
    m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum",
               slot_depth_fixed=3, slot_max_depth=8)
    recs = _reads(m, inp, layout)
    for t, r in enumerate(recs):
        assert float(r["read"][:, 0].abs().max()) == 0.0, \
            f"pass {t}: cell 0 read something — its window row is not empty"
        assert float(r["carry"][:, 0].abs().max()) == 0.0
        assert float(r["carry"][~layout.slot_valid].abs().max()) == 0.0, \
            f"pass {t}: a pad slot accumulated a carry"
    # Two-sided: a pad DOES read, so the zero above is the MASK's doing and not an
    # accident of an inert probe.
    assert float(recs[-1]["read"][~layout.slot_valid].abs().max()) > 0.0


# ── 3. THE CARRY DOES NOT WIDEN THE REACH ────────────────────────────────────

def _span_delta(mdl, ids, layout, back: int, k: int = 4) -> float:
    """max |Δlogit| over span k's TOKEN positions when span k−back is edited."""
    edited = _edit(ids, layout, 0, k - back)
    a, lay = _logits(mdl, ids)
    b, _ = _logits(mdl, edited)
    tgt = (~lay.slot_mask[0]) & (lay.bag_id[0] == k)
    assert bool(tgt.any()), "fixture: span k must hold token positions"
    d = (a[0] - b[0]).abs().nan_to_num(0.0)
    d = torch.where(a[0] != b[0], d, torch.zeros_like(d))
    return float(d[tgt].max())


@pytest.mark.parametrize("mode", ["sum", "gate"])
def test_the_carry_cannot_widen_one_pass_but_depth_still_carries(mode):
    """Depth 1: two spans back moves, three spans back does NOT. Depth 3: it does.

    `tests/test_tul_strict_geometry.py::
    test_reach_prev_plus_loop_reach_needs_depth_to_carry_three_spans` fixes the depth-1
    zero for the ruler. Here it must SURVIVE the carry: a carry that widened the per-pass
    reach would break the middle clause, and one that carried nothing would break the
    last.
    """
    ids, _inp, _lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry=mode,
              slot_max_depth=8)
    d1 = _model(**kw, slot_depth_fixed=1)
    assert _span_delta(d1, ids, layout, back=2) > 0.0, \
        "depth 1 did not carry an edit TWO spans back — the fixture is inert"
    assert _span_delta(d1, ids, layout, back=3) == 0.0, \
        f"loop_carry={mode!r} widened one pass: an edit three spans back reached span 4"
    d3 = _model(**kw, slot_depth_fixed=3)
    assert _span_delta(d3, ids, layout, back=3) > 0.0, \
        f"loop_carry={mode!r} at depth 3 carried nothing three spans back"


# ── 4. THE NORMALISATION IS LIVE ─────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["sum", "gate"])
def test_the_injected_carry_sits_at_the_carriers_rms_at_every_pass(mode):
    """`carry/inject_ratio_t{t}` is exactly 1 at every pass that injects anything.

    Read off the MODEL's own instrument, so a change to the rule that forgot the
    instrument, or the reverse, fails here.
    """
    _ids0, inp, lab, layout = _batch()
    m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry=mode,
               slot_depth_fixed=3, slot_max_depth=8)
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    ratios = {k: float(v) for k, v in out.items() if k.startswith("carry_inject_ratio_t")}
    # Pass 0 injects nothing (c(0) = 0), so depth 3 gives ratios at passes 1 and 2.
    assert sorted(ratios) == ["carry_inject_ratio_t1", "carry_inject_ratio_t2"], ratios
    for k, v in ratios.items():
        assert abs(v - 1.0) < 1e-4, f"{k} = {v}"


def test_rms_match_is_exact_on_nonzero_cells_and_exactly_zero_on_zero_ones():
    """The rule itself, away from the model: scale to the carrier's RMS, or to nothing."""
    torch.manual_seed(0)
    h = torch.randn(2, 5, 4, 8) * 3.0          # [B, S, n, C] carrier
    c = torch.randn(2, 5, 8)
    c[0, 2] = 0.0                              # a cell with an exactly zero carry
    term = carry_rms_match(c, h)
    rms_h = h.flatten(2).pow(2).mean(-1).sqrt()
    rms_t = term.pow(2).mean(-1).sqrt()
    ok = torch.ones(2, 5, dtype=torch.bool)
    ok[0, 2] = False
    assert torch.allclose(rms_t[ok], rms_h[ok], rtol=1e-5)
    assert float(term[0, 2].abs().max()) == 0.0
    # Direction preserved: the term is a positive multiple of the carry.
    cos = torch.nn.functional.cosine_similarity(term[ok], c[ok], dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_the_rms_match_gradient_is_finite_at_a_zero_carry():
    """The sqrt(0) trap: an infinite derivative times a masked zero cotangent is NaN.

    This is not hypothetical — before the two guards in `carry_rms_match` the whole
    model's total gradient read `nan` on the tiny fixture, because every pad slot and
    cell 0 of every row hold an exactly zero carry.
    """
    h = torch.randn(1, 3, 2, 8)
    c = torch.zeros(1, 3, 8, requires_grad=True)
    carry_rms_match(c, h).sum().backward()
    assert torch.isfinite(c.grad).all()
    assert float(c.grad.abs().max()) == 0.0, \
        "a zero carry took gradient through the match — it would ride a 1/eps scale"


def test_training_step_gradients_are_finite_on_both_modes():
    """End to end, with the backward: no NaN anywhere in the model."""
    _ids0, inp, lab, layout = _batch()
    for mode in ("sum", "gate"):
        m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry=mode).train()
        torch.manual_seed(7)
        res = m(inp, labels=lab, slot_layout=layout)
        res["loss"].backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{mode}: {n} has a non-finite grad"


def test_the_carry_state_grows_with_depth_and_is_reported():
    """`carry/rms_t{t}` is a real reading, not a constant: accumulation must show."""
    _ids0, inp, lab, layout = _batch()
    m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum",
               slot_depth_fixed=3, slot_max_depth=8)
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    r = [float(out[f"carry_rms_t{t}"]) for t in (1, 2, 3)]
    assert r[0] > 0 and r[1] > r[0] and r[2] > r[1], r


def test_gate_mean_is_reported_on_gate_and_absent_on_sum():
    _ids0, inp, lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1, slot_depth_fixed=3,
              slot_max_depth=8)
    with torch.no_grad():
        g = _model(**kw, loop_carry="gate")(inp, labels=lab, slot_layout=layout)
        s = _model(**kw, loop_carry="sum")(inp, labels=lab, slot_layout=layout)
    assert abs(float(g["carry_gate_mean_t1"]) - 0.5) < 1e-6
    assert not any(k.startswith("carry_gate_mean") for k in s)


# ── PERSIST (tul.loop_carry="persist", LXTUL-R Step 2) ───────────────────────
#
# `sum` / `gate` re-inject the accumulated read at the entry of EVERY later pass, and
# that re-supply is the measured failure this arm answers
# (lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md: carry RMS 0.5 -> 75).
# `persist` accumulates the SAME reads, the SAME way, but never hands them back to a
# pass — the whole sum is added ONCE, at the loop's exit. The tests below prove, in
# order: no parameter and no new state-dict keys; the per-pass MAP is bit-identical to
# `loop_carry: "none"` (the property `sum` does NOT have, from pass 1 on); the exit
# state is exactly `h_none + carry_rms_match(sum of reads, h_none)`; `carry/persist_ratio`
# is reported (and only on `persist`); persist is NOT refused on a Thought Register
# (`tul.slot_cells > 1` / `tul.fan_k > 0`), unlike `sum` / `gate`; and gradients are
# finite end to end.


def test_persist_builds_no_parameter_and_its_param_names_match_none():
    a = _model()
    b = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="persist")
    assert b.tul_carry is not None and b.tul_carry.mode == "persist"
    assert b.tul_carry.W_g is None
    assert not any(k.startswith("tul_carry.") for k in b.state_dict())
    assert set(a.state_dict()) == set(b.state_dict())


def _core_step_outputs(m: MORPHTransformer, inp, layout):
    """Spy on `_apply_core_step`; return the carrier it RETURNS at every pass — upstream
    of any exit-only add, so this is "the carrier after each pass" the docstring claims
    bit-identity for."""
    real = m._apply_core_step
    outs = []

    def spy(*a, **kw):
        h_out, rs = real(*a, **kw)
        outs.append(h_out.detach().clone())
        return h_out, rs

    m._apply_core_step = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
        return outs
    finally:
        m._apply_core_step = real


def test_persist_per_pass_trajectory_is_bit_identical_to_none_and_sum_is_not():
    """The test that distinguishes persist from sum: persist never injects at a pass
    entry, so the carrier `_apply_core_step` returns at every pass must be bit-identical
    to a `loop_carry: "none"` twin at the same seed and weights — unlike `sum`, which
    diverges starting at pass 1 (its carry(0) = 0, so pass 0 still agrees)."""
    _ids0, inp, _lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1, slot_depth_fixed=3,
              slot_max_depth=8)
    m_persist = _model(**kw, loop_carry="persist")
    m_none = _model(**kw)
    out_p = _core_step_outputs(m_persist, inp, layout)
    out_n = _core_step_outputs(m_none, inp, layout)
    assert len(out_p) == len(out_n) == 3
    for t, (a, b) in enumerate(zip(out_p, out_n)):
        assert torch.equal(a, b), f"pass {t}: persist's per-pass carrier diverged from none"
    m_sum = _model(**kw, loop_carry="sum")
    out_s = _core_step_outputs(m_sum, inp, layout)
    assert torch.equal(out_s[0], out_n[0]), "pass 0: sum must still agree (carry(0) = 0)"
    assert not torch.equal(out_s[1], out_n[1]), \
        "sum must diverge from none by pass 1 — persist's agreement above is not vacuous"


def _tul_core_exit_h(m: MORPHTransformer, inp, layout):
    """Spy on `_tul_core`; return its raw return value `h` (the second tuple entry, the
    compact-axis carrier `_forward_tul` renames `h_slots`) — AFTER any exit-only add."""
    real = m._tul_core
    captured: dict = {}

    def spy(*a, **kw):
        ret = real(*a, **kw)
        captured["h"] = ret[1]
        return ret

    m._tul_core = spy
    try:
        with torch.no_grad():
            m(inp, labels=None, slot_layout=layout)
        return captured["h"]
    finally:
        m._tul_core = real


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_persist_exit_is_the_raw_exit_plus_the_rms_matched_sum_of_reads(depth):
    """h_persist == h_none + carry_rms_match(sum of the T captured reads, h_none), where
    h_none is `_tul_core`'s raw return from a `loop_carry: "none"` twin at the same seed
    and weights. Exactly zero on cell 0 of every row (its window row is empty under reach
    1) and on every pad slot."""
    _ids0, inp, _lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1, slot_depth_fixed=depth,
              slot_max_depth=8)
    m_persist = _model(**kw, loop_carry="persist")
    m_none = _model(**kw)
    recs = _reads(m_persist, inp, layout)
    assert len(recs) == depth, f"forced depth {depth} ran {len(recs)} passes"
    # `recs[-1]["carry"]` IS the sum of the T reads, already masked to the slots whose
    # pass actually ran (`test_sum_carry_is_the_sum_of_its_reads_at_forced_depth` proves
    # that equality for `sum`, which accumulates identically). Re-summing `recs[*]["read"]`
    # here directly would NOT reproduce it: a pad slot READS something nonzero at every
    # pass (its window row holds a real cell) but never accumulates — the accumulator's
    # own `active & slot_valid` mask keeps its carry at exactly 0 — so an unmasked sum of
    # raw reads is the WRONG reference at every pad cell.
    total_accum = recs[-1]["carry"]
    h_none = _tul_core_exit_h(m_none, inp, layout)
    h_persist = _tul_core_exit_h(m_persist, inp, layout)
    # `h_none` is the Hyper-Connection carrier `[B, S, n, C]` on this fixture; the
    # production add goes through `_apply_injection`, which broadcasts the single-stream
    # `[B, S, C]` term onto the stream axis — a raw `+` would try to broadcast on the
    # wrong axis.
    expected = MORPHTransformer._apply_injection(
        h_none, carry_rms_match(total_accum, h_none))
    assert torch.allclose(h_persist, expected, atol=1e-5), \
        f"depth {depth}: persist's exit != h_none + carry_rms_match(sum reads, h_none)"
    v = layout.slot_valid
    assert float((h_persist[:, 0] - h_none[:, 0]).abs().max()) == 0.0, \
        "cell 0 of every row must get exactly zero persist term (empty window row)"
    assert float((h_persist[~v] - h_none[~v]).abs().max()) == 0.0, \
        "a pad slot must get exactly zero persist term"


def test_persist_ratio_is_reported_and_absent_on_sum_and_none():
    _ids0, inp, lab, layout = _batch()
    kw = dict(tg_coda_prefix_reach="prev", loop_reach=1, slot_depth_fixed=3,
              slot_max_depth=8)
    with torch.no_grad():
        p = _model(**kw, loop_carry="persist")(inp, labels=lab, slot_layout=layout)
        s = _model(**kw, loop_carry="sum")(inp, labels=lab, slot_layout=layout)
        n = _model(**kw)(inp, labels=lab, slot_layout=layout)
    assert "carry_persist_ratio" in p
    assert 0.0 < float(p["carry_persist_ratio"]) <= 1.0 + 1e-4
    assert not any(k.startswith("carry_persist_ratio") for k in s)
    assert not any(k.startswith("carry_") for k in n)


def test_persist_is_not_refused_at_slot_cells_greater_than_one():
    """The mirror of the `slot_cells=2` case in `test_loop_carry_refusals`: persist must
    NOT raise there, unlike sum/gate — LXTUL-R Step 2 is measured on the register."""
    _tul(loop_carry="persist", loop_reach=1, slot_cells=2, prefix_k=2)   # must not raise


def test_persist_training_step_gradients_are_finite():
    _ids0, inp, lab, layout = _batch()
    m = _model(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="persist").train()
    torch.manual_seed(7)
    res = m(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    n_finite = 0
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"persist: {name} has a non-finite grad"
            n_finite += 1
    assert n_finite > 0


# ── 5. THE REFUSALS ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(loop_carry="always"), "must be one of"),
    (dict(loop_carry="sum"), "loop_reach=0"),
    (dict(loop_carry="gate", loop_reach=1, slot_cells=2, prefix_k=2), "slot_cells=2"),
    (dict(loop_carry="sum", loop_reach=1, tokens_through_core=True),
     "tokens_through_core=True"),
    (dict(loop_carry="sum", loop_reach=1, loop_reads_tokens=True),
     "loop_reads_tokens=True"),
])
def test_loop_carry_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)


def test_scse_is_refused_at_build():
    cfg = _tiny(tul=_tul(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum"),
                scse_enabled=True)
    with pytest.raises(NotImplementedError, match="loop_carry under SCSE"):
        MORPHTransformer(cfg)


def test_a_coreless_model_is_refused_at_build():
    cfg = _tiny(tul=_tul(tg_coda_prefix_reach="prev", loop_reach=1, loop_carry="sum"),
                n_core=0)
    with pytest.raises(ValueError, match="loop_carry needs a core loop"):
        MORPHTransformer(cfg)


def test_the_module_refuses_to_be_built_in_none_mode():
    with pytest.raises(ValueError, match="'none' builds no module"):
        TULLoopCarry(8, "none")


def test_unknown_carry_keys_raise_through_tul_setup():
    """`loop_carry` is in KNOWN_TUL_KEYS and a typo next to it still raises."""
    reject_unknown_tul_keys({"loop_carry": "sum", "loop_reach": 1})
    with pytest.raises(ValueError, match="loop_carrry"):
        reject_unknown_tul_keys({"loop_carrry": "sum"})


def test_the_window_capture_is_refused_outside_tg_restrict():
    """The capture means 'the cross-cell reach read' only under the restriction."""
    m = _model(tg_restrict=False, tg_geometry="restrict")
    cap: dict = {}
    for blk in (m.prelude[0], m.prelude[1]):        # one CSA layer and one HCA layer
        with pytest.raises(NotImplementedError, match="tg_win_capture"):
            blk.attention(torch.randn(1, 16, 64), tg_win_capture=cap)
    assert cap == {}
    # The control: a tg_restrict model FILLS it, so the refusal above is about the
    # branch and not about the kwarg never being read.
    r = _model()
    r.prelude[0].attention(torch.randn(1, 16, 64), tg_win_capture=cap)
    assert cap["win"].shape == (1, 16, 64)
