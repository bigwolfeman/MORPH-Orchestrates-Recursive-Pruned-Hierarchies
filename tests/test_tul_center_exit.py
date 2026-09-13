"""`tul.center_exit` — the ROW-CENTERED EXIT (lever C1, 2026-09-13).

THE DEFECT. On the strict ruler a row's 64 written slot states sit at effective rank
5.7598 in 1024 dimensions with mean pairwise cosine 0.7104. A large SHARED component is
the cheapest account of the cosine. This lever subtracts the row's mean over its valid
slots from every valid slot's exit state and adds back one learned bias `b_center`.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. `center_exit: false` builds no parameter, draws no RNG and leaves the
   forward bit-identical.
2. THE STATE IS ACTUALLY CENTERED, at the point the CODA reads it. The per-row mean over
   valid slots of the state handed to `prefix_project` equals `b_center` (zero at init).
3. PADS ARE UNTOUCHED. Bit-exact against an OFF twin at the same weights.
4. EVERY READER SEES IT. The span decoder and the coda's prefix write read the centered
   state, and so does `tul_slot_state_probe` — the arm's own headline instrument, which
   re-runs the front and would otherwise report the loop's raw exit.
5. PER CELL INDEX at M = 4. Cell i is centered against cell i of the other slots and NOT
   against its own siblings, so the register's within-slot axis survives.
6. THE MEAN IS LIVE. A common shift of every valid slot's carrier leaves the output
   EXACTLY unchanged, forward and backward — a detached mean would pass the forward half
   and fail the backward half.
7. THE GRADIENT REACHES `b_center`.
8. THE REFUSALS: the paid loop, `loop_reads_tokens`, an FM planner, `n_core: 0`.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md
Note: .agents/notes/proposed/architecture/2026-09-13-rank-levers-center-and-contrast.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULCenterExit, TULConfig
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
    """`max_slots` 10 with a short token budget, so the row carries PAD slots."""
    spec = TulLayoutSpec(seq_len=64, prefix_k=max(M, 2), max_slots=10, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(_ids(seed=seed), _rule(), spec)
    return inp, lab, layout


def _model(M: int = 1, center: bool = False, seed: int = 99, **tul_kw):
    kw = dict(prefix_k=max(M, 2), center_exit=center)
    if M > 1:
        kw.update(slot_cells=M, slot_cell_init="distinct")
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


def _spy_written(model, inp, lab, layout):
    """The state `prefix_project` is handed — i.e. what the CODA gets, at the real seam.

    Returns ``(h_slots, cells)``: ``h_slots`` ``[B, S, *carrier]`` and, on a register
    model, ``cells`` ``[B, S, M, *carrier]``. Both are the values AFTER whatever the
    forward did to them, which is the whole point of spying here and not earlier.
    """
    seen: dict = {}
    orig = model.tul.prefix_project

    def spy(h_slots, layout_, l_total, cells=None):
        seen["h_slots"] = h_slots.detach().clone()
        seen["cells"] = None if cells is None else cells.detach().clone()
        return orig(h_slots, layout_, l_total, cells=cells)

    model.tul.prefix_project = spy
    try:
        out = model(inp, labels=lab, slot_layout=layout)
    finally:
        model.tul.prefix_project = orig
    return seen["h_slots"], seen["cells"], out


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

def test_off_builds_no_parameter():
    m = _model(1, center=False)
    assert m.cfg.tul.center_exit is False
    assert m.tul_center is None
    assert not any("tul_center" in k for k in m.state_dict())


def test_on_adds_exactly_one_tensor_and_moves_no_base_weight():
    """`b_center` is a `torch.zeros`, so it draws NO RNG: a center model's base weights
    must be byte-identical to its ruler's and the arm differs by the mechanism alone."""
    a, b = _model(1, center=False), _model(1, center=True)
    ka, kb = set(a.state_dict()), set(b.state_dict())
    assert kb - ka == {"tul_center.b_center"}
    assert ka - kb == set()
    for k in sorted(ka):
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), (
            f"{k} differs — building TULCenterExit drew from the GLOBAL RNG stream")


def test_b_center_is_zero_at_init_and_shaped_like_the_carrier():
    m = _model(1, center=True)
    bc = m.tul_center.b_center
    assert tuple(bc.shape) == (m._n_streams, m.cfg.d_model)
    assert float(bc.detach().abs().max()) == 0.0


def test_off_forward_is_bit_identical_to_a_model_without_the_knob():
    """The OFF claim as a NUMBER, not as an inspection. The two configs differ only in a
    flag that builds nothing, so loss, logit sum and grad sum must agree to the last bit."""
    inp, lab, layout = _batch()
    outs = []
    for center in (False, False):
        m = _model(1, center=center)
        torch.manual_seed(3)
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        outs.append((float(out["loss"]),
                     sum(float(p.grad.abs().sum()) for p in m.parameters()
                         if p.grad is not None)))
    assert outs[0] == outs[1]


# ── 2. THE STATE IS ACTUALLY CENTERED, WHERE THE CODA READS IT ───────────────

def test_written_state_has_row_mean_equal_to_b_center():
    m = _model(1, center=True)
    inp, lab, layout = _batch()
    h, cells, _out = _spy_written(m, inp, lab, layout)
    assert cells is None
    valid = layout.slot_valid                                   # [B, S]
    w = valid.reshape(*valid.shape, *([1] * (h.dim() - 2))).to(h.dtype)
    mu = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)       # [B, *carrier]
    target = m.tul_center.b_center.detach().expand_as(mu)
    assert torch.allclose(mu, target, atol=1e-6), (
        f"per-row mean over valid slots is {float(mu.abs().max()):.3e} from b_center — "
        f"the written state is NOT centered")


def test_a_nonzero_b_center_is_what_the_row_mean_becomes():
    """The bias is the row's mean, not a decoration: move it and the mean must follow."""
    m = _model(1, center=True)
    with torch.no_grad():
        m.tul_center.b_center.normal_(std=0.3,
                                      generator=torch.Generator().manual_seed(11))
    inp, lab, layout = _batch()
    h, _c, _o = _spy_written(m, inp, lab, layout)
    valid = layout.slot_valid
    w = valid.reshape(*valid.shape, *([1] * (h.dim() - 2))).to(h.dtype)
    mu = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    assert torch.allclose(mu, m.tul_center.b_center.detach().expand_as(mu), atol=1e-5)


def test_the_lever_actually_moves_the_forward():
    """A lever that changes no number is not a lever. OFF and ON share every base weight
    (proved above), so any difference here is the centering."""
    inp, lab, layout = _batch()
    a, b = _model(1, center=False), _model(1, center=True)
    torch.manual_seed(3)
    la = float(a(inp, labels=lab, slot_layout=layout)["loss"])
    torch.manual_seed(3)
    lb = float(b(inp, labels=lab, slot_layout=layout)["loss"])
    assert la != lb, "center_exit=True gave the ruler's loss — the lever is inert"


# ── 3. PADS ARE UNTOUCHED ────────────────────────────────────────────────────

def test_pad_slots_are_bit_identical_to_the_off_twin():
    inp, lab, layout = _batch()
    assert bool((~layout.slot_valid).any()), "fixture has no pad slot — nothing is tested"
    a, b = _model(1, center=False), _model(1, center=True)
    torch.manual_seed(3)
    ha, _ca, _oa = _spy_written(a, inp, lab, layout)
    torch.manual_seed(3)
    hb, _cb, _ob = _spy_written(b, inp, lab, layout)
    pad = ~layout.slot_valid
    assert torch.equal(ha[pad], hb[pad]), (
        "a PAD slot's state moved under center_exit — pads must be excluded from the "
        "mean AND left exactly as they were (the center_bag_mean dump-bin rule)")
    assert not torch.equal(ha[layout.slot_valid], hb[layout.slot_valid]), (
        "the VALID slots did not move either — the fixture proves nothing")


def test_pads_are_excluded_from_the_mean_itself():
    """Directly on the module: garbage at a pad must not move a valid slot's value."""
    torch.manual_seed(0)
    mod = TULCenterExit(8, n_streams=0)
    h = torch.randn(1, 6, 8)
    valid = torch.tensor([[True, True, True, True, False, False]])
    out_a = mod(h, valid)
    h2 = h.clone()
    h2[0, 4:] = 1e3
    out_b = mod(h2, valid)
    assert torch.equal(out_a[0, :4], out_b[0, :4]), (
        "a pad slot's value entered the row mean")


# ── 4. EVERY READER SEES IT ──────────────────────────────────────────────────

def test_the_span_decoder_reads_the_centered_state():
    """`spandec` is taken from `h_slots` BELOW the centering line, so its value must move.
    Base weights are shared, so this isolates the centering."""
    inp, lab, layout = _batch()
    a, b = _model(1, center=False), _model(1, center=True)
    torch.manual_seed(3)
    sa = float(a(inp, labels=lab, slot_layout=layout)["spandec"])
    torch.manual_seed(3)
    sb = float(b(inp, labels=lab, slot_layout=layout)["spandec"])
    assert sa != sb, "the span decoder read the UNcentered state"


def test_sigreg_reads_the_centered_state():
    inp, lab, layout = _batch()
    a = _model(1, center=False, sigreg_lambda=1e-3, sigreg_slices=16)
    b = _model(1, center=True, sigreg_lambda=1e-3, sigreg_slices=16)
    torch.manual_seed(3)
    sa = float(a(inp, labels=lab, slot_layout=layout)["sigreg"])
    torch.manual_seed(3)
    sb = float(b(inp, labels=lab, slot_layout=layout)["sigreg"])
    assert sa != sb, "SIGReg read the UNcentered state"


def test_the_state_probe_applies_the_centering():
    """`tul_slot_state_probe` re-runs the front, so it needs its OWN call. Without it the
    arm's headline instrument (`val/slot_eff_rank`, `val/slot_pairwise_cos`) would report
    the loop's raw exit and be blind to the only thing the arm does — the exact defect
    `cond_layers` was fixed for."""
    inp, _lab, layout = _batch()
    a, b = _model(1, center=False), _model(1, center=True)
    with torch.no_grad():
        b.tul_center.b_center.normal_(std=0.2,
                                      generator=torch.Generator().manual_seed(4))
    pa = a.eval().tul_slot_state_probe(inp, layout)
    pb = b.eval().tul_slot_state_probe(inp, layout)
    assert pa["slot_pairwise_cos"] != pb["slot_pairwise_cos"], (
        "the probe reported the same geometry with and without the centering")


# ── 5. PER CELL INDEX AT M = 4 ───────────────────────────────────────────────

def test_each_cell_index_is_centered_separately_at_m4():
    m = _model(4, center=True)
    inp, lab, layout = _batch(4)
    _h, cells, _out = _spy_written(m, inp, lab, layout)
    assert cells is not None and cells.shape[2] == 4
    valid = layout.slot_valid
    w = valid.reshape(*valid.shape, *([1] * (cells.dim() - 2))).to(cells.dtype)
    mu = (cells * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)     # [B, M, *carrier]
    target = m.tul_center.b_center.detach().expand_as(mu)
    assert torch.allclose(mu, target, atol=1e-6), (
        f"per-CELL-INDEX row mean is {float((mu - target).abs().max()):.3e} from "
        f"b_center — the cells were pooled into one mean")


def test_pooling_the_cells_into_one_mean_would_be_a_different_operation():
    """The per-cell choice is only meaningful if the cell means DIFFER from each other.
    Blank the cell-index structure (give every cell the same value) and the per-cell means
    collapse onto the pooled one; with the register's real cells they do not."""
    torch.manual_seed(0)
    mod = TULCenterExit(8, n_streams=0)
    valid = torch.tensor([[True] * 4 + [False]])
    h = torch.randn(1, 5 * 4, 8)                       # S=5, M=4, slot-major
    v = h.reshape(1, 5, 4, 8)
    per_cell = (v[:, :4] .mean(dim=1))                 # [1, 4, 8]
    pooled = v[:, :4].reshape(1, -1, 8).mean(dim=1)    # [1, 8]
    assert not torch.allclose(per_cell, pooled.unsqueeze(1).expand_as(per_cell), atol=1e-4)
    # ...and the module really uses the per-cell one.
    out = mod(h, valid, m_cells=4).reshape(1, 5, 4, 8)
    assert torch.allclose(out[:, :4].mean(dim=1), torch.zeros(1, 4, 8), atol=1e-6)


def test_the_cell_axis_is_slot_major_and_the_values_do_not_move_positions():
    """THE LAYOUT, pinned by VALUE. `_tul_core` returns the compact cell axis slot-major
    (index `s*M + i`, `TULSlotRegister`'s own contract). Reading it cell-major instead is a
    PERMUTATION of which slot's state lands where — and it still satisfies the "per-cell
    row mean equals b_center" identity, because the permutation maps groups onto groups.
    Found by sabotage D7, which the mean test missed. So this one checks the VALUES.

    Slot `s` and cell `m` carry `(s+1) * (1 + 0.01*m)`. The cell factor MULTIPLIES rather
    than adds, so it does not cancel out of the per-cell row mean and the two axes stay
    separable after centering: the mean at cell `m` is `mean_s(s+1) * (1 + 0.01*m)` and
    the output at `s*M+m` is `(s + 1 - mean_s(s+1)) * (1 + 0.01*m)`, exactly.
    """
    mod = TULCenterExit(8, n_streams=0)
    S, M, C = 5, 4, 8
    h = torch.zeros(1, S * M, C)
    for s in range(S):
        for m in range(M):
            h[0, s * M + m] = (s + 1) * (1.0 + 0.01 * m)
    out = mod(h, torch.ones(1, S, dtype=torch.bool), m_cells=M)
    mean_slot = sum(s + 1 for s in range(S)) / S
    for s in range(S):
        for m in range(M):
            want = (s + 1 - mean_slot) * (1.0 + 0.01 * m)
            assert torch.allclose(out[0, s * M + m], torch.full((C,), want), atol=1e-5), (
                f"cell {m} of slot {s} reads {float(out[0, s * M + m][0]):.4f}, expected "
                f"{want:.4f} — the cell axis is not slot-major, so a slot's state landed "
                f"at another slot's position")


def test_m4_off_is_bit_identical_to_the_register_ruler():
    inp, lab, layout = _batch(4)
    a, b = _model(4, center=False), _model(4, center=False)
    torch.manual_seed(3)
    la = float(a(inp, labels=lab, slot_layout=layout)["loss"])
    torch.manual_seed(3)
    lb = float(b(inp, labels=lab, slot_layout=layout)["loss"])
    assert la == lb


# ── 6. THE MEAN IS LIVE (forward AND backward invariance) ────────────────────

def test_a_common_shift_leaves_the_output_exactly_unchanged():
    torch.manual_seed(0)
    mod = TULCenterExit(8, n_streams=0)
    h = torch.randn(2, 6, 8)
    valid = torch.tensor([[True] * 5 + [False], [True] * 3 + [False] * 3])
    c = torch.randn(1, 1, 8)
    shifted = torch.where(valid.unsqueeze(-1), h + c, h)
    assert torch.allclose(mod(h, valid), mod(shifted, valid), atol=1e-6), (
        "centering is not invariant to a shared offset — it is not a centering")


def test_the_gradient_of_a_common_shift_is_exactly_zero():
    """THE DETACH TEST. A detached mean gives the SAME forward values and a DIFFERENT
    backward: the common mode would keep a live gradient the readers cannot see."""
    torch.manual_seed(0)
    mod = TULCenterExit(8, n_streams=0)
    h = torch.randn(2, 6, 8)
    valid = torch.tensor([[True] * 5 + [False], [True] * 3 + [False] * 3])
    w = torch.randn(2, 6, 8)
    c = torch.zeros(1, 1, 8, requires_grad=True)
    out = mod(torch.where(valid.unsqueeze(-1), h + c, h), valid)
    (out * w).sum().backward()
    assert float(c.grad.abs().max()) < 1e-6, (
        f"a shared offset carries gradient {float(c.grad.abs().max()):.3e} — the row mean "
        f"is DETACHED, so the optimizer is still pushing a direction no reader can read")


def test_gradient_reaches_b_center():
    m = _model(1, center=True)
    inp, lab, layout = _batch()
    m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    g = m.tul_center.b_center.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    assert bool(torch.isfinite(g).all())


def test_no_parameter_reads_a_nan_gradient():
    m = _model(4, center=True)
    inp, lab, layout = _batch(4)
    m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    assert bad == [], f"non-finite gradient at {bad}"


def test_an_all_pad_row_is_returned_untouched_and_finite():
    torch.manual_seed(0)
    mod = TULCenterExit(8, n_streams=0)
    h = torch.randn(1, 4, 8)
    valid = torch.zeros(1, 4, dtype=torch.bool)
    out = mod(h, valid)
    assert torch.equal(out, h) and bool(torch.isfinite(out).all())


# ── 7. THE REFUSALS ──────────────────────────────────────────────────────────

def test_center_exit_refuses_the_paid_loop():
    with pytest.raises(ValueError, match="center_exit"):
        TULConfig(prefix_k=2, slot_id=4, center_exit=True, tokens_through_core=True)


def test_center_exit_refuses_loop_reads_tokens():
    with pytest.raises(ValueError, match="center_exit"):
        TULConfig(prefix_k=2, slot_id=4, center_exit=True, loop_reads_tokens=True)


def test_center_exit_refuses_a_zero_core():
    with pytest.raises(ValueError, match="center_exit"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(center_exit=True)))


def test_center_exit_refuses_an_fm_planner():
    """The FM refusal must fire BEFORE the n_core one, because a planner arm legitimately
    runs n_core=0 — otherwise the message would send the reader to the wrong knob."""
    from morph.model.tul_fm import FMArmConfig

    # A bare TULConfig: `tg_geometry="strict"` refuses a planner on its own, and a test
    # that trips the WRONG refusal proves nothing about this one.
    tc = TULConfig(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                   mux_beta=0.0, center_exit=True)
    with pytest.raises(NotImplementedError, match="center_exit"):
        MORPHTransformer(_tiny(n_core=0, tul=tc,
                               fm=FMArmConfig(d_p=16, n_layers=1, n_heads=2, d_ff=32,
                                              cond_dim=16, max_slots=10, l_total=84)))


def test_center_exit_shape_mismatch_raises():
    mod = TULCenterExit(8, n_streams=0)
    with pytest.raises(ValueError, match="expected"):
        mod(torch.randn(1, 7, 8), torch.ones(1, 4, dtype=torch.bool), m_cells=1)
