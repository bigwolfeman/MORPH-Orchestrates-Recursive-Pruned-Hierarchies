"""`tul.core_token_aux` — the token CE through the core, in TRAINING only.

Arm `slot-spandec-strict-coretok`, 2026-09-12. In the slot loop the six shared core blocks
are trained by the SLOT losses alone; the plain looped model trains the same blocks on
1,024 next-token targets a row and earns 0.185 nats of depth. This knob adds a
TRAINING-ONLY second pass in which every position goes through the core, and charges the
ordinary weighted token CE on it.

What this file pins, one test per invariant:

* **Off is nothing.** No key, and the method is never called — proved by monkeypatching it
  to raise, not by reading the source.
* **On is PURELY ADDITIVE and RNG-NEUTRAL.** Every parameter is byte-identical to an
  off-model from the same seed (the knob builds no module), and `loss -
  core_token_aux_weighted` equals the off-model's loss BIT FOR BIT — the aux saves and
  restores the generator state, so it consumes nothing from the run's stream.
* **The aux term reaches EVERY core block parameter.** The arm's entire mechanism.
* **Eval never runs it**, and an eval forward is identical with the knob on and off.
* **THE LEAK TEST.** Inside the aux core a token must not read another span's tokens. One
  token id of span 0 is edited; with the slot cells' contribution removed from the aux
  path, nothing outside span 0 may move — bit-exact on CPU fp32. Its own two-sided control:
  with the cells left in, the edit DOES cross, and with the restriction widened to plain
  causal it crosses even with the cells removed.
* **`self._core_aux` and `self._jac_capture` survive the aux.** The aux core writes both;
  the slot loop's fixed-point term must still be the one charged, and a Jacobian capture
  must not mix two maps.

CPU only, fp32, tiny config — the `tests/test_tul_coda_span.py` fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.transformer as tfm
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import (BoundaryRule, TulLayoutSpec, slot_layout_from_ids,
                                    tg_reset_from_ids, tg_segment_ids, tg_strict_allow)

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


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict",
                slot_depth_fixed=3, slot_max_depth=4)
    base.update(kw)
    return TULConfig(**base)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _spec() -> TulLayoutSpec:
    return TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)


def _batch(B: int = 2, n: int = 120, seed: int = 0):
    return slot_layout_from_ids(_ids(B, n, seed), _rule(), _spec())


def _model(seed: int = 99, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    # `BigramEmbedding.lambdas` is zero-init, so the bigram route would be switched off and
    # the leak test below would pass without ever exercising it (the span-mask lesson).
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


def _run(m: MORPHTransformer):
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    return m(inp, labels=lab, slot_layout=layout)


# ── off is nothing ───────────────────────────────────────────────────────────

def test_off_never_calls_the_aux(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the core-token aux ran on a model that did not ask for it")

    monkeypatch.setattr(MORPHTransformer, "_tul_core_token_aux", boom)
    out = _run(_model())
    assert "core_token_aux" not in out and "core_token_aux_weighted" not in out


def test_on_adds_a_positive_term_that_carries_its_weight():
    m = _model(core_token_aux=True, core_token_aux_weight=0.25)
    out = _run(m)
    assert float(out["core_token_aux"]) > 0.0
    assert abs(float(out["core_token_aux_weighted"])
               - 0.25 * float(out["core_token_aux"])) < 1e-5
    assert float(out["core_token_aux_n"]) > 0.0
    # The honesty column: the aux CE against the model's OWN token CE on the same tokens.
    assert float(out["core_token_aux_ce"]) == pytest.approx(float(out["core_token_aux"]),
                                                            abs=1e-6)


def test_the_knob_builds_no_parameter():
    off, on = _model(), _model(core_token_aux=True)
    d_off, d_on = dict(off.named_parameters()), dict(on.named_parameters())
    assert set(d_off) == set(d_on), "the aux must build no module — it reuses the core"
    for n, p in d_off.items():
        assert torch.equal(p, d_on[n]), f"turning the aux on moved {n}"


# ── additive, bit for bit ────────────────────────────────────────────────────

def test_loss_minus_the_weighted_term_is_the_off_models_loss_bitwise():
    """The `spandec_weighted` contract, and the reason the aux restores the RNG state.

    The aux draws a per-sample Poisson depth inside `_core_region`. Without the save and
    restore that draw would shift every later consumer of the stream and this equality
    would hold only by the accident of nothing else drawing after it.
    """
    off, on = _model(), _model(core_token_aux=True)
    o_off, o_on = _run(off), _run(on)
    assert torch.equal(o_on["loss"] - o_on["core_token_aux_weighted"], o_off["loss"])


def test_the_aux_consumes_nothing_from_the_rng_stream():
    torch.manual_seed(11)
    inp, lab, layout, _ = _batch()
    m = _model(core_token_aux=True)
    torch.manual_seed(5)
    m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    after_on = torch.randn(4)
    m2 = _model(core_token_aux=False)
    torch.manual_seed(5)
    m2(inp, labels=lab, slot_layout=layout)["loss"].backward()
    after_off = torch.randn(4)
    assert torch.equal(after_on, after_off), (
        "the aux moved the global RNG stream; the run's later draws would differ from the "
        "ruler's for a reason that is not the mechanism")


# ── the gradient reaches the core ────────────────────────────────────────────

def _live_term(monkeypatch, **tul_kw):
    """The aux's LIVE tensor (`out["core_token_aux"]` is exposed detached)."""
    real = MORPHTransformer._tul_core_token_aux

    def spy(self, *a, **k):
        t = real(self, *a, **k)
        self._ca_term = t
        return t

    monkeypatch.setattr(MORPHTransformer, "_tul_core_token_aux", spy)
    m = _model(core_token_aux=True, **tul_kw)
    _run(m)
    return m


def test_the_aux_term_reaches_every_core_block_parameter(monkeypatch):
    """The arm's entire mechanism: the token CE must land on the SHARED core weights."""
    m = _live_term(monkeypatch)
    names = [n for n, _ in m.named_parameters() if n.startswith("core.")]
    assert names, "fixture: the model must have core blocks"
    params = [dict(m.named_parameters())[n] for n in names]
    g = torch.autograd.grad(m._ca_term, params, allow_unused=True, retain_graph=True)
    live = {n for n, x in zip(names, g) if x is not None and float(x.abs().sum()) > 0.0}
    for b in sorted({n.split(".")[1] for n in names}):
        assert any(n.split(".")[1] == b for n in live), \
            f"core block {b} gets NO gradient from the aux term"


def test_the_aux_term_does_not_reach_the_span_decoder(monkeypatch):
    """The aux is a token objective. It must not train the slot's own reader."""
    m = _live_term(monkeypatch)
    dec = [p for n, p in m.named_parameters() if n.startswith("tul_spandec.")]
    assert dec, "fixture: the span decoder must be built"
    g = torch.autograd.grad(m._ca_term, dec, allow_unused=True, retain_graph=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in g)


# ── eval pays nothing, and is identical on and off ───────────────────────────

def test_eval_never_runs_the_aux(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the core-token aux ran on an EVAL forward")

    monkeypatch.setattr(MORPHTransformer, "_tul_core_token_aux", boom)
    m = _model(core_token_aux=True).eval()
    inp, lab, layout, _ = _batch()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert "core_token_aux" not in out


def test_the_shipped_eval_forward_is_identical_on_and_off():
    """The claim the sweeps depend on: the knob changes TRAINING, not the model."""
    inp, lab, layout, _ = _batch()
    outs = []
    for on in (False, True):
        m = _model(core_token_aux=on).eval()
        with torch.no_grad():
            outs.append(m(inp, labels=lab, slot_layout=layout))
    assert torch.equal(outs[0]["loss"], outs[1]["loss"])
    with torch.no_grad():
        m_off, m_on = _model().eval(), _model(core_token_aux=True).eval()
        a = m_off(inp, labels=None, slot_layout=layout)["logits"]
        b = m_on(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(a, b)


# ── THE LEAK TEST: the aux core is not a DIRECT cross-span channel ───────────
#
# WHAT IS AND IS NOT CLAIMED, because the first draft of this test claimed the wrong thing
# and failed. Over SEVERAL passes information DOES cross a span boundary inside the aux
# core, and that is the design: a cell reads its own span at pass t and a later span's
# token reads that cell at pass t+1, which is exactly the reachability the slot cells have
# inside the shipped loop. What must NEVER happen is a token reading another span's TOKEN
# — the route `tg_geometry: strict` exists to cut.
#
# So the probe isolates ONE core application (`n_core: 1`, depth 1 at eval) and blanks the
# slot cells' carrier and injection sources, leaving the token relation as the only route
# that can move anything. Two two-sided controls keep it from passing vacuously: leave the
# cells in and the edit crosses; widen the relation to plain causal and it crosses with the
# cells still blanked.


def _leak_tiny(**kw) -> MORPHConfig:
    # ONE core layer, ONE pass: within a single application every position is computed from
    # the same input, so a blanked cell cannot have picked its span up yet.
    return _tiny(n_core=1, mean_depth=1, max_depth=1, **kw)


def _leak_model(seed: int = 99) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_leak_tiny(tul=_tul(core_token_aux=True, slot_depth_fixed=1,
                                             slot_max_depth=1)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _edit(ids: np.ndarray, layout, row: int, span: int) -> np.ndarray:
    """Change one NON-boundary token of ``span`` in ``row`` without moving the cut."""
    bag = layout.bag_id[row].numpy()
    tok = ~layout.slot_mask[row].numpy()
    pos = np.flatnonzero((bag == span) & tok)
    assert pos.size >= 3, f"span {span} of row {row} is too short to edit safely"
    p = int(pos[len(pos) // 2])
    out = ids.copy()
    raw = int(tok[:p].sum())
    assert out[row, raw] not in (DOT, 11)
    out[row, raw] = 12 if out[row, raw] != 12 else 13
    return out


def _aux_states(m: MORPHTransformer, ids: np.ndarray, cut_cells: bool):
    """The aux CORE's output at every position, for one batch of ids.

    The prelude runs under the STRICT prelude relation (same span only), exactly as the
    shipped forward runs it, so nothing has crossed a boundary before the core.

    ``cut_cells``: zero the slot cells' carrier and their per-layer injection sources, so
    a cell's key, value and injection are the SAME constant in both runs and carry nothing
    about the edit.
    """
    inp, _lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec())
    pre = tg_strict_allow(layout, "prelude")
    with torch.no_grad():
        x, x0, bg = m._tul_front(
            inp, layout,
            attn_kwargs={"tg_allow": pre, "tg_slot_mask": layout.slot_mask,
                         "tg_comp_allow": pre, "tg_seg": tg_segment_ids(layout)},
            ret_reset_mask=tg_reset_from_ids(tg_segment_ids(layout)))
        if cut_cells:
            sm = layout.slot_mask
            x = x * (~sm).view(*sm.shape, *([1] * (x.dim() - 2))).to(x.dtype)
            x0 = x0 * (~sm).view(*sm.shape, *([1] * (x0.dim() - 2))).to(x0.dtype)
            if bg is not None:
                bg = bg * (~sm).view(*sm.shape, *([1] * (bg.dim() - 2))).to(bg.dtype)
        # The SHIPPED relation, not a rebuild of it: the first version of this file built
        # its own kwargs, and a sabotage that widened the real ones to plain causal was
        # MISSED (2026-09-12). `_core_token_aux_kwargs` has ONE home and this reads it.
        xc = m._core_region(x, x0, bg, inp,
                            attn_kwargs=m._core_token_aux_kwargs(layout))
    return xc, layout


def _leak(m: MORPHTransformer, cut_cells: bool, row: int = 0, span: int = 0) -> float:
    ids = _ids()
    a, layout = _aux_states(m, ids, cut_cells)
    b, _ = _aux_states(m, _edit(ids, layout, row, span), cut_cells)
    bag, tok = layout.bag_id[row], ~layout.slot_mask[row]
    other = tok & (bag != span)
    own = tok & (bag == span)
    d = (a[row] - b[row]).abs().flatten(1)
    assert float(d[own].max()) > 0.0, "fixture: the edit did not move its OWN span"
    return float(d[other].max())


def test_the_aux_core_lets_no_token_read_another_spans_token():
    assert _leak(_leak_model(), cut_cells=True) == 0.0, (
        "one application of the aux core moved another span's token state with the slot "
        "cells blanked — a token is reading another span's token directly, and the arm's "
        "geometry claim is false")


def test_the_leak_probe_sees_the_route_when_the_cells_are_left_in():
    """Two-sided: with the cells carrying their seed the edit DOES cross in one
    application, so the test above is not passing because the probe is blind."""
    assert _leak(_leak_model(), cut_cells=False) > 0.0


def test_a_plain_causal_aux_core_leaks_through_the_same_probe(monkeypatch):
    """Second two-sided control: widen the relation to plain causal and the edit crosses
    with the cells still blanked. The MASK is what holds, not the probe."""
    def wide(layout, *a, **k):
        B, L = layout.bag_id.shape
        row = torch.arange(L).unsqueeze(1)
        col = torch.arange(L).unsqueeze(0)
        return (col <= row).view(1, 1, L, L).expand(B, 1, L, L).clone()

    monkeypatch.setattr(tfm, "tg_allow_mask", wide)
    assert _leak(_leak_model(), cut_cells=True) > 0.0, (
        "the probe cannot see a leak even at plain causal — it measures nothing")


def test_the_aux_core_does_cross_a_boundary_through_a_cell_over_two_passes():
    """The design, stated as a test so nobody reads the leak test as 'nothing crosses'.

    At depth 2 a cell has read its own span at pass 1 and a later span's token reads that
    cell at pass 2, even with the cells blanked at ENTRY.
    """
    torch.manual_seed(99)
    m = MORPHTransformer(_tiny(n_core=1, mean_depth=2, max_depth=2,
                               tul=_tul(core_token_aux=True, slot_depth_fixed=2,
                                        slot_max_depth=2)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    assert _leak(m.eval().float(), cut_cells=True) > 0.0


# ── the aux does not stomp the shipped path's state ──────────────────────────

def test_the_slot_loops_fixed_point_term_survives_the_aux():
    """`_core_region` writes `self._core_aux`; the slot loop wrote it first.

    Without the save/restore the aux's own fixed-point term would be the one charged, and
    `fixed_point` would stop meaning what it means on every other slot arm.
    """
    inp, lab, layout, _ = _batch()
    vals = []
    for on in (False, True):
        torch.manual_seed(99)
        m = MORPHTransformer(_tiny(tul=_tul(core_token_aux=on),
                                   core_fixed_point_lambda=1.0)).train().float()
        torch.manual_seed(3)
        vals.append(m(inp, labels=lab, slot_layout=layout))
    assert torch.equal(vals[0]["fixed_point"], vals[1]["fixed_point"]), (
        "the aux core's fixed-point term replaced the slot loop's")
    assert torch.equal(vals[0]["fp_weighted"], vals[1]["fp_weighted"])
    assert "core_token_aux_fp" in vals[1], "the aux's own reading must still be reported"


def test_a_jacobian_capture_never_sees_the_aux_core():
    m = _model(core_token_aux=True)
    m._jac_capture = []
    inp, lab, layout, _ = _batch()
    torch.manual_seed(3)
    m(inp, labels=lab, slot_layout=layout)
    S = int(layout.slot_index.shape[1])
    assert m._jac_capture, "fixture: the capture recorded nothing at all"
    for rec in m._jac_capture:
        assert rec["h"].shape[1] == S, (
            f"the capture holds a record of width {rec['h'].shape[1]} — the aux core's "
            f"full-axis operating point leaked into the slot map's probe")


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(core_token_aux=True, core_token_aux_weight=0.0), "core_token_aux_weight > 0"),
    (dict(core_token_aux=True, tokens_through_core=True, spandec=False, spandec_layers=2,
          spandec_max_tokens=0, tg_restrict=False, tg_geometry="restrict"),
     "SAME forward twice"),
    (dict(core_token_aux=True, coda_sees_slots=False), "FULL-AXIS coda"),
    (dict(core_token_aux=True, detach_z=True), "detach_z"),
    (dict(core_token_aux_weight=2.0), "silently ignored"),
])
def test_core_token_aux_config_refusals(kw, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _tul(**kw)


def test_a_coreless_model_refuses_the_aux():
    with pytest.raises(ValueError, match="needs a core loop"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(core_token_aux=True)))


def test_the_core_refuses_an_attention_kwarg_it_cannot_permute():
    """`_core_region` sorts its masks into active-set order. A key it does not handle
    would be dropped in silence, which is an unrestricted core wearing a restricted name."""
    m = _model()
    inp, _lab, layout, _ = _batch()
    with torch.no_grad():
        x, x0, bg = m._tul_front(inp, layout)
    with pytest.raises(NotImplementedError, match="does not thread"):
        m._core_region(x, x0, bg, inp, attn_kwargs={"tg_span": {"bag_id": layout.bag_id}})
