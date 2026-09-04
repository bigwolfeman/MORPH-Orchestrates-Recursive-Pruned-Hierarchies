"""SIGReg (LeJEPA arXiv 2511.08544) and the auxiliary warmup gates.

Contracts protected here: the statistic actually discriminates the pathology it
is meant to catch (collapse and wrong scale score far above isotropic); pad
slots are excluded; `sigreg_lambda=0` is bit-identical to the default build; the
composite loss decomposes as CE + sigreg_weighted; and a zeroed gate removes an
auxiliary's gradient entirely, which is what makes the warmup schedule real.
CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import numpy as np
import torch

from morph.model.sigreg import sigreg_epps_pulley
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
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


def _batch(B=2, n=60, seed=0):
    # max_slots deliberately ABOVE the number of spans this text produces, so
    # every row carries pad slots — test_sigreg_ignores_pad_slots needs them and
    # asserts they exist rather than assuming.
    spec = TulLayoutSpec(seq_len=48, prefix_k=2, max_slots=12, slot_id=4)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(tul, seed=1234) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul))


def _loss(m, x, y, layout, seed=7):
    torch.manual_seed(seed)
    m.eval()
    return m(x, labels=y, slot_layout=layout)


# ── the statistic itself ─────────────────────────────────────────────────────

def test_sigreg_scores_collapse_and_scale_far_above_isotropic():
    """The pathology SIGReg is here for is MEASURED in MORPH's slot states:
    effective rank 1.7-4.8 in 1024 dims, mean pairwise cosine +0.39..+0.71. A
    statistic that did not separate rank-1 data from isotropic data would be
    decorative, so this asserts a wide margin, not merely a different number."""
    torch.manual_seed(0)
    iso = torch.randn(512, 64)
    collapsed = torch.randn(512, 1) * torch.randn(1, 64)     # rank 1
    scaled = torch.randn(512, 64) * 20.0                     # right shape, wrong scale
    s_iso = float(sigreg_epps_pulley(iso, step=0))
    s_col = float(sigreg_epps_pulley(collapsed, step=0))
    s_scl = float(sigreg_epps_pulley(scaled, step=0))
    assert s_col > 10 * s_iso, f"collapse not separated: {s_col} vs {s_iso}"
    assert s_scl > 10 * s_iso, f"scale not separated: {s_scl} vs {s_iso}"
    assert s_iso >= 0.0


def test_sigreg_is_differentiable_and_pushes_toward_isotropy():
    """Gradient descent on the statistic must actually reduce it — otherwise the
    term is a metric, not a regulariser."""
    torch.manual_seed(0)
    z = (torch.randn(256, 1) * torch.randn(1, 32)).requires_grad_(True)
    before = float(sigreg_epps_pulley(z, step=0))
    for _ in range(25):
        loss = sigreg_epps_pulley(z, step=0)
        g, = torch.autograd.grad(loss, z)
        with torch.no_grad():
            z -= 0.05 * g
    assert float(sigreg_epps_pulley(z, step=0)) < before


def test_sigreg_needs_two_points():
    assert float(sigreg_epps_pulley(torch.randn(1, 8), step=0)) == 0.0


# ── wiring into the model ────────────────────────────────────────────────────

def test_sigreg_lambda_zero_is_bit_identical():
    x, y, layout, _ = _batch()
    o0 = _loss(_model(TULConfig(slot_id=4)), x, y, layout)
    o1 = _loss(_model(TULConfig(slot_id=4, sigreg_lambda=0.0)), x, y, layout)
    assert torch.equal(o0["loss"], o1["loss"])
    assert "sigreg" not in o1


def test_sigreg_loss_decomposes_as_ce_plus_weighted_term():
    x, y, layout, _ = _batch()
    o0 = _loss(_model(TULConfig(slot_id=4)), x, y, layout)
    o1 = _loss(_model(TULConfig(slot_id=4, sigreg_lambda=0.3, sigreg_slices=32)),
               x, y, layout)
    assert "sigreg" in o1 and torch.isfinite(o1["sigreg"]) and float(o1["sigreg"]) > 0
    assert torch.allclose(o1["sigreg_weighted"], 0.3 * o1["sigreg"], atol=1e-6)
    assert torch.allclose(o1["loss"] - o1["sigreg_weighted"], o0["loss"], atol=1e-5)


def test_sigreg_ignores_pad_slots():
    """Pad slots are a fixed-shape artefact. If they reached the statistic, the
    regulariser could lower its loss by moving vectors that mean nothing — and
    the count of real slots differs per row, so the value would drift with
    padding rather than with the model."""
    x, y, layout, _ = _batch()
    m = _model(TULConfig(slot_id=4, sigreg_lambda=0.3, sigreg_slices=32))
    torch.manual_seed(7)
    m.eval()
    xx, x0, bg = m._tul_front(x, layout)
    _xn, h_slots, _d, _g, *_ = m._tul_core(xx, x0, bg, layout)
    assert not bool(layout.slot_valid.all()), "fixture has no pad slots — test void"
    torch.manual_seed(3)
    a = float(m._tul_sigreg_loss(h_slots, layout))
    h2 = h_slots.clone()
    inv = ~layout.slot_valid
    h2[inv] = h2[inv] + 1000.0            # wreck ONLY the pad slots
    torch.manual_seed(3)
    b = float(m._tul_sigreg_loss(h2, layout))
    assert a == b, f"pad slots reached the statistic: {a} vs {b}"


# ── the warmup gate ──────────────────────────────────────────────────────────

def test_zero_gate_removes_the_auxiliary_gradient():
    """The warmup schedule is only real if a zeroed gate removes the term's
    gradient. Control: with the gate at 1.0 the same comparison must DIFFER."""
    x, y, layout, _ = _batch()

    def embed_grad(gate_value, lam=0.3, beta=0.0):
        m = _model(TULConfig(slot_id=4, sigreg_lambda=lam, sigreg_slices=32, mux_beta=beta))
        m.sigreg_gate.fill_(gate_value)
        m.mux_gate.fill_(gate_value)
        out = _loss(m, x, y, layout)
        out["loss"].backward()
        return m.embed.hybrid.euc_embed.weight.grad.clone(), float(out["loss"])

    g_off, l_off = embed_grad(0.0)
    g_on, l_on = embed_grad(1.0)
    base = _model(TULConfig(slot_id=4))
    o = _loss(base, x, y, layout)
    o["loss"].backward()
    g_base = base.embed.hybrid.euc_embed.weight.grad.clone()

    assert torch.allclose(g_off, g_base, atol=1e-6), "gate=0 still changed the gradient"
    assert not torch.allclose(g_on, g_base, atol=1e-6), "gate=1 changed nothing — test void"
    assert abs(l_off - float(o["loss"])) < 1e-5


# ── centered bag-mean (the root cause of slot collapse) ──────────────────────

def _slot_cosine(m, x, layout) -> float:
    """Mean pairwise cosine between the VALID slot input vectors of one row."""
    emb = m.embed.hybrid.euc_embed(x) if False else None      # unused; see below
    with torch.no_grad():
        xx, _x0, _bg = m._tul_front(x, layout)
    v = layout.slot_valid[0]
    idx = layout.slot_index[0][v]
    s = xx[0, idx].reshape(len(idx), -1)
    s = s / s.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    g = s @ s.t()
    off = ~torch.eye(len(idx), dtype=torch.bool)
    return float(g[off].mean())


def test_centering_lowers_the_slot_pairwise_cosine():
    """The collapse is arithmetic: a bag-mean shrinks per-token deviations by
    1/sqrt(span) but preserves the embedding table's common mean exactly, so
    every slot inherits it. Measured on a real checkpoint 2026-08-27:
    ||mean embedding|| 0.423 vs mean deviation 1.049, predicting cosines of
    0.394 (span 4) to 0.839 (span 32) against a measured +0.39..+0.71.

    Centering must therefore REDUCE the pairwise cosine. A model whose embedding
    table happens to be near zero-mean would make this vacuous, so the control
    asserts the uncentered cosine is materially positive first."""
    x, y, layout, _ = _batch()
    # give the embedding table a deliberate common mean, as a trained one has
    m_off = _model(TULConfig(slot_id=4))
    with torch.no_grad():
        m_off.embed.hybrid.euc_embed.weight += 0.5
    torch.manual_seed(1234)
    m_on = _model(TULConfig(slot_id=4, center_bag_mean=True))
    with torch.no_grad():
        m_on.embed.hybrid.euc_embed.weight += 0.5

    c_off = _slot_cosine(m_off, x, layout)
    c_on = _slot_cosine(m_on, x, layout)
    assert c_off > 0.3, f"control is vacuous: uncentered cosine only {c_off:.3f}"
    assert c_on < c_off, f"centering did not decollapse: {c_on:.3f} vs {c_off:.3f}"


def test_center_bag_mean_off_is_bit_identical():
    x, y, layout, _ = _batch()
    o0 = _loss(_model(TULConfig(slot_id=4)), x, y, layout)
    o1 = _loss(_model(TULConfig(slot_id=4, center_bag_mean=False)), x, y, layout)
    assert torch.equal(o0["loss"], o1["loss"])


def test_centering_does_not_touch_the_dump_bin():
    """bag_mean's documented invariant: the dump row is exactly 0 so tail pads
    receive E_slot ALONE. Centering must not hand them -mu instead.

    Driven with a synthetic full-width signal with a deliberate nonzero mean —
    that mean is exactly what centering subtracts, so if the dump bin were
    included the assertion below would fail by that vector."""
    x, y, layout, _ = _batch()
    m = _model(TULConfig(slot_id=4, center_bag_mean=True))
    B, L = x.shape
    C = m.cfg.d_model
    torch.manual_seed(0)
    sig = torch.randn(B, L, C) + 3.0            # large common mean
    with torch.no_grad():
        out = m.tul.slot_input(sig, layout, add_e_slot=True)
    dump = (layout.bag_id >= layout.slot_index.shape[1]) & layout.slot_mask
    if not bool(dump.any()):
        return                      # no tail pads in this fixture; nothing to check
    e = m.tul.E_slot.detach()
    e = e if e.dim() == 1 else e[0]
    assert torch.allclose(out[dump], e.expand_as(out[dump]), atol=1e-5)
