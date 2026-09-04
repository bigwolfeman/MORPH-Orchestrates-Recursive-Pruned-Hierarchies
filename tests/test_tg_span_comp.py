"""E-SAC — span-aligned compression (tul.tg_span_comp).

Prereg: lab/experiments/planned/2026-09-01-span-aligned-compression.md.
CPU only, eager, tiny config — same conventions as test_tg_restrict.py, whose
scaffolding this reuses. Covers: config validation, construction byte-identity
(zero new params / no RNG), the pool+visibility math against hand values, the
shipped forward+backward, flag-reaches-the-mask, and strict causality under a
fixed layout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.attention import _tg_span_attention
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=2,
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


def _spec() -> TulLayoutSpec:
    return TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)


def _batch(B=2, n=200, seed=0):
    spec, rule = _spec(), _rule()
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::10] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), rule, spec)


def _model(tul: TULConfig | None, seed=1234) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul))


def _sac_cfg(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, tg_restrict=True, tg_span_comp=True)
    base.update(kw)
    return TULConfig(**base)


def test_tg_span_comp_without_restrict_raises():
    with pytest.raises(ValueError, match="tg_span_comp.*requires.*tg_restrict"):
        TULConfig(prefix_k=2, slot_id=4, tg_span_comp=True)


def test_construction_byte_identity():
    """The flag builds nothing and draws no RNG: state dicts match bit for bit."""
    a = _model(_sac_cfg())
    b = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True))
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"weight mismatch at {k}"


def test_pool_and_visibility_hand_values():
    """Pooled K/V are span means; visibility is span-granular causal; a query
    with nothing visible resolves to the sink's zero value."""
    B, H, S, D = 1, 1, 5, 2
    k = torch.arange(S * D, dtype=torch.float32).reshape(1, 1, S, D)
    v = 10.0 * torch.arange(S * D, dtype=torch.float32).reshape(1, 1, S, D)
    q = torch.zeros(1, 1, S, D)                       # uniform scores over visible
    bag_id = torch.tensor([[0, 0, 1, 1, 2]])
    token_sel = torch.ones(1, S, dtype=torch.bool)
    # spans: 0 ends at pos 1, 1 ends at pos 3, 2 ends at pos 4; max_slots=3
    span_end = torch.tensor([[1, 3, 4, -1]])
    sink = torch.full((H,), -1e9)                     # kill the sink's share
    out = _tg_span_attention(q, k, v, bag_id, token_sel, span_end, sink, 1.0)
    v0m = v[0, 0, 0:2].mean(0)
    v1m = v[0, 0, 2:4].mean(0)
    # pos 0/1: nothing visible -> sink only -> zero output
    assert torch.allclose(out[0, 0, 0], torch.zeros(D), atol=1e-6)
    assert torch.allclose(out[0, 0, 1], torch.zeros(D), atol=1e-6)
    # pos 2/3: only span 0 visible
    assert torch.allclose(out[0, 0, 2], v0m, atol=1e-5)
    assert torch.allclose(out[0, 0, 3], v0m, atol=1e-5)
    # pos 4: spans 0 and 1, uniform weights (q=0)
    assert torch.allclose(out[0, 0, 4], (v0m + v1m) / 2, atol=1e-5)


def test_own_span_summary_is_invisible():
    """A token never attends its own span's summary (it would leak the span's
    future tokens): perturbing a LATER token of the same span leaves the
    earlier token's branch output unchanged."""
    B, H, S, D = 1, 1, 4, 2
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    bag_id = torch.tensor([[0, 0, 0, 0]])
    token_sel = torch.ones(1, S, dtype=torch.bool)
    span_end = torch.tensor([[3, -1]])
    sink = torch.zeros(H)
    out1 = _tg_span_attention(q, k, v, bag_id, token_sel, span_end, sink, 1.0)
    k2, v2 = k.clone(), v.clone()
    k2[0, 0, 3] += 100.0
    v2[0, 0, 3] += 100.0
    out2 = _tg_span_attention(q, k2, v2, bag_id, token_sel, span_end, sink, 1.0)
    assert torch.equal(out1[0, 0, :3], out2[0, 0, :3])


def test_forward_backward_run():
    """One real forward+backward on the SHIPPED path with tg_span_comp=True."""
    x, y, layout, _stats = _batch(seed=5)
    m = _model(_sac_cfg())
    out = m(x, labels=y, slot_layout=layout)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss under tg_span_comp: {loss}"
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for _n, g in grads), "non-finite gradient"


def test_forward_differs_from_slot_comp():
    """The flag must CHANGE the function — identical logits would mean the
    tg_span kwarg never reached the compressed branch and a training run would
    silently duplicate tul_g0c0."""
    x, _y, layout, _stats = _batch(seed=5)
    slot = _model(TULConfig(prefix_k=2, slot_id=4, tg_restrict=True))
    sac = _model(_sac_cfg())
    sac.load_state_dict(slot.state_dict())
    slot.eval(); sac.eval()
    with torch.no_grad():
        ls = slot(x, labels=None, slot_layout=layout)["logits"]
        lc = sac(x, labels=None, slot_layout=layout)["logits"]
    assert ls is not None and lc is not None
    assert not torch.allclose(ls, lc), "tg_span_comp did not change the forward"


def test_causality_under_fixed_layout():
    """Strict position-level causality: with the layout held fixed, perturbing
    the token at position p leaves every logit at positions < p unchanged."""
    x, _y, layout, _stats = _batch(B=1, seed=7)
    m = _model(_sac_cfg())
    m.eval()
    tokpos = torch.nonzero(~layout.slot_mask[0], as_tuple=False).flatten()
    p = int(tokpos[len(tokpos) * 2 // 3])             # a token position, late-middle
    x2 = x.clone()
    x2[0, p] = 6 if int(x[0, p]) != 6 else 7          # non-boundary, non-slot id
    with torch.no_grad():
        l1 = m(x, labels=None, slot_layout=layout)["logits"]
        l2 = m(x2, labels=None, slot_layout=layout)["logits"]
    # torch.equal, not subtraction: the slot_id logit column is masked to -inf
    # everywhere, and (-inf) - (-inf) = NaN would poison a difference-based check.
    assert torch.equal(l1[0, :p], l2[0, :p]), f"acausal leak before pos {p}"
    assert not torch.equal(l1[0, p:], l2[0, p:]), "perturbation had no effect at all"
