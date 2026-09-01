"""E-SAC-G — learned gated span pooling (tul.tg_span_gate).

Prereg binding: lab/experiments/failures/2026-09-01-span-aligned-compression.md
(P-S1 FALSE -> one learned-gated-pooling variant). Reuses the
test_tg_span_comp.py scaffolding. Covers: config validation, zero-init
equivalence to the mean-pool arm, gate params exist + get grads, the gate
changes the function, hand-value gated pooling, and strict causality.
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


def _batch(B=2, n=200, seed=0):
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=10, slot_id=4)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::10] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(tul: TULConfig | None, seed=1234) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul))


def _sac_cfg(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, tg_restrict=True, tg_span_comp=True)
    base.update(kw)
    return TULConfig(**base)


def test_tg_span_gate_without_span_comp_raises():
    with pytest.raises(ValueError, match="tg_span_gate.*requires.*tg_span_comp"):
        TULConfig(prefix_k=2, slot_id=4, tg_restrict=True, tg_span_gate=True)


def test_zero_init_equals_mean_pool():
    """Zero-init gates make the softmax pool uniform == the mean pool, so the
    gated model must reproduce the mean-pool model's logits when loaded with
    its weights (the ONLY extra keys are the gate params, all zeros)."""
    x, _y, layout, _stats = _batch(seed=5)
    mean = _model(_sac_cfg())
    gated = _model(_sac_cfg(tg_span_gate=True))
    res = gated.load_state_dict(mean.state_dict(), strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    assert res.missing_keys, "gated model added no parameters"
    assert all("tg_span_gate_w" in k for k in res.missing_keys), res.missing_keys
    mean.eval(); gated.eval()
    with torch.no_grad():
        lm = mean(x, labels=None, slot_layout=layout)["logits"]
        lg = gated(x, labels=None, slot_layout=layout)["logits"]
    fin = torch.isfinite(lm)
    assert torch.equal(fin, torch.isfinite(lg))
    assert torch.allclose(lm[fin], lg[fin], atol=1e-4), (
        f"max|d|={float((lm[fin] - lg[fin]).abs().max())}")


def test_gate_params_exist_and_train():
    x, y, layout, _stats = _batch(seed=5)
    m = _model(_sac_cfg(tg_span_gate=True))
    gates = [(n, p) for n, p in m.named_parameters() if "tg_span_gate_w" in n]
    assert gates, "no gate parameters built"
    assert all(torch.all(p == 0) for _n, p in gates), "gates not zero-init"
    out = m(x, labels=y, slot_layout=layout)
    out["loss"].backward()
    got = [n for n, p in gates if p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "no gate parameter received a nonzero gradient"


def test_nonzero_gate_changes_function():
    x, _y, layout, _stats = _batch(seed=5)
    m = _model(_sac_cfg(tg_span_gate=True))
    m.eval()
    with torch.no_grad():
        l0 = m(x, labels=None, slot_layout=layout)["logits"]
        for n, p in m.named_parameters():
            if "tg_span_gate_w" in n:
                p.add_(torch.randn_like(p))
        l1 = m(x, labels=None, slot_layout=layout)["logits"]
    assert not torch.equal(l0, l1), "gate weights are dead — function unchanged"


def test_gated_pool_hand_values():
    """With a nonzero gate the pooled v must equal the manually computed
    within-span softmax-weighted mean."""
    B, H, S, D = 1, 1, 4, 2
    torch.manual_seed(3)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    q = torch.zeros(B, H, S, D)                       # uniform over visible summaries
    bag_id = torch.tensor([[0, 0, 1, 1]])
    token_sel = torch.ones(1, S, dtype=torch.bool)
    span_end = torch.tensor([[1, 3, -1]])
    sink = torch.full((H,), -1e9)
    gate_w = torch.randn(H, D)
    out = _tg_span_attention(q, k, v, bag_id, token_sel, span_end, sink, 1.0,
                             gate_w=gate_w)
    g = (k[0, 0] @ gate_w[0])                         # [S]
    w0 = torch.softmax(g[0:2], dim=0)
    v0 = (w0.unsqueeze(-1) * v[0, 0, 0:2]).sum(0)     # span 0 gated pool
    # pos 2/3 see only span 0's summary; q=0 -> its weight is 1 (sink at -1e9)
    assert torch.allclose(out[0, 0, 2], v0, atol=1e-5)
    assert torch.allclose(out[0, 0, 3], v0, atol=1e-5)
    # pos 0/1: nothing visible -> sink only -> zero output
    assert torch.allclose(out[0, 0, 0], torch.zeros(D), atol=1e-6)


def test_causality_under_fixed_layout():
    x, _y, layout, _stats = _batch(B=1, seed=7)
    m = _model(_sac_cfg(tg_span_gate=True))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "tg_span_gate_w" in n:
                p.add_(torch.randn_like(p))           # exercise the gated path
    m.eval()
    tokpos = torch.nonzero(~layout.slot_mask[0], as_tuple=False).flatten()
    p = int(tokpos[len(tokpos) * 2 // 3])
    x2 = x.clone()
    x2[0, p] = 6 if int(x[0, p]) != 6 else 7
    with torch.no_grad():
        l1 = m(x, labels=None, slot_layout=layout)["logits"]
        l2 = m(x2, labels=None, slot_layout=layout)["logits"]
    # torch.equal, not subtraction: the slot_id column is -inf everywhere and
    # (-inf) - (-inf) = NaN would poison a difference-based check.
    assert torch.equal(l1[0, :p], l2[0, :p]), f"acausal leak before pos {p}"
    assert not torch.equal(l1[0, p:], l2[0, p:]), "perturbation had no effect"
