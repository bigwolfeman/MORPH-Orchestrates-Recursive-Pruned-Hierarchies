"""A2s — tg_restrict threaded through the token core region (tokens_through_core).

Prereg: lab/experiments/planned/2026-09-02-a2s-restricted-paid-loop.md.
CPU only, eager, tiny config — the test_tg_span_comp.py scaffolding. Covers:
construction (the old raise is gone), forward+backward, the masks actually
REACHING the core (dropping them changes the function), strict position-level
causality, and reorder-equivariance under the active-set permutation (the
mask-follows-the-sort property that makes A2s correct at nonuniform depths).
"""
from __future__ import annotations

import numpy as np
import torch

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


def _a2s_cfg(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, tg_restrict=True, tokens_through_core=True,
                token_state_dropout=0.0)
    base.update(kw)
    return TULConfig(**base)


def _model(tul: TULConfig, seed=1234) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul))


def test_construction_no_longer_raises():
    m = _model(_a2s_cfg())
    assert m._tg_restrict and m.cfg.tul.tokens_through_core


def test_forward_backward_run():
    x, y, layout, _stats = _batch(seed=5)
    m = _model(_a2s_cfg())
    out = m(x, labels=y, slot_layout=layout)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss under A2s: {loss}"
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for _n, g in grads), "non-finite gradient"


def test_masks_reach_the_core():
    """Dropping attn_kwargs at the _core_region seam must CHANGE the logits —
    identical logits would mean the threading silently fell out and an A2s
    training run would duplicate A2 with extra steps."""
    x, _y, layout, _stats = _batch(seed=5)
    m = _model(_a2s_cfg())
    m.eval()
    orig = m._core_region
    with torch.no_grad():
        l_masked = m(x, labels=None, slot_layout=layout)["logits"]
        m._core_region = lambda *a, **kw: orig(*a[:4], attn_kwargs=None)
        try:
            l_unmasked = m(x, labels=None, slot_layout=layout)["logits"]
        finally:
            m._core_region = orig
    assert not torch.equal(l_masked, l_unmasked), "attn_kwargs never reached the core"


def test_causality_under_fixed_layout():
    """Strict position-level causality on the A2s path: with the layout held
    fixed, perturbing the token at position p leaves every logit before p
    unchanged (torch.equal — the slot_id column is -inf everywhere, and
    (-inf)-(-inf) NaN-poisons a difference-based check)."""
    x, _y, layout, _stats = _batch(B=1, seed=7)
    m = _model(_a2s_cfg())
    m.eval()
    tokpos = torch.nonzero(~layout.slot_mask[0], as_tuple=False).flatten()
    p = int(tokpos[len(tokpos) * 2 // 3])
    x2 = x.clone()
    x2[0, p] = 6 if int(x[0, p]) != 6 else 7
    with torch.no_grad():
        l1 = m(x, labels=None, slot_layout=layout)["logits"]
        l2 = m(x2, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(l1[0, :p], l2[0, :p]), f"acausal leak before pos {p}"
    assert not torch.equal(l1[0, p:], l2[0, p:]), "perturbation had no effect at all"


def test_reorder_equivariance_at_nonuniform_depth():
    """THE active-set property: with per-sample depths forced nonuniform so the
    sort actually permutes, reversing the batch order must reverse the outputs
    and nothing else. If the masks did not follow the perm, the reversed batch
    would run sample A under sample B's visibility and the rows would differ."""
    x, _y, layout, _stats = _batch(B=2, seed=9)
    m = _model(_a2s_cfg())
    m.train()  # training path samples depths — patched below to force the perm

    def _run(inp, lay, depths):
        m._sample_depths = lambda B, dev: torch.tensor(depths, device=dev)
        with torch.no_grad():
            return m(inp, labels=None, slot_layout=lay)["logits"]

    rev = torch.tensor([1, 0])
    l_fwd = _run(x, layout, [1, 3])
    l_rev = _run(x[rev], layout.index_select(rev) if hasattr(layout, "index_select")
                 else _reindex(layout, rev), [3, 1])
    assert torch.allclose(l_fwd[0], l_rev[1], atol=1e-5), "sample 0 changed under reorder"
    assert torch.allclose(l_fwd[1], l_rev[0], atol=1e-5), "sample 1 changed under reorder"


def _reindex(layout, idx):
    import dataclasses
    kw = {}
    for f in dataclasses.fields(layout):
        v = getattr(layout, f.name)
        kw[f.name] = v[idx] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 2 else v
    return dataclasses.replace(layout, **kw)
