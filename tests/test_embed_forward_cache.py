"""`MORPHEmbedding.forward_cache` (2026-10-10, training-speed pass, exact).

The TUL forward opens one scope in which every parametrized table (`embed_quant`: the int-N
fake quantisation of the euclidean and bigram tables) is quantised once and the tied head
`lm_weight()` is built once. Before it, the lxtul_pointer step quantised the euclidean table
three times (front lookups, token CE, span decoder) and built the head twice.

What each test pins (CPU, fp32, the tiny `lxtul_pointer` winner composed by Hydra with its
quantisation applied, as tests/test_graph_step.py builds it):
  1. One training forward quantises each table ONCE and builds the head ONCE (spies on
     `IntNRowSTE.eager`, the CPU quantiser, and on the head's Lorentz log-map at table size);
     without the cache it is 3 + 1 quantisations and 2 heads. The scope caches the int-N
     quantisers ONLY: a ternary weight is rebuilt at every access as before (a global
     parametrize cache would sum its bf16 per-use gradients in bf16).
  2. The training forward and backward are BIT-IDENTICAL with and without the cache: the
     loss, every `out` tensor and every parameter gradient (`torch.equal`).
  3. The scope leaves nothing behind (depth 0, no cached head) and nests; outside a scope
     `lm_weight()` builds a new tensor on every call.
"""
from __future__ import annotations

import contextlib
import dataclasses

import torch

from test_graph_step import _TINY, _compose

from morph.model import embed_quant, embeddings


def _cpu_model(monkeypatch, seed=0):
    from morph.model.transformer import MORPHTransformer
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config
    cfg, rt = _compose(monkeypatch, "lxtul_pointer")
    mc = dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg), **_TINY)
    torch.manual_seed(seed)
    m = MORPHTransformer(mc)
    apply_quantization(m, cfg)
    m.tul_fan_target_build()
    m.tul_fan_target_sync()
    m.train()
    return m, rt


def _cpu_batch(rt):
    """One packed batch (test_graph_step's stream and packer), on the CPU."""
    import numpy as np
    from morph.model.tul_layout import TulLayoutSpec, pack_tul_batch
    from test_graph_step import B, SEQ, _stream
    from test_tul_strict_geometry import _rule
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=rt.data_cfg.prefix_k, max_slots=10,
                         slot_id=rt.data_cfg.slot_id)
    x, y, lay = pack_tul_batch(_stream(B * 400, seed=101, p_boundary=0.12), _rule(), spec, B)
    assert int(lay.slot_valid.sum()) > 0 and np.isfinite(float(x.float().mean()))
    return x, y, lay


def _step(m, x, y, lay, seed=11):
    m.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=lay)
    out["loss"].backward()
    grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}
    return {k: v.detach().clone() for k, v in out.items() if torch.is_tensor(v)}, grads


def _count(monkeypatch):
    n = {"quant": 0, "head": 0}
    real_q = embed_quant.IntNRowSTE.eager
    real_lm = embeddings._log_map_origin

    def q(self, w):
        n["quant"] += 1
        return real_q(self, w)

    def lm(x):
        if x.dim() == 2 and x.shape[0] >= 60:                    # the table, not a lookup
            n["head"] += 1
        return real_lm(x)
    monkeypatch.setattr(embed_quant.IntNRowSTE, "eager", q)
    monkeypatch.setattr(embeddings, "_log_map_origin", lm)
    return n


def test_one_forward_quantises_each_table_once_and_builds_the_head_once(monkeypatch):
    m, rt = _cpu_model(monkeypatch)
    x, y, lay = _cpu_batch(rt)
    n = _count(monkeypatch)
    torch.manual_seed(1)
    m(x, labels=y, slot_layout=lay)
    cached = dict(n)
    n["quant"] = n["head"] = 0
    monkeypatch.setattr(embeddings.MORPHEmbedding, "forward_cache",
                        lambda self: contextlib.nullcontext())
    torch.manual_seed(1)
    m(x, labels=y, slot_layout=lay)
    assert cached == {"quant": 2, "head": 1}, cached           # euclidean + bigram; one head
    assert n["quant"] >= 4 and n["head"] >= 2, dict(n)         # the uncached forward


def test_the_scope_leaves_parametrize_uncached(monkeypatch):
    import torch.nn.utils.parametrize as P
    m, rt = _cpu_model(monkeypatch)
    seen = []
    with m.embed.forward_cache():
        seen.append(P._cache_enabled)
    assert seen == [0]


def test_training_step_is_bit_identical_with_and_without_the_cache(monkeypatch):
    m, rt = _cpu_model(monkeypatch)
    x, y, lay = _cpu_batch(rt)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    gen = m.gain_pass_rng_state()                    # the hinge's pass draw: same pass twice
    out_c, g_c = _step(m, x, y, lay)
    m.load_state_dict(sd)
    m.set_gain_pass_rng_state(gen)
    monkeypatch.setattr(embeddings.MORPHEmbedding, "forward_cache",
                        lambda self: contextlib.nullcontext())
    out_u, g_u = _step(m, x, y, lay)
    assert set(out_c) == set(out_u)
    for k in out_c:
        assert torch.equal(out_c[k], out_u[k]) or (
            torch.isnan(out_c[k]).all() and torch.isnan(out_u[k]).all()), k
    assert torch.isfinite(out_c["loss"])
    n_live = 0
    for k in g_c:
        assert (g_c[k] is None) == (g_u[k] is None), k
        if g_c[k] is not None:
            assert torch.equal(g_c[k], g_u[k]), k
            n_live += 1
    # the two tables the head reads carry a live gradient (the bigram table's is exactly 0 at
    # init: its per-layer lambdas start at 0), so the comparison above is not of zeros
    emb = [k for k in g_c if "euc_embed" in k or "space_embed" in k]
    assert len(emb) == 2 and all(float(g_c[k].abs().max()) > 0 for k in emb), emb
    assert n_live > 50


def test_the_scope_leaves_nothing_behind_and_nests(monkeypatch):
    m, rt = _cpu_model(monkeypatch)
    h = m.embed.hybrid
    assert h.lm_weight() is not h.lm_weight()                  # no scope: a new head each call
    with m.embed.forward_cache():
        a = m.embed.lm_weight()
        with m.embed.forward_cache():
            assert m.embed.lm_weight() is a
        assert h._cache_depth == 1 and m.embed.lm_weight() is a
    assert h._cache_depth == 0 and h._lm_w is None
    with m.embed.forward_cache():
        assert m.embed.lm_weight() is not a                    # a new scope, a new head
    x, y, lay = _cpu_batch(rt)
    m(x, labels=y, slot_layout=lay)
    assert h._cache_depth == 0 and h._lm_w is None
    qs = [q for q in m.modules() if isinstance(q, embed_quant.IntNRowSTE)]
    assert len(qs) == 2 and all(q._scope_depth == 0 and q._scope_out is None for q in qs)


def test_a_no_grad_first_read_does_not_cut_the_gradient(monkeypatch):
    """Inside the scope a no-grad read of the table first, then an attached read: the
    attached read must carry the gradient (the cache is rebuilt, not reused)."""
    m, _rt = _cpu_model(monkeypatch)
    emb = m.embed.hybrid.euc_embed
    with m.embed.forward_cache():
        with torch.no_grad():
            a = emb.weight
        b = emb.weight
        c = emb.weight
    assert not a.requires_grad and b.requires_grad and c is b
    b.sum().backward()
    assert emb.parametrizations.weight.original.grad is not None
