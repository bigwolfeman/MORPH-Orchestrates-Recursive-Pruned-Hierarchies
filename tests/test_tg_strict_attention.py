"""``model.tg_fused_attention`` (2026-10-07, graph-step task 2.1): the fused strict TG attention.

morph/kernels/triton/tg_strict_attention.py computes a strict slot-loop model's two TG
attention branches (window + compressed with sink) in one Triton kernel pair, the relation
read from the layout's index tensors. It is NOT bit-identical to the eager path (a flash
online softmax sums the same terms in another order), so it is held to the eager path's own
error against fp64 instead.

What each test pins:
  1. fp64 parity at the winner's shapes (B 6, H 8, D 64; prelude / coda at L 1280 on a packed
     layout, the 256-cell register core): out_comp, out_win, dq, dk, dv, dsink. The kernel's
     max error against an fp64 reference built from the TREE's relations
     (`tg_strict_allow`, `slot_cell_relation`) is no larger than `_TOL` x the eager path's
     (`_tg_slot_attention` + `_window_fallback`, the exact calls the attention module makes)
     on the same inputs.
  2. Edge cases, same bound: a row with no slot at all, a row of minimum-length spans that
     runs out of slots (hundreds of tail pads), an L that is not a tile multiple, a
     2-cell register.
  3. Geometry: perturbing the k/v of ONE span's tokens, or of one slot's cells, changes
     exactly the output rows the tree's relation lets read them, and leaves every other row
     BIT-identical. The strict geometry's invariant: the loop is the only cross-span channel.
  4. Inside torch.compile (fullgraph, no graph break) and under CUDA-graph capture, the op
     returns the eager call's bits; the backward is bit-deterministic.
  5. The training step: a tiny winner (`lxtul_pointer_ditto` + graph_safe) with the key on
     against the key off (loss and every gradient within a relative bound), and the
     graph-captured replay of the key-on step equals its eager step bit for bit
     (tests/test_graph_step.py's own plan, run with the key on).
  6. Refusals (CPU): the key on a geometry the kernel does not implement; `tg_index` with a
     dense mask beside it.

Sabotage checks, run and reverted (session report, not committed):
  (a) the prelude relation drops `bag[i] == bag[j]` -> 1, 2 and 3 fail (prelude cases).
  (b) the causal bound is off by one (`diff >= -1`) -> 1, 2 and 3 fail (prelude and coda).
  (c) a coda token also reads earlier spans' TOKENS -> 1, 2 and 3 fail (coda cases).
  (d) a core cell also reads the next slot's cells -> 1 and 3 fail (core cases).
CUDA except 6. Run 6 on CPU with CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from morph.model.attention import _tg_slot_attention, _window_fallback
from morph.model.transformer import slot_cell_relation
from morph.model.tul_layout import (BoundaryRule, SlotLayout, TulLayoutSpec, pack_tul_row,
                                    tg_strict_allow)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel")

V, DOT, SLOT = 1000, 7, 4
H, D, W = 8, 64, 256
SCALE = D ** -0.5
_TOL = 1.0         # kernel error <= the eager error, per output and per gradient


def _rule(span_cap: int = 32) -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[DOT] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=span_cap, eos_id=0)


def _row(seed: int, p_boundary: float, l_tok: int, max_slots: int, every: int = 0,
         span_cap: int = 32):
    rng = np.random.default_rng(seed)
    n = l_tok + 4 * max_slots + 64
    ids = rng.integers(8, V, size=n)
    if every:
        ids[every - 1::every] = DOT
    else:
        ids[rng.random(n) < p_boundary] = DOT
    spec = TulLayoutSpec(seq_len=l_tok, prefix_k=4, max_slots=max_slots, slot_id=SLOT)
    arrays, _, _ = pack_tul_row(ids, _rule(span_cap), spec)
    return arrays


def _layout(kinds, l_tok=1024, max_slots=64) -> SlotLayout:
    """One row per entry: ("p", p_boundary) random spans, ("every", n) a boundary every n
    tokens (min_span 4 -> spans of exactly 4), ("none",) no boundary at all."""
    rows = []
    for i, k in enumerate(kinds):
        if k[0] == "p":
            rows.append(_row(i, k[1], l_tok, max_slots))
        elif k[0] == "every":
            rows.append(_row(i, 0.0, l_tok, max_slots, every=k[1]))
        else:            # no boundary and a cap past the row: not one slot
            rows.append(_row(i, 0.0, l_tok, max_slots, span_cap=1 << 20))
    return SlotLayout.from_rows(rows, 4).to("cuda")


_REAL = [("p", 1 / 12)] * 6
_EDGE = [("p", 1 / 12), ("none",), ("every", 4), ("p", 1 / 5)]


def _case(kind: str, lay: SlotLayout | None = None, cells: int = 4, n_slots: int = 64):
    """(index dict, eager fn, fp64 window allow, fp64 comp allow, S, B)."""
    from morph.kernels.triton.tg_strict_attention import tg_strict_index
    if kind == "cells":
        rel = slot_cell_relation(n_slots, cells, "cuda")[0]            # [1,1,S,S]
        S = n_slots * cells
        idx = tg_strict_index("cells", cells=cells)

        def eager(q, k, v, sink):
            return (_tg_slot_attention(q, k, v, None, sink, SCALE, relation=rel,
                                       dense_slot_cols=True),
                    _window_fallback(q, k, v, W, q.device, SCALE, 0, relation=rel))
        base, comp_cols = rel.squeeze(1), None
    else:
        allow = tg_strict_allow(lay, kind, **({"coda_prefix_reach": "all"}
                                              if kind == "coda" else {}))
        S = lay.slot_mask.shape[1]
        idx = tg_strict_index(kind, lay.bag_id, lay.slot_mask)

        def eager(q, k, v, sink):
            return (_tg_slot_attention(q, k, v, lay.slot_mask, sink, SCALE, extra_mask=allow,
                                       dense_slot_cols=True),
                    _window_fallback(q, k, v, W, q.device, SCALE, 0, extra_mask=allow))
        base, comp_cols = allow.squeeze(1), lay.slot_mask[:, None, :]
    ii = torch.arange(S, device="cuda")
    near = ((ii[:, None] - ii[None, :]).abs() < W) & (ii[:, None] != ii[None, :])
    aw = base & near
    ac = base if comp_cols is None else base & comp_cols
    return idx, eager, aw, ac, S


def _ref64(q, k, v, sink, aw, ac):
    q, k, v, sink = q.double(), k.double(), v.double(), sink.double()
    s = torch.einsum("bhid,bhjd->bhij", q, k) * SCALE
    pw = torch.softmax(s.masked_fill(~aw.unsqueeze(1), float("-inf")), -1).nan_to_num(0.0)
    sc = s.masked_fill(~ac.unsqueeze(1), float("-inf"))
    B, Hh, S, _ = s.shape
    sc = torch.cat([sc, sink.view(1, Hh, 1, 1).expand(B, Hh, S, 1)], -1)
    pc = torch.softmax(sc, -1)[..., :S]
    return pc @ v, pw @ v


def _inputs(B, S, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(B, H, S, D, device="cuda", generator=g).to(dtype)
               .requires_grad_(True) for _ in range(3))
    sink = torch.randn(H, device="cuda", generator=g).requires_grad_(True)
    gc, gw = (torch.randn(B, H, S, D, device="cuda", generator=g).to(dtype)
              for _ in range(2))
    return q, k, v, sink, gc, gw


def _errors(fn, q, k, v, sink, gc, gw, ref):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=q.dtype != torch.float32):
        c, w = fn(q, k, v, sink)
    g = torch.autograd.grad((c * gc).sum() + (w * gw).sum(), (q, k, v, sink))
    return [(t.double() - r).abs().max().item() for t, r in zip((c, w, *g), ref)]


def _parity(kind, lay=None, cells=4, n_slots=64, seed=0, dtype=torch.bfloat16, tol=_TOL):
    from morph.kernels.triton.tg_strict_attention import tg_strict_attention
    idx, eager, aw, ac, S = _case(kind, lay, cells, n_slots)
    B = 6 if lay is None else lay.slot_mask.shape[0]
    q, k, v, sink, gc, gw = _inputs(B, S, seed, dtype)
    q64, k64, v64, s64 = (t.detach().double().requires_grad_(True) for t in (q, k, v, sink))
    c64, w64 = _ref64(q64, k64, v64, s64, aw, ac)
    ref = (c64, w64) + torch.autograd.grad((c64 * gc.double()).sum() + (w64 * gw.double()).sum(),
                                           (q64, k64, v64, s64))
    e_old = _errors(eager, q, k, v, sink, gc, gw, ref)
    e_new = _errors(lambda *a: tg_strict_attention(*a, idx, W, SCALE), q, k, v, sink, gc, gw,
                    ref)
    for name, a, b in zip(("comp", "win", "dq", "dk", "dv", "dsink"), e_new, e_old):
        assert a <= tol * b, (kind, name, a, b)


# ── 1. real shapes ───────────────────────────────────────────────────────────────────────


@cuda
@pytest.mark.parametrize("kind", ["prelude", "coda"])
@pytest.mark.parametrize("seed", [0, 1])
def test_region_parity_at_real_shapes(kind, seed):
    _parity(kind, _layout(_REAL), seed=seed)


@cuda
@pytest.mark.parametrize("seed", [0, 1])
def test_core_cells_parity_at_real_shapes(seed):
    _parity("cells", seed=seed)


# ── 2. edge cases ────────────────────────────────────────────────────────────────────────


@cuda
@pytest.mark.parametrize("kind", ["prelude", "coda"])
def test_region_edge_rows(kind):
    lay = _layout(_EDGE)
    sm, bag = lay.slot_mask.cpu(), lay.bag_id.cpu()
    assert lay.slot_valid[1].sum() == 0                       # the no-slot row
    assert lay.slot_valid[2].sum() == 64 and (sm[2] & (bag[2] == 64)).sum() > 100  # pads
    _parity(kind, lay)


@cuda
@pytest.mark.parametrize("kind", ["prelude", "coda"])
def test_region_length_not_a_tile_multiple(kind):
    lay = _layout([("p", 1 / 6), ("every", 5)], l_tok=150, max_slots=12)   # L = 198
    assert lay.slot_mask.shape[1] % 64 != 0
    _parity(kind, lay)


@cuda
def test_core_small_register():
    _parity("cells", cells=2, n_slots=37)                     # S = 74, not a tile multiple


@cuda
@pytest.mark.parametrize("kind", ["coda", "cells"])
def test_fp32_inputs(kind):
    """An fp32 forward (no autocast: probes and tests, never the trainer, whose train AND eval
    run under bf16 autocast): full-precision dots and libdevice exp2. Bound 3x, not 1x: both
    paths sit at a few fp32 ulps, and the backward recomputes P from the stored fp32
    log-sum-exp, whose own rounding scales a whole row of P (measured 2026-10-07: comp 1.2x,
    cells dv 2.6x the eager error, 8e-6 absolute)."""
    _parity(kind, _layout(_EDGE) if kind == "coda" else None, dtype=torch.float32, tol=3.0)


# ── 3. geometry: no cross-span leak ──────────────────────────────────────────────────────


def _bits(t):
    return t.detach().contiguous().view(torch.int16)


def _affected(kind, idx, aw, ac, lay, cols, seed):
    """Rows whose output changes when k/v at `cols` ([B, S] bool) change, and the rows the
    relation predicts."""
    from morph.kernels.triton.tg_strict_attention import tg_strict_attention
    B, S = cols.shape
    q, k, v, sink, _, _ = _inputs(B, S, seed)
    with torch.no_grad():
        c0, w0 = tg_strict_attention(q, k, v, sink, idx, W, SCALE)
        noise = torch.randn_like(k)
        m = cols[:, None, :, None]
        c1, w1 = tg_strict_attention(q, torch.where(m, k + noise, k),
                                     torch.where(m, v - noise, v), sink, idx, W, SCALE)
    changed = ((_bits(c0) != _bits(c1)) | (_bits(w0) != _bits(w1))).any(-1).any(1)  # [B,S]
    pred = ((aw | ac) & cols[:, None, :]).any(-1)
    return changed, pred


@cuda
@pytest.mark.parametrize("kind", ["prelude", "coda"])
def test_no_cross_span_leak(kind):
    lay = _layout(_EDGE)
    idx, _, aw, ac, S = _case(kind, lay)
    sm, bag = lay.slot_mask, lay.bag_id
    for s in (0, 3, 17):
        for what, cols in (("tokens", ~sm & (bag == s)), ("cells", sm & (bag == s))):
            if not cols.any():
                continue
            changed, pred = _affected(kind, idx, aw, ac, lay, cols, seed=s)
            assert torch.equal(changed, pred), (kind, what, s,
                                                (changed ^ pred).nonzero()[:5].tolist())
            if kind == "prelude":       # a span's prelude reads its own span and nothing else
                assert not (changed & (bag != s)).any()
            if kind == "coda" and what == "tokens":   # tokens are read by their span only
                assert not (changed & ((bag != s) | sm)).any()


@cuda
def test_core_slot_reads_only_earlier_and_own_slots():
    idx, _, aw, ac, S = _case("cells")
    slot_of = torch.arange(S, device="cuda") // 4
    for s in (0, 9, 63):
        cols = (slot_of == s).unsqueeze(0).expand(6, S)
        changed, pred = _affected("cells", idx, aw, ac, None, cols, seed=s)
        assert torch.equal(changed, pred)
        assert torch.equal(changed[0], slot_of >= s)


# ── 4. compile, capture, determinism ─────────────────────────────────────────────────────


@cuda
def test_compiled_and_captured_calls_return_the_eager_bits():
    from morph.kernels.triton.tg_strict_attention import tg_strict_attention
    lay = _layout(_REAL)
    idx, _, _, _, S = _case("coda", lay)
    q, k, v, sink, gc, gw = _inputs(6, S, 5)

    def step(q, k, v, sink):
        c, w = tg_strict_attention(q, k, v, sink, idx, W, SCALE)
        g = torch.autograd.grad((c * gc).sum() + (w * gw).sum(), (q, k, v, sink))
        return (c, w) + g

    ref = step(q, k, v, sink)
    again = step(q, k, v, sink)
    assert all(torch.equal(a, b) for a, b in zip(ref, again)), "backward not deterministic"

    def fwd(q, k, v, sink):
        c, w = tg_strict_attention(q, k, v, sink, idx, W, SCALE)
        return c * 2 + w
    comp = torch.compile(fwd, fullgraph=True, dynamic=False)        # raises on a graph break
    assert torch.equal(comp(q, k, v, sink), fwd(q, k, v, sink))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qs, ks, vs, ss = (t.detach().clone().requires_grad_(True) for t in (q, k, v, sink))
        step(qs, ks, vs, ss)                                        # warm up on the side
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step(qs, ks, vs, ss)
    torch.cuda.current_stream().wait_stream(side)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(ref, out))


# ── 5. the training step ─────────────────────────────────────────────────────────────────


@cuda
def test_train_step_key_on_matches_key_off(monkeypatch):
    import test_graph_step as G
    out = {}
    for key in ("false", "true"):
        cfg, rt = G._compose(monkeypatch, "lxtul_pointer_ditto", *G._ON,
                             f"model.tg_fused_attention={key}")
        m, _ = G._build(cfg, rt, seed=0)
        assert bool(m.cfg.tg_fused_attention) == (key == "true")
        x, y, lay = G._batches(rt, 3)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = G._fwd(m)(x, y, lay)["loss"]
        loss.backward()
        out[key] = (loss.detach().float(),
                    {n: p.grad.detach().float() for n, p in m.named_parameters()
                     if p.grad is not None})
    (l0, g0), (l1, g1) = out["false"], out["true"]
    assert g0.keys() == g1.keys()
    assert (l1 - l0).abs() <= 1e-2 * l0.abs(), (l0, l1)
    n0 = torch.sqrt(sum((g ** 2).sum() for g in g0.values()))
    dn = torch.sqrt(sum(((g1[n] - g0[n]) ** 2).sum() for n in g0))
    assert dn <= 0.05 * n0, (dn.item(), n0.item())


@cuda
def test_graph_replay_with_the_key_is_its_eager_step(monkeypatch, request):
    import test_graph_step as G
    monkeypatch.setattr(G, "_ON", G._ON + ("model.tg_fused_attention=true",))
    _prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        G.test_replay_is_the_eager_step_bit_for_bit(monkeypatch, None, ())
    finally:
        torch.use_deterministic_algorithms(_prev)


# ── 6. refusals (CPU) ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("over,needle", [
    ("tul.tg_geometry=restrict", "tg_geometry=strict"),
    ("tul.tg_coda_prefix_reach=prev", "tg_coda_prefix_reach=all"),
    ("+tul.loop_reach=1", "loop_reach=0"),
])
def test_config_refusals(monkeypatch, over, needle):
    import test_graph_step as G
    from morph.model.transformer import MORPHTransformer
    from morph.training.train import build_morph_config
    cfg, rt = G._compose(monkeypatch, "lxtul_pointer_ditto", "model.tg_fused_attention=true",
                         over)
    mc = dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg), **G._TINY)
    with pytest.raises(ValueError, match=needle):
        MORPHTransformer(mc)


def test_index_with_a_dense_mask_raises():
    from morph.model.attention import _tg_index_guard
    m = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="replaces them all"):
        _tg_index_guard({"mode": "cells", "cells": 2}, m, None, None, None, None, True, 0)
    with pytest.raises(NotImplementedError, match="tg_restrict"):
        _tg_index_guard({"mode": "cells", "cells": 2}, None, None, None, None, None, False, 0)
    _tg_index_guard(None, m, m, None, None, None, True, 0)          # key off: untouched
