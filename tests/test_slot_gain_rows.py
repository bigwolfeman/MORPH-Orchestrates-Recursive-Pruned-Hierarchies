"""``model.slot_gain_rows``: the gain hinge on k of the B batch rows.

What each test pins:
  1. Key 0 (the default) builds no row counter and never reaches the row choice (CPU build).
  2. THE SLICE IS EXACT. A core application on the sliced rows equals the full application's
     rows (rows attend within themselves only), and the hinge with the key on equals the key-0
     hinge handed the SAME sliced inputs and the same RNG state, bit for bit (gain, penalty,
     per-slot gains).
  3. The kept rows ROTATE: three forwards read three different row sets that together cover
     the batch.
  4. A CUDA graph that records the row choice replays a NEW choice each time, and the whole
     training step (tests/test_graph_step.py, the `gain_rows` case) replays the eager step bit
     for bit with the key on.
  5. Refusals: k < 0, k with the hinge off, k with slot_compact off (CPU build); k >= B at the
     first hinge (CUDA).

Sabotage checks, run and reverted (session report, not committed): see the build note.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from test_graph_step import (_COMPACT, _ON, _TINY, B, _batches, _build, _compose, _fwd, cuda)

K = 2          # rows kept (of B = 3)


def _model(monkeypatch, rows, cpu=False):
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_COMPACT,
                       f"model.slot_gain_rows={rows}", "training.dropout=0.0")
    if cpu:
        from morph.model.transformer import MORPHTransformer
        from morph.training.train import build_morph_config
        return MORPHTransformer(dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg),
                                                    **_TINY))
    return (*_build(cfg, rt, seed=0), rt)


def test_key_zero_builds_no_counter(monkeypatch):
    m = _model(monkeypatch, 0, cpu=True)
    assert not hasattr(m, "_gain_rows_ctr") and m.cfg.slot_gain_rows == 0
    m1 = _model(monkeypatch, 1, cpu=True)
    assert m1._gain_rows_ctr.dtype == torch.long and "_gain_rows_ctr" not in m1.state_dict()


def test_refusals(monkeypatch):
    with pytest.raises(ValueError, match="slot_gain_rows must be >= 0"):
        _model(monkeypatch, -1, cpu=True)
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_COMPACT,
                       "model.slot_gain_rows=1", "model.slot_compact=off",
                       "model.slot_depth_stratified=false")
    from morph.model.transformer import MORPHTransformer
    from morph.training.train import build_morph_config
    mc = dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg), **_TINY)
    with pytest.raises(NotImplementedError, match="slot_compact"):
        MORPHTransformer(mc)
    with pytest.raises(ValueError, match="slot_gain_lambda"):
        MORPHTransformer(dataclasses.replace(
            mc, slot_compact="gather", slot_depth_stratified=True, slot_gain_lambda=0.0))


@cuda
def test_row_count_must_be_a_strict_subset(monkeypatch):
    m, _, rt = _model(monkeypatch, B)
    x, y, lay = _batches(rt, 3)[0]
    with pytest.raises(ValueError, match="strict subset"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _fwd(m)(x, y, lay)


@cuda
def test_slice_is_exact_and_rows_rotate(monkeypatch):
    mk, _, rt = _model(monkeypatch, K)
    m0, _, _ = _model(monkeypatch, 0)
    m0.load_state_dict(mk.state_dict())
    batches = _batches(rt, 3)
    seen, rows_used, checks = [], [], []
    orig_idx = mk._gain_rows_index

    def idx_spy(n_rows, k, device):
        r = orig_idx(n_rows, k, device)
        rows_used.append(r.tolist())
        return r
    mk._gain_rows_index = idx_spy
    orig = mk._slot_gain_penalty

    def spy(core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond, mask, lam, **kw):
        ctr = int(mk._gain_rows_ctr)
        cpu, cuda_s = torch.get_rng_state(), torch.cuda.get_rng_state()
        out = orig(core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond, mask, lam, **kw)
        ri = torch.tensor([(ctr * K + j) % B for j in range(K)], device=h_in.device)
        assert rows_used[-1] == ri.tolist()                   # the formula, written out here
        compact = kw["compact"]
        *plan, caches = compact
        sl = (*(None if p is None else p.index_select(0, ri) for p in plan),
              None if caches is None else tuple(c.index_select(0, ri) for c in caches))
        # reference: the key-0 hinge on the same sliced inputs and RNG state
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda_s)
        kw0 = {**kw, "compact": sl}
        if "renorm_to" in kw0:
            kw0["renorm_to"] = kw0["renorm_to"].index_select(0, ri)
        ref = m0._slot_gain_penalty(core_step, h_in.index_select(0, ri),
                                    e_arg.index_select(0, ri), inj_arg.index_select(1, ri),
                                    None, t, stage_cond, mask.index_select(0, ri), lam, **kw0)
        for key in ("gain", "gain_max", "g_slot"):
            assert torch.allclose(out[key], ref[key], rtol=0, atol=0, equal_nan=True), key
        assert torch.equal(out["penalty"].detach(), ref["penalty"].detach())
        # rows attend within themselves: the application on the kept rows is the full
        # application's rows (kernel tiling differs with the batch, so a tolerance)
        with torch.no_grad():
            full = core_step(h_in.detach(), e_arg.detach(), inj_arg.detach(), iter_idx=t,
                             compact=compact)[0].index_select(0, ri).float()
            part = core_step(h_in.detach().index_select(0, ri), e_arg.detach().index_select(0, ri),
                             inj_arg.detach().index_select(1, ri), iter_idx=t, compact=sl)[0].float()
        checks.append(float((full - part).norm() / full.norm()))
        spy.n += 1
        return out
    spy.n = 0
    mk._slot_gain_penalty = spy
    for x, y, lay in batches:
        mk.zero_grad(set_to_none=True)
        out = _fwd(mk)(x, y, lay)             # fp32: the comparison below is exact
        out["loss"].backward()
    assert spy.n == 3 and len(rows_used) == 3
    assert all(a != b for a, b in zip(rows_used, rows_used[1:])), rows_used
    assert {r for rs in rows_used for r in rs} == set(range(B)), rows_used
    assert max(checks) < 1e-4, checks


@cuda
def test_graph_replays_a_new_row_choice(monkeypatch):
    m, _, _ = _model(monkeypatch, K)
    m._gain_rows_ctr.zero_()
    for _ in range(2):                                         # warm up off the capture stream
        m._gain_rows_index(B, K, "cuda")
    m._gain_rows_ctr.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m._gain_rows_index(B, K, "cuda")
    got = []
    for _ in range(4):
        g.replay()
        got.append(out.tolist())
    assert got == [[(n * K + j) % B for j in range(K)] for n in range(4)], got
    assert int(m._gain_rows_ctr) == 4
