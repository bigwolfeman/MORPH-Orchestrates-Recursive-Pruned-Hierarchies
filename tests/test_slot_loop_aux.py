"""The slot loop (`_tul_core`) under the k=12 panel's three additions (2026-09-07):

the terminal fixed-point term ported from `_core_region` (base.yaml `core_fixed_point_lambda`),
`tul.slot_depth_fixed` (every valid slot loops exactly k times, train and eval), and the arc
E8 lookahead heads on the TUL coda, read on the compacted TOKEN stream. CPU only, tiny config.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.training.tul_setup import reject_unknown_tul_keys

V = 64


def _cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed=5, train=True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg(**kw))
    m.train(train)
    return m


def _layout(seed=0, B=2, n=90, **spec_kw):
    lut = np.zeros(V, dtype=bool)
    lut[[10, 11]] = True
    lut[0] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)
    spec = TulLayoutSpec(**{**dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4), **spec_kw})
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::6] = 10
    return slot_layout_from_ids(ids.astype(np.int64), rule, spec)


SLOT = dict(prefix_k=2, slot_id=4)


# ── fixed depth ───────────────────────────────────────────────────────────────────────
def test_fixed_depth_is_exact_on_every_valid_slot_in_train_and_eval():
    x, y, layout, _ = _layout()
    m = _model(tul=TULConfig(**SLOT, slot_depth_fixed=3))
    for train in (True, False):
        m.train(train)
        d = m._sample_slot_depths(layout, x.device)
        assert torch.equal(d[layout.slot_valid], torch.full_like(d[layout.slot_valid], 3))
        assert torch.equal(d[~layout.slot_valid], torch.ones_like(d[~layout.slot_valid]))
    # 0 keeps the Poisson draw: eval is the mean, training varies
    p = _model(tul=TULConfig(**SLOT))
    p.eval()
    assert torch.equal(p._sample_slot_depths(layout, x.device)[layout.slot_valid],
                       torch.full((int(layout.slot_valid.sum()),), 2, dtype=torch.long))


def test_fixed_depth_above_the_slot_max_refuses_to_build_and_the_key_is_known():
    with pytest.raises(ValueError, match="slot_depth_fixed"):
        _model(tul=TULConfig(**SLOT, slot_depth_fixed=4))          # max_depth 3
    _model(tul=TULConfig(**SLOT, slot_depth_fixed=12, slot_max_depth=12))
    reject_unknown_tul_keys(OmegaConf.create({"slot_depth_fixed": 12}))
    with pytest.raises(ValueError, match="slot_depth_fixd"):
        reject_unknown_tul_keys(OmegaConf.create({"slot_depth_fixd": 12}))


# ── the fixed-point term on the slot loop ─────────────────────────────────────────────
def test_slot_loop_fixed_point_term_adds_to_the_loss_and_reaches_the_core():
    x, y, layout, _ = _layout()
    tul = TULConfig(**SLOT, slot_depth_fixed=3)
    torch.manual_seed(1)
    off = _model(core_fixed_point_lambda=0.0, tul=tul)(x, labels=y, slot_layout=layout)
    torch.manual_seed(1)
    m = _model(core_fixed_point_lambda=2.0, tul=tul)
    on = m(x, labels=y, slot_layout=layout)
    assert "fixed_point" not in off
    assert on["fixed_point"] > 0 and torch.isfinite(on["loss"])
    assert torch.allclose(on["loss"] - on["fp_weighted"], off["loss"], atol=1e-6)
    assert torch.allclose(on["fp_weighted"], 2.0 * on["fixed_point"])
    on["fp_weighted"].backward()
    core = [p.grad for n, p in m.named_parameters() if n.startswith("core.") and p.grad is not None]
    assert core and any(g.abs().sum() > 0 for g in core)
    # eval never applies it; a label-less forward leaves no stash
    m.eval()
    with torch.no_grad():
        o = m(x, labels=y, slot_layout=layout)
    assert "fp_weighted" not in o
    m.train()
    m(x, labels=None, slot_layout=layout)
    assert m._core_aux is None


def test_slot_loop_fixed_point_term_matches_the_formula_at_depth_one():
    """At fixed depth 1 every valid slot finishes at iteration 0: the term is the mean over
    valid slots of ||h_1 - h_0||^2 / ||h_1||^2, recomputed here from the loop's own states."""
    x, y, layout, _ = _layout()
    m = _model(core_fixed_point_lambda=1.0, tul=TULConfig(**SLOT, slot_depth_fixed=1))
    m.train()
    out = m(x, labels=y, slot_layout=layout)
    # recompute: the entry state and one core step, through the model's own pieces
    xf, x0, bigram = m._tul_front(x, layout)
    xn = m.input_norm(xf)
    from morph.model.tul import gather_valid
    e = gather_valid(xn, layout.slot_index, layout.slot_valid)
    h0 = m.core_init(e)
    xn2, h1, depths, *_ = m._tul_core(xf, x0, bigram, layout)
    fn, fo = h1.flatten(2).float(), h0.flatten(2).float()
    rel = (fn - fo).pow(2).sum(-1) / (fn.pow(2).sum(-1) + 1e-6)
    expect = rel[layout.slot_valid].mean()
    assert torch.allclose(out["fixed_point"], expect, rtol=1e-4, atol=1e-6), \
        (float(out["fixed_point"]), float(expect))


def test_slot_loop_fixed_point_term_under_scse_builds_and_is_finite():
    x, y, layout, _ = _layout()
    m = _model(core_fixed_point_lambda=1.0, scse_enabled=True,
               tul=TULConfig(**SLOT, slot_depth_fixed=2))
    m.train()
    out = m(x, labels=y, slot_layout=layout)
    assert torch.isfinite(out["fixed_point"]) and 0.0 <= float(out["fixed_point"]) < 10.0


# ── lookahead heads on the TUL coda ───────────────────────────────────────────────────
def test_token_compact_keeps_token_order_and_drops_every_slot_label():
    x, y, layout, _ = _layout()
    B, L = y.shape
    xh = torch.arange(B * L, dtype=torch.float32).view(B, L, 1).expand(B, L, 3).clone()
    xt, lt = MORPHTransformer._tul_token_compact(xh, y, layout)
    for b in range(B):
        tok = (~layout.slot_mask[b]).nonzero().flatten()
        assert torch.equal(lt[b], y[b, tok])
        assert torch.equal(xt[b, :, 0], xh[b, tok, 0])
    assert lt.shape[1] == int((~layout.slot_mask[0]).sum())


def test_mtp_heads_on_the_tul_coda_add_to_the_loss_and_keep_ce_main_at_eval():
    x, y, layout, _ = _layout()
    tul = TULConfig(**SLOT, slot_depth_fixed=2)
    ref = _model(tul=tul)
    m = _model(mtp_heads=3, mtp_weight=0.5, tul=tul)
    torch.manual_seed(1)                      # the coda's token dropout draws here
    plain = ref(x, labels=y, slot_layout=layout)
    torch.manual_seed(1)
    out = m(x, labels=y, slot_layout=layout)
    assert "ce_mtp_2" in out and "ce_mtp_3" in out and "mtp_weighted" in out
    assert torch.allclose(out["loss"] - out["mtp_weighted"], plain["loss"], atol=1e-5)
    assert torch.allclose(out["mtp_weighted"], 0.5 * (out["ce_mtp_2"] + out["ce_mtp_3"]))
    out["mtp_weighted"].backward()
    assert all(h.proj.weight.grad is not None for h in m.mtp)
    # at eval the TUL metric ce_main (the token CE) survives the heads: identity-init heads
    # leave the backbone untouched, so it equals the head-less model's ce_main exactly
    ref.eval()
    m.eval()
    with torch.no_grad():
        ev = m(x, labels=y, slot_layout=layout)
        rv = ref(x, labels=y, slot_layout=layout)
    assert torch.equal(ev["ce_main"], rv["ce_main"])
    assert not torch.allclose(ev["ce_main"], ev["loss"])   # loss carries the heads


def test_mtp_heads_refuse_a_gathered_coda():
    x, y, layout, _ = _layout()
    m = _model(mtp_heads=2, tul=TULConfig(**SLOT, coda_sees_slots=False))
    with pytest.raises(NotImplementedError, match="gathered coda"):
        m(x, labels=y, slot_layout=layout)
