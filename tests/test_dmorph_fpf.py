"""dmorph v1.1 — self-conditioning carry + Fixed-Point Forcing (arXiv 2606.29150).

One test per contract of the design note
(``.agents/notes/rejected/architecture/2026-09-03-fixed-point-forcing-for-dmorph-and-the-loop.md``):
``fpf_p: 0`` is v1 bit-for-bit; the carry reaches the block only through ``W_s``; the
rollout carries no gradient and runs exactly the blocks between ``band(t_start)`` and
``band(t)``; the integrator with ``recur`` visits each block ``recur + 1`` times; the
residual AUROC helper is a real AUROC.
"""

from __future__ import annotations

import math

import pytest
import torch

from morph.model import dmorph as dm_mod
from morph.model.dmorph import (
    DmorphConfig, band_of_t, block_layers, carry_of, fpf_rollout, integrate, ladder,
    noisy_stream, residual_auroc, training_terms,
)
from _dmorph_common import D, batch, clean_pass, dm_cfg, model, wake_stream


def _spy_layers(m):
    calls: list[int] = []
    for gi, layer in enumerate(list(m.prelude) + list(m.coda)):
        orig = layer.forward

        def wrapped(h, *a, _orig=orig, _gi=gi, **kw):
            if "cla_kv" in (kw.get("attn_kwargs") or {}):
                calls.append(_gi)
            return _orig(h, *a, **kw)

        layer.forward = wrapped
    return calls


def _train_terms(m, x, y, layout, want_eval=False):
    xh, caps, ctx = clean_pass(m, x, layout)
    row_w, _p, _z = m._tul_half_weights(y, layout)
    return training_terms(m, xh=xh, labels=y, layout=layout, kv=caps, ctx=ctx, row_w=row_w,
                          want_eval=want_eval)


def test_config_rejects_bad_fpf_values():
    with pytest.raises(ValueError):
        dm_cfg(fpf_p=1.5)
    with pytest.raises(ValueError):
        dm_cfg(recur=-1)
    with pytest.raises(ValueError):
        dm_cfg(fpf_p=0.5, t_per_position=True)


def test_fpf_off_is_v1_bit_for_bit_and_w_s_never_trains():
    """With ``fpf_p: 0`` no carry is formed: the loss equals the v1 loss whatever W_s
    holds, W_s gets no gradient, and no ``dm_fpf_*`` term is emitted."""
    x, y, layout, _ = batch(B=3)
    torch.manual_seed(0)
    m0 = model(dm_cfg(fpf_p=0.0))
    wake_stream(m0)
    m0.train()
    torch.manual_seed(11)
    add0, g0 = _train_terms(m0, x, y, layout)
    m1 = model(dm_cfg(fpf_p=0.0))
    wake_stream(m1)
    with torch.no_grad():
        m1.dmorph.W_s.weight.normal_(0.0, 0.3)       # a non-zero W_s that must not matter
    m1.train()
    torch.manual_seed(11)
    add1, g1 = _train_terms(m1, x, y, layout)
    assert torch.equal(add0, add1)
    assert not any(k.startswith("dm_fpf") for k in g0)
    add1.backward()
    assert m1.dmorph.W_s.weight.grad is None or float(m1.dmorph.W_s.weight.grad.abs().max()) == 0.0


def test_null_carry_and_zero_w_s_are_no_ops_and_a_live_carry_is_not():
    m = model(dm_cfg())
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=2)
    _xh, caps, ctx = clean_pass(m, x, layout)
    B, L = x.shape
    g = torch.Generator().manual_seed(3)
    x_t = torch.randn(B, L, D, generator=g)
    s = carry_of(torch.randn(B, L, D, generator=g))
    t = torch.tensor([0.3, 0.7])
    band = band_of_t(t, m.dmorph.cfg.n_blocks)
    v_none = noisy_stream(m, x_t, t, band, caps, ctx, None)
    v_zero = noisy_stream(m, x_t, t, band, caps, ctx, torch.zeros_like(s))
    v_s0 = noisy_stream(m, x_t, t, band, caps, ctx, s)        # W_s == 0 at construction
    assert torch.equal(v_none, v_zero) and torch.equal(v_none, v_s0)
    with torch.no_grad():
        m.dmorph.W_s.weight.normal_(0.0, 0.3)
    v_live = noisy_stream(m, x_t, t, band, caps, ctx, s)
    assert not torch.allclose(v_live, v_none)
    assert torch.equal(noisy_stream(m, x_t, t, band, caps, ctx, torch.zeros_like(s)), v_none)


def test_rollout_carry_is_detached_and_w_s_trains_through_the_supervised_pass():
    x, y, layout, _ = batch(B=3)
    m = model(dm_cfg(fpf_p=1.0))
    wake_stream(m)
    with torch.no_grad():
        m.dmorph.W_s.weight.normal_(0.0, 0.3)
    m.train()
    xh, caps, ctx = clean_pass(m, x, layout)
    B, L = x.shape
    t = torch.tensor([0.2, 0.55, 0.9])
    x0 = torch.randn(B, L, D) / math.sqrt(D)
    yv = torch.nn.functional.normalize(torch.randn(B, L, D), dim=-1)
    use = torch.tensor([True, True, False])
    s, t_start = fpf_rollout(m, x0, yv, t, use, caps, ctx, w_head=m.embed.lm_weight().detach())
    assert s.requires_grad is False and s.grad_fn is None
    assert bool((t_start <= t).all()) and bool((t_start >= 0).all())
    assert float(s[2].abs().max()) == 0.0                      # null carry where not used
    assert torch.allclose(s[:2].norm(dim=-1), torch.ones(2, L), atol=1e-4)
    add, g = _train_terms(m, x, y, layout)
    assert "dm_fpf_frac" in g and float(g["dm_fpf_frac"]) == 1.0
    add.backward()
    assert m.dmorph.W_s.weight.grad is not None
    assert float(m.dmorph.W_s.weight.grad.abs().max()) > 0.0


def test_rollout_runs_exactly_the_blocks_between_band_t_start_and_band_t():
    m = model(dm_cfg(n_blocks=4))       # one layer per block in the tiny stack
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=2)
    _xh, caps, ctx = clean_pass(m, x, layout)
    B, L = x.shape
    x_in = torch.randn(B, L, D) / math.sqrt(D)
    w = m.embed.lm_weight()
    cases = [
        (torch.tensor([0.10, 0.10]), torch.tensor([0.20, 0.95]), {0: [0, 1], 1: [1], 2: [1], 3: [1]}),
        (torch.tensor([0.60, 0.30]), torch.tensor([0.70, 0.30]), {0: [], 1: [1], 2: [0], 3: []}),
    ]
    for t_start, t_end, want in cases:
        m.dmorph.cfg = DmorphConfig(**{**m.dmorph.cfg.__dict__})
        seen: dict[int, list[int]] = {b: [] for b in range(4)}
        orig = dm_mod.run_block

        def spy(model_, block, x_in_, cond, kv, ctx_, rows, _o=orig):
            seen[block].extend(range(B) if rows is None else rows.tolist())
            return _o(model_, block, x_in_, cond, kv, ctx_, rows)

        dm_mod.run_block = spy
        try:
            res = integrate(m, x_in, t_start, t_end, caps, ctx, bridge=True, w_head=w)
        finally:
            dm_mod.run_block = orig
        assert seen == want, (seen, want)
        n_want = torch.tensor([sum(r == i for rs in want.values() for r in rs) for i in range(B)],
                              dtype=torch.float32)
        assert torch.equal(res.n_updates, n_want)


def test_integrate_from_0_to_1_is_the_ladder_and_recur_repeats_each_block():
    for recur in (0, 2):
        m = model(dm_cfg(n_blocks=4, recur=recur))
        wake_stream(m)
        m.eval()
        x, y, layout, _ = batch(B=2)
        _xh, caps, ctx = clean_pass(m, x, layout)
        calls = _spy_layers(m)
        L = x.shape[1]
        d_last, x_final = ladder(m, caps, ctx, (2, L, D), bridge=True, w_head=m.embed.lm_weight())
        k = m.dmorph.layers_per_block
        want = [gi for b in range(4) for _ in range(recur + 1) for gi in block_layers(b, k)]
        assert calls == want, (recur, calls, want)
        assert torch.allclose(x_final.norm(dim=-1), torch.ones(2, L), atol=1e-4)


def test_eval_terms_emit_the_held_time_reads_only_under_fpf():
    x, y, layout, _ = batch(B=2)
    m = model(dm_cfg(fpf_p=0.5))
    wake_stream(m)
    m.eval()
    _add, g = _train_terms(m, x, y, layout, want_eval=True)
    for k in ("dm_ladder_ce_r0", "dm_ladder_ce_r2", "dm_ladder_acc_r2", "dm_resid_r2",
              "dm_resid_auroc_r2"):
        assert k in g, k
    assert torch.equal(g["dm_ladder_ce_r0"], g["dm_ladder_ce"])    # recur 0 is the main read
    assert "dm_fpf_frac" not in g                                   # eval keeps the null carry
    m0 = model(dm_cfg(fpf_p=0.0))
    wake_stream(m0)
    m0.eval()
    _add0, g0 = _train_terms(m0, x, y, layout, want_eval=True)
    assert not any(k.startswith(("dm_ladder_ce_r", "dm_resid")) for k in g0)
    assert torch.equal(g0["dm_ladder_ce"], g["dm_ladder_ce"])      # W_s = 0: same ladder as v1


def test_residual_auroc_is_a_real_auroc():
    resid = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    correct = torch.tensor([True, True, True, False, False, False])
    assert float(residual_auroc(resid, correct)) == 1.0
    assert float(residual_auroc(resid, ~correct)) == 0.0
    mixed = torch.tensor([True, False, True, False, True, False])
    # pairs (c, w) with resid_c < resid_w: c=0:{3 w}, c=2:{2}, c=4:{1} → 6/9
    assert abs(float(residual_auroc(resid, mixed)) - 6.0 / 9.0) < 1e-6
    assert math.isnan(float(residual_auroc(resid, torch.ones(6, dtype=torch.bool))))
