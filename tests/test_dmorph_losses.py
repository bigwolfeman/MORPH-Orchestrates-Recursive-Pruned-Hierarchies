"""Acceptance criterion 4: the FM loss on a zero-velocity head equals the analytic null
floor, and ``D̂ = x_t + (1 - t)·v̂`` recovers ``y`` exactly when ``v̂ = v*``. Plus the
loss-group contract every dmorph forward emits.
"""

from __future__ import annotations

import math

import torch

from morph.model.dmorph import aggregate_eval, eval_weight_key
from morph.model.dmorph import (DmCtx, band_of_t, fm_euler_step, noisy_stream, targets,
                                training_terms)
from _dmorph_common import D, V, batch, clean_pass, dm_cfg, model


def test_velocity_is_exactly_zero_at_construction():
    m = model(dm_cfg(n_blocks=2))
    m.eval()
    x, y, layout, _ = batch(B=2)
    _xh, caps, ctx = clean_pass(m, x, layout)
    L = x.shape[1]
    t = torch.tensor([0.2, 0.9])
    with torch.no_grad():
        v = noisy_stream(m, torch.randn(2, L, D), t, band_of_t(t, 2), caps, ctx)
    assert torch.equal(v, torch.zeros_like(v))


def test_fm_loss_at_the_zero_head_is_the_null_floor():
    """``E‖y − x0‖² = E‖y‖² + d·s² = 2`` for unit-L2 targets and the matched source;
    with ``v̂ ≡ 0`` the raw FM loss is the sample mean of that quantity, so
    ``dm_fm_rel`` sits at 1 ± the Monte-Carlo error, at EVERY t (the CFM target is
    scale-uniform in t)."""
    m = model(dm_cfg(n_blocks=2))
    m.eval()
    assert m.dmorph.null_floor == 1.0 + D * (1.0 / math.sqrt(D)) ** 2 == 2.0
    x, y, layout, _ = batch(B=8, n=130, seed=2)
    vals = []
    for seed in range(4):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = m(x, labels=y, slot_layout=layout)
        vals.append(float(out["dm_fm_rel"]))
        assert torch.equal(out["dm_fm"], out["dm_fm_rel"])
    mean = sum(vals) / len(vals)
    assert abs(mean - 1.0) < 0.05, f"zero-head FM loss {mean:.4f} is not at the floor 1.0"


def test_d_hat_recovers_y_exactly_when_v_hat_is_v_star():
    g = torch.Generator().manual_seed(0)
    y = torch.nn.functional.normalize(torch.randn(3, 7, D, generator=g), dim=-1)
    x0 = torch.randn(3, 7, D, generator=g) / math.sqrt(D)
    for t_val in (0.0, 0.3, 0.7, 0.999):
        t = torch.full((3, 1, 1), t_val)
        x_t = (1 - t) * x0 + t * y
        v_star = y - x0
        d_hat = x_t + (1 - t) * v_star
        assert torch.allclose(d_hat, y, atol=1e-6)
        # And the Euler step written in D̂ is Euler on the velocity.
        t1 = torch.full((3,), t_val)
        tn = torch.full((3,), min(1.0, t_val + 0.25))
        x_next = fm_euler_step(x_t, d_hat, t1, tn)
        assert torch.allclose(x_next, x_t + (tn - t1).view(3, 1, 1) * v_star, atol=1e-5)


def test_targets_are_unit_l2_and_detached_and_masked():
    for arm in ("tok", "hs"):
        m = model(dm_cfg(arm=arm))
        m.eval()
        x, y, layout, _ = batch()
        xh, caps, ctx = clean_pass(m, x, layout)
        xh = xh.detach().requires_grad_(True)
        row_w, _p, _z = m._tul_half_weights(y, layout)
        yy, valid, ce_labels, ce_w = targets(m, xh, y, layout, row_w)
        assert not yy.requires_grad, f"{arm}: the target is live"
        n = yy[valid].norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-5)
        assert torch.equal(yy[~valid], torch.zeros_like(yy[~valid]))
        assert ((ce_labels == -100) | valid).all()
        if arm == "tok":
            assert valid.equal((y != -100) & (row_w.view(*y.shape) > 0))
            assert torch.equal(ce_w, row_w)
        else:
            real = layout.slot_mask & (layout.bag_id != layout.max_slots)
            assert valid.equal(real)
            assert (ce_labels != -100).sum() == (real & (y != -100)).sum()


def test_loss_groups_carry_the_terms_and_the_total_is_their_sum():
    m = model(dm_cfg(arm="tok", lambda_fm=0.5, lambda_ce=2.0))
    m.train()
    x, y, layout, _ = batch()
    torch.manual_seed(1)
    out = m(x, labels=y, slot_layout=layout)
    total = out["loss_tokens_only"] + 0.5 * out["dm_fm"] + 2.0 * out["dm_ce"]
    assert torch.allclose(out["loss"], total, atol=1e-6)
    assert torch.allclose(out["dm_fm_weighted"], 0.5 * out["dm_fm"])
    assert torch.allclose(out["dm_ce_weighted"], 2.0 * out["dm_ce"])
    assert "dm_ladder_ce" not in out, "the ladder is an eval-only instrument"
    out["loss"].backward()
    # AdaLN-Zero warm start (DiT): with W_v at zero the FM/CE terms reach W_v and
    # head_scale and NOTHING upstream of W_v — the gates, the t MLP and the block's
    # layers get exactly zero from the stream until W_v leaves zero.
    for p in (m.dmorph.W_v.weight, m.dmorph.head_scale):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0
    for p in (m.dmorph.v_gate.to_mod.weight, m.dmorph.cond.mlp[0].weight):
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0
    # Once W_v is off zero the whole stream is trainable.
    from _dmorph_common import wake_stream
    m2 = model(dm_cfg(arm="tok"))
    wake_stream(m2)
    m2.train()
    torch.manual_seed(1)
    m2(x, labels=y, slot_layout=layout)["loss"].backward()
    for p in (m2.dmorph.W_v.weight, m2.dmorph.head_scale, m2.dmorph.v_gate.to_mod.weight,
              m2.dmorph.cond.mlp[0].weight, m2.dmorph.v_norm.weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0


def test_loss_scale_none_uses_the_raw_fm_term():
    m = model(dm_cfg(loss_scale="none"))
    m.eval()
    x, y, layout, _ = batch()
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    assert torch.equal(out["dm_fm"], out["dm_fm_raw"])
    assert not torch.equal(out["dm_fm"], out["dm_fm_rel"])


def test_eval_aggregation_weights_by_counts_and_never_averages_an_empty_band_as_zero():
    """Two eval batches: batch 0 has no row in band 0 (its band-0 term is emitted as 0),
    batch 1 has ten. A per-batch mean would report 1.0; the count-weighted read is 2.0.
    A band with no row anywhere is NaN, not 0. Counts and the head scale stay plain."""
    acc = {
        "val/dm_fm_band0": [0.0, 2.0], "val/dm_n_band0": [0.0, 10.0],
        "val/dm_ce_band0": [0.0, 1.5], "val/dm_n_ce_band0": [0.0, 4.0],
        "val/dm_fm_band1": [0.0, 0.0], "val/dm_n_band1": [0.0, 0.0],
        "val/dm_fm": [1.0, 3.0], "val/dm_n_fm": [10.0, 30.0],
        "val/dm_cos": [0.2, 0.6], "val/dm_worth_cost_zero": [0.5, 1.5], "val/dm_n_ce": [1.0, 3.0],
        "val/dm_head_scale": [16.0, 18.0],
        "val/ce_tokens": [4.0, 5.0],
    }
    out = aggregate_eval(acc)
    assert out["val/dm_fm_band0"] == 2.0 and out["val/dm_ce_band0"] == 1.5
    assert out["val/dm_fm_band1"] != out["val/dm_fm_band1"]           # NaN
    assert out["val/dm_fm"] == 2.5 and out["val/dm_cos"] == 0.5
    assert out["val/dm_worth_cost_zero"] == 1.25
    assert out["val/dm_head_scale"] == 17.0 and out["val/dm_n_fm"] == 20.0
    assert "val/ce_tokens" not in out
    assert eval_weight_key("dm_ladder_ce") == "dm_n_ce" and eval_weight_key("dm_sigreg") == "dm_n_fm"
    assert eval_weight_key("dm_n_band2") is None
    import pytest
    with pytest.raises(KeyError, match="dm_n_ce"):
        aggregate_eval({"val/dm_ladder_ce": [1.0]})


def test_ce_through_d_hat_starts_near_ln_v_at_every_source_scale():
    """The first panel run (dmorph-tok-s1-5k, 2026-09-03) started its CE-through-D̂ term at
    41.7 nats — four times ln V — because the raw ``D̂`` (norm ``s·sqrt(d)`` at low t)
    was read through a ``sqrt(d)`` gain: confidently WRONG, not ignorant. The readout must
    depend on D̂'s DIRECTION only, so at construction the term can never exceed the
    uniform head's ln V by more than the calibration slack, for the matched source AND
    for ``source_std 1.0`` (the panel's reshaped source). It is NOT pinned near ln V from
    below: at t → 1 the input already carries the target and the tied head decodes it
    (the decodability trap the prereg's P6 is about), so on a 64-token vocabulary the
    init value is small. The raw readout is kept as the negative control: at source 1.0
    it must exceed ln V, which is the defect this test exists for."""
    from morph.model.dmorph import hard_bridge, readout_state, targets
    from morph.model.fused_ce import fused_linear_cross_entropy
    x, y, layout, _ = batch(B=3)
    for src in (1.0 / math.sqrt(D), 1.0):
        m = model(dm_cfg(n_blocks=2, source_std=src))
        m.train()
        torch.manual_seed(0)
        out = m(x, labels=y, slot_layout=layout)
        ce = float(out["dm_ce"])
        assert ce <= math.log(V) + 0.5, (src, ce, math.log(V))
    # negative control: the pre-fix readout (raw D̂ · head_scale) at source 1.0 is
    # overconfident-wrong — a random unit-ish direction blown up to logit gaps of tens.
    m = model(dm_cfg(n_blocks=2, source_std=1.0))
    m.eval()
    torch.manual_seed(1)
    _xh, _caps, ctx = clean_pass(m, x, layout)
    w_head = m.embed.lm_weight().detach().float()
    B, L = y.shape
    d_hat_raw = torch.randn(B, L, D)            # source-scale noise at s=1: norm ~sqrt(D)
    lab = y.clone(); lab[layout.slot_mask] = -100
    raw = fused_linear_cross_entropy((d_hat_raw * m.dmorph.head_scale).reshape(-1, D), w_head,
                                     lab.reshape(-1), ignore_index=-100, mask_token_id=4)
    assert float(raw) > math.log(V) + 1.0, f"negative control did not fire: {float(raw):.2f}"
    scale = torch.tensor(8.0)
    big = torch.randn(4, 5, D) * 50.0
    small = torch.randn(4, 5, D) * 0.01
    for st in (big, small):
        r = readout_state(st, scale)
        assert torch.allclose(r.norm(dim=-1), torch.full((4, 5), 8.0), atol=1e-4)
    w = torch.randn(V, D)
    rows = hard_bridge(big, w, scale, mask_id=4)
    assert torch.allclose(rows.norm(dim=-1), torch.ones(4, 5), atol=1e-5)
    assert torch.equal(rows, hard_bridge(big * 1e-3, w, scale, mask_id=4)), \
        "the bridge must be invariant to the input's norm"
