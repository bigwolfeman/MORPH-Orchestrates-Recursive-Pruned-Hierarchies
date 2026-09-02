"""gamma slow-EMA on the ternary scale (tul cusp-vault fix, 2026-09-02).

The contract: beta=0 is bit-identical legacy behaviour; beta>0 makes the
forward PURE (reads a buffer, never mutates) with the buffer advanced only by
the explicit per-optimizer-step update; and a large one-step weight drift
re-thresholds gradually instead of mass-flipping codes.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from morph.model.ternary_qat import (
    TernarySTE, collect_scale_ema_stes, update_scale_emas,
)


def _ste(w, beta=0.0, group=0):
    return TernarySTE(threshold=0.5, weight_shape=tuple(w.shape), mode="symmetric",
                      group=group, scale_dtype="fp16", weight_init=w.detach(),
                      scale_ema_beta=beta)


def test_beta_zero_bit_identical():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    a, b = _ste(w, beta=0.0), _ste(w, beta=0.0)
    assert a._scale_ema is None
    assert torch.equal(a(w), b(w))


def test_init_matches_live_scale_bitwise():
    """At registration the EMA buffer equals mean|W|, so the first forward is
    bit-identical to the live-scale forward."""
    torch.manual_seed(1)
    w = torch.randn(64, 32)
    live, ema = _ste(w, beta=0.0), _ste(w, beta=0.99)
    assert torch.equal(live(w), ema(w))


def test_forward_is_pure():
    torch.manual_seed(2)
    w = torch.randn(64, 32)
    ste = _ste(w, beta=0.99)
    buf0 = ste._scale_ema.clone()
    o1 = ste(w)
    o2 = ste(w * 3.0)          # even a drifted weight must not move the buffer
    assert torch.equal(ste._scale_ema, buf0), "forward mutated the EMA buffer"
    o3 = ste(w)
    assert torch.equal(o1, o3)
    assert not torch.equal(o1, o2)


def test_update_math():
    torch.manual_seed(3)
    w = torch.randn(64, 32)
    ste = _ste(w, beta=0.9)
    g0 = ste._scale_ema.clone()
    w2 = w * 2.0
    ste.update_scale_ema(w2)
    expect = 0.9 * g0 + 0.1 * (2.0 * g0)   # mean|2w| = 2*mean|w| exactly
    assert torch.allclose(ste._scale_ema, expect, rtol=1e-6)


def test_vault_suppression():
    """THE property the fix exists for: flip CONTAGION via the shared scale.
    A large drift in a small subset of coords yanks the live gamma=mean|W|,
    re-thresholding the UNTOUCHED majority in the same step (the mass vault).
    Under the EMA, gamma barely moves, so untouched coords keep their codes."""
    torch.manual_seed(4)
    w = torch.randn(256, 128)
    def codes(x, scale):
        wn = x / scale
        return torch.sign(wn) * (wn.abs() > 0.5)
    live0 = codes(w, w.abs().mean())
    drift = w.clone()
    hot = torch.rand_like(w) < 0.05
    drift[hot] *= 10.0                     # 5% of coords blow up 10x in one step
    untouched = ~hot
    flips_live = (codes(drift, drift.abs().mean()) != live0)[untouched].float().mean()
    ste = _ste(w, beta=0.99)
    ste.update_scale_ema(drift)            # one post-step EMA advance
    g = ste._scale_ema.reshape([])
    flips_ema = (codes(drift, g) != live0)[untouched].float().mean()
    assert flips_live > 0.02, f"scenario failed to produce a vault ({flips_live})"
    assert flips_ema < flips_live * 0.2, (flips_ema.item(), flips_live.item())


def test_grouped_fp16_path():
    torch.manual_seed(5)
    w = torch.randn(64, 32)
    live, ema = _ste(w, beta=0.0, group=16), _ste(w, beta=0.95, group=16)
    assert torch.equal(live(w), ema(w))    # init parity on the vectorized path
    ema.update_scale_ema(w * 1.5)
    assert not torch.equal(live(w * 1.5), ema(w * 1.5))


def test_grouped_non_fp16_raises():
    w = torch.randn(64, 32)
    with pytest.raises(AssertionError, match="fp16"):
        TernarySTE(threshold=0.5, weight_shape=(64, 32), mode="symmetric",
                   group=16, scale_dtype="int8", weight_init=w, scale_ema_beta=0.9)


def test_collect_and_update_helpers():
    torch.manual_seed(6)
    lin = nn.Linear(32, 64, bias=False)
    ste = _ste(lin.weight, beta=0.9)
    parametrize.register_parametrization(lin, "weight", ste)
    model = nn.Sequential(lin)
    pairs = collect_scale_ema_stes(model)
    assert len(pairs) == 1
    g0 = pairs[0][0]._scale_ema.clone()
    with torch.no_grad():
        lin.parametrizations.weight.original.mul_(2.0)
    update_scale_emas(pairs)
    assert not torch.equal(pairs[0][0]._scale_ema, g0)
    # off-model: no EMA STEs -> empty, hook is a no-op
    lin2 = nn.Linear(8, 8, bias=False)
    parametrize.register_parametrization(lin2, "weight", _ste(lin2.weight, beta=0.0))
    assert collect_scale_ema_stes(nn.Sequential(lin2)) == []
