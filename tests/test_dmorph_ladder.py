"""Acceptance criterion 5: the hard bridge returns embedding rows of full norm, and the
ladder with ``B`` steps calls each block exactly once, in order.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from morph.model.dmorph import block_layers, hard_bridge, ladder
from _dmorph_common import D, V, batch, clean_pass, dm_cfg, model, wake_stream


def test_hard_bridge_returns_unit_rows_of_the_argmax_token():
    m = model(dm_cfg())
    w = m.embed.lm_weight().detach()
    g = torch.Generator().manual_seed(0)
    d_hat = torch.randn(2, 9, D, generator=g)
    out = hard_bridge(d_hat, w, m.dmorph.head_scale, mask_id=4, chunk=5)
    assert out.shape == d_hat.shape
    n = out.norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)
    logits = (d_hat * m.dmorph.head_scale) @ w.t()
    logits[..., 4] = float("-inf")
    idx = logits.argmax(-1)
    assert torch.allclose(out, F.normalize(w[idx], dim=-1), atol=1e-5)
    # The masked structural id is never chosen even when it would win.
    w2 = w.clone()
    w2[4] = 100.0 * d_hat[0, 0] / d_hat[0, 0].norm()
    out2 = hard_bridge(d_hat[:1, :1], w2, m.dmorph.head_scale, mask_id=4)
    assert not torch.allclose(out2[0, 0], F.normalize(w2[4], dim=-1))


def test_ladder_calls_each_block_once_in_order_and_bridges_between_steps():
    for n_blocks in (2, 4):
        m = model(dm_cfg(n_blocks=n_blocks))
        wake_stream(m)
        m.eval()
        x, y, layout, _ = batch(B=2)
        _xh, caps, ctx = clean_pass(m, x, layout)
        k = m.dmorph.layers_per_block
        calls: list[int] = []
        for gi, layer in enumerate(list(m.prelude) + list(m.coda)):
            orig = layer.forward

            def wrapped(h, *a, _orig=orig, _gi=gi, **kw):
                if "cla_kv" in (kw.get("attn_kwargs") or {}):
                    calls.append(_gi)
                return _orig(h, *a, **kw)

            layer.forward = wrapped
        w = m.embed.lm_weight()
        L = x.shape[1]
        d_last, x_final = ladder(m, caps, ctx, (2, L, D), bridge=True, w_head=w)
        want = list(range(m.cfg.n_prelude + m.cfg.n_coda))    # every layer once, in order
        assert want == [gi for b in range(n_blocks) for gi in block_layers(b, k)]
        assert calls == want, f"ladder ran layers {calls}, expected {want}"
        # With the hard bridge the final state is a unit embedding row.
        n = x_final.norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-4)
        # The read-out is the last block's UNBRIDGED estimate, not the one-hot row.
        assert not torch.allclose(d_last, x_final)
        # Same rows, same noise seed → the same ladder.
        d2, x2 = ladder(m, caps, ctx, (2, L, D), bridge=True, w_head=w)
        assert torch.equal(d2, d_last) and torch.equal(x2, x_final)


def test_ladder_without_bridge_ends_at_d_hat():
    m = model(dm_cfg(n_blocks=2, arm="hs"))
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=2)
    _xh, caps, ctx = clean_pass(m, x, layout)
    w = m.embed.lm_weight()
    L = x.shape[1]
    d_last, x_final = ladder(m, caps, ctx, (2, L, D), bridge=False, w_head=w)
    assert torch.allclose(d_last, x_final, atol=1e-5)


def test_dmorph_infer_masks_the_slot_id_on_both_heads():
    m = model(dm_cfg(n_blocks=2))
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=1)
    r = m.dmorph_infer(x, layout)
    assert r["logits"].shape == r["ladder_logits"].shape == (1, x.shape[1], V)
    assert torch.isinf(r["logits"][..., 4]).all() and torch.isinf(r["ladder_logits"][..., 4]).all()
    assert (r["logits"].argmax(-1) != 4).all() and (r["ladder_logits"].argmax(-1) != 4).all()
