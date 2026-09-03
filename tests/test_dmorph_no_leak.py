"""Acceptance criterion 2: the noisy stream cannot see its own target, and the clean
head cannot see the noisy stream. Ported from the testbed's ``tests/test_no_leak.py``
(perturbation proof with a POSITIVE CONTROL — a probe with no control can pass because
it is blind).

Noisy query ``i`` predicts ``label_i = input_ids[i+1]``. Its attention reads the CLEAN
stream's keys and values: rows ``j <= i`` in the window (the cross-stream read keeps
key ``i`` — XSA is a clean-path property), compressed blocks that end strictly before
``i`` (``_compressed_causal_mask``), and the clean current token through the shared
x0 / bigram / value-embed injections once their scales leave zero. So
changing clean token ``p`` must move the noisy stream at ``p-1`` by EXACTLY zero, and
must move it at ``p`` (reachability). The control re-runs the same probe against a
K/V bundle rolled two positions toward the past, which lets query ``i`` reach clean
``i+1``; the probe must fire.
"""

from __future__ import annotations

import torch

import morph.model.dmorph as dmorph_mod
from morph.model.dmorph import band_of_t, noisy_stream
from _dmorph_common import D, V, batch, clean_pass, dm_cfg, model, wake_stream


def _stream_at(m, x, layout, x_t, t):
    xh, caps, ctx = clean_pass(m, x, layout)
    band = band_of_t(t, m.dmorph.cfg.n_blocks)
    with torch.no_grad():
        return noisy_stream(m, x_t, t, band, caps, ctx), xh


def _movement(m, x, layout, x_t, t, p):
    base, _ = _stream_at(m, x, layout, x_t, t)
    x2 = x.clone()
    x2[:, p] = 5 + (x2[:, p] - 5 + 13) % (V - 5)
    x2[layout.slot_mask] = x[layout.slot_mask]         # never rewrite a slot position
    if torch.equal(x2, x):
        return None
    pert, _ = _stream_at(m, x2, layout, x_t, t)
    return (pert - base).abs().amax(dim=-1)[0]         # [L]


def _setup(n_blocks=2, seed=3):
    m = model(dm_cfg(n_blocks=n_blocks), retention_gate_init=0.0)
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=1, seed=seed)
    L = x.shape[1]
    g = torch.Generator().manual_seed(11)
    x_t = torch.randn(1, L, D, generator=g)
    return m, x, layout, x_t


def test_noisy_query_never_sees_its_target_and_reaches_its_context():
    m, x, layout, x_t = _setup()
    for t_val in (0.1, 0.6, 0.95):
        t = torch.full((1,), t_val)
        for p in range(1, x.shape[1]):
            if bool(layout.slot_mask[0, p]):
                continue
            mv = _movement(m, x, layout, x_t, t, p)
            if mv is None:
                continue
            assert float(mv[:p].max()) == 0.0, (
                f"LEAK at t={t_val}: clean token {p} moved a noisy query before it "
                f"(max {float(mv[:p].max()):.3e} at {int(mv[:p].argmax())})")
            assert float(mv[p]) > 1e-6, f"noisy query {p} cannot reach clean context {p}"


def test_positive_control_a_rolled_kv_is_detected(monkeypatch):
    """The same probe against a bundle whose keys are shifted two positions toward the
    past MUST fire, or the test above proves nothing."""
    m, x, layout, x_t = _setup()
    orig = dmorph_mod._select_kv

    def rolled(kv, idx, detach):
        out = orig(kv, idx, detach)
        for k in ("k", "v"):
            out[k] = torch.roll(out[k], shifts=-2, dims=2)
        out["k_lat"] = torch.roll(out["k_lat"], shifts=-2, dims=1)
        return out

    monkeypatch.setattr(dmorph_mod, "_select_kv", rolled)
    t = torch.full((1,), 0.6)
    fired = False
    for p in range(2, x.shape[1]):
        if bool(layout.slot_mask[0, p]):
            continue
        mv = _movement(m, x, layout, x_t, t, p)
        if mv is not None and float(mv[:p].max()) > 0.0:
            fired = True
            break
    assert fired, "the leak probe failed to detect a deliberately leaky K/V bundle"


def test_the_clean_head_never_sees_the_noisy_stream():
    """Perturbing labels (hence the targets, x_t, and the whole noisy stream) moves no
    clean-head number; and the clean CE's graph contains no dmorph parameter."""
    m = model(dm_cfg(arm="tok"))
    wake_stream(m)
    x, y, layout, _ = batch()
    y2 = y.clone()
    tok = (y2 != -100) & (~layout.slot_mask)
    y2[tok] = 5 + (y2[tok] - 5 + 7) % (V - 5)
    m.eval()
    with torch.no_grad():
        a = m(x, labels=y, slot_layout=layout)
        b = m(x, labels=y2, slot_layout=layout)
        la = m.dmorph_infer(x, layout)["logits"]
    assert not torch.equal(a["dm_fm"], b["dm_fm"]), "the probe is blind: targets did not move"
    # The clean logits are a function of x and the layout alone.
    with torch.no_grad():
        lb = m.dmorph_infer(x, layout)["logits"]
    assert torch.equal(la, lb)
    m.train()
    out = m(x, labels=y, slot_layout=layout)
    dm_params = [p for n, p in m.named_parameters() if n.startswith("dmorph.")]
    grads = torch.autograd.grad(out["loss_tokens_only"], dm_params, allow_unused=True)
    assert all(g is None for g in grads), "a dmorph parameter is in the clean CE's graph"


def test_x_t_reaches_later_noisy_positions_only_through_causal_channels():
    """The noisy stream has NO noisy -> noisy attention (the named deviation), but it is
    not position-isolated either: CCA's causal conv on the query (kernel 4) and the GLA
    scan are noisy -> noisy channels, both strictly causal. So perturbing ``x_t`` at
    ``q`` moves the stream at ``q`` and possibly at LATER positions (legal: a past noisy
    position carries a past token, ``y_j = E[ids_{j+1}]`` with ``j < i``), and NEVER at
    an earlier one. Recorded in the design note's Implementation record.
    """
    for retention in (False, True):
        m = model(dm_cfg(n_blocks=2), retention=retention, retention_gate_init=0.0)
        wake_stream(m)
        m.eval()
        x, y, layout, _ = batch(B=1, seed=4)
        L = x.shape[1]
        g = torch.Generator().manual_seed(2)
        x_t = torch.randn(1, L, D, generator=g)
        t = torch.full((1,), 0.4)
        base, _ = _stream_at(m, x, layout, x_t, t)
        for q in (0, 5, L - 1):
            x2 = x_t.clone()
            x2[:, q] += 3.0
            pert, _ = _stream_at(m, x, layout, x2, t)
            mv = (pert - base).abs().amax(dim=-1)[0]
            assert float(mv[q]) > 1e-6, q
            assert float(mv[:q].max() if q > 0 else 0.0) == 0.0, f"x_t at {q} moved an EARLIER position"
            reach = 2 * (m.cfg.conv_kernel - 1) * m.dmorph.layers_per_block
            if not retention and q + reach + 2 < L:
                # No GLA: the only noisy -> noisy channel is CCA's causal conv on q — two
                # stacked kernel-4 convs per layer (depthwise, then grouped), so the reach
                # is 2·(k−1) per layer; nothing beyond it can move.
                far = mv[q + reach + 1:]
                assert float(far.max()) == 0.0, f"x_t at {q} moved a position beyond the conv reach"
                assert float(mv[q + 1: q + reach + 1].max()) > 0.0, "the conv channel is dead"
