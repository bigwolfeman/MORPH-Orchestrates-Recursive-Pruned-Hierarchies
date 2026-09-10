"""What the MUX head reads off the Hyper-Connection carrier (``tul.mux_readout``).

The slot-geometry audit's finding F2
(``lab/experiments/results/2026-09-10-slot-geometry-audit/README.md``): the MUX head and
``TULSlots.unpack`` read ``h.mean(dim=2)`` over the four HC streams, and on
``slot-unpack-free`` the loop's UPDATE survives that mean at 0.139 of its per-stream norm
while the ENTRY survives at 0.972. ``TULSlots.prefix_project``, the reader that feeds the
coda, does not take the mean at all: it projects every stream separately.

``mux_readout: "full"`` moves the mean to the other side of the readout's RMSNorm: the
readout runs PER STREAM and the results are averaged, which is exactly the tied head
applied to each stream with the logits averaged (everything between the streams and the
head is linear, so the two are the same object and the cheap one is used).

One test per contract:

  1. ``"mean"`` is the forward from before the knob existed. Checked against the PRE-CHANGE
     tree, not by reading the diff: the tiny model at seed 3/7 with
     ``mux_stage_own_iters=2`` gives loss ``9.359314918518066`` and sha256
     ``189911591a8f60ee2f9dee7932461bca63b2bf5433d2355270763fe0461aefc0`` over all 208
     gradient tensors both at master ``cffe3bb`` and with the knob at its default after the
     change.
  2. ``"full"`` IS the per-stream readout, averaged — recomputed by hand, to 1e-6.
  3. fixture sensitivity BOTH ways: it changes the MUX loss when the streams differ, and it
     equals ``"mean"`` when all four streams are identical.
  4. the stream SUM would not be a different arm — recorded as a test because it is the
     reason the per-stream form was chosen.
  5. the refusals: an unknown value, and a carrier with no stream axis.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _tul                      # noqa: E402 (tests/ on sys.path)
from test_tul_mux_every_pass import _grads, _model, _run, _same   # noqa: E402


def _mux(readout: str = "mean", **kw):
    return _model(tul_mux_readout=readout, slot_gain_lambda=0.0, dropout=0.0, **kw)


def _slots(m):
    """The looped slot carrier ``[B, S, n, C]`` of one forward, off the real path."""
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(7)
    xx, x0, bg = m._tul_front(x, layout)
    h = m._tul_core(xx, x0, bg, layout)[1]
    assert h.dim() == 4 and h.shape[2] == m._n_streams
    return h, x, layout


# ── 1. "mean" is the forward from before the knob ────────────────────────────

def test_mean_is_bit_identical_loss_and_every_gradient():
    m_a = _model(tul_mux_stage_own_iters=2)                     # the default
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    rng_a = torch.get_rng_state()

    m_b = _model(tul_mux_stage_own_iters=2, tul_mux_readout="mean")
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()

    assert torch.equal(rng_a, torch.get_rng_state()), "the mean path consumed random numbers"
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert _same(_grads(m_a), _grads(m_b)), "the mean path changed a gradient"


# ── 2. "full" is the per-stream readout, averaged ────────────────────────────

def test_full_is_the_per_stream_readout_averaged():
    m = _mux("full")
    h, x, layout = _slots(m)
    by_hand = m.final_norm(m.lm_mixer(h)).mean(dim=2)
    assert torch.allclose(m._readout_per_stream(h), by_hand, atol=1e-6)
    # and it is NOT the shipped readout, which means first
    assert not torch.allclose(m._readout(h), by_hand, atol=1e-4)


def test_the_mux_loss_uses_it():
    """The knob has to reach the loss, not only the helper.

    Measured while writing this test, and worth stating because it bounds the arm: on the
    UNTRAINED tiny model the two losses differ by about 3e-6 nats, because the HC streams
    start almost equal (``_readout``'s own docstring: at init the mean readout exactly
    recovers the plain-residual output). The knob can only act once the streams diverge,
    which they do on a trained checkpoint — the audit reads stream-mean survival 0.581 for
    the entry and 0.577 for the update on `slot-mux-norm-match` at step 5,000, and
    0.972 / 0.139 on `slot-unpack-free`. So the strong assertion below uses a carrier whose
    streams differ by construction, and the raw carrier only has to show the branch is live
    at all."""
    m_mean, m_full = _mux("mean"), _mux("full")
    h, x, layout = _slots(m_full)
    with torch.no_grad():
        # `_model` builds both at the same seed, so the weights are the same weights
        assert torch.equal(m_mean.tul.E_slot, m_full.tul.E_slot)
    assert not torch.equal(m_mean._tul_mux_loss(h, x, layout),
                           m_full._tul_mux_loss(h, x, layout))

    scale = torch.tensor([0.25, 1.0, 2.0, 4.0]).view(1, 1, -1, 1)[:, :, :h.shape[2]]
    hs = (h * scale.to(h.dtype)).contiguous()      # streams of very different magnitude
    l_mean = m_mean._tul_mux_loss(hs, x, layout)
    l_full = m_full._tul_mux_loss(hs, x, layout)
    assert abs(float(l_mean.detach()) - float(l_full.detach())) > 1e-3, \
        (float(l_mean.detach()), float(l_full.detach()))


# ── 3. fixture sensitivity, both ways ────────────────────────────────────────

def test_identical_streams_make_the_two_readouts_agree():
    """The other half of the sensitivity check. With every stream equal there is nothing
    for a per-stream normalisation to reweight, so `full` MUST collapse onto `mean`. A
    difference here would mean the branch changes something other than where the mean
    sits."""
    m = _mux("full")
    h, x, layout = _slots(m)
    flat = h[:, :, :1].expand_as(h).contiguous()          # all four streams identical
    assert torch.allclose(m._readout_per_stream(flat), m._readout(flat), atol=1e-6)
    assert torch.allclose(m._tul_mux_loss(flat, x, layout),
                          _mux("mean")._tul_mux_loss(flat, x, layout), atol=1e-6)


def test_the_loss_and_the_gradients_move_on_a_real_forward():
    m_a = _mux("mean", tul_mux_stage_own_iters=2)
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    m_b = _mux("full", tul_mux_stage_own_iters=2)
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()
    assert not torch.equal(out_a["mux_local"], out_b["mux_local"])
    assert not _same(_grads(m_a), _grads(m_b))


# ── 4. why per-stream and not the stream sum ─────────────────────────────────

def test_the_stream_sum_would_not_be_a_different_arm():
    """`final_norm` is an RMSNorm (scale-invariant) and `lm_mixer` is linear, so reading the
    stream SUM through `_readout` gives exactly what reading the stream MEAN gives. That is
    why the "head on each stream, logits averaged" form is the one implemented: the sum
    could not have been a second option."""
    m = _mux("mean")
    h, _, _ = _slots(m)
    n = h.shape[2]
    mean_read = m._readout(h)
    sum_read = m.final_norm(m.lm_mixer(h.sum(dim=2) if h.dim() == 4 else h))
    assert torch.allclose(mean_read, sum_read, atol=1e-5), \
        (float((mean_read - sum_read).abs().max()), n)


# ── 5. refusals ──────────────────────────────────────────────────────────────

def test_an_unknown_value_is_refused():
    with pytest.raises(ValueError, match="mux_readout"):
        _tul(mux_readout="streams", mux_beta=1.0)


def test_a_carrier_with_no_stream_axis_is_refused():
    """The reachable half of the "needs a stream axis" contract. The config-level guard in
    `MORPHTransformer.__init__` cannot fire on this tree — `_is_hc` is set unconditionally
    True because HC-Cayley is the sole residual — so the guard that has teeth is the dim
    check inside the readout, and this test states both facts."""
    m = _mux("full")
    assert m._is_hc, "the tree grew a second residual mode; the config guard is now live"
    h, _, _ = _slots(m)
    with pytest.raises(RuntimeError, match="Hyper-Connection carrier"):
        m._readout_per_stream(h.mean(dim=2))
