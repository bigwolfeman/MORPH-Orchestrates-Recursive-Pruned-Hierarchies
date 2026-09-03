"""Acceptance criterion 3: the band→block map is ONE function used by training, eval and
the generator, and ``t`` in band ``b`` runs exactly layers ``[k·b, k·(b+1))`` and nothing
else — asserted by a layer-call spy, not by reading the code.

The testbed lost a 573M-token run to two σ→block functions that disagreed
(``lab/dmorph/research/2026-09-03-db-testbed-fidelity.md``, "block reversal"). Here
``band_of_t`` and ``block_layers`` are the only road and every caller goes through
``noisy_stream``.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.dmorph import (band_bounds, band_of_t, block_layers, eval_phase,
                                noisy_stream, sample_blocks, sample_t_in_band,
                                stratified_t)
from _dmorph_common import D, batch, clean_pass, dm_cfg, model, wake_stream


class _Spy:
    """Records (global layer index, had cla_kv) for every block call."""

    def __init__(self, m):
        self.calls: list[tuple[int, bool]] = []
        n_pre = m.cfg.n_prelude
        for gi, layer in enumerate(list(m.prelude) + list(m.coda)):
            orig = layer.forward

            def wrapped(h, *a, _orig=orig, _gi=gi, **kw):
                ak = kw.get("attn_kwargs") or {}
                self.calls.append((_gi, "cla_kv" in ak))
                return _orig(h, *a, **kw)

            layer.forward = wrapped
        assert n_pre + m.cfg.n_coda == len(list(m.prelude) + list(m.coda))


@pytest.mark.parametrize("n_blocks", [2, 4])
def test_t_in_band_b_runs_exactly_that_blocks_layers(n_blocks):
    m = model(dm_cfg(n_blocks=n_blocks))
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=1)
    _xh, caps, ctx = clean_pass(m, x, layout)
    k = m.dmorph.layers_per_block
    assert k * n_blocks == m.cfg.n_prelude + m.cfg.n_coda
    spy = _Spy(m)
    L = x.shape[1]
    for b in range(n_blocks):
        lo, hi = band_bounds(b, n_blocks, 0.0)
        for t_val in (lo + 1e-4, 0.5 * (lo + hi), hi - 1e-4):
            t = torch.full((1,), t_val)
            band = band_of_t(t, n_blocks)
            assert int(band) == b
            spy.calls.clear()
            with torch.no_grad():
                noisy_stream(m, torch.randn(1, L, D), t, band, caps, ctx)
            noisy = [gi for gi, had in spy.calls if had]
            # The expectation is written out by hand, NOT through block_layers: an
            # off-by-one in the map must fail here, not be mirrored (mutation-checked).
            want = list(range(b * k, (b + 1) * k))
            assert want == list(block_layers(b, k))
            assert noisy == want, f"t={t_val:.4f} (band {b}) ran layers {noisy}, expected {want}"
            assert all(had for _, had in spy.calls), "a clean (capture-less) layer call leaked in"


def test_every_row_runs_its_own_block_and_each_block_runs_once_per_forward():
    """Per-row t: a batch whose rows sit in different bands runs each hit block ONCE, on
    its own rows — the 1.25× accounting of the design note rests on this."""
    n_blocks = 2
    m = model(dm_cfg(n_blocks=n_blocks))
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=3)
    _xh, caps, ctx = clean_pass(m, x, layout)
    k = m.dmorph.layers_per_block
    spy = _Spy(m)
    t = torch.tensor([0.1, 0.9, 0.2])
    band = band_of_t(t, n_blocks)
    L = x.shape[1]
    x_t = torch.randn(3, L, D)
    with torch.no_grad():
        v = noisy_stream(m, x_t, t, band, caps, ctx)
    noisy = [gi for gi, had in spy.calls if had]
    assert noisy == list(range(0, k)) + list(range(k, 2 * k))
    # Row 1 (band 1) must equal the single-row run of band 1 on the same x_t.
    spy.calls.clear()
    with torch.no_grad():
        v1 = noisy_stream(m, x_t[1:2], t[1:2], band[1:2], [
            {kk: vv[1:2] for kk, vv in c.items()} for c in caps], ctx.rows(torch.tensor([1])))
    assert torch.allclose(v[1], v1[0], atol=1e-5, rtol=1e-5)


def test_training_samples_block_first_then_t_inside_the_widened_band():
    n_blocks, gamma = 4, 0.1
    g = torch.Generator().manual_seed(0)
    band = sample_blocks((4000,), n_blocks, None, torch.device("cpu"), generator=g)
    t = sample_t_in_band(band, n_blocks, gamma, generator=g)
    for b in range(n_blocks):
        lo, hi = band_bounds(b, n_blocks, gamma)
        tb = t[band == b]
        assert tb.numel() > 0
        assert float(tb.min()) >= lo - 1e-6 and float(tb.max()) <= hi + 1e-6
        # Without overlap the sampled t maps back to its own block; with it, only the
        # γ margin may map to a neighbour.
        inner = (tb >= b / n_blocks) & (tb < (b + 1) / n_blocks)
        assert (band_of_t(tb[inner], n_blocks) == b).all()
    counts = torch.bincount(band, minlength=n_blocks).float() / band.numel()
    assert (counts - 0.25).abs().max() < 0.03, "uniform visit is not uniform"
    visit = (0.7, 0.1, 0.1, 0.1)
    band2 = sample_blocks((4000,), n_blocks, visit, torch.device("cpu"), generator=g)
    c2 = torch.bincount(band2, minlength=n_blocks).float() / band2.numel()
    assert abs(float(c2[0]) - 0.7) < 0.03


def test_eval_t_is_a_deterministic_stratified_grid_that_covers_every_band():
    t = stratified_t((4,), torch.device("cpu"))
    assert torch.equal(t, torch.tensor([0.125, 0.375, 0.625, 0.875]))
    assert band_of_t(t, 4).tolist() == [0, 1, 2, 3]
    assert band_of_t(torch.tensor([0.0, 0.999, 1.0]), 4).tolist() == [0, 3, 3]
    # A phase rotates the grid (mod 1) and keeps its spacing.
    assert torch.allclose(stratified_t((4,), torch.device("cpu"), 0.5),
                          torch.tensor([0.625, 0.875, 0.125, 0.375]))


def test_eval_phase_rotates_across_batches_so_the_eval_set_covers_every_band():
    """At batch 2 a fixed grid (0.25, 0.75) never visits bands 0 and 2 of four; the
    data-derived phase must visit all four over an eval set, and be a function of the
    rows alone (same rows -> same t; the determinism test rests on it)."""
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(4, dtype=torch.long)
    phases = set()
    for _ in range(40):
        ids = torch.randint(5, 64, (2, 96), generator=g)
        ph = eval_phase(ids)
        assert 0.0 <= ph < 1.0 and ph == eval_phase(ids.clone())
        phases.add(ph)
        counts += torch.bincount(band_of_t(stratified_t((2,), torch.device("cpu"), ph), 4),
                                 minlength=4)
    assert len(phases) > 30, "the phase does not vary across batches"
    assert (counts > 0).all(), f"a band got no eval row across 40 batches: {counts.tolist()}"
    assert torch.equal(band_of_t(stratified_t((2,), torch.device("cpu"), 0.0), 4),
                       torch.tensor([1, 3])), "the un-phased grid is the pinned one"


def test_eval_forward_is_deterministic_on_the_same_rows():
    m = model(dm_cfg(n_blocks=2))
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch()
    with torch.no_grad():
        a = m(x, labels=y, slot_layout=layout)
        b = m(x, labels=y, slot_layout=layout)
    for k in ("loss", "dm_fm", "dm_ce", "dm_ladder_ce", "dm_fm_band0", "dm_fm_band1"):
        assert torch.equal(a[k], b[k]), k
