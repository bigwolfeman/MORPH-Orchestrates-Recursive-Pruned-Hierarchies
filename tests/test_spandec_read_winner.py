"""``tul.spandec_read`` (arm "spdwin", 2026-10-10): the span decoder grades the latent-selected
loop's WINNER cell, the one cell the coda reads, instead of the mean of the M exit cells.

Files: morph/model/tul.py (the key, ``_check_spandec_read``), morph/model/transformer.py
(``_spandec_read_winner``, ``_fan_winner_cell``, the ``_tul_spandec_loss`` call site in
``_forward_tul``), morph/training/tul_setup.py (``KNOWN_TUL_KEYS``, Hydra path, manifest),
morph/configs/lxtul_pointer_fast_spdwin.yaml.

What each test pins:
  a. ``"mean"`` (explicit) is bit-identical to the key being absent: loss and every grad.
  b. Under ``"winner"`` the decoder's input is ``_readout`` of the winner cell exactly, and
     the winner index is the one ``_fan_route_cells`` uses (train AND eval).
  c. Under ``"winner"`` the span-decoder loss alone sends ZERO gradient to the loser cells
     at the exit and a non-zero one to the winner. Under ``"mean"`` the same probe shows
     gradient on every cell, so the test fails if the code still reads the mean.
  d. Refusals.

CPU, fp32, the ``tests/test_tul_fan_lsel.py`` strict LXTUL fixtures (``M = 4`` fan cells).
"""
from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHTransformer
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M

M = int(M)


def _spy(m: MORPHTransformer) -> list:
    """Record every ``_tul_spandec_loss`` call: (z input, cells, returned loss)."""
    log: list = []
    orig = m._tul_spandec_loss

    def wrapped(h_slots, input_ids, layout, stats=None, cells=None):
        out = orig(h_slots, input_ids, layout, stats=stats, cells=cells)
        log.append({"h": h_slots, "cells": cells, "loss": out})
        return out

    m._tul_spandec_loss = wrapped
    return log


def _spy_route(monkeypatch) -> list:
    log: list = []
    orig = MORPHTransformer._fan_route_cells

    def wrapped(cells, route):
        log.append({"winner": route["winner"].detach().clone()})
        return orig(cells, route)

    monkeypatch.setattr(MORPHTransformer, "_fan_route_cells", staticmethod(wrapped))
    return log


def _train_fwd(m, seed=7):
    _ids, inp, lab, layout = _batch(M)
    m.train()
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout)


# ── a. mean is the tree before the key ──────────────────────────────────────────────


def test_mean_is_bit_identical_to_the_key_absent():
    a = _lsel("joint")
    b = _lsel("joint", spandec_read="mean")
    assert not a._spandec_read_winner and not b._spandec_read_winner
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    oa, ob = _train_fwd(a), _train_fwd(b)
    assert torch.equal(oa["loss"], ob["loss"])
    oa["loss"].backward()
    ob["loss"].backward()
    n = 0
    for (na, pa), (_nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None:
            assert pb.grad is None, na
            continue
        assert torch.equal(pa.grad, pb.grad), na
        n += 1
    assert n > 0


# ── b. the decoder reads the coda's cell ────────────────────────────────────────────


@pytest.mark.parametrize("train", [True, False])
def test_decoder_input_is_the_winner_cell_readout(monkeypatch, train):
    m = _lsel("joint", spandec_read="winner")
    assert m._spandec_read_winner
    sd, route = _spy(m), _spy_route(monkeypatch)
    _ids, inp, lab, layout = _batch(M)
    torch.manual_seed(3)
    if train:
        m.train()
        m(inp, labels=lab, slot_layout=layout)
    else:
        m.eval()
        with torch.no_grad():
            m(inp, labels=lab, slot_layout=layout)
    assert len(sd) == 1 and len(route) >= 1
    cells, h = sd[0]["cells"], sd[0]["h"]
    winner = route[0]["winner"]
    B, S = winner.shape
    assert cells.shape[:3] == (B, S, M)
    assert h.shape == (B, S, *cells.shape[3:])
    # the winner is non-trivial: not every slot picked the same cell
    assert len(set(winner.flatten().tolist())) > 1, "degenerate winner; fixture proves nothing"
    expect = torch.stack([cells[b, s, int(winner[b, s])]
                          for b in range(B) for s in range(S)]).reshape(B, S, *cells.shape[3:])
    assert torch.equal(h, expect)
    z_expect = m._readout(expect)
    z_mean = m._readout(cells.mean(dim=2))
    assert torch.equal(m._readout(h), z_expect)
    assert not torch.equal(z_expect, z_mean)


# ── c. gradient reaches the winner cell only ────────────────────────────────────────


def _decoder_grad_per_cell(m, monkeypatch):
    sd, route = _spy(m), _spy_route(monkeypatch)
    _train_fwd(m)
    cells, loss = sd[0]["cells"], sd[0]["loss"]
    (g,) = torch.autograd.grad(loss, cells, retain_graph=True, allow_unused=True)
    assert g is not None
    return g, route[0]["winner"]


def test_decoder_loss_alone_trains_the_winner_cell_only(monkeypatch):
    m = _lsel("joint", spandec_read="winner")
    g, winner = _decoder_grad_per_cell(m, monkeypatch)
    B, S = winner.shape
    gn = g.flatten(3).abs().sum(-1) if g.dim() > 3 else g.abs()      # [B, S, M]
    onehot = torch.nn.functional.one_hot(winner, M).bool()
    assert float(gn[~onehot].abs().max()) == 0.0
    assert float(gn[onehot].min()) >= 0.0
    assert float(gn[onehot].sum()) > 0.0


def test_mean_reader_trains_every_cell(monkeypatch):
    """The control that makes the test above bite: under "mean" the losers DO get decoder
    gradient, so a "winner" build that still read the mean would fail the zero check."""
    m = _lsel("joint")
    g, winner = _decoder_grad_per_cell(m, monkeypatch)
    gn = g.flatten(3).abs().sum(-1) if g.dim() > 3 else g.abs()
    onehot = torch.nn.functional.one_hot(winner, M).bool()
    assert float(gn[~onehot].abs().max()) > 0.0


# ── d. refusals ─────────────────────────────────────────────────────────────────────


def test_refusals():
    with pytest.raises(ValueError, match="'mean', 'winner', 'each' or 'best'"):
        _lsel("joint", spandec_read="worst")
    with pytest.raises(ValueError, match="spandec: true"):
        _lsel("joint", spandec=False, spandec_read="winner")
    with pytest.raises(ValueError, match="fan_loop_select != 'off'"):
        _lsel("off", spandec_read="winner")
    with pytest.raises(ValueError, match="fan_lsel_read == 'winner'"):
        _lsel("joint", fan_lsel_read="all", spandec_read="winner")
    with pytest.raises(ValueError, match="spandec_reads_cells"):
        _lsel("joint", spandec_reads_cells=True, spandec_read="winner")
