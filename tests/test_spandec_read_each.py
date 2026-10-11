"""``tul.spandec_read: each`` (2026-10-10): the span decoder grades EACH of the M exit cells
alone (cell axis folded into the batch axis, one pass over B*M rows); the loss is the mean of
the M per-cell losses.

Files: morph/model/transformer.py (``_spandec_read_each``, ``fold_cells`` in
``_tul_spandec_loss``, the call site in ``_forward_tul``), morph/model/tul.py
(``_check_spandec_read``), morph/configs/lxtul_pointer_fast_spdeach.yaml.

What each test pins:
  a. The loss equals the mean over i of the loss of the decoder run on cell i ALONE.
  b. The decoder loss alone sends cell i exactly (1/M) x the gradient of cell i's own loss,
     and the cells' gradients differ (under ``mean`` every cell gets the identical cotangent,
     so this fails if the code still reads the mean).
  c. The row cap scales with the folded batch (CE with ``spandec_ce_row_cap`` on equals off).
  d. Refusals. ``mean`` bit-identity stays pinned in test_spandec_read_winner.py.

CPU, fp32, the ``tests/test_tul_fan_lsel.py`` strict LXTUL fixtures (``M = 4`` fan cells).
"""
from __future__ import annotations

import pathlib

import pytest
import torch

from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M

M = int(M)


def _capture(m) -> dict:
    """Record the args of the one ``_tul_spandec_loss`` call and its returned loss."""
    cap: dict = {}
    orig = m._tul_spandec_loss

    def wrapped(h_slots, input_ids, layout, stats=None, cells=None, **kw):
        out = orig(h_slots, input_ids, layout, stats=stats, cells=cells, **kw)
        cap.update(h=h_slots, ids=input_ids, layout=layout, cells=cells, loss=out, kw=kw,
                   stats=stats)
        return out

    m._tul_spandec_loss = wrapped
    cap["orig"] = orig
    return cap


def _fwd(m, seed=7):
    _ids, inp, lab, layout = _batch(M)
    m.train()
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout)


def _own_loss(cap, i):
    """The decoder run on cell i ALONE through the unfolded path (the hand reference)."""
    cells = cap["cells"]
    return cap["orig"](cells[:, :, i], cap["ids"], cap["layout"], cells=cells)


def test_each_flag_is_set_only_by_each():
    assert _lsel("joint", spandec_read="each")._spandec_read_each
    assert not _lsel("joint", spandec_read="winner")._spandec_read_each
    assert not _lsel("joint")._spandec_read_each


def test_loss_is_the_mean_of_the_per_cell_losses():
    m = _lsel("joint", spandec_read="each")
    cap = _capture(m)
    _fwd(m)
    assert cap["kw"]["fold_cells"] is True
    assert cap["cells"].shape[2] == M
    with torch.no_grad():
        per_cell = torch.stack([_own_loss(cap, i) for i in range(M)])
    assert float(per_cell.max() - per_cell.min()) > 0.0, "cells identical; proves nothing"
    torch.testing.assert_close(cap["loss"].detach(), per_cell.mean(), rtol=1e-5, atol=1e-6)
    # and it is NOT the mean-reader's loss on the mean cell
    mean_loss = cap["orig"](cap["cells"].mean(dim=2), cap["ids"], cap["layout"],
                            cells=cap["cells"])
    assert not torch.allclose(cap["loss"].detach(), mean_loss.detach(), rtol=1e-6, atol=0)


def test_every_cell_gets_one_over_m_of_its_own_gradient():
    m = _lsel("joint", spandec_read="each")
    cap = _capture(m)
    _fwd(m)
    cells = cap["cells"]
    (g,) = torch.autograd.grad(cap["loss"], cells, retain_graph=True, allow_unused=True)
    assert g is not None
    gs = []
    for i in range(M):
        (gi,) = torch.autograd.grad(_own_loss(cap, i), cells, retain_graph=True)
        ref = gi[:, :, i] / M
        assert float(ref.abs().sum()) > 0.0
        torch.testing.assert_close(g[:, :, i], ref, rtol=1e-4, atol=1e-7)
        gs.append(g[:, :, i])
    # under the mean reader every cell gets the identical cotangent; here they differ
    for i in range(1, M):
        assert not torch.allclose(gs[0], gs[i], rtol=1e-3, atol=0), i


def test_mean_reader_gives_identical_cotangent_control():
    """The control that makes the test above bite."""
    m = _lsel("joint")
    cap = _capture(m)
    _fwd(m)
    cells = cap["cells"]
    (g,) = torch.autograd.grad(cap["loss"], cells, retain_graph=True)
    for i in range(1, M):
        torch.testing.assert_close(g[:, :, 0], g[:, :, i])


def test_row_cap_matches_uncapped():
    losses = []
    for cap_on in (False, True):
        m = _lsel("joint", spandec_read="each")
        m.cfg.spandec_ce_row_cap = cap_on
        cap = _capture(m)
        _fwd(m)
        losses.append(cap["loss"].detach())
    assert torch.isfinite(losses[1])
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)


def test_config_composes():
    p = pathlib.Path(__file__).resolve().parents[1] / "morph/configs/lxtul_pointer_fast_spdeach.yaml"
    txt = p.read_text()
    assert "spandec_read: each" in txt and "- lxtul_pointer_fast" in txt


def test_refusals():
    with pytest.raises(ValueError, match="spandec: true"):
        _lsel("joint", spandec=False, spandec_read="each")
    with pytest.raises(ValueError, match="fan_loop_select != 'off'"):
        _lsel("off", spandec_read="each")
    with pytest.raises(ValueError, match="fan_lsel_read == 'winner'"):
        _lsel("joint", fan_lsel_read="all", spandec_read="each")
    with pytest.raises(ValueError, match="spandec_reads_cells"):
        _lsel("joint", spandec_reads_cells=True, spandec_read="each")
