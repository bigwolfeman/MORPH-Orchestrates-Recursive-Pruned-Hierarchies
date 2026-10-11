"""``tul.spandec_read: best`` (2026-10-11): the span decoder decodes the next span from EACH of
the M cells alone and grades each SPAN on its lowest-CE cell. The loss is the token-weighted
mean over valid spans of that minimum; the gradient of a span reaches its argmin cell only.

Files: morph/model/transformer.py (``_spandec_read_best``, ``select_best`` in
``_tul_spandec_loss``, ``_spandec_best_of_cells``, the call site in ``_forward_tul``),
morph/model/tul.py (``_check_spandec_read``), morph/configs/nowta_pointer_fast_spdbest.yaml.

What each test pins:
  a. The loss equals a hand reference: the decoder on each cell ALONE (unfolded path), per-span
     token-mean CE by ``F.cross_entropy`` on the decoder's logits, min over cells, token weight.
  b. Only the argmin cell of a span gets gradient for it: a losing cell's cotangent at that
     slot is exactly zero, a winner's is not, and different cells win different spans.
  c. The win-fraction stats sum to 1 and match the argmin counts.
  d. Refusals. ``mean`` / ``each`` stay pinned in test_spandec_read_winner.py and
     test_spandec_read_each.py (run them).
  e. The row cap (``spandec_ce_row_cap``) gives the same loss and the same winners.

CPU, fp32, the ``tests/test_tul_fan_lsel.py`` strict fixtures: the ungraded fan and LXTUL
(``M = 4`` cells).
"""
from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn.functional as F

from test_spandec_read_each import _capture, _fwd, _model
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M
from morph.model.tul_spandec import horizon_span_slots

M = int(M)
FIXTURES = ["off", "joint"]          # off = the ungraded fan, joint = LXTUL


def _span_ce(m, cap, i):
    """Cell i ALONE through the unfolded path, by hand: ``(span_ce [B, S], n [B, S])``,
    ``F.cross_entropy`` on the decoder's logits with the slot id masked out of the softmax."""
    tc, dec = m.cfg.tul, m.tul_spandec
    z = m._readout(cap["cells"][:, :, i])
    w = m.embed.lm_weight()
    ids, valid = horizon_span_slots(cap["ids"], cap["layout"], dec.per_span_tokens,
                                    dec.horizon, start=dec.target_offset)
    st = dec.decode(z, ids, valid, w.detach())
    logits = st @ (w.detach() if tc.mux_detach_head else w).t()
    logits[..., tc.slot_id] = float("-inf")
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1),
                         reduction="none").reshape(ids.shape)
    n = valid.sum(-1)
    return (ce * valid).sum(-1) / n.clamp(min=1), n


def _reference(m, cap):
    per = [_span_ce(m, cap, i) for i in range(M)]
    ce = torch.stack([p[0] for p in per])                     # [M, B, S]
    n = per[0][1]
    best, win = ce.min(0)                                      # first minimum on ties
    loss = (best * n).sum() / n.sum()
    return loss, win, n, ce


def test_best_flag_is_set_only_by_best():
    assert _lsel("joint", spandec_read="best")._spandec_read_best
    assert not _lsel("joint", spandec_read="each")._spandec_read_best
    assert not _lsel("joint")._spandec_read_best


@pytest.mark.parametrize("sel", FIXTURES)
def test_loss_is_the_token_weighted_min_over_cells(sel):
    m = _model(sel, spandec_read="best")
    cap = _capture(m)
    _fwd(m)
    assert cap["kw"]["fold_cells"] is True and cap["kw"]["select_best"] is True
    with torch.no_grad():
        ref, _win, n, ce = _reference(m, cap)
    assert int((n > 0).sum()) > 1
    # the min must differ from the mean over cells, or the test cannot tell them apart
    mean_loss = ((ce.mean(0) * n).sum() / n.sum())
    assert abs(float(mean_loss - ref)) > 1e-4
    torch.testing.assert_close(cap["loss"].detach(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sel", FIXTURES)
def test_only_the_argmin_cell_gets_the_gradient(sel):
    m = _model(sel, spandec_read="best")
    cap = _capture(m)
    _fwd(m)
    cells = cap["cells"]
    with torch.no_grad():
        ref, win, n, _ce = _reference(m, cap)
    (g,) = torch.autograd.grad(cap["loss"], cells, retain_graph=True)
    gn = g.flatten(3).abs().sum(-1) if g.dim() > 4 else g.abs().sum(-1)   # [B, S, M]
    live = n > 0
    assert len(set(win[live].tolist())) > 1, "one cell wins every span; proves nothing"
    for i in range(M):
        won = live & (win == i)
        lost = live & (win != i)
        assert bool((gn[:, :, i][lost] == 0).all()), f"cell {i} got gradient on a span it lost"
        if bool(won.any()):
            assert bool((gn[:, :, i][won] > 0).all()), f"cell {i} won a span with no gradient"
        assert bool((gn[:, :, i][~live] == 0).all())
    # and the winner's gradient is the gradient of that cell's own reference loss
    for i in range(M):
        if not bool((live & (win == i)).any()):
            continue
        ce_i, _ = _span_ce(m, cap, i)
        mask = (live & (win == i)).to(ce_i.dtype)
        (gi,) = torch.autograd.grad((ce_i * n * mask).sum() / n.sum(), cells,
                                    retain_graph=True)
        torch.testing.assert_close(g[:, :, i], gi[:, :, i], rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("sel", FIXTURES)
def test_win_fractions_sum_to_one_and_match_the_argmin_counts(sel):
    m = _model(sel, spandec_read="best")
    cap = _capture(m)
    _fwd(m)
    with torch.no_grad():
        _ref, win, n, _ce = _reference(m, cap)
    live = n > 0
    fr = torch.stack([cap["stats"][f"spandec_best_win_c{i}"] for i in range(M)])
    assert f"spandec_best_win_c{M}" not in cap["stats"]
    assert all(not v.requires_grad for v in fr)
    torch.testing.assert_close(fr.sum(), torch.tensor(1.0), rtol=0, atol=1e-6)
    want = torch.stack([(live & (win == i)).sum() for i in range(M)]).float() / live.sum()
    torch.testing.assert_close(fr, want, rtol=0, atol=1e-6)


def test_row_cap_matches_uncapped():
    out = []
    for cap_on in (False, True):
        m = _model("off", spandec_read="best")
        m.cfg.spandec_ce_row_cap = cap_on
        cap = _capture(m)
        _fwd(m)
        out.append((cap["loss"].detach(),
                    torch.stack([cap["stats"][f"spandec_best_win_c{i}"] for i in range(M)])))
    assert torch.isfinite(out[1][0])
    torch.testing.assert_close(out[0][0], out[1][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out[0][1], out[1][1], rtol=0, atol=1e-6)


def test_config_composes():
    d = pathlib.Path(__file__).resolve().parents[1] / "morph/configs"
    txt = (d / "nowta_pointer_fast_spdbest.yaml").read_text()
    assert "spandec_read: best" in txt and "- nowta_pointer_fast" in txt
    assert "name: nowta-pointer-fast-spdbest" in txt


def test_refusals():
    with pytest.raises(ValueError, match="spandec: true"):
        _lsel("joint", spandec=False, spandec_read="best")
    with pytest.raises(ValueError, match="slot_cells > 1"):
        _model("off", spandec_read="best", slot_cells=1)
    with pytest.raises(ValueError, match="spandec_reads_cells"):
        _lsel("joint", spandec_reads_cells=True, spandec_read="best")
    with pytest.raises(ValueError, match="'mean', 'winner', 'each' or 'best'"):
        _lsel("joint", spandec_read="worst")
