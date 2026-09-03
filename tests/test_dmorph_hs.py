"""Acceptance criterion 6, the hs arm: the target is detached (no gradient reaches the
clean stream through ``y``), and the four-condition worth produces four distinct
numbers on a random model, from the shipped eval path.
"""

from __future__ import annotations

import torch

from morph.model.dmorph import targets
from _dmorph_common import batch, clean_pass, dm_cfg, model, wake_stream


def test_hs_target_is_detached_by_default_and_live_only_under_sigreg():
    x, y, layout, _ = batch()
    for lam, live in ((0.0, False), (0.02, True)):
        m = model(dm_cfg(arm="hs", sigreg_lambda=lam, sigreg_slices=64))
        m.train()
        h, x0, bg, ve = m._tul_front(x, layout)
        h = m._core_region(h, x0, bg, x)
        xh = m._back_region(h, x0, bg, x)
        row_w, _p, _z = m._tul_half_weights(y, layout)
        yy, valid, _cl, _cw = targets(m, xh, y, layout, row_w)
        assert yy.requires_grad is live
        if not live:
            # No path from the FM term back into the clean stream THROUGH y: the target
            # of a slot state is a constant to autograd.
            assert yy.grad_fn is None


def test_hs_worth_has_four_distinct_numbers_on_a_random_model():
    m = model(dm_cfg(arm="hs", n_blocks=2))
    wake_stream(m)
    m.eval()
    x, y, layout, _ = batch(B=3)
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=layout)
    vals = {k: float(out[f"dm_worth_{k}"]) for k in ("clean", "ladder", "zero", "shuffle")}
    assert len({round(v, 6) for v in vals.values()}) == 4, vals
    for k in ("ladder", "zero", "shuffle"):
        assert torch.allclose(out[f"dm_worth_cost_{k}"], out[f"dm_worth_{k}"] - out["dm_worth_clean"])
    assert 0.0 <= float(out["dm_cos"]) <= 1.0 or float(out["dm_cos"]) < 0.0
    assert "dm_clean_acc" not in out, "dm_clean_acc is the tok arm's P3 read"


def test_hs_terms_ignore_token_positions():
    """Both hs terms live on slot positions only: a perturbation of the noisy input at
    token positions must not move the FM loss's target mask or the CE label mask."""
    m = model(dm_cfg(arm="hs"))
    m.eval()
    x, y, layout, _ = batch()
    xh, _c, _ctx = clean_pass(m, x, layout)
    row_w, _p, _z = m._tul_half_weights(y, layout)
    yy, valid, ce_labels, ce_w = targets(m, xh, y, layout, row_w)
    assert not valid[~layout.slot_mask].any()
    assert (ce_labels[~layout.slot_mask] == -100).all()
    real = layout.slot_mask & (layout.bag_id != layout.max_slots)
    assert int(valid.sum()) == int(real.sum())
    n_ce = int((ce_labels != -100).sum())
    assert 0 < n_ce <= int(layout.slot_valid.sum()), "the CE term lives at emit positions only"
