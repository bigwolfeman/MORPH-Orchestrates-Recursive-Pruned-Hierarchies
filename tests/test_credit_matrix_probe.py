"""Contracts of `lab/divergence/credit_matrix.py` that fail silently if broken.

* `DetachTap` must hand back the LIVE tensor behind a detached scalar, so a folded loss
  term can be differentiated on its own; if it returned nothing or the wrong tensor, the
  matrix would book that term as token CE (the closure check catches a missing term, not a
  wrong one).
* `group_of` must put every layer's `x0_injects.i` with its own section (prelude / core /
  coda) and must refuse an unknown parameter instead of dropping it.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))
from credit_matrix import DetachTap, group_of  # noqa: E402


def test_detach_tap_returns_the_live_tensor_and_its_gradient():
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    with DetachTap() as tap:
        a = (w * 3.0).sum()             # 9.0
        b = (w ** 2).sum()              # 5.0
        da, db = a.detach(), b.detach()
        _ = w.detach()                  # not 0-dim: not recorded
    assert tap.live_of(da) is a and tap.live_of(db) is b
    assert tap.live_of(torch.tensor(9.0)) is None
    assert len(tap.pairs) == 2
    (g,) = torch.autograd.grad(tap.live_of(db), w)
    assert torch.equal(g, torch.tensor([2.0, 4.0]))


def test_group_of_maps_injections_by_layer_and_refuses_unknowns():
    np_, nc = 4, 6
    assert group_of("x0_injects.3.proj.weight", np_, nc) == "prelude"
    assert group_of("x0_injects.4.log_scale", np_, nc) == "loop_core"
    assert group_of("x0_injects.9.log_scale", np_, nc) == "loop_core"
    assert group_of("x0_injects.10.log_scale", np_, nc) == "coda"
    assert group_of("tul.W_prefix", np_, nc) == "W_prefix"
    assert group_of("tul.E_mask", np_, nc) == "coda"          # the token-dropout vector
    assert group_of("tul.E_slot", np_, nc) == "slot_seed_register"
    assert group_of("tul.W_sent.parametrizations.weight.original", np_, nc) == \
        "slot_seed_register"
    assert group_of("_orig_mod.core.2.norm_mlp.weight", np_, nc) == "loop_core"
    with pytest.raises(KeyError):
        group_of("some_new_module.weight", np_, nc)
