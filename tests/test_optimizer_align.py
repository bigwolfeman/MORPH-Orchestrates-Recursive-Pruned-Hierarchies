"""Re-indexing optimizer state when a resume adds parameters.

This exists because the first SCSE screen (2026-08-25) could not start: SCSE adds two
projections, so the optimizer's decay group grew and `load_state_dict` raised
"a parameter group that doesn't match the size of optimizer's group".

The raise is the LUCKY case. torch matches saved state to live parameters by POSITION, so
an added module whose parameters land in the middle of a group shifts every later index
and pairs each parameter with another parameter's moments — with no error at all if the
sizes happen to line up. These tests assert the MAPPING, not just that nothing throws.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from morph.training.optimizer import align_optimizer_state, param_group_names


class Tiny(nn.Module):
    """`a` and `c` are always present; `b` stands for a module added by an intervention
    and sits BETWEEN them in `named_parameters()` order, which is the hard case.

    `norm` and the biases exist so the decay / no-decay split is NON-TRIVIAL. Without them
    every parameter lands in group 0 and a `param_group_names` that ignored the split
    entirely still passed every test here — that escape was found by sabotage, not by
    reading."""

    def __init__(self, with_b: bool):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=True)
        self.norm = nn.LayerNorm(4)
        if with_b:
            self.b = nn.Linear(4, 4, bias=True)
        self.c = nn.Linear(4, 4, bias=True)


def _state_for(model, marker: float) -> dict:
    """An optimizer state dict whose moments identify their parameter, so a mis-mapping
    is visible rather than merely possible."""
    names = param_group_names(model)
    flat = [n for g in names for n in g]
    return {
        "state": {i: {"exp_avg": torch.full((2,), marker + i)} for i in range(len(flat))},
        "param_groups": [{"params": list(range(len(names[0]))), "weight_decay": 0.1},
                         {"params": list(range(len(names[0]),
                                               len(names[0]) + len(names[1]))),
                          "weight_decay": 0.0}],
    }


def test_no_added_parameters_returns_the_state_untouched():
    m = Tiny(with_b=False)
    st = _state_for(m, 100.0)
    out, added = align_optimizer_state(st, m, {n for g in param_group_names(m) for n in g})
    assert added == []
    assert out is st


def test_a_parameter_added_in_the_MIDDLE_shifts_only_the_ones_after_it():
    old = Tiny(with_b=False)
    new = Tiny(with_b=True)
    old_names = [n for g in param_group_names(old) for n in g]
    new_names = [n for g in param_group_names(new) for n in g]
    assert old_names == ["a.weight", "c.weight", "a.bias", "norm.weight", "norm.bias",
                         "c.bias"]
    assert new_names == ["a.weight", "b.weight", "c.weight", "a.bias", "norm.weight",
                         "norm.bias", "b.bias", "c.bias"]
    assert new_names.index("b.weight") < new_names.index("c.weight"), "b lands mid-group"

    st = _state_for(old, 100.0)
    out, added = align_optimizer_state(st, new, set(old_names))

    assert sorted(added) == ["b.bias", "b.weight"]
    # Every surviving name keeps ITS OWN moment, identified by the marker value.
    for old_i, name in enumerate(old_names):
        new_i = new_names.index(name)
        assert out["state"][new_i]["exp_avg"][0].item() == pytest.approx(100.0 + old_i), (
            f"{name} moved from index {old_i} to {new_i} and lost its moments")
    for name in added:
        assert new_names.index(name) not in out["state"]


def test_the_added_parameter_gets_no_state_so_it_starts_cold():
    new = Tiny(with_b=True)
    old_names = [n for g in param_group_names(Tiny(with_b=False)) for n in g]
    out, _ = align_optimizer_state(_state_for(Tiny(with_b=False), 100.0), new, set(old_names))
    b_index = [n for g in param_group_names(new) for n in g].index("b.weight")
    assert b_index not in out["state"], "an added parameter must start with fresh moments"


def test_group_index_lists_cover_every_live_parameter():
    new = Tiny(with_b=True)
    old_names = {n for g in param_group_names(Tiny(with_b=False)) for n in g}
    out, _ = align_optimizer_state(_state_for(Tiny(with_b=False), 100.0), new, old_names)
    names = param_group_names(new)
    flat = sum(len(g) for g in names)
    assert [len(g["params"]) for g in out["param_groups"]] == [len(g) for g in names]
    assert sorted(i for g in out["param_groups"] for i in g["params"]) == list(range(flat))


def test_hyperparameters_of_each_group_survive_the_realignment():
    new = Tiny(with_b=True)
    old_names = {n for g in param_group_names(Tiny(with_b=False)) for n in g}
    out, _ = align_optimizer_state(_state_for(Tiny(with_b=False), 100.0), new, old_names)
    assert out["param_groups"][0]["weight_decay"] == 0.1
    assert out["param_groups"][1]["weight_decay"] == 0.0


def test_a_REMOVED_parameter_raises_instead_of_guessing():
    """Not a pure insertion. Guessing here pairs parameters with the wrong moments."""
    m = Tiny(with_b=False)
    st = _state_for(Tiny(with_b=True), 100.0)     # checkpoint has MORE than the live model
    ckpt_names = {n for g in param_group_names(Tiny(with_b=True)) for n in g}
    with pytest.raises(ValueError, match="not a pure insertion"):
        align_optimizer_state(st, m, ckpt_names)


def test_a_changed_group_layout_raises():
    m = Tiny(with_b=True)
    st = _state_for(Tiny(with_b=False), 100.0)
    st["param_groups"] = st["param_groups"][:1]          # one group instead of two
    old_names = {n for g in param_group_names(Tiny(with_b=False)) for n in g}
    with pytest.raises(ValueError, match="optimizer groups"):
        align_optimizer_state(st, m, old_names)


def test_param_group_names_matches_the_real_optimizer_group_sizes():
    """The names view and the tensor view come from one walk; this pins them together, so
    a later change to the decay rule cannot silently desynchronise them."""
    from morph.training.optimizer import _param_groups
    m = Tiny(with_b=True)
    groups = _param_groups(m, weight_decay=0.1)
    names = param_group_names(m)
    assert [len(g["params"]) for g in groups] == [len(n) for n in names]
    # The split must be NON-TRIVIAL here, or this test proves nothing.
    assert len(names[0]) > 0 and len(names[1]) > 0
    assert "norm.weight" in names[1] and "a.bias" in names[1]
    assert "a.weight" in names[0]


def test_alignment_actually_loads_into_a_real_optimizer():
    """End to end: the aligned dict must be accepted by torch, and the surviving parameter
    must carry ITS OWN moment across, not its neighbour's."""
    old, new = Tiny(with_b=False), Tiny(with_b=True)
    opt_old = torch.optim.AdamW(_groups(old))
    for p in old.parameters():
        p.grad = torch.ones_like(p)
    opt_old.step()
    st = opt_old.state_dict()

    opt_new = torch.optim.AdamW(_groups(new))
    old_names = [n for g in param_group_names(old) for n in g]
    aligned, added = align_optimizer_state(st, new, set(old_names))
    opt_new.load_state_dict(aligned)            # raised before this function existed

    live = [n for g in param_group_names(new) for n in g]
    c_new = opt_new.state_dict()["state"][live.index("c.weight")]["exp_avg"]
    c_old = st["state"][old_names.index("c.weight")]["exp_avg"]
    assert torch.equal(c_new, c_old), "c must keep its own moments, not b's or a's"
    assert live.index("b.weight") not in opt_new.state_dict()["state"]


def _groups(m):
    from morph.training.optimizer import _param_groups
    return _param_groups(m, weight_decay=0.1)
