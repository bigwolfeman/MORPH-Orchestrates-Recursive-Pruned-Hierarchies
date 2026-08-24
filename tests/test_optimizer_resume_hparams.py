"""A resume must honour the CONFIG's optimizer hyperparameters, not the checkpoint's.

torch's `Optimizer.load_state_dict` replaces `param_groups` wholesale, hyperparameters
included. Every optimizer setting a resume passes on the command line is therefore silently
reverted to whatever the checkpoint was written with. Only `lr` escapes, because the LR
scheduler rewrites it every step.

Found 2026-08-24 the expensive way: an RCA arm resuming with ademamix_alpha_cap=1.0 against
a checkpoint written at 3.5 produced output BIT-IDENTICAL to its control across 229 steps
and 87 probe series. It looked like a null result about the optimizer; it was a silently
discarded flag.

train.py now snapshots the freshly-built optimizer's hyperparameters, lets load_state_dict
bring in the moment/EMA tensors, then puts the configured hyperparameters back. These tests
pin that behaviour on a plain optimizer, so they run anywhere and stay fast.
"""

from __future__ import annotations

import torch


def _reapply(optimizer, saved_state, cfg_groups):
    """The shape of the fix as train.py applies it."""
    optimizer.load_state_dict(saved_state)
    changed = {}
    for g, want in zip(optimizer.param_groups, cfg_groups):
        for k, v in want.items():
            if g.get(k) != v:
                changed.setdefault(k, (g.get(k), v))
            g[k] = v
    return changed


def _make(lr, weight_decay):
    p = torch.nn.Parameter(torch.zeros(4))
    return torch.optim.AdamW([p], lr=lr, weight_decay=weight_decay), p


def test_load_state_dict_alone_reverts_hyperparameters():
    """The defect itself. If torch ever stops doing this, the fix is unnecessary and this
    test says so loudly rather than the fix silently rotting."""
    old, p = _make(lr=1e-3, weight_decay=0.5)
    p.grad = torch.ones(4)
    old.step()
    saved = old.state_dict()

    new, _ = _make(lr=5e-5, weight_decay=0.01)      # the config the resume asked for
    new.load_state_dict(saved)
    assert new.param_groups[0]["weight_decay"] == 0.5, "torch no longer clobbers hparams"
    assert new.param_groups[0]["lr"] == 1e-3


def test_the_fix_restores_the_configured_hyperparameters():
    old, p = _make(lr=1e-3, weight_decay=0.5)
    p.grad = torch.ones(4)
    old.step()
    saved = old.state_dict()

    new, _ = _make(lr=5e-5, weight_decay=0.01)
    cfg = [{k: v for k, v in g.items() if k != "params"} for g in new.param_groups]
    changed = _reapply(new, saved, cfg)

    assert new.param_groups[0]["weight_decay"] == 0.01
    assert new.param_groups[0]["lr"] == 5e-5
    assert changed["weight_decay"] == (0.5, 0.01)
    assert changed["lr"] == (1e-3, 5e-5)


def test_the_fix_keeps_the_optimizer_STATE():
    """It must restore hyperparameters WITHOUT throwing away the moment estimates — those
    are the whole reason to resume rather than restart."""
    old, p = _make(lr=1e-3, weight_decay=0.5)
    p.grad = torch.ones(4)
    old.step()
    old.step()
    saved = old.state_dict()
    exp_avg = saved["state"][0]["exp_avg"].clone()
    steps = saved["state"][0]["step"]

    new, q = _make(lr=5e-5, weight_decay=0.01)
    cfg = [{k: v for k, v in g.items() if k != "params"} for g in new.param_groups]
    _reapply(new, saved, cfg)

    st = new.state[new.param_groups[0]["params"][0]]
    assert torch.equal(st["exp_avg"], exp_avg), "moment estimates were lost"
    assert st["step"] == steps


def test_nothing_is_reported_changed_when_nothing_changed():
    """A resume that does not override anything must be silent, so the log line means
    something when it does appear."""
    old, p = _make(lr=1e-3, weight_decay=0.5)
    p.grad = torch.ones(4)
    old.step()
    saved = old.state_dict()
    new, _ = _make(lr=1e-3, weight_decay=0.5)
    cfg = [{k: v for k, v in g.items() if k != "params"} for g in new.param_groups]
    assert _reapply(new, saved, cfg) == {}


def test_custom_hyperparameters_are_covered_too():
    """The fix is generic: it copies every non-`params` key, so optimizer-specific settings
    like ademamix's alpha_cap are covered without naming them."""
    p = torch.nn.Parameter(torch.zeros(4))
    old = torch.optim.AdamW([p], lr=1e-3)
    old.param_groups[0]["alpha_cap"] = 3.5
    p.grad = torch.ones(4)
    old.step()
    saved = old.state_dict()

    new = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=1e-3)
    new.param_groups[0]["alpha_cap"] = 1.0
    cfg = [{k: v for k, v in g.items() if k != "params"} for g in new.param_groups]
    changed = _reapply(new, saved, cfg)
    assert new.param_groups[0]["alpha_cap"] == 1.0
    assert changed["alpha_cap"] == (3.5, 1.0)
