"""Acceptance criterion 1: ``dmorph=None`` is bit-identical to the tree without dmorph
(loss, every parameter, every gradient), and a dmorph model's CLEAN stream is the
control's clean stream to the bit.

Three facts, each a separate assertion:

1. Construction. A dmorph model and a control built at the same seed share every base
   weight (the stream is built LAST and its params are the only difference).
2. The plain path. ``forward(x, labels)`` with no layout never reaches the stream:
   loss and every base gradient equal the control's; every ``dmorph.*`` grad is None.
3. The TUL path. On the same packed batch the dmorph model's ``loss_tokens_only`` (the
   clean CE, graph intact) equals the control's ``loss`` bitwise, in eval AND in train
   mode with dropout (the stream draws its RNG after the clean pass), and backprop of
   the clean CE alone reproduces the control's gradients bitwise — the K/V capture is
   a pure stash.

The cross-commit half of the claim (this tree's ``dmorph=None`` forward equals the
forward at ``5cb8fc5``) was checked by hand with the same batch before the change
landed and is recorded in the design note's Implementation record.
"""

from __future__ import annotations

import torch

from _dmorph_common import V, batch, dm_cfg, model


def _base_named(m):
    return [(n, p) for n, p in sorted(m.named_parameters()) if not n.startswith("dmorph.")]


def test_dmorph_construction_does_not_move_the_base_weights():
    ctl = model(None)
    dm = model(dm_cfg())
    assert dm.dmorph is not None and ctl.dmorph is None
    a = torch.cat([p.detach().flatten() for _, p in _base_named(ctl)])
    b = torch.cat([p.detach().flatten() for _, p in _base_named(dm)])
    assert torch.equal(a, b), "building the noisy stream perturbed the base init RNG"
    assert any(n.startswith("dmorph.") for n, _ in dm.named_parameters())


def test_plain_path_never_reaches_the_stream():
    x = torch.randint(5, V, (2, 32))
    y = torch.randint(5, V, (2, 32))
    outs, grads = [], []
    for cfg in (None, dm_cfg()):
        m = model(cfg, dropout=0.1)
        m.train()
        torch.manual_seed(99)
        out = m(x, labels=y)
        out["loss"].backward()
        outs.append(out["loss"].detach().clone())
        grads.append(torch.cat([p.grad.flatten() for _, p in _base_named(m)
                                if p.grad is not None]))
        for n, p in m.named_parameters():
            if n.startswith("dmorph."):
                assert p.grad is None, f"{n} received a gradient on the plain path"
        assert "dm_fm" not in out and "loss_tokens_only" not in out
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(grads[0], grads[1])


def test_tul_path_clean_stream_is_the_control_to_the_bit_in_eval():
    x, y, layout, _ = batch()
    ctl, dm = model(None), model(dm_cfg())
    ctl.eval(); dm.eval()
    with torch.no_grad():
        a = ctl(x, labels=y, slot_layout=layout)
        b = dm(x, labels=y, slot_layout=layout)
    assert "loss_tokens_only" not in a
    assert torch.equal(a["loss"], b["loss_tokens_only"])
    assert torch.equal(a["ce_tokens"], b["ce_tokens"])
    assert torch.equal(a["ce_emit"], b["ce_emit"])
    assert not torch.equal(a["loss"], b["loss"]), "the stream's terms must be IN the total"
    assert "dm_fm" in b and "dm_ce" in b


def test_tul_path_clean_gradients_are_the_controls_in_train_mode():
    x, y, layout, _ = batch()
    ctl, dm = model(None, dropout=0.1), model(dm_cfg(), dropout=0.1)
    ctl.train(); dm.train()
    torch.manual_seed(5)
    a = ctl(x, labels=y, slot_layout=layout)
    a["loss"].backward()
    torch.manual_seed(5)
    b = dm(x, labels=y, slot_layout=layout)
    b["loss_tokens_only"].backward()
    assert torch.equal(a["loss"], b["loss_tokens_only"])
    ga = torch.cat([p.grad.flatten() for _, p in _base_named(ctl) if p.grad is not None])
    gb = torch.cat([p.grad.flatten() for _, p in _base_named(dm) if p.grad is not None])
    assert torch.equal(ga, gb), "the clean CE's backward differs with the stream built"
    for n, p in dm.named_parameters():
        if n.startswith("dmorph."):
            assert p.grad is None, f"{n} is in the clean CE's graph"


def test_a_dmorph_model_refuses_a_core_or_a_missing_layout_config():
    import pytest
    with pytest.raises(ValueError, match="n_core == 0"):
        model(dm_cfg(), n_core=2)
    with pytest.raises(ValueError, match="requires tul"):
        model(dm_cfg(), tul=None)
