"""The depth-mask probe (`lxtul/depth_mask_eval.py`), CPU only.

A tiny random plain_pointer model (build() with `Module.cuda` patched to a no-op). Checks: an all-False
mask is the model's own depth 1, an all-True mask is its own depth D, a partial mask is neither, a mask
change at position t leaves every position before t unchanged, the wrapper is gone after the block, and
the boundary / next-4 / random-position helpers do what the probe's readouts assume.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_depth_mask_eval.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from lxtul.build import build
from lxtul.depth_mask_eval import (boundary_mask, depth_mask, next_k_mask, parcae_of, position_nll,
                                   random_like)
from lxtul.plain_pointer import PlainPointer

CONF = str(Path(__file__).resolve().parents[1] / "configs")
SEQ = 48
D = 3
TINY = ["shape.n_embd=64", "shape.num_attention_heads=2", "shape.num_key_value_heads=2",
        "shape.intermediate_size=128", f"shape.block_size={SEQ}", "run.name=t", "run.wandb_mode=disabled"]


@pytest.fixture(scope="module")
def tiny():
    mp = pytest.MonkeyPatch()
    mp.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    with initialize_config_dir(CONF, version_base=None):
        cfg = compose("train", overrides=["arm=plain_pointer", *TINY])
    torch.manual_seed(0)
    model, mcfg, _ = build(cfg, str(cfg.arm.kind), compile=False)
    mp.undo()
    with torch.no_grad():
        for p in model.parameters():                            # zero-init weights would hide wiring bugs
            if p.ndim >= 2 and not p.abs().any():
                p.normal_(0, 0.05)
    vocab = int(mcfg.model.vocab_size)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(20, vocab, (2, SEQ), generator=g)
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    return model.eval(), x, y, vocab


def nll(model, x, y, depth, deep, seed=7):
    return position_nll(model, x, y, depth, deep, slot_id=0, amp=False, seed=seed)


def test_all_false_mask_is_depth_one(tiny):
    m, x, y, _ = tiny
    assert isinstance(m, PlainPointer)
    off = torch.zeros(x.shape, dtype=torch.bool)
    assert torch.equal(nll(m, x, y, D, off), nll(m, x, y, 1, None))


def test_all_true_mask_is_depth_d(tiny):
    m, x, y, _ = tiny
    on = torch.ones(x.shape, dtype=torch.bool)
    full = nll(m, x, y, D, on)
    assert torch.equal(full, nll(m, x, y, D, None))
    assert not torch.equal(full, nll(m, x, y, 1, None))         # depth matters in this model


def test_partial_mask_is_neither_end(tiny):
    m, x, y, _ = tiny
    deep = torch.zeros(x.shape, dtype=torch.bool)
    deep[:, 10:20] = True
    part = nll(m, x, y, D, deep)
    assert not torch.equal(part, nll(m, x, y, 1, None))
    assert not torch.equal(part, nll(m, x, y, D, None))
    assert torch.equal(part[:, :10], nll(m, x, y, 1, None)[:, :10])   # before the deep block: depth 1


def test_mask_is_causal(tiny):
    m, x, y, _ = tiny
    t = 25
    a = torch.zeros(x.shape, dtype=torch.bool)
    a[:, 5] = True
    b = a.clone()
    b[:, t] = True                                              # differs from a only at position t
    na, nb = nll(m, x, y, D, a), nll(m, x, y, D, b)
    assert torch.equal(na[:, :t], nb[:, :t])
    assert not torch.equal(na[:, t:], nb[:, t:])                # and the change is visible at t or later


def test_wrapper_is_removed(tiny):
    m, x, y, _ = tiny
    p = parcae_of(m)
    before = nll(m, x, y, D, None)
    with depth_mask(m, torch.ones(x.shape, dtype=torch.bool)):
        assert "update_recurrent_state" in vars(p)
    assert "update_recurrent_state" not in vars(p)
    assert torch.equal(nll(m, x, y, D, None), before)


def test_wrong_mask_shape_raises(tiny):
    m, x, y, _ = tiny
    with pytest.raises(ValueError, match="does not match"):
        nll(m, x, y, D, torch.ones(2, SEQ - 1, dtype=torch.bool))


def _rule(size: int = 50):
    from morph.model.tul_layout import BoundaryRule
    lut = np.zeros(size, dtype=bool)
    lut[10] = True                                              # id 10 is the only punctuation token
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def test_boundary_positions_on_a_hand_written_row():
    ids = [5, 5, 10,                                            # 10 at span_len 3: below min_span, no cut
           5, 10,                                               # span_len 5: boundary at 4
           5, 5, 10,                                            # span_len 3 since 4: no cut
           5, 10,                                               # span_len 5: boundary at 9
           5, 0]                                                # EOS at 11 cuts at any length
    ids += [5] * 38                                             # no punctuation: the cap forces 11 + 32 = 43
    bnd = boundary_mask(torch.tensor([ids]), _rule())
    assert bnd[0].nonzero().flatten().tolist() == [4, 9, 11, 43]


def test_boundary_flags_are_causal():
    r = _rule()
    ids = torch.tensor([[5, 5, 5, 5, 10, 5, 5, 5, 5, 10, 5, 5]])
    full = boundary_mask(ids, r)
    for n in range(1, ids.shape[1] + 1):
        assert torch.equal(boundary_mask(ids[:, :n], r)[0], full[0, :n])


def test_next4_stops_at_the_next_boundary():
    bnd = torch.zeros(1, 20, dtype=torch.bool)
    bnd[0, [3, 6, 15]] = True
    got = next_k_mask(bnd)[0].nonzero().flatten().tolist()
    assert got == [4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19]        # 4..6 (cut at 6), 7..10, 16..19


def test_random_positions_match_the_boundary_count_and_avoid_boundaries():
    g = torch.Generator().manual_seed(0)
    bnd = torch.zeros(3, 40, dtype=torch.bool)
    bnd[0, [5, 9, 20]] = True
    bnd[1, [7]] = True
    valid = torch.ones_like(bnd)
    valid[2, 30:] = False
    deep = random_like(bnd, valid, g)
    assert deep.sum(1).tolist() == [3, 1, 0]
    assert not (deep & bnd).any() and not (deep & ~valid).any()
    again = random_like(bnd, valid, torch.Generator().manual_seed(0))
    assert torch.equal(deep, again)                             # a draw is fixed by its seed


def test_run_probe_scores_the_conditions_on_shared_position_sets(tiny):
    from types import SimpleNamespace
    from lxtul.depth_mask_eval import run_probe
    m, x, y, vocab = tiny
    rule = _rule(vocab)
    x = torch.where(torch.rand(x.shape, generator=torch.Generator().manual_seed(3)) < 0.15,
                    torch.full_like(x, 10), x)                  # sprinkle the punctuation id
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    lay = SimpleNamespace(slot_mask=torch.zeros(x.shape, dtype=torch.bool))
    res = run_probe(m, [(x, y, lay)], rule, slot_id=0, depth=D, draws=2, amp=False, device="cpu",
                    log=lambda *a, **k: None)
    valid = y >= 0
    bnd = boundary_mask(x, rule) & valid
    assert res["n"] == {"n_all": int(valid.sum()), "n_shallow": int((valid & ~bnd).sum()),
                        "n_next4": int((next_k_mask(bnd) & valid).sum()), "n_boundary_own": int(bnd.sum())}
    assert bnd.any()
    ce = res["ce"]
    d1, dD = nll(m, x, y, 1, None, seed=1000), nll(m, x, y, D, None, seed=1000)   # run_probe's batch-0 seed
    assert ce["all_1"]["ce_all"] == pytest.approx(float(d1.double().sum() / valid.sum()), abs=1e-9)
    assert ce[f"all_{D}"]["ce_all"] == pytest.approx(float(dD.double().sum() / valid.sum()), abs=1e-9)
    assert ce[f"all_{D}"]["ce_boundary_own"] == pytest.approx(float((dD.double() * bnd).sum() / bnd.sum()), abs=1e-9)
    assert set(ce[f"random_{D}"]["draws"]) == {"draw0", "draw1"}
    assert ce[f"random_{D}"]["mean"]["ce_all"] == pytest.approx(
        np.mean([v["ce_all"] for v in ce[f"random_{D}"]["draws"].values()]))
