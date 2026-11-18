"""OWT rows widened to the hop layout (`hops.base.widen_batch`), and the width guard (CPU).

The MORPH config stays stock (64 slots): OWT rows are packed as without the task. A slot arm trained with the hop
task has `max_slots` 256, so its OWT rows (training stream and eval set) are WIDENED: same tokens, slots and labels,
extra tail pads. Pinned here:
  1. the real positions, labels and slot arrays of a widened row equal the stock row's; the pads follow the packer's
     own tail-pad convention (read off a row the real packer ended early);
  2. on CPU with the smallest model: the lxtul forward on a stock row and on its widened copy gives the same hidden
     state at every real position and the same NLL sum (widening changes nothing);
  3. with the task ON, the OWT rows of the eval set hold the same tokens as with it OFF (lxtul widened, flat untouched);
  4. for each of the four arms composed with task=hops, every eval batch the trainer feeds the model (OWT eval rows and
     hop eval rows) is no wider than the model's block_size.

    PYTHONPATH=.:$MORPH_ROOT HF_HUB_OFFLINE=1 python -m pytest -q lxtul/tests/test_hops_widen.py
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import lxtul.evaluate as ev
import parcae_lm
from lxtul.build import build
from lxtul.data import layout_for_cells, morph_cfg_of, tul_runtime
from lxtul.hops.base import widen_batch
from lxtul.hops.eval import make_batches
from lxtul.hops.mix import build_task
from lxtul.model import LXTULParcae
from lxtul.tests.hops_support import SLOT, compose_cfg, real, real_task, tiny_rule
from lxtul.tests.test_compact import S, SEQ, _randomize, cpu_reference_kernels  # noqa: F401  (flex -> SDPA, fp64)
from lxtul.tests.test_hops_rows import _owt_like
from lxtul.tests.test_minimal import _keys, _row1

from morph.model.tul_layout import TulLayoutSpec, pack_tul_batch


def _stock_and_wide(cells=1):
    task, rt = real_task(), real()[2]
    stock_spec = layout_for_cells(rt, cells).data_cfg.spec_for(task.seq_len)
    batch = _owt_like(stock_spec, 1, 4, 3)[0]
    return batch, task.spec(cells), stock_spec


def test_real_positions_equal_and_pads_follow_the_packer_convention():
    (x, y, lay), wide, stock = _stock_and_wide(1)
    assert stock.max_slots == 64 and wide.max_slots == 256 and x.shape[1] == 1088
    wx, wy, wl = widen_batch((x, y, lay), wide)
    L0, S0 = x.shape[1], stock.max_slots
    assert wx.shape == wy.shape == wl.slot_mask.shape == wl.bag_id.shape == (4, 1280)
    assert wl.slot_index.shape == wl.slot_valid.shape == (4, 256)
    assert (wx[:, :L0] == x).all() and (wy[:, :L0] == y).all() and (wl.slot_mask[:, :L0] == lay.slot_mask).all()
    assert (wl.slot_index[:, :S0] == lay.slot_index).all() and (wl.slot_valid[:, :S0] == lay.slot_valid).all()
    assert not wl.slot_index[:, S0:].any() and not wl.slot_valid[:, S0:].any()
    real_b = lay.bag_id < S0                                                  # slot indices stay; the dump bin moves
    assert (wl.bag_id[:, :L0][real_b] == lay.bag_id[real_b]).all() and (wl.bag_id[:, :L0][~real_b] == 256).all()
    assert (wx[:, L0:] == SLOT).all() and (wy[:, L0:] == -100).all() and wl.slot_mask[:, L0:].all() and (wl.bag_id[:, L0:] == 256).all()
    # the convention itself, read off a row the real packer ended early (max_slots reached): a long pad tail
    rule = tiny_rule()
    spec = TulLayoutSpec(seq_len=64, prefix_k=1, max_slots=4, slot_id=SLOT)
    buf = np.tile([20, 21, 22, 12, 23, 24, 25, 26], 40).tolist()                # a boundary (id 12) every 4 tokens
    px, py, pl = pack_tul_batch(buf, rule, spec, 1)
    pad = pl.slot_mask[0] & (pl.bag_id[0] == 4)
    assert int(pad.sum()) > 20 and (px[0][pad] == SLOT).all() and (py[0][pad] == -100).all()
    w2 = widen_batch((px, py, pl), TulLayoutSpec(seq_len=64, prefix_k=1, max_slots=8, slot_id=SLOT))
    tail = slice(px.shape[1], None)
    assert (w2[0][0, tail] == SLOT).all() and (w2[1][0, tail] == -100).all() and w2[2].slot_mask[0, tail].all()
    assert (w2[2].bag_id[0][pad.numpy().nonzero()[0]] == 8).all() and (w2[2].bag_id[0, tail] == 8).all()
    with pytest.raises(ValueError, match="cannot widen"):
        widen_batch((x, y, lay), dataclasses.replace(wide, prefix_k=4))


def _model_s(base: LXTULParcae, slots: int) -> LXTULParcae:
    """The tiny model of `base` with a larger slot capacity, same weights (the rope table only grows)."""
    cfg = dataclasses.replace(base.config, block_size=SEQ + slots)
    m = LXTULParcae(cfg, dataclasses.replace(base.tul, max_slots=slots)).double().eval()
    sd, new = base.state_dict(), m.state_dict()
    m.load_state_dict({k: (v if v.shape == new[k].shape else new[k]) for k, v in sd.items()}, strict=True)
    assert {k for k in sd if sd[k].shape != new[k].shape} <= {"freqs_cis"}
    for k in ("freqs_cis",):
        if k in sd:                                          # a non-persistent table: same first rows
            n = sd[k].shape[1]
            assert torch.equal(sd[k], new[k][:, :n])
    return m


def test_widened_row_gives_the_same_hidden_states_and_nll():
    from lxtul.tests.test_minimal import _model
    base = _model(1).eval()
    (x, y, lay), spec = _row1(2, B=3)
    assert spec.max_slots == S == 16
    wide = TulLayoutSpec(seq_len=SEQ, prefix_k=1, max_slots=32, slot_id=SLOT)
    wx, wy, wl = widen_batch((x, y, lay), wide)
    m2 = _model_s(base, 32)
    with torch.no_grad():
        h1 = base(x, None, lay, depth=3)["hidden"]
        h2 = m2(wx, None, wl, depth=3)["hidden"]
        real = ~lay.slot_mask | (lay.bag_id < S)                               # tokens and valid slot cells
        assert real.any() and wl.slot_mask[:, x.shape[1]:].all() and wl.slot_mask.shape[1] > x.shape[1]    # the widened tail is pad
        assert torch.allclose(h1[real], h2[:, : x.shape[1]][real], rtol=1e-9, atol=1e-9)
        n1 = base.token_nll(h1, x, y, lay.slot_mask, lay)
        n2 = m2.token_nll(h2, wx, wy, wl.slot_mask, wl)
    assert abs(float(n1.sum()) - float(n2.sum())) < 1e-8 * max(1.0, float(n1.sum())) and float(n1.sum()) > 0
    sab = h2[:, : x.shape[1]].clone()
    sab[real] += 1e-3                                                           # the comparison would notice a change
    assert not torch.allclose(h1[real], sab[real], rtol=1e-9, atol=1e-9)


def test_owt_eval_rows_hold_the_same_tokens_with_the_task_on_or_off(monkeypatch):
    task, rt = real_task(), layout_for_cells(real()[2], 1)
    stock = _owt_like(rt.data_cfg.spec_for(task.seq_len), 3, 4, 5)
    monkeypatch.setattr(ev, "make_loader", lambda *a, **k: iter([tuple(b) for b in stock]))
    mcfg = real()[3]
    off = ev.eval_rows(mcfg, rt, 12, 4)
    on = ev.eval_rows(mcfg, rt, 12, 4, widen_to=task.spec(1))
    assert all(a[0] is b[0] for a, b in zip(off, stock)), "task off: the stock batches, untouched"
    for a, b in zip(off, on):
        L0 = a[0].shape[1]
        assert b[0].shape[1] == 1280 and (b[0][:, :L0] == a[0]).all() and (b[1][:, :L0] == a[1]).all()
        assert (b[2].slot_mask[:, :L0] == a[2].slot_mask).all() and (b[0][:, :L0][~a[2].slot_mask] == a[0][~a[2].slot_mask]).all()
        assert int((b[1] >= 0).sum()) == int((a[1] >= 0).sum())               # the same scored tokens


@pytest.mark.parametrize("arm,extra", [("gpt8_pointer", []), ("plain_pointer", []), ("minimal", []),
                                       ("minimal", ["+lxtul.mean_depth=1", "+lxtul.max_depth=1", "+lxtul.eval_depth=1"])])
def test_every_eval_batch_fits_the_models_block_size(arm, extra, monkeypatch):
    """The guard for the failure that reached the GPU: a batch wider than block_size. Built from the real configs."""
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    cfg = compose_cfg(arm, "task=hops", *extra)
    kind = str(cfg.arm.kind)
    ov = OmegaConf.to_container(cfg.arm.lxtul, resolve=True) if "lxtul" in cfg.arm else {}
    model, mcfg, rt = build(cfg, kind, overrides=ov, compile=False)
    block = int(model.config.block_size)
    eval_rt = rt if rt is not None else tul_runtime(morph_cfg_of(cfg))
    task = build_task(cfg.task, mcfg, eval_rt)
    seq = int(mcfg.data.seq_len)
    assert mcfg.tul.max_slots == 64, "the MORPH config must stay stock"
    stock_w = eval_rt.data_cfg.spec_for(seq).l_total                           # width of the OWT eval rows as MORPH packs them
    if kind == "lxtul":
        owt_w = task.spec(model.tul.cells).l_total                              # widened
        assert model.tul.max_slots == 256 and owt_w == seq + model.tul.cells * 256
    else:
        owt_w = stock_w                                                         # plain_view keeps the stock width
        assert task.seq_len <= block
    rows = [r for _, r in task.eval_rows()][:12]
    hop = make_batches(task, kind, eval_rt, rows, 6)
    assert owt_w <= block, (arm, owt_w, block)
    assert all(b.x.shape[1] <= block and b.y.shape == b.x.shape for b in hop), (arm, [b.x.shape for b in hop], block)
    if kind == "lxtul":
        assert all(b.x.shape[1] == block and b.lay.slot_index.shape[1] == model.tul.max_slots for b in hop)
    del model
