"""Onset-capture instruments (lab/experiments/planned/2026-09-03-tul-onset-capture.md).

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_onset_capture.py -v

Contracts, each chosen so the test FAILS when the shipped code breaks:

1. The cotangent hook (`_probe_cot`) records one norm per GRAD iteration of the slot loop,
   every value finite and positive, and leaves every parameter gradient BIT-IDENTICAL to
   the same forward/backward without the hook (the hook returns None).
2. The rank reading (`_probe_rank`) yields one entropy effective rank per iteration, each
   inside [1, min(n_active_slots, d_model)], and does not change the loss.
3. The pre-clip probe reader emits `loop/cot_norm_t*` and `loop/eff_rank_t*` rows from a
   real model after backward.
4. The batch-dump form of a SlotLayout round-trips through torch.save and rebuilds the
   dataclass (the trainer serialises `vars(layout)`).
5. The Jacobian probe dict carries the typical-gain keys next to sigma.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer
from morph.model.tul_layout import SlotLayout

MAX_DEPTH = 3


def _model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, cot: bool, rank: bool, seed: int = 7):
    x, y, layout, _ = _batch()
    m._probe_loop = True
    m._probe_cot = cot
    m._probe_rank = rank
    torch.manual_seed(seed)          # pins the Poisson depth draw
    m.eval()                         # dropout off; graph still built
    out = m(x, labels=y, slot_layout=layout)
    out["loss"].backward()
    grads = [p.grad.detach().clone() if p.grad is not None else None for p in m.parameters()]
    return out, grads, layout


# ── 1. the cotangent hook ─────────────────────────────────────────────────────

def test_cot_hook_records_every_grad_iteration_and_leaves_grads_bit_identical():
    m_hook = _model(seed=3)
    m_ref = _model(seed=3)
    out_h, g_h, _ = _run(m_hook, cot=True, rank=False)
    out_r, g_r, _ = _run(m_ref, cot=False, rank=False)
    assert torch.equal(out_h["loss"], out_r["loss"])
    for a, b in zip(g_h, g_r):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b), "the cotangent hook changed a gradient"
    cot = m_hook._loop_cot
    n_grad_iters = int(m_hook._loop_probe["core_gain"].numel())   # every iteration is a grad iteration at full BPTT
    assert sorted(cot) == list(range(n_grad_iters)), (sorted(cot), n_grad_iters)
    for t, v in cot.items():
        v = float(v)
        assert v == v and v > 0.0, f"cotangent at iteration {t} is {v}"
    assert not m_ref._loop_cot, "no hook was armed, so nothing may be recorded"


# ── 2. the rank reading ───────────────────────────────────────────────────────

def test_eff_rank_is_one_per_iteration_inside_its_bounds_and_does_not_move_the_loss():
    m_rank = _model(seed=5)
    m_ref = _model(seed=5)
    out_k, _, layout = _run(m_rank, cot=False, rank=True)
    out_r, _, _ = _run(m_ref, cot=False, rank=False)
    assert torch.equal(out_k["loss"], out_r["loss"])
    er = m_rank._loop_probe["eff_rank"]
    assert er is not None and er.numel() == m_rank._loop_probe["core_gain"].numel()
    n_slots = int(layout.slot_valid.numel())      # active set ⊆ all slot positions
    d = m_rank.cfg.d_model
    for v in er.tolist():
        assert 1.0 - 1e-4 <= v <= min(n_slots, d) + 1e-4, v
    assert m_ref._loop_probe["eff_rank"] is None


# ── 3. the probe reader ───────────────────────────────────────────────────────

def test_preclip_probe_reader_emits_cot_and_rank_rows():
    from morph.training.train import _preclip_probe
    m = _model(seed=9)
    _run(m, cot=True, rank=True)
    row = _preclip_probe(m)
    cot_keys = sorted(k for k in row if k.startswith("loop/cot_norm_t"))
    rank_keys = sorted(k for k in row if k.startswith("loop/eff_rank_t"))
    n = int(m._loop_probe["core_gain"].numel())
    assert cot_keys == [f"loop/cot_norm_t{t}" for t in range(n)]
    assert rank_keys == [f"loop/eff_rank_t{t}" for t in range(n)]
    assert "loop/delta_mean_t0" in row and row["loop/delta_mean_t0"] > 0.0


# ── 4. the batch dump form ────────────────────────────────────────────────────

def test_slot_layout_dump_round_trips():
    _, _, layout, _ = _batch()
    d = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(layout).items()}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "batch_000001.pt")
        torch.save({"step": 1, "layout": d}, p)
        back = torch.load(p, weights_only=False)
    rebuilt = SlotLayout(**back["layout"])
    for k, v in vars(layout).items():
        w = getattr(rebuilt, k)
        if torch.is_tensor(v):
            assert torch.equal(v.cpu(), w)
        else:
            assert v == w


# ── 5. the Jacobian dict ──────────────────────────────────────────────────────

def test_jacobian_probe_dict_carries_typical_gain():
    from morph.training.core_jacobian import CoreJacobianProbe
    from morph.training.train import _jacobian_probe
    m = _model(seed=11)
    x, y, layout, _ = _batch()
    m.eval()
    probe = CoreJacobianProbe(m, n_iter=4, seed=0, per_block=True)
    row = _jacobian_probe(m, probe, x, y, layout, 0, [0])   # bag_size 0 = no TST bag
    assert "jac/sigma_t0" in row and "jac/rms_t0" in row
    assert row["jac/rms_t0"] <= row["jac/sigma_t0"] + 1e-6
    assert "jac/rms_blockgeo_t0" in row and "jac/rms_b0_t0" in row
