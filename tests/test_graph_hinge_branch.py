"""graph_step's hinge branch (2026-10-10, phase-2 tracker C4): the pieces under the replay test.

morph/training/graph_step.py records a step whose forward reports the gain hinge as
``pre -> IF gain_active: A ELSE: B -> post``; B is the backward with the penalty's edge left
out. tests/test_graph_step.py pins the whole step bit for bit against eager, with the hinge
always active and never active. This file pins the parts:

  1. ``_cut_roots`` (CPU): the roots of a loss chain without the penalty's edge. A backward
     from them gives EXACTLY (torch.equal) the gradient of the loss without the penalty, for
     a chain with adds, a ``/ 1``, a ``None`` addend (a constant) and a bf16 addend; the
     penalty's own nodes never run (a hook on them does not fire). A penalty under any other
     node (a multiply), or not in the chain, raises.
  2. ``_TensorAttrs`` (CPU): a tensor attribute and a dict of tensors set by the second body
     are copied into the first body's objects and rebound; a changed shape raises.
  3. ``CondBranch`` (CUDA): over 24 steps whose flag a preceding torch graph computes on the
     device, the branch runs exactly the body the flag names (each body writes its own
     result; no host read between the two launches); a non-bool flag raises.
  4. The model's flag (CPU): ``out["gain_active"]`` is False when no row's gain is over the
     target, True when one is, and a NaN gain reads active.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from morph.training.graph_step import _cut_roots, _TensorAttrs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")


def _chain(w):
    """loss = ((ce + pen) + aux + const + bf16_term) / 1, every term reading w."""
    ce = (w * w).sum()
    pen = 100.0 * torch.relu(w.exp().mean() - 0.5) ** 2
    aux = 0.3 * w.sum()
    const = torch.zeros(())                                    # no grad: a None edge
    b16 = (w.to(torch.bfloat16) * 2).sum().float()
    loss = (((ce + pen) + aux) + const) + b16
    return ce, pen, aux, b16, loss / 1


def test_cut_roots_backward_is_the_loss_without_the_penalty():
    w = torch.randn(64, dtype=torch.float32, requires_grad=True)
    ce, pen, aux, b16, l1 = _chain(w)
    fired = []
    pen.grad_fn.register_prehook(lambda g: fired.append(1))
    roots = _cut_roots(l1, pen)
    assert len(roots) == 3                                    # ce, aux, b16 (const: None)
    torch.autograd.backward(roots, grad_tensors=[torch.ones(())] * len(roots))
    got = w.grad.clone()
    assert fired == []                                        # the penalty's nodes never ran
    w2 = w.detach().clone().requires_grad_(True)
    ce2, pen2, aux2, b162, _ = _chain(w2)
    (((ce2 + aux2) + torch.zeros(())) + b162).backward()
    assert torch.equal(got, w2.grad)
    # and the full backward does reach the penalty (its gradient is not 0 here)
    w3 = w.detach().clone().requires_grad_(True)
    _, pen3, _, _, l3 = _chain(w3)
    l3.backward()
    assert not torch.equal(w3.grad, got)


def test_cut_roots_refuses_a_penalty_outside_the_add_chain():
    w = torch.randn(8, requires_grad=True)
    pen = w.exp().sum()
    with pytest.raises(RuntimeError, match="not an addend"):
        _cut_roots(((w * w).sum() + 2.0 * pen) / 1, pen)      # under a multiply
    with pytest.raises(RuntimeError, match="not an addend"):
        _cut_roots((w * w).sum() / 1, pen)                    # not in the chain


def test_tensor_attrs_copy_the_second_bodys_values_into_the_first_bodys_objects():
    m = nn.Linear(2, 2)
    m.ref = torch.ones(3)
    m.table = {0: torch.zeros(2), 1: torch.ones(2)}
    first_ref, first_t0 = m.ref, m.table[0]
    snap = _TensorAttrs([m])
    m.ref = torch.full((3,), 7.0)                             # the second body's objects
    m.table[0] = torch.full((2,), 5.0)
    snap.restore()
    assert m.ref is first_ref and torch.equal(first_ref, torch.full((3,), 7.0))
    assert m.table[0] is first_t0 and torch.equal(first_t0, torch.full((2,), 5.0))
    assert torch.equal(m.table[1], torch.ones(2))
    m.ref = torch.ones(4)
    with pytest.raises(RuntimeError, match="set differently"):
        snap.restore()


@cuda
def test_cond_branch_runs_the_body_the_device_flag_names():
    from morph.training.graph_cond import CondBranch
    s = torch.cuda.Stream()
    pool = torch.cuda.MemPool()
    x = torch.zeros(4096, device="cuda")
    out = torch.zeros(4096, device="cuda")
    with torch.cuda.stream(s):
        x.normal_()
        out.zero_()
    torch.cuda.synchronize()
    g_pre = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_pre, pool=pool.id, stream=s):
        flag = x.sum() > 0
    bodies = []
    for fn in (lambda: out.copy_(x * 2.0), lambda: out.copy_(x + 1.0)):
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g, pool=pool.id, stream=s):
            fn()
        bodies.append(g)
    br = CondBranch(flag, bodies[0], bodies[1])
    try:
        seen = []
        for i in range(24):
            x.copy_(torch.randn(4096, device="cuda") + (2.0 if i % 3 else -2.0))
            g_pre.replay()
            br.launch()
            torch.cuda.synchronize()
            pos = float(x.sum()) > 0
            seen.append(pos)
            assert torch.equal(out, x * 2.0 if pos else x + 1.0), i
        assert set(seen) == {True, False}
    finally:
        br.close()
    with pytest.raises(ValueError, match="one CUDA bool"):
        CondBranch(x.sum(), bodies[0], bodies[1])


def test_model_flag_reads_inactive_active_and_nan():
    from test_gain_pass_draw import _fwd, _model
    for target, want in ((100.0, False), (0.0, True)):
        m = _model(seed=3, slot_gain_target=target)
        out, _t, _ = _fwd(m)
        assert out["gain_active"].dtype == torch.bool and bool(out["gain_active"]) is want
        assert out["gain_pen"].requires_grad
        assert (float(out["gain_pen"]) > 0) is want
    m = _model(seed=3, slot_gain_target=float("nan"))           # every hinge reads NaN
    out, _t, _ = _fwd(m)
    assert torch.isnan(out["gain_pen"]) and bool(out["gain_active"]) is True
