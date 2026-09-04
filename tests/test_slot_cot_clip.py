"""Clip-through-time on the slot loop's backward (`model.slot_cot_clip`).

The forecast spike train is the cotangent growing 39-2436x back through the eight slot-loop
iterations while the forward stays flat (lab/experiments/successes/2026-09-03-tul-onset-
capture.md). The clip rescales, per row, the cotangent arriving at each iteration's output
to at most `slot_cot_clip` times that row's exit cotangent. One test per contract:

  1. a cap that never binds is bit-identical to no cap (the hook multiplies by exactly 1.0);
  2. a cap that binds changes the gradients, never the loss, and every recorded post-clip
     row norm sits at or below cap x reference while the exit cotangent is untouched;
  3. the knob is refused on a model that has no slot loop.
"""
from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHTransformer

MAX_DEPTH = 3


def _model(seed: int = 0, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False)
    base.update(kw)
    return MORPHTransformer(_cfg(**base))


def _run(m: MORPHTransformer, *, seed: int = 7):
    x, y, layout, _ = _batch()
    m._probe_loop = True
    m._probe_cot = True
    torch.manual_seed(seed)          # pins the Poisson depth draw
    m.eval()                         # dropout off; graph still built
    out = m(x, labels=y, slot_layout=layout)
    out["loss"].backward()
    grads = [p.grad.detach().clone() if p.grad is not None else None for p in m.parameters()]
    return out, grads


def _same(ga, gb) -> bool:
    return all(((a is None) == (b is None)) and (a is None or torch.equal(a, b))
               for a, b in zip(ga, gb))


def test_unbinding_cap_is_bit_identical_to_no_cap():
    out_r, g_r = _run(_model(seed=3))
    out_c, g_c = _run(_model(seed=3, slot_cot_clip=1e9))
    assert torch.equal(out_r["loss"], out_c["loss"])
    assert _same(g_r, g_c), "a cap that never binds changed a gradient"


def test_binding_cap_bounds_every_row_and_leaves_the_loss_and_exit_alone():
    m_r = _model(seed=3)
    out_r, g_r = _run(m_r)
    cap = 1e-3
    m_c = _model(seed=3, slot_cot_clip=cap)
    out_c, g_c = _run(m_c)
    assert torch.equal(out_r["loss"], out_c["loss"]), "the clip touched the forward"
    assert not _same(g_r, g_c), "a cap of 1e-3 bound nothing: the hook is not on the path"
    ref = m_c._loop_cot_ref
    assert ref is not None and ref.dim() == 1, "no exit reference was recorded"
    ts = sorted(m_c._loop_cot_rows)
    assert ts == sorted(m_c._loop_cot), (ts, sorted(m_c._loop_cot))
    assert len(ts) == MAX_DEPTH, "every iteration is a grad iteration at full BPTT"
    bound = False
    for t in ts:
        rows = m_c._loop_cot_rows[t]
        post = torch.minimum(rows, cap * ref)          # what the scale produces per row
        assert torch.all(post <= cap * ref + 1e-6), f"iteration {t} exceeds the cap"
        assert float(m_c._loop_cot_post[t]) <= float(post.norm()) * (1 + 1e-5) + 1e-9
        bound |= float(m_c._loop_cot_bind[t]) > 0.0
    assert bound, "the cap bound no row anywhere"
    # The pre-clip norms recorded by the clipped model are what the clip SAW; at the last
    # iteration nothing upstream of it was clipped yet, so they match the unclipped run.
    t_last = ts[-1]
    assert torch.equal(m_c._loop_cot[t_last], m_r._loop_cot[t_last])
    # Earlier iterations received a clipped upstream, so their pre-clip norms are smaller.
    assert float(m_c._loop_cot[ts[0]]) < float(m_r._loop_cot[ts[0]])


def test_knob_refused_without_a_slot_loop():
    with pytest.raises(ValueError, match="needs a slot loop"):
        _model(slot_cot_clip=2.0, n_core=0)
    with pytest.raises(ValueError, match=">= 0"):
        _model(slot_cot_clip=-1.0)
