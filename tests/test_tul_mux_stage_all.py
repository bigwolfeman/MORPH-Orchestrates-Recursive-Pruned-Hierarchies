"""Staged targets at EVERY non-final pass (``tul.mux_stage_all``).

``tul.mux_stage_own_iters`` puts the own-span (memory) target on ONE intermediate pass k
and the next-span (forecast) target on the exit state. This knob changes what k means: the
own target is applied at every pass from k to T-1 — k is then the FIRST supervised pass —
on the slots whose realised depth reaches that pass, and the own terms are averaged. The
loss stays ``0.5 * (own + next)``, so ``mux_beta`` keeps its meaning and the reported
``mux_local`` still comes from the final forecast term.

Why: the toy study (``lab/toy_slot_loop/WRITEUP.md``) found this exact attachment is the
only one that solves an iteration-requiring chain (5/5 seeds; exit-only 2/5, the same
target at every pass 1/5). ``mux_stage_own_iters`` alone expresses the narrower k=3 form.

One test per contract:

  1. ``mux_stage_all = False`` is the forward from before the knob existed. Checked
     directly against the PRE-CHANGE tree, not by reading the diff: the tiny model at
     seed 3/7 with ``mux_stage_own_iters=2`` gives loss ``9.359314918518066`` and sha256
     ``189911591a8f60ee2f9dee7932461bca63b2bf5433d2355270763fe0461aefc0`` over all 208
     gradient tensors both at master ``fe42d85`` and with the knob off after the change
     (the knob ON gives ``9.340977668762207`` / ``1a10728e…5b77f7``). This file keeps the
     loss + gradient contract that would break if the mechanism leaked into the off path,
     and a sensitivity test so it cannot pass vacuously.
  2. the loss IS ``0.5 * (mean of the own terms) + 0.5 * next``, recomputed by hand from a
     hooked trajectory at a fixed depth, to 1e-6.
  3. the masks: a slot supervised at pass j has realised depth >= j, and a shallow slot
     drops out of the later passes.
  4. an eval forward is bit-identical to the single-k arm — the knob is TRAINING only, so
     the depth sweep's two final-state columns are unchanged.
  5. the own terms reach the shared core weights.
  6. the knob is refused where it has no meaning.

CPU only, tiny config, no tokenizer. The helpers come from ``test_tul_mux_every_pass``
(the same trajectory and the same masks feed both knobs) so the two files cannot drift.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _tul                      # noqa: E402 (tests/ on sys.path)
from test_tul_mux_every_pass import (                      # noqa: E402
    MAX_DEPTH, _core_spy, _fixed_depths, _grads, _model, _run, _same,
)


def _stage(k: int = 1, all_: bool = True, **kw):
    return _model(tul_mux_stage_own_iters=k, tul_mux_stage_all=all_, **kw)


# ── 1. off is the forward from before the knob ───────────────────────────────

def test_off_is_bit_identical_loss_and_every_gradient():
    """The identity contract. `mux_stage_all=False` must trace the single-k graph that
    arc E3 built, down to every gradient tensor."""
    m_a = _stage(k=2, all_=False)
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    rng_a = torch.get_rng_state()

    m_b = _model(tul_mux_stage_own_iters=2)          # the knob left at its default
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()

    assert torch.equal(rng_a, torch.get_rng_state()), "the off path consumed random numbers"
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert _same(_grads(m_a), _grads(m_b)), "the off path changed a gradient"
    assert m_b._loop_mux is None, "the off path wrote the per-pass counters"
    assert "mux_stage_terms" not in out_b


def test_the_check_is_sensitive_on_changes_the_loss_and_the_gradients():
    """Guards the test above: were the mechanism a no-op, it would pass for the wrong
    reason."""
    m_a = _stage(k=1, all_=False, slot_gain_lambda=0.0, dropout=0.0)
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    m_b = _stage(k=1, all_=True, slot_gain_lambda=0.0, dropout=0.0)
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()
    assert not torch.equal(out_a["mux_local"], out_b["mux_local"])
    assert not _same(_grads(m_a), _grads(m_b))


# ── 2. the loss is 0.5 * mean(own terms) + 0.5 * next ────────────────────────

def test_the_loss_is_the_mean_of_the_own_terms_plus_the_final_next_term():
    """At a fixed depth of 4 with k=1 the own side is the mean of the terms at passes 1, 2
    and 3 — the FINAL state (pass 4) belongs to the forecast, never to the memory."""
    m = _stage(k=1, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, lambda lay: torch.full(lay.slot_valid.shape, MAX_DEPTH,
                                            dtype=torch.long))
    seen = _core_spy(m)
    out, x, layout = _run(m)

    traj, keep, h = seen["traj"], seen["keep"], seen["h_slots"]
    assert len(traj) == MAX_DEPTH + 1 and len(keep) == MAX_DEPTH
    assert torch.equal(traj[-1], h), "the final state is not the loop's exit state"

    own_terms = [m._tul_mux_loss(traj[j], x, layout, slot_keep=keep[j - 1], target="own")
                 for j in (1, 2, 3)]
    nxt = m._tul_mux_loss(h, x, layout)
    expect = 0.5 * (torch.stack(own_terms).mean() + nxt)
    assert torch.allclose(out["mux_local"], expect, atol=1e-6)
    assert torch.allclose(out["mux_stage_own"], torch.stack(own_terms).mean(), atol=1e-6)
    assert torch.allclose(out["mux_stage_next"], nxt, atol=1e-6)
    assert float(out["mux_stage_terms"]) == 3.0


def _ladder(lay):
    """Depths 1, 2, 3, 4, 1, 2, ... across the slot axis — every own-term mask is then a
    PROPER subset of the valid slots, which is what makes the mask observable in the loss
    and not only in the returned mask tensor."""
    S = lay.slot_valid.shape[1]
    d = 1 + (torch.arange(S) % MAX_DEPTH)
    return d.unsqueeze(0).expand_as(lay.slot_valid).contiguous()


def test_the_loss_and_its_gradient_are_the_hand_built_staged_objective():
    """The sharp test, and the one the sabotage runs are aimed at. Under a ladder of
    realised depths the loss is rebuilt by hand from the hooked LIVE trajectory with the
    per-pass masks, and BOTH its value and its gradient on every core parameter must match
    the module's. Dropping the mask changes the value; detaching the trajectory leaves the
    value alone and changes the gradient."""
    m = _stage(k=1, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, _ladder)
    seen = _core_spy(m)
    out, x, layout = _run(m)
    traj, keep, h, depths = seen["traj"], seen["keep"], seen["h_slots"], seen["depths"]

    idxs = list(range(1, len(traj) - 1))
    assert len(idxs) >= 2, "the fixture did not produce a deep enough batch"
    n_valid = int(layout.slot_valid.sum())
    sizes = [int(keep[j - 1].sum()) for j in idxs]
    assert min(sizes) > 0, sizes                     # no term is empty
    assert min(sizes) < n_valid, (sizes, n_valid)    # and at least one mask really cuts

    own = [m._tul_mux_loss(traj[j], x, layout, slot_keep=keep[j - 1], target="own")
           for j in idxs]
    expect = 0.5 * (torch.stack(own).mean() + m._tul_mux_loss(h, x, layout))
    assert torch.allclose(out["mux_local"], expect, atol=1e-6)

    m.zero_grad(set_to_none=True)
    out["mux_local_live"].backward(retain_graph=True)
    g_mod = {n: p.grad.detach().clone() for n, p in m.named_parameters()
             if p.grad is not None and n.startswith("core.")}
    m.zero_grad(set_to_none=True)
    expect.backward()
    g_hand = {n: p.grad.detach().clone() for n, p in m.named_parameters()
              if p.grad is not None and n.startswith("core.")}
    assert g_mod and set(g_mod) == set(g_hand)
    for n in g_mod:
        assert torch.allclose(g_mod[n], g_hand[n], rtol=1e-5, atol=1e-7), n
    # And the gradient is not trivially zero, or the comparison above says nothing.
    assert max(float(v.abs().max()) for v in g_mod.values()) > 0.0
    assert int(depths.max()) == MAX_DEPTH


def test_k_selects_the_first_supervised_pass():
    """k is the FIRST supervised pass, not the only one: at k=3 and depth 4 exactly one own
    term survives, and it is pass 3."""
    m = _stage(k=3, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, lambda lay: torch.full(lay.slot_valid.shape, MAX_DEPTH,
                                            dtype=torch.long))
    seen = _core_spy(m)
    out, x, layout = _run(m)
    own = m._tul_mux_loss(seen["traj"][3], x, layout, slot_keep=seen["keep"][2],
                          target="own")
    assert float(out["mux_stage_terms"]) == 1.0
    assert torch.allclose(out["mux_stage_own"], own, atol=1e-6)


def test_the_counters_report_every_own_term():
    m = _stage(k=1, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, lambda lay: torch.full(lay.slot_valid.shape, MAX_DEPTH,
                                            dtype=torch.long))
    seen = _core_spy(m)
    out, x, layout = _run(m)
    assert set(m._loop_mux) == {"mux_stage_terms", "mux_stage_t1", "mux_stage_t2",
                                "mux_stage_t3"}
    assert m._loop_mux["mux_stage_terms"] == 3.0
    for j in (1, 2, 3):
        by_hand = m._tul_mux_loss(seen["traj"][j], x, layout, slot_keep=seen["keep"][j - 1],
                                  target="own")
        assert float(m._loop_mux[f"mux_stage_t{j}"]) == pytest.approx(
            float(by_hand.detach()), rel=1e-6)


# ── 3. the masks ─────────────────────────────────────────────────────────────

def test_a_shallow_slot_drops_out_of_the_passes_it_never_reaches():
    """A slot's state stops moving at its realised depth. Supervising the frozen state
    again at every later pass would over-weight shallow slots — the same rule
    `mux_every_pass` and `db_loop` carry. At k=2 a depth-2 slot is supervised at pass 2 and
    NOT at pass 3."""
    def draw(lay):
        d = torch.full(lay.slot_valid.shape, MAX_DEPTH, dtype=torch.long)
        d[0, 0] = 2                      # one shallow slot, deliberately
        return d

    m = _stage(k=2, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, draw)
    seen = _core_spy(m)
    out, _, _ = _run(m)
    keep, depths = seen["keep"], seen["depths"]
    _, _, layout, _ = _batch()
    valid = layout.slot_valid
    assert int(depths[0, 0]) == 2 and bool(valid[0, 0]), "the fixture slot is not valid"
    assert float(out["mux_stage_terms"]) == 2.0          # passes 2 and 3
    assert bool(keep[1][0, 0]), "the shallow slot lost its pass-2 term"
    assert not bool(keep[2][0, 0]), "a frozen slot was supervised past its depth"
    for j, km in enumerate(keep, start=1):
        assert torch.equal(km, valid & (depths >= j))


# ── 4. eval is the single-k arm ──────────────────────────────────────────────

def test_eval_falls_back_to_the_single_k_own_term():
    """`core_depth_sweep.py` reads `mux_local_own_final` / `mux_local_next_final` off the
    FINAL state at a forced depth. The knob is training-only, so an eval forward must be
    the single-k arm's, tensor for tensor — the arm must not change its own readout."""
    m_on, m_off = _stage(k=1, all_=True), _stage(k=1, all_=False)
    x, y, layout, _ = _batch()
    outs = []
    for m in (m_on, m_off):
        m.eval()
        torch.manual_seed(7)
        with torch.no_grad():
            outs.append(m(x, labels=y, slot_layout=layout))
    for key in ("mux_local", "mux_stage_own", "mux_stage_next",
                "mux_local_own_final", "mux_local_next_final"):
        assert torch.equal(outs[0][key], outs[1][key]), key
    assert "mux_stage_terms" not in outs[0]


# ── 5. the own terms reach the shared core weights ───────────────────────────

def test_a_mux_only_backward_moves_the_core_weights_differently():
    """Fixture sensitivity on the quantity the arm exists to change. Both models hold the
    same weights and see the same batch at the same seed; only the MUX term is
    backpropagated, so any difference in the core gradients comes from the extra own terms
    and from nothing else."""
    def core_grads(all_: bool):
        m = _stage(k=1, all_=all_, slot_gain_lambda=0.0, dropout=0.0)
        out, _, _ = _run(m)
        m.zero_grad(set_to_none=True)
        out["mux_local_live"].backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if p.grad is not None and n.startswith("core.")}

    g_off, g_on = core_grads(False), core_grads(True)
    assert g_off and set(g_off) == set(g_on)
    changed = [n for n in g_off if not torch.allclose(g_off[n], g_on[n], atol=0, rtol=0)]
    assert len(changed) > 0.5 * len(g_off), (len(changed), len(g_off))


# ── 6. refusals ──────────────────────────────────────────────────────────────

def test_the_knob_needs_a_first_pass():
    with pytest.raises(ValueError, match="mux_stage_own_iters > 0"):
        _tul(mux_stage_all=True, mux_beta=1.0)


def test_the_per_pass_mux_refuses_the_knob():
    with pytest.raises(ValueError, match="mux_every_pass"):
        _tul(mux_stage_all=True, mux_stage_own_iters=2, mux_beta=1.0, mux_every_pass=True)


@pytest.mark.parametrize("kw,pat", [(dict(db_loop=True), "db_loop"),
                                    (dict(tokens_through_core=True), "slot-loop lever"),
                                    (dict(mux_beta=0.0), "mux_beta")])
def test_the_stage_refusals_cover_the_modifier(kw, pat):
    """`mux_stage_all` is a modifier of `mux_stage_own_iters`, which is required > 0, so
    that knob's own refusals are the modifier's too — checked here rather than duplicated
    in the validator."""
    base = dict(mux_stage_all=True, mux_stage_own_iters=2, mux_beta=1.0)
    base.update(kw)
    with pytest.raises((ValueError, NotImplementedError), match=pat):
        _tul(**base)
