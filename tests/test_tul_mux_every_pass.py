"""The M-next MUX on EVERY pass of the slot loop (``tul.mux_every_pass``).

The slot loop gets gradient only at its EXIT state: the token CE through the coda and the
two prefix cells, and ONE MUX term computed on that exit state. No intermediate pass
carries a loss. This knob puts the configured MUX target on the state after every pass of
a LIVE carry — no detach anywhere — for the slots whose realised depth reaches that pass,
plus the final state for every valid slot, averaged with weights summing to 1 so
``mux_beta`` keeps its meaning.

One test per contract:

  1. ``mux_every_pass = False`` is the forward from before the knob existed. The identity
     against the PRE-CHANGE tree was checked directly, off the test tree: this tiny model
     at seed 3/7 gives loss ``9.30903148651123`` and sha256
     ``ee9ffa170414e8311cf714b65b989bfe1bfbf240906466e2d959b213e447bea2`` over all 208
     gradient tensors both at master ``f4a284f`` and with the knob off after the change.
     This file keeps the loss + gradient-hash contract that would break if the mechanism
     ever leaked into the off path, and a sensitivity test so it cannot pass vacuously.
  2. the loss IS the mean of the per-pass terms: recomputed by hand from a hooked
     trajectory at a fixed depth, to 1e-6.
  3. the per-pass terms reach the shared core weights — a MUX-only backward gives
     different core gradients with the knob on and off.
  4. the masks: a slot of realised depth 2 is supervised at pass 1 and pass 2 and NOT at
     pass 3, and a progressive no-grad prefix carries no term.
  5. the knob is refused where a per-pass term has no meaning.

CPU only, tiny config, no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)

from morph.model.transformer import MORPHConfig, MORPHTransformer  # noqa: F401

MAX_DEPTH = 4


def _model(seed: int = 3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    tul = _tul(tg_restrict=False, sigreg_lambda=0.0, mux_beta=1.0, mux_target="next",
               mux_detach_head=False, **tul_kw)
    base = dict(tul=tul, n_core=2, mean_depth=MAX_DEPTH, max_depth=MAX_DEPTH,
                bptt_depth=MAX_DEPTH, retention=False, dropout=0.1,
                core_fixed_point_lambda=1.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _core_spy(m: MORPHTransformer) -> dict:
    """Capture everything ``_tul_core`` returned, so a term can be recomputed by hand."""
    seen: dict = {}
    orig = m._tul_core

    def spy(*a, **k):
        res = orig(*a, **k)
        seen.update(xn=res[0], h_slots=res[1], depths=res[2].detach().clone(),
                    traj=res[4], keep=res[6])
        return res

    m._tul_core = spy
    return seen


def _fixed_depths(m: MORPHTransformer, per_slot):
    """Replace the Poisson draw with an explicit per-slot depth.

    ``per_slot(layout)`` returns a ``[B, S]`` long tensor for the VALID slots; pads are
    forced to 1 exactly as ``_sample_slot_depths`` does, so they never inflate the loop.
    """
    def draw(layout, device):
        d = per_slot(layout).to(device)
        return torch.where(layout.slot_valid, d, torch.ones_like(d))

    m._sample_slot_depths = draw


def _run(m: MORPHTransformer, *, seed: int = 7):
    x, y, layout, _ = _batch()
    m.train()
    torch.manual_seed(seed)
    out = m(x, labels=y, slot_layout=layout)
    return out, x, layout


def _grads(m: MORPHTransformer):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}


def _same(ga, gb) -> bool:
    return all(((ga[k] is None) == (gb[k] is None))
               and (ga[k] is None or torch.equal(ga[k], gb[k])) for k in ga)


# ── 1. off is the forward from before the knob ───────────────────────────────

def test_off_is_bit_identical_loss_and_every_gradient():
    m_a = _model()
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    rng_a = torch.get_rng_state()

    m_b = _model(tul_mux_every_pass=False)
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()

    assert torch.equal(rng_a, torch.get_rng_state()), "the off path consumed random numbers"
    assert torch.equal(out_a["loss"], out_b["loss"])
    assert _same(_grads(m_a), _grads(m_b)), "the off path changed a gradient"
    assert m_b._loop_mux is None, "the off path wrote the per-pass counters"
    assert "mux_pass_terms" not in out_b


def test_the_check_is_sensitive_on_changes_the_loss_and_the_gradients():
    """Guards the test above: were the mechanism a no-op, it would pass for the wrong
    reason."""
    m_a = _model(tul_mux_every_pass=False)
    out_a, _, _ = _run(m_a)
    out_a["loss"].backward()
    m_b = _model(tul_mux_every_pass=True)
    out_b, _, _ = _run(m_b)
    out_b["loss"].backward()
    assert not torch.equal(out_a["mux_local"], out_b["mux_local"])
    assert not _same(_grads(m_a), _grads(m_b))


def test_eval_keeps_the_single_final_term():
    """`core_depth_sweep.py` forces a depth and reads `mux_local` off the FINAL state. The
    knob must not change that column, so the every-pass ladder is training only."""
    m_on, m_off = _model(tul_mux_every_pass=True), _model(tul_mux_every_pass=False)
    x, y, layout, _ = _batch()
    outs = []
    for m in (m_on, m_off):
        m.eval()
        torch.manual_seed(7)
        with torch.no_grad():
            outs.append(m(x, labels=y, slot_layout=layout))
    assert torch.equal(outs[0]["mux_local"], outs[1]["mux_local"])
    assert "mux_pass_terms" not in outs[0]


# ── 2. the loss is the mean of the per-pass terms ────────────────────────────

def test_the_loss_is_the_mean_of_the_per_pass_terms_at_a_fixed_depth():
    """At a fixed depth of 3 the trajectory holds three distinct post-pass states. The
    loss is the mean of four terms — one per pass, on the slots whose depth reaches it,
    plus the final state on every valid slot (the ruler's own term). Pass 3 and the final
    term read the SAME state and differ only in their mask, which is why the count is
    four and not three."""
    m = _model(tul_mux_every_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, lambda lay: torch.full(lay.slot_valid.shape, 3, dtype=torch.long))
    seen = _core_spy(m)
    out, x, layout = _run(m)

    traj, keep, h = seen["traj"], seen["keep"], seen["h_slots"]
    assert len(traj) == 4 and len(keep) == 3, (len(traj), len(keep))
    assert len({id(t) for t in traj}) == 4
    assert torch.equal(traj[-1], h), "the final state is not the loop's exit state"

    terms = [m._tul_mux_loss(traj[j], x, layout, slot_keep=keep[j - 1])
             for j in range(1, len(traj))]
    terms.append(m._tul_mux_loss(h, x, layout))
    assert len(terms) == 4
    assert torch.allclose(out["mux_local"], torch.stack(terms).mean(), atol=1e-6)
    assert float(out["mux_pass_terms"]) == 4.0
    # The weights sum to 1, so `mux_beta` keeps its meaning: the mean of the terms is
    # bounded by their min and max, not by their sum.
    _vals = [float(t.detach()) for t in terms]
    assert min(_vals) <= float(out["mux_local"]) <= max(_vals)


def test_the_counters_report_every_term():
    m = _model(tul_mux_every_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, lambda lay: torch.full(lay.slot_valid.shape, 3, dtype=torch.long))
    seen = _core_spy(m)
    out, x, layout = _run(m)
    assert set(m._loop_mux) == {"mux_pass_terms", "mux_pass_t1", "mux_pass_t2",
                                "mux_pass_t3", "mux_pass_final"}
    assert m._loop_mux["mux_pass_terms"] == 4.0
    traj, keep = seen["traj"], seen["keep"]
    for j in (1, 2, 3):
        by_hand = m._tul_mux_loss(traj[j], x, layout, slot_keep=keep[j - 1])
        assert float(m._loop_mux[f"mux_pass_t{j}"]) == pytest.approx(
            float(by_hand.detach()), rel=1e-6)
    assert float(m._loop_mux["mux_pass_final"]) == pytest.approx(
        float(m._tul_mux_loss(seen["h_slots"], x, layout).detach()), rel=1e-6)


def test_the_carry_stays_live_every_pass_state_keeps_its_graph():
    """The contract that separates this arm from `db_loop`: nothing is detached, so the
    gradient of the final state with respect to pass 1's state is nonzero."""
    m = _model(tul_mux_every_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    seen = _core_spy(m)
    _run(m)
    traj = seen["traj"]
    assert len(traj) >= 3
    for j in range(1, len(traj)):
        assert traj[j].grad_fn is not None, f"pass {j} lost its graph"
    g = torch.autograd.grad(traj[-1].float().pow(2).sum(), traj[1],
                            retain_graph=True, allow_unused=True)[0]
    assert g is not None and float(g.abs().sum()) > 0.0


# ── 3. the per-pass terms reach the shared core weights ──────────────────────

def test_a_mux_only_backward_moves_the_core_weights_differently():
    """Fixture sensitivity, on the quantity the arm exists to change. Both models hold the
    same weights and see the same batch at the same seed; only the MUX term is
    backpropagated, so any difference in the core gradients comes from the per-pass terms
    and from nothing else."""
    def core_grads(on: bool):
        m = _model(tul_mux_every_pass=on, slot_gain_lambda=0.0, dropout=0.0)
        out, _, _ = _run(m)
        m.zero_grad(set_to_none=True)
        out["mux_local_live"].backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if p.grad is not None and n.startswith("core.")}

    g_off, g_on = core_grads(False), core_grads(True)
    assert g_off and set(g_off) == set(g_on)
    changed = [n for n in g_off if not torch.allclose(g_off[n], g_on[n], atol=0, rtol=0)]
    assert len(changed) > 0.5 * len(g_off), (len(changed), len(g_off))


# ── 4. the masks ─────────────────────────────────────────────────────────────

def test_a_depth_two_slot_is_supervised_at_passes_one_and_two_and_not_three():
    """The mask contract. A slot's state stops moving at its realised depth; supervising
    its frozen state again at every later pass would over-weight shallow slots (the same
    reason `db_loop` masks its intermediate picks)."""
    def draw(lay):
        d = torch.full(lay.slot_valid.shape, 3, dtype=torch.long)
        d[0, 0] = 2                      # one shallow slot, deliberately
        return d

    m = _model(tul_mux_every_pass=True, slot_gain_lambda=0.0, dropout=0.0)
    _fixed_depths(m, draw)
    seen = _core_spy(m)
    _run(m)
    keep, depths, valid = seen["keep"], seen["depths"], None
    x, y, layout, _ = _batch()
    valid = layout.slot_valid
    assert int(depths[0, 0]) == 2 and bool(valid[0, 0]), "the fixture slot is not valid"
    assert bool(keep[0][0, 0]) and bool(keep[1][0, 0]), "a live pass lost its term"
    assert not bool(keep[2][0, 0]), "a frozen slot was supervised past its depth"
    # And the general rule the fixture is one instance of.
    for j, k in enumerate(keep, start=1):
        assert torch.equal(k, valid & (depths >= j))


def test_a_progressive_no_grad_prefix_carries_no_term():
    """`tul.progressive_p` is off on this arm. If it is ever on, a detached prefix pass
    must not carry a MUX term: the term would train nothing through that slot and still be
    averaged into the loss, silently shrinking every other term's weight."""
    m = _model(tul_mux_every_pass=True, tul_progressive_p=1.0, slot_gain_lambda=0.0,
               dropout=0.0)
    _fixed_depths(m, lambda lay: torch.full(lay.slot_valid.shape, 3, dtype=torch.long))
    seen = _core_spy(m)
    _run(m)
    keep, depths = seen["keep"], seen["depths"]
    k = m._loop_prog_k
    assert k is not None and int((k > 0).sum()) > 0, "no slot drew a prefix at p=1"
    _, _, layout, _ = _batch()
    valid = layout.slot_valid
    n_cut = 0
    for j, km in enumerate(keep, start=1):
        assert torch.equal(km, valid & (depths >= j) & (k < j))
        n_cut += int((valid & (depths >= j) & (k >= j)).sum())
    assert n_cut > 0, "the check is vacuous — no pass was cut"


# ── 5. refusals ──────────────────────────────────────────────────────────────

def test_the_paid_loop_refuses_the_knob():
    with pytest.raises(NotImplementedError, match="SLOT-LOOP lever"):
        _tul(mux_every_pass=True, mux_beta=1.0, tokens_through_core=True)


def test_a_zero_mux_beta_refuses_the_knob():
    with pytest.raises(ValueError, match="mux_beta"):
        _tul(mux_every_pass=True, mux_beta=0.0)


def test_db_loop_and_staged_targets_refuse_the_knob():
    with pytest.raises(ValueError, match="db_loop"):
        _tul(mux_every_pass=True, mux_beta=1.0, db_loop=True)
    with pytest.raises(ValueError, match="mux_stage_own_iters"):
        _tul(mux_every_pass=True, mux_beta=1.0, mux_stage_own_iters=2)


def test_the_think_once_stack_refuses_the_knob():
    with pytest.raises(NotImplementedError, match="cond_layers"):
        _tul(mux_every_pass=True, mux_beta=1.0, cond_layers=1)
