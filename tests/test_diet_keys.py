"""Two FLOP-cutting keys that change the training objective on purpose (2026-10-13, "dietkeys").

  * ``tul.spandec_slot_keep`` (morph/model/tul_spandec.py ``slot_keep_mask``, the packed branch of
    ``MORPHTransformer._tul_spandec_loss``): the packed span decoder trains on a random subset of
    each row's slots, whole spans, up to a per-row token budget of keep x the row's graded tokens.
  * ``model.slot_gain_every`` (``_gain_cadence_on``, ``gain_pass_choices`` / ``draw_gain_pass`` /
    ``_gain_pass_for_step``): the one-pass gain hinge runs on every k-th step, lambda x k.

What each test pins:
  A1. The mask: kept slots are whole spans, a prefix of the row's random order, at most the row's
      budget and within one span of it; it differs from step to step and row to row, repeats for
      the same (seed, counter) and moves with the seed. (SABOTAGE: an all-True mask is caught.)
  A2. The packed capacity and rows shrink by keep, training only; eval keeps every slot.
  A3. The loss is the mean over the KEPT tokens, and every gradient equals the per-slot decoder's
      on the same kept spans (the reference run feeds the recorded mask to the per-slot path).
  A4. The counter advances once per training forward (not in eval), so a graph replay draws on.
  A5. A row over the cap's proof is a NaN loss, not a silent drop; refusals; the build off.
  B1. The cadence: the hinge's forward runs on steps 0, k, 2k... only; off steps call nothing.
  B2. On-steps charge lambda x k (the penalty is exactly k x the every-1 model's).
  B3. GraphStep's variants: the choices end with -1, the host draw follows the same cadence, a
      forward pinned to -1 runs no hinge. Refusals; the keys are wired.

CPU, fp32.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

import morph.model.transformer as tr
import morph.model.tul_spandec as sd
from morph.model.transformer import GAIN_OFF
from test_gain_pass_draw import MAX_DEPTH, _fwd, _model as _gain_model
from test_spandec_packed import (J_SMALL, M, _assert_same, _batch3, _model as _sd_model,
                                 _randomise_zero_inits, _run)
from test_tul_fan_lsel import _build, _nowta_kw

KEEP = 0.5
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _keep_model(keep: float = KEEP, packed: bool = True, **kw):
    return _build(model_kw={"spandec_packed": packed},
                  **_nowta_kw(fan_loop_select="off", spandec_reads_cells=True,
                              spandec_max_tokens=J_SMALL, spandec_slot_keep=keep, **kw))


def _valid3() -> torch.Tensor:
    inp, _lab, layout = _batch3()
    return sd.horizon_span_slots(inp, layout, J_SMALL, 1, start=1)[1]


# ── A1. the mask ────────────────────────────────────────────────────────────────────

def _check_budget(valid, kept, q):
    """Raise AssertionError unless ``kept`` is whole spans within each row's token budget and
    within one span of it (the prefix cut stops at the first span that does not fit)."""
    n = valid.sum(-1)
    assert not bool((kept & (n == 0)).any()), "kept an empty slot"
    for b in range(valid.shape[0]):
        budget = int(sd.keep_scaled(int(n[b].sum()), q))
        got = int(n[b][kept[b]].sum())
        assert got <= budget, (b, got, budget)
        assert got > budget - int(n[b].max()), (b, got, budget)
        assert int(kept[b].sum()) < int((n[b] > 0).sum()), "every slot kept at keep < 1"


def test_mask_is_whole_spans_within_the_row_budget():
    valid = _valid3()
    q = sd.keep_q_of(KEEP)
    assert q == 512
    for c in range(8):
        kept = sd.slot_keep_mask(valid, q, 7, torch.tensor(c))
        _check_budget(valid, kept, q)


def test_mask_varies_per_step_and_row_and_is_reproducible():
    valid = _valid3()
    q = sd.keep_q_of(KEEP)
    masks = [sd.slot_keep_mask(valid, q, 7, torch.tensor(c)) for c in range(16)]
    assert len({m.numpy().tobytes() for m in masks}) >= 12, "the subset did not move per step"
    assert torch.equal(masks[3], sd.slot_keep_mask(valid, q, 7, torch.tensor(3)))
    other = [sd.slot_keep_mask(valid, q, 8, torch.tensor(c)) for c in range(16)]
    assert any(not torch.equal(a, b) for a, b in zip(masks, other)), "the seed does nothing"
    # each slot with a span is kept about keep of the time
    live = (valid.sum(-1) > 0)
    rate = (torch.stack(masks).float().sum(0) * live).sum() / (16 * live.sum())
    assert 0.3 < float(rate) < 0.7, float(rate)


def test_sabotaged_mask_is_caught():
    valid = _valid3()
    with pytest.raises(AssertionError):
        _check_budget(valid, valid.any(-1), sd.keep_q_of(KEEP))


# ── A2 / A3. capacity, rows, the loss ──────────────────────────────────────────────

def _spy_decode(m):
    seen = []
    orig = m.tul_spandec.decode_packed

    def spy(z, ids, valid, emb, cap, **kw):
        out = orig(z, ids, valid, emb, cap, **kw)
        seen.append(dict(cap=cap, rows=out[0].shape[1], n=int(valid.sum()), over=bool(out[2])))
        return out

    m.tul_spandec.decode_packed = spy
    return seen


def test_decoder_rows_and_tokens_shrink_in_training_only():
    inp, lab, layout = _batch3()
    full, on = _keep_model(1.0), _keep_model(KEEP)
    sf, so = _spy_decode(full), _spy_decode(on)
    for m in (full, on):
        m.train()
        m(inp, labels=lab, slot_layout=layout)
    cap0 = full._spandec_packed_cap(inp.shape[1], layout, full.tul_spandec)
    assert sf[0]["cap"] == cap0
    assert so[0]["cap"] == int(sd.keep_scaled(cap0, sd.keep_q_of(KEEP)))
    assert so[0]["rows"] < sf[0]["rows"] and so[0]["rows"] <= -(-so[0]["cap"] // J_SMALL) * J_SMALL
    assert 0 < so[0]["n"] < sf[0]["n"] and not so[0]["over"]
    # eval grades every slot at the full capacity
    on.eval()
    with torch.no_grad():
        on(inp, labels=lab, slot_layout=layout)
    assert so[1]["cap"] == cap0 and so[1]["n"] == sf[0]["n"]


def test_loss_is_the_mean_over_kept_tokens_and_grads_match_the_per_slot_decoder(monkeypatch):
    inp, lab, layout = _batch3()
    # the per-slot reference has no keep branch: it gets the recorded mask instead
    on, ref = _keep_model(KEEP), _sd_model(False)
    _randomise_zero_inits(ref)
    on.load_state_dict(ref.state_dict())
    rec = []
    orig = tr.slot_keep_mask

    def recorded(*a, **kw):
        rec.append(orig(*a, **kw))
        return rec[-1]

    monkeypatch.setattr(tr, "slot_keep_mask", recorded)
    got = _run(on, inp, lab, layout)
    assert len(rec) == 1 and int(rec[0].sum()) > 0
    orig_slots = tr.horizon_span_slots

    def masked(*a, **kw):
        ids, valid = orig_slots(*a, **kw)
        return ids, valid & rec[0].unsqueeze(-1)

    monkeypatch.setattr(tr, "horizon_span_slots", masked)
    want = _run(ref, inp, lab, layout)
    _assert_same(want, got)
    # and the kept tokens are a strict part of the batch
    valid = _valid3()
    assert 0 < int((valid & rec[0].unsqueeze(-1)).sum()) < int(valid.sum())


# ── A4. the counter ─────────────────────────────────────────────────────────────────

def test_counter_advances_per_training_forward_only_and_the_subset_moves():
    inp, lab, layout = _batch3()
    m = _keep_model(KEEP)
    seen, orig = [], tr.slot_keep_mask
    m.train()
    import unittest.mock as mock
    with mock.patch.object(tr, "slot_keep_mask",
                           lambda *a, **kw: seen.append(orig(*a, **kw)) or seen[-1]):
        for _ in range(4):
            m(inp, labels=lab, slot_layout=layout)
        assert int(m._spandec_keep_ctr) == 4
        m.eval()
        with torch.no_grad():
            m(inp, labels=lab, slot_layout=layout)
        assert int(m._spandec_keep_ctr) == 4
    assert len(seen) == 4 and len({s.numpy().tobytes() for s in seen}) >= 3


# ── A5. capacity proof, refusals, build ────────────────────────────────────────────

def test_over_the_row_cap_is_nan_not_a_silent_drop(monkeypatch):
    inp, lab, layout = _batch3()
    on = _keep_model(KEEP)
    monkeypatch.setattr(type(on), "_spandec_packed_cap", staticmethod(lambda L, lay, dec: 40))
    assert torch.isnan(_run(on, inp, lab, layout)[0])


def test_refusals_and_the_off_build():
    with pytest.raises(ValueError, match="needs model.spandec_packed"):
        _keep_model(KEEP, packed=False)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="spandec_slot_keep"):
            _keep_model(bad)
    with pytest.raises(ValueError, match="resolution"):
        _keep_model(0.9999)
    with pytest.raises(ValueError, match="spandec"):
        _build(**_nowta_kw(spandec=False, spandec_slot_keep=0.5))
    off, on = _keep_model(1.0), _keep_model(KEEP)
    assert not hasattr(off, "_spandec_keep_ctr") and hasattr(on, "_spandec_keep_ctr")
    assert off.state_dict().keys() == on.state_dict().keys()


# ── B. slot_gain_every ──────────────────────────────────────────────────────────────

def _spy_hinge(m):
    calls = []
    orig = m._slot_gain_penalty

    def spy(*a, **kw):
        calls.append(len(calls))
        return orig(*a, **kw)

    m._slot_gain_penalty = spy
    return calls


@pytest.mark.parametrize("k", [2, 3])
def test_hinge_runs_on_every_kth_step_only(k):
    m = _gain_model(seed=3, slot_gain_every=k, slot_gain_pass_force=2)
    calls = _spy_hinge(m)
    steps = []
    for i in range(3 * k + 1):
        before = len(calls)
        out, t, _ = _fwd(m, seed=7 + i, backward=True)
        steps.append((len(calls) - before, t, "gain_est" in out, "gain_pen" in out))
    for i, (n, t, has_gain, has_pen) in enumerate(steps):
        if i % k == 0:
            assert (n, t, has_gain, has_pen) == (1, 2, True, True), (i, steps)
        else:
            assert (n, t, has_gain, has_pen) == (0, GAIN_OFF, False, False), (i, steps)


def test_on_steps_charge_lambda_times_k_and_every_one_is_untouched():
    k = 4
    base = _gain_model(seed=3, slot_gain_pass_force=2)
    cad = _gain_model(seed=3, slot_gain_every=k, slot_gain_pass_force=2)
    cad.load_state_dict(base.state_dict())
    a, _, _ = _fwd(base, seed=7)
    b, _, _ = _fwd(cad, seed=7)                     # step 0 of the cadence: the hinge runs
    torch.testing.assert_close(b["gain_pen"], k * a["gain_pen"], rtol=1e-5, atol=0)
    torch.testing.assert_close(b["gain_est"], a["gain_est"], rtol=1e-6, atol=0)
    assert float(a["gain_pen"].detach()) > 0.0, "the hinge must bind for this test to mean anything"
    # an off step is the model without the hinge's loss term
    c, _, _ = _fwd(cad, seed=8)
    assert "gain_pen" not in c
    d, _, _ = _fwd(_gain_model(seed=3, slot_gain_lambda=0.0, slot_gain_target=0.9), seed=8)
    assert "gain_pen" not in d
    # tail / floor lambdas ride the same cadence factor
    t3 = _gain_model(seed=3, slot_gain_every=3, slot_gain_pass_force=2, slot_gain_tail_lambda=2.0,
                     slot_gain_tail_target=0.01)
    t1 = _gain_model(seed=3, slot_gain_pass_force=2, slot_gain_tail_lambda=2.0,
                     slot_gain_tail_target=0.01)
    t3.load_state_dict(t1.state_dict())
    o1, o3 = _fwd(t1, seed=7)[0], _fwd(t3, seed=7)[0]
    torch.testing.assert_close(o3["gain_tail_pen"], 3 * o1["gain_tail_pen"], rtol=1e-5, atol=0)
    assert float(o1["gain_tail_pen"]) > 0.0


def test_graph_step_variants_end_with_the_off_choice_and_follow_the_cadence():
    m = _gain_model(seed=3, slot_gain_every=3)
    assert m.gain_pass_choices() == tuple(range(MAX_DEPTH)) + (GAIN_OFF,)
    assert _gain_model(seed=3, slot_gain_every=3,
                       slot_gain_pass_force=1).gain_pass_choices() == (1, GAIN_OFF)
    assert _gain_model(seed=3).gain_pass_choices() == tuple(range(MAX_DEPTH))
    draws = [m.draw_gain_pass() for _ in range(9)]
    assert [d == GAIN_OFF for d in draws] == [False, True, True] * 3, draws
    assert all(0 <= d < MAX_DEPTH for d in draws if d != GAIN_OFF)
    # the host draw is the eager cadence: same pass on the same steps, same generator
    e = _gain_model(seed=3, slot_gain_every=3)
    eager = [_fwd(e, seed=7 + i)[1] for i in range(9)]
    assert eager == draws, (eager, draws)
    # a forward pinned to the off choice runs no hinge and does not advance the cadence
    p = _gain_model(seed=3, slot_gain_every=3)
    calls = _spy_hinge(p)
    p.pin_gain_pass(GAIN_OFF)
    out, t, _ = _fwd(p)
    assert t is None and not calls and "gain_pen" not in out and p._gain_cad_ctr == 0
    p.pin_gain_pass(1)
    out, _, _ = _fwd(p)
    assert len(calls) == 1 and "gain_pen" in out


@pytest.mark.parametrize("kw,needle", [
    (dict(slot_gain_every=0), "must be >= 1"),
    (dict(slot_gain_every=2, slot_gain_lambda=0.0, slot_gain_target=0.9), "needs model.slot_gain_lambda"),
    (dict(slot_gain_every=2, slot_gain_all_iters=True), "needs model.slot_gain_lambda"),
])
def test_every_is_refused_when_inert_or_out_of_range(kw, needle):
    with pytest.raises(ValueError, match=needle):
        _gain_model(seed=3, **kw)


def test_keys_are_wired_through_hydra():
    base = (ROOT / "morph/configs/base.yaml").read_text()
    assert "slot_gain_every: 1" in base and "spandec_slot_keep: 1.0" in base
    assert "slot_gain_every=int(" in (ROOT / "morph/training/train.py").read_text()
    assert '"spandec_slot_keep"' in (ROOT / "morph/training/tul_setup.py").read_text()
