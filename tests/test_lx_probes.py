"""The two LX probes of 2026-09-26: the amplitude-matched K-curve
(`lab/divergence/lx_amp_matched.py`) and the LX-Carry Stage 0 re-scoring
(`lab/divergence/lx_carry_stage0.py`, the math in `morph/model/rollout_mixture.py`).

What each test pins, on the tiny strict `code_enum_k = 4` model in fp32 on CPU:
  * the code multiplier at 1 is the trained model bit for bit, any other value reaches the
    forward, and the trained ratio is restored after;
  * the bisection finds the multiplier of a monotone map and refuses a target it cannot
    bracket;
  * exit separation grows with the multiplier, and at depth 1 (the rollouts differ ONLY by
    one code term) all of it lies in the code subspace: sep_code == sep_abs;
  * Step 0's code share on the ctx slice is the hand-computed one;
  * the carried read at eps = 1 is `_enum_position_nll` bit for bit, at eps < 1 it moves
    every run but a row's first, and a row's carried NLL sums to the switching model's
    exact log-likelihood computed here by an independent loop;
  * an EOS run resets the carry to uniform;
  * the offline re-scoring of a forward's stored arrays reproduces that forward's coda
    CE (the stream-index join and the scatter are right).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "lab/divergence")
from lx_amp_matched import (  # noqa: E402
    code_ratio_scale, code_slice_energy, match_multiplier, ratio_ci, run_point)
from lx_carry_stage0 import rescore  # noqa: E402

from morph.model.rollout_mixture import (  # noqa: E402
    carried_position_nll, fixed_share_log_prior, span_segment_start)
from morph.model.transformer import MORPHTransformer  # noqa: E402
from morph.model.tul_code_enum import gram_schmidt_rows  # noqa: E402
from morph.model.tul_layout import slot_layout_from_ids  # noqa: E402
from test_tul_lxtul_e import K, _brute_rollout_lp, _model, _Spy, _table  # noqa: E402
from test_tul_strict_geometry import _ids, _pack, _rule, _spec  # noqa: E402


def _batches(ids: np.ndarray | None = None) -> list:
    """One batch in `val_batches` format ``(inp, labels, layout, idx)``; ``idx`` numbers the
    token positions in row order (row b from b * 10_000), -1 at slot positions."""
    if ids is None:
        _i, inp, lab, layout = _pack()
    else:
        inp, lab, layout, _s = slot_layout_from_ids(ids, _rule(), _spec())
    idx = torch.full(inp.shape, -1, dtype=torch.long)
    tok = ~layout.slot_mask
    for b in range(inp.shape[0]):
        p = tok[b].nonzero().flatten()
        idx[b, p] = b * 10_000 + torch.arange(p.numel())
    return [(inp, lab, layout, idx)]


def _rollout_lp(m, inp, lab, layout, depth: int = 2) -> torch.Tensor:
    spy = _Spy(m, "_enum_mix_losses")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, depth))
        return _brute_rollout_lp(m, spy.calls[0][0][0], lab)


# ── probe 1: the code multiplier ─────────────────────────────────────────────────────


def test_multiplier_one_is_the_trained_model_and_others_reach_the_forward():
    _i, inp, lab, layout = _pack()
    m = _model().eval()
    r0 = m.tul_code_enum.ratio
    assert r0 == m.cfg.tul.code_enum_ratio == 0.1
    tab = _table(layout, 3)

    def fwd():
        with torch.no_grad():
            lo = m(inp, labels=lab, slot_layout=layout, slot_depths=tab)
            lg = m(inp, slot_layout=layout, slot_depths=tab)["logits"]
        return lo["loss"], lg

    base_loss, base_logits = fwd()
    with code_ratio_scale(m, 1.0):
        l1, g1 = fwd()
    assert torch.equal(l1, base_loss) and torch.equal(g1, base_logits)
    with code_ratio_scale(m, 2.5):
        assert m.tul_code_enum.ratio == pytest.approx(0.25)
        l2, g2 = fwd()
    assert not torch.equal(g2, base_logits)
    assert abs(float(l2 - base_loss)) > 1e-6
    assert m.tul_code_enum.ratio == r0
    # restored even when the block raises
    with pytest.raises(RuntimeError, match="boom"):
        with code_ratio_scale(m, 3.0):
            raise RuntimeError("boom")
    assert m.tul_code_enum.ratio == r0
    # the term itself scales by a
    h = torch.randn(K * 2, 5, 4, m.cfg.d_model)
    valid = torch.ones(K * 2, 5, dtype=torch.bool)
    t1 = m.tul_code_enum.term(h, valid, K)
    with code_ratio_scale(m, 3.0):
        t3 = m.tul_code_enum.term(h, valid, K)
    torch.testing.assert_close(t3, 3.0 * t1, rtol=1e-6, atol=1e-7)


def test_match_multiplier_bisects_a_monotone_map_and_refuses_the_unbracketable():
    def f(a):
        return 2.0 * a ** 0.9

    for target in (7.0, 0.9, 150.0):
        a, trace = match_multiplier(f, target, tol_rel=0.005)
        assert abs(f(a) - target) <= 0.005 * target
        assert trace[-1][0] == a or any(t[0] == a for t in trace)
    with pytest.raises(RuntimeError, match="cannot bracket"):
        match_multiplier(f, 1e4)
    with pytest.raises(RuntimeError, match="cannot bracket"):
        match_multiplier(f, 1e-4)


def test_exit_separation_grows_with_the_multiplier_and_is_all_code_at_depth_one():
    m = _model().eval()
    bt = _batches()
    p = {x: run_point(m, bt, 1, x, "cpu", 2e-3) for x in (1.0, 2.0, 4.0)}
    assert p[1.0]["sep_abs"] < p[2.0]["sep_abs"] < p[4.0]["sep_abs"]
    # depth 1: the rollouts share every tensor up to the one code term, so their exit
    # difference IS r * rms * (u_i - u_j): entirely inside span(u_k)
    for x, r in p.items():
        assert r["sep_code"] == pytest.approx(r["sep_abs"], rel=1e-4), x
    # depth 3: later passes act on the code nonlinearly, so some separation leaves it
    p3 = run_point(m, bt, 3, 1.0, "cpu", 2e-3)
    assert p3["sep_code"] < p3["sep_abs"] * (1 - 1e-4)
    # a = 1 is the forward's own coda CE (score_arm's self-check ran inside, tol 2e-3)
    assert np.isfinite(p[1.0]["coda_mix"]).all()


def test_step0_share_is_the_hand_computed_one():
    m = _model().eval()
    s0 = code_slice_energy(m, n_null=50)
    s, e = m.injection.start, m.injection.end
    assert s0["slice"] == [s, e] and 0 < s < e < m.cfg.d_model
    q = gram_schmidt_rows(m.tul_code_enum.basis.detach().double())
    u = m.tul_code_enum.directions().double()
    want = ((u[:, s:e] ** 2).sum(1) / (u ** 2).sum(1)).tolist()
    assert s0["trained_u_share"] == pytest.approx(want, rel=1e-6)
    assert s0["trained_subspace_share"] == pytest.approx(
        float((q[:, s:e] ** 2).sum() / (K - 1)), rel=1e-9)
    # a fresh model's basis IS the init draw
    assert s0["init_u_share"] == pytest.approx(s0["trained_u_share"], rel=1e-9)
    with torch.no_grad():
        m.tul_code_enum.basis[:, s:e] = 0.0          # move every code off the slice
    assert code_slice_energy(m, n_null=10)["trained_subspace_share"] < 1e-12


def test_ratio_ci_is_the_ratio_of_the_two_differences():
    rng = np.random.default_rng(0)
    n = 400
    blocks = np.repeat(np.arange(40), 10)
    ref = rng.normal(size=n)
    un = ref + 0.02 + 0.01 * rng.normal(size=n)
    mt = ref + 0.005 + 0.01 * rng.normal(size=n)
    r = ratio_ci(mt, ref, un, ref, blocks, 500, 0)
    assert r["point"] == pytest.approx((mt.mean() - ref.mean()) / (un.mean() - ref.mean()))
    assert r["lo"] <= r["point"] <= r["hi"]


# ── probe 2: the carried read ────────────────────────────────────────────────────────


def _switching_nll(lp, bag, scored, eps, reset=None) -> float:
    """Independent reference: -log p(row) of the fixed-share model, one row, by runs of
    ``bag`` in order (plain Python, math.log)."""
    Kr = len(lp)
    pi = [1.0 / Kr] * Kr
    total = 0.0
    L = len(bag)
    p = 0
    while p < L:
        q = p
        while q < L and bag[q] == bag[p]:
            q += 1
        S = [sum(float(lp[k][i]) for i in range(p, q) if scored[i]) for k in range(Kr)]
        if any(scored[i] for i in range(p, q)):
            terms = [math.log(pi[k]) + S[k] for k in range(Kr)]
            mx = max(terms)
            lse = mx + math.log(sum(math.exp(t - mx) for t in terms))
            total += lse
            alpha = [math.exp(t - lse) for t in terms]
            pi = [(1 - eps) * alpha[k] + eps / Kr for k in range(Kr)]
            if reset is not None and any(reset[i] for i in range(p, q)):
                pi = [1.0 / Kr] * Kr
        p = q
    return -total


def test_eps_one_is_todays_read_bit_for_bit_and_smaller_eps_moves_later_runs():
    _i, inp, lab, layout = _pack()
    m = _model().eval()
    lp = _rollout_lp(m, inp, lab, layout)
    sc = (~layout.slot_mask) & (lab >= 0)
    today = MORPHTransformer._enum_position_nll(lp, inp, layout)
    assert torch.equal(carried_position_nll(lp, inp, layout, sc, 1.0), today)
    assert torch.equal(fixed_share_log_prior(lp, sc, span_segment_start(layout.bag_id), 1.0),
                       torch.zeros(lp.shape, dtype=torch.float64))
    car = carried_position_nll(lp, inp, layout, sc, 0.2)
    first_run = layout.bag_id == layout.bag_id[:, :1]
    first_run &= torch.cumprod(first_run.long(), dim=1).bool()          # the leading run
    assert torch.equal(car[first_run & sc], today[first_run & sc])
    moved = (car - today).abs()[sc & ~first_run]
    assert float(moved.max()) > 1e-4
    assert int((moved > 0).sum()) > 0.5 * int((sc & ~first_run).sum())


@pytest.mark.parametrize("eps", [0.5, 0.05, 0.0])
def test_a_rows_carried_nll_is_the_switching_models_likelihood(eps):
    _i, inp, lab, layout = _pack()
    m = _model().eval()
    lp = _rollout_lp(m, inp, lab, layout)
    sc = (~layout.slot_mask) & (lab >= 0)
    car = carried_position_nll(lp, inp, layout, sc, eps)
    for b in range(inp.shape[0]):
        got = float(car[b][sc[b]].double().sum())
        want = _switching_nll(lp[:, b].tolist(), layout.bag_id[b].tolist(), sc[b].tolist(),
                              eps)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-5)
    # and it is not today's (the restart's) likelihood
    today = MORPHTransformer._enum_position_nll(lp, inp, layout)
    assert abs(float(car[sc].sum() - today[sc].sum())) > 1e-4


def test_an_eos_run_resets_the_carry_to_uniform():
    ids = _ids(B=2)
    ids[0, 36] = 0                                   # EOS (the rule's eos_id) mid row 0
    (inp, lab, layout, _idx), = _batches(ids)
    m = _model().eval()
    lp = _rollout_lp(m, inp, lab, layout)
    sc = (~layout.slot_mask) & (lab >= 0)
    reset = (~layout.slot_mask) & (inp == 0)
    assert int(reset.sum()) == 1
    seg_start = span_segment_start(layout.bag_id)
    pr = fixed_share_log_prior(lp, sc, seg_start, 0.2, reset)
    pr_no = fixed_share_log_prior(lp, sc, seg_start, 0.2, None)
    p_eos = int(reset[0].nonzero()[0])
    run = layout.bag_id[0, p_eos]
    nxt = (layout.bag_id[0] == run + 1) & ~layout.slot_mask[0]
    assert nxt.any()
    assert torch.equal(pr[:, 0][:, nxt], torch.zeros_like(pr[:, 0][:, nxt]))
    assert float(pr_no[:, 0][:, nxt].abs().max()) > 1e-6
    # the EOS row's likelihood is the switching model's with the reset
    car = carried_position_nll(lp, inp, layout, sc, 0.2, reset)
    want = _switching_nll(lp[:, 0].tolist(), layout.bag_id[0].tolist(), sc[0].tolist(), 0.2,
                          reset[0].tolist())
    assert float(car[0][sc[0]].double().sum()) == pytest.approx(want, rel=1e-6, abs=1e-5)


def test_offline_rescoring_reproduces_the_forwards_coda_ce():
    """The Stage 0 path end to end: the Stage 1 scorer's stored arrays (coda_idx,
    coda_code = -lp, coda_mix) re-scored by `rescore` give back coda_mix at eps = 1."""
    from lxtul_e_stage1_score import score_arm
    m = _model().eval()
    bt = _batches()
    per = score_arm(m, bt, [2], "cpu", 2e-3)["per"][2]
    r = rescore(bt, per["coda_idx"], per["coda_code"], [1.0, 0.2], eos_id=0)
    np.testing.assert_allclose(r["today"], per["coda_mix"], rtol=0, atol=1e-5)
    assert np.array_equal(r["eps"][1.0], r["today"])
    assert np.abs(r["eps"][0.2] - r["today"]).max() > 1e-4
    assert r["n_rows"] == 2 and set(np.unique(r["row"]).tolist()) == {0, 1}
    # a stored array from DIFFERENT rows is refused, not silently joined
    with pytest.raises(RuntimeError, match="not in the stored coda_idx"):
        rescore(bt, per["coda_idx"] + 1, per["coda_code"], [1.0], eos_id=0)
