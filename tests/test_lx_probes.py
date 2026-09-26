"""The three LX probes of 2026-09-26: the amplitude-matched K-curve
(`lab/divergence/lx_amp_matched.py`), the LX-Carry Stage 0 re-scoring
(`lab/divergence/lx_carry_stage0.py`, the math in `morph/model/rollout_mixture.py`) and the
selection ceiling (`lab/divergence/lx_selection_ceiling.py`).

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
    CE (the stream-index join and the scatter are right);
  * selection ceiling: per span oracle <= span mixture <= oracle + log K <= ..., mixture -
    oracle = log(K c_max), mixture <= the mean single rollout, oracle <= best fixed <=
    mean single overall; the span mixture is the sum of an independently computed
    per-token Bayes read; credit KL is 0 when the rollouts tie; a hand-computed K = 2 case;
    on the tiny model the stored arrays' Bayes read sums to the span mixture per span, and
    grouping spans without the row is caught; the token-shuffle null keeps each token's
    values, is 0 when the rollouts tie, and removes a span-coherent rollout advantage.
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
from lx_selection_ceiling import (  # noqa: E402
    check_identities, credit, selection_reads, span_index, token_shuffle_null)

from morph.model.rollout_mixture import (  # noqa: E402
    carried_position_nll, fixed_share_log_prior, segment_index, span_segment_start)
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


# ── probe 3: the selection ceiling ───────────────────────────────────────────────────


def _synthetic(K: int = 4, G: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(lp [K, N], span [N])``: random log-probs, spans of 1-11 tokens in order."""
    rng = np.random.default_rng(seed)
    span = np.repeat(np.arange(G), rng.integers(1, 12, size=G))
    return -rng.gamma(2.0, 1.5, size=(K, span.size)), span


def _bayes_reference(lp: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Independent per-token Bayes read, uniform restart per span: at token i the weights
    are the softmax of the span's EARLIER tokens' summed log-probs (plain Python)."""
    K, N = lp.shape
    out = np.empty(N)
    for i in range(N):
        prev = [j for j in range(N) if span[j] == span[i] and j < i]
        C = [sum(lp[k][j] for j in prev) for k in range(K)]
        mx = max(C)
        w = [math.exp(c - mx) for c in C]
        z = sum(w)
        out[i] = -math.log(sum(w[k] / z * math.exp(lp[k][i]) for k in range(K)))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_inequalities_hold_on_random_scores(seed):
    lp, span = _synthetic(seed=seed)
    K, N = lp.shape
    G = int(span.max()) + 1
    rd = selection_reads(lp, span, G)
    mix, orc = rd["mix_span"], rd["oracle_span"]
    assert (orc <= mix + 1e-12).all()                         # max S >= log mean exp S
    assert (mix <= orc + math.log(K) + 1e-12).all()           # log mean exp S >= max S - log K
    assert (mix <= -rd["S"].mean(0) + 1e-12).all()            # Jensen: >= mean_k S_k
    assert (mix - orc > 1e-6).any()                           # not all ties on random data
    np.testing.assert_allclose(mix - orc, np.log(K * rd["c_max"]), rtol=0, atol=1e-12)
    totals = -lp.sum(1)
    assert rd["k_fixed"] == int(np.argmin(totals))
    assert rd["oracle"].sum() <= rd["fixed"].sum() == pytest.approx(totals.min())
    assert rd["fixed"].sum() <= totals.mean()
    assert rd["single_ce"] == pytest.approx((totals / N).tolist())
    for g in range(G):                                        # brute-force oracle
        m = span == g
        kb = int(np.argmax(lp[:, m].sum(1)))
        np.testing.assert_array_equal(rd["oracle"][m], -lp[kb, m])
    assert ((rd["kl"] >= -1e-12) & (rd["kl"] <= math.log(K) + 1e-12)).all()
    np.testing.assert_allclose(rd["kl"] + rd["entropy"], math.log(K), atol=1e-12)
    # the span mixture is the per-token Bayes read summed over the span (chain rule)
    today = _bayes_reference(lp, span)
    chk = check_identities(rd, span, today, 1e-9)
    assert chk["bayes_span_sum_vs_mixture_max_abs_dev"] < 1e-9
    assert today.mean() == pytest.approx(mix.sum() / N, rel=1e-12)
    # a read whose span sums are wrong is refused
    with pytest.raises(RuntimeError, match="span sums"):
        check_identities(rd, span, today[::-1].copy(), 1e-9)


def test_credit_is_uniform_and_the_oracle_is_the_mixture_when_rollouts_tie():
    lp1, span = _synthetic(K=1)
    lp = np.repeat(lp1, 4, axis=0)
    rd = selection_reads(lp, span, int(span.max()) + 1)
    np.testing.assert_allclose(rd["kl"], 0.0, atol=1e-12)
    np.testing.assert_allclose(rd["entropy"], math.log(4), atol=1e-12)
    np.testing.assert_allclose(rd["c_max"], 0.25, atol=1e-12)
    np.testing.assert_allclose(rd["mix_span"], rd["oracle_span"], atol=1e-12)
    np.testing.assert_array_equal(rd["oracle"], -lp1[0])
    np.testing.assert_array_equal(rd["fixed"], -lp1[0])
    # and one decisive rollout drives KL to log K
    c = credit(np.array([[0.0], [-80.0], [-80.0], [-80.0]]))
    assert c["kl"][0] == pytest.approx(math.log(4), abs=1e-12)
    assert c["c_max"][0] == pytest.approx(1.0) and c["argmax"][0] == 0


def test_selection_hand_computed_case():
    # K = 2; span 0 has two tokens, span 1 one token.
    #   S(span 0) = (-2, -5): c = (0.952574, 0.047426), H = 0.190865, KL = 0.502282,
    #     mixture = 2 + log 2 - log(1 + e^-3) = 2.644560, oracle 2.
    #   S(span 1) = (-4, -1.5): mixture 2.114257, oracle 1.5 (rollout 1).
    #   totals: rollout 0 = 6, rollout 1 = 6.5 -> best fixed 0; oracle 3.5.
    lp = np.array([[-1.0, -1.0, -4.0], [-2.0, -3.0, -1.5]])
    span = np.array([0, 0, 1])
    rd = selection_reads(lp, span, 2)
    np.testing.assert_allclose(rd["S"], [[-2.0, -4.0], [-5.0, -1.5]])
    assert rd["kl"][0] == pytest.approx(0.5022822094535027, rel=1e-12)
    assert rd["entropy"][0] == pytest.approx(0.19086497110644257, rel=1e-12)
    assert rd["c_max"][0] == pytest.approx(0.952574126822433, rel=1e-12)
    assert rd["argmax"].tolist() == [0, 1]
    np.testing.assert_allclose(rd["oracle"], [1.0, 1.0, 1.5])
    np.testing.assert_allclose(rd["mix_span"], [2.644559828986203, 2.1142574462673958],
                               rtol=1e-12)
    assert rd["k_fixed"] == 0
    np.testing.assert_allclose(rd["fixed"], [1.0, 1.0, 4.0])
    assert rd["single_ce"] == pytest.approx([2.0, 6.5 / 3])
    # span numbering: (row, seg) pairs, row-major; equal seg in two rows are two spans
    sp, first = span_index(np.array([0, 0, 0, 1, 1]), np.array([0, 0, 2, 0, 1]))
    assert sp.tolist() == [0, 0, 1, 2, 3] and first.tolist() == [0, 2, 3, 4]


def test_selection_reads_on_the_tiny_model_stored_arrays():
    """The ceiling path end to end: the Stage 1 scorer's stored arrays, rebuilt by
    ``rescore``, grouped by ``span_index``: the Bayes read sums to the span mixture on every
    span, the span count is the layout's, and a grouping that ignores the row is caught."""
    from lxtul_e_stage1_score import score_arm
    m = _model().eval()
    bt = _batches()
    per = score_arm(m, bt, [2], "cpu", 2e-3)["per"][2]
    r = rescore(bt, per["coda_idx"], per["coda_code"], [1.0], eos_id=0)
    span, first = span_index(r["row"], r["seg"])
    rd = selection_reads(-per["coda_code"].astype(np.float64), span, int(first.shape[0]))
    chk = check_identities(rd, span, r["today"], 1e-4)
    assert chk["bayes_span_sum_vs_mixture_max_abs_dev"] < 1e-4
    inp, lab, layout, _idx = bt[0]
    sc = (~layout.slot_mask) & (lab >= 0)
    seg = segment_index(span_segment_start(layout.bag_id))
    want = sum(int(seg[b][sc[b]].unique().numel()) for b in range(inp.shape[0]))
    assert first.shape[0] == want > inp.shape[0]
    assert r["today"].mean() == pytest.approx(rd["mix_span"].sum() / span.size, abs=1e-5)
    assert rd["oracle"].mean() < r["today"].mean()
    # grouping by run index alone merges row 0's and row 1's runs: refused
    _u, bad = np.unique(r["seg"], return_inverse=True)
    rd_bad = selection_reads(-per["coda_code"].astype(np.float64), bad, int(_u.shape[0]))
    with pytest.raises(RuntimeError, match="span sums"):
        check_identities(rd_bad, bad, r["today"], 1e-4)


def test_token_shuffle_null_removes_span_coherent_advantage_only():
    rng = np.random.default_rng(3)
    K, G, n = 4, 200, 10
    span = np.repeat(np.arange(G), n)
    N = span.size
    # coherent: in each span one rollout is 0.3 nats better on EVERY token
    lp = -rng.gamma(2.0, 1.5, size=(1, N)).repeat(K, 0) - 0.3
    win = rng.integers(0, K, size=G)
    lp[win[span], np.arange(N)] += 0.3
    rd = selection_reads(lp, span, G)
    obs = float((rd["mix_span"] - rd["oracle_span"]).sum() / N)
    nl = token_shuffle_null(lp, span, G, 3, 0)
    assert obs > 0.08                                   # log(K c_max) / 10 per token
    assert nl["bayes_minus_oracle_mean"] < 0.5 * obs    # the shuffle breaks the coherence
    assert nl["kl_mean"] < 0.5 * float(rd["kl"].mean())
    # each token keeps its K values (a permutation, not a resample)
    perm = np.argsort(np.random.default_rng(0).random((K, N)), axis=0)
    np.testing.assert_array_equal(np.sort(np.take_along_axis(lp, perm, 0), 0), np.sort(lp, 0))
    # tied rollouts: the null gap and KL are exactly what the observed ones are, 0
    lp1, sp1 = _synthetic(K=1)
    z = token_shuffle_null(np.repeat(lp1, 4, 0), sp1, int(sp1.max()) + 1, 2, 0)
    assert max(z["bayes_minus_oracle"]) == pytest.approx(0.0, abs=1e-12)
    assert max(z["kl"]) == pytest.approx(0.0, abs=1e-12)
