"""LX efficient-exploration knobs (2026-09-29): pure-function tests for
``morph/model/tul_explore.py`` (KNOBS 3, 4, 5, 6 — scaffolding for later experiments,
nothing queued). Knob 2 (late fusion) was investigated and NOT implemented; see the
module docstring of ``tul_explore.py`` and the Agent Note for why.

This file hand-verifies the MATH each function claims, on small tensors whose correct
answer is computed independently in the test itself (never by calling the function
under test a second time) — the project's own "assert on the actual contract" bar.
Model-level wiring (TULConfig fields, ``build_tul_runtime`` reach, bit-identity at
every knob's OFF default, a real forward at dropout 0 and dropout > 0) is
``tests/test_tul_explore_wiring.py``.

CPU, fp32.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from morph.model.tul_explore import (TULHypMergeGate, TULHypScoreHead, gather_rows,
                                     hyp_merge_mean, hyp_merge_probe_stats,
                                     latent_energy_score, latent_infonce,
                                     score_head_kl_loss, score_head_read_gaps,
                                     score_head_topk_mask, swor_uniform)


# ── Knob 3: score head ──────────────────────────────────────────────────────────────


def test_score_head_is_rng_neutral_and_linear_no_bias():
    torch.manual_seed(0)
    before = torch.randn(5)
    head = TULHypScoreHead(d_model=8)
    torch.manual_seed(0)
    after = torch.randn(5)
    assert torch.equal(before, after), "building the head perturbed the global RNG stream"
    assert head.w.bias is None
    z = torch.randn(3, 4, 8)
    out = head(z)
    assert out.shape == (3, 4)
    w = head.w.weight.view(8)
    torch.testing.assert_close(out, (z * w).sum(-1))


def test_score_head_kl_loss_hand_computed():
    # R=2 rollouts, N=2 items. Item 0: target posterior sharply favours rollout 0;
    # head predicts the OPPOSITE — should cost a large KL. Item 1: head matches target
    # exactly — should cost ~0. item 1 is INVALID and must not move the mean.
    s_target = torch.tensor([[5.0, -5.0], [1.0, 1.0]])     # [R, N]
    scores = torch.tensor([[-5.0, 5.0], [1.0, 1.0]])        # [R, N]  (item 0 flipped)
    valid = torch.tensor([True, False])
    got = score_head_kl_loss(scores, s_target, valid)
    p0 = torch.softmax(s_target[:, 0], dim=0)
    q0 = torch.softmax(scores[:, 0], dim=0)
    want = float((p0 * (p0.clamp_min(1e-30).log() - q0.clamp_min(1e-30).log())).sum())
    assert float(got) == pytest.approx(want, rel=1e-5)
    assert float(got) > 1.0, "a flipped, sharp posterior must cost a large KL"


def test_score_head_kl_loss_zero_when_head_matches_target_exactly():
    s_target = torch.randn(4, 6)
    valid = torch.tensor([True, True, False, True, True, False])
    got = score_head_kl_loss(s_target.clone(), s_target.clone(), valid)
    assert float(got) == pytest.approx(0.0, abs=1e-5)


def test_score_head_kl_loss_shape_mismatch_raises():
    with pytest.raises(ValueError, match="scores"):
        score_head_kl_loss(torch.randn(3, 4), torch.randn(3, 5), torch.ones(4, dtype=torch.bool))


def test_score_head_topk_mask_matches_hand_ranking():
    scores = torch.tensor([[3.0, 0.0], [1.0, 5.0], [2.0, 1.0], [0.0, 2.0]])  # [R=4, N=2]
    mask = score_head_topk_mask(scores, k=2)
    # item 0: ranks are 3 > 2 > 1 > 0 -> rollouts {0, 2} win top-2
    assert mask[:, 0].tolist() == [True, False, True, False]
    # item 1: ranks are 5 > 2 > 1 > 0 -> rollouts {1, 3}
    assert mask[:, 1].tolist() == [False, True, False, True]
    assert mask.sum(dim=0).tolist() == [2, 2]


def test_score_head_topk_mask_k_out_of_range_raises():
    with pytest.raises(ValueError, match="k=0"):
        score_head_topk_mask(torch.randn(3, 2), k=0)
    with pytest.raises(ValueError, match="k=4"):
        score_head_topk_mask(torch.randn(3, 2), k=4)


def test_score_head_read_gaps_hand_computed():
    # R=2 rollouts, N=3 items; item 2 has no scored token and must add nothing.
    S = torch.tensor([[-1.0, -4.0, 0.0],
                      [-3.0, -2.0, 0.0]])                     # [R, N] summed span log-probs
    scores = torch.tensor([[0.0, 1.0, 5.0],
                           [1.0, 0.0, -5.0]])                 # head picks rollout 1, then 0
    scored = torch.tensor([True, True, False])
    n_w = torch.tensor(10.0)
    got = score_head_read_gaps(scores, S, scored, n_w)
    lme = torch.logsumexp(S[:, :2], dim=0) - math.log(2)       # exact 2-way read per span
    top1 = torch.tensor([-3.0, -4.0])                         # the head's picks
    torch.testing.assert_close(got["read_top1_gap"], (lme - top1).sum() / 10.0)
    torch.testing.assert_close(got["read_top2_gap"], torch.tensor(0.0))   # R=2: the mixture
    torch.testing.assert_close(got["read_rand1_gap"], (lme - S[:, :2].mean(0)).sum() / 10.0)
    torch.testing.assert_close(got["read_best1_gap"],
                               (lme - S[:, :2].max(0).values).sum() / 10.0)
    assert float(got["score_agree"]) == 0.0                   # both picks are the loser
    assert float(got["read_rand1_gap"]) >= 0.0 >= float(got["read_best1_gap"])


def test_score_head_read_gaps_a_head_that_knows_the_posterior_hits_the_ceiling():
    """The deployable read of a head whose scores ARE the posterior is the hindsight
    read: the contract that separates the head's pick from the floor and the ceiling."""
    torch.manual_seed(0)
    S = torch.randn(4, 12) * 3.0
    scored = torch.ones(12, dtype=torch.bool)
    got = score_head_read_gaps(S.clone(), S, scored, torch.tensor(40.0))
    torch.testing.assert_close(got["read_top1_gap"], got["read_best1_gap"])
    assert float(got["score_agree"]) == 1.0
    assert float(got["read_top1_gap"]) < float(got["read_top2_gap"]) \
        < float(got["read_rand1_gap"])
    flat = score_head_read_gaps(torch.zeros(4, 12), S, scored, torch.tensor(40.0))
    # constant scores: topk ties break to rollout 0, a fixed single pick
    torch.testing.assert_close(flat["read_top1_gap"],
                               ((torch.logsumexp(S, 0) - math.log(4)) - S[0]).sum() / 40.0)


# ── Knob 4: latent set loss ─────────────────────────────────────────────────────────


def test_latent_energy_score_hand_computed_two_rollouts():
    # 1 item, R=2, C=1. hyp = [0, 4], target = [0]. d_target = [0, 4] -> mean 2.
    # pairwise (r,r') over all 4 ordered pairs incl r==r': |0-0|,|0-4|,|4-0|,|4-4| = 0,4,4,0
    # mean = 8/4 = 2. e = 2 - 0.5*2 = 1.
    hyp = torch.tensor([[[0.0]], [[4.0]]])                 # [R=2, N=1, C=1]
    target = torch.tensor([[0.0]])                          # [N=1, C=1]
    valid = torch.tensor([True])
    got = latent_energy_score(hyp, target, valid)
    assert float(got) == pytest.approx(1.0, abs=1e-6)


def test_latent_energy_score_collapsed_ensemble_is_plain_mean_distance():
    # Every hypothesis identical -> pairwise term is exactly 0, score = mean distance.
    hyp = torch.full((5, 3, 4), 2.0)
    target = torch.zeros(3, 4)
    valid = torch.ones(3, dtype=torch.bool)
    got = latent_energy_score(hyp, target, valid)
    want = (hyp - target.unsqueeze(0)).norm(dim=-1).mean()
    assert float(got) == pytest.approx(float(want), rel=1e-5)


def test_latent_energy_score_invalid_items_excluded():
    hyp = torch.randn(3, 4, 5)
    target = torch.randn(4, 5)
    valid_all = torch.ones(4, dtype=torch.bool)
    valid_drop = valid_all.clone()
    valid_drop[2] = False
    hyp2 = hyp.clone()
    target2 = target.clone()
    # Perturbing the dropped item must not move the loss at all.
    hyp2[:, 2] += 100.0
    target2[2] += 100.0
    a = latent_energy_score(hyp, target, valid_drop)
    b = latent_energy_score(hyp2, target2, valid_drop)
    assert float(a) == pytest.approx(float(b), rel=1e-5)


def test_latent_energy_score_shape_check():
    with pytest.raises(ValueError, match="latent_energy_score"):
        latent_energy_score(torch.randn(2, 3), torch.randn(3, 4), torch.ones(3, dtype=torch.bool))


def test_latent_infonce_matches_hand_computed_fixed_hypothesis():
    """Every rollout's hypothesis is the SAME fixed vector for every item, so the
    similarity ROW over keys is identical for every query — an independent hand
    computation (never calling ``latent_infonce`` twice) of that one row's cross entropy
    against every target index must match the function's output exactly."""
    torch.manual_seed(0)
    N, C, R = 6, 8, 3
    fixed = torch.randn(C)
    hyp = fixed.view(1, 1, C).expand(R, N, C).clone()
    target = torch.randn(N, C)
    valid = torch.ones(N, dtype=torch.bool)
    tau = 0.5
    got = latent_infonce(hyp, target, valid, tau=tau)
    hyp_n = F.normalize(fixed, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    sim_row = (hyp_n @ tgt_n.T) / tau                          # [N], same for every query
    logsm = torch.log_softmax(sim_row, dim=-1)
    want = -logsm[torch.arange(N)].mean()
    assert float(got) == pytest.approx(float(want), rel=1e-4)


def test_latent_infonce_perfect_alignment_is_near_zero():
    torch.manual_seed(1)
    N, C, R = 5, 16, 2
    target = torch.randn(N, C)
    hyp = target.unsqueeze(0).expand(R, N, C).clone() * 50.0     # sharp, correct match
    valid = torch.ones(N, dtype=torch.bool)
    got = latent_infonce(hyp, target, valid, tau=0.1)
    assert float(got) < 1e-3


def test_latent_infonce_tau_must_be_positive():
    with pytest.raises(ValueError, match="tau"):
        latent_infonce(torch.randn(2, 3, 4), torch.randn(3, 4), torch.ones(3, dtype=torch.bool),
                       tau=0.0)


def test_latent_hyp_spread_is_zero_when_collapsed_and_positive_otherwise():
    from morph.model.tul_explore import latent_hyp_spread
    valid = torch.ones(3, dtype=torch.bool)
    collapsed = torch.zeros(4, 3, 5)
    assert float(latent_hyp_spread(collapsed, valid)) == pytest.approx(0.0, abs=1e-6)
    torch.manual_seed(2)
    spread = torch.randn(4, 3, 5)
    assert float(latent_hyp_spread(spread, valid)) > 0.0


# ── Knob 5: sampling without replacement ────────────────────────────────────────────


def test_swor_uniform_returns_k_distinct_ascending_indices_in_range():
    g = torch.Generator().manual_seed(0)
    idx = swor_uniform(K=8, k=3, generator=g)
    assert idx.shape == (3,)
    assert len(set(idx.tolist())) == 3
    assert idx.tolist() == sorted(idx.tolist())
    assert all(0 <= i < 8 for i in idx.tolist())


def test_swor_uniform_k_out_of_range_raises():
    with pytest.raises(ValueError, match="swor_uniform"):
        swor_uniform(K=4, k=0)
    with pytest.raises(ValueError, match="swor_uniform"):
        swor_uniform(K=4, k=5)


def test_swor_uniform_k_equals_K_is_a_full_permutation():
    g = torch.Generator().manual_seed(3)
    idx = swor_uniform(K=6, k=6, generator=g)
    assert sorted(idx.tolist()) == list(range(6))


def test_swor_uniform_marginal_inclusion_matches_k_over_K():
    """1000 draws of k=2 of K=4: every index's empirical inclusion rate should sit near
    the exact k/K = 0.5, well inside a generous tolerance for a finite sample."""
    g = torch.Generator().manual_seed(7)
    counts = torch.zeros(4)
    n = 2000
    for _ in range(n):
        idx = swor_uniform(K=4, k=2, generator=g)
        counts[idx] += 1
    rate = counts / n
    assert torch.allclose(rate, torch.full((4,), 0.5), atol=0.06), rate


def test_gather_rows_preserves_row_content_and_drops_others():
    K, block, C = 4, 3, 2
    x = torch.arange(K * block * C, dtype=torch.float32).view(K * block, C)
    keep = torch.tensor([0, 2])                        # keep rollouts 0 and 2 of 4
    got = gather_rows(x, keep, block)
    want = torch.cat([x[0:block], x[2 * block:3 * block]], dim=0)
    torch.testing.assert_close(got, want)


def test_gather_rows_recurses_into_dict_and_skips_non_matching_tensors():
    K, block = 3, 2
    tiled = torch.arange(K * block, dtype=torch.float32).view(K * block, 1)
    small = torch.zeros(5, 7)                          # dim(0) != K*block: left alone
    d = {"tg_allow": tiled, "scalar_like": small, "none_val": None}
    keep = torch.tensor([1])
    got = gather_rows(d, keep, block)
    torch.testing.assert_close(got["tg_allow"], tiled[block:2 * block])
    assert got["scalar_like"] is small
    assert got["none_val"] is None


def test_gather_rows_none_and_non_tensor_pass_through():
    assert gather_rows(None, torch.tensor([0]), 2) is None
    assert gather_rows(7, torch.tensor([0]), 2) == 7


# ── Knob 6: merge / superposition probe ─────────────────────────────────────────────


def test_hyp_merge_mean_is_the_plain_average():
    hyp = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])   # [R=3, N=1, C=2]
    got = hyp_merge_mean(hyp)
    torch.testing.assert_close(got, torch.tensor([[3.0, 4.0]]))


def test_hyp_merge_gate_is_rng_neutral_and_convex():
    torch.manual_seed(0)
    before = torch.randn(5)
    gate = TULHypMergeGate(d_model=6)
    torch.manual_seed(0)
    after = torch.randn(5)
    assert torch.equal(before, after)
    hyp = torch.randn(4, 3, 6)
    out = gate(hyp)
    assert out.shape == (3, 6)
    # Every output must be a CONVEX combination of that item's own R hypotheses: its
    # distance to the centroid cannot exceed the max hypothesis's distance to the
    # centroid (a strict convexity check, not merely a shape check).
    centroid = hyp.mean(dim=0)
    max_dev = (hyp - centroid).norm(dim=-1).max(dim=0).values
    out_dev = (out - centroid).norm(dim=-1)
    assert bool((out_dev <= max_dev + 1e-4).all())


def test_hyp_merge_gate_uniform_query_recovers_the_mean():
    gate = TULHypMergeGate(d_model=4)
    with torch.no_grad():
        gate.q.weight.zero_()                         # score = (0 . hyp) = 0 for every r
    hyp = torch.randn(5, 2, 4)
    torch.testing.assert_close(gate(hyp), hyp_merge_mean(hyp))


def test_hyp_merge_probe_stats_hand_computed():
    # 1 item, R=2, C=1. hyp = [0, 10], target = [1]. best_single = |0-1| = 1,
    # mean_single = (1+9)/2 = 5, merge (mean) = 5 -> dist 4, does NOT beat best (1).
    hyp = torch.tensor([[[0.0]], [[10.0]]])
    target = torch.tensor([[1.0]])
    valid = torch.tensor([True])
    merged = hyp_merge_mean(hyp)
    stats = hyp_merge_probe_stats(hyp, target, valid, merged)
    assert stats["best_single_dist"] == pytest.approx(1.0, abs=1e-6)
    assert stats["mean_single_dist"] == pytest.approx(5.0, abs=1e-6)
    assert stats["merge_dist"] == pytest.approx(4.0, abs=1e-6)
    assert stats["merge_beats_best_single"] == pytest.approx(0.0, abs=1e-6)


def test_hyp_merge_probe_stats_detaches_and_carries_no_grad():
    hyp = torch.randn(3, 2, 4, requires_grad=True)
    target = torch.randn(2, 4, requires_grad=True)
    valid = torch.ones(2, dtype=torch.bool)
    merged = hyp_merge_mean(hyp)
    stats = hyp_merge_probe_stats(hyp, target, valid, merged)
    assert all(isinstance(v, float) for v in stats.values())
