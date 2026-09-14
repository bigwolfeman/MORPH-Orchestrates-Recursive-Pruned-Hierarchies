"""CPU unit tests for the four loop diagnostics (lab/divergence/{aa_score,
jacobian_gap_vs_k, sink_mass_per_pass, basin_map}.py, 2026-09-14).

Tiny config, no tokenizer, no checkpoint, no dataloader — same conventions as
``test_tul_forward.py`` / ``test_tul_gl1.py`` / ``test_tul_spandec.py``. These tests
exercise the instruments' own computational core directly (the functions a real
``--ckpt`` run also calls), not the CLI/checkpoint-loading path in ``_diag_common.Arm``,
which needs a real dataset and a real checkpoint file.

One test per script's headline mechanism, plus the shared pure-math helpers
(``jacobian_top2``, ``spearman_rho``, ``basin_map``'s settling-time / entropy).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "lab", "divergence"))

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
SLOT_ID = 4

_BASE = dict(
    d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
    n_prelude=1, n_core=2, n_coda=1, mean_depth=3, max_depth=4, bptt_depth=3,
    channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
    hca_compress_ratio=8, top_k=8, window_size=16,
    retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
    dropout=0.0,
)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=48, prefix_k=2, max_slots=6, slot_id=SLOT_ID)
    base.update(kw)
    return TulLayoutSpec(**base)


def _slot_batch(B=2, n=80, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == SLOT_ID] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), _spec())


def _slot_model(seed=3) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul = TULConfig(prefix_k=2, slot_id=SLOT_ID, slot_seed="boundary", tg_restrict=True,
                    tg_geometry="strict", tg_coda_prefix_reach="all", emit_weight=0.0,
                    plast_weight=1.0, token_state_dropout=0.0, spandec=True,
                    spandec_layers=1, spandec_max_tokens=4, spandec_heads=2)
    cfg = MORPHConfig(tul=tul, core_fixed_point_lambda=0.0, ckpt_grad_iters=0, **_BASE)
    return MORPHTransformer(cfg)


def _plain_model(seed=3) -> MORPHTransformer:
    torch.manual_seed(seed)
    cfg = MORPHConfig(core_state_init="noise", core_state_init_std=0.02,
                      injection_channels="all", injection_B=True,
                      core_fixed_point_lambda=0.0, ckpt_grad_iters=0, **_BASE)
    return MORPHTransformer(cfg)


def _plain_batch(B=2, n=48, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(5, V, size=(B, n)).astype(np.int64))


# ── shared math helpers ───────────────────────────────────────────────────────

def test_spearman_rho_matches_known_cases():
    from _diag_common import spearman_rho
    rho, n = spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert n == 5 and abs(rho - 1.0) < 1e-9
    rho, n = spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert abs(rho - (-1.0)) < 1e-9
    rho, n = spearman_rho([1, 2, 2, 4], [1, 2, 2, 4])   # ties, average ranks
    assert abs(rho - 1.0) < 1e-9


def test_jacobian_top2_recovers_known_singular_values():
    from _diag_common import jacobian_top2
    torch.manual_seed(0)
    n = 24
    # W with a KNOWN, well-separated top-2 singular spectrum: diag(5, 3, 1, 1, ..., 1).
    U, _ = torch.linalg.qr(torch.randn(n, n))
    Vt, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.ones(n)
    s[0], s[1] = 5.0, 3.0
    W = (U * s) @ Vt.T
    # S=1: a longer sequence axis would repeat this SAME W^T W block once per position,
    # giving every eigenvalue (including the top one) multiplicity >= S — a genuine gap
    # needs the flattened operator to be exactly W^T W, not a block-diagonal repeat of it.
    h0 = torch.randn(1, 1, n)
    mask = torch.ones(1, 1, 1)

    def fn(h):
        return h @ W.t()

    sigma1, sigma2 = jacobian_top2(fn, h0, mask, n_iter=60, seed=0)
    assert abs(sigma1 - 5.0) / 5.0 < 1e-3
    assert abs(sigma2 - 3.0) / 3.0 < 1e-2
    assert sigma1 >= sigma2 >= 0.0


def test_settling_time_and_entropy():
    from basin_map import _entropy, _settling_time

    assert _settling_time([7, 7, 7, 7]) == 1          # never changes: settles at 1
    assert _settling_time([1, 2, 2, 2]) == 2           # last change after depth 1
    assert _settling_time([1, 2, 3, 3]) == 3
    assert _settling_time([1, 2, 3, 4]) == 4           # still changing at the last depth

    # a perfectly flat field: entropy 0, every point in one bucket
    h_flat = _entropy({1: 100}, 100)
    assert h_flat == 0.0
    # maximally spread over 4 buckets of 25 each: entropy = log(4)
    import math
    h_spread = _entropy({1: 25, 2: 25, 3: 25, 4: 25}, 100)
    assert abs(h_spread - math.log(4)) < 1e-9


# ── aa_score.py ────────────────────────────────────────────────────────────────

def test_aa_score_slot_loop_cosines_are_bounded_and_self_consistent():
    from aa_score import _readout_vec, _run_once_and_twice_slot

    model = _slot_model()
    model.eval()
    inp, _y, layout, _ = _slot_batch()
    fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
    h_once, h_swap, h_noise, valid = _run_once_and_twice_slot(model, inp, layout, fkw, freset)
    assert valid.sum() > 0
    ro_once = _readout_vec(model, h_once)
    ro_swap = _readout_vec(model, h_swap)
    ro_noise = _readout_vec(model, h_noise)
    import torch.nn.functional as F
    c_swap = F.cosine_similarity(ro_once.float(), ro_swap.float(), dim=-1)[valid]
    c_noise = F.cosine_similarity(ro_once.float(), ro_noise.float(), dim=-1)[valid]
    assert torch.isfinite(c_swap).all() and torch.isfinite(c_noise).all()
    assert (c_swap >= -1.0001).all() and (c_swap <= 1.0001).all()
    assert (c_noise >= -1.0001).all() and (c_noise <= 1.0001).all()
    # h_once must be the model's OWN real forward, i.e. identical to a fresh call
    x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    with torch.no_grad():
        out2 = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    assert torch.equal(h_once, out2[1])


def test_aa_score_plain_core_runs_and_bounds_hold():
    from aa_score import _readout_vec, _run_once_and_twice_plain

    model = _plain_model()
    model.eval()
    inp = _plain_batch()
    h_once, h_swap, h_noise, valid = _run_once_and_twice_plain(model, inp)
    assert bool(valid.all())
    ro_once = _readout_vec(model, h_once)
    ro_swap = _readout_vec(model, h_swap)
    import torch.nn.functional as F
    c = F.cosine_similarity(ro_once.float(), ro_swap.float(), dim=-1)
    assert torch.isfinite(c).all()
    assert (c >= -1.0001).all() and (c <= 1.0001).all()


# ── jacobian_gap_vs_k.py ─────────────────────────────────────────────────────

def test_jacobian_gap_vs_k_end_to_end_on_slot_loop():
    from _diag_common import row_token_ce
    from jacobian_gap_vs_k import _capture_iter0, _sigma_at_point

    model = _slot_model()
    model.eval()
    inp, labels, layout, _ = _slot_batch(B=1)

    tc = model.cfg.tul
    orig_mean, orig_max = int(tc.slot_mean_depth), int(tc.slot_max_depth)
    tc.slot_mean_depth, tc.slot_max_depth = 1, max(1, orig_max or model.cfg.max_depth)
    s1_sum, s1_cnt = row_token_ce(model, inp, labels, layout, "cpu")
    tc.slot_mean_depth, tc.slot_max_depth = 3, max(3, orig_max or model.cfg.max_depth)
    s3_sum, s3_cnt = row_token_ce(model, inp, labels, layout, "cpu")
    tc.slot_mean_depth, tc.slot_max_depth = orig_mean, orig_max
    assert s1_cnt > 0 and s3_cnt > 0

    def fwd():
        fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
        x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        model._tul_core(x, x0, bigram, layout, input_ids=inp)

    point, root = _capture_iter0(model, fwd)
    assert int(point["iter_idx"]) == 0
    assert bool(point["active"].any())
    sigma1, sigma2 = _sigma_at_point(root, point, n_iter=15, seed=0)
    model._jac_capture = None
    assert sigma1 >= sigma2 >= 0.0
    assert sigma1 > 0.0    # a freshly-initialised block is not the zero map


def test_jacobian_gap_vs_k_plain_core():
    from jacobian_gap_vs_k import _capture_iter0, _sigma_at_point

    model = _plain_model()
    model.eval()
    inp = _plain_batch(B=1)

    def fwd():
        x, x0, bigram = model._front_region(inp)
        model._core_region(x, x0, bigram)

    point, root = _capture_iter0(model, fwd)
    sigma1, sigma2 = _sigma_at_point(root, point, n_iter=15, seed=0)
    model._jac_capture = None
    assert sigma1 >= sigma2 >= 0.0


# ── sink_mass_per_pass.py ────────────────────────────────────────────────────

def test_sink_mass_per_pass_slot_loop_bucketing_and_bounds():
    from sink_mass_per_pass import PassSinkStats, capture_sink_mass

    model = _slot_model()
    model.eval()
    inp, _y, layout, _ = _slot_batch(B=2)
    n_core = int(model.cfg.n_core)
    stats = PassSinkStats(n_core)
    sink_pos = torch.zeros(inp.shape[0], dtype=torch.long)   # compact slot index 0
    fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
    with torch.no_grad():
        x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        with capture_sink_mass(sink_pos, stats):
            model._tul_core(x, x0, bigram, layout, input_ids=inp)
    assert stats.per_pass, "no window-branch calls were captured"
    # exactly n_core calls captured per pass (the loop calls every core layer once/iter)
    for p, calls in stats.per_pass.items():
        assert len(calls) == n_core
    for p, calls in stats.per_pass.items():
        for c in calls:
            for v in c:
                assert -1e-6 <= v <= 1.0 + 1e-6


def test_sink_mass_per_pass_plain_core():
    from sink_mass_per_pass import PassSinkStats, capture_sink_mass

    model = _plain_model()
    model.eval()
    inp = _plain_batch(B=2)
    n_core = int(model.cfg.n_core)
    stats = PassSinkStats(n_core)
    sink_pos = torch.zeros(inp.shape[0], dtype=torch.long)
    with torch.no_grad():
        x, x0, bigram = model._front_region(inp)
        with capture_sink_mass(sink_pos, stats):
            model._core_region(x, x0, bigram)
    assert stats.per_pass
    for p, calls in stats.per_pass.items():
        assert len(calls) == n_core


# ── basin_map.py ─────────────────────────────────────────────────────────────

def test_basin_map_orthonormal_pair_is_orthonormal():
    from basin_map import _orthonormal_pair
    d1, d2 = _orthonormal_pair((4, 16), "cpu", seed=7)
    assert abs(float((d1 * d1).sum()) - 1.0) < 1e-5
    assert abs(float((d2 * d2).sum()) - 1.0) < 1e-5
    assert abs(float((d1 * d2).sum())) < 1e-5


def test_basin_map_slot_loop_small_grid_end_to_end():
    from basin_map import run_one_pair

    model = _slot_model()
    model.eval()
    inp, _y, layout, _ = _slot_batch(B=1)

    class _Arm:
        pass

    arm = _Arm()
    arm.model = model
    arm.is_slot_loop = True

    valid_idx = layout.slot_valid[0].nonzero().flatten()
    assert valid_idx.numel() >= 1
    target_idx = int(valid_idx[0])
    res = run_one_pair(arm, inp, None, layout, target_idx, grid=5, radius=1.0,
                       depths=[1, 2, 3], seed=0, chunk=8, device="cpu")
    assert res["n_points"] == 25
    assert sum(res["settling_time_counts"].values()) == 25
    assert 0.0 <= res["fraction_differing_from_centre"] <= 1.0
    assert res["settling_time_entropy_nats"] >= 0.0
    assert res["center_settling_time"] in (1, 2, 3)


def test_basin_map_plain_core_small_grid_end_to_end():
    from basin_map import run_one_pair

    model = _plain_model()
    model.eval()
    inp = _plain_batch(B=1)

    class _Arm:
        pass

    arm = _Arm()
    arm.model = model
    arm.is_slot_loop = False

    res = run_one_pair(arm, inp, None, None, target_idx=8, grid=5, radius=1.0,
                       depths=[1, 2, 3], seed=0, chunk=8, device="cpu")
    assert res["n_points"] == 25
    assert sum(res["settling_time_counts"].values()) == 25
    assert 0.0 <= res["fraction_differing_from_centre"] <= 1.0
