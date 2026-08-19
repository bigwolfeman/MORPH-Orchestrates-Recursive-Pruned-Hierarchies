"""DiffusionBlocks unit tests. CPU-only — no CUDA, no model build, no training.

Every test here asserts on a CONTRACT, not on a shape or a truthiness. The two that earn
their keep are :func:`test_euler_step_lands_at_next_sigma` and
:func:`test_clean_noisy_mask_never_leaks_the_target`: each catches an error that would
leave the loss curve looking healthy while the experiment taught the model nothing.

Docs: docs/diffusionblocks-plan-of-action.md, docs/diffusionblocks-reference-audit.md.
"""

import math

import pytest
import torch

from morph.model.diffusion_blocks import (
    P_MEAN,
    P_STD,
    SIGMA_MAX,
    SIGMA_MIN,
    AdaLNGate,
    DBConfig,
    DBSchedule,
    EDMPrecond,
    SigmaConditioning,
    SliceScaler,
    clean_noisy_mask,
    euler_step,
    expected_embedding,
)
from morph.training.flops import expected_clamped_poisson


# ── the decisive sign test ───────────────────────────────────────────────────

def test_euler_step_lands_at_next_sigma():
    """A PERFECT denoiser must move z from noise level σ to exactly σ_next.

    This is the test that settles the paper's sign typo (audit §2). With the paper's
    RENDERED Eq (3)-(5) sign, z would land at 2σ − σ_next, i.e. noise would INCREASE as
    the schedule descends. If someone "corrects" euler_step back to the paper, this fails.
    """
    torch.manual_seed(0)
    B, L, D = 4, 7, 16
    y = torch.randn(B, L, D)
    eps = torch.randn(B, L, D)
    sigma = torch.tensor([10.0, 5.0, 2.0, 0.5])
    next_sigma = torch.tensor([4.0, 2.0, 0.5, 0.1])

    z = y + sigma.view(B, 1, 1) * eps
    out = euler_step(z, denoised=y, sigma=sigma, next_sigma=next_sigma)
    expected = y + next_sigma.view(B, 1, 1) * eps

    assert torch.allclose(out, expected, atol=1e-5), (
        f"perfect denoiser did not land at σ_next; max err "
        f"{(out - expected).abs().max().item():.3e}"
    )


def test_euler_step_is_a_contraction_toward_the_denoiser():
    """The seam must be a CONVEX blend: α = σ_next/σ ∈ (0,1), α + (1−α) = 1.

    This is the ρ(J) ≤ 1 handle the nested-dynamics note asks for. A step that
    EXTRAPOLATES away from D (α > 1) would be an expansion and would break the argument.
    """
    z = torch.tensor([[[4.0]]])
    d = torch.tensor([[[1.0]]])
    sigma, next_sigma = torch.tensor([8.0]), torch.tensor([2.0])
    out = euler_step(z, d, sigma, next_sigma)
    alpha = 2.0 / 8.0
    assert torch.allclose(out, alpha * z + (1 - alpha) * d, atol=1e-6)
    # strictly between d and z => contraction, not extrapolation
    assert float(d) < float(out) < float(z)


# ── the mask cutoff ──────────────────────────────────────────────────────────

def test_clean_noisy_mask_never_leaks_the_target():
    """Noisy row i must NOT attend clean column i+1 — that column IS its target.

    Layout is [clean_0..clean_{L-1}, noisy_0..noisy_{L-1}] with
    y[t] = embed(labels[t]) = embed(input_ids[t+1]) and clean_t = embed(input_ids[t]).
    So clean_{i+1} is precisely the answer at noisy position i. Allowing it collapses the
    loss to ~0 and teaches nothing.
    """
    L = 6
    m = clean_noisy_mask(L, torch.device("cpu"))
    assert m.shape == (2 * L, 2 * L)
    for i in range(L):
        noisy_row = L + i
        assert m[noisy_row, i], f"noisy {i} must see clean {i} (its legitimate context)"
        if i + 1 < L:
            assert not m[noisy_row, i + 1], (
                f"LEAK: noisy row {i} can attend clean column {i+1}, which is its target")
        # no future clean at all
        for j in range(i + 1, L):
            assert not m[noisy_row, j]


def test_clean_noisy_mask_clean_stream_is_noise_free_and_causal():
    """Clean rows never attend the noisy half, and are causal among themselves."""
    L = 5
    m = clean_noisy_mask(L, torch.device("cpu"))
    assert not m[:L, L:].any(), "clean stream must not attend noisy positions"
    for i in range(L):
        for j in range(L):
            assert bool(m[i, j]) == (j <= i)


# ── schedule ─────────────────────────────────────────────────────────────────

def test_schedule_boundaries_span_the_full_range_and_ascend():
    sch = DBSchedule(DBConfig(mode="b3"), mean_depth=6)
    s = sch.sigmas
    assert len(s) == 4
    assert math.isclose(s[0], SIGMA_MIN, rel_tol=1e-9)
    assert math.isclose(s[-1], SIGMA_MAX, rel_tol=1e-9)
    assert all(s[i] < s[i + 1] for i in range(len(s) - 1)), f"not ascending: {s}"


def test_b3_mass_split_is_one_eighth_six_eighths_one_eighth():
    """1 prelude step + T̄ core steps + 1 coda step, each equal mass, at T̄ = 6."""
    sch = DBSchedule(DBConfig(mode="b3"), mean_depth=6)
    assert sch.mass_layer_order == pytest.approx((1 / 8, 6 / 8, 1 / 8))
    assert sum(sch.mass_layer_order) == pytest.approx(1.0)


def test_b3_mass_split_tracks_mean_depth():
    sch = DBSchedule(DBConfig(mode="b3"), mean_depth=4)
    assert sch.mass_layer_order == pytest.approx((1 / 6, 4 / 6, 1 / 6))


def test_equi_probability_partition_gives_each_block_its_mass():
    """Each block's σ interval must carry exactly its share of the log-normal mass."""
    sch = DBSchedule(DBConfig(mode="b3"), mean_depth=6)

    def cdf(x):
        return 0.5 * (1.0 + math.erf(((math.log(x) - P_MEAN) / P_STD) / math.sqrt(2.0)))

    span = cdf(SIGMA_MAX) - cdf(SIGMA_MIN)
    got = [(cdf(sch.sigmas[i + 1]) - cdf(sch.sigmas[i])) / span
           for i in range(len(sch.sigmas) - 1)]
    # sigmas ascend, mass_layer_order is prelude-first (highest σ) => reversed
    assert got == pytest.approx(list(reversed(sch.mass_layer_order)), abs=1e-9)


def test_b1_is_one_block_covering_everything():
    sch = DBSchedule(DBConfig(mode="b1"), mean_depth=6)
    assert sch.n_blocks == 1
    lo, hi = sch.block_range(0)
    assert lo == pytest.approx(SIGMA_MIN)
    assert hi == pytest.approx(SIGMA_MAX)


def test_gamma_overlap_widens_a_block_both_ways():
    narrow = DBSchedule(DBConfig(mode="b3", overlap_gamma=0.0), mean_depth=6)
    wide = DBSchedule(DBConfig(mode="b3", overlap_gamma=0.1), mean_depth=6)
    lo_n, hi_n = narrow.block_range(1)
    lo_w, hi_w = wide.block_range(1)
    assert lo_w < lo_n and hi_w > hi_n
    # never past the global range
    for b in range(3):
        lo, hi = wide.block_range(b)
        assert lo >= SIGMA_MIN - 1e-12 and hi <= SIGMA_MAX + 1e-12


def test_sampled_sigma_stays_inside_the_block_range():
    sch = DBSchedule(DBConfig(mode="b3"), mean_depth=6)
    for b in range(3):
        lo, hi = sch.block_range(b)
        s = sch.sample_sigma(b, 256, torch.device("cpu"))
        assert s.shape == (256,)
        assert float(s.min()) >= lo * (1 - 1e-6)
        assert float(s.max()) <= hi * (1 + 1e-6)


def test_block_of_sigma_inverts_block_range():
    """σ drawn from block b must map back to block b (ignoring the γ overlap)."""
    sch = DBSchedule(DBConfig(mode="b3", overlap_gamma=0.0), mean_depth=6)
    for b in range(3):
        s = sch.sample_sigma(b, 64, torch.device("cpu"))
        back = sch.block_of_sigma(s)
        assert (back == b).all(), f"block {b} round-trip gave {back.unique().tolist()}"


def test_inference_sigmas_descend():
    """euler_step reads its sign from next_sigma − sigma < 0, so order is load-bearing."""
    sch = DBSchedule(DBConfig(mode="b3"), mean_depth=6)
    s = sch.inference_sigmas(8)
    assert s.shape == (8,)
    assert all(s[i] > s[i + 1] for i in range(len(s) - 1)), f"not descending: {s}"


def test_visit_probs_uniform_vs_mass():
    u = DBSchedule(DBConfig(mode="b3", visit="uniform"), mean_depth=6).visit_probs()
    m = DBSchedule(DBConfig(mode="b3", visit="mass"), mean_depth=6).visit_probs()
    assert u == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert m == pytest.approx([1 / 8, 6 / 8, 1 / 8])
    # the point of the uniform default: it does NOT starve the prelude/coda
    assert u[0] > m[0]


# ── EDM preconditioning ──────────────────────────────────────────────────────

def test_edm_precond_reconstructs_a_perfect_denoiser():
    """With the ideal network output, c_out·hidden + c_skip·z must recover y exactly.

    EDM's coefficients are designed so the ideal network target is
    hidden* = (y − c_skip·z)/c_out. Feeding that back must give y. This catches any typo in
    c_skip / c_out / c_in.
    """
    pre = EDMPrecond(sigma_data=0.5)
    torch.manual_seed(0)
    y = torch.randn(3, 5, 8)
    eps = torch.randn(3, 5, 8)
    sigma = torch.tensor([0.01, 1.0, 50.0])
    z = y + sigma.view(3, 1, 1) * eps

    c_skip, c_out, c_in, _ = pre.coeffs(sigma)
    ideal = (y - c_skip.view(3, 1, 1) * z) / c_out.view(3, 1, 1)
    recon = ideal * c_out.view(3, 1, 1) + z * c_skip.view(3, 1, 1)
    assert torch.allclose(recon, y, atol=1e-4)


def test_edm_coeff_limits():
    pre = EDMPrecond(sigma_data=0.5)
    c_skip, c_out, c_in, c_noise = pre.coeffs(torch.tensor([1e-4, 1e4]))
    # σ → 0: trust the input (c_skip → 1). σ → ∞: ignore it (c_skip → 0).
    assert c_skip[0] > 0.999 and c_skip[1] < 1e-6
    assert c_in[1] < c_in[0]
    assert torch.allclose(c_noise, 0.25 * torch.tensor([1e-4, 1e4]).log())


def test_edm_weight_matches_the_paper_formula():
    pre = EDMPrecond(sigma_data=0.5)
    s = torch.tensor([0.1, 1.0, 10.0])
    expect = (s**2 + 0.25) / (s * 0.5) ** 2
    assert torch.allclose(pre.weight(s), expect)


# ── slice scaling (plan O1) ──────────────────────────────────────────────────

def test_slice_scaler_gives_each_slice_per_component_std_sigma_data():
    """The whole point of O1: σ_data must be LITERALLY true, per slice.

    Built so the Lorentz slice cannot be buried. A whole-vector L2 (what the authors do)
    would leave the two slices' relative scale untouched and fail this.
    """
    sd = 0.5
    euc_dim, lor_dim = 768, 256
    sc = SliceScaler((euc_dim, lor_dim), sigma_data=sd)
    torch.manual_seed(0)
    # deliberately pathological: the Lorentz slice starts ~200x smaller, as it does in
    # the real model (init std 0.005 vs ~1.0).
    x = torch.cat([torch.randn(4, 6, euc_dim),
                   torch.randn(4, 6, lor_dim) * 0.005], dim=-1)
    out = sc(x)

    euc, lor = out[..., :euc_dim], out[..., euc_dim:]
    assert euc.norm(dim=-1).allclose(
        torch.full((4, 6), sd * math.sqrt(euc_dim)), atol=1e-3)
    assert lor.norm(dim=-1).allclose(
        torch.full((4, 6), sd * math.sqrt(lor_dim)), atol=1e-3)
    # per-component RMS is sigma_data on BOTH slices — the imbalance is gone
    assert euc.pow(2).mean(-1).sqrt().allclose(torch.full((4, 6), sd), atol=1e-3)
    assert lor.pow(2).mean(-1).sqrt().allclose(torch.full((4, 6), sd), atol=1e-3)


def test_slice_scaler_rejects_wrong_width():
    sc = SliceScaler((8, 4))
    with pytest.raises(ValueError, match="expected last dim"):
        sc(torch.randn(2, 3, 11))


def test_whole_vector_l2_would_fail_the_same_check():
    """Documents WHY we deviate from the authors: their normalisation cannot fix this."""
    euc_dim, lor_dim = 768, 256
    x = torch.cat([torch.randn(1, 1, euc_dim),
                   torch.randn(1, 1, lor_dim) * 0.005], dim=-1)
    whole = torch.nn.functional.normalize(x, p=2, dim=-1)
    euc_rms = whole[..., :euc_dim].pow(2).mean(-1).sqrt()
    lor_rms = whole[..., euc_dim:].pow(2).mean(-1).sqrt()
    # the ratio survives normalisation -> the Lorentz slice is still ~orders down
    assert (euc_rms / lor_rms).item() > 20.0


# ── config validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,match", [
    (dict(mode="b7"), "db.mode"),
    (dict(conditioning="magic"), "db.conditioning"),
    (dict(visit="weighted"), "db.visit"),
    (dict(overlap_gamma=-0.1), "overlap_gamma"),
    (dict(sigma_data=0.0), "sigma_data"),
    (dict(mode="b1", block_mass=(0.5, 0.5)), "b1"),
    (dict(block_mass=(0.5, 0.4)), "sum to 1.0"),
    (dict(block_mass=(1.5, -0.5)), "> 0"),
])
def test_bad_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DBConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(cfg_scale=1.5), dict(self_conditioning=True)])
def test_unbuilt_arms_raise_instead_of_being_ignored(kwargs):
    """A silently-ignored config key is worse than a missing one (no-theater)."""
    with pytest.raises(NotImplementedError):
        DBConfig(**kwargs)


# ── σ conditioning ───────────────────────────────────────────────────────────

def test_adaln_gate_starts_as_exact_identity():
    """Zero-init means step 0 is the un-conditioned network — same discipline as HC init."""
    g = AdaLNGate(cond_dim=32, dim=16)
    x = torch.randn(2, 5, 16)
    cond = torch.randn(2, 32)
    assert torch.allclose(g(x, cond), x)


def test_adaln_gate_moves_once_trained():
    """And it must NOT be a permanent no-op — a gate that can't act is theater."""
    g = AdaLNGate(cond_dim=32, dim=16)
    torch.nn.init.normal_(g.to_mod.weight, std=0.1)
    x = torch.randn(2, 5, 16)
    assert not torch.allclose(g(x, torch.randn(2, 32)), x)


def test_sigma_conditioning_is_sigma_dependent_and_finite():
    m = SigmaConditioning(cond_dim=64, n_freq=32)
    a = m(torch.tensor([0.25 * math.log(0.01)]))
    b = m(torch.tensor([0.25 * math.log(10.0)]))
    assert a.shape == (1, 64) and torch.isfinite(a).all()
    assert not torch.allclose(a, b), "conditioning must vary with σ"


def test_sigma_conditioning_accepts_a_batch_of_sigmas():
    m = SigmaConditioning(cond_dim=64, n_freq=32)
    out = m(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert out.shape == (4, 64)


# ── sampler bridge ───────────────────────────────────────────────────────────

def test_expected_embedding_is_a_convex_combination():
    """softmax(logits) @ E must be inside the convex hull of the embedding rows."""
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    out = expected_embedding(torch.tensor([[0.0, 0.0, 0.0]]), E)
    assert torch.allclose(out, E.mean(0, keepdim=True), atol=1e-6)


def test_expected_embedding_recovers_a_row_when_confident():
    E = torch.tensor([[3.0, -1.0], [0.0, 1.0]])
    out = expected_embedding(torch.tensor([[100.0, 0.0]]), E)
    assert torch.allclose(out, E[:1], atol=1e-4)


def test_expected_embedding_shrinks_under_uncertainty_documenting_R9():
    """Risk R9: the bridge's norm collapses toward the table mean when unsure.

    Not a bug to fix — it is inherent to the method — but it IS a train/inference mismatch,
    because training always sees a full-scale y. This test exists so the behaviour is
    pinned and nobody 'discovers' it during a sampling run.
    """
    E = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    confident = expected_embedding(torch.tensor([[50.0, -50.0]]), E)
    unsure = expected_embedding(torch.tensor([[0.0, 0.0]]), E)
    assert confident.norm() > 0.99
    assert unsure.norm() < 1e-5


# ── the realized-depth constant the A3 gate asserts on ───────────────────────

def test_expected_clamped_poisson_is_not_the_lambda():
    """clamp(Poisson(6), 1, 8) has mean 5.688 — NOT 6.

    The A3 gate demanded "A0 reports exactly 44.0 passes/token", which a REALIZED counter
    can never produce. This is why flop_proxy (nominal) and layer_passes_per_token
    (realized) are two different keys.
    """
    got = expected_clamped_poisson(6.0, 1, 8)
    assert got == pytest.approx(5.688, abs=0.002), got
    assert got < 6.0
    # realized A0 layer passes = 4 + 6*5.688 + 4 ≈ 42.1, not 44
    assert 4 + 6 * got + 4 == pytest.approx(42.13, abs=0.05)


def test_expected_clamped_poisson_degenerate_bounds():
    assert expected_clamped_poisson(6.0, 3, 3) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        expected_clamped_poisson(6.0, 5, 2)
