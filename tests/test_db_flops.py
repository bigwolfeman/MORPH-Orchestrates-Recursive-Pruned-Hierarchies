"""FLOP-model tests (gate A3). CPU-only, no model build, no CUDA.

The contract these defend: ``flop_proxy`` is NOMINAL and comparable run to run, while
``layer_passes_per_token`` is REALIZED. The original A3 gate demanded "A0 reports exactly
44.0" from a realized counter, which is impossible — clamp(Poisson(6),1,8) has mean 5.688.
Conflating the two is the bug these tests exist to prevent.
"""

import pytest

from morph.training.flops import (
    FLOP_MODEL_VERSION,
    FlopModel,
    attention_flops_per_token,
    expected_clamped_poisson,
)


def _a0_model() -> FlopModel:
    """MORPH local shape: 4 prelude / 6 core / 4 coda, T̄ = 6, cap 8."""
    return FlopModel(
        version=FLOP_MODEL_VERSION, d_model=1024,
        n_prelude=4, n_core=6, n_coda=4,
        mean_depth=6, max_depth=8, seq_len=1024,
        gemm_prelude=4_000_000, gemm_core=6_000_000, gemm_coda=4_000_000,
        gemm_other=1_000_000,
    )


def test_a0_nominal_flop_proxy_is_exactly_44():
    """The pre-registered anchor: 4 + 6·6 + 4 = 44.0, no TUL, no concat."""
    assert _a0_model().flop_proxy() == pytest.approx(44.0)


def test_a0_realized_passes_are_about_42_not_44():
    fm = _a0_model()
    realized = expected_clamped_poisson(6.0, 1, 8)
    assert fm.layer_passes_per_token(realized) == pytest.approx(42.13, abs=0.05)
    assert fm.layer_passes_per_token(realized) < fm.flop_proxy()


def test_tul_shape_reproduces_the_ledger_anchor():
    """TUL: core on 64 of 1024 positions, L_total/seq = 1152/1024 = 1.125.

    The ledger's measured realized anchor is 10.68 layer-passes/token. This asserts the
    counter lands in that neighbourhood — if it does not, the counter is wrong, not the
    ledger (gate A3).
    """
    fm = _a0_model()
    realized = expected_clamped_poisson(6.0, 1, 8)
    got = fm.layer_passes_per_token(
        realized, positions_per_token=1.125, core_position_frac=64 / 1024)
    assert got == pytest.approx(10.68, abs=1.0), got


def test_db_b3_uniform_visit_is_the_preregistered_4_67():
    """DB-2's pre-registered nominal passes with x0_inject: (4 + 6 + 4)/3 = 4.67.

    A B=3 step runs exactly ONE section with the core applied ONCE, so the comparable
    number is the expectation over the visit distribution.
    """
    fm = _a0_model()
    got = fm.db_expected_passes([1 / 3, 1 / 3, 1 / 3], depth=1.0)
    assert got == pytest.approx(4.667, abs=0.01), got
    # ~9.4x under the A0 anchor -- the headline compute claim
    assert fm.flop_proxy() / got == pytest.approx(9.43, abs=0.1)


def test_db_b3_concat_is_the_preregistered_9_34():
    """With the clean|noisy concatenation: (8 + 12 + 8)/3 = 9.33."""
    fm = _a0_model()
    got = fm.db_expected_passes([1 / 3, 1 / 3, 1 / 3], depth=1.0,
                                positions_per_token=2.0)
    assert got == pytest.approx(9.33, abs=0.02), got


def test_mass_visits_are_cheaper_but_starve_the_ends():
    """Arm DB-12. Mass-proportional visits favour the core, which is the CHEAPEST section
    per step here (6 layers vs the prelude+coda's 8 combined), so the expected cost differs
    from uniform -- the arm is about update allocation, not only about FLOPs."""
    fm = _a0_model()
    uni = fm.db_expected_passes([1 / 3, 1 / 3, 1 / 3], depth=1.0)
    mass = fm.db_expected_passes([1 / 8, 6 / 8, 1 / 8], depth=1.0)
    assert mass == pytest.approx(5.5, abs=0.01)
    assert mass != pytest.approx(uni, abs=0.1)


def test_flop_proxy_does_not_double_count_positions():
    """Regression: an earlier version multiplied by positions_per_token TWICE.

    The prelude and coda scale linearly with positions and the core scales with its own
    position fraction. At ppt=2 the correct A0-shaped value is 8·2 + 6·6·2 = 88, i.e.
    exactly 2× — NOT 4× (which is what double-counting gives).
    """
    fm = _a0_model()
    one = fm.flop_proxy(positions_per_token=1.0)
    two = fm.flop_proxy(positions_per_token=2.0)
    assert one == pytest.approx(44.0)
    assert two == pytest.approx(88.0)
    assert two / one == pytest.approx(2.0, abs=1e-6)


def test_b1_mode_runs_every_section_with_one_core_pass():
    """mode='b1' is the whole net as one denoiser: 4 + 6·1 + 4 = 14 at ppt=1."""
    fm = _a0_model()
    got = fm.layer_passes_per_token(depth=1.0, positions_per_token=1.0)
    assert got == pytest.approx(14.0)
    assert fm.layer_passes_per_token(depth=1.0, positions_per_token=2.0) == pytest.approx(28.0)


def test_db_expected_passes_rejects_a_non_b3_visit_vector():
    fm = _a0_model()
    with pytest.raises(ValueError, match="B=3 form"):
        fm.db_expected_passes([1.0], depth=1.0)


def test_step_flops_scale_with_batch_and_depth():
    fm = _a0_model()
    a, _ = fm.step_flops(batch=1, seq_len=1024, depth=6.0)
    b, _ = fm.step_flops(batch=2, seq_len=1024, depth=6.0)
    c, _ = fm.step_flops(batch=1, seq_len=1024, depth=12.0)
    assert b == pytest.approx(2 * a, rel=1e-6)
    assert c > a


def test_density_scales_only_the_gemms():
    """Dense is 1.0 for the whole campaign (plan O3); the knob must still be wired."""
    fm = _a0_model()
    full, attn_full = fm.step_flops(batch=1, seq_len=1024, depth=6.0, density=1.0)
    half, attn_half = fm.step_flops(batch=1, seq_len=1024, depth=6.0, density=0.5)
    assert half < full
    assert attn_half == attn_full, "attention is not weight-sparse; density must not touch it"


def test_attention_model_reports_its_own_assumptions():
    """The modelled half must be self-documenting — it is the approximate term."""
    flops, assumptions = attention_flops_per_token(
        d_model=1024, n_heads=8, n_kv_heads=4, seq_len=1024,
        compression=2, csa_ratio=8, hca_ratio=256, top_k=256, window=256)
    assert flops > 0
    assert assumptions["attn_d_eff"] == 512
    assert "MODELLED" in assumptions["attn_note"]
    assert assumptions["attn_keys_total"] == (
        assumptions["attn_keys_csa"] + assumptions["attn_keys_hca"]
        + assumptions["attn_keys_local"])


def test_manifest_carries_the_version():
    """An MFU without its model version is not comparable to another run's."""
    m = _a0_model().manifest()
    assert m["flops/version"] == FLOP_MODEL_VERSION
    assert m["flops/nominal_proxy_a0"] == pytest.approx(44.0)
    assert m["flops/realized_depth"] == pytest.approx(5.688, abs=0.002)
