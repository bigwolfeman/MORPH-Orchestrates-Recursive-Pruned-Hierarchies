"""``tul.code_t_logit_mean`` / ``code_t_logit_std`` — the flow thinker's TRAINING noise
schedule — and ``val/code_ca``, LCM's contrastive accuracy on the sampled code.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_tul_code_tdist.py -q

WHAT THIS FILE HAS TO PROVE:

1. THE DEFAULT IS THE TREE. ``code_t_logit_mean: null`` draws the same numbers from the
   same RNG state and leaves the stream in the same place as the bare ``torch.rand`` the
   flow site used before the knob — checked against a manual draw AND against a monkey-
   patched pre-knob model forward (loss, flow term and RNG state all bit-identical).
2. THE DRAW IS THE ONE ASKED FOR. At ``mu = -1, sigma = 1`` the mean and ``P(t < 0.5)``
   match the analytic values (quadrature and ``Phi(1)``) on 200k draws, and the clamp
   holds at the extremes.
3. IT MOVES THE LOSS'S WEIGHT. The four ``code_fm_band{b}_rel`` bands are quarters of t,
   and under ``mu = -1`` the low two bands take 84 % of the slots against uniform's 50 %.
   A real model forward reports ``code_t_mean`` at the schedule's mean.
4. CA IS A REAL RETRIEVAL. Identical codes read 1.0, a shuffle reads chance, a prediction
   equal to its NEIGHBOUR's code is not a hit and the neighbour is not a candidate (the
   candidate COUNT, i.e. ``chance``, is checked against hand arithmetic), and pads are out.
5. REFUSALS: a non-positive sigma, the keys on a discrete code, the keys at
   ``code_fm_weight 0``, and the keys with ``code: false``.
6. The keys compose through Hydra + ``tul_setup`` and ``tul_code_cfg_tlow`` resolves.

CPU only, fp32, the ``tests/test_tul_code.py`` fixtures.
Prereg: lab/experiments/planned/2026-09-21-lctul-cfg-tlow.md
"""
from __future__ import annotations

import math

import pytest
import torch

from test_tul_code import _arm_head, _model  # noqa: E402  (tests/ is on sys.path)
from test_tul_gl1 import _batch, _tul  # noqa: E402

from morph.model.tul import TULConfig
from morph.model.tul_code import code_contrastive_accuracy, draw_flow_t


class _TC:
    """The two fields :func:`draw_flow_t` reads, free of a whole TULConfig."""

    def __init__(self, mean=None, std=1.0):
        self.code_t_logit_mean = mean
        self.code_t_logit_std = std


def _analytic_mean_t(mu: float, sigma: float, n: int = 400_001) -> float:
    """``E[sigmoid(mu + sigma*eps)]``, ``eps ~ N(0,1)``, by trapezoid quadrature over
    ``[-14, 14]`` — the reference the sampler is checked against (it is NOT sigmoid(mu),
    which is the MEDIAN)."""
    e = torch.linspace(-14.0, 14.0, n, dtype=torch.float64)
    pdf = torch.exp(-0.5 * e * e) / math.sqrt(2.0 * math.pi)
    t = torch.sigmoid(mu + sigma * e)
    return float(torch.trapz(t * pdf, e))


_PHI1 = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))          # 0.8413447...


# ── 1. the default is the tree ───────────────────────────────────────────────

def test_default_draw_is_the_bare_torch_rand_and_consumes_the_same_rng():
    torch.manual_seed(1234)
    st0 = torch.random.get_rng_state()
    t = draw_flow_t(5, 9, _TC(), torch.device("cpu"))
    st1 = torch.random.get_rng_state()
    torch.random.set_rng_state(st0)
    ref = torch.rand(5, 9, dtype=torch.float32)
    assert torch.equal(t, ref), "the default draw is not the pre-knob torch.rand"
    assert torch.equal(st1, torch.random.get_rng_state()), \
        "the default draw moved the RNG stream by a different amount"
    assert t.shape == (5, 9) and t.dtype == torch.float32


def test_default_model_forward_is_bit_identical_to_the_pre_knob_flow_site(monkeypatch):
    """The pre-knob line was ``t = torch.rand(B, S, device=..., dtype=float32)``. Patch it
    back in and every number of a phase-2 train step must agree to the last bit."""
    import morph.model.transformer as T
    x, y, lay, _ = _batch()

    def _run(patch: bool):
        m = _model()
        _arm_head(m)
        m.train()
        m.code_phase = 2
        if patch:
            monkeypatch.setattr(
                T, "draw_flow_t",
                lambda B, S, tc, device, generator=None:
                    torch.rand(B, S, device=device, dtype=torch.float32))
        else:
            monkeypatch.undo()
        torch.manual_seed(77)
        out = m(x, labels=y, slot_layout=lay)
        return (float(out["loss"].detach()), float(out["code_fm_raw"]),
                float(out["code_t_mean"]), torch.random.get_rng_state().clone())

    a = _run(False)
    b = _run(True)
    assert a[0] == b[0], f"loss moved: {a[0]} vs {b[0]}"
    assert a[1] == b[1], f"code_fm_raw moved: {a[1]} vs {b[1]}"
    assert a[2] == b[2], f"code_t_mean moved: {a[2]} vs {b[2]}"
    assert torch.equal(a[3], b[3]), "the RNG stream ended in a different place"
    # ... and the default schedule really is uniform, so the mean sits near 0.5
    assert 0.40 < a[2] < 0.60, a[2]


# ── 2. the draw is the one asked for ─────────────────────────────────────────

def test_logit_normal_draw_matches_its_analytic_mean_and_tail():
    torch.manual_seed(5)
    t = draw_flow_t(200, 1000, _TC(-1.0, 1.0), torch.device("cpu"))     # 200k draws
    ref_mean = _analytic_mean_t(-1.0, 1.0)
    assert abs(ref_mean - 0.303265) < 1e-4, ref_mean       # the quadrature itself
    got = float(t.mean())
    assert abs(got - ref_mean) < 0.01, f"mean t {got} vs analytic {ref_mean}"
    p_lo = float((t < 0.5).float().mean())
    assert abs(p_lo - _PHI1) < 0.01, f"P(t<0.5) {p_lo} vs Phi(1) {_PHI1}"
    # the median is sigmoid(mu), NOT the mean
    assert abs(float(t.median()) - 1.0 / (1.0 + math.exp(1.0))) < 0.01
    assert float(t.min()) >= 1e-4 and float(t.max()) <= 1.0 - 1e-4


def test_sigma_scales_the_spread_and_mu_shifts_the_mass():
    torch.manual_seed(6)
    narrow = draw_flow_t(100, 500, _TC(-1.0, 0.25), torch.device("cpu"))
    wide = draw_flow_t(100, 500, _TC(-1.0, 2.0), torch.device("cpu"))
    assert float(narrow.std()) < float(wide.std())
    assert abs(float(narrow.mean()) - _analytic_mean_t(-1.0, 0.25)) < 0.01
    assert abs(float(wide.mean()) - _analytic_mean_t(-1.0, 2.0)) < 0.01
    high = draw_flow_t(100, 500, _TC(3.0, 1.0), torch.device("cpu"))
    assert abs(float(high.mean()) - _analytic_mean_t(3.0, 1.0)) < 0.01
    assert float(high.mean()) > float(wide.mean()) > float(narrow.mean())


# ── 3. it moves the flow loss's weight ───────────────────────────────────────

def _band_counts(t: torch.Tensor) -> list[float]:
    """The four bands ``code_fm_band{b}_rel`` is reported over: ``[b/4, (b+1)/4)``
    (transformer.py's flow branch), as fractions of the draw."""
    n = float(t.numel())
    return [float(((t >= b / 4.0) & (t < (b + 1) / 4.0)).sum()) / n for b in range(4)]


def test_low_t_bands_take_the_weight_under_mu_minus_one():
    torch.manual_seed(7)
    uni = _band_counts(draw_flow_t(200, 1000, _TC(), torch.device("cpu")))
    low = _band_counts(draw_flow_t(200, 1000, _TC(-1.0, 1.0), torch.device("cpu")))
    assert all(abs(c - 0.25) < 0.01 for c in uni), uni
    assert abs(sum(uni[:2]) - 0.5) < 0.01, uni
    # P(t < 0.5) = Phi(1) = 0.8413 and P(t < 0.25) = 0.4607 (analytic)
    assert abs(sum(low[:2]) - _PHI1) < 0.01, low
    assert abs(low[0] - 0.4607) < 0.01, low
    assert low[3] < 0.02, low                      # the SNR>9 corner is nearly abandoned


def test_the_model_reports_code_t_mean_at_the_schedule_and_the_band_keys():
    x, y, lay, _ = _batch()
    m = _model(tul_code_t_logit_mean=-1.0)
    _arm_head(m)
    m.train()
    m.code_phase = 2
    torch.manual_seed(31)
    out = m(x, labels=y, slot_layout=lay)
    got = float(out["code_t_mean"])
    assert 0.15 < got < 0.45, f"code_t_mean {got}: the schedule did not reach the forward"
    assert any(f"code_fm_band{b}_rel" in out for b in range(4)), sorted(out)
    # the uniform twin on the SAME batch reads the uniform mean
    m2 = _model()
    _arm_head(m2)
    m2.train()
    m2.code_phase = 2
    torch.manual_seed(31)
    out2 = m2(x, labels=y, slot_layout=lay)
    assert float(out2["code_t_mean"]) > got + 0.10, (float(out2["code_t_mean"]), got)


# ── 4. contrastive accuracy ──────────────────────────────────────────────────

def _codes(n_rows: int, n_slots: int, m: int = 2, c: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_rows, n_slots, m, c, generator=g)
    row = torch.arange(n_rows).view(-1, 1).expand(n_rows, n_slots)
    slot = torch.arange(n_slots).view(1, -1).expand(n_rows, n_slots)
    return z, row, slot


def test_ca_identical_prediction_is_one():
    z, row, slot = _codes(4, 12, seed=1)
    ok = torch.ones(4, 12, dtype=torch.bool)
    acc, chance = code_contrastive_accuracy(z.clone(), z, ok, row, slot)
    assert acc == 1.0
    assert 0.0 < chance < 1.0


def test_ca_shuffled_prediction_reads_chance():
    z, row, slot = _codes(8, 16, c=32, seed=2)
    ok = torch.ones(8, 16, dtype=torch.bool)
    g = torch.Generator().manual_seed(9)
    flat = z.reshape(-1, 2, 32)
    perm = torch.randperm(flat.shape[0], generator=g)
    acc, chance = code_contrastive_accuracy(flat[perm].reshape(z.shape), z, ok, row, slot)
    # 128 valid slots. Per row: 2 end slots keep 127 candidates, 14 interior keep 126.
    want = (8 * 2 * (1.0 / 127.0) + 8 * 14 * (1.0 / 126.0)) / 128.0
    assert abs(chance - want) < 1e-6, (chance, want)
    assert acc <= 10.0 * chance, f"a shuffle scored {acc} against chance {chance}"


def test_ca_excludes_the_two_temporal_neighbours_of_the_same_row():
    """Designed codes, one orthonormal axis each, so every cosine below is exact."""
    n = 5
    flat = torch.eye(n, 16)
    z = flat.reshape(1, n, 2, 8)
    ok = torch.ones(1, n, dtype=torch.bool)
    row = torch.zeros(1, n, dtype=torch.long)
    slot = torch.arange(n).view(1, n)
    # (a) THE NEIGHBOUR IS NOT A CANDIDATE. A prediction pointing MOSTLY at its next
    # neighbour (cos 0.874) and a little at itself (cos 0.486) retrieves ITSELF — which is
    # only possible if the neighbour was taken out of the pool. With the pool intact every
    # one of these slots would retrieve the neighbour and acc would be 1/5.
    pred = flat.clone()
    pred[:4] = 0.9 * flat[1:] + 0.5 * flat[:4]
    acc, chance = code_contrastive_accuracy(pred.reshape(1, n, 2, 8), z, ok, row, slot)
    assert acc == 1.0, acc
    # ... and the candidate COUNT says the same: the two end slots keep 4 of 5, the three
    # interior slots keep 3, so chance = (2*(1/4) + 3*(1/3)) / 5 = 0.3
    assert abs(chance - 0.3) < 1e-6, chance
    # (b) A PREDICTION EQUAL TO ITS NEIGHBOUR'S CODE IS NOT A HIT. Slots 0..3 predict the
    # next slot's code (minus a little of their own, so the own code cannot win a tie);
    # only slot 4, which predicts its OWN code, may score.
    pred2 = flat.clone()
    pred2[:4] = flat[1:] - 0.1 * flat[:4]
    acc2, _ = code_contrastive_accuracy(pred2.reshape(1, n, 2, 8), z, ok, row, slot)
    assert abs(acc2 - 1.0 / n) < 1e-6, acc2


def test_ca_neighbour_exclusion_is_scoped_to_the_row_and_pads_are_out():
    z, row, slot = _codes(2, 5, seed=4)
    ok = torch.ones(2, 5, dtype=torch.bool)
    _acc, chance = code_contrastive_accuracy(z.clone(), z, ok, row, slot)
    # 10 valid slots; 4 end slots lose 1 candidate, 6 interior lose 2
    assert abs(chance - (4 * (1.0 / 9.0) + 6 * (1.0 / 8.0)) / 10.0) < 1e-6, chance
    ok2 = ok.clone()
    ok2[0, 4] = False                          # a pad: out of the pool AND not a query
    acc2, chance2 = code_contrastive_accuracy(z.clone(), z, ok2, row, slot)
    assert acc2 == 1.0
    # 9 valid slots. Row 0 keeps 0..3: slots 0 and 3 lose 1 neighbour, 1 and 2 lose 2.
    # Row 1 keeps 0..4: slots 0 and 4 lose 1, 1..3 lose 2.
    want = (2 * (1.0 / 8.0) + 2 * (1.0 / 7.0) + 2 * (1.0 / 8.0) + 3 * (1.0 / 7.0)) / 9.0
    assert abs(chance2 - want) < 1e-6, (chance2, want)
    # a pad slot's own cells cannot change the reading
    zp = z.clone()
    zp[0, 4] = 1e3
    assert code_contrastive_accuracy(zp, zp, ok2, row, slot) == (acc2, chance2)


def test_ca_empty_batch_is_zero_and_shape_mismatch_raises():
    z, row, slot = _codes(2, 3, seed=5)
    assert code_contrastive_accuracy(z, z, torch.zeros(2, 3, dtype=torch.bool),
                                     row, slot) == (0.0, 0.0)
    with pytest.raises(ValueError, match="pred"):
        code_contrastive_accuracy(z, z[:, :2], torch.ones(2, 3, dtype=torch.bool), row, slot)


def test_val_forward_reports_code_ca_against_its_own_chance():
    x, y, lay, _ = _batch()
    m = _model()
    _arm_head(m)
    m.eval()
    with torch.no_grad():
        out = m(x, labels=y, slot_layout=lay)
    assert "code_ca" in out and "code_ca_chance" in out, sorted(out)
    ca, ch = float(out["code_ca"]), float(out["code_ca_chance"])
    assert 0.0 <= ca <= 1.0 and 0.0 < ch < 1.0, (ca, ch)
    # the ENCODER mode hands the coda the true code, so retrieval there is exact — and it
    # is NOT reported, because CA is a statement about the SAMPLER.
    with torch.no_grad():
        enc = m.tul_forward_ablated(x, y, lay, code_mode="encoder")
    assert "code_ca" not in enc, sorted(enc)


# ── 5. refusals ──────────────────────────────────────────────────────────────

def _code_tul(**kw) -> TULConfig:
    base = dict(tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict",
                sigreg_lambda=0.0, mux_beta=0.0, token_state_dropout=0.0, code=True)
    base.update(kw)
    return _tul(**base)


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_non_positive_sigma_raises(std):
    with pytest.raises(ValueError, match="code_t_logit_std"):
        _code_tul(code_t_logit_std=std)


def test_the_keys_are_refused_on_a_discrete_code():
    with pytest.raises(ValueError, match="FLOW-thinker"):
        _code_tul(code_discrete=True, code_t_logit_mean=-1.0)
    with pytest.raises(ValueError, match="FLOW-thinker"):
        _code_tul(code_discrete=True, code_t_logit_std=2.0)


def test_the_keys_are_refused_when_the_flow_term_has_no_weight():
    with pytest.raises(ValueError, match="code_fm_weight"):
        _code_tul(code_fm_weight=0.0, code_t_logit_mean=-1.0)
    with pytest.raises(ValueError, match="code_fm_weight"):
        _code_tul(code_fm_weight=0.0, code_t_logit_std=2.0)
    _code_tul(code_fm_weight=0.0)                       # the default pair is fine there


def test_the_keys_are_refused_with_code_false():
    with pytest.raises(ValueError, match="code_t_logit_mean"):
        _tul(code=False, code_t_logit_mean=-1.0)
    with pytest.raises(ValueError, match="code_t_logit_std"):
        _tul(code=False, code_t_logit_std=2.0)


def test_a_set_schedule_builds_and_the_default_stays_none():
    assert _code_tul().code_t_logit_mean is None
    assert _code_tul().code_t_logit_std == 1.0
    tc = _code_tul(code_t_logit_mean=-1.0, code_t_logit_std=1.5)
    assert tc.code_t_logit_mean == -1.0 and tc.code_t_logit_std == 1.5


# ── 6. the config composes ───────────────────────────────────────────────────

def test_the_keys_compose_through_tul_setup_and_tlow_resolves():
    import os

    from hydra import compose, initialize_config_dir

    from morph.training.tul_setup import KNOWN_TUL_KEYS, build_tul_runtime
    assert "code_t_logit_mean" in KNOWN_TUL_KEYS and "code_t_logit_std" in KNOWN_TUL_KEYS
    cdir = os.path.abspath("morph/configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name="tul_code_cfg_tlow")
    assert cfg.wandb.name == "tul-code-cfg-tlow"
    rt = build_tul_runtime(cfg)
    assert rt.model_cfg.code_t_logit_mean == -1.0
    assert rt.model_cfg.code_t_logit_std == 1.0
    assert rt.model_cfg.code_cfg_drop == 0.1 and rt.model_cfg.code_cfg_scale == 2.0
    assert rt.manifest["code_t_logit_mean"] == -1.0
    assert rt.manifest["code_t_logit_std"] == 1.0
    # the parent keeps the uniform draw
    with initialize_config_dir(config_dir=cdir, version_base=None):
        parent = compose(config_name="tul_code_cfg")
    assert build_tul_runtime(parent).model_cfg.code_t_logit_mean is None
