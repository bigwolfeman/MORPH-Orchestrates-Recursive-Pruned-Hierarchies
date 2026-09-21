"""``tul.fan_seed_noise`` — LXTUL-P rung P1: the K streams as K SAMPLES.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. The default is 0.0; a model built with the key absent and one built
   with it 0.0 agree on loss, logits, total gradient AND the global RNG state after the
   forward — the last one is the point: a knob that draws nothing must CONSUME nothing,
   or every downstream draw in the run is shifted and the arm is not one factor.
2. THE SCALE IS THE DOCUMENTED ONE. At ``fan_seed_noise: 1.0`` the term added to the K
   cells' entry state has per-channel RMS 1.0 (the seed lives in the ``input_norm``'d
   field), so two streams of one slot differ by RMS sqrt(2) and each stream's deviation
   from its slot's stream-mean is sqrt((K-1)/K). All three are measured on the same
   sample, and the knob is linear (0.5 reads 0.5).
3. IT IS THE SEED AND NOT THE SOURCE. ``e`` — the tensor ``_tul_core`` hands to EVERY
   pass as the injection source — is bit-identical to the off arm's. Noise in ``e`` would
   be one draw re-injected at every pass: a random per-stream trigger, not a sample.
   This is the design decision, so it is a test and not a comment.
4. EVERY FORWARD DRAWS, AT TRAIN AND AT EVAL. Two eval forwards of the same model on the
   same batch differ. The streams ARE samples; a deterministic eval would read a
   different model from the one that trained.
5. PADS GET EXACTLY ZERO.
6. REFUSALS, by name: off a fan arm, at a negative std, under ``core_state_init:
   "noise"`` and under SCSE.
7. The key composes through ``tul_setup`` and the shipped config resolves — AND the
   diversity term it turns off keeps REPORTING (``fan_vol_t*`` / ``fan_epi_t*`` /
   ``fan_stream_cos_t*`` at ``fan_repel_lambda: 0``), which is what makes this arm
   readable against fan4-all on the same axes.

CPU only, fp32, the ``tests/test_tul_fan.py`` fixtures.
Prereg: lab/experiments/planned/2026-09-21-lxtul-fan4-all-noise.md
"""
from __future__ import annotations

import math

import pytest
import torch

from test_tul_fan import _batch, _model, _tul, _tiny, _finite_logit_sum
from morph.model.transformer import MORPHTransformer


K = 4


def _fan(**kw) -> MORPHTransformer:
    """A fan4-all model, the shipped arm's shape, at the tiny fixture's size."""
    base = dict(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all")
    base.update(kw)
    return _model(**base)


def _seed_state(m: MORPHTransformer, inp, layout, seed: int | None = 7):
    """``(h_0, e)`` of the real forward: the carrier ENTERING pass 1 and the injection
    source every pass is handed.

    Read off ``_jac_capture``, which the Jacobian probe already fills with the exact
    operating point of each pass — entry 0 is pass 1's. Nothing is recomputed: a bare
    front would score a strict arm from an unrestricted prelude
    (`instruments-must-use-the-models-tg-kwargs`).
    """
    m.eval()
    m._jac_capture = []
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        m(inp, labels=None, slot_layout=layout)
    caps = m._jac_capture
    m._jac_capture = None
    assert caps, "the fixture must run at least one pass"
    return caps[0]["h"], caps[0]["e"]


def _cells(h: torch.Tensor, n_slots: int) -> torch.Tensor:
    """``[B, S*K, *carrier, C]`` -> ``[B, S, K, C]``, the Hyper-Connection carrier reduced
    by the same mean every single-stream read of it takes (``_cell_readout``)."""
    z = h.reshape(h.shape[0], n_slots, K, *h.shape[2:])
    while z.dim() > 4:
        z = z.mean(dim=-2)
    return z


def _noise_sample(std: float, seeds=(0, 1, 2, 3)):
    """The ADDED term itself, over several batches: ``[N, K, C]`` on valid slots only,
    plus the count of pad cells it had to be zero on."""
    off, on = _fan(), _fan(fan_seed_noise=std)
    live, pads = [], 0
    for sd in seeds:
        _i, inp, lab, layout = _batch(K, seed=sd)
        h_off, e_off = _seed_state(off, inp, layout)
        h_on, e_on = _seed_state(on, inp, layout)
        assert torch.equal(e_off, e_on), (
            "the injection SOURCE moved: the noise must be added to the entry state "
            "`core_init(e)`, never to `e`, which every pass re-injects")
        d = h_on - h_off
        # the term is single-stream, broadcast into the carrier (`_apply_injection`)
        if d.dim() == 4:
            # `allclose`, not `equal`: the term is recovered by SUBTRACTING two carriers
            # whose Hyper-Connection streams hold different values, and (a_i + t) - a_i is
            # not t to the last bit. The add itself is exact and broadcast
            # (`_apply_injection`); 1e-5 is four orders above the fp32 error here.
            assert torch.allclose(d[:, :, 0], d[:, :, -1], rtol=0.0, atol=1e-5), (
                "the draw must be ONE per cell, broadcast over the HC streams")
        s = layout.slot_valid.shape[1]
        dc = _cells(d, s)                                    # [B, S, K, C]
        pads += int((~layout.slot_valid).sum()) * K
        assert float(dc[~layout.slot_valid].abs().max()) == 0.0, "a pad slot drew noise"
        live.append(dc[layout.slot_valid])                   # [N, K, C]
    return torch.cat(live), pads


# ── 1. off is nothing ────────────────────────────────────────────────────────

def _loss_logits_grad_rng(m: MORPHTransformer, seed: int = 7):
    _ids, inp, lab, layout = _batch(K)
    torch.manual_seed(seed)
    res = m.train()(inp, labels=lab, slot_layout=layout)
    rng = torch.random.get_rng_state().clone()
    res["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    return float(res["loss"].detach()), _finite_logit_sum(lg), g, rng


def test_default_is_zero_and_the_absent_key_is_bit_identical():
    a = _fan()
    assert a.cfg.tul.fan_seed_noise == 0.0
    assert a._fan_seed_noise == 0.0
    b = _fan(fan_seed_noise=0.0)
    la, lga, ga, ra = _loss_logits_grad_rng(a)
    lb, lgb, gb, rb = _loss_logits_grad_rng(b)
    assert (la, lga, ga) == (lb, lgb, gb)
    assert torch.equal(ra, rb), "a knob that draws nothing must consume no RNG"


def test_a_live_knob_consumes_the_global_stream_and_a_private_generator_does_not():
    """The draw is real, so it MOVES the global stream — stated as a test because the
    consequence (every later draw in the run is shifted relative to the off arm) is a
    thing the reader of a one-factor comparison has to know. A probe that needs the draw
    isolated attaches `_fan_noise_gen`, and then the global stream is untouched."""
    _l0, _g0, _s0, rng_off = _loss_logits_grad_rng(_fan())
    _l1, _g1, _s1, rng_on = _loss_logits_grad_rng(_fan(fan_seed_noise=1.0))
    assert not torch.equal(rng_off, rng_on)
    m = _fan(fan_seed_noise=1.0)
    m._fan_noise_gen = torch.Generator().manual_seed(0)
    _l2, _g2, _s2, rng_priv = _loss_logits_grad_rng(m)
    assert torch.equal(rng_off, rng_priv), (
        "with a private generator attached the global stream must not move")
    assert _l2 != _l0, "the private generator must still deliver a real draw"


# ── 2. the scale is the documented one ───────────────────────────────────────

def test_the_added_term_has_unit_per_channel_rms_at_one():
    z, pads = _noise_sample(1.0)
    n = z.shape[0]
    assert n >= 20 and pads > 0, f"fixture too small: {n} valid slots, {pads} pad cells"
    assert z.numel() >= 2000, f"only {z.numel()} samples: the tolerance below is luck"
    rms = float(z.double().pow(2).mean().sqrt())
    assert abs(rms - 1.0) < 0.1, f"per-channel RMS {rms:.4f}, want 1.0 +- 0.1"
    # per-channel, not only pooled: a term that was unit RMS overall but concentrated on a
    # few channels would pass the line above and fail this one.
    per_c = z.double().pow(2).mean(dim=(0, 1)).sqrt()
    assert abs(float(per_c.mean()) - 1.0) < 0.1
    assert 0.5 < float(per_c.min()) and float(per_c.max()) < 1.6


def test_two_streams_of_one_slot_differ_by_root_two_and_the_deviation_by_root_three_quarters():
    """The three readings of the SAME draw, so nobody has to guess which one
    'the streams differ by 1.0' meant: the TERM is unit RMS, a PAIR differs by sqrt(2),
    and a stream's deviation from its slot's stream-mean is sqrt((K-1)/K)."""
    z, _pads = _noise_sample(1.0)
    pair = float((z[:, 0] - z[:, 1]).double().pow(2).mean().sqrt())
    assert abs(pair - math.sqrt(2.0)) < 0.1, f"pairwise RMS {pair:.4f}, want sqrt(2)"
    dev = z - z.mean(dim=1, keepdim=True)
    d = float(dev.double().pow(2).mean().sqrt())
    want = math.sqrt((K - 1) / K)
    assert abs(d - want) < 0.1, f"deviation RMS {d:.4f}, want {want:.4f}"


def test_the_knob_is_the_standard_deviation_and_scales_linearly():
    z = _noise_sample(0.5)[0]
    rms = float(z.double().pow(2).mean().sqrt())
    assert abs(rms - 0.5) < 0.05, f"per-channel RMS {rms:.4f} at std 0.5"


# ── 4. every forward draws ───────────────────────────────────────────────────

def test_two_eval_forwards_of_one_model_differ():
    m = _fan(fan_seed_noise=1.0)
    _i, inp, lab, layout = _batch(K)
    # NO reseed between them: reseeding would hand the same stream to both draws and
    # the test would pass on a model that drew once and cached it.
    torch.manual_seed(7)
    a, _e = _seed_state(m, inp, layout, seed=None)
    b, _e2 = _seed_state(m, inp, layout, seed=None)
    assert not torch.equal(a, b), (
        "eval reused one draw: the K streams are SAMPLES, so eval must draw too")
    m.eval()
    with torch.no_grad():
        l1 = _finite_logit_sum(m(inp, labels=None, slot_layout=layout)["logits"])
        l2 = _finite_logit_sum(m(inp, labels=None, slot_layout=layout)["logits"])
    assert l1 != l2


def test_the_training_forward_draws_too():
    off = _loss_logits_grad_rng(_fan())[0]
    on = _loss_logits_grad_rng(_fan(fan_seed_noise=1.0))[0]
    assert on != off


# ── 6. refusals ──────────────────────────────────────────────────────────────

def test_refused_off_a_fan_arm_and_at_a_negative_std():
    with pytest.raises(ValueError, match="tul.fan_seed_noise"):
        _tul(fan_seed_noise=1.0)
    with pytest.raises(ValueError, match="tul.fan_seed_noise"):
        _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_seed_noise=-0.1)
    _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_seed_noise=1.0)


def test_refused_with_the_noise_entry():
    """`model.core_state_init: "noise"` ALREADY replaces h_0 with a fresh per-cell draw,
    so the two together are an entry noise at an unnamed scale."""
    tul = _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_seed_noise=1.0)
    with pytest.raises(ValueError, match="core_state_init"):
        MORPHTransformer(_tiny(tul=tul, core_state_init="noise", injection_channels="all"))


def test_scse_is_already_refused_upstream_by_the_register():
    """There is NO fan_seed_noise check for SCSE, and this test is why: a fan forces
    `slot_cells == fan_k >= 2`, and the register refuses SCSE at construction, BEFORE the
    knob is read. A second check would be unreachable code that looks like a guard."""
    tul = _tul(fan_k=K, slot_cells=K, prefix_k=K, fan_mix="all", fan_seed_noise=1.0)
    with pytest.raises(NotImplementedError, match="SCSE"):
        MORPHTransformer(_tiny(tul=tul, scse_enabled=True))


# ── 7. the key composes, and the term it replaces keeps reporting ────────────

def test_the_key_composes_through_tul_setup_and_the_config_resolves():
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import KNOWN_TUL_KEYS

    assert "fan_seed_noise" in KNOWN_TUL_KEYS
    cdir = os.path.abspath("morph/configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name="tul_slot_spandec_strict_fan4_all_noise")
    assert float(cfg.tul.fan_seed_noise) == 1.0
    assert float(cfg.tul.fan_repel_lambda) == 0.0, "the noise REPLACES the volume term"
    assert cfg.tul.fan_repel_mode == "epivol", (
        "the mode is kept so the instruments stay the fan4-all ones")
    assert cfg.tul.fan_mix == "all" and int(cfg.tul.fan_k) == 4
    assert cfg.wandb.name == "slot-spandec-strict-fan4-all-noise"


def test_the_diversity_instruments_still_report_with_the_term_off():
    """`fan_repel_lambda: 0` must turn off the CHARGE and nothing else: the volume, the
    epiplexity and the cosine are still computed every pass and reported, or this arm
    cannot be read against fan4-all on the same axes."""
    m = _fan(fan_seed_noise=1.0, fan_repel_lambda=0.0, fan_repel_mode="epivol")
    _i, inp, lab, layout = _batch(K)
    torch.manual_seed(7)
    res = m.train()(inp, labels=lab, slot_layout=layout)
    have = {k for k in res if k.startswith("fan_")}
    for pref in ("fan_vol_t", "fan_epi_t", "fan_stream_cos_t"):
        assert any(k.startswith(pref) for k in have), f"{pref}* is not reported: {sorted(have)}"
    assert "fan_repel" not in res and "fan_repel_weighted" not in res, (
        "the term must not be charged at lambda 0")
    assert torch.isfinite(res["loss"]).all()
