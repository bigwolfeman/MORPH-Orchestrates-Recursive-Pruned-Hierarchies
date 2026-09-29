"""LX efficient-exploration knobs (2026-09-29) — MODEL-LEVEL wiring tests. Pure-function
math is ``tests/test_tul_explore.py``; this file proves: every knob composes through
Hydra and reaches ``TULConfig`` (``build_tul_runtime``), every knob is bit-identical to
the tree before it AT ITS OWN DEFAULT, each knob's defining property holds on a real
forward+backward (dropout 0 AND dropout > 0 — project memory: "a real crash hid behind
dropout-0 fixtures"), and every refusal TULConfig documents actually fires.

Scaffolding for LATER experiments: nothing here is queued or trained. Knob 2 (late
fusion) was investigated and NOT implemented; ``tul.coda_fuse_layer`` accordingly has
exactly one legal value (0) and this file pins that refusal.

CPU, fp32, the ``tests/test_tul_fan.py`` / ``tests/test_tul_lxfan.py`` fixtures.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHTransformer
from test_tul_fan import _batch, _tul
from test_tul_lxfan import K, _build, _lx_kw


def _lx(**kw):
    """fp01-free plain LX: K=4 codes, plast_weight=1 (code_enum_k's own requirement)."""
    return _lx_kw(**kw)


# ── off is the tree (bit-identical at every knob's own default) ────────────────────


def test_every_knob_defaults_to_off_bit_identical():
    a = _build(model_kw={"dropout": 0.1}, **_lx()).eval()
    b = _build(model_kw={"dropout": 0.1}, **_lx(
        coda_fuse_layer=0, hyp_score_head=False,
        latent_set_loss="none", latent_set_weight=0.0, enum_decode_k=0,
        hyp_merge="none", hyp_merge_weight=0.0)).eval()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    assert set(a.state_dict()) == set(b.state_dict())
    _ids, inp, lab, layout = _batch(2)
    torch.manual_seed(11)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(11)
    ob = b(inp, labels=lab, slot_layout=layout)
    for k, v in oa.items():
        if torch.is_tensor(v):
            assert torch.equal(v, ob[k]), k


def test_no_new_modules_built_when_every_knob_is_off():
    m = _build(**_lx())
    assert m.tul_hyp_score_head is None
    assert m.tul_hyp_merge_gate is None


# ── KNOB 2: refused, not built ───────────────────────────────────────────────────────


def test_coda_fuse_layer_is_refused_at_any_nonzero_value():
    with pytest.raises(NotImplementedError, match="coda_fuse_layer"):
        _tul(coda_fuse_layer=1, code_enum_k=K, plast_weight=1.0)
    with pytest.raises(NotImplementedError, match="coda_fuse_layer"):
        _tul(coda_fuse_layer=-1, code_enum_k=K, plast_weight=1.0)


# ── KNOB 3: score head ──────────────────────────────────────────────────────────────


def test_hyp_score_head_builds_the_module_and_reaches_config():
    from test_tul_fan import _tiny
    tc = _tul(hyp_score_head=True, code_enum_k=K, plast_weight=1.0)
    assert tc.hyp_score_head is True
    m = MORPHTransformer(_tiny(tul=tc, d_ff=96))
    assert m.tul_hyp_score_head is not None


def test_hyp_score_head_trains_and_reports_kl(dropout=0.0):
    m = _build(model_kw={"dropout": dropout}, **_lx(hyp_score_head=True)).train()

    # Spy on the real call site: capture the exact tensors the score head is built from,
    # as they reach it during a real forward. `tul_code_enum.basis` legitimately carries
    # SOME nonzero gradient regardless of this knob (the ordinary token-mixture CE trains
    # the codes on every LX arm) -- comparing two separately-run models' `basis.grad` is
    # not a valid test of leakage, since Poisson core depth / code sampling / dropout draw
    # different random numbers between two independent forwards and would make the
    # gradients differ for reasons that have nothing to do with the score head. The
    # correct, direct proof is structural: the score head must receive tensors with no
    # `grad_fn` (true leaves w.r.t. autograd), so there is no graph edge for
    # ``score_loss.backward()`` to walk back through, no matter what else was sampled.
    captured = {}
    orig = m._tul_explore_pre_coda

    def _spy(h_slots, x, layout, n_rollouts, stats):
        ret = orig(h_slots, x, layout, n_rollouts, stats)
        captured["hyp"], captured["target"] = ret[1], ret[2]
        return ret

    m._tul_explore_pre_coda = _spy

    _ids, inp, lab, layout = _batch(2)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    assert "hyp_score_kl" in out and float(out["hyp_score_kl"]) >= 0.0

    assert captured["hyp"] is not None and captured["target"] is not None
    # A detached tensor's grad_fn is None: this is the structural guarantee, independent
    # of which random branch the forward took.
    assert captured["hyp"].grad_fn is None, "hyp reaching the score head is not detached"
    assert captured["target"].grad_fn is None, "target reaching the score head is not detached"
    assert captured["hyp"].requires_grad is False
    assert captured["target"].requires_grad is False

    out["loss"].backward()
    assert m.tul_hyp_score_head.w.weight.grad is not None
    assert torch.isfinite(m.tul_hyp_score_head.w.weight.grad).all()
    assert torch.isfinite(m.tul_code_enum.basis.grad).all()


def test_hyp_score_head_trains_at_dropout():
    test_hyp_score_head_trains_and_reports_kl(dropout=0.1)


def test_hyp_score_read_is_not_a_config_key():
    """A top1/top2 READ was once accepted as config and read by nothing. It is gone;
    the head's read is priced by the `hyp_read_*` instrument instead."""
    with pytest.raises(TypeError, match="hyp_score_read"):
        _tul(hyp_score_read="top1", hyp_score_head=True, code_enum_k=K, plast_weight=1.0)


def test_hyp_read_gaps_agree_with_the_independent_width_gain():
    """`hyp_read_rand1_gap` (built from the slot-indexed S re-indexing, the `scored` mask
    and `n_w`) must equal `enum_width_gain` (mean_k ce_code - ce_mix, built from the
    per-token log-probs by separate code). The tokens before the first slot read no cell,
    so they are rollout-invariant and add nothing to either side. A wrong group offset, a
    wrong mask or a per-span-vs-per-token slip breaks the equality."""
    m = _build(model_kw={"dropout": 0.0}, **_lx(hyp_score_head=True)).eval()
    _ids, inp, lab, layout = _batch(2)
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    for key in ("hyp_read_top1_gap", "hyp_read_top2_gap", "hyp_read_rand1_gap",
                "hyp_read_best1_gap", "hyp_score_agree", "enum_width_gain"):
        assert key in out, key
    assert float(out["enum_width_gain"]) > 1e-4, "fixture rollouts do not differ"
    torch.testing.assert_close(out["hyp_read_rand1_gap"].float(),
                               out["enum_width_gain"].float(), rtol=1e-4, atol=1e-6)
    assert float(out["hyp_read_best1_gap"]) <= 0.0 <= float(out["hyp_read_rand1_gap"])


def test_hyp_score_head_refused_under_fan_k():
    with pytest.raises(NotImplementedError, match="hyp_score_head"):
        _tul(hyp_score_head=True, fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all",
            code_enum_k=K, plast_weight=1.0)


# ── KNOB 4: set loss ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["energy", "infonce"])
def test_latent_set_loss_trains_and_moves_the_loop(mode, dropout=0.0):
    m = _build(model_kw={"dropout": dropout},
              **_lx(latent_set_loss=mode, latent_set_weight=0.1)).train()
    _ids, inp, lab, layout = _batch(2)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    assert "explore_set" in out and "explore_set_weighted" in out and "explore_set_spread" in out
    out["loss"].backward()
    # UNLIKE the score head, the set loss's hypothesis side is LIVE: it must reach the
    # loop's own codes.
    assert m.tul_code_enum.basis.grad is not None
    assert float(m.tul_code_enum.basis.grad.abs().sum()) > 0
    assert torch.isfinite(m.tul_code_enum.basis.grad).all()


@pytest.mark.parametrize("mode", ["energy", "infonce"])
def test_latent_set_loss_trains_at_dropout(mode):
    test_latent_set_loss_trains_and_moves_the_loop(mode, dropout=0.1)


def test_latent_set_weight_zero_is_off_bit_identical():
    a = _build(model_kw={"dropout": 0.1}, **_lx()).eval()
    b = _build(model_kw={"dropout": 0.1},
              **_lx(latent_set_loss="none", latent_set_weight=0.0)).eval()
    _ids, inp, lab, layout = _batch(2)
    torch.manual_seed(3)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(3)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(oa["loss"], ob["loss"])


def test_latent_set_loss_and_weight_must_be_set_together():
    with pytest.raises(ValueError, match="latent_set_loss"):
        _tul(latent_set_weight=0.1, code_enum_k=K, plast_weight=1.0)
    with pytest.raises(ValueError, match="latent_set_loss"):
        _tul(latent_set_loss="energy", code_enum_k=K, plast_weight=1.0)


def test_latent_set_loss_refused_under_fan_k():
    with pytest.raises(NotImplementedError, match="latent_set_loss"):
        _tul(latent_set_loss="energy", latent_set_weight=0.1, fan_k=4, slot_cells=4,
            prefix_k=4, fan_mix="all", code_enum_k=K, plast_weight=1.0)


# ── KNOB 5: sampled decodes ──────────────────────────────────────────────────────────


def test_enum_decode_k_zero_is_off_bit_identical():
    a = _build(model_kw={"dropout": 0.1}, **_lx()).eval()
    b = _build(model_kw={"dropout": 0.1}, **_lx(enum_decode_k=0)).eval()
    _ids, inp, lab, layout = _batch(2)
    torch.manual_seed(5)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(oa["loss"], ob["loss"])


def test_enum_decode_k_uniform_runs_only_k_rollouts_through_the_coda():
    """A spy on `_back_region` must see k*B0 rows on the coda call, not K*B0 — the
    actual compute-saving claim, not just a finite-loss smoke."""
    from test_tul_lxfan import _Spy
    m = _build(model_kw={"dropout": 0.0}, **_lx(enum_decode_k=2)).train()
    _ids, inp, lab, layout = _batch(2)
    B0 = inp.shape[0]
    spy = _Spy(m, "_back_region")
    out = m(inp, labels=lab, slot_layout=layout)
    rows = [int(a[0].shape[0]) for a, _kw in spy.calls]
    assert rows == [2 * B0], rows
    assert torch.isfinite(out["loss"])


def test_enum_decode_k_survives_real_dropout_on_a_non_multiple_batch():
    """Regression: k*B0 not a multiple of K used to raise inside RolloutSharedDropout
    (`rows % n_rep`) — found and fixed while building this knob, the exact class of gap
    project memory documents ("a real crash hid behind dropout-0 fixtures")."""
    m = _build(model_kw={"dropout": 0.3}, **_lx(enum_decode_k=3)).train()   # K=4, B0=2: 6%4!=0
    _ids, inp, lab, layout = _batch(2)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    n_bad = sum(1 for p in m.parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all())
    assert n_bad == 0


def test_enum_decode_k_with_score_head_is_refused():
    """A head-weighted draw has no importance correction in this tree."""
    with pytest.raises(NotImplementedError, match="importance correction"):
        _tul(enum_decode_k=2, hyp_score_head=True, code_enum_k=K, plast_weight=1.0)


def test_enum_decode_k_out_of_range_or_without_rollouts_raises():
    with pytest.raises(ValueError, match="enum_decode_k"):
        _tul(enum_decode_k=5, code_enum_k=4, plast_weight=1.0)
    with pytest.raises(ValueError, match="enum_decode_k"):
        _tul(enum_decode_k=1)                             # code_enum_k defaults to 1


def test_enum_decode_k_refused_under_fan_k():
    with pytest.raises(NotImplementedError, match="enum_decode_k"):
        _tul(enum_decode_k=2, fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all",
            code_enum_k=K, plast_weight=1.0)


def test_eval_never_uses_enum_decode_k():
    """Eval / generation always reads every rollout — `tc.enum_decode_k` only acts
    under `self.training`. A spy confirms the label-free forward's coda call sees the
    FULL K*B0 rows."""
    from test_tul_lxfan import _Spy
    m = _build(model_kw={"dropout": 0.0}, **_lx(enum_decode_k=2)).eval()
    _ids, inp, lab, layout = _batch(2)
    B0 = inp.shape[0]
    spy = _Spy(m, "_back_region")
    with torch.no_grad():
        m(inp, labels=None, slot_layout=layout)
    rows = [int(a[0].shape[0]) for a, _kw in spy.calls]
    assert rows == [K * B0], rows


# ── KNOB 6: merge / superposition probe ─────────────────────────────────────────────


def test_hyp_merge_probe_is_read_only_and_moves_no_gradient():
    m = _build(model_kw={"dropout": 0.0}, **_lx(hyp_merge="probe")).train()
    _ids, inp, lab, layout = _batch(2)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    assert "hyp_merge_dist" in out and "hyp_merge_beats_best_single" in out
    assert "explore_extra_weighted" not in out       # nothing folded: probe is read-only
    out["loss"].backward()
    assert m.tul_hyp_merge_gate is None


def test_hyp_merge_learned_builds_the_gate_and_trains_only_at_nonzero_weight():
    m0 = _build(model_kw={"dropout": 0.0}, **_lx(hyp_merge="learned")).train()
    assert m0.tul_hyp_merge_gate is not None
    _ids, inp, lab, layout = _batch(2)
    out0 = m0(inp, labels=lab, slot_layout=layout)
    assert "explore_extra_weighted" not in out0     # weight 0.0: built, not trained
    out0["loss"].backward()
    assert m0.tul_hyp_merge_gate.q.weight.grad is None

    m1 = _build(model_kw={"dropout": 0.0},
               **_lx(hyp_merge="learned", hyp_merge_weight=0.1)).train()
    out1 = m1(inp, labels=lab, slot_layout=layout)
    assert "explore_extra_weighted" in out1
    out1["loss"].backward()
    assert m1.tul_hyp_merge_gate.q.weight.grad is not None
    assert torch.isfinite(m1.tul_hyp_merge_gate.q.weight.grad).all()


def test_hyp_merge_learned_trains_at_dropout():
    m = _build(model_kw={"dropout": 0.1},
              **_lx(hyp_merge="learned", hyp_merge_weight=0.1)).train()
    _ids, inp, lab, layout = _batch(2)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    n_bad = sum(1 for p in m.parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all())
    assert n_bad == 0


def test_hyp_merge_weight_needs_learned_mode():
    with pytest.raises(ValueError, match="hyp_merge_weight"):
        _tul(hyp_merge_weight=0.1, code_enum_k=K, plast_weight=1.0)   # merge='none'
    with pytest.raises(ValueError, match="hyp_merge_weight"):
        _tul(hyp_merge="probe", hyp_merge_weight=0.1, code_enum_k=K, plast_weight=1.0)


def test_hyp_merge_bad_value_and_fan_k_refusal():
    with pytest.raises(ValueError, match="hyp_merge"):
        _tul(hyp_merge="bogus", code_enum_k=K, plast_weight=1.0)
    with pytest.raises(NotImplementedError, match="hyp_merge"):
        _tul(hyp_merge="probe", fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all",
            code_enum_k=K, plast_weight=1.0)


# ── combined ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("combo", [
    dict(hyp_score_head=True),       # the head and the decode subsample are refused
    dict(enum_decode_k=2),           # together, so each legal side is trained here
])
def test_every_legal_knob_combination_trains_a_real_step(combo):
    m = _build(model_kw={"dropout": 0.1}, **_lx(
        latent_set_loss="infonce", latent_set_weight=0.05,
        hyp_merge="learned", hyp_merge_weight=0.05, **combo)).train()
    _ids, inp, lab, layout = _batch(2)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    n_bad = sum(1 for p in m.parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all())
    assert n_bad == 0
    m.eval()
    with torch.no_grad():
        ev = m(inp, labels=None, slot_layout=layout)
        assert not torch.isnan(ev["logits"]).any()


# ── config compose / reach (build_tul_runtime) ──────────────────────────────────────


# NOTE ON THE PARENT CONFIG. The task brief's own naming template
# (`tul_slot_spandec_strict_lxfan4_wta_fp01_<knob>.yaml`) composes on top of arm (b), the
# WTA/fan arm — but knobs 3/4/5/6 REFUSE combination with `tul.fan_k` (see each knob's
# own TULConfig refusal, added to keep this whole package clear of `_tul_fan_all`'s
# WTA winner-pick machinery, which three other agents were editing the same day this was
# built). Composing on `tul_slot_spandec_strict_lxfan4_wta_fp01` would therefore raise at
# construction for every one of these configs. Built on
# `tul_slot_spandec_strict_e4probe_fp01` instead (the plain LX + fp01 constraint arm,
# `code_enum_k=4`, no fan) — a deliberate, documented deviation from the literal filename
# template, not an oversight.
@pytest.mark.parametrize("name,expect_key", [
    ("tul_slot_spandec_strict_e4probe_fp01_hypscore", "tul.hyp_score_head"),
    ("tul_slot_spandec_strict_e4probe_fp01_latentset", "tul.latent_set_loss"),
    ("tul_slot_spandec_strict_e4probe_fp01_decodek", "tul.enum_decode_k"),
    ("tul_slot_spandec_strict_e4probe_fp01_hypmerge", "tul.hyp_merge"),
])
def test_knob_configs_compose_and_reach_the_runtime(name, expect_key, monkeypatch):
    from test_tul_lxfan import _diff
    cfg, rt, diff = _diff(name, "tul_slot_spandec_strict_e4probe_fp01", monkeypatch)
    assert expect_key in diff, (name, sorted(diff))
    assert int(cfg.training.steps) == 3000
    tc = rt.model_cfg
    assert tc is not None and tc.code_enum_k == 4
