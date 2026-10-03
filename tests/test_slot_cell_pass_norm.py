"""``tul.slot_cell_pass_norm`` (2026-10-03): after every slot-loop pass each carried cell
is ``RMSNorm(h) * g``, ONE learned per-channel gain shared by every pass, so the loop can
rotate its cells but not grow them.

Files: morph/model/tul.py (the key, `_check_slot_cell_pass_norm`), morph/model/transformer.py
(`tul_cell_norm`, the model-level refusals, `_slot_cell_norm` (the ONE site in
`_tul_core`), `_cell_rms_reading`, `_fold_cell_norm_stats`), morph/training/tul_setup.py,
morph/training/train.py (`_CELL_NORM_KEYS`, the val scan), the two `_cnorm` configs.
Note: .agents/notes/proposed/architecture/2026-10-03-slot-cell-pass-norm.md

What each test pins:
  * OFF is the tree: loss, logit sum, eval loss, total |grad| and the gradient count of three
    slot-loop fixtures (the two parents' latent-selected loops and the strict single-cell
    ruler with fp01's hinge and fixed-point term), at dropout 0.0 and 0.1, against pins
    measured on the UNMODIFIED tree (master 48f89c8, before the key existed), with the key
    implicit and explicit.
  * ON builds ONE RMSNorm (eps 1e-6, weight ones), no RNG drawn, one new state-dict key.
  * The entry is NOT normed; pass 0's carried cell is exactly RMSNorm of the off model's
    pass-0 output (same weights), so the injection is inside the pass and the norm at its end.
  * EVERY pass's carried cell, divided by g, has per-stream RMS 1 (g set to a random
    positive vector), so every pass is normed by the SAME g; at init the RMS is rms(g) = 1.
  * g receives a finite, nonzero gradient; it is in the no-decay group and never ternary.
  * The fixed-point term equals a hand recomputation from the NORMED states.
  * The eval readings (`slot_cell_rms_t{t}` ~ 1 at init, the entry's is not, the gain's
    mean / std) exist at eval only and reach train.py's `evaluate` under `val/`.
  * Refusals, tul-level and model-level.
  * The two configs differ from their parents by this key and the run name only, and the
    key reaches MORPHConfig and the manifest.

CPU, fp32, the tests/test_tul_fan.py / test_tul_lxfan.py / test_tul_fan_lsel.py fixtures.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from morph.model.attention import RMSNorm
from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from test_tul_fan import _batch, _tiny, _tul
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M, _fan_pin_run
from test_tul_lxfan import _build

# Measured 2026-10-03 on the UNMODIFIED tree (worktree at master 48f89c8, before
# `tul.slot_cell_pass_norm` existed) by `_fan_pin_run` on these fixtures, one CPU thread:
# (train loss, eval logit sum, eval loss, total |grad|, number of grads).
PINS = {
    ("lsel_det_rf_lam1", 0.0): (12.307818412780762, 1127.7436597240157, 12.344327926635742,
                                1918.7259733589428, 203),
    ("lsel_det_rf_lam1", 0.1): (12.251749038696289, 1127.7436597240157, 12.344327926635742,
                                1924.0707942435026, 203),
    ("lsel_joint_rf_lam1_rank", 0.0): (12.307818412780762, 1127.7436597240157,
                                       12.344327926635742, 2003.3168870525822, 203),
    ("lsel_joint_rf_lam1_rank", 0.1): (12.251749038696289, 1127.7436597240157,
                                       12.344327926635742, 2026.3529562972428, 203),
    ("strict_ruler_fp01", 0.0): (5.114372253417969, 1424.483801516961, 5.110508441925049,
                                 708.2954810252289, 174),
    ("strict_ruler_fp01", 0.1): (5.099832057952881, 1424.483801516961, 5.110508441925049,
                                 708.4679551873319, 174),
}

_ARMS = {
    "lsel_det_rf_lam1": ("detached", dict(fan_lsel_train_follow="router",
                                          fan_lsel_lambda=1.0)),
    "lsel_joint_rf_lam1_rank": ("joint", dict(fan_lsel_train_follow="router",
                                              fan_lsel_lambda=1.0,
                                              fan_lsel_head_input="detached")),
    "strict_ruler_fp01": (None, {}),
}

# Per-stream RMS of RMSNorm(x) is sqrt(ms / (ms + eps)) with eps 1e-6; fp32 rounding adds
# ~1e-6. 1e-4 is two orders above both and far below any un-normed cell (the entry's
# per-stream RMS sits 0.05+ away from 1 on these fixtures, checked below).
_RMS_TOL = 1e-4


def _arm(name: str, dropout: float = 0.0, model_kw: dict | None = None, **kw):
    mode, akw = _ARMS[name]
    mk = {"dropout": dropout, **(model_kw or {})}
    if mode is None:
        return _build(model_kw=mk, **akw, **kw)
    return _lsel(mode, model_kw=mk, **akw, **kw)


def _stream_rms(h: torch.Tensor) -> torch.Tensor:
    """Per (row, cell, stream) RMS over the channel axis, fp64."""
    return h.double().pow(2).mean(-1).sqrt()


def _capture_eval(m: MORPHTransformer, inp, lab, layout):
    """One eval forward with the Jacobian capture on: the carrier ENTERING every pass
    (`_jac_capture`, detached, a Python-level hook that leaves the forward unchanged) and,
    on a latent-selected model, every pass's candidates before the reset (`_lsel_capture`
    "pre", which includes the LAST pass's exit candidates)."""
    m._jac_capture = []
    lsel = m._lsel_mode != "off"
    if lsel:
        m._lsel_capture = []
    try:
        with torch.no_grad():
            out = m(inp, labels=lab, slot_layout=layout)
        jac = list(m._jac_capture)
        pre = list(m._lsel_capture) if lsel else None
    finally:
        m._jac_capture = None
        if lsel:
            m._lsel_capture = None
    return out, jac, pre


# ── 1. OFF is the tree ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,dropout", sorted(PINS))
@pytest.mark.parametrize("explicit", [False, True])
def test_off_is_bit_identical_to_the_pre_key_tree(name, dropout, explicit):
    kw = {"slot_cell_pass_norm": "off"} if explicit else {}
    m = _arm(name, dropout, **kw)
    assert m.tul_cell_norm is None
    assert not any(k.startswith("tul_cell_norm") for k in m.state_dict())
    assert _fan_pin_run(m) == PINS[(name, dropout)]


# ── 2. the build ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_rms_builds_one_shared_gain_and_draws_no_rng(name):
    off = _arm(name)
    on = _arm(name, slot_cell_pass_norm="rms")
    assert isinstance(on.tul_cell_norm, RMSNorm)
    assert on.tul_cell_norm.eps == 1e-6
    assert torch.equal(on.tul_cell_norm.weight, torch.ones(on.cfg.d_model))
    assert on.tul_cell_norm.weight.dtype == torch.float32
    sd_off, sd_on = off.state_dict(), on.state_dict()
    assert set(sd_on) - set(sd_off) == {"tul_cell_norm.weight"}
    assert set(sd_off) <= set(sd_on)
    for k, v in sd_off.items():
        assert torch.equal(v, sd_on[k]), k
    # ONE gain: the only parameter the key adds.
    added = [n for n, _ in on.named_parameters() if n not in dict(off.named_parameters())]
    assert added == ["tul_cell_norm.weight"]


# ── 3. placement: the entry is not normed, the norm is the end of the pass ──────────


def test_entry_is_untouched_and_pass0_is_the_norm_of_the_off_step():
    """Strict single-cell ruler, eval, depth 3 (no reset, no fan): the off and rms models
    share every weight, so they enter the loop at the SAME state and pass 0 computes the
    SAME raw output; the rms model then carries exactly RMSNorm(that output) * g.

    The entry is `core_init(input_norm(prelude))`, and `input_norm` is itself an RMSNorm
    with weight 1 at init, so at init the entry ALREADY has per-stream RMS 1 and an RMS
    check cannot see a norm on it. Both models get the same non-constant `input_norm`
    weight first, which moves the entry off unit RMS (the weights stay shared)."""
    _ids0, inp, lab, layout = _batch(M)
    off = _arm("strict_ruler_fp01", slot_mean_depth=3).eval()
    on = _arm("strict_ruler_fp01", slot_mean_depth=3, slot_cell_pass_norm="rms").eval()
    gen = torch.Generator().manual_seed(9)
    w_in = 0.3 + 2.0 * torch.rand(off.cfg.d_model, generator=gen)
    with torch.no_grad():
        off.input_norm.weight.copy_(w_in)
        on.input_norm.weight.copy_(w_in)
    _o, jac_off, _ = _capture_eval(off, inp, lab, layout)
    _o, jac_on, _ = _capture_eval(on, inp, lab, layout)
    assert len(jac_off) == len(jac_on) == 3
    valid = jac_on[0]["active"]
    # the entry: identical, and NOT at unit RMS
    assert torch.equal(jac_on[0]["h"], jac_off[0]["h"])
    ent = _stream_rms(jac_on[0]["h"])[valid]
    assert float((ent - 1.0).abs().max()) > 0.05
    # pass 0's carried cell = RMSNorm(the off model's pass-0 output)
    want = on.tul_cell_norm(jac_off[1]["h"])
    torch.testing.assert_close(jac_on[1]["h"][valid], want[valid], rtol=1e-5, atol=1e-6)
    # and the off model's pass-0 output is NOT normed (the test can tell the two apart)
    assert float((_stream_rms(jac_off[1]["h"])[valid] - 1.0).abs().max()) > 0.05


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_every_pass_is_normed_at_init_to_rms_g(name):
    """At init g = 1, so every carried cell after every pass has per-stream RMS 1 = rms(g):
    the states entering passes 1 and 2 and, on the latent-selected loop, every pass's
    candidates including the exit."""
    _ids0, inp, lab, layout = _batch(M)
    m = _arm(name, slot_mean_depth=3, slot_cell_pass_norm="rms").eval()
    g_rms = float(m.tul_cell_norm.weight.detach().double().pow(2).mean().sqrt())
    assert g_rms == 1.0
    _out, jac, pre = _capture_eval(m, inp, lab, layout)
    assert len(jac) == 3
    states = [(f"enter{t}", c["h"], c["active"]) for t, c in enumerate(jac) if t >= 1]
    if pre is not None:
        states += [(f"pre{c['t']}", c["pre"], jac[0]["active"]) for c in pre]
        assert len(pre) == 3
    for tag, h, act in states:
        r = _stream_rms(h)[act]
        assert r.numel() > 0, tag
        assert float((r - g_rms).abs().max()) < _RMS_TOL, (tag, float((r - g_rms).abs().max()))


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_every_pass_is_normed_by_the_same_gain(name):
    """g set to a random positive vector: every pass's carried cell divided by g has
    per-stream RMS 1. A gain per pass, or a pass that is not normed, fails this."""
    _ids0, inp, lab, layout = _batch(M)
    m = _arm(name, slot_mean_depth=3, slot_cell_pass_norm="rms").eval()
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        m.tul_cell_norm.weight.copy_(0.3 + 2.0 * torch.rand(m.cfg.d_model, generator=gen))
    g = m.tul_cell_norm.weight.detach().double()
    _out, jac, pre = _capture_eval(m, inp, lab, layout)
    states = [(f"enter{t}", c["h"], c["active"]) for t, c in enumerate(jac) if t >= 1]
    if pre is not None:
        states += [(f"pre{c['t']}", c["pre"], jac[0]["active"]) for c in pre]
    for tag, h, act in states:
        r = _stream_rms(h.double() / g)[act]
        assert float((r - 1.0).abs().max()) < _RMS_TOL, (tag, float((r - 1.0).abs().max()))
    # and the carried cell's own RMS is NOT rms(g) for a non-constant g (so the test above
    # is not satisfied by a norm that ignores g)
    r_raw = _stream_rms(jac[1]["h"])[jac[1]["active"]]
    assert float((r_raw - float(g.pow(2).mean().sqrt())).abs().max()) > 1e-3


# ── 4. g trains ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_the_gain_gets_a_gradient_and_sits_in_the_no_decay_group(name):
    from morph.training.optimizer import _split_by_decay

    _ids0, inp, lab, layout = _batch(M)
    m = _arm(name, slot_cell_pass_norm="rms").train()
    torch.manual_seed(3)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    g = m.tul_cell_norm.weight.grad
    assert g is not None and bool(torch.isfinite(g).all())
    assert float(g.abs().sum()) > 0.0
    _d, _nd, _dn, nd_names = _split_by_decay(m)
    assert "tul_cell_norm.weight" in nd_names
    assert m.tul_cell_norm.weight.requires_grad


def test_the_gain_is_never_ternarized():
    from torch.nn.utils import parametrize

    from morph.model.ternary_qat import apply_ternary_qat

    m = _arm("lsel_det_rf_lam1", slot_cell_pass_norm="rms")
    apply_ternary_qat(m, scope="full", scale_mode="norm_match")
    assert not parametrize.is_parametrized(m.tul_cell_norm)
    assert any(parametrize.is_parametrized(mod) for mod in m.modules()), \
        "the QAT call ternarized nothing, so this test proves nothing"


# ── 5. the fixed-point term reads the NORMED states ─────────────────────────────────


def test_fixed_point_term_compares_the_normed_states(monkeypatch):
    """Train forward, strict ruler (fixed-point 0.1, full BPTT): the term equals
    mean over finishing slots of ||h_T - h_{T-1}||^2 / (||h_T||^2 + 1e-6) with h_T the
    normed exit and h_{T-1} the carried state entering the last pass (normed, or the
    un-normed entry at depth 1)."""
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("strict_ruler_fp01", slot_cell_pass_norm="rms").train()
    assert m.cfg.core_fixed_point_lambda > 0.0 and m.cfg.bptt_depth >= m.cfg.max_depth
    seen: dict = {}
    real = m._tul_core

    def spy(*a, **k):
        r = real(*a, **k)
        seen["h"], seen["depths"] = r[1].detach(), r[2].detach()
        return r
    monkeypatch.setattr(m, "_tul_core", spy)
    m._jac_capture = []
    torch.manual_seed(11)
    out = m(inp, labels=lab, slot_layout=layout)
    jac = list(m._jac_capture)
    m._jac_capture = None
    d, h = seen["depths"], seen["h"]
    valid = jac[0]["active"]
    terms = []
    for b in range(d.shape[0]):
        for s in range(d.shape[1]):
            if not bool(valid[b, s]):
                continue
            T = int(d[b, s])
            hT = h[b, s].double().flatten()
            hp = jac[T - 1]["h"][b, s].double().flatten()
            terms.append(((hT - hp).pow(2).sum() / (hT.pow(2).sum() + 1e-6)).item())
    want = sum(terms) / len(terms)
    assert len(terms) > 4 and len({int(x) for x in d[valid]}) > 1, "need mixed depths"
    assert abs(float(out["fixed_point"]) - want) < 1e-5 * max(1.0, want), \
        (float(out["fixed_point"]), want)
    # the exit is normed (so the recomputation above used normed states)
    assert float((_stream_rms(h)[valid] - 1.0).abs().max()) < _RMS_TOL


def test_the_gain_hinge_still_reads_the_raw_map():
    """The hinge was NOT changed: it re-runs `_core_step`, which does not contain the norm.
    At depth 1 the hinge probes pass 0, whose operating point (the entry) is identical in
    the off and rms models, so both read the SAME gain and penalty. A norm moved inside
    `_core_step` (so the hinge would difference RMSNorm(f)) fails this."""
    _ids0, inp, lab, layout = _batch(M)
    res = {}
    for v in ("off", "rms"):
        m = _arm("strict_ruler_fp01", slot_depth_fixed=1, slot_cell_pass_norm=v).train()
        assert m.cfg.slot_gain_lambda > 0.0
        torch.manual_seed(4)
        out = m(inp, labels=lab, slot_layout=layout)
        res[v] = (float(out["gain_est"]), float(out["gain_reg_weighted"]))
    assert res["off"] == res["rms"], res


def test_the_norm_takes_the_pass_grad_context(monkeypatch):
    """Truncated BPTT (bptt_depth 1, fixed depth 3): passes 0 and 1 run under no_grad, so
    their norm must too (no gradient into g from passes the objective cut away)."""
    _ids0, inp, lab, layout = _batch(M)
    m = _arm("strict_ruler_fp01", model_kw={"bptt_depth": 1}, slot_depth_fixed=3,
             slot_cell_pass_norm="rms").train()
    calls: list[bool] = []
    real = m.tul_cell_norm.forward

    def spy(x):
        calls.append(torch.is_grad_enabled())
        return real(x)
    monkeypatch.setattr(m.tul_cell_norm, "forward", spy)
    torch.manual_seed(2)
    m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    assert calls == [False, False, True], calls
    assert float(m.tul_cell_norm.weight.grad.abs().sum()) > 0.0


# ── 6. the eval readings ────────────────────────────────────────────────────────────


def test_eval_readings_exist_at_eval_only_and_reach_evaluate():
    from morph.training.train import evaluate

    _ids0, inp, lab, layout = _batch(M)
    m = _arm("lsel_det_rf_lam1", slot_mean_depth=3, slot_cell_pass_norm="rms").eval()
    # Move the entry off unit RMS (`input_norm` is an RMSNorm at weight 1, see the
    # placement test), so the entry reading and the per-pass readings can differ.
    with torch.no_grad():
        m.input_norm.weight.copy_(
            0.3 + 2.0 * torch.rand(m.cfg.d_model, generator=torch.Generator().manual_seed(9)))
        out = m(inp, labels=lab, slot_layout=layout)
    for t in range(3):
        # carrier RMS over (streams, channels) = rms(g) = 1 at init, per cell
        assert abs(float(out[f"slot_cell_rms_t{t}"]) - 1.0) < _RMS_TOL
        assert 0.0 < float(out[f"slot_cell_mean_rms_t{t}"]) <= 1.0 + _RMS_TOL
    assert abs(float(out["slot_cell_rms_entry"]) - 1.0) > 0.05
    assert float(out["slot_cell_norm_g_mean"]) == 1.0
    assert float(out["slot_cell_norm_g_std"]) == 0.0
    m.train()
    torch.manual_seed(1)
    out_tr = m(inp, labels=lab, slot_layout=layout)
    assert not any(str(k).startswith("slot_cell_rms") for k in out_tr)
    off = _arm("lsel_det_rf_lam1").eval()
    with torch.no_grad():
        out_off = off(inp, labels=lab, slot_layout=layout)
    assert not any(str(k).startswith(("slot_cell_rms", "slot_cell_norm")) for k in out_off)
    m.eval()
    extra: dict = {}
    evaluate(m, torch.device("cpu"), iter([(inp, lab, layout)]), n_batches=1, tul=True,
             extra=extra)
    assert abs(extra["val/slot_cell_rms_t0"] - 1.0) < _RMS_TOL
    assert "val/slot_cell_norm_g_std" in extra and "val/slot_cell_rms_entry" in extra


# ── 7. refusals ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,match", [
    (dict(tokens_through_core=True, tg_restrict=False, tg_geometry="none"),
     "tokens_through_core"),
    (dict(loop_reads_tokens=True), "loop_reads_tokens"),
    (dict(code=True), "tul.code"),
    (dict(core_stage_cond="iter"), "core_stage_cond"),
    (dict(gram=True), "tul.gram"),
    (dict(loop_carry="persist", loop_reach=1), "persist"),
    (dict(loop_denoise=True), "loop_denoise"),
    (dict(xhc_streams=8), "xhc_streams"),
])
def test_tul_level_refusals(kw, match):
    with pytest.raises(NotImplementedError, match=f"slot_cell_pass_norm='rms' with .*{match}"):
        _tul(slot_cell_pass_norm="rms", **kw)


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="slot_cell_pass_norm must be one of"):
        TULConfig(slot_cell_pass_norm="layer")


@pytest.mark.parametrize("model_kw,match", [
    (dict(n_core=0), "n_core=0"),
    (dict(scse_enabled=True), "scse_enabled"),
    (dict(slot_state_renorm=True), "slot_state_renorm"),
    (dict(core_gain_clip=2.0), "core_gain_clip"),
])
def test_model_level_refusals(model_kw, match):
    tc = _tul(slot_cell_pass_norm="rms")
    with pytest.raises(NotImplementedError, match=f"slot_cell_pass_norm='rms' with .*{match}"):
        MORPHTransformer(_tiny(tul=tc, **model_kw))
    MORPHTransformer(_tiny(tul=_tul(), **model_kw))   # the model alone builds


def test_fm_planner_refused():
    from morph.model.tul_fm import FMArmConfig

    tc = TULConfig(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                   mux_beta=0.0, slot_cell_pass_norm="rms")
    fm = FMArmConfig(d_p=16, n_layers=1, n_heads=2, d_ff=32, cond_dim=16, max_slots=10,
                     l_total=84)
    with pytest.raises(NotImplementedError, match="slot_cell_pass_norm='rms' with an FM"):
        MORPHTransformer(_tiny(n_core=0, tul=tc, fm=fm))


# ── 8. configs ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,parent", [
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1_cnorm",
     "tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1"),
    ("tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm",
     "tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank"),
])
def test_cnorm_configs_change_one_key_and_reach_the_model(name, parent, monkeypatch):
    from omegaconf import OmegaConf
    from test_slot_gain_tail import _leaves, _MISSING
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {"tul.slot_cell_pass_norm", "wandb.name"}, sorted(diff)
    assert c["wandb.name"] == p["wandb.name"] + "-cnorm"
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    pmc = build_morph_config(pcfg, tul=prt.model_cfg)
    tdiff = {f.name for f in dataclasses.fields(mc.tul)
             if getattr(mc.tul, f.name) != getattr(pmc.tul, f.name)}
    assert tdiff == {"slot_cell_pass_norm"}
    assert mc.tul.slot_cell_pass_norm == "rms" and pmc.tul.slot_cell_pass_norm == "off"
    assert rt.manifest.get("slot_cell_pass_norm") == "rms"
    assert prt.manifest.get("slot_cell_pass_norm") == "off"
