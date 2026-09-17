"""``tul.code_target`` — the slot loop regressed onto the frozen VAE code (spec §17), one
test per invariant, each failing when its mechanism is removed.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_tul_code_target.py -q

CPU only, fp32, the strict-geometry tiny fixture (tests/test_tul_strict_geometry.py).

What this file pins:

* **Off is the ruler.** ``code_target: false`` builds neither E nor the projection and
  every shared weight equals the on model's (no global RNG draw). On, ``W_prefix`` is
  built and inert.
* **The write.** The coda reads the PROJECTION's cells at the prefix positions and
  ``prefix_project`` is never called; ``zero`` zeroes them, ``shuffle`` permutes whole
  slots, ``code_mode="encoder"`` hands the coda E's own code, ``code_given`` overrides.
* **The term.** Positive, ``2 (1 - cos)`` on unit-RMS cells, in ``loss`` with its weighted
  twin exposed; E's parameters get NO gradient; under ``code_target_detach`` the token CE
  reaches nothing in the loop (the regression alone trains it), without it the CE does.
* **The instrument.** ``code_target_cos_l{t}`` for the entry state and every realised
  pass at train; none at eval.
* **The loader.** A VAE-stage (``tul.code``) checkpoint's thinker tensors are dropped
  LOUDLY into a code-target model and nothing else is homeless.
* **The configs.** ``tul_code_target`` / ``_ce`` compose, build and run.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_strict_geometry import _pack, _runtime, _tiny, _tul  # noqa: E402

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig, TULSlots
from morph.model.tul_code import TULCodeProj, code_rmsnorm, code_target_regression
from morph.training.train import (CODE_THINKER_PREFIXES, drop_code_thinker_keys,
                                  drop_retired_tul_keys)

TARGET_CONFIGS = ["tul_code_target", "tul_code_target_ce"]


def _model(seed: int = 5, on: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    base_tul = dict(tg_geometry="strict", code_target=on)
    base_tul.update(tul_kw)
    base = dict(tul=_tul(**base_tul), n_core=2, mean_depth=3, max_depth=4, bptt_depth=4,
                retention=False, dropout=0.0, core_fixed_point_lambda=0.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_tiny(**base)).float()


def _run(m: MORPHTransformer, seed: int = 3, **kw):
    _ids, inp, lab, layout = _pack()
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout, **kw)


def _grads(m: MORPHTransformer, prefix: str) -> float:
    return sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
               if n.startswith(prefix) and p.grad is not None)


# ── off is the ruler ─────────────────────────────────────────────────────────

def test_off_builds_nothing_and_shares_every_weight_with_the_ruler():
    on, off = _model(on=True), _model(on=False)
    assert off.tul_code_enc is None and off.tul_code_proj is None
    assert on.tul_code_enc is not None and on.tul_code_proj is not None
    assert on.tul_code_head is None and on.tul_code_time is None, "no thinker on a target model"
    off_p = dict(off.named_parameters())
    for n, p in on.named_parameters():
        if n.startswith(("tul_code_enc.", "tul_code_proj.")):
            continue
        assert torch.equal(p, off_p[n]), f"{n} differs: the target modules drew from the global RNG"
    assert on.tul.W_prefix is not None and not on.tul.W_prefix.requires_grad
    assert off.tul.W_prefix.requires_grad
    assert all(not p.requires_grad for p in on.tul_code_enc.parameters()), "E is frozen at build"
    assert torch.equal(on.tul_code_proj.W_code[0], torch.eye(on.cfg.d_model)), "identity init"


def test_refusals():
    with pytest.raises(NotImplementedError, match="code model has no slot loop"):
        _tul(tg_geometry="strict", code_target=True, code=True)
    with pytest.raises(NotImplementedError, match="slot geometry"):
        _tul(tg_geometry="strict", code_target=True, tokens_through_core=True)
    with pytest.raises(NotImplementedError, match="strict"):
        _tul(code_target=True)                       # geometry defaults to "restrict"
    with pytest.raises(ValueError, match="detach_z"):
        _tul(tg_geometry="strict", code_target=True, detach_z=True)
    with pytest.raises(ValueError, match="code_target_weight"):
        _tul(tg_geometry="strict", code_target=True, code_target_weight=-1.0)
    with pytest.raises(ValueError, match="silently ignored"):
        _tul(code_target=False, code_target_weight=2.0)
    with pytest.raises(ValueError, match="code_target_lambda"):
        # the OLD knob of the same prefix is the flow arm's and stays refused off-code
        _tul(code_target=False, code_target_lambda=0.5)
    m = _model().eval()
    _ids, inp, lab, layout = _pack()
    with torch.no_grad():
        with pytest.raises(NotImplementedError, match="no sampler"):
            m(inp, labels=lab, slot_layout=layout, code_steps=4)
        with pytest.raises(NotImplementedError, match="only 'encoder'"):
            m(inp, labels=lab, slot_layout=layout, code_mode="sampled")
    m.train()
    with pytest.raises(ValueError, match="EVAL-ONLY"):
        m(inp, labels=lab, slot_layout=layout, code_mode="encoder")


# ── the write ────────────────────────────────────────────────────────────────

def test_the_coda_reads_the_projection_and_prefix_project_is_never_called(monkeypatch):
    seen: list[torch.Tensor] = []
    orig = TULCodeProj.forward

    def spy(self, r, valid):
        out = orig(self, r, valid)
        seen.append(out.detach().clone())
        return out

    def boom(*a, **k):
        raise AssertionError("prefix_project ran on a code-target model")
    monkeypatch.setattr(TULCodeProj, "forward", spy)
    monkeypatch.setattr(TULSlots, "prefix_project", boom)
    m = _model().eval()
    with torch.no_grad():
        out = _run(m)
    assert len(seen) == 1, "eval: one projection (no per-pass readings)"
    assert torch.equal(out["code_cells"], seen[0])
    B, S, M, C = out["code_cells"].shape
    assert M == m.cfg.tul.prefix_k and C == m.cfg.d_model
    # unit RMS per valid cell, exactly 0 on a slot with no code
    _ids, inp, lab, layout = _pack()
    rms = out["code_cells"].pow(2).mean(-1).sqrt()
    valid = rms > 0
    assert torch.allclose(rms[valid], torch.ones_like(rms[valid]), atol=1e-4)
    assert not bool(valid[:, -1].any()), "the row's last slot precedes the tail: no code, cell 0"


def test_the_cells_move_the_coda_and_the_ablations_act_on_them():
    m = _model().eval()
    _ids, inp, lab, layout = _pack()
    with torch.no_grad():
        n = m(inp, labels=lab, slot_layout=layout)
        z = m.tul_forward_ablated(inp, lab, layout, plan_mode="zero")
        torch.manual_seed(0)
        s = m.tul_forward_ablated(inp, lab, layout, plan_mode="shuffle")
    assert float(z["code_cells"].abs().sum()) == 0.0
    assert float(n["loss"]) != float(z["loss"]), "zeroing the cells must move the coda"
    # shuffle: every row's cells are a permutation of the normal cells over slots
    for b in range(n["code_cells"].shape[0]):
        a = n["code_cells"][b].flatten(1).sum(-1).sort().values
        c = s["code_cells"][b].flatten(1).sum(-1).sort().values
        assert torch.allclose(a, c, atol=1e-5)
    assert not torch.equal(n["code_cells"], s["code_cells"])


def test_encoder_mode_hands_the_coda_e_code_and_code_given_overrides(monkeypatch):
    from morph.model.tul_code import TULCodeEncoder
    seen: dict = {}
    orig = TULCodeEncoder.forward

    def spy(self, xs, layout):
        z, ok = orig(self, xs, layout)
        seen["z"], seen["ok"] = z.detach().clone(), ok.clone()
        return z, ok
    monkeypatch.setattr(TULCodeEncoder, "forward", spy)
    m = _model().eval()
    _ids, inp, lab, layout = _pack()
    with torch.no_grad():
        own = m(inp, labels=lab, slot_layout=layout)
        orc = m(inp, labels=lab, slot_layout=layout, code_mode="encoder")
    assert torch.allclose(orc["code_cells"].float(), seen["z"].float(), atol=1e-6)
    assert not torch.allclose(own["code_cells"].float(), seen["z"].float(), atol=1e-3)
    assert float(orc["loss"]) != float(own["loss"])
    # code_given at slot 1 of row 0 replaces that slot's cells and nothing else
    given = torch.zeros_like(own["code_cells"])
    gmask = torch.zeros(given.shape[:2], dtype=torch.bool)
    given[0, 1] = code_rmsnorm(torch.randn(given.shape[2:]))
    gmask[0, 1] = True
    with torch.no_grad():
        g = m(inp, labels=lab, slot_layout=layout, code_mode="generate", code_given=given,
              code_given_mask=gmask)
    assert torch.allclose(g["code_cells"][0, 1], given[0, 1], atol=1e-6)
    keep = ~gmask
    assert torch.allclose(g["code_cells"][keep], own["code_cells"][keep], atol=1e-6)
    with pytest.raises(ValueError, match="code_given_mask"):
        m(inp, labels=lab, slot_layout=layout, code_mode="generate", code_given=given)


# ── the term ─────────────────────────────────────────────────────────────────

def test_the_regression_helper_is_two_one_minus_cos_on_unit_rms_cells():
    g = torch.Generator().manual_seed(0)
    p = code_rmsnorm(torch.randn(2, 3, 2, 16, generator=g))
    z = code_rmsnorm(torch.randn(2, 3, 2, 16, generator=g))
    ok = torch.tensor([[True, True, False], [True, False, False]])
    loss, cos, n = code_target_regression(p, z, ok)
    assert float(n) == 6.0
    assert abs(float(loss) - 2.0 * (1.0 - float(cos))) < 1e-5
    same, _c, _n = code_target_regression(z, z, ok)
    assert float(same) < 1e-6 and abs(float(_c) - 1.0) < 1e-5
    zero, _c0, n0 = code_target_regression(p, z, torch.zeros_like(ok))
    assert float(zero) == 0.0 and float(n0) == 0.0 and zero.requires_grad is False


def test_the_term_is_positive_in_the_loss_and_e_gets_no_gradient():
    m = _model(tul_code_target_weight=2.0).train()
    out = _run(m)
    ct = float(out["code_target"])
    assert ct > 0.0
    assert abs(float(out["code_target_weighted"]) - 2.0 * ct) < 1e-5
    assert abs(ct - 2.0 * (1.0 - float(out["code_target_cos"]))) < 1e-4, "2 (1 - cos)"
    m0 = _model(tul_code_target_weight=0.0).train()
    out0 = _run(m0)
    assert abs((float(out["loss"].detach()) - float(out0["loss"].detach()))
               - float(out["code_target_weighted"])) < 1e-4
    out["loss"].backward()
    assert all(p.grad is None for p in m.tul_code_enc.parameters()), "E must never train"
    assert _grads(m, "tul_code_proj.") > 0.0
    assert m.tul.W_prefix.grad is None, "W_prefix is inert"


def test_detach_decides_whether_the_token_ce_reaches_the_loop():
    # weight 0 isolates the CE's route into the loop
    a = _model(tul_code_target_weight=0.0, tul_code_target_detach=True).train()
    _run(a)["loss"].backward()
    assert _grads(a, "core.") == 0.0, "detached: the CE reaches nothing in the loop"
    assert _grads(a, "tul_code_proj.") == 0.0
    b = _model(tul_code_target_weight=0.0, tul_code_target_detach=False).train()
    _run(b)["loss"].backward()
    assert _grads(b, "core.") > 0.0, "undetached: the CE trains the loop through the cells"
    assert _grads(b, "tul_code_proj.") > 0.0
    c = _model(tul_code_target_weight=1.0, tul_code_target_detach=True).train()
    _run(c)["loss"].backward()
    assert _grads(c, "core.") > 0.0, "the regression alone reaches the loop"


# ── the instrument ───────────────────────────────────────────────────────────

def test_per_pass_cosines_exist_at_train_and_not_at_eval():
    m = _model().train()
    out = _run(m)
    keys = sorted(k for k in out if k.startswith("code_target_cos_l"))
    assert keys[0] == "code_target_cos_l0" and len(keys) >= 2
    for k in keys:
        assert -1.0 <= float(out[k]) <= 1.0
    m.eval()
    with torch.no_grad():
        o = _run(m)
    assert not any(k.startswith("code_target_cos_l") for k in o)
    assert "code_target_cos" in o and "code_target" in o


# ── the loader ───────────────────────────────────────────────────────────────

def test_a_vae_stage_checkpoint_loads_with_the_thinker_dropped_loudly(capsys):
    torch.manual_seed(1)
    vae = MORPHTransformer(_tiny(tul=_tul(tg_geometry="strict", code=True), n_core=2,
                                 mean_depth=3, max_depth=4, bptt_depth=4)).float()
    state = {k: v.clone() for k, v in vae.state_dict().items()}
    thinker = sorted(k for k in state if k.startswith(CODE_THINKER_PREFIXES))
    assert thinker, "the fixture's VAE model must carry a thinker"
    tgt = _model()
    assert drop_retired_tul_keys(state, tgt, "vae.pt") == [], "W_prefix lives on both"
    dropped = drop_code_thinker_keys(state, tgt, "vae.pt")
    assert sorted(dropped) == thinker
    _out = capsys.readouterr().out
    assert "dropped" in _out and "code-thinker" in _out, "the drop must be LOUD"
    missing, unexpected = tgt.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert missing and all(k.startswith("tul_code_proj.") for k in missing), missing
    # E's tensors came across
    for k, v in vae.state_dict().items():
        if k.startswith("tul_code_enc."):
            assert torch.equal(dict(tgt.state_dict())[k], v)
    # on any other model the drop is a no-op
    assert drop_code_thinker_keys({k: v for k, v in vae.state_dict().items()}, vae, "x") == []


# ── the configs ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", TARGET_CONFIGS)
def test_the_target_arms_compose_and_build_and_run(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    assert rt is not None
    tc = rt.model_cfg
    assert tc.code_target and not tc.code and tc.tg_geometry == "strict" and not tc.spandec
    assert tc.code_target_detach == (name == "tul_code_target")
    assert "tul_code_proj." in list(cfg.training.train_only)
    assert not any(p.startswith("tul_code_enc") for p in cfg.training.train_only)
    assert not bool(cfg.model.use_kernels)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, n_core=2, mean_depth=3, max_depth=4, bptt_depth=4)).eval().float()
    _ids0, inp, lab, layout = _pack()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]) and "code_target_cos" in out


# ── the code-ONLY arm (tul.code_target_skip_coda) and the InfoNCE term ───────

CODE_ONLY_CONFIGS = ["tul_code_only", "tul_code_only_nce"]


def test_skip_coda_train_forward_ends_at_the_projection():
    m = _model(tul_code_target_skip_coda=True).train()
    out = _run(m)
    assert out["logits"] is None and "ce_tokens" not in out, "no coda, no token CE at train"
    assert "code_target_cos" in out and "code_target_cos_l0" in out
    # the loss is the code term plus the loop's own constraint and nothing else
    expect = float(out["code_target_weighted"]) + float(out.get("gain_reg_weighted", 0.0))
    assert abs(float(out["loss"]) - expect) < 1e-5, (float(out["loss"]), expect)
    out["loss"].backward()
    assert _grads(m, "core.") > 0.0, "the code term reaches the loop"
    assert _grads(m, "prelude.") > 0.0, "and the prelude (train_only is the config's job)"
    assert _grads(m, "coda.") == 0.0, "the coda never ran"
    assert _grads(m, "tul_code_enc.") == 0.0, "E is the frozen target"


def test_skip_coda_eval_forward_still_runs_the_coda():
    m = _model(tul_code_target_skip_coda=True).eval()
    with torch.no_grad():
        out = _run(m)
    assert "ce_tokens" in out and "code_cells" in out and torch.isfinite(out["loss"])
    # and it is the SAME eval forward as the arm with the coda at train
    torch.manual_seed(5)
    ref = _model(tul_code_target_skip_coda=False).eval()
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        o2 = _run(ref)
    assert torch.allclose(out["loss"], o2["loss"]) and torch.allclose(out["code_cells"], o2["code_cells"])


def test_infonce_term_trains_the_loop_and_reports_its_readings():
    m = _model(tul_code_target_loss="infonce", tul_code_target_tau=0.1).train()
    out = _run(m)
    assert float(out["code_target"]) > 0.0
    assert 0.0 <= float(out["code_target_acc"]) <= 1.0
    assert -1.0 <= float(out["code_target_cos"]) <= 1.0
    assert "code_target_mse" in out, "the L2 reading is kept beside the term"
    out["loss"].backward()
    assert _grads(m, "core.") > 0.0 and _grads(m, "tul_code_proj.") > 0.0
    assert _grads(m, "tul_code_enc.") == 0.0


def test_infonce_helper_contract():
    from morph.model.tul_code import code_target_infonce
    torch.manual_seed(0)
    B, S, M, C = 2, 5, 2, 16
    z = code_rmsnorm(torch.randn(B, S, M, C))
    ok = torch.ones(B, S, dtype=torch.bool)
    ok[1, 4] = False
    # the own code IS the prediction: top-1 is perfect and the loss is the batch's floor
    loss, cos, n, acc = code_target_infonce(z.clone().requires_grad_(True), z, ok, 0.1)
    assert float(acc) == 1.0 and abs(float(cos) - 1.0) < 1e-5 and int(n) == 9 * M
    # a foreign prediction: chance-level top-1 and a loss near log(n_valid)
    p = code_rmsnorm(torch.randn(B, S, M, C)).requires_grad_(True)
    loss2, cos2, _, acc2 = code_target_infonce(p, z, ok, 1.0)
    assert float(loss2) > float(loss)
    loss2.backward()
    assert p.grad is not None and float(p.grad.abs().sum()) > 0.0
    # no valid slot: an exact 0 that still carries the graph
    loss0, _, n0, _ = code_target_infonce(p, z, torch.zeros_like(ok), 0.1)
    assert float(loss0) == 0.0 and int(n0) == 0 and loss0.requires_grad


def test_shuffled_cosine_is_reported_at_train_and_eval():
    m = _model().train()
    out = _run(m)
    assert "code_target_cos_shuf" in out and -1.0 <= float(out["code_target_cos_shuf"]) <= 1.0
    m.eval()
    with torch.no_grad():
        o = _run(m)
    assert "code_target_cos_shuf" in o


@pytest.mark.parametrize("kw", [
    dict(tul_code_target_skip_coda=True, tul_code_target_detach=False),
    dict(tul_code_target_loss="cosine"),
    dict(tul_code_target_tau=0.0),
    dict(tul_code_target_skip_coda=True, tul_code_target_weight=0.0),
])
def test_code_only_refusals(kw):
    with pytest.raises(ValueError):
        _model(**kw)


def test_code_only_knobs_refused_when_the_target_is_off():
    for kw in (dict(tul_code_target_skip_coda=True), dict(tul_code_target_loss="infonce"),
               dict(tul_code_target_tau=0.2)):
        with pytest.raises(ValueError):
            _model(on=False, **kw)


@pytest.mark.parametrize("name", CODE_ONLY_CONFIGS)
def test_the_code_only_arms_compose_and_build_and_run(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    tc = rt.model_cfg
    assert tc.code_target and tc.code_target_skip_coda and tc.code_target_detach
    assert tc.code_target_loss == ("infonce" if name.endswith("_nce") else "l2")
    assert list(cfg.training.train_only) == [], "everything trains except the frozen E"
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, n_core=2, mean_depth=3, max_depth=4, bptt_depth=4)).float()
    _ids0, inp, lab, layout = _pack()
    m.train()
    out = m(inp, labels=lab, slot_layout=layout)
    assert out["logits"] is None and "ce_tokens" not in out and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert _grads(m, "tul_code_enc.") == 0.0
    m.eval()
    with torch.no_grad():
        o = m(inp, labels=lab, slot_layout=layout)
    assert "ce_tokens" in o and "code_target_cos" in o
