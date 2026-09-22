"""LCTUL-J Stage 1 (``tul.code_target_ema`` / ``tul.code_enc_var_lambda`` /
``tul.code_enc_var_gamma``) — the moving EMA target and its online variance floor. One
test per bullet of the Stage 1 acceptance list in
``.agents/notes/proposed/architecture/2026-09-22-lctul-ema-target-and-factors.md``.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_tul_code_ema.py -q

CPU only, fp32, the strict-geometry tiny fixture (tests/test_tul_strict_geometry.py) and
the `tul.code_target_ref` fixture pattern (tests/test_tul_code_ref.py).

What this file pins:

* **Defaults are a no-op.** ``code_target_ema: 0.0`` / ``code_enc_var_lambda: 0.0`` change
  nothing the forward computes or the trainer does — loss, logits, parameter names, and
  (since the trainer never calls the EMA update at ``m == 0``) the twin's state all agree
  with the same model built with the two fields simply absent from the ``tul:`` kwargs
  (dataclass defaults, so "absent" and "explicit 0.0" are the same value by construction).
* **One EMA update** moves every twin tensor — plain parameters, a parametrised ternary
  shadow weight, and float buffers — to ``m * before + (1 - m) * live``, touches nothing
  on the live model, and moves the twin's forward output.
* **The name-set assertion** raises, with the missing/extra names named, when the live
  model and the twin stop agreeing on what tensors exist.
* **The checkpoint round trip**: the twin's state saves, reloads strictly, and a further
  EMA update continues from the reloaded tensors (not from a re-snapshot).
* **The floor** (:func:`code_enc_var_floor`, a pure function): matches a hand computation,
  reads an exact 0 when every coordinate's std already clears ``gamma``, is positive with
  finite gradients on a collapsed (all-equal) batch, ignores invalid slots exactly, and
  reads an exact 0 with fewer than two valid slots.
* **The refusals**: every ``TULConfig`` bound in the spec.
* **The configs**: ``tul_code_ema`` / ``tul_code_ema0`` compose, build, and both keys are
  in ``KNOWN_TUL_KEYS``.
* **A real training step** on the tiny ``code_target_ref`` model with both knobs on: finite
  loss, the four new stats keys present, the twin moved.
"""

from __future__ import annotations

import copy

import pytest
import torch

from test_tul_strict_geometry import _pack, _runtime, _tiny, _tul  # noqa: E402

from morph.model.tul import TULConfig
from morph.model.tul_code import code_enc_var_floor
from morph.model.transformer import MORPHTransformer
from morph.model.ternary_qat import apply_ternary_qat
from morph.training.tul_setup import KNOWN_TUL_KEYS

_FRONT = ("prelude.", "embed.")
CONFIGS = ["tul_code_ema", "tul_code_ema0"]


def _model(seed: int = 5, on: bool = True, code_ref: bool = True, ema: float = 0.0,
          var_lambda: float = 0.0, snapshot: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    base_tul = dict(tg_geometry="strict", code_target=on, code_target_ref=code_ref,
                    code_target_ema=ema, code_enc_var_lambda=var_lambda)
    base_tul.update(tul_kw)
    base = dict(tul=_tul(**base_tul), n_core=2, mean_depth=3, max_depth=4, bptt_depth=4,
                retention=False, dropout=0.0, core_fixed_point_lambda=0.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    m = MORPHTransformer(_tiny(**base)).float()
    if code_ref and snapshot:
        m.tul_code_ref_snapshot()
    return m


def _run(m: MORPHTransformer, seed: int = 3, **kw):
    _ids, inp, lab, layout = _pack()
    torch.manual_seed(seed)
    return m(inp, labels=lab, slot_layout=layout, **kw)


# ── defaults are a no-op ────────────────────────────────────────────────────

def test_defaults_are_bit_identical_to_the_fields_absent():
    """``code_target_ema`` / ``code_enc_var_lambda`` never appearing in a ``TULConfig``
    call (a ``TULConfig`` built the way every pre-Stage-1 code path built one, with no
    knowledge these fields exist) and the two fields set EXPLICITLY to their dataclass
    default (``0.0``) must produce THE SAME object — a dataclass has no way to represent
    "field absent", so this is the honest form of the claim: passing 0.0 is indistinguishable
    from never passing anything, at construction AND across 3 forward+backward+(no-op
    EMA-update) steps. The trainer only calls ``tul_code_ref_ema_update`` when
    ``code_target_ema > 0`` (morph/training/train.py), so at 0.0 the update is never
    invoked and the twin never moves, exactly as it never did before this key existed.
    """
    tul_absent = _tul(tg_geometry="strict", code_target=True, code_target_ref=True)
    tul_explicit = _tul(tg_geometry="strict", code_target=True, code_target_ref=True,
                        code_target_ema=0.0, code_enc_var_lambda=0.0,
                        code_enc_var_gamma=1.0)
    assert tul_absent == tul_explicit, "0.0 is not indistinguishable from unset"

    torch.manual_seed(11)
    a = MORPHTransformer(_tiny(tul=tul_absent, n_core=2, mean_depth=3, max_depth=4,
                               bptt_depth=4, retention=False, dropout=0.0,
                               core_fixed_point_lambda=0.0, ckpt_grad_iters=0)).float()
    a.tul_code_ref_snapshot()
    torch.manual_seed(11)
    b = MORPHTransformer(_tiny(tul=tul_explicit, n_core=2, mean_depth=3, max_depth=4,
                               bptt_depth=4, retention=False, dropout=0.0,
                               core_fixed_point_lambda=0.0, ckpt_grad_iters=0)).float()
    b.tul_code_ref_snapshot()
    assert set(n for n, _ in a.named_parameters()) == set(n for n, _ in b.named_parameters())
    for n, pa in a.named_parameters():
        pb = dict(b.named_parameters())[n]
        assert torch.equal(pa, pb), f"{n}: differs at construction"
    twin_a0 = copy.deepcopy(a.code_ref.state_dict())
    twin_b0 = copy.deepcopy(b.code_ref.state_dict())
    opt_a = torch.optim.SGD([p for p in a.parameters() if p.requires_grad], lr=0.05)
    opt_b = torch.optim.SGD([p for p in b.parameters() if p.requires_grad], lr=0.05)
    for step in range(3):
        opt_a.zero_grad()
        out_a = _run(a, seed=100 + step)
        out_a["loss"].backward()
        opt_a.step()
        # The trainer's own gate (train.py: `if _code_ema_m > 0.0:`) — at m=0.0 this
        # call never happens, which is what "0.0 keeps the frozen twin" means.
        opt_b.zero_grad()
        out_b = _run(b, seed=100 + step)
        out_b["loss"].backward()
        opt_b.step()
        assert torch.equal(out_a["loss"], out_b["loss"]), f"step {step}: loss diverged"
        assert torch.equal(out_a["code_target_cos"], out_b["code_target_cos"]), \
            f"step {step}: code_target_cos diverged"
    for n, pa in a.named_parameters():                    # the whole trajectory matched
        pb = dict(b.named_parameters())[n]
        assert torch.equal(pa, pb), f"{n}: diverged after 3 steps"
    for k, v in a.code_ref.state_dict().items():
        assert torch.equal(v, twin_a0[k]), f"{k}: twin A moved despite ema=0.0"
    for k, v in b.code_ref.state_dict().items():
        assert torch.equal(v, twin_b0[k]), f"{k}: twin B moved despite ema absent"
    assert set(a.code_ref.state_dict()) == set(b.code_ref.state_dict())
    for k, v in a.code_ref.state_dict().items():
        assert torch.equal(v, b.code_ref.state_dict()[k]), f"{k}: twins disagree"


# ── one EMA update ───────────────────────────────────────────────────────────

def test_one_update_moves_every_twin_tensor_to_the_lerp_and_nothing_else():
    m = _model(seed=5, ema=0.5, var_lambda=0.0, snapshot=False)
    apply_ternary_qat(m, scope="backbone")           # gives us a real shadow-weight name
    m.tul_code_ref_snapshot()                        # AFTER quantisation, per the contract
    before_live = {n: p.detach().clone() for n, p in m.named_parameters()}
    before_ref_p = {n: p.detach().clone() for n, p in m.code_ref.named_parameters()}
    before_ref_b = {n: b.detach().clone() for n, b in m.code_ref.named_buffers()}
    z_before = _code_z(m)
    with torch.no_grad():                            # simulate an optimizer step
        for _, p in m.named_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m.tul_code_ref_ema_update(0.5)
    live_now = dict(m.named_parameters())
    for n, live in before_live.items():
        assert not torch.equal(live_now[n], live), f"{n}: the perturbation is vacuous"
    ref_p_now = dict(m.code_ref.named_parameters())
    for n, before in before_ref_p.items():
        expect = 0.5 * before + 0.5 * live_now[n].detach()
        assert torch.allclose(ref_p_now[n], expect, atol=1e-6, rtol=1e-5), \
            f"{n}: not m*before + (1-m)*live"
    ref_b_now = dict(m.code_ref.named_buffers())
    for n, before in before_ref_b.items():
        if not torch.is_floating_point(before):
            continue
        live_bufs = dict(m.named_buffers())
        expect = 0.5 * before + 0.5 * live_bufs[n].detach()
        assert torch.allclose(ref_b_now[n], expect, atol=1e-6, rtol=1e-5), f"{n}: buffer not lerped"
    # the parametrised ternary shadow weight, checked explicitly by name
    shadow_names = [n for n in before_ref_p if n.endswith("parametrizations.weight.original")]
    assert shadow_names, "the ternary QAT wrap produced no parametrised weight to test"
    sn = shadow_names[0]
    expect = 0.5 * before_ref_p[sn] + 0.5 * live_now[sn].detach()
    assert torch.allclose(ref_p_now[sn], expect, atol=1e-6, rtol=1e-5)
    # the live model is untouched by the EMA CALL itself (it already held the perturbed
    # values before the call; assert they are EXACTLY what they were right after the
    # perturbation, i.e. the update wrote nothing back into them)
    for n, p in m.named_parameters():
        assert torch.equal(p.detach(), live_now[n]), f"{n}: the EMA update wrote the live model"
    assert not torch.equal(z_before, _code_z(m)), "the twin's forward output did not move"


def test_non_float_buffers_are_copied_not_lerped():
    m = _model(seed=5, ema=0.5, snapshot=False)
    m.tul_code_ref_snapshot()
    int_bufs = [n for n, b in m.code_ref.named_buffers() if not torch.is_floating_point(b)]
    if not int_bufs:
        pytest.skip("this tiny fixture registers no integer/bool buffers")
    n = int_bufs[0]
    with torch.no_grad():
        live_buf = dict(m.named_buffers())[n]
        live_buf.add_(1)
    m.tul_code_ref_ema_update(0.5)
    assert torch.equal(dict(m.code_ref.named_buffers())[n], dict(m.named_buffers())[n]), \
        f"{n}: a non-float buffer was lerped instead of copied"


def test_ema_update_refuses_m_outside_zero_one():
    m = _model(seed=5, ema=0.5, snapshot=False)
    m.tul_code_ref_snapshot()
    with pytest.raises(ValueError, match=r"m must be in \[0, 1\)"):
        m.tul_code_ref_ema_update(1.0)
    with pytest.raises(ValueError, match=r"m must be in \[0, 1\)"):
        m.tul_code_ref_ema_update(-0.1)


def test_ema_update_with_no_twin_is_a_no_op():
    m = _model(seed=5, code_ref=False)
    m.tul_code_ref_ema_update(0.9)                   # must not raise: `ref is None` -> return


# ── the name-set assertion ──────────────────────────────────────────────────

def test_the_name_set_assertion_raises_and_names_the_difference():
    m = _model(seed=5, ema=0.5, snapshot=False)
    m.tul_code_ref_snapshot()
    dropped = next(m.code_ref.named_parameters())[0]
    orig_np = m.code_ref.named_parameters

    def _missing_one(*a, **k):
        for n, p in orig_np(*a, **k):
            if n != dropped:
                yield n, p
    m.code_ref.named_parameters = _missing_one
    with pytest.raises(RuntimeError, match="do not name the same tensors") as ei:
        m.tul_code_ref_ema_update(0.5)
    assert dropped in str(ei.value)


def test_the_name_set_check_runs_only_once_per_instance():
    """The FIRST update asserts the name sets; later updates skip the (linear-time) check —
    proved by corrupting the twin's view AFTER a successful first update and confirming the
    second update does NOT raise (it would, if the check ran again)."""
    m = _model(seed=5, ema=0.5, snapshot=False)
    m.tul_code_ref_snapshot()
    m.tul_code_ref_ema_update(0.5)                    # first update: checks and passes
    assert m.__dict__.get("_code_ema_names_checked") is True
    dropped = next(m.code_ref.named_parameters())[0]
    orig_np = m.code_ref.named_parameters

    def _missing_one(*a, **k):
        for n, p in orig_np(*a, **k):
            if n != dropped:
                yield n, p
    m.code_ref.named_parameters = _missing_one
    m.tul_code_ref_ema_update(0.5)                    # must NOT raise: the check is cached


# ── the checkpoint round trip ────────────────────────────────────────────────

def test_checkpoint_round_trip_and_the_update_continues_from_it(tmp_path):
    from morph.training.train import save_checkpoint, read_code_ref_state
    m = _model(seed=5, ema=0.5, snapshot=False)
    m.tul_code_ref_snapshot()
    for _ in range(2):
        with torch.no_grad():
            for _, p in m.named_parameters():
                p.add_(torch.randn_like(p) * 0.02)
        m.tul_code_ref_ema_update(0.5)
    saved_ref = copy.deepcopy(m.tul_code_ref_state())
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, 2, m, opt, torch.amp.GradScaler("cpu", enabled=False), None)
    got = read_code_ref_state(path)
    assert got is not None and set(got) == set(saved_ref)
    for k, v in saved_ref.items():
        assert torch.equal(got[k], v), f"{k}: the checkpoint's twin does not match"
    # a fresh model resumes strictly from the checkpoint's twin (train.py's own contract)
    m2 = _model(seed=999, ema=0.5, snapshot=False)
    assert m2.tul_code_ref_snapshot(got) == "resume"
    for k, v in m2.code_ref.state_dict().items():
        assert torch.equal(v, saved_ref[k]), f"{k}: the resumed twin does not match the saved one"
    # and the NEXT update continues from the reloaded tensors, not from a re-snapshot of
    # m2's (differently seeded) live weights
    reloaded_ref_p = {n: p.detach().clone() for n, p in m2.code_ref.named_parameters()}
    with torch.no_grad():
        for _, p in m2.named_parameters():
            p.add_(torch.randn_like(p) * 0.02)
    m2.tul_code_ref_ema_update(0.5)
    live2 = dict(m2.named_parameters())
    for n, before in reloaded_ref_p.items():
        expect = 0.5 * before + 0.5 * live2[n].detach()
        assert torch.allclose(dict(m2.code_ref.named_parameters())[n], expect,
                              atol=1e-6, rtol=1e-5), f"{n}: did not continue from the reload"


# ── the floor ─────────────────────────────────────────────────────────────

def _code_z(m: MORPHTransformer):
    _ids, inp, lab, layout = _pack()
    m.eval()
    with torch.no_grad():
        return m(inp, labels=lab, slot_layout=layout)["code_z"].clone()


def test_floor_matches_a_hand_computation():
    B, S, M, C = 2, 2, 2, 4
    pred = torch.zeros(B, S, M, C, requires_grad=True)
    with torch.no_grad():
        pred[0, 0, 0] = torch.full((C,), 1.0)
        pred[0, 1, 0] = torch.full((C,), 3.0)
        pred[1, 0, 0] = torch.full((C,), 1.0)
        # cell 1 stays all-zero at every valid slot -> sigma = sqrt(eps)
    ok = torch.zeros(B, S, dtype=torch.bool)
    ok[0, 0], ok[0, 1], ok[1, 0] = True, True, True
    loss, std_mean, active = code_enc_var_floor(pred, ok, gamma=1.0, eps=1e-4)
    v = torch.tensor([1.0, 3.0, 1.0])
    sigma0 = float((v.var(unbiased=False) + 1e-4).sqrt())
    sigma1 = float((0.0 + 1e-4) ** 0.5)
    floor0, floor1 = max(0.0, 1.0 - sigma0), max(0.0, 1.0 - sigma1)
    expect_loss = (floor0 * C + floor1 * C) / (2 * C)
    expect_std = (sigma0 * C + sigma1 * C) / (2 * C)
    assert float(loss.detach()) == pytest.approx(expect_loss, abs=1e-6)
    assert float(std_mean) == pytest.approx(expect_std, abs=1e-6)
    assert float(active) == pytest.approx(1.0, abs=1e-6)   # both sigmas < gamma=1.0


def test_floor_is_exact_zero_when_every_std_clears_gamma():
    vals = torch.tensor([[[[-100.0]], [[100.0]], [[-100.0]], [[100.0]], [[-100.0]]]])
    ok = torch.ones(1, 5, dtype=torch.bool)
    loss, std_mean, active = code_enc_var_floor(vals.clone().requires_grad_(True), ok, gamma=1.0)
    assert float(loss.detach()) == 0.0
    assert float(active) == 0.0
    assert float(std_mean) > 90.0


def test_floor_is_positive_with_finite_gradients_on_a_collapsed_batch():
    B, S, M, C = 1, 4, 2, 3
    pred = torch.full((B, S, M, C), 0.7, requires_grad=True)   # every valid cell IDENTICAL
    ok = torch.ones(B, S, dtype=torch.bool)
    loss, std_mean, active = code_enc_var_floor(pred, ok, gamma=1.0)
    assert float(loss.detach()) > 0.0
    assert float(std_mean) == pytest.approx((1e-4) ** 0.5, abs=1e-6)
    assert float(active) == 1.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_floor_ignores_invalid_slots_exactly():
    p = torch.tensor([[[[1.0]], [[2.0]], [[999.0]]]], requires_grad=True)
    ok = torch.tensor([[True, True, False]])
    l_a, _, _ = code_enc_var_floor(p, ok, gamma=1.0)
    p2 = p.detach().clone()
    p2[0, 2, 0, 0] = -12345.0
    p2.requires_grad_(True)
    l_b, _, _ = code_enc_var_floor(p2, ok, gamma=1.0)
    assert torch.equal(l_a.detach(), l_b.detach())


def test_floor_is_exact_zero_with_fewer_than_two_valid_slots():
    B, S, M, C = 1, 2, 2, 3
    pred = torch.randn(B, S, M, C, requires_grad=True)
    ok0 = torch.zeros(B, S, dtype=torch.bool)
    loss0, std0, active0 = code_enc_var_floor(pred, ok0, gamma=1.0)
    assert float(loss0.detach()) == 0.0 and float(std0) == 0.0 and float(active0) == 0.0
    loss0.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) == 0.0

    pred1 = torch.randn(B, S, M, C, requires_grad=True)
    ok1 = torch.zeros(B, S, dtype=torch.bool)
    ok1[0, 0] = True
    loss1, std1, active1 = code_enc_var_floor(pred1, ok1, gamma=1.0)
    assert float(loss1.detach()) == 0.0 and float(std1) == 0.0 and float(active1) == 0.0


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,match", [
    (dict(code_target_ema=1.0, code_target=True, code_target_ref=True), r"in \[0, 1\)"),
    (dict(code_target_ema=-0.1, code_target=True, code_target_ref=True), r"in \[0, 1\)"),
    # code_target=False with code_target_ema set: `_check_code_target` runs BEFORE
    # `_check_code_ema` and would raise its OWN (different) message if code_target_ref
    # were also set here, so this case leaves code_target_ref at its default (False).
    (dict(code_target_ema=0.5, code_target=False), "needs tul.code_target"),
    (dict(code_target_ema=0.5, code_target=True, code_target_ref=False), "needs tul.code_target_ref"),
    (dict(code_enc_var_lambda=-0.1, code_target=True), "must be >= 0"),
    (dict(code_enc_var_gamma=0.0, code_target=True), "must be > 0"),
    (dict(code_enc_var_gamma=-1.0, code_target=True), "must be > 0"),
    (dict(code_enc_var_lambda=0.1, code_target=False), "needs tul.code_target"),
])
def test_config_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        _tul(tg_geometry="strict", **kw)


def test_defaults_pass_every_check():
    _tul(tg_geometry="strict", code_target=True, code_target_ref=True)   # must not raise
    TULConfig(prefix_k=2, slot_id=4)                                     # bare defaults


# ── the configs ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CONFIGS)
def test_the_configs_compose_and_build(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    assert rt is not None
    tc = rt.model_cfg
    assert tc.code_target and tc.code_target_ref and tc.tg_geometry == "strict"
    assert list(cfg.training.train_only) == [], "the whole model trains on this arm"
    if name == "tul_code_ema":
        assert tc.code_target_ema == pytest.approx(0.996)
        assert tc.code_enc_var_lambda == pytest.approx(0.02)
    else:
        assert tc.code_target_ema == 0.0
        assert tc.code_enc_var_lambda == 0.0
    assert tc.code_enc_var_gamma == pytest.approx(1.0)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc, n_core=2, mean_depth=3, max_depth=4, bptt_depth=4)).float()
    m.tul_code_ref_snapshot()
    _ids0, inp, lab, layout = _pack()
    m.train()
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])


def test_known_tul_keys_carries_the_new_fields():
    assert {"code_target_ema", "code_enc_var_lambda", "code_enc_var_gamma"} <= KNOWN_TUL_KEYS


# ── a real training step, both knobs on ─────────────────────────────────────

def test_a_training_step_with_both_knobs_on_runs_and_the_twin_moves():
    m = _model(seed=9, ema=0.5, var_lambda=0.02, snapshot=False)
    m.tul_code_ref_snapshot()
    twin0 = copy.deepcopy(m.code_ref.state_dict())
    m.train()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.05)
    opt.zero_grad()
    out = _run(m, seed=42)
    assert torch.isfinite(out["loss"])
    for k in ("code_enc_var", "code_enc_std", "code_enc_active", "code_tgt_std",
              "code_enc_var_weighted"):
        assert k in out, f"missing stat {k}"
        assert torch.isfinite(out[k]) if torch.is_tensor(out[k]) else out[k] == out[k]
    out["loss"].backward()
    opt.step()
    m.tul_code_ref_ema_update(0.5)
    moved = any(not torch.equal(v, twin0[k]) for k, v in m.code_ref.state_dict().items())
    assert moved, "the twin did not move after a real training step + EMA update"


def test_lambda_zero_reports_no_enc_var_term_but_always_reports_target_std():
    m = _model(seed=9, ema=0.0, var_lambda=0.0, snapshot=False)
    m.tul_code_ref_snapshot()
    m.train()
    out = _run(m, seed=42)
    assert "code_enc_var" not in out and "code_enc_std" not in out \
        and "code_enc_active" not in out
    assert "code_tgt_std" in out
