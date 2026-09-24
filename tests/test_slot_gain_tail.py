"""The slot map's PER-SLOT gain and the tail hinge on it (`model.slot_gain_tail_lambda`,
`model.slot_gain_tail_target`; LXTUL Stage 3, "move the slot map", 2026-09-24).

Files: morph/model/transformer.py (`MORPHConfig.slot_gain_tail_*`, `_slot_gain_penalty`,
`_slot_gain_reduce`, the `gain_slot_*` / `gain_tail_pen` groups), morph/training/train.py
(the parse and the logged keys), morph/configs/tul_slot_spandec_strict_e{1,4}probe*.yaml.

What each test pins:
  * tail lambda 0 is the PRE-CHANGE function: loss, every parameter gradient, the returned
    row gain / max and the global RNG after the step are bit-identical to a model whose
    `_slot_gain_penalty` and reducer are the pre-change code (kept verbatim below), on the
    single-sample hinge and on `slot_gain_all_iters`;
  * the per-slot gain g_s = ||(f(h + d) - f(h))_s|| / ||d_s|| equals an independent
    computation (the core step applied by the test to the two inputs the hinge used), on a
    synthetic cross-slot linear map over a 4-D carrier and on the real tiny model; the
    reducer's p50 / p90 / max / frac_gt1 match numpy on the valid values;
  * the tail term is lam * mean over VALID slots of relu(g_s - target)^2; perturbing a pad
    slot's state changes nothing; it is exactly zero when every g_s <= target; its gradient
    reaches the core and nothing upstream of the loop;
  * with `slot_gain_all_iters` every iteration contributes its own tail and its own slots;
  * the three Stage 3 configs compose, build and train one step, and differ from their
    parents by exactly the stated keys and the run name;
  * the refusals hold.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from test_slot_gain_reg import MAX_DEPTH, _model, _run, _same

from morph.model.transformer import MORPHTransformer

_NEW_KEYS = ("gain_slot_p50", "gain_slot_p90", "gain_slot_max", "gain_slot_frac_gt1",
             "gain_tail_pen")


# ── the pre-change code, VERBATIM (master 2f0c30d) ──────────────────────────────────────

def _pre_change_penalty(self, core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond,
                        mask, lam: float, carry=None) -> dict:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if h_in.is_cuda else None
    def _restore():
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
    hp = h_in.detach()
    e_arg = e_arg.detach() if torch.is_tensor(e_arg) else e_arg
    inj_arg = inj_arg.detach() if torch.is_tensor(inj_arg) else inj_arg
    ret_state = ret_state.detach() if torch.is_tensor(ret_state) else ret_state
    carry = carry.detach() if torch.is_tensor(carry) else carry
    m = mask.view(*mask.shape, *([1] * (hp.dim() - 2))).to(hp.dtype)
    v = torch.randn_like(hp) * m
    hn = hp.flatten(2).float().norm(dim=2)                                    # [B, S]
    vn = v.flatten(2).float().norm(dim=2)
    scale = (float(self.cfg.slot_gain_eps) * hn / (vn + 1e-6)).to(hp.dtype)
    d = v * scale.view(*scale.shape, *([1] * (hp.dim() - 2)))
    try:
        _restore()
        f0, _ = core_step(hp, e_arg, inj_arg, ret_state=ret_state, iter_idx=t,
                          stage_cond=stage_cond, carry=carry)
        _restore()
        f1, _ = core_step(hp + d, e_arg, inj_arg, ret_state=ret_state, iter_idx=t,
                          stage_cond=stage_cond, carry=carry)
    finally:
        _restore()
    num = ((f1 - f0) * m).float().flatten(1).norm(dim=1)                       # [B]
    den = d.float().flatten(1).norm(dim=1) + 1e-6
    gain = (num / den)                                                         # [B]
    hinge = torch.relu(gain - float(self.cfg.slot_gain_target))
    pen = lam * (hinge * hinge).mean()
    return {"gain": gain.detach().mean(), "gain_max": gain.detach().max(), "penalty": pen}


def _pre_change_reduce(terms: list[dict]) -> dict:
    out = {
        "gain": torch.stack([g["gain"] for g in terms]).mean(),
        "gain_max": torch.stack([g["gain_max"] for g in terms]).max(),
        "penalty": torch.stack([g["penalty"] for g in terms]).sum(),
        "n_iters": float(len(terms)),
    }
    # Bookkeeping only: the forward copies these five keys into its groups. They are NaN
    # here and the comparison never reads them.
    out.update({k: out["gain"].new_tensor(float("nan")) for k in _NEW_KEYS})
    return out


def _pre_change_model(**kw) -> MORPHTransformer:
    m = _model(**kw)
    m._slot_gain_penalty = _pre_change_penalty.__get__(m)
    m._slot_gain_reduce = _pre_change_reduce
    return m


@pytest.mark.parametrize("all_iters", [False, True])
@pytest.mark.parametrize("target", [0.0, 1e6])       # binding everywhere / never binding
@pytest.mark.parametrize("tail_target", [None, 1e-3])  # the default / one every slot exceeds
def test_tail_off_is_bit_identical_to_the_pre_change_function(all_iters, target, tail_target):
    kw = dict(seed=3, slot_gain_lambda=1.0, slot_gain_target=target,
              slot_gain_all_iters=all_iters)
    out_o, g_o, rng_o = _run(_pre_change_model(**kw))
    # slot_gain_tail_lambda defaults to 0. At a tail target every slot exceeds, a tail that
    # leaked into the loss at lambda 0 (or at the row lambda) would move the loss.
    tail_kw = {} if tail_target is None else dict(slot_gain_tail_target=tail_target)
    out_n, g_n, rng_n = _run(_model(**kw, **tail_kw))
    assert torch.equal(rng_o, rng_n), "the per-slot readings moved the global RNG stream"
    for k in ("loss", "gain_est", "gain_est_max", "gain_reg_weighted", "mux_local"):
        assert torch.equal(out_o[k], out_n[k]), k
    assert _same(g_o, g_n), "the per-slot readings changed a gradient"
    assert float(out_n["gain_tail_pen"]) == 0.0
    for k in _NEW_KEYS:
        assert torch.isfinite(out_n[k]), k


# ── the per-slot gain against an independent computation ─────────────────────────────────

B, S, N, C = 3, 7, 4, 8          # a 4-D carrier: [B, S, n, C], per-slot norm over (n, C)


class _Recorder:
    """A core step that records its inputs: the hinge calls it on h and on h + d."""

    def __init__(self, step):
        self.step, self.inputs = step, []

    def __call__(self, h, e, inj, **kw):
        self.inputs.append(h.detach().clone())
        return self.step(h, e, inj, **kw)


def _linear_step(c: torch.Tensor, A: torch.Tensor):
    """f(h) = c_s * h_s + sum_t A[s, t] h_t — per-slot scale plus a CROSS-slot mix, so a
    valid slot's direction reaches the pads' outputs (pads must be masked out, not merely
    zero by accident)."""
    def step(h, e, inj, ret_state=None, iter_idx=None, stage_cond=None, carry=None):
        return h * c.view(*c.shape, 1, 1) + torch.einsum("st,btnc->bsnc", A, h), None
    return step


def _mask(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(B, S, generator=g) > 0.3
    m[:, 0] = True
    m[0, -2:] = False                     # at least two pads
    return m


def _synthetic(tail_lambda: float = 2.0, tail_target: float = 1.1, c_lo: float = 0.3,
               c_hi: float = 1.8, mix: float = 0.05, seed: int = 0, h=None, mask=None):
    m = _model(seed=3, slot_gain_lambda=1.0, slot_gain_target=0.9,
               slot_gain_tail_lambda=tail_lambda, slot_gain_tail_target=tail_target)
    # fp64 carrier: the pad-perturbation test adds 50x to pads that the cross-slot mix
    # carries into every output, and fp32 cancellation in f(h + d) - f(h) would read as a
    # change at 1e-5 that is rounding, not leakage.
    g = torch.Generator().manual_seed(seed + 1)
    c = c_lo + (c_hi - c_lo) * torch.rand(B, S, generator=g, dtype=torch.float64)
    A = mix * torch.randn(S, S, generator=g, dtype=torch.float64)
    h = torch.randn(B, S, N, C, generator=g, dtype=torch.float64) if h is None else h
    mask = _mask(seed) if mask is None else mask
    rec = _Recorder(_linear_step(c, A))
    torch.manual_seed(11)
    res = m._slot_gain_penalty(rec, h, None, None, None, 0, None, mask, 1.0)
    return m, res, rec, mask, h


def _independent(rec: _Recorder, mask: torch.Tensor, step=None):
    """g_s from the two recorded inputs, the core step re-applied by the test, fp64."""
    h0, h1 = rec.inputs
    step = step or rec.step
    f0 = step(h0, None, None)[0].double()
    f1 = step(h1, None, None)[0].double()
    d = (h1 - h0).double()
    g = (f1 - f0).flatten(2).norm(dim=2) / d.flatten(2).norm(dim=2).clamp_min(1e-300)
    return g, mask


def test_per_slot_gain_matches_an_independent_computation_on_a_cross_slot_map():
    _m, res, rec, mask, _h = _synthetic()
    g_ind, _ = _independent(rec, mask)
    g = res["g_slot"]
    assert torch.isnan(g[~mask]).all(), "a pad slot carried a per-slot gain"
    assert torch.allclose(g[mask].double(), g_ind[mask], rtol=1e-4), (g[mask], g_ind[mask])
    # The gains straddle 1, so the readings below are not vacuous.
    gv = g_ind[mask].numpy()
    assert (gv > 1.0).any() and (gv < 1.0).any() and (gv > 1.1).any()


def test_reducer_quantiles_max_and_fraction_match_numpy():
    _m, res, rec, mask, _h = _synthetic()
    red = MORPHTransformer._slot_gain_reduce([res])
    gv = res["g_slot"][mask].double().numpy()
    assert float(red["gain_slot_p50"]) == pytest.approx(np.percentile(gv, 50), rel=1e-5)
    assert float(red["gain_slot_p90"]) == pytest.approx(np.percentile(gv, 90), rel=1e-5)
    assert float(red["gain_slot_max"]) == pytest.approx(gv.max(), rel=1e-6)
    assert float(red["gain_slot_frac_gt1"]) == pytest.approx((gv > 1.0).mean(), abs=1e-7)
    assert 0.0 < float(red["gain_slot_frac_gt1"]) < 1.0


def test_reducer_pools_every_iteration_and_reads_nan_on_an_empty_sample():
    _m, r1, _rec, m1, _h = _synthetic(seed=0)
    _m, r2, _rec, m2, _h = _synthetic(seed=5)
    red = MORPHTransformer._slot_gain_reduce([r1, r2])
    gv = np.concatenate([r1["g_slot"][m1].double().numpy(), r2["g_slot"][m2].double().numpy()])
    assert float(red["gain_slot_p90"]) == pytest.approx(np.percentile(gv, 90), rel=1e-5)
    assert float(red["gain_slot_frac_gt1"]) == pytest.approx((gv > 1.0).mean(), abs=1e-7)
    assert float(red["gain_tail_pen"]) == pytest.approx(
        float(r1["tail_pen"]) + float(r2["tail_pen"]), rel=1e-6)
    empty = dict(r1, g_slot=torch.full_like(r1["g_slot"], float("nan")))
    red0 = MORPHTransformer._slot_gain_reduce([empty])
    for k in ("gain_slot_p50", "gain_slot_p90", "gain_slot_max", "gain_slot_frac_gt1"):
        assert torch.isnan(red0[k]), f"{k} reported a number for an empty sample"


def test_tail_term_is_lambda_times_mean_over_valid_slots_of_the_squared_excess():
    lam, tgt = 2.0, 1.1
    _m, res, rec, mask, _h = _synthetic(tail_lambda=lam, tail_target=tgt)
    g_ind, _ = _independent(rec, mask)
    exp_tail = lam * (torch.relu(g_ind[mask] - tgt) ** 2).mean()
    assert float(res["tail_pen"]) == pytest.approx(float(exp_tail), rel=1e-4)
    # The penalty is the row hinge (independent too) plus the tail.
    h0, h1 = rec.inputs
    f0 = rec.step(h0, None, None)[0].double()
    f1 = rec.step(h1, None, None)[0].double()
    mm = mask.view(B, S, 1, 1).double()
    row = ((f1 - f0) * mm).flatten(1).norm(dim=1) / (h1 - h0).double().flatten(1).norm(dim=1)
    exp_row = (torch.relu(row - 0.9) ** 2).mean()
    assert float(res["penalty"]) == pytest.approx(float(exp_row + exp_tail), rel=1e-4)


def test_perturbing_a_pad_slot_changes_nothing():
    mask = _mask(0)
    g = torch.Generator().manual_seed(1)
    h = torch.randn(B, S, N, C, generator=g, dtype=torch.float64)
    h2 = h.clone()
    h2[~mask] += 50.0 * torch.randn(int((~mask).sum()), N, C, generator=g, dtype=torch.float64)
    _m, a, _r, _mk, _h = _synthetic(h=h, mask=mask)
    _m, b, _r, _mk, _h = _synthetic(h=h2, mask=mask)
    assert torch.allclose(a["g_slot"][mask], b["g_slot"][mask], rtol=1e-5)
    assert float(a["tail_pen"]) == pytest.approx(float(b["tail_pen"]), rel=1e-5)
    assert float(a["penalty"]) == pytest.approx(float(b["penalty"]), rel=1e-5)


def test_tail_is_exactly_zero_when_no_slot_exceeds_its_target():
    kw = dict(c_lo=0.2, c_hi=0.9, mix=0.0)
    _m, on, rec, mask, _h = _synthetic(tail_lambda=5.0, **kw)
    assert float(on["g_slot"][mask].max()) <= 1.1
    _m, off, _r, _mk, _h = _synthetic(tail_lambda=0.0, **kw)
    assert float(on["tail_pen"]) == 0.0
    assert torch.equal(on["penalty"], off["penalty"])


# ── the real tiny model ─────────────────────────────────────────────────────────────────

def _spy(m: MORPHTransformer):
    """Wrap `_slot_gain_penalty` to record, per call, its arguments, the core step's two
    inputs and its returned dict."""
    calls = []
    orig = m._slot_gain_penalty

    def spy(core_step, h_in, e_arg, inj_arg, ret_state, t, stage_cond, mask, lam, carry=None):
        rec = _Recorder(core_step)
        res = orig(rec, h_in, e_arg, inj_arg, ret_state, t, stage_cond, mask, lam, carry=carry)
        calls.append(dict(rec=rec, e=e_arg, inj=inj_arg, ret=ret_state, t=t, sc=stage_cond,
                          mask=mask, carry=carry, res=res))
        return res
    m._slot_gain_penalty = spy
    return calls


def test_per_slot_gain_on_the_real_model_matches_the_core_step_reapplied():
    m = _model(seed=3, dropout=0.0, slot_gain_lambda=1.0, slot_gain_target=0.0)
    calls = _spy(m)
    out, _g, _rng = _run(m)
    assert len(calls) == 1
    c = calls[0]
    h0, h1 = c["rec"].inputs

    def step(h):
        with torch.no_grad():
            return c["rec"].step(h, c["e"], c["inj"], ret_state=c["ret"], iter_idx=c["t"],
                                 stage_cond=c["sc"], carry=c["carry"])[0].double()
    f0, f1 = step(h0), step(h1)
    mask = c["mask"]
    g_ind = (f1 - f0).flatten(2).norm(dim=2) / (h1 - h0).double().flatten(2).norm(dim=2)
    g = c["res"]["g_slot"]
    assert mask.any() and (~mask).any()
    assert torch.allclose(g[mask].double(), g_ind[mask], rtol=1e-3), (g[mask], g_ind[mask])
    assert torch.isnan(g[~mask]).all()
    gv = g[mask].double().numpy()
    assert float(out["gain_slot_p90"]) == pytest.approx(np.percentile(gv, 90), rel=1e-5)
    assert float(out["gain_slot_frac_gt1"]) == pytest.approx((gv > 1.0).mean(), abs=1e-7)
    assert float(out["gain_slot_max"]) == pytest.approx(gv.max(), rel=1e-6)


def test_tail_gradient_reaches_the_core_and_nothing_upstream():
    # The row hinge never binds on both; the only difference is the tail (target 0: binds).
    base = dict(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6)
    out_a, g_a, rng_a = _run(_model(**base))
    out_b, g_b, rng_b = _run(_model(**base, slot_gain_tail_lambda=1.0,
                                    slot_gain_tail_target=1e-3))
    assert torch.equal(rng_a, rng_b), "the tail moved the global RNG stream"
    assert float(out_b["gain_tail_pen"]) > 0.0
    assert float((out_b["loss"] - out_a["loss"]).detach()) == pytest.approx(
        float(out_b["gain_tail_pen"]), rel=1e-4)
    core = [n for n in g_a if n.startswith("core.")]
    upstream = [n for n in g_a if n.startswith(("prelude.", "embed", "tok_emb", "input_norm"))]
    assert core and upstream
    assert not _same(g_a, g_b, core), "the tail left the core's gradients untouched"
    assert _same(g_a, g_b, upstream), "the tail leaked upstream of the loop"


def test_all_iters_every_iteration_contributes_its_tail_and_its_slots():
    lam = 0.5
    m = _model(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6, slot_gain_all_iters=True,
               slot_gain_tail_lambda=lam, slot_gain_tail_target=1e-3)
    calls = _spy(m)
    out, _g, rng = _run(m)
    assert len(calls) == MAX_DEPTH and float(out["gain_n_iters"]) == MAX_DEPTH
    for c in calls:
        gv = c["res"]["g_slot"][c["mask"]].double()
        exp = lam * (torch.relu(gv - 1e-3) ** 2).mean()
        assert float(c["res"]["tail_pen"]) == pytest.approx(float(exp), rel=1e-5)
    assert float(out["gain_tail_pen"]) == pytest.approx(
        sum(float(c["res"]["tail_pen"]) for c in calls), rel=1e-6)
    assert float(out["gain_reg_weighted"]) == pytest.approx(float(out["gain_tail_pen"]),
                                                            rel=1e-6)
    gv = np.concatenate([c["res"]["g_slot"][c["mask"]].double().numpy() for c in calls])
    assert float(out["gain_slot_p50"]) == pytest.approx(np.percentile(gv, 50), rel=1e-5)
    _o0, _g0, rng0 = _run(_model(seed=3, slot_gain_lambda=0.0))
    assert torch.equal(rng, rng0), "the every-iteration tail moved the global RNG stream"


@pytest.mark.parametrize("kw,match", [
    (dict(slot_gain_lambda=1.0, slot_gain_tail_lambda=-1.0), "slot_gain_tail_lambda must be >= 0"),
    (dict(slot_gain_lambda=1.0, slot_gain_tail_target=0.0), "slot_gain_tail_target"),
    (dict(slot_gain_lambda=1.0, slot_gain_tail_target=-0.5), "slot_gain_tail_target"),
    (dict(slot_gain_lambda=0.0, slot_gain_tail_lambda=1.0), "needs model.slot_gain_lambda > 0"),
])
def test_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        _model(**kw)


# ── the Stage 3 configs ─────────────────────────────────────────────────────────────────

_MAP = {"slot_gain_target": 0.98, "slot_gain_tail_lambda": 100.0, "slot_gain_tail_target": 1.1}
_MISSING = object()


def _leaves(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_leaves(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


@pytest.mark.parametrize("name,parent,section,keys,wb,parent_wb", [
    ("tul_slot_spandec_strict_e4probe_map", "tul_slot_spandec_strict_e4probe", "model", _MAP,
     "lxtul-e4probe-map", "lxtul-e4probe"),
    ("tul_slot_spandec_strict_e1probe", "tul_slot_spandec_strict_e1", "tul",
     {"spandec_parallel_detach": True}, "lxtul-e1probe", "lxtul-e1"),
    ("tul_slot_spandec_strict_e1probe_map", "tul_slot_spandec_strict_e1probe", "model", _MAP,
     "lxtul-e1probe-map", "lxtul-e1probe"),
])
def test_the_stage3_configs_compose_build_train_and_differ_from_parents_by_the_stated_keys(
        name, parent, section, keys, wb, parent_wb, monkeypatch):
    from omegaconf import OmegaConf
    from test_tul_lxtul_e import _D_FF
    from test_tul_strict_geometry import _pack, _runtime, _tiny

    from morph.training.train import build_morph_config

    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    assert c["wandb.name"] == wb and p["wandb.name"] == parent_wb
    for k, v in keys.items():
        assert c[f"{section}.{k}"] == v, (k, c[f"{section}.{k}"])
    # Every resolved leaf that differs, and nothing else. `slot_gain_tail_target: 1.1` is
    # stated in the map arms but equals base.yaml's default, so it is not a difference.
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    stated = {f"{section}.{k}" for k, v in keys.items()
              if p.get(f"{section}.{k}", _MISSING) != v} | {"wandb.name"}
    assert diff == stated, f"{name} vs {parent}: differs in {sorted(diff)}, stated {sorted(stated)}"

    # The parse reaches MORPHConfig (train.py's getattr defaults would hide a missed key).
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    gain_kw = {k: getattr(mc, k) for k in ("slot_gain_lambda", "slot_gain_target",
                                           "slot_gain_eps", "slot_gain_all_iters",
                                           "slot_gain_tail_lambda", "slot_gain_tail_target",
                                           "slot_cot_clip", "core_fixed_point_lambda")}
    exp_map = _MAP if section == "model" else {"slot_gain_target": 0.9,
                                                "slot_gain_tail_lambda": 0.0}
    for k, v in exp_map.items():
        assert gain_kw[k] == v, (k, gain_kw[k])
    assert gain_kw["slot_gain_lambda"] == 100.0 and gain_kw["slot_cot_clip"] == 4.0
    assert rt.model_cfg.spandec_parallel_detach is True

    # Build at tiny width with the arm's TUL block and gain knobs, and train one step.
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=_D_FF, **gain_kw)).train().float()
    _ids, inp, lab, layout = _pack()
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    for k in _NEW_KEYS:
        assert k in out and torch.isfinite(out[k]), k
