"""The FLOOR on the slot map's row gain (`model.slot_gain_floor_lambda`,
`model.slot_gain_floor_target`; LXTUL Stage 3 arm B, 2026-09-24).

Files: morph/model/transformer.py (`MORPHConfig.slot_gain_floor_*`, `_slot_gain_penalty`,
`_slot_gain_reduce`, the `gain_floor_pen` group), morph/training/train.py (the parse and
the logged key), morph/configs/tul_slot_spandec_strict_e4probe_{nofp,floor}.yaml.

What each test pins:
  * floor lambda 0 is the function without the floor: loss, every gradient and the RNG are
    bit-identical at a floor target every row sits below (the pre-tail, pre-floor code is
    pinned by tests/test_slot_gain_tail.py, whose bit-identity test reads this key too);
  * the floor term is lam * mean over rows WITH an active slot of relu(floor - g_row)^2,
    against an independent computation of the row gain; a row with no active slot adds
    nothing; the term is exactly zero when every row sits at or above the floor;
  * its gradient RAISES the gain (a step against it lifts every row below the floor) and
    reaches the core and nothing upstream of the loop;
  * with `slot_gain_all_iters` every iteration contributes its own floor;
  * the refusals hold;
  * the two Stage 3 arms compose, build and train one step, and differ from
    `tul_slot_spandec_strict_e4probe_map` by exactly the stated keys and the run name.
"""
from __future__ import annotations

import pytest
import torch

from test_slot_gain_reg import MAX_DEPTH, _model, _run, _same
from test_slot_gain_tail import B, C, N, S, _leaves, _linear_step, _mask, _MISSING, _Recorder, _spy

from morph.model.transformer import MORPHTransformer


def _floor_model(floor_lambda: float, floor_target: float, target: float = 1e6):
    return _model(seed=3, slot_gain_lambda=1.0, slot_gain_target=target,
                  slot_gain_floor_lambda=floor_lambda, slot_gain_floor_target=floor_target)


def _synthetic(floor_lambda: float, floor_target: float, c=None, mask=None, seed: int = 0):
    """The penalty on a synthetic per-slot linear map over a 4-D fp64 carrier. The row
    hinge never binds (target 1e6), so the penalty is the floor alone."""
    m = _floor_model(floor_lambda, floor_target)
    g = torch.Generator().manual_seed(seed + 1)
    if c is None:
        c = 0.3 + 0.9 * torch.rand(B, S, generator=g, dtype=torch.float64)
    A = 0.05 * torch.randn(S, S, generator=g, dtype=torch.float64)
    h = torch.randn(B, S, N, C, generator=g, dtype=torch.float64)
    mask = _mask(seed) if mask is None else mask
    rec = _Recorder(_linear_step(c, A))
    torch.manual_seed(11)
    res = m._slot_gain_penalty(rec, h, None, None, None, 0, None, mask, 1.0)
    return res, rec, mask


def _row_gain(rec: _Recorder, mask: torch.Tensor) -> torch.Tensor:
    """The row gain from the two recorded inputs, the core step re-applied by the test."""
    h0, h1 = rec.inputs
    f0 = rec.step(h0, None, None)[0].detach().double()
    f1 = rec.step(h1, None, None)[0].detach().double()
    mm = mask.view(B, S, 1, 1).double()
    return ((f1 - f0) * mm).flatten(1).norm(dim=1) / (h1 - h0).double().flatten(1).norm(dim=1)


def test_floor_off_is_bit_identical_at_a_floor_every_row_sits_below():
    # Row hinge at 1e6 never binds; a floor of 1e5 would bind on every row if it leaked.
    out_a, g_a, rng_a = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6))
    out_b, g_b, rng_b = _run(_floor_model(0.0, 1e5))
    assert torch.equal(rng_a, rng_b)
    for k in ("loss", "gain_est", "gain_est_max", "gain_reg_weighted", "mux_local"):
        assert torch.equal(out_a[k], out_b[k]), k
    assert _same(g_a, g_b)
    assert float(out_b["gain_floor_pen"]) == 0.0


def test_floor_term_matches_an_independent_computation_with_rows_on_both_sides():
    _r, rec, mask = _synthetic(0.0, 0.5)
    g_row = _row_gain(rec, mask)
    floor = float(g_row.median())            # one row below, one above, one at the floor
    assert (g_row < floor).any() and (g_row > floor).any()
    lam = 3.0
    res, rec, mask = _synthetic(lam, floor)
    g_row = _row_gain(rec, mask)
    exp = lam * (torch.relu(floor - g_row) ** 2).mean()
    assert float(res["floor_pen"]) > 0.0
    assert float(res["floor_pen"]) == pytest.approx(float(exp), rel=1e-4)
    assert float(res["penalty"]) == pytest.approx(float(exp), rel=1e-4)
    red = MORPHTransformer._slot_gain_reduce([res])
    assert torch.equal(red["gain_floor_pen"], res["floor_pen"])


def test_a_row_with_no_active_slot_adds_nothing_to_the_floor():
    mask = _mask(0)
    mask[1] = False                           # row 1's depth ran out at this pass
    lam, floor = 2.0, 5.0                     # every live row sits below the floor
    res, rec, mask = _synthetic(lam, floor, mask=mask)
    g_row = _row_gain(rec, mask)
    live = mask.any(dim=1)
    assert bool(torch.isnan(g_row[1])) and live.sum() == B - 1   # 0/0: no d, no f-difference
    exp = lam * (torch.relu(floor - g_row[live]) ** 2).mean()
    assert float(res["floor_pen"]) == pytest.approx(float(exp), rel=1e-4)


def test_floor_is_exactly_zero_when_every_row_sits_at_or_above_it():
    res, _rec, _m = _synthetic(5.0, 1e-3)
    off, _rec, _m = _synthetic(0.0, 1e-3)
    assert float(res["floor_pen"]) == 0.0
    assert torch.equal(res["penalty"], off["penalty"])


def test_a_step_against_the_floor_raises_every_row_below_it():
    g = torch.Generator().manual_seed(1)
    c0 = 0.3 + 0.9 * torch.rand(B, S, generator=g, dtype=torch.float64)
    _r, rec, mask = _synthetic(0.0, 0.5, c=c0)
    g0 = _row_gain(rec, mask)
    floor = float(g0.max()) + 0.1             # every row below
    c = c0.clone().requires_grad_(True)
    res, _rec, _m = _synthetic(1.0, floor, c=c)
    res["penalty"].backward()
    assert c.grad is not None and float(c.grad[mask].max()) <= 0.0
    assert float(c.grad[mask].min()) < 0.0
    _r, rec1, _m = _synthetic(0.0, 0.5, c=(c0 - 0.05 * c.grad).detach())
    g1 = _row_gain(rec1, mask)
    assert (g1 > g0).all(), (g0, g1)


def test_floor_gradient_reaches_the_core_and_nothing_upstream():
    out_a, g_a, rng_a = _run(_model(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6))
    out_b, g_b, rng_b = _run(_floor_model(1.0, 1e5))
    assert torch.equal(rng_a, rng_b), "the floor moved the global RNG stream"
    assert float(out_b["gain_floor_pen"]) > 0.0
    assert float((out_b["loss"] - out_a["loss"]).detach()) == pytest.approx(
        float(out_b["gain_floor_pen"]), rel=1e-4)
    core = [n for n in g_a if n.startswith("core.")]
    upstream = [n for n in g_a if n.startswith(("prelude.", "embed", "tok_emb", "input_norm"))]
    assert core and upstream
    assert not _same(g_a, g_b, core), "the floor left the core's gradients untouched"
    assert _same(g_a, g_b, upstream), "the floor leaked upstream of the loop"


def test_all_iters_every_iteration_contributes_its_floor():
    lam, floor = 0.5, 1e5
    m = _model(seed=3, slot_gain_lambda=1.0, slot_gain_target=1e6, slot_gain_all_iters=True,
               slot_gain_floor_lambda=lam, slot_gain_floor_target=floor)
    calls = _spy(m)
    out, _g, rng = _run(m)
    assert len(calls) == MAX_DEPTH and float(out["gain_n_iters"]) == MAX_DEPTH
    for c in calls:
        live = c["mask"].any(dim=1)
        assert live.any()
        # The row gain of each call, read back from its per-slot readings' pooled RMS is
        # not exact (d-weighted), so the call's own row gain comes from its recorded inputs.
        h0, h1 = c["rec"].inputs
        with torch.no_grad():
            kw = dict(ret_state=c["ret"], iter_idx=c["t"], stage_cond=c["sc"], carry=c["carry"])
            f0 = c["rec"].step(h0, c["e"], c["inj"], **kw)[0].double()
            f1 = c["rec"].step(h1, c["e"], c["inj"], **kw)[0].double()
        mm = c["mask"].view(*c["mask"].shape, *([1] * (h0.dim() - 2))).double()
        g_row = ((f1 - f0) * mm).flatten(1).norm(dim=1) / (h1 - h0).double().flatten(1).norm(dim=1)
        exp = lam * (torch.relu(floor - g_row[live]) ** 2).mean()
        assert float(c["res"]["floor_pen"]) == pytest.approx(float(exp), rel=1e-4)
    total = sum(float(c["res"]["floor_pen"]) for c in calls)
    assert float(out["gain_floor_pen"]) == pytest.approx(total, rel=1e-6)
    assert float(out["gain_reg_weighted"]) == pytest.approx(total, rel=1e-6)
    _o0, _g0, rng0 = _run(_model(seed=3, slot_gain_lambda=0.0))
    assert torch.equal(rng, rng0), "the every-iteration floor moved the global RNG stream"


@pytest.mark.parametrize("kw,match", [
    (dict(slot_gain_lambda=1.0, slot_gain_floor_lambda=-1.0), "slot_gain_floor_lambda must be >= 0"),
    (dict(slot_gain_lambda=1.0, slot_gain_floor_target=0.0), "slot_gain_floor_target"),
    (dict(slot_gain_lambda=0.0, slot_gain_floor_lambda=1.0), "needs model.slot_gain_lambda > 0"),
    (dict(slot_gain_lambda=1.0, slot_gain_target=0.9, slot_gain_floor_lambda=1.0,
          slot_gain_floor_target=0.9), "must be below"),
    (dict(slot_gain_lambda=1.0, slot_gain_target=0.9, slot_gain_floor_lambda=1.0,
          slot_gain_floor_target=0.95), "must be below"),
])
def test_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        _model(**kw)


# ── the Stage 3 configs ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,keys,wb", [
    ("tul_slot_spandec_strict_e4probe_nofp", {"core_fixed_point_lambda": 0.0},
     "lxtul-e4probe-nofp"),
    ("tul_slot_spandec_strict_e4probe_floor",
     {"slot_gain_floor_lambda": 100.0, "slot_gain_floor_target": 0.95}, "lxtul-e4probe-floor"),
])
def test_the_arms_compose_build_train_and_differ_from_the_map_arm_by_the_stated_keys(
        name, keys, wb, monkeypatch):
    from omegaconf import OmegaConf
    from test_tul_lxtul_e import _D_FF
    from test_tul_strict_geometry import _pack, _runtime, _tiny

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_e4probe_map"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    assert c["wandb.name"] == wb and p["wandb.name"] == "lxtul-e4probe-map"
    for k, v in keys.items():
        assert c[f"model.{k}"] == v, (k, c[f"model.{k}"])
    # `slot_gain_floor_target: 0.95` equals base.yaml's default, so it is not a difference.
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    stated = {f"model.{k}" for k, v in keys.items()
              if p.get(f"model.{k}", _MISSING) != v} | {"wandb.name"}
    assert diff == stated, f"{name} vs {parent}: differs in {sorted(diff)}, stated {sorted(stated)}"

    mc = build_morph_config(cfg, tul=rt.model_cfg)
    gain_kw = {k: getattr(mc, k) for k in ("slot_gain_lambda", "slot_gain_target",
                                           "slot_gain_eps", "slot_gain_all_iters",
                                           "slot_gain_tail_lambda", "slot_gain_tail_target",
                                           "slot_gain_floor_lambda", "slot_gain_floor_target",
                                           "slot_cot_clip", "core_fixed_point_lambda")}
    for k, v in keys.items():
        assert gain_kw[k] == v, (k, gain_kw[k])
    assert gain_kw["slot_gain_target"] == 0.98 and gain_kw["slot_gain_tail_lambda"] == 100.0
    assert gain_kw["core_fixed_point_lambda"] == keys.get("core_fixed_point_lambda", 1.0)
    assert gain_kw["slot_gain_floor_lambda"] == keys.get("slot_gain_floor_lambda", 0.0)

    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=_D_FF, **gain_kw)).train().float()
    _ids, inp, lab, layout = _pack()
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["gain_floor_pen"])
    if gain_kw["slot_gain_floor_lambda"] > 0.0:
        assert float(out["gain_floor_pen"]) >= 0.0
    else:
        assert float(out["gain_floor_pen"]) == 0.0


@pytest.mark.parametrize("name,section,keys,wb", [
    ("tul_slot_spandec_strict_e4probe_nofp_s2", "training", {"seed": 2}, "lxtul-e4probe-nofp-s2"),
    ("tul_slot_spandec_strict_e4probe_fp01", "model", {"core_fixed_point_lambda": 0.1},
     "lxtul-e4probe-fp01"),
])
def test_the_nofp_followups_differ_from_nofp_by_the_stated_keys(name, section, keys, wb,
                                                               monkeypatch):
    from omegaconf import OmegaConf
    from test_tul_lxtul_e import _D_FF
    from test_tul_strict_geometry import _pack, _runtime, _tiny

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_e4probe_nofp"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    assert c["wandb.name"] == wb and p["wandb.name"] == "lxtul-e4probe-nofp"
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    assert diff == {f"{section}.{k}" for k in keys} | {"wandb.name"}, sorted(diff)
    assert p["training.seed"] == 1 and p["model.core_fixed_point_lambda"] == 0.0
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    assert mc.core_fixed_point_lambda == keys.get("core_fixed_point_lambda", 0.0)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=_D_FF,
                               core_fixed_point_lambda=mc.core_fixed_point_lambda)).train().float()
    _ids, inp, lab, layout = _pack()
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
