"""`tul.slot_source_once` (map-cause intervention I-2, 2026-09-24): the slot loop's per-pass
source enters at pass 0 only.

Files: morph/model/tul.py (`TULConfig.slot_source_once`, `_check_source_once`),
morph/model/transformer.py (`DiagonalInjection.decay`, `_apply_core_step`'s
`source_decay_only`, `_tul_core`'s `_core_step` and the LXTUL-E code re-add),
morph/training/tul_setup.py (the key), morph/configs/tul_slot_spandec_strict_e4probe_{once,
inj9}.yaml (I-2 and I-1).

What each test pins:
  * `DiagonalInjection.decay(h)` is `forward(h, 0)`: the decay kept, the source gone;
  * on the real loop pass 0 takes the source and every later pass does not, at eval and in
    training, where the gain hinge re-runs the step at every pass and the checkpoint
    recomputes it: every call carries `source_decay_only == (iter_idx >= 1)`;
  * a pass t >= 1 does not read `e` at all, and it equals the decay followed by the core
    blocks with NO per-layer x0/bigram terms (the `source_free` blocks on `decay(h)`);
    pass 0 does read `e`;
  * the LXTUL-E code is added after pass 0 only (one call to the term at any depth), and
    the rollouts still differ from pass 2 on (the code lives in the state);
  * off, the per-pass calls are the ones from before the key (all `source_decay_only`
    False, the code once per pass);
  * the I-1 and I-2 configs compose, build and train one step, and differ from
    `tul_slot_spandec_strict_e4probe_map` by exactly the stated keys and the run name;
  * the refusals hold.
"""
from __future__ import annotations

import pytest
import torch

from test_slot_gain_tail import _leaves, _MISSING
from test_tul_lxtul_e import _D_FF, K, _Spy, _table, _tc
from test_tul_strict_geometry import _pack, _tiny

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig


def _model(once: bool, seed: int = 1234, **mkw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tc(K, slot_source_once=once), d_ff=_D_FF, **mkw))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.float()


def test_decay_is_forward_with_a_zero_source():
    torch.manual_seed(0)
    m = _model(True)
    with torch.no_grad():
        m.injection.log_A.uniform_(-1.5, -0.1)
        m.injection.log_dt.uniform_(-0.5, 0.5)
    h = torch.randn(2, 5, 4, 64)
    assert torch.equal(m.injection.decay(h), m.injection(h, torch.zeros_like(h)))
    s, e_ = m.injection.start, m.injection.end
    out = m.injection.decay(h)
    assert torch.equal(out[..., :s], h[..., :s]) and torch.equal(out[..., e_:], h[..., e_:])
    assert not torch.equal(out[..., s:e_], h[..., s:e_])


@pytest.mark.parametrize("once", [False, True])
def test_every_pass_at_eval_takes_the_source_only_at_pass_0(once):
    _ids, inp, lab, layout = _pack()
    m = _model(once).eval()
    spy = _Spy(m, "_apply_core_step")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, 3))
    got = [(k["iter_idx"], k.get("source_decay_only", False)) for _a, k in spy.calls]
    assert got == [(0, False), (1, once), (2, once)], got


def test_training_every_call_including_the_hinge_and_recompute_follows_the_rule():
    _ids, inp, lab, layout = _pack()
    m = _model(True, slot_gain_lambda=1.0, slot_gain_target=0.5,
               slot_gain_all_iters=True).train()
    spy = _Spy(m, "_apply_core_step")
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert len(spy.calls) > 3                   # the loop's passes and the hinge's re-runs
    ts = {k["iter_idx"] for _a, k in spy.calls}
    assert 0 in ts and max(ts) >= 1
    for _a, k in spy.calls:
        assert k.get("source_decay_only", False) == (int(k["iter_idx"]) >= 1), k["iter_idx"]


def test_a_later_pass_ignores_e_and_is_decay_then_bare_blocks():
    _ids, inp, lab, layout = _pack()
    m = _model(True).eval()
    spy = _Spy(m, "_apply_core_step")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, 3))
        by_t = {k["iter_idx"]: (a, k, o) for (a, k), o in zip(spy.calls, spy.outs)}
        a, k, o = by_t[2]
        h_in, e_in = a[0], a[1]
        assert k["source_decay_only"] is True
        # The e the loop passed is not read: a different e gives the same output.
        o_e = m._apply_core_step(h_in, e_in + 3.0 * torch.randn_like(e_in), *a[2:], **k)
        assert torch.equal(o[0], o_e[0])
        # Decay, then the blocks with no source (source_free skips the injection and
        # every per-layer term).
        k_free = {**k, "source_decay_only": False, "source_free": True, "inj_terms": None}
        o_ref = m._apply_core_step(m.injection.decay(h_in), e_in, *a[2:], **k_free)
        assert torch.equal(o[0], o_ref[0])
        # Pass 0 reads e.
        a0, k0, o0 = by_t[0]
        assert k0["source_decay_only"] is False
        o0_e = m._apply_core_step(a0[0], a0[1] + 3.0 * torch.randn_like(a0[1]), *a0[2:], **k0)
        assert not torch.equal(o0[0], o0_e[0])


def test_the_code_is_added_after_pass_0_only_and_rollouts_still_differ():
    _ids, inp, lab, layout = _pack()
    for once, calls in ((False, 3), (True, 1)):
        m = _model(once).eval()
        m._jac_capture = []
        spy = _Spy(m.tul_code_enum, "term")
        with torch.no_grad():
            m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, 3))
        assert len(spy.calls) == calls, (once, len(spy.calls))
        B0, v = layout.slot_valid.shape[0], layout.slot_valid
        for t, cap in enumerate(m._jac_capture):
            diff = (cap["h"][:B0] - cap["h"][B0:2 * B0])[v].abs().max()
            assert (diff == 0) if t == 0 else (diff > 1e-4), (once, t, float(diff))


def test_the_change_reaches_the_exit():
    _ids, inp, lab, layout = _pack()
    outs = []
    for once in (False, True):
        m = _model(once).eval()
        with torch.no_grad():
            outs.append(m(inp, labels=lab, slot_layout=layout,
                          slot_depths=_table(layout, 3))["loss"])
    assert not torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("kw,match", [
    (dict(tokens_through_core=True), "tokens_through_core"),
    (dict(fan_trigger_every_pass=True, fan_k=2), "fan_trigger_every_pass|fan"),
    (dict(reread=True), "reread"),
])
def test_refusals(kw, match):
    with pytest.raises((NotImplementedError, ValueError), match=match):
        TULConfig(slot_source_once=True, **kw)


@pytest.mark.parametrize("name,section,keys,wb", [
    ("tul_slot_spandec_strict_e4probe_once", "tul", {"slot_source_once": True},
     "lxtul-e4probe-once"),
    ("tul_slot_spandec_strict_e4probe_inj9", "model",
     {"injection_channels": "all", "injection_all_decay": 0.9}, "lxtul-e4probe-inj9"),
])
def test_the_arms_compose_build_train_and_differ_from_the_map_arm_by_the_stated_keys(
        name, section, keys, wb, monkeypatch):
    from omegaconf import OmegaConf
    from test_tul_strict_geometry import _runtime

    from morph.training.train import build_morph_config

    parent = "tul_slot_spandec_strict_e4probe_map"
    cfg, rt = _runtime(name, monkeypatch)
    pcfg, _prt = _runtime(parent, monkeypatch)
    c = _leaves(OmegaConf.to_container(cfg, resolve=True))
    p = _leaves(OmegaConf.to_container(pcfg, resolve=True))
    assert c["wandb.name"] == wb and p["wandb.name"] == "lxtul-e4probe-map"
    # `injection_all_decay: 0.9` equals base.yaml's default, so it is not a difference.
    diff = {k for k in c.keys() | p.keys() if c.get(k, _MISSING) != p.get(k, _MISSING)}
    stated = {f"{section}.{k}" for k, v in keys.items()
              if p.get(f"{section}.{k}", _MISSING) != v} | {"wandb.name"}
    assert diff == stated, f"{name} vs {parent}: differs in {sorted(diff)}, stated {sorted(stated)}"

    mc = build_morph_config(cfg, tul=rt.model_cfg)
    assert rt.model_cfg.slot_source_once is (name.endswith("_once"))
    assert mc.injection_channels == ("all" if name.endswith("_inj9") else "ctx")
    assert mc.slot_gain_target == 0.98 and mc.slot_gain_tail_lambda == 100.0

    mkw = {k: getattr(mc, k) for k in ("injection_channels", "injection_all_decay",
                                       "slot_gain_lambda", "slot_gain_target",
                                       "slot_gain_tail_lambda", "slot_gain_tail_target",
                                       "slot_cot_clip", "core_fixed_point_lambda")}
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=rt.model_cfg, d_ff=_D_FF, **mkw)).train().float()
    if name.endswith("_inj9"):
        A = m.injection.log_A.exp()
        s, e_ = m._ctx_start, m._ctx_end
        assert torch.allclose(A[s:e_], torch.full_like(A[s:e_], 0.447))
        rest = torch.cat([A[:s], A[e_:]])
        assert torch.allclose(rest, torch.full_like(rest, 0.9))
    _ids, inp, lab, layout = _pack()
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
