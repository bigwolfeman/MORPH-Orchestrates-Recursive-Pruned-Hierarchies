"""model.hc_carrier_bf16: the Hyper-Connection carrier stored in bf16.

What each test pins (2026-10-12):
  1. KERNELS. ``hc_region_entry`` / ``hc_region_exit`` on a bf16 carrier (loads widen to fp32,
     all mixing math fp32, only the store rounds) against the fp64 reference computed from the
     SAME bf16 values: output, carrier grad, projection weight and bias grad, sublayer weight
     grad and folded inject-term grad. Tolerance, from bf16 eps: a bf16 store has relative
     rounding error uniform in +-2**-9, RMS 2**-9 / sqrt(3) = 1.1e-3, and the region path adds
     the projection's own bf16 rounding (the same autocast convention the fp32 carrier
     already carries). Each error is held to the fp32-carrier path's error on the same inputs
     plus ``STORE_TOL`` = 2.0e-3, 1.8 RMS bf16 roundings. Measured excess over the fp32
     carrier: 0.5e-3 (out), 0.7-1.1e-3 (grads), 1.66e-3 (the inject term grad: a sum of four bf16-rounded gout streams). A kernel that rounds anywhere else (Hres, a
     partial sum) adds more and breaks it (sabotage b).
  1b. A bf16 carrier is the fp32 carrier holding the same values, rounded once at the store:
     x_bar, the exit output and the carrier grad equal the fp32 run's rounded to bf16 up to
     the rare last-bit flip of two compiles (< 1e-3 of the elements, <= 1.5 bf16 ulp); Hres and
     Hpost_row equal it to 2e-5 absolute (fp32 roundoff is 2e-7, a bf16 rounding of Hres 1e-3).
  2. DTYPES. Out and carrier grad are bf16; Hres and Hpost_row stay fp32 (the 2026-10-06
     audit: bf16 Hres is slightly expansive); the 48 x 4C projection weight grad is fp32.
  3. REFUSALS. No fallback: without bf16 autocast, on CPU, with ``use_kernel=False`` or
     without ``region_fused`` the key raises.
  4. THE MODEL. A tiny recipe model (``nowta_pointer_fast_spdcells_speed`` with the region
     keys): with the key on every Hyper-Connection residual of the main loop, the prelude and
     the coda reads a bf16 carrier, and the two applications of the gain hinge read an fp32
     one; with the key off every carrier is fp32. Step-0 loss agrees with the key-off model to
     1e-2 and the hinge's logged gain to 5e-3. Measured (tiny d=64 and d=256, 3 seeds each,
     scratch gain_probe.py): key on vs off differ by 0.0002-0.0032 (the operating point is a
     bf16-rounded state; seeds differ by 0.007 among themselves), a hinge on the bf16 carrier
     reads +0.039 to +0.046 and moves the loss by +0.02 to +0.09 (the trap of 2026-10-08).

  5. ENTRY POINTS (2026-10-12). `MORPHTransformer.forward` and every other public entry that
     runs the network is wrapped by `_carrier_autocast`: key on and autocast off, it enters
     bf16 autocast itself (same function as the training forward); without CUDA it raises;
     key off it is untouched. `generate_tul` (eager) emits tokens on the key-on fan model;
     the KV-cached decoders refuse the key. Sabotage (c): wrapper made a pass-through ->
     the no-autocast and the generate tests fail (the HC gate raises).

Sabotage checks, run and reverted (session report, not committed):
  (a) ``_slot_gain_penalty`` without the ``hp = hp.float()`` widening -> test 4 fails (the
      hinge applications read a bf16 carrier).
  (b) ``hc_region_entry`` rounding the returned Hres to bf16 on a bf16 carrier -> test 1b fails
      (30 % of the output elements differ); test 1 alone does NOT catch it (its tolerance is
      the bf16 store).
"""
import dataclasses
import os

import pytest
import torch

from morph.kernels.triton import fused_hyper_connection as K
from morph.model.hyper_connections import HyperConnectionResidual

C = 1024
STORE_TOL = 2.0e-3
ATOL = 1.0e-5
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _hc(**kw):
    torch.manual_seed(1)
    hc = HyperConnectionResidual(C, n_streams=4, use_kernel=True, region_fused=True, **kw).cuda()
    with torch.no_grad():                       # a mapping far from the identity init
        hc.proj.weight.normal_(0, 3.0 / (4 * C) ** 0.5)
        hc.proj.bias.normal_(0, 0.5)
    return hc


def _data(B, S, with_term):
    g = torch.Generator(device="cuda").manual_seed(0)
    h = (torch.randn(B, S, 4, C, device="cuda", generator=g) * 2).bfloat16()   # bf16 values
    wf = torch.randn(C, C, device="cuda", generator=g) / C ** 0.5
    r = torch.randn(B, S, 4, C, device="cuda", generator=g)
    term = torch.randn(B, S, C, device="cuda", generator=g) if with_term else None
    return h, wf, r, term


def _err(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def _run(hc, h0, wf0, r, term0, dtype):
    h = h0.to(dtype).detach().clone().requires_grad_(True)
    wf = wf0.detach().clone().requires_grad_(True)
    term = None if term0 is None else term0.detach().clone().requires_grad_(True)
    hc.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = hc(h, lambda x: x @ wf.to(torch.bfloat16), post_inject=term)
    (out.float() * r).sum().backward()
    gs = [h.grad, hc.proj.weight.grad, hc.proj.bias.grad, wf.grad]
    if term is not None:
        gs.append(term.grad)
    return out.detach(), gs


def _run64(hc, h0, wf0, r, term0):
    h = h0.detach().double().requires_grad_(True)
    w = hc.proj.weight.detach().double().requires_grad_(True)
    b = hc.proj.bias.detach().double().requires_grad_(True)
    wf = wf0.detach().double().requires_grad_(True)
    term = None if term0 is None else term0.detach().double().requires_grad_(True)
    xb, hres, hpr = K.hc_pre_map_reference(h, w, b, hc.tau, hc.cayley_alpha, 3, hc.eps)
    out = K.hc_post_reference(hres, hpr, h, xb @ wf, term)
    (out * r.double()).sum().backward()
    gs = [h.grad, w.grad, b.grad, wf.grad]
    if term is not None:
        gs.append(term.grad)
    return out.detach(), gs


CASES = [
    (6, 1280, False),   # prelude / coda block: 7680 positions
    (6, 256, False),    # core pass: 1536 positions
    (3, 37, True),      # a partial last tile + the folded inject term
]
NAMES = ["grad_h", "grad_proj_w", "grad_proj_b", "grad_sublayer_w", "grad_term"]


@cuda
@pytest.mark.parametrize("B,S,with_term", CASES)
def test_bf16_carrier_kernels_vs_fp64(B, S, with_term):
    h, wf, r, term = _data(B, S, with_term)
    hc_off, hc_on = _hc(), _hc(carrier_bf16=True)
    ref_out, ref_g = _run64(hc_off, h, wf, r, term)
    off_out, off_g = _run(hc_off, h, wf, r, term, torch.float32)
    on_out, on_g = _run(hc_on, h, wf, r, term, torch.bfloat16)
    assert on_out.dtype == torch.bfloat16 and off_out.dtype == torch.float32
    e_on, e_off = _err(on_out, ref_out), _err(off_out, ref_out)
    print(f"  out         fp32 carrier {e_off:.3e}  bf16 carrier {e_on:.3e}")
    assert e_on <= e_off + STORE_TOL, ("out", e_on, e_off)
    for name, a, b, rr in zip(NAMES, off_g, on_g, ref_g):
        e_off, e_on = _err(a, rr), _err(b, rr)
        print(f"  {name:15s} fp32 carrier {e_off:.3e}  bf16 carrier {e_on:.3e}")
        assert e_on <= e_off + STORE_TOL, (name, e_on, e_off)


@cuda
@pytest.mark.parametrize("B,S,with_term", [(2, 96, True), (3, 37, False)])
def test_bf16_carrier_is_the_fp32_carrier_rounded_at_the_store(B, S, with_term):
    """The kernels widen every load and do all math in fp32, so a bf16 carrier must give,
    bit for bit, what an fp32 carrier holding the SAME values gives, rounded once at the
    store: x_bar, the exit output and the carrier grad. Hres and Hpost_row are fp32 in both
    and must be equal (no rounding of the mixing matrices; the 2026-10-06 audit)."""
    h, wf, r, term = _data(B, S, with_term)
    hc = _hc(carrier_bf16=True)
    g = torch.Generator(device="cuda").manual_seed(3)
    y = torch.randn(B, S, C, device="cuda", generator=g).bfloat16()
    gx = torch.randn(B, S, C, device="cuda", generator=g).bfloat16()
    gout = torch.randn(B, S, 4, C, device="cuda", generator=g).bfloat16()
    res = {}
    for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        hh = h.to(dt).clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_bar, hres, hpr, link = K.hc_region_entry(
                hh, hc.proj.weight, hc.proj.bias, hc.tau, hc.cayley_alpha, 3, hc.eps, True)
            out = K.hc_region_exit(hres, hpr, link, y.to(x_bar.dtype), term, True)
        # the backward of one entry + one exit, fed the same bf16-valued cotangents
        torch.autograd.backward([x_bar, out], [gx.to(x_bar.dtype), gout.to(out.dtype)])
        res[name] = (x_bar.detach(), hres.detach(), hpr.detach(), out.detach(), hh.grad)
    for nm, a, b in zip(["x_bar", "out", "grad_h"], (res["bf16"][i] for i in (0, 3, 4)),
                        (res["fp32"][i] for i in (0, 3, 4))):
        want = b.to(a.dtype)                   # the fp32 result, rounded once at the store
        ulp = torch.finfo(a.dtype).eps * want.abs()
        worst = ((a.double() - want.double()).abs() / ulp.double().clamp_min(1e-30)).max().item()
        frac = (a != want).float().mean().item()
        print(f"  {nm:10s} differing {frac:.2e}  worst {worst:.2f} ulp")
        # Two compiles of the same math (a bf16 and an fp32 pointer) differ in the last fp32
        # bit (FMA contraction), which flips a bf16 rounding in ~1e-5 of the elements.
        assert frac < 1e-3 and worst < 1.5, (nm, frac, worst)
    for nm, i in (("hres", 1), ("hpost_row", 2)):
        d = (res["bf16"][i].double() - res["fp32"][i].double()).abs().max().item()
        print(f"  {nm:10s} max abs diff {d:.2e}")
        # fp32 roundoff of the mapping (about 1e-7); a bf16 round of Hres is 1e-3.
        assert d < 2e-5, (nm, d)
    assert res["bf16"][1].dtype == torch.float32 and res["bf16"][2].dtype == torch.float32


@cuda
def test_bf16_carrier_dtypes():
    h, wf, r, term = _data(2, 96, True)
    out, gs = _run(_hc(carrier_bf16=True), h, wf, r, term, torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert gs[0].dtype == torch.bfloat16            # carrier grad: autograd matches h
    assert gs[1].dtype == torch.float32 and gs[2].dtype == torch.float32   # proj weight, bias
    hc = _hc(carrier_bf16=True)
    hb = h.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x_bar, hres, hpr, link = K.hc_region_entry(
            hb, hc.proj.weight, hc.proj.bias, hc.tau, hc.cayley_alpha, 3, hc.eps, True)
    assert x_bar.dtype == torch.bfloat16
    assert hres.dtype == torch.float32 and hpr.dtype == torch.float32   # Hres never bf16
    assert link.dtype == torch.bfloat16


@cuda
def test_bf16_carrier_refuses_without_autocast():
    h, wf, r, term = _data(2, 33, False)
    hc = _hc(carrier_bf16=True)
    with pytest.raises(RuntimeError, match="hc_carrier_bf16"):
        hc(h.clone(), lambda x: x)                 # no autocast


def test_bf16_carrier_refuses_on_cpu():
    torch.manual_seed(1)
    hc = HyperConnectionResidual(C, n_streams=4, use_kernel=True, region_fused=True,
                                 carrier_bf16=True)
    h = torch.randn(1, 4, 4, C)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="hc_carrier_bf16"):
            hc(h, lambda x: x)


@pytest.mark.parametrize("kw", [dict(use_kernel=False, region_fused=False),
                                dict(region_fused=False)])
def test_constructor_refuses_unsupported(kw):
    base = dict(n_streams=4, use_kernel=True, region_fused=True, carrier_bf16=True)
    base.update(kw)
    with pytest.raises(ValueError, match="hc_(carrier_bf16|region_fused)"):
        HyperConnectionResidual(C, **base)


def test_model_refuses_without_region():
    from morph.model.transformer import MORPHConfig, MORPHTransformer
    cfg = MORPHConfig(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=64,
                      context_len=64, n_prelude=1, n_core=1, n_coda=1,
                      channel_dims=(32, 20, 12), d_ff=96, bigram_hash_vocab=64,
                      hc_carrier_bf16=True)
    with pytest.raises(ValueError, match="hc_carrier_bf16"):
        MORPHTransformer(cfg)
    assert MORPHConfig.__dataclass_fields__["hc_carrier_bf16"].default is False


# ── the model ─────────────────────────────────────────────────────────────────────────────

def _tiny_recipe(monkeypatch, on: bool):
    from test_graph_step import _TINY, _compose
    from morph.model.transformer import MORPHTransformer
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config
    keys = ("model.hc_region_fused=true", "model.cca_prologue_tiled=true",
            "model.inject_fold=true", f"model.hc_carrier_bf16={str(on).lower()}")
    cfg, rt = _compose(monkeypatch, "nowta_pointer_fast_spdcells_speed", *keys)
    mc = dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg), **_TINY)
    torch.manual_seed(0)
    m = MORPHTransformer(mc).cuda()
    apply_quantization(m, cfg)
    m.train()
    return m, rt


@cuda
def test_model_carrier_dtypes_and_step0_hinge(monkeypatch):
    from test_graph_step import _batches
    res = {}
    for on in (False, True):
        m, rt = _tiny_recipe(monkeypatch, on)
        assert m.cfg.hc_carrier_bf16 is on
        seen = {"hinge": [], "main": []}
        state = {"in_hinge": False}
        orig = m._slot_gain_penalty

        def _wrapped(*a, _orig=orig, **k):
            state["in_hinge"] = True
            try:
                return _orig(*a, **k)
            finally:
                state["in_hinge"] = False
        m._slot_gain_penalty = _wrapped
        hooks = [mod.register_forward_pre_hook(
                     lambda mod, args: seen["hinge" if state["in_hinge"] else "main"].append(
                         args[0].dtype))
                 for mod in m.modules() if isinstance(mod, HyperConnectionResidual)]
        x, y, lay = _batches(rt, 3)[0]
        torch.manual_seed(5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)
        for hk in hooks:                 # the backward's recompute is not the forward's dtype story
            hk.remove()
        out["loss"].backward()
        assert seen["main"] and seen["hinge"], "the hinge or the main path never ran"
        if on:
            assert set(seen["main"]) == {torch.bfloat16}, set(seen["main"])
            assert set(seen["hinge"]) == {torch.float32}, set(seen["hinge"])
        else:
            assert set(seen["main"] + seen["hinge"]) == {torch.float32}
        res[on] = (out["loss"].detach().float().item(), float(out["gain_est"]),
                   out["ce"].detach().float().item() if "ce" in out else None)
    (l0, g0, _), (l1, g1, _) = res[False], res[True]
    print(f"  step-0 loss off {l0:.5f} on {l1:.5f}; hinge gain off {g0:.5f} on {g1:.5f}")
    assert abs(l1 - l0) < 1e-2, (l0, l1)
    assert abs(g1 - g0) < 5e-3, (g0, g1)    # a bf16 hinge misses by 4e-2


# ── entry points outside the trainer (the key's contract: the function it trained with) ───────

def _eval_model(monkeypatch, on: bool):
    m, rt = _tiny_recipe(monkeypatch, on)
    return m.eval(), rt


@cuda
def test_no_autocast_forward_is_the_autocast_forward_bit_for_bit(monkeypatch):
    """Key on: an entry point called with autocast off enters bf16 autocast itself, so it
    returns what the same call returns inside the caller's autocast. The wrapper is
    `torch.autocast` around the same call, so the ops are the same ops. Not bit for bit,
    because two autocast calls of the SAME model already differ from each other by 2.4e-7
    on the logits (atomics in the slot gather; measured 2026-10-12, 3 repeats), so the bound
    is `ATOL` = 1e-5, 40x that floor. An fp32 carrier would differ by about 1e-2 (bf16
    store rounding), and the carrier-dtype check below pins which function ran. Covers
    `forward` (the label-free val forward) and `tul_forward_ablated` (an `evaluate`
    ablation forward that raised before the wrapper)."""
    from test_graph_step import _batches
    m, rt = _eval_model(monkeypatch, True)
    x, y, lay = _batches(rt, 3)[0]
    for call in (lambda: m(x, slot_layout=lay),
                 lambda: m.tul_forward_ablated(x, y, lay, plan_mode="zero")):
        with torch.no_grad():
            torch.manual_seed(7)
            assert not torch.is_autocast_enabled("cuda")
            got = call()
            torch.manual_seed(7)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                want = call()
        key = "logits" if got.get("logits") is not None else "loss"
        assert got[key].dtype == want[key].dtype
        a, b = got[key], want[key]
        # pad rows of the eval logits hold NaN in both runs: same mask, then the max error
        # over the rest.
        assert torch.equal(a.isnan(), b.isnan())
        err = (a - b).nan_to_num().abs().max().item()
        assert err <= ATOL, err
    # The wrapper is not a silent fp32 carrier: the carrier it ran is bf16.
    seen = []
    hooks = [mod.register_forward_pre_hook(lambda mod, a: seen.append(a[0].dtype))
             for mod in m.modules() if isinstance(mod, HyperConnectionResidual)]
    with torch.no_grad():
        m(x, slot_layout=lay)
    for hk in hooks:
        hk.remove()
    assert seen and set(seen) == {torch.bfloat16}, set(seen)


@cuda
def test_key_off_adds_no_autocast(monkeypatch):
    """Key off: the entry points run exactly as before, with autocast left off."""
    from test_graph_step import _batches
    m, rt = _eval_model(monkeypatch, False)
    x, y, lay = _batches(rt, 3)[0]
    seen = []
    hk = m.register_forward_pre_hook(lambda mod, a, kw: seen.append(
        torch.is_autocast_enabled("cuda")), with_kwargs=True)
    with torch.no_grad():
        m(x, slot_layout=lay)
    hk.remove()
    assert seen == [False]


def test_key_on_without_cuda_raises_clearly(monkeypatch):
    from morph.model.transformer import _carrier_autocast
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _Stub(torch.nn.Module):
        _carrier_bf16 = True

        @_carrier_autocast
        def forward(self, x):
            raise AssertionError("must not run")
    with pytest.raises(RuntimeError, match="needs CUDA.*no fp32 fallback"):
        _Stub()(1)


@cuda
def test_eager_generate_tul_on_the_fan_recipe_with_the_key(monkeypatch):
    """`generate_tul` (eager, no autocast anywhere in it) runs on the key-on fan model and
    emits tokens; the same call raised before the model entered autocast itself."""
    from test_graph_step import SEQ, _rule
    from morph.inference.tul_generate import generate_tul
    from morph.model.tul_layout import TulLayoutSpec
    m, rt = _eval_model(monkeypatch, True)
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=rt.data_cfg.prefix_k, max_slots=10,
                         slot_id=rt.data_cfg.slot_id)
    toks, builder = generate_tul(m, [5, 6, 7, 8, 10, 12, 13], _rule(), spec,
                                 max_new_tokens=6, temperature=0.0, device="cuda",
                                 emit_source="token")
    assert len(toks) == 6 and all(0 <= int(t) < 64 for t in toks), toks


@cuda
def test_cached_decoders_refuse_the_key(monkeypatch):
    from morph.inference.tul_generate_cached import _check_supported
    m, _ = _eval_model(monkeypatch, True)
    with pytest.raises(NotImplementedError, match="hc_carrier_bf16"):
        _check_supported(m)
