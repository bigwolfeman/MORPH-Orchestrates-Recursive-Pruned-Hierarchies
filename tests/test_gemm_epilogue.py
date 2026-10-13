"""``model.gemm_epilogue_fusion`` (2026-10-13; plan note
.agents/notes/proposed/architecture/2026-10-12-core-pass-persistent-kernel-plan.md, stage F1).

The dense MLP ``down(silu(gate) * up)`` becomes one split op whose Triton GEMMs carry the SwiGLU:
the ``gate_up`` forward writes ``h`` beside the pre-activation ``gu``; the ``down`` data-grad GEMM
forms ``d_gate`` and ``d_up`` from the saved ``gu``. Elementwise epilogue work only; the tile grid
stays. The key promises rounding-level equality, not bit equality, so each test pins a contract:

  1. THE KERNELS AGAINST fp64. ``swiglu_fwd`` (``gu`` and ``h``) and ``swiglu_bwd`` (``d_gu``) are
     as close to an fp64 reference as the path they replace, at the core, prelude and coda row
     counts (the unfused path = the Triton GEMM, then the compiled pointwise kernel inductor
     emits today).
  2. EVERY CONFIG GIVES THE SAME BITS (the tuner changes speed, never numbers).
  3. THE WHOLE OP against fp64: ``y``, ``dx`` and both weight grads, summed over two reads, no
     worse than the unfused split ops.
  4. GRAPH CAPTURE: an untuned shape RAISES, a tuned one replays bit for bit; a recording that
     meets untuned keys collects them, runs the cuBLAS equivalents, and the keys then tune.
  5. THE MODEL: a tiny model with the key on tracks the key-off Triton arm (loss and gradients
     close), calls the fused op, and a captured-and-replayed fused run equals its eager run bit
     for bit. Outside the training scope the plain MLP runs.
  6. REFUSALS at build and bind.

Sabotage checks, run and reverted (session report): (a) the forward epilogue drops the silu
(``hv = gf * uf``) -> tests 1 and 3 fail; (b) the backward epilogue swaps the ``d_gate`` and
``d_up`` stores -> tests 1 and 3 fail; (c) the op's backward hands ``gu`` where ``h`` belongs to
down's weight grad -> test 3 fails.
CUDA required.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from test_graph_step import (_batches, _compose, _fwd, _same, _snap, _variants, _zero_pen, cuda,
                             deterministic, _ON, _HINGE_ON)
from test_stream_overlap import _PLAIN, _SPLIT, _build, _explicit_up, _run_eager, _triton_mgr

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

D, DFF = 1024, 2816                      # the recipe's d_model and d_ff
# core passes (1536, 1104, 408), a small odd count, a prelude / coda row count (6 x 1280)
_ROWS = [1536, 1104, 408, 33, 7680]
_EPI = ("model.gemm_epilogue_fusion=true",)


def _single_rounding_forward(self, x, iter_idx=0):
    """The unfused MLP with the arithmetic inductor gives it in the compiled model: fp32 inside
    the SwiGLU, ONE rounding to bf16. Eager autograd would round after the silu AND after the
    product; on the test fixture (loss ~ 90, untrained) that double rounding alone moves the loss
    by 0.3 % and single gradients by 100 %, so it is not the reference for the fused arm."""
    gu = self.gate_up(x)
    gate, up = gu.float().chunk(2, dim=-1)
    return self.down((F.silu(gate) * up).to(torch.bfloat16))


def _err(t, ref):
    return (t.double() - ref).abs().mean().item()


@pytest.fixture(scope="module")
def pointwise():
    """The unfused SwiGLU as inductor compiles it today: one kernel forward, one backward."""
    def f(gu):
        gate, up = gu.chunk(2, dim=-1)
        return F.silu(gate) * up
    return torch.compile(f, dynamic=True)


# ── 1. the kernels against fp64 ─────────────────────────────────────────────────────────


@cuda
@pytest.mark.parametrize("m", _ROWS)
def test_swiglu_kernels_are_as_close_to_fp64_as_the_unfused_path(m, pointwise):
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(0)
    x = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)
    wgu = (torch.randn(2 * DFF, D, device="cuda") * 0.03).to(torch.bfloat16)
    wdn = (torch.randn(D, DFF, device="cuda") * 0.03).to(torch.bfloat16)
    dy = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)

    gu, h = sg.swiglu_fwd(x, wgu)
    gu_u = sg.linear_fwd(x, wgu)                                 # unfused: GEMM, then pointwise
    gu_u.requires_grad_()
    h_u = pointwise(gu_u)
    # fp64 reference of the whole chain from the same bf16 inputs
    gu64 = x.double() @ wgu.double().t()
    gate64, up64 = gu64.chunk(2, dim=-1)
    h64 = F.silu(gate64) * up64
    assert _err(gu, gu64) <= _err(gu_u, gu64) * 1.05 + 1e-6, "gu"
    assert _err(h, h64) <= _err(h_u, h64) * 1.05 + 1e-6 * h64.abs().mean().item(), "h"
    assert _err(h, h64) < 0.01 * h64.abs().mean().item()         # close, not just "no worse"

    dgu = sg.swiglu_bwd(dy, wdn, gu)
    dh_u = sg.linear_dgrad(dy, wdn)                              # unfused: GEMM, then pointwise bwd
    h_u2 = pointwise(gu_u)
    (dgu_u,) = torch.autograd.grad(h_u2, gu_u, dh_u)
    dh64 = dy.double() @ wdn.double()
    g64, u64 = gu64.chunk(2, dim=-1)
    sig = torch.sigmoid(g64)
    dgu64 = torch.cat([dh64 * u64 * sig * (1 + g64 * (1 - sig)), dh64 * g64 * sig], dim=-1)
    scale = dgu64.abs().mean().item()
    assert _err(dgu, dgu64) <= _err(dgu_u, dgu64) * 1.10 + 1e-6 * scale, "d_gu"
    assert _err(dgu, dgu64) < 0.02 * scale


# ── 2. every config, the same bits ──────────────────────────────────────────────────────


@cuda
def test_every_swiglu_config_gives_the_same_bits():
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(2)
    m = 1104
    x = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)
    wgu = (torch.randn(2 * DFF, D, device="cuda") * 0.03).to(torch.bfloat16)
    wdn = (torch.randn(D, DFF, device="cuda") * 0.03).to(torch.bfloat16)
    dy = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)
    outs = []
    for cfg in sg.SWIGLU_FWD_CONFIGS:
        gu = torch.empty(m, 2 * DFF, device="cuda", dtype=torch.bfloat16)
        h = torch.empty(m, DFF, device="cuda", dtype=torch.bfloat16)
        sg._launch(x, wgu.t(), gu, sg.MODE_SWIGLU_FWD, cfg, h)
        outs.append((gu, h))
    assert all(torch.equal(outs[0][0], o[0]) and torch.equal(outs[0][1], o[1]) for o in outs[1:])
    gu = outs[0][0]
    bwd = []
    for cfg in sg.CONFIGS:
        d = torch.empty_like(gu)
        sg._launch(dy, wdn, d, sg.MODE_SWIGLU_BWD, cfg, gu)
        bwd.append(d)
    assert all(torch.equal(bwd[0], o) for o in bwd[1:])


# ── 3. the whole op against fp64 ────────────────────────────────────────────────────────


def _cases(m):
    """Two reads of one MLP: a 3-D input (the model's layout) and a 2-D one."""
    return [(2, m // 2), (m,)] if m >= 64 else [(m,), (m,)]


def _mlp_run(path, xs, cs, wgu0, wdn0, pointwise):
    """sum_i <mlp(x_i), c_i> through ``path`` ('fused', 'unfused' or 'fp64'); returns the value,
    dx per read and the weight grads."""
    from morph.model.split_wgrad import SplitWeight, as_bf16, split_linear, split_swiglu_mlp
    if path == "fp64":
        wgu, wdn = wgu0.double().requires_grad_(), wdn0.double().requires_grad_()
        xr = [x.double().requires_grad_() for x in xs]
        loss = 0
        for x, c in zip(xr, cs):
            g, u = F.linear(x, wgu).chunk(2, dim=-1)
            loss = loss + (F.linear(F.silu(g) * u, wdn) * c.double()).sum()
    else:
        mgr = _triton_mgr()
        wgu, wdn = wgu0.clone().requires_grad_(), wdn0.clone().requires_grad_()
        sw_gu = SplitWeight(mgr, "gu", [wgu], [(0, wgu.shape[0])], wgu.shape[1], kind="callable",
                            source=lambda: wgu.detach().to(torch.bfloat16))
        sw_dn = SplitWeight(mgr, "dn", [wdn], [(0, wdn.shape[0])], wdn.shape[1], kind="callable",
                            source=lambda: wdn.detach().to(torch.bfloat16))
        xr = [x.clone().requires_grad_() for x in xs]
        loss = 0
        for x, c in zip(xr, cs):
            if path == "fused":
                y = split_swiglu_mlp(as_bf16(x), sw_gu, sw_dn)
            else:
                y = split_linear(pointwise(split_linear(as_bf16(x), sw_gu)), sw_dn)
            loss = loss + (y.double() * c.double()).sum()
    loss.backward()
    return loss.item(), [x.grad for x in xr], wgu.grad.clone(), wdn.grad.clone()


@cuda
@pytest.mark.parametrize("m", [1536, 408, 33])
def test_the_fused_op_matches_fp64_as_well_as_the_unfused_split_ops(m, pointwise):
    torch.manual_seed(5)
    shapes = _cases(m)
    xs = [torch.randn(*s, D, device="cuda", dtype=torch.bfloat16) for s in shapes]
    cs = [torch.randn(*s, D, device="cuda", dtype=torch.bfloat16) for s in shapes]
    wgu = (torch.randn(2 * DFF, D, device="cuda") * 0.03).to(torch.bfloat16).float()
    wdn = (torch.randn(D, DFF, device="cuda") * 0.03).to(torch.bfloat16).float()
    ref = _mlp_run("fp64", xs, cs, wgu, wdn, pointwise)
    fused = _mlp_run("fused", xs, cs, wgu, wdn, pointwise)
    unf = _mlp_run("unfused", xs, cs, wgu, wdn, pointwise)
    names = (["value"] + [f"dx{i}" for i in range(len(xs))] + ["dW_gate_up", "dW_down"])
    flat = lambda r: [torch.tensor(r[0])] + list(r[1]) + [r[2], r[3]]       # noqa: E731
    for name, f, u, r in zip(names, flat(fused), flat(unf), flat(ref)):
        scale = r.abs().mean().item()
        assert _err(f, r) <= _err(u, r) * 1.10 + 1e-6 * scale, (name, _err(f, r), _err(u, r))
        assert _err(f, r) < 0.05 * scale + 1e-9, (name, _err(f, r), scale)


# ── 4. capture ──────────────────────────────────────────────────────────────────────────


@cuda
def test_an_untuned_swiglu_shape_under_capture_raises_and_a_tuned_one_replays():
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(7)
    wgu = torch.randn(2 * 640, 256, device="cuda").to(torch.bfloat16)
    wdn = torch.randn(256, 640, device="cuda").to(torch.bfloat16)
    x = torch.randn(520, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(520, 256, device="cuda", dtype=torch.bfloat16)
    xo = torch.randn(528, 256, device="cuda", dtype=torch.bfloat16)        # never tuned
    gu_ref, h_ref = sg.swiglu_fwd(x, wgu)                                 # tunes (eager)
    d_ref = sg.swiglu_bwd(dy, wdn, gu_ref)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gu, h = sg.swiglu_fwd(x, wgu)
        d = sg.swiglu_bwd(dy, wdn, gu)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(gu, gu_ref) and torch.equal(h, h_ref) and torch.equal(d, d_ref)
    for fn in (lambda: sg.swiglu_fwd(xo, wgu), lambda: sg.swiglu_bwd(xo[:, :256], wdn, torch.empty(
            528, 2 * 640, device="cuda", dtype=torch.bfloat16))):
        g2 = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="not tuned"):
            with torch.cuda.graph(g2):
                fn()
        torch.cuda.synchronize()


@cuda
def test_a_recording_collects_swiglu_keys_runs_the_cublas_form_and_the_keys_then_tune():
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(8)
    m = 456                                                              # a row count nothing tuned
    wgu = (torch.randn(2 * 640, 256, device="cuda") * 0.05).to(torch.bfloat16)
    wdn = (torch.randn(256, 640, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    sg.clear_tuned()
    g = torch.cuda.CUDAGraph()
    with sg.CAPTURE_TUNER.recording():
        with torch.cuda.graph(g):
            gu, h = sg.swiglu_fwd(x, wgu)
            d = sg.swiglu_bwd(dy, wdn, gu)
    g.replay()
    torch.cuda.synchronize()
    # the collecting pass computed the cuBLAS form of both epilogues
    gu_c = (x.float() @ wgu.float().t()).to(torch.bfloat16)
    assert (gu.float() - gu_c.float()).abs().max() <= 0.02 * gu_c.float().abs().max()
    assert torch.equal(h, sg.swiglu_fwd_torch(gu))
    assert torch.equal(d, sg.swiglu_bwd_torch((dy @ wdn), gu))
    assert sg.CAPTURE_TUNER.pending() == 2
    assert sg.CAPTURE_TUNER.tune() == 2 and sg.CAPTURE_TUNER.pending() == 0
    gu2, h2 = sg.swiglu_fwd(x, wgu)                                       # now a plain tuned launch
    assert (h2.float() - h.float()).abs().max() <= 0.05 * h.float().abs().max() + 1e-6


# ── 5. the model ────────────────────────────────────────────────────────────────────────


@cuda
def test_fused_arm_tracks_the_unfused_triton_arm_and_graph_replay_equals_its_eager(monkeypatch,
                                                                                     deterministic):
    """Three eager steps of the fused arm track the unfused Triton arm (loss and every gradient
    close), the fused op really runs, and a captured-and-replayed fused run equals the eager
    fused run bit for bit."""
    import morph.model.transformer as T
    from morph.training.graph_step import GraphStep
    plan = [(True, "eager"), (False, "eager"), (False, "capture"), (False, "replay"),
            (True, "replay"), (False, "replay")]
    arm_u = _SPLIT + ("model.split_wgrad_triton=true",)
    arm_f = arm_u + _EPI
    cfg_u, rt_u = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_HINGE_ON, *arm_u)
    cfg_f, rt_f = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_HINGE_ON, *arm_f)
    batches = _batches(rt_u, len(plan))
    calls = []
    real = T.split_swiglu_mlp
    monkeypatch.setattr(T, "split_swiglu_mlp", lambda *a, **k: calls.append(1) or real(*a, **k))
    with monkeypatch.context() as mp:
        _explicit_up(mp)
        mp.setattr(T._SwiGLUMortar, "forward", _single_rounding_forward)
        _mu, _gu, ref_u = _run_eager(cfg_u, rt_u, batches, len(plan), strict=True,
                                     kinds=[k for k, _ in plan])
        assert not calls                                  # key off: the op is never built
    with monkeypatch.context() as mp:
        _explicit_up(mp)
        _mf, mgr_f, ref_f = _run_eager(cfg_f, rt_f, batches, len(plan), strict=True,
                                       kinds=[k for k, _ in plan])
    assert calls and mgr_f.triton and mgr_f.deferred > 0
    assert abs(ref_f[0]["loss"].item() - ref_u[0]["loss"].item()) < 1e-4 * abs(
        ref_u[0]["loss"].item())
    n_cmp = 0
    for a, b in zip(ref_u[0]["grads"], ref_f[0]["grads"]):
        if a is None:
            assert b is None
            continue
        assert (a.float() - b.float()).abs().max() <= 5e-2 * a.float().abs().max() + 1e-8
        n_cmp += 1
    assert n_cmp > 20
    del _mu, _mf

    mb, ob, mgr = _build(cfg_f, rt_f)
    clip = float(cfg_f.training.grad_clip)
    gs = GraphStep(mb, ob, grad_clip=clip, forward=_fwd(mb), kinds=(False, True),
                   set_kind=lambda k: setattr(mb, "_train_instruments", k),
                   loss_terms=_zero_pen, after_step=mb.tul_fan_after_step, first_capture=2,
                   keep_grads=(True, False), compiled=False, **_variants(mb))
    torch.manual_seed(1234)
    with monkeypatch.context() as mp:
        _explicit_up(mp)
        for s, ((kind, how), (x, y, lay)) in enumerate(zip(plan, batches)):
            mb._train_instruments = kind
            for g in ob.param_groups:
                g["lr"] = 1e-3 * (1 + 0.1 * s)
            ob.write_step_scalars()
            ob.zero_grad(set_to_none=True)
            out, loss, gn, mode = gs.step(s, kind, x, y, lay, context="tul")
            assert mode == how, (s, mode, how)
            _same(ref_f[s], _snap(mb, ob, out, loss, gn, [p.grad for p in mb.parameters()]),
                  f"fused step {s} ({how})")
            mb._gain_pass_last = None
    assert gs.stats.replayed == 4


@cuda
def test_outside_the_training_scope_the_plain_mlp_runs(monkeypatch):
    import morph.model.transformer as T
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_SPLIT,
                       "model.split_wgrad_triton=true", *_EPI)
    m, _opt, _mgr = _build(cfg, rt)
    calls = []
    real = T.split_swiglu_mlp
    monkeypatch.setattr(T, "split_swiglu_mlp", lambda *a, **k: calls.append(1) or real(*a, **k))
    mlp = next(mod for mod in m.modules() if isinstance(mod, T._SwiGLUMortar))
    assert mlp._swiglu_fused
    x = torch.randn(2, 16, mlp.d_model, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mlp(x).float().sum().backward()                   # grad mode, but no train_backward_scope
    assert not calls and x.grad is not None


# ── 6. refusals ─────────────────────────────────────────────────────────────────────────


@cuda
def test_the_key_refuses_without_the_triton_split_gemms(monkeypatch):
    from test_graph_step import _build as _base_build
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_SPLIT, *_EPI)
    with pytest.raises(ValueError, match="needs model.split_wgrad_triton=true"):
        _base_build(cfg, rt)


@cuda
def test_bind_refuses_epilogue_without_triton_and_a_routed_mlp(monkeypatch):
    from morph.model.split_wgrad import bind_split_wgrad
    from morph.model.ternary_qat import bind_ternary_step_cache
    from test_graph_step import _build as _base_build
    import morph.model.transformer as T
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_SPLIT,
                       "model.split_wgrad_triton=true")
    m, _opt = _base_build(cfg, rt)
    bind_ternary_step_cache(m, None)
    with pytest.raises(ValueError, match="needs model.split_wgrad_triton=true"):
        bind_split_wgrad(m, triton=False, epilogue=True)
    mlp = next(mod for mod in m.modules() if isinstance(mod, T._SwiGLUMortar))
    mlp.enable_routing(n_clusters=16, n_iters=1)
    with pytest.raises(ValueError, match="ReMoE router"):
        bind_split_wgrad(m, triton=True, epilogue=True)
    mlp.router = None
    bind_split_wgrad(m, triton=True, epilogue=True)
    assert mlp._swiglu_fused
    with pytest.raises(ValueError, match="fused MLP op has no gate"):
        mlp.enable_routing(n_clusters=16, n_iters=1)
