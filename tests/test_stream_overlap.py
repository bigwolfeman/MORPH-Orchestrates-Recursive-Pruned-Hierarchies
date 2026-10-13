"""``model.slot_gain_stream`` and ``model.split_wgrad`` (2026-10-12; note
.agents/notes/proposed/architecture/2026-10-12-streams-hinge-and-wgrad.md).

Both keys move work to a second CUDA stream and promise the SAME numbers as the inline path.
What each test pins:

  1. KEYS ON == KEYS OFF, EAGER, BIT FOR BIT. A tiny winner (the graph_step fixture, slot
     compaction, step-cached ternary MLP) trains three steps once with both keys off and once
     with each key and both keys on; the loss, the grad norm, every `out` reading, every
     gradient, parameter, buffer and optimizer state are equal bit for bit after every step.
     The hinge is active on every step (target 0) in one case and never (target 100) in the
     other, so both hinge-branch bodies' gradients are compared (the eager step runs the whole
     backward; the branch bodies are test 2).
  2. KEYS ON, CAPTURED AND REPLAYED == KEYS OFF, EAGER. GraphStep records the keys-on step
     (both kinds, every hinge pass, both hinge-branch bodies) and replays it; every step
     equals the keys-off eager step bit for bit.
  3. THE ORDER OF THE JOIN. A weight read twice sums its two weight grads in the reads' order;
     the side stream is made slow on purpose so a missing wait would show (the sabotage
     checks below).
  4. THE COVERAGE CONTRACT. With `strict` on, a backward whose covered weight already has a
     `.grad` (a read that bypassed the op) raises; no covered parameter reaches `.grad`
     through autograd alone.
  5. Refusals at build.

Sabotage checks, run and reverted (session report, not committed):
  (a) `_SplitLinear.backward` records `ready` AFTER `defer` (the side stream does not wait for
      `dy`) -> test 3 fails (a long sleep on the producing stream).
  (b) `WgradStream.finish` skips `cur.wait_event(done)` -> test 3 (slow side stream) fails.
  (c) `_tul_core` skips the main stream's `wait_event` before `_slot_gain_reduce` -> test 1
      (hinge stream made slow) fails.
CUDA required: these are stream tests.
"""
from __future__ import annotations

import dataclasses
import os

import pytest
import torch

from test_graph_step import (_COMPACT, _HINGE_OFF, _HINGE_ON, _ON, _batches, _compose, _eager_step,
                             _eq, _fwd, _same, _snap, _variants, _zero_pen, cuda, deterministic)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_PLAIN = _COMPACT + ("model.ternary_step_cache=true",)
_HINGE_STREAM = ("model.slot_gain_stream=true",)
_SPLIT = ("model.split_wgrad=true",)
_ARMS = {"hinge_stream": _HINGE_STREAM, "split_wgrad": _SPLIT,
         "both": _HINGE_STREAM + _SPLIT}


def _explicit_up(monkeypatch):
    """The plain W_up read with its bf16 cast as an explicit per-read op, which is what the
    compiled block does (inductor emits one cast per use). Eager autocast instead CACHES the
    cast of a leaf parameter for the whole forward and sums the reads' weight grads in bf16 at
    that one node, so eager keys-off W_up differs from every compiled run by design; the
    reference runs of the split arms use this form."""
    from morph.model.attention import _CCABase

    def up(self, z):
        return torch.nn.functional.linear(z, self.W_up.weight.to(torch.bfloat16))
    monkeypatch.setattr(_CCABase, "_up", up)


def _build(cfg, rt, seed=0, *, strict=False):
    """The graph_step fixture's build, then the step cache and (key on) the split-wgrad bind,
    in train.py's order."""
    from morph.model.split_wgrad import bind_split_wgrad
    from morph.model.ternary_qat import bind_ternary_step_cache
    from test_graph_step import _build as _base_build
    m, opt = _base_build(cfg, rt, seed)
    bind_ternary_step_cache(m, None)
    mgr = (bind_split_wgrad(m, strict=strict, triton=bool(m.cfg.split_wgrad_triton),
                            epilogue=bool(m.cfg.gemm_epilogue_fusion))
           if bool(m.cfg.split_wgrad) else None)
    return m, opt, mgr


def _run_eager(cfg, rt, batches, n, *, strict=False, kinds=None):
    m, opt, mgr = _build(cfg, rt, strict=strict)
    torch.manual_seed(1234)
    clip = float(cfg.training.grad_clip)
    snaps = []
    for s in range(n):
        m._train_instruments = bool(kinds[s]) if kinds else False
        for g in opt.param_groups:
            g["lr"] = 1e-3 * (1 + 0.1 * s)
        opt.write_step_scalars()
        out, loss, gn = _eager_step(m, opt, *batches[s], clip)
        snaps.append(_snap(m, opt, out, loss, gn, [p.grad for p in m.parameters()]))
        snaps[-1]["pass"] = m._gain_pass_last
        m._gain_pass_last = None
    return m, mgr, snaps


@cuda
@pytest.mark.parametrize("hinge", [_HINGE_ON, _HINGE_OFF], ids=["hinge_active", "hinge_inactive"])
@pytest.mark.parametrize("arm", list(_ARMS))
def test_keys_on_equal_keys_off_eager(monkeypatch, deterministic, arm, hinge):
    cfg0, rt0 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *hinge)
    cfg1, rt1 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *hinge, *_ARMS[arm])
    batches = _batches(rt0, 3)
    with monkeypatch.context() as mp:
        if "split_wgrad" in arm or arm == "both":
            _explicit_up(mp)
        _m0, _g0, ref = _run_eager(cfg0, rt0, batches, 3)
    m1, mgr, got = _run_eager(cfg1, rt1, batches, 3, strict=True)
    for s, (a, b) in enumerate(zip(ref, got)):
        _same(a, b, f"{arm} step {s}")
        assert a["pass"] == b["pass"]
    if "split_wgrad" in arm or arm == "both":
        # the op ran: every covered weight formed its gradient on the side stream
        assert mgr.deferred > 0 and mgr.joins == 3, (mgr.deferred, mgr.joins)


@cuda
@pytest.mark.parametrize("hinge", [(), _HINGE_ON, _HINGE_OFF],
                         ids=["winner", "hinge_active", "hinge_inactive"])
@pytest.mark.parametrize("arm", ["hinge_stream", "split_wgrad", "both"])
def test_keys_on_graph_step_equals_keys_off_eager(monkeypatch, deterministic, arm, hinge):
    from morph.training.graph_step import GraphStep
    plan = [(True, "eager"), (False, "eager"), (False, "capture"), (False, "replay"),
            (True, "replay"), (False, "replay")]
    cfg0, rt0 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *hinge)
    cfg1, rt1 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *hinge, *_ARMS[arm])
    batches = _batches(rt0, len(plan))
    with monkeypatch.context() as mp:
        if arm != "hinge_stream":
            _explicit_up(mp)
        _m0, _g0, ref = _run_eager(cfg0, rt0, batches, len(plan), kinds=[k for k, _ in plan])
    del _m0

    mb, ob, mgr = _build(cfg1, rt1)
    clip = float(cfg1.training.grad_clip)
    gs = GraphStep(mb, ob, grad_clip=clip, forward=_fwd(mb), kinds=(False, True),
                   set_kind=lambda k: setattr(mb, "_train_instruments", k),
                   loss_terms=_zero_pen, after_step=mb.tul_fan_after_step, first_capture=2,
                   keep_grads=(True, False), compiled=False, **_variants(mb))
    torch.manual_seed(1234)
    for s, ((kind, how), (x, y, lay)) in enumerate(zip(plan, batches)):
        mb._train_instruments = kind
        for g in ob.param_groups:
            g["lr"] = 1e-3 * (1 + 0.1 * s)
        ob.write_step_scalars()
        ob.zero_grad(set_to_none=True)
        out, loss, gn, mode = gs.step(s, kind, x, y, lay, context="tul")
        assert mode == how, (s, mode, how)
        _same(ref[s], _snap(mb, ob, out, loss, gn, [p.grad for p in mb.parameters()]),
              f"{arm} step {s} ({how})")
        mb._gain_pass_last = None
    assert gs.stats.replayed == 4
    if arm != "hinge_stream":
        # a recording defers inside the capture (the join sits inside both hinge bodies); the
        # replays run no Python, so the counter moves at the recordings and the eager steps
        assert mgr.deferred > 0


# ── 3. the order of the join, with a slow side stream ────────────────────────────────


class _SlowWgrad:
    """A WgradStream whose side stream sleeps before every weight GEMM, so a wait that is
    missing shows as a wrong number and not as a lucky schedule."""

    @staticmethod
    def make(ms_cycles: int = 100_000_000, group_reads: int = 8):
        from morph.model.split_wgrad import WgradStream
        from morph.model.streams import side_stream

        class Slow(WgradStream):
            def defer(self, sw, dy, x, ready):
                with torch.cuda.stream(self.stream):
                    torch.cuda._sleep(ms_cycles)      # on the SIDE stream, ahead of the GEMM
                super().defer(sw, dy, x, ready)
        return Slow(side_stream("wgrad-test"), strict=True, group_reads=group_reads)


def _reads(mgr, w, xs, delay_dy: bool):
    """``sum_i |x_i W^T|^2`` through one split read of the weight ``w`` per input."""
    from morph.model.split_wgrad import SplitWeight, as_bf16, split_linear
    sw = SplitWeight(mgr, "t", [w], [(0, w.shape[0])], w.shape[1], kind="callable",
                     source=lambda: w.detach().to(torch.bfloat16))

    class SlowGrad(torch.autograd.Function):
        """Identity whose backward is late: dy is produced after a long sleep, so a side
        stream that does not wait for ``ready`` reads garbage."""
        @staticmethod
        def forward(ctx, a):
            return a.view_as(a)

        @staticmethod
        def backward(ctx, g):
            torch.cuda._sleep(400_000_000)       # later than the side stream's own sleep
            return g * 1.0

    ys = [split_linear(as_bf16(x), sw) for x in xs]
    ys = [SlowGrad.apply(y) for y in ys] if delay_dy else ys
    return sum((y.float() ** 2).sum() for y in ys)


@cuda
@pytest.mark.parametrize("delay_dy", [False, True], ids=["slow_side_stream", "late_dy"])
def test_the_join_waits_and_the_reads_sum_in_the_engines_order(delay_dy):
    """Three reads of one weight whose weight grads differ by 6 orders of magnitude (so the
    fp32 sum depends on its association): the op's `.grad` is the plain autograd one, bit for
    bit, with a side stream that sleeps before every GEMM and (late_dy) a dy that is produced
    late."""
    torch.manual_seed(0)
    w = torch.randn(48, 32, device="cuda", requires_grad=True)
    xs = [torch.randn(40, 32, device="cuda") * 1e3, torch.randn(24, 32, device="cuda"),
          torch.randn(33, 32, device="cuda") * 1e-3]
    w2 = w.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ys = [torch.nn.functional.linear(x.to(torch.bfloat16), w2.to(torch.bfloat16))
              for x in xs]                      # one explicit cast per read (compiled form)
        sum((y.float() ** 2).sum() for y in ys).backward()
    # Pass 0 only loads the kernels: a first launch of a kernel loads its module, and the load
    # waits for every stream, which hides a missing wait. Pass 1 is the one that counts.
    for attempt in range(2):
        w.grad = None
        mgr = _SlowWgrad.make()
        xr = [x.clone().requires_grad_() for x in xs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = _reads(mgr, w, xr, delay_dy)
        loss.backward()
        assert mgr.joins == 1 and mgr.deferred == 3
        assert _eq(w.grad, w2.grad), (attempt, (w.grad - w2.grad).abs().max())


@cuda
def test_held_references_are_bounded_and_the_sum_is_unchanged():
    """12 reads of one weight with groups of 2: at most two closed groups plus the open one
    are alive at any time, and the gradient is still the plain autograd one bit for bit."""
    torch.manual_seed(0)
    w = torch.randn(48, 32, device="cuda", requires_grad=True)
    xs = [torch.randn(16 + i, 32, device="cuda") * (10.0 ** (3 - i % 7)) for i in range(12)]
    w2 = w.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ys = [torch.nn.functional.linear(x.to(torch.bfloat16), w2.to(torch.bfloat16))
              for x in xs]
        sum((y.float() ** 2).sum() for y in ys).backward()
    mgr = _SlowWgrad.make(1_000_000, group_reads=2)
    xr = [x.clone().requires_grad_() for x in xs]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = _reads(mgr, w, xr, False)
    loss.backward()
    assert mgr.deferred == 12 and mgr.joins == 1
    assert mgr.peak_held <= 6, mgr.peak_held
    assert _eq(w.grad, w2.grad)


# ── 4. coverage ─────────────────────────────────────────────────────────────────────


@cuda
def test_strict_join_raises_on_a_bypassing_read():
    from morph.model.split_wgrad import SplitWeight, as_bf16, split_linear
    mgr = _SlowWgrad.make(1)
    w = torch.randn(16, 8, device="cuda", requires_grad=True)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    sw = SplitWeight(mgr, "t", [w], [(0, 16)], 8, kind="callable",
                     source=lambda: w.detach().to(torch.bfloat16))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = split_linear(as_bf16(x), sw).float().sum() + torch.nn.functional.linear(x, w).float().sum()
    with pytest.raises(RuntimeError, match="bypassed the split op"):
        y.backward()


@cuda
def test_no_covered_parameter_reaches_grad_through_autograd(monkeypatch):
    """Every covered weight is read through the op: with the join disabled, a full backward
    leaves every covered parameter without a gradient."""
    from morph.model import split_wgrad
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_SPLIT)
    m, opt, mgr = _build(cfg, rt)
    covered = {id(p) for mod in m.modules()
               for sw in (getattr(mod, "_split", None), getattr(mod, "_split_in", None),
                          getattr(mod, "_split_up", None)) if sw is not None for p in sw.params}
    assert covered
    monkeypatch.setattr(split_wgrad.SplitWeight, "bind_grads", lambda self, strict: None)
    opt.zero_grad(set_to_none=True)
    x, y, lay = _batches(rt, 3)[0]
    from morph.model.split_wgrad import train_backward_scope
    with torch.autocast("cuda", dtype=torch.bfloat16), train_backward_scope():
        loss = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)["loss"]
    loss.backward()
    leaked = [n for n, p in m.named_parameters() if id(p) in covered and p.grad is not None]
    assert not leaked, leaked
    assert mgr.deferred > 0


# ── 4b. the op is opt-in by scope: a probe backward never meets it ───────────────────────


def _covered(m):
    return [p for mod in m.modules()
            for sw in (getattr(mod, "_split", None), getattr(mod, "_split_in", None),
                       getattr(mod, "_split_up", None)) if sw is not None for p in sw.params]


def _probe_grads(m, batch):
    """A full-model probe as the repo's probes run it: forward under enable_grad + bf16
    autocast OUTSIDE the training scope, then ``autograd.grad`` of the loss with respect to an
    activation (the first block's output), not a parameter."""
    x, y, lay = batch
    caught = []
    torch.manual_seed(7)                      # the probe draws dropout masks
    first = next(iter(m.prelude))
    h = first.register_forward_hook(
        lambda mod, inp, out: caught.append(out[0] if isinstance(out, tuple) else out))
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)["loss"]
    h.remove()
    (g,) = torch.autograd.grad(loss, caught[0])
    return g


@cuda
def test_a_probe_backward_outside_the_scope_never_reaches_grad(monkeypatch, deterministic):
    """Key on, a probe forward+``autograd.grad`` outside ``train_backward_scope``: no covered
    ``p.grad`` is set (not now, and not at the next step) and the returned grad equals the key
    off one bit for bit. The old code ran the op for any grad-mode bf16 read and bound its dW."""
    cfg0, rt0 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN)
    cfg1, rt1 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_SPLIT)
    batch = _batches(rt0, 3)[0]
    _explicit_up(monkeypatch)
    m0, _o0, _ = _build(cfg0, rt0)
    m1, _o1, mgr = _build(cfg1, rt1)
    covered = _covered(m1)
    assert covered
    g0 = _probe_grads(m0, batch)
    g1 = _probe_grads(m1, batch)
    assert mgr.deferred == 0 and mgr.joins == 0
    assert all(p.grad is None for p in covered)
    assert _eq(g0, g1)


@cuda
def test_a_double_backward_through_a_covered_linear_works(monkeypatch, deterministic):
    """``create_graph=True`` (core_jacobian.py's use) through a covered ternary linear outside
    the scope: no error, and both the first and the second derivative equal the key-off ones."""
    cfg0, rt0 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN)
    cfg1, rt1 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_SPLIT)
    m0, _o0, _ = _build(cfg0, rt0)
    m1, _o1, mgr = _build(cfg1, rt1)

    name = next(n for n, mm in m1.named_modules() if getattr(mm, "_split", None) is not None)

    def run(m):
        mod = dict(m.named_modules())[name]
        torch.manual_seed(3)
        x = torch.randn(24, mod.in_features, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x)
            (g,) = torch.autograd.grad((y.float() ** 2).sum(), x, create_graph=True)
            (gg,) = torch.autograd.grad((g.float() ** 2).sum(), x)
        return g, gg

    a, b = run(m0), run(m1)
    assert mgr.deferred == 0
    assert _eq(a[0], b[0]) and _eq(a[1], b[1])


# ── 5. the hinge stream is a real second stream ─────────────────────────────────────────


@cuda
def test_the_hinge_runs_on_its_own_stream_and_the_main_stream_joins(monkeypatch, deterministic):
    """The hinge's kernels run on the side stream (a probe in `_slot_gain_penalty` reads the
    current stream), made slow so a missing join would show in the loss."""
    from morph.model.streams import side_stream
    from morph.model.transformer import MORPHTransformer
    cfg0, rt0 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_HINGE_ON)
    cfg1, rt1 = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_HINGE_ON,
                         *_HINGE_STREAM)
    batches = _batches(rt0, 3)
    _m, _g, ref = _run_eager(cfg0, rt0, batches, 2)
    seen = []
    orig = MORPHTransformer._slot_gain_penalty

    def slow(self, *a, **k):
        seen.append(torch.cuda.current_stream())
        torch.cuda._sleep(400_000_000)         # the hinge's result is LATE
        return orig(self, *a, **k)
    monkeypatch.setattr(MORPHTransformer, "_slot_gain_penalty", slow)
    _m1, _g1, got = _run_eager(cfg1, rt1, batches, 2)
    side = side_stream("slot_gain", torch.device("cuda"))
    assert seen and all(s == side for s in seen), seen
    for s, (a, b) in enumerate(zip(ref, got)):
        _same(a, b, f"slow hinge stream step {s}")


# ── 6. refusals at build ──────────────────────────────────────────────────────────────


@cuda
def test_split_wgrad_refuses_without_the_step_cache(monkeypatch):
    from morph.model.split_wgrad import bind_split_wgrad
    from test_graph_step import _build as _base_build
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_COMPACT)
    m, _opt = _base_build(cfg, rt)
    with pytest.raises(ValueError, match="step-cached TernarySTE"):
        bind_split_wgrad(m)


@cuda
def test_configs_refuse(monkeypatch):
    for over, needle in [
        (("model.split_wgrad=true",), "ternary_step_cache"),
        (("model.slot_gain_stream=true",), "slot_gain_no_ckpt"),
        (("model.slot_gain_stream=true", "model.slot_gain_no_ckpt=true",
          "model.slot_gain_lambda=0.0"), "slot_gain_lambda > 0"),
    ]:
        with pytest.raises((ValueError, NotImplementedError), match=needle):
            cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *over)
            from test_graph_step import _build as _base_build
            _base_build(cfg, rt)


@cuda
def test_a_read_allocates_nothing_on_the_side_stream():
    """The side-stream work of a read is one GEMM into a persistent scratch and one sum into a
    persistent buffer: the allocator's allocation count over 6 reads' weight grads (backward
    with the join) grows only by what the autograd graph and the held refs cost, and does not
    grow with the number of reads' GEMM outputs. Measured as: the count over the backward of 6
    reads minus the count of 3 reads is the cost of 3 reads' own dx/dy tensors (same in the
    key-off backward), never a [N,K] per read."""
    torch.manual_seed(0)
    w = torch.randn(48, 32, device="cuda", requires_grad=True)
    w2 = w.detach().clone().requires_grad_()

    def count(n, split):
        w.grad = None
        xs = [torch.randn(40, 32, device="cuda").requires_grad_() for _ in range(n)]
        mgr = _SlowWgrad.make(1000)
        torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (_reads(mgr, w, xs, False) if split else
                    sum((torch.nn.functional.linear(x.to(torch.bfloat16),
                                                    w2.to(torch.bfloat16)).float() ** 2).sum()
                        for x in xs))
        torch.cuda.synchronize()
        a0 = torch.cuda.memory_stats()["allocation.all.allocated"]
        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["allocation.all.allocated"] - a0

    count(2, True), count(2, False)           # warm: kernels, the buffers
    per_read_split = (count(6, True) - count(3, True)) / 3
    per_read_plain = (count(6, False) - count(3, False)) / 3
    # the plain backward allocates dx, dW (bf16) and the fp32 cast per read; the split one may
    # not exceed it by a per-read [N,K] (it allocates fewer)
    # Measured 2026-10-12: 6 per read with the persistent scratch and buffer (the plain
    # backward: 8); an [N,K] GEMM output per read (the sabotage) makes it 7.
    assert per_read_split <= 6.0 and per_read_split < per_read_plain, (per_read_split,
                                                                       per_read_plain)


# ── 7. model.split_wgrad_triton: Triton GEMMs inside the split op ───────────────────────
#
# NOT bit-exact against cuBLAS (another accumulation order), so the contracts are: each GEMM
# is as close to an fp64 reference as cuBLAS is; the wgrad epilogue equals "bf16 round, then
# copy / fp32 add" of the same kernel's plain output, bit for bit; every config gives the same
# bits; an untuned shape under capture raises; the op's gradients match the cuBLAS split op;
# the captured replay of the Triton arm equals its eager run bit for bit.
# Sabotage checks, run and reverted (session report): (a) MODE_ADD stores without the old value
# -> test_wgrad_epilogue; (b) the tuned-key lookup is skipped inside a capture -> the raise
# test; (c) a config with a K block that skips the tail mask -> the parity test.

_REAL_SHAPES = [(1536, 5632, 1024), (1104, 1024, 2816), (408, 1280, 1024), (1464, 1024, 2816)]


def _ref64(a, b):
    return a.double() @ b.double()


@cuda
@pytest.mark.parametrize("m,n,k", _REAL_SHAPES, ids=lambda v: str(v))
def test_triton_gemms_are_as_close_to_fp64_as_cublas(m, n, k):
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.03).to(torch.bfloat16)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    cases = {
        "fwd": (sg.linear_fwd(x, w), x @ w.t(), _ref64(x, w.t())),
        "dgrad": (sg.linear_dgrad(dy, w), dy @ w, _ref64(dy, w)),
    }
    buf = torch.empty(n, k, device="cuda")
    sg.linear_wgrad(dy, x, buf, add=False)
    cases["wgrad"] = (buf, (dy.t() @ x).float(), _ref64(dy.t(), x))
    for name, (tri, cub, ref) in cases.items():
        e_tri = (tri.double() - ref).abs().mean().item()
        e_cub = (cub.double() - ref).abs().mean().item()
        scale = ref.abs().mean().item()
        assert e_tri <= e_cub * 1.05 + 1e-6 * scale, (name, e_tri, e_cub)
        assert e_tri < 0.01 * scale, (name, e_tri, scale)       # not just "no worse": close


@cuda
def test_wgrad_epilogue_is_round_to_bf16_then_copy_then_fp32_add():
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(1)
    n, k = 1024, 2816
    xs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * s
          for m, s in ((408, 1e3), (624, 1.0), (33, 1e-3))]
    dys = [torch.randn(x.shape[0], n, device="cuda", dtype=torch.bfloat16) for x in xs]
    plain = []
    for dy, x in zip(dys, xs):
        g = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
        sg.tuned_mm(dy.t(), x, g)                       # the same kernel, no epilogue
        plain.append(g)
    buf = torch.full((n, k), float("nan"), device="cuda")      # the first read must overwrite it
    expect = None
    for i, (dy, x) in enumerate(zip(dys, xs)):
        sg.linear_wgrad(dy, x, buf, add=i > 0)
        expect = plain[0].float() if i == 0 else expect + plain[i].float()
        assert torch.equal(buf, expect), i


@cuda
def test_every_config_gives_the_same_bits():
    from morph.kernels.triton import split_gemm as sg
    torch.manual_seed(2)
    a = torch.randn(1104, 2816, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2816, 1024, device="cuda", dtype=torch.bfloat16)
    outs = []
    for cfg in sg.CONFIGS:
        c = torch.empty(1104, 1024, device="cuda", dtype=torch.bfloat16)
        sg._launch(a, b, c, sg.MODE_PLAIN, cfg)
        outs.append(c)
    assert all(torch.equal(outs[0], o) for o in outs[1:])


@cuda
def test_an_untuned_shape_under_capture_raises_and_a_tuned_one_replays():
    from morph.kernels.triton import split_gemm as sg
    x = torch.randn(520, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1280, 1024, device="cuda", dtype=torch.bfloat16)
    xo = torch.randn(528, 1024, device="cuda", dtype=torch.bfloat16)       # never tuned
    ref = sg.linear_fwd(x, w)                                              # tunes (eager)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = sg.linear_fwd(x, w)                    # tuned: allowed
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, ref)
    g2 = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="not tuned"):
        with torch.cuda.graph(g2):
            sg.linear_fwd(xo, w)
    torch.cuda.synchronize()
    sg.linear_fwd(xo, w)                           # the capture failure left the process usable


def _triton_mgr():
    from morph.model.split_wgrad import WgradStream
    from morph.model.streams import side_stream
    return WgradStream(side_stream("wgrad-test-triton"), strict=True, triton=True)


@cuda
def test_the_split_op_with_triton_matches_the_cublas_op():
    """Three reads of one weight through the Triton op: dx and the summed weight grad are close
    to the cuBLAS split op's, and the Triton arm allocates no GEMM scratch."""
    from morph.model.split_wgrad import SplitWeight, as_bf16, split_linear
    torch.manual_seed(3)
    w0 = torch.randn(1280, 1024, device="cuda")
    xs = [torch.randn(m, 1024, device="cuda") for m in (408, 520, 33)]
    res = {}
    for name, mgr in (("cublas", _SlowWgrad.make(1000)), ("triton", _triton_mgr())):
        w = w0.clone().requires_grad_()
        sw = SplitWeight(mgr, name, [w], [(0, 1280)], 1024, kind="callable",
                         source=lambda w=w: w.detach().to(torch.bfloat16))
        xr = [x.clone().requires_grad_() for x in xs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = sum((split_linear(as_bf16(x), sw).float() ** 2).sum() for x in xr)
        loss.backward()
        res[name] = (loss.item(), [x.grad.float() for x in xr], w.grad.clone(), mgr)
    assert not res["triton"][3]._scratch and res["cublas"][3]._scratch
    assert abs(res["triton"][0] - res["cublas"][0]) < 2e-2 * abs(res["cublas"][0])
    for gt, gc in zip(res["triton"][1], res["cublas"][1]):
        assert (gt - gc).abs().max() < 3e-2 * gc.abs().max()
    wt, wc = res["triton"][2], res["cublas"][2]
    assert (wt - wc).abs().max() < 3e-2 * wc.abs().max()


@cuda
def test_triton_key_refuses_without_split_wgrad(monkeypatch):
    from test_graph_step import _build as _base_build
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN,
                       "model.split_wgrad_triton=true")
    with pytest.raises(ValueError, match="needs model.split_wgrad=true"):
        _base_build(cfg, rt)


@cuda
def test_triton_arm_close_to_cublas_arm_and_graph_replay_equals_its_eager(monkeypatch,
                                                                          deterministic):
    """The whole tiny model: three eager steps of split_wgrad+triton track the cuBLAS split arm
    (loss and every gradient close), and a captured-and-replayed Triton run equals the eager
    Triton run bit for bit (the tuning ran in the eager steps)."""
    from morph.training.graph_step import GraphStep
    plan = [(True, "eager"), (False, "eager"), (False, "capture"), (False, "replay"),
            (True, "replay"), (False, "replay")]
    arm_t = _SPLIT + ("model.split_wgrad_triton=true",)
    cfg_c, rt_c = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_HINGE_ON, *_SPLIT)
    cfg_t, rt_t = _compose(monkeypatch, "lxtul_pointer_ditto", *_ON, *_PLAIN, *_HINGE_ON, *arm_t)
    batches = _batches(rt_c, len(plan))
    with monkeypatch.context() as mp:
        _explicit_up(mp)
        _mc, _gc, ref_c = _run_eager(cfg_c, rt_c, batches, len(plan), kinds=[k for k, _ in plan])
        _mt, mgr_t, ref_t = _run_eager(cfg_t, rt_t, batches, len(plan), strict=True,
                                       kinds=[k for k, _ in plan])
    assert mgr_t.triton and mgr_t.deferred > 0
    # step 0 starts from equal weights: loss and the per-parameter grads are close
    assert abs(ref_t[0]["loss"].item() - ref_c[0]["loss"].item()) < 1e-2 * abs(ref_c[0]["loss"].item())
    for a, b in zip(ref_c[0]["grads"], ref_t[0]["grads"]):
        if a is None:
            assert b is None
            continue
        assert (a.float() - b.float()).abs().max() <= 5e-2 * a.float().abs().max() + 1e-8
    del _mc, _mt

    mb, ob, mgr = _build(cfg_t, rt_t)
    clip = float(cfg_t.training.grad_clip)
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
            _same(ref_t[s], _snap(mb, ob, out, loss, gn, [p.grad for p in mb.parameters()]),
                  f"triton step {s} ({how})")
            mb._gain_pass_last = None
    assert gs.stats.replayed == 4


@cuda
def test_the_recording_tunes_the_keys_the_warm_up_never_ran(monkeypatch, deterministic):
    """GraphStep records every variant on one step, but the eager warm-up ran only the one the
    first draws picked. The other variant's GEMM shapes are untuned when the recording reaches
    them: collect mode lets pass one finish, the keys are tuned on synthetic tensors, the
    graphs are recorded again, and every replay equals the eager run bit for bit."""
    from morph.kernels.triton import split_gemm as sg
    from morph.training.graph_step import GraphStep
    plan = [(True, "eager"), (False, "eager"), (False, "capture"), (False, "replay"),
            (True, "replay"), (False, "replay")]
    over = _ON + _PLAIN + _HINGE_ON + _SPLIT + ("model.split_wgrad_triton=true",)
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *over)
    batches = _batches(rt, len(plan))
    sg.clear_tuned()
    with monkeypatch.context() as mp:
        _explicit_up(mp)
        # eager reference: the pass pinned to the first choice, the way the replays draw it
        m0, opt0, _ = _build(cfg, rt)
        choices = m0.gain_pass_choices()
        if len(choices) < 2:                   # the fixture's hinge reads a drawn pass
            cfg2, rt2 = _compose(monkeypatch, "lxtul_pointer_ditto", *over)
            m0, opt0, _ = _build(cfg2, rt2)
            choices = m0.gain_pass_choices()
        assert len(choices) >= 2, choices
        v0, v1 = choices[0], choices[-1]
        m0.pin_gain_pass(v0)
        torch.manual_seed(1234)
        clip = float(cfg.training.grad_clip)
        ref = []
        for s in range(len(plan)):
            m0._train_instruments = bool(plan[s][0])
            for g in opt0.param_groups:
                g["lr"] = 1e-3 * (1 + 0.1 * s)
            opt0.write_step_scalars()
            out, loss, gn = _eager_step(m0, opt0, *batches[s], clip)
            ref.append(_snap(m0, opt0, out, loss, gn, [p.grad for p in m0.parameters()]))
        del m0, opt0
        sg.clear_tuned()                       # the reference run's tuning is not the test's
        mb, ob, _mgr = _build(cfg, rt)
        tuned_n = []
        real_tune = sg.CAPTURE_TUNER.tune
        mp.setattr(sg.CAPTURE_TUNER, "tune",
                   lambda: tuned_n.append(real_tune()) or tuned_n[-1], raising=False)
        gs = GraphStep(mb, ob, grad_clip=clip, forward=_fwd(mb), kinds=(False, True),
                       set_kind=lambda k: setattr(mb, "_train_instruments", k),
                       loss_terms=_zero_pen, after_step=mb.tul_fan_after_step, first_capture=2,
                       keep_grads=(True, False), compiled=False, variants=(v0, v1),
                       draw_variant=lambda: v0, set_variant=mb.pin_gain_pass,
                       capture_tuner=sg.CAPTURE_TUNER)
        torch.manual_seed(1234)
        for s, ((kind, how), (x, y, lay)) in enumerate(zip(plan, batches)):
            mb._train_instruments = kind
            for g in ob.param_groups:
                g["lr"] = 1e-3 * (1 + 0.1 * s)
            ob.write_step_scalars()
            ob.zero_grad(set_to_none=True)
            if how == "capture":
                # The fixture's two passes have the same GEMM shapes, so "the warm-up never ran
                # these keys" is forced: the tuned cache is emptied before the recording.
                sg.clear_tuned()
            out, loss, gn, mode = gs.step(s, kind, x, y, lay, context="tul")
            assert mode == how, (s, mode, how)
            _same(ref[s], _snap(mb, ob, out, loss, gn, [p.grad for p in mb.parameters()]),
                  f"retuned step {s} ({how})")
            mb._gain_pass_last = None
    assert tuned_n and tuned_n[0] > 0, tuned_n     # the recording did meet untuned keys
    assert sg.CAPTURE_TUNER.pending() == 0
    assert gs.stats.replayed == 4 and len(gs.graphs) == 4
