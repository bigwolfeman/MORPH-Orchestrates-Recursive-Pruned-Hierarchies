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
    mgr = bind_split_wgrad(m, strict=strict) if bool(m.cfg.split_wgrad) else None
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
