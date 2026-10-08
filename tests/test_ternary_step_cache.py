"""``model.ternary_step_cache`` (2026-10-07, task 2.3 of
.agents/notes/proposed/architecture/2026-10-07-graph-captured-training-step.md).

``morph.model.ternary_qat.TernaryStepCache`` computes every bound ternary weight's bf16 form
once per forward and hands it to every read; the gradient reaches the fp32 shadow per read.
The contract is BIT-IDENTITY with the per-access path (the key off).

What each test pins:
  1. Shared weights, two reads per forward, two optimizer steps (CPU, bf16 autocast, both
     closed-form rules): the loss, every gradient and every parameter after each step are
     equal bit for bit to the unbound twin's. A cache that is not refreshed after the
     optimizer step, or that cuts the gradient to the shadow weight, fails here.
  2. Reads outside bf16 autocast (saliency scoring, fp32 diagnostics) get the per-access
     fp32 value, not the cache.
  3. A deepcopy of a bound STE / model is unbound (the fan's EMA twin is a deepcopy: a bound
     copy would read a buffer nobody refreshes); ttq / dual are refused.
  4. CUDA, the tiny winner (`lxtul_pointer_graph` composed by Hydra: fan, latent-selected
     loop, pointer, gain hinge, the EMA twin, ternary STE) with its blocks compiled as the
     trainer compiles them: three training steps with the key on equal the key off bit for
     bit (loss, every gradient, every parameter, the twin). The compiled blocks' weights are
     filled by a compiled quantiser (an eager one differs from Inductor's in the scale's
     last ulp, measured 3/30 MLP draws).

Sabotage checks, run and reverted (session report, not committed):
  (a) `TernaryStepCache.refresh` fills only on its first call (no refresh after the
      optimizer step) -> tests 1 and 4 fail at the second step.
  (b) `_StepCachedWeight.backward` returns None for the shadow weight (the cache detached
      from it) -> tests 1 and 4 fail ("grad presence").
"""
from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from morph.model.ternary_qat import (
    TernarySTE,
    TernaryStepCache,
    bind_ternary_step_cache,
)

_BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float64: torch.int64}


def _eq(u: torch.Tensor, v: torch.Tensor) -> bool:
    """Bit equality (-0.0 differs from +0.0)."""
    if u.shape != v.shape or u.dtype != v.dtype:
        return False
    if u.dtype in _BITS:
        return torch.equal(u.contiguous().view(_BITS[u.dtype]), v.contiguous().view(_BITS[v.dtype]))
    return torch.equal(u, v)


class _Shared(nn.Module):
    """Two ternary Linears, each read twice per forward (a looped core in miniature), and a
    forward that refreshes the cache first, as ``MORPHTransformer._forward_single`` does."""

    def __init__(self, d: int, mode: str):
        super().__init__()
        self.a = nn.Linear(d, 2 * d, bias=False)
        self.b = nn.Linear(2 * d, d, bias=False)
        for lin in (self.a, self.b):
            P.register_parametrization(lin, "weight", TernarySTE(
                mode=mode, weight_shape=tuple(lin.weight.shape)))

    def forward(self, x):
        tsc = self.__dict__.get("_ternary_step_cache")
        if tsc is not None:
            tsc.refresh()
        h = x
        for _ in range(2):
            h = h + self.b(F.silu(self.a(h)))
        return h.float().pow(2).mean()


def _train(net: nn.Module, xs: list[torch.Tensor]) -> list[dict]:
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    snaps = []
    for x in xs:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = net(x)
        loss.backward()
        grads = [None if p.grad is None else p.grad.clone() for p in net.parameters()]
        opt.step()
        snaps.append({"loss": loss.detach().clone(), "grads": grads,
                      "params": [p.detach().clone() for p in net.parameters()]})
    return snaps


@pytest.mark.parametrize("mode", ["norm_match", "symmetric"])
def test_cached_equals_per_access_across_reads_and_steps(mode):
    torch.manual_seed(0)
    ref = _Shared(48, mode)
    net = copy.deepcopy(ref)
    cache = bind_ternary_step_cache(net, None)
    assert isinstance(cache, TernaryStepCache) and len(cache.entries) == 2
    assert all(not e[0] for e in cache.entries)                  # nothing compiled here
    xs = [torch.randn(4, 16, 48) for _ in range(3)]
    a, b = _train(ref, xs), _train(net, xs)
    for s, (u, v) in enumerate(zip(a, b)):
        assert _eq(u["loss"], v["loss"]), (s, "loss")
        for i, (gu, gv) in enumerate(zip(u["grads"], v["grads"])):
            assert (gu is None) == (gv is None), (s, "grad presence", i)
            assert gu is None or _eq(gu, gv), (s, "grad", i)
        for i, (pu, pv) in enumerate(zip(u["params"], v["params"])):
            assert _eq(pu, pv), (s, "param", i)
    # the weights moved, so a stale cache could not have passed
    assert not _eq(a[0]["params"][0], a[-1]["params"][0])


def test_reads_outside_bf16_autocast_are_per_access():
    torch.manual_seed(0)
    ref = _Shared(32, "norm_match")
    net = copy.deepcopy(ref)
    bind_ternary_step_cache(net, None)
    w_ref, w = ref.a.weight, net.a.weight            # no autocast: the fp32 per-access value
    assert w.dtype == torch.float32 and _eq(w, w_ref)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        wc = net.a.weight                           # bf16 autocast: the cache
        assert wc.dtype == torch.bfloat16
        assert _eq(wc, w_ref.detach().to(torch.bfloat16))
    with torch.autocast("cpu", dtype=torch.float16):
        assert net.a.weight.dtype == torch.float32  # another autocast dtype: per-access


def test_copies_are_unbound_and_learnable_scales_refused():
    torch.manual_seed(0)
    net = _Shared(32, "norm_match")
    bind_ternary_step_cache(net, None)
    twin = copy.deepcopy(net)
    for lin in (twin.a, twin.b):
        ste = lin.parametrizations.weight[0]
        assert "_wq" not in ste._buffers and "_forward_live" not in ste.__dict__
        assert ste._forward_fn.__func__ is TernarySTE._forward_norm_match
        assert ste._forward_fn.__self__ is ste
    assert twin.__dict__["_ternary_step_cache"] is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert twin.a.weight.dtype == torch.float32              # per-access, then autocast
    # the original keeps its cache
    assert "_wq" in net.a.parametrizations.weight[0]._buffers
    # the twin's checkpoint names are the original's (non-persistent buffer)
    assert net.state_dict().keys() == twin.state_dict().keys()
    with pytest.raises(RuntimeError, match="already bound"):
        bind_ternary_step_cache(net, None)
    lin = nn.Linear(8, 8, bias=False)
    P.register_parametrization(lin, "weight", TernarySTE(
        mode="ttq", weight_shape=(8, 8), weight_init=lin.weight.detach()))
    with pytest.raises(ValueError, match="learnable"):
        bind_ternary_step_cache(lin, None)


# ── CUDA: the tiny winner with compiled blocks ──────────────────────────────────────────

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled blocks on CUDA")


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    yield
    torch.use_deterministic_algorithms(prev)


def _winner(monkeypatch, step_cache: bool):
    """train.py's build order at the tiny width, then the trainer's compile_blocks wrap and
    (key on) the bind, which comes after the wrap."""
    from test_graph_step import _build, _compose
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_graph",
                       f"model.ternary_step_cache={str(step_cache).lower()}")
    m, opt = _build(cfg, rt)
    for group in (m.prelude, m.core, m.coda):
        for i in range(len(group)):
            if getattr(group[i], "retention", None) is not None:      # as train.py does
                group[i].retention.forward = torch.compiler.disable(group[i].retention.forward)
            group[i] = torch.compile(group[i], mode="default",
                                     dynamic=None if group is not m.core else False)
    if step_cache:
        assert m.cfg.ternary_step_cache
        cache = bind_ternary_step_cache(m, "default")
        n_c = sum(1 for e in cache.entries if e[0])
        assert 0 < n_c < len(cache.entries), "both fills must be exercised"
    return cfg, rt, m, opt


@cuda
def test_tiny_winner_compiled_blocks_bit_identical(monkeypatch, deterministic):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from test_graph_step import _batches, _eager_step

    runs = []
    for on in (False, True):
        cfg, rt, m, opt = _winner(monkeypatch, on)
        batches = _batches(rt, 3)
        torch.manual_seed(1234)
        snaps = []
        for x, y, lay in batches:
            m._train_instruments = False
            opt.write_step_scalars()
            out, loss, gn = _eager_step(m, opt, x, y, lay, float(cfg.training.grad_clip))
            tw = m.__dict__.get("_fan_target")
            snaps.append({
                "loss": loss.clone(), "gn": gn.clone(),
                "grads": [None if p.grad is None else p.grad.clone() for p in m.parameters()],
                "params": [p.detach().clone() for p in m.parameters()],
                "twin": [t.detach().clone() for t in tw.parameters()] if tw is not None else [],
            })
        runs.append(snaps)
        torch._dynamo.reset()
    for s, (u, v) in enumerate(zip(*runs)):
        assert _eq(u["loss"], v["loss"]), (s, "loss", u["loss"], v["loss"])
        assert _eq(u["gn"], v["gn"]), (s, "gnorm", u["gn"], v["gn"])
        for i, (gu, gv) in enumerate(zip(u["grads"], v["grads"])):
            assert (gu is None) == (gv is None), (s, "grad presence", i)
            assert gu is None or _eq(gu, gv), (s, "grad", i)
        for i, (pu, pv) in enumerate(zip(u["params"], v["params"])):
            assert _eq(pu, pv), (s, "param", i)
        assert len(u["twin"]) > 0
        for i, (tu, tv) in enumerate(zip(u["twin"], v["twin"])):
            assert _eq(tu, tv), (s, "twin", i)
