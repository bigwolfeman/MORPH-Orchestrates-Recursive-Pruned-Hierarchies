"""``model.fp8_ternary_fwd`` (2026-10-12): the ternary MLP forward GEMM on FP8 tensor cores.

``bind_ternary_step_cache(model, mode, fp8_fwd=True)`` gives every dense ternary
``CMSBlockLinear`` the STE's codes as e4m3 plus a per-row fp32 scale, refilled with the bf16
step cache; the layer forward is then ``_TernaryFP8Fwd`` (rowwise e4m3 activations,
``torch._scaled_mm``). Needs a CUDA device (sm_89+); every test skips without one.

What each test pins:
  1. Forward: against the bf16 step-cache twin, relative Frobenius error of a two-layer MLP is
     below 0.06. Why that bound: e4m3 keeps 3 mantissa bits, so one activation is rounded by
     at most 2^-4 = 6.25 % (rms ~1.8 %); the K-term sum averages the independent roundings, so
     a layer's relative output error sits near the rms (measured 2.6 % one layer, 3.8 % two with
     a SiLU between). The codes and the scale are exact, so nothing else contributes. A wrong
     scale, a wrong code or a transposed weight lands at 0.5-1.0. The output must also DIFFER
     from the twin's (the FP8 GEMM ran) and `torch._scaled_mm` must be called once per layer.
  2. Backward: for the same input and the same upstream gradient, dX and the shadow weight's
     gradient are bit-equal to the bf16 twin's (the backward is the F.linear backward on the
     saved bf16 x and w).
  3. Refresh: after the shadow weight changes, the next forward's e4m3 codes and scales follow
     (a cache filled once would not).
  4. Key off: a step-cached model without the key is bit-equal to the plain F.linear model.
  5. Refusals: bias, carved layer, dims not divisible by 16, no CMSBlockLinear in the model,
     an unbound deep copy (the fan's EMA twin) takes the F.linear path.
  6. torch.compile and CUDA-graph capture of forward + backward match the eager FP8 result.

Sabotage checks, run and reverted (session report, not committed):
  (a) `_TernaryFP8Fwd.forward` passes `scale_b=ws * 2` -> test 1 fails (error ~1.0).
  (b) `_fill_fp8_codes` skips the `w8.copy_` (stale codes) -> tests 1 and 3 fail.
  (c) `_TernaryFP8Fwd.backward` returns `go.mm(w8.float().to(bf16))` for dX -> test 2 fails.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from morph.model.layers.block_sparse import CMSBlockLinear
from morph.model.ternary_qat import TernarySTE, bind_ternary_step_cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="torch._scaled_mm e4m3 needs a CUDA device with FP8 tensor cores")

D, DFF = 256, 704   # divisible by 16; small so the file is quick


class _Mlp(nn.Module):
    """A SwiGLU-free two-layer MLP of dense ternary CMSBlockLinear, refreshing the cache first
    as ``MORPHTransformer._forward_single`` does."""

    def __init__(self, bias: bool = False, d: int = D, dff: int = DFF):
        super().__init__()
        self.up = CMSBlockLinear(d, dff, bias=bias, density=1.0, device="cuda")
        self.down = CMSBlockLinear(dff, d, bias=bias, density=1.0, device="cuda")
        for lin in (self.up, self.down):
            nn.init.normal_(lin.weight, std=0.05)
            P.register_parametrization(lin, "weight", TernarySTE(
                mode="norm_match", weight_shape=tuple(lin.weight.shape), device="cuda"))

    def forward(self, x):
        tsc = self.__dict__.get("_ternary_step_cache")
        if tsc is not None:
            tsc.refresh()
        return self.down(F.silu(self.up(x)))


def _pair(seed: int = 0):
    torch.manual_seed(seed)
    net = _Mlp()
    twin = copy.deepcopy(net)               # same weights; deepcopy of an unbound model
    bind_ternary_step_cache(twin, None)
    bind_ternary_step_cache(net, None, fp8_fwd=True)
    return net, twin


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _fwd(net, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net(x)


def test_forward_matches_bf16_within_fp8_tolerance(monkeypatch):
    net, twin = _pair()
    x = torch.randn(96, D, device="cuda")
    calls = []
    real = torch._scaled_mm
    monkeypatch.setattr(torch, "_scaled_mm", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    y, y_ref = _fwd(net, x), _fwd(twin, x)
    assert len(calls) == 2                                   # one FP8 GEMM per layer
    assert y.dtype == y_ref.dtype == torch.bfloat16
    assert not torch.equal(y, y_ref)                         # the FP8 path ran
    err = _rel(y, y_ref)
    assert 1e-3 < err < 0.06, err


def test_backward_is_the_bf16_backward():
    net, twin = _pair(1)
    x = torch.randn(3, 40, D, device="cuda")                 # 3-D input as the blocks feed it
    x_a, x_b = x.clone().requires_grad_(), x.clone().requires_grad_()
    y, y_ref = _fwd(net, x_a), _fwd(twin, x_b)
    go = torch.randn_like(y_ref)
    y.backward(go)
    y_ref.backward(go)
    # Through two layers the gradients differ from the twin's by the forward's activation noise
    # (layer 2's saved input follows layer 1's FP8 forward); the backward ITSELF is pinned
    # bit-equal on one layer in the next test.
    for gn, gr in ((net.up.parametrizations.weight.original.grad,
                    twin.up.parametrizations.weight.original.grad),
                   (net.down.parametrizations.weight.original.grad,
                    twin.down.parametrizations.weight.original.grad),
                   (x_a.grad, x_b.grad)):
        assert gn is not None and torch.isfinite(gn).all()
        assert _rel(gn, gr) < 0.15


def test_single_layer_backward_is_bit_equal():
    torch.manual_seed(2)
    lin = CMSBlockLinear(D, DFF, bias=False, density=1.0, device="cuda")
    nn.init.normal_(lin.weight, std=0.05)
    P.register_parametrization(lin, "weight", TernarySTE(
        mode="norm_match", weight_shape=tuple(lin.weight.shape), device="cuda"))
    twin = copy.deepcopy(lin)

    class Host(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            self.__dict__["_ternary_step_cache"].refresh()
            return self.m(x)

    a, b = Host(lin), Host(twin)
    bind_ternary_step_cache(b, None)
    bind_ternary_step_cache(a, None, fp8_fwd=True)
    x = torch.randn(2, 33, D, device="cuda")
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    go = torch.randn(2, 33, DFF, device="cuda", dtype=torch.bfloat16)
    _fwd(a, xa).backward(go)
    _fwd(b, xb).backward(go)
    assert torch.equal(xa.grad, xb.grad)
    assert torch.equal(lin.parametrizations.weight.original.grad,
                       twin.parametrizations.weight.original.grad)


def test_codes_and_scales_follow_the_shadow_weight():
    net, _ = _pair(3)
    ste = net.up.parametrizations.weight[0]
    x = torch.randn(32, D, device="cuda")
    _fwd(net, x)
    w8_before, ws_before = ste._wq8.clone(), ste._ws.clone()
    wq = ste._wq
    # the cache IS scale * code: the e4m3 codes times the row scale rebuild the bf16 weight
    assert torch.equal((ste._wq8.float() * ste._ws.t()).to(torch.bfloat16), wq)
    assert set(ste._wq8.float().unique().tolist()) <= {-1.0, 0.0, 1.0}
    with torch.no_grad():
        net.up.parametrizations.weight.original.mul_(0.5)
        net.up.parametrizations.weight.original[:8].add_(0.3)
    _fwd(net, x)
    assert not torch.equal(ste._ws, ws_before)               # per-tensor scale moved
    assert not torch.equal(ste._wq8, w8_before)              # and some codes crossed the threshold
    assert torch.equal((ste._wq8.float() * ste._ws.t()).to(torch.bfloat16), ste._wq)


def test_key_off_is_bit_equal_to_plain_linear():
    torch.manual_seed(4)
    net = _Mlp()
    plain = copy.deepcopy(net)
    bind_ternary_step_cache(net, None)                       # step cache on, key off
    x = torch.randn(48, D, device="cuda")
    assert torch.equal(_fwd(net, x), _fwd(plain, x))
    assert not net.up._fp8_tfwd and not hasattr(net.up.parametrizations.weight[0], "_wq8")


def test_refusals():
    torch.manual_seed(5)
    # no CMSBlockLinear at all: the key must not be a silent no-op
    lin = nn.Linear(D, DFF, bias=False, device="cuda")
    P.register_parametrization(lin, "weight", TernarySTE(
        mode="norm_match", weight_shape=tuple(lin.weight.shape), device="cuda"))
    host = nn.ModuleList([lin])
    with pytest.raises(ValueError, match="nothing would run on FP8"):
        bind_ternary_step_cache(host, None, fp8_fwd=True)
    # bias
    net = _Mlp(bias=True)
    with pytest.raises(ValueError, match="bias"):
        bind_ternary_step_cache(net, None, fp8_fwd=True)
    # dims not divisible by 16
    bad = CMSBlockLinear(24, 40, bias=False, density=1.0, tile_size=8, device="cuda")
    P.register_parametrization(bad, "weight", TernarySTE(
        mode="norm_match", weight_shape=(40, 24), device="cuda"))
    with pytest.raises(ValueError, match="divisible by 16"):
        bind_ternary_step_cache(nn.ModuleList([bad]), None, fp8_fwd=True)
    # not ternary
    plain = CMSBlockLinear(D, DFF, bias=False, density=1.0, device="cuda")
    with pytest.raises(ValueError, match="not ternary"):
        plain.enable_fp8_ternary_fwd()
    # carved
    net, _ = _pair()
    with pytest.raises(ValueError, match="carve"):
        net.up.carve(blocking=128)
    # a ttq model has no closed-form cache at all
    t = nn.Linear(D, DFF, bias=False, device="cuda")
    P.register_parametrization(t, "weight", TernarySTE(
        mode="ttq", weight_shape=(DFF, D), device="cuda"))
    with pytest.raises(ValueError):
        bind_ternary_step_cache(nn.ModuleList([t]), None, fp8_fwd=True)


def test_unbound_deepcopy_takes_the_linear_path():
    net, _ = _pair(6)
    twin = copy.deepcopy(net)                                # the fan EMA twin's situation
    assert not hasattr(twin.up.parametrizations.weight[0], "_wq8")
    assert twin.up._fp8_tfwd                                 # flag copies; the buffers do not
    plain = copy.deepcopy(twin)
    x = torch.randn(16, D, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = twin.up(x)
    ref = _Mlp()
    ref.load_state_dict(net.state_dict())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = ref.up(x)
    assert torch.equal(y, y_ref)                             # live STE path, no FP8
    del plain


def test_compile_and_cuda_graph_match_eager():
    net, _ = _pair(7)
    x = torch.randn(64, D, device="cuda").to(torch.bfloat16)
    go = torch.randn(64, D, device="cuda", dtype=torch.bfloat16)

    def run(m):
        xi = x.clone().requires_grad_()
        y = _fwd(m, xi)
        y.backward(go)
        g = net.up.parametrizations.weight.original.grad.clone()
        net.up.parametrizations.weight.original.grad = None
        net.down.parametrizations.weight.original.grad = None
        return y.detach(), xi.grad.clone(), g

    y_e, gx_e, gw_e = run(net)
    cnet = torch.compile(net, mode="default")
    y_c, gx_c, gw_c = run(cnet)
    # Not bit-equal: Inductor fuses the SiLU into the row quantiser and drops the bf16 rounding
    # between them, and e4m3 rounding amplifies that (~3 % of the activations land on the other
    # side of a grid midpoint). The difference stays at the activation-rounding noise level.
    assert _rel(y_c, y_e) < 0.05 and _rel(gx_c, gx_e) < 0.15 and _rel(gw_c, gw_e) < 0.15
    twin = _pair(7)[1]
    assert _rel(y_c, _fwd(twin, x)) < 0.06                   # and it IS the FP8 forward

    # CUDA graph: forward + backward captured once, replayed with the weight changed in place
    xs = x.clone().requires_grad_()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            _fwd(net, xs).backward(go)
    torch.cuda.synchronize()
    for p in net.parameters():
        p.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = _fwd(net, xs)
        yg.backward(go)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(yg, y_e)                              # a captured graph runs the same kernels
    y_before = yg.clone()
    with torch.no_grad():
        net.up.parametrizations.weight.original.mul_(-1.0)   # codes flip sign; scale same
    g.replay()
    torch.cuda.synchronize()
    assert not torch.equal(yg, y_before)                     # the replay re-read the new codes
