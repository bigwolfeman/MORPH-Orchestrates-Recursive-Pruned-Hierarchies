"""model.hc_bwd_fused: the region entry backward with its projection term on tensor cores.

The key swaps ONE kernel of the entry backward (``_hcr_bwd_carrier_kernel`` ->
``_hcr_bwd_carrier_tc_kernel``): the carrier grad's projection term ``graw @ W`` runs as a bf16
MMA of W (the bf16 copy the forward projection used) with graw split into bf16 hi + lo parts.
The forward, the reduce and mapping kernels, and the weight and bias grads are untouched.
Contract (hcbwd, 2026-10-13):

  1. FORWARD outputs are bit-identical with the key on (the key does not touch the forward).
  2. The carrier grad stays inside the old kernel's error against a float64 reference, for
     fp32 and bf16 carriers, at the production shapes (7680 and 1536 positions), a partial
     token tile (111) and a grid smaller than a tile (5).
  3. grad_w / grad_b (formed from the same ``graw``) are bit-identical to the old kernel's.
  4. Through ``HyperConnectionResidual`` (``bwd_fused=True``), with and without
     ``block_custom_ops``, the full chain matches the key-off chain.
  5. Refusals: the constructor without ``region_fused``; the entry on the reference path.
  6. The default is off.
"""

import pytest
import torch

from morph.kernels.triton import fused_hyper_connection as K
from morph.model.hyper_connections import HyperConnectionResidual

C = 1024
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CASES = [(6, 1280), (6, 256), (3, 37), (1, 5)]


def _hc(device="cuda", bf16_weight=False, **kw):
    torch.manual_seed(1)
    hc = HyperConnectionResidual(C, n_streams=4, use_kernel=True, region_fused=True, **kw).to(device)
    with torch.no_grad():                       # a mapping far from the identity init
        hc.proj.weight.normal_(0, 3.0 / (4 * C) ** 0.5)
        hc.proj.bias.normal_(0, 0.5)
        if bf16_weight:                         # W exactly representable in bf16
            hc.proj.weight.copy_(hc.proj.weight.bfloat16().float())
    return hc


def _err(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def _entry_exit(hc, h0, gx, gout, y, term, fused):
    """entry -> exit; backward from (gx, gout). Returns forward outputs and grads."""
    h = h0.clone().requires_grad_(True)
    w = hc.proj.weight.detach().clone().requires_grad_(True)
    b = hc.proj.bias.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xbar, hres, hpr, link = K.hc_region_entry(h, w, b, hc.tau, hc.cayley_alpha, 3, hc.eps,
                                                  h.dtype == torch.bfloat16, fused)
        out = K.hc_region_exit(hres, hpr, link, y, term, h.dtype == torch.bfloat16)
    torch.autograd.backward([xbar, out], [gx, gout])
    return [xbar.detach(), hres.detach(), hpr.detach(), out.detach()], [h.grad, w.grad, b.grad]


def _ref64(hc, h0, gx, gout, y, term):
    h = h0.double().requires_grad_(True)
    w = hc.proj.weight.detach().double().requires_grad_(True)
    b = hc.proj.bias.detach().double().requires_grad_(True)
    xb, hres, hpr = K.hc_pre_map_reference(h, w, b, hc.tau, hc.cayley_alpha, 3, hc.eps)
    out = K.hc_post_reference(hres, hpr, h, y.double(), None if term is None else term.double())
    torch.autograd.backward([xb, out], [gx.double(), gout.double()])
    return h.grad, w.grad, b.grad


def _data(B, S, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    h = (torch.randn(B, S, 4, C, device="cuda", generator=g) * 2).to(dtype)
    gx = torch.randn(B, S, C, device="cuda", generator=g).to(dtype)
    gout = torch.randn(B, S, 4, C, device="cuda", generator=g).to(dtype)
    y = torch.randn(B, S, C, device="cuda", generator=g).bfloat16()
    term = torch.randn(B, S, C, device="cuda", generator=g).bfloat16()
    return h, gx, gout, y, term


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S", CASES)
def test_fused_backward_matches_reference_and_leaves_the_rest_alone(B, S, dtype):
    hc = _hc()
    h, gx, gout, y, term = _data(B, S, dtype)
    f_off, g_off = _entry_exit(hc, h, gx, gout, y, term, False)
    f_on, g_on = _entry_exit(hc, h, gx, gout, y, term, True)
    for a, b in zip(f_off, f_on):                                    # 1. forward untouched
        assert torch.equal(a, b)
    ref = _ref64(hc, h, gx, gout, y, term)
    e_off, e_on = _err(g_off[0], ref[0]), _err(g_on[0], ref[0])
    d = _err(g_on[0], g_off[0].double())
    print(f"  {dtype} T={B * S}: grad_h err old {e_off:.3e} new {e_on:.3e} (new vs old {d:.3e})")
    assert g_on[0].dtype == h.dtype and g_on[0].shape == h.shape
    # 2. Measured 2026-10-13: the new error is the old one plus 2.0e-4 .. 2.3e-4 (the bf16
    # rounding of W, 1.0e-3 relative to the old grad, added in quadrature to a 2.5e-3 error the
    # forward's bf16 projection already puts on both). A wrong tile or a dropped term moves it
    # by 10x or more.
    assert e_on <= e_off * 1.02 + 3e-4, (e_on, e_off)
    assert torch.equal(g_off[1], g_on[1]) and torch.equal(g_off[2], g_on[2])      # 3.


@cuda
@pytest.mark.parametrize("B,S", [(6, 1280), (6, 256), (3, 37)])
def test_the_projection_term_is_what_changed(B, S):
    """A guard with teeth. With W exactly representable in bf16 the two kernels compute the
    same fp32 sum up to the hi/lo split of graw (2^-17) and summation order: fp32 carriers
    agree to 1e-5. Dropping the lo half of graw alone moves the carrier grad by ~1e-3, and a
    wrong W tile by O(1)."""
    hc = _hc(bf16_weight=True)
    h, gx, gout, y, term = _data(B, S, torch.float32)
    _, g_off = _entry_exit(hc, h, gx, gout, y, term, False)
    _, g_on = _entry_exit(hc, h, gx, gout, y, term, True)
    rel = _err(g_on[0], g_off[0].double())
    print(f"  fp32 carrier, bf16-exact W, new vs old grad_h: {rel:.3e}")
    assert rel < 1e-5, rel


@cuda
@pytest.mark.parametrize("ops", [False, True])
def test_through_the_module(ops):
    """HyperConnectionResidual(bwd_fused=True), with and without the custom ops: the same
    chain as bwd_fused=False to the projection term's rounding."""
    kw = dict(carrier_bf16=True, block_custom_ops=ops)
    h0 = torch.randn(2, 96, 4, C, device="cuda").bfloat16()
    wf = (torch.randn(C, C, device="cuda") / C ** 0.5).bfloat16()
    r = torch.randn(2, 96, 4, C, device="cuda")
    res = []
    for fused in (False, True):
        hc = _hc(bwd_fused=fused, **kw)
        h = h0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = hc(h, lambda x: x @ wf)
        (out.float() * r).sum().backward()
        res.append((out.detach(), h.grad, hc.proj.weight.grad, hc.proj.bias.grad))
    assert torch.equal(res[0][0], res[1][0])
    assert _err(res[1][1], res[0][1].double()) < 1e-2
    # the weight grads see a changed carrier grad only through earlier layers: here none.
    assert torch.allclose(res[0][2], res[1][2], rtol=2e-2, atol=1e-3)


def test_constructor_refuses_without_region():
    with pytest.raises(ValueError, match="hc_bwd_fused"):
        HyperConnectionResidual(C, n_streams=4, use_kernel=True, region_fused=False,
                                bwd_fused=True)


def test_entry_refuses_the_reference_path():
    h = torch.randn(1, 4, 4, C)
    w = torch.randn(48, 4 * C)
    b = torch.randn(48)
    with pytest.raises(RuntimeError, match="hc_bwd_fused"):
        K.hc_region_entry(h, w, b, 1.0, 0.1, 3, 1e-6, False, True)    # CPU, no autocast


def test_default_is_off():
    from morph.model.transformer import MORPHConfig
    assert MORPHConfig.__dataclass_fields__["hc_bwd_fused"].default is False
    assert HyperConnectionResidual(C, region_fused=True)._bwd_fused is False


@pytest.mark.parametrize("n", [2, 8, 16])
def test_other_stream_counts_are_refused(n):
    """The region kernels (and so this key) unroll the mapping for n = 4 (a 4x4 Cayley block, 20
    reductions, 48 projection rows). Another n must raise at build, never run a wrong kernel."""
    with pytest.raises(ValueError, match="hc_region_fused"):
        HyperConnectionResidual(C, n_streams=n, use_kernel=True, region_fused=True, bwd_fused=True)
