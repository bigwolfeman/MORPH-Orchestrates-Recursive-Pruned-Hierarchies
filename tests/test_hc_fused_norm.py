"""model.hc_fused_norm / model.hc_fused_grad: the HC pre-map kernel with the sublayer's
RMSNorm folded in, and the projection-path carrier grad accumulated by the GEMM.

Contracts:
  * hc_fused_grad alone is BIT-IDENTICAL to the default kernel path (forward outputs and
    every gradient), at the winner's real shapes and at edge shapes.
  * hc_fused_norm is not bit-identical (the unfused norm is a separate kernel with its own
    reduction order). Against an fp64 reference its error is no worse than the unfused
    path's (default pre-map kernel, then attention.RMSNorm, then the bf16 cast autocast
    does at the next GEMM), for the forward and for every gradient: carrier, mapping
    projection weight and bias, norm weight.
  * On CPU (the composed fallback) both keys change nothing: MORPHBlock outputs and grads
    are bit-identical with the keys on and off.
"""

import pytest
import torch
import torch.nn as nn

from morph.model.attention import RMSNorm
from morph.model.hyper_connections import HyperConnectionResidual
from morph.model.mhc import MORPHBlock

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")

TAU, ALPHA, ITERS, EPS, NEPS = 1.0, 0.1, 3, 1e-6, 1e-6
# (B, S, C, carrier scale): the winner's prelude/coda and core shapes, then edges: a
# token count with no power-of-two factor (one token per backward program), a C below
# its pow2 block (masked lanes), a near-zero carrier, a large one.
SHAPES = [
    (6, 1280, 1024, 1.0),
    (6, 256, 1024, 1.0),
    (1, 7, 1024, 1.0),
    (2, 5, 768, 1.0),
    (2, 64, 1024, 1e-3),
    (2, 64, 1024, 300.0),
]


def _inputs(B, S, C, scale, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(B, S, 4, C, device="cuda", generator=g) * scale
    pw = torch.randn(48, 4 * C, device="cuda", generator=g) * 0.02
    pb = torch.randn(48, device="cuda", generator=g) * 0.05
    nw = 1.0 + 0.1 * torch.randn(C, device="cuda", generator=g)
    # Upstream grads, bf16-representable so the bf16 output's grad is exactly this.
    gx = torch.randn(B, S, C, device="cuda", generator=g).bfloat16().float()
    gres = torch.randn(B, S, 4, 4, device="cuda", generator=g)
    gpr = torch.randn(B, S, 4, device="cuda", generator=g)
    return h, pw, pb, nw, gx, gres, gpr


def _leaves(*ts, dtype=None):
    return [t.detach().clone().to(dtype or t.dtype).requires_grad_(True) for t in ts]


def _backward(outs, gx, gres, gpr):
    x, res, pr = outs
    loss = (x.to(gx.dtype) * gx).sum() + (res.to(gres.dtype) * gres).sum() \
        + (pr.to(gpr.dtype) * gpr).sum()
    loss.backward()


@cuda
@pytest.mark.parametrize("B,S,C,scale", SHAPES)
def test_fused_grad_is_bit_identical(B, S, C, scale):
    from morph.kernels.triton.fused_hyper_connection import (
        _FusedHCPreMap, _FusedHCPreMapFold)
    h, pw, pb, _nw, gx, gres, gpr = _inputs(B, S, C, scale)
    a = _leaves(h, pw, pb)
    b = _leaves(h, pw, pb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        oa = _FusedHCPreMap.apply(*a, TAU, ALPHA, ITERS, EPS)
        ob = _FusedHCPreMapFold.apply(*b, None, TAU, ALPHA, ITERS, EPS, NEPS, torch.bfloat16)
    for x, y in zip(oa, ob):
        assert x.dtype == y.dtype and torch.equal(x, y)
    _backward(oa, gx, gres, gpr)
    _backward(ob, gx, gres, gpr)
    for name, x, y in zip(("h", "proj_w", "proj_b"), a, b):
        assert torch.equal(x.grad, y.grad), f"grad {name} differs"


@cuda
def test_fused_grad_module_bit_identical_with_post_path():
    """The whole residual (pre-map, sublayer, post) with the carrier grad summed by autograd
    over the pre and post paths: bit-identical with model.hc_fused_grad on and off."""
    torch.manual_seed(0)
    C = 1024
    ref = HyperConnectionResidual(C, use_kernel=True).cuda()
    fg = HyperConnectionResidual(C, use_kernel=True, fused_grad=True).cuda()
    fg.load_state_dict(ref.state_dict())
    lin = nn.Linear(C, C, bias=False).cuda()
    h0 = torch.randn(6, 256, 4, C, device="cuda")
    gout = torch.randn(6, 256, 4, C, device="cuda")
    res = []
    for mod in (ref, fg):
        lin.zero_grad()
        h = h0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(h, lin)
        (out * gout).sum().backward()
        res.append((out.detach(), h.grad, mod.proj.weight.grad, mod.proj.bias.grad,
                    lin.weight.grad.clone()))
    for x, y in zip(*res):
        assert torch.equal(x, y)


def _fp64_truth(h, pw, pb, nw):
    from morph.kernels.triton.fused_hyper_connection import hc_pre_map_reference
    h, pw, pb, nw = _leaves(h, pw, pb, nw, dtype=torch.float64)
    xb, res, pr = hc_pre_map_reference(h, pw, pb, TAU, ALPHA, ITERS, EPS)
    xa = xb * xb.pow(2).mean(-1, keepdim=True).add(NEPS).rsqrt() * nw
    return (h, pw, pb, nw), (xa, res, pr)


def _err(x, ref):
    """Relative L2 error of x against the fp64 reference."""
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


@cuda
@pytest.mark.parametrize("B,S,C,scale", SHAPES)
def test_fused_norm_error_no_worse_than_unfused(B, S, C, scale):
    from morph.kernels.triton.fused_hyper_connection import (
        _FusedHCPreMap, _FusedHCPreMapFold)
    h, pw, pb, nw, gx, gres, gpr = _inputs(B, S, C, scale)

    # unfused: the default kernel, then RMSNorm as the block runs it, then autocast's cast.
    norm = RMSNorm(C, eps=NEPS).cuda()
    with torch.no_grad():
        norm.weight.copy_(nw)
    u = _leaves(h, pw, pb) + [norm.weight]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xb, ures, upr = _FusedHCPreMap.apply(*u[:3], TAU, ALPHA, ITERS, EPS)
        uout = (norm(xb).to(torch.bfloat16), ures, upr)
    # folded
    f = _leaves(h, pw, pb, nw)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fout = _FusedHCPreMapFold.apply(*f, TAU, ALPHA, ITERS, EPS, NEPS, torch.bfloat16)
    assert fout[0].dtype == torch.bfloat16
    t_leaves, tout = _fp64_truth(h, pw, pb, nw)

    _backward(uout, gx, gres, gpr)
    _backward(fout, gx, gres, gpr)
    _backward(tout, gx.double(), gres.double(), gpr.double())

    names = ("x_norm", "Hres", "Hpost_row")
    rows = [(n, _err(fo, t), _err(uo, t)) for n, fo, uo, t in zip(names, fout, uout, tout)]
    for n, fl, ul, tl in zip(("grad_h", "grad_proj_w", "grad_proj_b", "grad_norm_w"),
                             f, u, t_leaves):
        rows.append((n, _err(fl.grad, tl.grad), _err(ul.grad, tl.grad)))
    msg = "\n".join(f"{n:12s} fused {fe:.3e}  unfused {ue:.3e}" for n, fe, ue in rows)
    print(f"\n[B={B} S={S} C={C} scale={scale}]\n{msg}")
    for n, fe, ue in rows:
        # "no worse" with a 10 % band for the two paths' independent fp32 roundings, and
        # an absolute floor at fp32 resolution where both errors are at roundoff.
        assert fe <= 1.10 * ue + 2e-7, f"{n}: fused {fe:.3e} > unfused {ue:.3e}\n{msg}"
    # Both paths carry the bf16 projection GEMM's error (~1e-3), which would hide a
    # norm-sized bug in the bound above. So also bound their DIRECT gap: the bf16 output
    # differs only by rare rounding flips (measured <= 2.6e-5), the fp32 grads by fp32
    # roundoff (measured <= 4.4e-7).
    def gap(a, b):
        return ((a.double() - b.double()).norm() / b.double().norm()).item()
    assert gap(fout[0], uout[0]) <= 1e-4, gap(fout[0], uout[0])
    for n, fl, ul in zip(("grad_h", "grad_proj_w", "grad_proj_b", "grad_norm_w"), f, u):
        assert gap(fl.grad, ul.grad) <= 2e-6, f"{n} gap {gap(fl.grad, ul.grad):.3e}"


# ── CPU: the composed fallback leaves the block's arithmetic unchanged ────────────────

def _block(fused_norm, fused_grad, C=64):
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(C, 2 * C, bias=False), nn.SiLU(),
                        nn.Linear(2 * C, C, bias=False))

    class _Attn(nn.Module):           # any [B,S,C] -> [B,S,C] sublayer
        def __init__(self):
            super().__init__()
            self.p = nn.Linear(C, C, bias=False)

        def forward(self, x):
            return self.p(x)

    return MORPHBlock(RMSNorm(C), _Attn(), RMSNorm(C), mlp, d_model=C,
                      hc_kwargs=dict(n_streams=4, use_kernel=True, fused_norm=fused_norm,
                                     fused_grad=fused_grad))


@pytest.mark.parametrize("fused_norm,fused_grad", [(True, False), (False, True), (True, True)])
def test_cpu_block_unchanged_by_keys(fused_norm, fused_grad):
    ref, blk = _block(False, False), _block(fused_norm, fused_grad)
    blk.load_state_dict(ref.state_dict())
    with torch.no_grad():                         # non-trivial norm weights
        for m in (ref, blk):
            m.norm_attn.weight.copy_(torch.linspace(0.5, 1.5, 64))
            m.norm_mlp.weight.copy_(torch.linspace(1.5, 0.5, 64))
    h0 = torch.randn(2, 9, 4, 64)
    outs = []
    for m in (ref, blk):
        h = h0.clone().requires_grad_(True)
        out = m(h)
        out.square().sum().backward()
        outs.append([out.detach(), h.grad] + [p.grad for p in m.parameters()])
    for x, y in zip(*outs):
        assert torch.equal(x, y)


def test_fold_keys_refuse_eager_and_foreign_norms():
    with pytest.raises(ValueError, match="hc_use_kernel"):
        HyperConnectionResidual(64, use_kernel=False, fused_norm=True)
    with pytest.raises(ValueError, match="hc_use_kernel"):
        HyperConnectionResidual(64, use_kernel=False, fused_grad=True)
    with pytest.raises(TypeError, match="attention.RMSNorm"):
        MORPHBlock(nn.LayerNorm(64), nn.Identity(), RMSNorm(64), nn.Identity(), d_model=64,
                   hc_kwargs=dict(use_kernel=True, fused_norm=True))
