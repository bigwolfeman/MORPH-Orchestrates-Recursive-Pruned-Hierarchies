"""The HC residual's two paths compute ONE function: model.hc_use_kernel=true (the fused
Triton pre-map / post kernels, what every production run trains) and
model.hc_use_kernel=false (the kernels' pure-PyTorch references: a debug opt-in on CUDA,
the only path on CPU; until 2026-10-08 also the path every model.use_kernels=false config
ran), and the decode engine's inline HC is that function too.

Contracts (hcgap, 2026-10-08):
  * Under bf16 autocast with an fp32 carrier, the carrier ops (x_bar = Hpre_cm·h and the
    skip Hres·h) stay in the carrier dtype on BOTH paths. Before the fix the references'
    einsums ran as autocast bf16 bmm: Hres rounded to bf16 (no longer orthogonal) and the
    skip stored in bf16, 2.4e-3 rel-RMS per HC op against 4e-8 for the kernel; on the
    winner that moved the step-0 loss by +2.64 nats through the slot gain hinge. The
    Cayley closed form runs with autocast off for the same reason.
  * Kernel vs eager under autocast, forward and every gradient (carrier, mapping weight
    and bias, the sublayer weight, the folded inject term), agree to fp32 roundoff at the
    winner's real shapes and at edge shapes. Both paths share one bf16 op by design, the
    mapping GEMV's forward (bf16 inputs and result, fp32 backward). Without autocast both
    forwards match an fp64 evaluation of the definition to fp32 roundoff.
  * model.use_kernels=false (the global force_eager) does not reach HC on CUDA.
  * StaticDecodeEngine._hc (the decode engine's HC) equals the training module's forward.
"""

import pytest
import torch
import torch.nn.functional as F

from morph.kernels.triton.fused_hyper_connection import hc_post_reference, hc_pre_reference
from morph.model.hyper_connections import HyperConnectionResidual

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")

TAU, ALPHA, EPS = 1.0, 0.1, 1e-6
# (B, S, C, carrier scale, with an inject term): the winner's prelude/coda and core
# shapes, then edges: an odd token count, C below its pow2 block (masked lanes), a
# near-zero carrier, a large one.
SHAPES = [
    (6, 1280, 1024, 1.0, False),
    (6, 256, 1024, 1.0, True),
    (1, 7, 1024, 1.0, True),
    (2, 5, 768, 1.0, False),
    (2, 64, 1024, 1e-3, True),
    (2, 64, 1024, 300.0, False),
]


def rel(a: torch.Tensor, r: torch.Tensor) -> float:
    a, r = a.double(), r.double()
    return ((a - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-300)).item()


def reference64(h, pw, pb, y, term):
    """The HC definition (hyper_connections.py module docstring) in fp64."""
    B, S, n, C = h.shape
    h, pw, pb = h.double(), pw.double(), pb.double()
    xf = h.reshape(B, S, n * C)
    rms = xf.pow(2).mean(-1, keepdim=True).add(EPS).sqrt()
    raw = (xf @ pw.t() / rms + pb).reshape(B, S, 3, n, n)
    hpre = torch.softmax(raw[:, :, 0] / TAU, -1)
    hpost = torch.softmax(raw[:, :, 1] / TAU, -2)
    a = raw[:, :, 2]
    bm = 0.5 * ALPHA * (a - a.transpose(-1, -2))
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    hres = torch.linalg.solve((eye - bm).transpose(-1, -2),
                              (eye + bm).transpose(-1, -2)).transpose(-1, -2)
    x_bar = torch.einsum("bsj,bsjc->bsc", hpre.mean(-2), h)
    out = torch.einsum("bsij,bsjc->bsic", hres, h) + hpost.sum(-1)[..., None] * y.double()[:, :, None]
    if term is not None:
        out = out + term.double()[:, :, None]
    return x_bar, out


def _inputs(B, S, C, scale, with_term, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(B, S, 4, C, device="cuda", generator=g) * scale
    pw = torch.randn(48, 4 * C, device="cuda", generator=g) * 0.02
    pb = torch.randn(48, device="cuda", generator=g) * 0.05
    wsub = torch.randn(C, C, device="cuda", generator=g) / C ** 0.5
    term = torch.randn(B, S, C, device="cuda", generator=g) * scale if with_term else None
    gout = torch.randn(B, S, 4, C, device="cuda", generator=g)
    return h, pw, pb, wsub, term, gout


def _run(use_kernel, h, pw, pb, wsub, term, gout, autocast=True):
    """One HC residual around an fp32 linear sublayer (autocast off inside it, so the
    sublayer adds no bf16 rounding of its own), under bf16 autocast like training (or
    without autocast: then no op on either path is bf16)."""
    C = h.shape[-1]
    mod = HyperConnectionResidual(C, n_streams=4, tau=TAU, cayley_alpha=ALPHA,
                                  use_kernel=use_kernel).cuda()
    with torch.no_grad():
        mod.proj.weight.copy_(pw)
        mod.proj.bias.copy_(pb)
    hl = h.detach().clone().requires_grad_(True)
    wl = wsub.detach().clone().requires_grad_(True)
    tl_ = None if term is None else term.detach().clone().requires_grad_(True)
    seen = {}

    def sub(xb):
        seen["x_bar"] = xb
        with torch.autocast("cuda", enabled=False):
            y = F.linear(xb.float(), wl)
        seen["y"] = y
        return y

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = mod(hl, sub, post_inject=tl_)
    (out.float() * gout).sum().backward()
    grads = {"h": hl.grad, "proj_w": mod.proj.weight.grad, "proj_b": mod.proj.bias.grad,
             "w_sub": wl.grad}
    if tl_ is not None:
        grads["term"] = tl_.grad
    return out.detach(), seen["x_bar"].detach(), seen["y"].detach(), grads


# Measured 2026-10-08 (worst over SHAPES, RTX 5090). Kernel vs eager under autocast: out
# 1.5e-7, x_bar 9.2e-8, grads h 1.8e-7, proj_w 4.7e-7, proj_b 3.4e-7, w_sub 3.8e-7, term 0.
# Either path vs fp64 without autocast: <= 1.8e-7. The fixes this pins, each measured as
# kernel vs eager: carrier einsums in bf16 (the 2026-10-07 code) fail on dtype; with only
# those fixed, the Cayley closed form's two matmuls still ran in bf16 (out 2.0e-3 to 2.9e-3,
# grad_h 2.8e-3); with that fixed too, plain autocast ran the mapping GEMV's backward in
# bf16 (grad_h 0.8e-3 to 1.1e-3, grad_proj_w 2.9e-3).
TOL_FWD = 1e-6
TOL_GRAD = {"h": 2e-6, "w_sub": 2e-6, "term": 2e-6, "proj_w": 2e-6, "proj_b": 2e-6}
TOL_DEF = 1e-6


@cuda
@pytest.mark.parametrize("B,S,C,scale,with_term", SHAPES)
def test_kernel_and_eager_hc_are_one_function(B, S, C, scale, with_term):
    h, pw, pb, wsub, term, gout = _inputs(B, S, C, scale, with_term)
    ok, xk, yk, gk = _run(True, h, pw, pb, wsub, term, gout)
    oe, xe, ye, ge = _run(False, h, pw, pb, wsub, term, gout)
    assert ok.dtype == oe.dtype == h.dtype and xk.dtype == xe.dtype == h.dtype
    errs = {"out": rel(oe, ok), "x_bar": rel(xe, xk)}
    errs.update({f"grad_{k}": rel(ge[k], gk[k]) for k in gk})
    bad = [k for k, v in errs.items()
           if v > (TOL_GRAD[k[5:]] if k.startswith("grad_") else TOL_FWD)]
    assert not bad, f"kernel vs eager beyond tolerance: {errs}"
    print(f"[kernel-vs-eager] {B}x{S}x{C} scale {scale}: "
          + " ".join(f"{k} {v:.2e}" for k, v in errs.items()))
    # Both are the definition. Without autocast no op on either path is bf16 (the mapping
    # GEMV is the one bf16 op under autocast, and a bf16 result flips at rounding
    # boundaries an fp64 sum does not share), so both forwards meet fp64 at fp32 roundoff.
    for name, uk in (("kernel", True), ("eager", False)):
        o, x, y32, _ = _run(uk, h, pw, pb, wsub, term, gout, autocast=False)
        x64, o64 = reference64(h, pw, pb, y32, term)
        ex, eo = rel(x, x64), rel(o, o64)
        print(f"[vs-fp64] {name}: x_bar {ex:.2e} out {eo:.2e}")
        assert ex < TOL_DEF and eo < TOL_DEF, f"{name}: x_bar {ex:.2e} out {eo:.2e} vs fp64"


def test_carrier_references_ignore_autocast():
    """CPU: the references compute the carrier ops in the carrier dtype even under bf16
    autocast (CPU autocast also runs einsum's bmm in bf16, the CUDA mechanism)."""
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 9, 4, 64, generator=g)
    hres = torch.linalg.qr(torch.randn(2, 9, 4, 4, generator=g))[0]
    hpost_row = torch.rand(2, 9, 4, generator=g) * 2
    hpre_cm = torch.softmax(torch.randn(2, 9, 4, generator=g), -1)
    y = torch.randn(2, 9, 64, generator=g).bfloat16()
    term = torch.randn(2, 9, 64, generator=g)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = hc_post_reference(hres, hpost_row, h, y, term)
        x_bar = hc_pre_reference(h, hpre_cm)
    o64 = (torch.einsum("bsij,bsjc->bsic", hres.double(), h.double())
           + hpost_row.double()[..., None] * y.double()[:, :, None] + term.double()[:, :, None])
    x64 = torch.einsum("bsj,bsjc->bsc", hpre_cm.double(), h.double())
    assert out.dtype == torch.float32 and x_bar.dtype == torch.float32
    assert rel(out, o64) < 1e-6 and rel(x_bar, x64) < 1e-6, (rel(out, o64), rel(x_bar, x64))
    # Without autocast the references are the ops from before, bit for bit.
    old = torch.einsum("bsij,bsjc->bsic", hres, h) + hpost_row[..., None] * y[:, :, None] \
        + term[:, :, None]
    assert torch.equal(hc_post_reference(hres, hpost_row, h, y, term), old)
    assert torch.equal(hc_pre_reference(h, hpre_cm), torch.einsum("bsj,bsjc->bsc", hpre_cm, h))


@cuda
@pytest.mark.parametrize("B,C,with_term,with_norm", [(1, 1024, False, False),
                                                      (4, 1024, True, True),
                                                      (3, 768, True, False),
                                                      (2, 4096, True, True)])
def test_decode_engine_hc_is_the_training_hc(B, C, with_term, with_norm):
    """StaticDecodeEngine._hc launches the premap/post kernels itself (C > 2048: the wide
    multi-CTA suite in fused_decode_step). It must compute the module's function: bias
    OUTSIDE the /rms divide, every pointer in its slot, the exact closed-form Cayley. The
    engine's GEMV reads bf16 weights against the fp32 carrier, so the module gets the
    same bf16-representable weights and the two differ by fp32 sum order only."""
    from morph.inference.engine import StaticDecodeEngine
    g = torch.Generator(device="cuda").manual_seed(1)
    h = torch.randn(B, 1, 4, C, device="cuda", generator=g)
    mod = HyperConnectionResidual(C, n_streams=4, tau=TAU, cayley_alpha=ALPHA).cuda().eval()
    with torch.no_grad():
        mod.proj.weight.copy_((torch.randn(48, 4 * C, device="cuda", generator=g) * 0.02)
                              .bfloat16().float())
        mod.proj.bias.copy_(torch.randn(48, device="cuda", generator=g) * 0.5)
    wsub = torch.randn(C, C, device="cuda", generator=g) / C ** 0.5
    term = torch.randn(B, 1, C, device="cuda", generator=g) if with_term else None
    nw = 1.0 + 0.1 * torch.randn(C, device="cuda", generator=g)
    seen = {}

    def sub(xb):
        seen.setdefault("x_bar", xb.clone())
        return F.linear(xb, wsub)

    eng = StaticDecodeEngine.__new__(StaticDecodeEngine)   # only _hc's own state
    eng._hcw = {id(mod): mod.proj.weight.detach().contiguous().to(torch.bfloat16)}
    eng._hcb = {id(mod): mod.proj.bias.detach().float().contiguous()}
    nout = torch.empty(B, C, device="cuda") if with_norm else None
    with torch.no_grad():
        out_e = eng._hc(mod, h, sub, term=term, norm_w=nw if with_norm else None,
                        norm_out=nout, norm_stride=C)
        x_e = seen.pop("x_bar")
        out_m = mod(h, sub, post_inject=term)
        x_m = seen.pop("x_bar")
    assert rel(x_e, x_m) < 1e-6 and rel(out_e, out_m) < 1e-6, (rel(x_e, x_m), rel(out_e, out_m))
    if with_norm:
        xf = x_m.float()
        want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
        assert rel(nout, want.reshape(B, C)) < 1e-6


@cuda
def test_use_kernels_false_still_runs_the_hc_kernels(monkeypatch):
    """model.use_kernels=false sets the process-global force_eager at build. It must not
    reach HC: on CUDA the residual runs the fused kernels whatever use_kernels says
    (owner decision 2026-10-08; the reference it used to pick rounded the skip to bf16)."""
    from morph.kernels.triton import fused_hyper_connection as fhc
    from morph.kernels.triton._eager_flag import force_eager, set_force_eager
    from morph.model.transformer import MORPHConfig, MORPHTransformer

    calls = []
    for name in ("_hc_pre_map_dispatch", "_hc_post_dispatch"):
        orig = getattr(fhc, name)

        def spy(*a, _o=orig, _n=name, **k):
            calls.append(_n)
            return _o(*a, **k)
        monkeypatch.setattr(fhc, name, spy)
    cfg = MORPHConfig(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=128, max_seq_len=64,
                      n_prelude=1, n_core=1, n_coda=1, mean_depth=1, max_depth=1,
                      bptt_depth=1, channel_dims=(32, 20, 12), retention=False,
                      bigram_hash_vocab=0, dropout=0.0, use_kernels=False, hc_use_kernel=True)
    try:
        torch.manual_seed(0)
        model = MORPHTransformer(cfg).cuda().train()
        assert force_eager(), "use_kernels=false no longer sets force_eager: test is moot"
        x = torch.randint(0, 128, (2, 32), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=x)
        out["loss"].backward()
    finally:
        set_force_eager(False)
    n_hc = sum(1 for m in model.modules() if isinstance(m, HyperConnectionResidual))
    assert calls.count("_hc_pre_map_dispatch") >= n_hc and calls.count("_hc_post_dispatch") >= n_hc, \
        (n_hc, calls)
