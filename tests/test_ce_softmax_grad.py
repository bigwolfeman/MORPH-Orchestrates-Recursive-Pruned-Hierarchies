"""`model.ce_softmax_kernel` (2026-10-04 speed pass): the chunked CE's elementwise body as
one Triton kernel (morph/kernels/triton/ce_softmax_grad.py).

It is NOT bit-identical to the eager body, so the pins here are accuracy bounds against an
fp64 reference computed from the SAME bf16 logits: the kernel's loss, grad_x and grad_w
must be at least as close to the reference as the eager body's are (within 25 % slack plus
an absolute floor), over the cases the model uses (padded vocab, the masked slot id,
per-row weights, ignored rows). The kernel's own tile output is also checked against the
fp64 softmax gradient directly. Off (the default) is the eager body: pinned bit for bit by
tests/test_fused_ce_sync_free.py.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.fused_ce import fused_linear_cross_entropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel, CUDA")


def _case(n=2500, d=192, V=49169, frac_valid=0.7, weights=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(n, d, generator=g) * 0.5).to("cuda", torch.bfloat16)
    w = (torch.randn(V, d, generator=g) * 0.3).to("cuda", torch.float32)
    lab = torch.randint(0, V, (n,), generator=g)
    lab[lab == 7] = 8                                   # 7 is the masked id: never a label
    lab[torch.rand(n, generator=g) > frac_valid] = -100
    wt = (torch.rand(n, generator=g) + 0.25).cuda() if weights else None
    return x, w, lab.cuda(), wt


def _run(x, w, lab, wt, kernel, mask, autocast=False, compact=False):
    """``autocast``: the model's real call, fp32 activations under bf16 autocast (the
    bf16 ``x`` upcast to fp32 is exact, so the GEMMs see the same bf16 operands and the
    fp64 reference is the same)."""
    xr = (x.float() if autocast else x.clone()).requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        loss = fused_linear_cross_entropy(xr, wr, lab, chunk_size=1000, mask_token_id=mask,
                                          weights=wt, softmax_kernel=kernel,
                                          compact_rows=compact)
    loss.backward()
    return loss.detach().double(), xr.grad.double(), wr.grad.double()


def _ref(x, w, lab, wt, mask):
    """fp64 loss and grads from the bf16 logits the kernel and the eager body both see."""
    V = w.shape[0]
    wb = w.to(torch.bfloat16)
    logits = (x @ wb.t()).double()                      # the bf16 GEMM output, exactly
    if mask >= 0:
        logits[:, mask] = float("-inf")
    valid = lab != -100
    rw = valid.double() if wt is None else wt.double() * valid.double()
    n = rw.sum().clamp_min(1e-6) if wt is not None else rw.sum().clamp_min(1)
    ls = lab.clamp(min=0)
    lse = torch.logsumexp(logits, -1)
    loss = ((lse - logits.gather(-1, ls[:, None])[:, 0]) * rw).sum() / n
    p = torch.softmax(logits, -1)
    p[torch.arange(len(ls)), ls] -= 1.0
    p = p * rw[:, None] / n
    return loss, p @ wb.double(), p.t() @ x.double(), V


def _err(a, b):
    return float((a - b).abs().max())


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("mask", [-1, 7])
@pytest.mark.parametrize("V", [49169, 4096])
def test_kernel_is_as_accurate_as_the_eager_body(weights, mask, V, autocast):
    x, w, lab, wt = _case(V=V, weights=weights, seed=V + mask)
    rl, rgx, rgw, _ = _ref(x, w, lab, wt, mask)
    el, egx, egw = _run(x, w, lab, wt, False, mask, autocast)
    kl, kgx, kgw = _run(x, w, lab, wt, True, mask, autocast)
    assert not torch.equal(kgw, egw)                     # the kernel really ran
    for name, k, e, r, floor in (("loss", kl, el, rl, 1e-6), ("grad_x", kgx, egx, rgx, 1e-7),
                                 ("grad_w", kgw, egw, rgw, 1e-7)):
        ek, ee = _err(k, r), _err(e, r)
        assert ek <= 1.25 * ee + floor, (name, ek, ee)
    # and the two paths agree with each other to a bf16-tile rounding
    assert _err(kl, el) < 1e-5
    assert _err(kgx, egx) <= 4 * float(egx.abs().max()) * 2 ** -8
    if mask >= 0:
        assert float(kgw[mask].abs().max()) == 0.0      # the masked row gets no gradient


def test_tile_is_the_weighted_softmax_gradient():
    from morph.kernels.triton.ce_softmax_grad import ce_softmax_grad_
    g = torch.Generator(device="cpu").manual_seed(3)
    c, Vp, V = 37, 5000, 4990                           # pad columns 4990..4999
    logits = (torch.randn(c, Vp, generator=g) * 4).to("cuda", torch.bfloat16)
    lab = torch.randint(0, V, (c,), generator=g).cuda()
    w = torch.rand(c, generator=g).cuda()
    w[::5] = 0.0                                        # ignored rows
    x = logits.double().clone()
    x[:, V:] = float("-inf")
    x[:, 11] = float("-inf")
    lse = torch.logsumexp(x, -1)
    p = torch.softmax(x, -1)
    p[torch.arange(c), lab] -= 1.0
    p = p * w.double()[:, None]
    loss_ref = (lse - x.gather(-1, lab[:, None])[:, 0]) * w.double()
    t = logits.clone()
    loss = ce_softmax_grad_(t, lab, w, V, 11)
    assert float((loss.double() - loss_ref).abs().max()) < 2e-5
    assert torch.equal(t[:, V:], torch.zeros_like(t[:, V:]))
    assert float(t[:, 11].abs().max()) == 0.0
    # within one bf16 rounding of the exact gradient
    assert float((t.double() - p).abs().max()) <= float(p.abs().max()) * 2 ** -8
    assert torch.equal(t[::5], torch.zeros_like(t[::5]))


def test_off_is_the_eager_body_bit_for_bit():
    x, w, lab, wt = _case(n=900, V=4096, weights=True)
    a = _run(x, w, lab, wt, False, 7)
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    loss = fused_linear_cross_entropy(xr, wr, lab, chunk_size=1000, mask_token_id=7,
                                      weights=wt)                       # no keyword at all
    loss.backward()
    assert torch.equal(a[0], loss.detach().double())
    assert torch.equal(a[1], xr.grad.double()) and torch.equal(a[2], wr.grad.double())


def test_the_key_reaches_the_model():
    import pathlib
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    cdir = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        on = compose(config_name="base", overrides=["model.ce_softmax_kernel=true"])
        off = compose(config_name="base")
    assert build_morph_config(on).ce_softmax_kernel is True
    assert build_morph_config(off).ce_softmax_kernel is False


@pytest.mark.parametrize("kernel", [False, True])
@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_compact_rows_is_as_accurate_as_all_rows(kernel, weights, autocast):
    """`model.ce_compact_rows`: the rows with no loss are dropped before the GEMMs. Against
    fp64 it must be as close as the all-rows path, and the dropped rows' grad_x is 0."""
    x, w, lab, wt = _case(V=49169, weights=weights, seed=11, frac_valid=0.5)
    if wt is not None:
        wt[::7] = 0.0                                    # zero-weight rows are dropped too
    rl, rgx, rgw, _ = _ref(x, w, lab, wt, 7)
    al, agx, agw = _run(x, w, lab, wt, kernel, 7, autocast, compact=False)
    cl, cgx, cgw = _run(x, w, lab, wt, kernel, 7, autocast, compact=True)
    for name, c, a, r, floor in (("loss", cl, al, rl, 1e-6), ("grad_x", cgx, agx, rgx, 1e-7),
                                 ("grad_w", cgw, agw, rgw, 1e-7)):
        assert _err(c, r) <= 1.25 * _err(a, r) + floor, (name, _err(c, r), _err(a, r))
    dropped = lab == -100
    if wt is not None:
        dropped = dropped | (wt == 0)
    assert float(cgx[dropped].abs().max()) == 0.0
    assert float(cgx[~dropped].abs().min(dim=-1).values.max()) > 0.0   # kept rows train


def test_compact_key_reaches_the_model():
    import pathlib
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    cdir = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        on = compose(config_name="base", overrides=["model.ce_compact_rows=true"])
        off = compose(config_name="base")
    assert build_morph_config(on).ce_compact_rows is True
    assert build_morph_config(off).ce_compact_rows is False
