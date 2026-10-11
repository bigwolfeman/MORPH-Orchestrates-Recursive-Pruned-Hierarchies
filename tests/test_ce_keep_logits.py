"""`fused_linear_label_logprob(keep_logits=True)` (2026-10-10, phase-2 tracker C3).

The token head of `_tul_group_losses` keeps the forward's logits tiles for the backward
instead of recomputing them (one vocab GEMM fewer per step). What each test pins:
  1. CPU, the eager body: out, grad_x and grad_w are bit-identical (`torch.equal`) with and
     without the kept tiles, for an upstream gradient of either sign with zero and ignored
     rows, at a chunk size that leaves a short last chunk.
  2. Nothing is kept when no backward can run (no_grad, or no input requiring grad).
  3. A second backward through a retained graph raises (the kernel path overwrites the kept
     tiles with the gradient; the eager path refuses the same way so the contract is one).
  4. CUDA, the Triton kernel path at the token head's real shape (7680 x 1024, V 49169 padded
     to 49280, masked slot id), under bf16 autocast: bit-identical out, grad_x, grad_w.
  5. The model's token head asks for it (source pin on `_tul_group_losses`).
"""
from __future__ import annotations

import inspect

import pytest
import torch

from morph.model.fused_ce import fused_linear_label_logprob

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels, CUDA")


def _case(n, d, V, mask, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g) * 0.6
    w = torch.randn(V, d, generator=g) * 0.05
    lab = torch.randint(0, V, (n,), generator=g)
    lab[lab == mask] = mask + 1
    lab[torch.rand(n, generator=g) > 0.85] = -100
    up = torch.randn(n, generator=g)
    up[::7] = 0.0
    return x.to(device), w.to(device), lab.to(device), up.to(device)


def _run(x, w, lab, up, keep, kernel=False, chunk=16, mask=3, autocast=False):
    xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = fused_linear_label_logprob(xr, wr, lab, chunk_size=chunk, mask_token_id=mask,
                                         softmax_kernel=kernel, keep_logits=keep)
    (out * up).sum().backward()
    return out.detach(), xr.grad, wr.grad


def test_cpu_kept_tiles_are_bit_identical_to_the_recompute():
    x, w, lab, up = _case(70, 24, 203, mask=3)          # 70 rows at chunk 16: last chunk 6
    a = _run(x, w, lab, up, keep=False)
    b = _run(x, w, lab, up, keep=True)
    for name, p, q in zip(("out", "grad_x", "grad_w"), a, b):
        assert torch.equal(p, q), name
    assert float(b[2][3].abs().max()) == 0.0            # the masked vocab row: no gradient
    assert float(b[1].abs().max()) > 0.0


def test_nothing_is_kept_without_a_backward():
    x, w, lab, _ = _case(40, 8, 50, mask=3)
    seen = []
    real = torch.autograd.function.FunctionCtx.save_for_backward

    def spy(ctx, *ts):
        seen.append(ts[-1])
        return real(ctx, *ts)
    torch.autograd.function.FunctionCtx.save_for_backward = spy
    try:
        with torch.no_grad():
            fused_linear_label_logprob(x.requires_grad_(True), w, lab, chunk_size=16,
                                       keep_logits=True)
        fused_linear_label_logprob(x.detach(), w, lab, chunk_size=16, keep_logits=True)
        fused_linear_label_logprob(x.detach().requires_grad_(True), w, lab, chunk_size=16,
                                   keep_logits=True)
    finally:
        torch.autograd.function.FunctionCtx.save_for_backward = real
    assert seen[0] is None and seen[1] is None          # no_grad; nothing requires grad
    assert seen[2] is not None and seen[2].shape == (40, 128)   # V 50 padded to 128


def test_second_backward_through_kept_tiles_raises():
    x, w, lab, up = _case(30, 8, 40, mask=3)
    xr = x.clone().requires_grad_(True)
    out = fused_linear_label_logprob(xr, w, lab, chunk_size=16, keep_logits=True)
    (out * up).sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="second backward"):
        (out * up).sum().backward()
    # without keep_logits a retained graph backs up twice, as before
    xr2 = x.clone().requires_grad_(True)
    out2 = fused_linear_label_logprob(xr2, w, lab, chunk_size=16)
    (out2 * up).sum().backward(retain_graph=True)
    (out2 * up).sum().backward()


@cuda
def test_cuda_kernel_path_bit_identical_at_the_token_head_shape():
    V, D, MASK = 49169, 1024, 4
    x, w, lab, up = _case(7680, D, V, MASK, device="cuda", seed=1)
    x = x.to(torch.bfloat16).float()
    up = up * 1e-3
    a = _run(x, w, lab, up, keep=False, kernel=True, chunk=1024, mask=MASK, autocast=True)
    b = _run(x, w, lab, up, keep=True, kernel=True, chunk=1024, mask=MASK, autocast=True)
    for name, p, q in zip(("out", "grad_x", "grad_w"), a, b):
        assert torch.equal(p, q), name
    # the eager body under autocast too (the kept tile is the bf16 GEMM output)
    c = _run(x[:2000], w, lab[:2000], up[:2000], keep=False, chunk=1024, mask=MASK,
             autocast=True)
    d = _run(x[:2000], w, lab[:2000], up[:2000], keep=True, chunk=1024, mask=MASK,
             autocast=True)
    for name, p, q in zip(("out", "grad_x", "grad_w"), c, d):
        assert torch.equal(p, q), name


def test_the_token_head_keeps_its_logits():
    from morph.model.transformer import MORPHTransformer
    src = inspect.getsource(MORPHTransformer._tul_group_losses)
    # one kwargs dict serves both routes (all rows, and model.lmhead_row_cap's static rows)
    i = src.index("_lp_kw = dict(")
    assert "keep_logits=True" in src[i:src.index(")", i)]
    assert src.count("**_lp_kw") == 2
