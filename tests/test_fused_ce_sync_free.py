"""The chunked CE's normaliser stays on the device (2026-10-04 speed pass).

`_FusedLinearCE.forward` used to call `int(valid.sum().item())`, a host sync at the head
of every CE call. On CUDA it now divides by a 0-dim device tensor through `_scale_inv`,
which must reproduce ATen's division of a CUDA tensor by a Python number BIT FOR BIT
(ATen multiplies by the fp32 reciprocal there). On CPU the old host division is kept.

The reference below is the pre-change forward, copied verbatim (loss, grad_x, grad_w),
so the pins compare the new kernel against the old one, not against a re-derivation.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.fused_ce import _scale_inv, fused_linear_cross_entropy


def _old_forward(x, w, labels, ignore_index, chunk_size, mask_token_id=-1, weights=None):
    """The pre-2026-10-04 `_FusedLinearCE.forward` body, verbatim (returns the three
    tensors the backward saves, plus the loss)."""
    import torch.nn.functional as F

    N, d = x.shape
    compute_dtype = x.dtype
    valid = labels != ignore_index
    if weights is None:
        n_valid = int(valid.sum().item())
        n_valid_f = float(max(n_valid, 1))
        row_w = None
    else:
        row_w = weights.to(torch.float32) * valid.to(torch.float32)
        n_valid_f = float(max(float(row_w.sum().item()), 1e-6))
    w_cast = w.to(compute_dtype)
    V = w_cast.shape[0]
    if V % 64 == 0:
        V_pad = V
    else:
        V_pad = ((V + 127) // 128) * 128
        w_cast = F.pad(w_cast, (0, 0, 0, V_pad - V))
    neg_inf = torch.finfo(torch.float32).min
    grad_x = torch.empty_like(x)
    grad_w = torch.zeros((V_pad, d), device=w.device, dtype=torch.float32)
    loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_c = x[start:end]
        lab_c = labels[start:end]
        valid_c = valid[start:end].float().unsqueeze(-1)
        logits_c = (x_c @ w_cast.t()).float()
        if V_pad != V:
            logits_c[:, V:] = neg_inf
        if mask_token_id >= 0:
            logits_c[:, mask_token_id] = neg_inf
        lse = torch.logsumexp(logits_c, dim=-1)
        lab_safe = lab_c.clamp(min=0)
        tgt = logits_c.gather(-1, lab_safe.unsqueeze(-1)).squeeze(-1)
        w_c = valid_c if row_w is None else row_w[start:end].unsqueeze(-1)
        loss_c = (lse - tgt) * w_c.squeeze(-1)
        loss_sum = loss_sum + loss_c.sum()
        probs = torch.softmax(logits_c, dim=-1)
        probs.scatter_add_(-1, lab_safe.unsqueeze(-1),
                           -torch.ones_like(lab_safe, dtype=probs.dtype).unsqueeze(-1))
        probs = probs * w_c
        probs_c = probs.to(compute_dtype)
        grad_x[start:end] = probs_c @ w_cast
        grad_w += (probs_c.t() @ x_c).float()
    loss = loss_sum / n_valid_f
    grad_x.div_(n_valid_f)
    grad_w = grad_w[:V] if V_pad != V else grad_w
    grad_w.div_(n_valid_f)
    return loss, grad_x, grad_w


def _case(device, dtype, n=300, d=64, V=203, frac_valid=0.6, weights=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g).to(device=device, dtype=dtype)
    w = (torch.randn(V, d, generator=g) * 0.2).to(device=device, dtype=torch.float32)
    lab = torch.randint(0, V, (n,), generator=g)
    lab[torch.rand(n, generator=g) > frac_valid] = -100
    lab = lab.to(device)
    wt = (torch.rand(n, generator=g) + 0.25).to(device) if weights else None
    return x, w, lab, wt


def _new(x, w, lab, wt, chunk, mask):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    loss = fused_linear_cross_entropy(x, w, lab, chunk_size=chunk, mask_token_id=mask,
                                      weights=wt)
    loss.backward()
    return loss.detach(), x.grad, w.grad


def _check(device, dtype):
    for weights in (False, True):
        for chunk in (64, 128, 1000):
            for mask in (-1, 7):
                x, w, lab, wt = _case(device, dtype, weights=weights, seed=chunk + mask)
                lo, gx, gw = _old_forward(x, w, lab, -100, chunk, mask, wt)
                ln, gxn, gwn = _new(x, w, lab, wt, chunk, mask)
                # backward multiplies the saved grads by grad_output = 1 and casts to the
                # input dtypes, exactly as the old backward did.
                assert torch.equal(lo, ln), (weights, chunk, mask)
                assert torch.equal(gx.to(dtype), gxn), (weights, chunk, mask)
                assert torch.equal(gw.to(torch.float32), gwn), (weights, chunk, mask)


def test_cpu_path_is_the_old_kernel():
    _check("cpu", torch.float32)
    _check("cpu", torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA division semantics")
def test_cuda_path_is_the_old_kernel_bit_for_bit():
    _check("cuda", torch.bfloat16)
    _check("cuda", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast")
def test_cuda_autocast_path_is_the_old_kernel_bit_for_bit():
    """The model's real call: fp32 activations under bf16 autocast. The GEMM operands are
    now cast once per call instead of once per chunk GEMM (2026-10-04); same casts, same
    bits."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _check("cuda", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA division semantics")
def test_scale_inv_matches_cuda_scalar_division():
    g = torch.Generator(device="cpu").manual_seed(1)
    for dtype in (torch.float32, torch.bfloat16):
        for n in list(range(1, 400)) + [1021, 4093, 12288, 7681]:
            t = torch.randn(2048, generator=g).to("cuda", dtype)
            a = t.clone().div_(float(n))
            b = t.clone()
            _scale_inv(b, torch.reciprocal(torch.tensor(float(n), device="cuda")))
            assert torch.equal(a, b), (dtype, n)


def test_cpu_scale_inv_is_division():
    t = torch.randn(512, dtype=torch.float32)
    a = t.clone().div_(7.0)
    b = t.clone()
    _scale_inv(b, 7.0)
    assert torch.equal(a, b)
