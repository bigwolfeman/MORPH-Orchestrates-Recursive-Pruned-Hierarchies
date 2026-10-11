"""The one-pass int-N fake quantisation (morph/kernels/triton/int_row_quant.py, 2026-10-10).

What each test pins (CUDA):
  1. Bit for bit (int32 view) equal to ``IntNRowSTE.eager`` at both production table shapes
     (49280 x 768 euclidean, 49152 x 1024 bigram) and both widths (int6, int8), on tables that
     hold an all-zero row (the 1e-8 scale floor), rows of exact half-steps (round half to
     even), a row with a NaN (the row reads NaN), a row with an inf, tiny and huge rows.
  2. The backward is the identity: the gradient reaching the shadow is the incoming one, the
     same as through the eager ``w + (q - w).detach()``.
  3. ``IntNRowSTE.forward`` takes the kernel on a CUDA fp32 table and the eager body on CPU.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.embed_quant import IntNRowSTE

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel, CUDA")


def _table(V, D, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(V, D, generator=g) * 0.02
    w[3] = 0.0                                                    # scale floor
    w[7] = torch.arange(D).float() * 1e-30                        # tiny
    w[8] = torch.randn(D, generator=g) * 1e30                     # huge
    w[11, 4] = float("nan")
    w[13, 2] = float("inf")
    return w.cuda()


def _ties(w, hi):
    """Row 5 = half steps of its own scale: w / s lands exactly on k + 0.5 (k = 0, +-1, ..)."""
    A = torch.tensor(0.75, dtype=torch.float32)
    # ATen's tensor / CPU-scalar is a multiply by the fp32 reciprocal
    s = A * (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(hi)))
    D = w.shape[1]
    k = (torch.arange(D - 1) % 9 - 4).float() + 0.5
    w[5, 0] = A
    w[5, 1:] = (k * s).to(w.device)
    return w


def _bits(t):
    return t.contiguous().view(torch.int32)


@cuda
@pytest.mark.parametrize("bits", [6, 8])
@pytest.mark.parametrize("V,D", [(49280, 768), (49152, 1024), (37, 50)])
def test_bit_identical_to_the_eager_quantiser(bits, V, D):
    q = IntNRowSTE(bits)
    w = _ties(_table(V, D, seed=V + D + bits), q.hi)
    ref = q.eager(w)
    got = q(w)
    assert torch.equal(_bits(got), _bits(ref))
    assert torch.isnan(got[11]).all() and torch.isnan(ref[11]).all()
    # the half-step row really exercises ties: w / s is exactly k + 0.5 for some k, by the
    # eager arithmetic itself
    s5 = w[5:6].abs().amax(dim=-1, keepdim=True).div(q.hi).clamp(min=1e-8)
    x = w[5:6] / s5
    assert int(((x - x.floor()) == 0.5).sum()) >= 2


@cuda
def test_backward_is_the_identity():
    q = IntNRowSTE(6)
    w = _table(300, 768).requires_grad_(True)
    up = torch.randn(300, 768, device="cuda")
    q(w).backward(up)
    g_kernel = w.grad.clone()
    w.grad = None
    q.eager(w).backward(up)
    assert torch.equal(_bits(g_kernel), _bits(w.grad)) and torch.equal(g_kernel, up)


def test_cpu_takes_the_eager_body():
    q = IntNRowSTE(6)
    w = torch.randn(20, 16)
    assert torch.equal(q(w), q.eager(w))
