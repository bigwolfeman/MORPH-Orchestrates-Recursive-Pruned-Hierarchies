"""``model.graph_safe``'s fixed-shape readers are BIT-IDENTICAL to the gathered eager ones on
CUDA (2026-10-07, the identity decomposition of the graph-captured training step,
.agents/notes/proposed/architecture/2026-10-07-graph-captured-training-step.md).

The captured step must train bit for bit like production. Its readers cannot gather a
host-counted row set, so they keep every row (live rows first, ``host_shadow.valid_first``)
and reproduce the eager result itself: ATen's CUDA reduction order is a function of the
row count, which ``morph.model.host_shadow`` evaluates at a fixed length for a count held on
the device. Each test compares the fixed form with the eager op it replaces, value and
gradient (a non-unit upstream gradient: MeanBackward multiplies by fl(1/n), which a
division would not match), at the real shapes of the lead arm.

What each test pins:
  * ``exact_mean_1d``: forward and backward of ``v[:n].mean()`` for every count regime
    (one thread per element below 128, the vectorised widths 32..256 above).
  * ``aten_rowsum_order`` / ``aten_rowsum_width``: ``x.sum(-1)`` at every row count 1..20.
  * ``exact_col_mean`` / ``exact_col_std``: ``x[:n].mean(0)`` (fp32, bf16, fp64, both split
    regimes, with gradient) and ``x[:n].std(0, unbiased=False)`` (Welford with FMAs).
  * The fan readers (cosine, vol pass, epi term with its ridge QR, mix entropy) and the
    fixed-point term, fixed against gathered, under the trainer's bf16 autocast.

CUDA only: the emulated order is the CUDA kernels'.
"""
from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction order")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DEV = "cuda"


@pytest.fixture(autouse=True)
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def _grad(fn, xs, g):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    y = fn(xs)
    gs = torch.autograd.grad(y, xs, torch.full_like(y, g), allow_unused=True)
    return y.detach(), [torch.zeros_like(x) if a is None else a for x, a in zip(xs, gs)]


def _same(a, b):
    ya, ga = a
    yb, gb = b
    assert torch.equal(ya, yb), (ya.item() if ya.dim() == 0 else ya, yb)
    for i, (x, y) in enumerate(zip(ga, gb)):
        assert torch.equal(x, y), f"grad {i}: max diff {(x.double() - y.double()).abs().max()}"


# ── 1. the reductions ─────────────────────────────────────────────────────────────────

_COUNTS = list(range(1, 140)) + list(range(140, 1537, 13)) + [255, 256, 257, 511, 512, 1024]


@pytest.mark.parametrize("chunk", range(4))
def test_exact_mean_1d_is_the_eager_mean(chunk):
    from morph.model.host_shadow import exact_mean_1d
    g = torch.Generator(device=DEV).manual_seed(chunk)
    L = 1536
    for n in _COUNTS[chunk::4]:
        v = torch.randn(n, device=DEV, generator=g) * 3 + 0.25
        buf = torch.cat([v, torch.randn(L - n, device=DEV, generator=g)])   # junk past n
        up = float(torch.randn((), generator=torch.Generator().manual_seed(n)))
        nd = torch.tensor(n, device=DEV)
        ref = _grad(lambda xs: xs[0].mean(), [v], up)
        got = _grad(lambda xs: exact_mean_1d(xs[0], nd), [buf], up)
        assert torch.equal(ref[0], got[0]), n
        assert torch.equal(ref[1][0], got[1][0][:n]), n
        assert bool((got[1][0][n:] == 0).all()), n


@pytest.mark.parametrize("cols", [256, 4096])
def test_rowsum_order_at_every_row_count(cols):
    from morph.model.host_shadow import aten_rowsum_order, aten_rowsum_width
    x = torch.randn(20, cols, device=DEV) * 2
    for r in range(1, 21):
        xr = x[:r].contiguous()
        assert torch.equal(xr.sum(-1), aten_rowsum_order(xr, aten_rowsum_width(r, cols))), r


@pytest.mark.parametrize("dtype,cols", [(torch.float32, 64), (torch.bfloat16, 64),
                                        (torch.float64, 1024)])
def test_exact_column_mean_and_std(dtype, cols):
    from morph.model.host_shadow import exact_col_mean, exact_col_std
    g = torch.Generator(device=DEV).manual_seed(7)
    L = 384
    for n in list(range(2, 140, 9)) + list(range(200, 385, 11)):
        x = (torch.randn(n, cols, device=DEV, generator=g) * 2 + 0.5).to(dtype)
        xp = torch.cat([x, torch.randn(L - n, cols, device=DEV, generator=g).to(dtype)])
        nd = torch.tensor(n, device=DEV)
        up = 0.37
        ref = _grad(lambda xs: (xs[0].mean(0) * torch.arange(cols, device=DEV)).sum(), [x], up)
        got = _grad(lambda xs: (exact_col_mean(xs[0], nd)
                                * torch.arange(cols, device=DEV)).sum(), [xp], up)
        assert torch.equal(ref[0], got[0]), (n, dtype)
        assert torch.equal(ref[1][0], got[1][0][:n]), (n, dtype)
        if dtype != torch.float64:
            assert torch.equal(x.std(0, unbiased=False), exact_col_std(xp, nd)), (n, dtype)


# ── 2. the fan readers at the lead arm's shapes ──────────────────────────────────────


def _fan_case(seed, n_valid):
    g = torch.Generator(device=DEV).manual_seed(seed)
    B, S, m, carrier, C = 6, 64, 4, 4, 1024
    valid = torch.zeros(B * S, dtype=torch.bool, device=DEV)
    valid[torch.randperm(B * S, generator=g, device=DEV)[:n_valid]] = True
    traj = [torch.randn(B, S * m, carrier, C, device=DEV, generator=g) for _ in range(3)]
    return valid.view(B, S), m, traj


@pytest.mark.parametrize("seed,n_valid", [(0, 212), (1, 267), (2, 384), (3, 40)])
def test_fan_readers_fixed_equal_gathered(seed, n_valid):
    from morph.model.tul_fan import (FanReservoir, TULFanMix, _fan_vol_pass, fan_epi_term,
                                     fan_stream_cos)
    valid, m, traj = _fan_case(seed, n_valid)
    res = FanReservoir(traj[0].shape[-1], 64, seed=3).to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, fn in (
                ("cos", lambda tr, fx: fan_stream_cos(tr[1], valid, m, fixed=fx)),
                ("vol", lambda tr, fx: _fan_vol_pass(tr[2], valid, m, 30.0, fixed=fx)),
                ("epi", lambda tr, fx: fan_epi_term(tr, valid, m, 2, res, 3.0, 30.0,
                                                   instruments=False, fixed=fx))):
            ref = _grad(lambda xs: fn(xs, False), traj, -0.05123)
            got = _grad(lambda xs: fn(xs, True), traj, -0.05123)
            try:
                _same(ref, got)
            except AssertionError as e:
                raise AssertionError(f"{name}: {e}") from None
        w = torch.softmax(torch.randn(*valid.shape, m, device=DEV), -1)
        assert torch.equal(TULFanMix.entropy(w, valid), TULFanMix.entropy(w, valid, fixed=True))


# ── 3. the fixed-point term ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(6))
def test_fixed_point_term_fixed_equals_gathered(seed):
    from morph.model.transformer import _fp_term_exact
    g = torch.Generator(device=DEV).manual_seed(seed)
    B, SM, T = 6, 256, 8
    u0 = torch.randn(B, SM, 4, 1024, device=DEV, generator=g)
    h0 = u0 + 0.1 * torch.randn(B, SM, 4, 1024, device=DEV, generator=g)
    valid = torch.rand(B, SM, device=DEV, generator=g) < 0.8
    # Small Poisson means put 1..15 finishing cells on the late passes (their own order).
    lam = (6.0, 1.5, 3.0)[seed % 3]
    depths = torch.poisson(torch.full((B, SM), lam, device=DEV), generator=g).clamp(1, T).long()

    def eager(xs):
        u, h = xs
        terms = []
        for t in range(T):
            fin = valid & (depths == t + 1)
            if fin.any():
                fn, fo = u[fin].flatten(1).float(), h[fin].flatten(1).float()
                terms.append((fn - fo).pow(2).sum(-1) / (fn.pow(2).sum(-1) + 1e-6))
        return torch.cat(terms).mean()

    def fixed(xs):
        u, h = xs
        acc = None
        for t in range(T):
            fin = valid & (depths == t + 1)
            fv, tp = fin.view(B, SM, 1, 1), torch.full_like(depths, t)
            if acc is None:
                acc = (torch.where(fv, u, torch.zeros_like(u)),
                       torch.where(fv, h, torch.zeros_like(h)), fin,
                       torch.where(fin, tp, torch.zeros_like(tp)))
            else:
                acc = (torch.where(fv, u, acc[0]), torch.where(fv, h, acc[1]), acc[2] | fin,
                       torch.where(fin, tp, acc[3]))
        return _fp_term_exact(acc, None, T)

    _same(_grad(eager, [u0, h0], 0.1), _grad(fixed, [u0, h0], 0.1))
