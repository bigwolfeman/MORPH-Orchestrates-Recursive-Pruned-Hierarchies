"""model.kv_flat_kernels: the re-tiled K/V-side backward (morph/kernels/triton/kv_flat.py).

What each test pins (2026-10-13):
  1. FP64 PARITY. Through the real entry points (``fused_cca_conv`` and ``fused_cca_prologue``
     as torch.library ops), every gradient of the flat path has an error against an fp64
     reference no worse than the in-tree path's (same rule as test_cca_prologue_tiled.py:
     REL_SLACK plus an fp32 noise floor). Production shapes (core pass 6 x 256, prelude/coda
     6 x 1280), odd shapes (tile and row-block remainders) and both prologue launch forms
     (kv groups split across programs below 192 row blocks, not split above).
  2. DETERMINISM. Two calls on the same inputs return the same bytes (no atomics).
  3. NO STALE WEIGHT GRAD. A second call with other grads must return the grads of the second
     inputs, not the first call's partials. The check is shown to catch a sabotaged sum.
  4. ROUTING. flat=True reaches the flat kernels; flat=False never does.
  5. REFUSALS, DEFAULT. The key needs block_custom_ops; the default is off.

Sabotage record (2026-10-13), each edit made on purpose, the named test failed, edit restored:
  (a) ``cca_conv_bwd_flat`` returned the FIRST call's ``dw_gp`` on every later call
      -> test_flat_conv_does_not_serve_a_stale_weight_grad.
  (b) ``_kvf_dw_bwd_kernel`` mapped ``part_ptr`` slabs with the tap and channel axes swapped
      -> test_conv_backward_matches_fp64.

CUDA only (Triton kernels).
"""

import pytest
import torch

from morph.kernels.triton import kv_flat
from morph.kernels.triton.fused_cca_conv import causal_conv_reference, fused_cca_conv
from morph.kernels.triton.fused_cca_prologue import cca_prologue_reference, fused_cca_prologue
from morph.model.transformer import MORPHConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

REL_SLACK, ABS_FLOOR = 0.02, 5e-7     # fp32 vector grads sum in another order: noise near 1e-7
H, HKV, D = 8, 4, 64
BF = torch.bfloat16


def _err(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def _conv_inputs(B, C, S, CG, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, S, C, device="cuda", generator=g).to(BF)          # [B, S, C], as the model
    w_dw = (torch.randn(C, 1, K, device="cuda", generator=g) * 0.3).to(BF)
    w_gp = (torch.randn(C, CG, K, device="cuda", generator=g) * 0.1).to(BF)
    r = torch.randn(B, C, S, device="cuda", generator=g)
    return x, w_dw, w_gp, r


def _conv_grads(x, w_dw, w_gp, r, CG, K, flat):
    xs = x.clone().requires_grad_(True)
    wd, wg = w_dw.clone().requires_grad_(True), w_gp.clone().requires_grad_(True)
    y = fused_cca_conv(xs.transpose(1, 2), wd, wg, w_gp.shape[0] // CG, K, custom_op=True,
                       flat=flat)
    (y.float() * r).sum().backward()
    return [xs.grad, wd.grad, wg.grad]


def _conv_ref(x, w_dw, w_gp, r, K):
    xs = x.double().requires_grad_(True)
    wd, wg = w_dw.double().requires_grad_(True), w_gp.double().requires_grad_(True)
    y = causal_conv_reference(xs.transpose(1, 2), wd, wg, K)
    (y * r.double()).sum().backward()
    return [xs.grad, wd.grad, wg.grad]


CONV_CASES = [
    (6, 768, 256, 64, 4),     # the core pass: 6 x 256 cells, q|k latents 512 + 256
    (3, 384, 200, 64, 4),     # S not a multiple of any tile
    (2, 128, 70, 32, 4),      # CG 32, short row
    (5, 256, 130, 64, 3),     # K = 3, one position past a 128 tile
]


@pytest.mark.parametrize("B,C,S,CG,K", CONV_CASES)
def test_conv_backward_matches_fp64(B, C, S, CG, K):
    x, w_dw, w_gp, r = _conv_inputs(B, C, S, CG, K)
    ref = _conv_ref(x, w_dw, w_gp, r, K)
    old = _conv_grads(x, w_dw, w_gp, r, CG, K, flat=False)
    new = _conv_grads(x, w_dw, w_gp, r, CG, K, flat=True)
    for name, a, b, q in zip(("dx", "dw_dw", "dw_gp"), old, new, ref):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        e_old, e_new = _err(a, q), _err(b, q)
        assert e_new <= e_old * (1 + REL_SLACK) + ABS_FLOOR, (
            f"{name}: flat error {e_new:.3e} > in-tree error {e_old:.3e}")
        assert e_new < 1e-2, (name, e_new)        # and absolutely sane: bf16 output rounding


def test_conv_backward_is_deterministic():
    x, w_dw, w_gp, r = _conv_inputs(6, 768, 256, 64, 4)
    a = _conv_grads(x, w_dw, w_gp, r, 64, 4, flat=True)
    b = _conv_grads(x, w_dw, w_gp, r, 64, 4, flat=True)
    for n, u, v in zip(("dx", "dw_dw", "dw_gp"), a, b):
        assert torch.equal(u, v), n


def _stale_check(B=6, C=768, S=256, CG=64, K=4):
    """Two calls with different grads; the second must match ITS fp64 reference. Returns the
    worst dw error of the second call."""
    x, w_dw, w_gp, r1 = _conv_inputs(B, C, S, CG, K, seed=1)
    _, _, _, r2 = _conv_inputs(B, C, S, CG, K, seed=2)
    _conv_grads(x, w_dw, w_gp, r1, CG, K, flat=True)
    got = _conv_grads(x, w_dw, w_gp, r2, CG, K, flat=True)
    ref = _conv_ref(x, w_dw, w_gp, r2, K)
    return max(_err(got[1], ref[1]), _err(got[2], ref[2]))


def test_flat_conv_does_not_serve_a_stale_weight_grad():
    assert _stale_check() < 1e-2


def test_the_stale_check_catches_a_stale_weight_grad(monkeypatch):
    """Sabotage in the test itself: a sum that returns the FIRST call's result for every later
    call. The check above must fail on it, or it proves nothing."""
    real, cache = kv_flat._sum_partials, {}

    def stale(bufs):
        key = tuple(tuple(b[0].shape) for b in bufs)
        if key in cache:
            for (_, out, _), old in zip(bufs, cache[key]):
                out.copy_(old)                   # serve the previous call's grads
            return
        real(bufs)
        cache[key] = [out.clone() for _, out, _ in bufs]

    monkeypatch.setattr(kv_flat, "_sum_partials", stale)
    assert _stale_check() > 1e-2


def _rows_inputs(B, S, seed=0):
    from test_cca_prologue_tiled import _inputs
    return _inputs(B, S, True, seed)


def _rows_run(ins, cos, sin, gos, n_skip, flat):
    xs = [t.detach().requires_grad_(True) for t in ins]
    outs = fused_cca_prologue(*xs[:6], *xs[6:9], cos, sin, H, HKV, D, n_skip, 1e-6, tiled=True,
                              custom_op=True, flat=flat)
    torch.autograd.backward(list(outs), [g.to(o.dtype) for g, o in zip(gos, outs)])
    return [x.grad for x in xs]


ROWS_CASES = [
    (6, 256, 0),       # core pass: 96 row blocks, kv groups split across programs
    (6, 1280, 0),      # prelude / coda: 480 row blocks, not split
    (3, 37, 5),        # rows not a multiple of the block, trailing positions skip RoPE
    (7, 100, 0),       # 700 rows, 44 blocks, split, ragged last block
]
NAMES = ["q_lat", "k_lat", "q_conv", "k_conv", "v_curr", "v_prev", "wq", "wk", "temp"]


@pytest.mark.parametrize("B,S,n_skip", ROWS_CASES)
def test_prologue_backward_matches_fp64(B, S, n_skip):
    from test_cca_prologue_tiled import _run64
    ins, cos, sin, gos = _rows_inputs(B, S)
    _, ref = _run64(ins, cos, sin, gos, n_skip)
    old = _rows_run(ins, cos, sin, gos, n_skip, flat=False)
    new = _rows_run(ins, cos, sin, gos, n_skip, flat=True)
    for name, a, b, q in zip(NAMES, old, new, ref):
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride(), name
        e_old, e_new = _err(a, q), _err(b, q)
        assert e_new <= e_old * (1 + REL_SLACK) + ABS_FLOOR, (
            f"{name}: flat error {e_new:.3e} > in-tree error {e_old:.3e}")


def test_prologue_backward_is_deterministic():
    ins, cos, sin, gos = _rows_inputs(6, 256)
    a = _rows_run(ins, cos, sin, gos, 0, flat=True)
    b = _rows_run(ins, cos, sin, gos, 0, flat=True)
    for n, u, v in zip(NAMES, a, b):
        assert torch.equal(u, v), n


@pytest.mark.parametrize("n,M,K", [(1, 700, 1), (4, 3072, 4), (37, 64, 1), (100, 4 * 130, 4)])
def test_sum_kernel_sums_in_the_documented_layout(n, M, K):
    g = torch.Generator(device="cuda").manual_seed(0)
    part = torch.randn(n, M, device="cuda", generator=g)
    out = torch.empty(M, device="cuda", dtype=BF)
    kv_flat._sum_partials([(part, out, K)])
    ref = part.double().sum(0).reshape(K, M // K).t().reshape(M)     # [K, M/K] -> [M/K, K]
    assert torch.allclose(out.double(), ref.to(BF).double(), atol=0, rtol=2 ** -7), (n, M, K)


def test_flat_reaches_the_flat_kernels_and_off_does_not(monkeypatch):
    calls = []
    real = kv_flat.cca_conv_bwd_flat
    monkeypatch.setattr(kv_flat, "cca_conv_bwd_flat",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    x, w_dw, w_gp, r = _conv_inputs(2, 128, 64, 32, 4)
    _conv_grads(x, w_dw, w_gp, r, 32, 4, flat=False)
    assert calls == []
    _conv_grads(x, w_dw, w_gp, r, 32, 4, flat=True)
    assert calls == [1]


def test_refusals_and_default():
    assert MORPHConfig.__dataclass_fields__["kv_flat_kernels"].default is False
    x, w_dw, w_gp, _ = _conv_inputs(2, 128, 64, 32, 4)
    with pytest.raises(ValueError, match="block_custom_ops"):
        fused_cca_conv(x.transpose(1, 2), w_dw, w_gp, 4, 4, custom_op=False, flat=True)
    ins, cos, sin, _ = _rows_inputs(2, 64)
    with pytest.raises(ValueError, match="block_custom_ops"):
        fused_cca_prologue(*ins[:6], *ins[6:9], cos, sin, H, HKV, D, tiled=True,
                           custom_op=False, flat=True)
    from morph.model.transformer import MORPHTransformer
    with pytest.raises(ValueError, match="kv_flat_kernels"):
        MORPHTransformer(MORPHConfig(
            d_model=256, n_heads=2, n_kv_heads=2, d_ff=688, vocab_size=512, max_seq_len=512,
            n_prelude=1, n_core=0, n_coda=1, channel_dims=(128, 80, 48), compression=2,
            csa_compress_ratio=8, hca_compress_ratio=256, top_k=256, window_size=256,
            retention=False, use_kernels=True, kv_flat_kernels=True))
