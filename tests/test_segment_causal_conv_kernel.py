"""Contracts for the fused segment-reset causal conv (morph/kernels/triton/
fused_seg_causal_conv.py) and for the eager reference it replaces.

Three tiers, by what they need:

  * CPU, always: the reference is byte-for-byte the function it was before the
    kernel landed (a pin computed from the pre-refactor body), the CPU dispatch
    still lands on it, and unsupported shapes are refused by the predicate the
    dispatcher asks.
  * CPU + TRITON_INTERPRET=1: the actual Triton kernels, forward and backward,
    run through Triton's interpreter on small shapes and are scored against the
    reference. This is the only numerical check of the kernel bodies that runs
    on a host with no GPU.
        TRITON_INTERPRET=1 PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES="" \
          python -m pytest -q tests/test_segment_causal_conv_kernel.py -k interpreter
  * CUDA: forward, gradients, the seg-free identity, the cross-segment zero and a
    micro-benchmark, at the shapes the model actually runs
    (tul_slot_spandec_strict_fan4_all_reach1: B=6, 64 slots x 4 cells = 256
    positions, latent_q 512 / latent_k 256, conv kernel 4).
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.:tests \
          python -m pytest -q -s tests/test_segment_causal_conv_kernel.py

The CUDA numeric tests score BOTH implementations against an fp64 oracle instead of
scoring the kernel against the reference — read ``_assert_no_worse_than_the_reference``
for the measurement that forced that, in both dtypes.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from morph.kernels.triton.fused_cca_conv import causal_conv_reference
from morph.kernels.triton.fused_seg_causal_conv import (
    fused_segment_causal_conv, seg_conv_available, seg_conv_shapes_supported)
from morph.model.attention import segment_causal_conv, segment_causal_conv_reference

_INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"
# Two invocations, never one: TRITON_INTERPRET is process-global, so under it the CUDA
# tests would score the INTERPRETER rather than the compiled kernel — which is not the
# thing under test, and is also where Triton 3.6 gets bf16 tl.dot wrong.
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or _INTERPRETING,
    reason="needs a CUDA device and the compiled (not interpreted) kernel")
INTERP = pytest.mark.skipif(
    not _INTERPRETING,
    reason="needs TRITON_INTERPRET=1 in the environment at process start")

# The arm's shapes: conv kernel 4, d_head 64, n_heads 8, n_kv_heads 4 →
# latent_q_dim 512 (G=8, Cg=64), latent_k_dim 256 (G=4, Cg=64); the core's compact
# axis is max_slots 64 x prefix_k 4 = 256 cells at batch 6.
MODEL_K = 4
MODEL_CASES = [
    # (B, C, S, Cg) — q stream / k stream on the core axis, then the coda axis.
    (6, 512, 256, 64),
    (6, 256, 256, 64),
    (6, 512, 1280, 64),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _weights(C, Cg, K, device, dtype, gen):
    w_dw = torch.randn(C, 1, K, device=device, dtype=dtype, generator=gen) * 0.5
    w_gp = torch.randn(C, Cg, K, device=device, dtype=dtype, generator=gen) * (Cg ** -0.5)
    return w_dw, w_gp


def _slot_major_seg(B, S, cells, device):
    """The id the register-reach path builds: `cells` consecutive positions per slot."""
    n = (S + cells - 1) // cells
    return torch.arange(n, device=device).repeat_interleave(cells)[:S].unsqueeze(0).expand(B, S)


def _random_seg(B, S, device, gen, n_ids=7):
    """Ragged segments: a random run-length partition, ids not monotone in position."""
    out = torch.empty(B, S, dtype=torch.long, device=device)
    for b in range(B):
        lens = torch.randint(1, 9, (S,), generator=gen, device=device)
        ids = torch.randint(0, n_ids, (S,), generator=gen, device=device)
        # S runs of length >= 1 always cover S positions.
        out[b] = torch.repeat_interleave(ids, lens)[:S]
    return out


def _case(B, C, S, Cg, K, device, dtype, seg_kind="slot", seed=0, requires_grad=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, C, S, device=device, dtype=dtype, generator=gen)
    w_dw, w_gp = _weights(C, Cg, K, device, dtype, gen)
    if seg_kind == "slot":
        seg = _slot_major_seg(B, S, 4, device)
    elif seg_kind == "random":
        seg = _random_seg(B, S, device, gen)
    elif seg_kind == "one":
        seg = torch.zeros(B, S, dtype=torch.long, device=device)
    else:  # pragma: no cover - typo guard
        raise ValueError(seg_kind)
    if requires_grad:
        x.requires_grad_(True)
        w_dw.requires_grad_(True)
        w_gp.requires_grad_(True)
    return x, w_dw, w_gp, seg




# ---------------------------------------------------------------------------
# The numeric contract on CUDA: an fp64 oracle, not the reference
# ---------------------------------------------------------------------------
# A direct `allclose(fused, reference)` compares TWO approximations of the same exact
# function and calls the difference an error of the newer one. Measured on the 5090
# (scratchpad/segconv/gpu_probe1.py, gpu_probe2.py) that is wrong in both dtypes:
#
#   * fp32. `F.conv1d` runs on cuDNN with `torch.backends.cudnn.allow_tf32` TRUE by
#     default, so the REFERENCE is the TF32 one at the shapes where cuDNN picks an
#     implicit-GEMM algorithm: its error against an fp64 oracle is 1.484e-03 at
#     (1,64,5,16) and 2.501e-03 at (2,128,37,32), dropping to 3.3e-07 / 1.2e-06 when
#     allow_tf32 is turned off. The kernel is at 4.3e-07 / 1.9e-06 either way — it
#     asks tl.dot for `input_precision="ieee"` (a raw [16,16]@[16,16] fp32 dot
#     measures 1.4e-06 with ieee and 9.7e-03 without).
#   * bf16. The reference rounds to bf16 after every tap and every stage; the kernel
#     accumulates the whole thing in fp32 and rounds twice. Against the fp64 oracle
#     the kernel is closer at EVERY shape measured: 2.2e-02 vs 5.1e-02 at
#     (6,512,256,64), 2.3e-02 vs 3.6e-02 at (2,128,37,32). The reference's error
#     grows with C; the kernel's does not.
#
# So the contract is: the kernel must be at least as close to the exact math as the
# reference is, OR inside the dtype's own round-off for this reduction. A kernel with
# a wrong mask or a wrong tap is O(scale) off and satisfies neither.
_ULP_BUDGET = {
    # One bf16 ulp (2^-8) x2: the measured worst relative error is 3.2e-03, one ulp
    # is 3.9e-03 — the kernel is inside a single rounding of the exact answer.
    torch.bfloat16: 2 ** -7,
    torch.float16: 2 ** -10,
    # fp32: measured worst relative error 6.2e-07 over a 256-term reduction; 1e-05
    # leaves 16x headroom and still fails a TF32 regression (1e-03) by 100x.
    torch.float32: 1e-5,
}


def _assert_no_worse_than_the_reference(got, ref, oracle, dtype, what):
    """`got` must beat `ref` against the fp64 `oracle`, or be inside the dtype's ulp."""
    scale = float(oracle.abs().max())
    assert scale > 0, f"{what}: the oracle is all zeros, the test would pass on anything"
    err_f = float((got.double() - oracle).abs().max())
    err_r = float((ref.double() - oracle).abs().max())
    budget = _ULP_BUDGET[dtype] * scale
    assert err_f <= max(err_r, budget), (
        f"{what}: fused is {err_f:.3e} from the exact answer, reference is "
        f"{err_r:.3e}, budget is {budget:.3e} (scale {scale:.3e})")
    return err_f, err_r


# ---------------------------------------------------------------------------
# (a) CPU: the reference is what it was
# ---------------------------------------------------------------------------

# Computed from the PRE-REFACTOR morph.model.attention.segment_causal_conv at commit
# c0f6bc4, before the rename, by scratchpad/segconv/pin_reference.py. fp32 CPU, and
# stable at 1 and 2 torch threads.
PIN_SUM = -57.39854145422578
PIN_WEIGHTED = -6747.147111669183
PIN_ABS = 1520.9606739841402
PIN_ROW0_C0 = [-2.9171905517578125, -4.924220085144043, -7.040716171264648,
               -11.904205322265625, -4.683709144592285, 1.8287930488586426]


def _pin_inputs():
    torch.manual_seed(1234)
    B, C, S, K, Cg = 2, 8, 17, 4, 4
    x = torch.randn(B, C, S, dtype=torch.float32)
    w_dw = torch.randn(C, 1, K, dtype=torch.float32)
    w_gp = torch.randn(C, Cg, K, dtype=torch.float32)
    seg = torch.tensor([[0] * 5 + [1] * 1 + [2] * 11,
                        [3] * 7 + [4] * 4 + [5] * 6], dtype=torch.long)
    return x, w_dw, w_gp, seg


def test_reference_output_matches_the_pre_refactor_pin():
    x, w_dw, w_gp, seg = _pin_inputs()
    out = segment_causal_conv_reference(x, w_dw, w_gp, seg)
    assert tuple(out.shape) == (2, 8, 17)
    assert float(out.double().sum()) == pytest.approx(PIN_SUM, rel=1e-7)
    assert float((out.double() * torch.arange(out.numel(), dtype=torch.float64)
                  .reshape(out.shape)).sum()) == pytest.approx(PIN_WEIGHTED, rel=1e-7)
    assert float(out.double().abs().sum()) == pytest.approx(PIN_ABS, rel=1e-7)
    for got, want in zip(out[0, 0, :6].tolist(), PIN_ROW0_C0):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6)


def test_cpu_dispatch_is_the_reference_bit_for_bit():
    x, w_dw, w_gp, seg = _pin_inputs()
    assert not seg_conv_available(x, w_dw, w_gp)          # no CUDA tensor -> eager
    assert torch.equal(segment_causal_conv(x, w_dw, w_gp, seg),
                       segment_causal_conv_reference(x, w_dw, w_gp, seg))


def test_unsupported_shapes_are_refused_by_the_predicate():
    C, K = 32, 4
    x = torch.randn(2, C, 12)
    ok_dw = torch.randn(C, 1, K)
    # Cg = 16 is the smallest tl.dot tile, so it is supported; 8 is not.
    assert seg_conv_shapes_supported(x, ok_dw, torch.randn(C, 16, K))
    assert not seg_conv_shapes_supported(x, ok_dw, torch.randn(C, 8, K))
    # Cg must be a power of two (the tl.dot tile shape is a constexpr).
    assert not seg_conv_shapes_supported(torch.randn(2, 48, 12), torch.randn(48, 1, K),
                                         torch.randn(48, 24, K))
    # one dtype for both tl.dot operands
    assert not seg_conv_shapes_supported(x.bfloat16(), ok_dw, torch.randn(C, 16, K))
    # kernel widths must agree between the stages
    assert not seg_conv_shapes_supported(x, ok_dw, torch.randn(C, 16, K + 1))


def test_shared_memory_budget_refuses_a_head_it_cannot_tile():
    """d_head 128 at fp32: the [Cg, Cg] weight tile alone is 64 KiB, so no legal time
    tile exists and the shapes must be refused rather than launched."""
    from morph.kernels.triton.fused_seg_causal_conv import _block_t_gp, _gp_tile_cap

    C, K = 512, 4
    x = torch.randn(2, C, 64)
    w_dw = torch.randn(C, 1, K)
    assert seg_conv_shapes_supported(x, w_dw, torch.randn(C, 64, K))          # fp32 Cg 64
    assert not seg_conv_shapes_supported(x, w_dw, torch.randn(C, 128, K))     # fp32 Cg 128
    assert seg_conv_shapes_supported(x.bfloat16(), w_dw.bfloat16(),
                                     torch.randn(C, 128, K).bfloat16())       # bf16 Cg 128
    # The production tile stays 128 at both the core axis and the coda axis.
    assert _gp_tile_cap(64, 2) == 256 and _gp_tile_cap(64, 4) == 128
    assert _block_t_gp(256, 64, 2) == 128 and _block_t_gp(1280, 64, 2) == 128


def test_unsupported_shapes_still_compute_through_the_reference():
    C, K, Cg = 32, 4, 8                                   # Cg 8 < the tl.dot minimum
    x = torch.randn(2, C, 21)
    w_dw = torch.randn(C, 1, K)
    w_gp = torch.randn(C, Cg, K)
    seg = _slot_major_seg(2, 21, 4, x.device)
    assert not seg_conv_shapes_supported(x, w_dw, w_gp)
    assert torch.equal(segment_causal_conv(x, w_dw, w_gp, seg),
                       segment_causal_conv_reference(x, w_dw, w_gp, seg))


def test_reference_cuts_every_cross_segment_tap():
    """The property the kernel must reproduce, stated on the reference itself."""
    B, C, S, K, Cg = 1, 16, 24, 4, 4
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(B, C, S, generator=gen)
    w_dw, w_gp = _weights(C, Cg, K, torch.device("cpu"), torch.float32, gen)
    seg = torch.zeros(B, S, dtype=torch.long)
    seg[:, 12:] = 1
    base = segment_causal_conv_reference(x, w_dw, w_gp, seg)
    x2 = x.clone()
    x2[:, :, 11] += 3.0
    pert = segment_causal_conv_reference(x2, w_dw, w_gp, seg)
    assert torch.equal(base[:, :, 12:], pert[:, :, 12:])
    assert not torch.equal(base[:, :, 11:12], pert[:, :, 11:12])


# ---------------------------------------------------------------------------
# (b')(c') CPU + Triton interpreter: the kernel bodies, small shapes
# ---------------------------------------------------------------------------

@INTERP
@pytest.mark.parametrize("seg_kind", ["slot", "random", "one"])
@pytest.mark.parametrize("B,C,S,Cg", [(2, 32, 21, 16), (1, 64, 5, 16), (3, 32, 16, 32)])
def test_interpreter_forward_matches_the_reference(seg_kind, B, C, S, Cg):
    K = 4
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, K, torch.device("cpu"), torch.float32,
                               seg_kind=seg_kind, seed=11)
    got = fused_segment_causal_conv(x, w_dw, w_gp, seg)
    want = segment_causal_conv_reference(x, w_dw, w_gp, seg)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), \
        f"max |Δ| = {float((got - want).abs().max())}"


@INTERP
def test_interpreter_gradients_match_the_reference():
    B, C, S, Cg, K = 2, 32, 21, 16, 4
    dev = torch.device("cpu")
    args_f = _case(B, C, S, Cg, K, dev, torch.float32, seg_kind="random", seed=5,
                   requires_grad=True)
    args_r = _case(B, C, S, Cg, K, dev, torch.float32, seg_kind="random", seed=5,
                   requires_grad=True)
    gen = torch.Generator().manual_seed(77)
    go = torch.randn(B, C, S, generator=gen)

    fused_segment_causal_conv(*args_f[:3], args_f[3]).backward(go)
    segment_causal_conv_reference(*args_r[:3], args_r[3]).backward(go)
    for name, a, b in zip(("dx", "dw_dw", "dw_gp"), args_f[:3], args_r[:3]):
        assert a.grad is not None and b.grad is not None, name
        assert a.grad.shape == b.grad.shape, name
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-4), \
            f"{name}: max |Δ| = {float((a.grad - b.grad).abs().max())}"


@INTERP
def test_interpreter_skips_the_kernels_for_inputs_that_want_no_grad():
    """Only w_gp needs a gradient: dx and dw_dw come back None, w_gp's is still right."""
    B, C, S, Cg, K = 2, 32, 17, 16, 4
    dev = torch.device("cpu")
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, K, dev, torch.float32, seed=9)
    w_gp.requires_grad_(True)
    xr, wr_dw, wr_gp, segr = _case(B, C, S, Cg, K, dev, torch.float32, seed=9)
    wr_gp.requires_grad_(True)
    gen = torch.Generator().manual_seed(4)
    go = torch.randn(B, C, S, generator=gen)

    fused_segment_causal_conv(x, w_dw, w_gp, seg).backward(go)
    segment_causal_conv_reference(xr, wr_dw, wr_gp, segr).backward(go)
    assert x.grad is None and w_dw.grad is None
    assert torch.allclose(w_gp.grad, wr_gp.grad, atol=1e-4, rtol=1e-4)


@INTERP
def test_interpreter_half_precision_plumbing():
    """A 16-bit dtype end to end: loads, the tl.dot operands and the cast-on-store.

    fp16, not bf16: Triton 3.6's INTERPRETER computes ``tl.dot`` wrong on bf16
    operands (measured here — a [16,16]x[16,16] bf16 dot comes back off by 2.3e10
    while fp16 and fp32 are exact), so bf16 is a CUDA-only test below. The kernel's
    bf16 path is the same code the shipped ``fused_cca_conv`` runs.
    """
    B, C, S, Cg, K = 2, 32, 21, 16, 4
    dev = torch.device("cpu")
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, K, dev, torch.float32, seg_kind="slot", seed=17)
    got = fused_segment_causal_conv(x.half(), w_dw.half(), w_gp.half(), seg)
    want = segment_causal_conv_reference(x, w_dw, w_gp, seg)
    assert got.dtype == torch.float16
    assert torch.allclose(got.float(), want, atol=2e-2, rtol=2e-2), \
        f"max |Δ| = {float((got.float() - want).abs().max())}"


@INTERP
def test_interpreter_multiple_time_tiles():
    """S > BLOCK_T: the per-(b, tile) weight-gradient slabs must reduce correctly and
    every tile must read its own segment ids."""
    B, C, S, Cg, K = 1, 32, 200, 16, 4          # _block_t(200) == 128 -> 2 tiles
    dev = torch.device("cpu")
    af = _case(B, C, S, Cg, K, dev, torch.float32, seg_kind="random", seed=31,
               requires_grad=True)
    ar = _case(B, C, S, Cg, K, dev, torch.float32, seg_kind="random", seed=31,
               requires_grad=True)
    gen = torch.Generator().manual_seed(32)
    go = torch.randn(B, C, S, generator=gen)
    out_f = fused_segment_causal_conv(*af[:3], af[3])
    out_r = segment_causal_conv_reference(*ar[:3], ar[3])
    assert torch.allclose(out_f, out_r, atol=1e-4, rtol=1e-4)
    out_f.backward(go)
    out_r.backward(go)
    for name, a, b in zip(("dx", "dw_dw", "dw_gp"), af[:3], ar[:3]):
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-4), \
            f"{name}: max |Δ| = {float((a.grad - b.grad).abs().max())}"


@INTERP
def test_interpreter_all_equal_segments_are_the_plain_causal_conv():
    B, C, S, Cg, K = 2, 32, 23, 16, 4
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, K, torch.device("cpu"), torch.float32,
                               seg_kind="one", seed=2)
    got = fused_segment_causal_conv(x, w_dw, w_gp, seg)
    want = causal_conv_reference(x, w_dw, w_gp, K)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)


@INTERP
def test_interpreter_no_cross_segment_influence():
    B, C, S, Cg, K = 1, 32, 24, 16, 4
    x, w_dw, w_gp, _ = _case(B, C, S, Cg, K, torch.device("cpu"), torch.float32, seed=6)
    seg = torch.zeros(B, S, dtype=torch.long)
    seg[:, 12:] = 1
    base = fused_segment_causal_conv(x, w_dw, w_gp, seg)
    x2 = x.clone()
    x2[:, :, 11] += 3.0
    pert = fused_segment_causal_conv(x2, w_dw, w_gp, seg)
    assert torch.equal(base[:, :, 12:], pert[:, :, 12:])
    assert not torch.equal(base[:, :, :12], pert[:, :, :12])


@INTERP
def test_interpreter_cca_call_site_reaches_the_kernel(monkeypatch):
    """The wiring, not the function: `_CCABase._cca_project(..., seg=...)` must send
    BOTH the q and the k stream through the kernel and land on the same q/k/v.

    On this host the dispatcher always picks the eager body (no CUDA tensor), so the
    device gate is patched open and the kernel runs under Triton's interpreter. The
    spy counts the calls, which is what proves the call site was rewired at all.
    """
    from morph.model import attention as attn

    torch.manual_seed(0)
    d_model, heads, kv_heads, comp, K = 128, 4, 2, 2, 4     # d_head 16 == the Cg floor
    cca = attn._CCABase(d_model, heads, kv_heads, comp, max_seq_len=64, context_len=32,
                        window_size=16, init_alpha=0.1, conv_kernel=K)
    B, S = 2, 24
    x = torch.randn(B, S, d_model)
    seg = _slot_major_seg(B, S, 4, x.device)

    ref = cca._cca_project(x, seg=seg)

    calls = []
    real = attn.fused_segment_causal_conv

    def spy(*args):
        calls.append(tuple(args[0].shape))
        return real(*args)

    monkeypatch.setattr(attn, "seg_conv_available", lambda *a: True)
    monkeypatch.setattr(attn, "fused_segment_causal_conv", spy)
    got = cca._cca_project(x, seg=seg)

    assert calls == [(B, heads * (d_model // (comp * heads)), S),
                     (B, kv_heads * (d_model // (comp * heads)), S)], calls
    for name, a, b in zip("qkv", got, ref):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), \
            f"{name}: max |Δ| = {float((a - b).abs().max())}"


# ---------------------------------------------------------------------------
# (b) CUDA forward
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seg_kind", ["slot", "random"])
@pytest.mark.parametrize("B,C,S,Cg", [(6, 512, 256, 64), (6, 256, 256, 64),
                                      (2, 128, 37, 32), (1, 64, 5, 16)])
def test_cuda_forward_matches_the_reference(dtype, seg_kind, B, C, S, Cg):
    dev = torch.device("cuda")
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, dtype, seg_kind=seg_kind, seed=1)
    assert seg_conv_available(x, w_dw, w_gp)
    got = segment_causal_conv(x, w_dw, w_gp, seg)
    ref = segment_causal_conv_reference(x, w_dw, w_gp, seg)
    oracle = segment_causal_conv_reference(x.double(), w_dw.double(), w_gp.double(), seg)
    _assert_no_worse_than_the_reference(got, ref, oracle, dtype, "forward")


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_forward_at_the_coda_axis(dtype):
    """The other call site: the full packed row, seq 1024 + 4x64 slot cells."""
    dev = torch.device("cuda")
    B, C, S, Cg = 6, 512, 1280, 64
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, dtype, seg_kind="random", seed=8)
    got = segment_causal_conv(x, w_dw, w_gp, seg)
    ref = segment_causal_conv_reference(x, w_dw, w_gp, seg)
    oracle = segment_causal_conv_reference(x.double(), w_dw.double(), w_gp.double(), seg)
    _assert_no_worse_than_the_reference(got, ref, oracle, dtype, "coda axis")


@CUDA
def test_cuda_expanded_segment_ids_need_no_materialisation():
    """`torch.arange(n).repeat_interleave(m).expand(B, L)` has batch stride 0. Reading
    it through the stride must give BIT-IDENTICAL output to the materialised copy."""
    dev = torch.device("cuda")
    B, C, S, Cg = 6, 512, 256, 64
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, torch.float32, seed=3)
    assert seg.stride(0) == 0
    assert torch.equal(segment_causal_conv(x, w_dw, w_gp, seg),
                       segment_causal_conv(x, w_dw, w_gp, seg.contiguous()))
    # and a [1, S] row broadcasts like the reference's mask does
    assert torch.equal(segment_causal_conv(x, w_dw, w_gp, seg[:1]),
                       segment_causal_conv(x, w_dw, w_gp, seg))


@CUDA
def test_cuda_force_eager_routes_to_the_reference():
    """use_kernels=False / MORPH_FORCE_EAGER must reach the eager body here too."""
    from morph.kernels.triton._eager_flag import set_force_eager

    dev = torch.device("cuda")
    B, C, S, Cg = 2, 128, 37, 32
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, torch.float32, seed=19)
    set_force_eager(True)
    try:
        assert not seg_conv_available(x, w_dw, w_gp)
        got = segment_causal_conv(x, w_dw, w_gp, seg)
    finally:
        set_force_eager(False)
    assert torch.equal(got, segment_causal_conv_reference(x, w_dw, w_gp, seg))
    assert seg_conv_available(x, w_dw, w_gp)


@CUDA
def test_cuda_unsupported_shape_falls_back_to_the_reference():
    dev = torch.device("cuda")
    B, C, S, Cg = 2, 64, 33, 8                            # Cg 8 < the tl.dot minimum
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, torch.float32, seed=23)
    assert not seg_conv_available(x, w_dw, w_gp)
    assert torch.equal(segment_causal_conv(x, w_dw, w_gp, seg),
                       segment_causal_conv_reference(x, w_dw, w_gp, seg))
    with pytest.raises(ValueError, match="unsupported shapes"):
        fused_segment_causal_conv(x, w_dw, w_gp, seg)


# ---------------------------------------------------------------------------
# (c) CUDA gradients
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("B,C,S,Cg", [(6, 512, 256, 64), (2, 128, 37, 32)])
def test_cuda_gradients_match_the_reference(dtype, B, C, S, Cg):
    dev = torch.device("cuda")
    af = _case(B, C, S, Cg, MODEL_K, dev, dtype, seg_kind="random", seed=4,
               requires_grad=True)
    ar = _case(B, C, S, Cg, MODEL_K, dev, dtype, seg_kind="random", seed=4,
               requires_grad=True)
    ao = _case(B, C, S, Cg, MODEL_K, dev, dtype, seg_kind="random", seed=4)
    ao = tuple(t.double().detach().requires_grad_(True) for t in ao[:3]) + (ao[3],)
    gen = torch.Generator(device=dev).manual_seed(21)
    go = torch.randn(B, C, S, device=dev, dtype=dtype, generator=gen)

    segment_causal_conv(*af[:3], af[3]).backward(go)
    segment_causal_conv_reference(*ar[:3], ar[3]).backward(go)
    segment_causal_conv_reference(*ao[:3], ao[3]).backward(go.double())
    for i, name in enumerate(("dx", "dw_dw", "dw_gp")):
        a, b, o = af[i], ar[i], ao[i]
        assert a.grad is not None, name
        assert a.grad.dtype == b.grad.dtype == dtype, name
        assert a.grad.shape == b.grad.shape == o.grad.shape, name
        _assert_no_worse_than_the_reference(a.grad, b.grad, o.grad, dtype, name)


@CUDA
def test_cuda_partial_requires_grad():
    dev = torch.device("cuda")
    B, C, S, Cg = 2, 128, 37, 32
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, torch.float32, seed=12)
    xr, wr_dw, wr_gp, segr = _case(B, C, S, Cg, MODEL_K, dev, torch.float32, seed=12)
    x.requires_grad_(True)
    xr.requires_grad_(True)
    gen = torch.Generator(device=dev).manual_seed(13)
    go = torch.randn(B, C, S, device=dev, generator=gen)
    xo, wo_dw, wo_gp, sego = _case(B, C, S, Cg, MODEL_K, dev, torch.float32, seed=12)
    xo = xo.double().detach().requires_grad_(True)
    segment_causal_conv(x, w_dw, w_gp, seg).backward(go)
    segment_causal_conv_reference(xr, wr_dw, wr_gp, segr).backward(go)
    segment_causal_conv_reference(xo, wo_dw.double(), wo_gp.double(),
                                  sego).backward(go.double())
    assert w_dw.grad is None and w_gp.grad is None
    _assert_no_worse_than_the_reference(x.grad, xr.grad, xo.grad, torch.float32, "dx")


# ---------------------------------------------------------------------------
# (d) CUDA: one segment == the unsegmented conv
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_all_equal_segments_are_the_plain_causal_conv(dtype):
    """One segment ⇒ the UNSEGMENTED conv — scored against its fp64 oracle, so the
    claim is about the function and not about cuDNN's rounding."""
    dev = torch.device("cuda")
    B, C, S, Cg = 6, 512, 256, 64
    x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, dtype, seg_kind="one", seed=7)
    got = segment_causal_conv(x, w_dw, w_gp, seg)
    ref = causal_conv_reference(x, w_dw, w_gp, MODEL_K)
    oracle = causal_conv_reference(x.double(), w_dw.double(), w_gp.double(), MODEL_K)
    _assert_no_worse_than_the_reference(got, ref, oracle, dtype, "seg-free identity")


# ---------------------------------------------------------------------------
# (e) CUDA: exact-zero cross-segment influence
# ---------------------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_no_cross_segment_influence(dtype):
    dev = torch.device("cuda")
    B, C, S, Cg = 6, 512, 256, 64
    x, w_dw, w_gp, _ = _case(B, C, S, Cg, MODEL_K, dev, dtype, seed=15)
    seg = _slot_major_seg(B, S, 4, dev)                   # 4 cells per slot
    base = segment_causal_conv(x, w_dw, w_gp, seg)
    x2 = x.clone()
    x2[:, :, 7] += 3.0                                    # slot 1, its last cell
    pert = segment_causal_conv(x2, w_dw, w_gp, seg)
    assert torch.equal(base[:, :, 8:], pert[:, :, 8:]), "a later slot moved"
    assert torch.equal(base[:, :, :7], pert[:, :, :7]), "an earlier position moved"
    assert not torch.equal(base[:, :, 7:8], pert[:, :, 7:8]), "nothing moved at all"


# ---------------------------------------------------------------------------
# (f) CUDA micro-benchmark — prints, never asserts
# ---------------------------------------------------------------------------

@CUDA
def test_cuda_microbenchmark_fused_vs_reference(capsys):
    dev = torch.device("cuda")
    iters = 200
    lines = ["", "segment_causal_conv: %d calls, fused vs eager reference" % iters]
    for B, C, S, Cg in MODEL_CASES:
        x, w_dw, w_gp, seg = _case(B, C, S, Cg, MODEL_K, dev, torch.bfloat16, seed=1,
                                   requires_grad=True)
        go = torch.randn_like(x)

        def run(fn, backward):
            for t in (x, w_dw, w_gp):
                t.grad = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out = fn(x, w_dw, w_gp, seg)
                if backward:
                    out.backward(go)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e3 / iters

        for fn in (fused_segment_causal_conv, segment_causal_conv_reference):
            run(fn, False)                                # warm up / autotune
        fwd_f = run(fused_segment_causal_conv, False)
        fwd_r = run(segment_causal_conv_reference, False)
        fb_f = run(fused_segment_causal_conv, True)
        fb_r = run(segment_causal_conv_reference, True)
        lines.append(
            f"  B={B} C={C} S={S} Cg={Cg}: fwd {fwd_f:.3f} vs {fwd_r:.3f} ms "
            f"({fwd_r / fwd_f:.2f}x) | fwd+bwd {fb_f:.3f} vs {fb_r:.3f} ms "
            f"({fb_r / fb_f:.2f}x)")
    with capsys.disabled():
        print("\n".join(lines))
