"""fused_cca_prologue precision: the kernel rounds once, at the store.

The first forward kernels rounded every intermediate to bf16 (qk-mean, pre-norm sum,
normalized value, norm * weight, exp(temp), cos/sin, RoPE products), whatever the input
dtype. Measured 2026-10-10 at the production widths (8 heads, 4 kv heads, d_head 64):
  * fp32 inputs: q / k relative error vs fp64 3.8e-3 / 4.6e-3; the eager reference 6e-8.
  * bf16 inputs, fp32 parameters (training autocast): q / k 3.9e-3 / 4.7e-3, about 2.5x the
    error of the fp64 result rounded once to bf16.
The fixed kernels compute in fp32 and round once to the output dtype. The bounds below
hold the two contracts, for both kernel families (per-(row, head) and row-tiled):
  1. fp32 inputs: q and k within 1e-6 relative of fp64 (fp32 arithmetic noise), v exact,
     every input gradient within 1e-5 relative of fp64.
  2. bf16 inputs: q and k no worse than 1.05x the one-rounding floor (fp64 rounded to
     bf16), v exact.
Both bounds fail on the pre-fix kernels by more than 1000x (contract 1) and 2x (contract 2).

CUDA only (Triton kernels).
"""

import pytest
import torch

from test_cca_prologue_tiled import NAMES, _err, _inputs, _run, _run64

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP32_OUT_TOL = 1e-6      # q / k rel. error, fp32 inputs (measured ~1e-7 after the fix)
FP32_GRAD_TOL = 1e-5     # every input grad rel. error, fp32 inputs
FLOOR_SLACK = 1.05       # bf16 inputs: error / one-rounding floor (measured ~1.00)

CASES = [
    (6, 1280, 0),     # prelude / coda block
    (6, 256, 0),      # core pass
    (3, 37, 5),       # rows not a multiple of the tile; trailing positions skip RoPE
]


@pytest.mark.parametrize("tiled", [False, True], ids=["per_row", "tiled"])
@pytest.mark.parametrize("B,S,n_skip", CASES)
def test_fp32_inputs_match_fp64(B, S, n_skip, tiled):
    ins, cos, sin, gos = _inputs(B, S, strided=False)
    ins = [t.float() for t in ins]
    o_ref, g_ref = _run64(ins, cos, sin, gos, n_skip)
    outs, grads = _run(ins, cos, sin, gos, n_skip, tiled=tiled)
    for name, o, r in zip("qkv", outs, o_ref):
        assert o.dtype == torch.float32, (name, o.dtype)
        e = _err(o, r)
        tol = 0.0 if name == "v" else FP32_OUT_TOL
        assert e <= tol, f"{name}: rel error {e:.3e} vs fp64 > {tol:.0e} (fp32 inputs)"
    for name, g, r in zip(NAMES, grads, g_ref):
        e = _err(g, r)
        assert e <= FP32_GRAD_TOL, f"grad_{name}: rel error {e:.3e} vs fp64 > {FP32_GRAD_TOL:.0e}"


@pytest.mark.parametrize("tiled", [False, True], ids=["per_row", "tiled"])
@pytest.mark.parametrize("B,S,n_skip", CASES)
def test_bf16_inputs_round_once(B, S, n_skip, tiled):
    # Production layouts; the norm weights and temp stay fp32, as under training autocast.
    ins, cos, sin, gos = _inputs(B, S, strided=True)
    o_ref, _ = _run64(ins, cos, sin, gos, n_skip)
    outs, _ = _run(ins, cos, sin, gos, n_skip, tiled=tiled)
    for name, o, r in zip("qkv", outs, o_ref):
        assert o.dtype == torch.bfloat16, (name, o.dtype)
        floor = _err(r.to(torch.bfloat16), r)      # fp64 rounded once to the output dtype
        e = _err(o, r)
        assert e <= FLOOR_SLACK * floor, (
            f"{name}: rel error {e:.3e} vs fp64 > {FLOOR_SLACK} x one-rounding floor "
            f"{floor:.3e} (bf16 inputs)")
