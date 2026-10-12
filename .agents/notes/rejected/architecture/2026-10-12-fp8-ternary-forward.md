# Agent Note: FP8 forward GEMMs for the ternary MLPs

Status: rejected — FP8 activation rounding inside the slot loop cost 55 % of K1-K6 and 0.007 CE@6 at 5k (lab/experiments/failures/2026-10-12-fan-speed-lever-split.md) for 1.7 % of the step

## Problem

MORPH trains ternary weights with a forward STE: codes in {-1, 0, +1} times one per-tensor
scale (`morph/model/ternary_rule.py`, `norm_match`). Those codes are exact in e4m3, so a forward
GEMM can run on the 5090's FP8 tensor cores (about 2x bf16) if only the activations are
quantised. The idea was that GEMMs are ~110 ms of a ~190 ms step and ~40 % of GEMM FLOPs are
forward.

## Phase 1 numbers (RTX 5090, sm_120, torch 2.11.0+cu128)

`torch._scaled_mm` on this stack supports e4m3 x e4m3, bf16 output, per-tensor scalar scales,
and row-wise scales (`scale_a` fp32 `[M,1]`, `scale_b` fp32 `[1,N]`; a row-wise `scale_a` with a
scalar `scale_b` is refused: "Invalid scaling configuration"). It captures in a CUDA graph.
Odd M (7, 401) works. Output error of a ternary-weight GEMM against bf16, row-wise e4m3
activations: 2.65 % relative Frobenius (activation rounding only; codes and scale are exact).

Microbench (`/home/wolfe/morph-scratch/abq/fp8-fwd/bench2.py`, graph-captured, row-wise
quantiser compiled by Inductor, quantiser INCLUDED), d_model 1024, d_ff 2816:

| GEMM | M=400 | M=800 | M=1500 | M=3072 | M=6144 |
|---|---|---|---|---|---|
| gate_up (1024 -> 5632) bf16 -> fp8 total (us) | 26.8 -> 29.7 (0.90x) | 57.4 -> 33.6 (1.71x) | 109.5 -> 56.8 (1.93x) | 211.7 -> 98.6 (2.15x) | 370.5 -> 186.7 (1.98x) |
| down (2816 -> 1024) | 27.1 -> 23.0 (1.18x) | 28.8 -> 25.0 (1.16x) | 108.8 -> 44.1 (2.47x) | 144.6 -> 84.1 (1.72x) | 287.2 -> 143.5 (2.00x) |

Gate passed: about 2x at the 6144-row shapes (prelude, coda), 1.5x to 2.5x at 1000 to 1500 rows,
break-even or worse at 400 rows or fewer. Per-tensor scales are 10 % faster still, but the
row-wise scale is the one that survives a token with a large outlier, so the build uses it.

## Proposal

Built behind `model.fp8_ternary_fwd` (default false), uncommitted in the worktree.

- `_TernaryFP8Fwd` (`morph/model/layers/block_sparse.py`, beside the old `_FP8DenseMatmul`):
  row-wise e4m3 activations, `torch._scaled_mm`, backward is the `F.linear` backward on the saved
  bf16 `x` and bf16 STE weight (STE dgrad and wgrad stay bf16).
- `CMSBlockLinear.enable_fp8_ternary_fwd` / `_forward_fp8_ternary`; `carve()` refuses the flag.
- The e4m3 codes and the `[1, out]` row scale ride the existing `TernaryStepCache`
  (`TernarySTE.bind_step_cache(fp8_fwd=True)`, `_fill_fp8_codes`): refilled once per forward from
  the bf16 cache (`sign(wq)` is the code, the row `amax` is the scale), so the FP8 forward and the
  bf16 backward share the same numbers. A deep copy (the fan EMA twin) is unbound and keeps the
  `F.linear` path.
- Covers the dense `CMSBlockLinear` MLPs (prelude, core, coda; 28 layers in the fan recipe).
  Refused: no step cache, a bias, dims not divisible by 16, a carved layer, no qualifying layer.

## Result

Tests: `tests/test_fp8_ternary_fwd.py` (8 tests), with three sabotage runs that each fail the
file. Key-off byte gate: `FINAL_SHA256 4a73ebf5...bd70d7`, identical to master.

Bench, `nowta_pointer_fast_spdcells_speed`, 300 steps, graph_step, interleaved, trainer `sps`
at step 280 (mean of the last 100 steps): key off 7.18 and 7.21 steps/s (139.3 and 138.7 ms),
key on 7.31 and 7.32 (136.8 and 136.6 ms). Gain: 2.5 ms per step, 1.8 %. Peak memory +0.11 GB.

Why only 1.8 %: the premise overcounted. In the fan recipe the ternary MLP forward GEMMs are
about 1.5 of 15.3 TFLOP per step (the LEDGER rows: prelude 0.69, coda 0.74, core passes 0.07 to
0.20 each, all fwd, and MLPs are only part of those). The big GEMM time is backward (2x the
forward FLOPs), the span decoder (3.7 TFLOP) and the LM head, none of which are ternary-forward.
Saving half of ~5 ms of forward MLP GEMM time gives the ~2.5 ms measured.

## Alternatives considered

- Per-tensor activation scale: 10 % faster GEMM, loses the per-token range. Not taken.
- FP8 dgrad (grad_out in e4m3/e5m2 times the exact codes): the same kind of saving on the other
  half of the MLP GEMMs, and the weight side stays exact. Not built (the task fixed the backward).
  This is the larger prize: dgrad FLOPs equal the forward FLOPs.
- Deriving the codes inside the STE instead of from the bf16 cache: would duplicate the rule that
  `ternary_rule.py` owns. Not taken.

## Acceptance criteria

The 5k quality check (orchestrator) must hold CE and K1-K6 inside the paired spread before this
key joins a recipe. The numerics are not bit-identical (activation rounding ~2.6 % per layer).

## Risks

- Activation noise in the looped core compounds over 6 passes; the 5k check is the only test.
- Small-row core passes (400 rows or fewer) can run slower in FP8; there is no row threshold.
- Not verified: eval/generation paths with FP8, `compile_core_dynamic` recompiles on new shapes
  beyond the 300-step runs, carve and the in-run MORTAR path (refused), the JAX mirror.
