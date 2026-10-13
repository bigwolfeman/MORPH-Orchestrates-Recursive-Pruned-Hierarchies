# Agent Note: Core-pass speed after S1-S6: Triton GEMMs and the flat K/V-side kernels, not a persistent kernel

Status: proposed

## Problem

Wolfe wants far better than 33-37 % MFU on the fan recipe
(`morph/configs/nowta_pointer_fast_spdcells_speed_bf16c.yaml`). The earlier note
[`2026-10-12-core-pass-megakernel-design.md`](2026-10-12-core-pass-megakernel-design.md)
deferred a persistent megakernel and proposed S1-S7. Status today: S1 (hinge stream) and S2
(weight grads on a side stream) are merged with keys off, bench 215.05 -> 196.80 ms/step.
S4 (HC exit + entry fusion) was exact and slower (213.8 -> 219.3 ms), because the fused kernel
re-read the skinny projection's weights from L2 and lost the exit's per-token parallelism.
S3, S5 and S6 are in build. The brief: a staged plan that takes the core passes plus the gain
hinge (the hinge runs the same blocks) from about 55-60 ms after S1-S6 toward 25-30 ms.

This note answers four questions with numbers. Scripts and raw output are in
`/home/wolfe/morph-scratch/abq/megakernel2/`. Every GPU run used the GPU lock. The absolute
desktop load was present, so read ratios, not absolute ms.

### Q1. Per-kernel fixed cost or per-kernel bytes?

The traces alone cannot tell. The old note called the M-independent time "about 2.5 ms per
pass of fixed cost". Two new measurements split it.

**Measurement 1, the cost of one kernel boundary** (`chain_bench.py`, `cg_sync.cu`). A chain
of 400 dependent `y = x + 1` kernels, captured in one CUDA graph and replayed, results checked
exact (fp32 `x + 1`, every chain correct):

| tensor (fp32, in + out bytes) | plain, us/kernel | PDL, us/kernel | effective BW |
|---|---|---|---|
| 1 K elements (8 KB) | 0.70 (CUDA C: 0.72) | 0.51 (CUDA C: 0.58) | launch only |
| 393 K (3 MB) | 1.13 (CUDA C: 1.28) | 0.89 (CUDA C: 1.21) | 2.8 TB/s |
| 1.57 M (12.6 MB) | 2.76 | 2.51 | 4.6 TB/s |
| 6.3 M (50 MB) | 8.94 | 8.81 (early trigger: 9.40) | 5.6 TB/s |

So one graph-replayed kernel boundary costs 0.70 us on this card. The L2 is 96 MiB, so a whole
core pass's carrier (1,536 rows x 4 streams x 1,024 x bf16 = 12.6 MB) stays in L2 and bytes
between kernels move at 4.6-5.6 TB/s.

**Measurement 2, the core-pass time that does not scale with rows**
(`scripts/pass_scaling.py` on the flame3 trace, a least-squares fit `ms = a + b * M_t` over
passes 1-7). Per pass, the M-independent part is 3.55 ms: GEMM 1.02, non-GEMM 2.53. Over the
8 passes that is 28.4 of 62.5 ms. Its parts:

| M-independent part, 8 passes | ms | what it is |
|---|---|---|
| kernel boundaries | about 5.1 | 7,246 kernels x 0.70 us (measured cost x counted kernels) |
| K/V side over all 1,536 cells | about 8.7 | CCA conv backward (`_gp_bwd_dw_kernel` 1.36, `_dw_bwd_kernel` 1.03, `_gp_bwd_dx_kernel` 0.36, its reduce 0.55), CCA prologue rows backward 1.31 + reduce 0.71, forward `_gp_fwd_kernel` 0.68 and `_cca_rows_fwd_kernel` 0.67, the flat part of `_bwd_dkdv_kernel` 0.68, copies 0.62 |
| weight-sized work | about 3.5 | per-read fp32 grad adds 1.92 (S2 moves them off the critical path), casts and copies 1.6 |
| GEMM | 8.2 | wave quantization and bad cuBLAS picks at small M (`wmma 32x32` kernels at M = 1,464, 1,320, 864, 624, 408) |
| rest | about 3 | HC map and reduce kernels, small inductor kernels |

Answer: per-kernel fixed cost is real but small, about 0.7 us x the kernel count, about 8 % of
the core passes (about 6.4 ms with the hinge). Bytes are cheap because they sit in L2. Most of
the waste is work that does not shrink with the active rows: the K/V side over all cells, and
a few kernels whose run time is set by low parallelism, not by bytes. Example:
`_gp_bwd_dw_kernel` takes 28 us per block and pass at pass 1 and at pass 7 alike, for a few MB
of L2-resident input. The second large part is GEMM efficiency at M = 408-1,536.

### Q2. Cheaper alternatives to a megakernel, measured

**PDL.** Triton 3.6.0 (installed) supports it: the `launch_pdl=True` launch option sets
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, and `tl.extra.cuda.gdc_wait` /
`gdc_launch_dependents` emit `griddepcontrol`. Inductor in torch 2.11 has
`TORCHINDUCTOR_ENABLE_PDL=1` (`config.triton.enable_pdl`, sm_90 and up; it inserts the wait
and the trigger itself, never in Triton GEMM templates). PDL survives `torch.cuda.graph`
capture on sm_120: the chain got faster and stayed exact. Measured gain: 0.19 us per trivial
boundary, 0.24 us at 3-12.6 MB, about 0 at 50 MB, and an EARLY `gdc_launch_dependents` costs
0.46 us at 50 MB (the waiting dependents occupy SM slots). At a cuBLAS boundary PDL gives
nothing: `mm` alone 12.09 us, `mm` + elementwise 13.03 plain and 13.12 with PDL (M = 408).
About 3,500 of the 7,246 core kernels are Triton or inductor kernels; the rest are ATen
elementwise, reduce and copy kernels and cuBLAS GEMMs, which cannot take PDL. Ceiling: about
0.7 ms in the core passes, about 1.5 ms over the step. A real-block PDL chain was NOT measured:
the custom kernels have no `gdc_wait` today, and launching them with PDL without it is a race.

**Grid barrier cost (the price of a persistent kernel).** Measured, not assumed:
`cooperative_groups::this_grid().sync()` costs 0.69 us at 170 CTAs (one per SM), 0.64-0.84 us
at 340 and 1.75 us at 680. A Triton cooperative launch with an atomic barrier costs 0.67, 0.90
and 1.86 us. A persistent Triton kernel that does the same `y = x + 1` phases with one barrier
per phase is NOT faster than the graph-replayed chain at any size: 1.38 vs 1.13 us at 3 MB,
2.97 vs 2.76 at 12.6 MB, 9.2-10.1 vs 8.9 at 50 MB. A grid barrier costs the same as a kernel
boundary under graph replay. A persistent kernel that keeps today's dependency structure saves
nothing.

**CUDA graph node tricks.** The replay trace shows 0.09-0.23 ms of idle per pass in 900+
kernels, so the graph executor adds no idle that node edits could remove. The 0.70 us boundary
is GPU-side (launch, ramp, drain). The only graph-level lever is parallel branches, which is
what S1 and S2 do with streams. Not measured separately.

**Fewer, larger kernels per sublayer.** A sublayer kernel that runs HC entry -> norm -> GEMM ->
HC exit must give each CTA whole rows across the GEMM's N. At M = 408 and 64-row tiles that is
7 CTAs on 170 SMs. That is the S4 failure at a larger scale. Rejected. Fusion that keeps the
GEMM's tile grid (elementwise epilogues) is safe.

**Persistent / small-tile GEMMs (the largest lever found).** `triton_gemm_bench.py`: a plain
tiled Triton `tl.dot` GEMM (mma.sync on sm_120), 11 autotune configs, at the core shapes (fwd,
dgrad and wgrad of `gate_up` 1,024 -> 5,632, `down` 2,816 -> 1,024, in-projection 1,024 ->
1,280; M = 1,536, 1,104, 408), each op graph-replayed. Sum over 21 shape-ops: cuBLAS 1,209.7 us
(141 TFLOPS), Triton 1,004.0 us (170 TFLOPS, 81 % of the 209 peak), ratio 0.830. Best case
x0.44 (`gate_up` wgrad at M = 408: 63.2 -> 28.0 us). Worst case x1.30 (`gate_up` dgrad at
M = 408: 28.7 -> 37.3 us). The winners are 64x64 and 64x128 tiles: more tiles fill 170 SMs
better than cuBLAS's 128x128 and 256x128. The difference from cuBLAS is at most one bf16 ulp of
the output (rel. 5e-3 of the max, from cuBLAS split-K). Not tested: the M = 1,464 and 1,320
shapes (S6 rounds those away), stream-K, and in-model numbers.

### Q3. Is a megakernel still the answer?

No, not on this card. The measured price of a grid barrier equals the measured price of a
kernel boundary, so persistence buys nothing by itself. The only form that could win is a
tile-dataflow kernel with per-row-tile counters instead of barriers, and its total prize is
bounded by the boundary cost, about 6.4 ms of core plus hinge. It would need, forward and
backward: the Cayley HC mapping, CCA projection and causal conv across cells, the prologue,
strict attention with compaction, the ternary SwiGLU, dropout and saved activations. The
dependency structure per block, read from the forward and backward code paths:

- Forward, grid-wide: attention keys and values over all cells (every query reads every key)
  and the conv halo across cells. Row-tile only: HC entry, norm, in-projection, gate,
  `W_up`, `gate_up`, SwiGLU, `down` (needs the whole d_ff row of its tile), HC exit.
- Backward, grid-wide: dK/dV over all queries; every weight grad (a reduction over all rows);
  the HC projection weight grad; the conv weight grads. Row-tile: the dgrad chain.

So about 2 grid-wide and 8 row-tile dependency points per block forward, about 6-8 grid-wide
and 10 row-tile in the backward. With barriers that is about 17 x 6 blocks x 8 passes = 816
barriers x 0.69 us = 0.56 ms of barriers alone, the same order as the boundaries they replace.

Language, if it is ever built: Triton is enough for the GEMM tiles (measured above: it beats
cuBLAS here). Triton 3.6 cannot express a cross-op tile scheduler with spin-waits on per-tile
flags and producer/consumer warps cleanly; that part needs CUDA C++ with mma.sync (CUTLASS's
sm_80-style collectives run on sm_120; wgmma and TMA do not exist on sm_120). ThunderKittens
has mma.sync kernels but was not checked on sm_120.

Constraints any fused kernel must keep (they bind the stages below too):

- Dropout: masks come from torch's philox stream. A custom kernel that draws its own masks
  changes the trained function (statistically equal, not equal) and needs a 2-seed 5k check
  against matched controls. Plan: no stage below moves dropout into a custom kernel.
- `slot_compact: gather`: queries on M_t rows, keys and values from the caches over all cells,
  `qpos` / `perm` in the strict kernel, grads through the caches. GEMMs see the gathered rows.
- Ternary: weights are read from `TernaryStepCache` (bf16, refreshed once per forward). A GEMM
  op reads that tensor as its B operand. The STE is the identity in the backward.
- `slot_cot_clip` hooks the carrier between passes. Nothing below crosses a pass boundary.
- `graph_step`: fixed shapes per pass, 16 hinge recordings, the C4 bodies, no RNG in a backward
  body, `fail_on_recompile`. Triton autotune must finish in the eager warm-up; a shape that
  is not tuned when capture starts must RAISE, not tune inside the capture.
- HC Cayley mapping: unchanged by every stage below.

### Q4. The ceiling, stated honestly

Core plus hinge GEMM work is about 4.3 TFLOP per step (core 3.50 from the shapes; hinge about
0.8, estimated from its GEMM time share of 7.18 / 30.93 ms). At the measured Triton rate
(170 TFLOPS) that is about 25 ms. At the 209 TFLOPS peak it is 20.6 ms. With every weight grad
hidden on the grad stream, the forward and dgrad GEMMs alone are about 17 ms, and the
non-GEMM traffic floor is about 7 ms (old note's estimate). So 25-30 ms for core plus hinge is
the asymptote of a perfect kernel set, not a reachable plan. The stages below reach about
42-50 ms. Reaching 25-30 needs fewer FLOPs, not better kernels: the hinge (15.6 ms busy, about
18 % of loop time), the K/V side over all cells, or a larger batch (no memory today). Those are
recipe changes for Wolfe to decide.

## Proposal

Start from the S1-S6 tree. Each stage is its own key, default off, byte-identical off (agent
rules 2026-10-12). Each is benched interleaved (off, on, off, on; 300 steps; skip 100) and
re-profiled with `flame3/chain.sh` so the next stage starts from measured numbers. The hinge
takes every stage for free because it runs the same blocks.

| stage | change | basis of the estimate | step ms, expected (range) | kill if |
|---|---|---|---|---|
| G0 | probe: inductor `max_autotune_gemm` with Triton templates on the compiled regions, S2 off. No code, a config probe. | none, it is the measurement | sets the G1 prior | in-model core GEMM time falls < 10 %: drop G1 |
| G1 | Triton GEMMs for the core, hinge, prelude and coda linears inside S2's `split_linear` op (dgrad on the main stream, wgrad on the grad stream) and a forward op; per-(M, N, K) autotune in warm-up; the wgrad epilogue rounds to bf16 and adds into the fp32 accumulator, which removes the cast and add kernels | isolated GEMMs: -17 % (measured); in-model: not measured | -5 (-3 to -8) | bench saves < 2 ms, or the fixed-weight readout moves CE@6 or K1-K6 by > 5 % |
| P0 | PDL: `TORCHINDUCTOR_ENABLE_PDL=1`, `launch_pdl` plus a `gdc_wait` before the first dependent load in every custom Triton kernel and in the G1 GEMMs; no early trigger | synthetic chain: -0.19 to -0.24 us per Triton-to-Triton boundary (measured); boundary count from the trace | -1 (-0.5 to -1.5) | bench saves < 0.5 ms |
| K1 | the flat K/V-side kernels: re-tile the CCA conv weight-grad kernels and the prologue rows backward for parallelism (split the position reduction, two-phase sum); one conv weight grad over all 8 passes, like S7; compare against the in-tree kernel at the real shapes first | trace: 5.8 ms flat over 8 passes in 6 kernels, about 1.2 more in the hinge (measured); their floor about 1 ms (estimated from bytes) | -4 (-3 to -5.5) | the new kernels are < 2x faster in a microbench at the real shapes |
| F1 | GEMM epilogue fusion that keeps the tile grid: SwiGLU in the `gate_up` forward epilogue, the SwiGLU backward in the `down` dgrad epilogue, the residual alpha scale | guessed from the inductor kernel times around these GEMMs | -2 (-1 to -3) | bench saves < 1 ms |

Sum: -12 (-8 to -18) ms per step. Core plus hinge from about 55-60 ms to about 42-50 ms; the
step from about 175 to about 160-167 ms. Only the isolated GEMM ratio, the boundary and PDL
costs, the barrier cost and the flat-kernel times are measured. Every in-model saving is an
estimate.

**First stage: G0 then G1.** G0 is a config probe that a builder can run in an hour. It gives
the in-model ceiling of Triton GEMMs before anyone writes an op. G1 then merges on its own: it
is one op path that S2 already owns, it has no RNG, and it makes the GEMM boundaries PDL-able
(cuBLAS boundaries are not), which P0 needs. It is rounding only (another accumulation order;
cuBLAS uses split-K on some shapes), so it takes the fixed-weight readout
(`core_depth_sweep.py --depths 1,6 --rows 480`) and one paired continuation, not the byte gate.

**Megakernel trigger.** Re-take the decision only if, after G1, P0, K1 and F1, the profile
shows the kernel-boundary share (kernels x 0.70 us) above 15 % of the core passes AND the
architecture is frozen. Then the design to cost is the tile-dataflow form, not barriers.

Exposure to the S4 lesson (small-M fusion loses parallelism or re-reads weights):

- G1: low. Smaller tiles give more CTAs, not fewer. At M = 408, N = 1,024 even 64x64 tiles give
  112 tiles on 170 SMs; that shape wants split-K or stream-K, which is untested.
- K1: medium. A rewrite that gives one CTA per channel group would repeat S4. The kill test is
  a microbench at the real shapes before any model change.
- F1: low while the epilogue only adds elementwise work. High if anyone moves the HC
  projection or a row norm into a GEMM prologue: that is S4 again.
- A megakernel: high. Its GEMM tiles must keep at least one full wave at M = 408.

## Alternatives considered

- **Persistent megakernel with grid barriers, forward only, for the core passes.** Rejected
  with numbers: a barrier (0.64-0.69 us at 170-340 CTAs, 1.75 us at 680) costs what a kernel
  boundary costs (0.70 us), and persistent phases ran 3-22 % slower than the kernel chain. The
  forward is also only about a quarter of a pass (2.4 of 9.1 ms at pass 0).
- **Tile-dataflow megakernel (per-tile counters, no barriers).** Deferred. The prize is bounded
  by the boundary cost, about 6.4 ms of core plus hinge, and the build is weeks of CUDA C++ that
  every recipe change would break. Trigger condition above.
- **PDL as the main lever.** Measured too small: about 0.2 us per eligible boundary, 0 at cuBLAS
  boundaries. Kept as P0 because it is cheap.
- **One kernel per sublayer.** Rejected: it gives each CTA whole rows across a GEMM, the S4
  failure mode.
- **CUTLASS or cuBLASLt algorithm search instead of Triton GEMMs.** Not tested. TunableOp on
  this build offered no cuBLASLt alternatives (old note). Triton already beats cuBLAS by 17 % on
  the shape set and keeps one language for the epilogues.
- **Stream-K GEMMs.** Not measured. The small-tile Triton kernel already reached 170 TFLOPS on
  the set; stream-K is the candidate for the shapes with fewer tiles than SMs (M = 408, 624).

## Acceptance criteria

- Each stage: key off byte-identical (`FINAL_SHA256 4a73ebf5...` on the 45-step gate);
  interleaved bench, two repeats, saving at least half the stage's expected value; reserved
  memory reported against the 0.34 GiB free after eval (S1+S2 left that little).
- G1: an autotune cache filled in warm-up; a test that an untuned shape under capture raises;
  fp64 parity of the Triton GEMM no worse than cuBLAS against an fp64 reference; fixed-weight
  readout CE@6 and K1-K6 within 5 %.
- P0: the byte gate with the key on prints the master SHA (PDL changes timing, not math).
- K1: fp64 parity per kernel; a sabotage test that serves a stale conv weight grad.
- After G1, P0 and K1: a fresh `flame3/chain.sh` profile; core plus hinge plus glue below 50 ms;
  then one 5k package run against the speed base (CE@6, K1-K6 within 5 %, 2 seeds, because one
  run cannot test the 5 % rule; see `lab/experiments/successes/2026-10-12-bf16-carrier-matched-controls.md`).

## Risks

- G1's 17 % is from isolated GEMMs with warm L2 and fixed inputs. In the model the S2 grad
  stream already hides part of the weight-grad time, and S6 removes the worst cuBLAS picks, so
  the in-model gain can be half the isolated one. G0 exists to measure this first.
- G1 autotune adds compile time per shape (8 passes x about 12 GEMM kinds) and must not run
  inside a capture.
- Triton GEMMs on the grad stream compete with main-stream kernels for SMs; better wave fill on
  the main stream leaves fewer idle SMs for the grad stream, so G1 and S2 savings do not add.
- K1's floor is a byte estimate. The conv weight-grad reduction may need fp32 atomics, which
  are not deterministic; a two-phase sum keeps it deterministic at one more kernel.
- The hinge FLOP figure (0.8 TFLOP) is inferred from its GEMM time share, not counted.
- All per-stage savings are estimates except the microbenchmark inputs listed in Q1 and Q2.
  The 4-step flame3 profile mixes hinge-active and hinge-inactive steps.
