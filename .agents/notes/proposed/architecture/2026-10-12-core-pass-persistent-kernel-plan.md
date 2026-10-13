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

## Status 2026-10-13: G0 passed, G1 built (key off, 5k check owed)

Scratch, logs and scripts: `/home/wolfe/morph-scratch/abq/gemm/`. Speed base
`nowta_pointer_fast_spdcells_speed_bf16c`, `model.slot_gain_pass_force=3` on every arm, 300 steps, skip 100,
2 interleaved repeats, shared GPU.

**G0 (inductor `max_autotune_gemm`, Triton backend only, S2 off).** Env
`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`.
Step: 186.20 ms (186.27, 186.08) to 175.49 ms (174.18, 175.79), -10.7 ms (-5.7 %).
Kineto window, steps 160-164: GEMM kernel time per step 95.99 to 84.48 ms (-12.0 %). The sum counts every GEMM
kernel in the step, not only the core's; the profile has no per-region labels in graph replay, so the core-only
share is not separated. Kill rule (< 10 %) not met, so G1 was built.

**G1 (`model.split_wgrad_triton`, default off, needs `split_wgrad`).** Triton GEMMs for forward, dgrad and wgrad
inside `split_linear` (`morph/kernels/triton/split_gemm.py`). The wgrad epilogue rounds to bf16 and copies or adds
into the fp32 sum. Tuning runs once per (mode, M, N, K, strides) in the eager warm-up; an untuned shape under
capture raises.

| check | result |
|---|---|
| byte gate, key off | `FINAL_SHA256 4a73ebf5426b9dfa889c81a8608a23d23a6785bb225ff32e2d5dd03476bd70d7` (matches) |
| pinned bench, S2 on, key off vs on | 174.81 ms (174.77, 174.86) vs 168.10 ms (168.09, 168.20): -6.7 ms (-3.8 %). Kill rule (< 2 ms) not met |
| profile GEMM time per step, S2 on | 103.60 to 91.45 ms (-11.7 %) |
| kernel launches per step | 14,322 to 13,976 (-346): the cast and add kernels go; no launch increase |
| fixed-weight readout through the split op's forward (train scope), 480 rows, step_5000 of `ab-bf16b` | CE@1 3.9118 vs 3.9118, CE@6 3.9059 vs 3.9060, K1-K6 +0.0058 vs +0.0058 (cuBLAS vs Triton forward) |
| memory, 135-step run with one eval | graph reserved 19.69 vs 19.63 GiB (key off vs on); device free after eval 6.93 vs 6.99 GiB on this run |
| tests (`tests/test_stream_overlap.py`) | 36 passed, 10 of them new; three sabotage edits each fail their test |

Not verified: the readout covers the forward GEMMs only (it has no backward); the backward is covered by the
Triton arm versus cuBLAS arm test on a tiny model (loss and grads close) and by the fp64 parity tests. No 5k
run: the quality check (2 seeds, CE@6 and K1-K6 within 5 %) is owed to the orchestrator. The 0.34 GiB free
figure of S1 + S2 was not reproduced by this run (desktop load differs); the on-versus-off difference is the
number to trust (+0.06 GiB). The G0 and G1 savings do not add: both replace the same cuBLAS GEMMs.

**G1 follow-up 2026-10-13.** `GraphStep` takes an optional `capture_tuner` (wired only under
`model.split_wgrad_triton`). A recording that meets an untuned GEMM key collects it and runs the cuBLAS
equivalent; the keys are tuned on synthetic tensors, the graphs are dropped, and the step is recorded again
(the CUDA generator state is restored, or the re-record would shift later draws). A second pass with a
pending key raises. Test: `test_the_recording_tunes_the_keys_the_warm_up_never_ran`. On the speed base with
`block_custom_ops` (S5), pinned bench: base 171.07 ms vs G1 164.21 ms, -6.9 ms (-4.0 %). Unpinned 300-step
runs of the speed base and of the `cellp3` config (64 graphs) both finished with no re-record, because the
warm-up had already tuned every key those recordings reach.

## Status 2026-10-13: K1 built (key off, 5k check owed), P0 built (key off, borderline)

Scratch, logs and scripts: `/home/wolfe/morph-scratch/abq/kvflat/`. Speed base
`nowta_pointer_fast_spdcells_speed_bf16c` (S1, S2, S5, G1 ON, cell count 4), `model.slot_gain_pass_force=3` on
every arm, 300 steps, skip 100, interleaved arms, shared GPU.

**Fresh K/V-side profile** (kineto window, steps 160-164, per step; the plan's 1.36 / 1.03 / 0.36 numbers were 8
passes only and predate S5 and G1). `_gp_bwd_dw_kernel` 1.84 ms (60 launches, 30.7 us, 144 programs),
`_dw_bwd_kernel` 1.53 (25.5 us, 9,216 programs), `_gp_bwd_dx_kernel` 0.49, `_cca_rows_bwd_kernel` 2.33
(68 launches, 34.2 us; the 96-program core calls and the 480-program prelude/coda calls average together),
`_gp_fwd_kernel` 0.81, `_cca_rows_fwd_kernel` 1.29. The core shapes are B = 6, C = 768 (q 512 + k 256),
S = 256, CG = 64, K = 4 for the conv and B = 6, S = 256, H = 8, Hkv = 4, D = 64 for the prologue. The torch
`sum` and cast kernels after the conv and prologue backward are not named in the trace.

**K1 kill test (microbench `lab/perf/bench_kv_flat.py`, CUDA-graph replay, warm L2).** Conv backward as the
model calls it (all kernels, reduces and casts): 61.9 us in tree, 20.6 us re-tiled, x3.00. Prologue rows backward
(with its partial sums): core shape 6 x 256 31.2 us to 14.7 us, x2.12; prelude/coda shape 6 x 1280 56.5 us to
36.0 us, x1.57 (that shape is not on the 2x rule; it is the 7 prelude and coda blocks only). Kill rule (< 2x at the
real shapes) not met, so K1 was built. What changed: the grouped conv dx and dw share one launch and dw is split
over (group, tap, position chunk) with the accumulator in registers; the depthwise backward tiles 4 channels; the
prologue rows backward splits the kv groups across programs below 192 row blocks; all weight-grad partials go
through one fixed-order sum kernel (no atomics) that also casts and transposes. The "one conv weight grad over
all 8 passes" idea was NOT built: after the re-tile the per-pass partials cost 0.31 + 0.86 ms per step in total
(`_kvf_dw_bwd_kernel`, `_kvf_sum_kernel`), and a deferred weight grad needs S2's side-stream plumbing for the conv.

| check | result |
|---|---|
| byte gate, key off (K1 code, then K1 + P0 code) | `FINAL_SHA256 4a73ebf5426b9dfa889c81a8608a23d23a6785bb225ff32e2d5dd03476bd70d7` both times |
| fp64 parity per kernel (`tests/test_kv_flat.py`) | flat error vs fp64 no worse than in-tree (2 % slack + 5e-7 floor) for dx, dw_dw, dw_gp and all nine prologue grads, 4 conv shapes and 4 prologue shapes. bf16 grads read 2.3e-3 (conv) and 1.7e-3 (prologue) for BOTH paths; the three fp32 vector grads read 1.4e-7 to 1.8e-7 for both, with d_temp at 6 x 1280 at 4.4e-7 vs 2.3e-7 (another sum order, inside the floor) |
| tests | `test_kv_flat.py` 18 passed; `test_block_custom_ops.py` 22, `test_cca_prologue_tiled.py` 5, `test_cca_prologue_precision.py` 12, `test_segment_causal_conv_kernel.py` 37 (16 skipped) all pass on the edited kernels |
| sabotage | (a) `cca_conv_bwd_flat` serving the first call's `dw_gp` -> `test_flat_conv_does_not_serve_a_stale_weight_grad` fails; (b) tap and channel axes swapped in the depthwise partial store -> `test_conv_backward_matches_fp64` fails. A third test shows the stale check itself catches a stale sum. All restored |
| fixed-weight readout, step_5000 of `ab-bf16b`, 480 rows, depths 1,6, compiled blocks | key off and on identical: CE@1 3.9117, CE@6 3.9060, K1-K6 +0.0057 [+0.0053, +0.0062], spandec K1-K6 +0.0317. EXPECTED: the key changes the backward only and the readout is forward only, so this row cannot test the key |
| gradient parity (the readout's batch, 3 rows, depth 6, bf16 autocast, plain backward, 415 grads) | key off twice: relative difference 1.4e-8 (deterministic). Key on vs off: global relative difference 3.95e-3, cosine 0.99999222, per-tensor worst 3 % (alpha) and 0.5-2 % elsewhere. Same size as the bf16 error of either path against fp64 (2.3e-3), so it reads as rounding, not a different gradient. Not checked: a per-tensor fp64 gradient of the whole model |
| pinned bench, 300 steps, 2 repeats, off vs on | 164.12 ms (164.17, 164.10) vs 160.20 ms (160.05, 160.25): -3.9 ms (-2.4 %). Second session, 3 repeats: 164.18 vs 160.26, -3.9 ms. Keep rule (>= 2 ms) met |
| profile (steps 160-164) | conv backward kernels 3.86 ms to 1.27 ms (`_kvf_gp_bwd` 0.96, `_kvf_dw_bwd` 0.31) plus `_kvf_sum` 0.86 (128 launches, also sums the prologue partials); `_cca_rows_bwd` 2.33 to 1.06; launches per step 13,216 to 12,840 |
| memory | graph pool 19.40 GiB both; device free after eval 6.96 (off) vs 6.97 GiB (on); one extra fp32 partial buffer per call (1.5 MB), freed at the end of the backward |
| 300-step quality proxy (not a verdict) | loss at steps 100 / 200 / 280, off: 16.836 / 14.247 / 12.805; on: 16.840 / 14.287 / 12.832. MORPH runs decorrelate within tens of steps, so a 0.03 gap is inside the noise (memory note "n=1 run comparisons are unreadable") |

Not verified for K1: a 5k run (CE@6 and K1-K6 against matched controls, 2 seeds) is owed to the orchestrator. The
fp32 vector grads (`wq`, `wk`, `temp`) sum in another order; no run longer than 600 steps used the key.

**P0 REJECTED 2026-10-13; its code is removed (`_pdl.py`, `tests/test_pdl.py`, the `PDL` kernel parameters, `model.pdl`). It broke training on the speed base past the byte gate: [`lab/experiments/failures/2026-10-13-k1-p0-5k.md`](../../../../lab/experiments/failures/2026-10-13-k1-p0-5k.md). The text below is the history.**

**P0 (`model.pdl`).** Triton 3.6.0 and torch 2.11.0 support it: `launch_pdl` is a launch option,
`tl.extra.cuda.gdc_wait` emits `griddepcontrol.wait`, and `torch._inductor.config.triton.enable_pdl` exists (the key
sets it; inductor then inserts its own waits). 35 custom kernels got `PDL: tl.constexpr = False`,
`if PDL: tl.extra.cuda.gdc_wait()` as the first statement and `**pdl_opts()` at every launch: HC region and
`_hc_post_fwd`, strict attention (4), CCA prologue rows (2), CCA conv (5), segment conv (5), kv-flat (3), G1
`_mm_kernel`, CE (3), int row quant, exact reduce (2) and the two AdEMAMix kernels. Left alone: decode kernels, the
non-step attention kernels (CSA, HCA, window, flash, GLA), the router kernels, `morph_set_cond`. Off, the kernels
compile as before (the constexpr is False and `pdl_opts()` is empty).

| check | result |
|---|---|
| byte gate, key OFF | `FINAL_SHA256 4a73ebf5...bd70d7` |
| byte gate, key ON (`model.pdl=true`) | `FINAL_SHA256 4a73ebf5426b9dfa889c81a8608a23d23a6785bb225ff32e2d5dd03476bd70d7` (same) |
| tests (`tests/test_pdl.py`) | 75 passed: static (all 35 kernels end their signature in `PDL: tl.constexpr = False`, wait first, every launch passes `**pdl_opts()`, `launch_pdl` appears only in `_pdl.py`), `griddepcontrol.wait` in the PTX with PDL on and absent with it off, conv + G1 GEMM chain bit-equal PDL on vs off in eager AND replayed from a CUDA graph on sm_120 |
| sabotage (on a scratch copy of the tree) | wait line deleted from `_kvf_sum_kernel` -> `test_every_pdl_kernel_waits_first` and `test_the_wait_reaches_the_ptx` fail; `**pdl_opts()` removed from the `_mm_kernel` launch -> `test_every_launch_passes_pdl_opts` fails |
| pinned bench 1 (3 arms, 2 repeats) | off 163.69 (163.62, 163.86) vs pdl 163.23 (162.84, 163.32): -0.46 ms |
| pinned bench 2 (4 arms, 3 repeats) | off 164.18 (164.17, 164.36, 163.71) vs pdl 163.23 (163.12, 163.23, 163.52): -0.95 ms. kv 160.26 (160.26, 160.48, 160.12) vs kv + pdl 159.58 (159.70, 159.43, 158.42): -0.68 ms |

Verdict on the keep rule (>= 0.5 ms): the two sessions read -0.46 and -0.95 ms for PDL alone and -0.68 ms on top of
K1; the five paired repeats give -0.78, -0.54, -1.05, -1.13, -0.19 ms (mean -0.74). It clears 0.5 ms on the pooled
mean and on three of four reads, and misses it by 0.04 ms on one session's medians. The plan expected -1 ms (-0.5 to
-1.5). It is small and near the noise floor of this bench (off arms differ by up to 0.7 ms between sessions); keep
it default off and decide on the next bench. Not verified: PDL under an eval forward in the trainer (the eval after
step 290 ran on a PDL-off build only), the 5k quality (PDL cannot change math, and the byte gate agrees), memory with the key on (not measured), and whether
inductor's early `gdc_launch_dependents` helps or hurts on its own (it is on whenever the key is on).

## Status 2026-10-13: F1 built (`model.gemm_epilogue_fusion`, key off, 5k check owed)

Scratch: `/home/wolfe/morph-scratch/abq/epi/`. Speed base `nowta_pointer_fast_spdcells_speed_bf16c` (S1, S2, S5, G1 on;
K1 and P0 off), `model.slot_gain_pass_force=3`. The MLP is one split op: SwiGLU in the `gate_up` forward epilogue
(two accumulators per tile, writes `gu` and `h`) and the SwiGLU backward in the `down` dgrad epilogue (reads saved `gu`,
writes `d_gu`). Saved tensors equal the unfused path's. The residual alpha scale sits inside `_hc_post_fwd_kernel`, not on a
GEMM output, so it was not touched; the dropout after `down` (0.38 ms) stays out (torch RNG). The new kernels take no PDL code.
Kill test (profile, steps 160-164, keys off): SwiGLU forward kernel 0.64 ms, backward 2.15 ms (sum 2.79 ms, 3.2 ms with the
fused-variant kernels) against the 1.5 ms rule. Pinned bench, 3 repeats: 164.28 ms off vs 162.83 ms on (-1.45 ms).
Not verified: 5k quality (2 seeds), the model's backward against a per-tensor fp64 gradient.
