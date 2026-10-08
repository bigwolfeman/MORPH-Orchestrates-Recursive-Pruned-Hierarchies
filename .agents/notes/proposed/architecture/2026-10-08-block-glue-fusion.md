# Agent Note: block glue fusion (HC region kernels, tiled CCA prologue, injection fold)

Status: proposed

## Problem

On the FAST2 graph-replayed step of `lxtul_pointer` (6 x 1280 positions, the 15 graph-step
keys), the non-GEMM work inside the blocks cost more than the attention and MLP math it
surrounds. The census (`perf/graph/blockfuse/census.md`, one prelude block and one core pass,
graph replay aligned to a labelled trace) put the HC residual at 55 ms of a 354 ms kernel sum
and the CCA prologue at 25 ms.

- HC. The kernels ran near DRAM bandwidth, but each sublayer touched the `[T, 4, 1024]` fp32
  carrier 3 times on entry (a bf16 copy for the projection GEMM, the GEMM, the pre-map kernel)
  and 11 times in the backward (two carrier-grad kernels, an `addmm_` that read and wrote the
  carrier grad, autograd's add of the two, an fp32 SIMT GEMM over the carrier for grad_w).
- CCA prologue. One 64-element program per (row, head), 8 warps each: the kernels moved 27 MB
  in 461 us forward. Launch geometry, not bandwidth. The backward added about 16 small host-side
  adds, reduces and casts per call.
- Injection. Every prelude and coda block added its injection term to the carrier in a separate
  pass (`_apply_injection`), and autograd summed the term's grad over the streams in another.

## Proposal

Three `model.` keys, default off. With all three off the step is the old step bit for bit.

- `model.cca_prologue_tiled` (`fused_cca_prologue.py`, `_FusedCCAPrologueRows`). One program
  holds 16 rows by every head as D = 64 tiles; the RoPE partner column is a second load at the
  rotated index. Inputs are read by their strides (no `.contiguous()` copies). The backward
  combines the Q and K paths in registers, stores input grads in their final dtype and layout,
  and writes weight and temperature grads as per-program partials summed once.
- `model.hc_region_fused` (`fused_hyper_connection.py`, `_HCRegionEntry` / `_HCRegionExit`).
  The entry returns an alias of the carrier (`h.view_as(h)`); the exit reads the alias, so the
  exit's carrier grad, gout itself, reaches the entry backward through autograd and the entry
  writes the whole carrier grad once. Forward: a per-stream projection kernel (bf16 tl.dot,
  fp32 accumulate, the result rounded to bf16 as the autocast mm rounded it) and a map kernel
  (mapping, x_bar). Backward: a reduce kernel (the 20 C-reductions per token), a map kernel
  (the mapping VJP) and a carrier kernel (Hres^T gout + x_bar path + rms path + graw @ W as an
  fp32 FMA dot). grad_w stays the fp32 GEMM the old path ran. A kernel splits C over a second
  grid axis when its token blocks alone leave SMs idle (the core's 1536 positions). The region
  path runs only on CUDA with n = 4, an fp32 carrier and bf16 autocast on; one shared predicate
  (`_hc_region_active`) decides for both entry and exit, and anywhere else both fall back to the
  reference path.
- `model.inject_fold` (`transformer.py` `_front_tail`, `_back_region`). Prelude block i's
  injection is added by block i-1's HC exit kernel (`next_inject_term`, an existing block
  argument) and block 0's by the stream expand; the same for coda blocks 1..3. Forward
  bit-identical (the same fp32 adds); the term's grad is summed in the exit kernel. Needs
  `hc_region_fused` (the old post kernel returned the term's grad in bf16).

## Alternatives considered

- One fused kernel per HC backward (reductions, mapping VJP and carrier write in one program,
  rows re-read from L2). Built first. It held about 200 per-token mapping values in registers
  (255 per thread, 1 program per SM, 16.7 % occupancy) and ncu read 77 % DRAM busy at a 38 %
  L2 hit rate: the re-read went to DRAM anyway. Split into three kernels it is equal at 7680
  positions and faster at 1536 (in-step HC kernel time 41.5 -> 33.8 ms with the forward split).
- One fused entry forward (projection then x_bar from the L2-resident rows, persistent grid).
  Equal at 7680 positions (190 vs 195 us cold), 1.6x slower at 1536 (96 programs of 16 tokens
  for 170 SMs, each streaming all of W). Removed; the two-kernel form is the only form.
- Exit forward fused with the next sublayer's entry (one kernel writes the carrier and
  projects it for the next sublayer). Estimated at about 2 ms per step: the post kernel is per
  token and the projection needs 16-token tiles to amortise W. Not built.
- Computing grad Hres in the exit backward (it reads gout anyway). Moves a carrier read from
  one kernel to another; traffic unchanged. Not built.
- A bf16 projection in the backward (graw @ W, grad_w). Faster, but its error is above the old
  fp32 path's; the brief's contract is error no worse than the existing path. Not built.

## Acceptance criteria

Run 2026-10-08 on `/home/wolfe/morph-wt-graph` (frozen snapshot
`perf/graph/blockfuse/snap_bf`, md5 of the sorted `morph/**/*.py` list `a9f1999d`):

- `tests/test_cca_prologue_tiled.py`: 5 passed. fp64 reference at 7680, 1536, 111 (tile tail,
  trailing positions without RoPE) and 128 (contiguous) positions; for q, k, v and every
  input grad the tiled error is no worse than the per-row kernels' error (2 % slack); grads
  keep their input layouts. Sabotage (RoPE VJP sign flipped) -> 4 failed, reverted.
- `tests/test_hc_region_fused.py`: 13 passed. fp64 reference through `HyperConnectionResidual`
  at 7680, 1536, 111 (+ inject term) and 10 positions: output and grads of carrier, proj
  weight, proj bias, sublayer weight and term no worse than the `hc_fused_grad` path (2 %;
  grad_proj_b 5 %, set from an 8-seed measurement, ratio 0.987-1.028, mean 0.999). Also the
  unused-outputs entry, the no-autocast fallback (bit-identical), the CPU fallback, the
  constructor refusals, and `inject_fold` bit-identical forward on the tiny winner. Sabotages:
  a wrong grad-Hres product -> 4 failed; stream 3's projection partial dropped -> 5 failed;
  the last prelude fold term dropped -> the fold test failed. All reverted.
- `tests/test_graph_step.py`: 15 passed, with a new `blockfuse_keys` case (all three keys on)
  whose captured replay equals the eager step bit for bit.
- 45-step deterministic gate (`gate.sh`, FAST2 overrides): keys off reproduces the 02:48
  `tf_all` gate's final state hash bit for bit (`1e77a754`). Keys on first differs at step 0
  (loss 24.32347 vs 24.32302, the bf16 projection rounding pattern); over 45 steps mean |dloss|
  0.011, max 0.054, mean of the last 10 losses 22.9893 vs 22.9948.
- Fixed-weight readout (`core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on
  `lxtul-pointer-rerun/step_5000.pt`, call counters confirm the new kernels ran): CE@6 3.918688
  vs 3.918730 (+4.2e-5), K1-K6 0.009450 vs 0.009410 (-0.42 %).
- Speed (`r1007/bench.sh`, 420 steps, FAST2 vs FAST2 + the three keys, alternating, twice,
  from the snapshot): 342.6 / 342.8 ms per step off, 300.8 / 301.1 on, so -41.8 ms (-12.2 %),
  17.93k -> 20.42k tok/s, reserved 21.18 -> 20.93 GB. 4-step graph profiles of the same tree
  (`perf/graph/blockfuse/prof/off_graph.json`, `final_on_graph.json`): kernel sum 340.2 ->
  298.0 ms. By family: CCA prologue -18.9, fp32 adds -8.3, SIMT GEMMs -4.7, bf16 copies -4.6,
  tensor-core GEMMs (the projection GEMV) -4.6, other -4.8, HC Triton kernels +3.6.

The brief's target was -50 ms. The measured cut is -41.8 ms. What is left of the in-block glue
after these keys (labelled profile, `prof/census_on.pkl`): HC Triton kernels 33.6 ms, about 10
carrier passes per core sublayer at bandwidth; core weight-grad accumulation adds inside the
compiled block backward 3.3 ms (autograd's AccumulateGrad over 10 passes, not block glue);
partial-sum reduces 2.5 ms; conv layout copies 2.8 ms; CCA prologue 3.6 ms. The largest
remaining step glue (16 ms adds, 11 ms copies) sits in the CE heads, which are out of scope here.

## Risks

- Not bit-identical with `hc_region_fused` or `cca_prologue_tiled` on. A run that turns them on
  mid-way changes the rounding pattern; the readout above says by about 4e-5 nats.
- `inject_fold` covers the prelude and coda only (the core's per-pass injection and coda block
  0's add stay separate). It refuses the old post kernel.
- The launch shapes (`_HCR_TUNE`) are tuned for the 5090 at C = 1024. Other widths compile
  (the tiny test model at C = 64 runs) but are not tuned.
- The region path changes the inject term's grad from bf16 (old post kernel) to the term's own
  dtype. That is a precision gain, and a change.
- Not measured: the cloud width (d 2048), other GPUs, `hc_fused_norm` (refused with the region
  key), long-run training curves.
