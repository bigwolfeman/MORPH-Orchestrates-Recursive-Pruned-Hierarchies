# Agent Note: Core-pass speed: overlap and fewer kernels, not a persistent megakernel

Status: proposed

Update 2026-10-12 evening: S1 + S2 are ON in the speed base (177.2 vs 199.3 ms), S4 measured slower and parked.
The follow-up plan measured kernel boundaries (0.70 us), grid barriers (0.69-1.75 us) and Triton vs cuBLAS GEMMs
at the core shapes, and finds that a persistent kernel does not pay on this card:
[`2026-10-12-core-pass-persistent-kernel-plan.md`](2026-10-12-core-pass-persistent-kernel-plan.md).

## Problem

Wolfe's target (2026-10-12): 9.1 TFLOP/step for everything except the training-only span
decoder, and far better than 33 % MFU. The speed base is
`morph/configs/nowta_pointer_fast_spdcells_speed_bf16c.yaml`: 198.9 ms/step wall on
`lab/perf/bench_step.py`. The brief asked whether a persistent megakernel for the slot loop's
core passes is the way to get there, and what the floor of those passes is.

The ledger the brief quoted (`/home/wolfe/morph-scratch/perf/flame2/LEDGER.md`, 261 ms span) is
from the tree before the speed package and the bf16 carrier. I profiled the current base again
with the same scripts.

**Fresh profile** (`/home/wolfe/morph-scratch/abq/megakernel/flame3/`, an export of master
`72145f6e`, graph replay of steps 61-64 aligned to a labelled eager-compiled trace, `chain.sh`
in that folder): profiled wall 202.4 ms/step, GPU busy 196.0 ms, 14,725 kernels per step, and
one idle gap of 3.8-4.3 ms per step between replays.

| region | GPU ms | kernels | note |
|---|---|---|---|
| core passes 0-7 | 62.5 | 7,246 | 9.1, 8.7, 9.5, 9.4, 7.9, 6.9, 6.1, 4.9 per pass |
| core glue (outside blocks) | 6.9 | 1,019 | row moves, freeze, fixed-point term, norms |
| gain hinge | 15.6 | 1,795 | two core applications at one pass; C4 skips its backward on inactive steps |
| prelude | 26.5 | 833 | GEMM 12.9 |
| coda | 26.5 | 707 | GEMM 13.0; strict attention backward 2.8 |
| span decoder (training only) | 24.8 | 780 | |
| LM head + CE | 18.1 | 352 | |
| optimizer + clip | 6.2 | 364 | |
| rest | 9.9 | 1,629 | |

Inside the core passes (node-level ledger, `node_ledger3.txt`): GEMM 30.9 ms (forward 9.0,
backward 20.8, the HC projection's fp32 weight-grad GEMM 1.2); custom Triton 16.6 ms (HC entry
backward 4.6, block forward kernels 4.4, CCA conv backward 3.0, strict attention backward 2.6,
CCA prologue backward 1.4, HC exit backward 0.6); inductor kernels 5.9 ms in 2,328 launches;
elementwise 4.6 ms, of which about 3.1 ms in 830 launches are the per-read fp32 gradient adds
into the shared core weights; copies and casts 2.2 ms; reductions 2.1 ms. A pass is about 920
kernels whatever its row count: pass 7 does 35 % of pass 0's FLOPs in 54 % of its time.

### The floor of the core passes (question 1)

| part | ms | how it was obtained |
|---|---|---|
| GEMM FLOPs at peak | 16.7 | 3.50 TFLOP over the 8 passes (the four core linears per block, fwd + dgrad + wgrad, `gemm_floor.py`) at 209 TFLOPS |
| GEMM at these shapes, cuBLAS, serial | 27.4 + 2.2 | measured (`gemm_floor.json`): fwd 8.5, dgrad 9.3, wgrad 9.6, plus 2.2 for the bf16 -> fp32 cast and add each read's weight grad takes today. In the model the same GEMMs take 30.9 |
| unavoidable memory traffic | about 6.7 | estimated, not measured: about 136 KB per active row per block (carrier read and write at each sublayer boundary, saved activations, carrier grad), 8,832 active rows x 6 blocks -> 4.0 ms; the K/V side over all 1,536 cells per pass 1.0 ms; weights and one fp32 grad accumulate per step 1.7 ms |
| per-kernel overhead | the rest | non-GEMM kernels average 5.1 us; about 2.5 ms per pass is fixed cost that does not shrink with the row count |

So the floor is about 17-18 ms with GEMM and memory traffic fully overlapped, and about 23 ms
with them serial. Today's 62.5 ms splits into: GEMM at peak 16.7, GEMM shape loss 14.2 (wave
quantization at 408-1,536 rows; cuBLAS runs these GEMMs at 87-190 TFLOPS and the rate jumps
with M: the down projection's forward runs at 87 TFLOPS at M = 1,464 and 168 at M = 1,104),
non-GEMM traffic floor about 6.7, and about 25 ms of non-GEMM work above that floor (kernel
count, carrier re-reads, fp32 glue). A realistic target for the core passes without a
megakernel is 38-42 ms; with the hinge and the glue, the loop-related total goes from 85 to
about 55-60 ms.

### Measurements made for this design

All under `flock /home/wolfe/morph-scratch/gpu.lock`, scripts in
`/home/wolfe/morph-scratch/abq/megakernel/scripts/`.

1. `gemm_floor.py`: the core linears at each pass's row count M_t = 24 C_t (C_t = 64, 63, 61,
   55, 46, 36, 26, 17), each op captured in a CUDA graph and replayed. Numbers above. A
   weight grad over all passes at once (K = 8,832, fp32 out) costs 8.0 ms against 11.8 ms for
   eight per-pass weight grads plus cast and add.
2. Same script under TunableOp (`PYTORCH_TUNABLEOP_ENABLED=1`, tuning on): every shape kept
   the `Default` algorithm (71 of 71 rows in the results CSV), totals within noise (29.97 vs
   29.59 ms). TunableOp on this CUDA build offers no cuBLASLt alternatives, so algorithm search
   is NOT tested by this.
3. `wgrad_overlap.py`: a stand-in for the 8-pass core backward (dgrad GEMMs plus carrier-sized
   elementwise passes on the main stream). Weight grads (GEMM + cast + add) on a second stream
   instead of inline: 30.6 / 30.9 ms -> 25.5 / 25.5 ms (two runs each). The stand-in's carrier
   kernels are torch elementwise ops of the right byte count, not the HC kernels.
4. The fresh profile above (`flame3/`).
5. Hinge on its own stream (`bench/`): a scratch-tree hack (`flame3/tree`, env
   `MK_HINGE_STREAM=1`, about 20 lines around `_slot_gain_penalty`'s call site in `_tul_core`)
   ran the hinge's two applications on a second CUDA stream; graph capture, the C4 hinge
   branch and 300 training steps ran without error. `bench_step.py`, 300 steps, skip 100,
   interleaved on, off, on, off: median 193.6 and 194.2 ms/step on, 197.5 and 199.7 off
   (p10-p90 spread about 20 ms in every run, 9 blocks each). Mean -4.7 ms (-2.4 %). Not
   checked: the byte gate, memory, the gradients.
6. `gemm_pad.py`: each core GEMM (fwd, dgrad, wgrad of `gate_up`, `down`, `W_up`) at M_t and at
   M_t rounded up to 128, 256 or 1,536 rows. The rate is a step function of M: at pass 2
   (M = 1,464) the `gate_up` dgrad takes 202.9 us and 136.6 us at M = 1,536; the `down` forward
   97.6 vs 67.0 us. The best padding per GEMM: 24.04 -> 22.45 ms over 8 passes x 6 blocks
   (-1.6 ms), most of it at passes 1-3.

## Proposal

No persistent megakernel now. Build the stages below in order. Each is its own `model.` or
`training.` key, default off and byte-identical off (the agent rules of 2026-10-12), each is
benched with `lab/perf/bench_step.py` (interleaved, 300 steps, skip 100, two repeats) and then
re-profiled with the `flame3/chain.sh` method so the next stage starts from measured numbers.

Expected savings are wall-clock ms/step against 198.9. The order is ms saved per unit of risk.

| stage | change | exactness | expected ms | gate |
|---|---|---|---|---|
| S1 | gain hinge on its own stream | exact | -4.7 (measured, hack) | byte gate |
| S2 | split backward: weight grads and their fp32 accumulation on a grad stream, all core / prelude / coda / hinge linears | exact | -7 (5-9) | byte gate |
| S3 | latent K/V cache: cache the attention in-projection's OUTPUT for frozen cells, run the projection on the active rows only | rounding only | -1 to -1.5 | fp64 parity test + fixed-weight readout |
| S4 | HC boundary fusion: exit of sublayer i and entry of sublayer i+1 in one kernel, forward and backward | rounding only | -4 (2-6) | fp64 parity + fixed-weight readout |
| S5 | the Triton autograd Functions (HC, CCA conv, prologue, strict attention) as `torch.library` custom ops, so each block compiles as one graph | rounding only | -4 (1-7) | fixed-weight readout + paired continuation |
| S6 | GEMM shape choice per pass: capacity 64 at passes 1-3 (M = 1,536), per-shape padding in the S2 op | rounding only | -1 to -1.5 | fixed-weight readout |
| S7 | one weight-grad GEMM per weight over all passes (K = sum of rows), only if S2's grad stream is on the critical path | rounding only, more precise | -1 to -3 | paired continuation |
| side | find the per-step 3.8-4.3 ms idle between replays | exact | up to -4 | byte gate |

Sum of the expected values: about -25 to -30 ms (S1 measured, the rest estimated), so about 170 ms/step (36 k tok/s) on today's
FLOPs. With the FLOP target met by the other tracker items, the MFU gain multiplies with it.

### S1. The gain hinge on its own stream

What: in `_tul_core`, the `_slot_gain_penalty` call runs under `torch.cuda.stream(hinge_stream)`
after `hinge_stream.wait_stream(current)`; an event recorded on the hinge stream is waited on by
the main stream just before `_slot_gain_reduce`; the returned tensors get `record_stream(main)`.
The hinge's forward then overlaps passes t+1..7 and the coda; autograd runs each backward op on
its forward op's stream, so the hinge's backward overlaps the main backward too.

Why exact: same kernels, same RNG offsets (the hinge already saves and restores the generator;
under capture the offsets are fixed at record time), and the autograd engine's node order and
gradient summation order do not depend on streams.

Builder must know: the stream is created once at build (never inside a capture); no stream or
event object may change across recordings (GraphStep compares module attributes: keep the
event in a local, not on the module, or reset it to None as the hack does); both C4 bodies
(A with the hinge backward, B without) must fork and join inside their capture; memory: the
hinge's activations are now live beside the main loop's, read `reserved` in the trainer log
(today 18.84 GiB) and keep 4 GB for the desktop. Test: byte gate with the key on must print the
master SHA (the change is exact); a graph-step test that replays the step with the key on
against its eager step.

### S2. Split backward: weight grads off the critical path

What: the core linears (the fused attention in-projection, `W_up`, `gate_up`, `down`; then the
prelude / coda blocks; the HC projection's weight grad) go through one custom op,
`morph::split_linear(x, w_bf16, sink)`, registered with `torch.library.custom_op` and
`register_autograd`, so inductor treats it as an opaque call. Its backward computes
`dx = dy @ w` on the current stream and enqueues on a per-device grad stream: wait on an event,
`g = mm(dy^T, x)` in bf16 (the GEMM autocast runs today), `acc.add_(g.float())`. It returns no
grad for the weight. The trainer's step body joins the grad stream after `backward()` and binds
each shadow weight's `.grad` to its accumulator. References to `x` and `dy` are kept in the sink
until the join, then dropped (no `record_stream`: inside a capture it defers every free to the
end of the capture and would grow the pool).

Why exact: today each read of a cached ternary weight has its own `_StepCachedWeight` node, which
casts that read's bf16 grad to fp32, and the engine sums the reads into `.grad` in engine order
(`morph/model/ternary_qat.py`). The op's backward is called in the same engine order and enqueues
the same bf16 GEMM, the same cast and the same adds in that order on one stream, so the sum is
the same bit for bit. The one difference is the sign of an exact zero (0 + -0), the same caveat
the C4 branch already carries. If the byte gate shows cuBLAS picking a different algorithm on the
second stream's handle, the change is rounding only and needs the fixed-weight readout.

Builder must know: every read of a weight must go through the op or none (mixed paths would
change the summation order): refuse at build. The fused in-projection reads five weights through
one `torch.cat`; its grad is split into the five accumulators by slices, as autograd does now.
The bf16 step cache (`TernaryStepCache`) is refreshed once per forward and is what the op reads;
the STE is the identity in the backward, so `.grad` of the shadow weight is the accumulator. The
join must sit inside both C4 bodies. Memory: one pass of `x` / `dy` held until the next join
point is enough; join per pass if the pool grows. Expected: the stand-in moved 5.2 ms of 30.7
off the core backward; prelude and coda add about 2 ms (their weight grads overlap the
bandwidth-bound HC backward kernels); it also removes the 830 per-read add launches and 144
casts from the critical path.

### S3. Latent K/V cache

What: under `slot_compact: gather` the core attention runs its fused in-projection
(1024 -> 1280: q_lat, k_lat, v_curr, v_prev, gate input) on all 1,536 cells every pass, because
the cache holds each frozen cell's attention INPUT (`compact_attn_input`). The projection is
row-wise, so a frozen cell's projection output is constant within a step. Cache the projection
output (q_lat, k_lat, v_curr, v_prev; the gate input is needed for active queries only) and run
the GEMM on the M_t active rows. The causal conv and the value shift still run over all
positions on the cached plus fresh latents.

Why rounding only: a row of a GEMM can round differently at another M (another kernel), and the
gradient of a frozen cell's keys is summed in latent space before the in-projection's dgrad
instead of after it (the same sum by linearity, another order).

Expected: the in-projection costs 4.2 ms over the 8 passes at 1,536 rows (`gemm_floor.json`);
at M_t rows about 2.9 ms. Also the K/V-side dgrad over frozen rows goes away. -1 to -1.5 ms.
Builder must know: the gain hinge reads the caches detached; the guard in
`validate_slot_compact` and the plans stay; the conv still reads across slot boundaries (that is
why K/V cannot be cached after the conv: a frozen slot's keys change when the slot before it is
active). Test: `tests/test_slot_compact.py`'s gather-vs-full contract, with a sabotage that
serves a stale latent.

### S4. HC boundary fusion

Measured 2026-10-12: built and exact, but SLOWER (bench 213.8 -> 219.3 ms, HC kernels 18.26 -> 21.62
ms; the projection's weight reads and the lost per-token parallelism cost more than the warm-L2
carrier read saves). Parked on `perf/hc-exit-entry` (1c220f33) to re-bench at larger width:
its note `.agents/notes/proposed/architecture/2026-10-12-hc-boundary-fusion.md` lives on that branch.

What: in a block chain, the exit of sublayer i writes the carrier and the entry of sublayer i+1
reads it back at once (projection partials, then mapping, `x_bar`). One kernel that does the
exit and the next entry's projection partials removes one carrier read per boundary; the
backward mirror fuses the entry backward's carrier kernel with the previous exit's backward. This
is the 8-kernel decode layer's `up_post` / `dn_post` pattern
(`2026-10-10-lx-decode-megakernel.md`) at training row counts. The 2026-10-08 estimate was about
2 ms for the forward alone, because the projection wants 16-token tiles.

Expected: HC Triton kernels are about 9 ms in the core passes, 2.5 in the hinge, about 9 in
prelude plus coda, about 2x their traffic floor. -4 ms (2-6). Builder must know: keep the bf16
store points of `model.hc_carrier_bf16` unchanged (the same roundings, only fewer reads); the
cotangent clip (`slot_cot_clip`) hooks the carrier between passes, so the fusion stays inside a
pass; the `inject_fold` term rides the exit kernel already. Test: fp64 references as in
`tests/test_hc_region_fused.py`, no worse than the unfused path.

### S5. One compiled graph per block

What: the HC, CCA conv, CCA prologue and strict-attention Triton autograd Functions are fenced
with `torch.compiler.disable` (`kernel_fence`), which cuts every block into many compiled pieces;
the core passes run 2,328 inductor kernels, about 290 per pass. Register them as
`torch.library.custom_op` (or `triton_op`) with `register_autograd`, so dynamo traces through
and inductor can fuse the casts, SiLU, dropout and norms between them.

Expected: the core's inductor, copy, reduce and small elementwise kernels are about 11.7 ms in
about 3,900 launches. If fusion halves the launches: -4 ms, with a wide range (1-7) because how
much inductor fuses across an opaque op is not measured. Risk: compile time, recompiles at the 8
pass shapes (the cache limit is already 256), `fail_on_recompile`.

### S6. GEMM shape choice

What: the compaction capacities C_t are 64, 63, 61, 55, ... slots per row, so passes 1-3 run
at M = 1,512, 1,464 and 1,320, where cuBLAS picks slow kernels (measurement 6). Raise C_t to
64 at passes 1-3 (rows past the active ones are inactive cells whose outputs are already
discarded, so no value the loop keeps changes; it is rounding only through the GEMM's kernel
choice). Measured per block: about 180 us saved at passes 1-3, x6 = 1.1 ms; the extra 3-14 %
rows cost a little in the non-GEMM kernels. Where padding to a non-capacity M helps (pass 5's
`gate_up` wgrad at M = 864: 138.8 -> 81.6 us at 1,024), the S2 op can pad its own GEMM inputs
with zero rows. A cuBLASLt heuristic enumeration or inductor `max-autotune` on the core blocks
is the untested next step of the same lever.

### S7. Deferred weight grads

Only if the S2 profile shows the grad stream on the critical path. One fp32-out GEMM per weight
over the rows of all passes (and the hinge): `gemm_floor.json` measured 8.0 ms against 11.8 ms
for the per-pass form. Needs every pass's `dy` kept to the end of the core backward, about 0.95
GB for the 6 core blocks; rounding only (the per-pass bf16 rounding of each weight grad goes
away).

### Side item: the idle gap between replays

The replay trace shows one gap of 3.8-4.3 ms per step with the GPU idle. Bench wall (198.9) minus
busy (196.0) agrees. Something on the host waits for the GPU once per step. Find it with
`torch.cuda.set_sync_debug_mode("warn")` around `GraphStep.step` and the trainer loop.

## Answers to the brief's questions

**Q2, options compared** (ms saved is wall-clock estimate for the whole step):

| option | ms | risk | build | constraints |
|---|---|---|---|---|
| (a) persistent megakernel per block or per pass, fwd + bwd | up to about -35 in the core, -50 with hinge and prelude / coda | high | weeks, CUDA C++ | everything below at once |
| (b) bigger GEMMs | see text | low | small | |
| (c) glue fused into block kernels | -4 (S4) to -8 (S4 + S5) | medium | medium | bf16 store points, cot clip, compaction caches |
| (d1) streams: hinge and weight grads | -10 to -13 (S1 + S2) | low | small to medium | C4 bodies, GraphStep attribute checks, memory |
| (d2) latent K/V cache | -1 to -1.5 | low | small | compaction caches, hinge |
| (d3) idle gap | up to -4 | low | small | |

(b): the 4 cells and the active slots of a pass are already one GEMM per linear: rows =
6 x C_t x 4. Fewer, larger GEMMs can come only from (i) the hinge's two applications as one call
of 2B rows, which changes their dropout masks unless the masks are copied (S1 overlaps them
instead and keeps the function), (ii) a weight grad over all passes (S7), (iii) a larger batch,
which the 18.84 GiB graph pool does not leave room for on the 5090 and which changes the recipe.
Grouped GEMM across a linear's dgrad and wgrad is what S2 gets with two streams, without a new
GEMM kernel.

**Constraints every option must respect** (training needs forward and backward; the backward is
about 2x the forward):

- Ternary STE: core weights are read through `_StepCachedWeight`, one autograd node per read,
  each read's bf16 grad cast to fp32 before the reads are summed in read order. Any change to
  where weight grads are formed must keep that order to stay exact (S2), or accept rounding
  (S7).
- No sparsity in this recipe (`prune_start`, `compact_step`, `route_start` all 999999999), so no
  BCSR path; a block kernel built now would need a MORTAR path later.
- `slot_compact: gather`: queries on M_t rows, keys and values over all cells from the caches,
  `qpos` / `perm` in the strict kernel, gradients through the caches to the pass that wrote them.
- The cotangent clip hooks the full carrier between passes; work moved across that point
  bypasses it (the slot-loop glue note measured a 60x larger gradient when it did).
- `graph_step`: fixed shapes, one graph per hinge pass (16 recordings), the C4 conditional bodies,
  no RNG draw in a backward body, `fail_on_recompile`, no module attribute that changes across
  recordings.
- Dropout masks come from torch's philox stream at the call site; a kernel that draws its own
  masks changes the trained function (statistically equal, not equal).

**Q3, which changes are exact.** Exact: S1, S2 (bit for bit, up to the sign of a zero), the idle
gap. Rounding only, no added bf16 rounding inside the loop: S3, S4, S5, S6, S7 (S7 removes one
bf16 rounding per weight grad). Function changes: a megakernel with its own dropout RNG, any FP8
inside the loop (measured: -55 % of K1-K6), any bf16 store point added to the carrier or the
hinge's fp32 path. Rounding-only changes here sit near 1e-7 relative in fp32 sums, against about
2^-4 for FP8 e4m3, so the FP8 result does not predict them; the block-glue keys of 2026-10-08
were of this kind and moved the fixed-weight readout by 4e-5 nats CE and -0.42 % K1-K6. The gate
for them is the fixed-weight readout plus one paired continuation each (the 2026-10-08 speed
plan's acceptance), not a 5k run per stage; one 5k package run at the end.

**Q5, what not to do and why.**

- A persistent fwd + bwd megakernel for a block or a pass, now. Its ceiling is real (core passes
  62.5 -> 25-30 ms). But it must re-implement, forward and backward: the Cayley HC mapping, the
  CCA projection with a causal conv across cells, the prologue (QK mean, norms, temperature,
  CoPE, GQA, value shift), the strict span-masked attention with compaction and sinks, the gate
  and residual alpha, the ternary SwiGLU MLP, dropout, saved activations for autograd, and the
  compaction caches. A block forward has at least 7 grid-wide dependencies (the decode
  megakernel note counted them); the backward has more, about 100 per block-pass in all, 4,800
  per step, at 1-3 us each: the same order as the kernel boundaries it removes. Its GEMMs on
  sm_120 are `mma.sync` code that must first match cuBLAS (87-190 TFLOPS at these shapes) before
  anything else pays, and it must be rewritten for Hopper (wgmma, TMA). Every recipe change of the
  last month (fan, pointer, decoder cells, bf16 carrier, compaction) would have meant a rewrite.
  Revisit only when (1) the architecture is frozen and (2) after S1-S5 the core passes still sit
  above 2x their floor (about 45 ms). S2's split backward, S3's latent cache and S4's boundary
  kernels are pieces such a kernel would reuse.
- FP8 activations inside the loop: measured cost to K1-K6 (-55 %) is far over the 5 % rule.
- Triton GEMMs in place of cuBLAS for the core linears, to fuse SwiGLU into the epilogue: the
  epilogue saves about 2 ms of inductor traffic, and a Triton GEMM that is 10 % slower than
  cuBLAS at these shapes loses 3 ms. Only after S5 shows what inductor already fuses.
- More single-op folds without removing launches (the 2026-10-07 measurement: 3 % for two weeks).
- L2 persistence windows (the 2026-06 campaign measured a net loss).

## Alternatives considered

- **Persistent megakernel (the brief's option a).** Rejected for now, reasons and trigger
  condition in Q5. The decode megakernel (`LxFusedDecoder`, 8 kernels per layer) did not build
  one either, for the same dependency count, and it had no backward.
- **Grouped GEMM or fixed M buckets for capture (option b).** The pass is already one GEMM per
  linear; capture already has fixed M per pass. The measured lever in this family is the GEMM's
  step-wise rate in M (S6), not grouping.
- **Algorithm search with TunableOp.** Tried: no alternatives offered on this CUDA build. A
  cuBLASLt heuristic enumeration or inductor `max-autotune` for the core blocks is the untested
  way to do it (S6).
- **A larger batch to raise M.** The largest lever for MFU on this model (per-step kernel cost
  is fixed, so it halves per token at 2x batch), but the graph pool (18.84 GiB) leaves no room on
  the 5090 and it changes the recipe. Wolfe's call; the cloud GPU is where it belongs.
- **Capacity bound over valid slots only.** About 17 % of the compacted rows at passes >= 1 are
  pad or inactive filler. Rejected in the compaction note: a capacity that depends on the valid
  count is not a fixed shape.
- **Caching K/V after the conv.** A frozen slot's keys change when the slot before it is active
  (the conv reads across slots); exact recompute set saves about 8 % of the in-projection work.
  Caching the projection output (S3) is exact in math and saves 28 %.

## Acceptance criteria

- Each stage: its key off is byte-identical (`FINAL_SHA256 4a73ebf5...` on the 45-step gate);
  S1 and S2 also byte-identical ON (else they are rounding only and take the readout gate);
  `bench_step.py` interleaved, two repeats, saving at least half of the stage's expected value
  or the stage is dropped; reserved memory reported.
- Rounding-only stages: fp64 parity of the new kernel no worse than the path it replaces;
  fixed-weight readout (`core_depth_sweep.py --depths 1,6 --rows 480`) CE@6 and K1-K6 within 5 %;
  one paired continuation.
- After S1-S5: a new `flame3/chain.sh` profile; the core passes plus hinge plus glue below
  60 ms; then the 5k package run against the speed base (CE@6, K1-K6 within 5 %).
- The megakernel decision is re-taken with that profile.

## Risks

- S1 and S2 add concurrency: the pool grows by whatever is live on both streams at once. The
  hinge stream run (measurement 5) did not record memory.
- S2 replaces how weight grads are formed for every linear it covers; a missed read path changes
  the summation order silently. The build-time refusal and the byte gate are the guard.
- S2 must work inside inductor-compiled regions; a custom op with a side effect in its backward
  can be reordered by AOTAutograd. Test the accumulation order directly (a test with two reads
  whose order matters in fp32).
- S5's gain is the least certain number here.
- The estimates for S2-S5 are from a stand-in and from the profile, not from built code. Only S1
  was run in the model, as a hack without a byte gate.
- The profile is 4 steps; the hinge's backward runs only on active steps, so its 15.6 ms is a mix.
