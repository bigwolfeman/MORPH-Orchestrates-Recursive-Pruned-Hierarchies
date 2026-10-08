# Agent Note: LXTUL training speed plan, from the flame graphs

Status: proposed

## Problem

The LXTUL + pointer winner trains at about 16k tok/s on one 5090 with the fastest recipe we
have (graph-captured, 390 ms per 6144-token step). Parcae's testbed runs the same model at the
same FLOPs at about 22k. Wolfe's planning arithmetic (2026-10-08): 600B tokens in about 80 hours
is 2M tok/s for a fleet; on this model that is 7.6 PFLOP/s sustained, about 130 5090s at today's
28 % MFU, about 66 at 55 %, about 19 H100s at 40 %. The cloud model (4:8:4, d 2048) costs about
4-5x more per token. Every speed factor won here divides the fleet and the bill.

Measured (2026-10-08, `/home/wolfe/morph-scratch/perf/flame/`, LEDGER.md and png/):

- 21.67 TFLOP per step; 390 ms GPU span, 386.5 busy; GEMM + attention 154 ms at 141 TFLOPS;
  everything else 232 ms. Ideal at 180 TFLOPS: 120 ms. The graph removed host time, so every
  further millisecond is GPU work.
- By region (ms): core loop 114.3, span decoder 65.7, LM head + CE 50.4, prelude 42.2, coda 39.6,
  gain hinge 29.5, EMA twin 19.2, fan diversity + lsel 11.2, rest 18.
- Parcae, same labels (ms): core 72.5, hinge 14.4, span decoder 26.0, token CE 25.9, prelude 17.3,
  coda 19.5, twin 5.5 (its 50.5 ms Muon optimizer is the only region where MORPH is faster).
- Removable compute (audit, `/home/wolfe/morph-scratch/perf/audit/AUDIT.md`), each with a
  bit-equality test and a positive control: span decoder dead rows (49.4 % of its rows) and its
  detached-head weight gradient (1.24 TF), the hinge on inactive steps (71 % of steps), checkpoint
  recompute (+3.1 GB to drop), core dead cells (18 % of cell-passes), LM-head slot rows, coda zero
  cells, twin slot rows: 6.2-7.2 TF, 28-32 % of the step's FLOPs. Discarded slot-pass work is
  40.7 % of the core's passes (the part beyond the dead cells is load-bearing: later slots read
  frozen slots' states).
- Noise floor (filed 2026-10-08): the same production code rerun at the same seed reads K1-K6
  +0.0095 against the winner's +0.0155. Single 5k runs cannot see a 5 % K1-K6 change (about
  0.0006).

## Proposal

Target for this phase: **at least 32k tok/s (190 ms/step) on one 5090 with the winner's
function**, then 40k+ with the near-equivalent cuts. The levers, in order of ms per unit of work.
Owners are subagents; every change is a default-off key until it passes its check.

### A. Exact or rounding-only (no change to the trained function), in flight tonight

| lever | region now | target | agent | measured 2026-10-08 (bench pairs, 420 steps) |
|---|---|---|---|---|
| span decoder: dead rows (static cap 6792, Lemma 9), no dW for the detached head, fused CE | 65.7 | ~15 | cehead | fed3aafc: 53.5 -> 28.3 ms (dW skip exact) |
| LM head CE: fused kernel reaches the pointer/DITTO loss (slot rows not dropped yet, Lemma 3) | 50.4 | ~22 | cehead | fed3aafc: 49.2 -> 25.3 ms; both keys 373 -> 324 ms |
| EMA twin: shared table reads (exact), compiled twin | 19.2 | ~6 | twinfan | 4284c374: exact part -4.3 ms; compile -1.6 ms; twin still 13.75 ms (HC/CCA kernels compile cannot reach) |
| fan diversity / lsel: fp32 batched, bf16 pick | 11.2 | ~3 | twinfan | 4284c374: fan/lsel 11.2 -> 4.3 ms; three keys together 388.8 -> ~377 ms |
| online prelude as router teacher (changes objective) | 13.75 | 0 | twinfan | -13 ms more (362 ms); needs a paired continuation |
| core + hinge checkpoint recompute off | ~30 | 0 | coreloop | 22ee2ffb: byte-identical gate; 373.8 -> 352.1 ms; +4.6 GiB pool (6.5 GB free) |
| hinge skipped when inactive | 29.5 | ~15 | coreloop | not built: needs a CUDA conditional node; the active-cell hinge is not exact (inactive cells serve K/V) |
| hinge reuses the loop's f(h) | | | coreloop | built, off, NOT near-equivalent on active steps (total-gradient cosine 0.85-0.95): rejected |
| HC eager reference fixed, HC always kernel | (correctness) | | hcgap | bc75eec1 |

Measured 2026-10-08 08:36 (bench pairs, frozen tree, 420 steps): FAST2 343.0 / 343.2 ms
(17.9k tok/s); every committed key together, `morph/configs/lxtul_pointer_fast.yaml`: 266.7 /
266.8 ms = **23.0k tok/s**, about 2x eager production (~525 ms) and level with Parcae's testbed
(23.5k). Compaction (f1ac1b9c, `slot_compact: gather` + `slot_depth_stratified`) is part of it.
Block glue (caab8d86, `hc_region_fused` + `cca_prologue_tiled` + `inject_fold`): -41.8 ms on
FAST2, and on the fast recipe 265 -> **230.0 / 229.8 ms = 26.7k tok/s** (09:43 bench pairs; the
trainer's replay rate reads ~30.5k). Left: LM-head slot rows (Lemma 3), the logits recompute GEMM,
the hinge's inactive-step backward (CUDA conditional node), a fused exit-plus-next-entry HC kernel
(~2 ms), core injection fold, and the carrier itself (HC with 1 stream was worth 60 ms at FAST).

### B. Carrier and sublayer glue (the HC/CCA Triton kernels are about 75 ms; copies/casts and
fp32 adds/muls about 100 ms across the step)

- Fuse each sublayer's HC pre-map + norm into the GEMM's input stage and the HC post-map into the
  GEMM's output stage, so a sublayer reads and writes the 4-stream carrier once. Triton (portable
  to Hopper for phase 3), cuBLAS kept for the big GEMMs where an epilogue cannot be attached.
- Fold the HC mixing gradients that run as fp32 SIMT / tiny GEMMs (22 ms) into the kernels.
- The core attention sublayer is wider than the core MLP (4.6 vs 3.75 ms per pass) at a quarter
  of the FLOPs: CCA Q/K Triton, copies and fp32 reduces around a 0.25 ms attention kernel.
- Measured ceiling (agent `carrier`, 2026-10-08, bench pairs within 3 ms): HC at 1 stream saves
  60 ms of 389 (HC Triton 32.8 -> 3.8, copy/cast -14, elementwise -8, GEMM -13); no HC at all
  saves 70 ms. So the carrier is a large lever but not the largest: the CE heads (116 ms) and the
  core loop + hinge (144 ms) come first. A bf16-STORED carrier with an exact fp32 mixer saves only
  21 ms; its step-0 total loss moves 24.32 -> 25.73 with CE unchanged, i.e. the finite-difference
  gain hinge reads bf16 storage as gain (the hcgap split: storing the skip in bf16 alone is +5.2e-2
  of step-0 loss). A bf16 carrier therefore also needs the hinge to probe in fp32 or by JVP.
- History: the only past bf16-carrier attempt (`e316c05f`, 2026-08-22, branch perf/bf16-carrier,
  never merged) measured x1.16 with loss traces within 0.005 nats and never had a quality run; no
  record calls it bad. It also rounded Hres to bf16, which the 2026-10-06 HC audit found slightly
  expansive (sigma_max 1.07 mean over 96 applications).
- Bug found on the way: `model.hc_streams=1` cannot be graph-captured (`_cayley_ref` calls
  `torch.linalg.solve`, a host sync; the 1x1 case is exactly 1).

### C. Near-equivalent cuts (change the trained function; need the paired check)

- Core compaction: run only the active cells of each pass (40.7 % of slot-pass work is discarded),
  with finished slots exposing their last pass's K/V to later slots (fixed-weight eval delta
  -2.0e-5 [-8.9e-5, +4.9e-5]). Static shapes for graph capture by drawing slot depths with fixed
  per-pass counts (stratified Poisson: the per-slot marginal stays Poisson(6) clamped to 1..8,
  only the joint changes), and the index-tensor attention kernel reading slots by id.
- Online prelude (stop-grad) as the router's teacher instead of the EMA twin (teacher pick agrees
  93-98 % per pass).
- FP8 forward GEMMs: the ternary weights are exactly representable in FP8 e4m3 (codes {-1,0,1}
  times one per-tensor scale), so only the activations are quantised; 5090 and H100 both have FP8
  tensor cores at about 2x bf16 throughput. GEMMs are 154 ms of the step today.

### Acceptance (Wolfe 2026-10-08: within 5 % of baseline loop contribution, not bit-exact)

- A and B (rounding only): fp64 parity per kernel; at FIXED WEIGHTS on the step_5000 checkpoint,
  CE@6 and K1-K6 within 5 % of the production readout (anchor 0, one run answers it); the 45-step
  gate's step-0 differences reported. No 5k run is needed: the noise floor shows that rounding
  chaos alone moves a 5k run's K1-K6 by 0.006.
- C (function changes): a PAIRED continuation: resume the production rerun's step_2500 twice, with
  and without the change, same data order and seed, 1000 steps; compare K1-K6 and CE@6 on the 480
  rows. Measured paired noise (no change, three draws, 2026-10-08, `lab/experiments/mixed/`):
  K1-K6 spread 0.0012, CE@6 0.0007, against 0.006 / 0.009 between independent 5k runs. 5 % of
  K1-K6 there is 0.00045, below the paired spread, so a cut needs three or more pairs and is
  judged on the mean; CE@6 is the sharper readout. The FAST2 package plus the online teacher
  passed this check (CE@6 unchanged in 3 runs; K1-K6 2 of 3 inside, one draw high). One full 5k run of the final
  combined recipe is read against the noise-floor runs.

### Lean

`leanprove` formalises the audit's nine lemmas (dead cells, coda zero cells, slot-row deadness,
twin slot rows, inactive hinge, hinge f0 value, span-decoder dead rows, detached dW, static row
cap) with each lemma's code-level conditions turned into runtime asserts or tests, so a cut cannot
silently become wrong when the architecture changes. Near-equivalences get a proven bound where
one exists.

### Phases (Wolfe's three)

1. Now (LXTUL speed): A, B, C as above.
2. After LXTUL closes: Wolfe's own low-hanging list.
3. Deployment target: multi-GPU data parallel with the captured step (NCCL inside the graph,
   sharded optimizer state), Hopper tuning (Triton autotune for SM90, FP8), and only then a
   megakernel if the profile still shows glue: the architecture must be frozen first, and a
   hand-written SM120 kernel would need a rewrite for H100 (wgmma/TMA).

## Alternatives considered

- **A training megakernel now.** After A, the step is about 240 ms with 154 ms of GEMM; a
  megakernel can only remove the remaining non-GEMM glue (about 85 ms), which B targets with
  per-sublayer fusion at a fraction of the cost and without freezing the architecture. Revisit in
  phase 3.
- **More single-op folds (phase 2 of 2026-10-07).** Measured: 3 % for two weeks of kernels; the
  costs are spread over thousands of small kernels per region, so only region-level fusion or
  removing the work pays.
- **Bigger batch to raise core GEMM efficiency (M = 1536 rows, 117 TFLOPS).** Memory-bound by the
  graph pool today; reconsider after the checkpoint and carrier changes settle memory.

## Acceptance criteria

- Phase A merged: >= 26k tok/s on the FAST recipe, fixed-weight CE@6 and K1-K6 within 5 %.
- Phase B: >= 32k tok/s.
- Phase C: >= 40k tok/s with the paired continuation inside its measured paired noise.

## Risks

- Memory: dropping checkpoints (+4.6 GB) and the graph pool must leave >= 4 GB for the desktop.
- Several agents edit one worktree; each benches from a frozen snapshot.
- The near-equivalent cuts change the training function; a paired check that cannot resolve 5 %
  would leave them unproven, and they stay off.
- The UPS: no multi-core CPU work while the GPU is saturated (compile threads capped at 2).
