# Agent Note: LXTUL-R Step 2, speed and contribution, the task list

Status: proposed

Wolfe, 2026-09-22, on the Step 1b readout: "This shouldn't be a failure it is a mixed
result. We need to solve the problems on this. Speed. The current step 1b is very slow.
Loop contribution. Remember 1/2 of mean k should give around 90% loop contribution.
80-94%. The contribution needs to be closer to 0.1." And on the plan below: "Agreed. Save
this to a task list, build and test with subagents, then run it end to end."

Parent: [`2026-09-21-lxtul-r-reach-composition.md`](2026-09-21-lxtul-r-reach-composition.md)
(Step 0, Step 1, Step 1b outcomes in its Outcome log). Filing that opens this note:
[`failures/2026-09-22-lxtul-r-step1b.md`](../../../../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md).

## Problem

Two, measured on `slot-spandec-strict-fan4-all-reach1` at 5k (commit 9b430d3, run u8xh91vd):

1. **Speed.** 1.03 steps/s (971 ms/step; fwd 494, bwd 416 wall) against fan4-all's 1.15 and
   the plain model's 1.98. The profile (`results/2026-09-22-lxtul-r-step1b/profile_step_c0f6bc4.txt`,
   kineto over steps 40 to 43) says the GPU is busy 86 % of the step at 8.8 TFLOPS: the
   arm is latency bound on small tensors (B=6, 256 cells), not FLOP bound. The buckets:
   `aten::mm` 28 % (about 2,400 calls per step at 96 us); copies and dtype casts 14 %; the
   EAGER segmented CCA conv path (cudnn conv fwd+bwd, grouped dgrad, layout transposes)
   about 175 ms/step, 18 %, which the fused prologue does in one kernel on the strict
   control; one unaligned gemm (`128x256_tn_align1`, V = 49169 is odd) at 1.18 ms x 36
   calls per step = the per-row `[L, V]` logits of the fan write-all winner replay; and
   host syncs (`cudaStreamSynchronize` 896 ms of CPU over 3 steps) from `float()` on
   fan stats inside the forward.
2. **Contribution.** Token K1-K6 +0.026 (K3-K6 +0.0057: at depth 3 the arm has 78 % of its
   depth-6 gain, so the SHAPE already meets Wolfe's half-depth rule; the SIZE does not).
   The far budget that a one-slot-per-pass relay can earn is [0.126, 0.197]; the arm
   recovers a fifth of its lower bound and sits 0.027 behind fan4-all-fp0 at depth 6
   (0.053 at depth 1). Carried content decays inside the cell after arrival (planted g=2
   kept 62 %, an instrument with a caveat), and the relay and the diversity term fight
   inside the same four cells.

## Proposal

The task list. A box is ticked in this note when the item is merged (build) or filed (run).

SPEED (objective-preserving; the loss is the same function)
- [x] S1 (cad4515) batch the K no-grad stream-scoring coda passes of `_tul_fan_all` into one
      `[K*B, L, C]` call (same FLOPs, a quarter of the launches, 4x gemm M).
- [x] S2 (cad4515) 8-aligned vocab for every raw `[L, V]` logits matmul on the fan/span-CE path
      (zero-row pad of the tied head, sliced before the CE).
- [x] S3 (cad4515) no host syncs in the training forward: fan stats become detached 0-dim tensors,
      `float()` in the trainer on logging steps.
- [x] S4 (merged 11:25; S5 reading 2: 227 ms/step off, the cuDNN conv path 468 -> 43 ms per 3 profiled steps) fused Triton segmented two-stage causal conv (fwd + bwd) for `segment_causal_conv`,
      the eager path every reach arm pays at every core layer of every pass.
- [x] S5 (reading 2 at d1c42f0, 15:05: 767 ms/step, 1.30 steps/s, bar MET at its edge) measure: a 48-step profile run of the same config at the merged commit; report
      steps/s and the region split beside the 971 ms baseline. Bar: >= 1.30 steps/s on the
      Step 1b config (fan4-all's 1.15 plus the reach cost removed). The compile/latency
      floor of the 256-cell core (compile_blocks bench 1.84x) is the next lever if S1-S4
      fall short; it is NOT in this list.

CONTRIBUTION (each a one-factor arm over Step 1b, 5k steps, then the winner at 20k)
- [x] C1 (merged 11:10, queued behind C2) `tul.loop_carry: persist` (Step 2 of the parent note): the accumulated layer-0
      cross-cell read is added ONCE at the loop exit, RMS-matched to the exit cell, and is
      never re-processed by the core (the map is bit-identical to `none`; the rejected
      `sum`/`gate` re-injected at every entry). Config
      `tul_slot_spandec_strict_fan4_all_reach1_persist.yaml`.
- [x] C2 (5b09669, queued 10:55) `tul.fan_history_streams: 1`: stream 0 is the history channel (reads stream 0 of
      slot k-1 at core layer 0, exempt from the epivol repulsion), streams 1..3 are plan
      streams (slot-local at every layer, they read history through their own slot's
      history cell, repelled among themselves). Config
      `tul_slot_spandec_strict_fan4_all_reach1_hist1.yaml`.
- [ ] C3 CLOSED (2026-09-22 15:00, not queued): neither C1 nor C2 moved the K-curve or the depth-6 CE (Wolfe's call: no C3 on this geometry unless C1 clears fp0 at depth 6; C1 reads +0.025 behind).
- [ ] C4 CLOSED (2026-09-22 15:00, not queued): no 20k run on the chain geometry (Wolfe's call, same reason).
- [x] I1 (84ad3b2) the span-swap instrument (`hop_distance_probe.py --swap`): span k-g's tokens
      replaced by natural text from another row, benefit = CE(target span | swapped) -
      CE(target span | original) at forced depths; replaces the out-of-context planted
      probe (control 15.5 nats, 4.7 above uniform) as the distance instrument.

Scorecard per arm (every number paired on identical rows): token K1-K6 and K3-K6
(shape: K3-K6 / K1-K6 in [0.06, 0.20] means 80-94 % by half depth); paired depth-6 and
depth-1 CE against fp0 (the depth-dependence guard; a K-curve bought by a depth-1 loss is
not a win); the swap table (benefit per g and depth, kept fraction); rate; entry norm and
tripwire.

Order: S1-S4 built in parallel by subagents, merged one patch at a time with the test
gates, S5 measured; C1, C2 and I1 built in parallel; then the runner queue C1, C2 (and C3)
at 5k with the S-merged code; readouts; C4.

S5 reading 1 (2026-09-22, 10:50, `results/2026-09-22-lxtul-r-step1b/profile_step_s123_84ad3b2.txt`):
S1–S3 merged leave the step at 994 ms against 971 (inside run-to-run noise). The unaligned
gemm is gone (about 20 ms/step), copies rose by about the same, and the step is bound by
about 830 ms of GPU kernel time spread over tens of thousands of small kernels. Launches
and host syncs were not what the GPU waited on. The remaining speed levers are S4 (the
conv path's launches) and the compile/latency floor of the 256-cell core; S5's bar
stands and is now expected to fail without the second.

S5 reading 2 (2026-09-22, 15:05, `results/2026-09-22-lxtul-r-step1b/profile_step_s4_d1c42f0.txt`):
S4 merged takes the step from 994 to 767 ms (1.30 steps/s, 7,707 tok/s against 6,129;
forward 510 → 413, backward 454 → 325 ms GPU). The cuDNN conv kernels and their layout
copies leave the table (468 → 43 ms per 3 profiled steps for the conv path; `aten::copy_`
397 → 305 ms); `aten::mm` is unchanged at 614 ms. Peak memory 17.7 → 17.0 GB. The S5 bar
is met at its edge. What remains is the same shape as reading 1: about 26k copy launches
and 7.5k small matmuls per 3 steps on 256-cell tensors; the compile/latency floor of the
core is the next lever and is outside this list. Not verified: the arms C1 and C2 ran
without S4 (queued before its merge); a training-run reading of the fused conv under the
reach geometry is the next arm's smoke.

**Wolfe's calls, 2026-09-22 (13:40 to 14:00), during the C1 run.** (1) The chain geometry
is a sliding window with a stride of one span per pass; it is INTENTIONAL as a measurement
of whether forced passes earn, and it is NOT a final design. Every arm on it sits behind
fp0 at every depth (the A1.8 guard reads negative on Step 1b and C2), and reaching g spans
back costs g full passes where one attention hop is free: relay is the wrong job for a
pass. C1 is read for the decay question; no C3 and no 20k run on this geometry unless C1
clears fp0 at depth 6. The open problem stays: keep direct access and find a pass job
that is not relay. (2) The span-swap instrument (I1) is a CORRUPTION: it inserts wrong
text, so its benefit overstates content use and its "decay" may be the later passes
discounting a mismatched span. Language is causal. Its exact zeros and arrival staircase
stand as the leak check; its kept fractions are not read as loop decay, and the sum of its
far rows is not compared with Step 0's far budget (a different instrument). The proposed
replacement is information REMOVAL: replace slot k-g's seed (the span summary entering the
loop) with the batch-mean seed, tokens untouched; not built until Wolfe decides.

## Alternatives considered

- **Widen the reach (2 slots per pass).** The leaky Step 1 arm effectively relayed four
  slots per pass and earned LESS (+0.020) than one slot per pass (+0.026): the loop's gain
  comes from the passes having a job, not from more content per pass. Not queued.
- **Winner replay on a fraction of rows or steps.** Changes the objective (which slots
  score their streams); S1 gets the same launches saving with the loss unchanged. Not
  queued unless S1-S4 miss the bar.
- **A TG-aware fused attention kernel.** The eager attention path costs less than the
  eager conv on the profile (the fused mem-efficient attention kernels are 3 % of CUDA
  time); the conv is the eager cost. Deferred behind S4's reading.
- **Longer horizon first.** The 5k to 20k table in the parent note says the horizon does
  not move a slot loop's K-curve on its own; C4 runs after C1/C2 pick the mechanism.
- **The two-channel design outside the fan (history through the coda's token reach).**
  C2 is the same idea inside the fan and keeps depth = reach by construction; the
  outside version lets the coda read history at depth 0 and gives the loop no job.

## Acceptance criteria

- Builds: every S and C item lands with tests that fail when the behaviour is broken;
  default-off keys are bit-identical (loss, logits, parameter names); geometry claims by
  perturbation at forced depth (exact zeros beyond the reach); the S items give the same
  loss up to bf16 noise (a numeric test each).
- S5: >= 1.30 steps/s on `tul_slot_spandec_strict_fan4_all_reach1` at the merged commit.
- C1/C2 preregs frozen and committed before launch (`lab/experiments/planned/`), with the
  K1-K6, K3-K6 shape, fp0-paired and swap bars stated there.

## Risks

- S1 raises the no-grad coda's peak by K x one block's activations; the 5090 has ~1 GB
  of slack on the fan arms. Measured at S5 before any queue.
- S4 is a new kernel on the hot path; the reference stays as the oracle and the GPU test
  (fwd, grads, exact-zero cross-segment) gates the merge.
- C1 gives every pass's read a direct gradient path from the coda; if the passes learn to
  write only for the exit add, the K-curve could rise while the raw trajectory flattens.
  The raw `db_traj` rank and the swap table at forced depths read that.
- C2 with one history stream makes that stream the only relay; if the epivol term on the
  three plan streams collapses without the fourth, `fan/plan_rank` reads it.

## Outcome (2026-09-22 15:05)

C1 and C2 filed together under
[`failures/2026-09-22-lxtul-r-step2-panel.md`](../../../../lab/experiments/failures/2026-09-22-lxtul-r-step2-panel.md).
Both fail the panel's headline bars: C2 hist1 token K1−K6 +0.0204 [+0.0194, +0.0214], C1
persist +0.0234 [+0.0222, +0.0246] (bar +0.035; Step 1b +0.0261); K3−K6 ratios 0.225 and
0.222 (bar 0.06 to 0.20); paired depth-6 CE against fp0 +0.0255 and +0.0248 (bar ≤ +0.015);
against Step 1b −0.0010 and −0.0018, both inside their intervals. P-1 holds on both with
no late spikes (the first spike-free reach runs); C1's persist term is live by construction
(`carry/persist_ratio` 0.98, the accumulator's RMS at pass 5 growing 0.54 → 2.63) and C2's
three plan streams sit at rank 1.99 of 3. C1 costs 0.6 % more rate than the bar allows
(6,106 against 6,335 tok/s; the second layer-0 attention call). Reading: Step 1b, C1 and
C2 have the same depth-6 CE within 0.002; persistence and stream separation do not change
what the coda gets from the loop; the relay itself is the limit. Per Wolfe's calls above,
the chain geometry is closed as a measurement: C3 and C4 are not queued. The speed items
stand on their own (S5 reading 2 below). Next design direction, from the JEPA-Anything
summary of the same day: keep fp0's direct access and give each pass one orthogonal
factor of the slot target under an EMA target encoder with per-coordinate variance floors
(the same recipe that addresses the LCTUL code-target collapse); to be written as its own
proposed note before any prereg.
