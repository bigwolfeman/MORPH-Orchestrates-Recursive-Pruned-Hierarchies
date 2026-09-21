# Planned: LXTUL-P rung P2, single-stream form: the slot loop as a DENOISER (`tul.loop_denoise`)

Status: planned

Date: 2026-09-21 (frozen before any GPU step of the arm). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
Part 2 change 1, rung P2. Partner: arm A, `tul-code-target`
([`2026-09-17-lctul-target-slot-loop.md`](2026-09-17-lctul-target-slot-loop.md), wandb
`edgwqcr1`, 20k). Arm A's front is frozen by its `train_only` list, so on arm A the
frozen reference twin (`code_target_ref`) computes the SAME reading the live front does;
arm A IS "arm A with the reference" and no second partner run is needed. Its depth-6
sweep and `tokens.npz` at 20k are at
`/home/wolfe/morph-scratch/arc/results/2026-09-17-lctul-target/` (copied from the Spark
2026-09-21 03:41). Its checkpoint is gone from every host; every reading here pairs on
the artifacts.

## Question

Fourteen strict slot-loop arms read token K3−K6 inside [−0.0002, +0.002] and one pass does
89–95 % of the work. Every target was a deterministic function of the past with a one-step
optimum. LCM's steps earn (MI rises with steps) because each step RECEIVES a noise level
and has the clean target at that level as its own target. If pass `i` of the slot loop
enters at `z_t = (1−t) z0 + t x0` with `t = (i−1)/T_s`, is charged `‖x0_hat_i − x0‖²`
independently of the other passes (teacher forcing at train), and at eval rolls out
DDIM-style from pure noise, does the frozen VAE-stage coda read the rolled-out SAMPLE
better than it reads arm A's regressed conditional mean?

## Hypothesis

The frozen coda was trained on TRUE codes noised at 3.0 and rms-renormed. Arm A hands it a
rank-17 conditional mean wrapped in a large constant, and the coda reads that WORSE than
no cell at all (zeroing arm A's cell is worth −4.608 nats at 20k). A rolled-out sample
has the truth's statistics (norm, rank ~70) rather than the mean's, so the same reader
should read it closer to how it reads a true code (1.3 nats). That is the LaDiR / LCM
claim and the reason the ladder puts P2 above P1. Against it: the rollout gap (train
teacher-forces, eval rolls out; no epsilon-scaling built), and the possibility that the
sample is a true-looking code of the WRONG span, which the reader will speak fluently and
wrongly.

## Method

`morph/configs/tul_slot_spandec_strict_denoise.yaml` over `tul_code_target` +
`code_target_ref: true`: `loop_denoise: true` (linear grid, weight 1.0),
`code_target_weight: 0.0` (the exit term is the last entry of the per-pass sum),
`core_fixed_point_lambda: 0.0` (under teacher forcing `h_T` and `h_{T−1}` are outputs of
two different inputs; the term would penalise the denoising motion itself),
`code_t_embed_scale: 16.0`, `train_only` restated with `tul_loop_denoise.` and
`tul_code_time.`. Opening as arm A: `training.init_from` the VAE stage
(`tul-code-vae/step_10000.pt`), step axis 0, the 1000-step ramp, `data_skip_batches`
10000. 20,000 steps, batch 6, seed as arm A. Build: `morph/model/tul_denoise.py`,
`_tul_core` (per-pass entry, prediction and rollout), `tests/test_tul_loop_denoise.py`
(30). Committed 2ef117c.

Readouts: wandb finals (`train/loop_denoise`, `loop_denoise_l2_t{t}`,
`val/code_target_cos`, `val/code_target_cos_shuf`, `val/ce_tokens`); the runner's
forced-depth sweep at 5k/10k/15k/20k with `tokens.npz`; `paired_vs_ruler.py` with arm
A's 20k sweep as the ruler at depth 6 (the RUNG SCORE); the worth profile (zero /
shuffle) at 20k.

**Theatre named before the run.** A denoiser's K-curve rises because pass 1 enters at
pure noise. K1−K6 is NOT this arm's score. Every per-pass loop instrument at train reads
an independently noised entry and is not comparable to any other arm (the config lists
them). At eval the rollout is a sample, so `val/code_target_cos` is a sample's cosine to
the truth, not a mean's, and is expected LOWER than arm A's 0.147 even when the sample is
good; read it beside the reader.

**Method amendment 2026-09-21 12:50 (before any training step; predictions untouched).**
The runner's 12-step smoke of this config FAILED at 12:34 (`queue.log`: "smoke FAILED:
draw not started") inside `warmup_compile_all_shapes`: the trainer's compile warmup runs
BEFORE `tul_code_ref_snapshot`, and the denoiser's pre-loop target call raised with no
twin and no live states. The write path has always fallen back to E on the live prelude
in that window (every `code_target_ref` arm's smoke took it). Fix: `_tul_code_target_encode_pre`
takes the same fallback with the exact `xn` the loop forms, at both pre-loop call sites;
test `test_a_forward_before_the_twin_is_snapshotted_runs_on_the_live_front` (31 pass in
the file, 35 in the ref + target files). The arm is re-queued at the fix commit behind
np0; the runner's smoke is the GPU check. No prediction is changed by this: the trained
forward never runs without the twin (the snapshot precedes step 0).

## Predictions (frozen)

Arm A at 20k for reference: depth-6 val CE 9.1099 on 480 rows; K1−K6 +0.0320 [+0.0235,
+0.0412]; K1−K16 +0.0959; `val/code_target_cos` 0.147; zero-cell worth −4.608 (the
no-cell floor is therefore about 4.50 nats); the true code reads 1.3 nats; rate 14,393
tok/s. Seed spread 0.024 nats. Probabilities are mine.

- **P-1 (healthy).** 20,000 steps, tripwire HEALTHY (`preclip/total` under 1e4 at every
  step ≥ 200), no divergence-guard abort (ceiling 1e8 ppl as arm A). **80 %.** The
  entries are unit-RMS mixtures of noise and a unit-RMS code; the hinge is on and was
  tuned on prelude-shaped entries. If this fails, the pre-registered fallback is the same
  config with `model.slot_gain_lambda: 0`, and the tripwire alone holds.
- **P-2 (the denoiser learns).** `train/loop_denoise` at 20k below **0.70×** its value
  averaged over steps 200–400. **85 %.** The sanity clause: a per-pass x0 loss that does
  not fall means the level or the entry did not reach the passes.
- **P-3 (THE RUNG'S REASON: the reader).** Paired depth-6 reader CE on the shared rows,
  denoise − arm A, at or below **−0.50** nats. **40 %.** The bar is deliberately large:
  arm A sits 4.6 nats above its own no-cell floor, so a sample that the coda merely
  tolerates clears 0.5 easily, and a 0.024-nat win would be inside the reader's own
  noise about a broken read.
- **P-4 (worth having).** Depth-6 val CE at 20k below the no-cell floor, **4.50**
  nats: the worth profile's `zero` total is NEGATIVE (zeroing the cell HURTS). **25 %.**
  This is the bar arm A failed. It is the reading that would say the loop's sample is
  worth more to the frozen coda than nothing.
- **P-5 (a sample, not a mean).** `val/code_target_cos` at 20k inside **[0.06, 0.20]**
  and the corpus-mean probe's prediction rank (`code_target_mean_*`, effective rank of
  the exit cells over 96 rows at depth 6) at or above **35** (arm A 17.0 / 18.1, truth
  77.5 / 70.2). **50 %.** A rollout that collapsed to the conditional mean reads rank
  ~17; one that samples reads higher.
- **P-6 (the K-curve rises, trivially).** Forced-depth K1−K6 at 20k at or above
  **+0.20** nats. **85 %.** Depth 1 is one pass from pure noise. This is an instrument
  reading, not a score; it is listed so no one quotes it as depth earning.
- **P-7 (rate).** Train rate at or above **0.80×** arm A's 14,393 tok/s. **75 %.** One
  extra projection and l2 per realised pass; the target is encoded once.

## Binding

If **P-3 and P-4 hold**: the sample beats the mean and is worth having. Next: (a) the
SONAR target on the denoiser (P3 on P2: `code_target_source: sonar` needs the code-target
seam, not the flow thinker's, and that seam is not built; it is a build), and (b) the
fan on the denoiser, which needs the register's cells and the code target's cells to be
one object (named NOT composable in the config; a build with its own note).

If **P-3 holds and P-4 fails**: better than arm A, still worse than nothing. The reader
is still the confound (`the-reader-was-the-limit`): the next run is a coda re-initialised
and trained alone on the frozen denoiser's rollouts, Wolfe's 2026-09-17 question, never
run.

If **P-3 fails** (the sample reads no better than the mean): rung P2 does not beat its
rung below, which is the note's falsifier for the ladder on the single-stream base. Do
NOT rescue with a schedule sweep. Record it in the note and go to Wolfe's fallback: the
np0 arm ([`2026-09-21-strict-np0.md`](2026-09-21-strict-np0.md)).

If **P-2 fails**: a build fault (level or entry not reaching the passes); fix, re-run
tests, re-queue with a fresh prereg. Nothing about the idea is read.

## Not verified before launch

- No GPU step of this arm; the runner's smoke is the first. The smoke runs the bare
  config WITHOUT `init_from`, so it exercises the build and not the opening.
- Whether the runner's forced-depth sweep is a faithful rollout at each forced depth
  (the level grid is per slot over the REALISED depth, so a forced depth T gives a
  T-step rollout by construction; asserted from the code, not measured).
- The pairing intersects arm A's 2026-09-17 sweep rows with the runner's 2026-09-21
  rows on `tok_index`; the overlap count is unmeasured. If it is under 300 of 480 rows,
  the pairing is void and the sweep must be re-run on arm A's rows.
- `torch.compile` on the new branch: the tests are eager.
- The exposure-bias gap (train teacher-forced vs eval rollout) has no instrument here
  beyond `val/code_target_cos`.
