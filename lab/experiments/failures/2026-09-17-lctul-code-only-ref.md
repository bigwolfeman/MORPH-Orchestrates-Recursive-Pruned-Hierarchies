# Experiment: the code-only arm, re-run against a frozen reference twin

Status: failure

Date: 2026-09-17 (frozen before any GPU step beyond a 12-step smoke on the Spark).
Replaces `2026-09-17-lctul-code-only.md`, whose two draws are void: with `train_only`
cleared the frozen encoder read the LIVE prelude, so its codes were not a fixed target and
collapsed to one direction (own cosine 0.99, shuffled 0.98 by step 2500). The design changed
(`tul.code_target_ref`), so this is a new planned file and not an amendment.
Note: `.agents/notes/proposed/architecture/2026-09-17-lctul-target-slot-loop.md` (item 5);
`docs/tul-code-spec.md` §17.1. Tests: `tests/test_tul_code_ref.py` (9),
`tests/test_tul_code_target.py` (26), `tests/test_tul_code_grade.py` (16), CPU.

## Question

Arm A froze the prelude, the embeddings, E and the coda, and read a cell that is a rank-17
conditional mean of the code, found in one pass by step 1500 and unchanged to 20k (own
cosine 0.154, shuffled 0.020, per-pass l6 − l1 +0.003, depth 16 minus depth 1 +0.002).
Its prelude came from the VAE stage, where it learned to serve a coda reading noised TRUE
codes, never to represent the past for prediction. With the prelude and embeddings training
under the loop's gradient, and the target computed by a FROZEN twin of the VAE stage so it
cannot drift, does the loop predict more of the next span's code, and does any of the extra
prediction arrive after pass 1?

## Hypothesis

H1: the frozen prelude was the bound. Own minus shuffled and the cells' effective rank both
rise well past arm A's.
H2 (depth): a representation the loop itself shapes gives the later passes something to do,
so the per-pass cosine curve stops being flat.
H0: the predictable part of the next span's code is small whatever the front, so the loop
lands near arm A's numbers with a different prelude, and the passes still finish in one.

## Predictions (frozen)

Arm `tul-code-only` (`tul_code_only.yaml`, now `code_target_ref: true`): `init_from`
`tul-code-vae/step_10000.pt`, 20k steps, batch 6, seq 1024, one seed, `train_only: []`
(E and the twin frozen; the coda gets no gradient because no coda runs at train),
`data_skip_batches` 10000. Baseline for every delta is arm A at 20k: own 0.1544 / 0.1550,
shuffled 0.0196 / 0.0454, rank 17.0 / 18.1, `code_target_cos` 0.1413 over the last 500
logged steps, l6 − l1 +0.0037, rate 14,393 tok/s.

- P-R1 (the collapse is gone): `train/code_target_cos_shuf` stays below 0.10 on the last 500
  logged steps at 20k (the void draw read 0.98 by step 2500). 85 %.
- P-R2 (the prelude was a bound): `train/code_target_cos` ≥ 0.25 over the last 500 logged
  steps at 20k. 40 %.
- P-R3 (specific, not generic): corpus-mean probe at 20k, own minus shuffled ≥ 0.18 on both
  cells (arm A: 0.135 / 0.110). 40 %.
- P-R4 (rank): effective rank of the predicted cells ≥ 25 on both cells at 20k (arm A: 17.0 /
  18.1). 35 %.
- P-R5 (the passes move): `tul/code_target_cos_l6` − `tul/code_target_cos_l1` ≥ 0.02 over the
  last 500 logged steps at 20k (arm A: +0.0037). 20 %.
- P-R6 (depth in the cell, not in the reader): depth-resolved corpus-mean probe at 20k, own
  cosine at depth 16 minus depth 1 ≥ 0.02 on both cells (arm A: +0.0023 / +0.0016). 20 %.
- P-R7 (rate): ≥ 1.1× arm A's tok/s — no coda runs at train, one extra frozen prelude pass
  does. 75 %.

## Binding

- P-R2 and P-R3 hold → the frozen prelude was the bound; the reader question (a coda trained
  on the loop's cells from step 0) becomes the next arm, on this front.
- P-R2 fails with P-R1 holding → the predictable part of the code is ~0.15 cosine whatever
  the front: the L2 target is exhausted and only a computed target (the graded arm,
  `2026-09-17-lctul-graded-target.md`) can change the question.
- P-R5 or P-R6 holds → the first per-pass signal in this record; the progressive arm re-runs
  on this front.
- P-R1 fails → the twin does not hold the target still either, and the whole
  trainable-front branch of the family goes to rejected with that reading.

## Method

`tul_code_only.yaml` composes `tul_code_target` and sets `code_target_skip_coda: true`,
`code_target_ref: true`, `train_only: []`. Queue line at the head of `recon_arms.txt` (kind
slot, commit recorded at insertion, EXTRA with `training.init_from` and the panel extras,
exactly arm A's line). Spark smoke of the config with `init_from` before insertion. Probes
at 10k and 20k on the Spark: the semantic probe (`--kind target`, on the code fixed at
14f004d, where the open slot finally carries a cell), the corpus-mean probe with
`--depths 1,6,16`, and the runner's forced-depth sweep (fixed at 2236ac4).

Amended 2026-09-17 20:45 (reason: the smoke ran and the line was inserted): Spark smoke of
`tul_code_only` at 92954ca with `init_from`, 12 steps: 0 Tracebacks, 0 nan, 467/469 tensors
matched, `[code_ref] frozen VAE-stage twin snapshot: 269.9M parameters, 0 trainable, eval
mode`, peak 7.17 GB, final val_loss 8.4375, train `code_target_cos` 0.0047 with
`_cos_shuf` 0.0053. Queue line inserted at the HEAD at 20:45 (commit 92954ca), ahead of the
graded arm and the continuations. Predictions unchanged.

## Not verified before launch

- The 51 CPU contracts of this family; the GPU smoke's exit code goes here before launch.
- The twin's memory cost on the 5090: 1.32 GB measured on the Spark, and the 5090's TUL
  arms have about 1 GB of slack at seq 4096 but run this panel at seq 1024 and 7.9 GB.
- One seed. No coda trains, so nothing here reads generation quality except through a coda
  frozen at the VAE stage. Amended 2026-09-18 00:10 (reason: the "7.4 nats" floor was
  WITHDRAWN at 8ea2a76 as never measured; this file predates that commit). The measured
  reading is arm A's worth profile at 20k: zeroing the cell moves the frozen coda's paired
  per-token CE by -4.608 nats, so that coda would rather have NO cell than the predicted
  one. Predictions unchanged.

## Results

Arm `tul-code-only`, wandb `wy38mb6y`, 20,000 steps, commit 92954ca, finished 2026-09-18
03:30. One seed. Readings are the mean of the 25 logged points at step 19,500 and beyond,
and the corpus-mean probe at 20k on the Spark (96 rows, depths 1/6/16).

### The twin held the target

The whole reason this file replaced its predecessor. `train/code_target_cos_shuf` ends at
**0.0380**, and it FELL during training (0.0571 at step 200, 0.0194 at 1000, 0.0238 at
3000). The void draw read 0.98 by step 2600 with the own cosine tracking it. The frozen
reference copy does what it was built to do.

### The frozen prelude was not the bound

| reading | arm A @20k | this arm @20k |
| --- | --- | --- |
| `train/code_target_cos` | 0.1413 | **0.1674** |
| `train/code_target_cos_shuf` | 0.0196 | 0.0380 |
| corpus-mean own − shuffled, cell 0 / cell 1 | 0.135 / 0.110 | 0.138 / 0.110 |
| effective rank of the prediction | 17.0 / 18.1 | 17.4 / 17.6 |
| `l6 − l1` | +0.0037 | +0.0029 |
| depth 16 − depth 1 | +0.0023 / +0.0016 | +0.0005 / +0.0003 |
| tok/s | 14,393 | 16,029 |

Training the prelude, the embeddings and the whole front for 20,000 steps moved the own
cosine by **+0.026** against a prediction of 0.25, and left rank, span-specificity and the
per-pass curve where arm A had them.

### Scoring

| | prediction | threshold | measured | |
| --- | --- | --- | --- | --- |
| P-R1 | the collapse is gone | shuf < 0.10 | 0.0380 | **HOLDS** |
| P-R2 | the prelude was a bound | cos ≥ 0.25 | 0.1674 | fails |
| P-R3 | specific, not generic | own − shuf ≥ 0.18 both | 0.1382 / 0.1100 | fails |
| P-R4 | rank | ≥ 25 both | 17.4 / 17.6 | fails |
| P-R5 | the passes move | l6 − l1 ≥ 0.02 | +0.0029 | fails |
| P-R6 | depth in the cell | d16 − d1 ≥ 0.02 both | +0.00045 / +0.00029 | fails |
| P-R7 | rate | ≥ 1.1x arm A | 1.114x | **HOLDS** |

Two of seven hold and both are machinery: the mechanism works and the arm is faster
without a coda. Every prediction about the IDEA failed.

### What the cell actually is

The corpus-mean probe at 20k, depth 6, is the most informative reading in the file. The
true codes have effective rank 77.5 and 70.2; the predictions have 17.4 and 17.6. The mean
prediction is nearly parallel to the mean code (`cos_pbar_zbar` 0.90 / 0.97) and a third to
a half of each prediction's norm is that constant (`norm_pbar_over_rms` 0.36 / 0.55). The
predictions correlate with EACH OTHER at 0.13 / 0.31 while the true codes correlate at
about zero.

But the signal is real. Remove the corpus mean and own cosine is **0.1495 / 0.1335**
against a shuffled floor of **0.0027 / 0.0013**. The cell is a genuine, span-specific,
~17-dimensional shadow of a ~73-dimensional code, wrapped in a large constant.

### Four instruments were void on this arm, and why

No coda runs at train, so the coda's weights never move, but the prelude trains under a
loss with no token term. Measured against the frozen twin inside this arm's own
checkpoint: the prelude drifted **0.400** of its own norm and the core **0.279**, while
`coda.` and `tul_code_enc.` moved EXACTLY 0.000000 (0 of 96 and 0 of 8 tensors). The reader
is fixed; its world moved.

`val/ce_tokens` climbed 9.36 → 11.09 → 12.64 and the arm's own oracle ceiling reached 8.3
nats against arm A's 1.41. `ln(49152) = 10.80`, so at 12.64 the coda is 1.84 nats WORSE
than uniform and is not a reader at all. Consequences:

- the forced-depth sweep read K1−K6 of +0.0698, −0.0014, +0.0414, +0.0291 at
  5k/10k/15k/20k, every interval clear of zero and the sign flipping between checkpoints;
- the worth profile read a shuffle cost of +0.2972 against arm A's +0.050;
- the semantic probe's ORACLE fell to 0.1694 against arm A's 0.5645, so the ceiling
  collapsed (OWN−SHUF still survived at +0.0196 [+0.0023, +0.0372], but on a probe with
  a third of its range);
- `tul.code_target_ref` was never rebuilt by `scripts/tul_samples.py::load_ckpt`, so every
  offline probe silently scored against the LIVE front. Fixed at 9263204. Before the fix
  this arm's codes read cosine 0.914 to their own corpus mean with rank 11; after, 0.057
  and rank 77, agreeing with the trainer.

The replacement is `lab/divergence/code_twin_read_probe.py` (5f0bd68): the cell is the only
thing taken from the live model and the frozen twin reads it. Cross-validated against
`worth_profile.py` on the same 96 rows — `zero` matched to four decimals, −4.6413 both.

| twin-read, 96 rows, 8 permutation draws | arm A @20k | this arm @20k | graded-l2 @20k |
| --- | --- | --- | --- |
| oracle, absolute | 1.2349 | 1.2349 | 1.2349 |
| zero, absolute | 4.4517 | 4.4517 | 4.4517 |
| own, absolute | 9.0930 | 9.0755 | 8.9747 |
| zero − own | −4.6413 | −4.6238 | −4.5230 |
| shuf − own | +0.1314 | +0.1876 | +0.0919 |

The `zero` and `oracle` absolutes are identical across three arms to four decimals, because
neither condition involves the live model and all three readers are the same VAE stage.

On this scale the arm is **0.056 nats MORE span-specific than arm A** (+0.1876 against
+0.1314, 8 draws each, shared rows, spreads 0.042 and 0.049), and it rose from +0.1055 at
10k, so it was still improving while the training cosine looked flat. The corpus-mean probe
disagrees and calls it a tie (0.138 / 0.110 against 0.135 / 0.110). Both instruments agree
the arm improved between 10k and 20k; they differ on the comparison with arm A, and neither
difference is large. Not resolved here.

What does NOT differ: `zero − own` is −4.64, −4.62, −4.52 across the three arms. A
competent reader would rather have NO cell than any of them, by about four and a half nats,
whatever the front and whatever the target.

## Verdict

**Failure.** Five of seven predictions missed, including every prediction about the idea.
H1 is refuted: the frozen prelude was not the bound. H0 stands — the predictable part of
the next span's code is a cosine of about 0.15 whatever the front is, found in one pass.

The binding clause written before the run applies as written: "P-R2 fails with P-R1 holding
→ the predictable part of the code is ~0.15 cosine whatever the front: the L2 target is
exhausted and only a computed target can change the question." The computed target then ran
(`2026-09-17-lctul-graded-target.md`, `tul-code-grade-l2`) and did not change it: its loop
predicts the WINNER's code at cosine 0.532 and the TRUTH's at 0.146, while the winner's code
is itself only 0.153 from the truth's. That arm's grader is sound (true rank 0.878), so this
is not a broken grader; the target is largely self-referential.

## Updated hypothesis

The loop can find a ~17-dimensional, genuinely span-specific projection of the next span's
~73-dimensional code, in one pass, and no amount of extra passes, extra freedom in the
front, or a computed target in place of the regression moves that. The cell is not empty and
it is not generic. It is a sharp shadow of the wrong size, and a fixed competent reader
would rather have nothing than have it — which is the reading that should drive what comes
next, because it says the failure is not "the loop learned too little" but "what the loop
learned is not what the reader needs".

Two questions this file cannot answer, in order:

1. **Can a reader trained ON these cells use them?** Every reading here uses a reader that
   never saw the loop's output during training. `tul-code-target-uf` (running 2026-09-18
   from arm A's 20k with the coda unfrozen and the prelude confirmed frozen, so the target
   cannot drift and CE stays valid) is that experiment. Its loss fell from 11.37 at resume
   to 5.59 by step 25,600, so the reader adapts fast; whether it adapts TO the cell or
   AROUND it is the thing to measure.
2. **Is 17 of 73 dimensions the ceiling of what the past determines?** Nothing here
   separates "the loop cannot find more" from "there is no more to find". The oracle gap
   (−7.84 nats) is the size of the prize, and no arm in this family has closed any of it.
