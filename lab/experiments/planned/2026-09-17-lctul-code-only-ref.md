# Experiment: the code-only arm, re-run against a frozen reference twin

Status: planned

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

## Verdict

## Updated hypothesis
