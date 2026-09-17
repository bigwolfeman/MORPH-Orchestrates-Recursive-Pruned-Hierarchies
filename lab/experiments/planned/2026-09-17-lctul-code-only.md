# Experiment: the code-ONLY arms — no coda at train, everything trains, L2 vs InfoNCE

Status: planned

Date: 2026-09-17 (frozen before any GPU step beyond a 12-step smoke on the Spark).
Note: `.agents/notes/proposed/architecture/2026-09-17-lctul-target-slot-loop.md` (item 5).
Spec: `docs/tul-code-spec.md` §17.1. Tests: `tests/test_tul_code_target.py` (24, CPU).
Parent: `2026-09-17-lctul-target-slot-loop.md` (arm A, the reading this amends).

## Question

Arm A (frozen prelude, L2 to the frozen VAE code, frozen coda reading the cell) found a
rank-16 conditional mean of the code in one pass by step 1500 and stopped (exit cosine
0.144 at 1500, 0.169 at 10k; l1 0.135, l6 0.132; own 0.145 vs shuffled 0.021; cos to the
corpus mean 0.33-0.53 vs the codes' 0.06-0.09). Two things bounded it: the prelude never
trained to represent the past for prediction, and L2's optimum is a mean that hedges.
When the prelude trains with the loop and no coda runs at train, does the loop predict
MORE of the code (own minus shuffled), and does a discriminative term (InfoNCE) change
what it predicts (rank, own minus shuffled) or whether later passes add anything?

## Hypothesis

H1: the frozen prelude was the bound. With the prelude training, the exit cosine and own
minus shuffled rise well past arm A's; the predicted cells' effective rank rises past 16.
H2: the term shapes the cell. InfoNCE gives a higher own-minus-shuffled and a higher rank
than L2 at a similar or lower own cosine (it stops hedging toward the mean).
H0 (depth): both arms still meet their target in one pass (`per-pass-targets-met-in-one-step`).

## Predictions (frozen)

Arms `tul-code-only` (L2, `tul_code_only.yaml`) and `tul-code-only-nce` (InfoNCE tau 0.1,
`tul_code_only_nce.yaml`): `init_from` `tul-code-vae/step_10000.pt`, 20k steps, batch 6,
seq 1024, one seed each, `train_only` empty (E frozen at build; the coda and heads get no
gradient because no coda runs), `data_skip_batches` 10000, the same slot-loop constraint as
arm A. Readings from wandb (`train/code_target_cos`, `train/code_target_cos_shuf`,
`tul/code_target_cos_l{t}`, `train/code_target_acc`), the corpus-mean probe
(`code_target_mean_probe.py`) and the semantic probe (`--kind target`) at 10k and 20k on
the Spark. The baseline for every delta is arm A's 10k probe (own 0.145 / 0.149, shuffled
0.021 / 0.048, rank 15.5 / 16.6) and its 20k wandb series once it lands.

- P-C1 (the prelude was a bound): `tul-code-only` `train/code_target_cos` ≥ 0.25 averaged
  over the last 500 logged steps at 20k (arm A: 0.136 over 9500-10100). 45 %.
- P-C2 (specific, not generic): `tul-code-only` corpus-mean probe at 20k, own minus
  shuffled ≥ 0.18 on both cells (arm A 0.124 / 0.101). 45 %.
- P-C3 (rank): `tul-code-only` effective rank of the predicted cells at 20k ≥ 25 on both
  cells (arm A 15.5 / 16.6). 40 %.
- P-C4 (InfoNCE learns to discriminate): `tul-code-only-nce` `train/code_target_acc` ≥ 0.05
  averaged over the last 500 logged steps at 20k (chance ≈ 1/380 ≈ 0.003). 55 %.
- P-C5 (InfoNCE stops hedging): at 20k, the nce arm's own-minus-shuffled exceeds the L2
  arm's by ≥ 0.03 on both cells, and its effective rank exceeds the L2 arm's by ≥ 5 on
  both cells. 40 %.
- P-C6 (the passes move): on EITHER arm, `tul/code_target_cos_l6` − `tul/code_target_cos_l1`
  ≥ 0.02 averaged over the last 500 logged steps at 20k (arm A: −0.002). 20 %.
- P-C7 (rate): both arms ≥ 1.25× arm A's tok/s (14,340) at batch 6: no coda at train. 70 %.
- P-C8 (the frozen coda is not the instrument): `val/loss` (the VAE coda's CE on the
  predicted cell) stays above 7.4 nats on both arms at 20k. 65 %. Reported, not a verdict:
  the prelude the coda was trained beside has moved.

## Binding

- P-C1 and P-C2 hold → the frozen prelude was the bound; the next arm trains a coda on the
  loop's cells from step 0 (the reader question), on the better of the two terms.
- P-C1 fails and P-C4 holds → the loop can discriminate but not regress: the L2 target is
  the wrong shape for a Poisson loop and the family moves to the InfoNCE term.
- P-C1 and P-C4 fail → the past bounds the code at ~0.15 cosine regardless of the prelude
  and the term: the predictable part of the next span's VAE code is small, and the family
  goes to rejected with that number.
- P-C6 holds on either arm → the first per-pass signal in the record; the progressive arm
  re-runs on that term. P-C6 fails on both → one pass again; the depth question moves off
  predicted targets entirely (a target that is computed, not predicted).

## Method

Both configs compose `tul_code_target` (`tul_code_only` adds `code_target_skip_coda: true`,
`train_only: []`; `_nce` adds `code_target_loss: infonce`, `code_target_tau: 0.1`). Spark
smoke of each with `init_from`, 12 steps, before the queue lines are written. Queue lines
at the HEAD of `recon_arms.txt` (kind slot, commit recorded at insertion, EXTRA with
`training.init_from` and the panel extras exactly as arm A's line), ahead of arm B and the
continuations. A watcher instance runs the semantic probe and the corpus-mean probe at 10k
and 20k. Pairing against arm A with `code_semantic_pair.py`.

Amended 2026-09-17 16:58 (reason: the smokes ran and the lines were inserted): Spark smokes of
`tul_code_only` and `tul_code_only_nce` at 20a4c3b with `init_from`, 12 steps, both: 0
Tracebacks, 0 nan, 467/469 tensors matched, guard ceiling 1e8, final val_loss 8.40 / 8.25
(the frozen coda reading a near-identity cell), train `code_target_cos` 0.0048 / 0.0019 with
`_cos_shuf` 0.0053 / 0.0024 (the floor instrument reads), `code_target_acc` 0.0015 (chance).
No `[train_only]` line prints with the empty list, so the trainable count stays unverified
until the run's own log. Queue lines inserted at the HEAD at 16:57 (commit 20a4c3b), ahead
of arm B and the continuations; watcher `spark_code_probes_only.sh` started 16:57 (semantic
+ corpus-mean probe at 10000 and 20000). Predictions unchanged.

Amended 2026-09-17 18:10 (reason: the L2 draw was killed at step 3500 and the InfoNCE line
pulled before it started; this draw is void, not a result): with the prelude and embeddings
trainable, the frozen E's target is NOT fixed. E reads the live prelude's states of the next
span; the prelude drifts under the loop's gradient, and E's codes collapse toward one
direction: train own cosine 0.61 / shuffled 0.56 at 500-1000, 0.99 / 0.98 by 2500-3000
(val 0.989 / 0.984 at 3000), regression loss 0.016, the frozen coda's oracle CE 13.1 nats
(1.4 on arm A). No gradient reaches E's output (it runs under `no_grad`), so this is drift
of E's INPUT, not an optimised collapse; either way the target measures nothing. The fix
(built next, `tul.code_target_ref`): a FROZEN reference copy of the VAE model computes the
target (and, on the graded arm, samples and grades) so E's code is the VAE stage's code
whatever the live front does. The rerun gets a new planned file. Run rate before the kill:
16,758 tok/s (1.17x arm A's 14,393; P-C7 would have failed at 1.25x) at peak 7.9 GB (10.9).
Trainable count printed by the run: none (no `[train_only]` line with the empty list).
Correction 18:12: the runner consumed the InfoNCE line at 18:04 before it was pulled; that
draw ran 200 steps (16,884 tok/s) and was killed at 18:12 by PID. Both draws void.

## Not verified before launch

- The 24 CPU contracts (`tests/test_tul_code_target.py`); the GPU smoke's exit code is
  recorded here before the queue lines are written.
- Whether `train_only: []` leaves any parameter other than E and `W_prefix` frozen
  (the smoke prints the trainable / frozen counts; recorded below).
- One seed per arm; no coda trains, so nothing here reads generation quality.

## Results

## Verdict

## Updated hypothesis
