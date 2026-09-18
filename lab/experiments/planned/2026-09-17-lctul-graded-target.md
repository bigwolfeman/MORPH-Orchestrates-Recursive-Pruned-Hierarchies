# Experiment: the GRADED-continuation target — the loop proposes, a blind grader ranks

Status: planned

Date: 2026-09-17 (frozen before any GPU step beyond a 12-step smoke on the Spark).
Note: [`.agents/notes/proposed/architecture/2026-09-17-lctul-graded-continuation-target.md`](../../../.agents/notes/proposed/architecture/2026-09-17-lctul-graded-continuation-target.md).
Spec: `docs/tul-code-spec.md` §17.2. Tests: `tests/test_tul_code_grade.py` (CPU).
Parent: [`2026-09-17-lctul-code-only.md`](2026-09-17-lctul-code-only.md) — this is the arm its
P-C6 binding names ("a target that is computed, not predicted").
Earlier graders: [`2026-09-12-arc-core-token-and-critic.md`](../failures/2026-09-12-arc-core-token-and-critic.md).

## Question

Every target the slot loop has been given is a deterministic function of the past, and
every one is met in one pass (arm A at 10k: `cos_l0` 0.06, `l1` 0.135, `l6` 0.132, exit
cosine flat at 0.17 since step 1500). If the target is instead COMPUTED — the loop's own
cell proposes `K` continuations through the frozen coda, a grader that never sees the true
span ranks them, and the loop is pushed toward the winner's code — does a per-pass signal
appear, and does the cell learn anything the L2 target did not?

## Hypothesis

A conditional mean is a point, and one pass reaches a point. An argmax over proposals is
not a point: it moves when the proposals move, so pass `t+1` can beat pass `t` on the same
objective. If the flat K-curve is a property of the TARGET's shape, the graded term moves
it. If the flat K-curve is a property of the MAP, nothing here moves and the target shape
is closed as an explanation.

Two failure modes are anticipated and instrumented rather than argued away. The grader may
not select for anything the code measures (`cos_best_true` ≈ `cos_worst_true`). And the
signal is thin by construction: about four graded slots per step-equivalent out of ~300
valid slots, which is the price of the 0.5× rate ceiling.

## Predictions (frozen)

Arms `tul-code-grade` (`tul_code_grade.yaml`: `code_grade_loss: pref`,
`code_grade_grader: coda_past`, `code_target_weight: 0`) and `tul-code-grade-l2`
(`tul_code_grade_l2.yaml`: the same with the ordinary `code_target` L2 term ON at 1.0).
Both `init_from` `tul-code-vae/step_10000.pt`, 20k steps, batch 6, seq 1024, one seed each,
`data_skip_batches` 10000, `train_only: []`, the slot-loop gain constraint as arm A.
Defaults `code_grade_k: 4`, `code_grade_tokens: 16`, `code_grade_rows: 1`,
`code_grade_every: 8`, `code_grade_temp: 1.0`, `code_grade_tau: 0.1`. Readings from wandb
(`train/code_grade_*`, `tul/code_grade_cos_l{t}`, `train/code_target_cos`,
`train/code_target_cos_shuf`) and the corpus-mean probe at 10k and 20k. Baseline for every
delta is arm A's 10k reading (own 0.145 / 0.149, shuffled 0.021 / 0.048, rank 15.5 / 16.6,
`l6 − l1` −0.002) and the code-only arms' 20k series when they land.

- P-G1 (grader sanity): `train/code_grade_true_rank` ≥ 0.25 averaged over the last 500
  logged steps at 20k on `tul-code-grade` — the fraction of non-degenerate candidates the
  TRUE continuation outscores. **65 %.**
  Stated as a rank and not as a difference on purpose. The candidates are drawn from the
  same model that grades them, so `E_p[log p] = −H` sits at or above the truth's `−CE` by
  Jensen and `code_grade_true` is BELOW `code_grade_mean` however good the grader is. The
  CPU fixture reads exactly that at init (best −2.84, mean −3.19, worst −3.55, true −5.16).
  A rank near 0 means the grader puts real text last, which is the failure this catches.
- P-G2 (the grade selects for the code): `train/code_grade_cos_best_true` exceeds
  `train/code_grade_cos_worst_true` by ≥ 0.02, same window, same arm. **50 %.**
- P-G3 (the passes move, to the computed target): `tul/code_grade_cos_l6` −
  `tul/code_grade_cos_l1` ≥ 0.02 on either arm, same window. Arm A's twin reading is
  −0.002. **25 %.**
- P-G4 (the passes move, to the truth): `tul/code_target_cos_l6` −
  `tul/code_target_cos_l1` ≥ 0.02 on either arm, same window. This is the reading that
  matters if P-G3 holds: a per-pass climb toward the WINNER that is not also a climb toward
  the TRUTH is the term fitting its own sampler. **15 %.**
- P-G5 (own minus shuffled): `tul-code-grade` `train/code_target_cos` −
  `train/code_target_cos_shuf` ≥ 0.10 at 20k (arm A 0.124 at the probe; the graded arm's L2
  term is OFF, so this is what the graded term alone teaches the cell). **35 %.**
- P-G6 (the graded term does not cost the L2 arm): `tul-code-grade-l2`
  `train/code_target_cos` at 20k is within 0.02 of `tul-code-only`'s. **55 %.**
- P-G7 (rate): both graded arms ≥ 0.50× `tul-code-only`'s tok/s at batch 6, measured over
  the same step window. The cost arithmetic says 0.53×. **70 %.**
- P-G8 (coverage): `train/code_grade_slot_frac` ≥ 0.30 averaged over the graded steps —
  the fraction of valid slots whose span is at most `code_grade_tokens` long. Below that,
  the defaults are wrong and `code_grade_tokens` is raised before the panel is read.
  **60 %.**

## Binding

- P-G3 and P-G4 both hold → the first per-pass signal in the record, and it is a signal
  about the truth. The family moves to the graded target: raise `code_grade_rows`, spend
  the rate, and run the K-curve.
- P-G3 holds and P-G4 fails → the loop is fitting its own sampler. The term is rejected as
  written and the next arm grades with a judge outside the model (the frozen plain
  checkpoint the note rules out on VRAM, on the Spark instead).
- P-G1 fails (rank below 0.25) → the grader is the bug, not the target. Nothing else in the run is readable;
  re-run with `code_grade_grader: coda_zero` to separate "the grader is blind" from "the
  context channel is empty".
- P-G2 fails while P-G1 holds → grading text does not select codes. The `E` channel cannot
  carry what the grader ranks, and the graded-target family goes to rejected with that
  number.
- P-G3 and P-G4 both fail with P-G1 and P-G2 holding → the target's SHAPE is closed as an
  explanation for the flat K-curve: a computed, state-dependent, refinable target is met in
  one pass too, and the next question is about the map.

## Method

`tul_code_grade.yaml` and `tul_code_grade_l2.yaml` both compose `tul_code_only` (no coda CE
at train, everything trains except the frozen `E`). The graded term runs inside
`_tul_code_target_write`, at the one seam where the predicted cells and the frozen code
already meet. A graded step: choose `code_grade_rows` rows from a step-seeded generator;
eligible slots are `code_target_valid` with a true span of 2..`code_grade_tokens` tokens;
`code_grade_k` candidates are decoded in parallel across EVERY eligible span of the chosen
rows (the strict geometry makes spans mutually invisible, so one forward advances them all);
the candidates are graded by the frozen coda with the graded slot's own cell zeroed (two
parity passes); `E` encodes the candidate rows; the winner's code is the target. Everything
up to the target runs under `no_grad` in eval mode, with the module's training flag and the
CUDA RNG state restored afterwards.

CPU contracts first (`tests/test_tul_code_grade.py`), then a 12-step Spark smoke of each
config with `init_from`, then the queue lines. The smoke's readings are recorded here by
amendment before any queue line is written.

Amended 2026-09-17 19:20 (reason: the Spark smokes and the rate pair ran, and two fixes
landed after the predictions were frozen). Nothing in Predictions is edited.

Smokes on the DGX Spark, 12 steps, `init_from` `tul-code-vae/step_10000.pt`, batch 6,
seq 1024, `data_skip_batches` 20. At 7a9dfe1: `tul_code_grade`, `tul_code_grade_l2` and
`tul_code_only` all exit 0, 0 Tracebacks, 0 nan. At 14f004d (the open-slot fix, below):
`tul_code_grade` and `tul_code_only` exit 0 with the SAME step-0 loss as at 7a9dfe1
(1.5973 and 2.1315) — the fix is eval-only and does not touch the training path. Every run
prints `[code_ref] frozen VAE-stage twin snapshot: 269.9M parameters, 0 trainable, eval
mode`. Step-0 graded readings: `n=20` slots, `frac=0.38`, best −5.00, mean −6.16, worst
−7.39, true −4.10, rank 0.89, cosB 0.147, cosW 0.116, degen 0.00. Peak memory with the twin
on: 7.46 GB (graded) and 7.17 GB (code-only) at step 0, 8.47 / 8.44 GB by step 40. The twin
costs 1.32 GB on its own (6.14 → 7.46 GB, measured). The code-only prereg's 7.9 GB is a
5090 number and is not on this box.

Rate, 60 steps each, same box, same flags, `tul_code_grade` against the same config with
`tul.code_grade=false`: 940 tok/s against 1578 at the step-40 window, and 407 s against
253 s of wall clock including an identical compile warmup. That is **0.60× on the step
window and 0.62× on wall clock**, above P-G7's 0.50 floor and below the 0.53× the
arithmetic predicted (the arithmetic counted block passes and ignored the graded step's
poorer utilisation). The panel runs on the 5090, so P-G7 is still scored there.

P-G1's THRESHOLD stands; its RATIONALE is refuted and is recorded as wrong here rather
than rewritten above. The prediction argued from Jensen that `code_grade_true` must sit
BELOW `code_grade_mean`. That holds only when the grader and the sampler read the same
cell. This grader zeroes the graded slot's own cell, so the truth can and does outscore
the samples: true −4.10 against mean −6.16 at step 0 (rank 0.89), true −4.88 against mean
−5.04 at step 40 (rank 0.51). A rank near 0 is still the failure P-G1 catches, so the
0.25 floor is unchanged.

Two fixes landed between the freeze and the queue, both eval-only:

1. `tul.code_target_ref` (7a9dfe1). `E` is frozen but its INPUT is not — it pools the LIVE
   prelude's states, and on `train_only: []` the prelude drifts and `E`'s codes collapse
   (own 0.99 / shuffled 0.98 by step 3000 on the killed code-only draw). Both graded
   configs set `code_target_ref: true`, so every VAE-stage reading — the target code, the
   coda that samples, the coda-with-zero-cell that grades, the tied head they score
   through — comes from a frozen deep copy of the model snapshotted at `init_from`.
   Consequence for P-G6: its baseline is the code-only RERUN with the twin on, not the
   killed draw.
2. The open slot (14f004d). `_tul_code_target_write` masked the PROJECTION by
   `ok = code_target_valid`, which is False for a row's last valid slot. At generation that
   slot is the OPEN span, so the coda wrote every generated span from a zero cell. The
   projection now takes `ok` at train and `layout.slot_valid` at eval; the loss and the
   cosines keep `ok`. Controlled A/B on the Spark, `tul_code_grade`, 12 steps,
   `eval_every=12`, identical flags, 7a9dfe1 against 14f004d: every `ok`-indexed instrument
   is unchanged to four decimals (`val/ce_tf` 1.5877, `val/code_target_cos` 0.0035,
   `val/code_target_cos_shuf` 0.0057, `val/code_eff_rank` 57.6613), and only the
   instruments that READ the predicted cells move — `val/ce_tokens` 8.5547 → 8.5923,
   `val/plan_worth_zero` −3.5673 → −3.6050, `val/plan_worth_shuffle` −0.0626 → −0.0312.
   Every OWN / SHUF / ZERO probe taken before 14f004d is void, arm A's and arm B's
   included.

## Not verified before launch

- The CPU contracts run at the tiny strict fixture (2 rows, 8 slots, `d_model` 64); no CPU
  test exercises the real 64-slot, 1024-token geometry.
- The rate claim is arithmetic plus a 12-step smoke, not a 20k measurement. The smoke's
  step count is too small to separate compile time from steady-state throughput; P-G7 is
  read from the run.
- Peak memory with `code_grade_rows · code_grade_k` extra rows resident under `no_grad` is
  read from the smoke only, on Spark memory, not on the 5090's 1 GB of slack.
- One seed per arm. Nothing here reads generation quality; no coda trains.
- The grader's absolute scale (`code_grade_true` in nats per token) is not calibrated
  against any external LM; only the ORDERING of the grades is used by the term.
- The open-slot fix is proved on CPU (three tests, one through `generate_tul`) and
  corroborated by the Spark A/B above. Nobody has yet re-run the OWN / SHUF / ZERO semantic
  probe on a real checkpoint with the fix in; until that lands the size of the cell's
  contribution at generation is unknown, only that it is no longer structurally zero.
- The graded arm has never run past 60 steps. Every graded reading quoted here is from a
  model 12 to 60 steps off the VAE checkpoint, with the LR still inside the ramp.

## Results

## Verdict

## Updated hypothesis
