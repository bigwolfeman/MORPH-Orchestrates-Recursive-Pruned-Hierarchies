# Planned: LCTUL-J Stage 1, the EMA code target with the online variance floor

Status: failure

Date: 2026-09-22 (drafted 16:40 while the build ran; frozen and committed before any
GPU step of either arm). Design note:
[`2026-09-22-lctul-ema-target-and-factors.md`](../../../.agents/notes/proposed/architecture/2026-09-22-lctul-ema-target-and-factors.md).
Filter statement (the campaign synthesis,
[`2026-09-22-slot-loop-campaign-synthesis.md`](../../../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md)):
condition A, the coda reads the predicted cells and the tokens directly, so the cell's
job (the next span's code) has the token path as a shallower route for most of what it
could carry; this arm does not remove that route and does not claim to. Condition B, the
regression term pays for the cell at weight 1.0. The arm's claim is narrower than depth:
that a code target which MOVES under the predictor can leave the 0.15-cosine ceiling a
fixed target holds, without collapsing.

Baselines (20k-step readings, quoted for scale; the in-panel control is `ema0` at 5k):
arm A (`tul-code-target`, frozen E, front frozen): `train/code_target_cos` 0.1413 over the
last 500 logged steps, shuffled 0.0196, effective rank of the prediction 17.0 / 18.1,
`val/ce_tf` 1.43 on the frozen coda. `tul-code-only-ref` (frozen twin, front trained):
own 0.1674, shuffled 0.0380, rank 17.4 / 17.6; its void draw without the twin read
shuffled 0.98 by step 2500. Slot-loop K1−K6 yardstick +0.002; fan4-all +0.0049.

## Question

Does a target encoder that follows the online model by EMA, with a per-coordinate
variance floor on the online cells, give the slot loop a code target that (1) stays
active, (2) moves toward what the context can predict, and (3) the coda can still decode,
and does the loop then use more than one pass on it?

## Hypothesis

A fixed reconstruction code is 85 % unpredictable, so the loop finds its predictable part
(0.15 cosine) in one pass and stops. Under an EMA target the front moves toward
representations whose next-span code the loop CAN predict, the target follows, and the
own-minus-shuffled cosine rises well above the fixed target's; the floor keeps the
per-coordinate spread of the online cells above `γ` so the drift cannot end in a point.
Whether the moved target needs more than one pass is the open question this arm reads and
does not assume.

## Method

Two arms on the runner (`run_recon.sh`, KIND `slot`, sweeps at 2500 and 5000 on 480 rows
at forced depths 1, 2, 3, 6, 9, 12, 16; worth profile and slot-state probe at 5000), both
seeded from `tul-code-vae/step_10000.pt` through `training.init_from` in the EXTRA field,
`data_skip_batches 10000`, 5000 steps, the panel settings, the whole model training
(no `train_only`), `code_target_ref: true`, `code_target_detach: true`,
`token_state_dropout 0.0`:

- `tul-code-ema` (`tul_code_ema.yaml`): `code_target_ema 0.996`, `code_enc_var_lambda
  0.02`, `code_enc_var_gamma 1.0`.
- `tul-code-ema0` (`tul_code_ema0.yaml`): `code_target_ema 0.0`, `code_enc_var_lambda
  0.0` (the frozen twin; the one-factor control).

Readouts after both finish: the self-paired K-curve and the pairing of `ema` against
`ema0`, `slot-spandec-strict` and `fan4-all-fp0` at 5k on the same 480 rows
(`core_depth_sweep.py`, block bootstrap); `code_target_mean_probe.py` at 5000 on both
(corpus-mean-removed own and shuffled cosine, effective rank of the predicted cells and
of the target's codes; beside the runner under `nice`, or on the Spark); the wandb
series `val/code_target_cos`, `val/code_target_cos_shuf`, `val/code_tgt_std`,
`val/code_eff_rank`, `val/ce_tf`, `tul/code_enc_var`, `tul/code_enc_std`,
`tul/code_enc_active`, `tul/code_target_cos_l{t}`, `preclip/total`, `loop/in_norm_t0`.
The commit is the one in the runner line and is printed by every readout.

## Predictions (frozen)

- P-1 (the target stays active): on `ema` at 5000, `val/code_tgt_std` ≥ 0.5 × its first
  logged value and `val/code_target_cos_shuf` < 0.10. 70 %.
- P-2 (the target moves toward the predictable): on `ema` at 5000, `val/code_target_cos`
  − `val/code_target_cos_shuf` exceeds `ema0`'s same difference by ≥ 0.05. 45 %.
- P-3 (decodable): `val/ce_tf` on `ema` at 5000 is within +0.30 nats of `ema0`'s. 50 %.
- P-4 (the passes move on the code): `tul/code_target_cos_l6` − `tul/code_target_cos_l1`
  ≥ 0.02 over the last 500 logged steps on `ema` (arm A +0.0037; `code-only-ref` failed
  the same bar). 25 %.
- P-5 (token depth): token K1−K6 at 5000 on `ema` ≥ +0.010 with a CI clear of +0.002
  (read through a coda that trains, so the frozen-coda genericity artefact does not
  apply). 25 %.
- P-6 (the token path is not taxed): paired depth-6 token CE, `ema` minus `ema0`, inside
  [−0.03, +0.03]. 60 %.
- P-7 (rate): `ema`'s tok/s at step 200 ≥ 0.90 × `ema0`'s (one `foreach` lerp per step
  over the twin). 85 %.
- P-8 (health): `preclip/total` < 1e4 at every step ≥ 200 on both arms; `loop/in_norm_t0`
  growth ≤ 2.0x over the run. 80 %.

## Binding

- P-1 fails: the recipe does not hold the target at `m 0.996`, `λ_enc 0.02`; ONE retry
  is allowed under this file (dated amendment naming the lever): either `m 0.999` or
  `code_enc_var_lambda 0.2` (the floor is 1 % of the objective at 0.02: the smoke reads
  the raw term 0.50 against a regression term near 1.7, and even a fully collapsed cell
  reads it at most `γ` = 1.0). If the retry fails too, both go to `failures/`, Stage 2 is
  not built on this target, and the note's "Risks" first bullet is the finding.
- P-1 holds and P-2 fails: the EMA moved nothing the fixed target did not already have;
  file under `failures/`; Stage 2 is not built; the moving-target lane closes at one arm.
- P-1 and P-2 hold and P-4 fails: the target is healthy and moved, and the loop still
  finds it in one pass; file under `failures/` for the depth question, and Stage 2 (the
  per-pass factor split) is the next build with its own prereg.
- P-1, P-2 and P-4 hold: file under `successes/`; the arm goes to 20k with `ema0` paired.
- P-3 failing on its own does not change the branch; it is recorded as the cost of the
  moved target to the reader.
- P-6 failing (a tax above 0.03) is recorded; it blocks the 20k run until read.

## Step-0 readings from the 12-step smoke (before the run; not a result)

`tul_code_ema` at 8443c79 plus the build, seeded from the VAE stage: `tul/code_enc_std`
0.499 (the online cells' per-coordinate std across the batch's slots), `tul/code_enc_active`
0.998 (the floor is active on almost every coordinate at init: the loop's exit cells share
a large common direction at step 0, the corpus-mean effect arm A read), `tul/code_tgt_std`
0.986 (the target's own spread; the target nearly meets `γ` on its own),
`tul/code_enc_var` 0.501, weighted 0.0100 = 0.02 × 0.501. Peak memory 10.3 GB allocated
(with the compile warmup), about 1.5 to 2.1 steps/s after warmup on the smoke's short
window. P-1's first-value reference for `code_tgt_std` is the run's own first logged
value, not this 0.986.

## Not verified before launch

- The paper states neither `m` nor `γ`; 0.996 and 1.0 are our choices.
- The EMA over parametrised (ternary shadow) tensors on the compiled model: covered by the
  CPU test and the 12-step smoke only.
- Memory with the whole model training beside the twin: `tul-code-only-ref` trained the
  front and the loop beside the twin at 20k, so the shape has run; the floor's extra
  tensors are small. The smoke's peak is the reading.
- The `ema0` control differs from `tul-code-only-ref` (the coda trains here and reads
  the predicted cells; that arm had no coda at train), so 20k numbers above are scale
  only.

## Amendment 2026-09-22 (17:14): the one retry, lever named

`tul-code-ema` (run twx0472q, 13a9535) finished 5000 steps healthy (tripwire max 24.3 at
4625, exit 0). P-1's first clause holds (`val/code_tgt_std` 0.837 at 4750 against a first
value of 0.985, ratio 0.85) and its second clause FAILS: `val/code_target_cos_shuf` 0.448
(bar 0.10). The own cosine reads 0.575 but the shuffled cosine rose with it from 0.073 at
step 250, so the target drifted toward one common direction (effective rank 58 → 24,
`val/ce_tf` 1.73 → 4.02, i.e. the coda's read of the TRUE code ended at the token CE
itself: the code became worth nothing to the reader). The online floor was active on
every coordinate the whole run and the online spread still fell (`tul/code_enc_std` 0.50
→ 0.44), so the lever is the floor's WEIGHT, not the momentum. Per the Binding, the one
retry is `tul_code_ema_l2.yaml` (`code_enc_var_lambda 0.2`, ten times the paper's value;
`m` 0.996 unchanged; one factor over `tul_code_ema.yaml`), wandb `tul-code-ema-l2`,
queued behind `ema0` at the same seed and data offset. The frozen predictions apply to the
retry unchanged, with `ema0` still the control. No other prediction is scored before
`ema0` lands. P-4 on the first draw reads +0.0017 (fails); recorded now so the number
cannot be re-read later.

## Amendment 2026-09-22 (17:19): Wolfe's call on the cosine bars

Wolfe, after the first draw: "I think you're reading tea leaves with the cosine. I
wouldn't pass or fail anything by it." Applied as follows, without touching the frozen
text above. P-1's second clause, P-2 and P-4 are cosine bars; they are scored as written
for the record and are NOT the basis of the verdict or the binding branch. The verdict
and the branch are decided on the nats and rank instruments already in the Method:
`val/ce_tf` against the arm's own token CE (the true code's worth to the reader; on the
first draw 1.73 → 4.02 = the token CE, so the code ended worth nothing), the worth
profile of the predicted cell (first draw: zero +0.168, shuffle +0.142 total), `val/
code_eff_rank` (58 → 24), the token K-curve (+0.0006 [+0.0002, +0.0010]) and the paired
depth-6 CE against `ema0`. The retry `tul-code-ema-l2` was queued on P-1's cosine clause
at 17:14; it stands on the nats reading alone (the reader lost the whole worth of the
true code and the rank halved), so it runs. The health readings for the retry are
`val/ce_tf` staying clear of the token CE and the rank staying up; the cosines are read
as diagnostics of why.

## Results

Three arms on the runner at 13a9535 (the retry at a55ddae, config only), each seeded from
`tul-code-vae/step_10000.pt`, 5000 steps, whole model training, the coda reading the
predicted cells with stop-gradient. Sweeps on 480 rows at 2500 and 5000; worth profile
and slot-state probe at 5000; the corpus-mean probe on the Spark (8c00cf3, `tg_seg` 24)
at 5000, 96 rows, depths 1 and 6. Artifacts: `../results/2026-09-22-lctul-ema/`. wandb
runs: `ema` twx0472q, `ema0` okcopzg0, `ema-l2` y6k36fqo. Per the 17:19 amendment the
verdict rests on nats and rank; the cosine clauses are scored for the record only.

| instrument at 5000 | ema (m 0.996, floor 0.02) | ema0 (frozen twin) | ema-l2 (floor 0.2) |
|---|---|---|---|
| paired depth-6 token CE, arm − ema0 (500 blocks) | +0.0038 [+0.0020, +0.0058] | 0 | −0.0049 [−0.0068, −0.0030] |
| paired depth-1 token CE, arm − ema0 | +0.0028 [+0.0009, +0.0048] | 0 | −0.0051 [−0.0071, −0.0032] |
| token K1−K6 (self-paired, 480 rows) | +0.0006 [+0.0002, +0.0010] | +0.0016 [+0.0012, +0.0020] | +0.0014 [+0.0010, +0.0018] |
| worth of the predicted cell, zeroed, total / first token | +0.168 / +0.608 | +0.165 / +0.591 | +0.174 / +0.609 |
| worth, shuffled, total / first token | +0.142 / +1.259 | +0.122 / +1.048 | +0.152 / +1.434 |
| `val/loss` at 4750 (the runner's val batches) | 4.104 | 4.103 | 4.101 |
| `val/ce_tf` (the coda on the TRUE code) 250 → 4750 | 1.73 → 4.02 | 1.67 → 4.12 | 4.29 → 3.98 (already at the token CE by the first val) |
| target per-coordinate spread `val/code_tgt_std` 250 → 4750 | 0.985 → 0.837 | 0.987 → 0.987 | 0.984 → 0.830 |
| target effective rank per cell (Spark probe, depth 6) | 39.8 / 38.3 | 77.5 / 70.2 | 39.6 / 37.1 |
| target's cosine to the corpus mean (probe) | 0.53 / 0.51 | 0.06 / 0.09 | 0.52 / 0.51 |
| prediction effective rank per cell (probe) / `val/code_eff_rank` | 16.6 / 16.3; 24.0 | 17.2 / 18.0; 25.9 | 17.1 / 16.8; 22.5 |
| online cell spread `tul/code_enc_std` 0 → last-500 mean | 0.499 → 0.442 | not charged | 0.499 → 0.541 |
| own / shuffled cosine (record only) | 0.575 / 0.448 | 0.150 / 0.037 | 0.588 / 0.424 |
| centred own / shuffled cosine (probe, depth 6, record only) | 0.300 / 0.002 | 0.141 / 0.002 | 0.317 / 0.005 |
| per-pass cosine l6 − l1, last 500 steps (record only) | +0.0017 | +0.0011 | −0.0009 |
| tok/s at step 200 | 12,419 | 12,625 | 12,451 |
| `preclip/total` max at step ≥ 200; entry-norm growth | 24.3; 1.16x | 22.1; 1.17x | 21.8; 1.15x |

Against the strict ruler and `fan4-all-fp0` at 5k on the same rows, `ema` reads −0.228
[−0.234, −0.222] and −0.175 [−0.182, −0.170] at depth 6: the VAE stage's 10k-step head
start, shared by all three arms, not the target.

Scoring of the frozen clauses (for the record). P-1 clause 1 holds on `ema` (spread ratio
0.85) and clause 2 fails (shuffled 0.448); P-2 holds on the raw cosine (gap 0.127 against
`ema0`'s 0.113 is +0.014, under the 0.05 bar: FAILS); P-3 holds (4.02 against 4.12); P-4
fails (+0.0017); P-5 fails (+0.0006); P-6 holds (+0.0038); P-7 holds (0.98x); P-8 holds.
Retry: P-1 clause 1 holds (ratio 0.84), clause 2 fails (0.424); P-2 fails (gap 0.164 against 0.113 is +0.051, at the bar's edge, and it is a cosine); P-3 holds; P-4 fails (−0.0009); P-5 fails (+0.0014); P-6 holds (−0.0049); P-7 holds (0.99x); P-8 holds. The retry is scored the same as the first draw..

What the nats and rank say. The EMA target moved: its rank halved (77 → 40 per cell) and
half of the movement went onto one shared direction (cosine to the corpus mean 0.06 →
0.53). Nothing of that reached the reader: the two arms have the same val loss to three
decimals through the whole run, the same cell worth within 0.02 nats, the same K-curve at
the floor, and `ema` is 0.004 nats WORSE paired at both depths. The coda's read of the true
code collapses to the token CE on the CONTROL as well (1.67 → 4.12), so `val/ce_tf` is a
reader statement (the coda trained on detached predicted cells forgets E's code, the
target-unfreeze finding) and not a target statement; it is not used. The retry at ten
times the floor weight held the online cells' spread at 0.54 (the first draw's fell to 0.44) and changed nothing else: the target drifted to the same rank (39.6 / 37.1) and the same mean axis (0.52), the paired token CE moved 0.005 the other way, the cell worth 0.174 against 0.165..

Why the floor did not hold the target, found while the retry ran and recorded here as an
error of the Stage 1 mapping: the paper's `L_enc` floors the ONLINE ENCODER's output, the
representation whose EMA is the target (eq. 9: "the online context representations z_c;
for token-valued contexts, over all valid context tokens"). In LCTUL the encoder whose EMA
makes the target is the front plus E, so the floor belongs on the online FRONT's token
states (what E pools), and Stage 1 put it on the loop's predicted cells (the predictor's
output) instead. That floor bounds the predictor, which never collapsed; the target
collapses through the twin front following the live front, which the regression pushes
toward representations whose next-span code is easy to predict. A weight of 0.2 held the
online cells' spread at 0.55 (against 0.44 at 0.02) and the target drifted the same. The
corrected placement is a different arm and is not run under this file.

## Verdict

**Failure**, the first draw and the one retry alike. On the instruments the 17:19 amendment
names: the EMA target moved (rank 77 → 40 per cell, cosine to the corpus mean 0.06 → 0.53,
on both draws) and the reader saw none of it (val loss equal to three decimals, cell worth
0.165 / 0.168 / 0.174, K-curve at the slot-loop floor on all three, paired token CE inside
±0.005 of the control). The binding's first branch applies as written: the retry failed
too, so both go to `failures/`, Stage 2 is not built on this target, and the note's first
Risk ("the floor acts on the online cells; the target inherits the online front by EMA, so
a slow drift of the target into a low-rank code is still possible") is the finding, with
one correction: the drift is not slow. Half the target's rank is gone by step 1500.

Two errors of my own are on this record. (1) The floor was placed on the predictor's
output; the paper floors the encoder's output (see Results, last paragraph). A weight
ten times the paper's could not reach the target from there, and the retry measured that.
(2) I named `val/ce_tf` as a verdict instrument at 17:19; the control showed within the
hour that it collapses on a frozen target too, so it is a reader statement. Wolfe's call
that no arm passes or fails on a cosine stands and was applied.

## Updated hypothesis

A code target that can move will move toward what the predictor finds easy, and with the
predictor's output as the only floored tensor it moves onto a shared direction. Whether a
floor on the FRONT's states (the tensor whose EMA is the target) holds the target's rank
is a different arm and an open question; it is not run under this file. What is settled
here for the slot loop: with the coda reading a stop-gradient cell and trained on tokens,
the identity of the code target is decoupled from the token loss, so no target, fixed or
moving, can change what the reader gets unless the cell's route into the token loss
changes. The 0.13-cosine span-specific part the loop predicts in one pass showed up on the
fixed target (centred 0.14) and doubled on the moved one (0.30 to 0.33) without any
change on the token side; per Wolfe it is not read as a result. The token K-curve on every
LCTUL-J arm is the floor (+0.0006 to +0.0016); the passes still do nothing.
