# Planned: LCTUL-J Stage 1, the EMA code target with the online variance floor

Status: planned

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
