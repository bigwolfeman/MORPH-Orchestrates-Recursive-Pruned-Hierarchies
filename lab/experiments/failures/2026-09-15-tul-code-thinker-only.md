# Experiment: TUL-Code thinker-only — a frozen encoder and coda, 40k thinker steps at 2x batch

Status: failure
Date: 2026-09-15
Owner: Claude (session f9558148), for Wolfe ("keep the decoder frozen ... push higher batch
and steps for a significantly higher token count")

## Question

In the joint runs the thinker's flow loss sat flat from step 4000 (the target, E's code,
kept moving under the coda's CE), and in the p 1.0 run — E frozen from 10000 — it kept
falling for the whole second half (0.33 → 0.21 of the target's variance by 20k) while the
sample's residual fell 1.8 → 1.5×. Is the thinker undertrained in that specific sense: too
few steps on a FIXED target? With E and the coda frozen at tul-code-20k's step 10000 and
the thinker alone trained for 40k more steps at batch 12 (491M tokens, 4× the panel's
123M), does the sample close on the truth, and does the frozen phase-2 coda — which trusts
codes fully — read it?

## Hypothesis

H-T1: the thinker keeps learning on the fixed target for the whole run; the sample's
residual falls well below the joint runs' 1.5–1.9×, and the trusting coda's sampled CE
falls with it, from about 12 nats toward the ruler.
H-T0 (null): the loss share flattens within 10k steps and the residual stops above 1.3×:
the target's unpredictable part, not training time, bounds the guess.

## Predictions (frozen)

Reference: the resume point tul-code-20k step 10000 (sampled one draw 12.8 on the trusting
coda, ce_tf 0.35, flow share 0.33, residual not measured at 10k; at 20k joint 1.65–1.92);
rollout1 at 20k after 10k steps on a fixed target: share 0.21, residual 1.49–1.64.

- P-T1. `train/code_fm_rel` at the end (step 50000) ≤ 0.15. 55 %.
- P-T2. Sample residual over z variance (rank-128 head) at 50000 ≤ 1.30. 50 %.
- P-T3. Sampled one-draw CE on the frozen trusting coda at 50000 ≤ 8.0 (from 12.8 at the
  resume point), i.e. the guess is read as partly right. 45 %.
- P-T4. The k-curve (8-draw marginal, k = 16 − k = 1) at 50000 ≤ −1.0 — on a trusting coda
  the sampler's depth keeps its worth (phase-2 readings −1.0 to −1.3). 65 %.
- P-T5. The flow share is still falling at the end: share(50000) ≤ share(40000) − 0.01.
  55 %.
- P-T6. Healthy; batch 12 fits (peak ≤ 28 GB); rate ≥ 40k tok/s (no backward through the
  token path). 70 %.

## Binding

- P-T1, P-T2 and P-T5 hold → time on a fixed target is the lever; the next runs train the
  thinker for longer still and only then unfreeze the coda (a short adaptation phase).
- P-T1 holds and P-T2 fails → the L2 flow floor falls but the sample does not close: the
  sampler (guidance, coupling) is the lever, not training time.
- P-T5 fails and P-T2 fails → the target bounds the guess (H-T0): the predictability arm
  (jepa) or a different target.

## Method

`morph/configs/tul_code_thinker.yaml`: `tul_code` + `training.train_only` = the core body,
its six injection terms (indices 4–9 of a 4:6×6:4 model), the diagonal injection, the
velocity head, the time embedding and the two cell markers; everything else frozen
(`morph/training/freeze.py`, before the optimizer is built). `training.resume` =
tul-code-20k `step_10000.pt`, `resume_fresh_optimizer: true` (fresh optimizer over the
trainable subset, no data replay), steps 50000 (40k after the resume point), batch 12,
seq 1024, `ademamix_t_beta3` 40000, `code_phase3_at` 1.0 (no rollout: the coda is
frozen), truth cells at the parent's statistic (no renorm; eval feeds the encoder code at
that statistic). Checkpoints every 10000; runner sweeps at 20000 … 50000; Spark k-sweep at
each, subspace probe and span samples at 50000. The trusting coda makes the sampled CE a
direct read of guess quality (a bad guess reads ~12, the truth reads 0.35).

Method amendment 1 (2026-09-15 16:10, before any checkpoint): the first launch spent two
thirds of its wall clock in validation (a val every 250 steps with the 8-draw marginal at
batch 12, about 90 s each; 6000 steps in 55 minutes). Relaunched from the same fork point
with `training.eval_every` 1000 and `tul.code_marginal_k` 4. The predictions are
unchanged; P-T4 reads the Spark k-sweep (8 draws), not the trainer's marginal.

## Not verified before launch

- The freeze under torch.compile and gradient checkpointing on the real model (the CPU
  test covers the toy model's gradient flow); the runner's 12-step smoke is the gate.
- Batch 12 memory with a frozen token path: estimated, not measured.
- The LR schedule is a function of the global step, so the fresh optimizer starts at the
  flat 1e-4 with no ramp; the model is trained, but the slow EMA starts empty.

## Results

Run: wandb `tul-code-thinker` (ywr5qrn5, its own run; the two earlier resumes into the
parent's id e2zl3cha lost every logged row, see the memory note), 50000 steps, exit 0,
runner verdict HEALTHY, 69.5k tok/s at batch 12, 5.4 GB peak. `train/loss` (the token CE
through the frozen coda) sat at 0.38–0.43 for the whole run, by construction. Probes on
the Spark, 96 validation rows (4900 valid slots), the same rows at every step; artifacts
in `lab/experiments/results/2026-09-15-tul-code-thinker/` (`readings.txt` has the table).

Flow share (`code_flow_probe.py`, train mode, dropout off, ± is the bootstrap half-width):

| step | share | band0 | band1 | band2 | band3 |
|---|---|---|---|---|---|
| 10000 (fork) | 0.320 | 0.485 | 0.362 | 0.230 | 0.212 |
| 20000 | 0.293 ± 0.003 | 0.461 | 0.324 | 0.207 | 0.189 |
| 30000 | 0.286 ± 0.003 | 0.456 | 0.313 | 0.201 | 0.183 |
| 40000 | 0.282 ± 0.003 | 0.453 | 0.307 | 0.198 | 0.180 |
| 50000 | 0.279 ± 0.003 | 0.450 | 0.302 | 0.195 | 0.179 |

Gains per ten thousand steps: 0.027, 0.007, 0.004, 0.003. The training-time series
(`train/code_fm_rel`, 1000-step means) agrees: 0.336 at 11k, 0.300 at 15k, and the
gradient norm fell from 0.94 to 0.11–0.26 within the first 3000 thinker-only steps at the
flat 1e-4 learning rate.

Sample residual over the code variance (`code_subspace_probe.py`, k = 8, cells 0 / 1;
2.0 is an independent draw, 0 a perfect guess):

| step | r = 4 | r = 32 | r = 128 | full |
|---|---|---|---|---|
| parent 20000 (joint) | 1.70 / 1.51 | 1.74 / 1.65 | 1.85 / 1.83 | 1.92 / 1.91 |
| rollout1 20000 | 1.25 / 1.25 | 1.49 / 1.48 | 1.62 / 1.62 | 1.65 / 1.64 |
| thinker 20000 | 1.57 / 1.49 | 1.62 / 1.58 | 1.70 / 1.69 | 1.74 / 1.75 |
| thinker 30000 | 1.54 / 1.45 | 1.60 / 1.56 | 1.69 / 1.69 | 1.74 / 1.74 |
| thinker 40000 | 1.54 / 1.47 | 1.59 / 1.56 | 1.68 / 1.68 | 1.72 / 1.73 |
| thinker 50000 | 1.53 / 1.46 | 1.59 / 1.55 | 1.68 / 1.68 | 1.72 / 1.73 |

Frozen phase-2 coda, sampled codes (`code_marginal_sweep.py`, 8 draws; the runner's
one-draw depth sweep at 20000 read 12.3–13.9 at every depth ≥ 1 and 0.36 at depth 0):

| step | one draw k = 8 | marginal k = 1 | marginal k = 16 | k16 − k1 |
|---|---|---|---|---|
| 20000 | 13.59 | 11.06 | 9.58 | −1.48 [−1.58, −1.39] |
| 30000 | 13.85 | 11.60 | 9.85 | −1.75 [−1.87, −1.62] |
| 40000 | 13.62 | 11.50 | 9.64 | −1.86 [−1.98, −1.73] |
| 50000 | 13.66 | 11.58 | 9.70 | −1.88 [−2.00, −1.73] |

Span samples at 50000 (`code_span_samples.py`, 10 cuts): the same picture as at the fork point (`code_span_samples_tul-code-20k_10000.txt` in the cond results): ORACLE returns the true span word for word from the encoder code, GREEDY and both SAMPLE lines are word salad from the first token on, and no cut is more on topic than at 10000 (`code_span_samples_tul-code-thinker_50000.txt`).

Scores:

- P-T1 (share ≤ 0.15 at 50000): FAIL, 0.279.
- P-T2 (rank-128 residual ≤ 1.30): FAIL, 1.68 (full rank 1.72–1.73; the joint parent read 1.92 at 20k, rollout1 1.65).
- P-T3 (one-draw sampled CE ≤ 8.0): FAIL, 13.66 one draw at k = 8 (12.8 at the resume point; the runner's one-draw sweep reads 12.5–13.8 at every depth ≥ 1).
- P-T4 (k16 − k1 ≤ −1.0): HOLDS, −1.88 [−2.00, −1.73].
- P-T5 (share(50000) ≤ share(40000) − 0.01): FAIL, the step is 0.003.
- P-T6 (healthy, batch 12 fits, ≥ 40k tok/s): HOLDS.

## Verdict

Status: failure. Two of six predictions held, and the two that decide the question
(P-T2, P-T5) failed. The prereg's binding for that pair: the target bounds the guess
(H-T0). Forty thousand thinker-only steps on a fixed target, 491M tokens at twice the
panel's batch, moved the flow share from 0.320 to 0.279 and the full-rank sample residual
from about 1.9 (joint) to 1.72–1.73, with the last twenty thousand steps worth 0.007 of
share and 0.02 of residual. The thinker reached its floor within the first ten thousand
steps. The frozen-target thinker also learned SLOWER per step than rollout1, where the
coda and prelude kept training beside it at half the batch (share 0.209 and residual
1.65 by 19–20k); so "a moving target starved the thinker" is not the story either. The
sampler's Euler depth keeps its worth on a trusting coda (P-T4), at a level (10–14 nats)
that no coda can use.

## Updated hypothesis

The code is a verbatim copy of the next span (the 20k panel's finding), and a
conditional flow field trained on the seed plus the tape can only fall to the entropy of
the span given the past, which on web text is most of the code. Training time, batch,
target motion and guidance are all measured and all fail to move the sample off an
unconditional draw by more than the top few code directions. The next arm must change
what the code IS: the predictability arm (`tul_code_jepa.yaml`, `code_target_lambda`
0.1, queued next on the runner) lets the flow loss shape the encoder toward what the past
determines; a code defined by the past rather than by the span is the alternative. Not
verified here: a second seed, a lower or decaying learning rate on the fixed target (the
early gradient-norm collapse argues against it), and a coda adaptation phase after the
thinker (pointless while the sample is an unconditional draw).
