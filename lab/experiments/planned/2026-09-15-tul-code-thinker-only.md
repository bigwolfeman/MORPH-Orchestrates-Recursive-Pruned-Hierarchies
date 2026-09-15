# Experiment: TUL-Code thinker-only — a frozen encoder and coda, 40k thinker steps at 2x batch

Status: planned
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

## Not verified before launch

- The freeze under torch.compile and gradient checkpointing on the real model (the CPU
  test covers the toy model's gradient flow); the runner's 12-step smoke is the gate.
- Batch 12 memory with a frozen token path: estimated, not measured.
- The LR schedule is a function of the global step, so the fresh optimizer starts at the
  flat 1e-4 with no ramp; the model is trained, but the slow EMA starts empty.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
