# Experiment v1a: does the MUX head make the plan load-bearing, without candidates?

Status: **planned, code implemented and unit-tested, run NOT started.** Predictions
frozen 2026-08-25 before the run. Split out of
[the soft-min arm](2026-08-25-gradient-flow-soft-min-arm.md) (now v1b) so each
lever family is tested alone — MUX-alone costs ≈ nothing, so it goes first.

## Question

The plan is empty because its only direct supervision is a one-token race it
loses (the 2026-08-25 pivot; core Shapley 0.0007 nats, plan-off 0.0191, `cf` < 0
everywhere). Two minimal changes: retire that race, and give z direct span-level
content gradient through the MUX local head (arXiv 2607.18264). No candidate
latents, no soft-min — z stays deterministic. Does that alone make the plan
load-bearing and keep the takeover from firing?

## Hypothesis

The MUX head fills z with next-span content (its gradient does not route through
the coda's suppressed readout), Prop 16 preserves the coda's routing to z as the
local loss falls, and retiring `ce_emit` removes the fuel of the 90%-norm/
zero-value gradient war. Deterministic-z ceiling accepted: the loss minimizer is
the expected bag given context (proper, CE-shaped); the K-candidate mode-committing
upgrade is v1b's question.

## Arm

Config: `morph/configs/tul_v1a.yaml` — `tul_a1` plus `emit_weight 0.0 /
plast_weight 1.0 / mux_beta 1.0 / mux_rho 0.9 / mux_tau 1.0` (the paper's values).
Fresh run, batch 6, 3500 steps, seed 1, `eval_every 250`, `ckpt_every 500` —
exactly the seedsweep protocol, so the CONTROL is the existing
`seedsweep-s1` run (its log and checkpoints at /home/wolfe/morph-scratch/seedsweep/).
`ademamix_t_beta3` stays null on BOTH: for fresh same-length runs the
`training.steps` fallback is matched by construction (the 5000 pin in v1b's notes
applies to resume-based arms only).

Implementation (shipped with this doc, all defaults bit-identical to A1 —
`tests/test_tul_mux.py`, 6 tests):
- `morph/model/tul.py` — `TULConfig.{emit,plast}_weight`, `mux_{beta,rho,tau}`,
  `mux_span_targets()` (sparse geometric targets; the dense |V| vector is never built).
- `morph/model/transformer.py` — `_tul_mux_loss` (reads h_slots through the
  model's OWN `_readout` → unembedding; zero new parameters), loss fold-in with
  `mux_weighted` exposed so train/loss and val loss stay the MODEL's CE
  (the spectral-penalty precedent).
- `morph/training/train.py` — subtracts `mux_weighted` from reported CE; logs
  `tul/mux_local` per step and `val/mux_local` at eval.

## Predictions (frozen 2026-08-25, before any run)

Baselines: plan-off worth 0.0191 nats and raw slot/token readout ratio 0.055
(ROLL_step_1750, same config family at batch 6); A1 aborts at 1800–2940 by seed
(seedsweep, batch 6, 3500 steps); control curves = `seedsweep-s1`.

- **P1 (survival):** v1a reaches step 3500 with no takeover abort and median
  core pre-clip gradient share < 0.5.
- **P2 (value):** plan-off ablation cost (`slot_path_worth.py`) at the step-3000
  checkpoint ≥ **0.04 nats** (>2x baseline).
- **P3 (readout):** raw slot/token Jacobian ratio (`readout_jacobian.py`) at
  step 3000 HIGHER than at the arm's own step 1500 — the falling trend reverses.
- **P4 (no tax):** `val/ce_tokens` at step 3000 within **+0.05 nats** of
  seedsweep-s1 at the same step.
- **P5 (head bites, refuter):** at step 3000 the mean mux local CE beats 0.8x the
  corpus **unigram prior's** CE against the same targets (best span-independent
  predictor, same batches). If it cannot, z carries nothing span-specific and
  the head is decorative regardless of P1–P4.

Failure reading: P1 holds but P2/P3/P5 fail → the war was `ce_emit` alone and
deterministic z stays empty → v1b (K candidates) is the next question, already
pre-registered.

## Method notes

- Sequential runs only on the 5090 (UPS). wandb on, full config (the tul manifest
  now carries the mux fields).
- Seed-matched single-run comparison is unreadable for CE deltas (6.5% spread
  memory) — P4's +0.05 tolerance is deliberately loose; P2/P3/P5 are within-run.
