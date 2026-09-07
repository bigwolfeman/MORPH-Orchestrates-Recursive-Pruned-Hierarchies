# Planned: ARC E6 — the plain loop at a deep recurrence draw (mean 16, truncated backprop 8)

Status: planned
Date: 2026-09-07 (frozen before any smoke; launch is Wolfe's call; ~3 GPU-hours)
Arc: `2026-09-04-loop-contribution-arc.md`, Method amendment 3. Follows the Huginn rule
in `failures/2026-09-04-huginn-loop-contribution.md`.

## Question

Every stable MORPH loop, trained at a Poisson draw of mean 6 (max 8), earns nothing past
iteration 3 on OpenWebText. Huginn-3.5B, trained at a draw centred on 32, earns 0.566 nats
from 3 to 6 and 0.206 from 6 to 16 on the same rows, and stops between 16 and 24. Both
stop near half their training mean. Is the training depth distribution the lever: does a
MORPH loop trained at mean 16 earn past iteration 3?

## Method

`morph/configs/notul_deep16.yaml`: `notul.yaml` (the plain arm, TUL never) with
`model.mean_depth 16`, `model.max_depth 24`, `model.bptt_depth 8` (truncated: the last 8
iterations carry grad, the earlier ones run under `no_grad`, as Huginn trains), the
1000-step ramp, seq 1024, batch 6, 5000 steps, seed 1, `ademamix_alpha_cap 3.5` (the
measured 0/9 ramp recipe; the cap is not varied here). Sustained tripwire
(`lab/divergence/tripwire_sustained.py`) on `probe.jsonl`. At step 5000 (and 2500):
`lab/divergence/token_depth_sweep.py --profile` at forced depths 1, 3, 6, 8, 12, 16, 24 on
the arc's 480 rows; paired-bootstrap CIs; `score_arc_e0.py` at (3, 12). Control: the
notul 5000-step sweep already in `results/2026-09-04-arc-e0/sweep_notul_5000.json` (mean
6, full BPTT). Output `results/2026-09-07-arc-e6/`.

Cost estimate, not measured: the core forward runs ~2.7x more iterations (16 against 6)
and the backward covers 8 either way, so about 2x the notul step time; ~3 h for 5000
steps.

## Predictions (frozen)

- **P6a.** Token K3−K6 at step 5000 > 0.01 nats with the paired CI above 0: **55 %**
  (the half-of-training-mean reading says a mean-16 loop earns to ~8).
- **P6b.** Token K6−K12 > 0.005 with CI above 0: **45 %**.
- **P6c.** Token K12−K16 has a CI covering 0 or below (saturation before the mean):
  **60 %**.
- **P6d.** Val CE at the trained depth (16) at step 5000 is WORSE than notul's at 5000
  by more than 0.02 nats (deep converges slower; matched steps, not matched wall clock):
  **65 %**.
- **P6e.** No detonation under the ramp (sustained tripwire never fires): **75 %** (the
  detonation grew like ρ^T at depth 6; 16–24 iterations under `no_grad` is untested).
- **P6f.** The offset profile of K3−K12 earning rises with offset-in-span (bin 16–31
  earns more than offset 0 with non-overlapping CIs), Huginn's shape: **40 %**.

## Binding

- P6a TRUE ⇒ the training depth distribution is the lever. The loop question reopens as
  a regime question: E5 (20k, matched wall clock) runs at the deep draw, and the arc's
  closing rule is rewritten around the draw, not the data.
- P6a FALSE and P6e TRUE ⇒ the training-depth explanation fails on MORPH at this width
  and length. Remaining candidates, in order: training length (Huginn 800B tokens, this
  arm 30M positions), width (5280 against 768). Neither is a 5090 experiment; the arc
  closes on Wolfe's call.
- P6e FALSE ⇒ file it under the divergence README's regime table; the deep draw is a
  stability experiment before it is a contribution one.

## Not verified before launch

Memory and step time at 24 no-grad iterations on seq 1024 batch 6 (no smoke has run);
whether `token_depth_sweep.py` accepts forced depths above the model's `max_depth` of the
control (it forces per-call, but the 24 cell is new); the Poisson clamp at 24 discards
~2.5 % of draws; whether the ramp's 0/9 record transfers to a loop whose forward is 2.7x
deeper.
