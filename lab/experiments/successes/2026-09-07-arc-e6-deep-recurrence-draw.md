# Success: ARC E6 — the plain loop at a deep recurrence draw (mean 16, truncated backprop 8)

Status: success
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

## Results (2026-09-07 01:36; `arc/run_e6_e7.sh` on worktree e0501b2; run 00:22–01:26, 5000 steps, exit 0, 1.34–1.37 steps/s, peak 10.06 GB; sweeps on the arc's 480 rows at 2500 and 5000; files in `results/2026-09-07-arc-e6/`)

Forced-depth CE, all 480 rows (the mean-6 ruler is `results/2026-09-04-arc-e0/sweep_notul_5000.json`, same rows, same step):

| depth | 1 | 3 | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| E6 @2500 | 5.198 | 4.622 | 4.560 | 4.549 | 4.539 | 4.537 | 4.540 |
| E6 @5000 | 4.939 | 4.397 | 4.120 | 4.095 | 4.080 | 4.075 | 4.079 |
| notul @5000 (mean 6) | 4.008 | 3.972 | 3.971 | 3.973 | | | |

Paired row bootstraps at 5000 (`paired_ci.txt`):

| pair | E6 | notul |
|---|---|---|
| K1−K3 | +0.541 [+0.532, +0.550] | +0.035 [+0.034, +0.036] |
| K3−K6 | +0.277 [+0.271, +0.284] | +0.0016 [+0.0011, +0.0020] |
| K6−K8 | +0.025 [+0.024, +0.026] | −0.0019 [−0.0022, −0.0017] |
| K8−K12 | +0.0154 [+0.0148, +0.0160] | |
| K12−K16 | +0.0045 [+0.0041, +0.0049] | |
| K16−K24 | −0.0040 [−0.0045, −0.0034] | |
| K6−K12 | +0.0405 [+0.0392, +0.0417] | |
| E6@16 − notul@6, same rows | **+0.104 [+0.101, +0.107]** (E6 worse) | |

Training: no detonation (`preclip/total` max 35.5 at step 226, HEALTHY by the sustained
tripwire). Run-log val at 5000: 4.1048 (last four 4.244, 4.038, 4.097, 4.142) against the
notul run's 4.1380 at 5000; those evals are on different batches and swing ±0.1 between
checkpoints, so the paired 480-row number above is the measurement. Throughput 1.34 sps
against notul's 1.65 (the 20k pair record), 1.23x the step cost.

`score_arc_e0.py` at (3, 12): Spearman(row CE_3, earning) −0.183 [−0.268, −0.088]; offset
profile 0.221 [0.207, 0.235] at offset 0, then 0.332, 0.320, 0.323, 0.335, 0.322, 0.317
(offset-0 / mean 0.68x); top-decile loss share 0.120, earning share 0.076. At (6, 16):
Spearman −0.191, offset-0 ratio 0.63x, profile flat from offset 1 on.

Scored:

- **P6a TRUE.** K3−K6 +0.277, CI above 0 (bar 0.01; notul 0.0016).
- **P6b TRUE.** K6−K12 +0.0405, CI above 0 (bar 0.005).
- **P6c FALSE.** K12−K16 +0.0045 [+0.0041, +0.0049] is above 0. The curve flattens between
  16 and 24 (K16−K24 −0.0040), at the training mean, not before it.
- **P6d TRUE.** E6 at its trained depth is 0.104 nats worse than notul at its own, on the
  same rows (bar 0.02).
- **P6e TRUE.** No detonation.
- **P6f TRUE by the letter** (bin 16–31 at 0.317 against offset 0 at 0.221, non-overlapping),
  but the profile is MORPH's shape, not Huginn's: the span's first token earns least and
  everything from offset 1 on is flat. Row difficulty still anti-correlates with earning.

## Verdict

Five of six held. The training depth draw IS the lever on the K-differences: trained at
mean 16, the same plain loop that earned 0.0016 nats from iteration 3 to 6 earns 0.277,
and keeps earning to 16. The Huginn reading "a loop earns to roughly half its training
mean" did not hold on MORPH: E6 saturates AT its mean (16–24), and P6c missed for that
reason.

The finding that matters more, and was not a prediction: **the K-differences measure
dependence on depth, not the value of depth.** E6 at depth 16 is 0.104 nats WORSE than the
mean-6 model at depth 6 on identical rows at identical steps, at 1.23x the step cost. Its
depth-1 CE is 0.93 nats worse than notul's. The deep draw moved the model's work INTO the
loop; it did not make the model do more work. Every "earning" number in this arc, and
Huginn's, is of this kind unless a matched-compute shallow control sits next to it. Huginn
has no such control.

Not verified: whether the 0.104 gap closes with training (deep models converge slower;
the 20k pair's A2 gap closed 0.132 → 0.012 from 5k to 20k); the 24-depth cell's sweep
runs the eval loop past the model's max, which the code allows and nothing audited;
the val-line discrepancy (4.105 against 4.138) is unexplained beyond batch noise.

## Updated hypothesis

Depth earning is a property of the training draw, and it is not free: at 5k the deep
model pays 0.104 nats and 1.23x compute for a loop that "earns" 0.86 nats over its own
shallow evaluation. The question the arc actually needs answered is matched-compute:
does a mean-16 loop at 20k beat a mean-6 loop at the same wall clock? That is E5 at the
deep draw (the binding rule as written). The block-loop (E7, running) asks the version
that could pay: a deep draw on 1/8 of the positions, with the tokens shallow.
