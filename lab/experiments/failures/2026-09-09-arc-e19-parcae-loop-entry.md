# Planned: ARC E19 — Parcae's loop entry on the plain model: does the recurrence earn depth once the state must be built from the injected input?

Status: failure
Date: 2026-09-09 (frozen before launch; Wolfe: "We misimplemented the looping in MORPH. It
has never been contributing."). Arc: `2026-09-04-loop-contribution-arc.md`, row E19.
Decision note: `.agents/notes/proposed/architecture/2026-09-09-depth-lobotomy-candidates.md`.

## Question

MORPH's plain core does one iteration of work (skipping it costs 0.317 nats) and then sits
at a fixed point: iterations 2–8 are worth 0.020 nats, the state moves 9.9 / 4.9 / 3.4 / 2.7
/ 2.3 / 2.1 / 2.0 % per iteration, the block branches move it 1–30 % (attention) and 4–13 %
(MLP), and from a zero or noise start the loop lands at 4.67 / 6.30 nats — the fixed point
depends on where the loop starts (E18 plain checkpoint, 2026-09-09; `results/2026-09-08-arc-e18/`).
Parcae-140m on the same corpus builds its state from near-zero noise (like-init, std 0.023
against a converged state RMS of 0.56), re-injects e on every one of its 768 dims through a
learned B (diagonal mean 2.4) each iteration, and each of its two recurrent blocks moves the
state 37–42 % per pass even at the fixed point; its per-iteration CE gains are 0.196 / 0.058 /
0.023 / 0.007 / 0.0015 (`~/parcae/outputs/parcae-140m-owt-10k/depth-010000.json`). MORPH's
entry is ``h_0 = e`` with the carry on 320 of 1024 dims: the identity loop is a free solution
and the coda reads the prelude whether or not the recurrence does anything. E19 gives the
plain model Parcae's entry and asks whether the recurrence then earns depth, and what the loop
is worth in TRAINING (the depth-1-trained control), which a forced-depth curve cannot say.

## Hypothesis

H19: with the state built from noise and e re-injected on every dim through B, the loop
becomes load-bearing: the forced-depth curve earns past iteration 3 and the per-iteration
state movement stays above 10 % at iteration 6, at an end CE within 0.05 nats of the E18
plain arm at 5,000 steps. H19′: the entry is not sufficient — the map stays weak (the
ternary / HC-gain candidate) and the curve is flat past 3 again. H19″ (the price): the plain
model trained at depth 1 is within 0.05 nats of the E18 plain arm at depth 6, i.e. MORPH's
loop has been worth almost nothing in training as well.

## Method

Three plain arms on the E18 recipe (`notul_e18`: seq 1024, batch 6, 5,000 steps, ramp 1000,
flat 1e-4, ternary backbone, AdEMAMix β1 = 0, Poisson core depth mean 6 / max 8, full BPTT):

- **loop** (`notul_parcae_entry.yaml`): `core_state_init: noise` (std 0.02), `injection_channels:
  all` with `injection_all_decay 0.447` / `injection_all_dt 0.8` on every dim, `injection_B:
  true` (identity init), `core_fixed_point_lambda 0.0` (Parcae has no such term).
- **fp0** (`notul_no_fixed_point.yaml`): the E18 plain recipe with only the fixed-point term off —
  the one-factor control for the loop arm's λ = 0.
- **d1** (`notul_depth1.yaml`): the E18 plain recipe trained at core depth 1 (mean = max =
  bptt = 1, λ = 0): the matched-tokens control that prices the loop.

Code: `morph/model/transformer.py` (`_NoiseInit`, `DiagonalInjection(use_B)`, the uniform
all-dim init; validation: noise requires "all"), `morph/training/train.py` (the config map),
`morph/configs/base.yaml` (the four keys with their defaults), tests in
`tests/test_injection_channels.py` (B identity reproduces the forward bit for bit; noise init
scale and seed determinism; the raises; the config map). Smokes 2026-09-09 02:10, 12 steps:
loop exit 0, peak 9.07 GB, 0 NaN, carry line "ALL 1024 dims (every dim decay 0.447, dt 0.8;
learned B)"; d1 exit 0, peak 7.95 GB.

Runner `arc/run_e19.sh` (commit pinned in `arc/E19_COMMIT`; worktree `/home/wolfe/morph-to`):
12-step smoke per arm, the sustained tripwire, `core_depth_sweep.py` at 2,500 and 5,000 (480
rows, depths 0,1,2,3,6,9,12,16, per-token files → `results/2026-09-09-arc-e19/`), then
`core_anatomy.py` (3 rows, depth 8) and `core_init_probe.py` (96 rows) on the 5,000
checkpoint. Order: loop, fp0, d1. Between-arm CE is token-paired (E18 amendment 1). Every
sweep's CE at the trained depth is checked against the trainer's `[VAL]` at the same step
before it is read (the E17 rule; the two are different samples of the validation stream, so
0.02 is sampling and 0.2 is a bug). The E18 plain arm (3.9804 at depth 6 on the same rows) is
the reference for every comparison.

**The Parcae side of the price** (Wolfe's call; needs the fork's power-cap launcher, which
uses sudo): `~/parcae/experiments/configs/owt_d1.yaml`, Parcae-140m at recurrence 1 for
5,000 steps; CE(parcae-140m-owt-10k @5000, depth 8 = 3.8685) − CE(this @1) prices Parcae's
loop at matched tokens. Not part of this file's predictions.

**Method amendment 1 (2026-09-09 02:36, loop arm at step ~250, nothing scored).** The plain
lineage inherits `training.grad_probe_every: 0`, so the first launch wrote no probe file and
the sustained tripwire was blind (E18's plain arm had the same gap, judged from its val
curve). The three E19 configs now set `grad_probe_every: 1` (0.5 % cost); the loop arm was
stopped at step ~250 and relaunched from step 0 at the re-pinned commit. Predictions untouched.

## Predictions (frozen)

- **P19a (survival).** Reaches 5,000 with the sustained tripwire silent: loop **70 %**
  (new dynamics under ternary + AdEMAMix; the noise start puts iteration 1 far from the fixed
  point every step), fp0 **90 %**, d1 **95 %**.
- **P19b (the loop arm earns depth).** At 5,000: K1−K6 > 0.15 nats: **65 %**. K3−K6 > 0.02
  with the CI above 0: **55 %**. K6−K12 within ±0.005 (converged by 6): **60 %**. CE at T=0
  (the coda reads the noise) worse than T=1 by more than 1.0 nats: **85 %**.
- **P19c (the loop arm's end point).** Token-paired CE at depth 6 vs the E18 plain arm at
  depth 6 (3.9804): within +0.05: **45 %**; better (below 0): **20 %**; more than 0.15 behind:
  **25 %**.
- **P19d (fp0 alone).** K3−K6 > 0.005 with the CI above 0: **25 %**. CE at depth 6 within
  0.02 of the E18 plain arm, token-paired: **70 %**.
- **P19e (the price of the loop).** CE(E18 plain @6) − CE(d1 @1), token-paired: > 0.05:
  **60 %**; > 0.15: **25 %**; < 0.02 (H19″, the loop was free in training too): **30 %**.
  CE(loop @6) − CE(d1 @1) < 0 (the rebuilt loop beats the depth-1 model): **50 %**.
- **P19f (anatomy).** Loop arm at 5,000, depth 8: consecutive state movement at iteration 6
  above 10 %: **60 %**; at least one core block with attention or MLP out/in above 0.2 at
  iteration 6: **55 %**; B's diagonal mean moves from 1.0 by more than 0.3: **60 %**.
  Init probe: from zero the loop arm reaches within 0.1 nats of its prelude-init CE by T=8:
  n/a (the loop arm HAS no prelude init) — instead: its CE from `noise_small` at T=8 equals
  its trained-init CE within 0.01 (they are the same init): **90 %** (a sanity check).
- **P19g (cost).** Loop arm ≤ 1.3 h: **70 %**; peak allocated under 16 GB: **80 %**.

## Binding

- P19b's first two clauses TRUE and P19c's first clause TRUE ⇒ the entry was the lobotomy:
  the next change ports the entry to `base.yaml` and to the paid TUL loop, and the arc's
  standing instruments become CE(T=∞) − CE(T=1) plus the depth-1-trained price.
- P19b TRUE and P19c's third clause TRUE (earns depth but 0.15+ behind) ⇒ the entry is
  right and the recipe is not: a longer run or the dense-core / HC-gain arms come next, with
  the loop arm as the base.
- P19b FALSE (K3−K6 < 0.01) ⇒ the entry is not sufficient; the dense-core and HC-gain arms
  (candidate 1) run next on the loop arm's config.
- P19e's third clause TRUE ⇒ the loop has been free in training too; the price becomes the
  headline number in the record.
- P19a FALSE on the loop arm ⇒ its trip step, gain and `loss/fixed_point`-free instruments go
  into the divergence README's second-hold table; the arm re-runs once at std 0.1.
- NO 20k run from this experiment.

## Not verified before launch

The 5,000-step path of the loop arm (only a 12-step smoke ran); whether `torch.compile` and
the fused kernels take the noise init and B without a recompile storm (the smoke ran eager
kernels off, kernels on for the run); the interaction of the noise init with the retention
state (retention is off in this recipe); the sweep's plain path at depth 0 on a
noise-init model (the coda reads the raw noise); the Parcae depth-1 control (not launched).

## Results

**Horizon caveat (Wolfe, 2026-09-09 10:20).** 5,000 steps is too short to rank a looped model against a depth-1 model, or one density against another, on final loss: deeper and sparser models converge slower and the break-even sits at a longer horizon. The CE differences below are reported as what they are, a matched-step reading at 5,000, and are NOT verdicts on looping or on ternary. The finding of this panel is the LOOP CONTRIBUTION: the K-curve, the branch ratios and the state movement.

Run 2026-09-09 02:33–05:19 at `3ed1ea2` (Method amendment 1: grad probe on). Runs renamed
at filing to `parcae-entry` (the loop arm), `plain-no-fixed-point` (fp0) and `plain-depth1`
(d1); the queue lines, wandb runs and every artifact carry those names. Scored by
`../results/2026-09-09-arc-e19/score_e19.py` (`score_e19.txt` is its output); token files
and probes under `ignored/experiment-artifacts/2026-09-09-arc-e19/`. Every sweep's CE at the
trained depth sits 0.023–0.030 under the trainer's held-out loss at the same step, the
same two-cut gap the E18 plain arm shows (0.026), so the E17 rule passes for all three arms.

**P19a (survival): TRUE ×3.** parcae-entry HEALTHY (probe peak 129 at step 226, then
under 10), plain-no-fixed-point HEALTHY (37 at 245), plain-depth1 HEALTHY (48 at 2936).

**P19b (the loop arm earns depth): FALSE on both depth clauses.** parcae-entry at 5,000:
K1−K6 +0.033 [+0.032, +0.034] (predicted > 0.15); K3−K6 +0.0009 [+0.0007, +0.0011]
(predicted > 0.02); K6−K12 −0.0001 (converged, TRUE); CE(T=0) − CE(T=1) = +4.84 (TRUE: the
coda cannot read the noise). The K-curve: 8.83 / 3.9905 / 3.9621 / 3.9582 / 3.9573 /
3.9575 / 3.9574 / 3.9575 at T = 0, 1, 2, 3, 6, 9, 12, 16. Flat to the fourth decimal from
depth 6 to 16, where the E18 plain arm drifts 0.014 worse.

**P19c (end point): TRUE, better.** parcae-entry@6 − E18 plain@6 = −0.0230 [−0.0255,
−0.0205], token-paired over 491,520 tokens (481 blocks). The trainer's own batches agree:
the loop arm trails by 0.008–0.061 through step 2,000 and leads by 0.017–0.044 from 2,500
on, final 3.9807 vs 4.0060.

**P19d (the fixed-point term alone): FALSE on depth, TRUE on CE.** plain-no-fixed-point
K3−K6 +0.0009 [+0.0005, +0.0013]; @6 − E18 plain@6 = −0.0004 [−0.0033, +0.0024]. Its
K1−K6 is +0.037 against the plain arm's +0.020 and its curve drifts −0.011 from 6 to 12: the
term costs no CE, halves the first pass's share and flattens the tail, and nothing more.

**P19e (the price of the loop at this horizon): H19″ TRUE (within 0.02).** Token-paired at 5,000:
E18 plain@6 − plain-depth1@1 = **+0.0192 [+0.0168, +0.0213]** (predicted < 0.02 at 30 %);
plain-no-fixed-point@6 − depth1@1 = +0.0188 [+0.0159, +0.0215]; parcae-entry@6 −
depth1@1 = **−0.0038 [−0.0066, −0.0013]** (the rebuilt loop ahead of the depth-1 model at 5,000,
predicted 50 %). The depth-1 model trained in 0.33 h against 1.00 h for the loop arms and
0.88 h for the plain ones; its held-out loss 3.9887 beats the plain looped arm's 4.0060.
Run deeper than trained it diverges (4.96 at depth 6, 6.61 at 16).

**P19f (anatomy): 1 of 3 TRUE; the sanity check TRUE.** Consecutive state movement
0.375 / 0.231 / 0.072 / 0.049 / 0.027 / 0.021 / 0.018 (iteration 6: 2.1 %, predicted
> 10 %: FALSE; Parcae's checkpoint reads 0.567 / 0.230 / 0.118 / 0.071 / 0.045 / 0.030).
Block out/in at iteration 6: attention 0.079–0.207, MLP 0.185–0.236 (> 0.2: TRUE; the E18
plain arm read 0.012–0.18 and ~0.10). B's diagonal mean 1.035 (> 0.3 from 1.0: FALSE);
decay 0.4436 (init 0.447), dt 0.802 (init 0.8): the carry did not train. Init probe on
the loop arm: prelude / RMS-noise / zero / small-noise starts all read 3.896 by T = 4 (the
fixed point is start-independent; the ordinary entry's zero start reads 5.50 at T = 6 and
the depth-1 model's 6.02). small-noise@T6 3.8960 vs the trained init 3.8959 (TRUE).

**P19g (cost): TRUE.** 1.00 h, 10.17 GB peak.

## Verdict

**failure** (the central prediction failed; the finding is decisive, not inconclusive).
The entry is NOT the lobotomy. Rebuilt the way Parcae builds it — state from noise, e
re-injected on every dimension through a learned B, no fixed-point term — the loop still
converges by pass 3 and earns 0.033 nats past pass 1 against Parcae's 0.29. What the entry
DID change: a better model (−0.023 nats at the same wall clock), a start-independent
fixed point, Parcae-shaped state movement, and attention in core block 0 awake (0.21 from
0.016). What it did not change: the carry never trains (B, decay, dt at init), and the
readout gains nothing from the state's continued movement after pass 2.

P19e at this horizon: the model trained at depth 1 sits within 0.02 nats of the looped
plain model at depth 6 (0.019 ahead of it, token-paired) and within 0.004 of the rebuilt loop
arm, at a third of the wall clock. That is a 5,000-step reading, not a verdict on looping:
a loop is a deep model and converges slower, and where the break-even lies is a longer-horizon
question this panel did not ask. What the reading DOES say is that at 5,000 steps the loop's
passes past the first carry 0.03 nats of the loss (the K-curve), and the depth-1 control is the
instrument that separates convergence time from computation on any horizon.

Binding applied: P19b FALSE ⇒ the dense-core and HC-gain arms were next; the HC-gain arm
was dropped at review (the hyper-connection residual is a plain residual at init and its
mixing map trained ~4× from init), and the depth-candidates panel
(`2026-09-09-arc-e20-loop-depth-candidates.md`) runs dense-core, a 20× carry learning
rate and Parcae's depth schedule on the loop arm's config, each scored against
plain-depth1. P19e's third clause TRUE ⇒ the price is reported above, with the horizon caveat.

## Updated hypothesis

The loop's flatness is not in its entry, its start, its fixed-point term or its stability.
Three candidates remain and the depth-candidates panel separates them: the training
schedule never asks for depth (full BPTT over a mean-6 draw trains the state to be readable
at pass 1; Parcae truncates backprop to the last 4 of mean 8), the ternary per-pass map is
too coarse to refine past pass 2, or the update never reaches the carry at lr 1e-4. Whatever
the panel says, the standing instruments from here are CE(T=∞) − CE(T=1) AND the price
against a depth-1-trained model at matched steps and matched wall clock, read as a
loop-contribution instrument at the horizon run, never as a ranking of looped against
unlooped models; a K-curve alone is not evidence of computation. The Parcae comparison needs the same control on Parcae
(`~/parcae/experiments/configs/owt_d1.yaml`, Wolfe's launch); until it runs, Parcae's 0.29
is depth DEPENDENCE, not a measured value of depth.
