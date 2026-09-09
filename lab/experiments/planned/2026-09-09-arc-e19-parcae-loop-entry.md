# Planned: ARC E19 — Parcae's loop entry on the plain model: does the recurrence earn depth once the state must be built from the injected input?

Status: planned
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

- **loop** (`notul_e19_loop.yaml`): `core_state_init: noise` (std 0.02), `injection_channels:
  all` with `injection_all_decay 0.447` / `injection_all_dt 0.8` on every dim, `injection_B:
  true` (identity init), `core_fixed_point_lambda 0.0` (Parcae has no such term).
- **fp0** (`notul_e19_fp0.yaml`): the E18 plain recipe with only the fixed-point term off —
  the one-factor control for the loop arm's λ = 0.
- **d1** (`notul_e19_d1.yaml`): the E18 plain recipe trained at core depth 1 (mean = max =
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

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
