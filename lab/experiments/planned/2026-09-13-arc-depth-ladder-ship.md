# Planned: the depth ladder — what core depth ships

Status: planned

Date: 2026-09-13 (frozen before any GPU step of the three new arms; the depth-6 rung is
the existing `norm-match-20k` run). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).

## Question

Wolfe, 2026-09-13 evening: "At this point I would just be happy to have something that
reduces compute enough that I can ship. I am not married to TUL." The slot loop is closed
on eleven-plus arms. The plain looped model is the other place the compute goes: a core of
6 blocks run T times (Poisson mean 6, max 8) is 44 block-passes per token against 14 at
depth 1, and the depth-1 model trains at 27,610 tok/s against 9,901 for the looped one
(`arc/plain-depth1/run.log`, `arc/norm-match-20k/run.log`, step-200 lines). What does the
loop buy at 20,000 steps, at matched steps AND at matched wall clock, and what is the
cheapest fixed depth whose 20k CE holds the looped model's?

## What is already known

- At 5,000 steps the depth-1-TRAINED plain model sits 0.0038 nats BEHIND the Parcae-entry
  looped model at depth 6, token-paired
  ([`failures/2026-09-09-arc-e20-loop-depth-candidates.md`](../failures/2026-09-09-arc-e20-loop-depth-candidates.md),
  P20d). That is a 5k reading and deep models converge slower; it is NOT a verdict.
- The looped model's own forced-depth sweep at 20k reads CE 3.6217 / 3.4990 / 3.4672 /
  3.4516 at eval depths 1 / 2 / 3 / 6 (`results/2026-09-09-norm-match-recipe-reads/
  sweep_norm-match-20k_20000.json`), K1−K6 0.1702 and rising with steps (0.136 at 5k).
  Those are EVAL depths of a depth-6-trained model; training depth moves CE and eval depth
  does not ([`2026-09-12` register/spandec twin reading]), so a depth-d-TRAINED rung will
  sit well under the depth-6 model's forced-depth-d number.

## Method

Three new arms, one factor (training depth) against the existing top rung. All on the
`notul_norm_match_20k` lineage: Parcae noise entry, `norm_match` ternary, seq 1024, batch 6,
seed 1, 1,000-step ramp, 20,000 steps, checkpoints every 5,000, retention off,
prune/carve/route off, `core_fixed_point_lambda` 0.0 (the lineage's value on every rung;
depth 1 has no terminal pair anyway).

| arm | config | mean = max = bptt depth | block-passes / token |
| --- | --- | --- | --- |
| `norm-match-20k-d1` | `notul_norm_match_20k_d1.yaml` | 1 | 14 |
| `norm-match-20k-d2` | `notul_norm_match_20k_d2.yaml` | 2 | 20 |
| `norm-match-20k-d3` | `notul_norm_match_20k_d3.yaml` | 3 | 26 |
| `norm-match-20k` (exists) | `notul_norm_match_20k.yaml` | Poisson mean 6, max 8 | ~44 |

Each checkpoint (5k, 10k, 15k, 20k) gets `core_depth_sweep.py --depths 1,2,3,6,9,12,16
--rows 480` (the runner's plain readout). Between-arm CE is token-paired on the 480 rows
(`span_budget_profile.py --full A.npz --span B.npz`, gap = span − full). Wall clock per
arm is the run log's epoch stamps, and the step-200 tok/s is the rate figure.

Matched WALL CLOCK is read by bracketing: the depth-6 run's 5k checkpoint costs about 103
min and its 10k about 207 min at 9,901 tok/s; the depth-1 run's 20k costs about 74 min at
27,610 tok/s. So "d1 at 20k vs d6 at matched wall clock" is d1@20k against d6@5k (d6 has
had MORE time), and the honest comparison is that d1@20k must beat d6@5k to be a
wall-clock win at all, and beat d6@10k to be a clear one. The same bracketing applies to
d2 and d3 with their measured rates.


### Method amendment (2026-09-14)

"Fixed depth d" on this ladder means `mean_depth = max_depth = d`, which the tree draws
as Poisson(d) clamped to [1, d]: d1 is truly fixed, d2 and d3 train with a share of rows
at lower depth (Poisson(3) puts about 20 % of rows at 1 and 22 % at 2 before the clamp).
The predictions were written under that construction and are scored under it. A truly
fixed depth-3 rung (`model.depth_fixed`, `notul_norm_match_20k_d3fixed`) is queued as the
LoopMTP control and is read in the LoopMTP prereg, not here.

## Predictions (frozen)

- **P-1 (matched steps, the loop's edge at 20k).** CE(d1@20k) − CE(d6@20k), token-paired,
  is in **[+0.02, +0.08]**. **55 %.** Reasoning: 0.004 at 5k, the loop's own K1−K6 grows
  0.136 → 0.170 from 5k to 20k, and a depth-1-trained model closes most but not all of a
  forced-depth gap. Residual: 25 % under 0.02 (the loop buys nothing at 20k either), 20 %
  over 0.08.
- **P-2 (matched wall clock).** CE(d1@20k) < CE(d6@10k) on the paired rows, i.e. the
  depth-1 model at 74 min beats the looped model at 207 min. **80 %.** Reasoning: d6@10k
  reads 3.69 at its own depth; d1 at 20k has seen 2x the tokens of d6@10k and P-1 says the
  loop's edge at equal steps is under 0.08 while the 10k→20k step gain is 0.24. Residual:
  20 %.
- **P-3 (the cheapest rung that holds).** d3@20k is within **0.02** of d6@20k,
  token-paired. **60 %.** d2@20k within 0.02: **40 %.** Reasoning: the depth-6 model's own
  forced-depth-3 sits 0.016 behind its depth 6, and a depth-3-TRAINED model does at least as
  well as that.
- **P-4 (the rungs use their depth).** The d2 and d3 rungs' own forced-depth sweeps read
  K1−Kd above **0.05** at 20k. **70 %.** Residual: a fixed-depth-trained model that ignores
  its own loop, which would say the loop earns only as a Poisson draw.
- **P-5 (rates).** Step-200 tok/s: d2 ≥ **17,000**, d3 ≥ **13,000**. **75 %.** Arithmetic
  from the block-pass counts against the measured 27,610 (14) and 9,901 (44).
- **P-6 (it fits).** All three run 20,000 steps on the 5090 at batch 6 under 12 GB peak.
  **90 %.** The d1 run peaked at 9.04 GB, d6 at 10.08.


### Sampled rungs (added 2026-09-14 after Wolfe's correction; frozen before their launch)

**The correction.** The d1/d2/d3 rungs (Poisson(d) clamped at d) and the `depth_fixed`
rungs are NOT the recipe's loop: a fixed-depth model learns a d-deep net with tied
weights, is out of distribution at every other depth (d3fixed at 20k: 3.71 / 3.55 / 3.44 /
3.48 at 1 / 2 / 3 / 6) and has no depth axis, while the sampled model is an attractor with a
usable curve at every depth (3.62 / 3.50 / 3.47 / 3.45) and a K1−K6 of 0.170 that still
grows with steps. d3fixed's −0.007 at depth 3 against the top rung at 20k is therefore
not a ship reading, and the fixed rungs stay in this file as controls only. The compute
question moves INSIDE the sampled regime.

**Arms.** `norm-match-20k-p3` (Poisson mean 3, max 6, BPTT 6) and `norm-match-20k-p4`
(mean 4, max 8, BPTT 8), the top rung's recipe with a cheaper draw; expected block passes
per token ~26 and ~32 against the top rung's ~44. Paired against `norm-match-20k` on its
existing sweeps.

**Predictions (frozen).**

- **S-1 (the curve survives).** `p3`'s own K1−K6 at 20k ≥ 0.10 (the top rung reads 0.170;
  a clamped-3 rung cannot read past 3): 60 %. `p4` ≥ 0.13: 60 %.
- **S-2 (extra eval depth still pays).** `p3`'s own K3−K6 at 20k > 0.005 (the model uses
  passes beyond its mean at eval; the clamped d3 rung reads −0.003): 55 %.
- **S-3 (paired CE at the mean depth).** `p3` at depth 3 minus the top rung at depth 6, 20k,
  480 rows: in [+0.010, +0.030]: 50 %; `p4` at depth 4: in [+0.005, +0.020]: 50 %.
- **S-4 (paired CE at depth 6).** `p3` evaluated at depth 6 minus the top rung at depth 6:
  ≤ +0.015: 50 % (it trained at depth 6 on ~8 % of rows).
- **S-5 (rate).** Step-200 tok/s: `p3` ≥ 16,000, `p4` ≥ 13,000: 65 % each.
- **S-6 (survival).** Both HEALTHY to 20k: 90 %.

**Binding.** If S-1 and S-2 hold, a lower-mean draw keeps the loop and its anytime axis
at fewer passes, and the recipe's `mean_depth` is the compute dial; the next rung is a
20k-vs-40k horizon check on `p3`. If S-1 fails, the draw's mean is load-bearing for the
curve and the cost of the loop is the cost of the mean.

## Binding

The ship depth is the smallest d whose 20k CE is within 0.02 of d6's (P-3) AND whose wall
clock is at most half of d6's. If P-2 holds, the loop is not a training-compute win at
this horizon, and at inference depth 1 is 3.1x fewer core block-passes by construction; the
deploy recipe then runs depth 1 (or the P-3 rung) with the ternary + MORTAR + route path,
whose conjunction is still unmeasured and becomes the next run. If P-2 FAILS, the looped
model wins per GPU-hour and the loop stays; the compute question moves to inference only
(a fixed shallower eval depth on a depth-6-trained model, whose price the existing sweep
already gives: 0.047 at depth 2, 0.016 at depth 3).

## Not verified before launch

- No GPU step of d2 or d3 exists; their rates are arithmetic.
- The depth-1 rate (27,610) is from `plain-depth1`, the E18 lineage, not this lineage; the
  two differ in the injection block, which is a small cost.
- 20k is one horizon. Nothing here says the ordering holds at 100k or at seq 4096.
- The three rungs share seed 1; MORPH runs decorrelate in ~11 steps and n=1 comparisons
  carry a 6.5 % median spread (memory `morph-n1-run-comparisons-unreadable`); the token
  pairing removes row noise, not seed noise. A 0.02 threshold is inside that spread's
  reach; the wall-clock prediction (P-2) is not.

## Results

(to be filled after the runs; predictions above are frozen)
