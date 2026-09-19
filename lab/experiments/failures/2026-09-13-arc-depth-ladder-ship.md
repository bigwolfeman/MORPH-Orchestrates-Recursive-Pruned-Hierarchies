# Planned: the depth ladder — what core depth ships

Status: failure

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

Scored 2026-09-19. Every rung trained its 20,000 steps and every checkpoint was swept
with `core_depth_sweep.py --depths 0,1,2,3,6,9,12,16 --rows 480`. All ten sweeps read
here carry the same 491,520 scored positions (480 rows x 1,024) in the same order:
`tok_index` is bit-identical across d1, d2, d3, d3fixed, d3fixed-s2, p3, p4 and the top
rung at 5k/10k/15k/20k, checked before every pair. Paired deltas and their 95 %
intervals come from `lab/divergence/ladder_score.py` (row bootstrap over the 480 rows,
2,000 draws, seed 0; tests in `tests/test_ladder_score.py`). Inputs and outputs:
[`results/2026-09-13-arc-depth-ladder/`](../results/2026-09-13-arc-depth-ladder/).

### The ladder at 20,000 steps, token-paired against the top rung

Top rung `norm-match-20k` (Poisson mean 6, max 8) reads **3.4516** at eval depth 6.

| arm | draw | passes / token | read at depth | CE | minus top rung | 95 % |
| --- | --- | --- | --- | --- | --- | --- |
| `d1` | fixed 1 | 14 | 1 | 3.5189 | **+0.0674** | [+0.0647, +0.0701] |
| `d2` | Poisson(2) clamped 2 | 20 | 2 | 3.4749 | **+0.0233** | [+0.0209, +0.0257] |
| `d3` | Poisson(3) clamped 3 | 26 | 3 | 3.4610 | **+0.0095** | [+0.0070, +0.0118] |
| `p3` | Poisson(3) max 6 | 26 | 3 | 3.4502 | **−0.0013** | [−0.0036, +0.0010] |
| `p3` | " | " | 6 | 3.4496 | **−0.0019** | [−0.0042, +0.0003] |
| `p4` | Poisson(4) max 8 | 32 | 3 | 3.4503 | **−0.0013** | [−0.0038, +0.0012] |
| `p4` | " | " | 6 | 3.4464 | **−0.0051** | [−0.0076, −0.0027] |
| `d3fixed` | `depth_fixed` 3 | 26 | 3 | 3.4447 | −0.0069 | [−0.0094, −0.0044] |
| `d3fixed-s2` | the same, seed 2 | 26 | 3 | 3.4684 | +0.0168 | (derived, see below) |

**The resolution of this table is 0.024 nats, not 0.002.** `d3fixed` and `d3fixed-s2`
are the SAME config at two seeds, and at their own training depth they differ by
**+0.0237 [+0.0208, +0.0266]** on the identical rows. Read at depth 6 the same pair
differs by only +0.0042 [+0.0009, +0.0074], so the spread is largest exactly where this
ladder ranks its arms. The prereg said a 0.02 threshold was inside the n=1 spread's
reach ("Not verified before launch", last bullet); it is, and now it is measured. Every
row above whose magnitude is under 0.024 is therefore a tie, which covers d2, d3 and the
whole sampled family. Only d1's +0.0674 clears it.

### Each arm's own forced-depth curve at 20,000 steps

| arm | depth 1 | 2 | 3 | 6 | own K1−K6 | 95 % |
| --- | --- | --- | --- | --- | --- | --- |
| `d1` | 3.5189 | 3.5677 | 3.6193 | 3.6606 | −0.1417 | [−0.1452, −0.1382] |
| `d2` | 3.4877 | 3.4749 | 3.4766 | 3.4859 | +0.0018 | [+0.0007, +0.0028] |
| `d3` | 3.5030 | 3.4668 | 3.4610 | 3.4643 | +0.0387 | [+0.0373, +0.0402] |
| `p3` | 3.4974 | 3.4572 | 3.4502 | 3.4496 | +0.0478 | [+0.0465, +0.0492] |
| `p4` | 3.5169 | 3.4616 | 3.4503 | 3.4464 | +0.0705 | [+0.0689, +0.0721] |
| top rung | 3.6217 | 3.4990 | 3.4672 | 3.4516 | +0.1702 | [+0.1672, +0.1729] |
| `d3fixed` (control) | 3.7143 | 3.5526 | 3.4447 | 3.4849 | +0.2294 | [+0.2260, +0.2331] |
| `d6fixed` (control) | 3.9505 | 3.7078 | 3.5738 | 3.4246 | +0.5259 | [+0.5152, +0.5365] |

The K1−K6 column is monotone in the draw's mean over the sampled family: 1 → −0.142,
2 → +0.002, 3 → +0.039 (clamped) / +0.048 (Poisson tail), 4 → +0.071, 6 → +0.170. The
final CE is not. That is the ladder's one strong result and it is stated in the verdict.

### Rates, memory and wall clock

`tok/s` at step 200 from each `run.log`; `peak` is the largest `peak=` over the whole
run; queue wall is DONE minus START from `queue.log`; the integrated column sums
`200 / sps` over the log and is the contention-free estimate.

| arm | tok/s @200 | passes/tok | peak GB | queue wall | integrated | vs top rung |
| --- | --- | --- | --- | --- | --- | --- |
| `d1` | 25,931 | 14 | 9.15 | 80.2 min | 75.7 min | 0.37x |
| `d2` | 19,931 | 20 | 9.61 | 104.8 min | 98.7 min | 0.48x |
| `d3` | 18,062 | 26 | 9.71 | 118.3 min | 112.3 min | 0.54x |
| `p3` | 12,892 | 26 | 9.99 | 149.4 min | 142.6 min | 0.69x |
| `p4` | 12,802 | 32 | 9.99 | 285.5 min | 194.1 min | 0.94x |
| top rung | 9,901 | 44 | 10.17 | 216.0 min | 206.8 min | 1.00x |

`p4`'s queue wall is 1.47x its integrated step time: it ran under contention and its
285 min is not a clean measurement. Every other arm's ratio is 1.04 to 1.06.

**Expected block passes do not buy wall clock one for one under a Poisson draw.** `d3`
and `p3` both average 26 passes per token, and `d3` runs 1.40x faster (18,062 against
12,892 tok/s; 112 min against 143 min). The core loop shrinks its active set each
iteration (`_core_region`, sorted descending by drawn depth), so a Poisson(3)-max-6 batch
still launches up to 6 iterations, the last of which carry one or two rows and leave the
GPU mostly idle. A clamp makes every iteration full width. The measured saving from
mean 6 to mean 3 inside the sampled family is 31 % of wall clock, against the 41 % the
pass count promises.

### Wall-clock arithmetic: a correction to the Method

The Method's bracketing is off by a factor of two. At 6,144 tokens per step and
9,901 tok/s the top rung needs 52 min for 5,000 steps and 207 min for 20,000, not
"103 min for 5k and 207 min for 10k". The measured checkpoint times, integrated from
its own log and scaled to its 216 min queue wall, are 5k = 53.6 min, 10k = 107.1 min,
15k = 161.4 min. P-2 is scored exactly as written (CE(d1@20k) < CE(d6@10k)) and the
honest matched-clock comparison is added beside it: at d1's own 80 min the top rung has
reached roughly step 7,500.

### Scorecard

Clamped rungs, against the top rung at 20,000 steps:

| clause | bar | measured | |
| --- | --- | --- | --- |
| **P-1** CE(d1@20k) − CE(top@20k) | in [+0.02, +0.08] | **+0.0674** [+0.0647, +0.0701] | **HOLDS** |
| **P-2** CE(d1@20k) < CE(top@10k) | < 0 | **−0.1728** [−0.1777, −0.1682] | **HOLDS** |
| **P-3a** d3@20k within 0.02 of top | \|.\| ≤ 0.02 | **+0.0095** [+0.0070, +0.0118] | **HOLDS** |
| **P-3b** d2@20k within 0.02 of top | \|.\| ≤ 0.02 | **+0.0233** [+0.0209, +0.0257] | **FAILS** |
| **P-4a** d2 own K1−K2 | > 0.05 | **+0.0128** [+0.0122, +0.0134] | **FAILS** |
| **P-4b** d3 own K1−K3 | > 0.05 | **+0.0420** [+0.0408, +0.0432] | **FAILS** |
| **P-5a** d2 step-200 tok/s | ≥ 17,000 | **19,931** | **HOLDS** |
| **P-5b** d3 step-200 tok/s | ≥ 13,000 | **18,062** | **HOLDS** |
| **P-6** all three under 12 GB peak, 20k steps | < 12 GB | **9.15 / 9.61 / 9.71 GB**, all HEALTHY at 19,999 | **HOLDS** |

Sampled rungs:

| clause | bar | measured | |
| --- | --- | --- | --- |
| **S-1a** p3 own K1−K6 | ≥ 0.10 | **+0.0478** [+0.0465, +0.0492] | **FAILS** |
| **S-1b** p4 own K1−K6 | ≥ 0.13 | **+0.0705** [+0.0689, +0.0721] | **FAILS** |
| **S-2** p3 own K3−K6 | > 0.005 | **+0.0006** [+0.0002, +0.0010] | **FAILS** |
| **S-3a** p3@3 − top@6 | in [+0.010, +0.030] | **−0.0013** [−0.0036, +0.0010] | **FAILS** |
| **S-3b** p4@4 − top@6 | in [+0.005, +0.020] | depth 4 NEVER SWEPT; bracketed by p4@3 **−0.0013** and p4@6 **−0.0051**, and CE falls monotonically from depth 3 to 6, so depth 4 lies in [−0.0051, −0.0013] | **FAILS** |
| **S-4** p3@6 − top@6 | ≤ +0.015 | **−0.0019** [−0.0042, +0.0003] | **HOLDS** |
| **S-5a** p3 step-200 tok/s | ≥ 16,000 | **12,892** | **FAILS** |
| **S-5b** p4 step-200 tok/s | ≥ 13,000 | **12,802** | **FAILS** |
| **S-6** both HEALTHY to 20k | both | both `DONE exit=0 verdict=HEALTHY last=19999` | **HOLDS** |

Seven clauses hold, ten fail. **This record is a failure**, and the useful half of it is
that the failures are one-directional: every CE prediction that placed a cheaper rung
BEHIND the top rung overshot, and every depth-curve prediction that expected the cheaper
draw to keep its depth axis overshot too.

### The one clause the method could not measure

S-3b names eval depth 4 and the sweep grid is 0, 1, 2, 3, 6, 9, 12, 16. Depth 4 was
never run on any arm, so the clause cannot be read as written. It is scored FAILS rather
than left open because both bracketing columns sit below the band and the curve between
them is monotone, so no value at depth 4 could land inside [+0.005, +0.020]. Adding
depth 4 and 5 to the sweep grid costs two more forwards per checkpoint and should be the
default from here; the fix is one string in the runner's `--depths`.

## Verdict

**`p3` (Poisson mean 3, max 6, full BPTT) is the rung that ships.** At 20,000 steps it
is level with the mean-6 top rung on 480 identical rows, at its own mean depth
(−0.0013 [−0.0036, +0.0010]) and at depth 6 (−0.0019 [−0.0042, +0.0003]), for 26 of
44 expected block passes, 1.30x the training throughput and 69 % of the wall clock, and
it keeps a monotone anytime curve down to depth 1. `p4` is the same trade one rung up
(−0.0051 at depth 6, 94 % of the wall clock) and is the safer choice if the K-curve
matters, because the curve, not the CE, is what the mean buys. The clamped ladder
answers the prereg's own binding rule with **no rung**: `d3` meets the CE bar (+0.0095)
but costs 54 % of the top rung's wall clock against a bar of 50 %, and `d2` meets the
clock (48 %) but not the CE bar (+0.0233, interval entirely above 0.02). That rule is
moot anyway under the 2026-09-14 correction, which takes the clamped and fixed rungs out
of the ship set: a clamped rung has no depth axis worth the name (`d2` reads K1−K6
+0.0018 and its curve turns back up past its clamp) and a `depth_fixed` rung is a tied-weight
d-deep net, not this recipe's loop.

**The loop is not a training-compute win at this horizon and the ladder says so twice.**
d1 at 20,000 steps and 80 min beats the top rung at 10,000 steps and 107 min by 0.173
nats and still beats it at 15,000 steps and 161 min by 0.025, so P-2 holds with room to
spare; and inside the sampled family, cutting the mean from 6 to 3 costs nothing
measurable in CE while returning 31 % of the wall clock. What the loop buys, and the
only thing it measurably buys, is depth dependence: K1−K6 rises 0.002 → 0.039 → 0.048 →
0.071 → 0.170 as the mean goes 2 → 3 → 3 (Poisson) → 4 → 6, on models whose final CE
is flat to within the 0.024-nat seed spread. Paying 1.7x the compute to make the
model 0.17 nats worse at depth 1 is only worth it if something downstream spends that
axis, and nothing in this tree does yet.

## Updated hypothesis

1. **`mean_depth` is a dial on the depth axis, not on final CE.** Before this run the
   working assumption (P-3, S-3) was that a cheaper draw would cost CE roughly the way
   the top rung's own forced-depth curve suggested, 0.016 at depth 3. It costs nothing:
   every sampled rung from mean 3 up is within the seed spread of mean 6 at 20,000
   steps, while K1−K6 scales almost linearly with the mean. The two quantities came
   apart, and the K-curve is the one the mean controls.
2. **The clamp and the Poisson tail are different animals at equal expected passes.**
   `d3` and `p3` both average 26 passes, and `d3` is 1.40x faster in wall clock and
   0.011 nats worse in CE with a smaller K-curve. Expected FLOPs is the wrong cost
   model for a per-sample depth draw at batch 6; the tail iterations run on a nearly
   empty active set.
3. **0.02 nats is below this lineage's resolution at n=1.** The seed pair measures
   0.0237 at the arms' own training depth. Any future ladder bar under about 0.03 nats
   needs a second seed per rung, and the prereg's own last bullet predicted this.
4. **What did NOT change.** The top rung's K-curve is still the largest of any sampled
   arm and still grows with steps, so nothing here argues against the loop as an anytime
   mechanism. It argues that at 20k on web text the mechanism has no CE customer.

Next planned experiment, in the prereg's own words for the S-branch: a 20k-vs-40k
horizon check on `p3`, now with a second seed on the top rung and on `p3`, and with
depths 4 and 5 in the sweep grid. Until that runs, the horizon caveat below stands.

### Not verified by this scoring

- **One seed per rung.** The seed pair exists only for `d3fixed`. `p3`, `p4`, `d1`,
  `d2`, `d3` and the top rung are n=1, and the measured spread (0.024) swallows every
  CE difference in this record except d1's.
- **One horizon.** 20,000 steps, seq 1024, batch 6. Nothing here says the ordering
  holds at 100k, at seq 4096, or under the `base.yaml` conjunction (TUL x TST x
  prune/carve/route).
- **`p4`'s wall clock.** Its run shared the machine; only the integrated step time
  (194 min) is usable, and that is an estimate from the `sps` column, not a stopwatch.
- **Eval depths 4 and 5 were never run**, on any arm.
- **Generation quality.** Every number here is validation CE on 480 rows. No `rep4`,
  no `distinct3`, no downstream task. A rung that ties on CE can still be worse to
  sample from.
- **Inference cost.** The ship argument counts training wall clock. The deploy path
  (ternary + MORTAR + route) is still unmeasured on any of these rungs, as the prereg's
  Binding section already said.
