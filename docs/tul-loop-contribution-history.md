# TUL loop contribution: the history

2026-09-23. Written by two agents from the committed record: hist-A wrote Part A
(2026-08-16 to 2026-09-12, through the strict-geometry panel) and the glossary; hist-B
wrote Part B (2026-09-12 to 2026-09-23). hist-A merged the two. Sources: every filed
experiment under [lab/experiments/](../lab/experiments/), the Agent Notes under [.agents/notes/](../.agents/notes/), the
campaign files under [lab/divergence/](../lab/divergence/), [lab/toy_slot_loop/WRITEUP.md](../lab/toy_slot_loop/WRITEUP.md),
[docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md), and, for the GK readings that are not filed
yet, the vlt thread `lxtul-fan4` and the probe JSONs in
`/home/wolfe/morph-scratch/arc/results/2026-09-23-lxtul-gk/` (outside the repo).

How to read this file:

- Links go to the file that holds the number. "ln" is the line in that file. A link to a
  filed experiment shows its path without the `lab/experiments/` prefix.
- A positive number means the loop helps, unless the row says otherwise.
- Status words: **stands** (no later file changes it), **qualified** (the number is
  real, but a named file shows it is not loop work of the kind claimed), **retracted**
  (a named file shows it is an artefact).
- Every number was copied from a markdown file or JSON that one of the two authors
  opened. Nothing was recomputed. Section 10 lists what we could not check.

## Contents

1. The short version
2. Era overview
3. Glossary
4. Part A eras (A1 to A8)
5. Part B eras (B1 to B9)
6. Positive readings, by date
7. Retractions and qualifications
8. Docs that still state a retracted number
9. Lessons that carried forward
10. What could not be verified

## 1. The short version

**What "loop contribution" means here.** MORPH loops a shared core T times (Poisson
mean 6). Loop contribution is what passes 2 to T add to the prediction of TOKENS. The
main instrument is the token K-curve: force the loop to run d passes, read token CE, and
subtract (K1−K6 = CE at depth 1 minus CE at depth 6). A K-curve measures depth
DEPENDENCE. To call it VALUE you need a control: a depth-1-trained twin or a ruler arm,
compared by paired CE. The TUL slot loop is "think once, decode cheap": only the slot
positions run the core, and the coda reads the slot's exit state. That is the design this
campaign tried to make earn.

**What ever earned on tokens, and under which conditions.**

| reading | number | conditions | source |
|---|---|---|---|
| Plain loop, first honest earning | K1−K6 0.220 at 4500 | leak fixed, GLA off, spectral cap off, no LR ramp, absmean | [successes/2026-08-31-loop-killer-bisect.md](../lab/experiments/successes/2026-08-31-loop-killer-bisect.md) ln 112-113 |
| Plain loop, norm_match | K1−K6 +0.1849 [+0.1816, +0.1883] at 5k | norm_match ternary rule, NOISE entry | [successes/2026-09-09-arc-per-pass-strength.md](../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md) ln 112 |
| Same rule, prelude entry | K1−K6 +0.033 at 5k | the entry the slot loop uses | [failures/2026-09-09-arc-slot-loop-norm-match.md](../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md) ln 147 |
| Plain loop over 20k | 0.1363 (5k) to 0.1702 (20k) | norm-match-20k, noise entry | [failures/2026-09-09-arc-norm-match-recipe-reads.md](../lab/experiments/failures/2026-09-09-arc-norm-match-recipe-reads.md) ln 141-144 |
| Plain loop VALUE, vs its depth-1 twin | looped better by 0.0674 [0.0647, 0.0701] at 20k | matched steps; at 5k (absmean, E19) the depth-1 twin was 0.0192 AHEAD | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) ln 168; [failures/2026-09-09-arc-e19-parcae-loop-entry.md](../lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md) ln 158 |
| Paid loop (tokens and slots through the core) | K1−K6 +0.1685 at 5k; 0.104 at 20k with the LR ramp | causal (future-leak probe clean); 0.022 behind plain on CE at 20k; rejected as "not TUL" | [successes/2026-09-01-a2-paid-loop.md](../lab/experiments/successes/2026-09-01-a2-paid-loop.md) ln 104; [failures/2026-09-02-warmup-20k-pair.md](../lab/experiments/failures/2026-09-02-warmup-20k-pair.md) ln 125-127 |
| Best slot-loop reading on tokens | K1−K6 +0.0261 [+0.0251, +0.0273] | LXTUL-R Step 1b, a forced relay; 0.0265 worse on CE at depth 6 than its partner | [failures/2026-09-22-lxtul-r-step1b.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md) ln 92, 96 |
| Slot loop, strict relay | K1−K6 +0.0163 [+0.0153, +0.0173] | prev-reach1: the coda reads only the previous cell | [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 289 |
| Slot loop, yardstick | about +0.002; twelve arms in [−0.0001, +0.0033] | every design without a forced relay | [.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md) |

Under absmean, three mask arms read more: E4 +0.0209, E13 +0.0489, E18 k8 +0.0486. That
dependence went away under norm_match: +0.0009
([failures/2026-09-10-arc-slot-mux-mask-norm-match.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md) ln 219). The slot
positives stay at about 0.026 or less because every slot design gave the coda a
shallower route than the passes. The same six core blocks earn +0.0102 when run over
tokens and +0.0005 when run over slot states
([failures/2026-09-13-arc-loop-reads-tokens.md](../lab/experiments/failures/2026-09-13-arc-loop-reads-tokens.md);
[failures/2026-09-12-arc-core-token-and-critic.md](../lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md)). The plain loop's
passes act on the token states the loss is charged on. No slot design gave its passes
that.

**The failure modes that came back again and again.**

1. **The coda ignores the loop.** The coda reads the cell and not the passes: worth(zero)
   +0.811 at offset 0 while token K1−K6 reads +0.0006
   ([failures/2026-09-09-arc-coda-reads-the-thought.md](../lab/experiments/failures/2026-09-09-arc-coda-reads-the-thought.md) ln 115-117). In
   LXTUL-G the coda reads the posterior's cell, and a prior cell is worse than an empty
   one ([failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md)).
2. **One pass does the work.** The cell gains 0.1074 from entry to exit on
   slot-unpack-free, and passes 2 to 6 add at most 0.002
   ([results/2026-09-10-slot-geometry-audit/README.md](../lab/experiments/results/2026-09-10-slot-geometry-audit/README.md) ln 311-313). gradpass
   takes 0.366 of its 0.385-nat fall in pass 1
   ([failures/2026-09-10-arc-slot-mnext-gradpass.md](../lab/experiments/failures/2026-09-10-arc-slot-mnext-gradpass.md) ln 296). Every
   per-pass target tried (staged, oracle, per-pass plan, critic, LoopMTP, progressive,
   graded) was met by pass 1.
3. **Teacher forcing and the exposure gap.** In the denoise arm, passes 2 to 6 learned
   to denoise an entry they were given; pass 1, which starts from noise, never learned
   ([failures/2026-09-21-lxtul-loop-denoise.md](../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md)). LXTUL-G's exposure gap
   grew from 0.7108 at 2500 to 0.9840 at 5000
   ([failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) ln 148).
4. **Leaks that looked like depth.** l2cap's K1−K6 +0.2328 read −1.1198 with the carry
   off ([successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md) ln 81-82). LXTUL-R
   Step 1 relayed through the conv and value shift, which an attention mask does not
   bound ([failures/2026-09-22-lxtul-r-step1.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1.md)). A frozen coda's depth
   curve rewards a bland cell: arm A +0.0323, denoise +0.5575
   ([.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md](../.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md)).
   Readings taken before a detonation (E2, E7, E14, X1) looked like earnings.

**Where we are on 2026-09-23.**

- **LXTUL-G failed on the exposure gap.** Gap 0.9840 at 5000; ce_post 4.0270 against
  ce_prior@1 5.0110; K1−K6 −0.0050 [−0.0083, −0.0018]
  ([failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) ln 148). Two leads survive as
  qualified: width gain 0.159 at 4 samples (ln 205) and depth 6 better than depth 1 by
  0.0123 under the 4-sample read (ln 207), both on trajectories the model was not
  trained to produce.
- **GK1 matched the ruler, and its noise collapsed.** Final val 4.4238 against the
  ruler's 4.4249 (vlt `lxtul-fan4` entry 139, not filed). The prior noise scale fell from
  0.1 to 0.00034 of the state; eval width gain at 4 samples is 0.00007
  (`lxtul_g_probe_lxtul-gk1_5000.json`: `sigma_ratio_prior` 0.000339, `width_gain@4`
  6.84e-5).
- **The first gk4 run is confounded.** Its K rollouts drew independent dropout masks, so
  the bound could earn "width" from dropout. Training width gain 0.0065 ([failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) ln 116-117; vlt entry 140).
  At eval, with dropout off, the width gain is 0.00006 and the noise scale 0.00029
  (`lxtul_g_probe_lxtul-gk4_5000.json`: `width_gain@4` 6.22e-5, `sigma_ratio_prior`
  0.000290). Fixed in 7d44ed7 (shared dropout masks across rollouts); prereg amendment
  f53e988 ([failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) ln 113-124).
- **GK filed as a failure (2026-09-23 21:24, [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)).** On lxtul-gk4-shared the exposure
  gap is gone (ce_iw@4 0.7533 better than lxtul-g's ce_prior@1), but the noise switched
  itself off (sigma/r 0.00026), width gain is 0.00005, and K1−K6 is +0.0036, at the top
  edge of the slot-loop floor. It sits 0.0033 to 0.0044 ahead of the ruler at depth 6,
  inside the ~0.004 seed floor. Learned-Gaussian loop arms are closed.

## 2. Era overview

The "best reading" column gives the strongest loop-contribution number of the era and
its source. The era sections below carry every other number.

| era | dates | what was tried | best loop-contribution reading | verdict |
|---|---|---|---|---|
| A1 | 08-16 to 08-24 | first slot-loop arms (A0/A1), span-length gate, the takeover campaign (about 20 hypotheses) | A1c beats dense A0c by 0.0562 val CE at 20k, pre-fix ([lab/tul/arms-result.md](../lab/tul/arms-result.md) ln 51) | the takeover is positional; no cure; every CE is on a leak-reliant model |
| A2 | 08-25 to 08-28 | region Shapley, MUX head, Thought Gestalt restriction, warmup/SIGReg/NTP dropout, FM planner | tul_v1a2b loop worth up to +0.0107, post-hoc, pre-fix ([failures/2026-08-27-warmup-sigreg-ntpdrop.md](../lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md) ln 268) | "the plan is empty": the core works only on its own one-token target |
| A3 | 08-29 to 08-31 | loop ladder, l2cap, carry-leak audit, loop-killer bisect | l2cap +0.2328 RETRACTED; honest plain BG0C0 0.220, slot loop 0.015 ([successes/2026-08-31-loop-killer-bisect.md](../lab/experiments/successes/2026-08-31-loop-killer-bisect.md) ln 112; [successes/2026-08-31-tul-vs-notul-20k.md](../lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md) ln 83) | the l2cap earning was the leak; the plain loop earns once GLA and the cap are off |
| A4 | 09-01 to 09-03 | write-side seeds, paid loop A2, LR ramp, 20k pair, think-once panel | A2 +0.1685 at 5k, 0.104 at 20k ([successes/2026-09-01-a2-paid-loop.md](../lab/experiments/successes/2026-09-01-a2-paid-loop.md) ln 104) | earning follows payment; paid loop shipped, then rejected as "not TUL" |
| A5 | 09-04 to 09-08 | slot-loop gain constraint; arc E0 to E18; Huginn as an external ruler | E13 mask +0.0489, absmean, later gone ([failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 145) | stable and empty; stability and contribution are separate axes |
| A6 | 09-09 to 09-10 | Parcae entry, bf16 core, per-pass strength, norm_match, entry panel | plain +0.1849 (noise entry), 0.1702 at 20k ([successes/2026-09-09-arc-per-pass-strength.md](../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md) ln 112) | the ternary rule limited the plain loop; the slot loop is unchanged |
| A7 | 09-10 to 09-11 | geometry audit, gradient probe, z-optimize, toy study, staged and other M-next levers | staged forecast +0.0670, tokens +0.0020 ([successes/2026-09-10-arc-slot-mnext-staged.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md) ln 212) | the core is not muted; the problem is credit assignment; no token value |
| A8 | 09-11 to 09-12 | span budget, span decoder, strict geometry panel | prev-reach1 +0.0163 ([failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 289) | budget 0.40 nats; depth use comes from reachability; a relay, not a design |
| B1 | 09-12 | energies, per-pass targets, core-token training, critic, depth-1 twin, math | Olympiad slot +0.0111 ([failures/2026-09-12-arc-math-under-norm-match.md](../lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md)), math later set aside | each lane closed: the loss is not steep in the slot state |
| B2 | 09-13 | Thought Register, trajectory prefix, reader size, VQ write, bootstrap, plain depth ladder, core on tokens | plain loop vs d1 twin 0.0674 at 20k ([failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) ln 168); core over tokens +0.0102 | capacity and reader are not the lever; p3 ships for the plain loop |
| B3 | 09-14 | AA score, spectral gap, sink mass, basin map, LoopMTP port | plain loop AA 0.9998 ([failures/2026-09-14-arc-loop-diagnostics.md](../lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md)) | attractor-ness goes with earning; horizon targets degenerate on web text |
| B4 | 09-14 to 09-18 | LCTUL: sampled code, regressed code target, graded target, unfrozen reader | cell worth −4.64 frozen to +0.177 adapted ([successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md)) | a sampled code carries nothing; the reader limits usability, not depth |
| B5 | 09-18 to 09-19 | hop-distance probe, set toys, sample oracle, loop carry | prev-reach1 per-hop h3 +0.0640 ([failures/2026-09-18-hop-distance-earning.md](../lab/experiments/failures/2026-09-18-hop-distance-earning.md) ln 236) | earning is gather- and carry-limited; the relay staircase is real |
| B6 | 09-19 to 09-20 | LXTUL fan: K streams, diversity terms, select, write-all | fan4-all 0.0342 better than pk4 (a reader gain); K1−K6 +0.0049 ([successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md)) | fixes the reader and selector, not depth |
| B7 | 09-21 | LXTUL-P ladder (fp0, trig, noise, lineage, denoise), np0, cfg-tlow, pk8 | fp0 +0.0102, all of it pass 1's ([failures/2026-09-21-lxtul-fan4-all-fp0.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-fp0.md)) | state levers closed; teacher forcing is a bypass; the depth law |
| B8 | 09-21 to 09-22 | reach split, LXTUL-R, seed twin, LCTUL-J, synthesis, token-like read, 20k horizon pair | LXTUL-R 1b +0.0261 (a refund of a tax); plain reach 1 +0.0554 ([failures/2026-09-21-span-reach-split.md](../lab/experiments/failures/2026-09-21-span-reach-split.md) ln 171) | relay closes; Conditions A and B; slot vs twin a constant 0.0075 |
| B9 | 09-23 | LXTUL-G (posterior-trained stochastic loop), LXTUL-GK (multi-sample bound) | LXTUL-G width gain 0.159, qualified ([failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) ln 205) | exposure gap; GK noise collapses (filed failure) |

## 3. Glossary

Written by hist-A. A probe script is the definition of its instrument. Where a doc and a script disagree, the script wins.

### 3.1 Instruments

#### Token K-curve (K1−K6, K3−K6, K6−K12, K1−Kmax)

- **What it is.** The trainer forces the loop to run exactly `d` passes. The probe reads
  token cross-entropy (CE) at each `d` on the same packed rows. `Ka−Kb` is CE at depth a
  minus CE at depth b.
- **Sign.** Positive means the deeper model predicts better. K1−K6 is the value of passes 2
  to 6. K3−K6 is the value past pass 3, the "depth on trained support" number.
- **Code.** [lab/divergence/core_depth_sweep.py](../lab/divergence/core_depth_sweep.py)`::_bootstrap_pairs` (slot-loop and TUL
  arms), [lab/divergence/token_depth_sweep.py](../lab/divergence/token_depth_sweep.py)`::ce_map` (plain looped model, forces
  `model.cfg.mean_depth`), [lab/divergence/a2_depth_sweep.py](../lab/divergence/a2_depth_sweep.py)`::ce_maps` (paid loop, forces
  per-sample core depth). CI math: [lab/divergence/_stats.py](../lab/divergence/_stats.py)`::paired_bootstrap_ci`.
- **Rows.** Script default is 48 rows. The production panel is 480 validation rows at seq
  1024. The August 2026 filings mostly read 48 rows. Read the row count before you compare
  two numbers.
- **Floor.** Twelve slot-loop arms read token K1−K6 inside [−0.0001, +0.0033]; the campaign
  calls about +0.002 the slot-loop yardstick
  ([.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md)).
- **Pitfalls.**
  - A K-curve measures depth DEPENDENCE, not depth VALUE. A model that is worse at every
    depth can have a large K-curve. E6 is the first measured case
    ([successes/2026-09-07-arc-e6-deep-recurrence-draw.md](../lab/experiments/successes/2026-09-07-arc-e6-deep-recurrence-draw.md): K3−K6 +0.277 on
    a model 0.104 nats worse than the mean-6 model). Pair it with a CE comparison.
  - A K-curve read on a model that trained with `retention_carry` on is not admissible.
    The carry leaks the future across passes
    ([.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md](../.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md),
    Consequences: "no depth/loop claim is admissible without a carry-off ... sweep").
  - A reading taken before a run detonates is not an earning. The E2 rule
    ([failures/2026-09-04-arc-e2-iteration-conditioning.md](../lab/experiments/failures/2026-09-04-arc-e2-iteration-conditioning.md)) says a
    pre-onset checkpoint reads the expansive map, not the loop's work.
  - A broken depth 1 inflates K1−K6. E16's mask arm reads +0.406 because its depth-1 CE is
    1.31 against the plain model's 0.76
    ([failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md](../lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md)).
  - On a frozen coda (LCTUL `code_target` arms) the K-curve tracks how generic the cell
    is, not how good it is (`core_depth_sweep.py::warn_if_frozen_reader`).
  - Forced-depth pad cells inflate a trajectory-prefix arm
    ([failures/2026-09-13-arc-trajectory-prefix.md](../lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md), B2).
  - The entry state changes the size of the K-curve. The same norm_match rule reads 0.185
    under the noise entry and 0.033 under the prelude entry. See "Loop entry" below.

#### Forecast K-curve and own-span K-curve (`mux_local`, M-next, M-own)

- **What it is.** The same forced-depth read, but the probe scores the slot's own
  auxiliary loss, not the coda's token CE. M-next is the next-span bag target. M-own is
  the slot's own-span target. In staged arms the "next" and "own" columns are both read.
- **Sign.** Positive means deeper passes lower the slot's own loss.
- **Code.** `core_depth_sweep.py` (`arm[f"ci_{m}"]`). The bootstrap resamples batches, not
  tokens, so run it with `--batch 1` when the interval matters.
- **Pitfall.** No token loss reads this head directly. A forecast K-curve can be large
  while the token K-curve is flat, and the exit forecast can equal a one-pass forecast
  ([successes/2026-09-10-arc-slot-mnext-staged.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md): exit 6.7698 against
  the ruler's 6.7724).

#### Own-loss K-curve

The think-once panel's name for the forecast or memory K-curve on the slot's own MUX loss
([failures/2026-09-03-tul-think-once-panel.md](../lab/experiments/failures/2026-09-03-tul-think-once-panel.md)). Same instrument as above.

#### Paired CE and the bootstrap CI

- **What it is.** Token-weighted mean CE difference between two readings on the same
  tokens, with a percentile CI from 2,000 resamples of units.
- **Units.** Rows when both readings come from one arm. 1,024-token stream blocks when two
  arms cut the stream differently ([lab/divergence/sweep_score.py](../lab/divergence/sweep_score.py)`::paired`, `BLOCK = 1024`).
- **Sign.** `paired(a, b)` is a minus b. Each filing states its own order. Always read the
  order before you read the sign.
- **Code.** [lab/divergence/_stats.py](../lab/divergence/_stats.py)`::paired_bootstrap_ci`, [lab/divergence/sweep_score.py](../lab/divergence/sweep_score.py),
  [lab/divergence/paired_vs_ruler.py](../lab/divergence/paired_vs_ruler.py).
- **Pitfall.** Rows of two different arms correlate at 0.098 across 480 rows
  (`sweep_score.py` docstring). A row-paired cross-arm CI is not paired. The output labels
  such a CI "UNPAIRED".

#### Worth (zero, shuffle, wrong_seed) and the worth profile

- **What it is.** Ablate what the slot writes into the coda's cells. `zero` sets the cells
  to zero. `shuffle` permutes cells across slots in a row. `wrong_seed` uses a cell from
  another context. Worth is ablated CE minus intact CE.
- **Sign.** Positive means the coda uses the cells.
- **Offset profile.** The stratified version bins by the predicted token's offset in its
  span. A real write effect decays with offset.
- **Code.** [lab/divergence/worth_profile.py](../lab/divergence/worth_profile.py) (modes `zero`, `shuffle`, `wrong_seed`).
  [lab/divergence/slot_path_worth.py](../lab/divergence/slot_path_worth.py) separates loop worth from whole-slot-path worth.
- **Older scalars.** `val/plan_worth_shuffle` (the all-token shuffle scalar, retired
  2026-08-29 as too noisy) and `val/plan_nats` (CE with slot positions removed minus CE
  with them).
- **Pitfall.** Worth says the coda reads the cell. It says nothing about the passes. A
  cell can be worth 0.8 nats while the K-curve is 0.0006
  ([failures/2026-09-09-arc-coda-reads-the-thought.md](../lab/experiments/failures/2026-09-09-arc-coda-reads-the-thought.md)).

#### Loop worth and plan worth (August 2026 sense)

- **What it is.** In `slot_path_worth.py`, "loop worth" is CE with the loop removed (the
  slot keeps its bag-mean seed) minus normal CE. "Plan worth" is CE with the whole slot
  path removed minus normal CE. Both on `ce_main` (token CE) unless stated.
- **Sign.** Positive means the loop (or the plan) helps the tokens.
- **Where used.** A2 (2026-08-25 to 2026-08-28), for example `tul_v1a2b` loop worth
  +0.0058 to +0.0107.

#### Region Shapley (prelude, core, coda)

- **What it is.** Exact 3-player Shapley value over 8 coalitions. Each region is ablated to
  the identity. The value of a coalition is nats saved against the empty coalition. The
  probe reads `ce_main` (token CE), `ce_plast` (the boundary token's own label) and
  `ce_emit` (the slot's own one-token target).
- **Sign.** Positive means the region saves nats.
- **Code.** [lab/divergence/region_shapley.py](../lab/divergence/region_shapley.py). The leave-one-out predecessor is
  [lab/divergence/delta_ablation.py](../lab/divergence/delta_ablation.py), which cannot tell "useless" from "redundant".
- **First use.** [results/2026-08-25-region-shapley/README.md](../lab/experiments/results/2026-08-25-region-shapley/README.md).

#### Depth-1 twin (d1 twin, norecur twin, plain-depth1)

- **What it is.** A second training run of the same config at a fixed depth of 1. It is a
  training-time control, not a probe. The comparison is a paired CE gap at matched steps:
  looped model at its own depth against the twin at depth 1.
- **Sign.** Filings use two conventions. `ruler@6 − norecur@1 = +0.0020` in
  [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) means the looped model is
  worse. Other tables print the twin minus the looped model. This document states "looped
  model better by X" in words.
- **Configs.** [morph/configs/notul_depth1.yaml](../morph/configs/notul_depth1.yaml), `notul_norm_match_20k_d1.yaml`,
  `tul_slot_spandec_norecur.yaml`.
- **First use.** [failures/2026-09-09-arc-e19-parcae-loop-entry.md](../lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md)
  (`plain-depth1`).
- **Pitfall.** Matched steps is not matched compute. A depth-1 model runs faster, and a
  deep model converges slower.

#### Future-leak probe and the carry-off sweep

- **What it is.** Corrupt the input tokens after position k. Score CE only before k. If
  depth earning survives, the loop uses the past. If it collapses, the loop read the future.
- **Carry-off sweep.** Re-read the same weights with `model.retention_carry=false` at eval.
  If the K-curve inverts, the earning was the leak.
- **Code.** [lab/divergence/future_leak_probe.py](../lab/divergence/future_leak_probe.py).
- **First use.** [successes/2026-08-31-future-leak-attribution.md](../lab/experiments/successes/2026-08-31-future-leak-attribution.md) and
  [successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md).

#### Core Jacobian, typical gain, alignment, `loop/core_gain_t0`, `gain_est`

- **What it is.** `sigma_max` is the spectral norm of one core step's Jacobian. Typical
  gain is `‖J‖_F/√n`, the gain a generic direction sees (`jac/rms_t3` in logs). Alignment
  is whole-step gain divided by the product of per-block gains. `loop/core_gain_t0` is a
  cheap training-time ratio of output to input state norm at the first pass. `gain_est` is
  the slot-loop gain hinge's finite-difference estimate.
- **Sign.** Above 1 means expansive.
- **Code.** [morph/training/core_jacobian.py](../morph/training/core_jacobian.py), [lab/divergence/jac_ladder.py](../lab/divergence/jac_ladder.py);
  procedure in [docs/cookbook/measuring-the-core-map.md](cookbook/measuring-the-core-map.md).
- **Pitfalls.** A pad slot at h=0 inflated the first reading to 1.5e6. The eager kernel
  path biases `gain_est` up by about 0.07
  ([.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md](../.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md)). These are
  stability instruments. They do not measure contribution.

#### Detonation and the abort rule

- **What it is.** `preclip/total` is the pre-clip global gradient norm. A value above 1e4
  at any step at or after 200 is a detonation. The rule caught 17 of 17 detonations by
  step 775 with 0 false positives in 44 healthy runs
  ([lab/divergence/DIVERGENCE-README.md](../lab/divergence/DIVERGENCE-README.md)).
- **Code.** [morph/training/train.py](../morph/training/train.py) (logging), [lab/divergence/tripwire_sustained.py](../lab/divergence/tripwire_sustained.py).

#### Takeover and core share

- **What it is.** The 2026-08 slot-loop failure. The core's share of the gradient climbs
  to about 1 and validation CE turns up. "Core share above 0.5" marks the onset in
  [failures/2026-08-24-tul-takeover-cure.md](../lab/experiments/failures/2026-08-24-tul-takeover-cure.md).

#### Cotangent effective positions and slot effective rank

- **Cotangent effective positions.** How many slot positions carry the backward cotangent
  (`jac_ladder.py::cotangent_rank`). It fell from about 13 to 2.5 at takeover.
- **Effective rank.** `(Σλ)²/Σλ²` of the centred covariance of the slot states
  ([morph/model/fm_planner.py](../morph/model/fm_planner.py)`::effective_rank`; per-stage version
  [lab/divergence/slot_rank_anatomy.py](../lab/divergence/slot_rank_anatomy.py); the probe-variant check is
  [lab/divergence/slot_rank_probe_variants.py](../lab/divergence/slot_rank_probe_variants.py)). Read it with the mean pairwise cosine.
- **Pitfalls.** Centred input rank and uncentred in-loop rank are different quantities. A
  2026-08-24 claim compared them and was withdrawn the same day
  ([lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md), "WITHDRAWN 2026-08-24"). The 5.7598 rank of
  `slot-spandec-strict` was a bare-front probe; see B2 and
  [lab/divergence/slot_rank_probe_variants.py](../lab/divergence/slot_rank_probe_variants.py) docstring.

#### Specificity (plan span specificity)

- **What it is.** Shuffle cost divided by zero cost of the plan cells. High means the
  cells carry span-specific content. Low means the cells act as a constant.
- **Code.** `plan_shuffled` in [lab/divergence/slot_path_worth.py](../lab/divergence/slot_path_worth.py).
- **First use.** [planned/2026-08-28-plan-span-specificity.md](../lab/experiments/planned/2026-08-28-plan-span-specificity.md) (the file
  is under `planned/` but carries results).

#### Token tax (reader-or-target sweep)

- **What it is.** Force token-state dropout p at eval, p in {0, 0.5, 0.9, 1.0}, and re-read
  the plan's shuffle cost. It asks whether the coda bypasses a full plan or the plan is
  empty. [planned/2026-08-28-reader-or-target.md](../lab/experiments/planned/2026-08-28-reader-or-target.md) (carries results).

#### Cancellation ratio

- **What it is.** `|Σ_t dW_t| / Σ_t |dW_t|` over the per-pass shares of the shared core
  weight gradient. 1 means every pass asks for the same update. Lower means the passes
  disagree.
- **Code.** [lab/divergence/slot_gradient_probe.py](../lab/divergence/slot_gradient_probe.py) (the parametrize "tap").
- **Pitfall.** The sign of "good" flipped. The gradient probe read low cancellation as
  bad. The toy study found solved runs read lower cancellation than stuck runs
  ([lab/toy_slot_loop/WRITEUP.md](../lab/toy_slot_loop/WRITEUP.md)).

#### Slot geometry audit, K0−K6, and "write contribution" (z-opt entry)

- **K0−K6.** Coda CE with the loop's ENTRY state written to the cell minus coda CE with the
  6-pass EXIT state. This is what the passes add to the cell.
  [lab/divergence/slot_geometry_audit.py](../lab/divergence/slot_geometry_audit.py).
- **Write contribution, "z-opt entry".** The same quantity from
  [lab/divergence/slot_z_optimize.py](../lab/divergence/slot_z_optimize.py): `ce_entry − ce_loop`. hist-B's filings call it
  "write contribution". Positive means the passes improve the cell.

#### Fitted z (z-optimize) and the causal fit

- **Fitted z, hindsight.** Freeze the model. Make the slot state a free variable. Fit it by
  gradient descent to the coda's token CE on the tokens it is then scored on.
  [lab/divergence/slot_z_optimize.py](../lab/divergence/slot_z_optimize.py). This is an upper bound on what the coda can read.
  It sees the answer.
- **Causal fit.** Fit z on continuations sampled from a frozen teacher, never the real
  one. Score on the real continuation. [lab/divergence/slot_z_causal_fit.py](../lab/divergence/slot_z_causal_fit.py).
- **Pitfall.** The hindsight fit read −0.944 and −2.626 nats on 2026-09-10. The causal fit
  read +0.2517 nats WORSE than the loop's own z on 2026-09-12
  ([failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md)). Never quote the
  hindsight number as headroom.

#### Span budget

- **What it is.** Two plain models that differ only in whether anything may cross a span
  boundary (`model.span_mask`: `row` against `span`). The paired CE gap is the nats that
  live across span boundaries. [lab/divergence/span_budget_profile.py](../lab/divergence/span_budget_profile.py).
- **First use.** [failures/2026-09-11-arc-span-budget.md](../lab/experiments/failures/2026-09-11-arc-span-budget.md): 0.3994 nats
  [0.3838, 0.4162] at 5k.

#### All-slot worth (`all_slots`)

- **What it is.** CE with every slot position and cell ablated minus normal CE. On the
  strict geometry it equals `zero` worth by construction
  ([failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md)).

#### spandec_ce and the horizon grid

- **spandec_ce.** The span decoder's per-token CE of the whole next span, read from the
  slot state (`tul/spandec_ce`; [morph/model/tul_spandec.py](../morph/model/tul_spandec.py)).
- **Horizon grid.** Holds the target fixed and moves depth
  ([lab/divergence/spandec_horizon_grid.py](../lab/divergence/spandec_horizon_grid.py)). Must use the model's own TG kwargs.

#### Hop distance probe, per-hop K1−K6, planted copy benefit

- **Per-hop K1−K6.** The corruption localiser finds, for each token, the hop distance h of
  the span it needs most. The token K1−K6 is then binned by h.
  [lab/divergence/hop_distance_probe.py](../lab/divergence/hop_distance_probe.py).
- **Planted copy benefit.** Plant a rare token id g spans back and as the next span's first
  token. Benefit = CE_control − CE_source(g, d). Positive means the loop carried the
  token. It reads exactly zero before the pass at which the content can arrive.
- **Pitfall.** `--seed` does not change the rows. Use `--row-offset`.

#### Attractor score (AA) and the basin map

- **AA.** Cosine between the loop's exit from its real entry and its exit from a swapped
  or noised entry. Near 1 means a path-independent fixed point. [lab/divergence/aa_score.py](../lab/divergence/aa_score.py).
- **Basin map.** Perturb the entry on a 2D grid and read the settling depth of the decoded
  argmax. [lab/divergence/basin_map.py](../lab/divergence/basin_map.py).

#### Fan instruments (Part B)

- **oracle − mixed, selector regret.** `fan/mixed_ce` (deployed all-cell read) minus
  `fan/oracle_ce` (per-span best stream, chosen after the fact). The bound on the part a
  reader can recover is log K per span. [lab/divergence/fan_mixture_probe.py](../lab/divergence/fan_mixture_probe.py).

#### LCTUL instruments (Part B)

- **K-draw marginal (k16 − k1).** Log of the mean over K sampled codes of the span
  likelihood, read at Euler step counts k. [lab/divergence/code_marginal_sweep.py](../lab/divergence/code_marginal_sweep.py),
  [morph/training/code_eval.py](../morph/training/code_eval.py)`::code_marginal_ce`.
- **Semantic OWN / SHUF / ZERO / ORACLE.** Greedy continuation from the thinker's own
  sample, a sample for another cut, a zero code, and the true code, scored by MiniLM
  cosine to the true span. [lab/divergence/code_semantic_probe.py](../lab/divergence/code_semantic_probe.py).
- **Twin read.** The frozen VAE-stage twin's reader scores the live loop's cell.
  [lab/divergence/code_twin_read_probe.py](../lab/divergence/code_twin_read_probe.py). Deltas are condition − own; positive means the
  own cell is better.

#### LXTUL-G and GK instruments (Part B)

All from [lab/divergence/lxtul_g_probe.py](../lab/divergence/lxtul_g_probe.py):
- `ce_post`: the coda on one posterior sample. It reads the answer. Never quote it alone.
- `ce_prior@1`: the coda on one prior sample.
- `ce_iw@N`: the exact per-token Bayesian read over N prior samples. Weights at token j are
  the softmax over samples of the log-likelihood of the span's earlier tokens.
- Exposure gap: `ce_prior@1 − ce_post`
  ([failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md), P-2).
- Width gain: `ce_prior@1 − ce_iw@4` (same file, P-4). Positive means extra samples help.

### 3.2 Terms used for arms and regimes

- **Plain loop (notul).** The looped model with no slots. Tokens run through the core.
- **Slot loop.** TUL as designed ("think once, decode cheap"). Only slot positions run the
  core (`tul.tokens_through_core: false`, `_tul_core`).
- **Paid loop (A2).** Tokens and slots both run the core (`tul.tokens_through_core: true`).
  Shipped 2026-09-03. Rejected by Wolfe as "not TUL".
- **Ruler.** The reference arm a panel pairs against, for example `slot-mux-norm-match`.
- **absmean, norm_match.** The two ternary scale rules. norm_match shipped 2026-09-09
  ([.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md](../.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md)).
- **Loop entry.** Noise entry (Parcae style, `core_state_init: noise`) against prelude
  entry (the loop starts from the prelude's state).
- **Mask (`tg_restrict`).** Tokens see only their own span and slots. The slot is the only
  cross-span route. Also called the bypass geometry when only prelude and coda are masked
  and the prelude cells can still skip the loop.
- **Strict geometry.** The prelude sees only its own span. The loop is the only
  cross-span channel. `prev` limits the coda to cell j−1. `loop_reach` limits how far a
  pass reaches.
- **retention_carry.** A GLA state carried across core passes. On by default until
  2026-08-31. It made every pass after the first able to read future tokens.

## 4. Part A eras (A1 to A8)

Author: hist-A. Range: TUL's first commit (`5d258e1`, 2026-08-16) through [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md). Each era gives what was tried and why, a readings table, what was concluded then, and what corrected it later.

One fact frames A1 to A3. Until 2026-08-31 every model trained with
`model.retention_carry` on. That GLA carry let every core pass after the first read future
tokens. The bug-fix note says every pre-fix checkpoint "is a different (leak-reliant)
model" and "no depth/loop claim is admissible without a carry-off ... sweep"
([.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md](../.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md),
Consequences, ln 80-88). Only the l2cap and tul-30k checkpoints were ever re-read with the carry off. Every
other loop reading from A1 and A2 is therefore marked "qualified (pre-fix)".

### A1. The first arms and the takeover campaign (2026-08-16 to 2026-08-24)

**What we tried and why.** TUL's spec started from a negative: decoding a span from one
vector plus an offset collapses. So the slot holds a latent, the core loops on the slots
only, and the coda reads the slot as a prefix ([.agents/specs/tul-spec.md](../.agents/specs/tul-spec.md) ln 52-54). The
first 20k-step pair put the slot-loop arm A1 against the dense baseline A0. A span-length
gate (TUL-gate) came next. Then arm A1 began to fail: validation CE turned up after step
1000 to 2000 while the core took over the gradient ("the takeover"). About twenty
hypotheses (spectral norm, AdEMAMix slow channel, under-determination, reader conflict,
forcing bias, attention sink) were tested against it
([lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md)).

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| 7.12 nats | span decoder CE, one vector plus offset | Huginn variant | n/a | [.agents/specs/tul-spec.md](../.agents/specs/tul-spec.md) ln 53 | stands (motivating negative) |
| A1c beats A0c by 0.0562 | val token CE, paired arms | tul-a1-acap1 vs tul-a0-acap1 | 20000 | [lab/tul/arms-result.md](../lab/tul/arms-result.md) ln 51 | qualified: n=1, "not the gate" (same file ln 86-89); pre-fix leak model (the +0.1433 leak was measured on tul-a0-acap1, retention-carry note ln 23) |
| −0.1196 | first_tok_counterfactual (ce_plast − ce_emit) | tul-a1-acap1 | 20000 | [lab/tul/arms-result.md](../lab/tul/arms-result.md) ln 74 | stands: the plan loses to the trivial channel at its own job |
| −0.1054 | val CE, gate − A1 | tul_gate vs tul_a1 | 20000 | [results/2026-08-23-tul-gate-bakeoff.md](../lab/experiments/results/2026-08-23-tul-gate-bakeoff.md) ln 29 | qualified: the leak, +0.1433, is larger than the headline (same file ln 186); single seed pair |
| 0.1900 vs 0.0045 | plan_nats | tul_gate vs tul_a1 | 20000 | same file ln 32 | qualified (pre-fix) |
| CW1 beats CW2 by 0.009 [0.0078, 0.0102] | eval screen: slots kept vs random tokens kept | tul-a1-acap1 | 20000 | [.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md](../.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md) ln 19, 53 | qualified: a stratified re-score favours random tokens in one stratum (same note ln 35); pre-fix |
| 1.7 to 4.8 | slot-state effective rank in 1024 dims | onset ladder | 1625 to 1866 | [failures/2026-08-24-tul-takeover-cure.md](../lab/experiments/failures/2026-08-24-tul-takeover-cure.md) ln 21 | stands as a number |
| "loop destroys ~10x more diversity than pooling" | centred input rank vs uncentred loop rank | onset ladder | n/a | [successes/2026-08-24-tul-span-pooling-law.md](../lab/experiments/successes/2026-08-24-tul-span-pooling-law.md) ln 103 | retracted the same day: two different measures ([lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md) ln 135-137) |
| 13 to 2.49 | cotangent effective positions, healthy to takeover | onset ladder | 1625 to 1866 | [failures/2026-08-24-tul-takeover-cure.md](../lab/experiments/failures/2026-08-24-tul-takeover-cure.md) ln 229, 234 | stands |
| ×2.9 alignment, +2.5 % gain | core Jacobian | onset ladder | 1625 to 1866 | same file ln 165-167 | stands |
| onset 1150 to 2225; best CE 0.78 and 0.46 below control | takeover step; val CE | b10-ctrl vs per_slot_embed arms | to ~3900 | same file ln 36-37 | stands; "not a cure" (fails one seed of two); removed from the tree 2026-09-03 |

**What we concluded then.** The loop is a stability problem. The takeover is positional:
the backward cotangent concentrates on a few slots while the core blocks' amplifying
directions align. Four spectral interventions failed and two made it worse. The best lever,
`per_slot_embed`, doubles time to failure and is not a cure
([.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md](../.agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md)).

**What corrected it later.** Two things. First, the "stability" frame gave way within a day
to a gradient-flow frame (A2). Second, the retention-carry leak (found 2026-08-23, fixed
2026-08-31) sits under every CE number of this era. The gate's win and the A1 over A0 win
were never re-read on a causal model. The slot-only arms of this era were deleted on
2026-09-03 (`d9e04e6` is the last commit that runs them).

---

### A2. What the slot and the loop carry: "the plan is empty" (2026-08-25 to 2026-08-28)

**What we tried and why.** The campaign stopped asking how to stop the takeover and asked
how much the core is worth. An exact Shapley split over prelude, core and coda answered
it. Then came a set of attacks on the "empty plan": a MUX span head, the Thought Gestalt
restriction (tokens see only their own span, so cross-span content must go through the
slot), warmup, SIGReg, NTP dropout, and a flow-matching (FM) planner line. SCSE and the
attention-sink hypothesis were tested and refuted as takeover mechanisms.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| core 0.0007 (ce_main) vs 0.3296 (ce_emit) | region Shapley | ROLL_step_1750, healthy A1 | 1750 | [results/2026-08-25-region-shapley/README.md](../lab/experiments/results/2026-08-25-region-shapley/README.md) ln 19, 28 | qualified (pre-fix) |
| core 0.2274 (ce_emit) | region Shapley | ROLL_step_1625 | 1625 | same file ln 18 | qualified (pre-fix) |
| loop 0.0051, whole plan 0.0191 | loop worth / plan worth on ce_main | onset ladder | ~1750 | same file ln 88-89; [lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md) ln 49-50 | qualified (pre-fix) |
| plan 0.0699 at p=1.0 | plan worth under token-state dropout | same | same | [lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md) ln 50-52 | qualified (pre-fix) |
| 2.6564 | worth of the core's direct target (ce_emit) | same | same | [lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md) ln 55 | qualified (pre-fix) |
| +0.0169 whole core; prelude +3.2205, coda +3.1051 | leave-one-out CE cost | ROLL_step_1750 | 1750 | [failures/2026-08-25-scse-arm-c-long.md](../lab/experiments/failures/2026-08-25-scse-arm-c-long.md) ln 152-154 | qualified (pre-fix); the "core loop is worth 0.017" prior |
| +0.0058, +0.0082, +0.0106, +0.0107 | loop worth on ce_main, 4 seeds | tul_v1a2b (detached MUX head, β 0.1) | 3500 | [failures/2026-08-27-warmup-sigreg-ntpdrop.md](../lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md) ln 268 | qualified: post-hoc (same file ln 273-276); the fresh-seed confirmation ([planned/2026-08-27-mux-matched-control-confirmation.md](../lab/experiments/planned/2026-08-27-mux-matched-control-confirmation.md)) has no results; pre-fix |
| p = 0.0286 | permutation test vs 3 controls | same | same | same file ln 269 | same |
| −0.0002 to +0.0042 | control loop worth | tul_a1 | 3500 | same file ln 261 | stands as the control band |
| +0.0276 → +0.0363; +0.0378 → +0.0237 | loop worth, 3000 → 3500 | tg2-s1; tg2-s2 | 3000, 3500 | [failures/2026-08-27-tg-restriction.md](../lab/experiments/failures/2026-08-27-tg-restriction.md) ln 78-79 | qualified: under the 0.05 line; plan worth rise withdrawn as a fallback confound (same file ln 105-113); later specificity 0.1-3.0 % |
| 65.1 % vs 0.1 to 3.0 % | plan span specificity | ctrlworth-s3 vs restricted arms | 3000 | [planned/2026-08-28-plan-span-specificity.md](../lab/experiments/planned/2026-08-28-plan-span-specificity.md) ln 103-106 | stands (the aux losses, not the mask, write content; in-file correction ln 179) |
| 0.0148 normal, 0.0867 fully starved | plan worth, token tax | ctrlworth-s3 | 3000 | [planned/2026-08-28-reader-or-target.md](../lab/experiments/planned/2026-08-28-reader-or-target.md) ln 136 | stands; aux-off arms ≤ 0.0104 |
| −0.0000 [−0.0001, +0.0001] | CE change with the TRUE target planted at the prefix | fm1-cw-s1 | 4500 | [successes/2026-08-28-oracle-prefix-probe.md](../lab/experiments/successes/2026-08-28-oracle-prefix-probe.md) ln 79-80 | stands: the additive prefix interface reads nothing |
| A3 best 4.0061 vs A1 best 4.8434 | val CE, no slots/no core vs full TUL | tul_a3 vs tul_a1 | 4250 | [planned/2026-08-28-does-the-slot-apparatus-pay.md](../lab/experiments/planned/2026-08-28-does-the-slot-apparatus-pay.md) ln 166-167 | stands: "The slot apparatus does not pay" |

**What we concluded then.** The plan is empty, not outcompeted. The core does real work
only on its own one-token target (`ce_emit`), and almost none of it reaches the token loss.
The objective split (2026-08-27) found no gradient conflict: "the coda's objective has
almost no achievable gain through the slot" ([failures/2026-08-27-objective-split.md](../lab/experiments/failures/2026-08-27-objective-split.md)
ln 218-220). The one positive loop-worth number is `tul_v1a2b`, and the file calls it
post-hoc. The FM line showed a planner can be 66 to 77 % accurate and still be worth zero
to the coda; even the true target pays nothing through the additive prefix.

**What corrected it later.** No later file retracts these numbers, but none survives the
carry rule: all are pre-fix. [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) marks `tul_v1a2b`
"standing, small, post-hoc" and region Shapley "standing" without the pre-fix caveat. I
disagree with that status and flag it. The "0 of 4 takeovers" claims of the TG arms were
qualified in-era: every "held" arm stopped at step 3500, and the aux-off control took over
57 steps after that point ([planned/2026-08-28-does-the-slot-apparatus-pay.md](../lab/experiments/planned/2026-08-28-does-the-slot-apparatus-pay.md) ln
119-137).

---

### A3. The loop ladder, l2cap, and the causality leak (2026-08-29 to 2026-08-31)

**What we tried and why.** The question became whether iteration itself kills training or
only uncontrolled iteration does. The loop ladder ran full BPTT with and without a hard
σ ≤ 1.5 spectral projection (l2cap). l2cap read the campaign's first large K-curve. A day
of ablations (conditioning, gates, truncated BPTT, DB init, interleaving) all flattened it,
which built the "identity-escape law". Then a 30k run read an impossible val CE of 1.24.
The carry-leak audit and the future-leak probe followed. After the leak fix, a bisect asked
which MORPH addition (GLA, the cap) killed depth earning on the plain loop.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| +0.2328 (4.6220 → 4.3892) | token K1−K6, 48 rows, carry on | tul-l2-cap | 4500 | [successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md) ln 81; first claimed [successes/2026-08-29-tul-loop-ladder.md](../lab/experiments/successes/2026-08-29-tul-loop-ladder.md) ln 150 | **retracted**: carry off, same weights, reads −1.1198 (4.6220 → 5.7418), same file ln 82 |
| +0.2580 clean, −0.0128 corrupt (105 % collapse) | future-leak probe, K1−K6 | l2cap-4500 | 4500 | [successes/2026-08-31-future-leak-attribution.md](../lab/experiments/successes/2026-08-31-future-leak-attribution.md) ln 89 | stands: the l2cap earning was future-reading |
| 3.85 nats | leak size at native depth | tul-30k | 30000 | [successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md); retention-carry note ln 24 | stands |
| +0.0171, clean = corrupt | K1−K6, causal carry | tul-l2nc (l2cap recipe trained causal) | 4500 | [successes/2026-08-31-future-leak-attribution.md](../lab/experiments/successes/2026-08-31-future-leak-attribution.md) ln 91 | stands: the honest l2cap recipe is flat |
| 0.0127 vs 0.233 | K1−K6 | tul-l2cap-cond vs l2cap | 4500 | [successes/2026-08-30-tul-ilv50-l2capcond.md](../lab/experiments/successes/2026-08-30-tul-ilv50-l2capcond.md) ln 92 | the 0.0127 stands; its comparator is retracted |
| 0.0085 "of l2cap's 0.233" | K1−K6 | tul-l2trunc | 4500 | [failures/2026-08-30-tul-l2-trunc.md](../lab/experiments/failures/2026-08-30-tul-l2-trunc.md) ln 42 | the 0.0085 stands; the comparator is retracted |
| 0.120 / 0.015 | K1−K6 / K3−K6 | notul-l2nc (plain, causal, GLA on, cap on) | 4500 | [successes/2026-08-31-loop-killer-bisect.md](../lab/experiments/successes/2026-08-31-loop-killer-bisect.md) ln 28 | stands (reference) |
| 0.142 | K1−K6 | BC0 (cap off, GLA on) | 4500 | same file ln 74 | stands |
| **0.220** / 0.017 | K1−K6 / K3−K6 | BG0C0 (plain, GLA off, cap off, carry none) | 4500 | same file ln 112-113 | stands: the first honest plain-loop K-curve; pre-ramp, pre-norm_match |
| **0.207** / 0.016 | K1−K6 / K3−K6 | notul-20k (winner recipe, plain) | 20000 | [successes/2026-08-31-tul-vs-notul-20k.md](../lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md) ln 80-82 | stands; saturates by K4 |
| 0.015 | K1−K6, core axis | tul-20k (slot loop, winner recipe) | 20000 | same file ln 83 | stands: the slot loop is flat on the honest recipe |
| 0.357 | val CE gap, notul beats tul | notul-20k vs tul-20k | 20000 | same file ln 62 | stands |

**What we concluded then.** On 2026-08-30: "Contractivity control is a REQUIREMENT for a
trainable iterated write" and l2cap is "the ONE recipe whose loop earns depth". On
2026-08-31, after the audit: "No recipe in the campaign has been shown to earn depth
honestly" (carry-leak audit ln 102-105). Then, the same day, the honest bisect found that
GLA and the cap together had killed earning on the plain loop, and BG0C0 earned 0.220.

**What corrected it later.** The correction banner exists only in
[.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md](../.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md) ln 5-18. Five
places still treat l2cap as a win and carry no correction, as of this read. Four state
0.233: the root [CLAUDE.md](../CLAUDE.md) ln 66 ("the l2cap recipe ... is the ONE recipe whose loop
earns depth (0.233 nats)"), [.agents/notes/implemented/architecture/2026-08-29-loop-ladder.md](../.agents/notes/implemented/architecture/2026-08-29-loop-ladder.md)
ln 53, [.agents/notes/archived/architecture/2026-08-30-dbfix-program.md](../.agents/notes/archived/architecture/2026-08-30-dbfix-program.md) ln 9 and
[.agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md](../.agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md) ln 8, 27. The fifth,
[.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md](../.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md) ln 32,
calls the l2cap checkpoint "the best substrate that will ever exist". The
"identity-escape law" was built from ablations of a leak; every "flattening" arm flattened
a leak, not a loop. Its conclusion (iteration is not fatal) happens to hold on the honest
BG0C0 number. The honest plain-loop curve itself was cut later by the 1000-step ramp and
absmean (A4 and A5) and lifted again by norm_match (A6).

---

### A4. The paid loop: earning follows payment (2026-09-01 to 2026-09-03)

**What we tried and why.** The slot loop was flat on the honest recipe (0.015). The
slot-channel recovery program tried the write side: bound seeds, span-aligned compression,
content seeds. The write-side ladder closed it: no seed mode moved slot-only earning above
0.011. The same night arm A2 sent tokens AND slots through the core, to test whether the
slot loop is flat because no token pays for it. A2 earned, and it detonated in 2 of 4
draws. A chain of stability runs found ternary QAT as the trigger and the 1000-step LR ramp
as the cure. A 20k matched pair then asked whether the paid loop beats the plain loop.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| 0.0113 / 0.0007 / 0.0019 | token K1−K6 | R0 / W1 / W2 (slot-only seeds) | 5000 | [failures/2026-09-01-write-side-ladder.md](../lab/experiments/failures/2026-09-01-write-side-ladder.md) ln 179-181 | stands: the free-ride floor |
| **+0.1685** | token K1−K6 | A2 (paid, warmup 0, absmean) | 5000 | [successes/2026-09-01-a2-paid-loop.md](../lab/experiments/successes/2026-09-01-a2-paid-loop.md) ln 104 | stands as a measurement; the arm is "not TUL" (see below) |
| 0.298 | K6 CE, A2 beats R0 | A2 vs R0 | 5000 | same file ln 115 | stands |
| 2 of 4 | paid-axis detonations | A2 + R1 | ~2040 | same file ln 112 | stands |
| max \|ΔCE\| = 0 (v1 and boundary); earning 0.1666 / 0.1817 | future-leak probe | A2 | 5000 | [successes/2026-09-02-a2-future-leak-probe.md](../lab/experiments/successes/2026-09-02-a2-future-leak-probe.md) ln 78-84, 95, 104 | stands: A2's earning is causal |
| 0.0463 / 0.0489 / 0.0456 vs 0.1209 | token K1−K6 | A2 + ramp, 3 draws vs flat A2 | 2500 | [failures/2026-09-02-a2-warmup-and-seq512.md](../lab/experiments/failures/2026-09-02-a2-warmup-and-seq512.md) ln 137-140 | stands: the ramp cuts earning to about a third |
| 0.058 vs 0.1685 | token K1−K6 | tul-a2-wu5k vs flat A2 | 5000 | [failures/2026-09-02-a2-warmup-5k-earning.md](../lab/experiments/failures/2026-09-02-a2-warmup-5k-earning.md) ln 57-58 | stands |
| **0.104** (480 rows) | token K1−K6 | tul-a2-20k-wu | 20000 | [failures/2026-09-02-warmup-20k-pair.md](../lab/experiments/failures/2026-09-02-warmup-20k-pair.md) ln 127 | stands; grows at every checkpoint (0.041 at 2500) |
| 0.041 | token K1−K6 | notul-20k-wu (plain, ramp) | 20000 | same file ln 127 | stands |
| +0.132 → +0.012 (48 rows); +0.022 (480 rows) | K6 CE gap, paid − plain, matched steps | pair | 5000 → 20000 | same file ln 141, 147, 125 | stands: the gap closes but never crosses by 20k |
| +0.0371 [+0.0355, +0.0388]; K3−K6 +0.0010 | own-loss K1−K6 | R3 M-own (slot loop) | 5000 | [failures/2026-09-03-tul-think-once-panel.md](../lab/experiments/failures/2026-09-03-tul-think-once-panel.md) ln 231 | qualified: token K1−K6 +0.0002 on the same row; fails the K3−K6 > 0.01 "thinks" bar |
| 0.0014 | token K1−K6 | R1 A1-wu (slot loop, ramp) | 5000 | same file ln 230 | stands |

**What we concluded then.** "Earning follows payment" ([successes/2026-09-01-a2-paid-loop.md](../lab/experiments/successes/2026-09-01-a2-paid-loop.md)
ln 124-127; [.agents/notes/implemented/architecture/2026-09-02-loop-earns-when-paid.md](../.agents/notes/implemented/architecture/2026-09-02-loop-earns-when-paid.md)).
On 2026-09-03 the paid loop shipped as `base.yaml`'s TUL forward and the slot-only code was
deleted ([.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md](../.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md)).
The honest caveat stood in the same note: at 20k the paid arm is 0.022 nats behind plain at
1.33x the wall clock.

**What corrected it later.** No number of this era is retracted. The identity of the arm
is. Wolfe rejected the paid loop the same evening ("totally wrong",
[.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md](../.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md)
ln 10). The slot loop came back on 2026-09-04 (the ship note's amendment). The 2026-09-22
synthesis states it as a standing call: "TUL means the slot loop. The paid loop is not
TUL." The lock-in record now lives at
[.agents/notes/rejected/architecture/tul-paid-loop-recipe.md](../.agents/notes/rejected/architecture/tul-paid-loop-recipe.md). The 40k continuation that
would test whether the paid gap crosses ([planned/2026-09-03-warmup-pair-continue-40k.md](../lab/experiments/planned/2026-09-03-warmup-pair-continue-40k.md))
never ran.

---

### A5. The slot loop returns stable and empty: the arc E0 to E18 (2026-09-04 to 2026-09-08)

**What we tried and why.** The slot loop came back with a forward gain constraint (gain
hinge at 0.9, cotangent clip 4.0). It trained clean to 5000 steps for the first time, and
it read nothing. The arc ([planned/2026-09-04-loop-contribution-arc.md](../lab/experiments/planned/2026-09-04-loop-contribution-arc.md)) then tried
every lever it could name, one factor at a time: the gain target (E1), per-iteration
conditioning (E2), the mask so tokens must read the slot (E4), a deep draw (E6, E13), a
multi-token target (E8), a wider carry (E9), loss terms for stability (E10, E11), an
expansive map (E14), Olympiad math and Sudoku (E16, E17), and slot write width (E18). As an
external ruler, Huginn-3.5B was scored with the same instruments.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| +0.0128 / +0.0135; K3−K6 −0.0006 / +0.0002 | forecast K1−K6 | Y1 / Y2 (constrained slot loop) | 5000 | [successes/2026-09-04-tul-forward-levers.md](../lab/experiments/successes/2026-09-04-tul-forward-levers.md) ln 172-173 | stands: "stable and empty" (ln 235) |
| +0.0138 / +0.0180 | forecast K1−K6 | E1 g95 / g98 | 5000 | [failures/2026-09-04-arc-e1-gain-target-dial.md](../lab/experiments/failures/2026-09-04-arc-e1-gain-target-dial.md) ln 94-95 | stands; K3−K6 ≤ 0.0014 |
| +0.0077 | forecast K3−K6 | E2, pre-onset draw | 2500 | [failures/2026-09-04-arc-e2-iteration-conditioning.md](../lab/experiments/failures/2026-09-04-arc-e2-iteration-conditioning.md) ln 200 | **retracted** in the same file: the stable rerun reads +0.0001 (ln 175, 200-202) |
| **+0.0209**; forecast +0.187 | token K1−K6 | E4, mask under the constraint (absmean) | 5000 | [successes/2026-09-04-arc-e4-mask-under-constraint.md](../lab/experiments/successes/2026-09-04-arc-e4-mask-under-constraint.md) ln 76 | qualified: 0.13 CE tax (ln 106); gone under norm_match, +0.0009 ([failures/2026-09-10-arc-slot-mux-mask-norm-match.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md) ln 219) |
| +0.0414 / +0.1037 | token K1−K6 | notul-20k-wu / tul-a2-20k-wu | 20000 | [failures/2026-09-04-arc-e0-where-depth-earns.md](../lab/experiments/failures/2026-09-04-arc-e0-where-depth-earns.md) ln 65, 67 | stands; earning spreads evenly over offsets |
| **+0.566**; K6−K16 +0.206; K1−K6 +1.906 | token K-curves | Huginn-0125 (external, eval only) | n/a | [failures/2026-09-04-huginn-loop-contribution.md](../lab/experiments/failures/2026-09-04-huginn-loop-contribution.md) ln 144-147 | stands: web text is not depth-flat; no depth-1 control, so dependence |
| +0.4185 / K3−K6 +0.0932 | own-loss K1−K6 | X1 (backward clip only) | 2500 | [failures/2026-09-04-tul-clip-through-time.md](../lab/experiments/failures/2026-09-04-tul-clip-through-time.md) ln 198 | **retracted** in the same file: a model 0.75 nats worse, "not a THINK in substance" (ln 216) |
| K3−K6 **+0.277**; K6−K12 +0.0405 | token K-curves | E6 notul_deep16 (plain, mean 16) | 5000 | [successes/2026-09-07-arc-e6-deep-recurrence-draw.md](../lab/experiments/successes/2026-09-07-arc-e6-deep-recurrence-draw.md) ln 83, 88 | qualified: 0.104 worse than mean 6 at its own depth (ln 89); 27 % of samples silent under the BPTT defect ([.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md](../.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md) ln 14) |
| +0.0263; forecast +0.823 | token / forecast K1−K6 | E7 mask + deep 16 | 2500 | [failures/2026-09-07-arc-e7-block-loop.md](../lab/experiments/failures/2026-09-07-arc-e7-block-loop.md) ln 97-98 | **retracted**: pre-onset, detonated at 2712 (same file ln 126-128) |
| +0.0526; next-token K3−K6 +0.0021 | token K1−K6 | E8-6 (4 MTP heads) | 5000 | [failures/2026-09-07-arc-e8-multi-token-coda.md](../lab/experiments/failures/2026-09-07-arc-e8-multi-token-coda.md) ln 99-100 | qualified: next-token CE +0.3445 worse (ln 101) |
| 0 of 6 vs 4 of 7 | detonations, warmup 0 | E10a fixed-point term vs controls | 1200 | [successes/2026-09-07-arc-e10-loop-loss-terms.md](../lab/experiments/successes/2026-09-07-arc-e10-loop-loss-terms.md) ln 97-98 | stands (stability, not contribution) |
| 0.0216 vs 0.0367; CE +0.0009 | token K1−K6; paired CE | E11 vs notul (ramp) | 5000 | [successes/2026-09-07-arc-e11-fixed-point-ramped.md](../lab/experiments/successes/2026-09-07-arc-e11-fixed-point-ramped.md) ln 62-65 | stands: the term is free and lowers dependence |
| **+0.0489**; forecast K3−K6 +0.0172 | token K1−K6 | E13 mask, mean 12 | 5000 | [failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 145, 129 | qualified: same CE as mean 6 (ln 131); absmean artefact per the A6 mask arm |
| +0.0425 / +0.0601 | token K1−K6 | E14 g102 / g102-rn | 2500 | [failures/2026-09-07-arc-e14-expansive-dial.md](../lab/experiments/failures/2026-09-07-arc-e14-expansive-dial.md) ln 133-134 | **retracted**: pre-onset, both detonated (3877, 4639); K3−K6 0.0007 |
| +0.406; answer K3−K6 −0.030 | token K1−K6 | E16 mask, Olympiad | 6000 | [failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md](../lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md) ln 161 | **retracted** as depth: depth 1 is broken, 1.31 nats (ln 202) |
| −0.0000 [−0.0006, +0.0005] | Sudoku accuracy@12 − accuracy@3 | notul_sud | 6000 | [successes/2026-09-08-arc-e17-sudoku-depth-grid.md](../lab/experiments/successes/2026-09-08-arc-e17-sudoku-depth-grid.md) ln 165 | stands: flat |
| +0.0243 / +0.0314 / +0.0486 | token K1−K6 | E18 mask k2 / k4 / k8 (absmean) | 5000 | [planned/2026-09-04-loop-contribution-arc.md](../lab/experiments/planned/2026-09-04-loop-contribution-arc.md) ln 339 | qualified: "an absmean artefact" (same row) |

**What we concluded then.** Stability and contribution are separate axes
([lab/divergence/BREAK-GLASS-IN-CASE-OF-DIVERGENCE-THE-SLOT-LOOP-GAIN-CONSTRAINT.md](../lab/divergence/BREAK-GLASS-IN-CASE-OF-DIVERGENCE-THE-SLOT-LOOP-GAIN-CONSTRAINT.md),
section 4). A K-curve measures dependence, not value (E6). Every slot-loop "earning" that
came with instability was pre-onset. The mask made tokens read the slot's depth, which was
the arc's best slot-loop reading.

**What corrected it later.** A6 removed the mask's token dependence: under norm_match
the mask arm reads +0.0009. The E13 row of [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) calls
+0.0172 a "forecast K6−K12" reading. The source file shows +0.0172 is forecast K3−K6, and
forecast K6−K12 is +0.0020 ([failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 129). The
eager-path gain hinge read +0.07 high on every mask arm (E4, E13, E14)
([.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md](../.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md)). E3, the staged
target, was built on 2026-09-04 and never ran on a GPU in this era. It first ran as
`slot-mnext-staged` on 2026-09-10 (A7; [successes/2026-09-10-arc-slot-mnext-staged.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md) ln 35). Wolfe later
called math and Sudoku "red herrings" for this question
([docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) ln 13).

---

### A6. norm_match and the entry confound (2026-09-09 to 2026-09-10)

**What we tried and why.** E19 rebuilt the loop entry the Parcae way (noise state init,
all-dim injection, no fixed-point term). It barely moved the curve. E20 ran three one-factor
candidates; only a bf16 core moved it. So the ternary core was the limiter: the absmean
rule shrank each ternary layer to 0.668 of its latent's norm. The per-pass strength panel
tested three ternary-preserving fixes. norm_match won and shipped. Then the same rule went
onto the slot loop, and a panel read the plain loop under the prelude entry the slot loop
uses.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| +0.033 / +0.0009 | token K1−K6 / K3−K6 | parcae-entry (absmean, noise entry) | 5000 | [failures/2026-09-09-arc-e19-parcae-loop-entry.md](../lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md) ln 141-142 | stands |
| +0.0192 [+0.0168, +0.0213] | E18 plain@6 − plain-depth1@1 (depth-1 twin ahead) | plain-depth1 | 5000 | same file ln 157-158 | stands: at 5k the twin is 0.019 ahead |
| **+0.168**; K3−K6 +0.0119; −0.0336 vs depth-1 | token K1−K6; paired CE | depthcand-dense-core (bf16 core) | 5000 | [failures/2026-09-09-arc-e20-loop-depth-candidates.md](../lab/experiments/failures/2026-09-09-arc-e20-loop-depth-candidates.md) ln 147, 161 | stands as a diagnostic; dense-then-ternary is not allowed in production |
| **+0.1849 [+0.1816, +0.1883]**; K3−K6 +0.0139 | token K1−K6 | scale-norm-match (noise entry) | 5000 | [successes/2026-09-09-arc-per-pass-strength.md](../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md) ln 112 | stands; CE@6 +0.0817 worse than absmean at 5k (same row) |
| +0.0410 / +0.0261 | token K1−K6 | scale-ttq / threshold-03 | 5000 | same file ln 113-114 | stands |
| +0.043 / +0.074 | token K1−K6 | density-half / density-quarter | 5000 | [failures/2026-09-09-arc-density-panel.md](../lab/experiments/failures/2026-09-09-arc-density-panel.md) ln 136-137 | qualified in-file: "dependence without computation" (ln 163) |
| 0.035 to 0.041 | token K1−K6 by step | old-entry notul, absmean | 2500 to 20000 | [failures/2026-09-09-arc-horizon-ternary-25k.md](../lab/experiments/failures/2026-09-09-arc-horizon-ternary-25k.md) ln 95-103 | stands: absmean earning does not grow |
| 0.1363 / 0.1562 / 0.1665 / **0.1702** | token K1−K6 | norm-match-20k (plain, noise entry) | 5k / 10k / 15k / 20k | [failures/2026-09-09-arc-norm-match-recipe-reads.md](../lab/experiments/failures/2026-09-09-arc-norm-match-recipe-reads.md) ln 141-144 | stands: earning grows with steps |
| −0.0100 [−0.0125, −0.0073] | CE@6, norm_match − absmean | norm-match-20k vs horizon-resume | 20000 | same file ln 144 | stands: the 5k CE deficit becomes a lead |
| +0.1200; +0.1252 worse than plain | token K1−K6; paired CE | tul-norm-match (paid) | 5000 | same file ln 155-158 | qualified: the paid arm is worse on CE and "not TUL" |
| **+0.033** / K3−K6 +0.0055 | token K1−K6 | plain-panel-norm-match (PRELUDE entry) | 5000 | [failures/2026-09-09-arc-slot-loop-norm-match.md](../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md) ln 147 | stands: the entry confound |
| 0.47 drop, ends 0.2 worse | depth 1 → 6 CE drop entered from zero | plain-panel-norm-match | 5000 | same file ln 150-151 | qualified: dependence |
| −0.0000 / +0.0001; forecast +0.0067 | token K1−K6 | slot-loop-norm-match / slot-mux-norm-match | 5000 | same file ln 143-145 | stands: +0.164 and +0.263 behind plain |
| −0.0001 | token K1−K6 | slot-loop-mask-norm-match (mask, no MUX) | 5000 | [failures/2026-09-10-arc-slot-loop-mask-norm-match.md](../lab/experiments/failures/2026-09-10-arc-slot-loop-mask-norm-match.md) ln 210 | stands |
| +0.0009 | token K1−K6 | slot-mux-mask-norm-match | 5000 | [failures/2026-09-10-arc-slot-mux-mask-norm-match.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md) ln 219 | stands: removes the absmean mask dependence |
| +0.811 at offset 0; K1−K6 +0.0006 | worth(zero); token K1−K6 | slot-unpack-norm-match | 5000 | [failures/2026-09-09-arc-coda-reads-the-thought.md](../lab/experiments/failures/2026-09-09-arc-coda-reads-the-thought.md) ln 115-117 | stands: the coda reads z; z does not depend on depth |
| +0.1168; CE@6 +0.052 worse | token K1−K6 | slot-unpack-fixed-depth | 5000 | [failures/2026-09-10-arc-slot-map-levers.md](../lab/experiments/failures/2026-09-10-arc-slot-map-levers.md) ln 143 | qualified in-file: "depth dependence without depth value" (ln 167) |

**What we concluded then.** The ternary scale rule was the limiter of the plain loop, and
norm_match fixes it ([.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md](../.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md)).
On 20k the norm_match loop's earning grows and its CE overtakes absmean. On the slot loop
the rule changes nothing: "the rule acts on token states, not on slot states"
([failures/2026-09-09-arc-slot-loop-norm-match.md](../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md) ln 173-176).

**What corrected it later.** The 0.185 is a noise-entry number. Under the prelude entry,
which the slot loop and its plain ruler use, the same rule reads 0.033. Later filings that
compare the slot loop against "the plain loop's 0.185" compare across entries. The E19
depth-1 twin (+0.019 ahead at 5k) is the first reading of the instrument that, at 20k,
became the plain loop's strongest value reading (+0.0674, see B2).

---

### A7. Slot M-next levers, the toy, and credit assignment (2026-09-10 to 2026-09-11)

**What we tried and why.** Three diagnostics opened the era. The geometry audit asked
whether the core is muted on the 64-cell slot sequence. The gradient probe read the
backward per pass. The z-optimize probe fitted a free slot state to the frozen coda. Then a
toy study on the 3070 built a task that needs iteration and varied the loss attachment. Its
winner, the staged target (own span at inner passes, next span at the exit), transferred
to the real tree as `slot-mnext-staged`. Every other lever ran one factor at a time on the
M-next ruler: progressive loss, per-pass LoRA, MUX on every pass, gradient-conditioned
passes (gradpass), noise entry, a Parcae core, the fixed-point term off, a re-read of the
prelude tokens, wider writes, a bag-mean seed, and the HCA fix.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| 0.0015 / 0.1074 / 0.0185 | K0−K6 (entry cell vs exit cell) | slot-mux-norm-match / slot-unpack-free / slot-mnext-noise-entry | 5000 | [results/2026-09-10-slot-geometry-audit/README.md](../lab/experiments/results/2026-09-10-slot-geometry-audit/README.md) ln 311-313 | stands: "The core is NOT muted" (ln 355) |
| 0.65 / 2.94 / 247x vs 1.04 / 2.78 / 161x | one-pass movement, slot vs token, same weights | same three | 5000 | same file ln 358 | stands |
| 0.520 | cancellation ratio | slot-mux-norm-match | 5000 | [results/2026-09-10-slot-gradient-probe/README.md](../lab/experiments/results/2026-09-10-slot-gradient-probe/README.md) ln 120 | stands; its reading ("passes fight") inverted by the toy |
| 0.016 → 0.523 by pass | token-CE cotangent share | slot-unpack-noise-entry | 5000 | same file ln 165 | stands: a 33x ramp under the noise entry |
| **−0.9442** / **−2.6258** | fitted z CE − loop z CE (hindsight) | slot-mux-norm-match / slot-unpack-norm-match | 5000 | [results/2026-09-10-slot-z-optimize/README.md](../lab/experiments/results/2026-09-10-slot-z-optimize/README.md) ln 139, 160 | **retracted** as headroom: hindsight; a causal fit on slot-spandec-strict reads +0.2517 worse than the loop's z ([failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) ln 449-453) |
| 5/5 vs 2/5 | chain solved, staged vs exit-only | toy, strict compose | 4000 | [lab/toy_slot_loop/WRITEUP.md](../lab/toy_slot_loop/WRITEUP.md) ln 136-137 | stands (toy) |
| 5/5 | chain solved, fixed-point term 1.0 | toy | 4000 | same file ln 193 | stands (toy) |
| **+0.0670**; own CE 6.06 → 3.36 at pass 3; pass-6 cosine −0.22 | forecast (next) K1−K6 | slot-mnext-staged | 5000 | [successes/2026-09-10-arc-slot-mnext-staged.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md) ln 212, 232 | qualified in-file: the exit forecast equals the ruler's (6.7698 vs 6.7724); token K1−K6 +0.0020 |
| +0.0637; tokens +0.0023; next K3−K6 +1.850 → +2.080 | forecast K1−K6 | slot-mnext-staged-20k | 20000 | [successes/2026-09-10-arc-slot-mnext-staged-20k.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged-20k.md) ln 266-269 | qualified: 0.161 behind plain, flat from 10k (ln 255, 311) |
| **+0.0236**; exit 0.017 better than ruler; 0.366 of 0.385 in pass 1 | forecast K1−K6 | slot-mnext-gradpass | 5000 | [failures/2026-09-10-arc-slot-mnext-gradpass.md](../lab/experiments/failures/2026-09-10-arc-slot-mnext-gradpass.md) ln 296, 306, 309 | stands as "one descent step"; token K1−K6 +0.0014 |
| +0.0145; K3−K6 0.0000; state expands 2.2x | forecast K1−K6 | slot-mux-fixed-point-off | 5000 | [failures/2026-09-10-arc-slot-mux-fixed-point-off.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-fixed-point-off.md) ln 225, 240 | stands: the term holds the exit, not the depth |
| +0.0604; tokens +0.0033; z-opt entry +0.0235 | forecast K1−K6 | slot-mnext-staged-mask | 5000 | [failures/2026-09-10-arc-slot-mnext-staged-mask.md](../lab/experiments/failures/2026-09-10-arc-slot-mnext-staged-mask.md) ln 193, 221 | stands; CE@6 4.4373 vs ruler 4.3290 |
| +0.0001 / +0.0000 / +0.0006 | token K1−K6 | mux-every-pass / per-pass LoRA / progressive | 5000 | failures files of 2026-09-10 (ln 223 / 212 / 171) | stands: flat |
| +0.0001; forecast +0.0019 | token K1−K6 | slot-mnext-parcae-core | 5000 | [successes/2026-09-10-arc-slot-mnext-parcae-core.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-parcae-core.md) ln 291 | stands: "the core is exonerated" |
| +0.0004 / +0.0001 / +0.0005 | token K1−K6 | hca-fix / prefix4 / bagmean | 5000 | 2026-09-10 successes files (ln 191 / 235 / 199) | stands: flat |
| +0.0003 | token K1−K6 | slot-unpack-reread | 5000 | [failures/2026-09-10-arc-slot-reread.md](../lab/experiments/failures/2026-09-10-arc-slot-reread.md) ln 76 | stands |

**What we concluded then.** The core is not muted and gradient reaches every pass. The
failure is credit assignment: the passes' updates disagree, and "the coda's CE with the
loop's exit equals its CE with the loop's entry". The toy said the attachment matters and
that low cancellation is healthy. On the real tree the toy's signatures transferred (a
forecast K-curve, a negative-cosine pass) but no value reached the exit. The hindsight z
fit said "the reader is not the bottleneck". By the end of the era the hypothesis moved to
the target: what the slot is asked for.

**What corrected it later.** The z-optimize "headroom" is retracted: a causal fit is worse
than the loop, so the 0.9 to 2.6 nats measure the coda's capacity, not reachable headroom.
Note the retraction was measured on a different arm (slot-spandec-strict) than the
hindsight numbers. The staged arm's forecast curve held flat to 20k and never paid CE.
gradpass's second-order form and `tg_restrict` twin were proposed and never run
([docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md), section 4).

---

### A8. The span budget, the span decoder, and strict geometry (2026-09-11 to 2026-09-12)

**What we tried and why.** First, measure the ceiling: how many nats of web text live
across a span boundary. Second, replace the MUX's bag-of-tokens target with a span decoder
that predicts the whole next span from the slot state. That moved the slot channel, and
the gain came through slot cells the coda reads directly, bypassing the loop. Third,
close that bypass. Strict geometry makes the loop the only cross-span channel. Seven arms
varied the coda's reach (`prev`: cell j−1 only), the loop's reach (`loop_reach` 1 or 2),
the target horizon, and an oracle target.

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| **0.3994 [0.3838, 0.4162]** | span budget, paired CE (span − full), depth 6 | budget-web-span vs budget-web-full | 5000 | [failures/2026-09-11-arc-span-budget.md](../lab/experiments/failures/2026-09-11-arc-span-budget.md) ln 311 | stands; still rising (0.1633 at 2500, ln 329) |
| +0.0279 / +0.0301 | token K1−K6 | budget-web-full / budget-web-span (plain, prelude entry) | 5000 | same file ln 339-340 | stands: "A context-starved model does not loop more" (same file ln 382-383) |
| −0.0720 [−0.0748, −0.0698] | paired CE, spandec-mask − mux-mask | slot-spandec-mask | 5000 | [successes/2026-09-11-arc-span-decoder.md](../lab/experiments/successes/2026-09-11-arc-span-decoder.md) ln 291 | stands: a channel gain, not a loop gain |
| ≤ 0.0013 (spandec-mask 0.0007) | token K1−K6 | every spandec arm | 5000 | same file ln 296-298 | stands: flat |
| 0.115 → 0.182 | all-slot worth | mux-mask → spandec-mask | 5000 | [.agents/notes/proposed/architecture/2026-09-11-span-decoder-target.md](../.agents/notes/proposed/architecture/2026-09-11-span-decoder-target.md) ln 221; results table ln 307, 379 | stands |
| 0.053 → 0.061 (disjoint CIs); CE +0.0068 | prefix-write worth, bin 8-15 | spandec-mask → spandec-chain-mask | 5000 | [successes/2026-09-11-arc-span-decoder.md](../lab/experiments/successes/2026-09-11-arc-span-decoder.md) ln 360, 379-380 | stands, small |
| +0.0016; worth 0.1865; CE −0.0001 vs mask | token K1−K6 | slot-spandec-strict | 5000 | [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 286 | stands: closing the bypass costs 0 |
| **+0.0163 [+0.0153, +0.0173]**; K3−K6 +0.0042; CE +0.0023 vs prev | token K1−K6 | slot-spandec-strict-prev-reach1 | 5000 | same file ln 289 | qualified: restriction; "carries history, not a thought" (ln 249-251, 342-344) |
| **+0.0127 [+0.0119, +0.0136]**; K3−K6 +0.0028; CE −0.0086 [−0.0109, −0.0063] vs prev, −0.0004 vs strict | token K1−K6; paired CE | slot-spandec-strict-prev-reach2 | 5000 | same file ln 290 | qualified: restriction; the one reach arm whose reach became CE value, but only to parity with strict |
| −0.014 to −0.028 | prev-reach1 vs its own depth-1 read, by offset | prev-reach1 | 5000 | same file ln 294-295 | stands |
| +0.1195 | paired CE, oracle target − strict | slot-spandec-strict-oracle | 5000 | same file ln 292 | stands: the oracle trajectory target is closed |
| 4 of 11 | predictions held | the panel | 5000 | same file ln 321 | stands |

**What we concluded then.** The budget is large (0.40 nats) and not front-loaded. A better
target fills more of it through the slot channel, but the loop does not do the work. In
strict geometry, "depth use comes from reachability, not from the target": every arm with
coda reach "all" reads K1−K6 ≤ 0.0016; the two reach-limited arms read 0.0163 and 0.0127
at CE parity ([failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 329-336). Wolfe's
reading of reach 1: "It proves that TUL can work for sure. But the amount of blindness here
is concerning" (ln 249-251). The reach arms are "a relay, not refinement, and are not a
design" ([.agents/notes/proposed/architecture/2026-09-12-strict-slot-geometry.md](../.agents/notes/proposed/architecture/2026-09-12-strict-slot-geometry.md) ln 187-188).

**What corrected it later.** See Part B. The depth-1 twin readings on the bypass and strict geometries are in B1 and B8. The hop staircase (B5) confirmed that reach-1 content arrives on schedule, and LXTUL-R (B8) showed the relay is a refund of a tax. Wolfe closed the restriction lane on 2026-09-22 ([.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md)). The strict run's logged slot rank 5.7598 was a bare-front probe artefact (B2). The matched-compute reading of 2026-09-12 (slot arm 0.2536 behind a plain depth-1 control) is carried in B1.

## 5. Part B eras (B1 to B9)

Author: hist-B. Range: every filing dated 2026-09-12 to 2026-09-23 except the strict-geometry panel (A8). "K1-K6" is the token K-curve on the same 480 rows. The slot-loop yardstick is about +0.002; the floor across twelve arms is [-0.0001, +0.0033]. The filings use two sign conventions for the depth-1 twin; this part reports "the looped arm is better by X". Unfiled GK numbers come from the vlt thread `lxtul-fan4` (entries 139 to 141), the GK prereg amendments, and the probe JSONs named in the rows.

### B1. 2026-09-12: give the passes an energy, a plan, or a trained core

**What we tried and why.** The strict geometry (A8) made the loop's write the only
cross-span channel. The channel was worth about 0.19 nats and the passes still read
K1-K6 +0.0016. Four panels on the same day asked four different "missing input"
questions, all on the strict partner `slot-spandec-strict`:

1. Does the pass lack a hard energy? The latent-z gradient arms (`egrad-recon`,
   `egrad-disc`) injected the gradient of an energy that one pass could not saturate.
   A linear probe gate asked first whether the slot state carries scoreable structure
   at all.
2. Does the pass lack a target that differs by pass? The per-pass plan arm grades pass t
   on spans s+1 to s+t. The parallel-coda arms (`codaspan`, J 8 and 32) ask the coda to
   emit a whole span from the cell with no token path.
3. Does the core lack training? The core-token auxiliary (`coretok`) trains the six core
   blocks on the token CE as well. A within-context critic grades pass t against pass t-1.
4. Is a depth-1-trained twin as good? `slot-spandec-norecur` (trained and read at depth
   1) is the control that separates "the loop iterates usefully" from "the slot cell is
   useful".

The same day two reference readings landed: the matched-compute plain control (part 2 of
the span-decoder filing, 03:27 on 09-12) and the math panel re-run under `norm_match`
(filed 09-12, run 09-13).

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| AUC 0.611 (entry) / 0.638 (pass 1) / 0.637 (exit), null p95 0.514 | linear probe for "next-span CE below median", mean reduction | slot-spandec-mask | 5000 | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | stands; the whole rise is at pass 1 |
| AUC 0.633 / 0.635 / 0.629 (falls) | same | slot-mux-norm-match (ruler) | 5000 | same | stands |
| looped better by 0.0235 [0.0209, 0.0263]; nil at offset 0, 0.031 at offsets 8+ | paired CE, arm at depth 6 vs d1 twin at depth 1 | slot-spandec-mask vs norecur (bypass geometry) | 5000 | same | qualified: bypass geometry only; the strict twin reads 0.0020 [-0.0006, +0.0044] in the same file |
| ruler better by 0.0020 [-0.0006, +0.0044] (the table labels it "ruler@6 - norecur@1 = +0.0020") | same | slot-spandec-strict vs strict-norecur | 5000 | same | qualified: CI crosses 0; re-read in B8 on a 20k-schedule pair as a constant 0.0075 ([failures/2026-09-22-arc-slot-token-like-read.md](../lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md)) |
| K1-K6 +0.0007 / +0.0019; K3-K6 -0.0003 / +0.0002 | token K-curve | egrad-recon / egrad-disc | 5000 | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | stands |
| +0.0495 / +0.0081 worse | paired CE vs ruler, depth 6 | egrad-recon / egrad-disc | 5000 | same | stands |
| +0.01353 / +0.01299 / +0.01492 | write contribution (ce_entry - ce_loop, z-opt) | strict / egrad-disc / egrad-recon | 5000 | same | stands; point estimates, no CI |
| -1.5516 (hindsight) | fitted z minus loop z (ce_zopt - ce_loop) | strict | 5000 | same | retracted as headroom: the causal fit is +0.2517 [+0.2265, +0.2730] WORSE than the loop's z ([results/2026-09-12-instruments/causal_fit_slot-spandec-strict_g1.txt](../lab/experiments/results/2026-09-12-instruments/causal_fit_slot-spandec-strict_g1.txt)) |
| per-pass cotangent share 0.16 to 0.19, flat; slot_cot_clip binds 0.000; core gets about 0.3 of the prelude's gradient norm | gradient probe | egrad arms | 5000 | same | stands |
| K1-K6 +0.0008 / +0.0012 / +0.0017 | token K-curve | perpass / codaspan / codaspan32 | 5000 | [failures/2026-09-12-arc-objective-arms.md](../lab/experiments/failures/2026-09-12-arc-objective-arms.md) | stands |
| +0.0172 / +0.0415 / +0.0255 worse | paired CE vs ruler, depth 6 | perpass / codaspan / codaspan32 | 5000 | same | stands |
| t1 4.513, t2 4.614, t3 4.646, t6 4.706 (rises) | per-pass training CE `spandec_pass_t` | perpass | 4500-5000 | same | qualified: each pass is scored on a different target (amendment 1) |
| pass_h1 -0.0200, pass_h6 -0.0075, exit -0.0102 | identical-target grid K1-K6, first run | perpass | 5000 | same | retracted: the instrument ran a bare `_tul_front`, so the strict prelude ran unrestricted; fixed at 7a24adf (same file) |
| pass_h1 +0.0000, pass_h6 +0.0015, exit +0.0013; ruler exit +0.0031 [+0.0026, +0.0036] | identical-target grid K1-K6, corrected | perpass; strict | 5000 | same | stands |
| +0.02195 vs ruler +0.01353 (1.6x); perpass +0.01135 | write contribution | codaspan | 5000 | same | stands; the P-6 bar (excess 0.010) missed by 0.0016 |
| 6.62 nats/token vs the coda's 4.39 | parallel span heads CE | codaspan | 5000 | same | stands |
| K1-K6 +0.0005 [+0.0002, +0.0007]; K3-K6 -0.0003 | token K-curve | slot-spandec-strict-coretok | 5000 | [failures/2026-09-12-arc-core-token-and-critic.md](../lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md) | stands |
| better by 0.0684 [0.0657, 0.0710] | paired CE vs strict, depth 6 | coretok | 5000 | same | stands; the dense objective on the shared prelude and coda, not the loop (K-curve and worth unchanged) |
| aux path 0.0164 better than shipped path; on the ruler the aux path is 1.1046 worse | aux token path through the core vs shipped path | coretok; strict | 5000 | same | stands |
| K1-K6 +0.0012; critic_agree 0.521; critic loss 0.689 (ln 2 = 0.693) | K-curve; critic accuracy | slot-spandec-strict-critic | 5000 | same | stands |
| 0.0012 to 0.0024 nats | measured worth of one pass (critic_gap_traj) | critic | 5000 | same | stands |
| gap +0.00145 [+0.00087, +0.00202]; win rate 0.477 | critic direction vs random, 10 % RMS step | critic | 5000 | same | stands; seven times under the 0.01 bar |
| slot arm behind by 0.2536 [0.2408, 0.2676]; mux-mask behind by 0.3253 | paired CE, slot arm at depth 6 vs plain depth-1 control at 14 block-passes per token | slot-spandec-mask vs plain-coda-matched | 5000 | [successes/2026-09-11-arc-span-decoder.md](../lab/experiments/successes/2026-09-11-arc-span-decoder.md) (part 2) | stands as a 5k reading; not a ranking (see "later") |
| K1-K6 +0.0647 [+0.0602, +0.0691] (absmean twin +0.0238) | token K-curve | oly-notul-nm (plain) | 6000 | [failures/2026-09-12-arc-math-under-norm-match.md](../lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md) | stands; Wolfe set math aside on 2026-09-22 ([docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md)) |
| K1-K6 +0.0111 [+0.0085, +0.0136]; K3-K6 +0.0024 | token K-curve | oly-spandec-strict | 6000 | same | stands; same caveat |
| K1-K6 +0.0049 [+0.0047, +0.0052]; plain +0.0011 | token K-curve | sud-spandec-strict; sud-notul-nm | 6000 | same | stands; same caveat |
| slot behind plain by +0.2161 (Olympiad), +0.2502 (Sudoku) | paired doc CE at depth 6 | oly/sud slot vs plain | 6000 | same | stands |
| -0.1393 / -0.1903 (zeroing HELPS) | worth(zero) | oly / sud spandec-strict | 6000 | same | stands; the damage is at offset 0 (-1.9 to -2.2 nats) |

**What we concluded then.** Each panel closed its lane. The energy arms moved the
direction of the passes (critic AUC 0.632, feature 0.019 of the state) and not the
amount the cell gains. The per-pass target was met by pass 1; its rising training
ladder was the changing target. The core-token arm made the six blocks a competent
token map (1.10 nats better than an untrained path) and the slot passes through the same
blocks still read 0.0005. The critic learned from ties. The shared reading, in the
core-token filing's words: "What is missing is a state the coda's loss is steep in, not
a pass that can move." The next lane was "the READ and the WRITE", and "a matched-compute
one-pass slot model priced against the looped one". The d1 twin said training depth
moves CE on the bypass geometry (0.0235) while eval depth does not (K1-K6 0.0007).

**What later corrected it.**

- The fitted-z headroom (-1.5 nats) was hindsight. The causal fit reads 0.25 worse than
  the loop's own z ([results/2026-09-12-instruments/causal_fit_slot-spandec-strict_g1.txt](../lab/experiments/results/2026-09-12-instruments/causal_fit_slot-spandec-strict_g1.txt),
  measured 09-13, quoted in the objective-arms filing). The synthesis note lists it as a
  downgraded instrument.
- The d1-twin effect is geometry-dependent. The strict twin at 5k crosses zero. In B8 a
  matched 20k-schedule twin pair reads a constant 0.0074 to 0.0083 from 5k to 20k, with
  no growth.
- The matched-compute reading (plain ahead by 0.25) stands as a number. A standing call
  later removed it as a verdict instrument: score arms on the token K-curve and paired
  depth-6 CE, "Do not score them on matched-compute nats"
  ([.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md),
  standing calls). The note does not date the call; I did not find the date in a repo file.
- The math positives (Olympiad +0.0111, Sudoku +0.0049) are the largest clean slot-loop
  K-curves of this era. No later arm ran on a math corpus. The positives ledger calls
  them "never built on"; its header records Wolfe's 2026-09-22 call that math was a red
  herring.
- The first identical-target grid was void because the instrument rebuilt the front
  without the strict relation. The same defect produced the slot rank 5.76 in B2.

### B2. 2026-09-13: the reader, the write's rank and width; the plain ladder; the core on tokens

**What we tried and why.** B1 pointed at the READ and the WRITE. A measured "defect" set
the day's agenda: the Thought Register prereg quoted the strict partner's row of slot
states at effective rank 5.7598 of 1024 and pairwise cosine 0.7104, "about six
directions". The day's arms attack that from both sides, all one factor from
`slot-spandec-strict`:

- The write's capacity: four mutable cells per span (Thought Register `m4`, `sameinit`),
  a discrete write of K VQ codes (`vq8`, `vq4`), a trajectory write that gives the coda
  every pass instead of the exit (`traj`, with width controls `trajrep` and `entryexit`).
- The reader's capacity: a non-shared 4-block stack between the loop and the cells
  (`cond4`), a one-block coda (`ultralight`), a mature coda (`bootstrap`: plain for 5k
  steps, then the slot loop).
- The target's distance: grade z on span s+2 or s+3 instead of s+1 (`off2`, `off3`).
- Two plain-loop questions: the depth ladder (what core depth ships, at 20k) and an eval
  probe that runs the coretok model's core over tokens at forced depths (the
  `loop_reads_tokens` training arm was cut by Wolfe because it costs about 44 block-passes
  per generated token and abandons "think once, decode cheap").

Several of these arms were filed on 09-13 and scored later (off2 09-19, off3, vq8, vq4 and
bootstrap 09-20/21, the ladder 09-19). They are placed here by their prereg date.

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| 5.7598 / 0.7104 | row slot rank / pairwise cos | slot-spandec-strict | 5000 | [failures/2026-09-13-arc-thought-register.md](../lab/experiments/failures/2026-09-13-arc-thought-register.md) (the quoted defect) | retracted: bare-front probe before 7a24adf; the corrected trainer-recipe reading is 13.8466 / 0.5201 ([results/2026-09-13-rank-anatomy/README.md](../lab/experiments/results/2026-09-13-rank-anatomy/README.md), discrepancy.md) |
| K1-K6 +0.00197 [+0.00163, +0.00233]; K3-K6 -0.00029 | token K-curve | slot-register-m4 | 5000 | [failures/2026-09-13-arc-thought-register.md](../lab/experiments/failures/2026-09-13-arc-thought-register.md) | stands |
| cell rank 1.2445 of 4 (init floor 1.2383), cell cos 0.9432 (init 0.8443); row rank 10.57 | register cell geometry | slot-register-m4 | 5000 | same | stands |
| 0.0217 better [0.0191, 0.0244]; m4 vs sameinit 0.0070 better | paired CE, depth 6 | m4 vs strict; m4 vs sameinit | 5000 | same | qualified: prefix width and packing (the filing names it "a wider prefix") |
| +0.00719 vs strict +0.00314 | span decoder's own K1-K6 | slot-register-m4 | 5000 | same | qualified: the decoder reads the mean of four cells; a readout property, not pass computation (same file) |
| +0.00026 / +0.00062 | token K-curve | strict-cond4 / register-m4-cond4 | 5000 | [failures/2026-09-13-arc-cond4-reader.md](../lab/experiments/failures/2026-09-13-arc-cond4-reader.md) | stands |
| +0.0057 / +0.0159 worse | paired CE vs partner | strict-cond4 / register-m4-cond4 | 5000 | same | stands |
| +0.00169; paired +0.0073 worse; worth 0.1823 vs 0.1865; worth(zero) at offset 0 +0.9493 vs +0.7567 | K-curve; paired CE; worth | slot-ultralight-coda | 5000 | [failures/2026-09-13-arc-ultralight-coda.md](../lab/experiments/failures/2026-09-13-arc-ultralight-coda.md) | stands |
| K1-K6 +0.0342 | token K-curve | traj (trajectory prefix) | 5000 | [failures/2026-09-13-arc-trajectory-prefix.md](../lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md) | retracted in the same file: at forced depth d the write fills d cells and pads the rest; the same-width `trajrep` reads +0.0019 |
| +0.0019 / +0.0014 | token K-curve | trajrep / entryexit | 5000 | same | stands |
| ruler to traj / trajrep / entryexit: 0.0271 / 0.0302 / 0.0311 better; entryexit - trajrep -0.0012 [-0.0035, +0.0011] | paired CE, depth 6 | trajectory panel | 5000 | same | qualified: six cells beat two whatever they hold; width, not loop |
| 0.0011 | pass-embedding gradient share of the prefix | traj | 5000 | same | stands |
| K1-K6 +0.0023 / +0.0009; paired 0.0035 better [0.0012, 0.0057] / -0.0008 [-0.0032, +0.0016] | K-curve; paired CE vs strict | strict-off2 / strict-off3 | 5000 | [failures/2026-09-13-arc-register-reader-and-downstream-target.md](../lab/experiments/failures/2026-09-13-arc-register-reader-and-downstream-target.md) | stands |
| +0.2599 / +0.3168 zeroed; +0.0358 / +0.0513 shuffled | span decoder CE with z zeroed / shuffled | off3 / off2 | 5000 | same | stands; the specific content of z is worth 0.04 to 0.05 to its own decoder |
| perplexity 105.7; row rank 35.09; cell rank 6.72 of 8; K1-K6 +0.0015; paired +0.0250 worse; worth 0.1397 | quantizer; geometry; K-curve; paired CE; worth | strict-vq8 | 5000 | [mixed/2026-09-13-arc-discrete-thought-vq.md](../lab/experiments/mixed/2026-09-13-arc-discrete-thought-vq.md) | stands |
| rank 17.13; K1-K6 +0.0017; paired +0.0577 worse; vq8 - vq4 -0.0328 [-0.0358, -0.0300] | same | strict-vq4 | 5000 | same | qualified: the width confound was resolved by pk8 (B7): the quantizer costs 0.059 beyond its width |
| K1-K6 +0.0019 [+0.0016, +0.0022]; K3-K6 +0.0002 | token K-curve | slot-strict-bootstrap | 5000 TUL steps | [failures/2026-09-13-arc-plain-then-tul-bootstrap.md](../lab/experiments/failures/2026-09-13-arc-plain-then-tul-bootstrap.md) | stands |
| worth(zero) 0.2287 vs 0.1865; offset 0 0.9016 vs 0.7567 | worth | bootstrap vs strict | 5000 | same | qualified: a better seed from the mature backbone; the passes still read +0.0002 (same file) |
| aux K1-K6 +0.0102 [+0.0093, +0.0110]; K3-K6 +0.0012; passes 1 to 2 give 0.0071 | K-curve of the coretok core run over tokens (eval probe) | slot-spandec-strict-coretok | 5000 | [failures/2026-09-13-arc-loop-reads-tokens.md](../lab/experiments/failures/2026-09-13-arc-loop-reads-tokens.md) | stands (hand-checked in [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md)); 20x the slot path's 0.0005 on the same blocks |
| d1 behind top by 0.0674 [0.0647, 0.0701] | paired CE, own training depth | plain `d1` vs `norm-match-20k` | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | stands at matched steps (hand-checked in the ledger); at matched wall clock d1@20k beats top@10k by 0.1728 [0.1682, 0.1777] |
| -0.1417 / +0.0018 / +0.0387 / +0.0478 / +0.0705 / +0.1702 [+0.1672, +0.1729] | own K1-K6 by draw: d1 / d2 / d3 / p3 / p4 / top (mean 6) | plain ladder | 20000 | same | qualified: a dial on depth dependence; every sampled rung from mean 3 up ties the top rung's final CE inside the 0.0237 seed spread |
| p3 vs top -0.0013 [-0.0036, +0.0010] at depth 3, -0.0019 at 6; p4 -0.0051 [-0.0076, -0.0027] at 6 | paired CE | p3 / p4 vs top | 20000 | same | stands; ties inside 0.024 |
| +0.2294 / +0.5259; d3fixed vs d3fixed-s2 0.0237 [0.0208, 0.0266] | own K1-K6; seed pair | d3fixed / d6fixed | 20000 | same | qualified: fixed depth is a tied-weight deep net, taken out of the ship set by the 2026-09-14 correction (same file) |
| slot arms 10.59 block-passes and 64.4 keys per token, 0.385 nats behind plain-depth1 (4.3466 vs 3.9612); budget-web-span 0.092 behind strict at 4.2x the passes | memory-cost table | seven 5k arms | 5000 | [planned/2026-09-13-arc-tul-as-memory.md](../lab/experiments/planned/2026-09-13-arc-tul-as-memory.md) | stands; the confound arm it plans has no results |

**What we concluded then.**

- Capacity is not the lever. Four cells behave as one (rank 1.24 of 4) and the seed does
  not matter (distinct vs same differ by 0.008 rank units). The row rank FELL from 13.85
  to 10.57.
- The reader is not the gate. A one-block coda leans on z no harder (worth 0.1823) and a
  4-block stack after the loop makes pass 1 matter less. cond4 was "the third arm in a day
  to move a rank or separation instrument with no downstream effect; rank of z is not the
  binding constraint".
- "The coda only sees the exit" is retired. The trajectory write is the worst of three
  contents at equal width. Prefix width is a real 0.03-nat lever, "to be named as width
  and not as loop contribution".
- Where z is graded does not move what the coda gets. The discrete write carries less at
  every bin, and rank by construction (35 on vq8) leaves K3-K6 at 0.0000.
- A mature coda makes a better seed and a more used z, and the passes are unchanged.
  "Reader maturity is closed as a confounder."
- The same six blocks earn 0.0102 on tokens and 0.0005 on slot states. That puts the
  fault in what the slot state gives the core to work on, not in the core.
- On the plain loop: `p3` ships (level with mean 6 at 69 % of the wall clock). The
  sampled mean is a dial on depth dependence, not on final CE. "At 20k on web text the
  mechanism has no CE customer."

**What later corrected it.**

- The rank that motivated the register was retracted the same day. The bootstrap's P-6
  and the register's P-2 bars were written against the retracted number; both filings
  record it. The instrument lesson ("instruments must use the model's TG kwargs") covers
  this and the void horizon grid of B1.
- The depth ladder's clamped and fixed rungs were taken out of the ship set on 2026-09-14
  ("fixed depth is not sampled depth", in the ladder filing and the synthesis note's
  standing calls).
- The vq panel's width confound was resolved on 2026-09-21 by the pk8 ruler (B7).
- The register reader arm never ran (a protocol failure, as its own filing says); the
  LXTUL fan (B6) took over the "K states per span" question.
- "The reader is not the limit" was true for the capacity of the strict coda. It was
  re-opened for a different reader in B4 (a frozen coda reads an LCTUL cell at -4.6 nats;
  the same coda adapted reads +0.177) and in B6 (write-all, where the coda's per-token read
  selects among candidates).

### B3. 2026-09-14: zero-training diagnostics and horizon-indexed passes

**What we tried and why.** The 2026-09-13 literature mining
([docs/references/looping-depth/2026-09-13-lit-mining/](references/looping-depth/2026-09-13-lit-mining/)) proposed four cheap instruments
that each give a different mechanistic reason iteration could matter or legitimately not:
the AA score (is the exit path-independent of the entry), the Jacobian spectral gap, the
attention sink mass per pass, and a basin map of the exit under entry perturbation. They
ran on the strict slot loop and the earning plain loop with no training. The same batch
ported LoopMTP (arXiv 2608.03624) to the slot loop: pass t gets a horizon-indexed target
(the tied-embedding mean of span i+t) and a gated readout over all six passes, against a
fixed-depth-6 control. A token-loop LoopMTP ladder was preregistered as the ship
question.

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| AA(noise) 0.8187 / AA(swap) 0.8644 | AA score | strict slot loop | 5000 | [failures/2026-09-14-arc-loop-diagnostics.md](../lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md) | stands |
| AA(noise) 0.9998 / AA(swap) 0.9777 | AA score | plain norm_match loop (notul_norm_match_20k) | 5000 | same | stands; the earning loop is the attractor |
| sigma1/sigma2 median 1.0521 (strict), 1.1051 (plain) | Jacobian spectral gap | both | 5000 | same | stands; no gap on either |
| max relative deviation, passes 2 to 6: 0.09 % (strict), 0.67 % (plain) | sink mass | both | 5000 | same | stands |
| 4 of 6 pairs flat; 2 of 6 at entropy about 1.6 nats, 59 to 92 % of grid points differing | basin map | strict | 5000 | same | stands, not followed; "not a depth-earning signal until a paired CE reading moves with them" (coordinator reading, same file) |
| K1-K6 +0.0040 [+0.0036, +0.0044]; K3-K6 +0.0002 | token K-curve | horizon-arm | 5000 | [failures/2026-09-14-arc-horizon-passes.md](../lab/experiments/failures/2026-09-14-arc-horizon-passes.md) | qualified: paired vs its control -0.0001 [-0.0023, +0.0025]; the gate reads 83 % from pass 1 |
| K1-K6 +0.278; K3-K6 +0.0086 | token K-curve | horizon-fixed6 (control) | 5000 | same | retracted as a depth reading: trained only at depth 6, out of distribution at every other eval depth (same file) |
| every pass 2 to 6 reads 0.638 to 0.645 cosine; the six horizon targets within 0.01 cosine of each other | pass-by-horizon cosine matrix | horizon-arm | 5000 | same | stands; the targets are degenerate on web text |
| |h| 747 to 2039, cosine to depth 1 falls to 0.24; horizon-arm |h| 340 to 391, cosine 0.88 | state probe | fixed6 / horizon-arm | 5000 | same | stands; a fixed draw turns the loop into a growing chain the coda does not use |
| no results filed | LoopMTP on the token loop | loopmtp rungs | - | [planned/2026-09-14-arc-loopmtp-token-loop.md](../lab/experiments/planned/2026-09-14-arc-loopmtp-token-loop.md) | open: an unfiled probe output exists at [results/2026-09-14-loopmtp-token-loop/](../lab/experiments/results/2026-09-14-loopmtp-token-loop/) (untracked); I did not read it as a result |

**What we concluded then.** The instruments did not agree with "one pass does all the
work". The strict loop is not an attractor and is multistable at a minority of slots.
The plain loop, which earns, is an attractor. So "attractor-ness goes WITH earning here,
and the spectral gap separates nothing". The LoopMTP port gave the passes no different
jobs because a span-mean embedding barely depends on the horizon t. "Any future per-pass
target must be shown non-degenerate (M's columns separated) before an arm is queued."

**What later corrected it.** Nothing retracts these readings. The basin pairs were never
followed. The plain loop's attractor reading reappears in B5 as evidence that a
contractive map cannot hold alternatives apart from one perturbed entry (sample oracle),
and that reading was itself corrected by Wolfe on 2026-09-19 (B5). The token-loop LoopMTP
ladder has no filed result.

### B4. 2026-09-14 to 2026-09-18: LCTUL, a code in the slot

**What we tried and why.** Wolfe, 2026-09-14: "if I just had ground truth latent values I
could do anything" ([failures/2026-09-14-arc-tul-code.md](../lab/experiments/failures/2026-09-14-arc-tul-code.md)). TUL-Code (renamed LCTUL on
09-16) gives each slot a ground-truth code: an encoder E reads the span the slot precedes
and writes its code into the cells; the coda speaks the span from the code; the core is
trained as a flow-matching field from noise to the code and samples it in k Euler passes.
In this family the "loop" is the sampler, so the depth instrument is the k-curve: the
8-draw marginal CE at k Euler steps, k16 - k1 (negative means more steps help). Three
sub-lanes followed:

1. **The sampled code** (09-14 to 09-16): 5k and 20k arms, a frozen-E rollout arm, a
   renorm fix, classifier-free guidance, predictability pressure on E (`jepa`), a
   thinker-only run to 50k, Explorative Modeling (`xm`), LeJEPA with SIGReg, a discrete
   code with a masked denoiser (LCTUL-D), a 24-bit plan code scored on generation
   (`dplan`), and the LaDiR recipe.
2. **The regressed code target** (09-17): the ordinary strict slot loop regressed onto a
   frozen E's code of the next span (arm A), with a progressive-loss twin, a code-only
   twin, and a graded-continuation target.
3. **The reader** (09-17 to 09-18): arm A continued 10k steps with the coda unfrozen.

**Readings.** (One-draw and marginal CE are nats per token on 501,106 paired tokens or
96 probe rows as the file states.)

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| +0.705 [+0.702, +0.709] | one sampled code vs strict ruler, paired | tul-code | 5000 | [failures/2026-09-14-arc-tul-code.md](../lab/experiments/failures/2026-09-14-arc-tul-code.md) | stands |
| ce_tf 0.57, 3.8 nats below the ruler | coda CE with the TRUE code (reconstruction ceiling) | tul-code | 5000 | same | stands; a ceiling, not an LM reading |
| 1.7 to 1.9 | sample residual over code variance (2.0 = independent draw) | tul-code | 5000 | same | stands |
| k16 - k1 = -0.004 [-0.006, -0.003] | 8-draw marginal | tul-code | 5000 | same (addendum 09-15) | stands; "present, and negligible" |
| -1.25 (5k), -1.29 (10k); rollout1 -0.98, -0.42 | phase-2 marginal k-curve on a coda that trusts the code | tul-code-20k; rollout1 | 5000 to 10000 | [failures/2026-09-14-arc-tul-code-20k.md](../lab/experiments/failures/2026-09-14-arc-tul-code-20k.md) | qualified: flat within 5000 rollout steps once the coda discounts samples |
| +0.626 [+0.618, +0.634]; rollout1 +0.359; rollout1 - tul-code -0.268 | one draw vs ruler | tul-code-20k / rollout1 | 20000 | same | stands; the rollout coda wins by ignoring its cells |
| +0.014 [+0.012, +0.016] / +0.005 | 8-draw marginal k16 - k1 | tul-code-20k / rollout1 | 20000 | same | stands (more steps slightly worse) |
| 0 of 10, 1 of 10, ruler 4 of 10 | span samples keeping a topic thread | tul-code-20k / rollout1 / ruler | 20000 | same | stands |
| fork - parent +0.033; k16 - k1 -0.010 [-0.013, -0.007] | one draw; marginal | tul-code renorm fork | 20000 | [successes/2026-09-15-tul-code-renorm-r10k.md](../lab/experiments/successes/2026-09-15-tul-code-renorm-r10k.md) | stands; the renorm is a correctness fix, not a lever |
| k16 - k1 -0.005 [-0.007, -0.002]; one draw vs ruler +0.648 | marginal; paired | tul-code-cfg | 20000 | [failures/2026-09-15-tul-code-conditioned-thinker.md](../lab/experiments/failures/2026-09-15-tul-code-conditioned-thinker.md) | stands |
| k16 - k1 -0.017 [-0.021, -0.014]; one draw vs ruler +0.654 | marginal; paired | tul-code-jepa | 20000 | same | stands |
| k16 - k1 -1.88 [-2.00, -1.73] at 9.6 to 11.6 nats | marginal on a frozen phase-2 coda | tul-code-thinker | 50000 | [failures/2026-09-15-tul-code-thinker-only.md](../lab/experiments/failures/2026-09-15-tul-code-thinker-only.md) | qualified: "at a level (10-14 nats) that no coda can use" (same file) |
| k16 - k1 +0.016 [+0.014, +0.019]; one-draw curve 4.439 to 4.479 | marginal; one draw | tul-code-xm | 20000 | [failures/2026-09-15-tul-code-xm.md](../lab/experiments/failures/2026-09-15-tul-code-xm.md) | stands (every extra step costs) |
| k1 - k8 = -0.162 [-0.176, -0.146]; context share 0.02 % | marginal (more rounds WORSE); share of loss on the past | tul-code-d (LCTUL-D) | 5000 | [failures/2026-09-16-lctul-d-first-arm.md](../lab/experiments/failures/2026-09-16-lctul-d-first-arm.md) | stands |
| k4 - k1 -0.0136 [-0.0164, -0.0108] (15k), -0.0119 [-0.0155, -0.0082] (20k) | marginal | tul-code-dplan | 15000 / 20000 | [failures/2026-09-16-lctul-dplan-semantic.md](../lab/experiments/failures/2026-09-16-lctul-dplan-semantic.md) | qualified: "real but tiny, and it does not reach the generated text" (same file) |
| OWN@4 - OWN@1 +0.009 [-0.008, +0.028]; OWN@4 - SHUF +0.008 [-0.006, +0.023]; OWN@4 vs ruler -0.030 | semantic probe (MiniLM cosine) | tul-code-dplan | 20000 | same | stands |
| k16 - k1 +0.319 [+0.225, +0.428]; OWN - SHUF -0.006 | marginal; semantic | LaDiR stage 2 | 40000 | [failures/2026-09-16-lctul-ladir-chain.md](../lab/experiments/failures/2026-09-16-lctul-ladir-chain.md) | stands |
| one draw k1 4.585, k2 4.486, k16 4.501; marginal k16 - k1 -0.0006 [-0.004, +0.003] | one-draw curve; marginal | tul-code-lejepa | 20000 | [failures/2026-09-16-tul-code-lejepa.md](../lab/experiments/failures/2026-09-16-tul-code-lejepa.md) | qualified: "should be read on a second seed before it is called anything" (same file); the marginal is flat |
| exit cosine to code 0.004 to 0.144 by step 1500, 0.160 at 20k; l6 - l1 +0.0052 | per-pass cosine to the code | tul-code-target (arm A) | 20000 | [planned/2026-09-17-lctul-target-slot-loop.md](../lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md) | stands as a diagnostic; Wolfe's call that no arm passes or fails on a cosine (applied in [failures/2026-09-22-lctul-ema-target.md](../lab/experiments/failures/2026-09-22-lctul-ema-target.md)) |
| OWN - SHUF +0.0230 [+0.0051, +0.0421]; OWN - ZERO +0.0054 [-0.0167, +0.0283] | semantic probe | arm A | 20000 | same | stands; "the FIRST positive own-versus-foreign reading in the LCTUL family" |
| worth(zero) -4.608 (zeroing HELPS) | worth through the frozen coda | arm A | 20000 | same | qualified: a statement about this frozen reader (see the unfreeze row) |
| K1-K6 +0.0323 [+0.0236, +0.0415]; depth 16 -0.0959 | forced-depth CE through the frozen coda | arm A | 20000 | same; [successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md) | retracted as depth: the cell drifts to the corpus mean with depth and the frozen coda rewards blandness ([.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md](../.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md)) |
| -4.64 (frozen) to +0.177 (adapted) | cell worth, same weights | tul-code-target-uf | 30000 | [successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md) | stands as a reader result |
| OWN vs arm A +0.0290 [+0.0040, +0.0533]; OWN - ZERO +0.0179 [+0.0025, +0.0329] | semantic probe | uf | 30000 | same | stands |
| K1-K6 +0.0047 [+0.0038, +0.0058]; ORACLE 0.5645 to 0.1463 | K-curve; oracle semantic | uf | 30000 | same | stands; "Unfreezing the reader cut the K-curve about sixfold" |
| cos l6 - l1 +0.0073 (arm A +0.0051) | per-pass cosine | tul-code-target-prog | 20000 | [failures/2026-09-17-lctul-target-progressive.md](../lab/experiments/failures/2026-09-17-lctul-target-progressive.md) | qualified: about fifty times too small to matter (same file); a cosine |
| K1-K6 -0.1173 [-0.1304, -0.1040]; OWN - ZERO +0.0222 [+0.0045, +0.0407] | K-curve through the frozen coda; semantic | prog | 20000 | same | the K-curve is read as genericity (the cell got SHARPER); the semantic row stands |
| own cosine 0.99, shuffled 0.98 by step 2500 | code target cosines | tul-code-only (first draws) | 2500 | [failures/2026-09-17-lctul-code-only-ref.md](../lab/experiments/failures/2026-09-17-lctul-code-only-ref.md) (describing the void draws) | retracted: a frozen E fed by the live prelude is not a fixed target; both draws void |
| code cos 0.1674; l6 - l1 +0.0029; twin read shuf - own +0.1876, zero - own -4.6238 | code target; twin read | tul-code-only (frozen ref) | 20000 | same | stands |
| K1-K6 +0.0698, -0.0014, +0.0414, +0.0291 (5k to 20k) with token CE 12.64 above ln 49152 = 10.80 | forced-depth sweep | tul-code-only | 5000 to 20000 | same | retracted: the reader is worse than uniform; the instrument is void (same file) |
| cos to winner 0.5323, to truth 0.1462; winner to truth 0.1529; true rank 0.8784; best - worst -0.004 | graded target | tul-code-grade-l2 | 20000 | [failures/2026-09-17-lctul-graded-target.md](../lab/experiments/failures/2026-09-17-lctul-graded-target.md) | stands |
| l0 0.4012, l1 0.5235, l6 0.5310 | per-pass cosine to the winner | grade-l2 | 20000 | same | qualified: met in one pass (+0.0075 after pass 1) |

**What we concluded then.**

- A sampled code carries nothing span-specific. The sample sits near an unconditional
  draw on every code tried (1024-d at two noise levels, 72-bit, 24-bit). The past does
  not determine the next span's code on web text beyond the 0.40-nat cross-span budget.
  Rate trades against predictability (dplan).
- A regressed code is not a sampled code. Arm A gave the family its first positive
  own-versus-foreign reading (+0.023).
- The reader was the limit on whether the cell is USABLE: -4.6 nats frozen, +0.177
  adapted. It is not the limit on depth: unfreezing cut the K-curve sixfold, and the
  cosine to the own code peaks at depth 6, the training mean, on every arm.
- The predictable part of a fixed code is about 0.15 cosine whatever the front, found in
  one pass. A computed (graded) target is met in one pass too. "Ranking TEXT and ranking
  CODES are not the same ordering."

**What later corrected it.**

- Arm A's K1-K6 +0.0323 was retracted as genericity on 2026-09-18 (the implemented
  testing note above). The same artefact later explained the denoiser's +0.5575 (B7).
- The code-only draws were void because a frozen encoder on a live front is not a fixed
  target. LCTUL-J (B8) tried a moving EMA target on purpose; the target moved and nothing
  reached the reader.
- The unfreeze's "reader was the limit" was tested against sampling in B5 (the adapted
  reader recovers 1.5x the frozen reader's best-of-16, not more).
- cfg-tlow (B7) is the last sampled-code arm in this range; it moved the context share
  from 1 % to 1.9 % and left the ceiling at +0.64.

### B5. 2026-09-18 to 2026-09-19: hop distance, the toys, the sample oracle, the carry

**What we tried and why.** Two ideas from outside the slot-loop ledger entered. First, a
mechanical hypothesis: a looped transformer earns depth through GATHER, one attention hop
per pass, so a pass can only pay for a token whose needed content is more than one hop
away. The hop-distance probe bins tokens by the span they depend on most and plants
rare-token copy pairs g spans back. It ran on three 5k checkpoints: strict (coda reads all
cells), `reach1` (loop reads one neighbour, coda reads all), and `prev-reach1` (loop reads
one neighbour, coda reads only the previous cell; A8's arm). Second, the
latent-exploration survey (2026-09-18) proposed that a loop earns by holding several
candidates and narrowing them. The sample-oracle gate draws N perturbed trajectories
without training and asks whether the best of N after the fact beats the deterministic
one. Toy tasks (`eliminate`, `eliminate6`) asked whether a slot loop can carry a SET. The
loop-carry arms then tried to stop the per-pass decay the hop probe found.

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| h1 +0.0109, h2 -0.0169, h3 +0.0640 [0.0610, 0.0671], h4 +0.0535, h5 +0.0288, h6 -0.0203 | per-hop K1-K6 | prev-reach1 | 5000 | [failures/2026-09-18-hop-distance-earning.md](../lab/experiments/failures/2026-09-18-hop-distance-earning.md) | qualified: a restriction relay (synthesis note: "R"); the staircase stands as a mechanism reading |
| largest bin 0.0031 (strict), spread 0.0076 (reach1) | per-hop K1-K6 | strict / reach1 | 5000 | same | stands; the controls are flat in every bin |
| exact 0.000 before pass g-1; g3 +0.064 at pass 2, g4 +0.040 at pass 3 (single token) | planted copy benefit | prev-reach1 | 5000 | same | stands |
| g3 +0.083 at pass 2, g4 +0.046 at pass 3; g0 +0.566 | planted pair benefit, depths 1 to 8 | prev-reach1 | 5000 | [failures/2026-09-19-hop-distance-plateau-and-dilution.md](../lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md) | stands |
| h1 +0.0127, h2 -0.0120, h3 +0.0665, h4 +0.0545, h5 +0.0302, h6 -0.0182 | per-hop K1-K6 on 480 disjoint rows | prev-reach1 | 5000 | same | stands; the staircase replicates |
| g2 0.148, 0.108, 0.088, 0.080 over depths 1 to 4; under the cut 0.148 to 0.035 | planted g=2 decay | prev-reach1 | 5000 | same | stands; about a third lost per pass over the first three passes |
| g1 +0.181 to +0.291 (depth 1 to 6) under the cut; +0.181 to +0.137 uncut | own-span content re-supplied by the injection | prev-reach1 | 5000 | same | stands; passes REFINE what the injection re-supplies |
| trained 0.999 to 1.000 vs random-init 0.668 (d96) at hop 4 | per-fact probe | toy eliminate6 | toy | [successes/2026-09-19-toy-eliminate6-hop-distance.md](../lab/experiments/successes/2026-09-19-toy-eliminate6-hop-distance.md) | stands (toy) |
| 8 of 15 (d192) vs 2 of 15 (d96) cells solved | solve count | toy eliminate6 | toy | same | stands (toy); width buys hops |
| entropy 0.989 of 1.0986 at the candidate slot (exit 0.352, random init 0.096) | candidate entropy under the fixed-point term | toy eliminate | toy | [failures/2026-09-18-toy-eliminate-deferred-commitment.md](../lab/experiments/failures/2026-09-18-toy-eliminate-deferred-commitment.md) | stands (toy) |
| gain(16) +0.0109 (sigma 0.3) / +0.0246 (sigma 1.0) at depth 6; depth 6 minus depth 1 -0.0008 / -0.0005 | best-of-16 oracle over entry noise | slot-spandec-strict | 5000 | [failures/2026-09-18-sample-oracle-gate.md](../lab/experiments/failures/2026-09-18-sample-oracle-gate.md) | stands; the loop turns none of the entry variation into alternatives |
| gain(16) +0.0149 / +0.0388; growth with depth -0.0008 / -0.0015; cos_exit 0.799 to 0.849 | same, adapted reader | tul-code-target-uf | 30000 | [failures/2026-09-19-sample-oracle-adapted-reader.md](../lab/experiments/failures/2026-09-19-sample-oracle-adapted-reader.md) | stands; the verdict's proxy role was withdrawn by Wolfe's correction the same day (same file) |
| K1-K6 +0.0404 / +0.0086 (ruler +0.0163) | token K-curve | carry-sum / carry-gate on prev-reach1 | 5000 | [failures/2026-09-19-loop-carry-prev-reach1.md](../lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md) | qualified: a worse depth 1, not a better depth 6 (depth-6 CE 0.098 / 0.050 behind the ruler) |
| h1 +0.1033, h2 +0.0846, h3 -0.0209, h6 -0.0957 | per-hop K1-K6 | carry-sum | 5000 | same | qualified: inverts the staircase (Spearman -1.000); far hops dilute |
| g2 under the cut +0.108 (d2) to +0.210 (d6), ruler +0.148 to +0.036 | planted g=2 | carry-sum | 5000 | same | stands: re-supply stops the decay |
| carry RMS 5 to 75 across passes; core_gain_t0 max 2.82e5 | trainer instruments | carry-sum | 5000 | same | stands; the state is unbounded |

**What we concluded then.**

- Depth earning on the slot family is gather-limited and carry-limited. The geometry
  decides whether a hop is forced; the per-pass carry decides how much arrives. "Every
  slot-loop arm should be scored on the per-hop K-curve, since a whole-arm mean of +0.015
  was hiding +0.064 and -0.020."
- Carried content decays under the cell's own later passes; own-span content that the
  injection re-supplies each pass is refined. The plateau file names the mechanism
  "per-pass decay of loop-carried content".
- A slot loop carries a set for one or two hops for free; beyond that, training buys it
  and width buys hops (toys).
- Sampling around the deterministic loop finds little (0.011 to 0.039 nats at best of 16)
  and the loop does not grow it with depth. The adapted-reader filing concluded "a
  contractive loop cannot hold K alternatives apart from one perturbed entry".
- The carry as built replaced the pass-1 read, diluted far hops and ran unbounded.

**What later corrected it.**

- The 2026-09-18 "dilution" wording was withdrawn for carried content the next day
  (plateau file): the loss is per-pass decay, and dilution applies only to the
  displacement of the cell's own content.
- Wolfe corrected the sample-oracle verdict on 2026-09-19 16:50 UTC: both checkpoints
  were trained deterministically with the fixed-point term at 1.0, so "the exits
  converging with depth is that term doing its job, not evidence about whether a slot
  loop trained WITH K streams and a repulsion can hold alternatives apart". The
  branch-opening rule was withdrawn as a proxy and the fan4 queue lines were restored
  (same file). That opened B6.
- The staircase was built on in B8 (LXTUL-R), where Wolfe's 2026-09-22 call closed the
  restriction lane: the relay is a refund of a tax, not a gain.
- The fixed-point term's toy result (entropy 0.989) did not transfer as a rank keeper on
  the fan: fp0 in B7 shows the term holds scale, not rank.

### B6. 2026-09-19 to 2026-09-20: the LXTUL fan, K streams through the shared core

**What we tried and why.** The survey's 2x2 ({one stream, K streams} x {deterministic,
stochastic}) left one cell unmeasured on this tree: K streams at TRAINING time with a
diversity term, a learned selector and an oracle-over-streams instrument. LXTUL fan4 runs
K = 4 latent streams per slot through the shared core. The instrument is within-model:
run the coda once per stream, take the per-span minimum after the fact, compare with the
deployed read. The diversity terms were tried in order (cosine repulsion, then
epiplexity, then epiplexity plus within-slot volume), then the selector (a hard
winner-takes-all per slot, `select`), then the gate writing its own pick (`select-gate`),
then every stream written into its own cell so the coda's per-token attention is the
selector (`fan4-all`, Wolfe 2026-09-20: "do the code/mux tokens hold the other
explorations? ... we should at least try that"). The width partner is `pk4` (the strict
ruler at `prefix_k` 4, no fan).

**Readings.** ("oracle - mixed" is how much the best stream after the fact beats the
deployed read.)

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| single - oracle 0.1069; oracle - mixed 0.0078 | oracle over streams | fan4 (cosine repel) | 5000 | [failures/2026-09-19-lxtul-fan4.md](../lab/experiments/failures/2026-09-19-lxtul-fan4.md) | qualified in the same file: 0.099 of the 0.107 is a constant offset per stream, 0.008 is choice |
| rank 1.047; sign pattern `++--` in 2536 of 2573 slots | stream geometry | fan4 | 5000 | same | stands; a rank-1 split on one global axis |
| K1-K6 +0.0023; paired +0.0073 [+0.0051, +0.0094] worse than pk4 | K-curve; paired CE | fan4 | 5000 | same | stands |
| oracle - mixed 0.0128 / 0.0162 | oracle over streams | fan4-norepel / fan4-mean | 5000 | same | stands; the no-term controls read the same choice number |
| K1-K6 +0.0061 [+0.0057, +0.0066]; K3-K6 +0.0007; oracle - mixed 0.0114; rank 1.0001 | K-curve; oracle; geometry | fan4-epi | 5000 | [failures/2026-09-20-lxtul-fan4-epi.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epi.md) | qualified: a worse depth 1 (0.006), not a better depth 6; the term was gamed by one live stream per slot |
| rank 2.823 at pass 1, 2.312 at pass 6; oracle - mixed 0.0182; K1-K6 +0.0047; depth-6 CE 4.3306 | geometry; oracle; K-curve; CE | fan4-epivol | 5000 | [failures/2026-09-20-lxtul-fan4-epivol.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md) | stands; not gamed; the coda learned to read one stream (mix entropy 0.410) |
| oracle 0.113 below the best single stream; the gate cashes 0.014; regret 0.099 | oracle; deployed read | fan4-select | 5000 | [failures/2026-09-20-lxtul-fan4-select.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-select.md) | qualified: the oracle gap has a near-copy floor of about 0.04 ([failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md)) |
| K1-K6 +0.0076; K3-K6 +0.0013 | K-curve | fan4-select | 5000 | same | stands |
| +0.0861 [+0.0830, +0.0894] worse | paired CE vs pk4 | fan4-select | 5000 | same | stands |
| core_gain_t0 1.35 to max 20.7 | trainer instrument | fan4-select | 5000 | same | stands; winner-takes-all climbs the first-iteration gain |
| +0.0205 [+0.0183, +0.0227] worse vs pk4; 0.0656 better than select; stream 0 written in 95 % of slots | paired CE; written share | fan4-select-gate | 5000 | [failures/2026-09-20-lxtul-fan4-select-gate.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-select-gate.md) | stands; the arm is pk4 with a dead fan |
| read cashes 0.056 of 0.099 oracle value; regret 0.042 | best single minus deployed; oracle over streams | fan4-all | 5000 | [successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md) | qualified: four near-copy streams (cos 0.99) reproduce regret 0.045 and an oracle 0.112 below the best single ([failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md)) |
| 0.0342 better [0.0316, 0.0367] at depth 6; 0.0293 better at depth 1 | paired CE vs pk4 | fan4-all | 5000 | same | stands as a 5k reader reading ("context, not a ranking", same file) |
| K1-K6 +0.0049 [+0.0044, +0.0055]; K3-K6 +0.0004 | token K-curve | fan4-all | 5000 | same | stands; the loop is as flat as every slot arm |
| +0.0314 [+0.0302, +0.0326] (select +0.0574) | span decoder's own K1-K6 | fan4-all | 5000 | same | stands as a within-run signal that z moves with depth for the decoder while tokens do not |
| rank 2.820 (pass 1) to 2.060 (pass 6) of a ceiling of 3 | stream rank (centred) | fan4-all | 5000 | same (correction 2026-09-20) | stands; the ceiling is 3, not 4 |

**What we concluded then.**

- After three diversity terms (cosine, epi, epivol) the oracle sat at 0.008 to 0.018 of
  the mixture. The epivol filing closed the K-stream branch: "on web text with this reader
  there is nothing for the loop to explore over at the span level."
- The select arm reopened it the same day. "Diversity terms are not the lever;
  responsibility is." One hard per-slot winner made four readable streams with 0.113 of
  oracle value; selection before the span cashed 0.014 of it ("the Jensen wall").
- The gate writing its own pick transfers to eval but collapses the read onto one
  stream. The select family closes on the deployed write.
- Writing all four cells lets the coda select per token with the span's own evidence.
  fan4-all was filed as a success: 0.034 better than pk4, regret 0.042. "The write-all arm
  fixes the reader and the selector; it does not make the loop earn depth."

**What later corrected it.**

- The oracle-over-streams instrument has a floor. On 2026-09-21 the noise arm's four
  streams sat at cosine 0.99 and still read regret 0.045 and an oracle 0.112 below the
  best single stream. "At least that much of every fan arm's oracle gap is a min-over-K on
  per-token noise, not candidate value" ([failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md)).
  This qualifies the select arm's 0.113 and fan4-all's 0.056-of-0.099. fan4-all's paired
  CE against pk4 is not touched by it.
- The synthesis note (2026-09-22) keeps "build candidates, not gates" as an open lane,
  quoting fan4-all's 0.056 of 0.099. It does not quote the near-copy floor. The two files
  disagree on how much of that reading is candidate value; I cite both.
- The pass-6 rank collapse (2.03 of 3) was attacked in B7 by removing the fixed-point term
  and by re-supplying the stream trigger every pass. Neither held the rank.

### B7. 2026-09-21: the LXTUL-P ladder, "what gives a pass a job"

**What we tried and why.** The LXTUL-P note
([.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md](../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md))
set a ladder of rungs on the fan4-all base. Part 1 probes asked what collapses the K
streams by pass 6: the fixed-point term (`fp0`: term off) or the lack of a per-stream
identity in the map (`trig`: re-supply the stream trigger every pass). Part 2 rungs asked
for a pass with a job: P1 the K streams as K samples (`noise`: independent unit-RMS draws
at the entry), P4 the K streams as K lineages (`lineage`), P2 the loop as a denoiser
(`loop_denoise`, single stream, teacher-forced per-pass targets, on the arm A code
target). Two side arms ran the same day: removing the prelude on the strict loop (`np0`,
Wolfe: "If these fail we should test removing the prelude"), and the LCTUL flow thinker
with its noise schedule shifted to high noise (`cfg-tlow`). The width ruler `pk8` resolved
the vq panel's confound.

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| K1-K6 +0.0102 [+0.0096, +0.0109]; K3-K6 +0.0008 | token K-curve | fan4-all-fp0 | 5000 | [failures/2026-09-21-lxtul-fan4-all-fp0.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-fp0.md) | qualified: the extra over fan4-all is pass 1's; K3-K6 inside the yardstick (same file) |
| 0.0058 better [0.0035, 0.0080] at depth 6; 0.0044 worse at depth 1 | paired CE vs fan4-all | fp0 | 5000 | same | stands; inside the seed spread |
| pass-6 rank 2.009 (fan4-all 2.060); stream norms grow 2.3 to 4.2x; early preclip 522 vs 29.4 | geometry; trainer | fp0 | 5000 | same | stands; the term holds SCALE, not rank |
| K1-K6 +0.0089 [+0.0082, +0.0096]; K3-K6 +0.0016 [+0.0013, +0.0018] | token K-curve | fan4-all-trig | 5000 | [failures/2026-09-21-lxtul-fan4-all-trig.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-trig.md) | qualified: same shape as fp0; K3-K6 is the largest of the fan family and still tiny |
| pass-6 rank 1.470 (probe 1.425); stream 0 ends 4.1x its siblings | geometry | trig | 5000 | same | stands; re-supply collapses the streams harder |
| stream cos +0.994 at pass 0; entry norm 1,365 to 15,417 over training | geometry; trainer | fan4-all-noise | 5000 | [failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md) | stands; the model escaped the fixed-std draw by scaling the state |
| regret 0.045; oracle 0.112 below the best single stream, on streams at cos 0.99 | oracle over streams | noise | 5000 | same | stands; the near-copy floor that qualifies B6 |
| K1-K6 +0.0045; paired +0.0017 [-0.0006, +0.0039] vs fan4-all | K-curve; paired CE | noise | 5000 | same | stands |
| K1-K6 +0.0029; paired +0.0057 worse vs fan4-all; winner persistence 0.300 vs chance 0.256 | K-curve; paired; persistence | fan4-all-lineage | 5000 | [failures/2026-09-21-lxtul-fan4-all-lineage.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-lineage.md) | stands |
| K1-K6 +0.0045 [+0.0034, +0.0054]; K3-K6 -0.0008; paired +0.1652 [+0.1614, +0.1690] worse | K-curve; paired CE vs strict | strict-np0 (no prelude) | 5000 | [failures/2026-09-21-strict-np0.md](../lab/experiments/failures/2026-09-21-strict-np0.md) | stands; the entry explanation is spent for the slot loop |
| wrong_seed worth 0.3602; row rank 12.93 | worth; geometry | np0 | 5000 | same | stands |
| K1-K6 +0.5575 (5k +0.688, 10k +0.662, 15k +0.722) | forced-depth CE through the frozen coda | tul-code-target-denoise | 20000 | [failures/2026-09-21-lxtul-loop-denoise.md](../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md) | retracted as depth in the same file: genericity (the frozen-coda artefact of B4) |
| +0.1260 [+0.0725, +0.1779] worse | paired CE vs arm A, depth 6 | denoise | 20000 | same | stands |
| pass-1 l2 1.91 (steps 200-400) and 1.96 (last 1,000); l2_t5 1.68 to 0.48 | per-pass denoise loss | denoise | 20000 | same | stands; pass 1 never learned under teacher forcing |
| context share 1.9 % (parent 1 %); CA 2.7x chance | flow probe; in-batch retrieval | tul-code-cfg-tlow | 20000 | [failures/2026-09-21-lctul-cfg-tlow.md](../lab/experiments/failures/2026-09-21-lctul-cfg-tlow.md) | stands |
| k16 - k1 -0.0068 [-0.0089, -0.0048] (parent -0.0047) | 8-draw marginal | cfg-tlow | 20000 | same | stands; small |
| one draw vs ruler +0.6384 [+0.6310, +0.6462]; runner K1-K6 -0.0054 | paired; one-draw sweep | cfg-tlow | 20000 | same | stands |
| K1-K6 +0.0010 [+0.0008, +0.0012]; pk8 - strict -0.0349 [-0.0378, -0.0319]; vq8 - pk8 +0.0592 | K-curve; paired CE | strict-pk8 | 5000 | [successes/2026-09-21-strict-pk8-ruler.md](../lab/experiments/successes/2026-09-21-strict-pk8-ruler.md) | stands; width buys the reader nats and the loop nothing |

**What we concluded then.**

- The state levers are closed. The exit rank follows the OBJECTIVE, not the entry or
  the hold: K exits stay distinct only if K distinct things are asked of them.
- Noise at the entry has no hold: the model rescales the carrier. The oracle-over-streams
  gap has a floor of about 0.04 on near-copies.
- Teacher forcing is a bypass. Passes 2 to 6 learned to denoise an entry they were given;
  pass 1, which starts at noise, never learned. The denoise filing states the campaign's
  depth law: "depth is earned in proportion to the loss share that has no shallower route
  and that pass 1 cannot satisfy alone (plain/noise 0.185, plain/prelude 0.033, the slot
  side channel 0.002)".
- Removing the prelude doubles the entry's rank and moves the state further with depth,
  with the same flat curve. "The slot loop's flatness is not the state it starts from ...
  it is what the passes are ASKED for."
- The flow schedule is not a lever on a lossless code (cfg-tlow).

**What later corrected it.** No reading here was retracted later. The depth law and
Condition A ("no shallower route") became the design filter of the 2026-09-22 synthesis
note. The "one-pass-easy" part of the law was qualified by Wolfe on 2026-09-22: a job
pass 1 cannot finish is a HELPER, "not a universal truth", because the plain loop earns
0.185 with a strong pass 1 (synthesis note, Decision).

### B8. 2026-09-21 to 2026-09-22: LXTUL-R, the synthesis, the token-like read and the horizon pair

**What we tried and why.** LXTUL-R
([.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md](../.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md)) put the
fan on the relay geometry the hop staircase had shown to work: each pass reaches one more
span back. Step 0 first measured the prize on plain models: how much of the 0.40-nat
cross-span budget is the previous span and how much lies further back. A seed twin
settled an odd Step 0 arm. Step 1 ran fan4-all with loop reach 1, coda reach `prev`, the
fixed-point term off and a slot-state renorm; Step 1b fixed a leak found in Step 1; Step 2
tried a persistent carry (C1) and a history stream (C2). In parallel, LCTUL-J tried an
EMA code target with a variance floor (from JEPA-Anything, arXiv 2609.20800). On 09-22
the campaign synthesis note set a design filter, and Wolfe named two positives to build
on from the new ledger ([docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md)): the horizon result and
a token-like read of the slot state. The token-like read (`bcast`, `bcast-all`) adds the
exit state to every token of the next span; the horizon pair trains the strict twin at
depth 1 for 20k steps beside the looped strict ruler.

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| far budget lower bound 0.1259 [0.1165, 0.1364]; previous-span upper bound 0.1794 | paired CE, reach arms | budget-web reach1 / reachall / span-h | 5000 | [failures/2026-09-21-span-reach-split.md](../lab/experiments/failures/2026-09-21-span-reach-split.md) | stands; brackets re-read by the seed twin |
| K1-K6 +0.0554 (reach 1), +0.0378 (reach all), +0.0325 (span-h), +0.0265 (reach1-coda) | plain-loop token K-curve | budget-web arms | 5000 | same | stands; hand-checked in the ledger; "the first plain arm on the ledger whose loop earns more under a geometry change" |
| reach1-coda better than span twin by 0.112 [0.108, 0.117] at seed 2; the two coda seeds differ by 0.152; span seeds differ by 0.004 | seed twin | budget-web-reach1-coda-s2 | 5000 | [failures/2026-09-22-coda-seed-twin.md](../lab/experiments/failures/2026-09-22-coda-seed-twin.md) | stands; withdraws Step 0's "learned single-block trade" (a bad draw); brackets become previous span [0.112, 0.179], far budget [0.126, 0.197], relay 0.071 |
| K1-K6 +0.0202 [+0.0193, +0.0212]; K3-K6 +0.0025 | token K-curve | LXTUL-R Step 1 (leaky) | 5000 | [failures/2026-09-22-lxtul-r-step1.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1.md) | qualified: method fault, the conv and value shift relayed about four slots per pass; the numbers stand only for the leaky chain |
| K1-K6 +0.0261 [+0.0251, +0.0273]; K3-K6 +0.0057 [+0.0053, +0.0062]; natural tokens +0.0255 | token K-curve | LXTUL-R Step 1b | 5000 | [failures/2026-09-22-lxtul-r-step1b.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md) | qualified: the largest slot-loop K-curve on the ledger, and a partial refund of a tax (next row); Wolfe: "a mixed result" |
| +0.0265 [+0.0239, +0.0290] worse at depth 6; +0.0527 [+0.0496, +0.0555] worse at depth 1 | paired CE vs fan4-all-fp0 | Step 1b | 5000 | same | stands |
| g3 exact 0.000 at depth 1, +0.048 at depth 2; g4 +0.036 at depth 3; g2 keeps 62 % | planted pair (valid run) | Step 1b | 5000 | same | stands as a leak check and arrival staircase; the planted CE sits 4.7 nats above uniform (Wolfe's caveat, same file) |
| K1-K6 +0.0234 [+0.0222, +0.0246]; vs Step 1b -0.0018 [-0.0039, +0.0004] | K-curve; paired CE | Step 2 C1 persist | 5000 | [failures/2026-09-22-lxtul-r-step2-panel.md](../lab/experiments/failures/2026-09-22-lxtul-r-step2-panel.md) | qualified: relay; same depth-6 CE as Step 1b |
| K1-K6 +0.0204 [+0.0194, +0.0214]; vs Step 1b -0.0010 [-0.0032, +0.0011] | K-curve; paired CE | Step 2 C2 hist1 | 5000 | same | qualified: relay; same depth-6 CE as Step 1b |
| K1-K6 +0.0006 / +0.0016 / +0.0014 | token K-curve | lctul-ema / ema0 / ema-l2 | 5000 | [failures/2026-09-22-lctul-ema-target.md](../lab/experiments/failures/2026-09-22-lctul-ema-target.md) | stands; the floor |
| target rank 77.5 to 39.8 per cell; target cosine to the corpus mean 0.06 to 0.53; val loss equal to three decimals | target geometry; val | lctul-ema vs ema0 | 5000 | same | stands; the target moved and nothing reached the reader |
| centred own cosine 0.141 (ema0) to 0.300 (ema) | cosine | lctul-ema | 5000 | same | not read as a result, per Wolfe (same file) |
| K1-K6 +0.0016 / +0.0013 (ruler +0.0016) | token K-curve | strict-bcast / strict-bcast-all | 5000 | [failures/2026-09-22-arc-slot-token-like-read.md](../lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md) | stands |
| 0.0166 [0.0142, 0.0191] / 0.0123 [0.0102, 0.0143] better | paired CE vs ruler, depth 6 | bcast / bcast-all | 5000 | same | stands; the read is used (mean abs gate 0.165), almost all of it at the span's first token; it is the read, not the passes |
| looped better by 0.0076 [0.0047, 0.0103], 0.0075, 0.0083, 0.0074 [0.0045, 0.0104] at 5k / 10k / 15k / 20k | d1 twin pair, looped@6 vs twin@1 (filed as "looped@6 minus twin@1 = -0.0076") | slot-spandec-strict-20k vs strict-norecur-20k | 5000 to 20000 | same | stands; constant, no growth; 0.006 to 0.007 of it survives with the looped model read at depth 1 |

**What we concluded then.**

- The prize is real. At least 0.126 nats lie beyond the previous span, at every offset,
  so the far content is topical, not only boundary induction.
- The plain loop earns MORE under reach 1 (+0.055) than under full reach (+0.038). "When
  the geometry routes far content through a relay, the plain loop's passes do the
  relaying."
- On the slot fan the relay moves the passes (+0.026 with K3-K6 +0.0057, 78 % of the gain
  by pass 3) but "the earning is a partial refund of a tax the geometry charges (0.05 nats
  at depth 1 against a direct read), never a gain over it". Persistence and history
  streams do not change the depth-6 CE. LXTUL-R closes.
- The campaign synthesis (2026-09-22) set two conditions every future slot-loop proposal
  must state on paper: **A, no shallower route**, and **B, consumed** (the loss reads the
  pass's output with a steep gradient). Wolfe's standing call: no more restriction
  geometries whose mechanism is to cut the coda's read.
- A direct read of the exit state is used and worth 0.012 to 0.017, and the passes still
  add nothing. The slot loop's small value over its depth-1 twin (0.0075) is a
  training-time effect present from 5k, not a horizon effect. "Every 5k slot-arm verdict
  on the ledger stands." What the plain loop has and the slot loop lacks is that its
  passes act on the token states themselves.

**What later corrected it.**

- The Step 0 filing's "learned single-block trade" was withdrawn within hours by the seed
  twin (a bad draw). Its brackets were re-read at seed 2.
- Step 1's first reading was withdrawn as a method fault the same night (the leak,
  fixed at 9b430d3). The lesson: an attention mask does not bound the conv and the value
  shift; check the geometry with a perturbation test.
- Step 1b's status: the file says `failure` by its frozen bars and records Wolfe's
  reading "This shouldn't be a failure it is a mixed result". The ledger calls it
  "suspect: a tax refund". I carry it as qualified.
- LCTUL-J's error of placement (the floor on the predictor's cells, not on the encoder's
  output) is recorded in its own filing; the corrected placement is a different arm and
  has not run.
- The horizon pair re-reads B1's strict d1 twin (+0.0020 at 5k on a 5k-schedule pair) as
  a constant 0.0075 on a 20k-schedule pair. The two pairs have different schedules, so the
  5k values differ; neither grows.

### B9. 2026-09-23: LXTUL-G and LXTUL-GK, a stochastic contractive loop

**What we tried and why.** The central failure is the coda ignoring the loop. LXTUL-G
(note [.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md](../.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md))
makes the slot loop a stochastic recursion trained as a latent-variable model in the GRAM
shape: each pass takes a learned Gaussian step; at training a posterior that sees the next
span proposes the steps; a KL (balancing 0.8, beta 0.1) ties the posterior to the loop's
own prior. The hope: the latent carries next-span information the tokens lack, so the
coda's cheapest route goes through it, and depth use emerges if the prior needs several
contractive passes to reach what the posterior proposes. The named risk was the exposure
gap. A beta-1 arm (`lxtul-g-b1`) was added mid-run; the mean-free and depth-1 controls
were pulled. A spectral-decoupling arm (`fan4-all-sd`) ran beside them.

LXTUL-GK ([failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)) drops the posterior and the KL
and trains on the deployed object: K prior rollouts per row under the multi-sample bound,
L = - sum over spans of log (1/K) sum_k exp(sum over the span's tokens of log p(tok | z_k)).
Arms: `lxtul-gk4` (K = 4) and `lxtul-gk1` (K = 1, the width control).

**Readings.**

| number | instrument | arm | step | source | status |
|---|---|---|---|---|---|
| gap 0.7108 (2500) to 0.9840 (5000); ce_post 4.0270, ce_prior@1 5.0110 | exposure gap | lxtul-g | 2500 / 5000 | [failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) | stands; the gap grows with training |
| KL 58.85 nats per slot | KL | lxtul-g | 5000 | same | stands; no collapse |
| worth(zero) on a prior sample -0.325; a zeroed cell 0.60 better than the own prior cell at the first token; a shuffled cell costs 1.12 there | worth | lxtul-g | 5000 | same | stands; the coda reads the POSTERIOR's cell |
| K1-K6 -0.0050 [-0.0083, -0.0018] (-0.1340 at 2500) | token K-curve, one seeded prior sample | lxtul-g | 5000 | same | stands |
| ce_iw@4 +0.5901 [+0.5796, +0.6012]; ce_iw@16 +0.5336 worse than the ruler | paired CE vs ruler, depth 6 | lxtul-g | 5000 | same | stands |
| 0.159 (ce_prior@1 - ce_iw@4); 0.215 at 16 samples | width gain inside the Bayesian read | lxtul-g | 5000 | same | qualified: "on a trajectory the model was not trained to produce" (same file) |
| depth 6 better than depth 1 by 0.0123 [0.0093, 0.0150] under ce_iw@4; ce_prior@1 depth 1 minus 6 -0.0010 [-0.0055, +0.0037] | depth under width | lxtul-g | 5000 | same | qualified: same caveat; the single-sample curve is flat |
| gap 2.0686; ce_iw@4 +1.3275 worse than ruler; K1-K6 -0.3580; loss/total 9.26 to 12.88 after step 3000 | exposure gap; paired; K-curve; objective | lxtul-g-b1 | 5000 | same | stands; the optimizer ascended its own objective by about 3.6 nats |
| gap 0.0008; width gain 0.0003; ce_zero - ce_post +0.116 | exposure gap; width gain; worth | lxtul-g-b1 | 2500 | same (addendum 16:13) | stands; the collapsed phase is a near-deterministic loop whose cells the coda uses |
| K1-K6 +0.0043 [+0.0038, +0.0047] (fan4-all +0.0049); +0.0141 worse vs fan4-all | K-curve; paired CE | fan4-all-sd | 5000 | same | stands; spectral decoupling closed as a lever |
| final val 4.4238 (ruler 4.4249); K1-K6 +0.0037 [0.0034, 0.0041]; d6 vs ruler +0.0038 [0.0014, 0.0062] worse | val; K-curve; paired CE | lxtul-gk1 (K = 1) | 5000 | [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) (filed 2026-09-23 21:24) | stands |
| prior sigma/r 0.1 to 0.00031 (trainer, vlt entry 139); probe sigma/r 0.000339; width gain at 4 samples 0.0000684; worth(zero) 0.1833 | noise scale; eval width gain; worth | lxtul-gk1 | 5000 | `/home/wolfe/morph-scratch/arc/results/2026-09-23-lxtul-gk/lxtul_g_probe_lxtul-gk1_5000.json` (outside the repo, 192 rows) | stands; "noise switched off" (vlt entry 139) |
| val 4.4277; sigma/r 2.8e-4; training width gain 0.0065 (0.0008 at step 1340) | val; noise; training width gain | lxtul-gk4, first run (43bb234) | 5000 | vlt lxtul-fan4 entry 140; [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) (amendment 18:59) | retracted as evidence: CONFOUNDED; the K rollouts drew independent dropout masks, so the bound could earn width from dropout; fixed in 7d44ed7 |
| width gain at 4 samples 0.0000622; probe sigma/r 0.000290; worth(zero) 0.1891; ce_iw@4 at depth 1 4.2646 vs depth 6 4.2616 | eval probe, dropout off | lxtul-gk4, first run (43bb234) | 5000 | `/home/wolfe/morph-scratch/arc/results/2026-09-23-lxtul-gk/lxtul_g_probe_lxtul-gk4_5000.json` (outside the repo, 192 rows) | stands as an eval reading of a confounded run: with dropout off the width is noise only, and the noise had collapsed |
| 4,999 tok/s = 0.43x the ruler | rate at step 200 | lxtul-gk4 first run | 200 | vlt lxtul-fan4 entry 139 | stands for that run; P-7 is scored on the rerun |
| ce_iw@4 −0.7533 [−0.7664, −0.7413] vs lxtul-g ce_prior@1; −0.0044 [−0.0079, −0.0007] vs ruler d6; −0.0083 vs gk1 | paired CE, probe tokens | lxtul-gk4-shared (7d44ed7) | 5000 | [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) | stands; the ruler and gk1 gaps are n = 1 and inside the ~0.004 seed floor; identical rollouts give the K = 1 gradient, so the gk1 gap is not width |
| sigma/r 0.00026; width gain at 4 samples 0.00005; ce_iw@4 depth 1 − 6 +0.0035 [0.0029, 0.0041]; K1−K6 +0.0036 [0.0032, 0.0039]; 5,594 tok/s = 0.476x | noise; width; depth; rate | lxtul-gk4-shared | 5000 | same | stands; P-3, P-4, P-5, P-7 failed |

**What we concluded then.**

- LXTUL-G: the latent does not collapse and the coda reads it, but it reads the
  POSTERIOR's cell. Training takes posterior steps and eval takes prior steps, so the
  reader is trained on cells the deployed loop never produces. The gap grows, a prior cell
  is worse than an empty one, and beta 1 went unstable. "A stochastic slot loop has to be
  trained on the rollouts it will be deployed with."
- Two readings were kept as leads: width earns inside the Bayesian read (0.159 at 4, 0.215
  at 16), and under that read depth 6 beats depth 1 by 0.0123 while the single-sample
  curve is flat.
- GK (unfiled, from the vlt thread): gk1 switched its own noise off (sigma/r 0.1 to 3e-4).
  The agent's working reading on entry 139: "the multi-sample bound does not pay for
  diagonal-Gaussian noise around a unimodal optimum ... Search needs a structured/discrete
  proposal (distinct modes), not isotropic noise." That is a hypothesis in a thread entry,
  not a filed verdict.

**What later corrected it.** The first gk4 run was found confounded the same evening
(dropout masks drew per rollout; every GK test ran at dropout 0.0). Its training width
gain (0.0065) is not evidence for width. Its eval width gain with dropout off is 0.00006
(`/home/wolfe/morph-scratch/arc/results/2026-09-23-lxtul-gk/lxtul_g_probe_lxtul-gk4_5000.json`). The rerun `lxtul-gk4-shared`
confirmed entry 139's reading and the panel is filed as a failure ([failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)): the smooth
multi-sample bound pays for spread only at second order, so learned sigma collapses (XM,
arXiv 2607.27372 App. F.1).

## 6. Positive readings, by date

One row per positive reading from both parts, sorted by date (stable within a date). A-rows come from hist-A, B-rows from hist-B; B83 was added at the merge. "Positive" means the loop's passes, or the loop's written slot state, made the prediction better on the named instrument. Many rows are qualified or retracted; the status column says why. hist-B's inclusion rule for B-rows: token K-curves above the slot-loop floor (+0.0033) or named in the ledger, any paired, worth, probe or sampler reading that favours the loop or its written state, and every retracted positive of the range. Rows: 61 A, 83 B.

| # | date | number | instrument | arm | step | source | status (correcting doc) |
|---|---|---|---|---|---|---|---|
| A1 | 08-18 | CW1 beats CW2 by 0.009 [0.0078, 0.0102] | eval screen: slot cells kept vs random tokens kept | tul-a1-acap1 | 20000 | [.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md](../.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md) ln 19, 53 | qualified: stratified re-score favours random tokens in one stratum (same note ln 35); pre-fix leak model |
| A2 | 08-23 | A1c beats A0c by 0.0562 | val token CE, slot-loop arm vs dense | tul-a1-acap1 vs tul-a0-acap1 | 20000 | [lab/tul/arms-result.md](../lab/tul/arms-result.md) ln 51 | qualified: n=1, "not the gate" (same file ln 86); pre-fix (retention-carry note ln 23: leak +0.1433 on A0c) |
| A3 | 08-23 | 0.1900 (A1 0.0045) | plan_nats | tul_gate | 20000 | [results/2026-08-23-tul-gate-bakeoff.md](../lab/experiments/results/2026-08-23-tul-gate-bakeoff.md) ln 32 | qualified: pre-fix; the leak (+0.1433) exceeds the headline gain (same file ln 186) |
| A4 | 08-25 | core 0.3296 (1750), 0.2274 (1625) on ce_emit; 0.0007 on ce_main | region Shapley | ROLL_step_1625/1750, A1 | 1625/1750 | [results/2026-08-25-region-shapley/README.md](../lab/experiments/results/2026-08-25-region-shapley/README.md) ln 18-19 | qualified: pre-fix; worth goes to the one-token emit target, not to tokens |
| A5 | 08-25 | loop 0.0051, whole plan 0.0191; plan 0.0699 at dropout p=1 | loop worth / plan worth (ce_main) | onset ladder, A1 | ~1750 | [results/2026-08-25-region-shapley/README.md](../lab/experiments/results/2026-08-25-region-shapley/README.md) README ln 88-89; [lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md) ln 49-51 | qualified: pre-fix |
| A6 | 08-25 | +0.0169 (prelude +3.2205, coda +3.1051) | leave-one-out region CE cost | ROLL_step_1750 | 1750 | [failures/2026-08-25-scse-arm-c-long.md](../lab/experiments/failures/2026-08-25-scse-arm-c-long.md) ln 152-154 | qualified: pre-fix |
| A7 | 08-27 | +0.0058 / +0.0082 / +0.0106 / +0.0107 (controls -0.0002 to +0.0042), p=0.0286 | loop worth (ce_main), 4 seeds | tul_v1a2b | 3500 | [failures/2026-08-27-warmup-sigreg-ntpdrop.md](../lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md) ln 261, 268-269 | qualified: post-hoc (same file ln 273-276); confirmation [planned/2026-08-27-mux-matched-control-confirmation.md](../lab/experiments/planned/2026-08-27-mux-matched-control-confirmation.md) never reported; pre-fix |
| A8 | 08-27 | +0.0276 to +0.0363 (s1), +0.0378 to +0.0237 (s2) | loop worth | tg2-s1 / tg2-s2 (Thought Gestalt restriction) | 3000 to 3500 | [failures/2026-08-27-tg-restriction.md](../lab/experiments/failures/2026-08-27-tg-restriction.md) ln 78-79 | qualified: under the 0.05 bar, seed direction split; plan worth rise withdrawn (same file ln 105-113); pre-fix |
| A9 | 08-28 | 0.0148 (0.0867 fully starved); specificity 65.1 % | plan worth, token tax; plan span specificity | ctrlworth-s3 | 3000 | [planned/2026-08-28-reader-or-target.md](../lab/experiments/planned/2026-08-28-reader-or-target.md) ln 136-137; [planned/2026-08-28-plan-span-specificity.md](../lab/experiments/planned/2026-08-28-plan-span-specificity.md) ln 103 | qualified: pre-fix; specificity comes from the aux losses, restricted arms 0.1-3.0 % (specificity ln 104) |
| A10 | 08-29 | +0.233 (carry-audit re-read +0.2328); worth_shuffle 0.146 | token K1-K6; worth_shuffle | tul-l2-cap | 4500 / 4250 | [successes/2026-08-29-tul-loop-ladder.md](../lab/experiments/successes/2026-08-29-tul-loop-ladder.md) ln 111, 150 | **retracted**: carry off reads -1.1198 ([successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md) ln 81-82); future-leak collapse 105 % ([successes/2026-08-31-future-leak-attribution.md](../lab/experiments/successes/2026-08-31-future-leak-attribution.md) ln 89). Uncorrected in CLAUDE.md ln 66 and 4 notes (section 8). The worth 0.146 was never re-read carry-off |
| A11 | 08-30 | 0.0127 / 0.0085 | token K1-K6 | tul-l2cap-cond / tul-l2trunc | 4500 | [successes/2026-08-30-tul-ilv50-l2capcond.md](../lab/experiments/successes/2026-08-30-tul-ilv50-l2capcond.md) ln 92; [failures/2026-08-30-tul-l2-trunc.md](../lab/experiments/failures/2026-08-30-tul-l2-trunc.md) ln 42 | qualified: pre-fix; their comparator (0.233) is retracted |
| A12 | 08-31 | +0.0171, clean = corrupt | token K1-K6, causal carry | tul-l2nc | 4500 | [successes/2026-08-31-future-leak-attribution.md](../lab/experiments/successes/2026-08-31-future-leak-attribution.md) ln 91 | stands (small, causal) |
| A13 | 08-31 | 0.120 / 0.015; BC0 0.142 | token K1-K6 / K3-K6 | notul-l2nc; BC0 (plain loop, causal) | 4500 | [successes/2026-08-31-loop-killer-bisect.md](../lab/experiments/successes/2026-08-31-loop-killer-bisect.md) ln 28, 74 | stands |
| A14 | 08-31 | **0.220** / 0.017 | token K1-K6 / K3-K6 | BG0C0 (plain, GLA off, cap off) | 4500 | [successes/2026-08-31-loop-killer-bisect.md](../lab/experiments/successes/2026-08-31-loop-killer-bisect.md) ln 112-113 | stands: first honest plain-loop earning; pre-ramp, absmean |
| A15 | 08-31 | 0.207 / 0.016 | token K1-K6 / K3-K6 | notul-20k (plain, winner recipe) | 20000 | [successes/2026-08-31-tul-vs-notul-20k.md](../lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md) ln 80-82 | stands |
| A16 | 08-31 | 0.015 | token K1-K6, core axis | tul-20k (slot loop) | 20000 | [successes/2026-08-31-tul-vs-notul-20k.md](../lab/experiments/successes/2026-08-31-tul-vs-notul-20k.md) ln 83 | stands; slot loop 0.357 behind plain (ln 61) |
| A17 | 09-01 | 0.0113 / 0.0007 / 0.0019 | token K1-K6 | R0 / W1 / W2 (slot-only seeds) | 5000 | [failures/2026-09-01-write-side-ladder.md](../lab/experiments/failures/2026-09-01-write-side-ladder.md) ln 179-181 | stands: the free-ride floor |
| A18 | 09-01 | +0.1937 | token K1-K6 | R1 notul (plain, warmup 0) | 5000 | [failures/2026-09-01-write-side-ladder.md](../lab/experiments/failures/2026-09-01-write-side-ladder.md) ln 182 | stands |
| A19 | 09-01 | **+0.1685**; K6 CE 0.298 below R0 | token K1-K6 | A2 paid loop (warmup 0, absmean) | 5000 | [successes/2026-09-01-a2-paid-loop.md](../lab/experiments/successes/2026-09-01-a2-paid-loop.md) ln 104, 115 | stands as a number; arm rejected as "not TUL" ([.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md](../.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md) ln 10; 2026-09-22 synthesis ln 73); 2 of 4 draws detonated (ln 112) |
| A20 | 09-02 | +0.1666 / +0.1817, max corrupt-clean 0 | future-leak probe (earning clean) | A2 | 5000 | [successes/2026-09-02-a2-future-leak-probe.md](../lab/experiments/successes/2026-09-02-a2-future-leak-probe.md) ln 78-84, 104 | stands: A2's earning is causal |
| A21 | 09-02 | 0.0463 / 0.0489 / 0.0456 (flat A2 0.1209); wu5k 0.058 | token K1-K6 | A2 + 1000-step ramp | 2500; 5000 | [failures/2026-09-02-a2-warmup-and-seq512.md](../lab/experiments/failures/2026-09-02-a2-warmup-and-seq512.md) ln 137-140; [failures/2026-09-02-a2-warmup-5k-earning.md](../lab/experiments/failures/2026-09-02-a2-warmup-5k-earning.md) ln 57 | stands: the ramp cuts earning |
| A22 | 09-02 | **0.104** (paid) / 0.041 (plain) | token K1-K6, 480 rows | tul-a2-20k-wu / notul-20k-wu | 20000 | [failures/2026-09-02-warmup-20k-pair.md](../lab/experiments/failures/2026-09-02-warmup-20k-pair.md) ln 127 | stands; paid is still 0.022 behind plain on CE (ln 125); matched-step gap 0.132 to 0.012 (ln 141, 147) |
| A23 | 09-03 | +0.0371 [+0.0355, +0.0388] (token +0.0002, K3-K6 +0.0010) | own-loss K1-K6 | R3 M-own (slot loop) | 5000 | [failures/2026-09-03-tul-think-once-panel.md](../lab/experiments/failures/2026-09-03-tul-think-once-panel.md) ln 231 | qualified: fails the K3-K6 > 0.01 bar; no token value |
| A24 | 09-04 | +0.0128 / +0.0135 | forecast K1-K6 | Y1 / Y2 (constrained slot loop) | 5000 | [successes/2026-09-04-tul-forward-levers.md](../lab/experiments/successes/2026-09-04-tul-forward-levers.md) ln 172-173 | stands: "stable and empty" (ln 235); K3-K6 about 0 |
| A25 | 09-04 | +0.0138 / +0.0180 | forecast K1-K6 | E1 g95 / g98 | 5000 | [failures/2026-09-04-arc-e1-gain-target-dial.md](../lab/experiments/failures/2026-09-04-arc-e1-gain-target-dial.md) ln 94-95 | stands; K3-K6 at most 0.0014 |
| A26 | 09-04 | +0.0077 | forecast K3-K6 | E2 (pre-onset draw) | 2500 | [failures/2026-09-04-arc-e2-iteration-conditioning.md](../lab/experiments/failures/2026-09-04-arc-e2-iteration-conditioning.md) ln 200 | **retracted**: stable rerun +0.0001 (same file ln 175, 202) |
| A27 | 09-04 | **+0.0209**; forecast +0.187 | token K1-K6 | E4 mask under the constraint (absmean) | 5000 | [successes/2026-09-04-arc-e4-mask-under-constraint.md](../lab/experiments/successes/2026-09-04-arc-e4-mask-under-constraint.md) ln 76 | qualified: 0.13 CE tax (ln 106); gone under norm_match, +0.0009 ([failures/2026-09-10-arc-slot-mux-mask-norm-match.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md) ln 219) |
| A28 | 09-04 | +0.0414 / +0.1037 | token K1-K6 | notul-20k-wu / tul-a2-20k-wu | 20000 | [failures/2026-09-04-arc-e0-where-depth-earns.md](../lab/experiments/failures/2026-09-04-arc-e0-where-depth-earns.md) ln 65, 67 | stands |
| A29 | 09-04 | K3-K6 **+0.566**; K6-K16 +0.206; K1-K6 +1.906 | token K-curves | Huginn-0125 (external, eval only) | n/a | [failures/2026-09-04-huginn-loop-contribution.md](../lab/experiments/failures/2026-09-04-huginn-loop-contribution.md) ln 144-147 | stands (external; no depth-1 control, so dependence) |
| A30 | 09-04 | +0.4185; K3-K6 +0.0932 | own-loss K1-K6 | X1 (backward clip only) | 2500 | [failures/2026-09-04-tul-clip-through-time.md](../lab/experiments/failures/2026-09-04-tul-clip-through-time.md) ln 198 | **retracted**: model 0.75 nats worse, "not a THINK in substance" (same file ln 216) |
| A31 | 09-07 | K3-K6 **+0.277**; K6-K12 +0.0405 | token K-curves | E6 notul_deep16 (plain, mean 16) | 5000 | [successes/2026-09-07-arc-e6-deep-recurrence-draw.md](../lab/experiments/successes/2026-09-07-arc-e6-deep-recurrence-draw.md) ln 83, 88 | qualified: 0.104 worse than mean 6 (ln 89); truncated-BPTT silences 27 % of samples ([.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md](../.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md) ln 14) |
| A32 | 09-07 | +0.0263; forecast +0.823 | token / forecast K1-K6 | E7 mask + deep 16 | 2500 | [failures/2026-09-07-arc-e7-block-loop.md](../lab/experiments/failures/2026-09-07-arc-e7-block-loop.md) ln 97-98 | **retracted**: pre-onset, detonated at 2712 (same file ln 126) |
| A33 | 09-07 | +0.0526 | token K1-K6 | E8-6 (MTP heads) | 5000 | [failures/2026-09-07-arc-e8-multi-token-coda.md](../lab/experiments/failures/2026-09-07-arc-e8-multi-token-coda.md) ln 100 | qualified: next-token CE +0.3445 worse (ln 101) |
| A34 | 09-07 | 0.0216 (notul 0.0367); paired CE +0.0009 | token K1-K6 | E11 fixed-point term, ramped | 5000 | [successes/2026-09-07-arc-e11-fixed-point-ramped.md](../lab/experiments/successes/2026-09-07-arc-e11-fixed-point-ramped.md) ln 62-65 | stands |
| A35 | 09-07 | **+0.0489**; forecast K3-K6 +0.0172 | token K1-K6 | E13 mask, mean 12 (absmean) | 5000 | [failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 145, 129 | qualified: same CE as mean 6 (ln 131); mask dependence gone under norm_match (A27). [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) ln 79 mislabels +0.0172 as forecast K6-K12; K6-K12 is +0.0020 (ln 129) |
| A36 | 09-07 | +0.0425 / +0.0601 | token K1-K6 | E14 g102 / g102-rn | 2500 | [failures/2026-09-07-arc-e14-expansive-dial.md](../lab/experiments/failures/2026-09-07-arc-e14-expansive-dial.md) ln 133-134 | **retracted**: pre-onset; both detonated (3877, 4639; ln 120) |
| A37 | 09-08 | +0.406; answer K3-K6 -0.030 | token K1-K6 | E16 mask, Olympiad | 6000 | [failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md](../lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md) ln 161 | **retracted** as depth: broken depth 1, 1.31 nats (ln 202); math set aside by Wolfe 2026-09-22 ([docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) ln 13) |
| A38 | 09-08 | +0.0243 / +0.0314 / +0.0486 | token K1-K6 | E18 mask k2 / k4 / k8 (absmean) | 5000 | [planned/2026-09-04-loop-contribution-arc.md](../lab/experiments/planned/2026-09-04-loop-contribution-arc.md) ln 339 | qualified: "an absmean artefact" (same row); own planned file has no Results |
| A39 | 09-09 | +0.033 / K3-K6 +0.0009 | token K1-K6 | parcae-entry (absmean, noise entry) | 5000 | [failures/2026-09-09-arc-e19-parcae-loop-entry.md](../lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md) ln 141 | stands; depth-1 twin is 0.0192 ahead at 5k (ln 158) |
| A40 | 09-09 | **+0.168**; K3-K6 +0.0119; beats depth-1 by 0.0336 | token K1-K6; d1 twin | depthcand-dense-core (bf16 core) | 5000 | [failures/2026-09-09-arc-e20-loop-depth-candidates.md](../lab/experiments/failures/2026-09-09-arc-e20-loop-depth-candidates.md) ln 147, 161 | stands as a diagnostic; a dense core is not a production option |
| A41 | 09-09 | **+0.1849 [+0.1816, +0.1883]**; K3-K6 +0.0139; ttq +0.0410; threshold-03 +0.0261 | token K1-K6 | scale-norm-match (noise entry) and siblings | 5000 | [successes/2026-09-09-arc-per-pass-strength.md](../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md) ln 112-114 | stands; CE@6 +0.0817 worse than absmean at 5k (ln 112), a lead of 0.0100 by 20k (A43) |
| A42 | 09-09 | +0.043 / +0.074 | token K1-K6 | density-half / density-quarter | 5000 | [failures/2026-09-09-arc-density-panel.md](../lab/experiments/failures/2026-09-09-arc-density-panel.md) ln 136-137 | qualified: "dependence without computation" (ln 163) |
| A43 | 09-09 | 0.1363 / 0.1562 / 0.1665 / **0.1702**; CE -0.0100 vs absmean at 20k | token K1-K6 | norm-match-20k (plain, noise entry) | 5k/10k/15k/20k | [failures/2026-09-09-arc-norm-match-recipe-reads.md](../lab/experiments/failures/2026-09-09-arc-norm-match-recipe-reads.md) ln 141-144 | stands; value read later by the d1 twin, +0.0674 (B16) |
| A44 | 09-09 | 0.035 to 0.041 | token K1-K6 over steps | old-entry notul (absmean) | 2500 to 20000 | [failures/2026-09-09-arc-horizon-ternary-25k.md](../lab/experiments/failures/2026-09-09-arc-horizon-ternary-25k.md) ln 95-103 | stands: absmean earning does not grow |
| A45 | 09-09 | +0.1200 | token K1-K6 | tul-norm-match (paid) | 5000 | [failures/2026-09-09-arc-norm-match-recipe-reads.md](../lab/experiments/failures/2026-09-09-arc-norm-match-recipe-reads.md) ln 156 | qualified: +0.1252 worse CE than plain (ln 158); paid loop "not TUL" |
| A46 | 09-09 | +0.033 / K3-K6 +0.0055 | token K1-K6 | plain-panel-norm-match (prelude entry) | 5000 | [failures/2026-09-09-arc-slot-loop-norm-match.md](../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md) ln 147 | stands: the entry confound, 0.185 (noise entry) vs 0.033 (prelude entry) |
| A47 | 09-09 | worth(zero) +0.811 at offset 0 (token K1-K6 +0.0006) | worth by offset | slot-unpack-norm-match | 5000 | [failures/2026-09-09-arc-coda-reads-the-thought.md](../lab/experiments/failures/2026-09-09-arc-coda-reads-the-thought.md) ln 115-117 | stands: the coda reads z; the passes do not fill it |
| A48 | 09-10 | +0.1168; K3-K6 +0.0067 | token K1-K6 | slot-unpack-fixed-depth | 5000 | [failures/2026-09-10-arc-slot-map-levers.md](../lab/experiments/failures/2026-09-10-arc-slot-map-levers.md) ln 143 | qualified: CE@6 +0.052 worse, "depth dependence without depth value" (ln 167) |
| A49 | 09-10 | 0.1074 / 0.0185 (mux 0.0015) | K0-K6 (entry cell vs exit cell) | slot-unpack-free / slot-mnext-noise-entry | 5000 | [results/2026-09-10-slot-geometry-audit/README.md](../lab/experiments/results/2026-09-10-slot-geometry-audit/README.md) ln 311-313 | stands: pass 1 does the work; passes 2-6 at most 0.002 |
| A50 | 09-10 | 5/5 chains solved (exit-only 2/5; fixed-point 1.0 5/5) | toy task solve rate | toy slot loop, staged | 4000 | [lab/toy_slot_loop/WRITEUP.md](../lab/toy_slot_loop/WRITEUP.md) ln 136-137, 193 | stands (toy only) |
| A51 | 09-10 | **+0.0670** (token +0.0020); own CE 6.06 to 3.36 by pass 3 | forecast (next) K1-K6; own ladder | slot-mnext-staged | 5000 | [successes/2026-09-10-arc-slot-mnext-staged.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md) ln 212, 232 | qualified: exit forecast equals the ruler's (6.7698 vs 6.7724, ln 212) |
| A52 | 09-10 | +0.0637; next K3-K6 +2.080; token +0.0023 | forecast K1-K6 | slot-mnext-staged-20k | 20000 | [successes/2026-09-10-arc-slot-mnext-staged-20k.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged-20k.md) ln 266-269 | qualified: 0.1609 behind plain, gap flat 10k to 20k (ln 255) |
| A53 | 09-10 | +0.0604; token +0.0033 | forecast K1-K6 | slot-mnext-staged-mask | 5000 | [failures/2026-09-10-arc-slot-mnext-staged-mask.md](../lab/experiments/failures/2026-09-10-arc-slot-mnext-staged-mask.md) ln 193 | qualified: CE@6 4.4373 vs ruler 4.3290 (ln 193) |
| A54 | 09-10 | +0.0145; K3-K6 0.0000 | forecast K1-K6 | slot-mux-fixed-point-off | 5000 | [failures/2026-09-10-arc-slot-mux-fixed-point-off.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-fixed-point-off.md) ln 225 | stands; state expands 2.2x (ln 240) |
| A55 | 09-11 | **+0.0236**; exit 0.017 better than ruler; 0.366 of the fall at pass 1 | forecast K1-K6 | slot-mnext-gradpass | 5000 | [failures/2026-09-10-arc-slot-mnext-gradpass.md](../lab/experiments/failures/2026-09-10-arc-slot-mnext-gradpass.md) ln 296, 306, 309 | stands as "one descent step"; token K1-K6 +0.0014 |
| A56 | 09-11 | +0.0279 / +0.0301 | token K1-K6 | budget-web-full / budget-web-span (plain, prelude entry) | 5000 | [failures/2026-09-11-arc-span-budget.md](../lab/experiments/failures/2026-09-11-arc-span-budget.md) ln 339-340 | stands: "A context-starved model does not loop more" (ln 382-383) |
| A57 | 09-11 | all-slot worth 0.115 to 0.182; CE -0.0720 vs mux-mask | all-slot worth; paired CE | slot-spandec-mask | 5000 | [successes/2026-09-11-arc-span-decoder.md](../lab/experiments/successes/2026-09-11-arc-span-decoder.md) ln 291, 307; [.agents/notes/proposed/architecture/2026-09-11-span-decoder-target.md](../.agents/notes/proposed/architecture/2026-09-11-span-decoder-target.md) ln 221 | qualified: a channel gain through the bypass cells, token K1-K6 0.0007 (ln 297) |
| A58 | 09-11 | 0.053 to 0.061 (disjoint CIs) | prefix-write worth, bin 8-15 | spandec-chain-mask | 5000 | [successes/2026-09-11-arc-span-decoder.md](../lab/experiments/successes/2026-09-11-arc-span-decoder.md) ln 379-380 | stands, small; CE cost +0.0068 (ln 360) |
| A59 | 09-12 | all_slots worth 0.1865; token K1-K6 +0.0016 | worth(zero); token K1-K6 | slot-spandec-strict | 5000 | [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 286 | stands: closing the bypass costs 0 |
| A60 | 09-12 | **+0.0163 [+0.0153, +0.0173]**; K3-K6 +0.0042 | token K1-K6 | slot-spandec-strict-prev-reach1 | 5000 | [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 289 | qualified: forced relay; "z has to hold the history" (ln 248-250); CE +0.0023 vs prev |
| A61 | 09-12 | **+0.0127 [+0.0119, +0.0136]**; K3-K6 +0.0028; CE -0.0086 [-0.0109, -0.0063] vs prev | token K1-K6; paired CE | slot-spandec-strict-prev-reach2 | 5000 | [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 290 | qualified: relay; CE only at parity with strict (-0.0004); restriction lane closed 2026-09-22 ([.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md)) |
| B1 | 09-12 | AUC 0.611 entry to 0.638 pass 1 (exit 0.637; null p95 0.514) | linear probe, "next-span CE below median", mean reduction | slot-spandec-mask | 5000 | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | qualified: the whole rise is pass 1; the ruler's AUC falls (same file) |
| B2 | 09-12 | looped better by 0.0235 [0.0209, 0.0263] | d1 twin, paired CE (arm@6 vs norecur@1) | slot-spandec-mask vs slot-spandec-norecur (bypass) | 5000 | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | qualified: bypass geometry only; strict twin 0.0020 [-0.0006, +0.0044] (same file) |
| B3 | 09-12 | ruler better by 0.0020 [-0.0006, +0.0044] | d1 twin (5k schedule) | slot-spandec-strict vs strict-norecur | 5000 | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | qualified: CI crosses 0; a 20k-schedule pair reads a constant 0.0075 ([failures/2026-09-22-arc-slot-token-like-read.md](../lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md)) |
| B4 | 09-12 | +0.01353 | write contribution (ce_entry - ce_loop) | slot-spandec-strict | 5000 | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | stands; point estimate, no CI |
| B5 | 09-12 | +0.02195 (1.6x the ruler) | write contribution | slot-spandec-strict-codaspan | 5000 | [failures/2026-09-12-arc-objective-arms.md](../lab/experiments/failures/2026-09-12-arc-objective-arms.md) | stands; no CI; missed its 0.010-excess bar by 0.0016 |
| B6 | 09-12 | +0.0031 [+0.0026, +0.0036] | identical-target grid, exit-column K1-K6 | slot-spandec-strict | 5000 | [failures/2026-09-12-arc-objective-arms.md](../lab/experiments/failures/2026-09-12-arc-objective-arms.md) | stands; at the slot-loop yardstick |
| B7 | 09-12 | +0.00145 [+0.00087, +0.00202]; win rate 0.477 | critic direction vs random (10 % RMS step) | slot-spandec-strict-critic | 5000 | [failures/2026-09-12-arc-core-token-and-critic.md](../lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md) | stands; seven times under its bar |
| B8 | 09-12 | 0.0012 to 0.0024 nats per pass | critic_gap_traj (measured worth of one pass) | slot-spandec-strict-critic | 5000 | [failures/2026-09-12-arc-core-token-and-critic.md](../lab/experiments/failures/2026-09-12-arc-core-token-and-critic.md) | stands; tiny |
| B9 | 09-12 | +0.0647 [+0.0602, +0.0691] | token K1-K6 | oly-notul-nm (plain, Olympiad) | 6000 | [failures/2026-09-12-arc-math-under-norm-match.md](../lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md) | stands; math set aside by Wolfe 2026-09-22 ([docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md)) |
| B10 | 09-12 | +0.0111 [+0.0085, +0.0136]; K3-K6 +0.0024 | token K1-K6 | oly-spandec-strict | 6000 | [failures/2026-09-12-arc-math-under-norm-match.md](../lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md) | stands; same caveat; slot gap to plain +0.2161 |
| B11 | 09-12 | +0.0049 [+0.0047, +0.0052] (plain +0.0011) | token K1-K6 | sud-spandec-strict | 6000 | [failures/2026-09-12-arc-math-under-norm-match.md](../lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md) | stands; same caveat; slot gap to plain +0.2502 |
| B12 | 09-13 | +0.00719 (strict +0.00314) | span decoder's own K1-K6 | slot-register-m4 | 5000 | [failures/2026-09-13-arc-thought-register.md](../lab/experiments/failures/2026-09-13-arc-thought-register.md) | qualified: the decoder reads the mean of four cells; token K3-K6 -0.00029 (same file) |
| B13 | 09-13 | +0.0342 | token K1-K6 | traj (trajectory prefix) | 5000 | [failures/2026-09-13-arc-trajectory-prefix.md](../lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md) | retracted: forced-depth pad cells; same-width trajrep +0.0019 (same file, T-3) |
| B14 | 09-13 (run 09-20) | worth 0.2287 vs 0.1865; offset 0 0.9016 vs 0.7567 | worth(zero) | slot-strict-bootstrap | 5000 | [failures/2026-09-13-arc-plain-then-tul-bootstrap.md](../lab/experiments/failures/2026-09-13-arc-plain-then-tul-bootstrap.md) | qualified: a better seed; passes K3-K6 +0.0002 (same file) |
| B15 | 09-13 | +0.0102 [+0.0093, +0.0110]; K3-K6 +0.0012 | aux K1-K6 (the coretok core run over tokens) | slot-spandec-strict-coretok | 5000 | [failures/2026-09-13-arc-loop-reads-tokens.md](../lab/experiments/failures/2026-09-13-arc-loop-reads-tokens.md) | stands (hand-checked in [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md)) |
| B16 | 09-13 (scored 09-19) | looped better by 0.0674 [0.0647, 0.0701] | d1 twin, paired CE at own depth | norm-match-20k vs plain d1 | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | stands at matched steps; at matched wall clock d1@20k beats top@10k by 0.1728 (same file) |
| B22 | 09-14 | AA(noise) 0.9998, AA(swap) 0.9777 (strict 0.8187 / 0.8644) | AA score | plain norm_match loop (notul_norm_match_20k) | 5000 | [failures/2026-09-14-arc-loop-diagnostics.md](../lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md) | stands; the earning loop is the attractor |
| B23 | 09-14 | entropy about 1.6 nats at 2 of 6 pairs | basin map | slot-spandec-strict | 5000 | [failures/2026-09-14-arc-loop-diagnostics.md](../lab/experiments/failures/2026-09-14-arc-loop-diagnostics.md) | stands; not followed |
| B24 | 09-14 | +0.0040 [+0.0036, +0.0044] | token K1-K6 | horizon-arm (LoopMTP port) | 5000 | [failures/2026-09-14-arc-horizon-passes.md](../lab/experiments/failures/2026-09-14-arc-horizon-passes.md) | qualified: paired vs its control -0.0001 [-0.0023, +0.0025]; the gate reads 83 % from pass 1 |
| B25 | 09-14 | +0.278; K3-K6 +0.0086 | token K1-K6 | horizon-fixed6 | 5000 | [failures/2026-09-14-arc-horizon-passes.md](../lab/experiments/failures/2026-09-14-arc-horizon-passes.md) | retracted as depth: trained only at depth 6, out of distribution elsewhere (same file) |
| B27 | 09-14 | -1.25 (5k), -1.29 (10k); rollout1 -0.98, -0.42 | phase-2 marginal k-curve, trusting coda | tul-code-20k / rollout1 | 5000 to 10000 | [failures/2026-09-14-arc-tul-code-20k.md](../lab/experiments/failures/2026-09-14-arc-tul-code-20k.md) | qualified: flat within 5000 rollout steps; the coda discounts samples (same file) |
| B26 | 09-15 | k16 - k1 -0.004 [-0.006, -0.003] | 8-draw marginal (sampler depth) | tul-code | 5000 | [failures/2026-09-14-arc-tul-code.md](../lab/experiments/failures/2026-09-14-arc-tul-code.md) (addendum) | stands; negligible |
| B28 | 09-15 | k16 - k1 -0.010 [-0.013, -0.007] | 8-draw marginal | tul-code renorm fork | 20000 | [successes/2026-09-15-tul-code-renorm-r10k.md](../lab/experiments/successes/2026-09-15-tul-code-renorm-r10k.md) | stands; small |
| B29 | 09-15 | k16 - k1 -0.005 [-0.007, -0.002] | 8-draw marginal | tul-code-cfg | 20000 | [failures/2026-09-15-tul-code-conditioned-thinker.md](../lab/experiments/failures/2026-09-15-tul-code-conditioned-thinker.md) | stands; small |
| B30 | 09-15 | k16 - k1 -0.017 [-0.021, -0.014] | 8-draw marginal | tul-code-jepa | 20000 | [failures/2026-09-15-tul-code-conditioned-thinker.md](../lab/experiments/failures/2026-09-15-tul-code-conditioned-thinker.md) | stands; small |
| B31 | 09-15 | k16 - k1 -1.88 [-2.00, -1.73] | 8-draw marginal on a frozen phase-2 coda | tul-code-thinker | 50000 | [failures/2026-09-15-tul-code-thinker-only.md](../lab/experiments/failures/2026-09-15-tul-code-thinker-only.md) | qualified: at 10 to 14 nats, "a level no coda can use" (same file) |
| B32 | 09-16 | one draw k1 4.585 to k2 4.486 | one-draw depth curve | tul-code-lejepa | 20000 | [failures/2026-09-16-tul-code-lejepa.md](../lab/experiments/failures/2026-09-16-tul-code-lejepa.md) | qualified: one seed; marginal k16 - k1 -0.0006 [-0.004, +0.003] (same file) |
| B33 | 09-16 | k4 - k1 -0.0136 [-0.0164, -0.0108] (15k), -0.0119 [-0.0155, -0.0082] (20k) | 8-draw marginal | tul-code-dplan | 15000 / 20000 | [failures/2026-09-16-lctul-dplan-semantic.md](../lab/experiments/failures/2026-09-16-lctul-dplan-semantic.md) | qualified: does not reach generated text; OWN@4 - OWN@1 +0.009 [-0.008, +0.028] (same file) |
| B35 | 09-17 | OWN - SHUF +0.0230 [+0.0051, +0.0421] | semantic probe | tul-code-target (arm A) | 20000 | [planned/2026-09-17-lctul-target-slot-loop.md](../lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md) | stands; first positive own-vs-foreign in LCTUL; OWN - ZERO +0.0054 crosses 0 |
| B36 | 09-17 | K1-K6 +0.0323 [+0.0236, +0.0415]; depth 16 -0.0959 | forced-depth CE, frozen coda | arm A | 20000 | [planned/2026-09-17-lctul-target-slot-loop.md](../lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md) | retracted as depth: genericity ([.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md](../.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md)) |
| B37 | 09-17 | exit cos 0.0604 (entry) to 0.1622 (l6); l6 - l1 +0.0052 | per-pass cosine to the code | arm A | 20000 | [planned/2026-09-17-lctul-target-slot-loop.md](../lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md) | qualified: pass 1 does it; a cosine |
| B41 | 09-17 | cos l6 - l1 +0.0073 (arm A +0.0051) | per-pass cosine | tul-code-target-prog | 20000 | [failures/2026-09-17-lctul-target-progressive.md](../lab/experiments/failures/2026-09-17-lctul-target-progressive.md) | qualified: about fifty times too small (same file); a cosine |
| B42 | 09-17 | OWN - ZERO +0.0222 [+0.0045, +0.0407] | semantic probe | prog | 20000 | [failures/2026-09-17-lctul-target-progressive.md](../lab/experiments/failures/2026-09-17-lctul-target-progressive.md) | stands |
| B38 | 09-18 | -4.64 (frozen) to +0.177 (adapted) | cell worth, same weights | tul-code-target-uf | 30000 | [successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md) | stands as a reader result |
| B39 | 09-18 | OWN - ZERO +0.0179 [+0.0025, +0.0329]; OWN vs arm A +0.0290 [+0.0040, +0.0533] | semantic probe | uf | 30000 | [successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md) | stands |
| B40 | 09-18 | +0.0047 [+0.0038, +0.0058] | token K1-K6 | uf | 30000 | [successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md) | stands; at the floor's edge |
| B43 | 09-18 | shuf - own +0.1876 (arm A +0.1314) | twin read | tul-code-only (frozen ref) | 20000 | [failures/2026-09-17-lctul-code-only-ref.md](../lab/experiments/failures/2026-09-17-lctul-code-only-ref.md) | qualified: zero - own -4.6238, a competent reader prefers no cell (same file) |
| B44 | 09-18 | +0.0698, -0.0014, +0.0414, +0.0291 (5k to 20k) | forced-depth CE | tul-code-only | 5000 to 20000 | [failures/2026-09-17-lctul-code-only-ref.md](../lab/experiments/failures/2026-09-17-lctul-code-only-ref.md) | retracted: the reader is above uniform (12.64 > 10.80); void (same file) |
| B45 | 09-18 | cos l0 0.4012, l1 0.5235, l6 0.5310 | per-pass cosine to the winner's code | tul-code-grade-l2 | 20000 | [failures/2026-09-17-lctul-graded-target.md](../lab/experiments/failures/2026-09-17-lctul-graded-target.md) | qualified: met in one pass |
| B52 | 09-18 | entropy 0.989 of 1.0986 (exit 0.352) | candidate entropy with the fixed-point term | toy eliminate | toy | [failures/2026-09-18-toy-eliminate-deferred-commitment.md](../lab/experiments/failures/2026-09-18-toy-eliminate-deferred-commitment.md) | stands (toy) |
| B17 | 09-19 | +0.1702 [+0.1672, +0.1729] | own K1-K6 | norm-match-20k (Poisson mean 6) | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | qualified: depth dependence; final CE ties the cheaper rungs inside the 0.0237 seed spread (same file) |
| B18 | 09-19 | +0.0705 [+0.0689, +0.0721] | own K1-K6 | p4 (Poisson mean 4) | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | qualified: same |
| B19 | 09-19 | +0.0478 [+0.0465, +0.0492] | own K1-K6 | p3 (Poisson mean 3, ships) | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | qualified: same |
| B20 | 09-19 | +0.0387 [+0.0373, +0.0402] | own K1-K6 | d3 (clamped mean 3) | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | qualified: same; clamped rungs out of the ship set (2026-09-14 correction, same file) |
| B21 | 09-19 | +0.2294 / +0.5259 | own K1-K6 | d3fixed / d6fixed | 20000 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | qualified: fixed depth is a tied-weight deep net, not the sampled loop (same file; synthesis note standing calls) |
| B46 | 09-19 | h3 +0.0640 [0.0610, 0.0671] (arrives pass 2) | per-hop K1-K6 | prev-reach1 | 5000 | [failures/2026-09-18-hop-distance-earning.md](../lab/experiments/failures/2026-09-18-hop-distance-earning.md) | stands as a mechanism reading; qualified as value: a relay refunds a tax ([failures/2026-09-22-lxtul-r-step1b.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md)) |
| B47 | 09-19 | h4 +0.0535 (pass 3); h5 +0.0288 (passes 3 to 6) | per-hop K1-K6 | prev-reach1 | 5000 | [failures/2026-09-18-hop-distance-earning.md](../lab/experiments/failures/2026-09-18-hop-distance-earning.md) | same |
| B48 | 09-19 | h3 +0.0665, h4 +0.0545, h5 +0.0302 on 480 disjoint rows | per-hop K1-K6 | prev-reach1 | 5000 | [failures/2026-09-19-hop-distance-plateau-and-dilution.md](../lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md) | stands; the staircase replicates |
| B49 | 09-19 | g3 +0.083 at pass 2; g4 +0.046 at pass 3; exact 0.000 before pass g-1 | planted pair benefit | prev-reach1 | 5000 | [failures/2026-09-19-hop-distance-plateau-and-dilution.md](../lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md) | stands (exact induction test) |
| B50 | 09-19 | g1 +0.181 (d1) to +0.291 (d6) under the cut | planted own-span content, re-supplied each pass | prev-reach1 | 5000 | [failures/2026-09-19-hop-distance-plateau-and-dilution.md](../lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md) | stands; passes refine re-supplied content |
| B51 | 09-19 | trained 0.999 to 1.000 vs random 0.668 at hop 4; d192 8 of 15 vs d96 2 of 15 | per-fact probe; solve count | toy eliminate6 | toy | [successes/2026-09-19-toy-eliminate6-hop-distance.md](../lab/experiments/successes/2026-09-19-toy-eliminate6-hop-distance.md) | stands (toy) |
| B53 | 09-19 | gain(16) +0.0246 (sigma 1.0) / +0.0109 (sigma 0.3) | best-of-16 oracle over entry noise | slot-spandec-strict | 5000 | [failures/2026-09-18-sample-oracle-gate.md](../lab/experiments/failures/2026-09-18-sample-oracle-gate.md) | qualified: growth with depth -0.0005 / -0.0008; inference-time sampling on a deterministically trained map (Wolfe's correction, [failures/2026-09-19-sample-oracle-adapted-reader.md](../lab/experiments/failures/2026-09-19-sample-oracle-adapted-reader.md)) |
| B54 | 09-19 | gain(16) +0.0388 (sigma 1.0) / +0.0149 (sigma 0.3) | same, adapted reader | tul-code-target-uf | 30000 | [failures/2026-09-19-sample-oracle-adapted-reader.md](../lab/experiments/failures/2026-09-19-sample-oracle-adapted-reader.md) | qualified: same |
| B58 | 09-19 | single - oracle 0.1069 | oracle over streams | fan4 (cosine repel) | 5000 | [failures/2026-09-19-lxtul-fan4.md](../lab/experiments/failures/2026-09-19-lxtul-fan4.md) | retracted as choice: 0.099 is a constant offset; oracle - mixed 0.0078 (same file) |
| B55 | 09-20 | K1-K6 +0.0404; h1 +0.1033, h2 +0.0846 | token and per-hop K1-K6 | carry-sum on prev-reach1 | 5000 | [failures/2026-09-19-loop-carry-prev-reach1.md](../lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md) | qualified: a worse depth 1; far hops negative; state unbounded (same file) |
| B56 | 09-20 | K1-K6 +0.0086; h2 +0.0530, h3 +0.0359 | token and per-hop K1-K6 | carry-gate | 5000 | [failures/2026-09-19-loop-carry-prev-reach1.md](../lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md) | qualified: same |
| B57 | 09-20 | g2 under the cut +0.108 (d2) to +0.210 (d6) (ruler +0.148 to +0.036) | planted pair | carry-sum | 5000 | [failures/2026-09-19-loop-carry-prev-reach1.md](../lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md) | stands: re-supply stops the decay |
| B59 | 09-20 | K1-K6 +0.0061 [+0.0057, +0.0066]; K3-K6 +0.0007 | token K1-K6 | fan4-epi | 5000 | [failures/2026-09-20-lxtul-fan4-epi.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epi.md) | qualified: a worse depth 1, not a better depth 6 (same file) |
| B60 | 09-20 | oracle - mixed 0.0182; rank 2.823 | oracle over streams; stream rank | fan4-epivol | 5000 | [failures/2026-09-20-lxtul-fan4-epivol.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md) | qualified: inside the 0.008 to 0.018 band every fan arm reads with or without a term (same file) |
| B61 | 09-20 | K1-K6 +0.0076; K3-K6 +0.0013 | token K1-K6 | fan4-select | 5000 | [failures/2026-09-20-lxtul-fan4-select.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-select.md) | qualified: 0.0861 behind pk4 at depth 6 (same file) |
| B62 | 09-20 | oracle 0.113 below the best single stream | oracle over streams | fan4-select | 5000 | [failures/2026-09-20-lxtul-fan4-select.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-select.md) | qualified: near-copy streams give 0.112 ([failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md)) |
| B63 | 09-20 | read cashes 0.056 of 0.099; regret 0.042 | best single minus deployed; oracle | fan4-all | 5000 | [successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md) | qualified: near-copy floor, regret 0.045 at cos 0.99 ([failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md)) |
| B64 | 09-20 | 0.0342 better [0.0316, 0.0367] at depth 6 | paired CE vs pk4 | fan4-all | 5000 | [successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md) | stands as a 5k reader reading (not a ranking) |
| B65 | 09-20 | +0.0049 [+0.0044, +0.0055]; span decoder K1-K6 +0.0314 | token K1-K6; decoder K1-K6 | fan4-all | 5000 | [successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md) | stands; the token curve is at the floor |
| B34 | 09-21 | k16 - k1 -0.0068 [-0.0089, -0.0048] | 8-draw marginal | tul-code-cfg-tlow | 20000 | [failures/2026-09-21-lctul-cfg-tlow.md](../lab/experiments/failures/2026-09-21-lctul-cfg-tlow.md) | stands; small |
| B66 | 09-21 | +0.0102 [+0.0096, +0.0109]; K3-K6 +0.0008 | token K1-K6 | fan4-all-fp0 | 5000 | [failures/2026-09-21-lxtul-fan4-all-fp0.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-fp0.md) | qualified: the extra is pass 1's (same file) |
| B67 | 09-21 | +0.0089 [+0.0082, +0.0096]; K3-K6 +0.0016 [+0.0013, +0.0018] | token K1-K6 | fan4-all-trig | 5000 | [failures/2026-09-21-lxtul-fan4-all-trig.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-trig.md) | qualified: same shape as fp0; exit rank collapses to 1.43 |
| B68 | 09-21 | +0.0045 [+0.0034, +0.0054]; K3-K6 -0.0008 | token K1-K6 | strict-np0 | 5000 | [failures/2026-09-21-strict-np0.md](../lab/experiments/failures/2026-09-21-strict-np0.md) | qualified: past pass 3 the curve goes the wrong way; 0.1652 behind strict |
| B69 | 09-21 | +0.5575 (5k +0.688) | forced-depth CE through the frozen coda | tul-code-target-denoise | 20000 | [failures/2026-09-21-lxtul-loop-denoise.md](../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md) | retracted as depth: genericity (same file) |
| B70 | 09-22 | +0.0554 (reach all +0.0378, span-h +0.0325) | plain-loop token K1-K6 | budget-web-reach1 | 5000 | [failures/2026-09-21-span-reach-split.md](../lab/experiments/failures/2026-09-21-span-reach-split.md) | stands (hand-checked in [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md)); an unscored observation |
| B71 | 09-22 | +0.0202 [+0.0193, +0.0212]; K3-K6 +0.0025 | token K1-K6 | LXTUL-R Step 1 (leaky) | 5000 | [failures/2026-09-22-lxtul-r-step1.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1.md) | qualified: method fault, about four slots relayed per pass (same file) |
| B72 | 09-22 | +0.0261 [+0.0251, +0.0273]; K3-K6 +0.0057 [+0.0053, +0.0062]; natural tokens +0.0255 | token K1-K6 | LXTUL-R Step 1b | 5000 | [failures/2026-09-22-lxtul-r-step1b.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md) | qualified: 0.0265 behind fp0 at depth 6 and 0.0527 at depth 1, a tax refund; Wolfe: "a mixed result" (same file) |
| B73 | 09-22 | g3 exact 0.000 at d1, +0.048 at d2; g4 +0.036 at d3 | planted pair (valid run) | Step 1b | 5000 | [failures/2026-09-22-lxtul-r-step1b.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md) | stands as a leak check; planted CE 4.7 nats above uniform (same file) |
| B74 | 09-22 | +0.0234 [+0.0222, +0.0246] / +0.0204 [+0.0194, +0.0214] | token K1-K6 | Step 2 C1 persist / C2 hist1 | 5000 | [failures/2026-09-22-lxtul-r-step2-panel.md](../lab/experiments/failures/2026-09-22-lxtul-r-step2-panel.md) | qualified: relay; the same depth-6 CE as Step 1b within 0.002 |
| B75 | 09-23 | 0.0166 [0.0142, 0.0191] / 0.0123 [0.0102, 0.0143] better | paired CE vs ruler, depth 6 | strict-bcast / strict-bcast-all | 5000 | [failures/2026-09-22-arc-slot-token-like-read.md](../lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md) | qualified: the direct read of the exit state, not the passes (K1-K6 +0.0016 / +0.0013) |
| B76 | 09-23 | looped better by 0.0076 [0.0047, 0.0103] (5k), 0.0075 (10k), 0.0083 (15k), 0.0074 [0.0045, 0.0104] (20k) | d1 twin pair (20k schedule) | slot-spandec-strict-20k vs strict-norecur-20k | 5000 to 20000 | [failures/2026-09-22-arc-slot-token-like-read.md](../lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md) | stands; constant, no growth with horizon |
| B77 | 09-23 | width gain 0.159 (4 samples), 0.215 (16) | ce_prior@1 - ce_iw@N | lxtul-g | 5000 | [failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) | qualified: on trajectories the model was not trained to produce; ce_iw@4 +0.5901 worse than the ruler (same file) |
| B78 | 09-23 | depth 6 better than depth 1 by 0.0123 [0.0093, 0.0150] | ce_iw@4 at forced depths | lxtul-g | 5000 | [failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) | qualified: same; single-sample curve flat |
| B79 | 09-23 | width gain 0.47 (4), 0.77 (16) | ce_prior@1 - ce_iw@N | lxtul-g-b1 | 5000 | [failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) | qualified: same; the run ascended its own objective after step 3000 |
| B80 | 09-23 | ce_zero - ce_post +0.116 (gap 0.0008) | worth of the cells in the collapsed phase | lxtul-g-b1 | 2500 | [failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) (addendum 16:13) | qualified: a near-deterministic loop; no verdict change |
| B81 | 09-23 | +0.0037 [0.0034, 0.0041] | token K1-K6 | lxtul-gk1 (K = 1) | 5000 | vlt thread lxtul-fan4 entry 139 (unfiled) | qualified: the loop switched its noise off (sigma/r 0.1 to 0.00031); +0.0038 worse than the ruler at depth 6 |
| B82 | 09-23 | training width gain 0.0065 | tul/gk_width_gain | lxtul-gk4, first run (43bb234) | 5000 | vlt thread lxtul-fan4 entry 140; [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) (amendment 18:59) | retracted: confounded, per-rollout dropout masks; fixed in 7d44ed7 |
| B83 | 09-23 | width gain at 4 samples 0.00006 (gk4 first run), 0.00007 (gk1) | eval width gain ce_prior@1 − ce_iw@4, dropout off | lxtul-gk4 (first run) / lxtul-gk1 | 5000 | `/home/wolfe/morph-scratch/arc/results/2026-09-23-lxtul-gk/lxtul_g_probe_lxtul-gk4_5000.json`, `lxtul_g_probe_lxtul-gk1_5000.json` (outside the repo) | qualified: the prior noise had collapsed (sigma/r 0.00029 / 0.00034); gk4 first run confounded (7d44ed7); added at merge |

lxtul-gk4-shared (filed 2026-09-23, [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)) adds no positive row: its K1−K6 +0.0036 sits at
the floor's top edge and its width gain is 0.00005. Its −0.0044 against the ruler is inside
the seed floor.

## 7. Retractions and qualifications

One table for both parts, in order of the original claim. "Same file" means the file in
the second column.

| claim | original number and source | correction and source | date corrected |
|---|---|---|---|
| "The loop destroys about 10x more diversity than pooling" | [successes/2026-08-24-tul-span-pooling-law.md](../lab/experiments/successes/2026-08-24-tul-span-pooling-law.md) ln 103 | withdrawn: two different measures (centred input rank vs uncentred loop rank), [lab/divergence/takeover-campaign.md](../lab/divergence/takeover-campaign.md) ln 135-137 | 2026-08-24 |
| Every loop, plan and Shapley reading before the leak fix (A1, A2, and pre-fix A3 arms) | the A1 and A2 tables | qualified: retention_carry leaked the future until the fix; "no depth/loop claim is admissible without a carry-off ... sweep", [.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md](../.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md) ln 81-88. None of these readings was re-read with the carry off | 2026-08-31 (fix) |
| Gate beats A1 by 0.1054 | [results/2026-08-23-tul-gate-bakeoff.md](../lab/experiments/results/2026-08-23-tul-gate-bakeoff.md) ln 29 | qualified: the leak (+0.1433) is larger than the headline, same file ln 186 | 2026-08-23 |
| CW1 beats CW2 by 0.009 | [.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md](../.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md) ln 19 | qualified: a stratified re-score favours random tokens in one stratum, same note ln 35 | not dated in the note |
| tul_v1a2b loop worth +0.0058 to +0.0107, p 0.0286 | [failures/2026-08-27-warmup-sigreg-ntpdrop.md](../lab/experiments/failures/2026-08-27-warmup-sigreg-ntpdrop.md) ln 268-269 | qualified: post-hoc, same file ln 273-276; the fresh-seed confirmation ([planned/2026-08-27-mux-matched-control-confirmation.md](../lab/experiments/planned/2026-08-27-mux-matched-control-confirmation.md)) has no results | 2026-08-27 |
| "The restriction moves the plan 2x to 10x" | [failures/2026-08-27-tg-restriction.md](../lab/experiments/failures/2026-08-27-tg-restriction.md) (original verdict) | withdrawn: the ablation's fallback differs between arms, same file ln 105-113 | 2026-08-28 |
| "0 of 4 tul_tg2 seeds held" (no takeover) | TG filings of 2026-08-27 | qualified: every "held" arm stopped at 3500; the control took over 57 steps later, [planned/2026-08-28-does-the-slot-apparatus-pay.md](../lab/experiments/planned/2026-08-28-does-the-slot-apparatus-pay.md) ln 119-137 | 2026-08-28 |
| l2cap earns depth | K1−K6 +0.2328, [successes/2026-08-29-tul-loop-ladder.md](../lab/experiments/successes/2026-08-29-tul-loop-ladder.md) ln 150; [successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md) ln 81 | retracted: carry off reads −1.1198, same audit ln 82; future corruption collapses the earning 105 %, [successes/2026-08-31-future-leak-attribution.md](../lab/experiments/successes/2026-08-31-future-leak-attribution.md) ln 89 | 2026-08-31 |
| "Contractivity control is a REQUIREMENT for a trainable iterated write" (the identity-escape law) | [.agents/notes/implemented/architecture/2026-08-29-loop-ladder.md](../.agents/notes/implemented/architecture/2026-08-29-loop-ladder.md) ln 53-56 | retracted as built: every "flattening" ablation flattened a leak, [.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md](../.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md) ln 5-18 | 2026-08-31 |
| The paid loop is TUL | A2 +0.1685, [successes/2026-09-01-a2-paid-loop.md](../lab/experiments/successes/2026-09-01-a2-paid-loop.md) ln 104; shipped 2026-09-03 | arm rejected, numbers stand: "totally wrong", [.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md](../.agents/notes/proposed/architecture/2026-09-03-tul-loop-contribution-drawing-board.md) ln 10; standing call in [.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md) ln 73 | 2026-09-03; standing 2026-09-22 |
| E2 forecast K3−K6 +0.0077 | [failures/2026-09-04-arc-e2-iteration-conditioning.md](../lab/experiments/failures/2026-09-04-arc-e2-iteration-conditioning.md) ln 200 | retracted: pre-onset; the stable rerun reads +0.0001, same file ln 175, 202 | 2026-09-04 |
| X1 own-loss +0.4185 | [failures/2026-09-04-tul-clip-through-time.md](../lab/experiments/failures/2026-09-04-tul-clip-through-time.md) ln 198 | retracted: a model 0.75 nats worse, "not a THINK in substance", same file ln 216 | 2026-09-04 |
| E6 K3−K6 +0.277 | [successes/2026-09-07-arc-e6-deep-recurrence-draw.md](../lab/experiments/successes/2026-09-07-arc-e6-deep-recurrence-draw.md) ln 83 | qualified: 0.104 worse than mean 6 (same file ln 89); 27 % of samples silenced by truncated BPTT, [.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md](../.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md) ln 14 | 2026-09-07 |
| E7 +0.0263, forecast +0.823 | [failures/2026-09-07-arc-e7-block-loop.md](../lab/experiments/failures/2026-09-07-arc-e7-block-loop.md) ln 97-98 | retracted: pre-onset; detonated at 2712, same file ln 126 | 2026-09-07 |
| E8 +0.0526 | [failures/2026-09-07-arc-e8-multi-token-coda.md](../lab/experiments/failures/2026-09-07-arc-e8-multi-token-coda.md) ln 100 | qualified: next-token CE +0.3445 worse, same file ln 101 | 2026-09-07 |
| E14 +0.0425 / +0.0601 | [failures/2026-09-07-arc-e14-expansive-dial.md](../lab/experiments/failures/2026-09-07-arc-e14-expansive-dial.md) ln 133-134 | retracted: pre-onset; both detonated (3877, 4639), same file ln 120 | 2026-09-07 |
| E16 mask +0.406 (Olympiad) | [failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md](../lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md) ln 161 | retracted as depth: depth 1 is broken (1.31 nats), same file ln 202 | 2026-09-08 |
| Mask-arm token dependence: E4 +0.0209, E13 +0.0489, E18 k8 +0.0486 | [successes/2026-09-04-arc-e4-mask-under-constraint.md](../lab/experiments/successes/2026-09-04-arc-e4-mask-under-constraint.md) ln 76; [failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 145; [planned/2026-09-04-loop-contribution-arc.md](../lab/experiments/planned/2026-09-04-loop-contribution-arc.md) ln 339 | qualified as an absmean artefact: +0.0009 under norm_match, [failures/2026-09-10-arc-slot-mux-mask-norm-match.md](../lab/experiments/failures/2026-09-10-arc-slot-mux-mask-norm-match.md) ln 219 | 2026-09-10 |
| The plain loop's 0.185 as the yardstick for slot arms | [successes/2026-09-09-arc-per-pass-strength.md](../lab/experiments/successes/2026-09-09-arc-per-pass-strength.md) ln 112 (noise entry) | qualified: under the prelude entry the same rule reads 0.033, [failures/2026-09-09-arc-slot-loop-norm-match.md](../lab/experiments/failures/2026-09-09-arc-slot-loop-norm-match.md) ln 147 | 2026-09-09 |
| Density arms +0.043 / +0.074 | [failures/2026-09-09-arc-density-panel.md](../lab/experiments/failures/2026-09-09-arc-density-panel.md) ln 136-137 | qualified: "dependence without computation", same file ln 163 | 2026-09-09 |
| slot-unpack-fixed-depth +0.1168 | [failures/2026-09-10-arc-slot-map-levers.md](../lab/experiments/failures/2026-09-10-arc-slot-map-levers.md) ln 143 | qualified: "depth dependence without depth value", same file ln 167 | 2026-09-10 |
| Staged forecast +0.0670 | [successes/2026-09-10-arc-slot-mnext-staged.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged.md) ln 212 | qualified: the exit forecast equals the ruler's (6.7698 vs 6.7724), same line; 0.1609 behind plain at 20k, [successes/2026-09-10-arc-slot-mnext-staged-20k.md](../lab/experiments/successes/2026-09-10-arc-slot-mnext-staged-20k.md) ln 255 | 2026-09-10 |
| z-optimize headroom −0.9442 / −2.6258 (hindsight) | [results/2026-09-10-slot-z-optimize/README.md](../lab/experiments/results/2026-09-10-slot-z-optimize/README.md) ln 139, 160 (slot-mux and slot-unpack arms); −1.5516 on the strict arm, [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) | retracted as headroom: a causal fit on the STRICT arm reads +0.2517 [+0.2265, +0.2730] worse than the loop's z, same latent-z file ln 449-453. The causal fit ran on a different arm than the first two hindsight numbers | 2026-09-12 (measured 09-13) |
| prev-reach1 +0.0163, prev-reach2 +0.0127 | [failures/2026-09-12-arc-strict-geometry.md](../lab/experiments/failures/2026-09-12-arc-strict-geometry.md) ln 289-290 | qualified: a forced relay, "z has to hold the history" (same file ln 248-250); restriction geometries closed by Wolfe, [.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md) | 2026-09-12; lane closed 2026-09-22 |
| Strict d1 twin +0.0020 at 5k | [failures/2026-09-12-arc-latent-z-gradient.md](../lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md) ln 415 | qualified: CI crosses 0; a 20k-schedule pair reads a constant 0.0074 to 0.0083, [failures/2026-09-22-arc-slot-token-like-read.md](../lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md) ln 115 | 2026-09-22 |
| First identical-target grid (pass_h1 −0.0200) | [failures/2026-09-12-arc-objective-arms.md](../lab/experiments/failures/2026-09-12-arc-objective-arms.md) | retracted: the instrument ran a bare `_tul_front` (strict prelude unrestricted); fixed at 7a24adf, same file | 2026-09-12/13 |
| Slot rank 5.7598 / cos 0.7104 | [failures/2026-09-13-arc-thought-register.md](../lab/experiments/failures/2026-09-13-arc-thought-register.md) | retracted: bare-front probe; the trainer-recipe reading is 13.8466 / 0.5201, [results/2026-09-13-rank-anatomy/README.md](../lab/experiments/results/2026-09-13-rank-anatomy/README.md) | 2026-09-13 |
| Trajectory prefix +0.0342 | [failures/2026-09-13-arc-trajectory-prefix.md](../lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md) | retracted: forced-depth pad cells; the same-width trajrep reads +0.0019, same file | 2026-09-13 |
| Register m4 0.0217 better; vq arms | [failures/2026-09-13-arc-thought-register.md](../lab/experiments/failures/2026-09-13-arc-thought-register.md); [mixed/2026-09-13-arc-discrete-thought-vq.md](../lab/experiments/mixed/2026-09-13-arc-discrete-thought-vq.md) | qualified: prefix width, not loop; the quantizer costs 0.059 beyond its width, [successes/2026-09-21-strict-pk8-ruler.md](../lab/experiments/successes/2026-09-21-strict-pk8-ruler.md) | 2026-09-13; 2026-09-21 |
| Fixed and clamped depth rungs +0.2294 / +0.5259 | [failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) | qualified: fixed depth is not sampled depth (2026-09-14 correction, same file) | 2026-09-14 |
| horizon-fixed6 +0.278 | [failures/2026-09-14-arc-horizon-passes.md](../lab/experiments/failures/2026-09-14-arc-horizon-passes.md) | retracted as depth: trained only at depth 6, out of distribution elsewhere, same file | 2026-09-14 |
| Arm A K1−K6 +0.0323 | [planned/2026-09-17-lctul-target-slot-loop.md](../lab/experiments/planned/2026-09-17-lctul-target-slot-loop.md) | retracted as depth: a frozen coda's curve measures genericity, [.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md](../.agents/notes/implemented/testing/2026-09-18-frozen-coda-k-curve-measures-genericity.md) | 2026-09-18 |
| Code-only draws (own cos 0.99) and code-only K-curve (+0.0698 ...) | [failures/2026-09-17-lctul-code-only-ref.md](../lab/experiments/failures/2026-09-17-lctul-code-only-ref.md) | retracted: a frozen E on a live front is not a fixed target; the reader sits above uniform CE (12.64 > 10.80), same file | 2026-09-17 / 2026-09-18 |
| "The reader is not the limit" | B2 (cond4, ultralight) | qualified: true for the strict coda's capacity; a frozen reader reads the same cell at −4.64, an adapted one at +0.177, [successes/2026-09-17-lctul-target-unfreeze.md](../lab/experiments/successes/2026-09-17-lctul-target-unfreeze.md) | 2026-09-17/18 |
| "Dilution" of carried content | [failures/2026-09-18-hop-distance-earning.md](../lab/experiments/failures/2026-09-18-hop-distance-earning.md) | withdrawn for carried content: per-pass decay, [failures/2026-09-19-hop-distance-plateau-and-dilution.md](../lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md) | 2026-09-19 |
| "A contractive loop cannot hold K alternatives apart" (sample oracle as a branch gate) | [failures/2026-09-19-sample-oracle-adapted-reader.md](../lab/experiments/failures/2026-09-19-sample-oracle-adapted-reader.md) | withdrawn as a proxy by Wolfe: the checkpoints were trained deterministic with the fixed-point term, same file | 2026-09-19 |
| fan4 single − oracle 0.1069 read as choice | [failures/2026-09-19-lxtul-fan4.md](../lab/experiments/failures/2026-09-19-lxtul-fan4.md) | qualified: 0.099 is a constant offset per stream, 0.008 is choice, same file | 2026-09-19 |
| epivol "closes the K-stream branch" | [failures/2026-09-20-lxtul-fan4-epivol.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-epivol.md) | reversed the same day by the select arm, [failures/2026-09-20-lxtul-fan4-select.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-select.md) | 2026-09-20 |
| select oracle 0.113; fan4-all cashes 0.056 of 0.099 | [failures/2026-09-20-lxtul-fan4-select.md](../lab/experiments/failures/2026-09-20-lxtul-fan4-select.md); [successes/2026-09-20-lxtul-fan4-all.md](../lab/experiments/successes/2026-09-20-lxtul-fan4-all.md) | qualified: near-copy streams (cos 0.99) read regret 0.045 and an oracle 0.112 below the best single, [failures/2026-09-21-lxtul-fan4-all-noise.md](../lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md). The synthesis note still quotes 0.056 of 0.099 without this floor | 2026-09-21 |
| Denoise K1−K6 +0.5575 | [failures/2026-09-21-lxtul-loop-denoise.md](../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md) | retracted as depth: genericity, same file | 2026-09-21 |
| "One-pass-easy" as a law (pass 1 must be unable to finish the job) | [failures/2026-09-21-lxtul-loop-denoise.md](../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md) | qualified by Wolfe: a helper, "not a universal truth", [.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md) (Decision) | 2026-09-22 |
| Step 0 "learned single-block trade" | [failures/2026-09-21-span-reach-split.md](../lab/experiments/failures/2026-09-21-span-reach-split.md) | withdrawn: a bad draw (coda seeds differ by 0.152), [failures/2026-09-22-coda-seed-twin.md](../lab/experiments/failures/2026-09-22-coda-seed-twin.md) | 2026-09-22 |
| LXTUL-R Step 1 +0.0202 | [failures/2026-09-22-lxtul-r-step1.md](../lab/experiments/failures/2026-09-22-lxtul-r-step1.md) | qualified as a method fault: the conv and value shift relayed about four slots per pass; fixed at 9b430d3, same file | 2026-09-22 |
| Math-corpus slot K-curves (Olympiad +0.0111, Sudoku +0.0049) | [failures/2026-09-12-arc-math-under-norm-match.md](../lab/experiments/failures/2026-09-12-arc-math-under-norm-match.md) | set aside: "math and Sudoku were red herrings", [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) ln 13 | 2026-09-22 |
| Matched-compute nats as a verdict (slot 0.2536 behind plain) | [successes/2026-09-11-arc-span-decoder.md](../lab/experiments/successes/2026-09-11-arc-span-decoder.md) ln 363 | the number stands; the standing call removes it as a verdict instrument, [.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md) | not dated in any repo file |
| LXTUL-G width gain 0.159, depth gain 0.0123 | [failures/2026-09-23-lxtul-g-panel.md](../lab/experiments/failures/2026-09-23-lxtul-g-panel.md) ln 205, 207 | qualified: "on a trajectory the model was not trained to produce", same file | 2026-09-23 |
| gk4 (first run) training width gain 0.0065 | [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) ln 116-117; vlt `lxtul-fan4` entry 140 | retracted as evidence: rollouts drew independent dropout masks; fixed 7d44ed7, amendment f53e988, [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) ln 113-124. Eval width gain with dropout off: 0.00006 (`lxtul_g_probe_lxtul-gk4_5000.json`) | 2026-09-23 |
| E13 "+0.0172 forecast K6−K12" | [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) ln 79 | mislabel: +0.0172 is forecast K3−K6; forecast K6−K12 is +0.0020, [failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 129 | 2026-09-23 (this history) |

## 8. Docs that still state a retracted number

These files were read on 2026-09-23. None was edited. Each one states a number or a
claim that a later file retracts or qualifies, and carries no correction.

| file | line | what it says | what corrects it |
|---|---|---|---|
| [CLAUDE.md](../CLAUDE.md) (repo root) | 66 | "the l2cap recipe ... is the ONE recipe whose loop earns depth (0.233 nats)" | [successes/2026-08-31-carry-leak-audit.md](../lab/experiments/successes/2026-08-31-carry-leak-audit.md) ln 81-82 (carry off −1.1198) |
| [.agents/notes/implemented/architecture/2026-08-29-loop-ladder.md](../.agents/notes/implemented/architecture/2026-08-29-loop-ladder.md) | 53 | l2cap is "the campaign's first load-bearing loop ... 0.233 nats of depth-earned CE" | same audit; the banner in [.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md](../.agents/notes/implemented/architecture/2026-08-30-l2cap-winning-recipe.md) ln 5-18 |
| [.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md](../.agents/notes/implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md) | 32 | the l2cap checkpoint is "the best substrate that will ever exist" | same audit |
| [.agents/notes/archived/architecture/2026-08-30-dbfix-program.md](../.agents/notes/archived/architecture/2026-08-30-dbfix-program.md) | 9 | "load-bearing loop (0.233 nats depth-earned CE)" | same audit (the note is archived, so it must not be edited; it needs a pointer from a live note instead) |
| [.agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md](../.agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md) | 8, 27 | "0.233 nats depth-earned"; "G0 ... the l2cap control ... 0.233 nats" | same audit |
| [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) | 79 | E13 "+0.0172" labelled "forecast K6-K12" | [failures/2026-09-07-arc-e13-m12-panel.md](../lab/experiments/failures/2026-09-07-arc-e13-m12-panel.md) ln 129: +0.0172 is forecast K3−K6; K6−K12 is +0.0020 |
| [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) | 111 | tul_v1a2b loop worth marked "standing, small, post-hoc" | no pre-fix caveat: [.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md](../.agents/notes/implemented/bug-fix/2026-08-23-retention-carry-breaks-causality.md) ln 81-88 |
| [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) | 142 | region Shapley marked "standing" | same pre-fix caveat |
| [docs/9-26-TUL-run-history-IMPORTANT.md](9-26-TUL-run-history-IMPORTANT.md) | 112 | TG2 loop worth marked "suspect: seed-inconsistent" | status is right, but it also lacks the pre-fix caveat |

## 9. Lessons that carried forward

Part A listed 12 lessons and Part B listed 11. Where two said the same thing, they are
merged here. The source part is named after each lesson.

1. **Run the causal check before you believe a depth curve.** l2cap's 0.233 was all
   future-reading. The carry-off sweep became a standing control on 2026-08-31, and
   every loop number from before that date comes from a leak-reliant model. The later
   form of the same lesson: an attention mask does not bound the conv and value shift, so
   check a geometry with a perturbation test (LXTUL-R Step 1). (A, B)
2. **A K-curve measures dependence, not value.** E6, the density arms, the fixed-depth
   slot arm, the zero-entry probe, the fixed depth rungs and horizon-fixed6 all read large
   K-curves on models that were not better. Pair a K-curve with paired CE against a
   control or a depth-1 twin. Fixed depth is not sampled depth. (A, B)
3. **A reading taken before a detonation is not an earning.** E2, E7, E14 and X1 read
   their best numbers before onset. (A)
4. **Stability is not contribution.** The gain constraint and the fixed-point term hold
   the map. Neither moved the K-curve. The fixed-point term holds scale, not rank (fp0).
   (A, B)
5. **Depth follows the loss share that has no shallower route (Condition A).** Plain
   loop with noise entry 0.185, with prelude entry 0.033, the slot side channel 0.002
   ([failures/2026-09-21-lxtul-loop-denoise.md](../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md)). The coda's direct read of
   a cell, the prelude cells in the mask geometry, the aux losses and teacher forcing
   were all shallower routes that took the job the loop was meant to do. (A, B)
6. **The loss must read the pass's output with a steep gradient (Condition B).** A reader
   that uses the cell is not a loop that fills it: worth 0.811 with a K-curve of 0.0006.
   Energies, critics, per-pass targets and a token-trained core all left the passes idle
   because the coda's loss is flat around the loop's states (0.001 to 0.003 nats per
   pass). The same six blocks earn 20x more on tokens (0.0102) than on slot states
   (0.0005). (A, B)
7. **A per-pass target is met in one step unless pass 1 cannot meet it.** Staged, oracle,
   gradpass, per-pass plan, critic, LoopMTP, progressive and graded targets were all met
   by pass 1. Wolfe qualified this on 2026-09-22: a job pass 1 cannot finish is a helper,
   not a law, because the plain loop earns with a strong pass 1. (A, B)
8. **The per-pass map must be strong enough to do work, and the entry sets the size of
   the curve.** absmean starved the core; norm_match moved the plain loop from 0.033 to
   0.185 under the noise entry. Under the prelude entry the same rule reads 0.033. Compare
   K-curves only within one entry. (A)
9. **Forced relays earn, and the earning is a refund.** Under reach-limited geometry,
   content h spans back arrives at pass h−1 with exact zeros before it (the hop
   staircase), and LXTUL-R reached +0.026. Every such arm sits behind a direct read on CE.
   Wolfe closed restriction geometries on 2026-09-22. (A, B)
10. **Width, rank and diversity move instruments, not the loop.** Four cells, VQ codes,
    eight prefix cells, three diversity terms, noise, lineages and trigger re-supply all
    moved a rank or a cosine and left K3−K6 at the floor. Prefix width is a real reader
    lever (0.022 at four cells, 0.035 at eight), and must be named as width. (B)
11. **Candidates must be read with the span's evidence, not chosen before it.** The
    select gate cashed 0.014 of 0.113; writing all cells cashed 0.056 of 0.099. The
    oracle-over-streams gap has a floor of about 0.04 on near-copies. (B)
12. **Train on what you deploy.** The fan's select arm (oracle write vs gate write) and
    LXTUL-G (posterior cells vs prior cells) failed the same way. GK applied the fix
    and closed the gap (0.7533 nats), but its learned noise then collapsed, so the fix
    alone earned no width or depth. (B)
13. **Horizon matters only for the loop that acts on tokens.** The paid loop's earning
    grew 0.041 to 0.104 over 20k steps; the norm_match plain loop grew 0.136 to 0.170 and
    its value over the depth-1 twin reached 0.0674. The slot loop's value over its twin
    stayed a constant 0.0075 from 5k to 20k. No slot-state positive grew with steps. (A, B)
14. **Instruments lie in specific, now-named ways.** A bare `_tul_front` ran the strict
    prelude unrestricted (rank 5.76, the void horizon grid). A frozen coda's depth curve
    measures genericity. A reader above uniform CE is not a reader. A hindsight z-fit
    uses the answer. A padded trajectory write is a width artefact. The oracle gap has a
    near-copy floor. Seed spread can be 0.15 on one config and 0.004 on another. Dropout
    draws per rollout unless shared. Read the source line, not the ledger: this history
    found one stale headline in the root [CLAUDE.md](../CLAUDE.md), four uncorrected notes, a
    mislabelled ledger row and a planned file with no results. (A, B)
15. **Standing calls.** Never pass or fail an arm on a cosine. Never read a 5k CE gap as a
    ranking. Never argue that web text does not need depth. These are collected in the
    synthesis note ([.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md](../.agents/notes/implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md)). (B)
16. **Positives to build on, as named at the end of the range:** the plain loop's
    matched-step win over its twin (0.0674 at 20k), the plain loop under reach 1
    (+0.0554), the core's token read (+0.0102), the math-corpus slot K-curves (+0.0111,
    +0.0049, set aside), and the width and depth leads under LXTUL-G's Bayesian read
    (0.159, 0.0123). GK tested them on deployed rollouts: they did not survive, because
    the learned noise collapses when the model trains on its own rollouts. (B)

## 10. What could not be verified

From the merge (2026-09-23):

- The GK noise scale differs by source. vlt `lxtul-fan4` entry 139 gives gk1 sigma/r
  0.00031; the probe JSON gives 0.000339. Entry 140 and the GK prereg amendment give gk4
  2.8e-4 at step 4980 (gk1 3.1e-4); the probe JSON gives 0.000290. The vlt numbers are trainer logs and the JSON numbers are
  eval probes on 192 rows. Both are cited.
- The gk1 final val (4.4238) and K1−K6 (+0.0037) were only in vlt entry 139 at the merge.
  They are now filed with the GK panel and recomputed from the sweep JSON. (The gk4 training width gain, 0.0065, is in the GK prereg amendment,
  [failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md) ln 116-117.)
- The depth-ladder prereg quotes "0.004 at 5k" for the plain loop against its depth-1
  twin ([failures/2026-09-13-arc-depth-ladder-ship.md](../lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md) ln 73). I did not
  find the filing that measured it.
- Resolved 2026-09-23 21:24: lxtul-gk4-shared is filed ([failures/2026-09-23-lxtul-gk-multisample.md](../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)); the GK sweeps, probes
  and paired readings now live in `lab/experiments/results/2026-09-23-lxtul-gk/`.

From Part A:

- No number was recomputed from JSON or wandb.
- About 10 line numbers in the A tables came from the extraction agents and were not
  grepped again by hist-A (mostly A1 and A2 context lines and some flat A7 arms).
- The Huginn span-decoder collapse figure (7.12 nats) appears only in the spec.
- [lab/tul/arms-result.md](../lab/tul/arms-result.md) (A1c over A0c, 0.0562) is not a filed experiment.
- No A1 or A2 reading was re-read with the carry off. The "qualified (pre-fix)" status
  rests on the bug-fix note's rule, not on a measurement.
- The tul_v1a2b confirmation has no results anywhere we searched.
- E18 exists only as a ledger row in the arc master and a mention in the prefix4 filing.
- Several A2 files under `planned/` carry filled Results with an unchanged Status. We
  read them as results.
- The later-correction search went by file basename and arm name. A correction that
  names neither could be missed.

From Part B:

- The LoopMTP token-loop result. An untracked probe output exists
  ([results/2026-09-14-loopmtp-token-loop/](../lab/experiments/results/2026-09-14-loopmtp-token-loop/)); the prereg has no Results,
  and hist-B did not read the probe as a result.
- The date of the "do not score on matched-compute nats" call. The synthesis note lists
  it without a date.
- Numbers that a filing quotes from other runs are cited through the quoting file. Not
  every quoted source was reopened.
- [planned/2026-09-13-arc-rank-levers-center-and-contrast.md](../lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md), the
  2026-09-16 planned files and [planned/2026-09-17-lctul-code-only.md](../lab/experiments/planned/2026-09-17-lctul-code-only.md)
  have no results; there is no evidence those arms ran under those files.
- Resolved in the merge: Part B could not source the gk4 eval width gain (0.00006). It is
  in `lxtul_g_probe_lxtul-gk4_5000.json` (`width_gain@4` 6.22e-5), cited above.
