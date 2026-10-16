# Parcae: the slot loop, plain and flat models at equal training compute

Status: failure (withdrawn 2026-10-14 before any run: Wolfe, loops need more tokens than 5k to reach the knee, so a 5k-budget equal-compute ranking cannot decide anything; the redesign replaced it)

## Question

Training compute is the problem TUL must solve. At equal steps the minimal slot loop (1 cell, pointer)
trails plain + pointer by 0.037 nats CE@6 (paired), at half the FLOPs per step. Does it win at equal
compute? Wolfe's caveat (2026-10-14): Parcae is torch-friendly and TUL is not, so measured wall clock
penalises TUL for engineering. Two rulers bracket the answer: equal measured wall clock (pessimistic for
TUL) and equal FLOPs (TUL at the flat arms' efficiency, optimistic).

## Budget

Plain + pointer row-compact at 5000 steps, batch 6 (bench4, `morph-scratch/abq/pz-bench/bench4.md`):
26.92 TFLOP per step (fwd+bwd 22.41 + optimizer 4.51), 240.5 ms per step. Budget 134.6 PFLOP, 1202 s.

| arm | TFLOP/step | ms/step | steps (ruler) |
| --- | --- | --- | --- |
| minimal, equal FLOPs | 13.19 | 153.4 | 10200 (FLOP) |
| minimal, equal wall clock | 13.19 | 153.4 | 7840 (wall) |
| flat14 + pointer | 13.52 | 123.3 | 9950 (FLOP; wall would give 9750) |
| gpt8 + pointer | 8.54 | 79.8 | 15760 (FLOP; wall would give 15070) |

Reference: plain + pointer 5k s2 (3.7273 CE@6 on minimal's rows). One seed each (seed 1). The schedule
scales with the step count (constant LR, linear cooldown over the last half). `exclude_eval_docs` is on, so
runs past doc 50k do not train on eval rows. The minimal fan has no synced FLOP bench and is not in this
panel.

## Predictions (CE@6 on the same rows via `lxtul.paired_eval --rows-from` the minimal equal-FLOP run)

1. Minimal at equal FLOPs beats plain + pointer 5k by at least 0.03 (70 %).
2. Minimal at equal wall clock beats plain + pointer 5k (60 %).
3. Flat14 + pointer at equal FLOPs beats minimal at equal FLOPs by at least 0.01 (55 %).
4. Gpt8 + pointer at equal FLOPs beats plain + pointer 5k (65 %).
5. Minimal K1-K6 stays within +-0.005 of 0 at both lengths (85 %).

## What each outcome changes

- 1 holds and 3 holds: the slot design saves training compute against the plain loop, but a flat model of
  the same FLOPs is better still; the loop adds nothing over flat, and loop contribution stays the open
  problem.
- 1 holds and 3 fails: the slot channel is worth something at equal compute on its own.
- 1 fails: the minimal slot loop does not solve the training problem at this scale even at TUL's
  speed-of-light FLOPs.

## Method

Queue `morph-scratch/abq/pz-panel/eqc.sh` after the shallow-coda panel (amended 2026-10-14 before any run: the shallow-coda arms go first); testbed checkout; one job at a time under
the GPU lock; paired eval at the end. Amendments get a date and a reason here.

## Results

## Verdict

## Updated hypothesis

(No run. Withdrawn before any data; see Status.)
