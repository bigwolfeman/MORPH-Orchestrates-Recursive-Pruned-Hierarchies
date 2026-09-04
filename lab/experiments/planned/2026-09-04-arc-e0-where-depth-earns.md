# Planned: ARC E0 — where does the loop earn, row by row and offset by offset (eval only)

Status: planned
Date: 2026-09-04 (frozen; eval-only GPU time, Wolfe's call)
Arc: `2026-09-04-loop-contribution-arc.md`, branch (b) TARGET.

## Question

The arc's hypothesis (b) says a loop earns only on the compute-limited part of a loss,
and a next-token or next-span target on web text is mostly entropy-limited. If that is
right, loop earning should NOT concentrate on the rows the model finds hardest (those are
the high-entropy rows), and it should concentrate on the positions where a token depends
on structure the prelude cannot resolve in one pass. This is the cheapest reading of (b)
we can take: no training, the kept checkpoints only.

## Method

Checkpoints: `morph-scratch/checkpoints-keep/notul-20k-wu/step_20000.pt` (the plain
token loop, K1−K6 = 0.04 on 480 rows: the only loop with enough earning to split) and
`step_5000.pt` of the same run (0.037); `tul-a2-20k-wu/step_20000.pt` (the paid loop,
0.100) as the second model. The slot arms have nothing to split (token K1−K6 ≤ 0.0006).

`lab/divergence/token_depth_sweep.py` gains two outputs behind flags, no change to its
numbers: `--per-row` (CE sum and token count per row per depth) and `--per-offset` (CE
sum per depth binned by offset-in-span, spans cut by `BoundaryRule` on the tokens; cap
32). 480 rows, depths 1, 3, 6, 8, batch 3, the same fixed rows the 20k tables used.

Readouts, per model:
1. Row-level: Spearman correlation between a row's depth-1 CE and its earning
   (CE₁ − CE₆), with a bootstrap CI over rows.
2. Offset profile: earning per offset-in-span for offsets 0..31, with the CI.
3. The earning share of the top-decile-by-CE rows against their loss share.

## Predictions (frozen)

- **P0a.** Spearman(row CE₁, row earning) on notul-20k-wu at 20k is ≤ +0.10: **60%**.
  (Earning does not concentrate on hard rows.)
- **P0b.** Earning at offset 0 (the first token after a boundary) exceeds the mean
  earning over offsets 4..31 by at least 2x on notul-20k-wu: **55%**.
- **P0c.** The top decile of rows by CE₁ carries a share of the total earning that is
  below its share of the total loss: **60%**.
- **P0d.** The paid loop (a2) shows the same sign on P0a as notul: **70%**.

## Binding

No launch decision hangs on E0. P0a/P0c TRUE ⇒ E3's staged target keeps the MEMORY job
(compute-limited by construction) on the early iterations. P0b TRUE ⇒ E3 supervises the
span-initial forecast harder than the rest (a ρ closer to 1 is wrong; a first-token
weight is right). All FALSE ⇒ the entropy-limited reading is wrong and E3's split is
chosen by F2 alone (memory early, forecast late).

## Not verified before launch

The per-offset binning on a slot-free model (the rule runs on tokens; never done); that
the 480 rows match the 20k tables' rows bit for bit (the sweep prints the row hash).
