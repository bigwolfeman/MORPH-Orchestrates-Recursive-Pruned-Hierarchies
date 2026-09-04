# Failure: ARC E0 — where does the loop earn, row by row and offset by offset (eval only)

Status: failure
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

## Results (2026-09-04 17:48; `arc/run_e0.sh` at master 40d0173; 480 rows, batch 3, depths 1/3/6/8; files in `results/2026-09-04-arc-e0/`)

The `--profile` path reproduces the ruler exactly (notul@5000: K1 4.0075, K3 3.9723, K6
3.9707 against the panel's 4.0074 / 3.9723 / 3.9707), so the old and new numbers are the
same numbers.

| model | K1−K6 (row mean) | P0a Spearman(row CE₁, row earning) | earning at offset 0 / 1 / 2 / 3 / 4–7 / 8–15 / 16–31 | offset-0 vs mean 4..31 | top-decile rows: loss share / earning share |
|---|---|---|---|---|---|
| notul-20k-wu @20000 | +0.0414 | +0.042 [−0.042, +0.127] | 0.037 / 0.045 / 0.040 / 0.043 / 0.041 / 0.041 / 0.042 | 0.89x | 0.126 / 0.106 |
| notul-20k-wu @5000 | +0.0367 | +0.054 [−0.039, +0.144] | 0.028 / 0.039 / 0.040 / 0.038 / 0.035 / 0.036 / 0.039 | 0.76x | 0.124 / 0.101 |
| tul-a2-20k-wu @20000 (paid loop) | +0.1037 | −0.180 [−0.270, −0.088] | 0.080 / 0.092 / 0.097 / 0.097 / 0.102 / 0.108 / 0.110 | 0.74x | 0.125 / 0.089 |

Every offset bin's CI excludes zero on every model; the profiles are FLAT to within
±15 % of their mean, and the span's first token earns the LEAST on all three.

Scored:

| prediction | credence | verdict |
|---|---|---|
| P0a Spearman ≤ +0.10 on notul@20k | 60% | TRUE (+0.042; the CI reaches +0.127) |
| P0b offset-0 earning ≥ 2x the mean over offsets 4..31 | 55% | FALSE (0.89x, 0.76x, 0.74x: offset 0 earns LESS) |
| P0c top-decile earning share below its loss share | 60% | TRUE on all three |
| P0d a2 shows the same sign on P0a as notul | 70% | FALSE by the letter (+0.04 vs −0.18); both say "not on hard rows" |

## Verdict

failure (P0b and P0d falsified). The reading that survives: depth earning does not
concentrate anywhere. It is spread evenly over every offset in the span and over rows of
every difficulty, and it is slightly SMALLER where the loss is largest (the span's first
token, the hardest decile of rows). The paid loop earns 2.5x the plain loop with the same
flat shape, and its earning tilts AWAY from hard rows (Spearman −0.18). The "hard positions
need more passes" picture is wrong for these targets on this data: extra passes refine the
predictable part of the loss uniformly.

## Updated hypothesis

The loop's earning is a uniform refinement of the predictable part of next-token
prediction, not extra computation where the target is hard. For E3 this means: do NOT
weight the span-initial forecast harder (P0b was the case for it, and it is false); keep
the memory stage (compute-limited by construction) as the early job. For the arc it
means condition (b) is stated wrongly as "a target that needs multi-step computation":
on web text the loop never finds such positions, so (b) needs a target whose whole loss
is compute-limited — the memory (multiplex) target is the one candidate on disk.
