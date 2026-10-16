# Parcae: the published spiral with one level looped at Poisson depth, at 5k

Status: success (filed 2026-10-16)

## Question

The published spiral (chunks 16, 8, 4, 2, one pass each) is the best loop arm against plain Parcae
(3.6912 vs 3.7256 at 0.49x the step), and it has no K-curve. The chunk loops showed depth pays at chunks of
2 and 4 (K1-K6 +0.0063 / +0.0030). Does looping one spiral level at Parcae's Poisson depth give the spiral a
K-curve, and does it keep or improve its CE?

## Arms (branch lxtul/spiral; 5000 steps; batch 6; 2 seeds; pointer head 4)

| arm | levels | train ms (bench, measured) |
| --- | --- | --- |
| spiral, Poisson chunks of 2 | 16, 8, 4 once; 2 at Poisson depth | 234.2 |
| spiral, Poisson chunks of 4 | 16, 8 once; 4 at Poisson depth; 2 once | 199.0 |

References (paired, 510,003 tokens): published spiral 3.6912 (138.0 ms), chunk-2 loop 3.6924 (191.6 ms),
plain + pointer s2 3.7256 (283.0 ms). Wolfe's bar (2026-10-15): judge against loops; flat rows are references.

## Predictions

1. Poisson chunks of 2: 2-seed CE@6 at most 3.6912 (50 %).
2. Poisson chunks of 4: 2-seed CE@6 at most 3.6912 (45 %).
3. Poisson chunks of 2: K1-K6 at least +0.004 (55 %).
4. Poisson chunks of 4: K1-K6 at least +0.002 (55 %).
5. Both arms beat plain + pointer s2 by at least 0.025 (80 %).
6. Head-off above head-on in every run (95 %); every run finite (90 %).

## Method

Configs `spiral_pub_poisson{2,4}.yaml`. Readouts as the chunk-loop filing (paired CE, K1-K6 with bootstrap,
head-off). Smoke (20 steps) passed for both, peak 18.6 / 16.8 GiB (`morph-scratch/abq/spiral/gpu_todo_pubpoisson.log`).

## Results

The predictions above use the old numeric format. They are graded here under the 2026-10-16 protocol. The
percent figures are not graded. CE@6 seed spread is about 0.010 nats, so a CE@6 line closer than that is
"noise-limited".

Sources: `morph-scratch/abq/full1015/chain.log` (train lines, head-off lines),
`morph-scratch/abq/full1015/eval/paired_full1015.json` (paired CE on 510,003 tokens; K1-K6 = ce["1"] - ce["6"]),
`morph-scratch/abq/spiral/eval/bench_pubpoisson.md` and `bench_v2.md` (step ms).

Paired CE and K-curve, 2-seed means (s1 / s2 in brackets):

| run | CE@6 | K1-K6 | train step ms (bench) | tok/s wall (chain.log, s1 / s2) |
| --- | --- | --- | --- | --- |
| spiral, Poisson chunks of 2 | 3.6907 (3.6880 / 3.6934) | +0.0070 (+0.0083 / +0.0058) | 234.2 | 29,129 / 29,213 |
| spiral, Poisson chunks of 4 | 3.6930 (3.6904 / 3.6955) | +0.0008 (+0.0008 / +0.0007) | 199.0 | 34,361 / 34,350 |
| published spiral (reference) | 3.6912 (3.6905 / 3.6919) | 0 (one pass, no depth axis) | 138.0 | not measured |
| chunk-2 loop (reference) | 3.6924 (3.6903 / 3.6946) | +0.0062 (+0.0064 / +0.0062) | 191.6 (from the plan) | not measured |
| plain + pointer s2 (one seed) | 3.7256 | +0.0376 | 283.0 | not measured |
| gpt8 + pointer (reference row) | 3.6996 | none | 79.8 | not measured |
| flat14 + pointer s1 (reference row) | 3.6731 | none | 123.2 | not measured |

Poisson chunks of 2 against the published spiral, paired per seed: -0.0025 (s1), +0.0014 (s2), mean -0.0005.
Chunks of 4: -0.0001 (s1), +0.0036 (s2), mean +0.0018. Both means are inside the 0.010 spread. The within-run
bootstrap intervals on K1-K6 in chain.log are narrow (for example +0.0083 [+0.0078, +0.0087]), but the
between-seed difference of K1-K6 for chunks of 2 is 0.0025, so use the seed spread, not the interval.

Chain K-curve of chunks of 2, seed 1 (chain.log): ce at depth 1, 2, 3, 4, 6, 8 =
3.6963, 3.6911, 3.6893, 3.6885, 3.6880, 3.6880. It flattens at depth 4, like the chunk-loop arms.

Prediction check:

1. Chunks of 2, CE@6 at most 3.6912: 3.6907 is under the line by 0.0005. Held on paper, noise-limited.
2. Chunks of 4, CE@6 at most 3.6912: 3.6930 is over it by 0.0018. Did not hold on paper, noise-limited.
3. Chunks of 2, K1-K6 at least +0.004: +0.0083 and +0.0058. Held in both seeds.
4. Chunks of 4, K1-K6 at least +0.002: +0.0008 and +0.0007. Did not hold. The K-curve is nearly flat.
5. Both beat plain + pointer s2 by at least 0.025: chunks of 2 by 0.0349, chunks of 4 by 0.0326. Held. This is
   one plain seed, and the margin is 3x the spread.
6. Head-off above head-on in every run: held in all 8 runs (for example 4.0676 off vs 3.6880 on, chunks of 2 s1;
   4.0693 vs 3.6904, chunks of 4 s1; chain.log head-off lines). Every run finite: held (chain.log). These are gates.

## Verdict

Supported for chunks of 2: it gives the published spiral a K-curve of +0.0070 (2-seed mean, +0.0083 and
+0.0058) with CE@6 3.6907 against 3.6912, a paired difference of -0.0005, inside the 0.010 spread. So CE is not
lost. Chunks of 4 get almost no K-curve (+0.0008).

Two things this does not show. First, the K-curve does not buy CE: the looped arm is not better than the
published spiral at depth 6 (inside noise), and it is no better than the chunk-2 loop (3.6924, K1-K6 +0.0062)
while it costs 234.2 ms against 191.6 ms and 138.0 ms for the published spiral (1.70x). Second, only two
seeds were run.

## Updated hypothesis

The spiral gets depth dependence from a looped chunk-2 level, and the published spiral's 3.69 is not hurt by
it. But a K-curve of +0.007 is the same size as the plain chunk loops show, and it does not move CE. The
spiral's gain over plain is the chunk hierarchy, not the loop. Next question: does the K-curve at chunks of 2
turn into a CE gain at a longer run (10k or more), where depth might pay? Until then the published spiral
(138.0 ms) stays the speed and CE base for spiral work.

## What this changes

We can say a spiral level can loop at Parcae's depth without losing CE. We cannot say the loop earns CE. The
loop is a cost (1.70x step) with no measured CE return at 5k, so the next spiral experiment should not spend
more on the loop without a longer run.

## Artifacts

- `/home/wolfe/morph-scratch/abq/full1015/chain.log`
- `/home/wolfe/morph-scratch/abq/full1015/eval/paired_full1015.json`
- `/home/wolfe/morph-scratch/abq/full1015/eval/spiral-pub-poisson{2,4}-5k-s{1,2}-head-off.json`
- `/home/wolfe/morph-scratch/abq/spiral/eval/bench_pubpoisson.md`, `bench_v2.md`
- runs in `/home/wolfe/parcae-runs/spiral-pub-poisson{2,4}-5k-s{1,2}`
