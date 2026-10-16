# Parcae: a Poisson-depth loop at chunk resolution, and the skip around the loop, at 5k

Status: failure (filed 2026-10-15; 5 of 10 predictions hold)

## Question

The Spiral filing (`failures/2026-10-15-parcae-spiral-5k.md`) found that the spiral's work happens at
chunks of 4 and 2, and that the span level does nothing. Every spiral chunk level ran one pass, so it has
no K-curve. Wolfe (2026-10-15): "let's try the Poisson sample chunks of 4". He also asked whether the
loop-contribution failure comes from the relationship between the loop and the coda.

Two facts frame the batch:

1. Plain Parcae's loop earns depth (K1-K6 +0.0377, paired, seed 2). It runs at token resolution, and its
   coda reads the loop output ALONE (series).
2. Every spiral arm uses Anchor: the coda reads `h0 + u`, so the prelude's state reaches the coda around
   the loop (parallel). Every spiral arm is flat.

So two things changed at once between "earns" and "flat": the resolution (1 to 4 or more tokens per
group) and the topology (series to parallel). This batch separates them.

Questions:
- Does a Poisson-depth loop at chunks of 4 (and of 2) earn loop contribution on the main CE alone?
- Does Anchor's skip, on its own, take plain Parcae's depth away (resolution held at 1 token)?

## Arms (branch lxtul/spiral; 5000 steps; batch 6; plain token rows; 2 seeds each; pointer head 4)

Same shape as the Spiral filing: prelude 4, core 6 (shared), coda 4, Anchor (`h = h0 + u`), learned
softmax pool, update `u = RMSNorm_level(C(x)) / sqrt(g)`. The chunk loop uses the span level's depth law:
Parcae's Poisson total (mean 6), truncated at max_depth 8, the last `mean_backprop_depth` passes with
gradient. One level per arm.

| arm | level | groups per row | layer-equiv / token (est.) |
| --- | --- | --- | --- |
| chunk-4 loop | chunks of 4 (offset 2, shift 3), Poisson depth | 256 | 8 + 36 / 4 = 17 |
| chunk-2 loop | chunks of 2 (offset 1, shift 1), Poisson depth | 512 | 8 + 36 / 2 = 26 |
| token-level skip loop | chunks of 1 (offset 0, shift 0), Poisson depth | 1024 | 8 + 36 = 44 |

The token-level skip loop is plain Parcae's loop (one group per token; a pool over one token is the
identity) with one change: the coda reads `h0 + u` instead of the loop output. It is the topology test.

References on the same 510,003 scored tokens: gpt8 + pointer 3.6996 (2 seeds), flat14 + pointer 3.6731
(1 seed), plain + pointer s2 3.7256 (K1-K6 +0.0377, paired CE@1 - CE@6), published spiral 3.6912 (2 seeds).

## Readouts

- CE per eval depth 1, 2, 3, 4, 6, 8 of the looped level; K1-K6 with a bootstrap interval.
- Paired CE (rows from `minfan-5k`, 510,003 tokens) against every reference above. Head-off CE.
- Bench median ms/step (synced, 200 after 100 warm-up), TFLOP/step, peak GiB.

## Predictions

1. Chunk-4 loop, 2-seed mean K1-K6 at least +0.005 (30 %).
2. Chunk-2 loop, 2-seed mean K1-K6 at least +0.005 (40 %).
3. Token-level skip loop, 2-seed mean K1-K6 at most +0.010, against plain's +0.0377 (55 %).
4. Token-level skip loop, 2-seed mean CE@6 within +-0.010 of plain + pointer s2's 3.7256 (50 %).
5. Chunk-4 loop, 2-seed mean CE@6 beats the published spiral's 3.6912 (40 %).
6. Chunk-4 loop, 2-seed mean CE@6 beats flat14 + pointer's 3.6731 (15 %).
7. K1-K6 falls with group size: token-level skip >= chunk-2 >= chunk-4 (2-seed means) (60 %).
8. Bench: chunk-4 loop at most 1.25 x flat14 + pointer's 123.2 ms (55 %).
9. Head-off CE above head-on in every run (95 %).
10. Every run finishes with finite losses (90 %).

## What each outcome changes

- 3 holds: the skip around the loop takes the depth away even at token resolution. The loop-coda
  relationship is a cause, and any new design puts the loop in series (no prelude path to the coda that
  goes around the loop). This is Wolfe's "Open Geo kills loop contribution" measured on one key.
- 3 fails (the skip loop keeps plain's depth): the skip is not the cause; resolution is. Depth is
  private per token (the boundary-depth probe), and a hierarchical design must still give each token
  its own deep state.
- 1 or 2 holds: depth survives at chunk resolution, so a loop can run at 1/g of the plain loop's cost
  and still earn. That is the training-cost lever.
- 1 and 2 fail with 3 failing: depth is per token and the chunk is too coarse; the hierarchy question
  moves to a series design at g = 1 or 2.

## Method

Code by a sonnet builder under `morph-scratch/abq/parcae-builder-rules-2026-10-14.md` (own worktree,
no git writes, CPU tests only, writes gpu_todo.sh and stops). New key: a chunk level takes
`depth: poisson` (default `fixed`, today's path byte-identical); at most one sampled-depth level per
model; chunk size 1 is allowed (offset 0, shift 0). Amendments get a date and a reason here.

Note 2026-10-15 11:10 (during the run, BEFORE the token-level skip loop's first result; the ten
predictions above are unchanged and are the only ones graded). An opus theory agent built a Lean model
(MORPH `lab/theory/parallel_loop/`, report `morph-scratch/abq/drawing-board/parallel-loop-lean.md`). Its
claim: an additive skip equals entering the loop at the prelude state, so plain Parcae's K1-K6 is mostly
the loop building the coda's input from a near-zero entry (depth dependence), which the skip removes.
Its predictions for the token-level skip loop, recorded here so they are blind: 2-seed mean K1-K6 +0.004
(80 % interval 0.000 to +0.010); CE@6 3.685 (80 % 3.672 to 3.700); CE@1 at most 3.700 (75 %), against
plain's 3.7633. A K1-K6 of +0.015 or more refutes its build reading. Its chunk-loop prediction (K1-K6 at
most +0.004 each) is NOT blind: chunk-4 s1 (+0.0030) and chunk-2 s1 (+0.0064) landed while it ran.

## Results

All six runs finish 5000 steps with finite, falling loss (commit 132d3c7, chain
`results/2026-10-15-parcae-chunk-loop-5k/gpu_todo_chunk.sh`). Paired eval on 510,003 tokens (rows from
`minfan-5k`; the token count and every reference reproduce the Spiral filing).

### CE on the paired set (pointer head on)

| arm | CE@6 s1 / s2 | 2-seed mean CE@6 | mean CE@1 | K1-K6 s1 / s2 (mean) | train ms/step |
| --- | --- | --- | --- | --- | --- |
| flat14 + pointer (reference) | 3.6731 | 3.6731 | 3.6731 | flat | 123.2 |
| published spiral (reference) | 3.6905 / 3.6919 | 3.6912 | 3.6912 | fixed depth | 138.0 |
| chunk-2 loop | 3.6945 / 3.6902 | 3.6924 | 3.6986 | +0.0064 / +0.0061 (+0.0063) | 191.6 |
| gpt8 + pointer (reference) | 3.7034 / 3.6958 | 3.6996 | 3.6996 | flat | 79.8 |
| chunk-4 loop | 3.7019 / 3.6994 | 3.7006 | 3.7036 | +0.0030 / +0.0029 (+0.0030) | 153.4 |
| token-level skip loop | 3.7125 / 3.7099 | 3.7112 | 3.7120 | +0.0007 / +0.0008 (+0.0008) | 267.1 |
| plain + pointer s2 (reference) | 3.7256 | 3.7256 | 3.7633 | +0.0377 | 283.0 |

Paired difference to gpt8 + pointer s1 at depth 6 (bootstrap 95 %): chunk-4 -0.0015 [-0.0033, +0.0004]
/ -0.0040; chunk-2 -0.0088 / -0.0131; skip +0.0091 / +0.0066; plain s2 +0.0223.

Head-off CE@6 sits 0.381 to 0.385 above head-on in all 6 runs. Bench (`bench_chunk.md`, same harness as
the Spiral bench, a separate session): chunk-4 153.4 ms, 10.78 TFLOP fwd+bwd, 16.4 layer-equiv per
token; chunk-2 191.6 ms, 15.83 TFLOP, 24.8; skip 267.1 ms, 26.59 TFLOP, 41.6. Peak 14.7 / 16.8 / 21.1
GiB. All three arms carry 582 M parameters (optimizer step 42.7 ms).

### Predictions

| # | prediction | result | verdict |
| --- | --- | --- | --- |
| 1 | chunk-4 K1-K6 >= +0.005 | +0.0030 | fails |
| 2 | chunk-2 K1-K6 >= +0.005 | +0.0063 | holds |
| 3 | skip K1-K6 <= +0.010 | +0.0008 | holds |
| 4 | skip CE@6 within +-0.010 of 3.7256 | 3.7112 (-0.0144, better) | fails |
| 5 | chunk-4 beats published 3.6912 | 3.7006 | fails |
| 6 | chunk-4 beats flat14 3.6731 | 3.7006 | fails |
| 7 | K1-K6: skip >= chunk-2 >= chunk-4 | 0.0008, 0.0063, 0.0030 | fails (skip is lowest) |
| 8 | chunk-4 bench <= 154.0 ms | 153.4 ms | holds by 0.6 ms, inside cross-session noise |
| 9 | head-off above head-on | 6 of 6 | holds |
| 10 | finite | 6 of 6 | holds |

### The Lean agent's blind predictions (recorded 11:10, not graded above)

| prediction | result | verdict |
| --- | --- | --- |
| skip K1-K6 +0.004, 80 % interval [0.000, +0.010] | +0.0008 | inside the interval |
| skip CE@6 3.685, 80 % interval [3.672, 3.700] | 3.7112 | outside (0.011 worse than the top) |
| skip CE@1 <= 3.700 | 3.7120 | fails |
| refuted if skip K1-K6 >= +0.015 | +0.0008 | not refuted |
| (not blind) chunk loops K1-K6 <= +0.004 each | chunk-2 +0.0063 | fails |

## Verdict

Failure: 5 of 10 hold (2, 3, 8, 9, 10).

- The skip takes the token-level loop's depth away. Same resolution as plain Parcae, same core, same
  Poisson depth: K1-K6 falls from +0.0377 to +0.0008. The coda's read of the prelude state around the
  loop is enough to make the loop flat. This is Wolfe's "Open Geo kills loop contribution" measured on
  one key at token resolution.
- The skip did NOT cost CE. The skip loop at depth 1 (3.7120) is 0.051 better than plain at depth 1
  and 0.014 better than plain at depth 6 (one plain seed; seed spread about 0.010). So most of plain's
  K-curve is the loop building the coda's input from its near-zero entry. That is depth dependence,
  and the skip delivers the same CE at depth 1. The Lean agent's direction holds; its CE number is too
  optimistic by 0.011 to 0.026.
- Coarse chunks earn more K1-K6 under the skip than tokens do (chunk-2 +0.0063, chunk-4 +0.0030, token
  +0.0008). A chunk summary is something the prelude state of one token does not carry, so the loop
  has a job the skip cannot do. The amounts are small: chunk-2 depth 1 to 6 is worth 0.006 nats.
- The flat model with the same unique layers wins. flat14 + pointer (prelude 4 + core 6 + coda 4, run
  once) is 0.019 better than the best loop arm here (chunk-2) at 0.64x its step time, and 0.038 better
  than the skip loop at 0.46x its step time. flat14 is the nearest depth-1-trained control of every
  arm in this batch (same 14 layers; it has 630 M parameters against 582 M, so it is not an exact match), so at 5k (31 M tokens) no loop arm returns value for its extra passes.
- Against plain Parcae (the publishing comparison), the chunk loops win on both axes: chunk-2 is 0.033
  better at 0.68x the step time, chunk-4 0.025 better at 0.54x. They lose on K1-K6, and the skip arm
  shows that plain's K1-K6 is not a measure of value.

## Updated hypothesis

1. K1-K6 measures how much the coda depends on depth, not how much depth adds. Topology alone moves it
   by 47x (0.0377 to 0.0008) while CE moves the other way. Any claim of "loop contribution" for the
   paper needs a measure that a skip cannot game: CE at the trained depth against a depth-1-trained
   control of the same unique layers (here flat14), at equal tokens.
2. On that measure every loop on this testbed is negative at 5k (plain -0.052, skip -0.038, chunk-2
   -0.019, chunk-4 -0.028, published spiral -0.018). Either loops need more tokens than 5k gives
   (Wolfe: the knee is far away), or the 4 + 6 + 4 split with a shared 6-layer core is a worse use of
   the same parameters than 14 distinct layers at this size. A 5k run cannot separate these.
3. Where depth does earn under a skip, it is where the loop computes something the skip cannot carry
   (a chunk summary). This supports a series hierarchy, where the coda's only route to coarse
   information is the loop's output, over adding more parallel levels.
