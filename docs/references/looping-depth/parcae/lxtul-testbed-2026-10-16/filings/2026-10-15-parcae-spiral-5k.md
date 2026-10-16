# Parcae: multi-resolution recursion (SpiralFormer) with spans as the coarse level, at 5k

Status: failure (5 of 12 predictions hold; written 2026-10-15 before any code or run of these arms)

## Question

SpiralFormer (arXiv 2602.11698, Yu et al.) runs one shared core at a different sequence resolution on
each pass: pool the token states into chunks, run the core on the short sequence, add the result back
into every token's state (causally shifted), coarse to fine. Plain CE, full backprop. At 410M it beats
a full-resolution looped model at 10 % fewer FLOPs and Pythia at equal parameters (PPL 9.07 to 8.73).

Two facts make it the right shape for TUL on Parcae:

1. The loop output is not a side position. It is part of the token state the coda reads, so the coda
   cannot route around it (the slot loop's shortcut; MORPH note 2026-09-07 Spiral-TUL).
2. A pass at resolution r costs r of a full pass. Parcae's plain loop runs its 6-layer core about 6
   times at full resolution (44 layer-equivalents per token at mean depth 6). A coarse loop costs a
   fraction of that.

Wolfe (2026-10-15) approved two variants: coarse to fine over passes (variant 1), and a loop over one
coarse level that then decodes to tokens (variant 2). Only the main CE buys depth: both are trained by
plain next-token CE and nothing else.

Questions:
- Does SpiralFormer as published reproduce on this testbed at 5k (against gpt8 + pointer and flat14 +
  pointer, at near-equal FLOPs)?
- Does a Poisson-depth loop over SPAN states (the TUL boundary rule), written back into the next
  span's tokens, earn loop contribution (K1-K6) where the slot loop was flat (minimal 0.0000)?
- Does adding finer single-pass levels after the span loop help?

## Arms (branch lxtul/spiral; 5000 steps; batch 6; plain token rows; 2 seeds each; pointer head 4)

Shape: prelude 4, core 6 (shared), coda 4, the testbed's Parcae blocks. Topology Anchor: every level
reads `h = h0 + u` where h0 is the prelude output and u the latest broadcast update (Parcae's own
injection runs inside each core loop). Down: learned softmax pool per level (Spiral's self-aggregation).
Up: uniform broadcast with gain sqrt(g) (Spiral's ablation: 9.03 vs 9.00, the cheaper operator).

| arm | levels (coarse to fine) | core passes per level | layer-equiv / token (est.) |
| --- | --- | --- | --- |
| published spiral | chunks of 16, 8, 4, 2 (half-chunk offset, shift g-1) | 1 each | 8 + 6 x 15/16 = 13.6 |
| span-level loop | spans by the TUL boundary rule | Poisson, Parcae's mean and max | 8 + 36 x (spans/L) ~ 10.3 |
| span-to-pair spiral | spans, then chunks of 4, then chunks of 2 | Poisson at spans, 1 at the chunks | ~ 10.3 + 4.5 = 14.8 |

Span causality: span j's pooled state reaches only the tokens AFTER span j's last token (Spiral's
no-overlap regime, which the slot geometry already has). Fixed chunks use Spiral's shift g - 1.

References on the same 510,003 scored tokens (plain rows): gpt8 + pointer 3.6996 (2 seeds, 8 layers),
flat14 + pointer 3.6731 (1 seed, 14 layers), plain + pointer 3.7206 (2 seeds, row compact, 44 layer-equiv).

## Readouts

- CE per eval depth 1, 2, 3, 4, 6, 8 of the span loop (the span arms; the chunk passes stay at 1);
  K1-K6 with a bootstrap interval. The published spiral has a fixed depth: CE only, no K-curve.
- Paired CE against every reference above. Head-off CE (pointer gate off).
- Bench median ms/step (synced, 200 after 100 warm-up), TFLOP/step from `flop_tally`, peak GiB.
- Mean spans per row (sets the span loop's real cost).

## Predictions

1. Published spiral, 2-seed mean CE beats gpt8 + pointer's 3.6996 (70 %).
2. Published spiral, 2-seed mean within +-0.015 of flat14 + pointer's 3.6731 (50 %).
3. Span-level loop, 2-seed mean beats gpt8 + pointer by at least 0.005 (45 %).
4. Span-level loop, 2-seed mean K1-K6 at least +0.005 (40 %).
5. Span-to-pair spiral, 2-seed mean beats the published spiral's (40 %).
6. Span-to-pair spiral, 2-seed mean K1-K6 at least +0.005 (35 %).
7. Every arm's 2-seed mean beats plain + pointer's 3.7206 (65 %).
8. Bench: span-level loop at most 1.35 x gpt8 + pointer's 79.8 ms; both spirals at most 1.25 x flat14 +
   pointer's 123.3 ms (60 %).
9. Head-off CE above head-on in every run (95 %).
10. Every run finishes with finite losses (90 %).

## What each outcome changes

- 1 and 2 hold: the paper reproduces here; the span arms are read against it.
- 3 and 4 hold: a span loop written back into the tokens earns depth on the main CE alone at about
  gpt8's cost. That is the LXTUL claim on Parcae (gains, little CE cost, loop contribution) and the
  next step is a second scale and the MeSH topology.
- 4 fails with 3 holding: the coarse pass helps as extra width, not as depth; test the repeats at
  the chunk levels instead of the span level.
- 1 fails: the reproduction is off at this scale or step count; read the chunk arm before the span arms.

## Method

Code by a sonnet builder under `morph-scratch/abq/parcae-builder-rules-2026-10-14.md` (own worktree,
no git writes, CPU tests only, writes gpu_todo.sh and stops). GPU chain `morph-scratch/abq/spiral/gpu_todo.sh`,
after the NCM chain. Amendments get a date and a reason here.

Amendment 2026-10-15 (before any code ran on GPU and before any run of these arms; Wolfe asked for it):
a fourth arm, "span-to-pair MeSH", 2 seeds. It is the span-to-pair spiral with Anchor replaced by MeSH
(arXiv 2510.07739, ICLR 2026; SpiralFormer appendix B): a buffer of B full-shape slots, slot 0 = the
token embeddings, a transitional write-read of the prelude output gives the first state, and each LEVEL
(span loop, chunk 4, chunk 2) writes its shifted update into the slots and reads the next state with
its own softmax routers (one Linear(d, B) each for write and read, per level). The coda reads a final
read. B = levels + 3 = 6 (MeSH's rule B = N_loop + 3, with the levels as its iterations). The Parcae
core loop inside the span level is unchanged. Why: under Anchor the span loop's update reaches the
coda only through the two chunk passes that pool and replace it; MeSH lets it persist in a slot.
Note on the published spiral: it runs Anchor, so its paper row is SpiralFormer-B* (Anchor), not the
MeSH rows; the reproduction claim is read that way.

Added predictions (the ten above are unchanged):
11. Span-to-pair MeSH, 2-seed mean CE beats span-to-pair spiral's (55 %).
12. Span-to-pair MeSH, 2-seed mean K1-K6 at least +0.005 (40 %).

Amendment 2026-10-15, round 1 void (a bug), round 2 run. The first seed-1 runs (`spiral-*-5k-s1`, commit
b963019) added C(zh) to h0 with gain sqrt(g) and no norm. The paper's eq. 20 is lambda / g with lambda =
sqrt(g), so the update was g times too large, and Parcae's loop state is not normed (plain Parcae's coda
reads C(x) instead of h0, so it never meets this). Measured on the published spiral's checkpoint (CPU,
2 rows): update RMS 536, 1.9e5, 4.6e7, 6.9e9 over the four levels against RMS(h0) 1.04; Anchor compounds
it because each level pools h0 + u. Round-1 CE@6: published 6.9509, span-to-pair 6.9301 (both broken);
span-level loop 3.7260 with K1-K6 +0.0000 and its update ignored (RMS 0.11; zeroing it moves CE by 0.0002).
The MeSH seed-1 run was stopped at about step 1500 and the chain before the seed-2 runs. Fix (commit
after this text): u = (1 / sqrt(g)) * RMSNorm_level(C(zh)), a learned per-level RMSNorm weight init 1;
spans use 1 / sqrt(span length). A CPU test pins the update RMS of a receiving token at exactly
1 / sqrt(g) at init, and fails with the norm removed. Round 2 reruns all four arms as `spiral-*-v2-5k-s*`
(chain `gpu_todo_v2.sh`, smokes skipped: the shapes are unchanged); a seed-1 run above 4.5 val CE skips
its seed 2. Predictions are unchanged and are read on round 2 only. Paired eval rows: dec2-5k (4-cell
rows, 510,003 tokens); references with checkpoints: gpt8 + pointer s1 and s2, flat14 + pointer s1,
plain + pointer s2 (`plain-pointer-norm-5k-s2`); plain + pointer s1 has no checkpoint, so prediction 7
reads plain + pointer s2 alone.

## Results

Round 2 only (`spiral-*-v2-5k-s{1,2}`, commit 23f0c27). Round 1 is void (the amendment above).

Note on the paired eval (2026-10-15, after the runs, before any result was read from it). The chain's
step 4 failed: this tree cannot build `dec2-5k` (its config has the decode branch's `decode_loop_layers`).
The rerun takes its rows from `minfan-5k` (4 cells, the same row layout, no decode keys). The token
count reads 510,003 and the references reproduce (gpt8 + pointer 3.6996 mean, flat14 + pointer 3.6731),
so it is the same token set. The rerun also drops the void round-1 `spiral-span-5k-s1` (it has no
`up_norm` weights and does not load). Script `results/2026-10-15-parcae-spiral-5k/paired_v2_rerun.sh`.

### CE on the paired set (510,003 tokens, plain token rows, pointer head on)

| arm | seed 1 CE@6 | seed 2 CE@6 | 2-seed mean CE@6 | mean CE@1 | K1-K6 (train-time eval, s1 / s2) |
| --- | --- | --- | --- | --- | --- |
| gpt8 + pointer | 3.7034 | 3.6958 | 3.6996 | 3.6996 | flat |
| flat14 + pointer | 3.6731 | | 3.6731 | 3.6731 | flat |
| plain + pointer s2 | | 3.7256 | 3.7256 | 3.7633 | +0.0377 (paired CE@1 - CE@6) |
| published spiral | 3.6905 | 3.6919 | 3.6912 | 3.6912 | fixed depth |
| span-level loop | 3.7054 | 3.7087 | 3.7070 | 3.7071 | +0.0000 / +0.0001 |
| span-to-pair spiral | 3.6897 | 3.6914 | 3.6905 | 3.6905 | +0.0000 / -0.0000 |
| span-to-pair MeSH | 3.6967 | 3.6926 | 3.6946 | 3.6948 | +0.0002 / +0.0001 |

Paired difference to gpt8 + pointer s1 (bootstrap 95 %): gpt8 s2 -0.0076 [-0.0094, -0.0058] (the seed
spread), flat14 -0.0303, published -0.0129 / -0.0115, span +0.0020 / +0.0053, span-to-pair -0.0137 /
-0.0120, MeSH -0.0067 / -0.0108.

Head-off (pointer gate off) CE@6 sits 0.378 to 0.381 above head-on in all 8 runs. All 8 runs finish
finite with falling loss. Mean spans per row 59.6 at eval (51.4 in the bench batch).

### Bench (batch 6, median of 200 synced steps after 100 warm-up; `bench.md`)

| arm | params M | train ms/step | TFLOP fwd+bwd | layer-equiv / token | infer ms (depth 6) |
| --- | --- | --- | --- | --- | --- |
| gpt8 + pointer | 404 | 79.8 | 5.96 | 8 | 18.4 |
| flat14 + pointer | 630 | 123.2 | 9.01 | 14 | 26.5 |
| plain + pointer | 582 | 283.0 | 28.50 | 44 | 69.5 |
| published spiral | 582 | 138.0 | 8.73 | 13.62 | 31.5 |
| span-level loop | 582 | 130.6 | 7.74 | 9.69 (11.15 padded) | 27.1 |
| span-to-pair spiral | 582 | 153.3 | 9.97 | 14.19 (15.65 padded) | 35.1 |
| span-to-pair MeSH | 582 | 162.3 | 9.97 | 14.19 (15.65 padded) | 38.2 |

The spiral arms carry Parcae's full parameter set (4 + 6 + 4 unique layers), so their optimizer step
costs 42.7 ms against gpt8's 24.4 ms. That is most of the span-level loop's gap to gpt8.

### Level ablation (CPU fp32, 6 rows, seed-1 checkpoints; `zero_level.txt`)

CE rise when one level's update u is set to zero (absolute CEs about 4.14 on this small set; only the
differences are read):

| arm | span level | chunks 16 | chunks 8 | chunks 4 | chunks 2 | all levels |
| --- | --- | --- | --- | --- | --- | --- |
| span-level loop | +0.0040 | | | | | +0.0040 |
| published spiral | | +0.0000 | +0.0045 | +0.0580 | +0.5819 | +0.5819 |
| span-to-pair spiral | +0.0005 | | | +0.0106 | +0.6220 | +0.6220 |
| span-to-pair MeSH | +0.0005 | | | +0.0078 | +0.1230 | +0.1895 |

Under Anchor, zeroing the last level leaves the coda with h0 alone, so the last column overstates that
level's own share. The ranking does not depend on it: the span level and the coarse chunks carry
almost nothing, and the work is at chunks of 4 and 2.

### Predictions

| # | prediction | result | verdict |
| --- | --- | --- | --- |
| 1 | published beats gpt8 + pointer 3.6996 | 3.6912 (-0.0084) | holds |
| 2 | published within +-0.015 of flat14 3.6731 | +0.0181 | fails |
| 3 | span loop beats gpt8 by >= 0.005 | 3.7070 (+0.0074, worse) | fails |
| 4 | span loop K1-K6 >= +0.005 | +0.0000 / +0.0001 | fails |
| 5 | span-to-pair beats published | 3.6905 vs 3.6912 (-0.0007) | holds by the letter, inside seed spread |
| 6 | span-to-pair K1-K6 >= +0.005 | +0.0000 / -0.0000 | fails |
| 7 | every arm beats plain + pointer s2 3.7256 | worst arm 3.7070 | holds |
| 8 | span <= 107.7 ms; spirals <= 154.1 ms | span 130.6 ms; spirals 138.0, 153.3 | fails (span part) |
| 9 | head-off above head-on | +0.378 to +0.381 in 8 of 8 | holds |
| 10 | finite | 8 of 8 | holds |
| 11 | MeSH beats span-to-pair | 3.6946 vs 3.6905 | fails |
| 12 | MeSH K1-K6 >= +0.005 | +0.0002 / +0.0001 | fails |

## Verdict

Failure: 5 of 12 predictions hold (1, 5, 7, 9, 10), and every loop-contribution prediction fails.

- The published spiral reproduces the paper's claim against a looped model: it is 0.034 better than
  plain + pointer s2 at 0.31x the training FLOPs and 0.49x the step time. It also beats gpt8 + pointer
  by 0.008. It does not reach flat14 + pointer: flat14 is 0.018 better, 11 % faster per step and has
  about the same FLOPs (9.01 vs 8.73 TFLOP). On this testbed at 5k, flat14 dominates every spiral arm.
- The span level does no work. The span-level loop is depth-flat (K1-K6 at most +0.0001) and is the
  worst spiral arm (+0.016 vs published). Inside the two span-to-pair arms, zeroing the span level
  moves CE by 0.0005. Writing the span state into the next span's tokens did not stop the bypass: the
  coda gets what it needs from the tokens.
- What the spirals gain comes from extra compute at fine resolution (chunks of 4 and 2), not from a
  coarse thought. The coarse levels (spans, chunks of 16 and 8) carry at most 0.005.
- MeSH costs 6 % more step time than Anchor and does not help (+0.004).

## Updated hypothesis

At 5k on Parcae, depth gain is per token and private (the boundary-depth probe, 2026-10-14), and this
filing extends that to written-back coarse states: a span or a 16-token chunk, pooled and broadcast to
later tokens, adds nothing the coda cannot get from those tokens. The cheapest loop compute that pays
is at near-token resolution. Two readings stay open and this run cannot tell them apart:

1. Coarse states have no value for next-token CE at this scale, so a span-level loop cannot earn depth
   from the main CE alone (this would make "difficulty lives at the span level" false for CE).
2. Coarse states have value, but at 5k steps the model has not learned to use them while the token
   path is open (the pre-knee reading).

The next test that separates them changes the information, not the schedule: a coarse level that
carries what the tokens do not (for example the span's future, as in the NCM plan arm) or a geometry
where later tokens cannot see earlier ones except through the coarse state. Both exist in other
forms: the minimal strict slot loop is flat (K1-K6 0.0000, 2026-10-14), and the NCM plan arm's K1-K6
(+0.0076 at m = 3) comes from its code NLL, while its token NLL does not depend on depth.
