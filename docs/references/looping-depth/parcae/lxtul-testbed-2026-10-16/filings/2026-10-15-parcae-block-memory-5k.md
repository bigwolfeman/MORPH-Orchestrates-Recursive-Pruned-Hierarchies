# Parcae: Block-Parcae with a loop memory, at 5k

Status: failure (filed 2026-10-16)

## Question

Block-Parcae (series: the coda sees no earlier token, only the previous chunk's loop state) trails the
parallel chunk-4 loop by 0.25 nats (`failures/2026-10-15-parcae-block-parcae-5k.md`, on branch lxtul/spiral).
Is the cost the series cut itself, or the one-vector channel? Here the coda attends over the loop states of
ALL earlier chunks (`spiral.block_memory`), still with no earlier token.

## Arms (branch lxtul/block-mem; 5000 steps; batch 6; 2 seeds; pointer head 4)

| arm | chunk | train ms (bench, measured) |
| --- | --- | --- |
| block memory, chunks of 4 | 4, Poisson depth | 168.5 |
| block memory, chunks of 2 | 2, Poisson depth | 213.8 |

References (paired): block-4 loop 3.9529 (K1-K6 +0.0076), block-2 loop 3.9098, chunk-4 Anchor loop 3.7006,
plain + pointer s2 3.7256.

## Predictions

1. Block memory, chunks of 4, closes at least half the series gap: CE@6 at most 3.8268 (55 %).
2. Block memory, chunks of 4, beats plain + pointer s2 3.7256 (20 %).
3. Block memory, chunks of 4, K1-K6 at least +0.0076 (50 %).
4. Chunks of 2 beat chunks of 4 by at least 0.010 (60 %).
5. Head-off above head-on in every run (95 %); every run finite (90 %).

## Method

Configs `spiral_blockmem{4,2}_loop.yaml`; flex mask `lxtul/attention.py::block_memory_mask`. Smoke passed
(peak 15.0 / 17.3 GiB, `morph-scratch/abq/blockmem/gpu_todo.log`). Readouts as the Block-Parcae filing.

## Results

The predictions above use the old numeric format. They are graded here under the 2026-10-16 protocol. The
percent figures are not graded. CE@6 seed spread is about 0.010 nats; K1-K6 between seeds varies by a few
thousandths.

Sources: `morph-scratch/abq/full1015/chain.log`, `.../eval/paired_full1015.json` (510,003 tokens, K1-K6 =
ce["1"] - ce["6"]), `morph-scratch/abq/blockmem/eval/bench_blockmem.md`, `morph-scratch/abq/spiral/eval/bench_block.md`.

| run | CE@6 (s1 / s2) | 2-seed mean | K1-K6 mean (s1 / s2) | train step ms (bench) |
| --- | --- | --- | --- | --- |
| block memory, chunks of 4 | 3.8942 / 3.8938 | 3.8940 | +0.0072 (+0.0073 / +0.0071) | 168.5 |
| block memory, chunks of 2 | 3.8515 / 3.8587 | 3.8551 | +0.0077 (+0.0079 / +0.0075) | 213.8 |
| block-4 loop (series, no memory) | 3.9516 / 3.9541 | 3.9528 | +0.0076 (+0.0082 / +0.0070) | 171.8 |
| block-2 loop (series, no memory) | not in the paired file | 3.9098 (from the plan above) | not measured here | 219.2 |
| chunk-4 Anchor loop (parallel) | from the plan above | 3.7006 | not measured here | not measured here |
| plain + pointer s2 (one seed) | 3.7256 | 3.7256 | +0.0376 | 283.0 |

Memory against no memory, paired by chunk size: chunks of 4 improve by 0.0588 (3.9528 to 3.8940); chunks of 2
improve by 0.0547 (3.9098 to 3.8551). Both are far outside the 0.010 spread, and both seeds agree (the
chunks-of-4 seeds sit 0.0003 apart; the chunks-of-2 seeds sit 0.0072 apart).

The series gap for chunks of 4, taken from the plan's own numbers, is 3.9528 - 3.7006 = 0.2522 against the
parallel chunk-4 loop. Memory closes 0.0588 of it, 23 percent. What is left: 0.1934 against the chunk-4 loop,
0.1684 against plain + pointer s2. For chunks of 2 the series gap is 3.9098 - 3.6924 = 0.2174 (chunk-2 loop
3.6924, `paired_full1015.json`); memory closes 0.0547, 25 percent.

Prediction check:

1. Chunks of 4 close at least half the series gap (CE@6 at most 3.8268): 3.8940. Did not hold, far outside noise.
2. Chunks of 4 beat plain + pointer s2 3.7256: 3.8940 is 0.1684 worse. Did not hold.
3. Chunks of 4 K1-K6 at least +0.0076: +0.0072 (+0.0073 / +0.0071). Under the line by 0.0004, which is inside
   the between-seed variation. Noise-limited.
4. Chunks of 2 beat chunks of 4 by at least 0.010: by 0.0389. Held, outside noise.
5. Head-off above head-on in every run: held in 4 of 4 runs (for example 4.3009 off vs 3.8942 on, chunks of 4
   s1; chain.log). Every run finite: held. These are gates.

Speed (bench): block memory costs 168.5 ms at chunks of 4 against 171.8 ms for the series loop, and 213.8 ms
against 219.2 ms at chunks of 2. Wall tok/s from chain.log: 39,254 / 39,421 (chunks of 4), 31,248 / 31,249
(chunks of 2).

## Verdict

Not supported. The main prediction was that memory closes at least half the series gap at chunks of 4. It
closes 23 percent (0.0588 of 0.2522 nats; CE@6 3.8940 against the 3.8268 line).

The memory does help, outside noise: the one-vector channel is a real part of the Block-Parcae loss. But most
of the loss (77 percent) stays when the coda has the loop states of all earlier chunks and no earlier token.
So the series cut itself is the main cost. Block-Parcae stays 0.17 nats behind plain + pointer.

## Updated hypothesis

The coda needs the earlier tokens, not only a summary of earlier chunk states. Reading all earlier chunk states
adds about 0.06 nats and no more. The K-curve is the same as without memory (+0.0072 against +0.0076), so the
memory did not make the loop earn more depth either. Next question: what does the coda lose when it sees no
earlier token? One test: give the coda the previous chunk's tokens only (a window of one chunk) with the
memory on, and see how much of the remaining 0.19 nats comes back.

## What this changes

Keep the token stream in the loop. Do not build Block-Parcae out as a replacement for the token loop. The
series design could stay as a speed idea only if a token window closes most of the remaining gap.

## Artifacts

- `/home/wolfe/morph-scratch/abq/full1015/chain.log`
- `/home/wolfe/morph-scratch/abq/full1015/eval/paired_full1015.json`
- `/home/wolfe/morph-scratch/abq/full1015/eval/spiral-blockmem{4,2}-loop-5k-s{1,2}-head-off.json`
- `/home/wolfe/morph-scratch/abq/blockmem/eval/bench_blockmem.md`, `/home/wolfe/morph-scratch/abq/spiral/eval/bench_block.md`
- runs in `/home/wolfe/parcae-runs/spiral-blockmem{4,2}-loop-5k-s{1,2}`
