# Parcae: Block-Parcae, a series hierarchy (the loop is the coda's only route between chunks), at 5k

Status: failure (filed 2026-10-15; 5 of 11 predictions hold)

## Question

The chunk-loop filing (`failures/2026-10-15-parcae-chunk-loop-5k.md`) showed that a skip around the loop
removes plain Parcae's K-curve (+0.0377 to +0.0008) and that CE does not suffer. Depth showed up only where
the loop computed something the skip could not carry (chunk summaries). Wolfe (2026-10-15): "maybe the full
model should be hierarchical all the way through then we have a separate decoder of some kind? So the loop
and the coda speak the same language." He accepted the defaults: the decoder sees only its own chunk's
tokens plus the coarse states; fixed chunks of 2 and 4; a separate decoder (the coda), not the shared core.

Questions:
- In a series hierarchy, does the chunk loop return CE value over the same model trained at depth 1?
- What does the series topology cost in CE against the parallel (Anchor) chunk loop at the same chunk size?

## Arms (branch lxtul/spiral; 5000 steps; batch 6; plain token rows; 2 seeds; pointer head 4)

Prelude 4 (full causal attention, feeds the pool only), core 6 (shared), coda 4. One chunk level, offset 0,
shift = size: the tokens of chunk c receive `RMSNorm(C(x_{c-1})) / sqrt(g)`, the loop state of chunk c - 1.
The coda input is the token EMBEDDING + that update (never the prelude state), and the coda attends only
inside its own chunk (each chunk runs as its own sequence of g tokens). Code: `topology: block` in
`lxtul/spiral.py`.

| arm | chunk | depth | layer-equiv / token (linear, est.) |
| --- | --- | --- | --- |
| block-4 loop | 4 | Poisson (mean 6, max 8) | 8 + 36 / 4 = 17 |
| block-4 depth-1 control | 4 | fixed, 1 pass | 8 + 6 / 4 = 9.5 |
| block-2 loop | 2 | Poisson | 8 + 36 / 2 = 26 |

The pointer head stays on, as in every reference. It reads the coda's final states of every earlier
token, so it is an output-side route between chunks for copying. It is the same head as in the references.

References on the same 510,003 tokens: flat14 + pointer 3.6731; published spiral 3.6912; chunk-2 loop
(Anchor) 3.6924, K1-K6 +0.0063; gpt8 + pointer 3.6996; chunk-4 loop (Anchor) 3.7006, K1-K6 +0.0030;
plain + pointer s2 3.7256, K1-K6 +0.0377.

## Readouts

CE at eval depths 1, 2, 3, 4, 6, 8 (the depth-1 control is scored once); K1-K6 with a bootstrap interval;
paired CE against every reference; head-off CE; bench ms/step, TFLOP, peak GiB.

Loop value (the measure proposed in the chunk-loop filing): block-4 depth-1 control CE minus block-4 loop
CE@6, 2-seed means. Positive = the loop's extra passes buy CE at equal tokens.

## Predictions

1. Block-4 loop K1-K6 at least +0.015 (2-seed mean) (70 %). Series makes the coda depend on the loop.
2. Block-4 loop K1-K6 above the chunk-4 Anchor loop's +0.0030 (90 %).
3. Loop value: block-4 loop CE@6 beats the block-4 depth-1 control by at least 0.010 (35 %).
4. Block-4 loop CE@6 is worse than the chunk-4 Anchor loop (3.7006) by at least 0.020 (65 %): the series
   cut costs CE at 5k.
5. Block-4 loop CE@6 beats plain + pointer s2 3.7256 (35 %).
6. Block-2 loop CE@6 beats block-4 loop by at least 0.015 (65 %).
7. Block-2 loop K1-K6 at least +0.015 (60 %).
8. Bench: block-4 loop at most 150 ms/step (chunk-4 Anchor loop 153.4) (60 %).
9. Bench: block-4 depth-1 control at most 115 ms/step (60 %).
10. Head-off CE above head-on in every run (95 %).
11. Every run finishes with finite losses (90 %).

## What each outcome changes

- 3 holds: the first loop on this testbed with CE value over its own depth-1 model. Block-Parcae becomes
  the design to scale and to tune (chunk size, coda width, a local encoder).
- 1 holds and 3 fails: series buys dependence but no value, as plain Parcae did. The hierarchy question
  then moves to what the chunk state is trained to hold, not where it is read.
- 4 fails (series costs little CE): the coda did not need the full-resolution prelude path, and the
  series design is cheap to adopt.

## Method

Code by me (Opus), not a builder: about 30 lines (`SpiralConfig.topology = "block"`, `_block_coda`), with
five tests in `lxtul/tests/test_spiral.py`: with the update zeroed no logit outside the perturbed token's
chunk moves (all three arms); the live loop carries earlier chunks to later ones; the same detector catches
the Anchor skip and a full-row coda; config guards. `test_spiral.py` 127 passed (CPU). Chain
`morph-scratch/abq/spiral/gpu_todo_block.sh`: smokes, seed 1 of the three arms, then seed 2, head-off,
paired eval, bench. It waits before any GPU step while `morph-scratch/abq/PAUSE` exists (Wolfe's game
pause, 2026-10-15).

## Results

All six runs finish 5000 steps with finite, falling loss (commit 78f6ac4). Paired eval on 510,003 tokens
(`results/2026-10-15-parcae-block-parcae-5k/paired_block.json`).

| arm | CE@6 s1 / s2 | mean CE@6 | mean CE@1 | K1-K6 (mean) | gap to gpt8 + pointer s1 | train ms/step |
| --- | --- | --- | --- | --- | --- | --- |
| block-2 loop | 3.9086 / 3.9110 | 3.9098 | 3.9206 | +0.0108 | +0.205 / +0.208 | 219.2 |
| block-4 loop | 3.9516 / 3.9541 | 3.9529 | 3.9604 | +0.0076 | +0.248 / +0.251 | 171.8 |
| block-4 depth-1 control | 3.9553 / 3.9548 | 3.9551 | 3.9551 | 0 | +0.252 / +0.252 | 126.8 |
| (ref) chunk-4 loop, Anchor | | 3.7006 | 3.7036 | +0.0030 | | 153.4 |
| (ref) plain + pointer s2 | | 3.7256 | 3.7633 | +0.0377 | +0.022 | 283.0 |

Head-off CE@6 sits 0.405 to 0.410 above head-on in all 6 runs. Bench (`bench_block.md`): the local coda
does not make the block arms cheaper than the chunk-4 Anchor loop. Forward 44.5 vs 40.8 ms, backward
75.7 vs 65.9 ms: the coda runs 1536 sequences of 4 tokens per layer, a poor shape for the attention kernel.

| # | prediction | result | verdict |
| --- | --- | --- | --- |
| 1 | block-4 K1-K6 >= +0.015 | +0.0076 | fails |
| 2 | block-4 K1-K6 above chunk-4 Anchor +0.0030 | +0.0076 | holds |
| 3 | block-4 loop beats its depth-1 control by >= 0.010 | 0.0022 | fails |
| 4 | block-4 worse than chunk-4 Anchor by >= 0.020 | +0.252 | holds |
| 5 | block-4 beats plain + pointer s2 3.7256 | 3.9529 | fails |
| 6 | block-2 beats block-4 by >= 0.015 | 0.043 | holds |
| 7 | block-2 K1-K6 >= +0.015 | +0.0108 | fails |
| 8 | block-4 bench <= 150 ms | 171.8 | fails |
| 9 | depth-1 control bench <= 115 ms | 126.8 | fails |
| 10 | head-off above head-on | 6 of 6 | holds |
| 11 | finite | 6 of 6 | holds |

## Verdict

Failure: 5 of 11 hold (2, 4, 6, 10, 11). Block-Parcae as built is not viable: it is 0.23 to 0.25 behind
plain Parcae, the loop it was built to compare against. Cutting the coda to one loop vector per chunk
costs 0.25 nats against the parallel chunk loop at the same chunk size. The series cut does raise the
coda's dependence on depth (K1-K6 +0.0030 to +0.0076 at chunks of 4), and halving the chunk recovers
0.043, which reads as a channel-width limit.

Note added at filing (Wolfe, 2026-10-15, after the runs): loop contribution is judged against other loops
(plain Parcae), not against flat models or a depth-1-trained twin at 5k, since loops need far more data
than a 5k run gives. Prediction 3 used the depth-1 twin as a bar; it stays graded as written, but it is
not a design verdict.

## Updated hypothesis

The channel between loop and coda, not the topology alone, sets the cost of a series design. The next
arm keeps the series property and widens the channel: the coda attends over the loop states of every
earlier chunk as a memory (`spiral.block_memory`, branch lxtul/block-mem, built 2026-10-15).
