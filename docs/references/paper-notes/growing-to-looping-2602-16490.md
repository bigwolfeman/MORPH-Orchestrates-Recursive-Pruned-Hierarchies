# Paper note: From Growing to Looping (depth growth and looping share iterative computation)

- **Title:** From Growing to Looping: A Unified View of Iterative Computation in LLMs (Kapl, Angelis, Maile, von Oswald, Bauer; 2026)
- **arXiv or URL:** 2602.16490
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: It compares looping with MIDAS-style depth growing at 360M and 1.7B, and reports a training-compute number for growing. Depth growing is a candidate training-speed lever for us.

## 2. What the paper shows

Read 2026-11-17 via the arXiv HTML page (first 100k of 101.7k characters; table numbers as the page gave them).

- 2026-11-17 (setup, Sec 3.2, Table 1): SmolLM-v1 style, 360M (32-layer baseline, 200B tokens) and 1.7B (24-layer baseline, 400B tokens). Loop(L x k) repeats an L-layer block k times. Loop(e-L x k-d) keeps unique layers around a looped middle.
- 2026-11-17 (growing recipe, App A): block size 4, PROP-1 stage budgets, 170k growth steps, duplicated layers copy weights and optimizer state, LR schedule not reset. LIDAS duplicates the exact layer-wise middle.
- 2026-11-17 (cost): grown models match the baseline on most categories at "roughly 80 % of the pre-training compute" (Table 1 caption). No FLOP table.
- 2026-11-17 (looping gain, Table 1): controls are iso-param (same unique layers, no loop) and iso-inference (same effective depth). Looping beats iso-param most on open-book QA, math word problems, reasoning primitives. At iso-inference it usually trails on knowledge and language modelling: 1.7B Loop(12x2) NLL 2.07 vs baseline 1.96. Reasoning primitives at 1.7B: Loop(4-4x4-4) 39.88 vs baseline 34.62; Loop(4x6) 33.84 (below baseline).
- 2026-11-17 (grown): 1.7B reasoning primitives LIDAS 48.02, MIDAS 40.88, baseline 34.62. At 360M LIDAS 31.58 vs 30.04.
- 2026-11-17 (Sec 4.2): repeating a middle block at inference, with no loop training, gives up to 2x on Copy Real Words for grown models; the baseline rarely gains.
- 2026-11-17 (Sec 5): grown and looped models gain more from extra in-context examples and from small SFT sets (variable-assignment code: 64 examples lift them above chance, baseline stays near random at 128). Math-heavy cooldown (last 30k steps, 20 % math) lifts grown models most; one extra pass on one block in cooldown gave LIDAS 38.84 math word problems.
- 2026-11-17: the paper does not say when in training the benefit appears.

## 3. Our speculation

- 2026-11-17 Speculation: growing is a training-cost lever for a FLAT model that is then looped or deep, not for our span loop. It stacks with a loop only if our loop sits on layers that can be duplicated.
- 2026-11-17 Speculation: the largest gains are on reasoning primitives and in-context tasks, not NLL. This matches our CE finding (loop does not help generic next-token loss at equal FLOPs).

## 4. Corrections

- 2026-11-17: reasoning-primitive numbers for the same baseline differ between this paper (34.62 at 1.7B) and 2512.08819 (42.77, column labelled ambiguously). Do not mix them.

## 5. Where our findings disagree

- None. Agrees with item 7 only loosely: no data on timing.
