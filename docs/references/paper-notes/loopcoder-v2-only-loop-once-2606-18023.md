# Paper note: LoopCoder-v2 (Parallel Loop Transformer loop count)

- **Title:** LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling (Jian Yang, Shawn Guo, Wei Zhang and 16 others)
- **arXiv or URL:** 2606.18023 (v1, 2026-06-16)
- **Local paper text:** docs/references/looping-depth/loopcoder-v2-only-loop-once-2606-18023/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: A 7B model trained from scratch on 18T tokens at loop counts 1 to 4, with per-loop instruments (update cosine, effective rank, fixed-point gap, attention KL). The architecture is the Parallel Loop Transformer (PLT), which differs from Parcae. Only the instruments and the non-monotonic loop count transfer.

## 2. What the paper shows

Read: Sec 1 to 4.3 and the start of Sec 5; appendices and the per-loop figure data beyond the values quoted here were not read.

- 2026-10-20 (Sec 2.2, 3.1): PLT loop r takes Embed(x) + shift(h(r-1)), so token i sees the previous loop's state of token i-1, which allows parallel loop execution; loops >= 2 use sliding-window attention (w=64) gated with the frozen loop-1 KV. Each model is trained AND evaluated at its own R (R=1 baseline, 2, 3, 4), then instruction tuned (6M examples). Loop counts are not FLOP matched: more loops, more compute.
- 2026-10-20 (Table 2): R=1 / 2 / 3 / 4. SWE-bench Verified 43.0 / 64.4 / 27.6 / 22.4. Terminal-Bench v1 26.3 / 34.2 / 30.0 / 26.3. BFCL 32.2 / 40.1 / 36.3 / 39.5. Average over the table 38.0 / 46.5 / 36.9 / 34.3. One run per R.
- 2026-10-20 (Fig 2, 3, Table 3; PLT4 model): effective rank peaks at loop 2 and falls for deeper loops (174.6, 172.5, 158.2 at r=2,3,4); update alignment cos theta is -0.72, -0.46, +0.04, so updates reverse (oscillate) in the refinement loops; output shift is largest at loop 2. The authors report the fixed offset cost is 30 to 45 times the per-loop output shift beyond loop 2.
- 2026-10-20 (Table 4): at R=2, thinking (explicit CoT) vs instruct: LiveCodeBench 62.3 vs 35.4, CRUX 93.5 vs 86.9. They call latent loop and explicit CoT complementary.
- Failure mode: R >= 3 regresses below the non-looped baseline on many tasks. The authors tie it to the PLT shift tax, not to loops in general.

## 3. Our speculation

- 2026-10-20 Speculation: the oscillation (cos < 0 between successive updates) is a cheap read to add to our K-curve panel: the cosine between pass-r and pass-(r+1) residual updates on the 480 rows, plain vs fan vs span-CALM. A negative cosine at passes 3 to 6 would say the extra passes of our core are not converging. Per the "never pass or fail on a cosine" rule, use it as a diagnostic only.
- 2026-10-20 Speculation: the confound is large (PLT shift, no FLOP match, one run per R). Treat "two loops are best" as one point about PLT, not as a law for Poisson-depth cores.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: They find loops past 2 hurt. We train with a Poisson mean-6 draw and see K1-K6 grow with depth on plain (+0.0377), so a deep draw is not harmful in our setting. The two setups differ in how the model is trained (fixed R vs a draw), so no direct conflict.
