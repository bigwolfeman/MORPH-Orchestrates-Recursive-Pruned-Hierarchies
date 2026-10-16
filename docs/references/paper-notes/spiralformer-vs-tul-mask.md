# Paper note: SpiralFormer (multi-resolution looped transformer)

- **Title:** SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion (Yu et al.), v2 Apr 2026
- **arXiv or URL:** arXiv 2602.11698, https://arxiv.org/abs/2602.11698
- **Local paper text:** none archived
- **Date first noted:** 2026-09-07

## 1. Why we reference it

- 2026-09-07: Wolfe maps our masked TUL arm onto SpiralFormer ("40% of SpiralFormer"). The TG restriction (tokens reach earlier spans only through slot states) uses the same routing idea.
- 2026-09-07: Our slot loop is one coarse level of the Spiral schedule. Our paid loop is the r=1 (full resolution) level. Both run as separate forwards in our design, not in one schedule.
- 2026-09-07: The untested arm is both levels in one loop, coarse first, shared weights, no mask.

## 2. What the paper shows

Full read on 2026-10-15. Section numbers not recorded. Ablations are at 410M parameters.

- 2026-10-15 (operators): Fixed chunks (not boundaries), a half-chunk offset, a learned softmax pool going down, and an output-routed broadcast going up with gain sqrt(g). The right shift is g-1.
- 2026-10-15 (topology): MeSH (multi-slot buffer, read and write routers) beats Anchor: 9.00 vs 9.13 PPL at 410M.
- 2026-10-15 (ablations, 410M PPL): fine-to-coarse order 9.24. No-overlap shift 9.10. Mean pool 9.01. Uniform broadcast 9.03. All non-learned operators 9.03. The paper says order and topology matter; the operators barely do.
- 2026-10-15 (savings): Spiral-B uses 7 to 11 percent fewer training FLOPs than the {1,1} looped model, at +0.01 nats. Spiral-L (no full-resolution pass, {1/16..1/2}, 0.94 passes) uses 3 to 10 percent fewer FLOPs than Pythia at equal parameters. PPL 9.07 to 8.73 at 410M.
- 2026-10-15 (setup): Fixed T=4, plain CE, full backprop. The paper has no depth sweep, since each pass has its own resolution.
- 2026-10-15 (other): The recurrence ratio optimum is 30 to 40 percent of layers shared. At decode, the coarse core fires once per chunk (cached).
- 2026-10-15 (our reproduction on Parcae, not the paper's result; filing lxtul/spiral e60a071, failure 5/12): The published spiral reaches CE@6 3.6912. It beats gpt8+ptr at 3.6996 and trails flat14+ptr at 3.6731, at the same FLOPs. Span-level loop depth is flat (K1-K6 +0.0001). Span level is worth 0.0005 inside span-to-pair. The gain is at chunks of 4 and 2. MeSH is 0.004 worse than Anchor. Plain+ptr scored 3.7256 (seed 2) and 3.7155 (seed 1) in the decode-loop redesign runs (memory note tul-decode-loop-redesign); training FLOPs per step: spiral 8.73 TFLOP vs plain+ptr 28.50 (0.31x), flat14+ptr 9.01 (bench, morph-scratch/abq/spiral/eval/bench_v2.md).
- 2026-10-15 (gotcha, ours): The update needs RMSNorm and 1/sqrt(g). Eq. 20 as written gives lambda/g. sqrt(g) with no norm blew up to RMS 6.9e9.

## 3. Our speculation

- 2026-09-07 Speculation: The Spiral savings are on training FLOPs. Our decode cost may not change under this design.

## 4. Corrections

- 2026-10-16: The 2026-09-07 summary said the Spiral arm gives "7 to 11 percent fewer FLOPs than a full-resolution looped baseline at equal or better PPL (9.07 to 8.73)." The full read (2026-10-15, section 2 above) shows two different comparisons. Spiral-B gives 7 to 11 percent fewer FLOPs at +0.01 nats, which is slightly worse, not equal or better. The 9.07 to 8.73 change is Spiral-L vs Pythia at 410M. The 2026-09-07 summary is wrong on that point.

## 5. Where our findings disagree

- 2026-10-15: Our measurement: on Parcae, the best flat arm (flat14+ptr, CE@6 3.6731) beats the published spiral (3.6912) at equal FLOPs. Paper's claim: Spiral-L beats Pythia (a flat model) at equal parameters with fewer FLOPs, PPL 9.07 to 8.73 at 410M. Scales differ (our 5k steps, 31M tokens), so this may be pre-knee. Source: filing lxtul/spiral e60a071, failure 5/12 (branch lxtul/spiral; not on master in this checkout).
- 2026-10-15: Our measurement: MeSH topology reads 0.004 nats worse than Anchor on Parcae. Paper's claim: MeSH beats Anchor (9.00 vs 9.13 PPL at 410M). Source: filing lxtul/spiral e60a071 (spiral_span_pair_mesh vs spiral_span_pair).
