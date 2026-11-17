# Paper note: LOOM (Looping Beyond Twice, looped Mixture-of-Experts)

- **Title:** Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts (arXiv 2610.01153v1)
- **arXiv or URL:** arXiv 2610.01153
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Recipe paper for stable deep loops (H up to 12) at 100M to 1.7B. Read for what makes a loop earn at more than 2 passes.

## 2. What the paper shows

Read 2026-11-17 from arxiv.org/html/2610.01153 (via a summarising fetch; section numbers as the fetch gave them).

- 2026-11-17 (models): about 100M (70M active, 5B tokens), 350M (141M active, 10B), 1.7B (0.63B active, 60B). A 700M iso-FLOP study at 10B tokens (Table 2).
- 2026-11-17 (recipe): residual scale gamma = 0.5/(H sqrt(M)); embedding re-injection with gate 0.5/(t sqrt(M)) from loop 2; a two-memory EMA "Looping Residual" (global persists, local resets); per-loop routers, shared experts and attention; extra RMSNorm on the MoE branch. LM loss at the end of each segment.
- 2026-11-17 (truncation): segmented BPTT, K = 3 loops per segment, state passed forward detached (Sec 3.3). At 350M, H = 12: peak memory 12.6 GiB vs 19.3 GiB unsegmented; wall clock 12.6 h vs 26.1 h (Fig 5). Unsegmented 9 loops fails: 81.04 ppl (Table 6), segmented 14.86.
- 2026-11-17 (gain, control = non-looped model, same width and M layers; no depth-1 twin): 700M iso-FLOP: 18.36 ppl flat, 17.37 at 2x, 16.58 at 4x, 16.54 at 5x, 16.57 at 6x (Table 2). Gain plateaus at 4 to 5 loops. 1.7B: flat 9.62, 9 loops 7.77 (Table 4). The "iso-FLOP" is near, not exact (90 vs 84 units).
- 2026-11-17 (partial recipe): at 100M, 3 loops with no technique or only one of the tricks is WORSE than the flat model (27.61 / 27.48 / 28.99 vs 26.26, Table 3), and deeper loops spike. Dropping residual scaling at 350M, 9 loops, step 5k costs 6 ppl (Table 5).
- 2026-11-17 (not reported): no token-budget crossover, no step-level time when the loop starts to pay. Only the step-5k ablation.

## 3. Our speculation

- 2026-11-17 Speculation: the recipe is a stabiliser. It shows deep loops fail without gain control, which matches our divergence work. It does not show the loop earns because of the recipe, only that it stops losing.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- Item 7 (value forms late): the paper gives no curve, so no test either way.
- Item 9: this paper adds no aux depth loss, it uses the main LM loss per segment. Consistent. Its per-segment loss at an intermediate loop is the one place it supervises a middle pass. Note this when we judge "only the main loss buys depth".
