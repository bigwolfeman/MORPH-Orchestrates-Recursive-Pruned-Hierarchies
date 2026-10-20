# Paper note: SMELT, scaling laws for compute-matched MoE looped transformers

- **Title:** SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers (Wang, Zhang, Luo, Wu, Liu, Liu, Huang, Yan, Li)
- **arXiv or URL:** 2609.01343 (v1 2026-09-01, v3 2026-09-11). Type: paper.
- **Local paper text:** docs/references/looping-depth/smelt-moe-looped-scaling-2609-01343/ (paper.txt, paper.pdf 2.4 MB, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is the cleanest published FLOP, parameter and KV-matched test of "does looping beat not looping", with a loop-span and loop-count sweep. Our loop uses prelude 4 / core 6 looped Poisson mean 6 / coda 4, which loops a middle slice, as SMELT recommends.
- 2026-10-20: Read the intro, related work, Sec. 3 (ablation table headings and text) and Sec. 4 start. Scaling-law fit details, Sec. 5 downstream and Sec. 6 mechanism only skimmed through the abstract and summary text.

## 2. What the paper shows

- 2026-10-20 (Sec. 3, Fig. 1): MoE decoder (top-8 experts, GQA). The looped model loops the middle 50 % of layers twice, narrows the hidden width, raises the expert count to restore total parameters, shrinks head size and raises GQA ratio to hold KV cache, and scales looped residual updates by 1/r.
- 2026-10-20 (Sec. 3.3 to 3.5, 200M ablations): looping the middle half beats full-stack looping (loss minimum near 50 % span at both S ~ 85 % and 95 % sparsity); a larger effective depth-to-width ratio than the baseline is optimal; two visits win, three and four regress because matched FLOPs force a thinner model (the regression is not ordered in loop count).
- 2026-10-20 (Sec. 4, Fig. 1b): 4 sizes (100M to 1.6B active, up to 54B non-embedding) x 4 sparsity levels, separate Chinchilla-style fits per architecture; SMELT is lower in every cell; compute saving 6.8 % to 18.0 % on the frontier (14.7 % at 1e21 FLOPs, S ~ 97 %).
- 2026-10-20 (Sec. 5, 6, abstract level): gains beyond what validation loss predicts on DCLM and MMLU, largest on code, growing with sample length and number of in-context examples; second visit has similar expert choice and attention targets, bigger aligned residual updates, less attention sink.
- Failure mode: advantage is sparsity-dependent; the dense S=0 control cell exists but its numbers were not read.

## 3. Our speculation

- 2026-10-20 Speculation: the loop-twice optimum at matched FLOPs does not transfer to us, since our compute is NOT matched (loop depth buys extra FLOPs, K1-K6 is the contribution test). It does support keeping a prelude and coda around the loop and scaling looped residual updates (our injection floor is the same kind of damping).
- 2026-10-20 Speculation: "second visit amplifies the first, residual grows and stays aligned" fits our reading that a plain Parcae pass is per-token refinement (+0.0377 K1-K6).
- 2026-10-20 Speculation: no training-speed information beyond FLOP matching; MoE is out of our stack.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None found. Their dense-baseline comparison is at a token-matched compute frontier at much larger scale than our 5k, 31M token tests.
