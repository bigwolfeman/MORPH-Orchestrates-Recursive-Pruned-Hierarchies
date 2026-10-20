# Paper note: Squeezing more from limited data with recursive transformers (RecursiveGPT)

- **Title:** Squeezing More from Limited Data with Recursive Transformers (Gulbahar, Edman, Fraser)
- **arXiv or URL:** 2608.26973 (v1, 2026-08-27, cs.CL). Type: paper. Code: github.com/serdardoesml/recursive-lm (not read).
- **Local paper text:** docs/references/looping-depth/recursive-transformers-limited-data-2608-26973/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: A looped LM with a measured depth sweep and a settle-depth readout, but at BabyLM scale with a data-constrained, multi-epoch setup and a non-Parcae design (one shared block, no prelude or coda, fixed R). Triage score 1.

## 2. What the paper shows

Read: abstract, Sections 1 to 6, Table 1, ablation text, Appendix C (prediction depth). Not read: Table 3 numbers (ablation of per-step norm sharing and one more component) beyond the text, Appendices B, D to F.

- 2026-10-20 (Table 1, 3 seeds, avg of BLiMP/COMPS/LAMBADA p@5): 10M words: recursive R=16 27.6M params 45.80 vs standard 44.93 (41.1M) and standard+factorized-embedding 45.07 (28.7M). 100M words: recursive R=24 124M 61.92 vs standard 1.22B 63.47; recursive-large 404M 63.93.
- 2026-10-20 (Fig 5): accuracy improves with R up to 16 then drops slightly at larger R (10 % subset of 100M corpus). Authors tie this to a too-near-identity or amplifying transition and cite residual-explosion work.
- 2026-10-20 (App C, R=24 large model, 65,536 tokens): top-1 settle depth under a logit lens: about half of tokens stable by depth 16, 70.1 % by 22, 80.9 % by 23; the last two depths matter for 29.9 %.
- 2026-10-20 Failure modes: per-step RMSNorm and bias are not shared (the only per-pass parameters); simple fixed R chosen because of "optimization difficulties with recursive weight sharing"; 100M models stay below the optimal scale for lack of compute.

## 3. Our speculation

- 2026-10-20 Speculation: the data-limited regime (10 epochs of 10M words) is not our regime (one pass over OpenWebText, 31M tokens at 5k steps), so the "recursion beats scaling parameters" claim does not transfer to a K1-K6 reading.
- 2026-10-20 Speculation: per-pass norm and bias as a cheap depth embedding is the same idea as the gate in the gated recurrent transformer note, at much lower cost. Neither has a K1-K6 reading.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None.
