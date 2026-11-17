# Paper note: Spherical Brownian Diffusion (SBD)

- **Title:** Spherical Brownian Diffusion for Language Modeling
- **arXiv or URL:** arXiv 2610.35101
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Checked for latent targets. Low value for TUL.

## 2. What the paper shows

Read 2026-11-17 from the arXiv HTML. Only the first 100k of 227k characters were read. Training cost sections (D.7, D.8) were not read.

- 2026-11-17 (Sec. 2.3, 3): Unit-norm token embeddings, corrupted by Brownian motion on the sphere. A transformer trained with CE predicts the clean-token posterior. One model gives continuous SDE/ODE sampling and categorical refinement.
- 2026-11-17 (Table 3): Hard Sudoku 83.36 (gated SBD) vs 1.0 AR greedy, 58.4 Duo, 81.65 FMLM+.
- 2026-11-17 (Sec. 5.2, 5.3): GSM8K (TinyGSM, 250k updates) trails Duo by about 6 points at 1024 NFE. OWT (1M updates, length 1024): MDLM and Duo keep a better quality-diversity trade-off over much of the range.
- 2026-11-17: No sentence-level latent. Per-token embeddings only.

## 3. Our speculation

- 2026-11-17 Speculation: Nothing here gives a coarse state a job. It is token diffusion with a sphere kernel.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-11-17: None. Consistent with item 5 (token diffusion is costly and does not beat AR on web text).
