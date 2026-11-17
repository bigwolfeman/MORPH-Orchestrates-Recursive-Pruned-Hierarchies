# Paper note: Binary Discrete Diffusion (candidate-set diffusion)

- **Title:** Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning
- **arXiv or URL:** arXiv 2610.28543
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Checked as a job with a natural per-pass step: narrow a candidate set per pass.

## 2. What the paper shows

Read 2026-11-17 from the arXiv HTML. The fetch tool read only the first 100k of 209k characters. Appendix F.6 (GenPPL) and any cost section were NOT read.

- 2026-11-17 (Sec. 3): A noisy position is a SET of candidate tokens (binary membership vector). A clean token is a singleton. Symmetric kernel adds and removes members. Absorbing kernel only adds. The net still outputs one V-way distribution per position, and all reverse rates cost O(V) (Sec. 3.3). A set is embedded as a normalized sum of token embeddings (Eq. 15).
- 2026-11-17 (Sec. 4, Alg. 1): Each sampling step mixes ancestral updates (keep the current set) with re-corruption (forget it). Confidence rules choose which positions and candidates to reopen.
- 2026-11-17 (Sec. 5, Table 1): 43M diffusion transformers. Sudoku-Extreme 96.27 / 92.66 (full / hard) for binary symmetric vs 87.96 / 76.12 masked. Boxoban 94.29 vs 91.26. Four-coloring of big graphs 64.13 vs 4.85 masked.
- 2026-11-17 (Table 2): Language: Text8, TinyStories, OpenWebText, 131B tokens each. Metric is MAUVE. Binary symmetric wins Text8 (0.913 vs 0.854), loses on TinyStories (0.882 vs 0.904 uniform) and OWT (0.840 vs 0.871 masked). No perplexity in the main text. No cost against autoregressive training was found in the part read.

## 3. Our speculation

- 2026-11-17 Speculation: A set that shrinks per pass is a real per-pass job on logic tasks (sudoku, graph coloring). On web text a span has too much entropy for the set to narrow to one token in a few passes, and a set state is still a token-level object (one set per position), so it does not make the state coarse. Training cost against AR is unknown from what we read, but token diffusion cost us 16 to 64x (item 5).

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-11-17: Compatible with item 6 (a hop-style task earns per pass). No LM result that beats a masked baseline.
