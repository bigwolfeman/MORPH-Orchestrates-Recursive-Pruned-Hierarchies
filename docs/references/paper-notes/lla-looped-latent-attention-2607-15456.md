# Paper note: Looped Latent Attention (cross-loop KV compression)

- **Title:** Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers (James O'Neill, Fergal Reid)
- **arXiv or URL:** 2607.15456 (v1 2026-07-16, v2 2026-07-24)
- **Local paper text:** docs/references/looping-depth/lla-looped-latent-attention-2607-15456/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: A post-training KV cache codec for looped decoders. Decode speed is not our priority and we have no cached loop decoding on the fan. It matters only for its measured facts on what a per-token loop trajectory looks like in pretrained looped LMs (Ouro, Huginn).

## 2. What the paper shows

Read: Sec 1 to 5 and Table 1; later sections (long rollouts, serving, appendices) from the abstract only.

- 2026-10-20 (Sec 4, Fig 2): in frozen Ouro-1.4B and Ouro-2.6B-Thinking the loop axis of the K/V cache has normalized effective rank about 0.61 to 0.63, one direction carries about 60 % of the energy; the head and layer axes are near full rank. Only 6 % of K variance and 17 % of V variance lie on the loop axis; most variation is content. V converges across loops more slowly than K.
- 2026-10-20 (Sec 4, Fig 4): the same pattern holds for Huginn-3.5B at up to 32 steps: successive steps converge toward a fixed point, each step refines less.
- 2026-10-20 (Table 1, Ouro-1.4B, T=4): at 4x compression, GSM8K: teacher 0.794, loop codec (LLA) 0.800, head-axis MLA 0.522, layer sharing 0.660, KV quant 0.702, and zero-parameter final-loop reuse 0.000 (MMLU 0.537 vs 0.682). At 21.33x LLA keeps GSM8K 0.740. GSM8K is flat near 0.58 from 4x to 2.67x in a cold-start sweep, then jumps to 0.75 at 2x.
- 2026-10-20 (abstract): serving batch capacity 32 to 768 sequences at 21.3x; on-policy refinement lifts MATH-500 at 4x from 0.43 to 0.66.
- Failure modes: final-loop reuse collapses generation; long-rollout fidelity needs an on-policy fix (teacher-forced KL is not enough).

## 3. Our speculation

- 2026-10-20 Speculation: "per-token trajectory across loops is low rank (one direction about 60 %) and V changes more than K" is a testable property of our core. Read the K and V of each pass (or the block output) for the 480 fixed rows and report effective rank along the pass axis for plain Parcae vs the fan. A near-rank-1 trajectory would mean passes mostly rescale, which fits the +0.0377 K1-K6 being small next to the 0.87 injection floor.
- 2026-10-20 Speculation: "final-loop reuse gives 0.000" says the intermediate loop states are not redundant even when the trajectory is low rank. It is one more point against reading a flat K-curve as "the loop does nothing".

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: None. Its models are pretrained Ouro and Huginn, not 5k-step Parcae. No test of loop contribution (no K-curve), only of cache fidelity.
