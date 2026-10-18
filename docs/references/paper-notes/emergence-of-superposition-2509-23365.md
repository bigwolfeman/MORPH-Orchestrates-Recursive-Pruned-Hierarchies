# Paper note: Emergence of Superposition

- **Title:** Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought (Zhu, Hao, Hu, Jiao, Russell, Tian; ICLR 2026)
- **arXiv or URL:** 2509.23365v3 (v1 2025-09-27, v3 2026-03-01)
- **Local paper text:** docs/references/looping-depth/latent-exploration/emergence-of-superposition-2509-23365/ (PDF and reading file); extracted text in ignore/papers/2509.23365v3-emergence-of-superposition.txt (private)
- **Date first noted:** 2026-10-18 (summarised from an uncached PDF on 2026-09-23; read in full 2026-10-18)

## 1. Why we reference it

- 2026-10-18: It explains WHY one continuous thought holds a set under ordinary
  cross-entropy: the index-matching gain `mu` stays bounded, so the readout keeps comparable
  weight on several candidates. It is the companion of Reasoning by Superposition (same
  authors).
- 2026-10-18: Two parts bear on the coarse-level particle design (K copies of the coarse
  loop from random starts, read as a set by a teacher-forced AR coda; Lean claims C1 to C6 in
  `lab/theory/tul_exploration/TulExploration/Particles.lean`). First, the gain dial between
  spreading over candidates and committing to one, which is the same question as whether the
  coarse map keeps particles in several basins. Second, the exact depth rule of Theorem 5:
  one hop per thought.

## 2. What the paper shows

- 2026-10-18 (Section 3, Eq. 1 to 3, Remark 1): the analysed model is a simplified
  two-layer transformer. Layer-1 copying is assumed learned, embeddings are orthonormal
  (Assumption 1), thought c is assumed to be the uniform superposition of the c-ball `N_c`,
  and layer 2 is LINEAR attention with `V = I`. The gain is `W = sum_v mu_v E(v) E_s(v)^T`;
  permutation averaging makes all `mu_v` one scalar `mu`.
- 2026-10-18 (proof of Lemma 3; Theorem 4): the next-node logits are
  `xi_v = lambda (1{v in N_c} + mu d_v)`, `lambda = 1/sqrt|N_c|`, with `d_v` the in-degree
  from sources in `N_c`. The readout is a Gibbs distribution over nodes with inverse
  temperature `lambda mu` on the in-degree.
- 2026-10-18 (Theorem 1, formal Theorem 4, Eq. 12 and 13): under one-demonstration
  cross-entropy (COCONUT, Eq. 6), `d mu/dt = alpha/(n sqrt K) (d* - F(mu))`, where `F(mu)`
  is the model's own expected in-degree under its readout (unreachable nodes counted at 0)
  and `d*` the demonstrated node's in-degree. `F` rises strictly from 0 to `d_max` and never
  reaches it. If `d* < d_max`, `mu` converges monotonically and exponentially to the unique
  `mu*` with `F(mu*) = d*`. If `d* = d_max`, `mu` grows at least logarithmically. `mu* > 0`
  also needs `F(0) < d*`.
- 2026-10-18 (Theorem 1, Lemma 4, Lemma 5): under the set loss (COCONUT-BFS, Eq. 5) the
  gradient is positive at every `mu`, and `mu(t) >= (1/(2 sqrt|N_c|)) ln(1 + alpha n^-3 e^-2 t)`.
- 2026-10-18 (Theorem 2, formal Theorem 5, Eq. 14): for any positive superposition on `N_c`
  and `mu > 0`, the next thought projects to
  `beta_v = lambda_v 1{v in N_c} + mu sum_{u in N_c} lambda_u 1{(u -> v) in E}`, carryover
  plus one-hop expansion, with support exactly `N_{c+1}`, on any graph.
- 2026-10-18 (Lemma 8, Proposition C.1, Theorem 6): the answer logits are residual carryover
  plus a candidate lift. Gradient flow goes to the max-margin direction with
  `mu_R / mu_A = lambda* + Delta_train`. It generalises to graphs where a non-candidate's
  weight overshoots the answer's by at most `Delta_train`.
- 2026-10-18 (Section 5, Table 3): 2-layer GPT-2 style, d_model 768, 8 heads, AdamW, LR 1e-4,
  batch 256, permuted labels; 150 epochs at stage 1, 25 per later stage, 350 in total.
  Test accuracy 96.2 %. ProsQA subset of 14,785 train problems, about 22.8 nodes.
- 2026-10-18 (Section 5.1, Figure 3): the frontier logit difference, the proxy for `mu`,
  saturates around 60 after about 125 epochs of stage 1.
- 2026-10-18 (Appendix E.2, Eq. 19, Figure 5): the BFS run uses a frontier-only variant
  (COCONUT-BFS-exp). Its logit difference keeps rising through 150 epochs; its test accuracy
  is 99.0 %. At c = 3 and 4 the gap to COCONUT shrinks.
- 2026-10-18 (Section 5.2, Figure 4): the prediction-stage gains plateau after about 5
  epochs, against the theory's unbounded growth. The authors leave this open.
- 2026-10-18 (Appendix F.1, Table 4): layers 2 / 4 / 8 / 12 give 98.8 / 97.3 / 96.5 / 67.4 %;
  d_model 384 / 768 / 1536 give 62.0 / 98.8 / 97.7 %; LR 2e-4 / 1e-4 / 5e-5 give
  58.1 / 98.8 / 62.1 %; tied and untied 98.8 %. With 400 stage-1 epochs and LR 5e-5 the
  12-layer model reaches 99.6 %.
- 2026-10-18 (Appendix F.2, Figure 8): in a 4-layer model the superposition is visible from
  layer 2 on.

## 3. Our speculation

- 2026-10-18 Speculation: `mu` is the Hopfield inverse temperature in FORM (a scalar gain on
  the query-key score of the matching head) but not in the proven model. The theorems use
  linear attention, so `mu` there is only the readout temperature, and the thought is linear
  in `mu` (Eq. 14). The commitment argument (Section 1, Section 2.1: "under large mu and
  commonly used softmax attention") is about softmax attention, outside the theorems. Our
  reading of how they meet: with softmax attention keyed on the summed state, a node's
  weight from in-degree enters the next hop's exponent, so large `mu` compounds in-degree
  into commitment and `mu = 0` spreads weight over every node. These are the polarisation
  and washout regimes of the mixture-collapse paper, with `mu` in the role of the coupling.
- 2026-10-18 Speculation: `mu*` is a moment-matching fit. It is finite exactly when the
  feature the state can see (in-degree) does not pick out the demonstrated target. So
  superposition under cross-entropy is a hedge on an unidentifiable target. The set loss is
  separable and has no finite optimum. For particles, the analogue is that a coarse map
  trained by plain CE through a mixture read should keep several candidates only where the
  context cannot tell them apart.
- 2026-10-18 Speculation: Theorem 5 is linear in the set weights
  (`beta = (I + mu A^T) lambda`). That supports the "near-linear in the set" half of Wolfe's
  criterion for when one superposed vector replaces K particles. The paper says nothing about
  the other half (branches with their own nonlinear multi-step computation).
- 2026-10-18 Speculation: Theorem 5 gives the depth rule: one thought per hop, so depth is
  needed exactly when the needed information is more lookups away than the state holds, and
  extra thoughts keep the answer because the sets are nested. The Parcae windowed fine-level
  run fits it: bounding the reach per pass made depth buy reach (K1-K6 +0.0116 against
  +0.0071; branch `lxtul/localfine` 4a9b3ce, recorded in the memory note
  spiral-levels-ignore-incoming-update). For particles this predicts that depth carries a
  particle into a basin only if the basin's evidence lies beyond one pass's reach.
- 2026-10-18 Speculation: for Lean C3 to C5 the paper is silent. Its readout is a linear
  max-margin read of candidate weights, not a set read over states, and its two losses differ
  in the target set, not in how credit is split across hypotheses.

## 4. Corrections

- 2026-10-18: Our 2026-09-23 summary (in the Reasoning by Superposition reading file) put
  the "bounded index-matching logit can balance exploration and exploitation" quote in
  Section 3. It is in Section 1 (and Section 2.1).
- 2026-10-18: The same summary said the BFS-style loss "did not saturate but kept
  increasing" (Appendix E.2) without saying that run uses the frontier-only variant Eq. 19,
  not the Eq. 5 loss of Theorem 1, and that it scores 99.0 % against COCONUT's 96.2 %.
- 2026-10-18: The same summary said "a loss that pays for 'any member of the set' destroys
  it". The paper shows the gain diverges (Theorem 1 (i)). It does not show the superposition
  is lost. In the proven model the thought's support is `N_{c+1}` for every `mu > 0`
  (Theorem 5), and nobody measured the spread under the set loss.
- 2026-10-18: The same summary listed "too deep per step" (Table 4) as a failure condition.
  Table 4's 12-layer row is 67.4 % at the default budget and 99.6 % with 400 stage-1 epochs at
  LR 5e-5 (Appendix F.1). Depth there is optimisation budget. The width row (384: 62.0 %)
  stands, but Table 4 measures accuracy, not superposition.
- 2026-10-18: Two inconsistencies inside the paper, recorded so nobody quotes one side
  alone: the main run reports 96.2 % (Section 5) while every default row of Table 4 reads
  98.8 %; and a 350-epoch schedule of 150 plus 25 per stage implies nine stages, against
  "never explicitly trained to generate more than two continuous thoughts" (Section 5.1).

## 5. Where our findings disagree

- None recorded yet.
