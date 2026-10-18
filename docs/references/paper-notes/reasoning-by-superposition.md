# Paper note: Reasoning by Superposition

- **Title:** Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought (Zhu, Hao, Hu, Jiao, Russell, Tian; NeurIPS 2025)
- **arXiv or URL:** 2505.12514; proceedings PDF https://proceedings.neurips.cc/paper_files/paper/2025/file/72c363c2a573ca2128bd176d3317696b-Paper-Conference.pdf
- **Local paper text:** docs/references/looping-depth/latent-exploration/reasoning-by-superposition/reasoning-by-superposition.pdf (reading file beside it); extracted text in ignore/papers/neurips2025-reasoning-by-superposition.txt (private)
- **Date first noted:** 2026-10-18 (paper first read 2026-09-18, re-read 2026-09-23)

## 1. Why we reference it

- 2026-10-18: It is the one published case where ONE vector holds a set of candidates and
  the set is used: a two-layer transformer runs breadth-first search with one vector per
  step. It was item 1 of the 2026-09-18 latent-exploration survey and the reason the Thought
  Register arm (`tul.slot_cells: 4`) was built.
- 2026-10-18: The coarse-level particle design on the Parcae spiral testbed runs K copies of
  the coarse loop `x^{t+1} = F(D x^t + Delta B z)` from independent random starts, read by a
  teacher-forced AR coda as a set (Lean claims C1 to C6 in
  `lab/theory/tul_exploration/TulExploration/Particles.lean`). This paper is the one-vector
  alternative to those K particles. The question it bears on: when does one superposed
  vector do the job of K particles?

## 2. What the paper shows

Numbers below come from the 2026-09-18 reading file. Spot-checked 2026-10-18 against
`ignore/papers/neurips2025-reasoning-by-superposition.txt`: Table 1 (all eight step-1 and
step-4 values), Section 5.2 (75 % and 83 %), Section 5.1 (25 epochs per stage, 300 in
total). All three match.

- 2026-10-18 (Lemma 2, Section 4.2): a construction exists in which the c-th continuous
  thought is exactly `(1/sqrt|V_c|) sum_{v in V_c} u_v`, the uniform superposition of every
  vertex within c hops of the root. Layer 1 has five fixed "attention chooser" heads that
  copy each edge's source and target onto its `<e>` token. Layer 2 has one head: the thought
  is the query, the edge source is the key, the edge target is the value. One pass is one
  frontier expansion.
- 2026-10-18 (Section 4.2, "MLP as filter"; Appendix B, Proposition B.2): softmax attention
  leaves noise mass on unwanted tokens. The MLP rotates into the vertex basis
  (`W1 = [u_1 ... u_V]^T`), applies a coordinate-wise threshold `1{x >= eps}`, and rotates
  back. LayerNorm restores the scale. The nonlinearity acts on each vertex separately.
- 2026-10-18 (Theorem 1, Section 4.3): with D continuous steps for a graph of diameter D and
  `d = O(|Voc|)`, the `<A>` token "measures" the superposition against the two candidates by
  inner product. The best known discrete-CoT result for constant depth needs `O(n^2)` steps
  (Section 1, citing Merrill and Sabharwal).
- 2026-10-18 (Section 5.1): GPT-2 style, 2 layers, d_model 768, 8 heads, from scratch,
  AdamW, constant LR 1e-4, ProsQA subset of 3 to 4 hops. Multi-stage curriculum, 25 epochs
  per stage, 300 epochs in total, previous stage mixed in at 0.1.
- 2026-10-18 (Section 5.2, Figure 4): Coconut near-perfect. Discrete CoT and no-CoT both
  about 75 % (chance 50 %). A 12-layer, 12-head CoT model reaches 83 %.
- 2026-10-18 (Section 5.4, Table 1): layer-2 attention by edge group, step 1 / step 4:
  not reachable 0.04 / 0.12, reachable 2.12 / 0.29, frontier 2.12 / 0.61, optimal 2.54 / 2.23.
- 2026-10-18 (Section 5.4, Figure 6): COCONUT-BFS, whose stage-i target is drawn uniformly
  from the nodes exactly i hops out, reaches the same near-perfect accuracy and a similar
  inner-product geometry. The trained thought is deterministic; nothing samples.

## 3. Our speculation

- 2026-10-18 Speculation: the step works on a set because it is ITEM-SEPARABLE, not because
  it is linear. Attention keyed on the summed state retrieves every matching edge at once,
  the MLP filter acts per vertex, and LayerNorm is one global scale. Two more conditions
  hold: every member runs the same program (look up my out-edges), and the items are
  near-orthogonal (`d = O(|Voc|)`). Our refinement of Wolfe's candidate criterion: one vector
  can stand in for K particles when the per-step map acts per item in a code where the items
  are near-orthogonal, and the width covers the set. K particles are needed when branches
  need different nonlinear computations whose cross terms do not cancel. This paper only
  shows the first half. None of the four papers we read on this (this one, Illusion,
  Emergence, mixture collapse) runs K separate states against one superposed state.
- 2026-10-18 Speculation: the MLP's re-equalisation is what lets a sharp attention gain
  avoid commitment here. With all set weights equal, every matching edge ties, so even a
  large gain spreads attention evenly. Without it (the Emergence model, Eq. 14 there), weights
  grow with in-degree hop after hop, and a large gain would pick the heaviest branch. A
  coarse particle loop has no such per-item filter.
- 2026-10-18 Speculation: Lemma 2 makes depth buy exactly one hop per pass. The Parcae
  windowed fine-level run found the same shape: bounding reach per pass made depth buy reach
  (K1-K6 +0.0116 against +0.0071, branch `lxtul/localfine` 4a9b3ce in the testbed; recorded
  in the memory note spiral-levels-ignore-incoming-update, filing not in this tree).

## 4. Corrections

- 2026-10-18: Our 2026-09-18 reading used `val/slot_eff_rank` 5.76 at cosine 0.7104 on
  `slot-spandec-strict` as evidence that our slot states are "low-rank and collapsed",
  against this paper's rank-k set. That number is a bare-front reading from the pre-7a24adf
  probe. The checkpoint under the shipped probe reads 13.85 at cosine 0.52
  (`lab/experiments/failures/2026-09-13-arc-cond4-reader.md`, Method amendment;
  `lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`). The contrast is weaker
  than the reading says. This corrects our evidence, not the paper.
- 2026-10-18: The 2026-09-23 re-read in the reading file summarised the companion papers
  (Emergence of Superposition, and Backour's mixture collapse) without caching them. Both
  are now read in full. The corrections are in
  [`emergence-of-superposition-2509-23365.md`](emergence-of-superposition-2509-23365.md)
  and [`continuous-mixture-collapse-2609-02049.md`](continuous-mixture-collapse-2609-02049.md),
  section 4 of each. The reading file has a dated pointer at its end.

## 5. Where our findings disagree

- 2026-10-18: Our measurement: four slot cells per span (`tul.slot_cells: 4`) read effective
  rank 1.2445 of 4 at within-slot cosine 0.9432, and seeding them apart or together moved the
  rank by 0.008. Paper's claim: one trained 768-wide two-layer model holds the whole
  reachable set (Lemma 2; Section 5.3, Figure 6). The settings differ (web-text next-token
  loss against ProsQA with a hop curriculum; M cells against one vector), so this is a
  contrast, not a refutation. Source: `lab/experiments/failures/2026-09-13-arc-thought-register.md`
  (measured 2026-09-13).
- 2026-10-18: Agreement, not disagreement. Our measurement: where the geometry forced hops
  (`slot-spandec-strict-prev-reach1` @ 5000), content h spans back arrived at pass h - 1 and
  kept improving for about one more pass; planted content g spans back read exactly zero
  before pass g - 1, on two disjoint 480-row draws. Paper's claim: one frontier expansion
  per pass (Lemma 2). Source: `lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md`,
  Verdict items 1 and 3 and P4 (measured 2026-09-19).
