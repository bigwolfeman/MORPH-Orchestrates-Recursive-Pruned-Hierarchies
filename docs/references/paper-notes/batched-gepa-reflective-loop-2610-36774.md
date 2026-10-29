# Paper note: Batching the Reflective Optimization Loop (parallel proposals for GEPA)

- **Title:** Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better (Zhang, Agrawal, Tan, Lee, Erdogan, Klein, Sen, Dimakis, Zaharia; UC Berkeley)
- **arXiv or URL:** 2610.36774 (v1, 2026-10-27). Type: paper (NeurIPS 2026 workshop, Resource-Aware Agentic AI). Read: the whole paper, 9 pages with appendices A to C.
- **Local paper text:** docs/references/inference-scaling/batched-gepa-reflective-loop-2610-36774/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-28

## 1. Why we reference it

- 2026-10-28: Wolfe flagged it as relevant to the draft-and-revise span loop with K sampled drafts. The overlap is the search pattern only: propose several candidates per step, score them in parallel, keep the winners.
- 2026-10-28: It is NOT a model-architecture paper. It optimizes prompts and agent programs with an LLM proposer (gpt-5-mini) and a black-box evaluator (gpt-4.1-mini). It has no loop state, no hidden state, no diffusion, no hypersphere and no training of weights. Treat it as search-systems evidence only.

## 2. What the paper shows

- 2026-10-28 (Section 2, Figure 2, Figure 3): standard GEPA does one proposal per step: sample a parent from a Pareto frontier, propose one reflective mutation, test on a mini-batch, run full validation if it beats the parent. The P x N strategy samples P parents, draws N mutations per parent, scores all P*N children concurrently, and updates the frontier once.
- 2026-10-28 (Section 3, eq. 1 to 3): runtime model. Chance that a step triggers full validation is q_k = 1 - (1 - a)^k for acceptance rate a and width k = P*N. Speedup is about k (L_step + a L_val) / (L_step + q_k L_val). With W workers, validation latency is Te if kV <= W, else (kV/W) Te. The model matches the measured times (Figure 4).
- 2026-10-28 (Section 4.1, Figure 4): LiveBench-Math, 5,000 metric calls, 512 workers. Single mutation 68.9 % test, 7.7 hours. 2x2 reaches 72.1 % test in 4.1 hours (1.9x speedup). Iterations fell 4.9x (219 to 45) but the share of steps that ran full validation rose from 17 % to 53 %. HoVer, 3,000 metric calls, 64 workers. Single mutation 49.0 % (share of test claims with all three gold pages retrieved). 8x1 reaches 60.0 %. Text claims "about 3 to 4x" cut in wall-clock at 16-way width.
- 2026-10-28 (Section 4.2): with P = 2, test score rises with N: 68.6 (2x1) to 71.6 to 72.1 (2x2 through 2x8) on LiveBench-Math; 52.5 to 55.0 on HoVer. Best split depends on the task (2x2 vs 8x1). The authors leave other budgets and parallelism levels for future work.
- 2026-10-28 (Section 4.3, Figures 5, 6): 2x4 vs single mutation at equal metric calls. Test gain +3.0 points (LiveBench-Math) and +6.0 points (HoVer). On LiveBench-Math single mutation scored higher on validation but lower on test; its validation-to-test drop was nine points against two for 2x4. Total spend stayed in the $12 to $16 range for all settings. 2x4 reaches its final validation score within about $4 on LiveBench-Math.
- 2026-10-28 (Appendix A): four dials per step: P parents, N mutations per parent, reflection mini-batch size (ComBEE, 2604.04247), reflection compute. How to split a fixed budget across the four is left open.
- Limits the authors state or show: one run per cell (no seeds or error bars are printed); two tasks; one proposer model; the best (P, N) split is task dependent and untuned beyond the sweep; the generalization story (fewer adaptive rounds against one holdout) is offered as "consistent with" a hypothesis, not tested.

## 3. Our speculation

- 2026-10-28 Speculation: the paper's one transferable idea is committing to several candidates per round before adapting to the selector. For K sampled drafts with scores, the analogue is scoring all K drafts against the same target before any gradient or selection step, so the selector is not tuned draw by draw. Their validation-to-test gap (9 points vs 2) is a selection-bias effect on a small holdout (100 to 150 examples). Our selector regret (selector-regret-is-hindsight-price) is a related effect, but we have no measurement that ties the two.
- 2026-10-28 Speculation: the choice between "more parents" (P) and "more mutations per parent" (N) maps loosely to K drafts of ONE span state vs drafts from several span states. The paper says N helped on the task with diverse problems and P helped on the task with complementary specialists. We have no such split in our fan.
- 2026-10-28 Speculation: Amdahl-style runtime model (eq. 2) applies to a K-draft loop only as a cost lever: K parallel drafts cost one latency if the GPU has room. It says nothing about whether the drafts earn CE.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-28: Nothing to compare. The paper has no language-model training, no CE or bpb number, no loop depth and no span model. It neither supports nor argues against draft-and-revise. Its one hint against K-draft gains is indirect: gains came at the same metric-call budget only because evaluation is cheap and parallel; in our loop a draft costs a full forward pass.

## 6. Triage

Score 1. Search-systems paper on prompt optimization with an external LLM. No architecture, objective or geometry content that changes what we run next.

Test ideas (control and predicted behavior):
1. None from this paper. The K-draft lever stays tested by our own explorative-modeling and fan work.
