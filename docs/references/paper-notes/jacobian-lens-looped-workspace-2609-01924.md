# Paper note: Looped transformers under the Jacobian lens (global workspace under recurrence)

- **Title:** Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence? (Wenlong Wang, Fergal Reid; Fin AI Research)
- **arXiv or URL:** 2609.01924 (v1, 2026-09-01, cs.AI). Type: paper.
- **Local paper text:** docs/references/looping-depth/jacobian-lens-looped-workspace-2609-01924/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is an interpretability study of released Ouro-2.6B and Huginn-0125, with Qwen3.6-27B as baseline. It speaks to where a loop keeps its answer and what a loop iteration can read. No training result. Triage score 1: weak link to a 5k-step test.

## 2. What the paper shows

Read: abstract, Sections 1, 4.1 to 4.5 headings, 5.2, 5.3, 7.3 text. Skimmed only: Sections 6 and 7.1 to 7.2, 7.4, 7.5 and the appendices (eleven causal experiment families are not individually checked).

- 2026-10-20 (Sec 5.2, Huginn, 4-layer core x 16): the latent answer (Tokyo) is rank 1 by recurrence 2 to 3 and persists through all 16; each lens reads reliably only in the one or two recurrences before its own target (transport is distance-limited). Harder prompts settle later.
- 2026-10-20 (Sec 5.3): logit-lens top-1 equals the emitted token at 15 to 25 % of positions flat across Huginn's recurrent stack (final coda state 63 %); Qwen never exceeds 9 % mid-stack; Ouro 1 to 7 % mid-loop, 64 to 71 % at its supervised checkpoints. The authors credit Huginn's sampled recurrence count (mean 32) as implicit deep supervision.
- 2026-10-20 (Sec 7.3): swapping a concept Huginn is about to emit gets reported in the top-5 at 98 % (124/126); injecting a concept it was not computing never becomes reportable (0 of 97 within rank 100). Ouro reports injected concepts top-1 at 31 %, the only model with explicit per-iteration loss. Ouro's loop boundary destroys the workspace component each loop and re-derives it (Sec 4.2); linear transport cannot cross it.
- 2026-10-20 Failure modes: two models, one family each; hand-picked 8-prompt suite; estimator for occupancy is the authors' own re-implementation (no released original).

## 3. Our speculation

- 2026-10-20 Speculation: Huginn's answer is formed by pass 2 to 3 and persists. That fits our measure that a coarse/span state carries its context in one pass ("necessary is not deep") and that the Parcae loop earns about +0.04 K1-K6 as per-token refinement. It is one more model where the answer exists early.
- 2026-10-20 Speculation: the claim that head pressure at sampled depths replaces explicit per-step loss is the same mechanism as our Poisson draw. It supports our rule that no pass-indexed auxiliary loss is needed to make mid-depth states decodable.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None.
