# Paper note: CosmicFish-HRM (adaptive reasoning in a compact language model)

- **Title:** CosmicFish-HRM: Adaptive Reasoning via Hierarchical Recurrent Mechanisms in Compact Language Models (Venkat Akhil Lakkapragada, Mistyoz AI)
- **arXiv or URL:** 2605.28919 (v1, 2026-05-27)
- **Local paper text:** docs/references/looping-depth/cosmicfish-hrm-2605-28919/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20
- **Triage score:** 1. LM-scale HRM loop, but no loop-off control, no seeds, no CE by step budget.

## 1. Why we reference it

- 2026-10-20: One of few papers that puts an HRM-style loop inside a decoder language model trained on web text. It is the right size class (82.8M) to compare with our testbed, so what it does not report matters as much as what it does.

## 2. What the paper shows

Read in full (about 5.5k words, 12 pages of main text). One training run, no seeds, no ablation.

- 2026-10-20 (Sec 3, Table 1): 6 input blocks, an HRM core (4 high plus 4 low blocks, 2 low cycles per high cycle, up to 16 steps), 6 output blocks, d=448, seq 512, 10B-token CosmicSet, 27.5k iterations at batch 128, final validation loss 3.36. Halting head on the mean-pooled high state, 10 % random deferral in training, +0.35 halt bias at inference, loss = LM + 0.01 x mean steps (Eq 15).
- 2026-10-20 (Table 2): zero-shot HellaSwag 26.2, PIQA 58.1, WinoGrande 50.7, against GPT-2 small 29.7 / 62.5 / 50.7 and OPT-125M 30.6 / 62.6 / 52.9. It is below same-size transformers on the first two. The author attributes this to overhead at small scale and states a hypothesis, not a measurement, that it improves with size.
- 2026-10-20 (Table 3): mean HRM steps per task 3.03 (HellaSwag, std 6.26), 1.87 (PIQA), 0.95 (WinoGrande), 0.80 (TriviaQA), overall 2.68 (std 5.95). A mean under 1 step with std near 3 to 6 means most items halt at zero or one step and a tail runs long. Table 4 gives four hand-picked prompts: 2.78 steps (capital of France), 4.77, 7.03, 8.40 (bat and ball).
- Failure modes reported: lower benchmark scores than the baselines; the author notes the benchmarks are shallow and do not test long reasoning. The paper does not report a no-HRM control at the same budget, loss by step count, or CE at fixed depth.

## 3. Our speculation

- 2026-10-20 Speculation: the one result that matters to us is missing: validation CE at step budget 1, 2, 4, 8 on one checkpoint. Without it, "non-uniform step use" is a statement about the halting head, not about the loop earning anything. A halting head with a step penalty can learn a token-length or entropy proxy and still give the loop zero value. Our K1-K6 on fixed rows is that missing measurement.
- 2026-10-20 Speculation: the Table 3 pattern (mean below 1 step) suggests the model mostly skips the core. That fits our "loop is worth 0.017 nats" and "skip flattens K-curve, CE improves" memories: a learned halting cost pushes the loop off.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper claim: adaptive depth is a promising alternative to parameter scale. The data in the paper (Table 2) show a loss to same-size fixed-depth models and no loop ablation, so the claim is unsupported by the paper's own numbers. Consistent with our rule that depth is judged by K-curves against a plain loop, not by step counts.
