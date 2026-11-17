# Paper note: LIFT (Latent Information Feedback Transformer), teacher-supervised pretraining

- **Title:** Pretraining Latent Information Feedback Transformers with Teacher Supervision (Tirosh, Amos, Geva)
- **arXiv or URL:** 2609.38149 (v1). Not the same as LiFT 2610.05538 (loop flow transformers). Read: HTML text through Appendix D and the cost section; the last 13k characters (late appendix) were not read.
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: It is a deep-to-shallow feedback channel trained in parallel, with a frozen teacher that supplies the fed-back state. That is the same shape as our one training-efficiency gain (frozen teacher top-k KL, lesson 8).
- 2026-11-17: Its own ablation says the teacher, not the loop, makes the gain. Useful for judging any "feedback" idea in TUL.

## 2. What the paper shows

- 2026-11-17 (Sec 3.1, Eq 4-5): the fed-back state is NOT a hidden layer. It is a next-token distribution: top-k (k = 1024) logits, softmax with temperature 1.5, tail mass in one extra outcome. It is projected through the embedding table and fused into the INPUT embedding of the next position (about 11 d^2 parameters, about 3 % of 1B). Path: last layer at position i to input layer at position i+1.
- 2026-11-17 (Sec 3.2, Eq 6): training. A frozen same-size teacher (10x Chinchilla tokens) runs once over the sequence in parallel and supplies the state for each position. Loss is next-token CE plus 1.5 x forward KL (teacher vs student top-k). Prefix state dropout p = 0.5 (Sec B.2). Last 10 % of training: teacher dropped, the model makes its own states with one or two no-grad passes (second with probability 0.5) and trains on them.
- 2026-11-17 (why a teacher): student states would need sequential rollouts. A teacher gives states parallel and independent of the student.
- 2026-11-17 (Table 1, Sec 5.2): C4 PPL, token-matched Transformer vs LIFT: 135M 32.43 vs 30.65, 350M 25.15 vs 23.82, 1B 19.79 vs 18.81 (-5.0 to -5.5 %). Compute-matched Transformer (teacher FLOPs included): 31.04, 24.32, 19.17, so LIFT still leads by 0.4 to 0.5 PPL. 33 of 36 downstream cells won. Three seeds only at 135M.
- 2026-11-17 (cost): fusion adds about 4.4 % per token in the trunk, the adaptation phase adds 52 % to its steps, whole run about +9.6 %. Teacher forward is charged in the compute-matched control. Inference about +4.5 % per token (97 MFLOPs at 1B), one refinement pass doubles prefill.
- 2026-11-17 (Appendix D.1, 135M, 3 seeds): full LIFT 0.47 PPL better than compute-matched. No teacher (own states only): +0.01, a tie. Teacher as input only: 0.19 behind. Teacher as target only (distillation): 0.77 behind. So about 98 % of the gain needs teacher and channel TOGETHER. D.2: channel width k = 1 keeps 13 % of the gain, k = 128 keeps 86 %. D.4: a stronger teacher changes little.
- 2026-11-17 (Sec 4, Figure 2): on S5 permutation composition, a 22M 2-layer LIFT solves lengths to N = 12 where an 8x-data Transformer fails from 12. A recurrent teacher gives 96 % at N = 32. The authors say it does not buy unbounded depth.
- 2026-11-17 (Sec 5.3, Table 2): vs T2MLR (Jacobi-style feedback) LIFT is equal or better at matched tokens and compute.

## 3. Our speculation

- 2026-11-17 Speculation: the feedback state is a soft next-token guess, so it is close to a token path. On web text the channel mostly passes the teacher's belief forward. The "no teacher = tie" row says the self-fed channel earns nothing by itself on LM. That matches our bypass law (item 1) and item 9 (only main loss buys depth).
- 2026-11-17 Speculation: LIFT is a distillation gain with a channel that makes the student able to USE the teacher at inference. Our lesson 8 gain (top-64 KL) is the first half of that. The channel is the second half and is untested in TUL.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-11-17: Paper: feedback gives work a Transformer cannot do (S5). Ours: on LM the gain without a teacher is 0 (D.1). The two agree only if the S5 gain comes from a serial task that tokens cannot bypass.

## 6. Triage

Score 2. See /home/wolfe/morph-scratch/abq/lit/2026-11-17-tul-papers/feedback-energy.md for the test.
