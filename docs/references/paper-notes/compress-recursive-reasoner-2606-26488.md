# Paper note: What Survives When You Compress a Recursive Reasoner for the Edge?

- **Title:** What Survives When You Compress a Recursive Reasoner for the Edge? (Pearse Jim, Steven Kolawole, Opegbemi M. Busoye, Glory Bagai, Virginia Smith; ML Collective, CMU)
- **arXiv or URL:** 2606.26488 (v1 2026-06-25, v2 2026-09-30)
- **Local paper text:** docs/references/looping-depth/compress-recursive-reasoner-2606-26488/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20
- **Triage score:** 1. Quantized-loop compounding and a label-free fidelity metric; weak link to our fp training, usable as a ternary-core probe.

## 1. Why we reference it

- 2026-10-20: It measures quantization noise compounding over loop passes in a weight-tied model, and proposes a label-free trajectory metric. MORPH runs a ternary STE core inside a loop, and "the ternary core is a weak per-pass map" is a standing finding. It is the only paper in this batch with a control on cycle count and precision at once.

## 2. What the paper shows

Read in full main text (Sections 1 to 5, Tables 1 to 3). Appendices B to O not read (they hold the per-config tables).

- 2026-10-20 (Table 1, primary protocol, Sudoku n=2000): TRM-MLP FP32 74.6, INT8 74.7, naive per-tensor INT4 6.4, per-channel calibrated INT4 74.5. TRM-Attn Sudoku 73.1 / 72.8 / 68.3 / 72.3. Maze (TRM-Attn) 87.0 / 87.1 / 86.2 / 86.7. The same task with an attention mixer survives naive INT4; the MLP-mixing token mixer has a 20.7x inter-channel range spread vs 11.1x for attention (Fig 4).
- 2026-10-20 (Sec 4.2, Fig 2, Table 9): on ARC-AGI-1 single-pass, one outer cycle (H=1, nsup=8) scores 35.75 FP32 and 35.25 INT8, against 36.00 at full depth H=3, nsup=16. That is 6x fewer FLOPs for -0.75 points. Sudoku keeps rising with depth (best at H=4), so there is no cheap cut there.
- 2026-10-20 (Sec 4.5, Fig 5, App I): carry-trajectory fidelity = cosine between the compressed and FP32 final carry states. Over a 50-configuration grid, Spearman -0.75 with accuracy damage, AUC 1.00 for "lost more than 10 points"; threshold 0.83 set on Sudoku, transferred to Maze with no error. Consecutive-state cosine is nearly flat across precision (change at most 0.064). An L2 divergence is equally predictive.
- 2026-10-20 (Sec 4.6): pruning (25 %), single-cycle distillation (13 to 15 % of parameters) and linear-attention substitution all give 0.0 % puzzle-exact while cell accuracy stays 50 to 87 %. QAT from the calibrated INT4 start (CE or KD, three learning rates) made held-out accuracy worse in every case.
- Failure modes reported: the compressed model keeps local token accuracy and loses global constraint satisfaction; the on-device INT4 result is an analytical weights-only projection (no INT4 run on device); fidelity needs an FP32 reference.

## 3. Our speculation

- 2026-10-20 Speculation: the metric ports directly. For a ternary MORPH core versus its FP32 twin, cosine of the depth-6 hidden state per row, and its trend over depths 1 to 6, would say whether the ternary map drifts off the dense trajectory as passes accumulate. Our norm_match rule was chosen from out/in gain ratios; this gives a second, label-free reading. Control: the dense twin at matched steps. Prediction: if the ternary loop is a weak map, fidelity would be high at depth 1 and fall with depth only if errors compound; flat fidelity would say the weakness is in the per-pass map, not in drift.
- 2026-10-20 Speculation: cell vs puzzle decoupling is the same shape as our token CE vs K1-K6: a model can keep token CE and lose the loop's contribution. It supports reporting K1-K6 beside CE, which we already do.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper claim: INT8 and calibrated INT4 recovery needs no retraining and QAT hurts. Our recipe trains ternary STE from scratch (no PTQ), so this does not apply to training; it only says that post-hoc rounding of a loop is fragile in MLP-mixing and sturdy in attention.
