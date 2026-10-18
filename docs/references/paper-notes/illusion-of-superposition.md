# Paper note: The Illusion of Superposition?

- **Title:** The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models (Rizvi-Martel, Rabusseau, Mosbach; COLM 2026)
- **arXiv or URL:** 2604.06374v2 (v1 2026-04-07, v2 2026-08-03)
- **Local paper text:** reading file docs/references/looping-depth/latent-exploration/illusion-of-superposition/illusion-of-superposition.md; PDF and extracted text only in ignore/papers/2604.06374v2-illusion-of-superposition.{pdf,txt} (private)
- **Date first noted:** 2026-10-18 (paper first read 2026-09-18)

## 1. Why we reference it

- 2026-10-18: It is the negative reading of one-vector superposition. Training-free soft
  tokens collapse in the first layers, fine-tuned Coconut models never use the latent, and
  only small from-scratch models on a graph task hold a set. The Thought Register arm was
  read against it on 2026-09-13.
- 2026-10-18: For the coarse-level particle design (K copies of the coarse loop from random
  starts, read by a teacher-forced AR coda as a set; Lean claims in
  `lab/theory/tul_exploration/TulExploration/Particles.lean`), it says when ONE vector fails
  to hold candidates. That is the case where K separate particles would be the alternative.
  Its width result also sets how wide a single coarse state must be to hold a set at all.

## 2. What the paper shows

Numbers below come from the 2026-09-18 reading file. Spot-checked 2026-10-18 against
`ignore/papers/2604.06374v2-illusion-of-superposition.txt`: Table 1 (both cosines and both
mixing entropies), Table 2 (all four fine-tuned rows and all four from-scratch rows), and
Table 3 (the 2L-768d and 12L-320d rows). All match.

- 2026-10-18 (Section 4.1, Figure 2): QwQ-32B on MATH500 (N = 500). Logit-lens entropy
  across layers is the same for Soft Thinking and discrete CoT. KL(soft, discrete) falls to
  about 1e-4 in the middle layers.
- 2026-10-18 (Section 4.2, Table 1): cosine between the soft-token pass and the argmax pass
  is 0.998 +- 0.013 (Qwen2-1.5B) and 0.996 +- 0.025 (QwQ-32B). Mixing-weight entropy is
  0.10 +- 0.26 and 0.18 +- 0.34 nats: the input mix was near-discrete already.
- 2026-10-18 (Section 5.1, Table 2a, Figure 3): fine-tuned Coconut on ProsQA, accuracy with
  latents / with latents removed: GPT-2 124M 99.0 / 99.0, SmolLM2-135M 93.3 / 92.3,
  SmolLM2-360M 98.7 / 98.0, SmolLM2-1.7B 100.0 / 100.0. Entity probing shows the target
  dominant from step 0. The model solves the task in one pass and the latents are unused.
- 2026-10-18 (Section 5.2, Table 2b, Figure 4): from scratch, 8 heads, 768 wide, with /
  without latents: 2 layers 94.5 / 13.8, 4 layers 96.2 / 16.0, 8 layers 95.2 / 53.0,
  12 layers 91.2 / 34.1. Here the latents are used and the belief evolves across steps.
- 2026-10-18 (Section 6, Table 3): parameter-matched at about 15M, 3 seeds: 2L-768d
  96.0 +- 0.9 / 30.9 +- 8.4, 4L-544d 91.8 +- 4.9 / 39.7 +- 20.8, 8L-384d 81.6 +- 4.3 /
  73.2 +- 1.4, 12L-320d 72.4 +- 7.1 / 70.0 +- 6.3. Narrow-deep models fall back to the
  no-latent shortcut.
- 2026-10-18 (Section 6, Table 4, Table 11): below the critical width (d <= 64), adding depth
  up to 12 layers gives at best 68.3 % with latents and 65.4 % without.
- 2026-10-18 (Section 6, Figure 5): uniform k-token mixes in Pythia-1B keep near-maximum
  entropy through a random-init model and collapse at the final layers of the pretrained
  one. Across pretraining checkpoints the collapse appears gradually.
- 2026-10-18 (Section 7): the authors ask whether token-level superposition is desirable at
  all; many Soft Thinking mixes combine unrelated tokens such as "(" and "{". They suggest
  meaningful exploration "likely requires superposition at a higher level of abstraction
  (over entire reasoning strategies, not individual tokens)". This is their speculation.

## 3. Our speculation

- 2026-10-18 Speculation: the one positive regime (from scratch, Section 5.2) is the graph
  task, whose per-step computation is item-separable (see the Reasoning by Superposition
  note, section 3). The negative regimes run the full nonlinear computation of a pretrained
  LM on a mixed input. That pattern fits the candidate criterion (one vector works when the
  step acts per item; separate particles are needed when branches need their own nonlinear
  computation). The paper never runs K separate states, so it is silent on whether
  particles fix the negative regime.
- 2026-10-18 Speculation: part of the depth effect in Tables 2b and 3 may be training
  budget. The Emergence paper's Table 4 has a 12-layer, 768-wide model at 67.4 % on the
  default budget and 99.6 % with 400 stage-1 epochs at LR 5e-5 (Emergence Appendix F.1).
  We did not check whether Illusion held the budget fixed across depths.
- 2026-10-18 Speculation: Section 6 blames next-token pretraining for final-layer
  commitment. A coarse particle state is not itself a next-token head; only the coda is. If
  that cause is the main one, it acts on the coda's read of the particles, not on the
  particles.

## 4. Corrections

- 2026-10-18: The spot checks found no number in the 2026-09-18 reading that disagrees with
  the paper text. The reading gave no table number for the fine-tuned and depth tables; they
  are Table 2a and Table 2b. The critical-width table it mentions is Table 11.

## 5. Where our findings disagree

- 2026-10-18: Our measurement: the Thought Register (`tul.slot_cells: 4`), trained from
  scratch with the slot loop, read rank 1.2445 of 4 at cosine 0.9432, and distinct against
  same init moved it 0.008. Paper's claim: from-scratch models do use superposition
  (Section 5.2, Table 2b); the fine-tuned ones do not (Section 5.1). Our from-scratch arm
  sits with their fine-tuned row, so from-scratch training is not the axis that separates
  us from their positive. The tasks differ (web text against ProsQA). Source:
  `lab/experiments/failures/2026-09-13-arc-thought-register.md` (measured 2026-09-13).
- 2026-10-18: Our measurement: on LCTUL arm A at step 20k the frozen twin coda preferred NO
  cell to the loop's own cell by 4.6413 nats (`zero - own`, twin-read probe, 96 rows). Paper's claim: deleting the latents
  costs at most 1.0 point (Table 2a). Both say the reader does not use the latent; ours is a
  sign flip, and a frozen reader is a confound they do not have. Source:
  `lab/experiments/failures/2026-09-17-lctul-code-only-ref.md` (measured 2026-09-17).
