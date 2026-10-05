# PonderLM (Pretraining Language Models to Ponder in Continuous Space): reading and local cache

Read 2026-10-05.

## Citation (verified)

Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang,
Zhiyu Li, Zhouhan Lin (corresponding). *PonderLM: Pretraining Language Models to Ponder in
Continuous Space*. [arXiv 2505.20674v3](https://arxiv.org/abs/2505.20674). v1 2025-05-27,
v2 2025-06-23, v3 2026-02-20. The arXiv comment field reads "ICLR 2026". Affiliations: LUMIA
Lab (Shanghai Jiao Tong University), Institute for Advanced Algorithms Research (Shanghai),
Shanghai Innovation Institute. Code: <https://github.com/LUMIA-Group/PonderingLM>. Weights:
<https://huggingface.co/zeng123/PonderingPythia-2.8B>.

The brief's title, "Pretraining Language Models to Ponder in Continuous Space", is the v1
title (checked on the v1 abstract page). v3 adds the "PonderLM:" prefix. Same paper.

Read in full: abstract, Sections 1 to 6, Appendices A to H, every table. The figures were
read from their captions and axis values in the extracted text, not measured off the plots.

## Local cache

- [PDF](../../../../ignore/papers/2505.20674v3-ponderlm-pretraining-lms-to-ponder-in-continuous-space.pdf)
- [Extracted text](../../../../ignore/papers/2505.20674v3-ponderlm-pretraining-lms-to-ponder-in-continuous-space.txt)
  (`pdftotext -layout`)
- PDF: 1,166,334 bytes, 22 pages. SHA256
  `ae260174abc91cf87a75244d80d2161489cb974f9e116afee5bb88f14ebd32fe`.
- Text: 84,678 bytes, 1,215 lines. SHA256
  `8af7f6eaf51c624ff4b4ff77a11a8ff6311e7ca9a812856785ebb3879b2eac2d`.

## What it actually does

One token prediction runs the WHOLE language model several times. Between runs, the model's
own output distribution is turned back into an input vector: the probability-weighted sum of
the token embeddings. That vector is ADDED to the original input embedding at the same
position, and the sum is fed to the same model again.

Section 2, with `V = [e_1, ..., e_|V|]` the input embedding matrix and `p` the predicted
distribution at one position:

    t   = sum_i p_i e_i                                   (eq. 3, "pondering embedding")
    T   = P V                                             (eq. 4, all positions at once)
    E_1 = E_0 + T                                         (eq. 5, residual)
    P_1 = LM(E_1)                                         (eq. 6)

    E_0,  P_0 = LM(E_0),  T_1 = P_0 V
    E_s = E_0 + sum_{i=1..s} T_i,   P_s = LM(E_s)         (eq. 7)

The cross-entropy loss is computed on `P_s` only. The intermediate distributions
`P_0 ... P_{s-1}` get no loss of their own: they are trained only through their effect on
`P_s`. There is no chain-of-thought data, no teacher, no RL and no curriculum (Table 3:
"General corpus ... Weight Sum of Embeddings ... Per token ... Pretrain"). Every position
ponders in parallel, so each pondering step is one more full forward over the sequence.

Implementation details that matter:

- Top-K mixture. Footnote 1: only the `K` highest-probability tokens enter the sum, `K = 100`
  (§4.4: `K = 10` gives perplexity 15.18, `K = 100` gives 14.21, the full vocabulary 14.20).
- `s = 3` steps by default (footnote 2).
- The pseudocode (Figure 2) uses one `nn.Parameter` embedding table both for the input and
  for the mixture. In the scaling runs the INPUT and OUTPUT embeddings are UNTIED
  (Appendix B). So the mixture uses the input table, weighted by the output head's softmax.
- The KL between consecutive steps' output distributions starts near 0.35 and falls
  monotonically; the cosine between consecutive `E_s` starts near 0.88 and approaches 1.0
  (Appendix F, Pythia-70M at 10 steps).

## The key numbers

Small-scale scaling (§3.1, GPT-2 and LLaMA, 405M to 1.4B, Pile subset, Chinchilla token
counts, context 2048): an 834M pondering model matches a vanilla model with "over 2x the
parameter-token product". Figure 4 labels 2.01x (GPT) and 2.26x (LLaMA); Figure 9 (LLaMA at
2, 3, 4 steps) labels 2.04x, 2.26x, 2.44x.

Pythia on the full 300B-token Pile, same architecture, tokenizer and hyperparameters as the
official suite (§3.2): the fitted curves put a 2.55B PonderPythia at Pythia-6.9B's loss (63 %
fewer parameters), and PonderPythia-1B at the baseline's final loss with 59 % fewer tokens.
At matched training FLOPs (70M; the vanilla model trained 4 epochs to match) PonderPythia
stays below.

Downstream averages, Table 1 (9 tasks):

    model              5-shot avg   0-shot avg
    Pythia-1B             50.4         50.6
    PonderPythia-1B       56.2         55.0
    Pythia-2.8B           57.6         57.3
    PonderPythia-2.8B     61.5         60.4
    Pythia-6.9B           60.1         59.1
    Pythia-12B            61.7         60.5

The ablation that matters to us, Table 2 (Pythia-70M, 30B tokens, Pile perplexity, lower is
better):

    variant                          3 extra steps   6 extra steps
    Pythia-70M baseline                  16.95            -
    Looped (whole stack reused)          15.33          15.18
    Pause tokens                         16.53          16.55
    Last hidden state as embedding       15.64          15.30      (Coconut-style)
    + linear projector                   15.64          15.29
    PonderPythia                         14.16          13.56

Same extra compute, same training signal (plain CE). Feeding the hidden state back is worse
than feeding back the embedding mixture, at 3 and at 6 steps.

Steps: 0, 1, 2, 3, 4, 5, 10 steps lower the validation loss monotonically (Figure 7 top). A
model trained with the step count drawn uniformly from 1 to 10 per batch gets better as the
step count rises at inference (Figure 7 bottom).

Continual pretraining of the official Pythia-1B on 30B tokens (Appendix D): average 51.5
against 50.7 (5-shot) and 51.5 against 50.8 (0-shot). Pretraining cost (Appendix A): 19,886,
31,680 and 109,570 GPU hours for 1B, 1.4B and 2.8B.

## How it maps onto MORPH

- **It is trace-free.** The only signal is next-token CE on the last step. This is the
  closest published case of a latent carrier that learns from CE alone, and it works where
  Coconut without its curriculum does not (Coconut reports its "w/o curriculum" variant no
  better than no-CoT; CODI reports its student-only task gains "only marginally").
- **The carrier is in the reader's input language.** That is the difference Table 2
  isolates: same loop, same loss, embedding mixture beats hidden state. MORPH's LM head is
  weight-tied to the input table (`lm_weight()` is the embedding table), so
  `softmax(c E^T / tau) E` is native to both the loop and the coda. PonderLM itself ran
  untied (Appendix B); the tied case is not tested in the paper.
- **The mixture is a NEXT-token distribution, refined in place.** PonderLM ponders at every
  position and adds the result to that position's own input. MORPH's need is different: the
  coda of a LATER span needs the CURRENT span's exact tokens (65 % of LXTUL's gap is bigram
  repeats 33 to 256 tokens back,
  [exact-recall probe](../../../../lab/experiments/successes/2026-10-04-exact-recall-gap-probe.md)).
  A mixture peaked on one of the span's own tokens is that token's embedding exactly, so the
  extractive cell (A3) is the vertex of the pondering mixture (A2). The Lean development
  [`lab/theory/tul_pseudotoken/`](../../../../lab/theory/tul_pseudotoken/) proves the parts of
  this that decide a design: one carrier position holds at most one exact token, a linear
  write of one cell cannot place two exact arbitrary tokens, a sharp softmax over the tied
  table can.
- **`E_s = E_0 + sum T_i` is an accumulating re-injection.** In
  [`tul_exploration`](../../../../lab/theory/tul_exploration/)'s terms (T5, `Washout.lean`) a
  code added every pass is not washed out by a contractive loop. PonderLM's steps add the
  current guess every step.
- **Cost.** Every step is a full forward over every position, like MORPH's paid loop. An
  LXTUL arm uses the mixture only on the slot channel, so it pays for N emissions per span,
  not for a re-run of the token path.

## What it does NOT establish for us

- Nothing about carrying exact tokens across spans. Its mixture refines the prediction at the
  same position. No experiment separates copy content from computed content, and no result is
  broken down by repetition or distance.
- Nothing about loop contribution in MORPH's sense. Its "more steps help" curve is a model
  trained at each step count (or a randomised one), not a truncation of one trained loop with
  a CI.
- Nothing about tied embeddings: the scaling runs untie them.
- Nothing about a strict geometry: the pondering embedding enters the same position's input,
  so every later position sees it through ordinary attention.
- No variance across seeds is reported for any number above.
