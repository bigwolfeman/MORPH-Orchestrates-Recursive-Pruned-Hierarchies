# Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation (DITTO)

- **Authors:** Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, Jian Li
- **Year:** 2022
- **Venue:** NeurIPS 2022 (36th Conference on Neural Information Processing Systems)
- **arXiv:** https://arxiv.org/abs/2206.02369
- **PDF:** [ditto.pdf](ditto.pdf)
- **MORPH / TUL uses:** the DITTO fine-tuning phase (`lxtul_pointer_ditto.yaml`).

## Summary

DITTO is PseuDo-RepetITion PenalizaTiOn. The authors find that a model assigns higher
probability to a token each time a sentence repeats (self-reinforcement). DITTO trains that
effect away on purpose.

- **Data.** Pick a random sentence s from the corpus and repeat it N+1 times until the
  sequence reaches the model's maximum length (1,536 tokens for open-ended generation). The
  previous context of the sentence is added as a prefix, which they find more stable.
- **Loss.** For the l-th token in the n-th repetition (Eq. 1):
  `L_DITTO = -log(1 - | P_theta(x_{n,l} | x_{<n,l}) - lambda * P*_theta(x_{n-1,l} | x_{<n-1,l}) |)`.
  P* is the same probability one repetition earlier, detached from the gradient.
- **Meaning of lambda.** The loss is minimal when P(n-th repeat) = lambda * P(previous repeat).
  lambda = 1 holds the probability flat. lambda < 1 makes it decay geometrically per repeat.
- **lambda used.** 0.5 by default (Wikitext-103 open-ended generation). 0.9 for the
  summarization task (Figure 5 discussion).
- **Mixing.** DITTO is applied as fine-tuning of an MLE-trained baseline: "equally mixing the
  sentence-level repetition penalization update and normal MLE loss update". Fine-tuning ran
  10k steps after 150k MLE steps.
- **Headline (Table 1, Wikitext-103 test, greedy decoding, 750M Transformer, 3 seeds).**
  MLE: PPL 25.68, Repetition-4 44.20 %, Repetition-Sen 14.50 %, MAUVE 0.34.
  DITTO: PPL 24.33, Repetition-4 22.00 %, Repetition-Sen 2.85 %, MAUVE 0.77.
  Human text: Repetition-4 1.10 %, Repetition-Sen 0.01 %. Perplexity falls, it does not rise.
- **Variants in the appendix.** DITTO-mse (Eq. 3) and DITTO-margin (Eq. 4); the paper says
  they do not beat Eq. 1.
- **Caveat for MORPH.** The paper does not penalize repetition that has a useful prefix
  source. They argue useful repeats act as positive samples, so the model learns to tell
  useful from useless repetition.
