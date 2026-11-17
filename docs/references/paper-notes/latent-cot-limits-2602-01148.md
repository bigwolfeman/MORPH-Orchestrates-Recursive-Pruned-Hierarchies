# Paper note: Capabilities and Fundamental Limits of Latent Chain-of-Thought

- **Title:** Capabilities and Fundamental Limits of Latent Chain-of-Thought (Zou, Xiong, Liu; 2026-02-01)
- **arXiv or URL:** 2602.01148
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Wolfe's summary: explicit tokens act as an error-correcting step, and latent error grows with step count. We need the exact theorem to judge a loop with tokens decoded between spans.

## 2. What the paper shows

Read 2026-11-17 via arXiv HTML (first 100k of 144k characters).

- 2026-11-17 (Theorem 4.8): with error E_M = h_M - h_M*, transition f_theta with Lipschitz constant L_F, and i.i.d. Gaussian noise N(0, sigma^2 I_d) added each step, E||E_M||^2 = (1 - L_F^(2M)) / (1 - L_F^2) * d * sigma^2. It grows with the step count M. For L_F > 1 it is exponential. For L_F <= 1 it still accumulates (bounded by M d sigma^2).
- 2026-11-17 (Theorem 4.7): argmax decoding acts as an error-correcting filter in explicit CoT, under Assumption 4.2 (kappa-concentrated Dirichlet step distribution) and Definition 4.6 (sub-decisional perturbation: the noise does not change the argmax).
- 2026-11-17 (other assumptions): A.3 latent states converge to a fixed point or compact attracting set (supported by a PCA plot only); Lemma A.4 non-degenerate output, max p <= 1 - delta.
- 2026-11-17 (Symbolic Index, Def 4.9): I_S = max over valid tokens of p(u|h,x). Thm 4.11: logit margin >= log(I_S/(1-I_S)). Thm 4.12 (exploration-execution trade-off): KL lower bound log B + I_S log I_S + (1-I_S) log((1-I_S)/(B-1)).
- 2026-11-17 (curriculum, Thm 5.1, 5.2): direct training on a biased distribution caps success below the expert for any data size; curriculum reaches it with rate O(sqrt((d log n + log 1/delta)/n)). Bounded valuation assumed.
- 2026-11-17 (experiments): GPT-2 124M, Coconut curriculum, 6 latent steps. GSM8K: CoT 42.9, Coconut 34.1, no curriculum 14.4. ProsQA: CoT 77.5, Coconut 97.0, no curriculum 76.1. ProntoQA Coconut 99.8, no curriculum 52.4. Some figures come from Hao et al. 2024. The theorems are not directly tested.
- 2026-11-17: the paper does not analyse decoding between latent steps. Argmax is analysed for explicit CoT only.

## 3. Our speculation

- 2026-11-17 Speculation: for a loop that runs T passes on one state and then decodes tokens, the bound applies inside the T passes only. With T <= 8 and L_F <= 1 the error is at most T d sigma^2. If the next span's state is rebuilt from decoded tokens, the argmax reset happens at span boundaries, as in CoT. It does not apply between passes. Whether the reset acts depends on the next state reading tokens, not the previous state.
- 2026-11-17 Speculation: the noise model (i.i.d. Gaussian per step) is an assumption, not measured on LMs.

## 4. Corrections

- 2026-11-17: the abstract headline numbers (ProsQA 97.0, GSM8K 34.1) are Coconut numbers partly quoted from prior work.

## 5. Where our findings disagree

- Item 5 agrees in spirit (a latent without a discrete step loses information). Item 2 is not touched.
