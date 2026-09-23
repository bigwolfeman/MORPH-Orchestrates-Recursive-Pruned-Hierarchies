# ReGuLaR (variational latent reasoning guided by rendered CoT): reading note

Read 2026-09-23, for the LXTUL-G build
([`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../../../.agents/notes/proposed/architecture/2026-09-23-lxtul-gram-stochastic-loop.md)).

## Citation (verified)

Fanmeng Wang, Haotian Liu, Guojiang Zhao, Hongteng Xu, Zhifeng Gao. *ReGuLaR: Variational
Latent Reasoning Guided by Rendered Chain-of-Thought*.
[arXiv 2601.23184v1](https://arxiv.org/abs/2601.23184), 2026-01-30 (PDF header dated
2026-02-02). Renmin University of China and DP Technology. No venue claimed.
Code: <https://github.com/FanmengWang/ReGuLaR>.

Read in full: all 22 pages, main text and Appendices A to C, including the training and
inference algorithms and every ablation table.

## Local cache

Not cached into `ignore/papers/` in this pass (the brief allowed only the note files).
Fetched from `https://arxiv.org/pdf/2601.23184` on 2026-09-23: 3,121,237 bytes, 22 pages,
SHA256 `a1e87ce35ef6cbb8fcaa37e31697db75a61511ff97866e14bfac2791edfd41b0`. Suggested cache
name: `ignore/papers/2601.23184v1-regular-rendered-cot.pdf`.

## Read this first: the names are swapped

ReGuLaR calls the model's own context-only latent distribution the "posterior" and the
target-conditioned distribution the "prior". In LXTUL-G's terms (and in GRAM's):

    ReGuLaR "posterior"  p_phi(z_k | Q, Z_<k)   context only        = our PRIOR
    ReGuLaR "prior"      p_gamma(z_k | R_k)     sees the k-th CoT    = our POSTERIOR (target side)
                                                segment

Every quote below uses their words; every recommendation uses ours.

## What it actually does

A pretrained LLM (LLaMA-3.2-1B-Instruct by default, frozen, LoRA r = 128, alpha = 32) replaces
its chain of thought `R` with K latent states `z_1..z_K`, K much smaller than the CoT length.

**The context-only latent.** A latent-reasoning head (an MLP on the last hidden state)
outputs `mu_k` and `log sigma_k` from the question and the earlier latents; the model
samples `z_k = mu_k + sigma_k * eps`, `eps ~ N(0, I)`, and feeds `z_k` back as the next input
embedding.

**The target side.** The CoT is cut into K segments (one per sentence in the main table).
Each segment is rendered to an image (the Glyph rendering config), encoded by the FROZEN
DeepSeek-OCR vision encoder (Tiny mode, 64 visual tokens mean-pooled to one vector of
1280 dims), and mapped by a trainable MLP adapter `g_gamma` to the LLM width (2048):

    z^_k = g_gamma( v( render(R_k) ) ),   p_gamma(z_k | R_k) = N( z^_k, I )

The variance is fixed at the identity. The encodings are precomputed offline.

**The objective (Eq. 7).**

    max  sum_i E_{Z ~ p_phi}[ log p(a_i | Q, Z, A_<i) ]                answer loss
       + sum_k sum_{r_j in R_k} E_{z_k}[ log p_psi(r_j | z_k) ]         per-state reconstruction
       - sum_k KL( p_phi(z_k | Q, Z_<k) || N(z^_k, I) )                 regulariser

- The answer decoder conditions on the question and the latents only. The CoT tokens are
  never in its context.
- The reconstruction term decodes ONE random token of segment `R_k` from `z_k` alone,
  through the language head. It is a bag-of-tokens loss per latent (Algorithm 1, line 23).
- The KL is taken in closed form (Eq. 9) but trained with the sampled surrogate of Eq. 10:

      (1/2) E_eps[ || mu_k + sigma_k * eps - z^_k ||^2 ] - log |diag(sigma_k)|

  which is a regression of the sampled latent onto the target encoding plus a term that
  holds sigma near 1. The KL weight is 1. No beta, no free bits, no annealing.

**Why the decoder uses the latent.** There is no other route. The CoT is removed from the
context, so everything the answer needs beyond the question must come through `Z`. The
paper does not frame this as collapse prevention, and it never discusses posterior
collapse; the architecture simply has no bypass.

**Inference (Algorithm 2).** Sample `z_k` from the context-only head, decode one token from
it with the language head, stop at an end-of-reasoning token, then decode the answer with
nucleus sampling (top-p 0.9, temperature 1.0). Five seeds per evaluation.

Training: AdamW, lr 1e-4, weight decay 0.01, constant schedule after 1000 warmup steps,
8x A100.

## The key numbers

Table 1, LLaMA-1B, accuracy (%) with reasoning length (latent steps):

                GSM8K-Aug     GSM-Hard      SVAMP         MultiArith
    Coconut     20.5 (6.00)   4.86 (6.00)   39.8 (6.00)   41.4 (6.00)
    CoLaR       26.6 (5.63)   6.23 (7.01)   47.1 (2.96)   87.0 (3.23)
    ReGuLaR     34.9 (3.69)   8.27 (4.12)   50.1 (2.02)   89.2 (2.28)

Average 45.6 % at 3.03 steps against CoLaR's 4.70 steps.

The learning-paradigm ablation (Table 8, average accuracy over the four sets) is the
number that matters most here:

    answer loss only                         12.8
    + reconstruction, no KL                  13.1
    + KL, no reconstruction                  41.9
    + KL + reconstruction (ReGuLaR)          45.6

Without the target-side KL, answer supervision alone trains a latent chain worth almost
nothing. Reconstruction without the KL does not rescue it.

Probabilistic against deterministic latents (Table 9): 45.6 against 44.2. The authors
attribute the drop to "mean collapse": a deterministic predictor outputs the average of
several valid next steps.

Dense against pooled target encoding (Table 10): rendered-image anchor 45.6, mean of the
segment's token embeddings 42.3.

Extreme compression (Table 2), one latent for the whole CoT: MATH average 11.9 % against
CoLaR 7.76 % at 62.2 steps; AQUA-RAT 39.8 against 31.2.

Robustness (Appendix B): font size 9 to 20 pt moves the average 44.5 to 45.6; DPI 72 to
300 moves it 45.2 to 45.6; the Tiny encoder mode matches Large.

## What LXTUL-G should take

- **One latent per segment, anchored by a target-side encoding of THAT segment.** Their
  segment is a CoT sentence; ours is the span the slot precedes. The structure is the
  same, and Table 8 says the target-side KL is the load-bearing part (12.8 to 41.9).
- **A dense span encoding beats a mean of token embeddings** (42.3 against 45.6). For
  LXTUL-G the posterior's view of the next span should be an attention pool over the
  prelude's states of those tokens, not a bag-mean of embeddings. MORPH's own warning
  applies on the other side: LCTUL's frozen encoder E pooled a live prelude and its codes
  collapsed (own cosine 0.61 to 0.99 by step 3000; the `tul.code_target_ref` entry in the
  root `CLAUDE.md`), so the span encoding must be trained with the ELBO or read from a
  frozen reference copy.
- **Keep the latent stochastic.** Table 9 is a second source, after GRAM's "guide only"
  row, that a deterministic latent pays for mean collapse.
- **The per-latent reconstruction term as an auxiliary.** Decoding tokens of the span from
  the latent alone, with no token path, forces the latent to carry the span. In LXTUL-G it
  would be a small head on the exit state predicting a random token of the next span. It
  is training-only and cheap. It is also the "regress onto the slot" family that MORPH
  treats with suspicion, so it belongs in a companion arm, not in the first build.
- **Fixed-variance target, if the posterior leaks.** A posterior with unit variance around
  an encoding cannot pass more than a bounded amount of information per channel. That is a
  blunt but real cap on the leak, and a fallback if a learned posterior variance shrinks
  to zero.

## What does not transfer

- **The bypass.** The answer decoder cannot see the CoT, so the latent is the only road.
  MORPH's coda reads every token of the span causally. The mechanism that makes their
  decoder use the latent does not exist in MORPH.
- **The KL direction and form.** Their KL pulls the context-only distribution toward a
  fixed-variance target anchor, and in practice it is a regression plus a sigma term. It is
  not a VAE KL between a posterior and a learned prior over the same step. The "never
  regress onto the slot state" lesson (LCTUL code target met in one pass) applies to this
  form directly.
- **The rendered image encoder.** The vision path is a compression trick for CoT text. Its
  advantage over pooled embeddings (3.3 points) comes from a large pretrained OCR encoder;
  MORPH has no such encoder, and the brief rules out a frozen one on a live front.
- **Scale and training regime.** A frozen 1B to 8B pretrained LLM with LoRA, supervised
  CoT on GSM8K-style data. No loop, no depth axis, no from-scratch training.
- **No exploration claim.** Samples at inference come from the context-only head, but the
  paper never measures diversity, coverage or sample selection. Its "probabilistic beats
  deterministic" is a 1.4-point single-sample result.
