# Paper note: Explorative Modeling (XM)

- **Title:** Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation (Gladstone, Ji, Du)
- **arXiv or URL:** arXiv 2607.27372, https://arxiv.org/pdf/2607.27372. Code: github.com/alexiglad/XM
- **Local paper text:** docs/references/tul-latent-emission/explorative-modeling/explorative-modeling.md
- **Date first noted:** 2026-10-16

## 1. Why we reference it

- 2026-10-16: XM trains on the best of K candidate matches between model outputs and data. This is best-of-K training, which is close to our LXTUL winner-take-all cell fan.
- 2026-10-16: Our LXTUL cell fan is a forward XM instance (the winner cell is chosen and trained on).
- 2026-10-16: A span-level plan is a candidate place for best-of-K over span candidates.

## 2. What the paper shows

Read on 2026-10-16: the abstract and section 7 (Limitations and Conclusion) only. The body was not read. Section 7 is the only language statement we read.

- 2026-10-16 (section 7, language): Autoregressive language models were the hardest case. The paper found them harder to improve with exploration, "likely because injecting a latent into them is less natural." It reports "initial modest results showing improved data efficiency" for autoregressive LLMs.
- 2026-10-16 (section 7, integration): Exploration needs a latent variable to search over. Continuous models benefited most. Masked diffusion models benefited once given a learned latent embedding.
- 2026-10-16 (abstract, claims not checked in the body): FLOP efficiency 4.1x, sample efficiency 6.2x, parameter efficiency 47 percent, and FID 1.43 on ImageNet without guidance. These are image results from the abstract. We did not check them in the body.

## 3. Our speculation

- 2026-10-16 Speculation: Best-of-K over span candidates could give a span-level plan that the reader does not have to decode token by token. Untested.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None recorded yet.
