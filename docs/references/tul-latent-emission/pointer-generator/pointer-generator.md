# Get To The Point: Summarization with Pointer-Generator Networks

- **Authors:** Abigail See, Peter J. Liu, Christopher D. Manning
- **Year:** 2017
- **Venue:** ACL 2017 (aclanthology.org/P17-1099; page title fetched and matched)
- **arXiv:** https://arxiv.org/abs/1704.04368
- **PDF:** [pointer-generator.pdf](pointer-generator.pdf)

## Summary

An attentional seq2seq summarizer that can copy source words by pointing and generate new
words from the vocabulary, plus a coverage mechanism against repetition.

- **Attention.** `e_i^t = v^T tanh(W_h h_i + W_s s_t + b_attn)` (Eq. 1), `a^t = softmax(e^t)` (Eq. 2).
- **Generation probability (Eq. 8).** `p_gen = sigma(w_h*^T h*_t + w_s^T s_t + w_x^T x_t + b_ptr)`.
- **Final distribution (Eq. 9).**
  `P(w) = p_gen * P_vocab(w) + (1 - p_gen) * sum_{i: w_i = w} a_i^t`.
  An out-of-vocabulary word has P_vocab = 0, so it can only be copied. A word absent from the
  source has zero copy mass.
- **Coverage vector (Eq. 10).** `c^t = sum_{t'=0}^{t-1} a^{t'}`. It feeds the attention
  (Eq. 11): `e_i^t = v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)`.
- **Coverage loss, composite (Eq. 13).**
  `loss_t = -log P(w*_t) + lambda * sum_i min(a_i^t, c_i^t)`. It penalizes only the overlap
  between the current attention and the coverage so far.
- **Training schedule.** lambda = 1. Coverage was NOT trained from the start. They trained the
  pointer-generator first, then "added the coverage mechanism with coverage loss weighted to
  lambda = 1 ... and trained for a further 3000 iterations (about 2 hours)"; the coverage loss
  fell from about 0.5 to 0.2. lambda = 2 was also tried: it lowered the coverage loss but raised the primary loss, so it was not used.
- **Cost.** The pointer adds 1153 parameters, coverage adds 512.
