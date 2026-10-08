# Pointer Sentinel Mixture Models

- **Authors:** Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher (MetaMind / Salesforce)
- **Year:** 2016 (arXiv v1, 26 Sep 2016)
- **Venue:** ICLR 2017 (NOT verified by fetch: the PDF text and arXiv metadata do not state a venue; OpenReview blocked the fetch. Taken from memory.)
- **arXiv:** https://arxiv.org/abs/1609.07843
- **PDF:** [pointer-sentinel.pdf](pointer-sentinel.pdf)

## Summary

A language model mixes a normal vocabulary softmax with a pointer over the last L tokens.
A learned sentinel makes the mixing weight come from the same softmax as the pointer.

- **Pointer scores.** Query `q = tanh(W h_{N-1} + b)` (Eq. 2), `z_i = q^T h_i` (Eq. 3),
  `a = softmax(z)` (Eq. 4). Mass on a word is the sum over its positions in the window (Eq. 5).
- **Mixture (Eq. 6).** `p(y_i | x_i) = g * p_vocab(y_i | x_i) + (1 - g) * p_ptr(y_i | x_i)`.
  g = 0 means pointer only, g = 1 means vocabulary softmax only.
- **Sentinel.** A learned vector s in R^H is appended to the pointer scores:
  `a = softmax([z ; q^T s])` (Eq. 7). a now has V+1 entries (the paper's V is the window length).
  The gate is the attention mass on that last slot: `g = a[V+1]`.
- **Renormalization (Eq. 8).** `p_ptr(y_i | x_i) = a[1:V] / (1 - g)`.
- **Effect.** The pointer and the vocabulary softmax compete in one normalization. The gate
  sees both the RNN state and the window states. The authors call this competition crucial to
  their best model.
- **Sharing.** Parameters of the softmax-RNN and the pointer are shared; the pointer also
  supervises the RNN.
- **Pointer-sum attention (Eq. 5)** lets the loss reach zero when mass sits on any occurrence
  of the target word.
