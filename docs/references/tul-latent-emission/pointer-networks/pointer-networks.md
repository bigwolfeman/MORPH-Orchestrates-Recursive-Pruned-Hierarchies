# Pointer Networks

- **Authors:** Oriol Vinyals, Meire Fortunato, Navdeep Jaitly
- **Year:** 2015 (arXiv v1 9 Jun 2015; stored PDF is v2, 2 Jan 2017)
- **Venue:** NIPS 2015 (papers.nips.cc page titled "Pointer Networks" was fetched; the PDF itself carries no venue line)
- **arXiv:** https://arxiv.org/abs/1506.03134
- **PDF:** [pointer-networks.pdf](pointer-networks.pdf)

## Summary

An output sequence whose elements are positions in the input (convex hull, Delaunay
triangulation, travelling salesman). The output vocabulary changes with input length, so a
fixed softmax cannot do it. The fix is to use the attention distribution itself as the output.

- **Factorization (Eq. 1).** `p(C^P | P; theta) = prod_i p_theta(C_i | C_1..C_{i-1}, P; theta)`.
- **Pointer attention (Eq. 3).** With encoder states e_j and decoder state d_i:
  `u_j^i = v^T tanh(W1 e_j + W2 d_i)`, `a_j^i = softmax(u_j^i)` for j in 1..n.
  Standard attention (Section 2.2) then forms `d'_i = sum_j a_j^i e_j` as a context vector.
  The Ptr-Net (Section 2.3) skips that: `p(C_i | C_1..C_{i-1}, P) = softmax(u^i)`, "an output
  distribution over the dictionary of inputs". This line is unnumbered in the PDF. The
  encoder state is not blended into the decoder; u^i are used directly as pointers.
- **Cost.** n operations per output step.
- **Limit for language use.** It can only output input positions, which is why later work
  (Pointer Sentinel, Pointer-Generator) mixes it with a vocabulary softmax.
