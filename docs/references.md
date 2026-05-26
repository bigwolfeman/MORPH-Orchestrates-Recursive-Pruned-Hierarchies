# MORPH Architecture — Paper References

**MORPH** (Orchestrates Recursive Pruned Hierarchies) is a production research model combining
Parcae-style looped transformers, Block-ELL structured sparsity, CCA+CSA+HCA triple-axis
attention compression, neural memory with SSM outer loop, mHC multi-channel residual streams,
STP geodesic regularization, LeJEPA z-latent prediction, hybrid hyperbolic/Euclidean embeddings,
STE ternary shadow weights, and ReMoE product-key routing.

Papers are grouped by architectural component. "Original work" entries are techniques developed
within the MORPH project with no external precedent.

---

## 1. Looping & Depth

### Parcae — Stable Looped Transformer
**Title:** Parcae: Scaling Laws For Stable Looped Language Models  
**Authors:** Sandy Huang, Mihir Kale, Trevor Gale, et al. (UCSD + Together AI)  
**Year:** 2026  
**arXiv:** [2604.12946](https://arxiv.org/abs/2604.12946)  
**MORPH uses:** The negative-diagonal injection parameterization that guarantees spectral radius
ρ(Ā) < 1 (via zero-order-hold / Euler discretization), enabling stable arbitrary-depth looping.
Also the per-sequence Poisson depth sampling during training, which stochastically varies the
number of loop iterations per batch to further reduce loss spikes.

### Poisson Depth Sampling
Documented within the Parcae paper (arXiv:2604.12946, §3.2). Parcae modifies training to sample
loop depth from a Poisson distribution independently per sequence in a batch, making the model
robust to variable iteration counts at inference. MORPH adopts this directly for the inner core
loop (T drawn per batch from Poisson(μ=8)).

---

## 2. Attention

### CCA — Compressed Convolutional Attention
**Title:** Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space  
**Authors:** Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, Beren Millidge (Zyphra)  
**Year:** 2025  
**arXiv:** [2510.04476](https://arxiv.org/abs/2510.04476)  
**MORPH uses:** Channel-dimension compression (down-project Q/K/V into a shared latent space
of size E/C), causal convolution over K, QK-mean pooling, v-shift (value shift: blend current
and t−1 value projection), learnable temperature, QK-RMSNorm, and CoPE Clipped RoPE inside the
compressed space — all operating before the attention softmax, simultaneously reducing KV-cache,
FLOPs, and parameter count by the compression factor C.

### CSA / HCA — Compressed Sparse & Heavily Compressed Attention
**Title:** DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence  
**Authors:** DeepSeek-AI  
**Year:** 2026  
**Report:** [DeepSeek-V4-Pro on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)  
*(No arXiv ID assigned; PDF hosted on HuggingFace model card, released April 24 2026)*  
**MORPH uses:** CSA (even layers): two-stream gated pooling every m=4 tokens → Lightning Indexer
top-k sparse global attention with −∞ causal masking before ReLU and a gathered validity mask.
HCA (odd layers): aggressive single-stream pooling every m=128 tokens → dense attention over the
compressed stream for broad global context. MORPH alternates CSA/HCA by layer index, following
the V4 interleaving pattern.

### XSA — Exclusive Self-Attention
**Title:** Exclusive Self Attention  
**Authors:** Shuangfei Zhai (Apple Machine Learning Research)  
**Year:** 2026  
**arXiv:** [2603.09078](https://arxiv.org/abs/2603.09078)  
**MORPH uses:** The two-line modification that excludes each token from attending to its own
position in the value sum, preventing the "attention similarity bias" where softmax allocates
excessive weight to the self-token and wastes capacity on identity transformation. Applied
inside the local sliding-window branch of every attention layer.

### Residual Attention (AttnRes)
**Title:** Attention Residuals (Technical Report)  
**Authors:** Chen et al., Moonshot AI (Kimi Team)  
**Year:** 2026  
**arXiv:** [2603.15031](https://arxiv.org/abs/2603.15031)  
**MORPH uses:** The per-head learned scalar α that additively combines the current layer's
attention output with a depth-weighted residual from earlier layer outputs, providing stable
gradient flow and bounded hidden-state magnitude as depth scales.

### CoPE — Clipped RoPE
**Title:** CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs  
**Authors:** Haoran Li, Sucheng Ren, Alan Yuille, Feng Wang  
**Year:** 2026  
**arXiv:** [2602.05258](https://arxiv.org/abs/2602.05258)  
**MORPH uses:** Soft cosine-taper attenuation of low-frequency RoPE components whose wavelength
exceeds the training context length, eliminating out-of-distribution position outliers and
enabling smooth extrapolation to contexts up to 256k tokens without fine-tuning.

### Attention Sinks — StreamingLLM
**Title:** Efficient Streaming Language Models with Attention Sinks  
**Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT + Meta)  
**Year:** 2023 (ICLR 2024)  
**arXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)  
**MORPH uses:** Prepending a small fixed number of learnable "sink" tokens to every sequence so
that the attention mechanism has stable receptacles for concentrated softmax mass, preventing
the instability that arises when initial-token KV entries are evicted from a sliding-window cache.

### Value Shift
The v-shift (blending the current-step value projection with the previous-step value projection)
is introduced and described in the CCA paper (Figliolia et al., arXiv:2510.04476) as part of the
CCA mechanism. No independent prior paper. See the CCA entry above.

---

## 3. Memory

### Titans — Neural Memory
**Title:** Titans: Learning to Memorize at Test Time  
**Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)  
**Year:** 2025  
**arXiv:** [2501.00663](https://arxiv.org/abs/2501.00663)  
**MORPH uses:** The Memory-Augmented Context (MAC) variant: gradient-based surprise update where
a deep MLP memory M is updated on the forward pass via momentum-accelerated associative loss
minimization — S_t = η_t·S_{t-1} − θ_t·∇‖M(k)−v‖², M_t = (1−α_t)·M_{t-1} + S_t — with
α/η/θ as sigmoid(Linear(chunk_pooled)) learned gates, L2-normalized K/Q, and SiLU activation.

---

## 4. Residual Streams

### mHC — Manifold-Constrained Hyper-Connections
**Title:** mHC: Manifold-Constrained Hyper-Connections  
**Authors:** DeepSeek-AI (19 researchers, led by Liang Wenfeng)  
**Year:** 2025  
**arXiv:** [2512.24880](https://arxiv.org/abs/2512.24880)  
**MORPH uses:** The multi-stream residual architecture where each layer reads from and writes
to N parallel residual channels via a doubly-stochastic mixing matrix (Sinkhorn-Knopp
normalized onto the Birkhoff polytope), preserving feature-mean conservation and bounded
signal propagation. Eliminates the catastrophic amplification (~3000×) that unconstrained
Hyper-Connections can exhibit at depth.

### Hyper-Connections (predecessor to mHC)
**Title:** Hyper-Connections  
**Authors:** Defa Zhu et al. (ByteDance)  
**Year:** 2024 (ICLR 2025)  
**arXiv:** [2409.19606](https://arxiv.org/abs/2409.19606)  
**MORPH uses:** The concept of expanding a single residual stream into N parallel streams with
learned inter-stream routing (generalized residuals). MORPH implements the mHC-stabilized
variant (arXiv:2512.24880), but this paper is the origin of the multi-stream residual idea.

---

## 5. Embeddings

### Lorentz / Hyperbolic Embeddings
**Title:** Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry  
**Authors:** Maximilian Nickel, Douwe Kiela (Facebook AI Research)  
**Year:** 2018 (ICML 2018, pp. 3776–3785)  
**arXiv:** [1806.03417](https://arxiv.org/abs/1806.03417)  
**MORPH uses:** The Lorentz (hyperboloid) model of hyperbolic geometry for embedding the
hierarchical component of the hybrid token embedding, enabling compact representation of
power-law / tree-structured semantic relationships that Euclidean embeddings require far more
dimensions to approximate.

### Hybrid (Mixed-Curvature) Embeddings
**Title:** Learning Mixed-Curvature Representations in Product Spaces  
**Authors:** Albert Gu, Frederic Sala, Beliz Gunel, Christopher Ré (Stanford)  
**Year:** 2019 (ICLR 2019)  
**OpenReview:** [HJxeWnCcF7](https://openreview.net/forum?id=HJxeWnCcF7)  
*(No arXiv preprint; canonical reference is OpenReview)*  
**MORPH uses:** The product-manifold formalism combining Euclidean and hyperbolic (Lorentz)
components in a single embedding space, giving the model heterogeneous curvature — flat space
for local syntactic structure, negatively curved space for hierarchical/ontological structure.
MORPH's `embeddings.py` implements this as eucl ⊕ Lorentz with learned mixing weights.

---

## 6. Sparsity & Routing

### Block-ELL Sparse Format
**Source:** NVIDIA cuSPARSE Library + NVIDIA Technical Blog  
**Reference:** "Accelerating Matrix Multiplication with Block Sparse Format and NVIDIA Tensor Cores" — NVIDIA Developer Blog (2023);  
cuSPARSE [Blocked-ELL Storage Format Documentation](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)  
**MORPH uses:** The Blocked-Ellpack (Blocked-ELL) storage format where nonzero weight sub-matrices
are stored in fixed-size dense tiles (32×32 on SM120/5090) with a companion column-index array.
MORPH's CMS pruning selects which tiles survive; surviving tiles are stored in Block-ELL for
hardware-efficient SpMM using Tensor Cores, achieving near-linear speedup proportional to sparsity.

### ReMoE — Differentiable MoE Routing
**Title:** ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing  
**Authors:** Ziteng Wang, Jianfei Chen, Jun Zhu (Tsinghua University)  
**Year:** 2024 (ICLR 2025)  
**arXiv:** [2412.14711](https://arxiv.org/abs/2412.14711)  
**MORPH uses:** ReLU-based continuous routing over macro tile-groups (32×32 Block-ELL tiles),
replacing the non-differentiable TopK gate with a differentiable L1-regularized ReLU that
naturally produces sparse expert selection without the gradient discontinuity of standard MoE
routing.

### PEER — Product Key Retrieval
**Title:** Large Memory Layers with Product Keys  
**Authors:** Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer,
Hervé Jégou (Facebook AI Research)  
**Year:** 2019 (NeurIPS 2019)  
**arXiv:** [1907.05242](https://arxiv.org/abs/1907.05242)  
**MORPH uses:** The product-key lookup mechanism (decomposed key = k₁ ⊗ k₂ for O(√N) search
over N=tile-group combinations) as the routing primitive for selecting which Block-ELL tile-groups
to activate per token. MORPH adopts the PEER routing mechanism (not the full PEER layer
computation — tile-groups remain full-rank, not rank-k projections).

---

## 7. Regularization & Self-Supervised Objectives

### STP — Semantic Tube Prediction (Geodesic Regularizer)
**Title:** Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA  
**Authors:** Hai Huang, Yann LeCun, Randall Balestriero (galilai-group / NYU)  
**Year:** 2026  
**arXiv:** [2602.22617](https://arxiv.org/abs/2602.22617)  
**MORPH uses:** The geodesic smoothness constraint applied to hidden-state trajectories during
pretraining — confining intermediate states to lie within a tubular neighborhood of the geodesic
connecting segment boundaries on the semantic manifold. MORPH applies STP during pretraining
(not fine-tuning as in the paper) with a multi-scale scheme (strides 1,2,4,…,τ=64).

Their theorem focuses on the value in SFT/RL, saying teacher forcing protects the tubes during pretraining. Ablations found massive AR generation improvements on base model training using STP.

### LeJEPA — Latent Prediction Without Collapse
**Title:** LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics  
**Authors:** Randall Balestriero, Yann LeCun (galilai-group / NYU)  
**Year:** 2025  
**arXiv:** [2511.08544](https://arxiv.org/abs/2511.08544)  
**MORPH uses:** The split_nsm z-latent prediction objective: backbone predicts mean(next segment
z_coda) while memory predicts the next segment prelude state. This JEPA-style latent-space
prediction prevents mode collapse without a teacher-student EMA setup or stop-gradients,
guided by the SIGReg anti-collapse regularizer from the same paper.

### SIGReg — Sketched Isotropic Gaussian Regularization
Introduced in the LeJEPA paper (Balestriero & LeCun, arXiv:2511.08544). SIGReg uses randomized
1D projections and characteristic-function matching to enforce that learned embeddings follow
an isotropic Gaussian distribution, preventing representation collapse with linear time and
memory complexity. MORPH applies SIGReg to z-latent embeddings in `prediction.py`.
See the LeJEPA entry above.

### LLM-JEPA
**Title:** LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures  
**Authors:** Hai Huang, Yann LeCun, Randall Balestriero (galilai-group)  
**Year:** 2025  
**arXiv:** [2509.14252](https://arxiv.org/abs/2509.14252)  
**MORPH uses:** The hybrid training objective combining next-token prediction loss with a
JEPA embedding-space prediction loss over related text views (e.g., code snippet ↔ docstring).
This is the direct application of LeJEPA principles to LLM pretraining that MORPH's
`prediction.py` extends.

---

## 8. Feed-Forward Networks

### SwiGLU — Gated Feed-Forward Activation
**Title:** GLU Variants Improve Transformer  
**Authors:** Noam Shazeer (Google)  
**Year:** 2020  
**arXiv:** [2002.05202](https://arxiv.org/abs/2002.05202)  
**MORPH uses:** SwiGLU (Swish-gated linear unit) as the MLP activation function in all
feed-forward sublayers — `FFN(x) = (xW₁ ⊙ Swish(xV)) · W₂` — providing a gated nonlinearity
that consistently outperforms GELU/ReLU variants on perplexity at equal parameter count.

---

## 9. Training Objectives

### MTP — Multi-Token Prediction
**Title:** Better & Faster Large Language Models via Multi-token Prediction  
**Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz,
Gabriel Synnaeve (Meta FAIR)  
**Year:** 2024  
**arXiv:** [2404.19737](https://arxiv.org/abs/2404.19737)  
**MORPH uses:** N independent output heads on top of a shared trunk, each predicting the token
n positions ahead (n = 1…N), as auxiliary training signal that densifies gradient information
and improves sample efficiency, especially on code. MORPH uses N=4 MTP heads during pretraining.

### STE Ternary — Straight-Through Estimator + BitNet b1.58
**Title:** The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits  
**Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang,
Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research)  
**Year:** 2024  
**arXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)  
**MORPH uses:** STE-based ternary quantization via shadow weights: full-precision AdamW
optimizer state maintained in fp32 shadow weights, which are quantized to {−1, 0, +1} for
the forward pass using absmean scaling, with straight-through gradients flowing back to the
shadow weights. This is the only ternary training method validated to work reliably at scale
(8 alternatives tested in prior ablations, all failed).

The implementation provided here extends across all learned weights of the backbone (not the neural memory).


---

## 10. Tokenization & Data

### StarCoder2 — Tokenizer
**Title:** StarCoder 2 and The Stack v2: The Next Generation  
**Authors:** Anton Lozhkov, Raymond Li, Loubna Ben Allal, et al. (BigCode / HuggingFace)  
**Year:** 2024  
**arXiv:** [2402.19173](https://arxiv.org/abs/2402.19173)  
**MORPH uses:** The StarCoder2 tokenizer (49,152-vocabulary BPE, fill-in-the-middle capable,
600+ programming languages) for code data in MORPH's mixed OpenWebText + code pretraining.
The 49k vocab cleanly stacks with a bigram hash-vocab prefix for rare byte patterns.

---

## 11. Inference Scaling

### Zyphra RSA — Markovian Recurrent Speculative Aggregation
**Title:** ZAYA1-8B Technical Report  
**Authors:** Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, Beren Millidge,
Krithik Puthalath, Danny Martinelli et al. (Zyphra)  
**Year:** 2026  
**arXiv:** [2605.05365](https://arxiv.org/abs/2605.05365)  
**MORPH uses:** The Markovian RSA test-time compute scheme: N parallel traces generated
simultaneously, then recursively aggregated; each reasoning chunk operates on a fixed-size
context window (Markovian: only the tail of the previous chunk is carried forward), enabling
unbounded reasoning with constant KV memory. MORPH's outer loop is designed to support RSA
harness deployment after RL training, currently deferred.

---

## Quick Reference Table

| # | Technique | Paper | arXiv |
|---|-----------|-------|-------|
| 1 | Parcae Loop | Huang et al. (UCSD+Together, 2026) | [2604.12946](https://arxiv.org/abs/2604.12946) |
| 2 | Block-ELL Format | NVIDIA cuSPARSE (2021+) | [developer.nvidia.com](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/) |
| 3 | CMS Topology | Original work — MORPH project | — |
| 4 | Neural Memory (Titans) | Behrouz, Zhong, Mirrokni (Google, 2025) | [2501.00663](https://arxiv.org/abs/2501.00663) |
| 5 | CCA | Figliolia et al. (Zyphra, 2025) | [2510.04476](https://arxiv.org/abs/2510.04476) |
| 6 | CSA / HCA | DeepSeek-AI (2026) | [HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| 7 | STP | Huang, LeCun, Balestriero (2026) | [2602.22617](https://arxiv.org/abs/2602.22617) |
| 8 | LeJEPA | Balestriero, LeCun (2025) | [2511.08544](https://arxiv.org/abs/2511.08544) |
| 9 | SIGReg | Balestriero, LeCun (2025) | [2511.08544](https://arxiv.org/abs/2511.08544) |
| 10 | Lorentz Embeddings | Nickel, Kiela (ICML 2018) | [1806.03417](https://arxiv.org/abs/1806.03417) |
| 11 | Hybrid Embeddings | Gu, Sala, Gunel, Ré (ICLR 2019) | [OpenReview](https://openreview.net/forum?id=HJxeWnCcF7) |
| 12 | CoPE (Clipped RoPE) | Li, Ren, Yuille, Wang (2026) | [2602.05258](https://arxiv.org/abs/2602.05258) |
| 13 | XSA | Zhai (Apple, 2026) | [2603.09078](https://arxiv.org/abs/2603.09078) |
| 14 | Residual Attention | Chen et al. / Kimi (2026) | [2603.15031](https://arxiv.org/abs/2603.15031) |
| 15 | SwiGLU | Shazeer (Google, 2020) | [2002.05202](https://arxiv.org/abs/2002.05202) |
| 16 | MTP | Gloeckle et al. (Meta, 2024) | [2404.19737](https://arxiv.org/abs/2404.19737) |
| 17 | STE Ternary (BitNet b1.58) | Ma et al. (Microsoft, 2024) | [2402.17764](https://arxiv.org/abs/2402.17764) |
| 18 | ReMoE | Wang, Chen, Zhu (Tsinghua, 2025) | [2412.14711](https://arxiv.org/abs/2412.14711) |
| 19 | PEER | Lample et al. (FAIR, 2019) | [1907.05242](https://arxiv.org/abs/1907.05242) |
| 20 | mHC | DeepSeek-AI (2025) | [2512.24880](https://arxiv.org/abs/2512.24880) |
| 20b | Hyper-Connections | Zhu et al. / ByteDance (2024) | [2409.19606](https://arxiv.org/abs/2409.19606) |
| 21 | Zyphra RSA | Figliolia et al. (Zyphra, 2026) | [2605.05365](https://arxiv.org/abs/2605.05365) |
| 22 | StarCoder2 Tokenizer | Lozhkov et al. (BigCode, 2024) | [2402.19173](https://arxiv.org/abs/2402.19173) |
| 23 | Nested Learning | Behrouz et al. (NeurIPS 2025) | [2512.24695](https://arxiv.org/abs/2512.24695) |
| 24 | Poisson Depth Sampling | Huang et al. (Parcae, 2026) | [2604.12946](https://arxiv.org/abs/2604.12946) |
| 25 | Attention Sinks | Xiao et al. (MIT/Meta, 2023) | [2309.17453](https://arxiv.org/abs/2309.17453) |
| 26 | Value Shift | Figliolia et al. (Zyphra, 2025) | [2510.04476](https://arxiv.org/abs/2510.04476) |
| 27 | LLM-JEPA | Huang, LeCun, Balestriero (2025) | [2509.14252](https://arxiv.org/abs/2509.14252) |
