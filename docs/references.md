# MORPH Architecture — Paper References

**MORPH** (Orchestrates Recursive Pruned Hierarchies) is a production research model combining
Parcae-style looped transformers with diagonal SSM injection, MORTAR BCSR structured sparsity,
CCA+CSA+HCA triple-axis attention compression, a gated-linear-attention (GLA) retention branch,
JPmHC orthogonal hyper-connection residuals, hybrid
hyperbolic/Euclidean embeddings, STE ternary shadow weights, ReMoE product-key routing, and
Token Superposition Training (TST) during the early pretraining phase.

Papers are grouped by architectural component. "Original work" entries are techniques developed
within the MORPH project with no external precedent.

Local markdown copies are grouped by topic in `[references/MANIFEST.md](references/MANIFEST.md)`.

> **Status note.** Several components cited below were evaluated but **removed** from the current
> architecture: the Titans neural-memory module (§3), Nested Learning (§3), the Multi-Rate Residual
> (§4), the Hyper-Connection Sinkhorn arm (§4), the LeJEPA / SIGReg / LLM-JEPA z-latent objectives,
> the STP / Semantic Step Prediction geodesic regularizers (§7), and MTP heads (§9).
> They are kept here as citations and marked **(removed)** in place, mirroring the Block-ELL
> "(superseded)" treatment in §6. The current cross-iteration memory is the GLA retention branch
> (`morph/model/gla.py`); the sole residual is the JPmHC Cayley hyper-connection.
> Some variables are not renamed in code to keep checkpoint consistency.

---

## 1. Looping & Depth

### Parcae — Stable Looped Transformer

**Title:** Parcae: Scaling Laws For Stable Looped Language Models  
**Authors:** Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu (UCSD + Together AI)  
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
loop (T drawn from a Poisson at `mean_depth`, capped at `max_depth` — 6/8 in the local  
`base.yaml`, 8 in `cloud.yaml`).  

**NOTES:** This has been incredibly stable, and in my testing is faster than equivalent layer depth.  
It also smokes other looped implementations on per bit intelligence.

Some foot guns about looped models: a learned gate seems ideal at face value, but it is generally worse.  
These tend to collapse, and if you solve the collapse problem the learned iterative map is less generalized.  
Poisson depth sampling forces the iterative map to generalize across a wide range of potential depths.

### DiffusionBlocks on recurrent-depth models — see §9

Applies a diffusion interpretation to Huginn's looped core and trains it as a single-pass denoiser
instead of looping with truncated BPTT. The mode most relevant to MORPH's Parcae core.
Assessment: [`diffusionblocks-morph-assessment.md`](diffusionblocks-morph-assessment.md).


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

**NOTES:** This does reduce throughput a noticable amount, the cache savings are immense and make much  
larger training tractable on smaller hardware. It seems to increase sensitivity to KV cache quantization, but  
this is still unclear from current testing. Out of all the techniques that have been ablated, this one is my favorite.  
It is quite clever.

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
**Authors:** Kimi Team (Moonshot AI)  
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

**NOTES:** Several positional methods were ablated. CoPE beat NoPE and RoPE by a considerable margin.  
One variant was quite different. CosFormer [https://arxiv.org/abs/2202.08791](https://arxiv.org/abs/2202.08791) provides position directly  
with how it handles attention. This worked as well as CoPE, but the other behaviors of the attention are a minor regression.

### Attention Sinks — StreamingLLM

**Title:** Efficient Streaming Language Models with Attention Sinks  
**Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT + Meta)  
**Year:** 2023 (ICLR 2024)  
**arXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)  
**MORPH uses:** Prepending a small fixed number of learnable "sink" tokens to every sequence so
that the attention mechanism has stable receptacles for concentrated softmax mass, preventing
the instability that arises when initial-token KV entries are evicted from a sliding-window cache.

**NOTES:** I am of the opinion that preventing attention sinking is better than providing a stable place  
for attention sinking. This needs further ablation.

### Value Shift

The v-shift (blending the current-step value projection with the previous-step value projection)
is introduced and described in the CCA paper (Figliolia et al., arXiv:2510.04476) as part of the
CCA mechanism. No independent prior paper. See the CCA entry above.

---

## 3. Memory

### GLA — Gated Linear Attention (retention branch)

**Title:** Gated Linear Attention Transformers with Hardware-Efficient Training  
**Authors:** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim  
**Year:** 2023 (ICML 2024)  
**arXiv:** [2312.06635](https://arxiv.org/abs/2312.06635)  
**MORPH uses:** Current cross-iteration memory path. `morph/model/gla.py` implements per-key-channel
gated linear attention with optional carry of the final state `S_T` across core-loop iterations;
output is gated and GroupNorm'd following the paper. Enabled by default (`retention: true` in
`base.yaml`) on configured section-local layers, beside CCA+CSA/HCA attention rather than as a
full-block interleave.

### Titans — Neural Memory (removed)

**Title:** Titans: Learning to Memorize at Test Time  
**Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)  
**Year:** 2025  
**arXiv:** [2501.00663](https://arxiv.org/abs/2501.00663)  
**MORPH history (removed):** An earlier MORPH version used the Titans Memory-Augmented Context
(MAC) variant: a deep-MLP memory M updated on the forward pass via momentum-accelerated
associative-loss minimization (S_t = η_t·S_{t-1} − θ_t·∇‖M(k)−v‖², M_t = (1−α_t)·M_{t-1} + S_t).
This module was **removed**; the current cross-iteration memory is the GLA retention branch above.

**NOTES:** This is extremely unstable. Spent months working with it across various modalities and architectures.  
It is very collapse prone. SigREG helps a lot but the fast weights still tend to collapse all the weights and rely on attention only.  
Then the backbone model attention sinks so heavily on the prepended tokens that it goes totally divergent.  
The worst part is that this behavior is sensitive to the data. If a data mix survives long enough the memory generally won't collapse.

My current thinking is that it needs another loss function to fight the divergent attraction. I ablated several  
JEPA-like hidden state predictors that seemed to help, but were still unstable. It is difficult to have a second  
loss function, as the fast weights update during forward pass and applying a second GD after that is a total lobotomy.  
The 2 loss values must modify the same gradient update. 

### Nested Learning (removed)

**Title:** Nested Learning: The Illusion of Deep Learning Architecture  
**Authors:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni  
**Year:** 2025 (NeurIPS 2025)  
**arXiv:** [2512.24695](https://arxiv.org/abs/2512.24695)  
**MORPH history (removed):** Nested Learning frames optimizers and architectures as multi-level
associative-memory processes (continuum memory systems, self-referential modules). It informed
early MORPH thinking around multi-timescale memory and the Titans lineage, but no Nested Learning
module ships in the current code. Kept as related work for the removed memory stack.  

**NOTES:** This works well, it is just slow. Lots of very interesting things can be done with this optimizer. AdEMAMix beats it outright. Muon generally matches. The CMS used for tile grading came from an implementation of nested learning, it is just an EMA.

---

## 4. Residual Streams

### Multi-Rate Residual (MRR) — MORPH's approach (removed)

An earlier MORPH version used a **Multi-Rate Residual (MRR)**: the d_model-dim hidden state split
into 3 sub-channels (compute 3N, context 2N, memory N) with per-channel learned γ gains on the
residual update — a simpler mechanism than HC/mHC, with no cross-stream mixing. It was **removed**
in favor of the JPmHC Cayley hyper-connection (below), which is now MORPH's sole residual. (The
block attributes are still named `mrr_attn`/`mrr_mlp` — a retained legacy name for checkpoint
compatibility — but they hold `HyperConnectionResidual` modules.)

**NOTES:** This was a Claude dumb-dumb moment. Claude said "mHC? I think we should dip our toes into this and do half of that" and proceeded to invent something new that works, but not as well as mHC. But the naming on checkpoints needed to remain consistent so this mHC implementation uses MRR for variable names now. To be addressed in a v2 or before final MORPH release as the last checkpoint breaking change before the final full training run.

### JPmHC — Jacobian-Preserving Manifold Hyper-Connections (Cayley)

**Title:** JPmHC: Dynamical Isometry via Orthogonal Hyper-Connections  
**Authors:** Biswa Sengupta, Jinhua Wang, Leo Brunswic (JP Morgan Chase LLM Suite Team)  
**Year:** 2026  
**arXiv:** [2602.18308](https://arxiv.org/abs/2602.18308)  
**MORPH uses:** The sole Hyper-Connection residual (selected unconditionally; the old
`residual_mode` config knob was removed). Widens the
residual stream to n parallel C-dim streams with input-dependent H^pre/H^post/H^res mappings.
Constrains H^res to the Stiefel manifold via Cayley transform of a skew-symmetric matrix,
giving exact dynamical isometry (all singular values = 1) without iterative Sinkhorn
normalisation — 5.2× cheaper H^res than the bistochastic variant at equal stream count.
Free-probability analysis predicts Jacobian spectra for structured skips; memory-efficient
implicit differentiation for the Cayley fixed-point projection. MORPH's `hyper_connections.py`
and fused Triton kernel (`fused_hyper_connection.py`) implement this mechanism.

### mHC — Manifold-Constrained Hyper-Connections (related work)

**Title:** mHC: Manifold-Constrained Hyper-Connections  
**Authors:** Zhenda Xie, Yixuan Wei, Huanqi Cao, et al. (DeepSeek-AI)  
**Year:** 2025  
**arXiv:** [2512.24880](https://arxiv.org/abs/2512.24880)  
**Relation to MORPH:** mHC uses n parallel full-C-dim streams with an n×n doubly-stochastic
mixing matrix (Sinkhorn-Knopp, Birkhoff polytope), plus aggregation (H^pre) and distribution
(H^post) vectors with dynamic input-dependent gating. MORPH evaluated this as an alternate
HC arm but **removed** the Sinkhorn variant; the Cayley JPmHC variant is the only residual.

### Hyper-Connections (predecessor to mHC, related work)

**Title:** Hyper-Connections  
**Authors:** Defa Zhu et al. (ByteDance)  
**Year:** 2024 (ICLR 2025)  
**arXiv:** [2409.19606](https://arxiv.org/abs/2409.19606)  
**Relation to MORPH:** Origin of the multi-stream residual concept. HC uses unconstrained n×n
mixing matrices; MORPH's manifold-constrained JPmHC (Cayley) variant descends from this line.
(MORPH's now-removed MRR was a much simpler per-channel-diagonal take on different update rates.)

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

**NOTES:** This is very sensitive to quantization for obvious reasons. Even fp8 is a major regression.

### Hybrid (Mixed-Curvature) Embeddings

**Title:** Learning Mixed-Curvature Representations in Products of Model Spaces  
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

### Lottery Ticket Hypothesis (LTH)

**Title:** The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks  
**Authors:** Jonathan Frankle, Michael Carbin (MIT CSAIL)  
**Year:** 2019 (ICLR 2019)  
**arXiv:** [1803.03635](https://arxiv.org/abs/1803.03635)  
**MORPH uses:** Conceptual foundation for structured sparsity. LTH shows that dense networks
contain sparse subnetworks ("winning tickets") that match full accuracy when retrained from
early initialization. MORPH does not implement the full iterative prune-and-rewind protocol;
instead CMS selects block-sparse topology (128×128 execution blocks) from importance scores accumulated
during dense pretraining, then compacts to hardware-efficient sparse weights. The hypothesis
motivates why aggressive block pruning can retain capability after the dense→prune→compact
schedule.

### Block-ELL Sparse Format (superseded)

**Source:** NVIDIA cuSPARSE Library + NVIDIA Technical Blog  
**Reference:** "Accelerating Matrix Multiplication with Block Sparse Format and NVIDIA Tensor Cores" — NVIDIA Developer Blog (2023);  
cuSPARSE [Blocked-ELL Storage Format Documentation](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)  
**MORPH history:** Early MORPH pruning used Blocked-Ellpack (Blocked-ELL) tiles with a companion
column-index array. This backend was removed 2026-06-11 in favor of MORTAR BCSR + MegaBlocks STK
kernels (see MegaBlocks entry below).

### MegaBlocks — Block-Sparse GPU Kernels (STK)

**Title:** MegaBlocks: Efficient Sparse Training with Mixture-of-Experts  
**Authors:** Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia (Stanford, Microsoft Research, Google Research)  
**Year:** 2022 (MLSys 2023)  
**arXiv:** [2211.15841](https://arxiv.org/abs/2211.15841)  
**MORPH uses:** The Sparse Toolkit (STK) block-sparse Triton kernels vendored in `morph/sparse/stk`,  
originally developed for dropless MoE routing via blocked-CSR/COO encodings and transpose-index  
tricks. MORPH repurposes these kernels for MORTAR 128×128 BCSR sparse matmul after carve() —  
measured 3.09× faster at 0.25 density, replacing the slower Block-ELL backend.

### ReMoE — Differentiable MoE Routing

**Title:** ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing  
**Authors:** Ziteng Wang, Jun Zhu, Jianfei Chen (Tsinghua University)  
**Year:** 2024 (ICLR 2025)  
**arXiv:** [2412.14711](https://arxiv.org/abs/2412.14711)  
**MORPH uses:** ReLU-based continuous routing over contiguous d_ff hidden-neuron clusters
(`TileRouter` in `morph/model/routing.py`), replacing the non-differentiable TopK gate with a
differentiable ReLU gate that naturally produces sparse expert selection without the gradient
discontinuity of standard MoE routing.

**NOTES:** I lit up when I read this paper. I knew the per tile nature of MORPH was ideal for this.  
I have generally seen that this per-token tile-selection routing is superior to full density activation.

The 16x16 tile granularity hits a wall at around 20% density where the model regresses dramatically. 25% is a safe density across tests and is with in error of baseline full density when pruned gradually.

### PEER — Product Key Retrieval

**Title:** Large Memory Layers with Product Keys  
**Authors:** Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer,  
Hervé Jégou (Facebook AI Research)  
**Year:** 2019 (NeurIPS 2019)  
**arXiv:** [1907.05242](https://arxiv.org/abs/1907.05242)  
**MORPH uses:** The product-key lookup mechanism (decomposed key = k₁ ⊗ k₂ for O(√N) search)  
as the routing primitive for selecting which d_ff neuron-clusters to activate per token  
(`TileRouter`). MORPH adopts the PEER routing mechanism (not the full PEER layer  
computation — clusters remain full-rank, not rank-k projections).

---

## 7. Regularization & Self-Supervised Objectives

### STP — Semantic Tube Prediction (Geodesic Regularizer) (removed)

**Title:** Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA  
**Authors:** Hai Huang, Yann LeCun, Randall Balestriero (galilai-group / NYU)  
**Year:** 2026  
**arXiv:** [2602.22617](https://arxiv.org/abs/2602.22617)  
**MORPH history (removed):** An earlier MORPH version applied the paper's geodesic smoothness
constraint to hidden-state trajectories during pretraining (not fine-tuning as in the paper)
with a multi-scale scheme (strides 1,2,4,…,τ=64). The regularizer was **removed** along with
`morph/model/prediction.py`; the current training loss is plain next-token cross-entropy. Extensive testing on this showed that potentially applying STP at punctuation boundaries (.,;!?--\n) during pretraining has some benefit to autoregressive generation.

**NOTES**: This was tested in pretraining and sft as well as Step STP in sft, and my own version punc STP (apply STP at punctuation boundaries) during pretraining and sft.  
These work great in SFT, especially punc-STP. I also observe consistent effects in pretraining out to 1B tokens, but its not clear there are down stream benefits. Using any STP during pretraining makes SFT need less data with or with out STP, but still using STP during SFT has massive improvements.

The winning recipe across all the STP/JEPA ablations I did was to do punc-STP during pretraining, then do punc-JEPA in reinforcement learning (it is JEPA like. Predict the final hidden state at next punc boundary from every position in the sequence leading up to it). I think my scale was too small to see major benefits here, I tested LoRA on several models and as I moved up it had stronger effects on reducing repetition and data quantity needed. The hidden state prediction needed the smoothness STP provided to really compound this effect. AR diversity was much better and stayed on topic for several more sentences on average (blind A/B with both human and LLM).

This was all removed because the interp work this needed and the scale testing it needed were too much.  
Punc-STP during pretraining has no noticable effect on ppl or training wall clock. I am considering using it during the cloud training deployement so that the regularization is in the base if I find other uses for it later.

### LeJEPA — Latent Prediction Without Collapse (removed)

**Title:** LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics  
**Authors:** Randall Balestriero, Yann LeCun (galilai-group / NYU)  
**Year:** 2025  
**arXiv:** [2511.08544](https://arxiv.org/abs/2511.08544)  
**MORPH history (removed):** An earlier MORPH version added a split_nsm z-latent prediction
objective (backbone predicts mean(next-segment z_coda); memory predicts the next-segment prelude
state) with the SIGReg anti-collapse regularizer. The z-latent / JEPA objectives were **removed**
(`morph/model/prediction.py` no longer exists). 
Several z-latent objectives were tested with varying success. LLM-JEPA was tested and a synth dataset for pretraining resembling LLM-JEPA was also tested. Results were mixed at this scale.

### SIGReg — Sketched Isotropic Gaussian Regularization (removed)

Introduced in the LeJEPA paper (Balestriero & LeCun, arXiv:2511.08544). SIGReg uses randomized
1D projections and characteristic-function matching to enforce that learned embeddings follow
an isotropic Gaussian distribution, preventing representation collapse with linear time and
memory complexity. MORPH applied SIGReg to z-latent embeddings; **removed** together with the
LeJEPA objective above.

### LLM-JEPA (removed)

**Title:** LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures  
**Authors:** Hai Huang, Yann LeCun, Randall Balestriero (galilai-group)  
**Year:** 2025  
**arXiv:** [2509.14252](https://arxiv.org/abs/2509.14252)  
**MORPH history (removed):** An earlier MORPH version applied the hybrid objective combining
next-token prediction with a JEPA embedding-space prediction loss over related text views
(e.g., code snippet ↔ docstring). **Removed** with the rest of the z-latent / JEPA stack
(`morph/model/prediction.py` no longer exists).

### Semantic Step Prediction (removed lineage)

**Title:** Semantic Step Prediction: Multi-Step Latent Forecasting in LLM Reasoning Trajectories
via Step Sampling  
**Authors:** Yidi Yuan  
**Year:** 2026  
**arXiv:** [2604.18464](https://arxiv.org/abs/2604.18464)  
**MORPH history (removed):** Step-boundary follow-up to STP (arXiv:2602.22617). Argues that
sampling the STP geodesic loss at semantic reasoning-step boundaries (rather than random token
positions) dominates the geometric outcome. MORPH never shipped this variant; it is archived as
related work for the removed STP / latent-prediction stack (`morph/model/prediction.py` no longer
exists). Local notes only (no PDF in the archive).

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

### MTP — Multi-Token Prediction (removed)

**Title:** Better & Faster Large Language Models via Multi-token Prediction  
**Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz,
Gabriel Synnaeve (Meta FAIR)  
**Year:** 2024  
**arXiv:** [2404.19737](https://arxiv.org/abs/2404.19737)  
**Status:** Not in the current code — there are no MTP heads in `morph/`; the training loss is
plain single-token next-token cross-entropy. Kept as a citation. Requires larger scale to benefit.

### STE Ternary — Straight-Through Estimator + BitNet b1.58

**Title:** The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits  
**Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang,
Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research)  
**Year:** 2024  
**arXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)  
**MORPH uses:** STE-based ternary quantization via shadow weights: full-precision  
optimizer state maintained in fp32 shadow weights, which are quantized to {−1, 0, +1} for  
the forward pass using absmean scaling, with straight-through gradients flowing back to the  
shadow weights. This is the only ternary training method validated to work reliably at scale  
(8 alternatives tested in prior ablations, STE ternary won).

The implementation covers the backbone scope (`ternary_scope: backbone` — MLP/mix/mhc
projections); attention projections stay bf16.

**NOTES:** I spent a long time trying everything under the sun to keep the weights as ternary with out keeping full shadow weights behind it. Nothing worked. If you know a method, please let me know. I even tried applying EGGROLL to the ternary weights and that didn't work. [https://arxiv.org/abs/2511.16652](https://arxiv.org/abs/2511.16652)

### Token Superposition Training (TST)

**Title:** Efficient Pre-Training with Token Superposition  
**Authors:** Bowen Peng, Théo Gigant, Jeffrey Quesnelle (Nous Research)  
**Year:** 2026  
**arXiv:** [2605.06546](https://arxiv.org/abs/2605.06546)  
**MORPH uses:** Two-phase Token Superposition Training as a drop-in pretraining schedule with no
architecture, tokenizer, or optimizer change. Phase 1 packs `tst_bag_size` contiguous tokens into
one position and trains with multi-hot cross-entropy (MCE) through the existing fused-CE path;
phase 2 recovers with standard next-token prediction (`bag_size=0`). Controlled by
`training.tst_bag_size` and `training.tst_ratio` in `base.yaml` (default bag 6, ratio 0.3 →
superposition for the first 30k of a 100k-step run). Eval and generation always use `bag_size=0`;
`tst_bag_size=0` is bit-identical to the pre-TST baseline. Curriculum configs deliberately disable
TST (`pretrain_curriculum.yaml`).

### DiffusionBlocks — block-wise training via diffusion interpretation (evaluating, nothing built)

**Title:** DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation  
**Authors:** Makoto Shing, Masanori Koyama, Takuya Akiba (Sakana AI, U. Tokyo)  
**Year:** 2026 (ICLR 2026; v1 2025)  
**arXiv:** [2506.14202](https://arxiv.org/abs/2506.14202)  
**MORPH uses:** NOTHING YET — under assessment, see
[`diffusionblocks-morph-assessment.md`](diffusionblocks-morph-assessment.md). Two separable modes.
(a) *Block-wise*: cut `L` layers into `B` σ-range-specialised blocks, each trained by its own
denoising objective, cutting params + grads + **optimizer state** by `B` (unlike gradient
checkpointing, App. G) — but with **no compute saving** (App. H: `L·K` layer evals either way).
(b) *Recurrent-depth* (§5.5, App. E.5): applied to **Huginn**, MORPH's closest published relative
(2 prelude / 4-layer recurrent core / 2 coda vs MORPH's 4:6:4). For that case the paper explicitly
does **NOT** partition into blocks — *"recurrent-depth models do not require block partitioning
since the entire network is applied recurrently"* — it trains the whole net as a single-pass
denoiser at a sampled σ, deleting `K` iterations and truncated BPTT from the training step
(~10× less compute; MAUVE 0.49 → 0.70, but at 3× the epochs and no true perplexity).
Key facts for us: the framework is **optimizer-agnostic** (AdamW throughout, nothing in the theory
needs it — so AdEMAMix is compatible, but every step-counted schedule stretches by `B`); the
inter-block seam is a *prescribed* convex contraction `z_b = α z_{b-1} + (1−α)D`, `α = σ_b/σ_{b-1}`,
which is a `ρ(J_core) ≤ 1` handle of exactly the kind
`Ai-notes/06-19-2026/MORPH-Iterative-Map-Dynamics/MENTAL-MODEL.md` asks for; equi-probability σ
partitioning matters far more than the layer split (FID 38.03 vs 43.53); quality holds to ~8
layers/block and degrades below. Costs: true perplexity becomes uncomputable, the causal-consistency
mask doubles sequence length (colliding with TUL's slot budget), and TUL's 1.6× conditional-compute
win is largely redundant under mode (b).  
**CAUTION:** as rendered in v4, Eq (3)–(5) carry a sign that makes noise *increase* down the
schedule; use the EDM (Karras et al. 2022, Alg. 2) form. See the assessment doc §4.5.

---

## 10. Optimizer

### AdEMAMix — Dual-EMA Adam Variant

**Title:** The AdEMAMix Optimizer: Better, Faster, Older  
**Authors:** Matteo Pagliardini, Pierre Ablin, David Grangier (EPFL, Apple)  
**Year:** 2024 (ICLR 2025)  
**arXiv:** [2409.03137](https://arxiv.org/abs/2409.03137)  
**MORPH uses:** Optional training optimizer (`cfg.training.optimizer: ademamix_b1zero` —
MORPH's β1=0 fork with a fused Triton kernel for 2-buffer 8-bit state and AdamW8bit memory
parity; the stock 3-buffer bitsandbytes `ademamix` path was removed). Extends AdamW with a
second very-slow momentum EMA (decay β3, `base.yaml` default 0.999) mixed into the update via
weight α (default 8.0): `update = (m₁ + α·m₂)/(√ν + ε) + λ·p`. α and β3 require their own warmup
schedulers (`t_alpha`, `t_beta3`) distinct from LR warmup — essential for stability under
MORPH's flat-LR recipe. Includes prune-aware dead-state masking for CMS-carved weights.

---

## 11. Tokenization & Data

### StarCoder2 — Tokenizer

**Title:** StarCoder 2 and The Stack v2: The Next Generation  
**Authors:** Anton Lozhkov, Raymond Li, Loubna Ben Allal, et al. (BigCode / HuggingFace)  
**Year:** 2024  
**arXiv:** [2402.19173](https://arxiv.org/abs/2402.19173)  
**MORPH uses:** The StarCoder2 tokenizer (49,152-vocabulary BPE, fill-in-the-middle capable,
600+ programming languages) for code data in MORPH's mixed OpenWebText + code pretraining.
The 49k vocab cleanly stacks with a bigram hash-vocab prefix for rare byte patterns.

**NOTES:** I ablated every tokenizer I am aware of between 16k-128k vocab. Starcoder 2 behaves the best per bit with Granite and Phi right behind. This was tested across 3 seeds to 100m tokens and 1 seed with 2 data mixtures.

---

## 12. Inference Scaling

### Zyphra RSA — Markovian Recurrent Speculative Aggregation

**Title:** ZAYA1-8B Technical Report  
**Authors:** Robert Washbourne, Rishi Iyer, Tomas Figliolia, et al., Beren Millidge (Zyphra)  
**Year:** 2026  
**arXiv:** [2605.05365](https://arxiv.org/abs/2605.05365)  
**MORPH uses:** The Markovian RSA test-time compute scheme: N parallel traces generated
simultaneously, then recursively aggregated; each reasoning chunk operates on a fixed-size
context window (Markovian: only the tail of the previous chunk is carried forward), enabling
unbounded reasoning with constant KV memory. MORPH's outer loop is designed to support RSA
harness deployment after RL training, currently deferred.

---

## 13. TUL — Thought Unpack Loop (latent emission & hierarchy) (spec, `experiments/tul`)

TUL loops the Parcae core over one **thought slot per span** and decodes tokens with
the slot's looped state visible as an attended prefix position. Spec:
[tul-spec.md](tul-spec.md). Local copies of every source below live in
`references/tul-latent-emission/`; the per-paper reading notes (31 papers, one templated
note each) are in `ignore/Ai-notes/08-16-2026/prior-art/`. Entries say what TUL takes and,
where a paper argues AGAINST something TUL does, say that too.

### Byte Latent Transformer (BLT)

**Title:** Byte Latent Transformer: Patches Scale Better Than Tokens  
**Authors:** Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer, Ghosh, Lewis, Holtzman, Iyer (Meta FAIR)  
**Year:** 2024  
**arXiv:** [2412.09871](https://arxiv.org/abs/2412.09871)  
**TUL uses:** the local-encoder / global-latent / local-decoder shape. Eq. 5–7: patch
queries initialised by mean-pooling the patch's byte embeddings, cross-attention over the
patch's bytes — TUL's slot input is the mean-pooled span and the prelude's own attention
over the span is that pooling. Eq. 9: the local decoder is seeded by the local ENCODER's
byte states (`D_0 = h_lE`) — TUL's coda input for token positions is the prelude state.
Table 7: patch vector at all decoder layers 0.846 BPB vs first-layer 0.861 vs none 0.866;
Table 9: 1 encoder + 9 decoder layers beat 5+5 — capacity belongs on the decode side.
§4: rule-based patching is "a very close competitor" to learned entropy patching; fixed
stride is worst.

### MegaByte

**Title:** MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers  
**Authors:** Yu, Simig, Flaherty, Aghajanyan, Zettlemoyer, Lewis (Meta AI)  
**Year:** 2023  
**arXiv:** [2305.07185](https://arxiv.org/abs/2305.07185)  
**TUL uses:** the global-over-patches / small-AR-local-per-token split and the fact that
the global output has no loss of its own. Table 7: removing the local AR model (one-shot
patch emission) 0.687 → 1.263 bpb — the datum behind "never decode a span blind". Eq. 4's
per-offset slices of the global vector are NOT used (weakest injection; see Block
Transformer / Hourglass). MegaByte's local `p = 0` position, which emits the first byte of
the patch from the global state, is TUL's slot label.

### H-Net

**Title:** Dynamic Chunking for End-to-End Hierarchical Sequence Modeling  
**Authors:** Hwang, Wang, Huo, Neubig, Dao (CMU / Cartesia)  
**Year:** 2025  
**arXiv:** [2507.07955](https://arxiv.org/abs/2507.07955)  
**TUL uses:** §2.3 signal-propagation recipe for a hierarchy — RMSNorm at the end of each
component ("norm balance"), a Linear on the residual path only with near-zero init,
per-stage LR modulation (outer stages higher). Table 1: fixed pooling worst, whitespace ≈
learned DC — supports the rule-based cut. The learned router + ratio loss + EMA smoothing
(§2.2) is TUL's deferred "learned boundaries" arm. The main network is supervised only
through decoded bytes — loss-free latent.

### Block Transformer

**Title:** Block Transformer: Global-to-Local Language Modeling for Fast Inference  
**Authors:** Ho, Bae, Kim, Ainslie, Lee, Yun, Ye, Kim (KAIST / Google DeepMind)  
**Year:** 2024  
**arXiv:** [2406.02657](https://arxiv.org/abs/2406.02657)  
**TUL uses:** the only BPE-level ablation of global→local injection. Fig 3f: prefix
positions the token decoder can refine (length 2–6) > single prefix > summation ≫
per-layer cross-attention to KV states (−0.18 nats). This is why TUL exposes the slot
state as an attended position rather than a cross-attention branch. Loss is log-linear in
block length (85M: vanilla 2.47, L_B=1/2/4/8 = 2.52/2.65/2.79/2.90); first token of a
block is the hardest position; §4.2 block-level MSE/contrastive losses on the latent HURT
(so TUL's slot-set warm-up is an arm, default off). Needs 2–3× params to match vanilla
PPL, buys 10–20× decode throughput — the honest expectation at 5090 scale.

### Dynamic Token Pooling

**Title:** Efficient Transformers with Dynamic Token Pooling  
**Authors:** Nawrot, Chorowski, Łańcucki, Ponti  
**Year:** 2022  
**arXiv:** [2211.09761](https://arxiv.org/abs/2211.09761)  
**TUL uses:** boundary-rule evidence (character-level): whitespace 1.133 BPC ≈ unigram
1.134 > Gumbel-learned 1.136 (unstable) > entropy 1.138 > vanilla 1.143 > fixed SF2/3/4
1.149/1.155/1.166 (Table 2). Deterministic content-aligned boundaries win; mean-pool >
take-last; boundary placed after the delimiter — all three are TUL's choices.

### Hourglass

**Title:** Hierarchical Transformers Are More Efficient Language Models  
**Authors:** Nawrot, Tworkowski, Tyrolski, Kaiser, Wu, Szegedy, Michalewski  
**Year:** 2021  
**arXiv:** [2110.13711](https://arxiv.org/abs/2110.13711)  
**TUL uses:** the causal-shift argument and Table 6 (1.128 vs 1.460 BPC without a
full-resolution layer after upsampling): position j cannot see positions < j of its own
group from the summary alone — the theory statement of why one-vector-per-span emission
needs a token path. Attention upsampling 1.132 > repeat 1.148 > per-offset linear 1.163.

### Patch-Level Training

**Title:** Patch-Level Training for Large Language Models  
**Authors:** Shao, Meng, Zhou (WeChat AI)  
**Year:** 2024  
**arXiv:** [2407.12665](https://arxiv.org/abs/2407.12665)  
**TUL uses:** the equivalence with MORPH's TST — mean-pooled input bag of K tokens, one
softmax scored on all K next tokens, hard switch to token level; K=2–4 best, K=8 −0.03,
K=16 −0.05 nats; λ optimum ≈ 1/4–3/8 (MORPH's 0.3). Justifies switching TUL on at the
TST boundary and mean-pooling the slot input.

### DeepSeek-V3 (Multi-Token Prediction module only)

**Title:** DeepSeek-V3 Technical Report  
**Authors:** DeepSeek-AI  
**Year:** 2024  
**arXiv:** [2412.19437](https://arxiv.org/abs/2412.19437)  
**TUL uses:** §2.2 as the shipped counter-example to blind multi-token heads: depth k
takes `[RMSNorm(h_{k-1}); RMSNorm(Emb(t_{i+k}))]` — the true previous token is fed back
per depth; helps already at 2.4B active. TUL's coda is token-fed by construction.

### Multi-token prediction (Gloeckle et al.) — see §9 (MTP, removed)

**TUL note:** the blind-heads variant helps only ≥3B (Table S7: worse at 0.3B/0.6B, even
at 1.3B) and hurts once next-token circuits form. Cited here as the reason TUL does not
decode blind; the entry stays in §9.

### Future Lens

**Title:** Future Lens: Anticipating Subsequent Tokens from a Single Hidden State  
**Authors:** Pal, Sun, Yuan, Wallace, Bau (Northeastern)  
**Year:** 2023  
**arXiv:** [2311.04897](https://arxiv.org/abs/2311.04897)  
**TUL uses:** the instrument for "does the slot carry the plan" (linear/soft-prompt
readout on a hidden state, always reported against `teacher_acc` and a bigram floor).
Table 2: blind linear probes 0.292/0.190/0.158 at t+2/3/4 (bigram 0.201); the
learned-prompt reader 0.484/0.437/0.469 is TOKEN-FED (Eq. 9/11) — evidence for the
fed-back decoder, not for blind decode. Fig 4 (read from the PDF): linear t+2 peaks late
in the stack; the mid-stack peak belongs to the token-fed reader.

### Blockwise Parallel Decoding

**Title:** Blockwise Parallel Decoding for Deep Autoregressive Models  
**Authors:** Stern, Shazeer, Uszkoreit (Google)  
**Year:** 2018  
**arXiv:** [1811.03115](https://arxiv.org/abs/1811.03115)  
**TUL uses:** the origin of predict-verify-accept from one state; a frozen trunk saturates
at ~1.8 accepted tokens/step for any k, fine-tuning + distillation gives 4.95 (Table 1).
Cited for the ceiling of a blind reader on a next-token-trained state.

### Medusa

**Title:** Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads  
**Authors:** Cai, Li, Geng, Peng, Lee, Chen, Dao  
**Year:** 2024  
**arXiv:** [2401.10774](https://arxiv.org/abs/2401.10774)  
**TUL uses:** nothing directly; cited so nobody anchors on it — the paper has no per-offset
head-accuracy table (only a blog figure). Speculative verification of span drafts is a
deferred emission-path idea, orthogonal to TUL's claims.

### Coconut

**Title:** Training Large Language Models to Reason in a Continuous Latent Space  
**Authors:** Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian (Meta FAIR)  
**Year:** 2024  
**arXiv:** [2412.06769](https://arxiv.org/abs/2412.06769)  
**TUL uses:** the latent as an ATTENDED POSITION (thoughts enter as previous positions in
the sequence and later tokens read them through the KV cache) — TUL's slots are positions
for the same reason; loss masked on latent thoughts. Also the warning: without the
language curriculum a slot with only downstream token loss learns nothing (14.4 vs No-CoT
16.5 GSM8k).

### CODI / CCoT (latent chain-of-thought)

**Titles:** CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
(Shen et al., 2025, [2502.21074](https://arxiv.org/abs/2502.21074)); Compressed Chain of
Thought (Cheng, Van Durme, 2024, [2412.13171](https://arxiv.org/abs/2412.13171))  
**TUL uses:** the finding that an untargeted latent step needs an intermediate target
(CODI without distillation 24.5 vs 43.7); CCoT: gold hidden states at punctuation
positions (5–10% of a chain) decode the chain losslessly, and feeding the LAST-layer
state back as the next input ≈ no thoughts (use a middle layer). Both inform TUL's
warm-up arms and the choice not to feed `h_i` back as an input.

### Reasoning with Latent Thoughts (looped transformers)

**Title:** Reasoning with Latent Thoughts: On the Power of Looped Transformers  
**Authors:** Saunshi, Dikkala, Li, Kumar, Reddi (Google)  
**Year:** 2025  
**arXiv:** [2502.17416](https://arxiv.org/abs/2502.17416)  
**TUL uses:** depth via looping beats params for reasoning at equal FLOPs ((12×2) 34.3 vs
(24×1) 29.3 math) with worse PPL; middle-loop (prelude/coda around the loop) is better.
Loops hit every token there; per-idea depth (TUL's C1) is untested.

### Latent Reasoning via Sentence Embedding Prediction

**Title:** Latent Reasoning via Sentence Embedding Prediction  
**Authors:** Hwang et al.  
**Year:** 2025  
**arXiv:** [2505.22202](https://arxiv.org/abs/2505.22202)  
**TUL uses:** the closest match at GPT-2 scale — a slot with no next-token loss trained
only by the CE of the step it must produce; one state reconstructs a 6–11 token step at
98.5–100% exact match through a full AR decoder. Also: the gap to CoT widens with model
size, which they name as the reason to move the objective into pretraining.

### Large Concept Models / SONAR

**Titles:** Large Concept Models: Language Modeling in a Sentence Representation Space
(Meta FAIR, 2024, [2412.08821](https://arxiv.org/abs/2412.08821)); SONAR
([2308.11466](https://arxiv.org/abs/2308.11466))  
**TUL uses:** the counter-example — MSE regression onto a fixed sentence embedding fails
(Table 3/4), the frozen SONAR decoder sees one vector and no context, fidelity collapses
past ~250 chars per unit (hence a hard span cap). Table 7: training the decoder on a
NOISED latent lifted AutoBLEU 79.5 → 88.0 (a control TUL keeps).

### CoCoMix

**Title:** LLM Pretraining with Continuous Concepts  
**Authors:** Tack et al. (Meta)  
**Year:** 2025  
**arXiv:** [2502.08524](https://arxiv.org/abs/2502.08524)  
**TUL uses:** Fig 6(d) as the ablation template — loss-only / mechanism-only / both /
neither; each alone near-null, only the pair gains. TUL's arms A0/A2/A4/A1 are that 2×2.
Fig 6b: regressing onto the concept loses.

### Sentence VAEs and posterior collapse

**Titles:** Generating Sentences from a Continuous Space (Bowman et al., 2015,
[1511.06349](https://arxiv.org/abs/1511.06349)); Lagging Inference Networks and Posterior
Collapse (He et al., 2019, [1901.05534](https://arxiv.org/abs/1901.05534)); Optimus (Li et
al., 2020, [2004.04092](https://arxiv.org/abs/2004.04092)); Fast Decoding with Discrete
Latent Variables (Kaiser et al., 2018, [1803.03382](https://arxiv.org/abs/1803.03382))  
**TUL uses:** the collapse recipe. Bowman Table 2: word dropout moves KL 0.01 → 20.9 nats;
the inputless decoder is 380 vs 119 PPL — TUL's token-state dropout is this tax. He §3.1:
the collapsed optimum is stable from init; Fig 5: measure usage with MI, not the loss —
TUL's `plan_nats` ablation. Optimus Fig 5: z as per-layer attendable memory beats
z-on-the-embedding by 0.5–0.8 nats/word. Kaiser Table 4: score reconstruction from the
TRUE code separately from end-to-end — oracle before predicted.

### Non-autoregressive / diffusion decoding

**Titles:** Non-Autoregressive NMT (Gu et al., 2017, [1711.02281](https://arxiv.org/abs/1711.02281));
LLaDA (Nie et al., 2025, [2502.09992](https://arxiv.org/abs/2502.09992)); Block Diffusion
(Arriola et al., 2025, [2503.09573](https://arxiv.org/abs/2503.09573)); Latent Diffusion for
Language Generation (Lovelace et al., 2023, [2212.09462](https://arxiv.org/abs/2212.09462))  
**TUL uses:** the price of parallel emission inside a unit. NAT: positional-only input ~2
BLEU, +4 with informative per-position input, +5 with AR-teacher distillation. BD3-LM
Table 3: PPL monotone in block size even with full-model diffusion and KV cache to past
blocks (AR 22.83 vs L'=4/8/16 = 28.23/29.83/30.60). LD4LG: a latent with no loss of its
own decoded by a pretrained AR decoder via cross-attention reaches Rouge-L 99.2. TUL
keeps AR emission inside the span; distillation from an AR teacher is a fallback if the
first-token position stays weak.

### Explorative Modeling (XM)

**Title:** Explorative Modeling  
**Authors:** Gladstone, Ji, Du  
**Year:** 2026  
**arXiv:** [2607.27372](https://arxiv.org/abs/2607.27372)  
**TUL uses:** the explanation of the `.` collapse — one-shot multi-target regression has
generative expressivity 1 and predicts the mean; MDLM XM-1 emits "the the the". Best-of-K
search over the latent is a deferred alternative to a warm-up loss.

### SpaceByte

**Title:** SpaceByte: Towards Deleting Tokenization from Large Language Modeling  
**Authors:** Slagle  
**Year:** 2024  
**arXiv:** [2404.14408](https://arxiv.org/abs/2404.14408)  
**TUL uses:** the closest published shape to a slot in one stream: global blocks run
only at the FIRST spacelike byte of a space/punctuation run (+BOS), their output is
truncated and residual-added at that same index, and the windowed local layers read it
through attention (Listing 1). Table 1 (1e19 FLOPs): 1.009 / 0.748 / 0.500 bpb
(PG-19 / arXiv / GitHub) vs SentencePiece 0.989 / 0.768 / 0.508; the fixed-stride
control loses 0.10 bpb on PG-19 — TUL's arm A5. Table 6: the global stack is billed at
1/6–1/8 of the byte rate. Cites ACT and Mixture-of-Depths only as generic layer
skipping — no loop at the boundary positions.

### AU-Net (Autoregressive U-Net)

**Title:** From Bytes to Ideas: Language Modeling with Autoregressive U-Nets  
**Authors:** Videau, Idrissi, Haziza, Wehrstedt, Copet, Teytaud, Lopez-Paz  
**Year:** 2025  
**arXiv:** [2506.14761](https://arxiv.org/abs/2506.14761)  
**TUL uses:** where the depth goes. Stage ≥2 keeps only the vector at each pretoken
boundary (the space BEFORE the word); Table 5: 75 % of layers at the coarse stages beats
50 % and 25 % (67.4 / 66.0 / 65.3 HellaSwag) and the byte stage stays at 3 layers to
1e22 FLOPs — depth in the slot loop, prelude/coda thin. Table 2: AU-Net-2 1B 69.9
HellaSwag at 3e21 vs BPE 70.2 at 4e21; TQA/MMLU lag at small scale. The unpool
(`hierarchical.py up()`) REPEATS the coarse vector over the following segment through
one of 16 per-offset linears and ADDS it to the byte skip stream, then 3 causal byte
layers; Table 4: boundary-only scatter (TUL's prefix route) ties at 2 stages (62.9 vs
63.5) and loses 5.4 at 3 — hence TUL's `bcast` arm, off by default. Sec 2.2: the split
must be "stable to rightward insertion" (= causal). Sec 6: byte-level models need
their own batch/LR scaling laws.

### Hierarchical Autoregressive Transformers (HAT)

**Title:** Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing for Robust, Adaptable Language Models  
**Authors:** Neitemeier, Deiseroth, Eichenberg, Balles  
**Year:** 2025  
**arXiv:** [2501.10322](https://arxiv.org/abs/2501.10322)  
**TUL uses:** the explicit-prefix decoder: the backbone output `p^i` is the FIRST
position of a 3–4 layer causal char decoder over word i+1 (Eq. 4). Table 1
(compute-matched vs 64k BPE at 1B / 3B / 7B): word accuracy 35.5 / 37.8 / 39.0 vs
35.3 / 37.7 / 39.2; LAMBADA +68 % relative at 7B; ARC −1..3. MegaByte's fixed 8-byte
split on the same architecture: −2.7 HellaSwag, −8.4 LAMBADA. Fig 3 is the metric
trap TUL avoids: a bigger char decoder raises byte accuracy but not word accuracy —
size the coda by whole-unit or first-token metrics, keep it ~2 % of params. TUL does
NOT copy its context-blind decoder (sees only `p^i` and the current word).

### STP / punc-STP — see §7 (STP, removed)

**TUL note:** the STP paper (2602.22617) has no boundary, no pretraining and no
decodability claim; punc-STP at `.;!?--\n` and the "next token ~80% decodable from the
boundary state" observation are MORPH's own (§7 notes). TUL carries punc-STP on the SLOT
trajectory as an arm (`tul.stp_lambda`), zero parameters.

---

## Quick Reference Table


| #   | Technique                              | Paper                                     | arXiv                                                                                                                                          |
| --- | -------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Parcae Loop                            | Prairie et al. (UCSD+Together, 2026)      | [2604.12946](https://arxiv.org/abs/2604.12946)                                                                                                 |
| 2   | Block-ELL Format (superseded)          | NVIDIA cuSPARSE (2021+)                   | [developer.nvidia.com](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/) |
| 2a  | MegaBlocks / STK                       | Gale et al. (Stanford, 2022)              | [2211.15841](https://arxiv.org/abs/2211.15841)                                                                                                 |
| 3   | CMS Topology                           | Original work — MORPH project             | —                                                                                                                                              |
| 4   | GLA Retention                          | Yang et al. (2023 / ICML 2024)            | [2312.06635](https://arxiv.org/abs/2312.06635)                                                                                                 |
| 4a  | Neural Memory (Titans) (removed)       | Behrouz, Zhong, Mirrokni (Google, 2025)   | [2501.00663](https://arxiv.org/abs/2501.00663)                                                                                                 |
| 5   | CCA                                    | Figliolia et al. (Zyphra, 2025)           | [2510.04476](https://arxiv.org/abs/2510.04476)                                                                                                 |
| 6   | CSA / HCA                              | DeepSeek-AI (2026)                        | [HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)                                                         |
| 7   | STP (removed)                          | Huang, LeCun, Balestriero (2026)          | [2602.22617](https://arxiv.org/abs/2602.22617)                                                                                                 |
| 8   | LeJEPA (removed)                       | Balestriero, LeCun (2025)                 | [2511.08544](https://arxiv.org/abs/2511.08544)                                                                                                 |
| 9   | SIGReg (removed)                       | Balestriero, LeCun (2025)                 | [2511.08544](https://arxiv.org/abs/2511.08544)                                                                                                 |
| 10  | Lorentz Embeddings                     | Nickel, Kiela (ICML 2018)                 | [1806.03417](https://arxiv.org/abs/1806.03417)                                                                                                 |
| 11  | Hybrid Embeddings                      | Gu, Sala, Gunel, Ré (ICLR 2019)           | [OpenReview](https://openreview.net/forum?id=HJxeWnCcF7)                                                                                       |
| 12  | CoPE (Clipped RoPE)                    | Li, Ren, Yuille, Wang (2026)              | [2602.05258](https://arxiv.org/abs/2602.05258)                                                                                                 |
| 13  | XSA                                    | Zhai (Apple, 2026)                        | [2603.09078](https://arxiv.org/abs/2603.09078)                                                                                                 |
| 14  | Residual Attention                     | Kimi Team (Moonshot AI, 2026)             | [2603.15031](https://arxiv.org/abs/2603.15031)                                                                                                 |
| 15  | SwiGLU                                 | Shazeer (Google, 2020)                    | [2002.05202](https://arxiv.org/abs/2002.05202)                                                                                                 |
| 16  | MTP (deffered, requires greater scale) | Gloeckle et al. (Meta, 2024)              | [2404.19737](https://arxiv.org/abs/2404.19737)                                                                                                 |
| 17  | STE Ternary (BitNet b1.58)             | Ma et al. (Microsoft, 2024)               | [2402.17764](https://arxiv.org/abs/2402.17764)                                                                                                 |
| 18  | ReMoE                                  | Wang, Zhu, Chen (Tsinghua, 2025)          | [2412.14711](https://arxiv.org/abs/2412.14711)                                                                                                 |
| 19  | PEER                                   | Lample et al. (FAIR, 2019)                | [1907.05242](https://arxiv.org/abs/1907.05242)                                                                                                 |
| 20  | mHC                                    | DeepSeek-AI (2025)                        | [2512.24880](https://arxiv.org/abs/2512.24880)                                                                                                 |
| 20a | JPmHC (Cayley HC)                      | Sengupta, Wang, Brunswic (JPMorgan, 2026) | [2602.18308](https://arxiv.org/abs/2602.18308)                                                                                                 |
| 20b | Hyper-Connections                      | Zhu et al. / ByteDance (2024)             | [2409.19606](https://arxiv.org/abs/2409.19606)                                                                                                 |
| 21  | Zyphra RSA (planned)                   | Washbourne et al. (Zyphra, 2026)          | [2605.05365](https://arxiv.org/abs/2605.05365)                                                                                                 |
| 22  | StarCoder2 Tokenizer                   | Lozhkov et al. (BigCode, 2024)            | [2402.19173](https://arxiv.org/abs/2402.19173)                                                                                                 |
| 23  | Nested Learning (removed)              | Behrouz et al. (NeurIPS 2025)             | [2512.24695](https://arxiv.org/abs/2512.24695)                                                                                                 |
| 24  | Poisson Depth Sampling                 | Prairie et al. (Parcae, 2026)             | [2604.12946](https://arxiv.org/abs/2604.12946)                                                                                                 |
| 25  | Attention Sinks (planned)              | Xiao et al. (MIT/Meta, 2023)              | [2309.17453](https://arxiv.org/abs/2309.17453)                                                                                                 |
| 26  | Value Shift                            | Figliolia et al. (Zyphra, 2025)           | [2510.04476](https://arxiv.org/abs/2510.04476)                                                                                                 |
| 27  | LLM-JEPA (removed)                     | Huang, LeCun, Balestriero (2025)          | [2509.14252](https://arxiv.org/abs/2509.14252)                                                                                                 |
| 28  | Lottery Ticket Hypothesis              | Frankle, Carbin (MIT, 2019)               | [1803.03635](https://arxiv.org/abs/1803.03635)                                                                                                 |
| 29  | AdEMAMix Optimizer                     | Pagliardini et al. (EPFL/Apple, 2024)     | [2409.03137](https://arxiv.org/abs/2409.03137)                                                                                                 |
| 30  | Token Superposition Training (TST)     | Peng, Gigant, Quesnelle (Nous, 2026)      | [2605.06546](https://arxiv.org/abs/2605.06546)                                                                                                 |
| 31  | Semantic Step Prediction (removed)     | Yuan (2026)                               | [2604.18464](https://arxiv.org/abs/2604.18464)                                                                                                 |
| 32  | BLT (TUL)                              | Pagnoni et al. (Meta FAIR, 2024)          | [2412.09871](https://arxiv.org/abs/2412.09871)                                                                                                 |
| 33  | MegaByte (TUL)                         | Yu et al. (Meta AI, 2023)                 | [2305.07185](https://arxiv.org/abs/2305.07185)                                                                                                 |
| 34  | H-Net (TUL)                            | Hwang et al. (CMU/Cartesia, 2025)         | [2507.07955](https://arxiv.org/abs/2507.07955)                                                                                                 |
| 35  | Block Transformer (TUL)                | Ho et al. (KAIST/GDM, 2024)               | [2406.02657](https://arxiv.org/abs/2406.02657)                                                                                                 |
| 36  | Dynamic Token Pooling (TUL)            | Nawrot et al. (2022)                      | [2211.09761](https://arxiv.org/abs/2211.09761)                                                                                                 |
| 37  | Hourglass (TUL)                        | Nawrot et al. (2021)                      | [2110.13711](https://arxiv.org/abs/2110.13711)                                                                                                 |
| 38  | Patch-Level Training (TUL)             | Shao, Meng, Zhou (2024)                   | [2407.12665](https://arxiv.org/abs/2407.12665)                                                                                                 |
| 39  | DeepSeek-V3 MTP module (TUL)           | DeepSeek-AI (2024)                        | [2412.19437](https://arxiv.org/abs/2412.19437)                                                                                                 |
| 40  | Future Lens (TUL)                      | Pal et al. (Northeastern, 2023)           | [2311.04897](https://arxiv.org/abs/2311.04897)                                                                                                 |
| 41  | Blockwise Parallel Decoding (TUL)      | Stern, Shazeer, Uszkoreit (2018)          | [1811.03115](https://arxiv.org/abs/1811.03115)                                                                                                 |
| 42  | Medusa (TUL, cited only)               | Cai et al. (2024)                         | [2401.10774](https://arxiv.org/abs/2401.10774)                                                                                                 |
| 43  | Coconut (TUL)                          | Hao et al. (Meta FAIR, 2024)              | [2412.06769](https://arxiv.org/abs/2412.06769)                                                                                                 |
| 44  | CODI (TUL)                             | Shen et al. (2025)                        | [2502.21074](https://arxiv.org/abs/2502.21074)                                                                                                 |
| 45  | CCoT (TUL)                             | Cheng, Van Durme (2024)                   | [2412.13171](https://arxiv.org/abs/2412.13171)                                                                                                 |
| 46  | Looped Transformers (TUL)              | Saunshi et al. (Google, 2025)             | [2502.17416](https://arxiv.org/abs/2502.17416)                                                                                                 |
| 47  | Sentence Embedding Prediction (TUL)    | Hwang et al. (2025)                       | [2505.22202](https://arxiv.org/abs/2505.22202)                                                                                                 |
| 48  | Large Concept Models (TUL, counter)    | Meta FAIR (2024)                          | [2412.08821](https://arxiv.org/abs/2412.08821)                                                                                                 |
| 48a | SONAR (TUL, supporting)                | Duquenne et al. (Meta, 2023)              | [2308.11466](https://arxiv.org/abs/2308.11466)                                                                                                 |
| 49  | CoCoMix (TUL)                          | Tack et al. (Meta, 2025)                  | [2502.08524](https://arxiv.org/abs/2502.08524)                                                                                                 |
| 50  | Sentence VAE (TUL)                     | Bowman et al. (2015)                      | [1511.06349](https://arxiv.org/abs/1511.06349)                                                                                                 |
| 51  | Lagging Inference / collapse (TUL)     | He et al. (2019)                          | [1901.05534](https://arxiv.org/abs/1901.05534)                                                                                                 |
| 52  | Optimus (TUL)                          | Li et al. (Microsoft, 2020)               | [2004.04092](https://arxiv.org/abs/2004.04092)                                                                                                 |
| 53  | Latent Transformer / discrete codes (TUL) | Kaiser et al. (Google, 2018)           | [1803.03382](https://arxiv.org/abs/1803.03382)                                                                                                 |
| 54  | Non-Autoregressive NMT (TUL)           | Gu et al. (2017)                          | [1711.02281](https://arxiv.org/abs/1711.02281)                                                                                                 |
| 55  | LLaDA (TUL)                            | Nie et al. (2025)                         | [2502.09992](https://arxiv.org/abs/2502.09992)                                                                                                 |
| 56  | Block Diffusion / BD3-LM (TUL)         | Arriola et al. (2025)                     | [2503.09573](https://arxiv.org/abs/2503.09573)                                                                                                 |
| 57  | Latent Diffusion for Language (TUL)    | Lovelace et al. (2023)                    | [2212.09462](https://arxiv.org/abs/2212.09462)                                                                                                 |
| 58  | Explorative Modeling (TUL)             | Gladstone, Ji, Du (2026)                  | [2607.27372](https://arxiv.org/abs/2607.27372)                                                                                                 |
| 59  | SpaceByte (TUL)                        | Slagle (2024)                             | [2404.14408](https://arxiv.org/abs/2404.14408)                                                                                                 |
| 60  | AU-Net (TUL)                           | Videau et al. (Meta FAIR, 2025)           | [2506.14761](https://arxiv.org/abs/2506.14761)                                                                                                 |
| 61  | Hierarchical AT (TUL)                  | Neitemeier et al. (Aleph Alpha, 2025)     | [2501.10322](https://arxiv.org/abs/2501.10322)                                                                                                 |


