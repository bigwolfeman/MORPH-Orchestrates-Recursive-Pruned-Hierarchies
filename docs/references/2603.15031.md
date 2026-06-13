# Attention Residuals (Technical Report)

- **Authors:** Kimi Team (Moonshot AI)
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2603.15031
- **MORPH uses:** The per-head learned scalar alpha that additively combines the current layer's attention output with a depth-weighted residual from earlier layer outputs, providing stable gradient flow and bounded hidden-state magnitude as depth scales.

---

## ATTENTION RESIDUALS 

TECHNICAL REPORT OF ATTENTION RESIDUALS 

## **Kimi Team** 

`https://github.com/MoonshotAI/Attention-Residuals` 

## **ABSTRACT** 

Residual connections [12] with PreNorm [60] are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer’s contribution [27]. We propose _Attention Residuals (AttnRes)_ , which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, inputdependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce _Block AttnRes_ , which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead. 

Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks. 

**==> picture [419 x 245] intentionally omitted <==**

**----- Start of picture text -----**<br>
Output Output Output<br>w α w α<br>MoE<br>MoE K V w<br>w α Q MoE α<br>| ie d l | w gp ii|le== 0<br>Attention Attention w<br>| EE w n Attention 0. α<br>w α<br>MoE MoE w<br>AttnRes Op  ( α ) MoE α<br>w α<br>Hi Attention el EE [= = w<br>Attention Attention α<br>w α Block  n -1<br>| ... C8 0 1 +<br>... Block  n -2<br>· [··]<br>Embedding Embedding Embedding<br>(a) Standard Residuals (b) Full Attention Residuals (c) Block Attention Residuals<br>**----- End of picture text -----**<br>


Figure 1: Overview of Attention Residuals. **(a)** Standard Residuals: standard residual connections with uniform additive accumulation. **(b)** Full AttnRes: each layer selectively aggregates all previous layer outputs via learned attention weights. **(c)** Block AttnRes: layers are grouped into blocks, reducing memory from _O_ ( _Ld_ ) to _O_ ( _Nd_ ). 

Attention Residuals 

TECHNICAL REPORT 

## **1 Introduction** 

Standard residual connections [12] are the _de facto_ building block of modern LLMs [35, 51, 9]. The update _**h** l_ = _**h** l−_ 1 + _fl−_ 1( _**h** l−_ 1) is widely understood as a _gradient highway_ that lets gradients bypass transformations via identity mappings, enabling stable training at depth. Yet residuals also play a second role that has received less attention. Unrolling the recurrence shows that every layer receives the same uniformly-weighted sum of all prior layer outputs; residuals define how information aggregates across depth. Unlike sequence mixing and expert routing, which now employ learnable input-dependent weighting [53, 20, 9], this depth-wise aggregation remains governed by fixed unit weights, with no mechanism to selectively emphasize or suppress individual layer contributions. 

In practice, PreNorm [60] has become the dominant paradigm, yet its unweighted accumulation causes hidden-state magnitudes to grow as _O_ ( _L_ ) with depth, progressively diluting each layer’s relative contribution [27]. Early-layer information is buried and cannot be selectively retrieved; empirically, a significant fraction of layers can be pruned with minimal loss [11]. Recent efforts such as scaled residual paths [54] and multi-stream recurrences [72] remain bound to the additive recurrence, while methods that do introduce cross-layer access [36, 56] are difficult to scale. The situation parallels the challenges that recurrent neural networks (RNNs) faced over the sequence dimension before attention mechanism provided an alternative. 

We observe a formal duality between depth-wise accumulation and the sequential recurrence in RNNs. Building on this duality, we propose **Attention Residuals (AttnRes)** , which replaces the fixed accumulation _**h** l_ = _i_ _**[v]**[i]_ with _**h** l_ = 2 _i[α][i][→][l][·]_ _**[ v]**[i]_[,][where] _[ α][i][→][l]_[are][ softmax][ attention weights computed from a single learned pseudo-query] _**w** l ∈_ R _[d]_ per layer. This lightweight mechanism enables selective, content-aware retrieval across depth with only one _d_ -dimensional vector per layer. Indeed, standard residual connections and prior recurrence-based variants can all be shown to perform depth-wise _linear_ attention; AttnRes generalizes them to depth-wise softmax attention, completing for depth the same linear-to-softmax transition that proved transformative over sequences (§6.2, §6.1). 

In standard training, Full AttnRes adds negligible overhead, since the layer outputs it requires are already retained for backpropagation. At scale, however, activation recomputation and pipeline parallelism are routinely employed, and these activations must now be explicitly preserved and communicated across pipeline stages. We introduce _Block AttnRes_ to maintain efficiency in this regime: layers are partitioned into _N_ blocks, each reduced to a single representation via standard residuals, with cross-block attention applied only over the _N_ block-level summaries. This brings both memory and communication down to _O_ ( _Nd_ ), and together with infrastructure optimizations (§4), Block AttnRes serves as a drop-in replacement for standard residual connections with marginal training cost and negligible inference latency overhead. 

Scaling law experiments confirm that AttnRes consistently outperforms the baseline across compute budgets, with Block AttnRes matching the loss of a baseline trained with 1 _._ 25 _×_ more compute. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on 1.4T tokens. Analysis of the resulting training dynamics reveals that AttnRes mitigates PreNorm dilution, with output magnitudes remaining bounded across depth and gradient norms distributing more uniformly across layers. On downstream benchmarks, our final model improves over the baseline across all evaluated tasks. 

## **Contributions** 

- **Attention Residuals.** We propose AttnRes, which replaces fixed residual accumulation with learned softmax attention over depth, and its scalable variant Block AttnRes that reduces memory and communication from _O_ ( _Ld_ ) to _O_ ( _Nd_ ). Through a unified structured-matrix analysis, we show that standard residuals and prior recurrence-based variants correspond to depth-wise _linear_ attention, while AttnRes performs depth-wise softmax attention. 

- **Infrastructure for scale.** We develop system optimizations that make Block AttnRes practical and efficient at scale, including cross-stage caching that eliminates redundant transfers under pipeline parallelism and a two-phase inference strategy that amortizes cross-block attention via online softmax [31]. The resulting training overhead is marginal, and the inference latency overhead is less than 2% on typical inference workloads. 

- **Comprehensive evaluation and analysis.** We validate AttnRes through scaling law experiments, component ablations, and downstream benchmarks on a 48B-parameter model pre-trained on 1.4T tokens, demonstrating consistent improvements over standard residual connections. Training dynamics analysis further reveals that AttnRes mitigates PreNorm dilution, yielding bounded hidden-state magnitudes and more uniform gradient distribution across depth. 

2 

Attention Residuals 

TECHNICAL REPORT 

## **2 Motivation** 

**Notation.** Consider a batch of input sequences with shape _B × T × d_ , where _B_ is the batch size, _T_ is the sequence length, and _d_ is the hidden dimension. For clarity, we write formulas for a single token: _**h** l ∈_ R _[d]_ denotes the hidden state entering layer _l_ , where _l ∈{_ 1 _, . . . , L}_ is the layer index and _L_ is the total number of layers. The token embedding is _**h**_ 1. The function _fl_ represents the transformation applied by layer _l_ . In Transformer models, we treat each self-attention or MLP as an individual _layer_ . 

## **2.1 Training Deep Networks via Residuals** 

**Residual Learning.** Residual learning [12] proves to be a critical technique in training deep networks as it allows gradients to bypass transformations. Specifically, each layer updates the hidden state as: 

**==> picture [102 x 11] intentionally omitted <==**

Expanding this recurrence, the hidden state at layer _l_ is the sum of the embedding and all preceding layer outputs: _l−_ 1 _**h** l_ = _**h**_ 1 + _i_ =1 _[f][i]_[(] _**[h]**[i]_[)][.][The key insight behind residual connections is] _[ identity mapping]_[:][each layer preserves a direct] path for both information and gradients to flow unchanged. During back-propagation, the gradient with respect to an intermediate hidden state is: 

**==> picture [125 x 32] intentionally omitted <==**

Expanding this product yields **I** plus higher-order terms involving the layer Jacobians _∂fj/∂_ _**h** j_ . The identity term is always preserved, providing a direct gradient path from the loss to any layer regardless of depth. 

**Generalizing Residuals.** While effective, the fixed unit coefficients in the residual update treat every layer’s contribution uniformly, offering no mechanism to adapt the mixing across depth. Highway networks [45] relax this by introducing learned element-wise gates: 

**==> picture [168 x 11] intentionally omitted <==**

where _**g** l ∈_ [0 _,_ 1] _[d]_ interpolates between the transformation and the identity path. More generally, both are instances of a weighted recurrence _**h** l_ = _αl ·_ _**h** l−_ 1 + _βl · fl−_ 1( _**h** l−_ 1), with residual setting _αl_ = _βl_ =1 and Highway setting _αl_ =1 _−_ _**g** l, βl_ = _**g** l_ . 

**Limitations.** Whether fixed or gated, both approaches share a fundamental constraint: each layer can only access its immediate input _**h** l−_ 1, a single compressed state that conflates all earlier layer outputs, rather than the individual outputs themselves. This entails several limitations: (1) _no selective access_ : different layer types (e.g., attention vs. MLP) receive the same aggregated state, despite potentially benefiting from different weightings; (2) _irreversible loss_ : information lost through aggregation cannot be selectively recovered in deeper layers; and (3) _output growth_ : later layers learn increasingly larger outputs to gain influence over the accumulated residual, which can destabilize training. These limitations motivate a mechanism that lets each layer selectively aggregate information from all preceding layers. 

## **3 Attention Residuals: A Unified View of Time and Depth** 

The limitations discussed above are reminiscent of similar bottlenecks in sequence modeling, suggesting that we seek similar solutions for the depth dimension. 

**The Duality of Time and Depth.** Like RNNs over time, residual connections compress all prior information into a single state _**h** l_ over depth. For sequence modeling, the Transformer improved upon RNNs by replacing recurrence with attention [3, 52], allowing each position to selectively access all previous positions with data-dependent weights. We propose the same methodology for depth: 

**==> picture [308 x 30] intentionally omitted <==**

_l−_ 1 where _αi→l_ are layer-specific attention weights satisfying _i_ =0 _[α][i][→][l]_[= 1][.][Unlike sequence length (which can reach] millions of tokens), network depth is typically modest ( _L <_ 1000), making _O_ ( _L_[2] ) attention over depth computationally feasible. We call this approach _Attention Residuals_ , abbreviated as _AttnRes_ . 

3 

Attention Residuals 

TECHNICAL REPORT 

## **3.1 Full Attention Residuals** 

The attention weights can be written as _αi→l_ = _ϕ_ ( _**q** l,_ _**k** i_ ) for a kernel function _ϕ_ : R _[d] ×_ R _[d] →_ R _≥_ 0, where _**q** l_ and _**k** i_ are query and key vectors [23, 70]. Different choices of _ϕ_ recover different residual variants (§6.2); we adopt _ϕ_ ( _**q** ,_ _**k**_ ) = exp ( _**q**[⊤]_ RMSNorm( _**k**_ ) ) [66] with normalization, yielding softmax attention over depth: 

**==> picture [286 x 29] intentionally omitted <==**

For each layer _l_ , we define: 

**==> picture [337 x 24] intentionally omitted <==**

where the query _**q** l_ = _**w** l_ is a layer-specific learnable vector in R _[d]_ . The RMSNorm inside _ϕ_ prevents layers with large-magnitude outputs from dominating the attention weights. The input to layer _l_ is then: 

**==> picture [273 x 30] intentionally omitted <==**

We call this form _full attention residuals_ . For each token, Full AttnRes requires _O_ ( _L_[2] _d_ ) arithmetic and _O_ ( _Ld_ ) memory to store layer outputs. Since depth is far smaller than sequence length, the arithmetic cost is modest. 

**Overhead.** The _O_ ( _Ld_ ) memory overlaps entirely with the activations already retained for backpropagation, so Full AttnRes introduces no additional memory overhead in vanilla training. At scale, however, activation recomputation and pipeline parallelism are widely adopted: layer outputs that would otherwise be freed and recomputed must now be kept alive for all subsequent layers, and under pipeline parallelism each must further be transmitted across stage boundaries. Both the memory and communication overhead then grow as _O_ ( _Ld_ ). 

**Blockwise optimization.** A deliberate design choice in Full AttnRes is that the _pseudo_ -query _**w** l_ is a learned parameter decoupled from the layer’s forward computation. This independence means that attention weights for any group of layers can be computed in parallel without waiting for their sequential outputs, and in particular permits grouping the _L_ layers into _N_ blocks of _S_ layers each and batching the attention computation within each block, reducing per-layer memory I/O from _O_ ( _Ld_ ) to _O_ (( _S_ + _N_ ) _d_ ) (we defer the detailed two-phase strategy to §4). Under current distributed training regimes, however, the dominant cost is not local memory bandwidth but cross-stage communication under pipeline parallelism: every layer output must still be transmitted between stages, and this _O_ ( _Ld_ ) communication overhead cannot be alleviated by local batching. This motivates the Block AttnRes variant introduced below, which reduces the number of cross-stage representations from _L_ to _N_ . We anticipate that future interconnect improvements will make the full _O_ ( _Ld_ ) communication practical, fully realizing the potential of Full AttnRes. 

## **3.2 Block Attention Residuals** 

We propose _Block Attention Residuals_ , which partitions the _L_ layers into _N_ blocks: within each block, the layer outputs are reduced to a single representation via summation, and across blocks, we apply full attention over only _N_ block-level representations and the token embedding. This reduces both memory and communication overhead from _O_ ( _Ld_ ) to _O_ ( _Nd_ ). 

**Intra-Block Accumulation.** Specifically, we divide the _L_ layers into _N_ blocks of _S_ = _L/N_ layers each, assuming _L_ is divisible by _N_ ; otherwise, the last block contains the remaining _L_ mod _N_ layers. Let _Bn_ denote the set of layer indices in block _n_ ( _n_ = 1 _, . . . , N_ ). To form a block, we sum all of its layer outputs: 

**==> picture [272 x 24] intentionally omitted <==**

We further denote _**b**[i] n_[as the partial sum over the first] _[ i]_[ layers in] _[ B][n]_[, so that] _**[ b]**[n]_[=] _**[ b]**[S] n_[.][When] _[ L]_[ is not divisible by] _[ N]_[,] the final partial sum is taken as the last block’s representation. As in Full AttnRes, the RMSNorm inside _ϕ_ prevents magnitude differences between complete blocks and partial sums from biasing the attention weights. 

4 

Attention Residuals 

TECHNICAL REPORT 

`1 def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: Linear, norm: RMSNorm) -> Tensor: 2 """ 3 Inter-block attention: attend over block reps + partial sum. 4 blocks: 5 N tensors of shape [B, T, D]: completed block representations for each previous block 6 partial_block: 7 [B, T, D]: intra-block partial sum (b_n^i) 8 """ 9 V = torch.stack(blocks + [partial_block]) # [N+1, B, T, D] 10 K = norm(V) 11 logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K) 12 h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V) 13 return h 14 15 def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]: 16 partial_block = hidden_states 17 # apply block attnres before attn 18 # blocks already include token embedding 19 h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm) 20 21 # if reaches block boundary, start new block 22 # block_size counts ATTN + MLP; each transformer layer has 2 23 if self.layer_number % (self.block_size // 2) == 0: 24 blocks.append(partial_block) 25 partial_block = None 26 27 # self-attention layer 28 attn_out = self.attn(self.attn_norm(h)) 29 partial_block = partial_block + attn_out if partial_block is not None else attn_out 30 31 # apply block attnres before MLP 32 h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm) 33 34 # MLP layer 35 mlp_out = self.mlp(self.mlp_norm(h)) 36 partial_block = partial_block + mlp_out 37 38 return blocks, partial_block` 

Figure 2: PyTorch-style pseudo code for Block Attention Residuals. `block_attn_res` computes softmax attention over block representations using a learned pseudo-query _**w** l_ ; `forward` is a single-layer pass that maintains `partial_block` ( _**b**[i] n_[, intra-block] residual) and `blocks` ([ _**b**_ 0 _, . . . ,_ _**b** n−_ 1], inter-block history). 

**Inter-Block Attention.** In Full AttnRes, the input to layer _l_ is computed by attending over all outputs up to _fl−_ 1( _**h** l−_ 1). The block-wise variant replaces these individual outputs with block representations, defining _**b**_ 0 = _**h**_ 1 so that the token embedding is always included as a source. For the _i_ -th layer in block _n_ , the value matrix is: 

**==> picture [368 x 25] intentionally omitted <==**

Keys and attention weights follow Eq. 3 and Eq. 2. The input of the very first layer of the network is the token embeddings, i.e. _**b**_ 0 = _**h**_ 1. In each block, the first layer receives the previous block representations and the token embeddings, and the subsequent layers additionally attend to the partial sum _**b**[i] n[−]_[1] . The final output layer aggregates all _N_ block representations. Fig. 2 provides PyTorch-style pseudocode for Block AttnRes. 

**Efficiency.** Since each layer now attends over _N_ block representations rather than _L_ individual outputs, memory reduces from _O_ ( _L_ ) to _O_ ( _N_ ) and computation from _O_ ( _L_[2] ) to _O_ ( _N_[2] ). The block count _N_ interpolates between two extremes: _N_ = _L_ recovers Full AttnRes, while _N_ = 1 reduces to standard residual connections with the embedding isolated as _**b**_ 0. Empirically, we find that _N ≈_ 8 recovers most of the benefit across model scales, requiring only eight stored hidden states per token (see § 5). 

Beyond memory and computation, the block structure also benefits inference latency: block boundaries define the dispatch granularity for the blockwise optimization described in §3, and the fixed block count _N_ bounds the KV cache size. The parallel inter-block results are merged with the sequential intra-block partial sums via online softmax [31], preserving exact equivalence (§4). 

## **4 Infrastructure Design** 

Block AttnRes introduces additional system challenges compared to standard residual connections. For large-scale model training, block representations must be propagated across pipeline stages, causing heavy communication in a 

5 

Attention Residuals 

TECHNICAL REPORT 

**==> picture [449 x 124] intentionally omitted <==**

**----- Start of picture text -----**<br>
VIRTUAL STAGE 0 VIRTUAL STAGE 1<br>RANK 0 [  b 0 ] [ ] 1 2 + [  b 1 ,  b 2 ] [ ] 1 2<br>RANK 1 [  b 0 ] [  b 1 ] i 1 2 + [  b 1 ,  b 2 ] [  b 3 ] - 1 2<br>RANK 2 [  b 0 ,  b 1 ] [ ] 1 2 + [  b 2 ,  b 3 ] [ ] 1 2<br>Cn Ca<br>RANK 3 [  b 0 ,  b 1 ] [  b 2 ] 1 2 + [  b 2 ,  b 3 ] [  b 4 ] 1 2<br>5 C-<br>**----- End of picture text -----**<br>


Figure 3: Cache-based pipeline communication example with 4 physical ranks and 2 virtual stages per rank, where hatched boxes denote end of AttnRes blocks. Numbers indicate micro-batch indices. Each rank caches previously received blocks; stage transitions only transmit incremental blocks (+[ _**b**_ 1 _,_ _**b**_ 2]) instead of the full history. 

naïve implementation. During inference, repeated access to accumulated block representations increases latency, while long-context prefilling amplifies the memory cost of caching block representations. We address these challenges with cross-stage caching in training, and with a two-phase computation strategy together with a memory-efficient prefilling scheme in inference. 

## **4.1 Training** 

For small-scale training, AttnRes adds a tiny computation overhead and no extra memory usage, as the activations need to be saved for backpropagation regardless. Under large-scale distributed training, pipeline parallelism poses the primary infrastructure challenge for AttnRes. Full AttnRes requires all _L_ layer outputs to be transmitted across stages; Block AttnRes reduces this to _N_ block representations, and the optimizations below further minimize the remaining overhead. 

**Pipeline communication.** With standard residual connections, pipeline parallelism [18] transfers a fixed-size hidden state between adjacent stages, independent of pipeline depth. Block AttnRes requires all accumulated block representations at each stage for inter-block attention, and naïvely transmitting the full history at every transition incurs redundant communication. 

Consider an interleaved pipeline schedule [33] with _P_ physical stages and _V_ virtual stages per physical stage. For simplicity, assume each physical stage produces on average _Np_ block representations of dimension _d_ per token.[1] With _C_ = _PV_ total chunks (each physical stage in each virtual stage), the _j_ -th chunk accumulates _jNp_ blocks. Naïvely transmitting all accumulated blocks at every transition incurs per-token communication cost: 

**==> picture [325 x 31] intentionally omitted <==**

**Cross-stage caching.** Since each physical stage processes multiple virtual stages in succession, we can eliminate this redundancy by caching blocks locally: blocks received during earlier virtual stages remain in local memory and need not be re-transmitted. The first virtual stage ( _v_ = 1) has no cache and accumulates normally; for _v ≥_ 2, each transition conveys only the _∼PNp_ incremental blocks accumulated since the receiver’s corresponding chunk in the previous virtual stage. Total communication reduces to: 

**==> picture [338 x 37] intentionally omitted <==**

Caching reduces peak per-transition cost from _O_ ( _C_ ) to _O_ ( _P_ ), a _V ×_ improvement that enables full overlap with computation during steady-state 1F1B. The backward pass benefits from the same scheme. Fig. 3 illustrates this optimization with _P_ =4 and _V_ =2: for the second virtual stage, caching eliminates 6 redundant block transmissions. 

> 1In practice, block boundaries need not align with physical stage boundaries. For example, in Fig. 3, each block spans two physical stages, so only every other transition involves a newly completed block. 

6 

Attention Residuals 

TECHNICAL REPORT 

**Algorithm 1:** Two-phase computation for block _n_ 

**Input:** Pseudo queries _{_ _**w** l}l∈Bn_ , block representations _{_ _**b**_ 0 _, . . . ,_ _**b** n−_ 1 _}_ `/* Phase 1: Parallel inter-block attention */ 1` **Q** _←_ [ _**w** l_ ] _l∈Bn_ `//` [ _S, d_ ] `2` **K** _,_ **V** _←_ [ _**b**_ 0; _. . ._ ; _**b** n−_ 1] `//` [ _n, d_ ] `3` _{_ _**o**_[(1)] _l , m_[(1)] _l , ℓ_[(1)] _l }l∈Bn ←_ ATTNWITHSTATS( **Q** _,_ **K** _,_ **V** ) `// Return LSE 4 /* Phase 2: Sequential intra-block attention + Online` softmax `merge */ 5` _i ←_ 0 `6` **for** _l ∈Bn_ **do** `7` **if** _i_ = 0 **then** `8` _**h** l ←_ _**o**_[(1)] _l /ℓ_[(1)] _l_ `// Inter-block only 9` **else** `10` _**o**_[(2)] _l , m_[(2)] _l , ℓ_[(2)] _l ←_ ATTNWITHSTATS( _**w** l,_ _**b**[i] n[,]_ _**[ b]**[i] n_[)] `// Intra-block 11` _ml ←_ max( _m_[(1)] _l , m_[(2)] _l_ ) _l −ml_ _**o**_[(1)] _l_ + _e[m] l_[(2)] _−ml_ _**o**_[(2)] _l_ `12` _**h** l ←[e] e[m][m]_[(1)] _l_[(1)] _−ml ℓ_[(1)] _l_ + _e[m] l_[(2)] _−ml ℓ_[(2)] _l_ `// Online softmax merge 13` _i ← i_ + 1 `14` _**b**[i] n[←]_ _**[b]**[i] n[−]_[1] + _fl_ ( _**h** l_ ) `// Update partial sum;` _**b**_[0] _n_[:=] **[ 0]** `15` **return** _{_ _**h** l}l∈Bn_ 

**Memory overhead.** With cross-stage caching, each block is stored exactly once across all _V_ virtual stages, which becomes negligible relative to standard per-layer activation cache. Crucially, the per-layer activation footprint remains identical to standard architectures, as activation checkpointing eliminates all inter-block attention intermediates, and the checkpointed input _**p** l_ matches the memory size of the hidden state _**h** l_ it replaces. 

In terms of wall-clock time, Block AttnRes adds negligible training overhead when pipeline parallelism is not enabled; under pipeline parallelism, the measured end-to-end overhead is less than 4%. 

## **4.2 Inference** 

The two-phase computation strategy described below applies to both Full and Block AttnRes: in either case, layers are grouped into blocks of size _S_ , with Phase 1 batching the inter-block queries and Phase 2 handling sequential intra-block lookback. For Full AttnRes, this reduces per-layer I/O from _O_ ( _Ld_ ) to _O_ (( _S_ + _N_ ) _d_ ) (detailed derivation shown in Appendix B); Block AttnRes further reduces the stored representations from _L_ to _N_ , since each block is compressed into a single vector. In what follows, we focus on Block AttnRes and detail the two-phase computation strategy together with a sequence-sharded prefilling scheme for long-context inputs. 

**Two-phase computation strategy.** The layer-wise attention computation of Block AttnRes resembles autoregressive decoding, where block representations serve as a shared KV cache reused across layers. A naïve implementation computes the attention residual at every layer, each requiring a full pass over all preceding blocks, resulting in _O_ ( _L · N_ ) memory accesses. Since the pseudo-query vectors are decoupled from the forward computation (§3), all _S_ = _L/N_ queries within a block can be batched into a single matrix multiplication, amortizing memory access from _S_ reads to 1. 

Algorithm 1 instantiates a two-phase computation strategy exploiting this property. 

- **Phase 1** computes inter-block attention for all _S_ layers simultaneously via a single batched query against the cached block representations, returning both outputs and softmax statistics (max and log-sum-exp). This amortizes the memory access cost, reducing reads from _S_ times to just once per block. 

- **Phase 2** computes intra-block attention sequentially for each layer using the evolving partial sum, then merges with Phase 1 outputs through online softmax [31]. Because the online-softmax merge is elementwise, this phase naturally admits kernel fusion with surrounding operations, further reducing I/O overhead. 

With the two-phase design, Phase 2 preserves an I/O footprint similar to that of standard residual connections, whereas the main additional cost arises from Phase 1 inter-block attention. Because these inter-block reads are amortized across 

7 

Attention Residuals 

TECHNICAL REPORT 

all layers in a block through batching, the total per-layer memory access cost remains only ( _[N] S_[+ 3)] _[d]_[ reads and][ 2] _[d]_ writes (Table 1). This is substantially lower than the residual-stream I/O of prior residual generalizations such as (m)HC under typical settings. In practice, Phase 1 can also partially overlap with the computation of the first layer in the block, further reducing its wall-clock impact. As a result, the end-to-end inference latency overhead is less than 2% on typical inference workloads. 

Table 1: Memory access cost per token per layer incurred by the residual mechanism under each scheme. The internal I/O of the layer function _fl_ is excluded. For AttnRes, both Full and Block variants use the two-phase inference schedule described in Appendix B; amortized costs are averaged over _N_ layers within a block. Typical values: _L_ =128, _N_ =8, _S_ = _L/N_ =16, _m_ =4. 

||Operation|Read|Write|Total I/O||
|---|---|---|---|---|---|
|||||Symbolic|Typical|
|Standard Residuals|Residual Merge|2_d_|_d_|3_d_|3_d_|
||Compute**_α_**_l_,**_β_**_l_,**A**_l_|_md_|_m_2+2_m_|||
||Apply**_α_**_l_|_md_+_m_|_d_|||
|mHC (_m_streams)|Apply**_β_**_l_|_d_+_m_|_md_|(8_m_+2)_d_+2_m_2+4_m_|34_d_|
||Apply**A**_l_|_md_+_m_2|_md_|||
||Residual Merge|2_md_|_md_|||
|AttnRes<br>Full|Phase 1 (amortized)<br>Phase 2|(_N−_1)_d_<br>(_S−_1)_d_|_d_<br>_d_|(_S_+_N_)_d_|24_d_|
|Block|Phase 1 (amortized)<br>Phase 2|_N_<br>_S d_<br>3_d_<br>~~-~~|_d_<br>_d_|_N_<br>_S_ +5 _d_<br>~~(=~~)|5_._5_d_|



**Memory-efficient prefilling.** Storing block representations during prefilling requires _N · T · d_ elements, which incurs 15 GB of memory for a 128K-token sequence with 8 blocks. We mitigate this by sharding these representations along the sequence dimension across _P_ tensor-parallel devices, allowing Phase 1 to execute independently on local sequence shards. The Phase 2 online-softmax merge then integrates into the standard TP all-reduce communication path: the output is reduce-scattered, merged locally, and reconstructed via all-gather, naturally admitting kernel fusion with operations like RMSNorm. This reduces the per-device memory footprint to _N ·_ ( _T/P_ ) _· d_ —lowering the 128K-context example from 15 GB to roughly 1.9 GB per device. Combined with chunked prefill (e.g., 16K chunk size), the overhead further reduces to under 0.3 GB per device. 

## **5 Experiments** 

**Architecture Details.** Our architecture is identical to Kimi Linear [69], a Mixture-of-Experts (MoE) Transformer following the Moonlight [28] / DeepSeek-V3 [9] design, which interleaves Kimi Delta Attention (KDA) and Multi-Head Latent Attention (MLA) layers in a 3:1 ratio, each followed by an MoE feed-forward layer. The only modification is the addition of AttnRes to the residual connections; all other components (model depth, hidden dimensions, expert routing, and MLP structure) remain unchanged. AttnRes introduces only one RMSNorm and one pseudo-query vector _**w** l ∈_ R _[d]_ per layer, amounting to a negligible fraction of the total parameter count. Crucially, all pseudo-query vectors must be initialized to zero. This ensures that the initial attention weights _αi→l_ are uniform across source layers, which reduces AttnRes to an equal-weight average at the start of training and prevents training volatility, as we validated empirically. 

## **5.1 Scaling Laws** 

We sweep five model sizes (Table 2) and train three variants per size: a PreNorm baseline, Full AttnRes, and Block AttnRes with _≈_ 8 blocks. They are trained with an 8192-token context window and a cosine learning rate schedule. Within each scaling law size group, all variants share identical hyperparameters selected under the baseline to ensure fair comparison; this setup intentionally favors the baseline and thus makes the comparison conservative. Following standard practice, we fit power-law curves of the form _L_ = _A × C[−][α]_ [22, 15], where _L_ is validation loss and _C_ is compute measured in PFLOP/s-days. 

**Scaling Behavior.** Fig. 4 presents the fitted scaling curves. The Baseline follows _L_ = 1 _._ 891 _× C[−]_[0] _[.]_[057] , while Block AttnRes fits _L_ = 1 _._ 870 _× C[−]_[0] _[.]_[058] , and Full AttnRes fits _L_ = 1 _._ 865 _× C[−]_[0] _[.]_[057] . All three variants exhibit a similar slope, but AttnRes consistently achieves lower loss across the entire compute range. Based on the fitted curves, at 5.6 

8 

Attention Residuals 

TECHNICAL REPORT 

Table 2: Baseline vs Block AttnRes ( _N_ = 8) vs Full AttnRes vs mHC(-lite) [64]: Model configurations, Hyperparameters, and Validation Loss. 

|Validation Loss.|Validation Loss.|||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|
|# Act.<br>Params_†_|Tokens|_Lb_|_H_|_d_model|_d_ff|lr|batch size_‡_|Baseline|Val. Loss<br>Block AttnRes|Val. Loss<br>Full AttnRes|mHC(-lite)|
|194M|038.7B|12|12|0896|400|2_._99_×_10_−_3|192|1.931|1.909|**1.899**|1.906|
|241M|045.4B|13|13|0960|432|2_._80_×_10_−_3|256|1.895|1.875|1.874|**1.869**|
|296M|062.1B|14|14|1024|464|2_._50_×_10_−_3|320|1.829|1.809|**1.804**|1.807|
|436M|087.9B|16|16|1168|528|2_._20_×_10_−_3|384|1.766|1.746|**1.737**|1.747|
|528M|119.0B|17|17|1264|560|2_._02_×_10_−_3|432|1.719|1.693|**1.692**|1.694|



> _†_ Denotes the number of activated parameters in our MoE models, excluding embeddings. 

> _‡_ All models were trained with a context length of 8192. 

> _⋆ Lb_ = _L/_ 2 denotes the number of Transformer blocks. 

**==> picture [206 x 200] intentionally omitted <==**

**----- Start of picture text -----**<br>
iN Baseline: 1 . 891  × C [−] [0] [.] [057]<br>~NS~\,x. .AN - Full AttnRes:Block AttnRes:1 . 1865 . 870  × × C C [−] [0] [−][.] [0][057] [.] [058]<br>1.9 *2 EN wk<br>hs ~<br>hd SAWN~<br>A.<br>SA)<br>>~<br>AAS<br>A AY<br>A DOR FN><br>S,\\~ 1.25 ×<br>1.8 RY: A > ~<br>S~<br>A~<br>A~<br>NAAS *~ AN<br>A] ~<br>*~ AS ~ ~ iN<br>AS AS<br>IS ¥<br>AS .<br>1.7 AN AS w, iS . .<br>0.5 1 2 5<br>PFLOP/s-days<br>Loss<br>**----- End of picture text -----**<br>


Figure 4: Scaling law curves for Attention Residuals. Both Full and Block AttnRes consistently outperform the baseline across all scales. Block AttnRes closely tracks Full AttnRes, recovering most of the gain at the largest scale. 

PFLOP/s-days, Block AttnRes reaches 1.692 versus the Baseline’s 1.714, equivalent to a 1 _._ 25 _×_ compute advantage. The gap between Full and Block AttnRes narrows with scale, shrinking to just 0.001 at the largest size. We also list mHC(-lite) [64] in Table 2 for reference. Full AttnRes outperforms mHC, while Block AttnRes matches it at lower memory I/O per layer: 5 _._ 5 _d_ versus 34 _d_ for mHC with _m_ =4 streams (Table 1). 

## **5.2 Main Results** 

**Training recipe.** The largest models we study are based on the full Kimi Linear 48B configuration: 27 Transformer blocks (54 layers) with 8 out of 256 routed experts plus 1 shared expert, yielding 48B total and 3B activated parameters. This model applies Block AttnRes with 6 layers per block, producing 9 blocks plus the token embedding for a total of 10 depth-wise sources. 

We follow the same data and training recipe as the Kimi Linear 1.4T-token runs [69]: all models are pre-trained with a 4096-token context window, the Muon optimizer [28], and a WSD (Warmup–Stable–Decay) learning rate schedule [16], with a global batch size of 8M tokens. Training of the final model proceeds in two stages: (i) a WSD pre-training phase on 1T tokens, followed by (ii) a mid-training phase on _≈_ 400B high-quality tokens, following the annealing recipe of Moonlight [28]. 

After mid-training, we continue training with progressively longer sequence length of 32K tokens. Since our architecture uses hybrid KDA/MLA attention [69], where MLA operates without positional encodings (NoPE) [61], context extension requires no modifications such as YaRN [37] or attention temperature rescaling. 

9 

|Attention Residuals<br>TECHNICALREPORT<br>1_._2<br>1_._3<br>1_._4<br>1_._5<br>(a) Validation Loss<br>Baseline<br>Block AttnRes<br>0<br>5<br>10<br>15<br>(b) Output Magnitude<br>0<br>1<br>2<br>3<br>(c) Gradient Magnitude (_×_10_−_5)<br>~~<TC~~|Attention Residuals<br>TECHNICALREPORT<br>1_._2<br>1_._3<br>1_._4<br>1_._5<br>(a) Validation Loss<br>Baseline<br>Block AttnRes<br>0<br>5<br>10<br>15<br>(b) Output Magnitude<br>0<br>1<br>2<br>3<br>(c) Gradient Magnitude (_×_10_−_5)<br>~~<TC~~|Attention Residuals<br>TECHNICALREPORT<br>1_._2<br>1_._3<br>1_._4<br>1_._5<br>(a) Validation Loss<br>Baseline<br>Block AttnRes<br>0<br>5<br>10<br>15<br>(b) Output Magnitude<br>0<br>1<br>2<br>3<br>(c) Gradient Magnitude (_×_10_−_5)<br>~~<TC~~|Attention Residuals<br>TECHNICALREPORT<br>1_._2<br>1_._3<br>1_._4<br>1_._5<br>(a) Validation Loss<br>Baseline<br>Block AttnRes<br>0<br>5<br>10<br>15<br>(b) Output Magnitude<br>0<br>1<br>2<br>3<br>(c) Gradient Magnitude (_×_10_−_5)<br>~~<TC~~|Attention Residuals<br>TECHNICALREPORT<br>1_._2<br>1_._3<br>1_._4<br>1_._5<br>(a) Validation Loss<br>Baseline<br>Block AttnRes<br>0<br>5<br>10<br>15<br>(b) Output Magnitude<br>0<br>1<br>2<br>3<br>(c) Gradient Magnitude (_×_10_−_5)<br>~~<TC~~|
|---|---|---|---|---|
|20k<br>40k<br>60k<br>80k<br>100k<br>0<br>10<br>20|0||10<br>20||
|Step<br>Transformer Block Index|||Transformer Block Index||
|Figure 5: Training dynamics of Baseline and Block AttnRes. **(a)**Validation loss during training.|||**(b)**Each transformer block’s output||
|magnitude at the end of training. **(c)**Each transformer block’s gradient magnitude.|||||
|**Training dynamics.**<br>We compare the training dynamics of our final Baseline and Block AttnRes models over 1T|We compare the training dynamics of our final Baseline and Block AttnRes models over 1T||We compare the training dynamics of our final Baseline and Block AttnRes models over 1T||
|tokens in Fig.5.|||||
|• **Validation loss:** AttnRes achieves consistently lower validation loss throughout training, with the gap widening|||||
|during the decay phase and resulting in a notably lower final loss.|||||
|• **Output magnitude:** The Baseline suffers from the PreNorm dilution problem [|The Baseline suffers from the PreNorm dilution problem [60,27]: as hidden-state magnitudes||||
|grow monotonically with depth, deeper layers are compelled to learn increasingly large outputs from fixed-scale||||grow monotonically with depth, deeper layers are compelled to learn increasingly large outputs from fixed-scale|
|normalized inputs to remain influential. Block AttnRes confines this growth within each block, as selective aggregation|||||
|at block boundaries resets the accumulation, yielding a bounded periodic pattern.|||||
|• **Gradient magnitude:** With all residual weights fixed to 1, the Baseline provides no means of regulating gradient|||||
|flow across depth, leading to disproportionately large gradients in the earliest layers.|||The learnablesoftmaxweights||
|in Block AttnRes (Fig.8) introduce competition among sources for probability mass, resulting in a substantially more|||) introduce competition among sources for probability mass, resulting in a substantially more||
|uniform gradient distribution.|||||
|Table 3: Performance comparison of AttnRes with the baseline, both after the same pre-training recipe. Best per-row results are|||||
|**bolded**.|||||
|Baseline<br>AttnRes|||||
|MMLU<br>73.5|**74.6**||||
|MMLU-Pro<br>**52.2**|**52.2**||||
|GPQA-Diamond<br>36.9|**44.4**||||
|_General_<br>BBH<br>76.3|**78.0**||||
|ARC-Challenge<br>64.6|**65.7**||||
|HellaSwag<br>83.2|**83.4**||||
|TriviaQA<br>69.9|**71.8**||||
|GSM8K<br>81.7|**82.4**||||
|MGSM<br>64.9|**66.1**||||
|_Math & Code_<br>Math<br>53.5<br>CMath<br>84.7|**57.1**<br>**85.1**||||
|HumanEval<br>59.1|**62.2**||||
|MBPP<br>72.0|**73.9**||||
|_Chinese_<br>CMMLU<br>82.0<br>C-Eval<br>79.6|**82.9**<br>**82.5**||||



Table 3: Performance comparison of AttnRes with the baseline, both after the same pre-training recipe. Best per-row results are **bolded** . 

**Downstream performance.** Following the evaluation protocol of Kimi Linear [69], we assess both models across three areas (Table 3): 

10 

Attention Residuals 

TECHNICAL REPORT 

Table 4: Ablation on key components of AttnRes (16-layer model). 

**==> picture [452 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 . 770<br>Baseline (1.766) Baseline<br>—— 1 . 765 Full AttnRes<br>Variant Loss Block AttnRes<br>1 . 760<br>Baseline (PreNorm) 1.766 1.757<br>DenseFormer [36] 1.767 1 . 755 1.753<br>mHC [59] 1.747<br>1 . 750 1.748<br>AttnRes Full 1.737 1.746 1.746<br>—<br>w/ input-dependent query 1.731 1 . 745<br>w/ input-independent mixing 1.749<br>w/  sigmoid 1.741 1 . 740<br>w/o  RMSNorm 1.743<br>SWA ( W = 1 + 8) 1.764 1 . 735 Full AttnRes i.e. S=1 (1.737)<br>32 16 8 4 2<br>Block ( S = 4) 1.746<br>w/ multihead (H = 16 ) 1.752 Block size ( S )<br>w/o  RMSNorm 1.750<br>Validation loss<br>**----- End of picture text -----**<br>


Figure 6: Effect of block size on validation loss (16-layer model). 

- **Language understanding and reasoning** : MMLU [13], MMLU-Pro Hard [55], GPQA-Diamond [41], BBH [48], ARC-Challenge [6], HellaSwag [65], and TriviaQA [21]. 

- **Reasoning (Code and Math)** : GSM8K [7], MGSM [44], Math [25], CMath [14], HumanEval [5], and MBPP [1]. 

- **Chinese language understanding** : CMMLU [26] and C-Eval [19]. 

As shown in Table 3, Block AttnRes matches or outperforms the baseline on all benchmarks. The improvements are particularly pronounced on multi-step reasoning tasks such as GPQA-Diamond (+7.5) and Minerva Math (+3.6), as well as code generation such as HumanEval (+3.1), while knowledge-oriented benchmarks such as MMLU (+1.1) and TriviaQA (+1.9) also show solid gains. This pattern is consistent with the hypothesis that improved depth-wise information flow benefits compositional tasks, where later layers can selectively retrieve and build upon earlier representations. 

## **5.3 Ablation Study** 

We conduct ablation studies on the 16-head model from Table 2 to validate key design choices in AttnRes (Table 4). All models share identical hyperparameters and compute budget. 

**Comparison with prior methods.** We compare AttnRes against the PreNorm baseline (loss 1.766) and two representative methods that generalize residual connections. DenseFormer [36] grants each layer access to all previous outputs but combines them with fixed, input-independent scalar coefficients; it shows no gain over the baseline (1.767), highlighting the importance of input-dependent weighting. mHC [59] introduces input dependence through _m_ parallel streams with learned mixing matrices, improving to 1.747. AttnRes takes this further with explicit content-dependent selection via softmax attention: Full AttnRes achieves 1.737 and Block AttnRes 1.746, outperforming both methods with only a single query vector per layer. 

**Cross-layer access.** We compare three granularities of cross-layer access. Full AttnRes follows directly from the time–depth duality (§ 3), applying attention over all previous layers, and achieves the lowest loss (1.737). A simple way to reduce its memory cost is sliding-window aggregation (SWA), which retains only the most recent _W_ =8 layer outputs plus the token embedding; it improves over baseline (1.764) but falls well short of both Full and Block AttnRes, suggesting that selectively accessing distant layers matters more than attending to many nearby ones. 

Block AttnRes offers a better trade-off: with block size _S_ =4 it reaches 1.746 while keeping memory overhead constant per layer. Fig. 6 sweeps _S_ across the full spectrum from _S_ =1 (i.e. Full AttnRes) to increasingly coarse groupings. Loss degrades gracefully as _S_ grows, with _S_ =2 _,_ 4 _,_ 8 all landing near 1.746 while larger blocks ( _S_ =16 _,_ 32) move toward baseline. In practice, we fix the number of blocks to _≈_ 8 for infrastructure efficiency (§ 4). As future hardware alleviates memory capacity constraints, adopting finer-grained block sizes or Full AttnRes represents a natural pathway to further improve performance. 

11 

Attention Residuals 

TECHNICAL REPORT 

**==> picture [460 x 181] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 . 7 2.017 1.909 1.875 1.851 1.858 1.954 1.890 1.843 1.828 1.824<br>2<br>0 . 6 1.990 1.902 1.862 1.852 1.862 1.931 1.863 1.830 1.817 1.818<br>1 . 96<br>0 . 5 1.973 1.883 1.859 1.849 1.854 1.917 1.841 1.819 1.812 1.817 1 . 92<br>1 . 88<br>0 . 4 1.952 1.868 1.850 1.849 1.857 1.893 1.823 1.815 1.813 1.813<br>1 . 84<br>0 . 3 1.926 1.857 1.851 1.847 1.858 1.877 1.816 1.802 1.806 1.820<br>15 30 45 60 75 15 30 45 60 75<br>d model /Lb d model /Lb<br>(a) Baseline (b) Attention Residuals<br>b<br>H/L<br>**----- End of picture text -----**<br>


Figure 7: Architecture sweep under fixed compute ( _≈_ 6 _._ 5 _×_ 10[19] FLOPs, _≈_ 2 _._ 3 _×_ 10[8] active parameters). Each cell reports validation loss for a ( _d_ model _/Lb, H/Lb_ ) configuration, where _Lb_ = _L/_ 2 is the number of Transformer blocks; the star marks the optimum. 

**Component design.** We further ablate individual components of the attention mechanism: 

- **Input-dependent query.** A natural extension is to make the query input-dependent by projecting it from the current hidden state. This further lowers loss to 1.731, but introduces a _d × d_ projection per layer and requires sequential memory access during decoding, so we default to the learned query. 

- **Input-independent mixing.** We removed the query and key and replaced them with learnable, input-independent scalars to weigh previous layers, which hurts performance (1.749 vs. 1.737). 

- softmax **vs.** sigmoid **.** Replacing softmax with sigmoid degrades performance (1.741). We attribute this to softmax’s competitive normalization, which forces sharper selection among sources. 

- **Multihead attention.** We test per-head depth aggregation ( _H_ =16) on Block AttnRes, allowing different channel groups to attend to different source layers. This hurts performance (1.752 vs. 1.746), indicating that the optimal depth-wise mixture is largely uniform across channels: when a layer’s output is relevant, it is relevant as a whole. 

- RMSNorm **on keys.** Removing RMSNorm degrades both Full AttnRes (1.743) and Block AttnRes (1.750). For Full AttnRes, it prevents individual layers with naturally larger outputs from dominating the softmax. This becomes even more critical for Block AttnRes, as block-level representations accumulate over more layers and can develop large magnitude differences; RMSNorm prevents these from biasing the attention weights. 

## **5.4 Analysis** 

## **5.4.1 Optimal Architecture** 

To understand how AttnRes reshapes optimal architectural scaling, we perform a controlled capacity reallocation study under a fixed compute and parameter budget. Our central question is whether AttnRes alters the preferred depth–width–attention trade-off, and in particular, given its potential strength on the depth dimension, whether it favors deeper models compared to conventional Transformer design heuristics. To isolate structural factors directly coupled to depth, we fix the per-expert MLP expansion ratio based on internal empirical observations ( _d_ ff _/d_ model _≈_ 0 _._ 45). We further fix total training compute (FLOPs _≈_ 6 _._ 5 _×_ 10[19] ) and active parameters ( _≈_ 2 _._ 3 _×_ 10[8] ), ensuring that any performance variation arises purely from architectural reallocation rather than overall capacity differences. Under this constrained budget, we enumerate 25 configurations on a 5 _×_ 5 grid over _d_ model _/Lb ∈{_ 15 _,_ 30 _,_ 45 _,_ 60 _,_ 75 _}_ and _H/Lb ∈{_ 0 _._ 3 _,_ 0 _._ 4 _,_ 0 _._ 5 _,_ 0 _._ 6 _,_ 0 _._ 7 _}_ , where _Lb_ = _L/_ 2 is the number of Transformer blocks and _H_ the number of attention heads. The results are shown in Fig. 7. 

Both heatmaps exhibit a shared pattern: loss decreases with growing _d_ model _/Lb_ and shrinking _H/Lb_ , and both methods reach their optima at _H/Lb ≈_ 0 _._ 3. Despite this shared trend, AttnRes achieves a lower loss than the baseline in each of the 25 configurations, by 0 _._ 019–0 _._ 063. The most apparent difference lies in the location of the optimum: the baseline achieves its lowest loss at _d_ model _/Lb ≈_ 60 (1 _._ 847), whereas AttnRes shifts it to _d_ model _/Lb ≈_ 45 (1 _._ 802). Under a fixed 

12 

Attention Residuals 

TECHNICAL REPORT 

**==> picture [471 x 325] intentionally omitted <==**

**----- Start of picture text -----**<br>
Full AttnRes, Pre-Attn Full AttnRes, Pre-MLP<br>Weight<br>1<br>2<br>3<br>4<br>5<br>6 0 . 8<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14 0 . 6<br>15<br>16<br>0 5 10 15 20 25 30 0 5 10 15 20 25 30<br>Source Index Source Index<br>Block AttnRes, Pre-Attn Block AttnRes, Pre-MLP 0 . 4<br>1<br>2<br>3<br>4<br>5<br>6<br>7 0 . 2<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15 0<br>16 og<br>0 1 2 3 4 5 6 7 8 0 1 2 3 4 5 6 7 8<br>Block Index Block Index<br>Figure 8: Depth-wise attention weight distributions for a 16-head model with full (top) and block (bottom) Attention Residuals,<br>averaged over tokens. The model has 16 attention and 16 MLP layers. Each row shows how the  l th attention (left) or MLP (right)<br>layer distributes weight over previous sources. Diagonal dominance indicates locality remains the primary information pathway,<br>while persistent weights on source 0 (embedding) and occasional off-diagonal concentrations reveal learned skip connections. Block<br>attention ( N = 8) recovers the essential structure with sharper, more decisive weight distributions.<br>Layer<br>Layer<br>**----- End of picture text -----**<br>


parameter budget, a lower _d_ model _/Lb_ corresponds to a deeper, narrower network, suggesting that AttnRes can exploit additional depth more effectively. We note that this preference for depth does not directly translate to a deployment recommendation, as deeper models generally incur higher inference latency due to their sequential computation [39]. Rather, this sweep serves as a diagnostic that reveals where AttnRes benefits most, and this depth preference can be factored into the architecture selection alongside inference cost. 

## **5.4.2 Analyzing Learned AttnRes Patterns** 

We visualize the learned weights _αi→l_ in Fig. 8 for the 16-head model (from Table 2) with both full and block ( _N_ =8) AttnRes. Each heatmap shows how the _l_ th attention or MLP layer (rows) allocates its attention over previous sources (columns), with pre-attention and pre-MLP layers shown separately. We highlight three key observations: 

- **Preserved locality.** Each layer attends most strongly to its immediate predecessor, yet selective off-diagonal concentrations emerge (e.g., layer 4 attending to early sources, layers 15–16 reaching back under the block setting), indicating learned skip connections beyond the standard residual path. 

- **Layer specialization.** The embedding _**h**_ 1 retains non-trivial weight throughout, especially in pre-attention layers. Pre-MLP inputs show sharper diagonal reliance on recent representations, while pre-attention inputs maintain broader receptive fields, consistent with attention routing information across layers and MLPs operating locally. 

- **Block AttnRes preserves structure.** Diagonal dominance, embedding persistence, and layer specialization all transfer from the full to the block variant, suggesting that block-wise compression acts as implicit regularization while preserving the essential information pathways. 

13 

Attention Residuals 

TECHNICAL REPORT 

Table 5: Comparison of residual update mechanisms. _Weight_ : whether the mixing coefficients are architecture-fixed, learned-static (fixed after training), or input-dependent (dynamic). _Source_ : which earlier representations layer _l_ can access. Normalization is omitted from most formulas for clarity. 

||**Method**||**Update rule**|**Update rule**|**Weight**|**Source**||
|---|---|---|---|---|---|---|---|
||_Single-state recurrence: layer l receives only_**_h_**_l−_1|||||||
||Residual [12]||**_h_**_l_ =**_h_**|**_h_**_l−_1+_fl−_1(**_h_**_l−_1)|Fixed|**_h_**_l−_1||
||ReZero [2]||**_h_**_l_ =**_h_**|**_h_**_l−_1+_αl · fl−_1(**_h_**_l−_1)|Static|**_h_**_l−_1||
||LayerScale [50]||**_h_**_l_ =**_h_**|**_h_**_l−_1+ diag(**_λ_**_l_)_· fl−_1(**_h_**_l−_1)|Static|**_h_**_l−_1||
||Highway [45]||**_h_**_l_ = (1|= (1_−_**_g_**_l_)_⊙_**_h_**_l−_1+**_g_**_l ⊙fl−_1(**_h_**_l−_1)|Dynamic|**_h_**_l−_1||
||DeepNorm [54]||**_h_**_l_ = Norm(|= Norm(_α_**_h_**_l−_1+_fl−_1(**_h_**_l−_1))|Fixed|**_h_**_l−_1||
||KEEL [4]||**_h_**_l_ = Norm(|= Norm(_α_**_h_**_l−_1+_fl−_1(Norm(**_h_**_l−_1)))|Fixed|**_h_**_l−_1||
||_Multi-state recurrence: layer l receives_||_receives m streams_|||||
||SiameseNorm [27]||**_h_**1<br>_l_ =Norm(**_h_**1<br>_l−_1+**_y_**_l−_1); **_h_**2<br>_l_ =**_h_**2<br>_l−_1+**_y_**_l−_1||Fixed|2 streams||
||HC/mHC [72,59]||**H**_l_ =**H**|**H**_l−_1**A**_l_+_fl−_1(**H**_l−_1**_α_**_l−_1)**_β_**_⊤_<br>_l−_1|Dynamic|_m_streams||
||DDL [67]||**H**_l_ = (|= (**I**_−βl_**_k_**_l_**_k_**_⊤_<br>_l_ )**H**_l−_1 +_βl_**_k_**_l_**_v_**_⊤_<br>_l_|Dynamic|_dv_ streams||
||_Cross-layer access: layer l can access individual earlier-layer outputs_|||_can access individual earlier-layer outputs_||||
||DenseNet [17]||**_h_**_l_ = ConvPool([|= ConvPool([**_h_**1; _f_1(**_h_**1); _. . ._; _fl−_1(**_h_**_l−_1)])|Static|[**_h_**1_, . . . ,_**_h_**_l−_1]||
||DenseFormer [36]||**_h_**_l_ =_α_|_α_0_→l_**_h_**1+<br>_l−_1<br>_i_=1 _αi→l fi_(**_h_**_i_)<br>>|Static|[**_h_**1_, . . . ,_**_h_**_l−_1]||
||MRLA [10]1||**_h_**_l_ =|_l−_1<br>_i_=1<br>ConvPool(_fl−_1(**_h_**_l−_1))<br>ConvPool(_fi_(**_h_**_i_)) Conv(_fi_(**_h_**_i_))<br>Dynamic<br>[**_h_**1_, . . . ,_**_h_**_l−_1]<br>=)||||
|Full2<br>AttnRes (ours)<br>Block3<br>1<br>~~I~~|||**_h_**_l ∝_<br>**_h_**_l ∝_<br>|_l−_1<br>_i_=0 _ϕ_(**_w_**_l,_**_k_**_i_)**_v_**_i_<br>_n−_1<br>_i_=0 _ϕ_(**_w_**_l,_**_k_**_i_)**_v_**_i_+_ϕ_(**_w_**_l,_**_k_**_j_<br>_n_)**_v_**_j_<br>_n_<br><br>~~x~~|Dynamic<br>Dynamic|[**_h_**1_, . . . ,_**_h_**_l−_1]<br>[**_b_**0_, . . . ,_**_b_**_n−_1_,_**_b_**_j_<br>_n_]||



> 1 ConvPool: pooling operation followed by convolution (channel projection). 2 _ϕ_ ( _**q** ,_ _**k**_ ) = exp ( _**q** ⊤_ RMSNorm( _**k**_ ) ) ; _**k** i_ = _**v** i_ ; _**v**_ 0 = _**h**_ 1, _**v** i≥_ 1 = _fi_ ( _**h** i_ ). softmax jointly normalized over all sources. 

> 3 Same _ϕ_ and normalization as Full; _**v** i_ = _**b** i_ , _**v** nj_[=] _**[ b]**[j] n_[.] 

## **6 Discussions** 

## **6.1 Sequence-Depth Duality** 

Residual connections propagate information over depth via a fixed recurrence _**h** l_ = _**h** l−_ 1 + _fl−_ 1( _**h** l−_ 1), much as RNNs propagate information over time. Test-Time Training (TTT) [46] formalizes the sequence side of this analogy (cf. Fast Weight Programmers [43, 32]), casting each recurrent step as gradient descent on a self-supervised loss: 

**W** _t_ = **W** _t−_ 1 _− η ∇ℓ_ ( **W** _t−_ 1; _**x** t_ ) _,_ (9) 

where a slow network parameterizes _ℓ_ and the state **W** is updated once per token. When _f_ is linear, this reduces to vanilla linear attention **S** _t_ = **S** _t−_ 1 + _**k** t_ _**v** t[⊤]_[.][The standard residual exhibits the same additive form along depth, with] _**[ h]**[l]_ serving as the state and each layer _fl_ acting as one “gradient step.” 

As noted by [4], this duality extends to richer variants (Table 5). Data-dependent gates on the sequence side [47, 63] correspond to Highway networks [45] on the depth side; the delta rule [42, 62, 69] corresponds to DDL [67]; and MRLA [10] mirrors GLA’s [63] gated linear attention. These methods all refine the recurrent update while remaining within the recurrence paradigm. AttnRes goes a step further and replaces depth-wise recurrence with direct cross-layer attention, just as Transformers replaced temporal recurrence with self-attention. Since the number of layers in current architectures remains well within the practical regime of softmax attention, we adopt vanilla depth-wise attention. Incorporating more expressive yet memory-efficient (e.g. linear-complexity) alternatives is a natural direction for future work. 

## **6.2 Residual Connections as Structured Matrices** 

The residual variants discussed above can all be viewed as weighted aggregations over previous layer outputs. We formalize this with a _depth mixing matrix_ **M** _∈_ R _[L][×][L]_ , where **M** _i→l_ is the weight that layer _l_ assigns to the output of layer _i_ . The variants differ in how these weights arise (fixed, learned, or input-dependent) and whether **M** is constrained to low rank or allowed to be dense. The semiseparable rank of **M** [8] offers a unified lens for comparing them. 

_l−_ 1 Concretely, the input to layer _l_ is _**h** l_ = 2 _i_ =0 **[M]** _[i][→][l]_ _**[ v]**[i]_[, where] _**[ v]**_[0][=] _**[ h]**_[1][(embedding) and] _**[ v]**[i]_[=] _[ f][i]_[(] _**[h]**[i]_[)][ for] _[ i][ ≥]_[1][.][Fig.][ 9] visualizes **M** for representative methods; we derive each below. 

14 

Attention Residuals 

TECHNICAL REPORT 

**==> picture [466 x 135] intentionally omitted <==**

Figure 9: Depth mixing matrices **M** for four residual variants ( _L_ =4; Block AttnRes uses block size _S_ =2). Highway is shown with scalar gates for clarity. AttnRes panels show unnormalized _ϕ_ scores; background colors group entries that share the same source (Full AttnRes) or the same source block (Block AttnRes). 

- Standard residual [12], _**h** l_ = _**h** l−_ 1 + _fl−_ 1( _**h** l−_ 1). Expanding gives _**h** l_ = 2 _li−_ =01 _**[v]**[i]_[, so] **[ M]** _[i][→][l]_[= 1][ for all] _[ i < l]_[ and] **[ M]** is an all-ones lower-triangular matrix: 

**==> picture [147 x 50] intentionally omitted <==**

- Highway [45], _**h** l_ = (1 _−gl_ ) _**h** l−_ 1 + _gl fl−_ 1( _**h** l−_ 1) (written here with scalar gates for clarity; the element-wise _l_ 

- extension is straightforward). Defining the carry product _γi[×] →l_[:=] II _j_ = _i_ +1[(1] _[ −][g][j]_[)][, the weights are] **[ M]**[0] _[→][l]_[=] _[ γ]_ 1 _[×] →l_ for the embedding and **M** _i→l_ = _gi_ +1 _γi[×]_ +1 _→l_[for] _[ i][ ≥]_[1][.][Since the cumulative products factor through scalar gates,] **[ M]** is 1-semiseparable [8], the same rank as the standard residual but with input-dependent weights. The weights sum to one by construction, making Highway a softmax-free depth-wise instance of stick-breaking attention [49]. 

- (m)HC [72, 59] maintain _m_ parallel streams **H** _l ∈_ R _[d][×][m]_ , updated via 

**==> picture [166 x 13] intentionally omitted <==**

where **A** _l ∈_ R _[m][×][m]_ is a learned transition matrix, _**α** l−_ 1 _∈_ R _[m]_ mixes streams into a single input for _fl−_ 1, and _**β** l−_ 1 _∈_ R _[m]_ distributes the output back across streams. Unrolling the recurrence gives the effective weight 

**==> picture [282 x 14] intentionally omitted <==**

where **A** _[×] i→j_[:=] II _jk_ = _i_ +1 **[A]** _[k]_[.][The] _[ m][×][m]_[ transitions render] **[ M]** _[ m]_[-semiseparable [][8][].][mHC [][59][,][ 64][] further constrains] each **A** _l_ to be doubly stochastic, stabilizing the cumulative products across depth. 

- Full AttnRes computes **M** _i→l_ = _αi→l_ via _ϕ_ ( _**w** l,_ _**k** i_ ) = exp ( _**w** l[⊤]_[RMSNorm(] _**[k]**[i]_[)] ) with normalization, where _**k** i_ = _**v** i_ are input-dependent layer outputs, yielding a dense, rank- _L_ **M** . 

- Block AttnRes partitions layers into _N_ blocks _B_ 1 _, . . . , BN_ . For sources _i_ in a completed earlier block _Bn_ , all share the block-level key/value _**b** n_ , so **M** _i→l_ = _αn→l_ for every _i ∈Bn_ . Within the current block, each layer additionally attends over the evolving partial sum _**b**[i] n[−]_[1] , introducing one extra distinct source per intra-block position. The effective rank of **M** therefore lies between _N_ and _N_ + _S_ (where _S_ is the block size), interpolating between standard residual ( _N_ =1) and Full AttnRes ( _N_ = _L_ ). 

**Practicality.** The structured-matrix perspective serves two purposes. First, it enables analytical insights that are not apparent from the recurrence form alone. The input-dependent **M** of AttnRes, for instance, reveals depth-wise attention sinks (§5.4.2), where certain layers consistently attract high weight regardless of input, mirroring the same phenomenon in sequence-wise attention [57]. Second, it informs new designs by exposing which properties of the kernel _ϕ_ matter. For example, when _ϕ_ decomposes as _ϕ_ ( _**q** ,_ _**k**_ ) = _φ_ ( _**q**_ ) _[⊤] φ_ ( _**k**_ ) for some feature map _φ_ [23], depth-wise attention collapses into a recurrence—precisely the structure underlying the MRLA–GLA and DDL–DeltaNet correspondences noted above. 

15 

Attention Residuals 

TECHNICAL REPORT 

**Prior Residuals as Depth-Wise Linear Attention** The structured-matrix perspective further relates to the sequencedepth duality by showing that existing residual variants are, in effect, instances of _linear_ attention over the depth axis. For example, the unrolled (m)HC weight **M** _i→l_ = _**β** i[⊤]_ **[A]** _[×] i_ +1 _→l_ _**[α]**[l]_[(Eq.][ 10][) admits a natural attention interpretation in] which _**α** l_ plays the role of a query issued by layer _l_ , _**β** i_ serves as a key summarizing the contribution of layer _i_ , and the cumulative transition **A** _[×] i_ +1 _→l_[acts as a depth-relative positional operator [][69][] governing the query–key interaction] across intervening layers. Notably, the _m_ parallel streams correspond to state expansion [40, 29] along the depth axis, expanding the recurrent state from _d_ to _d× m_ and thereby increasing the semiseparable rank of **M** . [58] show that replacing **A** _[×] i_ +1 _→l_[with the identity matrix still yields competitive performance, highlighting the role of state expansion.] Through this lens, methods like (m)HC thus act as depth-wise _linear_ attention with matrix-valued states, while AttnRes acts as depth-wise softmax attention. 

## **7 Related Work** 

**Normalization, Scaling, and Depth Stability.** The standard residual update _**h** l_ +1 = _**h** l_ + _fl_ ( _**h** l_ ) [12] presents a fundamental tension between _normalization placement_ and _gradient propagation_ . PostNorm [52] maintains bounded magnitudes but distorts gradients, as repeated normalization on the residual path compounds into gradient vanishing at depth [60]. PreNorm [34, 60] restores a clean identity path yet introduces unbounded magnitude growth: since _∥_ _**h** l∥_ grows as _O_ ( _L_ ), each layer’s relative contribution shrinks, compelling deeper layers to produce ever-larger outputs and limiting effective depth [27]. Subsequent work reconciles both desiderata via scaled residual paths [54], hybrid normalization [73], amplified skip connections [4], or learned element-wise gates [45] (see Table 5). AttnRes sidesteps this tension by replacing the additive recurrence with selective aggregation over individual earlier-layer outputs, avoiding both the cumulative magnitude growth of PreNorm and the repeated scale contraction of PostNorm. 

**Multi-State Recurrence.** All single-state methods above condition layer _l_ only on _**h** l−_ 1, from which individual earlier-layer contributions cannot be selectively retrieved. Several methods address this by widening the recurrence to multiple parallel streams: Hyper-Connections [72] and its stabilized variant mHC [59] maintain _m_ streams with learned mixing matrices; DDL [67] maintains a matrix state updated via a delta-rule erase-and-write mechanism; SiameseNorm [27] maintains two parameter-shared streams—one PreNorm and one PostNorm—to preserve identity gradients and bounded representations. While these methods alleviate information compression, they still condition on the immediate predecessor’s state; AttnRes is orthogonal, providing selective access to individual earlier-layer outputs while remaining compatible with any normalization or gating scheme. We discuss the formal connection to Hyper-Connections in § 6.2. 

**Cross-Layer Connectivity.** A separate line of work bypasses the single-state bottleneck by giving each layer direct access to individual earlier-layer outputs. The simplest approach uses static weights: DenseNet [17] concatenates all preceding feature maps; ELMo [38] computes a softmax-weighted sum of layer representations with learned scalar weights; DenseFormer [36] and ANCRe [68] assign learned per-layer scalar coefficients fixed after training. For input-dependent aggregation, MUDDFormer [56] generates position-dependent weights via a small MLP across four decoupled streams; MRLA [10] applies element-wise sigmoid gating over all previous layers, though its separable query–key product is closer to linear attention than softmax-based retrieval. Other methods trade full cross-layer access for more targeted designs: Value Residual Learning [71] accesses only a single earlier layer; LAuReL [30] augments the residual with low-rank projections over the previous _k_ activations; Dreamer [24] combines sequence attention with depth attention and sparse experts. AttnRes combines softmax-normalized, input-dependent weights with selective access to all preceding layers through a single _d_ -dimensional pseudo-query per layer, and introduces a block structure reducing cost from _O_ ( _L_[2] ) to _O_ ( _LN_ ). Cache-based pipeline communication and a two-phase computation strategy (§ 4) make Block AttnRes practical at scale with negligible overhead. 

## **Conclusion** 

Inspired by the duality between sequence and depth, we introduce AttnRes, which replaces fixed, uniform residual accumulation with learned, input-dependent depth-wise attention. We validate the method through ablation studies and scaling law experiments, showing that its gains persist across scales. Because Full AttnRes must access all preceding layer outputs at every layer, the memory footprint of cross-layer aggregation grows as _O_ ( _Ld_ ), which is prohibitive for large-scale models on current hardware. We therefore introduce Block AttnRes, which partitions layers into _N_ blocks and attends over block-level representations. Empirically, using about 8 blocks recovers most of the gains of Full AttnRes, while finer-grained blocking remains a promising direction as future hardware constraints relax. Together with cross-stage caching and a two-phase computation strategy, Block AttnRes is practical at scale, incurring only marginal training overhead and minimal inference overhead. 

16 

Attention Residuals 

TECHNICAL REPORT 

## **References** 

- [1] Jacob Austin et al. _Program Synthesis with Large Language Models_ . 2021. arXiv: `2108.07732 [cs.PL]` . URL: `https://arxiv.org/abs/2108.07732` . 

- [2] Thomas Bachlechner et al. _ReZero is All You Need: Fast Convergence at Large Depth_ . 2020. arXiv: `2003.04887 [cs.LG]` . URL: `https://arxiv.org/abs/2003.04887` . 

- [3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. _Neural Machine Translation by Jointly Learning to Align and Translate_ . 2016. arXiv: `1409.0473 [cs.CL]` . URL: `https://arxiv.org/abs/1409.0473` . 

- [4] Chen Chen and Lai Wei. _Post-LayerNorm Is Back: Stable, ExpressivE, and Deep_ . 2026. arXiv: `2601.19895 [cs.LG]` . URL: `https://arxiv.org/abs/2601.19895` . 

- [5] Mark Chen et al. _Evaluating Large Language Models Trained on Code_ . 2021. arXiv: `2107.03374 [cs.LG]` . URL: `https://arxiv.org/abs/2107.03374` . 

- [6] Peter Clark et al. “Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge”. In: _arXiv:1803.05457v1_ (2018). 

- [7] Karl Cobbe et al. _Training Verifiers to Solve Math Word Problems_ . 2021. arXiv: `2110.14168 [cs.LG]` . URL: `https://arxiv.org/abs/2110.14168` . 

- [8] Tri Dao and Albert Gu. “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”. In: _CoRR_ abs/2405.21060 (2024). DOI: `10.48550/ARXIV.2405.21060` . arXiv: `2405.21060` . URL: `https://doi.org/10.48550/arXiv.2405.21060` . 

- [9] DeepSeek-AI et al. _DeepSeek-V3 Technical Report_ . 2025. arXiv: `2412.19437 [cs.CL]` . URL: `https://arxiv. org/abs/2412.19437` . 

- [10] Yanwen Fang et al. _Cross-Layer Retrospective Retrieving via Layer Attention_ . 2023. arXiv: `2302.03985 [cs.CV]` . URL: `https://arxiv.org/abs/2302.03985` . 

- [11] Andrey Gromov et al. _The Unreasonable Ineffectiveness of the Deeper Layers_ . 2025. arXiv: `2403.17887 [cs.CL]` . URL: `https://arxiv.org/abs/2403.17887` . 

- [12] Kaiming He et al. _Deep Residual Learning for Image Recognition_ . 2015. arXiv: `1512.03385 [cs.CV]` . URL: `https://arxiv.org/abs/1512.03385` . 

- [13] Dan Hendrycks et al. _Measuring Massive Multitask Language Understanding_ . 2021. arXiv: `2009.03300 [cs.CY]` . URL: `https://arxiv.org/abs/2009.03300` . 

- [14] Dan Hendrycks et al. _Measuring Mathematical Problem Solving With the MATH Dataset_ . 2021. arXiv: `2103. 03874 [cs.LG]` . URL: `https://arxiv.org/abs/2103.03874` . 

- [15] Jordan Hoffmann et al. _Training Compute-Optimal Large Language Models_ . 2022. arXiv: `2203.15556 [cs.CL]` . URL: `https://arxiv.org/abs/2203.15556` . 

- [16] Shengding Hu et al. _MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies_ . 2024. arXiv: `2404.06395 [cs.CL]` . URL: `https://arxiv.org/abs/2404.06395` . 

- [17] Gao Huang et al. _Densely Connected Convolutional Networks_ . 2018. arXiv: `1608.06993 [cs.CV]` . URL: `https://arxiv.org/abs/1608.06993` . 

- [18] Yanping Huang et al. “GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism”. In: _Advances in NeurIPS_ . 2019. 

- [19] Yuzhen Huang et al. “C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models”. In: _Advances in NeurIPS_ 36 (2023), pp. 62991–63010. 

- [20] Robert A. Jacobs et al. “Adaptive Mixtures of Local Experts”. In: _Neural Computation_ 3.1 (1991), pp. 79–87. DOI: `10.1162/neco.1991.3.1.79` . 

- [21] Mandar Joshi et al. “Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension”. In: _arXiv preprint arXiv:1705.03551_ (2017). 

- [22] Jared Kaplan et al. _Scaling Laws for Neural Language Models_ . 2020. arXiv: `2001.08361 [cs.LG]` . URL: `https://arxiv.org/abs/2001.08361` . 

- [23] Angelos Katharopoulos et al. “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention”. In: _Proceedings of ICML_ . Ed. by Hal Daumé III and Aarti Singh. PMLR, 2020, pp. 5156–5165. URL: `https: //proceedings.mlr.press/v119/katharopoulos20a.html` . 

- [24] Jonas Knupp et al. _Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves_ . 2026. arXiv: `2601.21582 [cs.AI]` . URL: `https://arxiv.org/abs/2601.21582` . 

- [25] Aitor Lewkowycz et al. _Solving Quantitative Reasoning Problems with Language Models_ . 2022. arXiv: `2206. 14858 [cs.CL]` . URL: `https://arxiv.org/abs/2206.14858` . 

17 

Attention Residuals 

TECHNICAL REPORT 

- [26] Haonan Li et al. “CMMLU: Measuring massive multitask language understanding in Chinese”. In: _Findings of the Association for Computational Linguistics: ACL 2024_ . Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 11260–11285. DOI: `10 . 18653 / v1 / 2024 . findings - acl . 671` . URL: `https : / / aclanthology . org / 2024 . findings - acl.671/` . 

- [27] Tianyu Li et al. _SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm_ . 2026. arXiv: `2602.08064 [cs.LG]` . URL: `https://arxiv.org/abs/2602.08064` . 

- [28] Jingyuan Liu et al. _Muon is Scalable for LLM Training_ . 2025. arXiv: `2502.16982 [cs.LG]` . URL: `https: //arxiv.org/abs/2502.16982` . 

- [29] Brian Mak and Jeffrey Flanigan. _Residual Matrix Transformers: Scaling the Size of the Residual Stream_ . 2025. arXiv: `2506.22696 [cs.LG]` . URL: `https://arxiv.org/abs/2506.22696` . 

- [30] Gaurav Menghani, Ravi Kumar, and Sanjiv Kumar. _LAuReL: Learned Augmented Residual Layer_ . 2025. arXiv: `2411.07501 [cs.LG]` . URL: `https://arxiv.org/abs/2411.07501` . 

- [31] Maxim Milakov and Natalia Gimelshein. _Online normalizer calculation for softmax_ . 2018. arXiv: `1805.02867 [cs.PF]` . URL: `https://arxiv.org/abs/1805.02867` . 

- [32] Tsendsuren Munkhdalai et al. “Metalearned Neural Memory”. In: _ArXiv_ abs/1907.09720 (2019). URL: `https: //api.semanticscholar.org/CorpusID:198179407` . 

- [33] Deepak Narayanan et al. _Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM_ . 2021. arXiv: `2104.04473 [cs.CL]` . URL: `https://arxiv.org/abs/2104.04473` . 

- [34] Toan Q. Nguyen and Julian Salazar. “Transformers without Tears: Improving the Normalization of SelfAttention”. In: _Proceedings of IWSLT_ . Ed. by Jan Niehues et al. 2019. URL: `https : / / aclanthology . org/2019.iwslt-1.17/` . 

- [35] OpenAI et al. _GPT-4 Technical Report_ . 2024. arXiv: `2303.08774 [cs.CL]` . URL: `https://arxiv.org/abs/ 2303.08774` . 

- [36] Matteo Pagliardini et al. _DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging_ . 2024. arXiv: `2402.02622 [cs.CL]` . URL: `https://arxiv.org/abs/2402.02622` . 

- [37] Bowen Peng et al. “Yarn: Efficient context window extension of large language models”. In: _arXiv preprint arXiv:2309.00071_ (2023). 

- [38] Matthew E. Peters et al. “Deep Contextualized Word Representations”. In: _Proceedings of NAACL_ . 2018, pp. 2227–2237. URL: `https://aclanthology.org/N18-1202/` . 

- [39] Reiner Pope et al. _Efficiently Scaling Transformer Inference_ . 2022. arXiv: `2211.05102 [cs.LG]` . 

- [40] Zhen Qin et al. _HGRN2: Gated Linear RNNs with State Expansion_ . 2024. arXiv: `2404.07904 [cs.CL]` . 

- [41] David Rein et al. “Gpqa: A graduate-level google-proof q&a benchmark”. In: _First Conference on Language Modeling_ . 2024. 

- [42] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. “Linear Transformers Are Secretly Fast Weight Programmers”. In: _Proceedings of ICML_ . Ed. by Marina Meila and Tong Zhang. PMLR, 2021, pp. 9355–9366. URL: `https://proceedings.mlr.press/v139/schlag21a.html` . 

- [43] Jürgen Schmidhuber. “Learning to control fast-weight memories: An alternative to dynamic recurrent networks”. In: _Neural Computation_ 4.1 (1992), pp. 131–139. 

- [44] Freda Shi et al. _Language Models are Multilingual Chain-of-Thought Reasoners_ . 2022. arXiv: `2210.03057 [cs.CL]` . URL: `https://arxiv.org/abs/2210.03057` . 

- [45] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. _Highway Networks_ . 2015. arXiv: `1505.00387 [cs.LG]` . URL: `https://arxiv.org/abs/1505.00387` . 

- [46] Yu Sun et al. “Learning to (Learn at Test Time): RNNs with Expressive Hidden States”. In: _ArXiv_ abs/2407.04620 (2024). URL: `https://api.semanticscholar.org/CorpusID:271039606` . 

- [47] Yutao Sun et al. _Retentive Network: A Successor to Transformer for Large Language Models_ . 2023. arXiv: `2307.08621 [cs.CL]` . 

- [48] Mirac Suzgun et al. “Challenging big-bench tasks and whether chain-of-thought can solve them”. In: _arXiv preprint arXiv:2210.09261_ (2022). 

- [49] Shawn Tan et al. “Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study”. In: _Proceedings of ICLR_ . 2025. 

- [50] Hugo Touvron et al. _Going deeper with Image Transformers_ . 2021. arXiv: `2103.17239 [cs.CV]` . URL: `https: //arxiv.org/abs/2103.17239` . 

- [51] Hugo Touvron et al. _LLaMA: Open and Efficient Foundation Language Models_ . 2023. arXiv: `2302.13971 [cs.CL]` . 

18 

Attention Residuals 

TECHNICAL REPORT 

- [52] Ashish Vaswani et al. “Attention is All you Need”. In: _Advances in NeurIPS_ . Ed. by I. Guyon et al. Curran Associates, Inc., 2017. URL: `https://proceedings.neurips.cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf` . 

- [53] Ashish Vaswani et al. “Attention is All you Need”. In: _Advances in NeurIPS_ . Ed. by I. Guyon et al. Vol. 30. Curran Associates, Inc., 2017. URL: `https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf` . 

- [54] Hongyu Wang et al. _DeepNet: Scaling Transformers to 1,000 Layers_ . 2022. arXiv: `2203.00555 [cs.CL]` . URL: `https://arxiv.org/abs/2203.00555` . 

- [55] Yubo Wang et al. “Mmlu-pro: A more robust and challenging multi-task language understanding benchmark”. In: _Advances in NeurIPS_ 37 (2024), pp. 95266–95290. 

- [56] Da Xiao et al. “MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections”. In: _Proceedings of ICML_ . 2025. 

- [57] Guangxuan Xiao et al. “Efficient streaming language models with attention sinks”. In: _arXiv preprint arXiv:2309.17453_ (2023). 

- [58] Tian Xie. _Your DeepSeek mHC Might Not Need the “m”_ . Zhihu blog post. 2026. URL: `https://zhuanlan. zhihu.com/p/2010852389670908320` . 

- [59] Zhenda Xie et al. _mHC: Manifold-Constrained Hyper-Connections_ . 2026. arXiv: `2512.24880 [cs.CL]` . URL: `https://arxiv.org/abs/2512.24880` . 

- [60] Ruibin Xiong et al. _On Layer Normalization in the Transformer Architecture_ . 2020. arXiv: `2002.04745 [cs.LG]` . URL: `https://arxiv.org/abs/2002.04745` . 

- [61] Bowen Yang et al. _Rope to Nope and Back Again: A New Hybrid Attention Strategy_ . 2025. arXiv: `2501.18795 [cs.CL]` . URL: `https://arxiv.org/abs/2501.18795` . 

- [62] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. “Gated Delta Networks: Improving Mamba2 with Delta Rule”. In: _Proceedings of ICLR_ . 2025. URL: `https://openreview.net/forum?id=r8H7xhYPwz` . 

- [63] Songlin Yang et al. “Gated Linear Attention Transformers with Hardware-Efficient Training”. In: _Proceedings of ICML_ . PMLR, 2024. 

- [64] Yongyi Yang and Jianyang Gao. _mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations_ . 2026. arXiv: `2601. 05732 [cs.LG]` . URL: `https://arxiv.org/abs/2601.05732` . 

- [65] Rowan Zellers et al. “HellaSwag: Can a Machine Really Finish Your Sentence?” In: _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ . 2019. 

- [66] Biao Zhang and Rico Sennrich. “Root mean square layer normalization”. In: _Advances in NeurIPS_ 32 (2019). [67] Yifan Zhang et al. _Deep Delta Learning_ . 2026. arXiv: `2601.00417 [cs.LG]` . URL: `https://arxiv.org/ abs/2601.00417` . 

- [68] Yilang Zhang et al. _ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling_ . 2026. arXiv: `2602.09009 [cs.LG]` . URL: `https://arxiv.org/abs/2602.09009` . 

- [69] Yu Zhang et al. _Kimi Linear: An Expressive, Efficient Attention Architecture_ . 2025. arXiv: `2510.26692 [cs.CL]` . 

- [70] Shu Zhong et al. _Understanding Transformer from the Perspective of Associative Memory_ . 2025. arXiv: `2505. 19488 [cs.LG]` . URL: `https://arxiv.org/abs/2505.19488` . 

- [71] Zhanchao Zhou et al. “Value Residual Learning”. In: _Proceedings of ACL_ . Ed. by Wanxiang Che et al. Vienna, Austria, 2025, pp. 28341–28356. URL: `https://aclanthology.org/2025.acl-long.1375/` . 

- [72] Defa Zhu et al. _Hyper-Connections_ . 2025. arXiv: `2409.19606 [cs.LG]` . URL: `https://arxiv.org/abs/ 2409.19606` . 

- [73] Zhijian Zhuo et al. _HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization_ . 2025. arXiv: `2503.04598 [cs.CL]` . URL: `https://arxiv.org/abs/2503.04598` . 

19 

Attention Residuals 

TECHNICAL REPORT 

## **A Contributions** 

The authors are listed in order of the significance of their contributions, with those in project leadership roles appearing last. 

Guangyu Chen _[∗]_ Yu Zhang _[∗]_ Jianlin Su _[∗]_ Weixin Xu Siyuan Pan Yaoyu Wang Yucheng Wang Guanduo Chen Bohong Yin Yutian Chen Junjie Yan Ming Wei Y. Zhang Fanqing Meng Chao Hong Xiaotong Xie Shaowei Liu Enzhe Lu Yunpeng Tai 

Yanru Chen Xin Men Haiqing Guo Y. Charles Haoyu Lu Lin Sui Jinguo Zhu Zaida Zhou Weiran He Weixiao Huang Xinran Xu Yuzhi Wang Guokun Lai Yulun Du Yuxin Wu Zhilin Yang Xinyu Zhou 

> _∗_ Equal contribution 

20 

Attention Residuals 

TECHNICAL REPORT 

## **B Optimized Inference I/O for Full Attention Residuals** 

A naïve implementation of Full AttnRes scans all preceding layer outputs at every layer, so memory traffic scales linearly with depth. As noted in §4.2, however, the pseudo-query _**w** l_ is a learned parameter independent of both the input and the hidden state. We can therefore batch inter-block accesses across layers in a two-phase schedule, bringing total I/O well below the naïve bound. 

Note that the block partition introduced below is purely an inference scheduling device. Unlike Block AttnRes, it leaves the model architecture unchanged and does not replace per-layer sources with block summaries; it simply makes the amortization argument concrete. 

**Setup** Let the model have _L_ layers and hidden dimension _d_ , partitioned into _N_ contiguous blocks of size _S_ = _L/N_ . Inference proceeds one block at a time: Phase 1 jointly computes inter-block attention for all _S_ layers in the block against all preceding blocks, and Phase 2 walks through intra-block dependencies sequentially. 

## **Phase 1: Batched Inter-block Attention** 

Consider block _n_ with its _S_ layers. The queries _{_ _**w** l}l∈Bn_ are all known before execution begins, so the ( _n−_ 1) _S_ preceding key–value pairs need only be read once from HBM and reused across all _S_ queries. The read cost for block _n_ is therefore 

**==> picture [285 x 15] intentionally omitted <==**

where the factor of 2 accounts for both keys and values. Summing over all _N_ blocks and using _SN_ = _L_ : 

**==> picture [470 x 74] intentionally omitted <==**

in total. 

## **Phase 2: Sequential Intra-block Attention** 

Phase 1 covers all sources before the current block. Within the block, however, each layer depends on those before it, so these must be handled in order. Layer _t_ (1 _≤ t ≤ S_ ) reads _t−_ 1 intra-block key–value pairs at a cost of 2( _t−_ 1) _d_ . Summing over one block: 

**==> picture [469 x 50] intentionally omitted <==**

## **Total Amortized I/O per Layer** 

Summing both phases over all _N_ blocks: 

**==> picture [366 x 11] intentionally omitted <==**

Dividing by _L_ and using _SN_ = _L_ : 

**==> picture [406 x 36] intentionally omitted <==**

Batching inter-block reads thus brings per-layer I/O from _O_ ( _L_ ) down to _O_ ( _S_ + _N_ ). The schedule follows the same two-phase split as Block AttnRes: inter-block attention accounts for the bulk of the traffic, while sequential computation stays local within each block. 

21 

