# ZAYA1-8B Technical Report

- **Authors:** Robert Washbourne, Rishi Iyer, Tomas Figliolia, et al., Beren Millidge (Zyphra)
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2605.05365
- **MORPH uses:** The Markovian RSA (Recurrent Speculative Aggregation) test-time compute scheme: N parallel traces generated simultaneously, then recursively aggregated; each reasoning chunk operates on a fixed-size context window (Markovian: only the tail of the previous chunk is carried forward), enabling unbounded reasoning with constant KV memory. MORPH's outer loop is designed to support RSA harness deployment after RL training, currently deferred.

---

# ZAYA1-8B Technical Report 

Robert Washbourne[*] , Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, Sungyeon Yang, Pritish Yuvraj, Quentin Anthony, Yury Tokpanov, Xiao Yang, Ganesh Nanduru, Stephen Ebert, Praneeth Medepalli, Skyler Szot, Srivatsan Rajagopal, Alex Ong, Bhavana Mehta, Beren Millidge[*] 

## **Zyphra** 

San Francisco, CA 

> *Corresponding authors: rob@zyphra.com, beren@zyphra.com 

_**Abstract**_ **—** We present ZAYA1-8B, a reasoning-focused mixtureof-experts (MoE) model with 700M active and 8B total parameters, built on Zyphra’s MoE++ architecture. ZAYA1-8B’s core pretraining, midtraining, and supervised fine-tuning (SFT) were performed on a full-stack AMD compute, networking, and software platform. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, and remains competitive with substantially larger openweight reasoning models. ZAYA1-8B was trained from scratch for reasoning, with reasoning data included from pretraining onward using an answer-preserving trimming scheme. Post-training uses a four-stage RL cascade: reasoning warmup on math and puzzles; a 400-task RLVE-Gym curriculum; math and code RL with testtime compute traces and synthetic code environments built from competitive-programming references; and behavioral RL for chat and instruction following. We also introduce Markovian RSA, a test-time compute method that recursively aggregates parallel reasoning traces while carrying forward only bounded-length reasoning tails between rounds. In TTC evaluation, Markovian RSA raises ZAYA1-8B to 91.9% on AIME’25 and 89.6% on HMMT’25 while carrying forward only a 4K-token tail, narrowing the gap to much larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, and GPT-5-High. 

## I. INTRODUCTION 

In this paper, we introduce ZAYA1-8B, a 700M-active, 8B-total parameter mixture-of-experts (MoE) model. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, while remaining competitive with substantially larger open-weight reasoning models including OLMo-3.1-32B-Think, Nemotron-3-Nano-30BA3B, Mistral-Small-4-119B-2603, and Intellect-3-12A-106B (NVIDIA, 2025; Team et al., 2025c; Team, 2025a; Mistral AI, 2026). 

Moreover, using our test-time compute scheme, Markovian RSA, ZAYA1-8B narrows the gap on AIME’25 and HMMT’25 to substantially larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, Qwen3-235B-A22BThinking-2507, and GPT-5-High (Comanici et al., 2025; DeepSeek-AI, 2025c; Team, 2025b; OpenAI, 2025). These results suggest that competitive mathematical reasoning can be reached with under 1B active parameters when model architecture, reasoning-heavy training, verifiable RL, and testtime aggregation are co-designed. 

The system combines five design choices that we found important in practice: 

**Architecture:** ZAYA1-8B builds on Zyphra’s MoE++ architecture (Anthony et al., 2025), with three main changes relative to standard transformer MoE designs. First, ZAYA18B uses Compressed Convolutional Attention (CCA) (Figliolia et al., 2025), a FLOP- and memory-efficient attention variant that performs sequence mixing in a compressed latent space. Prior work showed that CCA performs well on perplexity and standard language modeling at small scale; ZAYA1-8B evaluates its behavior at larger scale and on more challenging reasoning and long-context tasks. Second, ZAYA1-8B uses the ZAYA1 router, which replaces the standard linear MoE router with a multi-layer MLP-based design, substantially increasing its expressiveness. In our experiments we find that increasing router capacity and expressiveness is a strong use of marginal parameters. A small number of router parameters controls a much larger number of expert parameters, and better routing decisions significantly reduce balancing instability and improve model quality. Third, ZAYA1-8B applies learned residual scaling to both the residual stream and the layer input at each block, which controls residual-norm growth through depth at negligible parameter and FLOP cost. 

**Reasoning-aware training across stages:** We designed ZAYA1-8B from scratch for reasoning. Motivated by evidence that including reasoning data during pretraining can produce gains that post-training alone does not recover (Akter et al., 2025), we include long chain-of-thought (CoT) data in all pretraining phases and during midtraining. To train on reasoning traces that exceed the pretraining context length, we introduce a novel answer-preserving trimming methodology, which truncates the tail of the reasoning trace while preserving the final answer, or drops the example if the answer alone does not fit. Unlike prior length-control methods that operate during inference or RL rollout generation (Khatri et al., 2025; Yang et al., 2025), AP-trimming is applied during training-data construction. 

**Cascaded reinforcement learning pipeline:** Post-training for ZAYA1-8B uses a four-stage RL cascade: reasoning warmup, a 400-task adaptive difficulty curriculum over the RLVE-Gym environment suite (Zeng et al., 2025b), math and code RL with test-time compute traces, and a final behavioral 

**==> picture [488 x 278] intentionally omitted <==**

**----- Start of picture text -----**<br>
LCB-v6<br>AIME'25 HMMT'25 Feb 25.02-25.05<br>94.6 92.5 75<br>74.1<br>92<br>94<br>93.1 72.5<br>92.3 90 89.6<br>91.9 88.3<br>92<br>88<br>90 +3.6 86 +6.9 70 69.2 68.7<br>83.9<br>84<br>88.0 82.5 +4.2<br>88 88.3 87.5 82 82.7<br>87.0<br>80 79.4 79.2 65<br>86 65.0<br>78<br>84 76 62<br>1<br>10<br>100<br>671<br>active params total params<br>ZAYA1-8B (single rollout) (0.7/8B) DeepSeek-R1-0528 (37/671B) Qwen3-Thinking-2507 (22/235B)<br>+ Markovian RSA boost Claude 4.5 Sonnet DeepSeek-V3.2 (37/671B)<br>Gemini-2.5 Pro GPT-5-High<br>Score (%)<br>Params (B)<br>**----- End of picture text -----**<br>


Fig. 1: ZAYA1-8B with Markovian RSA test-time compute vs. substantially larger reasoning models on AIME’25, HMMT’25, and LCB-v6. Hatched bars show the boost from Markovian RSA over single-rollout ZAYA1-8B. With 0.7B active parameters and the 40K/4K Markovian RSA configuration (Section VI-C), ZAYA1-8B reaches 91.9% on AIME’25 and 89.6% on HMMT’25, narrowing the gap to larger proprietary and open-weight reasoning models. ZAYA1-8B numbers (single-rollout and TTC) are evaluated in the Zyphra harness on the pre-behavioral checkpoint after math+code+TTC RL and before the final lightweight behavioral-RL polishing stage; comparator numbers are taken from official release materials (see Table XI for sources). The final behavioral stage targets chat style, instruction following, and preference behavior rather than math/code/TTC capability. 

RL stage. The cascade uses asynchronous PipelineRL (Piché et al., 2025; Khatri et al., 2025) with DPPO Binary-TV trustregion masking (Qi et al., 2026), Dr-GRPO sequence-level loss aggregation (Liu et al., 2024), MaxRL advantage estimation (Tajwar et al., 2026), and no KL regularization in the reward. Stable training required substantial precision, verifier, and data curation work, which we document throughout the report. 

**Test-time compute methods:** We introduce Markovian RSA, a novel test-time compute method that combines the recursive candidate-aggregation structure of RSA (Venkatraman et al., 2025) with the bounded-workspace principle of Markovian Thinking (Aghajohari et al., 2025). Markovian RSA turns long reasoning into staged batched inference: each stage generates _N_ candidates in parallel, each candidate has bounded decode length _β_ , and aggregation prefill depends only on _C_ carried-forward tails of length _τ_ , not on the full reasoning history. Crucially, we also integrate Markovian RSA into training: SFT data is constructed by reshuffling expertmodel rollouts into aggregation examples, and RL stages train both expert-model and policy-self-aggregation variants. The resulting model is trained for the Markovian RSA workflow 

at inference and we achieve substantial performance uplift by doing so. 

**AMD training stack:** Building on our prior work with AMD MI300X GPUs and AMD Pensando Pollara 400 networking for large-scale pretraining (Anthony et al., 2025), ZAYA1-8B was pretrained, midtrained, and supervised finetuned on this GPU/networking stack. This provides evidence that the stack can support sustained pretraining, longcontext midtraining, and supervised fine-tuning for an 8Btotal-parameter MoE reasoning model. We validate this stack at the ZAYA1-8B scale; validation for substantially larger models and broader parallelism regimes remains future work. 

The remainder of this report is organized as follows: Section II describes the ZAYA1-8B architecture. Section III describes pretraining, midtraining, and answer-preserving trimming. Section IV describes the SFT stage and RL cascade, including infrastructure, precision, optimizer, and stabilitymonitoring choices. Section V reports benchmark results and comparisons. Section VI describes our test-time compute approach. Section VII concludes with observations from training and open questions. 

2 

**==> picture [493 x 387] intentionally omitted <==**

**----- Start of picture text -----**<br>
HMMT'26 vs. Active Parameter Count<br>85<br>80 Qwen3-Next-80B-A3B-Think Total params<br>75 Intellect-3-12B-106B 8B<br>ZAYA1-8B Nemotron-3-Nano-A3-30B 30B<br>70<br>80B<br>Mistral-4-Small-6B-119B<br>65 119B<br>Qwen3.5-4B<br>60<br>0.7 1 2 3 4 6 10 20<br>Active parameters, B, log scale<br>AIME'26 vs. Active Parameter Count LCB-v6 vs. Active Parameter Count<br>70<br>92<br>Qwen3-Next-80B-A3B-Think<br>Qwen3-Next-80B-A3B-Think 68<br>90 ZAYA1-8B 66 ZAYA1-8B<br>Nemotron-3-Nano-A3-30B 64 Intellect-3-12B-106B<br>Nemotron-3-Nano-A3-30B<br>88<br>Mistral-4-Small-6B-119B 62<br>60 Mistral-4-Small-6B-119B<br>86<br>58<br>Intellect-3-12B-106B Qwen3.5-4B<br>84 Qwen3.5-4B 56<br>54<br>0.7 1 2 3 4 6 10 20 0.7 1 2 3 4 6 10 20<br>Active parameters, B, log scale Active parameters, B, log scale<br>HMMT26<br>AIME26 LCB-v6<br>**----- End of picture text -----**<br>


Fig. 2: Active-parameter scaling across HMMT’26, AIME’26, and LiveCodeBench-v6. ZAYA1-8B is shown at 0.7B active parameters and compared against larger open-weight and frontier models where available. Bubble area denotes total parameter count where available. 

## II. MODEL 

## _A. Architecture_ 

ZAYA1-8B uses an MoE architecture with three changes relative to contemporary MoE models: (1) CCA for the attention block, (2) the ZAYA1 router, and (3) residual scaling. In our ablations, these changes improve per-parameter perplexity relative to classical MoE architectures (Shazeer et al., 2016; Fedus et al., 2022) using MLA or GQA attention and a linear router (Dai et al., 2024). CCA also improves training speed relative to GQA and MLA and reduces prefill FLOPs while maintaining comparable KV-cache compression rates. 

_1) Compressed Convolutional Attention (CCA):_ CCA performs sequence mixing in a compressed latent space using a lightweight convolutional downprojector. This reduces compute requirements for training and prefill and reduces KVcache size for long-context decoding. CCA is competitive 

with attention variants such as MLA and GQA (Ainslie et al., 2023; DeepSeek-AI, 2025b). ZAYA1-8B’s reasoning and longcontext performance provides evidence that CCA remains effective at this scale and can support reasoning, in-context learning (ICL), and long-range recall. CCA also supports our long-context midtraining workloads at lower compute and communication cost, which was important for training ZAYA1-8B during midtraining and RL phases. Appendix C provides additional details. 

_2) ZAYA1 Router:_ We replace the standard linear router used in many large-scale MoE models with a more expressive router. First, we use an MLP in place of the linear router. Second, we mix the router representation with the previous layer’s routing representation using _Exponential Depth Averaging (EDA)_ , a variant of Depth-Weighted Averaging (Pagliardini et al., 2024). 

3 

|Property|ZAYA1-8B configuration|
|---|---|
|Architecture family|Decoder-only MoE Transformer, Zyphra MoE++|
|Active parameters|0.76B|
|Total parameters|8.4B|
|Transformer layers|40|
|Hidden dimension|2048|
|CCA query heads|8|
|KV heads|2|
|Head dimension|128|
|Attention variant|CCGQA with CCA preconditioner|
|Query compression|2_×_|
|KV-cache compression|8_×_ relative to full multi-head attention|
|Experts per MoE layer|16|
|Routing|Top-1, no residual expert|
|Expert FFN width|4096 pre-activation / 2048 post-activation|
|Router latent dimension|256|
|Position embeddings|50% RoPE on each head|
|Tokenizer|Gemma3 tokenizer, 262,272 vocabulary size|
|Primary training hardware|AMD MI300X with Pollara networking|



TABLE I: ZAYA1-8B model configuration. Exact parameter counts are shown; the rounded release convention refers to the model as 0.7B active and 8B total. Architectural constants follow the ZAYA1 base configuration used for pretraining and continued post-training. 

Fig. 3: ZAYA1-8B model architecture. Two of the three main architectural changes are shown here: CCA for the attention block and the ZAYA1 router. The ZAYA1 router replaces the linear router with an MLP-based router consisting of a down-projection, EDA, and a three-layer MLP. 

Given the residual stream input _xl ∈_ R _[B][×][S][×][D]_ , where _D_ is using a learned weight matrix _W_ down _∈_ R _[R][×][D]_ : the residual stream dimension, the ZAYA1 router first downprojects the residual stream to a smaller router dimension _R rl_ = _W_ down _xl ,_ 

**==> picture [157 x 9] intentionally omitted <==**

such that _rl ∈_ R _[B][×][S][×][R]_ . For ZAYA1-8B we set _R_ = 256. We then apply EDA, which combines the representation with that 

4 

of the previous layer using a learned coefficient _γ_ : 

**==> picture [161 x 10] intentionally omitted <==**

The EDA operation is followed by a three-layer MLP with GeLU activations to produce the final router scores _s ∈_ R _[B][×][S][×][E]_ , where _E_ is the number of experts: 

**==> picture [202 x 11] intentionally omitted <==**

The scores are then used to select experts through a top-k operation: 

**==> picture [169 x 11] intentionally omitted <==**

where _bl_ are learned bias-balancing vectors and topk selects the _k_ experts with the largest biased router scores for each token. In ZAYA1-8B, _k_ = 1, so (4) reduces to selecting arg max _e_ ( _sl,e_ + _bl,e_ ) for each token. The ZAYA1 router uses a bias-balancing scheme building on (DeepSeek-AI, 2025b). Routing biases are updated using a scheme inspired by proportional–integral–derivative (PID) controllers from classical control theory (Åström & Hägglund, 2006). The router enforces balancing across a global batch of expert choices. Our PID optimizer uses AdamW internally, where the error signal passed to the optimizer is the difference between the empirical routing probability distribution and the uniform distribution. Specifically, the gradient _∇bl,e_ , for expert _e_ at layer _l_ , is computed as: 

**==> picture [165 x 21] intentionally omitted <==**

where _pl,e_ is the actual fraction of tokens routed to expert _e_ in the current batch, and _E_ is the total number of experts. This gradient signal is then used by AdamW to update the bias terms, penalizing over-utilized experts and boosting under-utilized ones. This improved the convergence speed and stability of the PID loop relative to the classical DeepSeek implementation. 

In our experiments, the MLP router and EDA improve MoE performance and make balancing (Figure 4) and expert specialization easier. The additional MLP adds some FLOPs and parameters, but parameter-matched ablations show that the router is a strong target for marginal parameters compared with the experts or attention. The added router parameters and FLOPs remain small because the MLP operates in the downprojected latent space rather than in the full embedding dimension. Figure 4 illustrates the average balancing across layers from initialization of an experiment-sized model. Empirically, reduced time to convergence translated to increased recovery speed in the face of perturbations such as data distribution shifts throughout phases of training. This yields an improved router-load entropy convergence in the reported 1.8B ablation and reduced balancing failures in our training runs compared to linear routers. 

_3) ZAYA1 Residual Scaling:_ The final architectural change in ZAYA1-8B is residual scaling. We apply a learned bias _bl_ and gating coefficient _α ∈_ R _[D]_ both to the residual stream and to the output of each layer before the residual connection: 

Fig. 4: Normalized router-load entropy, averaged over MoE layers, as a function of training step from initialization of a 1.8B-parameter experimental model. For each global batch and layer, let _pi_ denote the fraction of routed tokens assigned to expert _i_ , with _E_ total experts. We report _H_ ( _p_ ) _/_ ln _E_ , where _H_ ( _p_ ) = _− Ei_ =1 _[p][i]_[ ln] _[ p][i]_[is][the][Shannon][entropy][of][the] empirical expert-load distribution. 

**==> picture [262 x 26] intentionally omitted <==**

Different gating coefficients and biases are applied to the residual stream and to the layer outputs. Residual scaling lets the model downweight parts of the residual stream and control how much prior residual information is retained. In our experiments, residual scaling provides similar benefits to Qwen’s attention gating scheme (Qiu et al., 2025), without the parameter or FLOP overhead of an explicit gating matrix. Residual scaling also helps control residual-norm growth through network depth, without observing any gradient vanishing. We initialize _α_ to ones and _β_ to zeros, as this initializes the model with default residual connections. Because residual scaling adds only 4 _× L × D_ parameters, its parameter and FLOP overhead are comparable to LayerNorm and are negligible. 

Beyond these architectural changes, we trained with 16 experts and a hidden-dimension expansion factor of 2. This relatively fine-grained expert configuration improved performance at fixed parameter count, consistent with prior work (Team et al., 2025b; DeepSeek-AI, 2025a; Dai et al., 2024; Tian et al., 2025). 

Unlike many contemporary MoEs, we trained with top-k equal to 1 and without residual experts (Rajbhandari et al., 2022; DeepSeek-AI, 2025b). In our experiments, the improved routing expressiveness of the ZAYA1 router and the resulting expert specialization make a residual expert unnecessary. FLOP-matched experiments also favored top-1 over higher top-k when using the ZAYA1 router. We hypothesize that the ZAYA1 router assigns more certain expert choices, with better expert specialization, so additional experts in parallel via top- 

5 

k are less useful. When larger values of _k_ are used, their contribution is further reduced by multiplication with the routing probability. ZAYA1-8B produces lower-entropy routing probabilities per token than linear routers, consistent with more confident routing. As a sanity check on expert redundancy, Appendix D measures within-layer expert subspace overlap for ZAYA1-8B and public MoE baselines. ZAYA1-8B is not an outlier toward higher expert overlap: its first-projection input overlap is 1 _._ 45 _×_ the random-subspace baseline, close to Qwen3-30B-A3B’s 1 _._ 48 _×_ , while its output-projection overlap is intermediate among the compared MoEs. For attention, we used CCGQA with a query compression rate of 2 _×_ and a KV compression rate of 8 _×_ . We applied RoPE (Su et al., 2023) to half the channels in each head, leaving the other half without position embeddings. ZAYA1-8B was trained with the Gemma3 tokenizer. 

Table I summarizes core architectural hyperparameters of the final release configuration. 

## III. PRETRAINING AND MIDTRAINING 

ZAYA1-8B was initialized from Zyphra’s ZAYA1 base architecture and trained through pretraining, context-extension midtraining, and SFT on an AMD MI300X cluster equipped with the AMD Pensando Pollara networking stack. Full details of the base-model pretraining system, hardware, checkpointing, context parallelism, and AMD-specific optimizer and kernel work are provided in (Anthony et al., 2025). 

Table II summarizes the main phases. Base pretraining used a broad web-crawl distribution with code, math, multilingual, and reasoning data mixed in progressively. The second base pretraining phase upweighted code, math, reasoning, and instruction-formatted data while still training at 4K context length. We then ran a reasoning-focused midtrain phase at 32K context for 1.2T tokens at a RoPE base frequency of 1M. This was followed by an SFT phase at 131K context for 660B tokens at a RoPE base frequency of 5M. We believe that training for a large number of tokens at longer contexts significantly improves the model’s native long-context capabilities and thus provides a stronger base for post-training and RL. The substantial reduction in prefill FLOPs we obtained through using CCA was instrumental in making this feasible at our compute scale. 

Table III reports coarse data categories for the reasoningfocused midtrain and SFT. Percentages are normalized over the nonzero mixture weights in the data cards; we report only category-level proportions rather than individual dataset names. To specialize the model for reasoning and provide as strong a base for RL as possible we utilized a very high fraction of long-CoT reasoning traces in the midtrain and SFT. 

For context extension, we used all-gather KV context parallelism with two ranks at 32K and eight ranks at 131K. CCA’s compressed KV representation kept activation and KVcache memory overhead low, while short asynchronous pointto-point exchanges handled the convolution and value-shift boundary conditions introduced by CCA. Across these phases, 

we trained with the Muon optimizer using AdamW RMS matching (Jordan et al., 2024; Liu et al., 2025). 

_A. Reasoning-aware pretraining and answer-preserving trimming_ 

Recent work suggests that introducing long chain-ofthought reasoning data during pretraining and midtraining, rather than only during post-training, can produce gains that subsequent fine-tuning does not recover (Akter et al., 2025). We follow this approach throughout ZAYA1-8B’s training pipeline: every pretraining and midtraining phase included long-CoT data and it was a majority of the mix for the midtraining phases. 

Including reasoning data at short pretraining contexts creates a practical challenge: reasoning traces from strong teacher models often exceed 10K tokens, with a long tail beyond 30K. At the initial 4K context length, each example must be handled in one of three ways: (i) drop it entirely, losing the reasoning signal; (ii) truncate naively, often preserving the reasoning prefix while losing the answer and thereby training the model on reasoning that never reaches a conclusion; or (iii) preserve the answer while truncating part of the reasoning. We use the third option and call the resulting scheme _answer-preserving (AP) trimming_ . 

Given a sample containing one or more assistant messages with <think>...</think> reasoning blocks followed by a final-answer section, AP-trimming applies the following procedure to fit the sample within a target context budget _C_ : 

- 1) **Keep unchanged.** If the full conversation fits within _C_ , retain it as-is. 

- 2) **Trim the tail of the last reasoning block.** If the conversation does not fit, truncate the final assistant turn’s reasoning trace from the tail, immediately before the answer. This preserves the start of the reasoning trace and the full answer section. The retained reasoning length is chosen so that the full sample fits within _C_ . 

- 3) **Drop prior reasoning blocks.** For multi-turn conversations, if step 2 is insufficient, remove the <think> blocks of earlier assistant turns while preserving their answer sections, then re-apply step 2. 

- 4) **Drop the sample.** If the answer sections alone exceed _C_ , discard the sample. 

The core idea is to truncate from the tail of the reasoning trace rather than from the middle. The beginning of a reasoning trace often contains problem decomposition, planning, and exploration of multiple approaches. The tail is usually more local, consolidating the selected approach into the final answer. Removing tail tokens therefore preserves more of the planning and decomposition signal while producing partial but coherent reasoning sequences whose beginning, truncated end, and final answer remain causally aligned. The transition between truncated reasoning and the answer is distributionally artificial, but in practice we did not observe obvious artifacts: passrate evaluations on reasoning benchmarks after pretraining 

6 

|Phase|Context|RoPE base|Token budget|Main emphasis|
|---|---|---|---|---|
|Base pretraining, phase 1|4K|10K|8T|Broad web, code, math, multilingual|
|Base pretraining, phase 2|4K|10K|4T|More code, math, reasoning, instruction data|
|32K midtraining|32K|1M|1.2T|Long-CoT reasoning, code, math, long-context data|
|SFT|131K|5M|660B|Chat template, reasoning, code, IF, TTC traces|



TABLE II: Training recipe summary. Base-pretraining details are summarized here for context and described in detail in (Anthony et al., 2025). 

|Category|32K midtraining|131K SFT|
|---|---|---|
|Long-CoT reasoning traces|86.1%|75.0%|
|Web, synthetic web, multilingual|5.7%|9.8%|
|Natively long-context data|0.8%|6.4%|
|Code corpus / code SFT|3.0%|5.0%|
|Math/STEM corpus|3.0%|2.6%|
|Short instruction / few-shot data|1.4%|1.2%|



TABLE III: Coarse midtraining data mixtures. The 32K context-extension mixture was trained for approximately 1.2T tokens, while SFT was trained for approximately 660B tokens; percentages denote normalized mixture weights. Individual source datasets are omitted. 

and midtraining remained strong, and we did not identify a truncation-specific failure mode in downstream evaluations. 

_a) Stage-aware re-trimming:_ AP-trimming is applied offline to each dataset at each context length where the data is used. As the training pipeline advances through 4K pretraining, 32K midtraining, and 131K context-extension SFT, we re-trim each dataset to the corresponding context length and progressively retain longer reasoning traces. Most reasoning datasets fit fully at 131K context, so late midtraining operates on nearcomplete traces; early pretraining uses the most aggressive trimming. 

_b) Relation to prior work:_ The closest related techniques operate during inference or RL rollout generation rather than during pretraining data construction. (Khatri et al., 2025) use forced length interruptions during RL rollouts: when a thinking trace approaches the budget, the environment appends an endof-thinking phrase that forces the model to produce a final answer. (Yang et al., 2025) use a similar mechanism for inference-time thinking-budget control. Both methods operate on rollouts during training or generation, not on training data before consumption. The closest training-data analogue is the answer-length-filtered subset of (Akter et al., 2025), which retains examples whose answer length exceeds 4K tokens as a proxy for reasoning depth. That is a selection strategy rather than a truncation strategy. AP-trimming addresses the complementary problem of using long-CoT reasoning data at training contexts shorter than the natural trace length by truncating reasoning while preserving the answer section. 

## IV. POST-TRAINING 

Post-training begins with SFT, followed by a four-stage RL cascade. The first three RL stages are almost entirely verifiable reasoning: a math/puzzle/TTC warmup, an RLVE-Gym adaptive difficulty curriculum, and a two-phase math+code+TTC 

stage. We defer general chat, style, and instruction-following optimization to the final behavioral RL stage. This ordering prioritizes capability extraction from verifiable signals before applying preference and instruction-following rewards. 

Two aspects of this ordering differ from common posttraining recipes. First, reasoning RL is front-loaded: most RL compute before behavioral RL is spent on verifiable math, puzzles, synthetic environments, and code. Second, the code stage uses several synthetic auxiliary environments constructed from competitive-programming references, including code input/output prediction, code reconstruction from test cases, and falsification. 

## _A. Supervised Fine-Tuning_ 

The SFT phase establishes the chat template used in subsequent post-training, improves instruction following, and continues reasoning supervision at 131K context. The stage consumed 660B tokens. We use a supervised mixture spanning chat, instruction following, code, math, reasoning, tool-calling traces, and TTC aggregation examples, but do not report individual dataset details. 

Because the SFT stage trains at 131K context, packing strategy mattered. We use optimized best-fit decreasing bin packing (Ding et al., 2024) rather than naively streaming examples into fixed-length windows and truncating at arbitrary boundaries. The packer fills each 131K window with complete examples whenever possible; over-length examples are handled by dataset-specific preprocessing before packing rather than by training on arbitrary suffixes created by a fixed-boundary truncation pass. This avoided hallucination artifacts we observed when models were trained on endings of mechanically truncated packed sequences. 

SFT also introduces aggregation-based examples used by Markovian RSA. These examples present the model with a 

7 

**==> picture [504 x 161] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 SFT 2 Reasoning 3 RLVE-Gym 4 Math + Code + 5 Math + Code + 6 Behavioral RL<br>warmup curriculum TTC TTC<br>CE loss verifiable reward verifiable reward verifiable reward verifiable reward RLAIF + verifiable<br>131K660B contexttokens 232 RL steps 400 steps 384 steps 464 steps 384 steps<br>• Sets chat template and • Reasoning Gym/Core 54.5% • 400 adaptive puzzle-like • 18,656 rows • 12,092 rows • 80K RM prompts × 1 epoch<br>supervised skills • Competition math 31.2% environments • Standard math 33.8% • Code-focused mix • Simple IF prompts × 1 epoch<br>• Mixture: chat, IF, code, math, • Enigmata puzzles 14.4% • Thompson/IRT calibration to • Standard code 20.3% • Standard code 31.4% • Hard IFBench-like prompts ×<br>reasoning, • examples • SelfAnswer-Preservingaggregationtool tracescoldtrimmingstart • solvable • unsolvableFilteredFilteredby(easy)(hard)bylowlargeeffortpromptspromptsmodelreasoning 0.5 • difficulty • environmentsOnlineSamplerpass rateschedulerupfavorsor downleast-sampledmoves •• 12.5%CodeMath aux/TTCTTC/RSA/PaCoRe33.4% ••• 19.4%CodeStandardMath aux/TTCTTC/RSA/PaCoRemath 22.5%26.8% 1 • checkerepochRM score gated by binary IF<br>Shared RL algorithmic spine<br>Same infrastructure across RL stages. Data, reward, filtering, and some hyperparameters vary.<br>PipelineRL rollouts/training async on disjoint GPUs DPPO δ = 0 . 1 Binary-TV TV; no reward KL Dr-GRPO seq-mean/token-sum aggregation Muon no momentum optimizer state, low memory MaxRL prevents advantage mode collapse Online reject uniform filtering pass/fail samples Batching 128 prompts × 16<br>**----- End of picture text -----**<br>


Fig. 5: Schematic of our post-training process for ZAYA1-8B. Post-training progressed through SFT followed by four sequential RL stages. The first stage built general reasoning capabilities on math and puzzles and was then followed by two stages of code + TTC training. The model was then polished through a short behavioral RLHF phase which focused more on chat and user interaction. 

problem and several candidate reasoning tails, then train it to produce a single improved solution. Section VI-B describes this construction in detail. 

## _B. Reinforcement Learning Cascade_ 

Post-training reinforcement learning is organized as a fourstage cascade. The cascade uses a shared algorithmic spine described in Section IV-B1; individual stages differ in data, reward signal, and stage length. 

_1) Algorithmic Spine:_ All RL stages and subphases share a common algorithmic spine. Per-stage differences are confined to data, reward signal, and a small number of hyperparameters. 

_a) PipelineRL:_ Rollout generation and gradient updates run fully asynchronously on disjoint GPU pools (Piché et al., 2025; Khatri et al., 2025). We allocate 2–5 _×_ more rollout workers than trainer workers, balanced per workload to match average response length to actor update time so that neither pool stalls. Trainer-to-rollout weight sync happens in place every 2 trainer iterations; in steady state, the rollout policy is bounded at 2 trainer updates behind the trainer policy. 

_b) Trust region:_ DPPO Binary-TV. We replace PPO’s per-token ratio clipping with Binary Total-Variation trustregion masking (Qi et al., 2026). Tokens for which the policydivergence estimate exceeds a threshold _δ_ are masked from the gradient while remaining tokens contribute as in standard policy-gradient updates. We use _δ_ = 0 _._ 1 in production. We tune this threshold against preserving the reward-growth trajectory of an unconstrained baseline, selecting the largest value that did not produce unconstrained reward growth in early training. The Binary-TV variant uses a deterministic indicator over a single divergence threshold rather than the continuous TV penalty or the Top- _K_ approximation, and adds negligible overhead relative to standard PPO. 

_c) Loss aggregation:_ Dr-GRPO SMTSN. Loss aggregation follows Dr-GRPO (Liu et al., 2024): sequence-mean over token-sum-norm (SMTSN). Token-level losses are summed 

within each rollout and then averaged across rollouts in the batch, rather than averaged per-token. This avoids the implicit length normalization in standard GRPO, which biases the gradient toward longer responses. 

_d) Advantage estimation:_ MaxRL. Advantages are computed as in (Tajwar et al., 2026). For each prompt, we sample a group of _G_ rollouts with task rewards _ri ∈{_ 0 _,_ 1 _}_ using dynamic sampling (Yu et al., 2025). The advantage normalizes by the per-prompt mean reward rather than the per-prompt reward standard deviation: 

**==> picture [154 x 21] intentionally omitted <==**

where _r_ ¯ = _G_ 1 � _Gj_ =1 _[r][j]_[is][the][group][mean][reward.][This] corresponds to the variance-reduced MaxRL estimator (Tajwar et al. (2026), Algorithm 1), which is unbiased for a truncated maximum-likelihood objective rather than for expected reward and produces stronger gradient signal on harder prompts. We use this normalization in all RL stages except the final behavioral RL stage (Section IV-B7), which uses standard GRPO with reward standard-deviation normalization. 

_e) Reward shape and length reward:_ Task rewards are binary across the cascade, with the exception of (i) RLVE-Gym environments that yield continuous solve rates near difficulty thresholds (Section IV-B4) and (ii) the behavioral RL stage, which uses a normalized reward-model score (Section IV-B7). All RL stages except behavioral RL also include the difficultyscaled length reward of Section IV-B2, applied additively as ∆ _ri_ to the task reward at the numerator of the advantage only — the denominator _r_ ¯ uses the unmodified task reward to avoid scale blow-up. The length-reward coefficient _c_ ramps from a small initial value during reasoning warmup to _c_ = 1 _._ 0 during the math+code+TTC stage, where the production reasoning length is established. 

_f) No KL in reward:_ The cascade applies no KL regularization to the reward; the trust region is enforced entirely 

8 

|Stage / phase|Length|Main data|Reward signal|
|---|---|---|---|
|Reasoning warmup|232 steps|Math, puzzles, TTC traces|Verifable task reward|
|RLVE-Gym curriculum|400 steps|400 adaptive environments|Environment verifer / solve rate|
|Math+Code+TTC, phase 1|384 steps|General math, code, TTC|Verifable task reward|
|Math+Code+TTC, phase 2|464 steps|Code-focused mix|Verifable task reward|
|Behavioral RL|384 steps|Chat/RM, simple IF, hard IF|RM score with IF gate for IF stages|



TABLE IV: RL cascade summary. The first three stages emphasize verifiable reasoning. Behavioral RL is run last to tune chat, style, and instruction-following behavior. 

by DPPO Binary-TV. In stress-testing with high KL-penalty coefficients, we observed a length-dependent bias attributable to applying a signed sequence-level log-ratio reward term to stale or mixed-policy rollouts under in-flight weight sync; Section IV-F describes the mechanism and possible mitigations. The production cascade avoids this configuration entirely by relying on the DPPO trust region alone. 

_g) Optimizer:_ All RL stages use Muon with momentum set to zero (GLM-5-Team et al., 2026), extending the GLM5 prescription of resetting the optimizer at each weight-sync boundary into a fully momentum-free regime. Section IV-E describes this choice in detail and discusses its motivation and memory implications. 

_h) Hyperparameters:_ Across all five stages, the cascade uses minibatches of 128 prompts with rollout group size _G_ = 16 responses per prompt. Per-rollout maximum response length is 81,920 tokens, except for the first half of the reasoning-warmup stage, which uses 65K. The maximum aggregation-prompt length is 20,480 tokens, sized to fit Markovian RSA round-1 prompts containing _C_ = 4 candidate tails. Trainer-to-rollout weight sync occurs every 2 trainer iterations. The trainer requires 2 batches of completed rollouts to be available in the buffer before pulling, and the buffer is capped with oldest-sample eviction; in steady state, on-policy staleness is bounded at 2 trainer updates. Learning rates are set per stage in the range 2 _×_ 10 _[−]_[6] to 1 _×_ 10 _[−]_[5] , with the smallest values used during behavioral RL. 

_2) Token efficiency:_ To encourage concise reasoning, we combine aspects from ALP (Xiang et al., 2025) and ShortRL (Yuan et al., 2025) to create a group-relative, difficulty-scaled length reward. Given a prompt with rollout group size _G_ , response reward _ri ∈{_ 0 _,_ 1 _}_ , and response length _ℓi_ , we compute the group solve rate _p_ = _G_ 1 � _Gi_ =1 _[r][i]_[and][the][shortest][correct][response] length _ℓ_ min. Similar to ShortRL, we define a linear length interpolation, with the distinction that _ℓ_ max is a constant: 

**==> picture [209 x 58] intentionally omitted <==**

Let _k_ =[�] _i[r][i]_[denote][the][number][of][correct][responses][in] the group. We apply the length reward only when at least two responses in the group are correct, so that there is a nontrivial 

comparison among correct response lengths. We adopt the following correctness and difficulty gate: 

**==> picture [223 x 13] intentionally omitted <==**

where the term _p[∗]_ denotes a running maximum solve rate for the corresponding data source or environment, the condition _p >_ 1 _/G_ is equivalent to _k ≥_ 2 for integer-valued binary rewards, and _T_ acc is a tolerance that prevents the length reward from activating far below the current observed capability frontier. We additionally scale the bonus by the solve rate _p_ , attenuating the length penalty on difficult problems and amplifying it on easier problems. The final additive reward is: 

**==> picture [170 x 12] intentionally omitted <==**

where _c_ is a scaling coefficient. This reward ∆ _ri_ is added to the task reward, biasing the policy toward shorter correct solutions while preserving task accuracy in our production runs. 

_3) Reasoning Warmup:_ The first RL stage is a 232-step reasoning warmup on math, puzzle, and TTC reasoning prompts. Its purpose is to adapt the SFT model to long verifiable rollouts before the broader RLVE and math+code stages. The warmup set contains 84,604 rows and is deliberately biased toward hard prompts: retained examples have prior pass rate at most 0.75, with most examples at substantially lower pass rates. Responses in this stage are long, with median replay response length around 17.6K tokens and a p90 near 30K tokens. 

Rewards are verifiable and task-specific. For math problems, the reward is based on final-answer correctness after normalization. For puzzle environments, the reward is supplied by the environment verifier. TTC prompts are formatted to match the Markovian RSA workflow described in Section VI, so the model begins RL already seeing aggregation-based reasoning prompts. 

_4) RLVE-Gym Difficulty Curriculum:_ The second RL stage trains for 400 steps on 400 adaptive and verifiable problem generators from RLVE-Gym (Zeng et al., 2025a). We integrated RLVE as a dataset in VeRL (Sheng et al., 2025). Although this stage has fewer optimizer steps than the later math+code stage, the average step is roughly twice as long because responses are long, with reasoning lengths around 50K tokens. We use this stage to expose the model to a broad distribution of puzzle-like verifiable environments while 

9 

|Category|Mix share|
|---|---|
|Reasoning-gym and reasoning-core puzzles|54.4%|
|Competition level math reasoning|31.2%|
|Enigmata puzzles|14.4%|



TABLE V: Coarse composition of the reasoning-warmup RL data. Percentages are computed over 84,604 warmup rows. 

keeping each environment near the model’s current difficulty boundary. 

During training, we used an online scheduler for problem difficulty, and we balanced environment selection using a weighted sampler for which the least sampled environments get the highest weight. Our difficulty scheduler differs slightly from the authors’ in that it uses a tighter bound on the difficulty _d_ and allows regressions. 

Let _y_ ˆ denote either the optimal solution or a reasonable heuristic when optimal solutions are intractable. We define 

**==> picture [208 x 100] intentionally omitted <==**

where _ri_ is reward per rollout in a group, _r_ ¯ = _G_ 1 � _i[r][i]_ is the group pass rate, _ϵ_ is an environment-specific numerical tolerance used to determine whether the rollout answer _y_ is close enough to the target _y_ ˆ to receive reward 1, _d_ is the current difficulty setting, and _d_ group is the observed difficulty of the last computed group. We constrain updates to _d_ group = _d_ to prevent stale difficulties from affecting the pass rate estimate. 

Crucially, we use an initial tuning step to avoid training on difficulties that are too easy for the model, and we aim to maximize the information content during training by initializing all environments to a difficulty that gives a 0.5 pass rate for the model. This tuning process is an adaptive search problem, and the search space is essentially unbounded. Some environments are solvable into the range of _d >_ 100, while others are rarely solvable even at 0. For this reason, we rely on Thompson Sampling (Thompson, 1933) as a reasonably efficient method to determine the 0.5 solve rate crossing point for every environment. We model the pass rate using the complement of the logistic curve as is commonly done in Item-Response-Theory (IRT) (Lord, 1980) and we sample from the midpoint with an _ε_ -greedy approach. Each verified response group yields an estimate of the pass rate. A parameter pool is maintained as with Thompson Sampling and a single Gaussian prior on _µ_ and _s_ is used for all environments based on empirically observed ranges. 

**==> picture [210 x 39] intentionally omitted <==**

Given a parameter pool, we perform weighted sampling proportional to the posterior weight of the candidates (initialized as uniform) in Θ and for each iteration we sample a candidate and use it to compute a difficulty _d_ at _p_ target: 

**==> picture [207 x 73] intentionally omitted <==**

where _**w**_ is the normalized posterior-weight vector over the candidate logistic-curve parameters in **Θ** . 

We use _p_ target = 0 _._ 5, the maximum Fisher information point of the logistic model (Lord, 1980), with _ε_ -greedy exploration to 0 _._ 5 _±_ 0 _._ 25. We then perform rollouts and verification at the sampled difficulty and close the loop by updating and renormalizing the posterior: 

**==> picture [205 x 26] intentionally omitted <==**

where _p_ success _,j_ is the current estimate for candidate _j_ . Groups are generated asynchronously using vLLM with the previous phase’s frozen model weights. If the effective sample size of the pool falls below a threshold, we resample with replacement from Θ, aggregate the observation history as a recencyweighted sum of successes and failures, then re-initialize likelihoods. 

This curriculum is intended to maximize useful verifier signal. Environments that are too easy produce mostly positive groups and little policy-gradient information; environments that are too hard produce mostly negative groups. The initial calibration and online difficulty updates keep each environment near a solvable but non-saturated regime, making the stage a bridge between the narrower reasoning warmup and the broader math+code+TTC RL stage. 

_5) Math, Code, and Test-Time Compute RL:_ The third RL stage is the main capability-building stage of the cascade. It combines olympiad-level math, competitive-programming code, Markovian RSA aggregation prompts, PaCoRe continuation prompts, and synthetic auxiliary code environments. We run this stage in two phases: a 384-step general math+code+TTC phase, followed by a 464-step code-focused phase. 

Table VI summarizes the two data mixtures. Phase 1 contains 18,656 rows and balances math and code while introducing TTC and PaCoRe variants. Phase 2 contains 12,092 rows and increases the code share while retaining math TTC data. 

10 

|Category|Phase 1: general|Phase 2: code-focused|
|---|---|---|
|Standard code prompts|20.3%|31.4%|
|Code auxiliary / code TTC prompts|33.4%|26.8%|
|Standard math prompts|33.8%|22.5%|
|Math TTC / RSA / PaCoRe prompts|12.5%|19.4%|



TABLE VI: Coarse composition of the math+code+TTC RL stage. Phase 1 uses 18,656 rows; phase 2 uses 12,092 rows. Percentages are grouped from source tags and rounded. 

The auxiliary code environments are constructed by transforming competitive-programming references into multiple verifiable tasks per source problem. Each seed problem contains a problem statement, input/output specification, accepted reference implementations, rejected or incorrect implementations when available, and test cases. From these seeds, we construct three auxiliary task families: 

- 1) **CodeI/O prediction** (Li et al., 2025). Given code and a set of inputs, the model predicts the outputs; in the reverse direction, given code and outputs, the model proposes inputs that produce them. Output-prediction rewards use exact normalized agreement with the reference execution. Input-prediction rewards execute the reference program on the generated input and check that the target output is produced while satisfying the input schema. 

- 2) **CodeARC reconstruction** (Wei et al., 2025). Given a problem description, input/output specification, and example test cases, the model synthesizes code. The verifier compiles or executes the generated solution and checks it against held-out tests. 

- 3) **Falsification.** Given a specification and a candidate implementation, the model must find an input that falsifies the implementation relative to the specification or a trusted correct implementation. The verifier checks that the generated input is valid and that it induces a disagreement or specification violation. 

These tasks target algorithmic reasoning primitives rather than only end-to-end competitive-programming solving. CodeI/O emphasizes execution tracing and inverse reasoning over program behavior. CodeARC emphasizes synthesis from sparse behavioral evidence. Falsification emphasizes adversarial test construction and spec-implementation comparison. All three are binary-verifiable and therefore fit the same RL objective as math and puzzle prompts. 

TTC prompts are included in the same RL stream. For Markovian RSA examples, the prompt contains the original problem and a small set of candidate reasoning tails. The policy generates a single aggregated solution and receives the standard verifiable reward for the final answer or produced code. This lets the stage train both ordinary single-rollout problem solving and the aggregation workflow used at inference time. 

_6) Agentic task scope:_ ZAYA1-8B does not include a dedicated multi-turn agentic RL stage in this release. We include some supervised agent, tool, and SWE traces during SFT, but the RL cascade is primarily optimized for verifiable 

reasoning, math, code, and instruction-following behavior. As a result, we expect agentic benchmarks such as BFCL-v4 and _τ_[2] to lag models whose post-training explicitly emphasizes multi-turn tool use. Scaling agentic data and agentic RL is left for future releases. 

_7) Behavioral RL:_ The final RL stage tunes general chat behavior, style, and instruction following after the verifiablereasoning stages have established the model’s math and code capabilities. Behavioral RL uses standard GRPO with reward standard-deviation normalization rather than the MaxRL normalization used in the verifiable-reasoning stages. It also does not use the length reward from Section IV-B2. 

We first train for one epoch on 80K behavioral prompts (Wang et al., 2024, 2025). This stage improves general response quality and chat behavior without changing the reasoning-focused data distribution of the earlier stages. 

We then run two instruction-following stages, each for one epoch. The first uses simpler instruction-following prompts; the second uses more difficult IFBench-like prompts. For these IF stages, the reward is gated by a binary instruction-following checker. If the completion fails the IF gate, its reward is set to zero. If it passes the gate, the completion is scored by the reward model. This prevents the reward model from assigning positive reward to fluent responses that fail the explicit instruction constraints. 

## _C. RL Infrastructure_ 

_a) Router replay:_ The single most important MoEspecific change for RL stability is router replay: the trainer reuses the expert routing assignments produced by vLLM at rollout time during its own forward pass over the rollout, rather than recomputing routing decisions from scratch. Even with the precision settings in Section IV-D, small numerical differences between the rollout engine and the trainer can produce different routing decisions for tokens near a router decision boundary. In an MoE with top-1 routing, a token routed to expert _e_ inference at rollout time but to a different expert _e_ train = _e_ inference at gradient time produces different per-token logits, which corrupts the on-policy gradient. Router replay eliminates this source of mismatch: by pinning the trainer’s expert selection to the rollout-time decision, enforcing _e_ train _≡ e_ inference, the gradient is computed against the same expert sequence that produced the rollout. We discuss the SNR view of this mismatch in Section VII-C. 

In practice, vLLM writes per-token and per-layer expert assignment indices to a shared memory buffer during decode. The write is overlapped with decode work to avoid slowing 

11 

rollout generation. Assignments are then packed alongside the rest of the rollout batch (token IDs, masks, etc.) when the batch is shipped to the trainer, so router replay introduces no separate transport step. 

_b) Memory and recompute strategy:_ For long-rollout training the dominant memory pressure comes from activations. We combine host-side activation offloading with gradient checkpointing: the hidden state tensors from each layer that autograd must retain for backward are temporarily offloaded to CPU memory during the forward pass, while checkpointed layer interiors discard their forward activations and reconstruct them by rerunning the layer forward during backward. This trades extra backward-time compute and host-to-device traffic for substantially lower peak GPU activation memory. In this configuration we use FSDP shard size 4 under the FSDP2 sharding strategy with sequence parallelism disabled; at this model size and per-rank rollout length, the extra cross-rank communication from sequence parallelism and ring attention is not worth the memory savings. 

_c) Packing and dynamic batching:_ We use sequence packing and variable length attention for the trainer. This allows the trainer to run with dynamic microbatch sizing: rather than fixing the number of rollouts per microbatch, we fix a token budget of 131,072 tokens per GPU per microbatch and pack rollouts into microbatches up to this budget. This avoids paying for the longest rollout in a fixedrollout-count microbatch when most rollouts are shorter, and keeps GPU memory utilization stable across batches even when rollout-length distributions shift between training stages. We additionally rebalance pack assignments across GPUs so that microbatches on different ranks contain comparable token counts; without this, the slowest rank gates the entire step, since synchronous gradient accumulation must wait for all ranks to finish. With balancing, per-step variance across ranks is small enough that no rank consistently bottlenecks training. 

_d) Buffer management:_ The trainer pulls completed rollouts from a shared buffer with a maximum capacity bound and oldest-sample eviction. As described in Section IV-B1, the trainer requires 2 batches of completed rollouts to be available before pulling. This combination keeps the rollout pool from running ahead of the trainer (which would inflate staleness and waste rollout compute), while ensuring the trainer is never blocked waiting for fresh rollouts under our 2–5 _×_ rollout-totrainer ratio. 

## _D. Precision_ 

The default precision regime for ZAYA1-8B RL is BF16 weights and activations, with a small set of operations promoted to FP32. The subset of operations in FP32 is identical between the trainer and vLLM, which is necessary for enginetrainer log-prob agreement within the regime needed for stable PipelineRL training (see Figure 6). 

_a) FP32 operation set:_ The following operations run with FP32 numerics on both trainer and inference paths: 

- **Loss/output:** fused cross-entropy accumulation and LMhead matmul. 

- **Attention/normalization:** CCA cache state, QK-norm, QKmean, and RMSNorm; see Section II. 

- **Routing/residuals:** router softmax and residual stream additions. 

The LM-head FP32 promotion follows precedents in (Khatri et al., 2025) and (Chen et al., 2025). The remaining FP32 ops were added incrementally to close engine-trainer logprob mismatch observed in early training runs; without them, mismatch produces grad-norm spikes and stale-policy artifacts under PipelineRL. 

_b) FP16 detour:_ Recent work argues that training– inference mismatch in RL fine-tuning can arise directly from floating-point precision, and proposes using FP16 uniformly rather than BF16 as a simple way to reduce mismatch (Qi et al., 2025). However, in our comparisons, we found that a hardened BF16 path with a small matched FP32 operation set on both the rollout engine and trainer achieved the engine– trainer agreement needed for stable PipelineRL training, while retaining BF16’s dynamic-range advantages. We therefore use BF16 weights and activations by default, promote only the operations listed above to FP32, as described previously. 

_c) Rollout Engine-trainer match:_ Figure 6 compares per-token log-probabilities computed by vLLM (used during rollout generation) and by the trainer’s prefill (used to compute gradients). At our default precision setup, the two distributions are nearly identical: KL divergence = 1 _._ 3 _×_ 10 _[−]_[4] and Pearson _r >_ 0 _._ 9996 over a 128-prompt, _G_ = 16 batch with 4K-token completions. This level of agreement is a precondition for stable PipelineRL training under our staleness regime; without the FP32 op set above, agreement degrades substantially and downstream training is unstable. 

## _E. Optimizer_ 

Let _Lt_ ( _W_ ) denote the actor training loss for the parameter matrix _W_ on rollout batch _t_ , and let _gt_ = _∇W Lt_ ( _W_ ) be the corresponding actor gradient. Let _mt_ denote Muon’s first moment buffer, and let _M_ ( _·_ ) denote the Muon orthogonalization step via Newton-Schulz (Jordan et al., 2024). Standard Muon uses the update 

**==> picture [171 x 24] intentionally omitted <==**

where _ηt_ is the learning rate at optimizer step _t_ . For actor updates, we set _µ_ = 0 so _mt_ = _gt_ and ∆ _Wt_ = _−ηtM_ ( _gt_ ). Thus each actor update depends on the current rollout batch and does not carry first-moment optimizer state across rollout batches. For embedding and output-head parameters, including the word embedding and LM head, we use AdamW rather than Muon. For the remaining matrix-valued actor weights, we use momentum-free Muon. 

The motivation differs from pretraining. Compared to AdamW, Muon stands as a more compute efficient optimizer that is well suited to the RL setting where updates to parameters are sparse (Mukherjee et al., 2026b). Furthermore, in next-token pretraining, adjacent minibatches are drawn from a comparatively stationary data distribution, so momentum can 

12 

**==> picture [390 x 155] intentionally omitted <==**

**----- Start of picture text -----**<br>
BF16 BF16+FP32 BF16+FP32+RR<br>10<br>08 }<br>py<br>;<br>{<br>06 x<br>7<br>04 A<br>2 rs3<br>02<br>#2%Ld Z% #7<br>0.0 - -<br>00 02 0.4 06 08 1.00.0 02 04 06 08 10 0.0 02 04 06<br>Engine probability Engine probability Engine probability<br>Prefill probability<br>**----- End of picture text -----**<br>


Fig. 6: Per-token probability comparison (log scaled frequency): vLLM (engine, used for rollout generation) vs. trainer prefill (used for gradients) with incremental precision improvements. **BF16** : naive uniform BF16 implementation in inference and prefill. **BF16+FP32** : addition of selective upcasting of a subset of operations to FP32. **BF16+FP32+RR** : additional improvement from implementing router replay on trainer prefill from cached indices of rollout. Each point is a token from a 128-prompt, _G_ = 16 evaluation batch with 4K-token completions on ZAYA1-8B. Identity line shown (dashes). For BF16+FP32+RR, KL divergence = 1 _._ 3 _×_ 10 _[−]_[4] , Pearson _r >_ 0 _._ 9996. 

average compatible gradient directions across steps. In RL, each actor update is tied to a rollout batch whose prompts, sampled trajectories, rewards, and generating policy snapshot may differ from neighboring batches. Following (GLM-5Team et al., 2026), we view optimizer-state reset as a useful stability heuristic for asynchronous RL. Our setting extends this idea: instead of resetting the optimizer state only at rolloutengine weight-sync boundaries, we make every actor update momentum-free. This makes each update depend only on the current rollout batch while retaining Muon’s normalized matrix update, ∆ _Wt_ = _−ηtM_ ( _gt_ ), rather than a raw SGD step (∆ _Wt_ = _−ηtgt_ ). We treat this as a practical stability and memory choice, not as evidence that zero momentum is generally optimal for RL. 

This choice also avoids maintaining a persistent firstmoment buffer for the Muon-updated actor weights during RL, reducing optimizer-state memory relative to momentum Muon. We did not include a controlled optimizer ablation in this report. A direct comparison against momentum Muon, AdamW, and SGD updates is left for future work. 

## _F. Monitoring and maintaining stability_ 

Reward and KL diagnostics describe the policy’s optimization dynamics but do not reflect the content of generated rollouts. We monitor a small set of auxiliary rollout-level statistics during RL training to fill this gap. A subset of these statistics also act as reward gates, zeroing a rollout’s task reward when its content is flagged as degenerate. 

_a) Streaming compressibility:_ Our primary canary is a sliding-window LZ77 compressibility metric computed per chunk on the raw token-ID bytes of each rollout. Compression uses zlib with a 2[10] = 1024-byte LZ77 window (wbits=-10), level-1 deflate, and Z_SYNC_FLUSH between chunks; the compressor is stateful, so each chunk’s compression ratio reflects compressibility relative to recent history 

bounded by the LZ77 window rather than whole-sequence redundancy. Each rollout is divided into fixed-size chunks of _C_ tokens (with the final short chunk merged into its predecessor to avoid Z_SYNC_FLUSH overhead inflating short-tail ratios), and the per-chunk compression ratio 

**==> picture [213 x 24] intentionally omitted <==**

is computed for each chunk _c_ . 

A small _rc_ indicates a chunk that compresses well against its preceding context, which is the signature of degenerate repetition or copying: the model has emitted a span of tokens already present in the LZ77 window. More generally, as is noted by (Lee et al., 2026), an effective compression algorithm also serves as a computable upper bound on Kolmogorov Complexity (Li & Vitányi, 2019), and both ends of the compressibility spectrum could arguably be filtered as either low information content or purely random. We choose LZ77 in particular over simpler n-gram or token counting methods because it takes into account sequence-level matching within the window, whereas language and domain-level n-gram biases can complicate simpler presence/frequency metrics. We flag a rollout if any chunk satisfies _rc < τ_ repeat, with a conservative _τ_ repeat = 0 _._ 05 in production. Flagged rollouts have their task reward zeroed before advantage computation, so the policy receives no positive learning signal for producing degenerate text even when the verifier accepts the (technically correct) final answer at the end of a long repetitive trace. The per-chunk granularity allows reward zeroing on rollouts where degenerate spans appear at any position rather than attempting to rely on coarser, full response compressibility. 

_b) Rare-token monitoring:_ As an independent signal, we track the fraction of tokens in each rollout whose token IDs fall in the top _X_ % of the tokenizer’s ID range. This is a lightweight proxy for unusual or rarely used tokens in our 

13 

tokenizer. In production monitoring we track several cutoffs, including 10%, 5%, 2%, and 1%, and use the top-10% tokenID region for gibberish canaries. A rising rare-token fraction often precedes other failure indicators and is cheap to compute. 

_c) Operational use:_ The low-ratio repetition canary and rare-token-fraction statistics are computed per batch during RL training and visible alongside reward and KL in WandB. The repetition canary additionally runs as a reward-zeroing gate: rollouts that exceed the low-ratio threshold have their rewards zeroed before advantage computation, regardless of verifier outcome. Canary signals do not adjust learning rate or any other optimizer setting. 

_d) Length bias from signed KL-in-reward under pipeline RL:_ Beyond rollout-level canaries, we also monitored response-length growth, which exposed an interaction between PipelineRL training and a sequence-level signed logratio reward penalty. In early stress tests combining the two, we observed runaway response-length growth: rollouts grew progressively longer over training without corresponding reward improvement. Our working explanation is specific to this estimator and aggregation choice. In pipeline RL, long completions can span multiple generator-policy snapshots: early tokens may be sampled from a stale generator policy _π_ gen _,c_ that is ∆ _c_ trainer updates behind the current actor _πθ_ , while later tokens may be sampled from fresher snapshots with smaller ∆ _c_ . 

The commonly used _K_ 1-estimator log-ratio KL term is 

**==> picture [213 x 12] intentionally omitted <==**

For tokens sampled from _π_ gen _,c_ ( _t_ ), this signed log-ratio is negative in expectation whenever the current policy differs from the generator policy: 

**==> picture [245 x 23] intentionally omitted <==**

For fresh tokens with small policy lag, _lt ≈_ 0. If these terms are aggregated into a sequence-level scalar, 

**==> picture [154 x 22] intentionally omitted <==**

and subtracted from reward as 

**==> picture [164 x 11] intentionally omitted <==**

then stale off-policy tokens can create a positive reward offset. Longer completions can accumulate more negative signed logratio terms, and when the resulting sequence-level adjusted advantage is broadcast back to all tokens, stale-prefix terms can affect the learning signal assigned to later suffix tokens. 

This produces a length-dependent bias through two interacting effects. 

_Stale-prefix contamination._ Longer sequences can contain more stale prefix tokens contributing negative _lt_ , making _S_ seq more negative. Since the signed log-ratio term enters as _−β_ KL _S_ seq, the negative sequence sum acts as a positive reward offset, inflating the advantage for longer sequences independent of task quality. 

_Staleness-dependent penalty scale._ The magnitude of the signed log-ratio can also depend on chunk staleness ∆ _c_ . Relatedly, Bartoldson derives a first-order EMA-reference approximation for asynchronous RL in which the log-ratio between the current policy and a ∆-old inference policy can be interpreted as a surrogate for KL regularization against an EMA reference, under local linearity and first-order Taylor assumptions (Bartoldson, 2026). This suggests that ∆ can change the effective scale of a stale-policy log-ratio penalty. In our setting, we use this only as intuition for lag-dependent penalty strength; the length-bias mechanism itself follows from applying a signed off-policy log-ratio at sequence level and subtracting it from reward. 

This mechanism should be distinguished from true KL regularization. A KL divergence is non-negative by construction, whereas the sampled signed log-ratio _lt_ can be arbitrarily negative on individual samples and is negative in expectation under the generator distribution when _π_ gen _,c_ ( _t_ ) = _πθ_ . The severity depends on both absolute staleness and withinsequence policy heterogeneity. In-flight synchronization can create prefix–suffix heterogeneity by allowing one completion to span multiple generator snapshots; holding the generator fixed for the entire completion removes this specific coupling, but does not remove the off-policy signed-log-ratio length offset if the fixed generator is stale relative to the trainer. 

_e) Possible mitigations:_ Two practical mitigations target the specific stale-prefix coupling described above. _Chunklocal signed-log-ratio isolation_ aggregates the signed log-ratio within each chunk rather than across the full sequence, so stale-prefix terms do not directly contaminate the advantage assigned to fresher suffix chunks: 

**==> picture [209 x 23] intentionally omitted <==**

This localizes the bias but does not by itself turn the off-policy signed log-ratio into a true KL penalty. 

_Staleness rescaling_ is an additional heuristic: divide the chunk term by an empirical staleness scale _g_ (∆ _c_ ), with _g_ (∆ _c_ ) _>_ 0, to reduce variation in effective penalty strength across chunks generated at different lags: 

**==> picture [199 x 26] intentionally omitted <==**

A simple first-order choice is _g_ (∆ _c_ ) = max(1 _,_ ∆ _c_ ), motivated by the local-linear lag dependence in Bartoldson’s EMA approximation (Bartoldson, 2026), but the correct scale is implementation- and dynamics-dependent. 

For ZAYA1-8B we did not implement either mitigation in production. Instead, we removed KL-in-reward entirely and rely on the DPPO Binary-TV trust region (Section IV-B1) for trust-region enforcement. This was sufficient for the trainingstability properties we required and avoided tracking chunk boundaries and per-chunk generator staleness. We document the mechanism here because it may arise in asynchronous or pipeline RL systems that combine stale or mixed-policy rollouts, a signed _K_ 1-estimator log-ratio in the reward, sequence- 

14 

level aggregation, and broadcast of the resulting adjusted advantage. 

## V. RESULTS 

Results are organized into three tables. Table VII compares ZAYA1-8B against open-weight reasoning models at comparable scale. Table VIII extends to open-weight models in the 26B–119B total-parameter range. Table XI reports testtime compute comparisons against open-weight models in the 235B–671B range plus Gemini-2.5 Pro and GPT-5-High. 

## _A. Evaluation Protocol_ 

Unless otherwise noted, ZAYA1-8B results are measured with the Zyphra evaluation harness. In-class comparator models are run in the same harness when feasible, using each model’s recommended sampling settings from its model card. For ZAYA1-8B single-rollout reasoning evaluations, we use temperature 1.0, top-p 0.95, top-k -1, and benchmark-specific maximum generation lengths. For thinking-mode Qwen comparators, we mirror the recommended thinking-mode settings from the corresponding model card. Results reported from external release materials are marked with _[†]_ . TTC evaluations use the checkpoint immediately following the math+code+TTC RL stage and before the final behavioral-RL polishing stage; the latter targets chat style, instruction following, and preference behavior rather than additional math/code/TTC capability. 

For pass-rate evaluations in the Zyphra harness, we report averages over multiple samples per problem. Math benchmarks, including AIME, HMMT, IMO-AnswerBench, and APEX-shortlist, are reported as avg@64. Code benchmarks, including LiveCodeBench-family tasks, are reported as avg@16. GPQA-Diamond and _τ_[2] are reported as avg@16 unless otherwise noted. MMLU-Pro, BFCL-v4, HLE, IFEval, IFBench, EQBench, and Creative Writing are reported as mean@1 or as the benchmark’s standard single-run score. We use avg@k to mean the mean correctness over k independently sampled completions, estimating single-sample pass rate under the stated sampler; it is not best-of-k/pass@k unless explicitly stated. Markovian RSA results use the TTC protocol in Section VI; its token counts are total newly generated decode tokens and exclude prompt/prefill tokens. Results from external release materials may use different sampling and reporting protocols and are marked with _[†]_ . 

## _B. Main Results: In-Class Comparison_ 

Table VII compares ZAYA1-8B against Qwen3-4BThinking-2507, Qwen3.5-4B, and Gemma-4-E4B-it. 

## _C. Scaling Comparison: Larger Open-Weight Models_ 

Table VIII compares ZAYA1-8B against larger openweight reasoning models: Arcee-Trinity-Mini, Nemotron-3Nano, OLMo-3.1-32B-Think, Qwen3-Next-80B-A3B-Think, Intellect-3, and Mistral-Small-4-119B-2603. 

## _D. Test-Time Compute Scaling_ 

Table XI compares ZAYA1-8B with Markovian RSA testtime compute against substantially larger reasoning models. With the headline Markovian RSA configuration ( _β_ = 40K, _τ_ = 4K, _T_ = 2, _N_ = 16, _C_ = 4), ZAYA1-8B reaches 91.9 on AIME’25 and 89.6 on HMMT’25 Feb. 

## _E. Effect of Post-Training_ 

To quantify the effect of post-training, we compare the 131K SFT checkpoint against the final ZAYA1-8B checkpoint using the same evaluation harness and sampling settings in Table IX. This comparison measures the aggregate effect of the RL cascade rather than isolating the contribution of each individual stage. We do not report per-stage ablations in this release. 

## VI. TEST-TIME COMPUTE 

Test-time compute (TTC) scaling — increasing inference compute per problem to improve answer quality — has become an important axis of capability scaling for reasoning models, alongside model scale and training compute. Two recent lines of work motivate the design space considered here. (Venkatraman et al., 2025) introduce Recursive SelfAggregation (RSA), a TTC scheme that maintains a population of candidate reasoning chains and refines them through repeated aggregation: at each iteration, the model is shown a random subset of candidates and produces an improved candidate, which seeds the next iteration’s population. Empirically, RSA allows smaller open-weight models to approach the performance of larger reasoning models when given sufficient inference compute. (Aghajohari et al., 2025) introduce Markovian Thinker, a reformulation of the RL thinking environment in which the policy reasons in fixed-size chunks with bounded carryover state between chunks, decoupling thinking length from context size. Their key observation is that longcontext reasoning can be factorized in a Markovian way: with sufficient training, a model can sometimes carry forward only the information needed in a bounded textual state and continue reasoning indefinitely. 

We introduce _Markovian RSA_ , a TTC method that combines RSA’s recursive candidate aggregation with the boundedworkspace principle of Markovian Thinker. We integrate it into ZAYA1-8B’s training pipeline so the model is trained to use the same workflow at inference. The method has three components: an algorithm that includes both RSA and MarkovianThinker for chunked reasoning as special cases (Section VI-A), a training-time integration that supplies verifier-free aggregation examples for SFT and verifiable aggregation prompts for RL (Section VI-B), and an inference-time scaling profile with bounded per-iteration aggregation context, capped attention costs, and predictable throughput (Section VI-C). 

## _A. Markovian RSA_ 

_a) Algorithm:_ Given a problem _q_ and a base policy _π_ , Markovian RSA proceeds over _T_ aggregation rounds, indexed _t_ = 0 _,_ 1 _, . . . , T_ . Each round maintains a population of _N_ 

15 

**==> picture [498 x 266] intentionally omitted <==**

**----- Start of picture text -----**<br>
AIME'26 HMMT'26 LCB-v6 IFEval GPQA-D<br>80 70 80<br>90 89.1 90.1 86.4 86.3 71.6 75.5 70.6 72.2 64.8 64.6 66.8 90 92.8 75.1 77.2 74.6<br>< 7 70 © = 60 (©. 57.9 85.6 84.0 70 71.0 <n HN.LY<br>81.2<br>80<br>60 80 Kd<br>50<br>60<br>70 50<br>70<br>40<br>50<br>40 33.3 62.0 46.8<br>36.9<br>59.6<br>60 30 2. 60 4. 4.<br>1 YW 2 NM WW| MY 1 122 MlWW7 VVA i 1 2hz WW7 07 07VY7 1 WW22 Ml 2 VVA BZ 1 WV a Ml | WW<br>3 3 3 3 3<br>10 10 10 10 10<br>30 30 30 30 30<br>100 100 100 100 100<br>active params total params<br>ZAYA1-8B (0.7/8B) Arcee-Trinity-Mini (3/26B) NVIDIA Nemotron 3 Nano (3/30B) Mistral-4-Small (6/119B) Intellect-3 (12/106B)<br>Score (%)<br>Params (B)<br>**----- End of picture text -----**<br>


Fig. 7: Comparison of ZAYA1-8B performance against open-weight reasoning models on various evaluations. The under-bar plots model sizes in active and total parameters on a log scale to give a sense of the scale of the various models. 

|Active / Total||0.7B / 8.0B|4.0B / 4.0B|4.0B / 4.0B|4.0B / _∗_8.0B|
|---|---|---|---|---|---|
|Category|Benchmark|ZAYA1-8B|Qwen3-4B-Thinking-2507|Qwen3.5-4B|Gemma-4-E4B-it|
||AIME’26|89.1|79.0|84.5|50.3|
|Math|HMMT’26 Feb.|71.6|53.6|63.6|32.1|
||IMO-AnswerBench|59.3|51.6|48.7|27.3|
||APEX-shortlist|32.2|17.1|21.35|6.1|
|Code|LiveCodeBench-v6|64.8|54.9|55.8_†_|54.2|
|Knowledge|GPQA-Diamond<br>MMLU-Pro|71.0<br>74.2|66.1<br>74.3|76.2<br>79.7|57.4<br>70.2|
|Instruction|IFEval|85.6|86.8|89.8|88.5|
||IFBench|52.6|52.9|59.2|42.7|
|Style & chat|EQBench<br>Creative Writing v3|73.0<br>63.0|79.6<br>58.6|79.5<br>72.9|80.2<br>83.8|
|Agentic|BFCL-v4<br>_τ_ 2|40.5<br>36.3|49.7<br>52.9|45.2<br>82.1|31.7<br>37.7|



> _∗_ Gemma4 includes 4B additional embedding parameters as a part of its total. 

> _†_ Qwen3.5-4B LiveCodeBench-v6 scores taken from release materials. 

TABLE VII: In-class comparison against models of comparable sizes. ZAYA1-8B used the following sampling settings: _T_ =1 _._ 0, top- _p_ =0 _._ 95, top- _k_ disabled for math, knowledge, and instruction; _T_ =0 _._ 6, top- _p_ =0 _._ 95, top- _k_ =20 for code, agentic, and style. We used the recommended sampling settings in the model cards for the other models in this table. EQBench and Creative Writing v3 use the official judge, anthropic/claude-3.7-sonnet. 

16 

|Model|Active|Total|AIME’26|HMMT’26|Feb.|LCB-v6*|IFEval|GPQA-D|MMLU-Pro|
|---|---|---|---|---|---|---|---|---|---|
|ZAYA1-8B|0.7B|8B|89.1|71.6||64.8|85.6|71.0|74.2|
|Arcee-Trinity-Mini|3B|26B|59.6|36.9||33.3|62.0|46.8|70.6|
|Nemotron-3-Nano-30B-A3B|3B|30B|90.1|75.5||64.6|92.8|75.1|78.9|
|OLMo-3.1-32B-Think|32B|32B|78.9|50.6||58.3|93.2|59.6|75.8|
|Qwen3-Next-80B-A3B-Think|3B|80B|90.2|79.3||67.8|88.5|76.7|82.6|
|Intellect-3|12B|106B|86.3|72.3||66.8|81.2|74.6|82.3|
|Mistral-Small-4-119B|6B|119B|86.4|70.6||57.9|84.0|77.2|81.6|



TABLE VIII: Scaling comparison against larger open-weight reasoning models, ordered by total parameter count. All numbers are run on the Zyphra evaluation harness. 

> _∗_ LCB-v6 denotes the 2025-02–2025-05 LiveCodeBench-v6 split. 

|Benchmark|SFT checkpoint|Final ZAYA1-8B|∆|
|---|---|---|---|
|AIME’26|68.30|89.10|+20.80|
|HMMT’26 Feb.|39.20|71.60|+32.40|
|LiveCodeBench-v6|54.80|64.84|+10.04|
|GPQA-Diamond|59.30|71.00|+11.70|
|MMLU-Pro|70.10|74.20|+4.10|
|IFEval|66.60|85.58|+18.98|
|IFBench|30.20|52.56|+22.36|
|EQBench|57.80|72.95|+15.15|
|Creative Writing v3|46.72|62.97|+16.25|
|BFCL-v4|33.41|40.50|+7.09|
|_τ_ 2|32.88|36.30|+3.42|



TABLE IX: Aggregate effect of post-training. SFT and final ZAYA1-8B checkpoints are evaluated with the same harness and benchmark-specific sampling settings. This table reports the aggregate effect of the post-training recipe; it is not a per-stage ablation. 

candidate reasoning traces. At round _t_ = 0, the model generates _N_ independent rollouts directly from _q_ , each with a per-rollout thinking budget _β_ . Each rollout’s reasoning trace is then reduced to its last _τ_ tokens, which we call the _tail_ . We write tail _τ_ ( _y_ ) for the operation that returns the final _τ_ tokens of reasoning trace _y_ , with _τ ≤ β_ . 

For rounds _t ≥_ 1, the algorithm operates on tails from the previous population. To generate each new candidate, it samples _C ≤ N_ tails uniformly at random, concatenates them into an aggregation prompt, and asks the model to reason over the candidate solutions and produce a single improved solution. The model generates a new reasoning trace under the same per-rollout budget _β_ . The trace is again reduced to its final _τ_ tokens, and the resulting tail enters the population for round _t_ . This process repeats until round _T_ , after which the final answer is extracted from the final round’s outputs using the standard answer-extraction procedure. The aggregation prompt simply asks the model to consider the candidates and produce the best solution; it does not require specialized parsing or verifier feedback. 

ward. Full-chain RSA passes the full reasoning chain[*] , so the aggregation prompt at round _t ≥_ 1 contains _C_ chains, each with length up to _β_ . Markovian RSA passes only the final _τ_ tokens of each chain, with _τ ≤ β_ chosen independently. This decouples per-rollout thinking depth from aggregationcontext size: _β_ controls how long each candidate may reason, while _τ_ controls how much of that reasoning is carried into the next round. Setting _τ ≪ β_ allows larger per-rollout thinking budgets while keeping aggregation prompts small. As a result, decode-attention cost, prefill-attention cost, and KVcache footprint are bounded by configuration constants rather than by reasoning length. 

_b) Default configuration:_ For ZAYA1-8B, we use ( _N, C, T_ ) = (16 _,_ 4 _,_ 2) with _β_ set per workload and _τ_ chosen as a fraction of _β_ (typically _τ ≤ β/_ 2). Both _β_ and _τ_ can be tuned per deployment to trade off per-round thinking depth against total inference budget. 

_c) Inference profile:_ Markovian RSA changes the inference workload from a single long, position-growing decode into a sequence of bounded-context batched decoding stages. At round 0, the model generates _N_ independent candidates 

Both Markovian RSA and full-chain RSA bound per-rollout generation cost: _β_ caps the number of tokens any single candidate generates. The difference is what gets passed for- 

> *One can add a summarization step to full-chain RSA to keep aggregation prompts short. We focus on fixed-tail forwarding because it gives a simple, bounded aggregation context without requiring an additional summarization model or parsing step. 

17 

from the original problem, so decode runs at batch size _N_ rather than batch size 1. At each later aggregation round, the model again generates _N_ candidates, but each candidate conditions only on the problem and _C_ carried-forward tails of length at most _τ_ . Thus the aggregation prefill length is bounded by 

**==> picture [183 x 11] intentionally omitted <==**

and the per-candidate decode length is bounded by _β_ , independent of the total amount of reasoning generated across all rounds. This gives a stable serving profile: prefill is short and predictable at every stage, decode uses high-throughput batched generation, and no stage attends over the full reasoning history. 

This profile differs from both single-rollout long-CoT and full-chain RSA. A single long rollout has batch size 1 and a decode position that grows with the full reasoning length. Full-chain RSA supports batched candidate generation, but its aggregation prefill grows with _Cβ_ because it forwards full reasoning chains. Markovian RSA keeps the batched candidate-generation structure of RSA while replacing fullchain forwarding with bounded tail forwarding, so increasing _β_ increases per-candidate thinking depth without increasing aggregation-context length. 

_d) Special cases:_ Markovian RSA contains several common TTC regimes as special cases: 

- **Parallel sampling /** _N_ **responses.** Setting _T_ = 0 removes aggregation and produces _N_ independent responses. If a verifier, answer-selection rule, or external scoring model is applied to these responses, this reduces to a Best-of- _N_ evaluation; otherwise it is simply parallel sampling. 

- **Full-chain RSA.** Setting _τ_ = _β_ forwards each full reasoning chain between rounds, recovering RSA. In this limit, aggregation prefill grows with the full reasoning budget. 

- **Delethink bounded continuation.** Setting _C_ = 1 removes cross-candidate aggregation while retaining bounded carryover. Each candidate continues from its own tail, giving a parallel version of Markovian/Delethink chunked reasoning. This isolates the effect of bounded continuation from the additional effect of cross-candidate aggregation. 

_e) Comparison with PaCoRe:_ (Hu et al., 2026) introduced PaCoRe, a related multi-round parallel-reasoning scheme that also bounds per-round aggregation context. PaCoRe compacts each trajectory by extracting its final-answer or conclusion section and passing this extracted message forward between rounds. Markovian RSA instead passes the final _τ_ tokens of the reasoning trace itself as the carryforward state, regardless of whether the trajectory reached a final conclusion. The two methods share the same goal of bounding aggregation context across rounds and differ in compaction mechanism: PaCoRe uses model-structured finalanswer extraction, while Markovian RSA uses a fixed-size suffix of generated reasoning. 

In practice, we also evaluate a PaCoRe hybrid compaction variant: when a candidate reaches a post-think answer section, we pass that compact answer forward; otherwise, we fall 

back to passing the partial reasoning chain. This hybrid keeps the compact-message advantage of PaCoRe when candidates finish, while avoiding the need to set _β_ large enough for every branch to reach a final answer. 

## _B. Training-Time Integration_ 

A TTC method may be more effective when the model is trained on the workflow it uses at inference. Markovian RSA’s aggregation prompt presents the model with a problem and several candidate reasoning tails, then asks it to produce a single improved solution. This behavior is rare in standard reasoning-model training data, where each example typically consists of one problem and one solution. To train ZAYA18B for Markovian RSA scaling, we construct aggregationbased examples from existing expert-model reasoning data and include them in SFT and RL. 

_a) SFT data construction from expert rollouts:_ Many open-source reasoning datasets used during midtraining and SFT include multiple expert-model rollouts per problem, often with _n_ = 8 rollouts (e.g., OpenMathReasoning, rStar-Coder, internal reasoning gym and enigmata data). For each problem _q_ with rollouts _{y_ 1 _, . . . , yn}_ from a teacher model, we construct a round-0-to-round-1 aggregation example as follows: sample _C_ rollouts from the _n_ available; extract their tails _{_ tail _τ_ ( _yi_ 1) _, . . . ,_ tail _τ_ ( _yiC_ ) _}_ ; form an aggregation prompt containing _q_ and the _C_ tails; and condition the teacher to produce a new aggregated rollout under the same prompt. The resulting aggregated rollout, including its reasoning trace and final answer, becomes the SFT target. 

This construction has two practical advantages. It is offline and reuses existing rollout pools: no new expert-model inference is needed for each round-0 sample beyond the aggregation step. It also does not require a verifier: the teacher’s aggregated rollout is used as the target regardless of whether the underlying answer is verifiable. This makes the technique applicable to puzzle, code, and reasoning domains where the post-think content is itself the answer and where finalanswer-only aggregation strategies such as PaCoRe’s message compaction are not directly applicable. 

_b) RL stage integration:_ During RL, Markovian RSA examples are folded into the standard prompt distribution and treated like other RL prompts. Two variants are used during the math+code+TTC stage (Section IV-B5): 

- **Expert-aggregation.** Round-1 prompts are constructed from expert-model rollouts as described above. The policy generates an aggregated rollout and is rewarded against the verifiable target. 

- **Self-aggregation.** For prompts where rollouts from the current SFT checkpoint or a prior-stage RL checkpoint are available, round-1 prompts are constructed from those self-rollouts. The policy aggregates over its own reasoning traces, or over traces from its predecessor. 

In both variants, the aggregation example is a standard RL prompt: the policy generates a single rollout, and verifiable reward is applied to its final answer. No special multi-round 

18 

|Method|Decode profile|Aggregation/context state|Special case|
|---|---|---|---|
|Single long-CoT|BS=1, position grows with total|Full prior trace|–|
||reasoning|||
|Parallel sampling / _N_|BS=_N_, one round|No aggregation|Markovian RSA with|
|responses|||_T_ = 0; becomes Best-of-|
||||_N_ only with an external|
||||selector or verifier|
|Delethink continuation|BS=_N_, chunked rounds|One bounded tail|Markovian RSA with|
||||_C_ = 1|
|Full-chain RSA|BS=_N_, aggregation rounds|_C_ full chains, length up to _Cβ_|Markovian RSA with|
||||_τ_ =_β_|
|Markovian RSA|BS=_N_, aggregation rounds|_C_ tails, length up to _Cτ_|General case|



TABLE X: Inference-profile view of Markovian RSA. Markovian RSA preserves the batched candidate-generation structure of RSA while bounding the state forwarded between rounds. Setting _T_ = 0 gives _N_ independent responses with no aggregation; this becomes Best-of- _N_ only if an external selector, verifier, or answer-selection rule is applied. Setting _C_ = 1 gives the Delethink bounded-continuation regime, and setting _τ_ = _β_ recovers full-chain RSA. 

Fig. 8: One round of Markovian RSA. From a population of _N_ candidate reasoning traces (left), we extract the final _τ_ tokens of each trace as its tail. To produce each new candidate for the next round, we sample _C_ tails uniformly at random and present them to the model as candidate solutions in an aggregation prompt. The model produces a new reasoning trace, whose tail joins the next round’s population. Aggregation context size and per-round attention cost depend only on _C_ and _τ_ , and are independent of the per-rollout thinking budget _β_ . 

RL machinery is required; the round structure is encoded in _C. Inference-Time Scaling_ the prompt construction rather than in the gradient update. We currently train on round-1 self-aggregation. Round-2-andWe evaluate Markovian beyond self-aggregation, where the policy aggregates rollouts two axes — per-rollout from a prior-stage version of itself in an online buffer, is a with the configuration natural extension left for future work. (16 _,_ 4 _,_ 2).. The sampling 

We evaluate Markovian RSA’s inference-time scaling along two axes — per-rollout reasoning budget _β_ and tail size _τ_ — with the configuration described in Section VI-A ( _N, C, T_ ) = (16 _,_ 4 _,_ 2).. The sampling settings are reported in Table XII. We also compare against full-chain RSA (Venkatraman et al., 2025), recovered as the _τ_ = _β_ limit of our algorithm. The final scores reported in this section are mean correctness over the final-round candidate outputs, not best-of- _N_ unless explicitly stated. 

_c) Domain coverage:_ Aggregation-based training data is included for math, code, reasoning gym, and enigmata puzzle problems. Directly aggregating over reasoning tails is useful in domains where the post-think content is the answer rather than a separate boxed result. This allows the same approach to apply across domains regardless of answer format. 

_a) Headline result:_ With Markovian RSA at ( _β, τ, T, N, C_ ) = (40K _,_ 4K _,_ 2 _,_ 16 _,_ 4), ZAYA1-8B reaches 91.9% on AIME’25 and 89.6% on HMMT’25 Feb. These 

19 

|Model|Active|Total|AIME’25|HMMT’25 Feb.|LCB-v6*|
|---|---|---|---|---|---|
|ZAYA1-8B (single rollout)|0.7B|8.0B|88.3|82.7|65.0|
|ZAYA1-8B + Markovian RSA (40K/4K)|0.7B|8.0B|91.9|89.6|69.2_‡_|
|DeepSeek-R1-0528_†_|37B|671B|87.5|79.4|68.7|
|Qwen3-235B-A22B-Thinking-2507_†_|22B|235B|92.3|83.9|74.1|
|Gemini-2.5 Pro_†_|–|–|88.0|82.5|72.5|
|DeepSeek-V3.2_†_|37B|671B|93.1|92.5|–|
|GPT-5-High_†_|–|–|94.6|88.3|–|



> _∗_ LCB-v6 denotes the 2025-02–2025-05 LiveCodeBench-v6 split. 

> _‡_ For LCB-v6 on the same pre-behavioral checkpoint after math+code+TTC RL, Markovian RSA improves ZAYA1-8B from 65% single-rollout to 69.2%, while our PaCoRe hybrid compaction variant reaches 71.1%. This variant is not an exact implementation of PaCoRe: when a candidate reaches a post-think answer section, we pass that compact answer forward; when it does not, we fall back to passing the partial reasoning chain rather than dropping the candidate. Since the model was trained with both RSA and PaCoRe-type aggregation examples, we do not attribute the gap to training exposure alone. 

TABLE XI: ZAYA1-8B single-rollout and TTC numbers in this table are evaluated on the pre-behavioral checkpoint after math+code+TTC RL and before the final lightweight behavioral-RL polishing stage, using the Zyphra evaluation harness. The final behavioral stage targets chat style, instruction following, and preference behavior rather than math/code/TTC capability. ZAYA1-8B + Markovian RSA uses the 40K/4K configuration from Section VI-C. Numbers for comparator models marked _†_ are taken from external sources. DeepSeek-R1-0528, Qwen3-235B-A22B-Thinking-2507, and Gemini-2.5 Pro are from the Qwen3-235B-A22B-Thinking-2507 model card (Qwen Team, 2025). DeepSeek-V3.2 is from the DeepSeek-V3.2 technical report (DeepSeek-AI, 2025c); the GPT-5-High row is reproduced from the comparison table in that report rather than from an OpenAI release table. 

runs use temperature 1.0, top-p 1.0, and a 40K-token finalresponse budget. The result holds while carrying forward only a 4K-token tail between aggregation rounds, approximately one-tenth of the per-rollout reasoning budget. 

_b) Configuration sweep:_ Table XII reports accuracy across four Markovian RSA configurations, alongside the _C_ = 1 Markovian Thinker baseline described in Section VI-A. At _T_ = 2 _, N_ = 16, and _C_ = 4, increasing the per-rollout reasoning budget _β_ from 8K to 16K to 40K improves both benchmarks at fixed tail size: AIME’25 advances from 86.5% to 88.8% to 91.9%, and HMMT’25 advances from 80.8% to 87.1% to 89.6%. HMMT’25 is especially responsive to longer per-rollout reasoning, with a 6.3-point gain from _β_ = 8K to _β_ = 16K. 

_c) Evaluation scope:_ The algorithmic definition, training construction, and serving profile above describe the Markovian RSA method. The remainder of this subsection reports empirical TTC scaling results, measuring how accuracy changes with the per-rollout reasoning budget _β_ , carried-forward tail length _τ_ , aggregation depth _T_ , and aggregation method under a fixed population size. 

_d) Generated-token accounting:_ We report realized total generated decode tokens separately from per-worker trajectory lengths. Markovian RSA generates multiple candidates in parallel at each non-final stage, so a per-worker or per-stage average length is not the total decode cost of the method. For a problem _q_ , let _gs,j_ ( _q_ ) be the number of newly generated tokens from worker _j_ in generation stage _s_ , and let _ns_ be the number of workers in that stage. The realized generated-token cost is 

**==> picture [206 x 30] intentionally omitted <==**

where _g_ ¯ _s_ ( _q_ ) is the average generated length per worker in stage _s_ . This count includes newly generated candidate and aggregation tokens across all workers, but excludes the original problem prompt, aggregation-prompt prefill, and copied carryforward tails. 

Under this accounting, with the final response budget of 40K, the reported AIME’25/HMMT’25 Markovian RSA evaluations use approximately 440K generated decode tokens per problem for the 16K/4K configuration and approximately 740K generated decode tokens per problem for the 40K/4K configuration. These are realized averages from the evaluation runs, not worst-case caps; they depend on early stopping, benchmark, sampling settings, and implementation details. We include them to avoid conflating per-worker trajectory length with total TTC cost. 

_e) Tail size and iteration depth:_ Increasing the tail size _τ_ from 4K to 8K at fixed _β_ = 40K does not improve accuracy on AIME’25 or HMMT’25: the 4K-tail configuration reaches 91.9%/89.6%, while the 8K-tail configuration reaches 90.8%/89.2%. Because these scores average over multiple generated candidates per problem, the comparison is less sensitive to a single unlucky rollout than a one-sample evaluation. We treat this as empirical evidence that, for these configurations and benchmarks, aggregation is not limited by retaining more than a 4K reasoning tail. 

This should not be read as a general claim that tail length never matters. On harder benchmarks, the aggregation depth ( _T_ ), the diversity of the candidate responses ( _N_ and _C_ ), _β_ , and _τ_ can all contribute to saturating the model’s capacity. We also evaluate higher-compute Markovian RSA settings on APEX-shortlist, a harder capacity-ceiling benchmark. Table XIII reports the three APEX configurations used for the light, high, and extra-high modes shown in Figure 10. The 

20 

|Configuration|_T_|_N_|_C_|_β_ (CoT)|_τ_|(Tail)|AIME’25|HMMT’25|
|---|---|---|---|---|---|---|---|---|
|Markovian baseline|4|16|1|8K||4K|82.1|75.0|
|Markovian RSA|2|16|4|8K||4K|86.5|80.8|
|Markovian RSA|2|16|4|16K||4K|88.8|87.1|
|Markovian RSA|2|16|4|40K||4K|**91.9**|**89.6**|
|Markovian RSA|2|16|4|40K||8K|90.8|89.2|



TABLE XII: Markovian RSA configuration sweep on AIME’25 and HMMT’25. _T_ is the number of aggregation rounds, _N_ is the population size, _C_ is the number of candidates sampled per aggregation prompt, _β_ is the per-rollout reasoning budget, and _τ_ is the carry-forward tail size. Markovian RSA rows use temperature 1.0 and top-p 1.0 with a 40K-token final-response budget. The Markovian baseline row uses _C_ = 1, removing cross-candidate aggregation while retaining _N_ = 16 independent chunked rollouts. 

Fig. 9: Accuracy vs. realized total newly generated decode tokens per problem for Markovian RSA configurations on AIME’25 (left) and HMMT’25 (right). The token axis excludes the original problem prompt, aggregation-prompt prefill, and copied carry-forward tails, and corresponds to _s[g]_[¯] _[s]_[(] _[q]_[)][,][using][notations][from][(][24][),][averaged][over][prob-] lems. Curves correspond to the configurations in Table XII: _β_ = 8K _, τ_ = 4K; _β_ = 16K _, τ_ = 4K; _β_ = 40K _, τ_ = 8K; and _β_ = 40K _, τ_ = 4K. All curves use _N_ = 16, _C_ = 4, _T_ = 2, temperature 1.0, and top-p 1.0. Allocating budget to longer per-rollout reasoning yields more accuracy per generated token than allocating to longer tail carryover in this sweep. 

**==> picture [235 x 166] intentionally omitted <==**

**----- Start of picture text -----**<br>
60<br>51.8<br>50 +19.6 48.6<br>777Z _ L 45.8<br>40 +13.9<br>30 -[I +1.1<br>32.2<br>20<br>Pe<br>e e<br>ZAYA1-8B DeepSeek-V3.2 GPT OSS 120B<br>(high)<br>APEX-Shortlist<br>ZAYA1-8B without TTC light TTC boost DeepSeek-V3.2<br>high TTC boost xHigh TTC boost GPT OSS 120B (high)<br>Score (%)<br>**----- End of picture text -----**<br>


Fig. 10: APEX-shortlist performance under the light, high, and extra-high Markovian RSA configurations from Table XIII. The extra-high configuration reaches 51.8% using approximately 5.5M newly generated decode tokens per problem. GPT-OSS-120B (high) and DeepSeek-V3.2 comparator scores are taken from the MathArena leaderboard (Dekoninck et al., 2026). These external comparator scores are shown for context only and may use different inference protocols. 

strongest setting, with _T_ = 8, _N_ = 32, _C_ = 4, _β_ = 32K, and _τ_ = 4K, reaches 51.8% on APEX-shortlist. This setting uses approximately 5.5M newly generated decode tokens per problem, so we treat it as a capability-ceiling evaluation rather than a recommended deployment setting. 

|_T_|_N_|_C_|_β_|(CoT)|_τ_|(Tail)|APEX-shortlist|
|---|---|---|---|---|---|---|---|
|2|16|4||8K||4K|33.3|
|8|16|4||16K||4K|46.1|
|8|32|4||32K||4K|51.8|



TABLE XIII: APEX-shortlist Markovian RSA configurations. These three rows define the light, high, and extra-high modes shown in Figure 10. 

_f) Markovian baseline and the value of aggregation:_ The _C_ = 1 row of Table XII runs the algorithm with a single previous tail conditioning each new candidate: each of the 

_N_ = 16 candidates carries forward only its own bounded textual state, with no cross-candidate aggregation. This is a parallel version of chunked Markovian-Thinker rollouts. Two observations follow. First, the _C_ = 1 baseline reaches 82.1% on AIME’25 and 75.0% on HMMT’25, indicating that bounded carryover preserves much of the model’s long-CoT reasoning capability without aggregation — consistent with (Aghajohari et al., 2025)’s finding that off-the-shelf reasoning models support Markovian traces zero-shot. Second, the gap between the _C_ = 1 baseline and the corresponding Markovian RSA configuration (4.4 points on AIME’25 and 5.8 points on HMMT’25, both at _β_ = 8K, _τ_ = 4K) quantifies the additional gain from cross-candidate aggregation on top of bounded continuation. The two effects compose: bounded carryover establishes that the inference workload can be capped without losing reasoning capability, and recursive aggregation provides further gains from cross-candidate refinement. 

21 

_g) Comparison to full-chain RSA:_ Markovian RSA recovers full-chain RSA in the limit _τ_ = _β_ , but the boundedtail setting substantially reduces aggregation-prompt length. At _β_ = 40K and _C_ = 4, full-chain RSA would carry up to 160K candidate-state tokens per aggregation item, whereas Markovian RSA with _τ_ = 4K carries only 16K. We therefore use Markovian RSA for the reported high-budget evaluations. A fully matched empirical comparison against full-chain RSA is left for future work. 

_h) Cross-model comparison:_ Training for TTC. The benefits of training for a TTC workflow are not specific to Markovian RSA. Figure 11 compares ZAYA1-8B against Qwen3-4B-Thinking-2507 under Markovian RSA, with both models running the same TTC procedure. This comparison is not intended as a method evaluation, since both models use the same procedure. We interpret any gap as evidence of trainingdesign effects rather than method differences: ZAYA1-8B is trained from midtraining onward with TTC aggregation traces in midtraining, SFT, and RL; Qwen3-4B-Thinking-2507 is not. 

Fig. 11: Test-time compute scaling for ZAYA1-8B and Qwen34B-Thinking-2507. Both models use the same TTC procedure (Markovian RSA with _T_ = 2, _β_ = 16K, and _τ_ = 4K) with the final response budget of 40K for both the TTC and non-TTC runs (the recommended sampling parameters in the model card were used for the Qwen model), so performance differences reflect model-level differences in the ability to exploit aggregation. ZAYA1-8B saw long-CoT reasoning data during midtraining and aggregation-based data during SFT and RL; Qwen3-4B-Thinking-2507 was not trained for this specific aggregation workflow. We interpret the gap as evidence of training-design effects rather than method differences. 

candidate length capped by _β_ . 

This differs from both single long-CoT and full-chain RSA. A single long rollout has batch size 1, with decode position and KV-cache length growing continuously with the full trace. Full-chain RSA batches candidates, but forwards full reasoning chains into aggregation prompts, so the carried candidate state grows with _Cβ_ . At _β_ = 40K and _C_ = 4, full-chain RSA would carry up to 160K candidate-state tokens per aggregation item, whereas Markovian RSA with _τ_ = 4K carries only 16K, a 90% reduction in carried candidate state before accounting for the shared problem prompt. Markovian RSA therefore lets us increase per-candidate reasoning depth through _β_ without increasing aggregation-prefill length. 

This bounded-context profile should be distinguished from total generated-token cost. Markovian RSA still spends a large aggregate decode budget because it generates many candidates across parallel workers and aggregation rounds. For the 40K/4K configuration, the realized total is approximately 740K newly generated decode tokens per problem across all workers on the reported AIME’25/HMMT’25 evaluation runs. We therefore compare TTC configurations using both activeparameter _×_ total generated-token cost and the serving-side context profile: per-candidate decode capped by _β_ , aggregation prefill bounded by _|q|_ + _Cτ_ , and no stage attending over the full generated reasoning history. 

_j) Recommended deployment configuration:_ For deployment, we recommend the 16K/4K configuration from Table XII as a lower-cost default. It provides a strong accuracy– cost tradeoff while using substantially fewer realized decode tokens than 40K/4K: approximately 440K vs. 740K generated tokens per problem on the reported AIME’25/HMMT’25 evaluations, excluding prompt, prefill, and copied-tail tokens. In our current serving setup, we also observed the lightweight Markovian RSA configuration ( _T_ = 2 _, N_ = 8 _, C_ = 4 _, β_ = 8K _, τ_ = 4K, and final response budget 40K) completing in roughly 0 _._ 4 _×_ the wall-clock time of our standard _N_ = 1 long-reasoning baseline on the same evaluation harness. We attribute this to serving the workload as bounded-context batched decoding rather than as one long position-growing trace. We report this wall-clock ratio as an implementationspecific observation, not as a hardware-independent throughput benchmark. The 40K/4K configuration is reserved for capability-ceiling evaluation. 

## VII. DISCUSSION 

_i) Serving profile and compute efficiency:_ Markovian RSA has a favorable serving profile because every stage is both batched and bounded. Round-0 generation runs _N_ independent candidates in parallel. Later aggregation rounds also run at batch size _N_ , with each item prefilling only the original problem plus _C_ carried-forward tails. With the deployment setting _N_ = 16, _C_ = 4, and _τ_ = 4K, the candidate-state portion of each aggregation prompt is bounded by _Cτ_ = 16K tokens, plus the original problem and formatting overhead. Prefill is therefore bounded and predictable at every round, while decode proceeds as batched generation with per- 

In this technical report, we presented ZAYA1-8B, the first and smallest model in the ZAYA-1 family. ZAYA1-8B is designed to maximize reasoning performance per active parameter, with a particular focus on reasoning-intensive mathematics and coding. In this target regime, the model is strongly competitive with systems that use far more active parameters, and reaches or exceeds the level of earlier frontier-scale reasoning models such as DeepSeek-R1-0528 and Gemini-2.5 Pro on several challenging math and code benchmarks. 

With its native Markovian RSA TTC mode, ZAYA1-8B approaches the mathematical performance of much larger 

22 

frontier reasoning models such as Gemini-2.5 Pro, DeepSeekV3.2, and GPT-5-High. ZAYA1-8B is also competitive with substantially larger open-weight models including OLMo3.1-32B-Think, Nemotron-3-Nano-30B-A3B, Intellect-3, and Mistral-Small-4-119B, with its clearest advantage on math/code reasoning density. 

We attribute ZAYA1-8B’s performance to a combination of its architecture, cascaded RL pipeline, reasoning-heavy training data, and an AMD training stack that supported longcontext pretraining, midtraining, and SFT. 

Finally, we introduced Markovian RSA, a test-time scaling method that combines recursive self-aggregation with bounded carry-forward state. The model generates and aggregates batches of candidate responses in parallel, while each aggregation round conditions only on a fixed number of bounded-length reasoning tails. This preserves RSA’s cross-candidate refinement benefits while keeping aggregation context bounded, avoiding attention over the full generated reasoning history. 

We believe TTC is an especially promising avenue for smaller reasoning-focused models and can make them competitive with substantially larger models in active-parameter _×_ generated-token cost for some reasoning workloads. If TTC methods reliably convert additional generated tokens into accuracy, then active-parameter count and inference-time reasoning tokens become complementary axes of scaling. 

Below, we discuss observations and lessons learned from the ZAYA1-8B training process. 

## _A. RL Sample Efficiency_ 

The reasoning RL portion of ZAYA1-8B post-training is short relative to the pretraining and midtraining compute that precedes it. The main verifiable-reasoning cascade uses 232 reasoning-warmup steps, 400 RLVE-Gym steps, 384 math+code+TTC phase-1 steps, and 464 math+code+TTC phase-2 steps, for 1,480 total reasoning-RL update steps before behavioral RL. Despite this small number of optimizer steps, the aggregate post-training gain is large. Relative to the SFT checkpoint, we observe roughly a 20–30 point gain on AIME-like math evaluations and roughly a 10-point gain on LiveCodeBench-v6, with smaller but positive changes on many other benchmarks. Achieving gains of this size with ordinary midtraining or SFT would likely require substantially more data and optimization compute, and identifying the right supervised distribution would itself be nontrivial. 

One interpretation is that RL takes a KL-minimal path to optimality during training (Shenfeld et al., 2025), keeping the parameters relatively close to their initial values, compared to SFT, which may move the model arbitrarily in parameter space. Thus, the pretraining and midtraining stages appear to install most of the latent capabilities needed for reasoning; RL then changes the policy’s sampling distribution so that these capabilities are expressed more reliably under long generation budgets. This view is consistent with our pass@k observations: in our previous base-model report, the reasoning checkpoint showed strong pass@64 behavior at a 30K generation budget, 

while the post-RL model moves much of that capability into average sampled performance at longer generation lengths. However, unlike prior work (Cui et al., 2025) that argued RL simply uses up the inherent entropy of the midtraining checkpoint, we observed pass@k staying stable or even increasing during RL training where it did not hit the performance ceiling. 

The sample efficiency of RL remains an open problem. One plausible contributor is that the optimization problem in verifiable RL is qualitatively different from dense cross-entropy training. In SFT, every target token supplies a supervised gradient. In verifiable RL, useful signal is concentrated in trajectory-level outcomes, group-relative comparisons, verifier acceptance, and trust-region filtering: only a subset of prompts and sampled rollouts produces informative contrast for a given update. This makes the effective learning signal sparse over prompts and trajectories, before considering how the optimizer transforms that signal into parameter updates. We discuss optimizer-dependent sparsity in the resulting parameter deltas in Section VII-B. 

We do not claim here that SFT on the same data would fail to recover the RL gains; we have not run this experiment in a controlled form for this release. The safer interpretation is that short verifiable-RL runs can move a strong pretrained/SFT model into a nearby region of policy space that is difficult to identify from supervised next-token prediction alone. 

## _B. Momentum-Free RL Optimization_ 

ZAYA1-8B uses Muon with momentum set to zero for matrix-valued actor weights during RL. This is unconventional from the perspective of pretraining, where momentum and adaptive optimizer state are standard tools for stabilizing dense next-token-prediction training. In our RL setting, however, momentum-free Muon worked well and was compatible with the stability requirements of PipelineRL. We did not run a controlled optimizer ablation for this report, so we present this as an empirical recipe choice rather than as a general optimizer recommendation. 

Our motivation is that asynchronous verifiable RL differs from pretraining in both signal structure and stationarity. Each actor update is tied to a rollout batch with sampled trajectories, verifier outcomes, group-relative advantages, and a generating policy snapshot that may differ from neighboring batches. In this setting, carrying first-moment state across batches can average gradient information collected under different prompt sets, sampled solutions, reward outcomes, and policy lags. This may be useful when the persistent component of the gradient dominates minibatch noise, but it may be less useful when the stationarity horizon of the RL signal is short relative to the optimizer’s momentum horizon. We therefore use momentumfree Muon as a simple optimizer-state-reset variant: each actor update depends only on the current rollout batch while retaining Muon’s normalized matrix update. 

This choice is related to recent observations that RL finetuning of LLMs can update a relatively small subset of parameters and that memory-light optimizers can remain competitive in 

23 

RL settings (Mukherjee et al., 2025, 2026a). In an informal matched-step RL diagnostic, SGD updates left 99.51% of parameters exactly unchanged and 99.94% below 10 _[−]_[5] , while AdamW left 92.82% exactly unchanged and 96.40% below 10 _[−]_[5] . We do not use this diagnostic to claim that SGD, momentum-free Muon, or sparse updates are generally preferable, nor to claim that RL is universally sparse relative to SFT. Rather, it suggests that in our setting the effective update can be highly concentrated, making optimizer-state design a relevant practical consideration. 

Momentum-free Muon also reduces optimizer-state memory because no persistent Muon first-moment buffer is maintained for the actor weights. For matrix-valued actor parameters, the update remains Muon’s normalized matrix update rather than a raw SGD step, while embedding and output-head parameters are optimized with AdamW under the standard matrix-parameter split. A direct comparison against momentum Muon, AdamW, and SGD actor updates is left for future work. 

## _C. MoE Logit Mismatch and Router Replay_ 

PipelineRL requires the gradient step to be computed against the same policy distribution that generated the rollout, up to bounded staleness. In practice, this is an SNR problem rather than a binary correctness problem. Small engine-trainer logit differences add noise to the policy-gradient estimate; when the rest of the recipe is stable, this noise tends to appear as slower learning, unstable high-learning-rate behavior, or plateaus rather than immediate collapse. 

MoE models introduce an additional source of mismatch beyond ordinary numerical differences between the rollout engine and trainer. In a dense model, a small numeric perturbation usually causes a small logit perturbation. In a top-1 MoE, the same perturbation can flip a token’s expert assignment, producing a discontinuous change in the computation path. A token generated by expert _e_ rollout but trained through expert _e_ train = _e_ rollout gives the actor gradient the wrong local model for that token. This mismatch is especially harmful in longrollout RL, where many such token-level routing errors can accumulate across a sequence. 

Router replay (Ma et al., 2025) addresses this MoE-specific mismatch by recording the per-token, per-layer expert choices made by vLLM during rollout generation and replaying those choices during all trainer forward passes over the rollout. In ZAYA1-8B training, router replay was a major stability improvement: high learning rates that were unstable without replay became usable with replay. Router replay does not remove all sources of mismatch, but it removes the discontinuous top-1 routing component. 

The remaining engine-trainer mismatch is handled by the FP32 operation set described in Section IV-D. We use enginevs-trainer probability scatter plots as the main diagnostic: before hardening and router replay, the scatter broadens and token probabilities deviate from the identity line; after hardening, the distributions align closely. In the final configuration, 

the engine-trainer comparison reaches KL divergence approximately 1 _._ 3 _×_ 10 _[−]_[4] and Pearson correlation above 0 _._ 9996 on a 128-prompt, _G_ = 16, 4K-completion diagnostic batch. We found substantial benefits in terms of training stability and final performance from being extremely careful about reducing sources of numerical error, even if they initially seemed small. 

## _D. Data and Verifier SNR_ 

The sample efficiency of verifiable RL depends strongly on data and verifier SNR. A verifier can provide binary reward at scale, but a binary reward is only useful when the prompt distribution produces informative variation across sampled rollouts. If most groups are solved by every rollout, the gradient has little contrast. If no rollout solves the prompt, the gradient is also weak. If the verifier accepts shortcuts or rewards a skewed answer distribution, the model can learn the shortcut rather than the intended reasoning behavior which is often described as ‘reward hacking’. 

We therefore treat difficulty curation as a fundamental part of the RL algorithm rather than as a preprocessing detail. The reasoning warmup and math+code+TTC stages use passrate filtering to select hard but not fully saturated prompts. The RLVE-Gym stage performs this filtering online through an adaptive difficulty scheduler targeting the high-information region of each environment. Between major stages, we refilter our datasets using the current RL policy: first with an instruction-mode filtering pass using more aggressive sampling settings, and then with a thinking-mode pass closer to the RL rollout setup but at a lower response-length limit. These filtering passes remove the high end of the distribution, progressively raising the difficulty floor as the model improves. 

Verifier quality becomes more important as the model approaches the ceiling of a benchmark or training distribution. At low capability, even a coarse verifier can provide useful signal because the model is far from saturation. Near saturation, small verifier errors, spurious binary ground-truth patterns, incomplete code tests, or skewed answer distributions can dominate the remaining gradient signal, often causing learning to plateau. In practice, we typically could trace back a plateau to one of three causes: prompts that were too easy, prompts that were effectively impossible under the current rollout budget, or verifier/data artifacts that made a shortcut easier than genuine reasoning. 

This observation also helps explain why short RL runs can be effective. When the prompt distribution is centered near the model’s current capability boundary and the verifier is high precision, each rollout batch contains useful contrast. When data or verifier SNR degrades, increasing RL steps alone is inefficient: the optimizer repeatedly sees low-information or misleading groups which ultimately diminish the signal to the point where learning plateaus at a noise floor. For this reason, the ZAYA1-8B cascade alternates capability-building stages with data curation and difficulty filtering, rather than treating the RL dataset as fixed. 

24 

## _E. Why ZAYA1-8B Benefits from Test-Time Compute_ 

ZAYA1-8B is a small-active-parameter MoE: each generated token uses roughly 700M active parameters while drawing on a larger 8B-parameter expert pool across tokens. This makes test-time compute especially attractive. Generating many candidate traces is cheap in active-parameter compute, while the total expert pool still gives the model more specialization capacity than a similarly active dense model. Inference compute is therefore better measured by active parameters multiplied by generated tokens than by total parameters alone. 

Markovian RSA exploits this regime. The method spends additional compute on many relatively cheap candidate rollouts and aggregation passes, while bounding aggregation context through the carried-forward tail. This lets ZAYA1-8B trade more decode tokens for higher accuracy without paying the per-token cost of a much larger dense or high-active-parameter model. The relevant deployment question is not just whether accuracy increases with more tokens, but whether the activeparameter _×_ generated-token product is favorable relative to larger alternatives. 

Training also matters. ZAYA1-8B sees TTC aggregation traces before inference: long-CoT data appears during midtraining, aggregation-based examples appear during SFT, and Markovian RSA prompts are included in the math+code+TTC RL stage. As a result, the model is not asked to discover aggregation behavior only at inference time. It has learned a strong prior for reading several candidate tails, reconciling partial reasoning paths, and producing an improved solution. 

This combination helps explain why Markovian RSA scales well on ZAYA1-8B. The model has low per-token active compute, enough total expert capacity to support diverse candidate trajectories, and explicit training exposure to the aggregation workflow. We have observed weaker TTC scaling from models not trained for this workflow under the same RSA, Markovian RSA, and PaCoRe procedures, but those comparisons are not fully controlled because the models differ in architecture, training data, and post-training recipe. We therefore treat the main claim as specific to ZAYA1-8B: its architecture and training recipe make test-time compute a particularly effective way to convert low-active-parameter rollouts into higher reasoning accuracy. 

## _F. KL-in-Reward and Length Bias under PipelineRL_ 

Section IV-F describes a length-bias failure mode we observed when combining PipelineRL with a sampled signed _K_ 1-estimator log-ratio term in the reward. In short, stale or mixed-policy rollouts can make the signed sequence-level logratio negative, so subtracting it from reward can create a positive length-dependent offset. When one completion spans multiple generator-policy snapshots, stale-prefix terms can also affect the sequence-level advantage assigned to fresher suffix tokens. For ZAYA1-8B, we avoided this configuration by removing KL-in-reward and relying on DPPO Binary-TV for trust-region control. More principled chunk-local signed-logratio handling and staleness rescaling are left for future work. 

## _G. Open Questions and Limitations_ 

Several open questions remain. We provide evidence for the viability of the AMD hardware and networking stack for pretraining at the 8B scale, which is larger than prior public pretraining runs on combined AMD GPU and networking hardware that we are aware of. However, training ZAYA1-8B required only data parallelism plus context parallelism during context-extension phases. Although we have stress-tested other parallelism strategies, including cross-node parallelism, further scaling is needed to validate the stack for substantially larger models. 

It was also unclear at the outset whether our architectural changes would support effective reasoning and long-context behavior. This was a particular concern for CCA, which we had not previously tested at long contexts. ZAYA1-8B’s performance on long-context and reasoning benchmarks suggests that CCA’s advantage over attention variants such as MLA and GQA can be maintained in these settings. However, 8B total parameters is still modest relative to frontier-scale models, and architectural behavior at larger scales remains to be tested. 

The evaluation profile of ZAYA1-8B is uneven in a useful way. The model is strongest on reasoning-heavy mathematics and code, where it is competitive with much larger models and, under TTC, approaches frontier mathematical performance. On knowledge-heavy and broad factual evaluations such as MMLU-Pro and GPQA-D, ZAYA1-8B remains strong for its active-parameter scale but does not fully close the gap to substantially larger models. This pattern is consistent with the intuition that reasoning performance and factual storage scale differently: a small-active model can express strong algorithmic reasoning while still having less capacity for broad memorized knowledge than much larger models. 

This motivates a useful direction for future systems: smallactive reasoning models may be especially effective when paired with test-time compute and external retrieval. Rather than requiring all capability to be stored in parameters, such systems can combine a compact reasoning core, cheap parallel inference, and external knowledge sources. ZAYA1-8B is one example of this tradeoff, but establishing the generality of this pattern requires further scaling and controlled comparisons. 

Some of ZAYA1-8B’s reasoning strength may come from its relatively deep architecture. ZAYA1-8B has 40 layers, compared with 36 layers for Qwen3-4B and 16 layers for OLMoE at a similar total-parameter scale. We hypothesize that this depth helps the model represent more serial computation within a single forward pass, which may be useful for reasoning. The residual-scaling mechanism may also help preserve the contribution of later layers by controlling residual-norm growth, while the ZAYA1 router improves routing stability and expert specialization. We treat these as architectural hypotheses supported by our training experience rather than as isolated causal claims; controlled ablations at larger scale are left for future work. 

On agentic tasks, ZAYA1-8B trails models whose posttraining emphasizes multi-turn tool use, especially on bench- 

25 

marks such as BFCL-v4 and _τ_[2] . This is an expected consequence of the release scope: our midtraining, SFT, and RL budgets prioritize math, code, TTC aggregation, and instruction following rather than dedicated multi-turn agentic RL. We view this as a data-and-training emphasis gap rather than an architectural limitation. Scaling agentic data and RL remains a priority for future releases. 

## ACKNOWLEDGEMENTS 

We would like to thank Paul White, Danny Martinelli, Steven Brook, Kristina Zhao, and Krithik Puthalath for help with the release. We would also like to thank Yuankai Chen and Yao Fu from AMD for their support and close technical collaboration. 

## REFERENCES 

- Milad Aghajohari, Kamran Chitsaz, Amirhossein Kazemnejad, Sarath Chandar, Alessandro Sordoni, Aaron Courville, and Siva Reddy. The markovian thinker: Architectureagnostic linear scaling of reasoning. _arXiv preprint arXiv:2510.06557_ , 2025. 

- Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. _arXiv preprint arXiv:2305.13245_ , 2023. 

- Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, and Bryan Catanzaro. Front-loading reasoning: The synergy between pretraining and post-training data. _arXiv preprint arXiv:2510.03264_ , 2025. 

- AMD. The AMD CDNA™3 architecture. White paper, AMD, 2024. URL https://www.amd.com/content/ dam/amd/en/documents/instinct-tech-docs/white-papers/ amd-cdna-3-white-paper.pdf. 

- AMD Pensando. AMD Pollara 400 Card. https://www.amd.com/content/dam/amd/en/ documents/pensando-technical-docs/product-briefs/ pollara-product-brief.pdf, 2024. 

- Quentin Anthony, Yury Tokpanov, Skyler Szot, Srivatsan Rajagopal, Praneeth Medepalli, Anna Golubeva, Vasu Shyam, Robert Washbourne, Rishi Iyer, Ansh Chaurasia, et al. Training foundation models on a full-stack amd platform: Compute, networking, and system design. _arXiv preprint arXiv:2511.17127_ , 2025. 

- Karl J Åström and Tore Hägglund. Pid control. _IEEE Control Systems Magazine_ , 1066:30–31, 2006. 

- Brian Bartoldson. Cheaply approximating kl against an ema: an async rl hack. _https://brianbartoldson.wordpress.com/2026/05/04/cheaplyapproximating-kl-against-an-ema-an-async-rl-hack/_ , 2026. 

- Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. _arXiv preprint arXiv:2506.13585_ , 2025. 

- Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_ , 2025. 

- Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. _arXiv preprint arXiv:2505.22617_ , 2025. 

- Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-ofexperts language models, 2024. URL https://arxiv.org/abs/ 2401.06066. 

- DeepSeek-AI. Deepseek-v3.2-exp: Boosting long-context efficiency with deepseek sparse attention, 2025a. 

- DeepSeek-AI. Deepseek-v3 technical report, 2025b. URL https://arxiv.org/abs/2412.19437. 

- DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models, 2025c. 

- Jasper Dekoninck, Nikola Jovanovi´c, Tim Gehrunger, Kári Rögnvalddson, Ivo Petrov, Chenhao Sun, and Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. 2026. URL https: //arxiv.org/abs/2605.00674. 

- Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, and Stefano Soatto. Fewer truncations improve language modeling, 2024. URL https: //arxiv.org/abs/2404.10830. 

- William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. _Journal of Machine Learning Research_ , 23(120):1–39, 2022. 

- Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, and Beren Millidge. Compressed convolutional attention: Efficient attention in a compressed latent space, 2025. URL https://arxiv.org/abs/2510.04476. 

- GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, 

26 

Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763. 

- Jingcheng Hu, Yinmin Zhang, Shijie Shang, Xiaobo Yang, Yue Peng, Zhewei Huang, Hebin Zhou, Xin Wu, Jie Cheng, Fanqi Wan, et al. Pacore: Learning to scale test-time compute with parallel coordinated reasoning. _arXiv preprint arXiv:2601.05593_ , 2026. 

- Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. _URL https://kellerjordan. github. io/posts/muon_ , 6, 2024. 

- Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S Dhillon, David Brandfonbrener, and Rishabh Agarwal. The art of scaling reinforcement learning compute for llms. _arXiv preprint arXiv:2510.13786_ , 2025. 

- Dan Lee, Seungwook Han, Akarsh Kumar, and Pulkit Agrawal. Training language models via neural cellular automata. _arXiv preprint arXiv:2603.10055_ , 2026. doi: 10. 48550/arXiv.2603.10055. URL https://arxiv.org/abs/2603. 10055. 

- Junlong Li, Daya Guo, Dejian Yang, Runxin Xu, Yu Wu, and Junxian He. Codei/o: Condensing reasoning patterns via code input-output prediction, 2025. URL https://arxiv.org/ abs/2502.07316. 

- Ming Li and Paul Vitányi. _An Introduction to Kolmogorov Complexity and Its Applications_ . Texts in Computer Science. Springer, Cham, 4 edition, 2019. ISBN 978-3-03011297-4. doi: 10.1007/978-3-030-11298-1. URL https: 

//doi.org/10.1007/978-3-030-11298-1. 

- Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. _arXiv preprint arXiv:2502.16982_ , 2025. 

- Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025. _URL https://arxiv. org/abs/2503.20783_ , 2024. 

- Frederic M. Lord. _Applications of Item Response Theory to Practical Testing Problems_ . Lawrence Erlbaum Associates, Hillsdale, NJ, 1980. ISBN 089859006X. 

- Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, and Fuli Luo. Stabilizing moe reinforcement learning by aligning training and inference routers, 2025. URL https://arxiv.org/abs/2510.11370. 

- Mistral AI. Introducing Mistral Small 4. https://mistral.ai/ news/mistral-small-4, 2026. Accessed: 2026-05-06. 

- Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models. _arXiv preprint arXiv:2505.11711_ , 2025. 

- Sagnik Mukherjee, Lifan Yuan, Pavan Jayasinha, Dilek Hakkani-Tür, and Hao Peng. Do we need adam? surprisingly strong and sparse reinforcement learning with sgd in llms. _arXiv preprint arXiv:2602.07729_ , 2026a. 

- Sagnik Mukherjee, Lifan Yuan, Pavan Jayasinha, Dilek Hakkani-Tür, and Hao Peng. Do we need adam? surprisingly strong and sparse reinforcement learning with sgd in llms, 2026b. URL https://arxiv.org/abs/2602.07729. 

- Minh Nhat Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, and Ravid Shwartz-Ziv. Turning up the heat: Min-p sampling for creative and coherent llm outputs. _arXiv preprint arXiv:2407.01082_ , 2024. doi: 10. 48550/arXiv.2407.01082. URL https://arxiv.org/abs/2407. 01082. 

NVIDIA. Nemotron 3 Nano: Open, efficient mixture-of-experts hybrid Mamba-Transformer model for Agentic reasoning, 2025. URL https://research.nvidia.com/labs/nemotron/files/ NVIDIA-Nemotron-3-Nano-Technical-Report.pdf. Technical report. 

- OpenAI. GPT-5 System Card. https://openai.com/index/ gpt-5-system-card/, August 2025. Accessed: 2026-05-06. 

- Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, and Martin Jaggi. Denseformer: Enhancing information flow in transformers via depth weighted averaging. _Advances in neural information processing systems_ , 37:136479–136508, 2024. 

- Alexandre Piché, Ehsan Kamalloo, Rafael Pardinas, Xiaoyin Chen, and Dzmitry Bahdanau. Pipelinerl: Faster on-policy reinforcement learning for long sequence generation. _arXiv preprint arXiv:2509.19128_ , 2025. 

- Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Defeating the traininginference mismatch via fp16, 2025. URL https://arxiv.org/ 

27 

abs/2510.26788. 

- Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in llm reinforcement learning. _arXiv preprint arXiv:2602.04879_ , 2026. 

- Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, et al. Gated attention for large language models: Nonlinearity, sparsity, and attention-sink-free. _arXiv preprint arXiv:2505.06708_ , 2025. 

- Qwen Team. Qwen3-235B-A22B-Thinking-2507. Hugging Face model card, 2025. URL https://huggingface.co/Qwen/ Qwen3-235B-A22B-Thinking-2507. Accessed: 2026-0506. 

- Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixture-of-experts inference and training to power nextgeneration ai scale. In _International conference on machine learning_ , pp. 18332–18346. PMLR, 2022. 

- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. In _International Conference on Learning Representations_ , 2016. URL https://arxiv.org/abs/1701. 06538. 

- Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. Rl’s razor: Why online reinforcement learning forgets less. _arXiv preprint arXiv:2509.04259_ , 2025. doi: 10.48550/arXiv. 2509.04259. URL https://arxiv.org/abs/2509.04259. 

- Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In _Proceedings of the Twentieth European Conference on Computer Systems (EuroSys 2025)_ , pp. 1279–1297. ACM, 2025. doi: 10.1145/3689031.3696075. arXiv preprint arXiv:2409.19256. 

- Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/abs/ 2104.09864. 

- Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, and Andrea Zanette. Maximum likelihood reinforcement learning, 2026. URL https: //arxiv.org/abs/2602.02710. 

- FAIR CodeGen Team, Jade Copet, Quentin Carbonneaux, Gal Cohen, Jonas Gehring, Jacob Kahn, Jannik Kossen, Felix Kreuk, Emily McMilin, Michel Meyer, Yuxiang Wei, David Zhang, Kunhao Zheng, Jordi Armengol-Estapé, Pedram Bashiri, Maximilian Beck, Pierre Chambon, Abhishek Charnalia, Chris Cummins, Juliette Decugis, Zacharias V. Fisches, François Fleuret, Fabian Gloeckle, Alex Gu, Michael Hassid, Daniel Haziza, Badr Youbi Idrissi, Christian Keller, Rahul Kindi, Hugh Leather, Gallil Maimon, Aram Markosyan, Francisco Massa, Pierre-Emmanuel 

Mazaré, Vegard Mella, Naila Murray, Keyur Muzumdar, Peter O’Hearn, Matteo Pagliardini, Dmitrii Pedchenko, Tal Remez, Volker Seeker, Marco Selvi, Oren Sultan, Sida Wang, Luca Wehrstedt, Ori Yoran, Lingming Zhang, Taco Cohen, Yossi Adi, and Gabriel Synnaeve. Cwm: An open-weights llm for research on code generation with world models. _arXiv preprint arXiv:2510.02387_ , 2025a. doi: 10.48550/arXiv.2510.02387. URL https://arxiv.org/abs/ 2510.02387. 

- Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. _arXiv preprint arXiv:2507.20534_ , 2025b. 

Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Chenxiao Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Qizheng Gu, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yang Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Haoyu Lu, Lijun Lu, Yashuo Luo, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Zeyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Lin Sui, Xinjie Sun, Flood Sung, Yunpeng Tai, Heyi Tang, Jiawen Tao, Qifeng Teng, Chaoran Tian, Chensi Wang, Dinglu Wang, Feng Wang, Hailong Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Si Wang, Xinyuan Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Haoning Wu, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Jing Xu, Jing Xu, Junjie Yan, Yuzi Yan, Hao Yang, Xiaofei Yang, Yi Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Siyu Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian Zhao, Yikai Zhao, Zijia Zhao, 

28 

Huabin Zheng, Shaojie Zheng, Longguang Zhong, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Jinguo Zhu, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence, 2026. URL https://arxiv.org/abs/2507.20534. 

- Olmo Team, A Ettinger, A Bertsch, B Kuehl, D Graham, D Heineman, D Groeneveld, F Brahman, F Timbers, H Ivison, et al. Olmo 3. _arXiv preprint arXiv:2512.13961_ , pp. 23–25, 2025c. 

Prime Intellect Team. Intellect-3: Technical report, 2025a. URL https://huggingface.co/PrimeIntellect/INTELLECT-3. Qwen Team. Qwen3 technical report, 2025b. URL https: //arxiv.org/abs/2505.09388. 

- William R. Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. _Biometrika_ , 25(3–4):285–294, 1933. doi: 10.1093/ biomet/25.3-4.285. 

- Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. Towards greater leverage: Scaling laws for efficient mixture-of-experts language models. _arXiv preprint arXiv:2507.17702_ , 2025. 

Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476. Danlong Yuan, Tian Xie, Shaohan Huang, Zhuocheng Gong, Huishuai Zhang, Chong Luo, Furu Wei, and Dongyan Zhao. Shorten after you’re right: Lazy length penalties for reasoning rl. _arXiv preprint arXiv:2505.12284v3_ , 2025. 

Zhiyuan Zeng, Hamish Ivison, Yiping Wang, Lifan Yuan, Shuyue Stella Li, Zhuorui Ye, Siting Li, Jacqueline He, Runlong Zhou, Tong Chen, Chenyang Zhao, Yulia Tsvetkov, Simon Shaolei Du, Natasha Jaques, Hao Peng, Pang Wei Koh, and Hannaneh Hajishirzi. Rlve: Scaling up reinforcement learning for language models with adaptive verifiable environments. _arXiv preprint arXiv:2511.07317_ , 2025a. 

Zhiyuan Zeng, Hamish Ivison, Yiping Wang, Lifan Yuan, Shuyue Stella Li, Zhuorui Ye, Siting Li, Jacqueline He, Runlong Zhou, Tong Chen, et al. Rlve: Scaling up reinforcement learning for language models with adaptive verifiable environments. _arXiv preprint arXiv:2511.07317_ , 2025b. 

- Siddarth Venkatraman, Vineet Jain, Sarthak Mittal, Vedant Shah, Johan Obando-Ceron, Yoshua Bengio, Brian R Bartoldson, Bhavya Kailkhura, Guillaume Lajoie, Glen Berseth, et al. Recursive self-aggregation unlocks deep thinking in large language models. _arXiv preprint arXiv:2509.26626_ , 2025. 

- Zhilin Wang, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, Jimmy J. Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. Helpsteer2: Open-source dataset for training top-performing reward models, 2024. 

- Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open humanannotated preference data across diverse tasks and languages, 2025. URL https://arxiv.org/abs/2505.11475. 

- Anjiang Wei, Tarun Suresh, Jiannan Cao, Naveen Kannan, Yuheng Wu, Kai Yan, Thiago S. F. X. Teixeira, Ke Wang, and Alex Aiken. Codearc: Benchmarking reasoning capabilities of llm agents for inductive program synthesis, 2025. URL https://arxiv.org/abs/2503.23145. 

- Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, and Nick Haber. Just enough thinking: Efficient reasoning with adaptive length penalties reinforcement learning. _arXiv preprint arXiv:2506.05256_ , 2025. 

- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ , 2025. 

- Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, 

29 

APPENDIX A CLUSTER DETAILS 

Table XIV summarizes the hardware configuration of the compute, storage, and login nodes. All nodes also include separate local drives for the operating system. 

## APPENDIX B 

## STORAGE NODE SIZING AND I/O CALCULATIONS 

We analyze shared-storage requirements for dataset reads during training, assuming Megatron-style pretokenized corpora accessed through mmap or buffered reads on a dedicated storage fabric. Large sequential checkpoint writes are throughputbound rather than IOPS-bound and are not considered here. 

Let _G_ be the global batch size, _s_ the sequence length, _b_ bytes per token, _P_ the storage page size, _t_ the iteration time, and _I_ max the sustainable IOPS capacity. We introduce a _scatter factor σ ≥_ 1 to capture how much real dataset access patterns, including metadata touches, index probes, page-cache misses, and small random seeks, deviate from perfectly contiguous reads. Each iteration reads _G · s · b_ bytes, requiring 

**==> picture [179 x 21] intentionally omitted <==**

effective I/O operations. The sustained IOPS requirement is therefore 

**==> picture [214 x 22] intentionally omitted <==**

and the break-even iteration time under budget _I_ max is 

**==> picture [186 x 23] intentionally omitted <==**

We can estimate the scatter factor _σ_ from the average number of additional page faults per sample. Let _m_ denote the average count of additional page faults from metadata, *.idx probes, document-boundary straddles, or cold reads. If the ideal pages per sample are ( _s·b_ ) _/P_ , a practical approximation is 

**==> picture [165 x 22] intentionally omitted <==**

This interpolates between contiguous, warm-cache access ( _σ →_ 1) and fragmented, small-document regimes ( _σ >_ 1). In our experience with well-packed Megatron datasets, _σ ∈_ [1 _,_ 2] is typical; heavily fragmented or multi-shard random-seek workloads can reach _σ ∈_ [2 _,_ 8]. 

For the ZAYA1 training run with _G_ =4096, _s_ =4096, _b_ =4 B, _P_ =4096 B, _t_ =2 _._ 5 s, and _I_ max=70 _,_ 000 IOPS, each iteration reads 64 MiB across 16,384 pages. This requires approximately 6 _,_ 554 _·σ_ IOPS, with break-even time _t_ break _≈_ 0 _._ 234 _·σ_ s. At the observed _t_ =2 _._ 5 s, the run remains above the breakeven time even for _σ_ =8, indicating that the 70K IOPS storage budget is sufficient. 

Fig. 12: We briefly review the Compressed Convolutional Attention block (Figliolia et al., 2025). 

## APPENDIX C COMPRESSED CONVOLUTIONAL ATTENTION 

CCA (shown in Figure 12) modifies the attention block so that attention is performed in a compressed latent space. This reduces both memory and FLOP costs. In prior work, CCA outperformed alternatives such as GQA and MLA on perplexity and training/inference FLOPs while enabling high KV-cache compression, which is important for fast decoding. 

CCA has several core components: 

- **Low-Rank Projections:** Low-rank down-projections reduce compute and memory. 

- **Sequence-Mixing Convolutions:** A short convolution and grouped head-wise convolution act as lightweight preconditioners before attention. 

- **Value Head Time-Delay:** A time delay of one token is applied to half of the value heads. 

- **Skip Connections and Normalization:** The architecture utilizes query/key mean skip connections to enforce representational similarity, coupled with an RMSNorm layer that applies a head-wise temperature strictly to the keys. To stabilize training, we modify the standard CCA mechanism by scaling the keys by a learned temperature _T_ instead of exp( _T_ ). Because exp( _T_ ) can easily grow excessively large, the query-key inner product becomes susceptible to unbounded growth. Our linear parameterization successfully mitigates the resulting maximum attention logit instability, drawing parallels to similar stabilization efforts in MLA (Team et al., 2026). The 

30 

|**Node**|**Component**|**Specifcation**|**Details**|
|---|---|---|---|
||GPUs|8_×_ AMD MI300X GPUs (AMD, 2024)|Connected via Infnity Fabric intra-node interconnect.|
||RAM|2 TB DDR5|16 _×_ 128GB Samsung M321RAJA0MB0-CWMNY|
|Compute|||DIMMs running at 5600 MT/s.|
||CPU|Dual-socket Intel Xeon Platinum 8570|Each socket has 56 physical cores and 2 threads per|
||||core, and is connected to 1 TB of RAM (8 DIMMs).|
||Networking|8_×_ Pollara 400 NICs (AMD Pensando, 2024) + 1|Each Pollara NIC provides 400 Gb/s; the Pensando NIC|
|||Pensando DSC 200 GbE NIC|is used for data and checkpoint transfer.|
||Storage|25.6 TB NVMe|8_×_ Micron MTFDKCC3T2TGQ-1BK1DABDB drives,|
||||3.2 TB each.|
||RAM|256 GB DDR5|16 _×_ 16GB<br>Samsung<br>M321R2GA3BB6-CQKET|
|Storage|CPU|Dual-socket Intel Xeon Gold 6426Y|DIMMs running at 4800 MT/s.<br>Each socket has 16 physical cores and 2 threads per|
||||core, and is connected to 128 GB of RAM (8 DIMMs).|
||Networking|1 Pensando DSC 100 GbE NIC|Used for data and checkpoint transfer.|
||Storage|120 TB RAID0|Single RAID0 array, /dev/md127, built from 16_×_|
||||Micron 7450 MTFDKCC7T6TFR NVMe drives, each|
||||with approximately 7.6 TB.|
||RAM|80 GB system memory|5_×_16GB DIMMs under QEMU.|
|Login|CPU|Virtualized dual-socket Intel Xeon (Sapphire Rapids)|Each socket has 8 cores and 2 threads per core; virtu-|
||||alized under KVM.|
||Storage|Approximately 1 TB|Three virtual disks: vda (100 GB), vdb (520 GB), and|
||||vdc (520 GB).|



TABLE XIV: Hardware configuration of the compute, storage, and login nodes. 

scaled query-key inner product is then upper bounded (assuming QK norm) to: 

**==> picture [152 x 24] intentionally omitted <==**

where _dh_ is the head dimension. 

These additions to precondition query and key, especially the convolutions, provide expressivity and nonlinearity that allow CCA to match or exceed full attention while requiring less compute and memory. 

The CCA block can also operate in ‘GQA-mode’, where multiple KV heads are shared across query heads. This further reduces decoding cost through additional KV-cache compression. We call the combined method CCGQA. For ZAYA1-8B, we use CCGQA with 2 KV heads for 8 query heads, on top of 2 _×_ query compression, for an 8 _×_ KV-cache compression ratio relative to full multi-head attention. 

## APPENDIX D 

## EXPERT REDUNDANCY DIAGNOSTIC 

We include a small diagnostic for expert redundancy in the MoE feed-forward blocks. This diagnostic is not an evaluation benchmark and should not be interpreted as a complete measure of expert specialization. Its purpose is narrower: to check whether ZAYA1-8B’s experts appear unusually collapsed or redundant relative to other public MoE checkpoints. 

For each MoE layer _l_ and expert _e_ , let _Wl,e_ denote the expert projection under study. We compute the top- _d_ singular subspace _Ql,e ∈_ R _[D][×][d]_ with _d_ = 128 and orthonormal columns. For the input-side metric, _Ql,e_ is the right singular subspace 

of the full first FFN projection. For gated FFNs with separate branches, we concatenate the gate and up projections along the FFN dimension before computing the right singular subspace; for ZAYA1-8B, this corresponds to the fused linear_fc1 projection. For the output-side metric, _Ql,e_ is the left singular subspace of the expert output projection, corresponding to linear_fc2, down_proj, or w2. 

For two experts _i_ and _j_ in the same layer, we define 

**==> picture [176 x 21] intentionally omitted <==**

where _∥· ∥F_ is the Frobenius norm. The reported score averages _sl,i,j_ over all off-diagonal expert pairs and all MoE layers. Larger values indicate that experts share more of the same input or output directions; smaller values indicate more distinct expert subspaces under this particular projectionspace diagnostic. For independent random _d_ -dimensional subspaces in R _[D]_ , the expected value of this metric is _d/D_ . All models in Table XV have _D_ = 2048 for both the input and output comparison spaces, so the random-subspace floor is 128 _/_ 2048 = 0 _._ 0625. We therefore report both the raw overlap and the random-normalized ratio 

**==> picture [148 x 21] intentionally omitted <==**

The “Var.” columns report the mean fraction of projection Frobenius energy captured by the top-128 singular directions; they are included as spectral context, not as a normalization of the overlap score. 

Overall, this diagnostic does not indicate unusual expert collapse in ZAYA1-8B. On the input-side first projection, 

31 

|Model|MoE|layers|Experts/layer|Input|overlap|Input _ρ_|Input Var.|Output overlap|Output _ρ_|Output Var.|
|---|---|---|---|---|---|---|---|---|---|---|
|ZAYA1-8B||40|16|0_._0904_±_|_±_0_._0244|1_._45_×_|17.8%|0_._0990_±_0_._0289|1_._58_×_|22.7%|
|LFM2-8B-A1B||22|32|0_._1328_±_|_±_0_._0502|2_._12_×_|25.4%|0_._1174_±_0_._0441|1_._88_×_|28.5%|
|OLMoE-1B-7B||16|64|0_._1031_±_|_±_0_._0397|1_._65_×_|29.7%|0_._0872_±_0_._0256|1_._40_×_|34.0%|
|Qwen3-30B-A3B||48|128|0_._0923_±_|_±_0_._0333|1_._48_×_|31.2%|0_._0851_±_0_._0226|1_._36_×_|39.2%|



TABLE XV: Mean within-layer expert subspace overlap with _d_ = 128. Input overlap uses the right singular subspace of the full first FFN projection; output overlap uses the left singular subspace of the expert output projection. Standard deviations are over all off-diagonal expert pairs across analyzed MoE layers. All rows use a 2048-dimensional comparison space, giving a random-subspace baseline of _d/D_ = 0 _._ 0625. The Qwen row uses Qwen3-30B-A3B-Thinking-2507. 

Fig. 13: Expert redundancy diagnostic. Panels (a) and (b) show global mean overlap for input and output projections; bar labels report the random-normalized ratio _ρ_ . Panels (c) and (d) show raw overlap averaged over depth quartiles. The dashed horizontal line is the random-subspace baseline _d/D_ = 0 _._ 0625. ZAYA1-8B is not an outlier toward higher expert overlap: its first-projection input overlap is close to Qwen and below LFM2 and OLMoE, while its output-projection overlap is intermediate between LFM2 and the lower-overlap OLMoE/Qwen group. 

ZAYA1-8B is 1 _._ 45 _×_ the random-subspace baseline, close to Qwen’s 1 _._ 48 _×_ and below LFM2 and OLMoE. On the output projection, ZAYA1-8B is intermediate at 1 _._ 58 _×_ random: higher than OLMoE and Qwen, but below LFM2. ZAYA18B’s input-side overlap also rises with depth, from 0 _._ 075 in the first depth quartile to 0 _._ 105 in the last, indicating more shared input directions in later MoE layers. We therefore use this appendix only as evidence against obvious expert collapse, not as a claim that ZAYA1-8B has stronger outputside specialization than all baselines. 

## APPENDIX E RARE-TOKEN SAMPLING ARTIFACTS 

We define gibberish as brief, recoverable episodes of seemingly random tokens that interrupt the CoT, and we argue that it is distinct from other forms of degeneration, which may involve cache corruption, numeric issues, repeats, etc. because 

these other issues are generally non-recoverable or at least persistent in the CoT. 

In this set of examples (Table XVI), it is clear that the gibberish token has no relationship to the prefix, and interestingly, the suffix seems to mostly ignore it. However, this does not mean it is benign, because we found that it is likely to reinforce itself as a result of successful (rewarded) traces containing gibberish tokens. 

As a heuristic for gibberish detection we rely on the method described in (Team et al., 2025a) which combines a low logprob mask (2 nats below uniform) with token ID mask (a proxy for token rarity). While this is a good indicator/metric in practice, it is a somewhat arbitrary hard threshold and in our experience it isn’t precise or sensitive enough to be used in training for reward zeroing or loss masking. 

We collected traces and their top-3 token probabilities using the vLLM streaming API and aborted the traces around any 

32 

|#|**Prefx**|**Gibberish**|**Suffx**|
|---|---|---|---|
|0|Let ’ s count underscores : – The outer parentheses : open before underscore : {} - Inside|<unused5148>|? Actually start of right part : ( _ | ( _ & (|
|1|, we have : - frst underscore : c - second underscore : inside the ’& ’? Actually|U+1490|. Wait after the " | (" we have ’_ -> maybe a variable|
|2|as : ( c | ( d & ( e ^ ( f | g )))) . Then outer parentheses|U+96BC|might not be needed . Thus we have outer parentheses around the|
|3|(_ | _) )))) ‘ which we can view as ’(’ followed by ’_ then ’| ’ then ’(’ then|<unused521>|etc followed by ’)) ’? Actually need to count correctly . Let ’|
|4|then ’^ ’, then underscore , then ")" -> you have an outer parentheses containing|U+52DF|?). That yields the expression "(" _ ^ _ ") "? That seems odd|
|5|the outer parentheses of the XOR is the one after "(" , we have "(" underscore "^ " then|U+12D6|? Let ’ s expression parse : The substring is " (_ ^ (_|
|6|just "(" "_ ", "^ ", "_" ") ". That ’ s parentheses open and close on a level containing|U+78C1|? Possibly they just open parentheses for each underscore : ’(’ then|



TABLE XVI: Traces at the gibberish token position. 

**==> picture [253 x 252] intentionally omitted <==**

**----- Start of picture text -----**<br>
Logprob around flagged token<br>0.0<br>2.5<br>5.0<br>7.5<br>10.0<br>12.5 abort median<br>abort p25/75<br>15.0 end (random baseline) median<br>end (random baseline) p25/75<br>Entropy around flagged token<br>2.0<br>1.5<br>1.0<br>0.5 abort median<br>abort p25/75<br>end (random baseline) median<br>0.0 end (random baseline) p25/75<br>1.0 Top-3 probability mass   abort (median)<br>0.8<br>0.6<br>0.4<br>0.2 abort top-1<br>abort top-2<br>abort top-3<br>0.0<br>250 200 150 100 50 0 50 100<br>tokens relative to flagged token<br>logprob<br>entropy (nats)<br>probability<br>**----- End of picture text -----**<br>


**==> picture [245 x 75] intentionally omitted <==**

**----- Start of picture text -----**<br>
part p = 0.34<br>char p = 0.23<br>line 3 has  ̀' | ' ̀ then spaces etc . The leftmost line p = 0.12<br>…<br>U+0FB3 p = 2 . 6  ×  10 [−] [7]<br>**----- End of picture text -----**<br>


Fig. 15: Top token choices for an otherwise coherent trace that sampled a random Tibetan token. In this example the model was attempting to reason about a challenging isometric ASCII drawing problem from the RLVE gym known as BlockImage. 

with a threshold of 10 _[−]_[5] reduces this to 0.1%. In training, to prevent engine/trainer mismatch, we recommend implementing min- _p_ replay, which exposes the kept/omitted token IDs to the trainer for consistency in the min- _p_ renormalization. 

Fig. 14: Aggregated tokenwise metrics for gibberish responses compared to normal responses. 

gibberish-flagged tokens as they arose using the heuristic mask. While entropy is spiking in those cases, the most probable tokens still constitute a significant portion of the total probability. Looking into the top- _k_ tokens themselves, we found that the top token was a coherent continuation of the prefix, with the most common case being simply "the" as with examples 0, 1, 4, 6 in the table above. In another more context-dependent case, any of the top 3 most likely tokens would have been a coherent alternative (Figure 15). 

In general, the baseline/prefix entropy of traces aborted with gibberish tokens tends to be higher (Figure 14). This could be because the model is challenged by a difficult problem and not confident as a result. Around the flagged token, entropy spikes, and while the probability mass in the top 3 tokens is median 0.7, a significant mass remains in the tail of the distribution. For this reason, we believe that despite the sampled token having an extremely low probability, it was chosen because it was one of many possible low-probability tokens. 

A natural solution is to use min- _p_ (Nguyen et al., 2024) sampling to eliminate extremely low probability choices from the sampling distribution. In our experiments, we find that even in the most degraded test checkpoint available to us, with a baseline rate of 19.9% of flagged responses, min- _p_ sampling 

33 

