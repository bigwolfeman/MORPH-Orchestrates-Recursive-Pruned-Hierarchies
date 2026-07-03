# Efficient Pre-Training with Token Superposition

- **Authors:** Bowen Peng, Theo Gigant, Jeffrey Quesnelle (Nous Research)
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2605.06546
- **MORPH uses:** Token Superposition Training (TST): packing multiple tokens into a single position via multi-hot cross-entropy, extending the fused-CE path with no new kernel. 

---

## **Efficient Pre-Training with Token Superposition** 

**Bowen Peng[*]** Nous Research `bloc@nousresearch.com` 

**Th´eo Gigant[*] Jeffrey Quesnelle** Nous Research Nous Research `theo@nousresearch.com emozilla@nousresearch.com` 

## **Abstract** 

Pre-training of Large Language Models is often prohibitively expensive and inefficient at scale, requiring complex and invasive modifications in order to achieve high data throughput. In this work, we present Token-Superposition Training (TST), a simple drop-in method that significantly improves the data throughput per FLOPs during pre-training without modifying the parallelism, optimizer, tokenizer, data, or model architecture. TST is done in two phases: (i) A highly efficient superposition phase where we combine many contiguous tokens into one bag and train using a multi-hot cross-entropy (MCE) objective, and (ii) a recovery phase where we revert back to standard training. We extensively evaluate TST on the scale of 270M and 600M parameters and validate on 3B and a 10B A1B mixture of experts model, demonstrating that it is highly robust in different settings. Ultimately, TST consistently outperforms baseline loss and downstream evaluations, and under equal-loss settings, TST yields up to a 2.5x reduction in total pre-training time at the 10B A1B scale. 

**==> picture [238 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
Return to<br>77 .. 50 baseline regime Baseline<br>6 . 5 TST<br>6 . 0<br>5 . 5<br>5 . 0<br>4 . 5<br>4 . 0<br>3 . 5<br>3 . 0<br>2 . 8<br>22 .. 64 2.5x speedup<br>2 . 2<br>Training Steps<br>0 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000110000120000<br>Training Loss<br>**----- End of picture text -----**<br>


Figure 1: Loss curves during the pre-training of two Qwen3-like MoE models (10B-A1B) with baseline pretraining and token superposition training (TST) where we stop the training early to match an equal training loss. The baseline training sees 1.05T tokens while the TST training sees 2T data tokens. Every step in all conditions are equal-FLOPs, thus the speedup can be directly computed w.r.t. the number of steps. More details are in Table 1. 

> *Equal contribution. 

Preprint. 

## **1 Introduction** 

The rapid proliferation of modern generalist Large Language Models (LLMs) has been driven not only by increases in model size, but critically, by aggressive data scaling [53, 21, 6, 34, 5, 57, 1], with recent training regimes often overtraining [16] well beyond compute-optimal estimates to maximize performance at inference time. In this data-hungry paradigm, one of the major concerns during pre-training is the efficiency in which raw text is consumed given a fixed amount of compute. 

Recent advances in language modeling pre-training efficiency can be broadly organized into three categories: 

1. **Information maximization** : maximizing the information given per sample through improved input priors and representations (better tokenization, BPE [48], SuperBPE [35], Unigram [29], _n_ -gram hashing [36, 9]) and richer training signals (auxiliary losses, including multi-token prediction [20, 34], order-augmented objectives [64]). 

2. **Compute sparsity** : keeping the input representation fixed but reducing the FLOPs required to process each token by activating only a subset of parameters or attending to a subset of positions. This constitutes a _compute-level_ prior: similar expressivity, less work per token (sparse mixture-of-experts [26, 13], sparse attention [59, 58]). 

3. **Compressive modeling** : learning to _further compress_ the representation within the model itself, reducing the number of representations that flow through the expensive layers. This constitutes a _representation-level_ prior: fewer tokens internally, but dense computation on each (e.g., Bolmo [44], H-Net [25], Byte-Latent Transformer [46], Autoregressive UNet [54] Perceiver-Resampler [2]). 

Note that some of these works also touch on inference-time efficiency, which we acknowledge as an important line of work but orthogonal to our focus on pre-training efficiency. While some methods above confound and mix both training-time and inference-time efficiency, other methods focus on inference-time efficiency only ( _e.g._ diffusion [45], speculative decoding [31]), and almost none of them have decoupled the training-time efficiency and focus solely on this latter part (except concurrent work from Zheng et al. [62]). 

Many recent works refute the training-time and inference-time efficiency equivalence, where they show that scaling up inference-time compute independently of training time compute can improve performance on downstream tasks (e.g. CoT/reasoning models [55], ParScale [8], looped language models [63]), thus it is important for our method to be only used during training, and keep the model architecture and expressivity “untouched” for inference compared to the baseline in order to minimize any confounding factors. 

Furthermore, recent evidence suggests that the performance advantage of subword-based over bytelevel language models is largely driven by increased training sample throughput [18]. Building on this insight, we investigate whether LLM training efficiency can be further optimized by maximizing training-time throughput independently of the model’s inference-time architecture. 

Finally, the monolithic pre-training paradigm is currently shifting towards two-stage or multi-stage pre-training schemes. Many recent works have found that better or more efficient training methods can be used for the first stage of training [44, 23, 30], and then find that the model is elastic enough to quickly adapt to the final desired model behavior, often saving on training cost and/or resulting in a better quality model. However, most prior works focus on the mid-training or post-training regime, but we contextualize this work by showing that the same ideas can be applied to pre-training. 

We introduce Token-Superposition Training (TST), a method that tries to improve training efficiency by increasing token throughput while still priming the model for the desired task of autoregressive prediction. Our work starts with this fundamentally different perspective: 

## **Perspective** 

Can we improve pre-training efficiency by forcing a higher token throughput during training, without modifying the final model architecture and its inference dynamics? 

2 

After completion of this work, we became aware of Shao et al. [50], which independently proposed the same core mechanism: averaging consecutive token embeddings as input, predicting all tokens in the next group via cross-entropy with a single head, and transferring to standard token-level training in a second phase, with only minor differences from ours on the algorithm side. Although our work converges to a very similar method, we propose a different view on the problem and extend the research in several directions, with more details in Section 2.4. 

## **2 Related works** 

## **2.1 Alternative Prediction Objectives** 

In contrast to standard autoregressive next-token prediction, several studies have explored alternative training objectives to improve representation learning. Tay et al. [52] introduced the _mixtureof-denoisers_ framework, which unifies diverse denoising tasks—such as span corruption and causal language modeling to provide superior generalization across architectures. In the context of encoder models, Gisserot-Boukhlef et al. [19] demonstrated that a two-stage pretraining schedule, transitioning from causal language modeling to masked language modeling (MLM), outperforms standard MLM baselines in some downstream tasks. 

## **2.2 Auxiliary Prediction of Future Representations** 

Recent pre-training literature introduced auxiliary losses to move beyond single-token targets, with the goal of increasing the information density per gradient step. Gloeckle et al. [20] introduced _multitoken prediction_ (MTP), using _k_ independent heads to predict the next _k_ tokens simultaneously. Although MTP improves sample efficiency in some cases and has been featured in major state-ofthe-art LLM pre-training runs [34], it has limited benefit to smaller models and requires the tuning of an extra hyper-parameter _k_ while introducing additional parameters. 

Other approaches explore using auxiliary losses with targets using representations involving future tokens. DeepseekV3 [34] uses a modified version of MTP, featuring cascaded predictions using additional MTP modules. Zuhri et al. [64] replace MTP with the prediction of the relative order of future tokens, relaxing the complexity of MTP by simplifying the task and requiring only a single additional head. Liu et al. [37] proposed _next concept prediction_ with predicted concepts covering segments spanning multiple tokens. Mahajan et al. [41] introduced _future summary prediction_ , where the model predicts a compressed representation of future tokens utilizing either hand-crafted bag-of-words representations or learned latent features from a reversed sequence model. Notably, an entry within the `modded-nanogpt` speedrun [27] has implemented an MTP-inspired loss that shares conceptual similarities with the _next bag-of-tokens_ prediction explored in our work. These works illustrate that additional signal involving predicting representations of future tokens can yield substantial gains in pre-training sample efficiency. 

## **2.3 Input Granularity** 

Input granularity is a fundamental hyperparameter in both vision and language modeling. In Vision Transformers [14], patch size serves as a primary control for the trade-off between FLOPs and performance. Anagnostidis et al. [4] showed that scheduling patch size from coarse to fine-grained during training improves isoFLOPs performance for Vision Transformers. In LLMs, this granularity is typically set by the tokenizer’s _fertility_ , _i.e._ the average number of tokens required to represent a word. Subword tokenizers provide a coarser-grained view of text compared to byte-level or unicode representations. Liu et al. [35] further merge BPE tokens into supertokens, resulting in coarser tokens exhibiting improved training performance compared to regular BPE. Recent work by Minixhofer et al. [44] explored coarse-to-fine distillation of LLMs by resuming subword-level pretraining with byte-level objectives. Furthermore, Zheng et al. [62] demonstrated that including mixed granularities at training-time, through both compressed and raw byte representations, resulted in better byte-level performance compared to raw byte representation only. Gigant et al. [18] investigated the performance gap between subword and byte-level models, attributing the subword advantage largely to increased _sample throughput_ at isoFLOPs, a direct consequence of the coarser tokens resulting from the sequence compression effect of subword tokenization. These findings suggest that coarser training-time input granularity is a key driver of modern LLM training speed. In this work, we lever- 

3 

age both the additional signal from future tokens representations and a coarser input granularity at training-time to improve the model pre-training efficiency. Crucially, during the second phase of training, we return to the baseline granularity and loss. 

## **2.4 Patch-Level Training** 

Most directly related to our work, Shao et al. [50] introduced patch-level training for LLMs, in which consecutive token embeddings are averaged into ”patches” and the model trained to predict all tokens in the next patch using a single output head with cross-entropy loss. After patch-level training, the model reverts to standard token-level training. This is algorithmically identical to what we term Token Superposition Training, but with major differences in background theory and execution, which we attribute to conceptual convergence. Shao et al. [50] frame patch-level training as reducing total FLOPs for a fixed dataset, whereas we propose TST as two orthogonal but additive processes that increase token throughput at constant per-step FLOPs (Section 5.2) following the throughput hypothesis of Gigant et al. [18], where we make careful implementation decisions that result in accurate comparisons which simplifies scaling, and justify the choice of the multi-hot loss function based on empirical evidence (Section 3.2). Shao et al. [50] demonstrated the approach with trainings up to 2.7B parameters on 360B tokens, achieving a total training cost of 0.5 _×_ . We extend this with an extensive search of the hyperparameter space and demonstrate successful scaling at the 10B parameters, 2T token scale. They also observed that maintaining the same architecture (without additional projection layers) across both phases is critical for successful recovery. This finding aligns with our representation alignment analysis (Section 5.3). 

## **3 Methodology** 

Token Superposition involves two small changes when compared to baseline next-token prediction training, illustrated in Figure 2. In the Token Superposition Training (TST) regime, an LLM processes superpositions of token embeddings obtained via averaging the embeddings of contiguous tokens in non-overlapping _s_ -grams. Each such superposed latent token, which we call ”s-tokens”, results in a single prediction, compared against the next non-overlapping bag-of-tokens. 

The superposition objective is semi-causal and semi-autoregressive. In other words, the model still broadly predicts the input sequence from left to right, but we lose the ordering of the tokens in the superposition bag and the sampling capabilities during inference. Not surprisingly, if used as-is, a model trained using only TST has nonsensical outputs which represent a mixed probability of any of the _s_ future tokens. In order to remedy this, we opt for the simplest method which is to revert to training with the standard next-token causal prediction objective after some amounts of steps, where we define _r_ as the ratio of steps where we train with TST. In the second stage, we resume training from the saved checkpoint with the TST code fully removed to avoid any possibility of experimental and results contamination. 

We leave more complex conversion methods or other potential uses cases for TST as future work, where for example a model trained with TST might have informative latents that can be used to efficiently predict or verify multiple tokens at once, or using a TST-ed model as a compressive prior or encoder model. 

## **3.1 Input Superposition: Bag-of-Token-Embeddings** 

First, the contiguous tokenized data sequence of shape _B×L×V_ is segmented into **non-overlapping** contiguous segments of _s_ tokens, called bags. Here, _B_ denotes the batch size, _L_ the data token sequence length and _V_ the vocabulary size of the tokenizer. This gives us a bagged view of the data in the shape of _B ×l×s×V_ , where _s_ is the superposition bag size and _l_ the latent ”s-token”sequence length. 

In the embedding layer of the model, a single latent ”s-token” is created by superposition of the tokens in a bag via the average of their embeddings, resulting in a final shape of _B × l × d_ , where _d_ is the residual dimension of the model. The code is shown in Appendix A. 

As the model performs computations over a coarser-grained representation of the input text, it processes _s_ times more data tokens for every FLOPs used on the latent ”s-tokens”. Therefore, in order 

4 

**==> picture [392 x 303] intentionally omitted <==**

**----- Start of picture text -----**<br>
...<br>ti +2 Token<br>ti +1 Embeddings<br>ti ei LLM pi CE Loss ti +1 Next TokenPrediction<br>... Next Token<br>Processed length: L Prediction<br>...<br>ti +2 Token ti + k<br>ti +1 Embeddings ti +2<br>Multi-Token<br>ti ei LLM pi ti +1<br>CE Loss Prediction [20]<br>... Multi-Token<br>Processed length: L Prediction<br>... Supertoken<br>ti + s Embeddings<br>ti +1 Super-token e [′] j LLM p [′] j CE Loss Sj +1 SuperBPE [35]<br>ti Next Supertoken<br>Prediction<br>... Processed length: L [′] < L<br>... Superposed Token ti +2 s<br>ti + s Embeddings ti + s +2 Token<br>ti +1 mean� e [′] ⌊i/s⌋ LLM p [′] ⌊i/s⌋ Multi-hotCE Loss ti + s +1 Superposition(ours)<br>ti Next Bag-of-Tokens<br>Prediction<br>... Processed length: ⌊L/s⌋<br>L<br>Sequence length:<br>L<br>Sequence length:<br>L<br>Sequence length:<br>L<br>Sequence length:<br>**----- End of picture text -----**<br>


Figure 2: Comparison between standard next token prediction, TST and a few methods that superficially resemble TST. Note that this comparison is illustrative only for the purpose of understanding TST. 

to make a valid comparison during training, we choose to make every TST step equal-FLOPs to the baseline training by increasing the data sequence length _L_ by _s_ times during the superposition phase. 

This is what allows the model to ingest tokens at a higher rate during the superposition phase for the same amount of FLOPs per step compared to standard training. A faster alternative would be to increase the micro batch size by _s_ instead, but it is not explored in this work. 

## **3.2 Output Superposition: Next Bag-of-Tokens Prediction with Multi-hot Cross-Entropy Loss** 

Second, to predict the next ”bag of tokens” instead of a single token with a single output head, we modify the standard one-hot cross-entropy (CE) loss into a **multi-hot** cross entropy (MCE) loss. 

We have the standard CE loss with respect to the predicted logits **z** and the label index _y_ : 

**==> picture [270 x 30] intentionally omitted <==**

One possible expansion of this one-hot CE loss into a multi-hot MCE loss is simply to target an equal probability 1 _/s_ for each valid label (that together sum to 1). In our case, the label bag size _|_ **y** _|_ is equal to _s_ . With some rearranging, we get the following: 

**==> picture [318 x 32] intentionally omitted <==**

5 

The expanded derivation is shown in Eq. 4, Appendix C. 

If we do not care about the absolute loss value during training and only care about the training dynamics, we can drop the log _|_ **y** _|_ term from the loss, as it’s gradient is 0. This yields the final simplified form that we use: 

**==> picture [265 x 27] intentionally omitted <==**

Finally, the standard next token prediction labels are shifted to the left by _s −_ 1 before splitting into non-overlapping bags to preserve causality. This ensures that each bag of _s_ tokens at the positions [ _t, t_ + _s −_ 1] predicts the next bag of tokens at [ _t_ + _s, t_ + 2 _s −_ 1]. 

The simplified form allows us to take advantage of the highly optimized and compiled CE loss kernels that exist in major pre-training libraries and use them without major modifications to the training code. Further optimization can be done to reuse the inner log[�] _i_[exp(] _[z][i]_[)][ term, but that is] not explored in this work as the overhead to summing outside the loss with a for loop is negligible and uses no additional memory. The code is shown in Appendix A. 

We also tried many variants of possible multi-hot bag losses, such as Hinge loss [12] and Binary Cross-Entropy (BCE) loss [49]. But these results were significantly worse than the MCE loss we picked and even worse than the baseline training without TST. Although an interesting alternative to the MCE loss shown here was explored, with more details in Appendix C.2, the final chosen MCE loss has the best balance between soundness and simplicity. 

## **4 Experiments** 

We perform a battery of experiments that cover a wide range of superposition settings, varying the superposition bag size and ratio at different model scales and total durations. For all trainings, the TorchTitan [33] pre-training library was used with FSDP [61] parallelism, running on 64 NVIDIA B200 GPUs for the bigger models, and 8 B200 GPUs for the smaller models. 

For pre-training the smaller models, we broadly follow the training procedures as outlined in SmolLM [3]. For the 270M and 600M parameter models, we adapt and use the shape of the SmolLM2 models to fit the Llama3 [21] modeling code, with two modifications: we use the Llama38B tokenizer, and we do not tie the weights of the input embeddings to the output head. All other settings were kept the same (including layers, LM heads, inner dimensions, etc.). The untied embeddings make our models bigger, with the 270M matching SmolLM2-135M and the 600M matching SmolLM2-360M. Similarly for the 3B run, we matched it with SmolLM3-3B with the same modifications applied. Finally, the DCLM [32] dataset was used as the pre-training dataset, with a standard batch size of 2M tokens[1] for the SmolLM-like models. More detail is shown in Table 1. 

**==> picture [362 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
7 . 5 Baseline 7 . 5 Baseline 7 . 5 Baseline<br>7 . 0 TST 7 . 0 TST 7 . 0 TST<br>6 . 5 6 . 5 6 . 5<br>6 . 0 6 . 0 6 . 0<br>5 . 5 5 . 5 5 . 5<br>5 . 0 5 . 0 5 . 0<br>4 . 5 4 . 5 4 . 5<br>4 . 0 4 . 0 4 . 0<br>3 . 5 3 . 5 3 . 5<br>322 ... 098 322 ... 098 322 ... 098<br>2 . 7 2 . 7 2 . 7<br>2 . 6 2 . 6 2 . 6<br>Training Steps Training Steps Training Steps<br>(a) Equal-FLOPs (b) Equal-Loss (c) Equal-Data<br>0 10000 20000 0 10000 20000 30000 0 10000 20000 30000 40000 50000<br>Training Loss Training Loss Training Loss<br>**----- End of picture text -----**<br>


Figure 3: Same constraints comparisons between different baseline training and one Token Superposition Training, using superposition bag size _s_ = 6 and step ratio _r_ = 0 _._ 3. These are the four 3B parameter runs described in Table 1. 

> 1or ”s-tokens”in the case of TST, in which we call both ”equivalent-tokens” in the figures. 

6 

|**Model**<br>**Params**<br>**TST**<br>**Total**<br>**TST**<br>**TST**<br>**Total**<br>**B200-Hours**<br>**Steps**<br>**Steps**<br>**Bag Size**<br>**Tokens**<br>**Tokens**<br>**(**_↓_**)**|**Final Loss**<br>**HellaSwag**<br>**ARC-E**<br>**ARC-C**<br>**MMLU**<br>**(**_↓_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**|
|---|---|
|Dense Baseline<br>270M<br>–<br>20000<br>–<br>–<br>42B<br>34<br>Dense TST<br>270M<br>6000<br>20000<br>6x<br>75B<br>105B<br>34|3.212<br>36.3<br>46.7<br>24.9<br>–<br>**3.142**<br>**38.6**<br>**47.6**<br>**26.4**<br>–|
|Dense Baseline<br>270M<br>–<br>100000<br>–<br>–<br>209B<br>170<br>Dense TST<br>270M<br>30000<br>100000<br>6x<br>377B<br>524B<br>170|3.092<br>40.2<br>47.5<br>**26.2**<br>–<br>**3.048**<br>**42.6**<br>**50.3**<br>25.5<br>–|
|Dense Baseline<br>600M<br>–<br>20000<br>–<br>–<br>42B<br>61<br>Dense TST<br>600M<br>6000<br>20000<br>6x<br>75B<br>105B<br>61|3.019<br>43.5<br>51.7<br>25.5<br>–<br>**2.943**<br>**48.2**<br>**52.5**<br>**26.9**<br>–|
|Dense Baseline<br>3B<br>–<br>20000<br>–<br>–<br>42B<br>**247**<br>Dense Baseline<br>3B<br>–<br>36000<br>–<br>–<br>75B<br>443<br>Dense Baseline<br>3B<br>–<br>50000<br>–<br>–<br>105B<br>622<br>Dense TST<br>3B<br>6000<br>20000<br>6x<br>75B<br>105B<br>**247**|2.808<br>57.6<br>60.6<br>31.9<br>31.2<br>2.677<br>62.3<br>65.9<br>34.9<br>32.7<br>**2.640**<br>**63.9**<br>**67.3**<br>**36.8**<br>**33.3**<br>2.676<br>62.4<br>66.3<br>36.0<br>32.8|
|MoE Baseline<br>10B A1B<br>–<br>125000<br>–<br>–<br>1.05T<br>12311<br>2.252<br>70.1<br>73.8<br>46.3<br>37.4<br>MoE TST<br>10B A1B<br>12483<br>49983<br>16x<br>1.68T<br>2T<br>**4768**<br>**2.236**<br>**71.2**<br>**74.2**<br>**47.3**<br>**39.0**||



Table 1: Overview of TST’s advantage across different configurations compared to standard training (baseline). The TST Tokens denote raw data token count before compression. All evals are 0-shot. More results are shown in Table 3, Appendix E. 

**==> picture [392 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
270M Parameters, 42B Equivalent-Tokens 270M Parameters, 210B Equivalent-Tokens 600M Parameters, 42B Equivalent-Tokens<br>r = 0<br>3 . 02 r = 0 . 1<br>3 . 21 3 . 09 r = 0 . 2<br>3 . 2 3 . 01 rrr = 0= 0= 0 ... 345<br>3 . 08 3 r = 0 . 6<br>3 . 19<br>2 . 99<br>3 . 18 3 . 07 2 . 98<br>3 . 17<br>2 . 97<br>3 . 16 3 . 06<br>2 . 96<br>3 . 15<br>3 . 05 2 . 95<br>3 . 14<br>2 . 94<br>1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12<br>Superposition Size Superposition Size Superposition Size<br>Loss<br>Final<br>**----- End of picture text -----**<br>


Figure 4: Superposition results with respect to loss at varying superposition bag sizes and superposition step ratio _r_ , where _r_ is the ratio of number of steps trained in the superposition regime, with 1 _− r_ trained in the normal regime. Each data point is a fully trained model using TST first and converted to a standard AR LM. The full set of results are attached in Appendix E. 

For the large 10B Mixture-of-Experts scaling validation run, we broadly follow the Qwen3 [57] training procedures, and use the Qwen3 architecture, but scaled down to match the size of a 10B total, 1B active parameter model trained to 1.05T tokens. A 50/50 mix of FineWeb-Edu [39] and DCLM was used for this larger training run, along with a constant batch size of 8M tokens per step. The TST variant was trained with the settings shown in Table 1, for a total of 2T data tokens. 

For the optimizer, we use AdamW [38], with the optimal learning rate for the 270M and 600M models found using a sweep shown in Figure 7, Appendix B. For the 3B and 10B models, we use the recommended learning rate of 2 _×_ 10 _[−]_[4] and 3 _×_ 10 _[−]_[4] respectively, as a full sweep is too expensive at this scale. For all training runs, we use the standard _β_ 1 = 0 _._ 9 and _β_ 2 = 0 _._ 95, along with the Warmup-Stable-Decay [24, 56] learning rate scheduler. We warm up for a constant 2000 steps and then decay for the last 10% steps. 

We run the standard set of LLM evaluations at the final step checkpoint after the recovery phase, including ARC [11], BoolQ [10], HellaSwag [60], MMLU [22], OpenBookQA [42], PIQA [7] and Winogrande [47]. All evaluations are performed using the Eleuther AI LM-Eval harness [17], using a 0-shot prompting setting. These experiments are illustrated in Table 1 and Figure 4 highlights the robustness and significant benefit of using TST, as it outperforms the baseline in all equal-FLOPs or equal-loss settings tested. 

## **5 Discussion** 

## **5.1 On Comparisons to Auxiliary-loss Methods** 

One may wonder about how TST compares empirically to multi-token prediction (MTP) [20] and related auxiliary-loss methods [34, 37, 41, 64]. We argue that such comparisons are not directly 

7 

**==> picture [392 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
270M Parameters, 42B Equivalent-Tokens 270M Parameters, 210B Equivalent-Tokens 600M Parameters, 42B Equivalent-Tokens<br>0 . 465 r = 0<br>0 . 48 0 . 505 rr = 0= 0 .. 12<br>0 . 46 r = 0 . 3<br>0 . 475 0 . 5 rr = 0= 0 .. 45<br>r = 0 . 6<br>0 . 455 0 . 495<br>0 . 47<br>0 . 45 0 . 49<br>0 . 465<br>0 . 485<br>0 . 445<br>0 . 46 0 . 48<br>0 . 44<br>0 . 455 0 . 475<br>0 . 435 0 . 47<br>0 . 45<br>1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12<br>Superposition Size Superposition Size Superposition Size<br>Accuracy<br>Average<br>**----- End of picture text -----**<br>


Figure 5: Downstream evals at varying superposition bag sizes and superposition step ratio _r_ , where the average is the arithmetic average of (arc-c, arc-e, boolq, hellaswag, openbookqa, piqa, winogrande). The full set of results are attached in Appendix E. 

meaningful for our setting. MTP and its variants do not increase training-time throughput: they process the same number of tokens per FLOP as the baseline, while adding parameters and an auxiliary loss term. Additionally, in the case of MTP and some of the related methods, emphasis is placed on inference-time gains via speculative decoding, an aspect that is not addressed in this work. TST occupies a different point in the design space: it strictly increases tokens-per-FLOP during training, leaves the inference-time architecture untouched, and we observe consistent gains from 270M to 10B parameters. We therefore view TST as orthogonal to, rather than competing with, auxiliary-loss methods, and combining the two is a natural direction for future work. 

## **5.2 Input and Output Superposition** 

We perform ablations adding the input-only, output-only and full superposition settings with a superposition bag size _s_ = 4 for a ratio _r_ = 0 _._ 5 of the total steps. The input-only method only bags the input token embeddings but only predicts one next token, and the output-only method processes individual tokens in the input but predicts the next bag of tokens in the output. 

Figure 6 illustrates that all superposition settings outperform the baseline, but emphasizes that input or output superposition alone does not capture the complete improvement of full superposition, and the combination of both yields a further gain without signs of interference. We understand this as evidence that TST is not a single trick but two orthogonal mechanisms: the input superposition changes the input granularity and FLOPs cost per unit of information, while the output side modifies the prediction target and the gradients. 

**Output Superposition** Next bag-of-tokens prediction is using bag-of-words summaries of the future, reminiscent of seminal works in word vector representation [43], extractive summarization [40] and information retrieval [51], repurposed as a local supervision signal for an autoregressive model. The closest work in the literature is _future summary prediction_ [41], which also targets a compressed view of the future. The important difference is architectural rather than conceptual: the authors attach an auxiliary head with a binary cross-entropy loss on top of the regular next-token objective, paying extra parameters and an extra loss term. We keep a single cross-entropy loss on the single main head and only replace the target. An even closer match, to our knowledge not in the peer-reviewed literature, is a `modded-nanogpt` speedrun entry [27] that concurrently proposed a next-bag-of-tokens loss. It differs from ours in two design choices: exponential weighting of the bag, and a smooth interpolation into next-token prediction rather than a hard switch. 

**Output Bag Weighting** Our experiments show U-shaped loss plots when varying the superposition bag size, highlighting the need to correctly tune this hyperparameter to find the optimal setting, with the risk of slightly suboptimal results if overshooting. We tried different weighting schemes to average the loss participation of each term in the bag-of-tokens, detailed in Appendix D. The power-law weighting scheme was the most promising at large superposition sizes, so we performed a new set of experiments and compared it to the uniform average and reported the results in Figure 8, 

8 

Appendix D. The power-law weighted average results in higher loss than the uniform average for smaller superposition sizes, but outperforms it and is more stable for _s ≥_ 8. 

**==> picture [199 x 140] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 275<br>3 . 250<br>3 . 225<br>3 . 200<br>3 . 175<br>3 . 150<br>3 . 125<br>3 . 100<br>3 . 075 Baseline<br>3 . 050 Input Superposition (Bag-of-Token-Embeddings)<br>3 . 025 Output Superposition (Next Bag-of-Tokens Prediction)<br>Full Superposition<br>3 . 000<br>Training Steps<br>20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000<br>Training Loss<br>**----- End of picture text -----**<br>


Figure 6: Input and Output Superposition ablations, only the recovery phase (ii) is represented. 

**Input Superposition** Input superposition has, as far as we know, little direct analog in the LLM pretraining literature. The underlying reason for its success is an open question. One interpretation we find plausible is that the first phase acts as a form of pre-pre-training as studied by Hu et al. [23] and Lee et al. [30]: before learning full-resolution language, the model is exposed to a simpler distribution that already shares coarse statistical structure with natural language ( _e.g._ local topic, co-occurrence), and carries that inductive prior into phase (ii). 

A second, not exclusive, interpretation is that averaging in embedding space implicitly regularizes the embedding geometry, since many random _s_ -grams must remain linearly separable once summed. We do not have the interpretability evidence to pick between these. 

Beyond language models, training first on a lower-resolution, higher-throughput version of the data and then on the full-resolution version is an old idea that recurs in recent work: Anagnostidis et al. [4] schedule patch size from coarse to fine for Vision Transformers, Minixhofer et al. [44] start from a pretrained subword-LLM that is then converted into a model capable of processing finer-grained UTF-8 bytes. 

Input superposition can be read as the same principle applied to token embeddings. We see this as evidence that the coarse-to-fine granularity schedule is a reusable ingredient of efficient pretraining recipes, rather than a property of any specific modality. 

## **5.3 Two-Phase Task Alignment** 

The natural question arises: why has this effect not been observed in numerous prior work that resembles TST? Much of the existing literature on multi-stage pre-training and transfer learning relies on an **alignment phase** , in which the main model is frozen and only a small adapter is trained before unfreezing everything the recovery phase. We argue that this design choice is precisely what has obscured the effect we exploit in TST. 

We hypothesize that the internal circuitry of a LLM is highly sensitive to its input and output representations. TST is, to our knowledge, one of the few compressive LLM pre-training method in which the input embedding and output LM head are shared without modification across the superposition and recovery phases, avoiding the representational mismatch that previous methods incur (and that adapter-based alignment is designed to patch over). 

To test this hypothesis, we run a Dense TST 3B experiment matching the setup in Table 1, but we randomly re-initialize the input embedding and output LM head at the start of the recovery phase. As shown in Table 2, perturbing the input/output representations between the two phases completely eliminates the gains of TST, and makes it even worse than the baseline training where the TST steps are completely wasted and do not contribute to the final model. Although not definitive, this result supports our hypothesis that representation alignment across the two phases is a key reason why TST succeeds where prior compressive approaches have required explicit alignment training. 

9 

|**Model**<br>**Params**<br>**TST**<br>**Total**<br>**TST**<br>**TST**<br>**Total**<br>**B200-Hours**<br>**Steps**<br>**Steps**<br>**Bag Size**<br>**Tokens**<br>**Tokens**<br>**(**_↓_**)**|**Final Loss**<br>**(**_↓_**)**|
|---|---|
|Dense Baseline<br>3B<br>–<br>20000<br>–<br>–<br>42B<br>247<br>Dense TST<br>3B<br>6000<br>20000<br>6x<br>75B<br>105B<br>247<br>Dense TST w/ Randomization<br>3B<br>6000<br>20000<br>6x<br>75B<br>105B<br>247|2.808<br>**2.676**<br>2.938|



Table 2: Comparison between TST and TST with randomization where we re-initialize the input embedding and LM head layers between the superposition phase and the recovery phase. 

## **6 Conclusion** 

In this work, we describe a high throughput training paradigm for LLMs: Token Superposition Training. During the token superposition regime, sample throughput is increased _s_ -fold without changing per-step FLOPs, parallelism, model architecture, tokenizer, or data. The following recovery phase is a return to the standard LLM pretraining regime, exhibiting a fast recovery period, quickly outperforming the loss of an equal-FLOPs baseline pretraining. TST significantly increases the pretraining efficiency compared to baseline pretraining at the same computational cost ( _c.f._ Figure 3a). Alternatively, the same loss can be achieved at around half the computational cost ( _c.f._ Figure 3b). Overall, we find this new paradigm to be robust to hyperparameter choice within a reasonable range (superposition bag size _s ∈_ [[4 _,_ 8]] and step ratio _r ∈_ [0 _._ 2 _,_ 0 _._ 4]). 

## **7 Limitations, Future Work and Broader Impacts** 

TST effectively trades more data consumption for better loss at a given computational cost. The underlying assumption is that LLM pretraining is performed under compute-bound constraints rather than data-bound constraints. Kim et al. [28] recently argued that this assumption will be wrong in the future, given current trends. In this alternative view, output-only superposition offers a significant advantage, as it outperforms the baseline pretraining regime without increasing data consumption. We leave the study of these settings and the comparison with auxiliary loss methods, such as MTP, for future work. 

Folding the initial sequence into a sequence of bags of _s_ tokens results in a longer effective context during TST compared to the baseline regime. This could likely have positive effects on long context performance that we did not evaluate, as there would be less truncation or splitting of native long context data, leaving this to future work. 

We also did not perform larger scale ablations or multiple identical runs to evaluate statistical significance due to limited compute resources. Future work could investigate scaling laws of token superposition, in order to predict the best TST settings for larger model sizes, including industryscale pretraining. 

We proposed some hypotheses on the phenomena involved in TST, but further interpretability work on this subject could definitely improve the understanding of the underlying mechanisms and the ramifications of token superposition. 

**Broader Impacts:** Our work improves the efficiency of large language model pre-training, reducing computational cost and energy usage, which may improve accessibility to a broader range of researchers. However, increased efficiency may also accelerate the development and deployment of such models, potentially amplifying known risks including misuse for generating harmful content and concerns around bias, fairness, and privacy. While our contribution is methodological and does not directly introduce new capabilities, it may indirectly increase the scale at which these systems are trained and used. We encourage future work to pair efficiency improvements with advances in safety and responsible deployment. 

## **References** 

- [1] S. Agarwal, L. Ahmad, J. Ai, S. Altman, A. Applebaum, E. Arbus, R. K. Arora, Y. Bai, B. Baker, H. Bao, and others. gpt-oss-120b & gpt-oss-20b model card. 

10 

- [2] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millicah, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan. Flamingo: a visual language model for few-shot learning. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , NIPS ’22, pages 23716–23736, Red Hook, NY, USA, Nov. 2022. Curran Associates Inc. ISBN 978-1-7138-7108-8. 

- [3] L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Bl´azquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydl´ıˇcek, A. P. Lajar´ın, V. Srivastav, J. Lochner, C. Fahlgren, X.-S. Nguyen, C. Fourrier, B. Burtenshaw, H. Larcher, H. Zhao, C. Zakka, M. Morlon, C. Raffel, L. v. Werra, and T. Wolf. SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model, Feb. 2025. URL `http://arxiv.org/abs/2502.02737` . arXiv:2502.02737 [cs]. 

- [4] S. Anagnostidis, G. Bachmann, I. Schlag, and T. Hofmann. Navigating scaling laws: compute optimality in adaptive model training. In _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _ICML’24_ , pages 1511–1530, Vienna, Austria, 2024. JMLR.org. 

- [5] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, and others. Qwen technical report. 

- [6] X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, and others. Deepseek llm: Scaling open-source language models with longtermism. 

- [7] Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pages 7432–7439, 2020. 

- [8] M. Chen, B. Hui, Z. Cui, J. Yang, D. Liu, J. Sun, J. Lin, and Z. Liu. Parallel Scaling Law for Language Models. Oct. 2025. URL `https://openreview.net/forum?id=dEi1S731lk` . 

- [9] X. Cheng, W. Zeng, D. Dai, Q. Chen, B. Wang, Z. Xie, K. Huang, X. Yu, Z. Hao, Y. Li, and others. Conditional memory via scalable lookup: A new axis of sparsity for large language models. 

- [10] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In _Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers)_ , pages 2924–2936, 2019. 

- [11] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- [12] C. Cortes and V. Vapnik. Support-vector networks. _Machine learning_ , 20(3):273–297, 1995. 

- [13] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. In L.-W. Ku, A. Martins, and V. Srikumar, editors, _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1280–1297, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.70. URL `https://aclanthology.org/2024.acl-long.70/` . 

- [14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Oct. 2020. URL `https://openreview.net/forum?id=YicbFdNTTy&utm_ campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=` 

   - `p2ANqtz--bIpWoAA0d8Ugha6WmwlzJEFeLwluYNZSx-7AAH9r5Kdq3UTcUJwY1X4RnbL0IOgx_ 32-d` . 

11 

- [15] W. Ebeling and T. P¨oschel. Entropy and Long-Range Correlations in Literary English. _Europhysics Letters_ , 26(4):241, May 1994. ISSN 0295-5075. doi: 10.1209/0295-5075/26/4/001. URL `https://doi.org/10.1209/0295-5075/26/4/001` . 

- [16] S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat, A. Fang, J. Li, S. Keh, and others. Language models scale reliably with over-training and on downstream tasks. 

- [17] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. A framework for few-shot language model evaluation, 12 2023. URL `https: //zenodo.org/records/10256836` . 

- [18] T. Gigant, B. Peng, and J. Quesnelle. Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation, Apr. 2026. URL `http://arxiv.org/ abs/2604.27263` . arXiv:2604.27263 [cs]. 

- [19] H. Gisserot-Boukhlef, N. Boizard, M. Faysse, D. M. Alves, E. Malherbe, A. Martins, C. Hudelot, and P. Colombo. Should We Still Pretrain Encoders with Masked Language Modeling? Oct. 2025. URL `https://openreview.net/forum?id=jpz7e3jhRq` . 

- [20] F. Gloeckle, B. Y. Idrissi, B. Rozi`ere, D. Lopez-Paz, and G. Synnaeve. Better & faster large language models via multi-token prediction. In _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _ICML’24_ , pages 15706–15734, Vienna, Austria, 2024. JMLR.org. 

- [21] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, and others. The llama 3 herd of models. 

- [22] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- [23] M. Y. Hu, J. Petty, C. Shi, W. Merrill, and T. Linzen. Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 9691–9709, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10. 18653/v1/2025.acl-long.478. URL `https://aclanthology.org/2025.acl-long.478/` . 

- [24] S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, X. Zhang, Z. L. Thai, K. Zhang, C. Wang, Y. Yao, C. Zhao, J. Zhou, J. Cai, Z. Zhai, N. Ding, C. Jia, G. Zeng, D. Li, Z. Liu, and M. Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024. URL `https://arxiv.org/abs/2404.06395` . 

- [25] S. Hwang, B. Wang, and A. Gu. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, July 2025. URL `http://arxiv.org/abs/2507.07955` . arXiv:2507.07955 [cs]. 

- [26] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mixtral of Experts, Jan. 2024. URL `http://arxiv.org/abs/2401.04088` . arXiv:2401.04088 [cs]. 

- [27] KellerJordan. New Record: Multi-token prediction and Untie LM Head 2/3rds through training (119.76 seconds) by varunneal · Pull Request #178 · KellerJordan/modded-nanogpt. URL `https://github.com/KellerJordan/modded-nanogpt/pull/178` . 

- [28] K. Kim, S. Kotha, P. Liang, and T. Hashimoto. Pre-training under infinite compute, Sept. 2025. URL `http://arxiv.org/abs/2509.14786` . arXiv:2509.14786 [cs]. 

- [29] T. Kudo and J. Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In _Proceedings of the 2018 conference on empirical methods in natural language processing: System demonstrations_ , pages 66–71. 

12 

- [30] D. Lee, S. Han, A. Kumar, and P. Agrawal. Training Language Models via Neural Cellular Automata, 2026. URL `https://arxiv.org/abs/2603.10055` . Version Number: 1. 

- [31] Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative decoding. In _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _ICML’23_ , pages 19274–19286, Honolulu, Hawaii, USA, 2023. JMLR.org. 

- [32] J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Gadre, H. Bansal, E. Guha, S. Keh, K. Arora, S. Garg, R. Xin, N. Muennighoff, R. Heckel, J. Mercat, M. Chen, S. Gururangan, M. Wortsman, A. Albalak, Y. Bitton, M. Nezhurina, A. Abbas, C.-Y. Hsieh, D. Ghosh, J. Gardner, M. Kilian, H. Zhang, R. Shao, S. Pratt, S. Sanyal, G. Ilharco, G. Daras, K. Marathe, A. Gokaslan, J. Zhang, K. Chandu, T. Nguyen, I. Vasiljevic, S. Kakade, S. Song, S. Sanghavi, F. Faghri, S. Oh, L. Zettlemoyer, K. Lo, A. El-Nouby, H. Pouransari, A. Toshev, S. Wang, D. Groeneveld, L. Soldaini, P. W. Koh, J. Jitsev, T. Kollar, A. G. Dimakis, Y. Carmon, A. Dave, L. Schmidt, and V. Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024. 

- [33] W. Liang, T. Liu, L. Wright, W. Constable, A. Gu, C.-C. Huang, I. Zhang, W. Feng, H. Huang, J. Wang, S. Purandare, G. Nadathur, and S. Idreos. TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining. Oct. 2024. URL `https://openreview. net/forum?id=SFN6Wm7YBI` . 

- [34] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, and others. Deepseek-v3 technical report. . 

- [35] A. Liu, J. Hayase, V. Hofmann, S. Oh, N. A. Smith, and Y. Choi. SuperBPE: Space travel for language models. . 

- [36] H. Liu, J. Zhang, C. Wang, X. Hu, L. Lyu, J. Sun, X. Yang, B. Wang, F. Li, Y. Qian, and others. Scaling embeddings outperforms scaling experts in language models. . 

- [37] Y. Liu, Y. Song, Y. Wang, K. Ge, A. Lamb, Q. Guo, K. Chen, B. Zhou, and Z. Lin. Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models, Feb. 2026. URL `http://arxiv.org/abs/2602.08984` . arXiv:2602.08984 [cs]. 

- [38] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- [39] A. Lozhkov, L. Ben Allal, L. von Werra, and T. Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL `https://huggingface.co/datasets/HuggingFaceFW/ fineweb-edu` . 

- [40] H. P. Luhn. The Automatic Creation of Literature Abstracts. _IBM Journal of Research and Development_ , 2(2):159–165, Apr. 1958. ISSN 0018-8646. doi: 10.1147/rd.22.0159. URL `https://ieeexplore.ieee.org/document/5392672` . Conference Name: IBM Journal of Research and Development. 

- [41] D. Mahajan, S. Goyal, B. Y. Idrissi, M. Pezeshki, I. Mitliagkas, D. Lopez-Paz, and K. Ahuja. Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries, Oct. 2025. URL `http://arxiv.org/abs/2510.14751` . arXiv:2510.14751 [cs]. 

- [42] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _Proceedings of the 2018 conference on empirical methods in natural language processing_ , pages 2381–2391, 2018. 

- [43] T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient Estimation of Word Representations in Vector Space. Jan. 2013. URL `https://www.semanticscholar.org/ paper/Efficient-Estimation-of-Word-Representations-in-Mikolov-Chen/ f6b51c8753a871dc94ff32152c00c01e94f90f09` . 

- [44] B. Minixhofer, T. Murray, T. Limisiewicz, A. Korhonen, L. Zettlemoyer, N. A. Smith, E. M. Ponti, L. Soldaini, and V. Hofmann. Bolmo: Byteifying the Next Generation of Language Models, Dec. 2025. URL `http://arxiv.org/abs/2512.15586` . arXiv:2512.15586 [cs]. 

13 

- [45] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, and C. Li. Large Language Diffusion Models. Oct. 2025. URL `https://openreview.net/forum? id=KnqiC0znVF` . 

- [46] A. Pagnoni, R. Pasunuru, P. Rodriguez, J. Nguyen, B. Muller, M. Li, C. Zhou, L. Yu, J. E. Weston, L. Zettlemoyer, G. Ghosh, M. Lewis, A. Holtzman, and S. Iyer. Byte Latent Transformer: Patches Scale Better Than Tokens. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 9238–9258, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.453. URL `https://aclanthology.org/2025.acl-long.453/` . 

- [47] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_ , 64(9):99–106, 2021. 

- [48] R. Sennrich, B. Haddow, and A. Birch. Neural machine translation of rare words with subword units. In _Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: long papers)_ , pages 1715–1725. 

- [49] C. E. Shannon. A mathematical theory of communication. _The Bell system technical journal_ , 27(3):379–423, 1948. 

- [50] C. Shao, F. Meng, and J. Zhou. Beyond next token prediction: Patch-level training for large language models. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL `https://openreview.net/forum?id=dDpB23VbVa` . 

- [51] K. Sparck Jones. A statistical interpretation of term specificity and its application in retrieval. _Journal of documentation_ , 28(1):11–21, 1972. 

- [52] Y. Tay, M. Dehghani, V. Q. Tran, X. Garcia, J. Wei, X. Wang, H. W. Chung, D. Bahri, T. Schuster, S. Zheng, D. Zhou, N. Houlsby, and D. Metzler. UL2: Unifying Language Learning Paradigms. Sept. 2022. URL `https://openreview.net/forum?id=6ruVLB727MC` . 

- [53] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, and others. Llama: Open and efficient foundation language models. 

- [54] M. Videau, B. Y. Idrissi, A. Leite, M. Schoenauer, O. Teytaud, and D. Lopez-Paz. From Bytes to Ideas: Language Modeling with Autoregressive U-Nets. Oct. 2025. URL `https: //openreview.net/forum?id=FnFf7Ru2ur` . 

- [55] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. Oct. 2022. URL `https://openreview.net/forum?id=_VjQlMeSB_J&trk=public_ post_comment-text` . 

- [56] K. Wen, Z. Li, J. Wang, D. Hall, P. Liang, and T. Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective, 2024. URL `https://arxiv.org/ abs/2410.05192` . 

- [57] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, and others. Qwen3 technical report. 

- [58] J. Yuan, H. Gao, D. Dai, J. Luo, L. Zhao, Z. Zhang, Z. Xie, Y. Wei, L. Wang, Z. Xiao, Y. Wang, C. Ruan, M. Zhang, W. Liang, and W. Zeng. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 23078–23097, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1126. URL `https://aclanthology.org/2025.acl-long.1126/` . 

14 

- [59] M. Zaheer, G. Guruganesh, A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed. Big bird: transformers for longer sequences. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , NIPS ’20, pages 17283–17297, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 978-1-71382954-6. URL `https://dl.acm.org/doi/10.5555/3495724.3497174` . 

- [60] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th annual meeting of the association for computational linguistics_ , pages 4791–4800, 2019. 

- [61] Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, A. Desmaison, C. Balioglu, P. Damania, B. Nguyen, G. Chauhan, Y. Hao, A. Mathews, and S. Li. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. _Proc. VLDB Endow._ , 16(12):3848–3860, 2023. ISSN 2150-8097. doi: 10.14778/3611540.3611569. URL `https://dl.acm.org/doi/10.14778/3611540.3611569` . 

- [62] L. Zheng, X. Li, Q. Liu, X. Feng, and L. Kong. Proxy Compression for Language Modeling, Feb. 2026. URL `http://arxiv.org/abs/2602.04289` . arXiv:2602.04289 [cs]. 

- [63] R.-J. Zhu, Z. Wang, K. Hua, T. Zhang, Z. Li, H. Que, B. Wei, Z. Wen, F. Yin, H. Xing, and others. Scaling latent reasoning via looped language models. 

- [64] Z. M. K. Zuhri, E. H. Fuadi, and A. F. Aji. Predicting the Order of Upcoming Tokens Improves Language Modeling, Feb. 2026. URL `http://arxiv.org/abs/2508.19228` . arXiv:2508.19228 [cs]. 

15 

## **A Code** 

1 2 `[...] # within train loop` 3 4 `if superposition_bag_size is not None and superposition_bag_size > 1:` 5 `bs , seq = inputs.shape` 6 `inputs = inputs.reshape(bs , seq // superposition_bag_size , superposition_bag_size )` 

Listing 1: Input folding in Pytorch 

1 2 `[...] # within model forward` 3 4 `# Using superposition` 5 `if len (tokens.shape) == 3:` 6 `bs , sp_seq , superposition_bag_size = tokens.shape` 7 8 `# Sum in float32 for better numerical precision` 9 `h = self. tok_embeddings (tokens [... , 0])` 10 `h_dtype = h.dtype` 11 `h = h. float ()` 12 `for i in range (1, superposition_bag_size ):` 13 `h = h + self. tok_embeddings (tokens [... , i]). float ()` 14 15 `h = (h / superposition_bag_size ).to(h_dtype)` 16 17 `else :` 18 `h = self. tok_embeddings (tokens)` 

Listing 2: Bag-of-Token embeddings input in Pytorch 

1 `import torch` 2 3 `def cross_entropy_loss (pred: torch.Tensor , labels: torch.Tensor) -> torch. Tensor:` 4 5 `# Compute shapes` 6 `bs , seq , dim = pred.shape` 7 `label_bs , label_seq = labels.shape` 8 `superposition_bag_size = label_seq // seq` 9 `superposition_offset = superposition_bag_size - 1` 10 11 `# Pre -flatten and perform causal padding` 12 `pred = pred.flatten (0, 1). float ()` 13 `labels = torch.nn.functional .pad(labels , (0, superposition_offset ), mode= ’constant ’ , value = -100) [... , superposition_offset :]. view ((bs , seq , superposition_bag_size ))` 14 15 `# Compute loss` 16 `loss = 0.` 17 `w_total = 0.` 18 `for i in range ( superposition_bag_size ):` 19 `w = 1 # uniform weighting` 20 `target = labels [... , i]. flatten (0, 1)` 21 `loss += w * torch.nn.functional . cross_entropy (pred , target)` 22 `w_total += w` 23 24 `return loss / w_total` 

Listing 3: Next bag-of-words prediction loss code in Pytorch 

16 

## **B Learning rate sweeps** 

**==> picture [391 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
270M Parameters, 42B Tokens 270M Parameters, 210B Tokens 600M Parameters, 42B Tokens<br>3 . 1 3 . 03<br>3 . 23<br>3 . 1 3 . 03<br>3 . 22 3 . 03<br>3 . 1<br>3 . 03<br>3 . 22 3 . 1 3 . 03<br>3 . 09 3 . 02<br>3 . 21<br>3 . 02<br>3 . 09<br>3 . 02<br>1 2 3 4 5 6 1 . 2 1 . 4 1 . 6 1 . 8 2 2 . 2 2 . 4 2 . 6 2 . 8 3 1 2 3 4 5<br>Learning Rate · 10 [−] [3] Learning Rate · 10 [−] [3] Learning Rate · 10 [−] [3]<br>Loss<br>Final<br>**----- End of picture text -----**<br>


Figure 7: Learning rate sweeps at varying model sizes, with the optimal learning rate being used for all the training runs. The final lr used in order from left to right are 2 _×_ 10 _[−]_[3] , 2 _×_ 10 _[−]_[3] and 1 _×_ 10 _[−]_[3] . 

## **C Loss Derivations** 

**Setup.** Given logits **z** _∈_ R _[V]_ , let _Z_ =[�] _[V] i_ =1[exp(] _[z][i]_[)][ and write the softmax probability as] _[ P]_[(] _[i]_[)][=] exp( _zi_ ) _/Z_ . For a target distribution **t** over the vocabulary, cross-entropy and KL divergence are 

**==> picture [272 x 22] intentionally omitted <==**

where **t** is the target probability vector, and where _H_ ( **t** ) = _−_[�] _i[t][i]_[ log] _[ t][i]_[. Cross-entropy vanishes at] its minimum only when _H_ ( **t** ) = 0, i.e. when **t** is one-hot. KL divergence always vanishes at _P_ = **t** regardless of the target. 

We now consider a bag **y** of _s_ valid target tokens, with two variants of a multi-hot cross-entropy (MCE) loss for a bag of _s_ valid target tokens **y** . 

## **C.1 MCE: equal-probability targets** 

The MCE loss used throughout the paper treats the bag as a multi-hot target and pushes each valid token’s probability towards 1 _/s_ , so that the bag tokens try to end up with an equal probability. Therefore, the target is the uniform distribution over the bag: 

**==> picture [93 x 25] intentionally omitted <==**

Standard cross-entropy with this new target is 

**==> picture [240 x 28] intentionally omitted <==**

Unlike the one-hot case, this target has nonzero entropy _H_ ( **t** ) = log _s_ , so plain CE bottoms out at log _s_ rather than 0. To recover the same ”vanishes at the optimum” behaviour as standard CE, we 

17 

subtract the entropy of the target (i.e. we use KL divergence): _L_ MCE( **z** _,_ **y** ) = KL( **t** _∥ P_ ) 

**==> picture [268 x 168] intentionally omitted <==**

Rearranging the terms, we get: 

**==> picture [318 x 98] intentionally omitted <==**

## **C.2 MCEAlt: sum-to-one probability targets** 

We also investigated a different formulation of the MCE loss we used, where we target the sum of the probabilities of all valid labels to be 1 instead of targeting them to be equal-probability. Here we do not pick a single target distribution; instead we only require that the total probability mass on the bag be 1. This effectively lets the model choose its own weighting across bag tokens. A natural way to express this is to treat the bag as a single composite label with probability 

**==> picture [152 x 29] intentionally omitted <==**

Cross-entropy against a one-hot target on this composite label is 

_L_ MCEAlt( **z** _,_ **y** ) = _−_ log _P_ ( **y** ) 

**==> picture [250 x 63] intentionally omitted <==**

Because the implicit target is a one-hot over composite labels (all mass in ”the bag”), its entropy is zero and no correction is needed: the loss vanishes exactly when[�] _y∈_ **y** _[P]_[(] _[y]_[) = 1][, just like ordinary] CE on a single label. 

In all the limited small-scale experiments we have done with MCEAlt, it seemed to perform identically[2] compared to the equal-probability loss of MCE. We therefore did not explore it further, since unlike MCE it does not reduce to a simplified form that uses CE, it would require a custom loss function, reduce the training speed, increase memory usage, and add unnecessary complexity to the final method. We leave a more thorough exploration of this variant for future work. 

> 2We trained models using one variant of MCE or the other, and we obtained the same final loss after the recovery phase if everything else is kept identical. 

18 

## **D Non-uniform Multi-hot Cross-Entropy** 

**==> picture [273 x 122] intentionally omitted <==**

**----- Start of picture text -----**<br>
Uniform Loss Power Law Loss<br>r = 0<br>3 . 02 3 . 02 r = 0 . 1<br>r = 0 . 2<br>3 . 01 3 . 01 rr = 0= 0 .. 34<br>r = 0 . 5<br>3 3 r = 0 . 6<br>2 . 99 2 . 99<br>2 . 98 2 . 98<br>2 . 97 2 . 97<br>2 . 96 2 . 96<br>2 . 95 2 . 95<br>2 . 94 2 . 94<br>1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12<br>Superposition Size Superposition Size<br>Loss<br>Final<br>**----- End of picture text -----**<br>


Figure 8: Comparison between superposition using uniform output loss and power law output loss at varying superposition window sizes and superposition ratio _r_ , with a 600M-parameter model trained at 42B tokens. The final loss is the final training loss using standard next-token Cross-Entropy Loss. 

For other multi-hot target distributions, we consider functions _g_ that are monotonically decreasing with the position _i_ in the bag. 

**==> picture [292 x 27] intentionally omitted <==**

- Uniform: _i �→_ 1 

- Power law: _i �→_[1] _i_ 

- Exponential: _i �→_ exp( _−i_ ), following [27] 

- First token: _i �→ δ_ 1( _i_ ) 

**==> picture [360 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
3.3 3.3<br>Baseline Baseline<br>Uniform Uniform<br>Power law Power law<br>Exponential Exponential<br>First token First token<br>3.2 3.2<br>3.1 3.1<br>3.0 3.0<br>Training Steps Training Steps<br>(a) Superposition bag size  s  = 4 (b) Superposition bag size  s  = 16<br>20000 30000 40000 20000 30000 40000<br>Training Loss Training Loss<br>**----- End of picture text -----**<br>


Figure 9: Training loss after resuming from weighted superposition 

At a superposition bag size _s_ = 16, the best setting involves a power-law distribution. However, at _s_ = 4, the uniform distribution works better. We did not find a distribution that would work best for all superposition ratios. 

However, we believe that the power law distribution of losses is reminiscent of the prediction difficulty of future tokens based on current context. Ebeling and P¨oschel [15] investigated mutual information between pairs of letters in literary English texts, which they found to decay with distance following a power law. Inspired by this, we compute mutual information between pairs of tokens sampled from texts in the DCLM dataset and fit a power law to modelize its decay. This mutual information decay between pairs of tokens in the DCLM dataset, and a fitted power law, are illustrated in Figure 10. This result correlates with our observation for relative weighting of tokens for next bag-of-tokens prediction with large superposition bag sizes. Using the values of the fitted 

19 

power law to weight the losses per position results in a slightly better loss than the power law tested earlier. 

**==> picture [318 x 192] intentionally omitted <==**

**----- Start of picture text -----**<br>
5 × 10 [0] Power law (fitted)<br>4 × 10 [0]<br>10 [0] 10 [1] 10 [2]<br>Distance (tokens)<br>Mutual Information<br>**----- End of picture text -----**<br>


Figure 10: Mutual information between pairs of tokens in DCLM decays with distance following a power law: _d �→ C_ 0 + _a ∗ d[k]_ , with _C_ 0 _≈_ 3 _._ 63, _a ≈_ 1 _._ 35 and _k ≈−_ 1 _._ 25 

## **E Additional Results** 

Unless mentioned, all evals are run with 0-shot prompting. 

**==> picture [392 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
270M Parameters, 42B Equivalent-Tokens 270M Parameters, 210B Equivalent-Tokens 600M Parameters, 42B Equivalent-Tokens<br>0 . 395 0 . 435 00 . 485 . 49 rrr = 0= 0= 0 .. 12<br>0 . 39 0 . 43 0 . 48 rr = 0= 0 .. 34<br>0 . 475 r = 0 . 5<br>0 . 385 0 . 425 0 . 47 r = 0 . 6<br>0 . 38 0 . 42 0 . 465<br>0 . 46<br>0 . 375 0 . 415 0 . 455<br>0 . 45<br>0 . 37 0 . 41 0 . 445<br>0 . 365 0 . 405 0 . 44<br>0 . 435<br>0 . 4 0 . 43<br>1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12<br>Superposition Size Superposition Size Superposition Size<br>270M Parameters, 42B Equivalent-Tokens 270M Parameters, 210B Equivalent-Tokens 600M Parameters, 42B Equivalent-Tokens<br>0 . 57 r = 0<br>0 . 5 r = 0 . 1<br>0 . 53 r = 0 . 2<br>0 . 56 r = 0 . 3<br>r = 0 . 4<br>0 . 49 0 . 52 0 . 55 rr = 0= 0 .. 56<br>0 . 51<br>0 . 54<br>0 . 48<br>0 . 5<br>0 . 53<br>0 . 47 0 . 49 0 . 52<br>0 . 48<br>0 . 51<br>0 . 46<br>0 . 47<br>0 . 5<br>1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12<br>Superposition Size Superposition Size Superposition Size<br>Accuracy<br>Hellaswag<br>Accuracy<br>Arc-Easy<br>**----- End of picture text -----**<br>


Figure 11: Rows from top to bottom: Hellaswag and ARC-Easy downstream evals at varying superposition bag sizes and superposition step ratio _r_ . 

20 

|**Model**<br>**Params**<br>**TST**<br>**Total**<br>**B200-Hours**<br>**Steps**<br>**Steps**<br>**(**_↓_**)**|**Final Loss**<br>**HellaSwag**<br>**ARC-E**<br>**ARC-C**<br>**MMLU**<br>**BoolQ**<br>**OpenBookQA**<br>**PIQA**<br>**Winogrande**<br>**(**_↓_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**<br>**(**_↑_**)**|
|---|---|
|Dense Baseline<br>270M<br>–<br>20000<br>34<br>Dense TST<br>270M<br>6000<br>20000<br>34|3.212<br>36.3<br>46.7<br>24.9<br>–<br>**56.7**<br>29.2<br>66.4<br>**51.3**<br>**3.142**<br>**38.6**<br>**47.6**<br>**26.4**<br>–<br>54.0<br>**29.8**<br>**67.0**<br>51.1|
|Dense Baseline<br>270M<br>–<br>100000<br>170<br>Dense TST<br>270M<br>30000<br>100000<br>170|3.092<br>40.2<br>47.5<br>**26.2**<br>–<br>**58.5**<br>30.6<br>67.3<br>51.5<br>**3.048**<br>**42.6**<br>**50.3**<br>25.5<br>–<br>57.9<br>**32.4**<br>**69.0**<br>**54.2**|
|Dense Baseline<br>600M<br>–<br>20000<br>61<br>Dense TST<br>600M<br>6000<br>20000<br>61|3.019<br>43.5<br>51.7<br>25.5<br>–<br>**56.6**<br>31.2<br>69.0<br>52.6<br>**2.943**<br>**48.2**<br>**52.5**<br>**26.9**<br>–<br>54.8<br>**35.0**<br>**70.6**<br>**53.9**|
|Dense Baseline<br>3B<br>–<br>20000<br>**247**<br>Dense Baseline<br>3B<br>–<br>36000<br>443<br>Dense Baseline<br>3B<br>–<br>50000<br>622<br>Dense TST<br>3B<br>6000<br>20000<br>**247**|2.808<br>57.6<br>60.6<br>31.9<br>31.2<br>58.4<br>36.6<br>74.5<br>56.5<br>2.677<br>62.3<br>65.9<br>34.9<br>32.7<br>62.1<br>36.4<br>74.7<br>60.0<br>**2.640**<br>**63.9**<br>**67.3**<br>**36.8**<br>**33.3**<br>**64.6**<br>38.0<br>75.5<br>**60.9**<br>2.676<br>62.4<br>66.3<br>36.0<br>32.8<br>60.0<br>**42.0**<br>**76.1**<br>59.6|
|MoE Baseline<br>10B A1B<br>–<br>125000<br>12311<br>MoE TST<br>10B A1B<br>12483<br>49983<br>**4768**|2.252<br>70.1<br>73.8<br>46.3<br>37.4<br>66.2<br>**44.0**<br>77.2<br>61.3<br>**2.236**<br>**71.2**<br>**74.2**<br>**47.3**<br>**39.0**<br>**69.4**<br>43.2<br>**77.4**<br>**63.0**|



Table 3: Expanded results of Table 1. All evals are 0-shot. 

Table 4: Final loss values with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>1.0|3.2124<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>3.1847<br>3.1646<br>3.1549<br>3.1539<br>3.1496<br>3.1480<br>3.1495<br>3.1487<br>3.1498<br>3.1489<br>3.1495<br>3.1505<br>3.1513<br>3.1493<br>3.1487<br>–<br>3.1751<br>3.1555<br>3.1480<br>3.1461<br>3.1425<br>3.1393<br>3.1419<br>3.1417<br>3.1434<br>3.1439<br>3.1437<br>3.1450<br>3.1451<br>3.1431<br>3.1455<br>–<br>3.1723<br>3.1517<br>3.1457<br>3.1434<br>3.1423<br>3.1392<br>3.1425<br>3.1417<br>3.1438<br>3.1450<br>3.1454<br>3.1464<br>3.1475<br>3.1456<br>3.1484<br>–<br>3.1720<br>3.1510<br>3.1467<br>3.1434<br>3.1444<br>3.1409<br>3.1452<br>3.1454<br>3.1481<br>3.1489<br>3.1488<br>3.1504<br>3.1520<br>3.1508<br>3.1534<br>–<br>3.1729<br>3.1527<br>3.1496<br>3.1461<br>3.1485<br>3.1450<br>3.1506<br>3.1502<br>3.1544<br>3.1546<br>3.1550<br>3.1574<br>3.1597<br>3.1583<br>3.1611<br>–<br>3.1755<br>3.1563<br>3.1548<br>3.1521<br>3.1545<br>3.1516<br>3.1587<br>3.1590<br>3.1638<br>3.1644<br>3.1655<br>3.1675<br>3.1708<br>3.1699<br>3.1734<br>–<br>3.1807<br>3.1635<br>3.1630<br>3.1621<br>3.1664<br>3.1635<br>3.1720<br>3.1730<br>3.1794<br>3.1806<br>3.1823<br>3.1847<br>3.1893<br>3.1882<br>3.1922<br>–<br>3.1904<br>3.1768<br>3.1774<br>3.1808<br>3.1879<br>3.1862<br>3.1970<br>3.1998<br>3.2075<br>3.2105<br>3.2138<br>3.2176<br>3.2239<br>3.2226<br>3.2280<br>–<br>3.2131<br>3.2110<br>3.2221<br>3.2324<br>3.2487<br>3.2501<br>3.2681<br>3.2765<br>3.2894<br>3.2985<br>3.3073<br>3.3146<br>3.3275<br>3.3254<br>3.3381<br>–<br>4.6099<br>5.2482<br>5.6289<br>5.8658<br>6.0365<br>6.1567<br>6.2477<br>6.3146<br>6.3677<br>6.4152<br>6.4529<br>6.4835<br>6.5082<br>6.5314<br>6.5519|



Table 5: Final loss values with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>1.0|3.0921<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>3.0844<br>3.0711<br>3.0603<br>3.0553<br>3.0512<br>3.0488<br>3.0488<br>3.0508<br>3.0506<br>3.0510<br>3.0511<br>3.0525<br>–<br>3.0812<br>3.0662<br>3.0534<br>3.0472<br>3.0486<br>3.0465<br>3.0479<br>3.0501<br>3.0503<br>3.0502<br>3.0517<br>3.0524<br>–<br>3.0783<br>3.0652<br>3.0519<br>3.0459<br>3.0480<br>3.0463<br>3.0485<br>3.0510<br>3.0515<br>3.0514<br>3.0538<br>3.0551<br>–<br>3.0772<br>3.0639<br>3.0517<br>3.0474<br>3.0493<br>3.0475<br>3.0506<br>3.0527<br>3.0540<br>3.0535<br>3.0571<br>3.0588<br>–<br>3.0771<br>3.0652<br>3.0533<br>3.0496<br>3.0513<br>3.0497<br>3.0534<br>3.0556<br>3.0575<br>3.0570<br>3.0611<br>3.0627<br>–<br>3.0780<br>3.0668<br>3.0557<br>3.0525<br>3.0553<br>3.0535<br>3.0577<br>3.0604<br>3.0623<br>3.0620<br>3.0660<br>3.0681<br>–<br>3.0805<br>3.0703<br>3.0601<br>3.0574<br>3.0608<br>3.0597<br>3.0642<br>3.0673<br>3.0697<br>3.0694<br>3.0738<br>3.0762<br>–<br>3.0852<br>3.0770<br>3.0683<br>3.0670<br>3.0711<br>3.0706<br>3.0761<br>3.0800<br>3.0827<br>3.0830<br>3.0881<br>3.0909<br>–<br>3.0984<br>3.0963<br>3.0908<br>3.0934<br>3.1009<br>3.1020<br>3.1097<br>3.1161<br>3.1209<br>3.1236<br>3.1306<br>3.1348<br>–<br>4.5120<br>5.2018<br>5.5772<br>5.8255<br>5.9952<br>6.1179<br>6.2156<br>6.2891<br>6.3461<br>6.3890<br>6.4285<br>6.4628|



Table 6: Final loss values with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>1.0|3.0186<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>2.9876<br>2.9697<br>2.9597<br>2.9541<br>2.9522<br>2.9525<br>2.9499<br>2.9517<br>2.9507<br>2.9503<br>2.9498<br>2.9514<br>–<br>2.9782<br>2.9595<br>2.9508<br>2.9461<br>2.9440<br>2.9447<br>2.9434<br>2.9450<br>2.9450<br>2.9437<br>2.9457<br>2.9465<br>–<br>2.9757<br>2.9569<br>2.9492<br>2.9461<br>2.9439<br>2.9445<br>2.9439<br>2.9458<br>2.9464<br>2.9467<br>2.9488<br>2.9501<br>–<br>2.9757<br>2.9576<br>2.9507<br>2.9480<br>2.9465<br>2.9466<br>2.9477<br>2.9497<br>2.9508<br>2.9523<br>2.9549<br>2.9558<br>–<br>2.9770<br>2.9601<br>2.9547<br>2.9523<br>2.9517<br>2.9518<br>2.9544<br>2.9563<br>2.9583<br>2.9607<br>2.9636<br>2.9647<br>–<br>2.9800<br>2.9648<br>2.9605<br>2.9593<br>2.9598<br>2.9605<br>2.9645<br>2.9668<br>2.9695<br>2.9724<br>2.9764<br>2.9779<br>–<br>2.9857<br>2.9723<br>2.9702<br>2.9708<br>2.9726<br>2.9747<br>2.9801<br>2.9832<br>2.9871<br>2.9911<br>2.9962<br>2.9987<br>–<br>2.9946<br>2.9865<br>2.9874<br>2.9918<br>2.9955<br>3.0000<br>3.0079<br>3.0132<br>3.0181<br>3.0243<br>3.0310<br>3.0356<br>–<br>3.0150<br>3.0194<br>3.0305<br>3.0441<br>3.0546<br>3.0659<br>3.0811<br>3.0915<br>3.1029<br>3.1146<br>3.1264<br>3.1366<br>–<br>4.3622<br>5.0131<br>5.4091<br>–<br>5.8477<br>5.9880<br>6.0884<br>6.1684<br>6.2315<br>6.2853<br>6.3318<br>6.3679|



21 

Table 7: Final loss values with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. The TST loss used the power-law weighting as described in Appendix D. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9<br>1.0|3.0186<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>2.9963<br>2.9794<br>2.9693<br>2.9608<br>2.9552<br>2.9537<br>2.9522<br>2.9513<br>2.9524<br>2.9513<br>2.9499<br>2.9497<br>–<br>2.9861<br>2.9685<br>2.9611<br>2.9537<br>2.9469<br>2.9466<br>2.9435<br>2.9439<br>2.9434<br>2.9435<br>2.9437<br>2.9442<br>–<br>2.9831<br>2.9659<br>2.9589<br>2.9529<br>2.9465<br>2.9471<br>2.9445<br>2.9451<br>2.9448<br>2.9448<br>2.9460<br>2.9471<br>–<br>2.9825<br>2.9656<br>2.9597<br>2.9537<br>2.9492<br>2.9494<br>2.9465<br>2.9488<br>2.9494<br>2.9489<br>2.9515<br>2.9528<br>–<br>2.9834<br>2.9678<br>2.9625<br>2.9571<br>2.9535<br>2.9531<br>2.9517<br>2.9550<br>2.9564<br>2.9558<br>2.9584<br>2.9606<br>–<br>2.9860<br>2.9721<br>2.9679<br>2.9632<br>2.9592<br>2.9609<br>2.9608<br>2.9636<br>2.9666<br>2.9665<br>2.9697<br>2.9721<br>–<br>2.9908<br>2.9786<br>2.9766<br>2.9733<br>2.9707<br>2.9741<br>2.9752<br>2.9781<br>2.9824<br>2.9828<br>2.9865<br>2.9895<br>–<br>2.9988<br>2.9904<br>2.9922<br>2.9911<br>2.9909<br>2.9967<br>2.9988<br>3.0038<br>3.0098<br>3.0111<br>3.0163<br>3.0202<br>–<br>3.0162<br>3.0184<br>3.0277<br>3.0334<br>3.0403<br>3.0510<br>3.0579<br>3.0672<br>3.0785<br>3.0829<br>3.0925<br>3.0997<br>–<br>3.0669<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|



Table 8: Final ARC-Challenge evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.2491<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.2543<br>0.2509<br>0.2432<br>0.2491<br>0.2645<br>0.2517<br>0.2594<br>0.2432<br>0.2594<br>0.2526<br>0.2526<br>0.2526<br>0.2568<br>0.2440<br>0.2611<br>–<br>0.2500<br>0.2483<br>0.2526<br>0.2509<br>0.2457<br>0.2483<br>0.2517<br>0.2679<br>0.2560<br>0.2517<br>0.2628<br>0.2415<br>0.2509<br>0.2500<br>0.2602<br>–<br>0.2449<br>0.2602<br>0.2577<br>0.2534<br>0.2637<br>0.2509<br>0.2534<br>0.2568<br>0.2526<br>0.2543<br>0.2645<br>0.2577<br>0.2577<br>0.2491<br>0.2577<br>–<br>0.2466<br>0.2474<br>0.2398<br>0.2594<br>0.2637<br>0.2491<br>0.2483<br>0.2679<br>0.2509<br>0.2534<br>0.2739<br>0.2594<br>0.2628<br>0.2483<br>0.2551<br>–<br>0.2526<br>0.2713<br>0.2551<br>0.2491<br>0.2560<br>0.2594<br>0.2534<br>0.2500<br>0.2483<br>0.2517<br>0.2765<br>0.2449<br>0.2491<br>0.2483<br>–<br>–<br>0.2449<br>0.2543<br>0.2594<br>0.2543<br>0.2551<br>0.2526<br>0.2577<br>0.2398<br>0.2619<br>0.2440<br>0.2637<br>0.2517<br>0.2568<br>0.2551<br>0.2449<br>–<br>0.2440<br>0.2474<br>0.2551<br>0.2440<br>0.2534<br>0.2526<br>0.2457<br>0.2389<br>0.2568<br>0.2551<br>0.2602<br>0.2483<br>0.2645<br>0.2602<br>0.2517<br>–<br>0.2423<br>0.2398<br>0.2585<br>0.2355<br>0.2483<br>0.2577<br>0.2594<br>0.2304<br>0.2662<br>0.2577<br>0.2594<br>0.2474<br>0.2543<br>0.2568<br>0.2483<br>–<br>0.2346<br>0.2440<br>0.2398<br>0.2517<br>0.2483<br>0.2585<br>0.2491<br>0.2440<br>0.2432<br>0.2389<br>0.2440<br>0.2526<br>0.2432<br>0.2398<br>0.2363|



Table 9: Final ARC-Easy evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.4672<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.4600<br>0.4756<br>0.4760<br>0.4764<br>0.4756<br>0.4785<br>0.4764<br>0.4743<br>0.4689<br>0.4701<br>0.4739<br>0.4714<br>0.4663<br>0.4722<br>0.4853<br>–<br>0.4701<br>0.4680<br>0.4840<br>0.4672<br>0.4827<br>0.4705<br>0.4819<br>0.4790<br>0.4832<br>0.4886<br>0.4861<br>0.4811<br>0.4802<br>0.4891<br>0.4773<br>–<br>0.4655<br>0.4731<br>0.4781<br>0.4798<br>0.4764<br>0.4903<br>0.4983<br>0.4697<br>0.4916<br>0.4811<br>0.4920<br>0.4697<br>0.4844<br>0.4865<br>0.4903<br>–<br>0.4773<br>0.4689<br>0.4739<br>0.4802<br>0.4705<br>0.4815<br>0.4806<br>0.4844<br>0.4907<br>0.4676<br>0.4827<br>0.5000<br>0.4752<br>0.4815<br>0.4823<br>–<br>0.4781<br>0.4575<br>0.4769<br>0.4726<br>0.4840<br>0.4836<br>0.4979<br>0.4844<br>0.4899<br>0.4903<br>0.4912<br>0.4899<br>0.4823<br>0.4781<br>–<br>–<br>0.4735<br>0.4722<br>0.4714<br>0.4726<br>0.4840<br>0.4865<br>0.5000<br>0.4827<br>0.4832<br>0.4769<br>0.4857<br>0.4954<br>0.4790<br>0.4899<br>0.4836<br>–<br>0.4684<br>0.4676<br>0.4705<br>0.4651<br>0.4823<br>0.4811<br>0.4941<br>0.4815<br>0.4886<br>0.4827<br>0.4907<br>0.4764<br>0.4865<br>0.4823<br>0.4756<br>–<br>0.4781<br>0.4646<br>0.4600<br>0.4689<br>0.4731<br>0.4735<br>0.5029<br>0.4735<br>0.4962<br>0.4790<br>0.4752<br>0.4697<br>0.4668<br>0.4832<br>0.4840<br>–<br>0.4676<br>0.4722<br>0.4562<br>0.4668<br>0.4651<br>0.4668<br>0.4819<br>0.4832<br>0.4827<br>0.4638<br>0.4646<br>0.4781<br>0.4693<br>0.4693<br>0.4621|



Table 10: Final BoolQ evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5667<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.6049<br>0.5893<br>0.5700<br>0.4606<br>0.5566<br>0.5734<br>0.5832<br>0.5532<br>0.6000<br>0.5208<br>0.5578<br>0.5141<br>0.5951<br>0.5927<br>0.5985<br>–<br>0.5911<br>0.5731<br>0.5502<br>0.5878<br>0.5391<br>0.5737<br>0.5982<br>0.5636<br>0.5994<br>0.5703<br>0.5865<br>0.5532<br>0.5456<br>0.5887<br>0.5630<br>–<br>0.5734<br>0.5884<br>0.5343<br>0.5575<br>0.5398<br>0.5930<br>0.5609<br>0.5624<br>0.5798<br>0.5752<br>0.5440<br>0.5881<br>0.5391<br>0.5856<br>0.5951<br>–<br>0.5379<br>0.5972<br>0.5242<br>0.5483<br>0.5190<br>0.5979<br>0.5538<br>0.5832<br>0.5792<br>0.5560<br>0.5700<br>0.5618<br>0.5765<br>0.5832<br>0.5841<br>–<br>0.5489<br>0.5902<br>0.5498<br>0.5780<br>0.5765<br>0.5817<br>0.5440<br>0.5437<br>0.5324<br>0.5618<br>0.5902<br>0.5578<br>0.5615<br>0.5948<br>–<br>–<br>0.5462<br>0.5563<br>0.5884<br>0.5642<br>0.5303<br>0.5823<br>0.5679<br>0.5477<br>0.5489<br>0.5349<br>0.5468<br>0.5606<br>0.5862<br>0.5327<br>0.5862<br>–<br>0.5853<br>0.5862<br>0.5856<br>0.5761<br>0.5382<br>0.5618<br>0.5752<br>0.5609<br>0.5306<br>0.5474<br>0.5428<br>0.5740<br>0.5520<br>0.5657<br>0.5135<br>–<br>0.5850<br>0.5869<br>0.5813<br>0.5807<br>0.5471<br>0.5869<br>0.5336<br>0.5633<br>0.5474<br>0.5682<br>0.5404<br>0.5465<br>0.5945<br>0.5480<br>0.5795<br>–<br>0.5789<br>0.5835<br>0.5810<br>0.5523<br>0.5737<br>0.5829<br>0.5924<br>0.5884<br>0.5587<br>0.5725<br>0.5737<br>0.5664<br>0.5826<br>0.5869<br>0.5722|



Table 11: Final HellaSwag evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.3634<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.3689<br>0.3766<br>0.3861<br>0.3832<br>0.3859<br>0.3865<br>0.3849<br>0.3858<br>0.3875<br>0.3841<br>0.3871<br>0.3853<br>0.3848<br>0.3852<br>0.3846<br>–<br>0.3775<br>0.3860<br>0.3905<br>0.3926<br>0.3900<br>0.3909<br>0.3931<br>0.3921<br>0.3881<br>0.3869<br>0.3909<br>0.3891<br>0.3886<br>0.3851<br>0.3900<br>–<br>0.3789<br>0.3806<br>0.3897<br>0.3935<br>0.3856<br>0.3899<br>0.3893<br>0.3898<br>0.3907<br>0.3869<br>0.3870<br>0.3902<br>0.3848<br>0.3873<br>0.3858<br>–<br>0.3756<br>0.3850<br>0.3914<br>0.3920<br>0.3892<br>0.3911<br>0.3911<br>0.3882<br>0.3872<br>0.3847<br>0.3903<br>0.3860<br>0.3922<br>0.3854<br>0.3857<br>–<br>0.3752<br>0.3847<br>0.3889<br>0.3902<br>0.3883<br>0.3923<br>0.3831<br>0.3884<br>0.3854<br>0.3819<br>0.3843<br>0.3826<br>0.3843<br>0.3844<br>–<br>–<br>0.3765<br>0.3805<br>0.3876<br>0.3886<br>0.3822<br>0.3809<br>0.3845<br>0.3837<br>0.3807<br>0.3785<br>0.3832<br>0.3816<br>0.3828<br>0.3758<br>0.3787<br>–<br>0.3745<br>0.3833<br>0.3865<br>0.3844<br>0.3781<br>0.3774<br>0.3777<br>0.3790<br>0.3773<br>0.3746<br>0.3745<br>0.3713<br>0.3743<br>0.3712<br>0.3731<br>–<br>0.3690<br>0.3781<br>0.3818<br>0.3786<br>0.3706<br>0.3704<br>0.3702<br>0.3705<br>0.3687<br>0.3638<br>0.3614<br>0.3675<br>0.3617<br>0.3609<br>0.3593<br>–<br>0.3665<br>0.3681<br>0.3676<br>0.3615<br>0.3540<br>0.3575<br>0.3505<br>0.3510<br>0.3439<br>0.3422<br>0.3420<br>0.3347<br>0.3342<br>0.3354<br>0.3360|



22 

Table 12: Final OpenBookQA evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.2920<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.2960<br>0.2960<br>0.3160<br>0.3020<br>0.3060<br>0.3060<br>0.3160<br>0.3060<br>0.3000<br>0.3140<br>0.2960<br>0.3080<br>0.2940<br>0.3240<br>0.3120<br>–<br>0.3060<br>0.3060<br>0.2980<br>0.2820<br>0.2960<br>0.3160<br>0.3000<br>0.2940<br>0.3040<br>0.3120<br>0.3080<br>0.2920<br>0.3000<br>0.3060<br>0.3040<br>–<br>0.2900<br>0.3120<br>0.3000<br>0.3060<br>0.2980<br>0.3080<br>0.3060<br>0.3120<br>0.3160<br>0.3180<br>0.3060<br>0.3100<br>0.2860<br>0.3040<br>0.2940<br>–<br>0.3100<br>0.2940<br>0.3060<br>0.3040<br>0.3100<br>0.2900<br>0.2960<br>0.2940<br>0.3140<br>0.3040<br>0.2840<br>0.3020<br>0.3020<br>0.3220<br>0.2920<br>–<br>0.2980<br>0.3020<br>0.3160<br>0.2900<br>0.3240<br>0.2860<br>0.2940<br>0.2900<br>0.3180<br>0.2800<br>0.2920<br>0.3060<br>0.2980<br>0.3100<br>–<br>–<br>0.3000<br>0.2860<br>0.3180<br>0.3180<br>0.3060<br>0.3040<br>0.2900<br>0.2880<br>0.3000<br>0.2940<br>0.2780<br>0.2880<br>0.2980<br>0.3140<br>0.2940<br>–<br>0.2980<br>0.2980<br>0.3100<br>0.3140<br>0.3060<br>0.2800<br>0.2880<br>0.2780<br>0.3060<br>0.2860<br>0.2920<br>0.2900<br>0.2960<br>0.2980<br>0.2900<br>–<br>0.2980<br>0.2780<br>0.3100<br>0.3100<br>0.3040<br>0.2860<br>0.2920<br>0.2740<br>0.2940<br>0.2940<br>0.2960<br>0.2940<br>0.2860<br>0.2980<br>0.2980<br>–<br>0.2980<br>0.2880<br>0.3100<br>0.3000<br>0.3040<br>0.2760<br>0.2800<br>0.2760<br>0.2860<br>0.3000<br>0.2960<br>0.2820<br>0.2900<br>0.3020<br>0.2880|



Table 13: Final PIQA evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.6638<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.6697<br>0.6736<br>0.6719<br>0.6654<br>0.6643<br>0.6714<br>0.6752<br>0.6703<br>0.6817<br>0.6730<br>0.6725<br>0.6757<br>0.6681<br>0.6659<br>0.6736<br>–<br>0.6703<br>0.6708<br>0.6741<br>0.6659<br>0.6703<br>0.6714<br>0.6725<br>0.6703<br>0.6681<br>0.6643<br>0.6801<br>0.6659<br>0.6785<br>0.6801<br>0.6692<br>–<br>0.6676<br>0.6659<br>0.6665<br>0.6790<br>0.6703<br>0.6708<br>0.6708<br>0.6708<br>0.6708<br>0.6703<br>0.6757<br>0.6708<br>0.6779<br>0.6779<br>0.6616<br>–<br>0.6621<br>0.6654<br>0.6643<br>0.6627<br>0.6790<br>0.6719<br>0.6817<br>0.6681<br>0.6763<br>0.6757<br>0.6708<br>0.6746<br>0.6692<br>0.6752<br>0.6665<br>–<br>0.6708<br>0.6768<br>0.6578<br>0.6741<br>0.6763<br>0.6665<br>0.6659<br>0.6736<br>0.6736<br>0.6670<br>0.6736<br>0.6730<br>0.6752<br>0.6725<br>–<br>–<br>0.6649<br>0.6676<br>0.6605<br>0.6665<br>0.6768<br>0.6714<br>0.6621<br>0.6741<br>0.6714<br>0.6703<br>0.6687<br>0.6681<br>0.6736<br>0.6741<br>0.6741<br>–<br>0.6649<br>0.6610<br>0.6632<br>0.6659<br>0.6719<br>0.6572<br>0.6714<br>0.6714<br>0.6752<br>0.6757<br>0.6638<br>0.6610<br>0.6801<br>0.6681<br>0.6567<br>–<br>0.6572<br>0.6676<br>0.6545<br>0.6621<br>0.6736<br>0.6654<br>0.6687<br>0.6621<br>0.6654<br>0.6703<br>0.6659<br>0.6578<br>0.6567<br>0.6649<br>0.6627<br>–<br>0.6610<br>0.6643<br>0.6513<br>0.6453<br>0.6600<br>0.6643<br>0.6502<br>0.6420<br>0.6534<br>0.6534<br>0.6442<br>0.6436<br>0.6507<br>0.6583<br>0.6534|



Table 14: Final Winogrande evals with varying _r_ and _s_ , for 270M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5130<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.5193<br>0.5312<br>0.5107<br>0.5130<br>0.5257<br>0.5114<br>0.5193<br>0.5099<br>0.5162<br>0.5375<br>0.5209<br>0.5264<br>0.5091<br>0.5154<br>0.5241<br>–<br>0.5114<br>0.4957<br>0.5185<br>0.5146<br>0.5264<br>0.5091<br>0.5391<br>0.5114<br>0.5067<br>0.5185<br>0.5162<br>0.5249<br>0.4988<br>0.5257<br>0.5257<br>–<br>0.5367<br>0.5091<br>0.5233<br>0.5233<br>0.5107<br>0.5020<br>0.5083<br>0.5280<br>0.5099<br>0.5280<br>0.5028<br>0.4957<br>0.5225<br>0.5107<br>0.5446<br>–<br>0.5130<br>0.5280<br>0.5130<br>0.5233<br>0.5130<br>0.5099<br>0.5241<br>0.5028<br>0.5217<br>0.5099<br>0.5272<br>0.5178<br>0.4925<br>0.5185<br>0.5028<br>–<br>0.5414<br>0.5154<br>0.5154<br>0.5138<br>0.5257<br>0.5367<br>0.5233<br>0.5028<br>0.5067<br>0.5178<br>0.5328<br>0.5146<br>0.5154<br>0.5391<br>–<br>–<br>0.5146<br>0.5067<br>0.5146<br>0.5162<br>0.5036<br>0.5059<br>0.5296<br>0.5075<br>0.5083<br>0.5162<br>0.5020<br>0.5107<br>0.5264<br>0.5233<br>0.5146<br>–<br>0.5170<br>0.5099<br>0.4988<br>0.5004<br>0.5178<br>0.5083<br>0.5059<br>0.5217<br>0.5043<br>0.5091<br>0.5280<br>0.5004<br>0.4988<br>0.5351<br>0.5051<br>–<br>0.5233<br>0.5051<br>0.4901<br>0.5114<br>0.5028<br>0.5178<br>0.5185<br>0.5146<br>0.5178<br>0.5170<br>0.5075<br>0.5091<br>0.5099<br>0.5012<br>0.5114<br>–<br>0.5209<br>0.5020<br>0.5028<br>0.5020<br>0.4972<br>0.5067<br>0.5130<br>0.5162<br>0.5146<br>0.5114<br>0.5114<br>0.4925<br>0.5249<br>0.5020<br>0.5020|



Table 15: Final ARC-Challenge evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|steps.|See Table 1 for trainingdetails.|
|---|---|
|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.2619<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.2568<br>0.2619<br>0.2534<br>0.2671<br>0.2654<br>0.2654<br>0.2654<br>0.2688<br>0.2645<br>0.2440<br>0.2543<br>0.2782<br>–<br>0.2517<br>0.2654<br>0.2662<br>0.2637<br>0.2645<br>0.2577<br>0.2619<br>0.2628<br>0.2773<br>0.2594<br>0.2730<br>0.2756<br>–<br>0.2611<br>0.2577<br>0.2705<br>0.2568<br>0.2551<br>0.2534<br>0.2696<br>0.2602<br>0.2662<br>0.2739<br>0.2688<br>0.2645<br>–<br>0.2628<br>0.2568<br>0.2517<br>0.2568<br>0.2611<br>0.2722<br>0.2739<br>0.2662<br>0.2654<br>0.2585<br>0.2611<br>0.2611<br>–<br>0.2449<br>0.2585<br>0.2577<br>0.2662<br>0.2645<br>0.2628<br>0.2602<br>0.2637<br>0.2602<br>0.2534<br>0.2628<br>0.2688<br>–<br>0.2500<br>0.2602<br>0.2543<br>0.2585<br>0.2577<br>0.2577<br>0.2611<br>0.2602<br>0.2773<br>0.2739<br>0.2611<br>0.2560<br>–<br>0.2534<br>0.2705<br>0.2637<br>0.2560<br>0.2662<br>0.2705<br>0.2637<br>0.2645<br>0.2628<br>0.2705<br>0.2713<br>0.2679<br>–<br>0.2432<br>0.2611<br>0.2619<br>0.2432<br>0.2628<br>0.2594<br>0.2645<br>0.2483<br>0.2662<br>0.2619<br>0.2722<br>0.2602<br>–<br>0.2551<br>0.2602<br>0.2747<br>0.2483<br>0.2517<br>0.2474<br>0.2491<br>0.2747<br>0.2585<br>0.2466<br>0.2611<br>0.2594|



Table 16: Final ARC-Easy evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.4747<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.4870<br>0.4941<br>0.4903<br>0.5029<br>0.5202<br>0.4954<br>0.5004<br>0.5198<br>0.5114<br>0.4992<br>0.4920<br>0.5088<br>–<br>0.4731<br>0.4924<br>0.4907<br>0.5332<br>0.5008<br>0.4895<br>0.4983<br>0.5130<br>0.4992<br>0.5051<br>0.5223<br>0.5034<br>–<br>0.4815<br>0.4979<br>0.5021<br>0.5160<br>0.5034<br>0.5181<br>0.5278<br>0.5173<br>0.5067<br>0.4945<br>0.5156<br>0.5114<br>–<br>0.4794<br>0.5025<br>0.4899<br>0.5051<br>0.5072<br>0.5017<br>0.5303<br>0.5223<br>0.5236<br>0.5181<br>0.5147<br>0.5021<br>–<br>0.4693<br>0.5038<br>0.5013<br>0.5189<br>0.4992<br>0.5156<br>0.5114<br>0.5072<br>0.5029<br>0.5177<br>0.5046<br>0.4983<br>–<br>0.4769<br>0.5093<br>0.5093<br>0.4920<br>0.5076<br>0.4962<br>0.5004<br>0.5059<br>0.4979<br>0.5004<br>0.5101<br>0.5114<br>–<br>0.4735<br>0.4933<br>0.4992<br>0.4832<br>0.4954<br>0.4987<br>0.4895<br>0.5000<br>0.5114<br>0.5122<br>0.4979<br>0.4827<br>–<br>0.4663<br>0.4836<br>0.4975<br>0.4886<br>0.4920<br>0.5067<br>0.5101<br>0.5042<br>0.4962<br>0.5059<br>0.5105<br>0.4987<br>–<br>0.4760<br>0.4920<br>0.4907<br>0.4823<br>0.4823<br>0.5021<br>0.5076<br>0.4996<br>0.4794<br>0.4815<br>0.5059<br>0.4853|



23 

Table 17: Final BoolQ evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5847<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.5223<br>0.5606<br>0.5245<br>0.5713<br>0.5618<br>0.5076<br>0.5933<br>0.5572<br>0.5621<br>0.5336<br>0.5550<br>0.5786<br>–<br>0.5260<br>0.6000<br>0.5318<br>0.5960<br>0.5431<br>0.5798<br>0.5462<br>0.5621<br>0.5083<br>0.5713<br>0.5746<br>0.5309<br>–<br>0.5391<br>0.5841<br>0.5535<br>0.5410<br>0.5789<br>0.5761<br>0.5566<br>0.5847<br>0.4927<br>0.5352<br>0.5190<br>0.5673<br>–<br>0.5131<br>0.5713<br>0.5575<br>0.5557<br>0.5187<br>0.5685<br>0.5557<br>0.5336<br>0.5382<br>0.5272<br>0.4838<br>0.5872<br>–<br>0.6000<br>0.5046<br>0.5612<br>0.5468<br>0.5700<br>0.5596<br>0.5676<br>0.5930<br>0.5554<br>0.5685<br>0.5468<br>0.5789<br>–<br>0.6141<br>0.5092<br>0.5083<br>0.5979<br>0.5957<br>0.5988<br>0.5416<br>0.5300<br>0.5297<br>0.5339<br>0.5367<br>0.5633<br>–<br>0.5966<br>0.5306<br>0.5113<br>0.5480<br>0.5899<br>0.5471<br>0.5694<br>0.5346<br>0.5080<br>0.5119<br>0.4957<br>0.5318<br>–<br>0.5869<br>0.5495<br>0.4936<br>0.5615<br>0.5511<br>0.5859<br>0.5544<br>0.5621<br>0.4966<br>0.4547<br>0.5220<br>0.5554<br>–<br>0.6162<br>0.5431<br>0.5024<br>0.5832<br>0.5388<br>0.5719<br>0.5758<br>0.5709<br>0.5116<br>0.5554<br>0.5462<br>0.5560|



Table 18: Final HellaSwag evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.4021<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.4107<br>0.4190<br>0.4237<br>0.4225<br>0.4252<br>0.4284<br>0.4281<br>0.4261<br>0.4276<br>0.4227<br>0.4292<br>0.4309<br>–<br>0.4122<br>0.4176<br>0.4260<br>0.4277<br>0.4266<br>0.4269<br>0.4267<br>0.4267<br>0.4262<br>0.4288<br>0.4243<br>0.4280<br>–<br>0.4143<br>0.4179<br>0.4306<br>0.4275<br>0.4261<br>0.4260<br>0.4265<br>0.4284<br>0.4238<br>0.4275<br>0.4243<br>0.4236<br>–<br>0.4137<br>0.4200<br>0.4265<br>0.4340<br>0.4279<br>0.4248<br>0.4290<br>0.4266<br>0.4261<br>0.4291<br>0.4246<br>0.4247<br>–<br>0.4145<br>0.4227<br>0.4247<br>0.4272<br>0.4246<br>0.4282<br>0.4219<br>0.4251<br>0.4219<br>0.4256<br>0.4219<br>0.4266<br>–<br>0.4144<br>0.4192<br>0.4226<br>0.4250<br>0.4244<br>0.4231<br>0.4256<br>0.4262<br>0.4200<br>0.4196<br>0.4180<br>0.4180<br>–<br>0.4124<br>0.4192<br>0.4214<br>0.4229<br>0.4195<br>0.4206<br>0.4188<br>0.4220<br>0.4141<br>0.4164<br>0.4160<br>0.4148<br>–<br>0.4102<br>0.4164<br>0.4224<br>0.4152<br>0.4153<br>0.4171<br>0.4175<br>0.4132<br>0.4120<br>0.4113<br>0.4068<br>0.4077<br>–<br>0.4100<br>0.4091<br>0.4089<br>0.4097<br>0.4019<br>0.4020<br>0.4049<br>0.3976<br>0.3950<br>0.3940<br>0.3921<br>0.3919|



Table 19: Final OpenBookQA evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|Table<br>steps.|19: Final OpenBookQA evals with varying_r_ and_s_, for 270M models trained for 100k total<br> See Table 1 for trainingdetails.|
|---|---|
|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.3060<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.3200<br>0.3020<br>0.3120<br>0.3100<br>0.3180<br>0.3080<br>0.2980<br>0.3140<br>0.3140<br>0.3140<br>0.3200<br>0.3340<br>–<br>0.3040<br>0.3220<br>0.3180<br>0.3220<br>0.3220<br>0.3020<br>0.3040<br>0.3100<br>0.3180<br>0.3220<br>0.3040<br>0.3240<br>–<br>0.3260<br>0.3400<br>0.3120<br>0.3260<br>0.3240<br>0.3220<br>0.3220<br>0.3180<br>0.3300<br>0.3060<br>0.3160<br>0.3160<br>–<br>0.2980<br>0.3100<br>0.3000<br>0.3220<br>0.3100<br>0.3080<br>0.3060<br>0.3200<br>0.3280<br>0.3040<br>0.3080<br>0.3040<br>–<br>0.3200<br>0.3160<br>0.3180<br>0.3240<br>0.3060<br>0.3220<br>0.3180<br>0.3000<br>0.3200<br>0.3140<br>0.3160<br>0.3260<br>–<br>0.3300<br>0.3300<br>0.3240<br>0.3040<br>0.3080<br>0.3120<br>0.3180<br>0.2980<br>0.3060<br>0.3080<br>0.3160<br>0.3240<br>–<br>0.2980<br>0.3080<br>0.3080<br>0.3180<br>0.3100<br>0.2900<br>0.3040<br>0.3060<br>0.2940<br>0.3080<br>0.3340<br>0.3020<br>–<br>0.3060<br>0.3160<br>0.3080<br>0.3220<br>0.3060<br>0.3120<br>0.2980<br>0.3200<br>0.2860<br>0.3320<br>0.2920<br>0.3020<br>–<br>0.2860<br>0.3280<br>0.3100<br>0.3180<br>0.3200<br>0.3060<br>0.3120<br>0.3140<br>0.2980<br>0.3080<br>0.2960<br>0.3020|



Table 20: Final PIQA evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.6730<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.6746<br>0.6844<br>0.6785<br>0.6785<br>0.6942<br>0.6855<br>0.6861<br>0.6774<br>0.6834<br>0.6861<br>0.6812<br>0.6904<br>–<br>0.6812<br>0.6779<br>0.6801<br>0.6779<br>0.6921<br>0.6861<br>0.6899<br>0.6850<br>0.6844<br>0.6844<br>0.6861<br>0.6801<br>–<br>0.6785<br>0.6817<br>0.6844<br>0.6801<br>0.6904<br>0.6752<br>0.6790<br>0.6855<br>0.6823<br>0.6877<br>0.6893<br>0.6910<br>–<br>0.6790<br>0.6839<br>0.6828<br>0.6823<br>0.6872<br>0.6779<br>0.6839<br>0.6937<br>0.6823<br>0.6964<br>0.6866<br>0.6861<br>–<br>0.6844<br>0.6779<br>0.6785<br>0.6801<br>0.6937<br>0.6855<br>0.6855<br>0.6921<br>0.6844<br>0.6915<br>0.6877<br>0.6899<br>–<br>0.6806<br>0.6834<br>0.6834<br>0.6882<br>0.6899<br>0.6834<br>0.6948<br>0.6888<br>0.6834<br>0.6823<br>0.6839<br>0.6828<br>–<br>0.6708<br>0.6752<br>0.6757<br>0.6779<br>0.6926<br>0.6763<br>0.6921<br>0.6866<br>0.6823<br>0.6806<br>0.6806<br>0.6741<br>–<br>0.6665<br>0.6752<br>0.6779<br>0.6817<br>0.6844<br>0.6855<br>0.6806<br>0.6823<br>0.6844<br>0.6768<br>0.6828<br>0.6795<br>–<br>0.6665<br>0.6654<br>0.6692<br>0.6844<br>0.6806<br>0.6779<br>0.6861<br>0.6757<br>0.6741<br>0.6714<br>0.6681<br>0.6659|



24 

Table 21: Final Winogrande evals with varying _r_ and _s_ , for 270M models trained for 100k total steps. See Table 1 for training details. 

|Table <br>steps.|21: Final Winogrande evals with varying _r_ and _s_, for 270M models trained for 100k total<br> See Table 1 for trainingdetails.|
|---|---|
|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5154<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.5170<br>0.5351<br>0.5036<br>0.5209<br>0.5343<br>0.5272<br>0.5272<br>0.5351<br>0.5328<br>0.5391<br>0.5446<br>0.5122<br>–<br>0.5193<br>0.5288<br>0.5304<br>0.5406<br>0.5454<br>0.5422<br>0.5272<br>0.5399<br>0.5288<br>0.5383<br>0.5193<br>0.5343<br>–<br>0.5185<br>0.5170<br>0.5320<br>0.5257<br>0.5422<br>0.5296<br>0.5414<br>0.5193<br>0.5304<br>0.5296<br>0.5375<br>0.5288<br>–<br>0.5288<br>0.5280<br>0.5201<br>0.5414<br>0.5257<br>0.5406<br>0.5383<br>0.5470<br>0.5201<br>0.5288<br>0.5375<br>0.5280<br>–<br>0.5075<br>0.5146<br>0.5335<br>0.5304<br>0.5320<br>0.5430<br>0.5320<br>0.5304<br>0.5162<br>0.5343<br>0.5264<br>0.5438<br>–<br>0.4957<br>0.5296<br>0.5241<br>0.5312<br>0.5383<br>0.5296<br>0.5375<br>0.5383<br>0.5288<br>0.5091<br>0.5462<br>0.5414<br>–<br>0.5130<br>0.5225<br>0.5375<br>0.5249<br>0.5264<br>0.5320<br>0.5391<br>0.5264<br>0.5225<br>0.5280<br>0.5296<br>0.5209<br>–<br>0.5233<br>0.5257<br>0.5083<br>0.5051<br>0.5430<br>0.5272<br>0.5272<br>0.5257<br>0.5304<br>0.5280<br>0.5359<br>0.5114<br>–<br>0.5280<br>0.5178<br>0.5170<br>0.5193<br>0.5178<br>0.5170<br>0.5083<br>0.5178<br>0.5075<br>0.5209<br>0.5067<br>0.5178|



Table 22: Final ARC-Challenge evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|Table<br>steps.|22: Final ARC-Challenge evals with varying_r_ and_s_, for 600M models trained for 20k total<br> See Table 1 for trainingdetails.|
|---|---|
|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.2551<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.2628<br>0.2671<br>0.2816<br>0.2747<br>0.2833<br>0.2952<br>0.2739<br>0.2892<br>0.2850<br>0.2765<br>0.2833<br>0.2790<br>–<br>0.2679<br>0.2765<br>0.2782<br>0.2782<br>0.2790<br>0.2867<br>0.3029<br>0.2867<br>0.2799<br>0.2918<br>0.3012<br>0.2927<br>–<br>0.2645<br>0.2765<br>0.2688<br>0.2705<br>0.2688<br>0.2944<br>0.2824<br>0.2824<br>0.2816<br>0.2961<br>0.2969<br>0.2816<br>–<br>0.2628<br>0.2782<br>0.2705<br>0.2850<br>0.2730<br>0.2867<br>0.2875<br>0.2910<br>0.2901<br>0.2850<br>0.2892<br>0.3063<br>–<br>0.2765<br>0.2807<br>0.2884<br>0.2782<br>0.2611<br>0.2961<br>0.2824<br>0.2910<br>0.2961<br>0.2807<br>0.3012<br>0.2952<br>–<br>0.2688<br>0.2696<br>0.2799<br>0.2773<br>0.2671<br>0.2910<br>0.2765<br>0.2858<br>0.2935<br>0.2824<br>0.2995<br>0.2892<br>–<br>0.2696<br>0.2688<br>0.2867<br>0.2688<br>0.2782<br>0.2816<br>0.2858<br>0.2816<br>0.2875<br>0.2961<br>0.2910<br>0.2918<br>–<br>0.2747<br>0.2875<br>0.2944<br>0.2799<br>0.2696<br>0.2637<br>0.2824<br>0.2884<br>0.2875<br>0.2867<br>0.2816<br>0.2782<br>–<br>0.2645<br>0.2722<br>0.2671<br>0.2816<br>0.2688<br>0.2628<br>0.2594<br>0.2705<br>0.2705<br>0.2773<br>0.2747<br>0.2730|



Table 23: Final ARC-Easy evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5168<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>0.5210<br>0.5530<br>0.5412<br>0.5307<br>0.5320<br>0.5286<br>0.5391<br>0.5219<br>0.5383<br>0.5370<br>0.5480<br>0.5526<br>-<br>0.5261<br>0.5328<br>0.5543<br>0.5429<br>0.5202<br>0.5362<br>0.5442<br>0.5501<br>0.5450<br>0.5644<br>0.5564<br>0.5593<br>-<br>0.5152<br>0.5391<br>0.5450<br>0.5261<br>0.5248<br>0.5383<br>0.5320<br>0.5543<br>0.5535<br>0.5492<br>0.5606<br>0.5455<br>-<br>0.5173<br>0.5438<br>0.5610<br>0.5265<br>0.5236<br>0.5379<br>0.5467<br>0.5400<br>0.5522<br>0.5467<br>0.5619<br>0.5556<br>-<br>0.5118<br>0.5349<br>0.5484<br>0.5362<br>0.5248<br>0.5442<br>0.5299<br>0.5547<br>0.5450<br>0.5425<br>0.5661<br>0.5593<br>-<br>0.5038<br>0.5459<br>0.5412<br>0.5455<br>0.5417<br>0.5467<br>0.5417<br>0.5543<br>0.5665<br>0.5467<br>0.5657<br>0.5518<br>-<br>0.5093<br>0.5366<br>0.5396<br>0.5425<br>0.5379<br>0.5539<br>0.5311<br>0.5614<br>0.5459<br>0.5421<br>0.5530<br>0.5366<br>-<br>0.5093<br>0.5400<br>0.5446<br>0.5375<br>0.5332<br>0.5463<br>0.5370<br>0.5341<br>0.5442<br>0.5497<br>0.5450<br>0.5332<br>-<br>0.5000<br>0.5370<br>0.5311<br>0.5236<br>0.5311<br>0.5290<br>0.5215<br>0.5223<br>0.5404<br>0.5265<br>0.5408<br>0.5434|



Table 24: Final BoolQ evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5661<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.5924<br>0.5924<br>0.5896<br>0.6098<br>0.5517<br>0.5920<br>0.6214<br>0.6031<br>0.5722<br>0.5697<br>0.5642<br>0.5789<br>–<br>0.5761<br>0.5991<br>0.5994<br>0.6254<br>0.6070<br>0.5783<br>0.5612<br>0.5364<br>0.5300<br>0.5327<br>0.6110<br>0.6012<br>–<br>0.6000<br>0.5810<br>0.5988<br>0.6024<br>0.5477<br>0.5336<br>0.5333<br>0.5483<br>0.5462<br>0.5456<br>0.5865<br>0.5670<br>–<br>0.5661<br>0.5872<br>0.5963<br>0.6037<br>0.5703<br>0.5297<br>0.5664<br>0.5703<br>0.5550<br>0.5532<br>0.5768<br>0.5911<br>–<br>0.5832<br>0.6009<br>0.6052<br>0.6162<br>0.5615<br>0.5697<br>0.5471<br>0.5070<br>0.5673<br>0.5119<br>0.5330<br>0.5985<br>–<br>0.5394<br>0.5823<br>0.6257<br>0.5942<br>0.5899<br>0.6024<br>0.5520<br>0.5654<br>0.5410<br>0.5346<br>0.5453<br>0.5664<br>–<br>0.5798<br>0.5627<br>0.6110<br>0.6043<br>0.5847<br>0.5624<br>0.5780<br>0.5777<br>0.5587<br>0.5125<br>0.6135<br>0.5884<br>–<br>0.5813<br>0.5838<br>0.6187<br>0.6168<br>0.5960<br>0.5758<br>0.5636<br>0.5847<br>0.5453<br>0.5211<br>0.5905<br>0.5755<br>–<br>0.5761<br>0.5869<br>0.5960<br>0.6183<br>0.5645<br>0.5355<br>0.5498<br>0.5905<br>0.5829<br>0.5951<br>0.6110<br>0.5648|



25 

Table 25: Final HellaSwag evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.4347<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.4554<br>0.4658<br>0.4780<br>0.4763<br>0.4736<br>0.4760<br>0.4756<br>0.4750<br>0.4771<br>0.4739<br>0.4787<br>0.4764<br>–<br>0.4628<br>0.4711<br>0.4806<br>0.4802<br>0.4810<br>0.4823<br>0.4855<br>0.4832<br>0.4813<br>0.4797<br>0.4809<br>0.4811<br>–<br>0.4620<br>0.4726<br>0.4788<br>0.4832<br>0.4817<br>0.4777<br>0.4840<br>0.4837<br>0.4825<br>0.4825<br>0.4793<br>0.4806<br>–<br>0.4613<br>0.4763<br>0.4797<br>0.4814<br>0.4802<br>0.4804<br>0.4812<br>0.4811<br>0.4792<br>0.4727<br>0.4749<br>0.4753<br>–<br>0.4642<br>0.4783<br>0.4812<br>0.4808<br>0.4781<br>0.4795<br>0.4767<br>0.4795<br>0.4757<br>0.4810<br>0.4718<br>0.4725<br>–<br>0.4614<br>0.4710<br>0.4780<br>0.4754<br>0.4755<br>0.4717<br>0.4695<br>0.4721<br>0.4706<br>0.4689<br>0.4671<br>0.4631<br>–<br>0.4588<br>0.4682<br>0.4740<br>0.4702<br>0.4706<br>0.4648<br>0.4679<br>0.4654<br>0.4590<br>0.4593<br>0.4556<br>0.4527<br>–<br>0.4561<br>0.4658<br>0.4685<br>0.4588<br>0.4582<br>0.4568<br>0.4524<br>0.4485<br>0.4465<br>0.4448<br>0.4414<br>0.4367<br>–<br>0.4494<br>0.4507<br>0.4482<br>0.4417<br>0.4343<br>0.4265<br>0.4257<br>0.4207<br>0.4107<br>0.4076<br>0.4033<br>0.4005|



Table 26: Final OpenBookQA evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|Table <br>steps.|26: Final OpenBookQA evals with varying _r_ and _s_, for 600M models trained for 20k total<br> See Table 1 for trainingdetails.|
|---|---|
|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.3120<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.3200<br>0.3220<br>0.3280<br>0.3440<br>0.3460<br>0.3320<br>0.3480<br>0.3160<br>0.3540<br>0.3560<br>0.3220<br>0.3480<br>–<br>0.3360<br>0.3260<br>0.3320<br>0.3380<br>0.3160<br>0.3300<br>0.3220<br>0.3280<br>0.3360<br>0.3260<br>0.3340<br>0.3380<br>–<br>0.3280<br>0.3340<br>0.3280<br>0.3100<br>0.3500<br>0.3340<br>0.3480<br>0.3340<br>0.3420<br>0.3240<br>0.3400<br>0.3180<br>–<br>0.3320<br>0.3340<br>0.3440<br>0.3320<br>0.3460<br>0.3140<br>0.3220<br>0.3340<br>0.3300<br>0.3400<br>0.3380<br>0.3420<br>–<br>0.3320<br>0.3420<br>0.3180<br>0.3300<br>0.3280<br>0.3260<br>0.3360<br>0.3420<br>0.3060<br>0.3220<br>0.3220<br>0.3200<br>–<br>0.3320<br>0.3340<br>0.3320<br>0.3200<br>0.3320<br>0.3260<br>0.3240<br>0.3500<br>0.3200<br>0.3260<br>0.3460<br>0.3320<br>–<br>0.3380<br>0.3380<br>0.3480<br>0.3000<br>0.3380<br>0.3300<br>0.3500<br>0.3500<br>0.3140<br>0.3320<br>0.3400<br>0.3240<br>–<br>0.3260<br>0.3340<br>0.3240<br>0.3180<br>0.3160<br>0.3200<br>0.3400<br>0.3480<br>0.3380<br>0.3380<br>0.3240<br>0.3200<br>–<br>0.3040<br>0.3280<br>0.3140<br>0.3060<br>0.3080<br>0.3160<br>0.3280<br>0.3440<br>0.3180<br>0.3340<br>0.3060<br>0.3140|



Table 27: Final PIQA evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.6904<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>0.6991<br>0.7062<br>0.7002<br>0.7024<br>0.6997<br>0.7155<br>0.7073<br>0.7095<br>0.7008<br>0.7057<br>0.7062<br>0.6964<br>-<br>0.7057<br>0.7089<br>0.6964<br>0.7067<br>0.7029<br>0.7127<br>0.6980<br>0.7095<br>0.7127<br>0.7089<br>0.7029<br>0.7089<br>-<br>0.7018<br>0.7095<br>0.7013<br>0.7024<br>0.7057<br>0.7127<br>0.7051<br>0.7144<br>0.7062<br>0.7051<br>0.7095<br>0.7127<br>-<br>0.7024<br>0.7067<br>0.7008<br>0.7095<br>0.6980<br>0.7013<br>0.7067<br>0.7138<br>0.7198<br>0.7100<br>0.7127<br>0.7078<br>-<br>0.6997<br>0.7057<br>0.7122<br>0.7084<br>0.6997<br>0.7040<br>0.7111<br>0.7127<br>0.7100<br>0.7002<br>0.7078<br>0.7008<br>-<br>0.6937<br>0.6986<br>0.7073<br>0.6970<br>0.7046<br>0.7067<br>0.7067<br>0.7127<br>0.7160<br>0.7084<br>0.7029<br>0.6915<br>-<br>0.6980<br>0.7062<br>0.7013<br>0.6953<br>0.7084<br>0.7029<br>0.7008<br>0.7160<br>0.7024<br>0.7057<br>0.7002<br>0.6959<br>-<br>0.7057<br>0.6921<br>0.7018<br>0.7024<br>0.6980<br>0.6991<br>0.7008<br>0.7073<br>0.7062<br>0.7018<br>0.7024<br>0.6991<br>-<br>0.6877<br>0.6975<br>0.6942<br>0.6866<br>0.6877<br>0.6834<br>0.6861<br>0.6948<br>0.6910<br>0.6882<br>0.6882<br>0.6757|



Table 28: Final Winogrande evals with varying _r_ and _s_ , for 600M models trained for 20k total steps. See Table 1 for training details. 

|_r_<br>_s_|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13|
|---|---|
|0.0<br>0.1<br>0.2<br>0.3<br>0.4<br>0.5<br>0.6<br>0.7<br>0.8<br>0.9|0.5257<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>0.5359<br>0.5446<br>0.5493<br>0.5438<br>0.5493<br>0.5533<br>0.5335<br>0.5478<br>0.5517<br>0.5604<br>0.5430<br>0.5422<br>–<br>0.5643<br>0.5280<br>0.5493<br>0.5675<br>0.5462<br>0.5478<br>0.5406<br>0.5359<br>0.5564<br>0.5485<br>0.5517<br>0.5399<br>–<br>0.5328<br>0.5320<br>0.5430<br>0.5509<br>0.5391<br>0.5620<br>0.5328<br>0.5320<br>0.5454<br>0.5501<br>0.5454<br>0.5596<br>–<br>0.5288<br>0.5422<br>0.5193<br>0.5509<br>0.5501<br>0.5667<br>0.5320<br>0.5383<br>0.5375<br>0.5422<br>0.5438<br>0.5422<br>–<br>0.5328<br>0.5493<br>0.5375<br>0.5375<br>0.5517<br>0.5604<br>0.5391<br>0.5501<br>0.5446<br>0.5399<br>0.5264<br>0.5754<br>–<br>0.5509<br>0.5391<br>0.5249<br>0.5501<br>0.5470<br>0.5509<br>0.5304<br>0.5454<br>0.5343<br>0.5580<br>0.5406<br>0.5596<br>–<br>0.5517<br>0.5549<br>0.5335<br>0.5438<br>0.5478<br>0.5509<br>0.5391<br>0.5335<br>0.5264<br>0.5406<br>0.5280<br>0.5422<br>–<br>0.5335<br>0.5399<br>0.5501<br>0.5359<br>0.5525<br>0.5383<br>0.5328<br>0.5304<br>0.5162<br>0.5296<br>0.5178<br>0.5241<br>–<br>0.5335<br>0.5414<br>0.5485<br>0.5335<br>0.5509<br>0.5406<br>0.5154<br>0.5296<br>0.5043<br>0.5162<br>0.5241<br>0.5107|



26 

