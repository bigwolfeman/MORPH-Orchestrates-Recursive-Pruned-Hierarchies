# CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs

- **Authors:** Haoran Li, Sucheng Ren, Alan Yuille, Feng Wang
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2602.05258
- **MORPH uses:** Soft cosine-taper attenuation of low-frequency RoPE components whose wavelength exceeds the training context length, eliminating out-of-distribution position outliers and enabling smooth extrapolation to contexts up to 256k tokens without fine-tuning.

---

# **CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

**Haoran Li**[1 2] **Sucheng Ren**[2] **Alan Yuille**[2] **Feng Wang**[2] 

## **Abstract** 

Rotary Positional Embedding (RoPE) is a key component of context scaling in Large Language Models (LLMs). While various methods have been proposed to adapt RoPE to longer contexts, their guiding principles generally fall into two categories: (1) _out-of-distribution (OOD) mitigation_ , which scales RoPE frequencies to accommodate unseen positions, and (2) _Semantic Modeling_ , which posits that the attention scores computed with RoPE should always prioritize semantically similar tokens. In this work, we unify these seemingly distinct objectives through a minimalist intervention, namely CoPE: soft clipping lowfrequency components of RoPE. CoPE not only eliminates OOD outliers and refines semantic signals, but also prevents spectral leakage caused by hard clipping. Extensive experiments demonstrate that simply applying our soft clipping strategy to RoPE yields significant performance gains that scale up to 256k context length, validating our theoretical analysis and establishing CoPE as a new state-of-the-art for length generalization. Our code, data, and models are available at https://github.com/hrlics/CoPE. 

## **1. Introduction** 

Long context Large Language Models (LLMs) have become a cornerstone of critical domains such as coding agents (Jimenez et al., 2024; Anthropic, 2025), agentic memory (Yu et al., 2025; Chhikara et al., 2025), and long-horizon reasoning (Qiao et al., 2025; Zhou et al., 2025; Sinha et al., 2025). To achieve context scaling, a long context training stage is often required after initial pre-training, where the frequencies within Rotary Positional Embedding (RoPE) (Su et al., 2024) are modified to fit the target context length, followed by continued training on long sequences. 

> 1Carnegie Mellon University 2Johns Hopkins University. Correspondence to: Haoran Li _<_ haoranl4@cs.cmu.edu _>_ , Feng Wang _<_ fwang60@jh.edu _>_ . 

**==> picture [228 x 122] intentionally omitted <==**

**----- Start of picture text -----**<br>
58.11 59.60 61.23 58.68<br>60<br>52.72<br>55.74 56.86 55.70<br>52.94<br>40 44.29<br>28.48<br>20<br>CoPE RoPE<br>14.37<br>8k 16k 32k 64k 128k 256k<br>Performance<br>2×<br>**----- End of picture text -----**<br>


_Figure 1._ **Performance comparison between CoPE and RoPE.** With a simple _soft_ clipping strategy, CoPE effectively improves RoPE’s performance both within the training range and during extrapolation. The training context length here is 64k. 

While existing works have proposed various methods to adapt RoPE to longer contexts, their guiding principles generally fall into two categories: _**OOD mitigation**_ and _**semantic modeling**_ . Specifically, RoPE divides the query and key vectors into two-dimensional chunks, and rotates each chunk at a specific frequency. For low-frequency components that do not complete a full rotation during pre-training, extrapolating to unseen positions leads to severe OOD issues. Therefore, several _**OOD mitigation**_ strategies, including Position Interpolation (PI) (Chen et al., 2023), NTK (bloc97, 2023), YaRN (Peng et al., 2024), and LongRoPE (Ding et al., 2024; Shang et al., 2025), are introduced to scale the frequencies so that extended contexts are mapped back to the original position range. In contrast, another line of research is inspired by _**semantic modeling**_ , which posits that the attention scores computed with RoPE should always prioritize semantically similar tokens. Men et al. (2024) show that the rotation matrix in attention would degrade the model’s ability to discriminate relevant tokens from irrelevant ones as the relative distance increases, motivating the use of a higher base frequency. The ABF technique (Xiong et al., 2024) arrives at the same strategy, claiming that increasing the base frequency mitigates the long-term decay in RoPE and improves long context modeling. 

Despite their improved performance, these two lines of research are typically treated as tackling distinct aspects of long context modeling. However, we argue that they stem 

_Preprint. February 6, 2026._ 

1 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

from the same underlying issue: _**the suboptimal behavior of low-frequency components in the extrapolation regime.**_ Through theoretical analysis of RoPE’s frequency spectrum, we show that low-frequency components simultaneously govern OOD behavior under extrapolation and the stability of semantic attention over long contexts. Motivated by this insight, we propose CoPE, a minimalist intervention that softly clips the low-frequency components of RoPE. This simple and effective strategy not only suppresses OOD outliers and refines semantic signals, but also prevents spectral leakage caused by hard clipping, providing a plug-andplay solution that can be seamlessly integrated into existing LLMs for better long context capability. 

To validate the effectiveness and compatibility of CoPE, we conduct extensive experiments that align with the long context recipe used in Qwen3 (Yang et al., 2025), i.e., employing the ABF technique (Xiong et al., 2024) during longcontext training and YaRN for test-time extrapolation. By simply replacing the standard RoPE with CoPE while keeping all other configurations unchanged, we observe consistent and significant improvements across diverse tasks and context lengths, as shown in Figure 1. Notably, at context lengths up to 256k tokens, CoPE achieves nearly **2** _×_ the performance of RoPE, while also maintaining superior performance within the training range. Together, our theoretical analysis and empirical results establish CoPE as a simple, general, and highly effective drop-in replacement for RoPE in long context LLMs. 

Our main contributions can be summarized as follows: 

- We provide a unified perspective on long context adaptations of RoPE, showing that both OOD mitigation and semantic modeling methods originate from the suboptimal behavior of low-frequency components in the extrapolation regime. 

- Based on this insight, we propose **CoPE** , a minimalist and principled modification to RoPE that softly attenuates low-frequency components, eliminating OOD outliers, refining semantic signals, and preventing the spectral leakage induced by hard clipping. 

- We conduct extensive experiments to demonstrate that CoPE is a _simple_ and _scalable_ drop-in replacement for RoPE, consistently improving performance across diverse tasks and context lengths up to 256k. 

## **2. Preliminaries** 

**Rotary Position Embedding (RoPE).** Transformer-based models (Vaswani et al., 2017) rely on Positional Encodings (PEs) to explicitly incorporate sequential information. Among various PEs, Rotary Position Embedding (RoPE) (Su et al., 2024) has become the dominant choice in modern LLMs. Let **x** _i ∈_ R _[d]_ denote the _d_ -dimensional token 

embedding of the _i_ -th token in a sequence. Consider the _n_ -th query vector **q** _n_ and the _m_ -th key vector **k** _m_ , RoPE partitions the dimensions into _d/_ 2 chunks, e.g., **q** _n_ = [ **q** _n_[(0)][;] **[ q]**[(1)] _n_[;] _[ . . .]_[ ;] **[ q]**[(] _n[d/]_[2] _[−]_[1)] ]. Each chunk is assigned a unique rotation frequency _θi_ = _b[−]_[2] _[i/d] , i ∈{_ 0 _,_ 1 _, . . . , d/_ 2 _−_ 1 _}_ , where _b_ is a pre-defined base frequency (typically set to 10 _,_ 000). The rotation is achieved through a rotation matrix **R** _n ∈_ R _[d][×][d]_ , which can be formulated as follows: 

**==> picture [234 x 57] intentionally omitted <==**

With this block-diagonal rotation matrix, the attention score[1] between **q** _n_ and **k** _m_ is computed as: 

**==> picture [236 x 34] intentionally omitted <==**

## **3. Analysis** 

In this section, we conduct a comprehensive theoretical analysis of existing methods that adapt RoPE to longer contexts. We begin by highlighting the underlying guiding principles of prior methods, namely _**OOD mitigation**_ and _**semantic modeling**_ . We then show that these two seemingly distinct objectives both originate from the same root cause: the suboptimal behavior of low-frequency components in the extrapolation regime. 

## **3.1. RoPE OOD Theory** 

**Background.** Recall that RoPE divides the query and key vectors into 2-dimensional chunks and rotates each chunk at a frequency of _θi_ = _b[−]_[2] _[i/d] , i ∈{_ 0 _,_ 1 _, . . . , d/_ 2 _−_ 1 _}_ , where _b_ is the base frequency and is usually set to 10 _,_ 000. Given the periodicity of sinusoidal functions, we know that for each chunk with frequency _θi_ , the corresponding period can be calculated as follows: 

**==> picture [138 x 23] intentionally omitted <==**

Since _θi_ decreases as the dimensional index _i_ increases, the low-frequency components in higher dimensions possess longer periods, potentially exceeding the pre-training context window. For example, the pre-training context window of Llama-3-8B (Grattafiori et al., 2024) is 8192, while the period of the 35-th chunk already slightly exceeds this length. Consequently, out of the 64 chunks, the last 29 lowfrequency chunks fail to experience a single complete period during the pre-training stage, leading to severe OOD issues 

> 1Here, we omit the softmax function and 1 _/_ ~~_√_~~ _d_ scaling in standard Transformer (Vaswani et al., 2017) for simplicity. 

2 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

**==> picture [454 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [0]<br>1.0<br>Clipping Onset<br>10 [2]<br>0.5<br>0.0 10 [4]<br>RoPE<br>-0.5 NoPE<br>Hard Clipping<br>CoPE (Ours)<br>-1.0<br>0 10<br>2000<br>4000<br>6000<br>8000 25<br>10000 12000 30 0 10 20 30 40 50 60<br>35<br>RoPE Dimension Index<br>(a)  RoPE Frequencies and OOD Issue. (b)  Spectral Comparison.<br>Context Length<br>RoPE Dimension<br>Frequency (Log Scale)<br>**----- End of picture text -----**<br>


_Figure 2._ **(a) Visualization of RoPE frequencies.** Low-frequency components in higher dimensions possess longer periods. The region shaded in red marks where the period exceeds the pre-training context window, leading to OOD extrapolation. **(b) Spectral comparison.** Unlike RoPE which keeps unstable low frequencies (blue), or Hard Clipping which causes an abrupt cut-off and spectral leakage, CoPE implements a soft decay strategy starting from the clipping onset, simultaneously eliminating OOD outliers and refines semantic signals. 

during extrapolation. In contrast, high-frequency components in lower dimensions complete multiple cycles during pre-training and remain well-behaved even in extrapolation. 

**Critical Dimension in Extrapolation.** Based on the above spectrum analysis and prior work (Liu et al., 2024; Shang et al., 2025), we formally define the critical dimension in RoPE-based extrapolation as follows: 

**Definition 3.1** (Critical Dimensions in Extrapolation) **.** For LLMs pre-trained with context window _Lpre_ , attention head dimension _d_ , and base frequency _b_ , only the first _dct_ dimensions perceive complete periodic patterns during pre-training: 

**==> picture [155 x 22] intentionally omitted <==**

As shown in Figure 2, for Llama-3-8B with _Lpre_ = 8192, _d_ = 128, _b_ = 500 _,_ 000, the critical dimension is 70, which corresponds to the 35-th rotation chunk as discussed earlier. 

**OOD Mitigation Methods.** To mitigate the OOD behavior of RoPE beyond the critical dimension, several methods have been proposed to scale the frequencies _θi_ so that extended contexts are mapped back to the original position range. For ease of notation, we denote the target context length as _Lt_ and the scaling factor for each frequency _θi_ as _si_ . Given the scaling factor, the scaled frequency can be calculated as: 

**==> picture [164 x 24] intentionally omitted <==**

Representative works include PI (Chen et al., 2023), NTK 

(bloc97, 2023), YaRN (Peng et al., 2024), and LongRoPE (Ding et al., 2024; Shang et al., 2025). _PI_ applies a uniform scaling factor across all RoPE frequencies, i.e., _si_ = _LLpret_[.] While easy to implement, this approach equally stretches all dimensions without considering the distinct behaviors of high- and low-frequency components of RoPE during extrapolation. As a result, it compresses high-frequency components, leading to a loss of local positional resolution. Inspired by the Neural Tangent Kernel (NTK) theory (Tancik et al., 2020), which states that neural networks have difficulties learning high-frequency features, _NTK_ proposes to scale high frequencies less and low frequencies more with the scaling factor _si_ = ( _L[L] pre[t]_[)][2] _[i/]_[(] _[d][−]_[2)][, effectively al-] leviating the loss of high-frequency information. Building on NTK, _YaRN_ further partitions the frequencies into three groups and applies the following strategy: no scaling for high-frequency components ( _si_ = 1), PI-style scaling for low-frequency components ( _si_ = _LLpret_[),][and][linear][inter-] polation between 1 and _LLpret_[for intermediate frequencies.] _LongRoPE_ adopts a perplexity-guided search-based method to estimate the optimal scaling factor _si_ for each frequency. 

**Takeaway 1.** OOD mitigation methods address extrapolation by interpolating low-frequency components, while minimally perturbing high frequencies. The primary distinction among these methods lies in their choice of per-frequency scaling factors. 

## **3.2. RoPE Semantic Modeling** 

**Background.** When RoPE was originally proposed (Su et al., 2024), it introduced an important inductive bias known as _**long-term decay**_ : the upper bound of the attention score 

3 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

**==> picture [189 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 RoPE<br>CoPE<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0.2<br>0 5000 10000 15000 20000 25000 30000<br>Relative Distance<br>)i<br>t<br>cos(<br>**----- End of picture text -----**<br>


_Figure 3._ **Long-term decay of semantic attention.** As relative distance increases, the model’s ability to prefer semantically similar tokens over random ones diminishes. Applying soft clipping to the low-frequency components (CoPE) effectively alleviates this decay, preserving semantic information over long contexts. 

between two tokens decreases as their relative distance increases. This property encourages each token to attend more to its neighbors. However, Men et al. (2024) observe that an undesirable decay property also exists: the ability to attend more to semantically similar tokens than random tokens also decays as the relative distance increases. Following Men et al. (2024), we denote this property as _**long-term decay of semantic attention**_ and formalize it as follows: 

**Theorem 3.2** (Long-term Decay of Semantic Attention) **.** _Assume query_ **q** _∈_ R _[d] and key_ **k** _∈_ R _[d] have distance_ ∆ _t and i.i.d. components with standard deviation σ. Let_ **k** _[′]_ = **q** + _ϵ denote a similar key to the query, where ϵ is a zero-mean perturbation. Then, we have:_ 

**==> picture [217 x 56] intentionally omitted <==**

The proof is provided in Appendix A.1. Note that the term _d/_ 2 _−_ 1 � _i_ =0 cos(∆ _tθi_ ) should ideally be greater than zero to ensure more attention is paid to similar tokens than random ones. However, this term does decrease as ∆ _t_ increases, as shown in Figure 3. Given this observation, Men et al. (2024) propose to use a higher base frequency _b_ , which in turn decreases _θi_ = _b[−]_[2] _[i/d]_ and alleviates this undesirable decay. Similarly, the ABF technique (Xiong et al., 2024) arrives at the same higher base frequency strategy, claiming that increasing the base frequency reduces the general long-term decay of RoPE and improves long context modeling. Given its simplicity and effectiveness, the higher base frequency strategy has been widely adopted in long context training (Grattafiori et al., 2024; Yang et al., 2025). More recently, several work has analyzed how different RoPE frequencies 

influence attention patterns, concluding that low-frequency components primarily carry semantic information (Barbero et al., 2025; Jin et al., 2025), as they are the most invariant to token relative distance. 

**Takeaway 2.** RoPE secretly induces a long-term decay of semantic attention that is primarily governed by the low-frequency components, revealing them as an unreliable semantic channel. 

## **3.3. All Roads Lead to Low-Frequency Components** 

Our analysis above reveals a unifying insight: both _OOD extrapolation_ and _long-term decay of semantic attention_ stem from the same root cause: _**the suboptimal behavior of low-frequency components in the extrapolation regime.**_ Specifically, from the OOD perspective, low-frequency components possess periods exceeding the pre-training context window, resulting in OOD extrapolation. Meanwhile, from the semantic modeling perspective, low frequencies serve as the semantic channel that distinguishes similar tokens from random ones, yet this ability decays as context length increases. Our unified perspective suggests a simple yet effective design principle: stabilizing the behavior of lowfrequency components is sufficient to mitigate OOD extrapolation and preserve long-range semantic attention. 

**Takeaway 3.** OOD extrapolation and long-term semantic decay are two manifestations of the **same** underlying issue: the suboptimal behavior of low-frequency components beyond the pre-training regime. 

## **4. CoPE: Clipped Rotary Position Embedding** 

Motivated by our analysis in Section 3, we propose Clipped Rotary Position Embedding ( **CoPE** ), a simple yet effective method that softly clips the low-frequency components of RoPE, as illustrated in Figure 2b. CoPE not only eliminates OOD outliers and refines semantic signals, but also prevents severe spectrum leakage induced by hard clipping, thereby scaling favorably with increased context window. 

## **4.1. Spectral Analysis** 

To stabilize low-frequency components, a straightforward approach is to directly set them to zero, i.e., hard clipping. For example, Babero et al. (2025) identify the low frequencies as the semantic channel and propose to stabilize them by clipping the lowest 25% or 75% frequencies, resulting in lower validation perplexity on a 2B-scale model with 8k context length. However, hard clipping introduces an abrupt spectral cutoff, which can distort the remaining frequency components and undermine the stability of positional information, particularly in long-context scenarios. To elabo- 

4 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

**==> picture [189 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 RoPE<br>Hard Clipping<br>CoPE (ours)<br>0.8<br>Ringing Artifacts<br>0.6<br>0.4<br>0.2<br>0 100 200 300 400 500<br>Relative Distance<br>Attention Score<br>**----- End of picture text -----**<br>


_Figure 4._ **Ringing artifacts caused by hard clipping.** Directly applying a hard clipping to the low-frequency components introduces an abrupt spectral cutoff, which causes spectral leakage and manifests as long-range oscillatory ringing in the attention signal (Gibbs phenomenon). 

rate, we first reframe the attention mechanism with RoPE through the lens of Non-Uniform Discrete Fourier Transform (NUDFT). As shown in Equation 2, the dot-product attention between the _n_ -th query vector **q** _n_ and the _m_ -th key vector **k** _n_ is calculated as: 

**==> picture [210 x 13] intentionally omitted <==**

which can be further transformed into 

**==> picture [228 x 42] intentionally omitted <==**

where _τ_ = _m − n_ denotes the relative distance. This formulation reveals that the attention score computed with RoPE achieves an inverse NUDFT with frequency components _θj_ = _b[−]_[2] _[j/d] , j ∈_ [0 _, d/_ 2). Now, we analyze the impact of hard clipping using a continuous approximation of _A_ ( _τ_ ) in the large- _d_ limit, which provides clearer theoretical insight. 

**Theorem 4.1** (Spectral Leakage from Hard Clipping) **.** _Let A_ ( _τ_ ) _be the continuous attention score. Hard highpass filter at cutoff θc yields a A_[˜] ( _τ_ ) = _A_ ( _τ_ ) + _E_ ( _τ_ ) _, where the error term is:_ 

**==> picture [185 x 25] intentionally omitted <==**

_The slow O_ (1 _/τ_ ) _decay of the sinc kernel introduces Gibbs oscillations, disrupting the general decay of A_ ( _τ_ ) _and inducing spurious long-range correlations._ 

The proof is provided in Appendix A.2. Theorem 4.1 shows that the slowly decaying _O_ (1 _/τ_ ) envelope of the sinc kernel 

is a direct consequence of the sharp spectral discontinuity introduced by hard clipping. As a result, the attention scores exhibit Gibbs ringing, where oscillatory artifacts disrupt the general monotonicity of decay and cause spurious longrange correlations, as illustrated in Figure 4. 

## **4.2. Soft Clipping Strategy** 

To address the above challenge, our CoPE introduces a soft clipping strategy, which applies a smooth spectral taper (e.g., a cosine window) to the low frequencies. By Fourier duality, this soft clipping yields a rapidly decaying kernel in the time domain, suppressing unstable low-frequency components without inducing long-range spurious correlations. 

Specifically, instead of applying a binary mask **1** _θ>θc_ , we assign a scalar weight _wj ∈_ [0 _,_ 1] to each frequency component _θj_ . To minimize spectral discontinuity, we employ a cosine-decay taper. The weights _wj_ are defined as a function of the frequency _θj_ : 

**==> picture [231 x 37] intentionally omitted <==**

where _θ_ start denotes the clipping onset and _θ_ min is the lowest frequency. This strategy is highly practical as it allows for seamless integration into modern LLM frameworks. By simply modifying the initialization of the RoPE frequency, CoPE can be applied as a drop-in replacement without altering the model architecture. This ensures full compatibility with optimized inference kernels, such as FlashAttention (Dao, 2024), while maintaining standard inference speeds. 

## **5. Experiment** 

In this section, we evaluate CoPE across various benchmarks to answer the following questions: **(1)** Does CoPE consistently outperform RoPE and the hard clipping strategy on real-world long context tasks? **(2)** Are synthetic benchmarks reliable proxies for real-world performance? **(3)** Can CoPE retain performance on short context benchmarks that assess general model capabilities? **(4)** How does the choice of clipping onset affect performance? 

## **5.1. Experimental Setups** 

**Evaluation Benchmarks.** For long context evaluation, we primarily utilize the HELMET benchmark (Yen et al., 2025), which improves upon purely synthetic benchmarks (e.g., RULER (Hsieh et al., 2024)) and benchmarks with limited real-world tasks (e.g., InfiniteBench (Zhang et al., 2024)), providing a more robust and realistic assessment. HELMET includes both synthetic recall and a diverse set of real-world tasks, including retrieval-augmented gener- 

5 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

_Table 1._ **Main results on HELMET benchmark across diverse real-world tasks.** Models are trained with 64k context length and evaluated up to 256k to assess length generalization. CoPE consistently outperforms RoPE and hard clipping, with performance gains scaling favorably with context length. The best results are **bold** , while “–” indicates unavailable benchmark data at that context length. 

|**Task**|**Method**|**Context Length**<br>**8k**<br>**16k**<br>**32k**<br>**64k**<br>**128k**<br>**256k**|
|---|---|---|
||||
|**Summ.**|RoPE<br>HardClip|29.18<br>28.46<br>21.76<br>11.10<br>6.31<br>9.06<br>25.68<br>25.70<br>18.55<br>6.93<br>9.33<br>8.60|
||**CoPE**|**29.76**<br>**30.81**<br>**32.78**<br>**30.88**<br>**27.89**<br>**32.37**|
||||
|**QA**|RoPE<br>HardClip|6.46<br>8.39<br>8.52<br>7.67<br>8.21<br>7.93<br>7.44<br>9.28<br>10.16<br>10.31<br>9.31<br>9.24|
||**CoPE**|**13.10**<br>**16.89**<br>**21.02**<br>**15.07**<br>**18.23**<br>**19.06**|
||||
|**ICL**|RoPE<br>HardClip|74.60<br>80.20<br>83.40<br>85.50<br>82.10<br>–<br>73.10<br>77.00<br>79.80<br>82.20<br>77.30<br>–|
||**CoPE**|**79.40**<br>**83.70**<br>**85.50**<br>**86.40**<br>**84.70**<br>–|
||||
|**Recall**|RoPE<br>HardClip|99.75<br>99.13<br>98.13<br>97.63<br>71.38<br>26.13<br>99.75<br>98.50<br>98.50<br>94.38<br>82.13<br>36.86|
||**CoPE**|99.63<br>98.88<br>**99.00**<br>**97.88**<br>76.00<br>34.00|
||||
|**RAG**|RoPE<br>HardClip|68.38<br>67.44<br>66.67<br>62.78<br>53.44<br>–<br>68.06<br>67.50<br>66.11<br>59.72<br>62.05<br>–|
||**CoPE**|**68.67**<br>**67.72**<br>**67.83**<br>**63.17**<br>56.78<br>–|
||||
|**Average**|RoPE<br>HardClip|55.74<br>56.86<br>55.70<br>52.94<br>44.29<br>14.37<br>54.81<br>55.60<br>54.62<br>50.71<br>48.02<br>18.23|
||**CoPE**|**58.11**<br>**59.60**<br>**61.23**<br>**58.68**<br>**52.72**<br>**28.48**|



ation (RAG), many-shot in-context learning (ICL), longdocument QA, and summarization. We also report results on synthetic tasks from RULER and InfiniteBench. For standard short context benchmarks, we adopt MMLU (Hendrycks et al., 2021), MMLU-Pro (Wang et al., 2024), GPQA (Rein et al., 2024), BIG-Bench Hard (Suzgun et al., 2022), and GSM8K (Cobbe et al., 2021). For more detailed benchmark descriptions, please refer to Appendix B.1. 

**Long Context Training Stage.** We employ Llama-3-8B (Grattafiori et al., 2024) as the backbone model, which is pre-trained with an 8k context window. We extend the models’ context length to 64k via continued pre-training on ProLong data (20B tokens) (Gao et al., 2025), followed by SFT on UltraChat (1B tokens) (Ding et al., 2023). Following Qwen3 and ProLong (Yang et al., 2025; Gao et al., 2025), we increase the base frequency from 5 _×_ 10[5] to 1 _×_ 10[7] using the ABF technique (Xiong et al., 2024). 

**Baselines.** We compare CoPE with the widely-used RoPE (Su et al., 2024) and a hard clipping strategy that directly sets some low frequencies to zero (Barbero et al., 2025). 

**Implementation Details.** For both continued pre-training and SFT, we adopt a batch size of 256 (16M tokens) and the AdamW optimizer (Loshchilov & Hutter, 2017) with a weight decay of 0 _._ 1 and ( _β_ 1 _, β_ 2) = (0 _._ 9 _,_ 0 _._ 95). Both stages are trained for one epoch, differing only in their learning 

rate schedules. Specifically, continued pre-training uses an initial learning rate of 1 _×_ 10 _[−]_[5] with a 10% warmup and cosine decay to 1 _×_ 10 _[−]_[6] , while SFT uses an initial learning rate of 2 _×_ 10 _[−]_[5] with a 5% warmup and cosine decay to 2 _×_ 10 _[−]_[6] . The clipping onset is set to 44 (64 frequencies in total). For evaluations beyond 64k, we leverage YaRN (Peng et al., 2024) with a scaling factor of 4. The training process takes approximately 1996 and 48 GPU hours on machines equipped with H100-80GB GPUs, respectively. 

## **5.2. Main Results** 

We evaluate CoPE across a diverse set of tasks, covering synthetic recall, RAG, ICL, QA, and summarization. The results are detailed in Table 1. 

**Performance on HELMET.** As shown in Table 1, CoPE consistently outperforms RoPE and HardClip across nearly all tasks and context lengths. Within the training range (64k), CoPE yields an average improvement of 10 _._ 84% over RoPE, indicating that soft clipping does not compromise in-distribution performance. When extrapolated to 256k context, CoPE achieves approximately 2 _×_ the performance of RoPE, demonstrating superior length generalization ability. In contrast, although the hard clipping strategy slightly improves performance at extreme context lengths (128k256k), it exhibits noticeable degradation within the training 

6 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

_Table 2._ **Performance comparison on synthetic tasks** sampled from InfiniteBench and RULER, which provide limited insights into real-world performance. 

|**Task**|**Method**|**Context Length**<br>**8k**<br>**16k**<br>**32k**<br>**64k**<br>**128k**<br>**256k**|
|---|---|---|
||||
|**RULER NIAH**|RoPE<br>HardClip|100.0<br>100.0<br>98.67<br>99.67<br>91.00<br>60.50<br>100.0<br>100.0<br>97.00<br>99.33<br>89.33<br>57.00|
||**CoPE**|**100.0**<br>**100.0**<br>**98.67**<br>**99.67**<br>**94.00**<br>**78.50**|
||||
|**RULER MK**|RoPE<br>HardClip|100.0<br>100.0<br>99.00<br>96.00<br>54.33<br>27.00<br>100.0<br>100.0<br>98.00<br>89.33<br>63.33<br>17.67|
||**CoPE**|**100.0**<br>**100.0**<br>98.67<br>**99.67**<br>59.67<br>**32.33**|
||||
|**InfBench KV**|RoPE<br>HardClip|6.20<br>12.80<br>24.60<br>24.20<br>16.40<br>16.40<br>6.20<br>12.80<br>24.80<br>11.20<br>21.40<br>21.40|
||**CoPE**|**6.20**<br>**12.80**<br>**25.40**<br>**31.40**<br>19.40<br>19.40|
||||
|**InfBench Math Find**|RoPE<br>HardClip|37.14<br>35.00<br>36.86<br>33.42<br>35.14<br>35.14<br>34.86<br>35.67<br>35.71<br>26.29<br>34.57<br>34.57|
||**CoPE**|35.43<br>**35.71**<br>36.00<br>**34.00**<br>**35.14**<br>**35.14**|



range (8k-64k). This behavior empirically validates our theoretical analysis in Theorem 4.1, which highlights that abrupt hard truncation would cause spectral leakage and introduce spurious correlations. Together, these results establish CoPE as a plug-and-play enhancement for vanilla RoPE in long context LLMs, effectively mitigating OOD outliers, refining long-range semantic signals, and preventing spectral leakage induced by hard clipping. 

**Scalable Performance Gain of CoPE.** Beyond higher _absolute_ performance, CoPE exhibits performance gains that scale favorably with increasing context length. In particular, the average performance gain is roughly 4 _._ 54% at shorter contexts (8-16k), increases to 10 _._ 39% within the training range (32k–64k), and further scales to 58 _._ 61% under longcontext extrapolation (128k–256k). This trend shows that soft clipping effectively suppresses unstable low-frequency behaviors that become pronounced as the context grows. 

## **5.3. Limitations of Synthetic Tasks** 

While synthetic recall tasks are widely adopted for long context evaluation, we find that they provide limited insights into real-world performance, as shown in Table 2. 

**Saturation Issue.** Many synthetic tasks quickly saturate within the training range, making them ineffective for distinguishing model capabilities. For example, RULER-NIAH and RULER-MK achieve near-perfect accuracy for all methods at 8k-64k context lengths, despite significant performance gaps on real-world tasks, as shown in Table 1. 

**Limited Discriminative Power.** Some synthetic tasks exhibit hardly distinguishable performance across methods by design. For example, on InfiniteBench KV, all methods achieve nearly identical accuracy at 8k-32k contexts, mak- 

_Table 3._ **Performance on standard benchmarks** that measure general model capabilities. Despite clipped low frequencies, CoPE preserves performance and even yields slight gains. 

|Method|MMLU|MMLU-Pro|GPQA|BBH|GSM8K|
|---|---|---|---|---|---|
|RoPE|62.22|33.52|28.75|64.47|52.38|
|HardClip|62.35|33.95|29.67|64.48|52.59|
|**CoPE**|**62.37**|**34.05**|29.31|**64.51**|52.46|



ing the task uninformative for comparing model capabilities. 

**Length Invariance.** Furthermore, some other synthetic tasks demonstrate insensitivity to context length. For instance, InfiniteBench Math Find is a variant of multiple numerical lookup and exhibits only minor performance differences across context lengths, i.e., maintaining _∼_ 35% accuracy from 8k to 256k context for all methods. 

Overall, synthetic tasks either saturate early or fail to capture meaningful distinctions between models, rendering them poor proxies for real-world long context performance. This observation aligns with prior findings (Gao et al., 2025) and motivates our adoption of the HELMET benchmark. 

## **5.4. Results on Standard Short Context Benchmarks** 

To verify that CoPE’s soft clipping strategy does not compromise general model capabilities, we evaluate it on a suite of standard short context benchmarks. As shown in Table 3, CoPE preserves performance and even yields slight gains on all benchmarks, which serve as proxies for broad reasoning and knowledge. The fact that CoPE does not trade off these capabilities indicates that soft clipping primarily suppresses _the suboptimal behavior of low-frequency components_ , rather than erasing semantically useful signal. These results, together with CoPE’s consistent gains across context 

7 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

_Table 4._ **Ablation results on HELMET.** While CoPE remains robust to the choice of clipping onset, we find that preserving some stable low frequencies generally yields better performance. CoPE-29 denotes softly clipping the last 29 frequencies, whose periods are longer than the pre-training context window. 

|Method<br>**CoPE**|8k<br>**58.11**|16k<br>**59.60**|32k<br>**61.23**|64k<br>**58.68**|128k<br>**52.72**|256k<br>**28.48**|
|---|---|---|---|---|---|---|
|RoPE|55.74|56.86|55.70|52.94|44.29|14.37|
|CoPE-29|55.92|57.15|58.02|56.28|49.71|21.76|
|CoPE-34|57.09|59.46|59.55|57.54|49.33|19.07|



lengths (Table 1), support our central claim: _soft_ clipping is a drop-in enhancement of RoPE that delivers consistent performance gains across tasks and context lengths. 

## **5.5. Ablation Study** 

To understand how the choice of clipping onset impacts performance, we conduct an ablation study by varying the number of frequencies that are softly clipped in CoPE. The results are summarized in Table 4. 

Specifically, we consider two variants, CoPE-29 and CoPE34, which softly clip a larger portion of the low-frequency components compared to the default configuration (CoPE20). In CoPE-29, all frequencies whose periods exceed the pre-training context window are clipped, while CoPE-34 further removes part of the moderately low-frequency band. 

According to Table 4, we observe that: (1) CoPE remains robust to the choice of clipping onset, with all variants outperforming vanilla RoPE across different context lengths. (2) The default CoPE configuration, which clips _∼_ 75% of the low frequencies, consistently yields the best performance, indicating that low-frequency suppression, while effective, should avoid being overly aggressive. 

## **6. Related Work** 

RoPE is widely adopted in modern LLMs and is deeply coupled with their length generalization ability. To enable context extension, prior work has proposed various modifications to RoPE. In this work, we highlight that their underlying guiding principles can be generally categorized into two classes: _OOD mitigation_ and _semantic modeling_ . 

**RoPE OOD Mitigation.** The low-frequency components in RoPE possess periods longer than the pre-training context window, which will lead to severe OOD issues during extrapolation. To mitigate this, a line of work has investigated different methods to scale RoPE frequencies so that extended contexts are mapped back to the original training range, including PI (Chen et al., 2023), NTK (bloc97, 2023), YaRN (Peng et al., 2024), and LongRoPE (Ding et al., 2024; Shang et al., 2025). As discussed in Section 3.1, these methods 

differ primarily in their choice of per-frequency scaling factors, and the key technique is to interpolate low frequencies while minimizing the impact on high frequencies, which have completed multiple cycles during pre-training. 

**RoPE Semantic Modeling.** Meanwhile, another line of work has investigated how the semantic information is carried within RoPE. As discussed in Section 3.2, Men et al. (2024) observe that besides the general decay of activations, RoPE also introduces an undesirable decay property: the ability to attend more to semantically similar tokens than random ones decays as the relative distance increases, which we refer to as _long-term decay of semantic attention_ . To alleviate this decay, they propose a higher base frequency strategy, which is also introduced in the ABF technique (Xiong et al., 2024). More recently, several studies analyze the attention patterns within different RoPE frequencies, revealing that low-frequency components primarily carry semantic information, as they are the most invariant to token relative distance (Barbero et al., 2025; Jin et al., 2025). 

In our work, we unify these seemingly diverging objectives and argue that they stem from the same issue: _the suboptimal behavior of low-frequency components in the extrapolation regime._ This is inspired by the fact that the low-frequency components are responsible for OOD extrapolation, while simultaneously serving as an unreliable semantic channel whose discriminative power decays with increasing relative distance. Given this insight, we propose a minimalist and principled enhancement, termed CoPE, which softly clips the low-frequency components of RoPE to suppress OOD outliers and refine long-range semantic signals. Importantly, softly clipping prevents spectral leakage induced by hard frequency truncation (Barbero et al., 2025), which can introduce ringing artifacts and spurious correlations. 

## **7. Conclusion** 

In this paper, we present a unified perspective on long context adaptations of RoPE. We first highlight that existing methods can be categorized into two paradigms: OOD mitigation and semantic modeling. Then, we point out that these two seemingly distinct objectives originate from the same issue: _the suboptimal behavior of low-frequency components in the extrapolation regime._ Motivated by this insight, we introduce CoPE, a plug-and-play enhancement for RoPE that softly clips the low-frequency components. CoPE not only suppresses OOD outliers and refines long-range semantic signals, but also avoids spectral leakage induced by hard frequency truncation. Extensive experiments on a diverse set of real-world tasks demonstrate that CoPE consistently outperforms RoPE and the hard clipping strategy across context lengths of up to 256k, confirming its effectiveness and moving beyond prior perplexity-based metrics, synthetic recall benchmarks, and short context evaluation. 

8 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## **References** 

Anthropic. Claude code, 2025. URL https://www. claude.com/product/claude-code. 

- Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., and Velickoviˇ c,´ P. Round and round we go! what makes rotary positional encodings useful? In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https://openreview.net/ forum?id=GtvuNrk58a. 

- bloc97. Ntk-aware scaled rope allows llama models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation, 2023. URL https://www.reddit.com/r/LocalLLaMA/ comments/14lz7j5/ntkaware_scaled_ rope_allows_llama_models_to_have/. 

- Chen, S., Wong, S., Chen, L., and Tian, Y. Extending context window of large language models via positional interpolation. _arXiv preprint arXiv:2306.15595_ , 2023. 

- Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. Mem0: Building production-ready ai agents with scalable long-term memory. _arXiv preprint arXiv:2504.19413_ , 2025. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International Conference on Learning Representations (ICLR)_ , 2024. 

- Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M., and Zhou, B. Enhancing chat language models by scaling high-quality instructional conversations, 2023. 

- Ding, Y., Zhang, L. L., Zhang, C., Xu, Y., Shang, N., Xu, J., Yang, F., and Yang, M. LongroPE: Extending LLM context window beyond 2 million tokens. In _Fortyfirst International Conference on Machine Learning_ , 2024. URL https://openreview.net/forum? id=ONOtpXLqqw. 

- Gao, T., Wettig, A., Yen, H., and Chen, D. How to train long-context language models (effectively). In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 7376–7399, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-2510. doi: 10.18653/v1/2025.acl-long.366. URL https: //aclanthology.org/2025.acl-long.366/. 

- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021. 

- Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., and Ginsburg, B. RULER: What’s the real context size of your long-context language models? In _First Conference on Language Modeling_ , 2024. URL https: //openreview.net/forum?id=kIoBbc76Sy. 

- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. R. SWE-bench: Can language models resolve real-world github issues? In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum? id=VTF8yNQM66. 

- Jin, M., Mei, K., Xu, W., Sun, M., Tang, R., Du, M., Liu, Z., and Zhang, Y. Massive values in self-attention modules are the key to contextual knowledge understanding. In _Forty-second International Conference on Machine Learning_ , 2025. URL https://openreview.net/ forum?id=1SMcxxQiSL. 

- Li, H., Qin, Y., Ou, B., Xu, L., and Xu, R. HoPE: Hybrid of position embedding for long context vision-language models. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_ , 2025. URL https: //openreview.net/forum?id=6TmLco2L2D. 

- Liu, X., Yan, H., An, C., Qiu, X., and Lin, D. Scaling laws of roPE-based extrapolation. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum? id=JO7k0SJ5V6. 

- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- Men, X., Xu, M., Wang, B., Zhang, Q., Lin, H., Han, X., and Chen, W. Base of rope bounds context length. _arXiv preprint arXiv:2405.14591_ , 2024. 

9 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

- Peng, B., Quesnelle, J., Fan, H., and Shippole, E. YaRN: Efficient context window extension of large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview. net/forum?id=wHBfxhZu1u. 

- Qiao, Z., Chen, G., Chen, X., Yu, D., Yin, W., Wang, X., Zhang, Z., Li, B., Yin, H., Li, K., et al. Webresearcher: Unleashing unbounded reasoning capability in long-horizon agents. _arXiv preprint arXiv:2509.13309_ , 2025. 

- Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. GPQA: A graduate-level google-proof q&a benchmark. In _First Conference on Language Modeling_ , 2024. URL https: //openreview.net/forum?id=Ti67584b98. 

- Shang, N., Zhang, L. L., Wang, S., Zhang, G., Lopez, G., Yang, F., Chen, W., and Yang, M. LongroPE2: Near-lossless LLM context window scaling. In _Fortysecond International Conference on Machine Learning_ , 2025. URL https://openreview.net/forum? id=jwMjzGpzi4. 

- Sinha, A., Arun, A., Goel, S., Staab, S., and Geiping, J. The illusion of diminishing returns: Measuring long horizon execution in llms. _arXiv preprint arXiv:2509.09677_ , 2025. 

- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_ , 568:127063, 2024. 

- Suzgun, M., Scales, N., Scharli,¨ N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., , and Wei, J. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ , 2022. 

- Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., and Ng, R. Fourier features let networks learn high frequency functions in low dimensional domains. _Advances in neural information processing systems_ , 33:7537–7547, 2020. 

   - Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B., Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., Wang, S., and Ma, H. Effective long-context scaling of foundation models. In Duh, K., Gomez, H., and Bethard, S. (eds.), _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 4643–4663, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long. 260. URL https://aclanthology.org/2024. naacl-long.260/. 

   - Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ , 2025. 

   - Yen, H., Gao, T., Hou, M., Ding, K., Fleischer, D., Izsak, P., Wasserblat, M., and Chen, D. Helmet: How to evaluate long-context language models effectively and thoroughly. In _International Conference on Learning Representations (ICLR)_ , 2025. 

   - Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.-Q., Ma, W.-Y., Liu, J., Wang, M., et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. _arXiv preprint arXiv:2507.02259_ , 2025. 

   - Zhang, X., Chen, Y., Hu, S., Xu, Z., Chen, J., Hao, M., Han, X., Thai, Z., Wang, S., Liu, Z., and Sun, M. _∞_ Bench: Extending long context evaluation beyond 100K tokens. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 15262–15277, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https: //aclanthology.org/2024.acl-long.814. 

   - Zhou, Z., Qu, A., Wu, Z., Kim, S., Prakash, A., Rus, D., Zhao, J., Low, B. K. H., and Liang, P. P. Mem1: Learning to synergize memory and reasoning for efficient longhorizon agents. _arXiv preprint arXiv:2506.15841_ , 2025. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. _arXiv preprint arXiv:2406.01574_ , 2024. 

10 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

## **A. Proofs** 

In this section, we provide detailed proofs for the theoretical statements presented in this paper. 

## **A.1. Long-term Decay of Semantic Attention** 

As discussed in Theorem 3.2, RoPE secretly induces a long-term decay of semantic attention, where the ability to attend more to semantically similar tokens than random ones decays as the relative distance increases. Here, we provide the derivation used in Equation 3.2. 

**Theorem 3.2 (** Long-term Decay of Semantic Attention **).** _Assume query_ **q** _∈_ R _[d] and key_ **k** _∈_ R _[d] have distance_ ∆ _t and i.i.d. components with standard deviation σ. Let_ **k** _[′]_ = **q** + _ϵ denote a similar key to the query, where ϵ is a zero-mean perturbation. Then, we have:_ 

**==> picture [344 x 32] intentionally omitted <==**

**==> picture [202 x 14] intentionally omitted <==**

_Proof._ 

**==> picture [429 x 195] intentionally omitted <==**

where _µ_ denotes the mean of the i.i.d. components in **q** and **k** . The term[�] _[d/] i_ =0[2] _[−]_[1] cos(∆ _tθi_ ) is oscillatory and thus not monotonic in ∆ _t_ , but exhibits a general decay as ∆ _t_ increases, as shown in Figure 3. 

## **A.2. Spectral Leakage from Hard Clipping** 

**Theorem 4.1 (** Spectral Leakage from Hard Clipping **).** _Let A_ ( _τ_ ) _be the continuous attention score. Hard high-pass filter at cutoff θc yields a A_[˜] ( _τ_ ) = _A_ ( _τ_ ) + _E_ ( _τ_ ) _, where the error term is:_ 

**==> picture [311 x 25] intentionally omitted <==**

_The slow O_ (1 _/τ_ ) _decay of the sinc kernel introduces Gibbs oscillations, disrupting the general decay of A_ ( _τ_ ) _and inducing spurious long-range correlations._ 

_Proof._ Let _F_ denote the Fourier transform and _F[−]_[1] the inverse Fourier transform. We define the operation of hard clipping at cutoff frequency _θc_ as applying an ideal high-pass filter, _H_ high( _ω_ ), in the frequency domain. This filter can be expressed 

11 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

as the complement of an ideal low-pass filter (rectangular window), _H_ low( _ω_ ): 

**==> picture [367 x 12] intentionally omitted <==**

Let _A_[ˆ] ( _ω_ ) = _F_ [ _A_ ( _τ_ )] be the spectrum of the continuous attention score. The spectrum of the filtered signal, _A_[ˆ˜] ( _ω_ ), is given by the element-wise product: 

**==> picture [312 x 49] intentionally omitted <==**

By the Convolution Theorem, multiplication in the frequency domain corresponds to convolution in the time domain. Applying the inverse Fourier transform _F[−]_[1] to both sides yields: 

**==> picture [336 x 14] intentionally omitted <==**

The inverse Fourier transform of the rectangular function _H_ low( _ω_ ) with cutoff _θc_ is the normalized sinc function: 

**==> picture [319 x 25] intentionally omitted <==**

Substituting this kernel back into the time-domain equation, we identify the error term _E_ ( _τ_ ) = _A_[˜] ( _τ_ ) _− A_ ( _τ_ ) as: 

**==> picture [321 x 25] intentionally omitted <==**

This concludes the derivation. The impulse response of the ideal low-pass filter is a sinc function, which decays asymptotically as _O_ (1 _/τ_ ). This slow decay manifests as Gibbs oscillations (ringing artifacts) in the time domain, disrupting the general decay of _A_ ( _τ_ ) and inducing suprious long-range correlations. This negative effect is also illustrated in Figure 4. 

## **B. Further Experimental Details** 

In this section, we provide further details of our experiments, including benchmark descriptions, additional results, and a case study. 

## **B.1. Benchmark Description** 

In this subsection, we provide detailed descriptions of the long context benchmarks we used in the experiments, including HELMET (Yen et al., 2025), RULER (Hsieh et al., 2024), and Infinite Bench (Zhang et al., 2024). 

- **HELMET** is a comprehensive benchmark for evaluating long context LLMs on real-world tasks, improving upon purely synthetic benchmarks (e.g., RULER) and benchmarks with limited real-world tasks (e.g., Infinite Bench). Specifically, HELMET comprises summarization, long-document QA, many-shot in-context learning (ICL), synthetic recall, retrieval-augmented generation (RAG), generation with citations, and passage re-ranking. Following ProLong (Gao et al., 2025), we select the five most representative tasks for evaluation. 

- **RULER** is a purely synthetic benchmark for long context evaluation, which expands upon the vanilla needle-in-ahaystack (NIAH) test to incorporate variations with diverse types and quantities of needles, resulting in a total of 13 synthetic tasks. However, as shown in recent work (Yen et al., 2025; Gao et al., 2025; Zhang et al., 2024; Shang et al., 2025) and our Section 5.3, synthetic tasks either saturate quickly within the training range or provide limited signals for real-world performance, rendering them poor proxies for long context capabilities. 

- **Infinite Bench** is a benchmark designed to evaluate LLMs on extremely long-context understanding, consisting of both synthetic and real-world tasks with an average length of _∼_ 200k. Infinite Bench covers domains such as novel understanding, code execution, and mathematical calculation. Nevertheless, as discussed in Section 5.3, we find that some tasks exhibit limited discriminative power among different methods (KV Retrieval) or insensitivity to context length (Math Find), which motivates our use of the more realistic HELMET benchmark. 

12 

**CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs** 

_Table 5._ **Performance comparison on the RULER benchmark.** The results are averaged across 13 tasks. 

|Method|8k|16k|32k|64k|128k|256k|
|---|---|---|---|---|---|---|
|RoPE|80.52|82.33|82.11|76.93|61.19|28.86|
|**CoPE**|**81.50**|**82.84**|**82.75**|76.71|**61.95**|**46.86**|



## **B.2. Additional Results** 

We report the quantitative results of CoPE and RoPE on the RULER benchmark in Table 5. We observe that, except under extremely long contexts (256k), where CoPE achieves a substantial improvement (up to +18 _._ 0), most RULER tasks exhibit limited discriminative power between different methods. In contrast, on real-world tasks from the HELMET benchmark, such as RAG, in-context learning, and long-form summarization, CoPE consistently yields significant performance gains, as shown in Table 1. These results suggest that synthetic recall benchmarks may fail to fully reflect practical long context capabilities, highlighting the importance of evaluating different methods on realistic downstream tasks. 

## **B.3. Case Study** 

Table 6 presents some long-document QA examples with RoPE and CoPE. We observe that RoPE exhibits repetitive and less informative responses under long-context settings, often missing fine-grained details, whereas CoPE produces more coherent and detail-preserving answers. 

_Table 6._ **Long-document QA examples with RoPE and CoPE.** 

|**Q:**Viola wonders about the fate of which animals<br>in Central Park?<br>**A:**The ducks in the lagoon.<br>**Input Length:** 256k|**CoPE:**Viola wonders about the fate of theducks<br>in the lagoonin Central Park.<br>**RoPE:**Viola wonders about the fate of theducks<br>[missing key details]in Central Park. I’m not sure<br>if he ever found out what happened to the ducks<br>in the winter, but he was worried about it. [non-<br>informative]|
|---|---|
|||
|**Q:**What does Raul call his speakeasy?<br>**A:**The Zebra Room.<br>**Input Length:** 256k|**CoPE:**The Zebra Room.<br>**RoPE:**The Zebra Room. ”Why did you have to<br>paint a church?” ”Why not?” ”Because I promised<br>to fx it up.” ”Why did you promise to fx it?” [non-<br>informative]|
|||
|**Q:**In Bellamy’s vision of a perfect society, what<br>is needed for artistic or literary success?<br>**A:**Talent.<br>**Input Length:** 256k|**CoPE:**The ability to produce work that is of high<br>quality and that is appreciated by the people. The<br>nation provides opportunities for education and<br>development oftalent, and the people vote on the<br>acceptance of work in these...<br>**RoPE:**The ability to produce work that is of high<br>quality and that is appreciated by the people. The<br>people are the sole judges of artistic and literary<br>merit... [missing key details]|



13 

