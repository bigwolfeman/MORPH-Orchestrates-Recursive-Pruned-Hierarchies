Title: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

URL Source: https://arxiv.org/html/2507.07955

Published Time: Wed, 16 Jul 2025 00:34:33 GMT

Markdown Content:
\addbibresource

references.bib

Brandon Wang

Cartesia AI 

brandon.wang@cartesia.ai 

Albert Gu

Carnegie Mellon University, Cartesia AI 

agu@cs.cmu.edu, albert@cartesia.ai

###### Abstract

Major progress on language models (LMs) in recent years has largely resulted from moving away from specialized models designed for specific tasks, to general models based on powerful architectures (e.g. the Transformer) that learn everything from raw data. Despite this trend, pre-processing steps such as tokenization remain a barrier to true end-to-end foundation models. We introduce a collection of new techniques that enable a dynamic chunking mechanism which automatically learns content- and context- dependent segmentation strategies learned jointly with the rest of the model. Incorporating this into an explicit hierarchical network (H-Net) allows replacing the (implicitly hierarchical) tokenization–LM–detokenization pipeline with a single model learned fully end-to-end. When compute- and data- matched, an H-Net with one stage of hierarchy operating at the byte level outperforms a strong Transformer language model operating over BPE tokens. Iterating the hierarchy to multiple stages further increases its performance by modeling multiple levels of abstraction, demonstrating significantly better scaling with data and matching the token-based Transformer of twice its size. H-Nets pretrained on English show significantly increased character-level robustness, and qualitatively learn meaningful data-dependent chunking strategies without any heuristics or explicit supervision. Finally, the H-Net’s improvement over tokenized pipelines is further increased in languages and modalities with weaker tokenization heuristics, such as Chinese and code, or DNA sequences (nearly 4×4\times 4 × improvement in data efficiency over baselines), showing the potential of true end-to-end models that learn and scale better from unprocessed data.

1 Introduction
--------------

A broad goal of deep learning is to learn meaningful patterns from raw data, automatically extracting features and building abstractions in an end-to-end fashion. However, fixed-vocabulary tokenization, the process of compressing raw text into predefined chunks through algorithms such as byte-pair encoding (BPE)\parencite BPE,SentencePiece, remains a pervasive handcrafted preprocessing step in modern language models (LMs)\parencite Llama3, GPT3. Tokenization comes with a host of well-documented drawbacks, from poor character-level understanding to lack of meaning and interpretability to degraded performance on complex languages and modalities\parencite petrov2023language, ahia2023all, belinkov2017synthetic, Adv-BERT, ByT5, Canine. 1 1 1 Many other edge cases have been discussed in informal online discourse rather than papers; we defer to Andrej Karpathy’s [lectures](https://x.com/karpathy/status/1657949234535211009) and [tweets](https://x.com/karpathy/status/1759996551378940395). Replacing the tokenization–LM–detokenization pipeline with a single end-to-end model would also adhere better to the spirit of deep learning, ideally scaling more powerfully with data and parameters (c.f. _the bitter lesson_)\parencite sutton2019bitter,peric2025bitter. However, tokenization remains an indispensable component of language models and other sequential data for its ability to compress and shorten sequences; as of yet, no _end-to-end_ tokenizer-free model has matched the performance of tokenizer-based language models when matched for computational budget.

A line of recent works has turned to overcoming tokenization in autoregressive sequence models, which requires addressing a series of difficult technical challenges: 2 2 2 An extended related work can be found in [Appendix A](https://arxiv.org/html/2507.07955v2#A1 "Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), which is summarized in [Table 6](https://arxiv.org/html/2507.07955v2#S3.T6 "In Comparison to Mixture-of-Experts. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling").

*   •Direct byte-level language modeling with isotropic architectures 3 3 3 Non-hierarchical models comprised of repeated blocks, such as the standard Transformer\parencite Transformer. can be improved with efficient sequence models such as MambaByte\parencite MambaByte, but still incur prohibitive computational costs while underperforming tokenized models in compute-matched settings. 
*   •To improve efficiency, hierarchical architectures such as Hourglass Transformer\parencite HourglassTransformer and MegaByte\parencite MegaByte use small byte-level models to compress raw inputs into subsampled sequences, which are then processed with a more powerful standard language model. However, simple pooling strategies such as compressing every k 𝑘 k italic_k inputs are not data-dependent, and perform poorly on modalities with variable information rates such as language. 
*   •SpaceByte\parencite SpaceByte and Byte Latent Transformer\parencite BLT introduce data-dependent chunking strategies such as delimiter- or entropy-based heuristics. These heuristics, however, rely on auxiliary _external_ boundary predictors, and are therefore modality-specific and not fully end-to-end. 
*   •Although jointly trainable boundary predictors are the ideal solution, they require optimizing discrete selection operations without supervision, which is fundamentally a challenging problem. Consequently, existing end-to-end approaches\parencite DPT exhibit training instabilities that preclude scaling beyond small models or nesting multi-level hierarchies. 

Fundamentally, creating a tokenizer-free architecture requires incorporating the data chunking process directly into the model, while overcoming challenges in efficiency, learnability, and stability at scale.

#### Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization

In this work, we introduce an end-to-end hierarchical network (H-Net) that compresses raw data through a recursive, data-dependent dynamic chunking (DC) process ([Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). H-Nets match the efficiency of tokenized pipelines while substantially improving modeling ability, by replacing handcrafted heuristics with content-aware and context-dependent segmentation learned from data.

##### Hierarchical Processing.

The H-Net adopts the hierarchical architecture from prior work\parencite SaShiMi,HourglassTransformer,SpaceByte, resembling an autoregressive U-Net\parencite U-Net: (i) raw data is processed by a small encoder network, (ii) then downsampled and passed through a main network operating on compressed chunks, (iii) and finally upsampled before being passed through a decoder network operating on the original resolution. This modularity creates a natural processing hierarchy where outer stages capture fine-grained patterns while inner stages operate on coarse representations akin to traditional tokens. Crucially, while the main network contains the bulk of parameters and can be any standard architecture designed for operating on tokenized language—such as a Transformer\parencite Transformer or state space model (SSM)\parencite Mamba—we show that the encoder and decoder networks are strongly improved by using SSMs, which have an inductive bias for compression\parencite gu2025tradeoffs.

##### Dynamic Chunking.

H-Net’s core is a novel dynamic chunking (DC) mechanism which interfaces between the main network and the encoder/decoder networks, learning how to segment data while using standard differentiable optimization. DC is composed of two complementary new techniques: (i) a routing module which predicts boundaries between adjacent elements through a similarity score (ii) and a smoothing module which interpolates representations using the router’s outputs, attenuating the effect of uncertain boundaries and significantly improving learnability. By combining these with a new auxiliary loss function that targets desired downsampling ratios, and modern techniques for gradient-based learning of discrete choices\parencite MoE,STE, DC lets an H-Net learn how to compress data in a fully end-to-end fashion.

##### Signal Propagation.

We introduce several architectural and training techniques to improve stability and scalability during end-to-end optimization. These include: (i) carefully placing projections and normalization layers to balance signal propagation between interacting sub-networks, and (ii) adjusting optimization parameters for each layer based on its dimensionality and effective batch size, which changes between stages of the hierarchical structure.

Altogether, H-Net learns segmentation strategies _optimized jointly_ with the main backbone, dynamically compressing input vectors based on contextual information into meaningful chunks. H-Net represents the first truly end-to-end, tokenizer-free language model: with a single stage of dynamic chunking, a _byte-level H-Net_ matches the perplexity and downstream performance of a strong _BPE-tokenized Transformer_ at sizes exceeding 1B parameters. Empirically, the dynamic chunking module naturally compresses data to a similar resolution as BPE tokenizers (4.5-5 bytes/chunk) and qualitatively learns meaningful boundaries, all without any external supervision or heuristics.

![Image 1: Refer to caption](https://arxiv.org/html/2507.07955v2/x1.png)

Figure 1: (left) Architectural overview of H-Net with a two-stage hierarchical design (S=2 𝑆 2 S=2 italic_S = 2). (right) Dynamic Chunking (DC). (bottom-right) Key components of a chunking layer: (a) a routing module for dynamically drawing chunk boundaries, and (b) a downsampler that selectively retains vectors based on boundary indicators, reducing sequence length while preserving semantically significant positions. (top-right) Key components of a dechunking layer: (c) a smoothing module for converting discrete chunks into interpolated representations, and (d) an upsampler that restores compressed vectors to their original resolution based on boundary indicators. 𝖫𝗂𝗇𝖾𝖺𝗋 𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear}sansserif_Linear in equation ([3](https://arxiv.org/html/2507.07955v2#S2.E3 "Equation 3 ‣ 2.1.1 Components of H-Net ‣ 2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) and 𝖲𝖳𝖤 𝖲𝖳𝖤\mathsf{STE}sansserif_STE in equation ([9](https://arxiv.org/html/2507.07955v2#S2.E9 "Equation 9 ‣ Upsampler. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) are omitted in the illustration for brevity. 

#### Hierarchical Chunking: From Data to Abstractions

Beyond addressing tokenization, H-Net improves general sequence modeling across a wide range of settings. Subword tokenization in language models is a special case of _chunking_—the process of building higher-level abstractions from low-level data—and is a central component of intelligence.4 4 4[Chunking](https://en.wikipedia.org/wiki/Chunking_(psychology)) is a formal concept from cognitive psychology central to human memory and cognition, and is the inspiration for this work’s terminology. Crucially, because H-Net is fully end-to-end, it can be iterated recursively: the main network can itself be an H-Net. Intuitively, more stages of chunking represent higher order meanings; just as characters can be combined into words, words can be combined into clauses, sentences, and beyond. Iterating the hierarchy should therefore lead to even more efficient use of compute and parameters, and more effective reasoning over compressed representations.

Recursive H-Nets represent a new class of foundation model architectures that not only overcome tokenization, but discover and operate over abstractions learned from raw data, leading to higher-quality models with less pre-processing. Iterating the 1-stage H-Net to 2 hierarchical stages further improves its capabilities and strongly outperforms all baselines, with steeper training curves and better scaling with data. A byte-level 2-stage H-Net overtakes the perplexity of a strong tokenized Transformer after just 30B training bytes, with the gap widening throughout training, and matches the downstream evaluations of the tokenized Transformer of twice its size.

Finally, H-Nets realize the benefits of overcoming tokenization:

*   •Robustness. Without special data mixes, the pretrained H-Net is dramatically more robust to textual perturbations than the token-based Transformer, as evaluated on the noisy HellaSwag suite of benchmarks. 
*   •Interpretability. Qualitative visualizations of learned boundaries reveal that H-Net automatically discovers semantically coherent units without explicit supervision, validating that end-to-end learning successfully detects the structural patterns traditionally imposed through handcrafted tokenization. 
*   •Other languages. H-Net’s improvements are even more pronounced on languages without obvious segmentation cues, including Chinese and code (59.9→66.3→59.9 66.3 59.9\to 66.3 59.9 → 66.3 on XWinograd-zh compared to tokenized Transformer) and DNA language modeling (3.6×3.6\times 3.6 × improved data efficiency compared to isotropic models). 

2 H-Net Architecture
--------------------

H-Nets are defined as hierarchical U-Net-like networks, but with data-dependent _dynamic subsampling_ that is learned end-to-end together with the rest of the model. We first introduce H-Net’s hierarchical architecture for multi-level processing, establishing key design principles ([Section 2.1](https://arxiv.org/html/2507.07955v2#S2.SS1 "2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). We then present our dynamic chunking mechanism that learns content-aware compression through standard optimization ([Section 2.2](https://arxiv.org/html/2507.07955v2#S2.SS2 "2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). Next, we detail architectural and optimization enhancements specifically tailored for hierarchical sequence modeling ([Section 2.3](https://arxiv.org/html/2507.07955v2#S2.SS3 "2.3 Improved Techniques for Hierarchical Sequence Modeling ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). Finally, we explain how H-Net preserves autoregressive properties throughout its hierarchical structure during both training and inference ([Section 2.4](https://arxiv.org/html/2507.07955v2#S2.SS4 "2.4 Autoregressive Training and Inference ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

### 2.1 Architectural Overview

#### 2.1.1 Components of H-Net

H-Net employs a hierarchical architecture comprising three primary components – encoder networks (ℰ ℰ\mathcal{E}caligraphic_E), main network (ℳ ℳ\mathcal{M}caligraphic_M), and decoder networks (𝒟 𝒟\mathcal{D}caligraphic_D) – where each component is implemented with a stack of sequence mixing layers (_e.g.,_ Transformers or state space models). In its simplest form, a single-stage H-Net consists of one encoder network, one main network, and one decoder network. Crucially, the architecture’s key characteristic lies in the main network’s unique property: it can itself be instantiated as a complete H-Net, enabling recursive construction of multi-level hierarchies.

This recursive design allows H-Net to scale to arbitrary depths. In an S 𝑆 S italic_S-stage model, we denote components at each stage using superscripts: encoder networks as ℰ s superscript ℰ 𝑠\mathcal{E}^{s}caligraphic_E start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT and decoder networks as 𝒟 s superscript 𝒟 𝑠\mathcal{D}^{s}caligraphic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT for stages 0≤s<S 0 𝑠 𝑆 0\leq s<S 0 ≤ italic_s < italic_S, with the main network ℳ ℳ\mathcal{M}caligraphic_M residing only at the final stage s=S 𝑠 𝑆 s=S italic_s = italic_S. For example, a two-stage model contains ℰ 0 superscript ℰ 0\mathcal{E}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT, ℰ 1 superscript ℰ 1\mathcal{E}^{1}caligraphic_E start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT, ℳ ℳ\mathcal{M}caligraphic_M, 𝒟 1 superscript 𝒟 1\mathcal{D}^{1}caligraphic_D start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT, and 𝒟 0 superscript 𝒟 0\mathcal{D}^{0}caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT, as illustrated in [Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(Left). Throughout this paper, we use superscripts to denote stage indices, though we omit them when all variables within an equation belong to the same stage.

Drawing inspiration from the U-Net architecture\parencite U-Net, H-Net progressively compresses input sequences into fewer vectors with richer semantic embeddings through a chunking layer, processes these representations in the main network, then decompresses the sequence back to its original resolution using a dechunking layer. Unlike traditional U-Net designs, however, H-Net dynamically determines chunking boundaries rather than using fixed-size pooling operations. The overall pipeline can be formalized as:

x^s=ℰ s⁢(x s),z^S=ℳ⁢(x S),z^s=𝒟 s⁢(z s),formulae-sequence superscript^𝑥 𝑠 superscript ℰ 𝑠 superscript 𝑥 𝑠 formulae-sequence superscript^𝑧 𝑆 ℳ superscript 𝑥 𝑆 superscript^𝑧 𝑠 superscript 𝒟 𝑠 superscript 𝑧 𝑠\hat{x}^{s}=\mathcal{E}^{s}(x^{s}),\qquad\qquad\hat{z}^{S}=\mathcal{M}(x^{S}),% \qquad\qquad\hat{z}^{s}=\mathcal{D}^{s}(z^{s}),over^ start_ARG italic_x end_ARG start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT = caligraphic_E start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) , over^ start_ARG italic_z end_ARG start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT = caligraphic_M ( italic_x start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT ) , over^ start_ARG italic_z end_ARG start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT = caligraphic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ( italic_z start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) ,(1)

where the chunking layer and the dechunking layer operations are defined as:

(x s+1,p s)=𝖢𝗁𝗎𝗇𝗄⁢(x^s),superscript 𝑥 𝑠 1 superscript 𝑝 𝑠 𝖢𝗁𝗎𝗇𝗄 superscript^𝑥 𝑠(x^{s+1},p^{s})=\mathsf{Chunk}(\hat{x}^{s}),( italic_x start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT , italic_p start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) = sansserif_Chunk ( over^ start_ARG italic_x end_ARG start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) ,(2)

z s=𝖣𝖾𝖼𝗁𝗎𝗇𝗄⁢(z^s+1,p s)+𝖫𝗂𝗇𝖾𝖺𝗋⁢(x^s).superscript 𝑧 𝑠 𝖣𝖾𝖼𝗁𝗎𝗇𝗄 superscript^𝑧 𝑠 1 superscript 𝑝 𝑠 𝖫𝗂𝗇𝖾𝖺𝗋 superscript^𝑥 𝑠 z^{s}=\mathsf{Dechunk}(\hat{z}^{s+1},p^{s})+\mathsf{Linear}(\hat{x}^{s}).italic_z start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT = sansserif_Dechunk ( over^ start_ARG italic_z end_ARG start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT , italic_p start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) + sansserif_Linear ( over^ start_ARG italic_x end_ARG start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) .(3)

The initial input to the model is x 0∈ℝ L 0×D 0 superscript 𝑥 0 superscript ℝ superscript 𝐿 0 superscript 𝐷 0 x^{0}\in\mathbb{R}^{L^{0}\times D^{0}}italic_x start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT × italic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT where L 0 superscript 𝐿 0 L^{0}italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT is the input sequence length and D 0 superscript 𝐷 0 D^{0}italic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT is the embedding dimension. Intuitively, p s∈[0,1]L s superscript 𝑝 𝑠 superscript 0 1 superscript 𝐿 𝑠 p^{s}\in[0,1]^{L^{s}}italic_p start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ∈ [ 0 , 1 ] start_POSTSUPERSCRIPT italic_L start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT represents the chunking router’s confidence that the token should be passed into the main stage. 7 7 7 We also sometimes refer to it as a _probability_—it is interpreted as such in [Appendix F](https://arxiv.org/html/2507.07955v2#A6 "Appendix F Distilling Token-Level Models to Byte-Level ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")—although we do not use it as a formal probability. This value is essential for both the chunk ([Section 2.2.1](https://arxiv.org/html/2507.07955v2#S2.SS2.SSS1 "2.2.1 Chunking Layer ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) and dechunk operations ([Section 2.2.2](https://arxiv.org/html/2507.07955v2#S2.SS2.SSS2 "2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

#### 2.1.2 Design Principles

##### Encoder and Decoder Networks.

The encoder and decoder networks in H-Net face unique design constraints due to their dual objectives and computational requirements. Each encoder must simultaneously (i) preserve fine-grained information for transmission to its corresponding decoder through residual connections ([3](https://arxiv.org/html/2507.07955v2#S2.E3 "Equation 3 ‣ 2.1.1 Components of H-Net ‣ 2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), and (ii) compress inputs into chunks of richer representations for the main network. The decoder, in turn, must effectively combine coarse-grained representations from the main network with fine-grained details from the encoder residuals.

Importantly, both encoders and decoders operate on uncompressed sequences, making computational efficiency a significant design constraint that shapes our architectural choices. Recent studies demonstrate that state space models (SSMs)\parencite S4, Mamba excel at processing fine-grained data including audio\parencite SaShiMi, DNA sequences\parencite Caduceus, and robotic control signals\parencite ssm-rl.

Based on these insights, we employ Mamba-2 layers\parencite Mamba2 as the primary building blocks for the encoder and decoder networks. This choice yields two significant benefits: effective handling of fine-grained inputs, and substantially improved efficiency when processing long, uncompressed sequences. Our ablation studies ([Section 3.3](https://arxiv.org/html/2507.07955v2#S3.SS3 "3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) confirm that SSM-based encoders/decoders significantly outperform Transformer layers, not just at the byte level but even on coarser inputs, which we attribute to their stronger inductive bias for compression which helps build abstractions\parencite gu2025tradeoffs.

##### Main Network.

H-Net’s computational efficiency stems from strategic parameter allocation. We concentrate the majority of model capacity in the main network, which operates on progressively compressed sequences. After S 𝑆 S italic_S stages of compression, the main network receives sequences where L S≪L 0 much-less-than superscript 𝐿 𝑆 superscript 𝐿 0 L^{S}\ll L^{0}italic_L start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT ≪ italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT, enabling much larger networks within the same computational budget. This design reflects two key principles: (i) compressed sequences allow more parameters and compute per chunk, and (ii) higher-level abstractions benefit from increased processing power.

The main network functions as a standard language model and can employ any sequence mixing architecture. We default to Transformer layers for two reasons: compressed representations align well with Transformers’ strengths in processing discrete, semantically-rich tokens, and this choice enables more controlled comparison with traditional BPE-based Transformer baselines in our experiments. However, the modular design also allows straightforward substitution with alternative architectures (_e.g.,_ a state space model, hybrid, or H-Net itself) as explored in our ablations.

##### Architectural Guidelines.

Compared to standard isotropic models, the H-Net’s structure introduces several new dimensions of architectural parameters to balance the parameter/compute allocation to each network. To simplify the search space, we follow a few general guidelines.

*   •First, we ensure the model width (often referred to as d model subscript 𝑑 model d_{\text{model}}italic_d start_POSTSUBSCRIPT model end_POSTSUBSCRIPT for isotropic architectures) is monotone in the hierarchy: D 0≤D 1≤⋯≤D S superscript 𝐷 0 superscript 𝐷 1⋯superscript 𝐷 𝑆 D^{0}\leq D^{1}\leq\dots\leq D^{S}italic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ≤ italic_D start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT ≤ ⋯ ≤ italic_D start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT. This allows increasing compute and parameters used in the main network without significantly increasing its depth. 
*   •Second, using efficient and powerful SSM layers in the outer networks allow reducing the number of layers used compared to similar prior architectures that only used Transformer layers\parencite SpaceByte; in this paper, we always stick to four layers (or the equivalent of four Mamba layers) in each encoder/decoder network. 

To handle the changes in dimensions without an additional linear layer, we adopt the technique used in SpaceByte\parencite SpaceByte with the marginal change: to expand dimensions (_i.e.,_ D s→D s+1→superscript 𝐷 𝑠 superscript 𝐷 𝑠 1 D^{s}\rightarrow D^{s+1}italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT → italic_D start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT), we append all vectors with a shared trainable vector of dimension D s+1−D s superscript 𝐷 𝑠 1 superscript 𝐷 𝑠 D^{s+1}-D^{s}italic_D start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT - italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT; to reduce dimensions (_i.e.,_ D s+1→D s→superscript 𝐷 𝑠 1 superscript 𝐷 𝑠 D^{s+1}\rightarrow D^{s}italic_D start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT → italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT), we take the first D s superscript 𝐷 𝑠 D^{s}italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT dimensions from each vector.

We note that H-Net’s performance can likely be improved with more careful tuning of the layer allocation and hyperparameters between sub-networks.

### 2.2 Dynamic Chunking (DC)

H-Net learns chunking boundaries through end-to-end training, allowing it to identify semantically meaningful units adaptively. Furthermore, this dynamic approach enables the model to allocate computational resources efficiently by compressing low-information regions while preserving high-information content at appropriate granularity.

#### 2.2.1 Chunking Layer

The chunking layer (𝖢𝗁𝗎𝗇𝗄 𝖢𝗁𝗎𝗇𝗄\mathsf{Chunk}sansserif_Chunk in equation ([2](https://arxiv.org/html/2507.07955v2#S2.E2 "Equation 2 ‣ 2.1.1 Components of H-Net ‣ 2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"))) contains a routing module and downsampler, as illustrated in [Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(bottom-right).

##### Routing Module.

In natural data, meaningful boundaries tend to emerge at points of contextual or semantic shift. From this observation, we add an inductive bias by measuring the similarity between adjacent representations: when context changes, consecutive vectors should exhibit lower similarity. The routing module implements this intuition through cosine similarity between adjacent encoder outputs. Given encoder outputs X^^𝑋\hat{X}over^ start_ARG italic_X end_ARG, it calculates boundary probabilities p t subscript 𝑝 𝑡 p_{t}italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT and boundary indicators b t subscript 𝑏 𝑡 b_{t}italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT as follows:

q t=W q⁢x^t,k t=W k⁢x^t,p t=1 2⁢(1−q t⊤⁢k t−1‖q t‖⁢‖k t−1‖)∈[0,1],b t=𝟙{p t≥0.5},formulae-sequence formulae-sequence subscript 𝑞 𝑡 subscript 𝑊 𝑞 subscript^𝑥 𝑡 formulae-sequence subscript 𝑘 𝑡 subscript 𝑊 𝑘 subscript^𝑥 𝑡 subscript 𝑝 𝑡 1 2 1 superscript subscript 𝑞 𝑡 top subscript 𝑘 𝑡 1 norm subscript 𝑞 𝑡 norm subscript 𝑘 𝑡 1 0 1 subscript 𝑏 𝑡 subscript 1 subscript 𝑝 𝑡 0.5 q_{t}=W_{q}\hat{x}_{t},\quad k_{t}=W_{k}\hat{x}_{t},\qquad p_{t}=\frac{1}{2}% \left(1-\frac{q_{t}^{\top}k_{t-1}}{\left\|q_{t}\right\|\left\|k_{t-1}\right\|}% \right)\in[0,1],\quad b_{t}=\mathds{1}_{\{p_{t}\geq 0.5\}},italic_q start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_W start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT over^ start_ARG italic_x end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_W start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT over^ start_ARG italic_x end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = divide start_ARG 1 end_ARG start_ARG 2 end_ARG ( 1 - divide start_ARG italic_q start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT italic_k start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT end_ARG start_ARG ∥ italic_q start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∥ ∥ italic_k start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT ∥ end_ARG ) ∈ [ 0 , 1 ] , italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = blackboard_1 start_POSTSUBSCRIPT { italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ≥ 0.5 } end_POSTSUBSCRIPT ,(4)

where p 1=1.0 subscript 𝑝 1 1.0 p_{1}=1.0 italic_p start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 1.0 by definition, ensuring the sequence begins with a boundary. This formulation scales cosine similarity into a boundary score or probability: ideally, when consecutive vectors x^t−1 subscript^𝑥 𝑡 1\hat{x}_{t-1}over^ start_ARG italic_x end_ARG start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT and x^t subscript^𝑥 𝑡\hat{x}_{t}over^ start_ARG italic_x end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT span a semantic boundary (_e.g.,_ between morphemes, words, or phrases), their projections q t subscript 𝑞 𝑡 q_{t}italic_q start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT and k t−1 subscript 𝑘 𝑡 1 k_{t-1}italic_k start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT diverge in the latent space, yielding low cosine similarity and consequently high boundary probability p t subscript 𝑝 𝑡 p_{t}italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT.

##### Downsampler.

The downsampler compresses encoder outputs x^s superscript^𝑥 𝑠\hat{x}^{s}over^ start_ARG italic_x end_ARG start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT into a reduced set of vectors x s+1 superscript 𝑥 𝑠 1 x^{s+1}italic_x start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT using boundary indicators {b t s}t=1 L s superscript subscript superscript subscript 𝑏 𝑡 𝑠 𝑡 1 superscript 𝐿 𝑠\{b_{t}^{s}\}_{t=1}^{L^{s}}{ italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT } start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT. Among potential compression strategies – including mean pooling, max pooling, or cross-attention – we adopt direct selection of boundary-marked vectors for its simplicity and effectiveness (see [Section E.1](https://arxiv.org/html/2507.07955v2#A5.SS1 "E.1 Different Downsampling Methods in the Chunking Layer ‣ Appendix E Additional Ablation Studies ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") for ablations).

As illustrated in [Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(b), this approach follows a straightforward selection rule: vectors where b t=1 subscript 𝑏 𝑡 1 b_{t}=1 italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 1 are retained in the compressed sequence x s+1 superscript 𝑥 𝑠 1 x^{s+1}italic_x start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT, while those where b t=0 subscript 𝑏 𝑡 0 b_{t}=0 italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 0 are discarded. Likewise, the same downsampler applies to boundary probabilities, compressing p s superscript 𝑝 𝑠 p^{s}italic_p start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT into P s+1 superscript 𝑃 𝑠 1 P^{s+1}italic_P start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT for use in a dechunking layer (see [Section 2.2.2](https://arxiv.org/html/2507.07955v2#S2.SS2.SSS2 "2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

#### 2.2.2 Dechunking

The dechunking layer (𝖣𝖾𝖼𝗁𝗎𝗇𝗄 𝖣𝖾𝖼𝗁𝗎𝗇𝗄\mathsf{Dechunk}sansserif_Dechunk in equation ([3](https://arxiv.org/html/2507.07955v2#S2.E3 "Equation 3 ‣ 2.1.1 Components of H-Net ‣ 2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"))) consists of a smoothing module and upsampler, as illustrated in [Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(top-right).

##### Smoothing Module.

The critical challenge in training a dynamic chunking module lies in the discrete nature of chunk boundaries, which impedes gradient flow during backpropagation. We introduce the smoothing module as a technique to address this problem. As illustrated in [Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(c), this component transforms discrete chunking operations into differentiable computations by creating smooth interpolations between chunks. Concretely, the smoothing module applies an exponential moving average (EMA) with the following definition:

z¯t=P t⁢z^t+(1−P t)⁢z¯t−1.subscript¯𝑧 𝑡 subscript 𝑃 𝑡 subscript^𝑧 𝑡 1 subscript 𝑃 𝑡 subscript¯𝑧 𝑡 1\bar{z}_{t}=P_{t}\hat{z}_{t}+(1-P_{t})\bar{z}_{t-1}.over¯ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT over^ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + ( 1 - italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) over¯ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT .(5)

![Image 2: Refer to caption](https://arxiv.org/html/2507.07955v2/x2.png)

Figure 2:  Comparison of decompression strategies on the example sequence "...new product!". ●●\CIRCLE● indicates a boundary with high confidence (P t=1.0 subscript 𝑃 𝑡 1.0 P_{t}=1.0 italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 1.0) and ◐◐\LEFTcircle◐ indicates a boundary with low confidence (P t=0.5 subscript 𝑃 𝑡 0.5 P_{t}=0.5 italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 0.5). As each letter in the example is unique, we use the letters in subscripts to denote expected semantics of chunks. (a) Optimal chunking with oracle boundaries identifying linguistically meaningful units. (b) Suboptimal chunking without a smoothing module. This creates misalignment during upsampling, causing information from incorrect contexts to propagate. (c) Improved decompression with a smoothing module, where low-confidence chunks are interpolated with weighted combinations of previous chunks, correcting the shaded regions. In panels (b) and (c), we interpret low-confidence boundaries cause the encoder network to embed broader contexts at subsequent positions. Specifically, the vectors at _ and ! encode new_ and duct!, respectively (instead of w_ and ct!). 

Our smoothing module performs several roles:

*   •Differentiable boundary learning: It transforms the discrete upsampling operation into a continuous one, enabling effective backpropagation through chunk boundaries during training without requiring stochastic exploration-based approaches\parencite GumbelSoftmax. 
*   •Adaptive error correction: Chunks with high confidence (P t≈1.0 subscript 𝑃 𝑡 1.0 P_{t}\approx 1.0 italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ≈ 1.0) maintain discrete boundaries (z¯t≈z t subscript¯𝑧 𝑡 subscript 𝑧 𝑡\bar{z}_{t}\approx z_{t}over¯ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ≈ italic_z start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT), while chunks with low confidence (P t≈0.5 subscript 𝑃 𝑡 0.5 P_{t}\approx 0.5 italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ≈ 0.5) are smoothed using information from previous chunks, creating a self-correcting mechanism. 
*   •Training stability: By smoothly interpolating between discrete choices based on confidence scores, a smoothing module prevents the model from overfitting to suboptimal chunking patterns early in training. 

[Figure 2](https://arxiv.org/html/2507.07955v2#S2.F2 "In Smoothing Module. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") illustrates this with the example "...new product!". The word "product" can be morphologically decomposed into "pro-" and "-duct"8 8 8 pro- – meaning _forward_ or _forth_, -duct – from Latin _ducere_, meaning _to lead_ or _to bring_. Without the smoothing module (see [Figure 2](https://arxiv.org/html/2507.07955v2#S2.F2 "In Smoothing Module. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(b)), suboptimal chunking (_e.g.,_"du" as shown with half-filled circles) creates alignment mismatches that disrupt information flow. With the smoothing module (see [Figure 2](https://arxiv.org/html/2507.07955v2#S2.F2 "In Smoothing Module. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(c)), chunks with low confidence are smoothed with previous context, ensuring proper information propagation and enabling the model to learn optimal chunk boundaries through gradient descent.

##### Upsampler.

We carefully design the upsampler (see [Figure 1](https://arxiv.org/html/2507.07955v2#S1.F1 "In Signal Propagation. ‣ Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization ‣ 1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-(d)) that decompresses z¯s+1 superscript¯𝑧 𝑠 1\bar{z}^{s+1}over¯ start_ARG italic_z end_ARG start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT to match the original resolution of inputs in the previous stage z s superscript 𝑧 𝑠 z^{s}italic_z start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT with the following definition:

c t subscript 𝑐 𝑡\displaystyle c_{t}italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT=p t b t⁢(1−p t)1−b t={p t if⁢b t=1,1−p t otherwise,absent superscript subscript 𝑝 𝑡 subscript 𝑏 𝑡 superscript 1 subscript 𝑝 𝑡 1 subscript 𝑏 𝑡 cases subscript 𝑝 𝑡 if subscript 𝑏 𝑡 1 1 subscript 𝑝 𝑡 otherwise\displaystyle=p_{t}^{b_{t}}(1-p_{t})^{1-b_{t}}=\begin{cases}p_{t}&\text{if }b_% {t}=1,\\ 1-p_{t}&\text{otherwise},\end{cases}= italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_POSTSUPERSCRIPT ( 1 - italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT 1 - italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_POSTSUPERSCRIPT = { start_ROW start_CELL italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_CELL start_CELL if italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 1 , end_CELL end_ROW start_ROW start_CELL 1 - italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT end_CELL start_CELL otherwise , end_CELL end_ROW(6)
𝖲𝖳𝖤⁢(c t)𝖲𝖳𝖤 subscript 𝑐 𝑡\displaystyle\mathsf{STE}(c_{t})sansserif_STE ( italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT )=c t+stopgradient⁢(1−c t),absent subscript 𝑐 𝑡 stopgradient 1 subscript 𝑐 𝑡\displaystyle=c_{t}+\text{stopgradient}(1-c_{t}),= italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + stopgradient ( 1 - italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ,(7)

z~t subscript~𝑧 𝑡\displaystyle\tilde{z}_{t}over~ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT=z¯∑k=1 t b k,absent subscript¯𝑧 superscript subscript 𝑘 1 𝑡 subscript 𝑏 𝑘\displaystyle=\bar{z}_{\sum_{k=1}^{t}b_{k}},= over¯ start_ARG italic_z end_ARG start_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT italic_b start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUBSCRIPT ,(8)
𝖴𝗉𝗌𝖺𝗆𝗉𝗅𝖾𝗋⁢(z¯,c)t 𝖴𝗉𝗌𝖺𝗆𝗉𝗅𝖾𝗋 subscript¯𝑧 𝑐 𝑡\displaystyle\mathsf{Upsampler}(\bar{z},c)_{t}sansserif_Upsampler ( over¯ start_ARG italic_z end_ARG , italic_c ) start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT=𝖲𝖳𝖤⁢(c t)⋅z~t.absent⋅𝖲𝖳𝖤 subscript 𝑐 𝑡 subscript~𝑧 𝑡\displaystyle=\mathsf{STE}\left(c_{t}\right)\cdot\tilde{z}_{t}.= sansserif_STE ( italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ⋅ over~ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT .(9)

Each component serves a specific purpose in enabling stable end-to-end learning:

*   •Confidence scoring ([6](https://arxiv.org/html/2507.07955v2#S2.E6 "Equation 6 ‣ Upsampler. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")): The coefficient c 𝑐 c italic_c quantifies the routing module’s confidence in its boundary decisions. For positions marked as boundaries (b t=1 subscript 𝑏 𝑡 1 b_{t}=1 italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 1), c t=p t subscript 𝑐 𝑡 subscript 𝑝 𝑡 c_{t}=p_{t}italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT rewards high boundary probabilities. In contrast, for non-boundary positions (b t=0 subscript 𝑏 𝑡 0 b_{t}=0 italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 0), c t=1−p t subscript 𝑐 𝑡 1 subscript 𝑝 𝑡 c_{t}=1-p_{t}italic_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 1 - italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT penalizes false boundary predictions. This formulation encourages the model to produce boundary probabilities near 1.0 1.0 1.0 1.0 at true boundaries and near 0.0 0.0 0.0 0.0 elsewhere. 
*   •Gradient stabilization ([7](https://arxiv.org/html/2507.07955v2#S2.E7 "Equation 7 ‣ Upsampler. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")): The Straight-Through Estimator (STE)\parencite STE is a well established technique from discrete representation learning\parencite VQ-VAE, GumbelSoftmax that rounds confidence scores to 1.0 1.0 1.0 1.0 in the forward pass while maintaining continuous gradients during backpropagation. While H-Net already demonstrates strong performance without STE, incorporating this technique provides an additional performance boost that empirically further stabilizes the optimization dynamics. 
*   •Causal expansion ([8](https://arxiv.org/html/2507.07955v2#S2.E8 "Equation 8 ‣ Upsampler. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")): The upsampling operation repeats each compressed vector until the next boundary position, ensuring that each reconstructed position receives information from its most recent chunk. This maintains the sequential flow of information while expanding the compressed representation back to its original length. 
*   •Confidence-weighted decompression ([9](https://arxiv.org/html/2507.07955v2#S2.E9 "Equation 9 ‣ Upsampler. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")): Multiplying upsampled vectors by their confidence scores incentivizes the routing module to make confident, accurate decisions. High-confidence boundaries create direct reward signals that encourage the model to sharpen its boundary predictions through gradient feedback. 

#### 2.2.3 Ratio Loss

Without explicit regularization, the model may converge to trivial solutions: either retaining nearly all vectors (negating computational benefits) or compressing excessively (losing critical information). Inspired by load balancing mechanisms in Mixture-of-Experts (MoE) models\parencite MoE, which face similar challenges in maintaining balanced expert utilization, we introduce a ratio loss to guide compression:

ℒ ratio=N N−1⁢((N−1)⁢F⁢G+(1−F)⁢(1−G)),F=1 L⁢∑t=1 L b t,G=1 L⁢∑t=1 L p t,formulae-sequence subscript ℒ ratio 𝑁 𝑁 1 𝑁 1 𝐹 𝐺 1 𝐹 1 𝐺 formulae-sequence 𝐹 1 𝐿 superscript subscript 𝑡 1 𝐿 subscript 𝑏 𝑡 𝐺 1 𝐿 superscript subscript 𝑡 1 𝐿 subscript 𝑝 𝑡\mathcal{L}_{\text{ratio}}=\frac{N}{N-1}\left((N-1)FG+(1-F)(1-G)\right),\qquad F% =\frac{1}{L}\sum_{t=1}^{L}b_{t},\quad G=\frac{1}{L}\sum_{t=1}^{L}p_{t},caligraphic_L start_POSTSUBSCRIPT ratio end_POSTSUBSCRIPT = divide start_ARG italic_N end_ARG start_ARG italic_N - 1 end_ARG ( ( italic_N - 1 ) italic_F italic_G + ( 1 - italic_F ) ( 1 - italic_G ) ) , italic_F = divide start_ARG 1 end_ARG start_ARG italic_L end_ARG ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L end_POSTSUPERSCRIPT italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , italic_G = divide start_ARG 1 end_ARG start_ARG italic_L end_ARG ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ,(10)

where F 𝐹 F italic_F represents the fraction of vectors actually selected, G 𝐺 G italic_G denotes the average boundary probability, and N 𝑁 N italic_N controls the target compression ratio. Mechanistically, although F 𝐹 F italic_F is not differentiable, the network can be trained toward targeted compression ratios through G 𝐺 G italic_G, which provides continuous feedback.

When F=G 𝐹 𝐺 F=G italic_F = italic_G, the loss attains a minimum of ℒ ratio=1 subscript ℒ ratio 1\mathcal{L}_{\text{ratio}}=1 caligraphic_L start_POSTSUBSCRIPT ratio end_POSTSUBSCRIPT = 1 when F=G=1 N 𝐹 𝐺 1 𝑁 F=G=\frac{1}{N}italic_F = italic_G = divide start_ARG 1 end_ARG start_ARG italic_N end_ARG. Interestingly, the loss can theoretically fall below 1 1 1 1 when F≠G 𝐹 𝐺 F\neq G italic_F ≠ italic_G (_e.g.,_ F=1 N+ϵ 𝐹 1 𝑁 italic-ϵ F=\frac{1}{N}+\epsilon italic_F = divide start_ARG 1 end_ARG start_ARG italic_N end_ARG + italic_ϵ and G=1 N−ϵ 𝐺 1 𝑁 italic-ϵ G=\frac{1}{N}-\epsilon italic_G = divide start_ARG 1 end_ARG start_ARG italic_N end_ARG - italic_ϵ), which we indeed observe during training. Despite this theoretical possibility, the loss effectively guides the model toward the desired compression ratio in practice. In practice, as our architectural design encourages the routing module to make confident decisions (_i.e.,_ boundary probabilities approaching 0 0 or 1 1 1 1), F 𝐹 F italic_F naturally converges toward G 𝐺 G italic_G, and the loss effectively guides the model toward the desired compression ratio.

Combined together with the autoregressive prediction loss (_i.e.,_ ℒ=ℒ AR+α⁢∑s=0 S−1 ℒ ratio s ℒ subscript ℒ AR 𝛼 superscript subscript 𝑠 0 𝑆 1 superscript subscript ℒ ratio 𝑠\mathcal{L}=\mathcal{L}_{\text{AR}}+\alpha\sum_{s=0}^{S-1}\mathcal{L}_{\text{% ratio}}^{s}caligraphic_L = caligraphic_L start_POSTSUBSCRIPT AR end_POSTSUBSCRIPT + italic_α ∑ start_POSTSUBSCRIPT italic_s = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_S - 1 end_POSTSUPERSCRIPT caligraphic_L start_POSTSUBSCRIPT ratio end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT), this mechanism preserves content-adaptive compression: the model learns which vectors to retain based on semantic importance rather than following predetermined patterns, distinguishing H-Net from fixed compression schemes. We fixed α=0.03 𝛼 0.03\alpha=0.03 italic_α = 0.03 in all experiments in this paper as it provides a good balance between prediction accuracy and chunking efficiency; however, in other settings, it may be important to choose this hyperparameter more carefully.

Notationally, we sometimes use (N 0,N 1,…,N s)superscript 𝑁 0 superscript 𝑁 1…superscript 𝑁 𝑠(N^{0},N^{1},\dots,N^{s})( italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT , italic_N start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT , … , italic_N start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT )-DC to denote the full dynamic chunking mechanism together with its targeted chunking ratios.

### 2.3 Improved Techniques for Hierarchical Sequence Modeling

We introduce several techniques that improve the overall architecture. These may generally be considered techniques to improve _signal propagation_ throughout the network, improving stability and learnability.

##### Norm Balance.

Modern large language models employ pre-normalization architectures\parencite GPT2, Llama, departing from the post-normalization design of the original Transformer\parencite Transformer. Following established best practices, these models typically include a final normalization layer after all residual blocks. H-Net adopts this convention through _network normalization_, by placing an RMSNorm\parencite RMSNorm at the end of each network component (ℰ s superscript ℰ 𝑠\mathcal{E}^{s}caligraphic_E start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT, 𝒟 s superscript 𝒟 𝑠\mathcal{D}^{s}caligraphic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT, and ℳ ℳ\mathcal{M}caligraphic_M).

This addition of a normalization layer addresses a critical challenge in hierarchical architectures. Pre-normalization allows residual stream magnitudes to grow unbounded through successive layers, with feature norms increasing monotonically. For H-Net, this poses a particular problem: the architecture leverages residual connections to preserve fine-grained information across stages. Without network normalization, outputs from deeper components (especially the many-layered main network) would dominate the residual signals from earlier encoder networks through imbalanced feature norms, neglecting the fine-grained details that are essential for decompression. The normalization layers restore balance between processed features and residual information, ensuring both contribute meaningfully to the final representation.

##### Separation of Two Streams.

Encoder outputs (x^^𝑥\hat{x}over^ start_ARG italic_x end_ARG) serve dual purposes in our architecture: passing fine-grained information to corresponding decoders through residual connections, and providing compressed representations as inputs to subsequent stages. This dual functionality creates a design challenge, as these two roles may benefit from different representations. We consider three options to address this: (i) apply a projection to the residual connection only, (ii) apply a projection to the main network inputs only, (iii) and apply a projection to both pathways.

As indicated in equation ([3](https://arxiv.org/html/2507.07955v2#S2.E3 "Equation 3 ‣ 2.1.1 Components of H-Net ‣ 2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), we adopt the first approach – adding a projection (𝖫𝗂𝗇𝖾𝖺𝗋 𝖫𝗂𝗇𝖾𝖺𝗋\mathsf{Linear}sansserif_Linear) only to the residual connection. This choice is motivated by the fundamental principle of designing deep learning models\parencite ResNet: maintaining intact gradient flow through the main computational path is crucial for effective training.

Empirically, we found that the third option underperforms despite additional parameters and computations, as the extra projections interfere with gradient propagation. The second option, while preserving residual gradients, disrupts the main network’s gradient flow and had worse training dynamics. Our chosen design maintains unimpeded gradients from deeper stages while allowing the residual connection to adapt its contribution through the learned projection. This encourages the model to leverage the main network’s computational depth while using residuals in a complementary role.

One additional detail is that this residual connection is initialized close to 0; earlier versions of H-Net found this to be an important detail, but it may be less important when combined with additional techniques such as LR modulation.

##### Learning Rate Modulation

The hierarchical design of H-Net requires careful adjustment of learning rates across stages to ensure balanced training dynamics. Modern theory establishes that neural network hyperparameters should be scaled in predictable ways for optimal trainability\parencite yang2020feature. Concretely, outer stages, which handle significantly longer input sequences, receive proportionally higher learning rates than inner stages operating on compressed representations. This scaling follows established principles that learning rates are adjusted based on effective batch size and model dimensions. The specific scaling factor we use accounts for both the total number of inputs processed at each stage and the corresponding hidden dimensions (see [Appendix C](https://arxiv.org/html/2507.07955v2#A3 "Appendix C Learning Rate Modulation ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). 9 9 9 We later realized that SpaceByte also followed muP LR scaling\parencite yang2020feature to account for model dimension D s superscript 𝐷 𝑠 D^{s}italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT\parencite[Appendix B.2]SpaceByte (but did not account for batch size scaling, as we do). Our reimplementation did not do this and therefore is not fully faithful to the original SpaceByte. With this modulation, the model achieves more stable training dynamics and improved convergence behavior across the entire hierarchy. In particular, we empirically find that since outer stages directly influence the chunk boundaries that inner stages depend on, the higher learning rates in the outer stages seem to accelerate learning the chunking mechanism.

### 2.4 Autoregressive Training and Inference

Every component of H-Net (_i.e.,_ encoder-, decoder-, main- networks, and the dynamic chunking mechanism) is carefully designed to preserve autoregressive properties essential for language modeling.

##### Training.

During training, H-Net employs standard causal masking across all sequence mixing layers. DC maintains causality by computing boundary probabilities based only on current and previous representations. Specifically, the boundary probability p t subscript 𝑝 𝑡 p_{t}italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT depends on q t subscript 𝑞 𝑡 q_{t}italic_q start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT and k t subscript 𝑘 𝑡 k_{t}italic_k start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT from the current and previous positions (equation ([4](https://arxiv.org/html/2507.07955v2#S2.E4 "Equation 4 ‣ Routing Module. ‣ 2.2.1 Chunking Layer ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"))), ensuring no information leakage from future tokens. The smoothing module similarly maintains causality through its recursive formulation (equation ([5](https://arxiv.org/html/2507.07955v2#S2.E5 "Equation 5 ‣ Smoothing Module. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"))), where each output depends only on past compressed representations.

##### Inference.

For inference, H-Net generates raw bytes (or whatever the outermost modality is) autoregressively with a modified procedure to handle its hierarchical structure.

Generation with a prompt proceeds as follows:

1.   1.Initial processing: During prefill, we generate chunks via the encoders (as in training). For each component (i.e. the isotropic components, and the routing module and dechunking layer), we generate a state. Isotropic state (e.g. KV cache for Transformer layers, SSM state for Mamba-2 layers) is generated as usual. 
2.   2.

DC state and DC step: As noted above, the DC modules have recursive formulations that maintain causality at train-time. These recursive formulations become autoregressive formulations at inference time.

    1.   (a)Routing Module: In order to compute p t subscript 𝑝 𝑡 p_{t}italic_p start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, we need k t−1 subscript 𝑘 𝑡 1 k_{t-1}italic_k start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT (see equation ([4](https://arxiv.org/html/2507.07955v2#S2.E4 "Equation 4 ‣ Routing Module. ‣ 2.2.1 Chunking Layer ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"))), so our state consists of the key value of the most recent token processed. 
    2.   (b)Dechunking Layer: In order to compute z~t subscript~𝑧 𝑡\tilde{z}_{t}over~ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, we need P t subscript 𝑃 𝑡 P_{t}italic_P start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT and z~t−1 subscript~𝑧 𝑡 1\tilde{z}_{t-1}over~ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t - 1 end_POSTSUBSCRIPT. Thus, the dechunking layer state should consist of the last z~~𝑧\tilde{z}over~ start_ARG italic_z end_ARG value. 

3.   3.

Token Generation:10 10 10 Here, we use _token_ in the autoregressive generation sense, referring to one time step, not in the literal BPE token sense. To perform a model step, we do the following for a 1-stage hierarchy:

    1.   (a)Pass the token through the encoder network, 
    2.   (b)Step the routing module to determine whether the token needs to be processed by the main network, 
    3.   (c)Step the main network if necessary, in which case we also need to step the dechunking layer. 
    4.   (d)Use the result of the dechunking layer to step the decoder network. 

A consequence of this inference formulation is that, at inference time, H-Net decides individually for each token how much compute to use when processing it. Therefore, H-Net can allocate more or less compute to different tokens as it deems necessary. A particular connection is that inference resembles speculative decoding\parencite SpeculativeDecoding, SpeculativeSampling, which also involves a small network (the _draft model_) stepping on every token, and a larger network (the _verification model_) only stepping on contiguous chunks of every few tokens.

3 Experiments
-------------

Table 1: Architectures for main language models, all data-/FLOP-matched.ℰ 0,𝒟 0 superscript ℰ 0 superscript 𝒟 0\mathcal{E}^{0},\mathcal{D}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT , caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT, ℰ 1,𝒟 1 superscript ℰ 1 superscript 𝒟 1\mathcal{E}^{1},\mathcal{D}^{1}caligraphic_E start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT , caligraphic_D start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT, ℳ ℳ\mathcal{M}caligraphic_M. T and M denote a T ransformer and a M amba-2 layer, respectively. For hierarchical byte-level models, the Tokenizer column lists the chunking mechanism. The numbers before DC indicate downsampling factor N 𝑁 N italic_N in equation ([10](https://arxiv.org/html/2507.07955v2#S2.E10 "Equation 10 ‣ 2.2.3 Ratio Loss ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")); for example, (3,3)-DC denotes N 0=N 1=3 superscript 𝑁 0 superscript 𝑁 1 3 N^{0}=N^{1}=3 italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = italic_N start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT = 3. The BPIC (Bytes-Per-Innermost-Chunk) measure shows that each chunk dynamically determined by our 1-stage comprises similar number of bytes to the GPT-2 tokenizer, despite aiming for N 0=6 superscript 𝑁 0 6 N^{0}=6 italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = 6. All Transformer layers in ℰ ℰ\mathcal{E}caligraphic_E or 𝒟 𝒟\mathcal{D}caligraphic_D networks, as well as LlamaByte, use Sliding Window Attention (SWA) with a window size of 1024 1024 1024 1024. 

Model Input Tokenizer L 0 superscript 𝐿 0 L^{0}italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT BPIC(L S/L 0 superscript 𝐿 𝑆 superscript 𝐿 0 L^{S}/L^{0}italic_L start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT / italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT)#Params Architecture d_model (D)
#FLOPs matched to GPT-3 Large
Transformer Token GPT2 1792 4.6 760M T24 1536 1536 1536 1536
LlamaByte—1.0 210M T16 1024 1024 1024 1024
MambaByte—1.0 190M M28 1024 1024 1024 1024
SpaceByte Spacelike 6.0 570M T8+T16+T8 768 768 768 768, 1536 1536 1536 1536
SpaceByte++Spacelike 6.0 850M M4+T28+M4 1024 1024 1024 1024, 1536 1536 1536 1536
H-Net (pool)Byte 6-Pool 8192 6.0 850M M4+T28+M4 1024 1024 1024 1024, 1536 1536 1536 1536
H-Net (space)Spacelike 6.0 850M M4+T28+M4 1024 1024 1024 1024, 1536 1536 1536 1536
H-Net (1-stage)6-DC 4.8 680M M4+T22+M4 1024 1024 1024 1024, 1536 1536 1536 1536
H-Net (2-stage)(3,3)-DC 7.0 870M M4+T1M4+T26+M4T1+M4 1024 1024 1024 1024, 1024 1024 1024 1024, 1536 1536 1536 1536
#FLOPs matched to GPT-3 XL
Transformer Token GPT2 1792 4.6 1.3B T24 2048 2048 2048 2048
SpaceByte++Spacelike 6.0 1.6B M4+T31+M4 1024 1024 1024 1024, 2048 2048 2048 2048
H-Net (space)Spacelike 6.0 1.6B M4+T31+M4 1024 1024 1024 1024, 2048 2048 2048 2048
H-Net (1-stage)Byte 6-DC 8192 4.7 1.3B M4+T24+M4 1024 1024 1024 1024, 2048 2048 2048 2048
H-Net (2-stage)(3,3)-DC 6.9 1.6B M4+T1M4+T27+M4T1+M4 1024 1024 1024 1024, 1536 1536 1536 1536, 2048 2048 2048 2048

We first describe our general experimental protocol for language modeling, used for the majority of our experiments. In [Section 3.1](https://arxiv.org/html/2507.07955v2#S3.SS1 "3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we evaluate on a high-quality English dataset, showing significantly stronger performance than baselines, as well as improved robustness and interpretability from avoiding tokenization. In [Section 3.2](https://arxiv.org/html/2507.07955v2#S3.SS2 "3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we extend our evaluation to diverse datasets including Chinese, code, and DNA, with even larger performance improvements, demonstrating H-Net’s versatility as a general sequence model architecture. In [Section 3.3](https://arxiv.org/html/2507.07955v2#S3.SS3 "3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we provide comprehensive ablations that study individual architectural components and design choices.

##### Models.

We compare against a standard tokenized Transformer following the Llama architecture\parencite Llama2,Llama3.11 11 11 This was called the ”Transformer++” in \textcite Mamba; since by now it is firmly established, we remove the ”++”. We additionally compare against several byte-level baselines:

*   •MambaByte\parencite MambaByte is an isotropic model using pure Mamba-2 layers. 
*   •LlamaByte is an isotropic model using pure Transformer layers. 
*   •SpaceByte\parencite SpaceByte represents the canonical hierarchical architecture with external boundary supervision, which chunks on spaces and "space-like" bytes. 12 12 12 BLT is another architecture with external supervision using entropy instead of delimiters, but is unfortunately too complex to set up and control as a baseline. We believe that the delimiter-based method is highly competitive. See [Section A.1.3](https://arxiv.org/html/2507.07955v2#A1.SS1.SSS3 "A.1.3 External Chunking ‣ A.1 Autoregressive Tokenizer-free Architectures ‣ Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"). On English, the space-like delimiter heuristic empirically has an average ratio of 6.0 6.0 6.0 6.0 bytes per chunk. 
*   •SpaceByte++ is our modification of SpaceByte that includes our architectural modifications to the hierarchical structure (from [Section 2.1](https://arxiv.org/html/2507.07955v2#S2.SS1 "2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). In particular, it changes the outer encoder/decoder networks to use Mamba-2, and modifies the layer counts and widths slightly to match the H-Net models below. 
*   •H-Net (space) further improves SpaceByte++ with our training improvements to the network ([Section 2.3](https://arxiv.org/html/2507.07955v2#S2.SS3 "2.3 Improved Techniques for Hierarchical Sequence Modeling ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), in particular, adding post-network norms, residual projections, and learning rate multipliers on the outer networks. H-Net (space) differs from our full H-Net only through the chunking function. 
*   •H-Net (pool) is a baseline ablating the effect of a simple chunking strategy that pools every k 𝑘 k italic_k tokens, which is expected to be weaker than all of the data-dependent chunking strategies. 
*   •H-Net (1-stage) is our full H-Net method with DC learned end-to-end ([Section 2.2](https://arxiv.org/html/2507.07955v2#S2.SS2 "2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) with compression target N 0=6 superscript 𝑁 0 6 N^{0}=6 italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = 6. 
*   •H-Net (2-stage) is our full H-Net method, iterated to two nested stages using N 0=3,N 1=3 formulae-sequence superscript 𝑁 0 3 superscript 𝑁 1 3 N^{0}=3,N^{1}=3 italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = 3 , italic_N start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT = 3. 

We provide results for two model scales, _Large (L)_ and _XL_. Each scale is FLOP-matched to the corresponding GPT-3\parencite GPT3 (_i.e.,_ GPT-3 L and GPT-3 XL) variant of the tokenized Transformer (760 760 760 760 M and 1.3 1.3 1.3 1.3 B parameters respectively).

##### Experimental Setup.

Following established practice\parencite ByT5, MambaByte, SpaceByte, we measure performance using bits-per-byte (BPB) to ensure comparability across different input representations. For tokenized models, this amounts to simply rescaling the total negative log likelihood of a sequence (in tokens) by the total number of bytes. In addition, we systematically control the data and compute budget for all models (see [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), matching all models carefully in both bytes-per-batch and FLOPs-per-byte:13 13 13 Another way to interpret this is that every model sees the exact same underlying data (regardless of tokenization) per minibatch, and every model aims to use the same number of FLOPs in every forward/backward pass.

*   •Data Budget: We train all models on the 100B token subset sampled from the FineWeb-Edu dataset\parencite FineWeb-Edu. All tokenizer-free models process 8192 8192 8192 8192 utf-8 encoded bytes per sequence, while the Transformer uses 1792 1792 1792 1792 tokens from the GPT2 tokenizer (roughly equivalent to 8192 8192 8192 8192 bytes). We use batch size 256 256 256 256 for all models; the total batch size is just under 0.5 0.5 0.5 0.5 M tokens per batch for the baseline BPE Transformer, roughly matching protocols from prior work\parencite Mamba. 
*   •Compute Budget: For calculating FLOPs, we follow standard methodology\parencite Chinchilla with an extension for Mamba-2 layers (see [Appendix B](https://arxiv.org/html/2507.07955v2#A2 "Appendix B FLOPs Computation ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). We use the BPE-tokenized Transformer’s #FLOPs as a reference, and the number of layers of the other models is adjusted accordingly to match the reference #FLOPs. 

Training employs AdamW\parencite AdamW optimizer with warmup-stable-decay (WSD)\parencite WSD scheduling with 10%percent 10 10\%10 % linear warmup and 20%percent 20 20\%20 % inverse-square-root decay\parencite Inverse-Square-Root-Decay. Following \textcite hagele2024scaling which recommends WSD schedulers with half the maximum learning rates as a cosine schedule, we adopt learning rates 2.5×2.5\times 2.5 × higher than GPT-3\parencite GPT2 standards; this corresponds to half of the maximum learning rate used in \textcite Mamba, yielding 6.25×10−4 6.25 superscript 10 4 6.25\times 10^{-4}6.25 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT for Large-scale models and 5.0×10−4 5.0 superscript 10 4 5.0\times 10^{-4}5.0 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT for XL-scale models. Architecture details include gated MLPs\parencite Llama2 in all Transformer layers and the main network’s Mamba layers, while Mamba layers in ℰ ℰ\mathcal{E}caligraphic_E and 𝒟 𝒟\mathcal{D}caligraphic_D are without an MLP.14 14 14 Just as in the original Mamba\parencite Mamba and Mamba-2\parencite Mamba2 blocks, our Mamba layers have roughly 6⁢(D s)2 6 superscript superscript 𝐷 𝑠 2 6(D^{s})^{2}6 ( italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT parameters and Transformer layers have 12⁢(D s)2 12 superscript superscript 𝐷 𝑠 2 12(D^{s})^{2}12 ( italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT parameters in stage s 𝑠 s italic_s. For Transformer layers in ℰ ℰ\mathcal{E}caligraphic_E and 𝒟 𝒟\mathcal{D}caligraphic_D, we use Sliding Window Attention (SWA)\parencite longformer with the window size of 1024 1024 1024 1024. As discussed [Section 2.1](https://arxiv.org/html/2507.07955v2#S2.SS1 "2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), ℰ ℰ\mathcal{E}caligraphic_E and 𝒟 𝒟\mathcal{D}caligraphic_D comprise mainly Mamba-2 layers.

### 3.1 Language Modeling

##### Training Curves.

Figure[3](https://arxiv.org/html/2507.07955v2#S3.F3 "Figure 3 ‣ XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") presents validation BPB metrics throughout training for both Large and XL model scales.

##### Large Scale.

At the _Large_ scale, we make note of the following comparisons.

*   •All isotropic models severely underperform hierarchical models. Among these, MambaByte is significantly better than LlamaByte, both the FLOP-matched sliding window attention (SWA) variant and even the global attention variant that is data-matched but uses 2×2\times 2 × the FLOPs. 
*   •H-Net (pool) is much worse than all other H-Net variants, validating that fixed-width chunking is not effective. 
*   •SpaceByte is much worse than SpaceByte++, validating our strategy for network design as well as usage of Mamba in the outer networks ([Section 2.1](https://arxiv.org/html/2507.07955v2#S2.SS1 "2.1 Architectural Overview ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). SpaceByte++ is in turn worse than H-Net (space), validating our improved signal propagation techniques ([Section 2.3](https://arxiv.org/html/2507.07955v2#S2.SS3 "2.3 Improved Techniques for Hierarchical Sequence Modeling ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). 
*   •H-Net (space) is a very strong model reaching the performance of the BPE Transformer, validating the effect of data-dependent chunking strategies together with a well-designed hierarchical architecture. 
*   •H-Net (1-stage) is stronger than H-Net (space), validating that our dynamic chunking mechanism successfully learns how to segment data in a _context-dependent_ way that improves over strong heuristics. 
*   •H-Net (2-stage) is significantly better than H-Net (1-stage), validating that iterated dynamic chunking can potentially learn a nested hierarchy of useful features, and leverage compute and parameters even more effectively. 

##### XL Scale.

At the XL scale, we zoom in more closely and compare only the strongest set of methods: SpaceByte++, H-Net (space), H-Net (1-stage), and H-Net (2-stage).

The same trends hold as at the _Large_ scale. Our SpaceByte++ baseline is strong, but slightly worse than the BPE Transformer baseline. On the other hand, all byte-level H-Net methods start off worse than the token-level Transformer, but scale better after enough data. H-Net (space), H-Net (1-stage), and H-Net (2-stage) cross over the tokenized Transformer after just 200 200 200 200 B bytes, 100 100 100 100 B bytes, and 30 30 30 30 B bytes respectively. Beyond these points, H-Net’s performance advantage widens progressively, demonstrating that the benefits of learnable dynamic chunking get strengthened with additional training data, as the model continuously refines its chunking strategy.

![Image 3: Refer to caption](https://arxiv.org/html/2507.07955v2/x3.png)

Figure 3: Validation Bits-per-byte (BPB) throughout training for different models at Large (760 760 760 760 M, left) and XL (1.3 1.3 1.3 1.3 B, right) scales with matched computational and data budgets for training. All models but Transformer take raw byte inputs (Transformer uses GPT-2 tokenizer). Vertical dotted lines indicate crossover points where H-Net begins to outperform Transformer with predefined BPE tokenization. From the curves we can clearly see the following: (1) all hierarchical models (_i.e.,_ SpaceByte++, H-Net variants) outperform the isotropic models (_i.e.,_ Transformer, MambaByte, LlamaByte); (2) dynamic chunking is more powerful than BPE tokenizers; and (3) DC is more effective than other chunking strategies. Furthermore, H-Net’s 2-stage variant consistently outperforms 1-stage across both scales, demonstrating the effectiveness of deeper hierarchies. See [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") for architectural details. 

Table 2: Zero-shot performance comparison across multiple benchmarks, all data-/FLOP-matched. Evaluation results on seven downstream tasks at both Large (760M) and XL (1.3B) scales. GFLOPs/Byte is measured during evaluation on FineWeb-Edu validation set. See [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") for architectural details. 

Model Input GFLOPs/F-Edu LMB.Hella.PIQA ARC-e ARC-c Wino.Open.Average
Byte bpb↓↓\downarrow↓acc↑↑\uparrow↑acc _n↑↑\uparrow↑acc↑↑\uparrow↑acc↑↑\uparrow↑acc _n↑↑\uparrow↑acc↑↑\uparrow↑acc _n↑↑\uparrow↑acc↑↑\uparrow↑
#FLOPs matched to GPT-3 Large
Transformer Token 0.42 0.756 45.0 54.5 72.3 69.9 36.3 55.9 38.8 53.3
LlamaByte 0.42 0.859 37.0 40.5 64.7 55.1 26.7 52.3 32.4 44.1
LlamaByte (Global)0.95 0.845 36.4 41.5 65.7 57.2 27.1 49.8 32.2 44.3
MambaByte 0.42 0.845 32.9 42.0 66.2 55.9 28.1 51.7 33.2 44.3
SpaceByte 0.41 0.791 43.0 49.0 69.0 63.3 33.5 53.3 35.0 49.4
SpaceByte++Byte 0.42 0.760 48.0 55.7 71.3 67.9 35.4 57.5 39.6 53.6
H-Net (pool)0.42 0.780 43.2 54.7 69.7 67.9 34.7 54.8 36.4 51.6
H-Net (space)0.42 0.755 46.7 55.9 72.4 68.8 34.6 57.6 38.0 53.4
H-Net (1-stage)0.43 0.755 46.2 55.5 71.0 68.1 35.6 58.6 40.0 53.6
H-Net (2-stage)0.43 0.743 46.9 57.4 72.0 71.7 39.2 60.4 40.6 55.5
#FLOPs matched to GPT-3 XL
Transformer Token 0.69 0.730 48.1 58.0 73.1 72.2 37.5 58.6 40.8 55.5
SpaceByte++0.72 0.733 51.3 60.1 72.4 71.8 38.0 58.5 40.6 56.1
H-Net (space)0.70 0.726 50.3 61.5 73.6 72.4 40.2 60.2 41.8 57.1
H-Net (1-stage)Byte 0.72 0.728 48.4 59.5 72.4 73.0 38.3 59.2 42.4 56.2
H-Net (2-stage)0.69 0.715 50.5 62.2 73.7 74.2 42.2 60.5 44.0 58.2

Table 3: Robustness evaluation on HellaSwag with textual perturbations, all data-/FLOP-matched. Zero-shot accuracy on five different perturbation types (AntSpeak, Drop, RandomCase, Repeat, UpperCase) for models trained exclusively on clean data without noise augmentation. Best and second best results in each column are denoted using bolded and underlined texts, respectively. The Robustness Score metric show that all byte-level models are more robust to adversarial text inputs than tokenizer-based Transformer. H-Net (2-stage) shows significantly enhanced robustness in textual perturbations, with the highest average accuracy across all noise types and highest robustness score. See [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") for architectural details, and [Section D.1](https://arxiv.org/html/2507.07955v2#A4.SS1 "D.1 Robustness Score ‣ Appendix D Additional Experimental Setup Details ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") for the definition of Robustness Score. 

Model Input HellaSwag Average↑↑\uparrow↑Robustness Score↑↑\uparrow↑
AntSpeak Drop RandomCase Repeat UpperCase
#FLOPs matched to GPT-3 Large
Transformer Token 31.1 29.9 27.1 27.8 38.9 30.9 20.2
LlamaByte (W1024)30.4 28.1 29.3 27.2 38.5 30.7 36.9
LlamaByte (Global)31.1 28.1 29.7 27.3 39.0 31.0 36.6
MambaByte 29.8 27.9 29.9 27.1 39.6 30.9 34.5
SpaceByte 30.7 29.8 33.5 29.5 47.8 34.3 38.1
SpaceByte++Byte 31.0 30.9 35.8 29.3 54.0 36.2 36.4
H-Net (pool)30.5 31.2 35.4 29.6 53.4 36.1 37.3
H-Net (space)30.8 31.2 38.6 29.4 54.0 36.8 38.2
H-Net (1-stage)31.2 31.1 35.4 29.9 54.1 36.4 37.2
H-Net (2-stage)30.8 32.1 39.3 30.4 57.1 38.0 39.0
#FLOPs matched to GPT-3 XL
Transformer Token 31.6 30.7 28.0 28.5 43.0 32.3 22.2
SpaceByte++30.9 32.1 40.3 30.6 58.5 38.5 38.5
H-Net (space)31.2 33.2 41.9 31.8 60.7 39.8 40.5
H-Net (1-stage)Byte 30.9 32.7 39.2 31.2 58.4 38.6 39.5
H-Net (2-stage)31.1 34.7 44.1 33.0 61.7 40.9 42.8

##### Downstream Evaluations.

[Table 2](https://arxiv.org/html/2507.07955v2#S3.T2 "In XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") presents zero-shot accuracy across diverse downstream benchmarks\parencite LAMBADA, HellaSwag, PIQA, ARC, WinoGrande, OpenBookQA using lm-evaluation-harness\parencite LMEvalHarness for models at Large and XL scales. SpaceByte++, H-Net (space), and H-Net (1-stage) all have similar performance to the BPE Transformer at _Large_ scale, and slightly outperform it at the _XL_ scale, consistent with their close training curves (and possibly reflecting some noise in the evaluations).

H-Net (2-stage) consistently achieves the highest performance across most tasks, outperforming 2.2 2.2 2.2 2.2% and 2.6 2.6 2.6 2.6% over the Transformer baseline at _Large_ and _XL_ scales respectively. Notably, the _Large_ H-Net (2-stage) matches the average downstream performance of the _XL_ BPE Transformer.

##### Robustness to Textual Perturbations.

[Table 3](https://arxiv.org/html/2507.07955v2#S3.T3 "In XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") evaluates model robustness on HellaSwag with various textual perturbations, following protocols from BLT\parencite BLT. Importantly, these are the same checkpoints trained on clean FineWeb-Edu data used to evaluate [Table 2](https://arxiv.org/html/2507.07955v2#S3.T2 "In XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), without any form of special data mix or augmentations that may improve character-level robustness. H-Net (2-stage) demonstrates substantially improved robustness compared to all baselines, with performance gaps exceeding those observed in standard benchmarks.

##### Visualization of Tokenized Positions.

In [Figure 4](https://arxiv.org/html/2507.07955v2#S3.F4 "In Visualization of Tokenized Positions. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we provide visualizations of the boundaries dynamically drawn by H-Net (1-stage) and H-Net (2-stage). The visualization offers several insights about how the model decides boundaries.

*   •Single-stage behavior: H-Net (1-stage) predominantly places boundaries at whitespace characters, closely mirroring the delimiters used by SpaceByte. This indicates that H-Net learns that word boundaries represent natural semantic units in text. This convergence to spacelike boundaries, discovered purely through end-to-end training, conversely validates SpaceByte’s strong empirical performance. 
*   •Hierarchical chunking patterns: The first stage of H-Net (2-stage) combines spacelike boundaries with first few characters of each word. This strategy helps the model because once the initial positions of a word are identified, the remaining characters become highly predictable. 
*   •Content-aware chunking: One might question if H-Net’s chunking decisions follow static rules, such as drawing boundaries only at certain fixed bytes (_e.g.,_ whitespace). However, as shown in the figure, H-Net often merges multiple words and spacelike characters based on content (examples include the backbone, such as, and (ii)). 
*   •Perturbation behavior:[Figure 16](https://arxiv.org/html/2507.07955v2#A6.F16 "In Results. ‣ Appendix F Distilling Token-Level Models to Byte-Level ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") shows the same example with textual perturbations such as removing whitespaces, which more prominently demonstrates that boundaries drawn by H-Net are based on content and context. In particular, it often still chunks in between semantic words even if the space is removed. 

![Image 4: Refer to caption](https://arxiv.org/html/2507.07955v2/x4.png)

(a) H-Net (1-stage), using 6 6 6 6-DC.

Figure 4: Visualization of boundaries drawn by H-Net. Numbers above indicate byte value, and the colored boxes indicate positions where b t=1 subscript 𝑏 𝑡 1 b_{t}=1 italic_b start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = 1. (a) H-Net (1-stage) tends to draw boundaries at spacelike bytes, which is very similar to SpaceByte. (b) The boundaries of the first stage in H-Net (2-stage) are focused on spacelike bytes, as well as starting characters of each word. The second stage of H-Net (2-stage) chunks the text into more meaningful units, such as words or numberings (_i.e.,_‘(ii)’). We can also observe that it often chunks multiple words that form one semantic group; for example, ‘the backbone’ and ‘such as’. 

![Image 5: Refer to caption](https://arxiv.org/html/2507.07955v2/x5.png)

(b) H-Net (2-stage), using (3,3)-DC.

### 3.2 Alternate Language Datasets

![Image 6: Refer to caption](https://arxiv.org/html/2507.07955v2/x6.png)

Figure 5: Validation Bits-per-byte (BPB) throughout training on Chinese language and code modeling. H-Net (space) and H-Net (2-stage) are byte-level, while the Transformers use the Llama-3 tokenizer which was designed for multilingual. H-Net clearly outperforms both Transformer and H-Net (space) on Chinese language modeling, which does not have space-like segmentation cues, with lower BPB than H-Net (space) throughout training and crossing over with Transformer after around 25B bytes. On code, both H-Net (2-stage) and H-Net (space) significantly outperform BPE Transformer. Final post-decay results can be found in [Table 4](https://arxiv.org/html/2507.07955v2#S3.T4 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"). 

Besides conventional language modeling, we also examine three other language modeling settings: Chinese, code, and DNA. These three settings present distinct challenges for traditional language-modeling pipelines:

*   •Chinese characters consist of 3 utf-8 encoded bytes each and Chinese language does not have natural spaces; thus, constructing a vocabulary or picking boundaries requires special consideration. 
*   •Code contains much more whitespace than typical language, which allows greater compressibility if handled properly. It also has latent hierarchical structure that can be leveraged for improved reasoning capabilities. 
*   •DNA does not have any natural tokenization cues and instead must be processed as raw base pairs. 

H-Net can operate on raw data without the need for handcrafted features (whether vocabulary or deliniation cues); it therefore provides a natural architecture that can operate naturally on any language.

##### Experimental setup for Chinese and code.

On Chinese and code, we use 46B token subset from FineWeb-Edu-Chinese-V2.1\parencite OpenCSG and Github subset from Pile\parencite pile to train three models at the 1.3B GPT-3 XL scale: H-Net (2-stage), H-Net (space), and Transformer. We maintain the same bytes per gradient step (256 batch size with 8192 utf-8 encoded bytes per example) as the main text experiments. For the H-Net (2-stage), we use the same target downsampling ratio (N 0=N 1=3 superscript 𝑁 0 superscript 𝑁 1 3 N^{0}=N^{1}=3 italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = italic_N start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT = 3) as the main experiments. Unlike BPE or spacelike-based tokenization, whose downsampling ratios can vary widely by dataset, H-Net allows for using similar compute budgets without much adjustment. For H-Net (space), we use the same definition of spacelike as the original SpaceByte paper \parencite SpaceByte, and for BPE, we use the Llama3 tokenizer\parencite Llama3, as the GPT2 tokenizer attains very poor downsampling ratios on both datasets. Despite this change, both H-Net (space) and Transformer (BPE) still have highly varied downsampling ratios between ordinary (primarily English) language, Chinese, and code. On the other hand, H-Net can adhere to a target ratio regardless of dataset, chunking into concepts at appropriate ratios.

Even with the Llama3 tokenizer, we find that H-Net (2-stage) scales better than BPE Transformer and H-Net (space) on both Chinese and code ([Figure 5](https://arxiv.org/html/2507.07955v2#S3.F5 "In 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), and achieves lower compression after the decay phase ([Table 4](https://arxiv.org/html/2507.07955v2#S3.T4 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). We additionally measure the performance of each Chinese-language model on the Chinese split of XWinograd, a multilingual Winograd Schema Challenge \parencite xwinograd, where H-Net (2-stage) is significantly better than H-Net (space) which in turn is better than Transformer ([Table 4](https://arxiv.org/html/2507.07955v2#S3.T4 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

Table 4: Architecture details and model benchmarks for Chinese and code models. BPIC (defined in [Table 2](https://arxiv.org/html/2507.07955v2#S3.T2 "In XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) denotes the compression between the main network and outermost stage (bytes). Each H-Net used (3,3)-DC, targeting an inner downsampling ratio of 9 9 9 9. However, the resulting BPIC was significantly different, indicating that code is much easier to compress than Chinese. In terms of results, H-Net (2-stage) performs better than both H-Net (space) and BPE Transformer on Chinese, which is reflected in the downstreams. On the other hand, H-Net (2-stage) achieves similar performance to H-Net (space) on code, and both H-Net models perform significantly better than Transformer. 

Model Chinese Code
BPIC Main arch.Val. BPB↓↓\downarrow↓XW-zh. acc.↑↑\uparrow↑BPIC Main arch.Val. BPB↓↓\downarrow↓
Transformer 3.62 T15 0.7404 0.599 3.58 T13 0.3376
H-Net (space)3.38 T19 0.7478 0.629 7.97 T40 0.3163
H-Net (2-stage)5.81 T30 0.7032 0.663 7.23 T28 0.3161

Model / Architecture Params.Final ppl.↓↓\downarrow↓
Transformer  (T9)29M 2.769
Mamba-2  (M10)33M 2.757
H-Net (M3T1+T15+M4)64M 2.705
H-Net (M3T1+M15+M4)66M 2.697

Table 5: Model details and final performance on HG38. We trained two isotropic models and two H-Net models, varying the main network architecture (Transformer or Mamba-2). Each H-Net model outperforms the corresponding isotropic model. We empirically find that the ℰ 0=superscript ℰ 0 absent\mathcal{E}^{0}=caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT =M3T1 encoder architecture slightly outperforms a pure Mamba-2 encoder ℰ 0=superscript ℰ 0 absent\mathcal{E}^{0}=caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT =M4 ([Section E.3](https://arxiv.org/html/2507.07955v2#A5.SS3 "E.3 DNA Architecture Ablations ‣ Appendix E Additional Ablation Studies ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). 

![Image 7: [Uncaptioned image]](https://arxiv.org/html/2507.07955v2/x7.png)

Figure 6: Scaling performance on HG38 during the stable phase of training. Each H-Net model achieves the same pre-decay perplexity of the corresponding isotropic model with approximately 3.6×3.6\times 3.6 × less data. 

![Image 8: Refer to caption](https://arxiv.org/html/2507.07955v2/x8.png)

Figure 7: Ablation study on key H-Net components showing validation BPB (left) and compression ratios for the first stage L 1/L 0 superscript 𝐿 1 superscript 𝐿 0 L^{1}/L^{0}italic_L start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT / italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT (center) and second stage L 2/L 1 superscript 𝐿 2 superscript 𝐿 1 L^{2}/L^{1}italic_L start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT / italic_L start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT (right) during training. Using H-Net (2-stage), we evaluate the impact of removing three components: the smoothing module (w/o smoothing), the similarity-based routing module (w/o cosine routing), and Straight-Through Estimator (w/o STE). 

##### DNA (Human Genome).

DNA is a setting that presents both a unique promise and challenge for hierarchical modeling. For one, handcrafted tokens do not work well on DNA, due to the lack of segmentation cues. Additionally, the same sequence of base pairs may serve different functions (_e.g.,_ depending on whether or not the pair is inside a gene or not). Consequently, a naive BPE-based approach may not work either. On the other hand, DNA _can_ exhibit higher resolution structure (_e.g.,_ codons, various regulatory elements), suggesting that there is room for principled hierarchical modeling. Indeed, state-of-the-art DNA models \parencite Evo2 operate directly on base pairs (A, C, G, T) with implicit hierarchical structure.

Thus, we evaluated four models on DNA: two isotropic models (pure Transformer and pure Mamba-2) operating at the base-pair level, and two corresponding H-Net (1-stage) with Transformer and Mamba-2 as the main network. Each model was trained on the HG38 dataset with a learning rate of 5⋅10−3⋅5 superscript 10 3 5\cdot 10^{-3}5 ⋅ 10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT for modules at the base-pair resolution. For the H-Net models, we used a downsampling ratio of N 0=3 superscript 𝑁 0 3 N^{0}=3 italic_N start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = 3. All models were trained with a d model subscript 𝑑 model d_{\text{model}}italic_d start_POSTSUBSCRIPT model end_POSTSUBSCRIPT of 512 512 512 512, which was used for all isotropic modules of H-Net (including the main network).

Previous work has shown that SSMs show improved DNA modeling ability compared to Transformers \parencite Mamba, and we find that this effect is preserved when examining Transformers vs. Mamba-2 as the main network (see [Table 5](https://arxiv.org/html/2507.07955v2#S3.T5 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). This finding suggests that existing layer selection principles can be applied when deciding main network architecture. In fact, by directly comparing the perplexity curves during the stable phase of training ([Figure 6](https://arxiv.org/html/2507.07955v2#S3.F6 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), we find that H-Net models can achieve similar performance to isotropic models with just 3.6×3.6\times 3.6 × the amount of data, a finding that holds for both choices of main network architecture.

![Image 9: Refer to caption](https://arxiv.org/html/2507.07955v2/x9.png)

Figure 8: Encoder-decoder architecture ablation using raw byte inputs. Validation BPB (left) and compression ratio L 1/L 0 superscript 𝐿 1 superscript 𝐿 0 L^{1}/L^{0}italic_L start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT / italic_L start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT (right) for H-Net (1-stage) throughout training. We evaluate four encoder-decoder (ℰ 0−𝒟 0 superscript ℰ 0 superscript 𝒟 0\mathcal{E}^{0}-\mathcal{D}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT - caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT) configurations: M4-M4 , M2T1-T1M2 and T1M2-M2T1, and T2-T2, where M denotes Mamba layers and T denotes Transformer layers. 

### 3.3 Ablation Studies

For ablation studies, we employ H-Net at _Large_ scale following the configurations in [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), training on 36 36 36 36 B tokens randomly sampled from FineWeb-Edu.

##### Importance of Components in H-Net.

[Figure 7](https://arxiv.org/html/2507.07955v2#S3.F7 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") illustrates the impact of each architectural component on both model performance and compression ratio (L s+1/L s superscript 𝐿 𝑠 1 superscript 𝐿 𝑠 L^{s+1}/L^{s}italic_L start_POSTSUPERSCRIPT italic_s + 1 end_POSTSUPERSCRIPT / italic_L start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT) stability during training. We conduct three targeted ablations: (i) using direct upsampling z~t=z^t subscript~𝑧 𝑡 subscript^𝑧 𝑡\tilde{z}_{t}=\hat{z}_{t}over~ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT = over^ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT by removing the smoothing module (_w/o smoothing_), (ii) replacing the routing module that is based on scaled cosine similarity, with direct probability prediction from individual inputs (_w/o cosine routing_), and (iii) skipping the straight-through estimator in equation ([9](https://arxiv.org/html/2507.07955v2#S2.E9 "Equation 9 ‣ Upsampler. ‣ 2.2.2 Dechunking ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) (w/o STE).

The smoothing module proves essential for stable training dynamics. Without this module, compression ratios fluctuate severely throughout training, preventing the model from learning consistent chunking boundaries. This instability directly manifests as substantial performance degradation, confirming that smooth gradient flow through the decompression process is crucial for effective end-to-end learning. While less critical than the smoothing module, both the similarity-based routing module and STE operation exhibit importance in training stability and final performance. These components help maintain consistent compression ratios and lead to more interpretable chunking patterns. The similarity-based approach particularly enables the model to identify natural linguistic boundaries (_e.g.,_ whitespaces, subwords) by comparing adjacent representations rather than making isolated predictions.

##### Encoder & Decoder Layer Selection.

The composition of sequence mixing layers in H-Net’s encoders and decoders substantially influences both compression efficiency and modeling capability. We systematically evaluate different architectural combinations using H-Net (1-stage) while fixing all other configurations in [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") the same. Four distinct encoder-decoder (ℰ 0 superscript ℰ 0\mathcal{E}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT-𝒟 0 superscript 𝒟 0\mathcal{D}^{0}caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT) pairings are tested: M4-M4, M2T1-T1M2, T1M2-M2T1, and T2-T2, where M denotes a Mamba-2 layer and T denotes a Transformer layer. These combinations are chosen by keeping the symmetry and replacing each Transformer layer with two Mamba-2 layers, as they contain equivalent parameter counts — 12⁢D 2 12 superscript 𝐷 2 12D^{2}12 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for Transformer (4⁢D 2 4 superscript 𝐷 2 4D^{2}4 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for the attention mechanism and 8⁢D 2 8 superscript 𝐷 2 8D^{2}8 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for an MLP) vs. ≈6⁢D 2 absent 6 superscript 𝐷 2\approx 6D^{2}≈ 6 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT per Mamba-2 layer (no MLP).

[Figure 8](https://arxiv.org/html/2507.07955v2#S3.F8 "In DNA (Human Genome). ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") and [Figure 9](https://arxiv.org/html/2507.07955v2#S3.F9 "In Encoder & Decoder Layer Selection. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") demonstrate that Mamba layers are essential for effective byte-level sequence processing. For both H-Net and SpaceByte++, the pure Transformer configuration (T2-T2) exhibits by far the worst performance despite using more FLOPs (it also down-compresses sequences poorly compared to other configurations, thus using more compute in the main network). This configuration struggles to compress byte sequences effectively, resulting in both computational waste and degraded modeling performance. Performance improves monotonically with increased Mamba layer allocation, achieving optimal results with the highest compression efficiency in the pure Mamba configuration (M4-M4). These findings align with recent research demonstrating SSMs’ advantages over Transformers for fine-grained sequence modeling\parencite SaShiMi, Caduceus, as corroborated by MambaByte’s superior performance over LlamaByte in [Figure 3](https://arxiv.org/html/2507.07955v2#S3.F3 "In XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling").

![Image 10: Refer to caption](https://arxiv.org/html/2507.07955v2/x10.png)

Figure 9: SpaceByte++ encoder-decoder architecture ablation using raw byte inputs. We evaluate four encoder-decoder (ℰ 0−𝒟 0 superscript ℰ 0 superscript 𝒟 0\mathcal{E}^{0}-\mathcal{D}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT - caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT) configurations: M4-M4 , M2T1-T1M2 and T1M2-M2T1, and T2-T2, where M denotes Mamba layers and T denotes Transformer layers. 

![Image 11: Refer to caption](https://arxiv.org/html/2507.07955v2/x11.png)

Figure 10: Encoder-decoder architecture ablation using BPE-tokenized inputs. Assuming that GPT-2 tokenizer serves as the outermost encoder-decoder (_i.e.,_ ℰ 0 superscript ℰ 0\mathcal{E}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT-𝒟 0 superscript 𝒟 0\mathcal{D}^{0}caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT), we evaluate six ℰ 1 superscript ℰ 1\mathcal{E}^{1}caligraphic_E start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT-𝒟 1 superscript 𝒟 1\mathcal{D}^{1}caligraphic_D start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT combinations: M6-M6, M4T1-T1M4, T1M4-M4T1, M2T2-T2M2, T2M2-M2T2, and T3-T3. 

A natural question arises: does the importance of Mamba layers (i) stem specifically from processing fine-grained byte inputs, or (ii) because they are better for compressing information into the next stage, even at coarser input resolutions? To investigate these hypotheses, we train a 1-stage H-Net on top of _BPE-tokenized_ inputs processed by the GPT-2 tokenizer. We then evaluate six different encoder-decoder combinations.

*   •If hypothesis (i) holds, then we would expect different combinations of Mamba/Transformer layers in the encoder/decoder to have similar performance, since it is known that they have similar performance on standard tokenized language modeling. 
*   •If hypothesis (ii) holds, then we would expect that encoders/decoders using some Mamba layers to be better than pure Transformer layers. 

As demonstrated in [Figure 10](https://arxiv.org/html/2507.07955v2#S3.F10 "In Encoder & Decoder Layer Selection. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), Mamba layers prove significantly important even when processing BPE tokens rather than raw bytes, providing evidence for the second hypothesis.

We hypothesize that this consistent advantage across input granularities stems from fundamental architectural differences between SSMs and Transformers. While Transformers naturally store complete key-value caches for all positions, SSMs are designed to compress information into fixed-size states. This compression-oriented architecture aligns naturally with our chunking mechanism, which requires aggregating multiple input vectors into consolidated representations. The inherent compression capability of Mamba layers makes them particularly well-suited for the encoder and decoder roles in our hierarchical architecture\parencite gu2025tradeoffs. Based on these findings, we employ Mamba layers throughout all encoders and decoders in our final H-Net configuration, as detailed in [Table 1](https://arxiv.org/html/2507.07955v2#S3.T1 "In 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling").

These findings transfer to more general hierarchical structures (such as a 2-stage H-Net at the byte level), in which case the outermost encoder and decoder layers (ℰ 0 superscript ℰ 0\mathcal{E}^{0}caligraphic_E start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT and 𝒟 0 superscript 𝒟 0\mathcal{D}^{0}caligraphic_D start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT) serve a similar role as the GPT-2 tokenizer and the inner layers (ℰ 1 superscript ℰ 1\mathcal{E}^{1}caligraphic_E start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT and 𝒟 1 superscript 𝒟 1\mathcal{D}^{1}caligraphic_D start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT) would share similar findings of benefiting from using Mamba layers.

##### Hybrid Architectures for the Main Network.

![Image 12: Refer to caption](https://arxiv.org/html/2507.07955v2/x12.png)

Figure 11: Hybrid main network. Bits-per-byte during the stable phase of training, for H-Net (2-stage) with Transformer main stage and with hybrid main stage. The hybrid main stage scales better, similar to findings for standard token-based language models. This finding suggests that design principles for isotropic (tokenized) models can carry over to choices of the main network.

![Image 13: Refer to caption](https://arxiv.org/html/2507.07955v2/x13.png)

Figure 12: Comparison to Mixtures-of-Experts. Bits-per-byte comparison of H-Net (both 1-stage and 2-stage) to LlamaByte-MoE, which is a FLOPs-matched MoE model that uses a similar number of parameters as H-Net (2-stage). Both H-Nets perform much better than LlamaByte-MoE, implying that H-Net’s capabilities do not just come from sparsity.

We also aimed to understand the role of architecture selection in the main network. To this end, we compared H-Net (2-stage) with an identical model where we replaced the Transformer stack with a hybrid model containing both 20 Mamba-2 and 7 Transformer layers interleaved in a 3:1 ratio. Hybrid architectures have shown promise in isotropic (BPE) models\parencite NvidiaMambaHybrid, and similarly perform better for our choice of main network ([Figure 11](https://arxiv.org/html/2507.07955v2#S3.F11 "In Hybrid Architectures for the Main Network. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

##### Comparison to Mixture-of-Experts.

H-Net can be viewed as a form of dynamic sparsity similar to Mixture-of-Experts (MoEs), in that they are able to improve performance by using more parameters, all while keeping the total FLOPs budget constant. We were interested in understanding whether or not its performance benefits were simply due to increasing sparsity. We compare against a sparsified version of LlamaByte (byte-level isotropic Transformer model) at the Large scale with a standard Mixture-of-Experts recipe\parencite MoE and similar parameter count as ours ([Figure 12](https://arxiv.org/html/2507.07955v2#S3.F12 "In Hybrid Architectures for the Main Network. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). While sparsity does improve LlamaByte performance, it is still far worse than either FLOPs-matched H-Net (1-stage) or H-Net (2-stage), even with similar parameter count. We interpret this result as: H-Net not only achieves sparsity, but does so in a more semantically meaningful manner, which allows for better scaling than even generic sparse methods.

Table 6: Related architectures. Comparison of related architectures, particularly those focused on byte-level modeling. H-Net is the first architecture that enables dynamic, multi-stage hierarchies. Extended discussion is provided in[Appendix A](https://arxiv.org/html/2507.07955v2#A1 "Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"). 

Class Autoregressive Chunking Mechanism Multi-stage Example Architectures
Hierarchy
Isotropic✗——ByT5
✓——MambaByte
Hierarchical (static)✗k 𝑘 k italic_k-width pooling✓Funnel-Transformer
Canine
Charformer
✓k 𝑘 k italic_k-width pooling✓Hourglass Transformer
SaShiMi
MegaByte
Block Transformer
MBLM
AU-Net 3
Hierarchical (external)✗delimiters✗eByte
WSF
✓delimiters✗DPT (Whitespaces)
SpaceByte
AU-Net 2
entropy✗DPT (Entropy)
BLT
Hierarchical (dynamic)✓soft matching✗MANTa
soft gating✗MrT5
stochastic reparameterization✗DPT (Gumbel)
Hierarchical (dynamic)✓dynamic chunking✓H-Net

4 Discussion
------------

##### Related Work.

[Table 6](https://arxiv.org/html/2507.07955v2#S3.T6 "In Comparison to Mixture-of-Experts. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") summarizes related models, particularly those motivated by byte-level language modeling. These methods are described in detail in [Appendix A](https://arxiv.org/html/2507.07955v2#A1 "Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), which provides an extended related work.

##### Distillation.

For new architectures, showing that they can be distilled from standard pretrained Transformers can result in stronger new models with substantially reduced training\parencite MOHAWK. In [Appendix F](https://arxiv.org/html/2507.07955v2#A6 "Appendix F Distilling Token-Level Models to Byte-Level ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we investigate this for H-Net by initializing the main network from a pretrained Llama checkpoint and learning the encoder and decoder networks. With less than 200B bytes of training, the resulting model shows strong performance much better than if it were trained from scratch, although still worse than the teacher model. Our distillation procedure is perhaps currently the most efficient way of creating an end-to-end byte-level model, but we expect that it can be further improved.

##### Efficiency.

Because of the dynamic nature of our model, it requires different considerations in making both the training pass and inference step efficient. Our implementation incorporates several engineering techniques already, such as handling variable sequence lengths within a mini-batch using specialized kernels provided by \textcite FlashAttention2, Mamba2. Because of the different architectural considerations, it is difficult to compare to more standard pipelines; our current implementation may be approximately up to 2×2\times 2 × slower than an isotropic model during training.

Note that the memory usage of our model is also dynamic, unlike standard sequence models, so other edge cases may happen, such as unlucky batches of sequences that are too long and overflow the device memory. Relatedly, one difficulty with stepping H-Net in batched mode is that different tokens in the batch may require different amounts of compute.

We believe that such considerations are not fundamental and will be an important subject of future work; just as how related dynamic sparsity and conditional compute methods such as Mixture-of-Experts and speculative decoding\parencite SpeculativeDecoding, SpeculativeSampling benefited from years of dedicated engineering improvements.

##### Deeper Hierarchies.

H-Net is the first _dynamic_ hierarchical model that can _recursively_ nest its chunking strategy (see [Table 6](https://arxiv.org/html/2507.07955v2#S3.T6 "In Comparison to Mixture-of-Experts. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") and [Appendix A](https://arxiv.org/html/2507.07955v2#A1 "Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). In this paper, we showed that iterating H-Net from 0 stages (i.e. an isotropic model) to 1 stage and from 1 stage to 2 stages consistently improves performance. We did not attempt a 3-stage H-Net at all for simplicity. Testing if H-Net can be iterated even deeper remains an immediate direction to explore.

##### Global Sequence Model Considerations.

Much research on sequence model architectures has focused on individual layers, where the tradeoffs are often quite direct. For example, recurrent models such as state space models \parencite gu2023thesis, Mamba and linear attention variants \parencite LA, GLA, DeltaNet, GatedDeltaNet compress arbitrarily long sequences into fixed-size hidden states, offering higher efficiency at the expense of precise retrieval of information (e.g. struggling with recall\parencite RepeatAfterMe).

H-Net, however, is a _global_ architectural design that is simultaneously orthogonal to, but may have interactive effects with, the choice of individual layers. For example, using deeper hierarchies with exclusively recurrent layers would preserve linear computation (in sequence length) but _logarithmic_ state size, resembling newer sequence model layers such as log-linear attention \parencite LogLinearAttn and Prefix Scannable Models \parencite PrefixScannableModels, but with dynamic hierarchies. Similarly, the recursive compression of sequence length may alleviate their limitations in retrieval on long sequences. This may be considered a form of _dynamic state allocation_. This paper has not focused on such implications, which would be a possible direction for future research.

##### Long Context.

Similarly, an effect of the global hierarchical structure may be improved long context abilities, which is a common motivation for hierarchical models\parencite CW-RNN, DilatedRNN. Much research on sequence models again focuses on long context at the layer level\parencite hyena, Mamba, Transformer, and we hypothesize that H-Nets may provide general long context improvements in an orthogonal direction.

##### Latent Test-Time Compute.

Test-time compute techniques, exemplified by Chain-of-Thought \parencite CoT, have been shown to improve model performance on a variety of reasoning benchmarks \parencite S1,o1preview. Recent work has explored including latent representations (as opposed to just tokens) in the reasoning process \parencite coconut, culminating in “recurrent depth" models that roll out an RNN for as many steps as needed before emitting a token \parencite ReccurentDepth. As discussed in [Section 2.4](https://arxiv.org/html/2507.07955v2#S2.SS4 "2.4 Autoregressive Training and Inference ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), H-Net is also capable of dynamically changing compute per output generated; thus, it can be viewed as a model that can dynamically allocate latent test-time compute as well. Additionally, as the motivation of H-Net is to recursively build higher-order abstractions, we hypothesize that it would be more effective as a reasoning model that operates over its own learned concepts instead of arbitrary token inputs.

##### Sparsity.

H-Net can be viewed as a form of dynamic sparsity or conditional computation, and is related to concepts such as mixture-of-experts (MoE)\parencite MoE, Original-MoE and mixture-of-depths\parencite Mixture-of-Depths. We showed that at the byte level, DC is much more effective than MoE when controlled for parameters and compute ([Figure 12](https://arxiv.org/html/2507.07955v2#S3.F12 "In Hybrid Architectures for the Main Network. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), and leave fleshing out further connections and comparisons for future work. We also note that H-Net can be viewed as orthogonal to MoE, which can be applied to sparsify any MLP layers within an H-Net.

##### Scale.

The largest models in this paper were FLOP-matched to the equivalent of a 1.3B parameter Transformer. While we believe that this provides sufficient evidence for the effectiveness of this approach, it remains to validate H-Net at larger model sizes of 3B, 7B, and beyond. We note that while we observed no instabilities at our model sizes, the added complexity of H-Net and inherent difficulties of learning end-to-end discrete selection problems may require more serious investigation of potential stability challenges at larger scale.

##### Scaling Laws.

Formally estimating the scaling behavior of a model requires calculating scaling law coefficients that sweep across a large range of model sizes and compute horizons\parencite Kaplan2020,, Chinchilla. We did not pursue this formal approach in this paper due to resource constraints.

Instead, we used a simpler heuristic for the scaling behavior of our models, at least with respect to data. We note that

*   •essentially all modern models live in the “overtrained" regime (with respect to the formal scaling laws) due to inference considerations at deployment\parencite Llama2; and 
*   •these overtrained models often use modern schedulers that have extended periods of constant learning rates\parencite WSD, DeepSeek-V3. 

Thus, we decided to use the models’ losses during the constant phase as a proxy for how quickly they improve with data. We believe this still provides useful insight into scaling behaviors, and a more dedicated analysis of formal scaling laws remains an important topic for future work.

##### BPB Calculation.

For baseline BPE tokenized models throughout this work, we used the standard bits-per-byte (BPB) calculation of simply rescaling the negative log-likelihood (or log perplexity) by the average number of bytes per token \parencite pile, MambaByte, SpaceByte. However, this is not strictly speaking a correct BPB estimate for tokenized models, as it assumes that the probability the model outputs a string is equal to the probability of the model outputting the greedy tokenization of the string.

Depending on how the model is trained, it is possible the model can output other tokenization sequences with nonzero probability. There are an exponential number of these, so computing the exact BPB is intractable; however, concurrent work \parencite vieira2024language shows that the standard BPB calculation indeed overestimates BPB. Due to the high computational overhead of estimating the true BPB, we only provide the standard (inexact) value; nevertheless, H-Net’s superior performance on downstreams provides supporting evidence that it scales better than BPE models.

5 Conclusion
------------

Major advances in deep learning have resulted from powerful architectural innovations enabling previously-handcrafted features to be learned from data, from CNNs learning visual features\parencite AlexNet to Transformers discovering linguistic patterns\parencite Transformer. H-Nets similarly unlock the ability to remove another layer of pre-processing, such as tokenizers, and instead learn them end-to-end. This ability results from a set of new techniques we introduce that work together to form a dynamic chunking mechanism, which is able to learn content- and context- dependent discrete segmentation strategies through standard gradient-based optimization. A single-stage byte-level H-Net already exceeds the performance of standard tokenized language models, and recursive H-Nets with multiple stages of dynamic chunking further improve its scaling. H-Nets substantially remedy issues with tokenizers, display very strong performance on diverse languages and language-like modalities, and more broadly may serve as the backbone of general foundation models that do _more learning_ with _less processing_.

Acknowledgements
----------------

We thank Nimit Sohoni, Eli Pugh, Justin Liu, Karan Goel, Arjun Desai, and Brandon Yang for feedback and support throughout the project. We thank Tri Dao for feedback on the ideas and earlier drafts. We thank Aakash Lahoti, Aviv Bick, and Kevin Li for feedback and discussions.

\printbibliography

Appendix A Related Work
-----------------------

The fundamental challenge of transforming raw sequential data into computationally efficient representations manifests across multiple domains through implicit chunking processes. In language modeling, this challenge is addressed through tokenization using static vocabularies derived from frequency-based algorithms such as Byte-Pair Encoding (BPE)\parencite BPE in GPT models\parencite GPT2, GPT3 and SentencePiece\parencite SentencePiece in Llama architectures\parencite Llama2, Llama3. Computer vision addresses similar challenges through spatial pooling operations\parencite U-Net that aggregate neighboring pixels into meaningful representations.

Despite achieving strong empirical performance, it is widely known that traditional tokenization approaches in language models suffer from fundamental limitations that constrain model capabilities. Fixed vocabularies exhibit biases toward high-resource languages, demonstrate fragility when handling adversarial inputs, and show lower performance on character-level tasks\parencite petrov2023language, ahia2023all, belinkov2017synthetic, Adv-BERT, ByT5. These limitations stem from the static nature of predefined vocabularies, which cannot adapt their chunking strategies to input content or context.

To address these constraints, _tokenizer-free_ methods have emerged that avoid the reliance on predefined vocabularies.

*   •In [Section A.1](https://arxiv.org/html/2507.07955v2#A1.SS1 "A.1 Autoregressive Tokenizer-free Architectures ‣ Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we discuss the most directly related prior work on autoregressive sequence models, extending the overview from [Section 1](https://arxiv.org/html/2507.07955v2#S1 "1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"). 
*   •In [Section A.2](https://arxiv.org/html/2507.07955v2#A1.SS2 "A.2 Non-Autoregressive Tokenizer-free Architectures ‣ Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we discuss non-autoregressive models. We note that essentially all autoregressive architectures can be turned into non-autoregressive architectures (including our proposed H-Net), and vice versa, which provide possible extensions of H-Net in future work. However, we provide this delineation because it marks an important difference in motivation that influences design considerations and downstream evaluations. 
*   •[Section A.3](https://arxiv.org/html/2507.07955v2#A1.SS3 "A.3 Other Tokenization-related Work ‣ Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") mentions other works in non-language modalities related to tokenization. 

We summarize our discussion on tokenizer-free architectures in [Table 6](https://arxiv.org/html/2507.07955v2#S3.T6 "In Comparison to Mixture-of-Experts. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling").

### A.1 Autoregressive Tokenizer-free Architectures

As outlined in [Section 1](https://arxiv.org/html/2507.07955v2#S1 "1 Introduction ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), prior work on autoregressive tokenizers for architectures can be divided into four categories:

1.   1.Non-hierarchical _isotropic_ architectures. 
2.   2.Hierarchical architectures with _static_ chunking strategies, where chunk boundaries are content-agnostic (usually some variant of fixed-width pooling). 
3.   3.Hierarchical architectures with _external_ chunking strategeies, where chunk boundaries are provided by an external function or module. 
4.   4.Hierarchical architectures with _dynamic_ chunking strategies, where chunk boundaries are content-dependent and learned end-to-end. 

#### A.1.1 Isotropic Architectures

The most direct approach to modeling language with tokenizers is to simply model raw byte sequences with a standard sequence model architecture. Since this naive approach suffers from computational challenges on long sequences, MambaByte \parencite MambaByte proposed using a state space model for its linear-time efficiency. We similarly use Mamba(-2) \parencite Mamba2 layers in the outer stages of an H-Net. Notably, through extensive ablations we show that Mamba is not just more efficient but also better at modeling high-resolution data such as text characters and DNA base pairs.

#### A.1.2 Static Chunking

To reduce sequence length, several approaches downsample the input sequence hierarchically. The most straightforward methods operate independently of input context, partitioning sequences using fixed-size intervals. Many strategies could be used to aggregate a width-k 𝑘 k italic_k window, including direct downsampling, average pooling, linear transformations that mix across the chunk, convolutions, and more; we lump these together as _pooling_ operations.

Hourglass Transformer\parencite HourglassTransformer and MegaByte\parencite MegaByte exemplify this strategy. Other recent variants include the Block Transformer\parencite ho2024block and Multiscale Byte Language Model (MBLM) \parencite egli2025multiscale, which use similar multi-stage static chunking architectures. Concurrently to H-Net, the MBLM also proposes using Mamba layers in the outer stages.

These approaches share conceptual similarity with spatial pooling operations in vision models that reduce resolution through fixed-window aggregation\parencite AlexNet, ResNet. While these content-agnostic methods have simple and efficient implementations, they face an inherent limitation: they do not reflect natural semantic boundaries in the data. Fixed-size chunking inevitably creates arbitrary separations that can split meaningful units such as words, morphemes, or phrases, thereby limiting model expressivity.

This class of models may also be called “autoregressive U-Nets”, characterized by the U-Net multi-scale architecture \parencite U-Net with additional considerations to maintain causality. Prior to these, the S4 and SaShiMi models \parencite S4,SaShiMi used the same architecture successfully in the vision and audio modalities, where fixed-window downsampling exhibits more appropriate inductive bias in contrast to language. SaShiMi specifically operated over 8-bit quantized audio inputs, hence also was a form of byte-level modeling that used BPB as a metric.

#### A.1.3 External Chunking

An improvement to hierarchical architectures with static downsampling is to use content-aware chunking strategies that attempt to identify natural token boundaries based on semantic or statistical properties of the input data. Several recent models propose using the boundaries provided by an external module, with two main variations appearing.

##### Delimiter-based methods.

The most intuitive content-aware approach segments on surface-level syntactical boundaries, which can be often implemented by simple rules or regular expressions.

Dynamic Pooling Transformer (DPT)\parencite DPT proposed a variant that segmented on whitespace characters, effectively making each word its own token. SpaceByte\parencite SpaceByte extends this to “space-like” delimiters (_e.g.,_/, ], :) as natural boundary signals. This approach provides semantically meaningful chunking for languages with explicit word separators such as English text and code.

However, delimiter-based methods cannot be used for inputs lacking explicit separators (e.g. many non-European languages, or other modalities such as DNA). Additionally, these approaches cannot be extended to multi-level hierarchical chunking due to ambiguities in defining natural delimiters at higher semantic levels. AU-Net \parencite AUNet is a concurrent work that augments SpaceByte with additional stages of hierarchy using fixed-width chunking. Specifically, AU-Net 2 is SpaceByte with minor architectural modifications, while AU-Net 3 (and AU-Net 4) add additional levels of hierarchical with width-2 downsampling.

In this work, we show that SpaceByte’s delimiter chunking strategy can be a very powerful baseline on appropriate languages – competitive with or outperforming traditional tokenizers on English and code – when augmented with several of H-Net’s additional techniques ([Section 3.1](https://arxiv.org/html/2507.07955v2#S3.SS1 "3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), [Section 3.3](https://arxiv.org/html/2507.07955v2#S3.SS3 "3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), [Figure 5](https://arxiv.org/html/2507.07955v2#S3.F5 "In 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), [Figure 9](https://arxiv.org/html/2507.07955v2#S3.F9 "In Encoder & Decoder Layer Selection. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

##### Entropy-based methods.

Another approach to circumvent the delimiter dependency is using the autoregressive conditional entropy as a heuristic to identify semantic boundaries. This was first proposed by the Dynamic Pooling Transformer (DPT)\parencite DPT, which detects entropy spikes that correlate with semantic transitions. The recent Byte Latent Transformer (BLT)\parencite BLT employs entropy thresholds computed by a separate pre-trained model to determine chunking boundaries.

Despite showing promise, these entropy-based approaches face several practical limitations. First, they require extensive domain-specific hyperparameter tuning to establish appropriate entropy thresholds, reducing their general applicability. Second, they still fall behind in performance; for example, BLT necessitates an extra 3B parameters (at the 8B scale) solely for multi-gram hash embeddings to match BPE Transformer baselines. Finally, these methods also cannot be extended hierarchically because computing cross-entropy loss requires access to target vocabularies, which are unavailable for intermediate latent representations in multi-stage architectures.

In this work, we do not compare against BLT because of its complexity: (i) necessitating training an auxiliary language model to provide proxy autoregressive conditional entropies (ii) converting it into an external neural tokenizer through tuning entropy heuristics (iii) using hash embeddings, which can be considered an orthogonal architectural component which may be incorporated into H-Net as well if desired.

Instead, we compared against SpaceByte (and our own stronger versions of SpaceByte), which we believe to be representative of the external-chunking family of methods and competitive to the entropy-based chunking strategy of BLT (for our main experiments such as English data).

#### A.1.4 Dynamic Chunking

The ideal tokenizer-free architecture would incorporate a _dynamic chunking method_ that attempts to learn optimal segmentation strategies directly from data through gradient-based optimization. Such a method would be optimized jointly together with the outer (fine-resolution) and inner (coarse-resolution) networks, and be able to create boundaries that are _content-_ and _context-_ aware.

The only prior work we are aware of that attempted a true dynamic chunking method is (one variant of) the Dynamic Pooling Transformer (DPT)\parencite DPT, which incorporates stochastic exploration mechanisms using Gumbel noise\parencite GumbelSoftmax,maddison2017concrete to enable differentiable boundary selection during training. Despite their theoretical flexibility, trainable methods encounter critical challenges. The stochastic exploration process requires careful tuning of noise magnitudes and introduces high-variance gradients that destabilize training, making it difficult to scale to larger model sizes.

In practice, the end-to-end (stochastic reparameterization) variant of DPT underperformed the external chunking variants (drawing boundaries on entropy spikes or whitespaces) \parencite DPT, illustrating the difficulty of this problem. Furthermore, the training instability prevented DPT from expanding to multiple hierarchical stages, constraining these methods to single-stage chunking.

We additionally highlight simple architectural modifications of DPT motivated by improved inference\parencite fleshman2023toucan or multilingual ability\parencite MAGNET. Such techniques can also be easily adapted to H-Nets in future work.

### A.2 Non-Autoregressive Tokenizer-free Architectures

Each class of autoregressive architectures from [Section A.1](https://arxiv.org/html/2507.07955v2#A1.SS1 "A.1 Autoregressive Tokenizer-free Architectures ‣ Appendix A Related Work ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") has corresponding non-autoregressive variants as well. Although these often have similar design principles, they are also motivated by different tasks, settings, and design considerations (e.g. no evaluation on large-scale autoregressive pretraining) and thus can be difficult to compare directly to autoregressive models. We include these for context and completeness.

##### Isotropic.

ByT5\parencite ByT5 directly models bytes using a bidirectional encoder-decoder architecture, showing improved performance with small models (because more power is moved into model parameters rather than vocabulary embeddings) and spelling-sensitive tasks.

##### Hierarchical (Static).

Funnel-Transformer\parencite dai2020funnel is an early architecture that uses a U-Net-like architecture for language, focusing on the non-causal setting. Canine \parencite Canine proposes a hierarchical model with convolution-based static downsampling; their method also targets non-autoregressive language models.

Charformer\parencite CharFormer presents a gradient-based subword tokenization (GBST) method that pools the input sequence at different resolutions, inducing an implicit ensemble of hierarchical models. It shows improved efficiency to performance trade-offs compared to models that use a single downsample resolution.

We note that these methods can also be endowed with implicit supervision from external tokenizers; for example, Canine proposes a variant that uses subword tokens in the _objective function_ (via masking out subwords in the masked language modeling objective), but does not need the tokenizer at inference time. We also note that such techniques are particular to non-autoregressive models, since they allow for variations in the modeling objective.

##### Hierarchical (External).

\textcite

thawani2023learn propose the eByte method, which resembles MegaByte but chunks on spaces with Transformer-based CLS-token pooling, and lacks the byte-level residual stream that enables autoregressive modeling. Word-based self-attention fusion (WSF)\parencite sreedhar2023local proposes a similar pooling strategy for encoder language models.

##### Hierarchical (Dynamic).

MANTa\parencite godey2022manta introduces an end-to-end method that predicts segmentation boundaries and pools bytes into blocks using a matching objective. MrT5\parencite kallinimrt5 is a recent method improving on ByT5 with a gating mechanism that allows for explicit dynamic token-merging at inference time, reducing sequence lengths by up to 80%.

### A.3 Other Tokenization-related Work

##### Tokenizers for Other Modalities.

While computer vision pipelines do not use tokenizers like BPE in the same way as language models do, they frequently need to turn raw perceptual data (images and videos) into shorter sequences of representations. One approach is the simple patchification step first introduced by the Vision Transformer (ViT)\parencite ViT. However, images, videos, and audio can have varying amounts of semantic content and non-uniform redundancies. A number of more recent approaches attempt to produce variable length tokenizations that adapt to the information content of the data, Which performs a more similar role to tokenization in language models. This can be done in the latent space of an autoencoder\parencite TiTok, ALIT or through explicit token merging (or "run length encoding") with heuristics\parencite ToMe, RLT. In the audio domain, SlowAE\parencite SlowAE proposes a joint autoencoder with autoregressive modeling that finds semantic segmentation boundaries, which resembles H-Net’s approach at a high level.

FAST\parencite lin2025autoregressive introduces a tokenizer for robotics, Which tokenizes continuous control actions by combining the Discrete Cosine Transform (DCT) with BPE.

##### Vocabulary Scaling.

While scaling laws for language models have generally kept tokenizers fixed\parencite ScalingLaw,Chinchilla,Llama3, recent works have showed that the tokenizer also warps scaling laws, in fact more so than model architecture changes\parencite mayilvahanan2025llms. \textcite tao2024scaling and \textcite huang2025over directly show that it is more optimal to scale an LLM’s vocabulary together with the rest of the model parameters.

In H-Nets, which are designed to operate over higher resolution raw data, the actual vocabulary can be kept minimal, but the chunking mechanism can be viewed as an implicit "tokenizer" with infinite vocabulary. As H-Nets scale in size, one expects that more iterations of hierarchy can be added (increasing effective chunk size), or the chunk size can directly be increased to leverage parameters more efficiently. This resembles the idea of increasing a vocabulary in tokenized models (which would generally increase the average length of tokens).

SuperBPE\parencite liu2025superbpe shows that allowing vocabulary tokens to cross whitespace boundaries can also improve performance. This is related to H-Net’s motivation of higher-level chunking of words into phrases; empirically, [Figure 4](https://arxiv.org/html/2507.07955v2#S3.F4 "In Visualization of Tokenized Positions. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") shows how the 2-stage H-Net finds semantic multi-word groups in the inner stage.

##### Cross-Tokenizer Transfer.

\textcite

minixhofer2024zero and \textcite minixhofer2025universal address the problem of _tokenizer transfer_, or adapting models across different tokenizers (for example for cross-language or cross-modality usage, or for knowledge distillation).

##### Other Effects of Tokenization.

\textcite

lee2024digitstodecisions discuss the effects that tokenization has on arithmetic in LLMs. For example, comparing the performance of left-to-right vs. right-to-left tokenization. \textcite hayase2024data show that examining the vocabulary of a BPE tokenizer leaks information about the data mix that it was trained on.

##### Tokenization Theory.

\textcite

schmidt2024tokenization examined the hypothesis that the primary role of tokenization is to shrink the input sequence length. They invented a new tokenizer that has even higher compression rates than BPE (actually, they keep the same vocabulary but simply find different segmentations that are more compressed) yet leads to worse language models, providing evidence against the hypothesis.

\textcite

rajaraman2024analysis showed that for certain data distributions, applying tokenization qualitatively changes what Transformers can learn.

\textcite

phan2024exact and \textcite vieira2024language propose various algorithms for converting a language model over tokens into a language model over characters or bytes. This helps alleviate some limitations of tokenizers such as the "prompt boundary" problem, the ability to compare different LLMs with different tokenizers, and simply produces better estimates of a language model’s true compressive ability (as measured by bits-per-byte). However, such algorithms are complex and expensive, and compared to direct byte-level models they are not practical for use during inference decoding (repeated autoregressive sampling).

Appendix B FLOPs Computation
----------------------------

We largely follow \textcite Chinchilla with two marginally updated computations: (1) add computations for Mamba-2\parencite Mamba2, and (2) modify computations in MLP blocks as we use the recent Transformer++ architecture. Assuming that all query, key, and value share the same num_heads and head_dim, we calculate the forward pass FLOPs as follows:

*   •Embeddings:2×seq_len×vocab_size×d_model 2 seq_len vocab_size d_model 2\times\text{seq\_len}\times\text{vocab\_size}\times\text{d\_model}2 × seq_len × vocab_size × d_model 
*   •

Attention:

    *   –Q⁢K⁢V 𝑄 𝐾 𝑉 QKV italic_Q italic_K italic_V projections:2×3×seq_len×d_model×(num_heads×head_dim)2 3 seq_len d_model num_heads head_dim 2\times 3\times\text{seq\_len}\times\text{d\_model}\times(\text{num\_heads}% \times\text{head\_dim})2 × 3 × seq_len × d_model × ( num_heads × head_dim ) 
    *   –Attention Logit Calculation:2×seq_len×seq_len×(num_heads×head_dim)2 seq_len seq_len num_heads head_dim 2\times\text{seq\_len}\times\text{seq\_len}\times(\text{num\_heads}\times\text% {head\_dim})2 × seq_len × seq_len × ( num_heads × head_dim ) 
    *   –Attention Score Softmax:3×num_heads×seq_len×seq_len 3 num_heads seq_len seq_len 3\times\text{num\_heads}\times\text{seq\_len}\times\text{seq\_len}3 × num_heads × seq_len × seq_len 
    *   –Score @ Query:2×seq_len×seq_len×(num_heads×head_dim)2 seq_len seq_len num_heads head_dim 2\times\text{seq\_len}\times\text{seq\_len}\times(\text{num\_heads}\times\text% {head\_dim})2 × seq_len × seq_len × ( num_heads × head_dim ) 
    *   –Output projection:2×seq_len×(num_heads×head_dim)×d_model 2 seq_len num_heads head_dim d_model 2\times\text{seq\_len}\times(\text{num\_heads}\times\text{head\_dim})\times% \text{d\_model}2 × seq_len × ( num_heads × head_dim ) × d_model 

*   •

Mamba-2:

    *   –X⁢Z 𝑋 𝑍 XZ italic_X italic_Z projections:2×seq_len×d_model×(2×expand×d_model)2 seq_len d_model 2 expand d_model 2\times\text{seq\_len}\times\text{d\_model}\times(2\times\text{expand}\times% \text{d\_model})2 × seq_len × d_model × ( 2 × expand × d_model ) 
    *   –B⁢C⁢Δ⁢t 𝐵 𝐶 Δ 𝑡 BC\Delta t italic_B italic_C roman_Δ italic_t projections:2×seq_len×d_model×(2×d_state+num_heads)2 seq_len d_model 2 d_state num_heads 2\times\text{seq\_len}\times\text{d\_model}\times(2\times\text{d\_state}+\text% {num\_heads})2 × seq_len × d_model × ( 2 × d_state + num_heads ) 
    *   –SSD:2×3×seq_len×(expand×d_model)×d_state 2 3 seq_len expand d_model d_state 2\times 3\times\text{seq\_len}\times(\text{expand}\times\text{d\_model})\times% \text{d\_state}2 × 3 × seq_len × ( expand × d_model ) × d_state 
    *   –Depthwise Convolution:2×seq_len×d_model×window_size 2 seq_len d_model window_size 2\times\text{seq\_len}\times\text{d\_model}\times\text{window\_size}2 × seq_len × d_model × window_size 
    *   –Gating:5×seq_len×d_model 5 seq_len d_model 5\times\text{seq\_len}\times\text{d\_model}5 × seq_len × d_model 
    *   –Output projection:2×seq_len×d_model×d_model 2 seq_len d_model d_model 2\times\text{seq\_len}\times\text{d\_model}\times\text{d\_model}2 × seq_len × d_model × d_model 

*   •

Gated MLP:

    *   –In, Gate, Out projections:2×seq_len×(3×d_model×ffw_size)2 seq_len 3 d_model ffw_size 2\times\text{seq\_len}\times(3\times\text{d\_model}\times\text{ffw\_size})2 × seq_len × ( 3 × d_model × ffw_size ) 
    *   –Gating:5×seq_len×d_model 5 seq_len d_model 5\times\text{seq\_len}\times\text{d\_model}5 × seq_len × d_model 

*   •Logit Prediction Head:2×seq_len×vocab_size×d_model 2 seq_len vocab_size d_model 2\times\text{seq\_len}\times\text{vocab\_size}\times\text{d\_model}2 × seq_len × vocab_size × d_model 

We assume the backward pass consumes twice the FLOPs of the forward pass.

Appendix C Learning Rate Modulation
-----------------------------------

As discussed in [Section 2.3](https://arxiv.org/html/2507.07955v2#S2.SS3 "2.3 Improved Techniques for Hierarchical Sequence Modeling ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we employ modulated learning rates for each stage by multiplying a scalar λ s superscript 𝜆 𝑠\lambda^{s}italic_λ start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT to the base learning rate. Empirically, we find that a reasonable set of multipliers (_e.g.,_ λ 0=2.0,λ 1=1.5,λ 2=1.0 formulae-sequence superscript 𝜆 0 2.0 formulae-sequence superscript 𝜆 1 1.5 superscript 𝜆 2 1.0\lambda^{0}=2.0,\lambda^{1}=1.5,\lambda^{2}=1.0 italic_λ start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = 2.0 , italic_λ start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT = 1.5 , italic_λ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT = 1.0) works well in general. To provide a more systematic experimental results across different architectural configurations, we follow previous works and set learning rates to be proportionally to the (1) square root of batch size\parencite malladi2022sdes,OLMo, and (2) inverse square root of hidden dimension\parencite Transformer,yang2020feature. Concretely, without heavy manual tuning, we define λ s superscript 𝜆 𝑠\lambda^{s}italic_λ start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT as follows:

λ s=N GPT⋅∏i=s S N i∏i=0 S N i⋅D S D s,N S=1.0 formulae-sequence superscript 𝜆 𝑠⋅superscript 𝑁 GPT superscript subscript product 𝑖 𝑠 𝑆 superscript 𝑁 𝑖 superscript subscript product 𝑖 0 𝑆 superscript 𝑁 𝑖 superscript 𝐷 S superscript 𝐷 𝑠 superscript 𝑁 𝑆 1.0\lambda^{s}=\sqrt{N^{\text{GPT}}\cdot\dfrac{\prod_{i=s}^{S}N^{i}}{\prod_{i=0}^% {S}N^{i}}\cdot\dfrac{D^{\text{S}}}{D^{s}}},\qquad N^{S}=1.0 italic_λ start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT = square-root start_ARG italic_N start_POSTSUPERSCRIPT GPT end_POSTSUPERSCRIPT ⋅ divide start_ARG ∏ start_POSTSUBSCRIPT italic_i = italic_s end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT italic_N start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT end_ARG start_ARG ∏ start_POSTSUBSCRIPT italic_i = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT italic_N start_POSTSUPERSCRIPT italic_i end_POSTSUPERSCRIPT end_ARG ⋅ divide start_ARG italic_D start_POSTSUPERSCRIPT S end_POSTSUPERSCRIPT end_ARG start_ARG italic_D start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT end_ARG end_ARG , italic_N start_POSTSUPERSCRIPT italic_S end_POSTSUPERSCRIPT = 1.0(11)

where N GPT superscript 𝑁 GPT N^{\text{GPT}}italic_N start_POSTSUPERSCRIPT GPT end_POSTSUPERSCRIPT is the average number of bytes per token of training dataset, which is 4.6 4.6 4.6 4.6 for the GPT-2 tokenizer on FineWeb-Edu.

We note that such principles for optimizing signal propagation as neural network hyperparameters change is an active area of research, and our scaling factors are just heuristics that can likely be improved.

Appendix D Additional Experimental Setup Details
------------------------------------------------

### D.1 Robustness Score

We introduce a metric called the _robustness score_ to measure the robustness of a model’s performance to textual perturbations, defined for Hellaswag as follows:

robustness score≔100⋅perturbed accuracy−0.25 max⁡(unperturbed accuracy−0.25,0).≔robustness score⋅100 perturbed accuracy 0.25 unperturbed accuracy 0.25 0\text{robustness score}\coloneqq 100\cdot\dfrac{\text{perturbed accuracy}-0.25% }{\max(\text{unperturbed accuracy}-0.25,0)}.robustness score ≔ 100 ⋅ divide start_ARG perturbed accuracy - 0.25 end_ARG start_ARG roman_max ( unperturbed accuracy - 0.25 , 0 ) end_ARG .

This score measures the percentage of original (unperturbed) performance that is captured by the model in the perturbed setting. We subtract by 0.25 0.25 0.25 0.25 as HellaSwag is multiple choice with 4 options, thus a model that scores 0.25 0.25 0.25 0.25 in the perturbed setting should be considered to have lost all of its original capability.

Appendix E Additional Ablation Studies
--------------------------------------

![Image 14: Refer to caption](https://arxiv.org/html/2507.07955v2/x14.png)

![Image 15: Refer to caption](https://arxiv.org/html/2507.07955v2/x15.png)

Figure 13: Compression Methods in chunking layer. Default: H-Net’s Downsample operation (left-a). Max/Mean: Channel-wise max and mean pooling within boundaries (left-b). XAttn: Cross-attention pooling within boundaries (left-c). +Res: Adds boundary vector residuals to compressed outputs. 

### E.1 Different Downsampling Methods in the Chunking Layer

Given the dynamically determined boundaries from the boundary predictor, we explore various compression strategies in the chunking layer. We compare the default Downsample operation of H-Net (see [Section 2.2.1](https://arxiv.org/html/2507.07955v2#S2.SS2.SSS1 "2.2.1 Chunking Layer ‣ 2.2 Dynamic Chunking (DC) ‣ 2 H-Net Architecture ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")) against three alternatives (see [Figure 13](https://arxiv.org/html/2507.07955v2#A5.F13 "In Appendix E Additional Ablation Studies ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-left): channel-wise max/mean pooling and cross-attention, all applied to vectors within the same boundary. Despite its simple design, the default compression in H-Net performs on-par with the other variants as demonstrated in [Figure 13](https://arxiv.org/html/2507.07955v2#A5.F13 "In Appendix E Additional Ablation Studies ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")-right. This shows that the sequence mixing layers in encoder are trained to implicitly compress necessary context into vectors at boundaries, without explicit compression mechanisms such as pooling or cross-attention.

### E.2 Details of Chinese and Code Experiments

In [Section 3.2](https://arxiv.org/html/2507.07955v2#S3.SS2 "3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we analyzed the performance of H-Net (2-stage) against Transformer and H-Net (space) on Chinese and on code, finding superior scaling for H-Net (2-stage) versus the other architectures. Here, we describe additional details from the experiment.

Besides measuring scaling behavior, we also measured final checkpoints on bits-per-byte compression ability. We also evaluated the Chinese-language models on the Chinese split of XWinograd, a Chinese language-understanding task. For model architecture, we primarily matched the settings from the GPT-3 XL, including d model subscript 𝑑 model d_{\text{model}}italic_d start_POSTSUBSCRIPT model end_POSTSUBSCRIPT and encoder/decoder architecture for H-Net models. However, we adjusted the number of layers in the main network of each model to account for slightly different compression ratios. Specifically, the Chinese-language models used a slightly higher total training flops target than the original language models, while the code models used a lower flops target. Full architecture details and results are also in [Table 4](https://arxiv.org/html/2507.07955v2#S3.T4 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling").

### E.3 DNA Architecture Ablations

H-Net (1-stage) with an M3T1 encoder achieves 3.6×3.6\times 3.6 × the data efficiency of an isotropic architecture ([Figure 6](https://arxiv.org/html/2507.07955v2#S3.F6 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). As mentioned in the caption of [Table 5](https://arxiv.org/html/2507.07955v2#S3.T5 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"), we found that an M3T1 encoder outperformed a pure Mamba-2 M4 encoder ([Table 7](https://arxiv.org/html/2507.07955v2#A5.T7 "In E.3 DNA Architecture Ablations ‣ Appendix E Additional Ablation Studies ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). Putting a Transformer in the encoder network does not appear to be helpful for text ([Figure 8](https://arxiv.org/html/2507.07955v2#S3.F8 "In DNA (Human Genome). ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). Thus, it is possible the Transformer being useful is a DNA-specific result.

Interestingly, the loss curve for the M4 encoder with a pure Mamba-2 main network was more unstable. We then also tried replacing the M15 in the main network with a T1M13T1 architecture, inspired by the finding that Transformer layers are good for dealing directly with compressed input (see [Figure 10](https://arxiv.org/html/2507.07955v2#S3.F10 "In Encoder & Decoder Layer Selection. ‣ 3.3 Ablation Studies ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). The new, principled main network architecture improved stability greatly ([Figure 14](https://arxiv.org/html/2507.07955v2#A5.F14 "In E.3 DNA Architecture Ablations ‣ Appendix E Additional Ablation Studies ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")).

Model Architecture Params.Final ppl.↓↓\downarrow↓
H-Net M3T1+T15+M4 64M 2.705
H-Net M3T1+M15+M4 66M 2.697
H-Net M4+T15+M4 62M 2.722
H-Net M4+M15+M4 64M 2.706
H-Net M4+T1M13T1+M4 64M 2.706

Table 7: Encoder architecture ablations on HG38. Switching the encoder architecture from M3T1 to M4 leads to worse performance across the board, though the results are still better than isotropic models ([Table 5](https://arxiv.org/html/2507.07955v2#S3.T5 "In Experimental setup for Chinese and code. ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")). Transformers in the encoder network do not appear to be helpful for text ([Figure 8](https://arxiv.org/html/2507.07955v2#S3.F8 "In DNA (Human Genome). ‣ 3.2 Alternate Language Datasets ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling")), suggesting that this finding may be modality-specific. 

![Image 16: [Uncaptioned image]](https://arxiv.org/html/2507.07955v2/x16.png)

Figure 14: Mamba-2-only encoder loss curves during the stable phase of training. The pure Mamba-2 model is more unstable with a loss spike. Adding Transformer layers to the main network near the DC modules can alleviate instabilities. H-Net (1-stage, principled) corresponds to the T1M13T1 main network architecture. 

Appendix F Distilling Token-Level Models to Byte-Level
------------------------------------------------------

![Image 17: Refer to caption](https://arxiv.org/html/2507.07955v2/x17.png)

Figure 15: Auxiliary loss strategy for training the encoder of a H-Net with pretrained main stage. In order to mimic the behavior of the tokenizer + embedding layer of a pretrained language model, we add supervision to both the routing module boundary probabilities and to the hidden states that we pass through to the main network. These losses encourage the encoder to tokenize once at the start of every token, while also passing the correct embedding into the main network near the start of the token, thus making maximal use of the next-token prediction ability. 

Table 8: Distilling Llama 3.2 3B to a byte level model. Average acc indicates average of the benchmarks measured in [Table 2](https://arxiv.org/html/2507.07955v2#S3.T2 "In XL Scale. ‣ 3.1 Language Modeling ‣ 3 Experiments ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"). H-Net loses performance across the board compared to the teacher, which is expected because we cannot quite replicate the exact behavior of the original model due to non-causality of BPE tokens. However, it is still much stronger than an H-Net trained from scratch on this small amount of data (189B bytes). 

Model LMB.Hella.PIQA ARC-e ARC-c Wino.Open.Average MMLU (5-shot)
acc↑↑\uparrow↑acc _n↑↑\uparrow↑acc↑↑\uparrow↑acc↑↑\uparrow↑acc _n↑↑\uparrow↑acc↑↑\uparrow↑acc _n↑↑\uparrow↑acc↑↑\uparrow↑acc↑↑\uparrow↑
Llama 3.2 3B (base)0.701 0.737 0.768 0.745 0.460 0.688 0.428 0.647 0.561
Distilled H-Net (1-stage)0.634 0.702 0.761 0.721 0.433 0.665 0.414 0.617 0.519

The role of the outer stages in H-Net is analagous to that of the tokenizer, embedding module, and LM head in a traditional BPE-based language model; together, these modules interconvert between raw text and an embedding space that the main model backbone can process. Given this similarity, we investigated whether it would be possible to convert a BPE-tokenized model directly into a byte level H-Net. To do this, we trained a 1-stage H-Net with frozen main network initialized from the backbone of Llama 3.2 3B (base). Our H-Net uses 4 Mamba-2 layers without MLPs for both the encoder and decoder with a hidden dimension of 1536 1536 1536 1536. Because the Llama model has a hidden dimension of 3072 3072 3072 3072, we add MLP adapters with hidden dimension 8192 8192 8192 8192 after chunking and right before dechunking (i.e. right before and after feeding into the main stage). We train the model for 90000 gradient steps with sequence length 8192 8192 8192 8192 and batch size 256 256 256 256, for a total of 189B bytes.

##### Aligning the encoder.

The primary difficulty in converting a tokenized model into a byte-level one is that the encoder and DC must produce chunks that the tokenized model can produce useful output with. Thus, our training (besides using the standard next-byte prediction loss), adds the following losses (see [Figure 15](https://arxiv.org/html/2507.07955v2#A6.F15 "In Appendix F Distilling Token-Level Models to Byte-Level ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling") for a visual).

1.   1.A binary cross-entropy boundary-prediction loss (with equal weight as the main loss) that operates on the routing module probabilities and targets the router to pass _the start of every real token_ through the main network. 
2.   2.A hidden state matching loss that matches the post-adapter hidden state with the “correct" hidden state. Here, if z^k subscript^𝑧 𝑘\hat{z}_{k}over^ start_ARG italic_z end_ARG start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT is the hidden representation that was passed into the main network at (byte) position t 𝑡 t italic_t, we try to match z k subscript 𝑧 𝑘 z_{k}italic_z start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT with the embedding of the token that the t 𝑡 t italic_t th byte was part of, _except_ when the t 𝑡 t italic_t h byte is the first byte of its token, in which case we match the z t subscript 𝑧 𝑡 z_{t}italic_z start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT with the _previous_ token’s embedding. Embedding matching is done with an L2 loss with a weight of 0.02. 

In the ideal case where both losses are zero, the router sends exactly the first byte of each token through to the main network with the right embedding. The main network would thus see exactly the representation it would see with a tokenizer + embedding setup. In practice, sending both losses to zero is literally impossible, as we discuss below. However, we still find that the boundary-prediction loss is crucial for learning a good matching, while the embedding-matching loss is helpful in speeding up training but not necessary. In fact, increasing the loss weight on the embedding-matching loss too much can harm the language-modeling loss.

##### Tokenization bias.

We are not able to send all auxiliary losses to zero because online prediction of BPE boundaries is an impossible task. \textcite phan2025exact coined the term “tokenization bias" to represent the fact that the tokenization process implicitly contains next-byte information. For example, the Llama 3 tokenizer tokenizes the strings _distill and _distinct into [_dist, ill] and [_distinct]. Prior use of this term has been to suggest that if an autoregressive language model is prompted with _dist, the nature of its training will be that it will never complete with inct (this is in fact a flaw of all tokenization-based models).

For us, however, tokenization bias implies that we cannot determine whether or not the i in _disti is the start of a new word until _after_ seeing the next character. In fact, the problem can be even worse–consider _applicable (becomes [_app, licable]) and _applicant (becomes [_applicant]): Determining whether l is the start of a token requires knowing the next two bytes as well.

While the H-Net does use the main network, it is not able to exactly match the behavior of the original tokenized model. Instead, it is finding slightly different representations of tokens to use in the main stage. Recent work has shown that tokenized language models can process tokenization sequences distinct from the “canonical" greedy tokenization \parencite vieira2024language, so it is possible our H-Net found another alternate representation that the pretrained model could process.

Remark. One might ask if our distilled model has simply learned to tokenize on spaces (since spaces are always the start of a new token). It has not. Simply tokenizing on spaces would yield a sub-95% boundary prediction accuracy; however, our distilled model gets boundary prediction accuracy above 99.5%. This suggests that the resulting H-Net is able to recognize some, but not all, subword boundaries.

##### Results.

The results from our distillation procedure are shown in [Table 8](https://arxiv.org/html/2507.07955v2#A6.T8 "In Appendix F Distilling Token-Level Models to Byte-Level ‣ Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"). H-Net is able to approximately match performance across almost all benchmarks; in general, H-Net is not able to replicate the behavior of the tokenized model exactly, so it is not unexpected that the benchmarks are slightly worse. Byte-Latent Transformer \parencite[Table 5]BLT performs a similar experiment, and they see a greater gap among most benchmarks (particularly PiQA, Arc-Easy, and Arc-Challenge) despite using a much larger amount of data (220B _tokens_ versus 189B _bytes_); it is possible that this performance difference is due to the fact that a BLT module cannot be supervised to align boundaries the way that end-to-end DC can.

![Image 18: Refer to caption](https://arxiv.org/html/2507.07955v2/x18.png)

Figure 16: Visualization of boundary positions dynamically drawn by H-Net (1-stage). The given text is perturbed that some whitespaces are missing. H-Net detects word boundaries even if they are not explicitly separated by whitespaces.
