Title: Towards Deleting Tokenization from Large Language Modeling

URL Source: https://arxiv.org/html/2404.14408

Markdown Content:
###### Abstract

Tokenization is widely used in large language models because it significantly improves performance. However, tokenization imposes several disadvantages, such as performance biases, increased adversarial vulnerability, decreased character-level modeling performance, and increased modeling complexity. To address these disadvantages without sacrificing performance, we propose SpaceByte, a novel byte-level decoder architecture that closes the performance gap between byte-level and subword autoregressive language modeling. SpaceByte consists of a byte-level Transformer model, but with extra larger transformer blocks inserted in the middle of the layers. We find that performance is significantly improved by applying these larger blocks only after certain bytes, such as space characters, which typically denote word boundaries. Our experiments show that for a fixed training and inference compute budget, SpaceByte outperforms other byte-level architectures and roughly matches the performance of tokenized Transformer architectures.

1 Introduction
--------------

Most language models are trained using tokenization, which partitions text into tokens that typically consist of words or subwords. Tokenization is useful because it significantly decreases the inference and training computational costs of large language models. However, tokenization also imposes several disadvantages, including performance penalties for languages that are priortizes less by the tokenizer [[1](https://arxiv.org/html/2404.14408v3#bib.bib1), [2](https://arxiv.org/html/2404.14408v3#bib.bib2), [3](https://arxiv.org/html/2404.14408v3#bib.bib3)]; increased vulnerability to adversarial attacks [[4](https://arxiv.org/html/2404.14408v3#bib.bib4)]; and worse character-level modeling performance [[5](https://arxiv.org/html/2404.14408v3#bib.bib5), [6](https://arxiv.org/html/2404.14408v3#bib.bib6)], and additional model complexity.1 1 1 See also Andrej Karpathy’s tweet [twitter.com/karpathy/status/1657949234535211009](https://twitter.com/karpathy/status/1657949234535211009) and video [youtube.com/watch?v=zduSFxRajkE&t=6725s](https://www.youtube.com/watch?v=zduSFxRajkE&t=6725s) on the disadvantages of tokenization.

Recently, MegaByte [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)], MambaByte [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)], and more [[8](https://arxiv.org/html/2404.14408v3#bib.bib8), [9](https://arxiv.org/html/2404.14408v3#bib.bib9), [10](https://arxiv.org/html/2404.14408v3#bib.bib10), [11](https://arxiv.org/html/2404.14408v3#bib.bib11), [12](https://arxiv.org/html/2404.14408v3#bib.bib12)] have been proposed as new byte-level autoregressive language models that model bytes instead of tokens. (See [[13](https://arxiv.org/html/2404.14408v3#bib.bib13), [14](https://arxiv.org/html/2404.14408v3#bib.bib14), [15](https://arxiv.org/html/2404.14408v3#bib.bib15), [16](https://arxiv.org/html/2404.14408v3#bib.bib16), [17](https://arxiv.org/html/2404.14408v3#bib.bib17), [18](https://arxiv.org/html/2404.14408v3#bib.bib18), [19](https://arxiv.org/html/2404.14408v3#bib.bib19), [20](https://arxiv.org/html/2404.14408v3#bib.bib20), [21](https://arxiv.org/html/2404.14408v3#bib.bib21)] for encoder and encoder-decoder byte-level modeling.) To address the longer context size resulting from modeling bytes instead of tokens, MegaByte uses multiscale modeling [[22](https://arxiv.org/html/2404.14408v3#bib.bib22), [23](https://arxiv.org/html/2404.14408v3#bib.bib23), [24](https://arxiv.org/html/2404.14408v3#bib.bib24)], while MambaByte uses Mamba blocks [[25](https://arxiv.org/html/2404.14408v3#bib.bib25)] instead of Transformer blocks. But although MegaByte and MambaByte have been shown to perform better than a standard byte-level Transformer, to our knowledge, no byte-level autoregressive large language model architecture has been shown to match the performance of tokenized models when controlling for compute costs.

In this work, we study the performance of byte-level and subword-level autoregressive models when trained using a fixed compute budget. We measure the performance in terms of the cross entropy (measured in bits-per-byte), which has been shown to be a strong predictor of down-stream performance [[26](https://arxiv.org/html/2404.14408v3#bib.bib26)]. In addition to controlling for training compute, we also control for inference compute costs (measured in FLOPs). We find that byte-level Transformer and MegaByte models can require roughly 10 times more training FLOPs to achieve the same performance as a subword-level Transformer. To close this substantial performance gap, we propose a new byte-level decoder architecture: SpaceByte.

SpaceByte also utilizes multiscale modeling to improve efficiency by grouping bytes into patches. But unlike MegaByte, which uses a fixed patch size, SpaceByte uses a simple rule to dynamically partition the bytes into patches that are aligned with word and other language boundaries. Our compute-controlled experiments show that this simple modification is crucial for performance, allowing SpaceByte to outperform other byte-level architectures and roughly match the performance of subword Transformers across a variety of text modalities.

Our experiments are performed on datasets consisting of English books, LaTeX formatted arXiv papers, and open-source code. For other data modalities, SpaceByte with our simple patching rule might not be as effective.

![Image 1: Refer to caption](https://arxiv.org/html/2404.14408v3/x1.png)

Figure 1:  An overview of the SpaceByte architecture. The embedding, local transformer blocks, and de-embedding (i.e. a layer norm and linear) are the standard Transformer decoder layers. SpaceByte modifies the standard transformer by applying “global” transformer blocks only after certain bytes, such as space characters. The intuition is that the first character of a word is typically the hardest to predict; thus this positioning of the global blocks should make the best use of the global blocks (which use a larger model dimension). 

2 SpaceByte
-----------

The SpaceByte architecture is summarized in Figure[1](https://arxiv.org/html/2404.14408v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"). In a nutshell, SpaceByte can be thought of as a byte-level Transformer model, but with extra “global” transformer blocks (with a larger model dimension) inserted in the middle, which are only applied a fraction of the time. While the MegaByte architecture applies the global transformer blocks every P∼8 similar-to 𝑃 8 P\sim 8 italic_P ∼ 8 bytes, we hypothesize that this fixed spacing hinders performance. Our intuition is that the first character of a word is typically significantly harder to predict than the following characters. We therefore expect that performance can be improved by applying the global blocks primarily at word boundaries.

Global Block Insertion Rule In this work, we consider a very simple rule to dynamically decide when to apply the global blocks. We assume that the text bytes are encoded using the UTF-8 encoding. We define a byte to be _spacelike_ if the byte does not encode a letter, number, or UTF-8 continuation byte 2 2 2 UTF-8 uses a variable number of bytes to encode a character. English letters or numbers consist of a single byte. Characters that are encoded using multiple bytes are encoded using a leading byte (which is spacelike by our definition) followed by one or more continuation bytes (which are not spacelike).. We apply the global blocks after any spacelike byte that is not preceded by another spacelike byte (and after any BOS token). See Figure[2](https://arxiv.org/html/2404.14408v3#S2.F2 "Figure 2 ‣ 2 SpaceByte ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for examples.

The most common spacelike byte is the space character. Thus, the global blocks are applied most frequently to predict the first character of a word, which we expect is the hardest character to predict [[27](https://arxiv.org/html/2404.14408v3#bib.bib27)] in a given word. With fixed patch size (e.g. as in MegaByte), the global blocks are typically inserted in the middle a word, which we expect is inefficient because predicting the rest of the word could likely be more efficiently accomplished using the local blocks. We define continuation bytes to be spacelike so that languages that do not use spaces between words can still benefit from the global blocks between multi-byte characters (e.g. Chinese characters consists of three bytes in UTF-8).

Although this very simple “spacelike” rule is likely not the optimal rule, we find that it works surprisingly well in practice for English text, LaTeX formatted papers, and code. Nevertheless, a critical future direction is to optimize [[28](https://arxiv.org/html/2404.14408v3#bib.bib28), [14](https://arxiv.org/html/2404.14408v3#bib.bib14), [9](https://arxiv.org/html/2404.14408v3#bib.bib9)] better rules using data rather than our simple heuristic.

PG-19:

the↓enemy!”••he exclaimed.“••Their capture must be prevented.Come with

arXiv:

where $q _ 1=q _ 2=\dots=q _\kappa$ and $V _ 1=V _ 2=\dots V _\kappa$.In this way,

Github:

exp += 2;↓↓mbf[3] = exp;↓mbf[2] = sign | (ieee[2] &0x7f);↓

Figure 2:  Examples of patch boundaries from datasets that we study. Spacelike bytes are underlined and colored blue. Patches boundaries are drawn above the text. Each patch ends after a spacelike byte that is not preceded by another spacelike byte. Consequently, each patch begins with zero or more spacelike bytes, followed by one or more non-spacelike bytes, and ends with a single spacelike byte. The global blocks predict the first character of each patch. The downward arrow (↓) denotes a newline byte. The left and right quotation characters, (“) and (”) in the PG-19 example, are encoded using three bytes in UTF-8. The first of the three bytes is spacelike, while the later two bytes are UTF-8 continuation bytes, which are not spacelike and are each denoted using a bullet point (•) above. 

Important Details Since the global blocks are not applied as often as the local transformer blocks, it is advantageous to use a larger model dimension for the global transformer blocks. To increase the dimensions of an activation vector before the global blocks, we simply pad the activation vector with zeros. To decrease the dimension, we truncate the activation vector.

In our experiments, we use a significantly larger context size than the model dimension D local subscript 𝐷 local D_{\text{local}}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT of the local transformer blocks. To prevent the attention mechanism from dominating the compute costs for the local model, we use an attention window [[29](https://arxiv.org/html/2404.14408v3#bib.bib29), [30](https://arxiv.org/html/2404.14408v3#bib.bib30), [31](https://arxiv.org/html/2404.14408v3#bib.bib31)] of length D local subscript 𝐷 local D_{\text{local}}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT for the local transformer blocks. The global blocks use a global attention that attends to all other global blocks.

See Appendix[C](https://arxiv.org/html/2404.14408v3#A3 "Appendix C Pseudocode ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for pseudocode. Additional details specific to our experiments are provided in Sections[4.1](https://arxiv.org/html/2404.14408v3#S4.SS1 "4.1 Models ‣ 4 Experiment Setup ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") and [4.2](https://arxiv.org/html/2404.14408v3#S4.SS2 "4.2 More Details ‣ 4 Experiment Setup ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") and Appendix[A](https://arxiv.org/html/2404.14408v3#A1 "Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling").

3 Related Work
--------------

The most straight-forward consequence of modeling bytes instead of subword tokens is that the length of a sequence typically increases by about a factor of four. This increased sequence length increases the training and inference compute cost for modeling a given long sequence of text for a Transformer due to the quadratic scaling of attention.

MegaByte The MegaByte architecture strives to use multiscale Transformer modeling to lessen these performance issues. In particular, MegaByte groups bytes into patches of a fixed patch size P 𝑃 P italic_P. Each patch of bytes is vectorized and then fed into a “global” Transformer model. The output of the global model is then fed into a “local” Transformer model that autoregressively outputs byte-level logits. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]

For a context size of T 𝑇 T italic_T bytes, MegaByte’s global Transformer model compresses the context into only T/P 𝑇 𝑃 T/P italic_T / italic_P patches, which can significantly decrease the compute cost for modeling long sequences. Similar to Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)], we also find that MegaByte outperforms a standard byte-level Transformer. However, we find that MegaByte’s performance is remarkably close to a stronger byte-level Transformer baseline that simply uses a sliding window attention mechanism [[29](https://arxiv.org/html/2404.14408v3#bib.bib29), [30](https://arxiv.org/html/2404.14408v3#bib.bib30), [31](https://arxiv.org/html/2404.14408v3#bib.bib31)] to increase the context size without increasing the compute costs.

Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)] do not compare MegaByte to subword-level Transformer in compute controlled experiments. In our compute controlled experiments, we find that MegaByte’s performance significantly lags behind a subword-level Transformer.

Compared to MegaByte, SpaceByte makes the crucial change that patches are dynamically sized to be commensurate with the text, e.g. with word boundaries. We also add an additional local model before the global model (while MegaByte only utilizes a single local model after the global model) to help the model deal with the dynamical patch sizes. We also use significantly longer attention windows for our local models. We find that these changes allow SpaceByte to significantly improve upon the performance of MegaByte and roughly match the performance of subword-level Transformers.

MambaByte The MambaByte architecture [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)] takes an alternative approach to avoiding the quadratic compute scaling of the attention mechanism by replacing the Transformer block with a Mamba block [[25](https://arxiv.org/html/2404.14408v3#bib.bib25)]. Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)] find that their byte-level MambaByte models outperform byte-level Transformer and byte-level MegaByte models. They perform one controlled experiment with a subword model, where they find that MambaByte slightly outperforms Mamba (using tokens) when controlling for the amount of model parameters and training data (which was 14 epochs of the PG-19 dataset). But this experiment was not controlled for compute as MambaByte was trained using roughly four times as much compute than Mamba. We view the Mamba and MambaByte architectures as complementary to our work, as the Mamba block could be integrated into SpaceByte (or MegaByte) in place of Transformer blocks.

Layer Skipping SpaceByte could be though of as a Transformer that employs a novel kind of text-dependent layer skipping [[32](https://arxiv.org/html/2404.14408v3#bib.bib32), [33](https://arxiv.org/html/2404.14408v3#bib.bib33), [34](https://arxiv.org/html/2404.14408v3#bib.bib34), [35](https://arxiv.org/html/2404.14408v3#bib.bib35), [36](https://arxiv.org/html/2404.14408v3#bib.bib36), [37](https://arxiv.org/html/2404.14408v3#bib.bib37), [38](https://arxiv.org/html/2404.14408v3#bib.bib38)] on the middle layers.

Word Boundary Prior works have shown utility in using word boundaries to partition patches for autoregressive multi-scale byte-level modeling [[9](https://arxiv.org/html/2404.14408v3#bib.bib9), [8](https://arxiv.org/html/2404.14408v3#bib.bib8), [11](https://arxiv.org/html/2404.14408v3#bib.bib11)] (and also [[18](https://arxiv.org/html/2404.14408v3#bib.bib18)] for encoder-decoder modeling). However, these works did not compare autoregressive byte-level models to subword-level models, nor did they identity a patch partitioning rule that could generically be applied to UTF-8 encoded text. Our primary contributions beyond these prior works is to show how to scale word-boundary byte-level modeling to more diverse text modalities while roughly matching the performance of subword-level models in compute-controlled experiments.

Nawrot et al. [[9](https://arxiv.org/html/2404.14408v3#bib.bib9)] and Fleshman and Van Durme [[11](https://arxiv.org/html/2404.14408v3#bib.bib11)] make use of the Hourglass Transformer architecture [[23](https://arxiv.org/html/2404.14408v3#bib.bib23)]. The SpaceByte architecture is similar to the Hourglass Transformer, except SpaceByte uses a simpler technique for shortening and upscaling the activations before and after the global blocks, and SpaceByte uses a sliding window attention [[29](https://arxiv.org/html/2404.14408v3#bib.bib29), [30](https://arxiv.org/html/2404.14408v3#bib.bib30), [31](https://arxiv.org/html/2404.14408v3#bib.bib31)] in the local blocks to improve performance for long context sizes.

4 Experiment Setup
------------------

Our experiments compare the performance of our byte-level SpaceByte architecture to subword-level Transformer and byte-level Transformer and MegaByte architectures. To fairly compare the performance between the byte and subword level models, we measure the cross-entropy of the test dataset in terms of bits-per-byte.3 3 3 The bits-per-byte (BPB) is equal to BPB=XE⁢N tokens/(N bytes⁢ln⁡2)BPB XE subscript 𝑁 tokens subscript 𝑁 bytes 2\text{BPB}=\text{XE}\,N_{\text{tokens}}/(N_{\text{bytes}}\ln 2)BPB = XE italic_N start_POSTSUBSCRIPT tokens end_POSTSUBSCRIPT / ( italic_N start_POSTSUBSCRIPT bytes end_POSTSUBSCRIPT roman_ln 2 ), i.e. the cross entropy per token (XE) times the number of tokens per byte (N tokens/N bytes)subscript 𝑁 tokens subscript 𝑁 bytes(N_{\text{tokens}}/N_{\text{bytes}})( italic_N start_POSTSUBSCRIPT tokens end_POSTSUBSCRIPT / italic_N start_POSTSUBSCRIPT bytes end_POSTSUBSCRIPT ) divided by ln⁡2 2\ln 2 roman_ln 2 (to convert nats to bits). N tokens subscript 𝑁 tokens N_{\text{tokens}}italic_N start_POSTSUBSCRIPT tokens end_POSTSUBSCRIPT and N bytes subscript 𝑁 bytes N_{\text{bytes}}italic_N start_POSTSUBSCRIPT bytes end_POSTSUBSCRIPT are the number of tokens and bytes in the dataset, respectively. For byte-level models, N tokens=N bytes subscript 𝑁 tokens subscript 𝑁 bytes N_{\text{tokens}}=N_{\text{bytes}}italic_N start_POSTSUBSCRIPT tokens end_POSTSUBSCRIPT = italic_N start_POSTSUBSCRIPT bytes end_POSTSUBSCRIPT since bytes are used in place of tokens. Given the substantial variation in inference compute costs across the models we study, we also compare their inference compute costs to provide a more comprehensive evaluation. We use FLOPs-per-byte as a simple software and hardware–independent proxy for inference compute costs, which is the number of FLOPs (see Appendix[A.1](https://arxiv.org/html/2404.14408v3#A1.SS1 "A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")) required to model a byte of text.4 4 4 We note that in order for memory bandwidth to not be a bottleneck during inference, the batch size must be sufficiently large and e.g. grouped-query attention [[39](https://arxiv.org/html/2404.14408v3#bib.bib39), [40](https://arxiv.org/html/2404.14408v3#bib.bib40)] must be used.

Note that by controlling for both total training compute and FLOPs-per-byte, we are also controlling for the amount of training data since (bytes trained)=(training FLOPs)/(training FLOPs-per-byte)bytes trained training FLOPs training FLOPs-per-byte(\text{bytes trained})=(\text{training FLOPs})/(\text{training FLOPs-per-byte})( bytes trained ) = ( training FLOPs ) / ( training FLOPs-per-byte ). The FLOPs-per-byte during training is equal to three times the FLOPs-per-byte during inference (due to the backwards pass during training).

We therefore study the Pareto frontier of lowest bits-per-byte and lowest FLOPs-per-byte. We train all models using a compute-controlled setup, using either 10 18 superscript 10 18 10^{18}10 start_POSTSUPERSCRIPT 18 end_POSTSUPERSCRIPT or 10 19 superscript 10 19 10^{19}10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT FLOPs. In order to effectively explore this Pareto frontier, we train models using a grid of different model dimensions and numbers of layers, as specified in Appendix[B.3](https://arxiv.org/html/2404.14408v3#A2.SS3 "B.3 Hyperparameter Grid ‣ Appendix B Training Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling").

Datasets Following the MegaByte [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)] and MambaByte [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)] experiments, we benchmark our models on a diverse range of long-form datasets: PG-19 (English-language books written before 1919) [[41](https://arxiv.org/html/2404.14408v3#bib.bib41)]; arXiv (papers from [ArXiv](https://arxiv.org/) written in LaTeX, extracted from the arXiv component of The Pile [[42](https://arxiv.org/html/2404.14408v3#bib.bib42)]); and Github (open-source code repositories, extracted from the Github component of The Pile [[42](https://arxiv.org/html/2404.14408v3#bib.bib42)]).

### 4.1 Models

The models we study tend to perform best when the compute cost is roughly evenly split between the attention and feedforward layers. To ensure this, we fix the context size (or attention window) to be equal to the model dimension for every layer. We detail our model setup below.

Notation For all models, we use T 𝑇 T italic_T to denote the context length, and D 𝐷 D italic_D to be the model dimension (of the global model for SpaceByte and MegaByte).

For SpaceByte and MegaByte, D local subscript 𝐷 local D_{\text{local}}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT is the dimension of the local model, and T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT is the maximum context size for the global model. The patch size P 𝑃 P italic_P is the number of bytes between global blocks. If the patch size is fixed (which is always the case for MegaByte), we naturally set the context size to be T=P⁢T global 𝑇 𝑃 subscript 𝑇 global T=PT_{\text{global}}italic_T = italic_P italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT.

Below, we describe each of the model architectures that we compare in our experiments.

SpaceByte We fix the global context size and global model dimension to be equal, T global=D subscript 𝑇 global 𝐷 T_{\text{global}}=D italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = italic_D, and we set the local attention window W local subscript 𝑊 local W_{\text{local}}italic_W start_POSTSUBSCRIPT local end_POSTSUBSCRIPT equal to the local model dimension, W local=D local subscript 𝑊 local subscript 𝐷 local W_{\text{local}}=D_{\text{local}}italic_W start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT. For the PG-19 and arXiv datasets, the average patch size is roughly 6, so we take T=6⁢T global 𝑇 6 subscript 𝑇 global T=6T_{\text{global}}italic_T = 6 italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT for these datasets; for the Github dataset, the average patch size is roughly 8, so we instead take T=8⁢T global 𝑇 8 subscript 𝑇 global T=8T_{\text{global}}italic_T = 8 italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT for the Github dataset.

For simplicity, we fix the number of global transformer blocks L global subscript 𝐿 global L_{\text{global}}italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT to be equal to the total number of local blocks, L local(1)+L local(2)superscript subscript 𝐿 local 1 superscript subscript 𝐿 local 2 L_{\text{local}}^{(1)}+L_{\text{local}}^{(2)}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT + italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 2 ) end_POSTSUPERSCRIPT, and we evenly split the number of local blocks before (L local(1)superscript subscript 𝐿 local 1 L_{\text{local}}^{(1)}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT) and after (L local(2)superscript subscript 𝐿 local 2 L_{\text{local}}^{(2)}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 2 ) end_POSTSUPERSCRIPT) the global blocks, i.e. we fix L local(1)=L local(2)=1 2⁢L global superscript subscript 𝐿 local 1 superscript subscript 𝐿 local 2 1 2 subscript 𝐿 global L_{\text{local}}^{(1)}=L_{\text{local}}^{(2)}=\tfrac{1}{2}L_{\text{global}}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 1 ) end_POSTSUPERSCRIPT = italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( 2 ) end_POSTSUPERSCRIPT = divide start_ARG 1 end_ARG start_ARG 2 end_ARG italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT.

SpaceByte (fixed patches) To clearly demonstrate the utility of dynamically aligning patch boundaries in SpaceByte, we also train a simplified version SpaceByte where the patches all have a fixed size. In order to roughly match SpaceByte’s average patch size, we take the fixed patch size to be P=6 𝑃 6 P=6 italic_P = 6 for all datasets except for the Github dataset, for which we use P=8 𝑃 8 P=8 italic_P = 8. We again use T global=D subscript 𝑇 global 𝐷 T_{\text{global}}=D italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = italic_D and T=P⁢T global 𝑇 𝑃 subscript 𝑇 global T=PT_{\text{global}}italic_T = italic_P italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT.

Byte-level Transformer For a simple baseline comparison (following Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]), we train byte-level Transformer models. We take the context size to be equal to the model dimension, T=D 𝑇 𝐷 T=D italic_T = italic_D.

Note that in our setup, a Transformer with model dimension D 𝐷 D italic_D only sees a context size of D 𝐷 D italic_D, which is significantly smaller than the context size of P⁢D 𝑃 𝐷 PD italic_P italic_D for SpaceByte (and MegaByte) with patch size P 𝑃 P italic_P.

Byte-level Transformer (Window Attention) Since a shorter context is a significant disadvantage for long-form datasets, we also compare against a stronger Transformer baseline that uses a sliding window attention [[29](https://arxiv.org/html/2404.14408v3#bib.bib29), [30](https://arxiv.org/html/2404.14408v3#bib.bib30), [31](https://arxiv.org/html/2404.14408v3#bib.bib31)] to efficiently increase the context size without increasing compute costs. We train each window attention enhanced Transformer using a context size T=P⁢D 𝑇 𝑃 𝐷 T=PD italic_T = italic_P italic_D and a sliding attention window size equal to D 𝐷 D italic_D, with P=6 𝑃 6 P=6 italic_P = 6 for all datasets except for the Github dataset for which P=8 𝑃 8 P=8 italic_P = 8.5 5 5 We also tried a simplified Sparse Transformer [[30](https://arxiv.org/html/2404.14408v3#bib.bib30)] where the first attention layer uses a sliding attention window of size D 𝐷 D italic_D; the second attention layer uses a strided attention with stride P 𝑃 P italic_P; and the remaining layers continue to alternate between sliding and strided attention. However in our setup, we found this to perform worse than just using a sliding window attention.

MegaByte We also compare to MegaByte [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]. Although MegaByte was originally trained using a patch size of P=8 𝑃 8 P=8 italic_P = 8, we found that a patch size of P=4 𝑃 4 P=4 italic_P = 4 was often better in our setup. We thus include both of these patch sizes (4 and 8) in our hyperparameter grid for MegaByte. For simplicity, we fix the number of layers in the global and local blocks to be equal, L global=L local subscript 𝐿 global subscript 𝐿 local L_{\text{global}}=L_{\text{local}}italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT, which is close to what was used by Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]. Similar to SpaceByte, we set the context size to T=P⁢D 𝑇 𝑃 𝐷 T=PD italic_T = italic_P italic_D, where D 𝐷 D italic_D is the global model dimension.

Subword Transformers Our most important baseline is the standard subword Transformer. We train subword Transformers using two different tokenizers (both with a vocabulary size of 50,257): (1) the GPT2 tokenizer [[43](https://arxiv.org/html/2404.14408v3#bib.bib43)], and (2) a SentencePiece [[44](https://arxiv.org/html/2404.14408v3#bib.bib44)] tokenizer using a byte-pair-encoding model [[45](https://arxiv.org/html/2404.14408v3#bib.bib45)] that was separately trained for each dataset. As usual, we set the context size to be equal to the model dimension, T=D 𝑇 𝐷 T=D italic_T = italic_D.

### 4.2 More Details

We use fairly standard Pre-LN [[46](https://arxiv.org/html/2404.14408v3#bib.bib46), [30](https://arxiv.org/html/2404.14408v3#bib.bib30), [47](https://arxiv.org/html/2404.14408v3#bib.bib47)] Transformer [[48](https://arxiv.org/html/2404.14408v3#bib.bib48)] blocks with no bias terms. Since MegaByte uses Rotary Position Embedding (RoPE) [[49](https://arxiv.org/html/2404.14408v3#bib.bib49)], we also use RoPE for all models (which slightly improves the loss). To prevent loss divergences during training, we use qk-layernorm [[50](https://arxiv.org/html/2404.14408v3#bib.bib50), [51](https://arxiv.org/html/2404.14408v3#bib.bib51), [52](https://arxiv.org/html/2404.14408v3#bib.bib52)] (which we strongly recommend) for all models; i.e. we add an extra layer-normalization to the query and key vectors in the self-attention layers.

All hyperparameters have been carefully tuned using grid and random searches. See Appendices[A](https://arxiv.org/html/2404.14408v3#A1 "Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") and [B](https://arxiv.org/html/2404.14408v3#A2 "Appendix B Training Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for more details.6 6 6 Our training code and data reproduction steps can be found at [github.com/kjslag/spacebyte](https://github.com/kjslag/spacebyte).

5 Results
---------

![Image 2: Refer to caption](https://arxiv.org/html/2404.14408v3/x2.png)

(a) PG-19 dataset

![Image 3: Refer to caption](https://arxiv.org/html/2404.14408v3/x3.png)

(b) arXiv dataset

![Image 4: Refer to caption](https://arxiv.org/html/2404.14408v3/x4.png)

(c) Github dataset

Figure 3: Pareto frontier of the cross-entropy bits-per-byte[3](https://arxiv.org/html/2404.14408v3#footnote3 "footnote 3 ‣ 4 Experiment Setup ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") vs FLOPs-per-byte during inference (details in Appendix[A.1](https://arxiv.org/html/2404.14408v3#A1.SS1 "A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")) for each model architecture trained using 10 18 superscript 10 18 10^{18}10 start_POSTSUPERSCRIPT 18 end_POSTSUPERSCRIPT (connected by thin lines) or 10 19 superscript 10 19 10^{19}10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT (thick lines) FLOPs on different datasets (on a log-log scale). Each dot describes a model with a different number of layers and/or model dimension. Lower and to the left is better. SpaceByte (red) outperforms all other byte-level architectures across the entire Pareto frontier for all datasets. SpaceByte roughly matches the performance of the subword Transformer using SentencePiece tokens, and outperforms the subword Transformer using GPT2 tokens. 

Table 1: Best bits-per-byte. Lowest bits-per-byte[3](https://arxiv.org/html/2404.14408v3#footnote3 "footnote 3 ‣ 4 Experiment Setup ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for each model architecture when trained using 10 19 superscript 10 19 10^{19}10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT FLOPs on different text modalities. The lowest bits-per-byte for each dataset are underlined; and the lowest within 2.5% are bolded. The largest statistical error (due to a finite number of evaluation samples) is 0.4%. SpaceByte significantly outperforms other byte-level architectures and performs on par with the SentencePiece subword Transformer. 

Model PG-19 arXiv Github
subword Transformer (GPT2 tokenizer)1.013 0.796 0.554
Transformer (SentencePiece)0.989 0.768 0.508
byte-level Transformer 1.138 0.909 0.655
Transformer (Window Attention)1.089 0.818 0.560
MegaByte 1.083 0.822 0.570
SpaceByte (fixed P 𝑃 P italic_P)1.112 0.804 0.552
SpaceByte 1.009 0.748 0.500

We now present our experimental data comparing the different model architectures in compute-controlled settings. Figure[3](https://arxiv.org/html/2404.14408v3#S5.F3 "Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") plots the Pareto frontier of lowest cross-entropy bits-per-byte and lowest FLOPs-per-byte (i.e. inference compute cost) for each architecture and training compute budget. We assume that the Pareto frontier is convex. Thus, for each architecture and compute budget, we perform a grid search over model dimension and number of layers; we then draw a piecewise-linear line connecting the best (i.e. minimal subset of) models such that all other models (not shown in figure) lie above and to the right of the line. Table[1](https://arxiv.org/html/2404.14408v3#S5.T1 "Table 1 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") summarizes the results for the lowest overall bits-per-byte for each architecture.

Across all datasets, training compute budgets, and inference compute budgets (i.e. FLOPs-per-byte), SpaceByte significantly outperforms all other byte-level architectures. SpaceByte also consistently outperforms the subword Transformer when using GPT2 tokens, and by a wide margin on the arXiv and Github datasets. SpaceByte roughly matches the performance of the most competitive baseline, the subword Transformer using the SentencePiece tokenizer, with SpaceByte performing slightly better on the arXiv and Github datasets. Figure[3](https://arxiv.org/html/2404.14408v3#S5.F3 "Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") also suggests that SpaceByte’s performance improves faster than the subword Transformer as the training compute budget increases.

Byte-level architectures other than SpaceByte perform significantly worse than SpaceByte or the SentencePiece Transformer. For example, for PG-19, the next best byte-level architecture is MegaByte; however, MegaByte trained using 10 19 superscript 10 19 10^{19}10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT FLOPs (thick green line in Figure[3a](https://arxiv.org/html/2404.14408v3#S5.F3.sf1 "In Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")) performs worse across nearly the entire Pareto frontier than the SentencePiece Transformer trained using only 10% as many training FLOPs (thin black line). Although the standard byte-level transformer (which is the primary baseline used by Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)], blue in Figure[3](https://arxiv.org/html/2404.14408v3#S5.F3 "Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")) performs significantly worse than the other byte-level models, we note that by simply using a sliding window attention mechanism to increase the context size to more closely match that of the other byte-level models, this stronger baseline (purple) performs almost as well as MegaByte. Nevertheless, SpaceByte still significantly outperforms this stronger baseline.

To verify the importance of dynamic patch sizes for SpaceByte’s performance, we compare SpaceByte to a variant of SpaceByte with fixed patch sizes (orange in Figure[3](https://arxiv.org/html/2404.14408v3#S5.F3 "Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")). We observe that fixing the patch size significantly degrades the performance of SpaceByte.

Note that on the arXiv and Github datasets, the subword Transformer performs significantly worse when using GPT2 tokens (which were trained on WebText [[43](https://arxiv.org/html/2404.14408v3#bib.bib43)]) than SentencePiece tokens (which were trained using the specific dataset). This exemplifies the bias that tokenization can introduce on data distributions different from what the tokenizer was trained on.

6 Comparison with Other Works
-----------------------------

Table 2: Comparison with other works. We compare SpaceByte to byte-level models trained in other works, along with a subword transformer that we train. All models are trained using roughly the same inference FLOPs-per-byte (≈728 absent 728\approx 728≈ 728 M). The bits-per-byte for the Transformer, PerceiverAR, and MegaByte models are taken from Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)], while MambaByte results are taken from Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)]. The best bits-per-byte for each dataset are underlined; and the lowest within 3% are bolded. The largest 1-sigma statistical error (due to a finite number of evaluation samples) for the models we train is less than 0.001. SpaceByte is the overall best performing byte-level model and consistently performs within a few percent of the subword Transformer. 

† These models used slightly different datasets for training and/or testing. For MambaByte-353M, we estimate that this difference very roughly amounts to an extra 3% statistical error. 

Model Context size Data trained Test bits-per-byte ↓↓\downarrow↓
PG-19 Stories arXiv Github
subword Transformer-1B 2048 tokens∼8192 similar-to absent 8192\sim 8192∼ 8192 bytes≈30 absent 30\approx 30≈ 30 B∗bytes 0.908 0.809 0.666 0.400
byte-level Transformer-320M [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]1024 80B 1.057 1.064 0.816†superscript 0.816†0.816^{\dagger}0.816 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT 0.575†superscript 0.575†0.575^{\dagger}0.575 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT
PerceiverAR-248M [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]8192 80B 1.104 1.070 0.791†superscript 0.791†0.791^{\dagger}0.791 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT 0.546†superscript 0.546†0.546^{\dagger}0.546 start_POSTSUPERSCRIPT † end_POSTSUPERSCRIPT
MegaByte-758M+262M [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]8192 80B 1.000 0.978 0.678†0.411†
MambaByte-353M [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)]8192 30B∗0.930 0.908†0.663†0.396†
SpaceByte-793M+184M 8192 30B∗0.918 0.833 0.663 0.411
(bytes)(bytes)

We also compare SpaceByte performance to byte-level models trained in other works. Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)] trained Transformer, PerceiverAR, and MegaByte models, each using the same amount of compute, FLOPs-per-byte, and data (80B bytes). Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)] additionally trained a MambaByte model using the same FLOPs-per-byte but only 30B bytes of data. We train SpaceByte-793M+184M (D=1536 𝐷 1536 D=1536 italic_D = 1536, D local=768 subscript 𝐷 local 768 D_{\text{local}}=768 italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = 768, L local=26 subscript 𝐿 local 26 L_{\text{local}}=26 italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = 26, L global=28 subscript 𝐿 global 28 L_{\text{global}}=28 italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = 28) using roughly the same inference FLOPs-per-byte (728M) but also only 30B bytes of data (following Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)]). Training these models thus requires roughly 3×728⁢M FLOPs-per-byte×30⁢B bytes≈6.5×10 19 3 728 M FLOPs-per-byte 30 B bytes 6.5 superscript 10 19 3\times 728\text{M FLOPs-per-byte}\times 30\text{B bytes}\approx 6.5\times 10^% {19}3 × 728 M FLOPs-per-byte × 30 B bytes ≈ 6.5 × 10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT FLOPS, where the factor of three comes from converting inference FLOPs-per-byte to training FLOPs-per-byte (which additionally requires a backwards pass). For this experiment, we set the context size of SpaceByte to 8192 bytes to follow the prior works. See Appendix[A](https://arxiv.org/html/2404.14408v3#A1 "Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for more details.

We also train subword Transformer-1B (D=1536 𝐷 1536 D=1536 italic_D = 1536) models using the SentencePiece tokenizer (except for the Stories dataset, for which we use the GPT2 tokenizer). The average number of bytes per token for the PG-19, Stories, arXiv, and Github datasets are 4.05, 4.39, 3.73, and 3.31, respectively. To match the FLOPs-per-byte of the subword Transformer-1B models to the byte-level models, we set the number of layers to 40, 44, 37, or 31, for Transformer-1B on these four respective datasets.

Results are shown in Table[2](https://arxiv.org/html/2404.14408v3#S6.T2 "Table 2 ‣ 6 Comparison with Other Works ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"). We show experiments for the PG-19 [[41](https://arxiv.org/html/2404.14408v3#bib.bib41)], Stories [[53](https://arxiv.org/html/2404.14408v3#bib.bib53)], arXiv (extracted from The Pile [[42](https://arxiv.org/html/2404.14408v3#bib.bib42)]), and Github (extracted from The Pile [[42](https://arxiv.org/html/2404.14408v3#bib.bib42)]) datasets.7 7 7 Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)] and Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)] also show results for a “Books” dataset [[42](https://arxiv.org/html/2404.14408v3#bib.bib42)] derived from Books3 (which is similar to PG-19 but also includes modern books). However, the legality of obtaining Books3 is questionable due to copyrights. We consequently exclude comparisons to this dataset.Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)] used proprietary “arXiv” and “Code” datasets, which we do not have access to. Following Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)], we compare Yu et al. [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]’s results to the similar (but likely slightly different) arXiv and Github components of The Pile [[42](https://arxiv.org/html/2404.14408v3#bib.bib42)]. However, Wang et al. [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)] use their own test splits to evaluate MambaByte-353M on Stories, arXiv, and Github. Due to the rather small test splits (∼100 similar-to absent 100\sim 100∼ 100 MB for the arXiv and Github datasets), this difference can be significant. For example, the validation (and test) bits-per-byte for SpaceByte-793M+184M on the Stories, arXiv, and Github datasets are 0.877 (0.833), 0.658 (0.663) and 0.397 (0.411), which differ by +5%percent 5+5\%+ 5 %, −1%percent 1-1\%- 1 %, and −3%percent 3-3\%- 3 %, respectively. Given this variation, the bits-per-byte of MambaByte-353M and SpaceByte-793M+184M are not statistically different on the arXiv or Github datasets.

Overall, we find that SpaceByte outperforms the byte-level models trained in other works. SpaceByte outperforms MegaByte, even though MegaByte was trained using 2.7 times as much compute and data. Moreover, SpaceByte’s performance is competitive with the subword Transformer-1B.

7 Conclusion
------------

We have proposed a new byte-level Transformer decoder architecture, SpaceByte. Our compute-controlled experiments show that SpaceByte outperforms all other byte-level architectures and roughly matches the performance of sub-word level Transformers.

Limitations SpaceByte uses a simple byte partitioning rule that relies on “spacelike” bytes, such as spaces which typically denote word boundaries. As such, SpaceByte should not be expected to perform well on arbitrary sequences of bytes, such as images or audio. Some languages, such as Chinese, do not use spaces between words. SpaceByte is somewhat robust to these languages, since e.g. Chinese characters are encoded using three bytes in UTF-8, which SpaceByte will group together. However, our preliminary experiments suggest that SpaceByte performs worse than subword transformers on Chinese text. It would therefore be desirable to improve upon and generalize SpaceByte’s global block insertion rule.

The variable spacing between global blocks makes it more challenging to design and implement an efficient batched inference sampling algorithm for SpaceByte.

Future Work SpaceByte uses multiscale modeling where the local model operates on bytes while the global model typically operates on words. Another natural extension of our work is to try recursively applying multiscale modeling at even longer scales, such as the sentence or paragraph level. It would also be fruitful to investigate if Mamba blocks [[25](https://arxiv.org/html/2404.14408v3#bib.bib25)] could further improve SpaceByte’s performance.

Acknowledgments and Disclosure of Funding
-----------------------------------------

We thank Tushaar Gangavarapu, Junxiong Wang, and Lili Yu for helpful conversations. This work was supported in part by the NSF Campus Cyberinfrastructure grant CC* Compute: Interactive Data Analysis Platform OAC-2019007 and by Rice University’s Center for Research Computing (CRC).

References
----------

*   Pourmostafa Roshan Sharami et al. [2023] J.Pourmostafa Roshan Sharami, D.Shterionov, and P.Spronck. A Systematic Analysis of Vocabulary and BPE Settings for Optimal Fine-tuning of NMT: A Case Study of In-domain Translation. March 2023. doi: 10.48550/arXiv.2303.00722. 
*   Petrov et al. [2023] Aleksandar Petrov, Emanuele La Malfa, Philip Torr, and Adel Bibi. Language model tokenizers introduce unfairness between languages. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023. URL [https://openreview.net/forum?id=78yDLKi95p](https://openreview.net/forum?id=78yDLKi95p). 
*   Ahia et al. [2023] Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David Mortensen, Noah Smith, and Yulia Tsvetkov. Do all languages cost the same? tokenization in the era of commercial language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 9904–9923, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.614. URL [https://aclanthology.org/2023.emnlp-main.614](https://aclanthology.org/2023.emnlp-main.614). 
*   Rumbelow and mwatkins [2023] Jessica Rumbelow and mwatkins. Solidgoldmagikarp (plus, prompt generation), 2023. URL [https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation). 
*   Welinder [2022] Peter Welinder, May 2022. URL [https://twitter.com/npew/status/1525900849888866307](https://twitter.com/npew/status/1525900849888866307). 
*   Wang et al. [2024] Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, and Alexander M Rush. MambaByte: Token-free Selective State Space Model. January 2024. doi: 10.48550/arXiv.2401.13660. 
*   Yu et al. [2023] Lili Yu, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis. MegaByte: Predicting Million-byte Sequences with Multiscale Transformers. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023. URL [https://openreview.net/forum?id=JTmO2V9Xpz](https://openreview.net/forum?id=JTmO2V9Xpz). 
*   Thawani et al. [2023] Avijit Thawani, Saurabh Ghanekar, Xiaoyuan Zhu, and Jay Pujara. Learn your tokens: Word-pooled tokenization for language modeling. In _Conference on Empirical Methods in Natural Language Processing_, 2023. URL [https://openreview.net/forum?id=O9zrG7NB3X](https://openreview.net/forum?id=O9zrG7NB3X). 
*   Nawrot et al. [2022] Piotr Nawrot, Jan Chorowski, Adrian Łańcucki, and Edoardo M. Ponti. Efficient Transformers with Dynamic Token Pooling. art. arXiv:2211.09761, November 2022. doi: 10.48550/arXiv.2211.09761. 
*   Lester et al. [2024] Brian Lester, Jaehoon Lee, Alex Alemi, Jeffrey Pennington, Adam Roberts, Jascha Sohl-Dickstein, and Noah Constant. Training LLMs over Neurally Compressed Text. art. arXiv:2404.03626, April 2024. 
*   Fleshman and Van Durme [2023] William Fleshman and Benjamin Van Durme. Toucan: Token-Aware Character Level Language Modeling. _arXiv e-prints_, art. arXiv:2311.08620, November 2023. doi: 10.48550/arXiv.2311.08620. 
*   Limisiewicz et al. [2024] Tomasz Limisiewicz, Terra Blevins, Hila Gonen, Orevaoghene Ahia, and Luke Zettlemoyer. MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling. _arXiv e-prints_, art. arXiv:2403.10691, March 2024. doi: 10.48550/arXiv.2403.10691. 
*   Xue et al. [2022] Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. ByT5: Towards a token-free future with pre-trained byte-to-byte models. _Transactions of the Association for Computational Linguistics_, 10:291–306, 2022. doi: 10.1162/tacl_a_00461. URL [https://aclanthology.org/2022.tacl-1.17](https://aclanthology.org/2022.tacl-1.17). 
*   Godey et al. [2022] Nathan Godey, Roman Castagné, Éric de la Clergerie, and Benoît Sagot. MANTa: Efficient Gradient-Based Tokenization for Robust End-to-End Language Modeling. art. arXiv:2212.07284, December 2022. doi: 10.48550/arXiv.2212.07284. 
*   Clark et al. [2021] Jonathan H. Clark, Dan Garrette, Iulia Turc, and John Wieting. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation. art. arXiv:2103.06874, March 2021. doi: 10.48550/arXiv.2103.06874. 
*   El Boukkouri et al. [2020] Hicham El Boukkouri, Olivier Ferret, Thomas Lavergne, Hiroshi Noji, Pierre Zweigenbaum, and Jun’ichi Tsujii. CharacterBERT: Reconciling ELMo and BERT for word-level open-vocabulary representations from characters. In _International Committee on Computational Linguistics_, pages 6903–6915, Barcelona, Spain (Online), December 2020. doi: 10.18653/v1/2020.coling-main.609. URL [https://aclanthology.org/2020.coling-main.609](https://aclanthology.org/2020.coling-main.609). 
*   Tay et al. [2021] Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon Baumgartner, Cong Yu, and Donald Metzler. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization. art. arXiv:2106.12672, June 2021. doi: 10.48550/arXiv.2106.12672. 
*   Edman et al. [2022] Lukas Edman, Antonio Toral, and Gertjan van Noord. Subword-Delimited Downsampling for Better Character-Level Translation. art. arXiv:2212.01304, December 2022. doi: 10.48550/arXiv.2212.01304. 
*   Sreedhar et al. [2023] Makesh Narsimhan Sreedhar, Xiangpeng Wan, Yu Cheng, and Junjie Hu. Local byte fusion for neural machine translation. In _Association for Computational Linguistics_, pages 7199–7214, Toronto, Canada, July 2023. doi: 10.18653/v1/2023.acl-long.397. URL [https://aclanthology.org/2023.acl-long.397](https://aclanthology.org/2023.acl-long.397). 
*   Edman et al. [2023] Lukas Edman, Gabriele Sarti, Antonio Toral, Gertjan van Noord, and Arianna Bisazza. Are Character-level Translations Worth the Wait? Comparing ByT5 and mT5 for Machine Translation. art. arXiv:2302.14220, February 2023. doi: 10.48550/arXiv.2302.14220. 
*   Cao [2023] Kris Cao. What is the best recipe for character-level encoder-only modelling? art. arXiv:2305.05461, May 2023. doi: 10.48550/arXiv.2305.05461. 
*   Subramanian et al. [2020] Sandeep Subramanian, Ronan Collobert, Marc’Aurelio Ranzato, and Y-Lan Boureau. Multi-scale Transformer Language Models. art. arXiv:2005.00581, May 2020. doi: 10.48550/arXiv.2005.00581. 
*   Nawrot et al. [2022] Piotr Nawrot, Szymon Tworkowski, Michał Tyrolski, Lukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk Michalewski. Hierarchical transformers are more efficient language models. In _Association for Computational Linguistics_, pages 1559–1571, Seattle, United States, July 2022. doi: 10.18653/v1/2022.findings-naacl.117. URL [https://aclanthology.org/2022.findings-naacl.117](https://aclanthology.org/2022.findings-naacl.117). 
*   Dai et al. [2020] Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V. Le. Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing. art. arXiv:2006.03236, June 2020. doi: 10.48550/arXiv.2006.03236. 
*   Gu and Dao [2024] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024. URL [https://openreview.net/forum?id=AL1fq05o7H](https://openreview.net/forum?id=AL1fq05o7H). 
*   Huang et al. [2024] Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. Compression Represents Intelligence Linearly. art. arXiv:2404.09937, April 2024. doi: 10.48550/arXiv.2404.09937. 
*   Al-Rfou et al. [2019] Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. Character-level language modeling with deeper self-attention. AAAI, 2019. doi: 10.1609/aaai.v33i01.33013159. URL [https://doi.org/10.1609/aaai.v33i01.33013159](https://doi.org/10.1609/aaai.v33i01.33013159). 
*   Mofijul Islam et al. [2022] Md Mofijul Islam, Gustavo Aguilar, Pragaash Ponnusamy, Clint Solomon Mathialagan, Chengyuan Ma, and Chenlei Guo. A vocabulary-free multilingual neural tokenizer for end-to-end task learning. Association for Computational Linguistics, May 2022. doi: 10.18653/v1/2022.repl4nlp-1.10. URL [https://aclanthology.org/2022.repl4nlp-1.10](https://aclanthology.org/2022.repl4nlp-1.10). 
*   Beltagy et al. [2020] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The Long-Document Transformer. April 2020. doi: 10.48550/arXiv.2004.05150. 
*   Child et al. [2019] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating Long Sequences with Sparse Transformers. April 2019. doi: 10.48550/arXiv.1904.10509. 
*   Press et al. [2021] Ofir Press, Noah A. Smith, and Mike Lewis. Shortformer: Better language modeling using shorter inputs. In _Association for Computational Linguistics_, pages 5493–5505, August 2021. doi: 10.18653/v1/2021.acl-long.427. URL [https://aclanthology.org/2021.acl-long.427](https://aclanthology.org/2021.acl-long.427). 
*   Graves [2016] Alex Graves. Adaptive Computation Time for Recurrent Neural Networks. art. arXiv:1603.08983, March 2016. doi: 10.48550/arXiv.1603.08983. 
*   Wang et al. [2017] Xin Wang, Fisher Yu, Zi-Yi Dou, Trevor Darrell, and Joseph E. Gonzalez. SkipNet: Learning Dynamic Routing in Convolutional Networks. art. arXiv:1711.09485, November 2017. doi: 10.48550/arXiv.1711.09485. 
*   Veit and Belongie [2017] Andreas Veit and Serge Belongie. Convolutional Networks with Adaptive Inference Graphs. art. arXiv:1711.11503, November 2017. doi: 10.48550/arXiv.1711.11503. 
*   Wu et al. [2017] Zuxuan Wu, Tushar Nagarajan, Abhishek Kumar, Steven Rennie, Larry S. Davis, Kristen Grauman, and Rogerio Feris. BlockDrop: Dynamic Inference Paths in Residual Networks. art. arXiv:1711.08393, November 2017. doi: 10.48550/arXiv.1711.08393. 
*   Schuster et al. [2021] Tal Schuster, Adam Fisch, Tommi Jaakkola, and Regina Barzilay. Consistent accelerated inference via confident adaptive transformers. In _Association for Computational Linguistics_, pages 4962–4979, Online and Punta Cana, Dominican Republic, November 2021. doi: 10.18653/v1/2021.emnlp-main.406. URL [https://aclanthology.org/2021.emnlp-main.406](https://aclanthology.org/2021.emnlp-main.406). 
*   Han et al. [2021] Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and Yulin Wang. Dynamic Neural Networks: A Survey. art. arXiv:2102.04906, February 2021. doi: 10.48550/arXiv.2102.04906. 
*   Raposo et al. [2024] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. art. arXiv:2404.02258, April 2024. 
*   Shazeer [2019] Noam Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. art. arXiv:1911.02150, November 2019. doi: 10.48550/arXiv.1911.02150. 
*   Ainslie et al. [2023] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In _Association for Computational Linguistics_, pages 4895–4901, Singapore, December 2023. doi: 10.18653/v1/2023.emnlp-main.298. URL [https://aclanthology.org/2023.emnlp-main.298](https://aclanthology.org/2023.emnlp-main.298). 
*   Rae et al. [2020] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In _International Conference on Learning Representations_, 2020. URL [https://openreview.net/forum?id=SylKikSYDH](https://openreview.net/forum?id=SylKikSYDH). 
*   Gao et al. [2020] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800GB Dataset of Diverse Text for Language Modeling. December 2020. doi: 10.48550/arXiv.2101.00027. 
*   Radford et al. [2019] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). 
*   Kudo and Richardson [2018] Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In _Association for Computational Linguistics_, pages 66–71, nov 2018. doi: 10.18653/v1/D18-2012. URL [https://aclanthology.org/D18-2012](https://aclanthology.org/D18-2012). 
*   Sennrich et al. [2016] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In _Association for Computational Linguistics_, pages 1715–1725, August 2016. doi: 10.18653/v1/P16-1162. URL [https://aclanthology.org/P16-1162](https://aclanthology.org/P16-1162). 
*   Baevski and Auli [2019] Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. In _International Conference on Learning Representations_, 2019. URL [https://openreview.net/forum?id=ByxZX20qFQ](https://openreview.net/forum?id=ByxZX20qFQ). 
*   Wang et al. [2019] Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep transformer models for machine translation. In _Association for Computational Linguistics_, pages 1810–1822, July 2019. doi: 10.18653/v1/P19-1176. URL [https://aclanthology.org/P19-1176](https://aclanthology.org/P19-1176). 
*   Vaswani et al. [2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_, volume 30, 2017. 
*   Su et al. [2021] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. April 2021. doi: 10.48550/arXiv.2104.09864. 
*   Henry et al. [2020] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Trevor Cohn, Yulan He, and Yang Liu, editors, _Findings of the Association for Computational Linguistics: EMNLP 2020_, pages 4246–4253, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.379. URL [https://aclanthology.org/2020.findings-emnlp.379](https://aclanthology.org/2020.findings-emnlp.379). 
*   Dehghani et al. [2023] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin F. Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Patrick Collier, Alexey Gritsenko, Vighnesh Birodkar, Cristina Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Pavetić, Dustin Tran, Thomas Kipf, Mario Lučić, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, and Neil Houlsby. Scaling Vision Transformers to 22 Billion Parameters. February 2023. doi: 10.48550/arXiv.2302.05442. 
*   Wortsman et al. [2024] Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie E Everett, Alexander A Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer training instabilities. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=d8w0pmvXbZ](https://openreview.net/forum?id=d8w0pmvXbZ). 
*   Trinh and Le [2018] Trieu H. Trinh and Quoc V. Le. A Simple Method for Commonsense Reasoning. art. arXiv:1806.02847, June 2018. doi: 10.48550/arXiv.1806.02847. 
*   Press and Wolf [2016] Ofir Press and Lior Wolf. Using the Output Embedding to Improve Language Models. August 2016. doi: 10.48550/arXiv.1608.05859. 
*   Loshchilov and Hutter [2019] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_, 2019. URL [https://openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7). 
*   Pascanu et al. [2012] Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training Recurrent Neural Networks. art. arXiv:1211.5063, November 2012. doi: 10.48550/arXiv.1211.5063. 
*   Dao et al. [2022] Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-efficient exact attention with IO-awareness. In _Advances in Neural Information Processing Systems_, 2022. URL [https://openreview.net/forum?id=H4DqfPSibmx](https://openreview.net/forum?id=H4DqfPSibmx). 
*   Dao [2024] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=mZn2Xyh9Ec](https://openreview.net/forum?id=mZn2Xyh9Ec). 
*   Levine et al. [2020] Yoav Levine, Noam Wies, Or Sharir, Hofit Bata, and Amnon Shashua. The Depth-to-Width Interplay in Self-Attention. art. arXiv:2006.12467, June 2020. doi: 10.48550/arXiv.2006.12467. 

Appendix A Model Details
------------------------

Table 3: Model hyperparameters. Hyperparameters for models shown in Table[1](https://arxiv.org/html/2404.14408v3#S5.T1 "Table 1 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"). 

Hyperparameters
Model Dataset Parameters FLOPs-per-byte L 𝐿 L italic_L(L global/L local)subscript 𝐿 global subscript 𝐿 local(L_{\text{global}}/L_{\text{local}})( italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT / italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT )D 𝐷 D italic_D(D/D local)𝐷 subscript 𝐷 local(D/D_{\text{local}})( italic_D / italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT )T 𝑇 T italic_T
Transformer(GPT2 tokenizer)PG-19 454M 279M 32 1024 1024
arXiv 253M 202M 16 1024 1024
Github 253M 253M 16 1024 1024
Transformer(SentencePiece)PG-19 454M 260M 32 1024 1024
arXiv 253M 155M 16 1024 1024
Github 253M 182M 16 1024 1024
Transformer PG-19 202M 470M 16 1024 1024
arXiv 202M 470M 16 1024 1024
Github 202M 470M 16 1024 1024
Transformer(Window Attention)PG-19 227M 529M 32 768 4608
arXiv 202M 470M 16 1024 6144
Github 202M 470M 16 1024 8192
MegaByte PG-19 201M+51M 219M 16/16 16 16 16/16 16 / 16 1024/512 1024 512 1024/512 1024 / 512 4096
arXiv 201M+51M 219M 16/16 16 16 16/16 16 / 16 1024/512 1024 512 1024/512 1024 / 512 4096
Github 201M+51M 219M 16/16 16 16 16/16 16 / 16 1024/512 1024 512 1024/512 1024 / 512 4096
SpaceByte(fixed P 𝑃 P italic_P)PG-19 201M+113M 343M 16/16 16 16 16/16 16 / 16 1024/768 1024 768 1024/768 1024 / 768 6144
arXiv 201M+113M 343M 16/16 16 16 16/16 16 / 16 1024/768 1024 768 1024/768 1024 / 768 6144
Github 201M+113M 323M 16/16 16 16 16/16 16 / 16 1024/768 1024 768 1024/768 1024 / 768 8192
SpaceByte PG-19 201M+50M 196M 16/16 16 16 16/16 16 / 16 1024/512 1024 512 1024/512 1024 / 512 6144
arXiv 201M+50M 196M 16/16 16 16 16/16 16 / 16 1024/512 1024 512 1024/512 1024 / 512 6144
Github 151M+38M 132M 12/12 12 12 12/12 12 / 12 1024/512 1024 512 1024/512 1024 / 512 8192

Table 4: Model hyperparameters. Hyperparameters for models shown in Table[2](https://arxiv.org/html/2404.14408v3#S6.T2 "Table 2 ‣ 6 Comparison with Other Works ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"). In order to roughly match the FLOPs-per-byte of the other models, for the subword-level Transformer-1B, we used 40, 44, 37, and 31 layers for the PG-19, Stories, arXiv, and Github datasets, respectively. 

Hyperparameters
Model Parameters FLOPs-per-byte L 𝐿 L italic_L(L global/L local)subscript 𝐿 global subscript 𝐿 local(L_{\text{global}}/L_{\text{local}})( italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT / italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT )D 𝐷 D italic_D(D/D local)𝐷 subscript 𝐷 local(D/D_{\text{local}})( italic_D / italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT )Others
Transformer 1B≈730 absent 730\approx 730≈ 730 M 40, 44, 37, or 31 1536 subword-level
Transformer 320M [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]732M 22 1024 byte-level
PerceiverAR 248M [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]17 1024 latents=1024 latents 1024\text{latents}=1024 latents = 1024
MegaByte 758M+262M [[7](https://arxiv.org/html/2404.14408v3#bib.bib7)]729M 14/18 14 18 14/18 14 / 18 2048/1024 2048 1024 2048/1024 2048 / 1024 P=8 𝑃 8 P=8 italic_P = 8
MambaByte 353M [[6](https://arxiv.org/html/2404.14408v3#bib.bib6)]713M 53 1024 n state=16 subscript 𝑛 state 16 n_{\text{state}}=16 italic_n start_POSTSUBSCRIPT state end_POSTSUBSCRIPT = 16
SpaceByte 793M+184M 728M 28/26 28 26 28/26 28 / 26 1536/768 1536 768 1536/768 1536 / 768 T global=1344 subscript 𝑇 global 1344 T_{\text{global}}=1344 italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = 1344

Hyperparameters for models shown in Table[1](https://arxiv.org/html/2404.14408v3#S5.T1 "Table 1 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") and Table[2](https://arxiv.org/html/2404.14408v3#S6.T2 "Table 2 ‣ 6 Comparison with Other Works ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") are summarized in Table[3](https://arxiv.org/html/2404.14408v3#A1.T3 "Table 3 ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") and Table[4](https://arxiv.org/html/2404.14408v3#A1.T4 "Table 4 ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"), respectively. In Figure[4](https://arxiv.org/html/2404.14408v3#A1.F4 "Figure 4 ‣ A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"), we show another perspective of Figure[3](https://arxiv.org/html/2404.14408v3#S5.F3 "Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") where we plot the bits-per-byte vs the bytes trained divided by the number of model parameters.

For the subword models (but not the byte-level models), we tie the input embedding weights with the output linear matrix weights [[54](https://arxiv.org/html/2404.14408v3#bib.bib54)]. In self-attention layers, we use a key dimension equal to 64. Although we apply RoPE embeddings, we also included trained position embeddings.

SpaceByte For SpaceByte, we include trained position embeddings just before the first local transformer block, and just before the first global transformer block.

Just like the other models, SpaceByte is trained using a fixed context size of T 𝑇 T italic_T bytes. At the same time, we also fix the maximum global context size of T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT patches. However, the number of patches in a given context of T 𝑇 T italic_T bytes is usually not exactly equal to T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT. To handle this mismatch during training, if the number of patches from applying the patching rule (see e.g. Figure[2](https://arxiv.org/html/2404.14408v3#S2.F2 "Figure 2 ‣ 2 SpaceByte ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")) to a context of T 𝑇 T italic_T bytes is greater T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT, then we simply ignore the bytes within these extra patches when calculating the cross entropy. Alternatively, if the number of patches is less than T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT, then we pad the activations for the global transformer blocks with zeroes and ignore the output of these global blocks. Thus, the input activations to the global blocks is always a tensor of same shape for each iteration. This discrepancy between the maximal global context size T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT and the actual number of patches results in a small fracton of wasted compute during training, which we roughly minimize by roughly tuning T/T global 𝑇 subscript 𝑇 global T/T_{\text{global}}italic_T / italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT. See Appendix[C](https://arxiv.org/html/2404.14408v3#A3 "Appendix C Pseudocode ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for pseudocode.

During inference, the model must stop predicting tokens before either the max number of bytes (T 𝑇 T italic_T) or the max number of patches (T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT) is reached.

### A.1 FLOPs

Table 5: Model non-embedding parameter counts. For MegaByte and SpaceByte, we separate the number of parameters (m 𝑚 m italic_m) into the global (m global subscript 𝑚 global m_{\text{global}}italic_m start_POSTSUBSCRIPT global end_POSTSUBSCRIPT) and local (m local subscript 𝑚 local m_{\text{local}}italic_m start_POSTSUBSCRIPT local end_POSTSUBSCRIPT) model contributions. We ignore embeddings and subleading parameters, such as layer norms, but include de-embedding parameters. See Section[A.1](https://arxiv.org/html/2404.14408v3#A1.SS1 "A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for symbol definitions. 

Architecture Parameters (non-embedding)Component
Transformer m=L×4⁢D 2 𝑚 𝐿 4 superscript 𝐷 2 m=L\times 4D^{2}italic_m = italic_L × 4 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Multi-head attention
+L×2⁢e ff⁢D 2 𝐿 2 subscript 𝑒 ff superscript 𝐷 2\quad\,+\,\,L\times 2e_{\text{ff}}D^{2}+ italic_L × 2 italic_e start_POSTSUBSCRIPT ff end_POSTSUBSCRIPT italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Feed-forward
+D⁢V 𝐷 𝑉\quad\,+\,\,DV+ italic_D italic_V De-embedding
MegaByte m global=L global×4⁢D 2 subscript 𝑚 global subscript 𝐿 global 4 superscript 𝐷 2 m_{\text{global}}=L_{\text{global}}\times 4D^{2}italic_m start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT × 4 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Global multi-head attention
+L global×2⁢e ff⁢D 2 subscript 𝐿 global 2 subscript 𝑒 ff superscript 𝐷 2\quad\quad\quad+\,L_{\text{global}}\times 2e_{\text{ff}}D^{2}+ italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT × 2 italic_e start_POSTSUBSCRIPT ff end_POSTSUBSCRIPT italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Global feed-forward
m local=D local⁢D P subscript 𝑚 local subscript 𝐷 local 𝐷 𝑃 m_{\text{local}}=D_{\text{local}}\frac{D}{P}italic_m start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT divide start_ARG italic_D end_ARG start_ARG italic_P end_ARG Global-to-local projection
+L local×4⁢D local 2 subscript 𝐿 local 4 superscript subscript 𝐷 local 2\quad\quad\;\;+\,L_{\text{local}}\times 4D_{\text{local}}^{2}+ italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT × 4 italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Local multi-head attention
+L local×2⁢e ff⁢D local 2 subscript 𝐿 local 2 subscript 𝑒 ff superscript subscript 𝐷 local 2\quad\quad\;\;+\,L_{\text{local}}\times 2e_{\text{ff}}D_{\text{local}}^{2}+ italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT × 2 italic_e start_POSTSUBSCRIPT ff end_POSTSUBSCRIPT italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Local feed-forward
+D local⁢V subscript 𝐷 local 𝑉\quad\quad\;\;+\,D_{\text{local}}V+ italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT italic_V De-embedding
SpaceByte m global=L global×4⁢D 2 subscript 𝑚 global subscript 𝐿 global 4 superscript 𝐷 2 m_{\text{global}}=L_{\text{global}}\times 4D^{2}italic_m start_POSTSUBSCRIPT global end_POSTSUBSCRIPT = italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT × 4 italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Global multi-head attention
+L global×2⁢e ff⁢D 2 subscript 𝐿 global 2 subscript 𝑒 ff superscript 𝐷 2\quad\quad\quad+\,L_{\text{global}}\times 2e_{\text{ff}}D^{2}+ italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT × 2 italic_e start_POSTSUBSCRIPT ff end_POSTSUBSCRIPT italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Global feed-forward
m local=L local×4⁢D local 2 subscript 𝑚 local subscript 𝐿 local 4 superscript subscript 𝐷 local 2 m_{\text{local}}=L_{\text{local}}\times 4D_{\text{local}}^{2}italic_m start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT × 4 italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Local multi-head attention
+L local×2⁢e ff⁢D local 2 subscript 𝐿 local 2 subscript 𝑒 ff superscript subscript 𝐷 local 2\quad\quad\;\;+\,L_{\text{local}}\times 2e_{\text{ff}}D_{\text{local}}^{2}+ italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT × 2 italic_e start_POSTSUBSCRIPT ff end_POSTSUBSCRIPT italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT Local feed-forward
+D local⁢V subscript 𝐷 local 𝑉\quad\quad\;\;+\,D_{\text{local}}V+ italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT italic_V De-embedding

Table 6: Inference FLOPs-per-token. We calculate the inference FLOPs-per-token in terms of the numbers of parameters (m 𝑚 m italic_m), shown in Table[5](https://arxiv.org/html/2404.14408v3#A1.T5 "Table 5 ‣ A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"). See Section[A.1](https://arxiv.org/html/2404.14408v3#A1.SS1 "A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for symbol definitions. 

Architecture inference FLOPs-per-token
Transformer 2⁢m+2⁢L⁢(2⁢W⁢D)2 𝑚 2 𝐿 2 𝑊 𝐷 2m+2L\,(2WD)2 italic_m + 2 italic_L ( 2 italic_W italic_D )
MegaByte 2⁢m global⁢1 P+2⁢L global⁢(2⁢T P⁢D)⁢1 P+2 subscript 𝑚 global 1 𝑃 limit-from 2 subscript 𝐿 global 2 𝑇 𝑃 𝐷 1 𝑃 2m_{\text{global}}\frac{1}{P}+2L_{\text{global}}\,(2\frac{T}{P}D)\frac{1}{P}+2 italic_m start_POSTSUBSCRIPT global end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_P end_ARG + 2 italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT ( 2 divide start_ARG italic_T end_ARG start_ARG italic_P end_ARG italic_D ) divide start_ARG 1 end_ARG start_ARG italic_P end_ARG +
2⁢m local+2⁢L local⁢(2⁢P⁢D local)2 subscript 𝑚 local 2 subscript 𝐿 local 2 𝑃 subscript 𝐷 local 2m_{\text{local}}+2L_{\text{local}}\,(2PD_{\text{local}})2 italic_m start_POSTSUBSCRIPT local end_POSTSUBSCRIPT + 2 italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT ( 2 italic_P italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT )
SpaceByte 2⁢m global⁢T global T local+2⁢L global⁢(2⁢T global⁢D)⁢T global T local+2 subscript 𝑚 global subscript 𝑇 global subscript 𝑇 local limit-from 2 subscript 𝐿 global 2 subscript 𝑇 global 𝐷 subscript 𝑇 global subscript 𝑇 local 2m_{\text{global}}\frac{T_{\text{global}}}{T_{\text{local}}}+2L_{\text{global}% }\,(2T_{\text{global}}D)\frac{T_{\text{global}}}{T_{\text{local}}}+2 italic_m start_POSTSUBSCRIPT global end_POSTSUBSCRIPT divide start_ARG italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT local end_POSTSUBSCRIPT end_ARG + 2 italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT ( 2 italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT italic_D ) divide start_ARG italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT local end_POSTSUBSCRIPT end_ARG +
2⁢m local+2⁢L local⁢(2⁢W local⁢D local)2 subscript 𝑚 local 2 subscript 𝐿 local 2 subscript 𝑊 local subscript 𝐷 local 2m_{\text{local}}+2L_{\text{local}}\,(2W_{\text{local}}D_{\text{local}})2 italic_m start_POSTSUBSCRIPT local end_POSTSUBSCRIPT + 2 italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT ( 2 italic_W start_POSTSUBSCRIPT local end_POSTSUBSCRIPT italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT )

The inference FLOPs-per-byte is the number of FLOPs required to output a byte of text during inference. We calculate the FLOPs-per-byte as the FLOPs-per-token divided by the average number of bytes per token (which is equal to 1 for byte-level models).

The FLOPs-per-token is the number of FLOPs required to output a token of text during inference (or byte of text for byte-level models). The FLOPs-per-token for the various architectures is shown in Table[6](https://arxiv.org/html/2404.14408v3#A1.T6 "Table 6 ‣ A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling").

Notation For all architectures, T 𝑇 T italic_T is the context length; D 𝐷 D italic_D is the model dimension (of the global model for SpaceByte and MegaByte); e ff=4 subscript 𝑒 ff 4 e_{\text{ff}}=4 italic_e start_POSTSUBSCRIPT ff end_POSTSUBSCRIPT = 4 is the model dimension expansion factor for feed-forward layers; and V 𝑉 V italic_V is the vocabulary size (which is 256 for byte-level models and 50257 for our subword models). For the transformer architecture, L 𝐿 L italic_L is the number of transformer blocks, and W 𝑊 W italic_W is the attention window size (which is equal to T 𝑇 T italic_T if window attention is not used). For SpaceByte and MegaByte, D local subscript 𝐷 local D_{\text{local}}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT is the dimension of the local model; L local subscript 𝐿 local L_{\text{local}}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT is the number of local transformer blocks; and L global subscript 𝐿 global L_{\text{global}}italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT is the number of global transformer blocks. For SpaceByte, T global subscript 𝑇 global T_{\text{global}}italic_T start_POSTSUBSCRIPT global end_POSTSUBSCRIPT is the maximum context size for the global model, and W local subscript 𝑊 local W_{\text{local}}italic_W start_POSTSUBSCRIPT local end_POSTSUBSCRIPT (which we set to D local subscript 𝐷 local D_{\text{local}}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT) is the attention window size for the local blocks. For MegaByte, P 𝑃 P italic_P is the patch size.

![Image 5: Refer to caption](https://arxiv.org/html/2404.14408v3/x5.png)

(a) PG-19 dataset

![Image 6: Refer to caption](https://arxiv.org/html/2404.14408v3/x6.png)

(b) arXiv dataset

![Image 7: Refer to caption](https://arxiv.org/html/2404.14408v3/x7.png)

(c) Github dataset

Figure 4: The Pareto frontier models from Figure[3](https://arxiv.org/html/2404.14408v3#S5.F3 "Figure 3 ‣ 5 Results ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling"), where we plot the bits-per-byte vs the number of bytes used for training divided by the number of non-embedding parameters (defined in Table[5](https://arxiv.org/html/2404.14408v3#A1.T5 "Table 5 ‣ A.1 FLOPs ‣ Appendix A Model Details ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling")). 

Appendix B Training Details
---------------------------

### B.1 Data

Each dataset prepared by downloaded it from Hugging Face 8 8 8[https://huggingface.co/datasets/pg19](https://huggingface.co/datasets/pg19)

[https://huggingface.co/datasets/lucadiliello/STORIES](https://huggingface.co/datasets/lucadiliello/STORIES)

[https://huggingface.co/datasets/monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted), concatenating sequences together, and separating sequences with a special BOS token. When preparing a training sample with context size T 𝑇 T italic_T, we uniformly and randomly sample a sub-sequence of length T 𝑇 T italic_T from the concatenated dataset. If a BOS token is found in this subset, we align the context with the first BOS token found; i.e. we take the context to be the first BOS token followed by the next T−1 𝑇 1 T-1 italic_T - 1 tokens in the concatenated dataset. If a BOS token is not found in the subset, we prepend a BOS token to the context. The context window is always full and always begins with a BOS token.

For the SpaceByte models, we always insert global blocks after a BOS token. A valid UTF-8 encoding never makes use of the byte values 254 or 255. We use 255 to encode the BOS token.

We train the SentencePiece tokenizers using the following command: 

`spm_train --input=train.txt --model_prefix=sp --model_type=bpe`

`--vocab_size=50257 --num_threads=32 --byte_fallback=True`

`--allow_whitespace_only_pieces=True --remove_extra_whitespaces=False`

`--normalization_rule_name=identity --input_sentence_size=10000000`

### B.2 Training

All models are trained using AdamW [[55](https://arxiv.org/html/2404.14408v3#bib.bib55)] with β 1=0.9 subscript 𝛽 1 0.9\beta_{1}=0.9 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 0.9, β 2=0.98 subscript 𝛽 2 0.98\beta_{2}=0.98 italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.98, batch size 64, weight decay of 0.01, and gradient clipping [[56](https://arxiv.org/html/2404.14408v3#bib.bib56)] with a maximum norm of 1.0. Trainable parameters are randomly initialized using a normal distribution with standard deviation σ init=1 subscript 𝜎 init 1\sigma_{\text{init}}=1 italic_σ start_POSTSUBSCRIPT init end_POSTSUBSCRIPT = 1 for all parameters except for linear weight matrices, which are initialized with standard deviation of σ init=1/d in subscript 𝜎 init 1 subscript 𝑑 in\sigma_{\text{init}}=1/\sqrt{d_{\text{in}}}italic_σ start_POSTSUBSCRIPT init end_POSTSUBSCRIPT = 1 / square-root start_ARG italic_d start_POSTSUBSCRIPT in end_POSTSUBSCRIPT end_ARG, where d in subscript 𝑑 in d_{\text{in}}italic_d start_POSTSUBSCRIPT in end_POSTSUBSCRIPT is the input dimension for the linear layer. We scale the learning rate for each parameter by its initialization standard deviation σ init subscript 𝜎 init\sigma_{\text{init}}italic_σ start_POSTSUBSCRIPT init end_POSTSUBSCRIPT.

With this setup, we found in our early hyperparameter search experiments that the optimal max learning rate for all models is approximately γ=0.005⁢B−0.5=0.000625 𝛾 0.005 superscript 𝐵 0.5 0.000625\gamma=0.005B^{-0.5}=0.000625 italic_γ = 0.005 italic_B start_POSTSUPERSCRIPT - 0.5 end_POSTSUPERSCRIPT = 0.000625, where B=64 𝐵 64 B=64 italic_B = 64 is the batch size. We therefore used γ=0.000625 𝛾 0.000625\gamma=0.000625 italic_γ = 0.000625 as the max learning rate for all models trained in this work. We applied a linear learning rate warmup over the first 1% of training iterations. We also multiply the learning rate by a “half-cosine” learning rate decay function cos⁡(π⁢x/2)𝜋 𝑥 2\cos(\pi x/2)roman_cos ( italic_π italic_x / 2 ), where 0≤x≤1 0 𝑥 1 0\leq x\leq 1 0 ≤ italic_x ≤ 1 is the fraction of training iterations completed.9 9 9 In our setup, we found cos⁡(π⁢x/2)𝜋 𝑥 2\cos(\pi x/2)roman_cos ( italic_π italic_x / 2 ) to slightly outperform the more standard cosine decay from 1 to 0.1.

Each model was trained using PyTorch on a single 40GB Nvidia A40 and A100 GPUs with mixed-precision (bfloat16 and float32) training and FlashAttention [[57](https://arxiv.org/html/2404.14408v3#bib.bib57), [58](https://arxiv.org/html/2404.14408v3#bib.bib58)]. SpaceByte-793M+184M took the longest to train, requiring about 10 days on an A100 GPU.10 10 10 We very roughly estimate that additional preliminary and failed experiments not shown in this work required roughly as many FLOPs as the experiments shown in this work.

### B.3 Hyperparameter Grid

We train models using a grid of different model dimensions and numbers of layers. In our early small-scale experiments, we found that the hyperparameter grid described below effectively explores the bits-per-byte and FLOPs-per-byte Pareto frontier for all models. To simplify the hyperparameter grid, we restrict ourselves to model dimensions and layer numbers to half-powers of two, i.e. a power of two times 1 or 3 2 3 2\frac{3}{2}divide start_ARG 3 end_ARG start_ARG 2 end_ARG.

For models trained using 10 18 superscript 10 18 10^{18}10 start_POSTSUPERSCRIPT 18 end_POSTSUPERSCRIPT FLOPs, we train model dimensions D∈{384,512,768}𝐷 384 512 768 D\in\{384,512,768\}italic_D ∈ { 384 , 512 , 768 }. For models trained using 10 19 superscript 10 19 10^{19}10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT FLOPs, we train model dimensions D∈{512,768,1024}𝐷 512 768 1024 D\in\{512,768,1024\}italic_D ∈ { 512 , 768 , 1024 }.

For SpaceByte and MegaByte, D 𝐷 D italic_D is the global model dimension. The local model dimension is chosen from D local∈{1 2⁢D,3 4⁢D}subscript 𝐷 local 1 2 𝐷 3 4 𝐷 D_{\text{local}}\in\{\frac{1}{2}D,\frac{3}{4}D\}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT ∈ { divide start_ARG 1 end_ARG start_ARG 2 end_ARG italic_D , divide start_ARG 3 end_ARG start_ARG 4 end_ARG italic_D } if D 𝐷 D italic_D is a power of two, or D local∈{1 2⁢D,2 3⁢D}subscript 𝐷 local 1 2 𝐷 2 3 𝐷 D_{\text{local}}\in\{\frac{1}{2}D,\frac{2}{3}D\}italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT ∈ { divide start_ARG 1 end_ARG start_ARG 2 end_ARG italic_D , divide start_ARG 2 end_ARG start_ARG 3 end_ARG italic_D } if D 𝐷 D italic_D is a power of two times 3 2 3 2\frac{3}{2}divide start_ARG 3 end_ARG start_ARG 2 end_ARG. However, in order to avoid excessively low FLOP utilization, we restrict D local≥256 subscript 𝐷 local 256 D_{\text{local}}\geq 256 italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT ≥ 256 (or D local≥384 subscript 𝐷 local 384 D_{\text{local}}\geq 384 italic_D start_POSTSUBSCRIPT local end_POSTSUBSCRIPT ≥ 384) for models trained using 10 18 superscript 10 18 10^{18}10 start_POSTSUPERSCRIPT 18 end_POSTSUPERSCRIPT FLOPs (or 10 19 superscript 10 19 10^{19}10 start_POSTSUPERSCRIPT 19 end_POSTSUPERSCRIPT FLOPs).

To set the number of layers, we roughly follow Levine et al. [[59](https://arxiv.org/html/2404.14408v3#bib.bib59)], who found that the compute-optimal number of layers for a Transformer roughly follows L∼12.5⁢log 2⁡(D/154)similar-to 𝐿 12.5 subscript 2 𝐷 154 L\sim 12.5\log_{2}(D/154)italic_L ∼ 12.5 roman_log start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ( italic_D / 154 ). We round this number to the nearest half-power of two to obtain L D subscript 𝐿 𝐷 L_{D}italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT, for which L 384=16 subscript 𝐿 384 16 L_{384}=16 italic_L start_POSTSUBSCRIPT 384 end_POSTSUBSCRIPT = 16, L 512=24 subscript 𝐿 512 24 L_{512}=24 italic_L start_POSTSUBSCRIPT 512 end_POSTSUBSCRIPT = 24, L 768=32 subscript 𝐿 768 32 L_{768}=32 italic_L start_POSTSUBSCRIPT 768 end_POSTSUBSCRIPT = 32, and L 1024=32 subscript 𝐿 1024 32 L_{1024}=32 italic_L start_POSTSUBSCRIPT 1024 end_POSTSUBSCRIPT = 32. For Transformer models, we choose the number of layers from L∈{1 2⁢L D,L D}𝐿 1 2 subscript 𝐿 𝐷 subscript 𝐿 𝐷 L\in\{\frac{1}{2}L_{D},L_{D}\}italic_L ∈ { divide start_ARG 1 end_ARG start_ARG 2 end_ARG italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT , italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT }.

For SpaceByte and MegaByte models, we choose the number of local and global layers from L local=L global∈{3 8⁢L D,1 2⁢L D}subscript 𝐿 local subscript 𝐿 global 3 8 subscript 𝐿 𝐷 1 2 subscript 𝐿 𝐷 L_{\text{local}}=L_{\text{global}}\in\{\frac{3}{8}L_{D},\frac{1}{2}L_{D}\}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT ∈ { divide start_ARG 3 end_ARG start_ARG 8 end_ARG italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT , divide start_ARG 1 end_ARG start_ARG 2 end_ARG italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT } if L D subscript 𝐿 𝐷 L_{D}italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT is a power of two, or L local=L global∈{1 3⁢L D,1 2⁢L D}subscript 𝐿 local subscript 𝐿 global 1 3 subscript 𝐿 𝐷 1 2 subscript 𝐿 𝐷 L_{\text{local}}=L_{\text{global}}\in\{\frac{1}{3}L_{D},\frac{1}{2}L_{D}\}italic_L start_POSTSUBSCRIPT local end_POSTSUBSCRIPT = italic_L start_POSTSUBSCRIPT global end_POSTSUBSCRIPT ∈ { divide start_ARG 1 end_ARG start_ARG 3 end_ARG italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT , divide start_ARG 1 end_ARG start_ARG 2 end_ARG italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT } if L D subscript 𝐿 𝐷 L_{D}italic_L start_POSTSUBSCRIPT italic_D end_POSTSUBSCRIPT is a power of two times 3 2 3 2\frac{3}{2}divide start_ARG 3 end_ARG start_ARG 2 end_ARG.

Appendix C Pseudocode
---------------------

See Listing[1](https://arxiv.org/html/2404.14408v3#LST1 "Listing 1 ‣ Appendix C Pseudocode ‣ SpaceByte: Towards Deleting Tokenization from Large Language Modeling") for Pytorch pseudocode for the SpaceByte forward method. The implementation of SpaceByte that we used in our experiments can be found at [github.com/kjslag/spacebyte](https://github.com/kjslag/spacebyte).

Listing 1: Pytorch pseudocode for SpaceByte

def forward(self,tokens,targets=None):

B,T=tokens.shape

T_global=self.global_context_size

D_local=self.local_model_dimension

D=self.global_model_dimension

x=self.token_embedding(tokens)

x=x+self.local_position_encoding

for block in self.initial_blocks:

x=block(x)

use_global=(

(tokens<ord(’0’))|

((ord(’9’)<tokens)&(tokens<ord(’A’)))|

((ord(’Z’)<tokens)&(tokens<ord(’a’)))|

((ord(’z’)<tokens)&(tokens<0 b1000_0000))|

(0 b1100_0000<=tokens))

use_global[:,1:]&=use_global[:,:-1].bitwise_not()

use_global|=tokens==self.BOS_token

num_global=torch.full((B,),-1)

global_idx=torch.full((B,T_global),T-1)

for b in range(B):

idx,=use_global[b].nonzero(as_tuple=True)

if targets is not None and len(idx)>T_global:

targets[b,idx[T_global]:]=-1

num_global[b]=len(idx[:T_global])

global_idx[b,:num_global[b]]=idx[:T_global]

y=x.gather(1,global_idx[:,:,None].expand(B,T_global,D_local))

y=torch.cat([torch.zeros(B,T_global,D-D_local),y],-1)

y=y+self.global_position_encoding

for block in self.global_blocks:

y=block(y)

x=torch.stack([

x[b].index_add(0,global_idx[b,:n],y[b,:n,-D_local:])

for b,n in enumerate(num_global)])

for block in self.final_blocks:

x=block(x)

logits=self.logits_linear(self.layer_norm(x))

cross_entropy_loss=None

if targets is not None:

cross_entropy_loss=torch.nn.functional.cross_entropy(

logits.view(B*T,256),targets.view(B*T),

ignore_index=-1).view(B,T)

return logits,cross_entropy_loss
