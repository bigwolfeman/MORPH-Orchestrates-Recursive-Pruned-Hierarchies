# Exclusive Self Attention

- **Authors:** Shuangfei Zhai (Apple Machine Learning Research)
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2603.09078
- **MORPH uses:** The two-line modification that excludes each token from attending to its own position in the value sum, preventing the "attention similarity bias" where softmax allocates excessive weight to the self-token and wastes capacity on identity transformation. Applied inside the local sliding-window branch of every attention layer.

---

## **Exclusive Self Attention** 

**Shuangfei Zhai** Apple `szhai@apple.com` 

## **Abstract** 

We introduce exclusive self attention (XSA), a simple modification of self attention (SA) that improves Transformer’s sequence modeling performance. The key idea is to constrain attention to capture only information orthogonal to the token’s own value vector (thus excluding information of self position), encouraging better context modeling. Evaluated on the standard language modeling task, XSA consistently outperforms SA across model sizes up to 2.7B parameters and shows increasingly larger gains as sequence length grows. 

## **1 Introduction** 

Transformers [Vaswani et al., 2017] consist of interleaved self attention (SA) and feed forward (FFN) layers, where SA aggregates information from the context, and FFN performs position wise feature updates. This SA/FFN design has remarkably stood the test of time and is continuing to serve as the building block for modern Transformer variants. 

In this work, we hypothesize that it is beneficial to further promote the division of labor between SA and FFN. We first expose a peculiar behavior of Transformers, where the output of attention tends to have a high cosine similarity with the self value vector – and we call it the _**attention similarity bias**_ . 

The prevalence of the _attention similarity bias_ suggests that SA spends a significant portion of its capacity modeling the point wise feature transformation. This is on the one hand unnecessary, because the information of the current position has a direction residual path to the following FFN layer; and on the other hand harmful, because it creates a competition between modeling the contextual vs point-wise feature. This reasoning directly motivates our solution: exclusive self attention (XSA), which explicitly excludes directions from the attention’s output along that of the self value vector. 

We evaluate XSA against standard Transformers on the language modeling task, and we show that XSA 1) introduces minimal computational overhead; 2) achieves better training/validation loss across three model sizes; 3) achieves better downstream evaluation results; 4) maintains consistent gains across different learning rates; 5) shows larger gains as sequence length increases; and 6) is robust w.r.t. the use of attention sinks. 

## **2 Motivation** 

We first define a standard (causal) self attention (SA) as _y_ = _f_ ( _x_ ): 

**==> picture [367 x 31] intentionally omitted <==**

where _Wq, Wk, Wv_ are the query, key and value projections, respectively. 

We next demonstrate the _attention similarity bias_ phenomenon, which exposes a hidden problem of SA. To see this, we take a trained language model (which has 1.3B parameters and 2048 sequence 

Technical report. 

Figure 1: Visualization of the _**attention similarity bias**_ of a 1.3B parameter language model of sequence length 2048 trained for 100B tokens, aggregated on 1024 random training sequences. **Left** : the average cosine similarity of value vectors _vi_ , _vj_ within a sequence; **middle** : the average diagonal attention value _ai,j_ ; **right** : the average cosine similarity of attention output _yi_ and the self value vector _vi_ . See Eq. 1 for notations. 

length, see Sect. 4 for details) and analyze each of its attention layers in Figure 1. Specifically, we plot three quantities: 1) the average cosine similarity of value vectors _< vi, vj >_ within the same head and sequence; 2) the average diagonal values of the attention _ai,i_ ; and 3) the average cosine similarity of the aggregated values _yi_ with the corresponding value _vi_ . All quantities are then averaged across attention heads and 1024 random training sequences and plotted for each layer. We can see that 1) value vectors tend to be positively correlated and 2) attention scores to the current position are relatively high. As a direct consequence, there is a high average _< yi, vi >_ , with an increasing trend w.r.t. the layer index. This suggests that standard SA tends to aggregate value vectors similar to what the self value _vi_ already encodes, which implicitly overlaps with role of FFN and consequently diminishes SA’s goal of context modeling. 

## **3 Method** 

We define exclusive self attention (XSA) as _z_ = _f_ ( _x_ ): 

**==> picture [367 x 56] intentionally omitted <==**

Note that the first line of Equation 2 corresponds to standard SA in Equation 1; XSA introduces an additional step to remove the projection of SA’s output _yi_ on self value vector _vi_ . Consequently, XSA’s output _zi_ no longer contains _vi_ itself, nor components from that context that’s correlated with _vi_ . Therefore, XSA completely removes the _attention similarity bias_ . 

Our core hypothesis is that, in the presence of residual connections and the FFN block, XSA 1) maintains the expressiveness of standard SA and 2) promotes modeling efficiency by allowing the attention layer to exclusively focus on contextual information. While it is also possible to provide theoretical groundings for it, in this work we defer to empirical evaluations as the main justification. 

XSA can be implemented with two lines of code change on top of SA, as demonstrated in Algorithm 1. 

## **4 Experiments** 

## **4.1 Setup** 

**Codebase** We conduct all experiments with the NanoGPT[1] codebase due to its ease of reproducibility. A few changes are made to the Transformer implementation: we replaced the learned position embeddings with RoPE [Su et al., 2023] whish is a common practice in modern language models; we 

> 1https://github.com/karpathy/nanoGPT 

2 

**Algorithm 1** PyTorch-style pseudocode for multi-head causal XSA 

```
#x:(B,T,D)
#Wq,Wk,Wv,Wo:(D,D)
#H:numberofheads
defexclusive_self_attention(x,Wq,Wk,Wv,Wo,H):
B,T,D=x.shape
#linearprojections
Q=(x@Wq).reshape(B,T,H,D//H).transpose(1,2)
K=(x@Wk).reshape(B,T,H,D//H).transpose(1,2)
V=(x@Wv).reshape(B,T,H,D//H).transpose(1,2)
#standardmulti-headattention
Y=torch.nn.functional.scaled_dot_product_attention(Q,K,V,is_causal=True)
#XSAmode
Vn=torch.nn.functional.normalize(V,dim=-1)
Z=Y-(Y*Vn).sum(dim=-1,keepdim=True)*Vn
#outputprojection
out=Z.transpose(1,2).reshape(B,T,D)@Wo
returnout
```

Table 1: Architectures and learning rates. All models are trained with a batch size of 0 _._ 5 _M_ tokens, for a total of 200K iterations. 

|Model Size|_n_layers|_d_model|_n_heads|_d_head|Learning Rate|
|---|---|---|---|---|---|
|0.7B|24|1536|6|256|5_._0_×_10_−_4|
|1.4B|24|2048|24|128|4_._0_×_10_−_4|
|2.7B|32|2560|24|128|3_._0_×_10_−_4|



insert an additional LayerNorm [Ba et al., 2016] right after the token embeddings which is found to improve training stability; we allow the number of attention heads and the head dimension to be configured independently to allow for more flexible model sizes. 

**Dataset** We use the FineWeb-100BT [Penedo et al., 2024][2] dataset which contains _∼_ 100 billion tokens. We preprocess the dataset following the NanoGPT protocol: namely tokenizing it with the GPT2 [Radford et al., 2019] tokenizer and randomly splitting 0 _._ 05% of the tokens as the validation set. 

**Training details** We closely follow the default training settings of NanoGPT, with a few modifications detailed below. We adopt a default context length of 2048, a global batch size of 256, and a training duration of 200K iterations. This amounts to 100B training tokens which is roughly one epoch over the training set. AdamW [Loshchilov and Hutter, 2017] is used with a linear learning rate warmup of 2K steps, followed by a cosine decay schedule to 101[x][of][the][max][learning][rate.] We perform a grid search of learning rate for each baseline model configuration and use it for the XSA variants. We experiment with 3 model sizes, ranging in 0.7B, 1.4B and 2.7B non-embedding parameters. The model configurations are specified in Table 1. 

## **4.2 Results** 

**Computational overhead** We first benchmark XSA in terms of its speed and memory efficiency. We run evaluations of an attention block (attention + FFN) on varied sequence lengths and model 

> 2https://huggingface.co/datasets/HuggingFaceFW/fineweb 

3 

Figure 2: Time and memory efficiency of XSA compared to standard attention. XSA introduces minimal computational overhead across various sequence lengths and model sizes _dmodel_ . 

**==> picture [382 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
3.2 3.2<br>0.7b_baseline  0.7b_baseline<br>3.1 0.7b_xsa  3.1 0.7b_xsa<br>1.3b_baseline  1.3b_baseline<br>1.3b_xsa  1.3b_xsa<br>3.0 2.7b_baseline  3.0 2.7b_baseline<br>2.7b_xsa  2.7b_xsa<br>2.9 2.9<br>2.8 2.8<br>2.7 2.7<br>2.6 2.6<br>2.5 2.5<br>Ne |<br>2.4 2.4<br>0 25 50 75 100 125 150 175 200 0 25 50 75 100 125 150 175 200<br>Training iteration (K) Training iteration (K)<br>Training loss Validation loss<br>**----- End of picture text -----**<br>


Figure 3: Training and validation loss curves of XSA against the baseline Transformer for three model sizes. 

Table 2: Downstream evaluation results of XSA vs baseline Transformer. 

|**Model**|**ARC-E**|**BoolQ**|**HSwag**|**LAMBADA**|**OBQA**|**PIQA**|**SocIQA**|**WinoGr**|**Avg**|∆**Avg**|
|---|---|---|---|---|---|---|---|---|---|---|
|**_0.7B_**|||||||||||
|Baseline<br>XSA|51.26<br>**52.69**|61.07<br>**61.19**|55.68<br>**56.29**|52.82<br>**54.07**|**35.00**<br>32.20|**74.05**<br>73.78|40.02<br>**41.45**|55.88<br>**56.20**|53.22<br>**53.48**|**+0.26**|
|**_1.3B_**|||||||||||
|Baseline<br>XSA|56.19<br>**58.84**|**65.47**<br>62.29|60.69<br>**62.41**|56.24<br>**58.57**|34.60<br>**36.00**|75.90<br>**76.61**|41.40<br>**42.84**|58.80<br>**59.98**|56.16<br>**57.19**|**+1.03**|
|**_2.7B_**|||||||||||
|Baseline<br>XSA|58.59<br>**60.65**|60.98<br>**64.86**|66.20<br>**67.40**|60.18<br>**62.04**|37.00<br>**38.40**|76.61<br>**77.80**|**42.94**<br>41.45|61.96<br>**62.75**|58.06<br>**59.42**|**+1.36**|



sizes, while switching XSA on and off. The experiments are ran on a B200 GPU with a batch size 

4 

32 and numerical precision of bfloat16. The results are shown in Figure 2. We see that, as expected, XSA introduces minimal overhead in terms of both speed and memory. 

**Model size** We next show the training and validation loss curves for the three model sizes in Figure 3. We observe that through the course of training XSA maintains a clear margin over the baseline in all three model sizes, across both training and validation. Furthermore, we conduct evaluations of the final checkpoints on 8 downstream tasks: ARC-Easy [Clark et al., 2018], BoolQ [Clark et al., 2018], HellaSwag [Zellers et al., 2019], LAMBADA [Paperno et al., 2016], OpenBookQA [Mihaylov et al., 2018], PIQA [Ba et al., 2016], SocialIQA [Sap et al., 2019] and WinoGrande [Sakaguchi et al., 2021], which cover language, knowledge and reasoning aspects of the models. We run all evaluations with the Language Model Evaluation Harness [Gao et al., 2024], and report the results w.r.t. accuracy (for BoolQ, LAMBADA, SocialIQA and WinoGrande) or length normalized accuracy (for ARC-Easy, HellaSwag, OpenBookQA and PIQA) in Table 2. We see that across three model sizes, XSA consistently outperforms the baseline Transformer in terms of the average accuracy, with a larger margin as the model size increases. Based on these observations, we speculate that XSA will remain advantageous in even larger scale training settings, both in terms of model size and training data size. 

**Learning rate** It is also important to make sure that the performance gain of XSA holds across different learning rates. In order to test it, we select the 1.3B model and compare XSA with the baseline with four different learning rates. The comparison is shown in Figure 4. Again, we see that there is a near constant margin across all learning rates, demonstrating the robustness of the XSA architecture. 

**==> picture [392 x 139] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.62 2.62<br>baseline baseline<br>2.60 xsa 2.60 xsa<br>2.58 2.58<br>2.56 2.56<br>2.54 2.54<br>2.52 2.52<br>2.50 2.50<br>1e-4 2e-4 4e-4 6e-4 1e-4 2e-4 4e-4 6e-4<br>Learning rate Learning rate<br>Training loss Validation loss<br>**----- End of picture text -----**<br>


Figure 4: Training and validation loss of XSA against the baseline Transformer for various learning rates evaluated with the 1.3B model. 

**==> picture [392 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.70 2.70<br>baseline baseline<br>2.65 xsa 2.65 xsa<br>2.60 2.60<br>2.55 2.55<br>2.50 2.50<br>2.45 2.45<br>512 1024 2048 4096 8192 16384 512 1024 2048 4096 8192 16384<br>Seq length Seq length<br>Training loss Validation loss<br>**----- End of picture text -----**<br>


Figure 5: Training and validation loss of XSA against the baseline Transformer for various sequence lengths evaluated with the 1.3B model. 

5 

**==> picture [392 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.58 2.58<br>baseline baseline<br>2.57 xsa 2.57 xsa<br>2.56 2.56<br>2.55 2.55<br>2.54 2.54<br>2.53 2.53<br>2.52 2.52<br>0 1 4 0 1 4<br>Num attn sinks Num attn sinks<br>Training loss Validation loss<br>**----- End of picture text -----**<br>


Figure 6: Training and validation loss of XSA against the baseline Transformer for various number of Attention Sinks evaluated with the 1.3B model. 

**Sequence length** Next, we evaluate XSA’s compatibility with long contexts. We again use the 1.3B model and train on six different sequence lengths, ranging in _{_ 512 _,_ 1024 _,_ 2048 _,_ 4096 _,_ 8192 _,_ 16384 _}_ . We use the same learning rate for all settings, and adjust the batch size such that the number of tokens per batch remains constant (0.5M). The result is shown in Figure 5. Interestingly, XSA claims larger gains as sequence length increases. We suspect that this is due to the increasing tension on context modeling for longer sequences, which makes the benefit of XSA more pronounced. This also suggests that XSA is a promising technique for long context modeling, one of the critical problems of scaling Transformers. 

**Comparison to Attention Sink** Incidentally, XSA is also related to Attention Sink [Xiao et al., 2023]. Instead of explicitly prepending to the sequence a set of learned sink tokens, XSA is able to allocate undesired attention scores to _ai,i_ . Therefore, XSA can be viewed as an implicit attention sink, and it would be interesting to compare it with standard Attention Sinks. We ran additional experiments by including learned attention sinks for both the baseline and XSA models, and report the results in Figure 6. We see that XSA maintains the loss margin in the existence of attention sinks. 

## **5 Discussions** 

We have shown the exclusive self attention (XSA) demonstrates promising performance on standard language modeling tasks. However, many questions remain: How does it work at even larger scale w.r.t. model and data? Is it compatible with other optimizers such as Muon [Jordan et al., 2024]? Does it work for other tasks/modalities besides language modeling? We hope that this study inspires future works that can properly answer these questions. 

## **References** 

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _arXiv preprint arXiv:1607.06450_ , 2016. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL `https://zenodo.org/records/12608602` . 

- Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL `https: //kellerjordan.github.io/posts/muon/` . 

6 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _Proceedings of the 2018 conference on empirical methods in natural language processing_ , pages 2381–2391, 2018. 

- Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In _Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: Long papers)_ , pages 1525–1534, 2016. 

- Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In _The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2024. URL `https://openreview.net/forum?id= n6SCkn2QaG` . 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_ , 64(9):99–106, 2021. 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In _Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)_ , pages 4463–4473, 2019. 

- J Su, Y Lu, S Pan, A Murtadha, B Wen, and Y Liu Roformer. Enhanced transformer with rotary position embedding., 2021. _DOI: https://doi. org/10.1016/j. neucom_ , 2023. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. _arXiv preprint arXiv:2309.17453_ , 2023. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th annual meeting of the association for computational linguistics_ , pages 4791–4800, 2019. 

7 

