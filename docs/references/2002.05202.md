# GLU Variants Improve Transformer

- **Authors:** Noam Shazeer (Google)
- **Year:** 2020
- **Source:** https://arxiv.org/abs/2002.05202
- **MORPH uses:** SwiGLU (Swish-gated linear unit) as the MLP activation function in all feed-forward sublayers - FFN(x) = (xW1 (.) Swish(xV)) . W2 - providing a gated nonlinearity that consistently outperforms GELU/ReLU variants on perplexity at equal parameter count.

---

**GLU Variants Improve Transformer** 

**==> picture [470 x 5] intentionally omitted <==**

Noam Shazeer Google noam@google.com 

February 14, 2020 

## **Abstract** 

Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feedforward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations. 

## **1 Introduction** 

The Transformer [Vaswani et al., 2017] sequence-to-sequence model alternates between multi-head attention, and what it calls "position-wise feed-forward networks" (FFN). The FFN takes a vector _x_ (the hidden representation at a particular position in the sequence) and passes it through two learned linear transformations, (represented by the matrices _W_ 1 and _W_ 2 and bias vectors _b_ 1 and _b_ 2). A rectified-linear (ReLU) [Glorot et al., 2011] activation function applied between the two linear transformations. 

**==> picture [346 x 11] intentionally omitted <==**

Following the T5 codebase [Raffel et al., 2019][1] , we use a version with no bias: 

**==> picture [322 x 11] intentionally omitted <==**

Subsequent work has proposed replacing the ReLU with other nonlinear activation functions such as Gaussian Error Linear Units, GELU( _x_ ) = _x_ Φ( _x_ ) [Hendrycks and Gimpel, 2016], and Swish _β_ ( _x_ ) = _xσ_ ( _βx_ ) [Ramachandran et al., 2017]. 

**==> picture [324 x 26] intentionally omitted <==**

## **2 Gated Linear Units (GLU) and Variants** 

[Dauphin et al., 2016] introduced Gated Linear Units (GLU), a neural network layer defined as the componentwise product of two linear transformations of the input, one of which is sigmoid-activated. They also suggest omitting the activation, which they call a "bilinear" layer and attribute to [Mnih and Hinton, 2007]. 

**==> picture [335 x 26] intentionally omitted <==**

We can also define GLU variants using other activation functions: 

> 1Also in the interest of ML fairness. 

1 

**==> picture [354 x 41] intentionally omitted <==**

In this paper, we propose additional variations on the Transformer FFN layer which use GLU or one of its variants in place of the first linear transformation and the activation function. Again, we omit the bias terms. 

**==> picture [346 x 71] intentionally omitted <==**

All of these layers have three weight matrices, as opposed to two for the original FFN. To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units _dff_ (the second dimension of _W_ and _V_ and the first dimension of _W_ 2) by a factor of 3[2][when][comparing][these] layers to the original two-matrix version. 

## **3 Experiments on Text-to-Text Transfer Transformer (T5)** 

We test the FFN variants we have described on the transfer-learning setup from [Raffel et al., 2019]. An encoder-decoder transformer model [Vaswani et al., 2017] is trained on a denoising objective of predicting missing text segments, and subsequently fine-tuned on various language understanding tasks. 

## **3.1 Model Architecture** 

We use the same code base, model architecture, and training task as the base model from [Raffel et al., 2019]. The encoder and decoder each consist of 12 layers, with _dmodel_ = 768. For the attention layers, _h_ = 12 and _dk_ = _dv_ = 64. The FFN layers have hidden size _dff_ = 3072. As we describe above, for the GLU-variant-based FFN layers, which have thee weight matrices instead of two, we reduce the hidden layer to _dff_ = 2048, so as to maintain the same parameter and operation counts as the base model. 

Table 1: Heldout-set log-perplexity for Transformer models on the segment-filling task from [Raffel et al., 2019]. All models are matched for parameters and computation. 

|Training Steps|65,536|524,288|
|---|---|---|
|FFNReLU(_baseline_)|1.997 (0.005)|1.677|
|FFNGELU|1.983 (0.005)|1.679|
|FFNSwish|1.994 (0.003)|1.683|
|FFNGLU|1.982 (0.006)|1.663|
|FFNBilinear|1.960 (0.005)|1.648|
|FFNGEGLU|**1.942** (0.004)|**1.633**|
|FFNSwiGLU|**1.944** (0.010)|**1.636**|
|FFNReGLU|1.953 (0.003)|1.645|



2 

## **3.2 Pre-Training and Perplexity Results** 

Identically to [Raffel et al., 2019], we pre-train for 524,288 steps on the span-filling objective on the C4 dataset. Each training batch consists of 128 examples, each of which has an input of 512 tokens and an output of 114 tokens, the output containing multiple spans of tokens which were deleted from the input[2] . Similarly to [Raffel et al., 2019], we use the Adafactor optimizer [Shazeer and Stern, 2018] and an inversesquare-root learning-rate schedule. We also decay the learning rate linearly for the final 10 percent of the training steps. Our main departure from [Raffel et al., 2019] is that we use no dropout during pre-training. We find this to produce superior results. We compute the log-perplexity on the training objective on a heldout shard of C4, which we believe to be a good indicator of model quality. For each model architecture, we also trained four models for a shorter period (65,536 steps) to measure inter-run variability. The results are listed in table 1. The GEGLU and SwiGLU variants produce the best perplexities. 

## **3.3 Fine-Tuning** 

We then fine-tune each fully-trained model once on an examples-proportional mixture of the Stanford Question-Answering Dataset (SQuAD) [Rajpurkar et al., 2016] and all the language understanding tasks in the GLUE [Wang et al., 2018] and SuperGlue [Wang et al., 2019] benchmarks.[3] Fine-tuning consists of 131072 steps with a learning rate of 10[−][3] . As in training, the input sequences for each step have a combined length of approximately 65,536 tokens. Following [Raffel et al., 2019], we use a dropout rate of 0 _._ 1 on the layer outputs, feed-forward hidden-layers and attention weights. The embedding matrices are fixed during fine-tuning. 

Tables 2, 3 and 4 show results on the development sets. For each task, we report the best score of any of the checkpoints recorded during fine-tuning. While the results are noisy, the new GLU-variants perform best on most of the tasks. For comparison, at the bottom of each of the tables we list the reuslts from [Raffel et al., 2019]. The model is identical to our FFNReLU model. Their results are notably worse, which we believe was caused by their use of dropout during pre-training. Also listed are the inter-run standard deviations measured by [Raffel et al., 2019]. 

Table 2: GLUE Language-Understanding Benchmark [Wang et al., 2018] (dev). 

||Score|CoLA|SST-2|MRPC|MRPC|STSB|STSB|QQP|QQP|MNLIm|MNLImm|QNLI|RTE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Average|MCC|Acc|F1|Acc|PCC|SCC|F1|Acc|Acc|Acc|Acc|Acc|
|FFNReLU|83_._80|51_._32|94_._04|**93.08**|**90.20**|89_._64|89_._42|89_._01|91_._75|85_._83|86_._42|92_._81|80_._14|
|FFNGELU|83_._86|53_._48|94_._04|92_._81|**90.20**|89_._69|89_._49|88_._63|91_._62|85_._89|86_._13|92_._39|80_._51|
|FFNSwish|83_._60|49_._79|93_._69|92_._31|89_._46|89_._20|88_._98|88_._84|91_._67|85_._22|85_._02|92_._33|81_._23|
|FFNGLU|84_._20|49_._16|94_._27|92_._39|89_._46|89_._46|89_._35|88_._79|91_._62|86_._36|86_._18|92_._92|**84.12**|
|FFNGEGLU|84_._12|53_._65|93_._92|92_._68|89_._71|90_._26|90_._13|89_._11|91_._85|86_._15|86_._17|92_._81|79_._42|
|FFNBilinear|83_._79|51_._02|**94.38**|92_._28|89_._46|90_._06|89_._84|88_._95|91_._69|**86.90**|**87.08**|92_._92|81_._95|
|FFNSwiGLU|84_._36|51_._59|93_._92|92_._23|88_._97|**90.32**|**90.13**|**89.14**|**91.87**|86_._45|86_._47|**92.93**|83_._39|
|FFNReGLU|**84.67**|**56.16**|**94.38**|92_._06|89_._22|89_._97|89_._85|88_._86|91_._72|86_._20|86_._40|92_._68|81_._59|
|[Rafel et al., 2019]|83_._28|53_._84|92_._68|92_._07|88_._92|88_._02|87_._94|88_._67|91_._56|84_._24|84_._57|90_._48|76_._28|
|ibid. stddev.|0_._235|1_._111|0_._569|0_._729|1_._019|0_._374|0_._418|0_._108|0_._070|0_._291|0_._231|0_._361|1_._393|



## **4 Conclusions** 

We have extended the GLU family of layers and proposed their use in Transformer. In a transfer-learning setup, the new variants seem to produce better perplexities for the de-noising objective used in pre-training, as well as better results on many downstream language-understanding tasks. These architectures are simple to implement, and have no apparent computational drawbacks. We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence. 

> 2Each training step took approximately 0.15 seconds on a 32-core TPUv2 cluster. 

> 3This departs from [Raffel et al., 2019], who fine-tuned separately on the different tasks. We chose one fine-tuning run for simplicity. 

3 

Table 3: SuperGLUE Language-Understanding Benchmark [Wang et al., 2019] (dev). 

||Score|BoolQ|CB|CB|CoPA|MultiRC|MultiRC|ReCoRD|ReCoRD|RTE|WiC|WSC|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Average|Acc|F1|Acc|Acc|F1|EM|F1|EM|Acc|Acc|Acc|
|FFNReLU|72_._76|80_._15|83_._37|89_._29|70_._00|76_._93|39_._14|73_._73|72_._91|83_._39|67_._71|77_._88|
|FFNGELU|72_._98|80_._64|86_._24|**91.07**|74_._00|75_._93|38_._61|72_._96|72_._03|81_._59|68_._34|75_._96|
|FFNSwish|72_._40|80_._43|77_._75|83_._93|67_._00|76_._34|39_._14|73_._34|72_._36|81_._95|68_._18|81_._73|
|FFNGLU|73_._95|80_._95|77_._26|83_._93|73_._00|76_._07|39_._03|74_._22|73_._50|84_._12|67_._71|**87.50**|
|FFNGEGLU|73_._96|81_._19|82_._09|87_._50|72_._00|**77.43**|**41.03**|75_._28|**74.60**|83_._39|67_._08|83_._65|
|FFNBilinear|73_._81|**81.53**|82_._49|89_._29|**76.00**|76_._04|40_._92|74_._97|74_._10|82_._67|**69.28**|78_._85|
|FFNSwiGLU|**74.56**|81_._19|82_._39|89_._29|73_._00|75_._56|38_._72|**75.35**|74_._55|**85.20**|67_._24|86_._54|
|FFNReGLU|73_._66|80_._89|**86.37**|**91.07**|67_._00|75_._32|40_._50|75_._07|74_._18|84_._48|67_._40|79_._81|
|[Rafel et al., 2019]|71_._36|76_._62|91_._22|91_._96|66_._20|66_._13|25_._78|69_._05|68_._16|75_._34|68_._04|78_._56|
|ibid. stddev.|0_._416|0_._365|3_._237|2_._560|2_._741|0_._716|1_._011|0_._370|0_._379|1_._228|0_._850|2_._029|



Table 4: SQuAD [Rajpurkar et al., 2016] v1.1 (dev). 

||EM|F1|
|---|---|---|
|FFNReLU|83_._18|90_._87|
|FFNGELU|83_._09|90_._79|
|FFNSwish|83_._25|90_._76|
|FFNGLU|82_._88|90_._69|
|FFNGEGLU|83_._55|91_._12|
|FFNBilinear|**83.82**|91_._06|
|FFNSwiGLU|83_._42|91_._03|
|FFNReGLU|83_._53|**91.18**|
|[Rafel et al., 2019]|80_._88|88_._81|
|ibid. Standard Deviation|0_._343|0_._226|



## **References** 

- Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. _CoRR_ , abs/1612.08083, 2016. URL http://arxiv.org/abs/1612.08083. 

- Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In _Proceedings of the fourteenth international conference on artificial intelligence and statistics_ , pages 315–323, 2011. 

- Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. _CoRR_ , abs/1606.08415, 2016. URL http://arxiv.org/abs/1606.08415. 

- Andriy Mnih and Geoffrey Hinton. Three new graphical models for statistical language modelling. In _Proceedings of the 24th international conference on Machine learning_ , pages 641–648, 2007. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _arXiv e-prints_ , 2019. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. _arXiv preprint arXiv:1606.05250_ , 2016. 

- Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. _arXiv preprint arXiv:1710.05941_ , 2017. 

- Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. _arXiv preprint arXiv:1804.04235_ , 2018. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_ , 2017. 

4 

- Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. _arXiv preprint arXiv:1804.07461_ , 2018. 

- Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. _arXiv preprint arXiv:1905.00537_ , 2019. 

5 

