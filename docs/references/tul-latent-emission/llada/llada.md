Title: Large Language Diffusion Models

URL Source: https://arxiv.org/html/2502.09992

Published Time: Tue, 21 Oct 2025 00:33:25 GMT

Markdown Content:
Shen Nie 1,2,3 Fengqi Zhu 1,2,3 1 1 footnotemark: 1 2 2 footnotemark: 2 Zebin You 1,2,3 2 2 footnotemark: 2 Xiaolu Zhang 4 Jingyang Ou 1,2,3

Jun Hu 4 3 3 footnotemark: 3 Jun Zhou 4 Yankai Lin 1,2,3 3 3 footnotemark: 3 Ji-Rong Wen 1,2,3 Chongxuan Li 1,2,3 3 3 footnotemark: 3

1 Gaoling School of Artificial Intelligence, Renmin University of China 

2 Beijing Key Laboratory of Research on Large Models and Intelligent Governance 

3 Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE 

4 Ant Group 

{nieshen,fengqizhu,chongxuanli}@ruc.edu.cn Equal contribution.Work done during an internship at Ant Group.Project leaders.Correspondence to Chongxuan Li.

###### Abstract

The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing _LLaDA_, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong _scalability_ and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in _in-context learning_ and, after SFT, exhibits impressive _instruction-following_ abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: [https://ml-gsai.github.io/LLaDA-demo/](https://ml-gsai.github.io/LLaDA-demo/).

1 Introduction
--------------

Large language models (LLMs)[[1](https://arxiv.org/html/2502.09992v3#bib.bib1)] fall entirely within the framework of generative modeling. Specifically, LLMs aim to capture the true but unknown language distribution p data​(⋅)p_{\textrm{data}}(\cdot) by optimizing a model distribution p θ​(⋅)p_{\theta}(\cdot) through maximum likelihood estimation, or equivalently KL divergence minimization between the two distributions:

max θ 𝔼 p data​(x)log p θ(x)⇔min θ KL(p data(x)||p θ(x))⏟Generative modeling principles.\displaystyle\underbrace{\max_{\theta}\mathbb{E}_{p_{\textrm{data}}(x)}\log p_{\theta}(x)\Leftrightarrow\min_{\theta}\textrm{KL}(p_{\textrm{data}}(x)||p_{\theta}(x))}_{\textrm{Generative modeling principles}}.(1)

The predominant approach relies on the autoregressive modeling (ARM)—commonly referred to as the “next-token prediction” paradigm—to define the model distribution:

p θ​(x)=p θ​(x 1)​∏i=2 L p θ​(x i∣x 1,…,x i−1)⏟Autoregressive formulation,\displaystyle\underbrace{p_{\theta}(x)=p_{\theta}(x^{1})\prod_{i=2}^{L}p_{\theta}(x^{i}\mid x^{1},\dots,x^{i-1})}_{\textrm{Autoregressive formulation}},(2)

where x x is a sequence of length L L, and x i x^{i} is the i i-th token. This paradigm has proven remarkably effective[[2](https://arxiv.org/html/2502.09992v3#bib.bib2), [3](https://arxiv.org/html/2502.09992v3#bib.bib3), [4](https://arxiv.org/html/2502.09992v3#bib.bib4), [5](https://arxiv.org/html/2502.09992v3#bib.bib5)] and has become the foundation of current LLMs. Despite its widespread adoption, a fundamental question remains unanswered: Is the autoregressive paradigm the only path to achieving the core capabilities of LLMs, such as scalability, in-context learning, and instruction-following?

![Image 1: Refer to caption](https://arxiv.org/html/2502.09992v3/x1.png)

![Image 2: Refer to caption](https://arxiv.org/html/2502.09992v3/x2.png)

Figure 1: Zero/Few‑Shot Benchmarks. We scale LLaDA to 8B parameters from scratch and observe competitive zero/few‑shot performance compared with strong autoregressive LLMs[[6](https://arxiv.org/html/2502.09992v3#bib.bib6)].

We argue that the answer is _not_ a simple “yes”. The key insight overlooked previously is: It is the _generative modeling principles_ (i.e., Eq. ([1](https://arxiv.org/html/2502.09992v3#S1.E1 "Equation 1 ‣ 1 Introduction ‣ Large Language Diffusion Models"))), _rather than the autoregressive formulation_ (i.e., Eq. ([2](https://arxiv.org/html/2502.09992v3#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Large Language Diffusion Models"))) itself, that fundamentally underpin the essential properties of LLMs.

In particular, we argue that _scalability_ is primarily a consequence of the interplay between Transformers[[7](https://arxiv.org/html/2502.09992v3#bib.bib7)], model size, data size, and _Fisher consistency_ 1 1 1 It suggests the ability to recover the true data distribution with infinite data, a sufficiently large network and optimal training.[[8](https://arxiv.org/html/2502.09992v3#bib.bib8)] induced by the generative principles in Eq.([1](https://arxiv.org/html/2502.09992v3#S1.E1 "Equation 1 ‣ 1 Introduction ‣ Large Language Diffusion Models")), rather than a unique result of the ARMs in Eq.([2](https://arxiv.org/html/2502.09992v3#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Large Language Diffusion Models")). The success of diffusion transformers[[9](https://arxiv.org/html/2502.09992v3#bib.bib9), [10](https://arxiv.org/html/2502.09992v3#bib.bib10)] on visual data[[11](https://arxiv.org/html/2502.09992v3#bib.bib11)] supports this claim. Furthermore, the _instruction-following_ and _in-context learning_[[4](https://arxiv.org/html/2502.09992v3#bib.bib4)] capabilities appear to be intrinsic properties of all conditional generative models on structurally consistent linguistic tasks, rather than exclusive advantages of ARMs. In addition, while ARMs can be interpreted as a _lossless data compressor_[[12](https://arxiv.org/html/2502.09992v3#bib.bib12), [13](https://arxiv.org/html/2502.09992v3#bib.bib13)], any sufficiently expressive probabilistic model can achieve similar capabilities[[14](https://arxiv.org/html/2502.09992v3#bib.bib14)].

However, certain inherent limitations of LLMs can be directly attributed to their autoregressive nature. For instance, the left-to-right generation process restricts their ability to handle reversal reasoning tasks[[15](https://arxiv.org/html/2502.09992v3#bib.bib15)], highlighting a representative failure in the generalization capabilities of current models.

Motivated by these insights, we introduce _LLaDA (Large Language Diffusion with mAsking)_ to investigate whether the capabilities exhibited by LLMs can emerge from generative modeling principles beyond ARMs, thereby addressing the fundamental question posed earlier. In contrast to traditional ARMs, LLaDA leverages a masked diffusion model (MDM)[[16](https://arxiv.org/html/2502.09992v3#bib.bib16), [17](https://arxiv.org/html/2502.09992v3#bib.bib17), [18](https://arxiv.org/html/2502.09992v3#bib.bib18), [19](https://arxiv.org/html/2502.09992v3#bib.bib19), [20](https://arxiv.org/html/2502.09992v3#bib.bib20)], which incorporates a forward data masking process and trains a _mask predictor_ to approximate its reverse process. This design enables LLaDA to construct a model distribution with bidirectional dependencies and optimize a variational lower bound of its log-likelihood, offering a principled and previously unexplored perspective on the core capabilities of LLMs discussed above.

We adopt the standard pipeline of data preparation, pre-training, supervised fine-tuning (SFT), and evaluation, scaling LLaDA to an unprecedented language diffusion of size 8B. In particular, LLaDA 8B was pre-trained from scratch on 2.3 trillion tokens using 0.13 million H800 GPU hours, followed by SFT on 4.5 million pairs. Across diverse tasks, including language understanding, math, code, and Chinese, LLaDA demonstrates the following contributions:

*   •LLaDA scales effectively to a compute budget of 10 23 10^{23} FLOPs, achieving comparable results to ARM baselines trained on the same data across six tasks, e.g., MMLU and GSM8K. 
*   •The pre-trained LLaDA 8B Base surpasses LLaMA2 7B Base[[21](https://arxiv.org/html/2502.09992v3#bib.bib21)] on nearly all 15 standard zero/few-shot learning tasks while performing on par with LLaMA3 8B Base[[6](https://arxiv.org/html/2502.09992v3#bib.bib6)], showcasing effective in-context learning capability. 
*   •LLaDA significantly enhances the ability to follow instructions after SFT, as demonstrated in case studies such as multi-turn dialogue. 
*   •LLaDA effectively breaks the reversal curse[[15](https://arxiv.org/html/2502.09992v3#bib.bib15)] with consistent performance across forward and reversal tasks. Notably, it outperforms GPT-4o in a reversal poem completion task. 

![Image 3: Refer to caption](https://arxiv.org/html/2502.09992v3/x3.png)

Figure 2: Overview of LLaDA. (a) Pre-training. LLaDA is trained on text with random masks applied independently to all tokens at the same ratio t∼U​[0,1]t\sim U[0,1]. (b) SFT. Only response tokens are possibly masked. (c) Sampling. LLaDA simulates a diffusion process from t=1 t=1 (fully masked) to t=0 t=0 (unmasked), predicting all masks simultaneously at each step with flexible remask strategies.

2 Approach
----------

In this section, we introduce the probabilistic formulation 2 2 2 Here, we focus on the approach of LLaDA. A rigorous formulation of MDM is provided in Appendix[A](https://arxiv.org/html/2502.09992v3#A1 "Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models") for interested readers., along with the pre-training, supervised fine-tuning, and inference procedures for LLaDA, as illustrated in Fig.[2](https://arxiv.org/html/2502.09992v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Large Language Diffusion Models").

### 2.1 Probabilistic Formulation

Unlike ARMs in Eq.([2](https://arxiv.org/html/2502.09992v3#S1.E2 "Equation 2 ‣ 1 Introduction ‣ Large Language Diffusion Models")), LLaDA defines a model distribution p θ​(x 0)p_{\theta}(x_{0}) through a _forward process_ and a _reverse process_[[16](https://arxiv.org/html/2502.09992v3#bib.bib16), [17](https://arxiv.org/html/2502.09992v3#bib.bib17), [18](https://arxiv.org/html/2502.09992v3#bib.bib18), [19](https://arxiv.org/html/2502.09992v3#bib.bib19), [20](https://arxiv.org/html/2502.09992v3#bib.bib20)]. The forward process gradually masks tokens independently in x 0 x_{0} until the sequence is fully masked at t=1 t=1. For t∈(0,1)t\in(0,1), the sequence x t x_{t} is partially masked, with each being masked with probability t t or remaining unmasked with probability 1−t 1-t. The reverse process recovers the data distribution by iteratively predicting masked tokens as t t moves from 1 1 to 0.

The core of LLaDA is a _mask predictor_, a parametric model p θ(⋅|x t)p_{\theta}(\cdot|x_{t}) that takes x t x_{t} as input and predicts all masked tokens (denoted as M) simultaneously. It is trained using a cross-entropy loss computed only on the masked tokens[[18](https://arxiv.org/html/2502.09992v3#bib.bib18), [19](https://arxiv.org/html/2502.09992v3#bib.bib19), [20](https://arxiv.org/html/2502.09992v3#bib.bib20)]:

ℒ​(θ)≜−𝔼 t,x 0,x t​[1 t​∑i=1 L 1​[x t i=M]​log⁡p θ​(x 0 i|x t)],\displaystyle\mathcal{L}(\theta)\triangleq-\mathbb{E}_{t,x_{0},x_{t}}\left[\frac{1}{t}\sum_{i=1}^{L}\textbf{1}[x_{t}^{i}=\textrm{M}]\log p_{\theta}(x_{0}^{i}|x_{t})\right],(3)

where x 0 x_{0} is a training sample, t t is a continuous random variable drawn uniformly from [0,1][0,1], x t x_{t} is sampled from the forward process and L L is the sequence length. The indicator function 1​[⋅]\textbf{1}[\cdot] ensures that the loss is computed only for masked tokens.

Once trained, we can simulate a reverse process (see Sec.[2.4](https://arxiv.org/html/2502.09992v3#S2.SS4 "2.4 Inference ‣ 2 Approach ‣ Large Language Diffusion Models") for details) parameterized by the mask predictor and define the model distribution p θ​(x 0)p_{\theta}(x_{0}) as the marginal distribution induced at t=0 t=0. The loss function in Eq.([3](https://arxiv.org/html/2502.09992v3#S2.E3 "Equation 3 ‣ 2.1 Probabilistic Formulation ‣ 2 Approach ‣ Large Language Diffusion Models")) has been proven to be an upper bound on the negative log-likelihood of the model distribution, making it a principled objective for generative modeling:

−𝔼 p data​(x 0)​[log⁡p θ​(x 0)]≤ℒ​(θ).\displaystyle-\mathbb{E}_{p_{\textrm{data}}(x_{0})}\left[\log p_{\theta}(x_{0})\right]\leq\mathcal{L}(\theta).(4)

Notably, LLaDA employs a masking ratio that varies randomly between 0 and 1 while BERT[[22](https://arxiv.org/html/2502.09992v3#bib.bib22)] uses a fixed ratio. The subtle differences have significant implications, especially at scale: as shown in Eq.([4](https://arxiv.org/html/2502.09992v3#S2.E4 "Equation 4 ‣ 2.1 Probabilistic Formulation ‣ 2 Approach ‣ Large Language Diffusion Models")), LLaDA is a principled generative model with the potential to perform in-context learning and instruction-following naturally, akin to LLMs. Moreover, its generative perspective implies strong scalability with large data and models as discussed in Sec.[1](https://arxiv.org/html/2502.09992v3#S1 "1 Introduction ‣ Large Language Diffusion Models"). In addition, MaskGIT[[23](https://arxiv.org/html/2502.09992v3#bib.bib23)] adopts a heuristic training objective, which misses the 1 t\frac{1}{t} term compared to Eq.([3](https://arxiv.org/html/2502.09992v3#S2.E3 "Equation 3 ‣ 2.1 Probabilistic Formulation ‣ 2 Approach ‣ Large Language Diffusion Models")), and lacks a theoretical link to maximum likelihood. We emphasize that it is precisely the theoretical foundation of maximum likelihood estimation that motivated us to scale discrete diffusion models for language modeling.

### 2.2 Pre-training

LLaDA employs a Transformer[[7](https://arxiv.org/html/2502.09992v3#bib.bib7)] as the mask predictor, similar to existing LLMs. However, LLaDA does not use a causal mask, as its formulation allows it to see the entire input for predictions.

We trained two variants of LLaDA with different sizes: 1B and 8B. We summarize the model architecture of LLaDA 8B and LLaMA3 8B[[6](https://arxiv.org/html/2502.09992v3#bib.bib6)] here, and details are provided in Appendix[B.2](https://arxiv.org/html/2502.09992v3#A2.SS2 "B.2 Details about Model Training ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). We have ensured consistency in most hyperparameters while making several necessary modifications. We use vanilla multi-head attention instead of grouped query attention[[24](https://arxiv.org/html/2502.09992v3#bib.bib24)] for simplicity, as LLaDA is incompatible with KV caching, resulting in a different number of key and value heads. Consequently, the attention layer has more parameters, and we reduce the FFN dimension to maintain a comparable model size. Additionally, the vocabulary size differs due to a tokenizer[[4](https://arxiv.org/html/2502.09992v3#bib.bib4)] adapted on our data.

The LLaDA model is pre-trained on a dataset comprising 2.3 trillion (T) tokens, adhering to a data protocol that aligns closely with existing LLMs[[25](https://arxiv.org/html/2502.09992v3#bib.bib25), [26](https://arxiv.org/html/2502.09992v3#bib.bib26)], without the incorporation of any special techniques. The data are derived from online corpora, with low-quality content filtered through manually designed rules and LLM-based approaches. Beyond general text, the dataset encompasses high-quality code, math, and multilingual data. Please refer to Appendix[B.1](https://arxiv.org/html/2502.09992v3#A2.SS1 "B.1 Data Collection and Preprocessing ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for more details about datasets. The mixing of data sources and domains is guided by scaled-down ARMs. The pre-training process utilizes a fixed sequence length of 4096 tokens, incurring a total computational cost of 0.13 million H800 GPU hours, similar to ARMs of the same scale and dataset size.

For a training sequence x 0 x_{0}, we randomly sample t∈[0,1]t\in[0,1], mask each token independently with the same probability t t to obtain x t x_{t} (see Fig.[2](https://arxiv.org/html/2502.09992v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Large Language Diffusion Models") (a)) and estimate Eq.([3](https://arxiv.org/html/2502.09992v3#S2.E3 "Equation 3 ‣ 2.1 Probabilistic Formulation ‣ 2 Approach ‣ Large Language Diffusion Models")) via the Monte Carlo method for stochastic gradient descent training. In addition, following Nie et al. [[27](https://arxiv.org/html/2502.09992v3#bib.bib27)], to enhance the ability of LLaDA to handle variable-length data, we set 1% of the pre-training data to a random length that is uniformly sampled from the range [1,4096][1,4096].

We adopted the Warmup-Stable-Decay[[28](https://arxiv.org/html/2502.09992v3#bib.bib28)] learning rate scheduler to monitor the training progress without interrupting continuous training. Specifically, we linearly increased the learning rate from 0 to 4×10−4 4\times 10^{-4} over the first 2000 iterations and maintained it at 4×10−4 4\times 10^{-4}. After processing 1.2T tokens, we decayed the learning rate to 1×10−4 1\times 10^{-4} and held it constant for the next 0.8T tokens to ensure stable training. Finally, we linearly reduced the learning rate from 1×10−4 1\times 10^{-4} to 1×10−5 1\times 10^{-5} for the last 0.3T tokens. Furthermore, we utilized the AdamW optimizer[[29](https://arxiv.org/html/2502.09992v3#bib.bib29)] with a weight decay of 0.1, a batch size of 1280, and a local batch size of 4 4 per GPU. The 8B experiment was executed once, without any hyperparameter tuning.

### 2.3 Supervised Fine-Tuning

We enhance the capability of LLaDA to follow instructions by supervised fine-tuning (SFT) with paired data (p 0,r 0)(p_{0},r_{0}), where p 0 p_{0} is the prompt and r 0 r_{0} denotes the response. This is the simplest and most basic post-training method for LLMs. Technically, this requires to model the conditional distribution p θ​(r 0|p 0)p_{\theta}(r_{0}|p_{0}) instead of p θ​(x 0)p_{\theta}(x_{0}) in pre-training.

The implementation is similar to pre-training. As shown in Fig.[2](https://arxiv.org/html/2502.09992v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Large Language Diffusion Models") (b), we leave the prompt unchanged and mask the tokens in the response independently, as done for x 0 x_{0}. Then, we feed both the prompt and the masked response r t r_{t} to the pre-trained mask predictor to compute the loss for SFT:

−𝔼 t,p 0,r 0,r t​[1 t​∑i=1 L′1​[r t i=M]​log⁡p θ​(r 0 i|p 0,r t)],\displaystyle-\mathbb{E}_{t,p_{0},r_{0},r_{t}}\left[\frac{1}{t}\sum_{i=1}^{L^{\prime}}\textbf{1}[r_{t}^{i}=\textrm{M}]\log p_{\theta}(r_{0}^{i}|p_{0},r_{t})\right],(5)

where L′L^{\prime} denotes a dynamic length specified later, and all other notations remain the same as before.

Note that this approach is fully compatible with pre-training. Essentially, the concatenation of p 0 p_{0} and r 0 r_{0} can be treated as clean pre-training data x 0 x_{0}, while the concatenation of p 0 p_{0} and r t r_{t} serves as the masked version x t x_{t}. The process is identical to pre-training, with the only difference being that all masked tokens happen to appear in the r 0 r_{0} portion.

The LLaDA 8B model undergoes SFT on a dataset comprising 4.5 million pairs. Consistent with the pre-training process, both data preparation and training follow the SFT protocols utilized in existing LLMs[[25](https://arxiv.org/html/2502.09992v3#bib.bib25), [26](https://arxiv.org/html/2502.09992v3#bib.bib26)], without introducing any additional techniques to optimize LLaDA’s performance. The dataset spans multiple domains, including code, mathematics, and instruction-following. We append |EOS||\text{EOS}| tokens to the end of short pairs in each mini-batch to ensure equal lengths across all data. We treat |EOS||\text{EOS}| as a normal token during training and remove it during sampling, enabling LLaDA to control the response length automatically. Please refer to Appendix[B.1](https://arxiv.org/html/2502.09992v3#A2.SS1 "B.1 Data Collection and Preprocessing ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for more details.

We train for 3 epochs on the SFT data using a similar schedule to the pre-training phase. The learning rate is linearly increased from 0 to 2.5×10−5 2.5\times 10^{-5} over the first 50 iterations and then kept constant. During the final 10%10\% of iterations, it is linearly reduced to 2.5×10−6 2.5\times 10^{-6}. Additionally, we set the weight decay to 0.1 0.1, the global batch size to 256 256, and the local batch size to 2 2 per GPU. The SFT experiment was executed once, without any hyperparameter tuning.

### 2.4 Inference

As a generative model, LLaDA can sample new text and evaluate the likelihood of candidate text _in a diffusion manner instead of the left-to-right autoregressive fashion_.

We begin with the reverse generation process. As illustrated in Fig.[2](https://arxiv.org/html/2502.09992v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Large Language Diffusion Models")(c), given a prompt p 0 p_{0}, we discretize the reverse process to sample from the model distribution p θ​(r 0|p 0)p_{\theta}(r_{0}|p_{0}), starting from a fully masked response. The total number of sampling steps is a hyperparameter, which naturally provides LLaDA with a trade-off between efficiency and sample quality, as analyzed in Sec.[3.3](https://arxiv.org/html/2502.09992v3#S3.SS3 "3.3 Reversal Reasoning and Analyses ‣ 3 Experiments ‣ Large Language Diffusion Models"). We employ uniformly distributed timesteps by default. In addition, the generation length is also treated as a hyperparameter, specifying the length of the fully masked sentence at the beginning of the sampling process. After generation, tokens appearing after the |EOS||\text{EOS}| token are discarded. As detailed in Appendix[B.5](https://arxiv.org/html/2502.09992v3#A2.SS5 "B.5 Ablation on Generated Length ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), since both pre-training and SFT are conducted using datasets with variable lengths, the final results are insensitive to this length hyperparameter.

At an intermediate step from time t∈(0,1]t\in(0,1] to s∈[0,t)s\in[0,t), we feed both p 0 p_{0} and r t r_{t} into the mask predictor and predict all masked tokens simultaneously. Subsequently, we remask s t\frac{s}{t} of the predicted tokens in expectation to obtain r s r_{s}, ensuring that the transition of the reverse process aligns with the forward process for accurate sampling[[18](https://arxiv.org/html/2502.09992v3#bib.bib18), [19](https://arxiv.org/html/2502.09992v3#bib.bib19), [20](https://arxiv.org/html/2502.09992v3#bib.bib20)]. In principle, the remasking strategy should be purely random. However, inspired by the annealing tricks of sampling in LLMs[[4](https://arxiv.org/html/2502.09992v3#bib.bib4), [30](https://arxiv.org/html/2502.09992v3#bib.bib30)], we adopt a low-confidence remasking strategy, where s t\frac{s}{t} of predicted tokens with the lowest confidence are remarked based on the predictions, same as the approach of Chang et al. [[23](https://arxiv.org/html/2502.09992v3#bib.bib23)].

We mention that LLaDA enables flexible sampling. In particular, it supports autoregressive and block diffusion[[31](https://arxiv.org/html/2502.09992v3#bib.bib31)] sampling directly after the pre-training or SFT processes described above, without requiring any further modifications or training. We provide a detailed analysis in Appendix[B.4](https://arxiv.org/html/2502.09992v3#A2.SS4 "B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). Nevertheless, the diffusion sampling (i.e., the reverse generation process) yields the best performance and is adopted as the default throughout this paper, especially for all experiments presented in Sec.[3](https://arxiv.org/html/2502.09992v3#S3 "3 Experiments ‣ Large Language Diffusion Models").

For conditional likelihood evaluation, we can naturally utilize the upper bound in Eq.([5](https://arxiv.org/html/2502.09992v3#S2.E5 "Equation 5 ‣ 2.3 Supervised Fine-Tuning ‣ 2 Approach ‣ Large Language Diffusion Models")). However, we find that the following equivalent form[[20](https://arxiv.org/html/2502.09992v3#bib.bib20)] exhibits lower variance and is more stable:

−𝔼 l,r 0,r l​[L l​∑i=1 L 1​[r l i=M]​log⁡p θ​(r 0 i|p 0,r l)],\displaystyle-\mathbb{E}_{l,r_{0},r_{l}}\left[\frac{L}{l}\sum_{i=1}^{L}\textbf{1}[r_{l}^{i}=\textrm{M}]\log p_{\theta}(r_{0}^{i}|p_{0},r_{l})\right],(6)

where L L is the sequence length of r 0 r_{0}, l l is uniformly sampled from {1,2,…,L}\{1,2,\dots,L\}, and r l r_{l} is obtained by uniformly sampling l l tokens from r 0 r_{0} without replacement for masking.

We present the training and inference algorithms, along with theoretical details, in Appendix[A](https://arxiv.org/html/2502.09992v3#A1 "Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models").

3 Experiments
-------------

![Image 4: Refer to caption](https://arxiv.org/html/2502.09992v3/x4.png)

![Image 5: Refer to caption](https://arxiv.org/html/2502.09992v3/x5.png)

![Image 6: Refer to caption](https://arxiv.org/html/2502.09992v3/x6.png)

![Image 7: Refer to caption](https://arxiv.org/html/2502.09992v3/x7.png)

![Image 8: Refer to caption](https://arxiv.org/html/2502.09992v3/x8.png)

![Image 9: Refer to caption](https://arxiv.org/html/2502.09992v3/x9.png)

Figure 3: Scalability of LLaDA. We evaluate the performance of LLaDA and our ARM baselines trained on the same data across increasing pre-training computational FLOPs. LLaDA exhibits strong scalability, matching the overall performance of ARMs on six tasks.

We evaluate the scalability, instruction-following, and in-context learning capabilities of LLaDA on standard benchmarks, followed by analyses and case studies to provide a comprehensive assessment.

### 3.1 Scalability of LLaDA on Language Tasks

We first investigate the _scalability_ of LLaDA on downstream tasks in comparison with the ARM baselines we constructed. Specifically, at the 1B scale, we ensured that LLaDA and ARM shared the same architecture, data, and all other configurations. At larger scales, we also report results for LLaDA and ARM models of slightly different sizes trained on the same data due to resource limitations. Please refer to Appendix[B.2](https://arxiv.org/html/2502.09992v3#A2.SS2 "B.2 Details about Model Training ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for more details. We use the pre-training computational cost as a unified scaling metric. For evaluation, we focused on six standard and diverse tasks.

Fig.[3](https://arxiv.org/html/2502.09992v3#S3.F3 "Figure 3 ‣ 3 Experiments ‣ Large Language Diffusion Models") shows that LLaDA demonstrates impressive scalability, with its overall trend highly competitive with ARMs. Notably, on tasks such as MMLU and GSM8K, LLaDA exhibits even stronger scalability. Even on relatively weaker tasks like PIQA, the performance gap with ARMs narrows as scale increases. To account for the influence of outliers, we opted not to fit quantitative curves, avoiding potential misinterpretation. Nevertheless, the results clearly demonstrate the scalability of LLaDA.

Considering LLaDA’s advantages on certain benchmarks, we hypothesize that this performance gain stems from a key architectural difference: while autoregressive models optimize only left-to-right conditional probabilities, LLaDA is trained to consider multiple conditioning directions, as detailed in Appendix[A.2](https://arxiv.org/html/2502.09992v3#A1.SS2 "A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"), which may offer greater flexibility and lead to better generalization. This hypothesis is motivated by LLaDA’s strong performance on reversal reasoning in Sec.[3.3](https://arxiv.org/html/2502.09992v3#S3.SS3 "3.3 Reversal Reasoning and Analyses ‣ 3 Experiments ‣ Large Language Diffusion Models") and the ablation studies on sampling strategies in Appendix[B.4](https://arxiv.org/html/2502.09992v3#A2.SS4 "B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models").

Nie et al. [[27](https://arxiv.org/html/2502.09992v3#bib.bib27)] suggests that MDM requires 16 times more computation than ARM to achieve the same likelihood. However, key differences make our findings more broadly applicable. In particular, likelihood is a relatively indirect metric for downstream task performance, and diffusion optimizes a bound of the likelihood, making it not directly comparable to ARM. Additionally, we extended the scaling range from 10 18∼10 20 10^{18}\sim 10^{20} FLOPs in Nie et al. [[27](https://arxiv.org/html/2502.09992v3#bib.bib27)] to 10 20∼10 23 10^{20}\sim 10^{23} FLOPs in this work.

### 3.2 Benchmark Results

To comprehensively evaluate the _in-context learning_ and _instruction-following_ capabilities of LLaDA 8B, we conducted detailed comparisons with existing LLMs[[6](https://arxiv.org/html/2502.09992v3#bib.bib6), [21](https://arxiv.org/html/2502.09992v3#bib.bib21), [25](https://arxiv.org/html/2502.09992v3#bib.bib25), [26](https://arxiv.org/html/2502.09992v3#bib.bib26), [32](https://arxiv.org/html/2502.09992v3#bib.bib32), [33](https://arxiv.org/html/2502.09992v3#bib.bib33)] of similar scale. Task selection and evaluation protocols followed existing studies, covering popular benchmarks in general tasks, mathematics, code, and Chinese. Further details are provided in Appendix[B.6](https://arxiv.org/html/2502.09992v3#A2.SS6 "B.6 Standard Benchmarks and Evaluation Details ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). For a more direct comparison, we re-evaluated representative LLMs[[6](https://arxiv.org/html/2502.09992v3#bib.bib6), [21](https://arxiv.org/html/2502.09992v3#bib.bib21)] in our implementation.

As shown in Tab.[1](https://arxiv.org/html/2502.09992v3#S3.T1 "Table 1 ‣ 3.2 Benchmark Results ‣ 3 Experiments ‣ Large Language Diffusion Models"), after pretraining on 2.3T tokens, LLaDA 8B Base demonstrates remarkable performance, surpassing LLaMA2 7B Base on nearly all tasks, and is overall competitive with LLaMA3 8B Base. LLaDA shows advantages in math and Chinese tasks. We conjecture that the strengths stem from the same factors as its relatively weaker performance in some tasks—differences in data quality and distribution, largely due to the closed-source situation of LLM datasets.

Table 1: Benchmark Results of Pre-trained LLMs.∗ indicates that models are evaluated under the same protocol, detailed in Appendix[B.6](https://arxiv.org/html/2502.09992v3#A2.SS6 "B.6 Standard Benchmarks and Evaluation Details ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). Results indicated by † and ¶ are sourced from Yang et al. [[25](https://arxiv.org/html/2502.09992v3#bib.bib25), [26](https://arxiv.org/html/2502.09992v3#bib.bib26)] and Bi et al. [[32](https://arxiv.org/html/2502.09992v3#bib.bib32)] respectively. The numbers in parentheses represent the number of shots used for in-context learning. “-” indicates unknown data.

LLaDA 8B∗LLaMA3 8B∗LLaMA2 7B∗Qwen2 7B†Qwen2.5 7B†Mistral 7B†Deepseek 7B¶Model Diffusion AR AR AR AR AR AR Training tokens 2.3T 15T 2T 7T 18T-2T General Tasks MMLU 65.9 (5)65.4 (5)45.9 (5)70.3 (5)74.2 (5)64.2 (5)48.2 (5)BBH 49.7 (3)62.1 (3)39.4 (3)62.3 (3)70.4 (3)56.1 (3)39.5 (3)ARC-C 45.9 (0)53.1 (0)46.3 (0)60.6 (25)63.7 (25)60.0 (25)48.1 (0)Hellaswag 70.5 (0)79.1 (0)76.0 (0)80.7 (10)80.2 (10)83.3 (10)75.4 (0)TruthfulQA 46.1 (0)44.0 (0)39.0 (0)54.2 (0)56.4 (0)42.2 (0)-WinoGrande 74.8 (5)77.3 (5)72.5 (5)77.0 (5)75.9 (5)78.4 (5)70.5 (0)PIQA 73.6 (0)80.6 (0)79.1 (0)---79.2 (0)Mathematics & Science GSM8K 70.3 (4)48.7 (4)13.1 (4)80.2 (4)85.4 (4)36.2 (4)17.4 (8)Math 31.4 (4)16.0 (4)4.3 (4)43.5 (4)49.8 (4)10.2 (4)6.0 (4)GPQA 25.2 (5)25.9 (5)25.7 (5)30.8 (5)36.4 (5)24.7 (5)-Code HumanEval 35.4 (0)34.8 (0)12.8 (0)51.2 (0)57.9 (0)29.3 (0)26.2 (0)HumanEval-FIM 73.8 (2)73.3 (2)26.9 (2)----MBPP 40.0 (4)48.8 (4)23.2 (4)64.2 (0)74.9 (0)51.1 (0)39.0 (3)Chinese CMMLU 69.9 (5)50.7 (5)32.5 (5)83.9 (5)--47.2 (5)C-Eval 70.5 (5)51.7 (5)34.0 (5)83.2 (5)--45.0 (5)

Table 2: Benchmark Results of Post-trained LLMs. LLaDA only employs an SFT procedure, while other models have extra reinforcement learning (RL) alignment. ∗ indicates models are evaluated under the same protocol, detailed in Appendix[B.6](https://arxiv.org/html/2502.09992v3#A2.SS6 "B.6 Standard Benchmarks and Evaluation Details ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). Results indicated by † and ¶ are sourced from Yang et al. [[26](https://arxiv.org/html/2502.09992v3#bib.bib26)] and Bi et al. [[32](https://arxiv.org/html/2502.09992v3#bib.bib32)] respectively. The numbers in parentheses represent the number of shots used for in-context learning. “-” indicates unknown data.

LLaDA 8B∗LLaMA3 8B∗LLaMA2 7B∗Qwen2 7B†Qwen2.5 7B†Gemma2 9B†Deepseek 7B¶Model Diffusion AR AR AR AR AR AR Training tokens 2.3T 15T 2T 7T 18T 8T 2T Post-training SFT SFT+RL SFT+RL SFT+RL SFT+RL SFT+RL SFT+RL Alignment pairs 4.5M--0.5M + -1M + 0.15M-1.5M + -General Tasks MMLU 65.5 (5)68.4 (5)44.1 (5)---49.4 (0)MMLU-pro 37.0 (0)41.9 (0)4.6 (0)44.1 (5)56.3 (5)52.1 (5)-Hellaswag 74.6 (0)75.5 (0)51.5 (0)---68.5 (-)ARC-C 88.5 (0)82.4 (0)57.3 (0)---49.4 (-)Mathematics & Science GSM8K 69.4 (4)78.3 (4)29.0 (4)85.7 (0)91.6 (0)76.7 (0)63.0 (0)Math 31.9 (0)29.6 (0)3.8 (0)52.9 (0)75.5 (0)44.3 (0)15.8 (0)GPQA 33.3 (5)31.9 (5)28.4 (5)34.3 (0)36.4 (0)32.8 (0)-Code HumanEval 49.4 (0)59.8 (0)16.5 (0)79.9 (0)84.8 (0)68.9 (0)48.2 (-)MBPP 41.0 (4)57.6 (4)20.6 (4)67.2 (0)79.2 (0)74.9 (0)35.2 (-)

Notably, we have carefully ruled out the possibility of data leakage by taking GSM8K as an example. First, as shown in Fig.[3](https://arxiv.org/html/2502.09992v3#S3.F3 "Figure 3 ‣ 3 Experiments ‣ Large Language Diffusion Models"), LLaDA outperformed ARM baselines regarding GSM8K. Moreover, the conclusion remains on a fully unseen GSM8K-like task[[34](https://arxiv.org/html/2502.09992v3#bib.bib34)] in Appendix[B.8](https://arxiv.org/html/2502.09992v3#A2.SS8 "B.8 Evaluation on iGSM Dataset ‣ Appendix B Experiments ‣ Large Language Diffusion Models").

Further, Tab.[2](https://arxiv.org/html/2502.09992v3#S3.T2 "Table 2 ‣ 3.2 Benchmark Results ‣ 3 Experiments ‣ Large Language Diffusion Models") compares the performance of LLaDA 8B Instruct with existing LLMs. SFT improved LLaDA’s performance on most downstream tasks. A few metrics, such as MMLU, showed declines, possibly due to the suboptimal quality of the SFT data. Overall, since we did not perform alignment with reinforcement learning (RL), our results are slightly behind LLaMA3 8B Instruct, though the gaps in many metrics remain small. Notably, even with only SFT, LLaDA demonstrates impressive instruction-following abilities, as detailed in Sec.[3.4](https://arxiv.org/html/2502.09992v3#S3.SS4 "3.4 Case Studies ‣ 3 Experiments ‣ Large Language Diffusion Models"). We leave RL-based alignment for future work.

All results in Sec.[3](https://arxiv.org/html/2502.09992v3#S3 "3 Experiments ‣ Large Language Diffusion Models") are based on pure diffusion methods, as they achieve better overall performance than approaches incorporating autoregressive components. Specifically, we use Eq.([6](https://arxiv.org/html/2502.09992v3#S2.E6 "Equation 6 ‣ 2.4 Inference ‣ 2 Approach ‣ Large Language Diffusion Models")) for conditional likelihood estimation and apply low-confidence remasking for sampling. For LLaDA 8B Instruct, block diffusion style sampling performs better on GSM8K and Math, with scores of 78.6 and 42.2, compared to 69.4 and 31.9 in Tab.[2](https://arxiv.org/html/2502.09992v3#S3.T2 "Table 2 ‣ 3.2 Benchmark Results ‣ 3 Experiments ‣ Large Language Diffusion Models"). This gain is due to extensive |EOS||\text{EOS}| token padding in the SFT data, causing early termination in low-confidence remasking. Please refer to Appendix[B.4](https://arxiv.org/html/2502.09992v3#A2.SS4 "B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for details.

Overall, despite the lack of data transparency, we have made every effort to adopt standardized procedures and introduce diverse tasks, we believe they sufficiently demonstrate the extraordinary capabilities of LLaDA, which is the only competitive non-autoregressive model to our knowledge.

### 3.3 Reversal Reasoning and Analyses

Table 3: Visualization of the Sampling Process and a Generated Multi-round Dialogue. In the response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.

To quantify the reversal reasoning[[15](https://arxiv.org/html/2502.09992v3#bib.bib15)] ability of models, we follow the protocol established in Allen-Zhu and Li [[35](https://arxiv.org/html/2502.09992v3#bib.bib35)]. Specifically, we construct a dataset of 496 famous Chinese poem sentence pairs. Given a sentence from a poem, models are tasked with generating the subsequent line (forward) or the preceding line (reversal) without additional fine-tuning. Examples can be found in[Section˜B.9](https://arxiv.org/html/2502.09992v3#A2.SS9 "B.9 Poem Completion Tasks ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). This setting provides a straightforward and more realistic evaluation compared to previous studies[[27](https://arxiv.org/html/2502.09992v3#bib.bib27), [36](https://arxiv.org/html/2502.09992v3#bib.bib36)].

As shown in Tab.[4](https://arxiv.org/html/2502.09992v3#S3.T4 "Table 4 ‣ 3.3 Reversal Reasoning and Analyses ‣ 3 Experiments ‣ Large Language Diffusion Models"), LLaDA effectively addresses the _reversal curse_[[15](https://arxiv.org/html/2502.09992v3#bib.bib15)], demonstrating consistent zero-shot performance across both forward and reversal tasks. In contrast, both Qwen 2.5 and GPT-4o exhibit a significant gap between the two. The results on forward generation confirm that both ARMs are strong, benefiting from significantly larger datasets and greater computational resources than LLaDA. However, LLaDA outperforms both by a large margin in the reversal task.

Table 4: Comparison on the Poem Completion task.

We did not design anything special for reversal tasks. Intuitively, LLaDA treats tokens uniformly without inductive bias, leading to balanced performance. See Appendix[A.2](https://arxiv.org/html/2502.09992v3#A1.SS2 "A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models") for details.

We also analyze the effect of different sampling strategies for LLaDA, including autoregressive sampling, block diffusion[[31](https://arxiv.org/html/2502.09992v3#bib.bib31)] sampling, and pure diffusion sampling, showing that pure diffusion sampling achieves the best overall performance, as detailed in Appendix[B.4](https://arxiv.org/html/2502.09992v3#A2.SS4 "B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models").

In addition, we examine LLaDA’s sampling speed and memory consumption, showing that it enables a flexible trade-off between generation quality and speed. See Appendix[B.7](https://arxiv.org/html/2502.09992v3#A2.SS7 "B.7 Analysis of Sampling Efficiency ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for more details.

Classifier-free guidance (CFG)[[37](https://arxiv.org/html/2502.09992v3#bib.bib37), [27](https://arxiv.org/html/2502.09992v3#bib.bib27)] is a widely used technique in diffusion models to improve generation quality. To ensure a fair comparison with ARMs, we do not apply CFG to LLaDA in the main text. However, we show that LLaDA is compatible with CFG and consistently benefits from its application. See Appendix[B.3](https://arxiv.org/html/2502.09992v3#A2.SS3 "B.3 Ablation on Classifier-free Guidance ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for more details.

### 3.4 Case Studies

We present samples generated by LLaDA 8B Instruct in Tab.[3](https://arxiv.org/html/2502.09992v3#S3.T3 "Table 3 ‣ 3.3 Reversal Reasoning and Analyses ‣ 3 Experiments ‣ Large Language Diffusion Models"), showcasing its instruction-following capabilities. First, the table illustrates LLaDA’s ability to generate coherent, fluent, and extended text in a non-autoregressive manner. Second, it highlights the model’s multi-turn dialogue capability, effectively retaining conversation history and producing contextually appropriate responses across multiple languages. Such _chat_ capabilities of LLaDA are impressive, as it departs from conventional ARMs for the first time, to the best of our knowledge. See more case studies in Appendix[B.10](https://arxiv.org/html/2502.09992v3#A2.SS10 "B.10 More Case Studies ‣ Appendix B Experiments ‣ Large Language Diffusion Models").

4 Related Work
--------------

Diffusion models[[38](https://arxiv.org/html/2502.09992v3#bib.bib38), [39](https://arxiv.org/html/2502.09992v3#bib.bib39), [40](https://arxiv.org/html/2502.09992v3#bib.bib40)] have achieved remarkable success in visual domains but remain unverified for large-scale (e.g., models trained with over 10 23 10^{23} FLOPs) language modeling, despite growing interest and extensive research efforts.

A simple approach is to continuousize text data and apply continuous diffusion models directly[[41](https://arxiv.org/html/2502.09992v3#bib.bib41), [42](https://arxiv.org/html/2502.09992v3#bib.bib42), [43](https://arxiv.org/html/2502.09992v3#bib.bib43), [44](https://arxiv.org/html/2502.09992v3#bib.bib44), [45](https://arxiv.org/html/2502.09992v3#bib.bib45), [46](https://arxiv.org/html/2502.09992v3#bib.bib46), [47](https://arxiv.org/html/2502.09992v3#bib.bib47), [48](https://arxiv.org/html/2502.09992v3#bib.bib48), [49](https://arxiv.org/html/2502.09992v3#bib.bib49), [50](https://arxiv.org/html/2502.09992v3#bib.bib50), [51](https://arxiv.org/html/2502.09992v3#bib.bib51)]. Alternatively, some methods model continuous parameters of discrete distributions instead[[52](https://arxiv.org/html/2502.09992v3#bib.bib52), [53](https://arxiv.org/html/2502.09992v3#bib.bib53), [54](https://arxiv.org/html/2502.09992v3#bib.bib54), [55](https://arxiv.org/html/2502.09992v3#bib.bib55), [56](https://arxiv.org/html/2502.09992v3#bib.bib56)]. However, scalability remains a significant challenge for these approaches. For instance, a 1B model may require 64 times the compute of an ARM to achieve comparable performance[[57](https://arxiv.org/html/2502.09992v3#bib.bib57)].

Another approach replaces continuous diffusion with discrete processes featuring new forward and reverse dynamics, leading to numerous variants[[58](https://arxiv.org/html/2502.09992v3#bib.bib58), [59](https://arxiv.org/html/2502.09992v3#bib.bib59), [60](https://arxiv.org/html/2502.09992v3#bib.bib60), [61](https://arxiv.org/html/2502.09992v3#bib.bib61), [62](https://arxiv.org/html/2502.09992v3#bib.bib62), [63](https://arxiv.org/html/2502.09992v3#bib.bib63), [64](https://arxiv.org/html/2502.09992v3#bib.bib64), [65](https://arxiv.org/html/2502.09992v3#bib.bib65), [66](https://arxiv.org/html/2502.09992v3#bib.bib66), [67](https://arxiv.org/html/2502.09992v3#bib.bib67), [68](https://arxiv.org/html/2502.09992v3#bib.bib68), [69](https://arxiv.org/html/2502.09992v3#bib.bib69), [70](https://arxiv.org/html/2502.09992v3#bib.bib70), [71](https://arxiv.org/html/2502.09992v3#bib.bib71)]. The original diffusion model paper[[38](https://arxiv.org/html/2502.09992v3#bib.bib38)] introduced both continuous-state and discrete-state transition kernels under a unified diffusion framework. Austin et al. [[16](https://arxiv.org/html/2502.09992v3#bib.bib16)] was among the pioneering works that introduced discrete diffusion models into language modeling, demonstrating the feasibility of this approach. Lou et al. [[17](https://arxiv.org/html/2502.09992v3#bib.bib17)] showed that masked diffusion, as a special case of discrete diffusion, achieves perplexity comparable to or surpassing ARMs at GPT-2 scale. Shi et al. [[18](https://arxiv.org/html/2502.09992v3#bib.bib18)], Sahoo et al. [[19](https://arxiv.org/html/2502.09992v3#bib.bib19)], Ou et al. [[20](https://arxiv.org/html/2502.09992v3#bib.bib20)] established fundamental theoretical results, which motivated our model design, training, and inference (see Appendix[A](https://arxiv.org/html/2502.09992v3#A1 "Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models") for details). Nie et al. [[27](https://arxiv.org/html/2502.09992v3#bib.bib27)] introduced the scaling laws for MDMs in language modeling and explored how MDMs can be leveraged for language tasks such as question answering at the GPT-2 scale. Gong et al. [[72](https://arxiv.org/html/2502.09992v3#bib.bib72)] demonstrated the potential of fine-tuning an ARM within the MDM framework. However, the improvements observed by Gong et al. [[72](https://arxiv.org/html/2502.09992v3#bib.bib72)] are limited to specific metrics, and their approach does not address the performance achievable through pure diffusion-based training. Concurrent work[[73](https://arxiv.org/html/2502.09992v3#bib.bib73)] demonstrates the potential of diffusion language models in code generation and highlights their advantages in inference efficiency. Nonetheless, as it is a closed-source product, specific details such as training procedures and sampling methods remain unknown.

In comparison, this study scales MDM to an unprecedented size of 8B parameters from scratch, achieving performance comparable to leading LLMs such as LLaMA 3.

Additionally, a parallel line of work on image generation[[23](https://arxiv.org/html/2502.09992v3#bib.bib23), [74](https://arxiv.org/html/2502.09992v3#bib.bib74), [75](https://arxiv.org/html/2502.09992v3#bib.bib75)] aligns well with the application of MDMs to text data. Moreover, MDMs have also shown promise in other domains such as protein generation[[76](https://arxiv.org/html/2502.09992v3#bib.bib76), [77](https://arxiv.org/html/2502.09992v3#bib.bib77)], where they have achieved promising results. Notably, a series of studies[[31](https://arxiv.org/html/2502.09992v3#bib.bib31), [78](https://arxiv.org/html/2502.09992v3#bib.bib78), [79](https://arxiv.org/html/2502.09992v3#bib.bib79), [80](https://arxiv.org/html/2502.09992v3#bib.bib80), [81](https://arxiv.org/html/2502.09992v3#bib.bib81), [82](https://arxiv.org/html/2502.09992v3#bib.bib82), [83](https://arxiv.org/html/2502.09992v3#bib.bib83), [84](https://arxiv.org/html/2502.09992v3#bib.bib84), [85](https://arxiv.org/html/2502.09992v3#bib.bib85), [86](https://arxiv.org/html/2502.09992v3#bib.bib86), [87](https://arxiv.org/html/2502.09992v3#bib.bib87)] have explored techniques such as architectural optimization, distillation, and sampling algorithm design to accelerate MDMs sampling.

5 Conclusion and Discussion
---------------------------

We introduce LLaDA, a diffusion language model trained from scratch with an unprecedented scale of 8B parameters. LLaDA demonstrates strong capabilities in scalability, in-context learning, and instruction-following, achieving performance comparable to strong LLMs such as LLaMA3. In addition, LLaDA offers unique advantages, such as bidirectional modeling and enhanced robustness, effectively addressing the relevant limitations of existing LLMs. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that these essential capabilities are inherently tied to ARMs. These results represent a new paradigm for language modeling and uncover novel insights, demonstrating a high degree of scientific innovation.

Limitations. While promising, the full potential of diffusion models remains to be fully explored. Several limitations of this work present significant opportunities for future research. The generation length is a user-specified hyperparameter. Although LLaDA is insensitive to this hyperparameter as detailed in Appendix[B.5](https://arxiv.org/html/2502.09992v3#A2.SS5 "B.5 Ablation on Generated Length ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), we believe that adopting an adaptive generation length would offer a more efficient solution. Due to computational constraints, direct comparisons between LLaDA and ARMs—such as training on identical datasets—were restricted to a computational budget of less than 10 23 10^{23} FLOPs. To allocate resources for training the largest possible LLaDA model and showcasing its potential, we were unable to scale the ARM baseline to the same extent. Moreover, no specialized attention mechanisms or position embeddings were designed for LLaDA, nor were any system-level architectural optimizations such as KV cache applied. On the inference side, more efficient and controllable[[37](https://arxiv.org/html/2502.09992v3#bib.bib37), [88](https://arxiv.org/html/2502.09992v3#bib.bib88), [89](https://arxiv.org/html/2502.09992v3#bib.bib89)] sampling algorithms remain preliminary. Furthermore, LLaDA has yet to undergo alignment with reinforcement learning[[90](https://arxiv.org/html/2502.09992v3#bib.bib90), [91](https://arxiv.org/html/2502.09992v3#bib.bib91)], which is crucial for improving its performance and alignment with human intent.

Looking ahead, both the model scale and the amount of training data for LLaDA remain smaller than those of leading ARM counterparts[[6](https://arxiv.org/html/2502.09992v3#bib.bib6), [26](https://arxiv.org/html/2502.09992v3#bib.bib26), [92](https://arxiv.org/html/2502.09992v3#bib.bib92), [93](https://arxiv.org/html/2502.09992v3#bib.bib93), [94](https://arxiv.org/html/2502.09992v3#bib.bib94), [95](https://arxiv.org/html/2502.09992v3#bib.bib95)], highlighting the need for further scaling to fully evaluate its capabilities. In addition, LLaDA’s ability to process multi-modal data remains unexplored. Its impact on prompt tuning techniques[[96](https://arxiv.org/html/2502.09992v3#bib.bib96)] and integration into agent-based systems[[97](https://arxiv.org/html/2502.09992v3#bib.bib97), [98](https://arxiv.org/html/2502.09992v3#bib.bib98)] is still not fully understood. Finally, a systematic investigation into post-training for LLaDA (e.g., O1-like systems[[99](https://arxiv.org/html/2502.09992v3#bib.bib99), [100](https://arxiv.org/html/2502.09992v3#bib.bib100)]) is needed to further unlock the potential of diffusion language models.

Acknowledgements
----------------

This work was supported by the National Natural Science Foundation of China (No. 92470118); Beijing Natural Science Foundation (No. L247030); Beijing Nova Program (No. 20220484044); and Ant Group Research Fund.

References
----------

*   Zhao et al. [2023] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. _arXiv preprint arXiv:2303.18223_, 2023. 
*   Radford [2018] Alec Radford. Improving language understanding by generative pre-training, 2018. 
*   Radford et al. [2019] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019. 
*   Brown [2020] Tom B Brown. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_, 2020. 
*   OpenAI [2022] OpenAI. ChatGPT: Optimizing Language Models for Dialogue. _OpenAI blog_, November 2022. URL [https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/). 
*   Dubey et al. [2024] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_, 2024. 
*   Vaswani [2017] Ashish Vaswani. Attention is all you need. _arXiv preprint arXiv:1706.03762_, 2017. 
*   Fisher [1922] Ronald A Fisher. On the mathematical foundations of theoretical statistics. _Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character_, 222(594-604):309–368, 1922. 
*   Bao et al. [2023] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 22669–22679, 2023. 
*   Peebles and Xie [2023] William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 4195–4205, 2023. 
*   Brooks et al. [2024] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL [https://openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators). 
*   [12] Gregoire Deletang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. Language modeling is compression. In _The Twelfth International Conference on Learning Representations_. 
*   Huang et al. [2024a] Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. Compression represents intelligence linearly. _arXiv preprint arXiv:2404.09937_, 2024a. 
*   Shannon [1948] Claude Elwood Shannon. A mathematical theory of communication. _The Bell system technical journal_, 27(3):379–423, 1948. 
*   Berglund et al. [2023] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". _arXiv preprint arXiv:2309.12288_, 2023. 
*   Austin et al. [2021a] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. _Advances in Neural Information Processing Systems_, 34:17981–17993, 2021a. 
*   Lou et al. [2023] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. _arXiv preprint arXiv:2310.16834_, 2023. 
*   Shi et al. [2024] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. _arXiv preprint arXiv:2406.04329_, 2024. 
*   Sahoo et al. [2024] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. _arXiv preprint arXiv:2406.07524_, 2024. 
*   Ou et al. [2024] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. _arXiv preprint arXiv:2406.03736_, 2024. 
*   Touvron et al. [2023] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023. 
*   Devlin [2018] Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_, 2018. 
*   Chang et al. [2022] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 11315–11325, 2022. 
*   Ainslie et al. [2023] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 4895–4901, 2023. 
*   Yang et al. [2024a] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024a. URL [https://arxiv.org/abs/2407.10671](https://arxiv.org/abs/2407.10671). 
*   Yang et al. [2024b] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. _arXiv preprint arXiv:2412.15115_, 2024b. 
*   Nie et al. [2024] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. _arXiv preprint arXiv:2410.18514_, 2024. 
*   Hu et al. [2024] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. _arXiv preprint arXiv:2404.06395_, 2024. 
*   Loshchilov [2017] I Loshchilov. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_, 2017. 
*   Holtzman et al. [2019] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. _arXiv preprint arXiv:1904.09751_, 2019. 
*   Arriola et al. [2025] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. _arXiv preprint arXiv:2503.09573_, 2025. 
*   Bi et al. [2024] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. _arXiv preprint arXiv:2401.02954_, 2024. 
*   Jiang et al. [2023] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_, 2023. 
*   Ye et al. [2024] Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process. _ArXiv e-prints_, abs/2407.20311, July 2024. Full version available at [http://arxiv.org/abs/2407.20311](http://arxiv.org/abs/2407.20311). 
*   Allen-Zhu and Li [2023] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 3.2, Knowledge Manipulation. _ArXiv e-prints_, abs/2309.14402, September 2023. Full version available at [http://arxiv.org/abs/2309.14402](http://arxiv.org/abs/2309.14402). 
*   Kitouni et al. [2024] Ouail Kitouni, Niklas Nolte, Diane Bouchacourt, Adina Williams, Mike Rabbat, and Mark Ibrahim. The factorization curse: Which tokens you predict underlie the reversal curse and more. _arXiv preprint arXiv:2406.05183_, 2024. 
*   Ho and Salimans [2022] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. _arXiv preprint arXiv:2207.12598_, 2022. 
*   Sohl-Dickstein et al. [2015] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In _International conference on machine learning_, pages 2256–2265. PMLR, 2015. 
*   Ho et al. [2020] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in neural information processing systems_, 33:6840–6851, 2020. 
*   Song et al. [2020] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint arXiv:2011.13456_, 2020. 
*   Li et al. [2022] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. _Advances in Neural Information Processing Systems_, 35:4328–4343, 2022. 
*   Gong et al. [2022] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. _arXiv preprint arXiv:2210.08933_, 2022. 
*   Han et al. [2022] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. _arXiv preprint arXiv:2210.17432_, 2022. 
*   Strudel et al. [2022] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, et al. Self-conditioned embedding diffusion for text generation. _arXiv preprint arXiv:2211.04236_, 2022. 
*   Chen et al. [2022] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. _arXiv preprint arXiv:2208.04202_, 2022. 
*   Dieleman et al. [2022] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. _arXiv preprint arXiv:2211.15089_, 2022. 
*   Richemond et al. [2022] Pierre H. Richemond, Sander Dieleman, and Arnaud Doucet. Categorical sdes with simplex diffusion, 2022. 
*   Wu et al. [2023] Tong Wu, Zhihao Fan, Xiao Liu, Yeyun Gong, Yelong Shen, Jian Jiao, Hai-Tao Zheng, Juntao Li, Zhongyu Wei, Jian Guo, Nan Duan, and Weizhu Chen. Ar-diffusion: Auto-regressive diffusion model for text generation, 2023. 
*   Mahabadi et al. [2024] Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E. Peters, and Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion, 2024. 
*   Ye et al. [2023a] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Mingxuan Wang. Dinoiser: Diffused conditional sequence learning by manipulating noises. _arXiv preprint arXiv:2302.10025_, 2023a. 
*   Zhang et al. [2023] Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, and Navdeep Jaitly. Planner: Generating diversified paragraph via latent language diffusion model. _Advances in Neural Information Processing Systems_, 36:80178–80190, 2023. 
*   Lou and Ermon [2023] Aaron Lou and Stefano Ermon. Reflected diffusion models, 2023. 
*   Graves et al. [2023] Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks. _arXiv preprint arXiv:2308.07037_, 2023. 
*   Lin et al. [2023] Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, and Weizhu Chen. Text generation with diffusion language models: A pre-training approach with continuous paragraph denoise. In _International Conference on Machine Learning_, pages 21051–21064. PMLR, 2023. 
*   Xue et al. [2024] Kaiwen Xue, Yuhao Zhou, Shen Nie, Xu Min, Xiaolu Zhang, Jun Zhou, and Chongxuan Li. Unifying bayesian flow networks and diffusion models through stochastic differential equations. _arXiv preprint arXiv:2404.15766_, 2024. 
*   Zhang et al. [2025] Ruixiang Zhang, Shuangfei Zhai, Yizhe Zhang, James Thornton, Zijing Ou, Joshua Susskind, and Navdeep Jaitly. Target concrete score matching: A holistic framework for discrete diffusion. _arXiv preprint arXiv:2504.16431_, 2025. 
*   Gulrajani and Hashimoto [2024] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Hoogeboom et al. [2021a] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. _Advances in Neural Information Processing Systems_, 34:12454–12465, 2021a. 
*   Hoogeboom et al. [2021b] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. _arXiv preprint arXiv:2110.02037_, 2021b. 
*   He et al. [2022] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. _arXiv preprint arXiv:2211.15029_, 2022. 
*   Campbell et al. [2022] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. _Advances in Neural Information Processing Systems_, 35:28266–28279, 2022. 
*   Meng et al. [2022] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. _Advances in Neural Information Processing Systems_, 35:34532–34545, 2022. 
*   Reid et al. [2022] Machel Reid, Vincent J. Hellendoorn, and Graham Neubig. Diffuser: Discrete diffusion via edit-based reconstruction, 2022. 
*   Sun et al. [2022] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. _arXiv preprint arXiv:2211.16750_, 2022. 
*   Kitouni et al. [2023] Ouail Kitouni, Niklas Nolte, James Hensman, and Bhaskar Mitra. Disk: A diffusion model for structured knowledge. _arXiv preprint arXiv:2312.05253_, 2023. 
*   Zheng et al. [2023] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. _ArXiv_, abs/2302.05737, 2023. 
*   Chen et al. [2023] Zixiang Chen, Huizhuo Yuan, Yongqian Li, Yiwen Kou, Junkai Zhang, and Quanquan Gu. Fast sampling via de-randomization for discrete diffusion models. _arXiv preprint arXiv:2312.09193_, 2023. 
*   Ye et al. [2023b] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Quanquan Gu. Diffusion language models can perform many tasks with scaling and instruction-finetuning. _arXiv preprint arXiv:2308.12219_, 2023b. 
*   Gat et al. [2024] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. _arXiv preprint arXiv:2407.15595_, 2024. 
*   Zheng et al. [2024a] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling, 2024a. URL [https://arxiv.org/abs/2409.02908](https://arxiv.org/abs/2409.02908). 
*   Kapur et al. [2024] Shreyas Kapur, Erik Jenner, and Stuart Russell. Diffusion on syntax trees for program synthesis. _arXiv preprint arXiv:2405.20519_, 2024. 
*   Gong et al. [2024] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. _arXiv preprint arXiv:2410.17891_, 2024. 
*   Khanna et al. [2025] Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, et al. Mercury: Ultra-fast language models based on diffusion. _arXiv preprint arXiv:2506.17298_, 2025. 
*   Chang et al. [2023] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. _arXiv preprint arXiv:2301.00704_, 2023. 
*   You et al. [2025] Zebin You, Jingyang Ou, Xiaolu Zhang, Jun Hu, Jun Zhou, and Chongxuan Li. Effective and efficient masked image generation models. _arXiv preprint arXiv:2503.07197_, 2025. 
*   Wang et al. [2024a] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Diffusion language models are versatile protein learners. _arXiv preprint arXiv:2402.18567_, 2024a. 
*   Wang et al. [2024b] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Dplm-2: A multimodal diffusion protein language model. _arXiv preprint arXiv:2410.13782_, 2024b. 
*   Kou et al. [2024] Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. _arXiv preprint arXiv:2403.00835_, 2024. 
*   Xu et al. [2025] Chenkai Xu, Xu Wang, Zhenyi Liao, Yishun Li, Tianqi Hou, and Zhijie Deng. Show-o turbo: Towards accelerated unified multimodal understanding and generation. _arXiv preprint arXiv:2502.05415_, 2025. 
*   Liu et al. [2024a] Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi Jaakkola, and Rafael Gómez-Bombarelli. Think while you generate: Discrete diffusion with planned denoising. _arXiv preprint arXiv:2410.06264_, 2024a. 
*   Zhu et al. [2025] Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, and Vicky Kalogeiton. Dimo: Distilling masked diffusion models into one-step generator. _arXiv preprint arXiv:2503.15457_, 2025. 
*   Ren et al. [2025] Yinuo Ren, Haoxuan Chen, Yuchen Zhu, Wei Guo, Yongxin Chen, Grant M Rotskoff, Molei Tao, and Lexing Ying. Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms. _arXiv preprint arXiv:2502.00234_, 2025. 
*   Hayakawa et al. [2024] Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. _arXiv preprint arXiv:2410.08709_, 2024. 
*   Zhao et al. [2024] Yixiu Zhao, Jiaxin Shi, Feng Chen, Shaul Druckmann, Lester Mackey, and Scott Linderman. Informed correctors for discrete diffusion models. _arXiv preprint arXiv:2407.21243_, 2024. 
*   Zheng et al. [2024b] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. _arXiv preprint arXiv:2409.02908_, 2024b. 
*   Park et al. [2024] Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, and Yuki Mitsufuji. Jump your steps: Optimizing sampling schedule of discrete diffusion models. In _The Thirteenth International Conference on Learning Representations_, 2024. 
*   Deschenaux and Gulcehre [2024] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time. _arXiv preprint arXiv:2410.21035_, 2024. 
*   Dhariwal and Nichol [2021] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. _Advances in neural information processing systems_, 34:8780–8794, 2021. 
*   Schiff et al. [2024] Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. _arXiv preprint arXiv:2412.10193_, 2024. 
*   Ouyang et al. [2022] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_, 35:27730–27744, 2022. 
*   Rafailov et al. [2024] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Achiam et al. [2023] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_, 2023. 
*   Google [2024] Google. Our next-generation model: Gemini 1.5, 2024. URL [https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024). 
*   Anthropic [2024] Anthropic. Claude 3.5 sonnet, 2024. URL [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet). 
*   Liu et al. [2024b] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. _arXiv preprint arXiv:2412.19437_, 2024b. 
*   Wei et al. [2022] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_, 35:24824–24837, 2022. 
*   Park et al. [2023] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th annual acm symposium on user interface software and technology_, pages 1–22, 2023. 
*   Wang et al. [2024c] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. _Frontiers of Computer Science_, 18(6):186345, 2024c. 
*   OpenAI [2024] OpenAI. Learning to reason with llms, 2024. URL [https://openai.com/index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/). 
*   Guo et al. [2025] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 
*   Uria et al. [2014] Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In _Proceedings of the 31th International Conference on Machine Learning_, 2014. 
*   Shih et al. [2022] Andy Shih, Dorsa Sadigh, and Stefano Ermon. Training and inference on any-order autoregressive models the right way. In _Proceedings of the 31th International Conference on Machine Learning_, 2022. 
*   Xu et al. [2024] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. _arXiv preprint arXiv:2406.08464_, 2024. 
*   Wei et al. [2023] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with oss-instruct. _arXiv preprint arXiv:2312.02120_, 2023. 
*   Zhang and Sennrich [2019] Biao Zhang and Rico Sennrich. Root mean square layer normalization. _Advances in Neural Information Processing Systems_, 32, 2019. 
*   Shazeer [2020] Noam Shazeer. Glu variants improve transformer. _arXiv preprint arXiv:2002.05202_, 2020. 
*   Su et al. [2024] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_, 568:127063, 2024. 
*   Kaplan et al. [2020] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_, 2020. 
*   Hoffmann et al. [2022] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. _arXiv preprint arXiv:2203.15556_, 2022. 
*   Hendrycks et al. [2020] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_, 2020. 
*   Suzgun et al. [2022] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_, 2022. 
*   Clark et al. [2018] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_, 2018. 
*   Zellers et al. [2019] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? _arXiv preprint arXiv:1905.07830_, 2019. 
*   Lin et al. [2021] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. _arXiv preprint arXiv:2109.07958_, 2021. 
*   Sakaguchi et al. [2021] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_, 64(9):99–106, 2021. 
*   Bisk et al. [2020] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_, 2020. 
*   Cobbe et al. [2021] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021. 
*   Hendrycks et al. [2021] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_, 2021. 
*   Rein et al. [2023] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. _arXiv preprint arXiv:2311.12022_, 2023. 
*   Chen et al. [2021] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_, 2021. 
*   Bavarian et al. [2022] Mohammad Bavarian, Heewoo Jun, Nikolas Tezak, John Schulman, Christine McLeavey, Jerry Tworek, and Mark Chen. Efficient training of language models to fill in the middle. _arXiv preprint arXiv:2207.14255_, 2022. 
*   Austin et al. [2021b] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_, 2021b. 
*   Li et al. [2023] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. _arXiv preprint arXiv:2306.09212_, 2023. 
*   Huang et al. [2024b] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. _Advances in Neural Information Processing Systems_, 36, 2024b. 
*   Gao et al. [2024] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL [https://zenodo.org/records/12608602](https://zenodo.org/records/12608602). 

Appendix A Formulation of Masked Diffusion Models
-------------------------------------------------

Algorithm 1 Pre-training of LLaDA

0: mask predictor

p θ p_{\theta}
, data distribution

p data p_{\rm{data}}

1:repeat

2:

x 0∼p data x_{0}\sim p_{\rm{data}}
# with a probability of 1%, the sequence length of x 0 x_{0} follows U​[1,4096]\text{U}[1,4096]

3:

t∼U​(0,1]t\sim\text{U}(0,1]

4:

x t∼q t|0​(x t|x 0)x_{t}\sim q_{t|0}(x_{t}|x_{0})
# q t|0 q_{t|0} is defined in Eq.([7](https://arxiv.org/html/2502.09992v3#A1.E7 "Equation 7 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"))

5: Calculate

ℒ=−1 t∗L​∑i=1 L 1​[x t i=M]​log⁡p θ​(x 0 i|x t)\mathcal{L}=-\frac{1}{t*L}\sum_{i=1}^{L}\textbf{1}[x_{t}^{i}=\textrm{M}]\log p_{\theta}(x_{0}^{i}|x_{t})
# L L is the sequence length of x 0 x_{0}

6: Calculate

∇θ ℒ\nabla_{\theta}\mathcal{L}
and run optimizer.

7:until Converged

8:Return

p θ p_{\theta}

Algorithm 2 Supervised Fine-Tuning of LLaDA

0: mask predictor

p θ p_{\theta}
, pair data distribution

p data p_{\rm{data}}

1:repeat

2:

p 0,r 0∼p data p_{0},r_{0}\sim p_{\rm{data}}
# please refer to Appendix[B.1](https://arxiv.org/html/2502.09992v3#A2.SS1 "B.1 Data Collection and Preprocessing ‣ Appendix B Experiments ‣ Large Language Diffusion Models") for details about the SFT dat

3:

t∼U​(0,1]t\sim\text{U}(0,1]

4:

r t∼q t|0​(r t|r 0)r_{t}\sim q_{t|0}(r_{t}|r_{0})
# q t|0 q_{t|0} is defined in Eq.([7](https://arxiv.org/html/2502.09992v3#A1.E7 "Equation 7 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"))

5: Calculate

ℒ=−1 t∗L′​∑i=1 L′1​[r t i=M]​log⁡p θ​(r 0 i|p 0,r t)\mathcal{L}=-\frac{1}{t*L^{\prime}}\sum_{i=1}^{L^{\prime}}\textbf{1}[r_{t}^{i}=\textrm{M}]\log p_{\theta}(r_{0}^{i}|p_{0},r_{t})
# L′L^{\prime} is the sequence length of r 0 r_{0}

6: Calculate

∇θ ℒ\nabla_{\theta}\mathcal{L}
and run optimizer.

7:until Converged

8:Return

p θ p_{\theta}

Algorithm 3 Conditional Log-likelihood Evaluation of LLaDA

0: mask predictor

p θ p_{\theta}
, prompt

p 0 p_{0}
, response

r 0 r_{0}
, the number of Monte Carlo estimations

n m​c n_{mc}

1:

log​_​likelihood=0\text{log}\_\text{likelihood}=0

2:for

i←1 i\leftarrow 1
to

n m​c n_{mc}
do

3:

l∼{1,2,…,L}l\sim\{1,2,\dots,L\}
# L L is the sequence length of r 0 r_{0}

4: Obtain

r l r_{l}
by uniformly sampling

l l
tokens from

r 0 r_{0}
without replacement for masking

5:

log​_​likelihood=log​_​likelihood+L l​∑i=1 L 1​[r l i=M]​log⁡p θ​(r 0 i|p 0,r l)\text{log}\_\text{likelihood}=\text{log}\_\text{likelihood}+\frac{L}{l}\sum_{i=1}^{L}\textbf{1}[r_{l}^{i}=\textrm{M}]\log p_{\theta}(r_{0}^{i}|p_{0},r_{l})

6:end for

7:

log​_​likelihood=log​_​likelihood/n m​c\text{log}\_\text{likelihood}=\text{log}\_\text{likelihood}/n_{mc}

8:Return

log​_​likelihood\text{log}\_\text{likelihood}

### A.1 Training

MDMs[[16](https://arxiv.org/html/2502.09992v3#bib.bib16), [17](https://arxiv.org/html/2502.09992v3#bib.bib17), [18](https://arxiv.org/html/2502.09992v3#bib.bib18), [19](https://arxiv.org/html/2502.09992v3#bib.bib19), [20](https://arxiv.org/html/2502.09992v3#bib.bib20)] define the model distribution p θ​(x 0)p_{\theta}(x_{0}) in a manner distinct from autoregressive models.

These models introduce a forward process {x t}\{x_{t}\} indexed by a time t∈[0,1]t\in[0,1]. This process gradually and independently masks all tokens in the sequence x 0 x_{0}. At time t=0 t=0, the data point x 0 x_{0} is fully observed with no masks, while for t∈(0,1]t\in(0,1], x t x_{t} represents latent variables with varying mask ratios in expectation.

Formally, the conditional distribution of x t x_{t} given x 0 x_{0} is defined by a fully factorized form:

q t|0​(x t|x 0)=∏i=1 L q t|0​(x t i|x 0 i),\displaystyle q_{t|0}(x_{t}|x_{0})=\prod_{i=1}^{L}q_{t|0}(x_{t}^{i}|x_{0}^{i}),(7)

where the conditional distribution for each token is given by:

q t|0​(x t i|x 0 i)={1−t,x t i=x 0 i,t,x t i=M.\displaystyle q_{t|0}(x_{t}^{i}|x_{0}^{i})=\begin{cases}1-t,&x_{t}^{i}=x_{0}^{i},\\ t,&x_{t}^{i}=\textrm{M}.\end{cases}(8)

Algorithm 4 Random Remasking Strategy of LLaDA

0: mask predictor

p θ p_{\theta}
, prompt

p 0 p_{0}
, answer length

L L
, sampling steps

N N

1: Set

r 1 r_{1}
is a fully masked sequence of length

L L
.

2:for

t←1 t\leftarrow 1
down to

1 N\frac{1}{N}
step

1 N\frac{1}{N}
do

3:

s=t−1 N s=t-\frac{1}{N}

4:

r 0=arg⁡max r 0⁡p θ​(r 0|p 0,r t)r_{0}=\arg\max_{r_{0}}p_{\theta}(r_{0}|p_{0},r_{t})
# we employ greedy sampling when predicting masked tokens

5:for

i←1 i\leftarrow 1
to

L L
do

6:if

r t i≠M r_{t}^{i}\neq\textrm{M}
then

7:

r 0 i=r t i r_{0}^{i}=r_{t}^{i}

8:else

9: with probability

s t\frac{s}{t}
,

r 0 i r_{0}^{i}
is set to M

10:end if

11:end for

12:

r s=r 0 r_{s}=r_{0}

13:end for

14:Return

r 0 r_{0}

Here, M denotes the mask token. Intuitively, each token either remains unchanged or is masked, with the probability of being masked increasing linearly as t t progresses from 0 to 1 1. At t=1 t=1, all tokens are guaranteed to be masked, meaning that x 1 x_{1} follows a Dirac distribution concentrated on a sequence of fully masked tokens. Notably, the linear masking probability is analogous to but distinct from, the noise schedule in continuous diffusion models[[38](https://arxiv.org/html/2502.09992v3#bib.bib38), [39](https://arxiv.org/html/2502.09992v3#bib.bib39), [40](https://arxiv.org/html/2502.09992v3#bib.bib40)]. This linearity is motivated by the assumption that the information in the text is proportional to the number of tokens on average, making it reasonable to lose information linearly during the forward process.

The forward process is not only reversible but also corresponds to a reverse process that is fully factorized across all tokens. The reverse process, from time t=1 t=1 to 0, generates new data from sequences of fully masked tokens. The conditional distribution for the reverse process, for 0≤s<t≤1 0\leq s<t\leq 1, is factorized as:

q s|t​(x s|x t)=∏i=1 L q s|t​(x s i|x t),\displaystyle q_{s|t}(x_{s}|x_{t})=\prod_{i=1}^{L}q_{s|t}(x_{s}^{i}|x_{t}),(9)

where the conditional distribution for each token is:

q s|t​(x s i|x t)={1,x t i≠M,x s i=x t i,s t,x t i=M,x s i=M,t−s t​q 0|t​(x s i|x t),x t i=M,x s i≠M,0,otherwise.\displaystyle q_{s|t}(x_{s}^{i}|x_{t})=\begin{cases}1,&x_{t}^{i}\neq\textrm{M},\,x_{s}^{i}=x_{t}^{i},\\ \frac{s}{t},&x_{t}^{i}=\textrm{M},\,x_{s}^{i}=\textrm{M},\\ \frac{t-s}{t}q_{0|t}(x_{s}^{i}|x_{t}),&x_{t}^{i}=\textrm{M},\,x_{s}^{i}\neq\textrm{M},\\ 0,&\textrm{otherwise}.\end{cases}(10)

Thus, the key function to estimate is the conditional distribution q 0|t​(x s i|x t)q_{0|t}(x_{s}^{i}|x_{t}), which predicts the original token if it is masked in the input x t x_{t}. This is analogous to the _data prediction_ form in continuous diffusion models.

As proven in[[20](https://arxiv.org/html/2502.09992v3#bib.bib20)], an equivalent yet _time-free_ parameterization can be derived as:

q 0|t​(x s i|x t)=p data​(x 0 i|x t UM),∀i​such that​x t i=M,\displaystyle q_{0|t}(x_{s}^{i}|x_{t})=p_{\textrm{data}}(x_{0}^{i}|x_{t}^{\textrm{UM}}),\quad\forall i\textrm{ such that }x_{t}^{i}=\textrm{M},(11)

where x t UM x_{t}^{\textrm{UM}} denotes the collection of unmasked tokens in x t x_{t}, which is identical to the corresponding tokens in the original data x 0 x_{0} since unmasked tokens are solely determined by x 0 x_{0} and are independent of time t t. Intuitively, this implies that estimating the data prediction function is equivalent to estimating the conditional distributions on clean data, which is time-invariant. Consequently, the time t t need not be provided as input to the parametric model.

Although the development of masked diffusion is nontrivial, the implementation is straightforward. We first introduce the _mask predictor_, a parametric model p θ(⋅|x t)p_{\theta}(\cdot|x_{t}) (e.g., a Transformer without causal mask), which takes x t x_{t} for any t t as input and predict all masked tokens simultaneously. Then, we define the model distribution p θ​(x 0)p_{\theta}(x_{0}) as follows: starting with x 1 x_{1} as a sequence of fully masked tokens, we simulate an approximate reverse process parameterized by p θ(⋅|x t)p_{\theta}(\cdot|x_{t}) from t=1 t=1 to 0. The marginal distribution induced at t=0 t=0 then represents the model distribution p θ​(x 0)p_{\theta}(x_{0}).

Formally, the mask predictor is trained using a cross-entropy loss with masking:

ℒ​(θ)≜−𝔼 t,x 0,x t​[1 t​∑i=1 L 1​[x t i=M]​log⁡p θ​(x 0 i|x t)],\displaystyle\mathcal{L}(\theta)\triangleq-\mathbb{E}_{t,x_{0},x_{t}}\left[\frac{1}{t}\sum_{i=1}^{L}\textbf{1}[x_{t}^{i}=\textrm{M}]\log p_{\theta}(x_{0}^{i}|x_{t})\right],(12)

where x 0 x_{0} is sampled from the training data, t t is sampled uniformly from [0,1][0,1], and x t x_{t} is sampled from q t|0​(x t|x 0)q_{t|0}(x_{t}|x_{0}). The indicator function 1​[⋅]\textbf{1}[\cdot] ensures that the cross-entropy loss is computed only for masked tokens. In Shi et al. [[18](https://arxiv.org/html/2502.09992v3#bib.bib18)], Sahoo et al. [[19](https://arxiv.org/html/2502.09992v3#bib.bib19)], Ou et al. [[20](https://arxiv.org/html/2502.09992v3#bib.bib20)], it has been proven that the loss function ℒ​(θ)\mathcal{L}(\theta) is an upper bound on the negative log-likelihood of the model distribution:

−𝔼 x 0∼p data​(x 0)​[log⁡p θ​(x 0)]≤ℒ​(θ).\displaystyle-\mathbb{E}_{x_{0}\sim p_{\textrm{data}(x_{0})}}\left[\log p_{\theta}(x_{0})\right]\leq\mathcal{L}(\theta).(13)

In summary, this principled approach trains a generative model by progressively masking tokens during a forward process and learning to recover the data distribution during a reverse process, all under the (approximate) maximum likelihood estimation framework.

Algorithm 5 Low-confidence Remasking Strategy of LLaDA

0: mask predictor

p θ p_{\theta}
, prompt

p 0 p_{0}
, answer length

L L
, sampling steps

N N

1: Set

r 1 r_{1}
is a fully masked sequence of length

L L
.

2:for

t←1 t\leftarrow 1
down to

1 N\frac{1}{N}
step

1 N\frac{1}{N}
do

3:

s=t−1 N s=t-\frac{1}{N}

4:for

i←1 i\leftarrow 1
to

L L
do

5:if

r t i≠M r_{t}^{i}\neq\textrm{M}
then

6:

r 0 i=r t i r_{0}^{i}=r_{t}^{i}
,

c i=1 c^{i}=1

7:else

8:

r 0 i=arg⁡max r 0 i⁡p θ​(r 0 i|p 0,r t)r_{0}^{i}=\arg\max_{r_{0}^{i}}p_{\theta}(r_{0}^{i}|p_{0},r_{t})

9:

c i=p θ​(r 0 i|p 0,r t)r 0 i c^{i}=p_{\theta}(r_{0}^{i}|p_{0},r_{t})_{r_{0}^{i}}

10:end if

11:end for

12:

n u​n=⌊L​(1−s)⌋n_{un}=\lfloor L(1-s)\rfloor
# the number of unmasked tokens is n u​n n_{un} in timestep s s

13:for

i←1 i\leftarrow 1
to

L L
do

14:if

c i∈Lowest−n u​n​({c i}1 L)c^{i}\in\text{Lowest}-n_{un}\left(\{c^{i}\}_{1}^{L}\right)
then

15:

r 0 i=M r_{0}^{i}=\textrm{M}
# the n u​n n_{un} positions with the least confidence are selected for remasking.

16:end if

17:end for

18:

r s=r 0 r_{s}=r_{0}

19:end for

20:Return

r 0 r_{0}

### A.2 Inference

The cross-entropy loss in Eq.([12](https://arxiv.org/html/2502.09992v3#A1.E12 "Equation 12 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) has several equivalent forms[[20](https://arxiv.org/html/2502.09992v3#bib.bib20)]. The first one is given by

−𝔼 l∼{1,2,…,L},x 0,x l​[L l​∑i=1 L 1​[x l i=M]​log⁡p θ​(x 0 i|x l)],\displaystyle-\mathbb{E}_{l\sim\{1,2,\dots,L\},x_{0},x_{l}}\left[\frac{L}{l}\sum_{i=1}^{L}\textbf{1}[x_{l}^{i}=\textrm{M}]\log p_{\theta}(x_{0}^{i}|x_{l})\right],(14)

where l l is uniformly sampled from {1,2,…,L}\{1,2,\dots,L\}, and x l x_{l} is obtained by uniformly sampling l l tokens from x 0 x_{0} without replacement for masking. Despite masking exactly l l tokens is different from masking each token independently with probability t t, these two masking methods lead to equivalent results in expectation [[20](https://arxiv.org/html/2502.09992v3#bib.bib20)].

While Eq.([12](https://arxiv.org/html/2502.09992v3#A1.E12 "Equation 12 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) and Eq.([14](https://arxiv.org/html/2502.09992v3#A1.E14 "Equation 14 ‣ A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) share the same expectation, their variances differ. Intuitively, in Eq.([12](https://arxiv.org/html/2502.09992v3#A1.E12 "Equation 12 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")), we expect x t x_{t} to have a fraction of t t tokens masked. However, the randomness of the forward process (i.e., Eq.([7](https://arxiv.org/html/2502.09992v3#A1.E7 "Equation 7 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"))) often causes deviations, especially when x t x_{t} contains few tokens. In contrast, in Eq.([14](https://arxiv.org/html/2502.09992v3#A1.E14 "Equation 14 ‣ A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")), the fraction of masked tokens in x l x_{l} is deterministically l L\frac{l}{L}. While a theoretical analysis depends on the data distribution, empirical results show that Eq.([12](https://arxiv.org/html/2502.09992v3#A1.E12 "Equation 12 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) requires over 1000 Monte Carlo estimates for stable results, whereas Eq.([14](https://arxiv.org/html/2502.09992v3#A1.E14 "Equation 14 ‣ A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) achieves stability with only 128 estimates. In addition, we can simply modify Eq.([14](https://arxiv.org/html/2502.09992v3#A1.E14 "Equation 14 ‣ A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) to its conditional version (i.e., Eq.([6](https://arxiv.org/html/2502.09992v3#S2.E6 "Equation 6 ‣ 2.4 Inference ‣ 2 Approach ‣ Large Language Diffusion Models"))) based on Eq.([5](https://arxiv.org/html/2502.09992v3#S2.E5 "Equation 5 ‣ 2.3 Supervised Fine-Tuning ‣ 2 Approach ‣ Large Language Diffusion Models")).

Any-order autoregressive models (AO-ARM)[[59](https://arxiv.org/html/2502.09992v3#bib.bib59), [101](https://arxiv.org/html/2502.09992v3#bib.bib101), [102](https://arxiv.org/html/2502.09992v3#bib.bib102)] characterize the joint distribution autoregressively for all possible orders π\pi of the L L variables. To learn such a distribution, an AO-ARM utilizes a weight-sharing neural network to model all univariate conditionals and employs mask tokens to represent absent variables. During training, the expected negative log-likelihood over the uniform distribution of all orders U π U_{\pi} is minimized:

−𝔼 x 0,π∼U π​[∑i=1 L log⁡p θ​(x 0 π​(i)|x 0 π(<i);π)].\displaystyle-\mathbb{E}_{x_{0},\pi\sim U_{\pi}}\left[\sum_{i=1}^{L}\log p_{\theta}(x_{0}^{\pi(i)}|x_{0}^{\pi(<i)};\pi)\right].(15)

Intuitively, x 0 π(<i)x_{0}^{\pi(<i)} can be understood as a masked token x t x_{t} with index in π(≥i){\pi(\geq i)} being masked. It can be further proved that Eq.([15](https://arxiv.org/html/2502.09992v3#A1.E15 "Equation 15 ‣ A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")) is equivalent to Eq.([12](https://arxiv.org/html/2502.09992v3#A1.E12 "Equation 12 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models")). This connection explains the bidirectional reasoning capabilities of LLaDA, even though it was never used explicitly in the inference procedure.

In addition, Nie et al. [[27](https://arxiv.org/html/2502.09992v3#bib.bib27)] introduces unsupervised classifier-free guidance (CFG), a plug-and-play technique that balances alignment with prompts and text diversity. Specifically, unsupervised CFG employs the following modified mask predictor for inference:

p~θ​(r 0|p 0,r t)∝p θ​(r 0|p 0,r t)1+w p θ​(r 0|m,r t)w,\displaystyle\tilde{p}_{\theta}(r_{0}|p_{0},r_{t})\propto\frac{p_{\theta}(r_{0}|p_{0},r_{t})^{1+w}}{p_{\theta}(r_{0}|m,r_{t})^{w}},(16)

where m m is a mask sequence of the same length as p 0 p_{0} and w w is a tunable hyperparameter that controls the strength of p 0 p_{0}. To ensure a fair comparison with ARMs, we do not apply CFG to LLaDA in the main text. However, as demonstrated in Appendix[B.3](https://arxiv.org/html/2502.09992v3#A2.SS3 "B.3 Ablation on Classifier-free Guidance ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), LLaDA is fully compatible with CFG and consistently exhibits improved performance when it is applied.

### A.3 Algorithms

In this section, we present the training and inference algorithms. Specifically, we introduce the pre-training and supervised fine-tuning algorithms in Algorithm[1](https://arxiv.org/html/2502.09992v3#alg1 "Algorithm 1 ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models") and Algorithm[2](https://arxiv.org/html/2502.09992v3#alg2 "Algorithm 2 ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"), respectively. In addition, the likelihood evaluation algorithm is provided in Algorithm[3](https://arxiv.org/html/2502.09992v3#alg3 "Algorithm 3 ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"). Finally, we present the reverse generation process in Algorithm[4](https://arxiv.org/html/2502.09992v3#alg4 "Algorithm 4 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models") and Algorithm[5](https://arxiv.org/html/2502.09992v3#alg5 "Algorithm 5 ‣ A.1 Training ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models"), which correspond to the random remasking and the low-confidence[[23](https://arxiv.org/html/2502.09992v3#bib.bib23)] remasking strategy, respectively.

Appendix B Experiments
----------------------

### B.1 Data Collection and Preprocessing

In this section, we first introduce the data collection and filtering processes for both pre-training and SFT. We then describe how LLaDA leverages these datasets during training.

Our pre-training corpus is constructed from diverse publicly available sources, including web data, books, academic papers, social media, encyclopedias, mathematics, and code, with approximately 11% Chinese, 61% English, and 28% code. The data cleaning process involves PDF text extraction, deduplication, and harmful content filtering. To further ensure quality, we fine-tune a BERT[[22](https://arxiv.org/html/2502.09992v3#bib.bib22)] model for automated data quality annotation, enabling the selection of higher-quality samples. Our SFT dataset consists of 1 million human-annotated samples and 3.5 million synthetic samples, generated using methods similar to those proposed in Xu et al. [[103](https://arxiv.org/html/2502.09992v3#bib.bib103)], Wei et al. [[104](https://arxiv.org/html/2502.09992v3#bib.bib104)].

We concatenate the collected documents in the pre-training corpus and segment the text into fixed-length sequences according to the predefined sequence length.

For SFT, a dynamic sequence length strategy is employed, where |EOS||\text{EOS}| tokens are appended to the end of shorter pairs to ensure uniform sequence lengths across all samples within each mini-batch. Notably, the padding |EOS||\text{EOS}| tokens are treated as part of the response, i.e., masked and included in the training objective. The |EOS||\text{EOS}| tokens are removed from the generated outputs during sampling. This strategy ensures that the model learns to control the length of its responses by generating |EOS||\text{EOS}|, enabling the response length to align effectively with the given prompt.

In addition, for n n-turn dialogues (p 0 0,r 0 0,p 0 1,r 0 1,…,p 0 n−1,r 0 n−1)(p_{0}^{0},r_{0}^{0},p_{0}^{1},r_{0}^{1},\dots,p_{0}^{n-1},r_{0}^{n-1}), we treat it as n n single-turn dialogue pairs, i.e., (p 0 0,r 0 0),(p 0 0​r 0 0​p 0 1,r 0 1),…,(p 0 0​r 0 0​p 0 1​r 0 1​…​p 0 n−1,r 0 n−1)(p_{0}^{0},r_{0}^{0}),(p_{0}^{0}r_{0}^{0}p_{0}^{1},r_{0}^{1}),\dots,(p_{0}^{0}r_{0}^{0}p_{0}^{1}r_{0}^{1}\dots p_{0}^{n-1},r_{0}^{n-1}) and randomly sample one. This data partitioning strategy not only equips LLaDA with multi-turn dialogue capabilities but also aligns with the above |EOS||\text{EOS}| padding strategy.

### B.2 Details about Model Training

This section provides the training details of LLaDA and the corresponding ARM baselines.

Firstly, for efficiency, we trained an ARM and an MDM, both with 1.5B parameters and identical architectures. Additionally, we scaled the MDM to 8B parameters. Due to computational resource constraints, we did not train an 8B autoregressive model with the same architecture. Instead, we utilized our previously trained 7B autoregressive model for comparison. These four models are utilized in the scalability analysis in Sec.[3.1](https://arxiv.org/html/2502.09992v3#S3.SS1 "3.1 Scalability of LLaDA on Language Tasks ‣ 3 Experiments ‣ Large Language Diffusion Models").

We adopted a Transformer architecture similar to LLaMA[[6](https://arxiv.org/html/2502.09992v3#bib.bib6), [21](https://arxiv.org/html/2502.09992v3#bib.bib21)] for the ARMs and MDMs we trained. Specifically, we employ RMSNorm[[105](https://arxiv.org/html/2502.09992v3#bib.bib105)] to stabilize training, use SwiGLU[[106](https://arxiv.org/html/2502.09992v3#bib.bib106)] as the activation function to enhance non-linearity, and integrate RoPE[[107](https://arxiv.org/html/2502.09992v3#bib.bib107)] for more expressive positional encoding. Tab.[5](https://arxiv.org/html/2502.09992v3#A2.T5 "Table 5 ‣ B.2 Details about Model Training ‣ Appendix B Experiments ‣ Large Language Diffusion Models") provides an overview of the model architectures.

For the 1B and 7B ARM baselines, as well as the 1B and 8B LLaDA models, we utilized the AdamW optimizer[[29](https://arxiv.org/html/2502.09992v3#bib.bib29)] with a weight decay of 0.1 and adopted the Warmup-Stable-Decay[[28](https://arxiv.org/html/2502.09992v3#bib.bib28)] learning rate scheduler. The learning rate was linearly increased from 0 to the maximum value over the first 2000 iterations and then held constant. For LLaDA 8B, to ensure stable training, the learning rate was reduced once during pre-training, as detailed in Sec.[2.2](https://arxiv.org/html/2502.09992v3#S2.SS2 "2.2 Pre-training ‣ 2 Approach ‣ Large Language Diffusion Models"). For the 1B ARM baseline and both the 1B and 8B LLaDA models, the maximum learning rate is set to 4×10−4 4\times 10^{-4} with a batch size of 1280, without any hyperparameter tuning. For the 7B ARM baseline, the maximum learning rate is set to 4.2×10−4 4.2\times 10^{-4} with a batch size of 4224, both selected via grid search.

Additionally, we employ the widely used 6​N​D 6ND formulation[[108](https://arxiv.org/html/2502.09992v3#bib.bib108), [109](https://arxiv.org/html/2502.09992v3#bib.bib109)] to calculate the training FLOPs in Fig.[3](https://arxiv.org/html/2502.09992v3#S3.F3 "Figure 3 ‣ 3 Experiments ‣ Large Language Diffusion Models"), where N N represents the number of non-embedding parameters, and D D denotes the total number of training tokens. The detailed results corresponding to Fig.[3](https://arxiv.org/html/2502.09992v3#S3.F3 "Figure 3 ‣ 3 Experiments ‣ Large Language Diffusion Models") are provided in Tab.[18](https://arxiv.org/html/2502.09992v3#A3.T18 "Table 18 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models") and Tab.[19](https://arxiv.org/html/2502.09992v3#A3.T19 "Table 19 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models").

Table 5: Model Architecture. We report the architectural configurations for our 1B and 7B ARM baselines, the 1B and 8B LLaDA models, and the 8B LLaMA3 model. 

### B.3 Ablation on Classifier-free Guidance

This section presents an ablation study on classifier-free guidance (CFG). Theoretical details about CFG can be found in the Appendix[A.2](https://arxiv.org/html/2502.09992v3#A1.SS2 "A.2 Inference ‣ Appendix A Formulation of Masked Diffusion Models ‣ Large Language Diffusion Models").

For simplicity, we select six representative benchmarks, including ARC-C, HellaSwag, TruthfulQA, WinoGrande, PIQA, and GPQA, and conduct experiments using LLaDA 8B Base. We search the CFG scale in {0.5,1,1.5,2}\{0.5,1,1.5,2\} for each task and report the best result. As shown in Tab.[6](https://arxiv.org/html/2502.09992v3#A2.T6 "Table 6 ‣ B.3 Ablation on Classifier-free Guidance ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), CFG consistently improves the performance of LLaDA. We emphasize that, to ensure a fair comparison with ARMs, CFG is not used in the main results reported in the paper.

Table 6: Ablation on CFG. CFG consistently improves the performance of LLaDA.

### B.4 Details and Ablation on Sampling Strategies

In this section, we first introduce the different sampling strategies supported by LLaDA. We then present ablation studies to evaluate the performance of these sampling strategies.

![Image 10: Refer to caption](https://arxiv.org/html/2502.09992v3/imgs/sample/ar_sample.jpg)

(a)Autoregressive.

![Image 11: Refer to caption](https://arxiv.org/html/2502.09992v3/imgs/sample/block_diffusion.jpg)

(b)Block Diffusion.

![Image 12: Refer to caption](https://arxiv.org/html/2502.09992v3/imgs/sample/block_diffusion_llada.jpg)

(c)Block Diffusion LLaDA.

Figure 4: Flexible Sampling Strategies Supported by LLaDA. Colored squares depict non‑masked tokens, while squares marked with ×\times denote masked tokens. In this illustration, the block length for both block diffusion and block diffusion LLaDA sampling is 4.

Table 7: Ablation on Sampling Strategies for LLaDA 8B Base.L′L^{\prime} is the block length. Pure diffusion sampling achieves the best overall performance.

Table 8: Ablation on Sampling Strategies for LLaDA 8B Instruct. The block length is set to 32 for efficiency. Pure diffusion sampling achieves the best overall performance.

Flexible Sampling Strategies. In Sec.[2.4](https://arxiv.org/html/2502.09992v3#S2.SS4 "2.4 Inference ‣ 2 Approach ‣ Large Language Diffusion Models"), Fig.[2](https://arxiv.org/html/2502.09992v3#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Large Language Diffusion Models")(c) illustrates the reverse generation process of LLaDA. As shown in Fig.[4](https://arxiv.org/html/2502.09992v3#A2.F4 "Figure 4 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), in addition to the reverse generation process, LLaDA also supports autoregressive and block diffusion[[31](https://arxiv.org/html/2502.09992v3#bib.bib31)] sampling directly after the pre-training or SFT stages, without requiring any further modifications or retraining. Block diffusion sampling applies the origin reverse process within each block and the autoregressive sampling across blocks. In the original block diffusion process, the sequence length varies dynamically. As shown in Fig.[4](https://arxiv.org/html/2502.09992v3#A2.F4 "Figure 4 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models")(c), LLaDA can also adopt a fixed-length block diffusion strategy, which we refer to as block diffusion LLaDA, also known as semi-autoregressive remasking.

Experimental Setup. We evaluate different sampling strategies using both LLaDA 8B Base and LLaDA 8B Instruct for comprehensive analysis. For LLaDA 8B Base, we use the five benchmarks in Tab.[1](https://arxiv.org/html/2502.09992v3#S3.T1 "Table 1 ‣ 3.2 Benchmark Results ‣ 3 Experiments ‣ Large Language Diffusion Models") that are evaluated based on sampling rather than likelihood estimation. For LLaDA 8B Instruct, we use the seven metrics in Tab.[2](https://arxiv.org/html/2502.09992v3#S3.T2 "Table 2 ‣ 3.2 Benchmark Results ‣ 3 Experiments ‣ Large Language Diffusion Models"), excluding MMLU and HellaSwag, since these two tasks only require the model to generate a single token (i.e., A, B, C, or D). In all settings, one token is generated per sampling step. For autoregressive and block diffusion sampling, generation terminates when the |EOS||\text{EOS}| token is produced. For block diffusion LLaDA (i.e., semi-autoregressive remasking) and pure diffusion sampling, the generation length is fixed at 1024 for LLaDA 8B Base, while for LLaDA 8B Instruct, it is tuned from {64, 256, 512} to balance efficiency and performance. Low-confidence remasking is applied to intra-block diffusion sampling in both block diffusion and block diffusion LLaDA, as well as to pure diffusion sampling. We also test different block lengths for LLaDA 8B Base. For LLaDA 8B Instruct, we only evaluate block length 32 for efficiency, as it yields the best results on LLaDA 8B Base.

Additionally, for LLaDA 8B Instruct, due to heavy padding of |EOS||\text{EOS}| tokens in the SFT data (as detailed in Sec.[B.1](https://arxiv.org/html/2502.09992v3#A2.SS1 "B.1 Data Collection and Preprocessing ‣ Appendix B Experiments ‣ Large Language Diffusion Models")), we observe that under pure diffusion sampling, the proportion of |EOS||\text{EOS}| tokens in model outputs becomes very high. This leads to extremely short generations and degrades model performance. To mitigate this issue, for HumanEval, MBPP, GSM8K, Math, and GPQA, we set the confidence score of the |EOS||\text{EOS}| token to zero during pure diffusion sampling. This adjustment helps maintain an appropriate ratio of |EOS||\text{EOS}| tokens during generation.

Finally, we conduct ablation studies to analyze the effects of random and low-confidence remasking strategies using the pure diffusion sampling. For efficiency, we use LLaDA 8B Base with generation length and sampling steps set to 256 in this analysis.

Results. As shown in Tab.[7](https://arxiv.org/html/2502.09992v3#A2.T7 "Table 7 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), for block diffusion sampling, overall performance improves as the block length increases. Moreover, both Tab.[7](https://arxiv.org/html/2502.09992v3#A2.T7 "Table 7 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models") and Tab.[8](https://arxiv.org/html/2502.09992v3#A2.T8 "Table 8 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models") show that block diffusion sampling consistently outperforms autoregressive sampling, and block diffusion LLaDA sampling further improves upon standard block diffusion sampling. Finally, pure diffusion sampling achieves the best overall performance.

In addition, Tab.[9](https://arxiv.org/html/2502.09992v3#A2.T9 "Table 9 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models") shows that the low-confidence remasking strategy consistently outperforms the random remasking strategy. We hypothesize that low-confidence remasking functions similarly to the annealed sampling method used by default in ARMs, improving accuracy by reducing the diversity of generated sentences.

Table 9: Analysis on Random and Low-confidence Remasking Strategies. The low-confidence remasking consistently outperforms the random remasking.

We discover that autoregressive sampling leads to very poor performance for LLaDA 8B Instruct. This is because each SFT data is a complete sentence, so given a sequence length, LLaDA 8B Instruct tends to generate a full sentence within that length. In contrast, LLaDA 8B Base does not suffer from this issue, as the pre-training data consists of truncated documents (as detailed in Appendix[B.1](https://arxiv.org/html/2502.09992v3#A2.SS1 "B.1 Data Collection and Preprocessing ‣ Appendix B Experiments ‣ Large Language Diffusion Models")) and the model is trained with random sequence lengths (as detailed in Sec.[2.2](https://arxiv.org/html/2502.09992v3#S2.SS2 "2.2 Pre-training ‣ 2 Approach ‣ Large Language Diffusion Models")). As a result, when given a short sequence length, LLaDA 8B Base tends to generate only part of a sentence, which can then be used as a prefix to continue generation.

Setting the block length to 8 in Tab.[8](https://arxiv.org/html/2502.09992v3#A2.T8 "Table 8 ‣ B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models") further improves the GSM8K score from 77.5 to 78.6.

### B.5 Ablation on Generated Length

Table 10: Ablation on Generation Length. The results are not sensitive to the length hyperparameter.

In this section, we conduct ablation studies on the generated length.

To ensure fairness, for each setting, we set the number of sampling steps equal to the generated length, ensuring that in each sampling step, just one tokens are transferred from the mask to the text. We conduct experiments using LLaDA 8B Base.

As reported in Tab.[10](https://arxiv.org/html/2502.09992v3#A2.T10 "Table 10 ‣ B.5 Ablation on Generated Length ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), the results are not sensitive to the length hyperparameter.

### B.6 Standard Benchmarks and Evaluation Details

![Image 13: Refer to caption](https://arxiv.org/html/2502.09992v3/x10.png)

![Image 14: Refer to caption](https://arxiv.org/html/2502.09992v3/x11.png)

![Image 15: Refer to caption](https://arxiv.org/html/2502.09992v3/x12.png)

![Image 16: Refer to caption](https://arxiv.org/html/2502.09992v3/x13.png)

Figure 5: Analysis of Sampling Efficiency. The generation length for LLaDA is set to 256, with sampling steps set to 32, 64, 128, and 256 across the figures. This corresponds to decoding 8, 4, 2, and 1 token(s) per forward pass, respectively. LLaDA enables a flexible trade-off between generation quality and sampling speed. 

In this section, we introduce the benchmarks we used and present the details of our evaluation process.

Following standard LLM[[25](https://arxiv.org/html/2502.09992v3#bib.bib25), [26](https://arxiv.org/html/2502.09992v3#bib.bib26)] evaluation practices, we assess LLaDA across four dimensions:

General ability: MMLU[[110](https://arxiv.org/html/2502.09992v3#bib.bib110)], BBH[[111](https://arxiv.org/html/2502.09992v3#bib.bib111)], ARC-C[[112](https://arxiv.org/html/2502.09992v3#bib.bib112)], Hellaswag[[113](https://arxiv.org/html/2502.09992v3#bib.bib113)], TruthfulQA[[114](https://arxiv.org/html/2502.09992v3#bib.bib114)], WinoGrande[[115](https://arxiv.org/html/2502.09992v3#bib.bib115)] and PIQA[[116](https://arxiv.org/html/2502.09992v3#bib.bib116)].

Math and science ability: GSM8K[[117](https://arxiv.org/html/2502.09992v3#bib.bib117)], Math[[118](https://arxiv.org/html/2502.09992v3#bib.bib118)] and GPQA[[119](https://arxiv.org/html/2502.09992v3#bib.bib119)].

Code generation: HumanEval[[120](https://arxiv.org/html/2502.09992v3#bib.bib120)], HumanEval-FIM[[121](https://arxiv.org/html/2502.09992v3#bib.bib121)] and MBPP[[122](https://arxiv.org/html/2502.09992v3#bib.bib122)].

Chinese understanding: CMMLU[[123](https://arxiv.org/html/2502.09992v3#bib.bib123)] and C-Eval[[124](https://arxiv.org/html/2502.09992v3#bib.bib124)].

For all the aforementioned benchmarks, we follow the widely adopted evaluation process[[125](https://arxiv.org/html/2502.09992v3#bib.bib125)] used in LLM assessments, primarily employing conditional likelihood estimation and conditional generation. Specifically, for certain benchmarks, a prompt and multiple candidate answers are provided, and the model is required to compute each candidate’s conditional likelihood. The candidate with the highest likelihood is then selected as the model’s final answer, and accuracy is used as the evaluation metric. For the remaining benchmarks, the model generates responses based on the given prompt, and performance is evaluated using metrics such as exact match and other relevant criteria.

For the base model, we use conditional likelihood estimation for MMLU, CMMLU, C-Eval, ARC-C, Hellaswag, TruthfulQA, WinoGrande, PIQA, and GPQA, while the remaining benchmarks are evaluated using conditional generation. For the instruct model, we evaluate all benchmarks using conditional generation.

For the base model, we use the widely adopted open-source evaluation framework lm-evaluation-harness[[125](https://arxiv.org/html/2502.09992v3#bib.bib125)], except for the HumanEval-FIM metric, which is not supported by the framework. For HumanEval-FIM on the base model and all evaluation metrics on the instruct model, we use an internal evaluation library. We choose the internal library as lm-evaluation-harness shows greater deviation from the LLaMA3 results reported by Yang et al. [[25](https://arxiv.org/html/2502.09992v3#bib.bib25)], relative to our internal evaluation.

For benchmarks evaluated via conditional likelihood estimation, we use Monte Carlo estimation to approximate Eq.([6](https://arxiv.org/html/2502.09992v3#S2.E6 "Equation 6 ‣ 2.4 Inference ‣ 2 Approach ‣ Large Language Diffusion Models")) for LLaDA. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128 Monte Carlo samples are adequate to produce stable results.

For benchmarks evaluated using conditional generation, we apply pure diffusion sampling with a low-confidence remasking strategy to both LLaDA Base and LLaDA Instruct. For LLaDA Base, we set both the generation length and the number of sampling steps to 1024. For LLaDA Instruct, the number of sampling steps is set equal to the answer length, which is configured as follows: 3 for MMLU and HellaSwag, 64 for GPQA, 256 for MBPP and MMLU-Pro, and 512 for HumanEval, GSM8K, Math, and ARC-C. As detailed in Appendix[B.4](https://arxiv.org/html/2502.09992v3#A2.SS4 "B.4 Details and Ablation on Sampling Strategies ‣ Appendix B Experiments ‣ Large Language Diffusion Models"), for HumanEval, MBPP, GSM8K, Math, and GPQA, we set the confidence of the |EOS||\text{EOS}| token to zero during sampling for LLaDA Instruct.

### B.7 Analysis of Sampling Efficiency

In this section, we first analyze the sampling efficiency of LLaDA, including both sampling speed and memory consumption. We then discuss potential optimizations to further improve its efficiency.

We use four representative open-ended generation benchmarks for sampling speed analysis: GSM8K, Math, HumanEval, and MBPP. We use the widely adopted throughput metric to measure generation speed, defined as the number of tokens generated per second. We compare LLaDA 8B Base and LLaMA3 8B Base, both using bfloat16 precision. All experiments in this section were conducted on a single A100-80GB GPU with a batch size of 1. For LLaDA, the output length is fixed to 256 tokens across all four benchmarks.

Fig.[5](https://arxiv.org/html/2502.09992v3#A2.F5 "Figure 5 ‣ B.6 Standard Benchmarks and Evaluation Details ‣ Appendix B Experiments ‣ Large Language Diffusion Models") shows that LLaDA enables a flexible trade-off between generation quality and speed by adjusting the number of sampling steps. Specifically, on the GSM8K and Math datasets, LLaDA 8B Base achieves comparable performance to LLaMA3 8B Base while delivering 1.5 and 1.8 times higher throughput, even though LLaMA3 uses KV Cache and LLaDA operates without any inference optimization techniques. For the HumanEval benchmark, LLaDA 8B Base performs comparably to LLaMA3 8B Base when the throughput is matched. On the MBPP benchmark, LLaDA 8B Base lags behind LLaMA3 8B Base.

For LLaMA3, the acceleration benefit provided by KV caching is notably weaker on the HumanEval dataset, which can be attributed to its relatively short prompt lengths. Specifically, the average prompt lengths for GSM8K, Math, MBPP, and HumanEval are 894, 680, 628, and 132 tokens, respectively.

Table 11: Analysis of Memory Consumption. Memory is measured in GB. Without any inference optimization techniques (e.g., KV Cache), LLaDA has memory usage comparable to LLaMA3, and slightly higher than LLaMA3 when the latter uses KV Cache.

Tab.[11](https://arxiv.org/html/2502.09992v3#A2.T11 "Table 11 ‣ B.7 Analysis of Sampling Efficiency ‣ Appendix B Experiments ‣ Large Language Diffusion Models") compares of memory consumption between LLaDA 8B Base and LLaMA3 8B Base. To avoid variations in generation length caused by differences in training data, we fix both the input and output token lengths during the memory analysis. For LLaDA, memory usage remains constant regardless of the number of sampling steps. Its memory consumption is comparable to LLaMA3 8B Base without KV cache, but slightly higher than with KV cache.

We emphasize that the goal of this study is not to propose a model that is faster than ARMs. Instead, we aim to show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities such as scalability, in-context learning, and instruction-following are inherently depend on ARMs. A substantial body of research[[31](https://arxiv.org/html/2502.09992v3#bib.bib31), [79](https://arxiv.org/html/2502.09992v3#bib.bib79), [80](https://arxiv.org/html/2502.09992v3#bib.bib80), [81](https://arxiv.org/html/2502.09992v3#bib.bib81), [82](https://arxiv.org/html/2502.09992v3#bib.bib82), [83](https://arxiv.org/html/2502.09992v3#bib.bib83), [84](https://arxiv.org/html/2502.09992v3#bib.bib84), [85](https://arxiv.org/html/2502.09992v3#bib.bib85), [86](https://arxiv.org/html/2502.09992v3#bib.bib86), [87](https://arxiv.org/html/2502.09992v3#bib.bib87)] has focused on improving the generation efficiency of MDMs through algorithmic or architectural innovations. We leave similar efficiency-oriented exploration for LLaDA to future work.

### B.8 Evaluation on iGSM Dataset

Table 12: Comparison on iGSM Dataset.

To further assess the mathematical capabilities of LLaDA, we test its performance on iGSM [[34](https://arxiv.org/html/2502.09992v3#bib.bib34)], an infinite, synthetic GSM8K-like dataset. iGSM is generated via specific rules, with parameters that control the difficulty of problems (i.e., the number of solution steps). For evaluation consistency, we append "#### $answer" to the final solution, adhering to the GSM8K format. Below is an example with solution steps set to 4:

(Question) The number of each North Star Elementary’s Cultural Studies Classroom equals 1. The number of each Westridge Elementary’s Dance Studio equals 3 times as much as the sum of each North Star Elementary’s Classroom and each North Star Elementary’s Cultural Studies Classroom. How many Dance Studio does Westridge Elementary have?

(Solution) Define North Star Elementary’s Cultural Studies Classroom as x; so x = 1.

Define North Star Elementary’s Classroom as m; so m = x = 1.

Define Westridge Elementary’s Dance Studio as n; w = m + x = 1 + 1 = 2;

so n = 3 * w = 3 * 2 = 1 #### 1

Since there are slight differences between GSM8K and iGSM (e.g., the use of a mod 5 algorithmic system), we follow [[34](https://arxiv.org/html/2502.09992v3#bib.bib34)] and provide a system prompt along with four-shot question-answer pairs for each problem.

(Prompt) You’re an expert at solving elementary math problems involving addition, subtraction, and multiplication. You solve all the problems in a uniform format. All calculations are done modulo 5. For example, 4 + 4 equals 3, 2 + 4 equals 1, 3 + 3 + 3 equals 4, 3 * 3 equals 4, and 2 * 2 equals 4. When providing your solution, please end with ’#### x.’ where x is your final answer, an integer between 0 and 4. You must solve all the problems using the same solution format. Our scenarios involve up to four categories of objects: schools, classrooms, backpacks and stationeries. Each school may contain classrooms, each classroom may contain backpacks, and each backpack may contain stationeries. We can specify quantities, such as ẗhe number of dance studios at each Lakeshore High.Ässume that every entity with the same name has an identical configuration; for example, each Lakeshore High contains the same number of dance studios. Another guiding principle is that what is not mentioned does not exist: when we refer to classrooms at Lakeshore High, we are only discussing the classrooms explicitly mentioned in our scenario. Furthermore, if Lakeshore High is not even mentioned, any classroom within it is automatically considered to be non-existent (i.e. 0).

For solution steps ranging from 4 to 6, we generate 100 questions for each case and report the corresponding accuracy in [Table˜12](https://arxiv.org/html/2502.09992v3#A2.T12 "In B.8 Evaluation on iGSM Dataset ‣ Appendix B Experiments ‣ Large Language Diffusion Models"). As shown in the table, LLaDA 8B Base demonstrates significant and consistent advantages over LLaMA3 8B Base on unseen mathematical problems, aligning with the results in Table[1](https://arxiv.org/html/2502.09992v3#S3.T1 "Table 1 ‣ 3.2 Benchmark Results ‣ 3 Experiments ‣ Large Language Diffusion Models").

### B.9 Poem Completion Tasks

In this section, we present examples from our poem completion dataset as follows.

Example 1: 

Prompt: 窈窕淑女的下一句是什么？直接输出句子即可。 

Answer: 君子好逑。

Example 2: 

Prompt: 不拘一格降人才的上一句是什么？直接输出句子即可。 

Answer: 我劝天公重抖擞。

### B.10 More Case Studies

In this section, we present additional case studies of LLaDA 8B Instruct. First, Tab.[13](https://arxiv.org/html/2502.09992v3#A3.T13 "Table 13 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models") shows the sampling process of the block diffusion LLaDA sampling, while Tab.[14](https://arxiv.org/html/2502.09992v3#A3.T14.fig1 "Table 14 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models") depicts the sampling process for multi-turn dialogues with random remasking. Additionally, Tab.[15](https://arxiv.org/html/2502.09992v3#A3.T15.fig1 "Table 15 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models") and Tab.[16](https://arxiv.org/html/2502.09992v3#A3.T16.fig1 "Table 16 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models") provide further examples of single-turn and multi-turn dialogues. Finally, Tab.[17](https://arxiv.org/html/2502.09992v3#A3.T17.fig1 "Table 17 ‣ Appendix C Impact Statement ‣ Large Language Diffusion Models") presents examples of poem reversal completions where the LLaDA 8B Instruct model succeeds, in contrast to the failure of GPT-4o.

Appendix C Impact Statement
---------------------------

Our work shows the promise of diffusion models for language modeling at scale and challenges the common assumption that core LLM capabilities such as scalability, in-context learning, and instruction-following are inherently dependent on ARMs. Our findings open new avenues for exploring alternative probabilistic paradigms in natural language processing, with potential applications in conversational AI, code generation, and complex reasoning tasks.

However, diffusion models, like traditional LLMs, raise similar societal concerns. These include the environmental impact of large-scale training, the potential misuse for generating harmful content, and the amplification of biases present in training data. Addressing these challenges is critical to ensuring the responsible development and deployment of diffusion language models.

Table 13: Visualization of the Block Diffusion LLaDA Sampling Process. In the response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.

Table 14: Visualization of the Multi-turn Dialogue. We employ random remasking strategy. In the response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.

Table 15: Single-turn Dialogue Cases of LLaDA 8B Instruct.

Table 16: Multi-turn Dialogue Cases of LLaDA 8B Instruct.

Table 17: Poem Reversal Completion Cases where LLaDA 8B Instruct Succeeds but GPT-4o Fails.

Table 18: Detailed results of LLaDA in Fig.[3](https://arxiv.org/html/2502.09992v3#S3.F3 "Figure 3 ‣ 3 Experiments ‣ Large Language Diffusion Models"). "-" indicates missing values, which do not affect the observations regarding the scalability of LLaDA. These missing values are due to hardware failures. 

Table 19: Detailed results of the autoregressive baelines in Fig.[3](https://arxiv.org/html/2502.09992v3#S3.F3 "Figure 3 ‣ 3 Experiments ‣ Large Language Diffusion Models").
