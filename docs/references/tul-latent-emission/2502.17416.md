Title: Reasoning with Latent Thoughts: On the Power of Looped Transformers

URL Source: https://arxiv.org/html/2502.17416

Published Time: Tue, 25 Feb 2025 03:05:29 GMT

Markdown Content:
# Reasoning with Latent Thoughts: On the Power of Looped Transformers

1.   [1 Introduction](https://arxiv.org/html/2502.17416v1#S1 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
2.   [2 Looped models on simple reasoning tasks](https://arxiv.org/html/2502.17416v1#S2 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    1.   [2.1 Experiments with simple reasoning problems](https://arxiv.org/html/2502.17416v1#S2.SS1 "In 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        1.   [$p$-hop induction.](https://arxiv.org/html/2502.17416v1#S2.SS1.SSS0.Px1 "In 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        2.   [i-GSM (Synthetic Grade School Math Problems).](https://arxiv.org/html/2502.17416v1#S2.SS1.SSS0.Px2 "In 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
            1.   [Question.](https://arxiv.org/html/2502.17416v1#S2.SS1.SSS0.Px3 "In Table 2 ‣ i-GSM (Synthetic Grade School Math Problems). ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
            2.   [Answer with CoT.](https://arxiv.org/html/2502.17416v1#S2.SS1.SSS0.Px4 "In Table 2 ‣ i-GSM (Synthetic Grade School Math Problems). ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

3.   [3 Language modeling with looped models](https://arxiv.org/html/2502.17416v1#S3 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    1.   [3.1 Experiments with 1B language modeling](https://arxiv.org/html/2502.17416v1#S3.SS1 "In 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    2.   [3.2 Inductive bias towards reasoning](https://arxiv.org/html/2502.17416v1#S3.SS2 "In 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    3.   [3.3 Middle looping variant and relationship with Gradual stacking](https://arxiv.org/html/2502.17416v1#S3.SS3 "In 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    4.   [3.4 Scaling behavior of looping](https://arxiv.org/html/2502.17416v1#S3.SS4 "In 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        1.   [Latent thoughts and connections to CoT reasoning.](https://arxiv.org/html/2502.17416v1#S3.SS4.SSS0.Px1 "In 3.4 Scaling behavior of looping ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

4.   [4 Looping-inspired regularization](https://arxiv.org/html/2502.17416v1#S4 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
5.   [5 Theoretical analysis for looped models](https://arxiv.org/html/2502.17416v1#S5 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    1.   [5.1 Preliminaries and Notations](https://arxiv.org/html/2502.17416v1#S5.SS1 "In 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    2.   [5.2 Group composition problem](https://arxiv.org/html/2502.17416v1#S5.SS2 "In 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    3.   [5.3 Looped models can simulate non-looped models](https://arxiv.org/html/2502.17416v1#S5.SS3 "In 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    4.   [5.4 Looped models can simulate CoT reasoning](https://arxiv.org/html/2502.17416v1#S5.SS4 "In 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

6.   [6 Related work](https://arxiv.org/html/2502.17416v1#S6 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
7.   [7 Conclusions, limitations and future work](https://arxiv.org/html/2502.17416v1#S7 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
8.   [A Experiments](https://arxiv.org/html/2502.17416v1#A1 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    1.   [A.1 Simple reasoning setup details](https://arxiv.org/html/2502.17416v1#A1.SS1 "In Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        1.   [$n$-ary addition.](https://arxiv.org/html/2502.17416v1#A1.SS1.SSS0.Px1 "In A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        2.   [$p$-hop induction.](https://arxiv.org/html/2502.17416v1#A1.SS1.SSS0.Px2 "In A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        3.   [i-GSM.](https://arxiv.org/html/2502.17416v1#A1.SS1.SSS0.Px3 "In A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

    2.   [A.2 Language modeling setup](https://arxiv.org/html/2502.17416v1#A1.SS2 "In Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    3.   [A.3 Results for each task group](https://arxiv.org/html/2502.17416v1#A1.SS3 "In Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

9.   [B Theoretical results](https://arxiv.org/html/2502.17416v1#A2 "In Reasoning with Latent Thoughts: On the Power of Looped Transformers")
    1.   [B.1 Detailed notations](https://arxiv.org/html/2502.17416v1#A2.SS1 "In Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        1.   [Multi-Head Self-Attention Mechanism:](https://arxiv.org/html/2502.17416v1#A2.SS1.SSS0.Px1 "In B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        2.   [Feed-Forward Network:](https://arxiv.org/html/2502.17416v1#A2.SS1.SSS0.Px2 "In B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        3.   [Finite-precision Modeling:](https://arxiv.org/html/2502.17416v1#A2.SS1.SSS0.Px3 "In B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

    2.   [B.2 Proofs](https://arxiv.org/html/2502.17416v1#A2.SS2 "In Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        1.   [B.2.1 Looped models can simulate non-looped models](https://arxiv.org/html/2502.17416v1#A2.SS2.SSS1 "In B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")
        2.   [B.2.2 Group composition.](https://arxiv.org/html/2502.17416v1#A2.SS2.SSS2 "In B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

    3.   [B.3 Connection to chain-of-thought reasoning](https://arxiv.org/html/2502.17416v1#A2.SS3 "In Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

\NewDocumentCommand\T
ooo T\IfNoValueF#1 [#1\IfNoValueF#2 ,#2\IfNoValueF#3 ,#3 ] \NewDocumentCommand\Cot oooo CoT\IfNoValueF#1 [#1\IfNoValueF#2 ,#2\IfNoValueF#3 ,#3 \IfNoValueF#4 ,#4 ] \NewDocumentCommand\Method oooo LoOPT\IfNoValueF#1 [#1\IfNoValueF#2 ,#2\IfNoValueF#3 ,#3 \IfNoValueF#4 ,#4 ]

# Reasoning with Latent Thoughts: On the Power of Looped Transformers

Nikunj Saunshi 1, Nishanth Dikkala 1, Zhiyuan Li 1,2, Sanjiv Kumar 1, Sashank J. Reddi 1

{nsaunshi, nishanthd, lizhiyuan, sanjivk, sashank}@google.com 

1 Google Research, 2 Toyota Technological Institute at Chicago 

Submitted on Oct. 1, 2024 

###### Abstract

Large language models have shown remarkable reasoning abilities and scaling laws suggest that large parameter count, especially along the depth axis, is the primary driver. In this work, we make a stronger claim — many reasoning problems require a large depth but not necessarily many parameters. This unlocks a novel application of looped models for reasoning. Firstly, we show that for many synthetic reasoning problems like addition, $p$-hop induction, and math problems, a $k$-layer transformer looped $L$ times nearly matches the performance of a $k ⁢ L$-layer non-looped model, and is significantly better than a $k$-layer model. This is further corroborated by theoretical results showing that many such reasoning problems can be solved via iterative algorithms, and thus, can be solved effectively using looped models with nearly optimal depth. Perhaps surprisingly, these benefits also translate to practical settings of language modeling — on many downstream reasoning tasks, a language model with $k$-layers looped $L$ times can be competitive to, if not better than, a $k ⁢ L$-layer language model. In fact, our empirical analysis reveals an intriguing phenomenon: looped and non-looped models exhibit scaling behavior that depends on their effective depth, akin to the inference-time scaling of chain-of-thought (CoT) reasoning. We further elucidate the connection to CoT reasoning by proving that looped models implicitly generate _latent thoughts_ and can simulate $T$ steps of CoT with $T$ loops. Inspired by these findings, we also present an interesting dichotomy between reasoning and memorization, and design a looping-based regularization that is effective on both fronts.

## 1 Introduction

Language models have shown a lot of promise in solving problems that require strong reasoning abilities like math, coding, common sense reasoning and logical puzzles(Brown et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib5), Team et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib60)). This has sparked interest in developing techniques to improve reasoning on harder problems (Wei et al., [2022b](https://arxiv.org/html/2502.17416v1#bib.bib62)) and has inspired theoretical studies on how Transformers are able to perform reasoning (Feng et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib16), Sanford et al., [2024a](https://arxiv.org/html/2502.17416v1#bib.bib49)). Reasoning abilities are often emergent in larger language models (Wei et al., [2022a](https://arxiv.org/html/2502.17416v1#bib.bib61)) – this aligns with various scaling laws (Kaplan et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib26), Hoffmann et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib24), Allen-Zhu & Li, [2024](https://arxiv.org/html/2502.17416v1#bib.bib1)) that show that the performance of language models is very strongly dependent on the model size (i.e., number of parameters) and much lesser on other architectural design choices. However, recent works have started to question this view. Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)) argue that scaling laws for reasoning are more subtle, and depth is very important in addition to parameter count – at the same parameter count, deeper but shallower models are better. This is a deviation from the conventional scaling law wisdom, but it intuitively makes sense because reasoning problems often requires multi-step compositional thinking, and thus, depth can play a crucial role.

In this work, we make a stronger claim – while depth is important, many reasoning problems do not necessarily require a lot of parameters. How does one solve reasoning problems with large depth but few parameters? We argue that looped models are perfectly suited for this, where the same function, parameterized with few parameters, is iteratively applied on the input. This leads us to our first important claim:

Claim 1: Many reasoning problems require depth but not necessarily parameters. That is, they can be solved via looped models

Looped models have been studied in the literature for parameter efficiency (Lan et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib30)), adaptive compute (Dehghani et al., [2018](https://arxiv.org/html/2502.17416v1#bib.bib11)), equilibrium models (Bai et al., [2019](https://arxiv.org/html/2502.17416v1#bib.bib2)) and for in-context learning (Yang et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib63), Gatmiry et al., [2024a](https://arxiv.org/html/2502.17416v1#bib.bib19)). In this work, we initiate the study of looped models in the context of reasoning. Admittedly, reasoning is not very well-defined and can be of various forms (Sun et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib57)). Acknowledging this hurdle, we focus on a non-exhaustive list of problems that intuitively require reasoning and that are inspired by reasoning benchmarks. Throughout the paper, we use the notation $\left(\right. k \bigotimes L \left.\right)$ to denote a $k$-layer model looped $L$ times (precise definition in [Section 2](https://arxiv.org/html/2502.17416v1#S2 "2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")), which has the same number of parameters as a $\left(\right. k \bigotimes 1 \left.\right)$ model and same flops as a $\left(\right. k ⁢ L \bigotimes 1 \left.\right)$ non-looped model (see [Figure 1](https://arxiv.org/html/2502.17416v1#S1.F1 "In 1 Introduction ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")). As a first step towards connecting looped models and reasoning, we empirically evaluate looped models on several simple reasoning tasks in the literature (e.g. [Section 2](https://arxiv.org/html/2502.17416v1#S2 "2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")). Perhaps surprisingly, we find that a $\left(\right. k \bigotimes L \left.\right)$ looped models does almost as well as, if not better than, a non-looped model $\left(\right. k ⁢ L \bigotimes 1 \left.\right)$ that has the same effective depth but $L$ times more parameters on these reasoning tasks. The looped model is also significantly better than a $\left(\right. k \bigotimes 1 \left.\right)$ model which has the same number of parameters. Our theoretical results on the expressiveness of looped models in representing iterative algorithms with short description further corroborate these empirical findings and provide strong support for our claim. This naturally raises an important question: do looped models benefit language modeling in a similar manner?

![Image 1: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/looping_illustration2.png)

Figure 1: Illustration of the simple and architecture agnostic looping mechanism that we consider. A $k$-layer block looped $L$ times (middle) is denoted by $\left(\right. k \bigotimes L \left.\right)$, which can essentially be viewed as a weighted shared model. The iso-param baseline, $\left(\right. k \bigotimes 1 \left.\right)$, is a $k$-layer model with the same number of _distinct_ parameters. The iso-FLOP baseline, $\left(\right. k ⁢ L \bigotimes 1 \left.\right)$, is a $k ⁢ L$-layer model with the same depth but $L$ times more parameters. Middle looping is a strategy that is inspired from prior works on model stacking (e.g. (Saunshi et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib52))).

Claim 2: For language modeling, looped models have an inductive bias towards good reasoning despite having worse perplexity and memorization to an iso-flop non-looped model

For the above claim, we again train a $\left(\right. k \bigotimes L \left.\right)$ looped model on causal language modeling and compare it to the iso-param $\left(\right. k \bigotimes 1 \left.\right)$ and iso-flop $\left(\right. k ⁢ L \bigotimes 1 \left.\right)$ non-looped baselines. While the looped model improves over the iso-param baseline, perhaps unsurprisingly, it ends up with worse perplexity than iso-flop baseline, since perplexity depends strongly on number of parameters. However, the downstream evaluations reveal an intriguing trend: looped models have a tendency to improve tasks that require reasoning a lot more than memorization tasks. Specifically, the looped model has reasoning performance much closer to the iso-flop baseline, sometimes even exceeding it despite having $L$ times fewer parameters and worse perplexity. This contrasting behavior between the pretraining and downstream metrics has been a subject of study lately (Saunshi et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib51), Liu et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib35)) and is attributed to the _inductive biases_ introduced due to different architectures and training algorithms. Our empirical analysis also uncovers an interesting phenomenon: accuracy on downstream tasks scale as logarithm of the effective depth. In particular, more loops enhances performance, and the relative benefit of loops is higher for tasks that require more reasoning. This is conceptually similar to inference time scaling discovered for CoT, but with looping as a central component. To further elucidate this interesting relationship of looped models with CoT, we present the following claim.

Claim 3: Looped models generate latent thoughts and can, in theory, simulate CoT reasoning

Note that CoT reasoning gives the model more time and compute by generating multiple thought tokens before the answer, and it has powered the recent paradigm of inference-time scaling for “thinking” models like O1 and DeepSeek’s R1 (Guo et al., [2025](https://arxiv.org/html/2502.17416v1#bib.bib23)). We make an observation about CoT reasoning – it is essentially a looped model that generates 1 thought token in each iteration. However, looped models seem to be much more powerful, since they can generate multiple _latent thoughts_ in each iteration. We translate this intuition into a theoretical result about how looped models can simulate CoT reasoning.

Motivated by these findings, we propose a regularization scheme that aims to tap into the inductive bias of looped models towards reasoning. This leads us to our final claim:

Claim 4: Looping-inspired regularization can leverage this inductive bias towards better reasoning

With the backdrop of these claims, we concretely present the contributions of the paper below:

*   •In this paper we study looped models – multilayer models with weight sharing – and their role in reasoning. In particular, we compare a $k$-layer model looped $L$ times, denoted by $\left(\right. k \bigotimes L \left.\right)$, with an iso-param$\left(\right. k \bigotimes 1 \left.\right)$ non-looped model with $k$ layers and an iso-flop$\left(\right. k ⁢ L \bigotimes 1 \left.\right)$ model with $k ⁢ L$ layers and $L$ times more parameters. 
*   •We conduct experiments on synthetic reasoning tasks like addition, $p$-hop induction and GSM-style math word problems in [Section 2](https://arxiv.org/html/2502.17416v1#S2 "2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). For these tasks, we surprisingly find that iso-flop looped models, despite having way fewer parameters, can nearly match or outperform a non-looped model. Supporting these experiments, in [Section 5](https://arxiv.org/html/2502.17416v1#S5 "5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") we present theoretical results for why looped models can solve such problems with almost optimal depth. 
*   •In [Section 3](https://arxiv.org/html/2502.17416v1#S3 "3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), we train looped models on causal language modeling at 1B parameter scale. Here, we show that looped models have an inductive bias towards doing well on reasoning benchmarks, despite having much worse perplexity. This finding is novel, since most prior work on looping focused more on perplexity metrics rather than downstream reasoning tasks. We validate this inductive bias by visualizing the perplexity vs downstream performance plots as training process. Additionally, we show that looped models demonstrate good scaling behavior on various benchmarks as the number of loops are increased, akin to CoT reasoning. Finally, we show that looped models, along with a scratchpad, can simulate chain-of-thought reasoning. 
*   •Inspired by this inductive bias, in [Section 4](https://arxiv.org/html/2502.17416v1#S4 "4 Looping-inspired regularization ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), we propose a regularization that encourages layers to be more similar to each other. We find that training with such a regularization inherits the inductive bias of looped models towards reasoning without affecting perplexity. 

## 2 Looped models on simple reasoning tasks

We first explore our hypothesis of looped models helping reasoning tasks on a set of tasks constructed in a procedural manner. The illustrative reasoning tasks we consider are: $n$-ary addition, $p$-hop induction head that tests the model’s ability to track back for $p$ steps, and i-GSM which consists of synthetically constructed grade-school math problems. While these obviously do not cover the whole spectrum of reasoning problems, they provide useful insights into looped models and provide a basis for the theoretical results in [Section 5](https://arxiv.org/html/2502.17416v1#S5 "5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

Looped models. While many variants of looped model have been proposed (Lan et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib30), Dehghani et al., [2018](https://arxiv.org/html/2502.17416v1#bib.bib11), Giannou et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib21), Yang et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib63), Mohtashami et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib40)), we use the vanilla version for simplicity of our exploration. For any sequence-to-sequence function $f$, we denote $f^{\left(\right. L \left.\right)} = f \circ f ⁢ ⋯ \circ f$ to be the function that is $f$ looped $L$ times. In general, the looping mechanism is independent of architectural choices for $f$. For the rest of the paper, we typically use $f$ to denote a $k$-layer Transformer backbone of a model. Thus, $f$ looped $L$ times is the same as a $k ⁢ L$ layer model with weight sharing between all $L$ blocks of $k$ consecutive layers. We denote such looped models with the notation $\left(\right. k \bigotimes L \left.\right)$. Please refer to [Figure 1](https://arxiv.org/html/2502.17416v1#S1.F1 "In 1 Introduction ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for a succinct illustration of the looping mechanism. [Section 5.1](https://arxiv.org/html/2502.17416v1#S5.SS1 "5.1 Preliminaries and Notations ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") provides a more formal definition of looped transformers that is used for theoretical analysis.

### 2.1 Experiments with simple reasoning problems

$n$-ary addition. We consider the problem of adding $n$ numbers with 3 digits each. Addition is popular in the literature to study aspects of reasoning with Transformers, such as use of scratchpad (Nye et al., [2021](https://arxiv.org/html/2502.17416v1#bib.bib42)), chain of thought reasoning (Lee et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib31), Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32)) and length generalization (Cho et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib7)). One reason for its popularity is that addition can have algorithmic solutions, which is a feature of many reasoning problems. For our experiments, we train on a uniform mixture on numbers of operands $n \in \left{\right. 2 , 4 , 8 , 16 , 32 \left.\right}$ and sample each 3-digit operand uniformly at random between $\left[\right. 0 , 999 \left]\right.$. We train all models directly on the input-output pair, without any chain-of-thought steps. Following is an example for $n = 4$:

Input: “$315 + 120 + 045 + 824 =$” ; Output = “$1304$”.

We train a standard Transformer-based baseline $\left(\right. 12 \bigotimes 1 \left.\right)$ model with 12 layers. Please refer to [Section A.1](https://arxiv.org/html/2502.17416v1#A1.SS1.SSS0.Px1 "𝑛-ary addition. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for details on the training setup. We also train $\left(\right. k \bigotimes 12 / k \left.\right)$ looped model and an iso-param $\left(\right. k \bigotimes 1 \left.\right)$ baseline models for comparison, and vary $k \in \left{\right. 2 , 3 , 4 , 6 \left.\right}$. All trained models are finally evaluated separately on each of $n \in \left{\right. 8 , 16 , 24 , 32 \left.\right}$ to measure accuracy on increasingly difficult problems. Results are presented in [Table 1](https://arxiv.org/html/2502.17416v1#S2.T1 "In 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

Table 1: Accuracy of looped and non-looped models on the addition problem (left) and $p$-hop induction (right), as described in [Section 2](https://arxiv.org/html/2502.17416v1#S2 "2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). Left. For addition, we report accuracies for different number of operands ($n$). For all budgets, a $\left(\right. k \bigotimes 12 / k \left.\right)$ looped model is significantly better than the iso-param $\left(\right. k \bigotimes 1 \left.\right)$ model and also nearly as good as the non-looped iso-flop $\left(\right. 12 \bigotimes 1 \left.\right)$ baseline model. Right. The findings are very similar for the $p$-hop problem for different values of $p$. Note that a random guessing baseline would get at least 25% accuracy (since only 4 choices for answer). This suggests that depth via looping and small number of parameters is very effective for these problems.

| Addition of $n$ numbers |  |
| --- |
|  | Params / | $n = 8$ | $n = 16$ | $n = 24$ | $n = 32$ |
|  | FLOPs |  |  |  |  |
| Base $\left(\right. 12 \bigotimes 1 \left.\right)$ | 12x / 12x | 100.0 | 100.0 | 100.0 | 100.0 |
| 1 layer model |  |  |  |
| Base $\left(\right. 1 \bigotimes 1 \left.\right)$ | 1x / 1x | 0.1 | 0.1 | 0.1 | 0.0 |
| Loop $\left(\right. 1 \bigotimes 12 \left.\right)$ | 1x / 12x | 99.9 | 100.0 | 99.9 | 99.6 |
| 2 layer model |  |  |  |
| Base $\left(\right. 2 \bigotimes 1 \left.\right)$ | 2x / 2x | 85.8 | 71.5 | 49.3 | 38.8 |
| Loop $\left(\right. 2 \bigotimes 6 \left.\right)$ | 2x / 12x | 100.0 | 99.8 | 99.7 | 99.5 |
| 3 layer model |  |  |  |
| Base $\left(\right. 3 \bigotimes 1 \left.\right)$ | 3x / 3x | 97.2 | 78.5 | 69.2 | 60.7 |
| Loop $\left(\right. 3 \bigotimes 4 \left.\right)$ | 3x / 12x | 100.0 | 99.1 | 97.0 | 96.6 |

| $p$-hop with $n$ tokens |
| --- |
|  | Params / | $p = 16$ | $p = 32$ |
|  | FLOPs | $n = 256$ | $n = 256$ |
| Base $\left(\right. 6 \bigotimes 1 \left.\right)$ | 6x / 6x | 99.9 | 99.6 |
| 1 layer model |  |
| Base $\left(\right. 1 \bigotimes 1 \left.\right)$ | 1x / 1x | 48.9 | 49.0 |
| Loop $\left(\right. 1 \bigotimes 6 \left.\right)$ | 1x / 6x | 99.9 | 99.5 |
| 2 layer model |  |
| Base $\left(\right. 2 \bigotimes 1 \left.\right)$ | 2x / 2x | 68.8 | 59.4 |
| Loop $\left(\right. 2 \bigotimes 3 \left.\right)$ | 2x / 6x | 99.9 | 99.8 |
| 3 layer model |  |
| Base $\left(\right. 3 \bigotimes 1 \left.\right)$ | 3x / 3x | 97.2 | 73.0 |
| Loop $\left(\right. 3 \bigotimes 2 \left.\right)$ | 3x / 6x | 99.9 | 99.5 |

We find that, while the shallower baselines $\left(\right. k \bigotimes 1 \left.\right)$ degrade with lower $k$, the looped model $\left(\right. k \bigotimes 12 / k \left.\right)$ performs very well, and nearly matches the iso-flop $\left(\right. 12 \bigotimes 1 \left.\right)$ baseline. In fact, even a 1-layer network looped 12 times is able to solve this, despite using merely $1 / 12^{t ⁢ h}$ of the parameters of the baseline. This suggests the addition problem primarily requires depth, but not necessarily more parameters.

##### $p$-hop induction.

The $p$-hop problem is a synthetic induction task studied in Sanford et al. ([2024b](https://arxiv.org/html/2502.17416v1#bib.bib50)), who were inspired by the analysis of induction heads from Elhage et al. ([2021](https://arxiv.org/html/2502.17416v1#bib.bib14)). Specifically, given a sequence of letters $𝒗 = \left(\right. v_{1} ⁢ \ldots ⁢ v_{n} \left.\right)$ from an alphabet $\Sigma$, an induction head tries to find the penultimate occurrence of $v_{n}$ (from left to right) in the sequence and output the character immediately succeeding it. The $p$-hop problem generalizes this idea to sequentially hop $p$ times. Intuitively, the $p$-hop problem tests a model’s ability to recursively backtrack and retrieve the answer for a given query. This is reminiscent of the reasoning abilities required to solve reading comprehension kind of problems. We present the formal definition of the $p$-hop problem in [Definition A.1](https://arxiv.org/html/2502.17416v1#A1.Thmdefinition1 "Definition A.1 (𝑝-𝗁𝗈𝗉, Sanford et al. (2024b)). ‣ 𝑝-hop induction. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). We perform experiments with looped and non-looped models on the $p$-hop problem, with alphabet size set to 4 and sequences of length $256$, and vary $p$ between 16 and 32 to control the problem difficulty. Our observations are presented in Table[1](https://arxiv.org/html/2502.17416v1#S2.T1 "Table 1 ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). Similarly to our findings on the addition task, reasonably deep looped models perform as well as the baseline using much fewer parameters.

##### i-GSM (Synthetic Grade School Math Problems).

Inspired by Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)), we built our own version of grade-school level math word problems. While we follow many of the design guidelines of Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)), we make a few simplifying changes. We generate the math problem as a DAG of arithmetic computations modulo 7, and restrict the depth of the graph to 4. For simplicity, we retain the problems in the symbolic form and do not map them to English (e.g., see Table[2](https://arxiv.org/html/2502.17416v1#S2.T2 "Table 2 ‣ i-GSM (Synthetic Grade School Math Problems). ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) 1 1 1 Similar to Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)), the simplified setting still allows for over 7 billion unique solution templates. We train models of depths 1, 2, 4 and 8 and compare them with different looped variants in Table[2](https://arxiv.org/html/2502.17416v1#S2.T2 "Table 2 ‣ i-GSM (Synthetic Grade School Math Problems). ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). The answer is computed modulo 7. Hence, a random guessing baseline would get at least 14% accuracy. [Section A.1](https://arxiv.org/html/2502.17416v1#A1.SS1.SSS0.Px3 "i-GSM. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") has more details. Remarkably, we again observe that a depth $k$ model looped $L$ times matches or outperforms a depth $k ⁢ L$ model and far outperforms a depth $k$ non-looped model. This suggests that even a more complex and realistic looking reasoning problem does not need too many parameters and can benefit from depth via looping.

Table 2: Left. Symbolic i-GSM problem and its solution. Right. Accuracy of looped and non-looped models on the i-GSM task from [Section 2](https://arxiv.org/html/2502.17416v1#S2 "2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). $\left(\right. k \bigotimes L \left.\right)$ looped model is significantly better than the iso-param $\left(\right. k \bigotimes 1 \left.\right)$ model and performs as well as non-looped iso-flop $\left(\right. k ⁢ L \bigotimes 1 \left.\right)$ model.

##### Question.

E#I := 4. E#J := E#I. K#N := I#N + J#O + F#K. F#K := E#J. J#O := F#K + K#O + E#J. H#J := E#J + F#K. I#P := L#M + I#N + K#O. I#M := J#O + J#P + F#K. J#P := H#J - F#K. L#M := I#N + J#P + F#K. I#N := 2 * J#P + H#J + E#I. K#O := J#P + I#N + E#J. I#P?

##### Answer with CoT.

E#I = 4. $\Longrightarrow$ E#I = 4. E#J = E#I. $\Longrightarrow$ E#J = 4. F#K = E#J. $\Longrightarrow$ F#K = 4. H#J = E#J+F#K. $\Longrightarrow$ H#J = 1. J#P = H#J-F#K. $\Longrightarrow$ J#P = 4. I#N = 2J#P+2H#J+2E#I. $\Longrightarrow$ I#N = 4. L#M = I#N+J#P+F#K. $\Longrightarrow$ L#M = 5. K#O = J#P+I#N+E#J. $\Longrightarrow$ K#O = 5. I#P = L#M+I#N+K#O. $\Longrightarrow$ I#P = 0.

|  | Params / FLOPs | Accuracy |
| --- |
| Base $\left(\right. 8 \bigotimes 1 \left.\right)$ | 8x / 8x | 73.2 |
| 1 layer model |
| Base $\left(\right. 1 \bigotimes 1 \left.\right)$ | 1x / 1x | 24.5 |
| Loop $\left(\right. 1 \bigotimes 2 \left.\right)$ | 1x / 2x | 52.3 |
| Loop $\left(\right. 1 \bigotimes 4 \left.\right)$ | 1x / 4x | 69.9 |
| Loop $\left(\right. 1 \bigotimes 8 \left.\right)$ | 1x / 8x | 73.2 |
| 2 layer model |
| Base $\left(\right. 2 \bigotimes 1 \left.\right)$ | 2x / 2x | 54.0 |
| Loop $\left(\right. 2 \bigotimes 2 \left.\right)$ | 2x / 4x | 66.9 |
| Loop $\left(\right. 2 \bigotimes 4 \left.\right)$ | 2x / 8x | 73.6 |
| 4 layer model |
| Base $\left(\right. 4 \bigotimes 1 \left.\right)$ | 4x / 4x | 71.3 |
| Loop $\left(\right. 4 \bigotimes 2 \left.\right)$ | 4x / 8x | 71.6 |

## 3 Language modeling with looped models

In this section, we pretrain and evaluate looped models for causal language models. We train models on 250B tokens of the Pile dataset (Gao et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib17)) and use a 24-layer 1B parameter model for most experiments, motivated by the setting in Tay et al. ([2022](https://arxiv.org/html/2502.17416v1#bib.bib59)) (refer to [Section A.2](https://arxiv.org/html/2502.17416v1#A1.SS2 "A.2 Language modeling setup ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for more details).

### 3.1 Experiments with 1B language modeling

For causal language modeling, we pretrain various looped models on the standard GPT2-style next token prediction objective (Radford et al., [2019](https://arxiv.org/html/2502.17416v1#bib.bib45)). We train models with different parameter budgets to make sure that the findings are robust. We remind the reader that the notation $\left(\right. k \bigotimes L \left.\right)$ corresponds to a $k$ layer model looped $L$ times. For each setting, we compare 3 models: (a)$\left(\right. 24 \bigotimes 1 \left.\right)$: 24-layer 1B model, (b)$\left(\right. k \bigotimes 1 \left.\right)$: $k$-layer model with the same configuration as the 24-layer model for other dimensions, (c)$\left(\right. k \bigotimes 24 / k \left.\right)$: $k$-layer model looped $24 / k$ times to match the parameter count of (b) and match the effective depth/FLOPs of (a). We run experiments for $k \in \left{\right. 4 , 6 , 8 , 12 \left.\right}$ to ensure that the findings are robust. After pretraining on Pile, we evaluate the models on validation perplexity and on downstream benchmarks using $k$-shot evaluations. Results are summarized in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")

Table 3: Downstream evaluations for language models trained on the Pile dataset. Comparisons include a 24-layer 1B-parameter baseline model, iso-flop looped models $\left(\right. k \bigotimes 24 / k \left.\right)$ for various parameter budgets $k$, and the corresponding iso-param baselines $\left(\right. k \bigotimes 1 \left.\right)$. Downstream evaluations are averaged over tasks within 4 task groups. We also include the % Gap metric for each $k$ to measure the gap between the iso-param and iso-flop baselines that is covered by the looped model (see [Equation 1](https://arxiv.org/html/2502.17416v1#S3.E1 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")). Overall the looped models are worse on perplexity and closed book QA (memorization benchmarks), but the % Gap is much higher for task groups that require reasoning (open book QA, math word problems). In fact for reasoning primitives, which are purely testing for reasoning skills, the looped models are much better than the 1B baseline for all $k$, despite having $24 / k \times$ fewer parameters.

Params /Perplexity ($\downarrow$)Closed Open Math Word All Tasks Reasoning
FLOPs(validation)Book QA ($\uparrow$)Book QA ($\uparrow$)Problems ($\uparrow$)Average ($\uparrow$)Primitives ($\uparrow$)
(4 tasks)(5 tasks)(6 tasks)(15 tasks)(4 tasks)
24 layers
Baseline 24x / 24x 7.40 11.2 33.9 29.3 26.0 47.5
12 layers
Base $\left(\right. 12 \bigotimes 1 \left.\right)$12x / 12x 8.16 8.2 26.9 26.7 21.8 35.7
Loop $\left(\right. 12 \bigotimes 2 \left.\right)$12x / 24x 7.90 9.3 30.8 34.3 26.5 51.2
% Gap 34 %37 %56 %282 %110 %131 %
Middle Loop 12x / 24x 7.81 11.0 32.3 28.3 25.0 56.5
$\left(\right. 4 \bigotimes 1 , 4 , 1 \left.\right)$
% Gap 46 %94 %78 %62 %95 %176 %
8 layers
Base $\left(\right. 8 \bigotimes 1 \left.\right)$8x / 8x 8.75 6.3 22.7 17.1 16.1 33.0
Loop $\left(\right. 8 \bigotimes 3 \left.\right)$8x / 24x 8.19 8.5 30.8 28.4 23.9 55.3
% Gap 41 %44 %72 %92 %78 %153 %
6 layers
Base $\left(\right. 6 \bigotimes 1 \left.\right)$6x / 6x 9.25 4.0 19.3 17.7 14.6 24.1
Loop $\left(\right. 6 \bigotimes 4 \left.\right)$6x / 24x 8.42 8.2 28.7 29.8 23.7 56.1
% Gap 44 %58 %64 %104 %80 %136 %
4 layers
Base $\left(\right. 4 \bigotimes 1 \left.\right)$4x / 4x 10.12 1.8 13.8 9.7 9.0 19.4
Loop $\left(\right. 4 \bigotimes 6 \left.\right)$4x / 24x 8.79 6.7 26.2 24.8 20.4 56.9
% Gap 48 %52 %61 %77 %67 %133 %

Evaluation metrics. We evaluate the models on perplexity metric after training is completed. Since there is growing evidence that perplexity, although very useful for training, is a narrow measure of model quality, we also track more holistic downstream evaluations(Liang et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib33)). Thus, we evaluate the model on 4 important slices: closed book QA, open book QA, math word problems and reasoning primitives. These comprise of 19 different tasks in total.

*   •Closed book QA: This includes tasks like TriviaQA (Joshi et al., [2017](https://arxiv.org/html/2502.17416v1#bib.bib25)), TydiQA-NoContext (Clark et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib9)), Natural Questions (Kwiatkowski et al., [2019](https://arxiv.org/html/2502.17416v1#bib.bib29)) and Web Questions (Talmor & Berant, [2018](https://arxiv.org/html/2502.17416v1#bib.bib58)) that test the model’s ability to answer questions without any context, and thus, primarily measure the memorization abilities of language models. 
*   •Open book QA: This includes tasks like TydiQA-GoldP (Clark et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib9)), SquadV2 (Rajpurkar et al., [2018](https://arxiv.org/html/2502.17416v1#bib.bib46)), Drop (Dua et al., [2019](https://arxiv.org/html/2502.17416v1#bib.bib12)), QuAC (Choi et al., [2018](https://arxiv.org/html/2502.17416v1#bib.bib8)), CoQA (Reddy et al., [2019](https://arxiv.org/html/2502.17416v1#bib.bib48)) that evaluate the model’s ability to infer the answer to a question from the extra context that is provided, akin to reading comprehension. 
*   •Math word problems: To evaluate the model’s ability to reason, we test them on math word problems considered in (Wei et al., [2022b](https://arxiv.org/html/2502.17416v1#bib.bib62)). This includes tasks like SVAMP (Patel et al., [2021](https://arxiv.org/html/2502.17416v1#bib.bib43)), ASDiv (Miao et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib39)), the MAWPS benchmark (Koncel-Kedziorski et al., [2016](https://arxiv.org/html/2502.17416v1#bib.bib27)). We report 5-shot evaluation for the pretrained model on these tasks. 
*   •Reasoning primitives: Saunshi et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib52)) introduced these datasets to study the inductive bias of stacking towards improving reasoning, by isolating simple reasoning abilities. One such primitive is depth-$k$ variable assignment that requires the model to resolve a chain of assignments of length $k$. An example of depth-0 var-assign is a=1, b=2, c=6, b=?, and example for depth-1 var-assign is a=1, b=2, c=a, d=b, d=?. We evaluate on the math and coding variants of the depth-0 and depth-1 problems using 5-shot evaluation. 

For each task group $G$ from above, in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") we report the average accuracy for that task group, denoted by $\text{Avg}_{G} \text{Avg}$. Furthermore, for each layer budget $k$, we report the % gap between the iso-param and iso-flop models that is covered by the looped model. More specifically

$\%\text{ Gap} = \frac{\text{Avg}_{G} ⁢ \left(\right. k \bigotimes 24 / k \left.\right) - \text{Avg}_{G} ⁢ \left(\right. k \bigotimes 1 \left.\right)}{\text{Avg}_{G} ⁢ \left(\right. 24 \bigotimes 1 \left.\right) - \text{Avg}_{G} ⁢ \left(\right. k \bigotimes 1 \left.\right)} . \%\text{ Gap} \text{Avg} \text{Avg} \text{Avg} \text{Avg}$(1)

This measures how effectively looping can bridge the gap between iso-param and iso-flops baselines. Implicitly, it measures how different task groups behave when a model only has a few parameters but is given depth through looping. % Gap being closer to 0% means that providing depth via looping does not benefit the task group, and number of parameters is the most important factor. On the other hand, % Gap closer to 100% means parameter count matters much less for the task group, and that depth via looping is more essential.

Perplexity results. Firstly we notice that all $\left(\right. k \bigotimes 24 / k \left.\right)$ looped models have better perplexity compared to the iso-param $\left(\right. k \bigotimes 1 \left.\right)$ baseline, but worse perplexity compared to the non-looped 24-layer baseline. The looped models only covers up roughly $34 - 50 \%$ of the perplexity gap between the iso-param and iso-flop baselines for various values of $k$. This perplexity gap is not too surprising since the looped model has $24 / k$ times fewer parameters, and thus, lower capacity than the 24-layer baseline. This was also been observed in prior works (Lan et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib30), Mohtashami et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib40)) and is the primary reason looped models have been ignored for language modeling. However, as we shall see shortly, the downstream metrics paint a more interesting and favorable picture.

Results on QA tasks. We first consider closed book and open book QA categories in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). Closed book QA tasks are primarily testing the model’s memorization abilities. Open book QA on the other hand tests the model’s ability to infer the answer from the additional context that is provided. Thus, intuitively, open book QA tasks require more reasoning. Firstly we notice that the % Gap for closed book QA (memorization) is very similar to % Gap for perplexity. The % Gap for open book QA, on the other hand, is much higher for all parameter budgets. This suggests that looped models relatively improve contextual QA much more than memorization based QA.

![Image 2: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/12L_closedQA.png)

![Image 3: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/12L_openQA.png)

![Image 4: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/12L_mwp.png)

![Image 5: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/12L_primitives.png)

![Image 6: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/24L_closedQA.png)

![Image 7: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/24L_openQA.png)

![Image 8: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/24L_mwp.png)

![Image 9: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/24L_primitives.png)

Figure 2: Downstream evaluation for various task groups on the x-axis, vs validation log perplexity on the y-axis (reversed), as training proceeds. The top plots compare a 12-layer baseline model $\left(\right. 12 \bigotimes 1 \left.\right)$ and the looped model $\left(\right. 12 \bigotimes 2 \left.\right)$. The second row compares the baseline 24-layer model and the 24-layer model trained with regularization using block size $k = 4$ and $\lambda_{\text{reg}} = 10 \text{reg}$ (See [Equation 4](https://arxiv.org/html/2502.17416v1#S4.E4 "In 4 Looping-inspired regularization ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")). For both comparisons we have similar observations. For closed book QA (memorization) tasks looping has very similar trends to baseline. For open book QA tasks and math word problems, looping has much better downstream performance at an equivalent log perplexity. This verifies the inductive bias of looping and regularization towards better reasoning abilities.

Math problems and reasoning primitves. We also present the % Gap for the math word problem in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). Surprisingly, we find that $\left(\right. k \bigotimes 24 / k \left.\right)$ looped model can almost match the baseline $\left(\right. 24 \bigotimes 1 \left.\right)$ model for $k \geq 6$, despite having $k$ times fewer parameters. In fact, the 12 layer model looped twice is even signficantly better ($34.3$) than the 24 layer baseline ($29.3$), despite having 50% of parameters and worse perplexity; suggesting that looping disproportionately improves mathematical reasoning.

To better understand the effect on reasoning, we direct the readers attention to the evaluations for reasoning primitives in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). The results are quite remarkable: $\left(\right. k \bigotimes 24 / k \left.\right)$ looped models are better than the iso-flop baseline $\left(\right. 24 \bigotimes 1 \left.\right)$ at reasoning primitives, for all values of $k$. This is a priori very surprising, since these are synthetic generated tasks and have nothing to do with the pretraining data or the model architecture. Thus, solving these tasks necessarily requires reasoning from context, and memorization abilities will not help here. These results clearly suggest that looped models have a bias towards improving reasoning, despite having worse perplexity and memorization abilities. Next, we formalize the inductive bias towards reasoning via isoplots.

### 3.2 Inductive bias towards reasoning

In this section, we formalize the inductive bias by plotting the perplexity vs downstream metric iso-plots, as introduced in Saunshi et al. ([2022](https://arxiv.org/html/2502.17416v1#bib.bib51)). [Section 3.1](https://arxiv.org/html/2502.17416v1#S3.SS1 "3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") showed that looped models have higher than expected performance on reasoning problems. However, since looped models are worse on perplexity, it is hard to make a direct comparison between various models. One way to bring parity between models is to look at their downstream performances at the same validation pretraining loss (Liu et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib35)). Saunshi et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib52)) proposed plotting pretraining loss vs downstream metrics as training proceeds, as a way to study the inductive bias of various methods. For each model, we evaluate the log perplexity and downstream metrics at every 20k steps, starting from 120k steps. We plot these values in a scatter plot and fit a linear function with log perplexity and the corresponding downstream metric being input and output respectively. Please refer to [Figures 2](https://arxiv.org/html/2502.17416v1#S3.F2 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") and[7](https://arxiv.org/html/2502.17416v1#A1.F7 "Figure 7 ‣ i-GSM. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for two sets of isoplots.

Findings. For all values of $k$, we observe the following:

*   •The isoplots for $\left(\right. k \bigotimes L \left.\right)$ looped model and $\left(\right. k \bigotimes 1 \left.\right)$ baseline are very aligned for closed book QA tasks (if extrapolated). This suggests that log perplexity is a very strong indicator of downstream performance on memorization based tasks. 
*   •For open book QA and math word problems, the isoplot line for the looped model is always higher than the baseline model. This suggests that at the same log perplexity, looped models will tend to have higher evaluation on these tasks that require more reasoning. 
*   •For reasoning primitives, there is a stark difference between looped and baseline models. The looped model seems to have good performance at most points in training. 

Overall this suggests a strong inductive bias of looped models towards improving reasoning.

### 3.3 Middle looping variant and relationship with Gradual stacking

Recently Saunshi et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib52)) introduced a gradual stacking (Gong et al., [2019](https://arxiv.org/html/2502.17416v1#bib.bib22), Reddi et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib47)) approach for training language models called MidAS. This approach gradually grows the model depth as training proceeds by duplicating certain layers of the model in each stacking operation. Surprisingly, they found that MidAS not only speeds up pretraining, but also improves reasoning in the same sense as [Figure 2](https://arxiv.org/html/2502.17416v1#S3.F2 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") – better reasoning at the same perplexity. Furthermore, the paper established a strong connection between stacking via MidAS and looped models, owing to the layer duplication operation, and conjectured that this is the reason for such an inductive bias. Our results from the previous section provides a compelling evidence for this conjecture by showing that looped models also show a very similar inductive bias, thus, further strengthening the connection between stacking and looped models. Why such an inductive bias occurs is still an open question, and we believe that understanding this is an important future direction.

Furthermore, inspired by their findings, we explore middle looping (see [Figure 1](https://arxiv.org/html/2502.17416v1#S1.F1 "In 1 Introduction ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for an illustration) — a variant of looping which maintains independent layers at the start and the end of the network, and perform looping on the middle block of layers. The high-level intuition from Saunshi et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib52)) is that the first and last layers play a special role in the model and thus, should be treated differently from the middle layers. In [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), we report results for a version of middle looping that is iso-param with a $\left(\right. 12 \bigotimes 1 \left.\right)$ baseline and iso-flop with a $\left(\right. 24 \bigotimes 1 \left.\right)$ baseline, just like the $\left(\right. 12 \bigotimes 2 \left.\right)$ model. Overall, we find that middle looping has better perplexity and more uniform improvements than the default looping of $\left(\right. 12 \bigotimes 2 \left.\right)$ (except for math word problems), and thus, might be a more practical looping approach. We leave the exploration of the best looping strategies for future work.

### 3.4 Scaling behavior of looping

![Image 10: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/depth_scaling_eval_perplexity.png)

![Image 11: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/depth_scaling_closedQA.png)

![Image 12: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/depth_scaling_openQA.png)

![Image 13: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/depth_scaling_eval_token_acc.png)

![Image 14: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/depth_scaling_mwp.png)

![Image 15: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/depth_scaling_primitives.png)

Figure 3: Scaling behavior for various task group as the effective depth increases. The blue curve shows how performance scales as the number of loops increases, without increasing parameters, using models of the form $\left(\right. 4 \bigotimes D / 4 \left.\right)$ for various values of $D$. The orange curve visualizes the scaling behavior of $\left(\right. D \bigotimes 1 \left.\right)$ which increases the depth by adding fresh parameters. For reasoning primitives, the looped model scales as well, or even better, than the baseline despite having $D / 4$ fewer parameters.

In this section, we discuss an intriguing scaling behavior of looping, specifically the impact of the number of loops on various evaluation metrics. In particular, we are interested in scaling behavior of: (a) accuracy as a function of number of loops and (b) comparison of looped and non-looped baseline of the same effective depth. To this end, we pretrain various looped language models of the form $\left(\right. 4 \bigotimes L \left.\right)$, i.e., 4-layer model looped $L$ times, for $L \in \left{\right. 1 , 2 , 3 , 6 , 9 , 12 \left.\right}$. To enable iso-FLOPs comparison, we also train baseline models of the form $\left(\right. 4 ⁢ L \bigotimes 1 \left.\right)$, which has $L$ times more parameters. For each task group, we plot the average accuracy as a function of the _effective depth_, i.e. $D = 4 ⁢ L$. From the results presented in [Figure 3](https://arxiv.org/html/2502.17416v1#S3.F3 "In 3.4 Scaling behavior of looping ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), we observe the following.

1.   1.In both cases, we find that the accuracies for all task groups continue to increase with more loops/depth, although, unsurprisingly, the returns are diminishing with depth for both looped and non-looped models. Interestingly, for both looped and non-looped models, we found that one can fit a simple scaling law of the following form:

$\text{Acc} = \alpha ⁢ log ⁡ \left(\right. D \left.\right) + \beta , \text{Acc}$(2)

where $D$ is the effective depth and $\alpha$ measures the impact of depth on downstream performance. 
2.   2.Furthermore, for each task group, we compute $\alpha_{l ⁢ o ⁢ o ⁢ p} / \alpha_{b ⁢ a ⁢ s ⁢ e}$ to assess the relative impact of “depth via looping” compared to “depth via additional parameters”. We find that more loops continue to help, and the relative benefit of loops is higher for reasoning tasks like open book QA and math problems. Remarkably, the impact of loops is even higher (1.19x) than impact of depth for reasoning primitives, which further consolidates the benefit of looped models for reasoning. 

##### Latent thoughts and connections to CoT reasoning.

We end this discussion with an important question: _why should we expect this interesting scaling behavior of looped models?_ We argue that looped models have a strong connection to chain-of-thought (CoT) reasoning. Such a connection is insightful because recent works on thinking have shown CoT can demonstrate inference-time scaling behavior, where the accuracy on reasoning datasets continues to improve as the length of the models chain of thought increases; i.e., generating more thoughts leads to better reasoning.

To establish this connection, we make a simple observation about CoT reasoning – it is essentially a looped model that generates a single thought token in each iteration. However, looped models can be more powerful since they can generate multiple “latent thoughts” in each iteration. This can be visualized in [Figure 4](https://arxiv.org/html/2502.17416v1#S5.F4 "In 5.4 Looped models can simulate CoT reasoning ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). We further translate this intuition into a theoretical result (see [Section 5.4](https://arxiv.org/html/2502.17416v1#S5.SS4 "5.4 Looped models can simulate CoT reasoning ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) about how looped models can in fact simulate CoT reasoning. Given this connection to CoT reasoning, it is believable that looping can scale well for harder reasoning. We believe that leveraging looping explicitly for inference-time scaling is a very promising future direction.

## 4 Looping-inspired regularization

In the previous section, we observed the looped models can improve reasoning with worse perplexity. Can we leverage this observation to improve reasoning without affecting perplexity? Here, we propose a simple approach: regularize the weights of the model to encourage them to be close to a looped model. This could have two advantages, (a) the model still has free parameters to improve perplexity, (b) the closeness to looped model can inherit the desirable inductive bias. In particular, if an $L$-layer model is denoted as $f_{0} \circ f_{1} \ldots , \circ f_{L / k - 1}$, where each $f_{i}$ is a block of $k$ layers, we add a regularization term that makes all the weights of the block $f_{i}$ close to $f_{i + 1}$ in terms of cosine similarity. For a parameter group $G$ (e.g. first feed-forward layer, or query matrix in Transformer), we use $\theta_{G}^{\left(\right. 0 \left.\right)} , \theta_{G}^{\left(\right. 1 \left.\right)} , \ldots , \theta_{G}^{\left(\right. L - 1 \left.\right)}$ to denotes the weights in all layers. Then the regularization term is

$\mathcal{R}_{G} ⁢ \left(\right. k \left.\right) = \frac{1}{L - k} ⁢ \sum_{i = 0}^{\frac{L}{k} - 2} \sum_{j = 0}^{k - 1} \text{Cosine} ⁢ \left(\right. \theta_{G}^{\left(\right. i ⁢ k + j \left.\right)} , \theta_{G}^{\left(\right. \left(\right. i + 1 \left.\right) ⁢ k + j \left.\right)} \left.\right) \text{Cosine}$(3)

The final loss function is a sum of the standard cross-entropy loss and the regularization term averaged over all groups, multiplied by a scalar hyperparameter. Let $\mathcal{G}$ denote the set of all parameter groups; $\mathcal{G} = \left{\right. \text{Attn}-\text{Q} , \text{Attn}-\text{K} , \ldots , \text{FFN}-\text{W2} \left.\right} \text{Attn}-\text{Q} \text{Attn}-\text{K} \text{FFN}-\text{W2}$

$\mathcal{L} = \mathcal{L}_{\text{xent}} + \lambda_{\text{reg}} ⁢ \left(\left|\right. \mathcal{G} \left|\right.\right)^{- 1} ⁢ \underset{G \in \mathcal{G}}{\sum} \mathcal{R}_{G} ⁢ \left(\right. k \left.\right) \text{xent} \text{reg}$(4)

In the above formulation, $\lambda_{\text{reg}} = 0 \text{reg}$ would recover standard training and $\lambda_{\text{reg}} \rightarrow \infty \text{reg}$ would converge to a fully looped model. Intermediate values of $\lambda_{\text{reg}} \text{reg}$ will lead to “approximately” looped models. For instance, to emulate the $\left(\right. 4 \bigotimes 6 \left.\right)$ looped model setting, we use pick $k = 4$, $L = 24$ and a large regularization strength like $\lambda_{\text{reg}} = 10 \text{reg}$. All other hyperparameters are kept the same as baseline training for a fair comparison. We tried options other than cosine similarity, like $ℓ_{2}$ norm, to bring the weights closer but found that cosine was more robust and effective.

Cosine similarities. Firstly, we check if the regularization had the right effect by measuring the cosine similarities between the successive blocks of $k$ layers at the end of training. We, indeed, find that for all parameter groups, the cosine similarity around 0.98 or higher (see [Figure 5](https://arxiv.org/html/2502.17416v1#A1.F5 "In 𝑝-hop induction. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")).

Inductive bias. To confirm the inductive bias of the regularizer, we visualize the log perplexity vs downstream isoplots for $\lambda_{\text{reg}} = 10 \text{reg}$ and baseline models in [Figure 2](https://arxiv.org/html/2502.17416v1#S3.F2 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). While the plots are similar for closed book QA, a strong inductive bias shows up for open book QA and reasoning problems. Crucially, the regularized model does well on reasoning without hurting perplexity (see [Table 4](https://arxiv.org/html/2502.17416v1#S4.T4 "In 4 Looping-inspired regularization ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")).

Table 4: Results for the 24-layer 1B model with and without the regularization introduced in [Section 4](https://arxiv.org/html/2502.17416v1#S4 "4 Looping-inspired regularization ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). We try various block sizes $k$ motivated by the looped model settings from [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). Overall, regularization helps retain the inductive bias towards reasoning, with notable improvements on math word problems and reasoning primitives, without almost neutral perplexity.

|  | Perplexity ($\downarrow$) | Closed | Open | Math Word | All Tasks | Reasoning |
| --- | --- | --- | --- | --- | --- | --- |
|  | (validation) | Book QA ($\uparrow$) | Book QA ($\uparrow$) | Problems ($\uparrow$) | Average ($\uparrow$) | Primitives ($\uparrow$) |
|  |  | (4 tasks) | (5 tasks) | (6 tasks) | (15 tasks) | (4 tasks) |
| Baseline | 7.40 | 11.2 | 33.9 | 29.3 | 26.0 | 47.5 |
| Regularized ($k = 4 , \lambda_{\text{reg}} = 1 \text{reg}$) | 7.41 | 11.2 | 34.8 | 31.6 | 27.2 | 42.5 |
| Regularized ($k = 4 , \lambda_{\text{reg}} = 10 \text{reg}$) | 7.38 | 12.5 | 36.2 | 36.4 | 30.0 | 57.2 |
| Regularized ($k = 6 , \lambda_{\text{reg}} = 10 \text{reg}$) | 7.40 | 12.0 | 35.8 | 31.0 | 27.5 | 55.8 |
| Regularized ($k = 8 , \lambda_{\text{reg}} = 10 \text{reg}$) | 7.43 | 11.3 | 34.4 | 32.8 | 27.6 | 56.3 |
| Regularized ($k = 12 , \lambda_{\text{reg}} = 10 \text{reg}$) | 7.51 | 10.1 | 34.1 | 32.3 | 27.0 | 50.7 |

## 5 Theoretical analysis for looped models

In this section, we present theoretical results to understand the phenomenon from the previous sections – why can looped model with few parameters match an iso-flops non-looped baseline on reasoning problems? While a complete theory is challenging, since “reasoning” is a very broad concept, the goal is to provide some intuition and formalization for the expressive power of looped models. First, we show that looped Transformers can effectively solve group composition (a generalization of the addition problem). Then we show a very general result on how a non-looped model with very few distinct layers can be simulated by a looped model with a small blowup in model size. This result is then used to solve the $p$-hop problem using a one-layer looped transformer. Our construction for group composition and $p$-hop problem are nearly optimal in terms of depth and much more efficient in terms of parameters compared to non-looped models.

### 5.1 Preliminaries and Notations

We first define the standard transformer architecture. Throughout the paper we will fix the dimension of the embedding to be $d \in \mathbb{N}^{+}$, the vocabulary to be $\mathcal{V}$ and maximum sequence length to be $n_{max}$. We will use $𝗂𝖽$ to denote the identity mapping. Here, we describe the high-level notation. Please refer to [Section B.1](https://arxiv.org/html/2502.17416v1#A2.SS1 "B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for detailed notations. We use $𝖥𝖥$ and $𝖬𝖧𝖠$ to denote the feed-forward and attention layers respectively, and $\theta_{𝖬𝖧𝖠}$ and $\theta_{𝖥𝖥}$ denote the parameters in these layers.

###### Definition 5.1(Transformer Block).

Given number of layers $L \in \mathbb{N}^{+}$ and parameter $\theta_{𝖳𝖡} = \left(\left(\right. \theta_{𝖬𝖧𝖠}^{\left(\right. l \left.\right)} , \theta_{𝖥𝖥}^{\left(\right. l \left.\right)} \left.\right)\right)_{l = 0}^{L - 1}$, $L$-layer transformer block $𝖳𝖡_{\theta_{𝖳𝖡}} : \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n} \rightarrow \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n}$ for any $n \in \mathbb{N}^{+}$ is defined as

$𝖳𝖡_{\theta_{𝖳𝖡}} \triangleq \left(\right. 𝗂𝖽 + 𝖥𝖥_{\theta_{𝖥𝖥}^{\left(\right. L - 1 \left.\right)}} \left.\right) \circ \left(\right. 𝗂𝖽 + 𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}^{\left(\right. L - 1 \left.\right)}} \left.\right) \circ ⋯ ⁢ \left(\right. 𝗂𝖽 + 𝖥𝖥_{\theta_{𝖥𝖥}^{\left(\right. 0 \left.\right)}} \left.\right) \circ \left(\right. 𝗂𝖽 + 𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}^{\left(\right. 0 \left.\right)}} \left.\right) ,$(5)

We also denote $𝖤𝖬𝖡𝖤𝖣$ and $𝖮𝖴𝖳𝖯𝖴𝖳$ to be the input embedding and output softmax layers respectively. Please refer to [Section B.1](https://arxiv.org/html/2502.17416v1#A2.SS1 "B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for precise definitions. Finally, we define the entire transformer model that maps a sequence of tokens to a distribution over tokens: $p_{\theta} : \cup_{n \leq n_{max}} \mathcal{V}^{n} \rightarrow \Delta^{\left|\right. \mathcal{V} \left|\right. - 1}$.

$p_{\theta} \triangleq 𝖮𝖴𝖳𝖯𝖴𝖳_{\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}} \circ 𝖳𝖡_{\theta_{𝖳𝖡}} \circ 𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖳𝖤} , \theta_{𝖯𝖤}}$(6)

where $\theta = \left(\right. \theta_{𝖳𝖡} , \theta_{𝖳𝖤} , \theta_{𝖯𝖤} , \theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \left.\right)$ denote all the transformer parameter. In particular, we use $𝖳𝖥_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \triangleq \left(arg ⁢ max\right)_{v \in \mathcal{V}} ⁡ p_{\theta} ⁢ \left(\right. v \left|\right. v_{1} , \ldots , v_{n} \left.\right)$ to denote the deterministic version of the transformer model. We now define a looped Transformer model that also subsumes a non-looped model.

###### Definition 5.2($\left(\right. L \bigotimes T \left.\right)$ Looped Transformer).

Given the number of loops $T \in \mathbb{N}^{+}$, parameters $\theta = \left(\right. \theta_{𝖳𝖡} , \theta_{𝖳𝖤} , \theta_{𝖯𝖤} , \theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \left.\right)$, where $\theta_{𝖳𝖥} = \left(\left(\right. \theta_{𝖬𝖧𝖠}^{\left(\right. l \left.\right)} , \theta_{𝖥𝖥}^{\left(\right. l \left.\right)} \left.\right)\right)_{l = 0}^{L - 1}$, we define a $\left(\right. L \bigotimes T \left.\right)$_looped Transformer_ as $p_{\theta , T} \triangleq 𝖮𝖴𝖳𝖯𝖴𝖳_{\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}} \circ \left(\left(\right. 𝖳𝖡_{\theta_{𝖳𝖡}} \left.\right)\right)^{T} \circ 𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖳𝖤} , \theta_{𝖯𝖤}}$.

### 5.2 Group composition problem

We consider the problem of composing $n$ elements from a group, and prove that a standard 1-layer transformer looped $\mathcal{O} ⁢ \left(\right. log ⁡ \left(\right. n \left.\right) \left.\right)$ times can solve this problem. This is a generalization of the modular addition problem and has a long history (see [Section B.2.2](https://arxiv.org/html/2502.17416v1#A2.SS2.SSS2 "B.2.2 Group composition. ‣ B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")). Recently, Liu et al. ([2022](https://arxiv.org/html/2502.17416v1#bib.bib34)) show that transformers with $log_{2} ⁡ n$ depth can compute composition over $n$ group elements, regardless of whether the group is solvable or not. However, the construction in Liu et al. ([2022](https://arxiv.org/html/2502.17416v1#bib.bib34)) uses different attention parameter for each layer. Here, we provide a more parameter efficient construction where we solve this problem by looping a one-layer transformer $log ⁡ \left(\right. n \left.\right)$ times.

###### Theorem 5.1.

For any finite group $G$ and every $n \in \mathbb{N}^{+}$, there exists a constant-precision looped transformer $𝖳𝖥_{\theta , T}$ computing the composition of $n$ elements from $G$ with a $1$-layer transformer block, $T = \lceil log_{2} ⁡ n \rceil$ loops, $G \cup \left{\right. \# \left.\right}$ being the vocabulary, $d = 3 ⁢ \left(\right. \lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil + \lceil log_{2} ⁡ n + 1 \rceil \left.\right)$ embedding size, $d_{𝖥𝖥} = \left(\left|\right. G \left|\right.\right)^{2} + 6 ⁢ \lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil$ hidden dimension in MLP, $d_{𝖠𝖳𝖳𝖭} = \lceil log_{2} ⁡ n \rceil$ hidden attention dimension, and $2$ attention heads. More specifically, for any $g_{1} , \ldots , g_{n} \in G$, $𝖳𝖥_{\theta} ⁢ \left(\right. \# , g_{1} , \ldots , g_{n} \left.\right) = g_{1} \circ ⋯ \circ g_{n}$.

The above result matches the depth upper bound shown for non-looped models by Liu et al. ([2022](https://arxiv.org/html/2502.17416v1#bib.bib34)), and is very close to the super constant lower bound shown there. Thus looped models can solve the problem with best known depth.

### 5.3 Looped models can simulate non-looped models

Our second theoretical result shows that a non-looped transformer with repeated layers can be simulated by a looped transformer with fewer parameters and same depth.

###### Theorem 5.2.

For any transformer $p_{\theta}$ with $L$ layers, $d$ embedding size, $d_{𝖥𝖥}$ hidden dimension for MLP, $H$ attention heads with $d_{𝖠𝖳𝖳𝖭}$ hidden dimension, at most $R$ distinct transformer layers and bounded activation value, there is a looped transformer $p_{\theta^{'} , L}$ working on the same vocabulary $\mathcal{V}$ plus a dummy token $\#$, which loops a 1-layer transformer block for $L$ times, with $d + R + 2$ embedding size, $R ⁢ h_{𝖥𝖥} + O ⁢ \left(\right. L \left.\right)$ hidden dimension for MLP, $R ⁢ H$ attention heads with $d_{𝖠𝖳𝖳𝖭}$ hidden dimension, such that for any string $v_{1} , \ldots , v_{n} \in \mathcal{V}$, $p_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) = p_{\theta^{'} , L} ⁢ \left(\right. \# , v_{1} , \ldots , v_{n} \left.\right)$.

We can also use the above theorem to convert the $O ⁢ \left(\right. log_{2} ⁡ n \left.\right)$ depth transformer that simulates group composition into a 1-layer looped transformer, although it is less parameter efficient than the result from the previous section. Please refer to [Section B.2.1](https://arxiv.org/html/2502.17416v1#A2.SS2.SSS1 "B.2.1 Looped models can simulate non-looped models ‣ B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for the full proof.

Theory for $p$-hop. The experiments on $p$-hop induction from [Section 2.1](https://arxiv.org/html/2502.17416v1#S2.SS1.SSS0.Px1 "𝑝-hop induction. ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") surprisingly show that a small model looped multiple times can solve it very effectively. We establish a theoretical basis for this finding. More specifically, we show that a constant layer transformer with $log ⁡ \left(\right. p \left.\right)$ loops suffices to solve $p$-hop induction problem. This result, in fact, matches the lower bound for layers required for non-looped models proved in Sanford et al. ([2024b](https://arxiv.org/html/2502.17416v1#bib.bib50)). The result follows from [Theorem 5.2](https://arxiv.org/html/2502.17416v1#S5.Thmtheorem2 "Theorem 5.2. ‣ 5.3 Looped models can simulate non-looped models ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") and the construction for non-looped models from Sanford et al. ([2024b](https://arxiv.org/html/2502.17416v1#bib.bib50)) (restated in [Theorem B.12](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem12 "Theorem B.12 (Restatement of Theorem 4.2, (Sanford et al., 2024b)). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")).

###### Corollary 5.3.

$p$-hop problem ([Definition A.1](https://arxiv.org/html/2502.17416v1#A1.Thmdefinition1 "Definition A.1 (𝑝-𝗁𝗈𝗉, Sanford et al. (2024b)). ‣ 𝑝-hop induction. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) can be solved by looping a 1-layer transformer $\lfloor log_{2} ⁡ p \rfloor + 2$ times, which has $O ⁢ \left(\right. log ⁡ n \left.\right)$ bits of precision, $d = d_{𝖥𝖥} = d_{𝖠𝖳𝖳𝖭} = O ⁢ \left(\right. 1 \left.\right)$ embedding size, hidden dimension for MLP and attention, and at most $3$ attention heads.

### 5.4 Looped models can simulate CoT reasoning

In [Section 3.4](https://arxiv.org/html/2502.17416v1#S3.SS4.SSS0.Px1 "Latent thoughts and connections to CoT reasoning. ‣ 3.4 Scaling behavior of looping ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), we discussed how looped models can be viewed as generating multiple latent thoughts in each iteration. The following theorem shows that one can use looped transformer with $L$ loops to simulate $L$ steps CoT of another transformer with similar sizes.

###### Theorem 5.4(Looped transformer simulates CoT).

For any $L$-layer non-looped transformer $𝖳𝖥_{\theta}$ with fixed input length $n$ and the number of CoT steps $m$, there exists a looped transformer $𝖳𝖥_{\theta^{'}}$ with $L + \mathcal{O} ⁢ \left(\right. 1 \left.\right)$ layers, $\Omega ⁢ \left(\right. log ⁡ \left(\right. n + m \left.\right) \left.\right)$, more embedding dimension and constantly many more attention heads, such that for any input $𝒗 = \left(\left(\right. v_{i} \left.\right)\right)_{i = 1}^{n}$, the output of non-looped transformer after $m$ steps of CoT, $𝖳𝖥_{\theta}^{m} ⁢ \left(\right. 𝒗 \left.\right)$, is the same as that of the looped transformer on input $x$ concatenated by $m$ dummy tokens with $m$ loops, $𝖳𝖥_{\theta^{'} , m} ⁢ \left(\right. 𝒗 , \#^{m} \left.\right)$.

Below we sketch the high-level idea behind [Theorem 5.4](https://arxiv.org/html/2502.17416v1#S5.Thmtheorem4 "Theorem 5.4 (Looped transformer simulates CoT). ‣ 5.4 Looped models can simulate CoT reasoning ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"); the full proof can be found in [Section B.3](https://arxiv.org/html/2502.17416v1#A2.SS3 "B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

*   •(Masking all-but-one) We maintain a counter $t$ ([Lemma B.10](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem10 "Lemma B.10. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) at each position for the current number of loop and for the $i$th position, we will only ”update” the embedding if $i - n \geq t$ and reset the embedding to the input to the looped transformer layer otherwise. This can be implemented by MLP. So similar to CoT, the first $n + i - 1$ embedding won’t be changed in the $i$th loop. 
*   •(Shift by one) We can use attention to obtain the output of the current loop at the previous position and use that as the input of the next loop at current position. ([Lemma B.9](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) 
*   •(Token decoding) We show that we can use MLP with ReLU activation to simulate the encoding and decoding process of CoT. ([Lemma B.7](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem7 "Lemma B.7 (Simulating Decoding and Embedding using MLP). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) 

![Image 16: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/cot_iteration.png)

![Image 17: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/latent_thoughts_iteration.png)

Figure 4: Left. Chain-of-thought reasoning can be viewed as a looped model, where each iteration produces one new thoughts token. The new tokens are highlighted in red. Right. A looped model can instead generate multiple latent thoughts in parallel and, in theory, can simulate CoT reasoning my masking the updates appropriately (see [Theorem 5.4](https://arxiv.org/html/2502.17416v1#S5.Thmtheorem4 "Theorem 5.4 (Looped transformer simulates CoT). ‣ 5.4 Looped models can simulate CoT reasoning ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"))

## 6 Related work

Reasoning is recognized as a core ability for intelligent and robustly model and has thus seen significant focus over the last few years. The synthetic reasoning problems we consider in this work have all been used in the prior works of Sanford et al. ([2024b](https://arxiv.org/html/2502.17416v1#bib.bib50)), Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)), Sanford et al. ([2024a](https://arxiv.org/html/2502.17416v1#bib.bib49)), Nogueira et al. ([2021](https://arxiv.org/html/2502.17416v1#bib.bib41)) to theoretically analyze the strengths and limitations of Transformers. There is also interest in the representation power of Transformers for computational problems (Liu et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib34), Strobl et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib56)) and for chain-of-thought reasoning (Merrill & Sabharwal, [2023a](https://arxiv.org/html/2502.17416v1#bib.bib37), Feng et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib15), Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32)). The necessity of model depth for performance has been remarked upon, for small models(Liu et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib36)) and reasoning (Chen & Zou, [2024](https://arxiv.org/html/2502.17416v1#bib.bib6), Ye et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib64), Petty et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib44)). In this work, we make a finer observation that albeit larger depth is necessary, this can be achieved with a limited parameter budget via looping.

Looping in transformer models has been studied since the works (Dehghani et al., [2018](https://arxiv.org/html/2502.17416v1#bib.bib11), Lan et al., [2020](https://arxiv.org/html/2502.17416v1#bib.bib30)) where they showed the benefits of looping for supervised learning tasks and BERT pretraining respectively. Looping also appears in (Schwarzschild et al., [2021](https://arxiv.org/html/2502.17416v1#bib.bib53), Bansal et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib3)) that study the extrapolation properties of looping for certain algorithmic tasks. More recently Giannou et al. ([2023](https://arxiv.org/html/2502.17416v1#bib.bib21)), de Luca & Fountoulakis ([2024](https://arxiv.org/html/2502.17416v1#bib.bib10)) have studied the theoretical properties of looped decoder models and show that looping can simulate arbitrary Turing machines. In addition Yang et al. ([2023](https://arxiv.org/html/2502.17416v1#bib.bib63)), Gao et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib18)), Gatmiry et al. ([2024a](https://arxiv.org/html/2502.17416v1#bib.bib19); [b](https://arxiv.org/html/2502.17416v1#bib.bib20)) study looped models as a way to simulate iterative algorithms for in-context learning. Recently, Mohtashami et al. ([2023](https://arxiv.org/html/2502.17416v1#bib.bib40)) introduced CoTFormer, which tries to improve the perplexity of looped language models and (Liu et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib36)) explore latency-efficiency parameter sharing for on-device LLMs. In contrast, our work focuses on the surprising inductive bias of looping to improve downstream reasoning tasks, and goes beyond algorithmic and in-context learning tasks.

Different training algorithms (e.g. gradient descent (Soudry et al., [2018](https://arxiv.org/html/2502.17416v1#bib.bib55))) and architectural choices (e.g. attention (Edelman et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib13))) have been shown to have certain implicit biases. There is increasing interest in such inductive biases during pretraining (Saunshi et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib51), Liu et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib35)). More recently, Saunshi et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib52)) showed an inductive bias of stacking (Reddi et al., [2023](https://arxiv.org/html/2502.17416v1#bib.bib47)) towards improving reasoning and hypothesize that a connection of stacking to looped models could be responsible for this. Our results provide further verification for this hypothesis.

## 7 Conclusions, limitations and future work

This work explores a new direction of “looped models for reasoning”. Not only are looped models able to solve many reasoning problems with very fewer parameters, they also have an inductive bias towards disproportionately improving the reasoning performance of language models, over memorization. The theoretical results on the expressivity of looped models start to provide some hints into their depth-optimality. While we test looped models on a subset of reasoning problems, a natural question is whether the results hold for many other forms of reasoning (e.g. multimodal and common-sense reasoning). In particular, a succinct formalization of reasoning problems itself is an interesting future direction. Furthermore, the inductive bias towards improved reasoning performance at the same perplexity is very intriguing and deserves further exploration. We find the scaling behavior of looped models very fascinating, and the connections to latent thoughts and CoT reasoning start to provide hints into this behavior. We hope this inspires future exploration on using looped models for more efficient inference-time scaling to aid with deeper reasoning.

## References

*   Allen-Zhu & Li (2024) Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws. _arXiv preprint arXiv:2404.05405_, 2024. 
*   Bai et al. (2019) Shaojie Bai, J Zico Kolter, and Vladlen Koltun. Deep equilibrium models. _Advances in neural information processing systems_, 2019. 
*   Bansal et al. (2022) Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, and Tom Goldstein. End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking. _Advances in Neural Information Processing Systems_, 2022. 
*   Barrington (1986) David A. Barrington. Bounded-width polynomial-size branching programs recognize exactly those languages in nc. pp. 1–5, 1986. 
*   Brown et al. (2020) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_, 2020. 
*   Chen & Zou (2024) Xingwu Chen and Difan Zou. What can transformer learn with varying depth? case studies on sequence learning tasks. _arXiv preprint arXiv:2404.01601_, 2024. 
*   Cho et al. (2024) Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, and Chulhee Yun. Position coupling: Leveraging task structure for improved length generalization of transformers. _arXiv preprint arXiv:2405.20671_, 2024. 
*   Choi et al. (2018) Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. QuAC: Question answering in context. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_, 2018. 
*   Clark et al. (2020) Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. _Transactions of the Association for Computational Linguistics_, 2020. 
*   de Luca & Fountoulakis (2024) Artur Back de Luca and Kimon Fountoulakis. Simulation of graph algorithms with looped transformers. _arXiv preprint arXiv:2402.01107_, 2024. 
*   Dehghani et al. (2018) Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In _International Conference on Learning Representations_, 2018. 
*   Dua et al. (2019) Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In _Proc. of NAACL_, 2019. 
*   Edelman et al. (2022) Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms. In _International Conference on Machine Learning_, pp. 5793–5831. PMLR, 2022. 
*   Elhage et al. (2021) Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. _Transformer Circuits Thread_, 1:1, 2021. 
*   Feng et al. (2023) Guhao Feng, Yuntian Gu, Bohang Zhang, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. _arXiv preprint arXiv:2305.15408_, 2023. 
*   Feng et al. (2024) Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Gao et al. (2020) Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_, 2020. 
*   Gao et al. (2024) Yihang Gao, Chuanyang Zheng, Enze Xie, Han Shi, Tianyang Hu, Yu Li, Michael K Ng, Zhenguo Li, and Zhaoqiang Liu. On the expressive power of a variant of the looped transformer. _arXiv preprint arXiv:2402.13572_, 2024. 
*   Gatmiry et al. (2024a) Khashayar Gatmiry, Nikunj Saunshi, Sashank J Reddi, Stefanie Jegelka, and Sanjiv Kumar. Can looped transformers learn to implement multi-step gradient descent for in-context learning? In _Forty-first International Conference on Machine Learning_, 2024a. 
*   Gatmiry et al. (2024b) Khashayar Gatmiry, Nikunj Saunshi, Sashank J Reddi, Stefanie Jegelka, and Sanjiv Kumar. On the role of depth and looping for in-context learning with task diversity. _arXiv preprint arXiv:2410.21698_, 2024b. 
*   Giannou et al. (2023) Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In _International Conference on Machine Learning_. PMLR, 2023. 
*   Gong et al. (2019) Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of BERT by progressively stacking. In _Proceedings of the 36th International Conference on Machine Learning_, 2019. 
*   Guo et al. (2025) Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 
*   Hoffmann et al. (2022) Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. _arXiv preprint arXiv:2203.15556_, 2022. 
*   Joshi et al. (2017) Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, 2017. 
*   Kaplan et al. (2020) Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_, 2020. 
*   Koncel-Kedziorski et al. (2016) Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In _Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies_, 2016. 
*   Krohn & Rhodes (1965) Kenneth Krohn and John Rhodes. Algebraic theory of machines. i. prime decomposition theorem for finite semigroups and machines. _Transactions of the American Mathematical Society_, 116, 1965. 
*   Kwiatkowski et al. (2019) Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. _Transactions of the Association for Computational Linguistics_, 2019. 
*   Lan et al. (2020) Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In _International Conference on Learning Representations_, 2020. 
*   Lee et al. (2024) Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee, and Dimitris Papailiopoulos. Teaching arithmetic to small transformers. In _The Twelfth International Conference on Learning Representations_, 2024. 
*   Li et al. (2024) Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=3EWTEy9MTM](https://openreview.net/forum?id=3EWTEy9MTM). 
*   Liang et al. (2023) Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. _Transactions on Machine Learning Research_, 2023. 
*   Liu et al. (2022) Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. _arXiv preprint arXiv:2210.10749_, 2022. 
*   Liu et al. (2023) Hong Liu, Sang Michael Xie, Zhiyuan Li, and Tengyu Ma. Same pre-training loss, better downstream: Implicit bias matters for language models. In _International Conference on Machine Learning_. PMLR, 2023. 
*   Liu et al. (2024) Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. _arXiv preprint arXiv:2402.14905_, 2024. 
*   Merrill & Sabharwal (2023a) William Merrill and Ashish Sabharwal. The expresssive power of transformers with chain of thought. _arXiv preprint arXiv:2310.07923_, 2023a. 
*   Merrill & Sabharwal (2023b) William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. _Transactions of the Association for Computational Linguistics_, 11:531–545, 2023b. 
*   Miao et al. (2020) Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, pp. 975–984, 2020. 
*   Mohtashami et al. (2023) Amirkeivan Mohtashami, Matteo Pagliardini, and Martin Jaggi. Cotformer: More tokens with attention make up for less depth. _arXiv preprint arXiv:2310.10845_, 2023. 
*   Nogueira et al. (2021) Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. Investigating the limitations of transformers with simple arithmetic tasks. _arXiv preprint arXiv:2102.13019_, 2021. 
*   Nye et al. (2021) Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_, 2021. 
*   Patel et al. (2021) Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2021. 
*   Petty et al. (2023) Jackson Petty, Sjoerd van Steenkiste, Ishita Dasgupta, Fei Sha, Dan Garrette, and Tal Linzen. The impact of depth and width on transformer language model generalization. _arXiv preprint arXiv:2310.19956_, 2023. 
*   Radford et al. (2019) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_, 2019. 
*   Rajpurkar et al. (2018) Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for SQuAD. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_, Melbourne, Australia, 2018. 
*   Reddi et al. (2023) Sashank Reddi, Sobhan Miryoosefi, Stefani Karp, Shankar Krishnan, Satyen Kale, Seungyeon Kim, and Sanjiv Kumar. Efficient training of language models using few-shot learning. In _Proceedings of the 40th International Conference on Machine Learning_, 2023. 
*   Reddy et al. (2019) Siva Reddy, Danqi Chen, and Christopher D. Manning. CoQA: A conversational question answering challenge. _Transactions of the Association for Computational Linguistics_, 2019. 
*   Sanford et al. (2024a) Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph algorithms. _arXiv preprint arXiv:2405.18512_, 2024a. 
*   Sanford et al. (2024b) Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. _arXiv preprint arXiv:2402.09268_, 2024b. 
*   Saunshi et al. (2022) Nikunj Saunshi, Jordan Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham Kakade, and Akshay Krishnamurthy. Understanding contrastive learning requires incorporating inductive biases. In _Proceedings of the 39th International Conference on Machine Learning_, 2022. 
*   Saunshi et al. (2024) Nikunj Saunshi, Stefani Karp, Shankar Krishnan, Sobhan Miryoosefi, Sashank J. Reddi, and Sanjiv Kumar. On the inductive bias of stacking towards improving reasoning. _arXiv preprint arXiv:2409.19044_, 2024. 
*   Schwarzschild et al. (2021) Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, and Tom Goldstein. Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks. _Advances in Neural Information Processing Systems_, 2021. 
*   Shazeer & Stern (2018) Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In _International Conference on Machine Learning_, pp. 4596–4604. PMLR, 2018. 
*   Soudry et al. (2018) Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. _Journal of Machine Learning Research_, 2018. 
*   Strobl et al. (2023) Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. Transformers as recognizers of formal languages: A survey on expressivity. _arXiv preprint arXiv:2311.00208_, 2023. 
*   Sun et al. (2023) Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, et al. A survey of reasoning with foundation models. _arXiv preprint arXiv:2312.11562_, 2023. 
*   Talmor & Berant (2018) Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_, 2018. 
*   Tay et al. (2022) Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara Bahri, Tal Schuster, Steven Zheng, et al. Ul2: Unifying language learning paradigms. In _The Eleventh International Conference on Learning Representations_, 2022. 
*   Team et al. (2023) Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_, 2023. 
*   Wei et al. (2022a) Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_, 2022a. 
*   Wei et al. (2022b) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_, 2022b. 
*   Yang et al. (2023) Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. _arXiv preprint arXiv:2311.12424_, 2023. 
*   Ye et al. (2024) Tian Ye, Zicheng Xu, Yuanzhi Li, and Zeyuan Allen-Zhu. Physics of language models: Part 2.1, grade-school math and the hidden reasoning process. _arXiv preprint arXiv:2407.20311_, 2024. 

## Appendix A Experiments

### A.1 Simple reasoning setup details

##### $n$-ary addition.

All experiments are run on a standard Transformer architecture with input dimension of 256, 8 heads and 1024 hidden dimension in the feed-forward layers. We train using Adafactor (Shazeer & Stern, [2018](https://arxiv.org/html/2502.17416v1#bib.bib54)) employing a linear warmup coupled with a cosine decay schedule for the learning rate. All runs use a batch size of 1024, learning rate of 0.005 and run for 200k steps. This corresponds to 200M examples, which is insignificant compared to the total possible examples ($> 10^{320}$). Thus memorization of the answer is not an issue. Since training is a bit noisy, for each setting, we run 3 different random seeds and pick the run with the maximum average accuracy. We pick maximum instead of average because we care about expressivity power of these models.

##### $p$-hop induction.

We formally define the $p$-hop problem below:

###### Definition A.1($p$-$𝗁𝗈𝗉$,Sanford et al. ([2024b](https://arxiv.org/html/2502.17416v1#bib.bib50))).

For a finite alphabet $\Sigma$, define the map $𝗁𝗈𝗉_{p} : \Sigma^{n} \rightarrow \Sigma \cup \left{\right. ⟂ \left.\right}$ as $𝗁𝗈𝗉_{p} ⁢ \left(\right. 𝒗 \left.\right) = v_{𝖿𝗂𝗇𝖽_{p} ⁢ \left(\right. 𝒗 , n \left.\right)}$ if $𝖿𝗂𝗇𝖽_{p} ⁢ \left(\right. 𝒗 , n \left.\right) \neq 0$ else $⟂$, where

$𝖿𝗂𝗇𝖽_{1} ⁢ \left(\right. 𝒗 , i \left.\right)$$= max ⁡ \left(\right. \left{\right. 0 \left.\right} \cup \left{\right. j \leq i , v_{j - 1} = v_{i} \left.\right} \left.\right)$
$𝖿𝗂𝗇𝖽_{p} ⁢ \left(\right. 𝒗 , i \left.\right)$$= 𝖿𝗂𝗇𝖽_{1} ⁢ \left(\right. 𝒗 , 𝖿𝗂𝗇𝖽_{p - 1} ⁢ \left(\right. 𝒗 , i \left.\right) \left.\right) ⁢ \textrm{ }\text{for}\textrm{ } ⁢ p \geq 2 . \textrm{ }\text{for}\textrm{ }$

![Image 18: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/Heatmap.Model_Pile_CosReg10_blk4.VariableKind_self_attention.post.w.png)

(a) Attn:Q

![Image 19: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/Heatmap.Model_Pile_CosReg10_blk4.VariableKind_ff_layer.ffn_layer1.linear.w.png)

(b) FFN:W1

![Image 20: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/Heatmap.Model_Pile_CosReg10_blk4.VariableKind_self_attention.post.w.png)

(c) Attn:PostNorm

Figure 5: Cosine similarities for different layers in the model trained with the regularization strength $\lambda_{\text{reg}} = 10 \text{reg}$ for block size $k = 4$ (see [Section 4](https://arxiv.org/html/2502.17416v1#S4 "4 Looping-inspired regularization ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for details). The 24 layer model will have 6 such blocks of size 4. The heatmap above shows the cosine similarities between weights for all pairs of blocks. Overall we find the final cosine similarity to be very high, thus suggesting a strong connection to looped models.

![Image 21: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/Heatmap.Model_Pile_Baseline_blk4.VariableKind_self_attention.post.w.png)

(a) Attn:Q

![Image 22: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/Heatmap.Model_Pile_Baseline_blk4.VariableKind_ff_layer.ffn_layer1.linear.w.png)

(b) FFN:W1

![Image 23: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/Heatmap.Model_Pile_Baseline_blk4.VariableKind_self_attention.post.w.png)

(c) Attn:PostNorm

Figure 6: Cosine similarities for different layers in the baseline model trained without any regularization strength. Overall the cosine similarities are very low for large matrices, as expected for high dimensions matrices.

For the $p$-hop problem we sample instances randomly while enforcing that there always exists $p$-hops present in the input sequence. We do this by first picking the sequence of $p$-hops randomly and then shuffling them around in a sequence with filler tokens to be filled by the remaining characters. After the shuffle, we sample the remaining characters to occur in place of the filler tokens while respecting the $p$-hop order. Our train set consists of 4M examples and our test and validation sets consist of 262k examples each. For all models we train on this dataset, the model dimension is 128, hidden dimension is 512 and 8 attention heads are used. Rotary positional encodings are used as well. We train using Adafactor for 200,000 steps with a batch size of 256 using a base learning rate of $10^{- 3}$ and use a linear warmup coupled with a cosine decay schedule for the learning rate.

##### i-GSM.

We describe how the i-GSM dataset is generated in more detail here. We start with a hierarchy of entities of depth 4 from which we build a randomly sampled structure graph where directed edges connect entities in level $i$ to those in level $i + 1$. Each edge in the structure graph defines a instance parameter which is an integer (for e.g. an edge between city center and car parks denotes the number of car parks in city center). We then construct a randomly sampled mathematical dependency graph which is a DAG over all the instance parameters by relating each to the others. Finally we pick one of the nodes of the dependency graph to query and the goal is to compute the numerical value of this node modulo some prime number $P$. For more details on the sampling process for the structure and dependency graphs, we refer the reader to Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)). We make 3 simplifications compared to Ye et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib64)): we phrase our problems in a symbolic language without the English construction of sentences (see [Table 2](https://arxiv.org/html/2502.17416v1#S2.T2 "In i-GSM (Synthetic Grade School Math Problems). ‣ 2.1 Experiments with simple reasoning problems ‣ 2 Looped models on simple reasoning tasks ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")); we do not allow abstract parameters in our problems; we perform arithmetic modulo $7$ as opposed to $23$. Our train dataset consists of around 4 million examples and we test on around 50k examples. Given the significantly larger number of unique solution templates possible, train-test overlap in the problem template space is going to be limited with high probability. For all models we train on this dataset, the model dimension is 128, hidden dimension is 512 and 8 attention heads are used. Rotary positional encodings are used as well. We train using Adafactor for 200,000 steps with a batch size of 256 using a base learning rate of $10^{- 3}$ and use a linear warmup coupled with a cosine decay schedule for the learning rate.

![Image 24: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/8L_closedQA.png)

![Image 25: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/8L_openQA.png)

![Image 26: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/8L_mwp.png)

![Image 27: Refer to caption](https://arxiv.org/html/extracted/6229618/Media/8L_primitives.png)

Figure 7: Downstream evaluation for various task groups on the x-axis, vs validation log perplexity on the y-axis (reversed), as training proceeds. The top plots compare a 8-layer baseline model $\left(\right. 8 \bigotimes 1 \left.\right)$ and the looped model $\left(\right. 8 \bigotimes 3 \left.\right)$. Similarly to [Figure 2](https://arxiv.org/html/2502.17416v1#S3.F2 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), for closed book QA (memorization) tasks looping has very similar trends to baseline. For open book QA tasks and math word problems, looping has much better downstream performance at an equivalent log perplexity.

### A.2 Language modeling setup

We train on the Pile data using causal language modeling objective. The dataset is pre-processed and cached to ensure that all models are trained on exactly the same tokens in the same order. For all experiments, we use a batch size of 512 and sequence length of 1280. We use a cosine learning rate schedule decaying over 400k steps with a peak learning rate of 0.01, tuned based on the baseline model. The base model is a 1.5B parameter decoder only Transformer model, with 24 layers, model dimensions of 2048, hidden dimension 5120 and 32 heads. For the shallower baselines and looped models, we only change the number of layers and keep all other hyperparameters the same.

### A.3 Results for each task group

In [Section 3](https://arxiv.org/html/2502.17416v1#S3 "3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") we discussed results for various task groups like closed book QA, math word problems etc. Here we present results for each individual task for completeness. Detailed results for closed book QA are in [Table 5](https://arxiv.org/html/2502.17416v1#A1.T5 "In A.3 Results for each task group ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), open book QA in [Table 6](https://arxiv.org/html/2502.17416v1#A1.T6 "In A.3 Results for each task group ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), math word problems in [Table 7](https://arxiv.org/html/2502.17416v1#A1.T7 "In A.3 Results for each task group ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") and reasoning primitives in [Table 8](https://arxiv.org/html/2502.17416v1#A1.T8 "In A.3 Results for each task group ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). These tables include, both, looped models from [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") and the models trained with regularization from [Table 4](https://arxiv.org/html/2502.17416v1#S4.T4 "In 4 Looping-inspired regularization ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

Table 5: Downstream evaluations for the models in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for closed book QA tasks.

|  | Params / FLOPs | TriviaQA | TydiQA- | Natural | Web | Average |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  | NoContext | Questions | Questions | Average |
| Baseline | 24x / 24x | 24.5 | 10.6 | 4.3 | 5.6 | 11.2 |
| Base $\left(\right. 12 \bigotimes 1 \left.\right)$ | 12x / 12x | 15.9 | 9.3 | 2.4 | 5.2 | 8.2 |
| Loop $\left(\right. 12 \bigotimes 2 \left.\right)$ | 12x / 24x | 18.6 | 9.6 | 3.4 | 5.8 | 9.3 |
| Regularized $\left(\right. k = 12 \left.\right)$ | 24x / 24x | 20.9 | 10.9 | 3.6 | 5.0 | 10.1 |
| Base $\left(\right. 8 \bigotimes 1 \left.\right)$ | 8x / 8x | 12.3 | 7.4 | 1.8 | 3.9 | 6.3 |
| Loop $\left(\right. 8 \bigotimes 3 \left.\right)$ | 8x / 24x | 17.3 | 8.8 | 2.5 | 5.2 | 8.5 |
| Regularized $\left(\right. k = 8 \left.\right)$ | 24x / 24x | 23.6 | 10.6 | 4.1 | 6.7 | 11.3 |
| Base $\left(\right. 6 \bigotimes 1 \left.\right)$ | 6x / 6x | 7.4 | 4.8 | 1.1 | 2.7 | 4.0 |
| Loop $\left(\right. 6 \bigotimes 4 \left.\right)$ | 6x / 24x | 16.0 | 9.3 | 2.7 | 4.9 | 8.2 |
| Regularized $\left(\right. k = 6 \left.\right)$ | 24x / 24x | 25.8 | 12.2 | 4.5 | 5.6 | 12.0 |
| Base $\left(\right. 4 \bigotimes 1 \left.\right)$ | 4x / 4x | 3.3 | 1.9 | 0.6 | 1.5 | 1.8 |
| Loop $\left(\right. 4 \bigotimes 6 \left.\right)$ | 4x / 24x | 11.8 | 8.8 | 2.4 | 3.5 | 6.7 |
| Regularized $\left(\right. k = 4 \left.\right)$ | 24x / 24x | 26.1 | 13.3 | 4.5 | 6.2 | 12.5 |

Table 6: Downstream evaluations for the models in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for open book QA tasks.

|  | Params / FLOPs | TydiQA-NoContext | SquadV2 | DROP | QuAC | CoQA | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 24x / 24x | 33.4 | 41.3 | 23.4 | 18.6 | 52.7 | 33.9 |
| Base $\left(\right. 12 \bigotimes 1 \left.\right)$ | 12x / 12x | 21.8 | 34.6 | 20.0 | 17.3 | 41.1 | 26.9 |
| Loop $\left(\right. 12 \bigotimes 2 \left.\right)$ | 12x / 24x | 27.7 | 39.5 | 22.3 | 17.6 | 46.8 | 30.8 |
| Regularized $\left(\right. k = 12 \left.\right)$ | 24x / 24x | 33.0 | 44.3 | 23.8 | 17.7 | 51.9 | 34.1 |
| Base $\left(\right. 8 \bigotimes 1 \left.\right)$ | 8x / 8x | 12.7 | 33.8 | 16.0 | 15.3 | 35.9 | 22.7 |
| Loop $\left(\right. 8 \bigotimes 3 \left.\right)$ | 8x / 24x | 29.8 | 38.1 | 21.6 | 17.6 | 46.6 | 30.8 |
| Regularized $\left(\right. k = 8 \left.\right)$ | 24x / 24x | 33.0 | 41.7 | 25.2 | 19.3 | 52.9 | 34.4 |
| Base $\left(\right. 6 \bigotimes 1 \left.\right)$ | 6x / 6x | 8.2 | 26.8 | 14.8 | 14.4 | 32.5 | 19.3 |
| Loop $\left(\right. 6 \bigotimes 4 \left.\right)$ | 6x / 24x | 26.6 | 34.6 | 20.8 | 18.1 | 43.7 | 28.7 |
| Regularized $\left(\right. k = 6 \left.\right)$ | 24x / 24x | 34.5 | 46.1 | 26.2 | 18.6 | 53.4 | 35.8 |
| Base $\left(\right. 4 \bigotimes 1 \left.\right)$ | 4x / 4x | 3.4 | 19.0 | 11.1 | 13.1 | 22.3 | 13.8 |
| Loop $\left(\right. 4 \bigotimes 6 \left.\right)$ | 4x / 24x | 22.0 | 32.4 | 20.1 | 15.8 | 40.7 | 26.2 |
| Regularized $\left(\right. k = 4 \left.\right)$ | 24x / 24x | 33.6 | 49.4 | 24.1 | 19.8 | 54.0 | 36.2 |

Table 7: Downstream evaluations for the models in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for math word problems.

|  | Params / FLOPs | ASDiv | MAWPS- | MAWPS- | MAWPS- | MAWPS- | SVAMP | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | AddSub | MultiArith | SingleEq | SingleOp |  |  |
| Baseline | 24x / 24x | 26.9 | 49.9 | 2.7 | 38.6 | 40.2 | 17.9 | 29.3 |
| Base $\left(\right. 12 \bigotimes 1 \left.\right)$ | 12x / 12x | 23.1 | 44.8 | 2.0 | 36.8 | 37.9 | 15.3 | 26.7 |
| Loop $\left(\right. 12 \bigotimes 2 \left.\right)$ | 12x / 24x | 31.5 | 55.9 | 2.0 | 47.4 | 50.2 | 18.5 | 34.3 |
| Regularized $\left(\right. k = 12 \left.\right)$ | 24x / 24x | 27.4 | 54.4 | 2.7 | 43.7 | 45.9 | 19.8 | 32.3 |
| Base $\left(\right. 8 \bigotimes 1 \left.\right)$ | 8x / 8x | 12.9 | 32.7 | 2.0 | 19.9 | 21.9 | 13.4 | 17.1 |
| Loop $\left(\right. 8 \bigotimes 3 \left.\right)$ | 8x / 24x | 23.2 | 54.9 | 2.3 | 36.2 | 38.3 | 15.6 | 28.4 |
| Regularized $\left(\right. k = 8 \left.\right)$ | 24x / 24x | 31.0 | 56.7 | 3.5 | 40.7 | 47.5 | 17.3 | 32.8 |
| Base $\left(\right. 6 \bigotimes 1 \left.\right)$ | 6x / 6x | 15.4 | 31.1 | 1.3 | 21.3 | 26.3 | 10.9 | 17.7 |
| Loop $\left(\right. 6 \bigotimes 4 \left.\right)$ | 6x / 24x | 24.5 | 51.6 | 0.7 | 38.0 | 47.7 | 16.3 | 29.8 |
| Regularized $\left(\right. k = 6 \left.\right)$ | 24x / 24x | 32.0 | 47.1 | 3.5 | 44.9 | 42.0 | 16.8 | 31.0 |
| Base $\left(\right. 4 \bigotimes 1 \left.\right)$ | 4x / 4x | 7.9 | 10.4 | 1.5 | 11.0 | 19.0 | 8.3 | 9.7 |
| Loop $\left(\right. 4 \bigotimes 6 \left.\right)$ | 4x / 24x | 19.9 | 49.1 | 1.7 | 29.3 | 35.9 | 12.6 | 24.8 |
| Regularized $\left(\right. k = 4 \left.\right)$ | 24x / 24x | 35.0 | 53.9 | 3.2 | 52.2 | 50.9 | 23.0 | 36.4 |

Table 8: Downstream evaluations for the models in [Table 3](https://arxiv.org/html/2502.17416v1#S3.T3 "In 3.1 Experiments with 1B language modeling ‣ 3 Language modeling with looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") for reasoning primitives.

|  | Params / FLOPs | Code | Math | Code | Math | Average |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | Depth-0 | Depth-0 | Depth-1 | Depth-1 |  |
| Baseline | 24x / 24x | 71.1 | 72.5 | 24.2 | 22.3 | 47.5 |
| Base $\left(\right. 12 \bigotimes 1 \left.\right)$ | 12x / 12x | 52.2 | 51.6 | 20.7 | 18.3 | 35.7 |
| Loop $\left(\right. 12 \bigotimes 2 \left.\right)$ | 12x / 24x | 73.5 | 86.5 | 21.3 | 23.6 | 51.2 |
| Regularized $\left(\right. k = 12 \left.\right)$ | 24x / 24x | 75.1 | 74.9 | 27.0 | 25.7 | 50.7 |
| Base $\left(\right. 8 \bigotimes 1 \left.\right)$ | 8x / 8x | 48.7 | 42.0 | 21.4 | 19.8 | 33.0 |
| Loop $\left(\right. 8 \bigotimes 3 \left.\right)$ | 8x / 24x | 88.4 | 86.9 | 23.1 | 22.9 | 55.3 |
| Regularized $\left(\right. k = 8 \left.\right)$ | 24x / 24x | 92.2 | 86.7 | 23.1 | 23.3 | 56.3 |
| Base $\left(\right. 6 \bigotimes 1 \left.\right)$ | 6x / 6x | 26.3 | 29.7 | 20.2 | 20.1 | 24.1 |
| Loop $\left(\right. 6 \bigotimes 4 \left.\right)$ | 6x / 24x | 90.6 | 88.1 | 24.1 | 21.7 | 56.1 |
| Regularized $\left(\right. k = 6 \left.\right)$ | 24x / 24x | 84.0 | 79.7 | 31.4 | 28.3 | 55.8 |
| Base $\left(\right. 4 \bigotimes 1 \left.\right)$ | 4x / 4x | 19.3 | 23.3 | 17.3 | 17.6 | 19.4 |
| Loop $\left(\right. 4 \bigotimes 6 \left.\right)$ | 4x / 24x | 87.9 | 90.0 | 24.8 | 24.8 | 56.9 |
| Regularized $\left(\right. k = 4 \left.\right)$ | 24x / 24x | 88.0 | 86.9 | 27.9 | 25.8 | 57.2 |

## Appendix B Theoretical results

### B.1 Detailed notations

###### Definition B.1(Embedding Layer).

Given a finite vocabulary $\mathcal{V}$, embedding dimension $d \in \mathbb{N}^{+}$, token embedding parameter $\theta_{𝖳𝖤} \in \mathbb{R}^{d \times \left|\right. \mathcal{V} \left|\right.}$ and position embedding parameter $\theta_{𝖯𝖤} \in \mathbb{R}^{d \times n_{max}}$, we define the _embedding layer_ as a sequence-to-sequence map, denoted by $𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖳𝖤} , \theta_{𝖯𝖤}} : \mathcal{V}^{n} \rightarrow \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n}$ for any $1 \leq n \leq n_{max}$, where

$𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖳𝖤} , \theta_{𝖯𝖤}} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) = \left(\right. \theta_{𝖳𝖤} ⁢ \left(\right. v_{1} \left.\right) + \theta_{𝖯𝖤} ⁢ \left(\right. 1 \left.\right) , \ldots , \theta_{𝖳𝖤} ⁢ \left(\right. v_{n} \left.\right) + \theta_{𝖯𝖤} ⁢ \left(\right. n \left.\right) \left.\right) .$(7)

##### Multi-Head Self-Attention Mechanism:

Given attention parameters $\theta_{𝖠𝖳𝖳𝖭} = \left{\right. W_{Q} , W_{K} , W_{V} , W_{O} \left.\right}$, where each $W_{Q}^{m} , W_{K}^{m} , W_{V}^{m} , W_{O}^{m} \in \mathbb{R}^{d_{𝖠𝖳𝖳𝖭} \times d}$, we define the _Self-Attention_ layer with a causal mask for a decoder-only transformer in [Algorithm 1](https://arxiv.org/html/2502.17416v1#alg1 "In Multi-Head Self-Attention Mechanism: ‣ B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). We also define a _Multi-Head Attention_ layer as a collection of self-attention layer with non-shared parameters $\theta_{𝖬𝖧𝖠} = \left(\left{\right. \theta_{𝖠𝖳𝖳𝖭}^{\left(\right. h \left.\right)} \left.\right}\right)_{h = 1}^{H}$, and its output is the sum of the outputs from each head. That is, $𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}} = \sum_{h = 1}^{H} 𝖠𝖳𝖳𝖭_{\theta_{𝖠𝖳𝖳𝖭}^{\left(\right. h \left.\right)}}$.2 2 2 Though in this paper we focus on attention with casual mask, our definition of looped transformer generalizes to the cases with other attention masks.

Algorithm 1 Causal Self-Attention, $𝖠𝖳𝖳𝖭$

1:Parameter $\theta_{𝖠𝖳𝖳𝖭} = \left(\right. W_{Q} , W_{K} , W_{V} , W_{O} \left.\right)$, Input embedding $x_{1} , \ldots , x_{n} \left.\right) \in \left(\right. \mathbb{R}^{d} \left.\right) ^{n}$. 

2:Output embedding $x^{'} = \left(\right. x_{1}^{'} , \ldots , x_{n}^{'} \left.\right) \triangleq 𝖠𝖳𝖳𝖭_{\theta_{𝖠𝖳𝖳𝖭}} ⁢ \left(\right. x_{1} , \ldots , x_{n} \left.\right)$. 

3:$q_{i} \triangleq W_{Q} ⁢ x_{i} , k_{i} \triangleq W_{K} ⁢ x_{i} , v_{i} \triangleq W_{V} ⁢ x_{i} , \forall i \in \left[\right. n \left]\right.$

4:$s_{i} \triangleq softmax ⁢ \left(\right. \langle q_{i} , k_{1} \rangle , \ldots , \langle q_{i} , k_{i} \rangle \left.\right) \parallel \left(\right. 0 , \ldots , 0 \left.\right)$. 

5:$h_{i}^{'} \triangleq W_{O}^{\top} ⁢ \sum_{j = 1}^{n} \left(\left(\right. s_{i} \left.\right)\right)_{j} ⁢ v_{j}$. 

##### Feed-Forward Network:

Given the parameters of the fully-connected feedforward network layer $\theta_{𝖥𝖥} = \left(\right. W_{1} , b_{1} , W_{2} , b_{2} \left.\right) \in \mathbb{R}^{x_{𝖥𝖥} \times d} \times \mathbb{R}^{d_{𝖥𝖥}} \times \mathbb{R}^{d \times d_{𝖥𝖥}} \times \mathbb{R}^{d_{𝖥𝖥}}$, we define the feedforward layer $𝖥𝖥_{\theta_{𝖥𝖥}} : \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$ as $𝖥𝖥_{\theta_{𝖥𝖥}} ⁢ \left(\right. h \left.\right) \triangleq W_{2} , 𝗋𝖾𝗅𝗎 ⁢ \left(\right. W_{1} ⁢ h + b_{1} \left.\right) + b_{2}$.

###### Definition B.2(Output Layer).

Given parameter $\theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \in \mathbb{R}^{d \times \left|\right. \mathcal{V} \left|\right.}$, we denote the output layer as $𝖮𝖴𝖳𝖯𝖴𝖳_{\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}} : \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n} \rightarrow \Delta^{\left|\right. \mathcal{V} \left|\right. - 1}$, where

$𝖮𝖴𝖳𝖯𝖴𝖳_{\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}} ⁢ \left(\right. x_{1} , \ldots , x_{n} \left.\right) \triangleq softmax ⁢ \left(\right. x_{n}^{\top} ⁢ \theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \left.\right)$(8)

Finally, we define the entire transformer model $p_{\theta} : \cup_{n \leq n_{max}} \mathcal{V}^{n} \rightarrow \Delta^{\left|\right. \mathcal{V} \left|\right. - 1}$ as

$p_{\theta} \triangleq 𝖮𝖴𝖳𝖯𝖴𝖳_{\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}} \circ 𝖳𝖡_{\theta_{𝖳𝖡}} \circ 𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖳𝖤} , \theta_{𝖯𝖤}}$(9)

for any $\theta = \left(\right. \theta_{𝖳𝖡} , \theta_{𝖳𝖤} , \theta_{𝖯𝖤} , \theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \left.\right)$.For convenience, we also write $\left[\right. p_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \left]\right. ⁢ \left(\right. v \left.\right)$ as $p_{\theta} ⁢ \left(\right. v \mid v_{1} , \ldots , v_{n} \left.\right)$. In particular, we use $𝖳𝖥_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \triangleq \left(arg ⁢ max\right)_{v \in \mathcal{V}} ⁡ p_{\theta} ⁢ \left(\right. v \left|\right. v_{1} , \ldots , v_{n} \left.\right)$ to denote the deterministic version of the transformer model.

##### Finite-precision Modeling:

In this paper we assume the transformer is of finite precision. More specifically, we follow the setting in Li et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib32)) and use the shorthand $\mathbb{F}_{s} \triangleq \left{\right. c \cdot k \cdot 2^{- s} \mid c \in \left{\right. - 1 , 1 \left.\right} , 0 \leq k \leq 2^{2 ⁢ s} - 1 , k \in \mathbb{N} \left.\right}$ to denote fixed-point numbers of constant precision $s$ and rounding operation $\left(\left[\right. \cdot \left]\right.\right)_{s} : \mathbb{R} \rightarrow \mathbb{F}_{s}$ to denote the correcting rounding, namely the mapping from $\mathbb{R}$ to the closest representable number in $\mathbb{F}_{s}$. (We break the tie by picking the number with smaller absolute value). We assume that (1). all the parameters of the transformer are in $\mathbb{F}_{s}$ and (2). correct rounding is performed after every binary operation in the forward pass of the transformer. We will refer the readers to Li et al. ([2024](https://arxiv.org/html/2502.17416v1#bib.bib32)) for detailed discussion on such finite-precision modeling and only list important notations and lemmas that will be used in this paper below.

We use $1_{s}$ to denote all-one vectors of length $s$. Similarly we define $\left(\langle \cdot , \cdot \rangle\right)_{s}$, $\times_{s}$, and $softmax_{s}$. We recall that for any $s \in \mathbb{N}^{+}$ and integer $0 \leq x \leq 2^{s} - 1$, we use $𝖻𝗂𝗇_{s} ⁢ \left(\right. x \left.\right) \in \left(\left{\right. 0 , 1 \left.\right}\right)^{s}$ to denote the usual binary encoding of integer $x$ using $s$ binary bits in the sense that $x = \sum_{i = 1}^{s} 2^{i} ⁢ \left(\left(\right. 𝖻𝗂𝗇_{s} ⁢ \left(\right. x \left.\right) \left.\right)\right)_{i}$ and $𝗌𝖻𝗂𝗇_{s} ⁢ \left(\right. x \left.\right) \in \left(\left{\right. - 1 , 1 \left.\right}\right)^{s}$ to denote the signed binary encoding, which is $2 ⁢ 𝖻𝗂𝗇_{s} ⁢ \left(\right. x \left.\right) - \left(\right. 1 , \ldots , 1 \left.\right)$. Finally we define $B_{s} = max ⁡ \mathbb{F}_{s} = 2^{s} - 2^{- s}$.

###### Lemma B.1.

[Lemma E.1, (Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32))] For any $s \in \mathbb{N}^{+}$, it holds that $\left(\left[\right. exp ⁡ \left(\right. - B_{s} \left.\right) \left]\right.\right)_{s} = 0$.

###### Lemma B.2.

[Lemma E.2, (Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32))] For any $s \in \mathbb{N}^{+}$, it holds that $\left(\left[\right. exp ⁡ \left(\right. B_{s} \left.\right) \left]\right.\right)_{s} = B_{s}$.

###### Lemma B.3.

[Lemma E.5, (Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32))] Unlimited-fanin $𝖠𝖭𝖣 , 𝖮𝖱$ (resp. $𝖬𝖠𝖩𝖮𝖱𝖨𝖳𝖸 \left.\right) : \left{\right. 0 , 1 \left.\right} ^{n} \rightarrow \left{\right. 0 , 1 \left.\right}$ can be simulated by some 2-layer feedforward ReLU network with constant (resp. $log ⁡ n$) bits of precision constant hidden dimension and additional $n$ constant inputs of value 1.

Mathematically, let $𝖥𝖥 ⁢ \left[\right. s ⁢ \left(\right. n \left.\right) \left]\right.$ be the set of functions $C : \left(\left{\right. 0 , 1 \left.\right}\right)^{n} \rightarrow \left{\right. 0 , 1 \left.\right}$ which can be a two-layer feedforward ReLU network with at most $s ⁢ \left(\right. n \left.\right)$ bits of precision and constant hidden dimension $𝖥𝖥_{\theta} : \left(\left{\right. 0 , 1 \left.\right}\right)^{2 ⁢ n} \rightarrow \left{\right. 0 , 1 \left.\right} , 𝖥𝖥_{\theta} ⁢ \left(\right. x^{'} \left.\right) = W_{2} \times_{s} 𝗋𝖾𝗅𝗎 ⁢ \left(\right. \left(\left[\right. W_{1} \times_{s} x^{'} + b_{1} \left]\right.\right)_{s} \left.\right)$, where $\theta = \left(\right. W_{2} , W_{1} , b_{1} \left.\right)$, such that for any $x \in \left(\left{\right. 0 , 1 \left.\right}\right)^{n}$,

$𝖥𝖥_{\theta} ⁢ \left(\right. x_{1} , 1 , x_{2} , 1 , \ldots , x_{n} , 1 \left.\right) = C ⁢ \left(\right. x \left.\right) .$(10)

We have unlimited-fanin $𝖠𝖭𝖣 , 𝖮𝖱 \in 𝖥𝖥 ⁢ \left[\right. 1 \left]\right.$ and $𝖬𝖠𝖩𝖮𝖱𝖨𝖳𝖸 \in 𝖥𝖥 ⁢ \left[\right. log ⁡ n \left]\right.$.

Given two vectors $x , y$ of the same length $s$, we use $x^{⌢} ⁢ y$ to denote their interleaving, that is, $\left(\left(\right. x^{⌢} ⁢ y \left.\right)\right)_{2 ⁢ i - 1} = x_{i} , \left(\left(\right. x^{⌢} ⁢ y \left.\right)\right)_{2 ⁢ i} = y_{i}$ for all $i \in \left[\right. e \left]\right.$.

###### Lemma B.4.

[Lemma E.3, (Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32))] For any $s \in \mathbb{N}^{+}$, let $q_{i} = 𝗌𝖻𝗂𝗇_{s} ⁢ \left(\left(\right. i \left.\right)\right)^{⌢} ⁢ 1_{s}$ and $k_{i} = B_{s} \cdot \left(\right. 𝗌𝖻𝗂𝗇_{s} ⁢ \left(\left(\right. i \left.\right)\right)^{⌢} ⁢ \left(\right. - 1_{s} \left.\right) \left.\right)$ for all $i \in \left[\right. 2^{s} - 1 \left]\right.$, it holds that $\left(\left[\right. exp ⁡ \left(\right. \left(\langle q_{i} , k_{j} \rangle\right)_{s} \left.\right) \left]\right.\right)_{s} = 𝟏 ⁢ \left[\right. i = j \left]\right.$ for all $i , j \in \left[\right. 2^{s} - 1 \left]\right.$.

### B.2 Proofs

#### B.2.1 Looped models can simulate non-looped models

###### Proof of [Theorem 5.2](https://arxiv.org/html/2502.17416v1#S5.Thmtheorem2 "Theorem 5.2. ‣ 5.3 Looped models can simulate non-looped models ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

We start by introduce some more notations. We will proof the theorem for any fixed sequence of $𝒗 = \left(\right. v_{1} , \ldots , v_{n} \left.\right)$. We use $𝒙^{\left(\right. l \left.\right)} = \left(\left(\right. x_{i}^{\left(\right. l \left.\right)} \left.\right)\right)_{i = 1}^{n}$ to denote the intermediate embedding of $p_{\theta}$ in the $l$th layer. More specifically, we define

$𝒙^{l} = \left(\right. 𝗂𝖽 + 𝖥𝖥_{\theta_{𝖥𝖥}^{\left(\right. l - 1 \left.\right)}} \left.\right) \circ \left(\right. 𝗂𝖽 + 𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}^{\left(\right. l - 1 \left.\right)}} \left.\right) \circ ⋯ ⁢ \left(\right. 𝗂𝖽 + 𝖥𝖥_{\theta_{𝖥𝖥}^{\left(\right. 0 \left.\right)}} \left.\right) \circ \left(\right. 𝗂𝖽 + 𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}^{\left(\right. 0 \left.\right)}} \left.\right) \circ 𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖤𝖬𝖡𝖤𝖣}} ⁢ \left(\right. 𝒗 \left.\right) .$(11)

We also use $𝒙^{\left(\right. l + 0.5 \left.\right)} = \left(\left(\right. x_{i}^{\left(\right. l + 0.5 \left.\right)} \left.\right)\right)_{i = 1}^{n}$ to denote the intermediate embedding of $p_{\theta}$ in the $l$th layer after the attention layer.

$𝒙^{l + 0.5} = \left(\right. 𝗂𝖽 + 𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}^{\left(\right. l - 1 \left.\right)}} \left.\right) ⁢ \left(\right. 𝒙^{l} \left.\right) .$(12)

Similarly, for the constructed looped transformer $p_{\theta , T}$, we use $𝒗^{'} = \left(\right. \# , 𝒗_{1} , \ldots , 𝒗_{n} \left.\right)$ to denote its input. For simplicity, we use the convention that $\#$ is at position $0$. The proof still works if $\#$ starts at position $1$ because we can just transfer the later tokens by $1$ position. We define $𝒙^{' \llbracket \left(\right. l \left.\right)} = \left(\right. x_{0}^{' \llbracket \left(\right. l \left.\right)} , x_{1}^{' \llbracket \left(\right. l \left.\right)} , \ldots , x_{n}^{' \llbracket \left(\right. l \left.\right)} \left.\right)$ as the intermediate embedding of $p_{\theta}$ in the $l$th layer and $𝒙^{' \llbracket \left(\right. l + 0.5 \left.\right)} = \left(\right. x_{0}^{' \llbracket \left(\right. l + 0.5 \left.\right)} , x_{1}^{' \llbracket \left(\right. l + 0.5 \left.\right)} , \ldots , x_{n}^{' \llbracket \left(\right. l + 0.5 \left.\right)} \left.\right)$ as the intermediate embedding of $p_{\theta}$ in the $l$th layer.

Below we first state the key properties that our construction will satisfy, which imply the correctness of the theorem and then we state our construction of $p_{\theta^{'} , T}$ and show the properties are actually satisfied:

*   •$x_{i}^{' \llbracket \left(\right. l \left.\right)} = \left(\right. x_{i}^{\left(\right. l \left.\right)} , 𝟏_{R} - e_{r ⁢ \left(\right. l \left.\right)} , l , 𝟏 ⁢ \left[\right. i = 0 \left]\right. \left.\right)$. 
*   •$x_{i}^{' \llbracket \left(\right. l + 0.5 \left.\right)} = \left(\right. x_{i}^{\left(\right. l + 0.5 \left.\right)} , 𝟏_{R} - e_{r ⁢ \left(\right. l \left.\right)} , l , 𝟏 ⁢ \left[\right. i = 0 \left]\right. \left.\right)$. 
*   •$x_{0}^{\left(\right. l \left.\right)} = x_{0}^{\left(\right. l + 0.5 \left.\right)} = 𝟎$.3 3 3 Here we abuse the notation for simplicity of presentation. $x_{0}^{\left(\right. l \left.\right)} = x_{0}^{\left(\right. l + 0.5 \left.\right)}$ are not defined in the original non-looped transformer. The key point here is that they are 0 vectors throughout the forward pass. 

To get the last coordinate $l$, which is a depth counter, we just need to add $1$ more hidden dimension in MLP.

Next we show we can use two-layer with $L + 2$ MLP to get the mapping from $ℓ \rightarrowtail 𝟏_{R} - e_{r ⁢ \left(\right. l \left.\right)}$. Let $\left(\left(\right. \theta_{𝖥𝖥}^{\left(\right. i \left.\right)} , \theta_{𝖬𝖧𝖠}^{\left(\right. i \left.\right)} \left.\right)\right)_{i = 1}^{R}$ be the parameters of the $R$ distinct layers in $\theta$. We assume in the $l$th layer, $r ⁢ \left(\right. l \left.\right)$’s parameters are used. This is because $e_{r ⁢ \left(\right. l \left.\right)} = \sum_{i = 1}^{L} e_{r ⁢ \left(\right. i \left.\right)} ⁢ 0.5 * \left(\right. \left(\left[\right. l - i + 1 \left]\right.\right)_{+} - 2 ⁢ \left(\left[\right. l - i \left]\right.\right)_{+} + \left(\left[\right. l - i - 1 \left]\right.\right)_{+} \left.\right)$.

Now we explain how to deactivate the undesired MLP neurons. In other words, our construction of $\theta_{𝖥𝖥}^{'}$ is essentially concatenation of $\theta_{𝖥𝖥}^{\left(\right. i \left.\right)}$ for $i \in \left[\right. r \left]\right.$ in the hidden dimension of $𝖥𝖥$, with the additional control that $𝖥𝖥_{\theta_{𝖥𝖥}^{'}} ⁢ \left(\right. \left(\right. x_{i}^{\left(\right. l \left.\right)} , 𝟏_{R} - e_{r ⁢ \left(\right. l \left.\right)} , l \left.\right) , 𝟏 ⁢ \left[\right. i = 0 \left]\right. \left.\right) = \sum_{i = 1}^{R} 𝖥𝖥_{\theta_{𝖥𝖥}^{\left(\right. i \left.\right)}} ⁢ \left(\right. x_{i}^{\left(\right. l \left.\right)} \left.\right) ⁢ 𝟏 ⁢ \left[\right. r ⁢ \left(\right. l \left.\right) = i , i \neq 0 \left]\right.$ at $l$th layer. This control can be done by subtracting $𝟏 - e_{r ⁢ \left(\right. l \left.\right)} + 𝟏 ⁢ \left[\right. i = 0 \left]\right.$ by a constant which is larger than the maximum pre-activation in the hidden layer.

Finally we explain how to deactivate the undesired attention. We will only use attention to update the first part of the embedding, which is $x_{i}^{\left(\right. l + 0.5 \left.\right)}$. A crucial step here is that we set the token embedding of $\#$ as $0$ We construct keys and queries as follows:

1.   1.$W_{}^{'} \left(\right. x_{i}^{' \llbracket \left(\right. l \left.\right)} \left.\right) = \left(\right. W_{Q}^{\left(\right. r^{'} \left.\right)} x_{i}^{\left(\right. l \left.\right)} , 1 - 𝟏 \left[\right. r^{'} = r \left(\right. l \left.\right) \left]\right.$ for $r^{'} \in \left[\right. R \left]\right.$ and $i = 0 , \ldots , n$ 
2.   2.$W_{}^{'} ⁢ \left(\right. x_{i}^{' \llbracket \left(\right. l \left.\right)} \left.\right) = \left(\right. W_{K}^{\left(\right. r^{'} \left.\right)} ⁢ x_{i}^{\left(\right. l \left.\right)} , - B ⁢ 𝟏 ⁢ \left[\right. i = 0 \left]\right. \left.\right)$ for $r^{'} \in \left[\right. R \left]\right.$ and $i = 0 , \ldots , n$, where $B$ is some constant larger than the maximum previous inner product in attention, $max_{l \in \left[\right. L \left]\right. , i , j} \langle \left(\right. W_{K} x_{i}^{\left(\right. l \left.\right)} , \left(\right. W_{Q} x_{i}^{\left(\right. l \left.\right)} \rangle$. 
3.   3.$W_{}^{'} ⁢ W_{}^{'} ⁢ \left(\right. x_{i}^{' \llbracket \left(\right. l \left.\right)} \left.\right) = \left(\right. W_{O}^{\left(\right. r^{'} \left.\right)} ⁢ W_{V}^{\left(\right. r^{'} \left.\right)} ⁢ x_{i}^{\left(\right. l \left.\right)} , 𝟎 , 0 , 0 \left.\right)$. 

This construction works because only the ‘desired’ attention head $r = r ⁢ \left(\right. l \left.\right)$ will be activated and behave as in the non-looped case, because otherwise all position in that attention head will be completely attended to position $0$ and returns a zero vector. (We can choose $B$ to be large enough and distribution calculated by the attention score is delta distribution) at position $0$, which yields a zero vector as its value. This completes the proof. ∎

#### B.2.2 Group composition.

Algorithm 2 Group Composition

1:Group elements $g_{0} , g_{1} , \ldots , g_{n} \in G$, where $g_{0} = e$. 

2:$g_{0} \circ g_{1} \circ \ldots ⁢ g_{n}$. 

3:$g_{i}^{\left(\right. 0 \left.\right)} = g_{i}$, $\forall 0 \leq i \leq n$. 

4:for$l = 1 \rightarrow \lceil log_{2} ⁡ n \rceil$do

5:$a_{i}^{\left(\right. l \left.\right)} = g_{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}}^{\left(\right. l - 1 \left.\right)} , b_{i}^{\left(\right. l \left.\right)} = g_{\left(\left[\right. 2 ⁢ i - n \left]\right.\right)_{+}}^{\left(\right. l - 1 \left.\right)}$$\forall 0 \leq i \leq n$. 

6:$g_{i}^{\left(\right. l \left.\right)} = a_{i}^{\left(\right. l \left.\right)} \circ b_{i}^{\left(\right. l \left.\right)}$, $\forall 0 \leq i \leq n$. 

7:end for

8:return$g_{n}^{\left(\right. \lceil log_{2} ⁡ n \rceil \left.\right)}$. 

The landmark result in automata theory, Krohn-Rhodes Decompotision Theorem(Krohn & Rhodes, [1965](https://arxiv.org/html/2502.17416v1#bib.bib28)), shows that all semi-automaton with solvable transformation group (which includes composition problem of solvable groups) can be simulated by a cascade of permutation-reset automata, which can be simulated by $𝖳𝖢^{0}$ circuits. (Liu et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib34)) further showed that such automaton with solvable transformation group can be continuously simulated by constant-depth transformers. However, it is also shown(Barrington, [1986](https://arxiv.org/html/2502.17416v1#bib.bib4)) that the composition problem of unsolvable groups are $𝖭𝖢^{1}$-complete, for example, the composition of permutation group over $5$ elements, $S_{5}$. Under the common hardness assumption that $𝖭𝖢^{1} \neq 𝖳𝖢^{0} \left.\right)$, constant depth transformer cannot solve composition of $S_{5}$ using a single forward pass(Merrill & Sabharwal, [2023b](https://arxiv.org/html/2502.17416v1#bib.bib38), Liu et al., [2022](https://arxiv.org/html/2502.17416v1#bib.bib34), Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32)). But with CoT, very shallow transformers (depth equal to one or two) can simulate the composition problem of any group(Li et al., [2024](https://arxiv.org/html/2502.17416v1#bib.bib32), Merrill & Sabharwal, [2023a](https://arxiv.org/html/2502.17416v1#bib.bib37)).

###### Proof of [Theorem 5.1](https://arxiv.org/html/2502.17416v1#S5.Thmtheorem1 "Theorem 5.1. ‣ 5.2 Group composition problem ‣ 5 Theoretical analysis for looped models ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

We will set the token embedding of $\#$ the same as that of $e$, which is the identity of $G$. In the following proof, we will just treat $\#$ as $e$. We will construct the transformer simulating group composition following the following algorithm[Algorithm 2](https://arxiv.org/html/2502.17416v1#alg2 "In B.2.2 Group composition. ‣ B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), which gives the high-level idea of the construction. The correctness of [Algorithm 2](https://arxiv.org/html/2502.17416v1#alg2 "In B.2.2 Group composition. ‣ B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") follows from the associativity of group composition. More concretely, we can verify by induction that $g_{0}^{l} \circ g_{1}^{l} \circ \ldots ⁢ g_{n}^{l}$ is the same for all $l = 0 , \ldots , \lceil log_{2} ⁡ n \rceil$ and in the final round, i.e., when $l = \lceil log_{2} ⁡ n \rceil$, $g_{i}^{\left(\right. l \left.\right)} = e$ for all $i < n$.

Below we show how to construct a transformer of the given sizes to simulate the above [Algorithm 2](https://arxiv.org/html/2502.17416v1#alg2 "In B.2.2 Group composition. ‣ B.2 Proofs ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). We will embed each $g \in G$ as a different vector $\bar{g} \in \left(\left{\right. - 1 , 1 \left.\right}\right)^{\lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil}$ and each position $0 \leq i \leq n$ as its binary representation in $\bar{i} \in \left(\left{\right. - 1 , 1 \left.\right}\right)^{\lceil log_{2} ⁡ n + 1 \rceil}$, which is a shorthand for $𝗌𝖻𝗂𝗇_{s} ⁢ \left(\right. i \left.\right)$ with $s = \lceil log_{2} ⁡ n + 1 \rceil$. We concatenate them to get $\left(\left{\right. x_{i}^{\left(\right. 0 \left.\right)} \left.\right}\right)_{i = 0}^{n}$, that is, $x_{i}^{\left(\right. 0 \left.\right)} = \left(\right. \bar{g_{i}} , \bar{i} , \bar{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}} , \bar{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}} , 0^{\lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil} , 0^{\lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil} \left.\right)$. For convenience, we will drop the 0’s in the end (also in the other proofs of the paper) and write it as $x_{i}^{\left(\right. 0 \left.\right)} = \left(\right. \bar{g_{i}} , \bar{i} , \bar{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}} , \bar{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}} \left.\right)$. Below we show we can construct 1-layer transformer block with parameter $\left(\right. \theta_{𝖬𝖧𝖠} , \theta_{𝖥𝖥} \left.\right)$ satisfying that

1.   1.$\left(\left[\right. 𝖬𝖧𝖠_{\theta_{𝖬𝖧𝖠}} ⁢ \left(\right. \left(\left(\right. \bar{g_{i}} , \bar{i} , \bar{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}} , \bar{\left(\left[\right. 2 ⁢ i - n - 1 \left]\right.\right)_{+}} \left.\right)\right)_{i = 0}^{n} \left.\right) \left]\right.\right)_{k} = \left(\right. 0^{\lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil + 3 ⁢ \lceil log_{2} ⁡ n + 1 \rceil} , \bar{g_{\left(\left[\right. 2 ⁢ k - n - 1 \left]\right.\right)_{+}}} , \bar{g_{\left(\left[\right. 2 ⁢ k - n \left]\right.\right)_{+}}} \left.\right)$

for all $g_{0} = e , g_{i} \in G ⁢ \forall i \in \left[\right. n \left]\right.$, $k = 0 , \ldots , n$; 
2.   2.$𝖥𝖥_{\theta_{𝖥𝖥}} ⁢ \left(\right. \bar{g} , \bar{i} , \bar{j} , \bar{k} , \bar{g^{'}} , \bar{g^{′′}} \left.\right) = \left(\right. \bar{g^{'} \circ g^{′′}} - \bar{g} , 0^{3 ⁢ \lceil log_{2} ⁡ n + 1 \rceil} , - \bar{g^{'}} , - \bar{g^{′′}} \left.\right)$, for all $i , j , k = 0 , \ldots , n$, $g , g^{'} , g^{′′} \in G$. 

The first claim is because we can use two attention heads to retrieve $\bar{g_{\left(\left[\right. 2 ⁢ k - n - 1 \left]\right.\right)_{+}}}$ and $\bar{g_{\left(\left[\right. 2 ⁢ k - n \left]\right.\right)_{+}}}$ respectively, where both of them use $\bar{k}$ as the key and use $- \bar{\left(\left[\right. 2 ⁢ k - n - 1 \left]\right.\right)_{+}}$ and $- \bar{\left(\left[\right. 2 ⁢ k - n \left]\right.\right)_{+}}$ as queries respectively. This is possible because all the required information are already in $x_{i}$. We further make attention temperature low enough so the probability returned by attention is a one-hot distribution at the position whose key is equal to the negative query after rounding.

Now we turn to the second claim about MLP. We will use $\left(\left|\right. G \left|\right.\right)^{2}$ neurons with ReLU activation and bias to simulate the product of $g^{'}$ and $g^{′′}$. We can index each neuron by $\left(\right. h , h^{'} \left.\right)$ for $h , h^{'} \in G$ and set its incoming weight $\left(\left[\right. W_{1} \left]\right.\right)_{\left(\right. h , h^{'} \left.\right) , :} = \left(\right. \bar{h} , \bar{h^{'}} \left.\right)$ and set bias $\left(\left(\right. b_{1} \left.\right)\right)_{\left(\right. h , h^{'} \left.\right)} = - 2 ⁢ \lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil + 1$, which ensures that the activation of neuron $\left(\right. h , h^{'} \left.\right)$ will only be $1$ when $g^{'} = h , g^{′′} = h^{'}$ and be $0$ otherwise. Then setting the outgoing weight of neuron $\left(\right. h , h^{'} \left.\right)$ as $\bar{h \circ h^{'}}$ and the bias in the second layer to be $0$ finishes the construction for simulating the group composition. Finally we use the remaining $6 ⁢ \lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil$ to simulate negative identity mapping $x \rightarrow - x$ for the remaining $3 ⁢ \lceil log_{2} ⁡ \left|\right. G \left|\right. \rceil$ embedding dimension. This completes the proof. ∎

### B.3 Connection to chain-of-thought reasoning

In this section, we establish a connection betwee looped models and CoT reasoning. We first define the recursion for CoT reasoning as follows:

$𝖳𝖥_{\theta}^{i} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \triangleq 𝖳𝖥_{\theta}^{i - 1} ⁢ \left(\right. v_{1} , \ldots , v_{n} , 𝖳𝖥_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \left.\right) ,$

for $i , n \in \mathbb{N}^{+}$ satisfying $i + n \leq n_{max} - 1$ along with the base case of $𝖳𝖥_{\theta}^{1} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \triangleq 𝖳𝖥_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right)$. For all $0 \leq i \leq n_{max} - n - 1$, the output with $i$ steps of CoT is $v_{n + i + 1} = 𝖳𝖥_{\theta}^{i + 1} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) = 𝖳𝖥_{\theta} ⁢ \left(\right. v_{1} , \ldots , v_{n} , v_{n + 1} , \ldots , v_{n + i} \left.\right)$.

We first give the formal statement below.

###### Theorem B.5(Looped transformer simulates CoT).

For any $L$-layer non-looped transformer $𝖳𝖥_{\theta}$, there exists a looped transformer $𝖳𝖥_{\theta^{'}}$ with $L + \mathcal{O} ⁢ \left(\right. 1 \left.\right)$ layers, constantly many more dimensions in embedding, MLP and attention layers and constantly many more attention heads, such that for any input $𝒗 = \left(\left(\right. v_{i} \left.\right)\right)_{i = 1}^{n}$ and integer $m$, the output of non-looped transformer after $m$ steps of CoT, $𝖳𝖥_{\theta}^{m} ⁢ \left(\right. 𝒗 \left.\right)$, is the same as that of the looped transformer on input $x$ concatenated by $m$ dummy tokens with $m$ loops, $𝖳𝖥_{\theta^{'} , m} ⁢ \left(\right. 𝒗 , \#^{m} \left.\right)$.

Below are some helping lemmas towards showing [Theorem B.5](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem5 "Theorem B.5 (Looped transformer simulates CoT). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") is at least as powerful as CoT.

###### Lemma B.6(Simulating $arg ⁢ max$ using MLP).

For every $d \in \mathbb{N}$ and precision $s \in \mathbb{N}^{+}$, there exists a 3-layer network with $𝗋𝖾𝗅𝗎$ activation and $d^{2}$ width $f$ with $s$ precision, such that for any $x \in \mathbb{F}_{s}^{d}$, if there is $k \in \left[\right. d \left]\right.$, such that $x_{k} > max_{j \neq k , j \in \left[\right. d \left]\right.} ⁡ x_{j}$, $f ⁢ \left(\right. x \left.\right) = e_{k}$.

###### Proof of [Lemma B.6](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem6 "Lemma B.6 (Simulating arg⁢max using MLP). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

Define $g_{i} = 2^{s} \cdot 𝗋𝖾𝗅𝗎 ⁢ \left(\right. 2^{- s} - \sum_{j \neq i} 𝗋𝖾𝗅𝗎 ⁢ \left(\right. x_{j} - x_{i} \left.\right) \left.\right)$ for each $i \in \left[\right. n \left]\right.$. We claim that if there is $k \in \left[\right. d \left]\right.$, such that $x_{k} - max_{j \neq k , j \in \left[\right. d \left]\right.} ⁡ x_{j} \geq 2^{- s}$, $g_{i} = 1$ iff $i = k$ for all $i \in \left[\right. d \left]\right.$. First $g_{k} = 2^{s} \cdot 𝗋𝖾𝗅𝗎 ⁢ \left(\right. 2^{- s} \left.\right) = 1$. Next for $i \neq k$, it clearly holds that $\sum_{j \neq i} 𝗋𝖾𝗅𝗎 ⁢ \left(\right. x_{j} - x_{i} \left.\right) \geq 2^{- s}$ and thus $g_{i} \leq 0$. This construction can clearly be implemented by a 3-layer $𝗋𝖾𝗅𝗎$ network with $s$ precision. ∎

###### Lemma B.7(Simulating Decoding and Embedding using MLP).

Given any $s$-precision $\theta_{𝖳𝖤} \in \mathbb{R}^{d \times \Sigma}$ and $\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}$, there is a 5-layer network $f : \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$ with $𝗋𝖾𝗅𝗎$ activation and $max ⁡ \left(\right. \left(\left|\right. \Sigma \left|\right.\right)^{2} \left.\right)$ width with $s$-precision, such that for all $s$-precision $x \in \mathbb{R}^{d}$ which admits unique $arg ⁢ max$ for $v \triangleq \left(arg ⁢ max\right)_{o \in \Sigma} ⁡ \left(\right. x^{\top} ⁢ \theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \left.\right) ⁡ \left(\right. o \left.\right)$, it holds that

$f ⁢ \left(\right. x \left.\right) = \theta_{𝖳𝖤} ⁢ \left(\right. v \left.\right) .$

###### Proof of [Lemma B.7](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem7 "Lemma B.7 (Simulating Decoding and Embedding using MLP). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

This is a simple application of [Lemma B.6](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem6 "Lemma B.6 (Simulating arg⁢max using MLP). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). ∎

###### Lemma B.8(Control Gate).

A 2-layer $𝗋𝖾𝗅𝗎$ network with precision $s$ can implement $F : \mathbb{F}_{s} \times \mathbb{F}_{s} \times \left{\right. 0 , 1 \left.\right} , F ⁢ \left(\right. x , y , M \left.\right) = M ⁢ x + \left(\right. 1 - M \left.\right) ⁢ y$.

###### Proof of [Lemma B.8](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem8 "Lemma B.8 (Control Gate). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

Note that $F ⁢ \left(\right. x , y , M \left.\right) = 𝗋𝖾𝗅𝗎 ⁢ \left(\right. x - 2^{s} \cdot \left(\right. 1 - M \left.\right) \left.\right) - 𝗋𝖾𝗅𝗎 ⁢ \left(\right. - x - 2^{s} \cdot \left(\right. 1 - M \left.\right) \left.\right) + 𝗋𝖾𝗅𝗎 ⁢ \left(\right. y - 2^{s} \cdot M \left.\right) - 𝗋𝖾𝗅𝗎 ⁢ \left(\right. - y - 2^{s} \cdot M \left.\right)$. The proof is completed. ∎

###### Definition B.3(Transformer Block with Mask).

Given number of layers $L \in \mathbb{N}^{+}$, parameter $\theta_{𝖳𝖡} = \left(\left(\right. \theta_{𝖬𝖧𝖠}^{\left(\right. l \left.\right)} , \theta_{𝖥𝖥}^{\left(\right. l \left.\right)} \left.\right)\right)_{l = 0}^{L - 1}$, and mask function $M : \mathbb{N} \rightarrow \left{\right. 0 , 1 \left.\right}$, we define the $L$-layer _transformer block with mask_$𝖳𝖡_{\theta_{𝖳𝖡} , M} : \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n} \rightarrow \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n}$ for any $n \in \mathbb{N}^{+}$ as

$\left(\left[\right. 𝖳𝖡_{\theta_{𝖳𝖡} , M} ⁢ \left(\right. 𝒙 \left.\right) \left]\right.\right)_{i} \triangleq \left(\right. 1 - M ⁢ \left(\right. i \left.\right) \left.\right) ⁢ x_{i} + M ⁢ \left(\right. i \left.\right) ⁢ \left(\left[\right. 𝖳𝖡_{\theta_{𝖳𝖡}} ⁢ \left(\right. 𝒙 \left.\right) \left]\right.\right)_{i}$(13)

###### Definition B.4(Looped Transformer with Mask).

Given number of loops $T \in \mathbb{N}^{+}$, parameters $\theta = \left(\right. \theta_{𝖳𝖡} , \theta_{𝖳𝖤} , \theta_{𝖯𝖤} , \theta_{𝖮𝖴𝖳𝖯𝖴𝖳} \left.\right)$, and mask functions $\left(\left{\right. M^{t} \left.\right}\right)_{t = 1}^{T}$, where $\theta_{𝖳𝖥} = \left(\left(\right. \theta_{𝖬𝖧𝖠}^{\left(\right. l \left.\right)} , \theta_{𝖥𝖥}^{\left(\right. l \left.\right)} \left.\right)\right)_{l = 0}^{L - 1}$, we define the _looped transformer with mask_ as $p_{\theta , T , M} \triangleq 𝖮𝖴𝖳𝖯𝖴𝖳_{\theta_{𝖮𝖴𝖳𝖯𝖴𝖳}} \circ 𝖳𝖡_{\theta_{𝖳𝖡} , M^{T}} \circ ⋯ ⁢ 𝖳𝖡_{\theta_{𝖳𝖡} , M^{1}} \circ 𝖤𝖬𝖡𝖤𝖣_{\theta_{𝖳𝖤} , \theta_{𝖯𝖤}}$ and the corresponding deterministic version as $𝖳𝖥_{\theta , T , M} ⁢ \left(\right. v_{1} , \ldots , v_{n} \left.\right) \triangleq \left(arg ⁢ max\right)_{v \in \mathcal{V}} ⁡ p_{\theta , T , M} ⁢ \left(\right. v \left|\right. v_{1} , \ldots , v_{n} \left.\right)$.

###### Definition B.5(Shifting Layer).

We define the _shifting layer_$𝖲𝖧𝖨𝖥𝖳 : \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n} \rightarrow \left(\left(\right. \mathbb{R}^{d} \left.\right)\right)^{n}$ as the following for any $d , n \in \mathbb{N}^{+}$ and $x_{1} , \ldots , x_{n} \in \mathbb{R}^{d}$:

$𝖲𝖧𝖨𝖥𝖳 ⁢ \left(\right. x_{1} , x_{2} , x_{3} , \ldots , x_{n} \left.\right) = \left(\right. x_{1} , x_{1} , x_{2} , x_{3} , \ldots , x_{n - 1} \left.\right) .$(14)

###### Lemma B.9.

For input sequence length up to some integer $n$, $𝖲𝖧𝖨𝖥𝖳$ could be implemented by a attention layer by concatenating each embedding $x_{i}$ with $\left(\right. 𝗌𝖻𝗂𝗇_{s} ⁢ \left(\right. i \left.\right) , 𝗌𝖻𝗂𝗇_{s} ⁢ \left(\right. f ⁢ \left(\right. i \left.\right) \left.\right) \left.\right)$, where $n = \lceil log_{2} ⁡ n + 1 \rceil$.

###### Proof of [Lemma B.9](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

It is equivalent to show we can construct an attention heads which computes $x_{f ⁢ \left(\right. i \left.\right)}$ at each position $i$.

To do so, we just need to invoke [Lemma B.6](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem6 "Lemma B.6 (Simulating arg⁢max using MLP). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") and use that to set key and query, so position $i$ attends to position $f ⁢ \left(\right. i \left.\right)$. We set value at position $i$ to be $x_{i}$. This completes the proof. ∎

###### Lemma B.10.

For any positive integer $s > 0$, there is a constant-depth MLP $F$ with $O ⁢ \left(\right. s \left.\right)$ hidden neurons per layer and parameters in $\mathbb{F}_{s}$, such that for any input $𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) \in \left(\left{\right. - 1 , + 1 \left.\right}\right)^{s}$, where $0 \leq x \leq 2^{s} - 1$, it holds that

$F ⁢ \left(\right. x \left.\right) = 𝖻𝗂𝗇 ⁢ \left(\right. x + 1 \left.\right) .$

###### Proof of [Lemma B.10](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem10 "Lemma B.10. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

By [Lemma B.3](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem3 "Lemma B.3. ‣ Finite-precision Modeling: ‣ B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), it suffices to show that we can simulate $𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) \rightarrowtail 𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) + 1$ using $O ⁢ \left(\right. s \left.\right)$ wide, constant-depth boolean circuits with $𝖠𝖭𝖣 , 𝖮𝖱 , 𝖭𝖮𝖳$ gates with unlimited fan-ins. This is immediate by noting that

$\left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. x + 1 \left.\right) \left]\right.\right)_{i} = \left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. x + 1 \left.\right) \left]\right.\right)_{i} \oplus \wedge_{j = 1}^{i - 1} \left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. x + 1 \left.\right) \left]\right.\right)_{j}$(15)

∎

###### Lemma B.11.

For any positive integer $s > 0$, there is a constant-depth MLP $F$ with $O ⁢ \left(\right. s \left.\right)$ hidden neurons per layer and parameters in $\mathbb{F}_{s}$, such that for any input $\left(\right. 𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) , 𝖻𝗂𝗇 ⁢ \left(\right. y \left.\right) \left.\right) \in \left(\left{\right. - 1 , + 1 \left.\right}\right)^{s}$, where $0 \leq x , y \leq 2^{s} - 1$, it holds that

$F ⁢ \left(\right. x , y \left.\right) = 𝟏 ⁢ \left[\right. x > y \left]\right. .$

###### Proof of [Lemma B.11](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem11 "Lemma B.11. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

By [Lemma B.3](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem3 "Lemma B.3. ‣ Finite-precision Modeling: ‣ B.1 Detailed notations ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), it suffices to show that we can simulate $𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) , 𝖻𝗂𝗇 ⁢ \left(\right. y \left.\right) \rightarrowtail 𝟏 ⁢ \left[\right. x > y \left]\right.$ using $O ⁢ \left(\right. s \left.\right)$ wide, constant-depth boolean circuits with $𝖠𝖭𝖣 , 𝖮𝖱 , 𝖭𝖮𝖳$ gates with unlimited fan-ins. This is immediate by noting that

$𝟏 ⁢ \left[\right. x > y \left]\right. = \underset{i \in \left[\right. s \left]\right.}{\vee} \left(\right. \left(\right. \left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) \left]\right.\right)_{i} = 1 \left.\right) \land \left(\right. \left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. y \left.\right) \left]\right.\right)_{i} = 0 \left.\right) \land \underset{1 \leq j < i}{\wedge} \left(\right. \left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. x \left.\right) \left]\right.\right)_{j} = \left(\left[\right. 𝖻𝗂𝗇 ⁢ \left(\right. y \left.\right) \left]\right.\right)_{j} \left.\right) \left.\right) .$(16)

∎

###### Proof of [Theorem B.5](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem5 "Theorem B.5 (Looped transformer simulates CoT). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers").

We consider mask $M^{t} ⁢ \left(\right. i \left.\right) = 𝟏 ⁢ \left[\right. i - t \geq n \left]\right.$, which we call it CoT masking. Let $s = \lfloor log ⁡ \left(\right. n + m \left.\right) + 1 \rfloor$, we use $\bar{i}$ to denote $𝗌𝖻𝗂𝗇_{s} ⁢ \left(\right. i \left.\right)$ for $1 \leq i \leq n + m$ for convenience. The embedding size of our newly constructed transformer is larger than the target transformer to be simulated by an additive constant of $3 ⁢ s$. We denote the new embedding at position $i$ after $t$th loop by $\left(\right. x_{i}^{\left(\right. t \left.\right)} , p_{i}^{\left(\right. t \left.\right)} \left.\right)$, where $x_{i}^{\left(\right. t \left.\right)}$ will be the original embedding of the transformer to be simulated, and $p_{i}^{\left(\right. t \left.\right)}$ is of dimension $3 ⁢ s$ and only depends on $i$ and $t$. In particular, we can show that we can set $p_{i}^{\left(\right. t \left.\right)} \triangleq \left(\right. \bar{i} , \bar{\left(\left[\right. i - 2 \left]\right.\right)_{+} + 1} , \bar{n + t} \left.\right)$ to save information about position and the number of loops — $p_{i}^{\left(\right. 0 \left.\right)}$ is from the positional embedding and the update is due to [Lemma B.10](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem10 "Lemma B.10. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). The size of hidden dimension of MLP and attention (key, query, value) will also be increased by $O ⁢ \left(\right. s \left.\right)$.

The proof contains two steps:

1.   1.To show that there is a transformer with CoT masks simulating the target transformer with $m$ steps of CoT and $L$ layers by looping its own $L + O ⁢ \left(\right. 1 \left.\right)$ layer block $m$ times. 
2.   2.To show that the above looped transformer with CoT masks can be simulated by a standard looped transformer without mask and with constantly many more layers. 

For the first claim, starting from the same parameter of the transformers with CoT $\theta$, we build a new looped model with parameter $\theta^{'}$ with constantly many more layers in each transformer block and at most constantly many more heads per attention layer. First, we can add constantly many more layers to use MLP to simulate the decoding-encoding process using [Lemma B.7](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem7 "Lemma B.7 (Simulating Decoding and Embedding using MLP). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"). Next, we can add one more transformer layer in each block and use the attention layer to simulate the shifting layer by[Lemma B.9](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), since we have the additional position embedding $p_{i}^{\left(\right.} t \left.\right)$. In particular, the embedding we get at position $n + t$ after $t$ loops, $x_{n + t}^{\left(\right. t \left.\right)}$, now simulates the token embedding of $n + t$ of the CoT transformer. By the way we define CoT mask $M$, for every $t \geq - n + 1$, the embedding $\left(\hat{x}\right)_{n + t}^{\left(\right. t^{'} \left.\right)}$ will keep the same for all $t^{'} \geq max ⁡ \left(\right. t , 0 \left.\right)$. In $t$th loop, the only embedding update that matters happens at $n + t$th position, because no updates happen at earlier positions, and updates at later positions $n + t^{'}$ for some $t^{'} > t$ will be overwritten eventually in the future loops $t^{'}$, by some value which is independent of their value at the current loop $t$. In the end, we know the embedding $x_{i}^{\left(\right. T \left.\right)}$ in [Definition B.4](https://arxiv.org/html/2502.17416v1#A2.Thmdefinition4 "Definition B.4 (Looped Transformer with Mask). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers") is exactly equal to that in CoT transformer, and so does the final output.

For the second claim, because CoT mask can be computed by a $O ⁢ \left(\right. log ⁡ \left(\right. n + m \left.\right) \left.\right)$ wide, constant-depth MLP([Lemma B.11](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem11 "Lemma B.11. ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")), together with [Lemma B.8](https://arxiv.org/html/2502.17416v1#A2.Thmtheorem8 "Lemma B.8 (Control Gate). ‣ B.3 Connection to chain-of-thought reasoning ‣ Appendix B Theoretical results ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers"), we know it suffices to increase the number of layers per transformer block and embedding size and hidden dimensions by constant to simulate the transformer with mask by a transformer without mask. ∎

###### Theorem B.12(Restatement of Theorem 4.2, (Sanford et al., [2024b](https://arxiv.org/html/2502.17416v1#bib.bib50))).

$p$-hop problem ([Definition A.1](https://arxiv.org/html/2502.17416v1#A1.Thmdefinition1 "Definition A.1 (𝑝-𝗁𝗈𝗉, Sanford et al. (2024b)). ‣ 𝑝-hop induction. ‣ A.1 Simple reasoning setup details ‣ Appendix A Experiments ‣ Reasoning with Latent Thoughts: On the Power of Looped Transformers")) can be solved by $\lfloor log_{2} ⁡ p \rfloor + 2$-layer non-looped transformer with $log ⁡ n$ bits of precision, at most $3$ different layers, $d = d_{𝖥𝖥} = d_{𝖠𝖳𝖳𝖭} = O ⁢ \left(\right. 1 \left.\right)$ embedding size, hidden dimension for MLP and attention, $1$ attention head.

Generated on Mon Feb 24 18:46:09 2025 by [L a T e XML![Image 28: Mascot Sammy](blob:http://localhost/70e087b9e50c3aa663763c3075b0d6c5)](http://dlmf.nist.gov/LaTeXML/)
