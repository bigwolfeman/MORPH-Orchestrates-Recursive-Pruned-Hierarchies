# ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing

- **Authors:** Ziteng Wang, Jun Zhu, Jianfei Chen (Tsinghua University)
- **Year:** 2024 (ICLR 2025)
- **Source:** https://arxiv.org/abs/2412.14711
- **MORPH uses:** ReLU-based continuous routing over macro tile-groups (32x32 Block-ELL tiles), replacing the non-differentiable TopK gate with a differentiable L1-regularized ReLU that naturally produces sparse expert selection without the gradient discontinuity of standard MoE routing.

---

Published as a conference paper at ICLR 2025 

## - - REMOE: FULLY DIFFERENTIABLE MIXTURE OF EXPERTS WITH RELU ROUTING 

## **Ziteng Wang, Jun Zhu, Jianfei Chen** _[∗]_ 

Dept. of Comp. Sci. and Tech., Institute for AI, BNRist Center, THBI Lab, Tsinghua-Bosch Joint ML Center, Tsinghua University wangzite23@mails.tsinghua.edu.cn; _{_ dcszj,jianfeic _}_ @tsinghua.edu.cn 

## ABSTRACT 

Sparsely activated Mixture-of-Experts (MoE) models are widely adopted to scale up model capacity without increasing the computation budget. However, vanilla TopK routers are trained in a discontinuous, non-differentiable way, limiting their performance and scalability. To address this issue, we propose ReMoE, a fully differentiable MoE architecture that offers a simple yet effective drop-in replacement for the conventional TopK+Softmax routing, utilizing ReLU as the router instead. We further propose methods to regulate the router’s sparsity while balancing the load among experts. ReMoE’s continuous nature enables efficient dynamic allocation of computation across tokens and layers, while also exhibiting domain specialization. Our experiments demonstrate that ReMoE consistently outperforms vanilla TopK-routed MoE across various model sizes, expert counts, and levels of granularity. Furthermore, ReMoE exhibits superior scalability with respect to the number of experts, surpassing traditional MoE architectures. The implementation based on Megatron-LM is available at https://github.com/thu-ml/ReMoE. 

## 1 INTRODUCTION 

Transformer models (Vaswani, 2017) consistently improve performance as the number of parameters increases (Kaplan et al., 2020). However, scaling these models is constrained by computation resources. Sparsely activated Mixture-of-Experts (MoE) (Shazeer et al., 2017) mitigates this challenge by employing a sparse architecture that selectively activates a subset of parameters during both training and inference. This conditional computation allows MoE models to expand model capacity without increasing computational costs, offering a more efficient alternative to dense models. 

The key component in MoE is the routing network, which selects the experts to activate for each token. Various routing methods (Shazeer et al., 2017; Lewis et al., 2021; Roller et al., 2021; Zhou et al., 2022) have been proposed, with TopK routing (Shazeer et al., 2017) being the most commonly adopted. However, the vanilla TopK router introduces a discrete and non-differentiable training objective (Shazeer et al., 2017; Zoph et al., 2022), limiting the performance and scalability. 

Recent works on fully-differentiable MoE aim to overcome this limitation. Soft MoE (Puigcerver et al., 2023) introduces token merging, while SMEAR (Muqeeth et al., 2023) proposes expert merging. However, both approaches break token causality, making them unsuitable for autoregressive models. Lory (Zhong et al., 2024) improves upon SMEAR and is applicable to autoregressive models. But it underperforms vanilla MoE with TopK routing. 

In this work, we address the discontinuities by introducing ReMoE, an MoE architecture that incorporates ReLU routing as a simple yet effective drop-in replacement for TopK routing. Unlike TopK routing, which computes a softmax distribution over the experts and calculates a weighted sum of the largest _K_ experts, ReLU routing directly controls the active state of each expert through a ReLU gate. The number of active experts is determined by the sparsity of the ReLU function. To maintain the desired sparsity, we propose adding a load-balancing refined _L_ 1 regularization to the router outputs, with an adaptively tuned coefficient. This approach ensures that ReMoE maintains the same computational costs as TopK-routed MoE. 

> _∗_ Corresponding author 

1 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 177] intentionally omitted <==**

**----- Start of picture text -----**<br>
TopK Routing ReLU Routing<br>Softmax TopK ReLU<br>.42 -.46 .05 .01 .36 .15 .25 .24 .36 0 .25 0 .42 -.46 .05 .01 .42 0 .05 .01<br>projection Expert1 multiplyand add projection Expert1 multiplyand add<br>Token 1 Token 1 Token 1 Token 1<br>Expert Expert<br>2 2<br>Expert Expert<br>3 3<br>Token  𝑻 Token  𝑻 Token  𝑻 Token  𝑻<br>projection Expert4 multiplyand add projection Expert4 multiplyand add<br>Softmax TopK ReLU<br>-.75 .81 -.59 -.40 .12 .57 .14 .17 0 .57 0 .17 -.75 .81 -.59 -.40 0 .81 0 0<br>· · · ·<br>· · · ·<br>· · · ·<br>**----- End of picture text -----**<br>


Figure 1: Compute flows of vanilla MoE with TopK routing and ReMoE with ReLU routing. Positive values are shown in orange, and negative values in blue, with deeper colors representing larger absolute values. Zeros, indicating sparsity and computation savings, are shown in white. The red dash arrows in TopK routing indicate discontinuous operations. Compared with TopK routing MoE, ReMoE uses ReLU to make the compute flow fully differentiable. 

Compared to TopK routing, ReLU routing is continuous and fully differentiable, as the ReLU function can smoothly transition between zero and non-zero values, indicating inactive and active. Besides, ReLU routing manages the “on/off” state of each expert independently, offering greater flexibility. Moreover, the number of activated experts can vary across tokens and layers, enabling a more efficient allocation of computational resources. Further analysis reveals that ReMoE effectively learns to allocate experts based on token frequency and exhibits stronger domain specialization. 

Our experiments on mainstream LLaMA (Touvron et al., 2023) architecture demonstrate that ReLU routing outperforms existing routing methods including TopK routing and fully-differentiable Lory. Through an extensive investigation across model structures, we find that ReMoE consistently outperforms TopK-routed MoE across a broad range of active model sizes (182M to 978M), expert counts (4 to 128), and levels of granularity (1 to 64) (Krajewski et al., 2024). Notably, in terms of scaling behavior, we observe that ReMoE exhibits a steeper performance improvement as the number of experts scales up, surpassing traditional MoE models. 

## 2 PRELIMINARIES 

## 2.1 MOE FOR DECODER-ONLY TRANSFORMER 

A typical decoder-only Transformer model consists of _L_ layers, each containing a Self-Attention module and a Feed-Forward Network (FFN) module. MoE modifies this structure by replacing each FFN module with an MoE module, which comprises a small router and several experts FFN1 _, . . . ,_ FFN _E_ , where each expert is equivalent to the original FFN and _E_ denotes the number of experts. Given the input _**x**[l]_ = ( _**x**[l] t_[)] _[T] t_ =1 _[∈]_[R] _[T][ ×][d]_[of the layer] _[ l]_[, where] _[ T]_[is the number of tokens in a] batch and _d_ is the hidden size, the output _**y**[l]_ = ( _**y** t[l]_[)] _[T] t_ =1[is computed as:] 

**==> picture [266 x 30] intentionally omitted <==**

Here, _R_ ( _·_ ) represents the routing function, and _dffn_ is the intermediate size of the FFN, typically set to _dffn_ = 4 _d_ . 

## 2.2 TOPK ROUTING 

TopK routing (Shazeer et al., 2017; Lepikhin et al., 2020; Fedus et al., 2022) is the most commonly used method for defining the routing function _R_ ( _·_ ). It introduces sparsity in the MoE computation 

2 

Published as a conference paper at ICLR 2025 

by forcibly zeroing out smaller elements: 

**==> picture [146 x 12] intentionally omitted <==**

(2) 

where _**W** l ∈_ R _[d][×][E]_ is the router’s weight matrix, and TopK( _·, k_ ) retains the top _k_ largest values while setting the rest to zero. This mechanism allows for skipping the computation of the FFN _e_ functions corresponding to the zeroed-out _R_ ( _**x**[l] t_[)] _[e]_[values in both the forward and backward passes.] 

## 3 OUR METHOD: REMOE 

## 3.1 MOTIVATION: FROM TOPK TO RELU 

For a given token _**x**_ = ( _xe_ ) _[E] e_ =1[after][Softmax,][TopK][introduces][a][jump][discontinuity][at][the] _[k]_[-th] largest value, denoted as _x_ [ _k_ ], by zeroing out the values smaller than _x_ [ _k_ ]. This can be expressed as: 

TopK( _**x** , k_ ) _e_ = _xe ·_ **1** _{xe ≥ t_ ( _**x** , k_ ) _}, t_ ( _**x** , k_ ) = _x_ [ _k_ ] (3) 

where **1** _{·}_ is the indicator function, returning 1 if the condition is met and 0 otherwise. 

As shown in Figure 2, the jump discontinuity can be eliminated by setting the breakpoint _t_ ( _**x** , k_ ) _≡_ 0, which actually corresponds to the ReLU function: 

**==> picture [154 x 11] intentionally omitted <==**

**==> picture [198 x 64] intentionally omitted <==**

**----- Start of picture text -----**<br>
TopK 𝒙, 𝑘 ! ReLU 𝒙 !<br>0 𝑥 " 𝑥! 0 𝑥!<br>**----- End of picture text -----**<br>


Figure 2: Comparison between TopK and ReLU. 

At a high level, ReLU improves upon TopK by aligning the breakpoints of all inputs and setting them to 0. This ensures that the output is continuous at 0, where the experts transition between active and inactive. As a result, the training pipeline becomes fully differentiable. 

## 3.2 DIFFERENTIABLE RELU ROUTING 

We define the ReLU routing function as follows: 

**==> picture [247 x 13] intentionally omitted <==**

with (1 _− E[k]_[)][ being the desired sparsity of ReLU, where] _[ k]_[ is the number of active experts and] _[ E]_[is] the total number of experts. This ensures that the computational cost remains equivalent to that of TopK routing. 

In vanilla TopK routers, the Softmax outputs sum to 1, representing the probabilities of selecting each expert, after which TopK eliminates those with lower probabilities. In contrast, ReLU routers discard the Softmax function, relying on ReLU’s naturally non-negative outputs. The outputs of ReLU routers represent the weights assigned to each expert, which can include 0. Instead of hardcoding expert selection with a discontinuous TopK function, ReLU allows the router to learn which experts to activate (i.e., when to produce 0s) in a fully differentiable manner. 

Another key difference is that in TopK routing, each token is routed to exactly _k_ experts, whereas in ReLU routing ReMoE, the routing decisions are independent, allowing tokens to be routed to a variable number of experts. This flexibility is advantageous, as not all tokens have the same level of difficulty. ReMoE can allocate more computational resources to more challenging tokens, a dynamic allocation strategy that we explore further in Section 5.1. 

TopK routing introduces a discrete loss function when the set of activated experts changes, whereas ReLU routing remains continuous and fully differentiable. For instance, in a two-expert Top1routing model, a small weight update that alters the softmax result from _**x**_ 1 = (0 _._ 51 _,_ 0 _._ 49) to _**x**_ 2 = (0 _._ 49 _,_ 0 _._ 51) shifts the TopK output from (0 _._ 51 _,_ 0) to (0 _,_ 0 _._ 51), creating a discontinuity. In contrast, ReLU routing only changes the activated experts when the routing output is near zero. For example, an output shift from (0 _._ 01 _,_ 0) to (0 _,_ 0 _._ 01) remains continuous. Further details on the stability analysis of these two routers can be found in Appendix A. 

A comparison of the compute flow between ReMoE and MoE is shown in Figure 1. 

3 

Published as a conference paper at ICLR 2025 

## 3.3 CONTROLLING SPARSITY VIA ADAPTIVE _L_ 1 REGULARIZATION 

ReMoE controls computational costs by managing the sparsity of the ReLU output, targeting a sparsity level of (1 _− E[k]_[)][.][However, directly training the ReLU router often results in lower sparsity,] as the model tends to activate more experts to increase capacity. To meet the desired budget, we need to enforce higher sparsity in the ReLU output. 

We achieve this by introducing a regularization loss, _Lreg_ , to the loss of language model, _Llm_ : 

**==> picture [240 x 11] intentionally omitted <==**

where _λi_ is an adaptive coefficient based on the current training step _i_ . Initially, we set _λ_ 0 to a small value and employ a simple zeroth-order algorithm to update it: 

**==> picture [255 x 14] intentionally omitted <==**

Here, _α >_ 1 is a preset update multiplier, and _Si_ denotes the average sparsity of all router outputs at the step _i_ : 

**==> picture [288 x 30] intentionally omitted <==**

The idea behind Equation 7 is that when the average sparsity _Si_ falls below the target sparsity (1 _− E[k]_[)][,][we increase] _[ λ][i]_[by a factor of] _[ α]_[, strengthening the regularization and encouraging higher] sparsity. Conversely, if the sparsity exceeds the target, _λi_ is reduced. We heuristically set _λ_ 0 = 1 _e[−]_[8] and _α_ = 1 _._ 2 in all our experiments, and demonstrate the robustness of these hyperparameters in Appendix B. 

The regularization term _Lreg_ uses the _L_ 1-norm, following prior work (Li et al., 2022; Song et al., 2024), to effectively encourage sparsity: 

**==> picture [316 x 31] intentionally omitted <==**

The second equation holds because the output of the ReLU function is non-negative. 

The term _Lreg_ represents the average value of all router outputs, including zeros. By taking the derivative of _λiLreg_ , we observe that the regularization effect adds _LT[λ][i]_[to the gradient of] each non-zero router output, effectively driving the outputs toward zero and enhancing sparsity. 

With this _L_ 1 regularization, we can control the sparsity around the desired level of (1 _− E[k]_[)][ with] only minor fluctuations, as shown in Figure 3. Consequently, ReMoE ensures that, on average, tokens are routed to _k_ experts across different layers and tokens, maintaining the same FLOPs as vanilla TopK-routed MoE from a statistical perspective. Our benchmarking results in Appendix D demonstrate that ReMoE can achieve nearly identical training and inference throughputs as conventional MoE, providing an efficient alternative without compromising speed. 

**==> picture [198 x 100] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>0.75<br>0.50<br>Sparsity=0.8751±0.0061<br>0.25 Desired Sparsity=0.875<br>0 10000 20000 30000 40000 50000 60000<br>Step<br>Sparsity<br>**----- End of picture text -----**<br>


Figure 3: The sparsity of ReMoE with _E_ = 8 _, k_ = 1 is effectively maintained around the desired target. Sparsity values for all steps are plotted without averaging or sampling. The mean and standard deviation are calculated excluding the first 100 warm-up steps. 

## 3.4 INTEGRATE LOAD BALANCING INTO _L_ 1 REGULARIZATION 

Load imbalance is a significant issue in MoE design, potentially leading to routing collapse (Shazeer et al., 2017; Muennighoff et al., 2024) and uneven computational distribution across multiple devices. The _L_ 1 regularization in Equation 9 treats the router output for each expert _e_ and each layer _l_ equally, which can contribute to load balancing problems. 

4 

Published as a conference paper at ICLR 2025 

**==> picture [394 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 Stage I Stage II Stage III 10 [1] Stage I Stage II Stage III 10 Stage I Stage II Stage III 10 [1]<br>0.80.6 1010 13 1010 [0] 1 86 101010 [0] 12<br>0.4 10 5 10 2 4 10 3<br>0.2 SparsityDesired Sparsity 10 7 ireg 10 3 2 ilmreg 10 4<br>0.0 10 4 0 10 5<br>0 50 100 150 200 60000 0 50 100 150 200 60000 0 50 100 150 200 60000<br>Step Step Step<br>(a) Sparsity  Si (b) Coefficient term λi and regular- (c) Language model loss Llm and<br>ization term  Lreg overall regularization  λiLreg<br>i reg lm ireg<br>Sparsity<br>**----- End of picture text -----**<br>


Figure 4: Natural Three Stage Training in ReMoE. 

To address this, we introduce a load-balancing refinement to the _L_ 1 regularization: 

**==> picture [274 x 65] intentionally omitted <==**

Here, _fl,e_ is non-differentiable and represents the average activation ratio of expert _e_ in layer _l_ , relative to the desired ratio _[k] E_[. This serves as a weight for the corresponding router output, modifying] the added gradient of non-zero router outputs to _[f][l] LT[,][e][λ][i]_[.][This mechanism penalizes experts receiving] more tokens by driving their router outputs toward zero more rapidly. 

Although derived from regularization, this formulation is _identical_ to the load-balancing loss in vanilla TopK routing (Fedus et al., 2022). In TopK routing, the outputs of Softmax sum to 1, giving the loss a lower bound of 1. In contrast, ReLU routing outputs can be arbitrarily small, making _Lreg,lb_ trivially bounded at 0. Therefore, unlike in MoE, we cannot fix the coefficient _λi_ in ReMoE, as this would lead to routing collapse toward 0. Thanks to the adaptive update of _λi_ , we can balance sparsity control and load balancing within a single formulation, as given in Equation 10. 

Further discussion on load balancing in ReMoE can be found in Section 5.2, and we adopt this load-balancing refined _L_ 1 regularization in our later experiments. 

## 3.5 NATURAL THREE-STAGE TRAINING IN REMOE 

With the regularization scheme described above, we observe a clear and naturally occurring threestage separation during the training of ReMoE as is depicted in Figure 4. 

The first stage is the warm-up stage, or the dense stage. During this stage, _λi_ is small, while _Llm_ is large and decreases rapidly. Training ReMoE at this stage is nearly equivalent to training its dense counterpart with the same total number of parameters. Each expert processes more than half of the tokens, allowing the experts to diversify from their random initializations. 

The second stage is the sparsifying stage, or the dense to sparse stage. At this point, the sparse regularization term _λiLreg_ becomes significant, causing the ReLU routers to activate fewer experts. This forces the experts to become more diverse without causing an increase in _Llm_ . 

The third stage is the stable stage, or the sparse stage. In this phase, the sparsity _Si_ stabilizes at the preset target. During this stage, _Llm_ is optimized while being softly guided along the sparse subspace by _Lreg_ . Both _Lreg_ and _λi_ change very slowly, with _Lreg_ gradually decreasing and _λi_ gradually increasing. However, the overall regularization term, _λiLreg_ , remains relatively constant. 

It should be noted that Stages I and II introduce additional computational cost and memory consumption since more experts are activated. However, the time overhead is negligible since they generally require only _∼_ 100 iterations ( _∼_ 0.17% of the total steps in our setting, benchmarking results are detailed in Appendix D). The memory overhead can be minimized by temporarily reducing the micro-batch size or by employing the activation checkpointing technique that avoids storing intermediate results of activated experts by recomputing them on-the-fly during the backward pass. 

5 

Published as a conference paper at ICLR 2025 

||||
|---|---|---|
|**Model**|**ARC-c**<br>**ARC-e**<br>**BoolQ**<br>**HellaSwag**<br>**LAMBADA**<br>**PIQA**<br>**RACE**|**Avg.**|
|Dense<br>Hash<br>Lory<br>SparseMixer-v2<br>EC<br>dMoE<br>ReMoE|19.45<br>43.35<br>54.40<br>28.61<br>31.09<br>61.97<br>28.52<br>19.28<br>45.45<br>54.95<br>29.68<br>31.44<br>63.06<br>27.66<br>**20.31**<br>42.97<br>49.54<br>28.75<br>32.35<br>62.24<br>27.75<br>19.80<br>**46.72**<br>45.96<br>30.24<br>34.12<br>62.89<br>29.00<br>18.86<br>42.97<br>**60.21**<br>29.14<br>29.26<br>61.92<br>27.37<br>20.05<br>45.16<br>57.83<br>29.83<br>32.97<br>**63.55**<br>28.33<br>20.22<br>46.68<br>54.16<br>**30.26**<br>**35.94**<br>**63.55**<br>**29.38**|38.20<br>38.79<br>37.70<br>38.39<br>38.53<br>39.67<br>**40.03**|
||||



Figure 5: Training curves of difTable 2: Zero-shot accuracy of different routing methods on ferent routing methods. downstream tasks. 

## 4 EXPERIMENTS 

## 4.1 SETUP 

**Infrastructure** We leverage Megatron-LM (Shoeybi et al., 2019) as our code base and implement ReLU routing as a drop-in replacement for the original TopK routing, supporting all forms of model parallelism: Data, Tensor, Pipeline, and Expert Parallelism (Shoeybi et al., 2019; Narayanan et al., 2021; Korthikanti et al., 2023). 

**Model Architecture.** We experiment with the mainstream LLaMA (Touvron et al., 2023) architecture, featuring grouped query attention (GQA) (Ainslie et al., 2023), SwiGLU (Shazeer, 2020) activation function, RoPE (Su et al., 2024) position embedding, and RMSNorm (Zhang & Sennrich, 2019). The context length is set to 1024, and the batch size is 512. We experiment with three different dense backbone sizes as shown in Table 1. For vanilla MoE we adopt a load balancing loss of weight 0 _._ 01 following Fedus et al. (2022). For ReMoE we use the adaptive load balancing _L_ 1 regularization in Equation 10. 

**Training Settings.** We train the models on The Pile (Gao et al., 2020), an 800 GB diverse corpus. All models are trained for 60k steps ( _∼_ 30B tokens), which exceeds the compute-optimal dataset size predicted by Krajewski et al. (2024) and is enough to converge. The byte pair encoding (BPE) tokenizer (Sennrich, 2015) is used. We adopt AdamW (Loshchilov, 2017) as the optimizer with _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 999 with ZeRO optimization (Rajbhandari et al., 2020). The learning rate is set to be 5 _e[−]_[4] with a cosine scheduler. All models are trained with 8 NVIDIA A100 GPUs. 

|**Size**|**#Parameters**<br>**hidden**<br>~~**s**~~**ize**<br>**num**<br>~~**l**~~**ayers**<br>**num**<br>~~**h**~~**eads**<br>**num**<br>~~**g**~~**roups**<br>**GFLOPs**|
|---|---|
|Smal<br>Mediu<br>Larg|l<br>182M<br>768<br>12<br>12<br>4<br>995<br>m<br>469M<br>1024<br>24<br>16<br>4<br>2873<br>e<br>978M<br>1536<br>24<br>16<br>4<br>5991|



Table 1: Configurations for the dense backbones. FLOPs are calculated with a single sequence according to Narayanan et al. (2021). 

## 4.2 COMPARISON WITH OTHER ROUTING METHODS 

We compare ReMoE against the following methods: (i) Token-choice dropless TopK routing (dMoE) (Gale et al., 2023) (ii) Expert-choice TopK routing (EC) (Zhou et al., 2022) (iii) Deterministic hash routing (Hash) (Roller et al., 2021) (iv) Fully-differentiable expert-merging routing (Lory) (Zhong et al., 2024) (v) TopK routing with improved gradient estimate (SparseMixer-v2) (Liu et al., 2024b). 

The performance of these methods is evaluated with active parameters _N_ = 182M and the expert count _E_ = 8. We fix the active expert count to _k_ = 1 for straightforward comparison with the dense counterpart. For the Hash method, we use mod _E_ hashing function. And for Lory, the segment length is set to 256, following the original paper. 

6 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
Dense 2.05 2.05<br>2.0 2.0421.936 MoE ReMoE 2.00 1.971 2.042 2.00 2.042 Dense MoE<br>1.9 1.921 1.95 1.961 1.936 1.907 1.95 1.936 ReMoE Dense×8<br>1.81.7 1.8861.7961.773 1.822 1.7301.715 1.901.851.80 DenseMoE ReMoE1.921 1.883 1.874 1.852 1.854 1.826 1.852 1.815 1.901.85 1.921 1.9121.9021.873 1.8991.896 1.8981.885 1.8951.879 1.8931.872 1.885 1.872<br>182M 469M 978M 4 8 16 32 64 128 1 2 4 8 16 32 64<br>Number of Parameters N Expert Count E Granularity G<br>(a) Scaling in  N (b) Scaling in  E (c) Scaling in  G<br>Valid Loss Valid Loss Valid Loss<br>**----- End of picture text -----**<br>


Figure 6: Scalability of ReMoE with respect to the number of active parameters ( _N_ ), expert count ( _E_ ), and granularity ( _G_ ). Default config is _N_ = 182M _, E_ = 8 _, G_ = 1 _, k_ = 1. The Y-axis represents the validation loss of each model after training on 30B tokens. ReMoE consistently outperforms MoE across all configurations. 

These models are trained on 30B tokens, with the training curves shown in Figure 5, We evaluate the zero-shot performance of the trained models on the following downstream tasks: ARC (Clark et al., 2018); BoolQ (Clark et al., 2019); HellaSwag (Zellers et al., 2019); LAMBADA (Paperno et al., 2016); PIQA (Bisk et al., 2020); RACE (Lai et al., 2017). 

The downstream accuracy results are summarized in Table 2. 

Our results show that all MoE models outperform the dense model. Deterministic hash routing performs worse than the learned routing methods. Among the Top-K approaches, token-choice dMoE outperforms expert-choice MoE and SparseMixer-v2 in evaluation. The differentiable routing method Lory surpasses Hash routing in training but underperforms in downstream tasks, with both methods falling short of the standard Top-K routing. Notably, ReMoE outperforms all methods, including the mainstream Top-K routing, while benefiting from differentiability. 

## 4.3 SCALABILITY OF REMOE 

In this section, we compare ReMoE with state-of-the-art dMoE (hereinafter referred to simply as MoE) across varying model parameters _N_ , expert counts _E_ , and granularity levels _G_ to demonstrate its scalability and universal superiority. Since ReMoE demands more computation in both Stage I and Stage II, we increase the number of training steps for the MoE baseline to match the total computation in each setting, ensuring a more equitable comparison. We present the final validation losses in Figure 6, with comprehensive downstream evaluation results available in Appendix E. 

**Scaling in active parameters** _N_ **.** To assess scalability with respect to the number of parameters _N_ , we fix _E_ = 8 and _k_ = 1, while varying active parameters _N_ from 182M to 975M, corresponding to the dense counterpart configurations in Table 1. The total parameters are 777M, 2.58B, 5.73B respectively. The results, shown in Figure 6a, indicate that ReMoE consistently outperforms MoE across all model sizes. The performance gap does not diminish as the model size increases, suggesting that ReMoE maintains its advantage at larger scales. 

**Scaling in expert count** _E_ **.** In this experiment, we fix the number of parameters at _N_ = 182M and set the number of active experts _k_ = 1, while varying the total number of experts _E_ from 4 to 128. The scaling curve in Figure 6b reveals that ReMoE consistently outperforms the standard MoE across all configurations of _E_ . 

Moreover, a key observation is the steeper slope in ReMoE’s performance as _E_ increases, compared to MoE. This suggests that ReMoE scales more effectively with the number of experts and derives greater benefits from larger expert pools. ReMoE’s differentiable routing strategy appears better suited for leveraging large expert groups, leading to significant improvements in model expressivity and generalization. 

**Scaling in granularity** _G_ **.** We also evaluate ReMoE and MoE in fine-grained settings. Finegrained MoE (Dai et al., 2024; Krajewski et al., 2024) with granularity _G_ is constructed by dividing 

7 

Published as a conference paper at ICLR 2025 

each expert into _G_ smaller experts, as formulated below: 

**==> picture [263 x 30] intentionally omitted <==**

**==> picture [280 x 13] intentionally omitted <==**

Fine-grained MoE outperforms vanilla MoE from a scaling law perspective (Krajewski et al., 2024) and has been adopted in subsequent works (Dai et al., 2024; Tan et al., 2024; Muennighoff et al., 2024). For fine-grained ReMoE, the routing function remains identical to Equation 5, and the target sparsity is still (1 _− E[k]_[)][.][The][only][distinction][lies][in][the][shape][of][the][weight][matrix,][with] _**[W]**[l][∈]_ R _[d][×][EG]_ . 

We conduct experiments with _N_ = 182M and _E_ = 8, varying _G_ from 1 to 64 for both fine-grained MoE and fine-grained ReMoE. In addition to comparing these models against the dense baseline with the same number of active parameters, we also evaluate their dense counterpart with the same total number of parameters. This is achieved by expanding the intermediate size of the FFN by a factor of _E_ , which we denote as _Dense×8_ . This configuration represents the strict upper bound for MoE and ReMoE, as it is equivalent to a Mixture-of-Experts with all experts activated (Dai et al., 2024). 

As illustrated in Figure 6c, fine-grained ReMoE consistently outperforms fine-grained MoE. Moreover, fine-grained ReMoE of _G_ = 32 and _G_ = 64 reach the performance of the theoretical upper bound, _Dense×8_ , while requiring significantly fewer FLOPs during both training and inference. In contrast, fine-grained MoE is unable to match in all settings, making ReMoE a more efficient and effective choice. 

## 5 DISCUSSION 

## 5.1 DYNAMIC EXPERT ALLOCATION IN REMOE 

In ReMoE, each token dynamically activates a subset of experts, allowing the model to adaptively allocate resources. We evaluate the performance of the _N_ = 182M _, E_ = 8 _, k_ = 1 ReMoE model and analyze the relationship between token frequency and the average number of active experts. As illustrated in Figure 7, the model tends to assign a higher number of experts to rarer tokens, such as ’©’, ’OTAL’, and ’@#’, while reducing the number of active experts for more frequent tokens like ’ ’, ’ _\_ n’, and ’the’. 

This adaptive behavior mirrors the principles of a Huffman tree Huffman (1952), where more frequent symbols are assigned shorter codes, and rarer symbols are assigned longer codes. Similarly, ReMoE tends to “cluster on” common tokens by activating fewer experts, effectively compressing the “representation” of these frequent tokens. In contrast, for rarer tokens, ReMoE activates a more diverse set of experts, “encoding” them as a richer linear combination at the expert level. This suggests that 

**==> picture [159 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [0]<br>3.0 Average Active Expert Count<br>Uniform Expert Assignment 10 1<br>2.5 Token Frequency<br>2.0 10 2<br>1.5 10 3<br>1.0<br>10 4<br>0.5<br>10 5<br>0 10000 20000 30000 40000 50000<br>Sorted Token ID<br>Token Frequency<br>Average Active Expert Count<br>**----- End of picture text -----**<br>


Figure 7: Correlation between expert allocation and token frequency in ReMoE. X-axis is sorted by average active expert count and token frequency is in log-scale. 

ReMoE learns to dynamically allocate computational resources, achieving an efficient balance between resource usage and the model’s capacity, optimizing performance under a constrained expert budget. Dynamic expert allocation is also evident at the domain level, as detailed in Appendix G. 

## 5.2 THE ROLE OF LOAD BALANCING IN REMOE 

Load imbalance can lead to routing collapse in the vanilla TopK-routed MoE, where the router tends to assign the same expert to all inputs, in which scenario the training objective becomes continuous and fully differentiable. As is shown in Figure 8a, there is a significant performance gap between MoE models with and without load balancing (LB). 

8 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 597] intentionally omitted <==**

**----- Start of picture text -----**<br>
Expert ID Expert ID<br>0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7<br>1 1<br>2.6 0 0 1.00<br>Dense 1 1 1 1<br>2.4 MMoE w. LBoE w.o. LB 23 21 23 21 0.95<br>ReMoE w.o. LB 4 4 4 4 0.90<br>2.2 ReMoE w. LB 56 18 56 18 0.85<br>2.0 78 161 78 161 0.80 ReMoE w. LB ReMoE w.o. LB<br>0 5 10 15 20 25 30 109 321 109 321 0.75 0 1 Avg. Sparsity2 3 4 5 6 7 8 9 10 11<br>#Tokens(B) 11 2 i 641 11 641 Layer ID<br>(a) Training curves of MoE and (b) Average routed to- (c) Average routed to- (d) Sparsity across different<br>ReMoE with and without load kens ratio of ReMoE kens ratio of ReMoE layers in ReMoE<br>balancing w.o. LB w. LB<br>Figure 8: Observations on the role of load balancing in MoE and ReMoE. White squares in (b)<br>represent inactive experts with fewer than 1/64 tokens routed to them.<br>While in ReLU routing, thanks to its differentiablity, even applying the  L 1 regularization from Equa- regularization from Equa-<br>tion 9 without load balancing yields comparable results with a well-tuned MoE with LB. However,<br>some experts in ReMoE without LB remain inactive, illustrated as white squares in Figure 8b which<br>shows the heat map of the  average routed tokens ratio  (i.e., the fraction of tokens routed to the  e -th<br>expert in the  l -th layer) over 50M tokens in test set. This inactivity can limit the model’s capacity.<br>When load balancing is incorporated into the refined L 1 regularization (Equation 10), the experi-<br>ments show a more even distribution of token assignments across experts, with all experts being<br>utilized, as shown in Figure 8c. The final loss in ReMoE decreases after introducing load balancing.<br>Besides, we observe ReMoE with LB can produce a smoother sparsity distribution across layers as<br>depicted in Figure 8d. This is because fl,el,e is computed based on the absolute number of routed<br>tokens, meaning denser layers receive stronger penalties.<br>Note that even ReMoE with load balancing (LB) does not yield a perfectly even distribution. How-<br>ever, the trade-off between load balancing and performance can be easily adjusted by modifying<br>the L 1 regularization in Equation 10. For instance, changing fl,el,e to fl,el,e [[2]] [[would]] [[make]] [[the]] [[model]]<br>more sensitive to load imbalance. Additionally, device-level load balancing techniques, as proposed<br>in Dai et al. (2024), could also be employed. Since load imbalance in ReMoE does not lead to<br>severe routing collapse, it primarily becomes a hardware utilization issue. As such, we leave the<br>exploration of these variants for future work.<br>Layer 0 Layer 5 Layer 11<br>0.8 Arxiv<br>0.6 Books<br>C4<br>0.4<br>Github<br>0.2<br>Stack<br>0.0<br>Wiki<br>0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7<br>Expert ID<br>(a) Domain specialization of MoE<br>Layer 0 Layer 5 Layer 11<br>0.8 Arxiv<br>0.6 Books<br>C4<br>0.4<br>Github<br>0.2<br>Stack<br>0.0<br>Wiki<br>0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7<br>Expert ID<br>(b) Domain specialization of ReMoE<br>Train Loss Layer ID Layer ID Sparsity<br>Routed Tokens Ratio<br>Routed Tokens Ratio<br>**----- End of picture text -----**<br>


Figure 8: Observations on the role of load balancing in MoE and ReMoE. White squares in (b) represent inactive experts with fewer than 1/64 tokens routed to them. 

While in ReLU routing, thanks to its differentiablity, even applying the _L_ 1 regularization from Equa- regularization from Equation 9 without load balancing yields comparable results with a well-tuned MoE with LB. However, some experts in ReMoE without LB remain inactive, illustrated as white squares in Figure 8b which shows the heat map of the _average routed tokens ratio_ (i.e., the fraction of tokens routed to the _e_ -th expert in the _l_ -th layer) over 50M tokens in test set. This inactivity can limit the model’s capacity. 

When load balancing is incorporated into the refined _L_ 1 regularization (Equation 10), the experiments show a more even distribution of token assignments across experts, with all experts being utilized, as shown in Figure 8c. The final loss in ReMoE decreases after introducing load balancing. 

Besides, we observe ReMoE with LB can produce a smoother sparsity distribution across layers as depicted in Figure 8d. This is because _fl,el,e_ is computed based on the absolute number of routed tokens, meaning denser layers receive stronger penalties. 

Note that even ReMoE with load balancing (LB) does not yield a perfectly even distribution. However, the trade-off between load balancing and performance can be easily adjusted by modifying the _L_ 1 regularization in Equation 10. For instance, changing _fl,el,e_ to _fl,el,e_[[2]][[would]][[make]][[the]][[model]] more sensitive to load imbalance. Additionally, device-level load balancing techniques, as proposed in Dai et al. (2024), could also be employed. Since load imbalance in ReMoE does not lead to severe routing collapse, it primarily becomes a hardware utilization issue. As such, we leave the exploration of these variants for future work. 

Figure 9: Average routed tokens ratio for MoE and ReMoE across 12 layers and 8 experts in different domains. The gray dashed lines indicate uniform distribution. ReMoE shows stronger domain specialization. 

9 

Published as a conference paper at ICLR 2025 

## 5.3 DOMAIN SPECIALIZATION IN REMOE 

The differentiability and dynamic allocation strategy of ReMoE facilitates the development of diverse experts that specialize in different domains. This allows the router to effectively perform ensemble learning by leveraging the expertise of various experts, as demonstrated in our experiments. 

In Figure 9, we plot the average routed tokens ratio across different experts, layers, and domains—namely Arxiv, Books, C4, Github, Stackexchange, and Wikipedia—for MoE and ReMoE models with _N_ = 182M _, E_ = 8. We focus on the first, middle, and last layers (with IDs 0, 5, and 11). The results for most experts in MoE (Figure 9a) show a roughly uniform distribution across all domains. In contrast, experts in ReMoE (Figure 9b) exhibit clear domain specialization, being activated with varying frequencies across different domains. For example, more than half of the tokens from Arxiv, Github, and StackExchange—domains that emphasize structured, non-natural languages like LaTeX and Python—are routed to Expert 6 in Layer 5, significantly more than in other domains. A more detailed result of domain specialization can be found in Appendix F. 

## 6 RELATED WORKS 

## 6.1 MIXTURE-OF-EXPERTS 

Mixture-of-Experts (MoE) was initially proposed in the early 1990s (Jacobs et al., 1991; Jordan & Jacobs, 1994) and later introduced into large-scale neural networks as a sparse submodule for efficiency (Shazeer et al., 2017). Advances like GShard (Lepikhin et al., 2020) and Switch Transformer (Fedus et al., 2022) integrated sparse MoE into Transformer models, achieving significant results. More recently, MoE has been used in commercial-scale language models such as Mixtral-8x7B (Jiang et al., 2024), DeepSeekMoE 16B (Dai et al., 2024), and Snowflake Arctic 17B (Snowflake, 2024). 

## 6.2 ROUTING MECHANISMS IN MOE 

Various routing methods have been developed for expert selection. Static routers, such as BASE (Lewis et al., 2021), use predefined rules like combinatorial optimization, while Hash routing (Roller et al., 2021) relies on deterministic hash functions, and THOR (Zuo et al., 2021) assigns experts randomly with regularization. Learned routers adaptively select experts based on token input, using approaches like REINFORCE (Bengio et al., 2013; Schulman et al., 2015; Clark et al., 2022) for reinforcement learning, and TopK routing (Shazeer et al., 2017; Zhou et al., 2022) for token or expert selection, though TopK introduces discontinuities that hinder gradient estimation. 

## 6.3 DIFFERENTIABLE MIXTURE-OF-EXPERTS 

Recent work on fully differentiable MoE models addresses the challenges of discrete optimization, basically through token merging and expert merging approaches. Soft MoE (Puigcerver et al., 2023) uses token merging, assigning fixed slots to each expert as a linear combination of input tokens. SMEAR (Muqeeth et al., 2023) merges experts into an ensemble via weighted averaging. However, both methods require a full probability map of input tokens, making them unsuitable for autoregressive models. Lory (Zhong et al., 2024) preserves autoregressiveness by segmenting sentences to merge experts but underperforms compared to TopK routing. 

## 7 CONCLUSION 

In this paper, we propose ReMoE, a fully differentiable MoE architecture with ReLU routing. The simple yet effective ReLU routing function acts as a drop-in replacement for the conventional TopK+Softmax routing, offering (i) continuity and differentiability, and (ii) dynamic expert allocation across tokens and layers. With the adaptive load balancing _L_ 1 regularization, ReMoE universally outperforms TopK-routed MoE across various model sizes, expert counts, and levels of granularity, demonstrating sharper performance gains as the number of experts scales. 

10 

Published as a conference paper at ICLR 2025 

## ACKNOWLEDGMENT 

The authors gratefully acknowledge Chao Du and Tianyu Pang for the insightful discussions. This work was supported by the NSFC Project (No. 62376131), Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University. J.Z is also supported by the XPlorer Prize. 

## REFERENCES 

- Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebr´on, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. _arXiv preprint arXiv:2305.13245_ , 2023. 

- Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. _arXiv preprint arXiv:1308.3432_ , 2013. 

- Vincent-Pierre Berges, Barlas O˘guz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi Gosh. Memory layers at scale. _arXiv preprint arXiv:2412.09764_ , 2024. 

- Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 7432–7439, 2020. 

- Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In _International conference on machine learning_ , pp. 4057– 4086. PMLR, 2022. 

- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. _arXiv preprint arXiv:1905.10044_ , 2019. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- R´obert Csord´as, Piotr Pikekos,[´] Kazuki Irie, and J¨urgen Schmidhuber. Switchhead: Accelerating transformers with mixture-of-experts attention. _Advances in Neural Information Processing Systems_ , 37:74411–74438, 2025. 

- Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. _arXiv preprint arXiv:2401.06066_ , 2024. 

- William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. _Journal of Machine Learning Research_ , 23(120):1–39, 2022. 

- Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. _Proceedings of Machine Learning and Systems_ , 5:288–304, 2023. 

- Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_ , 2020. 

- Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Hayden Kwok-Hay So, Ting Cao, Fan Yang, and Mao Yang. Seerattention: Learning intrinsic sparse attention in your llms. _arXiv preprint arXiv:2410.13276_ , 2024. 

- Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ , 2023. 

11 

Published as a conference paper at ICLR 2025 

Xu Owen He. Mixture of A Million Experts, July 2024. URL http://arxiv.org/abs/ 2407.04153. arXiv:2407.04153 [cs]. 

- Zihao Huang, Qiyang Min, Hongzhi Huang, Defa Zhu, Yutao Zeng, Ran Guo, and Xun Zhou. Ultrasparse memory network. _arXiv preprint arXiv:2411.12364_ , 2024. 

- David A Huffman. A method for the construction of minimum-redundancy codes. _Proceedings of the IRE_ , 40(9):1098–1101, 1952. 

- Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. _Neural computation_ , 3(1):79–87, 1991. 

- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. _Advances in Neural Information Processing Systems_ , 37:52481–52515, 2025. 

- Pengkun Jiao, Xinlan Wu, Bin Zhu, Jingjing Chen, Chong-Wah Ngo, and Yugang Jiang. Rode: Linear rectified mixture of diverse experts for food large multi-modal models. _arXiv preprint arXiv:2407.12730_ , 2024. 

- Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. _Neural computation_ , 6(2):181–214, 1994. 

- Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_ , 2020. 

- Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. _Proceedings of Machine Learning and Systems_ , 5:341–353, 2023. 

- Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pi´oro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Kr´ol, Tomasz Odrzyg´o´zd´z, Piotr Sankowski, et al. Scaling laws for fine-grained mixture of experts. _arXiv preprint arXiv:2402.07871_ , 2024. 

- Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pp. 785–794, 2017. 

- Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herv´e J´egou. Large memory layers with product keys. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. _arXiv preprint arXiv:2006.16668_ , 2020. 

- Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In _International Conference on Machine Learning_ , pp. 6265–6274. PMLR, 2021. 

- Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J Reddi, Ke Ye, Felix Chern, Felix Yu, Ruiqi Guo, et al. The lazy neuron phenomenon: On emergence of activation sparsity in transformers. _arXiv preprint arXiv:2210.06313_ , 2022. 

- Enshu Liu, Junyi Zhu, Zinan Lin, Xuefei Ning, Matthew B Blaschko, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs. _arXiv preprint arXiv:2407.00945_ , 2024a. 

12 

Published as a conference paper at ICLR 2025 

Liyuan Liu, Young Jin Kim, Shuohang Wang, Chen Liang, Yelong Shen, Hao Cheng, Xiaodong Liu, Masahiro Tanaka, Xiaoxia Wu, Wenxiang Hu, et al. Grin: Gradient-informed moe. _arXiv preprint arXiv:2409.12136_ , 2024b. 

- I Loshchilov. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. _arXiv preprint arXiv:2402.14800_ , 2024. 

- Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. Olmoe: Open mixture-of-experts language models. _arXiv preprint arXiv:2409.02060_ , 2024. 

- Mohammed Muqeeth, Haokun Liu, and Colin Raffel. Soft merging of experts with adaptive routing. _arXiv preprint arXiv:2306.03745_ , 2023. 

- Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ , pp. 1–15, 2021. 

- Denis Paperno, Germ´an Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fern´andez. The lambada dataset: Word prediction requiring a broad discourse context. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 1525–1534, 2016. 

- Joan Puigcerver, Carlos Riquelme, Basil Mustafa, and Neil Houlsby. From sparse to soft mixtures of experts. _arXiv preprint arXiv:2308.00951_ , 2023. 

- Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In _SC20: International Conference for High Performance Computing, Networking, Storage and Analysis_ , pp. 1–16. IEEE, 2020. 

- Stephen Roller, Sainbayar Sukhbaatar, Jason Weston, et al. Hash layers for large sparse models. _Advances in Neural Information Processing Systems_ , 34:17555–17566, 2021. 

- John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel. Gradient estimation using stochastic computation graphs. _Advances in neural information processing systems_ , 28, 2015. 

Rico Sennrich. Neural machine translation of rare words with subword units. _arXiv preprint arXiv:1508.07909_ , 2015. 

Noam Shazeer. Glu variants improve transformer. _arXiv preprint arXiv:2002.05202_ , 2020. 

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. _arXiv preprint arXiv:1701.06538_ , 2017. 

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_ , 2019. 

Snowflake. Arctic open: Efficient foundation language models at snowflake, April 2024. URL https://www.snowflake.com/blog/ arctic-open-efficient-foundation-language-models-snowflake/. 

Chenyang Song, Xu Han, Zhengyan Zhang, Shengding Hu, Xiyu Shi, Kuai Li, Chen Chen, Zhiyuan Liu, Guangli Li, Tao Yang, et al. Prosparse: Introducing and enhancing intrinsic activation sparsity within large language models. _arXiv preprint arXiv:2402.13516_ , 2024. 

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_ , 568:127063, 2024. 

13 

Published as a conference paper at ICLR 2025 

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive Network: A Successor to Transformer for Large Language Models, August 2023. URL http://arxiv.org/abs/2307.08621. arXiv:2307.08621 [cs]. 

- Shawn Tan, Yikang Shen, Rameswar Panda, and Aaron Courville. Scattered mixture-of-experts implementation. _arXiv preprint arXiv:2403.08245_ , 2024. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_ , 2017. 

- Xun Wu, Shaohan Huang, and Furu Wei. Mixture of lora experts. _arXiv preprint arXiv:2404.13628_ , 2024. 

- Ted Zadouri, Ahmet Ust¨un, Arash Ahmadian, Beyza Ermis¸, Acyr Locatelli, and Sara Hooker.[¨] Pushing mixture of experts to the limit: Extremely parameter efficient moe for instruction tuning. _arXiv preprint arXiv:2309.05444_ , 2023. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pp. 4791–4800, 2019. 

- Biao Zhang and Rico Sennrich. Root mean square layer normalization. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2 technical report: Accurate 4 bit attention for plug-and-play inference acceleration. _arXiv preprint arXiv:2411.10958_ , 2024a. 

- Jintao Zhang, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen, et al. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. _arXiv preprint arXiv:2410.02367_ , 2024b. 

- Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. Mixture of attention heads: Selecting attention heads per token. _arXiv preprint arXiv:2210.05144_ , 2022. 

- Zexuan Zhong, Mengzhou Xia, Danqi Chen, and Mike Lewis. Lory: Fully differentiable mixture-ofexperts for autoregressive language model pre-training. _arXiv preprint arXiv:2405.03133_ , 2024. 

- Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. _Advances in Neural Information Processing Systems_ , 35:7103–7114, 2022. 

- Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. _arXiv preprint arXiv:2202.08906_ , 2022. 

- Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, and Jianfeng Gao. Taming sparsely activated transformer with stochastic experts. _arXiv preprint arXiv:2110.04260_ , 2021. 

14 

Published as a conference paper at ICLR 2025 

## A STABILITY ANALYSIS OF TOPK AND RELU 

We introduce two metrics, “flip rate” and “flip count”, to evaluate the routing stability: 

**==> picture [279 x 39] intentionally omitted <==**

where _**M** i[l][∈]_[R] _[T][ ×][E]_[denotes the 0-1 mask matrix of the output of the router at layer] _[ l]_[and training] step _i_ , computed using a _fixed_ calibration set of tokens. 

The metric “flip rate” represents the percentage of expert activation states that change (from active to inactive or conversely) in a single update, while “flip count” indicates the average number of experts whose activation states change. 

We measure the two metrics on MoE and ReMoE with _N_ =182M and _E ∈{_ 8 _,_ 16 _,_ 32 _}_ training for 10B tokens. The results are presented in Figure 10, indicating that the ReLU router is more stable than the TopK router: 

**==> picture [396 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.05 0.4<br>MoE_8E MoE_8E<br>0.04 MoE_16E MoE_16E<br>MoE_32E 0.3 MoE _ 32E<br>0.03 ReMoE_8E ReMoE_8E<br>ReMoE_16E 0.2 ReMoE_16E<br>ReMoE_32E ReMoE_32E<br>0.02<br>0.1<br>0.01<br>0.00 0.0<br>0 2 4 6 8 10 0 2 4 6 8 10<br>#Tokens(B) #Tokens(B)<br>Flip Rate Flip Count<br>**----- End of picture text -----**<br>


Figure 10: Flip rate and flip count of MoE and ReMoE 

When _E_ = 8, we find the flip rate of MoE is higher than ReMoE, though the gap narrows as training progresses and the learning rate decreases. While for _E_ = 16 and _E_ = 32, the flip rate of MoE remains consistently 2 _−_ 3 _×_ higher compared to ReMoE throughout training. 

Moreover, the flip count of ReMoE is invariant with respect to _E_ , whereas the flip count of MoE is highly sensitive to the total number of experts and keeps increasing as _E_ grows. 

Notably, the flips in TopK-routed MoE are discontinuous (e.g.(0 _._ 51 _,_ 0) _→_ (0 _,_ 0 _._ 51)), while those in ReLU-routed ReMoE are continuous(e.g.(0 _._ 01 _,_ 0) _→_ (0 _,_ 0 _._ 01)), further underscoring the superiority of the ReLU router. 

- B INSENSITIVITY TO _λ_ 0 AND _α_ 

|_λ_0<br>1_e−_16<br>1_e−_12<br>1_e−_8<br>1_e−_4<br>1<br>Valid Loss<br>2.031<br>2.029<br>2.032<br>2.036<br>2.032<br>Settling time<br>138<br>136<br>110<br>55<br>92_†_<br>_†_ Overshoot observed in 8-92 steps.|_α_<br>1.05<br>1.1<br>1.2<br>1.3<br>1.5|
|---|---|
||Valid Loss<br>2.033<br>2.028<br>2.032<br>2.029<br>2.057_∗_<br>Settling time<br>414<br>211<br>110<br>80<br>52|
||_∗_A large oscillation amplitude in sparsity is observed.|



Table 3: Valid loss and settling time for different values of _λ_ 0 with _α_ = 1 _._ 2. 

Table 4: Valid loss and settling time for different values of _α_ with _λ_ 0 = 1 _e[−]_[8] . 

The ReMoE adaptation algorithm in Equation 7 includes two hyperparameters: _λ_ 0 and _α_ . _Settling time_ , defined as the total number of steps required in Stage I and Stage II (as outlined in Section 3.5), 

15 

Published as a conference paper at ICLR 2025 

is influenced by these parameters. For all experiments, we set _λ_ 0 = 1 _e[−]_[8] and _α_ = 1 _._ 2, but we show that performance remains stable as long as _λ_ 0 is small and _α_ is close to 1. 

Our experiments with _N_ = 182M, _E_ = 8, _G_ = 1, and _k_ = 1 ReMoE models trained for 20k steps ( _∼_ 10B tokens) reveal only minor variations in validation loss for different _λ_ 0 values (Table 3) and _α_ values (Table 4), except for _α_ = 1 _._ 5 which caused rapid regularization changes and excessive oscillation. Besides, although different _λ_ 0 and _α_ values affect settling time, the impact is minor compared to the overall training steps, proving the insensitivity. 

## C PERFORMANCE FOR LONGER TRAINING 

We conduct experiments of training MoE and ReMoE for a longer duration. We experiment with _N_ =469M, _E_ = 8, _k_ = 1 and train the models with a batch size of 4M tokens and training over 120B tokens. The results, as shown in Table 5, indicate that the superiority of ReMoE persists in longer training. 

|Model|Valid Loss|ARC-c<br>ARC-e<br>BoolQ<br>HellaSwag<br>LAMBADA<br>PIQA<br>RACE|Avg.|
|---|---|---|---|
|MoE<br>ReMoE|1.716<br>**1.689**|23.62<br>52.40<br>53.94<br>35.43<br>43.64<br>68.34<br>**31.48**<br>**25.34**<br>**55.22**<br>**55.96**<br>**36.76**<br>**45.82**<br>**68.93**<br>30.43|44.12<br>**45.49**|



Table 5: Performance of training _N_ =469M, _E_ = 8, _k_ = 1 models for 120B tokens. 

## D SPEED COMPARISON OF REMOE AND MOE 

We measure the end-to-end training time for MoE and ReMoE with models of _N_ =469M training over 120B tokens. The time consumption across stages is summarized in Table 6. We find Stage I and Stage II account for _∼_ 1.02% of the total training time and incur _∼_ 0.58% overhead. 

|Model|Stage I<br>Stage II<br>Stage III|Total|
|---|---|---|
|MoE<br>0.12<br>0.41<br>119.12<br>119.65<br>ReMoE<br>0.32<br>0.91<br>119.25<br>120.48|||



Table 6: End-to-end training time comparison across stages (in hours). The time is measured on _N_ = 469M, _E_ = 8, _k_ = 1 models training over 120B tokens. 

|**# Parameters**|**TP**|**Model**|**Train TFLOPS**<br>**Train Diff.**|**Infer TFLOPS**<br>**Infer Diff.**|
|---|---|---|---|---|
|182M|1|MoE<br>ReMoE|103.49<br>↑1.82%<br>105.38|78.47<br>↑2.19%<br>80.19|
|469M|1|MoE<br>ReMoE|138.58<br>↓1.37%<br>136.69|107.52<br>↑3.89%<br>111.71|
|978M|1|MoE<br>ReMoE|160.46<br>↓1.77%<br>157.61|153.11<br>↓0.23%<br>152.76|
|978M|2|MoE<br>ReMoE|133.40<br>↓0.68%<br>132.49|118.55<br>↓1.08%<br>117.27|
|978M|4|MoE<br>ReMoE|103.61<br>↓2.29%<br>101.23|85.96<br>↑2.33%<br>87.96|



Table 7: Throughput comparison between TopK-routed MoE and ReLU-routed ReMoE models. TP indicates the tensor parallel size. Train Diff. and Infer Diff. indicate the relative TFLOPS difference of ReMoE compared to MoE, where ↑ denotes ReMoE is faster, and ↓ denotes it is slower. 

16 

Published as a conference paper at ICLR 2025 

We further measure the throughput of ReMoE against TopK-routed MoE across different model sizes and tensor parallel sizes during Stage III. The results, presented in Table 7, indicate that ReMoE achieves comparable training and inference speeds with MoE, with a minor deviation ranging from _−_ 2 _._ 29% to +3 _._ 89%. This speed consistency is desirable, as ReMoE introduces only a minimal modification to the standard MoE architecture by adjusting the routing function, thereby avoiding additional computational overhead. 

## E DOWNSTREAM EVALUATION RESULTS 

This section provides the detailed downstream evaluation results for the main experiments of scalability of ReMoE in Section 4.3 and ablations on load balancing in Section 5.2. 

## E.1 SCALING IN ACTIVE PARAMETERS _N_ 

The downstream evaluation results for scaling with respect to the parameter count _N_ , as discussed in Section 4.3, are presented in Table 8. These results highlight the performance comparison with increasing model parameters. 

|**Model**<br>_N_|**ARC-c**<br>**ARC-e**<br>**BoolQ**<br>**HellaSwag**<br>**LAMBADA**<br>**PIQA**<br>**RACE**|**Avg.**|
|---|---|---|
|Dense<br>182M<br>469M<br>978M|19.45<br>43.35<br>54.40<br>28.61<br>31.09<br>61.97<br>28.52<br>21.50<br>49.12<br>56.88<br>31.12<br>36.74<br>64.47<br>30.53<br>21.93<br>50.88<br>**60.24**<br>32.42<br>41.06<br>67.46<br>**31.77**|38.20<br>41.48<br>43.68|
|MoE<br>182M<br>469M<br>978M|20.82<br>45.03<br>57.55<br>29.84<br>31.81<br>63.28<br>28.42<br>23.63<br>52.40<br>53.94<br>32.43<br>43.64<br>68.34<br>31.48<br>23.81<br>52.90<br>58.90<br>35.01<br>**44.42**<br>67.90<br>31.48|39.53<br>43.69<br>44.91|
|ReMoE<br>182M<br>469M<br>978M|20.22<br>46.68<br>54.16<br>30.26<br>35.94<br>63.55<br>29.38<br>21.67<br>53.16<br>58.75<br>33.80<br>40.66<br>67.95<br>31.20<br>**24.06**<br>**55.26**<br>57.28<br>**35.93**<br>**44.42**<br>**68.99**<br>30.43|40.03<br>43.88<br>**45.20**|



Table 8: Downstream results of scaling in active parameters _N_ . 

## E.2 SCALING IN EXPERT COUNT _E_ 

Table 9 contains the downstream evaluation results for scaling with respect to the expert count _E_ , as examined in Section 4.3. This analysis illustrates how varying the number of experts influences the overall model effectiveness of MoE and ReMoE. 

|**Model**<br>_E_|**ARC-c**<br>**ARC-e**<br>**BoolQ**<br>**HellaSwag**<br>**LAMBADA**<br>**PIQA**<br>**RACE**|**Avg.**|
|---|---|---|
|Dense<br>-|19.45<br>43.35<br>54.40<br>28.61<br>31.09<br>61.97<br>28.52|38.20|
|MoE<br>4<br>8<br>16<br>32<br>64<br>128|20.73<br>44.49<br>59.63<br>29.14<br>31.40<br>63.33<br>29.19<br>20.82<br>45.03<br>57.55<br>29.84<br>31.81<br>63.28<br>28.42<br>20.90<br>45.29<br>46.36<br>30.50<br>33.22<br>64.96<br>28.33<br>19.54<br>47.35<br>52.29<br>31.12<br>35.63<br>64.25<br>28.23<br>19.88<br>46.63<br>**60.06**<br>31.47<br>36.33<br>65.07<br>28.04<br>**20.99**<br>47.69<br>56.73<br>32.00<br>36.62<br>65.67<br>28.04|39.70<br>39.53<br>38.50<br>39.77<br>41.06<br>41.10|
|ReMoE<br>4<br>8<br>16<br>32<br>64<br>128|19.88<br>46.46<br>57.43<br>29.64<br>33.57<br>62.95<br>27.66<br>20.22<br>46.68<br>54.16<br>30.26<br>35.94<br>63.55<br>29.38<br>20.90<br>49.28<br>53.36<br>30.85<br>37.09<br>65.83<br>**30.05**<br>20.56<br>48.11<br>59.54<br>31.42<br>37.84<br>65.18<br>28.42<br>20.82<br>50.51<br>57.80<br>32.17<br>36.74<br>65.78<br>27.46<br>19.97<br>**51.05**<br>56.97<br>**32.40**<br>**37.92**<br>**66.70**<br>29.86|39.66<br>40.03<br>41.05<br>41.58<br>41.61<br>**42.12**|



Table 9: Downstream results of scaling in expert count _E_ . 

17 

Published as a conference paper at ICLR 2025 

## E.3 SCALING IN GRANULARITY _G_ 

The downstream evaluation results for scaling with respect to the granularity _G_ are shown in Table 10, based on the experiments in Section 4.3. These results demonstrate the superiority of finegrained ReMoE over fine-grained MoE. 

|**Model**<br>_G_|**ARC-c**<br>**ARC-e**<br>**BoolQ**<br>**HellaSwag**<br>**LAMBADA**<br>**PIQA**<br>**RACE**|**Avg.**|
|---|---|---|
|Dense<br>-<br>Dense_×_8<br>-|19.45<br>43.35<br>54.40<br>28.61<br>31.09<br>61.97<br>28.52<br>**22.78**<br>48.11<br>59.66<br>31.11<br>35.65<br>65.02<br>29.57|38.20<br>**41.70**|
|MoE<br>1<br>2<br>4<br>8<br>16<br>32<br>64|20.82<br>45.03<br>57.55<br>29.84<br>31.81<br>63.28<br>28.42<br>21.42<br>46.55<br>54.25<br>29.95<br>32.52<br>64.09<br>28.61<br>20.99<br>46.09<br>55.90<br>30.52<br>35.16<br>63.98<br>29.28<br>21.59<br>47.73<br>60.70<br>30.83<br>36.41<br>64.69<br>28.04<br>19.80<br>48.82<br>57.34<br>30.64<br>36.00<br>64.74<br>28.71<br>21.67<br>48.78<br>57.85<br>31.27<br>**37.10**<br>64.69<br>28.52<br>20.14<br>48.74<br>**61.50**<br>31.03<br>36.31<br>63.93<br>27.85|39.53<br>39.62<br>40.27<br>41.42<br>40.86<br>41.41<br>41.35|
|ReMoE<br>1<br>2<br>4<br>8<br>16<br>32<br>64|20.22<br>46.68<br>54.16<br>30.26<br>35.94<br>63.55<br>29.38<br>20.14<br>47.39<br>57.95<br>30.60<br>34.52<br>63.71<br>28.52<br>20.39<br>47.94<br>55.35<br>31.04<br>36.11<br>64.64<br>29.00<br>20.82<br>48.36<br>60.49<br>30.90<br>36.06<br>63.87<br>28.90<br>21.25<br>**49.41**<br>56.06<br>30.91<br>36.23<br>64.91<br>29.95<br>20.90<br>48.86<br>55.81<br>31.14<br>36.58<br>64.69<br>**30.05**<br>20.65<br>48.74<br>60.06<br>**31.56**<br>36.43<br>**65.40**<br>29.00|40.03<br>40.40<br>40.64<br>41.34<br>41.25<br>41.15<br>41.69|



Table 10: Downstream results of scaling in granularity _G_ . 

E.4 LOAD BALANCING ABLATIONS 

Table 11 presents the downstream evaluation results for the load balancing ablations, as discussed in Section 5.2. These results compare performance with and without load balancing, offering insights into the different roles of load balancing in MoE and ReMoE. 

|**Model**<br>**LB**|**ARC-c**<br>**ARC-e**<br>**BoolQ**<br>**HellaSwag**<br>**LAMBADA**<br>**PIQA**<br>**RACE**|**Avg.**|
|---|---|---|
|Dense<br>-<br>MoE<br>×<br>MoE<br>✓<br>ReMoE<br>×<br>ReMoE<br>✓|19.45<br>43.35<br>54.40<br>28.61<br>31.09<br>61.97<br>28.52<br>19.20<br>44.74<br>50.80<br>28.60<br>30.18<br>62.24<br>27.94<br>20.05<br>45.16<br>**57.83**<br>29.83<br>32.97<br>**63.55**<br>28.33<br>19.45<br>46.34<br>56.94<br>30.19<br>31.79<br>63.33<br>28.61<br>**20.22**<br>**46.68**<br>54.16<br>**30.26**<br>**35.94**<br>**63.55**<br>**29.38**|38.20<br>37.67<br>39.67<br>39.52<br>**40.03**|



Table 11: Downstream results of training with or without load balancing. 

## F DETAILED RESULTS FOR DOMAIN SPECIFICATION 

Figure 11 shows the average routed tokens ratio of MoE and ReMoE across all layers. ReMoE demonstrates significantly stronger domain specialization compared to MoE, where certain experts are more frequently activated for specific domains. This suggests that ReMoE is better at learning and exploiting the unique characteristics of different domains, allowing it to allocate computational resources more effectively. In contrast, MoE exhibits a more uniform expert activation across domains, indicating less differentiation in its expert specialization. 

18 

Published as a conference paper at ICLR 2025 

**==> picture [397 x 580] intentionally omitted <==**

**----- Start of picture text -----**<br>
Arxiv Books C4 Github Stack Wiki<br>Layer 0 Layer 1 Layer 2<br>0.8<br>0.4<br>0.0<br>Layer 3 Layer 4 Layer 5<br>0.8<br>0.4<br>0.0<br>Layer 6 Layer 7 Layer 8<br>0.8<br>0.4<br>0.0<br>Layer 9 Layer 10 Layer 11<br>0.8<br>0.4<br>0.0<br>0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7<br>Expert ID<br>(a) Domain specialization of MoE<br>Arxiv Books C4 Github Stack Wiki<br>Layer 0 Layer 1 Layer 2<br>0.8<br>0.4<br>0.0<br>Layer 3 Layer 4 Layer 5<br>0.8<br>0.4<br>0.0<br>Layer 6 Layer 7 Layer 8<br>0.8<br>0.4<br>0.0<br>Layer 9 Layer 10 Layer 11<br>0.8<br>0.4<br>0.0<br>0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7<br>Expert ID<br>(b) Domain specialization of ReMoE<br>Routed Tokens Ratio<br>Routed Tokens Ratio<br>**----- End of picture text -----**<br>


Figure 11: Detailed results of average routed tokens ratio for MoE and ReMoE in different domains. 

19 

Published as a conference paper at ICLR 2025 

We further analyze the experts in Layer 5 of ReMoE and observe that certain highly related, domainspecific vocabularies are consistently routed to the same expert. To investigate this, we calculate the routing probabilities of different tokens based on their IDs, defined as the ratio of the number of times a specific expert is utilized to the total occurrences of the token. The results are summarized in Table 12. 

Our findings reveal that the vocabularies exhibit clear specialization, reflecting domain-specific characteristics. For example, Expert 1, which is more frequently assigned to natural language domains (e.g., Books, C4), tends to route tokens such as husband, wife, and lover. In contrast, Expert 6, which is associated with non-natural language domains (e.g., Arxiv, Github, StackExchange), predominantly routes code-related tokens like variable, env, and HEAD. 

|Expert ID|Routed Tokens With High Probability|
|---|---|
|||
|0|End(100%);<br>folding(100%);<br>Fill(100%);<br>FILE(100%);<br>NULL(100%);<br>byte(100%);Release(99.36%);Del(99.80%)|
|||
|1|husband(100%); ife(100%); baby(100%); human(100%); lover(99.60%);<br>).(99.86%);),(99.71%);)...(98.425%)|
|||
|2|invest(100%);Fortune(100%);exec(100%);0000(100%);Sorry(100%);<br>bye(97.82%);If(97.74%);®(97.63%)|
|3<br>Conversely(100%); Methods(100%); flower(100%); Blossom(99.93%);<br>Argentina(100%);Georgian(100%);Uruguay(98.90%);African(100%)||
|||
|4|Spring(100%);<br>Summer(100%)<br>Autumn(100%);<br>Winter(100%);<br>seasons(99.02%);Temperature(100%);hot(97.98%);cold(100%)|
|||
|5|`e(100%);æ(99.80%);˚a(98.59%);Æ(97.67%)|
|||
|6|]);(100%);<br>gif(100%);<br>size(100%);<br>variable(100%);<br>env(100%);<br>begin(97.95%);HEAD(97.94%);|(97.83%)|
|||
|7|Kuala(100%);Tus(100%);Lama(100%);Riley(98.94%)|



Table 12: Routed tokens with high probability for experts in Layer 5 of ReMoE 

## G DOMAIN-LEVEL DYNAMIC EXPERT ALLOCATION IN REMOE 

We measure the average active expert count across different domains, as shown in Figure 12, and find that the computation allocation in ReMoE also varies at the domain level. Furthermore, this variation increases in deeper layers closer to the output. This is reasonable because deeper layers tend to capture more abstract and domain-specific features, leading to more pronounced specialization in expert activation. 

**==> picture [397 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
Arxiv Books C4 Github Stack Wiki<br>3.0<br>2.5<br>2.0<br>1.5<br>1.0<br>0.5<br>0.0<br>0 1 2 3 4 5 6 7 8 9 10 11 Avg<br>Layer ID<br>Average Active Expert Count<br>**----- End of picture text -----**<br>


Figure 12: Domain-level dynamic expert allocation 

20 

Published as a conference paper at ICLR 2025 

## H TRAINING MOE WITH NEAR-DENSE WARMUP 

In ReMoE, the training process naturally progresses through three stages, with the first two involving near-dense training where the majority of experts are active. To facilitate a fairer comparison, in Section 4.3, we train the MoE model for additional tokens to match the overall computational cost. In this section, we explore an alternative approach by introducing a similar near-dense warmup phase for MoE, referred to as ”MoE with warmup,” to align its computational footprint with ReMoE across each stage. Specifically, we train the MoE with _N_ = 182M, _E_ = 8, and _k_ = 6—approximately matching the average sparsity of ReMoE during Stages I and II, as depicted in Figure 4a—for the first 100 steps, before transitioning to _k_ = 1 for the remainder of the training process. 

Table 13 compares this warmup variant to both standard MoE and ReMoE. The results indicate that the warmup phase provides a modest improvement in validation loss compared to standard MoE, despite matching the overall computational cost. Nonetheless, ReMoE consistently outperforms both variants. This suggests that the three-stage training pipeline learned by ReMoE, with Stages I and II comprising only the first 100 steps, is beneficial to overall performance. 

|**Model**|**Valid**<br>**Loss**|**ARC-**<br>**c**<br>**ARC-**<br>**e**<br>**BoolQ**<br>**Hella-**<br>**Swag**<br>**LAM-**<br>**BADA**<br>**PIQA**<br>**RACE**|**Avg.**|
|---|---|---|---|
|||||
|MoE<br>MoE<br>with<br>warmup<br>ReMoE|1.936<br>1.928<br>1.921|20.82<br>45.03<br>57.55<br>29.84<br>31.81<br>63.28<br>28.42<br>20.73<br>46.38<br>52.35<br>30.28<br>33.90<br>63.76<br>27.66<br>20.22<br>46.68<br>54.16<br>30.26<br>35.94<br>63.55<br>29.38|39.53<br>39.29<br>40.03|



Table 13: Performance of MoE with near-dense warmup 

We further extend our experiments with MoE using warmup to configurations with larger _E_ , which increases the computational cost of near-dense training. The results, summarized in Table 14, show that as _E_ increases, the warmup setting consistently improves performance. However, ReMoE still outperforms both variants, maintaining a steeper performance scaling with respect to _E_ . 

|**Model,**<br>_E_ =8|**Valid**<br>**Loss**|**Avg.**<br>**Acc.**|**Model,**<br>_E_ =32|**Valid**<br>**Loss**|**Avg.**<br>**Acc.**|**Model,**<br>_E_ =128|**Valid**<br>**Loss**|**Avg.**<br>**Acc.**|
|---|---|---|---|---|---|---|---|---|
||||||||||
|MoE<br>MoE<br>with<br>warmup<br>ReMoE|1.936<br>1.928<br>1.921|39.53<br>39.29<br>40.03|MoE<br>MoE<br>with<br>warmup<br>ReMoE|1.874<br>1.869<br>1.852|39.77<br>40.06<br>41.58|MoE<br>MoE<br>with<br>warmup<br>ReMoE|1.852<br>1.841<br>1.815|41.10<br>41.34<br>42.12|



Table 14: Results for MoE with warmup under different expert count _E_ 

To further investigate the impact of warmup steps on MoE performance, we vary the number of warmup steps for the _E_ = 8 MoE configuration among 50, 100, 500, and 1000. The training curves of these models, along with standard MoE and ReMoE, are shown in Figure 13, and the final validation losses are summarized in Table 15. 

Our results reveal that performance does not improve monotonically with an increasing number of warmup steps, despite the additional computation. This behavior arises due to the discrepancy between the training objectives of _k_ = 6 (warmup phase) and _k_ = 1 (post-warmup phase). For instance, when warmup concludes after 100 steps, the transition between phases is smooth, with the loss changing minimally from 6 _._ 491 _→_ 6 _._ 751. However, extending warmup to 500 or 1000 steps leads to a more pronounced loss gap of 3 _._ 101 _→_ 5 _._ 827 and 2 _._ 695 _→_ 4 _._ 428, respectively. 

21 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.6<br>MoE<br>ReMoE<br>2.4 MoE Warmup 50 Model Warmup Steps Valid Loss<br>MoE Warmup 100<br>0 1.937<br>MoE Warmup 500<br>50 1.930<br>2.2 MoE Warmup 1000<br>MoE 100 1.928<br>500 1.930<br>1000 1.931<br>2.0<br>ReMoE - 1.921<br>0 5 10 15 20 25 30 Table 15: Final validation loss of<br>#Tokens(B) MoE with different warmup steps<br>Train Loss<br>**----- End of picture text -----**<br>


Figure 13: Training curves of MoE with different warmup steps 

In summary, near-dense warmup can enhance the performance of TopK MoE when training from scratch by providing a better initialization for the experts. However, the warmup phase should conclude while the language model loss is still decreasing rapidly. Prolonging the warmup can exacerbate the gap between the warmup and subsequent training phases, ultimately degrading performance. In contrast, ReMoE naturally determines the appropriate warmup steps and sparsity levels due to its continuous and differentiable training dynamics. 

## I FUTURE DIRECTIONS 

This work can be advanced in the following ways: 

- **ReLU Routing for Mixture-of-LoRAs (MoLoRA).** MoLoRA (Zadouri et al., 2023; Wu et al., 2024; Jiao et al., 2024) integrates MoE architectures to manage multiple Low-Rank Adaptation (LoRA) experts, dynamically activating task-specific adapters during inference. ReMoE’s fully differentiable routing mechanism could enhance MoLoRA by enabling smoother transitions between LoRA experts, particularly when adapters are trained on diverse tasks. Using ReLU straightforwardly in MoLoRA is explored in RoDE (Jiao et al., 2024), which can be further enhanced by scaling the expert count while controlling the sparsity as in ReMoE. 

- **ReLU Routing in Product-Key-Memory (PKM) Networks.** PKM (Lample et al., 2019; He, 2024; Berges et al., 2024; Huang et al., 2024) architectures treat individual neurons as ultra-fine-grained experts, leading to routing complexity at unprecedented scales (e.g., millions of experts). ReMoE’s differentiable routing and steep scaling properties are particularly suited to address PKM’s optimization challenges. 

- **Synergy with Efficient Attention Algorithms.** Merging ReMoE’s sparse, conditional feed-forward computation with efficient attention variants—such as quantized (Zhang et al., 2024b;a), linearized (Sun et al., 2023; Gu & Dao, 2023), sparse (Jiang et al., 2025; Gao et al., 2024), or mixture-of-attention (Zhang et al., 2022; Csord´as et al., 2025) mechanisms—could enable Transformers to scale efficiently in both sequence length and model capacity without incurring additional computational overhead. 

- **Dynamic Expert Pruning for ReMoE.** ReMoE’s differentiable training inherently promotes expert specialization, with significant variance in expert importance across domains. This property makes ReMoE more amenable to expert pruning (Lu et al., 2024; Liu et al., 2024a) compared to traditional TopK-routed MoE architectures. 

22 

