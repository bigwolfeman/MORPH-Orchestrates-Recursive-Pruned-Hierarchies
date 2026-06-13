# Hyper-Connections

- **Authors:** Defa Zhu et al. (ByteDance)
- **Year:** 2024 (ICLR 2025)
- **Source:** https://arxiv.org/abs/2409.19606
- **MORPH uses:** Related work / predecessor to mHC. Origin of the multi-stream residual concept. HC uses unconstrained n x n mixing matrices. MORPH's MRR was inspired by the concept of different update rates but uses a much simpler per-channel diagonal approach.

---

Published as a conference paper at ICLR 2025 

## HYPER-CONNECTIONS 

## **Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou** 

Seed-Foundation-Model Team, ByteDance {zhudefa,huanghongzhi.51,huangzihao.notabot,yutao.zeng, maoyunyao.myy,wubanggu,minqiyang,zhouxun}@bytedance.com 

## ABSTRACT 

We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections. This approach specifically addresses common drawbacks observed in residual connection variants, such as the seesaw effect between gradient vanishing and representation collapse. Theoretically, hyperconnections allow the network to adjust the strength of connections between features at different depths and dynamically rearrange layers. We conduct experiments focusing on the pre-training of large language models, including dense and sparse models, where hyper-connections show significant performance improvements over residual connections. Additional experiments conducted on vision tasks also demonstrate similar improvements. We anticipate that this method will be broadly applicable and beneficial across a wide range of AI problems. 

## 1 INTRODUCTION 

**==> picture [396 x 100] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss C4 en val. loss HellaSwag Acc. ARC Challenge Acc.<br>2.400 OLMoE-1B-7BOLMoE-1B-7B-DHCx4 2.85 OLMoE-1B-7BOLMoE-1B-7B-DHCx4 70.0 47.5 OLMoE-1B-7B OLMoE-1B-7B-DHCx4<br>45.0<br>2.375 2.80 67.5<br>42.5<br>2.350 65.0<br>2.75 40.0<br>2.325 62.5 37.5<br>2.70<br>2.3002.275 x1.8 0.027 2.65 x1.8 0.028 60.057.5 OLMoE-1B-7BOLMoE-1B-7B-DHCx4 35.032.5<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (Billions) Tokens (Billions) Tokens (Billions) Tokens (Billions)<br>Loss Loss<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: The performance of the baseline model OLMoE-1B-7B and the model with hyperconnections, OLMoE-1B-7B-DHC _×_ 4. **(1)** and **(2)** show the training loss (0.99 EMA smoothed) and the C4-en validation loss, respectively. Our method converges 1.8 times faster compared to the baseline and maintains a significant advantage at the 500B tokens. **(3)** and **(4)** show the accuracy curves on HellaSwag and ARC-Challenge, demonstrating the superior performance of the OLMoE-1B-7B-DHC _×_ 4 model. 

Deep learning has achieved tremendous success across various domains, where residual connections (He et al., 2016) have been instrumental in contemporary neural network architectures, including transformers and CNNs. Residual connections help mitigate the problem of gradient vanishing, enabling the effective training of very deep networks. However, it is important to acknowledge that residual connections are not infallible solutions and still present limitations that remain unresolved. 

The two main variants of residual connections, Pre-Norm and Post-Norm, each make distinct trade-offs between gradient vanishing and representation collapse. Pre-Norm applies normalization operations to the input before each residual block, effectively addressing the problem of gradient vanishing (Bengio et al., 1994; Glorot & Bengio, 2010). However, it can also lead to the issue of collapse in deep representations (Liu et al., 2020), where hidden features in deeper layers become highly similar, diminishing the contribution of additional layers as their number increases. In contrast, Post-Norm applies normalization after the output of each residual block, reducing the influence of a hidden state on subsequent layers. This approach can alleviate the issue of representation collapse but 

1 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
:  scalar, weights of hyper-connections :  hidden vectors layer :  attention or FFN<br>+ 𝛽# + 𝛽! + 𝛽# + 𝛽! + + +<br>layer layer layer layer<br>+ +<br>𝛼#,$ 𝛼!,$ 𝛼#,# 𝛼!,# 𝛼#,! 𝛼!,! 𝛼#,# 𝛼!,! 𝛼#,$ 𝛼!,$ 𝛼#,# 𝛼!,# 𝛼#,! 𝛼!,!<br>ℎ h# h! h# h! h# h!<br>(a) Residual connections (b) Hyper-connections (c) Depth-connections (d) Width-connections<br>**----- End of picture text -----**<br>


Figure 2: **Hyper-connections (HC) with an expansion rate of** _n_ = 2 **.** (a) Residual connections. (b) Hyper-connections: _β_ 1, _β_ 2, _α_ 0 _,_ 0, _α_ 0 _,_ 1, _α_ 1 _,_ 0, _α_ 1 _,_ 1, _α_ 2 _,_ 1, and _α_ 2 _,_ 2 are learnable scalars or scalars predicted by the network , depending on the specific HC version. These connections enable lateral information exchange and vertical integration of features across depths. The Transformer with HC is shown in Fig. 17. They can be decoupled into depth-connections and width-connections. (c) Depth-connections perform a weighted sum between the layer output and the hidden vector _h_ 1. (d) Width-connections allow information exchange between the hidden vectors _h_ 1 and _h_ 2. 

also reintroduces the problem of vanishing gradients. The vanishing gradient and the representation collapse are like two ends of a seesaw, with these two variants making respective trade-offs between these issues. The key issue is that residual connections, including both Pre-Norm and Post-Norm variants, predefine the strength of connections between the output and input within a layer. 

Driven by the limitations of residual connections, an important question arises: _Can neural networks autonomously learn the optimal strength of connections to improve performance?_ To address this, we propose hyper-connections (HC), which lead to significantly improved performance with a negligible increase in computation and parameters. We will show that both PostNorm and Pre-Norm variants can be expressed as specific nontrainable forms of hyper-connections, as discussed in § 3.1. 

The core idea of hyper-connections (HC) is to propose learnable _depth-connections_ and _width-connections_ , as depicted in Fig.2 (b). These connections flexibly integrate features vertically across depths, compared to the residual connections shown in Fig.2 (a). Depth-connections can be considered as a generalized residual connections, assigning weights to the connections between the inputs and outputs of each layer. To enable the network to model different depth-connections simultaneously, we expand the network’s input into _n_ copies, each having its own depth connection, as shown in Fig. 2 (b). This design allows multiple hidden vectors to reserve multiple patterns connecting preceding layers, as shown in § 4.5. Moreover, we establish width connections between the _n_ hidden vectors, allowing information exchange between hidden vectors within the same layer, as shown in Fig. 2 (b). We argue that _n_ ( _>_ 1) hidden states are necessary. As analyzed in Appendix F, the seesaw effect persists when _n_ = 1, and experiments show that it does not improve performance, as shown in Fig. 5. In contrast, when _n >_ 1, hyper-connections can not only learn to adjust the 

**==> picture [143 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>0.8<br>0.6<br>0.4<br>0.2 Pre-Norm<br>Hyper-Connection<br>0.0<br>0 5 10 15 20 25 30<br>Layer Index  i<br>)<br>+1 i 0<br> h ,<br>i 0<br>h<br>(<br>cos<br>**----- End of picture text -----**<br>


Figure 3: Cosine similarity between the input of the current and the previous layers for the OLMo-1B models (Groeneveld et al., 2024). The curve represents the median of similarity, while the shaded area indicates the range between the 5th and 95th percentiles. The red curve shows the model with Pre-Norm, and the blue curve shows that with hyper-connections. 

strength of residuals but also rearrange layers, either sequentially or in parallel, as discussed in § 3.2. To further enhance flexibility, we introduce dynamic hyper-connections (DHC), enabling the network to adjust connection weights according to the input. Notably, although HC seem to increase the network’s width by _n_ times, the additional parameters and computational cost are almost negligible, as analyzed in Appendix B. The Transformer with HC is shown in Fig. 17. 

Our research, primarily centered on large language models (LLMs) pre-training, also extends to visual generation and classification tasks. Using Pre-Norm as a baseline, we demonstrate the significant benefits of hyper-connections, including 1B and 7B dense models as well as 7B MoE models, as 

2 

Published as a conference paper at ICLR 2025 

detailed in § 4. The benefits are particularly prominent for OLMoE (Muennighoff et al., 2024) as presented in Fig.1. The model utilizing DHC converges **1.8** times faster and shows an improvement of **6** points on ARC-Challenge compared to the baseline trained with 500 B tokens. According to our visualization analysis, as shown in Fig.3, the baseline model tends toward representation collapse, characterized by high similarity between features of adjacent layers. In contrast, models with HC exhibit significantly lower similarity between features across adjacent layers and a wider range of similarities. This suggests that HC enhance the impact of each layer. Further discussion is provided in §4.5 and in Appendix F. These compelling pieces of evidence demonstrate the generality of the hyper-connections principle, and we anticipate their applicability in numerous other AI challenges. 

## 2 METHOD 

## 2.1 STATIC HYPER-CONNECTIONS 

Consider the hidden vector **h** _[k][−]_[1] _∈_ R _[d]_ (or **h** _[k][−]_[1] _∈_ R _[d][×]_[1] ) as the input to the _k_ -th layer, with the initial input **h**[0] to the network. Initially, **h**[0] _∈_ R _[d]_ is replicated _n_ times to form the initial _hyper hidden matrix_ **H**[0] = � **h**[0] **h**[0] _. . ._ **h**[0][�][⊺] _∈_ R _[n][×][d]_ . Here, _n_ is the expansion rate. For the _k_ -th layer, the input consists of the hyper hidden matrix from the previous layer **H** _[k][−]_[1] = � **h** _[k]_ 1 _[−]_[1] **h** _[k]_ 2 _[−]_[1] _. . ._ **h** _[k] n[−]_[1] �⊺ _∈_ R _n×d_ . Finally, we sum the last hyper hidden matrix row-wise to obtain the required hidden vector, which is then passed through a final projector to produce the final output of the network (i.e., a normalization layer and an unembedding layer in transformers). To simplify the notation in subsequent analysis, we omit the layer index and simply denote the hyper-hidden matrix as **H** = ( **h** 1 **h** 2 _. . ._ **h** _n_ )[⊺] . 

The hyper-connections (HC) can be represented by a matrix _HC_ , where each element defines the connection weight. The matrix is structured as follows: 

**==> picture [356 x 62] intentionally omitted <==**

Consider a network layer _T_ , it integrates self-attention layers and feed-forward networks within transformers. The output of the HC, denoted by **H[ˆ]** , can be simply formulated as follows: 

**==> picture [292 x 13] intentionally omitted <==**

We use **Am** as weights to perform a weighted sum on the input **H** = ( **h** 1 **h** 2 _. . ._ **h** _n_ )[⊺] to obtain the input **h** 0 of the current layer _T_ , which is given by: 

**==> picture [229 x 12] intentionally omitted <==**

While **Ar** is used to connect **H** and map it to a hyper hidden matrix **H** _[′]_ , as shown below: 

**==> picture [227 x 11] intentionally omitted <==**

Subsequently, the output is given by: 

**==> picture [246 x 13] intentionally omitted <==**

The **depth-connections** can be decoupled as the following matrix, which is shown at Fig 2 (a): 

**==> picture [320 x 25] intentionally omitted <==**

where the first row **B** represents the weights of the output of the current layer _T_ , and the last row diag( **Ar** ) represents the weights of the input. We use diag( **Ar** ) to represent the flatten vector of the diagonal entries of **Ar** . 

3 

Published as a conference paper at ICLR 2025 

The **width-connections** matrix can be defined as follows, which is shown at Fig 2 (b): 

**==> picture [266 x 13] intentionally omitted <==**

The algorithm that employs hyper-connections is presented in Algorithm 1. 

## 2.2 DYNAMIC HYPER-CONNECTIONS 

The entries of _HC_ can dynamically depend on the input **H** , which the matrix representation of dynamic hyper-connections (DHC) is defined as follows: 

**==> picture [265 x 25] intentionally omitted <==**

Similarly, given a layer _T_ and input **H** , we obtain the output of the DHC as follows: 

**==> picture [242 x 13] intentionally omitted <==**

In practice, we combine the dynamic and static matrices to achieve DHC. The dynamic parameters are obtained through a linear transformation. To stabilize the training process, we introduce normalization before the linear transformation and apply the tanh activation function after it, scaling it by a small initial learnable factor. The following equations detail how these dynamic parameters are computed: 

**==> picture [271 x 11] intentionally omitted <==**

**==> picture [294 x 44] intentionally omitted <==**

Our experiments in § 4 demonstrate that dynamic hyper-connections outperform static hyperconnections in language modeling tasks. The PyTorch implementations for both the static and dynamic variants of hyper-connections are detailed in Algorithm 2 and 3. 

## 2.3 INITIALIZATION 

In order to make the initialization of the hyper-connections equivalent to the Pre-Norm residual connections, we adopt the following initialization strategy. The dynamic parameters **W** _β_ , **W** _m_ , and **W** _r_ in Eqs. 11, 12, and 13 are initialized to 0, while the static matrices are initialized as follows: 

**==> picture [281 x 25] intentionally omitted <==**

where _k_ is the index of the layer. mod denotes the modulo operation. 

## 3 WHY HYPER-CONNECTIONS 

In this section, we elucidate the rationale behind hyper-connections. We explore how variants of residual connections, namely Pre-Norm and Post-Norm, can be viewed as non-trainable hyperconnections, and introduce the concept of sequential-parallel duality, demonstrating how hyperconnections can dynamically optimize layer arrangements to enhance network performance. A visulize analysis of hyper-connections through an unfolded view is discussed in § 4.5. 

## 3.1 RESIDUAL CONNECTIONS AS NON-TRAINABLE HYPER-CONNECTIONS 

The Pre-Norm and Post-Norm residual connections can be represented as the following hyperconnections matrices with an expansion rate _n_ = 1: 

4 

Published as a conference paper at ICLR 2025 

**==> picture [352 x 37] intentionally omitted <==**

where _σi_ and _σo_ denote the standard deviations of the input and output of the neural network layer, respectively, and _σio_ is the covariance between them. 

For Pre-Norm, its hyper-connection matrix is a 2 _×_ 2 matrix where the bottom right triangular part is filled with 1 and the rest is a placeholder 0. For Post-Norm, the weights depend on the variances and covariance of the input and output, forming a 2 _×_ 2 matrix. Therefore, their hyper-connection matrices are non-trainable. In this work, we propose hyper-connections that can be ( _n_ + 1) _×_ ( _n_ + 1) matrices, with weights that are trainable or even predicted based on the input. The complete derivation is provided in Appendix G. 

## 3.2 SEQUENTIAL-PARALLEL DUALITY 

Given a series of neural network modules, we have the option to arrange them either sequentially or in parallel. However, hyper-connections offer an approach that learns to rearrange these layers in a configuration blending both sequential and parallel arrangements. 

**==> picture [188 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
+<br>=<br>+ + +<br>layer 2 layer 2<br>=<br>+ + +<br>layer 1 layer 1<br>+<br>= =<br>(a) Sequential Arrangement  (b) Parallel Arrangement<br>**----- End of picture text -----**<br>


Figure 4: Sequential and parallel arrangements of hyper-connections with _n_ = 2. 

Without loss of generality, we set the expansion rate to _n_ = 2. If the hyper-connections are learned as the following matrix, the neural network will be arranged sequentially: 

**==> picture [241 x 31] intentionally omitted <==**

In this case, the depth connection degenerates into a residual connection, as shown in Fig. 4 (a). 

When the hyper-connections for odd and even layers (with layer numbering starting from 1) are defined by the following matrices, the neural network will be arranged in parallel every two consecutive layers, similar to the arrangement of parallel transformer blocks in transformers (Wang, 2021), as shown in Fig. 4 (b). The general and complete derivation is provided in Appendix H. 

**==> picture [349 x 31] intentionally omitted <==**

Thus, learning the hyper-connection matrix in various forms can create layer arrangements that surpass traditional sequential and parallel configurations, resulting in a soft-mixture or even dynamic 

5 

Published as a conference paper at ICLR 2025 

arrangement. For static hyper-connections, the layer arrangement within the network remains fixed after training. In contrast, dynamic hyper-connections allow the arrangement to adapt dynamically for each token. 

## 4 RESULTS 

**==> picture [397 x 117] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss vs Tokens Training Loss vs Tokens<br>2.60 OLMo-1B-baseline 2.60 OLMo-1B-baseline<br>OLMo-1B-DHCx1 OLMo-1B-DHCx1 W/O tanh<br>2.55 OLMoOLMo-1B-DHCx4-1B-DHCx2 2.55 OLMo-1B-DHCx2  W/O tanhOLMo-1B-DHCx4  W/O tanh<br>OLMo-1B-DHCx8 OLMo-1B-DHCx8  W/O tanh<br>2.50 2.50<br>2.45 2.45<br>2.40 2.40<br>100 150 200 250 300 350 400 450 500 100 150 200 250 300 350 400 450 500<br>Tokens (Billions) Tokens (Billions)<br>Training Loss Training Loss<br>**----- End of picture text -----**<br>


Figure 5: Comparison of training loss curves for different expansion rate. The left subfigure includes models with dynamic hyper-connections (DHC) at various expansion rates, while the right subfigure shows the effect of omitting the tanh function. Both subfigures illustrate how increasing the expansion rate leads to improved training loss performance over 500B tokens. Results are smoothed using an exponential moving average with a coefficient of 0.99. 

Table 1: Ablation study on expansion rates _n_ with training on 500 B tokens. 

|**Methods**||**V2 Eval**<br>**Loss**_↓_|**V2 Eval**<br>**PPL**_↓_|**V3 Eval**<br>**Loss**_↓_|**V3 Eval**<br>**PPL**_↓_|**Down Stream**<br>**Avg, Acc.** _↑_|
|---|---|---|---|---|---|---|
|OLMo-1B||2.811|18.023|2.544|14.229|62.5|
|OLMo-1B-DHC_×_1|W/O tanh|2.822|18.270|2.556|14.428|62.3|
|OLMo-1B-DHC_×_2|W/O tanh|2.792|17.663|2.537|14.033|63.8|
|OLMo-1B-DHC_×_4|W/O tanh|2.779|17.451|2.516|13.844|**64.4**|
|OLMo-1B-DHC_×_8|W/O tanh|**2.777**|**17.425**|**2.514**|**13.819**|63.8|
|OLMo-1B-DHC_×_1||2.819|18.125|2.556|14.418|62.3|
|OLMo-1B-DHC_×_2||2.802|17.950|2.534|14.114|63.0|
|OLMo-1B-DHC_×_4||2.781|17.509|**2.514**|**13.826**|**63.8**|
|OLMo-1B-DHC_×_8||**2.778**|**17.445**|2.516|13.843|62.8|



We primarily conduct experiments on pre-training of large language model, including dense and Mixture-of-Experts (MoE) (Shazeer et al., 2017) models, and extend to visual generation and classification tasks. Due to space constraints, we include the vision experiments in the Appendix E. 

**Experiment Settings.** We employ the experimental setup outlined by OLMo (Groeneveld et al., 2024) for dense models and by OLMoE (Muennighoff et al., 2024) for MoE models. **For dense models** , we use dolmap-v1.5-sample (Soldaini et al., 2024) as our training dataset. We conduct ablation studies on 1B models and assess the effectiveness of our method at the 7B model scale. **For MoE models** , we train the OLMoE-1B-7B model, both with and without hyper-connections, on the OLMOE-MIX dataset. These models activate 1 _._ 3B out of a total of 7B parameters. All experiments are trained on 500B tokens. 

**Implementation.** We maintain the training configuration of the baseline model, replacing the residual connections with hyper-connections. The static component in Eqs. 1, 11, 12, 13 does not utilize weight decay, whereas the dynamic component does. Since the hyper hidden vectors of the final transformer block are ultimately summed, we ensure that the standard deviation (std) of the output (before the final layernorm and unembedding layers) remains consistent with the original. At initialization, we scale the std of the weights of the output module at all layers, including those of the second linear layer of the feedforward network and the output projector of the attention module, by a factor of _[√] n_ , 

6 

Published as a conference paper at ICLR 2025 

where _n_ represents the expansion rate. The parameters and computational overhead introduced by hyper-connections is negligible, see Table. 7 and 8. 

**Metrics.** In accordance with the methodology of OLMo (Groeneveld et al., 2024), we report the average perplexities (PPL) and losses on both the V2 and V3 validation sets, along with the average metrics for zero-shot evaluation on downstream benchmarks (refer to Table 13). We observe significant volatility in the zero-shot performance indicators for the datasets (highlighted in grey in Table 13), with fluctuations exceeding 20% across neighboring checkpoints. For more reliable and consistent results, we excludes these volatile datasets from our analysis. For the MoE models, in line with OLMoE, we also present losses on V3 validation sets, and accuracies on downstream benchmarks (refer to Table 14). 

## 4.1 ABLATION STUDY 

We use the dynamic hyperconnections with an expansion rate of _n_ = 4 and include the tanh function as the default method, marked with the suffix -DHC, while -SHC denotes static hyper-connections. 

The evaluation results are presented in Table 1, and the training loss curves are depicted in Fig. 5. We observe that with an expansion rate of _n_ = 1, the performance of DHC is inferior to the baseline. However, for _n >_ 1, DHC significantly outperforms the baseline, achieving superior results at _n_ = 4, with the increase to _n_ = 8 providing minimal additional benefits. Notably, OLMo-1B-DHC _×_ 8 W/O tanh excels on both V2 and V3 validation sets, with a reduction in V2 Eval Loss by **0.034** and V3 Eval Loss by **0.029** compared to the baseline. Furthermore, the decline rate of training losses for DHC ( _n ≥_ 2) is steeper than that of the baseline, and DHC demonstrates greater stability, with no spikes observed in any DHC experiments. 

**Static and dynamic hyper-connections.** Table 2 presents an ablation study comparing SHC and DHC. All hyper-connection (HC) variants significantly outperform the baseline. At an expansion rate of 2, the improvements of DHC and SHC are similar. However, at an expansion rate of 4, DHC performs notably better than SHC. 

Table 2: Ablation study on static and dynamic hyper-connections with training on 500 B tokens. 

|**Methods**||**V2 Eval**<br>**Loss**_↓_|**V2 Eval**<br>**PPL**_↓_|**V3 Eval**<br>**Loss**_↓_|**V3 Eval**<br>**PPL**_↓_|**Down Stream**<br>**Avg, Acc.** _↑_|
|---|---|---|---|---|---|---|
|OLMo-1B||2.811|18.023|2.544|14.229|62.5|
|OLMo-1B-SHC_×_2||2.799|17.778|2.538|14.152|63.4|
|OLMo-1B-DHC_×_2||2.802|17.950|2.534|14.114|63.0|
|OLMo-1B-DHC_×_2|W/O tanh|**2.792**|**17.663**|**2.529**|**14.033**|**63.8**|
|OLMo-1B-SHC_×_4||2.791|17.671|2.528|14.025|63.6|
|OLMo-1B-DHC_×_4||2.781|17.509|**2.515**|**13.826**|63.8|
|OLMo-1B-DHC_×_4|W/O tanh|**2.779**|**17.451**|2.516|13.844|**64.4**|



**The importance of B and** _WC_ **.** As shown in Table 3, not training _WC_ leads to significant performance declines, with the V2 loss increasing by **0.021** and the V3 loss by **0.017** , as seen when comparing the 4th and 6th lines of Table 3. In contrast, the impact is less pronounced when **B** is not trained. Therefore, ensuring the trainability of both _WC_ and **B** is crucial. 

## 4.2 COMPARISON WITH RELATED WORKS 

We implemented the Altup (Baykal et al., 2024) and ResiDual (Xie et al., 2023) methods in OLMo. Altup is motivated to widen the hidden dimension while maintaining low computation cost by passing only a part of hidden state to transformer blocks. By contrast, ResiDual is proposed to combine both Pre- and Post-Norm in a two-stream style. Both methods expand the hidden size by _n_ times with negligible computational overhead, with ResiDual expanding it exactly 2 times. For a fair comparison, we set _n_ = 2 in our experiments. Unfortunately, although these methods show gains in the early stages of training, they are gradually surpassed by the baseline, as demonstrated by the results in Table 4 and the training loss curves in Fig. 15. 

7 

Published as a conference paper at ICLR 2025 

Table 3: Ablation study on OLMo-1B-DHC _×_ 4. In the **B** or _WC_ column, the symbol "✗" denotes parameters that are not trainable from initialization. 

|_WC_<br>**B**<br>**Tanh**|**V2 Eval**<br>**Loss**_↓_<br>**V2 Eval**<br>**PPL**_↓_<br>**V3 Eval**<br>**Loss**_↓_<br>**V3 Eval**<br>**PPL**_↓_<br>**Down Strea**<br>**Avg, Acc.**|
|---|---|
|||
|✗<br>✓<br>✗<br>✓<br>✗<br>✗<br>✓<br>✓<br>✗|2.804<br>17.912<br>2.537<br>14.145<br>62.5<br>2.781<br>17.493<br>2.518<br>13.874<br>63.6<br>**2.779**<br>**17.773**<br>**2.516**<br>**13.823**<br>**64.4**|
|||
|✗<br>✓<br>✓<br>✓<br>✗<br>✓<br>✓<br>✓<br>✓|2.802<br>17.914<br>2.532<br>14.072<br>63.4<br>2.783<br>17.504<br>2.520<br>13.906<br>63.4<br>**2.781**<br>**17.835**<br>**2.515**<br>**13.807**<br>**63.8**|



Table 4: Performance of related methods on OLMo-1B models. 

|**Methods**|**V2 Eval**<br>**Loss**_↓_|**V2 Eval**<br>**PPL**_↓_|**V3 Eval**<br>**Loss**_↓_|**V3 Eval**<br>**PPL**_↓_|**Down Stream**<br>**Avg, Acc.** _↑_|
|---|---|---|---|---|---|
|OLMo-1B|2.811|18.023|2.544|14.229|62.5|
|OLMo-1B-ResiDual|2.825|18.375|2.551|14.346|62.0|
|OLMo-1B-Altup_×_2|2.827|18.268|2.558|14.454|62.4|
|OLMo-1B-DHC_×_2|2.802|17.950|2.534|14.114|63.0|
|OLMo-1B-DHC_×_2W/O tanh|**2.792**|**17.663**|**2.529**|**14.033**|**63.8**|



## 4.3 7B MODELS 

**==> picture [396 x 80] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss C4-en Loss HellaSwag Acc. SciQ Acc.<br>2.4 OLMo-7B OLMo-7B-DHCx4 2.7 OLMo-7BOLMo-7B-DHCx4 70 9290<br>65<br>2.3 2.6 88<br>60 86<br>2.2 2.5 55 OLMo-7B OLMo-7B-DHCx4 8482 OLMo-7B OLMo-7B-DHCx4<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (Billions) Tokens (Billions) Tokens (Billions) Tokens (Billions)<br>Loss Loss<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 6: **(1)** and **(2)** Training loss (0.99 EMA smoothed) and C4-en validation loss for OLMo-7B and OLMo-7B-DHC _×_ 4 models. **(3)** and **(4)** Accuracy curves on hellaswag and sciq, demonstrating the superior performance of the OLMo-7B-DHC _×_ 4 model. 

We evaluate the effectiveness of hyper-connections on the 7B model, training a model with DHCs with an expansion rate of 4, denoted as OLMo-7B-DHC _×_ 4. According to Table 5, OLMo-7B-DHC _×_ 4 significantly outperforms the baseline OLMo-7B model in all average metrics. In the V2 evaluation, OLMo-7B-DHC _×_ 4 shows improvements of **0.022** for loss and **0.293** for PPL. Furthermore, the average score of downstream benchmarks **0.710** surpasses the baseline 0.701, with the results of specific tasks shown in Fig. 10. 

Based on Fig 6, the OLMo-7B-DHC _×_ 4 model consistently shows better metrics compared to baseline, including training and validation loss and accuracy in downstream benchmarks. Notably, after 400 B tokens, the model maintains its improvement without the gains diminishing. This indicates that the OLMo-7B-DHC _×_ 4 model continues to provide consistent benefits in reducing loss, even at higher token counts. Furthermore, according to Fig. 6, the baseline model exhibits frequent spikes, while our model with DHCs shows no spikes throughout the training. This shows that our approach not only achieves better loss but also ensures more stable training. 

## 4.4 MOE MODELS 

We evaluate the effectiveness of hyper-connections on the Mixture-of-Experts (MoE) model. We retrain the original OLMoE-1B-7B model as the baseline and train a model that applies Dynamic 

8 

Published as a conference paper at ICLR 2025 

Table 5: Performance of 7B models. FLOPs refers to the computation per token in the forward pass. 

|**Methods**|**Params**<br>**(B)**|**FLOPs**<br>**(G)**|**V2**<br>**Loss**_↓_|**V2**<br>**PPL**_↓_|**V3**<br>**Loss**_↓_|**V3**<br>**PPL**_↓_|**Tasks Avg.**<br>**Acc.** _↑_|
|---|---|---|---|---|---|---|---|
|OLMo-7B|6.9|13.36|2.581|14.316|2.322|11.324|70.1|
|OLMo-7B-DHC_×_4|6.9|13.38|**2.559**|**14.023**|**2.304**|**11.120**|**71.0**|



Hyper-Connections (DHC) with _n_ = 4, replacing the residual connections. The full results are shown in Fig. 9, which illustrates that hyper-connections outperform residual connections in almost all metrics. In many metrics, our method requires only **half** of the training tokens to achieve the same performance as the baseline. Fig. 1 and Table 6 highlight some of the results, such as a reduction in training loss of approximately **0.027** , a reduction in loss on the C4-en validation set of **0.028** , an improvement of **6** points on the ARC-Challengeand an improvement of **1.2** points on MMLU Var. 

Table 6: Downstream evaluations for MoE models training with 500B tokens under the OLMoE evaluation setting. ARC-C stands for ARC-Challenge, and ARC-E for ARC-Easy. MMLU Var is a modified version of MMLU that includes varying few-shot examples, providing stable feedback during early training, as outlined in the OLMoE setting (Muennighoff et al., 2024). 

|**Methods**|**MMLU**<br>**Var**|**Hella-**<br>**Swag**|**ARC-C**|**ARC-E**|**PIQA**|**Wino-**<br>**Grande**|**BoolQ**|
|---|---|---|---|---|---|---|---|
|OLMoE-1B-7B|38.5|69.5|41.8|72.8|77.6|64.4|65.4|
|OLMoE-1B-7B-DHC_×_4|**39.7**|**70.2**|**47.8**|**76.7**|**78.2**|**64.6**|**68.5**|



## 4.5 VISUALIZATION ANALYSIS 

**==> picture [394 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32<br>0 0 0 0 0 1.0<br>4 4 4 4 4<br>8 8 8 8 8 0.5<br>12 PTB 12 12 12 12<br>16 16 16 16 16 0.0<br>20 20 20 20 20<br>24 24 24 24 24 0.5<br>28 28 28 28 28<br>32 32 32 32 32 1.0<br>Hyper-Connection Post-Norm Pre-Norm Pre-Norm PTB Two-hop Residual<br>SIN NNN<br>**----- End of picture text -----**<br>


Figure 7: Visualization of connection matrices for hyper-connections and various related baseline methods. The attention layers, which have odd ids, are marked with green tick marks. 

In this section, we investigate the learned hyper-connection weights and show how the output of the former layer contributes to the latter ones. To this end, we convert hyper-connections to dense connections cross layers. Consider the input hidden vectors **h** _[k]_ 0[in] _[ k]_[-th layer, it can be unfolded as a] weighted summation over previous layer outputs: 

**==> picture [244 x 31] intentionally omitted <==**

where _c_[(0)] _kj_[describes how much layer-] _[j]_[(] _[T][j]_[) contributes to layer-] _[k]_[’s input] **[ h]** _[k]_ 0[.][Then,] **[ C]**[(0)][ denotes] a dense connection weight matrix. In particular, let layer-0 be the word embedding and _T_[0] be an identity mapping, layer- _L_ +1 be the hidden state before the unembedding layer, which is a summation over the last hidden vectors, i.e., **h** _[L]_ 0[+1] = _j_ **[h]** _[L] j_[.] 

OLMo-1B-DHC _×_ 4 model is adopted for visualization. We take the checkpoint at 500B tokens and forward random validation text to obtain dynamic hyper-connection weights. In addition, we show connection patterns for some related baseline methods. Finally, the visualization is illustrated in Fig. 13. We present the following findings, with more detailed discussions provided in Appendix F. 

9 

Published as a conference paper at ICLR 2025 

**Connection patterns for baseline methods.** For Pre-Norm baseline, the connection matrix is simply a lower triangular matrix with diagonal elements erased, because each transformer layer joins the residual equally. In the Pre-Norm parallel transformer block (PTB) baseline, the connection matrix appears jagged because the input to the FFN layer does not depend on the output of the previous attention layer. For Post-Norm baseline, the connection only holds for adjacent layers, as the weight for bottom layers decays every time the residual passes a post-norm layer. For the two-hop residual baseline (Ma et al., 2024), the outputs of attention layers are not added to residual and only contributes to the next one FFN layer, resulting in a vertical strip pattern in the connection matrix. 

> Λ **-shaped connection pattern.** In the connection matrix for hyper-connections, a long-term decay pattern can be observed, where layers are generally preferred to rely on a few adjacent layer outputs. Moreover, the bottom layers (e.g. layer 0,2) are observed frequently used in most of subsequent layers. Therefore, the two patterns together form a Λ-shaped connection pattern. Note that the long-term decay pattern is a Post-Norm style pattern, while the frequently accessed pattern is Pre-Norm style, indicating that the hyper-connection introduces a free mixture of Pre- and Post-Norm architecture. 

**Input word embedding is eliminated from model output.** As per the first column in the connection matrix for layer inputs, the input word embedding contributes to most of the layers except for the final one. This last layer, which products the model’s output, is used for next token prediction. In most cases, keeping a component of input embedding in model output is harmful to next token prediction, especially when using a tied word embedding such as that employed by OLMo-1B. Similar results are found in previous works (Ma et al., 2023). 

**Parallel transformer blocks are observed.** As discussed in § 3.2, parallel transformer block, which performs attention and FFN in parallel, is a special case for hyper-connection. In practice, PTB-like patterns, which can be identified by the local jagged pattern, are surprisingly observed to be learned by hyper-connections. For instance, layer 11 has a minimal contribution to the input of layer 12 (refer to row 12 in the hyper-connection connection matrix). This suggests that layers 11 and 12 can operate in parallel, thereby forming a PTB module. 

**Attention layers tend to have fewer long-term connections.** It is observed that attention layers at the bottom barely have long-term contribution, a trend that persists until layer 17. Upon examining the connection matrix for hyper hiddens (refer to Fig. 13 in the appendix), it’s evident that the outputs of the FFN layers have significantly greater magnitudes than those of the attention layers. This pattern resembles a two-hop residual connection design, wherein the attention output contributes to the input of the following FFN layer, but doesn’t join the main residual path. 

## 5 RELATED WORK 

**Transformers** (Vaswani et al., 2017) have revolutionized various fields, particularly natural language processing and computer vision. They rely heavily on residual connections to facilitate the training of deep models. Our hyper-connections approach can replace residual connections, providing stable training and consistent improvements in both natural language processing and computer vision. 

**The issues of gradient vanishing and representation collapse** (Bengio et al., 1994; Glorot & Bengio, 2010; Liu et al., 2020) have been extensively studied. The combinations of normalization techniques (Ioffe & Szegedy, 2015; Ba et al., 2016) and residual connections (He et al., 2016), like Pre-Norm and Post-Norm, actually reflects different emphases in solving these two issues. However, despite these advancements, the fundamental trade-off between gradient vanishing and representation collapse in deep networks remains a critical challenge. Building on these findings, our work introduces a novel approach that enables neural networks to autonomously learn the optimal strength of connections, potentially improving both gradient stability and representation quality. 

## 6 CONCLUSION 

In conclusion, we have introduced hyper-connections as an effective alternative to residual connections in transformers. Our analysis reveals that hyper-connections not only overcome the limitations of residuals but also enable dynamic adjustments in network architecture. Experimental results confirm their promising benefits across various tasks, including pre-training of large language model, image generation, and image classification. 

10 

Published as a conference paper at ICLR 2025 

## ACKNOWLEDGEMENTS 

This research was conducted at ByteDance Inc. We are grateful for the suggestions and assistance provided by Yaowei Zheng, Yuyu Zhang, Yunshui Li, Xiang Li, Bairen Yi, Zhenyi Lu and Xintian Han. 

## REFERENCES 

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. In _arXiv preprint arXiv:1607.06450_ , 2016. 

- Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, and Xin Wang. Alternating updates for efficient transformers. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. _IEEE transactions on neural networks_ , 5(2), 1994. 

- Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, 2020. 

- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. _arXiv preprint arXiv:1905.10044_ , 2019. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv:1803.05457v1_ , 2018. 

- Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In _Machine learning challenges workshop_ . Springer, 2005. 

- Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In _proceedings of Sinn und Bedeutung_ , volume 23, 2019. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ . Ieee, 2009. 

- Bill Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In _Third international workshop on paraphrasing (IWP2005)_ , 2005. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In _Proceedings of the thirteenth international conference on artificial intelligence and statistics_ . JMLR Workshop and Conference Proceedings, 2010. 

- Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. _arXiv preprint arXiv:2402.00838_ , 2024. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016. 

11 

Published as a conference paper at ICLR 2025 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021. 

- Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In _International conference on machine learning_ . PMLR, 2015. 

Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions. 2017. 

- Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. _arXiv preprint arXiv:2205.05198_ , 2022. 

- Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, and Jiawei Han. Understanding the difficulty of training transformers. _arXiv preprint arXiv:2004.08249_ , 2020. 

- Haoyan Ma, Xiang Li, Xia Yuan, and Chunxia Zhao. Denseformer: A dense transformer framework for person re-identification. _IET Computer Vision_ , 17(5), 2023. 

- Xuezhe Ma, Xiaomeng Yang, Wenhan Xiong, Beidi Chen, Lili Yu, Hao Zhang, Jonathan May, Luke Zettlemoyer, Omer Levy, and Chunting Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. _arXiv preprint arXiv:2404.08801_ , 2024. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _EMNLP_ , 2018. 

- Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060. 

- William Peebles and Saining Xie. Scalable diffusion models with transformers. _arXiv preprint arXiv:2212.09748_ , 2022. 

- Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In _2011 AAAI spring symposium series_ , 2011. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_ , 64(9), 2021. 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. _arXiv preprint arXiv:1904.09728_ , 2019. 

- N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. _Outrageously large neural networks_ , 2017. 

- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In _Proceedings of the 2013 conference on empirical methods in natural language processing_ , 2013. 

- Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. _arXiv preprint arXiv:2402.00159_ , 2024. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. _arXiv preprint arXiv:1811.00937_ , 2018. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in neural information processing systems_ , 2017. 

12 

Published as a conference paper at ICLR 2025 

Ben Wang. Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX. https://github.com/kingoflolz/mesh-transformer-jax, May 2021. 

- Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale transformer training instabilities. _arXiv preprint arXiv:2309.14322_ , 2023. 

Shufang Xie, Huishuai Zhang, Junliang Guo, Xu Tan, Jiang Bian, Hany Hassan Awadalla, Arul Menezes, Tao Qin, and Rui Yan. Residual: Transformer with dual residual connections. _arXiv preprint arXiv:2304.14802_ , 2023. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? _arXiv preprint arXiv:1905.07830_ , 2019. 

Biao Zhang and Rico Sennrich. Root mean square layer normalization. _Advances in Neural Information Processing Systems_ , 32, 2019. 

13 

Published as a conference paper at ICLR 2025 

## A TRANSFORMER WITH HYPER-CONNECTIONS 

**==> picture [397 x 467] intentionally omitted <==**

**----- Start of picture text -----**<br>
ℎ ['] ℎ [']<br>+<br>h"' h($<br>+ 𝛽"& + 𝛽$& +<br>FFN FFN<br>+<br>𝛼",!& 𝛼$,!& 𝛼","& 𝛼$,"& 𝛼",$& 𝛼$,$&<br>h"% h%$<br>+ 𝛽"% + 𝛽$% +<br>Attention Attention<br>+<br>𝛼",!% 𝛼$,!% 𝛼","% 𝛼$,"% 𝛼",$% 𝛼$,$%<br>h"$ h$$<br>+ 𝛽"$ + 𝛽$$ +<br>FFN FFN<br>+<br>𝛼",!$ 𝛼$,!$ 𝛼","$ 𝛼$,"$ 𝛼",$$ 𝛼$,$$<br>h"" h"$<br>+ 𝛽"" + 𝛽$" +<br>Attention Attention<br>+<br>𝛼",!" 𝛼$,!" 𝛼","" 𝛼$,"" 𝛼",$" 𝛼$,$"<br>h"! h!$<br>Repeat<br>ℎ [!] ℎ [!]<br>(b) Transformer with Residual Connections (b) Transformer with Hyper-Connections<br>**----- End of picture text -----**<br>


Figure 8: Comparison between transformers with hyper-connections and that with residual connections. 

14 

Published as a conference paper at ICLR 2025 

## B PARAMETERS, COMPUTATION AND MEMORY FOOTPRINT ANALYSIS 

**Static Hyper-Connections.** All learnable parameters are included in the hyper-connection matrix _HC_ in Eq. 1. The number of parameters in one _HC_ is given by: 

**==> picture [309 x 11] intentionally omitted <==**

where _n_ is the expansion rate, _|θ_ **B** _|_ is the number of parameters in **B** in SHC, and _|θ_ **A** _|_ is the number of parameters in **A** . Each layer contains two hyper-connection modules (one for the self attention and one for the feedforward network). Thus, the number of extra parameters is: 

**==> picture [251 x 11] intentionally omitted <==**

where _L_ is the number of layers. For example, in OLMo-1B-SHC _×_ 4, _P_ extra = 4 _×_ (4+2) _×_ 2 _×_ 16 = 768. 

**Dynamic Hyper-Connections.** The parameters of DHC are defined in Eqs. 10, 11, 12, and 13, and the number of parameters is given by: 

**==> picture [365 x 40] intentionally omitted <==**

where _d_ model is the dimension of the hidden states in the transformer, and _|θ_ norm _|_ depends on the type of normalization module. In OLMo models, there are no parameters for normalization, so _|θ_ norm _|_ = 0. In OLMoE, _|θ_ norm _|_ = _d_ model. Similar to the static hyper-connections, the number of extra parameters is: 

**==> picture [251 x 11] intentionally omitted <==**

For example, for OLMo-1B-DHC _×_ 4, _P_ extra = (0 + 2048 _×_ (4 + 2) + 4 _×_ (4 + 2) + 2) _×_ 2 _×_ 16 = 394 _,_ 048. 

The number of parameters for DHC and SHC used in the experiments is detailed in Table 7, while their corresponding FLOPs comparisons are provided in Table 8. Regardless of whether SHC or DHC is used, the additional parameters and computational overhead introduced are minimal and can be considered negligible. 

Table 7: Comparison of number of parameters. 

|**Method**|**HC Params(B)**|**Total Params(B)**|**Total Params**∆**rate (%)**|
|---|---|---|---|
|OLMo-1B|-|1.17676442|-|
|OLMo-1B-SHC_×_2|0.0000026|1.17676467|**+0.00002%**|
|OLMo-1B-SHC_×_4|0.0000077|1.17676518|**+0.00007%**|
|OLMo-1B-DHC_×_2|0.0002625|1.17702688|**+0.02230%**|
|OLMo-1B-DHC_×_4|0.0003940|1.17715846|**+0.03349%**|
|OLMo-7B|-|6.88809574|-|
|OLMo-7B-DHC_×_4|0.0013124|6.88967027|**+0.02286%**|
|OLMoE-1B-7B|-|6.91909427|-|
|OLMoE-1B-7B-DHC_×_4|0.0003940|6.91948832|**+0.00570%**|



**Computation Analysis.** The main computational cost of SHC and DHC lies in line 5 of Algorithm 1, where the complexity is _O_ ( _d_ model _× n ×_ ( _n_ + 1)). The computational cost of the FFN is _O_ (2 _× d_ model _× d_ ffn), and that of the projection part of attention is _O_ (4 _× d_ model _× d_ model). Since _O_ ( _d_ model _× n ×_ ( _n_ + 1)) _≪O_ (4 _× d_ model _× d_ model) _< O_ (2 _× d_ model _× d_ ffn), the computational cost of HC is negligible compared to the cost of both FFN and the attention projection part. Here, _d_ ffn is the inner dimension of the FFN. The detailed computation cost statistics are presented in Table 8. 

15 

Published as a conference paper at ICLR 2025 

Table 8: FLOPs per token in forward pass. 

|**Method**|**HC FLOPs (G)**|**Total FLOPs (G)**|**Total FLOPs**∆**rate (%)**|
|---|---|---|---|
|OLMo-1B|-|2.3536|-|
|OLMo-1B-SHC_×_2|0.0010|2.3545|**+0.038%**|
|OLMo-1B-SHC_×_4|0.0031|2.3566|**+0.127%**|
|OLMo-1B-DHC_×_2|0.0020|2.3554|**+0.076%**|
|OLMo-1B-DHC_×_4|0.0049|2.3583|**+0.200%**|
|OLMo-7B|-|13.3647|-|
|OLMo-7B-DHC_×_4|0.0197|13.3844|**+0.147%**|
|OLMoE-1B-7B|-|2.3580|-|
|OLMoE-1B-7B-DHC_×_4|0.0049|2.3629|**+0.208%**|



**Memory Footprint.** The introduction of HC results in a minor increase in activation memory usage during training. For a transformer model with _L_ layers, a model dimension of _d_ model, batch size _b_ , sequence length _s_ , and number of attention heads _a_ , the activation memory is calculated as _sbd_ model _L_ (34 + 5 _as/d_ model), as outlined in Korthikanti et al. (2022). Incorporating HC with an expansion rate of _n_ adds an extra memory overhead of 2 _nsbd_ model _L_ . For _n_ = 2, this contributes less than 15% to the total memory usage of a standard transformer. Notably, the memory consumption is mostly driven by the weight parameters, which experience only a slight increase with HC. Additionally, given HC’s low computational cost, the hidden states generated by HC can be discarded post forward pass and recomputed during backpropagation to further optimize memory usage. With this approach, the additional memory requirement is reduced to _nsbd_ model. During inference, the memory usage for activations is largely determined by the Key-Value cache, which is not impacted by the extra activations brought by HC. Moreover, the hidden states from earlier layers can be released as soon as the next layer’s computations start, significantly lowering memory requirements. The actual memory footprint is empirically measured on 8 GPUs, as shown in Table 9. 

Table 9: Measured Memory Footprint on 8 GPUs. 

|**Method**|**Memory (GB)**|**Memory**∆**Rate (**%**)**|**Micro Batch Size**<br>**(tokens per GPU)**|
|---|---|---|---|
|OLMo-1B|41.11|-|16,384|
|OLMo-1B-SHC_×_2|47.55|**+15.7%**|16,384|
|OLMo-1B-SHC_×_4|51.85|**+26.0%**|16,384|
|OLMo-1B-DHC_×_2|47.56|**+15.7%**|16,384|
|OLMo-1B-DHC_×_4|51.86|**+26.1%**|16,384|
|OLMo-7B|26.27|-|2,048|
|OLMo-7B-DHC_×_4|33.70|**+28.28%**|2,048|
|OLMoE-1B-7B|31.59|-|4,096|
|OLMoE-1B-7B-DHC_×_4|34.65|**+9.7%**|4,096|



16 

Published as a conference paper at ICLR 2025 

## C MOE 1B/7B MODEL EXPERIMENTS 

**==> picture [358 x 500] intentionally omitted <==**

**----- Start of picture text -----**<br>
training loss C4 en val. loss dolma books val. loss dolma cc val. loss<br>2.5 OLMoE-1B-7B OLMoE-1B-7B OLMoE-1B-7B OLMoE-1B-7B<br>OLMoE-1B-7B-DHCx4 2.8 OLMoE-1B-7B-DHCx4 2.7 OLMoE-1B-7B-DHCx4 2.90 OLMoE-1B-7B-DHCx4<br>2.4 2.85<br>2.6 2.80<br>2.7<br>2.3 2.75<br>2.5<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>dolma pes2o val. loss dolma reddit val. loss dolma stack val. loss dolma wiki val. loss<br>2.30 OLMoE-1B-7B 3.05 OLMoE-1B-7B 1.10 OLMoE-1B-7B OLMoE-1B-7B<br>2.25 OLMoE-1B-7B-DHCx4 3.00 OLMoE-1B-7B-DHCx4 OLMoE-1B-7B-DHCx4 2.4 OLMoE-1B-7B-DHCx4<br>2.20 1.05<br>2.95<br>2.15 1.00 2.3<br>2.90<br>2.10<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>2.8 ice val. lossOLMoEOLMoE-1B-7B-DHCx4-1B-7B 3.203.15 m2d2-OLMoE-1B-7B OLMoE-1B-7B-DHCx4 s2orc val. loss 2.202.15 pile val. lossOLMoE-1B-7B OLMoE-1B-7B-DHCx4 2.5 wikitext 103 val. lossOLMoE-1B-7B OLMoE-1B-7B-DHCx4<br>2.72.6 3.103.05 2.102.05 2.4<br>2.5 3.00 2.00 2.3<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>MMLU stem Var Acc. (%) MMLU hum. Var Acc. (%)MMLU soc. sci. Var Acc. (%)MMLU other Var Acc. (%)<br>34 42.5<br>34 50<br>32 40.0<br>30 32 37.5 45<br>OLMoE-1B-7B OLMoE-1B-7B OLMoE-1B-7B OLMoE-1B-7B<br>28 OLMoE-1B-7B-DHCx4 30 OLMoE-1B-7B-DHCx4 35.0 OLMoE-1B-7B-DHCx4 40 OLMoE-1B-7B-DHCx4<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>40 MMLU avg. Acc. (%) 70 HellaSwag Acc. (%) 94 SciQ Acc. (%) ARC Challenge Acc. (%OLMoE-1B-7B )<br>38 92 45 OLMoE-1B-7B-DHCx4<br>65<br>36 90 40<br>34 OLMoE-1B-7BOLMoE-1B-7B-DHCx4 60 OLMoEOLMoE-1B-7B-DHCx4-1B-7B 88 OLMoE-1B-7BOLMoE-1B-7B-DHCx4 35<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>ARC easy Acc. (%) PIQA Acc. (%) WinoGrande Acc. (%) Openbook QA Acc. (%)<br>44<br>78 65.0<br>75 42<br>76 62.5 40<br>70 60.0 38<br>OLMoE-1B-7B OLMoE-1B-7B OLMoE-1B-7B OLMoE-1B-7B<br>65 OLMoE-1B-7B-DHCx4 74 OLMoE-1B-7B-DHCx4 57.5 OLMoE-1B-7B-DHCx4 36 OLMoE-1B-7B-DHCx4<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>BoolQ Acc. (%) COPA Acc. (%) Commonsense QA Acc. (%)48 Social Iqa Acc. (%)<br>87.5 OLMoE-1B-7B 50.0<br>65 85.0 OLMoE-1B-7B-DHCx4 47.5 47<br>82.5 46<br>45.0<br>60 OLMoE-1B-7B 80.0 OLMoE-1B-7B 45 OLMoE-1B-7B<br>OLMoE-1B-7B-DHCx4 77.5 42.5 OLMoE-1B-7B-DHCx4 44 OLMoE-1B-7B-DHCx4<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B) Tokens (B)<br>**----- End of picture text -----**<br>


Figure 9: Loss curves in V3 validation sets and accuracy curves on downstream tasks for OLMoE-1B7B and OLMoE-1B7B-DHC _×_ 4 models. 

17 

Published as a conference paper at ICLR 2025 

## D 7B MODEL EXPERIMENTS 

**==> picture [326 x 521] intentionally omitted <==**

**----- Start of picture text -----**<br>
c4 en val. loss dolma books val. loss dolma cc val. loss<br>2.7 OLMo-7B-OLMo-7B-DHCx4 2.9 OLMo-7B-OLMo-7B-DHCx4 2.7 OLMo-7B-OLMo-7B-DHCx4<br>2.6 2.8 2.6<br>2.7<br>2.5 2.5<br>2.6<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B)<br>dolma pes2o val. loss dolma reddit val. loss dolma stack val. loss<br>OLMo-7B- 3.0 OLMo-7B- 1.05 OLMo-7B-<br>2.3 OLMo-7B-DHCx4 OLMo-7B-DHCx4 OLMo-7B-DHCx4<br>2.9 1.00<br>2.2<br>0.95<br>2.8<br>2.1<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B)<br>dolma wiki val. loss ice val. loss m2d2-s2orc val. loss<br>2.5 OLMo-7B- 2.8 OLMo-7B- 3.2 OLMo-7B-<br>OLMo-7B-DHCx4 OLMo-7B-DHCx4 OLMo-7B-DHCx4<br>2.4 2.7<br>3.1<br>2.3 2.6<br>3.0<br>2.5<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B)<br>pile val. loss wikitext 103 val. loss HellaSwag Acc. (%)<br>OLMo-7B- 2.7 OLMo-7B- 70<br>2.2 OLMo-7B-DHCx4 2.6 OLMo-7B-DHCx4 65<br>2.1 2.5 60<br>OLMo-7B-<br>2.4<br>55 OLMo-7B-DHCx4<br>2.0<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B)<br>SciQ Acc. (%) COPA Acc. (%) Openbook QA Acc. (%)<br>92.5<br>85 40.0<br>90.0<br>87.5 80 37.5<br>85.0 OLMo-7B- 75 OLMo-7B- 35.0 OLMo-7B-<br>82.5 OLMo-7B-DHCx4 70 OLMo-7B-DHCx4 32.5 OLMo-7B-DHCx4<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B)<br>PIQA Acc. (%) WinoGrande Acc. (%) ARC-Easy Acc. (%)<br>65<br>78 70<br>76 60 65<br>74 OLMo-7B- 55 OLMo-7B- 60 OLMo-7B-<br>72 OLMo-7B-DHCx4 OLMo-7B-DHCx4 55 OLMo-7B-DHCx4<br>100 200 300 400 500 100 200 300 400 500 100 200 300 400 500<br>Tokens (B) Tokens (B) Tokens (B)<br>**----- End of picture text -----**<br>


Figure 10: Loss curves in V3 validation set and accuracy curves on downstream tasks for OLMo-7B and OLMo-7B-DHC _×_ 4 models. 

18 

Published as a conference paper at ICLR 2025 

## E VISION EXPERIMENTS 

**Datasets.** We use the ILSVRC-2012 ImageNet dataset (Deng et al., 2009) with 1k classes and 1.3M images (see ImageNet in the following) for image generation and classification. 

## E.1 IMAGE GENERATION 

To investigate the generalizability of hyper-connections in image generation, our experiments are conducted using the DiT framework (Peebles & Xie, 2022) training the models for 1400 epochs. In order to save experimental costs, we use FP16 precision, introduce flash-attention to speed up training, and introduce QK-Norm (Wortsman et al., 2023) to stabilize training. 

Table 10: Benchmarking class-conditional image generation on ImageNet 256 _×_ 256, with cfg=1.50. **NP** , **P** , and **R** are short for Numerical Precision, Precision, and Recall, respectively. 

|**Method**|**NP**|**QK-Norm**|**Size (M)**|**FID**_↓_|**sFID**_↓_|**IS**_↑_|**P**_↑_|**R**_↑_|
|---|---|---|---|---|---|---|---|---|
|DiT-XL/2|FP32|✗|675|2.27|4.60|278.24|0.83|0.57|
|DiT-XL/2|FP16|✓|675|2.36|4.54|269.46|0.83|0.58|
|DiT-1B/2|FP16|✓|983|2.13|4.50|288.69|0.82|0.59|
|DiT-XL/2-SHC_×_2|FP16|✓|675|2.18|4.52|287.24|0.82|0.60|



Our experimental results demonstrate that DiT models incorporating hyper-connections exhibit comparable performance metrics to DiT models with 50% more parameters. This finding underscores the efficiency and efficacy of hyper-connections in enhancing model performance without increasing model size. 

## E.2 IMAGE CLASSIFICATION 

For the image classification experiments, we train ViT/16-Base and ViT/16-Large models with images at a resolution of 224 _×_ 224 for 300 epochs, following the experimental setup used by (Dosovitskiy et al., 2020).To speed up the training process, we use bfloat16 numerical precision. The training configuration is detailed in Table 12. Within this configuration, we replace the residual connections with static and dynamic hyper-connections, referred to as SHC and DHC, respectively, using an expansion rate of _n_ = 2. The top-1 accuracy results are presented in Table 11, and the training loss curves for ViT/16-Large and ViT/16-Large with DHC _×_ 2 are shown in Fig. 11. 

For the Base model (85M), our re-implemented ViT/16 achieves 76.38% accuracy on 224 _×_ 224 images. The SHC and DHC enhance performance to 77.60% and 77.26%, respectively. representing relative increases of **1.22%** and **0.88%** . For the Large model (307M parameters), ViT/16 achieves 77.25% accuracy. The SHC and DHC configurations further enhance accuracy to 78.38% and 79.94%, respectively. This corresponds to relative improvements of **1.13%** and **2.69%** , with DHC showing the highest performance. These results demonstrate that hyper-connections (SHC and DHC) significantly improve accuracy, especially in the Large model scale. 

Table 11: Accuracy on ImageNet. **ViT*/16** refers to the results reported by (Dosovitskiy et al., 2020), whereas **ViT/16** denotes our re-implemented baseline. SHC and DHC indicate that residual connections are replaced with static and dynamic hyper-connections, respectively. 

|**Model Scales**<br>**Params (M)**|**ViT*/16**|**ViT/16**<br>**ViT/16-SHC**_×_2<br>**ViT/16-DHC**_×_2<br>224_×_224|
|---|---|---|
||384_×_384||
|**Base**<br>85|77.91|76.38<br>**77.60**<br>77.26|
|**Large**<br>307|76.53|77.25<br>78.38<br>**79.94**|



19 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss<br>1.8 ViT/16-Lagre<br>1.6 ViT/16-Lagre-DHCx2<br>1.4<br>1.2<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>30000 40000 50000 60000 70000 80000 90000<br>Steps<br>Loss<br>**----- End of picture text -----**<br>


Figure 11: Training loss curves of ViT/16-Large and ViT/16-Large-DHC _×_ 2, smoothed using an Exponential Moving Average (EMA) with a decay rate of 0.999. The gain from Hyper-Connections decreases as training progresses, likely due to pass over the same dataset across many epochs, resulting in diminishing returns from the additional capacity provided by Hyper-Connections. 

## E.3 VISULIZATION OF DHC 

We randomly select three categories from the ImageNet dataset and sample the corresponding examples from the validation set. These samples are fed into the ViT-Base/16-DHC _×_ 2 model to compute the dynamic connection weights of the DHC in the final layer. As shown in Fig. 12, we visualize the distribution of these weights. We observe that the intra-class distribution of beta is highly concentrated, indicating that samples within the same category tend to have similar beta values. In contrast, the distribution of alpha is less concentrated, but the differences between the distributions of different categories are more pronounced, as exemplified by _α_ 2 _,_ 0. 

Table 12: Training hyperparameters for ViT. 

|**Hyperparameter**|**Value**|
|---|---|
|**Learning Rate (lr)**|0.003|
|**Batch Size**|4096|
|**Scheduler**|Cosine Annealing with Linear Warmup (10k steps)|
|**Data Augmentation**|Mixup (_α_= 0_._2)|
|**Epochs**|300|
|**Optimizer**|AdamW (_β_1 = 0_._9,_β_2 = 0_._999,_ϵ_= 1_e −_8)|
|**Gradient Clipping**|1.0|
|**Weight Decay**|0.3|
|**Dropout**|0.1|
|**Precision**|bf16|



20 

Published as a conference paper at ICLR 2025 

**==> picture [397 x 556] intentionally omitted <==**

**----- Start of picture text -----**<br>
33:loggerhead turtle 998:capitulum 779:school bus<br>30 40 40<br>30 30<br>20<br>20 20<br>10<br>10 10<br>0 0 0<br>0.95 1.00 1.05 1.10 1.15 1.20 0.95 1.00 1.05 1.10 1.15 1.20 0.95 1.00 1.05 1.10 1.15 1.20<br>1 1 1<br>33:loggerhead turtle 998:capitulum 779:school bus<br>30 50<br>25<br>25<br>40<br>20 20<br>15 15 30<br>10 10 20<br>5 5 10<br>0 0 0<br>0.95 1.00 1.05 1.10 1.15 1.20 0.95 1.00 1.05 1.10 1.15 1.20 0.95 1.00 1.05 1.10 1.15 1.20<br>2 2 2<br>33:loggerhead turtle 998:capitulum 779:school bus<br>5<br>8 30<br>25 4<br>6<br>20 3<br>4 15 2<br>10<br>2 5 1<br>0 0 0<br>0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.65 0.60 0.55 0.50 0.45 0.40 0.35<br>1, 0 1, 0 1, 0<br>33:loggerhead turtle 998:capitulum 779:school bus<br>5 4 50<br>4 3 40<br>3 30<br>2<br>2 20<br>1 1 10<br>0 0 0<br>0.9 1.0 1.1 1.2 1.3 0.9 1.0 1.1 1.2 1.3 0.9 1.0 1.1 1.2 1.3<br>1, 1 1, 1 1, 1<br>33:loggerhead turtle 998:capitulum 779:school bus<br>5 4 50<br>4 3 40<br>3 30<br>2<br>2 20<br>1 1 10<br>0 0 0<br>0.1 0.0 0.1 0.2 0.3 0.1 0.0 0.1 0.2 0.3 0.1 0.0 0.1 0.2 0.3<br>1, 2 1, 2 1, 2<br>33:loggerhead turtle 998:capitulum 779:school bus<br>6 4<br>5 15<br>3<br>4<br>10<br>3 2<br>2 5 1<br>1<br>0 0 0<br>2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40<br>2, 0 2, 0 2, 0<br>33:loggerhead turtle 998:capitulum 779:school bus<br>5 50<br>6<br>4 40<br>4 3 30<br>2 20<br>2<br>1 10<br>0 0 0<br>0.2 0.1 0.0 0.1 0.2 0.2 0.1 0.0 0.1 0.2 -0.20 -0.10 0.00 0.10 0.20<br>2, 1 2, 1 2, 1<br>Frequency Frequency Frequency<br>Frequency Frequency Frequency<br>Frequency Frequency Frequency<br>Frequency Frequency Frequency<br>Frequency Frequency Frequency<br>Frequency Frequency Frequency<br>Frequency Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 12: Distribution of weights of last DHC in ViT-Base/16-DHC _×_ 2 model. 

## F MORE VISUALIZATION AND ANALYSIS 

**Unfolding hyper-connections.** We first introduce how to determine the connection matrix **C**[(0)] for hyper-connections. To simplify writing, the layer output _T[k]_ ( **h** _[k]_ 0[)][ is denoted by] _[ T][k]_[for short.][The] 

21 

Published as a conference paper at ICLR 2025 

**==> picture [394 x 162] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32<br>0 0 0 0 0 1.0<br>4 C [(0)] 4 C [(1)] 4 C [(2)] 4 C [(3)] 4 C [(4)]<br>8 8 8 8 8 0.5<br>12 12 12 12 12<br>16 ty, 16 k 16 i. 16 Fe. 16 0.0<br>20 20 20 20 20<br>24 24 24 24 24 0.5<br>28 28 28 28 28<br>32 32 32 32 32 1.0<br>(a) Connection matrix for DHC model.<br>0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32<br>0 0 0 0 0 1.0<br>4 C [(0)] 4 C [(1)] 4 C [(2)] 4 C [(3)] 4 C [(4)]<br>8 8 8 8 8 0.5<br>12 12 12 12 12<br>16 16 . 16 by 16 be 16 0.0<br>20 20 20 20 20<br>24 24 24 24 24 0.5<br>28 28 28 28 28<br>32 32 32 32 32 1.0<br>**----- End of picture text -----**<br>


(b) Connection matrix for SHC model. 

Figure 13: **Visualization of unfolded connection matrix.** Matrices from left to right are **C**[(0)] (Connections for _{_ **h** _[j]_ 0 _[}][L] j_ =0[+1][),] **[C]**[(] _[i]_[)][(Connections][for] _[{]_ **[h]** _[′][j] i[}][L] j_ =0[+1][)][for] _[i][∈{]_[1] _[,]_[ 2] _[,]_[ 3] _[,]_[ 4] _[}]_[.][The][at-] tention layers, which have odd ids, are marked with green tick marks. 

recurrent form of hyper connection in Eq. 2 is expanded as follows: 

**==> picture [312 x 86] intentionally omitted <==**

Therefore, we obtain connection matrix _c_[(0)] _kj_[=] **[B]** _[j]_[(] I1 _kt_ = _−j_ 1+1 **[A][r]** _t_ ) **Am** _k_ . Similarly, the connection matrix **C**[(] _[i]_[)] for the _i_ -th hyper hidden from _k_ -th layer can be computed by substituting the last **Am** _k_ with **Ar** _k_ in Eq. 27, i.e., 

**==> picture [291 x 73] intentionally omitted <==**

**Visualization for hyper hidden.** We visualize connection matrices for hyper hiddens in Fig. 13 to reveal how hyper-connection maintains intermediate layer outputs. First of all, the four hyper hiddens are dissimilar and show completely different connection patterns. Then, we can see outputs from FFN layers are preserved long-termly in hyper hiddens, while attention layers are reserved less. It is also observed that the long-term connections are usually stored in pairs of hyper hiddens, where the connection is positive in one hyper hidden but negative in the other, for example, column 0 and 2 in **C**[(1)] _,_ **C**[(3)] . With such strategy, these connections can be easily eliminated in the sum-pooling operation before the unembedding layer. 

**SHC shares similar connection pattern with DHC.** We show the connection matrices for OLMo-1B-SHC _×_ 4 model in Fig. 13b. Comparing to DHC, as shown in Fig. 13a, SHC shares exactly the same connection patterns. Moreover, we observe many more PTB-like blocks in SHC, e.g., layers from 13 to 18. Note that the connection relation for SHC is token independent, and such PTB-like blocks can be physically reorganized to be parallelly computed. 

22 

Published as a conference paper at ICLR 2025 

**==> picture [391 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32 0 4 8 12 16 20 24 28 32<br>0 1.00 0 1.00 0 1.00<br>4 0.75 4 0.75 4 0.75<br>8 0.50 8 0.50 8 0.50<br>12 0.25 12 0.25 12 0.25<br>16 ↓ wasted 0.00 16 0.00 16 0.00<br>20 0.25 20 0.25 20 0.25<br>24 0.50 24 0.50 24 0.50<br>28 0.75 28 0.75 28 0.75<br>32 1.00 32 1.00 32 1.00<br>(a) OLMo-1B-DHC × 1 (b) OLMo-1B-DHC × 2 (c) OLMo-1B-DHC × 4<br>**----- End of picture text -----**<br>


Figure 14: Comparison of unfolded connection matrices for OLMo-1B-DHC _×_ 1, OLMo-1B-DHC _×_ 2 and OLMo-1B-DHC _×_ 4 model. 

**How HC** _×_ 1 **fails.** The OLMo-1B _×_ 1 model is observed to perform worse than baseline in our experiments. Its connection matrix is visualized in Fig. 14 to show how it fails. Above all, we observe that layer 17 is wasted, who has no connection to subsequent layers at all. Secondly, compared to HC _×_ 2 and HC _×_ 4 models, the Λ shaped pattern does not appear. Note that HC _×_ 1 does not support the pattern of Λ in its mathematical formulation, where the connections to previous layers must be weakened or strengthened simultaneously. Thus, the lack of connection from the early layers to the final layers may suffer from gradient vanishing, like post-norm style transformers, which leads to performance degeneration. 

23 

Published as a conference paper at ICLR 2025 

## G DERIVATION OF NON-TRAINABLE HYPER-CONNECTION MATRIX FOR RESIDUAL CONNECTIONS 

## G.1 PRE-NORM RESIDUAL CONNECTION 

In the Pre-Norm residual connection, the input to a layer is first normalized before being passed through the layer. The output of the layer is then added to the original input. This can be represented as: 

**==> picture [247 x 13] intentionally omitted <==**

By incorporating the normalization operator into the layer, _T_ := _T ◦_ Norm, we can express the entire process as: 

**==> picture [231 x 13] intentionally omitted <==**

To express this using hyper-connections, the matrix for Pre-Norm can be structured as follows: 

**==> picture [249 x 25] intentionally omitted <==**

Given hyper hidden matrix **H** = **h**[⊺] , we prove that the output of _HC_ PreNorm **H[ˆ]** = **h[ˆ]**[⊺] . 

_Proof._ 

**==> picture [262 x 55] intentionally omitted <==**

## G.2 POST-NORM RESIDUAL CONNECTION 

In the Post-Norm residual connection, the input to a layer is passed through the layer first, and then the output is normalized after being added to the original input. In matrix form, this can be represented as: 

**==> picture [222 x 12] intentionally omitted <==**

The summation of the input and the normalized output of the layer is: 

**==> picture [239 x 13] intentionally omitted <==**

We consider Norm to be LayerNorm (Zhang & Sennrich, 2019). The analysis process for RMSNorm is almost identical. In fact, the affine transformation can be incorporated into the subsequent layer, while the mean subtraction operation can be integrated into the current layer. 

**==> picture [232 x 10] intentionally omitted <==**

where _A_ is the affine transformation, and _C_ is the re-centering operator. Thus, the mean of the output of _T_ is 0. 

To express this using hyper-connections with an expansion rate _n_ = 1, we need a hyper-connection matrix _HC_ that encapsulates this operation: 

**==> picture [317 x 37] intentionally omitted <==**

24 

Published as a conference paper at ICLR 2025 

Similar to the previous proof, we prove that the output of _HC_ PostNorm is equivalent to the transpose of the output of the Post-Norm residual connection: 

**==> picture [218 x 13] intentionally omitted <==**

_Proof._ Note that 

**==> picture [262 x 19] intentionally omitted <==**

Given this fact, we can derive the Post-Norm: 

**==> picture [270 x 95] intentionally omitted <==**

For hyper-connections side, we have: 

**==> picture [332 x 73] intentionally omitted <==**

25 

Published as a conference paper at ICLR 2025 

## H SEQUENTIAL-PARALLEL DUALITY 

## H.1 HYPER-CONNECTION MATRIX OF SEQUENTIAL ARRANGEMENT 

In this section, we demonstrate that the following hyper-connection matrix will produce _n_ identical networks arranged sequentially with residual connections between them: 

**==> picture [249 x 25] intentionally omitted <==**

where **e** _n×n_ denotes an _n × n_ identity matrix, **e** _i ∈_ R _[n][×]_[1] represents the _i_ -th column of **e** _n×n_ , and **1** 1 _×n_ signifies a 1 _× n_ matrix of ones. 

We will use mathematical induction to prove that **h** _[k] i_[=] **[h]** _[k] j_[and] **[h]** _[k] i_[+1] = _T[k]_ ( **h** _[k] i_[) +] **[ h]** _[k] i_[,] _[∀][i, j][∈] {_ 0 _,_ 1 _, . . . , n}_ , _∀k ∈{_ 0 _,_ 1 _, . . . , L}_ , where _L_ is the number of layers. 

_Proof._ BASE CASE 

For _k_ = 0, we have the initial condition **h**[0] _i_[=] **[h]**[0] _j_[,] _[∀][i, j][∈{]_[0] _[,]_[ 1] _[, . . . , n][}]_[,][as][we][define] **[H]**[0][=] **h**[0] **h**[0] _. . ._ **h**[0][�][⊺] _∈_ R _[n][×][d]_ . � 

## INDUCTION HYPOTHESIS 

Assume that for some _k ∈{_ 1 _, . . . , L −_ 1 _}_ , we have **h** _[k] i_[=] **[h]** _[k] j_[and] **[h]** _[k] i_[=] _[T][k]_[(] **[h]** _i[k][−]_[1] ) + **h** _[k] i[−]_[1] , _∀i, j ∈{_ 0 _,_ 1 _, . . . , n}_ . 

INDUCTION STEP 

We have 

**==> picture [345 x 113] intentionally omitted <==**

Since **h** _[k] i_[=] **[ h]** _[k] j_[,] _[ ∀][i, j][∈{]_[0] _[,]_[ 1] _[, . . . , n][}]_[, it follows that] _[ T][k]_[(] **[h]** _[k]_ 1[) +] **[ h]** _[k] i_[=] _[ T][k]_[(] **[h]** _[k]_ 1[) +] **[ h]** _[k] j_[.][Thus, we have] 

**==> picture [227 x 15] intentionally omitted <==**

Since **h** _[k] i_[=] **[ h]** _[k] j_[,] _[ ∀][i, j][∈{]_[0] _[,]_[ 1] _[, . . . , n][}]_[, it follows that] **[ h]** _[k]_ 1[=] **[ h]** _[k] i_[,] _[ ∀][i][ ∈{]_[0] _[,]_[ 1] _[, . . . , n][}]_[.][Thus, we have] 

**==> picture [244 x 29] intentionally omitted <==**

26 

Published as a conference paper at ICLR 2025 

## H.2 HYPER-CONNECTION MATRIX OF PARALLEL ARRANGEMENT 

In this section, we demonstrate that the following hyper-connection matrix will produce a network where every _n_ adjacent layers are arranged in parallel, with each layer incorporating residual connections. We define a parallel-arranged network such that _n_ adjacent layers form a group, with layers within a group being parallel and groups arranged sequentially. The output of _k_ -th group is given by: 

**==> picture [267 x 28] intentionally omitted <==**

It can be proved that this arrangement can be described by the following hyper-connection matrices. **First, for** _k_ **where** _k −_ 1 _≡_ 0 (mod _n_ ) **:** 

**==> picture [288 x 25] intentionally omitted <==**

where the _HC_ matrix can be decomposed into two operations: 1) sum up all the outputs of the previous group and use it as the input of the current layer and as the residual of the subsequent layers; 2) sum up the output and input saving to the first hidden vector slot. 

**Next, for** _k_ **where** _k −_ 1 _≡ i_ (mod _n_ ) **and** _i ̸_ = 0 **:** 

**==> picture [296 x 25] intentionally omitted <==**

where the _HC_ matrix selects the _i_ -th hidden vector as the input of the current layer, and sums up the output and input, saving to the _i_ -th hidden vector slot. 

This means: 

**==> picture [272 x 60] intentionally omitted <==**

This can also be proved by mathematical induction; however, the conclusion is quite obvious through drawing, and the proof process is very tedious. Therefore, we don’t repeat the similar proof here. 

27 

Published as a conference paper at ICLR 2025 

## I PSEUDOCODE OF HYPER-CONNECTIONS 

**Algorithm 1** Network with Hyper-Connections 

**Require:** Initial hidden vector **h**[0] _∈_ R _[d]_ **Require:** Expansion rate _n_ **Ensure:** Final output **y** 1: **Initialize:** 2: **H**[0] _←_ **h**[0] **h**[0] _. . ._ **h**[0][�][⊺] _∈_ R _[n][×][d]_ � 3: **for** _k_ = 1 to _L_ **do** 4: **H** _←_ **H** _[k][−]_[1] 5: ( **h** 0 **H** _[′]_ ) _←WC[k]_[⊺] **H** 6: **h** _[′]_ 0 _[←T][k]_[(] **[h]**[0][)] 7: **ˆH** _←_ **B** _[k]_[⊺] **h** _[′]_ 0[+] **[ H]** _[′]_ 8: **H** _[k] ←_ **H[ˆ]** 9: **end for** 10: **Final Output:** 11: **h** _[L] ←_ sum rows of **H** _[L]_ 12: **h** _[L] ←_ Normalization Layer( **h** _[L]_ ) 13: **y** _←_ Output Layer( **h** _[L]_ ) 14: **return y** 

_▷_ For each layer _▷_ Width Connections _▷_ Layer Computation _▷_ Depth Connections 

28 

Published as a conference paper at ICLR 2025 

## J PYTORCH IMPLEMENTATION OF HYPER-CONNECTIONS 

**Algorithm 2** Pseudocode of hyper-connections in a PyTorch-like style. 

# h: hyper hidden matrix (BxLxNxD) class HyperConnection(nn.Module): def __init__(self, dim, rate, layer_id, dynamic, device=None): super(HyperConnection, self).__init__() self.rate = rate self.layer_id = layer_id self.dynamic = dynamic self.static_beta = nn.Parameter(torch.ones((rate,), device=device)) init_alpha0 = torch.zeros((rate, 1), device=device) init_alpha0[layer_id % rate, 0] = 1. self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), device= device)], dim=1)) if self.dynamic: self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate+1), device=device)) self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01) self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim, ), device=device)) self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01) self.layer_norm = LayerNorm(dim) def width_connection(self, h): # get alpha and beta if self.dynamic: norm_h = self.layer_norm(h) if self.dynamic: wc_weight = norm_h @ self.dynamic_alpha_fn wc_weight = F.tanh(wc_weight) dynamic_alpha = wc_weight * self.dynamic_alpha_scale alpha = dynamic_alpha + self.static_alpha[None, None, ...] else: alpha = self.static_alpha[None, None, ...] if self.dynamic: dc_weight = norm_h @ self.dynamic_beta_fn dc_weight = F.tanh(dc_weight) dynamic_beta = dc_weight * self.dynamic_beta_scale beta = dynamic_beta + self.static_beta[None, None, ...] else: beta = self.static_beta[None, None, ...] # width connection mix_h = alpha.transpose(-1, -2) @ h return mix_h, beta def depth_connection(self, mix_h, h_o, beta): h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :] return h 

**Algorithm 3** Pseudocode of transformer with hyper-connections in a PyTorch-like style. 

# h: hyper hidden matrix (BxLxNxD) # atten_hyper_connection, ffn_hyper_connection: hyper-connection modules # attn_norm, ffn_norm: normalization modules 

# Attention Block mix_h, beta = atten_hyper_connection.width_connection(h) h = attn_norm(mix_h[...,0,:]) h = self_attention(h) h = atten_hyper_connection.depth_connection(mix_h, dropout(h), beta) # FFN Block mix_h, beta = ffn_hyper_connection.width_connection(h) h = ffn_norm(mix_h[...,0,:]) h = ffn(h) h = ffn_hyper_connection.depth_connection(mix_h, dropout(h), beta) 

29 

Published as a conference paper at ICLR 2025 

## K VALIDATION SETS AND DOWNSTREAM TASKS 

Table 13: OLMo’s default configuration was evaluated using multiple metrics. Perplexity (PPL) and loss were used for the V2 and V3 Validation Sets, while zero-shot testing was applied to the Downstream Benchmarks. However, the grey benchmarks were excluded from our analysis due to the instability of their performance indicators. 

## **V2 Validation Sets** 

v2-small-4chan-validation v2-small-c4_100_domains-validation v2-small-c4_en-validation v2-small-gab-validation v2-small-ice-validation v2-small-m2d2_s2orc-validation v2-small-m2d2_wiki-validation v2-small-manosphere-validation v2-small-mc4_en-validation v2-small-pile-validation v2-small-ptb-validation v2-small-twitterAEE-validation v2-small-wikitext_103-validation 

## **V3 Validation Sets** 

v3-small-c4_en-validation v3-small-dolma_books-validation v3-small-dolma_common-crawl-validation v3-small-dolma_pes2o-validation v3-small-dolma_reddit-validation v3-small-dolma_stack-validation v3-small-dolma_wiki-validation v3-small-ice-validation v3-small-m2d2_s2orc-validation v3-small-pile-validation v3-small-wikitext_103-validation 

## **Downstream Benchmarks** 

piqa (Bisk et al., 2020) hellaswag (Zellers et al., 2019) winogrande (Sakaguchi et al., 2021) openbook_qa (Mihaylov et al., 2018) sciq (Johannes Welbl, 2017) arc_easy (Clark et al., 2018) copa (Roemmele et al., 2011) commitment_bank (De Marneffe et al., 2019) mrpc (Dolan & Brockett, 2005) rte (Dagan et al., 2005) sst2 (Socher et al., 2013) 

30 

Published as a conference paper at ICLR 2025 

Table 14: Downstream Benchmarks for OLMoE. 

**Downstream Benchmarks for OLMoE** piqa (Bisk et al., 2020) hellaswag (Zellers et al., 2019) winogrande (Sakaguchi et al., 2021) openbook_qa (Mihaylov et al., 2018) sciq (Johannes Welbl, 2017) arc_easy (Clark et al., 2018) arc_challenage (Clark et al., 2018) copa (Roemmele et al., 2011) boolq (Clark et al., 2019) commonsense_qa (Talmor et al., 2018) social_iqa (Sap et al., 2019) mmlu (Hendrycks et al., 2021) 

## L 1B MODEL EXPERIMENTS 

**==> picture [396 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss<br>2.9<br>OLMo-1B<br>OLMo-1B-ResiDual<br>2.8 OLMo-1B-Altupx2<br>OLMo-1B-DHCx2<br>OLMo-1B-DHCx2  W/O tanh<br>2.7<br>2.6<br>2.5<br>2.4<br>100 200 300 400 500<br>Tokens (Billions)<br>Loss<br>**----- End of picture text -----**<br>


Figure 15: Training loss curves of related works, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99. 

31 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss<br>2.60 OLMo-1B<br>OLMo-1B-DHCx1<br>OLMo-1B-DHCx2<br>2.55<br>OLMo-1B-DHCx4<br>OLMo-1B-DHCx8<br>2.50<br>2.45<br>2.40<br>2.35<br>200 400 600 800 1000<br>Tokens (Billions)<br>Loss<br>**----- End of picture text -----**<br>


Figure 16: Training loss curves of DHC with tanh over 500 billion tokens, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99. 

**==> picture [396 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss<br>2.60 OLMo-1B<br>OLMo-1B-DHCx1 W/O tanh<br>OLMo-1B-DHCx2  W/O tanh<br>2.55<br>OLMo-1B-DHCx4  W/O tanh<br>OLMo-1B-DHCx8  W/O tanh<br>2.50<br>2.45<br>2.40<br>2.35<br>200 400 600 800 1000<br>Tokens (Billions)<br>Loss<br>**----- End of picture text -----**<br>


Figure 17: Training loss curves of DHC without tanh over 500 billion tokens, smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99. 

32 

Published as a conference paper at ICLR 2025 

**==> picture [396 x 199] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Loss<br>2.9<br>OLMo-1B<br>OLMo-1B-PTB<br>2.8 OLMo-1B-DHCx4  W/O tanh<br>OLMo-1B-DHCx4<br>2.7<br>2.6<br>2.5<br>2.4<br>100 200 300 400 500<br>Tokens (Billions)<br>Loss<br>**----- End of picture text -----**<br>


Figure 18: Training loss curves comparied with parallel transformer blocks (PTB), smoothed using Exponential Moving Average (EMA) with a decay rate of 0.99. 

Table 15: Results on downstream benchmarks for 1B models. 

|**Method**||**arc_easy**|**copa**|**hellaswag**|**openbook_qa**|**piqa**|**sciq**|**winogrande**|**avg.**|
|---|---|---|---|---|---|---|---|---|---|
|OLMo-1B||56.8|76.0|56.1|33.8|74.4|85.1|55.6|62.5|
||||Scaling n in DHC W/O tanh|||||||
|OLMo-1B-DHCx1|W/O tanh|56.8|75.0|55.3|33.4|72.9|85.4|57.1|62.3|
|OLMo-1B-DHCx2|W/O tanh|63.0|74.0|57.1|34.6|73.5|86.0|58.2|63.8|
|OLMo-1B-DHCx4|W/O tanh|61.2|80.0|57.5|33.6|75.5|85.8|56.9|64.4|
|OLMo-1B-DHCx8|W/O tanh|61.1|75.0|57.6|35.4|73.8|85.2|58.5|63.8|
|||||Scaling n in|DHC|||||
|OLMo-1B-DHCx1||59.7|74.0|55.5|33.6|73.5|85.4|54.5|62.3|
|OLMo-1B-DHCx2||59.7|73.0|56.7|34.0|74.7|85.2|57.9|63.0|
|OLMo-1B-DHCx4||59.8|79.0|58.1|32.4|74.3|86.1|57.1|63.8|
|OLMo-1B-DHCx8||56.8|75.0|58.0|34.4|73.8|84.2|57.3|62.8|
|||||Scaling n in|SHC|||||
|OLMo-1B-SHCx2||59.1|77.0|56.6|35.4|74.2|85.3|56.4|63.4|
|OLMo-1B-SHCx4||59.3|77.0|56.7|34.0|74.3|86.6|57.1|63.6|
|||||Non-trainable_WC_||||||
|OLMo-1B-DHCx4||60.5|78.0|56.2|34.0|73.5|86.0|55.8|63.4|
|OLMo-1B-DHCx4|W/O tanh|59.1|72.0|56.8|35.0|73.3|86.0|55.5|62.5|
|||||Non-trainable**B**||||||
|OLMo-1B-DHCx4||59.5|77.0|57.9|33.8|73.3|85.6|56.6|63.4|
|OLMo-1B-DHCx4|W/O tanh|60.4|74.0|57.6|34.0|74.9|86.7|57.5|63.6|



33 

Published as a conference paper at ICLR 2025 

|OLMo-1B-DHCx4<br>2.296<br>2.594<br>2.742<br>3.348<br>2.684<br>3.051<br>2.569<br>3.008<br>2.497<br>2.221<br>2.917<br>3.627<br>2.622<br>2.783<br>OLMo-1B-DHCx4 W/O tanh<br>2.295<br>2.592<br>2.739<br>3.347<br>2.689<br>3.066<br>2.567<br>3.005<br>2.496<br>2.222<br>2.887<br>3.638<br>2.606<br>2.781|Non-trainable Beta|OLMo-1B-DHCx4<br>2.312<br>2.608<br>2.752<br>3.357<br>2.700<br>3.077<br>2.583<br>3.024<br>2.508<br>2.238<br>2.959<br>3.678<br>2.636<br>2.802<br>OLMo-1B-DHCx4 W/O tanh<br>2.308<br>2.609<br>2.755<br>3.357<br>2.710<br>3.100<br>2.585<br>3.025<br>2.510<br>2.240<br>2.945<br>3.663<br>2.644<br>2.804|Non-trainable WC|OLMo-1B-SHCx2<br>2.307<br>2.610<br>2.757<br>3.360<br>2.703<br>3.063<br>2.587<br>3.023<br>2.511<br>2.238<br>2.933<br>3.643<br>2.643<br>2.799<br>OLMo-1B-SHCx4<br>2.300<br>2.603<br>2.751<br>3.357<br>2.692<br>3.062<br>2.580<br>3.018<br>2.504<br>2.232<br>2.899<br>3.653<br>2.627<br>2.791|Scaling n in SHC|OLMo-1B-DHCx1<br>2.323<br>2.625<br>2.775<br>3.376<br>2.728<br>3.090<br>2.606<br>3.037<br>2.533<br>2.262<br>2.961<br>3.652<br>2.678<br>2.819<br>OLMo-1B-DHCx2<br>2.309<br>2.608<br>2.754<br>3.367<br>2.703<br>3.061<br>2.587<br>3.022<br>2.509<br>2.237<br>2.930<br>3.704<br>2.636<br>2.802<br>OLMo-1B-DHCx4<br>2.290<br>2.591<br>2.738<br>3.354<br>2.683<br>3.064<br>2.564<br>3.005<br>2.492<br>2.218<br>2.890<br>3.641<br>2.611<br>2.781<br>OLMo-1B-DHCx8<br>2.295<br>2.591<br>2.739<br>3.353<br>2.684<br>3.054<br>2.567<br>3.008<br>2.493<br>2.219<br>2.876<br>3.631<br>2.608<br>2.778|Scaling n in DHC|OLMo-1B-DHCx1 W/O tanh<br>2.320<br>2.626<br>2.773<br>3.379<br>2.725<br>3.102<br>2.609<br>3.036<br>2.531<br>2.264<br>2.948<br>3.703<br>2.672<br>2.822<br>OLMo-1B-DHCx2 W/O tanh<br>2.311<br>2.600<br>2.749<br>3.362<br>2.700<br>3.069<br>2.583<br>3.015<br>2.503<br>2.231<br>2.908<br>3.635<br>2.625<br>2.792<br>OLMo-1B-DHCx4 W/O tanh<br>2.295<br>2.591<br>2.735<br>3.344<br>2.686<br>3.056<br>2.562<br>3.005<br>2.492<br>2.221<br>2.898<br>3.632<br>2.610<br>2.779<br>OLMo-1B-DHCx8 W/O tanh<br>2.292<br>2.589<br>2.734<br>3.350<br>2.685<br>3.060<br>2.562<br>3.006<br>2.492<br>2.218<br>2.878<br>3.628<br>2.609<br>2.777|Scaling n in DHC W/O tanh|OLMo-1B<br>2.319<br>2.615<br>2.762<br>3.364<br>2.719<br>3.085<br>2.594<br>3.028<br>2.522<br>2.250<br>2.953<br>3.672<br>2.657<br>2.811|**Method**<br>**4chan**<br>**c4_100_domains**<br>**c4_en**<br>**gab**<br>**ice**<br>**m2d2_s2orc**<br>**m2d2_wiki**<br>**manosphere**<br>**mc4_en**<br>**pile**<br>**ptb**<br>**twitterAAE**<br>**wikitext_103**<br>**avg**|
|---|---|---|---|---|---|---|---|---|---|---|---|



34 

Published as a conference paper at ICLR 2025 

|OLMo-1B-DHCx4<br>9.927<br>13.354<br>15.475<br>28.417<br>14.722<br>21.454<br>13.021<br>20.185<br>12.135<br>9.228<br>17.932<br>38.005<br>13.553<br>17.493<br>OLMo-1B-DHCx4 W/O tanh<br>9.932<br>13.386<br>15.510<br>28.436<br>14.641<br>21.130<br>13.051<br>20.253<br>12.142<br>9.220<br>18.478<br>37.610<br>13.766<br>17.504|Non-trainable Beta|OLMo-1B-DHCx4<br>10.054<br>13.587<br>15.721<br>28.689<br>15.023<br>22.186<br>13.263<br>20.594<br>12.310<br>9.390<br>19.016<br>38.959<br>14.070<br>17.912<br>OLMo-1B-DHCx4 W/O tanh<br>10.092<br>13.566<br>15.666<br>28.704<br>14.873<br>21.696<br>13.242<br>20.579<br>12.276<br>9.377<br>19.272<br>39.570<br>13.963<br>17.914|Non-trainable WC|OLMo-1B-SHCx2<br>10.046<br>13.601<br>15.753<br>28.782<br>14.931<br>21.391<br>13.294<br>20.562<br>12.319<br>9.374<br>18.791<br>38.212<br>14.060<br>17.778<br>OLMo-1B-SHCx4<br>9.977<br>13.507<br>15.655<br>28.691<br>14.766<br>21.372<br>13.194<br>20.457<br>12.234<br>9.315<br>18.149<br>38.569<br>13.836<br>17.671|Scaling n in SHC|OLMo-1B-DHCx1<br>10.210<br>13.810<br>16.031<br>29.265<br>15.302<br>21.986<br>13.539<br>20.847<br>12.584<br>9.606<br>19.326<br>38.564<br>14.555<br>18.125<br>OLMo-1B-DHCx2<br>10.061<br>13.568<br>15.710<br>29.002<br>14.925<br>21.349<br>13.284<br>20.524<br>12.294<br>9.362<br>18.727<br>40.592<br>13.957<br>17.950<br>OLMo-1B-DHCx4<br>9.877<br>13.344<br>15.430<br>28.624<br>14.633<br>21.410<br>13.006<br>20.186<br>12.080<br>9.189<br>18.102<br>38.136<br>13.606<br>17.509<br>OLMo-1B-DHCx8<br>9.922<br>13.346<br>15.467<br>28.591<br>14.640<br>21.198<br>13.025<br>20.240<br>12.097<br>9.196<br>17.749<br>37.743<br>13.570<br>17.445|Scaling n in DHC|OLMo-1B-DHCx1 W/O tanh<br>10.174<br>13.815<br>16.004<br>29.328<br>15.259<br>22.231<br>13.587<br>20.823<br>12.562<br>9.620<br>19.071<br>40.580<br>14.462<br>18.270<br>OLMo-1B-DHCx2 W/O tanh<br>9.920<br>13.340<br>15.412<br>28.340<br>14.676<br>21.243<br>12.965<br>20.181<br>12.079<br>9.219<br>18.129<br>37.768<br>13.594<br>17.451<br>OLMo-1B-DHCx4 W/O tanh<br>10.082<br>13.470<br>15.625<br>28.848<br>14.882<br>21.521<br>13.234<br>20.392<br>12.217<br>9.312<br>18.321<br>37.905<br>13.806<br>17.663<br>OLMo-1B-DHCx8 W/O tanh<br>9.897<br>13.313<br>15.387<br>28.488<br>14.658<br>21.337<br>12.960<br>20.200<br>12.084<br>9.185<br>17.782<br>37.650<br>13.592<br>17.425|Scaling n in DHC W/O tanh|OLMo-1B<br>10.167<br>13.666<br>15.829<br>28.901<br>15.166<br>21.860<br>13.377<br>20.651<br>12.453<br>9.488<br>19.161<br>39.328<br>14.251<br>18.023|**Method**<br>**4chan**<br>**c4_100_domains**<br>**c4_en**<br>**gab**<br>**ice**<br>**m2d2_s2orc**<br>**m2d2_wiki**<br>**manosphere**<br>**mc4_en**<br>**pile**<br>**ptb**<br>**twitterAAE**<br>**wikitext_103**<br>**avg**|Table 17: Perplexities of V2 validation sets for 1B models.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|



35 

Published as a conference paper at ICLR 2025 

|OLMo-1B-DHCx4<br>2.679<br>2.880<br>2.697<br>2.306<br>2.961<br>1.025<br>2.458<br>2.684<br>3.188<br>2.204<br>2.612<br>2.518<br>OLMo-1B-DHCx4 W/O tanh<br>2.681<br>2.886<br>2.702<br>2.306<br>2.966<br>1.024<br>2.462<br>2.680<br>3.183<br>2.204<br>2.628<br>2.520|Non-trainable Beta|OLMo-1B-DHCx4<br>2.695<br>2.903<br>2.716<br>2.324<br>2.978<br>1.035<br>2.477<br>2.705<br>3.201<br>2.221<br>2.649<br>2.537<br>OLMo-1B-DHCx4 W/O tanh<br>2.692<br>2.899<br>2.714<br>2.321<br>2.976<br>1.032<br>2.474<br>2.695<br>3.189<br>2.219<br>2.641<br>2.532|Non-trainable WC|OLMo-1B-SHCx2<br>2.698<br>2.907<br>2.718<br>2.325<br>2.980<br>1.032<br>2.479<br>2.700<br>3.198<br>2.221<br>2.650<br>2.537<br>OLMo-1B-SHCx4<br>2.689<br>2.892<br>2.711<br>2.315<br>2.973<br>1.028<br>2.472<br>2.688<br>3.195<br>2.214<br>2.633<br>2.528|Scaling n in SHC|OLMo-1B-DHCx1<br>2.714<br>2.927<br>2.732<br>2.346<br>2.991<br>1.045<br>2.499<br>2.723<br>3.211<br>2.245<br>2.683<br>2.556<br>OLMo-1B-DHCx2<br>2.694<br>2.901<br>2.712<br>2.321<br>2.976<br>1.032<br>2.478<br>2.699<br>3.202<br>2.218<br>2.642<br>2.534<br>OLMo-1B-DHCx4<br>2.675<br>2.876<br>2.697<br>2.301<br>2.962<br>1.021<br>2.455<br>2.679<br>3.176<br>2.200<br>2.617<br>2.515<br>OLMo-1B-DHCx8<br>2.677<br>2.880<br>2.701<br>2.304<br>2.964<br>1.022<br>2.456<br>2.680<br>3.177<br>2.201<br>2.614<br>2.516|Scaling n in DHC|OLMo-1B-DHCx1 W/O tanh<br>2.712<br>2.928<br>2.732<br>2.349<br>2.991<br>1.045<br>2.499<br>2.721<br>3.219<br>2.246<br>2.677<br>2.556<br>OLMo-1B-DHCx2 W/O tanh<br>2.676<br>2.880<br>2.698<br>2.306<br>2.961<br>1.024<br>2.456<br>2.682<br>3.174<br>2.204<br>2.617<br>2.516<br>OLMo-1B-DHCx4 W/O tanh<br>2.689<br>2.890<br>2.706<br>2.317<br>2.969<br>1.030<br>2.471<br>2.697<br>3.200<br>2.213<br>2.633<br>2.529<br>OLMo-1B-DHCx8 W/O tanh<br>2.674<br>2.876<br>2.695<br>2.303<br>2.960<br>1.022<br>2.454<br>2.680<br>3.176<br>2.200<br>2.616<br>2.514|Scaling n in DHC W/O tanh|OLMo-1B<br>2.702<br>2.906<br>2.722<br>2.333<br>2.980<br>1.041<br>2.487<br>2.715<br>3.199<br>2.232<br>2.663<br>2.544|**Method**<br>**c4_en**<br>**dolma_books**<br>**dolma_common-crawl**<br>**dolma_pes2o**<br>**dolma_reddit**<br>**dolma_stack**<br>**dolma_wiki**<br>**ice**<br>**m2d2_s2orc**<br>**pile**<br>**wikitext_103**<br>**avg**|
|---|---|---|---|---|---|---|---|---|---|---|---|



36 

Published as a conference paper at ICLR 2025 

|OLMo-1B-DHCx4<br>14.574<br>17.820<br>14.840<br>10.038<br>19.320<br>2.787<br>11.677<br>14.647<br>24.233<br>9.059<br>13.621<br>13.874<br>OLMo-1B-DHCx4 W/O tanh<br>14.593<br>17.926<br>14.904<br>10.032<br>19.405<br>2.785<br>11.724<br>14.588<br>24.108<br>9.060<br>13.839<br>13.906|Non-trainable Beta|OLMo-1B-DHCx4<br>14.810<br>18.224<br>15.120<br>10.215<br>19.650<br>2.816<br>11.902<br>14.954<br>24.552<br>9.220<br>14.135<br>14.145<br>OLMo-1B-DHCx4 W/O tanh<br>14.756<br>18.160<br>15.095<br>10.191<br>19.613<br>2.806<br>11.868<br>14.807<br>24.273<br>9.203<br>14.021<br>14.072|Non-trainable WC|OLMo-1B-SHCx2<br>14.854<br>18.293<br>15.150<br>10.230<br>19.689<br>2.807<br>11.934<br>14.876<br>24.478<br>9.214<br>14.150<br>14.152<br>OLMo-1B-SHCx4<br>14.717<br>18.028<br>15.049<br>10.121<br>19.550<br>2.796<br>11.846<br>14.699<br>24.407<br>9.155<br>13.912<br>14.025|Scaling n in SHC|OLMo-1B-DHCx1<br>15.093<br>18.675<br>15.360<br>10.442<br>19.909<br>2.845<br>12.174<br>15.225<br>24.810<br>9.436<br>14.632<br>14.418<br>OLMo-1B-DHCx2<br>14.794<br>18.190<br>15.061<br>10.191<br>19.612<br>2.806<br>11.915<br>14.870<br>24.589<br>9.187<br>14.043<br>14.114<br>OLMo-1B-DHCx4<br>14.514<br>17.743<br>14.829<br>9.989<br>19.343<br>2.776<br>11.650<br>14.573<br>23.948<br>9.028<br>13.689<br>13.826<br>OLMo-1B-DHCx8<br>14.546<br>17.807<br>14.889<br>10.011<br>19.366<br>2.779<br>11.653<br>14.579<br>23.964<br>9.030<br>13.653<br>13.843|Scaling n in DHC|OLMo-1B-DHCx1 W/O tanh<br>15.064<br>18.699<br>15.356<br>10.473<br>19.909<br>2.843<br>12.167<br>15.191<br>25.013<br>9.451<br>14.540<br>14.428<br>OLMo-1B-DHCx2 W/O tanh<br>14.531<br>17.817<br>14.857<br>10.038<br>19.323<br>2.783<br>11.662<br>14.608<br>23.906<br>9.061<br>13.694<br>13.844<br>OLMo-1B-DHCx4 W/O tanh<br>14.711<br>17.996<br>14.975<br>10.146<br>19.479<br>2.800<br>11.830<br>14.839<br>24.524<br>9.146<br>13.917<br>14.033<br>OLMo-1B-DHCx8 W/O tanh<br>14.494<br>17.749<br>14.813<br>10.000<br>19.306<br>2.779<br>11.630<br>14.587<br>23.948<br>9.021<br>13.684<br>13.819|Scaling n in DHC W/O tanh|OLMo-1B<br>14.908<br>18.289<br>15.216<br>10.305<br>19.686<br>2.832<br>12.026<br>15.098<br>24.503<br>9.319<br>14.334<br>14.229|**Method**<br>**c4_en**<br>**dolma_books**<br>**dolma_common-crawl**<br>**dolma_pes2o**<br>**dolma_reddit**<br>**dolma_stack**<br>**dolma_wiki**<br>**ice**<br>**m2d2_s2orc**<br>**pile**<br>**wikitext_103**<br>**avg**|Table 19: Perplexities of V3 validation sets for 1B models.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|



37 

