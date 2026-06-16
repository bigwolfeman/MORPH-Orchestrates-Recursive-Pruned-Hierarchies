# Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA

- **Authors:** Hai Huang, Yann LeCun, Randall Balestriero (galilai-group / NYU)
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2602.22617
- **MORPH uses:** The geodesic smoothness constraint applied to hidden-state trajectories during pretraining - confining intermediate states to lie within a tubular neighborhood of the geodesic connecting segment boundaries on the semantic manifold. MORPH applies STP during pretraining (the paper only attempts fine-tuning) with a multi-scale scheme (strides 1,2,4,...,tau=64).

---

**Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA** 

**Hai Huang**[1] **Yann LeCun**[2] **Randall Balestriero**[3] 

## **Abstract** 

Large Language Models (LLMs) obey consistent scaling laws—empirical power-law fits that predict how loss decreases with compute, data, and parameters. While predictive, these laws are descriptive rather than prescriptive: they characterize typical training, not optimal training. Surprisingly few works have successfully challenged the data-efficiency bounds implied by these laws—which is our primary focus. To that end, we introduce the Geodesic Hypothesis, positing that token sequences trace geodesics on a smooth semantic manifold and are therefore locally linear. Building on this principle, we propose a novel Semantic Tube Prediction (STP) task, a JEPA-style regularizer that confines hidden-state trajectories to a tubular neighborhood of the geodesic. STP generalizes JEPA to language without requiring explicit multi-view augmentations. We show this constraint improves signal-to-noise ratio, and consequently preserves diversity by preventing trajectory collisions during inference. Empirically, STP allows LLMs to match baseline accuracy with 16 _×_ less training data on the NL-RX-SYNTH dataset, directly violating the data term of Chinchilla-style scaling laws and demonstrating that principled geometric priors can surpass brute-force scaling. Code is available at https://github.com/ galilai-group/llm-jepa#stp. 

## **1. Introduction** 

We argue that empirical scaling laws characterize _typical_ rather than _optimal_ training, suggesting the rigid power-law barrier is an artifact of current objectives. The core limitation is next-token prediction: a local objective that conflates surface statistical noise with global semantic signal. We 

1Atlassian 2NYU 3Brown. Correspondence to: Hai Huang _<_ hhuang3@atlassian.com _>_ , Randall Balestriero _<_ randall ~~b~~ alestriero@brown.edu _>_ . 

_Preprint. February 27, 2026._ 

**==> picture [218 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Semantic Tube<br>80<br>60<br>mmm mmm TTT TTT TTT TTT Tm mmm meee<br>40<br>20<br>_——— Cr NTP [+] Lr STP [ (Ours)] -@- 0.5Flop,2lr<br>——— NTP B 0.5Flop,2lr,2 2<br>—%- 1Flop —%- 1Flop<br>0 + 1Flop,2 A a 0.5Flop,2lr<br>1/32 1/16 1/8 1/4 1/2<br>Data Size<br>Accuracy (%)<br>**----- End of picture text -----**<br>


**==> picture [70 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  Data Efficiency<br>**----- End of picture text -----**<br>


_Figure 1._ Semantic Tube improves data efficiency. **(a)** We hypothesize that error-free hidden state trajectories are geodesics, which are locally linear and approximated by the Semantic Tube. The dotted line depicts a trajectory distorted by training loss. Deviations perpendicular to the tube constitute _noise_ , while the component along the geodesic represents the _signal_ . **(b)** With our approach ( _L_ NTP + _L_ STP), accuracy shows a negligible drop when the training dataset is halved, and it matches full-dataset standard fine-tuning ( _L_ NTP) accuracy using only 161[of the training data.][In] contrast, _L_ NTP degrades significantly when the dataset is halved. 

propose a fundamental shift: explicitly constraining hidden state dynamics to separate the error-free semantic trajectory from this noise. 

First, we formally demonstrate that, although tokens are discrete, token sequences can be modeled by an Ordinary Differential Equation (ODE). The Picard-Lindelof¨ (Exis- 

1 

**Semantic Tube Prediction** 

tence and Uniqueness) Theorem (Coddington & Levinson, 1955) guarantees that if the velocity is smooth enough, there is only one possible path forward from any starting point. In other words, trajectories originating from distinct initial states will never intersect. In the context of LLMs, if the ODE model holds, this implies that _error-free_ generations from distinct prompts maintain their semantic separation, theoretically ruling out mode collapse and preserving diversity. 

Next, we hypothesize that the Principle of Least Action (Lanczos, 1966) is at work. This principle states that the path taken by a system between two points minimizes the “Action” (the integral of the Lagrangian over time), resulting in a “straight line” or geodesic on the underlying manifold. We further hypothesize that, as the manifold is an artifact of the training process, it admits a smooth structure. Consequently, the geodesics are locally linear almost everywhere. In the context of LLMs, this implies that the trajectories of error-free token sequences—and by extension, the trajectories of error-free hidden states—are confined within a tube centered along a straight line. 

We designate this structure the **Semantic Tube** (Figure 1) and leverage it to regularize the LLM training process. The Semantic Tube posits that the noise—which causes deviations from the error-free trajectories—concentrates along the directions perpendicular to the tube. Let _s < r < t_ denote the indices of three tokens. We define the _noise_ term as ( _hr − hs_ ) _⊥ht−hs_ , representing the component of _hr − hs_ perpendicular to _ht − hs_ , and the _signal_ term as ( _hr − hs_ ) _∥ht−hs_ , representing the component parallel to _ht − hs_ . Minimizing the noise term is expected to improve the Signal-to-Noise Ratio (SNR) during training. We formulate this as an auxiliary loss term, the Semantic Tube Prediction (STP) loss _L_ STP, which can be seamlessly integrated into the training objective: 

**==> picture [93 x 10] intentionally omitted <==**

where _L_ NTP is the cross-entropy loss for Next Token Prediction (NTP) and _λ_ is a hyperparameter controlling the strength of the STP loss. 

Semantic Tube draws inspiration from the Joint-Embedding Predictive Architecture (JEPA) (Assran et al., 2023; Baevski et al., 2022), which learns to predict the representation of one view based on another. In our approach, we postulate that any segment of a token sequence aligns with the global trajectory; consequently, the predictor reduces to an identity function. 

If the Geodesic Hypothesis holds, it entails the following predictions: 

- (P1) _L_ NTP alone is insufficient for high-quality generation. Consequently, we expect to observe _L_ NTP 

plateau even as _L_ STP continues to decrease. 

- (P2) Semantic Tube improves SNR, resulting in superior data efficiency (Figure 1) and accuracy. 

- (P3) Semantic Tube preserves diversity. 

- (P4) We expect to see _λ ≪_ 1 to accommodate instances where the geodesic deviates from a straight line. 

- (P5) The identity function serves as a superior predictor compared to learned projections. 

We conducted extensive experiments validating predictions (P1) through (P5). These results provide a strong indication that the Geodesic Hypothesis represents a simplified form of self-consistency for autoregressive sequence models. Furthermore, they confirm the validity of the noise/signal decomposition (Figure 1) and establish Semantic Tube as an effective self-supervised learning objective for LLMs. 

## **2. Training and Inference Dynamics** 

In this section, we formally analyze training and inference dynamics, proposing that token sequence trajectories can be modeled by an Ordinary Differential Equation (ODE) characterized by ballistic trajectories. 

## **2.1. Training ODE** 

Let _x≤t_ denote a token sequence of length _t_ , where _xt_ represents the _t_ -th token, _ht_ is the corresponding hidden state, and _f_ ( _·_ ) denotes the neural network such that _ht_ = _f_ ( _x≤t_ ). Each hidden state _ht_ is subsequently unembedded to predict the next token _xt_ +1. 

During training, the predicted token _u_ ( _ht_ ) may diverge from the ground truth _xt_ +1; this discrepancy constitutes the training loss. However, due to teacher forcing, we invariably feed the ground truth sequence _x≤t_ +1 into _f_ ( _·_ ) to generate _ht_ +1. Consequently, assuming a converged network where the loss is minimized, the training dynamics can be modeled as: 

**==> picture [160 x 31] intentionally omitted <==**

where _f_[˚] and ˚ _u_ represent the functions of the converged network, and _ϵt_ denotes the residual unembedding error. 

If a time-indexed variable _zt_ follows the difference equation _zt_ +1 _− zt_ = _g_ ( _zt, t_ ), it can be approximated by an ODE of the form _dzt_ = _g_ ( _zt, t_ ) _dt_ . While the hidden state dynamics in Equation (2) do not fit this form (as _ht_ +1 depends on the entire history _x≤t_ rather than just _ht_ ), the sequence dynamics in Equation (1) do. Specifically, _x≤t_ +1 = _x≤t ⊕_ 

2 

**Semantic Tube Prediction** 

˚ _xt_ +1 = _x≤t ⊕ u ◦ f_[˚] ( _x≤t_ ), where _⊕_ denotes concatenation. Letting _⊖_ denote the prefix-removal operator, we obtain: 

**==> picture [115 x 13] intentionally omitted <==**

This formulation closely resembles the update rule _zt_ +1 _− zt_ = _g_ ( _zt, t_ ), suggesting that an ODE is a plausible model for the dynamics. 

Although tokens are discrete, their embeddings lie in a continuous vector space _xt ∈_ R _[d]_[model] . Let _T_ denote the maximum sequence length; then the sequence resides in R _[T][ ×][d]_[model] . In Section A, we demonstrate that under specific arrangements, the operation _x≤t_ +1 _⊖x≤t_ can be treated as vector subtraction _x≤t_ +1 _− x≤t_ . This leads to the following proposition: 

**Proposition 2.1** (Training ODE) **.** _The LLM training process can be modeled as a solution in the token sequence space_ R _[T][ ×][d]_[model] _to the ODE:_ 

**==> picture [91 x 14] intentionally omitted <==**

Theorem 2.1 models _x≤t_ as following a ballistic trajectory in R _[T][ ×][d]_[model] . The Picard-Lindelof Theorem guarantees that¨ if ˚ _u ◦ f_[˚] ( _·_ ) and its partial derivatives with respect to _x≤t_ are continuous, the ODE admits a unique solution for a given initial condition. Consequently, within this ODE framework, sequences generated from distinct prompts (initial conditions) cannot intersect, theoretically ruling out mode collapse, and preserving diversity. 

## **2.2. Mode Collapse at Inference Time** 

Let _h[∗]_ denote the optimal trajectory of hidden states, defined as: 

**==> picture [167 x 14] intentionally omitted <==**

If _f_[˚] ( _·_ ) is Lipschitz-continuous (Khalil, 2002), then the trajectory _h[∗]_ is also ballistic. 

However, _L_ NTP alone may not suffice to drive _ϵt_ to zero. Recall that the goal of _L_ NTP is to converge _u_ ( _ht_ ) to _xt_ +1. Since the hidden state _ht_ is continuous while the token _xt_ +1 is discrete, the training process can be modeled as finding the correct Voronoi cell (Okabe et al., 2000), without stipulating the exact location within the cell. This flexibility is necessary for the Picard-Lindelof Theorem to apply:¨ as illustrated in Figure 2, it allows error-free geodesics ( _h[∗] t_[) to] traverse the same Voronoi cell at distinct locations, thereby avoiding intersection. Nevertheless, _ht_ may drift onto an incorrect geodesic within the cell, leading to mode collapse. 

This analysis indicates that _L_ NTP alone is insufficient for generation quality, strongly motivating an additional loss 

_Figure 2._ Two hidden state trajectories with similar prefixes pass through the Voronoi cell of the “ **researcher** ” token at different locations, leading to different next hidden states and hence different next tokens. Since _L_ NTP cannot guarantee that _ht_ converges to _h[∗] t_ (optimal hidden state), _ht_ can be misplaced on another geodesic. This leads to mode collapse (the red dotted line mistakenly continues the generation, misattributing Hinton’s Nobel Prize to an arbitrary person, or if the error deviates in the opposite direction and precludes a winner). 

term ( _L_ STP) to explicitly minimize _ϵt_ . It also implies that within the correct Voronoi cell, _L_ NTP may plateau while _L_ STP continuously decreases. Therefore, (P1). 

In Section B, we demonstrate that in the infinite-width limit (Yang & Littwin, 2021), the inference process can be modeled as a Stochastic Differential Equation (SDE) with a Brownian motion term. 

## **3. Semantic Tube Prediction** 

A key challenge in minimizing the error _ϵt_ is that the optimal trajectory _h[∗]_ remains latent and unknown. To address this, we must postulate a structural property that allows us to estimate _h[∗]_ , leading us to the Geodesic Hypothesis. In this section, we formalize this hypothesis and subsequently introduce Semantic Tube Prediction (STP). 

## **3.1. Semantic Tube** 

If the Principle of Least Action holds, the trajectories of the token sequence _x≤t_ +1 in Equation (1) must be geodesics, which˚ are locally linear almost everywhere. Since _h[∗] t_[=] _f_ ( _x≤t_ ), when _f_[˚] ( _·_ ) is smooth enough, _h[∗] t_[is also expected] to be locally linear almost everywhere. Hence the Geodesic Hypothesis: 

_The trajectory of x≤t ∈_ R _[T][ ×][d]_[model] _is locally linear almost everywhere. Similarly, the trajectory ht − ϵt ∈_ R _[d] is locally linear almost everywhere._ 

3 

**Semantic Tube Prediction** 

We first formally define local linearity. Subsequently, we demonstrate that the Semantic Tube compresses the trajectory _ht_ within a tube centered at _h[∗] t_[.] 

**Definition 3.1** (Local Linearity) **.** A time-indexed trajectory _h[∗]_ is defined as locally linear if _∃τ, ∃ε_ such that for any time indices _s < r < t_ satisfying _|t − s| ≤ τ_ , we have: 

**==> picture [171 x 12] intentionally omitted <==**

where _x⊥y_ denotes the component of vector _x_ that is perpendicular to vector _y_ . 

Theorem 3.1 captures the intuition that if a trajectory is locally linear, each local segment can be approximated by a straight line connecting its endpoints. 

Next, we demonstrate that the Semantic Tube forces _h_ to approximate _h[∗]_ . 

**Lemma 3.2** (Straightening Lemma) **.** _If hs_ = _h[∗] s[,][ h][t]_[=] _[ h][∗] t[,] and L_ STP _≤ ϵ for all r satisfying s < r < t, then_ 

**==> picture [167 x 14] intentionally omitted <==**

## Proof is deferred to Section D. 

Let _∥hr − h[∗] ∥_ 2 = min _r′ ∥hr − h[∗] r[′][∥]_[2][denote the minimum] distance from _hr_ to the trajectory _h[∗]_ . We establish the following theorem: 

**Theorem 3.3** (Semantic Tube) **.** _If h[∗] is locally linear and for all r satisfying_ 0 _≤ s < r < t ≤ τ , L_ STP _→_ 0 _, then_ 

**==> picture [66 x 11] intentionally omitted <==**

_Proof Sketch._ Only prove for the case _hs_ = _h[∗] s_[and] _[ h][t]_[=] _h[∗] t_[.][In][this][scenario,] _[∥][h][r][−][h][s][∥]_[2][=] _[∥][h][r][−][h][∗] s[∥]_[.][Apply-] ing the triangle inequality yields _∥hr − h[∗] s[∥≤∥][h][∗] r[−] h[∗] s[∥]_[2][+] _[ ϵ][r]_[.][Notice] _[h][∗] r_[and] _[h][∗] s_[are][fixed,][by][Theorem][3.2][,] _∥_ ( _hr − hs_ ) _⊥h[∗] t[−][h][∗] s[∥]_[2] _[→]_[0][.][By Theorem][ 3.1][ and the trian-] gle inequality, it follows that _∥hr − h[∗] ∥_ 2 ≲ _ε_ 

In LLMs, it is standard to assume all sequences begin with <bos> and end with <eos>; thus, it is reasonable to assume the boundary conditions _h_ 0 = _h[∗]_ 0[and] _[ h][τ]_[=] _[ h][∗] τ_[.][This] is formally proven in Section E, which completes the proof of Theorem 3.3. 

In practice, the indices _s < r < t_ are selected randomly. Consequently, minimizing _L_ STP effectively drives E[1 _−_ cos( _ht − hr, hr − hs_ )] _→_ 0. By Markov’s inequality, for any _ϵ_ , _P_ (1 _−_ cos( _ht − hr, hr − hs_ ) _> ϵ_ ) _→_ 0. This leads to the following corollary: 

**Corollary 3.4** (Random Tube) **.** _For randomly selected s < r < t, if L_ STP _→_ 0 _, then for any ϵ,_ 

**==> picture [119 x 11] intentionally omitted <==**

Theorem 3.4 implies that if _L_ STP _→_ 0 for a given sequence, then with high probability, the trajectory of the sequence’s hidden states is confined within a tube centered around the optimal trajectory _h[∗]_ . 

However, at inference time, the Brownian motion term diverges into a cone whose radius scales as _∝ σt√t_ , see Section F for details. 

## **3.2. Practical Considerations** 

Since the forward pass naturally computes _hs_ , _hr_ , and _ht_ , the STP loss introduces negligible computational overhead—primarily the cost of computing cosine similarity. This is significantly more efficient than the fractional extra forward passes required by LLM-JEPA (Huang et al., 2025). Furthermore, because indices _s_ , _r_ , and _t_ can be selected randomly, STP eliminates the need for manual scaffolding of a two-view structure. In summary, STP effectively addresses the two primary limitations that have hindered the broader adoption of LLM-JEPA. Additionally, STP avoids the complexity of a predictor network (often a requirement in LLM-JEPA), as local linearity implies an identity predictor. Like LLM-JEPA, the STP loss is applied exclusively during training and is not required at inference time. 

Further implementation details are provided in Section G. 

## **3.3. Related Work** 

Our approach addresses the classic **Exposure Bias** problem (Bengio et al., 2015), originally identified in recurrent neural networks (RNNs) (Elman, 1990; Siegleman & Sontag, 1995). The problem arises because the model is trained with **Teacher Forcing** (Williams & Zipser, 1989)— conditioning on the ground-truth history—but must rely on its own potentially drifting predictions during inference. Although Maximum Likelihood Estimation ( _L_ NTP in the case of LLMs) is empirically effective, Huszar´ (2015) argues that it optimizes an objective different from generation quality, motivating our combined loss _L_ NTP + _L_ STP. 

**JEPAs** (Assran et al., 2023; Baevski et al., 2022) learn predictive representations across views, offering theoretical benefits (Littwin et al., 2024) despite the risk of dimensional collapse (Jing et al., 2021; Kenneweg et al., 2025). While recent works extend these objectives to LLMs (Barrault et al., 2024; Wang & Sun, 2025), LLM-JEPA (Huang et al., 2025) is bottlenecked by manual two-view scaffolding and the computational cost of additional forward passes, neither is a problem for _L_ STP. 

4 

**Semantic Tube Prediction** 

Our framework extends the philosophy of **Energy-Based Models** (EBMs) (LeCun et al., 2006), which learn to assign low energy to compatible configuration of variables. While EBMs and recent architectures like JEPA (LeCun, 2022) typically minimize energy at specific states, our approach invokes the Principle of Least Action to minimize the action—the integral of the Lagrangian along the generation trajectory. By enforcing geodesic constraints via _L_ STP, we generalize state-wise (or local) energy minimization to trajectory-wise action minimization, ensuring the generation follows the path of least resistance. 

**Scaling Laws** govern the power-law relationship between compute, data, and parameters in both pre-training (Kaplan et al., 2020; Hoffmann et al., 2022) and fine-tuning (Zhang et al., 2024). While recent data efficiency research emphasizes identifying high-density subsets (Sorscher et al., 2022) or synthetic curation (Gunasekar et al., 2023; Muennighoff et al., 2023), _L_ STP enhances the training SNR directly, obviating the need for explicit data subset selection. 

**SDE/ODE Perspective** : Kong et al. (2020) interpreted ResNets as “Neural SDEs” which has a Brownian motion term. While Tong et al. (2025) recently adapted ODEs for LLMs, they model evolution across network depth (layers). Our work takes an orthogonal approach, focusing instead on the temporal dynamics of hidden states across the token sequence. 

**The Linear Representation Hypothesis** (LRH) (Park et al., 2024; 2025) posits that simple concepts are encoded as directions in the representation space, whereas the Geodesic Hypothesis suggests that both simple and composed concepts (expressed as token sequences) follow locally linear trajectories. Consequently, the vector arithmetic observed in LRH ( _⃗vP aris − ⃗vF rance_ + _⃗vItaly ≈ ⃗vRome_ ) emerges naturally from path linearity ( _⃗vP aris,⃗vto,⃗vF rance,⃗vis,⃗vRome,⃗vto,⃗vItaly_ aligns on almost a straight line, see Figure 3). 

**The Manifold Hypothesis** (Kiani et al., 2024; Robinson et al., 2025; Whiteley et al., 2025) posits that learned representations form a simple and smooth manifold. Under the Geodesic Hypothesis, this structure is a natural consequence of the Principle of Least Action. 

**The Curvature Straightening Phenomenon** (Hosseini & Fedorenko, 2023; Henaff´ et al., 2021) observes that the training process tends to straighten the curvature between consecutive tokens. We interpret this as a manifestation of the underlying geodesic, which approximates a straight line. 

The **Neural Tangent Kernel (NTK)** simplifies infinitewidth dynamics (Jacot et al., 2018), a framework generalized to Transformers (Hron et al., 2020; Yang & Littwin, 2021) and compatible feature learning regimes (Yang & Hu, 2021). While Seleznova & Kutyniok (2022) note the importance of the depth-to-width ratio, modern LLMs typically operate in the requisite width _≫_ depth regime. 

The application of **geodesic geometry to LLMs** remains underexplored, with existing studies primarily restricted to interpolating representations across models (Deng et al., 2025; Yu et al., 2024). 

## **4. Experiments** 

We conduct extensive experiments to show the performance of Semantic Tube across models, datasets, and model sizes. We also show that accuracy barely budges when the training dataset is halved. Both accuracy and data efficiency are solid evidence that Semantic Tube improves SNR. We ablate on various setups, including LLM-JEPA style explicit twoviews and curvature straightening. Lastly we show how to tune _λ_ in practices. 

Implementing _L_ STP is straightforward with HuggingFace transformers. When computing loss, we grab pertoken hidden ~~s~~ tate _h_ from last layer, pick (random) indices _s < r < t_ , and compute 1 _−_ cos( _ht − hr, hr − hs_ ). Across all experiments, we follow LLM-JEPA (Huang et al., 2025) to pick 5 random seeds: 82, 23, 37, 84, and 4, and report both mean accuracy and standard deviation. This also allows us to report _p_ -value of paired, single-tailed _t_ -Test. We inherit optimal number of epochs and learning rate from LLM-JEPA. _λ_ is separately tuned. 

## **4.1. Loss Landscape** 

We begin by analyzing the loss landscape by fine-tuning Llama-3.2-1B-Instruct (Grattafiori et al., 2024) on the NLRX-SYNTH (Locascio et al., 2016) dataset. 

_Figure 3._ When the sentence aligns on a geodesic, the concept direction naturally aligns. 

Figure 4(a) demonstrates that in regular fine-tuning, minimizing _L_ NTP does not automatically minimize _L_ STP. With the Semantic Tube, however, _L_ STP continues to decrease even after _L_ NTP plateaus, corroborating (P1). Moreover, 

5 

**Semantic Tube Prediction** 

**==> picture [235 x 383] intentionally omitted <==**

**----- Start of picture text -----**<br>
5 NTP [ when minimize ] NTP<br>NTP [ when minimize ] NTP [+] STP<br>STP [ when minimize ] NTP<br>4 STP [ when minimize ] NTP [+] STP<br>3<br>2<br>1<br>0<br>0 50 100 150 200 250<br>Steps<br>(a)  Loss curve<br>1.4 STP<br>NTP [+] STP<br>1.2 NTP<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0 0.005 0.01 0.02 0.04 0.08<br>(b)  Loss vs. λ<br>Loss<br>Loss<br>**----- End of picture text -----**<br>


_Figure 4._ Loss landscape. **(a)** When _L_ NTP plateaus, _L_ STP continues to decrease. Furthermore, minimizing _L_ NTP does not automatically minimize _L_ STP. **(b)** Across a wide range of _λ_ , increasing _λ_ on a logarithmic scale reduces _L_ STP linearly, while _L_ NTP remains unchanged. 

while _L_ NTP remains comparable between regular and Semantic Tube fine-tuning, there is a significant gap in _L_ STP. This confirms that the SNR gain is driven by _L_ STP, validating the analysis in Section 2.2 that _L_ NTP alone is insufficient for generation quality and that _L_ STP acts as a necessary complement. 

Figure 4(b) illustrates that increasing _λ_ on a logarithmic scale reduces _L_ STP linearly across a wide range, while _L_ NTP remains stable. Given _L_ STP = 1 _−_ cos( _ht − hr, hr − hs_ ), a value of _L_ STP _>_ 1 _._ 0 implies that the trajectory vector _ht − hr_ diverges significantly (essentially reversing direction) relative to _hr − hs_ . At _λ_ = 0 (regular fine-tuning), _L_ STP _≈_ 1 _._ 4 indicates a trajectory resembling erratic Brownian motion. At _λ_ = 0 _._ 08, _L_ STP drops to 0 _._ 6, reflecting a substantially smoother path. Notably, while the optimal 

performance is achieved at _λ_ = 0 _._ 02 (Table 2), the accuracy at _λ_ = 0 _._ 08 is only marginally lower (Figure 7). 

## **4.2. Better Accuracy** 

**On Various Datasets** : We first fine-tune Llama-3.2-1BInstruct to demonstrate that Semantic Tube yields significant accuracy improvements over regular fine-tuning and LLMJEPA across diverse datasets: NL-RX-SYNTH, NL-RXTURK (Locascio et al., 2016), GSM8K (Cobbe et al., 2021), Spider (Yu et al., 2018), NQ-Open (Lee et al., 2019), and HellaSwag (Zellers et al., 2019). Figure 5(a) illustrates the superior performance of Semantic Tube compared to regular fine-tuning and LLM-JEPA. 

**On Various Model Families** : Next, we extend our evaluation to various model families. In addition to Llama, we evaluate gemma-2-2b-it (Team et al., 2024), OpenELM1 ~~1~~ B-Instruct (Mehta et al., 2024), and OLMo-2-0425-1BInstruct (OLMo et al., 2024) on NL-RX-SYNTH, as well as Qwen3-1.7B (Yang et al., 2025) and DeepSeek-R1-DistillQwen-1.5B (DeepSeek-AI et al., 2025) on GSM8K. The results are presented in Figure 5(b). 

**On Various Model Sizes** : Finally, we examine scalability across model sizes using Llama-3 1B, 3B, and 8B models. Results are shown in Figure 5(c). 

## **4.3. Data Efficiency** 

Data efficiency is another crucial metric demonstrating improved SNR. We randomly select subsets of[1] 2[,][1] 4[,][1] 8[,] 161[,] and 321[of][the][NL-RX-SYNTH][dataset][and][perform][both] Semantic Tube and regular fine-tuning on Llama-3 1B, 3B, and 8B models. To compensate for the reduced number of training steps, we scale the epochs proportionally: with a _n_ 1[dataset fraction, we run] _[ n][×]_[ epochs.][For Semantic Tube,] accuracy shows a negligible drop when the training dataset is halved and remains robust until the dataset is reduced to 1 16[, at which point it matches the accuracy of regular fine-] tuning on the full dataset. In contrast, regular fine-tuning suffers a significant drop immediately when the dataset is halved. See Figure 1 for 1B results and Figure 12 for 3B and 8B results. 

We also experimented with half compute ( _[n]_ 2 _[×]_[ epochs) com-] bined with a 2 _×_ learning rate. In both full and half compute scenarios, we also tested 2 _× λ_ . Interestingly, although the half-compute, double-learning-rate setting does not yield optimal accuracy at[1] 2[or full training data, it performs com-] paratively better when the dataset fraction is _<_[1][[.]] 

2[[.]] 

The improved accuracy and data efficiency provide strong evidence that Semantic Tube improves SNR (see Section H for formal proofs linking SNR to accuracy and data efficiency). This validates (P2) and supports the proposed noise/signal decomposition in Figure 1, where the com- 

6 

**Semantic Tube Prediction** 

**==> picture [235 x 576] intentionally omitted <==**

**----- Start of picture text -----**<br>
NTP<br>80<br>NTP [+] JEPA<br>70 NTP [+] STP [ (Ours)]<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>SYNTH TURK GSM8K Spider NQ-Open HellaSwag<br>(a)  Datasets<br>80<br>60<br>40<br>20<br>0<br>Llama3 Gemma2 OpenELM Qwen3 R1-Distill OLMo<br>(b)  Model families<br>100<br>80<br>60<br>40<br>20<br>0<br>Llama3.2 1B Llama3.2 3B Llama3.1 8B<br>(c)  Model sizes<br>Accuracy (%)<br>Accuracy (%)<br>Accuracy (%)<br>**----- End of picture text -----**<br>


_Figure 5._ Semantic Tube ( _L_ NTP + _L_ STP, our approach) demonstrates superior performance across **(a)** datasets, **(b)** model families, and **(c)** model sizes compared to regular fine-tuning ( _L_ NTP) and LLM-JEPA ( _L_ NTP + _L_ JEPA). 

ponent perpendicular to the tube represents noise. Consequently, it supports the hypothesis that the geodesic is locally linear; otherwise, it could not be effectively approximated by the tube. 

## **4.4. Preserving Diversity** 

In this section, we demonstrate that Semantic Tube preserves diversity. In the NL-RX-SYNTH dataset, some regular expressions end with “.*”, while others end with “.*.*”. Although functionally equivalent, these variations represent a nuanced preference by the dataset creator; a robust neural network should be able to learn and preserve this diversity. As shown in Table 1, we find that regular finetuning struggles to learn either pattern effectively. LLMJEPA learns the former pattern well but fails on the latter, likely because the former dominates the training set by a factor of 35 _×_ . In contrast, Semantic Tube successfully learns both patterns. We list representative samples from the SYNTH dataset ending with either “.*” or “.*.*” in Table 3. 

_Table 1._ Accuracy on functionally equivalent regular expression suffixes “.*” and “.*.*”. Semantic Tube effectively captures nuanced preferences, whereas LLM-JEPA exhibits mode collapse by biasing towards “.*”, which is 35 _×_ more prevalent in the training set than “.*.*”. 

|**Suffx**|**Semantic Tube**|**Regular**|**LLM-JEPA**|
|---|---|---|---|
|||||
|.*<br>.*.*|88.5%<br>68.0%|29.9%<br>28.0%|68.9%<br>32.0%|



Following LLM-JEPA, we compute the singular value decomposition (SVD) of Enc(Text) _−_ Enc(Code) to gain insight into the learned representations. Interestingly, we find (Figure 6) that Semantic Tube exhibits polymorphism: when the difference vectors Enc(Text) _−_ Enc(Code) are normalized, the singular value spectrum aligns with LLMJEPA; however, without normalization, it closely resembles regular fine-tuning. This indicates that Semantic Tube enforces structure on the directions (normalized vectors) while tolerating complexity on the raw vectors. We conjecture that this mechanism allows Semantic Tube to maintain flexibility and preserve diversity. 

Collectively, these results validate (P3). 

## **4.5. Tuning** _λ_ 

Semantic Tube introduces a single hyperparameter, _λ_ . Empirically, we observe that the accuracy vs. _λ_ curve is concave (Figure 7), typically peaking between 0 _._ 01 and 0 _._ 08 (Table 2). Notably, this behavior persists across other variations (see Section 4.6): the accuracy curves remain concave, and the optimal _λ_ consistently falls within the 0 _._ 01–0 _._ 08 range (see Figure 13). This validates (P4). 

7 

**Semantic Tube Prediction** 

**==> picture [235 x 383] intentionally omitted <==**

**----- Start of picture text -----**<br>
Base Model<br>NTP<br>10 [3] NTP [+] JEPA [,][ k][ = 1]<br>NTP [+] JEPA [,][ k][ = 0]<br>NTP [+] STP<br>10 [2]<br>10 [1]<br>0 20 40 60 80 100<br>Index<br>(a)  Without Normalization<br>Base Model<br>NTP<br>10 [1] NTP [+] JEPA [,][ k][ = 1]<br>NTP [+] JEPA [,][ k][ = 0]<br>NTP [+] STP<br>10 [0]<br>10 1<br>0 20 40 60 80 100<br>Index<br>(b)  With Normalization<br>Singular Value<br>Singular Value<br>**----- End of picture text -----**<br>


_Figure 6._ SVD decomposition demonstrating Semantic Tube’s polymorphism. **(a)** Without normalization, the SVD profile closely resembles regular fine-tuning. **(b)** With normalization, the SVD aligns with LLM-JEPA. Collectively, this indicates that Semantic Tube enforces a simple structure on the directions (normalized vectors) mapping Text to Code, while tolerating complexity in the unnormalized vectors. Note that the relative relationships among the base model, regular fine-tuning, and LLM-JEPA remain unchanged with or without normalization. 

_Table 2._ Optimal _λ_ values yielding maximum accuracy. 

|SYNTH|TURK|GSM8K|Spider|NQ|HS|
|---|---|---|---|---|---|
|0.02<br>0.04<br>0.005<br>0.04<br>0.16<br>0.02||||||
|Gemma2|Qwen3|R1 Dist|OLMo|OpenELM||
|||||||
|0.005|0.02|0.04|0.01|0.04||
|Llama3 3B|||Llama3 8B|||
|||||||
|0.01|||0.0025|||



**==> picture [235 x 176] intentionally omitted <==**

**----- Start of picture text -----**<br>
90<br>84.29±0.36 SYNTH<br>TURK<br>80<br>GSM8K<br>Spider<br>70 NQ-Open<br>HellaSwag<br>60 56.78±0.68<br>50<br>41.16±0.30<br>40 36.60±0.53<br>36.67±0.44<br>30 26.59±0.65<br>20<br>1/800 1/400 1/200 0.01 0.02 0.04 0.08 0.16 0.32 0.64<br>Accuracy (%)<br>**----- End of picture text -----**<br>


_Figure 7._ Impact of _λ_ tuning on Ll ~~a~~ ma-3 1B across various datasets. In most cases, peak performance is achieved within the range of 0 _._ 01 to 0 _._ 08. 

## **4.6. Ablation** 

We conducted extensive ablation studies on design decisions, establishing that _L_ STP yields superior performance compared to all variations (Figure 8). Full details are provided in Section L. We specifically note that the **Pred** variant—which trains a linear projector _P_ to minimize _L_ STP = 1 _−_ cos( _P_ ( _hr − hs_ ) _, ht − hr_ )—results in degraded performance in all configurations. This validates (P5). 

**==> picture [235 x 177] intentionally omitted <==**

**----- Start of picture text -----**<br>
100<br>Zero Inst Mean Sign<br>90 Pred Warmup Full<br>80<br>70<br>60<br>50<br>Semantic Tube Two View Mask Curvature<br>Accuracy (%)<br>**----- End of picture text -----**<br>


_Figure 8._ Ablation study. Semantic Tube (our approach) outperforms all variations. Within the Semantic Tube family, alternative configurations consistently degrade performance. 

## **5. Conclusion** 

This paper proposes the Geodesic Hypothesis, which posits that token sequence trajectories on the LLM manifold are locally linear geodesics. Based on it, we introduce Semantic Tube Prediction (STP)—a learning objective comple- 

8 

**Semantic Tube Prediction** 

mentary to Next Token Prediction—which compresses hidden state trajectories into a signal-rich tube centered on the geodesic. Our approach generalizes LLM-JEPA by eliminating the need for manual scaffolding of two-view structures, additional compute, or auxiliary predictors. Empirically, STP significantly improves Signal-to-Noise Ratio, allowing models to maintain accuracy even when training data is reduced to[1] 16[, thereby challenging standard Power Law scal-] ing. Our framework unifies the Linear Representation and Manifold Hypotheses under the Principle of Least Action. 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 

## **References** 

- Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. Self-supervised learning from images with a joint-embedding predictive architecture. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 15619– 15629, 2023. 

- Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., and Auli, M. Data2vec: A general framework for self-supervised learning in speech, vision and language. In _International conference on machine learning_ , pp. 1298–1312. PMLR, 2022. 

- Barrault, L., Duquenne, P.-A., Elbayad, M., Kozhevnikov, A., Alastruey, B., Andrews, P., Coria, M., Couairon, G., Costa-jussa, M. R., Dale, D., et al.` Large concept models: Language modeling in a sentence representation space. _arXiv preprint arXiv:2412.08821_ , 2024. 

- Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. Scheduled sampling for sequence prediction with recurrent neural networks. _Advances in neural information processing systems_ , 28, 2015. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Schulman, J., Hilton, J., Knight, M., Weller, A., Amodei, D., et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Coddington, E. A. and Levinson, N. _Theory of Ordinary Differential Equations_ . McGraw-Hill, New York, 1955. 

- Cover, T. M. and Thomas, J. A. _Elements of Information Theory_ . Wiley Series in Telecommunications. WileyInterscience, 1991. ISBN 0-471-06259-6. 

- DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948. 

- Deng, C., Bai, Y., and Ren, H. Chipalign: Instruction alignment in large language models for chip design via geodesic interpolation. In _2025 62nd ACM/IEEE Design Automation Conference (DAC)_ , pp. 1–7. IEEE, 2025. 

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)_ , pp. 4171–4186, 2019. 

- Elman, J. L. Finding structure in time. _Cognitive science_ , 14(2):179–211, 1990. 

- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

9 

**Semantic Tube Prediction** 

- Gunasekar, S., Zhang, Y., Aneja, J., Mendes, C. C. T., Del Giorno, A., Gopi, S., Javaheripi, M., Kauffmann, P., de Rosa, G., Saarikivi, O., et al. Textbooks are all you need. _arXiv preprint arXiv:2306.11644_ , 2023. 

- Henaff,´ O. J., Bai, Y., Charlton, J. A., Nauhaus, I., Simoncelli, E. P., and Goris, R. L. T. Primary visual cortex straightens natural video trajectories. _Nature Communications_ , 12(1):5982, oct 2021. ISSN 20411723. doi: 10.1038/s41467-021-25939-z. URL https: //doi.org/10.1038/s41467-021-25939-z. 

- Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. _arXiv preprint arXiv:2203.15556_ , 2022. 

- Hosseini, E. and Fedorenko, E. Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language. _Advances in Neural Information Processing Systems_ , 36: 43918–43930, 2023. 

- Hron, J., Bahri, Y., Sohl-Dickstein, J., and Novak, R. Infinite attention: Nngp and ntk for deep attention networks. In _International Conference on Machine Learning_ , pp. 4376– 4386. PMLR, 2020. 

- Huang, H., LeCun, Y., and Balestriero, R. Llm-jepa: Large language models meet joint embedding predictive architectures. _arXiv preprint arXiv:2509.14252_ , 2025. 

- Huszar, F.´ How (not) to train your generative model: Scheduled sampling, likelihood, adversary? _arXiv preprint arXiv:1511.05101_ , 2015. 

- Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel: Convergence and generalization in neural networks. _Advances in neural information processing systems_ , 31, 2018. 

- Jing, L., Vincent, P., LeCun, Y., and Tian, Y. Understanding dimensional collapse in contrastive self-supervised learning. _arXiv preprint arXiv:2110.09348_ , 2021. 

- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_ , 2020. 

- Kenneweg, T., Kenneweg, P., and Hammer, B. Jepa for rl: Investigating joint-embedding predictive architectures for reinforcement learning. _arXiv preprint arXiv:2504.16591_ , 2025. 

- Khalil, H. K. _Nonlinear systems_ . Prentice Hall, Upper Saddle River, N.J., 2002. 

- Kiani, B., Wang, J., and Weber, M. Hardness of learning neural networks under the manifold hypothesis. _Advances in Neural Information Processing Systems_ , 37:5661–5696, 2024. 

- Kim, T., Yoo, K. M., and Lee, S.-g. Self-guided contrastive learning for BERT sentence representations. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 2528–2540, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.197. URL https: //aclanthology.org/2021.acl-long.197/. 

- Kong, L., Sun, J., and Zhang, C. Sde-net: Equipping deep neural networks with uncertainty estimates. In _37th International Conference on Machine Learning, ICML 2020_ , pp. 5361–5371. International Machine Learning Society (IMLS), 2020. 

- Lanczos, C. _The Variational Principles of Mechanics_ . Number v. 1 in Mathematical expositions. University of Toronto Press, 1966. 

- LeCun, Y. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. _Open Review_ , 62(1):1–62, 2022. 

- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F., et al. A tutorial on energy-based learning. _Predicting structured data_ , 1(0), 2006. 

- Lee, K., Chang, M.-W., and Toutanova, K. Latent retrieval for weakly supervised open domain question answering. In Korhonen, A., Traum, D., and Marquez,` L. (eds.), _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pp. 6086–6096, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1612. URL https://aclanthology.org/P19-1612/. 

- Littwin, E., Saremi, O., Advani, M., Thilak, V., Nakkiran, P., Huang, C., and Susskind, J. How jepa avoids noisy features: The implicit bias of deep linear self distillation networks. _Advances in Neural Information Processing Systems_ , 37:91300–91336, 2024. 

- Locascio, N., Narasimhan, K., DeLeon, E., Kushman, N., and Barzilay, R. Neural generation of regular expressions from natural language with minimal domain knowledge. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pp. 1918–1923, 2016. 

10 

**Semantic Tube Prediction** 

- Mehta, S., Sekhavat, M. H., Cao, Q., Horton, M., Jin, Y., Sun, C., Mirzadeh, I., Najibi, M., Belenko, D., Zatloukal, P., et al. Openelm: An efficient language model family with open training and inference framework. _arXiv preprint arXiv:2404.14619_ , 2024. 

- Muennighoff, N., Rush, A., Barak, B., Le Scao, T., Tazi, N., Piktus, A., Pyysalo, S., Wolf, T., and Raffel, C. A. Scaling data-constrained language models. _Advances in Neural Information Processing Systems_ , 36:50358–50376, 2023. 

- Okabe, A., Boots, B., Sugihara, K., and Chiu, S. N. _Spatial Tessellations: Concepts and Applications of Voronoi Diagrams_ . Series in Probability and Statistics. John Wiley and Sons, Inc., 2nd ed. edition, 2000. 

- OLMo, T., Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., Bhagia, A., Gu, Y., Huang, S., Jordan, M., et al. 2 olmo 2 furious. _arXiv preprint arXiv:2501.00656_ , 2024. 

- Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models. In _Proceedings of the 41st International Conference on Machine Learning_ , ICML’24. JMLR.org, 2024. 

- Park, K., Choe, Y. J., Jiang, Y., and Veitch, V. The geometry of categorical and hierarchical concepts in large language models. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https: //openreview.net/forum?id=bVTM2QKYuA. 

- Robinson, M., Dey, S., and Chiang, T. Token embeddings violate the manifold hypothesis. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_ , 2025. 

- Seleznova, M. and Kutyniok, G. Neural tangent kernel beyond the infinite-width limit: Effects of depth and initialization. In _International Conference on Machine Learning_ , pp. 19522–19560. PMLR, 2022. 

- Shannon, C. E. A mathematical theory of communication. _The Bell System Technical Journal_ , 27(3):379–423, 1948. doi: 10.1002/j.1538-7305.1948.tb01338.x. 

- Siegleman, H. and Sontag, E. On the computational power of neural networks. _Journal of Computer and System Sciences_ , 50:132–150, 1995. 

- Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., and Morcos, A. Beyond neural scaling laws: beating power law scaling via data pruning. _Advances in Neural Information Processing Systems_ , 35:19523–19536, 2022. 

- Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Rame,´ A., et al. Gemma 2: Improving open 

language models at a practical size. _arXiv preprint arXiv:2408.00118_ , 2024. 

- Tong, A., Nguyen-Tang, T., Lee, D., Nguyen, D., Tran, T., Hall, D., KANG, C., and Choi, J. Neural ode transformers: Analyzing internal dynamics and adaptive finetuning. In _International Conference on Learning Representations (ICLR)_ . International Conference on Learning Representations, 2025. 

- Wang, B. and Sun, H. Is the reversal curse a binding problem? uncovering limitations of transformers from a basic generalization failure. _arXiv preprint arXiv:2504.01928_ , 2025. 

- Whiteley, N., Gray, A., and Rubin-Delanchy, P. Statistical exploration of the manifold hypothesis. _Journal of the Royal Statistical Society: Series B_ , 2025. 

- Williams, R. J. and Zipser, D. A learning algorithm for continually running fully recurrent neural networks. _Neural computation_ , 1(2):270–280, 1989. 

- Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388. 

- Yang, G. and Hu, E. J. Tensor programs iv: Feature learning in infinite-width neural networks. In _International Conference on Machine Learning_ , pp. 11727–11737. PMLR, 2021. 

- Yang, G. and Littwin, E. Tensor programs iib: Architectural universality of neural tangent kernel training dynamics. In _International conference on machine learning_ , pp. 11762– 11772. PMLR, 2021. 

- Yu, H., Inal, B., and Fumero, M. Connecting neural models latent geometries with relative geodesic representations. In _NeurIPS 2024 Workshop on Symmetry and Geometry in Neural Representations_ , 2024. 

- Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., et al. Spider: A large-scale human-labeled dataset for complex and crossdomain semantic parsing and text-to-sql task. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , 2018. 

11 

**Semantic Tube Prediction** 

- Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , 2019. 

- Zhang, B., Liu, Z., Cherry, C., and Firat, O. When scaling meets llm finetuning: The effect of data, model and finetuning method. In _The Twelfth International Conference on Learning Representations_ , 2024. 

12 

**Semantic Tube Prediction** 

## **A. Training ODE** 

In this section, we present a form of _u_ ( _·_ ) and _f_ ( _·_ ) such that _x≤t_ +1 _⊖ x≤t_ = _x≤t_ +1 _− x≤t_ . Throughout the section, we slightly abuse notation by letting _xt_ denote both a token and its embedding vector _xt ∈_ R _[d]_[model] , and letting _x≤t_ denote both a token sequence and its embedding vector _x≤t ∈_ R _[T][ ×][d]_[model] : 

**==> picture [107 x 11] intentionally omitted <==**

Let _f_ ( _x≤t_ ) _∈_ R _[d]_ . Let _u_ ( _·_ ) : R _[d] →_ R _[d]_[model] be the unembedding function that maps the hidden state back to the token embedding. 

Note that we need a function to lift _u_ ( _f_ ( _x≤t_ )) from R _[d]_[model] to R _[T][ ×][d]_[model] . Define _v_ ( _·, ·_ ) : R _[d]_[model] _×_ N _→_ R _[T][ ×][d]_[model] such that 

**==> picture [144 x 22] intentionally omitted <==**

Hence, we have 

**==> picture [103 x 11] intentionally omitted <==**

Define the _⊖_ operator as 

**==> picture [107 x 11] intentionally omitted <==**

By the definition of _v_ ( _·, ·_ ), we have 

**==> picture [120 x 10] intentionally omitted <==**

Note that the network is now in the form _v_ ( _u_ ( _f_ ( _x≤t_ )) _, t_ ), which can be written as _g_ ( _x≤t, t_ ) and satisfies the formulation of an ODE. 

## **B. Inference SDE** 

At training time, the unembedding error _ϵt_ does not propagate to the next token. However, at inference time, _ht_ +1 depends (indirectly) on _ht_ , causing _ϵt_ to accumulate into a Brownian motion term. 

Yang & Littwin (2021) established that in the limit of infinite width, the pre-activations of a neural network (and thus the hidden state) are well-approximated by Gaussian processes. Hence, we can assume _ϵt_ are i.i.d. Gaussian. Furthermore, as shown by (Yang & Littwin, 2021), _ϵt_ remains i.i.d. Gaussian when passed through a randomly initialized neural network, which remains constant in the infinite-width limit. Consequently, _ϵt_ accumulates to form a Brownian motion term _dWt_ . Thus the inference process can be modeled by a Stochastic Differential Equation (SDE). 

**Proposition B.1** (Inference SDE) **.** _The inference process of an LLM can be modeled by an SDE in the token sequence space_ R _[T][ ×][d]_[model] _,_ 

**==> picture [127 x 13] intentionally omitted <==**

Consider the example in Figure 2, if the Brownian motion shifts the top trajectory to the bottom, mode collapse occurs. Conversely, if the bottom trajectory shifts to the top, mode collapse occurs. This motivates the construction of an approach to explicitly suppress _ϵt_ . Indeed, Section 4.1 demonstrates that next token prediction alone is insufficient for high-quality generation, making our approach a necessary complement. 

## **C. Context-Aware Hidden State** 

We can view _ht − hs_ as the semantic evolution induced by the sub-sequence _x_ [ _s,t_ ] given the context _x≤s_ . In this sense, _ht − hs_ acts as a context-aware hidden state transition, which is significantly more informative than the static hidden state of the isolated sub-sequence _x_ [ _s,t_ ]. 

For example, given the prefix _⃗v_ The _,⃗v_ capital _,⃗v_ of , appending the token _⃗v_ France shifts the overall semantic trajectory toward _⃗vP aris_ . However, given a different prefix _⃗v_ The _,⃗v_ language _,⃗v_ of , appending the same token _⃗vF rance_ shifts the trajectory toward _⃗vF rench_ . If we were to compute the hidden state of _⃗vF rance_ in isolation, we would lose this contextual nuance and fail to capture the context-specific semantic semantic shift. 

Thus, _ht − hs_ serves as a context-aware representation of the added information. 

13 

**Semantic Tube Prediction** 

_Figure 9._ The same token _⃗vF rance_ directs the geodesic along different concept directions when appended to distinct prefixes, illustrating the necessity of the context-aware state difference _ht − hs_ . 

## **D. Proof of the Straightening Lemma** 

In this section, we provide the proof for Theorem 3.2. The objective is to show 

**==> picture [167 x 14] intentionally omitted <==**

_Figure 10._ Geometric illustration for the proof of Theorem 3.2 

Referring to Figure 10, we have 

**==> picture [172 x 12] intentionally omitted <==**

Since _θ[′] ≥ θ_ , it follows that 

**==> picture [174 x 13] intentionally omitted <==**

We also have 

**==> picture [95 x 11] intentionally omitted <==**

When _ϵ_ is sufficiently small, we can approximate cos _θ[′] ≈_ 1 _−[θ]_ 2 _[′]_[2][.][Hence] 

**==> picture [32 x 23] intentionally omitted <==**

Rearranging gives 

**==> picture [39 x 12] intentionally omitted <==**

Also, when _θ[′]_ is sufficiently small, sin _θ[′] ≈ θ[′]_ . Therefore 

**==> picture [293 x 14] intentionally omitted <==**

**==> picture [11 x 7] intentionally omitted <==**

**Semantic Tube Prediction** 

## **E. Proof of the Semantic Tube Theorem** 

We introduce two auxiliary tokens, <before-bos> and <after-eos>. The token <before-bos> appears only at the 0-th position and always precedes <bos>, while <after-eos> appears only at the _τ_ + 1-th position and always follows <eos>. This augmentation increases the total sequence length from _τ_ to _τ_ + 2. By anchoring the sequence with <before-bos> and <after-eos>, we ensure that the boundary conditions _h_ 0 = _h[∗]_ 0[and] _[ h][τ]_[+1][=] _[ h][∗] τ_ +1[are satisfied.] 

The proof follows from these conditions. □ 

## **F. Inference Cone** 

As STP explicitly reduces _ϵt_ , it lowers _σt_ in the Brownian Motion term of Theorem B.1. At inference time, the Brownian motion term causes the token sequence trajectory diverge into a cone whose radius grows at a rate _∝ σt√t_ . A lower _σt_ reduces the probability that the cone collides with another token sequence, which would causes mode collapse (Figure 11). 

_Figure 11._ The inference cone defines the probabilistic range of a Brownian motion, and its radius grows _∝ σt_ ~~_√_~~ _t_ . A larger _σt_ leads to a wider cone, which has a high probability of colliding with a token sequence trace that is far away (blue cone and green geodesic), while a smaller _σt_ leads to a narrower cone that may only collide with a nearby trace (yellow cone and red geodesic). The dotted red and green fine lines are the Brownian motions confined by the yellow and blue cones, respectively. 

**Proposition F.1** (Inference Cone) **.** _The distortion between ht and h[∗] t[behaves as a Gaussian process, where the scale of the] deviation grows as ∥ht − h[∗] t[∥]_[2] _[∝][σ] √t_ 

_Proof._ According to Theorem B.1, at inference time, we model the token sequence trajectory as following an SDE _dx≤t_ = ˚ _u ◦ f_[˚] ( _x≤t_ ) _dt_ + _σtdWt_ , where _σtdWt_ is a Brownian motion. Let _ht_ = _f_[˚] ( _x≤t_ ) be the hidden state. Let _x[∗] ≤t_[be the] error-free generation satisfying _dx[∗] ≤t_[= ˚] _[u][ ◦] f_[˚] ( _x[∗] ≤t_[)] _[dt]_[, and let] _[ h] t[∗]_[=] _f_[˚] ( _x[∗] ≤t_[)][ be the error-free hidden state.][We can quantify] the distortion between _ht_ and _h[∗] t_[by examining how the Brownian motion is transformed by] _f_[˚] . 

Yang & Littwin (2021) establishes that in the infinite-width limit, _f_[˚] converges to a Neural Tangent Kernel (NTK) determined by random initialization. It further showed that Gaussian noise remains Gaussian when passed through a randomly initialized network. Hence, a Brownian motion remains a Brownian motion when passed through _f_[˚] . Therefore, 

**==> picture [72 x 22] intentionally omitted <==**

where _ϵs_ are Gaussian noises. By Donsker’s theorem, when _t →∞_ , ~~_√_ —~~ 1 _t_ 2 _s≤t[ϵ][s][∼][N]_[(0] _[,]_[ Σ)][.][Consequently, the magnitude] of the distortion scales as 

**==> picture [358 x 43] intentionally omitted <==**

Theorem F.1 implies that with high probability, the trajectory of the generated hidden state _h_ is confined within a cone centered at _h[∗]_ whose radius grows at a rate _∝ σ√t_ . 

When mode collapse occurs at inference time, i.e., a generated sequence _x≤t_ collides with _y≤t[′]_ , then their corresponding hidden states _h_ and _g_ must collide. Let _∥h[∗] − g[∗] ∥_ 2 be the minimum distance between _h[∗]_ and _g[∗]_ . By Theorem F.1, _∀ε >_ 0, 

15 

**Semantic Tube Prediction** 

_∃c_ , 

**==> picture [123 x 13] intentionally omitted <==**

On the other hand, _L_ STP suppresses _ϵt_ and consequently reduces _σ_ , which decreases the lower bound of the probability of mode collapse. 

## **G. Implementation Details** 

If the training data already possesses a two-view structure, such as a ( _query, answer_ ) pair, one can leverage it by anchoring _s_ at the beginning of the _query_ and _t_ at the end of the _answer_ . However, we suggest that _r_ should be randomly selected to maximize the benefit of the STP loss. As demonstrated in our ablation study, fixing _r_ at the end of the _query_ yields lower accuracy. 

Typically, _ht − hs_ does not equal the hidden state of the isolated sub-sequence _x_ [ _s,t_ ]. However, as discussed Section C, we can view _ht − hs_ as the semantic evolution induced by the sub-sequence _x_ [ _s,t_ ] given the context _x≤s_ . In this sense, _ht − hs_ acts as a context-aware hidden state, which is significantly more informative than the hidden state of _x_ [ _s,t_ ] computed in isolation. For example, given the prefix _⃗v_ The _,⃗v_ capital _,⃗v_ of , appending the token _⃗v_ France shifts the overall meaning to _⃗v_ Paris. Conversely, given the prefix _⃗v_ The _,⃗v_ language _,⃗v_ of , appending _⃗v_ France shifts the meaning to _⃗v_ French. Computing the hidden state of _⃗v_ France separately loses this context and fails to capture the context-specific meaning of the tokens (see Figure 9). 

We can also leverage _ht − hs_ to bypass unwanted tokens. For example, setting _s >_ 0 allows us to skip the system prompt. Similarly, in multiple-choice Q&A, distractor choices that are semantically inconsistent with the _query_ are often located between the _query_ and the correct _answer_ . In such cases, we can pick _r_ and _r[′]_ such that _x_ [ _s,r_ ] is the _query_ and _x_ [ _r′,t_ ] is the correct _answer_ , computing the STP loss as: 

**==> picture [147 x 11] intentionally omitted <==**

This formulation effectively skips the irrelevant choice branches in the middle. 

Finally, the STP loss assumes that _hs_ , _hr_ , and _ht_ are collinear, which may not hold strictly in reality as geodesics can exhibit curvature. In practice, this implies that we must select a small _λ_ to tolerate the angular deviation between _ht − hr_ and _hr − hs_ . Indeed, our experiments consistently show that _λ ≈_ 0 _._ 01 is effective across various models, datasets, and model sizes. 

## **H. Signal-to-Noise Ratio** 

Directly measuring Signal-to-Noise Ratio (SNR) in the latent representations of LLMs is intractable. In self-supervised learning, the decomposition of activations into “semantic signal” and “nuisance noise” is not explicitly observable without access to the ground-truth data manifold. 

In this subsection, we formally show an information theoretic link between SNR and data efficiency and training accuracy. Hence we can validate our hypothesis via the predicted impact on them. 

We model LLM training process as extracting information about a discrete target _Y_ (tokens) from continuous latent representations _X_ (hidden states). Let _Y ∈V_ be the discrete target token from a vocabulary of size _|V|_ . Let _X[m]_ = _{Xi,_ 1 _≤ i ≤ m}_ be a set of _m_ hidden states that are conditionally i.i.d. given _Y_ . The training objective is to minimize cross-entropy, which is asymptotically equivalent to minimizing the conditional entropy _H_ ( _Y |X[m]_ ). 

**Lemma H.1** (Data Efficiency) **.** 

**==> picture [317 x 11] intentionally omitted <==**

_Proof._ The goal is to show: 

**==> picture [139 x 11] intentionally omitted <==**

By the definition of Mutual Information: 

**==> picture [137 x 11] intentionally omitted <==**

16 

**Semantic Tube Prediction** 

We need to bound _I_ ( _Y_ ; _X[m]_ ). Apply chain rule of mutual information, 

**==> picture [146 x 11] intentionally omitted <==**

Since _Xi_ are conditionally independent given _Y_ : 

**==> picture [113 x 29] intentionally omitted <==**

For the first term _H_ ( _X[m]_ ), by sub-additivity of entropy, the entropy of the joint distribution is always less than or equal to the sum of individual entropies (independence maximizes entropy): 

**==> picture [93 x 28] intentionally omitted <==**

Substitute these back into the Mutual Information expansion: 

**==> picture [168 x 102] intentionally omitted <==**

Since _Xi_ are identically distributed, _I_ ( _Y_ ; _Xi_ ) is the same for all _i_ : 

**==> picture [107 x 11] intentionally omitted <==**

Finally substitute this upper bound on Information back into step 1. Since we are subtracting a larger value, the result is a lower bound on entropy: 

**==> picture [238 x 11] intentionally omitted <==**

Suppose _H_ ( _Y |X[m]_ ) _≤ ϵ_ after training, we have 

**==> picture [163 x 11] intentionally omitted <==**

Recent theoretical work on infinite-width limits (Yang & Littwin, 2021) establishes that layer pre-activations converge to Gaussian distributions. Motivated by this, we model the local representation dynamics using a canonical Gaussian Channel approximation with additive noise. Specifically, we decompose _X_ = _Z_ + _N_ , where _Z_ is the latent signal, and _N ∼N_ (0 _, σ_[2] _I_ ) is the additive Gaussian noise. We define the Signal-to-Noise Ratio as 

**==> picture [72 x 25] intentionally omitted <==**

Under the Gaussian channel approximation, mutual information is a logarithmic function of SNR (Shannon, 1948): 

17 

**Semantic Tube Prediction** 

**==> picture [115 x 21] intentionally omitted <==**

Substituting this capacity into Theorem H.1, we have 

**Corollary H.2** (Signal-to-Noise Ratio) **.** 

**==> picture [290 x 26] intentionally omitted <==**

Theorem H.2 indicates that _m_ is inversely proportional to log(1 + SNR). Consequently, if the Semantic Tube works as expected, it will increase SNR and strictly lower the data requirement _m_ . 

Let _Y_[ˆ] = _f_ ( _X[m]_ ) be the estimator of _Y_ produced by the model. Let _Pe_ = _P_ ( _Y_[ˆ] = _Y_ ) be the probability of error (incorrect token generation). Fano’s Inequality (Cover & Thomas, 1991) provides a lower bound on the conditional entropy _H_ ( _Y |X[m]_ ) in terms of the error probability: 

**==> picture [164 x 11] intentionally omitted <==**

where _Hb_ ( _Pe_ ) is the binary entropy function. For LLMs, _|V| ≫_ 1, the term _Pe_ log _|V|_ dominates _Hb_ ( _Pe_ ). Hence we can simplify Fano’s inequality to be: 

**==> picture [305 x 11] intentionally omitted <==**

Plug Equation (5) into Equation (7), immediate we get 

**Corollary H.3** (Accuracy) **.** 

**==> picture [318 x 26] intentionally omitted <==**

Theorem H.3 indicates that if we observe significant improvement on training accuracy, we know that SNR is higher. 

## **I. Data Efficiency** 

**==> picture [487 x 24] intentionally omitted <==**

## **J. Regular Expression Samples** 

We list in Table 3 a few samples from the SYNTH dataset that end with either “.*” or “.*.*”, which are functionally equivalent. 

_Table 3._ Regular expression samples from the SYNTH dataset that end with either “.*” or “.*.*”, which are functionally equivalent. 

**Regular Expressions** .*([a-z]) _|_ ([AEIOUaeiou]) _|_ ([A-Za-z]).* .*([A-Za-z]).*([0-9]).*.* ((dog)(.*)).*([AEIOUaeiou]).* (dog).*((truck) _|_ ([A-Z]) _|_ ([0-9])).* .*(.)&([0-9])&(dog).* .*(dog).*((.)*).*.* .*dog.*[a-z].*.* 

18 

**==> picture [99 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
Semantic Tube Prediction<br>**----- End of picture text -----**<br>


**==> picture [488 x 213] intentionally omitted <==**

**----- Start of picture text -----**<br>
80 80<br>60 60<br>40 40<br>20 20<br>NTP [+] STP 1Flop,2 1Flop<br>0 NTPNTP [+] STP 1Flop,0.50.5Flop,2lr 1Flop0.5Flop,2lr 0 NTP 0.5Flop, 2 lr 0.5Flop, 2 lr<br>1Flop<br>1Flop 0.5Flop,2lr,0.5 0.5Flop, 2 lr,2<br>1/32 1/16 1/8 1/4 1/2 1/32 1/16 1/8 1/4 1/2<br>Data Size Data Size<br>(a)  Llama3 3B (b)  Llama3 8B<br>Figure 12. Semantic Tube (our approach) and regular fine-tuning with [1] 2 [,] [1] 4 [,] [1] 8 [,] 161 [, and] 321 [dataset on (a) Llama3 3B and (b) Llama3 8B.]<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


## **K. Tuning** _λ_ 

In this section, we present the accuracy vs. _λ_ curves for the various configurations of Semantic Tubes, Two Views, and Mask detailed in Section 4.6. As shown in Figure 13, we observe across all cases that the curve is concave, most of the time with a maximum reached at _λ_ values between 0 _._ 01 and 0 _._ 8. Furthermore, when _λ_ exceeds the optimal value, we occasionally observe a precipitous drop in accuracy accompanied by a drastic increase in standard deviation. Collectively, these results provide strong evidence supporting the validity of (P4). 

## **L. Ablation** 

**Semantic Tube** : We ablate several variations of the Semantic Tube configuration: 

- **Zero** : Instead of randomly picking _s_ , this variation fixes the start index _s_ = 0. The loss becomes _L_ STP = 1 _−_ cos( _hr − h_ 0 _, ht − hr_ ). 

- **Pred** : We introduce a learnable linear projector _P_ and modify the loss to _L_ STP = 1 _−_ cos( _P_ ( _hr − hs_ ) _, ht − hs_ ). aligns the approach more closely with the JEPA style, utlizing a non-identity predictor. _P_ is randomly initialized and trained during fine-tuning. 

- **Inst** : We incorporate instructions into the token sequence _x≤t_ . These instructions consist of system prompt such as "Convert natural language to regular expression". 

**Two Views** : This configuration adopts the LLM-JEPA style two-view structure, where _query_ and _answer_ represent two views of the same concept. Note that we retain the _L_ STP formulation but fix _s_ = 0 and set _r_ to the index of the last token of the _query_ . 

- **Warmup** : We linearly warm up _λ_ throughout the training process. 

- **Pred** : Identical to the Pred variation in the Semantic Tube configuration. 

- **Mean** : Instead of the difference vector _hr − hs_ , we use the average embedding _r−_ 1 _s_ +1 � _s≤i≤r[h][i]_[.][Consequently, the] loss becomes _L_ STP = 1 _−_ cos( _r−_ 1 _s_ +1 � _s≤i≤r[h][i][,] t−r_ 1+1 � _r≤j≤t[h][j]_[)][.][This is inspired by BERT Mean Pooling (][Kim] et al., 2021). 

**Mask** : This variation is inspired by BERT mask-and-recover training objective (Devlin et al., 2019). Given a token sequence _x≤t_ , we randomly pick a span [ _s, r_ ] and replace the tokens within this span with the [MASK] token. Let _y≤t_ denote the 

19 

**Semantic Tube Prediction** 

**==> picture [479 x 389] intentionally omitted <==**

**----- Start of picture text -----**<br>
90 81.39±0.65 8 1.58± 0 .89<br>80.5 4 ±0.49<br>80<br>85 83.80±0.62 8 4.29±0.36<br>8 1.93± 1 .05 7 6.07±0.73<br>80 75<br>76.11±1.66<br>75 70<br>70 65<br>65 Tube 60 Two Views<br>Tube Zero Two Views Pred<br>60 Tube Pred Two Views Mean<br>55<br>Tube Inst Two Views Warmup<br>55<br>1/800 1/400 1/200 0.01 0.02 0.04 0.08 0.16 0.32 1/800 1/400 1/200 0.01 0.02 0.04 0.08 0.16 0.32<br>(a)  Semantic Tube (b)  Two Views<br>Mask Inst Recov Pred<br>Mask Inst Full<br>75 74.14±3.45 Mask Inst<br>Mask<br>71.98 ± 1.60<br>70 69.42 ± 0.97<br>66.10 ± 2.70<br>65<br>60<br>1/800 1/400 1/200 0.01 0.02 0.04 0.08 0.16 0.32 0.64<br>(c)  Mask<br>Accuracy (%) Accuracy (%)<br>Accuracy (%)<br>**----- End of picture text -----**<br>


_Figure 13._ Tuning _λ_ for various configurations of (a) Semantic Tube, (b) Two Views, and (c) Mask. In all cases, the accuracy vs. _λ_ curve is concave. We also observe that when _λ_ exceeds the optimal value, accuracy declines rapidly while the standard deviation increases sharply, indicating that _λ ≪_ 1 is preferred. 

masked sequence and _gt_ = _f_ ( _y≤t_ ). The loss is defined as _L_ mask = 1 _−_ cos( _hr − hs, gt_ ). This can be interpreted as recovering the information of the masked tokens using the representation of the masked sequence _y≤t_ . 

- **Full** : Instead of aiming to match _hr − hs_ , we target _ht_ . The loss becomes _L_ Mask = 1 _−_ cos( _ht, gt_ ), corresponding to the recovery of the full masked sequence rather than just the masked span. 

- **Pred** : Identical to the Pred variation in the Semantic Tube configuration. 

- **Inst** : Identical to the Inst variation in the Semantic Tube configuration. 

**Curvature** : This variation is inspired by the curvature straightening objective (Henaff et al.´ , 2021). Let _θi_ be the angle between _hi − hi−_ 1 and _hi_ +1 _− hi_ . The loss is defined as _L_ Curvature =[1] _t_ � _i≤t[|][θ][i][|]_[.] 

- **Sign** : Replaces _|θi|_ with _θi_ (allowing for signed curvature). 

The fact that Pred yields inferior performance in both the Semantic Tube and Two Views configurations supports (P5). The _p_ -values comparing variations and options are presented in Tables 4 and 5. 

20 

**Semantic Tube Prediction** 

_Table 4._ Pairwise _p_ -values comparing variation families. A cell is populated only if the mean accuracy of the row method exceeds that of the column method. _p_ -values are computed using a paired, one-tailed _t_ -test, restricted to the best-performing variant from each family. 

||**Two View**<br>**Mask**<br>**Curvature**|
|---|---|
|||
|**LLM-JEPA2**<br>**Two View**<br>**Mask**|1.14e-3<br>1.77e-3<br>3.04e-5<br>4.76e-3<br>5.10e-5<br>1.28e-4|



_Table 5._ Pairwise _p_ -values comparing options within each variation family. A cell is populated only if the mean accuracy of the row option exceeds that of the column option. _p_ -values are computed using a paired, one-tailed _t_ -test. Values exceeding 0 _._ 05 are struck through. 

||**Zero**<br>**Pred**<br>**Inst**||**2View**<br>**Pred**<br>**Mean**|
|---|---|---|---|
|||||
|**LLM-JEPA2**<br><br>0.0534<br>2.03e-3<br>2.34e-4<br>**2View+Warmup**<br><br>0.265<br>0.0426<br>9.16e-6<br>**+Zero**<br>0.0185<br>5.56e-4<br>**2View**<br><br>0.0689<br>1.19e-6<br>**+Pred**<br>2.97e-4<br>**+Pred**<br>2.78e-4||||
|||||
||**Inst,Recov**<br>**Inst**<br>**Mask**||**Signed**|
|||||
|**Mask**+**_all_**<br>**-Pred**<br>**-Recov,Pred**|<br>0.0629<br>0.0159<br>4.02e-3<br>2.34e-3<br>1.18e-3<br>0.0103|**Curvature**|0.0368|



21 

