Title: Latent Reasoning via Sentence Embedding Prediction

URL Source: https://arxiv.org/html/2505.22202

Markdown Content:
${}^{*}$${}^{*}$footnotetext: Equal contribution
Hyeonbin Hwang 1∗ Byeongguk Jeon 1∗ Seungone Kim 2 Jiyeon Kim 1 Hoyeon Chang 1 Sohee Yang 3 Seungpil Won 4 Dohaeng Lee 4 Youbin Ahn 4 Minjoon Seo 1

1 KAIST 2 Carnegie Mellon University 3 University College London 4 LG AI Research 

{hbin0701, byeongguk, minjoon}@kaist.ac.kr

###### Abstract

Autoregressive Language Models (LMs) generate one token at a time, yet human reasoning operates over higher-level abstractions—sentences, propositions, and concepts. This contrast raises a central question: can LMs likewise learn to reason over structured semantic units rather than raw token sequences? In this work, we investigate whether pretrained LMs can be lifted into such abstract reasoning spaces building on their learned representations. We present a framework that adapts a pretrained token-level LM to operate in _sentence space_, by autoregressively predicting continuous embeddings of next sentences. We explore two embedding paradigms inspired by classical representation learning: semantic embeddings, learned via autoencoding to preserve surface meaning; and (ii) contextual embeddings, trained via next-sentence prediction to encode anticipatory structure. We evaluate both under two inference regimes: Discretized, which decodes each predicted embedding into text before re-encoding; and Continuous, which reasons entirely in embedding space for improved efficiency. Across four domains—mathematics, logic, commonsense, and planning—contextual embeddings under continuous inference show competitive performance with Chain-of-Thought (CoT) while reducing inference-time FLOPs in average by half. We also present early signs of scalability and modular adaptation. Finally, to visualize latent trajectories, we introduce SentenceLens, a diagnostic tool that decodes intermediate model states into interpretable sentences. Together, our results indicate that pretrained LMs can effectively transition to abstract, structured reasoning within latent embedding spaces.∗∗∗Our code is available [here](https://github.com/hbin0701/Pred-Sent).

1 Introduction
--------------

Autoregressive Language Models (LMs) have achieved remarkable success on complex reasoning tasks through a simple objective: Next-Token Prediction[[1](https://arxiv.org/html/2505.22202v2#bib.bib1)]. This success is further amplified by Chain-of-Thought (CoT), which generates explicit intermediate reasoning steps to guide the model[[2](https://arxiv.org/html/2505.22202v2#bib.bib2)]. Recent advancements demonstrate substantial gains in performance by scaling inference-time computation even further[[3](https://arxiv.org/html/2505.22202v2#bib.bib3), [4](https://arxiv.org/html/2505.22202v2#bib.bib4)]. However, next-token prediction requires generating long reasoning chains one token at a time, making it computationally inefficient. Also, it remains unanswered whether reasoning at such granularity is genuinely optimal.

![Image 1: Refer to caption](https://arxiv.org/html/2505.22202v2/x1.png)

Figure 1: Sentence-level reasoning framework. Training: the latent model reads the question tokens and previous embeddings, predicts h^t\hat{h}_{t}, and a frozen decoder reconstructs s t s_{t}; Inference: embedding can be rolled forward by (a) Discretized: decode →\rightarrow text →\rightarrow encode or (b) Continuous: pass-through.

While token-level generation has driven recent progress, human cognition typically operates over higher-level abstractions—such as concepts, propositions, or full sentences[[5](https://arxiv.org/html/2505.22202v2#bib.bib5), [6](https://arxiv.org/html/2505.22202v2#bib.bib6), [7](https://arxiv.org/html/2505.22202v2#bib.bib7)]. Prior works suggest that language models may similarly benefit from operating at these higher levels, potentially enabling more structured and computationally efficient reasoning[[8](https://arxiv.org/html/2505.22202v2#bib.bib8), [9](https://arxiv.org/html/2505.22202v2#bib.bib9)].

In this paper, we investigate whether pretrained language models can effectively build higher-level representations directly by abstracting over their existing token-level representations, without the prohibitive cost of pre-training from scratch. Specifically, we introduce a framework that repurposes pretrained next-token Transformers to reason in a latent sentence-level embedding space. Instead of producing outputs token-by-token, our approach predicts continuous embeddings for entire sentences, which can be decoded back into natural language yet primarily function as abstract conceptual representations.

To systematically explore viable latent representations, we draw inspiration from the well-established dichotomy in classical representation learning between reconstruction-based and prediction-based methods[[10](https://arxiv.org/html/2505.22202v2#bib.bib10), [11](https://arxiv.org/html/2505.22202v2#bib.bib11), [12](https://arxiv.org/html/2505.22202v2#bib.bib12)]. We define two embedding paradigms: (1) Semantic embeddings, which prioritize preserving textual fidelity through autoencoding, and (2) Contextual embeddings, which focus on capturing predictive context via next-sentence prediction.

We evaluate models trained with these embeddings under two inference regimes: Discretized, which decodes each predicted embedding into natural language before re-encoding it as the next input, and Continuous, which performs reasoning entirely within the continuous embedding space. Our empirical findings demonstrate that contextual embeddings consistently outperform semantic embeddings across diverse reasoning domains including mathematics, logic, commonsense, and planning tasks. Notably, contextual embeddings using Continuous inference show competitive performance to token level Chain of Thought reasoning while reducing inference time computational cost by half in average.

Finally, we introduce SentenceLens, a diagnostic tool that translates intermediate hidden states into readable sentences, thus providing intuitive transparency into the model’s internal reasoning trajectories. Overall, our analysis provides initial evidence that pretrained inductive biases acquired from token level modeling can be effectively adapted to structured, abstraction level reasoning within latent embedding spaces.

![Image 2: Refer to caption](https://arxiv.org/html/2505.22202v2/x2.png)

Figure 2: Illustration of the different types of sentence embeddings used in our framework.

2 Sentence embeddings for autoregressive modeling
-------------------------------------------------

Unsupervised and semi-supervised sequence representation learning has predominantly evolved along two primary paradigms: _reconstruction-based_ and _prediction-based_ methods[[10](https://arxiv.org/html/2505.22202v2#bib.bib10), [11](https://arxiv.org/html/2505.22202v2#bib.bib11), [12](https://arxiv.org/html/2505.22202v2#bib.bib12)]. Both methodologies have demonstrated strong empirical performance, yet each emphasizes distinct representational strengths. Reconstruction-based methods, typically employing autoencoder architectures, excel at semantic fidelity by explicitly encoding and reconstructing input sequences[[10](https://arxiv.org/html/2505.22202v2#bib.bib10)], whereas prediction-based methods prioritize capturing contextual semantics by modeling relations to subsequent sequences[[11](https://arxiv.org/html/2505.22202v2#bib.bib11)].

Previous research suggests that the optimal embedding strategy varies significantly depending on the target application[[13](https://arxiv.org/html/2505.22202v2#bib.bib13)]. In this light, we systematically explore both embedding paradigms within the context of sentence-level autoregressive modeling. Specifically, we adapt an autoregressive Language Model autoencoder framework to construct and evaluate two distinct embedding approaches: semantic embedding, derived through reconstruction objective, and contextual embedding, derived through predictive objective.

### 2.1 Sentence embedding construction

To ensure scalability and avoid vocabulary constraints inherent to discrete codebooks[[14](https://arxiv.org/html/2505.22202v2#bib.bib14)], we utilize a continuous embedding space. This approach facilitates flexible representational capacity scaling with embedding dimensionality[[15](https://arxiv.org/html/2505.22202v2#bib.bib15)]. We build upon the autoencoding framework proposed by ICAE[[16](https://arxiv.org/html/2505.22202v2#bib.bib16)] and adapt a decoder-only Transformer (e.g., GPT-2), employing shared parameters for encoding and decoding: θ ENC=θ DEC\theta_{\text{ENC}}=\theta_{\text{DEC}}.

Given an input sequence x=(x 1,…,x N)x=(x_{1},\dots,x_{N}), the encoder produces a sequence of hidden states H=(h 1,…,h N)H=(h_{1},\dots,h_{N}). We then define the embedding h[−1]:=h N h^{[-1]}:=h_{N} as the latent representation of the entire input sequence. This embedding conditions the decoder, trained autoregressively with cross-entropy loss:

y^=θ DEC​(h[−1])and ℒ CE=−∑t=1 N log⁡p​(y t∣y<t,h[−1])\hat{y}=\theta_{\text{DEC}}(h^{[-1]})\quad\text{and}\quad\mathcal{L}_{\text{CE}}=-\sum_{t=1}^{N}\log p(y_{t}\mid y_{<t},h^{[-1]})

Note that most reasoning tasks consist of a question or instruction q q, followed by an ordered sequence of reasoning steps (s 1,…,s n)(s_{1},\dots,s_{n}). In this light, we construct training examples tailored to each embedding type as follows (See Figure[2](https://arxiv.org/html/2505.22202v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Latent Reasoning via Sentence Embedding Prediction")):

#### Semantic embeddings.

Each reasoning step s i s_{i} independently forms the input and reconstruction target x=y=s i x=y=s_{i}. Training this way ensures the embedding h[−1]h^{[-1]} encapsulates complete and detailed semantics of the individual reasoning step.

#### Contextual embeddings.

We form context–target pairs, where context x x includes the question and preceding reasoning steps (q,s 1,…,s i−1)(q,s_{1},\dots,s_{i-1}), and the target is the current step y=s i y=s_{i}. Thus, embeddings must capture predictive cues essential for reasoning step generation.

Table 1: Performance of Semantic and Contextual Embeddings across datasets. For Semantic embeddings, we report exact match (EM). For Contextual embeddings, we compare final-answer accuracy (ACC) under different decoding schemes: CTX-B (unregularized), CTX-C (contrastive), and CoT (language-level chain-of-thought).

Optionally, to bridge semantic fidelity with predictive abstraction, we also try a contrastive regularization loss (InfoNCE), aligning contextual embeddings closer to corresponding semantic embeddings:

ℒ InfoNCE=−log⁡exp⁡(sim⁡(z^i,z i sem)/τ)∑j exp⁡(sim⁡(z^i,z j sem)/τ),\mathcal{L}_{\text{InfoNCE}}=-\log\frac{\exp\left(\operatorname{sim}(\hat{z}_{i},z_{i}^{\text{sem}})/\tau\right)}{\sum_{j}\exp\left(\operatorname{sim}(\hat{z}_{i},z_{j}^{\text{sem}})/\tau\right)},

where z^i\hat{z}_{i} is a contextual embedding and z i sem z_{i}^{\text{sem}} a semantic embedding. Negative examples z j sem z_{j}^{\text{sem}} are sampled within the batch. We refer to this regularized approach as Contextual-Contrastive (CTX-C) and the unregularized baseline as Contextual-Base (CTX-B).

### 2.2 Embedding evaluation

#### Setting

We evaluate our framework using GPT-2 across four distinct reasoning domains: mathematical reasoning (GSM8K[[17](https://arxiv.org/html/2505.22202v2#bib.bib17)]), commonsense reasoning (CommonsenseQA[[18](https://arxiv.org/html/2505.22202v2#bib.bib18)]), logical reasoning (ProsQA[[19](https://arxiv.org/html/2505.22202v2#bib.bib19)]), and planning (Blocksworld). For each domain, we train on the respective training split and report accuracy on the corresponding test set, analyzing how well our framework generalizes across diverse linguistic subspaces. (i.e., mathematical expressions, natural language, etc.)∗∗∗For CSQA restoration, we trained on a small subset of FineWeb-Edu[[20](https://arxiv.org/html/2505.22202v2#bib.bib20)] due to small CSQA training set. See Appendix [B](https://arxiv.org/html/2505.22202v2#A2 "Appendix B Dataset Description ‣ Latent Reasoning via Sentence Embedding Prediction") and [E](https://arxiv.org/html/2505.22202v2#A5 "Appendix E Experiment Details ‣ Latent Reasoning via Sentence Embedding Prediction") for more details.

To evaluate semantic embedding’s performance, we compute exact match (EM) between the original reasoning step s i s_{i} and the decoder output, assessing how faithfully the model reconstructs unseen steps. For contextual evaluation, as there could be multiple correct next steps that could lead to the correct answer, we roll out the model autoregressively: at each step, the generated output y y is appended to the current input x x, continuing until a terminal answer is produced. The final answer is then compared against the ground-truth answer. Results are reported in Table[1](https://arxiv.org/html/2505.22202v2#S2.T1 "Table 1 ‣ Contextual embeddings. ‣ 2.1 Sentence embedding construction ‣ 2 Sentence embeddings for autoregressive modeling ‣ Latent Reasoning via Sentence Embedding Prediction").

#### Results

Across all domains, we observe that the autoencoder successfully restores the original sentences with high fidelity. This aligns with findings from Kuratov et al. [[15](https://arxiv.org/html/2505.22202v2#bib.bib15)], who show—both theoretically and empirically—that language models can compress a substantial number of tokens into compact representations. Yet, as we form CommonsenseQA (CSQA) task’s semantic embedding using a subset of Fineweb-Edu corpus (∼\sim 100k documents), we highlight that larger language space (compared to synthetic, constrained, i.e. ProsQA and Blocksworld) involves a higher difficulty.

In the Contextual configuration, model performance approaches that of the CoT baseline on three out of four benchmarks, and notably surpasses it on Blocksworld across both contextual variants. Introducing the contrastive alignment term (CTX-C) leads to a nuanced pattern: scores remain largely unchanged on GSM8K and Blocksworld, improve modestly on CommonsenseQA, but decline on ProsQA. These trends appear closely tied to each task’s underlying semantic structure.

CommonsenseQA questions exhibit substantial lexical variety, so anchoring each latent vector to its semantic counterpart helps tame surface variability. In contrast, ProsQA benefits from simultaneously tracking multiple evolving states; consequently, enforcing a single semantic target at each step restricts its representational flexibility, which is consistent with earlier findings[[19](https://arxiv.org/html/2505.22202v2#bib.bib19), [21](https://arxiv.org/html/2505.22202v2#bib.bib21)]. GSM8K and Blocksworld are highly symbolic and lexically sparse—thus, the baseline contextual embedding already forms an unambiguous mapping, leaving little space for improvement through additional regularization.

3 Sentence-Level Reasoning Model
--------------------------------

Given the strong reconstruction and predictive capabilities of semantic and contextual embeddings, we now present a framework that leverages these embeddings for sentence-level reasoning. (Figure[1](https://arxiv.org/html/2505.22202v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Latent Reasoning via Sentence Embedding Prediction"))

### 3.1 Architecture

We adapt a pretrained decoder-only Transformer[[22](https://arxiv.org/html/2505.22202v2#bib.bib22)] to operate directly over continuous sentence embeddings instead of discrete natural language tokens. We refer to this model as the _Latent Model_ θ LAT\theta_{\mathrm{LAT}}. Formally, given a natural language question q q and a sequence of latent embeddings corresponding to previously generated sentences h 1,…,h t h_{1},\dots,h_{t}, the latent model predicts the embedding for the next sentence:

h^t+1=θ LAT​(q,h≤t).\hat{h}_{t+1}=\theta_{\mathrm{LAT}}(q,h_{\leq t}).

At inference time, predicted embeddings h^t+1\hat{h}_{t+1} are mapped to the next input embedding h t+1 h_{t+1} using a mapping function ℳ:ℝ d→ℝ d\mathcal{M}:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}, where d d denotes the embedding dimensionality:

h t+1=ℳ​(h^t+1).h_{t+1}=\mathcal{M}(\hat{h}_{t+1}).

This process continues autoregressively, forming a latent embedding trajectory that encodes the progression of reasoning steps. At each step, a sentence decoder θ DEC:ℝ d→𝒯\theta_{\text{DEC}}:\mathbb{R}^{d}\rightarrow\mathcal{T} can decode latent embeddings back into natural language text. However, decoding intermediate reasoning steps is optional; embeddings can remain in their latent form to enhance computational efficiency, particularly when only the final answer is required. To this end, a lightweight termination classifier can evaluate each predicted embedding h^t\hat{h}_{t} to determine when reasoning should conclude.∗∗∗We use an oracle termination classifier for simplicity. See Appendix[D](https://arxiv.org/html/2505.22202v2#A4 "Appendix D Termination Classifier ‣ Latent Reasoning via Sentence Embedding Prediction") for more details.

### 3.2 Training

A natural approach for this task is to train the transformer model to generate sentence embeddings by minimizing the Mean Squared Error (MSE) between predicted and target embeddings. However, a single context often allows for several valid yet distinctly different continuations. [[8](https://arxiv.org/html/2505.22202v2#bib.bib8)]. Under these conditions, MSE tends to blend these varied possibilities into a single averaged representation, thus blurring meaningful variation.

To address this, we employ a cross-entropy (CE) loss calculated over natural language targets generated by a frozen decoder. This encourages predicted embeddings to align with the manifold defined by such decoder:

ℒ CE=−∑t=1 n−1 log⁡p​(s t+1∣θ DEC​(h^t+1)).\mathcal{L}_{\text{CE}}=-\sum_{t=1}^{n-1}\log p\big(s_{t+1}\mid\theta_{\text{DEC}}(\hat{h}_{t+1})\big).

During training, the latent model conditions on the question q q and ground-truth sentence embeddings h i h_{i}, each computed using a fixed encoder θ ENC\theta_{\text{ENC}}. Additionally, to enhance the alignment between predicted and teacher-forced embeddings, we incorporate an InfoNCE loss[[14](https://arxiv.org/html/2505.22202v2#bib.bib14)]:

ℒ InfoNCE=−∑t=1 n−1 log⁡exp⁡(sim​(h^t+1,h t+1)/τ)∑j exp⁡(sim​(h^t+1,h j)/τ).\mathcal{L}_{\text{InfoNCE}}=-\sum_{t=1}^{n-1}\log\frac{\exp\big(\mathrm{sim}(\hat{h}_{t+1},h_{t+1})/\tau\big)}{\sum_{j}\exp\big(\mathrm{sim}(\hat{h}_{t+1},h_{j})/\tau\big)}.

The overall training objective combines both terms:

ℒ overall=ℒ CE+λ​ℒ InfoNCE.\mathcal{L}_{\text{overall}}=\mathcal{L}_{\text{CE}}+\lambda\,\mathcal{L}_{\text{InfoNCE}}.

To further improve training stability, we include shallow projection layers between the encoder output and latent model input, and between the latent model output and decoder input.

### 3.3 Inference

We explore two strategies for defining the mapping function ℳ\mathcal{M} during inference. Let L L represent the average token length per reasoning step, and R R the total number of steps in a reasoning trace.

#### (1) Discretized (Language-Level)

Inspired by SentenceVAE[[23](https://arxiv.org/html/2505.22202v2#bib.bib23)], we apply a decode-and-reencode procedure: ℳ​(h^t)=E​(D​(h^t))\mathcal{M}(\hat{h}_{t})=E(D(\hat{h}_{t})), where the predicted latent is first decoded into a sentence and then re-encoded into the model’s input space. We refer to this as the Discretized mode, as each step explicitly traverses the discrete natural language interface. This approach helps mitigate error compounding[[24](https://arxiv.org/html/2505.22202v2#bib.bib24)], but comes at a higher computational cost, with attention cost scaling as 𝒪​(L 2​R+R 2)\mathcal{O}(L^{2}R+R^{2}). A detailed complexity analysis can be found in Appendix[C](https://arxiv.org/html/2505.22202v2#A3 "Appendix C Computation Complexity Analysis ‣ Latent Reasoning via Sentence Embedding Prediction").

#### (2) Continuous (Latent-Level)

Following Coconut[[19](https://arxiv.org/html/2505.22202v2#bib.bib19)], we define the mapping as an identity function ℳ=I\mathcal{M}=I, directly propagating the predicted latent embedding h^t\hat{h}_{t} without intermediate decoding. In this Continuous mode, reasoning is entirely performed within the continuous embedding space, enabling significantly more efficient inference with attention complexity reduced to 𝒪​(R 2)\mathcal{O}(R^{2}).

Both methods offer computational advantages over natural language CoT, which incurs 𝒪​(L 2​R 2)\mathcal{O}(L^{2}R^{2}) attention complexity even under key-value caching. However, the savings in the Discretized mode are conditional: they occur only when either (1) the encoder-decoder are not too computation-heavy, or (2) attention dominates over MLP cost—typially when the total output length L​R LR is relatively long (e.g., Blocksworld). Otherwise, the repeated decoding and encoding introduce additional MLP overhead.∗∗∗Note that using a contextual encoder incurs greater computational cost than a semantic encoder.

### 3.4 Experiments

Building upon prior studies[[19](https://arxiv.org/html/2505.22202v2#bib.bib19), [21](https://arxiv.org/html/2505.22202v2#bib.bib21)], we select GPT-2 as our baseline model and evaluate its performance across four distinct reasoning domains detailed in Section[2](https://arxiv.org/html/2505.22202v2#S2 "2 Sentence embeddings for autoregressive modeling ‣ Latent Reasoning via Sentence Embedding Prediction"). To investigate optimal embedding strategies for latent reasoning, we examine Semantic and Contextual (both Ctx-B and Ctx-C) embeddings from Section[2](https://arxiv.org/html/2505.22202v2#S2 "2 Sentence embeddings for autoregressive modeling ‣ Latent Reasoning via Sentence Embedding Prediction"). We further explore a hybrid architecture—Sem (input) →\rightarrow Ctx (output)—which mirrors the natural separation of representational roles found in conventional language modeling.

For evaluation, we compare sentence-level reasoning models against three baseline models. First, CoT represents a fully supervised model trained with access to both intermediate reasoning steps and final answers. Second, No-CoT omits step-level supervision and is trained solely to predict final answers. Third, we include Coconut[[19](https://arxiv.org/html/2505.22202v2#bib.bib19)], which gradually forgoes explicit token-level targets with curriculum-based substitution of fixed number last hidden states.

### 3.5 Results

Again, our main objective is to examine whether a latent sentence-level reasoning framework can effectively generalize to higher-level abstractions while preserving the learned priors of the model. Achieving comparable performance to token-level Chain-of-Thought (CoT) would provide preliminary evidence toward this goal. To this end, we address the following three research questions.

Table 2: Performance on ProsQA, CSQA, GSM8K, and Blocksworld across different embedding paradigms. Bolded values indicate the best performance among our proposed methods within each section. Baseline results are highlighted with background colors.

#### Q1: Can sentence-level reasoning match token-level CoT performance?

We hypothesize that effective reasoning is driven more by transitions between high-level concepts than by fine-grained token-level details. Empirically, sentence-level models match or even exceed CoT performance on logical and commonsense reasoning tasks. On mathematical and planning benchmarks, performance is slightly lower, though the gap remains modest. We attribute this to the greater precision often required in these domains, where continuous latent representations may be more prone to fidelity loss.

#### Q2: How does sentence-level reasoning differ between language-level and latent-level inference?

To explore this, we compare model inference in the Discretized (language-level) space with that in the Continuous (latent-level) space. Results reveal complementary strengths: continuous models excel on logic and planning tasks, where reasoning benefits from uninterrupted latent-space composition and abstract state transitions. Conversely, discretized models show modest advantages on commonsense and mathematical benchmarks—likely due to the grounding effect of explicit linguistic representations. Still, the observed performance gaps are narrow—3.3% on commonsense and 0.7% on math—indicating that latent inference remains a viable and compute-efficient alternative. These findings suggest that effective reasoning need not always traverse explicit language space; continuous representations alone may support structured inference.

#### Q3: Can sentence-level reasoning reduce computational cost?

Table 3: Average inference-time compute cost (GFLOPs) for each dataset under CoT and Ctx-C Continuous Inference.

Table[8](https://arxiv.org/html/2505.22202v2#A6.T8 "Table 8 ‣ Appendix F Evaluation Prompt ‣ Latent Reasoning via Sentence Embedding Prediction") compares computational costs (FLOPs) between latent reasoning model and token-level CoT under forward-pass evaluation with key-value caching enabled. Latent reasoning employs an oracle answer classifier—executed via a single forward pass through the translator—that monitors the predicted embedding sequence and halts generation upon detecting a special answer token. The final latent embedding is decoded into natural language for evaluation.

Note that we measure computational costs across the full latent pipeline, including classifier and decoder components, which remain unoptimized.∗∗∗To see the cost with a lightweight classifier, please refer to Appendix [D](https://arxiv.org/html/2505.22202v2#A4 "Appendix D Termination Classifier ‣ Latent Reasoning via Sentence Embedding Prediction"). Thus, reported efficiency gains represent conservative estimate. Across tasks, Continuous inference achieves 1.5–2.5×\times better efficiency compared to token-level CoT. Notably, we highlight that even Discretized inference outperform CoT in longer reasoning tasks (e.g., Blocksworld w/ average trace length R∼9.1 R\sim 9.1: 52.26 GFLOPs vs. 58.69 GFLOPs). We expect this efficiency gap to grow as the length of reasoning trace increases.

![Image 3: Refer to caption](https://arxiv.org/html/2505.22202v2/figs/scale_chart.png)

(a)CoT vs. CTX-B on CommonsenseQA across GPT-2 variants.

![Image 4: Refer to caption](https://arxiv.org/html/2505.22202v2/figs/qual_analysis.png)

(b)GPT-4o Qualitative evaluation of the reasoning steps evaluated using a similar metric employed in [[25](https://arxiv.org/html/2505.22202v2#bib.bib25)], where SFT is trained using CoT and ours is using CTX-B.

4 Discussion
------------

### 4.1 Potential Scalability and Modularity

#### Scalability

We report preliminary observations that suggest our framework has potential to scale to increasing model capacity. Due to computational constraints, our experiments are limited to sub-1B models; we evaluate GPT-2 Medium (345M) and GPT-2 Large (775M) on the CommonsenseQA (CSQA) benchmark, which exhibits clear performance scaling under CoT fine-tuning. As shown in Figure[3(a)](https://arxiv.org/html/2505.22202v2#S3.F3.sf1 "In Q3: Can sentence-level reasoning reduce computational cost? ‣ 3.5 Results ‣ 3 Sentence-Level Reasoning Model ‣ Latent Reasoning via Sentence Embedding Prediction"), the Ctx-C configuration attains performance comparable to, and in some cases exceeding, CoT—despite operating entirely in latent space and incurring lower inference-time compute. While tentative, these findings suggest that latent reasoning could offer a more compute-efficient path toward generalization. However, we acknowledge that scaling to extensively pretrained models remains as a challenge, since stable adaptation under greater distribution shifts could be more difficult[[19](https://arxiv.org/html/2505.22202v2#bib.bib19)].

#### Using Off-the-Shelf Encoder–Decoder

We investigate whether the encoder–decoder can be decoupled from the latent model and replaced with smaller, fixed components. This modular design seeks to reduce the computational burden of Discretized inference—especially in settings where only the latent reasoning module requires adaptation. To evaluate this hypothesis, we paired a lightweight GPT-2 Small encoder–decoder (trained on Ctx-C) with a GPT-2 Medium latent model and assessed performance on GSM8K.∗∗∗GSM8K was selected based on preliminary findings that moderately sized datasets help stabilize shallow MLP mappings across heterogeneous embedding spaces.

This hybrid configuration achieved an accuracy of 42.23, compared to 47.69 for a fully fine-tuned GPT-2 Medium with CoT training. While accuracy decreases slightly, the results demonstrate that predictive embeddings can transfer across model architectures with reasonable degradation–supporing the feasibility of modular reuse. Given prior findings on general embedding space alignment across models[[26](https://arxiv.org/html/2505.22202v2#bib.bib26), [27](https://arxiv.org/html/2505.22202v2#bib.bib27)], further exploration with larger models and diverse tasks remains a promising direction.

Table 4: Latent Sentence Transitions with SentenceLens for GPT2-Large under the Ctx-C, Continuous setting. We visualize intermediate decoding across layers and reasoning steps. Highlighted rows represent the output from the final latent embedding at each step.

Table 5: Natural Language CoT Trace. Output from the CoT trained model (CoT)

### 4.2 SentenceLens: Towards Human-Readable Interpretability

We introduce SentenceLens, an intrepretability tool that decodes intermediate hidden representations by directly passing them through the trained sentence-level decoder. In contrast to token-level inspection methods such as Logit Lens[[28](https://arxiv.org/html/2505.22202v2#bib.bib28)], SentenceLens operates at the sentence level, offering a more human-readable view of the model’s evolving internal states across reasoning steps.

For example, in Table[6](https://arxiv.org/html/2505.22202v2#A1.T6 "Table 6 ‣ Appendix A SentenceLens Examples ‣ Latent Reasoning via Sentence Embedding Prediction"), we show how the model’s prediction shifts across layers during the transition from one reasoning step to the next. When making first step prediction h^1\hat{h}_{1}, Layer 19 introduces a general observation about eating and energy levels, while Layer 22 begins to center on the idea that hunger motivates goal-directed behavior. These intermediate activations reflect a gradual shift in conceptual focus, which in the last layer (36 t​h 36^{th}) develops as: If you are hungry, you are likely engaging in an activity that requires sustenance. Since the latent model frames reasoning as a continuous process, we hypothesize that intermediate latent states may become naturally decodable—allowing us to observe the progression of inference across steps. To see more examples, see Appendix[A](https://arxiv.org/html/2505.22202v2#A1 "Appendix A SentenceLens Examples ‣ Latent Reasoning via Sentence Embedding Prediction").

#### Qualitative Analysis

In addition, when decoding output embeddings at successive latent reasoning steps (e.g., Step 1 through Step 5), we find that the resulting sentences, while readily understandable, often lack the coherence and rigor characteristic of standard CoT responses. We compare two model outputs using GPT-4o evaluation with the rubric proposed by Ye et al. [[25](https://arxiv.org/html/2505.22202v2#bib.bib25)]. This scores Relevance, Fluency, Conciseness, Soundness, and Interpretability on a 1 to 5 Likert scale. It turns out that CTX-C model mostly produces reasoning chains of moderate quality (scores > 3); However, its performance falls short compared to CoT models trained directly in natural language space (Figure[3(b)](https://arxiv.org/html/2505.22202v2#S3.F3.sf2 "In Q3: Can sentence-level reasoning reduce computational cost? ‣ 3.5 Results ‣ 3 Sentence-Level Reasoning Model ‣ Latent Reasoning via Sentence Embedding Prediction")). The largest weakness appears in Soundness, which aligns with earlier observations that high-level concept models may exhibit reduced coherence even after extensive pretraining[[8](https://arxiv.org/html/2505.22202v2#bib.bib8)]. While we believe this tradeoff is a natural consequence of abstraction, bridging this gap remains an interesting direction for future research.

![Image 5: Refer to caption](https://arxiv.org/html/2505.22202v2/figs/robustness.png)

Figure 4: Performance Change when injecting a Gaussian random noise to different modes of inferencing, for Ctx-C model in GSM8K and CSQA datasets.

#### Future Directions

Another interesting direction is to self-train the model by using its own intermediate decoded sentences as auxiliary supervision targets. We also observe the correct answer often surfaces early in the reasoning trajectory. (see Appendix[A](https://arxiv.org/html/2505.22202v2#A1 "Appendix A SentenceLens Examples ‣ Latent Reasoning via Sentence Embedding Prediction")). In this light, these intermediate outputs could offer a novel training signal that could enhance both reasoning efficiency and stability. Furthermore, unlike prior latent reasoning approaches, our framework allows for sampling in the token-level after decoding. This opens the door to applying reinforcement learning or trajectory-level optimization over the latent reasoning chain.

### 4.3 Fragility of Continuous Embeddings

Latent reasoning operates over high-dimensional embedding manifolds, which tend to be more sensitive to perturbations than discrete token-level autoregression[[8](https://arxiv.org/html/2505.22202v2#bib.bib8), [24](https://arxiv.org/html/2505.22202v2#bib.bib24)]. To systematically assess this fragility, we introduce synthetic noise at inference time, following team et al. [[8](https://arxiv.org/html/2505.22202v2#bib.bib8)] with a 50% probability. We evaluate robustness across three intervention points in the reasoning pipeline: (1) Language-Level (Input): noise is applied to the input embedding; (2) Language-Level (Output): noise is added to the output embedding; and (3) Latent-Level: noise is directly injected into the predicted output embedding, which is then autoregressively consumed in the next step.

Empirically, we observe two key trends: (1) performance degrades more rapidly on GSM8K, where precise numerical reasoning amplifies the impact of noise; and (2) _Language-Level_ inference (i.e., decoding and re-encoding) consistently yields greater robustness than latent-only reasoning across both tasks. This supports the intuition that grounding in language acts as a regularizing prior, mitigating error accumulation at the cost of additional compute. These findings highlight a trade-off between efficiency and stability, motivating future work on approaches that help prevent error compounding.

5 Related Works
---------------

#### Sentence Representations

Sentence-level representation learning has historically followed two main paradigms: _reconstruction_ and _context prediction_. Early methods, such as sequence autoencoders[[10](https://arxiv.org/html/2505.22202v2#bib.bib10)] and Skip-Thought vectors[[11](https://arxiv.org/html/2505.22202v2#bib.bib11)], learned fixed-length sentence embeddings by reconstructing input or neighboring sentences. Subsequent research, exemplified by Quick-Thought[[29](https://arxiv.org/html/2505.22202v2#bib.bib29)], shifted towards contrastive prediction, focusing on distinguishing the correct sentence context from distractors.

Contrastive learning builds on these paradigms by explicitly aligning semantically related sentences while distinguishing unrelated examples. Models such as Sentence-BERT[[30](https://arxiv.org/html/2505.22202v2#bib.bib30)] and SimCSE[[31](https://arxiv.org/html/2505.22202v2#bib.bib31)], inspired by SimCLR[[32](https://arxiv.org/html/2505.22202v2#bib.bib32)], have produced robust sentence embeddings with excellent transfer performance. Our framework builds upon these developments by defining semantic and contextual embeddings and employing contrastive learning to align latent input-output pairs[[12](https://arxiv.org/html/2505.22202v2#bib.bib12)].

#### Sentence-Level Prediction

Several models move beyond token-level generation to predict entire sentences. Latent-variable approaches such as VAEs[[33](https://arxiv.org/html/2505.22202v2#bib.bib33)] and hierarchical decoders[[34](https://arxiv.org/html/2505.22202v2#bib.bib34)] generate sentences from continuous codes. LCM[[8](https://arxiv.org/html/2505.22202v2#bib.bib8)] autoregresses over sentence-level “concept” embeddings in a multilingual, multimodal space, while CoCoMix[[9](https://arxiv.org/html/2505.22202v2#bib.bib9)] injects sparse autoencoder-derived vectors into hidden states to improve interpretability and control. Our method similarly operates over latent embeddings but distinguishes itself by building upon pretrained models rather than training from scratch. This approach allows us to leverage existing language understanding capabilities while introducing latent reasoning mechanisms.

#### Latent-Space Reasoning

Efficiency and abstraction have motivated reasoning directly in embedding space, bypassing token generation. Joint embedding architectures[[35](https://arxiv.org/html/2505.22202v2#bib.bib35)] and predictive coding frameworks[[12](https://arxiv.org/html/2505.22202v2#bib.bib12)] model representation dynamics by forecasting future embeddings. This idea has recently been extended to language: Hao et al. [[19](https://arxiv.org/html/2505.22202v2#bib.bib19)] introduced continuous latent reasoning, where token-level embeddings are gradually replaced with continuous embeddings with the last-layer hidden states through a curriculum-based strategy from Deng et al. [[21](https://arxiv.org/html/2505.22202v2#bib.bib21)]. Further extensions include, among others, methods by Shen et al. [[36](https://arxiv.org/html/2505.22202v2#bib.bib36)] which guide latent rollouts using self-distillation; and Su et al. [[37](https://arxiv.org/html/2505.22202v2#bib.bib37)] which propose mixing discrete token embeddings from trained VQ-VAE[[14](https://arxiv.org/html/2505.22202v2#bib.bib14)] for inference efficiency.

Our work differs from these approaches primarily in three ways. (1) We provide explicit access to intermediate latent states through decoding, offering clearer insights into the reasoning trajectory. (2) Our method uniquely supports token-level sampling during latent-level reasoning, opening exciting research avenues such as self-training and reinforcement learning. (3) Whereas previous methods require iterative sampling of latent representations during training, which involves n+1 forward passes per iteration, our approach completes this in a single forward pass, significantly improving scalability.

6 Conclusion
------------

We present a framework that elevates pretrained language models from token-level generation to sentence-level reasoning by autoregressively predicting continuous embeddings of next-step sentences. This enables reasoning over more abstract conceptual units while retaining pretrained inductive biases. Our exploration of semantic and contextual embeddings reveals that contextual embeddings show competitive performance with token-level Chain-of-Thought (CoT) across diverse reasoning tasks, while significantly reducing inference-time computational costs under Continuous inference. Additionally, we demonstrate signs of scalability, modular reuse of encoder–decoder components, and enhanced interpretability through SentenceLens, which decodes latent embeddings into human-readable sentence-level traces. These findings suggest that pretrained language models could be effectively adapted for structured reasoning in latent embedding spaces, opening new directions for efficient latent reasoning systems.

Limitations
-----------

#### Need for Large-Scale Experiments

We conduct a preliminary exploration of sentence-level reasoning with GPT-2 variants. To keep experiments reproducible, we start with GPT-2 Small as our base model—following recent work on latent-level reasoning[[21](https://arxiv.org/html/2505.22202v2#bib.bib21), [19](https://arxiv.org/html/2505.22202v2#bib.bib19)] and then explore scalability by evaluating GPT-2 Medium, GPT-2 Large, and a lightweight hybrid that pairs a GPT-2 Medium latent core with a GPT-2 Small encoder–decoder.

During our experiments, we observed that larger models become somewhat more sensitive to hyperparameter choices which could often lead to increased performance gap between our method and CoT training. We note that this increased gap has been observed for similar preliminary researches when scaled to more competitive models (i.e. Llama 3[[38](https://arxiv.org/html/2505.22202v2#bib.bib38)]), and conjecture as one of the reasons why recent works have turned towards pretraining.

We hypothesize such challenge arises from the widening gap between the token-level embedding distributions learned during pretraining and the compact, coarser-grained manifold our adapter enforces. In effect, the very inductive biases that make large models robust in token space may conflict with sentence-level abstractions. A systematic study of this tension—and the design of transfer mechanisms that preserve high-capacity knowledge while avoiding overfitting to the latent manifold—remains an important avenue for future research.

#### Fragility of Latent Reasoning

As illustrated in Figure[4](https://arxiv.org/html/2505.22202v2#S4.F4 "Figure 4 ‣ Qualitative Analysis ‣ 4.2 SentenceLens: Towards Human-Readable Interpretability ‣ 4 Discussion ‣ Latent Reasoning via Sentence Embedding Prediction"), pure latent reasoning, as it is conducted entirely within a continuous embedding space, becomes notably fragile. Unlike Discrete-Step inference, which introduces a discrete decoding step that inherently quantizes minor perturbations, the continuous pathway lacks such built-in stabilization. This discrete bottleneck serves as a form of regularization, filtering out numerical noise and constraining the model’s trajectory to a finite set of linguistically meaningful sequences. However, this regularization comes at the expense of expressivity, limiting outputs to token sequences present in the vocabulary.

In a fully continuous framework, the model must learn to establish implicit attractors or decision boundaries to maintain trajectories within a coherent manifold—effectively performing a form of soft discretization. These learned boundaries, being approximate, may allow small deviations to persist and amplify over successive reasoning steps, potentially leading to significant semantic errors, especially in tasks demanding precision or extended reasoning chains. This vulnerability mirrors challenges observed in continuous control systems, where minor deviations can accumulate over time, resulting in substantial performance degradation unless addressed through specialized stabilization mechanisms[[24](https://arxiv.org/html/2505.22202v2#bib.bib24)]. Future work could explore hybrid framework that integrate discrete bottlenecks at critical junctures within the reasoning process, aiming to combine the robustness of discretization with the flexibility of continuous representations.

#### Training from Scratch

Training a model from scratch directly in the higher abstractions i.e. sentence embeddings space appears, at first glance, to be the cleanest path toward robust high-level reasoning. Prior work argues that models initialized on discrete-token objectives must later overcome a distribution shift when asked to operate over sentence-level abstractions, and this difficulty intensifies as model size—and pretraining data size—increase[[19](https://arxiv.org/html/2505.22202v2#bib.bib19), [21](https://arxiv.org/html/2505.22202v2#bib.bib21)] and therefore has leaned towards pretraining[[39](https://arxiv.org/html/2505.22202v2#bib.bib39), [36](https://arxiv.org/html/2505.22202v2#bib.bib36)]

Yet genuine intelligence might not rely on starting from a clean slate each time the abstraction level changes. We hypothesize that a system that truly generalizes beyond human capability must be able to climb the ladder of abstraction after exposure to raw experience, flexibly re-encoding its knowledge in coarser units. At the same time, safety considerations dictate that these higher-order representations remain interpretable—anchored to a manifold we can inspect and, when necessary, constrain.

Our adaptation framework takes a step in this direction: it shows that a pretrained token-level language model can be lifted, with modest additional supervision, onto an interpretable sentence-manifold without retraining everything from scratch. By demonstrating both the promise and the fragility of this approach, the present work highlights a critical research frontier: designing models that learn to abstract while preserving previously learned inductive bias.

Broader Impacts
---------------

This work introduces a novel framework for reasoning in continuous latent space, offering both practical and societal benefits. By avoiding token-level autoregressive decoding, it reduces computational overhead and may lower the environmental footprint of large-scale inference. Importantly, our method maintains interpretability by anchoring latent representations to human-readable abstractions.

Nonetheless, broader risks remain. If latent reasoning frameworks are deployed without transparency mechanisms, they may obscure decision processes—especially in high-stakes domains. Additionally, latent representations could encode and propagate biases present in pretraining data. As reasoning becomes more abstracted from language, care must be taken to ensure meaningful human oversight is preserved. We encourage future work to strengthen interpretability guarantees and explore safeguards that prevent misuse or unintended consequences.

Acknowledgment
--------------

We thank Seonghyeon Ye, Jinho Park, Seongyun Lee, and Jaehyeok Doo for their insightful discussions and valuable feedback.

References
----------

*   Bengio et al. [2003] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic language model. _J. Mach. Learn. Res._, 3(null):1137–1155, March 2003. ISSN 1532-4435. 
*   Wei et al. [2022] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_, 35:24824–24837, 2022. 
*   Jaech et al. [2024] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card, 2024. 
*   Guo et al. [2025] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 
*   Fodor [1975] Jerry Fodor. _The Language of Thought_. Harvard University Press, 1975. 
*   Mercier and Sperber [2011] Hugo Mercier and Dan Sperber. Why do humans reason? arguments for an argumentative theory. _Behavioral and Brain Sciences_, 34(2):57–74, 2011. doi: 10.1017/S0140525X10000968. 
*   Bengio [2019] Yoshua Bengio. The consciousness prior, 2019. URL [https://arxiv.org/abs/1709.08568](https://arxiv.org/abs/1709.08568). 
*   team et al. [2024] LCM team, Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R. Costa-jussà, David Dale, Hady Elsahar, Kevin Heffernan, João Maria Janeiro, Tuan Tran, Christophe Ropers, Eduardo Sánchez, Robin San Roman, Alexandre Mourachko, Safiyyah Saleem, and Holger Schwenk. Large concept models: Language modeling in a sentence representation space, 2024. URL [https://arxiv.org/abs/2412.08821](https://arxiv.org/abs/2412.08821). 
*   Tack et al. [2025] Jihoon Tack, Jack Lanchantin, Jane Yu, Andrew Cohen, Ilia Kulikov, Janice Lan, Shibo Hao, Yuandong Tian, Jason Weston, and Xian Li. Llm pretraining with continuous concepts, 2025. URL [https://arxiv.org/abs/2502.08524](https://arxiv.org/abs/2502.08524). 
*   Dai and Le [2015] Andrew M. Dai and Quoc V. Le. Semi-supervised sequence learning, 2015. URL [https://arxiv.org/abs/1511.01432](https://arxiv.org/abs/1511.01432). 
*   Kiros et al. [2015] Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Skip-thought vectors, 2015. URL [https://arxiv.org/abs/1506.06726](https://arxiv.org/abs/1506.06726). 
*   van den Oord et al. [2019] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. URL [https://arxiv.org/abs/1807.03748](https://arxiv.org/abs/1807.03748). 
*   Hill et al. [2016] Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences from unlabelled data, 2016. URL [https://arxiv.org/abs/1602.03483](https://arxiv.org/abs/1602.03483). 
*   van den Oord et al. [2018] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning, 2018. URL [https://arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937). 
*   Kuratov et al. [2025] Yuri Kuratov, Mikhail Arkhipov, Aydar Bulatov, and Mikhail Burtsev. Cramming 1568 tokens into a single vector and back again: Exploring the limits of embedding space capacity, 2025. URL [https://arxiv.org/abs/2502.13063](https://arxiv.org/abs/2502.13063). 
*   Ge et al. [2024] Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model, 2024. URL [https://arxiv.org/abs/2307.06945](https://arxiv.org/abs/2307.06945). 
*   Cobbe et al. [2021] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021. 
*   Talmor et al. [2019] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge, 2019. URL [https://arxiv.org/abs/1811.00937](https://arxiv.org/abs/1811.00937). 
*   Hao et al. [2024] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2024. URL [https://arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769). 
*   Penedo et al. [2024] Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL [https://arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557). 
*   Deng et al. [2024] Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step, 2024. URL [https://arxiv.org/abs/2405.14838](https://arxiv.org/abs/2405.14838). 
*   Vaswani et al. [2023] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. URL [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762). 
*   An et al. [2024] Hongjun An, Yifan Chen, Zhe Sun, and Xuelong Li. Sentencevae: Enable next-sentence prediction for large language models with faster speed, higher accuracy and longer context, 2024. URL [https://arxiv.org/abs/2408.00655](https://arxiv.org/abs/2408.00655). 
*   Simchowitz et al. [2025] Max Simchowitz, Daniel Pfrommer, and Ali Jadbabaie. The pitfalls of imitation learning when actions are continuous, 2025. URL [https://arxiv.org/abs/2503.09722](https://arxiv.org/abs/2503.09722). 
*   Ye et al. [2023] Seonghyeon Ye, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, Seungone Kim, Yongrae Jo, James Thorne, Juho Kim, and Minjoon Seo. Flask: Fine-grained language model evaluation based on alignment skill sets. _arXiv preprint arXiv:2307.10928_, 2023. 
*   Conneau et al. [2018] Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Word translation without parallel data, 2018. URL [https://arxiv.org/abs/1710.04087](https://arxiv.org/abs/1710.04087). 
*   Jha et al. [2025] Rishi Jha, Collin Zhang, Vitaly Shmatikov, and John X. Morris. Harnessing the universal geometry of embeddings, 2025. URL [https://arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540). 
*   nostalgebraist [2020] nostalgebraist. interpreting gpt: the logit lens, 2020. URL [https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). [https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). 
*   Logeswaran and Lee [2018] Lajanugen Logeswaran and Honglak Lee. An efficient framework for learning sentence representations, 2018. URL [https://arxiv.org/abs/1803.02893](https://arxiv.org/abs/1803.02893). 
*   Reimers and Gurevych [2019] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks, 2019. URL [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084). 
*   Gao et al. [2022] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings, 2022. URL [https://arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821). 
*   Chen et al. [2020] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations, 2020. URL [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709). 
*   Bowman et al. [2016] Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space, 2016. URL [https://arxiv.org/abs/1511.06349](https://arxiv.org/abs/1511.06349). 
*   Serban et al. [2016] Iulian Vlad Serban, Alessandro Sordoni, Ryan Lowe, Laurent Charlin, Joelle Pineau, Aaron Courville, and Yoshua Bengio. A hierarchical latent variable encoder-decoder model for generating dialogues, 2016. URL [https://arxiv.org/abs/1605.06069](https://arxiv.org/abs/1605.06069). 
*   Assran et al. [2023] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture, 2023. URL [https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243). 
*   Shen et al. [2025] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation, 2025. URL [https://arxiv.org/abs/2502.21074](https://arxiv.org/abs/2502.21074). 
*   Su et al. [2025] DiJia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, and Qinqing Zheng. Token assorted: Mixing latent and text tokens for improved language model reasoning, 2025. URL [https://arxiv.org/abs/2502.03275](https://arxiv.org/abs/2502.03275). 
*   Grattafiori et al. [2024] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783). 
*   Goyal et al. [2024] Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens, 2024. URL [https://arxiv.org/abs/2310.02226](https://arxiv.org/abs/2310.02226). 
*   Bohnet et al. [2024] Bernd Bohnet, Azade Nova, Aaron T Parisi, Kevin Swersky, Katayoon Goshvadi, Hanjun Dai, Dale Schuurmans, Noah Fiedel, and Hanie Sedghi. Exploring and benchmarking the planning capabilities of large language models. _arXiv preprint arXiv:2406.13094_, 2024. 

Appendix A SentenceLens Examples
--------------------------------

We include a representative SentenceLens example that highlights additional key observations. Specifically, the model often identifies the correct answer early in the latent trajectory; however, subsequent chain-of-thought (CoT) tokens exhibit a drift that ultimately leads to an incorrect prediction. (The correct answer is C.) This suggests room for improvement by using intermediate representations as explicit supervision targets, which could guide the construction of model centric datasets and self-training methods.

Table 6: Example of Latent Reasoning Trajectory inspected with SentenceLens. Although early steps’ intermediate layers demonstrate accurate associations with hair loss and balding, the final prediction selects an incorrect choice, showing a drift in reasoning at later stages.

Step Decoded Sentence(s)
Question Why would you take a bus to work?
A: commute B: flying C: get somewhere D: travel E: go home
0 →\rightarrow 1 Layer 19: A person spends time traveling between different locations.
Layer 20: A person spends time commuting to work.
Layer 21: A person spends time traveling, which often involves moving from one place to another.
Layer 22: A person spends time traveling, which often involves traveling across distances.
1 _People often take the bus to reach a destination._
_… not shown_
5 _### A (Correct)_

Table 7: Early Answer Emergence in Latent Reasoning. The model brings up the concept of “commuting” in the reasoning chain even before the first autoregressive step completes. This hints at potential efficiency gains by leveraging early, confident predictions as supervision signals in training.

Appendix B Dataset Description
------------------------------

#### Mathematics

We use the GSM8K dataset[[17](https://arxiv.org/html/2505.22202v2#bib.bib17)], which consists of grade-school math word problems originally comprising 7.8k training and 1.3k test samples. Following prior expansions[[19](https://arxiv.org/html/2505.22202v2#bib.bib19), [21](https://arxiv.org/html/2505.22202v2#bib.bib21)], we adopt an extended version containing approximately 370k training examples to support large-scale latent model training.

#### Planning

Following prior work[[40](https://arxiv.org/html/2505.22202v2#bib.bib40)], we use the Blocksworld environment for planning evaluation, but construct the dataset generation pipeline using our own Python implementation. We evaluate the model on 7-block configurations, ensuring that the initial and goal states do not overlap across the training, validation, and test sets. We use 9.9k samples for training, and 380 samples each for testing.

#### Logical

We adopt ProsQA[[19](https://arxiv.org/html/2505.22202v2#bib.bib19)], a synthetic dataset grounded in first-order logic. Each instance presents multiple distractors and requires multi-hop reasoning over a structured graph. Prior work highlights that latent models capable of multi-state tracking exhibit strong performance on this task. We use a 17.8k training set and 500 samples for evaluation.

#### Commonsense

We use CommonsenseQA[[18](https://arxiv.org/html/2505.22202v2#bib.bib18)], a multiple-choice benchmark that lacks explicit Chain-of-Thought (CoT) supervision. To enable training with intermediate reasoning steps, we augment the data using GPT-4o to generate CoT-style rationales. Our training split includes 8.5k examples, and for evaluation, we reserve 611 samples from the validation set.

Figure[6](https://arxiv.org/html/2505.22202v2#A6.F6 "Figure 6 ‣ Appendix F Evaluation Prompt ‣ Latent Reasoning via Sentence Embedding Prediction") illustrates representative examples from each dataset.

Appendix C Computation Complexity Analysis
------------------------------------------

#### Attention Complexity under KV‐caching

Let L L be the average number of tokens per sentence, R R the number of reasoning steps, and ignore the prompt length N 0 N_{0} in leading order.

1.   (1)Chain‐of‐Thought (CoT). Each step emits L L new tokens into the context. Before step t t, the context length is N 0+(t−1)​L N_{0}+(t-1)L, so

𝒞 CoT=∑t=1 R∑j=1 L[N 0+(t−1)​L+(j−1)]=𝒪​(L 2​R 2).\mathcal{C}_{\mathrm{CoT}}=\sum_{t=1}^{R}\sum_{j=1}^{L}\bigl[N_{0}+(t-1)L+(j-1)\bigr]=\mathcal{O}\bigl(L^{2}R^{2}\bigr). 
2.   (2)Contextual Embedding Mode. At each step the model (i) decodes one latent into an L L-token sentence and (ii) attends over all retained tokens to predict the next latent:

∑t=1 R∑j=1 L j⏟𝒪​(L 2​R)+∑t=1 R(N 0+(t−1)​L)⏟𝒪​(L​R 2)=𝒪​(L 2​R+L​R 2).\underbrace{\sum_{t=1}^{R}\sum_{j=1}^{L}j}_{\mathcal{O}(L^{2}R)}\;+\;\underbrace{\sum_{t=1}^{R}(N_{0}+(t-1)L)}_{\mathcal{O}(L\,R^{2})}=\mathcal{O}\bigl(L^{2}R+L\,R^{2}\bigr). 
3.   (3)Language‐Grounded Mode. Each step (i) processes only latents in the main chain (𝒪​(R 2))\bigl(\mathcal{O}(R^{2})\bigr) and (ii) decodes and re‐encodes an L L-token sentence (𝒪​(L 2​R))\bigl(\mathcal{O}(L^{2}R)\bigr), yielding

𝒞 LG=𝒪​(R 2+L 2​R).\mathcal{C}_{\mathrm{LG}}=\mathcal{O}\bigl(R^{2}+L^{2}R\bigr). 
4.   (4)Pure Latent Mode. Each step adds one latent vector; attending over t−1 t-1 latents gives

𝒞 latent=∑t=1 R(N 0+t−1)=𝒪​(R 2).\mathcal{C}_{\mathrm{latent}}=\sum_{t=1}^{R}(N_{0}+t-1)=\mathcal{O}\bigl(R^{2}\bigr). 

Summary of leading‐order costs:

𝒞 CoT=O​(L 2​R 2),𝒞 contextual=O​(L 2​R+L​R 2),𝒞 LG=O​(L 2​R+R 2),𝒞 latent=O​(R 2).\mathcal{C}_{\mathrm{CoT}}=O(L^{2}R^{2}),\quad\mathcal{C}_{\mathrm{contextual}}=O(L^{2}R+L\,R^{2}),\quad\mathcal{C}_{\mathrm{LG}}=O(L^{2}R+R^{2}),\quad\mathcal{C}_{\mathrm{latent}}=O(R^{2}).

#### MLP Overhead

In addition to attention cost, every decoded or re‐encoded token incurs feed‐forward (MLP) computation. More specifically:

*   •CoT & Contextual Embedding: emits L L tokens per step → processes L×R L\times R tokens through MLP → 𝒪​(L​R)\mathcal{O}(LR). 
*   •Language‐Grounded: With a _semantic_ encoder, each step decodes and re‐encodes L L tokens on compact codes—processing 2​L 2L tokens per step for an MLP cost of 𝒪​(L​R)\mathcal{O}(LR). If instead a _contextual_ encoder must re‐attend over up to N 0+(t−1)​L N_{0}+(t-1)L tokens each pass, it incurs an additional 𝒪​(L​R 2)\mathcal{O}(LR^{2}) MLP overhead, which can erode attention savings unless the encoder is shallow or non‐autoregressive. 
*   •Pure Latent: processes one latent per step → 𝒪​(R)\mathcal{O}(R). 

#### Concluding Remark

Under KV‐caching, the Language‐Grounded mode—with a semantic encoder—adds an 𝒪​(L 2​R)\mathcal{O}(L^{2}R) decode/re‐encode overhead, but makes it ideal for tasks sensitive to error‐compounding or instability (i.e. Mathematics.) In contrast, the Pure Latent mode eliminates all token‐level context (attention 𝒪​(R 2)\mathcal{O}(R^{2}), MLP 𝒪​(R)\mathcal{O}(R)), offering maximal efficiency when possible.

Appendix D Termination Classifier
---------------------------------

While we initially assume an oracle termination signal by using the first token generated by the decoder, we also demonstrate that this decision can be learned by a lightweight classifier. Specifically, we train a three-layer feedforward neural network (MLP) to identify the answer sentence during continuous inference. The MLP consists of linear layers with hidden dimensions of 192 and 48, each followed by a GELU activation, and outputs a single logit for binary classification (continue versus terminate). It is trained using binary cross-entropy loss with logits (BCEWithLogitsLoss). Note that the average inference GFLOPs, reported in Table[9](https://arxiv.org/html/2505.22202v2#A6.T9 "Table 9 ‣ Appendix F Evaluation Prompt ‣ Latent Reasoning via Sentence Embedding Prediction"), are lower than those reported in Table[3](https://arxiv.org/html/2505.22202v2#S3.T3 "Table 3 ‣ Q3: Can sentence-level reasoning reduce computational cost? ‣ 3.5 Results ‣ 3 Sentence-Level Reasoning Model ‣ Latent Reasoning via Sentence Embedding Prediction").

Appendix E Experiment Details
-----------------------------

Each dataset requires task-specific hyperparameter choices due to variation in problem structure and reasoning complexity. For all experiments, we report the best test-set accuracy across saved checkpoints (including baselines). When training all of our models (Latent Model, Encoder, and Decoder), we initialize them from the SFT checkpoint. The number of training epochs for each stage was selected based on convergence trends observed during early stage of experiments. Please note that we use small portion of Fineweb-Edu[[20](https://arxiv.org/html/2505.22202v2#bib.bib20)] for CSQA task’s restoration (i.e. training for semantic embeddings.) We report hyperparameters used in Table[10](https://arxiv.org/html/2505.22202v2#A6.T10 "Table 10 ‣ Appendix F Evaluation Prompt ‣ Latent Reasoning via Sentence Embedding Prediction") and[11](https://arxiv.org/html/2505.22202v2#A6.T11 "Table 11 ‣ Appendix F Evaluation Prompt ‣ Latent Reasoning via Sentence Embedding Prediction").

Appendix F Evaluation Prompt
----------------------------

Please refer to Figure[5](https://arxiv.org/html/2505.22202v2#A6.F5 "Figure 5 ‣ Appendix F Evaluation Prompt ‣ Latent Reasoning via Sentence Embedding Prediction").

Table 8: Dataset statistics for each reasoning benchmark across train, validation, and test splits.

Table 9: Average inference-time compute cost (GFLOPs) on each dataset under CoT and Ctx-C Continuous inference, with the accuracy of the trained classifier.

Stage GSM8K CSQA ProsQA Blocksworld
SFT*
Epochs 20 20 20 100
LR 1e-4 1e-4 1e-4 1e-4
Batch 64 64 64 64
Embedding: Restoration
Epochs 3 5 3 100
LR 5e-4 5e-4 5e-4 1e-4
Batch 256 512 128 1024(256*4)
Embedding: Prediction
Epochs 30 50 50 50
LR 5e-4 5e-4 5e-4 1e-4
Batch 128 128 96 64
Latent Autoreg.
Epochs 200 300 50 200
Eval Freq every 10 every 10 every 2 every 10
LR 5e-4 5e-4 5e-4 5e-4
Batch 128 128 32 64

Table 10: Training configurations of GPT-2 for each dataset and training stage. *SFT includes both CoT and No-CoT variants.

Stage GPT-2 Small GPT-2 Medium GPT-2 Large (LoRA)r=256 r{=}256, a=1024 a{=}1024)
SFT
Epochs 20 20 20
LR 1e-4 1e-4 1e-4
Batch 64 64 × 8 64 × 8
Embedding: Restoration
Epochs 5 5 5
LR 5e-4 5e-4 5e-4
Batch 512 128 128
Notes used FW subset used FW subset used FW subset
Embedding: Prediction
Epochs 50 50 50
LR 5e-4 5e-5 1e-4
Batch 128 128 64
Latent Autoreg.
Epochs 300 300 300
Eval Freq every 10 every 10 every 2
LR 5e-4 1e-4 1e-4
Batch 128 64 128
Notes——w. grad ckpting

Table 11: Training configurations by model size and stage. LoRA configuration used for GPT-2 Large.

![Image 6: Refer to caption](https://arxiv.org/html/2505.22202v2/figs/evaluation_prompt.png)

Figure 5: Evaluation Prompt used to GPT-4o for judging intermediate reasoning step’s quality.

![Image 7: Refer to caption](https://arxiv.org/html/2505.22202v2/x3.png)

Figure 6: Example instances from each dataset.
