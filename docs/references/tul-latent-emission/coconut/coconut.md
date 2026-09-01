Title: Training Large Language Models to Reason in a Continuous Latent Space

URL Source: https://arxiv.org/html/2412.06769

Markdown Content:
1]FAIR at Meta 2]UC San Diego \contribution[*]Work done at Meta

Sainbayar Sukhbaatar DiJia Su Xian Li Zhiting Hu Jason Weston Yuandong Tian [ [

###### Abstract

Large language models (LLMs) are restricted to reason in the “language space”, where they typically express the reasoning process with a chain-of-thought (CoT) to solve a complex reasoning problem. However, we argue that language space may not always be optimal for reasoning. For example, most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs. To explore the potential of LLM reasoning in an unrestricted latent space instead of using natural language, we introduce a new paradigm Coconut (C hain o f Con tin u ous T hought). We utilize the last hidden state of the LLM as a representation of the reasoning state (termed “continuous thought”). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space. This latent reasoning paradigm leads to the emergence of an advanced reasoning pattern: the continuous thought can encode multiple alternative next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT. Coconut outperforms CoT on certain logical reasoning tasks that require substantial search during planning, and shows a better trade-off between accuracy and efficiency.

1 Introduction
--------------

Large language models (LLMs) have demonstrated remarkable reasoning abilities, emerging from extensive pretraining on human languages (Dubey et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib8); Achiam et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib1)). While next token prediction is an effective training objective, it imposes a fundamental constraint on the LLM as a reasoning machine: the explicit reasoning process of LLMs must be generated in word tokens. For example, a prevalent approach, known as chain-of-thought (CoT) reasoning (Wei et al., [2022](https://arxiv.org/html/2412.06769v3#bib.bib42)), involves prompting or training LLMs to generate solutions step-by-step using natural language. However, this is in stark contrast to certain human cognition results. Neuroimaging studies have consistently shown that the language network – a set of brain regions responsible for language comprehension and production – remains largely inactive during various reasoning tasks (Amalric and Dehaene, [2019](https://arxiv.org/html/2412.06769v3#bib.bib2); Monti et al., [2012](https://arxiv.org/html/2412.06769v3#bib.bib30), [2007](https://arxiv.org/html/2412.06769v3#bib.bib28), [2009](https://arxiv.org/html/2412.06769v3#bib.bib29); Fedorenko et al., [2011](https://arxiv.org/html/2412.06769v3#bib.bib10)). Further evidence indicates that human language is optimized for communication rather than reasoning (Fedorenko et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib11)).

A significant issue arises when LLMs use language for reasoning: the amount of reasoning required for each particular token varies greatly, yet current LLM architectures allocate nearly the same computing budget for predicting every token. Most tokens in a reasoning chain are generated solely for fluency, contributing little to the actual reasoning process. By contrast, some critical tokens require complex planning and pose huge challenges to LLMs. While previous work has attempted to fix these problems by prompting LLMs to generate succinct reasoning chains (Madaan and Yazdanbakhsh, [2022](https://arxiv.org/html/2412.06769v3#bib.bib26)), or performing additional reasoning before generating some critical tokens (Zelikman et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib50)), these solutions remain constrained within the language space and do not solve the fundamental problems. On the contrary, it would be ideal for LLMs to have the freedom to reason without any language constraints, and then translate their findings into language only when necessary.

![Image 1: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/figure_1_meta_3.png)

Figure 1: A comparison of Chain of Continuous Thought (Coconut) with Chain-of-Thought (CoT). In CoT, the model generates the reasoning process as a word token sequence (e.g., [x i,x i+1,…,x i+j][x_{i},x_{i+1},...,x_{i+j}] in the figure). Coconut regards the last hidden state as a representation of the reasoning state (termed “continuous thought”), and directly uses it as the next input embedding. This allows the LLM to reason in an unrestricted latent space instead of a language space.

In this work we instead explore LLM reasoning in a latent space by introducing a novel paradigm, Coconut (Chain of Continuous Thought). It involves a simple modification to the traditional CoT process: instead of mapping between hidden states and language tokens using the language model head and embedding layer, Coconut directly feeds the last hidden state (a continuous thought) as the input embedding for the next token (Figure [1](https://arxiv.org/html/2412.06769v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Training Large Language Models to Reason in a Continuous Latent Space")). This modification frees the reasoning from being within the language space, and the system can be optimized end-to-end by gradient descent, as continuous thoughts are fully differentiable. To enhance the training of latent reasoning, we employ a multi-stage training strategy inspired by Deng et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib7)), which effectively utilizes language reasoning chains to guide the training process.

Interestingly, our proposed paradigm leads to an efficient reasoning pattern. Unlike language-based reasoning, continuous thoughts in Coconut can encode multiple potential next steps simultaneously, allowing for a reasoning process akin to breadth-first search (BFS). While the model may not initially make the correct decision, it can maintain many possible options within the continuous thoughts and progressively eliminate incorrect paths through reasoning, guided by some implicit value functions. This advanced reasoning mechanism surpasses traditional CoT, even though the model is not explicitly trained or instructed to operate in this manner, as seen in previous works (Yao et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib45); Hao et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib19)).

Experimentally, Coconut successfully enhances the reasoning capabilities of LLMs. For math reasoning (GSM8k, Cobbe et al., [2021](https://arxiv.org/html/2412.06769v3#bib.bib5)), using continuous thoughts is shown to be beneficial to reasoning accuracy, mirroring the effects of language reasoning chains. This indicates the potential to scale and solve increasingly challenging problems by chaining more continuous thoughts. On logical reasoning including ProntoQA (Saparov and He, [2022](https://arxiv.org/html/2412.06769v3#bib.bib33)), and our newly proposed ProsQA (Section [4](https://arxiv.org/html/2412.06769v3#S4 "4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")) which requires stronger planning ability, Coconut and some of its variants even surpasses language-based CoT methods, while generating significantly fewer tokens during inference. We believe that these findings underscore the potential of latent reasoning and could provide valuable insights for future research.

2 Related Work
--------------

Chain-of-thought (CoT) reasoning. We use the term chain-of-thought broadly to refer to methods that generate an intermediate reasoning process in language before outputting the final answer. This includes prompting LLMs (Wei et al., [2022](https://arxiv.org/html/2412.06769v3#bib.bib42); Khot et al., [2022](https://arxiv.org/html/2412.06769v3#bib.bib22); Zhou et al., [2022](https://arxiv.org/html/2412.06769v3#bib.bib51)), or training LLMs to generate reasoning chains, either with supervised finetuning (Yue et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib49); Yu et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib47)) or reinforcement learning (Wang et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib40); Havrilla et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib21); Shao et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib35); Yu et al., [2024a](https://arxiv.org/html/2412.06769v3#bib.bib46)). Madaan and Yazdanbakhsh ([2022](https://arxiv.org/html/2412.06769v3#bib.bib26)) classified the tokens in CoT into symbols, patterns, and text, and proposed to guide the LLM to generate concise CoT based on analysis of their roles. Recent theoretical analyses have demonstrated the usefulness of CoT from the perspective of model expressivity (Feng et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib12); Merrill and Sabharwal, [2023](https://arxiv.org/html/2412.06769v3#bib.bib27); Li et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib25)). By employing CoT, the effective depth of the transformer increases because the generated outputs are looped back to the input (Feng et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib12)). These analyses, combined with the established effectiveness of CoT, motivated our design of feeding the continuous thoughts back into the LLM as input embeddings. While CoT has proven effective for certain tasks, its autoregressive generation nature makes it challenging to mimic human reasoning on more complex problems (LeCun, [2022](https://arxiv.org/html/2412.06769v3#bib.bib23); Hao et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib19)), which typically require planning and search. There are works that equip LLMs with explicit tree search algorithms (Xie et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib43); Yao et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib45); Hao et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib20)), or train the LLM on search dynamics and trajectories (Lehnert et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib24); Gandhi et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib13); Su et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib37)). In our analysis, we find that after removing the constraint of a language space, a new reasoning pattern similar to BFS emerges, even though the model is not explicitly trained in this way.

Latent reasoning in LLMs. Previous works mostly define latent reasoning in LLMs as the hidden computation in transformers (Yang et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib44); Biran et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib4)). Yang et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib44)) constructed a dataset of two-hop reasoning problems and discovered that it is possible to recover the intermediate variable from the hidden representations. Biran et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib4)) further proposed to intervene the latent reasoning by “back-patching” the hidden representation. Shalev et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib34)) discovered parallel latent reasoning paths in LLMs. Another line of work has discovered that, even if the model generates a CoT to reason, the model may actually utilize a different latent reasoning process. This phenomenon is known as the unfaithfulness of CoT reasoning (Wang et al., [2022](https://arxiv.org/html/2412.06769v3#bib.bib39); Turpin et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib38)). To enhance the latent reasoning of LLMs, previous research proposed to augment it with additional tokens. Goyal et al. ([2023](https://arxiv.org/html/2412.06769v3#bib.bib17)) pretrained the model by randomly inserting a learnable <pause> tokens to the training corpus. This improves LLM’s performance on a variety of tasks, especially when followed by supervised finetuning with <pause> tokens. On the other hand, Pfau et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib31)) further explored the usage of filler tokens, e.g., “...”, and concluded that they work well for highly parallelizable problems. However, Pfau et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib31)) mentioned these methods do not extend the expressivity of the LLM like CoT; hence, they may not scale to more general and complex reasoning problems. Wang et al. ([2023](https://arxiv.org/html/2412.06769v3#bib.bib41)) proposed to predict a planning token as a discrete latent variable before generating the next reasoning step. Recently, it has also been found that one can “internalize” the CoT reasoning into latent reasoning in the transformer with knowledge distillation (Deng et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib6)) or a special training curriculum which gradually shortens CoT (Deng et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib7)). Yu et al. ([2024b](https://arxiv.org/html/2412.06769v3#bib.bib48)) also proposed to distill a model that can reason latently from data generated with complex reasoning algorithms. These training methods can be combined to our framework, and specifically, we find that breaking down the learning of continuous thoughts into multiple stages, inspired by iCoT (Deng et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib7)), is very beneficial for the training. Other work explores alternative architectures for latent reasoning, including looped transformers (Giannou et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib15); Fan et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib9)), diffusion models in sentence embedding space (Barrault et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib3)). Different from these works, we focus on general multi-step reasoning tasks and aim to investigate the unique properties of latent reasoning in comparison to language space. In addition to reasoning tasks, Pham et al. ([2023](https://arxiv.org/html/2412.06769v3#bib.bib32)) also explored using continuous space for multi-agent communication. Building on Coconut, Zhu et al. ([2025b](https://arxiv.org/html/2412.06769v3#bib.bib53)) developed a theoretical framework demonstrating that continuous CoT can be more efficient than discrete CoT on certain tasks by encoding multiple reasoning paths in superposition states. Subsequently, Zhu et al. ([2025a](https://arxiv.org/html/2412.06769v3#bib.bib52)) analyzed the training dynamics to explain how such superposition emerges under the Coconut training objective.

3 Coconut: Chain of Continuous Thought
--------------------------------------

In this section, we introduce our new paradigm Coconut (Chain of Continuous Thought) for reasoning in an unconstrained latent space. We begin by introducing the background and notation we use for language models. For an input sequence x=(x 1,…,x T)x=(x_{1},...,x_{T}), the standard large language model ℳ\mathcal{M} can be described as:

H t=Transformer​(E t)H_{t}=\text{Transformer}(E_{t})

ℳ​(x t+1∣x≤t)=softmax​(W​h t)\mathcal{M}(x_{t+1}\mid x_{\leq t})=\text{softmax}(Wh_{t})

where E t=[e​(x 1),e​(x 2),…,e​(x t)]E_{t}=[e(x_{1}),e(x_{2}),...,e(x_{t})] is the sequence of token embeddings up to position t t; H t∈ℝ t×d H_{t}\in\mathbb{R}^{t\times d} is the matrix of the last hidden states for all tokens up to position t t; h t h_{t} is the last hidden state of position t t, i.e., h t=H t​[t,:]h_{t}=H_{t}[t,:]; e​(⋅)e(\cdot) is the token embedding function; W W is the parameter of the language model head.

Method Overview. In the proposed Coconut method, the LLM switches between the “language mode” and “latent mode” (Figure [1](https://arxiv.org/html/2412.06769v3#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Training Large Language Models to Reason in a Continuous Latent Space")). In language mode, the model operates as a standard language model, autoregressively generating the next token. In latent mode, it directly utilizes the last hidden state as the next input embedding. This last hidden state represents the current reasoning state, termed as a “continuous thought”.

Special tokens <bot> and <eot> are employed to mark the beginning and end of the latent thought mode, respectively. As an example, we assume latent reasoning occurs between positions i i and j j, i.e., x i=x_{i}=<bot> and x j=x_{j}=<eot>. When the model is in the latent mode (i<t<j i<t<j), we use the last hidden state from the previous token to replace the input embedding, i.e., E t=[e​(x 1),e​(x 2),…,e​(x i),h i,h i+1,…,h t−1]E_{t}=[e(x_{1}),e(x_{2}),...,e(x_{i}),h_{i},h_{i+1},...,h_{t-1}]. After the latent mode finishes (t≥j t\geq j), the input reverts to using the token embedding, i.e., E t=[e​(x 1),e​(x 2),…,e​(x i),h i,h i+1,…,h j−1,e​(x j),…,e​(x t)]E_{t}=[e(x_{1}),e(x_{2}),...,e(x_{i}),h_{i},h_{i+1},...,h_{j-1},e(x_{j}),...,e(x_{t})]. It is worth noting that the last hidden states have been processed by the final normalization layer, so they are not too large in magnitude. ℳ​(x t+1∣x≤t)\mathcal{M}(x_{t+1}\mid x_{\leq t}) is not defined when i<t<j i<t<j, since the latent thought is not intended to be mapped back to language space. However, softmax​(W​h t)\mathrm{softmax}(Wh_{t}) can still be calculated for probing purposes (see Section [5](https://arxiv.org/html/2412.06769v3#S5 "5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space")).

Training Procedure. In this work, we focus on a problem-solving setting where the model receives a question as input and is expected to generate an answer through a reasoning process. We leverage language CoT data to supervise continuous thought by implementing a multi-stage training curriculum inspired by Deng et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib7)). As shown in Figure [2](https://arxiv.org/html/2412.06769v3#S3.F2 "Figure 2 ‣ 3 Coconut: Chain of Continuous Thought ‣ Training Large Language Models to Reason in a Continuous Latent Space"), in the initial stage, the model is trained on regular CoT instances. In the subsequent stages, at the k k-th stage, the first k k reasoning steps in the CoT are replaced with k×c k\times c continuous thoughts 1 1 1 If a language reasoning chain is shorter than k k steps, then all the language thoughts will be removed., where c c is a hyperparameter controlling the number of latent thoughts replacing a single language reasoning step. Following Deng et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib7)), we also reset the optimizer state when training stages switch. We insert <bot> and <eot> tokens (which are not counted towards c c) to encapsulate the continuous thoughts.

![Image 2: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/training_stage_figure.png)

Figure 2: Training procedure of Chain of Continuous Thought (Coconut). Given training data with language reasoning steps, at each training stage we integrate c c additional continuous thoughts (c=1 c=1 in this example), and remove one language reasoning step. The cross-entropy loss is then used on the remaining tokens after continuous thoughts.

During the training process, we optimize the normal negative log-likelihood loss, but mask the loss on questions and latent thoughts. It is important to note that the objective does not encourage the continuous thought to compress the removed language thought, but rather to facilitate the prediction of future reasoning. Therefore, it’s possible for the LLM to learn more effective representations of reasoning steps compared to human language.

Training Details. Our proposed continuous thoughts are fully differentiable and allow for back-propagation. We perform n+1 n+1 forward passes when n n latent thoughts are scheduled in the current training stage, computing a new latent thought with each pass and finally conducting an additional forward pass to obtain a loss on the remaining text sequence. While we can save any repetitive computing by using a KV cache, the sequential nature of the multiple forward passes poses challenges for parallelism. Further optimizing the training efficiency of Coconut remains an important direction for future research.

Inference Process. The inference process for Coconut is analogous to standard language model decoding, except that in latent mode, we directly feed the last hidden state as the next input embedding. A challenge lies in determining when to switch between latent and language modes. As we focus on the problem-solving setting, we insert a <bot> token immediately following the question tokens. For <eot>, we consider two potential strategies: a) train a binary classifier on latent thoughts to enable the model to autonomously decide when to terminate the latent reasoning, or b) always pad the latent thoughts to a constant length. We found that both approaches work comparably well. Therefore, we use the second option in our experiment for simplicity, unless specified otherwise.

4 Continuous Space Enables Latent Tree Search
---------------------------------------------

In this section, we provide a proof of concept of the advantage of continuous latent space reasoning. On ProsQA, a new dataset that requires extensive planning ability, Coconut outperforms language space CoT reasoning. Interestingly, our analysis indicates that the continuous representation of reasoning can encode multiple alternative next reasoning steps. This allows the model to perform a breadth-first search (BFS) to solve the problem, instead of prematurely committing to a single deterministic path like language CoT.

We start by introducing the experimental setup (Section [4.1](https://arxiv.org/html/2412.06769v3#S4.SS1 "4.1 Experimental Setup ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")). By leveraging Coconut’s ability to switch between language and latent space reasoning, we are able to control the model to interpolate between fully latent reasoning and fully language reasoning and test their performance (Section [4.2](https://arxiv.org/html/2412.06769v3#S4.SS2 "4.2 Overall Results ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")). This also enables us to interpret the latent reasoning process as tree search (Section [4.3](https://arxiv.org/html/2412.06769v3#S4.SS3 "4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")). Based on this perspective, we explain why latent reasoning can help LLMs make better decisions (Section [4.4](https://arxiv.org/html/2412.06769v3#S4.SS4 "4.4 Why is Latent Space Better for Planning? ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")).

### 4.1 Experimental Setup

Dataset. We introduce ProsQA (Pro of with S earch Q uestion-A nswering), a new logical reasoning dataset. A visualized example is shown in Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"). Each instance in ProsQA consists of a directed acyclic graph (DAG) of logical relationships between concepts, presented as natural language statements. The task requires models to determine logical relationships by finding valid paths through this graph, demanding sophisticated planning and search strategies. Unlike previous logical reasoning datasets like ProntoQA (Saparov and He, [2022](https://arxiv.org/html/2412.06769v3#bib.bib33)), ProsQA’s DAG structure introduces complex exploration paths, making it particularly challenging for models to identify the correct reasoning chain. More comprehensive details about the dataset construction and characteristics can be found in Appendix [7](https://arxiv.org/html/2412.06769v3#S7 "7 Datasets ‣ Training Large Language Models to Reason in a Continuous Latent Space").

Setup. We use a pre-trained GPT-2 model as the base model for all experiments. The learning rate is set to 1×10−4 1\times 10^{-4} while the effective batch size is 128. We train a Coconut model following the training procedure in Section [3](https://arxiv.org/html/2412.06769v3#S3 "3 Coconut: Chain of Continuous Thought ‣ Training Large Language Models to Reason in a Continuous Latent Space"). Since the maximum reasoning steps in ProsQA is 6, we set the number of training stages to N=6 N=6 in the training procedure. In each stage, we train the model for 5 epochs, and stay in the last stage until the 50 epochs. The checkpoint with the best accuracy in the last stage is used for evaluation. As reference, we report the performance of (1) CoT: the model is trained with CoT data, and during inference, the model will generate a complete reasoning chain to solve the problem. (2) no-CoT: the model is trained with only the question and answer pairs, without any reasoning steps. During inference, the model will output the final answer directly.

To understand the properties of latent and language reasoning space, we manipulate the model to switch between fully latent reasoning and fully language reasoning, by manually setting the position of the <eot> token during inference. When we enforce Coconut to use k k continuous thoughts, the model is expected to output the remaining reasoning chain in language, starting from the k+1 k+1 step. In our experiments, we test variants of Coconut on ProsQA with k∈{0,1,2,3,4,5,6}k\in\{0,1,2,3,4,5,6\}. Note that all these variants only differ in inference time while sharing the same model weights.

Metrics. We apply two sets of evaluation metrics. One of them is based on the correctness of the final answer, regardless of the reasoning process. It is also the main metric used in the later sections (Section [5.3](https://arxiv.org/html/2412.06769v3#S5.SS3 "5.3 Results and Discussion ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space")). To enable fine-grained analysis on ProsQA, we define another metric on the reasoning process. We classify a reasoning chain into (1) Correct Path: The output is one of the shortest paths to the correct answer. (2) Longer Path: A valid path that correctly answers the question but is longer than the shortest path. (3) Hallucination: The path includes nonexistent edges or is disconnected. (4) Wrong Target: A valid path in the graph, but the destination node is not the one being asked. These four categories naturally apply to the output from Coconut (k=0 k=0) and CoT, which generate the full path. For Coconut with k>0 k>0 that outputs only partial paths in language (with the initial steps in continuous reasoning), we classify the reasoning as a Correct Path if a valid explanation can complete it. Also, we define Longer Path and Wrong Target for partial paths similarly. If no valid explanation completes the path, it’s classified as Hallucination. In no-CoT and Coconut with larger k k, the model may only output the final answer without any partial path, and it falls into (5) Correct Label or (6) Incorrect Label. These six categories cover all cases without overlap.

### 4.2 Overall Results

![Image 3: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/figure_5_revised_1111.png)

Figure 3: The accuracy of final answer (left) and reasoning process (right) of multiple variants of Coconut and baselines on ProsQA.

Figure [3](https://arxiv.org/html/2412.06769v3#S4.F3 "Figure 3 ‣ 4.2 Overall Results ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space") presents a comparative analysis of various reasoning methods evaluated on ProsQA. The model trained using CoT frequently hallucinates non-existent edges or outputs paths leading to incorrect targets, resulting in lower answer accuracy. In contrast, Coconut, which leverages continuous space reasoning, demonstrates improved accuracy as it utilizes an increasing number of continuous thoughts. Additionally, the rate of correct reasoning processes (indicated by “Correct Label” and “Correct Path”) significantly increases. At the same time, there is a notable reduction in instances of “Hallucination” and “Wrong Target,” issues that typically emerge when the model makes mistakes early in the reasoning process.

An intuitive demonstration of the limitations of reasoning in language space is provided by the case study depicted in Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"). As shown, models operating in language space often fail to plan ahead or backtrack. Once they commit to an incorrect path, they either hallucinate unsupported edges or terminate with irrelevant conclusions. In contrast, latent reasoning avoids such premature commitments by enabling the model to iteratively refine its decisions across multiple reasoning steps. This flexibility allows the model to progressively eliminate incorrect options and converge on the correct answer, ultimately resulting in higher accuracy.

### 4.3 Interpreting the Latent Reasoning as Tree Search

To better understand Coconut, we probe the latent reasoning process by forcing the model to explicitly generate language reasoning steps following intermediate continuous thoughts (Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")). Using the example presented in Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"), at the initial reasoning step, the model must select which immediate child node of “Alex” to consider next, specifically from the set {“lempus”, “sterpus”, “zhorpus”, “grimpus”}. The distribution over these candidate next steps is visualized in Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"), left. In the subsequent reasoning step, these nodes expand further into an extended set of potential paths, including all grandchildren of “Alex” (Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"), right).

![Image 4: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/new_figure_6.png)

Figure 4: A case study of ProsQA. The model trained with CoT hallucinates an edge (Every yumpus is a rempus) after getting stuck in a dead end. Coconut (k=1) outputs a path that ends with an irrelevant node. Coconut (k=2) solves the problem correctly. 

![Image 5: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/new_figure_7.png)

Figure 5: An illustration of the latent search trees. The example is the same test case as in Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"). The height of a node (denoted as h h in the figure) is defined as the longest distance to any leaf nodes in the graph. We show the probability of the first concept predicted by the model following latent thoughts (e.g., “lempus” in the left figure). It is calculated as the multiplication of the probability of all tokens within the concept conditioned on previous context (omitted in the figure for brevity). This metric can be interpreted as an implicit value function estimated by the model, assessing the potential of each node leading to the correct answer.

We define the predicted probability of a concept following continuous thoughts as a value function (Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")), estimating each node’s potential for reaching the correct target. Interestingly, the reasoning strategy employed by Coconut is not greedy search: while “lempus” initially has the highest value (0.33) at the first reasoning step (Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"), left), the model subsequently assigns the highest value (0.87) to “rorpus,” a child of “grimpus,” rather than following “lempus” (Figure [5](https://arxiv.org/html/2412.06769v3#S4.F5 "Figure 5 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"), right). This characteristic resembles a breadth-first search (BFS) approach, contrasting sharply with the greedy decoding typical of traditional CoT methods. The inherent capability of continuous representations to encode multiple candidate paths enables the model to avoid making immediate deterministic decisions. Importantly, this tree search pattern is not limited to the illustrated example, but constitutes a fundamental mechanism underlying the consistent improvement observed with larger values of k k in Coconut.

Figure [6](https://arxiv.org/html/2412.06769v3#S4.F6 "Figure 6 ‣ 4.3 Interpreting the Latent Reasoning as Tree Search ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space") presents an analysis of the parallelism in the model’s latent reasoning across the first and second thoughts. For the first thoughts (left panel), the cumulative values of the top-1, top-2, and top-3 candidate nodes are computed and plotted against their respective percentiles across the test set. The noticeable gaps between the three lines indicate that the model maintains significant diversity in its reasoning paths at this stage, suggesting a broad exploration of alternative possibilities. In contrast, the second thoughts (right panel) show a narrowing of these gaps. This trend suggests that the model transitions from parallel exploration to more focused reasoning in the second latent reasoning step, likely as it gains more certainty about the most promising paths.

![Image 6: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/percentile.png)

Figure 6: Analysis of parallelism in the first two steps of the latent tree search. The three curves in each panel depict the cumulative value of the top-1, top-2, and top-3 candidate nodes.

### 4.4 Why is Latent Space Better for Planning?

Building upon the tree search perspective, we further examine why latent reasoning benefits planning tasks—specifically, why maintaining multiple candidate paths and postponing deterministic decisions enhances reasoning performance. Our hypothesis is that nodes explored in the early reasoning stages are inherently more challenging to evaluate accurately because they are farther from the final target nodes. In contrast, nodes positioned closer to potential targets, having fewer subsequent exploration possibilities, can be assessed accurately with higher confidence.

![Image 7: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/value_stats_meta_2.png)

Figure 7: The correlation between the predicted value of correct/incorrect nodes and their heights.

To systematically test this, we define the height of a node as its shortest distance to any leaf node and analyze the relationship between node height and the model’s estimated value. Ideally, a correct node—one that can lead to the target node—should receive a high estimated value, whereas an incorrect node—one that cannot lead to the target node—should receive a low value. Empirical results across the test set (Figure [7](https://arxiv.org/html/2412.06769v3#S4.F7 "Figure 7 ‣ 4.4 Why is Latent Space Better for Planning? ‣ 4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space")) support our hypothesis: nodes with lower heights consistently receive more accurate and definitive probability evaluations. Conversely, nodes with greater heights exhibit more ambiguous evaluations, reflecting increased uncertainty.

These findings underscore the advantage of latent space reasoning. By delaying deterministic decisions and allowing exploration to proceed toward terminal states, latent reasoning significantly enhances the model’s ability to differentiate correct paths from incorrect ones, thereby improving performance on complex, planning-intensive tasks compared to traditional greedy methods.

5 Empirical Results with Coconut
--------------------------------

After analyzing the promising parallel search pattern of Coconut, we validate the feasibility of LLM reasoning in a continuous latent space through more comprehensive experiments, highlighting its better reasoning efficiency over language space, as well as its potential to enhance the model’s expressivity with test-time scaling.

Table 1: Results on three datasets: GSM8k, ProntoQA and ProsQA. Higher accuracy indicates stronger reasoning ability, while generating fewer tokens indicates better efficiency. ∗The result is from Deng et al. ([2024](https://arxiv.org/html/2412.06769v3#bib.bib7)).

### 5.1 Experimental Setup

Math Reasoning. We use GSM8k (Cobbe et al., [2021](https://arxiv.org/html/2412.06769v3#bib.bib5)) as the dataset for math reasoning. It consists of grade school-level math problems. To train the model, we use a synthetic dataset generated by Deng et al. ([2023](https://arxiv.org/html/2412.06769v3#bib.bib6)). We use two continuous thoughts for each reasoning step (i.e., c=2 c=2). The model goes through 3 stages besides the initial stage. We then include an additional stage where still 3×c 3\times c continuous thoughts are used as in the previous stage, but with all the remaining language reasoning chain removed. This handles the long-tail distribution of reasoning chains longer than 3 steps. We train the model for 6 epochs in the initial stage, and 3 epochs in each remaining stage.

Logical Reasoning. Logical reasoning involves the proper application of known conditions to prove or disprove a conclusion using logical rules. We use the ProntoQA (Saparov and He, [2022](https://arxiv.org/html/2412.06769v3#bib.bib33)) dataset, and our newly proposed ProsQA dataset, which is more challenging due to more distracting branches. We use one continuous thought for every reasoning step (i.e., c=1 c=1). The model goes through 6 training stages in addition to the initial stage, because the maximum number of reasoning steps is 6 in these two datasets. The model then fully reasons with continuous thoughts to solve the problems in the last stage. We train the model for 5 epochs per stage.

For all datasets, after the standard schedule, the model stays in the final training stage, until reaching 50 epochs. We select the checkpoint based on the accuracy on the validation set. For inference, we manually set the number of continuous thoughts to be consistent with their final training stage. We use greedy decoding for all experiments.

### 5.2 Baselines and Variants of Coconut

We consider the following baselines: (1) CoT, and (2) No-CoT, which were introduced in Section [4](https://arxiv.org/html/2412.06769v3#S4 "4 Continuous Space Enables Latent Tree Search ‣ Training Large Language Models to Reason in a Continuous Latent Space"). (3) iCoT(Deng et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib7)): The model is trained with language reasoning chains and follows a carefully designed schedule that “internalizes” CoT. As the training goes on, tokens at the beginning of the reasoning chain are gradually removed until only the answer remains. During inference, the model directly predicts the answer. (4) Pause token(Goyal et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib17)): The model is trained using only the question and answer without a reasoning chain. However, different from No-CoT, special <pause> tokens are inserted between the question and answer, which provides the model with additional computational capacity to derive the answer. The number of <pause> tokens is set the same as continuous thoughts in Coconut.

We also evaluate some variants of Coconut: (1) w/o curriculum, which directly trains the model in the last stage. The model uses continuous thoughts to solve the whole problem. (2) w/o thought: We keep the multi-stage training, but don’t add any continuous latent thoughts. While this is similar to iCoT in the high-level idea, the exact training schedule is set to be consistent with Coconut, instead of iCoT, for a strict comparison. (3) Pause as thought: We use special <pause> tokens to replace the continuous thoughts, and apply the same multi-stage training curriculum as Coconut.

### 5.3 Results and Discussion

We show the overall results in Table [1](https://arxiv.org/html/2412.06769v3#S5.T1 "Table 1 ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space"). Using continuous thoughts effectively enhances LLM reasoning over the No-CoT baseline. For example, by using 6 continuous thoughts, Coconut achieves 34.1% accuracy on GSM8k, which significantly outperforms No-CoT (16.5%). We highlight several key findings below.

![Image 8: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/merged_efficiency.png)

Figure 8: Efficiency comparison of reasoning space and Coconut with different c c.

“Chaining” continuous thoughts enhances reasoning. Language CoT proves to increase the effective depth of LLMs and enhance their expressiveness (Feng et al., [2023](https://arxiv.org/html/2412.06769v3#bib.bib12)). Thus, generating more tokens serves as a way to inference-time scaling for reasoning (Guo et al., [2025](https://arxiv.org/html/2412.06769v3#bib.bib18); Snell et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib36)). This desirable property holds naturally for Coconut too. On GSM8k, Coconut outperformed other architectures trained with similar strategies, including Coconut (pause as thought) and Coconut (w/o thought). Particularly, it surpasses the latest baseline iCoT(Deng et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib7)), which requires a more carefully designed training schedule.

Additionally, we experimented with adjusting the hyperparameter c c, which controls the number of latent thoughts corresponding to one language reasoning step (Figure [8](https://arxiv.org/html/2412.06769v3#S5.F8 "Figure 8 ‣ 5.3 Results and Discussion ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space"), II). As we increased c c from 0 to 1 to 2, the model’s performance steadily improved.2 2 2 We discuss the case of larger c c in Appendix [9.1](https://arxiv.org/html/2412.06769v3#S9.SS1 "9.1 Using More Continuous Thoughts ‣ 9 More Discussion ‣ Training Large Language Models to Reason in a Continuous Latent Space"). This further validates the potential of continuous thoughts to scale up to harder problems. In two other synthetic tasks, we found that the variants of Coconut (w/o thoughts or pause as thought), and the iCoT baseline also achieve impressive accuracy. This indicates that the model’s computational capacity may not be the bottleneck in these tasks. In contrast, GSM8k involves more complex contextual understanding and modeling, placing higher demands on computational capability.

![Image 9: Refer to caption](https://arxiv.org/html/2412.06769v3/figures/figure_4_meta.png)

Figure 9: Decoding a continuous thought into language tokens in a math word problem. The decoded tokens correspond to intermediate variables that help solve the problem.

Continuous thoughts are efficient representations of reasoning. Compared to traditional CoT, Coconut generates fewer tokens while achieving higher accuracy on ProntoQA and ProsQA (Table [1](https://arxiv.org/html/2412.06769v3#S5.T1 "Table 1 ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space")). Although Coconut does not surpass CoT on GSM8k, it offers a superior trade-off between reasoning efficiency and accuracy (Figure [8](https://arxiv.org/html/2412.06769v3#S5.F8 "Figure 8 ‣ 5.3 Results and Discussion ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space"), I). To illustrate this, we train a series of CoT models that progressively “internalize” (Deng et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib7)) the initial m={0,1,2,3,ALL}m=\{0,1,2,3,\text{ALL}\} reasoning steps, and plot their accuracy versus the number of generated tokens (labeled as “language” in the figure). These models quickly lose accuracy as more reasoning steps are skipped. In contrast, by applying Coconut training strategy—replacing each language reasoning step with two continuous thoughts—the accuracy drop is substantially mitigated, maintaining higher performance even when fewer tokens are generated. Another interesting observation is that, when we decode the first continuous thought, it often corresponds to possible intermediate variables in the calculation (Figure [9](https://arxiv.org/html/2412.06769v3#S5.F9 "Figure 9 ‣ 5.3 Results and Discussion ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space")). This also suggests that the continuous thoughts are more efficient representations of reasoning.

The LLM still needs guidance to learn latent reasoning. In the ideal case, the model should learn the most effective continuous thoughts automatically through gradient descent on questions and answers (i.e., Coconut w/o curriculum). However, from the experimental results, we found the models trained this way do not perform any better than no-CoT.

On the contrary, with the multi-stage curriculum, Coconut is able to achieve top performance across various tasks. The multi-stage training also integrates well with pause tokens (Coconut- pause as thought). Despite using the same architecture and similar multi-stage training objectives, we observed a small gap between the performance of iCoT and Coconut (w/o thoughts). The finer-grained removal schedule (token by token) and a few other tricks in iCoT may ease the training process. We leave combining iCoT and Coconut as future work. While the multi-stage training used for Coconut has proven effective, further research is definitely needed to develop better and more general strategies for learning reasoning in latent space, especially without the supervision from language reasoning chains.

6 Conclusion
------------

In this paper, we introduce Coconut, a new paradigm for reasoning in continuous latent space. Experiments demonstrate that Coconut effectively enhances LLM performance across a variety of reasoning tasks. Reasoning in latent space gives rise to advanced emergent behaviors, where continuous thoughts can represent multiple alternative next steps. This enables the model to perform BFS over possible reasoning paths, rather than prematurely committing to a single deterministic trajectory as in language space CoT reasoning. Further research is needed to refine and scale latent reasoning to pretraining, which could improve generalization across a broader range of reasoning challenges. We hope our findings will spark continued exploration into latent reasoning, ultimately advancing the development of more capable machine reasoning systems.

Acknowledgement

The authors express their sincere gratitude to Jihoon Tack for his valuable discussions throughout the course of this work.

References
----------

*   Achiam et al. (2023) Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_, 2023. 
*   Amalric and Dehaene (2019) Marie Amalric and Stanislas Dehaene. A distinct cortical network for mathematical knowledge in the human brain. _NeuroImage_, 189:19–31, 2019. 
*   Barrault et al. (2024) Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R Costa-jussà, David Dale, et al. Large concept models: Language modeling in a sentence representation space. _arXiv preprint arXiv:2412.08821_, 2024. 
*   Biran et al. (2024) Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, and Amir Globerson. Hopping too late: Exploring the limitations of large language models on multi-hop queries. _arXiv preprint arXiv:2406.12775_, 2024. 
*   Cobbe et al. (2021) Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021. 
*   Deng et al. (2023) Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. _arXiv preprint arXiv:2311.01460_, 2023. 
*   Deng et al. (2024) Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step. _arXiv preprint arXiv:2405.14838_, 2024. 
*   Dubey et al. (2024) Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_, 2024. 
*   Fan et al. (2024) Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped transformers for length generalization. _arXiv preprint arXiv:2409.15647_, 2024. 
*   Fedorenko et al. (2011) Evelina Fedorenko, Michael K Behr, and Nancy Kanwisher. Functional specificity for high-level linguistic processing in the human brain. _Proceedings of the National Academy of Sciences_, 108(39):16428–16433, 2011. 
*   Fedorenko et al. (2024) Evelina Fedorenko, Steven T Piantadosi, and Edward AF Gibson. Language is primarily a tool for communication rather than thought. _Nature_, 630(8017):575–586, 2024. 
*   Feng et al. (2023) Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. _Advances in Neural Information Processing Systems_, 36, 2023. 
*   Gandhi et al. (2024) Kanishk Gandhi, Denise Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, and Noah D Goodman. Stream of search (sos): Learning to search in language. _arXiv preprint arXiv:2404.03683_, 2024. 
*   Geiping et al. (2025) Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. _arXiv preprint arXiv:2502.05171_, 2025. 
*   Giannou et al. (2023) Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In _International Conference on Machine Learning_, pages 11398–11442. PMLR, 2023. 
*   Gladstone et al. (2025) Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, and Tariq Iqbal. Energy-based transformers are scalable learners and thinkers. _arXiv preprint arXiv:2507.02092_, 2025. 
*   Goyal et al. (2023) Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. _arXiv preprint arXiv:2310.02226_, 2023. 
*   Guo et al. (2025) Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 
*   Hao et al. (2023) Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. _arXiv preprint arXiv:2305.14992_, 2023. 
*   Hao et al. (2024) Shibo Hao, Yi Gu, Haotian Luo, Tianyang Liu, Xiyan Shao, Xinyuan Wang, Shuhua Xie, Haodi Ma, Adithya Samavedhi, Qiyue Gao, et al. Llm reasoners: New evaluation, library, and analysis of step-by-step reasoning with large language models. _arXiv preprint arXiv:2404.05221_, 2024. 
*   Havrilla et al. (2024) Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large language models to reason with reinforcement learning. _arXiv preprint arXiv:2403.04642_, 2024. 
*   Khot et al. (2022) Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. _arXiv preprint arXiv:2210.02406_, 2022. 
*   LeCun (2022) Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. _Open Review_, 62(1):1–62, 2022. 
*   Lehnert et al. (2024) Lucas Lehnert, Sainbayar Sukhbaatar, Paul Mcvay, Michael Rabbat, and Yuandong Tian. Beyond a*: Better planning with transformers via search dynamics bootstrapping. _arXiv preprint arXiv:2402.14083_, 2024. 
*   Li et al. (2024) Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. _arXiv preprint arXiv:2402.12875_, 2024. 
*   Madaan and Yazdanbakhsh (2022) Aman Madaan and Amir Yazdanbakhsh. Text and patterns: For effective chain of thought, it takes two to tango. _arXiv preprint arXiv:2209.07686_, 2022. 
*   Merrill and Sabharwal (2023) William Merrill and Ashish Sabharwal. The expresssive power of transformers with chain of thought. _arXiv preprint arXiv:2310.07923_, 2023. 
*   Monti et al. (2007) Martin M Monti, Daniel N Osherson, Michael J Martinez, and Lawrence M Parsons. Functional neuroanatomy of deductive inference: a language-independent distributed network. _Neuroimage_, 37(3):1005–1016, 2007. 
*   Monti et al. (2009) Martin M Monti, Lawrence M Parsons, and Daniel N Osherson. The boundaries of language and thought in deductive inference. _Proceedings of the National Academy of Sciences_, 106(30):12554–12559, 2009. 
*   Monti et al. (2012) Martin M Monti, Lawrence M Parsons, and Daniel N Osherson. Thought beyond language: neural dissociation of algebra and natural language. _Psychological science_, 23(8):914–922, 2012. 
*   Pfau et al. (2024) Jacob Pfau, William Merrill, and Samuel R Bowman. Let’s think dot by dot: Hidden computation in transformer language models. _arXiv preprint arXiv:2404.15758_, 2024. 
*   Pham et al. (2023) Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A Plummer, Zhaoran Wang, and Hongxia Yang. Let models speak ciphers: Multiagent debate through embeddings. _arXiv preprint arXiv:2310.06272_, 2023. 
*   Saparov and He (2022) Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. _arXiv preprint arXiv:2210.01240_, 2022. 
*   Shalev et al. (2024) Yuval Shalev, Amir Feder, and Ariel Goldstein. Distributional reasoning in llms: Parallel reasoning processes in multi-hop reasoning. _arXiv preprint arXiv:2406.13858_, 2024. 
*   Shao et al. (2024) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_, 2024. 
*   Snell et al. (2024) Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. _arXiv preprint arXiv:2408.03314_, 2024. 
*   Su et al. (2024) DiJia Su, Sainbayar Sukhbaatar, Michael Rabbat, Yuandong Tian, and Qinqing Zheng. Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces. _arXiv preprint arXiv:2410.09918_, 2024. 
*   Turpin et al. (2024) Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always say what they think: unfaithful explanations in chain-of-thought prompting. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Wang et al. (2022) Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters. _arXiv preprint arXiv:2212.10001_, 2022. 
*   Wang et al. (2024) Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 9426–9439, 2024. 
*   Wang et al. (2023) Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, William Yang Wang, and Alessandro Sordoni. Guiding language model reasoning with planning tokens. _arXiv preprint arXiv:2310.05707_, 2023. 
*   Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_, 35:24824–24837, 2022. 
*   Xie et al. (2023) Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. Self-evaluation guided beam search for reasoning. _Advances in Neural Information Processing Systems_, 36, 2023. 
*   Yang et al. (2024) Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, and Sebastian Riedel. Do large language models latently perform multi-hop reasoning? _arXiv preprint arXiv:2402.16837_, 2024. 
*   Yao et al. (2023) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_, 36, 2023. 
*   Yu et al. (2024a) Fangxu Yu, Lai Jiang, Haoqiang Kang, Shibo Hao, and Lianhui Qin. Flow of reasoning: Efficient training of llm policy with divergent thinking. _arXiv preprint arXiv:2406.05673_, 2024a. 
*   Yu et al. (2023) Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_, 2023. 
*   Yu et al. (2024b) Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling system 2 into system 1. _arXiv preprint arXiv:2407.06023_, 2024b. 
*   Yue et al. (2023) Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_, 2023. 
*   Zelikman et al. (2024) Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D Goodman. Quiet-star: Language models can teach themselves to think before speaking. _arXiv preprint arXiv:2403.09629_, 2024. 
*   Zhou et al. (2022) Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. _arXiv preprint arXiv:2205.10625_, 2022. 
*   Zhu et al. (2025a) Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Emergence of superposition: Unveiling the training dynamics of chain of continuous thought. _arXiv preprint arXiv:2509.23365_, 2025a. 
*   Zhu et al. (2025b) Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition: A theoretical perspective on chain of continuous thought. _arXiv preprint arXiv:2505.12514_, 2025b. 

\beginappendix

7 Datasets
----------

### 7.1 Examples

We provide some examples of the questions and CoT solutions for the datasets used in our experiments.

### 7.2 Construction of ProsQA

To construct the dataset, we first compile a set of typical entity names, such as “Alex” and “Jack,” along with fictional concept names like “lorpus” and “rorpus,” following the setting of ProntoQA (Saparov and He, [2022](https://arxiv.org/html/2412.06769v3#bib.bib33)). Each problem is structured as a binary question: “Is [Entity] a [Concept A] or [Concept B]?” Assuming [Concept A] is the correct answer, we build a directed acyclic graph (DAG) where each node represents an entity or a concept. The graph is constructed such that a path exists from [Entity] to [Concept A] but not to [Concept B].

Algorithm [1](https://arxiv.org/html/2412.06769v3#alg1 "Algorithm 1 ‣ 7.2 Construction of ProsQA ‣ 7 Datasets ‣ Training Large Language Models to Reason in a Continuous Latent Space") describes the graph construction process. The DAG is incrementally built by adding nodes and randomly connecting them with edges. To preserve the validity of the binary choice, with some probability, we enforce that the new node cannot simultaneously serve as a descendant to both node 0 and 1 1. This separation maintains distinct families of nodes and balances their sizes to prevent model shortcuts.

After the graph is constructed, nodes without parents are assigned entity names, while other nodes receive concept names. To formulate a question of the form “Is [Entity] a [Concept A] or [Concept B]?”, we designate node 0 in the graph as [Entity], a leaf node labeled 1 1 as [Concept A], and a leaf node labeled 2 2 as [Concept B]. This setup ensures a path from [Entity] to [Concept A] without any connection to [Concept B], introducing a moderately complex reasoning path. Finally, to avoid positional biases, [Concept A] and [Concept B] are randomly permuted in each question.

Algorithm 1 Graph Construction for ProsQA

e​d​g​e​s←{}edges\leftarrow\{\}

n​o​d​e​s←{0,1}nodes\leftarrow\{0,1\}

l​a​b​e​l​s←{0:1,1:2}labels\leftarrow\{0:1,1:2\}

⊳\triangleright Labels: 1 (descendant of node 0), 2 (descendant of node 1), 3 (both), 0 (neither).

g​r​o​u​p​s←{0:{},1:{0},2:{1},3:{}}groups\leftarrow\{0:\{\},1:\{0\},2:\{1\},3:\{\}\}

i​d​x←2 idx\leftarrow 2

while

i​d​x<N idx<N
do⊳\triangleright For each new node, randomly add edges from existing nodes

n​_​i​n​_​n​o​d​e​s←poisson​(1.5)n\_in\_nodes\leftarrow\text{poisson}(1.5)

r​a​n​d←random​()rand\leftarrow\text{random}()

if

r​a​n​d≤0.35 rand\leq 0.35
then

c​a​n​d​i​d​a​t​e​s←g​r​o​u​p​s​[0]∪g​r​o​u​p​s​[1]candidates\leftarrow groups[0]\cup groups[1]
⊳\triangleright Cannot be a descendant of node 1.

else if

r​a​n​d≤0.7 rand\leq 0.7
then

c​a​n​d​i​d​a​t​e​s←g​r​o​u​p​s​[0]∪g​r​o​u​p​s​[2]candidates\leftarrow groups[0]\cup groups[2]
⊳\triangleright Cannot be a descendant of node 0.

else

c​a​n​d​i​d​a​t​e​s←n​o​d​e​s candidates\leftarrow nodes

end if

n​_​i​n​_​n​o​d​e​s←min⁡(len​(c​a​n​d​i​d​a​t​e​s),n​_​i​n​_​n​o​d​e​s)n\_in\_nodes\leftarrow\min(\text{len}(candidates),n\_in\_nodes)

w​e​i​g​h​t​s←[depth_to_root​(c)⋅1.5+1​∀c∈c​a​n​d​i​d​a​t​e​s]weights\leftarrow[\text{depth\_to\_root}(c)\cdot 1.5+1\;\forall c\in candidates]

⊳\triangleright Define sampling weights to prioritize deeper nodes.

⊳\triangleright This way, the solution reasoning chain is expected to be longer.

i​n​_​n​o​d​e​s←random_choice​(c​a​n​d​i​d​a​t​e​s,n​_​i​n​_​n​o​d​e​s,prob=w​e​i​g​h​t​s/sum​(w​e​i​g​h​t​s))in\_nodes\leftarrow\text{random\_choice}(candidates,n\_in\_nodes,\text{prob}=weights/\text{sum}(weights))

c​u​r​_​l​a​b​e​l←0 cur\_label\leftarrow 0

for

i​n​_​i​d​x∈i​n​_​n​o​d​e​s in\_idx\in in\_nodes
do

c​u​r​_​l​a​b​e​l←c​u​r​_​l​a​b​e​l∣l​a​b​e​l​s​[i​n​_​i​d​x]cur\_label\leftarrow cur\_label\mid labels[in\_idx]
⊳\triangleright Update label using bitwise OR.

e​d​g​e​s.append​((i​n​_​i​d​x,i​d​x))edges.\text{append}((in\_idx,idx))

end for

g​r​o​u​p​s​[c​u​r​_​l​a​b​e​l].append​(i​d​x)groups[cur\_label].\text{append}(idx)

l​a​b​e​l​s​[i​d​x]←c​u​r​_​l​a​b​e​l labels[idx]\leftarrow cur\_label

n​o​d​e​s←n​o​d​e​s∪{i​d​x}nodes\leftarrow nodes\cup\{idx\}

i​d​x←i​d​x+1 idx\leftarrow idx+1

end while

# Nodes# Edges Len. of Shortest Path# Shortest Paths
23.0 36.0 3.8 1.6

Table 2: Statistics of the graph structure in ProsQA.

### 7.3 Statistics

We show the size of all datasets in Table [3](https://arxiv.org/html/2412.06769v3#S7.T3 "Table 3 ‣ 7.3 Statistics ‣ 7 Datasets ‣ Training Large Language Models to Reason in a Continuous Latent Space").

Table 3: Statistics of the datasets.

8 Clock-Time Reasoning Efficiency Metric
----------------------------------------

We present a clock-time comparison to evaluate reasoning efficiency. The reported values represent the average inference time per test case (in seconds), with a batch size of 1, measured on an Nvidia A100 GPU. For the no-CoT and CoT baselines, we employ the standard generate method from the transformers 3 3 3[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers) library. Our results show that clock time is generally proportional to the number of newly generated tokens, as detailed in Table [1](https://arxiv.org/html/2412.06769v3#S5.T1 "Table 1 ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space").

Table 4: Inference time (in seconds) comparison across tasks and methods.

9 More Discussion
-----------------

### 9.1 Using More Continuous Thoughts

In Figure [8](https://arxiv.org/html/2412.06769v3#S5.F8 "Figure 8 ‣ 5.3 Results and Discussion ‣ 5 Empirical Results with Coconut ‣ Training Large Language Models to Reason in a Continuous Latent Space") (II), we present the performance of Coconut on GSM8k using c∈{0,1,2}c\in\{0,1,2\}. When experimenting with c=3 c=3, we observe a slight performance drop accompanied by increased variance. Analysis of the training logs indicates that adding three continuous thoughts at once – particularly during the final stage transition – leads to a sharp spike in training loss, causing instability. Future work will explore finer-grained schedules, such as incrementally adding continuous thoughts one at a time while removing fewer language tokens, as in iCoT (Deng et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib7)). Additionally, combining language and latent reasoning—e.g., generating the reasoning skeleton in language and completing the reasoning process in latent space—could provide a promising direction for improving performance and stability.

### 9.2 Coconut with Larger Models

We experimented with Coconut on GSM8k using Llama 3.2-3B and Llama 3-8B (Dubey et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib8)) with c=1 c=1. We train them for 3 epochs in Stage 0, followed by 1 epoch per subsequent stage. The results are shown in Table [5](https://arxiv.org/html/2412.06769v3#S9.T5 "Table 5 ‣ 9.2 Coconut with Larger Models ‣ 9 More Discussion ‣ Training Large Language Models to Reason in a Continuous Latent Space").

Table 5: Experimental results of applying Coconut to larger Llama models. We report performance comparisons between models without CoT reasoning (no-CoT) and our proposed Coconut method.

We observe consistent performance gains across both Llama 3.2-3B and Llama 3-8B models compared to the no-CoT baseline, though these improvements are not as pronounced as those previously demonstrated with GPT-2. One possible reason is that larger models have already undergone extensive language-focused pre-training, making the transition to latent reasoning more challenging.

We emphasize that the primary goal of this paper is to highlight the promising attributes of latent-space reasoning and to initiate exploration in this new direction. Universally surpassing language-based CoT likely requires significant research efforts dedicated to latent space pre-training. We are encouraged by recent progress in this area (Geiping et al., [2025](https://arxiv.org/html/2412.06769v3#bib.bib14); Barrault et al., [2024](https://arxiv.org/html/2412.06769v3#bib.bib3); Gladstone et al., [2025](https://arxiv.org/html/2412.06769v3#bib.bib16)). While these recent models provide scalable methods for latent representation learning, the latent spaces have not yet been explicitly optimized for reasoning. Integrating these recent advancements with Coconut presents an exciting and promising avenue for future research.
