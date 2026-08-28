Title: LLM Pretraining with Continuous Concepts

URL Source: https://arxiv.org/html/2502.08524

Markdown Content:
### 3.1 Main Results

In this section, we illustrate two core results: i) the comparison with NTP on a relatively large-scale pretraining setup and ii) the comparison with KD baseline, especially on weak-to-strong supervision scenarios where concepts extracted from a small model are used to guide a larger model.

Improving NTP with CoCoMix at scale. We first present the main result by applying CoCoMix to the NTP. Here, we consider training NTP and CoCoMix on 200B tokens. As shown in [Figure 3](https://arxiv.org/html/2502.08524v1#S3.F3 "Figure 3 ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), CoCoMix consistently and significantly improves the performance of overall downstream tasks on various model sizes. Our results also indicate that larger models (e.g., 386M and 1.38B) can benefit from using concepts extracted from a smaller 124M model, showing effective weak-to-strong supervision. Moreover, as shown in [Figure 2](https://arxiv.org/html/2502.08524v1#S3.F2 "Figure 2 ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), CoCoMix consistently improves the performance over the NTP on a billion-scale model. For instance, CoCoMix achieves similar performance to NTP while using 21.5% fewer tokens, demonstrating high sample efficiency. Finally, it is worth noting that the performance gain of using CoCoMix is increasing over the training steps, demonstrating strong generalization performance.

Comparison with KD baseline. We also compare CoCoMix with KD baseline across multiple scenarios, including (1) a stronger teacher model teaching a smaller student model; (2) weak-to-strong supervision, where a weaker teacher teaches a larger student model; and (3) distribution shift, where the student is trained on a corpus different from the teacher’s pretraining distribution. As shown in [section 3](https://arxiv.org/html/2502.08524v1#S3 "3 Experiments ‣ LLM Pretraining with Continuous Concepts"), CoCoMix demonstrates improvements over KD in all considered model configurations. In particular, CoCoMix shows significant performance gain in the weak-to-strong supervision setup, e.g., improving average perplexity of 2.8 in 386M, while KD does not show great improvement. This arises because a weaker teacher can introduce noisy or suboptimal knowledge, especially when the student surpasses the teacher in capability (Rawat et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib36)). This trend is also observable in [Figure 4](https://arxiv.org/html/2502.08524v1#S3.F4 "Figure 4 ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), where models trained with KD fall behind standard training midway through training as the student outpaces the teacher (especially in the distribution shift scenario). In contrast, CoCoMix selectively utilizes useful concepts, resulting in a consistent performance gain.

![Image 1: Refer to caption](https://arxiv.org/html/2502.08524v1/x16.png)

Figure 5: Qualitative demonstration of the concept steering effect. CoCoMix and GPT2 models are 350M and 124M parameter transformers, respectively, trained on the OpenWebText dataset. For CoCoMix, we manipulate the predicted concept logit 𝐳 𝐳{\mathbf{z}}bold_z, while for GPT2, we adjust the SAE concept space 𝐜 𝐜{\mathbf{c}}bold_c by increasing the activation of a specific concept index. This illustrates the impact of targeted concept steering on the respective model outputs.

### 3.2 Interpretability and Steerability of CoCoMix

Another core advantage of CoCoMix is its interpretability and model steering. Specifically, as the model is trained to predict concepts in its hidden state, we can analyze which concepts it focuses on based on the concept predictions. Furthermore, by amplifying the magnitude of the predicted concept 𝐳 t subscript 𝐳 𝑡{\mathbf{z}}_{t}bold_z start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, one can control the output generation of the model. Following Templeton et al.([2024](https://arxiv.org/html/2502.08524v1#bib.bib45)), we multiply 𝐳 t subscript 𝐳 𝑡{\mathbf{z}}_{t}bold_z start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT of a desired concept element by constants ranging from -10 to 10. To verify whether this steerability works as intended, we steer the activation of the same concept in the pretrained model’s SAE latent space 𝐜 𝐜{\mathbf{c}}bold_c and confirm whether the output exhibits the corresponding concept. Here, we use a 386M parameter model trained with CoCoMix, where the pretrained model is GPT-2. As shown in [Figure 5](https://arxiv.org/html/2502.08524v1#S3.F5 "Figure 5 ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), when the concept related to “website address” is amplified, both models start generating actual website addresses. This demonstrates that our model has successfully learned the GPT-2 aligned concepts. More examples of steering can be found in Appendix [7.2](https://arxiv.org/html/2502.08524v1#S7.SS2 "7.2 Additional Steerability Results ‣ 7 Additional Results ‣ 6 Experimental Details ‣ 5 Conclusion ‣ 4 Related Work ‣ 3.3 Analysis of CoCoMix’s effectiveness ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts").

![Image 2: Refer to caption](https://arxiv.org/html/2502.08524v1/x17.png)

(a) Effectiveness of the attribution score

![Image 3: Refer to caption](https://arxiv.org/html/2502.08524v1/x18.png)

(b) Concept vs. direct hidden state

![Image 4: Refer to caption](https://arxiv.org/html/2502.08524v1/x19.png)

(c) Compression layer’s weight analysis

![Image 5: Refer to caption](https://arxiv.org/html/2502.08524v1/x20.png)

(d) Component analysis

![Image 6: Refer to caption](https://arxiv.org/html/2502.08524v1/x21.png)

(e) Design choice for concept condition

![Image 7: Refer to caption](https://arxiv.org/html/2502.08524v1/x22.png)

(f) Comparison with Pause Token

Figure 6:  Analysis of CoCoMix: (a) Effectiveness of the attribution score for selecting concepts. (b) Comparison between concept prediction and direct hidden state prediction (i.e., predicting the hidden state with continuous loss rather than discretizing the hidden state with SAE). (c) The sparsity in the compression weight. (d) Component analysis by analyzing the contribution of concept prediction and mixing. (e) Design choices for concept conditioning by comparing adding the concept vector to the original hidden state and mixing (interleaving the concept vector with token hidden representation). (f) Comparison between CoCoMix and the Pause token (i.e., adding learnable tokens). We use a 69M transformer and train on 20B tokens from the OpenWebText dataset. 

### 3.3 Analysis of CoCoMix’s effectiveness

In this section, we provide a detailed analysis of CoCoMix to validate the effect of each proposed component. Unless otherwise specified, we use a 69M model and train on 20B tokens sampled from the OpenWebText dataset across all methods throughout this section.

Effectiveness of the attribution score. We first analyze whether the attribution score effectively extracts important concepts. To demonstrate this, we train CoCoMix using the activation value 𝐜 t subscript 𝐜 𝑡{\mathbf{c}}_{t}bold_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT for concept extraction, i.e., ℒ 𝚌𝚘𝚗𝚌𝚎𝚙𝚝⁢(𝐜 t)subscript ℒ 𝚌𝚘𝚗𝚌𝚎𝚙𝚝 subscript 𝐜 𝑡\mathcal{L}_{\mathtt{concept}}({\mathbf{c}}_{t})caligraphic_L start_POSTSUBSCRIPT typewriter_concept end_POSTSUBSCRIPT ( bold_c start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) in [Equation 2](https://arxiv.org/html/2502.08524v1#S2.E2 "Equation 2 ‣ 2.2 Continuous Concept Mixing ‣ 2 CoCoMix: Continuous Concept Mixing ‣ LLM Pretraining with Continuous Concepts"), instead of the attribution score 𝐚 t subscript 𝐚 𝑡{\mathbf{a}}_{t}bold_a start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT. Remark that the activation value also well reflects the importance of the concept (Bricken et al., [2023](https://arxiv.org/html/2502.08524v1#bib.bib4)). As shown in [6(a)](https://arxiv.org/html/2502.08524v1#S3.F6.sf1 "Figure 6(a) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), using attribution scores significantly improves performance, improving sample efficiency by 17.5% compared to activation value based selection. We believe it will be an interesting future direction to explore other selection criteria for improving CoCoMix’s performance or removing undesirable concepts to reduce bias, e.g., selectively removing unsafe concepts for safe LLM pretraining.

Comparison with direct hidden state predictions. To evaluate the importance of predicting the concept extracted from SAE, we compare CoCoMix with direct hidden state prediction strategies (i.e., predict the full hidden state without projecting into the concept space). To have a comparison under the same architecture as CoCoMix, we replace the concept prediction head M 𝑀 M italic_M with a two-layer multilayer perceptron (MLP), denoted g⁢(⋅)𝑔⋅g(\cdot)italic_g ( ⋅ ), which predicts the pretrained LLM’s hidden state 𝐡 𝚌𝚘𝚗 superscript 𝐡 𝚌𝚘𝚗{\mathbf{h}}^{\mathtt{con}}bold_h start_POSTSUPERSCRIPT typewriter_con end_POSTSUPERSCRIPT directly from the hidden state of the model 𝐡 𝐡{\mathbf{h}}bold_h. The predicted representation, g⁢(𝐡)𝑔 𝐡 g({\mathbf{h}})italic_g ( bold_h ), is then compressed into a continuous embedding for insertion to have the same architecture as CoCoMix. Here, we use continuous loss including, ℓ 1 subscript ℓ 1\ell_{1}roman_ℓ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT, ℓ 2 subscript ℓ 2\ell_{2}roman_ℓ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, and the cosine distance (e.g., |𝐡 𝚌𝚘𝚗−g⁢(𝐡)|2 2 superscript subscript superscript 𝐡 𝚌𝚘𝚗 𝑔 𝐡 2 2\lvert{\mathbf{h}}^{\mathtt{con}}-g({\mathbf{h}})\rvert_{2}^{2}| bold_h start_POSTSUPERSCRIPT typewriter_con end_POSTSUPERSCRIPT - italic_g ( bold_h ) | start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for ℓ 2 subscript ℓ 2\ell_{2}roman_ℓ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT) to predict the hidden state. As shown in [6(b)](https://arxiv.org/html/2502.08524v1#S3.F6.sf2 "Figure 6(b) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), direct hidden state prediction leads to a performance drop. We conjecture this to be due to SAE’s ability to decompose the latent into semantically meaningful concepts while predicting all hidden states may include noisy components, emphasizing the effectiveness of our method.

Compression layer weight analysis. Now, we analyze the weight of the compression layer 𝐖 𝐖{\mathbf{W}}bold_W in [Equation 3](https://arxiv.org/html/2502.08524v1#S2.E3 "Equation 3 ‣ 2.2 Continuous Concept Mixing ‣ 2 CoCoMix: Continuous Concept Mixing ‣ LLM Pretraining with Continuous Concepts") to show how CoCoMix utilize the predicted concept. To this end, we visualize the ℓ 2 subscript ℓ 2\ell_{2}roman_ℓ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT norm of each concept’s weights of 386M CoCoMix: for the weight matrix 𝐖∈ℝ d×C 𝐖 superscript ℝ 𝑑 𝐶{\mathbf{W}}\in\mathbb{R}^{d\times C}bold_W ∈ blackboard_R start_POSTSUPERSCRIPT italic_d × italic_C end_POSTSUPERSCRIPT, where d 𝑑 d italic_d is the hidden dimension and C 𝐶 C italic_C is the number of concepts, the norm is defined as: ∥𝐖:,c∥2=∑d=1 D W d,c 2 subscript delimited-∥∥subscript 𝐖:𝑐 2 superscript subscript 𝑑 1 𝐷 superscript subscript 𝑊 𝑑 𝑐 2\lVert{\mathbf{W}}_{:,c}\rVert_{2}=\sqrt{\sum_{d=1}^{D}W_{d,c}^{2}}∥ bold_W start_POSTSUBSCRIPT : , italic_c end_POSTSUBSCRIPT ∥ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = square-root start_ARG ∑ start_POSTSUBSCRIPT italic_d = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_D end_POSTSUPERSCRIPT italic_W start_POSTSUBSCRIPT italic_d , italic_c end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG. As shown in [6(c)](https://arxiv.org/html/2502.08524v1#S3.F6.sf3 "Figure 6(c) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), we found that a portion of concept weights are near zero, e.g., 5.8% has a norm less than 1⁢e-⁢2 1 e-2 1\text{e-}2 1 e- 2. This indicates that CoCoMix learns to ignore these concepts when compressing them into a continuous concept if it is not useful. We conjecture such a process enabled CoCoMix for strong weak-to-strong supervision as it learned to ignore ineffective concepts extracted from a weak model.

Component Analysis. We analyze the contributions of each component of our method: (a) concept prediction [Equation 2](https://arxiv.org/html/2502.08524v1#S2.E2 "Equation 2 ‣ 2.2 Continuous Concept Mixing ‣ 2 CoCoMix: Continuous Concept Mixing ‣ LLM Pretraining with Continuous Concepts"), and (b) concept insertion [Equation 4](https://arxiv.org/html/2502.08524v1#S2.E4 "Equation 4 ‣ 2.2 Continuous Concept Mixing ‣ 2 CoCoMix: Continuous Concept Mixing ‣ LLM Pretraining with Continuous Concepts"). The results in [6(d)](https://arxiv.org/html/2502.08524v1#S3.F6.sf4 "Figure 6(d) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts") demonstrate that both components are critical for performance improvement. Specifically, applying the concept prediction loss alone yields a modest reduction in perplexity. However, incorporating concept insertion alongside prediction enhances the effectiveness of the loss, resulting in further performance gains. This highlights the role of insertion in enabling the model to leverage the pretrained LLM’s latent reasoning effectively. Notably, while concept insertion increases parameter count, it has a limited impact on performance when used alone, emphasizing the critical role of concept prediction.

Continuous concept mixing method. We explored two methods for introducing the continuous concept 𝐜^t subscript^𝐜 𝑡\hat{{\mathbf{c}}}_{t}over^ start_ARG bold_c end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT into the model’s hidden state 𝐡 t subscript 𝐡 𝑡{\mathbf{h}}_{t}bold_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, referred to as concept conditioning. The first method, insertion, insert the continuous concept in front of the token embedding, thus enabling the model to have a mixed input of the token and concept. The other option is intervention, which directly modifies the hidden state by adding the concept vector, i.e., 𝐡 t←𝐡 t+𝐜^t←subscript 𝐡 𝑡 subscript 𝐡 𝑡 subscript^𝐜 𝑡{\mathbf{h}}_{t}\leftarrow{\mathbf{h}}_{t}+\hat{{\mathbf{c}}}_{t}bold_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ← bold_h start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + over^ start_ARG bold_c end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT. As illustrated in [6(e)](https://arxiv.org/html/2502.08524v1#S3.F6.sf5 "Figure 6(e) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), both methods enhance pretraining performance (compared to NTP in [6(d)](https://arxiv.org/html/2502.08524v1#S3.F6.sf4 "Figure 6(d) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts")), highlighting the importance of concept conditioning, where the insertion method performed better. By introducing a distinct concept vector, the insertion method enables the model to explicitly recognize and effectively utilize concepts during generation, enhancing its overall performance.

CoCoMix vs. pause tokens. Furthermore, we consider an additional baseline that jointly uses pause token (Goyal et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib15)). Specifically, the pause token uses an additional learnable token that is inserted into the token embedding, enabling the LLM to think more before predicting the next token, which is similar to our continuous concept insertion. To this end, we insert the pause token for every input token on the same hidden state layer as CoCoMix to ensure comparable computation. Moreover, we also consider training the pause token with KD. As shown in [6(f)](https://arxiv.org/html/2502.08524v1#S3.F6.sf6 "Figure 6(f) ‣ Figure 6 ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), CoCoMix consistently outperforms the pause token, indicating our inserted (or interleaved) continuous concept indeed consists of useful information to improve the performance.

4 Related Work
--------------

Beyond token-level guidance for language modeling. While next token prediction remains the standard paradigm for language modeling, recent approaches have begun to explore methods that provide guidance beyond language tokens. For instance, some methods explore a better target, such as leveraging multi-token predictions to capture long context dependencies (Gloeckle et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib14); DeepSeek-AI, [2024](https://arxiv.org/html/2502.08524v1#bib.bib9)) or predicting sequence embedding (Lee et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib25)). Additionally, methods explore new types of inputs, e.g., using latents (Hao et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib17)) or self-generated thought as inputs (Zelikman et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib53)), which have shown improving reasoning capabilities. Only recently, concept-level modeling using local encoder-decoder architectures has also been explored to represent a language at a higher abstraction level (LCM et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib22)). Other methods add extra tokens in the input space to increase computation at inference time Nye et al.([2021](https://arxiv.org/html/2502.08524v1#bib.bib31)); Wei et al.([2022](https://arxiv.org/html/2502.08524v1#bib.bib46)); Goyal et al.([2024](https://arxiv.org/html/2502.08524v1#bib.bib15)); Lanchantin et al.([2024](https://arxiv.org/html/2502.08524v1#bib.bib21)). In contrast to other works, we propose a pretraining approach that integrates next token prediction with continuous concepts, connecting high level concepts with fine-grained token guidance.

Sparse Autoencoders (SAEs). SAEs extend the autoencoder by enforcing sparsity constraints in the latent space (Lee et al., [2006](https://arxiv.org/html/2502.08524v1#bib.bib24)). The features learned by SAEs are often interpretable and disentangled, making them useful across various domains, including language modeling (Bricken et al., [2023](https://arxiv.org/html/2502.08524v1#bib.bib4)). Additionally, SAEs have gained attention in mechanistic interpretability due to their ability to capture coherent semantic concepts (Marks et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib29)). This property has enabled practical advancements in identifying and manipulating semantic concepts and facilitating steering for controlled model outputs (Lieberum et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib26)). Among SAE variants, TopK SAEs (Makhzani and Frey, [2014](https://arxiv.org/html/2502.08524v1#bib.bib28)) enforce explicit sparsity using a TopK activation function, demonstrating effectiveness even for large models (Gao et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib13)). In this work, we leverage SAE and, to the best of our knowledge, are the first to apply it to LLM pretraining, achieving strong performance while enhancing the interpretability and controllability of the trained model.

Knowledge distillation (KD). Our method can also be related to KD, i.e., transfers the expertise of a teacher model to a student model to enhance performance (Hinton et al., [2015](https://arxiv.org/html/2502.08524v1#bib.bib18); Zagoruyko and Komodakis, [2017](https://arxiv.org/html/2502.08524v1#bib.bib52)), as CoCoMix extracts high-level semantic features from a pretrained model which is used to train a base model. Recently, KD for LLMs has garnered increasing attention, leveraging knowledge from a teacher to improve the generative and encoding capabilities of a student (Sanh et al., [2019](https://arxiv.org/html/2502.08524v1#bib.bib39); Ko et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib20)). Especially, applying KD to LLM pretraining remains challenging due to the massive token scales (billions to trillions), forcing most current methods to resort to naive token-level probability matching (Team, [2024](https://arxiv.org/html/2502.08524v1#bib.bib44); Gu et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib16)). Additionally, while pretrained models contain a vast amount of learned information and are thus beneficial to use, reusing knowledge from smaller teacher models remains challenging (Burns et al., [2023](https://arxiv.org/html/2502.08524v1#bib.bib6)). In this work, we show CoCoMix can even leverage the concept extracted from small models to train a large model showing weak-to-strong supervision.

5 Conclusion
------------

We propose Continuous Concept Mixing (CoCoMix), a new LLM pretraining framework that augments next token prediction with continuous concepts. By leveraging concepts extracted from a pretrained SAE as targets, our model predicts both the next token and the associated concept. Predicted concepts are then compressed into a continuous concept vector, which is then mixed into the hidden state. This approach enhances interpretability and controllability by enabling direct probing of the distilled concepts. Experimental results show that CoCoMix consistently improves performance across benchmarks, particularly in challenging generalization scenarios such as weak-to-strong supervision. Future work could explore learning continuous concepts during pretraining without the need for distillation.

References
----------

*   Bachmann and Nagarajan (2024) Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. In _International Conference on Machine Learning_, 2024. 
*   Baehrens et al. (2010) David Baehrens, Timon Schroeter, Stefan Harmeling, Motoaki Kawanabe, Katja Hansen, and Klaus-Robert Müller. How to explain individual classification decisions. _The Journal of Machine Learning Research_, 2010. 
*   Bisk et al. (2020) Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In _AAAI Conference on Artificial Intelligence_, 2020. 
*   Bricken et al. (2023) Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html. 
*   Brown et al. (2020) Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In _Advances in Neural Information Processing Systems_, 2020. 
*   Burns et al. (2023) Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. _arXiv preprint arXiv:2312.09390_, 2023. 
*   Clark et al. (2018) Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_, 2018. 
*   Cunningham et al. (2023) Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. _arXiv preprint arXiv:2309.08600_, 2023. 
*   DeepSeek-AI (2024) DeepSeek-AI. Deepseek-v3 technical report. _arXiv preprint arXiv:2412.19437_, 2024. 
*   Deng et al. (2023) Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. _arXiv preprint arXiv:2311.01460_, 2023. 
*   Dubey et al. (2024) Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_, 2024. 
*   Gao et al. (2023) Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A general multi-modal assistant that can plan, execute, inspect, and learn. _arXiv preprint arXiv:2306.08640_, 2023. 
*   Gao et al. (2024) Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. _arXiv preprint arXiv:2406.04093_, 2024. 
*   Gloeckle et al. (2024) Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In _International Conference on Machine Learning_, 2024. 
*   Goyal et al. (2024) Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. In _International Conference on Learning Representations_, 2024. 
*   Gu et al. (2024) Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Miniplm: Knowledge distillation for pre-training language models. _arXiv preprint arXiv:2410.17215_, 2024. 
*   Hao et al. (2024) Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. _arXiv preprint arXiv:2412.06769_, 2024. 
*   Hinton et al. (2015) Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_, 2015. 
*   Kim and Rush (2016) Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In _Annual Conference of the Association for Computational Linguistics_, 2016. 
*   Ko et al. (2024) Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. Distillm: Towards streamlined distillation for large language models. In _International Conference on Machine Learning_, 2024. 
*   Lanchantin et al. (2024) Jack Lanchantin, Shubham Toshniwal, Jason Weston, Sainbayar Sukhbaatar, et al. Learning to reason and memorize with self-notes. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   LCM et al. (2024) LCM, Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews, Mariano Coria, Guillaume Couairon, Marta R Costa-jussà, et al. Large concept models: Language modeling in a sentence representation space. _arXiv preprint arXiv:2412.08821_, 2024. 
*   LeCun (2022) Yann LeCun. A path towards autonomous machine intelligence. _Open Review_, 2022. 
*   Lee et al. (2006) Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Ng. Efficient sparse coding algorithms. In _Advances in Neural Information Processing Systems_, 2006. 
*   Lee et al. (2024) Hyunji Lee, Doyoung Kim, Jihoon Jun, Sejune Joo, Joel Jang, Kyoung-Woon On, and Minjoon Seo. Semiparametric token-sequence co-supervision. In _Annual Conference of the Association for Computational Linguistics_, 2024. 
*   Lieberum et al. (2024) Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. _arXiv preprint arXiv:2408.05147_, 2024. 
*   Loshchilov (2017) I Loshchilov. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_, 2017. 
*   Makhzani and Frey (2014) Alireza Makhzani and Brendan Frey. K-sparse autoencoders. In _International Conference on Learning Representations_, 2014. 
*   Marks et al. (2024) Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. _arXiv preprint arXiv:2403.19647_, 2024. 
*   Merity et al. (2017) Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In _International Conference on Learning Representations_, 2017. 
*   Nye et al. (2021) Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_, 2021. 
*   Paperno et al. (2016) Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In _Annual Conference of the Association for Computational Linguistics_, 2016. 
*   Paster et al. (2023) Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. _arXiv preprint arXiv:2310.06786_, 2023. 
*   Radford et al. (2018) Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 
*   Radford et al. (2019) Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 
*   Rawat et al. (2024) Ankit Singh Rawat, Veeranjaneyulu Sadhanala, Afshin Rostamizadeh, Ayan Chakrabarti, Wittawat Jitkrittum, Vladimir Feinberg, Seungyeon Kim, Hrayr Harutyunyan, Nikunj Saunshi, Zachary Nado, et al. A little help goes a long way: Efficient llm training by leveraging small lms. _arXiv preprint arXiv:2410.18779_, 2024. 
*   Roziere et al. (2023) Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_, 2023. 
*   Sakaguchi et al. (2020) Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In _AAAI Conference on Artificial Intelligence_, 2020. 
*   Sanh et al. (2019) Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. In _Advances in Neural Information Processing Systems_, 2019. 
*   Sap et al. (2019) Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In _Conference on Empirical Methods in Natural Language Processing_, 2019. 
*   Shrikumar et al. (2016) Avanti Shrikumar, Peyton Greenside, Anna Shcherbina, and Anshul Kundaje. Not just a black box: Learning important features through propagating activation differences. In _International Conference on Machine Learning_, 2016. 
*   Simonyan and Zisserman (2014) Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_, 2014. 
*   Sundararajan et al. (2017) Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In _International Conference on Machine Learning_, 2017. 
*   Team (2024) Gemma Team. Gemma 2: Improving open language models at a practical size. _arXiv preprint arXiv:2408.00118_, 2024. 
*   Templeton et al. (2024) Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C.Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. _Transformer Circuits Thread_, 2024. [https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). 
*   Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_, 35:24824–24837, 2022. 
*   Wu et al. (2024) Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. Reft: Representation finetuning for language models. In _Advances in Neural Information Processing Systems_, 2024. 
*   Xuan-Quy et al. (2023) Dao Xuan-Quy, Le Ngoc-Bich, Phan Xuan-Dung, Ngo Bac-Bien, and Vo The-Duy. Evaluation of chatgpt and microsoft bing ai chat performances on physics exams of vietnamese national high school graduation examination. _arXiv preprint arXiv:2306.04538_, 2023. 
*   Yang et al. (2024) Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, and Sebastian Riedel. Do large language models latently perform multi-hop reasoning? In _Annual Conference of the Association for Computational Linguistics_, 2024. 
*   Yu et al. (2024) Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling system 2 into system 1. _arXiv preprint arXiv:2407.06023_, 2024. 
*   Yun et al. (2021) Zeyu Yun, Yubei Chen, Bruno A Olshausen, and Yann LeCun. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. _arXiv preprint arXiv:2103.15949_, 2021. 
*   Zagoruyko and Komodakis (2017) Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In _International Conference on Learning Representations_, 2017. 
*   Zelikman et al. (2024) Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D Goodman. Quiet-star: Language models can teach themselves to think before speaking. In _Conference on Language Modeling_, 2024. 
*   Zellers et al. (2019) Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In _Annual Conference of the Association for Computational Linguistics_, 2019. 
*   Zou et al. (2023) Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. _arXiv preprint arXiv:2310.01405_, 2023. 

\beginappendix

6 Experimental Details
----------------------

Architecture details. Our method and baseline both utilize the GPT-2-based Transformer architecture and tokenizer (Radford et al., [2019](https://arxiv.org/html/2502.08524v1#bib.bib35)), with a context length of 1024. We consider three model sizes, defined by the number of activated parameters: 69M, 386M, and 1.38B. For CoCoMix, the hidden state dimensions d 𝑑 d italic_d are 512, 1024, and 2028, with 8, 24, and 24 layers, respectively. The number of attention heads is set to 8, 16, and 16 for the three configurations. Baseline models are configured to match the number of activated parameters by employing the same number of layers and attention heads, with hidden state dimensions of 624, 1072, and 2096, respectively. We leverage an open-source pretrained TopK Sparse Autoencoder (SAE) (Gao et al., [2024](https://arxiv.org/html/2502.08524v1#bib.bib13)) for concept extraction, where K 𝚌𝚘𝚗𝚌𝚎𝚙𝚝 subscript 𝐾 𝚌𝚘𝚗𝚌𝚎𝚙𝚝 K_{\mathtt{concept}}italic_K start_POSTSUBSCRIPT typewriter_concept end_POSTSUBSCRIPT is set to 32 and the concept activation size is 32,768. Consequently, our models introduce an additional (C+K 𝚌𝚘𝚗𝚌𝚎𝚙𝚝)×d 𝐶 subscript 𝐾 𝚌𝚘𝚗𝚌𝚎𝚙𝚝 𝑑(C+K_{\mathtt{concept}})\times d( italic_C + italic_K start_POSTSUBSCRIPT typewriter_concept end_POSTSUBSCRIPT ) × italic_d activated parameters on top of the base Transformer parameters. The GPT-2 model (a teacher model for KD and a concept extraction model for CoCoMix) has 12 layers, a hidden dimension size of 768, and 124M parameters. The concept extraction layer L 𝚌𝚘𝚗 subscript 𝐿 𝚌𝚘𝚗 L_{\mathtt{con}}italic_L start_POSTSUBSCRIPT typewriter_con end_POSTSUBSCRIPT is configured as the 6th middle layer of GPT-2 for all model sizes. For CoCoMix, the concept prediction is done at the 4th layer for the 69M model and the 6th layer for the larger 386M and 1.38B models.

Training details. We mainly followed the training details outlined in (Brown et al., [2020](https://arxiv.org/html/2502.08524v1#bib.bib5)). For the main results presented in Section[2](https://arxiv.org/html/2502.08524v1#S2 "2 CoCoMix: Continuous Concept Mixing ‣ LLM Pretraining with Continuous Concepts"), models were trained on 200B tokens over approximately 382K training steps, while other experiments were conducted on 20B tokens. We used OpenWebText as the training dataset to match the same training corpus as GPT-2. The learning rate schedule included a warm-up phase over the first 1/300 of the total training steps, followed by a cosine decay to 10% of the maximum learning rate at the end of training. The maximum learning rates were set to 6⁢e-⁢4 6 e-4 6\text{e-}4 6 e- 4, 3⁢e-⁢4 3 e-4 3\text{e-}4 3 e- 4, and 2⁢e-⁢4 2 e-4 2\text{e-}4 2 e- 4 for the 69M, 386M, and 1.38B models, respectively. The batch sizes were configured to 0.5M, 0.5M, and 1M tokens for the three model sizes. A weight decay of 0.1 was applied across all configurations, and we utilized the AdamW optimizer (Loshchilov, [2017](https://arxiv.org/html/2502.08524v1#bib.bib27)) with β 1=0.9 subscript 𝛽 1 0.9\beta_{1}=0.9 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 0.9 and β 2=0.95 subscript 𝛽 2 0.95\beta_{2}=0.95 italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.95. For training CoCoMix, the concept prediction loss [Equation 4](https://arxiv.org/html/2502.08524v1#S2.E4 "Equation 4 ‣ 2.2 Continuous Concept Mixing ‣ 2 CoCoMix: Continuous Concept Mixing ‣ LLM Pretraining with Continuous Concepts") was scaled by a hyperparameter λ=0.1 𝜆 0.1\lambda=0.1 italic_λ = 0.1, and K 𝚊𝚝𝚝𝚛𝚒 subscript 𝐾 𝚊𝚝𝚝𝚛𝚒 K_{\mathtt{attri}}italic_K start_POSTSUBSCRIPT typewriter_attri end_POSTSUBSCRIPT was set to 4. For the KD baseline, we employed the vanilla KD loss, where the output probabilities of the teacher model ℳ 𝚌𝚘𝚗 subscript ℳ 𝚌𝚘𝚗\mathcal{M}_{\mathtt{con}}caligraphic_M start_POSTSUBSCRIPT typewriter_con end_POSTSUBSCRIPT and the student model ℳ ℳ\mathcal{M}caligraphic_M were matched using the Kullback-Leibler (KL) divergence. Specifically, given an input 𝐱 𝐱{\mathbf{x}}bold_x, the KD loss is defined as −log ℳ(x t+1|𝐱)+λ KD⋅KL(ℳ 𝚌𝚘𝚗(⋅|𝐱)∥ℳ(⋅|𝐱))-\log\mathcal{M}(x_{t+1}|{\mathbf{x}})+\lambda_{\text{KD}}\cdot\mathrm{KL}(% \mathcal{M}_{\mathtt{con}}(\cdot|{\mathbf{x}})\|\mathcal{M}(\cdot|{\mathbf{x}}))- roman_log caligraphic_M ( italic_x start_POSTSUBSCRIPT italic_t + 1 end_POSTSUBSCRIPT | bold_x ) + italic_λ start_POSTSUBSCRIPT KD end_POSTSUBSCRIPT ⋅ roman_KL ( caligraphic_M start_POSTSUBSCRIPT typewriter_con end_POSTSUBSCRIPT ( ⋅ | bold_x ) ∥ caligraphic_M ( ⋅ | bold_x ) ), where λ KD=0.1 subscript 𝜆 KD 0.1\lambda_{\text{KD}}=0.1 italic_λ start_POSTSUBSCRIPT KD end_POSTSUBSCRIPT = 0.1 consistently demonstrated strong performance.

7 Additional Results
--------------------

### 7.1 Detailed Performance During Training

![Image 8: Refer to caption](https://arxiv.org/html/2502.08524v1/x23.png)

(a) OpenWebText, PPL (↓↓\downarrow↓)

![Image 9: Refer to caption](https://arxiv.org/html/2502.08524v1/x24.png)

(b) LAMBADA, PPL (↓↓\downarrow↓)

![Image 10: Refer to caption](https://arxiv.org/html/2502.08524v1/x25.png)

(c) WikiText-103, PPL (↓↓\downarrow↓)

![Image 11: Refer to caption](https://arxiv.org/html/2502.08524v1/x26.png)

(d) HellaSwag, Acc-n (↑↑\uparrow↑)

![Image 12: Refer to caption](https://arxiv.org/html/2502.08524v1/x27.png)

(e) PIQA, Acc (↑↑\uparrow↑)

![Image 13: Refer to caption](https://arxiv.org/html/2502.08524v1/x28.png)

(f) SIQA, Acc (↑↑\uparrow↑)

![Image 14: Refer to caption](https://arxiv.org/html/2502.08524v1/x29.png)

(g) Arc-Easy, Acc (↑↑\uparrow↑)

![Image 15: Refer to caption](https://arxiv.org/html/2502.08524v1/x30.png)

(h) WinoGrande, Acc (↑↑\uparrow↑)

Figure 7: CoCoMix vs. NTP performance at different training checkpoints on 69M parameter model. Each model is trained on the 200B tokens sampled from the OpenWebText dataset. The plot shows the result of (a) OpenWebText, (b) LAMBADA, (c) WikiText-103, (d) HellaSwag, (e) PIQA, (f) SIQA, (g) Arc-Easy, and (h) WinoGrande datasets. We use the concepts extracted from a 124M-sized model for training CoCoMix. 

![Image 16: Refer to caption](https://arxiv.org/html/2502.08524v1/x31.png)

(a) OpenWebText, PPL (↓↓\downarrow↓)

![Image 17: Refer to caption](https://arxiv.org/html/2502.08524v1/x32.png)

(b) LAMBADA, PPL (↓↓\downarrow↓)

![Image 18: Refer to caption](https://arxiv.org/html/2502.08524v1/x33.png)

(c) WikiText-103, PPL (↓↓\downarrow↓)

![Image 19: Refer to caption](https://arxiv.org/html/2502.08524v1/x34.png)

(d) HellaSwag, Acc-n (↑↑\uparrow↑)

![Image 20: Refer to caption](https://arxiv.org/html/2502.08524v1/x35.png)

(e) PIQA, Acc (↑↑\uparrow↑)

![Image 21: Refer to caption](https://arxiv.org/html/2502.08524v1/x36.png)

(f) SIQA, Acc (↑↑\uparrow↑)

![Image 22: Refer to caption](https://arxiv.org/html/2502.08524v1/x37.png)

(g) Arc-Easy, Acc (↑↑\uparrow↑)

![Image 23: Refer to caption](https://arxiv.org/html/2502.08524v1/x38.png)

(h) WinoGrande, Acc (↑↑\uparrow↑)

Figure 8: CoCoMix vs. NTP performance at different training checkpoints on 368M parameter model. Each model is trained on the 200B tokens sampled from the OpenWebText dataset. The plot shows the result of (a) OpenWebText, (b) LAMBADA, (c) WikiText-103, (d) HellaSwag, (e) PIQA, (f) SIQA, (g) Arc-Easy, and (h) WinoGrande datasets. We use the concepts extracted from a 124M-sized model for training CoCoMix. 

![Image 24: Refer to caption](https://arxiv.org/html/2502.08524v1/x39.png)

(a) OpenWebText, PPL (↓↓\downarrow↓)

![Image 25: Refer to caption](https://arxiv.org/html/2502.08524v1/x40.png)

(b) LAMBADA, PPL (↓↓\downarrow↓)

![Image 26: Refer to caption](https://arxiv.org/html/2502.08524v1/x41.png)

(c) WikiText-103, PPL (↓↓\downarrow↓)

![Image 27: Refer to caption](https://arxiv.org/html/2502.08524v1/x42.png)

(d) HellaSwag, Acc-n (↑↑\uparrow↑)

![Image 28: Refer to caption](https://arxiv.org/html/2502.08524v1/x43.png)

(e) PIQA, Acc (↑↑\uparrow↑)

![Image 29: Refer to caption](https://arxiv.org/html/2502.08524v1/x44.png)

(f) SIQA, Acc (↑↑\uparrow↑)

![Image 30: Refer to caption](https://arxiv.org/html/2502.08524v1/x45.png)

(g) Arc-Easy, Acc (↑↑\uparrow↑)

![Image 31: Refer to caption](https://arxiv.org/html/2502.08524v1/x46.png)

(h) WinoGrande, Acc (↑↑\uparrow↑)

Figure 9: CoCoMix vs. NTP performance at different training checkpoints on 1.38B parameter model. Each model is trained on the 200B tokens sampled from the OpenWebText dataset. The plot shows the result of (a) OpenWebText, (b) LAMBADA, (c) WikiText-103, (d) HellaSwag, (e) PIQA, (f) SIQA, (g) Arc-Easy, and (h) WinoGrande datasets. We use the concepts extracted from a 124M-sized model for training CoCoMix. 

In this section, we present the performance tracking during training on 200B tokens, including validation perplexity and the perplexity and accuracy of various downstream tasks, including LAMBADA, WikiText-103, HellaSwag, PIQA, SIQA, Arc-Easy, and WinoGrande datasets. As shown in [Figure 7](https://arxiv.org/html/2502.08524v1#S7.F7 "Figure 7 ‣ 7.1 Detailed Performance During Training ‣ 7 Additional Results ‣ 6 Experimental Details ‣ 5 Conclusion ‣ 4 Related Work ‣ 3.3 Analysis of CoCoMix’s effectiveness ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), [Figure 8](https://arxiv.org/html/2502.08524v1#S7.F8 "Figure 8 ‣ 7.1 Detailed Performance During Training ‣ 7 Additional Results ‣ 6 Experimental Details ‣ 5 Conclusion ‣ 4 Related Work ‣ 3.3 Analysis of CoCoMix’s effectiveness ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), and [Figure 9](https://arxiv.org/html/2502.08524v1#S7.F9 "Figure 9 ‣ 7.1 Detailed Performance During Training ‣ 7 Additional Results ‣ 6 Experimental Details ‣ 5 Conclusion ‣ 4 Related Work ‣ 3.3 Analysis of CoCoMix’s effectiveness ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"), we compare CoCoMix with the next token prediction (NTP) across different active parameter sizes: 69M, 386M, and 1.38B. In most of the graphs, CoCoMix consistently demonstrates performance gains. Notably, our results show that CoCoMix achieves stable improvements in perplexity across all tasks. Furthermore, CoCoMix exhibits sample efficiency; for instance, in the 1.38B model, CoCoMix reaches the same OpenWebText validation perplexity as NTP while requiring approximately 43B fewer tokens (a 21.5% improvement in token efficiency).

### 7.2 Additional Steerability Results

![Image 32: Refer to caption](https://arxiv.org/html/2502.08524v1/x47.png)

(a) Main figure prompt, ‘$ dollar’ and ‘Phone’ concept

![Image 33: Refer to caption](https://arxiv.org/html/2502.08524v1/x48.png)

(b) New prompt, ‘year/month’ and ‘Phone’ concept

![Image 34: Refer to caption](https://arxiv.org/html/2502.08524v1/x49.png)

(c) New prompt, ‘Politic/law’ and ‘aggressive tone’ concept

Figure 10: More qualitative demonstration of the concept steering effect. CoCoMix and GPT2 models are 350M and 124M parameter transformers, respectively. For CoCoMix, we manipulate the predicted logit 𝐳 𝐳{\mathbf{z}}bold_z, while for GPT2, we adjust the SAE concept space 𝐜 𝐜{\mathbf{c}}bold_c by increasing the activation of a specific concept index

To further analyze the steerability enabled by CoCoMix, we conducted experiments using both the same prompt as in the main figure and a new prompt (in [Figure 10](https://arxiv.org/html/2502.08524v1#S7.F10 "Figure 10 ‣ 7.2 Additional Steerability Results ‣ 7 Additional Results ‣ 6 Experimental Details ‣ 5 Conclusion ‣ 4 Related Work ‣ 3.3 Analysis of CoCoMix’s effectiveness ‣ 3.2 Interpretability and Steerability of CoCoMix ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts")). For consistency, we first applied steering on additional concepts identified during our analysis—“$ dollar” and “Phone”—using the same prompt as in [Figure 5](https://arxiv.org/html/2502.08524v1#S3.F5 "Figure 5 ‣ 3.1 Main Results ‣ 3 Experiments ‣ LLM Pretraining with Continuous Concepts"). These experiments confirmed that the model could effectively modulate its output based on these newly identified concepts, producing coherent and concept-aligned generations. Next, to verify whether the identified concepts generalize to different contexts, we experimented with a new prompt: “Latent concept modeling is a” and steered the model using the previously identified concepts “month/year” and “Phone.” The results showed that the model successfully reproduced outputs aligned with these concepts, further supporting the robustness of our method. Additionally, we explored whether new concepts could be identified and steered using the same prompt. In this case, we identified two new concepts: “politics/law” and “aggressive tone.” Steering the model with these new concepts demonstrated that the outputs could be effectively controlled to exhibit characteristics aligned with the corresponding concepts. These findings further highlight the flexibility and interpretability of our approach.
