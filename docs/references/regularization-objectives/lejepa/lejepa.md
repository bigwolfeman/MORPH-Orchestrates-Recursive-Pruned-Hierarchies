# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

- **Authors:** Randall Balestriero, Yann LeCun (galilai-group / NYU)
- **Year:** 2025
- **Source:** https://arxiv.org/abs/2511.08544
- **MORPH uses:** Attempts at using LeJEPA style joint prediction to predict future HCA and CSA blocks was attempted. Some runs showed improvement, but the loss function was being trivially fit very quickly. Predicting future tokens is not the same thing as JEPA, so this was dropped to prevent confusion and save FLOPS on a dead loss function contributing little to the model.

---

# **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics** 

Randall Balestriero[1,2] ,* Yann LeCun[3,2] ,* 

1 Brown University 3 New York University (NYU) 2 Meta-FAIR 

* Equal contribution 

Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in **LeJEPA** , a lean, scalable, and theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution that JEPAs’ embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel objective– **Sketched Isotropic Gaussian Regularization** (SIGReg)–to constrain embeddings to reach that ideal distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and - practical benefits: (i) single trade off hyperparameter, (ii) linear time and memory complexity, (iii) stability across - hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop gradient, no teacher–student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring only ≈50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will reestablish self-supervised pre-training as a core pillar of AI research (GitHub repo). 

**==> picture [207 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Spearman corr.: 94.52% (ViT/base-8 i net1k)<br>[J<br>10 [1] λ<br>[ [J]<br>$ 0 . 04<br>2 ° 0 . 08<br>5leiefe e 0 . 12<br>10 [0] ConYeoy-9.~® @)) NP, °[J 00 .. 1620<br>@G 2.<br>IO<br>@ 2<br>0 20 40 60<br>Test acc. (%)<br>(log-scale)<br>loss<br>Train<br>**----- End of picture text -----**<br>


**==> picture [255 x 268] intentionally omitted <==**

**----- Start of picture text -----**<br>
ViT-g/14, ImageNet-1K, LeJEPA<br>8<br>60<br>6<br>40<br>4<br>2 20<br>0<br>0<br>0 14 28 43 57 72<br>Epoch<br>Full FT Frozen<br>Method 1-sh Full 1-sh Full<br>LeJEPA (in-domain)<br>ConvNeXt-V2 Nano 29.42 82.72 28.74 76.52<br>ResNet-34 24.27 83.28 31.08 78.17<br>Frontier (transfer)<br>DINOv2 ViT-S/16 21.05 78.34 27.68 67.62<br>DINOv3 ViT-S/16 24.71 81.60 30.17 71.38<br>(%)<br>Loss Accuracy<br>**----- End of picture text -----**<br>


**Figure 1. LeJEPA overview. Top-left:** Training loss exhibits strong correlation with downstream linear probe performance on ImageNet-1k (ViT-base), providing the first practical loss for model selection without supervised probing. **Top-right:** Training stability without heuristics even on 1.8B ViT-g models, stable training loss. **Bottom-left:** PCA features from ImageNet-1k pretrained LeJEPA ViT-Large demonstrate clear semantic relationships. **Bottom-right:** Galaxy10 in-domain results showcasing LeJEPA’s in-domain pretraining consistently outperforms state-of-the-art frontier foundation models transfer learning (DINOv2/v3 trained on natural images) across data regimes from 1-shot to full supervision. This demonstrates that _domain-specific SSL beats generic transfer learning_ , even against massive-scale frontier models, when the framework scales effortlessly to any domain, model, and data scale. 

**Sec 1: Intro** | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

## **1 Introduction** 

Learning manipulable representations of the world and - its dynamics is a long standing question in AI, with roots dating back centuries ago [Von Helmholtz, 1867, Tolman, 1948, Gregory, 1980, Sutton, 1991, Friston, 2010]. Across domains, e.g., image recognition, robotics, physics, space exploration, the unifying question is _how to learn an orga- nized and actionable high dimensional embedding space from observations?_ Using Deep Networks–parameterized nonlinear operators _𝑓_ _**𝜽**_ –to map observations to embeddings is a standard first piece of that puzzle [LeCun et al., 2015, Goodfellow et al., 2016]. The second, less standardized, piece of that puzzle is _how to train 𝑓_ _**𝜽**_ . Joint-Embedding Predictive Architectures (JEPAs) suggest training _𝑓_ _**𝜽**_ by maximizing predictive agreement between the embeddings of semantically related _views_ [Bromley et al., 1993, LeCun, 2022, Balestriero et al., 2023]. Views can come in two forms: transformations or corruptions. They can involve masking, cropping, blurring, temporal or spatial translations, geometric or photometric transformations, viewpoint changes, views from different sensor modalities, etc. The supervised forms involve human-produced components such as image-caption pairs, text-code pairs, etc [Tian et al., 2020]. In any case, views are expected to share some degree of semantic relationship to allow the prediction task to align _𝑓_ _**𝜽**_ ’s embeddings towards the underlying knowledge present in the data. Alas, JEPA’s prediction task admits failure modes, such as representation collapse, where _𝑓_ _**𝜽**_ maps all inputs to nearly identical embeddings ( _complete collapse_ ) or to a lowdimensional subspace ( _dimensional collapse_ ) [Jing et al., 2021][Jing et al., 2021, Cosentino et al., 2022, Balestriero and LeCun, 2022]. To mitigate such shortcut solutions, state-of-the-art recipes rely on heuristics–stop-gradient [Chen et al., 2020a], asymmetric view generation [Wang et al., 2022], teacher–student networks with carefully tuned EMA schedules [Caron et al., 2021, Tian et al., 2021], explicit normalization and whitening layers [Ermolov et al., 2021, Chen et al., 2021]–and a delicate balance of hyperparameters. As a result, today’s JEPA training is brittle and most research has shifted toward scaling data [Vo et al., 2024], models [Fan et al., 2025] and even post-training Rodas et al. [2025] while leaving the theoretical foundations of JEPAs largely unexplored. 

Our study proposes to break that cycle by questioning some of the fundamental design principles underpinning JEPAs. That introspection will start by asking _what are the necessary conditions that JEPAs should abide by?_ Those minimal conditions will then act as _axioms_ for us to design a novel and lean JEPA. We identify two axioms: (i) solving the prediction task while (ii) enforcing an isotropic Gaussian distribution of the embeddings 

(Section 3). While (i) follows standard practice [Balestriero and LeCun, 2022], we introduce in Section 4 a novel distribution matching objective–Sketched Isotropic Gaussian Regularization (SIGReg)–to enforce (ii). The use of SIGReg not only removes the need for the numerous heuristics previously employed to prevent representation collapse, but SIGReg also exhibits favorable scaling properties as its _memory and computational complexity is linear in dimension and sample size_ . Crucially, SIGReg’s isotropic Gaussian enforcement solves the collapsed shortcut solution and provably minimizes the model’s expected risk over the space of downstream tasks to be encountered post-training. The resulting JEPA solution–coined Latent-Euclidean JEPA (LeJEPA)–is introduced in Section 5. Beyond theoretical optimality, LeJEPA offers numerous benefits such as (i) provable statistical guarantees, (ii) removal of heuristics such as teacher-student networks, (iii) linear memory and computational complexity, and most importantly (iv) a unified design with a single trade-off parameter that works out of the box across datasets, architectures and scales (see Section 6). We summarize our contributions below. 

**Contribution 1: We prove the optimal embedding distribution for foundation models.** We establish that the isotropic Gaussian uniquely minimizes downstream prediction risk across broad task families. In Section 3, we derive this result rigorously for both linear (Section 3.1) and nonlinear probes (Section 3.2), providing the first principled answer to what distribution _𝑓_ _**𝜽**_ ’s embeddings should follow. This theoretical result transforms JEPA design from heuristic exploration to targeted optimization. **Contribution 2: We introduce SIGReg, a distribution matching objective that uniquely combines provable correctness with computational efficiency at scale.** We present _Sketched Isotropic Gaussian Regularization_ (SIGReg), a novel objective that enforces distributional alignment via random projections and characteristic-function matching (Section 4 and Figure 2). SIGReg provides statistical guarantees (Sections 4.1 and 4.2) while achieving linear complexity and bounded gradients—a combination that existing distribution matching methods do not offer. Critically, its projection-based construction defeats the curse of dimensionality (Section 4.3), making it both theoretically sound and practically efficient for high-dimensional embeddings. 

**Contribution 3: We design LeJEPA, a statistically optimal JEPA that eliminates collapse by construction.** By combining JEPA’s predictive objective with SIGReg targeting the isotropic Gaussian, we introduce _LeJEPA_ —LatentEuclidean JEPA (Section 5). LeJEPA requires only a single hyperparameter, eliminates representational collapse without stop-gradients or teacher-student architectures, and transfers across architectures and datasets without hyperparameter tuning. This demonstrates that principled 

2 

## **LeJEPA:** 

Sec 1: Intro | **Sec 2: Background** | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [14 x 24] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝑓 𝜽<br>→<br>**----- End of picture text -----**<br>


**Figure 2. Sketched Isotropic Gaussian Regularization (SIGReg):** Given some arbitrary input data with density _𝑝𝑥_ with support that may or may not lie on a manifold ( **left** ), a Deep network (DN) encoder ( _𝑓_ _**𝜽**_ ) produces embeddings _𝒛_ = _𝑓_ _**𝜽**_ ( _𝒙_ ) with some distribution _𝒛_ ∼ _𝑝𝑧_ ( **middle** ). Our proposed Backward Cramér-Wold Statistics (Section 4) objective pushes _𝑝𝑧_ to match a target distribution _𝑝𝑡_ by projecting the embeddings along 1 _𝑑_ directions ( **middle, arrows** ) and enforcing that the univariate densities ( **right, colored lines** ) match the distribution of _𝑝𝑡_ , projected along the same directions. Any popular statistical test (provided in Section 4.2) can assess the goodness-of-fit–in practice we argue for characteristic function tests (Section 4.2). By using SIGReg with _𝑝𝑡_ isotropic Gaussian ( **right, black lines** ), we introduce a lean and provably optimal (Section 3) JEPA, coined LeJEPA, free of numerous heuristics and able to produce competitive performances (Sections 5 and 6). 

theory directly yields practical simplicity. 

**Contribution 4: We validate LeJEPA at scale across diverse architectures and establish in-domain pretraining as viable.** Our experiments (Section 6) span ViTs, ConvNeXts, ResNets, MaxViTs, and Swin Transformers at scales approaching 1 billion parameters, where LeJEPA matches or exceeds state-of-the-art methods while maintaining training simplicity and robustness. Critically, on domain-specific datasets (Galaxy10, Food101), LeJEPA outperforms DINOv2-based transfer learning when pretrained directly on target data. This challenges the transfer learning paradigm and demonstrates that principled SSL can unlock effective in-domain pretraining—previously considered impractical for small datasets. 

## **2 Background and Notations** 

We start by introducing some of the notations we will be using throughout our manuscript (Section 2.1), followed by a review of JEPAs (Section 2.2), and existing literature studying their design (Section 2.3). 

## **2.1 Notations and Definitions** 

**Data.** We are in possession of a dataset of shape ( _𝑁, 𝑉, 𝐷_ ) ∈ N[∗][3] where _𝑁_ is the number of samples, _𝑉_ is the number of views, and _𝐷_ is the dimension. One entry of this dataset is accessed via _𝒙𝑛,𝑣,𝑑_ . Those dimensions are often interpreted as follows: ( **N** ) is the number of independent samples, e.g., different images or different videos, ( **V** ) is the number of _views_ , e.g., data-augmentations for images, frames for videos, and ( **D** ) is the dimension of each _𝒙𝑛,𝑣_ , e.g., number of RGB pixels for images. In many cases the ordering over _𝑉_ is given by _time_ –but in some cases, e.g., data-augmentation of an image, ordering becomes 

irrelevant. Our study does not require any particular choice to organize one’s dataset into a ( _𝑁, 𝑉, 𝐷_ ) tensor– _and none of our theory and implementation assumes a particular design decision for that tensor_ . However, we will rely on the following two properties, ( _independence_ ) the samples _𝒙𝑛 , 𝒙𝑛_[′] have been obtained independently from each other ∀ _𝑛_ ≠ _𝑛_[′] , and ( _identically distributed_ ) the sampling process was identical among _𝒙𝑛 ,_ ∀ _𝑛_ . 

**Deep Networks.** Today’s AI solutions rely on _Deep (Neural) Networks_ (DNs), which are compositions of a large number of parameterized linear and nonlinear operators. We denote the DN’s mapping as _𝑓_ _**𝜽**_ : R _[𝐷]_ → R _[𝐾]_ with _𝐾_ the dimension of the embedding space. The internals of _𝑓_ _**𝜽**_ are designed by the researcher to incorporate as much prior knowledge about the data as possible. The details of _𝑓_ _**𝜽**_ are irrelevant to our study–as we will see the proposed LeJEPA works out-of-the-box on any _𝑓_ _**𝜽**_ . In any case, all the _learnable parameters_ are gathered in the vector _**𝜽**_ ∈ R _[𝑃]_ , with _𝑃_ counting the total number of parameters. A central challenge in AI research is to design the right architecture and training objective so that _**𝜽**_ can be learned from gradient descent to ultimately produce a useful system, or foundation model, _𝑓_ _**𝜽**_ . 

**JEPAs.** A foundation model is any system, e.g., a DN, able to solve numerous downstream tasks without requiring any change in its internal parameters _**𝜽**_ . This is in sharp contrast with a supervised model that only considers its training task. JEPAs have formally been introduced by LeCun [2022] as a vehicle to produce foundation models. The core building blocks of JEPAs rely on numerous wellestablished techniques such as siamese networks [Bromley et al., 1993] and predictive coding [Helmholtz et al., 1867, Bruner and Postman, 1949]. While the exact blueprint of 

3 

**LeJEPA:** 

Sec 1: Intro | **Sec 2: Background** | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **Definition 1: JEPA** 

**==> picture [449 x 12] intentionally omitted <==**

**----- Start of picture text -----**<br>
JEPA( 𝒙 ) ⇐⇒ Enc ( 𝒙𝑛,𝑡 +1 ,. ) is predictable from Enc ( 𝒙𝑛,𝑡,. ) ,  ∀ 𝑛, 𝑡 and Enc () 𝒙.,.,. is not degenerate . (1)<br>**----- End of picture text -----**<br>


JEPAs varies greatly between use-cases, they all rely on two core principles: (i) being able to predict the embedding of a view _𝒙𝑛,𝑣_ from the embedding of another view _𝒙𝑛,𝑣_[′] _, 𝑣_[′] ≠ _𝑣_ , all while (ii) ensuring that the embeddings do not become degenerate. Concretely, once a JEPA is designed and trained, it should be able to solve numerous downstream tasks in zero or few shots. The JEPA objective function, along with some examples for _𝒙_ , is provided in Equation (1). The _predictability_ criterion can be done by directly comparing the embeddings of the partial views _𝐸𝑛𝑐_ ( _𝒙𝑛,𝑣,._ ) and _𝐸𝑛𝑐_ ( _𝒙𝑛,𝑣_[′] _,._ ) with a metric, e.g., _ℓ𝑝_ . In some cases, an additional DN coined _Pred_ , is employed to compare _𝑃𝑟𝑒𝑑_ ( _𝐸𝑛𝑐_ ( _𝒙𝑛,𝑣,._ )) against _𝐸𝑛𝑐_ ( _𝒙𝑛,𝑣_[′] _,._ )–which is only justified when there exists an asymmetry between the information content of the different views, e.g., by conditioning the predictions on observed actions from robotics data [Khazatsky et al., 2024]. 

## **2.2 The Need for Reliable Pretraining** 

The JEPA’s prediction task is designed based on a priori knowledge of the data. Its design is often quite natural since it is relatively intuitive to form _𝒙_ so that its views share the relevant information content one hope to capture. On the other hand, the design of the “anti-collapse” criterion is much closer to a game of Whac-A-Mole. Today’s designs rely on many different under-specified safeguards which are carefully combined in the hope that degenerate shortcut solutions are avoided during training. Such mechanisms include (i) feature whitening [Ermolov et al., 2021, Bardes et al., 2021], (ii) negative samples [Chen et al., 2020a, He et al., 2020], and (iii) asymmetric views and teacher-student networks with stop-gradient [Caron et al., 2021, Assran et al., 2023]. Those mechanisms all suffer from at least two of the following limitations: (i) 

under-specification, i.e., the criteria can be minimized while embeddings are in a degenerate configuration, (ii) quadratic time and memory complexity with mini-batch size and/or embedding dimension, (iii) sensitivity to data distribution, hyperparameters, architecture, and (iv) lack of theoretical understanding and guarantees. 

## **2.3 The Need for Actionable Theory** 

For decades, the two major solutions for AI were supervised learning [LeCun et al., 2015] and learning by reconstruction [Rumelhart et al., 1986]–sometimes combined together, e.g., for semi-supervised learning [Kingma et al., 2014]. In supervised learning, the labels both ensure that semantically similar samples are close to each other in embedding space while preventing complete representation collapse. In particular, it is possible to measure the amount of collapse in supervised learning as a function of the number of classes [Papyan et al., 2020]. The reconstruction objective is similarly well suited to prevent representation collapse as the original input must be recovered from the embeddings, i.e., the embeddings must be as informative about the input as possible–up to some optional denoising tasks that users can setup as part of the training [Vincent et al., 2010]. 

Because supervised and reconstruction-based learning have been widely studied for decades, there exists a large body of work to explain and inform practical designs–as well as studying their limitations in producing foundation models [Balestriero and LeCun, 2024, Van Assel et al., 2025]. This is not the case for the more recent JEPAs where empirical advances quickly outpace anyone hoping to delve into their inner workings. This dynamic led the community to focus on post-hoc theoretical justification of already found solutions [Liu et al., 2021, Shwartz Ziv 

4 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | **Sec 3: Why Gaussian?** | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

and LeCun, 2024, Shwartz-Ziv et al., 2022, Zhang et al., 2023]. In most cases, those studies involve the _Mutual Information (MI)_ [Shannon, 1948, Cover, 1999] whose different bounds recover established methods [Gutmann and Hyvärinen, 2010, Ma and Collins, 2018, Oord et al., 2018, Poole et al., 2019, Hjelm et al., 2018, McAllester and Stratos, 2020]. Because existing studies focus on explaining and interpreting already developed JEPAs, too little principled guidance and innovation has been brought forward. Instead, most of the recent empirical advances take the form of collecting larger dataset, scaling up pre-existing training recipes [Goyal et al., 2019, Chen et al., 2020b, Oquab et al., 2023, Fan et al., 2025], and deriving novel data curation processes [Vo et al., 2024, Kerdreux et al., 2025]. 

In contrast, our goal in the following Sections 3 to 5 will be to derive a novel JEPA solution from first principles, i.e., whose design relies on proved necessary conditions for optimality, and with a pretraining recipe that can finally reconcile exploratory research, scalability, and state-of-theart performances. 

## **3 Latent Euclidean: Embeddings Should be Isotropic Gaussian** 

We address a fundamental question: _which distribution should_ Enc( _𝒙_ ) _follow to minimize empirical risk on any downstream task?_ We prove that the isotropic Gaussian is the unique optimal distribution for both linear (Section 3.1) and nonlinear probing (Section 3.2), with geometric intuition provided in Section 3.3. This theoretical result establishes the necessary design principle for our JEPA; Section 4 then provides the practical implementation to achieve it. 

## **3.1 Linear Probing** 

We begin by identifying the optimal distribution for _𝑓_ _**𝜽**_ ’s embeddings by analyzing linear probes–one of the most popular methods for frozen encoder evaluation. Specifically, we ask: _which distribution for 𝑓_ _**𝜽**_ ( _𝒙_ ) _would be most favorable for solving arbitrary downstream tasks, i.e., for any realization of targets 𝒚?_ 

Denote as _𝒁_ ∈ R _[𝑁]_[×] _[𝐾]_ the matrix of _𝑁_ embeddings, each _𝐾_ -dimensional, from _𝑓_ _**𝜽**_ ( _𝒙𝑛_ ). The _unknown_ corresponding labels are denoted as _𝒚_ ∈ R _[𝑁]_ . Without loss of generality, we consider univariate targets; the following analysis extends to multivariate targets. The linear probe minimizes the following least square problem [Bishop and Nasrabadi, 2006] 

**==> picture [193 x 23] intentionally omitted <==**

where _𝛽_[ˆ] is the optimal probe parameters, and _𝜆_ ≥ 0 is an hyperparameter controlling the Tikhonov regularizer strength [Bishop, 1995, Golub et al., 1999]. Despite 

not knowing _𝒚_ , it is possible to describe the bias and variance of the estimator _𝛽_[ˆ] as a function of the distribution of _𝒁_ . Consider two embeddings with identical column spans _𝒁_ aniso _, 𝒁_ iso. _𝒁_ aniso’s covariance matrix eigenvalues are given by { _𝜆𝑘_ } _[𝐾] 𝑘_ =1[with][at][least][two][distinct] values, while _𝒁_ iso’s covariance matrix eigenvalues are all equal to _𝐾_[1] � _𝐾𝑘_ =1 _[𝜆][𝑘]_[.][Hence, the two candidate embeddings] _𝒁_ aniso _, 𝒁_ iso capture the same intrinsic features and have same energy, but different geometries. 

## **Lemma 1: Anisotropy amplifies bias** 

Whenever _𝜆𝐾 > 𝜆_ 1, there always exists a downstream task ( _𝒚_ ) for which _𝒁_ aniso produces a higher bias estimator than _𝒁_ iso for _𝜆>_ 0. (Proof in Section B.1.) 

## **Lemma 2: Anisotropy amplifies variance** 

With _𝜆_ = 0, the total variance of _𝛽_[ˆ] (OLS) is minimized for _𝒁_ iso with tr(Var( _**𝜷**_[ˆ] aniso)) _>_ tr(Var( _**𝜷**_[ˆ] iso)). (Proof in Section B.2.) 

From the above lemmas. 1 and 2 we obtain that the distribution of features must be isotropic. We now move to nonlinear probing where the standard Gaussian will emerge as the unique optimum. 

## **3.2 Nonlinear Probing** 

To allow for more flexible evaluation of the pretrained encoder _𝑓_ _**𝜽**_ , it has become increasingly common to work with a nonlinear probe. We analyze two widely-used nonlinear methods: radius-based k-NN [Taunk et al., 2019, Sun and Huang, 2010, Zhang et al., 2017, Abu Alfeilat et al., 2019] for its simplicity and kernel methods [Nadaraya, 1964, Watson, 1964] for their theoretical tractability. 

As in Section 3.1, we ask ourselves which distribution of embeddings would be preferable for a foundation model. We first define our prediction function. The training data consists of the _𝑁_ embeddings along with their training labels {( _𝒛𝑛 , 𝒚𝑛_ )} _𝑛[𝑁]_ =1[.][The][prediction,][using][radius-based] k-NN for a query vector _𝒒_ is formed as 

**==> picture [185 x 30] intentionally omitted <==**

where 𝒩 _𝑟_ 0( _𝒒_ ) = { _𝑛_ : ∥ _𝒛𝑛_ − _𝒒_ ∥≤ _𝑟_ 0}. The specific choice of radius _𝑟_ 0 controls how many neighbors predictions are averaged to form the query’s prediction. The kernel’s prediction at a query _𝒒_ ∈ R _[𝐾]_ is given by 

**==> picture [204 x 30] intentionally omitted <==**

We search over all distributions of Z subject to a fixed total variance constraint, e.g., Tr(Cov( _𝒁_ )) = _𝜅_ 1 or ∥Cov( _𝒁_ )∥ _𝐹_ = _𝜅_ 2. The specific value of _𝜅_ does not affect the optimal dis- 

5 

**LeJEPA:** Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | **Sec 4: SIGReg** | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [252 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
Isotropic, Var( β [ˆ] ) = 0 . 0056 Anisotropic, Var( β [ˆ] ) = 0 . 0801<br>4 Condition #: 20<br>2<br>0<br>− 2<br>True boundary True boundary<br>Learned boundaries Learned boundaries<br>− 4<br>− 4 − 2 0 2 4 − 4 − 2 0 2 4<br>x 1 x 1<br>x 2<br>**----- End of picture text -----**<br>


**Figure 3.** Illustration of lemma. 2 showcasing how anisotropic ( **right** ) embeddings lead to higher variance estimator compared to isotropic embeddings ( **left** ). We sample 100 training points for the 2-class classification task and fit a logistic regression–repeating the process over numerous training set sample. Each sampling results in a decision boundary ( **purple** ). 

tribution shape. Following the same type of derivations as done in the linear regime–with the exception of some additional regularity conditions–we are able to precisely identify the isotropic Gaussian as the unique optimum to minimize bias as formalized below. 

## **Theorem 1: isotropic Gaussian Optimality** 

The integrated square bias (ISB) over query points is given by 

**==> picture [230 x 52] intentionally omitted <==**

and among distributions with a scalar-based covariance constraint, the isotropic Gaussian is the unique minimizer of the integrated square bias. (Proof in Sections B.4 and B.7.) 

Numerous additional details and discussions on the regularity assumptions we employed are provided in Section A. Together, these results establish the isotropic Gaussian distribution as the optimal design to minimize the worst-case risk of a foundation model across downstream tasks. 

parameters equals 1 only for isotropic distributions, degrading for anisotropic cases regardless of sample size or regularization strength. Regarding variance (lemma. 2), we show in Figure 3 that learned parameters vary significantly more across training sets when the covariance is anisotropic (right) compared to isotropic (left)—even when using logistic regression instead of OLS. Figure 17 further illustrates this effect, showing the distribution of learned _𝛽_ parameters across different training samples for both cases. The anisotropic distribution clearly produces higher-variance estimators. 

These theoretical and empirical results establish our design principle for LeJEPA: _embeddings 𝑓_ _**𝜽**_ ( _𝒙_ ) _should follow an isotropic Gaussian distribution to minimize worst-case risk across downstream tasks encountered post-training_ . Section 4 introduces a novel regularizer to achieve this distribution. 

## **4 SIGReg: Reliable Isotropic Gaussian Regularization in High-Dimension** 

Having established the isotropic Gaussian as the optimal embedding distribution (Section 3), we now introduce _Sketched Isotropic Gaussian Regularization_ (SIGReg)–a distribution matching objective that is simultaneously (i) _differentiable_ , (ii) _scalable_ , (iii) _provable_ , and (iv) _interpretable_ . SIGReg builds on three key innovations. First, we formulate distribution matching as a statistical test under the null hypothesis _𝑃_ _**𝜽**_ = _𝑄_ (Section 4.1). Second, we identify a test that guarantees bounded gradients and curvature while maintaining linear complexity and efficient multi-GPU scaling (Section 4.2). Third, SIGReg bypasses the curse of dimensionality, eliminating collapsed shortcut solutions entirely (Section 4.3). 

## **4.1 Hypothesis Testing as a Judge** 

Asking for _𝑓_ _**𝜽**_ ( _𝒙_ )’s distribution _𝑃_ _**𝜽**_ to match a target distribution _𝑄_ is typically done by creating various measures of distance or divergence, and estimating them in highdimension. We propose a different starting point grounded in statistics. Consider the hypothesis testing framework [Fisher, 1928, Neyman and Pearson, 1933] given by 

## **3.3 Geometric and Practical Insights** 

We now empirically validate that the isotropic Gaussian is optimal when no information about downstream tasks is available. We focus on linear probing (Section 3.1), where all considered distributions have the same total variance. 

When employing a linear probe, an anisotropic distribution increases both bias (with Tikhonov regularization) and variance. Examining bias first (lemma. 1), we present in Figure 18 visualizations for both continuous regression and discrete classification tasks. We observe that the cosine similarity between estimated and ground-truth 

**==> picture [197 x 10] intentionally omitted <==**

with _𝐻_ 0 being referred to as the _null hypothesis_ . That is, we are asking in Equation (2) if there is enough empirical evidence to reject the null. To answer that question, one (i) employs a _test-statistic_ , i.e., a single scalar value summarizing the evidence from the empirical samples, (ii) determines a critical value _𝜏𝛼_ for the test-statistic based on the probability _𝛼_ of Type I error, i.e., of mistakenly rejecting a true null hypothesis, (iii) compares the test-statistic to 

6 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | **Sec 4: SIGReg** | Sec 5: LeJEPA | Sec 6: Experiments 

the critical value _𝜏𝛼_ ; if the test-statistic exceeds _𝜏𝛼_ , reject the null hypothesis. If the null is not rejected, we can only claim that _there is not sufficient empirical evidence against 𝑃_ _**𝜽**_ = _𝑄_ . 

As it stands, Equation (2) remains impractical in large dimension as existing tests have at least quadratic complexity with the number of samples considered (more details in Section F). We thus propose to derive a sketching strategy by decomposing Equation (2) into simpler univariate tests. Denoting the push-forward distributions _𝑃_ _**𝜽**_[(] _[𝒂]_[)] ≜ ( _𝒂_[⊤] )# _𝑃_ _**𝜽**_ and _𝑄_[(] _[𝒂]_[)] ≜ ( _𝒂_[⊤] )# _𝑄_ , we can define the following _directional_ univariate test 

**==> picture [222 x 16] intentionally omitted <==**

for a given directional unit-norm vector _𝒂_ ∈𝒮 _[𝐾]_[−][1] . The corresponding _directional test-statistic_ of Equation (3) is computed as _𝑇_ ({ _𝒂_[⊤] _𝑓_ _**𝜽**_ ( _𝒙𝑛_ )} _𝑛[𝑁]_ =1[)][.][Examples of tests] _[ 𝑇]_[will] be provided in the later Section 4.2. Repeating that process over a set of _𝑀_ directions A ≜ { _𝒂_ 1 _, . . . , 𝒂𝑀_ } and aggregating the individual values lead to the following _global test-statistic_ 

**==> picture [218 x 18] intentionally omitted <==**

We now provide a formal statement asserting the consistency of Equation (4) to test the original multivariate null hypothesis from Equation (2). Our result leverages the well-known union-intersection principle [Roy, 1953], and a slightly modified Cramér-Wold theorem. We denote by _𝑑_ = equality in distribution. 

## **Lemma 3: Hyperspherical Cramér-Wold** 

Let _𝑋, 𝑌_ be R _[𝑑]_ -valued random vectors, then ⟨ _𝒖, 𝑋_ ⟩ = _[𝑑]_ ⟨ _𝒖, 𝑌_ ⟩ _,_ ∀ _𝒖_ ∈ S _[𝑑]_[−][1] ⇐⇒ _𝑋_ = _[𝑑] 𝑌._ 

Convergence in distribution also holds. (Proof in Section B.8.) 

## **Theorem 2: Sufficiency of directional tests** 

Equation (4) is a valid statistical test for Equation (3) as 

**==> picture [237 x 62] intentionally omitted <==**

The assumptions required in the proof of thm. 2 hold for classical consistent univariate tests _𝑇_ such as the ones presented in the following Section 4.2. 

**Figure 4.** Examples of distributions living on the surface of the sphere with varying Sobolev smoothness coefficients _𝛼_ . As per thm. 5, the greater _𝛼_ is, the more global will be the impact of SIGReg for a given number of directions _𝑀_ . Practically, this represents the distribution of the encoder’s output. Because the target density (isotropic Gaussian) is smooth, the _𝛼_ coeffcients of the embedding will quickly grow hereby making SIGReg (def. 2) immune to the curse of dimensionality. 

## **4.2 SIGReg: Sketching the Epps-Pulley Test is Stable and Scalable** 

Our proposed regularizer–coined Sketched Isotropic Gaussian Regularization (SIGReg)–follows directly from thm. 2 using any statistical test _𝑇_ targeted towards the isotropic Gaussian, illustrated in Figures 2 and 5, and formalized below. 

## **Definition 2: SIGReg (PyTorch code in algorithm 1)** 

**==> picture [238 x 10] intentionally omitted <==**

**==> picture [229 x 38] intentionally omitted <==**

where we recommend the Epps-Pulley test (Section 4.2.3) for _𝑇_ . 

We replace the maximum over _𝒂_ ∈ A in thm. 2 by an average in (SIGReg) to avoid sparse gradient over the directions in A. We now delve on the choice of _𝑇_ for which we compare well-known candidate tests in the field of statistics that are categorized into (i) moment based (Section 4.2.1), (ii) CDF based (Section 4.2.2), and (iii) CF based (Section 4.2.3) statistics–ultimately justifying our choice of the Epps-Pulley statistic. 

## **4.2.1 Moments are Unstable and Insufficient** 

The first family of statistics we consider are moment-based. Taking the standard Gaussian as an instanciation for the moments, we can define the Jarque-Bera [Jarque and Bera, 1980] test that compares the third and fourth moments, 

7 

**LeJEPA:** Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | **Sec 4: SIGReg** | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [520 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
vcreg ext j arque b eta watson cramer v on m ises anderson d arling epps p ulley<br>1.0<br>2 i:9<br>150 i:8<br>i:7<br>i:6 0.6<br>100 0 i:5<br>i:4<br>50 i:3 0.2<br>− 2 i:2<br>i:1<br>0 i:0<br>− 2 0 2 − 2 0 2 − 5 0 i:0 i:1 i:2 i:3 i:4 i:5 i:6 i:7 i:8 i:9<br>x 1 (blue) — x 2 (red) x 1 ⟨x, ai⟩<br>) ⟩i ℓ 2<br>Count x 2 ⟨ ( x, ap ℓ and1<br>**----- End of picture text -----**<br>


**Figure 5.** Constructed data density with “X” distribution whose marginals are standard Gaussian and whose covariance is identity ( **left densities** ). Applying _𝑀_ = 10 projections on the half circle directions produces 10 univariate distributions that can be compared against a standard Gaussian ( **left** ) using any preferred statistic from Section 4.2. The appropriate direction is able to capture the degenerate distribution of the data hereby creating a spike in the statistic value. 

i.e., skewness and kurtosis, as 

**==> picture [243 x 38] intentionally omitted <==**

where skew[�] is the skewness computed from the data as _𝑛_ 1 � _𝑛𝑖_ =1[(] _[𝑥][𝑖]_[−ˆ] _[𝜇]_[)] 3 _𝑛_ 1 � _𝑛𝑖_ =1[(] _[𝑥][𝑖]_[−ˆ] _[𝜇]_[)] 4 _𝜎_ ˆ[3] and kurt[�] is the kurtosis _𝜎_ ˆ[4] . Typically, the (Jarque-Bera) test is used to see if a density follows a Gaussian distribution of any mean and variance–hence it only looks at moments 3 and 4. In our case we aim for a standard Gaussian test and thus add the usual statistics on the first two moments, leading to the extended test 

**==> picture [232 x 41] intentionally omitted <==**

The (Extended Jarque-Bera) acts as a moment matching problem over the first four moments. Such moment matching methods have proven powerful not only for statistical tests but also as mean to learn parametric and nonparametric models of data. 

**The Stability and Identifiability Conundrum.** We now explain why moment-based tests–albeit powerful–will not be suited for LeJEPA. The _𝑘[𝑡ℎ]_ of a distribution _𝑃_ is denoted as _𝑚𝑘_ ( _𝑃_ ). The first observation is that wellbehaved distributions abiding the Carleman’s condition ∞ � _𝑘_ =1 _[𝑚]_[2] _[𝑘]_[(] _[𝑄]_[)][−][1][/(][2] _[𝑘]_[)][=][ ∞][[Carleman, 1926], such as the Gaus-] sian, or for distributions with finite interval [Hausdorff, 1923] are uniquely determined by their moments. However, using a finite number of moments creates the following non-identifiability issue which well-known in statistics and often used as a motivation to use _all_ moments [Lehmann and Romano, 2005]. 

**Theorem 3: Insufficiency of K Moments** 

**==> picture [248 x 85] intentionally omitted <==**

Hence thm. 3 prescribes us with the guideline to employ as large _𝐾_ as possible to remove collapsed shortcut solution by making sure our distribution matching is accurate. Yet, doing so leads to unstable gradient-based training due to the gradient norm scaling as _𝑂_ ( _𝑘_ ), and the variance of Monte Carlo gradient estimates growing as _𝑂_ ( _𝑘_[2] _𝑚_ 2( _𝑘_ −1)) for the _𝑘_ -th moment since ��∇ _𝜃𝑚𝑘_ ( _𝑃_ _**𝜽**_ ( _𝒂_ )[)] �� = ∥E� _𝑘_ ( _𝒂_[⊤] _𝑓_ _**𝜽**_ ( _𝒙_ )) _[𝑘]_[−][1] _𝒂_[⊤] _𝐽 𝑓_ _**𝜽**_ ( _𝒙_ )�∥, with _𝐽 𝑓_ _**𝜽**_ ( _𝒙_ ) ∈ R _[𝐾]_[×] _[𝑃]_ the Jacobian matrix–hereby creating an impractical situation where training stability and identifiability can not be achieved simultaneously. 

## **4.2.2 Cumulative Density Functions are Impractical** 

The second family of tests acts upon the CDF. Because those tests require sorting, let’s denote the _𝑘_[th] order-statistics of _𝑁_ samples by _𝑥𝑘_ : _𝑁_ . Two highly standard tests are quadratic Empirical Density Function statistics with different weighting known as Cramér-von Mises [Cramér, 1928, Von Mises, 1981] and Anderson Darling [Anderson and Darling, 1952], and given by 

**==> picture [226 x 57] intentionally omitted <==**

where _𝑤_ ( _𝑥_ ) is a weighting function. Adding the _𝑈_[2] statistics on top of Equation (Cramér-von Mises) recovers the 

8 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | **Sec 4: SIGReg** | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [508 x 210] intentionally omitted <==**

**----- Start of picture text -----**<br>
original data VCReg ExtendedJarqueBera CramerVonMises Watson AndersonDarling EppsPulley<br>4<br>E ® eo ‘e ° ®e © e0 8<br>2<br>0<br>X oa © oo I Ted o® %ea’® © oq °<br>− 2 R R S= ® : v.. ° SERAR3 oBI3 krPE p=Bi% °<br>os ‘ “eo < .<br>0 5 0 5 0 5 0 5 0 5 0 5 0 5<br>dim 1 dim 1 dim 1 dim 1 dim 1 dim 1 dim 1<br>4<br>2<br>20 ° °Y ® ° H , i . ° T= - 4 $l .<br>− 02 ©% So °°2(od » K 4 :%°c Sd J 8) ® 8o, oN<br>N 8% o “a °° LP ° ® de ° ° 8% ® °°,<br>− 4<br>− 2 . 5 0 . 0 2 . 5 − 2 . 5 0 . 0 2 . 5 − 2 . 5 0 . 0 2 . 5 − 2 . 5 0 . 0 2 . 5 − 2 . 5 0 . 0 2 . 5 − 2 . 5 0 . 0 2 . 5 − 2 . 5 0 . 0 2 . 5<br>dim 3 dim 3 dim 3 dim 3 dim 3 dim 3 dim 3<br>2<br>dim<br>4<br>dim<br>**----- End of picture text -----**<br>


**Figure 6.** _𝑁_ = 100 samples are drawn from a 1024-dimensional standard Gaussian, and the first 2 coordinates are altered to produce the “X” distribution from Figure 5 ( **left-most column** ). For each statistic ( **all other columns** ), we perform gradient descent on the samples to minimize their value, at each iteration step with sample _𝑀_ = 10 random directions to evaluate SIGReg (recall def. 2). We obtain that albeit this is a high-dimensional distribution with limited number of samples, SIGReg is able to capture the degenerate subspace and adapt the data accordingly to match an isotropic Gaussian distribution. Additional figures with varying dimensions and number of 1d projections are provided in Figure 16. 

Watson test [Watson, 1961] 

**==> picture [197 x 25] intentionally omitted <==**

We do not consider the Kolmogorov-Smirnov test [Kolmogorov, 1933] as it employs the _ℓ_ ∞-norm instead of the _ℓ_ 2-norm hereby producing sparse gradients. Another common test is the Shapiro-Wilk test [Shapiro and Wilk, 1965] which we found to be unstable in practice–details are provided in Section E. 

**Lack of Scalability and Differentiability.** CDF-based tests require sorting that have been highly optimized, e.g., with the 𝒪( _𝑁_ log( _𝑁_ )) Quicksort algorithm [Hoare, 1962] but that nonetheless breaks the embarrassingly parallel nature of SGD–especially on multi-GPU [Tanasic et al., 2013, Maltenberger et al., 2022] due to synchronization requirements. Moreover, these tests involve non-differentiable operations (sorting and order statistics), making them unsuitable for gradient-based optimization without relaxations [Cuturi et al., 2019, Grover et al., 2019, Petersen et al., 2022]. While there exists intricate sketching solutions [Dunning and Ertl, 2019, Masson et al., 2019, Dunning, 2021], each of those solutions introduce numerous additional hyper-parameters–going against our first motivation for LeJEPA. 

## **4.2.3 Characteristic Functions are Stable, Scalable and Identifiable** 

The third family of tests is concerned with Empirical Characteristic Functions (ECF) which are the Fourier transform of the density function. The Epps–Pulley test [Epps and Pulley, 1983] is one of the most popular test and simply compares in weighted _ℓ_ 2-norm the ECF of the data against a target CF 

**==> picture [234 x 25] intentionally omitted <==**

The first crucial observation is that the ECF being defined as _𝜙_[ˆ] _𝑋_ ( _𝑡_ ) = ~~-~~ _𝑛_[1] 2 _𝑛𝑗_ =1 _[𝑒][𝑖𝑡𝑋][𝑗]_[is naturally differentiable and easily] computed in distributed settings via efficient `all_reduce` operations, as the ECF is a simple average of complex exponentials. The weight function is typically Gaussian, such as _𝑤_ ( _𝑡_ ) = _𝑒_[−] _[𝑡]_[2][/] _[𝜎]_[2] with _𝜎_ commonly set to 1. 

Other tests, e.g., based on the Entropy [Székely and Rizzo, 2005] are not considered here as they require numerous additional design choices for the univariate Entropy estimation [Silverman, 2018, Beirlant et al., 1997], e.g., using kernels [Joe, 1989], or M-estimators [Miller, 2003]. 

**Epps-Pulley has bounded loss, gradient and curvature.** We now consider the remaining two families of tests: moment-based and CF-based. First, recall that moments are polynomial in the data and with extreme growth rate 

9 

**LeJEPA:** Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | **Sec 4: SIGReg** | Sec 5: LeJEPA | Sec 6: Experiments 

**Algorithm 1.** SIGReg with Epps-Pulley statistic with DDP support and 𝒪( _𝑁_ ) time and memory complexity. x is a (N, K) tensor, num_slices is |A| in def. 2, ‘global_step‘ is used for sync. sampling across GPUs and can be omited for single-GPU training. An optimized implementation with caching is also provided in our official codebase, computation times provided in Table 6. 

|**def**|SIGReg( x ,<br>global_step ,<br>num_slices =256) :|SIGReg( x ,<br>global_step ,<br>num_slices =256) :|SIGReg( x ,<br>global_step ,<br>num_slices =256) :|||
|---|---|---|---|---|---|
||_# s l i c e_<br>_sampling −−synced _|_across devices _|||_−−_|
||dev = **dict**( device=x . device|)||||
||g = torch . Generator (∗∗dev )|||||
||g . manual_seed ( global_step )|||||
||proj_shape = ( x . size (1) , num_slices )|||||
||A = torch . randn ( proj_shape|,<br>generator=g ,|||∗∗dev )|
||A /= A. norm ( p=2 , dim=0)|||||
||_# −−Epps−Pulley_<br>_s t a t ._<br>_see _|_Sec ._<br>_4.3_|_f o r_|_a_|_l t . −−_|
||_# i n t e g r a t i o n_<br>_points_|||||
||t = torch . linspace ( −5 , 5 ,|17 , ∗∗dev )||||
||_# t h e o r e t i c a l CF f o r N(0 ,_|_1) and Gauss . _||_window_||
||exp_f = torch . exp( −0.5 ∗<br>t|∗∗2)||||
||_# empirical CF −−gathered _|_across devices _|||_−−_|
||x_t = ( x @ A) . unsqueeze (2)|∗<br>t<br>_# (N, M, T)_||||
||ecf = (1 j<br>∗x_t ) . exp ( ) .mean(0)|||||
||ecf = all_reduce ( ecf , op="AVG")|||||
||_# weighted L2 distance_|||||
||err = ( ecf −exp_f ) .**abs**( ) .|square ( ) . mul ( exp_f )||||
||N = x . size (0) ∗world_size|||||
||T = torch . trapz ( err ,<br>t ,<br>dim=1) ∗N|||||
||**return** T|||||



for higher moment–assuming they even exist. Even for well-behaved distributions, raising values to a power of _𝑘_ can quickly lead to exploding gradients. This comes in sharp contrast with the ECF which is always bounded and with bounded gradients for any input distribution for the projected samples _𝑧𝑖_ = _𝒂_[⊤] _𝑓𝜃_ ( _𝒙𝑛_ ), _𝑛_ = 1 _, . . . , 𝑁_ . 

**Theorem 4: Stability of Epps-Pulley Test** 

**==> picture [248 x 77] intentionally omitted <==**

By the chain rule, thm. 4 directly gives ∥∇ _𝜃𝐸𝑃_ ( **a** )∥≤ 4 _𝜎_[2] _𝑁 𝑁_ � _𝑖_ =1[∥] **[a]**[⊤][∇] _[𝜃][𝑓][𝜃]_[(] **[x]** _[𝑖]_[)∥][, providing stable gradients.][The lim-] itations of moment-based and CDF-based tests coupled with thm. 4 justifies our choice of the (Epps–Pulley): (i) DDP-friendly and scalable, (ii) uniformly bounded gradients and curvature regardless of input distribution, and (iii) hyper-parameter free implementation. Lastly, we highlight that _our implementation has a linear memory and computational complexity of_ 𝒪( _𝑁_ ) _, with 𝑁 the minibatch size_ . The implementation of SIGReg using that statistical test is provided in algorithm 1, along with computation times of the forward-backward pass in Table 6. 

**==> picture [252 x 176] intentionally omitted <==**

**----- Start of picture text -----**<br>
1500 β =  − 2 . 79 β =  − 285 . 91<br>R [2] = 0 . 87 R [2] = 0 . 96<br>1000<br>500<br>random<br>fixed<br>0<br>10 [2] 10 [3]<br>M (log-scale)<br>�<br>)<br>=1<br>N n<br>}<br>)<br>n<br>x<br>(<br>θ<br>f<br>⊤<br>a<br>{<br>(<br>T<br>�<br>a<br>E<br>**----- End of picture text -----**<br>


**Figure 7.** Expected directional statistic at the end of training ( **y-axis** ) for varying _𝑀_ (number of directions used at each training step, **x-axis** ). The _𝑀_ directions are either resampled ( **green** ) or kept fixed ( **blue** ) at each training step. While for fixed directions we benefit from thm. 5 bound where increasing _𝑀_ reduces the overall expected loss, being able to resample at every step provides significant coverage boost for free. 

study the requirements on the number of directions (|A|) for (2) to be effective in high-dimension. 

## **4.3 How SIGReg Beats the Curse of Dimensionality** 

This last section seeks to characterize how many slices in A one must sample for (SIGReg) to be an effective statistical test. That design is crucial if we hope for LeJEPA to successfully converge towards isotropic Gaussian embeddings. 

## **Smoothness Beats the Curse of Dimensionality** 

Our first argument arguing for a favorable scaling of |A| with the embedding dimension _𝐾_ relies on the smoothness of _𝑃_ _**𝜽**_ as measured by its Sobolev regularity _𝛼_ [Adams and Fournier, 2003]. We formalize below a bound on the directional test from Equation (3) over all possible directions _𝒂_ when the test statistic is minimized over |A| = _𝑀_ directions. While we provide bounds on the expected discrepancy over random directions _𝒂_ when the EP test is satisfied (equals zero) on a finite set of directions, the provided proof includes the case of moment-based and CDF-based tests as well. 

As a last step before introducing LeJEPA, we ought to 

10 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | **Sec 5: LeJEPA** | Sec 6: Experiments 

## **Theorem 5: Unified Error Bounds** 

**==> picture [239 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
Let 𝑝 𝜽 ∈ 𝐻 [𝛼] (R [𝐾] ), 𝒂 ∼𝒰(𝒮 [𝐾] [−][1] ), and (Epps–Pulley)= 0, i.e.,<br>𝑃𝜃 [(] [a] [)] =  𝑄 [(] [a] [)] ,  ∀ 𝒂 ∈ A, then<br>2<br>E 𝒂 �� 𝜑𝑎 ( 𝑡 ) − 𝜑 𝒩 ( 𝑡 )��  𝑑𝑡 ≤ 𝐶 ( 𝐾, 𝛼 )|A| [−][2] [𝛼] [/(] [𝐾] [−][1][)]<br>�∫R �<br>∞<br>2<br>× ∫0 �� 𝜑 ·( 𝑟 ) − 𝜑 𝒩 ( 𝑟 )�� 𝐻 [𝛼] (𝒮 [𝐾] [−][1] ) [𝑑𝑟,]<br>(Proof in Section B.10.)<br>**----- End of picture text -----**<br>


As |A| →∞, the bound decays as |A|[−][2] _[𝛼]_[/(] _[𝐾]_[−][1][)] , showing that |A| = _𝑂_ ( _𝐾_ ) directions suffice for _𝜖_ -approximation when _𝛼_ is large. Some examples of embedding densities with varying _𝛼_ are provided in Figure 4. The following statement characterizes how the _𝑀_ directions actually constrain the entire space as a function of _𝛼_ . The constant 2[2] _[𝛼] 𝜋_[(] _[𝐾]_[−][1][)/][2] Γ( _𝛼_ + _[𝐾]_ 2[−][1][)] _𝐶_ ( _𝐾, 𝛼_ ) = ( _𝐾_ −1)Γ( _𝛼_ )Γ( _[𝐾]_ 2[−][1][)] is visualized in Figure 15 (left) depicting how _𝛼_ and |A| interact. In words, we obtain that thanks to the natural smoothness of DN–either stemming from the architecture or the implicit and explicit regularizers used during training–applying SIGReg on |A| directions can be sufficient to tightly constrain the entire space. We note that considering the worst case over _𝒂_ or using low-discrepancy sequences for _𝒂_ does not impact the asymptotic bounds, details provided in Section D. 

## **SGD Beats the Curse of Dimensionality** 

Our second argument leverages the iterative nature of DN training. Although we may use only |A| to be a few hundreds, the cumulative number of sampled directions grows linearly with training time. This resampling effect (illustrated in Figure 7, bottom) enables rapid convergence. Even small |A| achieves tight distributional matching compared to keeping the set A fixed throughout minibatches (recall thm. 5). Our experiments show that even with |A| as low as 16 can easily outperform a fixed set with |A| of order of thousands thanks to the compounding effect of resampling at each minibatch. 

## **Empirical Validation on Synthetic Data** 

We conclude this section with a controlled experiment applying (SIGReg) with gradient-based training to produce isotropic embeddings. In this setup, we directly consider embeddings _𝒁_ which we will differentiate and optimized to minimize (SIGReg). By directly optimizing the embeddings we are able to observe the impact of the loss without any possible constraint and regularization that would come from the architecture. We sample _𝑁_ i.i.d. samples _𝒙𝑛_ in a _𝐷_ -dimensional space. This sampling is based on an isotropic Gaussian distribution–but the first 

**Algorithm 2.** LeJEPA implementation–works out-of-the-box on any dataset, with DDP, with any backbone, e.g., torchvision or timm. For non-ViT architectures (e.g., ResNet), set global_views = all_views. We use bs for the minibatch size, SIGReg is from algorithm 1. 

**def** LeJEPA( global_views , all_views , lambd ) : " " " global_views and all_views are l i s t s of tensors , lambd i s a scalar " " " _# embedding of global views_ g_emb = forward ( torch . cat ( glob_views ) ) _# embedding of l o c a l views # i f resnet : skip with a_emb=g_emb_ a_emb = forward ( torch . cat ( all_views ) ) _# LeJEPA loss_ centers = g_emb . view ( −1 , bs , K) .mean(0) a_emb = a_emb . view ( −1 , bs , K) sim = ( centers −a_emb) . square ( ) .mean ( ) sigreg = mean(SIGReg(emb, global_step ) **for** emb **in** a_emb) **return** (1−lambd ) ∗sim + lambd∗sigreg 

two dimensions are again set to the adversarial “X” shape. That is, among the _𝐷_ dimensions, only two must be transformed as all the other ones already obey the isotropic Gaussian target distribution. We then make the samples _𝒙𝑛_ differentiable and optimize then to minimize the value of the different statistical tests compute on _𝑀_ random _𝑀_ random directions. Those directions are resampled after each gradient step–which follows the procedure we will employ in LeJEPA. We present the results in Figure 6 demonstrating that even in challenging case, i.e., _𝐷_ = 512 and _𝑀_ = 16, SIGReg is able to detect the two degenerate dimensions and unfold them back to how they should look like under the target distribution. 

## **5 LeJEPA: Stable and Scalable Implementation** 

Having established that isotropic Gaussians are the optimal embedding distribution for foundation models (Section 3) and introduced SIGReg to achieve this distribution (def. 2), we now present the complete LeJEPA framework. We first evaluate candidate statistical tests (Sections 4.2.1 and 4.2.2) and identify characteristic function-based tests as optimal for gradient-based training (Section 4.2.3). The full LeJEPA implementation follows in Section 5.1. 

## **5.1 LeJEPA: SIGReg + Prediction Loss** 

We now discuss the implementation of LeJEPA starting with SIGReg and followed by the prediction and total losses. 

**The SIGReg Loss.** We chose (Epps–Pulley) for its provable boundedness (thm. 4) and its scalability. Its implementation follows exactly the equation except for the integrate which is estimated using a quadrature approximation. We 

11 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | **Sec 6: Experiments** 

find that the simple trapezoidal quadrature rule is sufficient even with as few knots as 17, as ablated in Figure 20. In particular, we leverage the symmetry of the integrand to double the number of knots for free, see the official code. On the other hand, the use of minibatches introduces a bias vanishing at rate 𝒪(1/ _𝑁_ ), as formalized below. 

**Theorem 6: Vanishing gradient bias** 

**==> picture [248 x 81] intentionally omitted <==**

Hence, the gradients we obtain from using (Epps–Pulley) are biased by an explicit 𝒪(1/ _𝑁_ ) term. We found this bias to be minimal and not a concern even for minibatches as small as 16. Unbiased alternatives include using U- statistic debiasing of | _𝜙𝜃_ |[2] or sample splitting, which we do not explore in this study. Our final implementation of the SIGReg term with Epps-Pulley statistic is provided in algorithm 1. 

**The Prediction Loss.** To standardize notations, we adopt the DINO [Caron et al., 2021] setup of generating _𝑉𝑔_ global views and _𝑉𝑙_ local views, leading to a total of _𝑉_ = _𝑉𝑔_ + _𝑉𝑙_ views. We set the first 1 _, . . . , 𝑉𝑔_ indices of each _𝒛𝑛,𝑣_ as the global views. For the cases without local views, simply set _𝑉𝑙_ = 0. The prediction loss is then given by having all views predict the global views as 

**==> picture [227 x 113] intentionally omitted <==**

where we denote _**𝝁** 𝑛_ ≜ _𝑉_ 1 _𝑔_ � _𝑉𝑣_ = _𝑔_ 1 _[𝒛][𝑛,𝑣]_[,][the][Equation][(5)][to] Equation (6) derivations are detailed in Section B.6. **LeJEPA Loss.** The final total loss simply combines the above prediction loss along with SIGReg on each views as per 

**==> picture [243 x 67] intentionally omitted <==**

We present (LeJEPA)’s implementation in algorithm 2. Altogether, the entire implementation–besides the usual model definitions, optimizers, and data loaders–only takes a few dozens lines in PyTorch (algorithms 1 and 2). The absence of prototypes, stop-gradients, and teacher-student networks makes (LeJEPA) appealing as it only contains one hyperparameter, _𝜆_ , balancing the trade-off between the prediction and isotropic Gaussian terms. 

## **5.2 Relation to Prior Work** 

Prior to presenting our experiments (Section 6), we conclude by discussing how our proposed LeJEPA and SIGReg objective relate to existing frameworks in the literature. 

While there is no existing solution employing such slicing and distribution matching for JEPAs, there exists similar pipelines for generative models and optimal transport. Notably, the Sliced Score Matching [Song et al., 2020] proposes to leverage univariate slicing of the space to ease the estimation of a density for generative models. In a similar vein, the sliced Wasserstein distance [Bonneel et al., 2015, Nguyen and Ho, 2023] uses such strategy to speed up and improve optimal transport. Furthermore, when the integral of the (Epps–Pulley) test is computed exactly, as opposed to our quadrature, each slice loss value recovers the kernel MMD [Sriperumbudur et al., 2010, Gretton et al., 2012, Chwialkowski et al., 2016] measuring the distance between two distributions–albeit with a quadratic complexity. Lastly, it is possible to recover some existing SSL frameworks in the limit by employing LeJEPA with a particular test–instead of the preferred (Epps–Pulley). For example, Setting _𝑇_ ({ _𝑥𝑛_ } _[𝐵] 𝑛_ =1[)][ =][ mean][({] _[𝑥][𝑛]_[}] _[𝐵] 𝑛_ =1[)][2][ + (][std][({] _[𝑥][𝑛]_[}] _[𝐵] 𝑛_ =1[) −][1][)][2] and using that _𝑇_ with SIGReg in LeJEPA recovers the VICReg SSL method in the limit of large number of slices. In fact, SIGReg will enforce in expectation that E[ **Z** ] = **0** and Cov( **Z** ) = **I** _𝑑_ , where **I** _𝑑_ denotes the _𝑑_ × _𝑑_ identity matrix–derivations provided in Section B.14. And since our invariance term is simply the _ℓ_ 2 distance between the views’ embeddings, LeJEPA recovers VICReg for this degenerate statistical test. Based on thm. 3, we however strongly advocate against such a setting as it would lead to shortcut solutions–a phenomenon already observed in VICReg. 

## **6 LeJEPA: Empirical Validation** 

We now use the LeJEPA implementation described in Section 5.1 to demonstrate its effectiveness through comprehensive experiments. We show that LeJEPA: (i) trains reliably across diverse architectures and datasets (Section 6.1), (ii) provides an informative training loss for model selection (Section 6.2), (iii) outperforms frontier vision models on small-scale in-domain pretraining (Section 6.3), (iv) scales successfully to nearly 1 billion parameters on ImageNet-1k (Section 6.4), and (v) learns rich 

12 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | **Sec 6: Experiments** 

**==> picture [252 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
ResNet50-inet100 acc. vs λ<br>86<br>2 Views<br>4 Views<br>84 8 Views<br>82<br>80<br>78<br>76<br>74<br>10 [−] [3] 10 [−] [2] 10 [−] [1]<br>λ (log-scale)<br>(%)<br>acc.<br>top1<br>**----- End of picture text -----**<br>


**Figure 8.** Inet100 with 400 pretraining epochs and resnet50 backbone. We depict linear probe performances as a function of _𝜆_ and the number of views _𝑉_ (recall (LeJEPA)). We observe that performances are stable over _𝜆_ –with **peak performance obtain by slightly adjust** _𝜆_ **proportionally to the number of views** . The corresponding performance values are provided in Table 7. 

semantic segmentation features without explicit supervision. 

## **6.1 LeJEPA’s Stability Across Hyper-Parameters and Architectures** 

We now demonstrate LeJEPA’s stability across hyperparameters, architectures, and experimental setups. Additional cross-domain stability results are presented in Section 6.3. 

**Stability across standard hyperparameters.** We begin by evaluating LeJEPA on ImageNet-100 and ImageNet1K. On ImageNet-100, we train a ResNet-50 and vary the number of views and the loss weighting _𝜆_ (Figure 8). Performance remains stable across both dimensions, leading us to recommend _𝜆_ = 0 _._ 05 as a robust default. On ImageNet-1K, we train a ViT-Large/14 and explore batch size, as well as the number of global ( _𝑉_ g) and local ( _𝑉_ l) views (Table 1b). We find that the configuration commonly used in prior work ( _𝑉_ g = 2 _, 𝑉_ l = 8) transfers well to LeJEPA. Notably, LeJEPA achieves competitive performance with batch sizes as small as 128 on ImageNet-1K (Table 1c), suggesting reduced memory requirements compared to existing methods. _We thus recommend to use 𝜆_ = 0 _._ 05 _, 𝑉_ g = 2 _, 𝑉_ l = 8 _, and batch size_ ≥ 128 _as starting points_ . 

**Stability across Epps-Pulley hyperparameters.** We next examine hyperparameters specific to LeJEPA: the number of slices |𝒜| in SIGReg, the integration domain for the Epps-Pulley test (Epps–Pulley), and the number of quadrature points for numerical integration. Table 1a shows ablations on ImageNet-1K with ViT-Large/14. Both the integration domain and number of quadrature points have negligible impact on performance. This is expected: since the characteristic function is accurate at zero, the 

**Table 1.** ViT/Large-14, on inet1k pretraining for 100 epochs and evaluated with frozen backbone linear probing (top1 accuracy, %). **LeJEPA’s performance is stable across all its hyperparameters** and while some may slightly improve performance, e.g., the number of slices |A| and the projector sizes, none of the choices lead to a catastrophic collapse. 

**(a)** (Epps–Pulley) **parameters** 

|integration<br>num_slices|confg/bstat_n_points<br>5<br>17<br>41|
|---|---|
|[−1_,_1]<br>512<br>2048<br>[−3_,_3]<br>512<br>2048<br>[−5_,_5]<br>512<br>2048|71.82<br>72.13<br>72.04<br>72.88<br>72.30<br>72.69<br>73.95<br>74.16<br>74.04<br>75.02<br>74.68<br>74.77<br>73.71<br>74.21<br>74.15<br>74.50<br>74.80<br>74.77|



## **(b) Number of local/global views** 

|||# global_views (_𝑉_g)<br>1<br>2<br>4<br># views (_𝑉_=_𝑉_g+_𝑉_l)<br>4<br>53.06<br>72.26<br>–<br>6<br>58.65<br>73.07<br>73.68<br>8<br>64.46<br>74.24<br>73.94<br>10<br>68.97<br>74.06<br>75.08<br>**(c) Mini-batch size**<br>batch_size<br>128<br>256<br>512<br>1024<br>72.20<br>74.15<br>74.72<br>74.07<br>**(d) Embedding/Projector dim.**<br>num_slices<br>1024<br>4096<br>emb. dim.<br>512<br>2048<br>512<br>2048<br>proj. dim.<br>64<br>75.29<br>75.32<br>75.50<br>75.65<br>128<br>74.77<br>75.09<br>75.26<br>75.47<br>256<br>74.56<br>74.66<br>75.08<br>75.02<br>512<br>73.94<br>74.11<br>74.81<br>74.65<br>1024<br>73.65<br>73.94<br>74.71<br>74.79<br>**(e) Register tokens**|
|---|---|---|
||||
||||
|reg_tokens<br>0<br>1<br>2<br>4<br>8<br>num_slices|||
|1024<br>75.14<br>75.18<br>75.08<br>75.34<br>75.23<br>4096<br>75.61<br>75.58<br>75.67<br>75.63<br>75.84|||



moments of the distribution are well-characterized even with a modest integration range. The number of slices |𝒜| has a modest effect—while more slices slightly improve performance, even 512 slices yield competitive results. _We thus recommend to use 17 integration points, an integration domain of_ [−5 _,_ 5] _, and 1024 slices as starting points_ . 

13 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | **Sec 6: Experiments** 

Inet10 – LeJEPA pretrained, frozen backbone, linear eval – 50 architectures ( < 20M params.) 

**==> picture [513 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
95.0 maxvit˙rmlp˙pico˙rw˙256 resnet14t resnet26d resnet26tmaxvit˙nano˙rw˙256maxvit˙rmlp˙nano˙rw˙256 Model Family<br>maxvit˙pico˙rw˙256 convnext<br>94.5 resnet26 efficientnet<br>resnet18d inception<br>94.0 resnext26ts<br>levit<br>93.5 efficientnet˙b0˙g8˙gn resnet32ts maxvit<br>93.0 efficientnet˙b0˙gn vit˙pe˙core˙tiny˙patch16˙384 levit˙192 levit˙conv˙256 maxxvitresnet<br>92.5 inception˙next˙atto levit˙128 resnetblur18 vit<br>convnext˙atto˙ols convnext˙femto<br>92.0 convnextv2˙femto levit˙128s convnext˙pico˙ols convnext˙nano resnet33ts<br>convnext˙zepto˙rms<br>91.5 convnext˙zepto˙rms˙olsconvnext˙atto˙rms convnext ˙ nano ˙ ols<br>2M 5M 8M 10M 12M 15M 18M<br>Parameters (Millions)<br>Top-1 Accuracy (%)<br>**----- End of picture text -----**<br>


**Figure 9.** INet10 pretraining and frozen backbone linear evaluation across 50 timm models using LeJEPA out of the box. We cross-validate the learning rate and weight-decay. While there is a small variation between the best and worst performing model, we clearly see that **across** 50 **models spanning** 8 **families, LeJEPA is able to produce non-trivial representations able to solve the downstream task at SOTA levels** . 

**Stability across architectures.** A key advantage of LeJEPA over recent methods (e.g., ĲEPA, DINOv2) is its architecture-agnostic design. While most modern selfsupervised methods are tailored to Vision Transformers, LeJEPA works across diverse architecture families without modification. To validate this claim, we pretrain approximately 50 architectures from 8 different families on ImageNet-10, selecting all models in the timm library with fewer than 20M parameters. All models are able to learn high-quality representations reaching between 91.5% to 95% top 1 accuracy with frozen backbone linear probing. It seems that models performing well in supervised learning setups are also the ones to favor for LeJEPA, such as resnets and ViTs. _We thus recommend to use standard architectures such as ResNets and ViTs over specialized models like EfficientNet as stating point._ 

**Removal of popular heuristics.** In addition to providing reliable performance across models and datasets, LeJEPA’s provable construction enables us to _remove_ many heuristics traditionally used to prevent collapse. First, prior work has shown both empirically and theoretically that predictors in image JEPA (without asymmetric information) and teacher-student architectures serve primarily to prevent collapse [Grill et al., 2020, Jing et al., 2021, Tian et al., 2021, Caron et al., 2021, Chen et al., 2021]. Removing these components produces _collapsed_ encoders, i.e., with performances at chance-level. Thanks to LeJEPA’s SIGReg loss, we can remove both the predictor and teacher-student architecture without suffering from collapse, as shown in Table 4. While a teacher-student configuration does provide a small performance boost for ViT models—consistent with observations in supervised learning via Stochastic 

Weight Averaging [Izmailov et al., 2019]—it is not necessary to prevent collapse. In our setup, we apply SWA on the encoder producing _𝜇_ in Equation (6). Second, recent work demonstrated that register tokens are needed to prevent training instabilities in vision models [Oquab et al., 2023, Siméoni et al., 2025, Darcet et al., 2023]. We show in Table 1 that such instabilities likely stem from poorly conditioned training objectives. In contrast, LeJEPA _does not_ require register tokens and achieves stable performance with or without them. _We thus recommend training without a predictor or register tokens, and optionally applying SWA with ViTs for a possible performance gain._ 

## **Experiment Details 1** 

We strive for **simplicity** and thus adopt a unified pretraining pipeline. The following parameters apply to _all_ experiments and figures unless stated otherwise in the corresponding caption and come from Section 6.1: 

- LeJEPA’s implementation is given in algorithm 2 with hyperparameter _𝜆_ 

- All backbones are from timm and all optimizers/schedulers are from PyTorch without modifications 

- We employ eight views ( _𝑉_ = 8) containing two global views ( _𝑉_ g = 2) with resolution 224x224 and 96x96 for the local views 

- AdamW optimizer with lr ∈{5 _𝑒_ − 3 _,_ 5 _𝑒_ − 4} and wd ∈ {1 _𝑒_ − 1 _,_ 1 _𝑒_ − 2 _,_ 1 _𝑒_ − 5}–no scheduler on weight-decay, standard linear warm-up cosine-annealing for lr 

## **6.2 LeJEPA’s Training Loss is Informative of Downstream Performance** 

A major challenge in SSL pretraining is the lack of reliable signals conveying the quality of the learned representation. As a result, it is common to monitor a supervised 

14 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | **Sec 6: Experiments** 

**==> picture [512 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
resnet50 - galaxy10 79 . 21 resnet50 - inet10 94 . 03 ViT/base-14 - inet1k 72 . 90<br>10 [0]<br>64 . 13 80 . 15 54 . 73<br>10 [−] [1]<br>49 . 06 66 . 27 36 . 56<br>10 [−] [1]<br>da Ak \ feNe | LANE<br>33 . 98 52 . 39 18 . 39<br>10 [−] [1]<br>10 [−] [2]<br>18 . 91 38 . 51 0 . 23<br>10 [0] 10 [1] 10 [0] 10 [1] 10 [0] 10 [1]<br>SIGReg loss (log-scale) SIGReg loss (log-scale) SIGReg loss (log-scale)<br>(log-scale) (log-scale) (log-scale)<br>loss Accuracy loss Accuracy loss Accuracy<br>Pred. Pred. Pred.<br>**----- End of picture text -----**<br>


**Figure 10.** (SIGReg, prediction loss) 2 _𝑑_ -plane with downstream task accuracy shown with colors from **blue** (low) to **red** (high). We clearly observe that within this plane, **there exists trade-off fronts between the two terms of LeJEPA producing similar downstream performance** corresponding to different values of _𝜆_ . Yet, those fronts are linear and pointed towards the lower left corner, i.e., LeJEPA’s training loss informs of downstream test performance across models and datasets ( **columns** ). Additional models and datasets provided in Figure 21. 

**==> picture [242 x 170] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 . 0 A<br>0 . 8<br>0 . 6<br>0 . 4<br>resnet18 f lowers102: 0.60 → 0 . 95<br>resnet50 g alaxy10: 0.85 → 0 . 98<br>0 . 2 resnet50 inet10: 0.81 → 0 . 99<br>ViT/base-8 i net1k: 0.88 → 0 . 93<br>ViT/s-8 g alaxy10: 0.88 → 0 . 98<br>ViT/s-8 i net10: 0.90 → 0 . 98<br>− 3 − 2 − 1 0 1 2 3<br>Alignment coefficient ( α )<br>acc)<br>test<br>,<br>α<br>/λ<br>loss<br>Corr(train<br>**----- End of picture text -----**<br>


**Figure 11.** Spearman correlation ( **y-axis)** between LeJEPA’s training loss and downstream accuracy on the dataset’s classification task with a frozen backbone and linear evaluation. The **x-axis** varies _𝛼_ in Equation (8) following our scaling law of the loss w.r.t. _𝜆_ . Using _𝛼_ = 0 recovers the plain training loss. We clearly observe a very high correlation already for _𝛼_ = 0, which further increases up to 99% for _𝛼_ = 0 _._ 4. The entire set of points is obtained across numerous hyper-parameters such as learning rate, weight decay, number of epochs, _𝜆_ –demonstrating how **LeJEPA’s training loss is strongly predictive of downstream performance** which can be used for label-free cross-validation. 

downstream task performance, sometimes supplemented with unsupervised embedding statistics [Agrawal et al., 2022, Garrido et al., 2023, Thilak et al., 2023]. This process is highly limiting since it requires labeled data that is costly and overly specialized. This is further exacerbated in the latest JEPA models where training losses exhibit low correlation with downstream performance–and may not even decrease monotonically during training. 

In contrast, we find that LeJEPA’s training loss behaves much more favorably–providing us with a meaningful signal on model quality. First, we provide in Figure 10, the 2D plane spanned by the SIGReg and prediction losses 

where a clear trend with downstream task accuracy can be observed. More strikingly, the combined training loss (LeJEPA) with mixing coefficient _𝜆_ exhibits very high Spearman correlation [Spearman, 1961], denoted as _𝜌𝑠_ , of about 85% with downstream accuracy–which is considered a strong signal. This strong relationship holds across datasets and architectures. As a result, a lower LeJEPA training loss reliably indicates a better downstream performance. 

We can further improve this correlation through a simple scaling law based upon the trade-off weighting hyperparameter _𝜆_ 

**==> picture [209 x 26] intentionally omitted <==**

By setting _𝛼_ ≈ 0 _._ 4, LeJEPA’s training loss is able to achieve nearly 99% correlation with downstream performance across multiple datasets and models. We depict the changes in _𝐶_[(] _[𝛼]_[)] as a function of _𝛼_ on multiple datasets and models in Figure 11, as well as the training LeJEPA loss against downstream performance in Figure 19. **The strong alignment between LeJEPA’s training loss and model quality enables label-free SSL model selection and cross-validation** . 

## **6.3 In-Domain LeJEPA Outperforms Frontier Model Transfer Learning** 

A key promise of self-supervised learning is to learn universal representations that generalize across tasks and domains. However, current frontier foundation models (e.g., DINOv2/v3, ĲEPA) are pretrained on natural images forcing practitioners in specialized domains to collect large amount of labels for supervised finetuning. In fact, most frontier models can not be trained directly on those domains as the number of samples may be small and searching again for the hyper-parameters would be cum- 

15 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | **Sec 6: Experiments** 

## **LeJEPA:** 

**==> picture [509 x 177] intentionally omitted <==**

**----- Start of picture text -----**<br>
Full finetuning Frozen backbone<br>80<br>70<br>60 LeJEPA convnextv2 n ano (galaxy10)<br>LeJEPA levit 1 28 (galaxy10)<br>EO o o<br>50 / a LeJEPA resnet18 (galaxy10)<br>40 4 + LeJEPADINOv2 resnet34ViT/s (LVD142M)(galaxy10)<br>DINOv3 ViT/s (LVD1.7B)<br>30<br>20<br>1 2 5 10 100 1000 all 1 2 5 10 100 1000 all<br>Samples per class Samples per class<br>(%)<br>Accuracy<br>**----- End of picture text -----**<br>


**Figure 12. Small architecture in-domain (Galaxy10) LeJEPA pretraining** with linear probe evaluation using frozen backbone or full finetuning ( **columns** ) and with varying number of samples per class ( **x-axis)** . We compare against state-of-the-art foundation models (DINOv2/v3, ĲEPA) over 3 different random seeds. We observe that **LeJEPA enables in-domain pretraining out of the box across architectures and able to outperform frontier foundation models** . Corresponding numbers are provided in Table 3. 

**Table 2.** Few-shot classification accuracy (percentages) on 8 datasets spanning textures, objects, and fine-grained categories. **Our LeJEPA achieves superior performance on fine-grained tasks (DTD, flowers102, food101) while requiring only 100 pretraining epochs compared to I-JEPA’s 300 epochs—a 3× reduction in training time and computational resources without sacrificing downstream task performance.** This efficiency gain is particularly valuable for practical applications where training budget is limited. Bold indicates best performance within the IN-1K comparison group, all numbers are percentages. 

||||||||||Dataset|||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|shots|model|params|pretrain|epochs|DTD<br>aircr.<br>cars cifar10 cifar100 flowers102<br>~~EE~~||||||food|pets|avg.|
||LeJEPA ViT-L|304M|IN-1K|100|**33.21**|9.37|3.40|51.65|27.01|48.53|17.14|46.11|29.55|
||LeJEPA ConvNeXtV2-H|660M|IN-1K|100|32.15|8.07|4.28|50.95|31.48|**48.74 **|**17.95**|58.98|31.58|
|1|I-JEPA ViT-H|632M|IN-1K|300|27.71|9.86|4.33|**56.52**|30.58|44.69|14.53|53.38|30.20|
||I-JEPA ViT-H + STOP|632M|IN-1K|300|26.60|**11.18**|**4.75**|56.27|**35.20**|47.17|15.75|**59.47**|32.05|
||_I-JEPA ViT-H (22K)_<br>_632M_<br>_IN-22K_<br>_900_<br>_27.98 13.00_<br>_3.45_<br>_61.84_<br>_34.70_<br>_89.72 19.62 30.86_<br>_35.15_<br>~~EE~~|||||||||||||
||LeJEPA ViT-L|304M|IN-1K|100|**64.72**|35.25|22.25|85.15|59.77|**92.53 **|**50.90**|77.00|60.95|
||LeJEPA ConvNeXtV2-H|660M|IN-1K|100|61.84|30.67|24.46|85.74|63.29|91.78|49.32|78.53|60.70|
|10|I-JEPA ViT-H|632M|IN-1K|300|57.68|33.82|21.96|88.77|66.42|88.24|43.97|83.23|60.51|
||I-JEPA ViT-H + STOP|632M|IN-1K|300|57.00|**39.77 **|**25.21**|**90.09**|**70.32**|90.16|45.68|**85.13**|62.92|
||_I-JEPA ViT-H (22K)_<br>_632M_<br>_IN-22K_<br>_900_<br>_58.74 43.52 18.27_<br>_94.83_<br>_75.23_<br>_98.94 49.06 67.66_<br>63.28<br>~~Cy~~|||||||||||||
||LeJEPA ViT-L|304M|IN-1K|100|**78.30**|57.01|57.28|96.50|83.71|**91.21 **|**82.05**|89.74|79.48|
||LeJEPA ConvNeXtV2-H|660M|IN-1K|100|76.60|52.99|54.88|96.15|81.34|91.11|77.64|89.76|77.56|
|all|I-JEPA ViT-H|632M|IN-1K|300|73.32|56.61|54.47|97.54|86.42|86.47|81.02|92.11|78.50|
||I-JEPA ViT-H + STOP|632M|IN-1K|300|73.87|**61.95 **|**61.27**|**98.02**|**87.78**|88.08|81.72|**92.88**|80.70|
|_I-JEPA ViT-H (22K)_<br>_632M_<br>_IN-22K_<br>_900_<br>_75.67 65.39 49.79_<br>_98.46_<br>_89.95_<br>_98.54 81.58 87.19_<br>_80.82_<br>~~I—————~~||||||||||||||



**Figure 13. Emergent Object Segmentation via Last Layer Thresholding.** LeJEPA naturally learns to segment and track salient objects (shown in attention maps on the right of each video) without explicit supervision. The results display impressive visual quality and strong temporal consistency across video frames ( _videos provided on our project page_ ). This emergent capability demonstrates the rich semantic representations learned through our self-supervised approach. 

16 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | **Sec 6: Experiments** 

bersome yet necessary [Assran et al., 2022]. 

To demonstrate LeJEPA’s versatility and ability to resolve that current pain-point, we propose to pretrain directly on a new domain without any change in the loss or the pretraining pipeline. We select the Galaxy10 dataset, a galaxy morphology classification task that differs significantly from natural images in both visual structure and statistical properties [Balestriero et al., 2025]. The dataset contains 11,000 training samples across 10 galaxy types. For LeJEPA, we use the default hyper-parameters and pretrain for 400 epochs a variety of backbones. We compare against the latest DINOv2, DINOv3 and ĲEPA. We report in Figure 12 the top1 accuracy for linear probing both with frozen backbone and full-finetuning. We observe that **in-domain pretraining with LeJEPA substantially outperforms state-of-the-art frontier models (DINOv2, DINOv3) on both linear probing and full finetuning** . Additional datasets and backbones are provided in Table 5 depicting LeJEPA’s ability to train in-domain, even with a dataset with 1000 samples (flowers102). Coupling this result with the stability of LeJEPA across architectures and hyper-parameters should offer a promising alternatives in domains not yet accounted for by the latest frontier models. 

## **6.4 LeJEPA Scales Across Data and Models** 

**Figure 14. LeJEPA learns rich semantic representations through self-supervised learning.** PCA visualization of last-layer features from LeJEPA (ViT-Large, 100 epochs on ImageNet-1K). For each image, features are independently projected to RGB using the first 3 principal components. Without any supervision, LeJEPA spontaneously develops semantically meaningful representations: notice how warm colors (red/magenta/pink) consistently capture foreground objects (parrot bodies, dog face), while cool colors (cyan/green/yellow) represent backgrounds and foliage. This emergent object-background separation and perceptual grouping discovered the visual structure of the world purely from unlabeled data. 

We now propose to apply LeJEPA over a larger pretraining dataset, i.e., Imagenet-1k, and over larger backbones such as ViT/Large (0.3B), ConvNextV2-Huge (0.6B). For those two models, we reach an online linear probe accuracy on inet1k of 77.1% and 78.5% respectively. Beyond in-distribution performances, we also explore transfer learning. For those experiments, our baselines are ĲEPA with a ViT-Huge (0.6B) which is the closest to our setup, and we also include a recent improved version of ĲEPA including additional stochastic prediction tasks [Bar et al., 2023] that is coined ĲEPA + STOP. For LeJEPA, we employ the same recipe as described in Section 6.1 and report transfer learning performances with frozen backbone in Table 2. We observe that we consistently outperform ĲEPA while employed a smaller model and shorted training schedule. Beyond top1 accuracy, we also echo our findings from Section 6.2 about LeJEPA’s training loss quality. In our setup, we observe a very stable and smooth training curve indicating a stable optimization landscape removing the need for careful hyperparameter selection (recall thm. 4). We provide an example on a ViT-gigantic (1.8B parameters) in Figure 1. 

## **6.5 Emergent Semantic Structure in LeJEPA Representations** 

A hallmark of successful self-supervised learning is the emergence of semantically meaningful attention patterns 

17 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

without explicit supervision [Caron et al., 2021]. To assess whether LeJEPA learns such structure, we visualize the attention maps of the learned representations. Following DINO [Caron et al., 2021], we apply PCA to the embeddings and visualize the first principal components, which reveal clear correspondence to object boundaries and salient regions (Figure 14). Furthermore, we explore whether these attention patterns can enable unsupervised video segmentation—a challenging task requiring temporal consistency and object understanding. By thresholding the self-attention maps of the [CLS] token, we obtain binary masks that track objects across frames without any segmentation labels during training. As shown in Figure 13, **LeJEPA’s attention naturally segments foreground objects from background with remarkable temporal coherence** , suggesting that the learned representations capture both spatial semantics and temporal structure. This emergent capability demonstrates that LeJEPA’s stabilityfocused objective does not sacrifice the semantic richness of learned features. 

## **7 Conclusion** 

We have established a principled theoretical framework for JEPA-based self-supervised learning that fundamentally resolves its core pathologies. Our contributions span theory and practice: we proved that isotropic Gaussian embeddings uniquely minimize worst-case downstream risk, introduced SIGReg as a tractable and provably correct method to enforce this distribution, and demonstrated that this approach eliminates representational collapse by design–and not through ad-hoc combinations of teacherstudent networks, stop-gradients, or asymmetric architectures. 

We validate LeJEPA across domains and over 60 architectures including gigantic versions with 1.8B parameters. In spite of its simplicify , LeJEPA matches state-of-the-art performance while requiring fewer than 50 lines of core implementation. Critically, our approach provides what SSL has long needed: a mathematically rigorous foundation that directly informs practical algorithm design. 

## **Acknowledgments** 

We would like to thank Mike Rabbat and Lucas Maes for providing valuable feedbacks on the manuscript. 

## **References** 

- Haneen Arafat Abu Alfeilat, Ahmad BA Hassanat, Omar Lasassmeh, Ahmad S Tarawneh, Mahmoud Bashir Alhasanat, Hamzeh S Eyal Salman, and VB Surya Prasath. Effects of distance measure choice on k-nearest neighbor classifier performance: a review. _Big data_ , 7(4):221–248, 2019. 

- Robert A Adams and John JF Fournier. _Sobolev spaces_ , volume 140. Elsevier, 2003. 

- Kumar K Agrawal, Arnab Kumar Mondal, Arna Ghosh, and Blake Richards. a-req: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. _Advances in Neural Information Processing Systems_ , 35:17626–17638, 2022. 

- Theodore W Anderson and Donald A Darling. Asymptotic theory of certain" goodness of fit" criteria based on stochastic processes. _The annals of mathematical statistics_ , pages 193–212, 1952. 

- Mahmoud Assran, Randall Balestriero, Quentin Duval, Florian Bordes, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, and Nicolas Ballas. The hidden uniform cluster prior in self-supervised learning. _arXiv preprint arXiv:2210.07277_ , 2022. 

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15619–15629, 2023. 

- Randall Balestriero and Yann LeCun. Contrastive and noncontrastive self-supervised learning recover global and local spectral embedding methods. _Advances in Neural Information Processing Systems_ , 35:26671–26685, 2022. 

- Randall Balestriero and Yann LeCun. Learning by reconstruction produces uninformative features for perception. _arXiv preprint arXiv:2402.11337_ , 2024. 

- Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al. A cookbook of self-supervised learning. _arXiv preprint arXiv:2304.12210_ , 2023. 

- Randall Balestriero, Nicolas Ballas, Mike Rabbat, and Yann LeCun. Gaussian embeddings: How jepas secretly learn your data density. _arXiv preprint arXiv:2510.05949_ , 2025. 

- Amir Bar, Florian Bordes, Assaf Shocher, Mahmoud Assran, Pascal Vincent, Nicolas Ballas, Trevor Darrell, Amir Globerson, and Yann LeCun. Stochastic positional embeddings improve masked image modeling. _arXiv preprint arXiv:2308.00566_ , 2023. 

- Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for selfsupervised learning. _arXiv preprint arXiv:2105.04906_ , 2021. 

18 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

- Jan Beirlant, Edward J Dudewicz, László Györfi, Edward C Van der Meulen, et al. Nonparametric entropy estimation: An overview. _International Journal of Mathematical and Statistical Sciences_ , 6(1):17–39, 1997. 

- Chris M Bishop. Training with noise is equivalent to tikhonov regularization. _Neural computation_ , 7(1):108– 116, 1995. 

- Christopher M Bishop and Nasser M Nasrabadi. _Pattern recognition and machine learning_ , volume 4. Springer, 2006. 

- Gunnar Blom. _Statistical estimates and transformed betavariables_ . PhD thesis, Almqvist & Wiksell, 1958. 

- Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and radon wasserstein barycenters of measures. _Journal of Mathematical Imaging and Vision_ , 51(1):22–45, 2015. 

- Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah. Signature verification using a" siamese" time delay neural network. _Advances in neural information processing systems_ , 6, 1993. 

- Jerome S Bruner and Leo Postman. On the perception of incongruity: A paradigm. _Journal of personality_ , 18(2): 206–223, 1949. 

- Russel E Caflisch. Monte carlo and quasi-monte carlo methods. _Acta numerica_ , 7:1–49, 1998. 

- Torsten Carleman. _Les Fonctions quasi analytiques: leçons professées au College de France_ . Gauthier-Villars, 1926. 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 9650–9660, 2021. 

- Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In _International conference on machine learning_ , pages 1597–1607. PmLR, 2020a. 

- Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. _Advances in neural information processing systems_ , 33:22243–22255, 2020b. 

- Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 9640–9649, 2021. 

- Kacper Chwialkowski, Heiko Strathmann, and Arthur Gretton. A kernel test of goodness of fit. In _International conference on machine learning_ , pages 2606–2615. PMLR, 2016. 

- Romain Cosentino, Anirvan Sengupta, Salman Avestimehr, Mahdi Soltanolkotabi, Antonio Ortega, Ted Willke, and Mariano Tepper. Toward a geometrical understanding of self-supervised contrastive learning. _arXiv preprint arXiv:2205.06926_ , 2022. 

- Thomas M Cover. _Elements of information theory_ . John Wiley & Sons, 1999. 

- Harald Cramér. On the composition of elementary errors: First paper: Mathematical deductions. _Scandinavian Actuarial Journal_ , 1928(1):13–74, 1928. 

- Harald Cramér and Herman Wold. Some theorems on distribution functions. _Journal of the London Mathematical Society_ , 1(4):290–294, 1936. 

- Marco Cuturi, Olivier Teboul, and Jean-Philippe Vert. Differentiable ranking and sorting using optimal transport. _Advances in neural information processing systems_ , 32, 2019. 

- Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. _arXiv preprint arXiv:2309.16588_ , 2023. 

- Josef Dick and Friedrich Pillichshammer. _Digital nets and sequences: discrepancy theory and quasi–Monte Carlo integration_ . Cambridge University Press, 2010. 

- Ted Dunning. The t-digest: Efficient estimates of distributions. _Software Impacts_ , 7:100049, 2021. 

- Ted Dunning and Otmar Ertl. Computing extremely accurate quantiles using t-digests. _arXiv preprint arXiv:1902.04023_ , 2019. 

- Gustav Elfving. The asymptotical distribution of range in samples from a normal population. _Biometrika_ , 34(1/2): 111–119, 1947. 

- Thomas W Epps and Lawrence B Pulley. A test for normality based on the empirical characteristic function. _Biometrika_ , 70(3):723–726, 1983. 

- Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for self-supervised representation learning. In _International conference on machine learning_ , pages 3015–3024. PMLR, 2021. 

- David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling language-free visual representation learning. _arXiv preprint arXiv:2504.01017_ , 2025. 

19 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

- Ronald Aylmer Fisher. _Statistical methods for research workers_ . Number 5. Oliver and Boyd, 1928. 

- Karl Friston. The free-energy principle: a unified brain theory? _Nature reviews neuroscience_ , 11(2):127–138, 2010. 

- Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank. In _International conference on machine learning_ , pages 10929–10974. PMLR, 2023. 

- Gene H Golub, Per Christian Hansen, and Dianne P O’Leary. Tikhonov regularization and total least squares. _SIAM journal on matrix analysis and applications_ , 21(1): 185–194, 1999. 

- Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. _Deep learning_ , volume 1. MIT press Cambridge, 2016. 

- Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In _Proceedings of the ieee/cvf International Conference on computer vision_ , pages 6391– 6400, 2019. 

- Richard Langton Gregory. Perceptions as hypotheses. _Philosophical Transactions of the Royal Society of London. B, Biological Sciences_ , 290(1038):181–197, 1980. 

- Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. _The journal of machine learning research_ , 13(1):723–773, 2012. 

- Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in neural information processing systems_ , 33:21271–21284, 2020. 

- Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. Stochastic optimization of sorting networks via continuous relaxations. _arXiv preprint arXiv:1903.08850_ , 2019. 

- AK Gupta. Estimation of the mean and standard deviation of a normal population from a censored sample. _Biometrika_ , 39(3/4):260–273, 1952. 

- Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In _Proceedings of the thirteenth international conference on artificial intelligence and statistics_ , pages 297–304. JMLR Workshop and Conference Proceedings, 2010. 

- JM Hammersley and KW Morton. The estimation of location and scale parameters from grouped data. _Biometrika_ , 41(3/4):296–301, 1954. 

- Felix Hausdorff. Momentprobleme für ein endliches intervall. _Mathematische Zeitschrift_ , 16(1):220–248, 1923. 

- Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9729–9738, 2020. 

- H von Helmholtz et al. Handbook of physiological optics. _Voss, Leipzig_ , 1867. 

- R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. _arXiv preprint arXiv:1808.06670_ , 2018. 

- C. A. R. Hoare. Quicksort. _The Computer Journal_ , 5(1):10–16, 01 1962. ISSN 0010-4620. doi: 10.1093/comjnl/5.1.10. URL `https://doi.org/10.1093/comjnl/5.1.10` . 

- Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization, 2019. URL `https://arxiv.org/abs/1803.05407` . 

- Carlos M Jarque and Anil K Bera. Efficient tests for normality, homoscedasticity and serial independence of regression residuals. _Economics letters_ , 6(3):255–259, 1980. 

- Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive selfsupervised learning. _arXiv preprint arXiv:2110.09348_ , 2021. 

- Harry Joe. Estimation of entropy and other functionals of a multivariate density. _Annals of the Institute of Statistical Mathematics_ , 41(4):683–697, 1989. 

- Thomas Kerdreux, Alexandre Tuel, Quentin Febvre, Alexis Mouche, and Bertrand Chapron. Efficient selfsupervised learning for earth observation via dynamic dataset curation. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 3017–3027, 2025. 

- Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. _arXiv preprint arXiv:2403.12945_ , 2024. 

20 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

- Diederik P Kingma, Danilo J Rezende, Shakir Mohamed, and Max Welling. Semi-supervised learning with deep generative models. _Advances in neural information processing systems_ , 27, 2014. 

- A. N. Kolmogorov. Sulla determinazione empirica di una legge di distribuzione. _Giornale dell’Istituto Italiano degli Attuari_ , 4:83–91, 1933. 

- Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. _Open Review_ , 62(1):1–62, 2022. 

- Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. _nature_ , 521(7553):436–444, 2015. 

- Erich Leo Lehmann and Joseph P Romano. _Testing statistical hypotheses_ . Springer, 2005. 

- Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. Self-supervised learning: Generative or contrastive. _IEEE transactions on knowledge and data engineering_ , 35(1):857–876, 2021. 

- Zhuang Ma and Michael Collins. Noise contrastive estimation and negative sampling for conditional models: Consistency and statistical efficiency. _arXiv preprint arXiv:1809.01812_ , 2018. 

- Tobias Maltenberger, Ivan Ilic, Ilin Tolovski, and Tilmann Rabl. Evaluating multi-gpu sorting with modern interconnects. In _Proceedings of the 2022 International Conference on Management of Data_ , pages 1795–1809, 2022. 

- George Marsaglia. Choosing a point from the surface of a sphere. _The Annals of Mathematical Statistics_ , 43(2): 645–646, 1972. 

- Charles Masson, Jee E Rim, and Homin K Lee. Ddsketch: A fast and fully-mergeable quantile sketch with relativeerror guarantees. _arXiv preprint arXiv:1908.10693_ , 2019. 

- David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. In _International Conference on Artificial Intelligence and Statistics_ , pages 875–884. PMLR, 2020. 

- H Mhaskar, F Narcowich, and J Ward. Spherical marcinkiewicz-zygmund inequalities and positive quadrature. _Mathematics of computation_ , 70(235):1113– 1130, 2001. 

- Erik G Miller. A new class of entropy estimators for multi-dimensional densities. In _2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP’03)._ , volume 3, pages III–297. IEEE, 2003. 

- Frederick Mosteller. _On some useful “inefficient” statistics_ . Springer, 2006. 

- Elizbar A Nadaraya. On estimating regression. _Theory of Probability & Its Applications_ , 9(1):141–142, 1964. 

- Francis J Narcowich, Pencho Petrushev, and Joseph D Ward. Localized tight frames on spheres. _SIAM Journal on Mathematical Analysis_ , 38(2):574–594, 2006. 

- Jerzy Neyman and Egon Sharpe Pearson. Ix. on the problem of the most efficient tests of statistical hypotheses. _Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character_ , 231(694-706):289–337, 1933. 

- Khai Nguyen and Nhat Ho. Energy-based sliced wasserstein distance. _Advances in Neural Information Processing Systems_ , 36:18046–18075, 2023. 

- Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 

- Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 

- Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. _Proceedings of the National Academy of Sciences_ , 117(40):24652–24663, 2020. 

- Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver Deussen. Monotonic differentiable sorting networks. _arXiv preprint arXiv:2203.09630_ , 2022. 

- RoL Plackett. Linear estimation from censored data. _The Annals of Mathematical Statistics_ , 29(1):131–142, 1958. 

- Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of mutual information. In _International conference on machine learning_ , pages 5171–5180. PMLR, 2019. 

- M Mahibbur Rahman and Z Govindarajulu. A modification of the test of shapiro and wilk for normality. _Journal of Applied Statistics_ , 24(2):219–236, 1997. 

- Bryan Rodas, Natalie Montesino, Jakob Ambsdorf, David Klindt, and Randall Balestriero. Diet-cp: Lightweight and data efficient self supervised continued pretraining. _arXiv preprint arXiv:2509.06990_ , 2025. 

- Samarendra Nath Roy. On a heuristic method of test construction and its use in multivariate analysis. _The Annals of Mathematical Statistics_ , 24(2):220–238, 1953. 

21 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

- David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. _nature_ , 323(6088):533–536, 1986. 

- Claude E Shannon. A mathematical theory of communication. _The Bell system technical journal_ , 27(3):379–423, 1948. 

- Samuel S Shapiro and RS Francia. An approximate analysis of variance test for normality. _Journal of the American statistical Association_ , 67(337):215–216, 1972. 

- Samuel Sanford Shapiro and Martin B Wilk. An analysis of variance test for normality (complete samples). _Biometrika_ , 52(3-4):591–611, 1965. 

- Ravid Shwartz Ziv and Yann LeCun. To compress or not to compress—self-supervised learning and information theory: A review. _Entropy_ , 26(3):252, 2024. 

- Ravid Shwartz-Ziv, Randall Balestriero, and Yann LeCun. What do we maximize in self-supervised learning? _arXiv preprint arXiv:2207.10081_ , 2022. 

- Bernard W Silverman. _Density estimation for statistics and data analysis_ . Routledge, 2018. 

- Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cĳo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. _arXiv preprint arXiv:2508.10104_ , 2025. 

- Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In _Uncertainty in artificial intelligence_ , pages 574–584. PMLR, 2020. 

- Charles Spearman. The proof and measurement of association between two things. 1961. 

- Bharath K Sriperumbudur, Arthur Gretton, Kenji Fukumizu, Bernhard Schölkopf, and Gert RG Lanckriet. Hilbert space embeddings and metrics on probability measures. _The Journal of Machine Learning Research_ , 11: 1517–1561, 2010. 

- Shiliang Sun and Rongqing Huang. An adaptive k-nearest neighbor algorithm. In _2010 seventh international conference on fuzzy systems and knowledge discovery_ , volume 1, pages 91–94. IEEE, 2010. 

- Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. _ACM Sigart Bulletin_ , 2 (4):160–163, 1991. 

- Gábor J Székely and Maria L Rizzo. A new test for multivariate normality. _Journal of Multivariate Analysis_ , 93(1): 58–80, 2005. 

- Ivan Tanasic, Lluís Vilanova, Marc Jordà, Javier Cabezas, Isaac Gelado, Nacho Navarro, and Wen-mei Hwu. Comparison based sorting for systems with multiple gpus. In _Proceedings of the 6th Workshop on General Purpose Processor Using Graphics Processing Units_ , pages 1–11, 2013. 

- Kashvi Taunk, Sanjukta De, Srishti Verma, and Aleena Swetapadma. A brief review of nearest neighbor algorithm for learning and classification. In _2019 international conference on intelligent computing and control systems (ICCS)_ , pages 1255–1260. IEEE, 2019. 

- Vimal Thilak, Chen Huang, Omid Saremi, Laurent Dinh, Hanlin Goh, Preetum Nakkiran, Joshua M Susskind, and Etai Littwin. Lidar: Sensing linear probing performance in joint embedding ssl architectures. _arXiv preprint arXiv:2312.04000_ , 2023. 

- Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? _Advances in neural information processing systems_ , 33:6827–6839, 2020. 

- Yuandong Tian, Xinlei Chen, and Surya Ganguli. Understanding self-supervised learning dynamics without contrastive pairs. In _International Conference on Machine Learning_ , pages 10268–10278. PMLR, 2021. 

- Edward C Tolman. Cognitive maps in rats and men. _Psychological review_ , 55(4):189, 1948. 

- Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani, Aviv Regev, and Randall Balestriero. Joint embedding vs reconstruction: Provable benefits of latent space prediction for self supervised learning. _arXiv preprint arXiv:2505.12477_ , 2025. 

- Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol, and Léon Bottou. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. _Journal of machine learning research_ , 11(12), 2010. 

- Huy V Vo, Vasil Khalidov, Timothée Darcet, Théo Moutakanni, Nikita Smetanin, Marc Szafraniec, Hugo Touvron, Camille Couprie, Maxime Oquab, Armand Joulin, et al. Automatic data curation for self-supervised learning: A clustering-based approach. _arXiv preprint arXiv:2405.15613_ , 2024. 

- Hermann Von Helmholtz. _Handbuch der physiologischen Optik_ , volume 9. L. Voss, 1867. 

- Richard Von Mises. _Probability, statistics, and truth_ . Courier Corporation, 1981. 

22 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

- Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, and Xinlei Chen. On the importance of asymmetry for siamese representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 16570–16579, 2022. 

- Geoffrey S Watson. Smooth regression analysis. _Sankhya:¯ The Indian Journal of Statistics, Series A_ , pages 359–372, 1964. 

- George S Watson. Goodness-of-fit tests on a circle. _Biometrika_ , 48(1/2):109–114, 1961. 

- S Weisburg and C Binham. An approximate analysis of variance test for non-normality suitable for machine computation. _Technometrics_ , 17:133–134, 1975. 

- Shichao Zhang, Xuelong Li, Ming Zong, Xiaofeng Zhu, and Ruili Wang. Efficient knn classification with different numbers of nearest neighbors. _IEEE transactions on neural networks and learning systems_ , 29(5):1774–1785, 2017. 

- Yifan Zhang, Zhiquan Tan, Jingqin Yang, Weiran Huang, and Yang Yuan. Matrix information theory for selfsupervised learning. _arXiv preprint arXiv:2305.17326_ , 2023. 

23 

## **LeJEPA** 

## **Appendix** 

## **A Additional Details on Nonlinear Probing** 

## **A.1 kNN Probing** 

To allow for more flexible evaluation of the pretrained encoder _𝑓_ _**𝜽**_ , it is standard to work with a _𝑘_ -NN prober [Taunk et al., 2019], both for regression and classification. We rely on the radial _𝑘_ -NN variation that leverages a sample-dependent _𝑘_ –improving performance for non uniform distributions of samples [Sun and Huang, 2010, Zhang et al., 2017, Abu Alfeilat et al., 2019]. 

We denote the underlying embedding density as _𝑝𝑧_ ∈ _𝐶_[3] with derivatives of order up to 3 bounded, and finite Fisher information and covariance. This regularity condition is fulfilled by current encoders. The _unknown_ labels come from the target function _𝜂_ : R _[𝐾]_ → R, assumed _𝐶_[2] . We handle classification tasks by setting _𝜂_ ( _𝒛_ ) = P( _𝑌_ = 1 | _𝒛_ ). The training consists of the _𝑁_ embeddings along with their training labels {( _𝒛𝑛 , 𝜂_ ( _𝒛𝑛_ ))} _𝑛[𝑁]_ =1[, where we will denote] _[ 𝒚][𝑛]_[≜] _[𝜂]_[(] _[𝒛][𝑛]_[)][.][The] prediction for a query vector _𝒒_ is formed as 

**==> picture [320 x 31] intentionally omitted <==**

with _𝒚_ ( _𝒒_ ) ≜ #{ _𝑛_ : �� _𝒛𝑛_ − _𝒒_ �� ≤ _𝑟_ 0} counting the number of samples within a _𝑟_ -radius ball around _𝒒_ . The radius _𝑟_ controls how many neighbors predictions are averaged to form the query’s prediction. As per the linear probing’s lemma. 1, we can characterize the bias of the estimator Equation (kNN) at a particular query point, as formalized below. 

## **Lemma 4: k-NN Pointwise Bias** 

The (kNN) estimator has bias at query _𝒒_ given by _𝑟_[2] 0 Bias( _𝒒_ ) = _𝑑_ + 2 �∇ _𝜂_ ( _𝒒_ )[⊤] ∇ log _𝑝𝑧_ ( _𝒒_ ) +[1] 2[Δ] _[𝜂]_[(] _[𝒛]_[)] � where the remainder _𝑜_ ( _𝑟_ 0[2][)][ is uniform in] _[ 𝒒]_[.][(Proof in Section B.3.)] 

**==> picture [31 x 14] intentionally omitted <==**

To obtain the integrated bias, i.e., over the distribution of query points, we consider the following two properties. First, the distribution of query points follow the training distribution, i.e., _𝒒_ ∼ _𝑝𝑧_ , second, target function _𝜂_ has gradient which is mean-zero and isotropic with E�∇ _𝜂_ ( _𝒛_ )∇ _𝜂_ ( _𝒛_ )[⊤][�] = _𝜏_[2] _𝑔[𝐼][𝑑]_[with] _[ 𝜏]_[2] _𝑔_[∈(][0] _[,]_[ ∞)][ uniformly in] _[ 𝒛]_[.][We also have any finite] scalar-constraint on the covariance of the embeddings such as Tr(Σ) = _𝑐_ or ∥Σ∥ _𝐹_ = _𝑐_ for a finite constant _𝑐_ . 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **Theorem 7: k-NN isotropic Gaussian Optimality** 

The integrated squared bias of (kNN) satisfies 

**==> picture [409 x 45] intentionally omitted <==**

## As a result, we now have a unique minimizer for the optimal embedding density for both the linear and k-NN probes. **A.2 Kernel Probing** 

As an alternative to (kNN), it is also common to leverage kernel methods, which we consider in this section. Consider a kernel _𝐾_ : R _[𝐾]_ → R with the following standard properties 

**==> picture [319 x 107] intentionally omitted <==**

for some _𝜇_ 2( _𝐾_ ) ∈(0 _,_ ∞), some bandwidth _ℎ >_ 0 and denoting _𝐾 ℎ_ ( _𝑡_ ) ≜ _ℎ_[−] _[𝑑] 𝐾_ ( _𝑡_ / _ℎ_ ), we remind the reader that the Nadaraya-Watson estimator, introduced in Nadaraya [1964], Watson [1964], at a query _𝒒_ ∈ R _[𝑑]_ is 

**==> picture [321 x 31] intentionally omitted <==**

Similarly to (kNN), we will see that the performance of (NW) depends crucially on the distribution of the training points. We have access to our dataset of inputs from _𝑝𝑧_ and for each sample _𝒛𝑛_ the corresponding target is given from _𝜂_ ( _𝒛𝑛_ ) = E[ _𝑌𝑛_ | _𝒛𝑛_ ]. We also denote the corresponding conditional variance of the target function at that point as _𝑣_ ( _𝑥_ ) = Var( _𝑌𝑖_ | _𝑋𝑖_ = _𝑥_ ). We follow the regularity conditions of the k-NN probing derivations and additionally assume that _𝑝_ has sufficiently light tails so that for each coordinate _𝑗_ , lim∥ _𝑥_ ∥→∞ _𝑝_ ( _𝑥_ ) = 0 and lim∥ _𝑥_ ∥→∞ _𝑥 𝑗 𝑝_ ( _𝑥_ ) = 0. We first derive the pointwise bias and variance for � _𝒚_ ( _𝒒_ ). 

## **Lemma 5: Kernel Bias and Variance** 

**==> picture [237 x 12] intentionally omitted <==**

**==> picture [413 x 68] intentionally omitted <==**

We now show that, under a fixed mean and total-covariance constraint on _𝑝𝑧_ , the isotropic Gaussian distribution uniquely minimizes the bias and variance of the kernel regression estimator at any test point. We restrict the smoothness class of the target function using 

**==> picture [168 x 20] intentionally omitted <==**

|Δ _𝒚_ ( _𝒒_ )| ≤ _𝐵,_ ∀ _𝒒_ ∈ R _[𝑑]_[�] _,_ 

25 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

allowing us to formalize below the worst case integrated bias and the optimal density for _𝑧_ . 

**Theorem 8: Kernel isotropic Gaussian Optimality** 

**==> picture [496 x 50] intentionally omitted <==**

and the integrated variance is independent of _𝑝_ . Among all densities _𝑝_ on R _[𝑑]_ with total-variance constrained, e.g., Tr(Σ) = _𝑐_ , the isotropic Gaussian is the unique minimizer. (Proof in Section B.7.) 

## **B Proofs** 

## **B.1 Proof of lemma. 1** 

> _Proof._ Our proof follows standard derivations when it comes to studying the bias of an estimator. Let’s consider the ridge regression problem (Tikhonov regularized least squares estimator) with close form estimator 

**==> picture [315 x 13] intentionally omitted <==**

The labels are formed from the ground truth parameter _𝛽_ true with centered error, as per **Y** = **X** _**𝜷**_ true + _**𝜺**_ where E[ _**𝜺**_ ] = **0** . We can now look at the bias of our estimator given by 

**==> picture [180 x 64] intentionally omitted <==**

We will now compare that bias when _𝑿_ has isotropic and anisotropic covariance with same total variance: 

**==> picture [310 x 26] intentionally omitted <==**

For any anisotropic covariance matrix of _𝑿_ , denote by _𝒒_ 1 the eigenvector with smallest eigenvalue, and let’s denote by _𝜅>_ 0 a positive constant. We now define 

**==> picture [291 x 11] intentionally omitted <==**

leading to 

**==> picture [166 x 52] intentionally omitted <==**

Since _𝜆𝑝 < 𝜆_[¯] (strict inequality when not isotropic): 

**==> picture [90 x 25] intentionally omitted <==**

we obtain that 

**==> picture [164 x 14] intentionally omitted <==**

As a result, whenever the covariance matrix of _𝑿_ is anisotropic, there will be downstream tasks for which the estimator bias is increased compared to having isotropic covariance matrix. Anisotropic covariance structure thus amplifies regularization bias when the true parameter vector aligns unfavorably with the data’s covariance structure. □ 

26 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

□ 

## **B.2 Proof of lemma. 2** 

_Proof._ We use the same formula as in Section B.1 with _𝜆_ wd = 0. We first see that the estimator is unbiased. We will now leverage that result to compute the covariance matrix of the estimator 

**==> picture [176 x 80] intentionally omitted <==**

leading to the total variance 

**==> picture [147 x 32] intentionally omitted <==**

where we used the eigendecomposition: 

**==> picture [51 x 12] intentionally omitted <==**

The function _𝑓_ ( _𝑥_ ) = _𝑥_[1][is strictly convex on][ (][0] _[,]_[ ∞)][ allowing us to leverage Jensen’s Inequality:] 

**==> picture [150 x 124] intentionally omitted <==**

The inequality is strict whenever the eigenvalues { _𝜆𝑗_ } _[𝑝] 𝑗_ =1[are not all equal.] 

## **B.3 Proof of lemma. 4** 

_Proof._ Under PPP, conditional expectations of � _𝜂_ ( _𝑥_ ) coincide with the normalized ball average 

**==> picture [266 x 35] intentionally omitted <==**

which is the key surrogate used below. **Ball integrals.** For computations we use (by symmetry) for any _𝑟 >_ 0: 

**==> picture [322 x 29] intentionally omitted <==**

Fix _𝑥_ ∈ R _[𝑑]_ and write _𝑧_ ∈ B(0 _, 𝑟_ 0) for local displacements. Assume _𝑝_ ∈ _𝐶_[3] , _𝜂_ ∈ _𝐶_[2] with bounded derivatives on the region of interest, and expand a second-order Taylor expansion: 

**==> picture [222 x 31] intentionally omitted <==**

with remainders satisfying | _𝑅𝜂_ ( _𝑥_ ; _𝑧_ )| ≤ _𝐶𝜂_ ∥ _𝑧_ ∥[3] and | _𝑅𝑝_ ( _𝑥_ ; _𝑧_ )| ≤ _𝐶𝑝_ ∥ _𝑧_ ∥[3] uniformly for ∥ _𝑧_ ∥≤ _𝑟_ 0. Using the ball identities 

27 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

∫ _𝐵_ (0 _,𝑟_ ) _[𝑧𝑑𝑧]_[=][ 0 and] ∫ _𝐵_ (0 _,𝑟_ ) _[𝑧𝑧]_[⊤] _[𝑑𝑧]_[=] _[𝑣][𝑑] 𝑑[𝑟]_ + _[𝑑]_ 2[+][2] _[𝐼][𝑑]_[and collecting terms up to order] _[ 𝑟]_ 0 _[𝑑]_[+][2] , we simplify the denominator as 

**==> picture [257 x 86] intentionally omitted <==**

since ∫ _𝑧𝑑𝑧_ = 0 and ∫ _𝑧_[⊤] _𝐻𝑝𝑧𝑑𝑧_ = tr( _𝐻𝑝_ ) _𝑣𝑑𝑑𝑟_ +0 _[𝑑]_ 2[+][2][and the denominator as] 

**==> picture [452 x 84] intentionally omitted <==**

Cubic terms vanish by symmetry, and quartic terms are _𝑂_ ( _𝑟_ 0 _[𝑑]_[+][4] ). Subtract _𝜂_ ( _𝑥_ )𝒟( _𝑥_ ) to obtain the bias numerator: 

**==> picture [292 x 26] intentionally omitted <==**

Write 𝒟( _𝑥_ ) = _𝑣𝑑𝑟_ 0 _[𝑑][𝑝]_[(] _[𝑥]_[)][�] 1 + _𝛼_ ( _𝑥_ ) _𝑟_ 0[2][+] _[ 𝑂]_[(] _[𝑟]_ 0[3][)][�] where _𝛼_ ( _𝑥_ ) := 2( _𝑑_ +21) _𝑝_ ( _𝑥_ )[tr][(] _[𝐻𝑝]_[(] _[𝑥]_[))][.][Then] 

**==> picture [285 x 99] intentionally omitted <==**

uniformly on 𝒦 . This gives the bias formula 

**==> picture [266 x 25] intentionally omitted <==**

completing the proof. 

**==> picture [8 x 6] intentionally omitted <==**

## **B.4 Proof of thm. 7** 

_Proof._ Recall from Section B.3 that the bias term as sample _𝒙_ is given by 

**==> picture [264 x 55] intentionally omitted <==**

28 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

where we defined _𝐴_ ( _𝑥_ ) ≜ ∇ _𝜂_ ( _𝑥_ ) · ∇ log _𝑝_ ( _𝑥_ ) and _𝐶_ ( _𝑥_ ) ≜[1] 2[Δ] _[𝜂]_[(] _[𝑥]_[)][.][We now square and take expectation of] _[ 𝑋]_[∼] _[𝑝]_[and the] isotropic gradient prior 

**==> picture [437 x 83] intentionally omitted <==**

We will derive each term separately, recalling that we assume an isotropic gradient prior for _𝜂_ , i.e., E�∇ _𝜂_ ( _𝑥_ )� = 0 and E�∇ _𝜂_ ( _𝑥_ )∇ _𝜂_ ( _𝑥_ )[⊤][�] = _𝜏_[2] _𝑔[𝐼][𝑑]_[, for some] _[ 𝜏]_[2] _𝑔_[∈(][0] _[,]_[ ∞)][.] 

**1) The score-gradient term** E[ _𝐴_ ( _𝑋_ )[2] ] **.** Using _𝑣_ ( _𝑥_ ) := ∇ log _𝑝_ ( _𝑥_ ) for brevity: 

**==> picture [207 x 164] intentionally omitted <==**

recovering the Fisher-information functional _𝐽_ ( _𝑝_ ), scaled by _𝜏_[2] _𝑔_ 

**2) The cross term** 2E[ _𝐴_ ( _𝑋_ ) _𝐶_ ( _𝑋_ )] **.** We have 

**==> picture [146 x 21] intentionally omitted <==**

Under the prior, ∇ _𝜂_ is mean-zero and isotropic; if, additionally, Δ _𝜂_ is uncorrelated with ∇ _𝜂_ and has zero mean (or is bounded and mean-zero after centering), then E _𝜂_ [ _𝐴_ ( _𝑥_ ) _𝐶_ ( _𝑥_ )] = 0. If one does _not_ assume the orthogonality/vanishing covariance above, then E[ _𝐴_ ( _𝑋_ ) _𝐶_ ( _𝑋_ )] is a finite constant (depending on the joint law of derivatives of _𝜂_ ), and the cross term contributes 

**==> picture [151 x 34] intentionally omitted <==**

not _𝑜_ ( _𝑟_ 0[4][)][.][In that general case, the leading] _[ 𝑝]_[-dependent term of][ E][[][Bias][(] _[𝑋]_[)][2][]][ is still the] _[ score-gradient][ 𝜏]_[2] _𝑔[𝐽]_[(] _[𝑝]_[)][.] 

**3) The curvature term** E[ _𝐶_ ( _𝑋_ )[2] ] **.** 

**==> picture [133 x 39] intentionally omitted <==**

which is independent of _𝑝_ , hence E� _𝐶_ ( _𝑋_ )[2][�] = _𝑂_ (1) 

29 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**Putting it together.** Substituting into (13): 

**==> picture [206 x 65] intentionally omitted <==**

We show that, among all mean-zero distributions _𝑝_ on R _[𝑑]_ with a given _scalar_ constraint on the covariance (trace, determinant, Frobenius norm, or spectral radius), the density that minimizes the Fisher-information functional 

**==> picture [138 x 25] intentionally omitted <==**

is the Gaussian with _isotropic_ covariance satisfying the same scalar constraint. We proceed in two steps: (i) for fixed 

covariance matrix Σ ≻ 0, _𝐽_ ( _𝑝_ ) is minimized by the Gaussian 𝒩(0 _,_ Σ) and attains the value tr(Σ[−][1] ); (ii) for each scalar constraint, tr(Σ[−][1] ) is minimized by Σ = _𝑠𝐼𝑑_ for the appropriate scalar _𝑠 >_ 0. 

## **Lemma 6: Special case: Recovery of VCReg** 

Let _𝑝_ be a mean-zero probability density on R _[𝑑]_ with covariance Σ = E[ _𝑋𝑋_[⊤] ] ≻ 0. Then _𝐽_ ( _𝑝_ ) ≥ tr(Σ[−][1] ) _,_ with equality if and only if _𝑝_ = 𝒩(0 _,_ Σ). 

_Proof._ Consider the location family _𝑝𝜃_ ( _𝑥_ ) := _𝑝_ ( _𝑥_ − _𝜃_ ), _𝜃_ ∈ R _[𝑑]_ . Its Fisher-information matrix at _𝜃_ is 

**==> picture [304 x 13] intentionally omitted <==**

so that _𝐽_ ( _𝑝_ ) = trℐ( _𝜃_ ). The estimator _𝑇_ ( _𝑋_ ) ≡ _𝑋_ is unbiased for _𝜃_ under _𝑝𝜃_ , with Cov( _𝑇_ ) = Σ. The matrix Cramér–Rao bound gives Cov( _𝑇_ ) ⪰ℐ( _𝜃_ )[−][1] , i.e., ℐ( _𝜃_ ) ⪰ Σ[−][1] . Taking traces yields _𝐽_ ( _𝑝_ ) ≥ tr(Σ[−][1] ). Equality in the matrix Cramér–Rao bound holds if and only if the score is an _affine_ function of _𝑋_ − _𝜃_ , i.e., ∇ log _𝑝𝜃_ ( _𝑋_ ) = _𝐴_ ( _𝑋_ − _𝜃_ ) a.s. for some matrix _𝐴_ ; integrating this identity shows _𝑝𝜃_ is Gaussian with precision matrix − _𝐴_ , hence _𝑝_ = 𝒩(0 _,_ Σ). □ 

## **Step 2: Optimizing over covariance shapes under scalar constraints** 

Write the eigenvalues of Σ as _𝜆_ 1 _, . . . , 𝜆𝑑 >_ 0. Then 

**==> picture [73 x 32] intentionally omitted <==**

We now solve min[�] _𝑖_[1][/] _[𝜆][𝑖]_[under each scalar constraint; in every case the minimum is attained when all] _[ 𝜆][𝑖]_[are equal, i.e.,] Σ = _𝑠𝐼𝑑_ . 

**(a) Trace constraint.** Given tr(Σ) =[�] _𝑖[𝜆][𝑖]_[=] _[ 𝑡][>]_[ 0, by Cauchy–Schwarz,] 

**==> picture [151 x 35] intentionally omitted <==**

with equality if and only if _𝜆_ 1 = · · · = _𝜆𝑑_ . Hence 

**==> picture [214 x 25] intentionally omitted <==**

30 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**(b) Determinant constraint.** Given det(Σ) =[�] _𝑖[𝜆][𝑖]_[=] _[𝛿>]_[0][,][set] _[𝜇][𝑖]_[:][=][1][/] _[𝜆][𝑖]_[so][that][�] _𝑖[𝜇][𝑖]_[=] _[𝛿]_[−][1][.][By][the][AM–GM] inequality, 

**==> picture [136 x 35] intentionally omitted <==**

with equality iff _𝜇_ 1 = · · · = _𝜇𝑑_ , i.e., _𝜆_ 1 = · · · = _𝜆𝑑_ . Thus 

**==> picture [246 x 19] intentionally omitted <==**

**(c) Frobenius-norm constraint.** Given ∥Σ∥[2] _𝐹_[=][�] _𝑖[𝜆]_[2] _𝑖_[=] _[𝑐]_[2] _[>]_[0][,][minimize] _[𝑓]_[(] _[𝜆]_[)][:][=][�] _𝑖_[1][/] _[𝜆][𝑖]_[over] _[𝜆][𝑖][>]_[0][subject][to] _𝑔_ ( _𝜆_ ) :=[�] _𝑖[𝜆]_[2] _𝑖_[=] _[𝑐]_[2][.][The Lagrangian] 

**==> picture [145 x 32] intentionally omitted <==**

has first-order conditions − _𝜆_[−] _𝑖_[2] + 2 _𝜈𝜆𝑖_ = 0 for all _𝑖_ , i.e., _𝜆_[3] _𝑖_[=] 21 _𝜈_[, so all] _[ 𝜆][𝑖]_[are equal.][Imposing][ �] _[𝜆]_[2] _𝑖_[=] _[𝑐]_[2][yields] _[ 𝜆][𝑖]_[=] _[𝑐]_[/] ~~√~~ _𝑑_ , hence 

**==> picture [274 x 31] intentionally omitted <==**

**(d) Spectral-radius constraint.** Let the spectral radius be constrained by _𝜌_ (Σ) = max _𝑖 𝜆𝑖_ ≤ _𝑟_ for some _𝑟 >_ 0. Since _𝑥_ ↦→ 1/ _𝑥_ is strictly decreasing on (0 _,_ ∞), 

**==> picture [98 x 31] intentionally omitted <==**

with equality if and only if _𝜆𝑖_ = _𝑟_ for all _𝑖_ . Therefore 

**==> picture [206 x 25] intentionally omitted <==**

(The same conclusion holds if the constraint is _𝜌_ (Σ) = _𝑟_ , since one may take all eigenvalues equal to _𝑟_ .) 

## **Conclusion: Isotropic Gaussian is optimal** 

Combining Lemma 6 with the solutions (a)–(d), we obtain: 

31 

## **LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **Theorem 9: Special case: Recovery of VCReg** 

- Fix one of the following scalar covariance constraints for a mean-zero distribution _𝑝_ on R _[𝑑]_ : • trace: tr(Cov( _𝑋_ )) = _𝑡_ , • determinant: det(Cov( _𝑋_ )) = _𝛿_ , • Frobenius norm: ∥Cov( _𝑋_ )∥ _𝐹_ = _𝑐_ , • spectral radius upper bound: _𝜌_ (Cov( _𝑋_ )) ≤ _𝑟_ . 

- Then the Fisher-information functional _𝐽_ ( _𝑝_ ) is minimized over all such _𝑝_ by the isotropic Gaussian _𝑝𝐺_ = 𝒩(0 _, 𝑠𝐼𝑑_ ) with _𝑠_ chosen to satisfy the constraint. The minimal values are: trace _𝑡_ : _𝐽_ min = _[𝑑]_[2] _𝑠_ = _[𝑡] 𝑡[,] 𝑑[,]_ 

- determinant _𝛿_ : _𝐽_ min = _𝑑𝛿_[−][1][/] _[𝑑] , 𝑠_ = _𝛿_[1][/] _[𝑑] , 𝑐_ 

- Frobenius _𝑐_ : _𝐽_ min = _[𝑑]_[3][/][2] _, 𝑠_ = _, 𝑐_ ~~√~~ _𝑑_ 

- spectral radius _𝑟_ : _𝐽_ min = _[𝑑] 𝑠_ = _𝑟. 𝑟[,]_ 

- In each case, _𝑝𝐺_ is the unique minimizer (up to null sets). 

_Proof._ For any admissible _𝑝_ with covariance Σ, Lemma 6 gives _𝐽_ ( _𝑝_ ) ≥ tr(Σ[−][1] ). Minimizing the right-hand side under the stated scalar constraint yields Σ = _𝑠𝐼𝑑_ by the calculations in (a)–(d). Equality in Lemma 6 holds if and only if _𝑝_ is Gaussian with that covariance, hence _𝑝𝐺_ uniquely attains the bound. □ □ 

## **B.5 Proof of lemma. 5** 

� _Proof._ Write the numerator and denominator of _𝑚_ ( _𝑥_ ) as 

**==> picture [238 x 29] intentionally omitted <==**

� so that _𝑚_ ( _𝑥_ ) = _𝐴[𝐵][𝑛] 𝑛_[(] ( _[𝑥] 𝑥_[)] )[.] _[Bias.]_[Compute expectations using independence and change of variables.][For the denominator,] 

**==> picture [285 x 146] intentionally omitted <==**

32 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

where we used symmetry ∫ _𝑡𝐾_ ( _𝑡_ ) _𝑑𝑡_ = 0 and isotropy ∫ _𝑡𝑡_[⊤] _𝐾_ ( _𝑡_ ) _𝑑𝑡_ = _𝜇_ 2( _𝐾_ ) _𝐼𝑑_ , which implies ∫ _𝑡_[⊤] ∇[2] _𝑝_ ( _𝑥_ ) _𝑡𝐾_ ( _𝑡_ ) _𝑑𝑡_ = _𝜇_ 2( _𝐾_ )tr(∇[2] _𝑝_ ( _𝑥_ )) = _𝜇_ 2( _𝐾_ )Δ _𝑝_ ( _𝑥_ ). Similarly, for the numerator, 

**==> picture [327 x 106] intentionally omitted <==**

where the last step uses the fact that tr[�] ∇[2] ( _𝑚𝑝_ )[�] = _𝑝_ Δ _𝑚_ + _𝑚_ Δ _𝑝_ + 2∇ _𝑚_[⊤] ∇ _𝑝_ by the product rule and symmetry of mixed derivatives. 

Now expand the ratio E[E] [[[] _𝐴[𝐵][𝑛] 𝑛_[(] ( _[𝑥] 𝑥_[)]] )][using the identity] 

**==> picture [207 x 28] intentionally omitted <==**

with _𝑎_ 0 = _𝑚_ ( _𝑥_ ) _𝑝_ ( _𝑥_ ), _𝑎_ 2 = _[𝜇]_[2] 2[(] _[𝐾]_[)] � _𝑝_ Δ _𝑚_ + _𝑚_ Δ _𝑝_ + 2∇ _𝑚_[⊤] ∇ _𝑝_[�] ( _𝑥_ ), _𝑏_ 0 = _𝑝_ ( _𝑥_ ), and _𝑏_ 2 = _[𝜇]_[2] 2[(] _[𝐾]_[)] Δ _𝑝_ ( _𝑥_ ). This yields 

**==> picture [321 x 58] intentionally omitted <==**

which recovers our statement. _Variance._ Linearize � _𝑚_ ( _𝑥_ ) = _𝐵𝑛_ ( _𝑥_ )/ _𝐴𝑛_ ( _𝑥_ ) around (E[ _𝐵𝑛_ ( _𝑥_ )] _,_ E[ _𝐴𝑛_ ( _𝑥_ )]) and use independence. To leading order, 

**==> picture [111 x 24] intentionally omitted <==**

Compute 

**==> picture [340 x 102] intentionally omitted <==**

while 

**==> picture [116 x 12] intentionally omitted <==**

Therefore, 

completing the proof. 

**==> picture [388 x 43] intentionally omitted <==**

## **B.6 Proof of Equation (5) to Equation (6)** 

¯ 1 _𝑉𝑔 Proof._ Let **z** = _𝑉𝑔_ � _𝑣_ =1 **[z]** _[𝑛,𝑣]_[denote the mean of the first] _[ 𝑉][𝑔]_[vectors.] 

33 

## **LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

We prove that: 

**==> picture [364 x 33] intentionally omitted <==**

Expanding the left-hand side: 

**==> picture [394 x 147] intentionally omitted <==**

Expanding the right-hand side: 

**==> picture [356 x 67] intentionally omitted <==**

To complete the proof, we verify that: 

**==> picture [305 x 33] intentionally omitted <==**

Expanding the right-hand side: 

**==> picture [320 x 117] intentionally omitted <==**

Therefore, LHS = RHS, completing the proof. 

**==> picture [8 x 6] intentionally omitted <==**

## **B.7 Proof of thm. 8** 

_Proof._ For each _𝑥_ , 

**==> picture [262 x 24] intentionally omitted <==**

34 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

Square and integrate against _𝑝_ ( _𝑥_ ): 

**==> picture [392 x 84] intentionally omitted <==**

where we used ( _𝑎_ + _𝑏_ )[2] ≤ 2 _𝑎_[2] + 2 _𝑏_[2] pointwise. Since |Δ _𝑚_ ( _𝑥_ )| ≤ _𝐵_ for all _𝑥_ , we have 

**==> picture [114 x 24] intentionally omitted <==**

For the second term, first use Cauchy–Schwarz and then integrate against _𝑝_ ( _𝑥_ ) to obtain 

**==> picture [316 x 41] intentionally omitted <==**

which can be combined with the bounds above to obtain the desired result. We similarly have for the integrated variance 

**==> picture [431 x 44] intentionally omitted <==**

which is independent of _𝑝_ . 

## **B.8 Proof of lemma. 3** 

_Proof._ We first start by reminding the reader about the original Cramér-Wold theorem that is a function of all possible directions (not unit-norm ones). 

## **Theorem 10: Cramér-Wold Cramér and Wold [1936]** 

Let _𝑋_ and _𝑌_ be random vectors in R _[𝐷]_ : 

**==> picture [329 x 14] intentionally omitted <==**

_𝑑_ Our proof will follow the same proof as for thm. 10. Necessity is immediate: if _𝑋_ = _𝑌_ , then every measurable function of _𝑋_ has the same distribution as the corresponding function of _𝑌_ , from which the linear mapping _𝑥_ ↦→⟨ _𝑢, 𝑥_ ⟩ for _𝑢_ ∈ S _[𝑑]_[−][1] is a special case. For sufficiency, assume ⟨ _𝑢, 𝑋_ ⟩ = _𝑑_ ⟨ _𝑢, 𝑌_ ⟩ for all _𝑢_ ∈ S _𝑑_ −1. Let _𝜑𝑋_ ( _𝑡_ ) := E� _𝑒[𝑖]_[⟨] _[𝑡,𝑋]_[⟩][�] and _𝜑𝑌_ ( _𝑡_ ) := E� _𝑒[𝑖]_[⟨] _[𝑡,𝑌]_[⟩][�] denote the characteristic functions of _𝑋_ and _𝑌_ . Fix an arbitrary _𝑡_ ∈ R _[𝑑]_ ; if _𝑡_ = 0, then _𝜑𝑋_ (0) = _𝜑𝑌_ (0) = 1. If _𝑡_ ≠ 0, write _𝑑 𝑡_ = _𝑠𝑢_ with _𝑠_ := ∥ _𝑡_ ∥ _>_ 0 and _𝑢_ := _𝑡_ /∥ _𝑡_ ∥∈ S _[𝑑]_[−][1] . By the assumption, ⟨ _𝑢, 𝑋_ ⟩ = ⟨ _𝑢, 𝑌_ ⟩, hence for this _𝑢_ and _𝑠_ we have 

**==> picture [310 x 14] intentionally omitted <==**

Thus _𝜑𝑋_ ( _𝑡_ ) = _𝜑𝑌_ ( _𝑡_ ) for all _𝑡_ ∈ R _[𝑑]_ , i.e., _𝜑𝑋_ ≡ _𝜑𝑌_ on R _[𝑑]_ . By the uniqueness theorem for characteristic functions, this _𝑑_ implies _𝑋_ = _𝑌_ . (ii) Define _𝜓𝑛,𝑡_ := E� _𝑒[𝑖]_[⟨] _[𝑡,𝑋][𝑛]_[⟩][�] and _𝜓𝑡_ := E� _𝑒[𝑖]_[⟨] _[𝑡,𝑋]_[⟩][�] . Fix _𝑡_ ∈ R _[𝑑]_ and decompose _𝑡_ = _𝑠𝑢_ with _𝑠_ := ∥ _𝑡_ ∥≥ 0 and _𝑢_ ∈ S _[𝑑]_[−][1] (take, e.g., _𝑢_ = _𝑡_ /∥ _𝑡_ ∥ if _𝑡_ ≠ 0, and any _𝑢_ if _𝑡_ = 0). The map _𝑔𝑠_ : R → R, _𝑔𝑠_ ( _𝑥_ ) = _𝑠𝑥_ , is continuous. By the _𝑑_ continuous mapping theorem applied to the real-valued random variables ⟨ _𝑢, 𝑋𝑛_ ⟩ −→⟨ _𝑢, 𝑋_ ⟩, we obtain 

**==> picture [174 x 15] intentionally omitted <==**

35 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

_𝑑_ Hence, for every fixed _𝑡_ ∈ R _[𝑑]_ , the one-dimensional projections satisfy ⟨ _𝑡, 𝑋𝑛_ ⟩ −→⟨ _𝑡, 𝑋_ ⟩, which in turn yields pointwise convergence of characteristic functions: 

**==> picture [258 x 13] intentionally omitted <==**

_𝑑_ Therefore, by Lévy’s continuity theorem, _𝑋𝑛_ −→ _𝑋_ . This completes the proof. □ 

## **B.9 Proof of thm. 2** 

_Proof._ We first formulate the following assumptions required for the proof–all of this are satisfied by typical univariate statistical tests. 

_𝑃_ = _𝑄_ if and only if _𝑃𝑎_ = _𝑄𝑎_ for all _𝑎_ ∈ _𝑆[𝑑]_[−][1] (population-level equivalence of laws). 

_𝐴𝑛_ are finite sets with mesh Δ( _𝐴𝑛_ ) := sup _𝑢_ ∈ _𝑆𝑑_ −1 min _𝑎_ ∈ _𝐴𝑛_ ∥ _𝑢_ − _𝑎_ ∥→ 0 as _𝑛_ →∞. 

If _𝑃_ ≠ _𝑄_ , there exists a separating direction _𝑎[★]_ ∈ _𝑆[𝑑]_[−][1] and a neighborhood _𝑈_ of _𝑎[★]_ such that 

**==> picture [128 x 16] intentionally omitted <==**

(Intuitively: near a truly separating direction, the 1D statistic eventually exceeds the global null threshold with probability → 1.) 

(i) Under _𝐻_ 0 : _𝑃_ = _𝑄_ , our assumption implies no separating direction exists at the population level, and the calibration of _𝑢𝑛_ ( _𝛼_ ) ensures Pr( _𝑀𝑛_ ≥ _𝑢𝑛_ ( _𝛼_ )) ≤ _𝛼_ for all _𝑛_ , hence lim sup _𝑛_ →∞ Pr(Ψ _𝑛_ = 1) ≤ _𝛼_ . (ii) Suppose _𝑃_ ≠ _𝑄_ . Our assumption guarantees that there exists at least one separating direction _𝑎[★]_ with _𝑃𝑎★_ ≠ _𝑄𝑎★_ . Our assumption guarantees a neighborhood _𝑈_ of _𝑎[★]_ in which the projection statistics exceed the global null threshold with probability tending to 1: 

**==> picture [134 x 17] intentionally omitted <==**

By assumption, for all large _𝑛_ the set _𝐴𝑛_ contains at least one direction _𝑎𝑛_ ∈ _𝑈_ (dense coverage). Therefore, 

**==> picture [266 x 13] intentionally omitted <==**

which proves consistency. 

□ 

## **B.10 Proof of thm. 5** 

_Proof._ For each case, consider the function _𝑔_ ( _𝑎_ ) on S _[𝐷]_[−][1] defined by the quantity of interest (CF, CDF, or moment) at a fixed _𝑡_ or _𝑘_ . Since _𝑓_ ∈ _𝐻[𝛼]_ (R _[𝐷]_ ), the mapping _𝑎_ ↦→ _𝑔_ ( _𝑎_ ) is in _𝐻[𝛼]_ (S _[𝐷]_[−][1] ) for each fixed _𝑡_ or _𝑘_ . 

Given _𝑀_ samples { _𝑎𝑖_ } _𝑖[𝑀]_ =1[on the sphere, the best possible reconstruction of] _[𝑔]_[from its values at these points is given by] spherical interpolation. By classical results on Sobolev spaces and spherical harmonics (see, e.g., Narcowich et al. [2006]), the _𝐿_[2] interpolation error for functions in _𝐻[𝛼]_ (S _[𝐷]_[−][1] ) using _𝑀_ points is bounded by 

**==> picture [224 x 17] intentionally omitted <==**

where _𝑔_[∗] is the interpolant matching _𝑔_ at the _𝑀_ sampled points. The interpolation error bound on the sphere follows from the theory of spherical harmonics and Marcinkiewicz–Zygmund (MZ) inequalities . Any _𝑓_ ∈ _𝐻[𝛼]_ (S _[𝑑]_ ) admits a spherical harmonics expansion, and the best _𝐿_[2] approximation by harmonics of degree at most _𝐿_ satisfies 

**==> picture [170 x 14] intentionally omitted <==**

where _𝑃𝐿 𝑓_ is the projection onto harmonics of degree ≤ _𝐿_ [Narcowich et al., 2006, Lemma 2.1]. If _𝑀_ points are distributed quasi-uniformly on S _[𝑑]_ , then for _𝐿_ ∼ _𝑐𝑀_[1][/] _[𝑑]_ , the set forms a Marcinkiewicz–Zygmund (MZ) set for degree _𝐿_ [Mhaskar et al., 2001, Theorem 1.1]. This allows reconstruction of any function in the space of harmonics of degree at most _𝐿_ from its values at these points, and the _𝐿_[2] interpolation error for _𝑓_ is bounded by 

**==> picture [178 x 14] intentionally omitted <==**

36 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

where _𝐼𝑀 𝑓_ is any interpolant matching _𝑓_ at the _𝑀_ points [Narcowich et al., 2006, Theorem 3.1]. Substituting _𝐿_ ∼ _𝑐𝑀_[1][/] _[𝑑]_ yields the rate _𝑀_[−] _[𝛼]_[/] _[𝑑]_ , and thus 

**==> picture [204 x 16] intentionally omitted <==**

with explicit _𝐶_ ( _𝑑, 𝛼_ ) as in the main theorem. Integrating (or summing) over _𝑡_ (for CF and CDF) or _𝑘_ (for moments, with weights _𝑤𝑘_ ) yields the stated bounds. The explicit constant _𝐶_ ( _𝐷, 𝛼_ ) arises from the theory of spherical Sobolev spaces and is given above. 

For the moment case, the sum over _𝑘_ is weighted to ensure convergence, as higher moments may grow rapidly. The weights _𝑤𝑘_ can be chosen, for example, as _𝑤𝑘_ = 1/ _𝑘_ !. 

This completes the proof. □ 

## **B.11 Proof of thm. 3** 

Pick distinct _𝑥_ 0 _, . . . , 𝑥𝐾_ +1 ∈ R and consider the linear map _𝐴_ : R _[𝐾]_[+][2] → R _[𝐾]_[+][1] , ( _𝐴𝑝_ ) _𝑟_ =[�] _[𝐾] 𝑗_ =[+] 0[1] _[𝑝][𝑗][𝑥][𝑟] 𝑗_[for] _[ 𝑟]_[=][ 0] _[, . . . , 𝐾]_[.][Then] rank( _𝐴_ ) ≤ _𝐾_ + 1, so ker( _𝐴_ ) ≠ {0}. Let _𝑣_ ∈ ker( _𝐴_ ) \ {0}; from ( _𝐴𝑝_ )0 =[�] _𝑗[𝑝][𝑗]_[, we get][ �] _𝑗[𝑣][𝑗]_[=][0][, hence] _[ 𝑣]_[has positive and] negative entries. Choose a strictly positive probability vector _𝑝_ and _𝜀>_ 0 small such that _𝑝_[±] := _𝑝_ ± _𝜀𝑣_ remain probability vectors. Then _𝐴𝑝_[+] = _𝐴𝑝_[−] , so the distributions supported on { _𝑥 𝑗_ } with masses _𝑝_[±] are distinct yet match moments up to order _𝐾_ . 

## **B.12 Proof of thm. 4** 

_Proof._ Fix the Gaussian weight 

**==> picture [108 x 14] intentionally omitted <==**

and define the population CF distance 

**==> picture [162 x 25] intentionally omitted <==**

Let the empirical CF be 

**==> picture [88 x 31] intentionally omitted <==**

and consider the V-statistic estimator 

**==> picture [144 x 25] intentionally omitted <==**

We use only that | _𝑒[𝑖𝑡𝑋]_ | = 1, | _𝜑𝑃_ ( _𝑡_ )| ≤ 1, | _𝜑𝐺_ ( _𝑡_ )| ≤ 1, and integrability of _𝑤𝑠_ . For each _𝑖_ differentiate under the integral (dominated convergence applies because the integrand and its derivative are bounded) 

**==> picture [217 x 56] intentionally omitted <==**

since | _𝜑_ � _𝑁_ ( _𝑡_ )| ≤ 1 and | _𝜑𝐺_ ( _𝑡_ )| ≤ 1, 

**==> picture [184 x 85] intentionally omitted <==**

37 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

using ∫R _[𝑒]_[−] _[𝑠]_[2] _[𝑡]_[2][|] _[𝑡]_[|] _[𝑑𝑡]_[=][ 1][/] _[𝑠]_[2][.] 

**==> picture [171 x 94] intentionally omitted <==**

Moreover, differentiating once more in _𝑋𝑖_ and using | _𝜑_ � _𝑁_ ( _𝑡_ )| ≤ 1, | _𝜑𝐺_ ( _𝑡_ )| ≤ 1 gives a global Lipschitz bound 

for some absolute constant _𝐶_ arising from bounded factors and product rule. Hence ECF gradients are uniformly bounded and Lipschitz, with scale controlled only by ( _𝑁, 𝑠_ ). 

(B) (Moment sample-gradients are polynomial in _𝑋𝑖_ and unbounded for _𝑘_ ≥ 2.) Let _𝐷_[�] _𝑉_ be as above. Define the moment objective 

**==> picture [326 x 31] intentionally omitted <==**

for a symmetric positive semidefinite _𝑊_ ∈ R _[𝑘]_[×] _[𝑘]_ and Gaussian target moments _𝜇_ = E _𝐺_ [ _𝜙_ ( _𝑌_ )]. For each _𝑖_ , 

**==> picture [151 x 52] intentionally omitted <==**

The gradient formula follows by the chain rule and linearity of _𝜙_[¯] . Let _𝑐_ := _𝑊_ ( _𝜙_[¯] − _𝜇_ ) and write _𝑐𝑟_ for its _𝑟_ -th coordinate. Then 

**==> picture [98 x 31] intentionally omitted <==**

which is a polynomial in _𝑋𝑖_ of degree deg = max{ _𝑟_ − 1 : _𝑐𝑟_ ≠ 0} ≤ _𝑘_ − 1. In particular, if _𝑐𝑘_ ≠ 0 (the generic case when the top-weighted deviation is nonzero), then 

**==> picture [136 x 33] intentionally omitted <==**

The expression is a nonconstant polynomial in _𝑋𝑖_ of degree deg ≤ _𝑘_ − 1 whenever some _𝑐𝑟_ ≠ 0 with _𝑟_ ≥ 2. Thus the gradient cannot be uniformly bounded on R. If _𝑐𝑘_ ≠ 0, the leading term dominates and the magnitude grows like | _𝑋𝑖_ | _[𝑘]_[−][1] , proving unboundedness for _𝑘_ ≥ 2. □ 

## **B.13 Proof of thm. 6** 

_𝑛 Proof._ A direct calculation shows Fix _𝑡_ ∈ R _[𝑑]_ and abbreviate _𝑍𝑗_ � _𝑒_[i] _[𝑡]_[⊤] _[𝑋][𝑗]_ , so that _𝜙𝑛_ ( _𝑡_ ) = _𝑛_[1] � _𝑗_ =1 _[𝑍][𝑗]_[.][Note that][ |] _[𝑍][𝑗]_[|][=][1] almost surely (since _𝑡_[⊤] _𝑋𝑗_ ∈ R), and E[ _𝑍𝑗_ ] = _𝜙𝜃_ ( _𝑡_ ) for all _𝑗_ . We start from the algebraic identity 

**==> picture [266 x 16] intentionally omitted <==**

38 

## **LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

Taking expectations term by term gives 

**==> picture [384 x 175] intentionally omitted <==**

**==> picture [18 x 11] intentionally omitted <==**

Since the _𝑍𝑗_ are i.i.d., 

**==> picture [219 x 33] intentionally omitted <==**

hence 

**==> picture [143 x 78] intentionally omitted <==**

Plugging these, we obtain 

**==> picture [232 x 90] intentionally omitted <==**

Under Dominated convergence, E[∇ _𝜃𝐷𝑛_ ( _𝑡_ )] = ∇ _𝜃_ E[ _𝐷𝑛_ ( _𝑡_ )], hence 

**==> picture [214 x 24] intentionally omitted <==**

concluding the proof. 

In practice one replaces ∫R _[𝑤]_[(] _[𝑡]_[)(·)] _[𝑑𝑡]_[by a deterministic quadrature on a uniform grid] _[ 𝑡][𝑘]_[∈[−] _[𝑇, 𝑇]_[]][ with weights] _[ 𝜔][𝑘]_[(e.g.] trapezoidal rule) and a Gaussian window _𝑤_ ( _𝑡_ ) = _𝑒_[−] _[𝛼][𝑡]_[2] . All statements above remain valid with the integral replaced by � _𝑘[𝜔][𝑘]_[(·)][:] 

**==> picture [280 x 24] intentionally omitted <==**

39 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

and the bias term becomes 

**==> picture [140 x 28] intentionally omitted <==**

Since the grid and weights are deterministic, they do not affect unbiasedness with respect to sampling; they only introduce a deterministic approximation error to the target functional _𝐿_ ( _𝜃_ ). 

**==> picture [8 x 6] intentionally omitted <==**

## **B.14 Proof of VICReg’s Recovery** 

_Proof._ We prove this result in two parts. 

**Part I:** E[ **X** ] = **0** Given that E[⟨ **X** _,_ **a** ⟩] = 0 for all unit vectors **a** , and noting that ⟨ **X** _,_ **a** ⟩ = **a** _[𝑇]_ **X** , we have: 

**==> picture [346 x 12] intentionally omitted <==**

By linearity of expectation: 

**==> picture [334 x 12] intentionally omitted <==**

Let _**𝝁**_ = E[ **X** ]. We claim that _**𝝁**_ = **0** . Suppose, for the sake of contradiction, that _**𝝁**_ ≠ **0** . Then ∥ _**𝝁**_ ∥2 _>_ 0. Define the unit vector: 

**==> picture [282 x 24] intentionally omitted <==**

Since **a**[∗] is a unit vector, equation (33) implies: 

**==> picture [283 x 13] intentionally omitted <==**

However, substituting the definition of **a**[∗] : 

**==> picture [360 x 29] intentionally omitted <==**

This contradiction establishes that _**𝝁**_ = **0** . 

**Part II:** Cov( **X** ) = **I** _𝑑_ Since E[ **X** ] = **0** , we have: 

**==> picture [341 x 12] intentionally omitted <==**

Expanding the quadratic form: 

**==> picture [521 x 54] intentionally omitted <==**

We now show that 𝚺 = **I** _𝑑_ . _Step 1: Diagonal entries._ For _𝑖_ ∈{1 _,_ 2 _, . . . , 𝑑_ }, let **e** _𝑖_ denote the _𝑖_ -th standard basis vector. Setting **a** = **e** _𝑖_ in equation (40): 

**==> picture [295 x 14] intentionally omitted <==**

Therefore, all diagonal entries of 𝚺 equal 1. _Step 2: Off-diagonal entries._ For distinct indices _𝑖, 𝑗_ ∈{1 _,_ 2 _, . . . , 𝑑_ }, consider the unit vector: 

**==> picture [312 x 25] intentionally omitted <==**

Applying equation (40): 

**==> picture [331 x 21] intentionally omitted <==**

40 

## **LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

Expanding the quadratic form and using the symmetry of 𝚺: 

**==> picture [327 x 102] intentionally omitted <==**

Therefore, all off-diagonal entries of 𝚺 equal zero, establishing that 𝚺 = **I** _𝑑_ . 

**==> picture [8 x 7] intentionally omitted <==**

## **C Background** 

**Foundation: The Linear Regression Model** We start with the standard linear regression model: 

**==> picture [48 x 10] intentionally omitted <==**

where: 

- **y** = [ _𝑦_ 1 _, 𝑦_ 2 _, . . . , 𝑦𝑛_ ] _[𝑇]_ ∈ R _[𝑛]_ is the response vector 

- **X** ∈ R _[𝑛]_[×] _[𝑝]_ is the design matrix with **X** _𝑖𝑗_ = _𝑥𝑖𝑗_ 

- _**𝜷**_ = [ _𝛽_ 1 _, 𝛽_ 2 _, . . . , 𝛽𝑝_ ] _[𝑇]_ ∈ R _[𝑝]_ is the parameter vector 

- _**𝜺**_ = [ _𝜀_ 1 _, 𝜀_ 2 _, . . . , 𝜀𝑛_ ] _[𝑇]_ ∼𝒩( **0** _, 𝜎_[2] **I** _𝑛_ ) is the error vector 

The error assumption means: 

**==> picture [222 x 13] intentionally omitted <==**

**Step 1: Deriving the OLS Estimator** To find the OLS estimator, we minimize the sum of squared residuals: 

**==> picture [194 x 29] intentionally omitted <==**

Expanding this quadratic form: 

Taking the derivative with respect to _**𝜷**_ : 

**==> picture [337 x 61] intentionally omitted <==**

Setting equal to zero and solving: 

Assuming **X** _[𝑇]_ **X** is invertible: 

**==> picture [88 x 71] intentionally omitted <==**

## **D Details on Low-Discrepancy Sequences** 

Quasi-Monte Carlo (QMC) methods, such as the Sobol sequence, are widely used to generate low-discrepancy samples in the unit hypercube, providing improved uniformity over purely random sampling. To obtain samples uniformly distributed on the hypersphere, each QMC point is mapped to a standard normal vector via the inverse cumulative 

41 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [256 x 183] intentionally omitted <==**

**----- Start of picture text -----**<br>
Decay of Error Bound Constant vs. Number of Directions (D=5)<br>250 α =1 α =81 α =141<br>α =21 α =101 α =161<br>200 α =41 α =121 α =181<br>α =61<br>150<br>100<br>50<br>0<br>− 50<br>− 100<br>25 50 75 100 125 150 175<br>Number of directions M<br>scale)<br>(log<br>α/d 2 −<br> M<br> ·<br>)<br>d, α<br>(<br>C<br>**----- End of picture text -----**<br>


**Figure 15.** Depiction of the expected BCS loss upper bound (thm. 5) for various smoothness values _𝛼_ . We clearly see that as the smoothness increases ( **blue to red** ), as the upper bound decreases more and more rapidly with _𝑀_ . 

**Table 3.** Performance metrics across different sample sizes from Figure 12 

|**Freeze Backbone**|**Model Name**|<br>**Samples per Class**<br>**All**<br>**1**<br>**2**<br>**5**<br>**10**<br>**100**<br>**1000**|
|---|---|---|
|**No**|_LeJEPA (Ours_<br>ConvNeXt-V<br>LeViT-128<br>ResNet-18<br>ResNet-34|_)_<br>2 Nano<br>82.72<br>**29.42**<br>**36.65**<br>**50.94**<br>**59.85**<br>**75.34**<br>81.97<br>79.41<br>18.45<br>24.08<br>33.11<br>41.76<br>64.59<br>77.59<br>82.15<br>23.34<br>31.56<br>43.82<br>54.64<br>73.53<br>81.41<br>**83.28**<br>24.27<br>31.51<br>44.23<br>53.95<br>74.93<br>**82.32**<br>ll<br>78.34<br>21.05<br>21.71<br>30.33<br>36.23<br>60.81<br>75.55<br>S/16<br>81.60<br>24.71<br>29.43<br>37.71<br>44.71<br>69.87<br>80.54|
||_Baselines_<br>DINOv2 Sma<br>DINOv3 ViT-||
|**Yes**|_LeJEPA (Ours_<br>ConvNeXt-V<br>LeViT-128<br>ResNet-18<br>ResNet-34|_)_<br>2 Nano<br>76.52<br>28.74<br>36.65<br>50.60<br>59.50<br>72.62<br>77.24<br>69.00<br>25.85<br>33.30<br>45.52<br>52.43<br>64.37<br>69.39<br>75.95<br>30.48<br>38.22<br>50.85<br>58.86<br>72.70<br>76.39<br>**78.17**<br>**31.08**<br>**38.33**<br>**52.26**<br>**60.63**<br>**74.77**<br>**78.62**<br>ll<br>67.62<br>27.68<br>32.22<br>40.72<br>47.72<br>62.49<br>67.89<br>S/16<br>71.38<br>30.17<br>36.65<br>45.74<br>51.51<br>65.90<br>71.35|
||_Baselines_<br>DINOv2 Sma<br>DINOv3 ViT-||



**Table 4.** Top 1 accuracy (in %) with LeJEPA pretraining on Imagenet-100 for 400 epochs (All values are percentages) 

|**Table 4.** Top 1 acc|uracy (in %) with LeJEPA pretraining on Imagenet-100 for 400 epochs (All values are percentages)|uracy (in %) with LeJEPA pretraining on Imagenet-100 for 400 epochs (All values are percentages)|uracy (in %) with LeJEPA pretraining on Imagenet-100 for 400 epochs (All values are percentages)|
|---|---|---|---|
|backbone<br>Projector<br>w/predictor<br>w/ SWA|resnet50<br>vit_small_patch8_224<br>vit_tiny_patch8_224<br>1-layer<br>2-layer<br>3-layer<br>1-layer<br>2-layer<br>3-layer<br>1-layer<br>2-layer<br>3-layer|||
|||||
|False<br>False<br>True<br>True<br>False<br>True|79.71<br>82.44<br>79.79<br>82.69<br>79.41<br>82.44<br>78.87<br>82.04|83.93<br>76.59<br>80.77<br>81.07<br>83.50<br>79.96<br>83.63<br>84.12<br>83.57<br>77.58<br>79.41<br>81.91<br>82.82<br>77.11<br>81.77<br>82.58|71.79<br>76.87<br>80.37<br>75.86<br>82.36<br>80.50<br>67.74<br>77.64<br>80.73<br>69.53<br>78.27<br>79.77|



distribution function (CDF), and then projected onto the sphere by normalization. This approach leverages the rotational invariance of the multivariate normal distribution, ensuring that the resulting directions are uniformly distributed on 

42 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

**Table 5. Small architecture in-domain LeJEPA pretraining** from random initialization across datasets and architectures, with frozen backbone linear evaluation. First, **LeJEPA is able to produce near state-of-the-art performances on tiny dataset with only a thousand samples** , e.g., flowers102. Second, **on non-natural image data, LeJEPA clearly outperforms the latest frontier vision models** , e.g., Galaxy10. See Figure 12 for additional experiments with varying number of training samples and with full finetuning. 

||Pretraining|fowers102|cifar100|food101|inet10|cifar10|galaxy10|
|---|---|---|---|---|---|---|---|
||# train. samples|1020|50000|75750|13000|50000|11008|
|LeJEPA (convnextv2_nano) 14M|in-domain|64.34|69.26|69.59|90.81|92.22|76.05|
|LeJEPA (resnet18) 11M|in-domain|74.57|69.94|73.57|92.36|92.51|75.32|
|LeJEPA (resnet34) 21M|in-domain|71.85|70.44|74.95|92.80|93.16|77.29|
|LeJEPA (resnext26ts) 8M|in-domain|82.19|69.10|76.77|92.82|91.59|73.78|
|LeJEPA (swin_tiny) 27M|in-domain|63.94|65.08|78.40|92.87|92.67|74.89|
|ĲEPA-inet22k (ViT-H/14) 630M|inet1k|85.76|86.93|81.06|98.65|97.77|62.93|



**Table 6.** Time (in millisecond) to compute the proposed SIGReg loss from algorithm 1 on a Tesla V100-SXM2-16GB for varying mini-batch size ( _𝑁_ ), number of slices ( _𝑀_ ), integration points. Results are computed over 10 runs. 

||N<br>M<br># integration<br>points<br>mean (ms)<br>std (ms)<br>512<br>512<br>16<br>0.465236<br>0.011642<br>512<br>512<br>64<br>0.461317<br>0.003894<br>512<br>512<br>256<br>0.627644<br>0.003337<br>2048<br>512<br>16<br>1.406441<br>0.002415<br>8192<br>512<br>16<br>6.188304<br>0.007226<br>8192<br>8192<br>16<br>8.685009<br>0.038829<br>32768<br>512<br>16<br>26.373118<br>0.012732<br>512<br>2048<br>16<br>0.465614<br>0.005274<br>512<br>8192<br>16<br>0.670379<br>0.006854<br>**Table 7.** Number of Figure 8.|
|---|---|
|_𝜆_<br>0.001<br>0.005<br>0.010<br>#views|resnet50<br><br>0.020<br>0.025<br>0.050<br>0.100<br>0.150<br>0.200<br>0.300<br>0.400<br>0.500|
|2<br>81.41<br>82.73<br>83.49<br>4<br>79.88<br>83.04<br>84.36<br>8<br>76.67<br>81.58<br>83.59|82.99<br>82.23<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br><br>84.68<br>84.33<br>83.00<br>82.91<br>81.05<br>78.58<br>-<br>-<br>-<br><br>83.49<br>83.76<br>84.32<br>83.66<br>83.07<br>82.16<br>81.00<br>79.25<br>77.72|



the sphere’s surface. While the low-discrepancy property is not strictly preserved under this nonlinear mapping, the resulting samples are empirically more uniform than random samples and are standard in high-dimensional applications Marsaglia [1972], Dick and Pillichshammer [2010], Caflisch [1998]. 

**Require:** Number of points _𝑁_ , dimension _𝑑_ **Ensure:** Points { **y** _𝑖_ } _[𝑁] 𝑖_ =1[quasi-uniformly distributed on][ S] _[𝑑]_[−][1] 

- 1: **for** _𝑖_ = 1 to _𝑁_ **do** 

- 2: Generate **x** _𝑖_ ∈[0 _,_ 1] _[𝑑]_ as the _𝑖_ -th point of a Sobol sequence 3: Transform each component: _𝑧𝑖,𝑗_ = Φ[−][1] ( _𝑥𝑖,𝑗_ ) for _𝑗_ = 1 _, . . . , 𝑑 ⊲_ Φ[−][1] is the inverse CDF of the standard normal 

- 4: Normalize: **y** _𝑖_ = **z** _𝑖_ /∥ **z** _𝑖_ ∥2 5: **end for** 

## **E Shapiro-Wilk Test** 

Let X1 < X2 < . . . < Xn denote an ordered random sample of size n from a standard normal distribution. Also, let mÂ 5 (m1,m2,...,mn) be the vector of expected values of standard normal order statistics, and let V 5 (vĳ ) be the corresponding 

43 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

n 3 n covariance matrix, so that 

**==> picture [386 x 13] intentionally omitted <==**

The W test statistic Shapiro and Wilk [1965] for normality is then denoted by 

**==> picture [361 x 52] intentionally omitted <==**

Shapiro and Francia [1972] suggested replacing the covariance matrix V by the identity matrix I, because for large samples, the observations Yi may be treated as if they are independent (see Gupta [1952]). Another asymptotic extension was suggested by Weisburg and Binham [1975] 

**==> picture [354 x 33] intentionally omitted <==**

building atop Elfving [1947]’s approximation but using 3/8 instead of _𝜋_ /8. 

Rahman and Govindarajulu [1997] proposed another variation using the approximation for the expected values of order statistics given by Blom [1958] and the approximations for the elements of the variance± covariance matrix given by Blom [1958], Mosteller [2006]. These approximations are 

**==> picture [357 x 26] intentionally omitted <==**

**==> picture [390 x 26] intentionally omitted <==**

**==> picture [284 x 22] intentionally omitted <==**

We know (see Hammersley and Morton [1954], Plackett [1958]) 

**==> picture [454 x 82] intentionally omitted <==**

## **F Multivariate Statistics** 

We ideally would like to compare the distributions. One slight variation is to compare the Characteristic function of the distributions. Given samples _𝒙_ 1 _, . . . , 𝒙𝑁_ , the Empirical Characteristic Function (ECF) is defined as 

**==> picture [100 x 31] intentionally omitted <==**

We can now compare our ECF to the one of the target distribution and build the statistic 

**==> picture [266 x 24] intentionally omitted <==**

44 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [466 x 103] intentionally omitted <==**

with _𝛽>_ 0, Baringhaus-Henze-Epps-Pulley. From[1] leading to the HZ test[2] uses 

**==> picture [324 x 14] intentionally omitted <==**

the same can be done with the moment generating function[3] 

**==> picture [243 x 78] intentionally omitted <==**

here with _𝛽>_ 2 There is also one combining both[4] ! 

**==> picture [324 x 26] intentionally omitted <==**

**==> picture [410 x 222] intentionally omitted <==**

and its simplified version 

Also one testing the derivative[5] 

**==> picture [354 x 24] intentionally omitted <==**

> 1 `https://www.routledge.com/Density-Estimation-for-Statistics-and-Data-Analysis/Silverman/p/book/9780412246203?srsltid= AfmBOoodlL-CtlqL0JVC-LcP6mOWw6VTt51_YstdZOW4W3iuicu1VFyg` 

> 2 `https://www.tandfonline.com/doi/abs/10.1080/03610929008830400` 

> 3 `https://arxiv.org/pdf/1711.07199` 

> 4 `https://arxiv.org/pdf/1706.03029` 

> 5 `https://arxiv.org/pdf/1901.03986` 

45 

## **LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

skewness[6] : 

skewness[7] : 

**==> picture [429 x 144] intentionally omitted <==**

which should be 0 for Gaussian and Kurtosis which should be d(d+2) 

**==> picture [302 x 31] intentionally omitted <==**

> 6 `https://www.jstor.org/stable/2334770` 

> 7 `https://link.springer.com/article/10.1007/s13171-020-00211-6` 

46 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

**==> picture [457 x 633] intentionally omitted <==**

**----- Start of picture text -----**<br>
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|dimension=128, slices=10|
|original|data|VCReg|ExtendedJarqueBera|CramerVonMises|Watson|AndersonDarling|EppsPulley|
|2|“\|&|.|8°|,|oo|.|-|.|.|
|0|
|All|gewy’|||||8|||ob|||||3x|8|-|fo|
|−|2|
|.|Tee,|Ces|°|ee|°°|op|20°|
|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|
|dim|1|dim|1|dim|1|dim|1|dim|1|dim|1|dim|1|
|2|
|0|
|° 5s|ri|$3ea|id oy|.°|Ee,|oc. .|po|
|−|2|ZN...|yoy|=A|Ey|®|ol|{oeKk|Co|Eo|
|©|°0|©|%%,®|*|Ph)|0®°®0,|R|.°|®|8oS|°ee|©|Tol:|°|oe|®0°|
|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|
|dim|3|dim|3|dim|3|dim|3|dim|3|dim|3|dim|3|
|dimension=128, slices=100|
|original|data|VCReg|ExtendedJarqueBera|CramerVonMises|Watson|AndersonDarling|EppsPulley|
|2|°|Pp|Ld|R|
|AY|(]|©|e|©|®|%|®|
|°o|qt|Seo|A|°|5|¢|°|©|I|
|0|
|Ky|4|?.|e?|5|®|.|%|x|oo |||° ofaNee|
|−|2|“s|t|ate||||ENe||NW|||F|KX|oa|
|es|x|4|’|Wg®|i°|°%0¥8|:|Es°|>|o|3X®|(]|°|
|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|
|dim|1|dim|1|dim|1|dim|1|dim|1|dim|1|dim|1|
|2|
|Je|g..|°|Sea|0 va|e085,|R|®,|0|°|o|.|
|0|©|“%|Ce|®e|0|We|H|©|0&5|R|3X|&|be|
|−|2|
|Joe|©|©|¢|Ek|‘|“e|°°|Ach)|oP|°°|VG Pd|
|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|
|dim|3|dim|3|dim|3|dim|3|dim|3|dim|3|dim|3|
|dimension=1024, slices=100|
|original|data|VCReg|ExtendedJarqueBera|CramerVonMises|Watson|AndersonDarling|EppsPulley|
|4|
|2|ales|“38%|EN|.|la|
|0|o®|°|o Co|8|o|-|.|$|4|°|Ps|
|xXo,|[KZ]|°S|°%|8q°"|8|o00|%|°|°|
|−|2|
|−|4|‘|°|© &|oop ©|oe|®e|id|
|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|
|dim|1|dim|1|dim|1|dim|1|dim|1|dim|1|dim|1|
|2|
|0|. I|o|°e I) NY|ya|i|e°,|.|)|°|RT|AL|
|°|"8K iy|®°|RRig.|7,|H|.|’|«3%oH|'d“|°°rs ©|wt,|oe:|3“¢“|RYnies:4b:|odeo|2.Be‘|
|−|2|
|- .|8|.|®|as|Ee|.|Fire|BS|Abe o|a|[f:]|
|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|−|2|.|5|0|.|0|2|.|5|
|dim|3|dim|3|dim|3|dim|3|dim|3|dim|3|dim|3|

**----- End of picture text -----**<br>


**Figure 16.** Reprise of Figure 6 for additional dimensions and number of 1d projections. 

47 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [416 x 623] intentionally omitted <==**

**----- Start of picture text -----**<br>
Distribution of Estimator β [ˆ] (Binary Y) (Normalized)<br>1 . 5 Isotropic<br>True β<br>1 . 0<br>0 . 5<br>0 . 0<br>1 . 5<br>1 . 0<br>0 . 5<br>0 . 0<br>0 . 5 1 . 0 1 . 5 0 . 5 1 . 0 1 . 5 0 . 5 1 . 0 1 . 5 0 . 5 1 . 0 1 . 5<br>β 1 β 1 β 1 β 1<br>Distribution of Estimator β [ˆ] (Linear Y)<br>− 0 . 10<br>Isotropic<br>− 0 . 15 True β<br>− 0 . 20<br>− 0 . 25<br>− 0 . 30<br>− 0 . 10<br>− 0 . 15<br>− 0 . 20<br>− 0 . 25<br>− 0 . 30<br>− 1 . 05 − 1 . 00 − 0 . 95 − 0 . 90 − 1 . 05 − 1 . 00 − 0 . 95 − 0 . 90 − 1 . 05 − 1 . 00 − 0 . 95 − 0 . 90 − 1 . 05 − 1 . 00 − 0 . 95 − 0 . 90<br>β 1 β 1 β 1 β 1<br>Distribution of Estimator β [ˆ] (Smooth Y)<br>Isotropic<br>2<br>0<br>2<br>0<br>0 2 0 2 0 2 0 2<br>β 1 β 1 β 1 β 1<br>β 2<br>β 2<br>β 2<br>β 2<br>β 2<br>β 2<br>**----- End of picture text -----**<br>


**Figure 17.** Depiction of the distribution of optimized _𝛽_ values from OLS when comparing _𝒁_ iso and _𝒁_ aniso from lemmas. 1 and 2. We clearly observe that the anisotropic version ( **blue** ) provides much lower variance compared to the isotropic case ( **red** ). We consider a binary classification (linear separable class) ( **top row** ), a linear regression task ( **middle row** ), and a nonlinear regression task with smooth targets ( **bottom row** ). For each case, we resample the training samples numerous times and produce an estimate for _𝛽_ each time. Because the data is 2-dimensional, we can visualize the _𝛽_ distribution directly. 

48 

**LeJEPA:** 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

**==> picture [565 x 610] intentionally omitted <==**

**----- Start of picture text -----**<br>
Test Accuracy vs Regularization ( N = 50, Linear Classification) Test Accuracy vs Regularization ( N = 200, Linear Classification) Test Accuracy vs Regularization ( N = 1000, Linear Classification)<br>1 . 00 1 . 00<br>0 . 98<br>== IN 0 . 98 —\ 0 . 99<br>0 . 96 0 . 98<br>0 . 96<br>0 . 94 0 . 97<br>0 . 94<br>0 . 96<br>0 . 92<br>0 . 92<br>Isotropic ( κ  = 1) \L Isotropic ( κ  = 1) 0 . 95 I Isotropic ( κ  = 1)<br>0 . 90 AnisotropicAnisotropic (( κκ  = 5) = 10) 0 . 90 ++ Anisotropic Anisotropic ( ( κ κ  = 5)  = 10) += AnisotropicAnisotropic (( κκ  = 5) = 10)<br>Anisotropic ( κ  = 50) *= Anisotropic ( κ  = 50) 0 . 94 = Anisotropic ( κ  = 50)<br>10 [−] [4] 10 [−] [3] 10 [−] [2] 10 [−] [1] 10 [0] 10 [1] 10 [2] 10 [−] [4] 10 [−] [3] 10 [−] [2] 10 [−] [1] 10 [0] 10 [1] 10 [2] 10 [−] [4] 10 [−] [3] 10 [−] [2] 10 [−] [1] 10 [0] 10 [1] 10 [2]<br>Regularization λ Regularization λ Regularization λ<br>Directional Alignment vs Regularization ( N = 50, Linear Regression) Directional Alignment vs Regularization ( N = 200, Linear Regression) Directional Alignment vs Regularization ( N = 1000, Linear Regression)<br>1 . 0 1 . 00 1 . 00<br>0 . 9 0 . 95 0 . 98<br>0 . 90 0 . 96<br>0 . 8<br>0 . 85 0 . 94<br>0 . 7<br>—&+ IsotropicAnisotropic( κ  = 1)( κ  = 5) 0 . 80 &+ IsotropicAnisotropic( κ  = 1)( κ  = 5) 0 . 92 rs.Ea IsotroAnisotropicpic ( κ  = 1( κ  = 5))<br>+ Anisotropic ( κ  = 10) - Anisotropic ( κ  = 10) Anisotropic ( κ  = 10)<br>0 . 6 = Anisotropic ( κ  = 50) 0 . 75 Anisotropic ( κ  = 50) 0 . 90 Anisotropic ( κ = 50)<br>10 [−] [4] 10 [−] [3] 10 [−] [2] 10 [−] [1] 10 [0] 10 [1] 10 [2] 10 = [−] [4] 10 [−] [3] 10 [−] [2] 10 [−] [1] 10 [0] 10 [1] 10 [2] 10 + [−] [4] 10 [−] [3] 10 [−] [2] 10 [−] [1] 10 [0] 10 [1] 10 [2]<br>Regularization λ Regularization λ Regularization λ<br>Figure 18. Depiction of accuracy ( top ) and cosine similarity between estimated and true estimator ( bottom ) for the OLS setting with varying<br>strength of Tikhonov regularization ( x-axis)  comparing isotropic and anisotropic embeddings. As per thm. 6, the anisotropic distribution creates a bias<br>in the OLS estimation for nonzero regularization.<br>Spearman corr.: 98.53% (resnet50 g alaxy10) Spearman corr.: 99.16% (resnet50 i net10) Spearman corr.: 94.52% (ViT/base-8 i net1k)<br>®©<br>λ 10 [1] λ<br>10 [1]<br>10 [1] 3 ph) 0 . 04 o® H: 0 . 04<br>0 . 08 λ 0 . 08<br>3 A ° 0 . 12 0 . 04 - 0 . 12<br>’ ° 0 . 16 1) 0 . 08 LL tae, ® ° 0 . 16<br>10 [0] Le pa CRE Cas:ely. ° 0 . 20 10 [0] ° 00 .. 1216 EERE24 eC 23,® 10 [0] (Rr Yo,LSPS tm { opm 0 . 20<br>Ee) ° 0 . 20 5<br>20 40 60 80 40 60 80 0 20 40 60<br>Test acc. (%) Test acc. (%) Test acc. (%)<br>Spearman corr.: 97.63% (ViT/s-8 g alaxy10) Spearman corr.: 97.97% (ViT/s-8 i net10) Spearman corr.: 93.82% (resnet18 f lowers102)<br><H λ 10 [2] ° λ λ<br>- 0 . 04 :° 0 . 04 “ % 0.01<br>© 0 . 08 0 . 08 0, (TY 0.02<br>10 [1] LC<br>20 e@ ‘ 0 . 12 10 [1] ° ° 0 . 12 ee) 0.05<br>0 . 16 0 . 16 0.1<br>0 . 20 0 . 20 10 [0] 0.2<br>10 [0] (at EYRCCL os -wy,(CW ) 10 [0] YourRLSiam Ay%, = ae,oe<br>20 40 60 80 20 40 60 80 20 40 60<br>Test acc. (%) Test acc. (%) Test acc. (%)<br>Accuracy Accuracy Accuracy<br>Test Test Test<br>)] [cos( ˆ ∗ E  ww , )] [cos( ˆ ∗ E  ww , )] [cos( ˆ ∗ E  ww ,<br>(log-scale) (log-scale) (log-scale)<br>loss loss loss<br>Train Train Train<br>(log-scale) (log-scale) (log-scale)<br>loss loss loss<br>Train Train Train<br>**----- End of picture text -----**<br>


**Figure 19.** Additional figures provides in Figure 19 

49 

Sec 1: Intro | Sec 2: Background | Sec 3: Why Gaussian? | Sec 4: SIGReg | Sec 5: LeJEPA | Sec 6: Experiments 

## **LeJEPA:** 

**==> picture [509 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
N (0 ,  1) 0 . 5 N ( − 2 ,  0 . 5 [2] ) + 0 . 5 N (2 ,  0 . 5 [2] ) Student- t ( ν = 3)<br>Trapezoid ( T true = 1 . 94) 10 [3] Trapezoid ( T true = 55109 . 93) 10 [3] Trapezoid ( T true = 925 . 27)<br>10 [−] [1]<br>10 [2]<br>10 [−] [2] 10 [2]<br>10 [1]<br>10 [−] [3]<br>10 [1]<br>10 [0]<br>10 [−] [4]<br>10 [0] 10 [−] [1]<br>10 [−] [5]<br>10 [−] [2]<br>10 [−] [6] 10 [−] [1]<br>10 [−] [3]<br>20 40 60 80 100 20 40 60 80 100 20 40 60 80 100<br>Number of quadrature points Number of quadrature points Number of quadrature points<br>| true<br>T<br> −<br>n<br> ˆ |T<br>error<br>Absolute<br>**----- End of picture text -----**<br>


**Figure 20.** Proposed trapezoid quadrature for the Epps-Pulley statistic as implemented in algorithm 1. We depict the approximation error of the integral for various distributions, demonstrate rapid convergence (faster than quadratic show in **grey line** ) across possible embedding distributions. 

**==> picture [512 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
ViT/s-8 - galaxy10 79 . 88 ViT/s-8 - inet10 92 . 77 10 [0] resnet18 - flowers102 72 . 18<br>10 [1] o &<br>10 [1] p<br>62 . 41 72 . 08 56 . 90<br>10 [0] 44 . 94 10 [0] 51 . 38 41 . 63<br>10 [−] [1]<br>27 . 47 30 . 69 26 . 35<br>AEE AN PS HW NY 3 [3][ KGa]<br>10 [−] [1]<br>10 [−] [1]<br>10 . 00 10 . 00 11 . 07<br>10 [0] 10 [1] 10 [1] 10 [1]<br>SIGReg loss (log-scale) SIGReg loss (log-scale) SIGReg loss (log-scale)<br>(log-scale) (log-scale) (log-scale)<br>loss Accuracy loss Accuracy loss Accuracy<br>Pred. Pred. Pred.<br>**----- End of picture text -----**<br>


**Figure 21.** Additional figures for Figure 10. 

50 

