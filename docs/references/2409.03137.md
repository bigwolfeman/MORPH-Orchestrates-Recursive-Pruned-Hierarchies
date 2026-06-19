# The AdEMAMix Optimizer: Better, Faster, Older

- **Authors:** Matteo Pagliardini, Pierre Ablin, David Grangier (EPFL, Apple)
- **Year:** 2024 (ICLR 2025)
- **Source:** https://arxiv.org/abs/2409.03137
- **MORPH uses:** Optional optimizer (`cfg.training.optimizer`: `ademamix` via bitsandbytes, or `ademamix_b1zero` — MORPH's β1=0 fork with fused Triton kernel for AdamW8bit memory parity). AdamW plus a second very-slow momentum EMA (β3) mixed with the fast EMA via α; α/β3 warmup schedulers required for stability under flat LR. 

---

Arxiv preprint 

## THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER 

**Matteo Pagliardini** _[†]_ EPFL 

**Pierre Ablin David Grangier** Apple Apple 

## ABSTRACT 

Momentum based optimizers are central to a wide range of machine learning applications. These typically rely on an Exponential Moving Average (EMA) of gradients, which decays exponentially the present contribution of older gradients. This accounts for gradients being local linear approximations which lose their relevance as the iterate moves along the loss landscape. This work questions the use of a single EMA to accumulate past gradients and empirically demonstrates how this choice can be sub-optimal: a single EMA cannot simultaneously give a high weight to the immediate past, and a non-negligible weight to older gradients. Building on this observation, we propose AdEMAMix, a simple modification of the Adam optimizer with a mixture of two EMAs to better take advantage of past gradients. Our experiments on language modeling and image classification show— quite surprisingly—that gradients can stay relevant for tens of thousands of steps. They help to converge faster, and often to lower minima: e.g., a 1 _._ 3B parameter AdEMAMix LLM trained on 101B tokens performs comparably to an AdamW model trained on 197B tokens (+95%). Moreover, our method significantly slowsdown model forgetting during training. Our work motivates further exploration of different types of functions to leverage past gradients, beyond EMAs. 

## 1 INTRODUCTION 

With large neural networks, deep-learning has revolutionized numerous fields, such as computer vision and natural language processing. At the heart of this paradigm lies the challenge of optimizing complex, non-convex loss functions using noisy gradient estimates. This optimization process is typically carried out using variants of Stochastic Gradient Descent (SGD) (Robbins & Monro, 1951) or adaptive methods such as Adam and AdamW (Kingma & Ba, 2015; Loshchilov & Hutter, 2019), which have become ubiquitous in training state-of-the-art models (Devlin et al., 2019; Brown et al., 2020; Dosovitskiy et al., 2021a; Radford et al., 2021; Zhai et al., 2022; Dehghani et al., 2023; Touvron et al., 2023; Dubey et al., 2024). 

A key component in many of these iterative optimization algorithms is momentum, which has long been shown to accelerate convergence (Nemirovskii & Nesterov, 1985) and often leads to solutions with superior generalization properties (Sutskever et al., 2013). By accumulating gradient vectors over successive optimization steps, momentum helps overcome small local variations of the loss landscape, potentially escaping shallow local minima, and accelerate in plateau regions (Qian, 1999; Ruder, 2016; Goh, 2017). Both SGD with momentum (SGD+M) and Adam incorporate momentum under the form of Exponential Moving Averages (EMAs) of past gradients _G[T]_ = _{_ _**g**_[(0)] _, . . . ,_ _**g**_[(] _[T]_[ )] _}_ : 

**==> picture [369 x 30] intentionally omitted <==**

Two considerations support the use of EMAs. From a practical standpoint, the recursive formula of EMA allows for efficient implementations, which do not require maintaining a buffer of past gradients. From a theoretical standpoint, gradient descent with momentum leads to optimal convergence rates for quadratics (Polyak, 1964; Nesterov, 1983). However, those results do not guarantee any optimality for general non-quadratic cases (Goujaud et al., 2023). 

The use of momentum in optimization is grounded in the varying nature of gradients. As local linear approximations of the loss landscape, their information can quickly become outdated as the 

> _†_ Work done while interning at Apple. 

1 

Arxiv preprint 

**==> picture [397 x 140] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 1<br>3 . 4 AdamW { 17B ,  26B trained ,  33B } on tokens AdamW { 25B ,  39Btrained ,  49B } ontokens 2 . 8 AdamW { 34B ,  101B trained ,  131B on } tokens<br>AdEMAMix { 17B ,  26B ,  33Btrained } tokenson 3 . 0 AdEMAMix { 25B ,  39B ,  49Btrained } tokenson 2 . 7 AdEMAMix { 34B ,  101B ,  131Btrained } tokenson<br>3 . 2 2 . 9 2 . 6<br>2 . 8<br>2 . 5<br>3 . 0 2 . 7<br>2 . 4<br>2 . 6<br>2 . 8 2 . 3 AdamW 197B<br>2 . 5<br>0 256 k 400 k 500 k 0 256 k 400 k 500 k 0 256 k 770 k 1M<br>Iterations Iterations Iterations<br>(a) 110M parameters. (b) 330M parameters. (c) 1 . 3B parameters.<br>Loss Loss Loss<br>**----- End of picture text -----**<br>


Figure 1: **Comparing AdamW and AdEMAMix on language modeling.** In **(a,b,c)** , we plot the loss obtained using AdamW and AdEMAMix (our optimizer) to train Transformer models of various sizes on the Redpajama dataset. In **(a)** , we train multiple baselines for 256 _k_ , 400 _k_ , and 500 _k_ iterations, resulting in processing from 17B to 33B tokens. Two AdamW runs with different number of iterations look very different as we use a cosine-decay for the learning rate. We compare those baselines to training AdEMAMix for 256 _k_ iterations. We observe that our method reaches a similar loss as an AdamW model trained on nearly twice the number of tokens. Analogous comparisons can be derived from **(b)** and **(c)** . Notably, in **(c)** , a 1 _._ 3B parameter AdEMAMix model trained on 101B tokens performs comparably to an AdamW model trained on 197B tokens (95% more, blue horizontal line). See § 4.1 and App. B.1 for a detailed description of our experimental setup, including hyperparameters. 

optimization process progresses (Pascanu et al., 2013). For this reason, practitioners typically employ moderate momentum values (i.e. _β ≈_ 0 _._ 8 or 0 _._ 9), effectively creating a moving average of recent gradients while discarding older information. Selecting larger _β_ values seems counter-intuitive, as it would suggest that older gradients maintain their relevance over extended periods of training. While it is tempting to see the use of small _β_ s as a confirmation of the limited temporal relevance of gradients, our work reveals instead that older gradients can efficiently be used. When we increase _β_ , we decrease the relative importance of recent gradients, and the iterate now fails to respond to local changes in the loss landscape. We observe that a single EMA cannot both give a significant weight to recent gradients, and give a non-negligible weight to older gradients (see Fig. 3a). However, a linear combination between a “fast-changing” (e.g. _β_ = 0 _._ 9 or _β_ 1 = 0) and a “slow-changing” (e.g. _β_ = 0 _._ 9999) EMA allows the iterate to beneficiate from (i) the great speedup provided by the larger (slow-changing) momentum, while (ii) still being reactive to small changes in the loss landscape (fast-changing). More precisely, we find the following statement to convey an important intuition behind this approach: 

_While changing the direction of the slow momentum is difficult, any adjustment orthogonal to that direction is easy—which favors fast progress in sinuous canyon-like landscapes._ 

A toy illustration of this can be seen in Fig. 2. Based on this idea, we propose AdEMAMix ( **Ad** aptive **EMA Mix** ture), a novel Adam based optimizer which successfully leverages very old gradients to reach better solutions. 

**Contributions.** Our contributions can be summarized as follows: 

- We propose AdEMAMix, a novel optimizer which better leverages past gradients by avoiding a common pitfall of EMA-based optimizers (see § 3). 

- We empirically demonstrate the superiority of our method over Adam by training ViT and language models (Transformers and Mamba) of up to 1 _._ 3B parameters (see § 4). In addition, we show gains from switching mid-training from Adam to AdEMAMix (see Fig. 5). 

- We show AdEMAMix forgets the training data slower when compared to Adam (see Fig. 4). 

- More broadly, our findings contribute to a deeper understanding of the optimal balance between using historical gradients and adapting to the rapidly changing loss landscape. Our work invites further research in methods combining old and recent gradients, beyond EMAs. 

2 

Arxiv preprint 

**==> picture [382 x 133] intentionally omitted <==**

**----- Start of picture text -----**<br>
Adam β 1 = 0 . 9 AdEMAMix β 3 = 0 . 999<br>10 [3] Adam β 1 ∈{ 0 β. 29 , = 0 . 099 . 999 ,  0 . 999 ,  0 . 9999 } Adam β 1 = 0 . 99 AdEMAMix β 3 = 0 . 9999<br>10 [2] AdEMAMix α = 9 , β 3 ∈{β 01 . 999 , β 2 ,  0= . 99990 . 9 ,  0 }. 999 start AdamAdam ββ 11 == 00 .. 9999999 start<br>10 [1]<br>10 [0]<br>10 [−] [1]<br>10 [−] [2] YESIma @A<br>10 [−] [3]<br>INI 0 2000 4000 \<br>Iterations<br>(a)  ∥ x [(] [t] [)] − x [⋆] ∥ 2. (b) Adam trajectories. (c) AdEMAMix trajectories.<br>solution<br>to<br>Distance<br>**----- End of picture text -----**<br>


Figure 2: **Comparing Adam and AdEMAMix on the Rosenbrock function.** Starting from _**x**_[(0)] = [ _−_ 3 _,_ 5], we minimize the Rosenbrock function _f_ ( _x_ 1 _, x_ 2) = (1 _− x_ 1)[2] + 100( _x_ 2 _− x_[2] 1[)][2][.][The] global minimum ( _⋆_ ) is _**x**[⋆]_ = [1 _,_ 1]. We use _β_ 2 = 0 _._ 999 for Adam and ( _β_ 1 _, β_ 2 _, α_ ) = (0 _._ 9 _,_ 0 _._ 999 _,_ 9) for AdEMAMix (see § 3). We reduce the learning rate for AdEMAMix to compensate for the influence of _α_ . We do a sweep over _β_ 1 (resp. _β_ 3) for Adam (resp. for AdEMAMix). In **(b)** , When Adam’s _β_ 1 is small (e.g. 0 _._ 9), the iterates do not oscillate but convergence is slow. Increasing _β_ 1 makes the iterates move faster but with large oscillations. In contrast, for AdEMAMix in **(c)** , we observe that despite _β_ 3 being large, the iterates moves fast and without oscillations. This results in reaching better solutions faster as can be seen in **(a)** . 

## 2 RELATED WORK 

**Works on understanding momentum.** From the seminal work of Polyak (1964), many publications analyzed the effect of gradient descent + momentum (GD+M) in both convex and non-convex settings (Ghadimi et al., 2015; Flammarion & Bach, 2015; Kidambi et al., 2018; Defazio, 2020; Sebbouh et al., 2021). While the acceleration in the noise-free setting has been long theorized for convex functions, several publications indicate this effect might not necessarily extend to stochastic settings (Yuan et al., 2016; Kidambi et al., 2018; Leclerc & Madry, 2020), emphasizing instead a link between momentum and effective learning rate. Recent work have been seeking to understand the impact of momentum on generalization through studying the implicit bias of momentum methods (Ghosh et al., 2023; Papazov et al., 2024), exposing a preference of SGD+M for lower norm solutions. Those further exposed a link between higher momentum and higher effective learning rate and higher variance reduction. Despite the volume of prior work on the subject, our understanding of momentum methods in non-convex stochastic settings is still incomplete (Yuan et al., 2016). Oscillatory behaviours, and the sometimes ambiguous effect of variance on optimization render the analysis tedious. From a theoretical standpoint, our work raises several questions. First, given that we gain from averaging very old gradients, what can it reveal of the loss landscape and the consistency of one batch’s gradient during training? Second, would our approach not decrease the variance up to a point that is harming generalization (Ghosh et al., 2023)? While no answer to those questions is given in this work, we provide a toy justification which indicates that large momentums can have a positive impact in noise-free non-convex settings (see Fig. 2)—indicating the improvement of our approach is at least partially explainable without considering variance-reduction effects. We moreover expose a link between momentum and forgetting the training data (see Fig. 4), which to our knowledge is novel. 

**Works on deep-learning optimizers.** Despite the popularity of Adam and AdamW (Kingma & Ba, 2015; Loshchilov & Hutter, 2019) in training deep neural networks, optimizer design is a rich field of research and we focus on a few of the works most relevant to this study. Adafactor (Shazeer & Stern, 2018) improves Adam’s memory efficiency by factorizing the second moment estimate. Lamb (You et al., 2020) extends Adam by adding layerwise normalized updates. Chen et al. (2023a) use algorithm discovery to derive the Lion optimizer. Contrary to Adam, Lion uses a single momentum term and the sign function to produce updates with the same magnitude across dimensions. Interestingly, Chen et al. (2023a) also report better scores are obtained when using a slightly larger momentum term ( _β_ = 0 _._ 99). In this work we show how increasing the momentum well beyond this value can still be beneficial. See App. C.3.2 for a detailed comparison between AdEMAMix and Lion. Recently, 

3 

Arxiv preprint 

Liu et al. (2023) introduced Sophia, a scalable second-order optimizer designed for LLM training. Sophia uses a Hessian-based pre-conditioner which better normalizes the step size, penalizing steps in high curvature direction and accelerating in low curvature directions. Understanding in which circumstances those novel optimizers bring improvements is still being investigated (Kaddour et al., 2023), and Adam’s dominance remains mostly unchallenged. 

**Work incorporating an additional momentum term.** Lee et al. (2024) introduce Grokfast, which uses a pre-filtering step on the gradient to amplify the low frequencies and help solve groking. When combined with Adam, it effectively applies the Adam’s EMAs on top of another gradient averaging method (e.g. EMA for Grokfast-EMA). Somewhat similarly, Chen et al. (2023b) refer to the Double EMA (DEMA) used in some finance applications (Mulloy, 1994) as one motivation for their AdMeta optimizer. Our motivation behind AdEMAMix is to combine both a high sensitivity to the recent gradients as well as incorporating very distant gradient, in this respect, using nested EMAs is not the right candidate as it reduces the influence of recent gradients. A more detailed review of AdMeta and DEMA is in App. C.3.1. Most relevant to us, Lucas et al. (2019, AggMo) also observe that using a combination of EMAs can enable the use of larger _β_ s, and incorporates a sum of _K_ momentum terms into _GD_ They show their approach reaches similar performances as baseline optimizers, with a faster convergence. In contrast, we modify _Adam_ , and introduce schedulers that are critical to reaching good performances at larger scales. This allows us to outperform Adam by a significant margin. In App. C.3.3, we notice no further improvement by adding more momentum terms. Finally, Szegedy et al. (2024) propose a general framework to derive and study optimizers with linear combinations of memory vectors—which encompasses EMA mixtures. 

**Works on distributed optimization.** Perhaps surprisingly, recent work on distributed optimization— DiLoCo and SlowMo (Douillard et al., 2023; Wang et al., 2020)—are relevant to our work. _N_ workers _**θ**_ 1[(] _[t]_[)] _[, . . . ,]_ _**[ θ]** N_[(] _[t]_[)][are trained independently for] _[ K]_[steps (e.g.] _[K]_[=][500][).][Every] _[ K]_[steps, the] delta updates _{_ ∆ _**θ** i}[N] i_ =1[≜] _[{]_ _**[θ]** i_[(] _[t]_[+] _[K]_[)] _−_ _**θ** i_[(] _[t]_[)] _[}] i[N]_ =1[are averaged and applied to each worker using an] outer optimizer _with momentum β_ : _**θ** i_[(] _[t]_[+] _[K]_[)] = _**θ** i_[(] _[t]_[)] _− η ·_ Opt( _N_[1] � _i_[∆] _**[θ]**[i][, β]_[)][.][The][application][of] the outer momentum every 500 steps increases the importance of old gradients in the optimization trajectory. We believe this observation might in parts explain the strong results provided by those methods, and further motivated our work. 

## 3 OUR METHOD: ADEMAMIX 

**Setup & notations.** Let _L_ _**θ**_ : _X �→_ R be a loss function parameterized by _**θ**_ , and mapping inputs _**x** ∈X_ to R. Given a sampled batch _**x**_ , let _**g**_ = _∇_ _**θ** L_ _**θ**_ ( _**x**_ ) be a stochastic gradient of the loss w.r.t. _**θ**_ . To minimize the empirical loss, the Adam optimizer (Kingma & Ba, 2015) relies on first and second moments, resp. _**m**_ and _**ν**_ , estimated via two EMAs parametrized by ( _β_ 1 _, β_ 2) _∈_ [0 _,_ 1[[2] . A weight-decay parameter _λ ∈_ R[+] is often used as in Loshchilov & Hutter (2019): 

**==> picture [317 x 51] intentionally omitted <==**

With _t >_ 0 being the timestep, _η_ being the learning rate, and _ϵ_ a small constant. Initially _**m**_[(] _[t]_[=0)] = _**ν**_[(] _[t]_[=0)] = **0** . To prevent the bias induced by the initial _**m**_[(] _[t]_[=0)] and _**ν**_[(] _[t]_[=0)] , the outputs of the two EMAs are corrected into _**m**_ ˆ[(] _[t]_[)] and _**ν**_ ˆ[(] _[t]_[)] . Those are used to compute the final weight update, scaled by the learning rate. 

**How far to look into the past?** A typical value for _β_ 1 is 0 _._ 9. Fig. 3a shows the weights given to past gradients for different values of _β_ . The larger the _β_ , the more uniform the average is. To put this in perspective—observing that[�] _[∞] i_ =0 _[β][i]_[(1] _[ −][β]_[) = 1][ for] _[ β][∈]_[[0] _[,]_[ 1[][—the number of successive previous] steps receiving a cumulative weight of 0 _._ 5, is _thalf_ =[ln] ln([(][0] _β[.]_[5] )[)] _[−]_[1][.][For] _[ β]_[= 0] _[.]_[9][,] _[ t][half][≈]_[6][, meaning] that half of the weight is given to the previous six gradients. This observation can also be extended to SGD with e.g. polyak or nesterov momentums (Polyak, 1964; Nesterov, 1983), which typically relies on similar _β_ values. The value of _β_ 1 is rarely increased beyond _∼_ 0 _._ 9. In our experiments with AdamW, increasing _β_ 1 further degraded the performance (see App. C.1.6). Does this mean 

4 

Arxiv preprint 

**==> picture [397 x 151] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 . 0 [×] [10] [−] [3] 2 . 7<br>EMA1: β = 0 . 9 AdamW AdamW trained on<br>0 . 8 EMA EMA EMA234: : : β β β = = = 0 0 0 . . . 99 999 9999 2 . 6 AdEMAMixAdEMAMix100 k steps of ββ linear33 == 00 .. 999999999 η decay 3 . 6 { AdEMAMix { 8B8B ,,  16B 16B }} tokenstokenstrained on<br>3 . 5<br>0 . 6<br>2 . 5<br>3 . 4<br>0 . 4<br>2 . 4 3 . 3<br>0 . 2<br>3 . 2<br>2 . 3<br>0 . 0 3 . 1<br>0 5000 10000 0 500 k =  Tα,β 3 1M 1 . 3M 0 64 k 128 k<br>t Iterations Iterations<br>(a) Limitation of EMAs. (b) Constant  η -scheduler. (c) Mamba results (168M).<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 3: **Limitation of EMAs, constant** _η_ **-scheduler, & Mamba results.** In **(a)** , we plot the weights _wt_ —for each past gradient _**g**_[(] _[t]_[)] —given by different EMAs after 10 _k_ steps. For a given _β_ , EMA( _β,_ _**g**_[(0)] _, . . . ,_ _**g**_[(] _[T]_[ )] ) =[�] _[T] i_ =0 _[β][i]_[(1] _[ −][β]_[)] _**[g]**_[(] _[T][ −][i]_[)][, which decays the contribution of past gradients] exponentially. A small _β_ (e.g. 0 _._ 9) will give a high weight to the immediate past and negligible weights to older timesteps. In contrast, a high _β_ (e.g. 0 _._ 9999) is giving a relatively uniform, yet non-negligible weight to all gradients. No _β_ value can simultaneously give a high weight to the immediate past and a non-negligible weight to very old timesteps. In **(b)** , we train multiple 1 _._ 3B language models using 3 _k_ steps of warmup and then a constant learning rate _η_ = 10 _[−]_[4] . This allows us to observe the gap between AdamW and AdEMAMix without the cosine decay as a confounder. We still use schedulers for _α_ and _β_ 3 with _Tα,β_ 3 = 500 _k_ , _α_ = 5. Similar to Zhai et al. (2022); Hu et al. (2024); Hägele et al. (2024), we decay the learning rate linearly at _t_ = 1M and _t_ = 1 _._ 3M. The loss-gap between AdamW and AdEMAMix increases at first, and then remains constant. AdEMAMix still outperforms AdamW after decaying the learning rate. See App.C.1.9 to see the impact of the linear decay duration. In **(c)** , we train 168M parameter Mamba models, showing how AdEMAMix’s performances can generalize outside of the Transformer architecture. 

older gradients are outdated? We show that this is not the case, rather, increasing beta is reducing the sensitivity to recent gradients too much. We design AdEMAMix such that the sensitivity to recent gradients is kept, while also incorporating information from much older gradients using an additional momentum term. This allows for the use of much larger _β_ values e.g. 0 _._ 9999. To compare, for _β_ = 0 _._ 9999, _thalf ≈_ 6 _,_ 930, spreading half of the mass over the previous 6 _,_ 930 past gradients. 

**AdEMAMix.** To keep a high sensitivity to recent gradients, while also incorporating information from older gradients, we add a second EMA (changes compared to AdamW are in Blue): 

**==> picture [345 x 68] intentionally omitted <==**

In our experiments, while the values of _β_ 1 _, β_ 2 remain similar to those of equation AdamW, we often use _β_ 3 = 0 _._ 9999. We find _α ∈_ [4 _,_ 10] to work well in practice. 

**Tackling early training instabilities.** Early training instabilities are commonplace when training deep models, and empirically motivated the introduction of methods such as learning rate warmup and gradient clipping. Gilmer et al. (2022) show how the use of learning rate warmup can be justified from a curvature perspective; allowing the iterates to move to parts of the optimization landscape where larger learning rates are stable. While we use learning rate warmup in all our experiments, we still noticed AdEMAMix models using a large _β_ 3 would diverge early. This, despite not using bias correction over _**m**_ 2, which lets the momentum buffer fill itself slowly. Those failed runs are characterized by updates of large magnitudes in the early phase of training (see App. C.1.9, Fig. 27). For this reason, we progressively increase the values of _β_ 3 and _α_ using schedulers. For _α_ we use a linear scheduler. A linear scheduler for _β_ 3 would be ill-fitted as the same increment of _β_ 3 has a 

5 

Arxiv preprint 

different impact for different values of _β_ 3. For instance, observe that an increase of _β_ of _δβ_ = 0 _._ 0001 barely increases the _thalf_ for _β_ = 0 _._ 9, while 0 _._ 999 _→_ 0 _._ 999 + _δβ_ increases the _thalf_ of 77. For this reason, we design the _β_ 3 scheduler to _increase thalf linearly_ (see App. A.1 for a derivation of that scheduler). The two schedulers are summarized below: 

**==> picture [380 x 53] intentionally omitted <==**

With _Tα_ and _Tβ_ 3 are resp. the warmup times for _α_[(] _[t]_[)] and _β_ 3[(] _[t]_[)] to reach their final and maximal values. In practice we always set those two to the same value: _Tα_ = _Tβ_ 3 = _Tα,β_ 3, and we typically use _Tα,β_ 3 = _T_ , with _T_ being the total number of iterations. _βstart_ is always set to _β_ 1 in our experiments. The use of those schedulers is not always required, especially, we found those have no impact when AdEMAMix is activated later during training (see Fig. 5). The full AdEMAMix optimizer, including the schedulers, is shown in Alg. 1. 

## **Algorithm 1** AdEMAMix optimizer. Differences with AdamW are in blue. 

1: **Input:** Data distribution _D_ . Initial model parameters _**θ**_[(0)] . Number of iterations _T_ . Learning rate _η_ . _ϵ_ a small constant. AdamW parameters: _β_ 1, _β_ 2 and _λ_ . AdEMAMix parameters _β_ 3, _α_ . Warmup parameter _Tα,β_ 3, note that we usually set it to _T_ . _βstart_ is usually set to _β_ 1. 

2: Initialize timestep: _t ←_ 0 

3: Initialize EMAs: _**m**_[(0)] 1 _←_ **0** , _**m**_[(0)] 2 _←_ **0** , _**ν**_[(0)] _←_ **0** 4: **for** _t ∈_ [ _T_ ] **do** 5: _t ← t_ + 1 6: Optional: use schedulers _η_[(] _[t]_[)] , _β_ 3[(] _[t]_[)] _← fβ_ 3( _t, β_ 3 _, βstart, Tα,β_ 3) and _α_[(] _[t]_[)] _← fα_ ( _t, α, Tα,β_ 3) 7: Sample batch: _x ∼D_ 8: Compute gradient: _**g**_[(] _[t]_[)] _←∇_ _**θ** L_ _**θ**_ ( _t−_ 1)( _**x**_ ) 9: Update the fast EMA _**m**_ 1: _**m**_[(] 1 _[t]_[)] _← β_ 1 _**m**_[(] 1 _[t][−]_[1)] + (1 _− β_ 1) _**g**_[(] _[t]_[)] 10: Update the slow EMA _**m**_ 2: _**m**_[(] 2 _[t]_[)] _← β_ 3[(] _[t]_[)] _**[m]**_[(] 2 _[t][−]_[1)] + (1 _− β_ 3[(] _[t]_[)][)] _**[g]**_[(] _[t]_[)] 11: Update the second moment estimate: _**ν**_[(] _[t]_[)] _← β_ 2 _**ν**_[(] _[t][−]_[1)] + (1 _− β_ 2) _**g**_[(] _[t]_[)2] 12: Apply bias corrections: _**m**_ ˆ[(] 1 _[t]_[)] _←_ 1 _**[m]** −_ 1[(] _β[t]_[)] 1 _[t]_[,] _**[ν]**_[ˆ] 1[(] _[t]_[)] _←_ 1 _**ν** −_ 1[(] _[t] β_[)] 2 _[t]_ 13: Update parameters: _**θ**_[(] _[t]_[)] _←_ _**θ**_[(] _[t][−]_[1)] _− η_[(] _[t]_[)][�] _**m**_[ˆ][(] 1 _[t]_[)][+] ˆ _[α]_[(] _[t]_[)] _**[m]**_[(] 2 _[t]_[)] + _λ_ _**θ**_[(] _[t][−]_[1)][�] ~~_√_~~ _**ν**_[(] _[t]_[)] + _ϵ_ 14: **end for** 15: **Return** optimized parameters _**θ**_[(] _[T]_[ )] 

**Using** _β_ 1 = 0 **to save memory.** By setting _β_ 1 = 0, we can save on memory by replacing _**m**_ ˆ[(] 1 _[t]_[)] by _**g**_[(] _[t]_[)] . In this case, the memory cost of AdEMAMix is the same as for AdamW. We show in App. C.1.4 (Fig. 15b) and App. C.1.7 that using _β_ 1 = 0 often works, at the cost of potentially less stable training. 

**Computational overheads.** Adding an additional EMA requires additional memory and compute. The added compute is negligible in comparison to what is required for the forward-backward, and has little impact over the total runtime as shown in Fig. 5a. Moreover—when considering larger distributed setups—it is worth noting that AdEMAMix is not increasing communication (gradient reduction) over Adam. Therefore, we expect the runtime overhead of AdEMAMix to shrink in those settings, as data movements occupy a larger fraction of the total runtime. A more significant overhead is in terms of memory when _β_ 1 = 0, as AdEMAMix requires to allocate both _**m**_ 1 and _**m**_ 2, which are of the same size as the model parameters _**θ**_ . We believe this issue is of lesser consequences as Fully-Sharded-Data-Paralellism (Zhao et al., 2023, FSDP) can always be used to distribute the optimizer states across compute nodes. In the cases where memory remains an issue, one mitigation strategy could be to apply factorization strategies as in Shazeer & Stern (2018); Zhao et al. (2024). 

**Hyperparameter sensitivity.** While we introduce up to four new hyperparameters: _α_ , _β_ 3, _Tα_ , and _Tβ_ 3, in practice we always set _Tα_ = _Tβ_ 3 = _Tα,β_ 3, and use _Tα,β_ 3 = _T_ in most cases. We show in App. C.1.4 that _Tα_ and _Tβ_ 3 should only be large enough to prevent instabilities early during training. 

6 

Arxiv preprint 

**==> picture [380 x 185] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 2 AdamWwithout B training<br>AdamW training<br>3 . 1 using B at t = tB<br>tB<br>3 . 0<br>2 . 9<br>2 . 8 NENG<br>90 k 256 k 70k 170 k 256 k 70k 190 k 256 k 70k 230 k<br>3 . 2 AdEMAMixwithout B training<br>AdEMAMix training<br>3 . 1 using B at t = tB<br>tB<br>3 . 0<br>2 . 9<br>2 . 8 SNE<br>90 k 256 k 70k 170 k 256 k 70k 190 k 256 k 70k 230 k<br>(a)  tB = 90 k . (b)  tB = 170 k . (c)  tB = 190 k . (d)  tB = 230 k .<br>B<br>batch<br>on<br>Loss<br>B<br>batch<br>on<br>Loss<br>**----- End of picture text -----**<br>


Figure 4: **Measuring forgetting using a held-out batch** _B_ **.** The top row is for AdamW, the bottom row is for AdEMAMix. We trained one AdamW and AdEMAMix model on a RedPajama dataset _not_ containing the batch _B_ , those runs are in blue. We then run multiple experiments where we inject _B_ in the training data at a specific timestep _tB_ . Those runs are in orange. To inspect how much influence _B_ had when it is injected at _tB_ , we can observe the evolution of the gap between the blue and the orange curves. For both optimizers, we observe a rapid decrease of the loss on _B_ right after training on _B_ . The sharpness of this decrease in loss is more pronounced for AdamW compared to AdEMAMix. However, when using AdamW, the loss on _B_ then increases faster, which we interpret as the model forgetting _B_ faster. In contrast, the curves for AdEMAMix are smoother, the loss on _B_ goes back up slower, and ultimately _B_ had a bigger impact on the training when using AdEMAMix—as can be seen by looking at the larger gap between the orange and blue curves for the last iteration. Finally, the forgetting behaviour evolve during training, with the later training batches being remembered better. See App. C.1.2 for a more detailed analysis of forgetting as training progresses. 

While all of our experiments on language modeling use _β_ 3 = 0 _._ 9999, other values such as 0 _._ 999 or even 0 _._ 99999 still can outperform the AdamW baseline (see App. C.1.4, Fig. 12). On vision tasks, the scatter plots in Fig. 6 show all the AdEMAMix models trained for those experiments, highlighting how easy it can be to find good _α_ and _β_ 3 values. Overall, we find the ranges of values of _α, β_ 3 and _Tα,β_ 3 providing improvements over AdamW to be wide. See App. C.1.4 for hyperparameter sweeps. 

**Limitations.** AdEMAMix consists in leveraging very old gradients. Therefore, the method is best suited to settings where the number of iterations is important. We report on this effect in App. C.1.5, additionally showing how smaller values of _β_ 3 (e.g. _β_ 3 = 0 _._ 999) can be better for low iterations scenarios. Moreover, retaining gradient information over many thousands steps can pose a problem in domains requiring fast adaptation to a sudden distribution shift. 

## 4 RESULTS 

In this section we use AdEMAMix to train language models (§ 4.1 and § 4.2) and vision transformers (§ 4.3) ranging from 24M to 1 _._ 3B parameters. 

## 4.1 TRANSFORMER LLM TRAINING 

**Experimental setup.** We use a transformer architecture (Vaswani et al., 2017) with learnt positional encoding. All of our experiments are using sequences of 1 _,_ 024 tokens. We experiment with three sizes of transformers: 110M, 335M, and 1 _._ 3B. For the learning rate, we use 3k warmup steps followed by—unless specified—a cosine decay to 10 _[−]_[5] . We extensively tuned the hyperparameters for both AdamW and AdEMAMix models (see App. B.1). We use the RedPajama v2 (Computer, 2023) 

7 

Arxiv preprint 

**==> picture [396 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
60 AdamW AdEMAMix 3 . 1 AdamWAdEMAMixAdEMAMix33Bfromfromtokens300330 kk 22 .. 5550 AdamWAdEMAMixAdEMAMix101Bfromfromtokens500600 kk<br>AdEMAMix from 360 k AdEMAMix from 0<br>50 AdEMAMix from 390 k 2 . 45<br>AdEMAMix from 0<br>3 . 0<br>2 . 40<br>40<br>2 . 9 2 . 35<br>30<br>2 . 8 2 . 30 AdamW 1M iterations<br>20 2 . 25<br>110M 1 . 3B 300 k 390 k 500 k 500 k 600 k 770 k<br>Iterations Iterations<br>(a) Train time for 256 k  iters. (b) From AdamW (110M). (c) From AdamW (1 . 3B).<br>(hours)<br>time Loss Loss<br>Training<br>**----- End of picture text -----**<br>


Figure 5: **Training time comparison & starting AdEMAMix from AdamW.** In **(a)** , we compare the time required to train 110M and 1 _._ 3B parameter models for 256 _k_ iterations. The additional EMA renders AdEMAMix slightly slower than AdamW. However, if we were to train AdamW longer to compensate for this gap, we would only train for an additional 2379 and 5421 iterations for resp. 110M and 1 _._ 3B parameter models. Those additional iterations would not be sufficient to close the gap (see Fig. 1). In **(b)** and **(c)** , we show—for two different model sizes—the effect of switching from AdamW to AdEMAMix during training. AdEMAMix’s additional parameter _**m**_ 2 is initialized to **0** , no scheduler is used for _α_ or _β_ 3. For both model sizes, we observe the loss increases slightly at first before decreasing and outperforming the baseline. In both cases, the earlier AdEMAMix is used, the better the final loss. See App. C.1.9 for results using schedulers. 

dataset for all of our experiments. We use batch sizes of 64, 96 and 128 for respectively our 110M, 335M, and 1 _._ 3B parameter models. Depending on the model, we vary the number of iterations from 256 _k_ to 1 _._ 5M. For AdEMAMix, we use _β_ 3 = 0 _._ 9999 and _α ∈{_ 5 _,_ 8 _,_ 10 _}_ depending on the model. A full description of the architecture and hyperparameters used is in App. B.1. We train on up to 8 A100 NVIDIA GPUs using data-parallelism. 

**Why not simply increasing AdamW’s** _β_ 1 **?** While our toy experiment in Fig. 2 already gives some intuition on why increasing Adam’s _β_ 1 is likely not to have the same effect as having an additional EMA as in AdEMAMix, we verify this intuition by training multiple 110M models using Adam with large _β_ 1 _∈{_ 0 _._ 99 _,_ 0 _._ 999 _,_ 0 _._ 9999 _,_ 0 _._ 99999 _}_ . When we use a large _β_ 1 from the beginning of training, we observe instabilities for larger _β_ 1 values and no _β_ 1 _>_ 0 _._ 9 improves upon the AdamW baseline. One can imagine this to be due to increasing _β_ 1 too early. Therefore, we also modify AdamW and add the same scheduler on _β_ 1 as we use on AdEMAMix’s _β_ 3. _β_ 1 is now increased steadily over the entire training duration. While this mostly stabilizes the training, none of the experiments outperformed the baseline using _β_ 1 = 0 _._ 9. Moreover, to rule out any effect that could be due to early training instabilities, we run the same experiments starting from a pre-trained AdamW checkpoint trained for 300 _k_ iterations. We simply resume training and either increase _β_ 1 suddenly or using a scheduler. Here again—unlike when using AdEMAMix—none of those experiments outperform the baseline. The details of those experiments are available in App. C.1.6. Those experiments show that simply increasing the _β_ 1 value in AdamW is not enough, which justifies our design of AdEMAMix. 

**Better perplexity for the same number of steps.** For all model sizes, all the number of iterations used—ranging from 256k to 1M—, AdEMAMix always outperforms the AdamW baseline. In Fig. 1, we show the validation loss curves for AdamW and AdEMAMix models trained—for each size—on various numbers of tokens. For 110M parameter models, training for 256 _k_ iterations gives similar results as training an AdamW model for 500 _k_ iterations. The gap between baseline and our method seems to be increasing as we increase the number of iterations. For 1 _._ 3B parameter models, training using 770 _k_ steps is on par with training the baseline for 1 _._ 5M iterations—reaching the same performance with 51% fewer tokens (an economy of 96M tokens). In Fig. 3b, we observe similar improvements when using a constant and then linear decay learning rate scheduler. 

**Training speed comparison.** We measure the time per iteration for all of our experiments. In Fig. 5a we plot the time required to train our 110M and 1 _._ 3B parameter models for 256 _k_ iterations. We observe that the impact of using an additional EMA on the training speed is negligible. If we were to 

8 

Arxiv preprint 

train new models with a fix time budget, the extra iterations of the baseline would not be sufficient to close the gap. For instance, to match a 110M parameter AdEMAMix model trained for 256 _k_ iterations, we need to train an AdamW model for 500 _k_ iterations, and the time advantage of AdamW would only allow us to do 2 _,_ 379 additional iterations. Moreover, as mentioned in § 3, we expect the time overhead to decrease when multi-node training is used, as IOs would become an important bottleneck, and AdEMAMix is not increasing IOs. 

**Consistency of the gain across token-budgets.** In Fig. 1, given that enough tokens have been used w.r.t. the model size, we observe consistent gains accross token budgets. It seems our method is able to bring a constant improvement over the baseline. This can be seen more clearly in Fig. 3b, showing results when using a constant learning rate scheduler—which removes the confounder of the cosine learning rate decay. We observe how, after an initial phase in which the gap grows, this gap becomes seemingly constant after a sufficient number of iterations. 

**Resume from AdamW vs. training AdEMAMix from scratch.** So far we trained AdEMAMix models from scratch. We show it is also possible to switch from an AdamW to an AdEMAMix state in the middle of training. When switching to AdEMAMix at step _tswitch_ , we initialize _**m**_[(] 2 _[t][switch]_[)] = **0** and replace _t_ by _t − tswitch_ in the scheduler equations—if those are used. However, we find that schedulers are not required when resuming training and report results without them in the main paper (see App. C.1.9 for more details). In Fig 5b and Fig.5c, we show that (i) it is possible to improve upon the baseline when activating AdEMAMix later during training, and (ii) the earlier the switch, the larger the gain, with diminishing returns. This indicates that the improvement of AdEMAMix cannot be attributed solely to early training dynamics, but rather, late training dynamics seem to play an important role. This is further corroborated by the reverse experiment—which switches from AdEMAMix to AdamW mid-training and show a performance degradation (see results in App. C.1.3). 

**AdEMAMix models forget the training data slower.** As an attempt to understand the reason for AdEMAMix’s improvements over AdamW, we study how fast a training batch is forgotten after being used during training. We focus on following one specific batch _B_ . We start by removing _B_ from the RedPajama training data and train AdamW and AdEMAMix models. For those runs, _B_ is therefore akin to a batch from the validation set. We measure the loss on _B_ while training. Now we can resume training from various checkpoints, and inject _B_ into the training data at various times _tB_ . By comparing the two runs—one having trained on _B_ , while the other never saw _B_ —we can visualize how _B_ is learned, and how it is forgotten. After training on _B_ , we expect the loss on _B_ to decrease suddenly and then increase as the model forgets the contribution of that batch. When comparing the forgetting curves for AdamW and AdEMAMix in Fig. 4, we see striking differences. AdamW models forget much faster—the loss over _B_ increases faster—than AdEMAMix models. Moreover, at the end of training, batches processed by AdEMAMix see their loss being improved over many thousands of iterations. Additional experiments on forgetting can be found in App. C.1.2. 

## 4.2 MAMBA LM TRAINING 

**Experimental setup & results.** Our experimental setup is similar to § 4.1, except that we now train 168M parameter Mamba models (Gu & Dao, 2023) using the FineWeb dataset (Penedo et al., 2024). See App. B.2 for more details. In Fig. 3c the improvement using AdEMAMix is consistent with our experiments on Transformer models. This shows how AdEMAMix’s gains can extend beyond Transformer models. 

## 4.3 VIT TRAINING 

**Experimental setup.** In this section we consider a different setting in which the data is now a limited resource, and we do multiple epochs (e.g. 37 or 320). We use two subsets of the ImageNet (Russakovsky et al., 2015) dataset: (i) the widely used ImageNet-1k subset, consisting of 1 _._ 3M images and 1 _,_ 000 possible classes; (ii) a filtered and pre-processed version of the ImageNet-21k (Ridnik et al., 2021) containing 11M images corresponding to 10 _,_ 450 classes. For each, we measure the test loss on a held-out test set. We use the ViT architecture (Dosovitskiy et al., 2021b) at two different scales: 24M and 86M parameters. Importantly, if it is common in the vision literature to pre-train large ViT models and finetune them on smaller datasets, this work focuses on pretraining optimization and we therefore train and test on the same distribution. The models’ hyperparameters are detailed in App. B.3, we use a batch size of 4096 for all our experiments. We used training hyperparameters 

9 

Arxiv preprint 

**==> picture [396 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 . 60 2 . 45 1 . 3<br>AdamW AdamW AdamW<br>AdEMAMix β 3 = 0 . 999 AdEMAMix β 3 = 0 . 999 AdEMAMix β 3 = 0 . 999<br>AdEMAMix β 3 = 0 . 9999 2 . 40 AdEMAMix β 3 = 0 . 9999 AdEMAMix β 3 = 0 . 9999<br>2 . 55 1 . 2<br>2 . 35<br>2 . 50 2 . 30 1 . 1<br>2 . 25<br>2 . 45 1 . 0<br>2 . 20<br>2 . 40 0 . 9<br>2 . 1 2 . 2 2 . 3 1 . 6 1 . 8 2 . 0 0 . 1 0 . 2 0 . 3<br>Training loss Training loss Training loss<br>(a) 24M params, 11M images. (b) 86M params, 11M images. (c) 86M params, 1 . 3M images.<br>(ImageNet-21k) (ImageNet-21k) (ImageNet-1k)<br>loss loss loss<br>Test Test Test<br>**----- End of picture text -----**<br>


Figure 6: **ViT results for different capacity/data ratio.** As described in § 4.3, we compare the train vs. test loss between AdEMAMix and AdamW, on multiple scenarios **(a,b,c)** . Those have different capacity/data ratios. For each setting, we first tune an AdamW baseline by testing multiple hyperparameters, those are represented by blue dots. We then pick the hyperparameters of the best baseline—corresponding to the lowest test loss—and train multiple AdEMAMix models by simply testing various _α ∈{_ 1 _,_ 5 _,_ 10 _,_ 15 _,_ 20 _}_ and _β_ 3 _∈{_ 0 _._ 999 _,_ 0 _._ 9999 _}_ . Those are represented by orange squares and red triangles. See App. B.3 for detailed hyperparameters. The red area in the above plots represents the area in which we achieve both a better training and test loss compared to the best baseline. The diagonal black line represents the line we would follow when we improve upon the baseline without adding any overfitting (i.e. an improvement in train loss yields the same improvement in test loss). **(a)** represents the most desirable setting, in which we have relatively a lot of data compared to the model size. In this setting, AdEMAMix outperforms the baseline for all hyperparameters tested. If we increase the model size as in **(b)** , most AdEMAMix hyperparameters are improving upon the baseline. If we now decrease the dataset size, using a 86M model with 1 _._ 3M images as in **(c)** , doing a total of 320 epochs, we do not observe an improvement from using AdEMAMix. See App. C.2 to see classification accuracies and loss curves for those experiments. 

from Dosovitskiy et al. (2021b) as a starting point and did some additional tuning of the learning-rate, dropout, and weight decay for our AdamW baselines. We then use the hyperparameters of the best AdamW baseline and train AdEMAMix models with various values of _α ∈{_ 1 _,_ 5 _,_ 10 _,_ 15 _,_ 20 _}_ and _β_ 3 _∈{_ 0 _._ 999 _,_ 0 _._ 9999 _}_ . We train models for 320 and 37 epochs for resp. the ImageNet-1k and ImageNet-21k datasets, corresponding in both cases to 100 _k_ iterations. Data-augmentation techniques have been shown to be central to the efficient training of ViTs Touvron et al. (2021; 2022). We use simple data-augmentations, which includes mixup (Zhang et al., 2018). We train on 8 A100 NVIDIA GPUs using Pytorch Fully Sharded Data-Parallelism (Zhao et al., 2023, FSDP). 

**AdEMAMix for different capacity/data ratios.** We consider three scenarios that differ by their capacity/data ratios. First, we trained 24M parameter models on 11M images (ImageNet-21k), for 37 epochs. In this setting, as can be seen in Fig. 6a, it is trivial to find AdEMAMix parameters outperforming the baseline in terms of both training and test accuracy. We now increase the model size to 86M parameters. Fig. 6b shows it is still easy to find parameters outperforming the baseline. We now keep the model size of 86M parameters and switch to the smaller ImageNet-1k dataset (1 _._ 3M images), which—given our 100 _k_ iterations—increases the number of epochs to 320. In this setting, Fig. 6c shows that outperforming the baseline becomes difficult. These experiments show how AdEMAMix seems to perform best in scenarios with large volumes of data w.r.t. the capacity of the model. Overall, we found AdEMAMix to consistently reduce the training loss more efficiently than AdamW. When this decrease in training loss correlates with a decrease in test loss, AdEMAMix outperforms the AdamW baseline. 

## 5 CONCLUSION 

In this work, we find that old gradients can be leveraged to efficiently train large language models and ViTs. Our proposed optimizer combines two momentum terms. A slow (large _β_ ) momentum 

10 

Arxiv preprint 

gathers information over many timestep, while a fast (low _β_ ) momentum can adapt the trajectory of the iterates to the rapidly changing loss landscape. We demonstrate the superiority of our optimizer over AdamW through a set of experiments on text modeling and image classification. We moreover reveal how our optimizer forgets the training data at a slower pace. 

## 6 ACKNOWLEDGEMENTS 

We thank Alaaeldin Mohamed Elnouby Ali for guiding us throughout the training of our ViT models, as well as Federico Danieli and Abhinav Moudgil for their help training Mamba models. We also thank Angelos Katharopoulos and Ronan Collobert for their insightful comments and feedback. 

## REFERENCES 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In _NAACL-HLT (1)_ , pp. 2357–2367. Association for Computational Linguistics, 2019. 

- Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, and Peter Clark. Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge. _CoRR_ , abs/2102.03315, 2021. 

- Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In _AAAI_ , pp. 7432–7439. AAAI Press, 2020. 

- Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. Gpt-neox-20b: An open-source autoregressive language model. _CoRR_ , abs/2204.06745, 2022. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , 2020. 

- Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic discovery of optimization algorithms. In _NeurIPS_ , 2023a. 

- Yineng Chen, Zuchao Li, Lefei Zhang, Bo Du, and Hai Zhao. Bidirectional looking with A novel double exponential moving average to adaptive and non-adaptive momentum optimizers. In _ICML_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 4764–4803. PMLR, 2023b. 

- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In _NAACL-HLT (1)_ , pp. 2924–2936. Association for Computational Linguistics, 2019. 

- Together Computer. Redpajama: an open dataset for training large language models, October 2023. URL https://github.com/togethercomputer/RedPajama-Data. 

- DeepMind, Igor Babuschkin, Kate Baumli, Alison Bell, Surya Bhupatiraju, Jake Bruce, Peter Buchlovsky, David Budden, Trevor Cai, Aidan Clark, Ivo Danihelka, Antoine Dedieu, Claudio Fantacci, Jonathan Godwin, Chris Jones, Ross Hemsley, Tom Hennigan, Matteo Hessel, Shaobo Hou, Steven Kapturowski, Thomas Keck, Iurii Kemaev, Michael King, Markus Kunesch, Lena Martens, Hamza Merzic, Vladimir Mikulik, Tamara Norman, George Papamakarios, John Quan, Roman Ring, Francisco Ruiz, Alvaro Sanchez, Laurent Sartran, Rosalia Schneider, Eren Sezener, Stephen Spencer, Srivatsan Srinivasan, Miloš Stanojevi´c, Wojciech Stokowiec, Luyu Wang, Guangyao Zhou, and Fabio Viola. The DeepMind JAX Ecosystem, 2020. URL http://github.com/google-deepmind. 

11 

Arxiv preprint 

- Aaron Defazio. Understanding the role of momentum in non-convex optimization: Practical insights from a lyapunov analysis. _CoRR_ , abs/2010.00406, 2020. URL https://arxiv.org/abs/ 2010.00406. 

- Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme Ruiz, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin Fathy Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Collier, Alexey A. Gritsenko, Vighnesh Birodkar, Cristina Nader Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Pavetic, Dustin Tran, Thomas Kipf, Mario Lucic, Xiaohua Zhai, Daniel Keysers, Jeremiah J. Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters. In _ICML_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 7480–7512. PMLR, 2023. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT (1)_ , pp. 4171–4186. Association for Computational Linguistics, 2019. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ . OpenReview.net, 2021a. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ . OpenReview.net, 2021b. 

- Arthur Douillard, Qixuang Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen. Diloco: Distributed lowcommunication training of language models. _CoRR_ , abs/2311.08105, 2023. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, and Kevin Stone. The llama 3 herd of models. _CoRR_ , abs/2407.21783, 2024. 

- Nicolas Flammarion and Francis R. Bach. From averaging to acceleration, there is only a step-size. In _COLT_ , volume 40 of _JMLR Workshop and Conference Proceedings_ , pp. 658–695. JMLR.org, 2015. 

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/10256836. 

12 

Arxiv preprint 

Euhanna Ghadimi, Hamid Reza Feyzmahdavian, and Mikael Johansson. Global convergence of the heavy-ball method for convex optimization. In _ECC_ , pp. 310–315. IEEE, 2015. 

- Avrajit Ghosh, He Lyu, Xitong Zhang, and Rongrong Wang. Implicit regularization in heavy-ball momentum accelerated stochastic gradient descent. In _ICLR_ . OpenReview.net, 2023. 

- Justin Gilmer, Behrooz Ghorbani, Ankush Garg, Sneha Kudugunta, Behnam Neyshabur, David Cardoze, George Edward Dahl, Zachary Nado, and Orhan Firat. A loss curvature perspective on training instabilities of deep learning models. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum?id=OcKMT-36vUs. 

- Gabriel Goh. Why momentum really works. _Distill_ , 2017. doi: 10.23915/distill.00006. URL http://distill.pub/2017/momentum. 

- Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation. In _ICLR (Poster)_ . OpenReview.net, 2019. 

- Baptiste Goujaud, Adrien Taylor, and Aymeric Dieuleveut. Provable non-accelerations of the heavyball method. _arXiv preprint arXiv:2307.11291_ , 2023. 

- Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ , 2023. 

- Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. _CoRR_ , abs/2405.18392, 2024. 

- Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zhen Leng Thai, Kai Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies. _CoRR_ , abs/2404.06395, 2024. 

- Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. In _EMNLP/IJCNLP (1)_ , pp. 2567–2577. Association for Computational Linguistics, 2019. 

- Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, and Matt J. Kusner. No train no gain: Revisiting efficient training algorithms for transformer-based language models. In _NeurIPS_ , 2023. 

- Rahul Kidambi, Praneeth Netrapalli, Prateek Jain, and Sham M. Kakade. On the insufficiency of existing momentum schemes for stochastic optimization. In _ICLR_ . OpenReview.net, 2018. 

- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _ICLR (Poster)_ , 2015. 

- Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Eduardo Blanco and Wei Lu (eds.), _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pp. 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2012. URL https://aclanthology.org/D18-2012. 

- Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Raghavi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. Rewardbench: Evaluating reward models for language modeling. _CoRR_ , abs/2403.13787, 2024. 

- Guillaume Leclerc and Aleksander Madry. The two regimes of deep network training. _CoRR_ , abs/2002.10376, 2020. 

- Jaerin Lee, Bong Gyun Kang, Kihoon Kim, and Kyoung Mu Lee. Grokfast: Accelerated grokking by amplifying slow gradients. _CoRR_ , abs/2405.20233, 2024. 

13 

Arxiv preprint 

- Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In _ACL (1)_ , pp. 3214–3252. Association for Computational Linguistics, 2022. 

- Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training. _CoRR_ , abs/2305.14342, 2023. 

- Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In _IJCAI_ , pp. 3622–3628. ijcai.org, 2020. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR (Poster)_ . OpenReview.net, 2019. 

- James Lucas, Shengyang Sun, Richard S. Zemel, and Roger B. Grosse. Aggregated momentum: Stability through passive damping. In _ICLR (Poster)_ . OpenReview.net, 2019. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In _EMNLP_ , pp. 2381–2391. Association for Computational Linguistics, 2018. 

- Patrick G. Mulloy. Smoothing data with faster moving averages. _Technical Analysis of Stocks and Commodities Magazine_ , 12, 1994. 

- Arkaddii S Nemirovskii and Yu E Nesterov. Optimal methods of smooth convex minimization. _USSR Computational Mathematics and Mathematical Physics_ , 25(2):21–30, 1985. 

- Yurii Nesterov. A method for unconstrained convex minimization problem with the rate of convergence _o_ (1 _/k_[2] ). In _Doklady Akademii Nauk SSSR_ , 1983. URL https://api. semanticscholar.org/CorpusID:202149403. 

- Hristo Papazov, Scott Pesme, and Nicolas Flammarion. Leveraging continuous time to understand momentum when training diagonal linear networks. In _AISTATS_ , volume 238 of _Proceedings of Machine Learning Research_ , pp. 3556–3564. PMLR, 2024. 

- Razvan Pascanu, Tomás Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In _ICML (3)_ , volume 28 of _JMLR Workshop and Conference Proceedings_ , pp. 1310–1318. JMLR.org, 2013. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In _NeurIPS_ , pp. 8024–8035, 2019. 

- Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL https://arxiv.org/abs/2406.17557. 

- B.T. Polyak. Some methods of speeding up the convergence of iteration methods. _USSR Computational Mathematics and Mathematical Physics_ , 3(4):864–878, 1964. ISSN 0041-5553. doi: https://doi.org/10.1016/0041-5553(63)90382-3. URL https://www.sciencedirect. com/science/article/pii/0041555363903823. 

- Ning Qian. On the momentum term in gradient descent learning algorithms. _Neural Networks_ , 12(1):145–151, 1999. ISSN 0893-6080. doi: https://doi.org/10. 1016/S0893-6080(98)00116-6. URL https://www.sciencedirect.com/science/ article/pii/S0893608098001166. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _ICML_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 8748–8763. PMLR, 2021. 

14 

Arxiv preprint 

- Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses, 2021. 

- H. Robbins and S. Monro. A stochastic approximation method. _Annals of Mathematical Statistics_ , 22:400–407, 1951. 

- Sebastian Ruder. An overview of gradient descent optimization algorithms. _CoRR_ , abs/1609.04747, 2016. 

- Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. _International Journal of Computer Vision (IJCV)_ , 115 (3):211–252, 2015. doi: 10.1007/s11263-015-0816-y. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In _AAAI_ , pp. 8732–8740. AAAI Press, 2020. 

- Othmane Sebbouh, Robert M. Gower, and Aaron Defazio. Almost sure convergence rates for stochastic gradient descent and stochastic heavy ball. In _COLT_ , volume 134 of _Proceedings of Machine Learning Research_ , pp. 3935–3971. PMLR, 2021. 

- Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In _ICML_ , volume 80 of _Proceedings of Machine Learning Research_ , pp. 4603–4611. PMLR, 2018. 

- Guijin Son, Hanwool Lee, Sungdong Kim, Seungone Kim, Niklas Muennighoff, Taekyoon Choi, Cheonbok Park, Kang Min Yoo, and Stella Biderman. KMMLU: measuring massive multitask language understanding in korean. _CoRR_ , abs/2402.11548, 2024. 

- Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In Sanjoy Dasgupta and David McAllester (eds.), _Proceedings of the 30th International Conference on Machine Learning_ , volume 28 of _Proceedings of Machine Learning Research_ , pp. 1139–1147, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR. URL https://proceedings.mlr.press/v28/sutskever13.html. 

- Balázs Szegedy, Domonkos Czifra, and Péter K˝orösi-Szabó. Dynamic memory based adaptive optimization, 2024. URL https://arxiv.org/abs/2402.15262. 

- Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In _ICML_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 10347–10357. PMLR, 2021. 

- Hugo Touvron, Matthieu Cord, and Hervé Jégou. Deit III: revenge of the vit. In _ECCV (24)_ , volume 13684 of _Lecture Notes in Computer Science_ , pp. 516–533. Springer, 2022. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_ , pp. 5998–6008, 2017. 

- Jianyu Wang, Vinayak Tantia, Nicolas Ballas, and Michael G. Rabbat. Slowmo: Improving communication-efficient distributed SGD with slow momentum. In _ICLR_ . OpenReview.net, 2020. 

15 

Arxiv preprint 

- Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In _NUT@EMNLP_ , pp. 94–106. Association for Computational Linguistics, 2017. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. _CoRR_ , abs/1910.03771, 2019. 

- Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training BERT in 76 minutes. In _ICLR_ . OpenReview.net, 2020. 

- Kun Yuan, Bicheng Ying, and Ali H. Sayed. On the influence of momentum acceleration on online learning. _Journal of Machine Learning Research_ , 17(192):1–66, 2016. URL http: //jmlr.org/papers/v17/16-157.html. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In _ACL (1)_ , pp. 4791–4800. Association for Computational Linguistics, 2019. 

- Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In _CVPR_ , pp. 1204–1213. IEEE, 2022. 

- Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In _ICLR (Poster)_ . OpenReview.net, 2018. 

- Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, and Sham M. Kakade. Deconstructing what makes a good optimizer for language models. _CoRR_ , abs/2407.07972, 2024. 

- Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023. URL https://arxiv.org/abs/2304.11277. 

16 

|**Contents**|**Contents**|**Contents**||
|---|---|---|---|
|**1**|**Introduction**||**1**|
|**2**|**Related Work**||**3**|
|**3**|**Our**|**method: AdEMAMix**|**4**|
|**4**|**Results**||**7**|
||4.1|Transformer LLM Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|7|
||4.2|Mamba LM Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|9|
||4.3|ViT Training. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|9|
|**5**|**Conclusion**||**10**|
|**6**|**Acknowledgements**||**11**|
|**A **|**Implementation details**||**18**|
||A.1|Deriving the_β_3scheduler . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||A.2|Pytorch implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|19|
||A.3|Optax implementation<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
|**B**|**Experimental details**||**21**|
||B.1|Transformer LLM experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|21|
||B.2|Mamba LM experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
||B.3|ViT experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|24|
|**C **|**Additional results**||**25**|
||C.1|LLM experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|25|
|||C.1.1<br>In-Context Learning (ICL) results . . . . . . . . . . . . . . . . . . . . . .|25|
|||C.1.2<br>More results on forgetting . . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|||C.1.3<br>Removing the second EMA mid-training<br>. . . . . . . . . . . . . . . . . .|28|
|||C.1.4<br>Hyperparameter sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . .|29|
|||C.1.5<br>Impact of training for fewer iterations . . . . . . . . . . . . . . . . . . . . .|31|
|||C.1.6<br>Limitations of a single EMA . . . . . . . . . . . . . . . . . . . . . . . . .|32|
|||C.1.7<br>Removing_m_1by setting_β_1 = 0 . . . . . . . . . . . . . . . . . . . . . . .|34|
|||C.1.8<br>Comparing_m_1+_αm_2versus(1_−α_)_m_1+_αm_2 . . . . . . . . . . . . . .|35|
|||C.1.9<br>Miscellaneous results . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|37|
||C.2|ViT experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|39|
||C.3|Comparison with other methods<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|40|
|||C.3.1<br>Comparison with AdMeta and DEMA . . . . . . . . . . . . . . . . . . . .|40|
|||C.3.2<br>Comparison with Lion . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|41|
|||C.3.3<br>Adding a third momentum term (_∼_AggMo) . . . . . . . . . . . . . . . . .|42|



17 

Arxiv preprint 

## A IMPLEMENTATION DETAILS 

A.1 DERIVING THE _β_ 3 SCHEDULER 

Let’s consider _S_ ( _t_ ), the sum of the weights given to the last _t_ gradients by an EMA parameterized by _β ∈_ [0 _,_ 1[: 

**==> picture [90 x 30] intentionally omitted <==**

We want to know which timestep _t_ would correspond to a cumulative weight of 0 _._ 5: 

**==> picture [228 x 30] intentionally omitted <==**

Let _f_ ( _β_ ) =[ln] ln([(][0] _β[.]_[5] )[)] _[−]_[1][.][This][function][provides][how][many][past][consecutive][gradients][receive][a] cumulative weight of 0 _._ 5. 

Its inverse is: 

**==> picture [68 x 14] intentionally omitted <==**

We want a scheduler which increases _β_ from _βstart_ to _βend_ such that _f_ ( _β_ ) increases linearly. Given an interpolating parameter _µ ∈_ [0 _,_ 1], this scheduler can be written as: 

**==> picture [178 x 12] intentionally omitted <==**

By replacing _f_ and _f[−]_[1] by their respective formula, one can arrive to: 

**==> picture [190 x 24] intentionally omitted <==**

By setting _βend_ = _β_ 3 and _µ_ = _Tβt_ 3[, we arrive to the] _[ β]_[3][-scheduler introduced in §][ 3][.][We show the] shape of our scheduler and compare it to a linear scheduler in Fig. 7. 

**==> picture [358 x 149] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 . 00<br>0 . 98<br>0 . 96<br>0 . 94 Our β -scheduler: β start = 0 . 9 , β end = 0 . 9999<br>Linear scheduler: βstart = 0 . 9 , βend = 0 . 9999<br>0 . 92 Our β -scheduler: βstart = 0 . 9 , βend = 0 . 999<br>Linear scheduler: βstart = 0 . 9 , βend = 0 . 999<br>0 . 90<br>0 . 0 0 . 2 0 . 4 0 . 6 0 . 8 1 . 0<br>µ<br>)<br>µ<br>(<br>β<br>**----- End of picture text -----**<br>


Figure 7: **AdEMAMix’s** _β_ 3 **scheduler.** We compare our scheduler to a linear scheduler for _βstart_ = 0 _._ 9 and _βend ∈{_ 0 _._ 999 _,_ 0 _._ 9999 _}_ . While our scheduler looks more aggressive at first glance, it increases fast for smaller values of _β_ , and slowly for larger ones associated. This makes sense as the same increase of _β_ for larger _β_ values has a bigger impact than that same increase applied to a smaller value of _β_ . The two linear schedulers look practically the same, despite values of _βend_ differing by one order of magnitude. This is not the case with our scheduler. 

18 

Arxiv preprint 

## A.2 PYTORCH IMPLEMENTATION 

The following is a code skeleton for our AdEMAMix optimizer in Pytorch (Paszke et al., 2019). The full implementation of AdEMAMix in Pytorch is provided in the following repository: https: //github.com/apple/ml-ademamix 

Listing 1: AdEMAMix code skeleton using Pytorch 

**==> picture [391 x 461] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 import math<br>2 import torch<br>3 from torch.optim import Optimizer<br>4<br>5<br>6 class AdEMAMix(Optimizer):<br>7<br>8 def __init__(self,<br>9 params,<br>10 lr=1e-3,<br>11 betas=(0.9,0.999,0.9999),<br>12 alpha=5.0,<br>13 T_beta3=0,<br>14 T_alpha=0,<br>15 eps=1e-8,<br>16 weight_decay=0.0):<br>17 # init the optimizer<br>18<br>19 @torch.no_grad()<br>20 def step(self):<br>21<br>22 for group in self.param_groups:<br>23<br>24 lr = group["lr"]<br>25 lmbda = group["weight_decay"]<br>26 eps = group["eps"]<br>27 beta1, beta2, beta3_final = group["betas"]<br>28 T_beta3 = group["T_beta3"]<br>29 T_alpha = group["T_alpha"]<br>30 alpha_final = group["alpha"]<br>31<br>32 for p in group["params"]:<br>33<br>34 grad = p.grad<br>35 state = self.state[p]<br>36<br>37 # State initialization<br>38 if len(state) == 0:<br>39 state["step"] = 0 # step counter used for bias correction<br>40 state["m1"] = torch.zeros_like(p) # fast EMA<br>41 state["m2"] = torch.zeros_like(p) # slow EMA<br>42 state["nu"] = torch.zeros_like(p) # second moment estimate<br>43<br>44 m1, m2, nu = state["m1"], state["m2"], state["nu"]<br>45<br>46 # Bias correction: no correction for beta3’s EMA<br>47 state["step"] += 1<br>48 bias_correction1 = 1 - beta1 ** state["step"]<br>49 bias_correction2 = 1 - beta2 ** state["step"]<br>50<br>51 # Calling the schedulers for alpha and beta3<br>52 alpha = alpha_scheduler(state["step"], start=0, end=alpha_final, T=T_alpha)<br>53 beta3 = beta3_scheduler(state["step"], start=beta1, end=beta3_final, T=T_beta3)<br>54<br>55 # Update the EMAs<br>56 m1.mul_(beta1).add_(grad, alpha=1 - beta1)<br>57 m2.mul_(beta3).add_(grad, alpha=1 - beta3)<br>58 nu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)<br>59<br>60 # Compute step<br>61 denom = (nu.sqrt() / math.sqrt(bias_correction2)).add_(eps)<br>62 update = (m1.div(bias_correction1) + alpha * m2) / denom<br>63<br>64 # Add weight decay<br>65 update.add_(p, alpha=lmbda)<br>66<br>67 # Apply the update scaled by the learning rate<br>68 p.add_(-lr * update)<br>69<br>70 return loss<br>**----- End of picture text -----**<br>


19 

Arxiv preprint 

## A.3 OPTAX IMPLEMENTATION 

The following is a code skeleton for our AdEMAMix optimizer in Optax, an optimization library based on JAX (DeepMind et al., 2020). The full implementation of AdEMAMix in Jax is provided in the following repository: https://github.com/apple/ml-ademamix 

Listing 2: AdEMAMix code skeleton using JAX and Optax 

**==> picture [391 x 507] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 from typing import NamedTuple<br>2 import chex<br>3 from optax._src import transform, combine, base, numerics<br>4 from jax import tree_util as jtu<br>5 import jax.numpy as jnp<br>6<br>7<br>8 class ScaleByAdemamixState(NamedTuple):<br>9 count: chex.Array<br>10 count_m2: chex.Array<br>11 m1: base.Updates<br>12 m2: base.Updates<br>13 nu: base.Updates<br>14<br>15<br>16 def ademamix(lr,<br>17 b1=0.9,<br>18 b2=0.999,<br>19 b3=0.9999,<br>20 alpha=5.0,<br>21 b3_scheduler=None,<br>22 alpha_scheduler=None,<br>23 eps=1e-8,<br>24 weight_decay=0.0):<br>25 return combine.chain(<br>26 scale_by_ademamix(b1, b2, b3, alpha, b3_scheduler, alpha_scheduler, eps),<br>27 transform.add_decayed_weights(weight_decay),<br>28 transform.scale_by_learning_rate(lr),<br>29 )<br>30<br>31<br>32 def scale_by_ademamix(b1,<br>33 b2,<br>34 b3,<br>35 alpha,<br>36 b3_scheduler,<br>37 alpha_scheduler,<br>38 eps):<br>39<br>40 def init_fn(params):<br>41 m1 = tree_zeros_like(params) # fast EMA<br>42 m2 = tree_zeros_like(params) # slow EMA<br>43 nu = tree_zeros_like(params) # second moment estimate<br>44 return ScaleByAdemamixState(<br>45 count=jnp.zeros([], jnp.int32),<br>46 count_mu2=jnp.zeros([], jnp.int32),<br>47 m1=m1,<br>48 m2=m2,<br>49 nu=nu<br>50 )<br>51<br>52 def update_fn(updates, state, params=None):<br>53 del params<br>54 c_b3 = b3_scheduler(state.count_m2) if b3_scheduler is not None else b3<br>55 c_alpha = alpha_scheduler(state.count_m2) if alpha_scheduler is not None else alpha<br>56 m1 = tree_update_moment(updates, state.m1, b1, 1) # m1 = b1 * m1 + (1-b1) * updates<br>57 m2 = tree_update_moment(updates, state.m2, c_b3, 1)<br>58 nu = tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)<br>59 count_inc = numerics.safe_int32_increment(state.count)<br>60 count_m2_inc = numerics.safe_int32_increment(state.count_m2)<br>61 m1_hat = tree_bias_correction(m1, b1, count_inc)<br>62 nu_hat = tree_bias_correction(nu, b2, count_inc)<br>63 updates = jtu.tree_map(<br>64 lambda m1_, m2_, v_: (m1_+c_alpha*m2_)/(jnp.sqrt(v_)+eps),<br>65 m1_hat,<br>66 m2,<br>67 nu_hat<br>68 )<br>69 return updates, ScaleByAdemamixState(<br>70 count=count_inc,<br>71 count_m2=count_m2_inc,<br>72 m1=m1,<br>73 m2=m2,<br>74 nu=nu<br>75 )<br>76<br>77 return base.GradientTransformation(init_fn, update_fn)<br>**----- End of picture text -----**<br>


20 

Arxiv preprint 

## B EXPERIMENTAL DETAILS 

## B.1 TRANSFORMER LLM EXPERIMENTS 

**Architecture details.** Our architecture is based on the transformer decoder of Vaswani et al. (2017). We use learnt positional encoding. We use a SentencePiece (Kudo & Richardson, 2018) tokenizer with a vocabulary of 32000 tokens. The model specific parameters for the different sizes are summarized in Table 1. Our implementation is using Jax (DeepMind et al., 2020), and we train using bfloat16, except for normalization modules and softmax which use float32. The optimizer states and operations are in float32. 

Table 1: **Model parameters for our LLM experiments.** 

|Model parameters|110M<br>330M<br>1_._3B|
|---|---|
|||
|Hidden size<br>FFW expansion factor<br>Attention heads<br>Layers|768<br>1024<br>2048<br>4<br>4<br>4<br>12<br>16<br>16<br>12<br>24<br>24|



**How did we tune the hyperparameters?** Starting from our smallest models (110M parameters), we first tuned hyperparameters for our AdamW models. We then use the best hyperparameters found as a starting point for AdEMAMix and tuned _β_ 3 and _α_ . When we use schedulers for _α_ and _β_ 3, we set _Tα_ = _Tβ_ 3 = _T_ , with _T_ being the total number of iterations. Table 2 summarizes the hyperparameters we tried for this model size. The impact of AdEMAMix’s hyperparameters is discussed extensively in App. C.1.4. For our 330M parameter models, we mostly kept the best hyperparameters found for the 110M model and re-tuned the learning rate and gradient clipping. For AdEMAMix, we additionally tested multiple _β_ 3 and _α_ values. We summarize this process in Table 3. Finally, for our 1 _._ 3B parameter models, we re-iterated the same process, re-tuning only the learning rate and gradient clipping parameters for our AdamW runs. When trying to transfer the best learning rate found to AdEMAMix, we found it to be too high, causing instabilities we couldn’t fix using gradient clipping. For this reason, we also tuned the learning rate for AdEMAMix for this model size. This process is described in Table 4. 

Table 2: **Hyperparameter tuning for our** 110 **M parameter LLM models.** In this table we report the hyperparameters we tuned for our 110M parameter models. When multiple values are given, we bold the parameters we found to give the best results. 

|Hyperparameter|Value|
|---|---|
|||
|Learning rate_η_<br>Number of warmup steps<br>Sequence length<br>Weight decay_λ_<br>Learning rate decay scheduler<br>Batch size<br>Gradient clipping|0_._005,0_._002,**0**_._**001**,0_._0005,0_._0001<br>2000,**3000**,4000,5000,6000<br>1024<br>**0**_._**1**,0_._0<br>no-decay,**cosine-decay**<br>64<br>None,5_._0,1_._0,**0**_._**5**|
|||
|AdamW_β_1<br>AdamW_β_2|**0**_._**9**,0_._99,0_._999,0_._9999<br>0_._95,**0**_._**999**|
|||
|AdEMAMix_β_1<br>AdEMAMix_β_2<br>AdEMAMix_β_3<br>AdEMAMix_α_|0_._9<br>0_._999<br>0_._999,**0**_._**9999**,0_._99999<br>2,4,6,**8**,**10**,15,20|



**Hyperparameters for experiments switching from AdamW and AdEMAMix.** For experiments in Fig. 5b and Fig. 5c, when we switch from AdamW to AdEMAMix during training, for our 110M parameter models (Fig. 5b), we use _α_ = 2 _, β_ 3 = 0 _._ 9999 and _Tα,β_ 3 = 0. For our 1 _._ 3B parameter 

21 

Arxiv preprint 

Table 3: **Hyperparameter tuning for our** 330 **M parameter LLM models.** In this table we report the hyperparameters we tuned for our 330M parameter models. When multiple values are given, we bold the parameters we found to give the best results. 

|Hyperparameter|Value|
|---|---|
|||
|Learning rate_η_<br>Number of warmup steps<br>Sequence length<br>Weight decay_λ_<br>Learning rate decay scheduler<br>Batch size<br>Gradient clipping|0_._001,**0**_._**0005**,0_._0001<br>3000<br>1024<br>0_._1<br>cosine-decay<br>96<br>1_._0,0_._5,**0**_._**1**|
|||
|AdamW_β_1<br>AdamW_β_2|0_._9<br>0_._999|
|||
|AdEMAMix_β_1<br>AdEMAMix_β_2<br>AdEMAMix_β_3<br>AdEMAMix_α_|0_._9<br>0_._999<br>0_._999,**0**_._**9999**,0_._99999<br>5,**8**,10,15|



Table 4: **Hyperparameter tuning for our** 1 _._ 3 **B parameter LLM models.** In this table we report the hyperparameters we tuned for our 1 _._ 3B parameter models. When multiple values are given, we bold the parameters we found to give the best results. 

|Hyperparameter|Value|
|---|---|
|||
|Number of warmup steps<br>Sequence length<br>Weight decay_λ_<br>Learning rate decay scheduler<br>Batch size<br>Gradient clipping|3000<br>1024<br>0_._1<br>cosine-decay<br>128<br>1_._0,0_._5,**0**_._**1**|
|||
|Learning rate_η_for AdamW<br>AdamW_β_1<br>AdamW_β_2|0_._001,**0**_._**0005**,0_._0003,0_._0001,0_._00005<br>0_._9<br>0_._999|
|||
|Learning rate_η_for AdEMAMix<br>AdEMAMix_β_1<br>AdEMAMix_β_2<br>AdEMAMix_β_3<br>AdEMAMix_α_|0_._0005,**0**_._**0003**<br>0_._9<br>0_._999<br>0_._999,**0**_._**9999**,0_._99999<br>1,3,**5**,8,10,15|



models (Fig. 5c), we use _α_ = 1 _, β_ 3 = 0 _._ 9999 and _Tα,β_ 3 = 0. Other hyperparameters are inherited from the AdamW model we are switching from. 

**Hyperparameters used our constant learning rate scheduler experiments.** For Fig. 3b we use _η_ = 10 _[−]_[4] , the remaining of the hyperparameters are identical as those used for our other 1 _._ 3B experiments and provided in Table 4. 

**Hyperparameters used in our forgetting experiments.** For the experiments in Fig. 4, we used 110M parameter models and hyperparameters from Table 2. 

22 

Arxiv preprint 

- B.2 MAMBA LM EXPERIMENTS 

**Architecture details.** We use a Mamba architecture (Gu & Dao, 2023) with an embedding dimension of 768, an expansion factor of 2, a state size of 16, and a depth of 24. We use a batch size of 120 sequences of 1024 tokens. We use the EleutherAI/gpt-neox-20b tokenizer from Black et al. (2022), using the HuggingFace library (Wolf et al., 2019). 

**Hyperparameter details.** We used parameters mostly taken from Gu & Dao (2023): 

- Learning rate: _η_ = 0 _._ 0006, 

- Warmup steps: 3 _k_ , 

- _η_ -scheduler type: cosine decay to 10 _[−]_[5] , 

- Weight-decay: _λ_ = 0 _._ 1, 

- Gradient clipping value: 1, 

- Total training steps _T ∈{_ 64 _k,_ 128 _k}_ . 

For experiments with AdamW, we use _β_ 1 = 0 _._ 9 and _β_ 2 = 0 _._ 999. For AdEMAMix, we use _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 999 _, β_ 3 = 0 _._ 9999 _, Tα,β_ 3 = _T_ and _α_ = 8. 

23 

Arxiv preprint 

## B.3 VIT EXPERIMENTS 

**Architecture details.** We use a ViT architecture following Dosovitskiy et al. (2021b). The architecture details for both our 24M and 86M parameter models can be seen in Table 5. We do not use an EMA of the iterates as our final model. Our implementation is in Pytorch (Paszke et al., 2019), and use bfloat16. 

Table 5: **Model parameters for our ViT experiments.** 

|Model parameters|24M<br>86M|
|---|---|
|||
|Patch size<br>Number of patches<br>Image size<br>Embedding dim.<br>MLP dim.<br>Layers<br>Number of heads|16<br>16<br>14_×_14<br>14_×_14<br>224<br>224<br>384<br>768<br>1536<br>3072<br>12<br>12<br>6<br>12|



**Data augmentation and Mixup.** While more sophisticated augmentation methods exist (Touvron et al., 2022), we simply use random-resized cropping, random horizontal flip ( _p_ = 0 _._ 5), and random color jitter (applied with a probability _p_ = 0 _._ 8). 

**How did we tune the hyperparameters?** For each model size, we started by tuning the hyperparameters for the AdamW baseline. The hyperparameters we tuned and the values we retained for our three different settings are in Table 6. For our experiments on ImageNet-1k, given that all models overfit the dataset, we select the best model according to the minimum validation loss, akin to using early stopping. For AdEMAMix, for each setting, we use the hyperparameters of the best AdamW model and only tune _α_ and _β_ 3. For each setting, we train 8 models using _α ∈{_ 1 _,_ 5 _,_ 10 _,_ 15 _,_ 20 _}_ and _β_ 3 _∈{_ 0 _._ 999 _,_ 0 _._ 9999 _}_ . All of the AdEMAMix models are shown in Fig. 6 and Fig. 30. Only the best AdamW model is shown in Fig. 30. 

Table 6: **Hyperparameters tuning for our ViT AdamW experiments.** We started our hyperparameter search from the values given in (Dosovitskiy et al. (2021b), Table 3), which recommends using (learning-rate, weight decay, dropout)= (0 _._ 003 _,_ 0 _._ 3 _,_ 0 _._ 1) when training on ImageNet-1k and (0 _._ 001 _,_ 0 _._ 03 _,_ 0 _._ 1) when training on ImageNet-21k. The various hyperparameters we experimented with are in the following table, values which gave us the lowest validation loss are in bold. For each setting, AdEMAMix experiments use the same hyperparameters as the best AdamW baselines. 

|Setting|24M params,11M images<br>(ImageNet-21k)|86M params,11M images<br>(ImageNet-21k)|86M params,1_._3M images<br>(ImageNet-1k)|
|---|---|---|---|
|||||
|Learning rate<br>Weight decay<br>Dropout<br>AdamW_β_1<br>Batch size|0_._001_,_**0**_._**003**_,_0_._005<br>**0**_._**03**_,_0_._1_,_0_._3<br>0_._1<br>**0**_._**9**_,_0_._99_,_0_._999<br>4096|**0**_._**001**_,_0_._003<br>0_._03_,_**0**_._**1**_,_0_._3<br>0_._1<br>**0**_._**9**_,_0_._99_,_0_._999<br>4096|0_._001_,_0_._003_,_**0**_._**005**<br>0_._03_,_0_._1_,_**0**_._**3**_,_0_._5_,_0_._7<br>0_._1<br>**0**_._**9**_,_0_._99_,_0_._999<br>4096|



24 

Arxiv preprint 

## C ADDITIONAL RESULTS 

## C.1 LLM EXPERIMENTS 

## C.1.1 IN-CONTEXT LEARNING (ICL) RESULTS 

**In-Context Learning (ICL) results.** We evaluate our largest (1 _._ 3B) LLM models on in-context learning tasks. We use the lm-eval package (Gao et al., 2023). We consider the following tasks: 

HellaSwag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2020), ARC (Bhakthavatsalam et al., 2021), BoolQ (Clark et al., 2019), LogiQA (Liu et al., 2020), MathQA (Amini et al., 2019), MMLU (Son et al., 2024), OpenbookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), PubmedQA (Jin et al., 2019), RewardBench (Lambert et al., 2024), Sciq (Welbl et al., 2017), TruthfulQA (Lin et al., 2022). 

The two AdEMAMis and AdamW models we benchmark have been trained for 1M steps, with a batch size of 128, corresponding to around 131B tokens. The results of the evaluation are in Table 7. We observe how the model trained with AdEMAMix is outperforming the AdamW model on nearly all of the tasks, sometimes by a large margin for e.g. PubmedQA. In addition to Table 7, we track the evolution of some of those scores during training for MMLU, RewardBench, ARC-Challenge, and ARC-Easy, results can be seen in Fig. 8. For that figure, we modify the way MMLU is evaluated. Instead of appending all the multiple answers, we concatenate the question prompt and each answer separately, and pick the most likely answer. We found it allows a much better comparison of smaller models, which otherwise fluctuates around random guessing. 

Table 7: **In-context learning results for our** 1 _._ 3 **B parameter models.** We compare AdamW and AdEMAMix models trained on 131B tokens. We use the lm-eval package. 

|Task|AdamW<br>AdEMAMix|Task|AdamW<br>AdEMAMix|
|---|---|---|---|
|||||
|ARC-Challenge<br>ARC-Easy<br>BoolQ<br>HellaSwag<br>LogiQA<br>MathQA<br>Winogrande<br>MMLU|0_._262<br>**0**_._**274**<br>0_._612<br>**0**_._**619**<br>0_._569<br>**0**_._**576**<br>0_._426<br>**0**_._**436**<br>**0**_._**235**<br>0_._225<br>0_._226<br>**0**_._**236**<br>0_._563<br>**0**_._**580**<br>0_._244<br>**0**_._**248**|OpenbookQA<br>PIQA<br>PubmedQA<br>RewardBench<br>RewardBench (reasoning)<br>Sciq<br>TruthfulQA|**0**_._**240**<br>0_._238<br>**0**_._**715**<br>**0**_._**715**<br>0_._556<br>**0**_._**632**<br>0_._569<br>**0**_._**573**<br>0_._617<br>**0**_._**630**<br>0_._903<br>**0**_._**907**<br>**0**_._**361**<br>0_._352|



**==> picture [396 x 104] intentionally omitted <==**

**----- Start of picture text -----**<br>
AdamW 0 . 63 AdamW AdamW AdamW<br>AdEMAMix AdEMAMix AdEMAMix 0 . 60 AdEMAMix<br>0 . 62 0 . 26<br>0 . 30<br>0 . 61 0 . 24 0 . 55<br>0 . 28 0 . 60 0 . 22<br>0 . 50<br>0 . 59<br>0 . 20<br>15B 75B 131B 15B 75B 131B 15B 75B 131B 15B 75B 131B<br>Tokens Tokens Tokens Tokens<br>(a) MMLU. (b) RewardBench. (c) ARC-Challenge. (d) ARC-Easy.<br>Accuracy Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 8: **In-context learning results for our** 1 _._ 3 **B models.** We periodically measure the performances of AdamW and AdEMAMix models on in-context learning tasks. Except for a couple exceptions, AdEMAMix is performing better than AdamW. 

25 

Arxiv preprint 

## C.1.2 MORE RESULTS ON FORGETTING 

**Evolution of forgetting during training.** In Fig. 4 in the main paper, we follow the loss on one specific batch over the entire training. In the following experiment, we do a closeup on the forgetting of different batches as training progresses. The goal being to visualize how later batches are ultimately more remembered, and compare the forgetting profiles of AdamW and AdEMAMix. For this, in Fig. 9a and Fig. 9b, we follow fixed batches at different stages of the training process. Every 10 _k_ iterations, we measure the loss over a specific batch _B_ before and after training on that batch. 

**==> picture [295 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
250 k<br>0 . 0<br>0 . 0<br>− 0 . 5<br>− 0 . 2 200 k<br>− 1 . 0<br>− 0 . 4<br>− 0 . 6 a −− 12 .. 50 150 k<br>100 k<br>−− 01 .. 80 | | i −− 23 .. 50 NoNN || 50 k<br>EE I| - -<br>0 5000 10000 0 5000 10000<br>Iterations after training on the batch Iterations after training on the batch<br>(a) AdamW. (b) AdEMAMix.<br>loss batch<br>tracked<br>of<br>Normalized<br>Index<br>**----- End of picture text -----**<br>


Figure 9: **Comparing forgetting between AdamW and AdEMAMix.** In **(a)** and **(b)** , we follow— over 10 _k_ iterations—the loss of training batches _Bτ_ for _τ ∈{_ 30 _k,_ 40 _k, . . . ,_ 250 _k}_ . We measure the loss right before training on _Bτ_ , and then monitor this loss over the next 10 _k_ steps. For instance, the most yellow curve represents the loss on batch _B_ 250 _k_ that we track for the last 6 _k_ iterations of training (we train for 256 _k_ iterations). The loss is normalized such that the loss for _Bτ_ is 0 and the one at time _τ_ + 50 is _−_ 1. This allows us to overlap curves and observe the tendency of the model to forget batches throughout training. We observe that AdEMAMix improves its loss on training batches _Bτ_ over many timesteps, a striking difference compared to AdamW, which has its loss on _Bτ_ increasing faster. Moreover—likely due to the learning rate scheduler—the models forget significantly slower at the end of training. In this later training stage ( _τ >_ 180 _k_ ), the difference between the two optimizers is especially pronounced. While this visualization allows to efficiently compare forgetting at different stages of training, the normalization prevents us from comparing the drop in loss between AdEMAMix and AdamW. To compare loss values, see Fig. 4. 

**Why are training batches forgotten more at the end of training?** Given the previous observation about how both AdamW and AdEMAMix models are forgetting the training batches at different paces given different stages of training, one can wonder what is causing this phenomenon? To answer this question, we contrast the results of Fig. 4 with results obtained using a different scheduler with a constant learning rate and a linear decay. We train 110M parameter models for 300 _k_ iterations, using a max learning rate of 0 _._ 001 and a batch size of 64. Results for those experiments are in Fig. 10. Those results indicate that the decaying learning rate might be the main parameter controlling how much a batch is remembered during training. This has interesting implications when selecting which learning rate decay to use. A cosine decay, with a rather long period of decay, might remember more than a constant learning rate scheduler with only a small number of steps of linear decay at the end. 

26 

Arxiv preprint 

**==> picture [376 x 188] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 1<br>AdamW training<br>without B<br>3 . 0 AdamW training<br>using B at t = tB<br>2 . 9 tB<br>2 . 8<br>2 . 7<br>siEsE<br>70k 230 k 300 k 70k 270 k 70k 280 k 70k 290 k<br>3 . 1<br>AdEMAMix training<br>without B<br>3 . 0 AdEMAMix training<br>using B at t = tB<br>2 . 9 tB<br>2 . 8<br>2 . 7<br>SHEE<br>70k 230 k 300 k 70k 270 k 70k 280 k 70k 290 k<br>(a)  tB = 230 k . (b)  tB = 270 k . (c)  tB = 280 k . (d)  tB = 290 k .<br>B<br>batch<br>on<br>Loss<br>B<br>batch<br>on<br>Loss<br>**----- End of picture text -----**<br>


Figure 10: **Measuring forgetting using a held-out batch** _B_ **, using a constant** _η_ **-scheduler with linear decay.** The top row is for AdamW, the bottom row is for AdEMAMix. The setup is similar to the one from Fig. 4, except that we now use a different learning rate scheduler. We use 3000 steps of linear warmup, them 247 _k_ steps of constant learning rate _η_ = 0 _._ 001, and finally 50 _k_ steps of linear decay (represented by the shaded green area in the above plots). In Fig. 4, we observed that the batches used later during training are remembered more by both AdamW and AdEMAMix. Multiple hypotheses could explain this phenomenon: (i) it could be that the learning rate decay is the main parameter controlling forgetting, or (ii) it could be that a sufficient number of steps are required to reach a part of the optimization landscape where gradients are more orthogonal and baches are remembered. The present experiment refutes that second hypothesis. Indeed, before the start of the _η_ -decay (as seen in **(a)** ), training batches are not remembered at the end of training. They only start to be remembered when the decay periods starts, as seen in **(b,c,d)** . Moreover, we can make the same observation as in Fig. 4: compared to AdamW, AdEMAMix models tend to keep the training data longer in memory. 

27 

Arxiv preprint 

## C.1.3 REMOVING THE SECOND EMA MID-TRAINING 

**Removing the second EMA mid-training.** In Fig. 5b and Fig. 5c, we looked at what is happening when we switch from AdamW to AdEMAMix in the middle of training. In this section, we will look at the opposite conversion: removing the _**m**_ 2 parameter of AdEMAMix during training, effectively switching back to AdamW. Results for this experiment can be seen in Fig. 11. Right after the switch, we observe a drop of the loss, followed by an increase and finally back to convergence. Ultimately, the final loss value is in between the ones obtained training only using AdamW and only using AdEMAMix. 

**==> picture [199 x 145] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4<br>tswitch<br>AdamW<br>3 . 3 AdEMAMix α = 10 . 0, β 3 = 0 . 9999, T α = T β 3 = 256 k<br>AdEMAMix α = 10 . 0, β 3 = 0 . 9999, Tα = Tβ 3 = 90 k<br>3 . 2<br>3 . 1<br>3 . 0<br>2 . 9<br>0 50000 100000 150000 200000 250000<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 11: **Switching from AdEMAMix to AdamW at** _tswitch_ = 128 _k_ **.** Using our AdEMAMix optimizer, we train two 110M parameters models with different _Tα_ = _Tβ_ 3 _∈{_ 90 _k,_ 256 _k}_ . For historical reasons, those experiments were done using a batch size of 32 instead of 64 (as we used for all the other 110M parameter experiments in this paper). We switch from AdEMAMix to AdamW at _tswitch_ = 128 _k_ . At first, the removal of the + _α_ _**m**_ 2 contribution in the update is suddenly decreasing the effective learning rate, causing the loss to drop suddenly. The loss then increases slightly and ends up in between the AdamW only (blue curve) and AdEMAMix only (grey curves) losses. 

28 

Arxiv preprint 

## C.1.4 HYPERPARAMETER SENSITIVITY 

**Hyperparameter sensitivity.** Depending on whether the _α_ and _β_ 3 schedulers are used, AdEMAMix adds up to 4 new hyperparameters: _α_ , _β_ 3, _Tα_ and _Tβ_ 3. In all our experiments we tied _Tα_ = _Tβ_ 3 = _Tα,β_ 3. In this section we analyze the sensitivity of AdEMAMix to those hyperparameters. We study the impact of _α_ and _β_ 3 in Fig. 12, revealing wide ranges of values for which hyperparameters are outperforming the AdamW baseline. We study the sensitivity of _Tα,β_ 3 as well as justify the choice of using a scheduler on _α_ in Fig. 13. When training from scratch _Tα,β_ 3 needs simply to be large enough to avoid early divergence. In Fig. 15 we study the sensitivity to the gradient clipping and AdEMAMix’s _β_ 1 value. While gradient clipping can help stabilize training and smooth the loss curves, it has little impact over the final loss value. Reducing the value of _β_ 1, we observe some loss spikes, yet the final loss value is relatively unaffected. 

**==> picture [324 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4 AdamW AdamW<br>AdEMAMix β 3 = 0 . 9999, α = 2 3 . 4 AdEMAMix β 3 = 0 . 99, α = 8<br>3 . 3 AdEMAMix β 3 = 0 . 9999, α = 6 AdEMAMix β 3 = 0 . 999, α = 8<br>AdEMAMix β 3 = 0 . 9999, α = 10 AdEMAMix β 3 = 0 . 9999, α = 8<br>3 . 2 AdEMAMixAdEMAMixAdEMAMix βββ 3 33 === 000 ... 9999,9999,9999, ααα === 152030 3 . 2 AdEMAMixAdEMAMix ββ 3 3 == 00 .. 99999,999995, αα ==88<br>3 . 1 AdEMAMix β 3 = 0 . 9999, α = 40<br>3 . 0 3 . 0<br>2 . 9<br>2 . 8 2 . 8<br>0 50000 100000 150000 200000 250000 0 50000 100000 150000 200000 250000<br>Iterations Iterations<br>(a) Sensitivity to  α . (b) Sensitivity to  β 3.<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 12: **Sensitivity of AdEMAMix to** _α_ **and** _β_ 3 **.** We test the sensitivity of 110M parameter models to the values of _α_ and _β_ 3. In **(a)** , we vary _α ∈{_ 2 _,_ 6 _,_ 10 _,_ 15 _,_ 20 _,_ 30 _,_ 40 _}_ while keeping all other parameters equal ( _β_ 3 = 0 _._ 9999 _, Tα,β_ 3 = 256 _k,_ clipping = 0 _._ 5 _, η_ = 0 _._ 001 _, λ_ = 0 _._ 1). We see that the larger the _α_ , the more significant is the loss decrease early during training, yet this does not necessarily translates into a better final loss. The last iterate’s loss is larger for large values of _α_ . For this model, a sweet spot seems to be reach for _α_ = 10. Moreover, the span of values resulting in good results is very large. In **(b)** , we now vary _β_ 3 _∈{_ 0 _._ 99 _,_ 0 _._ 999 _,_ 0 _._ 9999 _,_ 0 _._ 99999 _,_ 0 _._ 999995 _}_ . We first observe that—except for _β_ 3 = 0 _._ 999995—all those values are outperforming the AdamW baseline. Moreover, we observe that there is a sweet spot, at _β_ 3 = 0 _._ 9999, after which the final loss increases. 

**==> picture [324 x 122] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 8<br>AdamW AdamW<br>3 . 6 AdEMAMix AdEMAMix T T αα = = T T ββ 33 = = 10 30 k k 3 . 4 AdEMAMixAdEMAMix TTββ 33 == 256256 kk ,, TTαα == 010 k<br>AdEMAMix Tα = Tβ 3 = 50 k AdEMAMix Tβ 3 = 256 k , Tα = 50 k<br>3 . 4 AdEMAMixAdEMAMix TTαα == TTββ 33 == 7090 kk 3 . 2 AdEMAMix AdEMAMix T T β β 33 = = 256 256 k k , , T T α α = = 70 256 k k<br>AdEMAMix Tα = Tβ 3 = 128 k<br>3 . 2<br>3 . 0<br>3 . 0<br>2 . 8 2 . 8<br>0 50000 100000 150000 200000 250000 0 50000 100000 150000 200000 250000<br>Iterations Iterations<br>(a) Sensitivity to  Tα =  Tβ 3 =  Tα,β 3 . (b) Sensitivity to  Tα .<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 13: **Sensitivity of AdEMAMix to** _Tα_ **and** _Tβ_ 3 **.** For all the AdEMAMix experiments in this figure, we used _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 999 _, β_ 3 = 0 _._ 9999 _, α_ = 10. In **(a)** , we test the sensitivity of 110M parameter models to the values of _Tα_ and _Tβ_ 3. We tied the two values as in all the experiments in the main paper: _Tα_ = _Tβ_ 3 = _Tα,β_ 3. We observe that when not enough warmup steps are used, the iterates either diverge or converge to slightly worse loss values. In **(b)** we set _Tβ_ 3 = 256 _k_ and only vary _Tα_ . We observe that _Tα_ needs to be large enough to avoid divergence. This demonstrates the necessity of using a warmup on _α_ . This being said, the necessity of increasing _α_ linearly depends on the value of _α_ . Using a smaller _α_ value can work with _Tα_ = 0, but using an _α_ -scheduler alleviates any instability issue and increases the range of optimal _α_ values (see Fig. 14). Moreover, in other experiments, we observe how not using schedulers yields detrimental large updates in the early training phase, see App. C.1.9, Fig. 27. 

29 

Arxiv preprint 

**==> picture [324 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4 AdamW 3 . 2 AdEMAMix Tα = 0 , α = 1<br>AdEMAMix Tβ 3 = 256 k , Tα = 0, α = 2 AdEMAMix Tα = 0 , α = 3<br>3 . 3 AdEMAMixAdEMAMix TTα,ββ 3 =3 =256256 k , kT , αα ==0,2 α = 6 3 . 0 AdEMAMixAdEMAMix TTαα == 0256 , αk, α = 5= 5<br>3 . 2 AdEMAMix T α,β 3 = 256 k , α = 6<br>2 . 8<br>3 . 1<br>3 . 0 2 . 6<br>2 . 9<br>2 . 4<br>2 . 8<br>0 50000 100000 150000 200000 250000 0 50000 100000 150000 200000 250000<br>Iterations Iterations<br>(a) No  α -scheduler (110M). (b) No  α -scheduler (1 . 3B).<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 14: **The** _α_ **-scheduler reduces the sensitivity to** _α_ **.** We compare AdEMAMix experiments with and without _α_ -scheduler. The experiments with _α_ -scheduler are more stable. In **(b)** , for 1 _._ 3B parameter models, only _α_ = 1 converges in a stable fashion (the purple curve with _α_ = 5 diverged early), and reaches a final loss slightly worse than the AdEMAMix model trained using an _α_ -scheduler over 256 _k_ steps. In **(a)** , for smaller 110M parameter models, we observe (i) that it is possible to converge without _α_ -scheduler for small _α_ values, as well as (ii) that it seems easier to reach good loss values with a scheduler (the dotted lines have a lower final loss on average). 

**==> picture [324 x 122] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4<br>AdamW AdamW<br>3 . 3 AdEMAMixAdEMAMix ggrad-clip=rad-clip= 01 .. 10,, αα == 88 3 . 4 AdEMAMix AdEMAMix β β 1 1 = = 0 0 . . 0 1<br>AdEMAMix grad-clip= 5 . 0, α = 8 AdEMAMix β 1 = 0 . 3<br>3 . 2 AdEMAMix grad-clip= 0 . 1, α = 2 AdEMAMix β 1 = 0 . 5<br>AdEMAMix grad-clip= 1 . 0, α = 2 3 . 2 AdEMAMix β 1 = 0 . 7<br>3 . 1 AdEMAMix grad-clip= 5 . 0, α = 2 AdEMAMixAdEMAMix ββ 11 == 00 .. 999<br>3 . 0 3 . 0<br>2 . 9<br>2 . 8<br>2 . 8<br>0 50000 100000 150000 200000 250000 0 50000 100000 150000 200000 250000<br>Iterations Iterations<br>(a) Sensitivity to gradient clipping. (b) Sensitivity to  β 1.<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 15: **Sensitivity of AdEMAMix to** _β_ 1 **and the gradient clipping value.** Unless specified in the caption, all the AdEMAMix experiments in this figure used _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 999 _, β_ 3 = 0 _._ 9999 _, α_ = 10 _, Tα,β_ 3 = 256 _k_ . In **(a)** we vary the amount of gradient clipping used during training. We notice how gradient clipping can help to smooth the curves, yet bringing only minor improvements. In **(b)** we perform a sweep over _β_ 1 _∈{_ 0 _,_ 0 _._ 1 _,_ 0 _._ 3 _,_ 0 _._ 5 _,_ 0 _._ 7 _,_ 0 _._ 9 _,_ 0 _._ 99 _}_ . The value _β_ 1 = 0 _._ 0 is especially interesting as it allows us to remove entirely _**m**_ 1, therefore saving memory. We observe that smaller _β_ 1 values yield noisier curves, with multiple loss spikes. At the 110M parameter scale, those spike do not significantly impact the convergence, we conjecture that they could become a problem at larger scales. We also notice that slightly better losses could be obtained using smaller _β_ 1 values (e.g. _β_ 1 = 0 _._ 3), we keep _β_ 1 = 0 _._ 9 in all of our experiments for the ease of comparison with AdamW. 

30 

Arxiv preprint 

## C.1.5 IMPACT OF TRAINING FOR FEWER ITERATIONS 

**Sensitivity to the number of iterations.** As we rely on old gradients, on question that arises is whether AdEMAMix would still perform well when reducing the number of iterations. In this section we compare the loss obtained by 110M parameter models when halving the number of iterations and doubling the batch size, in such a way that the number of tokens consumed for training is always the same (17B). In Fig. 16, we observe that decreasing the number of iterations too much increases the final loss of the model. This effect is more pronounced for AdEMAMix. However, at both 32 _k_ and 64 _k_ iterations, AdEMAMix still outperforms AdamW. In Fig. 17 we observe that reducing _β_ 3 can mostly alleviate the problem. 

**==> picture [396 x 140] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4 AdamW 256 k steps, bs = 64 3 . 4 AdEMAMix 256 k steps, bs = 64 AdamW 17B<br>AdamW 128 k steps, bs = 128 AdEMAMix 128 k steps, bs = 128 AdEMAMix 17B<br>AdamWAdamW 6432 kk steps,steps, bsbs == 256512 AdEMAMixAdEMAMix 6432 kk steps,steps, bsbs == 256512 2 . 86<br>3 . 2 3 . 2<br>2 . 84<br>3 . 0 3 . 0<br>2 . 82<br>2 . 8 2 . 8<br>2 . 80<br>0 5B 10B 17B 0 5B 10B 17B 32 k 64 k 128 k 256 k<br>Tokens Tokens Number of training iterations<br>(a) AdamW with fewer steps. (b) AdEMAMix with fewer steps. (c) Comparing the final loss.<br>loss<br>Loss Loss<br>Final<br>**----- End of picture text -----**<br>


Figure 16: **Impact of using fewer iterations on AdEMAMix vs. AdamW.** We train multiple 110M parameter models. All models are trained on 17B tokens, but using different batch sizes ( _bs ∈ {_ 64 _,_ 128 _,_ 256 _,_ 512 _}_ ), which results in different numbers of iterations: _{_ 32 _k,_ 64 _k,_ 128 _k,_ 256 _k}_ . For both AdamW and AdEMAMix, we use hyperparameters from Table 2 ( _β_ 3 = 0 _._ 9999 for AdEMAMix). In **(a)** , we observe a drop in performance when reducing the number of steps, as the loss of the final iterates is higher for the model trained for 32 _k_ steps compared to the ones trained for more iterations. In **(b)** we observe a similar story for AdEMAMix models, with a slightly more pronounced performance degradation. Yet, as it can be seen in **(c)** , the gap between AdEMAMix and AdamW still remains, even when training for 32 _k_ iterations. Importantly, this experiment does not suggest that AdEMAMix is performing worse when using larger batches. In our ViT experiments, we successfully use a batch size of 4096. 

**==> picture [396 x 150] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 4 AdEMAMix 256 k steps, bs = 64 3 . 4 AdEMAMix 256 k steps, bs = 64 AdamW 17B<br>AdEMAMix 128 k steps, bs = 128 AdEMAMix 128 k steps, bs = 128 AdEMAMix 17B, β 3 = 0 . 9999<br>AdEMAMixAdEMAMix 6432 kk steps,steps, bsbs == 256512 AdEMAMixAdEMAMix 6432 kk steps,steps, bsbs == 256512 2 . 86 AdEMAMix 17B, β 3 = 0 . 999<br>3 . 2 3 . 2<br>2 . 84<br>3 . 0 3 . 0<br>2 . 82<br>2 . 8 2 . 8<br>2 . 80<br>0 5B 10B 17B 0 5B 10B 17B 32 k 64 k 128 k 256 k<br>Tokens Tokens Number of training iterations<br>(a) AdEMAMix with fewer steps (b) AdEMAMix with fewer steps (c) Comparing the final loss.<br>β 3 = 0 . 9999 (same as Fig. 16b). β 3 = 0 . 999.<br>loss<br>Loss Loss<br>Final<br>**----- End of picture text -----**<br>


Figure 17: **Reducing** _β_ 3 **can work better with low numbers of iterations.** We take the same experimental protocol as Fig. 16 but this time we compare training AdEMAMix models with _β_ = 0 _._ 999 and _β_ = 0 _._ 9999. In **(a,b)** we observe how using a smaller _β_ 3 = 0 _._ 999 solves—in a large part—the problem of decreasing performances when using a small number of iterations. In **(c)** , while final loss for _T ∈{_ 128 _k,_ 256 _k}_ is lower using _β_ 3 = 0 _._ 9999, using _β_ 3 = 0 _._ 999 is better for _T ∈{_ 32 _k,_ 64 _k}_ . Using _β_ 3 = 0 _._ 999, the gap between AdEMAMix and AdamW increases as the number of iterations decreases. 

31 

Arxiv preprint 

## C.1.6 LIMITATIONS OF A SINGLE EMA 

**Results on a 2D toy example.** A natural question that arises from our method is whether it is possible to obtain the same results without the additional EMA. In this section we aim to convince the reader that the two EMAs are required. First, we propose to study a small toy 2D optimization problem: 

**==> picture [370 x 21] intentionally omitted <==**

This function was introduced by Liu et al. (2023) as a function with sharp curvature along the _x_ -axis, and flatter curvature along the _y_ -axis. Initializing the first iterate _**x**_[(0)] = (0 _._ 3 _,_ 1 _._ 5), we start optimizing with (i) Adam with a large _β_ 1 = 0 _._ 999, and (ii) AdEMAMix with _β_ 1 = 0 _._ 9 and _β_ 3 = 0 _._ 999. In both cases _β_ 2 = 0 _._ 999. To make the experiment interesting, we initialize the EMA buffers for both methods to either ( _−_ 3 _,_ 0) or ( _−_ 0 _._ 8 _, −_ 3). This has for effect to give an initial "speed" to the first iterate. As a result of this speed, Adam with a large _β_ 1 is unable to correct his trajectory. In contrast, using two EMAs, one with a small, and one with a large _β_ —as in AdEMAMix—converges to the solution. The fast changing EMA can correct for the bias introduced by the slow changing EMA. See Fig. 18 for results. 

**==> picture [342 x 166] intentionally omitted <==**

**----- Start of picture text -----**<br>
Adam β 1 = 0 . 999 Adam β 1 = 0 . 999<br>AdEMAMix β 1 = 0 . 9 , β 3 = 0 . 999 AdEMAMix β 1 = 0 . 9 , β 3 = 0 . 999<br>ly<br>start start<br>(a)  m [(0)] 1 =  m [(0)] 2 = [ − 3 ,  0]. (b)  m [(0)] 1 =  m [(0)] 2 = [ − 0 . 8 , − 3].<br>**----- End of picture text -----**<br>


Figure 18: **Showing the necessity of a second momentum term on a toy example.** In this experiment, we use Adam and AdEMAMix to optimize a simple function (same function as in Fig.2 in Liu et al. (2023), _⋆_ is a local optimum). We want to understand what is preventing the use of large _β_ 1 values in Adam, and how AdEMAMix is overcoming this problem. Starting from _**x**_[(0)] = [0 _._ 3 _, −_ 1 _._ 5], we initialize _**m**_ 1 (resp. _**m**_ 1 and _**m**_ 2) for Adam (resp. AdEMAMix) to either [ _−_ 3 _,_ 0] or [ _−_ 0 _._ 8 _, −_ 3]—shown with black arrows (not to scale). _**ν**_[(0)] is still initialized to **0** . This is similar to giving an initial speed to the iterate. In both cases **(a)** and **(b)** , the iterates following the Adam trajectory fail to converge in the given number of iterations. This is due to the large _β_ 1 which requires many hundreds of iterations to significantly alter _**m**_ 1. As a result, the iterates mostly follow the initial direction imposed by _**m**_[(0)] 1 ( _**ν**_ is still used to scale _**m**_ 1). In contrast, AdEMAMix converges to the solution. The fast (smaller _β_ ) EMA of AdEMAMix corrects the trajectory of the slow (high _β_ ) EMA. Interestingly, AdEMAMix initially overshoots the optimum. It has to wait for the influence of _**m**_[(0)] 2 to fade away before finally converging to the solution. While this experiment does not explain why a larger momentum would be beneficial, it illustrates how using two EMAs can enable the use of large _β_ values. A better intuition behind why a large momentum could be beneficial is presented in Fig. 2. 

**Results training** 110 **M parameters LMs.** To further demonstrate that simply increasing _β_ 1 in AdamW does not provide nearly the same gains as AdEMAMix, we run several additional experiments. In Fig. 19 we show what we obtain when training from scratch using a single EMA with a _β_ 1 _∈ {_ 0 _._ 99 _,_ 0 _._ 999 _,_ 0 _._ 9999 _,_ 0 _._ 99999 _}_ . Naively increasing _β_ 1 in AdamW does not work; adding a scheduler on _β_ 1 to smoothly increase its value during the entirety of the training also fails. In Fig. 20, we show results when increasing _β_ 1 in the middle of training, with and without scheduler on _β_ 1. This differs from the previous setting as we bypass the initial training phase capable of causing instabilities (see 

32 

Arxiv preprint 

§ 3), as well as bypass the initial iterations during which the bias correction done by AdamW can have an impact. Here again we observe increasing the _β_ 1 value does not provide any noticeable gain. 

**==> picture [324 x 122] intentionally omitted <==**

**----- Start of picture text -----**<br>
6 . 0 3 . 4<br>AdamW β 1 = 0 . 9 AdamW β 1 = 0 . 9, β 2 = 0 . 999<br>5 . 5 AdamWAdamW ββ 11 == 00 .. 99999 3 . 3 AdamWAdamW ββ 11 == 00 .. 99999,, ββ 2 2==00 . 999 . 999<br>AdamW β 1 = 0 . 9999 AdamW β 1 = 0 . 9999, β 2 = 0 . 999<br>5 . 0 AdamW β 1 = 0 . 99999 3 . 2 AdamW β 1 = 0 . 99995, β 2 = 0 . 999<br>AdamW β 1 = 0 . 99999, β 2 = 0 . 999<br>4 . 5 3 . 1 AdamW β 1 = 0 . 99999, β 2 = 0 . 9999<br>4 . 0<br>3 . 0<br>3 . 5<br>2 . 9<br>3 . 0<br>2 . 8<br>0 50000 100000 150000 200000 250000 0 50000 100000 150000 200000 250000<br>Iterations Iterations<br>(a) Vanilla AdamW. (b) AdamW with  β 1-scheduler.<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 19: **Increasing** _β_ 1 **in AdamW.** In **(a)** we perform a simple sweep over the _β_ 1 hyperparameter of AdamW, we observe how the optimizer is unable to cope with large values of _β_ 1, diverging for _β_ 1 _>_ 0 _._ 999. In **(b)** we modify AdamW to incorporate the tricks we used in AdEMAMix, i.e. we add a scheduler on _β_ 1. While this prevents divergence, the final loss is getting worse as we increase _β_ 1. We had to increase _β_ 2 to stabilise the training when using _β_ 1 = 0 _._ 99999. Those experiments support the importance of the additional EMA in AdEMAMix. 

**==> picture [324 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
AdamW β 1 = 0 . 9 from scratch AdamW β 1 = 0 . 9 from scratch<br>3 . 2 AdamW β 1 = 0 . 99 3 . 2 AdamW β 1 = 0 . 99, Tβ 1 = 200 k<br>AdamW β 1 = 0 . 999 AdamW β 1 = 0 . 999, Tβ 1 = 200 k<br>3 . 1 AdamW AdamW β β 11 = = 0 0 . . 9999 99999 3 . 1 AdamWAdamW ββ 11 == 00 .. 9999,99999, TTβ 1 β 1==200200 kk<br>3 . 0 3 . 0<br>2 . 9 2 . 9<br>2 . 8 2 . 8<br>200000 250000 300000 350000 400000 450000 500000 200000 250000 300000 350000 400000 450000 500000<br>Iterations Iterations<br>(a) Increasing  β 1 without warmup. (b) Increasing  β 1 with warmup.<br>Loss Loss<br>**----- End of picture text -----**<br>


Figure 20: **Increasing** _β_ 1 **in AdamW later during training.** In **(a)** we load an AdamW checkpoint and resume training using AdamW with a larger _β_ 1. We observe no gain over the baseline from increasing the value of _β_ 1. In **(b)** , we add a scheduler over _β_ 1 (similar to the scheduler over _β_ 3 for AdEMAMix) to smooth the transition between _β_ 1 = 0 _._ 9 and larger values. We set _Tβ_ 1 = 200 _k_ iterations. Despite the smoother transition, no gain over the baseline is observed. 

33 

Arxiv preprint 

## C.1.7 REMOVING _m_ 1 BY SETTING _β_ 1 = 0 

**Effect of the batch size when** _β_ 1 = 0 **.** In this section we investigate the impact of setting _β_ 1 = 0, which allows to replace _**m**_[(] 1 _[t]_[)] by the current gradient _**g**_[(] _[t]_[)] , saving memory. In this scenario the memory complexity of AdEMAMix is the same as the one of AdamW. We noticed in Fig. 15b how reducing _β_ 1 can make the training slightly more unstable, triggering some loss spikes. In Fig. 21 we study the impact of the batch size on those instability, revealing that _β_ 1 = 0 is less stable when using larger batch sizes. However, despite the spikes, the final loss for _β_ 1 = 0 and _β_ 1 = 0 _._ 9 are very similar— _β_ 1 = 0 _._ 9 yielding slightly better results for smaller batch sizes. 

**==> picture [358 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 8 AdEMAMixAdEMAMix batch-size=batch-size= 44 , β, β 11 == 00 . 9<br>AdEMAMix batch-size= 8 , β 1 = 0<br>3 . 6 AdEMAMix batch-size= 8 , β 1 = 0 . 9<br>AdEMAMix batch-size= 16 , β 1 = 0<br>AdEMAMix batch-size= 16 , β 1 = 0 . 9<br>3 . 4 AdEMAMix batch-size= 32 , β 1 = 0<br>AdEMAMix batch-size= 32 , β 1 = 0 . 9<br>AdEMAMix batch-size= 64 , β 1 = 0<br>3 . 2 AdEMAMix batch-size= 64 , β 1 = 0 . 9<br>AdEMAMix batch-size= 128 , β 1 = 0<br>3 . 0 AdEMAMix batch-size= 128 , β 1 = 0 . 9<br>2 . 8<br>0 128 k 256 k<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 21: **Effect of batch size with** _β_ 1 = 0 **.** For a 110M parameter model, we use _β_ 1 _∈{_ 0 _._ 0 _,_ 0 _._ 9 _}_ and vary the batch-size _∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ . Lighter curves use _β_ 1 = 0, darker curves are their _β_ 1 = 0 _._ 9 counterpart. For small batch sizes, no instabilities are observed, and _β_ 1 = 0 _._ 9 reaches slightly better performances. For larger batch sizes, instabilities are observed with using _β_ 1 = 0, yet the final loss values are similar as the ones obtained with _β_ 1 = 0 _._ 9. This indicates that one can potentially save memory without compromising results. 

**Setting** _β_ 1 = 0 **for** 1 _._ 3 **B parameter models.** In this section, using a 1 _._ 3B parameter model, we re-do the same experiment as in Fig. 1, but we set _β_ 1 = 0. In Fig. 22, we show the results of this experiment. 

**==> picture [238 x 136] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 . 8 AdamW 101B tokens<br>AdEMAMix 101B tokens β 1 = 0 . 9 , β 3 = 0 . 9999<br>2 . 7 AdEMAMix 101B tokens β 1 = 0 . 0 , β 3 = 0 . 9999<br>2 . 6<br>2 . 5<br>2 . 4<br>2 . 3<br>0 200 k 500 k 770 k<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 22: **Using** _β_ 1 = 0 **with a** 1 _._ 3 **B parameter model.** We use _β_ 2 = 0 _._ 999 for all those experiments. We observe how it is possible to converge despite using _β_ 1 = 0. The final loss with _β_ 1 = 0 is slightly better than the one with _β_ 1 = 0 _._ 9. 

34 

Arxiv preprint 

## C.1.8 COMPARING _m_ 1 + _αm_ 2 VERSUS (1 _− α_ ) _m_ 1 + _αm_ 2 

In the equation AdEMAMix, we combine _**m**_ 1 and _**m**_ 2 using _**m**_ 1 + _α_ _**m**_ 2. In this section we contrast this to using a convex combination (1 _− α_ ) _**m**_ 1 + _α_ _**m**_ 2, with _α ∈_ [0 _,_ 1]. 

**Are those two somewhat equivalent?** Given the combination _η_ ( _**m**_ 1 + _α_ _**m**_ 2), it is possible to rewrite it as a convex combination as such: 

**==> picture [326 x 20] intentionally omitted <==**

Therefore, the two formulas are equivalent up-to a reparametrization. However, in practice, we use schedulers on _η_ and _α_ . While at any timestep _t_ , it is possible to reparametrize one formula into the other, there is not one single reparametrization that holds for all timesteps. Assuming that we do not change the nature of the schedulers for _α_ and _η_ , those two formulations are therefore _not equivalent_ . In Fig. 23 we show the evolution of the weights given to _**m**_ 1 and _**m**_ 2 for the two approaches. We assume a cosine scheduler for _η_ and a linear scheduler for _α_ . We can see how the two formulations give very different weights to _**m**_ 1 and _**m**_ 2. 

**==> picture [256 x 121] intentionally omitted <==**

**----- Start of picture text -----**<br>
× 10 [−] [3] × 10 [−] [3]<br>1 . 0<br>2 weight for m 1: η [(] [t] [)] (1  − α [(] [t] [)] )<br>weight for m 2: η [(] [t] [)] α [(] [t] [)]<br>weight for m 1: η [(] [t] [)]<br>1 weight for m 2: η [(] [t] [)] α [(] [t] [)] 0 . 5<br>0 0 . 0<br>0 100000 200000 0 100000 200000<br>t t<br>(a)  η [(] [t] [)] ( m 1 +  α [(] [t] [)] m 2). (b)  η [(] [t] [)] ((1 −α [(] [t] [)] ) m 1+ α [(] [t] [)] m 2).<br>**----- End of picture text -----**<br>


Figure 23: **Weights given to** _**m**_ 1 **and** _**m**_ 2 **for the two formulations.** In **(a)** , we show the weights given to _**m**_ 1 and _**m**_ 2 when we use _η_[(] _[t]_[)] ( _**m**_ 1 + _α_[(] _[t]_[)] _**m**_ 2). We use _α_ = 8 ( _α_ is the final value reached by the scheduler _α_[(] _[t]_[)] ). In **(b)** , we show the weights given to _**m**_ 1 and _**m**_ 2 when we use _η_[(] _[t]_[)] ((1 _− α_[(] _[t]_[)] ) _**m**_ 1 + _α_[(] _[t]_[)] _**m**_ 2). We use _α_ = 0 _._ 9. Importantly, in both cases, we assume a cosine scheduler for _η_[(] _[t]_[)] and a linear scheduler for _α_[(] _[t]_[)] . Given this choice of schedulers, there is no reparametrization which can make those two formulation equal. In Fig. 24 we show that using the convex combination formulation yields worse results. 

**Is the convex combination better?** Now that we established that the two formulations are different, which one is better? To answer this we run two grid searches, varying _η ∈ {_ 0 _._ 4 _,_ 0 _._ 6 _,_ 0 _._ 8 _,_ 1 _,_ 1 _._ 2 _,_ 1 _._ 4 _,_ 1 _._ 6 _}_ and _α ∈{_ 2 _,_ 4 _,_ 6 _,_ 8 _,_ 10 _,_ 12 _,_ 14 _}_ for the formulation in equation AdEMAMix, and _α ∈{_ 0 _._ 3 _,_ 0 _._ 4 _,_ 0 _._ 5 _,_ 0 _._ 6 _,_ 0 _._ 7 _,_ 0 _._ 8 _,_ 0 _._ 9 _}_ for the convex combination formulation. Results in Fig. 24 show how the formulation used in AdEMAMix provides a better loss _for all the pairs_ ( _η, α_ ) _tested_ , also outperforming AdamW for all of those. In contrast, for the convex combination, while most ( _η, α_ ) pairs outperform AdamW, the improvement is smaller. 

**Getting the same results by changing the schedulers for the convex combination.** We concluded before that—unless we tamper with the schedulers—it is not possible to find a reparametrization that would work for all timesteps and make the two formulations equivalent. What if we could change the schedulers? Which schedulers would we need to use with the convex formulation, to get the same results as with the other formulations. The answer is simple, let _η_[(] _[t]_[)] and _α_[(] _[t]_[)] denote the cosine and linear schedulers for resp. _η_ and _α_ . The convex combination would need to use the following schedulers: 

**==> picture [178 x 25] intentionally omitted <==**

Therefore, to get equivalent results as for the original AdEMAMix formulation, the learning rate scheduler would need to be _η_ ˆ[(] _[t]_[)] , which is not a cosine scheduler. The _α_ scheduler, _α_ ˆ[(] _[t]_[)] , would not be linear. We plot those schedulers in Fig. 25. 

35 

Arxiv preprint 

**==> picture [314 x 467] intentionally omitted <==**

**----- Start of picture text -----**<br>
14 0 . 9<br>12 3 . 07 0 . 8 3 . 07<br>10 0 . 7<br>3 . 06 3 . 06<br>8 0 . 6<br>6 3 . 05 0 . 5 3 . 05<br>4 0 . 4<br>3 . 04 3 . 04<br>2 0 . 3<br>I<br>0 . 4 0 . 6 0 . 8 1 1 . 2 1 . 4 1 . 6 0 . 4 0 . 6 0 . 8 1 1 . 2 1 . 4 1 . 6<br>η ×  10 [3] η ×  10 [3]<br>(a)  m 1 +  α m 2. (b) (1  − α ) m 1 +  α m 2.<br>η - α  hyperparameter sweep for the two formulations. For pairs of ( ( η, α ), we train small<br>-layers) transformer LMs on RedPajama v2 data.<br>and is normalized between the two plots.<br> η  to fine the value giving the best results for AdamW. We use a green level curve to show<br>Points in the interior of the green level curve represents<br> (a) , we color the entire frame in green as all the pairs of<br>hyperparameters are outperforming AdamW. It is clear from those two figures that using<br>yields better results, more consistently, compared to using (1 (1  − α ) m 1 + +  α m 2..<br>× 10 [−] [3]<br>7 . 5 α [(] [t] [)]<br>2 α ˆ [(] [t] [)] = αα [(] [t] [(][)] [t] +1 [)]<br>5 . 0<br>η [(] [t] [)]<br>η ˆ [(] [t] [)] = η [(] [t] [)] ( α [(] [t] [)] + 1)<br>1 2 . 5<br>0 0 . 0<br>0 100000 200000 0 100000 200000<br>t t<br>(a)  η  schedulers. (b)  α  schedulers.<br>AdamW<br>AdamW<br>α α<br>**----- End of picture text -----**<br>


Figure 24: _η_ **-** _α_ **hyperparameter sweep for the two formulations.** For pairs of ( ( _η, α_ ), we train small (6-layers) transformer LMs on RedPajama v2 data. We plot the results for the two formulations, the color represents the final validation loss, and is normalized between the two plots. We also do a sweep over _η_ to fine the value giving the best results for AdamW. We use a green level curve to show the performance of the best AdamW model. Points in the interior of the green level curve represents configurations outperforming AdamW. In **(a)** , we color the entire frame in green as all the pairs of hyperparameters are outperforming AdamW. It is clear from those two figures that using _**m**_ 1 + _α_ _**m**_ 2 yields better results, more consistently, compared to using (1 (1 _− α_ ) _**m**_ 1 + + _α_ _**m**_ 2.. 

Figure 25: **Using schedulers** _η_ ˆ[(] _[t]_[)] **and** _α_ ˆ[(] _[t]_[)] **with the convex combination formulation is the same as using the original AdEMAMix.** The original AdEMAMix forumulation uses a cosine and linear schedulers for resp. _η_ and _α_ . Those schedulers are in blue. For the weights given to _**m**_ 1 and _**m**_ 2 to be the same when using the convex combination formulation, one need to use different schedulers _η_ ˆ[(] _[t]_[)] and _α_ ˆ[(] _[t]_[)] (in orange). Using the original AdEMAMix formulation allows to use simpler schedulers. 

36 

Arxiv preprint 

## C.1.9 MISCELLANEOUS RESULTS 

**Impact of** _β_ 3 **and** _α_ **schedulers when starting AdEMAMix from AdamW.** As mentioned in § 3, we do not necessarily need the _α_ and _β_ 3 schedulers when resuming training from a sufficiently late checkpoint. Fig. 5b and Fig. 5c do not use schedulers, unless we start using AdEMAMix from scratch. In Fig. 26, we vary the warmup periods for _α_ and _β_ 3 when starting AdEMAMix from an AdamW checkpoint at _tswitch_ = 300 _k_ . We observe how increasing _Tβ_ 3 = _Tα_ only slows down the convergence. This serves to illustrate that the schedulers for _α_ and _β_ 3 are only required to stabilise the optimization during the early training phase. 

**==> picture [199 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
AdamW<br>3 . 2 AdEMAMix β 3 = 0 . 9999, α = 2 . 0, Tα = Tβ 3 = 0<br>AdEMAMix β 3 = 0 . 9999, α = 2 . 0, Tα = Tβ 3 = 25 k<br>3 . 1 AdEMAMix β 3 = 0 . 9999, α = 2 . 0, T α = T β 3 = 50 k<br>AdEMAMix β 3 = 0 . 9999, α = 2 . 0, Tα = Tβ 3 = 100 k<br>3 . 0<br>2 . 9<br>2 . 8<br>200000 250000 300000 350000 400000 450000 500000<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 26: **No need to use schedulers when switching from AdamW to AdEMAMix sufficiently late.** For a 110M parameter model, we switch from AdamW to AdEMAMix after _tswitch_ = 300 _k_ iterations. We vary _Tβ_ 3 = _Tα ∈{_ 0 _,_ 25 _k,_ 50 _k,_ 100 _k}_ , the other prameters are fixed: _β_ 3 = 0 _._ 9999 _, α_ = 2, the learning rate is inherited from AdamW and stays on his decaying course following a cosine decay. AdEMAMix’s _**m**_ 2 is initialized to **0** . The larger _Tβ_ 3 = _Tα_ , the slower the convergence is, the final loss value are very similar. This shows that the schedulers for _α_ and _β_ 3 are not important when starting AdEMAMix from a sufficiently late checkpoint. 

**AdEMAMix from scratch without schedulers.** In this section we provide more justification for the use of schedulers on _α_ and _β_ 3. We reveal that, without the use of schedulers, the norms of the updates increases significantly, even for small _α_ values. Unstable and large weight updates are known to occur in the early phases of training when learning rate warmup is not used (Gotmare et al., 2019). Using large momentum values too early seem to increase this phenomenon. See Fig. 27. 

**==> picture [340 x 127] intentionally omitted <==**

**----- Start of picture text -----**<br>
6 . 0<br>AdamW<br>AdEMAMix α = 0 . 9, β 3 = 0 . 9999, Tα,β 3 = 0 5 . 5<br>10 [4] AdEMAMix α = 0 . 6, β 3 = 0 . 9999, Tα,β 3 = 0<br>10 [3] AdEMAMixAdEMAMix αα == 00 .. 3,1, ββ 33 == 00 .. 9999,9999, TTα,βα,β 33 == 00 54 .. 05 AdamWAdEMAMixAdEMAMix αα == 00 .. 9,6, ββ 33 == 00 .. 9999,9999, TTα,βα,β 33 == 00<br>4 . 0 AdEMAMixAdEMAMix αα == 00 .. 3,1, ββ 33 == 00 .. 9999,9999, TTα,βα,β 33 == 00<br>10 [2] 3 . 5<br>3 . 0<br>0 50000 100000 150000 200000 250000 0 50000 100000 150000 200000 250000<br>Iterations Iterations<br>(a) Norm of 1000 consecutive updates. (b) Loss.<br>∆ θ<br>of Loss<br>Norm<br>**----- End of picture text -----**<br>


Figure 27: **The norm of the updates is exploding without schedulers.** For historical reasons, those experiments were done with a 110M parameter model using a batch size of 32, and without weight decay ( _λ_ = 0). We use _β_ 2 = 0 _._ 999 and _β_ 1 = 0 _._ 9. We use 3000 steps of learning rate warmup for all the experiments. In **(a)** , we monitor the norm of 1000 consecutive updates: _∥_ ∆ _**θ**_[(] _[t]_[)] _∥_ 2 = _∥_ _**θ**_[(] _[t]_[+1000)] _−_ _**θ**_[(] _[t]_[)] _∥_ 2. Even when using small values of _α_ , the norm of ∆ _**θ**_[(] _[t]_[)] is increasing significantly at the beginning of training. As a result, in **(b)** , we observe how—without the use of a scheduler on _α_ and _β_ 3—AdEMAMix models are either diverging or fail to recover and improve upon AdamW. 

37 

Arxiv preprint 

**Impact of the linear decay duration when using a constant** _η_ **-scheduler.** In Fig. 3b we show results using AdEMAMix with a linear warmup _→_ constant _→_ linear decay learning rate scheduler. We used 100 _k_ of linear decay. In this section we experiment with 200 _k_ steps of linear decay, the rest of the parameters are the same: we use a 1 _._ 3B parameter model with a max learning rate of 10 _[−]_[4] and remaining hyperparameters as in Table 4. Results can be seen in Fig. 28. 

**==> picture [199 x 139] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 . 8<br>AdamW<br>AdEMAMix β 3 = 0 . 9999<br>2 . 7 AdEMAMix β 3 = 0 . 99999<br>200 k steps of linear η decay<br>2 . 6<br>2 . 5<br>2 . 4<br>2 . 3<br>2 . 2<br>0 500 k =  Tα,β 3 900 k 1 . 1M<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 28: **Testing a different decay duration compared to Fig. 3b.** Increasing the length of the learning rate decay improves the results for each methods. The gap between AdEMAMix using _β_ 3 = 0 _._ 99999 and _β_ 3 = 0 _._ 9999 is shrinking, but the gap between AdEMAMix models and AdamW remains. 

**Same figure as Fig. 1, including the AdamW trained on** 197 **B tokens.** In Fig. 1 we represent the AdamW experiment trained on 197B tokens by a blue horizontal line for aesthetic reasons. In Fig. 29 we include this missing curve to the same plot. 

**==> picture [298 x 144] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 . 8 AdamW trained on<br>{ 34B ,  101B ,  131B ,  197B } tokens<br>2 . 7 AdEMAMix { 34B ,  101B ,  131Btrained } tokenson<br>2 . 6<br>2 . 5<br>2 . 4<br>2 . 3<br>AdamW 197B<br>0 256 k 770 k 1M 1 . 5M<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 29: **Same as Fig. 1 adding AdamW** 197 **B.** 

38 

Arxiv preprint 

## C.2 VIT EXPERIMENTS 

**Top-k accuracies.** In Fig. 6, we plot the test and train loss for the final iterates. In Fig. 30, we give a more detailed view by reporting the evolution of the training loss, test loss, and top-1 accuracy. Looking at the first row, the training loss for AdEMAMix is systematically better than the AdamW baseline. The second row shows that in cases where the test loss correlates well with the train loss, AdEMAMix works well. The top-1 accuracy carries the same message. 

**==> picture [397 x 327] intentionally omitted <==**

**----- Start of picture text -----**<br>
3 . 5 3 . 5 4<br>AdamW AdamW AdamW<br>AdEMAMix β 3 = 0 . 999 AdEMAMix β 3 = 0 . 999 AdEMAMix β 3 = 0 . 999<br>AdEMAMix β 3 = 0 . 9999 3 . 0 AdEMAMix β 3 = 0 . 9999 3 AdEMAMix β 3 = 0 . 9999<br>3 . 0<br>2 . 5 2<br>2 . 5<br>2 . 0<br>1<br>2 . 0 1 . 5<br>0<br>0 50000 100000 0 50000 100000 0 50000 100000<br>3 . 50 3 . 5 4<br>3 . 25<br>3 . 0 3<br>3 . 00<br>2 . 75<br>2 . 5 2<br>2 . 50<br>2 . 25 2 . 0 1<br>0 50000 100000 0 50000 100000 0 50000 100000<br>45 50 80<br>70<br>40 45<br>60<br>35 40<br>50<br>30 35 40<br>0 50000 100000 0 50000 100000 0 50000 100000<br>Iterations Iterations Iterations<br>(a) 24M params, 11M images. (b) 86M params, 11M images. (c) 86M params, 1 . 3M images.<br>(ImageNet-21k) (ImageNet-21k) (ImageNet-1k)<br>Loss Loss Loss<br>Train Train Train<br>Loss Loss Loss<br>Test Test Test<br>Accuracy Accuracy Accuracy<br>Top-1 Top-1 Top-1<br>**----- End of picture text -----**<br>


Figure 30: **AdEMAMix & capacity/data ratio.** These figures complement Fig. 6 and provide the top-1 accuracy as well as the training and test losses for the three experimental setups used in our ViT experiments (see § 4.3). From left to right, the ratio between data and model capacity worsen. In **(a)** , we train a 24M parameter model on a dataset of 11M images (Imagenet-21k), doing a total of 37 epochs. In this setting the training and test loss curves are very similar, signaling no blatant overfitting. We observe AdEMAMix trivially outperforms the AdamW baseline. As the ratio between data and model capacity worsen, in **(b)** (more capacity) and **(c)** (more capacity and less data), it becomes more and more difficult to find hyperparameters outperforming the baseline. Yet, across all those settings, the training loss for AdEMAMix is always lower than that of the AdamW baseline. When decreasing training loss correlates well with decreasing test loss, AdEMAMix can be expected to work well. 

39 

Arxiv preprint 

## C.3 COMPARISON WITH OTHER METHODS 

## C.3.1 COMPARISON WITH ADMETA AND DEMA 

**Double Exponential Moving Average (DEMA).** Originally introduced by Mulloy (1994), a DEMA originally aimed to emphasize the weight of recent asset price fluctuations, making the DEMA indicator more reactive to changes compared to simple EMAs. Given notations introduced in the main paper, let EMA[(] _β[T][ −][N]_[:] _[T]_[ )] ≜ EMA( _β,_ _**g**_[(] _[T][ −][N]_[)] _, . . . ,_ _**g**_[(] _[T]_[ )] ), let _N_ be the window size representing how many consecutive values are considered in the average, the formula for DEMA can be written as follows: 

**==> picture [378 x 15] intentionally omitted <==**

If a simple EMA tends to give a significant weight to more recent observations, a DEMA emphasizes this behaviour by removing some of the weight given to older observations. This is not what we suggest doing in this work, we want both high sensitivity to recent observations and non-negligible weights given to older observations. 

**AdMeta.** Chen et al. (2023b) take inspiration over DEMA and use nested EMAs in their AdMeta-S optimizer: 

**==> picture [277 x 57] intentionally omitted <==**

With _µ_ and _κ_ parameterized by _β_ 1 _∈_ [0 _,_ 1[ as such: 

**==> picture [94 x 49] intentionally omitted <==**

In their AdMeta-S experiments, they use _β_ 1 = 0 _._ 9, corresponding to ( _µ, κ_ ) = (4 _._ 88 _,_ 2 _._ 11). _β_ 2 takes values ranging from 0 _._ 1 to 0 _._ 4. As a results, unlike AdEMAMix, AdMeta is not leveraging very old gradients. Analysing the AdMeta algorithm, we see that it consists in two nested EMAs. We show the shape of nested EMAs in Fig. 31. In sharp contrast with our approach we observe that (i) it reduces the weights given to recent gradients and (ii) it gives a small weight to old gradients. 

**==> picture [284 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 . 08 NestedSum of EMAs:two EMAs: β 1 = β 01 . 9= , β 02 . 9= , β 02 . 4= 0 . 999 10 [−] [1] NestedSum of EMAs:two EMAs: β 1 = β 01 . 9= , β 02 . 9= , β 02 . 4= 0 . 999<br>10 [−] [3]<br>0 . 06<br>10 [−] [5]<br>0 . 04<br>10 [−] [7]<br>0 . 02<br>10 [−] [9]<br>0 . 00<br>0 50 100 150 200 0 50 100 150 200<br>timesteps t timesteps t<br>(a) Linear scale. (b) Logarithmic scale.<br>timestep timestep<br>each each<br>to to<br>given given<br>Weight Weight<br>**----- End of picture text -----**<br>


Figure 31: **Comparison between nested EMAs and a linear combination of EMAs.** In **(a)** we observe how two nested EMAs actually decrease the influence of recent timesteps. This is the justification for the DEMA method, which aims to increase the sensitivity to recent gradients by subtracting nested EMAs. In **(b)** , using a log-scale for the _y_ -axis, we see how, for the values used in AdMeta, the older gradients receive a negligible weight. In our work, we show how very old gradients can be leveraged to get better results by keeping a high sensitivity to recent gradients while giving non-negligible weights to older gradients. 

40 

Arxiv preprint 

## C.3.2 COMPARISON WITH LION 

**The Lion optimizer.** Chen et al. (2023a) derived the following optimizer. We change notations to facilitate the comparison with AdEMAMix: 

**==> picture [336 x 26] intentionally omitted <==**

The Lion optimizer uses a sign function and updates its EMA after updating the parameters. Moreover, it does not normalize the updates. While Lion and AdEMAMix are quite different from each others, we can draw one similarity. Indeed, the interpolation _α_ _**m**_[(] _[t][−]_[1)] + (1 _− α_ ) _**g**_[(] _[t]_[)] is similar to combining two EMAs, one of them using _β_ = 0. We also explore the possibility of setting AdEMAMix’s _β_ 1 to 0 in § 3, we show in App. C.1.7 and App. C.1.4 (Fig. 15b) that this often works despite a more unstable training. Interestingly, Chen et al. (2023a) find that larger _β_ = 0 _._ 99 values work best. Beside this similarity, the two optimizers behave very differently, the biggest difference being the use of the sign function, which Chen et al. (2023a) claim can help regularize the training. We test the Lion optimizer on language modeling. To tune the hyperparameters, we took values from Chen et al. (2023a) as a starting point, as well as recipes provided by the Optax Jax library. A summary of our hyperparameter tuning is in Table 8: 

Table 8: **Hyperparameter tuning for Lion** 110 **M parameter models.** When multiple values are given, we bold the parameters we found to give the best results. 

|Hyperparameter|Value|
|---|---|
|||
|Learning rate_η_<br>Number of warmup steps<br>Sequence length<br>Weight decay_λ_<br>Learning rate decay scheduler<br>Batch size<br>Gradient clipping|0_._00005_,_0_._0001_,_0_._0002_,_**0**_._**0004**_,_0_._0006_,_0_._0008_,_0_._001<br>3000<br>1024<br>0_._01_,_0_._125_,_0_._166_,_**0**_._**25**_,_0_._5_,_1_._0<br>cosine-decay<br>64<br>1_._0_,_**0**_._**5**|
|||
|Lion_α_<br>Lion_β_|0_._5_,_0_._7_,_**0**_._**9**<br>**0**_._**99**_,_0_._9999|



The training curve associated to the best hyperparameters is in Fig. 32. We observe that Lion is not outperforming our carefully tuned AdamW baseline. Moreover, no attempt to increase _β_ beyond 0 _._ 99 was successful, as those models mostly diverged. This emphasizes one of our main contributions: the introduction of schedulers enabling the use of large momentum values. 

**==> picture [199 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
AdamW<br>3 . 6 Best Lion model<br>3 . 4<br>3 . 2<br>3 . 0<br>2 . 8<br>0 50000 100000 150000 200000 250000<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 32: **Comparing AdamW with the best Lion model found.** Hyperparameters used are in Table 8. We train many Lion models and select the one with the lowest final validation loss. Lion is not outperforming our carefully tuned AdamW baseline. 

41 

Arxiv preprint 

## C.3.3 ADDING A THIRD MOMENTUM TERM ( _∼_ AGGMO) 

Lucas et al. (2019, AggMo) propose to add an arbitrary number ( _K_ ) of momentum terms to the gradient descent algorithm: 

**==> picture [292 x 30] intentionally omitted <==**

In order to see whether more momentum terms can be beneficial, we modify our AdEMAMix optimizer to include a third momentum _**m**_ 3. We name the resulting optimizer Ad **3** EMAMix: 

**==> picture [348 x 84] intentionally omitted <==**

We Train a 110M parameter model with same hyperparameters as in Table 2, but we use _α_ = 4 instead of _α_ = 8. We apply the same scheduler to _β_ 4 as for _β_ 3. In Fig. 33 we show the resulting validation loss curves for various ( _β_ 1 _, β_ 3 _, β_ 4) triplets. Our experiments show no advantage of adding an extra _β_ 4-EMA. Rather than carrying the message that we should use more momentum terms, our work shows how—in order to use a large momentum—a term should be added to stay sensitive to local fluctuations of the loss landscape. In that sense, we believe two terms should be enough. 

**==> picture [278 x 160] intentionally omitted <==**

**----- Start of picture text -----**<br>
AdamW<br>3 . 3 AdEMAMix<br>Ad3EMAMix β 1 , β 3 , β 4 = 0 . 9 ,  0 . 9999 ,  0 . 99999<br>Ad3EMAMix β 1 , β 3 , β 4 = 0 . 9 ,  0 . 999 ,  0 . 99999<br>3 . 2 Ad3EMAMix β 1 , β 3 , β 4 = 0 . 9 ,  0 . 999 ,  0 . 9999<br>Ad3EMAMix β 1 , β 3 , β 4 = 0 . 9 ,  0 . 99 ,  0 . 9999<br>Ad3EMAMix β 1 , β 3 , β 4 = 0 . 9 ,  0 . 99 ,  0 . 99999<br>3 . 1<br>3 . 0<br>2 . 9<br>2 . 8<br>0 128 k 256 k<br>Iterations<br>Loss<br>**----- End of picture text -----**<br>


Figure 33: **Adding an extra momentum to AdEMAMix does not provide better results.** We train multiple 110M parameter Ad3EMAMix models with various ( _β_ 1 _, β_ 3 _, β_ 4) triplets (lighter curves). None of those models end up outperforming AdEMAMix, indicating that the additional _**m**_ 3 momentum term is seemingly not bringing any advantage. 

42 

