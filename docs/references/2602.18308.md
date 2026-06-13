# JPmHC: Dynamical Isometry via Orthogonal Hyper-Connections

- **Authors:** Biswa Sengupta, Jinhua Wang, Leo Brunswic (JP Morgan Chase LLM Suite Team)
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2602.18308
- **MORPH uses:** Default Hyper-Connection residual (`residual_mode: hc_cayley`). Widens the residual stream to n parallel C-dim streams with input-dependent H^pre/H^post/H^res mappings; constrains H^res to the Stiefel manifold via Cayley transform of a skew-symmetric matrix, giving exact dynamical isometry (all singular values = 1) without iterative Sinkhorn normalisation. MORPH's fused Triton kernel implements the Cayley fixed-point projection from this paper.

---

# JPmHC Dynamical Isometry via Orthogonal Hyper-Connections 

Biswa Sengupta LLM Suite Team, JP Morgan Chase & Co. `biswa.sengupta@jpmorgan.com` 

Jinhua Wang LLM Suite Team, JP Morgan Chase & Co. `jinhua.wang@jpmorgan.com` 

Leo Brunswic LLM Suite Team, JP Morgan Chase & Co. `leo.brunswic@jpmorgan.com` 

February 2026 

## **Abstract** 

Recent advances in deep learning, exemplified by Hyper-Connections (HC), have expanded the residual connection paradigm by introducing wider residual streams and diverse connectivity patterns. While these innovations yield significant performance gains, they compromise the identity mapping property of residual connections, leading to training instability, limited scalability, and increased memory overhead. To address these challenges, we propose **JPmHC** ( **J** acobian-spectrum **P** reserving **m** anifold-constrained **H** yper- **C** onnections), a framework that replaces identity skips with a trainable linear mixer acting on _n_ parallel streams while explicitly controlling gradient conditioning. By constraining the mixer _M_ on spectrum-controlled manifolds (e.g. Stiefel, Grassmann), JPmHC prevents gradient pathologies and enhances stability. 

JPmHC introduces three key contributions: (i) a free-probability analysis that predicts Jacobian spectra for structured skips, providing actionable design rules for mixer selection; (ii) memory-efficient implicit differentiation for fixed-point projections, reducing activation memory and synchronization overhead; and (iii) a Stiefel-constrained mixer via Cayley transforms, ensuring orthogonality without post-hoc normalization. Empirical evaluations on ARC-AGI demonstrate that JPmHC achieves faster convergence, higher accuracy, and lower computational cost compared to bistochastic baselines, with a rank- _p_ Grassmannian variant tracking between the two—consistent with the spectral theory predictions. As a flexible and scalable extension of HC, JPmHC advances spectrum-aware, stable, and efficient deep learning, offering insights into topological architecture design and foundational model evolution. 

_Disclaimer: This paper was prepared for informational purposes by the LLM Suite group of JP Morgan Chase and its affiliates (‘JPMC’) and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation, warranty or undertaking whatsoever and disclaims all liability for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation, offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful_ 

1 

© 2026 JP Morgan Chase & Co. All rights reserved. 

## **1 Introduction** 

The residual connection [He et al., 2016]—the per-layer update _x[l]_[+1] = _F_ ( _x[l]_ ) + _x[l]_ —is a defining feature of modern deep learning, underpinning Transformers [Vaswani et al., 2017] and virtually every large-scale architecture deployed today. Its variants—Pre-Norm, DeepNorm [Wang et al., 2024]—have enabled training at thousands of layers by smoothing loss landscapes [Li et al., 2018] and stabilizing gradient flow [Pennington et al., 2017, Tarnowski et al., 2019]. However, the identity skip biases layerwise mappings toward the identity, anchoring the function class and limiting expressivity. 

A natural generalization replaces the identity skip with a learned linear map, 

**==> picture [277 x 12] intentionally omitted <==**

increasing expressivity but risking gradient instability if the operator norm _∥H_ res _∥_ and the singular spectrum of the end-to-end Jacobian are not controlled. To decouple expressivity from identity anchoring while preserving trainability at scale, Hyper-Connections (HC) [Zhu et al., 2024] split the hidden state into _n_ parallel streams and mix them through a small _n × n_ matrix. 

Let each stream live in R _[p]_ and stack the streams so that _x ∈_ R _[n] ⊗_ R _[p][∼]_ = R _[np]_ , with _n ≪ p_ (typically _n_ = 4, _p_ = 512). The HC block takes the form 

**==> picture [374 x 14] intentionally omitted <==**

where _H_ res( _x_ ) _, H_ pre( _x_ ) _, H_ post( _x_ ) _∈_ R _[n][×][n]_ are small mixing matrices that depend on the input _x_ . The extra cost scales with _n_ and remains negligible since _F_ is evaluated once per block on a stream mixture and re-distributed. The network learns _which_ information flows where— a strictly richer connection pattern. The gains are most dramatic in the Mixture-of-Experts (MoE) setting, where HC halved the training tokens needed to match baseline on OLMoE and improved BBH and GSM8K by +7 points on DeepSeek’s 27B MoE model [Zhu et al., 2024, Xie et al., 2025]. Both MoE and HC are learnable routing mechanisms—one for tokens across experts, the other for residual streams across layers—and both face the same stability challenge: unconstrained, they diverge (signal gains exceeding 3000 _×_ at 27B scale [Xie et al., 2025]). 

Manifold-Constrained Hyper-Connections (mHC) [Xie et al., 2025] addressed this instability by projecting _H_ res onto the Birkhoff polytope of doubly stochastic matrices via the Sinkhorn– Knopp iteration. Doubly stochastic mixers are appealing because (i) their operator norm is bounded by 1, preventing gradient explosion, and (ii) they act as transport plans [Villani, 2003], intuitively preserving information across streams. At 27B-parameter scale, mHC demonstrated strong results with minimal overhead. However, two limitations remain: (1) operator-norm boundedness does not preclude _vanishing_ gradients—a full singular-spectrum analysis of the end-to-end Jacobian is absent; and (2) backpropagating through iterative projections introduces memory and synchronization overhead in distributed training. 

This is where the argument breaks down. Training deep networks requires that the singular values of the input-output Jacobian _J_ =[�] _[L] l_ =1 _[Y][l]_[remain concentrated near][ 1][—a property called] _dynamical isometry_ [Saxe et al., 2014, Pennington et al., 2017]. Without it, expressivity capacity is lost or, worse, gradients may either explode or vanish exponentially. Tarnowski et al. [2019] proved that for scalar skip connections, dynamical isometry is universal: for any activation function, it is achieved when a single condition on the weight variance is met. A generalization of their free probability method to general twisting of the skip-connection seems within reach to go beyond operator-norm boundedness. For clarity-sake, we analyse the spectra for a simplified (2): 

**==> picture [327 x 15] intentionally omitted <==**

2 

where each _A[l] n[∈]_[R] _[n][×][n]_[is][a][fixed][mixing][matrix][(independent][of] _[x]_[)][for][theoretical][analysis.] We extend the theory to the operator-valued setting via operator-valued free probability [Voiculescu, 1995, Dykema, 2007], where the Kronecker structure of (3) collapses the spectral problem from network width _N_ = _np_ to twist dimension _n_ . This reveals two failure modes of doubly stochastic skip connections. The first is _eigenvalue contraction_ : a doubly stochastic matrix has its Perron eigenvalue pinned at one, but generically all others lie strictly inside the unit disk, and deep composition drives _|λ|[L] →_ 0. The second is _eigenspace misalignment_ : eigenbases of successive layers are unrelated, so composition scrambles directions and accelerates the collapse beyond what per-layer spectra predict. Together, these produce a partial spectral collapse—a growing fraction of the Jacobian’s singular values drifting toward zero—that no reparametrisation of the Birkhoff polytope can escape. 

Orthogonal matrices eliminate both failure modes: all eigenvalues lie on the unit circle, so no contraction is possible, and group closure under composition prevents misalignment at any depth. We propose replacing the Birkhoff constraint with the orthogonal group _O_ ( _n_ ), parametrised via the Cayley transform [Li et al., 2020a, Lezcano-Casado, 2019], which maps skew-symmetric matrices to orthogonal ones via ( _I − S_ )( _I_ + _S_ ) _[−]_[1] . Beyond spectral preservation, this provides a strictly richer function class (the linear span of _O_ ( _n_ ) is the full algebra _Mn_ (R), dimension _n_[2] , versus ( _n−_ 1)[2] + 1 for the Birkhoff polytope) and implicit geometric nonlinearity from the curvature of the orthogonal manifold. 

## **Contributions.** 

1. **Spectral diagnosis.** We identify eigenvalue contraction and eigenspace misalignment as the mechanisms by which doubly stochastic skip connections break dynamical isometry, and show that this collapse converts to concrete capacity loss in modern training ( _spectral stalling_ ). 

2. **Cayley-transform Stiefel projection.** We instantiate an orthogonality-preserving mixer by projecting _H_ res onto the Stiefel manifold via a small, fixed number of Cayley iterations (as few as _s_ = 2), yielding norm-preserving mixing with exact gradients and negligible overhead [Li et al., 2020a, Lezcano-Casado, 2019]. 

3. **Grassmannian subspace mixer.** We develop a rank- _p_ variant with _O_ ( _np_ ) parameters that mixes through a learned _p_ -dimensional subspace, optimized with a Cayley retraction for efficient Riemannian updates. 

4. **Implicit differentiation for fixed-point projections.** We design a custom backward pass for iterative normalizations and projections (e.g., Sinkhorn for bistochastic constraints, Cayley for orthogonal constraints), reducing activation memory from _O_ ( _T_ ) to _O_ (1) and eliminating distributed data-parallel synchronization stalls, while remaining compatible with CUDA graphs and mixed precision [Eisenberger et al., 2022a]. 

5. **Operator-valued Dyson pipeline.** We develop the first numerical implementation of the full operator-valued free probability pipeline—from the matrix Dyson equation through Dykema’s twisted S-transform multiplicativity to multi-layer spectral densities. 

6. **Experimental validation.** We confirm the spectral predictions against Monte Carlo simulation and validate the practical consequence on a modified Tiny Recursive Model [JolicoeurMartineau, 2025] evaluated on ARC-AGI-1 [Chollet, 2019]: orthogonal skip connections (Cayley) converge faster and reach higher accuracy than bistochastic ones (Sinkhorn), while the rank- _p_ Grassmannian variant tracks between the two, consistent with the spectral theory predictions. 

3 

## **2 Spectral Analysis** 

We now develop the spectral machinery for predicting the singular-value distribution of the endto-end Jacobian in deep networks with structured skip connections. The key insight is that free probability [Voiculescu, 1991, Nica and Speicher, 2006] reduces the spectral analysis of _L_ -layer compositions to fixed-point equations on order parameters, and the Kronecker structure _An ⊗ Ip_ further collapses the problem from network width _N_ = _np_ to twist dimension _n_ . 

## **2.1 Scalar Dyson equation and dynamical isometry** 

Consider a standard residual network with scalar skip connection _a_ and layer-wise update _x[l]_[+1] = _ϕ_ ( _W[l] x[l]_ ) + _ax[l]_ . The linearized layer map is _Y[l]_ = _D[l] W[l]_ + _aIN_ , where _D[l]_ = diag( _ϕ[′]_ ( _h[l] i_[))][and] _W[l] ∈_ R _[N][×][N]_ has i.i.d. Gaussian entries with variance _σw_[2] _[/N]_[.][The][end-to-end][Jacobian][is] _[J]_[=] � _Ll_ =1 _[Y][l]_[.] 

In the mean-field limit ( _N →∞_ ), the activation derivatives _D[l]_ concentrate around their expectation, making _D[l] W[l]_ effectively isotropic [Pennington et al., 2017]. Free probability theory [Voiculescu, 1991] then predicts the limiting spectral density of _J[⊤] J_ via the _Cauchy transform G_ ( _z_ ) := lim _N →∞ N_[1][E][Tr(] _[zI][−][J][⊤][J]_[)] _[−]_[1][and][the] _[S-transform]_[,][which][linearizes][free][multiplicative] convolution: _SJ ⊤J_ ( _w_ ) =[�] _[L] l_ =1 _[S] Y[l][⊤] Y[l]_[(] _[w]_[)][.][One][can][deduce] _[G]_[from] _[S]_[and][vice-versa,] For scalar skip connections, Tarnowski et al. [Tarnowski et al., 2019] derived a scalar fixedpoint equation for the single-layer Cauchy transform. The order parameter _m_ ( _z_ ) satisfies the _scalar Dyson equation_ : 

**==> picture [351 x 24] intentionally omitted <==**

and _q_ is the forward signal variance, _Z ∼N_ (0 _,_ 1). The Cauchy transform is _GY ⊤Y_ ( _z_ ) = _m_ ( _z_ ) _/z_ . For _L_ identical layers, the _z_ 1-mapping converts _SY ⊤Y_ ( _w_ ) _[L]_ back to _GJ ⊤J_ ( _z_ ) without numerically fragile S-transform inversion. 

A network achieves _dynamical isometry_ when the singular values of its Jacobian _J_ concentrate near 1. For scalar skip connections, regardless of activation function or depth [Tarnowski et al., 2019], dynamical isometry is achieved via suitable scaling of layers weights. 

This universality breaks down for structured skip connections: when _a_ is replaced by an _n × n_ 1 matrix _An_ , the scalar trace _N_[Tr][averages][over] _[A][n]_[’s][spectral][sectors.] Scalar approximation distinguishes bistochastic and orthogonal mixer but predictions are inaccurate for bistochastic (mHC) and general linear (HC) mixers, see figure 1. 

## **2.2 Operator-valued extension: Kronecker collapse** 

Hyper-Connections [Zhu et al., 2024] replace the scalar skip with a Kronecker product _A_ = _An ⊗ Ip_ , where _An ∈_ R _[n][×][n]_ mixes _n_ parallel streams of dimension _p_ , and _N_ = _np_ . The layer-wise Jacobian becomes 

**==> picture [285 x 15] intentionally omitted <==**

where _W[l] ∈_ R _[N][×][N]_ remains isotropic Gaussian, but the skip structure is now block-diagonal with _n × n_ blocks. 

**Why scalar theory fails.** The scalar Cauchy transform _G_ ( _z_ ) = _N_[1][Tr(] _[zI][N][−][M]_[)] _[−]_[1][computes] an average over all _N_ eigenvalues. When _M_ = ( _An ⊗ Ip_ ) + noise, this trace averages over the _n_ spectral sectors induced by _An_ , collapsing eigenvalue structure that is critical for gradient flow. For instance, if _An_ is bistochastic with eigenvalues _{_ 1 _, λ_ 2 _, . . . , λn}_ where _|λi| <_ 1 for _i ≥_ 2, the scalar theory sees only the average behavior, not the sector-wise contraction that drives vanishing gradients. 

4 

**Operator-valued free probability.** The solution is to work over the base algebra _B_ = _Mn_ (C) [Voiculescu, 1995, Speicher, 1998]. We promote the order parameter from a scalar _m_ ( _z_ ) _∈_ C to a matrix _M_ ( _z_ ) _∈ Mn_ (C), and the Cauchy transform to a _B_ -valued functional. The Kronecker structure _An ⊗ Ip_ ensures that the self-consistent equation for _M_ ( _z_ ) depends only on _An_ and the noise variance _σ_[2] , not on the stream dimension _p_ . 

Critically, the S-transform multiplicativity rule becomes _twisted_ in the operator-valued setting [Dykema, 2007]: 

**==> picture [348 x 15] intentionally omitted <==**

This conjugation by _SY_ ( _B_ ) encodes the eigenspace rotation between successive layers when _A[l] n_ do not commute—precisely the misalignment effect absent in scalar theory. 

**Proposition 2.1** (Kronecker collapse) **.** _Under the Kronecker structure Y[l]_ = ( _A[l] n[⊗][I][p]_[) +] _[ D][l][W][ l] and mean-field isotropy, the B-valued order parameter M_ ( _z_ ) _∈ Mn_ (C) _defined for z ∈ Mn_ (C) _satisfies the matrix fixed-point equation_ 

**==> picture [326 x 16] intentionally omitted <==**

_where Ah_ ( _z_ ) := _An_ + _σ_[2] _z is the_ dressed matrix _. The scalar Cauchy transform is recovered by −_ 1 _G_ ( _z_ ) = _n_[1][Tr] _[n]_ � _zIn − Ah_ ( _z_ ) _[⊤] Ah_ ( _z_ )� _. When n_ = 1 _, this reduces to_ (4) _._ 

**Proof sketch.** The key steps are: (i) the Kronecker structure implies E[ _W[l]_ ( _A[l] n[⊗][I][p]_[)]][=][0][by] isotropy; (ii) the self-energy Σ( _z_ ) has the block form Σ _n ⊗ Ip_ where Σ _n ∈ Mn_ (C); (iii) inserting the ansatz _M_ ( _z_ ) = _Mn_ ( _z_ ) _⊗ Ip_ into the Dyson-Schwinger equation and tracing over the _p × p_ blocks yields (7). 

**Computational complexity.** Solving (7) requires Newton iteration in C _[n]_[2] at cost _O_ ( _n_[6] ) per step (matrix inversion dominates). Since _n ≪ p_ (typically _n_ = 4, _p_ = 512), this collapses the spectral problem from _O_ ( _N_[3] ) = _O_ (( _np_ )[3] ) to _O_ ( _n_[6] ), a reduction of factor ( _p/n_ )[3] _≈_ 10[5] at typical scales. This makes exhaustive spectral analysis tractable for networks of arbitrary width _N_ . 

## **2.3 Numerical pipeline** 

We now describe the computational methods for solving the scalar and operator-valued Dyson equations and extracting spectral densities. 

**Scalar solver.** For fixed _z ∈_ C[+] (upper half-plane), equation (4) is solved by Newton’s method with the iteration 

**==> picture [308 x 37] intentionally omitted <==**

We sweep a grid of _z_ -values from large _|z|_ to small _|z|_ , using each solution to seed the next ( _branch continuation_ ). This ensures convergence even near the spectral edges where the Cauchy transform has poles. Convergence is typically achieved in 3–5 iterations with tolerance 10 _[−]_[12] . 

**Multi-layer: the** _z_ 1 **-mapping.** For _L_ identical layers, the S-transform multiplicative property gives _SJ ⊤J_ ( _w_ ) = _SY ⊤Y_ ( _w_ ) _[L]_ . The _z_ 1-mapping inverts this directly: given _GY ⊤Y_ ( _z_ 1), we _w_ +1 solve for _w_ such that _χ_ ( _w_ ) := _wSY ⊤Y_ ( _w_ )[=] _[z]_[(the] _[G][↔][S]_[relation),][then][compute] _[S][J][⊤][J]_[(] _[w]_[)][=] _w_ +1 _SY ⊤Y_ ( _w_ ) _[L]_ , and finally solve _wSJ⊤J_ ( _w_ )[=] _[z][′]_[to][obtain] _[G][J][⊤][J]_[(] _[z][′]_[)][.] This avoids numerical S- transform inversion, which is ill-conditioned near _w_ = 0. 

5 

**Matrix Dyson solver.** For Kronecker skip connections with fixed _An_ across layers, equation (7) is a coupled system of _n_[2] complex equations. We vectorize _M ∈ Mn_ (C) to **m** _∈_ C _[n]_[2] and apply Newton’s method: 

**==> picture [317 x 14] intentionally omitted <==**

where _F_ ( **m** ) = **m** _−_ vec� _Ah_ ( _M_ )( _zIn − Ah_ ( _M_ ) _[⊤] Ah_ ( _M_ )) _[−]_[1][�] and _JF_ is the _n_[2] _× n_[2] Jacobian computed via automatic differentiation. The cost is _O_ ( _n_[6] ) per iteration due to the matrix inverse in (7). Branch continuation from large to small _|z|_ remains essential for stability. 

**Operator-valued multi-layer pipeline.** For heterogeneous layers ( _A[l] n_[1] = _A[l] n_[2] ), the twisted S- transform multiplicativity (6) requires iterating the composition _SJ ⊤J_ ( _B_ ) = _SY_ 1 _⊤Y_ 1( _B[′]_ ) _SY_ 2 _⊤Y_ 2( _B[′′]_ ) with conjugation updates _B[′]_ = _SY_ 2( _B_ ) _[−]_[1] _BSY_ 2( _B_ ). Each conjugation requires solving the operator-valued _G ↔ S_ relation, itself a matrix fixed-point problem. The full pipeline has complexity _O_ ( _Ln_[10] ) per _z_ -point (nested matrix inversions). In practice, for _n ≤_ 4 and _L ≤_ 100, this completes in _<_ 1 second per _z_ -point on a CPU. 

**Validation.** Figure 1 compares the theoretical predictions to Monte Carlo histograms of singular values sampled from finite networks ( _n_ = 4, _p_ = 25, 500 samples). At _L_ = 1, the scalar theory (red curves) matches Monte Carlo perfectly for all mixer types. At _L_ = 10, the scalar theory fails for bistochastic and Gaussian mixers, which develop spectral mass near zero (eigenvalue contraction), while orthogonal mixers maintain dynamical isometry. The operator-valued theory correctly predicts the sector-wise collapse for bistochastic matrices. 

**Spectral density extraction.** The spectral density _ρ_ ( _x_ ) is recovered from the imaginary part of the Cauchy transform via the Stieltjes inversion formula: 

**==> picture [298 x 24] intentionally omitted <==**

We evaluate _G_ ( _x_ + _iε_ ) for small _ε ≈_ 0 _._ 01 on a dense real grid _x ∈_ [ _λ_ min _, λ_ max] and extract _ρ_ ( _x_ ) = _−_ Im _G_ ( _x_ + _iε_ ) _/π_ . Numerical integration confirms normalization � _ρ_ ( _x_ ) _dx_ = 1 to within 10 _[−]_[3] . 

## **3 Cayley Twisted Skip-Connections** 

The spectral analysis of Section 2 reveals two failure modes of doubly stochastic skip connections: eigenvalue contraction and eigenspace misalignment. Orthogonal matrices eliminate both—all eigenvalues lie on the unit circle and group closure prevents misalignment at any depth. We therefore constrain the residual mixer **H** res to the orthogonal group _O_ ( _n_ ) via an iterative Cayley transform [Li et al., 2020b,a, Lezcano-Casado, 2019]. 

## **3.1 Iterative Cayley Projection** 

> The Cayley transform maps a skew-symmetric matrix **W** = _−_ **W** _[⊤]_ to an orthogonal matrix via ( **I** _−_ **W** _/_ 2)( **I** + **W** _/_ 2) _[−]_[1] . The closed-form requires a matrix inverse that is expensive for batched, per-token computation. Following Li et al. [2020b], we replace the inverse with a fixed-point iteration that converges to the same retraction. 

Given an unconstrained parameter matrix **H**[˜] _∈_ R _[n][×][n]_ : 

1. **Skew-symmetrize: W** = **H**[˜] _−_ **H**[˜] _[⊤]_ , guaranteeing **W** _∈_ so( _n_ ). 

2. **Initialize: Y** 0 = **I** _n_ + _α_ **W** , with step-size _α >_ 0 (default 0 _._ 1). 

6 

Figure 1: **Scalar and OV theories vs. Monte Carlo singular value densities.** Panels show four skip-connection types ( _n_ = 4 streams, _p_ = 25 per stream, _c_ 2 _L_ = 0 _._ 05, _η_ = 0 _._ 02, 500 samples) at depths _L ∈{_ 1 _,_ 2 _,_ 10 _}_ (rows) for mixers _An ∈{_ Identity _,_ Bistochastic _,_ Orthogonal _, }_ (columns). The scalar Dyson prediction (dotted orange curve) matches Monte Carlo histograms at _L_ = 1 for all cases. At _L ∈_ 2 _,_ 10, bistochastic and Gaussian mixers develop mass near zero (spectral collapse), while orthogonal mixers preserve dynamical isometry. Scalar theory fails while Operator-value theory is able to fully catch the spectrum, regularization parameter _η_ used to reduce numerical instabilities smooth out the distribution: it pushes it away from zero and reduces the spikes thus increases the mass allocated to the 1.0 mode. The scaling _Lc_ 2 = const ensures weights _W[l] ∼N_ (0 _, σw_[2] _[/L]_[)][maintain][constant][forward][signal][variance.][This] normalization is shown to be accurate as spectra have a main mode bounded away from 0 an infinity. 

## 3. **Iterate** _s_ times: 

**==> picture [334 x 13] intentionally omitted <==**

In practice _s_ = 2 iterations suffice, achieving _∥_ **Y** _[⊤]_ **Y** _−_ **I** _∥_ max _<_ 10 _[−]_[3] (Section I). Each step is a single fused multiply-add ( `baddbmm` ), and all matrices are _n × n_ with _n_ = 4, so the overhead relative to the _p_ -dimensional sub-layer _F_ is negligible. 

7 

## **3.2 Layer Architecture** 

A single linear projection produces three unconstrained _n × n_ matrices per token from the flattened stream representation **x** flat _∈_ R _[np]_ : 

**==> picture [395 x 15] intentionally omitted <==**

Each matrix is then projected onto its respective constraint manifold: 

**==> picture [486 x 27] intentionally omitted <==**

The pre-mixer **H** pre is row-stochastic (aggregates streams), the post-mixer **H** post is columnstochastic (fans output back), and the residual mixer **H** res is orthogonal (norm-preserving skip). The forward pass implements (2) as: 

**==> picture [426 x 12] intentionally omitted <==**

where **x** ¯in denotes the stream-averaged input (a mean over the _n_ streams) and _F_ is the sub-layer (multi-head attention or feed-forward network), evaluated _once_ on a single _p_ -dimensional vector. The final combination is fused into a single `baddbmm` call. 

**==> picture [453 x 217] intentionally omitted <==**

**----- Start of picture text -----**<br>
q parallel streams HC-FeedForward Residual Block<br>Twisted Skip<br>� H res( x )  ⊗ Ip � · x<br>+ Layer Output<br>Norm x out ∈ R [N]<br>Stream qH → proj1 stream( Project x ) ·x Feed-ForwardNetwork Layer( · ) Stream 1 → ( q· )  ⊗ streams Expand1 H out Gate ( x ) · ( · )<br>p dims width p back to N  = qp<br>q parallel streams HC-Attention Residual Block<br>Twisted Skip<br>� H res( x )  ⊗ Ip � · x<br>Input + Layer<br>x ∈ R [N] , N  = qp Norm<br>Stream qH → proj1 stream( Project x ) ·x Multi-HeadAttention Layer( · ) Stream 1 → ( q· )  ⊗ streams Expand1 H out Gate ( x ) · ( · )<br>p dims width p back to N  = qp<br>**----- End of picture text -----**<br>


Figure 2: Hyper-Connected Transformer Encoder Block. Block 1 (Multi-Head Attention, bottom) feeds into Block 2 (Feed-Forward, top). Within each block the input forks into a twisted skip path (thick arrows, _H_ res) and a compute path (thin arrows) that projects _q→_ 1 streams, applies the layer, expands 1 _→q_ , and gates via _H_ out; both paths merge at the + node before Layer Norm. 

## **4 Experimental Setup** 

## **4.1 Task: ARC-AGI** 

We evaluate JPmHC on the **Abstraction and Reasoning Corpus** (ARC-AGI) [Chollet, 2019], a benchmark designed to measure general fluid intelligence. Each task presents a small number of demonstration input–output grid pairs and one or more test inputs; the solver must infer the latent transformation rule and produce the exact output grid (Figure 3). Grids are rectangular 

8 

(a) Task 1: Input (b) Task 1: Output (c) Task 2: Input (d) Task 2: Output Figure 3: Two representative ARC-AGI tasks. Each task requires discovering a latent rule (here, pattern tiling and region filling) from a few demonstrations, then applying it to a novel test input. 

matrices of integers 0–9 (visualized as colors), with dimensions up to 30 _×_ 30. A task is solved only when _every_ test output is reproduced cell-for-cell, including its dimensions. 

ARC-AGI is particularly suited to stress-test our spectral claims for two reasons. First, each task has a _unique_ underlying rule, so the benchmark is resistant to memorization and demands systematic generalization—precisely the regime where gradient conditioning determines whether a model can learn compositional abstractions. Second, the all-or-nothing exact-match criterion amplifies the practical consequence of partial spectral collapse: even a small fraction of vanishing singular values can corrupt a few output cells and turn a near-correct grid into a failure. 

For all experiments, we use the full ARC-AGI-1 corpus of 1000 tasks, split evenly between training and evaluation (400 training, 400 evaluation, plus 200 for ablation and validation). This ensures that models are evaluated on held-out tasks with unseen rules, and that the results reflect true generalization rather than memorization. 

## **4.2 Model and Training** 

We adapt the **Tiny Recursive Model (TRM)** [Jolicoeur-Martineau, 2025], a 7M-parameter recursive transformer, by expanding each transformer block with _n_ = 4 parallel streams. The attention and FFN residual sub-blocks are each wrapped by a JPmHC module (Figure 2): a read mapping _H_ pre aggregates streams into a single sublayer input, a write mapping _H_ post fans the output back to all streams, and a residual mixer _H_ res _∈_ R _[n][×][n]_ —constrained to a chosen manifold—mixes the original streams before addition. Hidden dim per stream is _d_ =512 (effective dim _nd_ =2048). Two unique weight-tied blocks are each applied 6 times (12 total recursive passes) with Adaptive Computation Time (ACT) halting, yielding 4 unique JPmHC modules reused across all 12 recursions. 

This architecture is a stringent test bed for mixer design: the 12-fold weight-tied recursion means the _same_ mixing matrix is composed with itself repeatedly, directly exposing the eigenvalue contraction and eigenspace misalignment phenomena analyzed in Section 1. All variants share identical training hyperparameters: AdamAtan2 optimizer [Kunstner et al., 2023] with lr 10 _[−]_[4] , global batch size 768, bfloat16 mixed precision, and PyTorch DDP with `torch.compile` on 8 _×_ NVIDIA B200 GPUs. Full configuration details are provided in Section G. 

## **4.3 Ablated Variants** 

We compare four JPmHC mixer constraints, summarized in Table 1. The key design axes are the manifold constraint on the residual mixer _H_ res and the resulting computational cost per module. 

9 

Table 1: JPmHC variant configurations and per-module compute cost. 

|Variant|Manifold|Key param|FLOPs/module|Norm|Constraint|
|---|---|---|---|---|---|
|Sinkhorn|Birkhof polytope|_T_=20, _k_=16|576|RMSNorm|sigmoid|
|Cayley|Stiefel _O_(_n_)|_s_=2, _α_=0_._1|256|LayerNorm|softmax|
|Grassmann|Gr(_n, p_)|_p_=2, _α_=0_._01|—|LayerNorm|softmax|



## **4.4 Evaluation Metrics** 

We report three complementary metrics: 

- **Exact accuracy** : fraction of tasks where the greedy prediction matches the ground-truth grid cell-for-cell—the strictest measure, directly sensitive to spectral health. 

- **Pass@** _k_ ( _k ∈{_ 1 _,_ 2 _,_ 5 _,_ 10 _,_ 100 _,_ 1000 _}_ ): probability that at least one of _k_ i.i.d. samples is _N −c N_ 

- correct, estimated via pass@ _k_ = E 1 _−_ � _k_ � _/_ � _k_ �[�] . � 

- **Eval LM loss** : stablemax cross-entropy on output grid tokens, providing a smooth proxy for per-token prediction quality. 

## **5 Results** 

The Cayley and Sinkhorn variants have completed training ( _∼_ 516K and _∼_ 511K steps respectively, with near-identical compute budgets). The Grassmann variant has recently started training and has completed _∼_ 111K steps; its results are preliminary but already informative. 

Table 2: ARC-AGI concept evaluation metrics (best observed per metric). Cayley and Sinkhorn have completed training ( _∼_ 516K and _∼_ 511K steps respectively); Grassmann training is ongoing ( _∼_ 111K steps). _[∗]_ Grassmann numbers are preliminary. 

|Metric|Sinkhorn|Cayley|Grassmann_∗_|Cayley/Sink|
|---|---|---|---|---|
|Training Steps|510,951|515,902|110,756|—|
|Eval Accuracy|86.5%|**86.9%**|82.2%|1_._00_×_|
|Exact Accuracy (greedy)|27.9%|**31.4%**|12.8%|1_._13_×_|
|Pass@1|36.5%|**40.5%**|27.5%|1_._11_×_|
|Pass@2|41.7%|**45.4%**|31.2%|1_._09_×_|
|Pass@5|45.5%|**50.5%**|35.9%|1_._11_×_|
|Pass@10|46.8%|**53.1%**|39.1%|1_._14_×_|
|Pass@100|51.9%|**59.2%**|46.0%|1_._14_×_|
|Pass@1000|56.1%|**62.7%**|49.0%|1_._12_×_|
|Eval LM Loss|0.817|**0.643**|1.067|1_._27_×_|



## **5.1 Analysis** 

**Key Observations.** 

1. **Cayley leads across all metrics at convergence.** With both Cayley and Sinkhorn now trained to comparable step counts ( _∼_ 516K and _∼_ 511K), the Cayley variant achieves 40.5% pass@1 (best checkpoint at step 380,755) versus Sinkhorn’s 36.5% at step 406,903— a 1 _._ 11 _×_ gap. The advantage is more pronounced in exact-match accuracy (1 _._ 13 _×_ : 31.4% vs. 27.9%), confirming that Cayley produces more _consistently_ correct full-grid predictions. 

10 

Table 3: Cayley evaluation metrics at each checkpoint. Bold indicates the best value for each metric. Training ran for _∼_ 516K steps. 

|Step|Exact Acc.|Pass@1|Pass@2|Pass@10|Pass@100|Pass@1000|
|---|---|---|---|---|---|---|
|68,951|10.6%|—|—|—|—|—|
|81,903|12.7%|—|—|—|—|—|
|94,854|15.7%|—|—|—|—|—|
|107,805|17.6%|—|—|—|—|—|
|161,952|23.4%|35.4%|40.1%|47.4%|52.7%|54.1%|
|174,904|24.5%|35.6%|40.3%|47.7%|53.9%|57.1%|
|188,951|25.1%|36.3%|39.6%|45.4%|51.2%|52.9%|
|201,902|25.4%|36.8%|40.7%|46.1%|53.8%|56.9%|
|214,853|26.3%|36.3%|40.4%|47.5%|54.9%|58.0%|
|231,951|26.0%|36.9%|39.8%|47.4%|52.5%|53.8%|
|244,902|26.4%|37.3%|41.1%|48.9%|55.5%|57.6%|
|257,853|27.8%|38.5%|41.6%|50.0%|56.2%|58.5%|
|270,804|27.0%|37.7%|41.7%|50.5%|57.4%|59.5%|
|283,755|26.9%|38.5%|43.1%|52.0%|58.0%|60.5%|
|296,707|28.1%|38.7%|43.8%|51.9%|**59.2%**|62.0%|
|309,659|28.3%|39.0%|43.6%|52.6%|**59.2%**|**62.7%**|
|328,951|28.9%|36.4%|40.3%|46.8%|52.9%|54.4%|
|341,902|27.3%|37.9%|42.5%|49.6%|56.4%|58.1%|
|354,853|28.6%|38.1%|43.8%|52.2%|58.4%|60.1%|
|367,804|29.0%|39.5%|43.5%|52.7%|58.6%|60.6%|
|380,755|29.6%|**40.5%**|43.6%|**53.1%**|58.9%|61.4%|
|392,952|29.3%|37.7%|42.1%|49.6%|55.1%|56.6%|
|405,903|29.2%|38.7%|43.8%|51.7%|57.7%|60.5%|
|418,854|**31.4%**|38.2%|**45.4%**|52.5%|58.7%|61.6%|
|431,806|31.0%|38.9%|45.1%|52.1%|59.0%|62.6%|
|456,952|30.9%|38.5%|43.4%|48.4%|55.0%|55.3%|
|469,904|31.0%|38.6%|43.3%|49.6%|55.9%|57.9%|
|482,855|31.4%|38.6%|43.9%|50.0%|56.4%|58.4%|
|502,951|31.3%|36.4%|41.5%|47.5%|53.1%|53.3%|
|515,902|31.4%|36.9%|42.5%|49.3%|54.0%|55.6%|



2. **Scaling with** _k_ **.** All three variants benefit from increased sampling budget with diminishing returns. For Cayley: pass@1 _→_ pass@1000 improves by +22 _._ 2 percentage points (pp). For Sinkhorn: +19 _._ 6 pp. For Grassmann (preliminary): +21 _._ 5 pp. The pass@ _k_ ratio between Cayley and Sinkhorn narrows from 1 _._ 11 _×_ at _k_ = 1 to 1 _._ 12 _×_ at _k_ = 1000, indicating comparable prediction variance at convergence. 

3. **Sinkhorn narrowed the gap but did not close it.** At the earlier reporting point ( _∼_ 349K for Sinkhorn, _∼_ 419K for Cayley), the pass@1 ratio was 1 _._ 19 _×_ and exact-accuracy ratio 1 _._ 41 _×_ . With Sinkhorn now extended to _∼_ 511K steps (matched compute), the pass@1 gap narrowed to 1 _._ 11 _×_ and exact-accuracy to 1 _._ 13 _×_ . Importantly, Sinkhorn’s best pass@1 (36.5%) and pass@1000 (56.1%) were achieved at steps _∼_ 407K and _∼_ 242K respectively; later checkpoints showed _declining_ pass@ _k_ despite continued exact-accuracy gains, suggesting overfitting to greedy decoding at the expense of sampling diversity. 

4. **High token accuracy, divergent task accuracy.** All variants exceed 82% per-token eval accuracy (Cayley 86.9%, Sinkhorn 86.5%, Grassmann 82.2%), yet task-level pass@1 

11 

**==> picture [373 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
Cayley (best ckpt)<br>60<br>Sinkhorn (best ckpt)<br>Grassmann [∗] (111K steps)<br>40<br>20<br>0<br>k =1 k =2 k =5 k =10 k =100 k =1000<br>Sampling Budget<br>7 .<br>2 . 62<br>4 . 550 . 5 . 153 . 8 . 59 951 . 156 . 49<br>(%) 405 . 5 . 45 741 . 45 9 . 46 139 . 46<br>k 36 5 . 231 . 35<br>27<br>Pass@<br>**----- End of picture text -----**<br>


Figure 4: Pass@ _k_ scaling comparison. Cayley consistently outperforms Sinkhorn across all sampling budgets _k_ . Grassmann results are preliminary ( _∼_ 111K steps vs. _∼_ 500K+ for others); at matched step counts, Grassmann tracks ahead of Sinkhorn’s early trajectory. The gap between Cayley and Sinkhorn narrows at higher _k_ , indicating Sinkhorn has higher prediction variance. 

ranges from 27.5% to 40.5%. This confirms that the advantage of orthogonal mixing lies in producing _coherent_ complete solutions, not merely better per-token predictions. 

5. **Computational efficiency.** The Cayley JPmHC module requires _∼_ 2 _._ 25 _×_ fewer FLOPs than Sinkhorn (Table 6), achieving higher accuracy with lower per-module cost—a clear Pareto improvement. 

6. **Lower evaluation loss.** The Cayley variant achieves a best evaluation LM loss of 0 _._ 643 compared to Sinkhorn’s 0 _._ 817—a 21% reduction (Figure 6). This 1 _._ 27 _×_ loss ratio exceeds the pass@1 ratio, suggesting that the accuracy advantage is driven by fundamentally better language modeling on the ARC grid tokens. Both variants show mild loss increase after their respective optima, consistent with slight overfitting at late training stages. 

7. **Faster convergence.** The Cayley variant surpassed Sinkhorn’s _final best_ exact-match accuracy (27.9%) at approximately step _∼_ 297K—58% of Sinkhorn’s training budget. For pass@1, Cayley exceeded Sinkhorn’s best (36.5%) at step _∼_ 202K—only 40% of Sinkhorn’s budget—demonstrating substantially higher sample efficiency. 

8. **Grassmann: promising early trajectory.** Despite only _∼_ 111K steps of training, the Grassmann variant already achieves 27.5% pass@1 and 12.8% exact-match accuracy. At the comparable step count ( _∼_ 113K), Sinkhorn had 22.2% pass@1 and 9.9% exact-match, while Cayley (at _∼_ 108K) had 17.6% exact-match (pass@1 not yet measured). This places Grassmann’s early convergence rate between the two completed variants, consistent with its status as a rank- _p_ orthogonal projection—a middle ground between full orthogonal mixing (Cayley) and full bistochastic mixing (Sinkhorn). 

## **5.2 Compute-Accuracy Tradeoff** 

> _†_ Grassmann backward includes Cayley retraction on _n × p_ frame; values for _n_ = 4, _p_ = 2. 

## **6 Related Work** 

**Hyper-Connections and structured skip connections** Beyond the references on which our work is buit [Zhu et al., 2024, Xie et al., 2025], two concurrent works explore related direc- 

12 

**==> picture [455 x 178] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Per-Token Accuracy (b) Exact-Match (Full Grid)<br>90<br>30<br>80<br>20<br>70<br>Cayley 10 Cayley<br>60<br>Sinkhorn Sinkhorn<br>Grassmann [∗] Grassmann [∗]<br>50 0<br>0 100 200 300 400 500 0 100 200 300 400 500<br>Training Steps ( × 10 [3] ) Training Steps ( × 10 [3] )<br>(%)<br>(%)<br>Accuracy<br>Accuracy<br>Eval<br>Exact<br>**----- End of picture text -----**<br>


Figure 5: Evaluation accuracy curves. **(a)** Per-token accuracy shows Cayley and Sinkhorn both exceeding 86% at convergence, with Grassmann tracking a steep early trajectory. **(b)** Exactmatch accuracy reveals a persistent gap: Cayley plateaus at _∼_ 31% while Sinkhorn saturates at _∼_ 28%. Grassmann at 111K steps (12.8%) tracks ahead of Sinkhorn’s comparable point (9.9% at 113K). _[∗]_ Grassmann training is ongoing. 

tions: Yang and Gao [2026] parametrises doubly stochastic matrices as Kronecker products of smaller bistochastic factors via the Birkhoff-von Neumann theorem, and Alonso [2026] proposes an operator-constrained framework. All of these remain within the Birkhoff polytope. Our work departs from this line by showing that the polytope’s contractive geometry causes spectral collapse, and proposes the orthogonal group as the correct constraint manifold. **Optimization on Matrix Manifolds.** Optimization over structured matrix sets has a rich history. The Cayley transform for parameterizing orthogonal matrices dates to Cayley [1846], with modern applications in neural networks [Li et al., 2020b]. Absil et al. [2008] provides principled gradient descent on curved spaces. Ablin et al. [2024] study the soft landing approach. The Sinkhorn operator and its differentiable variants have been extensively studied for optimal transport [Sinkhorn, 1967, Eisenberger et al., 2022b]. 

**Orthogonal constraints in neural networks** Orthogonal and unitary weight constraints have a rich history in recurrent networks, where they prevent gradient decay across time steps [Arjovsky et al., 2016, Wisdom et al., 2016]. Lezcano-Casado [2019] developed a general framework for gradient-based optimisation on matrix manifolds via trivializations (exponential map, Cayley transform), enabling efficient training with hard orthogonal constraints. We apply these parametrisation techniques not to weight matrices but to the skip connection matrices _Aq_ in Hyper-Connections, motivated by our spectral analysis showing that orthogonality of _Aq_ is the decisive property for preserving dynamical isometry. 

**Signal propagation and mean-field theory** The mean-field theory of deep networks Schoenholz et al. [2017] studies the propagation of pre-activation moments, identifying the edge-of-chaos phase transition. Yang and Schoenholz [2017] extended this to residual networks. Pennington et al. [2017] connected signal propagation to the Jacobian’s singular value distribution, introducing dynamical isometry for nonlinear networks. These works characterise signal propagation through scalar order parameters; our work computes the full spectral density, revealing failure modes—such as partial spectral collapse in bistochastic skip connections—that are invisible to mean-field analysis. To the best of our knowledge, operator valued free probability calculus 

13 

**==> picture [285 x 190] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 . 5<br>Cayley<br>Sinkhorn<br>Grassmann [∗]<br>2<br>1 . 5<br>1<br>0 . 5<br>0 50 100 150 200 250 300 350 400 450 500<br>Training Steps ( × 10 [3] )<br>token)<br>(per<br>Loss<br>LM<br>Eval<br>**----- End of picture text -----**<br>


Figure 6: Evaluation LM loss (per-token cross-entropy, lower is better). The Cayley variant achieves the lowest loss (0 _._ 643 at 419K steps) with a 1 _._ 27 _×_ advantage over Sinkhorn’s best (0 _._ 817). Grassmann’s steep descent suggests it may approach Sinkhorn-level loss with continued training. Both Cayley and Sinkhorn show mild loss increase after their respective optima, suggesting slight overfitting at late training stages. 

for Neural Networks Jabobian spectrum comnputation has been introduced in Yang [2020]. In particular, Operator-valued freeness is proved there. 

**Recursive Reasoning Models.** The Hierarchical Reasoning Model (HRM) Wang et al. [2025] introduced recursive multi-step reasoning with deep supervision for puzzle-solving tasks. The Tiny Recursive Model (TRM) Jolicoeur-Martineau [2025] simplified HRM to a single 2-layer network with weight-tied recursion, achieving state-of-the-art results on ARC-AGI-1 (45% accuracy) with only 7M parameters. We adopt the TRM architecture as our evaluation platform, extending it with structured multi-stream mixing. 

## **7 Discussion** 

## **7.1 Why Does Cayley Outperform Sinkhorn?** 

With both variants now trained to convergence at matched compute budgets ( _∼_ 516K and _∼_ 511K steps), the results show a consistent advantage for the Cayley JPmHC variant over implicit Sinkhorn, with the gap most pronounced in exact-match accuracy (1 _._ 13 _×_ ) and evaluation LM loss (1 _._ 27 _×_ ). We identify two contributing factors supported by both theory and empirical evidence: 

**Bistochastic Induces Spectral Stalling.** Spectral Stalling is the phenomenon by which directions associated with small singular values are ignored during the gradient descent: there is a hard cutoff of the spectrum. Effectively, the spectrum acts as a filter on the parameter space, the model is trained only on the subspace associated with singular values above the threshold. As shown in Section 2, Orthogonal and Bistochastic skip-connections show substantial differences in Jacobian singular spectrum. On the one hand, orthogonal skip-connection is undistinguishable from identity skip-connection, therefore dynamical isometry is achieved, the whole spectrum is above threshold. On the other hand, bistochastic skip-connection shows that more than 75% of the spectral mass is concentrated around 0, which suggest the same fraction of the weights are ignored: the model capacity is reduced. 

14 

**Empirical Gradient Evidence.** The gradient statistics from training corroborate the spectral stalling prediction. Despite achieving _worse_ evaluation loss, the Sinkhorn variant exhibits _∼_ 4 _×_ larger gradient norms than Cayley throughout training (average dense gradient norm: 0 _._ 39 vs. 0 _._ 10). The Grassmann variant, at its earlier training stage, shows even larger norms (0 _._ 84). This pattern—larger gradients with worse loss reduction—is consistent with a significant fraction of gradient energy being directed into spectral sectors with near-zero Jacobian singular values, where parameter updates produce little functional change. In contrast, Cayley’s smaller but more _efficient_ gradients are concentrated in the full-rank spectral region, producing more effective parameter updates per step. The per-layer gradient statistics further support this: Sinkhorn’s maximum per-layer gradient norm (0 _._ 21 avg) is 4 _._ 2 _×_ larger than Cayley’s (0 _._ 05 avg), indicating that gradient energy in the Sinkhorn variant is not only larger in total but also more heterogeneously distributed across layers. 

**Orthogonal Has Full Mixing Expressivity** The respective intrinsic dimension of Orthogonal matrices and Bistochastic matrices are _q_ ( _q −_ 1) _/_ 2 and _q_[2] _−_ 2 _q_ suggesting that Bistochastic matrices are more expressive. However, the latter form a polytope (a linear object) while the former form a spherical domain (non-linear). The span of Orthogonal matrices is the whole space of mixing matrices with dimension _q_[2] while the span of Bistochastic matrices has dimension _q_[2] _−_ 2 _q_ . The non-linear structure of orthogonal matrices has full mixing expressivity, while bistochastic matrices do not. 

**Computational Efficiency.** The Cayley JPmHC variant requires _∼_ 2 _._ 25 _×_ fewer FLOPs per module (Table 6), enabling more optimization steps per unit of wall-clock time. 

## **7.2 Grassmann: A Middle Ground** 

The preliminary Grassmann results ( _∼_ 111K steps) reveal an intriguing pattern. The Grassmann variant uses a rank- _p_ orthogonal projector **UU** _[⊤]_ (Section J), which shares the orthogonality structure of Cayley but with reduced rank. At matched step counts, Grassmann tracks ahead of Sinkhorn’s early trajectory (27.5% vs. 22.2% pass@1 at _∼_ 111K steps) but behind Cayley’s. This ordering—Cayley _>_ Grassmann _>_ Sinkhorn at matched steps—is consistent with the spectral theory prediction: orthogonal projections (full-rank or rank- _p_ ) preserve more of the gradient spectrum than bistochastic matrices, with the full-rank Cayley variant preserving the most. 

The Grassmann variant also offers the lowest per-module FLOPs (72 vs. 256 for Cayley), making it a potentially attractive efficiency–accuracy trade-off. Whether Grassmann’s asymptotic performance at convergence matches or exceeds Sinkhorn’s will be determined as training continues. 

## **7.3 Implicit Differentiation: Correctness and Efficiency** 

The custom backward pass for Sinkhorn (Section H) achieves two goals simultaneously: 

1. **Memory reduction** : From _O_ ( _T_ ) intermediate tensors to _O_ (1) (only the output **P** ). 

2. **DDP compatibility** : Elimination of 128K autograd nodes that caused synchronization stalls in distributed training. 

The key insight is that Sinkhorn’s fixed-point structure admits a closed-form implicit derivative. The Jacobian-vector product of the Sinkhorn operator at its fixed point can be expressed as a linear system involving only the fixed point **P** itself, bypassing the need to unroll through _T_ iterations. 

15 

**Self-Stabilization.** An important property of the implicit gradient formula (23) is _self-stabilization_ : the Hadamard product **P** _⊙_ ( _·_ ) automatically zeros out gradient contributions to entries where _Pij ≈_ 0, preventing gradient flow through near-zero mixing weights. This is analogous to the “straight-through” behavior of hard attention, but arises naturally from the fixed-point structure. 

## **7.4 Comparison with Hyper-Connections and mHC** 

Our JPmHC framework extends both the original HC [Zhu et al., 2025] and the concurrent mHC [Xie et al., 2025] in several dimensions: 

## **7.5 Limitations** 

**Late-Stage Overfitting.** Both Cayley and Sinkhorn show increasing evaluation LM loss after their respective best checkpoints (Cayley after _∼_ 310K, Sinkhorn after _∼_ 420K), while exactmatch accuracy continues to improve. This divergence between loss and accuracy, combined with near-zero training loss ( _<_ 0 _._ 002), suggests mild overfitting that manifests as reduced sampling diversity (declining pass@ _k_ ) despite improved greedy performance. 

**Pre/Post Architecture Confound.** As noted, the Cayley and Sinkhorn variants differ in pre/post normalization and mapping architecture, making it impossible to attribute the entire performance gap to the manifold choice alone. 

**Single Architecture.** All experiments use the 7M-parameter TRM on ARC-AGI. Generalization to larger models, different architectures, and other tasks (language modeling, vision) remains to be validated. 

**Incomplete Grassmann Training.** The Grassmann variant has completed only _∼_ 111K of the planned _∼_ 500K+ steps. While early results are promising, conclusions about its asymptotic performance relative to Cayley and Sinkhorn are premature. 

**Small** _n_ **.** With _n_ = 4 streams, the _n × n_ mixing matrices are small enough for exact spectral analysis. Scaling to _n ≥_ 8 may require approximate methods for the operator-valued Dyson pipeline. 

## **8 Conclusion** 

We have presented JPmHC, a unified framework of manifold-constrained mixing strategies for multi-stream residual architectures, extending the mHC framework [Xie et al., 2025] with novel projection methods and efficient differentiation. Our contributions include **implicit Sinkhorn differentiation** , **Cayley transform projection** , and **Grassmannian subspace optimization** —addressing complementary challenges: the first eliminates DDP synchronization stalls, the second provides norm-preserving orthogonal mixing with exact gradients, and the third offers parameter-efficient subspace mixing via Riemannian optimization. 

## **8.1 Summary of Key Results** 

- The Cayley JPmHC variant achieves 40.5% pass@1 and 31.4% exact-match accuracy—a persistent 1 _._ 11 _×_ /1 _._ 13 _×_ advantage over Sinkhorn at matched compute budgets ( _∼_ 500K+ steps each). 

16 

- The Cayley variant reaches a 21% lower evaluation LM loss (0 _._ 643 vs. 0 _._ 817) and surpasses Sinkhorn’s _final_ best pass@1 (36.5%) at only 40% of Sinkhorn’s training budget, demonstrating superior sample efficiency. 

- The Cayley JPmHC module requires 2 _._ 25 _×_ fewer FLOPs than Sinkhorn while achieving higher accuracy—a Pareto improvement in both compute and quality. 

- The Sinkhorn variant reaches 36.5% pass@1 and 27.9% exact-match at _∼_ 511K steps, significantly improving from earlier checkpoints but unable to close the gap to Cayley. 

- The Grassmann variant, at only _∼_ 111K steps, already achieves 27.5% pass@1—exceeding Sinkhorn’s performance at matched step counts and offering the lowest per-module FLOPs (72 vs. 256 for Cayley). 

- Empirical gradient statistics corroborate the spectral stalling theory: the Sinkhorn variant exhibits 4 _×_ larger gradient norms than Cayley despite achieving worse loss, consistent with gradient energy dissipating in near-zero spectral sectors. 

- All JPmHC variants exceed 82% per-token accuracy, confirming the viability of structured mixing for recursive reasoning. 

## **8.2 Broader Impact** 

This work demonstrates that _geometric structure_ —manifold constraints, group-theoretic analysis, implicit differentiation—can be profitably applied to architectural components typically treated as unconstrained parameters. By restricting mixing matrices to well-understood mathematical objects (orthogonal matrices, doubly-stochastic matrices, Grassmannians), we obtain models that are more computationally efficient and more effective. This approach is orthogonal to advances in attention mechanisms, normalization, and activation functions, suggesting potential for broader adoption in multi-stream architectures. 

## **8.3 Future Work** 

- **Complete Grassmann training** : Extend the Grassmann run to _∼_ 500K+ steps to determine its asymptotic performance relative to Cayley and Sinkhorn. 

- **Pre/post ablation** : Isolate the contribution of manifold choice from pre/post architecture differences. 

- **Scale experiments** : Larger models ( _n ≥_ 8 streams, _d ≥_ 1024) and additional benchmarks (language modeling, ARC-AGI-2 [Chollet et al., 2025]). 

- **Adaptive variant selection** : Learn which mixing strategy to apply at each layer during training. 

- **Overfitting mitigation** : Investigate regularization strategies to prevent the late-stage loss/accuracy divergence observed in both Cayley and Sinkhorn. 

## **References** 

Pierre Ablin, Simon Vary, Bin Gao, and Pierre-Antoine Absil. Infeasible deterministic, stochastic, and variance-reduction algorithms for optimization under orthogonality constraints. _Journal of Machine Learning Research_ , 25(389):1–38, 2024. URL `http://jmlr.org/papers/v25/ 23-0451.html` . 

17 

- Pierre-Antoine Absil, Robert Mahony, and Rodolphe Sepulchre. _Optimization Algorithms on Matrix Manifolds_ . Princeton University Press, 2008. 

- A. Noguer I Alonso. Operator-constrained residual connections. Technical Report 6048614, SSRN, 2026. 

- D. G. Anderson. Iterative procedures for nonlinear integral equations. _Journal of the ACM_ , 12 (4):547–560, 1965. 

- G. W. Anderson, A. Guionnet, and O. Zeitouni. _An Introduction to Random Matrices_ . Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2010. 

- M. Arjovsky, A. Shah, and Y. Bengio. Unitary evolution recurrent neural networks. In _Proceedings of the International Conference on Machine Learning_ , pages 1120–1128, 2016. 

- L. Armijo. Minimization of functions having Lipschitz continuous first partial derivatives. _Pacific Journal of Mathematics_ , 16(1):1–3, 1966. 

- Z. Bai and J. W. Silverstein. _Spectral Analysis of Large Dimensional Random Matrices_ . Springer Series in Statistics. Springer, New York, second edition, 2010. 

- S. T. Belinschi and H. Bercovici. A new approach to subordination results in free probability. _Journal d’Analyse Mathématique_ , 101:357–365, 2007. 

- S. T. Belinschi, R. Speicher, J. Treilhard, and C. Vargas. Operator-valued free multiplicative convolution: analytic subordination theory and applications to random matrix theory. _International Mathematics Research Notices_ , 2015(14):5933–5958, 2015. arXiv:1209.3508. 

- S. T. Belinschi, T. Mai, and R. Speicher. Analytic subordination theory of operator-valued free additive convolution and the solution of a general random matrix problem. _Journal of the European Mathematical Society_ , 19(8):2241–2312, 2017. 

- H. Bercovici and D. Voiculescu. Free convolution of measures with unbounded support. _Indiana University Mathematics Journal_ , 42(3):733–773, 1993. 

- P. Biane. Processes with free increments. _Mathematische Zeitschrift_ , 227:143–174, 1998. 

- Z. Burda, A. Jarosz, G. Livan, M. A. Nowak, and A. Swiech. Eigenvalues and singular values of products of rectangular Gaussian random matrices. _Physical Review E_ , 82(6):061114, 2010. 

- Arthur Cayley. Sur quelques propriétés des déterminants gauches. _Journal für die reine und angewandte Mathematik_ , 32:119–123, 1846. 

- François Chollet. On the measure of intelligence. _arXiv preprint arXiv:1911.01547_ , 2019. 

- François Chollet, Mike Knoop, Greg Kamradt, Bryan Landers, and Hansueli Pinkard. ARCAGI-2: A new challenge for frontier AI reasoning systems. _arXiv preprint arXiv:2505.11831_ , 2025. 

- K. Dykema. On the S-transform over a Banach algebra. _Journal of Functional Analysis_ , 231(1): 90–110, 2006. arXiv:math/0501083v2. 

- K. Dykema. Multilinear function series and transforms in free probability. _Advances in Mathematics_ , 208(1):351–407, 2007. 

- M. Eisenberger, A. Toker, L. Leal-Taixé, F. Bernard, and D. Cremers. A unified framework for implicit Sinkhorn differentiation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 509–519, 2022a. arXiv:2205.06688. 

18 

- Marvin Eisenberger, Aysim Toker, Laura Leal-Taixé, Florian Bernard, and Daniel Cremers. A unified framework for implicit Sinkhorn differentiation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022b. 

- G. H. Golub and J. H. Welsch. Calculation of Gauss quadrature rules. _Mathematics of Computation_ , 23(106):221–230, 1969. 

- U. Haagerup and S. Möller. The law of large numbers for the free multiplicative convolution. _Operator Theory: Advances and Applications_ , 149:157–186, 2005. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 770–778, 2016. 

- J. W. Helton, R. Far, and R. Speicher. Operator-valued semicircular elements: solving a quadratic matrix equation with positivity constraints. _International Mathematics Research Notices_ , 2007(22):rnm086, 2007. 

- Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks. _arXiv preprint arXiv:2510.04871_ , 2025. 

- D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In _International Conference on Learning Representations_ , 2015. arXiv:1412.6980. 

- Frederik Kunstner, Jacques Chen, Jonathan Wilder Lavington, and Mark Schmidt. Noise is not the main factor behind the gap between SGD and Adam on transformers, but sign descent might be. _arXiv preprint arXiv:2304.13960_ , 2023. 

- M. Ledoux. _The Concentration of Measure Phenomenon_ , volume 89 of _Mathematical Surveys and Monographs_ . American Mathematical Society, 2001. 

- M. Lezcano-Casado. Trivializations for gradient-based optimization on manifolds. In _Advances in Neural Information Processing Systems_ , volume 32, pages 9154–9164, 2019. 

- Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. _Advances in neural information processing systems_ , 31, 2018. 

- J. Li, F. Li, and S. Todorovic. Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform. In _International Conference on Learning Representations_ , 2020a. arXiv:2002.01113. 

- Jun Li, Fuxin Li, and Sinisa Todorovic. Efficient riemannian optimization on the Stiefel manifold via the Cayley transform. In _International Conference on Learning Representations (ICLR)_ , 2020b. 

- V. A. Marchenko and L. A. Pastur. Distribution of eigenvalues in certain sets of random matrices. _Matematicheskii Sbornik_ , 72(4):507–536, 1967. 

- R. A. Meyer, C. Musco, C. Musco, and D. P. Woodruff. Hutch++: Optimal stochastic trace estimation. _Proceedings of the Symposium on Simplicity in Algorithms (SOSA)_ , pages 142–155, 2021. arXiv:2010.09649. 

- Y. Nakatsukasa, O. Sète, and L. N. Trefethen. The AAA algorithm for rational approximation. _SIAM Journal on Scientific Computing_ , 40(3):A1494–A1522, 2018. 

- Alexandru Nica and Roland Speicher. _Lectures on the combinatorics of free probability_ , volume 13. Cambridge University Press, 2006. 

19 

- J. Nocedal and S. J. Wright. _Numerical Optimization_ . Springer Series in Operations Research. Springer, New York, second edition, 2006. 

- J. Pennington and P. Worah. Nonlinear random matrix theory for deep learning. In _Advances in Neural Information Processing Systems_ , volume 30, 2017. 

- J. Pennington, S. Schoenholz, and S. Ganguli. Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. In _Advances in Neural Information Processing Systems_ , volume 30, 2017. 

- P. Pulay. Convergence acceleration of iterative sequences. The case of SCF iteration. _Chemical Physics Letters_ , 73(2):393–398, 1980. 

- N. R. Rao and R. Speicher. Multiplication of free random variables and the S-transform: the case of vanishing mean. _Electronic Communications in Probability_ , 12:248–258, 2007. 

- L. F. Richardson. The approximate arithmetical solution by finite differences of physical problems involving differential equations, with an application to the stresses in a masonry dam. _Philosophical Transactions of the Royal Society A_ , 210:307–357, 1911. 

- A. M. Saxe, J. L. McClelland, and S. Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. _arXiv preprint arXiv:1312.6120_ , 2014. 

- S. S. Schoenholz, J. Gilmer, S. Ganguli, and J. Sohl-Dickstein. Deep information propagation. In _International Conference on Learning Representations_ , 2017. 

- Richard Sinkhorn. Diagonal equivalence to matrices with prescribed row and column sums. _The American Mathematical Monthly_ , 74(4):402–405, 1967. 

- R. Speicher. Combinatorial theory of the free product with amalgamation and operator-valued free probability theory. _Memoirs of the American Mathematical Society_ , 132(627), 1998. 

- Wojciech Tarnowski, Piotr Warchoł, Stanisław Jastrzębski, Jacek Tabor, and Maciej Nowak. Dynamical isometry is achieved in residual networks in a universal way for any activation function. In _The 22nd International Conference on Artificial Intelligence and Statistics_ , pages 2221–2230. PMLR, 2019. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_ , volume 30, 2017. 

- C. Villani. _Topics in Optimal Transport_ , volume 58 of _Graduate Studies in Mathematics_ . American Mathematical Society, Providence, RI, 2003. 

- D. Voiculescu. Operations on certain non-commutative operator-valued random variables. _Astérisque_ , 232:243–275, 1995. 

- D. Voiculescu, K. J. Dykema, and A. Nica. _Free Random Variables_ , volume 1 of _CRM Monograph Series_ . American Mathematical Society, Providence, RI, 1992. 

- Dan Voiculescu. Limit laws for random matrices and free products. _Inventiones mathematicae_ , 104(1):201–220, 1991. 

- H. F. Walker and P. Ni. Anderson acceleration for fixed-point iterations. _SIAM Journal on Numerical Analysis_ , 49(4):1715–1735, 2011. 

20 

- Guoxin Wang, Jiaqi Li, Yifan Sun, Xiang Chen, Chang Liu, Yang Wu, Ming Lu, Shuqiang Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model. _arXiv preprint arXiv:2506.21734_ , 2025. 

- H. Wang, S. Ma, L. Dong, S. Huang, D. Zhang, and F. Wei. DeepNet: Scaling transformers to 1,000 layers. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2024. arXiv:2203.00555. 

- S. Wisdom, T. Powers, J. Hershey, J. Le Roux, and L. Atlas. Full-capacity unitary recurrent neural networks. In _Advances in Neural Information Processing Systems_ , volume 29, 2016. 

- M. A. Woodbury. Inverting modified matrices. _Memorandum Report_ , 42, 1950. 

- Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang. mHC: Manifold-constrained hyper-connections. _arXiv preprint arXiv:2512.24880_ , 2025. 

- G. Yang and S. Schoenholz. Mean field residual networks: on the edge of chaos. In _Advances in Neural Information Processing Systems_ , volume 30, 2017. 

- Greg Yang. Tensor programs iii: Neural matrix laws. _arXiv preprint arXiv:2009.10685_ , 2020. 

- X. Yang and X. Gao. KromHC: Manifold-constrained hyper-connections with Kronecker-product residual matrices. _arXiv preprint arXiv:2601.21579_ , 2026. 

- Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou. Hyper-connections. _arXiv preprint arXiv:2409.19606_ , 2025. v2: 2512.24880. 

- Z. Zhu, Y. He, L. Liu, J. Xu, Z. Xie, A. Dai, and D. Dai. Hyper-connections. _arXiv preprint arXiv:2409.19606_ , 2024. Accepted at ICLR 2025. 

## **A Matrices Manifolds** 

## **A.1 Doubly-Stochastic Matrices and the Birkhoff Polytope** 

**Definition A.1** (Doubly-Stochastic Matrix) **.** _A matrix_ **P** _∈_ R _[n][×][n] is_ doubly stochastic _if Pij ≥_ 0 _for all i, j,_ **P1** = **1** _, and_ **P** _[⊤]_ **1** = **1** _._ 

The set of all _n × n_ doubly-stochastic matrices forms the _Birkhoff polytope Bn_ . By the Birkhoff-von Neumann theorem, _Bn_ is the convex hull of the _n_ ! permutation matrices: 

**==> picture [286 x 12] intentionally omitted <==**

**==> picture [131 x 12] intentionally omitted <==**

## **A.2 Sinkhorn-Knopp Algorithm** 

The Sinkhorn-Knopp algorithm Sinkhorn [1967] projects an arbitrary non-negative matrix onto the Birkhoff polytope via alternating row and column normalization. In log-space (for numerical stability): 

21 

**Algorithm 1** Sinkhorn-Knopp Projection (Log-Space) 

**Require:** Unconstrained logit matrix **X** _∈_ R _[n][×][n]_ , iterations _T_ **Ensure:** Doubly-stochastic matrix **P** _∈Bn_ 

1: log **M** _←_ clamp( **X** _, −_ 10 _,_ 10) 

2: **for** _t_ = 1 _, . . . , T_ **do** 

3: log **M** _←_ log **M** _−_ LSErow(log **M** ) 4: log **M** _←_ log **M** _−_ LSEcol(log **M** ) 5: **end for** 

_▷_ Row normalize _▷_ Column normalize 

6: **P** _←_ exp(log **M** ) 

- 7: **return P** 

where LSErow( **A** ) _ij_ = log[�] _k_[exp(] _[A][ik]_[)][broadcasts][along][rows.] 

## **A.3 The Stiefel Manifold** 

**Definition A.2** (Stiefel Manifold) **.** _The Stiefel manifold_ St( _n, p_ ) _is the set of n × p matrices with orthonormal columns:_ 

**==> picture [312 x 14] intentionally omitted <==**

When _p_ = _n_ , St( _n, n_ ) = _O_ ( _n_ ) is the orthogonal group. Points on St( _n, p_ ) can be parametrized via the Cayley transform of skew-symmetric matrices. 

## **A.4 The Grassmann Manifold** 

**Definition A.3** (Grassmann Manifold) **.** _The Grassmann manifold_ Gr( _n, p_ ) _is the set of p- dimensional subspaces of_ R _[n] :_ 

**==> picture [287 x 12] intentionally omitted <==**

_where O_ ( _p_ ) _acts by right multiplication. Two matrices_ **U** _,_ **V** _∈_ St( _n, p_ ) _represent the same point on_ Gr( _n, p_ ) _if_ **V** = **UQ** _for some_ **Q** _∈ O_ ( _p_ ) _._ 

The canonical representation of a Grassmannian point is the orthogonal projector **P** = **UU** _[⊤]_ , which is invariant to the _O_ ( _p_ ) fiber action. 

## **A.5 Cayley Transform** 

The Cayley transform maps skew-symmetric matrices to orthogonal matrices: 

**==> picture [312 x 13] intentionally omitted <==**

where **W** = _−_ **W** _[⊤]_ is skew-symmetric. This mapping is a diffeomorphism from the space of skewsymmetric matrices to the connected component of _O_ ( _n_ ) containing the identity (i.e., det = +1), minus the set where **I** _−_ **W** _/_ 2 is singular. 

22 

## **B Detailed Algorithm Pseudocode** 

## **B.1 Complete Sinkhorn Implicit Backward** 

**Algorithm 2** Complete Sinkhorn Implicit Backward Pass 

**Require:** Saved output **P** _∈_ R _[B][×][n][×][n]_ , upstream gradient _∂∂ℓ_ **P Require:** Number of Gauss-Seidel iterations _k_ (default: 4 _n_ = 16) **Ensure:** Gradient _∂ℓ ∂_ **M** 1: **G** _← ∂[∂ℓ]_ **P** _▷_ Upstream gradient 2: **H** _←_ **P** _⊙_ **G** _▷_ Element-wise product 3: _h_ row _←_ **H** _·_ **1** _▷_ Row sums: ( _B, n,_ 1) 4: _h_ col _←_ **H** _[⊤] ·_ **1** _▷_ Col sums: ( _B, n,_ 1) 5: **v** _←_ **0** _▷_ Initialize dual variable 6: **for** _i_ = 1 _, . . . , k_ **do** 7: **u** _← h_ row _−_ **P** _·_ **v** _[⊤] ▷_ Update _u_ from _v_ 8: **v** _← h_ col _−_ **P** _[⊤] ·_ **u** _[⊤] ▷_ Update _v_ from _u_ 9: **end for** _∂ℓ_ 10: _∂_ **M** _[←]_ **[H]** _[ −]_ **[u]** _[ ·]_ **[ P]** _[ −]_ **[v]** _[ ·]_ **[ P]** _▷_ Gradient w.r.t. logit matrix _∂ℓ_ 11: **return** _∂_ **M** 

## **B.2 Complete Cayley Transform** 

**Algorithm 3** Iterative Cayley Transform 

**Require:** Input matrix **H** _∈_ R _[B][×][n][×][n]_ , step size _α_ , iterations _s_ **Ensure:** Orthogonal matrix **Q** _∈ O_ ( _n_ ) 1: **H**[˜] _←_ **H** _._ view( _B, n, n_ ) 2: **W** _←_ **H**[˜] _−_ **H**[˜] _[⊤]_ 3: **Y** _←_ **I** + _α_ **W** 4: **for** _i_ = 1 _, . . . , s_ **do** 5: **Y** _←_ **I** + _[α]_ 2 **[W]**[(] **[I]**[ +] **[ Y]**[)] _▷_ 6: **end for** 7: **Q** _←_ **Y** 8: **return Q** 

_▷_ Skew-symmetrize _▷_ Initialize: _Y_ 0 = _I_ + _αW ▷_ Fixed-point iteration via `baddbmm` 

## **B.3 Grassmannian Riemannian Step** 

**Algorithm 4** Riemannian Gradient Step on Gr( _n, p_ ) 

**Require:** Basis **U** _∈_ St( _n, p_ ), Euclidean gradient _∇E_ , momentum **M Require:** Step size _α_ , momentum coefficient _β_ 1 **Ensure:** Updated basis **U** _[′] ∈_ St( _n, p_ ) 1: _∇_ hor _←_ ( **I** _−_ **UU** _[⊤]_ ) _∇E ▷_ Horizontal (Riemannian) gradient 2: **M** _← β_ 1 **M** + (1 _− β_ 1) _∇_ hor _▷_ Momentum update 3: **W** _←_ **MU** _[⊤] −_ **UM** _[⊤] ▷_ Skew-symmetric generator 4: **U** _[′] ←_ **U** _▷_ Initialize Cayley retraction 5: **for** _i_ = 1 _, . . . , s_ **do** 6: **U** _[′] ←_ **U** + _[α]_ 2 **[W]**[(] **[U]**[ +] **[ U]** _[′]_[)] _▷_ Cayley retraction iteration 7: **end for** 8: **return U** _[′]_ 

23 

## **C Spectral Gap Computation Details** 

## **C.1 Algorithm for Exhaustive Spectral Gap Search** 

**Algorithm 5** Exhaustive Search for Maximum Spectral Gap Generating Set 

**Require:** Group order _n_ , number of generators _K_ **Ensure:** Generating set _S[∗] ⊆ Sn_ with _|S[∗] |_ = _K_ and maximum spectral gap 1: _G ← Sn ▷_ Symmetric group on _n_ elements 2: elements _←_ list( _G_ ), _N ←|G|_ = _n_ ! 3: ∆ _[∗] ←_ 0, _S[∗] ←∅_ 4: **for** each _S ⊆ G_ with _|S|_ = _K_ **do** 5: **if** _S_ does not generate _G_ via BFS closure **then** 6: **continue** 7: **end if** 8: **T** _←_ **0** _[N][×][N] ▷_ Transition matrix 9: **for** _g ∈ G_ , _σ ∈ S_ **do** 10: **T** [ _g, gσ_ ] += 1 _/K ▷_ Right-multiplication walk 11: **end for** 12: _λ_ 1 _, λ_ 2 _, . . . ←_ eigenvalues( **T** ), sorted by _|λ|_ 13: ∆ _←|λ_ 1 _| −|λ_ 2 _|_ 14: **if** ∆ _>_ ∆ _[∗]_ **then** 15: ∆ _[∗] ←_ ∆, _S[∗] ← S_ 16: **end if** 17: **end for** 18: **return** _S[∗]_ , ∆ _[∗]_ 

## **C.2 Complexity Analysis** 

For _n_ streams and _K_ generators: 

- _n_ ! 

- • Number of candidate subsets: � _K_ � 

- BFS closure check: _O_ ( _n_ ! _· K_ ) per subset 

- Eigendecomposition: _O_ (( _n_ !)[3] ) per valid generating set 

- Total: _O Kn_ !� _·_ ( _n_ !)[3][�] �� 

## **D JPmHC Module Parameter Counts** 

With 4 unique JPmHC modules per model (2 per unique layer, weight-tied across 6 recursive cycles), the total JPmHC parameter overhead ranges from _∼_ 115K (Perm Mix) to _∼_ 410K (Cayley), which is 1 _._ 6–5 _._ 8% of the _∼_ 7M total model parameters. 

## **E Convergence of Gauss-Seidel Solver** 

The convergence rate of the Gauss-Seidel solver for the implicit Sinkhorn backward depends on the spectral radius _ρ_ of the iteration matrix. For a doubly-stochastic matrix **P** with entries bounded away from 0 and 1, the spectral radius satisfies _ρ <_ 1, ensuring convergence. 

Our default of _k_ = 4 _n_ = 16 iterations achieves _<_ 0 _._ 1% gradient error, which is sufficient for stable training with standard learning rates. 

24 

## **F Derivation of Dyson Equation for mHC (Scalar case)** 

We derive the generalized Green’s function governing the singular value spectrum of random matrices of the form _Y_ = _A_ + _X_ , where _A_ is deterministic and _X_ is an isotropic random matrix. Using a block linearization and a resolvent expansion, we prove a matrix Dyson equation (Schwinger–Dyson / Pastur equation) for a deterministic equivalent resolvent. Isotropy forces the self-energy to collapse to scalar multiples of the identity, reducing the spectral analysis to a small set of order parameters. 

## **F.1 Problem Setup** 

Let 

**==> picture [102 x 13] intentionally omitted <==**

where _A_ is deterministic and _X_ is centered random. We study singular values of _Y_ via 

**==> picture [53 x 12] intentionally omitted <==**

For _z ∈_ C[+] define the Stieltjes transform 

**==> picture [123 x 23] intentionally omitted <==**

We will access _GS_ through a block linearization. 

## **F.2 Block Linearization** 

Define the 2 _N ×_ 2 _N_ matrix 

**==> picture [109 x 27] intentionally omitted <==**

**Schur complement.** The (1 _,_ 1) block of _L_ ( _z_ ) _[−]_[1] equals the desired resolvent: 

**==> picture [138 x 14] intentionally omitted <==**

Hence 

**==> picture [123 x 23] intentionally omitted <==**

## **F.3 Generalized Green’s Function** 

For a block matrix 

define the block trace 

**==> picture [183 x 70] intentionally omitted <==**

Define the generalized Green’s function 

**==> picture [176 x 27] intentionally omitted <==**

Then _GS_ ( _z_ ) = _g_ 11( _z_ ). 

25 

## **F.4 Isotropy Assumption** 

We assume _X_ is isotropic in the second-moment sense: there exists _σ_[2] _>_ 0 such that 

**==> picture [309 x 32] intentionally omitted <==**

This holds for i.i.d. Gaussian _Xij ∼N_ (0 _, σ_[2] _/N_ ) and more generally for left-right orthogonally invariant ensembles with the same covariance. 

## **F.5 Resolvent, Resolvent Identity, and the Dyson Equation** 

This section expands the resolvent machinery and proves the Dyson equation used later. 

## **F.5.1 Definition of the resolvent** 

For any (square) matrix _H ∈_ R _[m][×][m]_ and any _w ∈_ C not in the spectrum of _H_ , the _resolvent_ is 

**==> picture [119 x 14] intentionally omitted <==**

In our setting, the primary object is the _block resolvent_ of _L_ ( _z_ ): 

**==> picture [78 x 14] intentionally omitted <==**

When we split _L_ ( _z_ ) into a deterministic part and a random perturbation, we also use the _bare resolvent_ 

**==> picture [87 x 14] intentionally omitted <==**

## **F.5.2 Splitting into deterministic and random parts** 

Write 

**==> picture [173 x 44] intentionally omitted <==**

Thus _L_ = _L_ 0 _−X_ . 

## **F.5.3 Resolvent identity (exact)** 

The following is a standard identity. 

**Lemma (resolvent identity).** If _B_ is invertible and _C_ is arbitrary such that _B − C_ is 

invertible, then 

**==> picture [182 x 13] intentionally omitted <==**

Apply this with _B_ = _L_ 0 and _C_ = _X_ : 

**==> picture [270 x 12] intentionally omitted <==**

Taking expectation (and using that _R_ 0 is deterministic), 

**==> picture [284 x 12] intentionally omitted <==**

Equation (20) is exact but not closed because E[ _XR_ ] depends on correlations between _X_ and _R_ . 

26 

**F.5.4 Dyson equation via a self-energy (Gaussian / Wick closure)** To close (20), one introduces the _self-energy_ operator Σ[ _·_ ]. For Gaussian (or Wick-type) ensembles, the closure is governed by second moments. 

We state a standard large- _N_ closure (often proved with Gaussian integration by parts / Stein’s lemma, or with cumulant expansions and planar diagrammatics). We use it here as the key computational step. 

**Assumption (Gaussian/Wick closure).** Assume _X_ has entries with variance _σ_[2] _/N_ and satisfies Wick’s rule (e.g. i.i.d. Gaussian). Then for resolvents _R_ , the leading-order contribution to E[ _XR_ ] can be written as 

**==> picture [158 x 12] intentionally omitted <==**

in an entrywise or normalized-trace sense (depending on the regularity assumptions). Under this closure, (20) becomes 

**==> picture [145 x 12] intentionally omitted <==**

Rearranging, 

**==> picture [129 x 15] intentionally omitted <==**

hence 

**==> picture [125 x 17] intentionally omitted <==**

This motivates defining a deterministic equivalent _M_ ( _z_ ) as the solution to the _matrix Dyson equation_ : 

**==> picture [304 x 22] intentionally omitted <==**

In many models (including i.i.d. Gaussian), one can show E[ _R_ ( _z_ )] _−M_ ( _z_ ) _→_ 0 in normalized trace, uniformly for Im( _z_ ) _≥ η >_ 0. 

## **F.5.5 Computing the self-energy under isotropy** 

Let 

**==> picture [99 x 28] intentionally omitted <==**

Because _X_ is off-diagonal, Σ[ _M_ ] is also off-diagonal at leading order. Under isotropy, the contraction identities imply: 

**==> picture [300 x 25] intentionally omitted <==**

This yields the self-energy map 

**==> picture [366 x 28] intentionally omitted <==**

Therefore, even for general deterministic _A_ , isotropy collapses the random correction to scalar multiples of the identity, reducing the random-matrix effect to two scalar order parameters. 

27 

## **F.6 Closed Dyson Equation and Order Parameters** 

Combining (21) with the isotropic self-energy gives 

i.e. 

**==> picture [226 x 75] intentionally omitted <==**

The order parameters satisfy the self-consistency equations 

**==> picture [200 x 23] intentionally omitted <==**

Finally, 

**==> picture [107 x 24] intentionally omitted <==**

## **F.7 Special Case: Scalar Skip Connection** 

If _A_ = _aI_ , then the Dyson equation collapses from 2 _N ×_ 2 _N_ to a 2 _×_ 2 system because all blocks commute and _Mij_ are scalar multiples of _I_ . 

## **F.8 Conceptual Takeaway** 

The key mechanism is: 

1. linearize _Y Y[⊤]_ into a 2 _N ×_ 2 _N_ operator, 

2. write the exact resolvent identity, 

3. close it via Wick contractions into a Dyson equation, 

4. use isotropy to reduce the self-energy to scalars _×I_ . 

This identifies the minimal set of order parameters controlling the singular spectrum of _Y_ = _A_ + _X_ . 

## **G Training and Architecture Details** 

## **G.1 TRM Architecture Configuration** 

## **G.2 Training Hyperparameters** 

**AdamAtan2 Optimizer.** AdamAtan2 replaces the standard Adam update _mt/_ ( _[√] vt_ + _ϵ_ ) with atan2( _mt,[√] vt_ ), providing more stable gradient scaling and eliminating sensitivity to the _ϵ_ hyperparameter. For the Grassmann variant, we additionally employ a `GrassmannianOptimizer` wrapper that applies Riemannian gradient steps to the subspace basis parameters **U** . 

**DeepSpeed compatibility.** Although our framework supports DeepSpeed ZeRO-2/ZeRO-3, profiling showed that for 7M parameters, vanilla DDP with `torch.compile` is faster due to lower communication overhead. All JPmHC variants are validated compatible with both backends. 

28 

## **G.3 Permutation Basis Construction** 

For the Perm Mix variant with _n_ = 4 streams and _K_ = 6 permutations, the default permutation basis is: 

1. **Identity** _e_ : (0 _,_ 1 _,_ 2 _,_ 3) 

2. **Adjacent transpositions** : (1 _,_ 0 _,_ 2 _,_ 3), (0 _,_ 1 _,_ 3 _,_ 2) 

3. **Cyclic shifts** : (1 _,_ 2 _,_ 3 _,_ 0), (3 _,_ 0 _,_ 1 _,_ 2) 

4. **Random fill** (seed 42): remaining permutations sampled to reach _K_ = 6 

This basis includes generators of _S_ 4 (adjacent transpositions generate the full symmetric group) while adding cyclic structure for efficient mixing. 

## **H Sinkhorn Variant with Implicit Differentiation** 

The Sinkhorn variant projects the residual mixing matrix **H** res onto the Birkhoff polytope _Bn_ of doubly-stochastic matrices. Our key contribution is a custom backward pass that eliminates the autograd graph explosion of the standard Sinkhorn-Knopp implementation. 

## **H.1 Problem: Autograd Graph Explosion** 

The standard implementation records all _T_ Sinkhorn iterations in the PyTorch autograd graph. Each iteration involves two log-sum-exp operations, each of which PyTorch decomposes into multiple elementary ops (exp, sum, log, subtract). For _T_ = 20 iterations on _n × n_ = 4 _×_ 4 matrices, this creates approximately 128 _,_ 000 backward nodes. 

These nodes produce only _microsecond-scale_ GPU kernels—far too small to overlap with DDP AllReduce communication. Our profiling on NVIDIA B200 GPUs revealed that 55% of the total backward pass time was spent on DDP synchronization stalls, waiting for AllReduce operations to complete because no substantial compute was available to overlap with them. 

## **H.2 Solution: Custom Autograd Function with Implicit Differentiation** 

We implement a custom `torch.autograd.Function` that decouples the forward and backward passes: 

**Forward Pass.** The Sinkhorn iterations run under `torch.no_grad()` , so no autograd graph is recorded. Only the final doubly-stochastic matrix **P** is saved for the backward pass: 

**Algorithm 6** Implicit Sinkhorn — Forward Pass 

**Require:** Logit matrix **X** _∈_ R _[n][×][n]_ , iterations _T_ **Ensure: P** _∈Bn_ (doubly-stochastic) 1: **with** `torch.no_grad()` : _▷_ No autograd recording 2: log **M** _←_ clamp( **X** _, −_ 10 _,_ 10) 3: **for** _t_ = 1 _, . . . , T_ **do** 4: log **M** _←_ log **M** _−_ LSErow(log **M** ) 5: log **M** _←_ log **M** _−_ LSEcol(log **M** ) 6: **end for** 7: **P** _←_ exp(log **M** ) 8: **save P** for backward 9: **return P** 

29 

**Backward Pass.** Gradients are computed via implicit differentiation of the fixed-point conditions, following the framework of Eisenberger et al. Eisenberger et al. [2022b]. At the doublystochastic fixed point, the constraints are: 

**==> picture [287 x 13] intentionally omitted <==**

**Proposition H.1** (Implicit Sinkhorn Gradient Eisenberger et al. [2022b]) **.** _Let_ **P** = Π _B_ ( **X** ) _be the Sinkhorn projection of_ **X** _onto the Birkhoff polytope. Given the upstream gradient ∇_ **P** _L, the gradient with respect to the input is:_ 

**==> picture [308 x 24] intentionally omitted <==**

_where_ **u** _∈_ R _[n][×]_[1] _and_ **v** _∈_ R[1] _[×][n] solve the coupled linear system:_ 

**==> picture [271 x 32] intentionally omitted <==**

_with_ 

**==> picture [334 x 14] intentionally omitted <==**

## **H.3 Gauss-Seidel Solver for the Coupled System** 

The coupled system (24)–(25) can be solved efficiently via Gauss-Seidel iteration. Starting from **v**[(0)] = **0** : 

**Algorithm 7** Implicit Sinkhorn — Backward Pass (Gauss-Seidel) 

**Require:** Saved **P** _∈Bn_ , upstream gradient _∇_ **P** _L_ **Ensure:** _∇_ **X** _L_ 

1: **H** _←_ **P** _⊙∇_ **P** _L ▷_ Elementwise product 2: **h** row _←_ **H1** _▷_ Row sums, shape ( _. . . , n,_ 1) 3: **h** col _←_ **1** _[⊤]_ **H** _▷_ Column sums, shape ( _. . . ,_ 1 _, n_ ) 4: _k ←_ GaussSeidelIters( _n_ ) _▷_ See Proposition H.2 5: **v** _←_ **0** _▷_ Shape ( _. . . ,_ 1 _, n_ ) 6: **for** _i_ = 1 _, . . . , k_ **do** 7: **u** _←_ **h** row _−_ **Pv** _[⊤] ▷_ Shape ( _. . . , n,_ 1) 8: **v** _←_ **h** col _−_ ( **P** _[⊤]_ **u** ) _[⊤] ▷_ Shape ( _. . . ,_ 1 _, n_ ) 9: **end for** 

10: _∇_ **X** _L ←_ **H** _−_ **u** _⊙_ **P** _−_ **v** _⊙_ **P** 

11: **return** _∇_ **X** _L_ 

## **H.4 Convergence Analysis** 

**Proposition H.2** (Gauss-Seidel Convergence Rate) **.** _For the coupled linear system arising from an n × n doubly-stochastic matrix_ **P** _, the spectral radius of the Gauss-Seidel iteration matrix is bounded by:_ 

**==> picture [252 x 23] intentionally omitted <==**

_To achieve a relative residual ϵ ≤_ 0 _._ 01 _, the required number of iterations is:_ 

**==> picture [302 x 27] intentionally omitted <==**

This yields the following iteration counts for practical values of _n_ : 

For our production configuration with _n_ = 4 streams, _k_ = 16 Gauss-Seidel iterations suffice. We enforce bounds _k ∈_ [10 _,_ 50] as a safety net. 

30 

## **H.5 Self-Stabilization Property** 

An important property of the implicit gradient formula (23) is its **self-stabilization** : the factor **P** _⊙_ ( _·_ ) ensures that entries where _Pij ≈_ 0 automatically produce near-zero gradients, regardless of the accuracy of **u** and **v** . This means no clamping of **P** is needed in the backward pass, and the formula is robust to numerical precision issues in the Sinkhorn forward iterations. 

**Remark H.3.** _Empirically, we verified that clamped (Pij ≥_ 10 _[−]_[8] _) and unclamped versions produce identical gradients (cosine similarity difference <_ 10 _[−]_[6] _) across all tested gradient magnitudes._ 

## **H.6 Fused Projection Optimization** 

All three mapping projections ( _ϕ_ pre _, ϕ_ post _, ϕ_ res) are computed via a single fused linear layer: 

**==> picture [335 x 13] intentionally omitted <==**

where **W** fused _∈_ R[(] _[n]_[+] _[n]_[+] _[n]_[2][)] _[×][nd]_ . This reads the input tensor once instead of three times, reducing memory bandwidth by _∼_ 3 _×_ and cutting kernel launch overhead from 3 to 1. 

## **H.7 Complexity Comparison** 

## **I Cayley Transform Variant** 

The Cayley variant replaces the doubly-stochastic constraint (Birkhoff polytope) with the _orthonormality_ constraint (Stiefel manifold). Instead of requiring **H** _[⊤]_ res **[1]**[=] **[1]**[and] **[H]**[res] **[1]**[=] **[1]**[,][we] require **H** _[⊤]_ res **[H]**[res][=] **[ I]** _[n]_[,][which][provides] **[norm-preserving]**[stream][mixing:] _[∥]_ **[H]**[res] **[x]** _[∥]_[=] _[ ∥]_ **[x]** _[∥]_[.] 

## **I.1 Mathematical Formulation** 

The Cayley transform (18) maps skew-symmetric matrices **W** = _−_ **W** _[⊤]_ to orthogonal matrices. However, the closed-form requires a matrix inverse ( **I** _−_ **W** _/_ 2) _[−]_[1] , which is computationally expensive for batched computation and not amenable to GPU parallelism. Following Li et al. Li et al. [2020b], we use an iterative approximation. 

## **I.2 Iterative Cayley Transform** 

Given an unconstrained matrix **H**[˜] _∈_ R _[n][×][n]_ , the projection proceeds in three steps: 

**Step 1: Skew-Symmetrization.** Ensure the input lies in the Lie algebra of _O_ ( _n_ ): 

**==> picture [262 x 14] intentionally omitted <==**

This guarantees **W** = _−_ **W** _[⊤]_ , which is necessary for the Cayley transform to map to _O_ ( _n_ ). 

**Step 2: Initialization.** Since we start from the identity ( **X** = **I** _n_ ), the initial estimate simplifies: 

**==> picture [265 x 12] intentionally omitted <==**

where _α >_ 0 is a step-size parameter (default _α_ = 0 _._ 1). This saves one matrix multiplication compared to the general case **Y** 0 = **X** + _α_ **WX** . 

31 

**Step 3: Fixed-Point Iteration.** The iterate converges to the Cayley retraction: 

**==> picture [342 x 21] intentionally omitted <==**

The full algorithm is: 

**Algorithm 8** Iterative Cayley Transform Projection 

**Require:** Unconstrained matrix **H**[˜] _∈_ R _[n][×][n]_ , step-size _α_ , iterations _s_ **Ensure:** Approximately orthonormal matrix **Y** _∈_ R _[n][×][n]_ ( **Y** _[⊤]_ **Y** _≈_ **I** ) 1: **W** _←_ **H**[˜] _−_ **H**[˜] _[[⊤]]_ 

1: **W** _←_ **H** _−_ **H** _[[⊤]] ▷_ Skew-symmetrize 2: **Y** _←_ **I** _n_ + _α_ **W** _▷_ Initialize (saves one matmul since **X** = **I** ) 3: **for** _i_ = 1 _, . . . , s_ **do** 4: **Y** _←_ **I** _n_ + _[[α]]_ **[[W]]**[[(]] **[[I]]** _[[n]]_[[ +]] **[[ Y]]**[[)]] _▷_ Fixed-point step via `baddbmm` 

3: **for** _i_ = 1 _, . . . , s_ **do** 4: **Y** _←_ **I** _n_ + _[[α]]_ 2 **[[W]]**[[(]] **[[I]]** _[[n]]_[[ +]] **[[ Y]]**[[)]] 5: **end for** 

- 6: **return Y** 

## **I.3 Properties of the Cayley Projection** 

**Proposition I.1** (Norm Preservation) **.** _For_ **Y** _produced by Algorithm 8, ∥_ **Yx** _∥≈∥_ **x** _∥ for all_ **x** _∈_ R _[n] , with the approximation improving with more iterations s._ 

**Proposition I.2** (Determinant) **.** _The determinant satisfies |_ det( **Y** ) _| ≈_ 1 _, approaching exactness as s →∞._ 

In practice, _s_ = 2 iterations suffice for deep learning applications Li et al. [2020b], achieving orthonormality deviation _∥_ **Y** _[⊤]_ **Y** _−_ **I** _∥_ max _<_ 10 _[−]_[3] . 

## **I.4 CUDA Graph Compatibility** 

A critical implementation detail is **pre-allocation of the identity matrix** . When using `torch.compile(mode=’reduce-overhead’)` , PyTorch captures CUDA graphs that require static tensor shapes. Dynamic calls to `torch.eye()` or `torch.ones()` during forward pass break graph capture. 

We solve this by registering the identity matrix as a persistent buffer during module initialization: 

`self.register_buffer("_identity", torch.eye` ( _n_ ) `, persistent=False)` (33) 

For batched operation (when the batch size _B · L_ is known), we pre-expand: 

**==> picture [368 x 12] intentionally omitted <==**

This ensures the same memory is reused across forward passes, enabling CUDA graph capture. 

## **I.5 Layer Architecture** 

The full Cayley JPmHC layer computes three mapping matrices per token: 

1. **Fused projection** : A single linear layer produces all three unconstrained matrices: 

**==> picture [334 x 15] intentionally omitted <==**

where **W** fused _∈_ R[3] _[n]_[2] _[×][nd]_ and the output is split and reshaped to ( _B · L, n, n_ ) for each. 

32 

2. **Constraint projection** : 

**==> picture [309 x 50] intentionally omitted <==**

Note that **H** pre and **H** post use softmax (row-stochastic and column-stochastic respectively, with temperature _τ_ ) rather than sigmoid, since the Cayley variant operates on full _n × n_ mixing matrices for pre/post. 

3. **Forward computation** : 

**==> picture [308 x 78] intentionally omitted <==**

The final combination uses `torch.baddbmm` for a fused residual-plus-write operation, avoiding a separate `bmm` followed by addition. 

## **I.6 Comparison: Doubly-Stochastic vs. Orthonormal Mixing** 

The norm-preserving property of orthonormal mixing is particularly beneficial for recursive models where the same layer is applied multiple times—it prevents representation collapse or explosion across recursion steps. 

## **J Grassmannian Variant** 

The Grassmannian variant provides a **parameter-efficient** alternative to both Sinkhorn and Cayley by learning a rank- _p_ subspace projection instead of a full _n × n_ mixing matrix. The residual mapping is represented as: 

**==> picture [329 x 14] intentionally omitted <==**

where _p ≤ n_ (default _p_ = _⌊n/_ 2 _⌋_ ). This projector is **idempotent** ( **H**[2] res[=] **[H]**[res][),] **[symmetric]** ( **H** res = **H** _[⊤]_ res[),][and][has] **[rank][exactly]** _[p]_[.] 

## **J.1 Parameter Efficiency** 

The key advantage is the reduction in parameters for the residual mixing: 

While the savings are modest for _n_ = 4, they become significant for larger _n_ and when accumulated across all layers in a recursive model. 

## **J.2 Riemannian Optimization via Cayley ADAM** 

Standard gradient descent in Euclidean space does not preserve the Stiefel constraint **U** _[⊤]_ **U** = **I** _p_ . We employ a full Riemannian optimization scheme that combines **horizontal projection** , **momentum** , and **Cayley retraction** . 

33 

## **J.2.1 Step 1: Horizontal Projection** 

On the Grassmann manifold Gr( _n, p_ ) = St( _n, p_ ) _/O_ ( _p_ ), the tangent space at **U** decomposes into: 

**==> picture [313 x 30] intentionally omitted <==**

The vertical space corresponds to rotations _within_ the subspace (i.e., right multiplication by _O_ ( _p_ )), which do not change the projector **UU** _[⊤]_ . The horizontal projection removes this component: 

**==> picture [343 x 14] intentionally omitted <==**

This is essential for Grassmannian (as opposed to Stiefel) optimization: it ensures we move in directions that actually change the subspace, not just rotate the basis within it. 

## **J.2.2 Step 2: Cayley ADAM Momentum** 

We maintain exponential moving averages of the horizontally-projected gradient: 

**==> picture [301 x 13] intentionally omitted <==**

**==> picture [296 x 15] intentionally omitted <==**

where _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 999, and _∇_[2] hor _,t_[denotes][elementwise][squaring.] 

With optional adaptive learning rate (analogous to Adam Kingma and Ba [2015]), the scaled momentum is: 

**==> picture [286 x 30] intentionally omitted <==**

Without adaptive scaling (the default), we simply use **M**[ˆ] _t_ = **M** _t_ . 

## **J.2.3 Step 3: Skew-Symmetric Direction** 

The Cayley retraction requires a skew-symmetric direction matrix: 

**==> picture [335 x 15] intentionally omitted <==**

This **W** defines a curve on the Stiefel manifold through the current point **U** . 

## **J.2.4 Step 4: Iterative Cayley Retraction** 

The Cayley retraction maps the tangent vector back to the manifold without an explicit matrix inverse: 

**==> picture [336 x 39] intentionally omitted <==**

After _s_ iterations (typically _s_ = 2), we update **U** _←_ **Y** _s_ . 

## **J.2.5 Step 5: QR Retraction (Post-Step Correction)** 

For robustness under DDP gradient synchronization and mixed-precision training, we also apply a QR retraction after each optimizer step: 

**==> picture [304 x 28] intentionally omitted <==**

34 

The sign correction ensures a unique representative on the Stiefel manifold (the _Q_ -factor from QR is unique up to column sign flips). 

The complete Riemannian optimization step is given in Algorithm 9. 

**Algorithm 9** Cayley ADAM Riemannian Optimization Step 

**Require:** Current basis **U** _∈_ St( _n, p_ ), Euclidean gradient _∇_ **U** _L_ **Require:** Momentum state **M** _,_ **v** _, t_ ; hyperparameters _α, β_ 1 _, β_ 2 _, ϵ_ **Ensure:** Updated **U** _∈_ St( _n, p_ ) 1: _t ← t_ + 1 2: _▷ Step 1: Horizontal projection_ 3: _∇_ hor _←∇_ **U** _L −_ **U** ( **U** _[⊤] ∇_ **U** _L_ ) 4: _▷ Step 2: Momentum update_ 5: **M** _← β_ 1 **M** + (1 _− β_ 1) _∇_ hor 6: **v** _← β_ 2 **v** + (1 _− β_ 2) _∇_[2] hor _▷_ Optional adaptive LR 7: **if** adaptive LR **then** ˆ **M** _/_ (1 _−β_ 1 _[t]_[)] 8: **M** _←_ ~~_√_~~ **v** _/_ (1 _−β_ 2 _[t]_[)+] _[ϵ]_ 9: **else** 10: **M** ˆ _←_ **M** 11: **end if** 12: _▷ Step 3: Skew-symmetric direction_ 13: **W** _←_ **MU**[ˆ] _[⊤] −_ **UM**[ˆ] _[⊤]_ 14: _▷ Step 4: Iterative Cayley retraction_ 15: **Y** _←_ **U** + _α_ **M**[ˆ] 16: **for** _i_ = 1 _, . . . , s_ **do** 17: **Y** _←_ **U** + _[α]_ 2 **[W]**[(] **[U]**[ +] **[ Y]**[)] 18: **end for** 19: **U** _←_ **Y** 20: _▷ Step 5: QR retraction (optional, for DDP compatibility)_ 21: **Q** _,_ **R** _←_ QR( **U** ) 

22: **U** _←_ **Q** _·_ diag(sign(diag( **R** ))) 

## **J.3 DDP/DeepSpeed Integration** 

The Grassmannian optimization is implemented as a **post-step optimizer wrapper** ( `GrassmannianOptimizer` that: 

1. Automatically discovers all `GrassmannianProjection` modules in the model (handling DDP/DeepSpeed wrapper unwrapping). 

2. After each standard optimizer step ( `optimizer.step()` ), applies QR retraction to project **U** back onto St( _n, p_ ). 

3. Maintains its own state dict for momentum ( **M** ), second moment ( **v** ), and time step ( _t_ ), supporting full checkpoint save/load. 

4. All tensor operations use in-place ops and avoid CPU-GPU synchronization (the time step _t_ is kept as a GPU tensor to avoid `.item()` calls that trigger sync). 

## **J.4 Geometric Interpretation** 

The Grassmannian projector **H** res = **UU** _[⊤]_ filters the _n_ -dimensional residual stream through a learned _p_ -dimensional subspace. Geometrically: 

- Components of **x** _within_ the subspace col( **U** ) are preserved. 

35 

- Components _orthogonal_ to the subspace are projected to zero. 

- The sublayer output is added back, potentially reintroducing orthogonal components. 

This provides an implicit form of **information bottleneck** in the residual stream, where the model learns which _p_ directions carry the most useful information across layers. 

## **K Computational Pipeline Overview** 

The spectral density of the end-to-end Jacobian _J_ = _YL · · · Y_ 1 is computed through a modular pipeline comprising four stages: 

1. **Signal propagation.** A forward recursion determines the per-layer operating point (preactivation variance _q[ℓ]_ , post-activation variance _v[ℓ]_ , and self-energy _e[ℓ]_ ), following the meanfield theory of Schoenholz et al. [2017], Pennington et al. [2017]. 

2. **Per-layer Dyson equation.** For each layer _ℓ_ , the Stieltjes transform _Gℓ_ ( _z_ ) of the squared singular-value distribution of _Yℓ_ = _Aℓ_ + _σℓ_[2] _[X][ℓ]_[is obtained from a self-consistent subordi-] ~~�~~ 

nation equation Tarnowski et al. [2019], Voiculescu [1991]. In the scalar case ( _q_ = 1) this reduces to a single complex equation; for Kronecker-structured skip connections ( _A_ = _Aq⊗Ip_ ) it becomes a _q × q_ matrix equation, following the operator-valued framework of Speicher [1998], Belinschi et al. [2017], Helton et al. [2007]. 

3. **Free multiplicative convolution.** The per-layer Stieltjes transforms are composed to obtain the _L_ -layer transform _GL_ ( _z_ ). Three composition strategies are available: the _z_ 1- mapping for identical layers (derived from the S-transform of Voiculescu [1991], Voiculescu et al. [1992]), the Belinschi–Speicher subordination iteration for heterogeneous scalar layers Belinschi et al. [2015], and the Dykema twisted multiplicativity theorem for noncommuting operator-valued layers Dykema [2006]. 

4. **Spectral inversion.** The density is recovered via the Stieltjes inversion formula, optionally refined by Richardson extrapolation Richardson [1911] or AAA rational approximation Nakatsukasa et al. [2018]. 

The appropriate solver is selected automatically based on the twist dimension _q_ , depth _L_ , layer homogeneity, and available hardware. 

## **L Activation Functions and Signal Propagation** 

The activation function _ϕ_ and its derivative _ϕ[′]_ enter the theory through three Gaussian moments, computed by Gauss–Hermite quadrature Golub and Welsch [1969] with 100 nodes: 

**==> picture [298 x 51] intentionally omitted <==**

The function _ψ_ controls the self-energy (effective noise variance) and _κ_ the second moment of the layer output. The effective cumulant appearing in the Dyson equation is _c_ 2 = _σw_[2] _[ψ]_[(] _[q]_[)][.][These] moments are well-defined for standard activation functions including ReLU, tanh, sigmoid, and leaky-ReLU; see Pennington and Worah [2017] for the general framework and Tarnowski et al. [2019] for the residual-network specialisation. 

36 

The operating point of each layer is determined by a forward recursion over the pre-activation variance _q[ℓ]_ , the post-activation variance _v[ℓ]_ , and the self-energy _e[ℓ]_ Schoenholz et al. [2017], Yang and Schoenholz [2017], Tarnowski et al. [2019]: 

**==> picture [286 x 32] intentionally omitted <==**

**==> picture [287 x 27] intentionally omitted <==**

with initial conditions _v_[0] (the input variance) and _e_[0] = _ψ_ ( _σw,_[2] 1 _[v]_[0][)][.][The][per-layer][Dyson][self-] energy is then _σℓ_[2][=] _[ σ] w,ℓ_[2] _[ψ]_[(] _[q][ℓ]_[)][.][The recursion accepts either a shared or per-layer skip matrix] _[ A][ℓ]_ and weight standard deviation _σw,ℓ_ . 

## **M Scalar Dyson Equation Solver** 

## **M.1 Subordination form** 

Under the isotropy assumption, the Stieltjes transform _G_ ( _ζ_ ) = _p_[1][Tr] �( _ζI − Y[T] Y_ ) _[−]_[1][�] of a single layer _Y_ = _A_ 0 + _√σ_[2] _X_ satisfies the subordination equation Tarnowski et al. [2019], Voiculescu [1991], Marchenko and Pastur [1967] 

**==> picture [364 x 31] intentionally omitted <==**

where _{si}[p] i_ =1[are the eigenvalues of] _[ A]_ 0 _[T][A]_[0][, precomputed once in] _[ O]_[(] _[p]_[3][)][ time.][This is an instance] of the general subordination phenomenon in free probability Biane [1998], Belinschi and Bercovici [2007]. 

## **M.2 Newton iteration** 

**==> picture [452 x 38] intentionally omitted <==**

The analytical Jacobian is 

**==> picture [344 x 32] intentionally omitted <==**

Newton’s method is applied with Armijo backtracking Armijo [1966], Nocedal and Wright [2006] using step sizes _α ∈{_ 1 _,_[1] 2 _[,]_ 4[1] _[,]_[1] 8 _[}]_[to][prevent][divergence][near][spectral][edges.] 

## **M.3 Batch solver** 

The batch solver uses a two-pass strategy: 

1. **Sequential continuation pass** : sweep the _z_ -grid from right to left (large to small _|z|_ ), using the converged solution at point _j_ as the initial guess for point _j_ +1. This continuation technique Nocedal and Wright [2006] provides a good warm start in approximately 5 Newton iterations per point. 

2. **Vectorised refinement** : apply Newton iterations to all grid points simultaneously in a batched array operation until global convergence. 

37 

## **M.4 Woodbury acceleration** 

When _A_ 0 = _aI_ + _UV[T]_ has low-rank perturbation of rank _r_ , the Woodbury matrix identity Woodbury [1950] allows precomputation of the eigenvalues of _A[T]_ 0 _[A]_[0][in] _[O]_[(] _[pr]_[2][)][time,][reducing][setup] cost from _O_ ( _p_[3] ) for large _p_ . 

## **N Operator-Valued Dyson Solver** 

For Kronecker-structured skip connections _A_ = _Aq ⊗ Ip_ with _Aq ∈_ R _[q][×][q]_ and _N_ = _qp_ , the conditional expectation _EB_ = id _q ⊗_ tr _p_ reduces the Dyson equation to a _q × q_ matrix problem within the operator-valued free probability framework Speicher [1998], Voiculescu [1995], Belinschi et al. [2017]. 

## **N.1 Scalar subordination path** 

Because the noise is isotropic in the _p_ -directions, the self-energy is proportional to _Iq_ : Σ = _σ_[2] _G_ scalar _Iq_ , where _G_ scalar =[1] _q_[Tr(] _[G]_[(] _[B]_[)][)][.][The] _[q][ ×][ q]_[Green’s][function][is][then] 

**==> picture [357 x 17] intentionally omitted <==**

with _u_ = 1 _− σ_[2] _G_ scalar and _ω_ = _z u_[2] . This is a scalar subordination equation in _G_ scalar with the same structure as (61), but summing over the _q_ eigenvalues of _A[T] q[A][q]_[.] 

## **N.2 Matrix spectral parameter** 

For general _b ∈ Mq_ (C) (needed by the Ψ-inversion in the operator-valued S-transform, cf. §O.2), a 2-scalar Schur complement iteration solves for the self-energies ( _g_ 11 _, g_ 22): 

**==> picture [321 x 33] intentionally omitted <==**

where _g_ 11 =[1][and] _[g]_[22][is][similarly][defined.][This][Schur][reduction,][inspired][by][the][fixed-] _q_[Tr(] _[G]_[(] _[B]_[)][)] point characterisation of Helton et al. [2007], costs _O_ ( _q_[3] ) per iteration (dominated by the _q × q_ matrix inverse), compared to _O_ ( _q_[6] ) for a full _q_[2] -dimensional Newton. Typically 5–30 iterations with adaptive damping suffice. 

## **O S-Transform and Free Multiplicative Convolution** 

## **O.1 Scalar S-transform** 

The S-transform, introduced by Voiculescu [1991], Voiculescu et al. [1992], is defined implicitly by 

**==> picture [294 x 26] intentionally omitted <==**

and linearises free multiplicative convolution Bercovici and Voiculescu [1993], Nica and Speicher [2006]: 

**==> picture [285 x 12] intentionally omitted <==**

The computational procedure involves three steps: 

1. **Cauchy-to-S** : compute _S_ ( _w_ ) from _G_ ( _z_ ) by solving _w_ = _zG_ ( _z_ ) _−_ 1 for _z_ via Newton’s method with continuation threading. 

38 

2. **S-product** : form the pointwise product[�] _ℓ[S][ℓ]_[(] _[w]_[)][.] 

3. **S-to-Cauchy** : recover _G_ ( _z_ ) from _S_ ( _w_ ) by solving the implicit equation _z S_ ( _w_ ) _w−w−_ 1 = 0 for _w_ given _z_ . 

See Rao and Speicher [2007], Haagerup and Möller [2005] for analytic properties of the S- transform relevant to the vanishing-mean case. 

## **O.2 Operator-valued S-transform (Dykema)** 

For the subalgebra _B_ = _Mq_ (C), the operator-valued S-transform Dykema [2006], Voiculescu [1995] is defined via the Ψ-inversion problem: given _W ∈ Mq_ , find _b ∈ Mq_ such that 

**==> picture [345 x 15] intentionally omitted <==**

Dykema’s twisted multiplicativity theorem (Dykema [2006], Theorem 1.1) gives the composition rule for non-commuting layers: 

**==> picture [348 x 15] intentionally omitted <==**

For _L_ layers, the cumulative S-transform is built by folding from layer _L_ (innermost) to layer 1 (outermost): 

**==> picture [426 x 31] intentionally omitted <==**

This twisted fold is the operator-valued analogue of the scalar product (67) and reduces to it when all _Aq,ℓ_ commute. The non-commutativity of the twisted product is essential for capturing the eigenspace misalignment effect described in the main text. 

## **P Multi-Layer Composition: Identical Layers** 

When all _L_ layers are identical (same _A_ and _σ_[2] ), the _L_ -layer Stieltjes transform _GL_ ( _zL_ ) is related to the single-layer _G_ 1( _z_ 1) by the subordination (“ _z_ 1-mapping”) formula, derived from the S-transform identity (67) Burda et al. [2010], Tarnowski et al. [2019]: 

**==> picture [305 x 28] intentionally omitted <==**

**==> picture [282 x 16] intentionally omitted <==**

This avoids the numerically fragile S-transform round-trip _G → S → S[L] → G_ . 

39 

**Algorithm 10** _z_ 1-mapping for identical layers 

**Require:** Single-layer solver for _G_ 1( _z_ ), depth _L_ , grid _{zj}[n] j_ =1[with][Im(] _[z][j]_[)] _[ >]_[ 0] **Ensure:** _GL_ ( _zj_ ) for _j_ = 1 _, . . . , n_ 1: _z_ 1 _,_ prev _←_ None 2: **for** _j_ = 1 _, . . . , n_ **do** 3: Construct multi-start guesses: _z_ 1 _,_ prev, geometric range _zj_[1] _[/L] , . . . , zj_ , fixed heuristics 4: **for** each guess _z_ 1[(0)] **do** 5: **if** _L ≥_ 5 **then** 6: Set _w ←_ log( _z_ 1) _▷_ Log parameterisation 7: Newton on _f_ log( _w_ ) = _w_ + ( _L−_ 1) log _ω −_ log _zj_ 8: **else** 9: Newton on _f_ ( _z_ 1) = _z_ 1 _ω[L][−]_[1] _− zj ▷ω_ = _z_ 1 _−_ 1 _/G_ 1( _z_ 1) 10: **end if** 11: Armijo backtracking: halve step until _|f | < |f_ prev _|_ and Im( _z_ 1) _>_ 0 12: **end for** 13: Select _z_ 1 _[∗]_[with][smallest][residual] 14: _GL_ ( _zj_ ) _← G_ 1( _z_ 1 _[∗]_[)] _[L][ /]_[ (] _[z]_ 1 _[∗][G]_[1][(] _[z]_ 1 _[∗]_[)] _[ −]_[1)] _[L][−]_[1] 15: _z_ 1 _,_ prev _← z_ 1 _[∗] ▷_ Continuation seed 16: **end for** 

17: Interpolate isolated unconverged points (linear in Re/Im) 

The analytical Jacobian _df/dz_ 1 is obtained via implicit differentiation on the Dyson equation: 

**==> picture [268 x 27] intentionally omitted <==**

where _F_ ( _G, z_ ) = 0 is the Dyson residual (62), eliminating finite-difference approximations. 

## **Q Multi-Layer Composition: Heterogeneous Layers** 

## **Q.1 Subordination iteration (Belinschi–Speicher)** 

For heterogeneous layers with distinct _Aℓ_ and _σℓ_[2][,][the][composition][uses][the][subordination][it-] eration of Belinschi, Speicher, Treilhard, and Vargas Belinschi et al. [2015]. For each spectral parameter _z_ , the algorithm iterates in the _w_ -domain: 

**==> picture [284 x 28] intentionally omitted <==**

where _Sℓ_ ( _w_ ) = _Gℓ_ ( _zℓ_ ) _/w_ with _zℓ_ satisfying _zℓGℓ_ ( _zℓ_ ) _−_ 1 = _w_ . Newton’s method on _w_ drives _zL_ ( _w_ ) _→ z_ . 

40 

**Algorithm 11** Subordination for heterogeneous layers 

**Require:** Per-layer solvers _{Gℓ_ ( _·_ ) _}[L] ℓ_ =1[,][grid] _[{][z][j][}]_ **Ensure:** _GL_ ( _zj_ ) for each _j_ 

1: Sweep _z_ -grid from large to small (continuation) 

2: **for** each _z_ = _zj_ **do** 3: Initialise _w_ from previous converged point or _w ← z G_ 1( _z_ ) _−_ 1 4: **for** iter = 1 _, . . . , N_ max **do** 5: **for** _ℓ_ = 1 _, . . . , L_ **do** 6: Solve _zℓGℓ_ ( _zℓ_ ) _−_ 1 = _w_ for _zℓ ▷_ Scalar Newton 7: _Sℓ_ ( _w_ ) _← Gℓ_ ( _zℓ_ ) _/w_ 8: **end for** 9: _zL_ ( _w_ ) _←_ ( _w_ + 1) _/_ ( _w_[�] _ℓ[S][ℓ]_[(] _[w]_[))] 10: **if** _|zL_ ( _w_ ) _− z| <_ tol _·_ max( _|z|,_ 1) **then** 11: _GL_ ( _z_ ) _←_ ( _w_ + 1) _/z_ ; **break** 12: **end if** 13: Newton update: _w ← w −_ ( _zL_ ( _w_ ) _− z_ ) _/_ ( _dzL/dw_ ) _▷_ FD derivative 14: **end for** 15: **end for** 

Convergence is probed on a small subset (10 uniformly-spaced points); if _≥_ 80% converge, the subordination iteration is run on the full grid. Otherwise, the algorithm falls back to the S-transform round-trip _Gℓ → Sℓ →_[�] _Sℓ → GL_ . This fallback exploits the analyticity properties established in Belinschi and Bercovici [2007], Bercovici and Voiculescu [1993]. 

## **Q.2 Anderson/DIIS acceleration** 

Fixed-point iterations for subordination are optionally accelerated by Anderson mixing Anderson [1965], Walker and Ni [2011], also known as DIIS (Direct Inversion in the Iterative Subspace) Pulay [1980] in the computational chemistry literature. Given a history of iterates _{xk}_ and residuals _{rk}_ with _rk_ = _g_ ( _xk_ ) _− xk_ , the accelerated estimate minimises _∥_[�] _i[c][i][r][i][∥]_[2][subject] to[�] _i[c][i]_[=][1][,][solved][via][an][(] _[m]_[+1)] _[ ×]_[ (] _[m]_[+1)][linear][system][with][Lagrange][multiplier][(history] depth _m_ = 5). 

## **R Operator-Valued Multi-Layer Pipeline** 

For Kronecker-structured heterogeneous layers with non-commuting _Aq,ℓ_ , the full operatorvalued pipeline computes _G_[(] _L[B]_[)][(] _[z]_[)] _[ ∈][M][q]_[(][C][)][via][a][triple-nested][Newton][structure:] 

1. **Outer Newton** (over _W ∈ Mq_ , _q_[2] complex unknowns): solve 

**==> picture [325 x 16] intentionally omitted <==**

where _S_ prod( _W_ ) is the twisted S-product (70). 

2. **Middle loop** : evaluates the twisted fold (70) over _L_ layers, applying Dykema’s composition rule (69) at each step. 

3. **Inner Newton** (per-layer Ψ-inversion): for each layer, solves the Ψ-inversion problem (68) _b G_[(] _[B]_[)] ( _b_ ) _− Iq − W_ twisted = 0 for _b ∈ Mq_ via Newton’s method with multiple initial guesses. Each iteration requires a matrix Dyson solve (§N). 

Convergence is managed via: 

41 

1. **Analytical Jacobians** (where feasible) with finite-difference fallback (step _h_ = 10 _[−]_[7] ); see Nocedal and Wright [2006] for the general theory of inexact Newton methods. 

2. **Damped Newton with Armijo backtracking** Armijo [1966]: _α ∈{_ 1 _,_ 0 _._ 5 _,_ 0 _._ 25 _,_ 0 _._ 1 _,_ 0 _._ 05 _,_ 0 _._ 02 _,_ 0 _._ 01 _}_ . 

3. **Continuation threading** : solutions ( _W, bℓ, Mℓ_ ) from the previous _z_ -point warm-start the next, reducing per-layer Newton iterations from _∼_ 20 (cold) to _∼_ 3–5 (warm). 

4. **Per-layer caching** : each layer carries its converged ( _b, M_ ) pair through all nesting levels, enabling efficient re-use across the middle-loop evaluations. 

For large spectral grids, the _z_ -points are sorted in descending order and partitioned into chunks, with a sequential pre-sweep seeding every _K_ -th chunk boundary. Chunks are then processed in parallel with work-stealing scheduling. 

## **S Spectral Density Recovery** 

## **S.1 Stieltjes inversion** 

The spectral density is recovered from the Stieltjes transform via the standard inversion formula Anderson et al. [2010], Bai and Silverstein [2010]: 

**==> picture [291 x 24] intentionally omitted <==**

followed by clipping to [0 _, ∞_ ) and normalisation to unit integral. 

## **S.2 Richardson extrapolation** 

To sharpen spectral edges without reducing _η_ to the point of numerical instability, a Nevilletableau Richardson extrapolation Richardson [1911] is applied. Densities are computed at _ηk_ = _η_ base _·_ 2 _[k]_ for _k_ = 0 _, . . . , n_ levels _−_ 1, then combined via 

**==> picture [337 x 25] intentionally omitted <==**

assuming _O_ ( _η_[2] ) error scaling. The final estimate is _Tn_ levels _−_ 1 _, n_ levels _−_ 1. 

## **S.3 AAA rational approximation** 

An alternative high-accuracy path fits a barycentric rational approximant to _G_ ( _z_ ) via the AAA algorithm Nakatsukasa et al. [2018] on Chebyshev nodes, then evaluates the approximant on the full grid. This avoids grid-based artifacts and provides uniform accuracy near spectral edges with typically _O_ (20–50) evaluations of _G_ . 

## **T GPU-Accelerated Solvers** 

All solvers described above admit natural GPU parallelisation over the spectral grid _{zj}_ , since each _z_ -point involves an independent fixed-point problem. The GPU implementation provides: 

1. **Batched scalar Dyson solver** : the subordination equation (61), Newton updates, and Armijo backtracking are expressed as batched tensor operations, processing all _z_ -points simultaneously. 

2. **Batched operator-valued Dyson solver** : the Schur complement iteration (65) is parallelised over _z_ -points with batched _q × q_ matrix inversions. 

42 

3. **Multi-layer pipeline** : the two-pass strategy (continuation + refinement) for identical layers, and the _w_ -domain Newton for heterogeneous layers, are adapted with per-chunk GPU batching and work-stealing. 

Memory management uses heuristic batch sizing based on available GPU memory, with automatic fallback to smaller batches upon memory exhaustion. 

## **U Monte Carlo Validation** 

Theoretical predictions are validated against direct Monte Carlo simulation of the random matrix product: 

1. Construct _n_ samples realisations of _J_ = _YL · · · Y_ 1 with _Yℓ_ = _Aℓ_ + _DℓWℓ_ , where _Wℓ_ has i.i.d. standard Gaussian entries and _Dℓ_ = _σℓI_ . 

2. Compute eigenvalues of _J[T] J_ for each realisation. 

3. Pool all eigenvalues and estimate the density via Gaussian KDE or histogram. 

4. Compare to the theoretical prediction using _L_[1] , _L_[2] , _L[∞]_ , and Kolmogorov–Smirnov metrics. 

The GPU variant uses batched eigendecomposition with automatic batch-size estimation based on available VRAM. As the network width _N →∞_ , the empirical spectral distribution converges to the theoretical prediction by the concentration of measure phenomenon Ledoux [2001], Anderson et al. [2010]. 

## **V Operator-Valued Spectral Density Panel** 

The validation panel (Figure 1) displays a 3 _×_ 4 grid comparing the theoretical spectral density with Monte Carlo simulation for Kronecker-structured skip connections _A_ = _Aq ⊗ Ip_ across depths _L ∈{_ 1 _,_ 2 _,_ 10 _}_ and four families of _q × q_ twist matrices: identity, random bistochastic, random Haar-orthogonal, and normalised Gaussian. The full pipeline is detailed in Algorithm 12. 

All panels use _q_ = 4, _p_ = 25 (so _N_ = _qp_ = 100), and a fixed noise budget _L · σ_[2] = 0 _._ 05, giving per-layer self-energy _σℓ_[2][= 0] _[.]_[05] _[/L]_[.][The imaginary regularisation is] _[ η]_[= 0] _[.]_[02][.][Monte Carlo] sample counts are _n_ samples _∈{_ 300 _,_ 200 _,_ 100 _}_ for _L ∈{_ 1 _,_ 2 _,_ 10 _}_ respectively. 

## **V.1 Operator-valued z** 1 **-mapping** 

For identical layers with _L >_ 1, the OV multi-layer composition exploits a key structural property: because the noise is isotropic in the _p_ -directions, the subordination variable _b_ 1 remains on the scalar manifold _b_ 1 = _z_ 1 _· Iq_ . The scalar subordination equation 

**==> picture [370 x 18] intentionally omitted <==**

is solved for _z_ 1 by multi-start Newton iteration (as in Algorithm 10, using the analytical derivatives from the OV Dyson solver §N), and the _L_ -layer _q × q_ Green’s function is reconstructed via eigendecomposition: 

**==> picture [343 x 17] intentionally omitted <==**

where the matrix powers are computed via eigendecomposition of the _q × q_ matrices _G_[(] 1 _[B]_[)] and _z_ 1 _G_[(] 1 _[B]_[)] _− Iq_ . 

43 

## **V.2 Panel generation algorithm** 

For clarity, Algorithm 12 describes the computation for a single twist matrix _Aq_ and depth _L_ with _L identical_ (homogeneous) layers sharing the same _Aq_ and _σ_[2] . The full 3 _×_ 4 panel is obtained by repeating this procedure over _L ∈{_ 1 _,_ 2 _,_ 10 _}_ and _Aq ∈{_ identity, bistochastic, orthogonal, Gaussian _}_ . 

**Algorithm 12** OV spectral density: single cell (homogeneous layers) 

**Require:** Twist matrix _Aq_ ( _q × q_ ), depth _L_ , width _p_ , self-energy _σ_[2] = _c/L_ , regularisation _η_ **Ensure:** Singular-value density plot with theory vs. MC overlay 

1: _A_ full _← Aq ⊗ Ip ▷N × N_ Kronecker expansion _— Monte Carlo phase —_ 2: **for** _s_ = 1 _, . . . , n_ samples **do** 3: _Y ←_[�][1] _ℓ_ = _L_[(] _[A]_[full][ +] _√σ_[2] _Ws,ℓ_ ) _▷Ws,ℓ_ : i.i.d. _N_ (0 _,_ 1 _/N_ ) entries 4: Record eigenvalues of _Y[T] Y_ 5: **end for** 6: Pool all eigenvalues _→_ empirical distribution _— Adaptive grid calibration —_ 7: _x_ max _←_ max�1 _._ 5 _·_ max(MC eigenvalues) _,_ 10� 8: _x_ -grid _←_ uniform [0 _._ 01 _, x_ max], max(400 _, ⌊_ 40 _x_ max _⌋_ ) points _— Theory phase (OV Dyson + z_ 1 _-mapping) —_ 9: Precompute eigenvalues of _A[T] q[A][q] ▷O_ ( _q_[3] ), once 10: _z_ -grid _← x_ -grid + _iη_ 11: **if** _L_ = 1 **then** 12: _{G_ scalar _,j} ←_ OV-Dyson( _Aq, σ_[2] _, z_ -grid) _▷_ §N 13: **else** 14: _{G_ scalar _,j} ←_ OV- _z_ 1-Mapping( _Aq, σ_[2] _, L, z_ -grid) _▷_ Alg. 10 / (79) 15: **end if** _— — Spectral density recovery_ 16: _ρ_ ( _xj_ ) _←− π_[1][Im] � _G_ scalar _,j_ �; clip to [0 _, ∞_ ); normalise � _ρ_ = 1 _— Change of variables —_ 17: _ρ_ sv( _σ_ ) _←_ 2 _σ ρ_ ( _σ_[2] ); renormalise _— Validation —_ 18: _W_ 1 _←_ Wasserstein-1(theory, MC) in eigenvalue domain 19: Plot: MC histogram + theory curve _ρ_ sv; annotate _W_ 1 

## **V.3 Heterogeneous layers** 

When layers have _distinct_ (non-commuting) twist matrices _Aq,_ 1 _, . . . , Aq,L_ with per-layer selfenergies _σ_ 1[2] _[, . . . , σ] L_[2][,][the] _[z]_[1][-mapping][of][Algorithm][12][no][longer][applies.][Two][operator-valued] composition methods are available, selected automatically by the dispatcher (§X): 

**OV subordination (default).** The scalar subordination variable _w_ still lives in C (isotropic noise), so the per-layer _S_ -products remain scalar. Algorithm 13 details the procedure. 

**Twisted S-transform (fallback).** For validation or when the subordination Newton diverges, the full Dykema twisted fold _S_ [[(] _L[B]_ ][)][(] _[W]_[)][=] _[S][L]_[(] _[W]_[)] _[ ·][ S][L][−]_[1][(] _[S] L[−]_[1] _[WS][L]_[)] _[ · · ·]_[is][used,][with][an][outer] Newton over _W ∈ Mq_ (C) ( _q_[2] complex unknowns). 

44 

**Algorithm 13** OV spectral density: single cell (heterogeneous layers) 

**Require:** Twist matrices _Aq,_ 1 _, . . . , Aq,L_ ( _q × q_ ), per-layer self-energies _σ_ 1[2] _[, . . . , σ] L_[2][,][width] _[p]_[,][reg-] ularisation _η_ **Ensure:** Singular-value density plot with theory vs. MC overlay 1: _A_ full _,ℓ ← Aq,ℓ ⊗ Ip_ for _ℓ_ = 1 _, . . . , L — Monte Carlo phase —_ 2: **for** _s_ = 1 _, . . . , n_ samples **do** 3: _Y ←_[�][1] _ℓ_ = _L_[(] _[A]_[full] _[,ℓ]_[+] _σℓ_[2] _[W][s,ℓ]_[)] ~~�~~ 4: Record eigenvalues of _Y[T] Y_ 5: **end for** 6: Pool all eigenvalues _→_ empirical distribution _— Adaptive grid & theory phase —_ 7: Calibrate _x_ -grid from MC support (as in Alg. 12, lines 8–9) 8: Build per-layer solvers: scalar _Dℓ_ = DysonSolver( _Aq,ℓ, σℓ_[2][)][,] matrix _Mℓ_ = MatrixDysonSolver( _Aq,ℓ, σℓ_[2][)] _— OV subordination (sweep large z to small) —_ 9: **for** each _zj_ in _z_ -grid (descending, with continuation) **do** 10: Initialise _w_ from previous point (or bootstrap _w_ = _zjG_ 1( _zj_ ) _−_ 1) 11: **repeat** 12: **for** _ℓ_ = 1 _, . . . , L_ **do** 13: Solve _zℓGℓ_ ( _zℓ_ ) _−_ 1 = _w_ for _zℓ ▷_ Newton 14: _Sℓ ← Gℓ_ ( _zℓ_ ) _/ w_ 15: **end for** 16: _zL_ ( _w_ ) _←_ ( _w_ + 1) _/_ ( _w ·_[�] _ℓ[S][ℓ]_[)] 17: Newton update: _w ← w −_ � _zL_ ( _w_ ) _− zj_ � _/_ � _∂zL/∂w_ � _▷_ FD derivative 18: **until** _|zL_ ( _w_ ) _− zj| <_ tol 19: _G_ scalar _,j ←_ ( _w_ + 1) _/ zj_ 20: **end for** _— Spectral density recovery & plotting —_ 21: _ρ, ρ_ sv _, W_ 1, plot (as in Alg. 12, lines 16–19) 

When _q_ = 1, the pipeline reduces to the scalar _z_ 1-mapping of §P. 

## **W Stochastic Trace Estimation** 

For large-dimensional problems where explicit eigendecomposition is prohibitive, the Hutch++ algorithm Meyer et al. [2021] provides stochastic trace estimation: 

**==> picture [364 x 31] intentionally omitted <==**

where _Q_ is obtained from the QR decomposition of _A_ Ω for a random Gaussian matrix Ω _∈_ R _[n][×][k]_ , and _{gi}_ are i.i.d. complex Gaussian vectors (normalised to unit norm). Applied to the resolvent _A_ = ( _zI − M_ ) _[−]_[1] , this yields stochastic estimates of Tr�( _zI − M_ ) _[−]_[1][�] = _N G_ ( _z_ ), enabling Stieltjes transform computation without full diagonalisation. 

## **X Solver Selection** 

The computational pipeline automatically selects the appropriate solver based on problem parameters: 

45 

|**Condition**|**Method**|
|---|---|
|_q_ = 1, _L_= 1|Scalar Dyson (§M)|
|_q_ = 1, _L >_1, identical layers|_z_1-mapping (§P)|
|_q_ = 1, _L >_1, heterogeneous layers|Subordination iteration (§Q)|
|_q >_1, _L_= 1|Operator-valued Dyson (§N)|
|_q >_1, _L >_1|OV multi-layer pipeline (§R)|
|GPU available|GPU-accelerated variants (§T)|



## **Y Numerical Stability Techniques** 

Several techniques are employed throughout the pipeline to ensure numerical stability: 

1. **Continuation threading** : every grid-based solver seeds each new _z_ -point with the converged solution from the previous point Nocedal and Wright [2006], reducing Newton iterations from _∼_ 20 to _∼_ 3–5. 

2. **Multi-start initialisation** : critical solvers try multiple initial guesses (continuation seed, heuristic estimates, geometric range, fixed fallbacks) and select the solution with smallest residual, mitigating the basin-of-attraction problem inherent in Newton’s method. 

3. **Armijo backtracking** Armijo [1966]: Newton steps are damped by halving the step size until the residual decreases, preventing divergence near spectral edges and singular points. 

4. **Logarithmic parameterisation** : for _L ≥_ 5, the _z_ 1-mapping uses _w_ = log( _z_ 1) to compress _O_ ( _e[L]_ ) dynamic range to _O_ ( _L_ ), avoiding overflow in ( _z_ 1 _−_ 1 _/G_ 1( _z_ 1)) _[L][−]_[1] . 

5. **Tikhonov regularisation** : when the Jacobian matrix in the operator-valued Newton has condition number _>_ 10[10] , a regularisation term _λI_ with _λ_ = 10 _[−]_[8] _∥J∥F_ is added. 

6. **Non-finite value guarding** : solver outputs are checked for non-finite values; any detected NaN or infinity is replaced by a safe fallback, preventing propagation through the pipeline. 

7. **Post-processing interpolation** : isolated unconverged grid points ( _<_ 10% of total) are linearly interpolated in Re/Im parts from neighbouring converged values. 

46 

Table 4: Sinkhorn evaluation metrics at each checkpoint. Bold indicates the best value for each metric. Training ran for _∼_ 511K steps; Sinkhorn continued to improve in exact-match accuracy through the final checkpoint. 

|Step|Exact Acc.|Pass@1|Pass@2|Pass@10|Pass@100|Pass@1000|
|---|---|---|---|---|---|---|
|52,951|1.4%|5.2%|7.5%|12.1%|14.9%|16.1%|
|65,902|3.1%|9.0%|11.6%|16.2%|21.4%|22.6%|
|78,854|5.0%|11.6%|16.2%|21.1%|26.8%|29.5%|
|99,951|8.5%|19.1%|22.8%|29.4%|36.6%|39.6%|
|112,902|9.9%|22.2%|26.2%|33.1%|40.6%|43.9%|
|125,853|11.2%|23.7%|28.0%|35.5%|42.8%|45.5%|
|138,805|13.0%|26.6%|30.3%|38.6%|44.9%|48.8%|
|151,757|14.3%|28.6%|31.7%|40.3%|46.0%|49.8%|
|164,709|15.9%|29.0%|33.0%|41.2%|47.4%|51.5%|
|177,661|16.2%|30.0%|34.6%|41.6%|47.5%|51.7%|
|190,612|17.4%|30.8%|34.7%|42.3%|48.5%|53.0%|
|203,563|17.1%|31.0%|35.5%|42.8%|48.8%|53.8%|
|216,514|18.2%|31.3%|35.5%|42.4%|50.5%|54.1%|
|229,465|18.6%|31.7%|36.0%|43.0%|50.9%|55.4%|
|242,416|18.7%|31.2%|35.6%|43.4%|51.1%|**56.1%**|
|258,951|19.3%|30.8%|33.4%|42.3%|47.5%|48.0%|
|271,902|19.4%|31.9%|34.5%|44.1%|48.9%|51.0%|
|284,853|20.3%|33.6%|37.6%|44.7%|50.1%|52.2%|
|297,805|20.1%|34.1%|38.0%|45.5%|51.4%|54.3%|
|322,952|21.2%|31.5%|35.0%|42.5%|48.1%|49.1%|
|335,903|22.0%|33.6%|37.6%|44.6%|50.1%|51.4%|
|348,854|22.2%|34.0%|38.5%|45.6%|50.2%|53.0%|
|361,805|22.6%|34.7%|39.5%|46.4%|51.7%|54.3%|
|374,756|23.0%|36.3%|40.5%|45.9%|**51.9%**|54.5%|
|393,952|22.9%|33.8%|38.5%|43.9%|48.8%|49.8%|
|406,903|24.0%|**36.5%**|41.1%|46.3%|51.1%|52.6%|
|419,854|26.0%|36.0%|**41.7%**|**46.8%**|51.9%|53.4%|
|434,952|26.8%|35.7%|40.1%|45.8%|49.5%|49.8%|
|459,952|26.7%|35.0%|40.4%|43.6%|48.0%|48.2%|
|472,904|27.7%|36.1%|41.2%|45.6%|48.9%|49.4%|
|485,855|27.6%|35.9%|41.2%|46.4%|49.3%|50.6%|
|510,951|**27.9%**|36.4%|40.6%|45.1%|48.9%|49.5%|



47 

Table 5: Grassmann evaluation metrics at each checkpoint (training ongoing at _∼_ 111K steps). Bold indicates the best value for each metric. At matched step counts, Grassmann tracks ahead of Sinkhorn’s early trajectory (cf. Sinkhorn at 113K: 9.9% exact, 22.2% pass@1). 

|Step|Exact Acc.|Pass@1|Pass@2|Pass@10|Pass@100|Pass@1000|
|---|---|---|---|---|---|---|
|12,951|0.2%|0.6%|1.3%|3.2%|3.9%|4.6%|
|28,951|0.8%|3.6%|5.5%|9.9%|12.4%|14.0%|
|41,902|2.6%|8.9%|12.1%|16.0%|18.6%|21.1%|
|58,951|5.5%|15.7%|18.6%|24.0%|29.4%|31.5%|
|71,902|7.4%|20.0%|24.1%|30.4%|35.7%|38.4%|
|84,854|9.6%|22.2%|27.3%|34.7%|40.4%|43.8%|
|97,805|10.9%|26.1%|29.1%|36.5%|43.0%|45.9%|
|110,756|**12.8%**|**27.5%**|**31.2%**|**39.1%**|**46.0%**|**49.0%**|



Table 6: Forward and backward pass compute estimates per JPmHC module. 

|Variant|Forward FLOPs|Backward FLOPs|Total per module|
|---|---|---|---|
|Sinkhorn (implicit)|_O_(_Tn_2) = 320|_O_(_kn_2) = 256|576|
|Cayley|_O_(_sn_3) = 128|_O_(_sn_3) = 128|256|
|Grassmann|_O_(_np_) = 8|_O_(_n_3)_†_ = 64|72|



Table 7: Feature comparison with HC and mHC. 

|Feature|HC / mHC|JPmHC (Ours)|
|---|---|---|
|Mixing parameterization|Learned _n × n_ / Sinkhorn|Stiefel, Grassmann, Birkhof|
|Manifold constraint|Birkhof polytope only|Stiefel, Grassmann, Birkhof|
|Implicit diferentiation|—|Sinkhorn backward|
|Riemannian optimization|—|Grassmann (Cayley ADAM)|
|Spectral analysis|—|Generating set selection|
|CUDA graph compatible|Not addressed|All variants|
|Distributed training|DualPipe (mHC)|DDP + DeepSpeed ZeRO|



Table 8: Tractability of exhaustive spectral gap search. 

|_n_|_n_!|�_n_!<br>4<br>�|Matrix size|Feasibility|
|---|---|---|---|---|
|3|6|15|6_×_6|_<_0_._01s|
|4|24|10,626|24_×_24|_∼_0_._3s|
|5|120|8,214,570|120_×_120|_∼_1 hour|
|6|720|_∼_1010|720_×_720|Intractable|



Table 9: Parameter count per JPmHC module ( _n_ = 4, _d_ = 512, _nd_ = 2048). 

|Variant|_D_res|Total params|Notes|
|---|---|---|---|
|Sinkhorn|_n_2 = 16|(_n_+_n_+ 16)_× nd_+ 36 = 49_,_188|Fused _ϕ_|
|Cayley|3_n_2 = 48|48_× nd_+LayerNorm= 102_,_400|Fused _ϕ_ + LN|
|Grassmann|_np_= 8|(_n_+_n_+ 8)_× nd_+ 28 = 32_,_796|_p_= 2|
|Perm Mix|_K_ = 6|(_n_+_n_+ 6)_× nd_+ 22 = 28_,_694|+ perm indices|



48 

Table 10: Gauss-Seidel convergence for _n_ = 4 with typical **P** matrices. 

|Iterations|_k_|Relative|residual _∥_**r**_∥/∥_**r**0_∥_|Gradient error (%)|
|---|---|---|---|---|
|1|||7_._5_×_10_−_1|43.2|
|2|||5_._6_×_10_−_1|28.1|
|4|||3_._2_×_10_−_1|11.5|
|8|||1_._0_×_10_−_1|2.8|
|16|||1_._0_×_10_−_2|0.08|
|32|||1_._0_×_10_−_4|_<_0_._001|



Table 11: Full TRM architecture configuration. 

|Parameter|Value|
|---|---|
|Total parameters|_∼_7M|
|Hidden dimension _d_|512|
|Number of streams _n_|4|
|Efective hidden dim _nd_|2048|
|Unique transformer layers (weight-tied)|2|
|Recursive cycles per layer|6|
|Total recursive applications|12|
|Attention heads|8|
|FFN expansion ratio|4_×_|
|Halting mechanism|Adaptive Computation Time (ACT)|
|ACT max recursion depth|16|
|ACT exploration probability|0.1|
|JPmHC modules per layer|2 (pre-attention, pre-FFN)|
|Total JPmHC modules|4 (shared via weight tying)|
|Positional encoding|RoPE|
|Flash Attention|Enabled (with SDPA fallback)|



Table 12: Training hyperparameters (shared across all variants). 

|Hyperparameter|Value|
|---|---|
|Optimizer|AdamAtan2 [Kunstner et al., 2023]|
|Learning rate|1_×_10_−_4|
|Global batch size|768|
|Weight decay|0.1|
|Gradient clipping|1.0|
|LR schedule|Step decay at 80% and 90% of training|
|LR decay factors|0_._316_×_ and 0_._1_×_|
|Warmup steps|2000|
|Precision|bfoat16 (mixed precision)|
|Framework|PyTorch DDP + `torch.compile`|
|Compile mode|`default`|
|Hardware|NVIDIA B200 192GB GPUs (_×_8)|
|Puzzle embedding optimizer|SignSGD (lr=10_−_2)|
|Puzzle embedding dim|512 (_×_ 16 tokens)|



49 

Table 13: Required Gauss-Seidel iterations for _ϵ_ = 0 _._ 01 convergence. 

|_n_|2|3|4|5|6|8|
|---|---|---|---|---|---|---|
|_ρ_= 1_−_1_/n_|0.500|0.667|0.750|0.800|0.833|0.875|
|Iterations _k_|7|12|16|21|26|37|



Table 14: Backward pass complexity: standard vs. implicit Sinkhorn. 

||Standard Sinkhorn|Implicit Sinkhorn (Ours)|
|---|---|---|
|Autograd nodes|_O_(_T · n_2) (_∼_128K)|_O_(1) (constant)|
|Backward kernels|_∼_128K microsecond-scale|_∼_20 millisecond-scale|
|DDP overlap|Poor (55% stalls)|Excellent|
|Memory (saved tensors)|_O_(_T · n_2)|_O_(_n_2) (just **P**)|
|Gradient accuracy|Exact|_ϵ ≤_0_._01 (controlled)|



Table 15: Properties of doubly-stochastic (Sinkhorn) vs. orthonormal (Cayley) mixing. 

|Property|Doubly-Stochastic (_Bn_)|Orthonormal (_O_(_n_))|
|---|---|---|
|Norm behavior|Contractive (_∥_**Hx**_∥≤∥_**x**_∥_)|Preserving (_∥_**Hx**_∥_=_∥_**x**_∥_)|
|Entries|Non-negative|Unconstrained sign|
|Row/col sums|Both equal **1**|Not constrained|
|det(**H**)|_∈_[0_,_1]|_±_1|
|Convex hull|Permutation matrices|Not a convex set|
|Gradient fow|May attenuate|Preserved|
|Backward cost|_O_(_T_) or implicit|Standard autograd|



Table 16: Parameter count for residual mixing matrix ( _n_ = 4). 

|Variant|Parameters|_n_= 4|
|---|---|---|
|Sinkhorn (full _n × n_)|_n_2|16|
|Cayley (full _n × n_)|_n_2|16|
|Grassmannian (_n × p_, _p_=_n/_2)|_np_|8|



50 

