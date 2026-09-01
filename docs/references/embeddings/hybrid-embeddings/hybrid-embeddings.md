# Learning Mixed-Curvature Representations in Products of Model Spaces

- **Authors:** Albert Gu, Frederic Sala, Beliz Gunel, Christopher Re (Stanford)
- **Year:** 2019 (ICLR 2019)
- **Source:** https://openreview.net/forum?id=HJxeWnCcF7
- **MORPH uses:** The product-manifold formalism combining Euclidean and hyperbolic (Lorentz) components in a single embedding space, giving the model heterogeneous curvature - flat space for local syntactic structure, negatively curved space for hierarchical/ontological structure. MORPH's embeddings.py implements this as eucl (+) Lorentz with learned mixing weights. (No arXiv preprint; canonical reference is OpenReview.)

---

Published as a conference paper at ICLR 2019 

## LEARNING MIXED-CURVATURE REPRESENTATIONS IN PRODUCTS OF MODEL SPACES 

## **Albert Gu, Frederic Sala, Beliz Gunel & Christopher R´e** 

Computer Science Department Stanford University Stanford, CA 94305 _{_ albertgu,fredsala,bgunel _}_ @stanford.edu, chrismre@cs.stanford.edu 

## ABSTRACT 

The quality of the representations achieved by embeddings is determined by how well the geometry of the embedding space matches the structure of the data. Euclidean space has been the workhorse for embeddings; recently hyperbolic and spherical spaces have gained popularity due to their ability to better embed new types of structured data—such as hierarchical data—but most data is not structured so uniformly. We address this problem by proposing learning embeddings in a product manifold combining multiple copies of these _model spaces_ (spherical, hyperbolic, Euclidean), providing a space of heterogeneous curvature suitable for a wide variety of structures. We introduce a heuristic to estimate the sectional curvature of graph data and directly determine an appropriate signature—the number of component spaces and their dimensions—of the product manifold. Empirically, we jointly learn the curvature and the embedding in the product space via Riemannian optimization. We discuss how to define and compute intrinsic quantities such as means—a challenging notion for product manifolds—and provably learnable optimization functions. On a range of datasets and reconstruction tasks, our product space embeddings outperform single Euclidean or hyperbolic spaces used in previous works, reducing distortion by 32 _._ 55% on a Facebook social network dataset. We learn word embeddings and find that a product of hyperbolic spaces in 50 dimensions consistently improves on baseline Euclidean and hyperbolic embeddings, by 2.6 points in Spearman rank correlation on similarity tasks and 3.4 points on analogy accuracy. 

## 1 INTRODUCTION 

With four decades of use, Euclidean space is the venerable elder of embedding spaces. Recently, non-Euclidean spaces—hyperbolic (Nickel & Kiela, 2017; Sala et al., 2018) and spherical (Wilson et al., 2014; Liu et al., 2017)—have gained attention by providing better representations for certain types of structured data. The resulting embeddings offer better reconstruction metrics: higher mean average precision (mAP) and lower distortion compared to their Euclidean counterparts. These three spaces are the _model spaces_ of constant curvature (Lee, 1997), and this improvement in representation fidelity arises from the correspondence between the structure of the data (hierarchical, cyclical) and the geometry of non-Euclidean space (hyperbolic: negatively curved, spherical: positively curved). The notion of _curvature_ plays the key role. 

To improve representations for a variety of types of data—beyond hierarchical or cyclical—we seek spaces with heterogeneous curvature. The motivation for such _mixed_ spaces is intuitive: our data may have complicated, varying structure, in some regions tree-like, in others cyclical, and we seek the best of all worlds. We expect mixed spaces to match the geometry of the data and thus provide higher quality representations. However, to employ these spaces, we face several key obstacles. We must perform a challenging manifold optimization to learn both the curvature and the embedding. Afterwards, we also wish to operate on the embedded points. For example, analogy operations for word embeddings in Euclidean vector space (e.g., _a − b_ + _c_ ) must be lifted to manifolds. 

1 

Published as a conference paper at ICLR 2019 

Figure 1: Three component spaces: sphere S[2] , Euclidean plane E[2] , and hyperboloid H[2] . Thick lines are geodesics; these get closer in positively curved ( _K_ = +1) space S[2] , remain equidistant in flat ( _K_ = 0) space E[2] , and get farther apart in negatively curved ( _K_ = _−_ 1) space H[2] . 

We propose embedding into _product spaces_ in which each component has constant curvature. As we show, this allows us to capture a wider range of curvatures than traditional embeddings, while retaining the ability to globally optimize and operate on the resulting embeddings. Specifically, we form a Riemannian product manifold combining hyperbolic, spherical, and Euclidean components and equip it with a decomposable Riemannian metric. While each component space in the product has constant curvature (positive for spherical, negative for hyperbolic, and zero for Euclidean), the resulting mixed space has _non-constant_ curvature. However, selecting appropriate curvatures for the embedding space is a potential challenge. We directly learn the curvature for each component space along with the embedding (via Riemannian optimization), recovering the correct curvature, and thus the matching geometry, directly from data. We show empirically that we can indeed recover non-uniform curvatures and improve performance on reconstruction metrics. 

Another technical challenge is to select the underlying number of components and dimensions of the product space; we call this the _signature_ . This concept is vacuous in Euclidean space: the product of E _[r]_[1] _, . . . ,_ E _[r][n]_ is identical to the single space E _[r]_[1][+] _[...]_[+] _[r][n]_ . However, this is _not_ the case with spherical and hyperbolic spaces. For example, the product of the spherical space S[1] (the circle) with itself is the torus S[1] _×_ S[1] , which is topologically distinct from the sphere S[2] . We address this challenge by introducing a theory-guided heuristic estimator for the signature. We do so by matching an empirical notion of discrete curvature in our data with the theoretical distribution of the sectional curvature, a fine-grained measure of curvature on Riemannian manifolds that is amenable to analysis in products. We verify that this approach recovers the correct signature on reconstruction tasks. 

Standard techniques such as PCA require centering so that the embedded directions capture components of variation. Centering in turn needs an appropriate generalization of the mean. We develop a formulation of mean for embedded points that exploits the decomposability of the distance and has theoretical guarantees. For _T_ = _{p_ 1 _, . . . , pn}_ in a manifold _M_ with dimension _r_ , the mean is _µ_ ( _T_ ) := arg min _p i[d]_[2] _M_[(] _[p, p][i]_[)][.][We give a global existence result:][under symmetry conditions on] the distribution of the points in _T_ on the spherical components, gradient descent recovers _µ_ ( _T_ ) with error _ε_ in time _O_ ( _nr_ log _ε[−]_[1] ). 

We demonstrate the advantages of product space embeddings through a variety of experiments; products are at least as good as single spaces, but can offer significant improvements when applied to structures not suitable for single spaces. We measure reconstruction quality (via mAP and distortion) for synthetic and real datasets over various allocations of embedding spaces. We observe a 32 _._ 55% improvement in distortion versus any single space on a Facebook social network graph. Beyond reconstruction, we apply product spaces to skip-gram word embeddings, a popular technique with numerous downstream applications, which crucially require the use of the manifold structure. We find that products of hyperbolic spaces improve performance on benchmark evaluations—suggesting that words form multiple smaller hierarchies rather than one larger one. We see an improvement of 3.4 points over baseline single spaces on the Google word analogy benchmark and of 2.6 points 

2 

Published as a conference paper at ICLR 2019 

in Spearman rank correlation on a word similarity task using the WS-353 corpus. Our results and initial exploration suggest that mixed product spaces are a promising area for future study. 

## 2 PRELIMINARIES & BACKGROUND 

**Embeddings** For metric spaces[1] _U, V_ equipped with distances _dU , dV_ , an _embedding_ is a mapping _f_ : _U → V_ . The quality of an embedding is measured by various _fidelity measures_ . A standard measure is _average distortion D_ avg. The distortion of a pair of points _a, b_ is ( _|dV_ ( _f_ ( _a_ ) _, f_ ( _b_ )) _− dU_ ( _a, b_ ) _|_ ) _/dU_ ( _a, b_ ), and _D_ avg is the average over all pairs of points. 

Distortion is a global metric; it considers the explicit value of all distances. At the other end of the global-local spectrum of fidelity measures is _mean average precision_ (mAP), which applies to unweighted graphs. Let _G_ = ( _V, E_ ) be a graph and node _a ∈ V_ have neighborhood _Na_ = _{b_ 1 _, . . . , b_ deg( _a_ ) _}_ , where deg( _a_ ) is the degree of _a_ . In the embedding _f_ , define _Ra,bi_ to be the smallest ball around _f_ ( _a_ ) that contains _bi_ (that is, _Ra,bi_ is the smallest set of nearest points required to 1 1 _|Na|_ retrieve the _i_ th neighbor of _a_ in _f_ ). Then, mAP( _f_ ) = _|V |_ � _a∈V_ deg( _a_ ) � _i_ =1 _[|N][a][ ∩][R][a,b] i[|][/][|][R][a,b] i[|][.]_ 

Note that mAP does not track explicit distances; it is a ranking-based measure for local neighborhoods. Observe that mAP( _f_ ) _≤_ 1 (higher is better) while _d_ avg _≥_ 0 (lower is better). 

**Riemannian Manifolds** We briefly review some notions from manifolds and Riemannian geometry. A more in-depth treatment can be found in standard texts (Lee, 2012; do Carmo, 1992). Let _M_ be a smooth manifold, _p ∈ M_ be a point, and _TpM_ be the tangent space to the point _p_ . If _M_ is equipped with a Riemannian metric _g_ , then the pair ( _M, g_ ) is called a _Riemannian manifold_ . The shortest-distance paths on manifolds are called _geodesics_ . To compute distance functions on a Riemannian manifold, the _metric tensor g_ is integrated along the geodesic. This is a smoothly varying function (in _p_ ) _g_ : _TpM × TpM →_ R that induces geometric notions such as length and angle by defining an inner product on the tangent space. For example, the norm of _v ∈ TpM_ is defined as 1 _∥v∥g_ := _gp_ ( _v, v_ ) 2 _._ In Euclidean space R _[d]_ , each tangent space _Tp_ R _[d]_ is canonically identified with R _[d]_ , and the metric tensor _g[E]_ is simply the normal inner product. 

**Product Manifolds** Consider a sequence of smooth manifolds _M_ 1, _M_ 2, _. . . , Mk_ . The product manifold is defined as the Cartesian product _M_ = _M_ 1 _× M_ 2 _× . . . × Mk_ . Notationally, we write points _p ∈ M_ through their coordinates _p_ = ( _p_ 1 _, . . . , pk_ ) : _pi ∈ Mi_ , and similarly a tangent vector _v ∈ TpM_ can be written ( _v_ 1 _, . . . , vk_ ) : _vi ∈ TpiMi_ . If the _Mi_ are equipped with metric tensor _gi_ , then the product _M_ is also Riemannian with metric tensor _g_ ( _u, v_ ) =[�] _[k] i_ =1 _[g][i]_[(] _[u][i][, v][i]_[)] _[.]_[ That is,][the] product metric decomposes into the sum of the constituent metrics. 

**Geodesics and Distances** Optimization on manifolds requires a notion of taking a step. This step can be performed in the tangent space and transferred to the manifold via the _exponential map_ Exp _p_ : _TpM → M_ . In a product manifold _P_ , for tangent vectors _v_ = ( _v_ 1 _, . . . , vk_ ) at _p_ = ( _p_ 1 _, . . . , pk_ ) _∈ M_ , the exponential map simply decomposes, as do squared distances (Ficken, 1939; Turaga & Srivastava, 2016): 

**==> picture [347 x 30] intentionally omitted <==**

In other words, the shortest path between points in the product travels along the shortest paths in each component simultaneously. Note the analogy to Euclidean products R _[d] ≡_ (R[1] ) _[d]_ . 

**Hyperbolic and Spherical Model** We use the _hyperboloid model_ of hyperbolic space, with points in R _[d]_[+1] . Let _J ∈_ R[(] _[d]_[+1)] _[×]_[(] _[d]_[+1)] be the diagonal matrix with _J_ 00 = _−_ 1 and _Jii_ = 1 : _i >_ 0. For _p, q ∈_ R _[d]_[+1] , the Minkowski inner product is _⟨p, q⟩∗_ := _p[T] Jq_ = _−p_ 0 _q_ 0 + _p_ 1 _q_ 1 + _. . ._ + _pdqd_ , and 1 the corresponding norm is _∥p∥∗_ = _⟨p, p⟩∗_ 2 . For any _K >_ 0, the hyperboloid H _dK_[is defined on the] 

> 1In this paper, we use the language of graphs; note that any discrete metric space can be identified with a weighted graph, and all of our algorithms operate on weighted graphs. 

3 

Published as a conference paper at ICLR 2019 

subset _{p ∈_ R _[d]_[+1] : _∥p∥∗_ = _−K_[1] _[/]_[2] _, p_ 0 _>_ 0 _}_ . When the subscript _K_ is omitted, it is taken to be 1. The hyperbolic distance on H _[d]_ is _dH_ ( _p, q_ ) = acosh( _−⟨p, q⟩∗_ ). 

Similarly, spherical space S _[d] K_[is][most][easily][defined][when][embedded][in][R] _[d]_[+1][.][The][manifold][is] defined on the subset _{p ∈_ R _[d]_[+1] : _∥p∥_ 2 = _K_[1] _[/]_[2] _}_ , with metric _g[S]_ induced by the Euclidean metric on R _[d]_[+1] . The spherical distance on S _[d]_ is _dS_ ( _p, q_ ) = arccos( _⟨p, q⟩_ ) . 

## 3 PRODUCT SPACES AND CONSTRUCTIONS 

We now tackle the challenges of mixed spaces. First, we introduce a product manifold embedding space _P_ composed of multiple copies of simple model spaces, providing heterogeneous curvature. Next, in Section 3.1, given the signature of _P_ (the number of components of each type and their dimensions), we describe how to simultaneously learn an embedding and the curvature for each component through optimization. In Section 3.2, we provide a heuristic to choose the signature by estimating a discrete notion of curvature for given data. Finally, in Section 3.3, given an embedding in _P_ , we introduce a Karcher-style mean which can be recovered efficiently. 

Let S _[d] K_[and][H] _[d] K_[be][the][spherical][and][hyperbolic][spaces][of][dimension] _[d]_[and][curvature] _[K,][ −][K]_[,][re-] spectively, and E _[d]_ the Euclidean space of dimension _d_ .[2] We describe our main embedding space: for sequences of dimensions _s_ 1 _, s_ 2 _, . . . , sm_ , _h_ 1 _, . . . , hn_ , and _e_ , we write 

_P_ = S _[s]_[1] _×_ S _[s]_[2] _× · · · ×_ S _[s][m] ×_ H _[h]_[1] _×_ H _[h]_[2] _× · · · ×_ H _[h][n] ×_ E _[e] ,_ a product manifold with _m_ + _n_ + 1 component spaces and total dimension[�] _i[s][i]_[ +][ �] _j[h][j]_[ +] _[ e]_[.][We] refer to each S _[s][i] ,_ H _[h][i] ,_ E _[e]_ as _components_ or _factors_ . We refer to the decomposition, e.g., (H[2] )[2] = H[2] _×_ H[2] , as the _signature_ . For convenience, let _M_ 1 _, . . . , Mm_ + _n_ +1 refer to the factors in the product. 

**Distances on** _P_ As discussed in Section 2, the product _P_ is a Riemannian manifold defined by the structure of its components. For _p, q ∈P_ , we write _dMi_ ( _p, q_ ) for the distance _dMi_ restricted to the appropriate components of _p_ and _q_ in the product. In particular, the squared distance in the product decomposes via (1). In other words, _dP_ is simply the _ℓ_ 2 norm of the component distances _dMi_ . 

We note that _P_ can also be equipped with different distances (ignoring the Riemannian structure), leading to a different embedding space. Without the underlying manifold structure, we cannot freely operate on the embedded points such as taking geodesics and means, but some simple applications only interact through distances. For such settings, we consider the _ℓ_ 1 distance _dP,ℓ_ 1( _p, q_ ) =[�] _[s] i_ =1 _[m][d][S] i_[(] _[p, q]_[) +][ �] _[h] i_ =1 _[n][d][H] i_[(] _[p, q]_[) +] _[ d][E]_[(] _[p, q]_[)][and][the][min][distance] _[d][P][,]_[min][(] _[p, q]_[)][=] min _{dS_ 1( _p, q_ ) _, . . . , dH_ 1( _p, q_ ) _, . . . , dE_ ( _p, q_ ) _}_ . These distances provide simple and interpretable embedding spaces using _P_ , enabling us to introduce _combinatorial constructions_ that allow for embeddings without the need for optimization. We give an example below and discuss further in the Appendix. We then focus on the Riemannian distance, which allows Riemannian optimization directly on the manifold, and enables full use of the manifold structure in generic downstream applications. 

**Example** Consider the graph _G_ shown on the right of Figure 2. This graph has a backbone cycle with 9 nodes, each attached to a tree; such topologies are common in networking. If a single edge ( _a, b_ ) is removed from the cycle, the result is a tree embeddable arbitrarily well into hyperbolic space (Sala et al., 2018). However, _a, b_ (and their subtrees) would then incur an additional distance of 8 _−_ 1 = 7, being forced to go the other way around the cycle. But using the _ℓ_ 1 distance, we can embed _G_ tree into H[2] and _G_ cycle into S[1] , yielding arbitrarily low distortion for _G_ . We give the full details and another combinatorial construction for the min-distance in the Appendix. 

## 3.1 OPTIMIZATION & COMPONENT CURVATURES 

To compute embeddings, we optimize the placement of points through an auxiliary loss function. Given graph distances _{dG_ ( _Xi, Xj_ ) _}ij_ , our loss function of choice is 

**==> picture [283 x 32] intentionally omitted <==**

> 2We write E for our Euclidean embedding space component to distinguish it from R, since our models of hyperbolic and spherical geometry also use R as an ambient space. 

4 

Published as a conference paper at ICLR 2019 

|**Algorithm 1**R-SGD inproducts|**Algorithm 1**R-SGD inproducts||||
|---|---|---|---|---|
|1:|**Input: Loss function**_L_:_P →_R||||
|2:|Initialize_x_(0) _∈P_ randomly||||
|3:|**for**_t_= 0_, . . . , T −_1**do**||||
|4:|_h ←∇L_(_x_(_t_))||||
|5:|**for**_i_= 1_, . . . , m_**do**||||
|6:|_vi ←_proj_S_<br>_x_(_t_)<br>_i_ (_hi_)||_G_||
|7:<br>8:|**for**_i_=_m_+ 1_, . . . , m_+_n_**do**<br>_vi ←_proj_H_<br>_x_(_t_)<br>_i_ (_hi_)||||
|9:|_vi ←Jvi_||||
|10:|_vm_+_n_+1 _←hm_+_n_+1||||
|11:|**for**_i_= 1_, . . . , m_+_n_+ 1**do**||||
|12:|_x_(_t_+1)<br>_i_<br>_←_Exp_x_(_t_)<br>_i_ (_vi_)|_G_tree||_G_cycle|
|13:|**return**_x_(_T_)||||



Figure 2: Left: Riemannian SGD decomposes per component. Subscripts _i_ index components in the product. Right: Ring of trees graph _G_ . Neither hyperbolic nor spherical space is suitable for _G_ , but the product H _×_ S captures it with low distortion. Note the decomposition into tree and cycle. 

which captures the average distortion. (2) depends on hyperbolic distance _dH_ (for which the gradient is unstable) only through the square _d_[2] _H_[, which is continuously differentiable (Sala et al., 2018).] 

In any Riemannian manifold, a loss function can be optimized through standard Riemannian optimization methods such as RSGD (Bonnabel, 2013) and RSVRG (Zhang et al., 2016). We write down the full RSGD specialized to our product spaces in Algorithm 1. This proceeds by first computing the Euclidean gradient _∇L_ ( _x_ ) with respect to the ambient space of the embedding (Step 4), and then converting it to the Riemannian gradient by applying the Riemannian correction (multiply by the inverse of the metric tensor _gP[−]_[1][).][This overall strategy has been detailed in previous work in the] hyperboloid model (Nickel & Kiela, 2018; Wilson & Leimeister, 2018), and the same calculations apply to our hyperbolic components. 

Since _gP_ is block diagonal on a product manifold, it suffices to apply the correction and perform the gradient step in each component _Mi_ independently. In the spherical and hyperboloid models, which have smaller dimension than the ambient space, this is performed by first projecting the gradient vector _h_ onto the tangent space _TxM_ via proj _[S] x_[(] _[h]_[) =] _[ h][ −⟨][h, x][⟩][x]_[ (Step 6) and proj] _[H] x_[(] _[h]_[) =] _h_ + _⟨h, x⟩∗x_ (Step 8). In the hyperboloid model, a final rescaling by the inverse of the metric _J_ is needed (Step 9). This is not required in the spherical model since it inherits the same metric from the ambient Euclidean space. 

**Learning the Curvature** There exists a spherical model for every curvature _K >_ 0 (for example, the sphere S _[d] K_[of radius] _[ K][−]_[1] _[/]_[2][) and a hyperbolic model for every] _[ K][<]_[0][ (the hyperboloid][ H] _[d] −K_[).] We jointly optimize the curvature _Ki_ of every non-Euclidean factor _Mi_ along with the embeddings. 

The idea is that distances on the spherical and hyperboloid models of arbitrary curvature can be emulated through distances on the standard models S _,_ H of curvature 1. For example, given _p, q_ on the sphere S1 _/R_ 2 of radius _R_ , then _d_ ( _p, q_ ) = _R · d_ S1( _p/R, q/R_ ) where _p/R, q/R_ lie on the unit sphere. Therefore the radius _R_ , which is monotone in the curvature _K_ , can be treated as a parameter as well, so that we can optimize _K_ and implicitly represent points lying on the manifold of curvature _K_ , while explicitly only needing to store and optimize points in the standard model of curvature 1 via Algorithm 1. The hyperboloid model is analogous. Moreover, the loss (2) depends only on squared distances on the product manifold, which are simple functions of distances in the components through (1), so we can optimize the curvature of each factor in _P_ . 

5 

Published as a conference paper at ICLR 2019 

**==> picture [352 x 59] intentionally omitted <==**

**----- Start of picture text -----**<br>
c c c<br>m m m<br>a b a b a b<br>**----- End of picture text -----**<br>


Figure 3: Geodesic triangles in differently curved spaces: compared to Euclidean geometry in which it satisfies the parallelogram law (Center), the median _am_ is longer in cycle-like positively curved space (Left), and shorter in tree-like negatively curved space (Right). The relative length of _am_ can be used as a heuristic to estimate discrete curvature. 

## 3.2 ESTIMATING THE SIGNATURE 

To choose the signature of an appropriate space _P_ corresponding to given data, we again turn to curvature. We use the _sectional curvature_ , a finer-grained notion defined over all two-dimensional subspaces passing through a point. Unlike coarser notions like scalar curvature, this is not constant in a product of basic spaces. Given linearly independent _u, v ∈ TpM_ spanning a two-dimensional subspace _U_ , the sectional curvature _Kp_ ( _u, v_ ) or _Kp_ ( _U_ ) is defined as the Gaussian curvature of the surface Exp( _U_ ) _⊆ M_ . Intuitively, this captures the rate that geodesics on the surface emanating from _p_ spread apart, which relates to volume growth. In Appendix C.2, we show that the sectional curvature of _P_ interpolates between the sectional curvatures of the factors, enabling us to better capture a wider range of structures in our embeddings: 

**Lemma 1.** _Let M_ = _M_ 1 _× M_ 2 _where Mi has constant curvature Ki. For any u, v ∈ TpM , if K_ 1 _, K_ 2 _are both non-negative, the sectional curvature satisfies K_ ( _u, v_ ) _∈_ [0 _,_ max _{K_ 1 _, K_ 2 _}_ ] _. If K_ 1 _, K_ 2 _are both non-positive, the sectional curvature satisfies K_ ( _u, v_ ) _∈_ [min _{K_ 1 _, K_ 2 _},_ 0] _. If Ki <_ 0 _and Kj >_ 0 _for i ̸_ = _j, then K_ ( _u, v_ ) _∈_ [ _Ki, Kj_ ] _._ 

Our estimation technique employs a triangle comparison theorem following from Toponogov’s theorem and the law of cosines, which characterizes sectional curvature through the behavior of small triangles (note that a triangle determines a 2-dimensional submanifold). Let _abc_ be a geodesic triangle in manifold (or metric space) _M_ and _m_ be the (geodesic) midpoint of _bc_ , and consider the quantity 

**==> picture [347 x 13] intentionally omitted <==**

This is non-negative (resp. non-positive) when the curvature is non-negative (resp. non-positive). Note that consequently the equality case occurs exactly when the curvature is 0, and equation 3 becomes the _parallelogram law_ of Euclidean geometry (Figure 3). 

Analogous to sectional curvature, which is a function of a point _p_ and two directions _x, y_ from _p_ , in an undirected graph _G_ we define an analog for every node _m_ and two neighbors _b, c_ . Given a reference node _a_ we set: _ξG_ ( _m_ ; _b, c_ ; _a_ ) = 2 _dG_ (1 _a,m_ ) _[ξ][G]_[(] _[a, b, c]_[)][.][This][is][exactly][the][expression] from equation 3, normalized suitably so as to yield the correct scaling for trees and cycles. Our 1 curvature estimation is then a simple average _ξG_ ( _m_ ; _b, c_ ) = _|V |−_ 1 � _a_ = _m[ξ][G]_[(] _[m]_[;] _[ b, c]_[;] _[ a]_[)][.] 

Importantly, _ξG_ recovers the right curvature for graph atoms such as lines, cycles, and trees (Appendix C.2, Lemma 4,5), and the correct sign for other special discrete objects like polyhedra (Thurston, 1998). The curvature is zero for lines, positive for cycles, and negative for trees. 

For a generic graph _G_ , we use this to generate a potential product manifold to embed in. An empirical sectional curvature of _G_ is estimated via Algorithm 3, which is based off the _homogeneity_ of product manifolds (i.e. isometries act transitively), implying that it suffices to analyze the curvature at a random point. In particular, we moment-match the distributions of sectional curvature through uniformly random 2-planes in the graph and in the manifold through Algorithms 3,2 (Appendix C.2). 

## 3.3 MEANS IN THE PRODUCT MANIFOLD 

A critical operation on manifolds is that of _taking the mean_ ; it is necessary for many downstream applications, including, for example, analogy tasks with word embeddings, for clustering, and for 

6 

Published as a conference paper at ICLR 2019 

centering before applying PCA. Even in simple settings like the circle S[1] , defining a mean is nontrivial. A classic approach is to take the Euclidean mean (in E[2] ) of the points and to project back onto S[1] —but this operation fails in the case where the points are uniformly spaced on S[1] . A further roadblock is the varying curvature of _P_ . Fortunately, we can exploit the decomposability of the distance on _P_ , reducing the challenge to breaking symmetries in the component spaces. To do so, we introduce the following Karcher-style weighted mean. Let _T_ = _{p_ 1 _, p_ 2 _, . . . , pn}_ be a set of points in _P_ and _w_ 1 _, . . . , wn_ be positive weights satisfying[�] _[n] i_ =1 _[w][i]_[=][1][.][Then][the][mean] _[µ]_[(] _[T]_[)][is] arg min _p∈P_ � _ni_ =1 _[w][i][d]_[2] _P_[(] _[p, p][i]_[)][.][In special cases, this matches commonly used means (the centroid] in the Euclidean case E _[d]_ , the spherical average for S[2] in Buss & Fillmore (2001)). We further note that when _wi ≥_ 0, the squared-distance components above are individually convex: this is trivial in the Euclidean term, holds in the hyperbolic case (cf. Theorem 4.1 (Bishop & O’Neill, 1969)), and holds in the spherical case under certain restrictions, e.g., when the points in _T_ lie entirely in one hemisphere of S _[r]_ (Buss & Fillmore, 2001). Moreover, in this case, peforming the optimization on the mean with gradient descent via the exponential map offers linear rate convergence: 

**Lemma 2.** _Let P be a product of model spaces of total dimension r, T_ = _{p_ 1 _, . . . , pn} points in P and w_ 1 _, . . . , wn weights satisfying wi ≥_ 0 _and_[�] _[n] i_ =1 _[w][i]_[=][1] _[.][Moreover,][let the components of] the points in P, pi|_ S _j restricted to each spherical component space_ S _[j] fall in one hemisphere of_ S _[j] . Then, Riemannian gradient descent recovers the mean µ_ ( _T_ ) _within distance ϵ in time O_ ( _nr_ log _ϵ[−]_[1] ) _._ 

This is a _global_ result; with weaker assumptions, we can derive local results; for example, in the case where some of the _wi_ are negative, which is useful for analogy operations. 

In summary, we offer the following key takeaways of our development: 

- Product manifolds of model spaces capture heterogeneous curvature while providing tractable optimization, 

- Each component’s curvature can be learned empirically through a reparametrization, 

- A signature for the product can be found by matching discrete notions of curvature on graphs with sectional curvature on manifolds, 

- There exists an easily-computed formulation of mean with theoretical guarantees. 

## 4 EXPERIMENTS 

We evaluate the proposed approach, comparing the representation quality of synthetic graphs and real datasets among different embedding spaces by measuring the reconstruction fidelity (through average distortion and mAP). We expect that mixed product spaces perform better for nonhomogeneous data. We consider the curvature of graphs, reporting the curvatures learned through optimization as well as the theoretical allocation from Section 3.2. Beyond reconstruction, we evaluate the intrinsic performance of product space embeddings in a skip-gram word embedding model, by defining tasks with generic manifold operations such as means. 

## 4.1 GRAPH RECONSTRUCTION 

**Datasets** We examine synthetic datasets—trees, cycles, the ring of trees shown in Figure 1, confirming that each matches its theoretically optimal embedding space. We then compare on several real-world datasets with describable structure, including the USCA312 dataset of distances between North American cities (Burkardt); a tree-like graph of computer science Ph.D. advisor-advisee relationships (De Nooy et al., 2011) reported in previous hyperbolics work (Sala et al., 2018); a powergrid distribution network with backbone structure (Watts & Strogatz, 1998); and a dense social network from Facebook (McAuley & Leskovec, 2012). For the former two graphs with well-defined structure, we expect optimal embeddings in spaces of positive and negative curvature, respectively. We hypothesize that the backbone network embeds well into simple products of hyperbolic and spherical spaces as in Figure 2, and the dense graph also benefits from a mixture of spaces. 

**Approaches** We minimize the loss (2) using Algorithm 1. We fix a total dimension _d_ and consider the most natural ways to construct product manifolds of the given dimension, through iteratively 

7 

Published as a conference paper at ICLR 2019 

Table 1: **Matching geometries** : Average distortion on canonical graphs (tree, cycle, ring of trees) with 40 nodes, comparing four spaces with total dimension 3. The best distortion is achieved by the space with matching geometry. 

||**Cycle**<br>**Tree**<br>**Ring of Trees**|
|---|---|
|(E3)1<br>(H3)1<br>(S3)1<br>(H2)1 _×_(S1)1|_|V |_= 40_, |E|_= 40<br>_|V |_= 40_, |E|_= 39<br>_|V |_= 40_, |E|_= 40|
||0.1064<br>0.1483<br>0.0997<br>0.1638<br>**0**_._**0321**<br>0.0774<br>**0**_._**0007**<br>0.1605<br>0.1106<br>0.1108<br>0.0538<br>**0**_._**0616**|



doubling the number of factors. These models include the products consisting of only a constantcurvature base space, ranging to various combinations of S _[d/]_ 2[2] _,_ H _[d/]_ 2[2] comprising factors of dimension 2.[3] For a given signature, the curvatures are initialized to the appropriate value in _{−_ 1 _,_ 0 _,_ 1 _}_ and then learned using the technique in Section 3.1. We additionally compare to the outputs of Algorithms 2,3 for heuristically selecting a combination of spaces in which to embed these datasets. 

**Quality** We focus on the average distortion—which our loss function (2) optimizes—as our main metric for reconstruction, and additionally report the mAP metric for the unweighted graphs. As expected, for the synthetic graphs (tree, cycle, ring of trees), the matching geometries (hyperbolic, spherical, product of hyperbolic and spherical) yield the best distortion (Table 1). Next, we report in Table 2 the quality of embedding different graphs across a variety of allocations of spaces, fixing total dimension _d_ = 10 following previous work (Nickel & Kiela, 2018). We confirm that the structure of each graph informs the best allocation of spaces. In particular, the cities graph—which has intrinsic structure close to S[2] —embeds well into any space with a spherical component, and the treelike Ph.D.s graph embeds well into hyperbolic products. We emphasize that even for such datasets that theoretically match a single constant-curvature space, the products thereof perform no worse. In general, the product construction achieves high quality reconstruction: the traditional Euclidean approach is often well below several other signatures. We additionally report the learned curvatures associated with the optimal signature, finding that the resulting curvatures are non-uniform even for products of identical spaces (cf. Ph.D.s). Finally, Table 3 reports the signature estimations of Algorithms 2, 3 for the unweighted graphs. Among the signatures over two components, the estimated curvature signs agree with best distortion results from Table 2. 

## 4.2 WORD EMBEDDINGS 

To investigate the performance of product space embeddings in applications requiring the underlying manifold structure, we learned word embeddings and evaluated them on benchmark datasets for word similarity and analogy. In particular, we extend results on hyperbolic skip-gram embeddings from Leimeister & Wilson (2018) (LW), who found that hyperbolic embeddings perform favorably against Euclidean word vectors in low dimensions ( _d_ = 5 _,_ 20), but less so in higher dimensions ( _d_ = 50 _,_ 100). Building on these results, we hypothesize that in high dimensions, a product of multiple smaller-dimension hyperbolic spaces will substantially improve performance. 

**Setup** We use the standard skip-gram model (Mikolov et al., 2013) and extend the loss function to a generic objective suitable for arbitrary manifolds, which is a variant of the objective proposed by LW. Concretely, given a word _u_ and target _w_ , with label _y_ = 1 if _w_ is a context word for _u_ and _y_ = 0 if it is a negative sample, the model is _P_ ( _y|w, u_ ) = _σ_ �( _−_ 1)[1] _[−][y]_ ( _−_ cosh( _d_ ( _αu, γw_ )) + _θ_ )� _._ 

Training followed the setup of LW, building on the _fastText_ skip-gram implementation. Euclidean results are reported directly from _fastText_ . Aside from choice of model, the training setup including hyperparameters (window size, negative samples, etc.) is identical to LW for all models. 

**Word Similarity** We measure the Spearman rank correlation _ρ_ between our scores and annotated ratings on the word similarity datasets WS-353 (Finkelstein et al., 2001), Simlex-999 (Hill et al., 

> 3Note that S1 and H1 are metrically equivalent to R, so these are not considered. 

8 

Published as a conference paper at ICLR 2019 

Table 2: **Graph reconstruction** : fidelity measures for graph embeddings using _d_ = 10 total dimensions, with varying allocations of spaces and dimensions. Our loss function (2) targets distortion, and for each dataset the best model reflects the structure of the data. Even on near-perfectly spherical or hierarchical data, products of S (resp. H) perform no worse than the single copy. 

||**Cities**<br>**CS PhDs**<br>**Power**<br>**Facebook**|
|---|---|
|E10<br>H10<br>S10|_|V |_=312<br>_|V |_=1025_, |E|_=1043<br>_|V |_=4941_, |E|_=6594<br>_|V |_=4039_, |E|_=88234|
||_D_avg<br>_D_avg<br>mAP<br>_D_avg<br>mAP<br>_D_avg<br>mAP|
||0.0735<br>0.0543<br>0.8691<br>0.0917<br>0.8860<br>0.0653<br>0.5801<br>0.0932<br>0.0502<br>0.9310<br>0.0388<br>0.8442<br>0.0596<br>0.7824<br>0.0598<br>0.0569<br>0.8329<br>0.0500<br>0.7952<br>0.0661<br>0.5562|
|(H5)2<br>(S5)2<br>H5 _×_S5<br>(H2)5<br>(S2)5<br>(H2)2_×_E2_×_(S2)2|0.0756<br>0.0382<br>0.9628<br>0.0365<br>0.8605<br>0.0430<br>0.7742<br>**0.0593**<br>0.0579<br>0.7940<br>0.0471<br>0.8059<br>0.0658<br>0.5728<br>0.0622<br>0.0509<br>0.9141<br>**0.0323**<br>0.8850<br>**0.0402**<br>0.7414<br>0.0687<br>**0.0357**<br>0.9694<br>0.0396<br>0.8739<br>0.0525<br>0.7519<br>0.0638<br>0.0570<br>0.8334<br>0.0483<br>0.8818<br>0.0631<br>0.5808<br>0.0765<br>0.0391<br>0.8672<br>0.0380<br>0.8152<br>0.0474<br>0.5951|
|**Best model**|S5<br>1_._0_×_S5<br>1_._1<br>H2<br>_._3_×_H2<br>_._6_×_H2<br>1_._5_×_(H2<br>1_._2)2<br>H5<br>3_._4 _×_S5<br>12_._6<br>H5<br>0_._3 _×_S5<br>3_._5|
|_D_**avg improvement**<br>**over single space**|0.8%<br>28.89%<br>16.75%<br>32.55%|



Table 3: **Heuristic allocation:** estimated signatures for embedding unweighted graphs from Table 2 into two factors, using Algorithms 2,3 to match the empirical distribution of graph curvature. The resulting curvature signs agree with results from Table 2 for choosing among two-component spaces. 

||CS PhDs|Power|Facebook|
|---|---|---|---|
|Estimated Signature|H5<br>1_._3 _×_H5<br>0_._2|H5<br>1_._8 _×_S5<br>1_._7|H5<br>0_._9 _×_S5<br>1_._6|



2015) and MEN (Bruni et al., 2014). The results are in Table 4. Notably, we find that hyperbolic word embeddings are consistently competitive with or better than the Euclidean embeddings, and the improvement increases with more factors in the product. This suggests that word embeddings implicitly contain multiple distinct but smaller hierarchies rather than forming a single larger one. 

**Analogies** In manifolds, there is no exact analog of the “word arithmetic” of conventional word embeddings arising from vector space structure. However, analogies can still be defined via intrinsic product manifold operations. In particular, note that the loss function depends on the embeddings solely through their pairwise distances. We thus define analogies _a_ : _b_ :: _c_ : _d_ by matching the distances _d_[2] ( _a, b_ ) = _d_[2] ( _c, d_ ) and _d_[2] ( _a, c_ ) = _d_[2] ( _b, d_ ) through constructing an analog of the parallelogram, by geodesically reflecting _a_ through the geodesic midpoint (i.e. mean) _m_ of _b, c_ . Note that this defines both the loss function and the intrinsic tasks purely in terms of distances and manifold operations. Hence, unlike traditional word embeddings, this formulation is generic to any space. 

Our evaluation, shown in Table 5, uses the standard Google word analogy benchmark (Mikolov et al., 2013). We observe a 22% accuracy improvement over single-space hyperbolic embeddings in 50 dimensions and similar improvements over a single hyperbolic space in 100 dimensions. As with similarity, accuracy on the analogy task consistently improves as the number of factors increases. 

## 5 CONCLUSION 

Product spaces enable improved representations by better matching the geometry of the embedding space to the structure of the data. We introduced a tractable Riemannian product manifold class that combines Euclidean, spherical, and hyperbolic spaces. We showed how to learn embeddings and curvatures, estimate the product signature, and defined a tractable formulation of mean. We hope that our techniques encourage further research on non-Euclidean embedding spaces. 

9 

Published as a conference paper at ICLR 2019 

Table 4: Spearman rank correlation on similarity datasets. Top: Previous results from embeddings into spaces of fixed curvature. Bottom: Embeddings into products of H with fixed total dimension. 

||Dim 50<br>Dim 100|
|---|---|
||WS-353<br>Simlex<br>MEN<br>WS-353<br>Simlex<br>MEN|
|Euclidean<br>Hyperbolic|0.6628<br>0.2738<br>0.7217<br>0.6986<br>0.2923<br>0.7473<br>0.6787<br>0.2784<br>0.7117<br>0.6846<br>0.2832<br>0.7217|
|2 Hyperbolics<br>5 Hyperbolics|0.6955<br>**0.2870**<br>0.7246<br>0.7297<br>0.3168<br>0.7450<br>**0.7048**<br>0.2837<br>**0.7270**<br>**0.7379**<br>**0.3212**<br>**0.7530**|



Table 5: Accuracy on the Google word analogy dataset. Taking products of smaller hyperbolic spaces significantly improves performance. Unlike conventional embeddings, the operations in hyperbolic and product spaces are defined solely through distances and manifold operations. 

|Total Dim_d_/ Model|R_d_|(H_d_)1|(H_d/_2)2|(H_d/_5)5|(H2)_d/_2|
|---|---|---|---|---|---|
|50|0.3866|0.3424|0.3928|0.4181|**0.4209**|
|100|**0.5513**|0.3738|0.4310|0.4731|0.5216|



## ACKNOWLEDGMENTS 

We gratefully acknowledge the support of DARPA under Nos. FA87501720095 (D3M) and FA86501827865 (SDH), NIH under No. N000141712266 (Mobilize), NSF under Nos. CCF1763315 (Beyond Sparsity) and CCF1563078 (Volume to Velocity), ONR under No. N000141712266 (Unifying Weak Supervision), the Moore Foundation, NXP, Xilinx, LETI-CEA, Intel, Google, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, the Okawa Foundation, and American Family Insurance, and members of the Stanford DAWN project: Intel, Microsoft, Teradata, Facebook, Google, Ant Financial, NEC, SAP, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of DARPA, NIH, ONR, or the U.S. Government. 

## REFERENCES 

- R. L. Bishop and B. O’Neill. Manifolds of negative curvature. _Trans. American Mathematical Society_ , 145:1–49, 1969. 

- S. Bonnabel. Stochastic gradient descent on Riemannian manifolds. _IEEE Trans. Automatic Control_ , 58(9):2217–2229, 2013. 

- M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst. Geometric deep learning: Going beyond Euclidean data. _IEEE Signal Processing Magazine_ , 34:18–42, 2017. 

- E. Bruni, N.-K. Tran, and M. Baroni. Multimodal distributional semantics. _J. Artificial Intelligence Research_ , 49:1–47, 2014. 

J Burkardt. Cities–city distance datasets. 

- S. Buss and J. P. Fillmore. Spherical averages and applications to spherical splines and interpolation. _ACM Trans. Graphics_ , 20(2):95–126, 2001. 

- B. P. Chamberlain, J. R. Clough, and M. P. Deisenroth. Neural embeddings of graphs in hyperbolic space. _arXiv preprint, arXiv:1705.10359_ , 2017. 

- H. Cho, B. Demeo, J. Peng, and B. Berger. Large-margin classification in hyperbolic space. _CoRR_ , abs/1806.00437, 2018. 

10 

Published as a conference paper at ICLR 2019 

- W. De Nooy, A. Mrvar, and V. Batagelj. _Exploratory social network analysis with Pajek_ . Cambridge University Press, 2011. 

- B. Dhingra, C. J. Shallue, M. Norouzi, A. M. Dai, and G. E. Dahl. Embedding text in hyperbolic spaces. In _TextGraphs@NAACL-HLT_ , 2018. 

- M. P. do Carmo. _Riemannian Geometry_ . Birkh¨auser, 1992. 

- Y. Enokida, A. Suzuki, and K. Yamanishi. Stable geodesic update on hyperbolic space and its application to Poincar´e embeddings. _CoRR_ , abs/1805.10487, 2018. 

- Frederick Arthur Ficken. The Riemannian and affine differential geometry of product-spaces. _Annals of Mathematics_ , pp. 892–913, 1939. 

- L. Finkelstein, E. Gabrilovich, Y. Matias, E. Rivlin, Z. Solan, G. Wolfman, and E. Ruppin. Placing search in context: the concept revisited. In _WWW_ , 2001. 

- P. Fletcher, C. Lu, S. Pizer, and S. Joshi. Principal geodesic analysis for the study of nonlinear statistics of shape. _IEEE Transactions on Medical Imaging_ , 23(8):995–1005, 2004. 

- O. Ganea, G. B´ecigneul, and T. Hofmann. Hyperbolic entailment cones for learning hierarchical embeddings. In _35th International Conference on Machine Learning (ICML)_ , pp. 1646–1655, Stockholm, Sweden, 2018a. 

- Octavian Ganea, Gary B´ecigneul, and Thomas Hofmann. Hyperbolic neural networks. In _Advances in Neural Information Processing Systems_ , pp. 5350–5360, 2018b. 

- C. Gulcehre, M. Denil, M. Malinowski, A. Razavi, R. Pascanu, K. M. Hermann, P. Battaglia, V. Bapst, D. Raposo, A. Santoro, and N. de Freitas. Hyperbolic attention networks. _arXiv preprint, arXiv:1805.09786_ , 2018. 

- Felix Hill, Roi Reichart, and Anna Korhonen. Simlex-999: Evaluating semantic models with (genuine) similarity estimation. _Computational Linguistics_ , 41:665–695, 2015. 

- S. Huckemann, T. Hotz, and A. Munk. Intrinsic shape analysis: Geodesic PCA for Riemannian manifolds modulo isometric Lie group actions. _Statistica Sinica_ , 20(1):1–58, 2010. 

- J. Lamping and R. Rao. Laying out and visualizing large trees using a hyperbolic space. In _Proc. of the 7th annual ACM Symposium on User Interface Software and Technology (UIST 94)_ , pp. 13–14, Marina del Rey, California, 1994. 

- J. Lee. _Riemannian Manifolds: An Introduction to Curvature_ . Springer, 1997. 

- J. Lee. _Introduction to Smooth Manifolds_ . Springer, 2012. 

- M. Leimeister and B. J. Wilson. Skip-gram word embeddings in hyperbolic space. _arXiv preprint, arXiv:1809.01498_ , 2018. 

- Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. Sphereface: Deep hypersphere embedding for face recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 212–220, 2017. 

- J. J. McAuley and J. Leskovec. Learning to discover social circles in ego networks. In _Advances in Neural Information Processing Systems 25 (NIPS 2012)_ , pp. 4599–4607, Lake Tahoe, NV, 2012. 

- Tomas Mikolov, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. _CoRR_ , abs/1301.3781, 2013. 

- M. Nickel and D. Kiela. Poincar´e embeddings for learning hierarchical representations. In _Advances in Neural Information Processing Systems 30 (NIPS 2017)_ , Long Beach, CA, 2017. 

- M. Nickel and D. Kiela. Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. In _35th International Conference on Machine Learning (ICML)_ , pp. 3779–3788, Stockholm, Sweden, 2018. 

11 

Published as a conference paper at ICLR 2019 

- Yann Ollivier. Ricci curvature of Markov chains on metric spaces. _Journal of Functional Analysis_ , 256(3):810–864, 2009. 

- Yann Ollivier. A visual introduction to Riemannian curvatures and some discrete generalizations. _Analysis and Geometry of Metric Measure Spaces: Lecture Notes of the 50th S´eminaire de Math´ematiques Sup´erieures (SMS), Montr´eal_ , 56:197–219, 2011. 

- X. Pennec. Hessian of the Riemannian squared distance: Supplement A of barycentric subspace analysis on manifolds. 

- X. Pennec. Barycentric subspace analysis on manifolds. _The Annals of Statistics_ , 46(6A):2711– 2746, 2018. 

- F Sala, C. De Sa, A. Gu, and C. R´e. Representation tradeoffs for hyperbolic embeddings. In _35th International Conference on Machine Learning (ICML)_ , pp. 4460–4469, Stockholm, Sweden, 2018. 

- Y. Tay, L. A. Tuan, and S. C. Hui. Hyperbolic representation learning for fast and efficient neural question answering. In _Proc. of the Eleventh ACM International Conference on Web Search and Data Mining (WSDM 2018)_ , pp. 583–591, Los Angeles, California, 2018. 

- William P Thurston. Shapes of polyhedra and triangulations of the sphere. _Geometry and Topology monographs_ , 1:511–549, 1998. 

Pavan K Turaga and Anuj Srivastava. _Riemannian computing in computer vision_ . Springer, 2016. 

- C. Udriste. _Convex Functions and Minimization Methods on Riemannian Manifolds_ . Springer, 1994. 

- J. A. Walter. H-MDS: a new approach for interactive visualization with multidimensional scaling in the hyperbolic space. _Information Systems_ , 29(4):273–292, 2004. 

- D. J. Watts and S. H. Strogatz. Collective dynamics of small-world networks. _Nature_ , 393:440–442, 1998. 

Melanie Weber, Emil Saucan, and J¨urgen Jost. Characterizing complex networks with forman-ricci curvature and associated geometric flows. _Journal of Complex Networks_ , 5(4):527–550, 2017. 

Benjamin Wilson and Matthias Leimeister. Gradient descent in hyperbolic space. _arXiv preprint arXiv:1805.08207_ , 2018. 

- R.C. Wilson, E.R. Hancock, E. Pekalska, and R. Duin. Spherical and hyperbolic embeddings of data. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 36(11):2255–2269, 2014. 

- H. Zhang and S. Sra. First-order methods for geodesically convex optimization. In _Proc. of the 29th Conference on Learning Theory (COLT)_ , pp. 1617–1638, New York, NY, 2016. 

- H. Zhang and S. Sra. Towards Riemannian accelerated gradient methods. _CoRR_ , abs/1806.02812, 2018. 

- H. Zhang, S. J. Reddi, and S. Sra. Riemannian SVRG: Fast stochastic optimization on Riemannian manifolds. In _Advances in Neural Information Processing Systems 29 (NIPS 2016)_ , pp. 4599– 4607, Barcelona, Spain, 2016. 

12 

Published as a conference paper at ICLR 2019 

The Appendix starts with a glossary of symbols and a discussion of related work. Afterwards, we provide the proof of Lemma 2. We continue with a more in-depth treatment of the curvature estimation algorithm. We then introduce two combinatorial constructions—embedding techniques that do not require optimization—that rely on the alternative product distances. We give additional details on our experimental setup. Finally, we additionally evaluate the interpretability of these embeddings (i.e., do the separate components in the embedding manifold capture intrinsic qualities of the data?) through visualizations of the synthetic example from Figure 1. 

## A GLOSSARY OF SYMBOLS 

We provide a glossary of commonly-used terms in our paper. 

|Symbol|Used for|
|---|---|
|mAP(_f_)|the mean average precision fdelity measure of the embedding_f_|
|_D_(_f_)|the distortion fdelity measure of the embedding_f_|
|_D_wc(_f_)|the worst-case distortion fdelity measure of the embedding_f_|
|_G_|a graph, typically with node set_V_ and edge set_E_|
|_T_|a tree|
|_a, b, c_|nodes in a graph or tree|
|_f_|an embedding|
|_Na_|neighborhood around node_a_in a graph|
|_Ra,b_|the smallest set of closest points to node_a_in an embedding_f_ that contains node_b_|
|_M_|a manifold; when equipped with a metric_g_,_M_ is Riemannian|
|_p_|a point in a manifold,_p ∈M_|
|_TpM_|the tangent space of point_p_in_M_ (a vector space)|
|_g_|a Riemannian metric defning an inner product on_TpM_|
|E_d_|_d_-dimensional Euclidean space|
|S_d_|_d_-dimensional spherical space|
|H_d_|_d_-dimensional hyperbolic space|
|_P_|product manifold consisting of spherical, Euclidean, hyperbolic factors|
|Exp_x_(_v_)|the exponential map for tangent vector_v_at point_x_|
|_R_|the Riemannian curvature tensor|
|_K_(_x, y_)|the sectional curvature for a subspace spanned by linearly independent_x, y ∈TpM_|
|_dE_|metric distance between two points in Euclidean space|
|_dS_|metric distance between two points in spherical space|
|_dH_|metric distance between two points in hyperbolic space|
|_dU_|metric distance between two points in metric space_U_|
|_dG_|metric distance between two points in a graph_G_= (_V, E_)|
|_µ_(_T_)|mean of a set of points_T_ =_{p_1_, . . . , pn}_in_P_|
|I_n_|the_n × n_identity matrix|



Table 6: Glossary of variables and symbols used in this paper. 

## B RELATED WORK 

Hyperbolic space has recently been proposed as an alternative to Euclidean space to learn embeddings in cases where there is a (possibly latent) hierarchical structure. In fact, many types of data (from various domains) such as social networks, word frequencies, metabolic-mass relationships, and phylogenetic trees of DNA sequences exhibit a non-Euclidean latent structure, as shown in Bronstein et al. (2017). 

Initial works on hyperbolic embeddings include Nickel & Kiela (2017) and Chamberlain et al. (2017). In Chamberlain et al. (2017), neural graph embeddings are performed in hyperbolic space and used to classify the vertices of complex networks. A similar application is link prediction in Nickel & Kiela (2017) for the lexical database WordNet; this work also measured predicted lexical entailment on the HyperLex benchmark dataset. The follow-up work Nickel & Kiela (2018) performs optimizations in the hyperboloid (i.e. Lorentz) model instead of the Poincar´e model. 

13 

Published as a conference paper at ICLR 2019 

Tay et al. (2018) proposed a neural ranking based question answering (Q/A) system in hyperbolic space that outperformed many state-of-the-art models using fewer parameters compared to competitor learning models. Ganea et al. (2018a) proposed hyperbolic embeddings of entailment relations, described by directed acyclic graphs by applying hyperbolic cones as a heuristic and showed improvements over baselines in terms of representational capacity and generalization. Sala et al. (2018) developed a combinatorial construction for efficiently embedding trees and tree-like graphs without optimization, studied the fundamental tradeoffs of hyperbolic embeddings, and explored PCA-like algorithms in hyperbolic space. 

Unlike Euclidean space, most Riemannian manifolds are not vector spaces, and thus even basic operations such as vector addition, vector translation and matrix multiplication do not have universal interpretations. In more complex geometries, closed form expressions for basic objects like distances, geodesics, and parallel transport do not exist. As a result, standard machine learning or deep learning tools, such as convolutional neural networks, long short term memory networks (LSTMs), logistic regression, support vector machines, and attention mechanisms, do not have exact correspondences in these complex geometries. 

A pair of recent approaches seek to formulate standard machine learning methods in hyperbolic space. Gulcehre et al. (2018) introduces a hyperbolic version of the attention mechanism using the hyperboloid model. This work shows improvements in terms of generalization on several downstream applications including neural machine translation, learning on graphs and visual question answering tasks, while having compact neural representations. Ganea et al. (2018b) formulates basic machine learning tools in hyperbolic space including multinomial logistic regression, feed-forward and recurrent neural networks like gated recurrent units and LSTMs in order to embed sequential data and perform classification in hyperbolic space. They demonstrate empirical improvements on textual entailment and noisy-prefix recognition tasks using hyperbolic sentence embeddings. Cho et al. (2018) introduced a hyperbolic formulation for support vector machine classifiers and demonstrated performance improvements for multi-class prediction tasks on real-world complex networks as well as simulated datasets. 

Zipf’s law states that word-frequency distributions obey a power law, which defines a hierarchy based on semantic specificities. Concretely, semantically general words that occur in a wider range of contexts are closer to the root of the hierarchy while rarer words are further down in the hierarchy. In order to capture the latent hierarchy in the natural language, there has been several proposals for training word embeddings in hyperbolic space. Dhingra et al. (2018) trains word embeddings using the algorithm from Nickel & Kiela (2017). They show that resulting hyperbolic word embeddings perform better on inferring lexical entailment relation than Euclidean embeddings trained with skip-gram model which is a standard method for training word embeddings, initially proposed by Mikolov et al. (2013). Leimeister & Wilson (2018) formulated the skip-gram loss function in hyperboloid model of hyperbolic space and evaluated on the standard the intrinsic evaluation tasks for word embeddings such as similarity and analogy in hyperbolic space. 

Finally, the popularity of hyperbolic embeddings has stimulated interest in descent methods suitable for hyperbolic space optimization. In addition to tools like Bonnabel (2013) and Zhang et al. (2016), Zhang & Sra (2016) offers convergence rate analysis for a variety of algorithms and settings for Hadamard manifolds. Enokida et al. (2018) proposes an explicit update rule along geodesics in a hyperbolic space with a theoretical guarantee on convergence, and Zhang & Sra (2018) introduces an accelerated Riemannian gradient methods. 

Our work also touches on previous work on maximum distance scaling (MDS) and PCA-like algorithms in hyperbolic, spherical, and more general manifolds. MDS-like algorithms in hyperbolic space are developed for visualization in Walter (2004) and Lamping & Rao (1994). Embeddings into spherical or into hyperbolic space with a PCA-like loss function were developed in Wilson et al. (2014). General forms of PCA include Geodesic PCA (Huckemann et al., 2010) and principal geodesics analysis (PGA) (Fletcher et al., 2004). A very general study of PCA-like algorithms is found in Pennec (2018). 

## C MANIFOLD CONCEPTS AND PROOFS 

Below, we include proofs of our results and further discuss manifold notions such as curvature. 

14 

Published as a conference paper at ICLR 2019 

## C.1 MEANS IN PRODUCT SPACES 

We begin with Lemma 2, restated below for convenience. 

**Lemma 2.** _Let P be a product of model spaces of total dimension r, T_ = _{p_ 1 _, . . . , pn} points in P and w_ 1 _, . . . , wn weights satisfying wi ≥_ 0 _and_[�] _[n] i_ =1 _[w][i]_[=][1] _[.][Moreover,][let the components of] the points in P, pi|_ S _j restricted to each spherical component space_ S _[j] fall in one hemisphere of_ S _[j] . Then, Riemannian gradient descent recovers the mean µ_ ( _T_ ) _within distance ϵ in time O_ ( _nr_ log _ϵ[−]_[1] ) _._ 

_Proof._ Consider the squared distance _d_[2] ( _p, q_ ) for _p, q ∈ M_ for a manifold _M_ . Fix _q_ . We denote the Hessian in _p_ by _Hp,M_ ( _q_ ). Then, we have the following expressions for the Hessian of the squared distance of a sphere, derived in Pennec 

**==> picture [194 x 22] intentionally omitted <==**

where I _r_ is the identity matrix, _θ_ = acos( _p · q_ ) is the distance _d_ S _r_ ( _p, q_ ) and _u_ = (I _r − pp[T]_ ) _q/_ sin _θ_ . In Pennec, it is shown that the eigenvectors of _Hp,_ S _r_ ( _q_ ) are 0 _,_ 1, and _θ_ cot( _θ_ ); thus the Hessian is bounded and if _θ ∈_ [0 _, π/_ 2], it is also positive definite (PD). 

For hyperbolic space (under the hyperboloid model), the Hessian is 

**==> picture [210 x 13] intentionally omitted <==**

Here _θ_ = acosh( _−⟨p, q⟩∗_ ), _J_ is the matrix associated with the Minkowski inner product, i.e., _⟨p, q⟩∗_ = _p[T] Jq_ , and _u_ = log _p_ ( _q_ ) _/θ_ . The log here refers to the logarithmic map. That is, if _q_ = exp _p_ ( _v_ ), then _v_ = log _p_ ( _q_ ). Moreover, exact expressions for the eigenvalues of _Hp,_ H _r_ ( _q_ ) in terms of _θ_ imply that it is always bounded and PD. 

The Hessian for Euclidean space is _Hp,_ E _[r]_ ( _q_ ) = 2I _r_ , which is also PD. 

Now we can express the Hessian of the weighted mean. We write _Hp,P_ for the Hessian of the weighted variance[�] _[n] i_ =1 _[w][i][d]_[2] _P_[(] _[p, p][i]_[)][(recall][that] _[µ]_[(] _[p]_[1] _[, . . . , p][n]_[)][=][arg min] _[p]_ � _ni_ =1 _[w][i][d]_[2] _P_[(] _[p, p][i]_[)][).] We have, by the decomposability of the distance, that 

**==> picture [332 x 32] intentionally omitted <==**

Taking the Hessian, 

**==> picture [254 x 31] intentionally omitted <==**

Now, by assumption, the spherical components for our points in each of the spheres, _pi|_ S _j_ , fall within one hemisphere, and we may initialize our gradient descent (that is, our _p_ 0) within this hemisphere. Then, the angle _θ_ in each of the spherical distances is in [0 _, π/_ 2], so that the corresponding Hessians are PD. 

Since each term in the sum is PD and the weights satisfy _wi ≥_ 0, with at least one positive weight, _Hp,P_ is also PD. Moreover, these Hessians are bounded. Then we apply Theorem 4.2 (Chap. 7.4) in Udriste (1994)), which shows linear rate convergence, as desired. 

## C.2 CURVATURE ESTIMATION 

We discuss the notions of curvature relevant to our product manifold in more depth. We start with a high-level overview of various definitions of curvature. Afterwards, we introduce the formal definitions for curvature and apply them to the product construction. 

**Definitions of Curvature** There are multiple notions of curvature, with varying granularity. Some of these notions are suitable for working with manifolds abstractly (without reference to an ambient space, that is, intrinsic). Others, in particular older definitions pre-dating the development of 

15 

Published as a conference paper at ICLR 2019 

the formal mechanisms underpinning differential geometry, require the use of the ambient space. Gauss defined the first intrinsic notion of curvature, _Gaussian_ curvature. It is the product of the _principal curvatures_ , which can be thought of as the smallest and largest curvature in different directions.[4] Below we consider several such notions. 

_Scalar_ curvature is a single value associated with a point _p ∈ M_ and intuitively relates to the area of geodesic balls. Negative curvature means volumes grow faster than in Euclidean space, positive means volumes grow slower. 

A more fine-grained notion of curvature is that of _sectional curvature_ : it varies over all “sheets” passing through _p_ . Note that curvature is inherently a notion of two-dimensional surfaces, and the sectional curvature fully captures the most general notion of curvature (the Riemannian curvature tensor). More formally, for every two dimensional subspace _U_ of the tangent space _TpM_ , the sectional curvature _K_ ( _U_ ) is equal to the Gaussian curvature of the sheet Exp _p_ ( _U_ ). Intuitively, it measures how far apart two geodesics emanating from _p_ diverge. In positively curved spaces like the sphere, they diverge more slowly than in flat Euclidean space. 

The Ricci curvature of a tangent vector _v_ at _p_ is the average of the sectional curvature _K_ ( _U_ ) over all planes _U_ containing _v_ . Geometrically the Ricci curvature measures how much the volume of a small cone around direction _v_ compares to the corresponding Euclidean cone. Positive curvature implies smaller volumes, and negative implies larger. Note that this is natural from the way geodesics bend in various curvatures. The scalar curvature is in fact defined as an average over the Ricci curvature, giving the intuitive relation between scalar curvature and volume. It is thus also an average over the sectional curvature. 

**Discrete Analogs of Curvature** Discrete data such as graphs do not have manifold structure. The goal of curvature analogs such as _ξ_ is to provide a discrete analog of curvature which satisfies similar properties to curvature; we use this to facilitate choosing an appropriate Riemannian manifold to embed discrete data into. In this work, we focus on the sectional curvature, but discrete versions of other curvatures have been proposed such as the the Forman-Ricci (Weber et al., 2017) and OllivierRicci (Ollivier, 2009) curvatures. 

The input to the discrete curvature estimation from Section 3.2 is analogous to other discrete curvature analogs. For example, the Ricci curvature is defined for a point _p_ and a tangent vector _u_ , and the coarse Ricci curvature is defined for a node _p_ and neighbor _x_ (Ollivier, 2011). Similarly, the sectional curvature is defined for a point and two tangent vectors, and _ξ_ is defined for a a node and two neighbors. 

**Sectional Curvature in Product Spaces** Now we are ready to tackle the question of curvature in our proposed product space. Let _M_ be our Riemannian manifold and _X_ ( _M_ ) be the set of vector fields on _M_ . The curvature _R_ of _M_ assigns a function _R_ ( _X, Y_ ) : _X_ ( _M_ ) _→X_ ( _M_ ) to each pair of vector fields ( _X, Y_ ) _∈X × X_ . For a vector field _Z_ in _X_ ( _M_ ), the function _R_ ( _X, Y_ ) can be written 

**==> picture [198 x 12] intentionally omitted <==**

Here _∇_ is the Riemannian connection for the manifold _M_ , and [ _X, Y_ ] is the Lie bracket of the vector fields _X, Y_ . 

For convenience, we shall write the inner product _⟨R_ ( _X, Y_ ) _Z, T ⟩_ as ( _X, Y, Z, T_ ); this is the _Riemannian curvature tensor_ . Then, the sectional curvature is defined as follows. Let us take _V_ to be a two-dimensional subspace of _TpM_ and _x, y ∈V_ be linearly independent (so that they span _V_ ). Then, the sectional curvature at _p_ for subspace _V_ is 

**==> picture [266 x 25] intentionally omitted <==**

The model spaces S _,_ H _,_ E are the spaces of constant curvature, where _K_ is constant for all points _p_ and 2-subspaces _V_ . 

> 4The curvature of a curve can be found by considering the _osculating circles_ which match it to second order. 

16 

Published as a conference paper at ICLR 2019 

For simplicity, suppose we are working with _M_ = _M_ 1 _× M_ 2; the approach extends easily for larger products. We write _x_ = ( _x_ 1 _, x_ 2) for _x ∈ TpM_ . Similarly, let _R_ 1 _, R_ 2 be the curvatures and _K_ 1 _, K_ 2 be the sectional curvatures of _M_ 1 _, M_ 2 at _p_ , respectively. Then the curvature tensor decomposes as 

**==> picture [294 x 11] intentionally omitted <==**

Our goal is to evaluate the sectional curvature _K_ (( _x_ 1 _, x_ 2) _,_ ( _y_ 1 _, y_ 2)) for the product manifold _M_ . We show the following, re-stated for convenience: 

**Lemma 1.** _Let M_ = _M_ 1 _× M_ 2 _where Mi has constant curvature Ki. For any u, v ∈ TpM , if K_ 1 _, K_ 2 _are both non-negative, the sectional curvature satisfies K_ ( _u, v_ ) _∈_ [0 _,_ max _{K_ 1 _, K_ 2 _}_ ] _. If K_ 1 _, K_ 2 _are both non-positive, the sectional curvature satisfies K_ ( _u, v_ ) _∈_ [min _{K_ 1 _, K_ 2 _},_ 0] _. If Ki <_ 0 _and Kj >_ 0 _for i ̸_ = _j, then K_ ( _u, v_ ) _∈_ [ _Ki, Kj_ ] _._ 

_Proof._ Let us start with the numerator of equation (4): 

**==> picture [205 x 11] intentionally omitted <==**

**==> picture [182 x 39] intentionally omitted <==**

Here, we used equation 5 in the third line. 

Note that when _x_ 1 _, y_ 1 are linearly independent, then _⟨R_ 1( _x_ 1 _, y_ 1) _x_ 1 _, y_ 1 _⟩_ = _K_ 1( _∥x_ 1 _∥_[2] _∥y_ 1 _∥_[2] _− ⟨x_ 1 _, y_ 1 _⟩_[2] ) by (4). Otherwise, this still holds since it is 0. So we can relate the above to _K_ 1 _, K_ 2: 

( _x, y, x, y_ ) = _K_ 1( _∥x_ 1 _∥_[2] _∥y_ 1 _∥_[2] _−⟨x_ 1 _, y_ 1 _⟩_[2] ) + _K_ 2( _∥x_ 2 _∥_[2] _∥y_ 2 _∥_[2] _−⟨x_ 2 _, y_ 2 _⟩_[2] ) _._ 

For convenience, we write _αi_ = _∥xi∥_[2] _∥yi∥_[2] _−⟨xi, yi⟩_[2] for _i_ = 1 _,_ 2. Then the numerator is simply _K_ 1 _α_ 1 + _K_ 2 _α_ 2. Next, we consider the denominator of (equation 4): 

**==> picture [326 x 57] intentionally omitted <==**

where we set _β_ = _∥x_ 1 _∥_[2] _∥y_ 2 _∥_[2] + _∥x_ 2 _∥_[2] _∥y_ 1 _∥_[2] . Thus, we have that 

**==> picture [311 x 23] intentionally omitted <==**

Now, note that _β >_ 0, since we assumed that _x_ 1 _, y_ 1 and _x_ 2 _, y_ 2 are linearly independent. By CauchySchwarz, _αi ≥_ 0. Then, if _Ki ≥_ 0, we have that ( _αiKi_ ) _/_ ( _α_ 1 + _α_ 2 + _β_ ) _≤_ ( _αiKi_ ) _/_ ( _α_ 1 + _α_ 2), so that 

**==> picture [314 x 21] intentionally omitted <==**

Thus, we relate the product sectional curvature to a convex combination of the factor sectional curvatures _K_ 1 _, K_ 2. We have for non-negative _K_ 1 _, K_ 2 (e.g., Euclidean and spherical spaces) that _K_ (( _x_ 1 _, x_ 2) _,_ ( _y_ 1 _, y_ 2)) _∈_ [0 _,_ max _{K_ 1 _, K_ 2 _}_ ]. A similar result holds for the non-positive (Euclidean and hyperbolic) case. The last case (one negative, one positive space) follows along the same lines. 

**Distribution of** _K_ The range of curvatures from Lemma 1 can be easily extended to a more refined distributional analysis. In particular, consider sampling any point _p_ and a random plane _V ⊆ TpM_ . By homogeneity, we can equivalently fix _p_ . The 2-subspaces of _TpM ≃_ R _[d]_ forms the Grassmannian manifold **Gr** (2 _, TpM_ ). The uniform measure on this (i.e. invariant to multiplication by an orthogonal matrix) can be recovered from the Haar measure on the orthogonal group O( _d_ ), which 

17 

Published as a conference paper at ICLR 2019 

**Algorithm 2** Sectional curvature distribution 

1: **Input: Dimensions** _d_ 1 _, d_ 2 2: _a_ 1 _← χ_[2] ( _d_ 1 _−_ 1) 3: _b_ 1 _← χ_[2] ( _d_ 1 _−_ 1) 4: _t_ 1 _←_ Beta(( _d_ 1 _−_ 1) _/_ 2 _,_ ( _d_ 1 _−_ 1) _/_ 2) 5: _c_ 1 _← a_[1] 1 _[/]_[2] _b_[1] 1 _[/]_[2] (2 _t_ 1 _−_ 1) 6: _a_ 2 _← χ_[2] ( _d_ 2 _−_ 1) 7: _b_ 2 _← χ_[2] ( _d_ 2 _−_ 1) 8: _t_ 2 _←_ Beta(( _d_ 2 _−_ 1) _/_ 2 _,_ ( _d_ 2 _−_ 1) _/_ 2) 9: _c_ 2 _← a_[1] 2 _[/]_[2] _b_[1] 2 _[/]_[2] (2 _t_ 2 _−_ 1) 10: _α_ 1 _← a_ 1 _b_ 1 _− c_[2] 1 11: _α_ 2 _← a_ 2 _b_ 2 _− c_[2] 2 12: _β ← a_ 1 _b_ 2 + _a_ 2 _b_ 1 13: **return** _α_ 1+ _αα_ 12+ _β[K]_[1][ +] _α_ 1+ _αα_ 22+ _β[K]_[2] 

itself can be constructed by orthonormalizing independent random normal vectors. In particular, it suffices to consider _V_ spanned by independent Gaussians _x, y ∼N_ (0 _, I_ ). 

Furthermore, we do not actually need to sample _d_ -dimensional vectors _x, y_ to compute the relevant curvature in equation 6. It suffices to sample the quantitities _⟨x_ 1 _, y_ 1 _⟩, ⟨x_ 1 _, x_ 1 _⟩, ⟨y_ 1 _, y_ 1 _⟩_ and _⟨x_ 2 _, y_ 2 _⟩, ⟨x_ 2 _, x_ 2 _⟩, ⟨y_ 2 _, y_ 2 _⟩_ directly. Note that _α_ := _⟨x_ 1 _, x_ 1 _⟩_ and _β_ := _⟨y_ 1 _, y_ 1 _⟩_ are _χ_[2] -distributed, while _⟨x_ 1 _, y_ 1 _⟩∼[√] αβγ_ , where _γ_ is distributed as the dot product of two uniformly random unit vectors. By rotational invariance, this is the same as the first coordinate of a random unit vector, which in turn is distributed as _X_ 1[2] _[/]_[(] _[X]_ 1[2][+] _[· · ·]_[+] _[X] d_[2][)][for][independent][normal] _[X][i]_[,][and][therefore] ( _γ_ + 1) _/_ 2 _∼_ Beta(( _d −_ 1) _/_ 2 _,_ ( _d −_ 1) _/_ 2). 

Thus a random _K_ ( _V_ ) can be computed by sampling from well known distributions in constant time, via Algorithm 2. 

Furthermore, without knowing _K_ 1 _, K_ 2 a priori, an estimate for these curvatures can be found by matching the distribution of sectional curvature from Algorithm 2 to the empirical curvature computed from Algorithm 3. In particular, Algorithm 2 can be used to generate samples for _α_ 1+ _αα_ 12+ _β_ and _α_ 2[The overall moments are then simple functions of] _[ K]_[1] _[, K]_[2][, and the sample moments] _α_ 1+ _α_ 2+ _β_[.] of the above quantities, so that _K_ 1 _, K_ 2 can then be found by matching moments. 

**Curvature Estimation** We prove the facts we mentioned in the main body of the paper relating to the evaluation of _ξ_ over fundamental pieces of graphs: lines, cycles, and trees. 

**Lemma 3.** _Suppose a lies on the same geodesic line as b, m, c; in other words, WLOG dG_ ( _a, b_ ) _≤ dG_ ( _a, c_ ) _and suppose dG_ ( _a, c_ ) = _dG_ ( _a, b_ ) + _dG_ ( _b, m_ ) + _dG_ ( _m, c_ ) _. Then ξ_ ( _m_ ; _b, c_ ; _a_ ) = 0 _._ 

**Lemma 4.** _Consider a cycle graph C with nodes b, m, c such that_ ( _m, b_ ) _and_ ( _m, c_ ) _are neighbors. Then for all a ∈ C, ξ_ ( _m_ ; _b, c_ ; _a_ ) _is either_ 0 _or positive._ 

_Proof._ Without loss of generality, let the cycle have an even number of vertices _n_ . Let _k_ be the node diametrically opposite from _m_ . Note that for any vertex _a ̸_ = _n_ , _a, b, m, c_ lie on a geodesic line, and therefore _ξ_ ( _m_ ; _b, c_ ; _a_ ) = 0. On the other hand, 

**==> picture [306 x 25] intentionally omitted <==**

The case when _n_ is odd is similar, where we find that two nodes _a_ satisfy _ξ_ ( _m_ ; _b, c_ ; _a_ ) = _n/_ ( _n −_ 1) and the rest are 0. 

**Lemma 5.** _Consider a tree graph T with nodes b, m, c such that_ ( _m, b_ ) _and_ ( _m, c_ ) _are neighbors. Then for all a ∈ T , ξ_ ( _m_ ; _b, c_ ; _a_ ) _is either_ 0 _or negative._ 

18 

Published as a conference paper at ICLR 2019 

**Algorithm 3** Empirical estimation of sectional curvature distribution 

1: **Input: Graph** _G_ = ( _V, E_ ) 

2: _m ←_ Uniform( _V_ ) 

3: _b ←_ Uniform( _N_ ( _m_ )) _{N_ ( _v_ ) is the neighbor set of _v}_ 

4: _c ←_ Uniform( _N_ ( _m_ )) 

5: _a ←_ Uniform( _V_ ) 

6: _K ← ξ_ ( _m_ ; _b, c_ ; _a_ ) 

- 7: **return** _K_ 

_Proof._ Due to the tree structure, _a_ is either geodesic with _b, m, c_ , or _a_ is connected to _m_ with a path that does not pass through _b_ or _c_ . In the former case, _ξ_ ( _m_ ; _b, c_ ; _a_ ) = 0. In the latter case, 

**==> picture [188 x 22] intentionally omitted <==**

where _d_ = _dG_ ( _a, m_ ). 

Given a graph, the distribution of _ξ_ ( _m_ ; _b, c_ ) over random “planes” (i.e. pairs of neighbors _b, c_ ) is easily calculable. This yields a distribution that can then be averaged over _m_ to obtain an average sectional curvature distribution. To simplify this, we find the distribution via sampling (Algorithm 3) in the calculations for Table 3, before being fed into Algorithm 2 to estimate _Ki_ . 

As a corollary of Lemma 5, note that the _ξ_ becomes more negative for trees of higher degree, matching the intuition that higher degree trees are more appropriate for hyperbolic space embeddings (Sala et al., 2018). 

We additionally note that a line of work has studied discrete analogs of curvature for regular objects such as triangular planar tessellations (including polyhedra) (Thurston, 1998). For example, these notions assign positive curvature to regular polyhedra (the tetrahedron, octahedron, icosahedron), zero curvature to the flat planar tessellation, and positive curvature to the order-7 triangular tiling of the hyperbolic plane. It is easily checked that Algorithm 3 assigns the right curvature sign to each of these objects. 

## C.3 COMBINATORIAL CONSTRUCTIONS 

The _ℓ_ 1 and min-based distances are suitable for combinatorial constructions where we do not learn embeddings by optimizing a surrogate loss function, but rather by directly placing points in the product space, often via recursive procedures. Such constructions offer superior speed and other benefits. On the other hand,they are only available for certain classes of graphs. Additionally, since the constructions rely on the _ℓ_ 1 and min distances, they do not take advantage of the Riemannian structure, and thus do not have the same applicability downstream. 

We often exploit the combinatorial construction for trees from Sala et al. (2018) as a building block; it offers worst-case distortion[5] 1 + _ε_ when embedding a tree into H _[r]_ for all _r ≥_ 2, where we can control _ε_ . 

**Hanging Tree Construction** Consider the class of graphs _G_ = ( _V, E_ ) where _V_ = _B ∪ T_ 1 _∪ T_ 2 _. . . ∪ T|B|_ , so that _B_ is a base set of nodes and for each node _a ∈ B_ , there is a tree _Ta_ connected to _a_ (the hanging trees). We show how to use _P_ with the _ℓ_ 1 distance to reduce the cost of embedding _G_ to that of embedding the subgraph induced by _B_ . 

We embed _G_ into the product space _P_ = _P[′] ×_ H _[r]_ equipped with the _ℓ_ 1 distance. Here, _P[′]_ is some product manifold. We do the embedding in two steps: 

> 5The worst-case distortion _D_ wc is a commonly-considered variant of distortion 

**==> picture [195 x 16] intentionally omitted <==**

**==> picture [155 x 10] intentionally omitted <==**

The worst-case distortion is the ratio of the worst expansion and the worst contraction of distances; note that it is scale-invariant. Here, the best worst-case distortion is _D_ wc( _f_ ) = 1. 

19 

Published as a conference paper at ICLR 2019 

1. Embed the subgraph induced by _B_ into _P[′]_ by any method; let the resulting worst-case distortion be 1 + _δ_ . Embed every node in _Ti_ into the embedded image of node _i_ , 

2. Form the tree _T_ by connecting each of the _T_ 1 _, . . . , T|B|_ to a single node (equivalent to crushing all the nodes in _B_ into a single node), and embed _T_ into H _[r]_ by using the combinatorial construction. Additionally, all of the nodes in _B_ are embedded into the image of the single central node in H _[r]_ . 

We can check the distortion. For nodes _xa, yb_ in subtrees hanging off _a, b ∈ B_ , the distance is _dG_ ( _xa, yb_ ) = _dT_ ( _xa, yb_ )+ _dB_ ( _x, y_ ). Since the distortion for the two embeddings are given by 1+ _δ_ and 1 + _ε_ , it is easy to check that the overall distortion is at most max _{_ 1 + _δ,_ 1 + _ε}_ . 

As a concrete example, consider the ring of trees in Figure 1. Then, _B_ = _Cr_ , the cycle on _r_ nodes. In this case, we can embed _B_ into _P[′]_ = S[1] . Let the nodes of _B_ be indexed _a_ 1 _, . . . , ar_ . We embed _ai_ into _Ai_ = (cos([2] _d[πi]_[)] _[,]_[ sin(][2] _d[πi]_[))][.][Then, for] _[ i < j]_[,] 

_dS_ ( _Ai, Aj_ ) = acos( _Ai · Aj_ ) 

**==> picture [247 x 77] intentionally omitted <==**

Thus indeed, the embedding has worst-case distortion 1. Thus, the overall distortion for the ring of trees is 1 + _ε_ . Since we control _ε_ , we can achieve arbitrarily good distortion for the ring of trees. The complexity of this algorithm is linear in the number of nodes, since embedding the trees and ring is linear time. 

**General Graph Construction** Now we use the min distance space to construct an embedding of _any_ graph _G_ on _r_ nodes with arbitrarily low distortion via the space _P_ = H[2] _×_ H[2] _× . . . ×_ H[2] with _r −_ 1 copies. As we shall see, this construction is ideal (arbitrarily low distortion, any graph) other than requiring _O_ ( _r_ ) spaces. 

Let the nodes of _G_ be _V_ = _{a_ 1 _, . . . , ar}_ . Now, for each _ai_ , 1 _≤ i ≤ r −_ 1, form the minimum distance tree _Ti_ rooted at _ai_ . Then, embed _Ti_ into the _i_ th copy of H[2] via the combinatorial construction. Then, for any nodes _ai, aj ∈ V_ , the distance _dG_ ( _ai, aj_ ) is attained by _dTi_ ( _ai, aj_ ), or _dTj_ ( _aj, ai_ ) in the case _i_ = _r_ . Since at least one of _Ti_ or _Tj_ , say _Ti_ , is embedded in H[2] with distortion 1 + _ε_ , if we make _ε_ small enough, the smallest distance among the embedded copies is indeed that for _Ti_ , so our overall distortion is still 1 + _ε_ . 

## D VISUALIZATIONS AND INTERPRETABILITY 

The combinatorial construction using the _ℓ_ 1 distance (Section C.3) can embed the hanging tree graph arbitrarily well, unlike any single type of space. Unlike a single space, this also lends more interpretability to the embedding, as each component displays different qualitative aspects of the underlying graph structure. Figure 4 shows that this phenomenon does in fact happen empirically, even using the optimization approach over the _ℓ_ 2 (Riemannian) instead of _ℓ_ 1 distance. 

## E EXPERIMENTAL DETAILS 

We provide some additional details for our experimental setups. 

**Graph Reconstruction** The optimization framework was implemented in PyTorch. The loss function (2) was optimized with SGD using minibatches of 65536 edges for the real-world datasets, and ran for 2000 epochs. For the Cities graph, the learning rate was chosen among 

20 

Published as a conference paper at ICLR 2019 

Figure 4: Ring of trees graph embedding into (H[2] )[1] _×_ (S[1] )[1] ; left: early epoch, right: completion. Only accessing graph distances, the optimization separates the different intrinsic structures of the underlying graph—the cycle and the trees—into interpretable components. Compare to Figure 2. 

_{_ 0 _._ 001 _,_ 0 _._ 003 _,_ 0 _._ 01 _}_ . For the rest of the datasets, the learning rate was chosen from a grid search among _{_ 10 _,_ 30 _,_ 100 _,_ 300 _,_ 1000 _}_ for each method.[6] 

Each point in the embedding is initialized randomly according to a uniform or Normal distribution in each coordinate with standard deviation 10 _[−]_[3] . (In the hyperboloid and spherical models, all but the first coordinate is chosen randomly, and the first coordinate is a function of the rest.) 

Table 2 uses only Algorithm 1, and initializes the curvatures to _−_ 1 for hyperbolic components and 1 for spherical components. These curvatures are learned using the method described in Section 3.1, and the “Best model” row reports the final curvatures of the best signature. 

**Word Embeddings** Following LW, the input corpus is a 2013 dump of Wikipedia that has been preprocessed by lower casing and removing punctuation, and filtered to remove articles few page views. All other hyperparameters are chosen exactly as in as LW, including their numbers for Euclidean embeddings from _fastText_ . The datasets used for similarity (WS-353, Simlex-999, MEN) and analogy (Google) are also identical to the previous setup. 

> 6Note that the high LR stems from the particular choice of normalization for (2) in our implementation. 

21 

