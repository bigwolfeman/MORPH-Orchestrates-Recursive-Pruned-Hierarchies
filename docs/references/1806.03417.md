# Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry

- **Authors:** Maximilian Nickel, Douwe Kiela (Facebook AI Research)
- **Year:** 2018 (ICML 2018, pp. 3776-3785)
- **Source:** https://arxiv.org/abs/1806.03417
- **MORPH uses:** The Lorentz (hyperboloid) model of hyperbolic geometry for embedding the hierarchical component of the hybrid token embedding, enabling compact representation of power-law / tree-structured semantic relationships that Euclidean embeddings require far more dimensions to approximate.

---

## **Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

## **Maximilian Nickel**[1] **Douwe Kiela**[1] 

## **Abstract** 

We are concerned with the discovery of hierarchical relationships from large-scale unstructured similarity scores. For this purpose, we study different models of hyperbolic space and find that learning embeddings in the Lorentz model is substantially more efficient than in the Poincaré-ball model. We show that the proposed approach allows us to learn high-quality embeddings of large taxonomies which yield improvements over Poincaré embeddings, especially in low dimensions. Lastly, we apply our model to discover hierarchies in two real-world datasets: we show that an embedding in hyperbolic space can reveal important aspects of a company’s organizational structure as well as reveal historical relationships between language families. 

## **1. Introduction** 

Hierarchical structures are ubiquitous in knowledge representation and reasoning. For example, starting with Linnaeus, taxonomies have long been used in biology to categorize and understand the relationships between species (Mayr, 1968). In social science, hierarchies are used to understand interactions in humans and animals or to analyze organizational structures such as companies and governments (Dodds et al., 2003). In comparative linguists, evolutionary trees are used to describe the origin of languages (Campbell, 2013), while ontologies are used to provide rich categorizations of entities in semantic networks (Antoniou & Van Harmelen, 2004). Hierarchies are also known to provide important information for learning and classification (Silla & Freitas, 2011). In cognitive development, the results of Inhelder & Piaget (1964) suggest that the classification structure in children’s thinking is hierarchical in nature. 

Hierarchies can therefore provide important insights into 

1Facebook AI Research, New York, NY, USA. Correspondence to: Maximilian Nickel <maxn@fb.com>. 

_Proceedings of the 35[th] International Conference on Machine Learning_ , Stockholm, Sweden, PMLR 80, 2018. Copyright 2018 by the author(s). 

systems of concepts. However, explicit information about such hierarchical relationships is unavailable for many domains. In this paper, we therefore consider the problem of discovering concept hierarchies from unstructured observations, specifically in the following setting: 

1. We focus on discovering pairwise hierarchical relations between concepts, where all superior and subordinate concepts are observed. 

2. We aim to infer concept hierarchies only from pairwise similarity measurements, which are relatively easy and cheap to obtain in many domains. 

Examples of hierarchy discovery that adhere to this setting include the creation of taxonomies from similarity judgments (e.g., genetic similarity of species or cognate similarity of languages) and the recovery of organizational hierarchies and dominance relations from social interactions. 

To infer hierarchies from similarity judgments, we propose to model such relationships as a combination of two separate aspects: relatedness and generality. Concept A is a parent (a superior) to concept B if both concepts are related and A is more general than B. By separating these aspects, we can then discover concept hierarchies via hyperbolic embeddings. In particular, we build upon ideas of Poincaré embeddings (Nickel & Kiela, 2017) to learn continuous representations of hierarchies. Due to its geometric properties, hyperbolic space can be thought of as continuous analogue to discrete trees. By embeddings concepts in such a way that their similarity order is preserved, we can then identify (soft) hierarchical relationships from the embedding: relatedness is captured via the distance in the embedding space, while generality is captured via the norm of the embeddings. 

To learn high-quality embeddings, we propose a new optimization approach based on the Lorentz model of hyperbolic space. The Lorentz model allows for an efficient closedform computation of the geodesics on the manifold. This facilitates the development of an efficient optimizer that directly follows these geodesics, rather than doing a first-order approximation as in (Nickel & Kiela, 2017). It allows us also to avoid numerical instabilities that arise from the Poincaré distance. As we will show experimentally, this optimization 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

method leads to a substantially improved embedding quality, especially in low dimensions. Simultaneously, we retain the attractive properties of hyperbolic embeddings, i.e., learning continuous representations of hierarchies via gradient-based optimization while scaling to large datasets. 

The reminder of this paper is organized as follows. In Section 2, we discuss related work regarding hyperbolic and ordered embeddings. In Section 2, we introduce our model and algorithm to compute the embeddings. In Section 4 we evaluate the efficiency of our approach on large taxonomies. Furthermore, we evaluate the ability of our model to discover meaningful hierarchies on real-world datasets. 

## **2. Related Work** 

Hyperbolic geometry has recently received attention in machine learning and network science due to its attractive properties for modeling data with latent hierarchies. Krioukov et al. (2010) showed that typical properties of complex networks (e.g., heterogeneous degree distributions and strong clustering) can be explained by assuming an underlying hyperbolic geometry and, moreover, developed a framework to model networks based on these properties. Furthermore, Kleinberg (2007) and Boguñá et al. (2010) proposed hyperbolic embeddings for greedy shortest-path routing in communication networks. Asta & Shalizi (2015) used hyperbolic embeddings of graphs to compare the global structure of networks. Sun et al. (2015) proposed to learn representations of non-metric data in pseudo-Riemannian space-time, which is closely related to hyperbolic space. 

Most similar to our work are the recently proposed Poincaré embeddings (Nickel & Kiela, 2017), which learn hierarchical representations of symbolic data by embedding them into an _n_ -dimensional Poincaré ball. The main focus of that work was to model the link structure of symbolic data efficiently, i.e., to find low-dimensional embeddings via exploiting the hierarchical structure of hyperbolic space. Here, we build upon this idea and extend it in various ways. First, we propose a new model to compute hyperbolic embeddings in the Lorentz model of hyperbolic geometry. This allows us to develop an efficient Riemannian optimization method that scales well to large datasets and provides better embeddings, especially in low dimensions. Second, we consider inferring hierarchies from real-valued similarity scores, which generalize binary adjacency matrices as considered by Nickel & Kiela (2017). Third, in addition to preserving similarity (e.g., local link structure), we also focus on recovering the correct hierarchical relationships from the embedding. 

Simultaneously to the present work, De Sa et al. (2018) analyzed the representation trade-offs for hyperbolic embeddings and proposed a new combinatorial embedding approach as well as a new approach to Multi-Dimensional 

Scaling (MDS) in hyperbolic space. Furthermore, Ganea et al. (2018) extended Poincaré embeddings using geodesically convex cones to model asymmetric relations. 

Another related method is Order Embeddings (Vendrov et al., 2015), which was proposed to learn visual-semantic hierarchies over words, sentences, and images from ordered input pairs. In contrast, we are concerned with learning hierarchical embeddings from less supervision: namely, from _unordered_ (symmetric) input pairs that provide no direct information about the partial ordering in the hierarchy. 

Further work on embedding order-structures include Stochastic Triplet Embeddings (Van Der Maaten & Weinberger, 2012), Generalized Non-Metric MDS (Agarwal et al., 2007), and Crowd Kernels (Tamuz et al., 2011). In the context of word embeddings, Vilnis & McCallum (2015) proposed Gaussian Embeddings to learn improved representations. By mapping words to densities, this model is capable of capturing uncertainty, assymmetry, and (hierarchical) entailment relations. 

To discover structural forms (e.g., trees, grids, chains) from data, Kemp & Tenenbaum (2008) proposed a model for making probabilistic inferences over a space of graph grammars. Recently, Lake et al. (2018) proposed an alternative approach to this work based on structural sparsity. Additionally, hierarchical clustering has a long history in machine learning and data mining (Duda et al., 1973). Bottom-up agglomerative clustering assigns each data point to its own cluster and then iteratively merges the two closest points according to a given distance measure (e.g., single link, average link, max link). As such, hierarchical clustering provides a hierarchical partition of the input space. In contrast, we are concerned with discovering direct hierarchical relationships between the input data points. 

## **3. Methods** 

In the following, we describe our approach for learning continuous hierarchies from unstructured observations. 

## **3.1. Hyperbolic Geometry & Poincaré Embeddings** 

Hyperbolic space is the unique, complete, simply connected Riemannian manifold with constant negative sectional curvature. There exist multiple equivalent[1] models for hyperbolic space and one can choose the model whichever is best suited for a given task. Nickel & Kiela (2017) based their approach for learning hyperbolic embeddings on the Poincaré ball model, due to its conformality and convenient parameterization. The Poincaré ball model is the Riemannian manifold _P[n]_ = ( _B[n] , gp_ ), where _B[n]_ = _{_ _**x** ∈_ R _[n]_ : _∥_ _**x** ∥ <_ 1 _}_ is the 

1Meaning that there exist transformations between the different models that preserve all geometric properties including isometry. 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

**==> picture [407 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
p 5<br>p 3<br>p 4<br>p 1<br>p 2<br>(a) Geodesics in the Poincaré disk. (b) Lorentz model of hyperbolic geometry.<br>**----- End of picture text -----**<br>


Figure 1: a) Geodesics in the Poincaré disk model of hyperbolic space. Due to the negative curvature of the space, geodesics between points are arcs that are perpendicular to the boundary of the disk. For curved arcs, midpoints are closer to the origin of the disk (p1) than the associated points, e.g. (p3, p5). b) Points (p,q) lie on the surface of the upper sheet of a two-sheeted hyperboloid. Points (u, v) are the mapping of (p, q) onto the Poincaré disk using Equation (11). 

_open n_ -dimensional unit ball and where 

**==> picture [228 x 73] intentionally omitted <==**

The distance function on _P_ is then defined as 

It can be seen from Equation (1), that the distance within the Poincaré ball changes smoothly with respect to the norm of _**x**_ and _**y**_ . This locality property of the distance is key for learning continuous embeddings of hierarchies. For instance, by placing the root node of a tree at the origin of _B[n]_ , it would have relatively small distance to all other nodes, as its norm is zero. On the other hand, leaf nodes can be placed close to the boundary of the ball, as the distance between points grows quickly with a norm close to one. 

denote the _Lorentzian scalar product_ . The Lorentz model of _n_ -dimensional hyperbolic space is then defined as the Riemannian manifold _L[n]_ = ( _H[n] , gℓ_ ), where 

**==> picture [207 x 13] intentionally omitted <==**

denotes the upper sheet of a two-sheeted _n_ -dimensional hyperboloid and where 

**==> picture [179 x 81] intentionally omitted <==**

The associated distance function on _L_ is then given as 

**==> picture [228 x 12] intentionally omitted <==**

**==> picture [177 x 13] intentionally omitted <==**

## **3.2. Riemannian Optimization in the Lorentz Model** 

In the following, we propose a new method to compute hyperbolic embeddings based on the Lorentz model of hyperbolic geometry. The main advantage of this parameterization is that it allows us to perform Riemannian optimization very efficiently. An additional advantage is that its distance function (see Equation (5)) avoids numerical instabilities that arise from the fraction in the Poincaré distance. 

## 3.2.1. THE LORENTZ MODEL OF HYPERBOLIC SPACE 

In the following, let _**x**_ , _**y** ∈_ R _[n]_[+1] and let 

**==> picture [179 x 29] intentionally omitted <==**

## 3.2.2. RIEMANNIAN OPTIMIZATION 

To derive the Riemannian SGD (RSGD) algorithm for the Lorentz model, we will first review the necessary concepts of Riemannian optimization. A Riemannian manifold ( _M, g_ ) is a real, smooth manifold _M_ equipped with a Riemannian metric _g_ . Furthermore, for each _**x** ∈M_ , let _T_ _**x** M_ denote the associated _tangent space_ . The metric _g_ induces then a inner product _⟨·, ·⟩_ _**x**_ : _T_ _**x** M × T_ _**x** M →_ R. _Geodesics γ_ : [0 _,_ 1] _→M_ are the generalizations of straight lines to Riemannian manifolds, i.e., constant speed curves that are locally distance minimizing. The _exponential map_ exp _**x**_ : _T_ _**x** M →M_ maps a tangent vector _**v** ∈T_ _**x** M_ onto _M_ such that exp _**x**_ ( _**v**_ ) = _**y**_ , _γ_ (0) = _**x**_ , _γ_ (1) = _**y**_ 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

and _γ_ ˙ (0) = _∂t[∂][γ]_[(0) =] _**[ v]**_[.][For a] _[ complete manifold][ M]_[, the] exponential map is defined for all points _**x** ∈M_ . 

Furthermore, let _f_ : _M →_ R be a smooth real-valued function over parameters _θ ∈M_ . In Riemannian optimization, we are then interested in solving problems of the form 

**Algorithm 1** Riemannian Stochastic Gradient Descent 

|**Input**Learning|rate_η_, number of epochs_T_.|
|---|---|
|for_t_= 1_, . . . , T_<br>**_h_**_t_<br>grad_f_(_θt_)<br>_θt_+1|_←g−_1<br>_θt ∇f_(_θt_)<br>_←_proj_θt_(**_h_**_t_)<br>_←_exp_θt_(_−η_grad_f_(_θt_))|



**==> picture [139 x 15] intentionally omitted <==**

## 3.2.3. EQUIVALENCE OF MODELS 

Following Bonnabel (2013), we minimize Equation (7) using Riemannian SGD. In RSGD, updates to the parameters _θ_ are computed via 

**==> picture [178 x 11] intentionally omitted <==**

where grad _f_ ( _θt_ ) _∈TθM_ denotes the _Riemannian gradient_ and _η_ denotes the learning rate. 

For the Lorentz model, the tangent space is defined as follows: For a point _**x** ∈L[n]_ , the tangent space _T_ _**x** L[n]_ consists of all vectors orthogonal to _**x**_ , where orthogonality is defined with respect to the Lorentzian scalar product. Hence, 

**==> picture [113 x 11] intentionally omitted <==**

Furthermore, let _**v** ∈ T_ _**x** L[n]_ . The exponential map exp _**x**_ : _T_ _**x** L[n] →L[n]_ is then defined as 

**==> picture [212 x 21] intentionally omitted <==**

where _∥_ _**v** ∥L_ = ~~�~~ _⟨_ _**v** ,_ _**v** ⟩L_ denotes the norm of _**v**_ in _T_ _**x** L[n]_ . 

To compute parameter updates as in Equation (7), we need additionally the Riemannian gradient of _f_ at _θ_ . For this purpose, we first compute the direction of steepest descent from the Euclidean gradient _∇f_ ( _θ_ ) via 

**==> picture [149 x 13] intentionally omitted <==**

Since _gℓ_ is an involutory matrix (i.e., _gℓ[−]_[1] = _gℓ_ ), the inverse in Equation (10) is trivial to compute. To derive the Riemannian gradient from _**h** ∈_ R _[n]_[+1] , we then use the orthogonal projection proj _θ_ : R _[n]_[+1] _→TθL[n]_ from the ambient Euclidean space onto the tangent space of the current parameter. This projection is computed as 

**==> picture [181 x 25] intentionally omitted <==**

since _∀x ∈H[n]_ : _⟨_ _**x** ,_ _**x** ⟩L_ = _−_ 1 (Robbin & Salamon, 2017). Using Equation (9) and section 3.2.2, we can then estimate the parameters _θ_ using RSGD as in Algorithm 1. We initialize the embeddings close to the origin of _H[n]_ by sampling from the uniform distribution _U_ ( _−_ 0 _._ 001 _,_ 0 _._ 001) and by setting _x_ 0 according to Equation (6). 

The Lorentz and Poincaré disk model both have specific strengths: the Poincare disk provides a very intuitive method for visualizing and interpreting hyperbolic embeddings. The Lorentz model on the other hand is well-suited for Riemannian optimization. Due to the equivalence of both models, we can exploit their individual strengths simultaneously: points in the Lorentz model can be mapped into the Poincaré ball via the diffeomorphism _p_ : _H[n] →P[n]_ , where 

**==> picture [187 x 24] intentionally omitted <==**

Furthermore, points in _P[n]_ can be mapped into _H[n]_ via 

**==> picture [186 x 26] intentionally omitted <==**

We will therefore learn the embeddings via Algorithm 1 in the Lorentz model and visualize the embeddings by mapping them into the Poincaré disk using Equation (11). See also Figure 1b for an illustration of Lorentz model and its connections to the Poincaré disk. 

## **3.3. Inferring Concept Hierarchies from Similarity** 

Nickel & Kiela (2017) embedded unweighted undirected graphs in hyperbolic space. In the following, we extend this approach to a more general setting, i.e., inferring continuous hierarchies from _pairwise similarity measurements_ . 

Let _C_ = _{ci}[m] i_ =1[be][a][set][of][concepts][and] _[X][∈]_[R] _[m][×][m]_ be a dataset of pairwise similarity scores between these concepts. We also assume that the concepts can be organized according to an unobserved hierarchy ( _C, ⪯_ ), where _ci ⪯ cj_ defines a _partial order_ over the elements of _C_ . Since partial order is a reflexive, anti-symmetric, and transitive binary relation, it is well suited to define hierarchical relations over _C_ . If _ci ⪯ cj_ or _cj ⪯ ci_ , then the concepts _ci_ , _cj_ are _comparable_ (e.g., located in the same subtree). Otherwise they are incomparable (e.g., located in different subtrees). For concepts _ci ≺ cj_ , we will refer to _ci_ as the superior and to _cj_ as the subordinate node. 

Given this setting, our goal is then to recover the partial order ( _C, ⪯_ ) from _X_ . For this purpose, we separate the semantics of the partial order relation into two distinct aspects: 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

First, whether two concepts _ci, cj_ are comparable (denoted by _ci ∼ cj_ ) and, second, whether concept _ci_ is more _general_ than _cj_ (denoted by _cj_ ⊏ _ci_ ). Combining both aspects provides us with the usual interpretation of partial order. 

By explicitly distinguishing between the aspects of comparability and generality, we can then make the following structural assumptions on _X_ to infer hierarchies from pairwise similarities: 1) Comparable (and related) concepts are more similar to each other than incomparable concepts (i.e., _Xij ≥ Xik_ if _ci ⪯ cj ∧ci ̸⪯ ck_ ); and 2) We assume that general concepts are similar to more concepts than less general ones. Both are mild assumptions given that the similarity scores _X_ describe concepts that are organized in a latent hierarchy. For instance, 1) simply follows from the assumption that concepts in the same subtree of the ground-truth hierarchy are more similar to each other than to concepts in different subtrees. This is also used in methods that use pathlengths in taxonomies to measure the _semantic similarity_ of concepts (e.g., see Resnik et al., 1999). 

It follows from assumption 1) the we want to preserve the similarity orderings in the embedding space in order to predict comparability. In particular, let _**u** i_ denote the embedding of _ci_ and let _N_ ( _i, j_ ) = _{k_ : _Xik < Xij} ∪{j}_ denote the set of concepts that are _less_ similar to _ci_ then _cj_ (including _cj_ ). Based only on pairwise similarities in _X_ , it is difficult to make global decisions about the likelihood that _ci ∼ cj_ is true. However, it follows from assumption 1) that we can make _local ranking_ decisions, i.e., we can infer that _ci ∼ cj_ is the most likely among all _ck ∈N_ ( _i, j_ ). For this purpose, let 

**==> picture [113 x 19] intentionally omitted <==**

be the nearest neighbor of _ci_ in the set _N_ ( _i, j_ ). We then learn embeddings Θ = _{_ _**u** }[m] i_ =1[by optimizing] 

**==> picture [183 x 23] intentionally omitted <==**

where 

**==> picture [182 x 28] intentionally omitted <==**

For computational efficiency, we follow (Jean et al., 2015) and randomly subsample _N_ ( _i, j_ ) on large datasets. 

Equation (12) is a ranking loss that aims to preserve the neighborhood structures in _X_ . For each pair of concepts ( _i, j_ ), this loss induces embeddings where ( _i, j_ ) is closer in the embedding space than pairs ( _i, k_ ) that are less similar. Since we compute the embedding in a metric space, we also retain transitive relations approximately. We can therefore identify the comparability of concepts _ci ∼ cj_ by their distance _d_ ( _**u** i,_ _**u** j_ ) in the embedding. 

Table 1: Taxonomy Statistics. The number of edges refers to the full transitive closure of the respective taxonomy. 

||**Taxonomy**|**Nodes**|**Edges**|**Depth**|
|---|---|---|---|---|
||WORDNETNouns<br>WORDNETVerbs<br>EUROVOC(en)<br>ACM|82,115<br>13,542<br>7,084<br>2,299|769,130<br>35,079<br>10,547<br>6,526|19<br>12<br>5<br>5|
||MESH|28,470|191,849|15|



Moreover, by optimizing Equation (12) in hyperbolic space, we are also able to infer the generality of concepts from their embeddings. According to assumption 2), we can can assume that general objects will be close to many different concepts. Since Equation (12) optimizes the local similarity ranking for all concepts, we can also assume that this ordering is preserved. We can see from Equation (1) that points with a small distance to many different points are located close to the center. We can therefore identify the generality of a concept _ci_ simply via the norm of its embedding _∥_ _**u** i∥_ . 

We have now cast the problem of hierarchy discovery as a simple embedding problem whose objective is to preserve local similarity orderings 

## **4. Evaluation** 

## **4.1. Embedding Taxonomies** 

In the following experiments, we evaluate the performance of the Lorentz model for embedding large taxonomies. For this purpose, we compore its embedding quality to Poincaré embeddings (Nickel & Kiela, 2017) on the following realworld taxonomies 

- **WordNet** _⃝_ **R** (Miller & Fellbaum, 1998) is a large lexical database which, amongst other relations, provides hypernymy (is-a) relations. In our experiments, we embedded the noun and verb hierarchy of WordNet. 

- **EuroVoc** is a mulitlingual thesaurus maintained by the European Union. It contains keywords organized in 21 domains and 127 sub-domains. In our experiments, we used the English section of EuroVoc.[2] 

- **ACM** The ACM computing classification system is a hierarchical ontology which is used by various ACM journals to organize subjects by area. 

- **MeSH** Medical Subject Headings (MeSH; (Rogers, 1963)) is a medical thesaurus which is created, maintained and provided by the U.S. National Library of Medicine. In our experiments we used the 2018 MeSH hierarchy. 

   - 2Available at http://eurovoc.europa.eu 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

Table 2: Evaluation of Taxonomy Embeddings. MR = Mean Rank, MAP = Mean Average Precision _ρ_ = Spearman rank-order correlation. ∆% indicates the relative improvement of optimization in the Lorentz model. 

||**WORDNET Nouns**<br>2<br>5<br>10|**WORDNET Verbs**<br>2<br>5<br>10|**EUROVOC**|**ACM**|**MESH**<br>2<br>5<br>10|
|---|---|---|---|---|---|
||||2<br>5<br>10|2<br>5<br>10||
|**MR**<br>Poincaré<br>Lorentz<br>∆%|90.7<br>4.9<br>4.02<br>22.8<br>3.18<br>2.95<br>74.8<br>35.1<br>36.2|10.71<br>1.39<br>1.35<br>3.64<br>1.26<br>1.23<br>66.0<br>9.6<br>8.9|2.83<br>1.25<br>1.23<br>1.63<br>1.24<br>1.17<br>42.4<br>6.1<br>3.4|4.14<br>1.8<br>1.71<br>3.05<br>1.67<br>1.63<br>26.3<br>7.2<br>4.8|61.11<br>14.05<br>12.8<br>38.99<br>14.13<br>12.42<br>36.2<br>-0.5<br>2.9|
|**MAP**<br>Poincaré<br>Lorentz<br>∆%|11.8<br>82.8<br>86.5<br>30.5<br>92.3<br>92.8<br>61.3<br>10.3<br>6.8|36.5<br>91.0<br>91.2<br>57.9<br>93.5<br>93.3<br>58.6<br>2.7<br>2.3|64.3<br>94.0<br>94.4<br>87.1<br>95.8<br>96.5<br>35.6<br>1.6<br>2.0|69.3<br>94.1<br>94.8<br>82.9<br>96.6<br>97.0<br>19.6<br>2.7<br>2.3|19.5<br>76.3<br>79.4<br>34.8<br>77.7<br>79.9<br>43.9<br>1.8<br>0.6|
|**_ρ_**<br>Poincaré<br>Lorentz|13.8<br>57.2<br>58.5<br>41.0<br>58.9<br>59.5|11.0<br>54.1<br>55.1<br>47.9<br>55.5<br>56.6|37.5<br>57.5<br>61.4<br>54.5<br>61.7<br>67.5|59.8<br>63.5<br>62.9<br>65.9<br>65.9<br>65.9|42.2<br>69.9<br>74.9<br>64.5<br>71.4<br>76.3|



## Statistics for all taxonomies are provided in Table 1. 

In our evaluation, we closely follow the setting of Nickel & Kiela (2017): First, we embed the undirected transitive closure of these taxonomies, such that the hierarchical structure is not directly visible from the observed edges but has to be inferred. To measure the quality of the embedding, we compute for each observed edge ( _u, v_ ) the corresponding distance _d_ ( _**u** ,_ _**v**_ ) in the embedding and rank it among the distances of all _unobserved_ edges for _u_ , i.e., among _{d_ ( _**u** ,_ _**v**[′]_ ) : ( _u, v[′]_ ) _̸∈D}_ . We then report the mean rank (MR) and mean average precision (MAP) of this ranking. 

In addition, we also evaluate how well the norm of the embeddings (i.e., our indicator for generality), correlates with the ground-truth ranks in the embedded taxonomy. Since different subtrees can have very different depths, we normalize the rank of each concept by the depth of its subtree and measure the Spearman rank-order correlation _ρ_ of the normalized rank with the norm of the embedding. We compute the normalized rank in the following way: Let sp( _c_ ) denote the shortest path to the root node from _c_ , and let lp( _c_ ) denote the longest path from _c_ to any of its children.[3] The normalized rank of _c_ is the given as 

**==> picture [102 x 24] intentionally omitted <==**

To learn the embeddings in the Lorentz model, we employ the Riemannian optimization method as described in Section 3.2. For Poincaré embeddings, we use the official opensource implementation.[4] Both methods were cross-validated over identical sets of hyperparameters. 

Table 2 shows the results of our evaluation. It can be seen that both methods are very efficient in embedding these 

> 3Since all taxonomies in our experiments are DAGs, it is possible to compute the longest path in the graph 

> 4Source code available at https://github.com/ facebookresearch/poincare-embeddings 

large taxonomies. However, the Lorentz model shows consistently higher-quality embeddings and especially so in low dimensions. The relative improvement of the twodimensional Lorentz embeddings over the Poincaré embedding amounts to 74.8% on the WORDNET noun hierarchy and 42.4% on EUROVOC. Similar improvements can be observed on all taxonomies. Furthermore, on the most complex taxonomy (WORDNET nouns), the 10-dimensional Lorentz embeddings already out-performs the best reported numbers reported in (Nickel & Kiela, 2017) (which went up to 200 dimensions). This suggests that the full Riemannian optimization approach can be very helpful for obtaining good embeddings. This is especially the case in low dimensions where it is harder for the optimization procedure to escape from local minima. 

## **4.2. Enron Email Corpus** 

In addition to the taxonomies in Section 4.1, we are interested in discovering hierarchies from real-world graphs that have not been generated from a clean DAG. For this purpose, we embed the communication graph of the Enron email corpus (Priebe et al., 2006) which consists of 125,409 emails that have been exchanged between 184 email addresses and 150 unique users.[5] From this data, we construct a graph where weighted edges represent the total number of emails that have been exchanged between two users. The dataset includes also the organizational roles for 130 users, based largely on the information collected by Shetty & Adibi (2005). 

Figure 2a shows the two-dimensional embedding of this graph. It can be seen that the embedding captures important properties of the organizational structure. First, the nodes are approximately arranged according to the organizational hierarchy of the company. Executive roles such as 

> 5This dataset has been created by Priebe et al. (2006) from the full Enron email corpus which has been released into public domain by the Federal Energy Regulatory Commission (FERC). 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

**==> picture [374 x 215] intentionally omitted <==**

**----- Start of picture text -----**<br>
PnJ ‘ d % “. : d Louise K. (P) [Richard°° sa. (vp) COO / ViceCEOPresident/ President/ Director LevelLevel 54<br>In-House Lawyer Level 3<br>Manager / Trader Level 2<br>(T)<br>Specialist / Analyst Level 1<br>F. (M) 5 Y Serer?) Jes O- (U | Employee Level 0<br>(T) 8 Richard Sh. (VP)<br>Dana D. (VP) v 1d » (b) Org. hierarchy<br>{ Harpreet A. (VP)<br>° 0 . 6 0 . 589 0 . 536 0 . 555<br>. Joe Q. (T) d Stanley H. (P) 0 . 5 B 0 . 463<br>re —<br>(0 X shelley c. (vp) ) 0 . 4 | | | i<br>0 . 319<br>Thomas M. (VP) 0 . 3 ‘11K fn<br>(a) Embedding of the Enron communication graph (c) Rank-order correlation<br>Hyperbolic DegreeCloseness EigenBetweeness<br> ρ<br>Spearman<br>**----- End of picture text -----**<br>


Figure 2: Embedding of the Enron email corpus. Abbreviations in parentheses indicate organizational role: CEO = Chief Executive Officer, COO = Chief Operating Officer, P = President, VP = Vice President, D = Director, M = Manager, T = Trader. Blue lines indicate edges in the graph. Node size indicates node degree. 

CEOs, COOs, and (vice) presidents are embedded close to the origin, while other employees (e.g., traders and analysts) are located closer to the boundary. Figure 2c shows the Spearman correlation of the norms of the embedding with the organizational rank. It can be seen that the norm correlates well with the ground-truth ranking and is on-par or better than commonly-used centrality measures on graphs. We also observe that the embedding provides a meaningful clustering of users. For instance, the lower left of the disk shows a cluster of traders. Above that cluster (i.e., closer to the origin), are managers (e.g., John F.) and vice presidents (e.g., Fletcher S., Kevin P.) who have been associated with the trading arm of Enron. This illustrates that, in addition to the notion of rank in a hierarchy, the embedding provides also insight into the similarity of nodes within the hierarchy. 

## **4.3. Historical Linguistics Data** 

The field of historical linguistics is concerned with the history of languages, their relations, and their origins. An important concept to determine the relations between lan- 

guages are so-called _cognates_ , i.e., words that are shared across different languages (but not borrowed) and which indicate common ancestry in the history of languages. To be classified as cognate, words must have similar meaning and systematic sound correspondences. Languages are assumed to be related if they share a large number of cognates. 

The goal of our experiments was to discover the historical relationships between languages (which are assumed to follow a hierarchical tree structure) by embedding cognate similarity data. For this purpose, we used the lexical cognate data provided by Bouckaert et al. (2012), which consists of 103 Indo-European languages and 6280 cognate sets in total. Since the number of cognate sets grew over time, not all languages are annotated with all possible sets. For this reason, we computed the cognate similarity between two languages in the following way. Let _c_ ( _ℓ_ 1 _, ℓ_ 2) denote the number of common cognates in languages _ℓ_ 1 _, ℓ_ 2. Furthermore, let _a_ ( _ℓ_ 1) denote the number of cognate annotations for _ℓ_ 1. We then compute the cognate similarity of _ℓ_ 1 _, ℓ_ 2 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

Figure 3: Embedding of the IELex lexical cognate data. 

simply as 

**==> picture [139 x 25] intentionally omitted <==**

Figure 3 shows a two-dimensional embedding of these cognate similarity scores. It can be seen that the embedding allows us to discover a meaningful hierarchy that corresponds well with the assumed origin of languages. First, the embedding shows clear clusters of high-level language families such as Celtic, Romance, Germanic, Balto-Slavic, Hellenic, Indic, and Iranian. Moreover, each of these cluster displays meaningful internal hierarchies such as (Gothic _→_ Old High German _→_ German), (Old Prussian _→_ Old Church Slavonic _→_ Bulgarian), (Latin _→_ Italian), or (Ancient Greek _→_ Greek). Closer to the center of the disc, we also find a number of ancient languages. For instance, Oscan and Umbrian are two extinct sister languages of Latin and located above the Romance cluster, Similarly, Avestan and Vedic-Sanskrit are two ancient languages that separated early in the pre-historic era before 1800 BCE (Baldi, 1983). After separation, Avestan developed in ancient Persia while Vedic-Sanskrit developed independently in ancient India. In the embedding, both languages are close to the center and to each other. Furthermore, Avestan is close to the Iranian cluster while Vedic-Sanskrit is close to the Indic cluster. 

## **5. Conclusion** 

We introduced a new method for learning continuous concept hierarchies from unstructured observations. We exploited the properties of hyperbolic geometry in such a way that we can discover hierarchies from pairwise similarity scores – under the assumption that concepts in the same subtree of the ground-truth hierarchy are more similar to each other than to concepts in different subtrees. To learn the embeddings, we developed an efficient Riemannian optimization approach based on the Lorentz model of hyperbolic space. Due to the more principled optimization approach, we were able to substantially improve the quality of the embeddings compared to the method proposed by Nickel & Kiela (2017) – especially in low dimensions. We further showed on two real-world datasets, that our method can discover meaningful hierarchies from nothing but pairwise similarity information. 

## **Acknowledgments** 

The authors thank Joan Bruna, Martín Arjovsky, Eryk Kopczy´nski, and Laurens van der Maaten for helpful discussions and suggestions. 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

## **References** 

- Agarwal, S., Wills, J., Cayton, L., Lanckriet, G., Kriegman, D., and Belongie, S. Generalized non-metric multidimensional scaling. In _Artificial Intelligence and Statistics_ , pp. 11–18, 2007. 

- Antoniou, G. and Van Harmelen, F. Web ontology language: Owl. In _Handbook on ontologies_ , pp. 67–92. Springer, 2004. 

- Asta, D. M. and Shalizi, C. R. Geometric network comparisons. In Meila, M. and Heskes, T. (eds.), _Proceedings of the Thirty-First Conference on Uncertainty in Artificial Intelligence, UAI_ , pp. 102–110, 2015. 

- Baldi, P. _An introduction to the Indo-European languages_ . SIU Press, 1983. 

- Boguñá, M., Papadopoulos, F., and Krioukov, D. Sustaining the internet with hyperbolic mapping. _Nature communications_ , 1:62, 2010. 

- Bonnabel, S. Stochastic gradient descent on Riemannian manifolds. _IEEE Trans. Automat. Contr._ , 58(9):2217– 2229, 2013. 

- Bouckaert, R., Lemey, P., Dunn, M., Greenhill, S. J., Alekseyenko, A. V., Drummond, A. J., Gray, R. D., Suchard, M. A., and Atkinson, Q. D. Mapping the origins and expansion of the indo-european language family. _Science_ , 337(6097):957–960, 2012. 

- Campbell, L. _Historical linguistics_ . Edinburgh University Press, 2013. 

- De Sa, C., Gu, A., Ré, C., and Sala, F. Representation tradeoffs for hyperbolic embeddings. _arXiv preprint arXiv:1804.03329_ , 2018. 

- Dodds, P. S., Watts, D. J., and Sabel, C. F. Information exchange and the robustness of organizational networks. _Proceedings of the National Academy of Sciences_ , 100 (21):12516–12521, 2003. 

- Duda, R. O., Hart, P. E., Stork, D. G., et al. _Pattern classification_ , volume 2. Wiley New York, 1973. 

- Ganea, O.-E., Bécigneul, G., and Hofmann, T. Hyperbolic entailment cones for learning hierarchical embeddings. _arXiv preprint arXiv:1804.01882_ , 2018. 

- Inhelder, B. and Piaget, J. _The growth of logic in the child_ . Routledge & Paul, 1964. 

- Jean, S., Cho, K., Memisevic, R., and Bengio, Y. On using very large target vocabulary for neural machine translation. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the_ 

_7th International Joint Conference on Natural Language Processing_ , volume 1, pp. 1–10, 2015. 

- Kemp, C. and Tenenbaum, J. B. The discovery of structural form. _Proceedings of the National Academy of Sciences_ , 105(31):10687–10692, 2008. 

- Kleinberg, R. Geographic routing using hyperbolic space. In _INFOCOM 2007. 26th IEEE International Conference on Computer Communications. IEEE_ , pp. 1902–1909. IEEE, 2007. 

- Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., and Boguná, M. Hyperbolic geometry of complex networks. _Physical Review E_ , 82(3):036106, 2010. 

- Lake, B. M., Lawrence, N. D., and Tenenbaum, J. B. The emergence of organizing structure in conceptual representation. _Cognitive Science_ , 2018. 

- Mayr, E. The role of systematics in biology. _Science_ , 159 (3815):595–599, 1968. 

- Miller, G. and Fellbaum, C. Wordnet: An electronic lexical database, 1998. 

- Nickel, M. and Kiela, D. Poincaré embeddings for learning hierarchical representations. pp. 6338–6347, 2017. 

- Priebe, C. E., Conroy, J. M., Marchette, D. J., and Park, Y. Enron data set, 2006. URL http://cis.jhu.edu/ ~parky/Enron/enron.html. 

- Resnik, P. et al. Semantic similarity in a taxonomy: An information-based measure and its application to problems of ambiguity in natural language. _J. Artif. Intell. Res.(JAIR)_ , 11:95–130, 1999. 

- Robbin, J. W. and Salamon, D. A. Introduction to differential geometry. _ETH, Lecture Notes, preliminary version, October_ , 2017. 

- Rogers, F. Medical subject headings. _Bulletin of the Medical Library Association_ , 51:114–116, 1963. 

- Shetty, J. and Adibi, J. Enron employee status report, 2005. URL http://www.isi.edu/~adibi/ Enron/EnronEmployeeStatus.xls. 

- Silla, C. N. and Freitas, A. A. A survey of hierarchical classification across different application domains. _Data Mining and Knowledge Discovery_ , 22(1-2):31–72, 2011. 

- Sun, K., Wang, J., Kalousis, A., and Marchand-Maillet, S. Space-time local embeddings. In _Advances in Neural Information Processing Systems 28_ , pp. 100–108, 2015. 

- Tamuz, O., Liu, C., Belongie, S., Shamir, O., and Kalai, A. T. Adaptively learning the crowd kernel. In _Proceedings of the 28th International Conference on International Conference on Machine Learning_ , pp. 673–680, 2011. 

**Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry** 

- Van Der Maaten, L. and Weinberger, K. Stochastic triplet embedding. In _Machine Learning for Signal Processing (MLSP), 2012 IEEE International Workshop on_ , pp. 1–6. IEEE, 2012. 

- Vendrov, I., Kiros, R., Fidler, S., and Urtasun, R. Orderembeddings of images and language. _arXiv preprint arXiv:1511.06361_ , 2015. 

- Vilnis, L. and McCallum, A. Word representations via gaussian embedding. In _International Conference on Learning Representations (ICLR)_ , 2015. 

