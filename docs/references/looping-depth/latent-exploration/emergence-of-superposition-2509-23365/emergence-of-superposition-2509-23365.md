# Emergence of Superposition: reading and local cache

Read 2026-10-18, in full: main body, Appendix B (all proofs for Section 3), Appendix C
(prediction-stage proofs), Appendix D, and the experiment appendices E and F. Read for the
coarse-level particle design on the Parcae spiral testbed (Lean claims C1 to C6 in
[`lab/theory/tul_exploration/TulExploration/Particles.lean`](../../../../../lab/theory/tul_exploration/TulExploration/Particles.lean)).

## Citation (verified)

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, Yuandong Tian.
*Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous
Thought*. [arXiv 2509.23365v3](https://arxiv.org/abs/2509.23365), first posted
2025-09-27, v3 dated 2026-03-01. Affiliations: UC Berkeley, UCSD, Nvidia, Meta AI.

Verified 2026-10-18 against the arXiv export API
(`https://export.arxiv.org/api/query?id_list=2509.23365`): title and all six authors match,
and the API comment reads "32 pages, 9 figures, accepted by ICLR 2026". Every PDF page
carries the header "Published as a conference paper at ICLR 2026". Cite it as ICLR 2026.

It is the companion of [Reasoning by Superposition](../reasoning-by-superposition/reasoning-by-superposition.md)
(same six authors). That paper builds the weights by hand. This one asks whether gradient
flow finds them.

## Local cache

- [PDF in this folder](emergence-of-superposition-2509-23365.pdf)
- [Private PDF](../../../../../ignore/papers/2509.23365v3-emergence-of-superposition.pdf)
- [Private extracted text](../../../../../ignore/papers/2509.23365v3-emergence-of-superposition.txt)
  (`pdftotext -layout`)
- 1,554,632 bytes, 32 pages, 2,527 text lines.
- SHA256: `0618e8beba581d69a8b8483fc36d8930855935373137d66090116eebb07bb01e`.

## What it actually does

The task is the same graph reachability task as the companion paper. The model is a
SIMPLIFIED two-layer transformer. The paper assumes layer 1 already copies each edge's
source and target onto the edge token `<e>` (Eq. 1, Assumption 1: orthonormal source,
target and content embeddings). It also assumes, by induction, that thought `c` is the
uniform superposition of the c-ball `N_c` (all nodes within `c` hops of the root). It then
studies ONE parameter of layer 2.

Layer 2 is attention-only and LINEAR: the score function is the identity, `V = I`
(Eq. 2, "we adopt the linear attention"). The key-query matrix is reparameterised as
`W = sum_v mu_v E(v) E_s(v)^T` (Eq. 3). So `mu` is a scalar gain on the score between the
thought and an edge whose source matches a node in the thought. The paper calls it the
index-matching logit. Permutation averaging over vertex labels makes every `mu_v` equal,
so one scalar `mu` remains (Lemma 1, Theorem 4.1).

With this model the next-node logits are (proof of Lemma 3, Theorem 4):

    xi_v = lambda * ( 1{v in N_c} + mu * d_v ),    lambda = 1/sqrt(|N_c|)

where `d_v` is the in-degree of `v` counting only sources inside `N_c`. So the readout is
a Gibbs distribution over nodes with the in-degree as the score and `lambda * mu` as the
inverse temperature, plus a carryover bonus for nodes already in the set. Nodes outside
`N_{c+1}` sit at logit 0.

Two losses are compared (Eq. 5 and Eq. 6):

    COCONUT-BFS:  l = -log( sum_{v in N_{c+1}} exp(xi_v) / sum_{v in V} exp(xi_v) )
    COCONUT:      l = -log( exp(xi_{p_{c+1}})     / sum_{v in V} exp(xi_v) )

The first pays for mass anywhere in the (c+1)-ball. The second is ordinary cross-entropy
on the one demonstrated next node `p_{c+1}` of one shortest path.

## The key results

### Theorem 1 (informal), Theorem 4 and Lemma 5 (formal): when `mu` stays bounded

Under COCONUT, gradient flow from `mu(0) = 0` follows (Theorem 4.1, Eq. 12):

    d mu / dt = alpha / (n sqrt(K)) * ( d* - F(mu) ),   K = |N_c|

    F(mu) = sum_{u in N_{c+1}} d_u exp(xi_u) / ( sum_{u in N_{c+1}} exp(xi_u) + c0 ),
    c0 = n - |N_{c+1}| >= 1

`F(mu)` is the model's own expected in-degree under its readout softmax, with the `c0`
unreachable nodes counted at in-degree 0. `d* = d_{p_{c+1}}` is the in-degree of the
demonstrated node. Theorem 4.2: `F` is smooth and strictly increasing, `F(-inf) = 0`,
`F(+inf) = d_max`, and `0 < F(mu) < d_max` for every finite `mu`.

- Theorem 4.3: if `d* < d_max`, a unique `mu*` with `F(mu*) = d*` exists, `mu(t)` converges
  to it monotonically, and the fixed point is locally exponentially stable.
- Theorem 4.4: if `d* = d_max`, `mu` diverges, at least as
  `(1/(lambda d_max)) ln(1 + alpha lambda d_max^2 c0 e^-lambda t / (2 n^2 sqrt(K)))` (Eq. 13).
- `mu* > 0` needs in addition `F(0) < d*` (text after Theorem 1).

Under COCONUT-BFS, Lemma 5 gives `mu(t) >= c1 ln(1 + alpha c2 t)` with
`c1 = 1/(2 sqrt(|N_c|))`, `c2 = n^-3 e^-2`. The gradient is positive for every `mu`
(Lemma 4), so `mu` never stops.

In plain words: under one-demonstration cross-entropy, `mu*` is the inverse temperature at
which the model's average in-degree matches the demonstrated node's in-degree. It is a
moment-matching fit. It is finite exactly when the in-degree does NOT single out the
demonstrated node. If the demonstrated node always has the top in-degree, the target is
separable by that feature and `mu` runs to infinity. The set loss is separable by
construction (any node in the ball counts, the root included), so it always diverges.

Scope of the theorem, stated by the paper or read from the proofs:

- The analysis is per instance: one `(G, r, p)` averaged over vertex permutations. A
  training set of many graphs is not analysed.
- Linear attention, `V = I`, cross terms `mu_2 = 0` (Remark 1), layer-1 copying already
  learned, and the induction hypothesis that thought `c` is exactly the superposition of
  `N_c`.
- Gradient flow, not Adam.

### Theorem 2 (informal) and Theorem 5 (formal): one hop per thought

If thought `c` is any positive superposition on `N_c` (weights `lambda_u > 0`, not
necessarily equal), then with `mu > 0` the next thought's token projection is (Eq. 14):

    beta_v = lambda_v 1{v in N_c}  +  mu * sum_{u in N_c} lambda_u 1{(u -> v) in E}
             (carryover)              (one-hop expansion)

Its support is exactly `N_{c+1}`, and every node in `N_{c+1}` gets positive mass. This
holds for any graph, including graphs not in the training set.

Three consequences that the paper's text does not spell out, read from Eq. 14 (inference):

1. The step is LINEAR in the set weights: `beta = (I_{N_c} + mu A^T) lambda`, with `A` the
   adjacency restricted to sources in the set. Expanding a sum equals summing the
   expansions.
2. Among newly reached nodes, the weight ratio is the ratio of weighted in-degrees and does
   NOT depend on `mu`. In this linear model a large `mu` does not shrink the support. It only
   shrinks the carryover's share and sharpens the readout softmax.
3. The sets are nested (`N_c` is inside `N_{c+1}`), so a thought past the needed hop count
   keeps the answer in the support. The weights are not equalised (no MLP filter here,
   unlike the companion construction), so nodes with many incoming paths gain weight hop
   after hop.

### Theorem 3 and Theorem 6: the prediction stage

The answer token's logits are `xi_v = mu_A lambda_v 1{v in N_C} + mu_R 1{v in {c1, c2}}`
(Lemma 2, Lemma 8): residual carryover plus a lift on the two candidates. The reachable
candidate is the only node with both. Gradient flow on this separable problem goes to the
max-margin direction with `mu_R / mu_A = lambda* + Delta_train` (Proposition C.1,
Lemma 11). Generalisation to an unseen graph needs
`max_v lambda_v - lambda_{c*} <= Delta_train` (Theorem 6). So the readout tolerates only as
much overshoot of a non-candidate node's weight as it saw in training.

### Experiments (Section 5, Appendices E and F)

- Model and recipe (Section 5): two-layer GPT-2 style decoder, d_model 768, 8 heads, from
  scratch, AdamW (0.9, 0.95, wd 1e-2), constant LR 1e-4, batch 256. Vertex labels permuted
  in train and test. 150 epochs at stage 1, 25 per later stage, 350 epochs in total, earlier
  stages mixed in at probability 0.1. Final test accuracy 96.2 %.
- ProsQA subset (Table 3): 14,785 train problems, about 22.8 nodes and 36.5 edges, mean
  solution length 3.5.
- Figure 3, Section 5.1: the frontier-minus-other attention logit difference (the proxy for
  `mu`) "saturates around 60 after ~125 epochs" at stage 1. At stage 2 it forms faster. The
  pattern appears at c = 3 and c = 4, which the text describes as length generalisation
  ("never explicitly trained to generate more than two continuous thoughts").
- Appendix E.2: the BFS run uses a VARIANT, COCONUT-BFS-exp (Eq. 19), whose target is the
  frontier `N_{c+1} \ N_c` only, to avoid the trivial "predict the root" solution. Its
  stage-1 logit difference keeps rising through 150 epochs with no saturation (Figure 5).
  At c = 3, 4 the gap to COCONUT shrinks. Its test accuracy is **99.0 %**, above COCONUT's
  96.2 %.
- Figure 4, Section 5.2: the prediction-stage proxies for `mu_A` and `mu_R` rise and
  plateau after about 5 epochs. Theory predicts unbounded growth here. The authors suggest
  the interaction with thought generation as the cause and leave it open.
- Table 4, Appendix F.1 (accuracy %, one change at a time): layers 2 / 4 / 8 / 12 give
  98.8 / 97.3 / 96.5 / 67.4. Heads 4 / 8 / 12 give 98.0 / 98.8 / 98.8. d_model 384 / 768 /
  1536 give 62.0 / 98.8 / 97.7. LR 2e-4 / 1e-4 / 5e-5 give 58.1 / 98.8 / 62.1. Tied and
  untied give 98.8. The text then says the 12-layer model reaches **99.6 %** with 400
  stage-1 epochs and LR 5e-5, and LR 5e-5 reaches 97.0 % with 300 stage-1 epochs. So the
  12-layer and LR rows are optimisation budget, not capacity.
- Appendix F.2, Figure 8: in a 4-layer model, layer 1 copies (induction-head pattern) and
  the superposition shows from layer 2 onward.

Two internal inconsistencies I could not resolve from the text. The main run reports
96.2 % (Section 5), while every default row of Table 4 reads 98.8 %. The schedule of 150
epochs plus 25 per later stage to 350 epochs implies nine stages, which sits poorly with
"never explicitly trained to generate more than two continuous thoughts".

## Where it sits for the coarse-level particles

The particle design runs K copies of the coarse loop `x^{t+1} = F(D x^t + Delta B z)` from
independent random starts and lets a teacher-forced AR coda read them as a set. This paper
is the one-vector alternative: ONE state holding a weighted set, with no sampling, and the
set grows by one hop per step.

### Is `mu` the Hopfield inverse temperature?

In form, yes. `mu` multiplies the query-key score of the matching head, and a scalar gain
on attention scores is the inverse temperature `beta` of a modern Hopfield update
`xi <- X softmax(beta X^T xi)`. In substance, the theorem does not cover that case. The
proven model has linear attention, so there is no softmax across edges and no retrieval
saturation. In the proven model `mu` is the inverse temperature of the READOUT softmax over
next nodes (the Gibbs form of `xi_v` above), and the thought itself is linear in `mu`
(Eq. 14). The exploration-versus-commitment argument is made in Section 1 and Section 2.1
("the difference in weights will be significant under large mu and commonly used softmax
attention"). It is an argument about softmax attention that the theorems do not model.

My reading of how the two pictures meet (inference): with softmax attention keyed on the
summed state, the next step's score for edges out of `v` is about `mu * beta_v`. Weight
differences from in-degree then pass through an exponential at every hop. Large `mu`
compounds them into commitment to the high in-degree branch. `mu = 0` gives the paper's
"same weight on almost every node" (text after Theorem 2). That is the polarisation and
washout pair of the mixture-collapse paper
([`../continuous-mixture-collapse-2609-02049/continuous-mixture-collapse-2609-02049.md`](../continuous-mixture-collapse-2609-02049/continuous-mixture-collapse-2609-02049.md)),
with `mu` in the role of the coupling `L_t`. `mu*` sits between the two. The companion
construction avoids the problem differently: its MLP filter re-equalises the weights every
step, so ties stay exact and a large attention gain cannot pick a winner.

### What decides when depth is needed

Theorem 5 makes the depth rule exact for this task. One thought adds one hop. The answer
needs as many thoughts as its hop distance from the root (the companion paper's Theorem 1
asks for C at least the graph diameter). Depth is needed when the information the readout
needs is more lookups away than the state already holds. It buys nothing once the set
contains the answer, and extra steps keep the answer because the sets are nested. One
thought is one pass of the two-layer net because only layer 2 reads the thought. The paper
does not test whether a deeper net expands more than one hop per thought.

### For the Lean claims (inference unless marked)

- C1 (spread at most `rho^T` under contraction): not addressed. This paper's state is
  deterministic and there is one of it.
- C3, C4 (mixture read monotone in K; `log K` ceiling and the sequential Bayes read): not
  addressed. The readout here is a linear max-margin read of the candidates' weights
  (Theorem 6), not a set read over states.
- C5 (soft credit against winner-take-all): not addressed as such. The two losses differ in
  the TARGET set, not in how credit is split across the model's hypotheses. One parallel,
  and it is only an analogy: a target the model cannot single out keeps the gain finite,
  and a target that any member satisfies drives it to infinity.

## Corrections to our 2026-09-23 summary

The re-read section of
[`../reasoning-by-superposition/reasoning-by-superposition.md`](../reasoning-by-superposition/reasoning-by-superposition.md#re-read-2026-09-23)
summarised this paper from the arXiv PDF without caching it. Checked against the text on
2026-10-18:

- Correct: Eq. 5 and Eq. 6, Theorem 1 (i) and (ii) with the `d* < d_max` condition, the
  Theorem 2 carryover plus one-hop form, the "around 60 after ~125 epochs" saturation
  (Section 5.1), and every Table 4 number it quotes.
- Wrong section: the "bounded index-matching logit can balance exploration and
  exploitation" quote is from Section 1, not Section 3. Section 2.1 repeats it.
- Incomplete: the "did not saturate but kept increasing" run (Appendix E.2) uses
  COCONUT-BFS-exp (Eq. 19, frontier-only target), not the Eq. 5 loss the theorem analyses.
  It reaches 99.0 % test accuracy, above COCONUT's 96.2 %.
- Over-claim: "a loss that pays for 'any member of the set' destroys it". The paper shows
  the gain diverges. It does not show the superposition is lost. In the proven model the
  thought's support does not depend on how large `mu` is (Theorem 5).
- Over-claim: "too narrow or too deep per step" as a failure condition. Width holds
  (d_model 384 gives 62.0 %). Depth does not: the 12-layer row is 67.4 % at the default
  budget and 99.6 % with 400 stage-1 epochs at LR 5e-5 (Appendix F.1).
- Missing scope: the theorem uses linear attention with `V = I` and one instance under
  permutation averaging. The exploration-versus-commitment argument is about softmax
  attention, which the theorem does not model.

## What it does NOT establish for us

- It does not show that a set loss destroys superposition. Theorem 1 shows the gain
  diverges. In the proven linear model the support stays `N_{c+1}` for any `mu > 0`, and the
  BFS-exp run scores 99.0 % against 96.2 %. Nobody measured the thought's spread under the
  BFS-exp loss.
- It does not analyse softmax attention, an MLP, LayerNorm, Adam, or a training set of many
  graphs. Each of these is outside Theorems 1 to 6.
- It does not analyse layer 1. Copying is assumed learned.
- It does not touch stochastic states or K separate states.
- Everything is graph reachability on a 22-node symbolic vocabulary.
