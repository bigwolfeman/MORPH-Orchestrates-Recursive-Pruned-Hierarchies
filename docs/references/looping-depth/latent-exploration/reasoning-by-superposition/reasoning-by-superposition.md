# Reasoning by Superposition: reading and local cache

Read 2026-09-18.

## Citation (verified)

Hanlin Zhu, Shibo Hao (equal contribution), Zhiting Hu, Jiantao Jiao, Stuart Russell,
Yuandong Tian. *Reasoning by Superposition: A Theoretical Perspective on Chain of
Continuous Thought*. 39th Conference on Neural Information Processing Systems
(NeurIPS 2025). The page header of the proceedings PDF states the venue.
Affiliations: UC Berkeley, UCSD, Meta AI.

- Proceedings PDF: <https://proceedings.neurips.cc/paper_files/paper/2025/file/72c363c2a573ca2128bd176d3317696b-Paper-Conference.pdf>
- Code: <https://github.com/Ber666/reasoning-by-superposition>

Fetched by the orchestrator and handed over as a local file. Read in full from the
extracted text (main body plus the proof sketches in Section 4). The appendices B.4
through B.6 and C were skimmed, not read line by line.

## Local cache

- [PDF](../../../../../ignore/papers/neurips2025-reasoning-by-superposition.pdf)
- [Extracted text](../../../../../ignore/papers/neurips2025-reasoning-by-superposition.txt)
- 3,686,463 bytes, 33 pages, 2,227 text lines.
- SHA256: `d40150700bdbfbb721e65bc728c473fc266d716cba5479f5e1387d5a88870849`.

## What it actually does

The task is directed graph reachability. A prompt lists m edges as
`source target <e>` triples, then a question token, two candidate destinations, and a
root node r. The model must say which candidate r reaches.

The paper proves that a TWO-LAYER transformer with D steps of continuous chain of
thought solves this for any graph of diameter D. Continuous CoT here is Coconut's rule:
the transformer's output embedding is appended as the next input embedding, with no
sampling step.

The construction (Lemma 2) is the whole content. Let `V_c` be the set of vertices
reachable from r within c steps. The claim is that the c-th continuous thought is
exactly the normalised uniform superposition of that set:

    [t_c] = (1/sqrt(|V_c|)) * sum over v in V_c of u_v

Layer 1 uses five fixed "attention chooser" heads to copy each edge's source and target
embeddings onto that edge's own `<e>` token, into two buffer subspaces. Layer 2 uses one
head: the current thought is the query, each `<e>` token's source is the key, so the
thought attends to exactly those edges whose source is already in `V_c`, and the value
read is the TARGET stored in buffer 2. That is one BFS frontier expansion per pass. An
MLP then acts as a threshold filter that drops low-weight noise vertices and re-equalises
the survivors, and LayerNorm restores the normalisation.

    pass c            pass c+1
    [t_c] = sum V_c --------> attends to edges with source in V_c
                              reads their targets
                              MLP filters, LayerNorm renormalises
                              [t_{c+1}] = sum V_{c+1}

At the end an answer token `<A>` "measures" the superposition against the two
candidates and picks whichever has the larger inner product.

The comparison point is that the best known result for a constant-depth transformer
with DISCRETE CoT on the same problem needs O(n^2) decoding steps, where n is the vertex
count and D < n. A discrete token is a collapsed state: it forces one branch, so the
search becomes depth-first with backtracking.

## The key numbers

- Trained model: GPT-2 style, TWO layers, d_model 768, 8 heads, from scratch, AdamW,
  constant LR 1e-4, on a ProsQA subset needing 3 to 4 hops.
- Coconut reaches near-perfect accuracy. Discrete CoT and no-CoT both sit near 75 %
  (chance is 50 %). A 12-layer, 12-head discrete-CoT model improves only to 83 %.
- Layer 2 attention mass at step i, by edge group (Table 1, step 1 / step 4):
  not reachable 0.04 / 0.12, reachable 2.12 / 0.29, frontier 2.12 / 0.61,
  optimal 2.54 / 2.23. The model concentrates on reachable edges and biases further
  toward the current frontier.
- Training is a multi-stage curriculum: stage i trains the model to use i continuous
  thoughts and then predict the i-th node of the gold CoT.
- COCONUT-BFS replaces that supervision with a node drawn UNIFORMLY from the true hop-i
  frontier. It reaches the same near-perfect accuracy and the same inner-product
  geometry. The frontier structure is therefore not an artefact of optimal-path
  supervision.
- The embedding dimension needed is LINEAR in graph size, unlike the earlier arithmetic
  result of Gozeten et al. which needs exponential width.

## Where it sits in MORPH's 2x2

Deterministic, K streams, where "K streams" is inside one vector rather than across K
vectors. This is the cell the Thought Register was built for, and it is the only
published case in this batch where that cell WORKS.

The distinction matters. PLR and GRAM give a model K separate state vectors. This paper
gives one vector holding a weighted set. MORPH has run both shapes: `tul.prefix_k > 1`
with `tul.slot_cells > 1` is the K-vector shape, and the plain slot cell at
`slot_cells: 1` is the one-vector shape. Neither showed the set structure this paper
proves and measures.

## What in MORPH already tested this

The register arm (`tul.slot_cells: 4`, filed 2026-09-13,
`lab/experiments/failures/2026-09-13-arc-thought-register.md`) is the direct local test
and it failed: the four cells read effective rank 1.24 of 4 with mean pairwise cosine
0.94, whether seeded apart (`slot_cell_init: distinct`) or not (`same`), the two arms
differing by 0.008 rank units. The token K-curve was the ruler's, K1-K6 +0.0020.
The measured 0.022 CE win was prefix WIDTH, not cell content.

The per-pass evidence is the second local test. Their thought at pass c is a strictly
larger set than at pass c-1, so pass c cannot be produced in one step. On MORPH every
per-pass target so far is met in ONE pass: seven targets all read at most 0.002 on
passes 2 to 6 (`per-pass-targets-met-in-one-step`), and the code-target arms do 95.1 %
(arm A) and 92.4 % (progressive) of their cosine gain in pass 1.

The state-geometry test is the third. `val/slot_eff_rank` on `slot-spandec-strict` reads
5.76 for a row's 64 written slot states in 1024 dimensions, at mean pairwise cosine
0.7104. A superposition of k vertices in their construction is rank k by design. Ours is
low-rank and collapsed.

## What a MORPH arm implementing it would need

The pieces that exist:

- `tul.slot_cells` with `tul.prefix_k == slot_cells`, which already gives M seeded-apart
  mutable cells looping together, with cell i of slot k reading every cell of slots < k
  per `tul.loop_reach` and every cell of its own slot.
- `tul.vq_codes` and the codebook knobs, which already make a span's thought a set of
  discrete symbols. That is the closest existing thing to a vertex set, because two
  spans that pick different codes are exactly as far apart as those codes.
- `tul.tg_restrict` and `tul.tg_geometry`, the strict geometry that makes a row's spans
  reach each other only through the loop.

The piece that does not exist is a TASK with the frontier property. Their frontier at
hop c is defined by the graph, not by the text. The span analogue has to be stated
before any arm is built. The honest candidate is: cell set at pass c = the set of
EARLIER SPANS of the row that the current span depends on at distance at most c in a
dependency graph over spans. That makes the hop-distance probe
(`lab/divergence/hop_distance_probe.py`, prereg pending, owned by another agent) the
gate, because the probe measures whether a token's useful context sits one span back or
several. If the useful distance is 1, MORPH has D = 1 and their theorem predicts one
pass, which is exactly what we measure.

Nothing in the tree builds a per-cell threshold filter. Their MLP filter is what keeps
the set from smearing, and MORPH's core has no analogue.

## The instrument it implies

A SET-MEMBERSHIP readout on the cell, not a rank readout. Take the row's earlier spans,
compute each one's frozen code or prelude mean, and measure the inner product of the
cell at pass t against each. The prediction is that the set of spans above threshold
GROWS with t. Their Figure 6 is that histogram, split by reachable, frontier and
optimal. Ours would split by hop distance from the current span.

This is a different instrument from `val/slot_eff_rank`. Rank says how many directions
the cells span. Set membership says WHICH earlier spans are in the cell and whether that
set grows per pass. A collapsed cell can still have rank 4 if it collapses onto four
fixed directions, and a genuine 4-element set of the same four spans every row would
also read rank 4. Only the per-row membership separates them.

## What it does NOT establish for us

It does not establish that superposition helps on next-token text. Every result is
graph reachability on ProsQA with a symbolic vocabulary.

It does not establish that the structure is unsupervised. The paper's claim, stated
precisely, is that the SUPERPOSITION emerges without supervision to explore multiple
paths. The DEPTH structure is supervised: stage i trains thought i against the hop-i
node. MORPH ran per-pass targets of exactly that shape and they were met in one pass.
The difference is in the target, not the mechanism: hop i on a graph genuinely needs i
expansions over the edge list, whereas the conditional mean of the next span does not.

It does not give a training recipe that transfers. The curriculum runs 25 epochs per
stage over 300 epochs on a task with a 40-token vocabulary.

It does not address stochasticity. Every thought here is deterministic given the prompt.

## Re-read 2026-09-23

Re-read for the LXTUL-GK prereg
([`2026-09-23-lxtul-gk-multisample.md`](../../../../../lab/experiments/planned/2026-09-23-lxtul-gk-multisample.md))
and the LXTUL-G failure
([`2026-09-23-lxtul-g-panel.md`](../../../../../lab/experiments/failures/2026-09-23-lxtul-g-panel.md)).
The question was whether this paper solves the noise collapse: the slot loop's learned
Gaussian step dies (sigma/r 0.1 to 0.0003) under the multi-sample bound. The first
reading above covers the construction. It left out the training objective, the
companion paper that explains WHY superposition emerges, and what that means for a
sampled latent. Those are added here.

### Identity and cache (verified)

arXiv [2505.12514](https://arxiv.org/abs/2505.12514) (v1 2025-05-18, v3 2025-11-01) is
this paper. The NeurIPS 2025 proceedings PDF is now cached next to this note as
[`reasoning-by-superposition.pdf`](reasoning-by-superposition.pdf), fetched 2026-09-23
from the proceedings URL above. Its SHA256 is `d4015070...0849`, identical to the
`ignore/papers/` copy. The arXiv v3 text has the same training section and the same
COCONUT-BFS section (checked by grep, not line by line).

### Where it sits in the LXTUL lineage (verified from the tree)

This paper was item 1 of the reading order in the 2026-09-18 survey
([`../2026-09-18-latent-exploration-survey.md`](../2026-09-18-latent-exploration-survey.md)),
the batch read after the LaDiR recipe (2026-09-16). The same day it was cited as "the
prior, in theorem form" of the hop-distance prereg
([`2026-09-18-hop-distance-earning.md`](../../../../../lab/experiments/failures/2026-09-18-hop-distance-earning.md))
and as the motivation of the toy `eliminate` task
([`2026-09-18-toy-eliminate-deferred-commitment.md`](../../../../../lab/experiments/failures/2026-09-18-toy-eliminate-deferred-commitment.md)).
The first LXTUL design note
([`2026-09-19-lxtul-fan-streams.md`](../../../../../.agents/notes/proposed/architecture/2026-09-19-lxtul-fan-streams.md))
does not cite it by name. It cites the survey and takes its SHAPE from PLR: K separate
vectors, a repulsion and a gate. This paper's shape is the other one: ONE vector that
holds the set. LXTUL-G then took GRAM's stochastic shape. So the one-vector form of this
paper was never built as an LXTUL arm.

### How the model is trained (verified, Section 5.1 and 5.4)

- Two-layer GPT-2 style decoder, d_model 768, 8 heads, from scratch, AdamW
  (beta 0.9/0.95, weight decay 1e-2), constant LR 1e-4. ProsQA subset with 3 to 4 hops;
  every graph node is its own token.
- A Coconut multi-stage curriculum: "Stage i teaches the model to use i continuous
  thoughts before predicting the i-th node in the given chain of thought as the next
  token." 25 epochs per stage, 300 in total, the previous stage's data mixed in with
  probability 0.1.
- The loss at every stage is ordinary cross-entropy on ONE node: the i-th node of the
  single demonstrated path. Nothing supervises the set.
- COCONUT-BFS (Section 5.4): the stage-i target is "drawn uniformly at random from the
  frontier nodes exactly i hops from the root". It reaches the same near-perfect accuracy
  and the same inner-product geometry.
- The thought is deterministic. There is no sampling anywhere.

The Coconut paper itself (arXiv 2412.06769, Section 5.3, cached in
[`../../../tul-latent-emission/coconut/coconut.md`](../../../tul-latent-emission/coconut/coconut.md))
reports that without the curriculum the model does "not perform any better than no-CoT".
So the superposition is emergent. The depth schedule is supervised.

### Why superposition emerges: the companion paper (verified)

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, Yuandong Tian.
*Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous
Thought*. [arXiv 2509.23365](https://arxiv.org/abs/2509.23365) (v3 2026-03-01).
"Published as a conference paper at ICLR 2026" on every page. Read from the arXiv PDF
(Sections 3 to 5, Appendix E.2); not cached in the repo.

It compares two losses on the thought's next-node logits `xi_v` (Eq. 5, Eq. 6):

    COCONUT-BFS:  l = -log( sum_{v in N_{c+1}} exp(xi_v) / sum_{v in V} exp(xi_v) )
    COCONUT:      l = -log( exp(xi_{p_{c+1}}) / sum_{v in V} exp(xi_v) )

The first rewards mass anywhere in the reachable set. The second is plain cross-entropy on
the one demonstrated node. Theorem 1: under COCONUT-BFS the index-matching logit mu
"grows at least logarithmically in t, leading to unbounded attention logits". Under
COCONUT, if the demonstrated node's in-degree is not the maximum (d* < d_max), mu
converges to a finite mu* and "all attention logits remain uniformly bounded".

Their reading (Section 3): "a bounded index-matching logit can balance exploration and
exploitation: if the logit is too small, the model cannot even perform local search ...
if the logit is too large, the model might over-confidently commit to one of the
plausible search traces merely depending on local features ... and thus early discard
the correct path." Theorem 2: with mu > 0 the next thought's token projection is
`beta_v = lambda_v 1{v in N_c} + mu * sum_{u in N_c} lambda_u 1{(u -> v) in E}`, a
carryover term plus a one-hop expansion.

Measured (Section 5.1): the frontier logit difference "saturates around 60 after ~125
epochs"; the BFS-style loss "did not saturate but kept increasing" (Appendix E.2).
Stages 3 and 4 reuse the stage 1-2 mechanism with no further training ("length
generalization"). Table 4 ablation, accuracy %: L=2 98.8, L=4 97.3, L=8 96.5, L=12 67.4;
d_model 384 62.0, 768 98.8, 1536 97.7; tied and untied weights both 98.8.

In plain words: superposition is what cross-entropy does to a deterministic state when
the single demonstrated target is uncertain given what the state can see. It is the
Bayes hedge of a proper scoring rule, held in one vector. It is not produced by noise,
and a loss that pays for "any member of the set" destroys it.

### Conditions for superposition to fail (verified, three sources)

- Target identifiable from local features (d* = d_max): logits diverge, the thought
  commits (Emergence, Theorem 1).
- A set-style loss: same divergence (Emergence, Theorem 1 and Appendix E.2).
- Too narrow or too deep per step: Emergence Table 4 above; the Illusion paper's
  parameter-matched Table 3 (2L-768d 96.0 with latents against 12L-320d 72.4).
- Pretrained or fine-tuned models shortcut the latent (Illusion, Section 5.1).
- A contractive recursion erases the differences between mixtures. Backour, *The
  Dynamics of Continuous Mixture Collapse in Language Models*,
  [arXiv 2609.02049](https://arxiv.org/abs/2609.02049) (2026-09-02), Theorem 2: with
  coupling `L_t <= L_max < 2`, `|u_T| <= (L_max/2)^T |u_0| + (1/2) sum_t (L_max/2)^(T-1-t) |b_t|`,
  so a mixture not re-supplied by the field `b_t` goes to 0 at a geometric rate. Above the
  threshold (Theorem 1) one component takes over. Their setting is soft-token feedback in
  pretrained LLMs, not a hidden-state loop, so the mapping to MORPH is inferred.

### What this changes for MORPH (inferred unless marked)

1. **The noise collapse is the expected optimum, not a defect.** The coda is a
   teacher-forced AR decoder. A mixture over span hypotheses factorises as
   `p(x_t | x_<t) = sum_k w_k(x_<t) p_k(x_t | x_<t)`, and the weights update from the true
   prefix. One deterministic state that stores the branches, read by a coda that
   re-weights them as tokens arrive, is the same model as the sampled mixture. Noise then
   buys nothing and costs curvature. That is this paper's picture, with the coda as the
   "measurement" of Section 4.3. LXTUL-GK1 matched the ruler (val 4.4238 against 4.4249)
   while its sigma/r fell to 0.000339, which is what this view predicts. The XM reading
   (no multimodality worth holding) predicts the same numbers, so GK alone cannot tell the
   two apart.
2. **"Conditional mean" and "superposition" are not opposites.** Lemma 2's thought is a
   normalised SUM of item embeddings, a weighted mean. It stays decodable because the
   items are near-orthogonal (d = O(|Voc|), Theorem 1). A blurred mean is a mean in a code
   where the candidates overlap. So a CE target (span decoder, token CE) asks the slot for
   a distribution. An MSE regression onto a dense code (the LCTUL code-target arms) asks
   for a point and is the anti-superposition objective.
3. **Depth.** The thought needs pass c only because hop c's query is the set found at hop
   c-1 and each step is two layers. On the strict ruler every earlier cell is one
   attention hop away and a pass is six core blocks, so the theorem's D is about 1. Pass 1
   doing 89-95 % of the work fits that. Where the geometry forced hops
   (`prev-reach1`), content h spans back arrived at pass h-1, exactly this paper's
   schedule (`hop-staircase-on-prev-reach1`). Wolfe closed restriction geometries on
   2026-09-22, so the non-restricting way to raise D is shallower passes (fewer core
   blocks per pass, more passes), untested on the slot loop.
4. **Instrument.** Rank cannot see this. The readout that can: bucket spans by whether
   the true first token was the coda's top-1 or top-2 choice given the cell, then measure
   the cell's worth (own cell against a shuffled cell) on the rest of the span. A
   superposition helps on both buckets. A committed point helps only on top-1. A blur
   helps on neither beyond the shuffle control. The set-membership readout proposed in
   the first reading above is the per-pass version.
