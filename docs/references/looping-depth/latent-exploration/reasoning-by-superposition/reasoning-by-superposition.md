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
