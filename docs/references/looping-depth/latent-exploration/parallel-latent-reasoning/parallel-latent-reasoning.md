# Parallel Latent Reasoning (PLR): reading and local cache

Read 2026-09-18.

## Citation (verified)

Jiakai Tang, Xu Chen, Wen Chen, Jian Wu, Yuning Jiang, Bo Zheng. *Parallel Latent
Reasoning for Sequential Recommendation*.
[arXiv 2601.03153v1](https://arxiv.org/abs/2601.03153), submitted 2026-01-06.
Affiliations: GSAI, Renmin University of China, and Alibaba Group. The PDF uses an ACM
template with the venue placeholders unfilled ("Conference acronym XX, Woodstock, NY",
copyright year 2018), so no venue can be cited from the source.

Read in full: abstract through Section 6.5, including the theory section and every
table. The proofs in Appendix A were not read.

## Local cache

- [PDF](../../../../../ignore/papers/2601.03153v1-plr-parallel-latent-reasoning-seqrec.pdf)
- [Extracted text](../../../../../ignore/papers/2601.03153v1-plr-parallel-latent-reasoning-seqrec.txt)
- 1,260,246 bytes, 12 pages, 896 text lines.
- SHA256: `a6d9d97cfb60b29afe920538828322e3df36d831fbd399274fbcc79ee743676c`.

## What it actually does

The setting is next-item recommendation, not language modelling, but the machinery is
precisely the arm MORPH has not built.

A transformer encodes a user's item sequence. The last position's state h0 is the entry
to a latent reasoning loop of T steps with weights SHARED with the encoder. PLR adds a
width axis on top:

    h_{0,m} = h0 + tau_m            m = 1..M     M learnable trigger tokens
    h_{t,m} = Attn(q = h_{t-1,m} + r_t,
                   k,v = [X_0 ; h_{1,m} ; ... ; h_{t-1,m}]) + h_{t-1,m}

Each stream gets its own start point by ADDING a learned trigger vector. Attention is
stream-isolated and causal within a stream, with the encoder's key-value pairs shared
across all streams (so a KV cache serves every stream).

Three mechanisms sit on top.

1. Global reasoning regularisation. Every reasoning state is projected to a softmax over
   the item vocabulary and all T*M states are pushed apart by pairwise bidirectional KL.
   The penalty is global: it separates steps WITHIN a stream and streams from each other
   at the same time.
2. Reasoning contrastive learning. Two forward passes with independent dropout on both
   the hidden states and the attention scores; in-batch InfoNCE makes the two views of
   the same user agree.
3. Mixture-of-Reasoning-Streams. A softmax gate reads h0 and produces M weights; the
   final representation is the gated sum of the per-stream mean-pooled states.

At inference the score uses `h0 + z_rea`, the encoder state plus the reasoning output,
which they call dual-process. At TRAINING only `z_rea` is used, to stop the model routing
around the loop.

The theory section is the useful part. Theorem 4.1 is Jensen: the ensemble loss is at
most the mean individual loss minus a specialisation benefit that is zero exactly when
all streams agree. Theorem 4.4 is the one to take away:

    D(T) = L^{2T} D(0) + o(L^{2T})

for an L-Lipschitz shared reasoning map, where D(t) is mean pairwise stream distance at
step t. If L < 1 the streams collapse EXPONENTIALLY in depth. Corollary 4.5 makes the
trade-off explicit: refinement improves with T while the diversity benefit decays like
exp(-2 gamma T).

## The key numbers

Best-baseline improvements on three Amazon Review 2023 domains across three backbones.
The strongest single cell is UniSRec on CDs & Vinyl, Recall@10 +14.91 %. SASRec on
CDs & Vinyl gives Recall@20 +12.07 %. Several cells are negative
(BERT4Rec Video & Games Recall@10 -3.53 %, UniSRec Video & Games NDCG@10 -3.40 %).

Ablation, SASRec, Recall@20 (CDs & Vinyl / Video & Games), full PLR is 0.0873 / 0.1033:
without MoRS 0.0785 / 0.0997, without contrastive 0.0782 / 0.0970, without KL
0.0853 / 0.0979. The gate is the largest single contributor; the KL repulsion is the
smallest and they say so.

Sensitivity: the optimum is M = 2 streams and T = 2 steps. Both DEGRADE beyond that.
The KL weight optimum is lambda = 0.1, and they report that too much repulsion hurts by
pushing streams apart faster than they can converge. Dropout 0.2 to 0.5.

Cost: +5.22 % FLOPs and +5.80 % latency over the base model, because the encoder KV
cache is shared and the M streams are vectorised.

The number that should govern any MORPH arm is the oracle-ceiling analysis (their
Figure 4). Evaluating every reasoning step independently and taking the best per user,
PLR and depth-only ReaRec reach the SAME ceiling, NDCG@20 in the range 0.0402 to 0.0416.
PLR wins by closing the gap to that ceiling, not by raising it.

## Where it sits in MORPH's 2x2

Deterministic, K streams, with an explicit diversity term and a learned selector. It is
the exact shape of the proposed empty-cell arm, built and measured on a different task.

## What in MORPH already tested this

The K-stream half is the Thought Register, and the collapse it measured is Theorem 4.4
realised: four cells that start apart, share one map, and end at rank 1.24 of 4 with
cosine 0.94. PLR's answer to the same theorem is a repulsion penalty, and their own
ablation says it is their WEAKEST component.

The repulsion half exists in the tree but in the wrong place.
`tul.row_contrast_lambda` / `tul.row_contrast_tau` is an InfoNCE that asks which of a
ROW's slots a given next span belongs to, so it separates slots, not cells. Its own
docstring records that at `slot_cells > 1` it reads the cells' MEAN.

The selector half exists outside the loop. `tul.code_grade` ranks K coda samples with a
grader; the loop is still pushed toward one point. `tul.mux_readout` mixes per-pass
states, not per-stream states.

The trigger half exists and is the one piece that already matches: `TULSlotRegister`
uses M learned queries to pool M different vectors from the span's own prelude states,
which is PLR's per-stream trigger with more structure.

## What a MORPH arm implementing it would need

Exists: `tul.slot_cells` (M cells), `tul.slot_cell_init` (distinct or shared queries,
the seeding control), `tul.prefix_k` (must equal `slot_cells`), `tul.loop_reach`
(the cross-slot reach), `tul.spandec_reads_cells` (a decoder that cross-attends the
cells instead of grading their mean).

Does not exist:

- A per-CELL repulsion at chosen passes. `row_contrast` is across slots.
  A new key, say `slot_cell_repel_lambda` with a pass mask, is required, and
  `KNOWN_TUL_KEYS` will refuse it until it is added.
- A learned softmax MIXTURE over cells into the coda's read. Today cell i is written 1:1
  into prefix cell i through `W_prefix[i]`, and every other reader (MUX, span decoder,
  SIGReg, the energy) takes the MEAN. PLR's gate is a third option and it is the one
  their ablation says carries the most.
- The stream-isolated attention pattern. In MORPH, cell i of slot k reads EVERY cell of
  slots < k and every cell of its own slot. PLR isolates streams entirely. Making cells
  isolated across passes is a forward change in `_tul_core`, not a config knob.

## The instrument it implies

Oracle-over-stream, which is their Figure 4 and the one number that keeps this honest.
Decode from each cell separately, score each against the true next span, take the best
per span, and compare with the mixture's score and with the K = 1 ruler. Their result is
that the oracle ceiling did NOT move. If ours also does not move, the streams are not
exploring anything, whatever the mixture gains.

The second instrument is stream distance per pass, D(t), which is directly Theorem 4.4.
We already log `val/slot_cell_eff_rank`; D(t) resolved BY PASS is the missing half,
because the theorem predicts a specific exponential shape and a measured L.

## What it does NOT establish for us

It does not establish that width helps on language. Sequential recommendation over
sparse behaviour data is a different problem and the paper says its own gains are
largest on the sparsest dataset.

It does not establish that width raises the ceiling. Their own oracle analysis says it
does not.

It does not establish that more streams is better. M = 2 is optimal and M > 2 degrades,
so a K = 4 proposal is already outside their measured range.

It does not establish that the repulsion term is what makes streams diverge. Removing
the KL costs 0.0020 Recall@20 on one dataset and 0.0054 on the other, both smaller than
removing the gate.

Their theory does not prove that a shared map MUST contract. Assumption 4.3 assumes
L-Lipschitz and the interesting case assumes L < 1; the exponential collapse is a
consequence of that assumption, not of transformers in general.
