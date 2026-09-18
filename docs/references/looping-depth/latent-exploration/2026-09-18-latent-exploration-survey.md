# Latent exploration: nine papers against MORPH's flat slot loop

Written 2026-09-18. Nine papers, all fetched and verified from source, all cached under
`ignore/papers/`. Per-paper notes sit beside this file.

The question this batch was pulled to answer: MORPH's TUL slot loop earns nothing from
depth (twelve arms at K1-K6 about +0.002 nats, against the plain looped model's 0.033),
and the working hypothesis is that every slot target so far has a ONE-STEP OPTIMUM. Each
target is the conditional mean of some future quantity, so nothing in training asks pass
2 to differ from pass 1. The literature question is whether anyone has made a latent
loop explore, and under what conditions.

## Verification summary

| # | Paper | Verified id | Venue as found |
|---|---|---|---|
| 1 | Reasoning by Superposition | NeurIPS 2025 proceedings | NeurIPS 2025 (stated in PDF) |
| 2 | The Illusion of Superposition? | arXiv 2604.06374v2 | **COLM 2026** (brief said unplaced) |
| 3 | LLMs are Single-threaded Reasoners | arXiv 2508.03440v4 | ICLR 2026 via OpenReview `ASLuOoP78o`; the PDF still says "Preprint" |
| 4 | GRAM, Generative Recursive Reasoning | arXiv 2605.19376v2 | Preprint, no venue claimed |
| 5 | Parallel Latent Reasoning (PLR) | arXiv 2601.03153v1 | ACM template, venue placeholders unfilled |
| 6 | Parallel Test-Time Scaling for Latent Reasoning | arXiv 2510.07745v4 | ACL 2026 Main (stated in PDF and arXiv comment) |
| 7 | Latent Thought Credit (LTC) | arXiv 2608.01593v1 | No venue claimed |
| 8 | Latent Thought Flow (LTF) | arXiv 2606.16222v1 | Preprint, no venue claimed |
| 9 | LSRL | ACL Anthology `2025.findings-emnlp.669` | Findings of EMNLP 2025, pp. 12534-12545 |

Corrections to the third-party list that was the starting point: the Illusion paper is
published at COLM 2026; LSRL is NOT on arXiv and has ONE author, Hangliang Ren; the
Single-threaded paper's own PDF has not been updated to reflect its ICLR acceptance.
No paper in the list turned out not to exist.

## The 2x2, with every cell filled

Axes: how many latent trajectories the model carries, and whether the transition is
deterministic.

```
                    ONE STREAM                        K STREAMS
              +--------------------------------+--------------------------------+
              | MORPH: the strict slot loop    | MORPH: Thought Register        |
              |   tul_slot_spandec_strict      |   tul.slot_cells: 4            |
              |   K1-K6 +0.002, pass 1 does    |   cells rank 1.24 of 4,        |
DETERMINISTIC |   89-95% of the work           |   cos 0.94, K-curve = ruler    |
              |                                |                                |
              | PAPERS: LSRL (grade every      | PAPERS: Reasoning by           |
              |   depth, r=8 works, r=4 flat)  |   Superposition (frontier set  |
              |   Illusion (fine-tuned Coconut |   in ONE vector, proven and    |
              |   latents worth <= 1.0 point)  |   measured, d_model 768)       |
              |   Single-threaded (the forward |   PLR (M triggers + KL         |
              |   pass prunes a mixture in     |   repulsion + gate; M=2 and    |
              |   2-3 layers)                  |   T=2 optimal, both degrade)   |
              +--------------------------------+--------------------------------+
              | MORPH: LCTUL flow thinker and  | MORPH: NOTHING.                |
              |   LCTUL-D                      |   tul.code_grade is selection  |
              |   tul.code, tul.code_discrete  |   OUTSIDE the loop over K coda |
              |   context-blind: past worth    |   samples; the loop is still   |
STOCHASTIC    |   1% of the flow loss;         |   regressed to one point.      |
              |   Euler depth provably flat    |                                |
              |                                | PAPERS: GRAM (learned Gaussian |
              | PAPERS: Latent Thought Flow    |   guidance + target-conditioned|
              |   (GFlowNet over depth; its    |   posterior + LPRM selector)   |
              |   sampler CHOSE depth ~1.9)    |   Parallel-TTS (MC-dropout or  |
              |                                |   AGN at test time + LatentRM) |
              |                                |   LTC (K thoughts x M answers, |
              |                                |   per-thought advantage)       |
              +--------------------------------+--------------------------------+
```

The empty cell is stochastic x K streams, with the streams present at TRAINING time, an
explicit diversity term, a learned selector, and an oracle-over-stream instrument.
`tul.code_grade` is the nearest thing in the tree and it is not in this cell: it grades K
continuations produced by the CODA from one cell, so the loop still has a point target.

## The two negative results, and our local replication

Two papers say the mixture does not survive.

**The Illusion of Superposition (COLM 2026)** measures three regimes. Off-the-shelf
models process a soft token indistinguishably from its argmax: KL falls to about 1e-4 in
the middle layers and cosine similarity is 0.996. A fine-tuned Coconut model does not use
its latents at all: deleting them costs at most 1.0 point across GPT-2 and three SmolLM2
sizes, and entity probing shows the final answer dominating from step 0. Only from-scratch
models use them, and there WIDTH decides: at matched 15M parameters, 2L-768d scores 96.0
with latents and 30.9 without, while 12L-320d scores 72.4 and 70.0.

**LLMs are Single-threaded Reasoners (ICLR 2026)** gives the mechanism. Three forward
passes at the same step show that the soft token's prediction matches its top-1 token's
prediction (JS near 0) and not its second's (JS near maximal). Logit Lens shows both
paths alive for two or three layers, then the top-1 path rising to 1.0. The forward pass
is a pruner. Their fix, Gumbel-Softmax at tau 0.5, is the only variant that beats sampled
token CoT (QwQ average 83.04 against 82.35, vanilla soft 80.06).

**The Thought Register is MORPH's replication of both.** Four cells per span, seeded apart
by four learned queries, looped together through the shared core, written 1:1 into four
prefix cells. At 5000 steps they read effective rank 1.24 of 4 and mean pairwise cosine
0.94. The distinct-init and same-init arms land 0.008 rank units apart, so seeding them
apart changed nothing. The per-row rank FELL, 13.85 to 10.57, while the cell count
quadrupled. The token K-curve was the ruler's, K1-K6 +0.0020. The measured 0.022 CE win
was prefix WIDTH, and the clean width control (`tul_slot_spandec_strict_pk8.yaml`) is
still queued. That is the collapse the negative papers describe, on our stack, under our
recipe.

PLR supplies the closed form for why. Under an L-Lipschitz shared reasoning map, mean
pairwise stream distance obeys `D(T) = L^(2T) D(0)`, so at L < 1 streams collapse
exponentially in depth. MORPH's core is measured contractive on the healthy arms (the
slot-loop gain constraint holds the map at 0.89). A contractive shared map and a demand
for divergent streams are in direct opposition, and the register lost that argument.

LTC adds the same finding as a downstream-utility measurement rather than a geometric
one: between-thought variance of correctness falls 0.0244 to 0.0043 from initial to final
policy, while within-thought answer variance rises 0.0376 to 0.0568. Ratio 1.54 to 13.10.
Training removes the differences between the K thoughts.

## Reading order for Wolfe

1. **Reasoning by Superposition** (NeurIPS 2025). The only positive existence proof in
   the batch, and it is a proof: D continuous steps for a D-diameter graph, with the
   frontier set visible in the trained model's attention and inner products, at d_model
   768 and two layers.
2. **The Illusion of Superposition** (COLM 2026). The counterweight, and the one that
   tells us which knob the positive result actually hangs on, which is width, not depth.
3. **PLR** (arXiv 2601.03153). The empty cell already built: triggers, repulsion, gate,
   a theorem for why the register collapsed, and an oracle-ceiling result that says width
   did not raise the ceiling.
4. **LLMs are Single-threaded Reasoners** (ICLR 2026). Read after the first two because
   it supplies the layer-by-layer mechanism of collapse and the one fix that measurably
   works.
5. **Parallel Test-Time Scaling** (ACL 2026). The cheapest probe we can run: no
   retraining, and coverage@N is the oracle-over-stream instrument we want.
6. **GRAM** (arXiv 2605.19376). The most ambitious version of the empty cell, with the
   ablation that says randomness alone is not enough and a target-conditioned
   deterministic map scores zero.
7. **LTC** (arXiv 2608.01593). Read for its Table 4 alone, which is the collapse measured
   in utility units, and for the budget split saying answer replication beats more
   thoughts.
8. **LSRL** (Findings of EMNLP 2025). Read for one number: the whole recipe is flat at
   r = 4 and works at r = 8. A depth threshold for per-depth supervision.
9. **Latent Thought Flow** (arXiv 2606.16222). Read last, for the uncomfortable fact: a
   sampler that was free to choose depth under an accuracy reward chose 1.9 steps.

## Proposal for the empty cell, on the strict spandec ruler

Base arm: `tul_slot_spandec_strict.yaml` at `prefix_k: 4`, which is the standing ruler
for the register comparison. One new arm, four factors, all four required because the
literature says each alone fails.

**K = 4 streams through the shared core.** `tul.slot_cells: 4` with
`tul.prefix_k: 4`. Exists today; this is the register's forward, unchanged.

**Per-stream learned trigger.** `tul.slot_cell_init: distinct` already gives four learned
queries pooling four different vectors from the span's own prelude states. That is PLR's
trigger with more structure, and we know from the sameinit control that it is not
sufficient on its own.

**Cosine repulsion at passes 1 and 2 only.** This does NOT exist.
`tul.row_contrast_lambda` separates a row's SLOTS, not a span's CELLS, and its own
docstring records that at `slot_cells > 1` it reads the cells' MEAN. The arm that pairs
it with the register (`tul_slot_register_m4_contrast.yaml`,
`lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md`) is still
`Status: planned` and has never run. A new key is needed, and `KNOWN_TUL_KEYS` will
refuse it until it is added deliberately. Apply it at the EARLY passes because PLR's
Theorem 4.4 says the collapse is exponential in depth: repelling at pass 6 fights a
factor `L^12`, repelling at pass 1 fights `L^2`.

**Learned softmax mixture into the coda's read.** Also does not exist. Today cell i goes
1:1 into prefix cell i through `W_prefix[i]`, and every other reader takes the MEAN.
PLR's ablation says the gate is its LARGEST contributor (Recall@20 0.0873 to 0.0785
without it), larger than the repulsion.

Expected cost: the register arm's cost, plus a gate of `d_model * 4` parameters, plus one
pairwise-cosine term over four cells at two passes. Roughly the register's wall clock.

### The three instruments

1. **Oracle-over-stream.** Decode from each of the four cells separately through the
   coda, score the per-span CE of each, and take the per-span minimum. Report three
   numbers on identical rows: the K = 1 ruler's CE, the mixture's CE, and the oracle's
   CE. This is PLR's Figure 4 and Parallel-TTS's coverage@N in the units MORPH is scored
   in. Measure the oracle's own selection bias by choosing the stream on the first half
   of a span and scoring on the second.
2. **Stream distance per pass, D(t).** Mean pairwise cosine distance among the four cells
   at each pass t, per span. PLR's theorem predicts `D(t) = L^(2t) D(0)`; fit L and report
   it. Today we log `val/slot_cell_eff_rank`, which sums over passes and cannot see the
   shape. Read this per pass, per the `depth-summing-instruments-hide-pass-trades` rule.
3. **Per-stream delayed resolution.** For each cell, the entry-anchored displacement at
   each pass, and the pass index at which its own CE contribution stops improving. The
   hypothesis that would justify K streams is that different cells resolve at different
   passes. The null is that all four resolve in pass 1, which is what every MORPH arm has
   read so far (pass 1 does 89.2 % of the plain fixed recipe's displacement, 95.1 % of
   arm A's cosine gain, 92.4 % of progressive's; dead arms read 97 to 105 %).

Report pass-1 level and per-pass slope as two numbers and never their sum.

### The falsifying prediction

**If oracle-over-stream at K = 4 does not beat K = 1 by more than the mixture's own cost,
there is nothing to explore over on this corpus, and the whole width branch closes.**

The threshold has to be stated before the run. The mixture's own cost is the CE a K = 1
model pays for having its readout widened to four gated cells, which the pk8 width control
already isolates. If the oracle sits inside that, the four cells are four copies with a
wider readout, and that is the register's verdict repeated at higher cost.

The reason to state the prediction this way rather than on the mixture's CE: PLR's
oracle-ceiling result says their width did NOT raise the ceiling, only closed the gap to
it. If MORPH's oracle ceiling is also unmoved, a better gate would buy CE without buying
exploration, and we would be reading a readout improvement as a depth result. That is the
exact failure the `trajectory-prefix-is-width-plus-pad-artefact` note already caught once.

A second falsifier, cheaper and available first: run the Parallel-TTS probe on the
EXISTING `slot-spandec-strict` checkpoint at 5000 steps. Perturb the cell with dropout or
additive Gaussian noise at each pass, draw N = 16 trajectories, and take the oracle CE.
No training. If a trained deterministic loop has no reachable better state under
perturbation, the training-time version is unlikely to find one either. This is not proof
(the causal-fit probe already found no headroom by gradient search, +0.252 nats WORSE
than the loop, and gradient search and sampling can disagree), but it costs about an hour
and it gates a multi-day arm.

## What the literature does not settle

**It does not establish that the conditional-mean target is the cause.** This is the
session's parent hypothesis and no paper tests it. The closest evidence is circumstantial
and points in the right direction without reaching it:

- GRAM's mechanism ablation scores 0.00 on both tasks for "guidance only", a deterministic
  map conditioned on the target. That says a target-conditioned deterministic map fails,
  not that a conditional-mean target causes flat depth.
- Reasoning by Superposition's target at pass i is the hop-i node, which genuinely needs i
  frontier expansions over the edge list. Its DEPTH STRUCTURE IS SUPERVISED by the
  multi-stage curriculum; what emerges unsupervised is the superposition, not the depth.
  So the paper is consistent with "targets without a one-step optimum earn depth", and
  it does not isolate that factor.
- LTF's sampler, free to choose depth under a reward that pays for accuracy, chose 1.9
  steps on math word problems. That is the one-step optimum arrived at by a different
  route on a different corpus, and it is evidence about those tasks, not about ours.
- LSRL is flat at r = 4 and works at r = 8, which is a DEPTH threshold, not a target
  property, and it is one comparison with no seeds.

**It does not establish that width transfers to next-token text.** Every positive result
in this batch is on a task with a checkable answer: graph reachability, Sudoku, N-Queens,
graph colouring, GSM8K, next-item recommendation. Coverage@N and pass@k are natural there
and have no clean analogue in per-token CE. The standing rule
(`never-argue-nlp-does-not-need-loop-depth`) forbids concluding the reverse from that, and
this section does not: it says the transfer is unmeasured, in either direction.

**It does not establish that a diversity penalty stops the collapse.** PLR's own ablation
says the KL repulsion is its weakest component and that too much of it hurts. GRAM's
"stochasticity only" row holds on Sudoku (94.88) and collapses on N-Queens (50.27).

**It does not establish that more streams is better.** PLR's optimum is M = 2 and M > 2
degrades. LTC's budget split at B = 8 prefers 2 thoughts x 4 answers over 4 thoughts x 2
answers by 1.97 points. A K = 4 proposal is already outside the measured optimum of both,
and the honest reason to run K = 4 anyway is that MORPH's register arm was K = 4 and the
comparison should be one factor from it.

**It does not resolve our reader confound.** `the-reader-was-the-limit` measured the same
cell at -4.6413 nats to a frozen coda and +0.1768 to one given 10k steps to adapt. Every
oracle-over-stream number in the proposal above is a property of a (cell, reader) PAIR.
The arm must either train the reader alongside the streams or report both readings, and no
paper in this batch handles that distinction.

## Cross-references

- Coconut and Sentence Embedding Prediction are already in the library and carry a short
  "2026-09-18 exploration reading" section appended for this survey:
  [`../../tul-latent-emission/coconut/coconut.md`](../../tul-latent-emission/coconut/coconut.md),
  [`../../tul-latent-emission/sentence-embedding-prediction/sentence-embedding-prediction.md`](../../tul-latent-emission/sentence-embedding-prediction/sentence-embedding-prediction.md).
- Loop stability, a separate axis and settled separately:
  [`../2026-09-07-loop-stability-survey.md`](../2026-09-07-loop-stability-survey.md).
- Depth and scaling:
  [`../virtual-logical-depth.md`](../virtual-logical-depth.md).
