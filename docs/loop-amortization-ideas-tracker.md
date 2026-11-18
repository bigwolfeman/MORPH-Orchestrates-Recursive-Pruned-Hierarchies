# Loop amortization ideas: tracker

Status: LIVE since 2026-11-17.

## Purpose

This file tracks the ideas for getting loop contribution at low training cost, and the result of each test. The goal behind them is a
1 to 4 billion parameter model that is affordable to train: the loop must earn something against a depth-1 twin, CE must stay near a
flat transformer or a plain token loop, and the training cost per token must be clearly below a plain looped model. Most ideas come
from an external review by Google's Carbon model on 2026-11-17. Its full text and our spot checks are in
[`references/external-reviews/2026-11-17-carbon-loop-amortization-ideas.md`](references/external-reviews/2026-11-17-carbon-loop-amortization-ideas.md).
The laws these ideas must respect are in [`lessons-learned-loops-and-spans.md`](lessons-learned-loops-and-spans.md).

## Diagnosis in plain words

Why a loop on one state per span earns little on next-token CE, while a loop on every token earns:

1. **The cross-span budget is small once copying leaves the hidden state.** Cutting every path across span boundaries costs about
   0.40 nats at 5k steps. Most of that is exact copying, and the pointer head takes it in one hop without the loop. What is left for
   the slot channel is small.
2. **The slot is blind to the next span's query.** A slot finishes all its passes before any token of the next span exists. It can
   only build a general summary of the past, and a general summary is done in one pass. A token loop runs its passes after it has
   seen the words directly before the prediction.
3. **A token loop earns as capacity per token.** Each token gets more nonlinear capacity on its own state. The slot loop has about
   21 times fewer positions, so it has fewer gradient samples and no more working memory per pass.

Point 2 is a proposed law. It is in the 2026-11-17 additions of the lessons document.

## Ideas

Costs are Carbon's estimates against a plain 6-pass token loop, in FLOPs. They are not measured, and they ignore gather and scatter
overhead.

| Idea | Mechanism in one line | Strict TUL? | Claimed cost (Carbon's estimate) | Collides with (repo result) | Status |
|---|---|---|---|---|---|
| Hop-task slot loop test | Mix hop chains and friendship graphs into web text; read accuracy by hop count for the slot loop against its depth-1 twin | Yes | 0.22x (the strict slot loop) | Slot loop K1-K6 near 0 on web text (`failures/2026-10-14-parcae-minimal-slot-loop-5k.md` in the testbed) | Running. Gate passed 2026-11-17: the flat arm reads 1.000 at 1 and 2 hops after a web-text start. Plan file in the Parcae testbed; snapshot under `references/looping-depth/parcae/lxtul-testbed-2026-11-17-hops/` |
| Shared pass-1 cache with routing (Carbon rank 1) | Every token does pass 1; a learned router sends the top 20 % of tokens through later passes, which read the shared pass-1 keys and values | No | 0.43x to 0.45x | H-Net selective token depth probe (lost 0.008 bpb, eval only); boundary depth probe | Building 2026-11-17, one seed, 5k |
| Narrow core (Carbon rank 2) | Loop every token, at half width inside the weight-tied core; prelude and coda stay full width | No | 0.37x to 0.39x | Density panel (`lab/experiments/failures/2026-09-09-arc-density-panel.md`); SMELT narrowing (arXiv 2609.01343) | Building 2026-11-17, one seed, 5k |
| Slot pointer composition (Carbon rank 3) | Each slot keeps pointer tables over token positions; each pass composes them through slot attention, so a pass chases one more hop of exact token ids | Yes | 0.22x | Pointer cell key (-0.0074 paired CE); hop-distance earning (carry loses a third per pass) | Building 2026-11-17, one seed, 5k |
| Teacher top-64 KL (Carbon rank 4) | A frozen looped teacher's top-64 next-token distribution enters the main loss as a KL term (and, in the full form, the input) | Either | 0.34x strict student; 0.56x 2-pass token student | Bypass law if the teacher input has no dropout; H-Net teacher result (top-64 KL beat CE by 0.058 bpb) | Building 2026-11-17, one seed, 5k |
| Delayed depth (Carbon's "wrongly closed" idea) | Train at depth 1 for the first 75 % of tokens, then switch on mixed depth with the optimizer intact | Either | 0.49x on a plain token loop | The standing rule against depth curricula; see Rulings and cautions | Building 2026-11-17, one seed, 5k |
| Slot fast-weight adapter (Carbon rank 5) | The slot loop writes a rank-8 linear adapter that the next span's tokens run in the coda | Yes | 0.23x | Slot reread (+0.0003 K1-K6); trajectory prefix pad-cell artefact | Not started |
| Four-token chunk loop (Carbon rank 6) | Fixed 4-token chunks loop in series; the coda sees its chunk and the previous chunk's tokens; a sparse top-64 bridge between passes | No | 0.39x | Chunk loop with skip (K1-K6 fell to near 0); block memory (0.168-nat CE gap) | Not started |
| Grown mean-3 loop (Carbon rank 7) | Start with 3 core blocks at 1 pass, duplicate them to 6, then loop at mean depth 3 with the shared pass-1 cache | No | about 0.42x | Depth ladder ship (un-clamped mean-3 lost wall clock) | Not started |

## Rulings and cautions

- **Delayed depth, Wolfe 2026-11-17.** He is hesitant. In his words, the first 600 to 3,000 steps "are some of the most important
  steps that really start to solidify what the weights are going to represent, especially as the LR warms up", and a depth-1 start
  gives that up for the loop. He also said: "That doesn't mean that it won't work". So it is allowed as a test.
- **One seed first.** Each idea gets one seed at 5k steps with its control. One seed cannot settle a small gap; a second seed
  follows only where the first reading is worth it.
- **Kill thresholds in Carbon's text are not ours.** Our protocol reads behaviors against a control (direction, ordering, sign).
  It does not use invented thresholds. Each test gets its own planned file with one main prediction before it runs.
- **Rank 3 caution.** The slot pointer composition builds the hop mechanism by hand. A positive reading on a hop task says less
  than the same reading from an arm that was not shaped for the test.

## Results log

| Date | Idea | Arms | Reading | Verdict word | Filing |
|---|---|---|---|---|---|
| | | | | | |
