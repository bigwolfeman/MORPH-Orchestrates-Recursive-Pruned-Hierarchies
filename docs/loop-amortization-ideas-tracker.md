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
| Shared pass-1 cache with routing (Carbon rank 1) | Every token does pass 1; a learned router sends the top 20 % of tokens through later passes, which read the shared pass-1 keys and values | No | 0.43x to 0.45x | H-Net selective token depth probe (lost 0.008 bpb, eval only); boundary depth probe | Ran 2026-11-18, see Results log |
| Narrow core (Carbon rank 2) | Loop every token, at half width inside the weight-tied core; prelude and coda stay full width | No | 0.37x to 0.39x | Density panel (`lab/experiments/failures/2026-09-09-arc-density-panel.md`); SMELT narrowing (arXiv 2609.01343) | Ran 2026-11-18, see Results log |
| Slot pointer composition (Carbon rank 3) | Each slot keeps pointer tables over token positions; each pass composes them through slot attention, so a pass chases one more hop of exact token ids | Yes | 0.22x | Pointer cell key (-0.0074 paired CE); hop-distance earning (carry loses a third per pass) | Ran 2026-11-18, see Results log |
| Teacher top-64 KL (Carbon rank 4) | A frozen looped teacher's top-64 next-token distribution enters the main loss as a KL term (and, in the full form, the input) | Either | 0.34x strict student; 0.56x 2-pass token student | Bypass law if the teacher input has no dropout; H-Net teacher result (top-64 KL beat CE by 0.058 bpb) | Ran 2026-11-18, see Results log |
| Delayed depth (Carbon's "wrongly closed" idea) | Train at depth 1 for the first 75 % of tokens, then switch on mixed depth with the optimizer intact | Either | 0.49x on a plain token loop | The standing rule against depth curricula; see Rulings and cautions | Ran 2026-11-18, see Results log |
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
| 2026-11-18 | Hop task, slot loop (seed 1 of 2; token loop pending) | Slot loop, its depth-1 twin, flat model; web-text start, 20k mixed stream | Chain accuracy at 1 / 2 / 3 / 4 hops: flat 1.00 / 1.00 / 0.91 / 0.49; slot loop 1.00 / 0.37 / 0.27 / 0.25; twin 1.00 / 0.60 / 0.30 / 0.25. Hard hops (6 to 32) and graph questions: both slot arms at chance (0.25, 0.5); flat reads reachability 0.885. The slot loop's 2-hop accuracy is 0.26 at eval depth 1 and 0.37 at depth 2 and above, the only depth signal. Web CE: flat 3.511, slot loop 3.594, twin 3.592 | Open (seed 1 leans not supported: loop not above twin) | Parcae `lxtul/hops`, `docs/experiments/planned/2026-11-17-parcae-hop-tasks-slot-loop.md` |
| 2026-11-18 | Narrow core (rank 2) | Narrow token loop, its depth-1 twin; filed full-width loop | CE: narrow loop 3.7375 at depth 6 (K1-K6 +0.0217), twin 3.7342 at depth 1, full-width loop 3.707 to 3.726. ms/step 157 / 98 / 283. Peak 14.8 GiB against 21.1 | can't tell (twin 0.003 ahead, inside the band) | `...-narrow-core-token-loop.md` |
| 2026-11-18 | Teacher top-64 KL (rank 4) | Slot loop, slot twin, flat model, each with the 42k looped teacher; no-teacher twin control | CE with teacher: slot loop 3.6216, twin 3.6205, flat 3.5625. Without: 3.7646 (filed), 3.7651, 3.7034 (filed). The teacher buys 0.141 to 0.145 nats on every student. Loop and twin tie with and without it; the loop's K-curve stays flat. The teacher forward adds about 145 ms/step | can't tell on the main prediction (lead unchanged, inside the band); both other predictions hold | `...-teacher-top64-kl.md` |
| 2026-11-18 | Slot pointer composition (rank 3), 5k | Composing slot loop, its twin, plain slot loop; hop stream from web-text weights | No reading: the slot keys stayed closed (logit offset -6 at start, -5.6 to -6.05 at the end) and 5k is too short (the control also reads chance at 2 hops). Rerun queued with open keys at 20k (dated amendment) | broken run, not graded | `...-slot-pointer-composition.md` |
| 2026-11-18 | Routed loop (rank 1) | Routed loop (19 % of tokens loop), Poisson(2) token loop, all-token shared cache | CE@6: routed 3.7251, Poisson(2) control 3.7311 (3.7267 at depth 2), all-token shared cache 3.7253, filed full loop 3.707 to 3.726. ms/step 227 / 177 / 270 / 283. The routed K-curve is flat (3.7250 at depth 1, 3.7251 at depth 6) and the router scores collapsed (std 0.0002), so the routed passes add nothing measurable. The shared cache read costs nothing | can't tell (inside the band); the K-curve prediction failed | `...-shared-cache-routed-loop.md` |
| 2026-11-18 | Delayed depth, 5k | Full loop with the switch at step 3,750; depth-1 twin; narrow loop with the switch (reading) | CE@6: delayed 3.7203, filed full-depth loop 3.707 to 3.726, depth-1 twin 3.7195 at depth 1. K1-K6: delayed 0.046 after 1,250 deep steps, full 0.048. ms/step 136 before the switch, 293 after (about 0.62x the time of the full run). Narrow with switch: 3.7347 | can't tell (inside the band); 20k test queued (`2026-11-18-parcae-delayed-depth-20k.md`) | `...-delayed-depth-switch.md` |
| 2026-11-18 | Slot pointer composition (rank 3), open keys, 20k | Composing slot loop, its depth-1 twin; controls = the hop-task slot loop and twin from the same start | Chain accuracy at 1 / 2 / 3 / 4 hops: composing loop 1.00 / 1.00 / 0.99 / 0.69; its twin 0.94 / 0.31 / 0.23 / 0.24; plain slot loop 1.00 / 0.37 / 0.27 / 0.25. Hard hops (6 to 32): chance for all. No staircase: one eval pass already reads 0.99 / 0.93 / 0.59 at 2 / 3 / 4 hops, and nothing is gained past two passes. Graph questions at chance. Web CE 3.5833 against 3.5940 | not supported on the preregistered hard-hop metric; a large loop-against-twin gap at 2 to 4 hops that the metric did not name | `...-slot-pointer-composition.md` (2026-11-18 amendment) |
| 2026-11-18 | Hop task, token loop (seed 1) | Plain token loop + pointer, web-text start, 20k mixed stream | A staircase in eval depth. Chain hops read at 0.95 or more: 2 hops at depth 1, 4 at depth 2, 8 at depth 3, 12 at depth 4, 16 at depth 6 and above. Past the trained range (chains up to 16, depth up to 8): 24 hops reads 0.62 and 32 hops 0.52 at depth 12. Reachability 0.67 (flat 0.885), no depth trend. Web CE 3.4373 at depth 6 against 3.5114 flat, K1-K6 +0.104 | reading for the hop plan's ordering prediction: token loop above slot loop, holds in seed 1 | Parcae `lxtul/hops` hop plan |

Notes on the 2026-11-18 rows (one seed each, 5,000 steps unless stated):

- **The depth-1 twin of the full token loop reads 3.7195 at 136 ms/step.** No looped arm of the night beats it at
  5,000 steps. So a 5,000-step CE cannot rank loop levers against a twin on this testbed; the K-curve and a longer
  run can. This agrees with the standing rule that a 5k CE gap is not a verdict for a deep model.
- **The teacher term is the one large effect.** It is a training-cost lever for every arm (0.14 nats), and it is
  not a loop lever. Owed: an equal-wall-clock comparison, because the teacher forward costs about 145 ms/step.
- **The delayed loop rebuilt the full K-curve in 1,250 deep steps.** Depth use on the token loop forms fast once
  depth is on. Wolfe's caution about the early steps is not tested at 5k; the 20k run tests it.
- **The token loop is the hop reference.** With a web-text start it walks chains at about 2 to 4 hops per pass and
  keeps going past its trained depth and chain length. The slot loop (one state per statement) does not. The
  pointer-composition arm closes part of that gap at 2 to 4 hops; a code reading says its core blocks may carry the
  chain and the tables only copy. Owed: an eval with the table update skipped.
- Plan files with `...` are in the Parcae worktree `parcae-wt-hops`, `docs/experiments/planned/2026-11-17-parcae-<name>`.
