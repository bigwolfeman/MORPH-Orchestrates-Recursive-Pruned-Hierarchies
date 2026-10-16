# Parcae minimal slot loop: does a shallow coda make the loop use depth?

Status: failure (withdrawn 2026-10-14 before any run: the LCM-Parcae reference tests span-level depth with a decoder that cannot do cross-span work, more cleanly; the decode loop redesign reshapes the coda)

## Question

Under strict geometry the coda's tokens read every earlier slot, so the coda's own 4 layers can do the
cross-span work (thoughts up to span s -> what span s+1 needs). The loop then only summarises a span, a
one-pass job: the minimal arm is flat (K1-K6 0.0000, 3 runs). Wolfe's principle (2026-10-14): only the
main CE may buy depth; no auxiliary terms. This arm keeps the main CE and strict geometry and moves coda
layers into the span-local prelude, so the token path keeps 8 layers and the FLOPs stay equal, but the
part of the model that reads the slots is shallow. If the cross-span work needs depth, it must now happen in
the loop.

## Arms (5k, `arm=minimal`, seed 1)

| arm | prelude / core / coda |
| --- | --- |
| minimal, 2-layer coda | `shape.n_layers_in_prelude=6 shape.n_layers_in_coda=2` |
| minimal, 1-layer coda | `shape.n_layers_in_prelude=7 shape.n_layers_in_coda=1` |

Reference: minimal 4/6/4, CE@6 3.7646 / 3.7615 (2 seeds), K1-K6 0.0000.

## Predictions

1. 2-layer coda: K1-K6 at least +0.01 (35 %).
2. 2-layer coda: CE@6 within +-0.015 of 3.763 (55 %).
3. 1-layer coda: K1-K6 at least +0.01 (45 %).
4. 1-layer coda: CE@6 at least 0.02 worse than 3.763 (60 %).

## What each outcome changes

- Depth appears with CE@6 at or below 3.763: the main CE alone buys depth once the coda cannot do the
  cross-span work; scale this lever (and price it at equal compute).
- Depth appears and CE@6 is worse: the tax refund again, now from the layer split.
- Both flat: the loop's first pass does the cross-span work even with a 1-layer reader; at this scale that
  work is shallow, and loop depth must come from another job.

## Method

`morph-scratch/abq/pz-panel/coda.sh`, after the epivol split and before the equal-compute panel; testbed
checkout; one job at a time under the GPU lock. Amendments get a date and a reason here.

## Results

## Verdict

## Updated hypothesis

(No run. Withdrawn before any data; see Status.)
